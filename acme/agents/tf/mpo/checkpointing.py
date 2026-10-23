# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint pieces that TensorFlow does not track on its own.

The learner's Adam moments and MPO dual variables are created on the first
update. The checkpointer restores during construction, so those variables have
to exist before then. Python, NumPy, and simulator state are not TensorFlow
variables; they are stored next to the networks so a restarted process keeps
the same random stream and the same episode.
"""

import copy
import pickle
import random
from typing import Any, List, Optional

from absl import logging
from acme.wrappers import base
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree

# Open Ant task fields that change during an episode and affect the next reward.
_TASK_FIELDS = (
    'last_pos',
    'last_action',
    'reward_direction_I',
    'previous_pos_timestamp',
    'last_heading_vector',
    'last_seen',
    'last_position_timestamp',
)


def stateless_seed() -> tf.Tensor:
  """An int32[2] seed from the checkpointed global generator.

  `tf.random.uniform` ignores that generator, so `distribution.sample()` with
  no seed would not continue after a restart. TFP treats this tensor as a
  stateless seed, and the generator advances so the next draw is new.
  """
  seeds = tf.random.get_global_generator().make_seeds(1)
  return tf.cast(seeds[:, 0], tf.int32)


def materialize_learner_variables(learner) -> None:
  """Create optimizer moments and MPO dual variables before restore.

  Args:
    learner: An `MPOLearner` whose networks have already been built with
      `create_variables`.
  """
  loss = learner._policy_loss_module  # pylint: disable=protected-access
  if hasattr(loss, 'create_dual_variables_once'):
    dummy = _zeros_like_signature(learner._policy_network)  # pylint: disable=protected-access
    distribution = learner._policy_network(*dummy)  # pylint: disable=protected-access
    if getattr(loss, '_per_dim_constraining', True):
      shape = distribution.event_shape
    else:
      shape = (1,)
    loss.create_dual_variables_once(shape, tf.float32)

  _initialize_optimizer(
      learner._critic_optimizer,  # pylint: disable=protected-access
      learner._observation_network.trainable_variables +  # pylint: disable=protected-access
      learner._critic_network.trainable_variables)  # pylint: disable=protected-access
  _initialize_optimizer(
      learner._policy_optimizer,  # pylint: disable=protected-access
      learner._policy_network.trainable_variables)  # pylint: disable=protected-access
  _initialize_optimizer(
      learner._dual_optimizer,  # pylint: disable=protected-access
      loss.trainable_variables)


def _initialize_optimizer(optimizer, variables: List[tf.Variable]) -> None:
  if not variables:
    return
  initialize = getattr(optimizer, '_initialize', None)
  if initialize is not None:
    initialize(list(variables))


def _zeros_like_signature(network: snt.Module):
  """Batch-1 zeros matching the signature stored by `create_variables`."""
  def _zeros(spec):
    shape = tuple(1 if dim is None else dim for dim in spec.shape)
    return tf.zeros(shape, dtype=spec.dtype)
  return tree.map_structure(_zeros, network._input_signature)  # pylint: disable=protected-access


class RandomState(tf.train.experimental.PythonState):
  """Python and NumPy RNGs, and the actor's simulator snapshot."""

  def __init__(self, environment: Optional[dm_env.Environment] = None):
    self._environment = environment

  def serialize(self) -> bytes:
    payload = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }
    if self._environment is not None:
      snapshot = snapshot_environment(self._environment)
      if snapshot is not None:
        payload['environment'] = snapshot
    return pickle.dumps(payload)

  def deserialize(self, string_value: bytes) -> None:
    payload = pickle.loads(string_value)
    random.setstate(payload['python'])
    np.random.set_state(payload['numpy'])
    snapshot = payload.get('environment')
    if snapshot is not None and self._environment is not None:
      self._environment.queue_resume(snapshot)


class ResumableEnvironment(base.EnvironmentWrapper):
  """Returns the checkpointed timestep from the first `reset` after a restart.

  `EnvironmentLoop` always resets when it starts. For a simulator, that reset
  is replaced by the saved physics state and the observation the actor had
  already seen, so the episode continues. A real robot has no physics snapshot
  and keeps the normal reset.
  """

  def __init__(self, environment: dm_env.Environment):
    super().__init__(environment)
    self._resume = None
    self._last_timestep = None

  def queue_resume(self, snapshot: Any) -> None:
    self._resume = snapshot

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    self._last_timestep = timestep
    return timestep

  def reset(self) -> dm_env.TimeStep:
    if self._resume is None:
      timestep = self._environment.reset()
      self._last_timestep = timestep
      return timestep

    snapshot = self._resume
    self._resume = None
    saved = snapshot['timestep']
    # The episode already ended. A normal reset starts the next one.
    if saved.last():
      timestep = self._environment.reset()
      self._last_timestep = timestep
      logging.info('Saved episode had already ended; starting a new one.')
      return timestep
    apply_snapshot(self._environment, snapshot)
    # The loop always passes this timestep to adder.add_first, which accepts
    # only a FIRST step. A mid-episode checkpoint is a MID step, so keep its
    # observation and mark it as the start of the continued episode.
    timestep = saved if saved.first() else dm_env.restart(saved.observation)
    self._last_timestep = timestep
    logging.info('Restored simulator state; continuing the saved episode.')
    return timestep


def snapshot_environment(environment: dm_env.Environment) -> Optional[dict]:
  """Capture simulator, task, and RNG state. None when there is nothing to resume."""
  if environment._last_timestep is None:  # pylint: disable=protected-access
    return None
  inner = environment._environment  # pylint: disable=protected-access
  chain = list(_chain(inner))
  nodes = [_capture_node(env) for env in chain]
  if not any(node and node.get('simulator') for node in nodes):
    return None
  return {
      'timestep': copy.deepcopy(environment._last_timestep),  # pylint: disable=protected-access
      'nodes': nodes,
  }


def apply_snapshot(environment: dm_env.Environment, snapshot: dict) -> None:
  chain = list(_chain(environment))
  nodes = snapshot['nodes']
  if len(chain) != len(nodes):
    raise ValueError(
        'Cannot restore the environment: the wrapper stack has '
        f'{len(chain)} levels and the checkpoint has {len(nodes)}.')
  for env, node in zip(chain, nodes):
    if node:
      _apply_node(env, node)


def _chain(environment):
  seen = set()
  current = environment
  while current is not None and id(current) not in seen:
    yield current
    seen.add(id(current))
    data = getattr(current, '__dict__', {})
    child = None
    for key in ('_environment', 'env'):
      if key in data and data[key] is not None:
        child = data[key]
        break
    current = child


def _capture_node(env) -> Optional[dict]:
  data = vars(env)
  node = {}
  if 'model' in data and 'data' in data:
    mujoco_state = _mujoco_state(data['model'], data['data'])
    if mujoco_state is not None:
      node['mujoco'] = mujoco_state
      node['simulator'] = True
  physics = data.get('_physics', data.get('physics'))
  if physics is not None and hasattr(physics, 'get_state'):
    node['physics'] = np.array(physics.get_state(), copy=True)
    node['simulator'] = True
  if '_reset_next_step' in data:
    node['reset_next_step'] = bool(data['_reset_next_step'])
  if '_previous_yaw' in data:
    node['previous_yaw'] = data['_previous_yaw']
  if '_step_count' in data and isinstance(data['_step_count'], (int, np.integer)):
    node['step_count'] = int(data['_step_count'])
  generator = data.get('_np_random')
  if generator is not None:
    node['np_random'] = copy.deepcopy(generator.bit_generator.state)
  task = data.get('task', data.get('_task'))
  if task is not None:
    task_state = _task_state(task)
    if task_state:
      node['task'] = task_state
  if not node:
    return None
  return node


def _apply_node(env, node: dict) -> None:
  data = vars(env)
  if 'mujoco' in node and 'model' in data and 'data' in data:
    _set_mujoco_state(data['model'], data['data'], node['mujoco'])
  if 'physics' in node:
    physics = data.get('_physics', data.get('physics'))
    physics.set_state(node['physics'])
    if hasattr(physics, 'forward'):
      physics.forward()
  if 'reset_next_step' in node:
    env._reset_next_step = node['reset_next_step']  # pylint: disable=protected-access
  if 'previous_yaw' in node:
    env._previous_yaw = node['previous_yaw']  # pylint: disable=protected-access
  if 'step_count' in node:
    env._step_count = node['step_count']  # pylint: disable=protected-access
  if 'np_random' in node:
    # A restarted process has not seeded the environment yet, so the generator
    # may not exist. Touching the property creates it; the saved state replaces it.
    if data.get('_np_random') is None:
      getattr(env, 'np_random')
    env._np_random.bit_generator.state = copy.deepcopy(  # pylint: disable=protected-access
        node['np_random'])
  if 'task' in node:
    task = data.get('task', data.get('_task'))
    _set_task_state(task, node['task'])


def _task_state(task) -> dict:
  state = {}
  for name in _TASK_FIELDS:
    if hasattr(task, name):
      value = getattr(task, name)
      if isinstance(value, np.ndarray):
        value = value.copy()
      state[name] = value
  generator = getattr(task, 'random', None)
  if isinstance(generator, np.random.RandomState):
    state['random'] = generator.get_state()
  return state


def _set_task_state(task, state: dict) -> None:
  for name, value in state.items():
    if name == 'random':
      task.random.set_state(value)
    elif hasattr(task, name):
      setattr(task, name, value.copy() if isinstance(value, np.ndarray) else value)


def _mujoco_spec():
  import mujoco  # pylint: disable=g-import-not-at-top
  return int(mujoco.mjtState.mjSTATE_PHYSICS) | int(
      mujoco.mjtState.mjSTATE_TIME) | int(mujoco.mjtState.mjSTATE_CTRL)


def _mujoco_state(model, data) -> Optional[dict]:
  try:
    import mujoco  # pylint: disable=g-import-not-at-top
    spec = _mujoco_spec()
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
  except (ImportError, AttributeError, TypeError, ValueError):
    return None
  return {'spec': spec, 'state': state}


def _set_mujoco_state(model, data, saved: dict) -> None:
  import mujoco  # pylint: disable=g-import-not-at-top
  mujoco.mj_setState(model, data, np.ascontiguousarray(saved['state']),
                     int(saved['spec']))
  mujoco.mj_forward(model, data)
