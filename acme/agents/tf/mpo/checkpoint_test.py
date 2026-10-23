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

"""Tests that a restart restores optimizers, duals, RNGs, and simulator state."""

import random
import tempfile

from acme import specs
from acme import types
from acme.agents.tf.mpo import checkpointing
from acme.agents.tf.mpo import learning
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import dm_env
import mujoco
import numpy as np
import sonnet as snt
import tensorflow as tf

from absl.testing import absltest


def _make_networks(obs_spec, action_spec):
  num_dimensions = int(np.prod(action_spec.shape))
  observation = tf2_utils.to_sonnet_module(tf.identity)
  policy = snt.Sequential([
      networks.LayerNormMLP((8, 8)),
      networks.MultivariateNormalDiagHead(num_dimensions),
  ])
  target_policy = snt.Sequential([
      networks.LayerNormMLP((8, 8)),
      networks.MultivariateNormalDiagHead(num_dimensions),
  ])
  critic = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP((8, 8, 1)))
  target_critic = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP((8, 8, 1)))
  target_observation = tf2_utils.to_sonnet_module(tf.identity)

  emb_spec = tf2_utils.create_variables(observation, [obs_spec])
  tf2_utils.create_variables(policy, [emb_spec])
  tf2_utils.create_variables(target_policy, [emb_spec])
  tf2_utils.create_variables(critic, [emb_spec, action_spec])
  tf2_utils.create_variables(target_critic, [emb_spec, action_spec])
  tf2_utils.create_variables(target_observation, [obs_spec])
  return observation, policy, critic, target_observation, target_policy, target_critic


class _Replay:

  def __init__(self, transition):
    self._transition = transition

  def __iter__(self):
    return self

  def __next__(self):
    return self

  @property
  def data(self):
    return self._transition


class LearnerCheckpointTest(absltest.TestCase):

  def test_restart_restores_optimizers_duals_and_rng(self):
    obs_spec = specs.Array(shape=(3,), dtype=np.float32)
    action_spec = specs.Array(shape=(2,), dtype=np.float32)
    batch = 4
    transition = types.Transition(
        observation=tf.zeros([batch, 3]),
        action=tf.zeros([batch, 2]),
        reward=tf.zeros([batch]),
        discount=tf.ones([batch]),
        next_observation=tf.zeros([batch, 3]),
        extras=tf.constant(0.),
    )
    dataset = _Replay(transition)
    directory = tempfile.mkdtemp()

    def new_learner():
      nets = _make_networks(obs_spec, action_spec)
      return learning.MPOLearner(
          policy_network=nets[1],
          critic_network=nets[2],
          observation_network=nets[0],
          target_policy_network=nets[4],
          target_critic_network=nets[5],
          target_observation_network=nets[3],
          discount=0.99,
          num_samples=5,
          target_policy_update_period=25,
          target_critic_update_period=25,
          dataset=dataset,
          logger=loggers.make_default_logger('learner', save_data=False),
          save_directory=directory,
          checkpoint_time_delta_minutes=1e9,
      )

    random.seed(123)
    np.random.seed(123)
    learner = new_learner()
    learner.step()

    # Nonzero moments and a moved dual, so a missed restore is obvious.
    learner._policy_optimizer.m[0].assign(  # pylint: disable=protected-access
        tf.ones_like(learner._policy_optimizer.m[0]))  # pylint: disable=protected-access
    learner._dual_optimizer.step.assign(11)  # pylint: disable=protected-access
    learner._policy_loss_module._log_temperature.assign([1.25])  # pylint: disable=protected-access
    learner._num_steps.assign(9)  # pylint: disable=protected-access
    _ = tf.random.uniform([1])
    learner._checkpointer.save(force=True)  # pylint: disable=protected-access

    expected_m = learner._policy_optimizer.m[0].numpy().copy()  # pylint: disable=protected-access
    expected_step = int(learner._dual_optimizer.step.numpy())  # pylint: disable=protected-access
    expected_temperature = (
        learner._policy_loss_module._log_temperature.numpy().copy())  # pylint: disable=protected-access
    expected_weight = learner._policy_network.trainable_variables[0].numpy(  # pylint: disable=protected-access
    ).copy()
    expected_updates = int(learner._num_steps.numpy())  # pylint: disable=protected-access
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    def draw_from_generator():
      seed = checkpointing.stateless_seed()
      return tf.random.stateless_normal(shape=[4], seed=seed).numpy()

    expected_sample = draw_from_generator()

    random.seed(999)
    np.random.seed(999)
    tf.random.set_global_generator(tf.random.Generator.from_seed(7))
    restored = new_learner()

    np.testing.assert_allclose(
        restored._policy_optimizer.m[0].numpy(), expected_m)  # pylint: disable=protected-access
    self.assertEqual(int(restored._dual_optimizer.step.numpy()), expected_step)  # pylint: disable=protected-access
    np.testing.assert_allclose(
        restored._policy_loss_module._log_temperature.numpy(),  # pylint: disable=protected-access
        expected_temperature)
    np.testing.assert_allclose(
        restored._policy_network.trainable_variables[0].numpy(),  # pylint: disable=protected-access
        expected_weight)
    self.assertEqual(int(restored._num_steps.numpy()), expected_updates)  # pylint: disable=protected-access
    self.assertEqual(random.getstate(), expected_python)
    restored_numpy = np.random.get_state()
    self.assertEqual(restored_numpy[0], expected_numpy[0])
    np.testing.assert_array_equal(restored_numpy[1], expected_numpy[1])
    self.assertEqual(restored_numpy[2:], expected_numpy[2:])
    np.testing.assert_allclose(draw_from_generator(), expected_sample)

    # The pre-created duals and moments must still train.
    restored.step()


_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="hinge" type="hinge"/>
      <geom size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Task:

  def __init__(self):
    self.last_pos = np.array([1.0, 2.0])
    self.last_action = np.zeros(2)
    self.reward_direction_I = np.array([0.0, 1.0])
    self.previous_pos_timestamp = 3.5


class _Sim(dm_env.Environment):

  def __init__(self):
    self.model = mujoco.MjModel.from_xml_string(_XML)
    self.data = mujoco.MjData(self.model)
    self._np_random = np.random.default_rng(1)
    self._previous_yaw = 0.4
    self._reset_next_step = False
    self._step_count = 7
    self.task = _Task()

  def reset(self):
    self.data.qpos[0] = 0.0
    self._step_count = 0
    return dm_env.restart(np.zeros(1, np.float32))

  def step(self, action):
    del action
    self.data.qpos[0] += 0.25
    self._step_count += 1
    self._np_random.random()
    return dm_env.transition(
        np.float32(0.), np.array([self.data.qpos[0]], np.float32))

  def observation_spec(self):
    return specs.Array(shape=(1,), dtype=np.float32)

  def action_spec(self):
    return specs.Array(shape=(1,), dtype=np.float32)


class SimulatorCheckpointTest(absltest.TestCase):

  def test_restart_continues_the_same_episode(self):
    environment = checkpointing.ResumableEnvironment(_Sim())
    environment.reset()
    timestep = environment.step(np.zeros(1))
    blob = checkpointing.RandomState(environment).serialize()

    restarted = checkpointing.ResumableEnvironment(_Sim())
    random.seed(5)
    np.random.seed(5)
    checkpointing.RandomState(restarted).deserialize(blob)
    resumed = restarted.reset()

    sim = restarted._environment  # pylint: disable=protected-access
    self.assertAlmostEqual(float(sim.data.qpos[0]), 0.25)
    self.assertEqual(sim._step_count, 1)  # pylint: disable=protected-access
    self.assertAlmostEqual(sim._previous_yaw, 0.4)  # pylint: disable=protected-access
    self.assertFalse(sim._reset_next_step)  # pylint: disable=protected-access
    np.testing.assert_allclose(sim.task.last_pos, [1.0, 2.0])
    self.assertTrue(resumed.first())
    np.testing.assert_allclose(resumed.observation, timestep.observation)

    # The saved generator was advanced once inside step. The next draw matches.
    fresh = np.random.default_rng(1)
    fresh.random()
    self.assertEqual(float(sim._np_random.random()), float(fresh.random()))  # pylint: disable=protected-access


if __name__ == '__main__':
  absltest.main()
