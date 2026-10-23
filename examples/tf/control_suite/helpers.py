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

"""Environment factories for the control suite and the Open Ant simulator."""

import atexit
import os
import sys
import threading
from typing import Optional

from absl import logging
from acme import specs
from acme import types
from acme import wrappers
import dm_env
import numpy as np


def make_environment(
    evaluation: bool = False,
    domain_name: str = 'cartpole',
    task_name: str = 'balance',
    from_pixels: bool = False,
    frames_to_stack: int = 3,
    flatten_stack: bool = False,
    num_action_repeats: Optional[int] = None,
) -> dm_env.Environment:
  """Implements a control suite environment factory."""
  # Load dm_suite lazily not require Mujoco license when not using it.
  from dm_control import suite  # pylint: disable=g-import-not-at-top
  from acme.wrappers import mujoco as mujoco_wrappers  # pylint: disable=g-import-not-at-top

  # Load raw control suite environment.
  environment = suite.load(domain_name, task_name)

  # Maybe wrap to get pixel observations from environment state.
  if from_pixels:
    environment = mujoco_wrappers.MujocoPixelWrapper(environment)
    environment = wrappers.FrameStackingWrapper(
        environment, num_frames=frames_to_stack, flatten=flatten_stack)
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)

  if num_action_repeats:
    environment = wrappers.ActionRepeatWrapper(
        environment, num_repeats=num_action_repeats)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment


def _open_ant_root() -> str:
  return os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'open-ant'))


def _load_open_ant():
  """Import the Open Ant simulator without requiring it at process start."""
  root = _open_ant_root()
  sim_dir = os.path.join(root, 'sim')
  if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)
  from ant_mujoco import AntEnv  # pylint: disable=g-import-not-at-top
  from embodied_ant_env import BackAndForthTask  # pylint: disable=g-import-not-at-top
  from embodied_ant_env import ForwardTask  # pylint: disable=g-import-not-at-top
  return AntEnv, ForwardTask, BackAndForthTask


class GymnasiumWrapper(dm_env.Environment):
  """Adapts a Gymnasium environment to the dm_env API used by Acme."""

  def __init__(self, environment):
    self._environment = environment
    self._reset_next_step = True
    self._last_info = None
    self._observation_spec = _convert_space(
        environment.observation_space, name='observation')
    self._action_spec = _convert_space(environment.action_space, name='action')

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    observation, info = self._environment.reset()
    self._last_info = info
    return dm_env.restart(self._cast_observation(observation))

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()

    observation, reward, terminated, truncated, info = self._environment.step(
        action)
    self._reset_next_step = bool(terminated or truncated)
    self._last_info = info
    observation = self._cast_observation(observation)
    reward = np.asarray(reward, dtype=self.reward_spec().dtype)

    if terminated:
      return dm_env.termination(reward, observation)
    if truncated:
      return dm_env.truncation(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._action_spec

  def close(self):
    self._environment.close()

  def _cast_observation(self, observation: types.NestedArray) -> types.NestedArray:
    return _cast_like_spec(observation, self._observation_spec)


def _cast_like_spec(value, spec):
  if isinstance(spec, dict):
    return {key: _cast_like_spec(value[key], spec[key]) for key in spec}
  return np.asarray(value, dtype=spec.dtype)


def _convert_space(space, name: Optional[str] = None) -> types.NestedSpec:
  """Converts a Gymnasium space to a dm_env spec."""
  spaces = getattr(space, 'spaces', None)
  if isinstance(spaces, dict):
    return {key: _convert_space(value, key) for key, value in spaces.items()}
  if hasattr(space, 'n') and not hasattr(space, 'low'):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)
  return specs.BoundedArray(
      shape=tuple(space.shape),
      dtype=space.dtype,
      minimum=np.array(space.low, dtype=space.dtype),
      maximum=np.array(space.high, dtype=space.dtype),
      name=name)


def make_open_ant_environment(
    evaluation: bool = False,
    task_name: str = 'forward',
    control_dt: float = 0.02,
    terminate_on_upside_down: bool = True,
    radius: float = 1.0,
    origin=(0.0, 0.0),
    record_video: bool = False,
    record_every: int = 10,
    video_steps: int = 200,
    video_dir: str = '',
    reward_scale: float = 1.0,
) -> dm_env.Environment:
  """Open Ant simulator as a dm_env environment.

  Args:
    evaluation: When True, this is the evaluator and video is not recorded.
    task_name: `forward` or `back_and_forth`.
    control_dt: Simulator control period in seconds.
    terminate_on_upside_down: End the episode when the body up-axis points
      below the horizontal.
    radius: Back-and-forth circle radius in meters.
    origin: Back-and-forth circle center (x, y) in meters.
    record_video: Record one training actor when evaluation is False.
    record_every: Start a clip every this many training-actor steps.
    video_steps: Steps written into each clip before the mp4 is saved.
    video_dir: Output directory. Empty uses ~/acme/<acme_id>/videos.
    reward_scale: Multiplier applied to the reward returned to the agent.
  """
  ant_env_cls, forward_task, back_and_forth_task = _load_open_ant()
  if task_name == 'forward':
    task = forward_task()
  elif task_name == 'back_and_forth':
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (2,):
      raise ValueError(f'origin must have shape (2,), got {origin.shape}.')
    task = back_and_forth_task(radius=radius, origin=origin)
  else:
    raise ValueError(
        f'Unknown Open Ant task {task_name!r}. Expected forward or back_and_forth.')

  record_this_actor = _claim_training_video(record_video, evaluation)
  render_mode = 'rgb_array' if record_this_actor else None
  environment = ant_env_cls(
      control_dt=control_dt,
      render_mode=render_mode,
      terminate_on_upside_down=terminate_on_upside_down,
      task=task)
  if record_this_actor:
    _frame_video_camera(environment)
    environment = _record_open_ant_video(
        environment,
        record_every=record_every,
        video_steps=video_steps,
        video_dir=video_dir)
  if reward_scale != 1.0:
    import gymnasium as gym  # pylint: disable=g-import-not-at-top
    logging.info('Scaling Open Ant rewards by %s', reward_scale)
    environment = gym.wrappers.TransformReward(
        environment, lambda reward, scale=reward_scale: reward * scale)
  environment = GymnasiumWrapper(environment)
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def _video_directory(video_dir: str) -> str:
  if video_dir:
    path = os.path.expanduser(video_dir)
    os.makedirs(path, exist_ok=True)
    return path
  from acme.utils import paths  # pylint: disable=g-import-not-at-top
  return paths.process_path('~/acme', 'videos')


_VIDEO_LOCK = threading.Lock()
_VIDEO_SPEC_SEEN = False
_VIDEO_ATTACHED = False


def _claim_training_video(record_video: bool, evaluation: bool) -> bool:
  """Record sampled actions from one training actor.

  DistributedMPO builds a throwaway environment before any actor starts. That
  call is skipped, as are the evaluator and the other training actors.
  """
  global _VIDEO_SPEC_SEEN, _VIDEO_ATTACHED
  if not record_video or evaluation:
    return False
  with _VIDEO_LOCK:
    if not _VIDEO_SPEC_SEEN:
      _VIDEO_SPEC_SEEN = True
      return False
    if _VIDEO_ATTACHED:
      return False
    _VIDEO_ATTACHED = True
    return True


def _frame_video_camera(environment):
  """Hold the camera at a fixed 4 m. It does not follow the ant.

  The default distance is the model extent, about 22 m. At 4 m the 1 m
  radius circle fits in the frame, and the aim point stays where it started.
  """
  environment.mujoco_renderer.default_cam_config = {'distance': 4.0}


def _record_open_ant_video(environment,
                           record_every: int,
                           video_steps: int,
                           video_dir: str):
  """Record evaluator clips with Gymnasium's RecordVideo wrapper.

  Open Ant does not end episodes, so clips start from the step counter.
  Each clip is video_steps long and the next one begins record_every steps
  after the previous start. record_every is raised to video_steps when it is
  smaller, so a new clip does not cut the current one short. The recorded
  actor samples actions, which is the data used for training.
  """
  import gymnasium as gym  # pylint: disable=g-import-not-at-top
  if record_every < 1:
    raise ValueError(f'record_every must be >= 1, got {record_every}.')
  if video_steps < 1:
    raise ValueError(f'video_steps must be >= 1, got {video_steps}.')
  # RecordVideo saves the current clip when a new one starts. The gap between
  # starts has to cover the clip itself.
  interval = max(record_every, video_steps)
  directory = _video_directory(video_dir)
  logging.info(
      'Recording %d-step training-actor clips every %d steps to %s',
      video_steps, interval, directory)
  recorded = gym.wrappers.RecordVideo(
      environment,
      directory,
      episode_trigger=lambda episode_id: False,
      step_trigger=lambda step_id: step_id % interval == 0,
      video_length=video_steps,
      name_prefix='actor',
  )
  atexit.register(recorded.close)
  return recorded


_REWARD_LOG_LOCK = threading.Lock()
_REWARD_LOG_ATTACHED = False
_SPEC_ENV_SEEN = False


class AverageRewardLogWrapper(dm_env.Environment):
  """Logs average reward to SimEmbodiedAnt_average_rewards.csv.

  Columns are step, raw_reward, reward, mean_return. `reward` is the mean
  reward-per-second over a 120 second window. `raw_reward` is the simulator
  reward before `reward_scale` is applied.
  """

  def __init__(self,
               environment: dm_env.Environment,
               control_dt: float,
               log_folder: str,
               env_id: str = 'SimEmbodiedAnt',
               time_window: float = 120.0,
               flush_every: int = 4000,
               reward_scale: float = 1.0):
    self._environment = environment
    self._reward_scale = float(reward_scale)
    self._flush_every = flush_every
    self._steps_since_flush = 0
    self._episode_return = 0.0
    agents_dir = os.path.join(_open_ant_root(), 'agents')
    if agents_dir not in sys.path:
      sys.path.insert(0, agents_dir)
    from reward import RewardTracker  # pylint: disable=g-import-not-at-top
    self._tracker = RewardTracker(
        env_dt=control_dt,
        env_id=env_id,
        time_window=time_window,
        log_folder=log_folder)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    reward = float(np.asarray(timestep.reward)) / self._reward_scale
    self._tracker.update(reward)
    self._episode_return += reward
    self._steps_since_flush += 1
    if timestep.last():
      self._tracker.record_episode_return(self._episode_return)
      self._episode_return = 0.0
    if self._steps_since_flush >= self._flush_every:
      self._tracker.log()
      self._steps_since_flush = 0
    return timestep

  def reset(self) -> dm_env.TimeStep:
    return self._environment.reset()

  def observation_spec(self) -> types.NestedSpec:
    return self._environment.observation_spec()

  def action_spec(self) -> types.NestedSpec:
    return self._environment.action_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def close(self):
    self._tracker.log()
    self._environment.close()


def maybe_log_average_reward(
    environment: dm_env.Environment,
    evaluation: bool,
    control_dt: float,
    reward_scale: float = 1.0,
) -> dm_env.Environment:
  """Attach the average-reward CSV to one training actor.

  DistributedMPO builds a throwaway environment to read the spec before any
  actor starts. That first call is left unlogged. The next training actor
  writes SimEmbodiedAnt_average_rewards.csv beside the Acme run directory.
  """
  global _REWARD_LOG_ATTACHED, _SPEC_ENV_SEEN
  if evaluation:
    return environment
  with _REWARD_LOG_LOCK:
    if not _SPEC_ENV_SEEN:
      _SPEC_ENV_SEEN = True
      return environment
    if _REWARD_LOG_ATTACHED:
      return environment
    _REWARD_LOG_ATTACHED = True
  from acme.utils import paths  # pylint: disable=g-import-not-at-top
  log_folder = os.path.join(
      os.path.expanduser('~/acme'), paths.get_unique_id()[0])
  return AverageRewardLogWrapper(
      environment, control_dt, log_folder, reward_scale=reward_scale)
