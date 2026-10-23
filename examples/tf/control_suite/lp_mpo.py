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

"""Launch MPO on the control suite or the Open Ant simulator via Launchpad."""

import functools
from typing import Dict, Sequence

from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import types
from acme.agents.tf import mpo
from acme.utils import paths
import helpers
from acme.tf import losses
from acme.tf import networks
from acme.tf import utils as tf2_utils
import launchpad as lp
import numpy as np
import sonnet as snt


# Flags which modify the behavior of the launcher.
FLAGS = flags.FLAGS
_MAX_ACTOR_STEPS = flags.DEFINE_integer(
    'max_actor_steps', 500_000,
    'Number of actor steps to run; defaults to None for an endless loop.')
_ENV = flags.DEFINE_enum(
    'env', 'control_suite', ['control_suite', 'open_ant'],
    'Which environment factory to use.')
_DOMAIN = flags.DEFINE_string('domain', 'cartpole',
                              'Control suite domain name.')
_TASK = flags.DEFINE_string('task', 'balance', 'Control suite task name.')
_ANT_TASK = flags.DEFINE_enum(
    'ant_task', 'back_and_forth', ['forward', 'back_and_forth'],
    'Open Ant task. Used when --env=open_ant.')
_CONTROL_DT = flags.DEFINE_float(
    'control_dt', 0.12, 'Open Ant control timestep in seconds.')
_TERMINATE_ON_UPSIDE_DOWN = flags.DEFINE_boolean(
    'terminate_on_upside_down', True,
    'End the Open Ant episode when the body up-axis points below the '
    'horizontal. The simulator then resets.')
_RADIUS = flags.DEFINE_float(
    'radius', 1.0,
    'Open Ant back-and-forth circle radius in meters.')
_ORIGIN = flags.DEFINE_list(
    'origin', ['0', '0'],
    'Open Ant back-and-forth origin x,y in meters. Example: --origin=0,0.')
_XML = flags.DEFINE_string(
    'xml', '',
    'MuJoCo XML for --env=open_ant. Empty uses the Open Ant default model. '
    'Set this to restart a checkpoint on another robot model.')
_RECORD_VIDEO = flags.DEFINE_boolean(
    'record_video', False,
    'Record one Open Ant training actor with Gymnasium RecordVideo. '
    'The evaluator is not recorded.')
_RECORD_EVERY = flags.DEFINE_integer(
    'record_every', 2000,
    'Start a training-actor clip every this many steps. Open Ant does not end '
    'episodes, so this is a step count, not an episode count.')
_VIDEO_STEPS = flags.DEFINE_integer(
    'video_steps', 200,
    'Training-actor steps saved in each clip. The mp4 is written when the clip '
    'reaches this length.')
_VIDEO_DIR = flags.DEFINE_string(
    'video_dir', '',
    'Directory for --record_video. Empty uses ~/acme/<acme_id>/videos.')
# Same default as the embodied-Ant trainer in open-ant/agents/mpo.
_OPEN_ANT_REWARD_SCALE = 100.0


def make_policy_loss() -> snt.Module:
  """MPO dual constraints from open-ant/agents/mpo/mpo_acme.py."""
  return losses.MPO(
      epsilon=1e-1,
      epsilon_mean=2.5e-3,
      epsilon_stddev=1e-6,
      init_log_temperature=10.,
      init_log_alpha_mean=10.,
      init_log_alpha_stddev=1000.,
      action_penalization=True,
      epsilon_penalty=1e-3,
  )


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
) -> Dict[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          init_scale=0.7,
          min_scale=1e-4,
          use_tfd_independent=False)  # TODO: the code was not running when use_tfd_independent was True
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  multiplexer = networks.CriticMultiplexer(
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential([
      multiplexer,
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(1),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': tf2_utils.batch_concat,
  }


def _make_open_ant_environment(evaluation: bool = False):
  if len(_ORIGIN.value) != 2:
    raise ValueError(
        f'--origin expects two values x,y, got {_ORIGIN.value}.')
  environment = helpers.make_open_ant_environment(
      evaluation=evaluation,
      task_name=_ANT_TASK.value,
      control_dt=_CONTROL_DT.value,
      terminate_on_upside_down=_TERMINATE_ON_UPSIDE_DOWN.value,
      radius=_RADIUS.value,
      origin=(float(_ORIGIN.value[0]), float(_ORIGIN.value[1])),
      xml_path=_XML.value,
      record_video=_RECORD_VIDEO.value,
      record_every=_RECORD_EVERY.value,
      video_steps=_VIDEO_STEPS.value,
      video_dir=_VIDEO_DIR.value,
      reward_scale=_OPEN_ANT_REWARD_SCALE)
  return helpers.maybe_log_average_reward(
      environment, evaluation, _CONTROL_DT.value, _OPEN_ANT_REWARD_SCALE)


def main(_):
  if _RECORD_VIDEO.value and _ENV.value != 'open_ant':
    raise ValueError('--record_video is only supported for --env=open_ant.')
  if _ENV.value == 'open_ant':
    make_environment = _make_open_ant_environment
  else:
    make_environment = functools.partial(
        helpers.make_environment,
        domain_name=_DOMAIN.value,
        task_name=_TASK.value)

  acme_id = paths.get_unique_id()[0]
  logging.info(
      'Learner weights, optimizer moments, dual variables, replay, step '
      'counts, and random state are saved under ~/acme/%s. Each actor also '
      'saves its random state and, in simulation, the episode state. Restart '
      'this run with --acme_id=%s', acme_id, acme_id)
  # A full replay checkpoint can take longer than Launchpad's 10s stop window.
  # A second Ctrl-C still kills the process immediately.
  if not FLAGS['lp_termination_notice_secs'].present:
    FLAGS.lp_termination_notice_secs = 180

  # Learning hyperparameters match the defaults in
  # open-ant/agents/mpo/mpo_acme.py. Actor count stays at 4: that file's
  # --num_envs is a vectorized-env setting, and the samples-per-insert ratio
  # does not depend on it.
  program_builder = mpo.DistributedMPO(
      make_environment,
      make_networks,
      num_actors=4,
      batch_size=512,
      min_replay_size=200,
      max_replay_size=1_000_000,
      samples_per_insert=1536.0,
      n_step=5,
      num_samples=64,
      additional_discount=0.92,
      target_policy_update_period=100,
      target_critic_update_period=100,
      policy_loss_factory=make_policy_loss,
      policy_learning_rate=3e-4,
      critic_learning_rate=1e-3,
      dual_learning_rate=1e-3,
      max_actor_steps=_MAX_ACTOR_STEPS.value)

  lp.launch(programs=program_builder.build())


if __name__ == '__main__':
  app.run(main)
