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
from acme import specs
from acme import types
from acme.agents.tf import mpo
import helpers
from acme.tf import networks
from acme.tf import utils as tf2_utils
import launchpad as lp
import numpy as np
import sonnet as snt


# Flags which modify the behavior of the launcher.
FLAGS = flags.FLAGS
_MAX_ACTOR_STEPS = flags.DEFINE_integer(
    'max_actor_steps', 100_000,
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
_RADIUS = flags.DEFINE_float(
    'radius', 1.0,
    'Open Ant back-and-forth circle radius in meters.')
_ORIGIN = flags.DEFINE_list(
    'origin', ['0', '0'],
    'Open Ant back-and-forth origin x,y in meters. Example: --origin=0,0.')
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
          num_dimensions, init_scale=0.7, use_tfd_independent=False) # TODO: the code was not running when use_tfd_independent was True
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
      radius=_RADIUS.value,
      origin=(float(_ORIGIN.value[0]), float(_ORIGIN.value[1])),
      record_video=_RECORD_VIDEO.value,
      record_every=_RECORD_EVERY.value,
      video_steps=_VIDEO_STEPS.value,
      video_dir=_VIDEO_DIR.value)
  return helpers.maybe_log_average_reward(
      environment, evaluation, _CONTROL_DT.value)


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

  # Construct the program.
  program_builder = mpo.DistributedMPO(
      make_environment,
      make_networks,
      target_policy_update_period=25,
      max_actor_steps=_MAX_ACTOR_STEPS.value,
      num_actors=4)

  lp.launch(programs=program_builder.build())


if __name__ == '__main__':
  app.run(main)
