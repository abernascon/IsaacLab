# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import G1FlatEnvCfg
from .mdp import UniformHeightCommandCfg, track_height_l2


@configclass
class G1LowHeightEnvCfg(G1FlatEnvCfg):
    """G1 flat env with a height command.

    Identical to G1FlatEnvCfg except:
    - a ``target_height`` command is added, resampled uniformly in [0.45, 0.80] m each episode
    - a ``track_height_l2`` reward penalises deviation from that command
    Everything else (velocity commands, terrain, all other rewards…) is unchanged.
    """

    def __post_init__(self):
        super().__post_init__()

        # Add height command: resampled every episode, uniform in [0.45, 0.80] m
        self.commands.target_height = UniformHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            ranges=UniformHeightCommandCfg.Ranges(height=(0.45, 0.80)),
        )

        # Add height-tracking reward (stronger weight to create meaningful gradient signal)
        self.rewards.track_height_l2 = RewTerm(
            func=track_height_l2,
            weight=-3.0,
            params={"command_name": "target_height", "asset_cfg": SceneEntityCfg("robot")},
        )

        # Update multi-head critic fields to include the new reward
        self.reward_components = sum(
            isinstance(getattr(self.rewards, attr), RewTerm)
            for attr in dir(self.rewards)
            if not attr.startswith("__")
        )
        self.reward_component_names = [
            attr for attr in dir(self.rewards)
            if isinstance(getattr(self.rewards, attr), RewTerm) and not attr.startswith("__")
        ]
        # ONLY true command-tracking objectives are "task" rewards.
        # Everything else (energy, smoothness, joint limits, orientation, etc.) is a
        # regulariser.  GCR-PPO's priority-aware PCGrad projects regulariser gradients
        # so they don't oppose task gradients — but NOT vice versa.
        # Labelling too many terms as "task" makes the projection nearly symmetric
        # (vanilla PCGrad), removing the benefit of priority.
        self.reward_component_task_rew = [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "track_height_l2",
            "termination_penalty",
        ]


class G1LowHeightEnvCfg_PLAY(G1LowHeightEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
