# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import G1FlatEnvCfg
from .mdp import UniformHeightCommandCfg, track_height_l2, track_height_rbf


@configclass
class G1LowHeightEnvCfg(G1FlatEnvCfg):
    """G1 flat env with a height command.

    Identical to G1FlatEnvCfg except:
    - a ``target_height`` command is added, resampled uniformly in [0.60, 0.75] m each episode
    - a ``track_height_rbf`` reward uses RBF kernel to reward proximity to commanded height
    - an ``alive`` reward incentivizes staying upright longer
    Everything else (velocity commands, terrain, all other rewards…) is unchanged.
    """

    def __post_init__(self):
        super().__post_init__()

        # Add height command:
        self.commands.target_height = UniformHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            ranges=UniformHeightCommandCfg.Ranges(height=(0.45, 0.55)),
        )

        # Add height-tracking reward using RBF kernel 
        self.rewards.track_height_rbf = RewTerm(
            func=track_height_rbf,
            weight=3.0,
            params={
                "command_name": "target_height",
                "asset_cfg": SceneEntityCfg("robot"),
                "sigma": 0.1,  
            },
        )
        
        # Add survival reward 
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.25)

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

        self.reward_component_task_rew = [
            "track_height_rbf",
            "alive",
        ]


class G1LowHeightEnvCfg_PLAY(G1LowHeightEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
