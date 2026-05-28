# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import math

from isaaclab_assets.robots.unitree import G1_CFG
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import G1FlatEnvCfg
from .mdp import UniformHeightCommandCfg, track_height_l2, track_height_rbf, flat_feet_orientation


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

        # switch robot to G1 full mesh
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Add height command:
        self.commands.target_height = UniformHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            ranges=UniformHeightCommandCfg.Ranges(height=(0.55, 0.55)),
        )

        # Add height-tracking reward using RBF kernel 
        self.rewards.track_height_rbf = RewTerm(
            func=track_height_rbf,
            weight=1.0,
            params={
                "command_name": "target_height",
                "asset_cfg": SceneEntityCfg("robot"),
                "sigma": 0.3,  
            },
        )
        
        # Add survival reward 
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.25)

        #Terimantions for full body mesh
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(60)},
        )
        self.terminations.low_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.3},
        )
        

        # -- multi-head critic support --
        self.reward_component_names = [
            name for name, val in self.rewards.__dict__.items()
            if isinstance(val, RewTerm)
        ]
        self.reward_components = len(self.reward_component_names)
        self.reward_component_task_rew = ["track_height_rbf", "alive", "track_lin_vel_xy_exp", "track_ang_vel_z_exp"]


class G1LowHeightEnvCfg_PLAY(G1LowHeightEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Fixed command: 1 m/s pure forward, no lateral, no yaw
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # Disable standing-env fraction so all envs receive the fixed 1 m/s command
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Height command already fixed at 0.55 m in the training config
