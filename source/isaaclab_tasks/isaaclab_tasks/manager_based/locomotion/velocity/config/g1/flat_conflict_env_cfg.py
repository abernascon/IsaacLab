# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG  # isort: skip
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg, ObservationsCfg

from .mdp import ScaledVelocityCommandCfg
from .rough_env_cfg import G1RoughEnvCfg, G1Rewards


##
# Config subclasses that declare the new conflict-task fields.
##

@configclass
class G1FlatConflictCommandsCfg(CommandsCfg):
    """Extends the base command set with a negated conflict velocity."""

    conflict_velocity: ScaledVelocityCommandCfg = ScaledVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,  # derived from source — no independent heading sampling
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
        source_command_name="base_velocity",
        x_scale=1.0,   # forward component is shared — both commands go forward
        y_scale=-1.0,  # only lateral direction is flipped: base=left, conflict=right
        yaw_scale=-1.0,    # yaw: turn in the opposite direction (conflicting heading)
    )


@configclass
class G1FlatConflictPolicyCfg(ObservationsCfg.PolicyCfg):
    """Adds the conflict velocity observation so the policy can distinguish both objectives."""

    conflict_velocity: ObsTerm = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "conflict_velocity"},
    )


@configclass
class G1FlatConflictObsCfg(ObservationsCfg):
    policy: G1FlatConflictPolicyCfg = G1FlatConflictPolicyCfg()


@configclass
class G1FlatConflictRewardsCfg(G1Rewards):
    """Extends G1Rewards with a survival term and two conflicting velocity-tracking objectives."""

    alive: RewTerm = RewTerm(func=mdp.is_alive, weight=0.25)

    # Mirror of the primary tracking rewards but targeting the negated command.
    # Equal weights create a symmetric, genuinely infeasible conflict for GCR-PPO.
    track_lin_vel_xy_conflict_exp: RewTerm = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "conflict_velocity", "std": 0.5},
    )
    track_ang_vel_z_conflict_exp: RewTerm = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "conflict_velocity", "std": 0.5},
    )


##
# Environment configuration
##

@configclass
class G1FlatConflictEnvCfg(G1RoughEnvCfg):
    rewards: G1FlatConflictRewardsCfg = G1FlatConflictRewardsCfg()
    commands: G1FlatConflictCommandsCfg = G1FlatConflictCommandsCfg()
    observations: G1FlatConflictObsCfg = G1FlatConflictObsCfg()

    def __post_init__(self):
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Use full mesh
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        # Terminations for full mesh
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(60)},
        )
        self.terminations.low_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.3},
        )

        # Tune reward weights
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # Remove feet_air_time: it references base_velocity and would create an
        # asymmetric bias toward the primary tracking objective, breaking the
        # symmetry required for a genuine conflict task.
        self.rewards.feet_air_time = None
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # Tune command ranges (also sync conflict_velocity ranges so they match)
        # base_velocity: forward-left (positive x and positive y)
        # conflict_velocity mirrors with y_scale=-1.0: forward-right (same x, negated y)
        # Equal XY magnitude is guaranteed since conflict is a pure y sign-flip.
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.2, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.conflict_velocity.ranges = self.commands.base_velocity.ranges

        # -- multi-head critic support --
        self.reward_component_names = [
            name for name, val in self.rewards.__dict__.items()
            if isinstance(val, RewTerm)
        ]
        self.reward_components = len(self.reward_component_names)
        self.reward_component_task_rew = [
            "track_lin_vel_xy_conflict_exp",
            "track_ang_vel_z_conflict_exp",
            "alive",
        ]


class G1FlatConflictEnvCfg_PLAY(G1FlatConflictEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
