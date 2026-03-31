# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import G1FlatEnvCfg

from isaaclab_assets import G1_CFG  # isort: skip
from .mdp import palm_orientation_proj_gravity, palm_lin_vel_penalty, track_palm_lin_vel_xy_yaw_frame_exp, track_palm_ang_vel_z_world_exp, WaiterVelocityCommandCfg  # , plate_drop_penalty


@configclass
class G1WaiterEnvCfg(G1FlatEnvCfg):
    """G1 waiter task: balance a circular plate fixed to the right palm while locomoting.

    Built on top of G1FlatEnvCfg (flat terrain, full locomotion reward stack).

    Plate attachment:
        The plate cylinder is spawned at ``{ENV_REGEX_NS}/Robot/right_palm_link/plate``,
        making it a **direct USD child of the palm link**.  Because it carries no
        ``RigidBodyAPI`` of its own, PhysX merges it into the palm link's rigid body
        as a compound collision/mass shape.  The plate therefore:
          - follows the palm exactly (no joints, no events, no teleport)
          - contributes its 400 g mass to the palm rigid body, creating real arm load
          - has visual geometry (off-white ceramic cylinder)

        ``init_state.pos`` is the plate's local offset inside the palm frame.
        Tune this once you inspect ``right_palm_link``'s frame gizmo in Isaac Sim.

    Rewards added vs. flat env:
        - ``alive``:               survival reward
        - ``plate_orientation_exp``: keep palm z-axis pointing world-up (plate flat)
        - ``plate_drop_penalty``:  large penalty when tilt > 30°

    Arm initialisation:
        Right arm is set to a ~90° forward waiter pose so training starts from a
        tray-compatible configuration.  The inherited ``joint_deviation_arms``
        penalty uses this as its reference.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace base_velocity command with waiter variant that logs plate tilt error
        old = self.commands.base_velocity
        self.commands.base_velocity = WaiterVelocityCommandCfg(
            palm_body_name="right_palm_link",
            asset_name=old.asset_name,
            resampling_time_range=old.resampling_time_range,
            rel_standing_envs=old.rel_standing_envs,
            rel_heading_envs=old.rel_heading_envs,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            debug_vis=old.debug_vis,
            ranges=old.ranges,
        )

        # Switch to full G1 mesh (g1.usd) for accurate mass distribution and
        # self-collision geometry. G1_MINIMAL_CFG strips most collision shapes.
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.init_state.joint_pos["right_elbow_roll_joint"] = 1.57  # supinate forearm → palm faces up
        # Override the wildcard ".*_elbow_pitch_joint" to set right side differently
        self.scene.robot.init_state.joint_pos["left_elbow_pitch_joint"] = 0.87  # keep left at default
        self.scene.robot.init_state.joint_pos["right_elbow_pitch_joint"] = 0.5  # a bit bent forward → more natural waiter pose
        del self.scene.robot.init_state.joint_pos[".*_elbow_pitch_joint"]  # remove wildcard to avoid conflict

        # ------------------------------------------------------------------
        # Scene: plate as a compound shape on the palm link
        # ------------------------------------------------------------------
        # Spawned as a CHILD of right_palm_link → no RigidBodyAPI → it becomes
        # part of the palm's rigid body in PhysX.
        #   radius = 0.15 m  (30 cm plate)
        #   height = 0.01 m  (1 cm thin)
        #   mass   = 0.40 kg (ceramic plate)
        # Collision is disabled to avoid self-collision with other robot links;
        # the mass is still applied via PhysicsMassAPI.
        # Set collision_props=sim_utils.CollisionPropertiesCfg() to re-enable.
        self.scene.plate = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_palm_link/plate",
            spawn=sim_utils.CylinderCfg(
                radius=0.15,
                height=0.01,
                rigid_props=None,          # no separate rigid body → merged with palm
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                collision_props=sim_utils.CollisionPropertiesCfg(),  # enabled: prevents arm compenetrating body
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.93, 0.88),  # off-white / ceramic
                    metallic=0.0,
                    roughness=0.6,
                ),
            ),
            # Local offset and rotation inside the palm frame.
            # Try each rot until the flat face is horizontal (like a tray):
            #   identity          (1.000, 0.000, 0.000, 0.000)  palm Z  = plate normal
            #   90° around X      (0.707, 0.707, 0.000, 0.000)  palm Y  = plate normal
            #   90° around Z      (0.707, 0.000, 0.000, 0.707)  palm X  = plate normal
            # Also update PLATE_LOCAL_ROT below to match whichever you pick.
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.1, 0.05, 0.0),           # offset along palm +Y (the "up" axis in waiter pose)
                rot=(0.707, -0.707, 0.0, 0.0),  # -90° around X: cylinder flat face ⊥ palm +Y
            ),
        )

        # ------------------------------------------------------------------
        # Terminations — replace contact-based check with geometry-independent ones
        # ------------------------------------------------------------------
        # bad_orientation fires when the torso tilts > 60° from upright.
        # root_height_below_minimum fires when the base drops below 0.5 m.
        # Both are instantaneous and don't depend on collision mesh quality.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(60)},
        )
        self.terminations.low_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.4},
        )

        # ------------------------------------------------------------------
        # Rewards
        # ------------------------------------------------------------------
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.25)

        # Override velocity tracking: track right palm velocity instead of torso
        palm_cfg = SceneEntityCfg("robot", body_names="right_palm_link")
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=track_palm_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": palm_cfg},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=track_palm_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": palm_cfg},
        )

        # Reduce feet_air_time weight to prevent GCR-PPO from exploiting
        # single-stance balancing (inherited weight=0.25 is too high)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.1,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 0.4,
            },
        )

        # Extend joint position limits penalty to arm joints
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-100.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_pitch_joint",
                        ".*_elbow_roll_joint",
                    ],
                ),
            },
        )

        # Penalize arm deviation from initial pose (both arms)
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_elbow_roll_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "right_elbow_roll_joint",
                    ],
                ),
            },
        )

        # Projected-gravity reward: palm +Y points world +Z when flat (tray pose)
        self.rewards.plate_orientation_exp = RewTerm(
            func=palm_orientation_proj_gravity,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_palm_link"),
                "sigma": 0.5,
            },
        )

        # ------------------------------------------------------------------
        # Multi-head critic bookkeeping (GCR-PPO)
        # ------------------------------------------------------------------
        self.reward_components = sum(
            isinstance(getattr(self.rewards, attr), RewTerm)
            for attr in dir(self.rewards)
            if not attr.startswith("__")
        )
        self.reward_component_names = [
            attr for attr in dir(self.rewards)
            if isinstance(getattr(self.rewards, attr), RewTerm) and not attr.startswith("__")
        ]
        self.reward_component_task_rew = ["alive", "termination_penalty", "plate_orientation_exp"]


class G1WaiterEnvCfg_PLAY(G1WaiterEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
