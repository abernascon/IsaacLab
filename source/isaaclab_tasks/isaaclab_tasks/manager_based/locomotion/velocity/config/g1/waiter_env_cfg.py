# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import G1FlatEnvCfg
from .mdp import plate_orientation_exp, palm_lin_vel_penalty  # , plate_drop_penalty


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

        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

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
        # Rewards
        # ------------------------------------------------------------------
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.25)

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
            weight=-1.0,
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

        # Penalize shoulder roll deviation more strongly to prevent arm closing into torso
        self.rewards.joint_deviation_shoulder = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_shoulder_roll_joint"],
                ),
            },
        )

        # palm +Y points world +Z when the palm faces up (waiter pose).
        PALM_UP_LOCAL = (0.0, 1.0, 0.0)

        # Power-law reward: palm +Y aligned with world +z (plate flat)
        self.rewards.plate_orientation_exp = RewTerm(
            func=plate_orientation_exp,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_palm_link"),
                "exponent": 2.0,
                "palm_up_local": PALM_UP_LOCAL,
            },
        )


        # Small penalty on right palm linear velocity to reduce oscillation
        self.rewards.palm_lin_vel = RewTerm(
            func=palm_lin_vel_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_palm_link"),
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
        self.reward_component_task_rew = ["plate_orientation_exp", "alive", "palm_lin_vel"]


class G1WaiterEnvCfg_PLAY(G1WaiterEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
