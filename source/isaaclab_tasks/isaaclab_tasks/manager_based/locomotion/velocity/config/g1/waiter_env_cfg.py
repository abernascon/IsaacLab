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
from .mdp import plate_orientation_rbf, plate_drop_penalty


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
        - ``plate_orientation_rbf``: keep palm z-axis pointing world-up (plate flat)
        - ``plate_drop_penalty``:  large penalty when tilt > 30°

    Arm initialisation:
        Right arm is set to a ~90° forward waiter pose so training starts from a
        tray-compatible configuration.  The inherited ``joint_deviation_arms``
        penalty uses this as its reference.
    """

    def __post_init__(self):
        super().__post_init__()

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
                collision_props=None,      # disable to avoid self-collision artefacts
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
                pos=(0.1, 0.0, 0.0),
                rot=(0.707, -0.707, 0.0, 0.0),  # -90° around X: palm +Y = plate normal (flat when palm faces up)
            ),
        )

        # ------------------------------------------------------------------
        # Right arm waiter pose initialisation
        # ------------------------------------------------------------------
        # The base G1 config has ".*_elbow_pitch_joint": 0.87 (wildcard).
        # Adding "right_elbow_pitch_joint" on top would create a double-match.
        # Fix: remove the wildcard and set left/right explicitly.
        self.scene.robot.init_state.joint_pos.pop(".*_elbow_pitch_joint", None)
        self.scene.robot.init_state.joint_pos.update({
            "right_shoulder_pitch_joint": 0.35,   # standard natural pose
            "right_shoulder_roll_joint": -0.16,   # standard natural pose
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.87,       # keep left at default
            "right_elbow_pitch_joint": 0.0,      
            "right_elbow_roll_joint": 1.57,       # supinate forearm → palm faces up
        })

        # ------------------------------------------------------------------
        # Rewards
        # ------------------------------------------------------------------
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.25)

        # PLATE_LOCAL_ROT defines which palm axis is the plate surface normal.
        # In the standard G1 pose, palm Y points world +Y. In the waiter pose (palm up),
        # palm Y should point world +Z. So we track palm +Y = world +Z.
        # Rx(-90°) * [0,0,1] = [0,+1,0] → reward checks R(q_palm)*[0,1,0] = world +z.
        PLATE_LOCAL_ROT = (0.707, -0.707, 0.0, 0.0)  # -90° around X: track palm +Y pointing world +z

        # RBF reward: plate z-axis (after local rotation) aligned with world +z
        self.rewards.plate_orientation_rbf = RewTerm(
            func=plate_orientation_rbf,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_palm_link"),
                "sigma": 0.2,
                "plate_local_rot": PLATE_LOCAL_ROT,
            },
        )

        # Large penalty when plate tilts more than 30° (plate "drops")
        self.rewards.plate_drop_penalty = RewTerm(
            func=plate_drop_penalty,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_palm_link"),
                "max_tilt_angle": 0.5236,  # 30 degrees
                "plate_local_rot": PLATE_LOCAL_ROT,
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
        self.reward_component_task_rew = ["plate_orientation_rbf", "alive", "plate_drop_penalty"]


class G1WaiterEnvCfg_PLAY(G1WaiterEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
