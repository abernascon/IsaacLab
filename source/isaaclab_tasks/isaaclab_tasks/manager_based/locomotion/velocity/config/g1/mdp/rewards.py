# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for G1 locomotion tasks."""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def track_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the squared difference between the commanded height and the robot's root z.

    Args:
        env: The RL environment.
        command_name: Key used to register the height command in CommandsCfg.
        asset_cfg: Scene entity for the robot.

    Returns:
        Per-environment L2 penalty, shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def track_height_rbf(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proximity to commanded height using a Radial Basis Function (Gaussian) kernel.

    Returns a positive reward in [0, 1] that decays smoothly as the robot moves away from
    the target height. This provides a clearer learning signal than L2 penalties.

    Args:
        env: The RL environment.
        command_name: Key used to register the height command in CommandsCfg.
        sigma: Standard deviation of the Gaussian kernel. Controls how quickly the reward
            decays with distance. Smaller values = steeper decay.
        asset_cfg: Scene entity for the robot.

    Returns:
        Per-environment RBF reward in [0, 1], shape ``(num_envs,)``.
        Returns 1.0 at exact target height, decaying to ~0 as distance increases.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]
    height_error = asset.data.root_pos_w[:, 2] - target_height
    # Gaussian RBF: exp(-distance^2 / (2*sigma^2))
    return torch.exp(-torch.square(height_error) / (2.0 * sigma**2))


def flat_feet_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat feet orientation.

    This function penalizes the deviation of the feet orientation from the flat orientation (z-up).
    It computes the sum of squares of the x and y components of the projected Z-axis,
    which corresponds to the non-yaw components of the rotation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get feet orientation
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 4)

    # Penalize x and y components of the quaternion (w, x, y, z)
    # This assumes the target orientation is z-up (flat), allowing for yaw rotation.
    # A rotation purely around Z-axis has x=0 and y=0.
    return torch.sum(torch.square(feet_quat_w[:, :, 1:3]), dim=-1).sum(dim=1)


def _plate_tilt_sq(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    plate_local_rot: tuple,
) -> torch.Tensor:
    """Compute tilt² of the plate's z-axis from world +z.

    The plate is mounted on the palm with a fixed local rotation ``plate_local_rot``
    (quaternion w,x,y,z relative to the palm frame).  The true plate orientation is:
        q_plate = q_palm ⊗ q_local
    and tilt is measured as x²+y² of q_plate (zero when plate z = world z).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]  # (N, 4)

    q_local = torch.tensor(plate_local_rot, dtype=torch.float32, device=palm_quat.device)
    q_local = q_local.unsqueeze(0).expand(palm_quat.shape[0], -1)   # (N, 4)

    plate_quat = math_utils.quat_mul(palm_quat, q_local)             # (N, 4)
    # x²+y² of the plate quaternion = 0 when plate z-axis points world +z
    return torch.square(plate_quat[:, 1]) + torch.square(plate_quat[:, 2])  # (N,)


def plate_orientation_rbf(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.2,
    plate_local_rot: tuple = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Reward keeping the plate horizontal.

    Computes the true plate orientation as q_palm ⊗ q_local (where ``plate_local_rot``
    is the plate's rotation relative to the palm body frame), then rewards alignment
    of the plate z-axis with world +z via an RBF kernel.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.
        sigma: RBF decay width in radians.
        plate_local_rot: Quaternion (w,x,y,z) of the plate relative to the palm frame.
                         Must match the ``rot`` set in the plate's ``init_state``.

    Returns:
        Per-environment RBF reward in [0, 1], shape ``(num_envs,)``.
    """
    tilt_sq = _plate_tilt_sq(env, asset_cfg, plate_local_rot)
    return torch.exp(-tilt_sq / (2.0 * sigma**2))


def plate_drop_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_angle: float = 0.5236,
    plate_local_rot: tuple = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Binary penalty when the plate tilts past ``max_tilt_angle`` from horizontal.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.
        max_tilt_angle: Tilt threshold in radians (default 30°).
        plate_local_rot: Quaternion (w,x,y,z) of the plate relative to the palm frame.
                         Must match the ``rot`` set in the plate's ``init_state``.

    Returns:
        Per-environment binary penalty (0 or 1), shape ``(num_envs,)``.
    """
    tilt_sq = _plate_tilt_sq(env, asset_cfg, plate_local_rot)
    threshold = math.sin(max_tilt_angle / 2.0) ** 2
    return (tilt_sq > threshold).float()
