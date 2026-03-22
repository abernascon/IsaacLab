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


def _palm_tilt_sq(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    palm_up_local: tuple = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Directional tilt of the palm's local up-axis from world +Z.

    Rotates ``palm_up_local`` into world frame and returns (1 - z) / 2,
    which is 0 when the axis points straight up and 1 when pointing straight down.
    Unlike x²+y², this is NOT symmetric: palm-down and palm-up give different values.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]  # (N, 4)

    up_local = torch.tensor(palm_up_local, dtype=torch.float32, device=palm_quat.device)
    up_local = up_local.unsqueeze(0).expand(palm_quat.shape[0], -1)  # (N, 3)

    up_world = math_utils.quat_apply(palm_quat, up_local)             # (N, 3)
    return (1.0 - up_world[:, 2]) / 2.0                               # (N,) in [0, 1]


def plate_orientation_rbf(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.2,
    palm_up_local: tuple = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Reward keeping the plate horizontal via an RBF kernel.

    Checks that the palm's local up-axis (``palm_up_local``) points world +z.
    Since the plate is fixed to the palm, this directly measures plate tilt.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.
        sigma: RBF decay width (in sin-space). Smaller = steeper penalty for tilt.
        palm_up_local: Palm-frame unit vector that should point world +z when tray is flat.

    Returns:
        Per-environment RBF reward in [0, 1], shape ``(num_envs,)``.
    """
    tilt_sq = _palm_tilt_sq(env, asset_cfg, palm_up_local)
    return torch.exp(-tilt_sq / (2.0 * sigma**2))


def plate_drop_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_angle: float = 0.5236,
    palm_up_local: tuple = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Binary penalty when the plate tilts past ``max_tilt_angle`` from horizontal.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.
        max_tilt_angle: Tilt threshold in radians (default 30°).
        palm_up_local: Palm-frame unit vector that should point world +z when tray is flat.

    Returns:
        Per-environment binary penalty (0 or 1), shape ``(num_envs,)``.
    """
    tilt_sq = _palm_tilt_sq(env, asset_cfg, palm_up_local)
    threshold = math.sin(max_tilt_angle / 2.0) ** 2
    return (tilt_sq > threshold).float()
