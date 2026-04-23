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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


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


def palm_orientation_proj_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """Reward keeping the plate horizontal via projected gravity in the palm frame.

    Projects world gravity into the palm's local frame (same operation as
    ``projected_gravity_b`` used for the robot base). When the palm is flat
    (tray pose, palm +Y pointing world +Z), gravity in the palm frame is
    ``(0, -1, 0)``. Any tilt rotates gravity into the X-Z plane; the penalty
    term is ``sin²θ = g_palm_x² + g_palm_z²``, zero when perfectly flat.

    An RBF kernel converts this into a positive reward in (0, 1]:
        reward = exp(-sin²θ / (2·σ²))

    This is directly analogous to ``flat_orientation_l2`` for the base and uses
    the same representation as the ``palm_projected_gravity_b`` observation term,
    so the policy receives the exact same signal it is being rewarded on.

    Args:
        env:       The RL environment.
        asset_cfg: Scene entity with ``body_names`` set to the palm link.
        sigma:     RBF width controlling tilt sensitivity.
                   σ=0.5 → reward 0.61 at 30°, 0.37 at 45°
                   σ=0.25 → reward 0.13 at 30°, 0.51 at 15°

    Returns:
        Per-environment reward in (0, 1], shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]        # (N, 4)
    g_palm = math_utils.quat_apply_inverse(palm_quat, asset.data.GRAVITY_VEC_W)  # (N, 3)
    # Target: g_palm = (0, -1, 0) when palm +Y faces world +Z (tray pose).
    # Use full squared distance from target direction, normalised to [0, 1]:
    #   ||g_palm - (0,-1,0)||² / 4  →  0 when flat, 1 when fully flipped.
    # This breaks the palm-up / palm-down symmetry that X²+Z² alone cannot.
    tilt_sq = (g_palm[:, 0].square() + (g_palm[:, 1] + 1.0).square() + g_palm[:, 2].square()) / 4.0
    return torch.exp(-tilt_sq / (2.0 * sigma ** 2))


def plate_orientation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    exponent: float = 2.0,
    palm_up_local: tuple = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Reward keeping the plate horizontal using a power-law on the alignment score.

    Computes ``((1 + dot(palm_up_world, world_z)) / 2) ^ exponent``.
    This maps the dot product from [-1, 1] to [0, 1] and raises it to a power,
    giving exactly 0 when fully upside-down and a smooth gradient everywhere else.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.
        exponent: Power to raise the alignment score to. Higher = stricter near upright.
        palm_up_local: Palm-frame unit vector that should point world +z when tray is flat.

    Returns:
        Per-environment reward in [0, 1], shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]  # (N, 4)

    up_local = torch.tensor(palm_up_local, dtype=torch.float32, device=palm_quat.device)
    up_local = up_local.unsqueeze(0).expand(palm_quat.shape[0], -1)  # (N, 3)

    up_world = math_utils.quat_apply(palm_quat, up_local)  # (N, 3)
    alignment = (1.0 + up_world[:, 2]) / 2.0  # (N,) in [0, 1]
    return alignment.pow(exponent)


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


def track_palm_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using the palm's velocity.

    Transforms the palm's world-frame linear velocity into the gravity-aligned
    (yaw-only) frame of the palm, then compares XY components against the command.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]       # (N, 4)
    palm_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]  # (N, 3)
    # Project palm velocity into the palm's yaw-aligned frame
    vel_yaw = quat_apply_inverse(yaw_quat(palm_quat), palm_lin_vel_w)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_palm_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using the palm's angular velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]  # (N, 3)
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - palm_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def palm_height_penalty(
    env: ManagerBasedRLEnv,
    target_height: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of the right palm height from a target height.

    Returns the squared difference between the palm's world-frame z position
    and the target height.

    Args:
        env: The RL environment.
        target_height: Desired palm height in world frame (meters).
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.

    Returns:
        Per-environment penalty, shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]  # (N,)
    return torch.square(palm_z - target_height)


def palm_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 1.0,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the palm at a target height using an exponential (RBF) kernel.

    Returns exp(-(palm_z - target)² / sigma²), which is 1.0 at the target and
    decays smoothly. With sigma=0.1 the reward is ~0.37 at 10 cm off-target and
    ~0.02 at 20 cm off-target, giving a much stronger learning signal than a
    squared penalty near the goal.

    Args:
        env: The RL environment.
        target_height: Desired palm height in world frame (meters).
        sigma: RBF width — smaller = sharper peak, stronger gradient near target.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.

    Returns:
        Per-environment reward in (0, 1], shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]  # (N,)
    return torch.exp(-torch.square(palm_z - target_height) / (sigma ** 2))


def palm_lin_vel_yaw_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Palm linear velocity projected into the palm's yaw-aligned frame, shape ``(N, 3)``.

    Matches the frame used by :func:`track_palm_lin_vel_xy_yaw_frame_exp`, so the
    policy observes the exact velocity component it is being rewarded on.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    palm_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    return quat_apply_inverse(yaw_quat(palm_quat), palm_lin_vel_w)


def palm_lin_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity of the palm link to reduce unnecessary hand movement.

    Returns the squared norm of the palm's world-frame linear velocity.

    Args:
        env: The RL environment.
        asset_cfg: Scene entity for the robot, with ``body_names`` set to the palm link.

    Returns:
        Per-environment penalty, shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    palm_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]  # (N, 3)
    return torch.sum(torch.square(palm_vel), dim=-1)
