# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waiter velocity command — extends UniformVelocityCommand with a plate-tilt error metric."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class WaiterVelocityCommand(UniformVelocityCommand):
    """Velocity command targeting a specific body (e.g. the waiter's palm) with a plate tilt metric.

    Differences vs. :class:`UniformVelocityCommand`:
      - ``error_vel_xy`` / ``error_vel_yaw`` metrics are computed against the
        *palm* velocity (in its yaw-aligned frame) rather than the robot base,
        so they reflect how well the command is being followed by the body
        this command is actually driving.
      - ``error_plate_tilt`` is added as an extra metric: 0 when the palm +Y
        axis points world +Z (plate flat), up to 1 when fully inverted.
    """

    cfg: WaiterVelocityCommandCfg

    def __init__(self, cfg: WaiterVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        palm_ids, _ = self.robot.find_bodies(cfg.palm_body_name)
        self._palm_body_idx: int = palm_ids[0]
        self.metrics["error_plate_tilt"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_palm_height"] = torch.zeros(self.num_envs, device=self.device)

    def _update_command(self):
        """Compute angular velocity from heading error using the **palm's** yaw, not the robot root.

        For heading-command envs the desired yaw rate is a P-controller on the
        error between the sampled heading target and the palm body's current
        yaw orientation.  This ensures the palm turns to face its own heading
        target independently of the torso.
        """
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # Palm yaw in world frame
            palm_quat = self.robot.data.body_quat_w[:, self._palm_body_idx, :]  # (N, 4)
            forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=palm_quat.device).expand(palm_quat.shape[0], -1)
            forward_w = math_utils.quat_apply(palm_quat, forward_local)  # (N, 3)
            palm_heading = torch.atan2(forward_w[:, 1], forward_w[:, 0])  # (N,)

            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - palm_heading[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (zero velocity) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _update_metrics(self):
        # NOTE: intentionally do NOT call super() — the parent computes error
        # against the robot root; we want error against the palm body instead.
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        palm_quat = self.robot.data.body_quat_w[:, self._palm_body_idx, :]           # (N, 4)
        palm_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._palm_body_idx, :]   # (N, 3)
        palm_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._palm_body_idx, :]   # (N, 3)

        # Linear velocity error in the palm's yaw-aligned frame (matches the reward).
        palm_lin_vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(palm_quat), palm_lin_vel_w)
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - palm_lin_vel_yaw[:, :2], dim=-1) / max_command_step
        )
        # Yaw rate error in world frame (matches the reward).
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - palm_ang_vel_w[:, 2]) / max_command_step
        )

        # Plate tilt metric: full squared distance from the flat-tray target direction.
        g_palm = math_utils.quat_apply_inverse(palm_quat, self.robot.data.GRAVITY_VEC_W)  # (N, 3)
        self.metrics["error_plate_tilt"] = (
            g_palm[:, 0].square() + (g_palm[:, 1] + 1.0).square() + g_palm[:, 2].square()
        ) / 4.0

        # Palm height metric: absolute deviation from configured target height.
        palm_z = self.robot.data.body_pos_w[:, self._palm_body_idx, 2]  # (N,)
        self.metrics["error_palm_height"] = torch.abs(palm_z - self.cfg.palm_target_height)


@configclass
class WaiterVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for WaiterVelocityCommand."""

    class_type: type = WaiterVelocityCommand

    palm_body_name: str = MISSING
    """Name of the palm rigid body whose orientation is monitored."""

    palm_target_height: float = 1.0
    """Target palm height (meters) used for the ``error_palm_height`` metric."""


class ScaledVelocityCommand(UniformVelocityCommand):
    """Velocity command that mirrors another command with independent XY and yaw scale factors.

    Instead of sampling its own velocities, this command reads from a source
    command (looked up by name in the command manager) every step and scales
    the linear XY components by ``speed_scale`` and the yaw component by
    ``yaw_scale``.

    Setting both to -1.0 creates a full velocity conflict for GCR-PPO: every
    component of the command points in the opposite direction of the source.
    """

    cfg: ScaledVelocityCommandCfg

    def __init__(self, cfg: ScaledVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._source_command_name = cfg.source_command_name
        self._speed_scale = cfg.speed_scale
        self._yaw_scale = cfg.yaw_scale

    def _resample_command(self, env_ids):
        # Sync from source for the resampled envs (covers reset path).
        source_cmd = self._env.command_manager.get_command(self._source_command_name)
        self.vel_command_b[env_ids, :2] = source_cmd[env_ids, :2] * self._speed_scale
        self.vel_command_b[env_ids, 2] = source_cmd[env_ids, 2] * self._yaw_scale

    def _update_command(self):
        # Full sync every step (covers mid-episode resampling of the source).
        source_cmd = self._env.command_manager.get_command(self._source_command_name)
        self.vel_command_b[:, :2] = source_cmd[:, :2] * self._speed_scale
        self.vel_command_b[:, 2] = source_cmd[:, 2] * self._yaw_scale


@configclass
class ScaledVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for ScaledVelocityCommand."""

    class_type: type = ScaledVelocityCommand

    source_command_name: str = MISSING
    """Name of the command to mirror (must be registered before this one)."""

    speed_scale: float = 0.5
    """Multiplier applied to the source command's linear XY velocities."""

    yaw_scale: float = 1.0
    """Multiplier applied to the source command's yaw velocity. Set to -1.0 to conflict."""
