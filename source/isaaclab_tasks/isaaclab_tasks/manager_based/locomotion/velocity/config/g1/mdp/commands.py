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


@configclass
class WaiterVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for WaiterVelocityCommand."""

    class_type: type = WaiterVelocityCommand

    palm_body_name: str = MISSING
    """Name of the palm rigid body whose orientation is monitored."""
