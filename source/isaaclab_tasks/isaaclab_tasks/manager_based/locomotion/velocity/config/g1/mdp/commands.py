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
    """Velocity command that additionally tracks plate tilt error as a TensorBoard metric.

    Adds ``error_plate_tilt`` to ``self.metrics``, logged as
    ``Metrics/base_velocity/error_plate_tilt``.  Value is 0 when the plate is
    perfectly flat and increases with tilt (same normalisation as the velocity
    error metrics).
    """

    cfg: WaiterVelocityCommandCfg

    def __init__(self, cfg: WaiterVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        palm_ids, _ = self.robot.find_bodies(cfg.palm_body_name)
        self._palm_body_idx: int = palm_ids[0]
        self.metrics["error_plate_tilt"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        palm_quat = self.robot.data.body_quat_w[:, self._palm_body_idx, :]  # (N, 4)
        g_palm = math_utils.quat_apply_inverse(palm_quat, self.robot.data.GRAVITY_VEC_W)  # (N, 3)
        # tilt_sq = 0 when palm +Y faces world +Z (plate flat), up to 1 when fully inverted.
        # Identical to the tilt measure used in palm_orientation_proj_gravity.
        self.metrics["error_plate_tilt"] = (
            g_palm[:, 0].square() + (g_palm[:, 1] + 1.0).square() + g_palm[:, 2].square()
        ) / 4.0


@configclass
class WaiterVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for WaiterVelocityCommand."""

    class_type: type = WaiterVelocityCommand

    palm_body_name: str = MISSING
    """Name of the palm rigid body whose orientation is monitored."""
