# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height command term — mirrors the structure of UniformVelocityCommand."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformHeightCommand(CommandTerm):
    """Resamples a scalar target height uniformly each episode.

    The command is a 1-D tensor ``(num_envs, 1)`` holding the desired root z height
    in the world frame.  It is resampled at the standard command resampling rate.
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # buffer: shape (num_envs, 1)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired root height in world frame. Shape: (num_envs, 1)."""
        return self.height_command

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        pass  # no post-processing needed

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_height"] += (
            torch.abs(self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2]) / max_command_step
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass  # no visualisation for a scalar command


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for UniformHeightCommand."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """Range (min, max) of the desired root height in metres."""

    ranges: Ranges = MISSING
