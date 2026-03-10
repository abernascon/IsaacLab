# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward term that tracks the height command."""

from __future__ import annotations

import torch

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
