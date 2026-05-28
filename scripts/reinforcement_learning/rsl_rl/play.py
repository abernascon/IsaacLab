# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval",
    action="store_true",
    default=False,
    help="Collect and print evaluation metrics (mean reward, episode length, per-reward-term stats).",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=50,
    help="Number of episodes to collect when --eval is enabled.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # convert agent cfg to dict and inject multi-head / PCGrad flags
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["use_critic_multi"] = getattr(args_cli, "use_critic_multi", False)
    agent_cfg_dict["use_pcgrad"] = getattr(args_cli, "use_pcgrad", False)

    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ------------------------------------------------------------------
    # Eval state — only used when --eval is passed
    # ------------------------------------------------------------------
    import math as _math

    do_eval = args_cli.eval
    num_eval_episodes = args_cli.eval_episodes
    num_envs = env.unwrapped.num_envs

    # Map from reward term name → (inversion kernel, display label).
    # We read reward_manager._step_reward directly (per-env, per-step, no dt scaling)
    # and invert per step before averaging — this is unbiased.
    # infos["log"] is NOT used: it is a scalar averaged across resetting envs,
    # only populated on reset steps, and stale otherwise.
    #
    # Kernels match the reward functions in rewards.py:
    #   track_palm_lin_vel_xy_yaw_frame_exp : r = exp(-‖error_xy‖²/std²), std=0.5
    #   track_palm_ang_vel_z_world_exp      : r = exp(-error_yaw²/std²),   std=0.5
    #   palm_orientation_proj_gravity       : r = exp(-sin²θ/(2·σ²)),      σ=0.2
    _PHYS_TERMS: dict[str, tuple] = {
        # Waiter env: palm velocity tracking
        # r = exp(-‖error_xy‖²/std²), std=0.5  →  ‖error_xy‖ in m/s
        "track_hand_lin_vel_xy_exp": (
            lambda r: _math.sqrt(max(-0.5**2 * _math.log(max(r, 1e-8)), 0.0)),
            "palm_lin_vel_xy_error  (m/s)",
        ),
        # r = exp(-error_yaw²/std²), std=0.5  →  |error_yaw| in rad/s
        "track_hand_ang_vel_z_exp": (
            lambda r: _math.sqrt(max(-0.5**2 * _math.log(max(r, 1e-8)), 0.0)),
            "palm_ang_vel_z_error   (rad/s)",
        ),
        # r = exp(-sin²θ/(2·σ²)), σ=0.2  →  θ in degrees
        "plate_orientation_exp": (
            lambda r: _math.degrees(
                _math.asin(min(_math.sqrt(max(-2 * 0.2**2 * _math.log(max(r, 1e-8)), 0.0)), 1.0))
            ),
            "plate_tilt_angle       (deg)",
        ),
        # Low-height env: base height tracking
        # r = exp(-error²/(2·σ²)), σ=0.3  →  |height_error| in m
        "track_height_rbf": (
            lambda r: _math.sqrt(max(-2 * 0.3**2 * _math.log(max(r, 1e-8)), 0.0)),
            "height_error           (m)",
        ),
        # Base velocity tracking (low-height env uses torso velocity, not palm)
        # r = exp(-‖error_xy‖²/std²), std=0.5, weight=1.0  →  ‖error_xy‖ in m/s
        "track_lin_vel_xy_exp": (
            lambda r: _math.sqrt(max(-0.5**2 * _math.log(max(r, 1e-8)), 0.0)),
            "base_lin_vel_xy_error  (m/s)",
        ),
        # weight=2.0 → _step_reward stores func_output*2.0; divide by weight before inverting
        # r = exp(-error_yaw²/std²), std=0.5, weight=2.0  →  |error_yaw| in rad/s
        "track_ang_vel_z_exp": (
            lambda r: _math.sqrt(max(-0.5**2 * _math.log(max(min(r / 2.0, 1.0), 1e-8)), 0.0)),
            "base_ang_vel_z_error   (rad/s)",
        ),
    }

    # Resolve term names → column indices in reward_manager._step_reward
    rew_mgr = env.unwrapped.reward_manager
    term_names: list[str] = rew_mgr._term_names
    phys_col: dict[str, int] = {}   # term_name → column index
    for term_name in _PHYS_TERMS:
        if term_name in term_names:
            phys_col[term_name] = term_names.index(term_name)

    # per-env running accumulators
    eval_ep_reward = torch.zeros(num_envs, device=env.unwrapped.device)
    eval_ep_length = torch.zeros(num_envs, device=env.unwrapped.device)
    # accumulate raw reward function output for every term (for summary table)
    eval_ep_term = torch.zeros(num_envs, len(term_names), device=env.unwrapped.device)
    eval_ep_phys: dict[str, torch.Tensor] = {
        name: torch.zeros(num_envs, device=env.unwrapped.device) for name in phys_col
    }

    # completed episode records
    finished_rewards: list[float] = []
    finished_lengths: list[float] = []
    finished_terms: dict[str, list[float]] = {}   # mean per-step reward per term
    finished_phys: dict[str, list[float]] = {}

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # ------------------------------------------------------------------
        # Eval bookkeeping
        # ------------------------------------------------------------------
        if do_eval:
            # with multi-head critic rewards has shape (num_envs, num_components) — sum to get total
            total_rewards = rewards.sum(dim=-1) if rewards.dim() > 1 else rewards
            eval_ep_reward += total_rewards
            eval_ep_length += 1

            # Read per-step reward values directly from the reward manager.
            # _step_reward shape: (num_envs, num_terms), values = func_output * weight (no dt).
            # For our terms weight=1.0, so this equals the raw reward function output in [0,1].
            # Invert the kernel per step per env → unbiased physical metric.
            step_rew = rew_mgr._step_reward.detach()
            # accumulate raw func output for all terms
            eval_ep_term += step_rew
            # accumulate physical metrics (inverted per step — unbiased)
            for term_name, col in phys_col.items():
                kernel, _ = _PHYS_TERMS[term_name]
                per_env_r = step_rew[:, col]
                eval_ep_phys[term_name] += torch.tensor(
                    [kernel(v) for v in per_env_r.tolist()],
                    device=eval_ep_phys[term_name].device,
                )

            # on each env that finished an episode, record and reset
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids:
                i = int(idx.item())
                ep_len = max(eval_ep_length[i].item(), 1)
                finished_rewards.append(eval_ep_reward[i].item())
                finished_lengths.append(eval_ep_length[i].item())
                for col, name in enumerate(term_names):
                    finished_terms.setdefault(name, []).append(
                        (eval_ep_term[i, col] / ep_len).item()
                    )
                for term_name in eval_ep_phys:
                    finished_phys.setdefault(term_name, []).append(
                        (eval_ep_phys[term_name][i] / ep_len).item()
                    )
                eval_ep_reward[i] = 0.0
                eval_ep_length[i] = 0.0
                eval_ep_term[i] = 0.0
                for term_name in eval_ep_phys:
                    eval_ep_phys[term_name][i] = 0.0

                n = len(finished_rewards)
                print(
                    f"[EVAL] Episode {n}/{num_eval_episodes} — "
                    f"ep_reward: {finished_rewards[-1]:.2f}  "
                    f"ep_length: {finished_lengths[-1]:.0f}"
                )

            if len(finished_rewards) >= num_eval_episodes:
                break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Eval summary
    # ------------------------------------------------------------------
    if do_eval and finished_rewards:
        import statistics

        def _fmt(vals: list[float]) -> str:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mean:>10.4f} ± {std:.4f}"

        print("\n" + "=" * 60)
        print(f"  Evaluation Summary  ({len(finished_rewards)} episodes, {num_envs} envs)")
        print("=" * 60)
        print(f"  {'episode_reward':<40s} {_fmt(finished_rewards)}")
        print(f"  {'episode_length':<40s} {_fmt(finished_lengths)}")

        # Per-term mean reward (raw func output, mean per step over episode)
        if finished_terms:
            print("  " + "-" * 58)
            for name, vals in finished_terms.items():
                print(f"  {name:<40s} {_fmt(vals)}")

        # Physical metrics: inverted per step from _step_reward, then averaged — unbiased
        if finished_phys:
            print("  " + "-" * 58)
            for term_name, vals in finished_phys.items():
                _, label = _PHYS_TERMS[term_name]
                print(f"  {label:<40s} {_fmt(vals)}")

        print("=" * 60 + "\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
