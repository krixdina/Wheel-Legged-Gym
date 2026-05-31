# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import argparse
import os
import sys

import isaacgym
from isaacgym.torch_utils import *
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


DEFAULT_KEY_FRAME_DIR = "frames"


def extract_recording_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--key_frame_dir",
        "--key-frame-dir",
        "--key_frame_folder",
        "--key-frame-folder",
        dest="key_frame_dir",
        default=DEFAULT_KEY_FRAME_DIR,
        help="Folder name under logs/<experiment>/exported for saved key-frame images.",
    )
    recording_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_argv]
    recording_args.key_frame_dir = validate_key_frame_dir(recording_args.key_frame_dir)
    return recording_args


def validate_key_frame_dir(folder_name):
    folder_name = folder_name.strip()
    if not folder_name:
        raise ValueError("--key_frame_dir must not be empty.")
    if os.path.isabs(folder_name):
        raise ValueError("--key_frame_dir expects a folder name, not an absolute path.")
    if os.sep in folder_name or (os.altsep is not None and os.altsep in folder_name):
        raise ValueError("--key_frame_dir expects a folder name without path separators.")
    if folder_name in {".", ".."}:
        raise ValueError("--key_frame_dir must be a normal folder name.")
    return folder_name


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 20
    env_cfg.env.fail_to_terminal_time_s = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.1, 0.2]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    # env_cfg.control.feedforward_force = 80.0
    # env_cfg.control.kp_theta = 80.0
    # env_cfg.control.kd_theta = 4.0
    # env_cfg.control.kp_l0 = 700.0
    # env_cfg.control.kd_l0 = 70.0

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        # 这里导出的是已经从 checkpoint 恢复出来的 actor_critic 网络
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 9  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    latent = None
    frame_dir = None
    if RECORD_FRAMES:
        frame_dir = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            getattr(args, "key_frame_dir", DEFAULT_KEY_FRAME_DIR),
        )
        os.makedirs(frame_dir, exist_ok=True)
        print(f"Frame output directory: {frame_dir}")

    CoM_offset_compensate = False
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)

    for i in range(1000 * int(env.max_episode_length)):
        # 如果策略网络带有时序结构，就同时使用当前观测和历史观测，
        # 并取回网络内部估计出的隐变量；否则只基于当前观测直接输出动作。
        if ppo_runner.alg.actor_critic.is_sequence:
            actions, latent = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        # 分别设置前向速度、机身高度和偏航角速度目标。
        env.commands[:, 0] = 1.8
        env.commands[:, 2] = 0.15  # + 0.07 * np.sin(i * 0.01)
        env.commands[:, 3] = 0

        # 先生成一个分阶段上升的参考速度，再根据期望前向速度与实际前向速度的误差微调控制指令。
        if CoM_offset_compensate:
            if i > 2000 and i < 600:
                # 200 到 600 步之间，速度目标从 0 平滑爬升到 2.5
                vel_cmd[:] = 2.5 * np.clip((i - 200) * 0.05, 0, 1)
            elif i < 2000 :
                vel_cmd[:] = 0
            else: 
                vel_cmd[:] = 2.5
                
            # vel_err_intergral 表示“速度误差的积分补偿量”：
            # 这里使用离散形式的积分近似，也就是每一步都按“当前误差 * 时间步长 env.dt”累加。
            # 因此它不是只看当前这一拍误差，而是在持续记住过去一段时间里“总是偏快还是总是偏慢”。
            #
            # 最后一项 ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5) 是一个按环境分别生效的 0/1 门控：
            # 当“目标前向速度与实际前向速度”的误差绝对值小于 0.5 时，该项为 1，允许继续积分微调；
            # 当误差过大时，该项为 0，本步不再累加积分。
            # 这样做的目的，是避免机器人还远没跟上目标时就过度累积积分量，减小积分饱和和补偿过猛的风险。
            vel_err_intergral += (
                (vel_cmd - env.base_lin_vel[:, 0])
                * env.dt
                * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
            )
            vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
            env.commands[:, 0] = vel_cmd + vel_err_intergral

        obs, _, rews, dones, infos, obs_history = env.step(actions)
        # 如果开启录帧，就定期把当前查看器画面保存成图片，便于后续合成视频或逐帧分析。
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(frame_dir, f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        # 如果开启相机跟随，就把观察视角移动到指定机器人附近，使画面持续跟踪目标机器人。
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        # 在一段时间内记录单台机器人和单个关节的状态，
        # 这里的 logger.log_states(...) 不是立刻把内容打印出来，
        # 而是把“当前这一仿真步的字段名和数值”追加保存到 Logger 内部的 state_log 中。
        # 因此同一个字段在很多步里反复记录后，会形成一条按 dt 时间排列的历史序列，
        # 如 state_log["dof_pos"] = [第1步的值, 第2步的值, 第3步的值, ...]
        # 后续如果调用 logger.plot_states()，横轴通常就是，纵轴就是这里记录的对应物理量。
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_yaw": env.commands[robot_index, 1].item(),
                    "command_height": env.commands[robot_index, 2].item(),
                    "base_height": env.base_height[robot_index].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            if CoM_offset_compensate:
                logger.log_states({"command_x": vel_cmd[robot_index].item()})
            else:
                logger.log_states({"command_x": env.commands[robot_index, 0].item()})
            # 如果策略返回了隐变量，就额外记录网络估计出来的机身线速度；
            # 当隐变量维度更高且观测噪声开启时，还会对比“观测值”和“估计值”的差异。
            if latent is not None:
                logger.log_states(
                    {
                        "est_lin_vel_x": latent[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": latent[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": latent[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
                if latent.shape[1] > 3 and env_cfg.noise.add_noise:
                    logger.log_states(
                        {
                            "base_vel_yaw_obs": obs[robot_index, 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_obs": obs[robot_index, 9 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_obs": obs[robot_index, 15 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
                    logger.log_states(
                        {
                            "base_vel_yaw_est": latent[robot_index, 3 + 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_est": latent[
                                robot_index, 3 + 9 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_est": latent[
                                robot_index, 3 + 15 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
        elif i == stop_state_log:
            # logger.plot_states()
            pass
        # 在若干回合结束后累计每回合奖励，并在设定步数达到阈值时打印平均奖励统计。
        if 0 < i < stop_rew_log:
            # infos["episode"] 只有在这一仿真步里有并行环境的 episode 刚好结束时才会有有效统计；
            # 它携带的是这一批刚结束回合的奖励分项信息
            if infos["episode"]:
                # 因此这里求和得到的是“这一仿真步总共结束了多少个 episode”，
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    # log_rewards(...) 会把这一批 episode 的奖励统计按 num_episodes 做累计，
                    # 后续 print_rewards() 再用“累计奖励总和 / 累计结束的 episode 总数”得到平均奖励。
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    MOVE_CAMERA = True
    recording_args = extract_recording_args()
    args = get_args()
    args.key_frame_dir = recording_args.key_frame_dir
    play(args)
