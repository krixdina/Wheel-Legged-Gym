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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from wheel_legged_gym.rsl_rl.algorithms import PPO
from wheel_legged_gym.rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticSequence,
)
from wheel_legged_gym.rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic

        if self.cfg["policy_class_name"] == "ActorCriticSequence":
            num_critic_obs += self.policy_cfg["latent_dim"]

        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 将配置中的算法类名字符串解析为当前作用域中的类对象，从字符串"PPO" -> PPO 类。
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        # 有多少个并行环境
        # 每轮采样多少步
        # 每个 observation 多大
        # critic observation 多大
        # 历史 observation 多大
        # action 多大
        # 初始化 rollout 存储器，方便后面收集一批轨迹再进行统一训练
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [self.env.obs_history_length * self.env.num_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    # 作用：执行 PPO 的主训练循环，循环中先收集一批环境交互数据，再计算回报并更新策略网络。
    # 输入：num_learning_iterations 表示本次调用要进行多少轮迭代，迭代的过程包括两步，数据采集和网络更新；
    #      init_at_random_ep_len 表示是否把各并行环境的初始回合长度打散，用来避免所有环境同步结束。
    # 输出：没有返回值；它会更新策略网络、价值网络、日志统计、checkpoint 文件和当前训练迭代位置。
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            # episode_length_buf 这个列表中的每个元素都是一个并行环境的回合计步器，后续环境每 step 一次都会让它加一。
            # 这里随机化的只是“当前回合已经运行了多少步”，不会改变机器人的真实物理状态。
            # 不同环境的计步器初值不同，就会在不同时间达到最大回合长度，从而避免所有环境同步超时重置。
            # 生成一个和 episode_length_buf 形状相同的随机整数张量，随机值范围是：[0, max_episode_length)
            #
            # 这个计步器后续主要在环境内部被用于：
            # - 判断是否达到单回合最大步数，从而置位 time_out_buf；
            # - 按固定时间间隔决定哪些环境需要重新采样 commands。
            #
            # max_episode_length 表示“一个 episode 最多允许多少个环境控制步”，
            # 它在环境初始化时按 ceil(episode_length_s / dt) 计算得到。
            # 以当前默认配置为例：
            # episode_length_s = 20 秒，sim.dt = 0.005 秒，decimation = 2，
            # 所以环境步长 dt = 0.01 秒，max_episode_length = ceil(20 / 0.01) = 2000。
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取训练开始时的环境观测，并选择供价值网络使用的观测；若环境提供特权观测，价值网络优先使用它。
        obs, obs_history = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, obs_history, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            critic_obs.to(self.device),
        )
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        # 准备日志统计缓存，用于记录最近结束回合的奖励、长度以及环境额外返回的 episode 指标。
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                # 用当前策略在所有并行环境中采样固定步数，并把每一步转移数据存入 PPO 的 rollout 缓冲区。
                # num_steps_per_env 用于表示每次执行 PPO 更新前要在每个环境中采集多少步数据；不是 max_episode_length 表示的回合上限步数
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, obs_history = (
                        self.env.step(actions)
                    )
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, obs_history, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
                        # Book keeping
                        # 只在写日志时维护回合级统计；某个环境结束后，把它的累计奖励和累计长度放入滑动窗口。
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # 为价值网络准备最后一个状态的观测，用于估计采样片段末尾之后的价值并计算优势函数。
                if self.cfg["policy_class_name"] == "ActorCriticSequence":
                    critic_obs__ = torch.cat(
                        (critic_obs, self.alg.actor_critic.encode(obs_history)), dim=-1
                    )
                else:
                    critic_obs__ = critic_obs
                self.alg.compute_returns(critic_obs__)

            mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss = (
                self.alg.update()
            )
            stop = time.time()
            learn_time = stop - start
            # 完成一轮更新后记录训练曲线、打印终端摘要，并按间隔保存当前模型。
            if self.log_dir is not None:
                # 以字典形式将当前循环的局部变量传入 log()
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()
        # 记录本次训练结束后的迭代位置，并额外保存一个最终 checkpoint。
        self.current_learning_iteration = num_learning_iterations
        self.save(
            os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations))
        )

    # 作用：在每次策略更新后汇总训练指标，同时写入 TensorBoard 并打印到终端。
    # 输入：locs 是 learn() 当前循环的局部变量字典，里面包含本轮采样耗时、学习耗时、损失、
    #      回合奖励缓存和回合长度缓存；width 和 pad 控制终端日志的显示宽度与对齐间距。
    # 输出：没有返回值；它会更新累计步数和累计时间，并通过 writer 记录曲线、通过 print 输出日志。
    def log(self, locs, width=80, pad=35):
        # 统计到目前为止已经采样的环境交互步数和真实耗时，用于展示训练进度与预计剩余时间。
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # 汇总环境在 episode 结束时返回的指标，例如各奖励项或任务统计量，并记录每个指标的平均值。
        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean {key}:':>{pad}} {value:.4f}\n"""
        # 计算策略动作分布的平均噪声大小和本轮训练速度，用于观察探索强度与训练性能。
        mean_std = self.alg.actor_critic.std.mean()
        # 这里的 fps 不是渲染帧率，而是训练吞吐率：
        # 本轮采样出的 transition 总数，除以“采样 rollout + PPO 更新”的总耗时。
        # 因此它反映的是整体训练流程每秒处理多少条环境交互数据。
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # 将损失、学习率、策略分布、KL 散度和性能耗时写入 TensorBoard，方便训练过程中画曲线观察。
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        # Perf 是 Performance 的缩写，用来把训练运行效率相关曲线归到同一组。
        # total_fps 是包含学习耗时后的总体 transition 吞吐率；
        # collection time 只统计 rollout 采样耗时；
        # learning_time 只统计本轮 PPO/网络更新耗时。
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )

        # 组装终端中显示的训练摘要；有完整 episode 结束时才会额外显示平均回合奖励和平均回合长度。
        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        # 在日志末尾追加累计交互步数、单轮耗时、总耗时和按当前平均速度估算的剩余时间。
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    # 作用：把当前策略网络、优化器状态和训练迭代位置保存成 checkpoint 文件，供之后续训或评估加载。
    # 输入：path 表示 checkpoint 保存路径；infos 表示调用方希望一并保存的额外信息，可以为空。
    # 输出：没有返回值；结果是磁盘上生成一个包含模型参数、优化器状态和迭代编号的文件。
    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    # 作用：从 checkpoint 文件恢复策略网络参数，并可选择是否恢复优化器状态，让训练能从保存点继续。
    # 输入：path 表示 checkpoint 文件路径；load_optimizer 表示是否恢复优化器内部状态，续训时通常需要恢复。
    # 输出：返回 checkpoint 中保存的额外信息，同时更新当前 runner 记录的训练迭代位置。
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    # 作用：取得用于测试或部署的策略推理函数，只根据观测输出动作，不执行训练更新。
    # 输入：device 表示希望把策略网络移动到哪个计算设备；为空时保持当前设备不变。
    # 输出：返回 actor_critic 中的推理接口，play.py 会用它在环境中直接计算动作。
    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
