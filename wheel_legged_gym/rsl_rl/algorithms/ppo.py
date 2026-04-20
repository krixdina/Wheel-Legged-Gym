#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from wheel_legged_gym.rsl_rl.modules import ActorCritic
from wheel_legged_gym.rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        extra_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        kl_decay=0,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.kl_decay = max(kl_decay, 0)
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.actor.parameters()},
                {"params": self.actor_critic.critic.parameters()},
                {"params": self.actor_critic.std},
            ],
            lr=learning_rate,
        )
        self.extra_optimizer = None
        if self.actor_critic.is_sequence:
            self.extra_optimizer = optim.Adam(
                [
                    {"params": self.actor_critic.encoder.parameters()},
                ],
                lr=extra_learning_rate,
            )
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    # 作用：在 rollout 采样阶段根据当前观测生成动作，并暂存 PPO 更新所需的旧策略信息和价值估计。
    # 输入：obs 表示 actor 当前可见的普通观测；obs_history 表示一段历史观测，供带编码器的策略提取隐含状态；
    #      critic_obs 表示 critic 用来估计状态价值的观测，启用特权观测时会比 actor 看到的信息更多。
    # 输出：返回当前策略采样得到的动作；同时把动作、价值、动作概率和相关观测写入临时转移对象。
    def act(self, obs, obs_history, critic_obs):

        # 如果当前策略是循环网络，先保存 actor 和 critic 的隐藏状态，后续写入 rollout 缓冲区用于按轨迹训练。
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Compute the actions and values
        # 序列策略会从历史观测中编码出 latent 隐变量，actor 用它辅助生成动作，critic 也拼接它来估计价值。
        if self.actor_critic.is_sequence:
            self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
            latent = self.actor_critic.get_latent()
            critic_obs = torch.cat((critic_obs, latent), dim=-1)
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()

        # 用 critic 观测估计当前状态价值，并记录采样动作在旧策略分布下的概率信息，供之后计算 PPO 损失。
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        # 缓存执行动作前的观测；环境执行动作后会返回下一时刻观测，PPO 需要用动作前后的状态组成一条转移。
        self.transition.observations = obs.clone()
        self.transition.observation_history = obs_history.clone()
        self.transition.critic_observations = critic_obs.clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        if self.kl_decay != 0:
            self.desired_kl = max(self.desired_kl - self.kl_decay, 0.001)
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0

        # 这里只创建生成器，不执行函数体内的任何逻辑
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        # yield 让 mini_batch_generator(...) 变成一个可以被 for 循环逐批使用的数据生成器，
        # 每次产出一个 mini-batch，PPO 每拿到一个 mini-batch 就做一次损失计算和参数更新。
        #
        # 调用 mini_batch_generator(...) -> 只得到 generator 对象，函数体还没真正运行
        #
        # 进入 for (...) in generator
        # -> 第一次 next(generator)，next(generator) 的意思是，
        #    运行 generator 内部的 mini_batch_generator(...) 函数体，直到遇到 yield 语句为止
        # -> flatten 各种 rollout 数据
        # -> 进入 epoch 和 mini-batch 循环
        # -> 生成第 1 个 obs_batch/actions_batch/...
        # -> yield 第 1 个 batch，然后暂停
        #
        # for 循环体执行 PPO 更新
        # -> 用第 1 个 batch 算 loss、反向传播、更新参数
        #
        # for 进入下一轮
        # -> 第二次 next(generator)
        # -> 从上一次 yield 后面继续执行，直到遇到下一个 yield
        #
        # 如此重复

        for (
            obs_batch,
            obs_history_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,  # 仅循环网络使用，非循环网络时为 None
            masks_batch,       # 仅循环网络使用，非循环网络时为 None
        ) in generator:
            if self.actor_critic.is_sequence:
                self.actor_critic.act(
                    obs_batch,
                    obs_history_batch,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )
            else:
                self.actor_critic.act(
                    obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            num_updates += 1

        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for next_obs_batch, critic_obs_batch, obs_history_batch in generator:
                if self.actor_critic.is_sequence:
                    latent_batch = self.actor_critic.encode(obs_history_batch)
                    vel_est_loss = (
                        (latent_batch[:, :3] - critic_obs_batch[:, :3]).pow(2).mean()
                    )
                    if self.actor_critic.latent_dim > 3:
                        obs_denoise_loss = (
                            (
                                latent_batch[:, 3 : self.actor_critic.latent_dim]
                                - critic_obs_batch[:, 3 : self.actor_critic.latent_dim]
                            )
                            .pow(2)
                            .mean()
                        )
                        extra_loss = vel_est_loss + obs_denoise_loss
                    else:
                        extra_loss = vel_est_loss

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.1)
                self.extra_optimizer.step()

                mean_extra_loss += extra_loss.item()
                num_updates_extra += 1

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        self.storage.clear()

        return (mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss)
