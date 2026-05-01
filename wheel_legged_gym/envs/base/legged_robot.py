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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.base_task import BaseTask
from wheel_legged_gym.utils.terrain import Terrain
from wheel_legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from wheel_legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):
    def __init__(
        self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    # 作用：
    # 这个函数是轮腿机器人环境在运行阶段的单步推进入口。
    # 它负责接收当前策略给出的动作，经过裁剪和延迟处理后转换为关节力矩，
    # 再推动底层物理仿真前进若干个物理子步，最后统一完成奖励、终止、重置和观测更新。
    # 训练时，PPO 采样循环会反复调用这个函数收集轨迹；
    # 初始化时，BaseTask.reset() 也会用全零动作调用它一次，用来刷新初始观测。
    #
    # 输入：
    # actions：策略网络为所有并行环境输出的动作张量；
    #          每一行代表一个并行机器人当前时刻的控制指令，
    #          后续会被裁剪到允许范围内，并转换成当前步真正施加的执行器力矩。
    #
    # 输出：
    # 这个函数返回一个 6 元组，供训练器继续向前滚动环境：
    # self.obs_buf：当前策略真正看到的普通观测，也就是 actor 网络的主要输入。
    # self.privileged_obs_buf：给 critic 网络使用的特权观测；如果当前任务没有启用特权观测，这里可能是 None。
    # self.rew_buf：所有并行环境在这一步得到的奖励。
    # self.reset_buf：所有并行环境在这一步结束后是否需要重置的标记。
    # self.extras：额外统计信息，例如 episode 日志和超时标记。
    # self.obs_history：拼接后的历史观测序列，供带历史输入的策略使用。
    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 先把策略输出的动作限制在配置允许的范围内，并放到当前环境张量所在的设备上。
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 在真正推进物理仿真之前，先处理可视化窗口和一些与本步开始相关的预处理逻辑。
        # step physics and render each frame
        self.render()
        self.pre_physics_step()

        # 一个环境步对应多个底层仿真子步：同一个策略动作会持续 decimation 个底层仿真步，以较高频率推进物理世界。
        # 每个子步都真正调用一次 self.gym.simulate(self.sim)
        # 动作延迟本质上也是域随机化的一部分
        for _ in range(self.cfg.control.decimation):
            # 记录每个并行环境已经运行了多少个底层物理子步，
            # 后面一些按时间触发的逻辑会用这个计数器做判断。
            self.envs_steps_buf += 1

            # 维护动作历史队列 action_fifo：
            # 这个张量的三个维度分别表示“并行环境编号、历史动作在时间队列中的位置、动作向量的各个分量”。
            # 其中第 1 维的长度由最大动作延迟决定，保存最近若干个物理子步的动作，
            # 目的是为了模拟控制指令不能立刻生效的执行器延迟，而不是单纯为了记录日志。
            #
            # self.actions 的形状原本是 [num_envs, num_actions]，
            # unsqueeze(1) 会在中间插入一个长度为 1 的新维度，把它变成 [num_envs, 1, num_actions]，
            # 这样当前动作就能被看成“一帧新的历史动作”，与 action_fifo 在时间维上对齐。
            #
            # self.action_fifo[:, :-1, :] 表示保留旧队列中除最后一帧之外的所有历史动作；
            # torch.cat(..., dim=1) 表示沿着“历史时间维”拼接，
            # 因此拼接后的结果就是：
            # 最新动作放在最前面，原来的历史动作整体后移一格，最老的一帧被丢弃。
            # 后面再通过 action_delay_idx 从这个队列中选出“当前真正生效的那一帧动作”。
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )

            # 根据延迟后的动作计算本次真正施加到关节上的力矩。
            # 这里的 action_delay_idx 不是手写固定值，而是在初始化/重置阶段按 delay_ms_range 随机生成的，
            # 先把“毫秒级延迟范围”除以 sim_params.dt，也就是底层物理仿真单步时长，
            # 再四舍五入成整数索引，因此它表示“延迟了多少个底层物理步”，而不是延迟多少个环境步。
            #
            # 下面这段高级索引的含义是：
            # torch.arange(self.num_envs) 会生成 [0, 1, 2, ..., num_envs-1]，
            # 也就是“每个并行环境自己的编号”。
            # action_delay_idx 则给出“每个环境应该取历史队列中的第几帧动作”。
            # ':' 表示把这一帧动作的全部动作分量都取出来。
            #
            # 例如当 num_envs=3、action_delay_idx=[0, 2, 1] 时，
            # torch.arange(self.num_envs) 就相当于 [0, 1, 2]，
            # 整体索引就等价于：
            # action_fifo[[0, 1, 2], [0, 2, 1], :]
            # 它实际取出的分别是：
            # 第 0 个环境的第 0 帧历史动作、
            # 第 1 个环境的第 2 帧历史动作、
            # 第 2 个环境的第 1 帧历史动作。
            # 因此最终得到的是一个形状为 [num_envs, num_actions] 的张量，
            # 表示“每个并行环境当前真正生效的那一帧延迟动作”，再交给 _compute_torques(...) 转成关节力矩。
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)

            # 把算出的关节力矩写入 Isaac Gym，让下一次物理积分按这些执行器输入推进。
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )

            # 如果启用了外力扰动随机化，就按配置周期给机器人施加随机推力。
            if self.cfg.domain_rand.push_robots:
                self._push_robots()

            # 真正推进一次底层物理仿真。
            self.gym.simulate(self.sim)

            # 当仿真主要跑在 CPU 上时，需要显式等待并取回这一步的结果。
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)

            # 刷新关节状态张量，并基于最新位置估计速度，供后续奖励和观测计算使用。
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()

        # 物理子步全部完成后，统一刷新机器人参考系状态(root_state)、检查终止、计算奖励、重置需要结束的环境，并生成新观测。
        self.post_physics_step()

        # 在返回给训练器之前，对普通观测和特权观测做数值裁剪，
        # 避免异常大数值破坏策略网络和价值网络的稳定性。
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
        )

    # 作用：
    # 这个函数在每个底层物理仿真步刷新关节状态后，根据当前关节角度和上一物理步缓存的关节角度估算关节速度。
    # 它额外处理旋转关节跨越 -pi/pi 边界时的角度跳变，避免速度估计因为角度表示方式突然变化而出现异常大值。
    #
    # 输入：
    # 无显式参数；它读取 self.dof_pos 表示的当前关节角度、
    # self.last_dof_pos 表示的上一物理步关节角度缓存，以及 self.sim_params.dt 表示的底层物理仿真步长。
    #
    # 输出：
    # 无显示返回值；它会更新由 self.dof_pos_dot 表示的关节速度估计。
    # 如果参数 self.cfg.env.dof_vel_use_pos_diff 为 True，表示使用差分速度替代 Isaac Gym 状态张量中直接提供的关节速度。
    # 最后它会刷新 self.last_dof_pos 表示的上一物理步关节角度缓存，供下一次差分使用。
    def compute_dof_vel(self):
        # 计算相邻两个物理步之间的最短角度差。
        # 这里把角度差包回 [-pi, pi) 范围，是为了让旋转关节在跨越圆周边界时仍得到连续的小角度变化。
        diff = (
            torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi)
            - self.pi
        )
        self.dof_pos_dot = diff / self.sim_params.dt

        # 根据配置选择是否用“位置差除以物理步长”得到的速度估计，
        # 替代 Isaac Gym 状态张量中直接提供的关节速度。
        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        # 保存本物理步的关节角度，作为下一次速度差分的历史参考。
        self.last_dof_pos[:] = self.dof_pos[:]

    # 作用：
    # 这个函数负责在action被施加并持续指定的仿真物理子步后，完成“仿真后处理”。
    # 它会先从仿真器刷新
    # - 机器人参考系状态(root_state)、
    # - 每个刚体 link 的净接触力(净接触力：把某个刚体在上一仿真步里由于所有接触产生的力合并后，得到一个总的 3D 力向量)
    # - 整台机器人所有刚体 link 的状态，
    # 再整理奖励与观测计算需要的派生物理量，随后执行命令更新、终止判定、奖励累计、环境重置和新观测生成。
    #
    # 输入：
    # 这个函数没有显式参数；它直接读取当前环境对象里已经由前面物理子步更新好的状态张量，
    # 例如根部位姿、关节状态、接触力、上一时刻缓存值以及当前动作。
    #
    # 输出：
    # 这个函数没有显式返回值；它通过修改当前环境对象的内部状态产生效果。
    # 具体会更新奖励缓冲区、重置标记、普通观测、特权观测、历史观测以及若干“上一时刻”缓存，
    # 这些结果会在 step() 返回时被训练器直接读取。
    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        # 先从 Isaac Gym 刷新本步结束后的关键状态张量：
        # 包括机器人参考系状态(root_state)、所有刚体净接触力和整台机器人所有刚体 link 的状态，
        # 后面的终止判定、奖励计算和观测构造都会依赖这些最新结果。
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 维护回合级和全局级的步数计数器。
        # episode_length_buf 表示每个并行环境当前回合已经运行了多少步；
        # common_step_counter 表示整个环境对象从启动以来累计执行了多少个环境步。
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 根据刚刚刷新的机器人参考系状态(root_state)和关节状态，整理后续频繁使用的派生物理量：
        # 例如机体姿态四元数、机体坐标系下的线速度与角速度、重力在机体系下的投影，以及关节加速度。
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

        # 用两条腿的关节角计算虚拟腿几何量：
        # theta1 和 theta2 表示两级连杆的关节角组合，
        # L0 表示虚拟腿长度，theta0 表示虚拟腿相对机体的摆角。
        # 这些量会参与奖励设计，也会被 VMC 版本作为更核心的状态表达。
        theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        theta2 = torch.cat(
            (
                (self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (-self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )
        end_x = (
            self.cfg.asset.offset
            + self.cfg.asset.l1 * torch.cos(theta1)
            + self.cfg.asset.l2 * torch.cos(theta1 + theta2)
        )
        end_y = self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            theta1 + theta2
        )
        self.L0 = torch.sqrt(end_x**2 + end_y**2)
        self.theta0 = torch.arctan2(end_y, end_x) - self.pi / 2

        # 执行“物理步之后但在奖励/观测之前”的通用回调：
        # 这里会处理命令重采样、朝向命令换算、地形高度测量等共享逻辑。
        self._post_physics_step_callback()

        # 依次完成强化学习环境真正关心的几件事：
        # 1. 检查哪些并行环境需要结束；
        # 2. 计算本步奖励；
        # 3. 对需要结束的环境执行局部重置；
        # 4. 生成下一步策略要读取的新观测。
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 把本步结果保存成“上一时刻缓存”，供下一步计算速度、加速度、动作变化率和历史观测时使用。
        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # 如果启用了 viewer 可视化窗口、同步渲染以及调试显示，
        # 就在这里把辅助调试图形画出来，例如地形采样点等可视化信息。
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """Check if environments need to be reset"""
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fail_buf |= self.projected_gravity[:, 2] > -0.1
        self.fail_buf *= fail_buf
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.edge_reset_buf
        )

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
            if self.cfg.commands.curriculum:
                time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
                self.update_command_curriculum(time_out_env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.obs_history[env_ids] = 0
        obs_buf = self.compute_proprioception_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["a_flat_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.flat_idx, 1].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["a_smooth_slope_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
            self.extras["episode"]["a_rough_slope_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.rough_slope_idx, 1].float()
            )
            self.extras["episode"]["a_stair_up_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.stair_up_idx, 1].float()
            )
            self.extras["episode"]["a_stair_down_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.stair_down_idx, 1].float()
            )
            self.extras["episode"]["a_discrete_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.discrete_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # 作用：
    # 这个函数负责在每个环境控制步（不是底层仿真步）结束时，汇总当前所有已启用的奖励项，
    # 累计各奖励项在当前控制步内的具体奖励值，生成每一个并行环境当前控制步内的总奖励
    # 通过 self.rew_buf 把奖励返回给强化学习训练器，用于 PPO 采样与更新。
    #
    # 输入：
    # 无显式参数；函数主要读取当前环境对象里的几类运行时状态：
    # self.reward_functions 表示已经根据配置筛选并注册好的奖励函数列表，其中每个元素都对应一个具体的奖励公式；
    # self.reward_scales 表示每个奖励项的权重，数值已经在初始化阶段按一个环境步对应的真实时长做过缩放；
    # self.reward_names 表示奖励函数列表中每一项对应的奖励名字；
    # 另外还会读取 self.cfg.rewards 里的裁剪和总奖励后处理配置。
    #
    # 输出：
    # 这个函数没有显式返回值；它通过修改当前环境对象的内部状态产生效果。
    # self.rew_buf 会被写成“每个并行环境在当前这一步的总奖励”；
    # self.episode_sums 会继续累计“每个并行环境在当前 episode 内各奖励项的累计值”，
    # 供回合结束时记录日志和课程学习逻辑使用。
    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        # 先清空当前控制步的总奖励缓存，准备重新累加本步所有已启用的奖励项。
        self.rew_buf[:] = 0.0
        # 按初始化阶段整理好的顺序，逐项计算奖励：
        # 先调用具体奖励公式得到当前状态下的原始奖励值，
        # 再乘上该奖励项的权重，并限制单项奖励在当前一步内的最大影响范围，
        # 最后同时写入“本步总奖励”和“当前回合分项累计奖励”。
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # reward_function 列表内存储的是奖励函数名称，取出后通过函数调用符 () 调用它。
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = torch.clip(
                rew,
                -self.cfg.rewards.clip_single_reward * self.dt,
                self.cfg.rewards.clip_single_reward * self.dt,
            )
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # 如果配置要求总奖励不能为负，就在所有普通奖励项累加完成后，
        # 统一把每个并行环境当前这一步的总奖励下限截到 0。
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        # 终止奖励单独放在最后处理：
        # 它不会参与前面的普通奖励逐项裁剪流程，而是在总奖励基本形成后，
        # 根据“是否真正失败终止而不是单纯超时”补上额外奖惩。
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_proprioception_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        return obs_buf

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = self.compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.obs_buf,
                    self.last_actions[:, :, 0],
                    self.last_actions[:, :, 1],
                    self.dof_acc * self.obs_scales.dof_acc,
                    heights,
                    self.torques * self.obs_scales.torque,
                    (self.base_mass - self.base_mass.mean()).view(self.num_envs, 1),
                    self.base_com,
                    self.default_dof_pos - self.raw_default_dof_pos,
                    self.friction_coef.view(self.num_envs, 1),
                    self.restitution_coef.view(self.num_envs, 1),
                ),
                dim=-1,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        update_idx = (
            (self.envs_steps_buf / self.cfg.control.decimation)
            % self.cfg.env.obs_history_dec
        ) == 0
        self.obs_history[update_idx, :] = torch.cat(
            (self.obs_history[update_idx, self.num_obs :], self.obs_buf[update_idx, :]),
            dim=-1,
        )

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (num_buckets, 1),
                    device=self.device,
                )
                self.friction_coef = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coef[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                (
                    min_restitution,
                    max_restitution,
                ) = self.cfg.domain_rand.restitution_range
                self.restitution_coef = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_restitution - min_restitution)
                    + min_restitution
                )
            for s in range(len(props)):
                props[s].restitution = self.restitution_coef[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                self.base_add_mass = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_add_mass - min_add_mass)
                    + min_add_mass
                )
                self.base_mass = props[0].mass + self.base_add_mass
            props[0].mass += self.base_add_mass[env_id]
        else:
            self.base_mass[:] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                self.base_com[:, 0] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_x * 2)
                    - com_x
                )
                self.base_com[:, 1] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_y * 2)
                    - com_y
                )
                self.base_com[:, 2] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_z * 2)
                    - com_z
                )
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 1] = torch.clip(
                1.5 * wrap_to_pi(self.commands[:, 3] - heading), -5, 5
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["height"][env_ids, 1]
            - self.command_ranges["height"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "height"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        pos_ref = actions * self.cfg.control.pos_action_scale
        pos_ref[:, 2] *= 0
        pos_ref[:, 5] *= 0
        vel_ref = actions * self.cfg.control.vel_action_scale
        vel_ref[:, :2] *= 0
        vel_ref[:, 3:5] *= 0
        torques = self.p_gains * (
            pos_ref + self.default_dof_pos - self.dof_pos
        ) + self.d_gains * (vel_ref - self.dof_vel)
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids, :]
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        """Random pushes the robots."""
        env_ids = (
            (
                self.envs_steps_buf
                % int(self.cfg.domain_rand.push_interval_s / self.sim_params.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return

        max_push_force = (
            self.base_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[env_ids, 0, 0:3] = quat_rotate(
            self.base_quat[env_ids], rigid_body_external_forces[env_ids]
        )
        self.rigid_body_external_forces[env_ids, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s
            < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.4
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        mask = self.terrain_levels[env_ids] >= self.max_terrain_level
        self.success_ids = env_ids[mask]
        mask = self.terrain_levels[env_ids] < 0
        self.fail_ids = env_ids[mask]
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]
        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"][self.fail_ids, 0] = torch.clip(
                self.command_ranges["lin_vel_x"][self.fail_ids, 0] + 0.25,
                -self.cfg.commands.basic_max_curriculum,
                -1,
            )
            self.command_ranges["lin_vel_x"][self.fail_ids, 1] = torch.clip(
                self.command_ranges["lin_vel_x"][self.fail_ids, 1] - 0.25,
                1,
                self.cfg.commands.basic_max_curriculum,
            )

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if self.cfg.terrain.curriculum and len(self.success_ids) != 0:
            # self.basic_terrain_idx = torch.cat((self.stair_up_idx, self.discrete_idx))
            # self.advanced_terrain_idx
            mask = (
                self.episode_sums["tracking_lin_vel"][self.success_ids]
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_lin_vel"]
            )
            success_ids = self.success_ids[mask]
            basic_ids = torch.any(
                success_ids.unsqueeze(1) == self.basic_terrain_idx.unsqueeze(0), dim=1
            )
            basic_ids = success_ids[basic_ids]
            self.command_ranges["lin_vel_x"][success_ids, 0] -= 0.05
            self.command_ranges["lin_vel_x"][success_ids, 1] += 0.05
            self.command_ranges["lin_vel_x"][basic_ids, 0] -= 0.45
            self.command_ranges["lin_vel_x"][basic_ids, 1] += 0.45

            self.command_ranges["lin_vel_x"][self.basic_terrain_idx, :] = torch.clip(
                self.command_ranges["lin_vel_x"][self.basic_terrain_idx, :],
                -self.cfg.commands.basic_max_curriculum,
                self.cfg.commands.basic_max_curriculum,
            )
            self.command_ranges["lin_vel_x"][self.advanced_terrain_idx, :] = torch.clip(
                self.command_ranges["lin_vel_x"][self.advanced_terrain_idx, :],
                -self.cfg.commands.advanced_max_curriculum,
                self.cfg.commands.advanced_max_curriculum,
            )
        if self.cfg.terrain.curriculum == False:
            if (
                torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_lin_vel"]
                and torch.mean(self.episode_sums["tracking_ang_vel"][env_ids])
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_ang_vel"]
                * 0.8
            ):
                self.command_ranges["lin_vel_x"][:, 0] = torch.clip(
                    self.command_ranges["lin_vel_x"][:, 0] - 0.1,
                    -self.cfg.commands.basic_max_curriculum,
                    0.0,
                )
                self.command_ranges["lin_vel_x"][:, 1] = torch.clip(
                    self.command_ranges["lin_vel_x"][:, 1] + 0.1,
                    0.0,
                    self.cfg.commands.basic_max_curriculum,
                )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # noise_vec[3 : 3 + 3] = (
        #     noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # )
        # noise_vec[3 + 3 : 6 + 3] = noise_scales.gravity * noise_level
        # noise_vec[6 + 3 : 8 + 3] = 0.0  # commands
        # noise_vec[8 + 3 : 14 + 3] = (
        #     noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # )
        # noise_vec[14 + 3 : 20 + 3] = (
        #     noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # )
        # noise_vec[20 + 3 : 26 + 3] = 0.0  # previous actions
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:8] = 0.0  # commands
        noise_vec[8:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[14:20] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[20:26] = 0.0  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    # ----------------------------------------
    # 作用：
    # 这个函数负责把 Isaac Gym 底层仿真返回的状态张量，整理成当前环境对象后续训练会持续复用的成员变量。
    # 上面一层的 step()、reset_idx()、compute_observations()、compute_reward() 以及 PPO 采样流程，
    # 都依赖这里准备好的状态视图、缓存张量、控制参数和随机化结果。
    #
    # 输入：
    # 这个函数没有显式参数；它读取的输入主要来自当前环境对象已经准备好的上下文，
    # 包括 self.sim 表示的仿真实例、self.cfg 表示的环境与控制配置、
    # self.num_envs 和 self.num_dof 表示的并行环境规模与关节规模，
    # 以及 create_sim() 阶段已经创建好的机器人、地形和刚体信息。
    #
    # 输出：
    # 这个函数没有显式返回值；它的结果是把当前对象初始化成“可以正式进入训练循环”的状态。
    # 执行完成后，环境对象会持有：
    # 1. 与 Isaac Gym 状态张量共享内存的视图，供每一步读取机器人姿态、关节和接触信息；
    # 2. 观测、控制、命令、历史动作和调试统计所需的变量；
    # 3. 由配置和 domain randomization 决定的 PD 参数、默认关节位置和动作延迟设置。
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # 从 Isaac Gym 取出机器人参考系状态(root_state)、关节状态和接触力这三类使用 Isaac Gym 内置类型表示的数据，
        # 并立即刷新一次，确保后面包装出的张量视图看到的是当前最新仿真状态。
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 把 Isaac Gym 内置类型表示的数据包装成 PyTorch 视图，并切出后续高频使用的字段：
        # 例如机器人参考系状态(root_state)、关节位置与速度、机身姿态四元数、以及每个刚体的接触力。
        # 这一步决定了后续环境逻辑可以直接用 torch 方式处理 Isaac Gym 的实时状态。
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # 初始化环境级共享缓存：
        # 这里包括噪声缩放、重力方向、机体前向方向，以及策略输出力矩、PD 增益、
        # 当前动作和历史动作等高频中间量。它们共同支撑 step() 里的控制计算与观测构造。
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        # 初始化命令相关状态。
        # 这些张量描述策略给出的机器人跟踪目标，例如前进速度、偏航角速度和目标高度；
        # curriculum 与命令重采样逻辑会直接读写这里的数据。
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands + 1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
            ],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.command_ranges["lin_vel_x"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_x"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x
        )
        self.command_ranges["ang_vel_yaw"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(
            self.cfg.commands.ranges.ang_vel_yaw
        )
        self.command_ranges["height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["height"][:] = torch.tensor(self.cfg.commands.ranges.height)
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )

        # 为外力扰动、重力投影和动作延迟机制准备缓存。
        # 其中动作延迟队列会在 step() 中维护，用来模拟控制命令并不是立即到达执行器的现实情况；
        # 这是整个项目 domain randomization 的一部分。
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # 如果任务启用了地形高度测量，就预先生成机体周围的采样点。
        # 后续 compute_observations() 会用这些采样点构造高度相关观测，
        # 让策略感知粗糙地形的局部形状。
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        # 预留腿部几何中间量。
        # 这些量在普通 LeggedRobot 中用于根据关节角恢复腿长和腿摆角，
        # 在 VMC 子类里则会进一步扩展成更完整的虚拟腿状态。
        self.L0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        # 根据配置文件为每个关节建立默认位置和 PD 增益。
        # 将配置中的机器人定义翻译成控制器可直接使用的数值张量，
        # 后面的 _compute_torques() 会直接读取这些参数生成控制力矩。
        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # 这里按“当前仿真内部的自由度索引顺序”逐个处理关节。
        # 配置文件中的 default_joint_angles、stiffness 和 damping 是“按关节名字组织”的，
        # 但控制器在运行时需要的是“按索引排好、可直接参与张量运算”的参数。
        # 所以下面这段循环本质上是在做一次从“名字配置”到“批量数值张量”的映射。
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]

            # raw_default_dof_pos 保存“原始配置里定义的默认关节位置”，只保留一份机器人本体参数；
            # default_dof_pos 形状为 (num_envs, num_dof) ，在这里我们把同一个默认值复制到所有并行环境上，
            # 方便后续控制器一次性对全部环境进行张量计算。
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False

            # 这里不是直接用关节索引查增益，而是用当前关节名字去匹配配置里的 stiffness 和 damping。
            # 匹配成功后，就把这个关节对应的 PD 参数写入 p_gains 和 d_gains，
            # 让后面的 _compute_torques() 可以直接按“环境批次 x 关节索引”读取控制增益。
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True

            # 如果配置里没有为当前关节找到 PD 参数，就显式置零。
            # 这样做的含义不是“这个关节不存在”，而是“这个关节在当前位置控制或速度控制模式下不施加对应的 PD 控制增益”。
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )

        # 最后应用 domain randomization。
        # 这里会随机扰动控制器增益、执行器力矩缩放、默认关节零位和动作延迟，
        # 目的是让训练出来的策略对真实硬件中的参数偏差和控制链路延迟更鲁棒。
        if self.cfg.domain_rand.randomize_Kp:
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.p_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_Kd:
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_motor_torque:
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_action_delay:
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()

    # 作用：
    # 这个函数负责在环境初始化阶段，把配置文件里声明的奖励项权重表整理成运行时可直接调用的奖励函数列表。
    # 后面的 compute_reward() 不再每一步临时查配置和拼函数名，而是直接遍历这里准备好的函数列表。
    #
    # 输入：
    # 这个函数没有显式参数；它主要读取 self.reward_scales 这个奖励权重字典。
    # 这个字典通常来自配置文件中的 rewards.scales，并且键名对应各个奖励项的名字，
    # 例如 tracking_lin_vel、orientation、torques 等。
    #
    # 输出：
    # 这个函数没有显式返回值；它会修改和创建三类运行时状态：
    # 1. self.reward_scales：移除权重为零的奖励项，并把保留下来的权重按仿真步长 self.dt 做时间尺度缩放；
    # 2. self.reward_functions 和 self.reward_names：建立“奖励名字 -> 具体成员函数”的调用顺序表，供 compute_reward() 逐项累加；
    # 3. self.episode_sums：为每个并行环境记录各奖励项在一个 episode 内的累计值，供 reset_idx() 统计日志。
    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 先清理奖励权重表：
        # 权重为 0 的奖励项说明当前任务不启用，可以直接移除，避免后面每一步做无意义计算；
        # 保留下来的非零奖励项会乘以仿真步长 self.dt，将单位时间的奖励换算成当前控制步持续时长下的奖励
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # 根据奖励名字动态收集对应的成员函数。
        # 例如配置里有 tracking_lin_vel，就去查找当前对象上的 _reward_tracking_lin_vel 方法。
        # 这样 compute_reward() 后续只需要按固定顺序遍历 reward_functions 即可。
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # termination 奖励项单独处理：
            # 它不进入普通奖励循环，而是在 compute_reward() 的末尾单独计算并叠加。
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            # getattr(self, name) 的意思是：用字符串形式的属性名 name，
            # 去当前对象 self 上动态查找同名属性。
            # 这里查找到是对应的成员方法对象，
            # 例如当 name 是 "_reward_torques" 时，取到的是 self._reward_torques 这个方法。
            # append(...) 加入列表的也是这个方法对象，因此后面可以通过 self.reward_functions[i]() 直接调用。
            self.reward_functions.append(getattr(self, name))

        # 为每个奖励项准备 episode 级累计缓存。
        # 这里的结构是“奖励项名字 -> 每个并行环境各自的累计奖励”，
        # 后面 compute_reward() 会持续往里累加，reset_idx() 会在回合结束时读取并清零，用于日志统计。
        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    # URDF 文件在该函数中被加载到仿真中
    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 根据配置里的关键字，从 URDF/资产刚体名称中筛出需要计入碰撞惩罚的身体部位。
        # 这里保存的是刚体名称，后面会转换成 Isaac Gym 的刚体索引。
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.friction_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitution_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_com = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            # num_cols = 20
            # terrain types: [flat, smooth slope, rough slope, stairs up, stairs down, discrete]
            # terrain types: [0 1 2 3, 4 5 6 7, 8 9 10 11, 12 13, 14 15 16 17, 18 19]
            # terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.2, 0.1]
            self.flat_idx = (self.terrain_types < 4).nonzero(as_tuple=False).flatten()
            self.smooth_slope_idx = (
                ((4 <= self.terrain_types) * (self.terrain_types < 8))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.rough_slope_idx = (
                ((8 <= self.terrain_types) * (self.terrain_types < 12))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.stair_up_idx = (
                ((12 <= self.terrain_types) * (self.terrain_types < 14))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.stair_down_idx = (
                ((14 <= self.terrain_types) * (self.terrain_types < 18))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.discrete_idx = (
                ((18 <= self.terrain_types) * (self.terrain_types < 20))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.basic_terrain_idx = torch.cat(
                (
                    self.flat_idx,
                    self.smooth_slope_idx,
                    self.rough_slope_idx,
                    self.stair_down_idx,
                )
            )
            self.advanced_terrain_idx = torch.cat(
                (self.stair_up_idx, self.discrete_idx)
            )
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
            self.terrain_x_max = (
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_x_min = -self.cfg.terrain.border_size
            self.terrain_y_max = (
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_y_min = -self.cfg.terrain.border_size
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
            self.flat_idx = torch.arange(self.num_envs, device=self.device)

    # 作用：
    # 这个函数负责把环境配置对象里的原始超参数，转换成当前环境实例在后续运行时直接使用的内部字段。
    # 它主要完成三类准备工作：统一时间尺度、提取观测与奖励相关配置、以及把若干“以秒为单位”的配置换算成“以控制步为单位”的值。
    # 这些结果会被 step()、奖励计算、命令采样、课程学习和终止判定等流程反复使用。
    #
    # 输入：
    # cfg：环境配置对象，按设计这里代表当前任务的配置来源；
    #      虽然函数体里实际主要读取 self.cfg，也就是当前环境实例保存的完整配置，
    #      但这个参数表达了这个函数的职责是“解析当前任务配置”。
    #
    # 输出：
    # 这个函数没有显式返回值；它通过修改当前环境对象的内部状态产生效果。
    # 具体会写入 dt，也就是一次策略控制对应的真实时间间隔；
    # 写入 obs_scales，也就是各类观测量在送入策略前使用的缩放系数；
    # 写入 reward_scales，也就是各奖励项的权重表；
    # 写入 command_ranges，也就是速度、高度等指令的采样范围；
    # 还会写入 max_episode_length 和 push_interval 这类按“控制步数”表示的运行参数。
    def _parse_cfg(self, cfg):
        # 这里的 self.dt 表示“强化学习环境视角下一步控制”对应的真实时间长度，
        # 不是底层物理引擎单次积分的最小时间步。
        # self.sim_params.dt 表示物理仿真一步前进多少秒；
        # self.cfg.control.decimation 表示同一个策略动作会连续作用多少个物理仿真步。
        # 因此二者相乘后，得到一次 env.step(...) 实际覆盖的时间：
        # 例如物理步长是 0.005 秒、同一动作持续 2 个物理步，
        # 那么环境控制步的时间 self.dt 就是 0.01 秒。
        self.dt = self.cfg.control.decimation * self.sim_params.dt

        # 取出观测缩放系数、奖励权重和指令范围配置，供后续观测拼接、奖励计算和命令采样直接使用。
        # reward_scales 和 command_ranges 额外转成普通字典，是为了后面按名字动态访问和修改更方便。
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        # 课程学习里的地形难度推进只在高度场或三角网格地形上有意义。
        # 如果当前不是这两类复杂地形，就直接关闭 terrain.curriculum，也就是地形课程学习开关。
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False

        # 先保留“以秒为单位”的单回合最大时长，
        # 再结合上面算出的环境步长 dt，换算出“一个 episode 最多能跑多少个控制步”。
        # 后面的超时终止、日志统计和训练器初始化都会依赖这个步数版本的上限。
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # 把“每隔多少秒推一次机器人”的随机扰动配置，
        # 换算成“每隔多少个环境控制步执行一次”的内部表示，方便 step 循环直接取模判断。
        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(
                    self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]
                )
                .cpu()
                .numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        # base_lin_vel[:, 0/1/2]： base_link 坐标系下 x/y/z 方向线速度
        # 惩罚机身竖直方向速度，抑制上下弹跳。
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        # base_ang_vel[:, 0/1/2]： base_link 坐标系下 roll/pitch/yaw 方向角速度
        # 惩罚 roll/pitch 角速度，让机身别大幅前后/左右甩动。
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # projected_gravity[:, 0/1/2]： 重力在 base_link 坐标系下 x/y/z 方向的投影分量
        # 重力投影在机体系 x/y 上越大，说明机身越倾斜；这个项约束机身保持水平姿态。
        # 如果机器人完全站正、机身不倾斜，那么重力在机体系里大致会接近 [0, 0, -9.81]
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # print(self.commands[0, 2], self.base_height[0])
        if self.reward_scales["base_height"] < 0:
            return torch.abs(self.base_height - self.commands[:, 2])
        else:
            # 鼓励机身高度贴近命令高度。误差越小越接近 1。
            base_height_error = torch.square(self.base_height - self.commands[:, 2])
            return torch.exp(-base_height_error / 0.001)

    def _reward_base_height_enhance(self):
        # 这是“机身高度主奖励”的辅助塑形项，而不是一个更强的主奖励。
        # 这里的 enhance 更适合在中文里理解成“增强训练信号的辅助项”：
        # 当机器人离目标高度还比较远时，它能提供一个衰减更慢、更平滑的修正信号，
        # 帮助策略先学会把高度往正确方向调回来；真正负责精确贴近目标高度的，仍是上面的 _reward_base_height(...)。

        # base_height_error 表示“当前机身实际高度”和“当前命令给出的目标高度”之间的平方误差。
        # 误差越小，说明机器人当前高度越接近想要的高度；误差越大，说明偏差越明显。
        base_height_error = torch.square(self.base_height - self.commands[:, 2])

        # 这里使用指数衰减来构造一个平滑的辅助惩罚项：
        # 1. 分母比主高度奖励更大，因此误差变大时衰减得更慢，能在较大误差范围内保留训练信号；
        # 2. 末尾减去 1 之后，返回值范围从原来的 (0, 1] 变成 (-1, 0]：
        #    - 当高度完全贴近目标时，返回 0，表示这项辅助惩罚可以自然退出；
        #    - 当高度偏差较大时，返回接近 -1，表示持续给出负反馈。
        # 所以它的角色更像“平滑的高度误差辅助惩罚”，而不是单独的正向奖励。
        #
        # 当前代码库里这个奖励项默认没有在 rewards.scales 中启用；
        # 通常只有在“主高度奖励太尖锐，训练前期高度误差较大、学习信号不够稳定”时，
        # 才会考虑把它和 _reward_base_height(...) 一起启用，并且一般只给较小权重。
        return torch.exp(-base_height_error / 0.001 / 10) - 1

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_power(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :2]), dim=1) + torch.sum(
            torch.square(self.dof_vel[:, 3:5]), dim=1
        )

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :, 0] - self.actions), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions[:, :2]
                - 2 * self.last_actions[:, :2, 0]
                + self.last_actions[:, :2, 1]
            ),
            dim=1,
        ) + torch.sum(
            torch.square(
                self.actions[:, 3:5]
                - 2 * self.last_actions[:, 3:5, 0]
                + self.last_actions[:, 3:5, 1]
            ),
            dim=1,
        )

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, :2] - self.dof_pos_limits[:2, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, :2] - self.dof_pos_limits[:2, 1]).clip(
            min=0.0
        )
        out_of_limits += -(self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 1]).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_enhance(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_enhance(self):
        # Tracking of angular velocity commands (x axes)
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev
        )
        # return lin_vel_error
        return delta_phi

    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev
        )
        # return ang_vel_error
        return delta_phi

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_nominal_state(self):
        # return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        if self.reward_scales["nominal_state"] < 0:
            return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        else:
            ang_diff = torch.square(self.theta0[:, 0] - self.theta0[:, 1])
            return torch.exp(-ang_diff / 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )
