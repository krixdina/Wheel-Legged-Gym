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
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.terrain import Terrain
from wheel_legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from wheel_legged_gym.utils.helpers import class_to_dict
from .wheel_legged_vmc_config import WheelLeggedVMCCfg


class LeggedRobotVMC(LeggedRobot):
    def __init__(
        self, cfg: WheelLeggedVMCCfg, sim_params, physics_engine, sim_device, headless
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
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            # 先根据当前仿真返回的关节位置和速度，计算虚拟腿的实际状态：
            # self.L0 / self.theta0 是当前真实腿长和等效腿摆角，
            # self.L0_dot / self.theta0_dot 是对应变化率。
            # 它们是 VMC 控制器的状态量；策略 actions 中的腿长和摆角维度则是目标参考量。
            # 后续 _compute_torques(...) 会把动作目标转换成 l0_ref / theta0_ref，
            # 再用“目标 - 当前实际状态”的误差生成虚拟腿力和力矩。
            self.leg_post_physics_step()
            self.envs_steps_buf += 1
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

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

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

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


        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def leg_post_physics_step(self):
        # Isaac Gym 返回的 self.dof_pos 是 URDF revolute joint 的原始关节角：
        # 正方向由 URDF 中每个 joint frame 的旋转轴 <axis> 按右手定则确定。
        # 左右腿在 URDF 里是镜像安装的，右腿 joint frame 与左腿方向相反，
        # 所以左右腿同一个几何姿态通常表现为左腿关节角为正、右腿关节角为负。
        # 这里将右腿角度取负，是为了把左右腿统一到同一个平面二连杆几何坐标系中。
        self.theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        # theta2 是二连杆模型中的第二段连杆相对水平方向(在本例中为x方向)的夹角。
        # 而在 URDF 中，第二段连杆与水平方向在定义时就相差 90°，
        # 因此我们需要将第二段连杆关节旋转角加上 pi/2 来转换成第二段连杆与水平方向的夹角。
        # 因此先用负号统一右腿镜像方向，再统一加 pi/2 把 URDF 关节角转换成几何夹角。
        self.theta2 = torch.cat(
            (
                (self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (-self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )

        # 速度项必须使用与 theta1/theta2 完全一致的符号约定；
        # pi/2 是常量偏置，对速度求导后为 0，因此这里只需要对右腿速度取负。
        theta1_dot = torch.cat(
            (self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta2_dot = torch.cat(
            (self.dof_vel[:, 1].unsqueeze(1), -self.dof_vel[:, 4].unsqueeze(1)), dim=1
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )

        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt

    # 作用：
    # 这个函数根据两级连杆的关节角，计算每条等效腿末端相对髋部参考点的极坐标状态，
    # 也就是“当前腿长有多长”以及“整条腿相对机体竖直方向偏转了多少角度”。
    # 在当前项目里，这一步是虚拟模型控制的几何基础；上层会先用它恢复等效腿长度和摆角，
    # 再进一步估计对应的变化速度，用于后续的腿长控制和摆角控制。
    #
    # 输入：
    # theta1：第一段连杆的关节角，表示从髋部出发的上游连杆当前转到了什么位置。
    # theta2：第二段连杆相对第一段连杆的关节角，表示下游连杆相对上游连杆继续弯折的程度。
    #
    # 输出：
    # L0：等效腿长度，表示从髋部参考点到腿部末端接触点的直线距离。
    # theta0：等效腿摆角，表示这条“髋部到足端”的等效连线相对机体竖直方向的偏转角。
    def forward_kinematics(self, theta1, theta2):
        # 先把两段刚性连杆在平面内展开，求出腿部末端相对髋部参考点的二维坐标。
        # 其中 offset 表示髋部参考点到第一段连杆安装点的固定水平偏置，
        # l1 和 l2 分别表示两段连杆的长度。
        end_x = (
            self.cfg.asset.offset
            + self.cfg.asset.l1 * torch.cos(theta1)
            + self.cfg.asset.l2 * torch.cos(theta1 + theta2)
        )
        end_y = self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            theta1 + theta2
        )
        # 再把二维直角坐标转换成极坐标：
        # 直线距离对应等效腿长，方向角再减去 pi/2 后，转换成相对机体竖直方向的摆角。
        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.arctan2(end_y, end_x) - self.pi / 2
        return L0, theta0

    # 相对父类 LeggedRobot.reset_idx(...) 的变化：
    # 这个函数的主体流程基本沿用父类实现，仍然负责课程难度更新、机器人状态重置、命令重采样、
    # episode 缓冲区清零以及训练日志统计。VMC 子类没有在这里新增专门的关节或机身重置逻辑。
    #
    # 真正的行为差异来自当前类重写了 compute_proprioception_observations(...)：
    # 父类初始化历史观测时使用的是原始关节位置误差和原始关节速度；
    # 当前类初始化 self.obs_history 这个供历史观测编码器使用的缓存时，使用的是虚拟腿摆角 theta0、
    # 虚拟腿摆角速度 theta0_dot、虚拟腿长 L0、虚拟腿长速度 L0_dot，以及左右轮关节状态。
    #
    # 注意：这个函数本身没有显式重新计算 theta0、theta0_dot、L0、L0_dot 这些虚拟腿状态；
    # 它们通常由每个物理子步前的 leg_post_physics_step(...) 根据最新关节状态更新。
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

    # 相对父类 LeggedRobot.compute_proprioception_observations(...) 的变化：
    # 父类普通观测会直接包含全部关节的位置误差 (self.dof_pos - self.default_dof_pos)
    # 和全部关节速度 self.dof_vel，也就是让策略网络直接看到 6 个自由度的原始关节状态。
    #
    # 当前 VMC 子类把腿部原始关节状态替换成虚拟腿状态：
    # self.theta0 表示左右虚拟腿相对机体竖直方向的摆角；
    # self.theta0_dot 表示左右虚拟腿摆角变化速度；
    # self.L0 表示左右虚拟腿长度；
    # self.L0_dot 表示左右虚拟腿长度变化速度。
    # 这样 actor 学到的是“虚拟腿层”的控制输入，而不是直接基于腿部关节角做端到端控制。
    #
    # 轮子仍然保留原始关节状态：self.dof_pos[:, [2, 5]] 表示左右轮关节位置，
    # self.dof_vel[:, [2, 5]] 表示左右轮关节速度，因为 VMC 只替换腿部几何控制，轮速仍直接参与底层控制。
    def compute_proprioception_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                # (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                # self.dof_vel * self.obs_scales.dof_vel,
                self.theta0 * self.obs_scales.dof_pos,
                self.theta0_dot * self.obs_scales.dof_vel,
                self.L0 * self.obs_scales.l0,
                self.L0_dot * self.obs_scales.l0_dot,
                self.dof_pos[:, [2, 5]] * self.obs_scales.dof_pos,
                self.dof_vel[:, [2, 5]] * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        return obs_buf

    # 相对父类 LeggedRobot.compute_observations(...) 的变化：
    # 第一处变化是普通观测 self.obs_buf 的内容已经由当前类的 compute_proprioception_observations(...)
    # 改成 VMC 观测结构，因此 actor 主要看到机体角速度、重力方向、运动命令、虚拟腿状态、轮关节状态和动作历史。
    #
    # 第二处变化是特权观测 self.privileged_obs_buf 额外保留了完整原始关节信息：
    # (self.dof_pos - self.default_dof_pos) 表示全部关节相对默认姿态的位置偏差，
    # self.dof_vel 表示全部关节速度。也就是说，actor 不直接看全部腿部原始关节状态，
    # 但 critic 仍可以在训练时使用这些更完整的物理状态来估计价值函数。
    #
    # 第三处变化是历史观测 self.obs_history 每次都会滑动追加当前普通观测；
    # 父类会根据 obs_history_dec 这个历史观测降采样参数决定是否更新，当前类没有使用这个降采样条件。
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
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
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

        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs :], self.obs_buf), dim=-1
        )

    # 作用：
    # 这个函数把策略网络输出的动作转换成 Isaac Gym 仿真器要施加到 6 个真实关节上的力矩命令。
    # 与父类直接用关节位置误差和关节速度误差做 PD 控制不同，当前 VMC 子类先把腿部动作解释为
    # 虚拟腿摆角目标和虚拟腿长度目标，再通过虚拟模型控制把等效腿上的力和力矩映射到真实腿部关节。
    #
    # 输入：
    # actions：策略网络在当前控制步输出的动作张量，每一行对应一个并行机器人；
    #          第 0/3 维表示左右虚拟腿摆角目标的归一化动作，第 1/4 维表示左右虚拟腿长度目标的归一化动作，
    #          第 2/5 维表示左右轮速度目标的归一化动作。
    #
    # 输出：
    # 返回值：一个形状与真实自由度数量一致的关节力矩张量；它会在 step(...) 中写入 Isaac Gym，
    #        作为下一次物理积分实际施加到机器人各关节上的执行器力矩。
    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 先把策略动作从无量纲范围转换成 VMC 控制器使用的物理目标：
        # 左右虚拟腿摆角目标用于控制腿相对机体竖直方向的偏转；
        # 左右虚拟腿长度目标用于控制髋部到轮/足端等效连线的长度；
        # 左右轮速度目标仍然直接用于轮关节速度控制。
        theta0_ref = (
            torch.cat(
                (
                    (actions[:, 0]).unsqueeze(1),
                    (actions[:, 3]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_theta
        )
        l0_ref = (
            torch.cat(
                (
                    (actions[:, 1]).unsqueeze(1),
                    (actions[:, 4]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_l0
        ) + self.cfg.control.l0_offset
        wheel_vel_ref = (
            torch.cat(
                (
                    (actions[:, 2]).unsqueeze(1),
                    (actions[:, 5]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )

        # 用虚拟腿层面的 PD 控制律生成等效控制量：
        # 摆角误差被转换成虚拟腿摆动方向的力矩，腿长误差被转换成沿虚拟腿方向的轴向力；
        # 轮关节不经过虚拟腿几何映射，仍然用轮速误差直接生成轮关节力矩。
        self.torque_leg = (
            self.theta_kp * (theta0_ref - self.theta0) - self.theta_kd * self.theta0_dot
        )
        self.force_leg = self.l0_kp * (l0_ref - self.L0) - self.l0_kd * self.L0_dot
        self.torque_wheel = self.d_gains[:, [2, 5]] * (
            wheel_vel_ref - self.dof_vel[:, [2, 5]]
        )
        T1, T2 = self.VMC(
            self.force_leg + self.cfg.control.feedforward_force, self.torque_leg
        )

        # 将虚拟模型控制得到的左右腿两段关节力矩和左右轮力矩拼回真实机器人 6 个自由度的顺序。
        # 右腿的两个腿部关节取负号，是为了把统一虚拟腿坐标系下的力矩方向转换回 URDF 中镜像安装的右腿关节方向。
        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                -T1[:, 1].unsqueeze(1),
                -T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        # 最后应用电机力矩随机缩放并限制到关节允许的力矩范围内，避免向仿真器发送超出执行器能力的命令。
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def VMC(self, F, T):
        theta0 = self.theta0 + self.pi / 2
        t11 = self.cfg.asset.l1 * torch.sin(
            theta0 - self.theta1
        ) - self.cfg.asset.l2 * torch.sin(self.theta1 + self.theta2 - theta0)

        t12 = self.cfg.asset.l1 * torch.cos(
            theta0 - self.theta1
        ) - self.cfg.asset.l2 * torch.cos(self.theta1 + self.theta2 - theta0)
        t12 = t12 / self.L0

        t21 = -self.cfg.asset.l2 * torch.sin(self.theta1 + self.theta2 - theta0)

        t22 = -self.cfg.asset.l2 * torch.cos(self.theta1 + self.theta2 - theta0)
        t22 = t22 / self.L0

        T1 = t11 * F - t12 * T
        T2 = t21 * F - t22 * T

        return T1, T2

    # 作用：
    # 这个函数为策略网络实际接收的普通观测构造逐维噪声幅值表，用于训练时给观测加入随机扰动，
    # 使策略对传感器误差和状态估计误差更鲁棒。
    #
    # 输入：
    # cfg：调用方传入的环境参数对象；当前实现没有直接读取这个参数，而是使用当前环境对象保存的参数，
    #      其中包含是否启用观测噪声、各类观测量的基础噪声强度以及观测归一化比例。
    #
    # 输出：
    # noise_vec：返回一个与单个环境普通观测同长度的张量；每个位置表示该观测维度
    #            中允许加入的最大噪声幅值(已进行缩放），后续会和 [-1, 1] 区间内的随机数相乘，
    #            再加到策略网络看到的普通观测上，因此第 i 维实际加噪范围是 [-noise_vec[i], +noise_vec[i]]。
    #
    #            例如原始机体角速度是 2.0 rad/s，观测缩放比例是 0.25，那么策略网络看到的是 0.5；
    #            如果原始角速度噪声上限是 0.2 rad/s，那么这里保存的缩放后噪声上限就是 0.2 * 0.25 = 0.05，
    #            最终会给 0.5 这个缩放后观测值加入 [-0.05, +0.05] 范围内的随机扰动。
    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # 先创建与单个环境普通观测同长度的噪声幅值表，并从环境配置中取出总开关、
        # 各类物理量的基础噪声强度和全局噪声倍率。
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
        # 下面的切片顺序必须和当前类构造普通观测的顺序保持一致：
        # 机体角速度、重力方向、运动命令、虚拟腿摆角和腿长、轮关节状态以及上一时刻策略动作。
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:8] = 0.0  # commands
        noise_vec[8:10] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[10:12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12:14] = noise_scales.l0 * noise_level * self.obs_scales.l0
        noise_vec[14:16] = noise_scales.l0_dot * noise_level * self.obs_scales.l0_dot
        noise_vec[16:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18:20] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[20:26] = 0.0  # previous actions
        # 如果当前地形配置启用了高度测量观测，则为局部地形高度采样部分也设置噪声幅值；
        # 这些高度采样值用于让策略感知机器人周围地形起伏。
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

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
        self.theta_kp = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.theta_kd = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l0_kp = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l0_kd = torch.zeros(
            self.num_envs,
            2,
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
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        self.L0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.L0_dot = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta0_dot = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta1 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta2 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

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
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.theta_kp[:] = self.cfg.control.kp_theta
        self.theta_kd[:] = self.cfg.control.kd_theta
        self.l0_kp[:] = self.cfg.control.kp_l0
        self.l0_kd[:] = self.cfg.control.kd_l0
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
            self.theta_kp *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.theta_kp.shape,
                device=self.device,
            )
            self.l0_kp *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.l0_kp.shape,
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
            self.theta_kd *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.theta_kd.shape,
                device=self.device,
            )
            self.l0_kd *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.l0_kd.shape,
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

    # ------------ reward functions----------------
