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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch


# Base class for RL tasks
class BaseTask:

    # 作用：
    # 这个构造函数负责完成所有强化学习环境实例都共用的底层初始化工作，
    # 包括连接 Isaac Gym 仿真接口、确定计算与渲染设备、读取环境规模配置、
    # 创建训练阶段要反复复用的张量缓冲区，以及建立仿真与可视化窗口。
    #
    # 输入：
    # cfg：环境配置对象，里面的 env 子配置给出并行环境数量、观测维度、动作维度、
    #      历史观测长度以及是否使用特权观测等基础规格。
    # sim_params：Isaac Gym 的仿真参数对象，决定时间步长、是否启用 GPU 流水线等底层运行方式。
    # physics_engine：物理引擎类型，用来告诉 Isaac Gym 采用哪一种物理求解器创建仿真。
    # sim_device：仿真设备字符串，表示物理仿真准备运行在哪块 CPU 或 GPU 上。
    # headless：是否无界面运行；为 True 时不创建可视化窗口，只保留训练所需的仿真。
    #
    # 输出：
    # 这个函数没有显式返回值；它的作用是把当前环境对象初始化成“可被训练器直接使用”的状态。
    # 初始化完成后，当前对象会持有仿真句柄、设备信息、观测与奖励等缓冲区，
    # 以及后续 step()、reset() 和渲染流程依赖的基础资源。
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 获取 Isaac Gym 的全局接口对象，后续创建仿真、环境、viewer 都通过它完成。
        self.gym = gymapi.acquire_gym()

        # 保存仿真运行时最基础的输入参数，供子类和后续仿真创建流程复用。
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # 决定环境内部张量实际放在哪个设备上：
        # 只有“仿真本身跑在 GPU 上”并且“启用了 GPU 流水线”时，
        # Isaac Gym 返回的状态张量才会直接留在 GPU；否则训练侧统一按 CPU 处理。
        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # 图形设备默认跟随仿真设备；如果是无界面模式，就把图形设备禁用掉。
        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        # 从配置中读取并行环境规模和观测/动作规格，这些值会决定后面所有缓冲区的形状。
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.obs_history_length = cfg.env.obs_history_length
        
        # 关闭 PyTorch JIT 的 profiling 优化路径，避免在这个高频仿真训练场景里引入额外开销。
        # Just-In-Time Compilation，即“即时编译”。
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # 为并行强化学习流程预先分配所有常用缓冲区：
        # 包括当前观测、历史观测、奖励、是否重置、超时标记以及特权观测等。
        # 这些张量会在整个训练期间被原地复用，避免每一步都重复申请内存。
        # allocate buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_obs * self.obs_history_length,
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.fail_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.envs_steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.edge_reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        # extras 用来承载额外信息，例如每个 episode 的统计量或超时标记。
        self.extras = {}

        # 先让子类通过 create_sim() 实现具体的仿真创建，再调用 Isaac Gym 的准备流程，
        # 这样后续就能安全访问底层状态张量。
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # 初始化 viewer 的控制状态；默认启用“仿真与画面同步刷新”。
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # 如果允许显示窗口，就创建 viewer，并注册常用快捷键：
        # ESC 用于退出程序，V 用于暂停画面刷新（但仿真继续运行），方便调试和观察。
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

    def get_observations(self):
        return (
            self.obs_buf,
            self.obs_history,
        )

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    # 面向所有环境对象的全量初始化，通常在训练开始前调用一次。
    # Python 会根据 self 的真实类型做动态分派。
    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    # 作用：
    # 这个函数负责在训练或测试过程中维护可视化窗口，
    # 包括处理窗口关闭与键盘事件、同步仿真结果到图形系统，以及按需要刷新画面。
    # 它通常在每次环境 step() 开始时被调用，让仿真在推进前先把上一时刻的可视化状态处理完。
    #
    # 输入：
    # sync_frame_time：是否让显示速度与真实时间同步；为 True 时会主动等待，
    #                  使画面播放更接近实时速度；为 False 时尽量不做这层限速。
    #
    # 输出：
    # 这个函数没有显式返回值；它的效果是更新 viewer 相关状态。
    # 如果用户关闭窗口或按下退出快捷键，程序会直接结束；
    # 如果开启了画面同步，还会把当前仿真结果绘制到窗口中。
    def render(self, sync_frame_time=True):
        if self.viewer:
            # 只有在创建了 viewer 可视化窗口时才需要做渲染相关工作；
            # 无界面训练时 self.viewer 为空，这个函数会直接跳过。
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # 处理窗口输入事件：
            # 关闭命令会直接终止程序，切换命令会改变“画面是否跟随仿真同步刷新”的开关状态。
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # 当环境内部状态张量放在 GPU 上时，先把上一轮仿真的结果取回到图形系统可读取的位置，
            # 否则 viewer 可能拿不到最新的仿真状态。
            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # 如果当前启用了画面同步，就推进图形更新并真正把当前仿真场景绘制到窗口；
            # 可选的 sync_frame_time 用来限制刷新速度，避免画面播放过快。
            # 如果关闭了画面同步，则只轮询窗口事件，不主动重绘，这样可以减少渲染开销。
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
