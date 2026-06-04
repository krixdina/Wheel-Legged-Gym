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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from .wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc_fyt.wheel_legged_vmc_fyt import LeggedRobotVMCFYT
from .wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)
from .wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)
from .wheel_legged_fyt.wheel_legged_fyt_config import (
    WheelLeggedFYTCfg,
    WheelLeggedFYTCfgPPO,
)
from .wheel_legged_vmc_fyt.wheel_legged_vmc_fyt_config import (
    WheelLeggedVMCFYTCfg,
    WheelLeggedVMCFYTCfgPPO,
)
from .wheel_legged_vmc_flat_fyt.wheel_legged_vmc_flat_fyt_config import (
    WheelLeggedVMCFlatFYTCfg,
    WheelLeggedVMCFlatFYTCfgPPO,
)
from .ablation_experiment import (
    FYTAblationNoSequenceCfg,
    FYTAblationNoSequenceCfgPPO,
    FYTAblationNoCurriculumCfg,
    FYTAblationNoCurriculumCfgPPO,
    FYTAblationTerrainCurriculumOnlyCfg,
    FYTAblationTerrainCurriculumOnlyCfgPPO,
    FYTAblationNoEnhanceRewardsCfg,
    FYTAblationNoEnhanceRewardsCfgPPO,
)


import os

from wheel_legged_gym.utils.task_registry import task_registry

# 通过 register() 将所有环境类和配置对象注册到 task_registry 中（存储在对应的字典中）
# 训练时会根据命令行指定的任务名在字典中查找对应的环境类和配置对象，并创建仿真环境和 PPO 训练器。
# 在注册表中建立“任务名 -> 环境构造器 -> 环境配置对象 -> 训练配置对象”的映射。
# 第二个参数传入的是环境类本身，而不是环境实例；真正创建环境时还需要仿真参数、
# 运行设备等信息，所以会在 task_registry.make_env() 中再调用这个类来实例化。
# 后两个参数是配置对象，注册时就可以直接创建，用来保存环境参数和 PPO 训练参数。
task_registry.register(
    "wheel_legged", LeggedRobot, WheelLeggedCfg(), WheelLeggedCfgPPO()
)
task_registry.register(
    "wheel_legged_vmc", LeggedRobotVMC, WheelLeggedVMCCfg(), WheelLeggedVMCCfgPPO()
)
task_registry.register(
    "wheel_legged_vmc_flat",
    LeggedRobotVMC,  # 环境类作为稍后创建仿真环境的入口，此处不要加括号提前实例化。
    WheelLeggedVMCFlatCfg(),  # 环境配置对象，保存地形、观测、奖励等任务参数。
    WheelLeggedVMCFlatCfgPPO(),  # 训练配置对象，保存策略网络、PPO 算法和 runner 参数。
)
task_registry.register(
    "wheel_legged_fyt", LeggedRobot, WheelLeggedFYTCfg(), WheelLeggedFYTCfgPPO()
)
task_registry.register(
    "wheel_legged_vmc_fyt",
    LeggedRobotVMCFYT,
    WheelLeggedVMCFYTCfg(),
    WheelLeggedVMCFYTCfgPPO(),
)
task_registry.register(
    "wheel_legged_vmc_flat_fyt",
    LeggedRobotVMCFYT,
    WheelLeggedVMCFlatFYTCfg(),
    WheelLeggedVMCFlatFYTCfgPPO(),
)

# FYT ablation task: replace ActorCriticSequence with ActorCritic to remove
# history-based latent velocity estimation while keeping the baseline environment.
task_registry.register(
    "ablation_fyt_no_sequence",
    LeggedRobotVMCFYT,
    FYTAblationNoSequenceCfg(),
    FYTAblationNoSequenceCfgPPO(),
)

# FYT ablation task: disable both terrain curriculum and command curriculum.
task_registry.register(
    "ablation_fyt_no_curriculum",
    LeggedRobotVMCFYT,
    FYTAblationNoCurriculumCfg(),
    FYTAblationNoCurriculumCfgPPO(),
)

# FYT ablation task: keep terrain curriculum enabled, but disable command curriculum.
task_registry.register(
    "ablation_fyt_terrain_curriculum_only",
    LeggedRobotVMCFYT,
    FYTAblationTerrainCurriculumOnlyCfg(),
    FYTAblationTerrainCurriculumOnlyCfgPPO(),
)

# FYT ablation task: disable the enhance reward terms while keeping the baseline
# sequence policy and curriculum settings.
task_registry.register(
    "ablation_fyt_no_enhance_rewards",
    LeggedRobotVMCFYT,
    FYTAblationNoEnhanceRewardsCfg(),
    FYTAblationNoEnhanceRewardsCfgPPO(),
)
