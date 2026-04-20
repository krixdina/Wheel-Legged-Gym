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

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
# 作用：只保留四元数表示的偏航角，用它把向量从机器人局部朝向旋转到世界水平面方向。
# 输入：quat 表示机器人姿态四元数；vec 表示需要按偏航方向旋转的向量或向量批次。
# 输出：返回旋转后的向量，调用方通常用它得到随机器人朝向变化的水平采样点。
def quat_apply_yaw(quat, vec):
    # 构造仅包含水平偏航信息的四元数，去掉横滚和俯仰对向量的影响。
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
# 作用：把角度值折叠到以 pi 为边界的周期范围内，便于计算最短方向的角度误差。
# 输入：angles 表示一个或一批弧度制角度。
# 输出：返回规整后的角度，数值会落在不超过一个整圆的等价表示中。
def wrap_to_pi(angles):
    # 先消除完整圈数，再把大于 pi 的角度移到负方向的等价角度上。
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
# 作用：在给定范围内生成随机数，使采样更偏向区间两端。
# 输入：lower 和 upper 表示采样范围边界；shape 表示随机张量形状；device 表示张量所在设备。
# 输出：返回位于 lower 到 upper 之间的随机张量，形状由 shape 决定。
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    # 对指定形状的张量进行采样，先在 [-1, 1] 中均匀采样。torch.rand() 采样范围是 [0, 1)
    r = 2*torch.rand(*shape, device=device) - 1

    # 如果 r 中某个元素 < 0，就使用 -torch.sqrt(-r)，否则使用 torch.sqrt(r)
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    # 将变换后的数值映射到 [0, 1]，最后缩放到调用方指定的实际范围。
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower
