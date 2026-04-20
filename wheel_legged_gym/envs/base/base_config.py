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

import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)

    # 这段代码的作用是：
    # 遍历一个对象 obj 的所有属性名
    # 找出其中“值是类”的属性
    # 把这些类实例化
    # 再递归处理这些实例内部的类
    # 
    # 精简示例:
    # class MyCfg(BaseConfig):
    #     class env:
    #         num_envs = 4096
    #     class terrain:
    #         class generator:
    #             slope = 15
    #
    # cfg = MyCfg() 之后，这个函数会把 cfg.env 从“类”变成“实例”，
    # 也会把 cfg.terrain.generator 从“类”变成“实例”，于是可以统一写成:
    # cfg.env.num_envs
    # cfg.terrain.generator.slope
    #
    # 这样做的原因:
    # self.init_member_classes(self) 中传入的 self 就是“当前对象本身”，
    # 目的是从当前对象开始，递归地把内部类转换成一棵可直接访问、可修改的对象树。
    #
    # 如果不这样做:
    # 1. cfg.env、cfg.terrain 等仍然只是类定义，而不是实例的一部分。
    # 2. 不同配置对象可能共享同一个内部类，修改一个对象的子配置可能污染另一个对象。
    #    例如 cfg1 = MyCfg(), cfg2 = MyCfg() 时，如果 env 仍然是类而不是实例，
    #    那么 cfg1.env 和 cfg2.env 实际上都指向同一个 MyCfg.env。
    #    此时若执行 cfg1.env.num_envs = 2048，cfg2.env.num_envs 也可能一起变成 2048，
    #    因为修改的是“共享的类属性”，而不是 cfg1 独有的一份子配置。
