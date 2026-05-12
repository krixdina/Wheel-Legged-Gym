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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


# 机器人在课程地形上的初始分配策略：
# 1. 这个 Terrain 类只负责生成一张 num_rows x num_cols 的大地图；
#    默认配置中 num_rows = 10 表示 10 个难度行，num_cols = 20 表示 20 个地形类型列。
# 2. 具体把每个并行机器人放到哪一块子地形上，是在 LeggedRobot._get_env_origins() 中完成的：
#    terrain_levels 表示每个机器人所在的难度行，terrain_types 表示每个机器人所在的地形类型列。
# 3. terrain_types 由环境编号按 num_cols 均匀划分；默认 num_envs=4096、num_cols=20，
#    所以每个地形类型列大约分到 204 或 205 个机器人。
# 4. 默认 curriculum=True 且 max_init_terrain_level=5，因此训练刚开始时：
#    terrain_levels ∈ {0, 1, 2, 3, 4, 5}，也就是每列分配到的机器人只从前 6 个难度行中随机初始化。
# 5. 机器人最终出生点由 terrain_origins[terrain_levels, terrain_types] 查表得到；
#    后续课程学习主要改变 terrain_levels，让机器人在同一地形类型列中上下移动难度。
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        """地形构建器。

        这个类处在“配置”与“Isaac Gym 仿真地形对象”之间，负责：
        1. 读取 `LeggedRobotCfg.terrain` 中的地形配置；
        2. 生成整张训练地图对应的高度图 `height_field_raw`；
        3. 为每个子地形计算机器人初始摆放原点 `env_origins`；
        4. 在 `mesh_type == "trimesh"` 时，把高度图进一步转换成三角网格。

        它不会直接参与策略推理或奖励计算，但会决定：
        - 训练时机器人脚下的地形长什么样；
        - 每个并行环境被摆放在大地图的什么位置；
        - 高度采样 `_get_heights()` 所依据的底层高度数据是什么。

        输入：
        - cfg: 地形相关配置，来自 `LeggedRobotCfg.terrain`
        - num_robots: 并行环境中的机器人数量。当前文件中主要用于保存上下文，
          真正决定地图划分的是 `cfg.num_rows * cfg.num_cols`

        输出：
        - self.height_field_raw / self.heightsamples: 整张地图的离散高度图
        - self.env_origins: 每个子地形中心平台的世界坐标原点
        - self.vertices, self.triangles: 仅在 trimesh 模式下生成的三角网格
        """

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        # `none` 和 `plane` 不需要在这里生成复杂地形数据：
        # - `plane` 直接在 simulator 中创建无限平面
        # - `none` 表示不提供地形网格，通常也不能做高度采样
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # 这里把“各类地形的概率占比”转成累积概率区间形式，后续 `make_terrain()`
        # 用一个 [0, 1) 的随机数 `choice` 来决定落入哪类地形。
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]
    
        # 整张训练地图由 `num_rows * num_cols` 个子地形拼接而成。
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # `env_origins[i, j]` 保存第 i 行、第 j 列子地形中机器人出生点的世界坐标。
        # 创建了一个 num_rows*num_cols*3 维的数组，每个位置存3个值，即为(x,y,z)坐标
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        
        # 把每块子地形从“米”换算成“高度图网格数”。 
        # horizontal_scale 表示每个网格点对应多少米。
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        
        # 将外围边界的宽度也换算为网格数
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)

        # 算整张大地图的网格大小 = 所有子地形拼起来 + 外围边界。
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 这是整张大地图的底层高度图，单位不是米，而是“离散高度格”。
        # 真实高度需要再乘以 `vertical_scale`。
        # 创建了一个 tot_rows*tot_cols 的数组，这个数组中的每个元素代表了离散后的高度格的数量
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # 下面三种模式决定“如何把各个子地形填进整张地图”：
        # - curriculum: 行表示难度递增，列表示地形类型递进
        # - selected: 所有格子都使用同一种指定地形
        # - randomized_terrain: 每个格子独立随机
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        # `LeggedRobot._create_heightfield()` / `_create_trimesh()` 会直接读取这里的数据。
        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            # Isaac Gym 的三角网格地形最终也是从高度图转换而来。
            self.vertices, self.triangles = (
                terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
            )

    def randomized_terrain(self):
        """随机生成整张地图。

        每个子地形独立随机采样：
        - `choice` 决定地形类别；
        - `difficulty` 决定该类别下的难度强度。
        """
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            # 将一维索引还原为多维数组中的坐标
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        """按课程学习方式生成地图。

        约定：
        - 行 `i` 控制难度，越往后越难；
        - 列 `j` 控制地形类型，越往后 `choice` 越大。
        
        具体来说：
        - 同一列 j 的 choice 相同，所以大体属于同一类地形。
        - 同一列中，行索引 i 越大，difficulty = i / num_rows 越大，地形越难。
        - 同一行中，难度相同，但不同列对应不同地形类型。

        这样做的结果是：整张大地图天然带有“从简单到困难”的分布，
        后续环境 reset 时可以通过 `terrain_levels` / `terrain_types`
        选择把机器人放到哪一块地形上。
        """
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        """使用配置中指定的一种地形，铺满所有子地形。"""
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        """生成一块子地形。

        输入：
        - choice: [0, 1) 的实数，用于按比例选择地形类别
        - difficulty: 难度系数，控制坡度、台阶高度、坑深等参数

        输出：
        - terrain_utils.SubTerrain 实例，其中 `height_field_raw` 已被写入
        """
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        # 下面这些参数都由 difficulty 推出。difficulty 越大，地形通常越激进。
        slope = difficulty * 0.404
        random_height = 0.05 + difficulty * 0.05
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.05
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        # `choice` 与累积概率区间 `self.proportions` 对比后，
        # 决定当前子地形属于哪一种类别。
        if choice < self.proportions[0]:
            # 第一段比例区间生成近似平地：这里仍调用金字塔坡地生成器，
            # 但坡度设为 0，只保留中心平台和基本高度图结构。
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.0)
        elif choice < self.proportions[1]:
            # 第二段比例区间生成平滑坡地；区间前半段把坡度取反，
            # 用同一类地形覆盖两列相反方向的金字塔坡面地形。
            if (
                choice
                < self.proportions[0] + (self.proportions[1] - self.proportions[0]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[2]:
            # 第三段比例区间生成粗糙坡地：先生成较缓的坡面，
            # 再叠加随机高度扰动，让地表不再是光滑平面。
            if (
                choice
                < self.proportions[1] + (self.proportions[2] - self.proportions[1]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope * 0.485, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_height,
                max_height=random_height,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[4]:
            # 第四、第五段比例区间共同生成台阶地形；
            # 前一段把台阶高度取反，后一段保持正值，用于得到两种相反高度方向的金字塔式台阶。
            if choice < self.proportions[3]:
                step_height *= -1
            two_step_pyramid_stairs_terrain(
                terrain, step_height=step_height, platform_size=4.0
            )
        elif choice < self.proportions[5]:
            # 第六段比例区间生成离散障碍地形，在中心平台外随机放置若干矩形障碍块。
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[6]:
            # 如果配置中继续扩展了地形比例区间，这一段可生成踏石地形。
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[7]:
            # 如果配置中继续扩展了地形比例区间，这一段可生成中间带缺口的地形。
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            # 剩余比例区间生成坑洞地形；当前默认比例配置下通常不会进入这里。
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        """把一块子地形写入整张大地图，并计算该子地形的环境原点。

        输入：
        - terrain: 单块 `SubTerrain`
        - row, col: 该子地形在大地图中的网格位置

        输出：
        - 无显式返回值；副作用是更新：
          1. `self.height_field_raw`
          2. `self.env_origins[row, col]`
        """
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        # 把子地形的高度图复制到大地图对应的切片区域。
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # 这里给每个子环境定义一个世界坐标原点：
        # x/y 取子地形中心；
        # z 取中心附近一块平台区域的最高点，避免机器人出生时嵌入地面。
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.0):
    """在子地形中央挖一个“壕沟/缺口”。

    输入：
    - terrain: 待修改的 `SubTerrain`
    - gap_size: 缺口尺寸，单位米
    - platform_size: 中间保留的可站立平台尺寸，单位米

    输出：
    - 无显式返回值；直接原地修改 `terrain.height_field_raw`
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def two_step_pyramid_stairs_terrain(terrain, step_height, platform_size=1.0):
    """生成“二级台阶”版本的金字塔台阶地形。

    相比原始的多级同心台阶，这里只保留两级高度跃迁：
    - 外圈平地
    - 中间一圈过渡台阶
    - 中心平台

    这样仍能保留明显的台阶结构，但难度低于原始多级台阶，高于单级台阶。
    """
    platform_size = int(platform_size / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    half_platform = platform_size // 2

    x1_inner = center_x - half_platform
    x2_inner = center_x + half_platform
    y1_inner = center_y - half_platform
    y2_inner = center_y + half_platform

    half_outer = min(
        terrain.length // 2,
        terrain.width // 2,
        int(np.ceil(platform_size * 0.75)),
    )
    x1_outer = center_x - half_outer
    x2_outer = center_x + half_outer
    y1_outer = center_y - half_outer
    y2_outer = center_y + half_outer

    terrain.height_field_raw[:, :] = 0
    terrain.height_field_raw[x1_outer:x2_outer, y1_outer:y2_outer] = step_height
    terrain.height_field_raw[x1_inner:x2_inner, y1_inner:y2_inner] = 2 * step_height


def pit_terrain(terrain, depth, platform_size=1.0):
    """在子地形中央挖一个“坑”。

    输入：
    - terrain: 待修改的 `SubTerrain`
    - depth: 坑深，单位米
    - platform_size: 坑区域边长的一半相关尺度，单位米

    输出：
    - 无显式返回值；直接原地修改 `terrain.height_field_raw`
    """
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
