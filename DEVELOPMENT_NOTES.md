# 开发文档

本文件用于在开发过程中逐步记录值得回顾的设计决策与技术要点，供后续阅读参考。

## Isaac Gym（FYT VMC 训练）

### FYT 大腿关节不能设置为 `continuous`

**背景**：FYT 机器人的大腿关节在机械结构上理论上可以无限旋转，因此一开始容易认为 URDF
中应把大腿关节写成 `continuous`。但在 Isaac Gym 训练中，这会带来一个很隐蔽的问题：
`continuous` 关节导入后没有位置上下限，大腿的 `self.dof_pos_limits` 会变成 `[-inf, inf]`。

**关键影响**：项目中的 `_reward_dof_pos_limits()` 会根据 `self.dof_pos_limits` 判断关节是否接近
或超过软限位。如果大腿是 `continuous`，那么对大腿而言：

```text
self.dof_pos_limits = [-inf, inf]
```

这会导致大腿对应的 `dof_pos_limits` 奖励项始终为 0。换句话说，训练过程完全不知道“大腿角度
过大是不好的”。

**因果理解**：

- 大腿本身可以无限旋转，不等于训练 URDF 中应该设置为 `continuous`。
- 设置为 `continuous` 不是直接导致大腿无法运动，也不是直接导致大腿失去探索能力。
- 真正的问题是：`continuous` 让 Isaac Gym 不再使用大腿的角度边界，进而让
  `_reward_dof_pos_limits()` 对大腿失效。
- 当训练无法感知大腿角度边界时，策略缺少“靠近大腿边界是不良状态”的奖励信号，也就失去了一部分
  调节大腿探索分布的约束。
- 在 FYT VMC 中，腿长虚拟力不会直接作用到髋关节；如果再叠加过大的 `action_scale_theta`、
  过大的 `kp_theta` 或过软的腿长控制，运动更容易被小腿吸收，大腿探索幅度反而偏小，最终可能收敛到
  劈叉或低奖励策略。

**结论**：FYT 大腿关节在训练 URDF 中应设置为 `revolute`，并保留合理的 `lower/upper`。这并不是
为了否认机械结构上大腿可无限旋转，而是为了让 Isaac Gym 和奖励函数在训练期间拥有有限的角度边界信号。

所以更准确地说：

`continuous` 并不会直接限制大腿运动；相反，它允许大腿无限旋转。问题在于，大腿被建模为 `continuous` 后，训练过程无法感知其角度边界这一物理约束，`dof_pos_limits` 相关奖励也就无法对大腿角度过大的状态产生反馈。这样会削弱策略对大腿关节进行有效探索和约束的训练信号，最终可能导致大腿在训练过程中探索不足，并使策略收敛到失败或低奖励行为。

## sim2sim（MuJoCo 验证）

### 轮子碰撞用实测圆柱基元替代 mesh

**背景**：把训练用的 URDF 转成 MuJoCo 的 MJCF 时，轮子在 URDF 里是用三角网格
（mesh）描述的。如果让轮子继续用 mesh 做**碰撞几何**，轮地接触在 MuJoCo 里既慢又
不稳定。

**处理方式**：
- **视觉几何**仍用轮子原始 mesh，保证外形真实。
- **碰撞几何**改用 `cylinder` 几何基元，参数 `半径=0.0579, 半宽=0.019`。这两个数是从
  `left_wheel_link.STL` 的实际包围盒量出来的，不是估计值。
- 圆柱沿 link 局部 z 轴放置，正好是轮子的旋转轴。
- 显式设置摩擦 `friction="1.0 0.005 0.0001"`。

**关键认知**：仿真里一个部件可以同时挂"视觉几何"和"碰撞几何"两套，它们用途不同
（一个用于渲染、另一个用于物理接触），因此可以使用不同类型——视觉用精细 mesh、碰撞用
简单基元，是机器人仿真的通用范式。

相关实现见 `sim2sim/scripts/build_mjcf.py`。

### 用 ROS pub/sub 类比理解 `contype` 和 `conaffinity`

MuJoCo 中每个参与碰撞的 `geom` 都可以设置 `contype` 和 `conaffinity`。一个便于记忆的
类比是：

- `contype` 类似 ROS 中的 publisher：这个几何体“发布”自己属于哪些碰撞类型。
- `conaffinity` 类似 ROS 中的 subscriber：这个几何体“订阅”哪些碰撞类型，愿意和哪些
  类型发生接触。

两个几何体 A 和 B 是否会产生碰撞接触，MuJoCo 判断的是：

```text
(A.contype & B.conaffinity) != 0
或者
(B.contype & A.conaffinity) != 0
```

只要任意一个方向成立，二者就允许生成接触。

这个类比和 ROS pub/sub 的主要区别是：ROS 通常通过同一个 topic 名称进行通信匹配，而
`contype` / `conaffinity` 不是字符串 topic，也不是要求两个值完全相等，而是通过**位掩码**
判断是否存在交集。

例如：

```text
A.contype = 2      # 二进制 0010
B.conaffinity = 3  # 二进制 0011
```

虽然 `2 != 3`，但：

```text
2 & 3 = 2
```

结果非 0，因此 A 发布的某个碰撞类型被 B 订阅，二者可以碰撞。

当前 sim2sim 中使用这个机制关闭机器人内部自碰撞，同时保留机器人与地面的接触：

```text
地面:       contype=1, conaffinity=1
机器人几何: contype=2, conaffinity=1
```

因此：

```text
机器人 vs 地面:
robot.contype & floor.conaffinity = 2 & 1 = 0
floor.contype & robot.conaffinity = 1 & 1 = 1
=> 可以碰撞

机器人 vs 机器人:
robot_link_1.contype & robot_link_2.conaffinity = 2 & 1 = 0
robot_link_2.contype & robot_link_1.conaffinity = 2 & 1 = 0
=> 不发生内部自碰撞
```

这个设置用于对齐 Isaac Gym 训练侧关闭自碰撞的配置，避免 MuJoCo 中 `base_link` 的碰撞盒
和腿部碰撞几何互相卡住，导致关节运动迟滞和 sim2sim 失稳。

### MuJoCo 中 visual geom、collision geom 与 `group` 的区别

MuJoCo 里并没有强制把 `geom` 分成“视觉几何”和“碰撞几何”两种类型。它们本质上都是
`geom`，区别主要来自用途和属性设置：

- **visual geom**：用于渲染外观，通常使用较精细的 STL mesh，并设置
  `contype="0" conaffinity="0"`，使其不参与碰撞。
- **collision geom/contact geom**：用于物理接触检测，通常使用更简单、更稳定的几何表示，
  例如 `box`、`cylinder`，或者简化后的 mesh。

例如当前 `wheel_legged_v4.xml` 中的视觉几何：

```xml
<geom conaffinity="0" contype="0" group="1" mesh="base_link" rgba="0.75 0.75 0.78 1" type="mesh"/>
```

其中真正让它“不参与碰撞”的是：

```text
contype="0"
conaffinity="0"
```

而不是 `group="1"`。

相对地，机器人碰撞几何类似：

```xml
<geom conaffinity="1" contype="2" group="0" pos="0 0 0.215" rgba="0 0.25 1 0.55" size="0.30 0.196 0.236" type="box"/>
```

这类 `geom` 参与物理接触检测，并且为了调试方便，被设置为蓝色半透明。

`group` 的作用主要是 **viewer 显示分组**，可以理解为可视化图层。它用于在 MuJoCo viewer
中批量显示或隐藏某一类 `geom`，不直接决定碰撞关系。当前模型采用的约定是：

```text
group=0: collision geom/contact geom
group=1: visual geom
```

因此在 viewer 中可以：

- 只打开 `group 1`：查看机器人真实外观 mesh。
- 只打开 `group 0`：查看碰撞几何。
- 同时打开 `group 0` 和 `group 1`：对比视觉模型和碰撞模型是否对齐。
- 打开 contact points/contact forces：查看当前仿真帧中实际发生的接触点和接触力。

调试碰撞问题时，推荐同时查看碰撞几何和 contact points/contact forces。碰撞几何告诉我们
“MuJoCo 用什么形状做接触检测”，contact points/contact forces 则直接显示“当前哪些位置真的
发生了接触”。对于 `type="mesh"` 的碰撞几何，viewer 中的凸包可视化可以帮助理解 MuJoCo
实际用于 mesh 碰撞检测的近似外形；但是否真的发生接触，仍应结合 contact points/contact
forces 判断。
