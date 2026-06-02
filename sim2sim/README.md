# Sim2Sim 验证（MuJoCo）

把在 Isaac Gym 中训练好的 FYT 轮腿策略放到 MuJoCo 里独立跑一遍，用来验证策略
对仿真器差异的鲁棒性（sim2sim）。本目录不依赖 Isaac Gym，只依赖 `mujoco` 与
`torch`。

## 目录结构

```
sim2sim/
├── model/
│   ├── model_8000.pt          # 被验证的策略（env_nums dr-true 主模型）
│   └── wheel_legged_v4.xml    # 由 URDF 自动生成的 MJCF（build_mjcf.py 产物）
├── scripts/
│   ├── build_mjcf.py          # URDF -> MJCF 转换（一次性）
│   ├── config/
│   │   └── sim2sim.yaml       # 所有冻结参数（与训练快照逐项核对）
│   ├── policy.py              # 加载 .pt，重建 encoder+actor 做推理
│   ├── wl_controller.py       # VMC/五连杆FK/PD 控制律（numpy 版，对齐训练代码）
│   └── play_mujoco.py         # 主验证脚本
└── README.md
```

## 运行环境

已安装在 `isaac_gym` conda 环境（`mujoco==2.3.6`，与现有 `numpy 1.19.5` 兼容，
未升级 numpy 因此不影响 Isaac Gym）。

## 使用方法

均从仓库根目录运行。

带可视化窗口、原地站立：

```bash
conda run -n isaac_gym python sim2sim/scripts/play_mujoco.py
```

下发前进速度命令（m/s）/ 偏航角速度（rad/s）/ 机体高度（m）：

```bash
conda run -n isaac_gym python sim2sim/scripts/play_mujoco.py --vx 0.8 --wz 0.0 --height 0.18
```

无界面、跑固定时长（适合服务器/快速回归）：

```bash
conda run -n isaac_gym python sim2sim/scripts/play_mujoco.py --headless --seconds 6 --vx 1.0
```

重新生成 MJCF（仅当 URDF 改变时）：

```bash
conda run -n isaac_gym python sim2sim/scripts/build_mjcf.py
```

## 已验证结果（headless，6s）

| 命令 vx | 稳态实测 vx | 结论 |
|---:|---:|---|
| 0.0 | ≈0.0 | 原地平衡 |
| 0.5 | ≈0.44 | 跟踪正常 |
| 1.0 | ≈0.97 | 跟踪正常 |
| -0.5 | ≈-0.72 | 反向跟踪正常 |

机体在全过程保持直立（base z 远高于 0.05 的跌倒阈值）。

## 关键对齐点（与训练代码一致）

- **控制频率**：sim 200 Hz（dt=0.005），策略 100 Hz（decimation=2）。
- **观测 27 维**：`base_ang_vel(3) | projected_gravity(3) | commands(3) |
  theta0(2) | theta0_dot(2) | L0(2) | L0_dot(2) | wheel_pos(2) | wheel_vel(2) |
  last_actions(6)`，缩放系数见 `scripts/config/sim2sim.yaml`。
- **历史观测**：5 帧滑动窗口（135 维）输入 encoder 估计 latent（含机体线速度）。
- **控制律**：动作 -> theta0/L0/轮速参考 -> 阻抗 PD + VMC 闭式雅可比 -> 6 关节力矩，
  左腿镜像取负。几何 l1=0.21, l2=0.25, offset=0。
- **MJCF 取舍**：轮子碰撞用实测圆柱（r=0.0579, 半宽0.019）；base_link 视觉用真实
  mesh（原 STL 106 万面超过 MuJoCo 20 万上限，已用 MeshLab 减面到 19 万面），但
  碰撞仍用不可见 box（mesh 碰撞会被转成凸包裹住腿部关节）；其余连杆视觉+碰撞均用
  mesh。

## 注意

- 该模型是 `dr-true`（带域随机化）主模型，更适合 sim2sim。若换模型/任务，需同步核对
  `scripts/config/sim2sim.yaml` 中的维度、缩放与增益，否则 sim2sim 失真。
- 第三维命令是机体高度，归一化用的是 `height_measurements=5.0`（与训练一致）。
