# sim2real 调试可视化（ROS2）

把 sim2real 部署主循环每个控制步的「上行状态 + 缩放前/后动作」发布成 ROS2 话题，
供 Foxglove / PlotJuggler / rqt_plot 实时可视化，并与 sim2sim 参考轨迹逐字段对比。

这是 **纯调试** 功能：默认关闭，只有 `config/sim2real.yaml` 里 `debug: true` 时才启用；
`rclpy` 在 `debug=true` 时才被惰性导入，因此非调试部署在 `isaac_gym`(Python 3.7) 下
完全不受影响、不引入任何 ROS 依赖。

## 为什么必须用 Python 3.10 环境

ROS2 Humble 的 `rclpy` 是为 **Python 3.10** 构建的，无法在 `isaac_gym`(Python 3.7) 中导入。
NUC 端 `sim2real/setup.py` 已约束部署环境为 Python 3.10，因此调试部署请在 Python 3.10 环境中运行
（例如 conda 环境 `sim2real_deploy_py310_test`，其中 `torch/pyserial/yaml` 与 `rclpy` 已验证可共存）。

## 自定义消息

| 消息 | 用途 | 话题 |
|---|---|---|
| `DebugState` | 21 维原始上行状态（字段名对齐 sim2sim 参考轨迹 tag） | `/sim2real_debug/state` |
| `DebugAction` | 6 维动作，**缩放前后共用同一类型** | `/sim2real_debug/action_raw`、`/sim2real_debug/action_scaled` |

`action_raw` 是裁剪后的策略原始输出（无量纲）；`action_scaled` 是下发给下位机的物理动作
（`theta0_ref [rad]`、`l0_ref [m]`、`wheel_vel_ref [rad/s]`）。三条消息共用同一时间戳，便于
时间序列工具对齐。

## 一次性：构建消息包

```bash
source /opt/ros/humble/setup.bash
cd sim2real/ros2
colcon build
```

`build/`、`install/`、`log/` 已被 `.gitignore` 忽略，不会误提交。

## 运行（开启调试）

1. 把 `sim2real/config/sim2real.yaml` 的 `debug` 改为 `true`。
2. 在 **Python 3.10 + 已 source ROS2** 的环境中启动部署：

```bash
conda activate sim2real_deploy_py310_test          # 任意装好 deploy 依赖的 py3.10 环境
source /opt/ros/humble/setup.bash
source sim2real/ros2/install/setup.bash             # 让 wheel_legged_msgs 可被导入
python sim2real/python/deploy.py --device cpu
```

启动时会打印 `debug=true: publishing ... to ROS2 (/sim2real_debug/*)`。

## 无串口上位机自测

在连接下位机之前，可以先发布一段构造的 sim2real debug 数据，验证 Python 3.10、ROS2 Humble、
`wheel_legged_msgs`、三个 topic 和后台发布线程是否工作正常。该命令不打开串口、不加载策略模型、不下发动作：

```bash
conda activate sim2real_deploy_py310_test
source /opt/ros/humble/setup.bash
source sim2real/ros2/install/setup.bash
python sim2real/python/debug_fake_data.py --seconds 10 --vx 0.5
```

如果已安装 `sim2real/setup.py` 中的 console script，也可以运行：

```bash
sim2real-debug-fake-data --seconds 10 --vx 0.5
```

## 查看数据

```bash
source /opt/ros/humble/setup.bash
source sim2real/ros2/install/setup.bash
ros2 topic list | grep sim2real_debug              # 三个话题
ros2 topic echo /sim2real_debug/state              # 文本查看
ros2 topic hz   /sim2real_debug/action_scaled      # 频率核验（≈100 Hz）
```

- **Foxglove**：用 Foxglove Bridge 或 `ros2 bag` 录制后打开，Raw Messages / Plot 面板逐字段绘图。
- **PlotJuggler**：`ros2 run plotjuggler plotjuggler`，选 ROS2 数据源，拖拽字段绘图。

## 不阻塞策略推理

部署主循环只调用 `publish_step()`，它把一条记录 `put_nowait` 进有界队列即返回（队列满则丢弃，
调试数据尽力而为），**绝不调用 rclpy、绝不阻塞**。真正的建节点与发布全部在后台工作线程完成。
实测在 100 Hz 下三话题各 200/200 不丢、稳定 ~99 Hz。实现见 `sim2real/python/debug_publisher.py`。
