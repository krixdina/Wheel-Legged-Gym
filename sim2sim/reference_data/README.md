# sim2sim 参考数据（sim2real 对照基线）

在 MuJoCo 里用与 [`../scripts/play_mujoco.py`](../scripts/play_mujoco.py) **完全相同**的管线跑策略，
把每个 100 Hz 控制步的**状态量**与**动作量**记录到 TensorBoard。sim2real 调试初期，可把实时从串口
**收到的状态量**和**下发的动作量**与这份已知良好的基线对照，快速判断数据是否正确。

## 采集

从仓库根目录运行：

```bash
conda run -n isaac_gym python sim2sim/reference_data/collect_reference_data.py --seconds 30 --vx 0.5
```

参数：`--seconds`（时长，默认 30）、`--vx/--wz/--height`（命令）、`--logdir`（默认 `runs/`）。
事件文件写入 [`runs/`](runs/)（重复运行会追加新的事件文件，TensorBoard 会一并显示）。

## 在浏览器查看

```bash
conda run -n isaac_gym tensorboard --logdir sim2sim/reference_data/runs
# 打开 http://localhost:6006
```

## 记录的标量（共 33 个）

- `state/*`（21 个）：下位机上行的**原始物理量**（未缩放）——`base_ang_vel_{x,y,z}`、
  `projected_gravity_{x,y,z}`、`command_{vx,wz,height}`、`theta0_{L,R}`、`theta0_dot_{L,R}`、
  `L0_{L,R}`、`L0_dot_{L,R}`、`wheel_pos_{L,R}`、`wheel_vel_{L,R}`。
- `action_raw/*`（6 个）：策略原始输出（裁剪后），顺序 `[L_theta, L_l0, L_wheel, R_theta, R_l0, R_wheel]`。
- `action_phys/*`（6 个）：NUC 下发的**带物理含义的动作量**——`theta0_ref_{L,R}` [rad]、
  `l0_ref_{L,R}` [m]、`wheel_vel_ref_{L,R}` [rad/s]，缩放公式与 controller / sim2real 一致。

横轴为控制步序号；同一 step 上 `state[k]` 对应产生 `action[k]`，与实时循环一致。
