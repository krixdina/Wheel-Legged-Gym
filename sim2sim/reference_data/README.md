# sim2sim 参考数据（sim2real 对照基线）

在 MuJoCo 里用与 [`../scripts/play_mujoco.py`](../scripts/play_mujoco.py) **完全相同**的管线跑策略，
把每个 100 Hz 控制步的**状态量**与**动作量**记录到 TensorBoard。sim2real 调试初期，可把实时从串口
**收到的状态量**和**下发的动作量**与这份已知良好的基线对照，快速判断数据是否正确。

## 采集

从仓库根目录运行：

```bash
conda run -n isaac_gym python sim2sim/reference_data/collect_reference_data.py --seconds 30 --vx 0.5
```

参数：`--seconds`（时长，默认 30）、`--vx/--wz/--height`（命令）、`--logdir`（父目录，默认 `runs/`）、
`--run_name`（自定义本次 run 子目录名，覆盖自动命名）。

每次采集都会在 `--logdir` 下生成**单独的命名子目录**作为本次 run，名称自动带上日期时间与命令值，例如：

```text
runs/20260609-201530_vx0p50_wz0p00_h0p18_t30s/
```

即"`日期-时间_vx<值>_wz<值>_h<值>_t<时长>s`"。命令值按两位小数编码，小数点写作 `p`、负号写作 `m`
（如 `-0.50` → `m0p50`），保证目录名是单一合法路径段。这样无需逐个打开就能在文件系统和 TensorBoard
的 run 列表中区分每次采集属于哪条命令。需要固定名称时用 `--run_name` 覆盖。

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
