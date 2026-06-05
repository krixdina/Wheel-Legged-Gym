# Sim2Sim 网络推理最大频率测试

本文档测量 sim2sim 策略网络的**最大推理频率**,即 `play_mujoco.py` 每个控制步调用的
`SequencePolicy.act(obs, obs_history)`（encoder + actor 前向，`torch.no_grad()`，numpy
进 / numpy 出）能达到的极限速率。测量**只包含网络推理本身**，不含 MuJoCo 物理步进与
VMC/PD 控制律。

## 测试结论

| 设备 | 完整 `act()` 延迟 | **完整 `act()` 最大频率** | 纯前向延迟 | 纯前向最大频率 |
|---|---:|---:|---:|---:|
| **CPU** | 约 53–55 μs | **约 18,000–18,600 Hz** | 约 48 μs | 约 20,700–20,900 Hz |
| CUDA (RTX 3050 Laptop) | 约 121–138 μs | 约 7,200–8,200 Hz | 约 89–93 μs | 约 10,800–11,300 Hz |

> **CPU 上完整 `act()` 的最大推理频率约为 18 kHz**，这是部署实际可达的上限（含 numpy↔tensor
> 数据搬运）。纯前向（输入已是 on-device tensor）约 21 kHz，作为去除数据搬运后的理论上界参考。

### 关键发现：CPU 比 GPU 快

本策略网络极小（encoder `135→128→64→3`，actor `30→128→64→32→6`，单样本 batch=1），
计算量远小于 GPU 每次 kernel 启动与 host↔device 数据传输的固定开销。因此 GPU 的固定开销
反成瓶颈，CPU 反而更快。这也解释了 sim2sim 默认使用 `--device cpu`（见 `play_mujoco.py`
的 `--device` 默认值）的原因。

### 与实际控制频率的对照

sim2sim 实际策略控制频率由 `sim2sim.yaml` 的 `control_timing` 决定：

```
sim_dt = 0.005 s, decimation = 2  ->  策略控制频率 = 1 / (0.005 × 2) = 100 Hz
```

CPU 推理上限约 18 kHz，是实际所需 100 Hz 的约 **180 倍**。因此**网络推理在 sim2sim 中
完全不是性能瓶颈**，瓶颈在 MuJoCo 物理步进与（viewer 模式下的）实时节流。

## 测试方法

- **被测对象**：`sim2sim/scripts/policy.py` 中的 `SequencePolicy`，权重为 `sim2sim/model/model_8000.pt`。
- **两种口径**：
  - *完整 `act()`*：与 `play_mujoco.py` 每步调用完全一致，含 numpy↔tensor 转换与
    `.cpu().numpy()` 回传，反映真实可达频率。
  - *纯前向*：输入预先放到目标设备，仅计时 encoder+actor，去除数据搬运，作为上界参考。
- **预热**：每项测量前先空跑 200 次（GPU 需预热以排除 CUDA 首次初始化开销）。
- **计时**：`time.perf_counter()` 对 20000 次调用取平均；GPU 在计时前后调用
  `torch.cuda.synchronize()` 等待异步 kernel 完成。
- **复现性**：连续运行两次，CPU 完整 `act()` 频率偏差 < 3%，结果稳定。

## 运行环境

- conda 环境：`isaac_gym`
- Python 3.7.12
- PyTorch 1.13.1+cu117
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU（CUDA 可用）

## 复现命令

```bash
conda run -n isaac_gym python sim2sim/scripts/benchmark_inference.py
```

基准测试脚本：`sim2sim/scripts/benchmark_inference.py`。
