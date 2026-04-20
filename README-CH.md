# Wheel-Legged-Gym 中文测试说明

本文档记录了在当前机器上对训练显存占用的实际测试结果，目的是找出 `--num_envs` 的可用上限，避免训练时因为显存不足而报错。

## 1. 测试机器

- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`
- 显存: `4096 MiB`
- 驱动: `535.230.02`
- Python 环境: `conda run -n isaac_gym`
- Python 版本: `3.7.12`
- PyTorch 版本: `1.13.1+cu117`
- 测试日期: `2026-04-08`

## 2. 测试方法

使用如下方式测试每一种组合：

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=<task_name> \
  --num_envs=<N> \
  --max_iterations=1
```

如果测试 `headless` 模式，则额外加上：

```bash
--headless
```

判定标准如下：

- 成功：训练能完整启动，并成功完成 `1` 次训练迭代后退出。
- 失败：出现 `CUDA out of memory`、`illegal memory access`、PhysX CUDA error、段错误等。

说明：这里测得的是“当前机器上可以成功启动并完成 1 次训练迭代的上限”。为了长期稳定训练，建议实际使用时比上限再留一点余量。

## 3. 测试结果

| 任务 | 模式 | 可用 `num_envs` 上限 | 结论 |
|---|---|---:|---|
| `wheel_legged_vmc_flat` | `--headless` | `3657` | `3657` 成功，`3658` OOM |
| `wheel_legged_vmc_flat` | 非 `--headless` | `2247` | `2247` 成功，`2248` OOM |
| `wheel_legged_vmc` | `--headless` | `969` | `969` 成功，`970` OOM |
| `wheel_legged_vmc` | 非 `--headless` | `0` | 即使 `--num_envs=1` 也会触发 PhysX/CUDA 非法内存访问并崩溃 |

## 4. 结论总结

如果你要在这台 4G 显存的机器上正常训练，可以直接按下面理解：

- `wheel_legged_vmc_flat --headless`：最多可用 `3657` 个环境。
- `wheel_legged_vmc_flat` 非 `--headless`：最多可用 `2247` 个环境。
- `wheel_legged_vmc --headless`：最多可用 `969` 个环境。
- `wheel_legged_vmc` 非 `--headless`：当前测试下无法正常启动，`num_envs=1` 也不行。

也就是说：

- 你如果训练 `wheel_legged_vmc_flat`，可以根据是否开 viewer 选择 `3657` 或 `2247` 作为极限值。
- 你如果训练 `wheel_legged_vmc`，建议只使用 `--headless`，并把环境数控制在 `969` 以内。
- `wheel_legged_vmc` 在非 `--headless` 模式下，不是“环境数太大”，而是当前机器上连最小环境数也无法稳定启动。

## 5. 推荐实际使用值

虽然上面给出了“能跑起来的极限值”，但实际长期训练建议保守一点，留出显存余量：

- `wheel_legged_vmc_flat --headless`：建议用 `3500`
- `wheel_legged_vmc_flat` 非 `--headless`：建议用 `2000`
- `wheel_legged_vmc --headless`：建议用 `900`
- `wheel_legged_vmc` 非 `--headless`：不建议使用

## 6. 推荐命令

### 6.1 平地任务，headless

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged_vmc_flat \
  --headless \
  --num_envs=3500
```

### 6.2 平地任务，带 viewer

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged_vmc_flat \
  --num_envs=2000
```

### 6.3 粗糙地形任务，headless

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged_vmc \
  --headless \
  --num_envs=900
```

## 7. 额外说明

- `wheel_legged_vmc_flat` 明显比 `wheel_legged_vmc` 更省显存，这和 rough terrain、测高点、地形相关计算的额外开销有关。
- 非 `--headless` 模式下，viewer/rendering 会进一步占用显存，所以同一任务下可用环境数会明显下降。
- 如果训练时同时开了浏览器、桌面特效、视频软件或其他占 GPU 的程序，稳定可用的环境数可能还要再调低一些。
