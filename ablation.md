# FYT 消融实验设置

本文档记录 FYT VMC 粗糙地形任务的消融实验设置。所有新增消融任务均从现有 baseline `wheel_legged_vmc_fyt` 派生，避免修改 baseline 参数。

正式训练统一使用以下公共参数：

```bash
--headless --num_envs=4096 --max_iterations=8000
```

项目运行环境为 conda 虚拟环境 `isaac_gym`。

## Baseline

baseline 使用现有任务：

```text
wheel_legged_vmc_fyt
```

baseline 设置保持不变：

- 策略网络：`ActorCriticSequence`
- 使用历史观测构建 latent 速度估计
- 启用地形课程学习
- 启用指令课程学习
- 保留 `tracking_lin_vel_enhance` 与已有奖励配置

第二组消融实验中的第三组“地形学习 + 指令学习均启用”即为该 baseline，已经训练完成，不需要再启动新的训练命令。

## 实验一：历史观测 latent 速度估计消融

目的：验证 `ActorCriticSequence` 接收历史观测并构建 latent 速度估计的效果。

消融设置：

- 任务名：`ablation_fyt_no_sequence`
- 环境配置：继承 `wheel_legged_vmc_fyt`
- 策略网络：由 `ActorCriticSequence` 替换为 `ActorCritic`
- 其他环境、课程学习和奖励配置保持 baseline 设置

正式训练命令：

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=ablation_fyt_no_sequence \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000
```

## 实验二：地形课程学习与指令课程学习消融

目的：比较地形课程学习和指令课程学习对训练效果的影响。

### 2.1 不启用地形学习，也不启用指令学习

消融设置：

- 任务名：`ablation_fyt_no_curriculum`
- `terrain.curriculum = False`
- `commands.curriculum = False`
- `commands.ranges.lin_vel_x = [-2.5, 2.5]`

速度范围直接设为 `[-2.5, 2.5]`，避免关闭指令课程后速度范围停留在初始 `[-1.0, 1.0]`。

正式训练命令：

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=ablation_fyt_no_curriculum \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000
```

### 2.2 启用地形学习，不启用指令学习

消融设置：

- 任务名：`ablation_fyt_terrain_curriculum_only`
- `terrain.curriculum = True`
- `commands.curriculum = False`
- `commands.ranges.lin_vel_x = [-2.5, 2.5]`

正式训练命令：

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=ablation_fyt_terrain_curriculum_only \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000
```

### 2.3 启用地形学习，也启用指令学习

该组为 baseline：

```text
wheel_legged_vmc_fyt
```

该模型已经训练完成，本轮不再重复训练。

## 实验三：enhance 奖励项消融

目的：验证 enhance 奖励项对训练效果的关键作用。

消融设置：

- 任务名：`ablation_fyt_no_enhance_rewards`
- `rewards.scales.tracking_lin_vel_enhance = 0.0`
- `rewards.scales.base_height_enhance = 0.0`
- 其他策略网络、课程学习和环境配置保持 baseline 设置

正式训练命令：

```bash
conda run -n isaac_gym python wheel_legged_gym/scripts/train.py \
  --task=ablation_fyt_no_enhance_rewards \
  --headless \
  --num_envs=4096 \
  --max_iterations=8000
```

## 新增任务汇总

| 消融目的 | 任务名 | 主要改动 |
|---|---|---|
| 历史观测 latent 速度估计 | `ablation_fyt_no_sequence` | `ActorCriticSequence -> ActorCritic` |
| 双课程关闭 | `ablation_fyt_no_curriculum` | 关闭地形课程与指令课程，速度范围固定为 `[-2.5, 2.5]` |
| 仅地形课程 | `ablation_fyt_terrain_curriculum_only` | 开启地形课程，关闭指令课程，速度范围固定为 `[-2.5, 2.5]` |
| enhance 奖励关闭 | `ablation_fyt_no_enhance_rewards` | 关闭 `tracking_lin_vel_enhance` 与 `base_height_enhance` |
