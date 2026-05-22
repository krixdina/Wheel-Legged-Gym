# Rewards 对照表

## 1. 当前项目的奖励是怎么接起来的

- 奖励函数统一定义在 `wheel_legged_gym/envs/base/legged_robot.py` 的 `_reward_*` 方法中。
- 配置入口统一来自 `wheel_legged_gym/envs/base/legged_robot_config.py` 的 `class rewards` 和 `class rewards.scales`。
- 初始化时，`_prepare_reward_function()` 会把 `rewards.scales` 中权重为 `0` 的奖励项移除，并把非零权重乘上 `dt` 后注册到运行时奖励列表中。
- 每个控制步里，`compute_reward()` 会遍历这些“已注册的非零奖励项”逐项计算、乘权重、做单项裁剪、累加到总奖励；`termination` 如果启用，会在总奖励裁剪后单独追加。

对应代码：

- 奖励注册：`wheel_legged_gym/envs/base/legged_robot.py:1414`
- 奖励累计：`wheel_legged_gym/envs/base/legged_robot.py:466`
- 默认奖励配置：`wheel_legged_gym/envs/base/legged_robot_config.py:187`

## 2. 当前“是否被调用”的判定口径

项目当前注册了 6 个任务：

- `wheel_legged`
- `wheel_legged_vmc`
- `wheel_legged_vmc_flat`
- `wheel_legged_fyt`
- `wheel_legged_vmc_fyt`
- `wheel_legged_vmc_flat_fyt`

这些任务都没有单独覆写 `class rewards` 或 `class rewards.scales`，因此它们当前使用的奖励开关完全一致，统一继承自 `LeggedRobotCfg.rewards.scales`。

也就是说：

- 某奖励项在 `rewards.scales` 里有非零权重：当前会被调用。
- 某奖励项没有出现在 `rewards.scales` 里，或者权重为 `0`：当前不会被调用。

当前实际启用的奖励项共有 15 个：

- `tracking_lin_vel`
- `tracking_lin_vel_enhance`
- `tracking_ang_vel`
- `base_height`
- `nominal_state`
- `lin_vel_z`
- `ang_vel_xy`
- `orientation`
- `dof_vel`
- `dof_acc`
- `torques`
- `action_rate`
- `action_smooth`
- `collision`
- `dof_pos_limits`

## 3. 奖励相关的全局配置参数

| 参数 | 默认值 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `rewards.only_positive_rewards` | `False` | 普通奖励项累加后，是否把总奖励下限裁到 `0` | 已被调用 |
| `rewards.clip_single_reward` | `1` | 每个奖励项在单步内的裁剪幅度上限，实际裁剪范围是 `[-clip_single_reward * dt, clip_single_reward * dt]` | 已被调用 |
| `rewards.tracking_sigma` | `0.25` | 速度跟踪类指数奖励的误差尺度 | 已被调用 |
| `rewards.soft_dof_pos_limit` | `0.97` | 关节位置软限位比例，影响 `dof_pos_limits` 的触发阈值 | 已被调用 |
| `rewards.soft_dof_vel_limit` | `1.0` | 关节速度软限位比例，供 `dof_vel_limits` 使用 | 已实现，但当前未生效 |
| `rewards.soft_torque_limit` | `1.0` | 力矩软限位比例，供 `torque_limits` 使用 | 已实现，但当前未生效 |
| `rewards.base_height_target` | `0.18` | 预留参数 | 当前未被任何奖励代码读取 |
| `rewards.max_contact_force` | `100.0` | 足端接触力阈值，供 `feet_contact_forces` 使用 | 已实现，但当前未生效 |

## 4. 奖励项总表

### 4.1 当前已启用的奖励项

- `tracking_lin_vel`
  定义位置：`legged_robot.py::_reward_tracking_lin_vel`
  当前 scale：`1.0`
  奖励项意义：鼓励机器人前向速度跟踪命令值，误差越小奖励越高。
  相关 config 参数：`rewards.scales.tracking_lin_vel`，`rewards.tracking_sigma`，`commands.ranges.lin_vel_x`
  当前是否被调用：是

- `tracking_lin_vel_enhance`
  定义位置：`legged_robot.py::_reward_tracking_lin_vel_enhance`
  当前 scale：`1.0`
  奖励项意义：前向速度跟踪的辅助训练信号。它不是额外主奖励，而是一个衰减更慢的平滑负反馈，帮助在误差较大时也保留学习信号。
  相关 config 参数：`rewards.scales.tracking_lin_vel_enhance`，`rewards.tracking_sigma`，`commands.ranges.lin_vel_x`
  当前是否被调用：是

- `tracking_ang_vel`
  定义位置：`legged_robot.py::_reward_tracking_ang_vel`
  当前 scale：`1.0`
  奖励项意义：鼓励偏航角速度跟踪命令值。
  相关 config 参数：`rewards.scales.tracking_ang_vel`，`rewards.tracking_sigma`，`commands.ranges.ang_vel_yaw`
  当前是否被调用：是

- `base_height`
  定义位置：`legged_robot.py::_reward_base_height`
  当前 scale：`1.0`
  奖励项意义：鼓励机身高度贴近高度命令；当前配置下是正向指数奖励，越接近命令高度越接近 `1`。
  相关 config 参数：`rewards.scales.base_height`，`commands.ranges.height`
  当前是否被调用：是

- `nominal_state`
  定义位置：`legged_robot.py::_reward_nominal_state`
  当前 scale：`-0.1`
  奖励项意义：惩罚左右虚拟腿摆角不一致，鼓励保持对称站立/运动姿态。
  相关 config 参数：`rewards.scales.nominal_state`
  当前是否被调用：是

- `lin_vel_z`
  定义位置：`legged_robot.py::_reward_lin_vel_z`
  当前 scale：`-2.0`
  奖励项意义：惩罚机身竖直方向速度，抑制上下弹跳。
  相关 config 参数：`rewards.scales.lin_vel_z`
  当前是否被调用：是

- `ang_vel_xy`
  定义位置：`legged_robot.py::_reward_ang_vel_xy`
  当前 scale：`-0.05`
  奖励项意义：惩罚机身 `roll/pitch` 角速度，减少前后和左右甩动。
  相关 config 参数：`rewards.scales.ang_vel_xy`
  当前是否被调用：是

- `orientation`
  定义位置：`legged_robot.py::_reward_orientation`
  当前 scale：`-10.0`
  奖励项意义：惩罚机身倾斜，鼓励底盘保持水平。
  相关 config 参数：`rewards.scales.orientation`
  当前是否被调用：是

- `dof_vel`
  定义位置：`legged_robot.py::_reward_dof_vel`
  当前 scale：`-5e-5`
  奖励项意义：惩罚腿部关节速度过大，降低腿部摆动过快的问题；当前实现不统计轮速。
  相关 config 参数：`rewards.scales.dof_vel`
  当前是否被调用：是

- `dof_acc`
  定义位置：`legged_robot.py::_reward_dof_acc`
  当前 scale：`-2.5e-7`
  奖励项意义：惩罚关节加速度过大，抑制动作过于激烈。
  相关 config 参数：`rewards.scales.dof_acc`
  当前是否被调用：是

- `torques`
  定义位置：`legged_robot.py::_reward_torques`
  当前 scale：`-0.0001`
  奖励项意义：惩罚关节力矩平方和，约束“用力过猛”。
  相关 config 参数：`rewards.scales.torques`
  当前是否被调用：是

- `action_rate`
  定义位置：`legged_robot.py::_reward_action_rate`
  当前 scale：`-0.01`
  奖励项意义：惩罚相邻控制步动作跳变过大。
  相关 config 参数：`rewards.scales.action_rate`
  当前是否被调用：是

- `action_smooth`
  定义位置：`legged_robot.py::_reward_action_smooth`
  当前 scale：`-0.01`
  奖励项意义：用二阶差分惩罚动作变化过快，比 `action_rate` 更强调“动作变化过程的平滑性”。
  相关 config 参数：`rewards.scales.action_smooth`
  当前是否被调用：是

- `collision`
  定义位置：`legged_robot.py::_reward_collision`
  当前 scale：`-1.0`
  奖励项意义：惩罚指定机体部位发生接触；用于抑制腿部或机身撞地/撞障碍。
  相关 config 参数：`rewards.scales.collision`，`asset.penalize_contacts_on`
  当前是否被调用：是

- `dof_pos_limits`
  定义位置：`legged_robot.py::_reward_dof_pos_limits`
  当前 scale：`-1.0`
  奖励项意义：惩罚关节位置逼近或越过软限位，防止动作贴边和越界。
  相关 config 参数：`rewards.scales.dof_pos_limits`，`rewards.soft_dof_pos_limit`
  当前是否被调用：是

### 4.2 已实现但当前未启用的奖励项

- `base_height_enhance`
  定义位置：`legged_robot.py::_reward_base_height_enhance`
  奖励项意义：高度主奖励的辅助项。在高度误差较大时提供更平滑、更慢衰减的修正信号。
  相关 config 参数：`rewards.scales.base_height_enhance`，`commands.ranges.height`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `power`
  定义位置：`legged_robot.py::_reward_power`
  奖励项意义：惩罚机械功率消耗，约束高力矩高转速同时出现。
  相关 config 参数：`rewards.scales.power`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `termination`
  定义位置：`legged_robot.py::_reward_termination`
  奖励项意义：对“真正失败终止”追加终止奖惩，不对单纯超时追加。
  相关 config 参数：`rewards.scales.termination`，`asset.terminate_after_contacts_on`，`env.fail_to_terminal_time_s`，`env.episode_length_s`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `dof_vel_limits`
  定义位置：`legged_robot.py::_reward_dof_vel_limits`
  奖励项意义：惩罚关节速度逼近软速度上限。
  相关 config 参数：`rewards.scales.dof_vel_limits`，`rewards.soft_dof_vel_limit`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `torque_limits`
  定义位置：`legged_robot.py::_reward_torque_limits`
  奖励项意义：惩罚力矩逼近软力矩上限。
  相关 config 参数：`rewards.scales.torque_limits`，`rewards.soft_torque_limit`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `tracking_ang_vel_enhance`
  定义位置：`legged_robot.py::_reward_tracking_ang_vel_enhance`
  奖励项意义：偏航角速度跟踪的辅助训练信号，在误差较大时保持更平滑的负反馈。
  相关 config 参数：`rewards.scales.tracking_ang_vel_enhance`，`rewards.tracking_sigma`，`commands.ranges.ang_vel_yaw`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `tracking_lin_vel_pbrs`
  定义位置：`legged_robot.py::_reward_tracking_lin_vel_pbrs`
  奖励项意义：基于前向速度跟踪“比上一控制步有没有变好”的增量奖励。
  相关 config 参数：`rewards.scales.tracking_lin_vel_pbrs`，`rewards.tracking_sigma`，`commands.ranges.lin_vel_x`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `tracking_ang_vel_pbrs`
  定义位置：`legged_robot.py::_reward_tracking_ang_vel_pbrs`
  奖励项意义：基于偏航角速度跟踪“比上一控制步有没有变好”的增量奖励。
  相关 config 参数：`rewards.scales.tracking_ang_vel_pbrs`，`rewards.tracking_sigma`，`commands.ranges.ang_vel_yaw`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `stumble`
  定义位置：`legged_robot.py::_reward_stumble`
  奖励项意义：惩罚足端撞击近似垂直表面，抑制绊腿/磕碰。
  相关 config 参数：`rewards.scales.stumble`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `stand_still`
  定义位置：`legged_robot.py::_reward_stand_still`
  奖励项意义：当速度命令接近零时，惩罚关节偏离默认站立姿态，鼓励静止站稳。
  相关 config 参数：`rewards.scales.stand_still`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

- `feet_contact_forces`
  定义位置：`legged_robot.py::_reward_feet_contact_forces`
  奖励项意义：惩罚足端接触力过大，抑制落地冲击过猛。
  相关 config 参数：`rewards.scales.feet_contact_forces`，`rewards.max_contact_force`
  当前是否被调用：否
  未调用原因：`rewards.scales` 中未配置

## 5. 和当前配置有关的几个关键结论

1. 当前项目所有任务的奖励启用集合完全相同，因为所有任务都继承了同一份 `LeggedRobotCfg.rewards.scales`，没有任务级覆写。
2. `tracking_lin_vel_enhance` 当前虽然名字里带 `enhance`，但它本质上返回的是 `(-1, 0]` 范围内的辅助负反馈，作用更像“平滑惩罚项”，不是单独的正向主奖励。
3. `base_height` 当前不是使用 `base_height_target`，而是直接跟踪 `self.commands[:, 2]` 对应的高度命令，所以 `rewards.base_height_target` 现在是未使用参数。
4. `soft_dof_vel_limit`、`soft_torque_limit`、`max_contact_force` 这些参数虽然已经实现，但因为对应奖励项没有启用，所以当前训练中不会实际生效。
5. 如果后续你在某个任务配置里新增 `class rewards(LeggedRobotCfg.rewards)` 覆写 `scales`，那这张表里的“当前是否被调用”就需要按该任务重新核对。

## 6. scripts
```
tensorboard --logdir logs/wheel_legged_vmc_fyt --host 0.0.0.0 --port 6007 --reload_interval 5
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat_fyt --num_envs=4096 --headless
python wheel_legged_gym/scripts/play_keyboard.py \
  --task wheel_legged_vmc_fyt \
  --experiment_name experiment_data/env_nums_experiment \
  --load_run May11_18-43-09_num-envs-4096_dr-false 
``` 