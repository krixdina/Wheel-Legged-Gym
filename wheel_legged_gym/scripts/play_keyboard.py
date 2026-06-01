# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import argparse
import os
import sys
import time

import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry

import numpy as np
import torch


# 键盘控制命令的默认值，启动和重置时都会回到这些状态。
DEFAULT_FORWARD_VELOCITY = 0.0
DEFAULT_YAW_RATE = 0.0
DEFAULT_BODY_HEIGHT = 0.15
COMMAND_EPSILON = 1e-6

# 各控制量的步进和边界，避免按键连续输入后命令超出环境可接受范围。
FORWARD_VELOCITY_STEP = 0.1
FORWARD_VELOCITY_MIN = -2.5
FORWARD_VELOCITY_MAX = 2.5

YAW_RATE_STEP = 0.1
YAW_RATE_MIN = -3.1
YAW_RATE_MAX = 3.1

BODY_HEIGHT_STEP = 0.03
BODY_HEIGHT_MIN = 0.15
BODY_HEIGHT_MAX = 0.32

DEFAULT_KEY_FRAME_DIR = "keyboard_frames"
IDLE_SETTLE_TIME_S = 0.25
KEYBOARD_ACTION_NAMES = {
    "keyboard_forward",
    "keyboard_backward",
    "keyboard_turn_left",
    "keyboard_turn_right",
    "keyboard_height_up",
    "keyboard_height_down",
    "keyboard_reset",
}


# 保存当前键盘命令状态，后续会同步写入环境的 commands 和 command_ranges。
@dataclass
class KeyboardCommand:
    forward_velocity: float = DEFAULT_FORWARD_VELOCITY
    yaw_rate: float = DEFAULT_YAW_RATE
    body_height: float = DEFAULT_BODY_HEIGHT

    def reset(self):
        # 将所有手动控制量恢复到脚本启动时的默认命令。
        self.forward_velocity = DEFAULT_FORWARD_VELOCITY
        self.yaw_rate = DEFAULT_YAW_RATE
        self.body_height = DEFAULT_BODY_HEIGHT

    def is_idle(self):
        return (
            abs(self.forward_velocity - DEFAULT_FORWARD_VELOCITY) < COMMAND_EPSILON
            and abs(self.yaw_rate - DEFAULT_YAW_RATE) < COMMAND_EPSILON
            and abs(self.body_height - DEFAULT_BODY_HEIGHT) < COMMAND_EPSILON
        )


def clamp(value, lower, upper):
    # 将单个控制命令限制在预设范围内，防止键盘累加越界。
    return max(lower, min(upper, value))


def extract_keyboard_recording_args():
    # 先解析录制专用参数，再把剩余参数交回项目通用参数解析器。
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--key_frame_dir",
        "--key-frame-dir",
        "--key_frame_folder",
        "--key-frame-folder",
        dest="key_frame_dir",
        default=DEFAULT_KEY_FRAME_DIR,
        help="Folder name under logs/<experiment>/exported for saved key-frame images.",
    )
    keyboard_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_argv]
    # 输出目录只允许是简单文件夹名，避免意外写到日志目录之外。
    keyboard_args.key_frame_dir = validate_key_frame_dir(keyboard_args.key_frame_dir)
    return keyboard_args


def validate_key_frame_dir(folder_name):
    # 校验截图子目录名称，保证后续 os.path.join 得到的是安全的相对目录。
    folder_name = folder_name.strip()
    if not folder_name:
        raise ValueError("--key_frame_dir must not be empty.")
    if os.path.isabs(folder_name):
        raise ValueError("--key_frame_dir expects a folder name, not an absolute path.")
    if os.sep in folder_name or (os.altsep is not None and os.altsep in folder_name):
        raise ValueError("--key_frame_dir expects a folder name without path separators.")
    if folder_name in {".", ".."}:
        raise ValueError("--key_frame_dir must be a normal folder name.")
    return folder_name


def configure_env_for_play(env_cfg):
    # 为交互式播放缩短回合、限制并行环境数量，并保留可视化地形多样性。
    env_cfg.env.episode_length_s = 20
    env_cfg.env.fail_to_terminal_time_s = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = env_cfg.terrain.mesh_type != "plane"

    # 关闭训练时常用的噪声和域随机化，使键盘调试时的机器人响应更稳定。
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.1, 0.2]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # 固定命令采样范围，后续由键盘事件直接覆盖这些命令值。
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [
        DEFAULT_FORWARD_VELOCITY,
        DEFAULT_FORWARD_VELOCITY,
    ]
    env_cfg.commands.ranges.ang_vel_yaw = [DEFAULT_YAW_RATE, DEFAULT_YAW_RATE]
    env_cfg.commands.ranges.height = [DEFAULT_BODY_HEIGHT, DEFAULT_BODY_HEIGHT]


# Purpose: 在 Isaac Gym 可视化窗口中注册键盘控制入口，把物理按键转换成后续循环可识别的动作名称。
# Inputs: env 表示当前仿真环境对象；这里使用它持有的可视化窗口 viewer 来接收键盘输入，并使用它持有的 Isaac Gym 接口完成事件订阅。
def subscribe_keyboard_events(env):
    # 只有创建了 viewer 时才能订阅键盘事件；headless 运行时直接提示并跳过。
    if env.viewer is None:
        print("No viewer was created; keyboard input is unavailable.")
        return

    # 将常用移动、转向、高度和重置按键映射为 Isaac Gym viewer action。
    # 这里的字符串是按键事件的语义标签：例如 W 键被标记为“前进命令”，后续事件处理逻辑会读取这个标签，并据此增加机器人的目标前进速度。
    key_actions = (
        (gymapi.KEY_W, "keyboard_forward"),
        (gymapi.KEY_S, "keyboard_backward"),
        (gymapi.KEY_A, "keyboard_turn_left"),
        (gymapi.KEY_D, "keyboard_turn_right"),
        (gymapi.KEY_Q, "keyboard_height_up"),
        (gymapi.KEY_E, "keyboard_height_down"),
        (gymapi.KEY_R, "keyboard_reset"),
    )
    for key, action in key_actions:
        # 订阅后，可视化窗口会在用户按下对应按键时产生带有动作名称的事件；主循环随后查询这些事件，并根据动作名称进入相应的命令更新分支。
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)


def install_keyboard_event_callback(env, command):
    if env.viewer is None:
        return

    def handle_event_from_render(evt):
        handled, changed = process_keyboard_event(env, command, evt)
        if changed:
            apply_keyboard_command(env, command)
        return handled

    env.viewer_action_callback = handle_event_from_render


def apply_keyboard_command(env, command):
    # commands 的前几维分别写入前进速度、偏航角速度、机体高度和 heading 占位。
    env.commands[:, 0] = command.forward_velocity
    env.commands[:, 1] = command.yaw_rate
    env.commands[:, 2] = command.body_height
    env.commands[:, 3] = 0.0

    # 同步收窄命令范围，避免环境内部重采样把键盘命令覆盖成其他值。
    env.command_ranges["lin_vel_x"][:, 0] = command.forward_velocity
    env.command_ranges["lin_vel_x"][:, 1] = command.forward_velocity
    env.command_ranges["ang_vel_yaw"][:, 0] = command.yaw_rate
    env.command_ranges["ang_vel_yaw"][:, 1] = command.yaw_rate
    env.command_ranges["height"][:, 0] = command.body_height
    env.command_ranges["height"][:, 1] = command.body_height


# Purpose: 在键盘命令被修改后，重新生成策略下一次推理要读取的观测，确保观测中的目标速度、目标高度等命令量已经是最新值。
# Inputs: env 表示当前仿真环境对象；这里会读取它保存的最新机器人状态、键盘命令、观测裁剪范围和历史观测缓存。
# Outputs: 返回当前普通观测和历史观测；同时会更新环境内部的普通观测缓存，并在存在历史观测缓存时把最新一帧写入历史窗口末尾。
def refresh_policy_observations(env):
    # 键盘命令变化后，立即重算策略观测，保证下一次推理能看到最新命令。
    if hasattr(env, "compute_proprioception_observations"):
        clip_obs = env.cfg.normalization.clip_observations
        env.obs_buf = torch.clip(
            env.compute_proprioception_observations(), -clip_obs, clip_obs
        )
        # 序列策略依赖历史观测，这里只替换最新一帧，保持历史缓冲区结构不变。
        if getattr(env, "obs_history", None) is not None:
            env.obs_history[:, -env.num_obs :] = env.obs_buf

    return env.get_observations()


def print_command_state(key, effect, command):
    # 每次有效按键后打印本次影响和完整命令状态，方便交互调试。
    print(f"Key '{key}' pressed: {effect}")
    print(
        "Command state -> "
        f"forward_velocity: {command.forward_velocity:.2f} m/s, "
        f"yaw_rate: {command.yaw_rate:.2f} rad/s, "
        f"body_height: {command.body_height:.2f} m"
    )


def handle_keyboard_events(env, command):
    # 读取 viewer 中累积的键盘事件，并把有效事件转换为命令状态变化。
    if env.viewer is None:
        return False

    changed = False
    for evt in env.gym.query_viewer_action_events(env.viewer):
        _, event_changed = process_keyboard_event(env, command, evt)
        changed = changed or event_changed

    if changed:
        # 只有命令发生变化时才写回环境，减少无意义的状态更新。
        apply_keyboard_command(env, command)

    return changed


def process_keyboard_event(env, command, evt):
    # 只处理按下事件，忽略释放或无效值，避免一次按键触发两次调整。
    if evt.value <= 0:
        return evt.action in KEYBOARD_ACTION_NAMES, False

    # 保留 Isaac Gym viewer 的退出和同步切换行为。
    if evt.action == "QUIT":
        sys.exit()
    if evt.action == "toggle_viewer_sync":
        env.enable_viewer_sync = not env.enable_viewer_sync
        return True, False

    # W/S 调整 x 方向线速度命令，正值为前进，负值为后退。
    if evt.action == "keyboard_forward":
        command.forward_velocity = clamp(
            command.forward_velocity + FORWARD_VELOCITY_STEP,
            FORWARD_VELOCITY_MIN,
            FORWARD_VELOCITY_MAX,
        )
        print_command_state(
            "w", "increased forward velocity command by +0.10 m/s.", command
        )
        return True, True
    if evt.action == "keyboard_backward":
        command.forward_velocity = clamp(
            command.forward_velocity - FORWARD_VELOCITY_STEP,
            FORWARD_VELOCITY_MIN,
            FORWARD_VELOCITY_MAX,
        )
        print_command_state(
            "s", "decreased forward velocity command by -0.10 m/s.", command
        )
        return True, True
    # A/D 调整偏航角速度命令，用于控制机器人左转或右转。
    if evt.action == "keyboard_turn_left":
        command.yaw_rate = clamp(
            command.yaw_rate + YAW_RATE_STEP, YAW_RATE_MIN, YAW_RATE_MAX
        )
        print_command_state(
            "a",
            "increased yaw rate command for a left turn by +0.10 rad/s.",
            command,
        )
        return True, True
    if evt.action == "keyboard_turn_right":
        command.yaw_rate = clamp(
            command.yaw_rate - YAW_RATE_STEP, YAW_RATE_MIN, YAW_RATE_MAX
        )
        print_command_state(
            "d",
            "decreased yaw rate command for a right turn by -0.10 rad/s.",
            command,
        )
        return True, True
    # Q/E 调整机体高度命令，用于在允许范围内抬高或降低机器人身体。
    if evt.action == "keyboard_height_up":
        command.body_height = clamp(
            command.body_height + BODY_HEIGHT_STEP, BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
        )
        print_command_state("q", "increased body height command by +0.03 m.", command)
        return True, True
    if evt.action == "keyboard_height_down":
        command.body_height = clamp(
            command.body_height - BODY_HEIGHT_STEP, BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
        )
        print_command_state("e", "decreased body height command by -0.03 m.", command)
        return True, True
    # R 将所有键盘命令恢复到默认值。
    if evt.action == "keyboard_reset":
        command.reset()
        print_command_state("r", "reset all commands to their defaults.", command)
        return True, True

    return False, False


def print_controls(frame_dir=None):
    # 启动时输出交互说明，让用户确认 viewer 焦点、按键映射和录制路径。
    print("Keyboard control mode is ready.")
    print("Focus the Isaac Gym viewer before pressing control keys.")
    print("W/S: increase/decrease forward velocity command by 0.10 m/s.")
    print("A/D: increase/decrease yaw rate command by 0.10 rad/s.")
    print("Q/E: increase/decrease body height command by 0.03 m.")
    print("R: reset all commands.")
    if RECORD_FRAMES:
        print(
            f"Frames will be saved every 2 environment steps at "
            f"{FRAME_CAPTURE_WIDTH}x{FRAME_CAPTURE_HEIGHT} for 50 FPS video composition."
        )
        if frame_dir is not None:
            print(f"Frame output directory: {frame_dir}")


def update_camera_follow(env, env_cfg, robot_index):
    # 使用 viewer 配置中的相机偏移量，让相机持续对准指定机器人。
    camera_offset = np.array(env_cfg.viewer.pos, dtype=np.float64)
    target_position = np.array(env.base_position[robot_index, :].to(device="cpu"))
    camera_position = target_position + camera_offset
    env.set_camera(camera_position, target_position)
    return camera_position, target_position


def create_recording_camera(env, env_index):
    # 为指定环境创建离屏相机，用于按照固定分辨率保存关键帧。
    camera_props = gymapi.CameraProperties()
    camera_props.width = FRAME_CAPTURE_WIDTH
    camera_props.height = FRAME_CAPTURE_HEIGHT
    camera_handle = env.gym.create_camera_sensor(env.envs[env_index], camera_props)
    if camera_handle == -1:
        raise RuntimeError("Failed to create recording camera sensor.")
    return camera_handle


def update_recording_camera(
    env, camera_handle, env_index, camera_position, target_position
):
    # 录制相机跟随当前 viewer 相机位置，保证截取的画面与观察视角一致。
    env.gym.set_camera_location(
        camera_handle,
        env.envs[env_index],
        gymapi.Vec3(*camera_position.tolist()),
        gymapi.Vec3(*target_position.tolist()),
    )


def render_idle_frame(env, command):
    # 空闲时只刷新 viewer 和键盘事件，避免策略长时间在零速度命令下进入难以起步的静止状态。
    env.render()
    apply_keyboard_command(env, command)
    time.sleep(env.dt)


def play_keyboard(args):
    # 载入任务配置并改写为适合键盘交互播放的确定性环境设置。
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_env_for_play(env_cfg)

    # 创建环境、选择被相机跟随的机器人，并把初始键盘命令写入环境。
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    camera_robot_index = min(CAMERA_ROBOT_INDEX, env.num_envs - 1)
    command = KeyboardCommand()
    subscribe_keyboard_events(env)
    install_keyboard_event_callback(env, command)
    apply_keyboard_command(env, command)

    # 以 resume 模式加载训练好的 PPO 策略，后续循环只做推理和环境步进。
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    apply_keyboard_command(env, command)
    obs, obs_history = refresh_policy_observations(env)
    img_idx = 0
    frame_dir = None
    recording_camera = None
    if RECORD_FRAMES:
        # 截图输出到当前实验日志的 exported 子目录，文件夹名来自命令行参数。
        frame_dir = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            getattr(args, "key_frame_dir", DEFAULT_KEY_FRAME_DIR),
        )
        os.makedirs(frame_dir, exist_ok=True)
        recording_camera = create_recording_camera(env, camera_robot_index)

    # 初始化 viewer 相机和录制相机，确保第一帧之前就对准目标机器人。
    print_controls(frame_dir)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_target = np.array(env_cfg.viewer.lookat, dtype=np.float64)
    if MOVE_CAMERA:
        camera_position, camera_target = update_camera_follow(
            env, env_cfg, camera_robot_index
        )
    if RECORD_FRAMES:
        update_recording_camera(
            env, recording_camera, camera_robot_index, camera_position, camera_target
        )

    idle_settle_steps = 0
    idle_settle_steps_after_motion = max(1, int(IDLE_SETTLE_TIME_S / env.dt))

    for i in range(1000 * int(env.max_episode_length)):
        # 每个仿真步前先处理一次按键，让策略推理尽快使用最新命令。
        handle_keyboard_events(env, command)
        apply_keyboard_command(env, command)
        obs, obs_history = refresh_policy_observations(env)

        if env.viewer is not None and command.is_idle():
            if idle_settle_steps <= 0:
                render_idle_frame(env, command)
                obs, obs_history = refresh_policy_observations(env)
                if command.is_idle():
                    continue
            else:
                idle_settle_steps -= 1
        else:
            idle_settle_steps = idle_settle_steps_after_motion

        # 序列策略需要观测历史，普通策略只需要当前观测。
        if ppo_runner.alg.actor_critic.is_sequence:
            actions, _ = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        # 在 step 前再次写入命令，抵消环境内部可能发生的命令重采样。
        apply_keyboard_command(env, command)
        obs, _, _, _, _, obs_history = env.step(actions)
        # step 内部可能触发 reset 或命令重采样；立刻恢复键盘命令并刷新下一次推理的观测。
        apply_keyboard_command(env, command)
        obs, obs_history = refresh_policy_observations(env)
        if MOVE_CAMERA:
            # 仿真推进后更新跟随相机，使视角始终锁定同一个机器人。
            camera_position, camera_target = update_camera_follow(
                env, env_cfg, camera_robot_index
            )
        if RECORD_FRAMES:
            # 每隔两个环境步保存一张图，用于后续按 50 FPS 合成视频。
            update_recording_camera(
                env, recording_camera, camera_robot_index, camera_position, camera_target
            )
            if i % 2:
                filename = os.path.join(frame_dir, f"{img_idx}.png")
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                env.gym.write_camera_image_to_file(
                    env.sim,
                    env.envs[camera_robot_index],
                    recording_camera,
                    gymapi.IMAGE_COLOR,
                    filename,
                )
                img_idx += 1
        # step 后再处理一次按键，降低交互输入在画面和命令上的延迟。
        if handle_keyboard_events(env, command):
            apply_keyboard_command(env, command)
            obs, obs_history = refresh_policy_observations(env)


if __name__ == "__main__":
    # 脚本入口处集中定义播放选项，便于临时调整相机、录制和截图分辨率。
    MOVE_CAMERA = False
    CAMERA_ROBOT_INDEX = 33
    RECORD_FRAMES = False
    FRAME_CAPTURE_WIDTH = 1280
    FRAME_CAPTURE_HEIGHT = 960
    keyboard_args = extract_keyboard_recording_args()
    args = get_args()
    args.key_frame_dir = keyboard_args.key_frame_dir
    play_keyboard(args)
