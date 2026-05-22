# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import sys

import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry

import numpy as np
import torch


DEFAULT_FORWARD_VELOCITY = 0.0
DEFAULT_YAW_RATE = 0.0
DEFAULT_BODY_HEIGHT = 0.18

FORWARD_VELOCITY_STEP = 0.1
FORWARD_VELOCITY_MIN = -2.5
FORWARD_VELOCITY_MAX = 2.5

YAW_RATE_STEP = 0.1
YAW_RATE_MIN = -3.1
YAW_RATE_MAX = 3.1

BODY_HEIGHT_STEP = 0.2
BODY_HEIGHT_MIN = 0.15
BODY_HEIGHT_MAX = 0.32


@dataclass
class KeyboardCommand:
    forward_velocity: float = DEFAULT_FORWARD_VELOCITY
    yaw_rate: float = DEFAULT_YAW_RATE
    body_height: float = DEFAULT_BODY_HEIGHT

    def reset(self):
        self.forward_velocity = DEFAULT_FORWARD_VELOCITY
        self.yaw_rate = DEFAULT_YAW_RATE
        self.body_height = DEFAULT_BODY_HEIGHT


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def configure_env_for_play(env_cfg):
    env_cfg.env.episode_length_s = 20
    env_cfg.env.fail_to_terminal_time_s = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
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

    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [
        DEFAULT_FORWARD_VELOCITY,
        DEFAULT_FORWARD_VELOCITY,
    ]
    env_cfg.commands.ranges.ang_vel_yaw = [DEFAULT_YAW_RATE, DEFAULT_YAW_RATE]
    env_cfg.commands.ranges.height = [DEFAULT_BODY_HEIGHT, DEFAULT_BODY_HEIGHT]


def subscribe_keyboard_events(env):
    if env.viewer is None:
        print("No viewer was created; keyboard input is unavailable.")
        return

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
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)


def apply_keyboard_command(env, command):
    env.commands[:, 0] = command.forward_velocity
    env.commands[:, 1] = command.yaw_rate
    env.commands[:, 2] = command.body_height
    env.commands[:, 3] = 0.0

    env.command_ranges["lin_vel_x"][:, 0] = command.forward_velocity
    env.command_ranges["lin_vel_x"][:, 1] = command.forward_velocity
    env.command_ranges["ang_vel_yaw"][:, 0] = command.yaw_rate
    env.command_ranges["ang_vel_yaw"][:, 1] = command.yaw_rate
    env.command_ranges["height"][:, 0] = command.body_height
    env.command_ranges["height"][:, 1] = command.body_height


def refresh_policy_observations(env):
    if hasattr(env, "compute_proprioception_observations"):
        clip_obs = env.cfg.normalization.clip_observations
        env.obs_buf = torch.clip(
            env.compute_proprioception_observations(), -clip_obs, clip_obs
        )
        if getattr(env, "obs_history", None) is not None:
            env.obs_history[:, -env.num_obs :] = env.obs_buf

    return env.get_observations()


def print_command_state(key, effect, command):
    print(f"Key '{key}' pressed: {effect}")
    print(
        "Command state -> "
        f"forward_velocity: {command.forward_velocity:.2f} m/s, "
        f"yaw_rate: {command.yaw_rate:.2f} rad/s, "
        f"body_height: {command.body_height:.2f} m"
    )


def handle_keyboard_events(env, command):
    if env.viewer is None:
        return False

    changed = False
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue

        if evt.action == "QUIT":
            sys.exit()
        if evt.action == "toggle_viewer_sync":
            env.enable_viewer_sync = not env.enable_viewer_sync
            continue

        if evt.action == "keyboard_forward":
            command.forward_velocity = clamp(
                command.forward_velocity + FORWARD_VELOCITY_STEP,
                FORWARD_VELOCITY_MIN,
                FORWARD_VELOCITY_MAX,
            )
            print_command_state(
                "w", "increased forward velocity command by +0.10 m/s.", command
            )
            changed = True
        elif evt.action == "keyboard_backward":
            command.forward_velocity = clamp(
                command.forward_velocity - FORWARD_VELOCITY_STEP,
                FORWARD_VELOCITY_MIN,
                FORWARD_VELOCITY_MAX,
            )
            print_command_state(
                "s", "decreased forward velocity command by -0.10 m/s.", command
            )
            changed = True
        elif evt.action == "keyboard_turn_left":
            command.yaw_rate = clamp(
                command.yaw_rate + YAW_RATE_STEP, YAW_RATE_MIN, YAW_RATE_MAX
            )
            print_command_state(
                "a",
                "increased yaw rate command for a left turn by +0.10 rad/s.",
                command,
            )
            changed = True
        elif evt.action == "keyboard_turn_right":
            command.yaw_rate = clamp(
                command.yaw_rate - YAW_RATE_STEP, YAW_RATE_MIN, YAW_RATE_MAX
            )
            print_command_state(
                "d",
                "decreased yaw rate command for a right turn by -0.10 rad/s.",
                command,
            )
            changed = True
        elif evt.action == "keyboard_height_up":
            command.body_height = clamp(
                command.body_height + BODY_HEIGHT_STEP, BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
            )
            print_command_state(
                "q", "increased body height command by +0.20 m.", command
            )
            changed = True
        elif evt.action == "keyboard_height_down":
            command.body_height = clamp(
                command.body_height - BODY_HEIGHT_STEP, BODY_HEIGHT_MIN, BODY_HEIGHT_MAX
            )
            print_command_state(
                "e", "decreased body height command by -0.20 m.", command
            )
            changed = True
        elif evt.action == "keyboard_reset":
            command.reset()
            print_command_state("r", "reset all commands to their defaults.", command)
            changed = True

    if changed:
        apply_keyboard_command(env, command)

    return changed


def print_controls():
    print("Keyboard control mode is ready.")
    print("Focus the Isaac Gym viewer before pressing control keys.")
    print("W/S: increase/decrease forward velocity command by 0.10 m/s.")
    print("A/D: increase/decrease yaw rate command by 0.10 rad/s.")
    print("Q/E: increase/decrease body height command by 0.20 m.")
    print("R: reset all commands.")


def update_camera_follow(env, env_cfg, robot_index):
    camera_offset = np.array(env_cfg.viewer.pos, dtype=np.float64)
    target_position = np.array(env.base_position[robot_index, :].to(device="cpu"))
    camera_position = target_position + camera_offset
    env.set_camera(camera_position, target_position)


def play_keyboard(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_env_for_play(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    camera_robot_index = min(CAMERA_ROBOT_INDEX, env.num_envs - 1)
    command = KeyboardCommand()
    subscribe_keyboard_events(env)
    apply_keyboard_command(env, command)
    obs, obs_history = refresh_policy_observations(env)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    print_controls()
    if MOVE_CAMERA:
        update_camera_follow(env, env_cfg, camera_robot_index)

    for _ in range(1000 * int(env.max_episode_length)):
        if handle_keyboard_events(env, command):
            obs, obs_history = refresh_policy_observations(env)

        if ppo_runner.alg.actor_critic.is_sequence:
            actions, _ = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        apply_keyboard_command(env, command)
        obs, _, _, _, _, obs_history = env.step(actions)
        if MOVE_CAMERA:
            update_camera_follow(env, env_cfg, camera_robot_index)
        if handle_keyboard_events(env, command):
            obs, obs_history = refresh_policy_observations(env)


if __name__ == "__main__":
    MOVE_CAMERA = False
    CAMERA_ROBOT_INDEX = 21
    args = get_args()
    play_keyboard(args)
