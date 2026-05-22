import os
from datetime import datetime

import numpy as np

import isaacgym
from isaacgym import gymapi, gymtorch, gymutil
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils.task_registry import task_registry


DEFAULT_TASK = "wheel_legged_vmc_fyt"
DEFAULT_SCREENSHOT_DIR = os.path.join(
    WHEEL_LEGGED_GYM_ROOT_DIR, "logs", "terrain_renders"
)
PREVIEW_BORDER_SIZE = 80.0


def parse_args():
    args = gymutil.parse_arguments(
        description="Show Isaac Gym terrains without visible robots.",
        custom_parameters=[
            {
                "name": "--task",
                "type": str,
                "default": DEFAULT_TASK,
                "help": "Registered task whose terrain configuration will be displayed.",
            },
            {
                "name": "--headless",
                "action": "store_true",
                "default": False,
                "help": "Kept for CLI compatibility. This script requires a viewer and will reject headless mode.",
            },
            {
                "name": "--rl_device",
                "type": str,
                "default": "cuda:0",
                "help": "Device used by the RL algorithm.",
            },
            {
                "name": "--num_envs",
                "type": int,
                "help": "Ignored by this script. Terrain preview always creates one hidden robot env.",
            },
            {
                "name": "--seed",
                "type": int,
                "help": "Random seed. Overrides config file if provided.",
            },
            {
                "name": "--max_iterations",
                "type": int,
                "help": "Unused compatibility argument.",
            },
            {
                "name": "--resume",
                "action": "store_true",
                "default": False,
                "help": "Unused compatibility argument.",
            },
            {
                "name": "--experiment_name",
                "type": str,
                "help": "Unused compatibility argument.",
            },
            {
                "name": "--run_name",
                "type": str,
                "help": "Unused compatibility argument.",
            },
            {
                "name": "--load_run",
                "type": str,
                "help": "Unused compatibility argument.",
            },
            {
                "name": "--checkpoint",
                "type": int,
                "help": "Unused compatibility argument.",
            },
            {"name": "--exptid", "type": str, "default": "", "help": "Unused compatibility argument."},
            {
                "name": "--camera",
                "type": str,
                "default": "top",
                "help": "Camera preset: top or angled.",
            },
            {
                "name": "--save_image",
                "type": str,
                "default": "",
                "help": "Optional path for an automatic screenshot after startup.",
            },
            {
                "name": "--frames_before_capture",
                "type": int,
                "default": 10,
                "help": "Number of viewer frames before the automatic screenshot is saved.",
            },
            {
                "name": "--quit_after_save",
                "action": "store_true",
                "default": False,
                "help": "Exit right after the automatic screenshot is saved.",
            },
            {
                "name": "--use_task_layout",
                "action": "store_true",
                "default": False,
                "help": "Use the task's original terrain layout instead of forcing a full curriculum gallery.",
            },
        ],
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def validate_args(args):
    if args.headless:
        raise ValueError("Terrain preview requires a viewer, so --headless is not supported.")
    if args.camera not in {"top", "angled"}:
        raise ValueError("--camera must be either 'top' or 'angled'.")
    if args.frames_before_capture < 0:
        raise ValueError("--frames_before_capture must be non-negative.")


def configure_env_for_terrain_view(env_cfg, use_task_layout):
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.border_size = max(
            float(env_cfg.terrain.border_size), PREVIEW_BORDER_SIZE
        )
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"] and not use_task_layout:
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.selected = False


def get_camera_pose(env_cfg, preset):
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        total_length = env_cfg.terrain.num_rows * env_cfg.terrain.terrain_length
        total_width = env_cfg.terrain.num_cols * env_cfg.terrain.terrain_width
        center = np.array(
            [
                total_length / 2.0,
                total_width / 2.0,
                0.0,
            ],
            dtype=np.float64,
        )
        max_dim = max(total_length, total_width)
    else:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        max_dim = 20.0

    if preset == "top":
        position = center + np.array([0.0, -max_dim * 0.375, max_dim * 0.375], dtype=np.float64)
    else:
        position = center + np.array(
            [-max_dim * 0.15625, -max_dim * 0.40625, max_dim * 0.3125],
            dtype=np.float64,
        )
    return position, center


def hide_robot(env):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.root_states[env_ids, 0] = -20.0
    env.root_states[env_ids, 1] = -20.0
    env.root_states[env_ids, 2] = 2.0
    env.root_states[env_ids, 7:13] = 0.0

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def get_default_screenshot_path(task_name):
    os.makedirs(DEFAULT_SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_SCREENSHOT_DIR, f"{task_name}_{timestamp}.png")


def save_viewer_image(env, path):
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    env.gym.write_viewer_image_to_file(env.viewer, abs_path)
    print(f"Saved terrain screenshot to: {abs_path}")


def print_controls(args):
    print(f"Showing terrain for task: {args.task}")
    print("Robots are moved outside the camera view for terrain-only rendering.")
    print("Viewer controls:")
    print("  ESC: quit")
    print("  V: toggle viewer sync")
    print("  R: reset camera to the selected preset")
    print("  P: save a screenshot to logs/terrain_renders/")


def show_terrains(args):
    validate_args(args)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    configure_env_for_terrain_view(env_cfg, args.use_task_layout)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if env.viewer is None:
        raise RuntimeError("Failed to create Isaac Gym viewer.")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "reset_camera")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "save_image")

    camera_position, camera_target = get_camera_pose(env_cfg, args.camera)
    env.set_camera(camera_position, camera_target)

    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, requires_grad=False
    )
    env.step(zero_actions)
    hide_robot(env)
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    print_controls(args)

    auto_save_path = os.path.abspath(args.save_image) if args.save_image else ""
    should_auto_save = bool(auto_save_path)
    frame_count = 0

    while not env.gym.query_viewer_has_closed(env.viewer):
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.value <= 0:
                continue
            if evt.action == "QUIT":
                return
            if evt.action == "toggle_viewer_sync":
                env.enable_viewer_sync = not env.enable_viewer_sync
            elif evt.action == "reset_camera":
                env.set_camera(camera_position, camera_target)
            elif evt.action == "save_image":
                save_viewer_image(env, get_default_screenshot_path(args.task))

        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        if env.enable_viewer_sync:
            env.gym.sync_frame_time(env.sim)
        else:
            env.gym.poll_viewer_events(env.viewer)

        if should_auto_save and frame_count >= args.frames_before_capture:
            save_viewer_image(env, auto_save_path)
            should_auto_save = False
            if args.quit_after_save:
                return

        frame_count += 1


if __name__ == "__main__":
    show_terrains(parse_args())
