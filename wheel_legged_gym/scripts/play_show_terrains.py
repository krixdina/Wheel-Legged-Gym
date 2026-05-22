import os
from datetime import datetime

import numpy as np

import isaacgym
from isaacgym import gymapi, gymutil

import wheel_legged_gym.envs  # noqa: F401
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from wheel_legged_gym.utils.task_registry import task_registry
from wheel_legged_gym.utils.terrain import Terrain


DEFAULT_TASK = "wheel_legged_vmc_fyt"
DEFAULT_SCREENSHOT_DIR = os.path.join(
    WHEEL_LEGGED_GYM_ROOT_DIR, "logs", "terrain_renders"
)


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": DEFAULT_TASK,
            "help": "Registered task whose terrain configuration will be displayed.",
        },
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
            "default": 5,
            "help": "Number of rendered frames before the automatic screenshot is saved.",
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
    ]
    args = gymutil.parse_arguments(
        description="Show Isaac Gym terrains without robots.",
        custom_parameters=custom_parameters,
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


def get_env_cfg(task_name):
    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    if env_cfg.terrain.mesh_type is None:
        raise ValueError(f"Task '{task_name}' does not define a terrain mesh.")
    return env_cfg


def configure_terrain_layout(env_cfg, use_task_layout):
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"] and not use_task_layout:
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.selected = False


def create_sim(gym, args, env_cfg):
    sim_cfg = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_cfg)
    graphics_device_id = getattr(args, "graphics_device_id", args.sim_device_id)
    sim = gym.create_sim(
        args.sim_device_id,
        graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation.")
    return sim


def build_terrain(env_cfg):
    if env_cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
        return None
    num_tiles = env_cfg.terrain.num_rows * env_cfg.terrain.num_cols
    return Terrain(env_cfg.terrain, num_tiles)


def add_plane(gym, sim, env_cfg):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = env_cfg.terrain.static_friction
    plane_params.dynamic_friction = env_cfg.terrain.dynamic_friction
    plane_params.restitution = env_cfg.terrain.restitution
    gym.add_ground(sim, plane_params)


def add_heightfield(gym, sim, env_cfg, terrain):
    hf_params = gymapi.HeightFieldParams()
    hf_params.column_scale = terrain.cfg.horizontal_scale
    hf_params.row_scale = terrain.cfg.horizontal_scale
    hf_params.vertical_scale = terrain.cfg.vertical_scale
    hf_params.nbRows = terrain.tot_cols
    hf_params.nbColumns = terrain.tot_rows
    hf_params.transform.p.x = -terrain.cfg.border_size
    hf_params.transform.p.y = -terrain.cfg.border_size
    hf_params.transform.p.z = 0.0
    hf_params.static_friction = env_cfg.terrain.static_friction
    hf_params.dynamic_friction = env_cfg.terrain.dynamic_friction
    hf_params.restitution = env_cfg.terrain.restitution
    gym.add_heightfield(sim, terrain.heightsamples, hf_params)


def add_trimesh(gym, sim, env_cfg, terrain):
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]
    tm_params.transform.p.x = -terrain.cfg.border_size
    tm_params.transform.p.y = -terrain.cfg.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = env_cfg.terrain.static_friction
    tm_params.dynamic_friction = env_cfg.terrain.dynamic_friction
    tm_params.restitution = env_cfg.terrain.restitution
    gym.add_triangle_mesh(
        sim,
        terrain.vertices.flatten(order="C"),
        terrain.triangles.flatten(order="C"),
        tm_params,
    )


def add_terrain_to_sim(gym, sim, env_cfg, terrain):
    mesh_type = env_cfg.terrain.mesh_type
    if mesh_type == "plane":
        add_plane(gym, sim, env_cfg)
    elif mesh_type == "heightfield":
        add_heightfield(gym, sim, env_cfg, terrain)
    elif mesh_type == "trimesh":
        add_trimesh(gym, sim, env_cfg, terrain)
    else:
        raise ValueError(
            "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]."
        )


def get_camera_pose(env_cfg, terrain, preset):
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"] and terrain is not None:
        total_length = env_cfg.terrain.num_rows * env_cfg.terrain.terrain_length
        total_width = env_cfg.terrain.num_cols * env_cfg.terrain.terrain_width
        center = np.array([total_length / 2.0, total_width / 2.0, 0.0], dtype=np.float64)
        max_dim = max(total_length, total_width)
    else:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        max_dim = 20.0

    if preset == "top":
        position = center + np.array([0.0, 0.0, max_dim * 1.25 + 6.0], dtype=np.float64)
    else:
        position = center + np.array(
            [max_dim * 0.45, -max_dim * 0.95, max_dim * 0.70 + 4.0],
            dtype=np.float64,
        )
    return position, center


def set_camera(gym, viewer, position, target):
    cam_pos = gymapi.Vec3(position[0], position[1], position[2])
    cam_target = gymapi.Vec3(target[0], target[1], target[2])
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def get_default_screenshot_path(task_name):
    os.makedirs(DEFAULT_SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_SCREENSHOT_DIR, f"{task_name}_{timestamp}.png")


def save_viewer_image(gym, viewer, path):
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    gym.write_viewer_image_to_file(viewer, abs_path)
    print(f"Saved terrain screenshot to: {abs_path}")


def print_controls(args):
    print(f"Showing terrain for task: {args.task}")
    print("No robot actors are created in this preview.")
    print("Viewer controls:")
    print("  ESC: quit")
    print("  V: toggle viewer sync")
    print("  R: reset camera to the selected preset")
    print("  P: save a screenshot to logs/terrain_renders/")


def show_terrains(args):
    validate_args(args)
    env_cfg = get_env_cfg(args.task)
    configure_terrain_layout(env_cfg, args.use_task_layout)
    set_seed(env_cfg.seed)

    gym = gymapi.acquire_gym()
    sim = create_sim(gym, args, env_cfg)
    viewer = None

    try:
        terrain = build_terrain(env_cfg)
        add_terrain_to_sim(gym, sim, env_cfg, terrain)
        gym.prepare_sim(sim)

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer.")

        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_camera")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "save_image")

        camera_position, camera_target = get_camera_pose(env_cfg, terrain, args.camera)
        set_camera(gym, viewer, camera_position, camera_target)
        print_controls(args)

        auto_save_path = os.path.abspath(args.save_image) if args.save_image else ""
        should_auto_save = bool(auto_save_path)
        viewer_sync = True
        frame_count = 0

        while not gym.query_viewer_has_closed(viewer):
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value <= 0:
                    continue
                if evt.action == "QUIT":
                    return
                if evt.action == "toggle_viewer_sync":
                    viewer_sync = not viewer_sync
                elif evt.action == "reset_camera":
                    set_camera(gym, viewer, camera_position, camera_target)
                elif evt.action == "save_image":
                    save_viewer_image(gym, viewer, get_default_screenshot_path(args.task))

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            if viewer_sync:
                gym.sync_frame_time(sim)
            else:
                gym.poll_viewer_events(viewer)

            if should_auto_save and frame_count >= args.frames_before_capture:
                save_viewer_image(gym, viewer, auto_save_path)
                should_auto_save = False
                if args.quit_after_save:
                    return

            frame_count += 1
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    show_terrains(get_args())
