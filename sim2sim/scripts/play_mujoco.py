"""Sim2sim entry point: run the trained policy inside MuJoCo.

Pipeline per policy step (100 Hz):
    1. read MuJoCo state -> base ang vel / projected gravity / joint pos,vel
    2. compute virtual-leg states (theta0, L0, rates) via wl_controller
    3. build the 27-dim observation, push into the 5-frame history buffer
    4. policy: latent = encoder(history); action = actor([obs, latent])
    5. controller: action -> 6 joint torques (VMC + PD)
    6. apply torques for DECIMATION physics steps (200 Hz)

Run from the repo root (the script also fixes sys.path to its own dir).
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 确保从任何工作目录运行时都能正确导入 policy.py 和 wl_controller.py
# 如果运行 python sim2sim/scripts/play_mujoco.py 那么 python 会自动加载 sim2sim/scripts/policy_mujoco.py 这个文件，
# 并在全局作用域中自动提供 __file__ = "sim2sim/scripts/play_mujoco.py"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import yaml

import wl_controller as ctrl
from policy import SequencePolicy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_XML = os.path.join(REPO_ROOT, "sim2sim/model/wheel_legged_v4.xml")
POLICY_PT = os.path.join(REPO_ROOT, "sim2sim/model/model_8000.pt")
CONFIG_PATH = Path(__file__).with_name("config") / "sim2sim.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

network = CONFIG["network"]
clipping = CONFIG["clipping"]
control_timing = CONFIG["control_timing"]
initial_state = CONFIG["initial_state"]
default_command = CONFIG["default_command"]
safety = CONFIG["safety"]
logging_config = CONFIG["logging"]

# MuJoCo qpos stores generalized positions:
# [base_x, base_y, base_z, base_qw, base_qx, base_qy, base_qz,
#  left_thigh_pos, left_leg_pos, left_wheel_pos, right_thigh_pos, right_leg_pos, right_wheel_pos].
# MuJoCo qvel stores generalized velocities:
# [base_vx, base_vy, base_vz, base_wx, base_wy, base_wz,
#  left_thigh_vel, left_leg_vel, left_wheel_vel, right_thigh_vel, right_leg_vel, right_wheel_vel].
#
# build_mjcf.py writes the six MJCF hinge joints in the same order as the URDF:
# [left_thigh, left_leg, left_wheel, right_thigh, right_leg, right_wheel].
# MuJoCo then stores qpos[7:] and qvel[6:] in this MJCF joint order, so these
# slices have the same DOF order as Isaac Gym.
QPOS_JOINT_START = 7
QVEL_JOINT_START = 6


# Purpose: 将世界坐标系下的三维向量旋转到机器人机体坐标系，用于构造与 Isaac Gym 训练侧一致的角速度和重力方向观测。
# Inputs: quat_wxyz 表示 MuJoCo 提供的基座姿态四元数，排列顺序是 [w, x, y, z]；vec 表示需要从世界坐标系变换到机体系的三维向量。
# Outputs: 返回旋转后的三维向量，其物理含义是同一个向量在 base_link 机体坐标系下的表达。
def quat_rotate_inverse_wxyz(quat_wxyz, vec):
    # 将 MuJoCo 四元数拆成标量部分和向量部分；这里的四元数顺序是 [w, x, y, z]，而训练侧工具函数内部使用的是标量和向量分开的公式。
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z])
    q_w = w
    # 使用四元数逆旋转的闭式公式计算结果，避免在部署循环中创建额外的旋转矩阵。
    a = vec * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, vec) * q_w * 2.0
    c = q_vec * (q_vec @ vec) * 2.0
    return a - b + c


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="run without viewer")
    p.add_argument("--seconds", type=float, default=60.0, help="sim duration")
    p.add_argument("--vx", type=float, default=0.0, help="forward velocity command [m/s]")
    p.add_argument("--wz", type=float, default=0.0, help="yaw rate command [rad/s]")
    p.add_argument("--height", type=float, default=default_command["height"], help="body height command [m]")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def reset_state(data):
    """Set base height and default joint angles, matching env init_state."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = initial_state["base_pos"]
    data.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion [w,x,y,z]
    data.qpos[QPOS_JOINT_START:] = initial_state["default_dof_pos"]


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    model.opt.timestep = control_timing["sim_dt"]
    # 加载 MJCF 模型，创建一个变量 data 用于存储仿真运行时数据
    data = mujoco.MjData(model)
    reset_state(data)
    # 根据 reset 后的 qpos 和 qvel 计算初始衍生状态（如各个 body 的世界坐标、geom的位置等）
    mujoco.mj_forward(model, data)

    policy = SequencePolicy(POLICY_PT, device=args.device)

    commands = np.array([args.vx, args.wz, args.height], dtype=np.float32)
    last_actions = np.zeros(network["num_actions"], dtype=np.float32)

    def read_observation():
        quat = data.qpos[3:7]                      # [w,x,y,z]
        world_ang_vel = data.qvel[3:6]             # base angular velocity (world)
        base_ang_vel = quat_rotate_inverse_wxyz(quat, world_ang_vel)
        projected_gravity = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))
        dof_pos = data.qpos[QPOS_JOINT_START:].copy()
        dof_vel = data.qvel[QVEL_JOINT_START:].copy()
        legs = ctrl.leg_states(dof_pos, dof_vel)
        obs = ctrl.build_observation(
            base_ang_vel, projected_gravity, commands, legs, dof_pos, dof_vel, last_actions
        )
        return np.clip(obs, -clipping["observations"], clipping["observations"]), legs, dof_vel

    obs, legs, dof_vel = read_observation()
    # History buffer filled with the first observation
    # (matches env reset, which repeats the first proprio obs across the whole history window).
    obs_history = np.tile(obs, network["obs_history_length"])

    n_policy_steps = int(args.seconds / (control_timing["sim_dt"] * control_timing["decimation"]))

    def run_loop(viewer=None):
        nonlocal obs, obs_history, last_actions, legs, dof_vel
        for step in range(n_policy_steps):
            if viewer is not None and not viewer.is_running():
                break
            t0 = time.time()

            action = policy.act(obs, obs_history)
            action = np.clip(action, -clipping["actions"], clipping["actions"])

            # Apply the same torque for DECIMATION physics steps.
            for _ in range(control_timing["decimation"]):
                dof_pos = data.qpos[QPOS_JOINT_START:].copy()
                dof_vel = data.qvel[QVEL_JOINT_START:].copy()
                legs = ctrl.leg_states(dof_pos, dof_vel)
                data.ctrl[:] = ctrl.compute_torques(action, legs, dof_vel)
                mujoco.mj_step(model, data)

            last_actions = action
            obs, legs, dof_vel = read_observation()
            # Slide history: drop oldest frame, append newest.
            obs_history = np.concatenate([obs_history[network["num_obs"]:], obs])

            if data.qpos[2] < safety["fall_reset_height"]:
                print(f"[step {step}] base fell (z={data.qpos[2]:.3f}), resetting")
                reset_state(data)
                mujoco.mj_forward(model, data)
                last_actions[:] = 0.0
                obs, legs, dof_vel = read_observation()
                obs_history = np.tile(obs, network["obs_history_length"])

            if viewer is not None:
                # 把当前仿真状态同步到 MuJoCo viewer 窗口里，让我们看到机器人当前位置、姿态和运动结果。
                viewer.sync()
                # Real-time pacing for a watchable viewer.
                # 示例：sim_dt=0.005 且 decimation=2 时，一个策略控制步代表 0.01 秒仿真时间。
                # 如果本轮循环中的计算实际耗时 0.003 秒，则暂停 0.007 秒，让 viewer 播放速度接近真实时间。
                dt = control_timing["sim_dt"] * control_timing["decimation"] - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
            elif step % logging_config["status_interval_steps"] == 0:
                print(f"[step {step}] z={data.qpos[2]:.3f} vx_cmd={commands[0]:.2f}")

    if args.headless:
        run_loop(None)
        print(f"Done. final base z={data.qpos[2]:.3f}")
    else:
        from mujoco import viewer as mj_viewer
        
        # 打开 MuJoCo viewer 窗口，把窗口对象命名为 viewer，然后运行 run_loop(viewer)；
        # 当这段代码结束后，自动关闭/清理 viewer 相关资源。
        with mj_viewer.launch_passive(model, data) as viewer:
            run_loop(viewer)


if __name__ == "__main__":
    main()
