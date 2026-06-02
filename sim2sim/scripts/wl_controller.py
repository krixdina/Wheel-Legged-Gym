"""VMC geometry + control law for the FYT wheel-legged robot, ported to numpy.

This is a faithful single-environment port of LeggedRobotVMCFYT
(wheel_legged_vmc_fyt.py) and the observation builder in
LeggedRobotVMC.compute_proprioception_observations(). Keeping it byte-for-byte
equivalent to the training code is what makes sim2sim meaningful, so the sign
conventions and the closed-form VMC Jacobian are reproduced exactly.

Right-leg-referenced VMC plane (see LeggedRobotVMCFYT docstring):
    theta1 = [-q_left_thigh,  q_right_thigh]
    theta2 = [-q_left_leg + pi/2,  q_right_leg + pi/2]
The left leg is mirrored, hence its angles/torques are negated on the way in
and out of the VMC frame.
"""
import math
from pathlib import Path

import numpy as np
import yaml


CONFIG_PATH = Path(__file__).with_name("config") / "sim2sim.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

network = CONFIG["network"]
observation_scales = CONFIG["observation_scales"]
action_scales = CONFIG["action_scales"]
control = CONFIG["control"]
geometry = CONFIG["geometry"]
limits = CONFIG["limits"]
dof_indices = CONFIG["dof_indices"]
kinematics = CONFIG["kinematics"]

fk_diff_dt = kinematics["fk_diff_dt"]  # finite-difference step for L0_dot / theta0_dot (matches env)


def forward_kinematics(theta1, theta2):
    """Two-link planar FK in the VMC frame. theta1/theta2 are length-2 (L,R).

    Returns (L0, theta0), each length-2.
    """
    phi = theta1 + theta2
    end_x = geometry["offset"] + geometry["l1"] * np.cos(theta1) + geometry["l2"] * np.cos(phi)
    end_y = geometry["l1"] * np.sin(theta1) + geometry["l2"] * np.sin(phi)
    L0 = np.sqrt(end_x ** 2 + end_y ** 2)
    theta0 = np.arctan2(end_y, end_x) - math.pi / 2.0
    return L0, theta0


def leg_states(dof_pos, dof_vel):
    """Map the 6 raw joint pos/vel to virtual-leg states (theta0, L0 and rates).

    dof_pos / dof_vel: length-6 arrays in Isaac Gym DOF order.
    Returns dict with theta0, theta0_dot, L0, L0_dot (each length-2, L/R).
    """
    theta1 = np.array([-dof_pos[dof_indices["left_thigh"]], dof_pos[dof_indices["right_thigh"]]])
    theta2 = np.array(
        [
            -dof_pos[dof_indices["left_leg"]] + math.pi / 2.0,
            dof_pos[dof_indices["right_leg"]] + math.pi / 2.0,
        ]
    )
    # Velocity sign convention matches theta1/theta2; the pi/2 offset drops out
    # under differentiation, so only the left side is negated.
    theta1_dot = np.array(
        [-dof_vel[dof_indices["left_thigh"]], dof_vel[dof_indices["right_thigh"]]]
    )
    theta2_dot = np.array(
        [-dof_vel[dof_indices["left_leg"]], dof_vel[dof_indices["right_leg"]]]
    )

    L0, theta0 = forward_kinematics(theta1, theta2)
    L0_next, theta0_next = forward_kinematics(
        theta1 + theta1_dot * fk_diff_dt, theta2 + theta2_dot * fk_diff_dt
    )
    L0_dot = (L0_next - L0) / fk_diff_dt
    theta0_dot = (theta0_next - theta0) / fk_diff_dt
    return {
        "theta1": theta1,
        "theta2": theta2,
        "theta0": theta0,
        "theta0_dot": theta0_dot,
        "L0": L0,
        "L0_dot": L0_dot,
    }


def _vmc(F, T, L0, theta2):
    """Closed-form virtual-model-control map (offset=0): [T1,T2] from [F,T].

    [[T1],[T2]] = [[0, 1],[J21, J22]] [[F],[T]], applied per leg (length-2).
    """
    L0_sq = L0 ** 2
    J21 = -(geometry["l1"] * geometry["l2"] * np.sin(theta2) / L0)
    J22 = geometry["l2"] * (geometry["l1"] * np.cos(theta2) + geometry["l2"]) / L0_sq
    T1 = T
    T2 = J21 * F + J22 * T
    return T1, T2


def compute_torques(actions, legs, dof_vel):
    """Reproduce LeggedRobotVMCFYT._compute_torques for a single environment.

    actions: length-6 policy output [Lθ, Ll0, Lwheel, Rθ, Rl0, Rwheel].
    legs:    dict from leg_states().
    dof_vel: length-6 raw joint velocities (for wheel velocity tracking).
    Returns length-6 joint torques in Isaac Gym DOF order.
    """
    theta0_ref = np.array([actions[0], actions[3]]) * action_scales["theta"]
    l0_ref = np.array([actions[1], actions[4]]) * action_scales["l0"] + control["l0_offset"]
    wheel_vel_ref = np.array([actions[2], actions[5]]) * action_scales["vel"]

    torque_leg = control["kp_theta"] * (theta0_ref - legs["theta0"]) - control["kd_theta"] * legs["theta0_dot"]
    force_leg = control["kp_l0"] * (l0_ref - legs["L0"]) - control["kd_l0"] * legs["L0_dot"]
    torque_wheel = control["wheel_kd"] * (
        wheel_vel_ref - np.array([dof_vel[dof_indices["left_wheel"]], dof_vel[dof_indices["right_wheel"]]])
    )

    t1, t2 = _vmc(force_leg + control["feedforward_force"], torque_leg, legs["L0"], legs["theta2"])

    # Map VMC-frame torques back to the 6 URDF DOFs; left leg torques negated.
    torques = np.array(
        [
            -t1[0],
            -t2[0],
            torque_wheel[0],
            t1[1],
            t2[1],
            torque_wheel[1],
        ]
    )
    return np.clip(torques, -np.array(limits["torque_limits"]), np.array(limits["torque_limits"]))


def build_observation(base_ang_vel, projected_gravity, commands, legs, dof_pos, dof_vel, last_actions):
    """Assemble the 27-dim proprioceptive observation (already scaled, unclipped).

    Order matches LeggedRobotVMC.compute_proprioception_observations().
    base_ang_vel:      length-3 body angular velocity [rad/s]
    projected_gravity: length-3 gravity unit vector in base frame
    commands:          length-3 [vx, wz, height]
    legs:              dict from leg_states()
    dof_pos/dof_vel:   length-6 raw joint state (wheel pos/vel used)
    last_actions:      length-6 previous policy action
    """
    obs = np.concatenate(
        [
            base_ang_vel * observation_scales["ang_vel"],
            projected_gravity,
            commands * np.array(observation_scales["commands_scale"]),
            legs["theta0"] * observation_scales["dof_pos"],
            legs["theta0_dot"] * observation_scales["dof_vel"],
            legs["L0"] * observation_scales["l0"],
            legs["L0_dot"] * observation_scales["l0_dot"],
            np.array([dof_pos[dof_indices["left_wheel"]], dof_pos[dof_indices["right_wheel"]]])
            * observation_scales["dof_pos"],
            np.array([dof_vel[dof_indices["left_wheel"]], dof_vel[dof_indices["right_wheel"]]])
            * observation_scales["dof_vel"],
            last_actions,
        ]
    )
    return obs.astype(np.float32)
