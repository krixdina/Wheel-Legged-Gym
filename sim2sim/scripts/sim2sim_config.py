"""Load the frozen FYT wheel-legged sim2sim parameters from YAML.

The source of truth is ``config/sim2sim.yaml`` next to this file.  This module
keeps the existing uppercase constants as a compatibility layer for
``play_mujoco.py``, ``policy.py`` and ``wl_controller.py``.

Observation layout (num_obs = 27), matching
LeggedRobotVMC.compute_proprioception_observations():
    [ 0: 3] base_ang_vel    * ang_vel(0.25)
    [ 3: 6] projected_gravity (unit gravity direction in base frame)
    [ 6: 9] commands[vx, wz, h] * [lin_vel(2.0), ang_vel(0.25), height(5.0)]
    [ 9:11] theta0 (virtual leg swing angle, L/R) * dof_pos(1.0)
    [11:13] theta0_dot                           * dof_vel(0.05)
    [13:15] L0 (L/R)                              * l0(4.5)
    [15:17] L0_dot (L/R)                          * l0_dot(0.25)
    [17:19] wheel dof_pos (L/R)                   * dof_pos(1.0)
    [19:21] wheel dof_vel (L/R)                   * dof_vel(0.05)
    [21:27] last actions (6)
"""
import math
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).with_name("config") / "sim2sim.yaml"


def _load_config(path=CONFIG_PATH):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"sim2sim config must be a mapping: {path}")
    return data


def _section(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise KeyError(f"missing or invalid sim2sim config section: {name}")
    return value


CONFIG = _load_config()

_network = _section(CONFIG, "network")
_clipping = _section(CONFIG, "clipping")
_control_timing = _section(CONFIG, "control_timing")
_observation_scales = _section(CONFIG, "observation_scales")
_action_scales = _section(CONFIG, "action_scales")
_control = _section(CONFIG, "control")
_geometry = _section(CONFIG, "geometry")
_limits = _section(CONFIG, "limits")
_initial_state = _section(CONFIG, "initial_state")
_dof_indices = _section(CONFIG, "dof_indices")
_default_command = _section(CONFIG, "default_command")

# ----- network / observation dims (verified against model_8000.pt) -----
NUM_OBS = int(_network["num_obs"])
NUM_ACTIONS = int(_network["num_actions"])
OBS_HISTORY_LENGTH = int(_network["obs_history_length"])
NUM_ENCODER_OBS = int(_network["num_encoder_obs"])
LATENT_DIM = int(_network["latent_dim"])
ENCODER_HIDDEN_DIMS = list(_network["encoder_hidden_dims"])
ACTOR_HIDDEN_DIMS = list(_network["actor_hidden_dims"])
ACTIVATION = str(_network["activation"])

expected_encoder_obs = OBS_HISTORY_LENGTH * NUM_OBS
if NUM_ENCODER_OBS != expected_encoder_obs:
    raise ValueError(
        "num_encoder_obs must equal obs_history_length * num_obs "
        f"({NUM_ENCODER_OBS} != {expected_encoder_obs})"
    )

CLIP_OBSERVATIONS = float(_clipping["observations"])
CLIP_ACTIONS = float(_clipping["actions"])

# ----- control timing -----
SIM_DT = float(_control_timing["sim_dt"])
DECIMATION = int(_control_timing["decimation"])

# ----- observation scales -----
OBS_SCALE_LIN_VEL = float(_observation_scales["lin_vel"])
OBS_SCALE_ANG_VEL = float(_observation_scales["ang_vel"])
OBS_SCALE_DOF_POS = float(_observation_scales["dof_pos"])
OBS_SCALE_DOF_VEL = float(_observation_scales["dof_vel"])
OBS_SCALE_L0 = float(_observation_scales["l0"])
OBS_SCALE_L0_DOT = float(_observation_scales["l0_dot"])
OBS_SCALE_HEIGHT = float(_observation_scales["height"])
COMMANDS_SCALE = list(_observation_scales["commands_scale"])

# ----- action scales / VMC control law -----
ACTION_SCALE_THETA = float(_action_scales["theta"])
ACTION_SCALE_L0 = float(_action_scales["l0"])
ACTION_SCALE_VEL = float(_action_scales["vel"])
L0_OFFSET = float(_control["l0_offset"])
FEEDFORWARD_FORCE = float(_control["feedforward_force"])

KP_THETA = float(_control["kp_theta"])
KD_THETA = float(_control["kd_theta"])
KP_L0 = float(_control["kp_l0"])
KD_L0 = float(_control["kd_l0"])
WHEEL_KD = float(_control["wheel_kd"])

# ----- five-bar / VMC geometry -----
L1 = float(_geometry["l1"])
L2 = float(_geometry["l2"])
OFFSET = float(_geometry["offset"])

# ----- torque limits and default pose -----
TORQUE_LIMITS = list(_limits["torque_limits"])
DEFAULT_DOF_POS = list(_initial_state["default_dof_pos"])

PI = math.pi

# DOF index map within the 6-vector (matches Isaac Gym dof ordering).
LEFT_THIGH = int(_dof_indices["left_thigh"])
LEFT_LEG = int(_dof_indices["left_leg"])
LEFT_WHEEL = int(_dof_indices["left_wheel"])
RIGHT_THIGH = int(_dof_indices["right_thigh"])
RIGHT_LEG = int(_dof_indices["right_leg"])
RIGHT_WHEEL = int(_dof_indices["right_wheel"])
WHEEL_IDX = list(_dof_indices["wheel_idx"])

# ----- default command for interactive / scripted play -----
DEFAULT_HEIGHT = float(_default_command["height"])
