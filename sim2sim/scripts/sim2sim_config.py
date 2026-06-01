"""Frozen parameter set for the FYT wheel-legged sim2sim验证.

Every value here was cross-checked against the training-time config snapshot in
logs/.../May12_13-34-06_num-envs-4096_dr-true/ and against the actual layer
shapes of model_8000.pt, so the MuJoCo side reproduces exactly the观测/动作/控制
管线 the policy was trained with. Do not edit casually: changing any scale or
gain here silently breaks sim2sim fidelity.

Observation layout (num_obs = 27), matching
LeggedRobotVMC.compute_proprioception_observations():
    [ 0: 3] base_ang_vel    * ang_vel(0.25)
    [ 3: 6] projected_gravity (unit gravity direction in base frame)
    [ 6: 9] commands[vx, wz, h] * [lin_vel(2.0), ang_vel(0.25), height(5.0)]
    [ 9:11] theta0 (virtual leg swing angle, L/R) * dof_pos(1.0)
    [11:13] theta0_dot                           * dof_vel(0.05)
    [13:14] L0 (left virtual leg length)         * l0(4.5)
    [14:15] L0 (right)                            * l0(4.5)   -> stored as 2 dims
    [15:17] L0_dot (L/R)                          * l0_dot(0.25)
    [17:19] wheel dof_pos (L/R)                   * dof_pos(1.0)
    [19:21] wheel dof_vel (L/R)                   * dof_vel(0.05)
    [21:27] last actions (6)
Note: L0 and L0_dot are 2-vectors (left,right); the slice numbers above collapse
them for readability. The concatenation order in code is the source of truth.
"""
import math

# ----- network / observation dims (verified against model_8000.pt) -----
NUM_OBS = 27
NUM_ACTIONS = 6
OBS_HISTORY_LENGTH = 5
NUM_ENCODER_OBS = OBS_HISTORY_LENGTH * NUM_OBS  # 135
LATENT_DIM = 3
ENCODER_HIDDEN_DIMS = [128, 64]
ACTOR_HIDDEN_DIMS = [128, 64, 32]
ACTIVATION = "elu"

CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 100.0

# ----- control timing -----
SIM_DT = 0.005          # MuJoCo physics step
DECIMATION = 2          # policy runs every DECIMATION physics steps -> 100 Hz

# ----- observation scales -----
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
OBS_SCALE_L0 = 4.5
OBS_SCALE_L0_DOT = 0.25
OBS_SCALE_HEIGHT = 5.0  # commands[2] (height) uses height_measurements scale
# commands_scale = [lin_vel, ang_vel, height_measurements]
COMMANDS_SCALE = [OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL, OBS_SCALE_HEIGHT]

# ----- action scales / VMC control law (wheel_legged_vmc_fyt_config snapshot) -----
ACTION_SCALE_THETA = 0.6
ACTION_SCALE_L0 = 0.1
ACTION_SCALE_VEL = 6.0
L0_OFFSET = 0.2
FEEDFORWARD_FORCE = 80.0   # [N]

KP_THETA = 80.0   # [N*m/rad]
KD_THETA = 4.0    # [N*m*s/rad]
KP_L0 = 900.0     # [N/m]
KD_L0 = 45.0      # [N/m*s]
WHEEL_KD = 0.25   # wheel velocity PD damping (damping["wheel"])

# ----- five-bar / VMC geometry (asset, offset=0 -> closed-form Jacobian) -----
L1 = 0.21
L2 = 0.25
OFFSET = 0.0

# ----- torque limits, from URDF joint effort (thigh/leg 30, wheel 4.5) -----
# Order: [left_thigh, left_leg, left_wheel, right_thigh, right_leg, right_wheel]
TORQUE_LIMITS = [30.0, 30.0, 4.5, 30.0, 30.0, 4.5]

# ----- default joint angles (init_state.default_joint_angles), DOF order -----
DEFAULT_DOF_POS = [-0.4, -0.6, 0.0, 0.4, 0.6, 0.0]

PI = math.pi

# DOF index map within the 6-vector (matches Isaac Gym dof ordering).
LEFT_THIGH, LEFT_LEG, LEFT_WHEEL = 0, 1, 2
RIGHT_THIGH, RIGHT_LEG, RIGHT_WHEEL = 3, 4, 5
WHEEL_IDX = [LEFT_WHEEL, RIGHT_WHEEL]

# ----- default command for interactive / scripted play -----
DEFAULT_HEIGHT = 0.18  # within commands.ranges.height [0.1, 0.2]
