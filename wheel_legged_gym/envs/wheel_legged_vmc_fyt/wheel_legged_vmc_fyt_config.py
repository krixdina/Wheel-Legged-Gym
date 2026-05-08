from wheel_legged_gym.envs.wheel_legged_fyt.wheel_legged_fyt_config import (
    WheelLeggedFYTCfg,
    WheelLeggedFYTCfgPPO,
)


class WheelLeggedVMCFYTCfg(WheelLeggedFYTCfg):
    class env(WheelLeggedFYTCfg.env):
        num_privileged_obs = (
            WheelLeggedFYTCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3
        )

    class domain_rand(WheelLeggedFYTCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = True
        randomize_Kd_range = [0.9, 1.1]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 10]
        
    class control(WheelLeggedFYTCfg.control):
        action_scale_theta = 0.6
        action_scale_l0 = 0.1
        action_scale_vel = 6.0

        l0_offset = 0.22
        feedforward_force = 80.0  # [N]

        kp_theta = 80.0  # [N*m/rad]
        kd_theta = 4.0  # [N*m*s/rad]
        kp_l0 = 900.0  # [N/m]
        kd_l0 = 45.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"thigh": 0.0, "leg": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"thigh": 0.0, "leg": 0.0, "wheel": 0.25}  # [N*m*s/rad]

    class normalization(WheelLeggedFYTCfg.normalization):
        class obs_scales(WheelLeggedFYTCfg.normalization.obs_scales):
            l0 = 4.5
            l0_dot = 0.25

    class noise(WheelLeggedFYTCfg.noise):
        class noise_scales(WheelLeggedFYTCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1
    
    class rewards(WheelLeggedFYTCfg.rewards):
        class scales(WheelLeggedFYTCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0

            base_height = 1.0
            base_height_enhance = 1.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -10.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1.0

class WheelLeggedVMCFYTCfgPPO(WheelLeggedFYTCfgPPO):

    class algorithm(WheelLeggedFYTCfgPPO.algorithm):
        kl_decay = (
            WheelLeggedFYTCfgPPO.algorithm.desired_kl - 0.002
        ) / WheelLeggedFYTCfgPPO.runner.max_iterations

    class runner(WheelLeggedFYTCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc_fyt"
