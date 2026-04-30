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
        randomize_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        randomize_inertia = False
        randomize_base_com = False
        push_robots = False
        randomize_Kp = False
        randomize_Kd = False
        randomize_motor_torque = False
        randomize_default_dof_pos = False
        randomize_action_delay = False
        
    class control(WheelLeggedFYTCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0

        l0_offset = 0.255
        feedforward_force = 40.0  # [N]

        kp_theta = 50.0  # [N*m/rad]
        kd_theta = 3.0  # [N*m*s/rad]
        kp_l0 = 900.0  # [N/m]
        kd_l0 = 20.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"thigh": 0.0, "leg": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"thigh": 0.0, "leg": 0.0, "wheel": 0.5}  # [N*m*s/rad]

    class normalization(WheelLeggedFYTCfg.normalization):
        class obs_scales(WheelLeggedFYTCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25

    class noise(WheelLeggedFYTCfg.noise):
        class noise_scales(WheelLeggedFYTCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1


class WheelLeggedVMCFYTCfgPPO(WheelLeggedFYTCfgPPO):

    class algorithm(WheelLeggedFYTCfgPPO.algorithm):
        kl_decay = (
            WheelLeggedFYTCfgPPO.algorithm.desired_kl - 0.002
        ) / WheelLeggedFYTCfgPPO.runner.max_iterations

    class runner(WheelLeggedFYTCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc_fyt"
