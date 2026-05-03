from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class WheelLeggedFYTCfg(LeggedRobotCfg):
    # FYT v4 keeps the original task/control values and swaps in the updated
    # wheel_legged_fyt robot asset plus name-based joint/contact configuration.

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "left_thigh_joint": -0.4,
            "left_leg_joint": -0.6,
            "left_wheel_joint": 0.0,
            "right_thigh_joint": 0.4,
            "right_leg_joint": 0.6,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"thigh": 40.0, "leg": 40.0, "wheel": 0}  # [N*m/rad]
        damping = {"thigh": 1.0, "leg": 1.0, "wheel": 0.5}  # [N*m*s/rad]

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            height = [0.1, 0.2]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged_fyt/urdf/wheel_legged_v4_isaac.urdf"
        name = "WheelLeggedFYT"
        offset = 0.0
        l1 = 0.21
        l2 = 0.25
        # 这里以子串的方式匹配机器人所有 link 中需要添加接触惩罚的link。
        penalize_contacts_on = [
            "left_thigh",
            "left_leg",
            "right_thigh",
            "right_leg",
            "base",
        ]
        # base 接触地面或障碍视为失败终止；腿部接触只扣 collision 奖励，不直接终止。
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # 仅影响训练时模型的可视化效果，不影响仿真物理碰撞。
        flip_visual_attachments = False


class WheelLeggedFYTCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_fyt"
