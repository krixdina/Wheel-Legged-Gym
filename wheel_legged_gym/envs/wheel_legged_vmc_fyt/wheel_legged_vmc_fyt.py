import torch

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC


class LeggedRobotVMCFYT(LeggedRobotVMC):
    """FYT 机器人专用的 VMC 几何适配层。

    FYT 与原机器人拓扑一致，但腿部连杆在 URDF 中的几何展开方向不同：
    thigh -> leg 这一段沿局部 -x 方向，而原项目按 +x 方向建模。
    因此这里只覆盖 VMC 相关的几何量计算和力矩映射，不修改原始 LeggedRobotVMC。
    """

    def leg_post_physics_step(self):
        # 将 Isaac Gym 返回的左右腿 URDF 关节角转换到统一的平面二连杆坐标系。
        # FYT 已将左右 leg_joint 的 axis 调整为 0 0 1，因此小腿角速度/角度不再需要额外反号；
        # 右大腿仍按原项目约定取负号，用来把镜像安装的左右腿统一到同一几何坐标系。
        self.theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        self.theta2 = torch.cat(
            (
                (self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )

        theta1_dot = torch.cat(
            (self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta2_dot = torch.cat(
            (self.dof_vel[:, 1].unsqueeze(1), self.dof_vel[:, 4].unsqueeze(1)), dim=1
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )

        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt

    def forward_kinematics(self, theta1, theta2):
        # FYT 的第一段腿从大腿关节指向小腿关节时沿局部 -x 方向；
        # 第二段仍用 theta1 + theta2 表示驱动轮相对髋部的末端方向。
        phi = theta1 + theta2
        end_x = (
            self.cfg.asset.offset
            - self.cfg.asset.l1 * torch.cos(theta1)
            + self.cfg.asset.l2 * torch.cos(phi)
        )
        end_y = -self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            phi
        )

        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.arctan2(end_y, end_x) - self.pi / 2
        return L0, theta0

    def _compute_torques(self, actions):
        # action 仍沿用原 VMC 定义：
        # [左腿摆角, 左腿长, 左轮速, 右腿摆角, 右腿长, 右轮速] 的无量纲偏移。
        theta0_ref = (
            torch.cat(
                (
                    (actions[:, 0]).unsqueeze(1),
                    (actions[:, 3]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_theta
        )
        l0_ref = (
            torch.cat(
                (
                    (actions[:, 1]).unsqueeze(1),
                    (actions[:, 4]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_l0
        ) + self.cfg.control.l0_offset
        wheel_vel_ref = (
            torch.cat(
                (
                    (actions[:, 2]).unsqueeze(1),
                    (actions[:, 5]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )

        self.torque_leg = (
            self.theta_kp * (theta0_ref - self.theta0) - self.theta_kd * self.theta0_dot
        )
        self.force_leg = self.l0_kp * (l0_ref - self.L0) - self.l0_kd * self.L0_dot
        self.torque_wheel = self.d_gains[:, [2, 5]] * (
            wheel_vel_ref - self.dof_vel[:, [2, 5]]
        )
        T1, T2 = self.VMC(
            self.force_leg + self.cfg.control.feedforward_force, self.torque_leg
        )

        # 将 VMC 几何坐标系下的广义力矩映射回 FYT URDF 的 6 个 DOF。
        # theta1 = [q0, -q3]，theta2 = [q1 + pi/2, q4 + pi/2]；
        # 因此右大腿力矩需要取负，小腿两侧不再因 axis 反向而取负。
        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                -T1[:, 1].unsqueeze(1),
                T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def VMC(self, F, T):
        # 根据 FYT forward_kinematics 对 L0 和 theta0 求雅可比。
        # 使用虚功关系 tau = F * dL/dq + T * dtheta0/dq，
        # 保证前向运动学和虚拟力到关节力矩的映射使用同一套几何约定。
        phi = self.theta1 + self.theta2
        x = (
            self.cfg.asset.offset
            - self.cfg.asset.l1 * torch.cos(self.theta1)
            + self.cfg.asset.l2 * torch.cos(phi)
        )
        y = -self.cfg.asset.l1 * torch.sin(self.theta1) + self.cfg.asset.l2 * torch.sin(
            phi
        )

        dx_dtheta1 = self.cfg.asset.l1 * torch.sin(
            self.theta1
        ) - self.cfg.asset.l2 * torch.sin(phi)
        dy_dtheta1 = -self.cfg.asset.l1 * torch.cos(
            self.theta1
        ) + self.cfg.asset.l2 * torch.cos(phi)
        dx_dtheta2 = -self.cfg.asset.l2 * torch.sin(phi)
        dy_dtheta2 = self.cfg.asset.l2 * torch.cos(phi)

        L0_sq = self.L0**2
        dL_dtheta1 = (x * dx_dtheta1 + y * dy_dtheta1) / self.L0
        dL_dtheta2 = (x * dx_dtheta2 + y * dy_dtheta2) / self.L0
        dtheta0_dtheta1 = (x * dy_dtheta1 - y * dx_dtheta1) / L0_sq
        dtheta0_dtheta2 = (x * dy_dtheta2 - y * dx_dtheta2) / L0_sq

        T1 = dL_dtheta1 * F + dtheta0_dtheta1 * T
        T2 = dL_dtheta2 * F + dtheta0_dtheta2 * T

        return T1, T2
