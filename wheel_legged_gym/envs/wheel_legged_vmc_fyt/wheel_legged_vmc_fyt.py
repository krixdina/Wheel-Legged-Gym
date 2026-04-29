import torch

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC


class LeggedRobotVMCFYT(LeggedRobotVMC):
    """FYT 轮腿构型专用 VMC 几何适配层。

    这里使用右腿参考的 VMC 平面坐标系：
    - VMC +x 指向 base_link 的 -x 方向；
    - VMC +y 指向竖直向下方向；
    - 右腿 URDF 关节正方向与该 VMC 坐标系一致，左腿是镜像侧，因此左腿角度和力矩需要取负映射。
    """

    def leg_post_physics_step(self):
        # 将 Isaac Gym 返回的 URDF 原始关节角 q 映射到右腿参考的 VMC 坐标系下，记为 theta。
        # theta1 表示第一段连杆在 VMC 平面中的绝对方向角；
        # theta2 表示第二段连杆相对第一段连杆的转角，使 theta1 + theta2 为第二段连杆方向角。
        self.theta1 = torch.cat(
            (
                (-self.dof_pos[:, 0]).unsqueeze(1),
                self.dof_pos[:, 3].unsqueeze(1),
            ),
            dim=1,
        )
        self.theta2 = torch.cat(
            (
                (-self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )

        # 因为速度项需要转到 VMC 坐标系下参与 theta0_dot 的计算
        # 所以速度项必须使用与 theta1/theta2 完全一致的符号约定,
        # pi/2 是常量偏置，对速度求导后为 0，因此这里只需要对左腿速度取负。
        theta1_dot = torch.cat(
            (
                (-self.dof_vel[:, 0]).unsqueeze(1),
                self.dof_vel[:, 3].unsqueeze(1),
            ),
            dim=1,
        )
        theta2_dot = torch.cat(
            (
                (-self.dof_vel[:, 1]).unsqueeze(1),
                self.dof_vel[:, 4].unsqueeze(1),
            ),
            dim=1,
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )

        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt

    def forward_kinematics(self, theta1, theta2):
        # 在右腿参考 VMC 坐标系中，FYT 退化为标准平面二连杆：
        # 第一段方向为 theta1，第二段方向为 theta1 + theta2。
        phi = theta1 + theta2
        end_x = (
            self.cfg.asset.offset
            + self.cfg.asset.l1 * torch.cos(theta1)
            + self.cfg.asset.l2 * torch.cos(phi)
        )
        end_y = (self.cfg.asset.l1 * torch.sin(theta1) 
            + self.cfg.asset.l2 * torch.sin(phi))

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

        # 将 VMC 坐标系下的力矩映射回 URDF 的 6 个 DOF。
        # theta1 = [-q0, q3]，theta2 = [pi/2 - q1, pi/2 + q4]；
        # 因此左侧腿部关节力矩取负，右侧腿部关节力矩保持正号。
        torques = torch.cat(
            (
                -T1[:, 0].unsqueeze(1),
                -T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                T1[:, 1].unsqueeze(1),
                T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def VMC(self, F, T):
        # theta0 不加 pi/2?
        phi = self.theta1 + self.theta2
        x = (
            self.cfg.asset.offset
            + self.cfg.asset.l1 * torch.cos(self.theta1)
            + self.cfg.asset.l2 * torch.cos(phi)
        )
        y = self.cfg.asset.l1 * torch.sin(self.theta1) + self.cfg.asset.l2 * torch.sin(
            phi
        )

        dx_dtheta1 = -self.cfg.asset.l1 * torch.sin(
            self.theta1
        ) - self.cfg.asset.l2 * torch.sin(phi)
        dy_dtheta1 = self.cfg.asset.l1 * torch.cos(
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
