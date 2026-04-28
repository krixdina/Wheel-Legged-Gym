import torch

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC


class LeggedRobotVMCFYT(LeggedRobotVMC):
    """FYT-specific VMC geometry adapter.

    The FYT URDF keeps the same 6-DOF topology as the original robot, but its
    lower-leg joints rotate around -z and the thigh-to-leg joint offset points
    along -x. This class adapts the VMC kinematics and torque mapping without
    changing the original URDF or the upstream LeggedRobotVMC implementation.
    """

    def leg_post_physics_step(self):
        # Unify left/right thigh angles into one planar VMC convention.
        self.theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )

        # FYT lower-leg joints use axis="0 0 -1", so positive URDF joint motion
        # corresponds to negative planar rotation in the VMC convention.
        self.theta2 = torch.cat(
            (
                (-self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (-self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )

        theta1_dot = torch.cat(
            (self.dof_vel[:, 0].unsqueeze(1), -self.dof_vel[:, 3].unsqueeze(1)), dim=1
        )
        theta2_dot = torch.cat(
            (-self.dof_vel[:, 1].unsqueeze(1), -self.dof_vel[:, 4].unsqueeze(1)), dim=1
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )

        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt

    def forward_kinematics(self, theta1, theta2):
        # FYT thigh-to-leg joint offset points along -x, while the wheel offset
        # still forms the second segment after the lower-leg joint rotation.
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

        # Map VMC generalized torques back to FYT URDF DOF torque signs:
        # theta1 = [q0, -q3], theta2 = [-q1 + pi/2, -q4 + pi/2].
        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                -T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                -T1[:, 1].unsqueeze(1),
                -T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def VMC(self, F, T):
        # Virtual work mapping for the FYT forward kinematics:
        # tau_q = F * dL/dq + T * dtheta0/dq.
        phi = self.theta1 + self.theta2
        x = (
            self.cfg.asset.offset
            - self.cfg.asset.l1 * torch.cos(self.theta1)
            + self.cfg.asset.l2 * torch.cos(phi)
        )
        y = -self.cfg.asset.l1 * torch.sin(self.theta1) + self.cfg.asset.l2 * torch.sin(
            phi
        )

        dx_dtheta1 = self.cfg.asset.l1 * torch.sin(self.theta1) - self.cfg.asset.l2 * torch.sin(
            phi
        )
        dy_dtheta1 = -self.cfg.asset.l1 * torch.cos(self.theta1) + self.cfg.asset.l2 * torch.cos(
            phi
        )
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
