"""sim2real policy-I/O controller for the wheel-legged robot (NUC side).

Unlike the sim2sim controller, this one does NOT run forward kinematics or the
VMC/PD torque law: the lower machine already computes the virtual-leg states
(theta0/L0 and rates) and the gravity projection and sends them up raw, and the
lower machine also turns the 6-dim action into joint torques. This controller is
purely the NUC-side policy interface:

    - scale the 21 raw uplink states into the 27-dim observation (last_action appended)
    - clip the observation,
    - maintain the 5-frame history window the encoder consumes,
    - remember the last action that was sent (it is fed back into the next obs),
    - scale the raw policy action into a physical command (theta0_ref / l0_ref /
      wheel_vel_ref) for the downlink -- the action scaling that used to run on
      the lower machine now happens here on the NUC.

All scales / dims come from the shared config.
"""
import numpy as np

from config import CONFIG


class InvalidUplinkState(ValueError):
    """Raised when a decoded uplink frame is not a physically plausible state."""


class Sim2RealController:
    """Build (obs, obs_history) from raw uplink state; track history + last action."""

    def __init__(self, config=CONFIG):
        self._scales = config["observation_scales"]
        self._commands_scale = np.array(self._scales["commands_scale"], dtype=np.float32)
        self._clip_obs = config["clipping"]["observations"]
        network = config["network"]
        self._num_obs = network["num_obs"]
        self._num_actions = network["num_actions"]
        self._history_length = network["obs_history_length"]
        # Action scaling (raw policy output -> physical command). Per-element gain
        # and offset for the 6-dim action [L_theta, L_l0, L_wheel, R_theta, R_l0,
        # R_wheel]; only the l0 entries carry the l0_offset, matching sim2sim.
        a = config["action_scales"]
        self._action_scale = np.array(
            [a["theta"], a["l0"], a["vel"], a["theta"], a["l0"], a["vel"]], dtype=np.float32
        )
        self._action_offset = np.array(
            [0.0, a["l0_offset"], 0.0, 0.0, a["l0_offset"], 0.0], dtype=np.float32
        )
        # last_action starts at zero (matches the env reset); 
        # _obs_history is lazily filled on the first observation so it repeats the first frame.
        self._last_action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs_history = None

    @staticmethod
    def _validate_state(state):
        """Reject garbage frames before scaling them into the policy observation.

        With CRC disabled, a binary payload byte can occasionally look like SOF
        during resync. The frame length/EOF check may then accept a false frame,
        whose floats can be NaN/Inf or wildly outside the robot's physical range.
        Dropping those frames keeps bad serial sync from contaminating policy IO.
        """
        for name, values in state.items():
            arr = np.asarray(values, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                raise InvalidUplinkState(f"{name} contains non-finite values: {arr}")

        gravity_norm = float(np.linalg.norm(state["projected_gravity"]))
        if not 0.5 <= gravity_norm <= 1.5:
            raise InvalidUplinkState(f"projected_gravity norm {gravity_norm:.3g} is not near 1")

        l0 = np.asarray(state["L0"], dtype=np.float32)
        if np.any((l0 < 0.01) | (l0 > 2.0)):
            raise InvalidUplinkState(f"L0 out of plausible range [0.01, 2.0] m: {l0}")

        bounded_fields = {
            "base_ang_vel": 1.0e3,
            "commands": 1.0e3,
            "theta0": 1.0e3,
            "theta0_dot": 1.0e4,
            "L0_dot": 1.0e4,
            "wheel_vel": 1.0e4,
        }
        for name, limit in bounded_fields.items():
            arr = np.asarray(state[name], dtype=np.float32)
            if np.any(np.abs(arr) > limit):
                raise InvalidUplinkState(f"{name} magnitude exceeds {limit:g}: {arr}")

    def _build_observation(self, state):
        """Raw uplink state dict -> scaled, clipped 27-dim observation (float32).

        Order and scales mirror LeggedRobotVMC.compute_proprioception_observations
        / sim2sim build_observation. projected_gravity and last_action are unscaled.
        """
        s = self._scales
        obs = np.concatenate(
            [
                state["base_ang_vel"] * s["ang_vel"],
                state["projected_gravity"],
                state["commands"] * self._commands_scale,
                state["theta0"] * s["dof_pos"],
                state["theta0_dot"] * s["dof_vel"],
                state["L0"] * s["l0"],
                state["L0_dot"] * s["l0_dot"],
                state["wheel_pos"] * s["dof_pos"],
                state["wheel_vel"] * s["dof_vel"],
                self._last_action,
            ]
        ).astype(np.float32)
        return np.clip(obs, -self._clip_obs, self._clip_obs)

    def observe(self, state):
        """Raw uplink state dict -> (obs(27), obs_history(135)).

        On the first call the history is filled by repeating the first observation,
        matching the env reset that tiles the first proprio frame.
        """
        self._validate_state(state)
        obs = self._build_observation(state)
        if self._obs_history is None:
            self._obs_history = np.tile(obs, self._history_length)
        else:
            # Slide the window: drop the oldest frame, append the newest.
            self._obs_history = np.concatenate([self._obs_history[self._num_obs :], obs])
        return obs, self._obs_history

    def set_last_action(self, action):
        """Store the RAW action just produced; it becomes last_action in the next obs.

        This is the unscaled policy output (matching the training observation),
        NOT the physical command from scale_action().
        """
        self._last_action = np.asarray(action, dtype=np.float32)

    def scale_action(self, action):
        """Raw policy action -> physical command sent over the downlink.

        Per leg (order [L_theta, L_l0, L_wheel, R_theta, R_l0, R_wheel]):
            theta0_ref [rad]      = action_theta * theta
            l0_ref [m]            = action_l0    * l0 + l0_offset
            wheel_vel_ref [rad/s] = action_wheel * vel
        Mirrors LeggedRobotVMCFYT._compute_torques / sim2sim, so the lower machine
        receives ready-to-use physical targets instead of raw actions.
        """
        scaled = np.asarray(action, dtype=np.float32) * self._action_scale + self._action_offset
        return scaled.astype(np.float32)

    def reset(self):
        """Clear history and last action; call on (re)start or after a fault."""
        self._last_action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs_history = None
