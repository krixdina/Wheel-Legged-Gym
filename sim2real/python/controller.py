"""sim2real policy-I/O controller for the wheel-legged robot (NUC side).

Unlike the sim2sim controller, this one does NOT run forward kinematics or the
VMC/PD torque law: the lower machine already computes the virtual-leg states
(theta0/L0 and rates) and the gravity projection and sends them up raw, and the
lower machine also turns the 6-dim action into joint torques. This controller is
purely the NUC-side policy interface:

    - scale the 21 raw uplink states into the 27-dim observation (last_action
      appended), in the exact order of 状态量与动作量说明.md,
    - clip the observation,
    - maintain the 5-frame history window the encoder consumes,
    - remember the last action that was sent (it is fed back into the next obs).

All scales / dims come from the shared config.
"""
import numpy as np

from config import CONFIG


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
        # last_action starts at zero (matches the env reset); _obs_history is
        # lazily filled on the first observation so it repeats the first frame.
        self._last_action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs_history = None

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

        On the first call the history is filled by repeating the first
        observation, matching the env reset that tiles the first proprio frame.
        """
        obs = self._build_observation(state)
        if self._obs_history is None:
            self._obs_history = np.tile(obs, self._history_length)
        else:
            # Slide the window: drop the oldest frame, append the newest.
            self._obs_history = np.concatenate([self._obs_history[self._num_obs :], obs])
        return obs, self._obs_history

    def set_last_action(self, action):
        """Store the action just sent; it becomes last_action in the next obs."""
        self._last_action = np.asarray(action, dtype=np.float32)

    def reset(self):
        """Clear history and last action; call on (re)start or after a fault."""
        self._last_action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs_history = None
