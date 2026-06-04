#include "sim2real/controller.hpp"

#include <algorithm>
#include <cstddef>

namespace sim2real {

Sim2RealController::Sim2RealController(const Config& config)
    : scales_(config.observation_scales),
      clip_obs_(config.clipping.observations),
      history_length_(static_cast<std::size_t>(config.network.obs_history_length)),
      last_action_{},          // zero, matching the env reset
      history_initialized_(false) {}

Observation Sim2RealController::buildObservation(const UplinkState& state) const {
  Observation obs;
  std::size_t k = 0;
  // Helper: append the scaled elements of a fixed-size array, advancing the cursor.
  auto append = [&obs, &k](const auto& values, float scale) {
    for (float v : values) {
      obs[k++] = v * scale;
    }
  };

  // Exact order and scales of LeggedRobotVMC.compute_proprioception_observations
  // / the Python build_observation. projected_gravity and last_action are unscaled.
  append(state.base_ang_vel, scales_.ang_vel);
  append(state.projected_gravity, 1.0f);
  for (std::size_t i = 0; i < 3; ++i) {
    obs[k++] = state.commands[i] * scales_.commands_scale[i];
  }
  append(state.theta0, scales_.dof_pos);
  append(state.theta0_dot, scales_.dof_vel);
  append(state.l0, scales_.l0);
  append(state.l0_dot, scales_.l0_dot);
  append(state.wheel_pos, scales_.dof_pos);
  append(state.wheel_vel, scales_.dof_vel);
  append(last_action_, 1.0f);

  // Clip every observation element to the configured bound.
  for (float& v : obs) {
    v = std::clamp(v, -clip_obs_, clip_obs_);
  }
  return obs;
}

Observation Sim2RealController::observe(const UplinkState& state) {
  const Observation obs = buildObservation(state);
  if (!history_initialized_) {
    // First frame: fill the whole window by repeating it (matches the env reset
    // that tiles the first proprio frame across the history).
    obs_history_.clear();
    obs_history_.reserve(NUM_OBS * history_length_);
    for (std::size_t frame = 0; frame < history_length_; ++frame) {
      obs_history_.insert(obs_history_.end(), obs.begin(), obs.end());
    }
    history_initialized_ = true;
  } else {
    // Slide the window: drop the oldest frame, append the newest.
    obs_history_.erase(obs_history_.begin(), obs_history_.begin() + NUM_OBS);
    obs_history_.insert(obs_history_.end(), obs.begin(), obs.end());
  }
  return obs;
}

void Sim2RealController::setLastAction(const Action& action) { last_action_ = action; }

void Sim2RealController::reset() {
  last_action_.fill(0.0f);
  obs_history_.clear();
  history_initialized_ = false;
}

}  // namespace sim2real
