#ifndef SIM2REAL_CONTROLLER_HPP
#define SIM2REAL_CONTROLLER_HPP

#include <vector>

#include "sim2real/config.hpp"
#include "sim2real/protocol.hpp"

namespace sim2real {

// NUC-side policy I/O, a faithful port of the Python Sim2RealController. It does
// NOT run FK or the VMC/PD law (the lower machine does): it only scales the raw
// uplink state into the 27-dim observation (with last_action appended), clips it,
// maintains the 5-frame (135-dim) history window, and remembers the last action.
class Sim2RealController {
 public:
  explicit Sim2RealController(const Config& config);

  // Build the observation from one raw state and advance the history window.
  // On the first call the history is filled by repeating the first observation
  // (matching the env reset that tiles the first proprio frame); afterwards it
  // slides: drop the oldest NUM_OBS values, append the newest.
  // Returns the current 27-dim observation; pair it with obsHistory().
  Observation observe(const UplinkState& state);

  // The 135-dim history window matching the most recent observe() call.
  const std::vector<float>& obsHistory() const { return obs_history_; }

  // Store the action just sent; it becomes last_action in the next observation.
  void setLastAction(const Action& action);

  // Clear history and last action; call on (re)start or after a fault.
  void reset();

 private:
  // Scale + clip one raw state into the 27-dim observation, in the exact field
  // order of the Python build_observation.
  Observation buildObservation(const UplinkState& state) const;

  ObservationScales scales_;
  float clip_obs_;
  std::size_t history_length_;
  Action last_action_;
  std::vector<float> obs_history_;  // size NUM_ENCODER_OBS once initialised
  bool history_initialized_;
};

}  // namespace sim2real

#endif  // SIM2REAL_CONTROLLER_HPP
