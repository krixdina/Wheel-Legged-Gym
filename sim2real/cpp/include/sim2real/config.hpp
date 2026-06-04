#ifndef SIM2REAL_CONFIG_HPP
#define SIM2REAL_CONFIG_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace sim2real {

// Serial port settings (config.serial).
struct SerialConfig {
  std::string port;
  int baudrate;
};

// Frame protocol settings (config.frame). The payload sizes are derived from the
// struct format strings so they stay in lockstep with the Python side.
struct FrameConfig {
  std::uint8_t sof;
  std::uint8_t eof;
  bool use_crc8;
  std::uint8_t crc8_poly;
  std::uint8_t crc8_init;
  std::string uplink_format;
  std::string downlink_format;
  std::size_t uplink_payload_size;    // bytes, derived from uplink_format
  std::size_t downlink_payload_size;  // bytes, derived from downlink_format
};

// Policy network settings (config.network).
struct NetworkConfig {
  std::string onnx_model_path;  // absolute path, resolved against the repo root
  std::string onnx_model_path_raw;  // as written in yaml, kept for error messages
  int num_obs;
  int num_actions;
  int obs_history_length;
  int num_encoder_obs;
};

// Observation scaling (config.observation_scales). projected_gravity and
// last_action are intentionally unscaled (factor 1), matching training/sim2sim.
struct ObservationScales {
  float ang_vel;
  float dof_pos;
  float dof_vel;
  float l0;
  float l0_dot;
  std::array<float, 3> commands_scale;  // [vx, wz, height]
};

// Observation/action clipping bounds (config.clipping).
struct ClippingConfig {
  float observations;
  float actions;
};

// Control-loop timing and link-loss policy (config.control_timing).
struct ControlTiming {
  int policy_rate_hz;
  int max_missed_frames;
};

// Parsed view of sim2real/config/sim2real.yaml, shared by all C++ modules.
struct Config {
  SerialConfig serial;
  FrameConfig frame;
  NetworkConfig network;
  ObservationScales observation_scales;
  ClippingConfig clipping;
  ControlTiming control_timing;

  // Load and validate the shared yaml.
  // config_path: path to sim2real.yaml; the repo root used to resolve the model
  //              path is derived from it (yaml sits at <repo>/sim2real/config/).
  // return:      the fully parsed Config.
  // Throws std::runtime_error if the file is missing, a key is absent, a frame
  // format is unsupported, or a dimension disagrees with the compile-time
  // protocol constants in protocol.hpp.
  static Config load(const std::string& config_path);
};

}  // namespace sim2real

#endif  // SIM2REAL_CONFIG_HPP
