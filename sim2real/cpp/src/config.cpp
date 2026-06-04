#include "sim2real/config.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "sim2real/protocol.hpp"

namespace sim2real {
namespace {

// Fetch a required child node, throwing a clear error naming the missing key
// instead of yaml-cpp's opaque conversion failure.
YAML::Node require(const YAML::Node& parent, const std::string& key, const std::string& where) {
  if (!parent[key]) {
    throw std::runtime_error("config: missing key '" + key + "' in section '" + where + "'");
  }
  return parent[key];
}

// Parse a byte that may be written as decimal or 0x.. hex in the yaml. yaml-cpp
// does not reliably parse hex scalars, so read the raw text and let std::stoul
// auto-detect the base (0 => recognises the 0x prefix).
std::uint8_t parseByte(const YAML::Node& node, const std::string& key, const std::string& where) {
  const std::string text = require(node, key, where).as<std::string>();
  const unsigned long value = std::stoul(text, nullptr, 0);
  if (value > 0xFF) {
    throw std::runtime_error("config: '" + key + "' = " + text + " does not fit in a byte");
  }
  return static_cast<std::uint8_t>(value);
}

// Derive the payload byte count from a struct format string of the form "<Nf"
// (little-endian, N float32). Only this subset is used by the protocol; anything
// else is rejected so a malformed format fails loudly.
std::size_t payloadBytesFromFormat(const std::string& fmt, const std::string& key) {
  if (fmt.size() < 3 || fmt.front() != '<' || fmt.back() != 'f') {
    throw std::runtime_error("config: unsupported " + key + " '" + fmt + "'; expected \"<Nf\"");
  }
  const std::string count_text = fmt.substr(1, fmt.size() - 2);
  const std::size_t count = static_cast<std::size_t>(std::stoul(count_text));
  return count * sizeof(float);
}

}  // namespace

Config Config::load(const std::string& config_path) {
  namespace fs = std::filesystem;
  if (!fs::exists(config_path)) {
    throw std::runtime_error("config: file not found: " + config_path);
  }
  const YAML::Node root = YAML::LoadFile(config_path);
  Config cfg;

  // --- serial ---
  const YAML::Node serial = require(root, "serial", "root");
  cfg.serial.port = require(serial, "port", "serial").as<std::string>();
  cfg.serial.baudrate = require(serial, "baudrate", "serial").as<int>();

  // --- frame ---
  const YAML::Node frame = require(root, "frame", "root");
  cfg.frame.sof = parseByte(frame, "sof", "frame");
  cfg.frame.eof = parseByte(frame, "eof", "frame");
  cfg.frame.use_crc8 = require(frame, "use_crc8", "frame").as<bool>();
  cfg.frame.crc8_poly = parseByte(frame, "crc8_poly", "frame");
  cfg.frame.crc8_init = parseByte(frame, "crc8_init", "frame");
  cfg.frame.uplink_format = require(frame, "uplink_format", "frame").as<std::string>();
  cfg.frame.downlink_format = require(frame, "downlink_format", "frame").as<std::string>();
  cfg.frame.uplink_payload_size = payloadBytesFromFormat(cfg.frame.uplink_format, "uplink_format");
  cfg.frame.downlink_payload_size = payloadBytesFromFormat(cfg.frame.downlink_format, "downlink_format");

  // --- network ---
  const YAML::Node network = require(root, "network", "root");
  cfg.network.onnx_model_path_raw = require(network, "onnx_model_path", "network").as<std::string>();
  cfg.network.num_obs = require(network, "num_obs", "network").as<int>();
  cfg.network.num_actions = require(network, "num_actions", "network").as<int>();
  cfg.network.obs_history_length = require(network, "obs_history_length", "network").as<int>();
  cfg.network.num_encoder_obs = require(network, "num_encoder_obs", "network").as<int>();
  // Resolve the model path relative to the repo root. The yaml lives at
  // <repo>/sim2real/config/sim2real.yaml, so the root is three levels up.
  const fs::path repo_root = fs::absolute(config_path).parent_path().parent_path().parent_path();
  cfg.network.onnx_model_path = (repo_root / cfg.network.onnx_model_path_raw).string();

  // --- observation_scales ---
  const YAML::Node scales = require(root, "observation_scales", "root");
  cfg.observation_scales.ang_vel = require(scales, "ang_vel", "observation_scales").as<float>();
  cfg.observation_scales.dof_pos = require(scales, "dof_pos", "observation_scales").as<float>();
  cfg.observation_scales.dof_vel = require(scales, "dof_vel", "observation_scales").as<float>();
  cfg.observation_scales.l0 = require(scales, "l0", "observation_scales").as<float>();
  cfg.observation_scales.l0_dot = require(scales, "l0_dot", "observation_scales").as<float>();
  const YAML::Node cmd_scale = require(scales, "commands_scale", "observation_scales");
  if (cmd_scale.size() != 3) {
    throw std::runtime_error("config: observation_scales.commands_scale must have 3 entries");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    cfg.observation_scales.commands_scale[i] = cmd_scale[i].as<float>();
  }

  // --- clipping ---
  const YAML::Node clipping = require(root, "clipping", "root");
  cfg.clipping.observations = require(clipping, "observations", "clipping").as<float>();
  cfg.clipping.actions = require(clipping, "actions", "clipping").as<float>();

  // --- control_timing ---
  const YAML::Node timing = require(root, "control_timing", "root");
  cfg.control_timing.policy_rate_hz = require(timing, "policy_rate_hz", "control_timing").as<int>();
  cfg.control_timing.max_missed_frames =
      require(timing, "max_missed_frames", "control_timing").as<int>();

  // --- validate dimensions against the compile-time protocol constants ---
  // Any mismatch means the C++ binary and the yaml disagree on the protocol, so
  // fail loudly rather than corrupt observations at runtime.
  auto expect = [](const std::string& name, long got, long want) {
    if (got != want) {
      throw std::runtime_error("config: " + name + " = " + std::to_string(got) +
                               " but this build expects " + std::to_string(want));
    }
  };
  expect("network.num_obs", cfg.network.num_obs, static_cast<long>(NUM_OBS));
  expect("network.num_actions", cfg.network.num_actions, static_cast<long>(NUM_ACTIONS));
  expect("network.obs_history_length", cfg.network.obs_history_length,
         static_cast<long>(HISTORY_LENGTH));
  expect("network.num_encoder_obs", cfg.network.num_encoder_obs,
         static_cast<long>(NUM_ENCODER_OBS));
  expect("frame.uplink payload floats", static_cast<long>(cfg.frame.uplink_payload_size),
         static_cast<long>(UPLINK_FLOATS * sizeof(float)));
  expect("frame.downlink payload floats", static_cast<long>(cfg.frame.downlink_payload_size),
         static_cast<long>(NUM_ACTIONS * sizeof(float)));

  return cfg;
}

}  // namespace sim2real
