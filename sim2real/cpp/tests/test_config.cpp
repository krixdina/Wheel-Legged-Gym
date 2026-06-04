#include <string>

#include "sim2real/config.hpp"
#include "sim2real/protocol.hpp"
#include "test_util.hpp"

// SIM2REAL_CONFIG_FILE is the absolute path to the repo's sim2real.yaml, injected
// by CMake so the test does not depend on the working directory.
#ifndef SIM2REAL_CONFIG_FILE
#error "SIM2REAL_CONFIG_FILE must be defined by the build"
#endif

int main() {
  using namespace sim2real;
  using sim2real_test::check;

  const Config cfg = Config::load(SIM2REAL_CONFIG_FILE);

  // Serial.
  check(cfg.serial.port == "/dev/ttyUSB0", "serial.port");
  check(cfg.serial.baudrate == 115200, "serial.baudrate");

  // Frame markers / CRC (hex scalars parsed correctly).
  check(cfg.frame.sof == 0xFF, "frame.sof == 0xFF");
  check(cfg.frame.eof == 0x0D, "frame.eof == 0x0D");
  check(cfg.frame.use_crc8 == true, "frame.use_crc8");
  check(cfg.frame.crc8_poly == 0x07, "frame.crc8_poly == 0x07");
  check(cfg.frame.crc8_init == 0x00, "frame.crc8_init == 0x00");

  // Payload sizes derived from the format strings.
  check(cfg.frame.uplink_payload_size == UPLINK_FLOATS * sizeof(float), "uplink payload = 84 bytes");
  check(cfg.frame.downlink_payload_size == NUM_ACTIONS * sizeof(float), "downlink payload = 24 bytes");

  // Network dims.
  check(cfg.network.num_obs == 27, "network.num_obs");
  check(cfg.network.num_actions == 6, "network.num_actions");
  check(cfg.network.obs_history_length == 5, "network.obs_history_length");
  check(cfg.network.num_encoder_obs == 135, "network.num_encoder_obs");
  // onnx_model_path resolved to an absolute path ending in the configured name.
  check(!cfg.network.onnx_model_path.empty() && cfg.network.onnx_model_path.front() == '/',
        "network.onnx_model_path is absolute");
  check(cfg.network.onnx_model_path.find("sim2real/model/policy.onnx") != std::string::npos,
        "network.onnx_model_path keeps the configured suffix");

  // Scales.
  check(sim2real_test::nearFloat(cfg.observation_scales.ang_vel, 0.25f), "scale.ang_vel");
  check(sim2real_test::nearFloat(cfg.observation_scales.dof_vel, 0.05f), "scale.dof_vel");
  check(sim2real_test::nearFloat(cfg.observation_scales.l0, 4.5f), "scale.l0");
  check(sim2real_test::nearFloat(cfg.observation_scales.commands_scale[0], 2.0f), "commands_scale[0]");
  check(sim2real_test::nearFloat(cfg.observation_scales.commands_scale[2], 5.0f), "commands_scale[2]");

  // Clipping + timing.
  check(sim2real_test::nearFloat(cfg.clipping.observations, 100.0f), "clip.observations");
  check(sim2real_test::nearFloat(cfg.clipping.actions, 100.0f), "clip.actions");
  check(cfg.control_timing.policy_rate_hz == 100, "policy_rate_hz");
  check(cfg.control_timing.max_missed_frames == 10, "max_missed_frames");

  return sim2real_test::report("test_config");
}
