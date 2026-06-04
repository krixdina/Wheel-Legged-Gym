#ifndef SIM2REAL_PROTOCOL_HPP
#define SIM2REAL_PROTOCOL_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sim2real {

// Fixed protocol dimensions shared by every module. They mirror the policy in
// sim2real/config/sim2real.yaml and are validated against the config at load time
// (see Config::load), so a yaml/firmware mismatch fails loudly instead of silently.
constexpr std::size_t UPLINK_FLOATS = 21;    // raw state floats per uplink frame
constexpr std::size_t NUM_ACTIONS = 6;       // action floats per downlink frame
constexpr std::size_t NUM_OBS = 27;          // assembled observation length
constexpr std::size_t HISTORY_LENGTH = 5;    // history window length, in frames
constexpr std::size_t NUM_ENCODER_OBS = NUM_OBS * HISTORY_LENGTH;  // 135

using Action = std::array<float, NUM_ACTIONS>;
using Observation = std::array<float, NUM_OBS>;

// Raw, unscaled state decoded from one uplink frame. Field order and sizes match
// the 21-float payload layout documented in config.frame.uplink_format. These are
// SI physical quantities (rad/s, unit vector, rad, m, ...); scaling happens later
// in Sim2RealController, exactly as in the Python pipeline.
struct UplinkState {
  std::array<float, 3> base_ang_vel;
  std::array<float, 3> projected_gravity;
  std::array<float, 3> commands;
  std::array<float, 2> theta0;
  std::array<float, 2> theta0_dot;
  std::array<float, 2> l0;
  std::array<float, 2> l0_dot;
  std::array<float, 2> wheel_pos;
  std::array<float, 2> wheel_vel;
};

// Decode a little-endian uplink payload (UPLINK_FLOATS float32 values) into an
// UplinkState. The payload is the frame body only, i.e. SOF/CRC/EOF already
// stripped by FrameCodec.
// payload: must be exactly UPLINK_FLOATS * 4 bytes, else std::invalid_argument.
// return:  the decoded raw state.
UplinkState decodeUplink(const std::vector<std::uint8_t>& payload);

// Encode a 6-dim policy action into a little-endian downlink payload.
// action: the policy output [L_theta, L_l0, L_wheel, R_theta, R_l0, R_wheel].
// return: NUM_ACTIONS * 4 payload bytes (no SOF/CRC/EOF; FrameCodec adds those).
std::vector<std::uint8_t> encodeDownlink(const Action& action);

}  // namespace sim2real

#endif  // SIM2REAL_PROTOCOL_HPP
