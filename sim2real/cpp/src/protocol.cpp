#include "sim2real/protocol.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace sim2real {
namespace {

// Read one little-endian float32 from p[0..4). Done bit-by-bit (not a raw cast)
// so the result is correct regardless of host endianness and never relies on
// struct layout/padding.
float readLeFloat(const std::uint8_t* p) {
  std::uint32_t bits = static_cast<std::uint32_t>(p[0]) |
                       (static_cast<std::uint32_t>(p[1]) << 8) |
                       (static_cast<std::uint32_t>(p[2]) << 16) |
                       (static_cast<std::uint32_t>(p[3]) << 24);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Write one little-endian float32 into out[0..4).
void writeLeFloat(float value, std::uint8_t* out) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  out[0] = static_cast<std::uint8_t>(bits & 0xFF);
  out[1] = static_cast<std::uint8_t>((bits >> 8) & 0xFF);
  out[2] = static_cast<std::uint8_t>((bits >> 16) & 0xFF);
  out[3] = static_cast<std::uint8_t>((bits >> 24) & 0xFF);
}

// Decode `count` consecutive little-endian floats starting at byte `offset`.
template <std::size_t N>
std::array<float, N> readFloats(const std::vector<std::uint8_t>& payload, std::size_t offset) {
  std::array<float, N> out;
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = readLeFloat(payload.data() + offset + i * sizeof(float));
  }
  return out;
}

}  // namespace

UplinkState decodeUplink(const std::vector<std::uint8_t>& payload) {
  constexpr std::size_t expected = UPLINK_FLOATS * sizeof(float);
  if (payload.size() != expected) {
    throw std::invalid_argument("decodeUplink: payload is " + std::to_string(payload.size()) +
                                " bytes, expected " + std::to_string(expected));
  }
  // Field offsets follow the documented 21-float layout: each std::array consumes
  // its size in floats, in order, with no gaps.
  UplinkState state;
  state.base_ang_vel = readFloats<3>(payload, 0);
  state.projected_gravity = readFloats<3>(payload, 3 * sizeof(float));
  state.commands = readFloats<3>(payload, 6 * sizeof(float));
  state.theta0 = readFloats<2>(payload, 9 * sizeof(float));
  state.theta0_dot = readFloats<2>(payload, 11 * sizeof(float));
  state.l0 = readFloats<2>(payload, 13 * sizeof(float));
  state.l0_dot = readFloats<2>(payload, 15 * sizeof(float));
  state.wheel_pos = readFloats<2>(payload, 17 * sizeof(float));
  state.wheel_vel = readFloats<2>(payload, 19 * sizeof(float));
  return state;
}

std::vector<std::uint8_t> encodeDownlink(const Action& action) {
  std::vector<std::uint8_t> payload(NUM_ACTIONS * sizeof(float));
  for (std::size_t i = 0; i < NUM_ACTIONS; ++i) {
    writeLeFloat(action[i], payload.data() + i * sizeof(float));
  }
  return payload;
}

}  // namespace sim2real
