#include <cstring>
#include <memory>
#include <vector>

#include "sim2real/config.hpp"
#include "sim2real/frame_codec.hpp"
#include "sim2real/protocol.hpp"
#include "sim2real/robot_serial_link.hpp"
#include "sim2real/serial_transport.hpp"
#include "sim2real/transport.hpp"
#include "test_util.hpp"

#ifndef SIM2REAL_CONFIG_FILE
#error "SIM2REAL_CONFIG_FILE must be defined by the build"
#endif

namespace {

using sim2real::ITransport;

// In-memory transport: rx feeds bytes to the link, tx captures what it writes.
class MockTransport : public ITransport {
 public:
  std::vector<std::uint8_t> rx;
  std::vector<std::vector<std::uint8_t>> tx;

  std::vector<std::uint8_t> readAvailable() override {
    std::vector<std::uint8_t> out = rx;
    rx.clear();
    return out;
  }
  void write(const std::vector<std::uint8_t>& data) override { tx.push_back(data); }
  void close() override {}
};

void putFloat(std::vector<std::uint8_t>& v, float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  v.push_back(static_cast<std::uint8_t>(bits & 0xFF));
  v.push_back(static_cast<std::uint8_t>((bits >> 8) & 0xFF));
  v.push_back(static_cast<std::uint8_t>((bits >> 16) & 0xFF));
  v.push_back(static_cast<std::uint8_t>((bits >> 24) & 0xFF));
}

// Serialize an UplinkState into the 21-float payload (matches decodeUplink order).
std::vector<std::uint8_t> makeUplinkPayload(const sim2real::UplinkState& s) {
  std::vector<std::uint8_t> p;
  for (float v : s.base_ang_vel) putFloat(p, v);
  for (float v : s.projected_gravity) putFloat(p, v);
  for (float v : s.commands) putFloat(p, v);
  for (float v : s.theta0) putFloat(p, v);
  for (float v : s.theta0_dot) putFloat(p, v);
  for (float v : s.l0) putFloat(p, v);
  for (float v : s.l0_dot) putFloat(p, v);
  for (float v : s.wheel_pos) putFloat(p, v);
  for (float v : s.wheel_vel) putFloat(p, v);
  return p;
}

}  // namespace

int main() {
  using namespace sim2real;
  using sim2real_test::check;
  using sim2real_test::nearFloat;

  const Config cfg = Config::load(SIM2REAL_CONFIG_FILE);

  // --- poll() returns the NEWEST frame and drops the older backlog ---
  {
    auto mock = std::make_unique<MockTransport>();
    MockTransport* raw = mock.get();
    RobotSerialLink link(cfg, std::move(mock));

    FrameCodec uplink_enc(cfg.frame.uplink_payload_size, cfg.frame);
    UplinkState s1{};
    s1.commands = {0.1f, 0.0f, 0.18f};
    UplinkState s2{};
    s2.commands = {0.9f, 0.0f, 0.18f};
    // Two frames queued in one go; poll must return s2 (the newer).
    const auto f1 = uplink_enc.encode(makeUplinkPayload(s1));
    const auto f2 = uplink_enc.encode(makeUplinkPayload(s2));
    raw->rx.insert(raw->rx.end(), f1.begin(), f1.end());
    raw->rx.insert(raw->rx.end(), f2.begin(), f2.end());

    const std::optional<UplinkState> got = link.poll();
    check(got.has_value(), "poll returns a state when frames present");
    check(got.has_value() && nearFloat(got->commands[0], 0.9f), "poll returns the newest frame");

    // No more bytes -> nullopt.
    check(!link.poll().has_value(), "poll returns nullopt with no fresh frame");
  }

  // --- sendAction() writes one decodable downlink frame ---
  {
    auto mock = std::make_unique<MockTransport>();
    MockTransport* raw = mock.get();
    RobotSerialLink link(cfg, std::move(mock));

    const Action action = {0.5f, -0.3f, 2.0f, -0.5f, 0.3f, -2.0f};
    link.sendAction(action);
    check(raw->tx.size() == 1, "sendAction writes exactly one frame");

    FrameCodec downlink_dec(cfg.frame.downlink_payload_size, cfg.frame);
    const auto payloads = downlink_dec.feed(raw->tx.front());
    check(payloads.size() == 1, "downlink frame decodes to one payload");
    // Decode the action back (little-endian floats) and compare.
    bool match = payloads.size() == 1;
    if (match) {
      const auto& p = payloads[0];
      for (std::size_t i = 0; i < NUM_ACTIONS; ++i) {
        std::uint32_t bits = std::uint32_t(p[i * 4]) | (std::uint32_t(p[i * 4 + 1]) << 8) |
                             (std::uint32_t(p[i * 4 + 2]) << 16) | (std::uint32_t(p[i * 4 + 3]) << 24);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        if (!nearFloat(f, action[i])) match = false;
      }
    }
    check(match, "downlink action round-trips");
  }

  // --- SerialTransport on a missing port throws (no hardware needed) ---
  {
    bool threw = false;
    try {
      SerialTransport bad("/dev/sim2real_nonexistent_port", 115200);
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "opening a missing serial port throws cleanly");
  }

  return sim2real_test::report("test_serial_link");
}
