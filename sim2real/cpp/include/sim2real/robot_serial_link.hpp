#ifndef SIM2REAL_ROBOT_SERIAL_LINK_HPP
#define SIM2REAL_ROBOT_SERIAL_LINK_HPP

#include <memory>
#include <optional>

#include "sim2real/config.hpp"
#include "sim2real/frame_codec.hpp"
#include "sim2real/protocol.hpp"
#include "sim2real/transport.hpp"

namespace sim2real {

// End-to-end serial link for the deployment loop, built from config. Ties the
// transport, the two frame codecs (uplink/downlink), and the protocol together,
// exactly like the Python RobotSerialLink.
class RobotSerialLink {
 public:
  // Open a real serial port from config.serial.
  explicit RobotSerialLink(const Config& config);

  // Inject a transport (used by tests with a mock). Takes ownership.
  RobotSerialLink(const Config& config, std::unique_ptr<ITransport> transport);

  // Drain the transport, parse frames, and return the NEWEST decoded state, or
  // std::nullopt if no complete frame arrived this cycle. Returning only the
  // latest frame keeps the policy on current state and drops stale backlog.
  std::optional<UplinkState> poll();

  // Encode and send one downlink action frame.
  void sendAction(const Action& action);

  void close();

 private:
  std::unique_ptr<ITransport> transport_;
  FrameCodec uplink_codec_;
  FrameCodec downlink_codec_;
};

}  // namespace sim2real

#endif  // SIM2REAL_ROBOT_SERIAL_LINK_HPP
