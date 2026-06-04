#include "sim2real/robot_serial_link.hpp"

#include <utility>

#include "sim2real/serial_transport.hpp"

namespace sim2real {

RobotSerialLink::RobotSerialLink(const Config& config)
    : RobotSerialLink(config,
                      std::make_unique<SerialTransport>(config.serial.port, config.serial.baudrate)) {}

RobotSerialLink::RobotSerialLink(const Config& config, std::unique_ptr<ITransport> transport)
    : transport_(std::move(transport)),
      uplink_codec_(config.frame.uplink_payload_size, config.frame),
      downlink_codec_(config.frame.downlink_payload_size, config.frame) {}

std::optional<UplinkState> RobotSerialLink::poll() {
  const std::vector<std::vector<std::uint8_t>> payloads = uplink_codec_.feed(transport_->readAvailable());
  if (payloads.empty()) {
    return std::nullopt;
  }
  // Only the newest frame matters; older backlog is dropped so the policy always
  // acts on current state.
  return decodeUplink(payloads.back());
}

void RobotSerialLink::sendAction(const Action& action) {
  transport_->write(downlink_codec_.encode(encodeDownlink(action)));
}

void RobotSerialLink::close() { transport_->close(); }

}  // namespace sim2real
