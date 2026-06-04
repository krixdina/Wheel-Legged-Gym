#ifndef SIM2REAL_TRANSPORT_HPP
#define SIM2REAL_TRANSPORT_HPP

#include <cstdint>
#include <vector>

namespace sim2real {

// Byte-stream transport abstraction, mirroring rm_serial_driver's
// TransporterInterface. Decoupling the wire from the framing lets RobotSerialLink
// be exercised against a mock transport in tests, with no real serial device.
class ITransport {
 public:
  virtual ~ITransport() = default;

  // Non-blocking: return all bytes currently buffered (possibly empty).
  virtual std::vector<std::uint8_t> readAvailable() = 0;

  // Write the whole buffer to the device.
  virtual void write(const std::vector<std::uint8_t>& data) = 0;

  virtual void close() = 0;
};

}  // namespace sim2real

#endif  // SIM2REAL_TRANSPORT_HPP
