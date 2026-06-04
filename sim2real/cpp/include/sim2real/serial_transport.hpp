#ifndef SIM2REAL_SERIAL_TRANSPORT_HPP
#define SIM2REAL_SERIAL_TRANSPORT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "sim2real/transport.hpp"

namespace sim2real {

// POSIX termios serial port, implementing ITransport. Configured 8N1, no flow
// control, raw mode (after rm_serial_driver's UartTransporter), but with
// VMIN=0/VTIME=0 so reads are non-blocking and return whatever is buffered --
// the right behaviour for a fixed-rate control loop that polls every cycle.
class SerialTransport : public ITransport {
 public:
  // Open and configure the port; throws std::runtime_error on failure (port
  // missing, permission denied, or an unsupported baudrate).
  SerialTransport(const std::string& port, int baudrate);
  ~SerialTransport() override;

  std::vector<std::uint8_t> readAvailable() override;
  void write(const std::vector<std::uint8_t>& data) override;
  void close() override;

 private:
  int fd_;
};

}  // namespace sim2real

#endif  // SIM2REAL_SERIAL_TRANSPORT_HPP
