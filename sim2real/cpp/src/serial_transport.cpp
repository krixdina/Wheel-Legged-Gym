#include "sim2real/serial_transport.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sim2real {
namespace {

// Map an integer baudrate to its termios speed constant. Only the rates a NUC
// link realistically uses are listed; an unknown rate is rejected with a clear
// message rather than silently mis-configuring the port.
speed_t toSpeed(int baudrate) {
  switch (baudrate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    default:
      throw std::runtime_error("SerialTransport: unsupported baudrate " + std::to_string(baudrate));
  }
}

}  // namespace

SerialTransport::SerialTransport(const std::string& port, int baudrate) : fd_(-1) {
  // O_NDELAY: open without blocking on carrier-detect; the read mode is set to
  // non-blocking below via VMIN/VTIME, so a poll loop never stalls on the port.
  fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd_ < 0) {
    throw std::runtime_error("SerialTransport: cannot open '" + port + "': " + std::strerror(errno));
  }

  struct termios options;
  if (tcgetattr(fd_, &options) != 0) {
    const std::string msg = std::strerror(errno);
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("SerialTransport: tcgetattr failed: " + msg);
  }

  const speed_t speed = toSpeed(baudrate);
  cfsetispeed(&options, speed);
  cfsetospeed(&options, speed);

  // 8N1, local line, receiver enabled.
  options.c_cflag |= (CLOCAL | CREAD);
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;        // 8 data bits
  options.c_cflag &= ~PARENB;    // no parity
  options.c_iflag &= ~INPCK;
  options.c_cflag &= ~CSTOPB;    // 1 stop bit
  options.c_cflag &= ~CRTSCTS;   // no hardware flow control

  // Raw mode: no output post-processing, no canonical/echo/signal handling, and
  // pass bytes such as 0x0D/0x11/0x13 through untouched (they are protocol data).
  options.c_oflag &= ~OPOST;
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  options.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON | IXOFF | IXANY);

  // VMIN=0, VTIME=0: fully non-blocking read -- return immediately with whatever
  // bytes are buffered (possibly none). Matches the Python timeout=0 semantics.
  options.c_cc[VMIN] = 0;
  options.c_cc[VTIME] = 0;

  tcflush(fd_, TCIFLUSH);
  if (tcsetattr(fd_, TCSANOW, &options) != 0) {
    const std::string msg = std::strerror(errno);
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("SerialTransport: tcsetattr failed: " + msg);
  }
}

SerialTransport::~SerialTransport() { close(); }

std::vector<std::uint8_t> SerialTransport::readAvailable() {
  if (fd_ < 0) {
    return {};
  }
  // Query how many bytes are buffered (like pyserial's in_waiting), then read
  // exactly that many so no partial-frame data is left stranded in the OS buffer.
  int pending = 0;
  if (ioctl(fd_, FIONREAD, &pending) != 0 || pending <= 0) {
    return {};
  }
  std::vector<std::uint8_t> data(static_cast<std::size_t>(pending));
  const ssize_t n = ::read(fd_, data.data(), data.size());
  if (n <= 0) {
    return {};
  }
  data.resize(static_cast<std::size_t>(n));
  return data;
}

void SerialTransport::write(const std::vector<std::uint8_t>& data) {
  if (fd_ < 0) {
    throw std::runtime_error("SerialTransport::write: port is closed");
  }
  // Loop until the whole buffer is written; ::write may accept fewer bytes.
  std::size_t written = 0;
  while (written < data.size()) {
    const ssize_t n = ::write(fd_, data.data() + written, data.size() - written);
    if (n < 0) {
      throw std::runtime_error(std::string("SerialTransport::write failed: ") + std::strerror(errno));
    }
    written += static_cast<std::size_t>(n);
  }
}

void SerialTransport::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

}  // namespace sim2real
