#include "sim2real/frame_codec.hpp"

#include <stdexcept>
#include <string>

namespace sim2real {

FrameCodec::FrameCodec(std::size_t payload_size, const FrameConfig& frame_cfg)
    : payload_size_(payload_size),
      sof_(frame_cfg.sof),
      eof_(frame_cfg.eof),
      use_crc8_(frame_cfg.use_crc8),
      crc_poly_(frame_cfg.crc8_poly),
      crc_init_(frame_cfg.crc8_init),
      // frame = SOF + payload (+ CRC if enabled) + EOF
      frame_size_(payload_size + (frame_cfg.use_crc8 ? 3 : 2)) {}

std::uint8_t FrameCodec::crc8(const std::uint8_t* data, std::size_t len) const {
  // Bit-by-bit CRC-8 with the configured polynomial/initial value; identical to
  // the Python _crc8 so both ends agree on the checksum.
  std::uint8_t crc = crc_init_;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 0x80) {
        crc = static_cast<std::uint8_t>((crc << 1) ^ crc_poly_);
      } else {
        crc = static_cast<std::uint8_t>(crc << 1);
      }
    }
  }
  return crc;
}

std::vector<std::uint8_t> FrameCodec::encode(const std::vector<std::uint8_t>& payload) const {
  if (payload.size() != payload_size_) {
    throw std::invalid_argument("FrameCodec::encode: payload is " + std::to_string(payload.size()) +
                                " bytes, expected " + std::to_string(payload_size_));
  }
  std::vector<std::uint8_t> frame;
  frame.reserve(frame_size_);
  frame.push_back(sof_);
  frame.insert(frame.end(), payload.begin(), payload.end());
  if (use_crc8_) {
    // CRC8 covers SOF + payload (the bytes currently in `frame`).
    frame.push_back(crc8(frame.data(), frame.size()));
  }
  frame.push_back(eof_);
  return frame;
}

std::vector<std::vector<std::uint8_t>> FrameCodec::feed(const std::vector<std::uint8_t>& chunk) {
  // A single read can yield half a frame, a whole frame, or several, so buffer
  // the new bytes and parse as many complete frames as are now available.
  buffer_.insert(buffer_.end(), chunk.begin(), chunk.end());

  std::vector<std::vector<std::uint8_t>> payloads;
  while (buffer_.size() >= frame_size_) {
    // Resync: if the head is not a SOF, drop one byte and keep searching.
    if (buffer_[0] != sof_) {
      buffer_.erase(buffer_.begin());
      continue;
    }
    bool valid;
    if (use_crc8_) {
      // CRC covers SOF + payload (frame_size_ - 2 bytes); then check trailing EOF.
      const std::uint8_t expected_crc = crc8(buffer_.data(), frame_size_ - 2);
      valid = buffer_[frame_size_ - 2] == expected_crc && buffer_[frame_size_ - 1] == eof_;
    } else {
      valid = buffer_[frame_size_ - 1] == eof_;
    }
    if (valid) {
      payloads.emplace_back(buffer_.begin() + 1, buffer_.begin() + 1 + payload_size_);
      buffer_.erase(buffer_.begin(), buffer_.begin() + frame_size_);  // consume whole frame
    } else {
      // False SOF or corrupted frame: slide one byte forward and resync.
      buffer_.erase(buffer_.begin());
    }
  }
  return payloads;
}

}  // namespace sim2real
