#ifndef SIM2REAL_FRAME_CODEC_HPP
#define SIM2REAL_FRAME_CODEC_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sim2real/config.hpp"

namespace sim2real {

// Fixed-length frame [SOF][payload][CRC8(optional)][EOF] with a resync parser.
// Faithful port of the Python FrameCodec: a single serial read may split or merge
// frames, so feed() buffers bytes, finds the SOF, validates CRC8/EOF, and slides
// one byte forward on any mismatch until it resynchronises onto a real frame.
class FrameCodec {
 public:
  // payload_size: number of business bytes between SOF and the trailing CRC/EOF.
  // frame_cfg:    SOF/EOF markers and CRC settings from the shared config.
  FrameCodec(std::size_t payload_size, const FrameConfig& frame_cfg);

  // Wrap a payload into one complete frame. When CRC is enabled the CRC8 covers
  // SOF + payload, matching the Python encoder.
  // payload: exactly payload_size bytes, else std::invalid_argument.
  std::vector<std::uint8_t> encode(const std::vector<std::uint8_t>& payload) const;

  // Append a chunk of raw bytes and return every complete, valid payload it now
  // contains. Partial trailing bytes are kept buffered for the next call.
  std::vector<std::vector<std::uint8_t>> feed(const std::vector<std::uint8_t>& chunk);

 private:
  // CRC-8 over data[0..len), using the configured polynomial and initial value.
  std::uint8_t crc8(const std::uint8_t* data, std::size_t len) const;

  std::size_t payload_size_;
  std::uint8_t sof_;
  std::uint8_t eof_;
  bool use_crc8_;
  std::uint8_t crc_poly_;
  std::uint8_t crc_init_;
  std::size_t frame_size_;  // payload_size_ + (use_crc8_ ? 3 : 2)
  std::vector<std::uint8_t> buffer_;
};

}  // namespace sim2real

#endif  // SIM2REAL_FRAME_CODEC_HPP
