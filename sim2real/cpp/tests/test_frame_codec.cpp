#include <cstdint>
#include <vector>

#include "sim2real/config.hpp"
#include "sim2real/frame_codec.hpp"
#include "test_util.hpp"

namespace {

using sim2real::FrameCodec;
using sim2real::FrameConfig;

FrameConfig makeFrameConfig(bool use_crc8) {
  FrameConfig f;
  f.sof = 0xFF;
  f.eof = 0x0D;
  f.use_crc8 = use_crc8;
  f.crc8_poly = 0x07;
  f.crc8_init = 0x00;
  return f;
}

std::vector<std::uint8_t> samplePayload(std::size_t n) {
  std::vector<std::uint8_t> p(n);
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = static_cast<std::uint8_t>((i * 7 + 1) & 0xFF);  // deterministic, includes 0xFF/0x0D
  }
  return p;
}

}  // namespace

int main() {
  using sim2real_test::check;
  constexpr std::size_t kPayload = 8;
  const std::vector<std::uint8_t> payload = samplePayload(kPayload);

  // --- CRC enabled: round trip ---
  {
    FrameCodec codec(kPayload, makeFrameConfig(true));
    const std::vector<std::uint8_t> frame = codec.encode(payload);
    check(frame.size() == kPayload + 3, "crc-on frame size = payload + 3");
    const auto out = codec.feed(frame);
    check(out.size() == 1, "crc-on: one payload decoded");
    check(out.size() == 1 && out[0] == payload, "crc-on: payload round-trips");
  }

  // --- CRC disabled: frame is one byte shorter, no checksum ---
  {
    FrameCodec codec(kPayload, makeFrameConfig(false));
    const std::vector<std::uint8_t> frame = codec.encode(payload);
    check(frame.size() == kPayload + 2, "crc-off frame size = payload + 2");
    const auto out = codec.feed(frame);
    check(out.size() == 1 && out[0] == payload, "crc-off: payload round-trips");
  }

  // --- Half-frame reassembly: feed the frame in 3-byte slices ---
  {
    FrameCodec codec(kPayload, makeFrameConfig(true));
    const std::vector<std::uint8_t> frame = FrameCodec(kPayload, makeFrameConfig(true)).encode(payload);
    std::size_t decoded = 0;
    std::vector<std::uint8_t> last;
    for (std::size_t i = 0; i < frame.size(); i += 3) {
      const std::vector<std::uint8_t> slice(frame.begin() + i,
                                            frame.begin() + std::min(i + 3, frame.size()));
      for (auto& p : codec.feed(slice)) {
        ++decoded;
        last = p;
      }
    }
    check(decoded == 1, "reassembly: exactly one frame across slices");
    check(last == payload, "reassembly: payload correct");
  }

  // --- Multiple frames in one feed ---
  {
    FrameCodec codec(kPayload, makeFrameConfig(true));
    FrameCodec enc(kPayload, makeFrameConfig(true));
    std::vector<std::uint8_t> two = enc.encode(payload);
    const std::vector<std::uint8_t> second = FrameCodec(kPayload, makeFrameConfig(true)).encode(payload);
    two.insert(two.end(), second.begin(), second.end());
    const auto out = codec.feed(two);
    check(out.size() == 2, "two frames in one feed");
  }

  // --- Noise bytes before a frame: must resync and still decode ---
  {
    FrameCodec codec(kPayload, makeFrameConfig(true));
    std::vector<std::uint8_t> noisy = {0x00, 0xFF, 0x12, 0x34};  // junk incl. a false SOF
    const std::vector<std::uint8_t> frame = FrameCodec(kPayload, makeFrameConfig(true)).encode(payload);
    noisy.insert(noisy.end(), frame.begin(), frame.end());
    const auto out = codec.feed(noisy);
    check(out.size() == 1 && out[0] == payload, "resync past noise + false SOF");
  }

  // --- Corrupted CRC: frame must be rejected ---
  {
    FrameCodec codec(kPayload, makeFrameConfig(true));
    std::vector<std::uint8_t> frame = FrameCodec(kPayload, makeFrameConfig(true)).encode(payload);
    frame[3] ^= 0xFF;  // flip a payload byte; CRC no longer matches
    const auto out = codec.feed(frame);
    check(out.empty(), "corrupted CRC frame is dropped");
  }

  return sim2real_test::report("test_frame_codec");
}
