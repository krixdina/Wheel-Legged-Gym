// Stub policy backend, compiled when the build cannot find ONNX Runtime.
//
// It keeps deploy.cpp and the tests buildable on a machine without ORT, but any
// attempt to construct the runner fails immediately with an actionable message
// rather than silently doing nothing -- exactly the "no silent failure" the
// deployment requires.
#include "sim2real/onnx_policy_runner.hpp"

#include <stdexcept>

namespace sim2real {

struct OnnxPolicyRunner::Impl {};

OnnxPolicyRunner::OnnxPolicyRunner(const Config& /*config*/) {
  throw std::runtime_error(
      "OnnxPolicyRunner: this binary was built WITHOUT ONNX Runtime, so policy "
      "inference is unavailable. Install ONNX Runtime (or pass "
      "-DONNXRUNTIME_ROOT=/path/to/onnxruntime) and reconfigure CMake; see "
      "sim2real/cpp/README.md.");
}

OnnxPolicyRunner::~OnnxPolicyRunner() = default;

Action OnnxPolicyRunner::act(const Observation& /*obs*/, const std::vector<float>& /*obs_history*/) {
  // Unreachable: construction always throws in this build.
  throw std::runtime_error("OnnxPolicyRunner: built without ONNX Runtime");
}

}  // namespace sim2real
