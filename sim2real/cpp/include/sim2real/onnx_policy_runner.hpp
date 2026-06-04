#ifndef SIM2REAL_ONNX_POLICY_RUNNER_HPP
#define SIM2REAL_ONNX_POLICY_RUNNER_HPP

#include <memory>
#include <vector>

#include "sim2real/config.hpp"
#include "sim2real/protocol.hpp"

namespace sim2real {

// CPU policy inference backed by ONNX Runtime, the C++ counterpart of the Python
// SequencePolicy. The implementation is hidden behind a pImpl so this header does
// not pull in ONNX Runtime: a build without ORT compiles a stub implementation
// that throws a clear, actionable error instead of failing silently.
//
// The ONNX model is expected to be a single graph with two float inputs named
// "obs" [1, NUM_OBS] and "obs_history" [1, NUM_ENCODER_OBS] and one float output
// "action" [1, NUM_ACTIONS] -- exactly what sim2real/python/export_onnx.py emits.
class OnnxPolicyRunner {
 public:
  // Loads the model named by config.network.onnx_model_path.
  // Throws std::runtime_error if the model is missing, is not a .onnx file, or
  // if this binary was built without ONNX Runtime.
  explicit OnnxPolicyRunner(const Config& config);
  ~OnnxPolicyRunner();

  OnnxPolicyRunner(const OnnxPolicyRunner&) = delete;
  OnnxPolicyRunner& operator=(const OnnxPolicyRunner&) = delete;

  // Run one inference: latent = encoder(obs_history); action = actor([obs, latent]).
  // obs:         the 27-dim observation.
  // obs_history: the 135-dim history window.
  // return:      the 6-dim policy action (pre-clip; the caller clips).
  Action act(const Observation& obs, const std::vector<float>& obs_history);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sim2real

#endif  // SIM2REAL_ONNX_POLICY_RUNNER_HPP
