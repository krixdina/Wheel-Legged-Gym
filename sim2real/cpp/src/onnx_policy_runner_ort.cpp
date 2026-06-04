// Real policy backend, compiled only when CMake finds ONNX Runtime.
//
// Loads a single merged ONNX graph (inputs "obs"/"obs_history", output "action")
// once, then reuses fixed input buffers each step so the control loop does not
// allocate or rebuild the session. CPU execution, single-threaded ops for low,
// predictable latency.
#include "sim2real/onnx_policy_runner.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace sim2real {

struct OnnxPolicyRunner::Impl {
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;
  Ort::MemoryInfo memory_info;
  std::string obs_input_name;
  std::string hist_input_name;
  std::string output_name;
  // Reusable input buffers; ONNX tensors wrap these without copying.
  std::array<float, NUM_OBS> obs_buf{};
  std::array<float, NUM_ENCODER_OBS> hist_buf{};

  explicit Impl(const std::string& model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "sim2real_policy"),
        session(nullptr),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, model_path.c_str(), session_options);

    // Resolve input/output names by querying the model, so the C++ side matches
    // whatever export_onnx.py produced. The two inputs must be named "obs" and
    // "obs_history"; the first output is taken as the action.
    Ort::AllocatorWithDefaultOptions allocator;
    std::string found;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
      const std::string name = session.GetInputNameAllocated(i, allocator).get();
      found += (found.empty() ? "" : ", ") + name;
      if (name == "obs") {
        obs_input_name = name;
      } else if (name == "obs_history") {
        hist_input_name = name;
      }
    }
    if (obs_input_name.empty() || hist_input_name.empty()) {
      throw std::runtime_error(
          "OnnxPolicyRunner: model must have inputs named 'obs' and 'obs_history'; "
          "found [" + found + "]. Re-export with sim2real/python/export_onnx.py.");
    }
    output_name = session.GetOutputNameAllocated(0, allocator).get();
  }

  Action run(const Observation& obs, const std::vector<float>& obs_history) {
    if (obs_history.size() != NUM_ENCODER_OBS) {
      throw std::invalid_argument("OnnxPolicyRunner: obs_history has wrong length");
    }
    std::copy(obs.begin(), obs.end(), obs_buf.begin());
    std::copy(obs_history.begin(), obs_history.end(), hist_buf.begin());

    const std::array<std::int64_t, 2> obs_shape{1, static_cast<std::int64_t>(NUM_OBS)};
    const std::array<std::int64_t, 2> hist_shape{1, static_cast<std::int64_t>(NUM_ENCODER_OBS)};
    Ort::Value inputs[2] = {
        Ort::Value::CreateTensor<float>(memory_info, obs_buf.data(), obs_buf.size(),
                                        obs_shape.data(), obs_shape.size()),
        Ort::Value::CreateTensor<float>(memory_info, hist_buf.data(), hist_buf.size(),
                                        hist_shape.data(), hist_shape.size()),
    };
    const char* input_names[2] = {obs_input_name.c_str(), hist_input_name.c_str()};
    const char* output_names[1] = {output_name.c_str()};

    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs, 2, output_names, 1);
    if (outputs.empty() ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != NUM_ACTIONS) {
      throw std::runtime_error("OnnxPolicyRunner: model output is not a length-6 action");
    }
    const float* data = outputs[0].GetTensorMutableData<float>();
    Action action;
    std::copy(data, data + NUM_ACTIONS, action.begin());
    return action;
  }
};

OnnxPolicyRunner::OnnxPolicyRunner(const Config& config) {
  namespace fs = std::filesystem;
  const std::string& path = config.network.onnx_model_path;
  // Fail loudly and specifically rather than letting ORT throw something opaque.
  if (fs::path(path).extension() != ".onnx") {
    throw std::runtime_error(
        "OnnxPolicyRunner: model path '" + config.network.onnx_model_path_raw +
        "' is not a .onnx file. The C++ policy needs ONNX, not a PyTorch .pt; "
        "export it first with sim2real/python/export_onnx.py.");
  }
  if (!fs::exists(path)) {
    throw std::runtime_error(
        "OnnxPolicyRunner: ONNX model not found at '" + path +
        "'. Export the policy first: python sim2real/python/export_onnx.py.");
  }
  impl_ = std::make_unique<Impl>(path);
}

OnnxPolicyRunner::~OnnxPolicyRunner() = default;

Action OnnxPolicyRunner::act(const Observation& obs, const std::vector<float>& obs_history) {
  return impl_->run(obs, obs_history);
}

}  // namespace sim2real
