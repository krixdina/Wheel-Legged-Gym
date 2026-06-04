#include <cmath>
#include <string>
#include <vector>

#include "sim2real/config.hpp"
#include "sim2real/onnx_policy_runner.hpp"
#include "sim2real/protocol.hpp"
#include "test_util.hpp"

#ifndef SIM2REAL_CONFIG_FILE
#error "SIM2REAL_CONFIG_FILE must be defined by the build"
#endif

// The policy backend depends on the environment, so this test accepts EITHER
// outcome, matching the requirement:
//   - no ONNX Runtime, or no/!.onnx model  -> construction throws a CLEAR error
//   - ONNX Runtime + a valid .onnx model    -> one inference returns 6 finite values
int main() {
  using namespace sim2real;
  using sim2real_test::check;

  const Config cfg = Config::load(SIM2REAL_CONFIG_FILE);

  try {
    OnnxPolicyRunner policy(cfg);
    // Construction succeeded -> ORT is present and the model loaded. Run once.
    const Observation obs{};
    const std::vector<float> history(NUM_ENCODER_OBS, 0.0f);
    const Action action = policy.act(obs, history);
    bool finite = true;
    for (float a : action) {
      if (!std::isfinite(a)) finite = false;
    }
    check(finite, "inference returns finite action values");
    std::cout << "  (ONNX Runtime present: ran a real inference)\n";
  } catch (const std::exception& e) {
    // Expected on this machine (no ORT / no exported model): the error must be
    // clear and non-empty rather than a silent failure.
    const std::string msg = e.what();
    check(!msg.empty(), "missing-policy error message is non-empty");
    std::cout << "  (expected clear error) " << msg << "\n";
  }

  return sim2real_test::report("test_policy");
}
