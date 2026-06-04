// sim2real deployment main loop (CPU NUC side): the glue between serial and policy.
//
// Per control step (config.control_timing.policy_rate_hz):
//   1. poll the serial link for the newest uplink state frame
//   2. fresh frame  -> build obs/history, run policy, clip action, store last_action
//      no new frame -> re-send the previous action (keeps the lower machine fed)
//   3. send the action; sleep to hold the loop rate
// If max_missed_frames consecutive frames are lost the link is declared lost and
// the loop stops. On any exit (Ctrl-C, link loss, error) a zero action is sent
// and the port is closed, so the robot is never left driven on stale state.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "sim2real/config.hpp"
#include "sim2real/controller.hpp"
#include "sim2real/onnx_policy_runner.hpp"
#include "sim2real/protocol.hpp"
#include "sim2real/robot_serial_link.hpp"

namespace {

// Set by SIGINT/SIGTERM so the control loop can exit cleanly and still send a
// zero action on the way out.
std::atomic<bool> g_stop{false};
void onSignal(int /*signum*/) { g_stop = true; }

std::string parseConfigPath(int argc, char** argv) {
  // Default mirrors the Python convention of running from the repo root.
  std::string path = "sim2real/config/sim2real.yaml";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      path = argv[++i];
    }
  }
  return path;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace sim2real;

  const std::string config_path = parseConfigPath(argc, argv);

  Config config;
  try {
    config = Config::load(config_path);
  } catch (const std::exception& e) {
    std::cerr << "config error: " << e.what() << "\n";
    return 1;
  }

  const std::size_t max_missed_frames = static_cast<std::size_t>(config.control_timing.max_missed_frames);
  const auto period = std::chrono::duration<double>(1.0 / config.control_timing.policy_rate_hz);

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  // Construct the policy first: this is where a missing ONNX model or a build
  // without ONNX Runtime surfaces, and we want a clear error before opening the
  // serial port.
  std::unique_ptr<OnnxPolicyRunner> policy;
  try {
    policy = std::make_unique<OnnxPolicyRunner>(config);
  } catch (const std::exception& e) {
    std::cerr << "policy error: " << e.what() << "\n";
    return 1;
  }

  std::unique_ptr<RobotSerialLink> link;
  try {
    link = std::make_unique<RobotSerialLink>(config);
  } catch (const std::exception& e) {
    std::cerr << "serial error: " << e.what() << "\n";
    return 1;
  }

  Sim2RealController controller(config);
  Action action{};  // last action, re-sent on a missed frame; starts at zero
  std::size_t missed = 0;

  std::cout << "sim2real deploy: " << config.control_timing.policy_rate_hz << " Hz, port "
            << config.serial.port << ", model " << config.network.onnx_model_path << ". Ctrl-C to stop.\n";

  std::string stop_reason = "interrupted";
  try {
    while (!g_stop) {
      const auto t0 = std::chrono::steady_clock::now();

      if (const std::optional<UplinkState> state = link->poll()) {
        missed = 0;
        const Observation obs = controller.observe(*state);
        Action raw = policy->act(obs, controller.obsHistory());
        for (float& a : raw) {
          a = std::clamp(a, -config.clipping.actions, config.clipping.actions);
        }
        action = raw;
        controller.setLastAction(action);
      } else {
        // No fresh frame this step: re-send the last action.
        if (++missed >= max_missed_frames) {
          stop_reason = "no uplink frame for " + std::to_string(missed) + " steps; link lost";
          break;
        }
      }

      link->sendAction(action);

      // Hold the control rate.
      const auto sleep_left = period - (std::chrono::steady_clock::now() - t0);
      if (sleep_left.count() > 0) {
        std::this_thread::sleep_for(sleep_left);
      }
    }
  } catch (const std::exception& e) {
    stop_reason = std::string("error: ") + e.what();
  }

  std::cout << "\nstopping: " << stop_reason << "\n";
  // Always leave the robot with a zero command, then close the port.
  try {
    link->sendAction(Action{});
  } catch (const std::exception& e) {
    std::cerr << "warning: failed to send zero action: " << e.what() << "\n";
  }
  link->close();
  std::cout << "sent zero action and closed serial port.\n";
  return 0;
}
