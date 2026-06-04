#include "sim2real/config.hpp"
#include "sim2real/controller.hpp"
#include "sim2real/protocol.hpp"
#include "test_util.hpp"

#ifndef SIM2REAL_CONFIG_FILE
#error "SIM2REAL_CONFIG_FILE must be defined by the build"
#endif

namespace {

// A raw state with distinct values so each scaled slot is checkable.
sim2real::UplinkState makeState() {
  sim2real::UplinkState s;
  s.base_ang_vel = {0.1f, -0.2f, 0.3f};
  s.projected_gravity = {0.0f, 0.0f, -1.0f};
  s.commands = {0.8f, 0.1f, 0.18f};
  s.theta0 = {-0.05f, 0.05f};
  s.theta0_dot = {0.1f, -0.1f};
  s.l0 = {0.18f, 0.18f};
  s.l0_dot = {0.0f, 0.0f};
  s.wheel_pos = {1.2f, -3.4f};
  s.wheel_vel = {5.0f, -5.0f};
  return s;
}

}  // namespace

int main() {
  using namespace sim2real;
  using sim2real_test::check;
  using sim2real_test::nearFloat;

  const Config cfg = Config::load(SIM2REAL_CONFIG_FILE);

  // --- Exact observation values vs hand-computed scaling ---
  {
    Sim2RealController ctrl(cfg);
    const Action last = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    ctrl.setLastAction(last);
    const Observation obs = ctrl.observe(makeState());

    // base_ang_vel * 0.25
    check(nearFloat(obs[0], 0.025f) && nearFloat(obs[1], -0.05f) && nearFloat(obs[2], 0.075f),
          "obs[0:3] = base_ang_vel * ang_vel");
    // projected_gravity unscaled
    check(nearFloat(obs[3], 0.0f) && nearFloat(obs[5], -1.0f), "obs[3:6] = projected_gravity");
    // commands * [2, 0.25, 5]
    check(nearFloat(obs[6], 1.6f) && nearFloat(obs[7], 0.025f) && nearFloat(obs[8], 0.9f),
          "obs[6:9] = commands * commands_scale");
    // theta0 * 1.0
    check(nearFloat(obs[9], -0.05f) && nearFloat(obs[10], 0.05f), "obs[9:11] = theta0");
    // theta0_dot * 0.05
    check(nearFloat(obs[11], 0.005f) && nearFloat(obs[12], -0.005f), "obs[11:13] = theta0_dot * dof_vel");
    // L0 * 4.5
    check(nearFloat(obs[13], 0.81f) && nearFloat(obs[14], 0.81f), "obs[13:15] = L0 * l0");
    // L0_dot * 0.25
    check(nearFloat(obs[15], 0.0f) && nearFloat(obs[16], 0.0f), "obs[15:17] = L0_dot * l0_dot");
    // wheel_pos * 1.0
    check(nearFloat(obs[17], 1.2f) && nearFloat(obs[18], -3.4f), "obs[17:19] = wheel_pos");
    // wheel_vel * 0.05
    check(nearFloat(obs[19], 0.25f) && nearFloat(obs[20], -0.25f), "obs[19:21] = wheel_vel * dof_vel");
    // last_action unscaled
    check(nearFloat(obs[21], 0.1f) && nearFloat(obs[26], 0.6f), "obs[21:27] = last_action");
  }

  // --- History: first frame tiled, then slide ---
  {
    Sim2RealController ctrl(cfg);
    const Observation obs1 = ctrl.observe(makeState());
    const std::vector<float>& h1 = ctrl.obsHistory();
    check(h1.size() == NUM_ENCODER_OBS, "history size = 135");
    bool tiled = true;
    for (std::size_t f = 0; f < HISTORY_LENGTH; ++f) {
      for (std::size_t i = 0; i < NUM_OBS; ++i) {
        if (!nearFloat(h1[f * NUM_OBS + i], obs1[i])) tiled = false;
      }
    }
    check(tiled, "first frame fills history by repetition");

    // Change last_action so the next obs differs, then slide.
    ctrl.setLastAction({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const Observation obs2 = ctrl.observe(makeState());
    const std::vector<float>& h2 = ctrl.obsHistory();
    check(h2.size() == NUM_ENCODER_OBS, "history size stays 135 after slide");
    bool oldest_is_obs1 = true, newest_is_obs2 = true;
    for (std::size_t i = 0; i < NUM_OBS; ++i) {
      if (!nearFloat(h2[i], obs1[i])) oldest_is_obs1 = false;
      if (!nearFloat(h2[(HISTORY_LENGTH - 1) * NUM_OBS + i], obs2[i])) newest_is_obs2 = false;
    }
    check(oldest_is_obs1, "slide keeps obs1 as oldest frame");
    check(newest_is_obs2, "slide appends obs2 as newest frame");
    check(!nearFloat(obs2[21], obs1[21]), "last_action change propagates into obs");
  }

  // --- Clipping ---
  {
    Sim2RealController ctrl(cfg);
    UplinkState s = makeState();
    s.base_ang_vel[0] = 1000.0f;  // * 0.25 = 250 -> clipped to 100
    const Observation obs = ctrl.observe(s);
    check(nearFloat(obs[0], cfg.clipping.observations), "observation clipped to bound");
  }

  return sim2real_test::report("test_controller");
}
