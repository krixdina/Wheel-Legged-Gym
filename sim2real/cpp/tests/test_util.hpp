#ifndef SIM2REAL_TEST_UTIL_HPP
#define SIM2REAL_TEST_UTIL_HPP

#include <cmath>
#include <iostream>
#include <string>

// Minimal, dependency-free test helpers (no test framework, no macros beyond the
// include guard) so the test binaries stay light on a NUC. Each test main runs
// checks then returns report(name): 0 if all passed, 1 otherwise.
namespace sim2real_test {

inline int& failureCount() {
  static int count = 0;
  return count;
}

inline void check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "  FAIL: " << what << "\n";
    ++failureCount();
  }
}

inline bool nearFloat(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }

inline int report(const std::string& name) {
  if (failureCount() == 0) {
    std::cout << "[PASS] " << name << "\n";
    return 0;
  }
  std::cout << "[FAIL] " << name << " (" << failureCount() << " failure(s))\n";
  return 1;
}

}  // namespace sim2real_test

#endif  // SIM2REAL_TEST_UTIL_HPP
