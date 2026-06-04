cmake_minimum_required(VERSION 3.16)
project(sim2real_cpp LANGUAGES CXX)

# Export compile_commands.json so clangd / VSCode can resolve headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# --- yaml-cpp (config loading) via pkg-config: robust to target-name changes ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# Absolute path to the shared config yaml, injected into the test binaries so
# they do not depend on the working directory.
get_filename_component(SIM2REAL_CONFIG_YAML
  "${CMAKE_CURRENT_SOURCE_DIR}/../config/sim2real.yaml" ABSOLUTE)

# --- core library: config + protocol + framing + serial + controller ---
add_library(sim2real_core STATIC
  src/config.cpp
  src/protocol.cpp
  src/frame_codec.cpp
  src/serial_transport.cpp
  src/robot_serial_link.cpp
  src/controller.cpp
)
target_include_directories(sim2real_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${YAMLCPP_INCLUDE_DIRS}
)
target_link_directories(sim2real_core PUBLIC ${YAMLCPP_LIBRARY_DIRS})
target_link_libraries(sim2real_core PUBLIC ${YAMLCPP_LIBRARIES})

# --- policy library: ONNX Runtime backend, with a graceful stub fallback ---
# If ONNX Runtime is available we build the real backend; otherwise we build a
# stub that throws a clear, actionable error at runtime (so everything else still
# builds and runs on a machine without ORT).
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(ONNXRuntime QUIET)

add_library(sim2real_policy STATIC "")
target_link_libraries(sim2real_policy PUBLIC sim2real_core)

if(ONNXRuntime_FOUND)
  message(STATUS "ONNX Runtime found (${ONNXRuntime_LIBRARY}): building real policy backend")
  target_sources(sim2real_policy PRIVATE src/onnx_policy_runner_ort.cpp)
  target_link_libraries(sim2real_policy PUBLIC onnxruntime::onnxruntime)
else()
  message(WARNING
    "ONNX Runtime NOT found: building a policy STUB that errors at runtime. "
    "Install ONNX Runtime (or pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime) and "
    "reconfigure to enable C++ inference.")
  target_sources(sim2real_policy PRIVATE src/onnx_policy_runner_stub.cpp)
endif()

# --- deploy main program ---
add_executable(deploy src/deploy.cpp)
target_link_libraries(deploy PRIVATE sim2real_core sim2real_policy)

# --- tests ---
enable_testing()

# Helper: build a test that links the core and knows the config path.
function(add_sim2real_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE sim2real_core)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_definitions(${name} PRIVATE SIM2REAL_CONFIG_FILE="${SIM2REAL_CONFIG_YAML}")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_sim2real_test(test_config)
add_sim2real_test(test_frame_codec)
add_sim2real_test(test_controller)
add_sim2real_test(test_serial_link)

# test_policy also needs the policy library (ORT backend or stub).
add_executable(test_policy tests/test_policy.cpp)
target_link_libraries(test_policy PRIVATE sim2real_core sim2real_policy)
target_include_directories(test_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_definitions(test_policy PRIVATE SIM2REAL_CONFIG_FILE="${SIM2REAL_CONFIG_YAML}")
add_test(NAME test_policy COMMAND test_policy)
