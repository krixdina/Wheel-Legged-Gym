# Locate ONNX Runtime (C/C++ API).
#
# Honours the ONNXRUNTIME_ROOT cache/environment variable and common install
# locations. On success defines:
#   ONNXRuntime_FOUND, ONNXRuntime_INCLUDE_DIRS, ONNXRuntime_LIBRARIES
#   and the imported target onnxruntime::onnxruntime
#
# Install hint (Linux x64): download the release tarball from
#   https://github.com/microsoft/onnxruntime/releases
# and either install under /usr/local or pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime.

find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
  PATHS /usr/local /usr /opt/onnxruntime
)

find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib lib64
  PATHS /usr/local /usr /opt/onnxruntime
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)

if(ONNXRuntime_FOUND)
  set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
  set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
