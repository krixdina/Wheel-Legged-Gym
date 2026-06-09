cmake_minimum_required(VERSION 3.8)
project(wheel_legged_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate the C++/Python interfaces for the two debug messages. std_msgs is a
# dependency because both messages carry a std_msgs/Header for timestamping.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DebugState.msg"
  "msg/DebugAction.msg"
  DEPENDENCIES std_msgs
)

ament_package()
