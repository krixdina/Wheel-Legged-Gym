# 6-dim wheel-legged action, used for BOTH the raw policy output and the scaled
# physical command (one layout serves both, per the sim2real downlink order
# [L_theta, L_l0, L_wheel, R_theta, R_l0, R_wheel]).
#   raw    : unitless clipped policy output.
#   scaled : theta -> theta0_ref [rad], l0 -> l0_ref [m], wheel -> wheel_vel_ref [rad/s].
# The publisher distinguishes the two by topic (action_raw vs action_scaled), so
# the same message type can carry either; header.stamp lets time-series tools
# align them with the matching state.
std_msgs/Header header
float32 left_theta
float32 left_l0
float32 left_wheel
float32 right_theta
float32 right_l0
float32 right_wheel
