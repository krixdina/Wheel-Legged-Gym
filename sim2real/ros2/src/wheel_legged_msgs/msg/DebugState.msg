# All 21 raw (unscaled, SI) uplink state quantities the lower machine sends to the
# NUC, gathered into one message. Field names mirror the sim2sim reference-trace
# tags (see sim2sim/reference_data/collect_reference_data.py) so a live serial
# trace can be compared against the sim2sim baseline field-for-field.
std_msgs/Header header

# body angular velocity [rad/s] (roll/pitch/yaw rates)
float32 base_ang_vel_x
float32 base_ang_vel_y
float32 base_ang_vel_z

# gravity unit vector projected into the base frame (z ~ -1 when upright)
float32 projected_gravity_x
float32 projected_gravity_y
float32 projected_gravity_z

# motion command: forward velocity [m/s], yaw rate [rad/s], body height [m]
float32 command_vx
float32 command_wz
float32 command_height

# virtual-leg swing angle [rad] (left / right)
float32 theta0_l
float32 theta0_r

# virtual-leg swing angular velocity [rad/s]
float32 theta0_dot_l
float32 theta0_dot_r

# virtual-leg length [m]
float32 l0_l
float32 l0_r

# virtual-leg length rate [m/s]
float32 l0_dot_l
float32 l0_dot_r

# wheel joint position [rad]
float32 wheel_pos_l
float32 wheel_pos_r

# wheel joint velocity [rad/s]
float32 wheel_vel_l
float32 wheel_vel_r
