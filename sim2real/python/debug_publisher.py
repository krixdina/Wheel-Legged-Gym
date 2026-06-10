"""Optional ROS2 debug publisher for sim2real bring-up (debug-only).

This module is imported ONLY when config.debug is true

    /sim2real_debug/state          (DebugState)  - decoded uplink state (21 raw values)
    /sim2real_debug/action_raw     (DebugAction) - clipped raw policy output
    /sim2real_debug/action_scaled  (DebugAction) - scaled physical downlink command

Non-blocking by construction (the 100 Hz control loop must never stall when using ROS):

publish_step() only enqueues one record per control step and drops it if the queue
is full (debug data is best-effort); This function never calls into rclpy. 
This keeps ROS2/DDS serialization, publish scheduling, and possible blocking away from
the control path; the 100 Hz loop only performs a queue write, so debug visualization cannot slow it down.
The worker thread owns the queue draining and all message publishing. 
"""
import queue
import threading

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from wheel_legged_msgs.msg import DebugAction, DebugState

# ROS2 node / topic names. Kept local: this is the only place that talks to ROS,
# so there is no value in threading them through the shared config.
NODE_NAME = "sim2real_debug"
TOPIC_STATE = "sim2real_debug/state"
TOPIC_ACTION_RAW = "sim2real_debug/action_raw"
TOPIC_ACTION_SCALED = "sim2real_debug/action_scaled"

# Bounded queue: ~2 s of slack at 100 Hz. If the worker ever falls behind, new
# samples are dropped (best-effort debug data) instead of blocking the loop.
QUEUE_MAXSIZE = 200


class DebugPublisher:
    """Publish (state, raw action, scaled action) to ROS2 without blocking the loop."""

    def __init__(self):
        # Best-effort, keep-last QoS: this is high-rate telemetry, so an occasional
        # dropped sample is preferable to reliable-delivery backpressure.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # init the global rclpy context once; tolerate it already being up.
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node(NODE_NAME)
        self._pub_state = self._node.create_publisher(DebugState, TOPIC_STATE, qos)
        self._pub_raw = self._node.create_publisher(DebugAction, TOPIC_ACTION_RAW, qos)
        self._pub_scaled = self._node.create_publisher(DebugAction, TOPIC_ACTION_SCALED, qos)

        self._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        # 线程间同步信号，用于通知工作线程停止运行
        self._stop = threading.Event()
        self._dropped = 0  # samples dropped because the queue was full
        # Daemon so a hung worker can never keep the process alive past shutdown().
        self._thread = threading.Thread(target=self._run, name=NODE_NAME, daemon=True)
        self._thread.start()

    def publish_step(self, state, raw_action, scaled_action):
        """Enqueue one control step's data; non-blocking, drops on a full queue.

        state:         dict of raw uplink arrays from decode_uplink 
        raw_action:    length-6 clipped policy output.
        scaled_action: length-6 scaled physical command
        """
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait((state, raw_action, scaled_action))
        except queue.Full:
            self._dropped += 1

    def shutdown(self):
        """Stop the worker and tear down the node/context. Safe to call once."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._dropped:
            print(f"debug publisher: dropped {self._dropped} samples (queue full)")
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _run(self):
        """Worker loop: drain the queue and publish. The only thread that publishes."""
        while not self._stop.is_set():
            try:
                state, raw_action, scaled_action = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # One shared timestamp for the whole step, so the state and the two
            # actions line up exactly in time-series tools.
            stamp = self._node.get_clock().now().to_msg()
            self._pub_state.publish(self._state_msg(state, stamp))
            self._pub_raw.publish(self._action_msg(raw_action, stamp))
            self._pub_scaled.publish(self._action_msg(scaled_action, stamp))

    @staticmethod
    def _action_msg(action, stamp):
        """length-6 action [L_theta, L_l0, L_wheel, R_theta, R_l0, R_wheel] -> DebugAction."""
        msg = DebugAction()
        msg.header.stamp = stamp
        msg.left_theta = float(action[0])
        msg.left_l0 = float(action[1])
        msg.left_wheel = float(action[2])
        msg.right_theta = float(action[3])
        msg.right_l0 = float(action[4])
        msg.right_wheel = float(action[5])
        return msg

    @staticmethod
    def _state_msg(state, stamp):
        """decode_uplink state dict -> DebugState (field order mirrors the uplink layout)."""
        msg = DebugState()
        msg.header.stamp = stamp
        bav = state["base_ang_vel"]
        msg.base_ang_vel_x, msg.base_ang_vel_y, msg.base_ang_vel_z = map(float, bav)
        pg = state["projected_gravity"]
        msg.projected_gravity_x, msg.projected_gravity_y, msg.projected_gravity_z = map(float, pg)
        cmd = state["commands"]
        msg.command_vx, msg.command_wz, msg.command_height = map(float, cmd)
        msg.theta0_l, msg.theta0_r = map(float, state["theta0"])
        msg.theta0_dot_l, msg.theta0_dot_r = map(float, state["theta0_dot"])
        msg.l0_l, msg.l0_r = map(float, state["L0"])
        msg.l0_dot_l, msg.l0_dot_r = map(float, state["L0_dot"])
        msg.wheel_pos_l, msg.wheel_pos_r = map(float, state["wheel_pos"])
        msg.wheel_vel_l, msg.wheel_vel_r = map(float, state["wheel_vel"])
        return msg
