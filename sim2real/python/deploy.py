"""sim2real deployment main loop (CPU NUC side): the glue between serial and policy.

This is the piece that wires the otherwise-independent layers into one fixed-rate
control loop:

    RobotSerialLink  (serial_comm) -- bytes in/out over the wire
    Sim2RealController (controller) -- raw state -> 27-dim obs, history, last_action
    SequencePolicy     (policy)     -- encoder + actor inference

Per control step (config.control_timing.policy_rate_hz, e.g. 100 Hz):
    1. poll the serial link for the newest uplink state frame
    2. fresh frame  -> build obs, run the policy, clip the RAW action (stored as
       last_action for the next obs), then scale it into a physical command;
       no new frame -> re-send the previous physical command
    3. send the physical command downstream
    4. sleep to hold the loop rate

Lost-frame policy: re-send the last command. If too many consecutive frames are
missed (config.control_timing.max_missed_frames) the link is treated as lost --
the loop stops and sends the neutral physical command (scaled zero action) so the
lower machine is not driven on stale state.

"""
import argparse
import time

import numpy as np
import serial  # pyserial; SerialException is raised when the port cannot be opened

from config import CONFIG
from controller import InvalidUplinkState, Sim2RealController
from policy import SequencePolicy
from serial_comm import RobotSerialLink


def _open_serial_link(reconnect_interval_s):
    """Open the serial link, retrying until the port is available.

    Opening fails (raising serial.SerialException) when the device is not
    present, Instead of crashing with a traceback we print
    a clear "no serial port found" message and retry every reconnect_interval_s
    seconds, so the deployment can be started before the cable is connected and
    recovers on hot-plug. Ctrl-C during the wait exits cleanly.
    """
    port = CONFIG["serial"]["port"]
    while True:
        try:
            return RobotSerialLink()
        except serial.SerialException:
            print(f"无法找到串口 {port}，{reconnect_interval_s:g}s 后重试…（Ctrl-C 退出）")
            time.sleep(reconnect_interval_s)


def _make_debug_publisher():
    """Construct the ROS2 debug publisher, or return None if debug is off.

    rclpy (and the custom messages) are imported here, lazily, so that a normal
    deployment with debug=false never needs ROS2 and keeps running under Python
    3.7 / isaac_gym. With debug=true this requires a Python 3.10 environment with
    ROS2 Humble and wheel_legged_msgs sourced; the import error is left to
    propagate so the missing setup is reported clearly instead of silently
    falling back.
    """
    if not CONFIG.get("debug", False):
        return None
    from debug_publisher import DebugPublisher

    publisher = DebugPublisher()
    print("debug=true: publishing state/action_raw/action_scaled to ROS2 (/sim2real_debug/*)")
    return publisher


def run(device="cpu"):
    num_actions = CONFIG["network"]["num_actions"]
    clip_action = CONFIG["clipping"]["actions"]
    timing = CONFIG["control_timing"]
    period = 1.0 / timing["policy_rate_hz"]
    # Consecutive missed uplink frames tolerated before the link is declared lost;
    # beyond this, re-sending the stale action is unsafe, so the loop stops and zeros.
    max_missed_frames = timing["max_missed_frames"]

    controller = Sim2RealController()
    policy = SequencePolicy(device=device)
    # Optional ROS2 debug publishing; None unless config.debug is true.
    debug_publisher = _make_debug_publisher()
    # Wait for the serial port instead of crashing if it is not present yet.
    try:
        link = _open_serial_link(timing["serial_reconnect_interval_s"])
    except KeyboardInterrupt:
        print("\nstopping: interrupted while waiting for serial port")
        if debug_publisher is not None:
            debug_publisher.shutdown()
        return

    # Physical "neutral" command = scaled zero action (legs at default length l0_offset, wheels stopped).
    neutral_action = controller.scale_action(np.zeros(num_actions, dtype=np.float32))
    action = neutral_action.copy()  # physical command re-sent on a missed frame
    missed = 0
    invalid_frames = 0

    frame_cfg = CONFIG["frame"]
    print(
        f"sim2real deploy: {1.0 / period:.0f} Hz, device={device}, "
        f"port={CONFIG['serial']['port']}, crc8={frame_cfg['use_crc8']}. Ctrl-C to stop."
    )
    try:
        while True:
            t0 = time.time()

            state = link.poll()
            if state is not None:
                try:
                    obs, obs_history = controller.observe(state)
                except InvalidUplinkState as exc:
                    invalid_frames += 1
                    missed += 1
                    if invalid_frames <= 5 or invalid_frames % 50 == 0:
                        print(f"invalid uplink frame dropped ({invalid_frames}): {exc}")
                    if missed >= max_missed_frames:
                        raise TimeoutError(
                            f"no valid uplink frame for {missed} steps; "
                            "check frame length/CRC/SOF/EOF and payload field order"
                        )
                else:
                    missed = 0
                    # Clip the RAW network output, store it as last_action (it feeds the next observation),
                    # then scale it into the physical downlink command.
                    raw_action = np.clip(policy.act(obs, obs_history), -clip_action, clip_action)
                    controller.set_last_action(raw_action)
                    action = controller.scale_action(raw_action)
                    # Debug only: hand this step's state and both actions to ROS2.
                    # Non-blocking (enqueue + drop-on-full); no-op when debug is off.
                    if debug_publisher is not None:
                        debug_publisher.publish_step(state, raw_action, action)
            else:
                # No fresh frame this step: re-send the last physical command.
                missed += 1
                if missed >= max_missed_frames:
                    raise TimeoutError(
                        f"no uplink frame for {missed} steps; "
                        "serial bytes may still be present, but no frame matched the configured protocol"
                    )

            link.send_action(action)

            # Hold the control rate.
            sleep_left = period - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except (KeyboardInterrupt, TimeoutError) as exc:
        print(f"\nstopping: {exc if str(exc) else 'interrupted'}")
    finally:
        # Always leave the robot with the neutral physical command, then close.
        link.send_action(neutral_action)
        link.close()
        if debug_publisher is not None:
            debug_publisher.shutdown()
        print("sent neutral action and closed serial port.")


def parse_args():
    p = argparse.ArgumentParser(description="sim2real deployment loop")
    p.add_argument("--device", default="cpu", help="torch device for inference")
    return p.parse_args()


def main():
    run(parse_args().device)


if __name__ == "__main__":
    main()
