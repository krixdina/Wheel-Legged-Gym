"""Publish fake sim2real debug topics without serial hardware.

This script is for upper-computer ROS2 debug bring-up before connecting the lower
machine. It generates deterministic, physically plausible samples with the same
field layout as the sim2real uplink and action topics, then feeds them through
DebugPublisher exactly like deploy.py does.
"""
import argparse
import math
import time

import numpy as np

from config import CONFIG
from controller import Sim2RealController


def parse_args():
    parser = argparse.ArgumentParser(description="Publish fake sim2real ROS2 debug data")
    parser.add_argument("--seconds", type=float, default=10.0, help="publish duration")
    parser.add_argument("--rate", type=float, default=None, help="publish rate; defaults to policy_rate_hz")
    parser.add_argument("--vx", type=float, default=0.5, help="forward velocity command [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="yaw rate command [rad/s]")
    parser.add_argument("--height", type=float, default=0.18, help="body height command [m]")
    return parser.parse_args()


def _gravity(t):
    gravity = np.array(
        [
            0.035 * math.sin(2.0 * math.pi * 0.35 * t),
            0.025 * math.cos(2.0 * math.pi * 0.27 * t),
            -1.0,
        ],
        dtype=np.float32,
    )
    return gravity / np.linalg.norm(gravity)


def _fake_state(t, args):
    """Build one 21-value raw uplink state, using sim2sim-like nominal ranges."""
    phase = 2.0 * math.pi * 1.2 * t
    wheel_radius = 0.0579
    wheel_speed = args.vx / wheel_radius if wheel_radius else 0.0
    return {
        "base_ang_vel": np.array(
            [
                0.04 * math.sin(phase),
                0.03 * math.cos(0.7 * phase),
                args.wz + 0.02 * math.sin(0.5 * phase),
            ],
            dtype=np.float32,
        ),
        "projected_gravity": _gravity(t),
        "commands": np.array([args.vx, args.wz, args.height], dtype=np.float32),
        "theta0": np.array(
            [
                -0.13 + 0.015 * math.sin(phase),
                -0.13 + 0.015 * math.sin(phase + math.pi),
            ],
            dtype=np.float32,
        ),
        "theta0_dot": np.array(
            [
                0.015 * 2.0 * math.pi * 1.2 * math.cos(phase),
                0.015 * 2.0 * math.pi * 1.2 * math.cos(phase + math.pi),
            ],
            dtype=np.float32,
        ),
        "L0": np.array(
            [
                0.20 + 0.01 * math.sin(phase + 0.4),
                0.20 + 0.01 * math.sin(phase + math.pi + 0.4),
            ],
            dtype=np.float32,
        ),
        "L0_dot": np.array(
            [
                0.01 * 2.0 * math.pi * 1.2 * math.cos(phase + 0.4),
                0.01 * 2.0 * math.pi * 1.2 * math.cos(phase + math.pi + 0.4),
            ],
            dtype=np.float32,
        ),
        "wheel_pos": np.array([wheel_speed * t, wheel_speed * t], dtype=np.float32),
        "wheel_vel": np.array([wheel_speed, wheel_speed], dtype=np.float32),
    }


def _fake_raw_action(t):
    """Build a small clipped raw policy action in sim2real downlink order."""
    phase = 2.0 * math.pi * 1.2 * t
    return np.array(
        [
            -0.02 + 0.04 * math.sin(phase),
            0.03 * math.cos(phase),
            0.10 * math.sin(0.5 * phase),
            -0.02 + 0.04 * math.sin(phase + math.pi),
            0.03 * math.cos(phase + math.pi),
            0.10 * math.sin(0.5 * phase + math.pi),
        ],
        dtype=np.float32,
    )


def main():
    args = parse_args()
    rate = args.rate if args.rate is not None else CONFIG["control_timing"]["policy_rate_hz"]
    period = 1.0 / rate
    steps = int(args.seconds * rate)
    controller = Sim2RealController()

    # Import only when publishing starts, so --help works without a sourced ROS2 environment.
    from debug_publisher import DebugPublisher

    publisher = DebugPublisher()
    print(
        f"publishing fake sim2real debug data for {args.seconds:g}s at {rate:g} Hz "
        "on /sim2real_debug/{state,action_raw,action_scaled}"
    )
    try:
        start = time.monotonic()
        for step in range(steps):
            t = step * period
            state = _fake_state(t, args)
            raw_action = _fake_raw_action(t)
            scaled_action = controller.scale_action(raw_action)
            publisher.publish_step(state, raw_action, scaled_action)

            next_tick = start + (step + 1) * period
            sleep_left = next_tick - time.monotonic()
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\nstopping: interrupted")
    finally:
        time.sleep(0.3)
        publisher.shutdown()
        print("fake debug publisher stopped")


if __name__ == "__main__":
    main()
