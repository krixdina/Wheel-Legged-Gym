"""sim2real deployment main loop (CPU NUC side): the glue between serial and policy.

This is the piece that wires the otherwise-independent layers into one fixed-rate
control loop:

    RobotSerialLink  (serial_comm) -- bytes in/out over the wire
    Sim2RealController (controller) -- raw state -> 27-dim obs, history, last_action
    SequencePolicy     (policy)     -- encoder + actor inference

Per control step (config.control_timing.policy_rate_hz, e.g. 100 Hz):
    1. poll the serial link for the newest uplink state frame
    2. if a fresh frame arrived: build the observation, run the policy, clip the
       action; if not: re-send the previous action (keeps the lower machine fed)
    3. send the action downstream and record it as last_action
    4. sleep to hold the loop rate

Lost-frame policy: re-send the last action. If too many consecutive frames are
missed (config.control_timing.max_missed_frames) the link is treated as lost --
the loop stops and commands a zero action so the lower machine is not driven on
stale state.

"""
import argparse
import time

import numpy as np

from config import CONFIG
from controller import Sim2RealController
from policy import SequencePolicy
from serial_comm import RobotSerialLink


def run(device="cpu"):
    num_actions = CONFIG["network"]["num_actions"]
    clip_action = CONFIG["clipping"]["actions"]
    timing = CONFIG["control_timing"]
    period = 1.0 / timing["policy_rate_hz"]
    # Consecutive missed uplink frames tolerated before the link is declared lost;
    # beyond this, re-sending the stale action is unsafe, so the loop stops and zeros.
    max_missed_frames = timing["max_missed_frames"]

    link = RobotSerialLink()
    controller = Sim2RealController()
    policy = SequencePolicy(device=device)

    # Last action re-sent on a missed frame; starts at the safe zero action.
    action = np.zeros(num_actions, dtype=np.float32)
    missed = 0

    print(f"sim2real deploy: {1.0 / period:.0f} Hz, device={device}. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.time()

            state = link.poll()
            if state is not None:
                missed = 0
                obs, obs_history = controller.observe(state)
                action = np.clip(policy.act(obs, obs_history), -clip_action, clip_action)
                controller.set_last_action(action)
            else:
                # No fresh frame this step: re-send the last action.
                missed += 1
                if missed >= max_missed_frames:
                    raise TimeoutError(f"no uplink frame for {missed} steps; link lost")

            link.send_action(action)

            # Hold the control rate.
            sleep_left = period - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except (KeyboardInterrupt, TimeoutError) as exc:
        print(f"\nstopping: {exc if str(exc) else 'interrupted'}")
    finally:
        # Always leave the robot with a zero command, then close the port.
        link.send_action(np.zeros(num_actions, dtype=np.float32))
        link.close()
        print("sent zero action and closed serial port.")


def parse_args():
    p = argparse.ArgumentParser(description="sim2real deployment loop")
    p.add_argument("--device", default="cpu", help="torch device for inference")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args().device)
