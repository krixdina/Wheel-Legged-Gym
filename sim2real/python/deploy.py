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

The downlink carries physically meaningful targets (theta0_ref / l0_ref /
wheel_vel_ref), not raw actions: the action scaling that used to run on the lower
machine now happens here on the NUC (Sim2RealController.scale_action).

Lost-frame policy: re-send the last command. If too many consecutive frames are
missed (config.control_timing.max_missed_frames) the link is treated as lost --
the loop stops and sends the neutral physical command (scaled zero action) so the
lower machine is not driven on stale state.

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

    controller = Sim2RealController()
    policy = SequencePolicy(device=device)
    link = RobotSerialLink()

    # Physical "neutral" command = scaled zero action (legs at default length
    # l0_offset, wheels stopped). It is the physical equivalent of the old raw-zero
    # stop, used as the initial command and as the safe command on exit.
    neutral_action = controller.scale_action(np.zeros(num_actions, dtype=np.float32))
    action = neutral_action.copy()  # physical command re-sent on a missed frame
    missed = 0

    print(f"sim2real deploy: {1.0 / period:.0f} Hz, device={device}. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.time()

            state = link.poll()
            if state is not None:
                missed = 0
                obs, obs_history = controller.observe(state)
                # Clip the RAW network output, store it as last_action (it feeds the
                # next observation), then scale it into the physical downlink command.
                raw_action = np.clip(policy.act(obs, obs_history), -clip_action, clip_action)
                controller.set_last_action(raw_action)
                action = controller.scale_action(raw_action)
            else:
                # No fresh frame this step: re-send the last physical command.
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
        # Always leave the robot with the neutral physical command, then close.
        link.send_action(neutral_action)
        link.close()
        print("sent neutral action and closed serial port.")


def parse_args():
    p = argparse.ArgumentParser(description="sim2real deployment loop")
    p.add_argument("--device", default="cpu", help="torch device for inference")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args().device)
