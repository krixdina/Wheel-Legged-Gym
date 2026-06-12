"""Read-only serial probe for sim2real uplink bring-up.

This tool opens the configured serial port, decodes incoming bytes with the
current frame settings, and validates the 21-float state without loading the
policy or sending any downlink command. It is intended for diagnosing protocol
issues before running deploy.py on the real robot.
"""
import argparse
import struct
import time

from config import CONFIG
from controller import InvalidUplinkState, Sim2RealController
from serial_comm import FrameCodec, SerialLinkError, SerialTransport, decode_uplink


def parse_args():
    parser = argparse.ArgumentParser(description="Probe sim2real serial uplink frames")
    parser.add_argument("--seconds", type=float, default=5.0, help="probe duration")
    parser.add_argument("--port", default=None, help="override config serial.port")
    return parser.parse_args()


def main():
    args = parse_args()
    serial_cfg = dict(CONFIG["serial"])
    if args.port:
        serial_cfg["port"] = args.port
    frame_cfg = CONFIG["frame"]
    uplink_format = frame_cfg["uplink_format"]
    payload_size = struct.calcsize(uplink_format)
    codec = FrameCodec(payload_size, frame_cfg)

    print(
        f"probing {serial_cfg['port']} @ {serial_cfg['baudrate']} baud for {args.seconds:g}s; "
        f"uplink_payload={payload_size} bytes, crc8={frame_cfg['use_crc8']}, "
        f"frame_size={payload_size + (3 if frame_cfg['use_crc8'] else 2)} bytes"
    )

    bytes_read = 0
    decoded = 0
    valid = 0
    invalid = 0
    first_state = None
    first_invalid = None

    transport = None
    try:
        transport = SerialTransport(serial_cfg["port"], serial_cfg["baudrate"])
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            chunk = transport.read_available()
            if not chunk:
                time.sleep(0.001)
                continue
            bytes_read += len(chunk)
            for payload in codec.feed(chunk):
                decoded += 1
                state = decode_uplink(payload, uplink_format)
                try:
                    Sim2RealController._validate_state(state)
                except InvalidUplinkState as exc:
                    invalid += 1
                    if first_invalid is None:
                        first_invalid = str(exc)
                else:
                    valid += 1
                    if first_state is None:
                        first_state = state
    except SerialLinkError as exc:
        print(f"serial error: {exc}")
        return 1
    finally:
        if transport is not None:
            try:
                transport.close()
            except SerialLinkError as exc:
                print(f"serial close warning: {exc}")

    print(f"bytes_read={bytes_read}, decoded_frames={decoded}, valid_states={valid}, invalid_states={invalid}")
    if first_state is not None:
        print(
            "first valid state: "
            f"gravity={first_state['projected_gravity']}, "
            f"L0={first_state['L0']}, "
            f"commands={first_state['commands']}"
        )
    if first_invalid is not None:
        print(f"first invalid state reason: {first_invalid}")
    if bytes_read and not decoded:
        print("raw bytes arrived, but no frame matched the configured SOF/EOF/CRC/length")
    if decoded and not valid:
        print("frames matched the byte protocol, but decoded floats are not physically plausible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
