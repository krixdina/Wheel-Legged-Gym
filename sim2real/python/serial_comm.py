"""Serial communication for the wheel-legged sim2real deployment (CPU NUC side).

Three responsibilities, cleanly separated (after the rm_serial_driver design):

    SerialTransport  - pyserial byte stream (open/read/write); the ONLY I/O layer.
    FrameCodec       - fixed-length framing [SOF][payload][CRC8(optional)][EOF];
                       frame sync + resync/reassembly (a single read may split or
                       merge frames), optional CRC8 integrity check.
    protocol funcs   - serialize/deserialize the business fields with struct.
    RobotSerialLink  - wires the three together for the deployment loop.

All parameters (port, baudrate, frame markers, CRC poly, payload layouts) come
from the shared config so a firmware/protocol change is a one-line yaml edit.

Two fixed-length frames (layouts in config.frame):
    uplink   (lower machine -> NUC): every network state EXCEPT last_action, raw
        and unscaled: base_ang_vel(3) | projected_gravity(3) | commands(3) |
        theta0(2) | theta0_dot(2) | L0(2) | L0_dot(2) | wheel_pos(2) | wheel_vel(2).
    downlink (NUC -> lower machine): action(6).

All multi-byte fields are little-endian, matching the MCU's memcpy serialization
(NUC and STM32 are both little-endian).

Deploy note: `pip install pyserial` on the NUC.
"""
import struct

import numpy as np
import serial  # pyserial

from config import CONFIG

_frame = CONFIG["frame"]

# Field offsets inside the 21-float uplink payload (fixed by the protocol order
# documented above and in 状态量与动作量说明.md).
_UPLINK_SLICES = {
    "base_ang_vel": (0, 3),
    "projected_gravity": (3, 6),
    "commands": (6, 9),
    "theta0": (9, 11),
    "theta0_dot": (11, 13),
    "L0": (13, 15),
    "L0_dot": (15, 17),
    "wheel_pos": (17, 19),
    "wheel_vel": (19, 21),
}


class SerialTransport:
    """pyserial byte stream -- the only layer that touches the device.

    8 data bits, no parity, 1 stop bit, no flow control, raw mode. `timeout=0`
    makes read() non-blocking (returns whatever is already buffered), which suits
    a fixed-rate poll loop.
    """

    def __init__(self, port, baudrate):
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            rtscts=False,
            dsrdtr=False,
            timeout=0.0,
        )

    def read_available(self):
        """Return all bytes currently buffered (never blocks)."""
        pending = self._serial.in_waiting
        # 如果串口接收缓冲区里有数据，就把这些数据读出来；如果没有数据，就返回一个空的 bytes。
        return self._serial.read(pending) if pending else b""

    def write(self, data):
        self._serial.write(data)

    def close(self):
        self._serial.close()


class FrameCodec:
    """Fixed-length frame [SOF][payload][CRC8(optional)][EOF] with a resync parser.

    The frame markers / CRC behaviour are taken from config.frame; when
    use_crc8 is false the frame carries no checksum byte and feed() only checks
    SOF/EOF. encode() builds an outgoing frame; feed() buffers incoming bytes and
    returns every complete, valid payload, sliding one byte forward on any
    mismatch (needed because one read can split or merge frames).
    """

    def __init__(self, payload_size, frame_cfg):
        self._payload_size = payload_size
        self._sof = frame_cfg["sof"]
        self._eof = frame_cfg["eof"]
        self._use_crc8 = frame_cfg["use_crc8"]
        self._crc_poly = frame_cfg["crc8_poly"]
        self._crc_init = frame_cfg["crc8_init"]
        # frame = SOF + payload (+ CRC if enabled) + EOF
        self._frame_size = payload_size + (3 if self._use_crc8 else 2)
        # 创建一个可变的字节缓存区，用来暂存从串口读到、但还没有成功解析成完整帧的数据。
        self._buffer = bytearray()

    def _crc8(self, data):
        """Bytewise CRC-8 using the config poly/init (firmware must match)."""
        crc = self._crc_init
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ self._crc_poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        return crc

    # 目的：把序列化完成后的数据字节包装成完整串口帧，供串口写入层直接发送给下位机。
    # 输入：payload 是待发送的有效数据字节，在下行通信中通常承载策略输出的 6 维动作数据。
    # 输出：返回完整帧字节，内容依次包含 SOF 这个约定的帧头标记、payload 这段有效数据字节、CRC 校验字节（若启用）和 EOF 这个约定的帧尾标记。
    def encode(self, payload):
        """payload(bytes) -> full frame bytes. When enabled, CRC8 covers SOF+payload."""
        # 先确认 payload 这段有效数据字节的长度符合当前编解码器配置，避免发送字段数量或字节数不匹配的帧。
        if len(payload) != self._payload_size:
            raise ValueError(f"payload must be {self._payload_size} bytes, got {len(payload)}")
        # body 表示要参与 CRC 校验的主体字节区域，由 SOF 这个帧头标记和 payload 这段有效数据字节组成。
        body = bytes([self._sof]) + payload
        # 启用校验时在主体字节后追加 crc8 这个按约定算法算出的校验字节，再加 EOF 这个帧尾标记；未启用校验时只追加帧尾。
        if self._use_crc8:
            return body + bytes([self._crc8(body), self._eof])
        return body + bytes([self._eof])

    # 目的：把串口读取到的原始字节追加到内部缓存 self._buffer 中，并按固定帧格式恢复出所有校验通过的 payload(承载有效数据的字节)，
    #      供上层继续解析传感器数据。
    # 输入：chunk 是本次从串口读到的一段原始字节，可能只包含半帧、整帧、多帧，或者混有需要跳过的噪声字节。
    # 输出：返回 payloads 这个列表；列表中的每一项都是去掉帧头、校验字节和帧尾后的 payload(承载有效数据的字节)，内部缓存 self._buffer 会保留尚未凑齐一帧的剩余字节。
    def feed(self, chunk):
        """Append raw bytes, return a list of all complete valid payloads."""
        # 先把 chunk 这段新收到的原始串口字节接到内部缓存 self._buffer 末尾，因为一次串口读取可能只拿到一帧的一部分。
        self._buffer.extend(chunk)
        payloads = []
        # 只有当内部缓存 self._buffer 中的字节数量已经达到 frame_size 这个固定帧总长度时，才尝试解析候选帧。
        while len(self._buffer) >= self._frame_size:
            # 如果缓存开头不是 SOF 这个约定的帧头标记，就丢掉一个字节继续寻找下一处可能的帧起点。
            if self._buffer[0] != self._sof:
                del self._buffer[0]
                continue
            # 取出一个候选帧；启用校验时由帧头与 payload(承载有效数据的字节)组成校验输入，未启用校验时只需核对帧尾。
            frame = self._buffer[: self._frame_size]
            if self._use_crc8:
                body = frame[:-2]  # SOF + payload, the bytes the CRC covers
                valid = frame[-2] == self._crc8(body) and frame[-1] == self._eof
            else:
                valid = frame[-1] == self._eof
            # 候选帧通过校验（或在无校验模式下帧尾正确）时，才把中间的 payload(承载有效数据的字节)交给调用者。
            if valid:
                payloads.append(bytes(frame[1 : 1 + self._payload_size]))
                del self._buffer[: self._frame_size]  # consume the whole frame
            else:
                # 如果候选帧校验失败或帧尾不匹配，只滑过当前帧头位置，保留后续字节继续重新同步。
                del self._buffer[0]
        return payloads


def decode_uplink(payload, uplink_format):
    """uplink payload -> dict of raw (unscaled) state arrays (float32), body frame.

    Keys match the observation components the NUC controller consumes; values are
    SI physical quantities: base_ang_vel [rad/s], projected_gravity [unit vector],
    commands [vx m/s, wz rad/s, height m], theta0 [rad], theta0_dot [rad/s],
    L0 [m], L0_dot [m/s], wheel_pos [rad], wheel_vel [rad/s].
    """
    # 按照 uplink_format 指定的二进制格式，把 payload 这段字节解析成一组 Python 数值。
    values = struct.unpack(uplink_format, payload)
    # 再按协议固定的字段偏移 _UPLINK_SLICES，把这组数值切分成各状态量并转成 float32 数组。
    return {
        name: np.array(values[lo:hi], dtype=np.float32)
        for name, (lo, hi) in _UPLINK_SLICES.items()
    }


def encode_downlink(action, downlink_format):
    """6-dim policy action -> downlink payload bytes (little-endian floats)."""
    # 按照 downlink_format 指定的二进制格式，把 action 这 6 维动作打包成下行 payload 字节。
    return struct.pack(downlink_format, *(float(a) for a in action))


class RobotSerialLink:
    """End-to-end serial link for the deployment loop, built from config.

    poll() drains the serial buffer, parses uplink frames, and returns the NEWEST
    decoded state (or None if no fresh frame this cycle) -- returning only the
    latest frame keeps the policy on current state and drops stale backlog.
    send_action() writes one downlink frame with the policy's 6-dim action.
    """

    def __init__(self, config=CONFIG):
        serial_cfg = config["serial"]
        frame_cfg = config["frame"]
        self._uplink_format = frame_cfg["uplink_format"]
        self._downlink_format = frame_cfg["downlink_format"]
        self._transport = SerialTransport(serial_cfg["port"], serial_cfg["baudrate"])
        self._uplink = FrameCodec(struct.calcsize(self._uplink_format), frame_cfg)
        self._downlink = FrameCodec(struct.calcsize(self._downlink_format), frame_cfg)

    def poll(self):
        payloads = self._uplink.feed(self._transport.read_available())
        return decode_uplink(payloads[-1], self._uplink_format) if payloads else None

    def send_action(self, action):
        payload = encode_downlink(action, self._downlink_format)
        self._transport.write(self._downlink.encode(payload))

    def close(self):
        self._transport.close()
