"""Offscreen video recorder for sim2sim playback.

MuJoCo renders each captured frame offscreen through mujoco.Renderer (no
interactive window is needed), and the raw RGB frames are piped to the system
ffmpeg, which encodes them into an H.264 mp4. Keeping the encoder external means
no extra Python dependency is required, and offscreen rendering lets the clip be
produced even on a headless server (export MUJOCO_GL=egl there).

A tracking camera follows the robot base so the forward-driving robot stays
framed for the whole clip.
"""
import shutil
import subprocess
from pathlib import Path

import mujoco

# mujoco.Renderer.render() returns an (H, W, 3) uint8 RGB array; this is the
# matching ffmpeg raw-frame pixel format.
RAW_PIXEL_FORMAT = "rgb24"


class VideoRecorder:
    """Render the live MuJoCo state offscreen and stream it to an mp4 file."""

    def __init__(self, model, output_path, width, height, fps, camera_cfg, track_body="base_link"):
        """
        model:       loaded MjModel, used to build the offscreen renderer.
        output_path: mp4 file to write; parent directories are created.
        width/height: frame size in pixels (both must be even for yuv420p).
        fps:         frame rate written into the mp4 container.
        camera_cfg:  dict with distance/azimuth/elevation for the tracking camera.
        track_body:  body name the camera follows so the robot stays in frame.
        """
        # MuJoCo's default offscreen framebuffer is 640x480; enlarge it to the
        # requested frame size before building the renderer, otherwise rendering
        # above the default raises a framebuffer-size error.
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)

        # Tracking camera: locked onto the robot body, so the robot does not
        # leave the frame while it drives forward.
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        self._camera.distance = camera_cfg["distance"]
        self._camera.azimuth = camera_cfg["azimuth"]
        self._camera.elevation = camera_cfg["elevation"]

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; cannot encode the recording.")
        self._output_path = str(output_path)
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)

        # Pipe raw RGB frames into ffmpeg's stdin; ffmpeg encodes H.264 into mp4.
        self._proc = subprocess.Popen(
            [
                ffmpeg, "-y",
                "-f", "rawvideo",
                "-pix_fmt", RAW_PIXEL_FORMAT,
                "-s", "{}x{}".format(width, height),
                "-r", "{}".format(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                self._output_path,
            ],
            stdin=subprocess.PIPE,
        )

    def capture(self, data):
        """Render the current sim state and append one frame to the video."""
        self._renderer.update_scene(data, camera=self._camera)
        frame = self._renderer.render()  # (H, W, 3) uint8 RGB
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        """Flush the encoder, finish the mp4, and return its path."""
        self._proc.stdin.close()
        self._proc.wait()
        return self._output_path
