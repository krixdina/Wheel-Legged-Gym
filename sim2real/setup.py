"""Minimal Python package metadata for CPU-side sim2real deployment.

This setup file is intentionally separate from the repository-root setup.py:
the root package targets Isaac Gym training, while this package only installs
the dependencies needed on the NUC deployment machine. The NUC-side package is
intended to run with ROS2 Humble debug support, whose rclpy binaries target the
system Python 3.10 ABI on Ubuntu 22.04.
"""

from setuptools import setup


CPU_TORCH_1131 = [
    "torch @ https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp310-cp310-linux_x86_64.whl ; "
    'python_version == "3.10" and platform_system == "Linux" and platform_machine == "x86_64"',
]

NUMPY = [
    "numpy==1.23.5 ; python_version >= '3.10' and python_version < '3.11'",
]


setup(
    name="wheel-legged-sim2real-deploy",
    version="0.1.0",
    description="Minimal CPU NUC dependencies and entry points for sim2real deployment",
    py_modules=[
        "config",
        "controller",
        "debug_fake_data",
        "debug_publisher",
        "deploy",
        "export_onnx",
        "policy",
        "probe_serial",
        "serial_comm",
    ],
    package_dir={"": "python"},
    python_requires=">=3.10,<3.11",
    install_requires=[
        *NUMPY,
        "PyYAML==6.0",
        "pyserial==3.5",
        "onnx==1.14.1",
        "protobuf==4.21.8",
        *CPU_TORCH_1131,
    ],
    entry_points={
        "console_scripts": [
            "sim2real-deploy=deploy:main",
            "sim2real-debug-fake-data=debug_fake_data:main",
            "sim2real-export-onnx=export_onnx:main",
            "sim2real-probe-serial=probe_serial:main",
        ]
    },
)
