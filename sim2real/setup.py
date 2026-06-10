"""Minimal Python package metadata for CPU-side sim2real deployment.

This setup file is intentionally separate from the repository-root setup.py:
the root package targets Isaac Gym training, while this package only installs
the dependencies needed on the NUC deployment machine.
"""

from setuptools import setup


CPU_TORCH_1131 = [
    "torch @ https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp37-cp37m-linux_x86_64.whl ; "
    'python_version == "3.7" and platform_system == "Linux" and platform_machine == "x86_64"',
    "torch @ https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp38-cp38-linux_x86_64.whl ; "
    'python_version == "3.8" and platform_system == "Linux" and platform_machine == "x86_64"',
    "torch @ https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp39-cp39-linux_x86_64.whl ; "
    'python_version == "3.9" and platform_system == "Linux" and platform_machine == "x86_64"',
    "torch @ https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp310-cp310-linux_x86_64.whl ; "
    'python_version == "3.10" and platform_system == "Linux" and platform_machine == "x86_64"',
]

NUMPY = [
    "numpy==1.19.5 ; python_version < '3.10'",
    "numpy==1.23.5 ; python_version >= '3.10' and python_version < '3.11'",
]


setup(
    name="wheel-legged-sim2real-deploy",
    version="0.1.0",
    description="Minimal CPU NUC dependencies and entry points for sim2real deployment",
    py_modules=[
        "config",
        "controller",
        "deploy",
        "export_onnx",
        "policy",
        "probe_serial",
        "serial_comm",
    ],
    package_dir={"": "python"},
    python_requires=">=3.7,<3.11",
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
            "sim2real-export-onnx=export_onnx:main",
            "sim2real-probe-serial=probe_serial:main",
        ]
    },
)
