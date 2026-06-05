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
        "serial_comm",
    ],
    package_dir={"": "python"},
    python_requires=">=3.7,<3.9",
    install_requires=[
        "numpy==1.19.5",
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
        ]
    },
)
