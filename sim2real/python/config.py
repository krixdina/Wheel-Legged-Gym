"""Load the shared sim2real deployment config once for all Python modules.

Every Python deployment module imports CONFIG from here instead of re-reading the
yaml, so the file is parsed in exactly one place 
"""
from pathlib import Path

import yaml

# config/sim2real.yaml sits one level above this python/ directory.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sim2real.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Repo root (…/Wheel-Legged-Gym), used to resolve repo-relative paths like the
# checkpoint in network.model_path.
REPO_ROOT = Path(__file__).resolve().parents[2]
