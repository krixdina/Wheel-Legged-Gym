"""Export the trained PyTorch policy to a single ONNX graph for the C++ runtime.

The C++ OnnxPolicyRunner expects one merged model with two float inputs named
"obs" [1, num_obs] and "obs_history" [1, num_encoder_obs] and one output "action"
[1, num_actions], computing exactly what the Python SequencePolicy does:

    latent = encoder(obs_history)
    action = actor(cat(obs, latent))

Output path comes from network.onnx_model_path in the shared config (resolved
against the repo root). Run from the repo root:

    conda run -n isaac_gym python sim2real/python/export_onnx.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from config import CONFIG, REPO_ROOT
from policy import SequencePolicy

_network = CONFIG["network"]


class _MergedPolicy(nn.Module):
    """Wrap the encoder + actor into one forward(obs, obs_history) -> action."""

    def __init__(self, sequence_policy):
        super().__init__()
        self.encoder = sequence_policy.encoder
        self.actor = sequence_policy.actor

    def forward(self, obs, obs_history):
        latent = self.encoder(obs_history)
        return self.actor(torch.cat((obs, latent), dim=-1))


def main():
    policy = SequencePolicy(device="cpu")
    model = _MergedPolicy(policy).eval()

    out_path = REPO_ROOT / _network["onnx_model_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_obs = torch.zeros(1, _network["num_obs"], dtype=torch.float32)
    dummy_history = torch.zeros(1, _network["num_encoder_obs"], dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_obs, dummy_history),
        str(out_path),
        input_names=["obs", "obs_history"],
        output_names=["action"],
        opset_version=11,
        dynamic_axes=None,  # fixed batch size 1, matching the C++ runner
    )
    print(f"exported ONNX policy -> {out_path}")


if __name__ == "__main__":
    main()
