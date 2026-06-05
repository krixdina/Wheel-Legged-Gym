"""Benchmark the maximum inference frequency of the sim2sim policy network.

"Inference" here means exactly what play_mujoco.py calls every control step:
SequencePolicy.act(obs, obs_history) -- encoder(135->latent) followed by
actor([obs, latent] -> action) under torch.no_grad(), numpy in / numpy out.
MuJoCo physics and the control law are deliberately excluded; this measures
only how fast the network itself can be evaluated.

For each device we report:
  - full act(): the real per-step cost including the numpy<->tensor copies that
    play_mujoco.py incurs, so 1/latency is the realistic max policy rate;
  - forward-only: encoder+actor on pre-made tensors, an upper bound that drops
    the host<->device data movement.

Run inside the isaac_gym conda env:
    conda run -n isaac_gym python sim2sim/scripts/benchmark_inference.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from policy import SequencePolicy, network, CONFIG

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POLICY_PT = os.path.join(REPO_ROOT, CONFIG["model"]["policy_path"])

# Iteration counts: enough to average out timer noise on these tiny MLPs.
WARMUP_ITERS = 200
MEASURE_ITERS = 20000


def _sync(device):
    # CUDA kernels are async; wait for them before reading the wall clock.
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_full_act(policy, obs, obs_history):
    """Time SequencePolicy.act(), i.e. the exact call play_mujoco.py makes."""
    device = policy.device
    for _ in range(WARMUP_ITERS):
        policy.act(obs, obs_history)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        policy.act(obs, obs_history)
    _sync(device)
    return (time.perf_counter() - t0) / MEASURE_ITERS


def bench_forward_only(policy, obs, obs_history):
    """Time encoder+actor on tensors already on-device (no numpy conversions)."""
    device = policy.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    hist_t = torch.as_tensor(obs_history, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            policy.actor(torch.cat((obs_t, policy.encoder(hist_t)), dim=-1))
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(MEASURE_ITERS):
            policy.actor(torch.cat((obs_t, policy.encoder(hist_t)), dim=-1))
        _sync(device)
    return (time.perf_counter() - t0) / MEASURE_ITERS


def run_device(device_name):
    policy = SequencePolicy(POLICY_PT, device=device_name)
    # Deterministic dummy inputs with the deployment shapes.
    obs = np.zeros(network["num_obs"], dtype=np.float32)
    obs_history = np.zeros(network["num_encoder_obs"], dtype=np.float32)

    full = bench_full_act(policy, obs, obs_history)
    fwd = bench_forward_only(policy, obs, obs_history)
    return {
        "device": device_name,
        "full_latency_us": full * 1e6,
        "full_hz": 1.0 / full,
        "forward_latency_us": fwd * 1e6,
        "forward_hz": 1.0 / fwd,
    }


def main():
    print(f"torch {torch.__version__}, iters={MEASURE_ITERS}, warmup={WARMUP_ITERS}")
    print(f"encoder {network['num_encoder_obs']}->{network['encoder_hidden_dims']}->{network['latent_dim']}, "
          f"actor {network['num_obs'] + network['latent_dim']}->{network['actor_hidden_dims']}->{network['num_actions']}")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print()
    for dev in devices:
        r = run_device(dev)
        print(f"[{r['device']:>4}] full act(): {r['full_latency_us']:8.2f} us  "
              f"-> {r['full_hz']:10.0f} Hz   |   "
              f"forward-only: {r['forward_latency_us']:8.2f} us -> {r['forward_hz']:10.0f} Hz")


if __name__ == "__main__":
    main()
