"""Python policy inference for sim2real (CPU): load encoder + actor, run forward.

Rebuilds the two deployment MLPs with the exact widths from the shared config
(verified against the checkpoint) and loads the matching weights out of
model_state_dict. The critic is training-only and is not loaded.

Inference mirrors ActorCriticSequence.act_inference():
    latent = encoder(obs_history)            # estimates base lin vel etc.
    action = actor(cat(obs, latent))         # deterministic mean action
"""
import torch
import torch.nn as nn

from config import CONFIG, REPO_ROOT

_network = CONFIG["network"]


def _activation(name):
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "selu": nn.SELU(), "tanh": nn.Tanh()}[name]


def _build_mlp(input_dim, hidden_dims, output_dim, activation):
    """MLP with `activation` between layers and none after the output layer.

    Matches the Sequential layout of ActorCriticSequence so the saved state_dict
    keys (actor.0, actor.2, ...) line up exactly.
    """
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
    return nn.Sequential(*layers)


class SequencePolicy:
    """Encoder + actor inference wrapper, numpy in / numpy out."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        activation = _activation(_network["activation"])

        self.encoder = _build_mlp(
            _network["num_encoder_obs"], _network["encoder_hidden_dims"], _network["latent_dim"], activation
        ).to(self.device)
        self.actor = _build_mlp(
            _network["num_obs"] + _network["latent_dim"],
            _network["actor_hidden_dims"],
            _network["num_actions"],
            activation,
        ).to(self.device)

        model_path = REPO_ROOT / _network["model_path"]
        state = torch.load(model_path, map_location=self.device)["model_state_dict"]
        self.encoder.load_state_dict(
            {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        )
        self.actor.load_state_dict(
            {k[len("actor.") :]: v for k, v in state.items() if k.startswith("actor.")}
        )
        self.encoder.eval()
        self.actor.eval()

    @torch.no_grad()
    def act(self, obs, obs_history):
        """obs: (num_obs,), obs_history: (num_encoder_obs,) -> action (num_actions,)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        hist_t = torch.as_tensor(obs_history, dtype=torch.float32, device=self.device)
        latent = self.encoder(hist_t)
        action = self.actor(torch.cat((obs_t, latent), dim=-1))
        return action.cpu().numpy()
