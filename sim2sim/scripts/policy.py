"""Load the trained ActorCriticSequence policy for deterministic sim2sim play.

Only the encoder and actor are needed for deployment (the critic is training
only). This rebuilds those two MLPs with the exact layer widths from
sim2sim_config (verified against model_8000.pt) and loads the matching weights
out of the checkpoint's model_state_dict.

Inference mirrors ActorCriticSequence.act_inference():
    latent  = encoder(observation_history)          # estimates base lin vel etc.
    action  = actor(cat(observation, latent))        # deterministic mean action
"""
import torch
import torch.nn as nn

import sim2sim_config as C


def _activation(name):
    return {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh(),
    }[name]


def _build_mlp(input_dim, hidden_dims, output_dim, activation):
    """MLP with activation between layers, none after the output layer.

    Matches the Sequential layout used in ActorCriticSequence so that the
    saved state_dict keys (e.g. actor.0, actor.2, ...) line up exactly.
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

    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        act = _activation(C.ACTIVATION)

        self.encoder = _build_mlp(
            C.NUM_ENCODER_OBS, C.ENCODER_HIDDEN_DIMS, C.LATENT_DIM, act
        ).to(self.device)
        self.actor = _build_mlp(
            C.NUM_OBS + C.LATENT_DIM, C.ACTOR_HIDDEN_DIMS, C.NUM_ACTIONS, act
        ).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
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
        """obs: (NUM_OBS,), obs_history: (NUM_ENCODER_OBS,) -> action (NUM_ACTIONS,)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        hist_t = torch.as_tensor(obs_history, dtype=torch.float32, device=self.device)
        latent = self.encoder(hist_t)
        action = self.actor(torch.cat((obs_t, latent), dim=-1))
        return action.cpu().numpy()
