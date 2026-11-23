"""Standalone inference loader for EncoderJointVectorMlpPolicy checkpoints."""

from dataclasses import dataclass
import pickle
import types

import numpy as np
import torch
from torch import nn
import yaml

from .checkpoints import resolve_checkpoint


class _TrainerMetadata:
    """Inert placeholder for unused trainer metadata embedded in .pt files."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass


class _InferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        metadata = {
            ("trl.experimental.ppo.ppo_trainer", "OnlineTrainerState"),
            ("trl.trainer.ppo_config", "PPOConfig"),
            *(
                ("transformers.trainer_utils", key)
                for key in ("IntervalStrategy", "SchedulerType", "SaveStrategy", "HubStrategy")
            ),
            ("transformers.trainer_pt_utils", "AcceleratorConfig"),
            ("transformers.training_args", "OptimizerNames"),
            ("accelerate.state", "PartialState"),
            ("accelerate.utils.dataclasses", "DistributedType"),
        }
        if (module, name) in metadata:
            return _TrainerMetadata
        return super().find_class(module, name)


_checkpoint_pickle = types.ModuleType("grail_inference_pickle")
_checkpoint_pickle.Unpickler = _InferenceUnpickler
_checkpoint_pickle.load = pickle.load


def load_checkpoint(path):
    # Replace unused trainer metadata with inert objects in trusted checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_checkpoint_pickle)


def fsq(latent):
    """vector_quantize_pytorch.FSQ: two tokens, 32 channels, 32 levels.

    The even-level half-bin offset and tanh shift are essential. Returns codes,
    not integer codebook indices; ONNX/C++ consumes these codes directly.
    """
    z = latent.reshape(-1, 2, 32).float()
    # Keep the reference's tensor arithmetic, including FP32 rounding of
    # half_l before division. A Python scalar changes exact bin boundaries.
    levels = torch.full((32,), 32, dtype=torch.int32, device=z.device)
    half_l = (levels - 1) * (1 + 1e-3) / 2
    offset = torch.where(levels % 2 == 0, 0.5, 0.0)
    shift = torch.atanh(offset / half_l)
    return (torch.round(torch.tanh(z + shift) * half_l - offset) / (levels // 2)).reshape(-1, 64)


class GrailPolicy(nn.Module):
    def __init__(self, checkpoint, device="cpu"):
        super().__init__()
        path = resolve_checkpoint(checkpoint)
        self.checkpoint = path
        self.config = yaml.safe_load(path.with_name("config.yaml").read_text())
        actor = self.config["algo"]["config"]["actor"]
        cfg = actor["backbone"]["module_config_dict"]
        if (cfg["proprio_input_dim"], cfg["privileged_input_dim"]) != (805, 79):
            raise ValueError("GRAIL carry requires 805D proprio and 79D task observations")
        dims = [884, *cfg["mlp_hidden_dims"], 66]
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.action_head = nn.Sequential(*layers)
        ckpt = load_checkpoint(path)
        state = ckpt["policy_state_dict"]
        prefix = "actor_module."
        self.action_head.load_state_dict(
            {k: state[prefix + "action_head." + k] for k in self.action_head.state_dict()},
            strict=True,
        )
        for name in ("mean", "std"):
            self.register_buffer(
                "latent_" + name, state[prefix + "latent_normalizer._" + name].clone()
            )
        self.eps = float(cfg["latent_norm_eps"])
        self.to(device).eval()

    @torch.inference_mode()
    def forward(self, proprio, task):
        device = self.latent_mean.device
        p = torch.as_tensor(proprio, dtype=torch.float32, device=device).reshape(-1, 805)
        t = torch.as_tensor(task, dtype=torch.float32, device=device).reshape(-1, 79)
        output = self.action_head(torch.cat((p, t), dim=-1))
        # No residual scaling; action_mean in GRAIL is already denormalized.
        return torch.cat(
            (output[:, :64] * (self.latent_std + self.eps) + self.latent_mean, output[:, 64:]),
            dim=-1,
        )


@dataclass
class _EpisodeAction:
    episode_id: str | int
    released: bool = False
    raw: np.ndarray | None = None
    token: np.ndarray | None = None


class GrailActor:
    """Keep an independent latent hold for each environment until it resets."""

    def __init__(self, policy, release_override=True):
        self.policy = policy
        self.release_override = release_override
        self._states = {}

    def act(self, packet):
        return self.act_batch([packet])[0]

    def act_batch(self, packets):
        states, pending = [], []
        for packet in packets:
            env_id = packet.get("env_id", 0)
            episode_id = packet["episode_id"]
            state = self._states.get(env_id)
            if state is None or state.episode_id != episode_id:
                state = self._states[env_id] = _EpisodeAction(episode_id)
            state.released |= self.release_override and packet["release_requested"]
            states.append(state)
            # An initial packet that already requests release needs one action
            # to seed the hold. Otherwise use the last action before release.
            if not state.released or state.raw is None:
                pending.append((packet, state))

        if pending:
            proprio = np.stack([packet["proprio"] for packet, _ in pending])
            task = np.stack([packet["task"] for packet, _ in pending])
            raw = self.policy(proprio, task)
            tokens = fsq(raw[:, :64]).cpu().numpy()
            for (_, state), action, token in zip(pending, raw.cpu().numpy(), tokens):
                state.raw = action.copy()
                state.token = token.copy()

        results = []
        for state in states:
            raw = state.raw.copy()
            if state.released:
                raw[64:66] = -1.0
            results.append((raw, state.token.copy()))
        return results


def finger_primitive_targets(config, side, closed):
    """Map named Dex3 primitive targets to physical radians."""
    primitive = config["manager_env"]["config"]["finger_primitive"]["primitive_action_map"][side]
    positions = primitive["pos_1" if closed else "pos_0"]
    return {
        name: value * (0.30625 if "thumb_0" in name else 0.7)
        for name, value in zip(primitive["dof_names"], positions)
    }


def finger_targets(action, config):
    """Physical radians in Dex3 thumb0/1/2, middle0/1, index0/1 order."""
    result = []
    for i, side in enumerate(("left", "right")):
        targets = finger_primitive_targets(config, side, action[64 + i] >= 0)
        order = [
            f"{side}_hand_{part}_joint"
            for part in (
                "thumb_0",
                "thumb_1",
                "thumb_2",
                "middle_0",
                "middle_1",
                "index_0",
                "index_1",
            )
        ]
        result.append(np.array([targets[name] for name in order], dtype=np.float32))
    return result
