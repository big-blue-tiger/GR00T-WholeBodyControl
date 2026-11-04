"""Standalone inference loader for EncoderJointVectorMlpPolicy checkpoints."""

from pathlib import Path
import pickle
import types

import numpy as np
import torch
from torch import nn
import yaml


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
        path = Path(checkpoint).resolve()
        self.config = yaml.safe_load(path.with_name("config.yaml").read_text())
        actor = self.config["algo"]["config"]["actor"]
        cfg = actor["backbone"]["module_config_dict"]
        dims = [892, *cfg["mlp_hidden_dims"], 66]
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.action_head = nn.Sequential(*layers)
        # Only load trusted, explicitly selected checkpoints. Unused trainer
        # metadata becomes inert objects; TRL/Transformers/Isaac are not imported.
        ckpt = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_checkpoint_pickle
        )
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
        t = torch.as_tensor(task, dtype=torch.float32, device=device).reshape(-1, 87)
        output = self.action_head(torch.cat((p, t), dim=-1))
        # No residual scaling; action_mean in GRAIL is already denormalized.
        return torch.cat(
            (output[:, :64] * (self.latent_std + self.eps) + self.latent_mean, output[:, 64:]),
            dim=-1,
        )


def finger_targets(action, config):
    """Physical radians in Dex3 thumb0/1/2, middle0/1, index0/1 order."""
    result = []
    for i, side in enumerate(("left", "right")):
        primitive = config["manager_env"]["config"]["finger_primitive"]["primitive_action_map"][
            side
        ]
        positions = primitive["pos_1"] if action[64 + i] >= 0 else primitive["pos_0"]
        targets = {
            name: value * (0.30625 if "thumb_0" in name else 0.7)
            for name, value in zip(primitive["dof_names"], positions)
        }
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
