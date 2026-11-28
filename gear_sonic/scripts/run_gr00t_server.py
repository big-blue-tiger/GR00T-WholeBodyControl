"""Serve the red-bin GR00T policy in the gr00t-n17 environment."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

from gear_sonic.utils.grail.rollout_manifest import checkpoint_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--compile", action="store_true", help="Compile and warm up the action head")
    args = parser.parse_args()

    repo = args.gr00t_repo.resolve()
    model = args.model_dir.resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"GR00T model config missing from {model}")
    sys.path.insert(0, str(repo))
    os.environ.setdefault("HF_HUB_CACHE", str(repo / "models/hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
    os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    import torch

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    torch.set_num_threads(4)
    print(f"[GR00T] Loading {model} on {args.device}", flush=True)
    checkpoint = checkpoint_identity(model)
    policy = Gr00tPolicy("UNITREE_G1_SONIC", str(model), device=args.device)
    active_episode = None

    def get_action_rgb(episode_id, step, rgb, state, prompt):
        nonlocal active_episode
        if episode_id != active_episode:
            policy.reset()
            active_episode = episode_id
        observation = dict(
            video={"ego_view": np.frombuffer(rgb, dtype=np.uint8).reshape(1, 1, 480, 640, 3)},
            state={key: np.frombuffer(value, dtype="<f4")[None, None] for key, value in state.items()},
            language={"annotation.human.task_description": [[prompt]]},
        )
        action, _ = policy.get_action(observation)
        actions = np.concatenate(
            (action["motion_token"][0], action["left_hand_joints"][0], action["right_hand_joints"][0]),
            axis=1,
        )
        return dict(episode_id=episode_id, step=step, actions=actions.astype("<f4", copy=False).tobytes())

    if args.compile:
        from gear_sonic.utils.grail.offline_task import DEFAULT_TASK_TEMPLATE, describe_task

        dit = policy.model.action_head.model
        dit.forward = torch.compile(dit.forward, dynamic=True)
        print("[GR00T] Warming up the compiled action head", flush=True)
        state = {
            "left_leg": bytes(6 * 4),
            "right_leg": bytes(6 * 4),
            "waist": bytes(3 * 4),
            "left_arm": bytes(7 * 4),
            "right_arm": bytes(7 * 4),
            "left_hand": bytes(7 * 4),
            "right_hand": bytes(7 * 4),
            "projected_gravity": np.array([0, 0, -1], dtype="<f4").tobytes(),
        }
        rgb = bytes(480 * 640 * 3)
        for step, (object_key, turn) in enumerate((("plastic_bottle", 0), ("apple", 1))):
            prompt = describe_task(DEFAULT_TASK_TEMPLATE, object_key, turn)
            get_action_rgb(-1, step, rgb, state, prompt)

    with PolicyServer(policy, host="127.0.0.1", port=args.port) as server:
        server.register_endpoint("get_action_rgb", get_action_rgb)
        server.register_endpoint("get_checkpoint_identity", lambda: {"checkpoint": checkpoint}, requires_input=False)
        if args.ready_file is not None:
            args.ready_file.touch()
        server.run()


if __name__ == "__main__":
    main()
