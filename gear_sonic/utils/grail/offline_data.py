"""On-disk handoff between dynamics, rendering, and dataset export."""

import json
from pathlib import Path

import numpy as np


def episodes(root):
    return sorted(Path(root).glob("episode_*/episode.json"))


def load_episode(path):
    path = Path(path)
    metadata = json.loads(path.read_text())
    with np.load(path.parent / "trace.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    n = metadata["transitions"]
    if trace["action"].shape != (n, 78) or len(trace["qpos"]) != n + 1:
        raise ValueError(f"invalid state/action boundary counts: {path}")
    if not np.allclose(trace["sim_time"], np.arange(n + 1) / 50, rtol=0, atol=1e-9):
        raise ValueError(f"invalid simulation timestamps: {path}")
    return metadata, trace


def label_episode(root, index, label):
    path = Path(root) / f"episode_{index:06d}" / "episode.json"
    metadata = json.loads(path.read_text())
    metadata["label"] = label
    path.write_text(json.dumps(metadata, indent=2))


def select_episodes(root, labels):
    return [p for p in episodes(root)
            if labels == "all" or json.loads(p.read_text())["label"] in labels.split(",")]
