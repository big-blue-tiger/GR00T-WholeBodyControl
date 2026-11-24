"""On-disk handoff between dynamics, rendering, and dataset export."""

import json
from pathlib import Path
from stat import S_IMODE
from tempfile import NamedTemporaryFile

import numpy as np

SUCCESS_DISTANCE_M = 0.2


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
    update_episode_metadata(path, metadata)


def update_episode_metadata(path, metadata):
    # Publish the metadata as a single replacement so readers always see complete JSON.
    temporary = NamedTemporaryFile(mode="w", dir=path.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(metadata, temporary, indent=2)
        temporary_path.chmod(S_IMODE(path.stat().st_mode))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def object_position_columns(model):
    joint_id = model.joint("grail_object_joint").id
    start = model.jnt_qposadr[joint_id]
    return slice(start, start + 3)


def distance_label(object_positions, release_target):
    target = np.asarray(release_target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("release target requires three finite world coordinates")
    if (
        object_positions.ndim != 2 or object_positions.shape[1] != 3
        or len(object_positions) == 0 or not np.isfinite(object_positions).all()
    ):
        raise ValueError("trajectory requires finite 3D object positions")
    minimum = float(np.linalg.norm(object_positions - target, axis=1).min())
    return dict(
        label="success" if minimum < SUCCESS_DISTANCE_M else "failure",
        min_target_distance_m=minimum,
    )


def auto_label(root):
    import mujoco

    paths = episodes(root)
    if not paths:
        raise ValueError(f"no episodes found in {root}")
    position_columns = {}
    totals = dict(success=0, failure=0, skipped=0)
    for path in paths:
        metadata = json.loads(path.read_text())
        if metadata["label"] != "unlabeled":
            totals["skipped"] += 1
            continue
        try:
            model_path = (path.parent / metadata["model"]).resolve()
            if model_path not in position_columns:
                model = mujoco.MjModel.from_binary_path(str(model_path))
                position_columns[model_path] = object_position_columns(model)
            with np.load(path.parent / "trace.npz", allow_pickle=False) as archive:
                qpos = archive["qpos"]
            if qpos.ndim != 2 or len(qpos) != metadata["transitions"] + 1:
                raise ValueError("invalid state count")
            result = distance_label(qpos[:, position_columns[model_path]], metadata["reset"]["release"])
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"cannot auto-label {path}: {error}") from error
        metadata.update(result)
        update_episode_metadata(path, metadata)
        totals[result["label"]] += 1
    print(
        f"[Auto-label] {totals['success']} success, {totals['failure']} failure, "
        f"{totals['skipped']} already labeled (threshold < {SUCCESS_DISTANCE_M:g} m)",
        flush=True,
    )


def select_episodes(root, labels):
    return [p for p in episodes(root)
            if labels == "all" or json.loads(p.read_text())["label"] in labels.split(",")]
