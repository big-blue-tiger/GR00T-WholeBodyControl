"""Single-writer LeRobot v2.1 assembly for GR00T N1.7 SONIC."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from .offline_data import load_episode, select_episodes, SUCCESS_DISTANCE_M
from .offline_sim import BODY_NAMES, HAND_NAMES, joint_addresses


def export(args):
    selected = select_episodes(args.traces, args.labels)
    if not selected:
        raise ValueError(f"no episodes match labels={args.labels}")
    if args.output.exists():
        raise FileExistsError(f"use a new dataset output directory: {args.output}")

    # Publish the dataset only after all episodes and training statistics are ready.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lerobot-", dir=args.output.parent) as temporary:
        dataset = Path(temporary) / "dataset"
        _export_episodes(args.traces, args.renders, dataset, selected)
        dataset.rename(args.output)
    print(f"[Export] {len(selected)} episodes -> {args.output}", flush=True)


def _export_episodes(traces, renders, output, selected):
    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_g1_robot_model,
        get_modality_config_sonic_vla,
    )
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity

    robot = get_g1_robot_model()
    full_features = get_features_sonic_vla(robot)
    keys = ("observation.images.ego_view", "observation.state", "observation.projected_gravity",
            "observation.root_orientation", "action.motion_token",
            "teleop.left_hand_joints", "teleop.right_hand_joints")
    features = {key: full_features[key] for key in keys}
    features.update({
        "next.observation.state": dict(features["observation.state"]),
        "next.observation.projected_gravity": dict(features["observation.projected_gravity"]),
        "next.done": dict(dtype="bool", shape=(1,), names=None),
        "source.episode_index": dict(dtype="int64", shape=(1,), names=None),
    })
    full_modality = get_modality_config_sonic_vla(robot)
    modality = {
        "state": {key: value for key, value in full_modality["state"].items()
                  if key in ("left_leg", "right_leg", "waist", "left_arm", "right_arm",
                             "left_hand", "right_hand", "projected_gravity")},
        "action": {key: full_modality["action"][key]
                   for key in ("motion_token", "left_hand_joints", "right_hand_joints")},
        "video": full_modality["video"], "annotation": full_modality["annotation"],
    }
    scene = json.loads((traces / "scene.json").read_text())
    is_gr00t_rollout = scene.get("policy") == "gr00t"
    exporter = Gr00tDataExporter.create(
        save_root=output, fps=50, features=features, modality_config=modality,
        task=scene["task"], script_config=scene, robot_type="unitree_g1_sonic", preencoded_videos=True,
    )
    model_path = None
    mapping = []
    outcomes = []
    for path in selected:
        meta, trace = load_episode(path)
        episode_model = (path.parent / meta["model"]).resolve()
        if episode_model != model_path:
            model = mujoco.MjModel.from_binary_path(str(episode_model))
            qadr, _ = joint_addresses(model, BODY_NAMES + HAND_NAMES)
            model_path = episode_model
        n = meta["transitions"]
        rendered = renders / path.parent.name
        info = json.loads((rendered / "render.json").read_text())
        with np.load(rendered / "frames.npz", allow_pickle=False) as frames:
            if not np.array_equal(frames["frame_index"], np.arange(n)) or not np.array_equal(
                frames["timestamp"], trace["sim_time"][:n]
            ) or info["frames"] != n:
                raise ValueError(f"rendered frames do not match trajectory: {path}")
        state = np.stack([
            robot.get_configuration_from_actuated_joints(
                body_actuated_joint_values=q[qadr[:29]],
                left_hand_actuated_joint_values=q[qadr[29:36]],
                right_hand_actuated_joint_values=q[qadr[36:]],
            ) for q in trace["qpos"]
        ])
        gravity = np.stack([compute_projected_gravity(q[3:7]) for q in trace["qpos"]])
        columns = {
            "timestamp": trace["sim_time"][:n],
            "observation.state": state[:-1], "next.observation.state": state[1:],
            "observation.projected_gravity": gravity[:-1],
            "next.observation.projected_gravity": gravity[1:],
            "observation.root_orientation": trace["qpos"][:-1, 3:7],
            "action.motion_token": trace["action"][:, :64].astype(np.float64),
            "teleop.left_hand_joints": trace["action"][:, 64:71],
            "teleop.right_hand_joints": trace["action"][:, 71:],
            "next.done": (np.arange(n) == n - 1)[:, None],
            "source.episode_index": np.full((n, 1), meta["episode_index"], dtype=np.int64),
        }
        exporter.save_preencoded_episode(
            columns, {"observation.images.ego_view": rendered / "ego_view.mp4"},
            {"observation.images.ego_view": info["video_stats"]}, task=meta["task"],
        )
        entry = dict(episode_index=exporter.meta.total_episodes - 1,
                     source_episode_index=meta["episode_index"], label=meta["label"],
                     source=str(path.resolve()))
        if "object_key" in meta:
            entry["object_key"] = meta["object_key"]
        mapping.append(entry)
        if is_gr00t_rollout:
            distance = meta["min_target_distance_m"]
            outcomes.append(dict(
                episode_index=entry["episode_index"], source_episode_index=meta["episode_index"],
                min_target_distance_m=distance, success=distance < SUCCESS_DISTANCE_M,
            ))
    (output / "meta" / "source_episodes.json").write_text(json.dumps(mapping, indent=2))
    write_gr00t_stats(output)
    if is_gr00t_rollout:
        distances = [outcome["min_target_distance_m"] for outcome in outcomes]
        p25, median, p75 = np.percentile(distances, [25, 50, 75]).tolist()
        successes = sum(outcome["success"] for outcome in outcomes)
        summary = dict(
            num_episodes=len(outcomes), num_success=successes,
            success_rate=successes / len(outcomes), success_threshold_m=SUCCESS_DISTANCE_M,
            min_target_distance_m=dict(mean=float(np.mean(distances)), median=median, p25=p25, p75=p75),
            episodes=outcomes,
        )
        (output / "rollout_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def write_gr00t_stats(root):
    """N1.7 reads low-dimensional quantiles from meta/stats.json."""
    import pandas as pd

    root = Path(root)
    features = json.loads((root / "meta/info.json").read_text())["features"]
    paths = sorted(root.glob("data/*/*.parquet"))
    stats = {}
    for key, feature in features.items():
        if "float" not in feature["dtype"]:
            continue
        values = np.concatenate([
            np.vstack(pd.read_parquet(path, columns=[key])[key].to_numpy()).astype(np.float32)
            for path in paths
        ])
        stats[key] = {name: function(values, axis=0).tolist() for name, function in (
            ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max),
        )}
        stats[key].update(q01=np.quantile(values, .01, axis=0).tolist(),
                          q99=np.quantile(values, .99, axis=0).tolist())
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=2))
