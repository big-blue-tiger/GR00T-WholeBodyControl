"""Single-writer LeRobot v2.1 assembly for GR00T N1.7 SONIC."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from .offline_data import SUCCESS_DISTANCE_M, load_episode, score_rows, select_episodes
from .offline_sim import BODY_NAMES, HAND_NAMES, joint_addresses


def export(args):
    selected = select_episodes(args.traces, args.labels)
    if not selected:
        raise ValueError(f"no episodes match labels={args.labels}")
    if args.output.exists():
        raise FileExistsError(f"use a new dataset output directory: {args.output}")
    scores = None
    if args.value_scores is not None:
        scores = score_rows(args.value_scores.resolve(), args.traces.resolve())
        for path in selected:
            row = scores.get(path.parent.name)
            if row is None:
                raise ValueError(f"missing TD Advantage score for {path}")
            source = Path(row.get("trace", ""))
            if source.parts[-3:] != path.resolve().parts[-3:]:
                raise ValueError(f"TD Advantage score for {source} does not belong to {path}")

    # Publish the dataset only after all episodes and training statistics are ready.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lerobot-", dir=args.output.parent) as temporary:
        dataset = Path(temporary) / "dataset"
        _export_episodes(args.traces, args.renders, dataset, selected, scores)
        dataset.rename(args.output)
    print(f"[Export] {len(selected)} episodes -> {args.output}", flush=True)


def load_td_advantage(score_path, trace):
    with np.load(score_path, allow_pickle=False) as scores:
        if not {"td_advantage", "time_s", "pelvis_pos"}.issubset(scores.files):
            raise ValueError(f"missing TD Advantage or replay boundaries in {score_path}")
        if not np.array_equal(scores["time_s"], trace["sim_time"][1:]) or not np.array_equal(
            scores["pelvis_pos"], trace["qpos"][1:, :3]
        ):
            raise ValueError(f"score replay boundaries do not match the trajectory: {score_path}")
        advantages = scores["td_advantage"].astype(np.float32)
    if advantages.shape != (len(trace["action"]),) or not np.isfinite(advantages).all():
        raise ValueError(f"invalid per-transition TD Advantage in {score_path}")
    return advantages


def chunk_advantage_weights(advantages):
    """Match the 40-frame sampler and loss weighting in Isaac-GR00T."""
    coefficients = np.linspace(2.0, 0.0, 40, dtype=np.float32)
    if len(advantages) < len(coefficients):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty
    windows = np.lib.stride_tricks.sliding_window_view(advantages, len(coefficients))
    chunk_advantage = (windows * coefficients).mean(axis=-1)
    loss_weights = np.exp(np.minimum(6.578815 * chunk_advantage, 0.0))
    return chunk_advantage, loss_weights


def save_advantage_plot(path, advantages, episode_index, source_episode_index):
    from matplotlib.figure import Figure

    chunk_advantage, loss_weights = chunk_advantage_weights(advantages)
    figure = Figure(figsize=(12, 5), layout="constrained")
    axis = figure.subplots()
    weight_axis = axis.twinx()
    raw_line, = axis.plot(
        advantages, color="C0", alpha=0.65, label="Per-frame TD Advantage",
        marker="o" if len(advantages) == 1 else None,
    )
    chunk_line, = axis.plot(
        chunk_advantage, color="C1", label="40-frame weighted TD Advantage (x)",
        marker="o" if len(chunk_advantage) == 1 else None,
    )
    weight_line, = weight_axis.plot(
        loss_weights, color="C2", linestyle="--", label=r"Loss weight $e^{\min(6.578815\,x,\,0)}$",
        marker="o" if len(loss_weights) == 1 else None,
    )
    axis.set_title(f"Episode {episode_index:06d}  |  Source episode {source_episode_index:06d}")
    axis.set_xlabel("Frame index (chunk curves use the start frame)")
    axis.set_ylabel("TD Advantage")
    axis.grid(alpha=0.2)
    weight_axis.set_ylabel("Loss weight", color="C2")
    weight_axis.tick_params(axis="y", labelcolor="C2")
    weight_axis.set_ylim(-0.02, 1.02)
    weight_axis.legend(handles=[raw_line, chunk_line, weight_line], loc="best", framealpha=1)
    figure.savefig(path, dpi=140)


def _export_episodes(traces, renders, output, selected, scores=None):
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
    if scores is not None:
        features["td_advantage"] = dict(dtype="float32", shape=(1,), names=None)
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
    if scores is not None:
        advantage_debug = output / "debug" / "advantage"
        advantage_debug.mkdir(parents=True)
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
        if scores is not None:
            columns["td_advantage"] = load_td_advantage(scores[path.parent.name]["local_score"], trace)
        exporter.save_preencoded_episode(
            columns, {"observation.images.ego_view": rendered / "ego_view.mp4"},
            {"observation.images.ego_view": info["video_stats"]}, task=meta["task"],
        )
        episode_index = exporter.meta.total_episodes - 1
        if scores is not None:
            save_advantage_plot(
                advantage_debug / f"episode_{episode_index:06d}.png",
                columns["td_advantage"], episode_index, meta["episode_index"],
            )
        entry = dict(episode_index=episode_index,
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
