"""Build and validate paired GR00T evaluation cases and rollout artifacts."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

from .offline_render import digest_file

MANIFEST_VERSION = 1
SCENE_INPUTS = ("initial_pose", "robot_scene", "scene_config")
PLACEMENT_FIELDS = {"object_initial": 7, "robot_initial": 7, "release": 3}


def read_json(path):
    return json.loads(path.read_text())


def json_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def scene_identity(args):
    return {name: digest_file(getattr(args, name)) for name in SCENE_INPUTS}


def checkpoint_identity(path):
    path = path.resolve(strict=True)
    processor = path / "processor"
    if not processor.is_dir() or (path / "processor_config.json").is_file():
        processor = path
    weights = sorted({*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")})
    if not weights:
        raise FileNotFoundError(f"Missing GR00T weights in {path}")
    # Model and processor inputs determine the identity. Training logs can be updated independently.
    configs = {path / "config.json", *path.glob("*.index.json"), *processor.glob("*config.json"),
               *processor.glob("*stats.json")}
    files = {str(p.relative_to(path)): digest_file(p) for p in sorted(configs)}
    for weight in weights:
        stat = weight.stat()
        files[weight.name] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
    return {"path": str(path), "fingerprint": json_digest(files)}


def load_manifest(path):
    manifest = read_json(path)
    if (not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION
            or not isinstance(manifest.get("episodes"), list) or not manifest["episodes"]):
        raise ValueError(f"Expected a nonempty version {MANIFEST_VERSION} evaluation manifest: {path}")
    inputs = manifest.get("scene_inputs", {})
    if not isinstance(inputs, dict) or set(inputs) != set(SCENE_INPUTS) or any(
        not isinstance(value, str) or len(value) != 64 for value in inputs.values()
    ):
        raise ValueError(f"Missing or invalid scene hashes in {path}")
    seeds = set()
    for index, row in enumerate(manifest["episodes"]):
        if not isinstance(row, dict) or type(row.get("episode_index")) is not int or row["episode_index"] != index:
            raise ValueError(f"Expected contiguous episode index {index} in {path}")
        seed, key, task = (row.get(name) for name in ("seed", "object_key", "task"))
        if not isinstance(key, str) or not key or key in (".", "..") or Path(key).name != key:
            raise ValueError(f"Invalid object key for episode {index} in {path}")
        if type(seed) is not int or not 0 <= seed < 2**32 or seed in seeds:
            raise ValueError(f"Invalid or repeated seed for episode {index} in {path}")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Missing task for episode {index} in {path}")
        seeds.add(seed)
        for name, size in PLACEMENT_FIELDS.items():
            vector = row.get(name)
            if not isinstance(vector, list) or len(vector) != size or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in vector
            ):
                raise ValueError(f"Invalid {name} for episode {index} in {path}")
    return manifest


def training_episodes(dataset):
    mapping = sorted(read_json(dataset / "meta/source_episodes.json"), key=lambda row: row["episode_index"])
    if not mapping:
        raise ValueError(f"Dataset contains no source episodes: {dataset}")
    for index, row in enumerate(mapping):
        if index != row["episode_index"]:
            raise ValueError(f"Invalid source episode numbering in {dataset}")
        metadata = read_json(dataset / row["source"])
        if metadata["object_key"] != row["object_key"] or type(metadata.get("seed")) is not int:
            raise ValueError(f"Missing object or seed provenance for {row['source']}")
        yield metadata


def build_manifest(args):
    import mujoco
    import numpy as np

    from .episode import load_initial_pose, reset_scene
    from .offline_collect import build_offline_scene, collection_objects
    from .offline_task import DEFAULT_TASK_TEMPLATE, describe_task
    from .plan import CarryPlannerConfig

    inputs = scene_identity(args)
    training = list(training_episodes(args.dataset))
    info = read_json(args.dataset / "meta/info.json")
    tasks = [json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()
             if line.strip()]
    if not len(training) == len(tasks) == info["total_episodes"]:
        raise ValueError("Dataset info, task metadata and source episode counts disagree")
    for index, (exported, source) in enumerate(zip(tasks, training)):
        if exported["episode_index"] != index or exported["tasks"] != [source["task"]]:
            raise ValueError(f"Exported task and source task disagree for episode {index}")
    used_seeds = set()
    training_positions = defaultdict(set)
    excluded = (training_episodes(dataset) for dataset in args.exclude_dataset)
    for group in (training, *excluded):
        for ep in group:
            used_seeds.add(ep["seed"])
            training_positions[ep["object_key"]].add(tuple(ep["reset"]["object_initial"][:3]))

    by_object = defaultdict(list)
    for index, row in enumerate(training):
        by_object[row["object_key"]].append((index, row["task"]))
    keys = collection_objects(args.dataset_dir, by_object, args.cache_dir)
    initial = load_initial_pose(args.initial_pose)
    rng = np.random.default_rng(args.seed)
    episodes = [None] * len(training)
    for key in keys:
        model, config, asset, target = build_offline_scene(args, key)
        planner = CarryPlannerConfig(**config["planner"])
        data = mujoco.MjData(model)
        blocked_positions = training_positions[key]
        for index, task in by_object[key]:
            for _ in range(10_000):
                seed = int(rng.integers(0, 2**32))
                if seed in used_seeds:
                    continue
                plan, reset = reset_scene(
                    model, data, config, asset, initial, np.random.default_rng(seed), planner,
                    release_position=target,
                )
                if describe_task(DEFAULT_TASK_TEMPLATE, key, plan.sparse[0, 1]) != task:
                    continue
                position = tuple(reset["object_initial"][:3])
                if position in blocked_positions:
                    continue
                break
            else:
                raise ValueError(f"Could not draw a new placement for episode {index} ({key}, {task})")
            used_seeds.add(seed)
            blocked_positions.add(position)
            episodes[index] = dict(episode_index=index, object_key=key, task=task, seed=seed,
                                   object_initial=reset["object_initial"], robot_initial=data.qpos[:7].tolist(),
                                   release=reset["release"])
    if scene_identity(args) != inputs:
        raise ValueError("Scene inputs changed while generating the evaluation manifest")
    return dict(version=MANIFEST_VERSION, scene_inputs=inputs, episodes=episodes)


def publish_manifest(path, manifest):
    """Publish the complete manifest with an exclusive creation of its final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent) as temporary:
        json.dump(manifest, temporary, indent=2, allow_nan=False)
        temporary.write("\n")
        temporary.flush()
        path.hardlink_to(temporary.name)


def check_case(episode, meta, location):
    for key in ("episode_index", "object_key", "seed", "task"):
        if meta.get(key) != episode[key]:
            raise ValueError(f"Episode {episode['episode_index']} {key} disagrees with the manifest: {location}")


def check_placement(episode, reset, robot_initial):
    import numpy as np

    placement = {**reset, "robot_initial": robot_initial}
    for name, size in PLACEMENT_FIELDS.items():
        vector = np.asarray(placement[name])
        if vector.shape != (size,) or not np.allclose(vector, episode[name], rtol=0, atol=1e-9):
            raise ValueError(f"Episode {episode['episode_index']} {name} differs from the frozen placement")


def episode_artifacts(output, index):
    name = f"episode_{index:06d}"
    return output / "traces" / name, output / "renders" / name


def episode_complete(output, episode, checkpoint, steps):
    trace_dir, render_dir = episode_artifacts(output, episode["episode_index"])
    metadata = trace_dir / "episode.json"
    if not metadata.is_file():
        return False

    import numpy as np

    from .offline_data import load_episode

    try:
        meta, trace = load_episode(metadata)
        info = read_json(render_dir / "render.json")
        if (meta["transitions"], meta["fps"], info["frames"]) != (steps, 50, steps):
            return False
        if meta["label"] not in ("success", "failure") or not (render_dir / "ego_view.mp4").stat().st_size:
            return False
        if not all((trace_dir / meta[name]).is_file() for name in ("scene", "model")):
            return False
        root_pose = trace["qpos"][0, :7]
        with np.load(render_dir / "frames.npz", allow_pickle=False) as frames:
            if not np.array_equal(frames["frame_index"], np.arange(steps)) or not np.array_equal(
                frames["timestamp"], trace["sim_time"][:-1]
            ):
                return False
    except (OSError, EOFError, ValueError, KeyError, IndexError, TypeError, BadZipFile):
        return False
    check_case(episode, meta, metadata)
    if not meta.get("checkpoint") or (trace_dir / meta["checkpoint"]).resolve() != checkpoint:
        raise ValueError(f"Episode belongs to a different checkpoint: {metadata}")
    check_placement(episode, meta["reset"], root_pose)
    return True


def pending_episodes(output, episodes, checkpoint, steps):
    expected = {f"episode_{ep['episode_index']:06d}" for ep in episodes}
    for kind in ("traces", "renders"):
        if extras := {p.name for p in (output / kind).glob("episode_*")} - expected:
            raise ValueError(f"Unexpected {kind} in {output}: {sorted(extras)}")
    return [ep for ep in episodes if not episode_complete(output, ep, checkpoint, steps)]
