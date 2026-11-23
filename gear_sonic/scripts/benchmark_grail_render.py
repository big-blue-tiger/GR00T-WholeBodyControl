"""Repeat complete offline-render trials with fixed inputs and decoded-video validation."""

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from gear_sonic.scripts.grail_offline import positive_int


def main():
    from gear_sonic.scripts.validate_grail_render import validate
    from gear_sonic.utils.grail.offline_data import episodes
    from gear_sonic.utils.grail.offline_render import render
    from gear_sonic.utils.grail.render_pipeline import atomic_json

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--traces", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--episodes", type=positive_int, default=100)
    cli.add_argument("--trials", type=positive_int, default=3)
    cli.add_argument("--render-envs", type=positive_int, default=1,
                     help="manual isolated Isaac concurrency limit (default: 1)")
    cli.add_argument("--encode-workers", type=positive_int, default=4)
    args = cli.parse_args()
    paths = episodes(args.traces)
    if len(paths) < args.episodes:
        cli.error(f"only {len(paths)} episodes available, requested {args.episodes}")
    if args.output.exists():
        raise FileExistsError(f"use a new benchmark output directory: {args.output}")
    args.output.mkdir(parents=True)
    subset = args.output / "traces"
    subset.mkdir()
    for path in paths[: args.episodes]:
        (subset / path.parent.name).symlink_to(path.parent.resolve(), target_is_directory=True)
    trials = []
    for index in range(args.trials):
        destination = args.output / f"trial_{index:02d}"
        render(
            SimpleNamespace(
                traces=subset,
                output=destination,
                render_envs=args.render_envs,
                encode_workers=args.encode_workers,
                overwrite=False,
                resume=False,
            )
        )
        atomic_json(destination / "validation.json", validate(subset, destination))
        metrics = json.loads((destination / "batch_metrics.json").read_text())
        trials.append(dict(trial=index, **metrics))
        summary = dict(
            trials=trials,
            episodes=args.episodes,
            median_total_wall_s=statistics.median(t["wall_s"] for t in trials),
            median_rendering_wall_s=statistics.median(t["rendering_wall_s"] for t in trials),
            median_pipeline_fps=statistics.median(t["pipeline_frames_per_second"] for t in trials),
            median_rendering_fps=statistics.median(
                t["rendering_frames_per_second"] for t in trials
            ),
        )
        atomic_json(args.output / "benchmark.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
