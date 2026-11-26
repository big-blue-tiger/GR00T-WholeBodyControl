"""Plot scored GR00T rollouts and save the ego-view frame at each suggested takeover."""

import argparse
import json
from pathlib import Path

import av
from matplotlib.figure import Figure
import numpy as np

from gear_sonic.utils.grail.gr00t_recovery import RecoveryConfig
from gear_sonic.utils.grail.offline_data import episodes


def moving_mean(values, window):
    mean = np.full(values.shape, np.nan)
    if len(values) >= window:
        finite = np.where(np.isfinite(values), values, np.nan)
        mean[window - 1:] = np.lib.stride_tricks.sliding_window_view(finite, window).mean(axis=-1)
    return mean


def plot_scores(score_path, decision, destination, config):
    with np.load(score_path, allow_pickle=False) as scores:
        time = scores["time_s"]
        curves = (scores["value"], scores["td_advantage"])

    figure = Figure(figsize=(11, 6), layout="constrained")
    axes = figure.subplots(2, 1, sharex=True)
    title = "No takeover suggested" if decision is None else (
        f"Takeover at {decision['time_s']:.2f} s (step {decision['step']})"
    )
    figure.suptitle(f"{score_path.stem}  |  {title}")
    for ax, values, label in zip(axes, curves, ("Value", "TD advantage")):
        mean = moving_mean(values, config.window_steps)
        ax.plot(time, values, color="C0", alpha=0.35, label="Raw")
        ax.plot(time, mean, color="C0", label=f"{config.window_steps}-step moving mean")
        if decision is not None:
            ax.axvline(decision["time_s"], color="C3", linestyle="--", label="Takeover")
            ax.scatter(decision["time_s"], mean[decision["step"] - 1], color="C3", zorder=3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[1].axhline(config.td_threshold, color="0.45", linestyle=":", label="Recovery threshold")
    axes[1].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.legend(loc="upper right")
    figure.savefig(destination, dpi=140)


def save_recovery_frame(render_dir, time_s, destination):
    with np.load(render_dir / "frames.npz", allow_pickle=False) as frames:
        # Scores are timestamped at transition end; video starts at boundary zero.
        matches = np.flatnonzero(np.isclose(frames["timestamp"], time_s, rtol=0, atol=1e-6))
        if len(matches) != 1:
            raise ValueError(f"no unique recorded frame at {time_s:.3f} s")
        index = int(frames["frame_index"][matches[0]])

    with av.open(str(render_dir / "ego_view.mp4")) as video:
        stream = video.streams.video[0]
        pts = (stream.start_time or 0) + round(index / stream.average_rate / stream.time_base)
        video.seek(pts, stream=stream)
        for frame in video.decode(stream):
            if frame.pts == pts:
                frame.to_image().save(destination)
                return
            if frame.pts is not None and frame.pts > pts:
                break
    raise ValueError(f"cannot decode recorded frame {index} at {time_s:.3f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True, help="Episode directory next to renders/")
    parser.add_argument(
        "--value-scores", type=Path, required=True, help="Score directory containing episodes.jsonl and NPZ files"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for plots and screenshots")
    parser.add_argument("--episode", type=int, action="append", help="Plot only these episode indices")
    args = parser.parse_args()

    scores, output = args.value_scores, args.output
    renders = args.traces.parent / "renders"
    manifest = scores / "episodes.jsonl"
    if not manifest.is_file():
        parser.error(f"score manifest not found: {manifest}")
    selected = {path.parent.name for path in episodes(args.traces)}
    if args.episode:
        selected.intersection_update(f"episode_{index:06d}" for index in args.episode)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if Path(row["score"]).stem in selected]
    if not rows:
        parser.error("no matching scored episodes")

    output.mkdir(parents=True, exist_ok=True)
    config = RecoveryConfig()
    snapshots, failed = 0, 0
    for row in rows:
        score_path = scores / Path(row["score"]).name
        episode = score_path.stem
        decision = row["recovery_decision"]
        plot_scores(score_path, decision, output / f"{episode}.png", config)
        if decision is not None:
            try:
                save_recovery_frame(
                    renders / episode, decision["time_s"], output / f"{episode}_recovery.png"
                )
                snapshots += 1
            except (OSError, ValueError, av.FFmpegError) as error:
                print(f"[Plot] {episode}: {error}")
                failed += 1
    print(f"[Plot] {len(rows)} plots, {snapshots} takeover frames, {failed} frame errors -> {output}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
