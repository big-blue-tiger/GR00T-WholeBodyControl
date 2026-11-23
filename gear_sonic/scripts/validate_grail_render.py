"""Decode every offline video and verify source alignment.

Example:
    python -m gear_sonic.scripts.validate_grail_render --traces outputs/my_grail_100 \
      --renders outputs/my_grail_100_render_v2 --report outputs/render_validation.json
"""

import argparse
from fractions import Fraction
from functools import cache
import json
from pathlib import Path
import time


def validate(traces, renders):
    import av
    import numpy as np

    from gear_sonic.utils.grail.offline_data import episodes, load_episode
    from gear_sonic.utils.grail.offline_render import digest_file, source_fingerprint

    started = time.perf_counter()
    results = []
    env_ids = set()
    total = 0
    digest = cache(digest_file)
    for path in episodes(traces):
        meta, trace = load_episode(path)
        destination = renders / path.parent.name
        result = json.loads((destination / "render.json").read_text())
        n = meta["transitions"]
        if result["frames"] != n or result["fps"] != 50:
            raise ValueError(f"incorrect completion marker: {path}")
        if result["source_fingerprint"] != source_fingerprint(path, digest=digest):
            raise ValueError(f"changed source: {path}")
        if result["timeline_time"] != 0:
            raise ValueError(f"Isaac timeline advanced: {path}")
        with np.load(destination / "frames.npz") as frames:
            if not np.array_equal(frames["frame_index"], np.arange(n)) or not np.array_equal(
                frames["timestamp"], trace["sim_time"][:n]
            ):
                raise ValueError(f"rendered frames do not match trajectory: {path}")
        count = 0
        with av.open(str(destination / "ego_view.mp4")) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            if stream.average_rate != 50 or stream.width != 640 or stream.height != 480:
                raise ValueError(f"incorrect video format: {path}")
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.pts * frame.time_base != Fraction(index, 50):
                    raise ValueError(f"incorrect presentation timestamp: {path}, {index}")
                if (frame.height, frame.width) != (480, 640):
                    raise ValueError(f"incorrect image shape: {path}, {index}")
                count += 1
        if count != n:
            raise ValueError(f"video truncated: {path}, {count}/{n}")
        env_ids.add(meta["env_id"])
        total += count
        results.append(
            dict(
                episode=meta["episode_index"],
                env_id=meta["env_id"],
                frames=count,
            )
        )
        print(f"[ValidateRender] {len(results)} episodes, {total} decoded frames", flush=True)
    if not results:
        raise ValueError("no episodes to validate")
    return dict(
        passed=True,
        episodes=len(results),
        source_environments=len(env_ids),
        frames=total,
        frame_alignment_passed=True,
        wall_s=time.perf_counter() - started,
        results=results,
    )


def main():
    from gear_sonic.utils.grail.render_pipeline import atomic_json

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--traces", type=Path, required=True)
    cli.add_argument("--renders", type=Path, required=True)
    cli.add_argument("--report", type=Path, required=True)
    args = cli.parse_args()
    try:
        result = validate(args.traces, args.renders)
    except Exception as error:
        atomic_json(args.report, dict(passed=False, error=str(error)))
        raise
    atomic_json(args.report, result)


if __name__ == "__main__":
    main()
