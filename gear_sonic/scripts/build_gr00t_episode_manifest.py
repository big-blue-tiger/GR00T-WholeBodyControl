"""Create a frozen evaluation manifest from a LeRobot dataset's source episodes."""

import argparse
from pathlib import Path

from gear_sonic.utils.grail.rollout_manifest import build_manifest, publish_manifest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset that defines the evaluation cases")
    parser.add_argument("--exclude-dataset", type=Path, action="append", required=True,
                        help="Also exclude placements and seeds from this training dataset (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261128, help="Seed used to draw new evaluation placements")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "gear_sonic/data/grail/100obj")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--initial-pose", type=Path, default=ROOT / "gear_sonic/config/grail/carry_initial_pose.json")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "gear_sonic/config/grail/apple_2.yaml")
    parser.add_argument("--robot-scene", type=Path,
                        default=ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a new manifest path: {args.output}")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    args.dataset_dir = args.dataset_dir.resolve()
    manifest = build_manifest(args)
    publish_manifest(args.output, manifest)
    print(f"[GR00T] Wrote {len(manifest['episodes'])} evaluation cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()
