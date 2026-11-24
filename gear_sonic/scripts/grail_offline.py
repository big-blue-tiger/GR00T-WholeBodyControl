"""Collect GRAIL dynamics, render saved states, and export GR00T N1.7 data."""

import argparse
from pathlib import Path

from gear_sonic.utils.grail.checkpoints import DEFAULT_CHECKPOINT

ROOT = Path(__file__).resolve().parents[2]
GRAIL_ASSETS = ROOT / "gear_sonic/data/grail"


def positive_int(value):
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    p = commands.add_parser("collect", help="record complete simulation-clock episodes")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help=".pt file or directory (newest complete .pt)")
    p.add_argument("--dataset-dir", type=Path, default=GRAIL_ASSETS / "100obj")
    p.add_argument("--initial-pose", type=Path, default=ROOT / "gear_sonic/config/grail/carry_initial_pose.json")
    p.add_argument("--scene-config", type=Path, default=ROOT / "gear_sonic/config/grail/apple_2.yaml")
    p.add_argument("--robot-scene", type=Path, default=ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
    p.add_argument("--object-key", action="append",
                   help="select an object; repeat for a subset (default: all dataset objects)")
    p.add_argument("--cache-dir", type=Path, help="override the preconverted asset directory")
    p.add_argument("--decoder-model", type=Path, default=ROOT / "gear_sonic_deploy/policy/grail_model/model_decoder_grail.onnx")
    p.add_argument("--decoder-executable", type=Path, default=ROOT / "gear_sonic_deploy/target/release/grail_offline_decoder")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--episodes-per-object", "--num-episodes", type=positive_int, default=5,
                   help="number of recorded trajectories per object (default: 5)")
    p.add_argument("--cpu-workers", type=int, default=4)
    p.add_argument("--duration", type=float, default=25.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--task", default="Pick up the object from the table and release it above the red bin.")
    p.add_argument("--no-release-override", action="store_true",
                   help="Disable forced-open hands and latent hold on release requests")
    commands.add_parser("benchmark", parents=[p], add_help=False,
                        help="collect trajectories and report collection throughput")
    p = commands.add_parser("render")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render-envs", type=positive_int, default=1,
                   help="manual concurrency limit: one isolated Isaac process per environment (default: 1)")
    p.add_argument("--encode-workers", type=positive_int, default=4)
    existing = p.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="render all episodes again, replacing existing videos")
    existing.add_argument("--resume", action="store_true", help="skip completed episodes and regenerate unfinished output")
    p = commands.add_parser("export")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labels", default="success", help="comma-separated success,failure,unlabeled, or all")
    p = commands.add_parser("label")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--label", choices=("success", "failure", "unlabeled"), required=True)
    return cli


def main():
    args = parser().parse_args()
    if args.command in ("collect", "benchmark"):
        from gear_sonic.utils.grail.offline_collect import collect
        collect(args)
    elif args.command == "render":
        from gear_sonic.utils.grail.offline_render import render
        render(args)
    elif args.command == "export":
        from gear_sonic.utils.grail.offline_export import export
        export(args)
    else:
        from gear_sonic.utils.grail.offline_data import label_episode
        label_episode(args.traces, args.episode, args.label)


if __name__ == "__main__":
    main()
