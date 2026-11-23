"""Prepare the selected RoboCasa USD using usd-core, NumPy and trimesh only."""

import argparse
from pathlib import Path
import sys

from gear_sonic.utils.grail.assets import prepare_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--object-key", default="pickup_table__alcohol_4__000")
    selection.add_argument("--all", action="store_true", help="convert every USD in object_usd/")
    parser.add_argument("--cache-dir", type=Path, help="override the prepared asset directory")
    args = parser.parse_args()
    if not args.all:
        print(prepare_assets(args.dataset_dir, args.object_key, args.cache_dir))
        return
    objects = sorted((args.dataset_dir / "object_usd").glob("*.usd"))
    if not objects:
        parser.error(f"no USD files found in {args.dataset_dir / 'object_usd'}")
    failures = []
    for usd in objects:
        try:
            prepare_assets(args.dataset_dir, usd.stem, args.cache_dir)
        except Exception as error:
            failures.append(usd.stem)
            print(f"{usd.stem}: {error}", file=sys.stderr, flush=True)
    print(f"Prepared {len(objects) - len(failures)}/{len(objects)} GRAIL objects.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
