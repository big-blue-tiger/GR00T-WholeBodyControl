"""Prepare the selected RoboCasa USD using usd-core, NumPy and trimesh only."""

import argparse

from gear_sonic.utils.grail.assets import prepare_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--object-key", default="pickup_table__alcohol_4__000")
    parser.add_argument("--cache-dir", default="/tmp/sonic-grail-assets")
    args = parser.parse_args()
    print(prepare_assets(args.dataset_dir, args.object_key, args.cache_dir))


if __name__ == "__main__":
    main()
