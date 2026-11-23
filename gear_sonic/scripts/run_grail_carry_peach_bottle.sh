#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export DEVICE="${DEVICE:-cuda:0}"
export DATASET_DIR="$ROOT/gear_sonic/data/grail/wahaha_ad_peach_bottle"
export OBJECT_KEY=pickup_table__wahaha_ad_peach_bottle_0__000
export GRAIL_CACHE_DIR="${GRAIL_CACHE_DIR:-$DATASET_DIR/object_obj}"

exec bash "$SCRIPT_DIR/run_grail_carry.sh" "$@"
