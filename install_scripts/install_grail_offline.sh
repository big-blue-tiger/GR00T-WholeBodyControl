#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/tide/miniconda3/envs/gr00t-sim/bin/python}

"$PYTHON" -m pip install --no-deps \
  'lerobot @ git+https://github.com/huggingface/lerobot.git@a445d9c9da6bea99a8972daa4fe1fdd053d711d2'
"$PYTHON" -m pip install \
  -c "$ROOT/install_scripts/requirements_grail_offline.txt" \
  -r "$ROOT/install_scripts/requirements_grail_offline.txt"

cmake -S "$ROOT/gear_sonic_deploy" -B "$ROOT/gear_sonic_deploy/build"
cmake --build "$ROOT/gear_sonic_deploy/build" \
  --target grail_offline_decoder -j4
