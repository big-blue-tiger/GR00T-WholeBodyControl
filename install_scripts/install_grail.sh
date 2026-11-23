#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
REQUIREMENTS="$ROOT/install_scripts/requirements_grail.txt"
python_prefix=$("$PYTHON" -c 'import sys; print(sys.prefix)')

isaacsim_requirement=$(grep '^isaacsim\[' "$REQUIREMENTS")
usd_requirement=$(grep '^usd-core==' "$REQUIREMENTS")
lerobot_requirement=$(grep '^lerobot @ ' "$REQUIREMENTS")
opencv_requirement=$(grep '^opencv-python==' "$REQUIREMENTS")

# Install Kit before the pinned MuJoCo and desktop OpenCV stack.
"$PYTHON" -m pip install --extra-index-url https://pypi.nvidia.com "$isaacsim_requirement"
CYCLONEDDS_HOME="$python_prefix" "$PYTHON" -m pip install \
  -r <(sed -E '/^(isaacsim\[|usd-core==|lerobot @ )/d' "$REQUIREMENTS") \
  -e "$ROOT/external_dependencies/unitree_sdk2_python"
"$PYTHON" -m pip install --no-deps --force-reinstall "$opencv_requirement"

# LeRobot uses the dependencies pinned here for the offline exporter.
"$PYTHON" -m pip install --no-deps "$lerobot_requirement"

# Only the converter adds this directory to sys.path. Kit supplies its own pxr.
"$PYTHON" -m pip install --upgrade --no-deps \
  --target "$python_prefix/share/grail-usd" "$usd_requirement"

cmake -S "$ROOT/gear_sonic_deploy" -B "$ROOT/gear_sonic_deploy/build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/gear_sonic_deploy/build" \
  --target g1_deploy_onnx_ref grail_offline_decoder --parallel 4
