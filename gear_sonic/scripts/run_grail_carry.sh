#!/usr/bin/env bash
# Launch the five terminals from docs/source/tutorials/grail_carry.md.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

OBJECT_KEY=${OBJECT_KEY:-pickup_table__soap_dispenser_0__001}
DEVICE=${DEVICE:-cpu}
DEFAULT_CHECKPOINT=gear_sonic_deploy/policy/grail_grasp_policy/1119robocasa_carry_table-randomized-20261119_081005/last.pt
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
REWARD_VALUE_CHECKPOINT=${REWARD_VALUE_CHECKPOINT:-$DEFAULT_CHECKPOINT}
SIM_PYTHON=${SIM_PYTHON:-$(command -v python)}
DATASET_DIR=${DATASET_DIR:-gear_sonic/data/grail/pickup_table_slowturn_300}
INITIAL_POSE=${INITIAL_POSE:-gear_sonic/data/grail/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl}
cache_args=()
if [[ -n ${GRAIL_CACHE_DIR:-} ]]; then
  cache_args=(--grail-cache-dir "$GRAIL_CACHE_DIR")
fi

"$SIM_PYTHON" -c 'import sys; from gear_sonic.utils.grail.assets import load_assets; load_assets(*sys.argv[1:])' \
  "$DATASET_DIR" "$OBJECT_KEY" ${GRAIL_CACHE_DIR:+"$GRAIL_CACHE_DIR"}
mkdir -p outputs

gnome-terminal --window --title="GRAIL - Simulation" --working-directory="$ROOT" -- \
  "$SIM_PYTHON" -m gear_sonic.scripts.run_sim_loop \
  --env-name grail \
  --dataset-dir "$DATASET_DIR" \
  --initial-pose-file "$INITIAL_POSE" \
  --scene-config gear_sonic/config/grail/apple_2.yaml \
  --object-key "$OBJECT_KEY" \
  "${cache_args[@]}" --grail-seed 42 \
  --grail-reward-value-checkpoint "$REWARD_VALUE_CHECKPOINT" \
  --enable-image-publish --render-backend both --isaac-headless \
  --grail-log outputs/grail-episodes.jsonl --verbose-timing

gnome-terminal --window --title="GRAIL - Policy" --working-directory="$ROOT" -- \
  "$SIM_PYTHON" -m gear_sonic.scripts.run_grail_policy \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" --threads 1

gnome-terminal --window --title="GRAIL - Decoder (press ] then Enter)" \
  --working-directory="$ROOT/gear_sonic_deploy" -- bash -ec '
    exec ./target/release/g1_deploy_onnx_ref lo \
      policy/grail_model/model_decoder_grail.onnx reference/example \
      --obs-config policy/release/observation_config.yaml \
      --policy-precision 32 \
      --input-type zmq --zmq-conflate --disable-crc-check --output-type zmq
  '

for camera in Isaac:5555 MuJoCo:5559; do
  port=${camera#*:}
  gnome-terminal --window --title="GRAIL - ${camera%:*} Camera" \
    --working-directory="$ROOT" -- bash -ec '
      port=$1
      shift
      echo "等待相机端口 $port 就绪……"
      until (: > /dev/tcp/127.0.0.1/"$port") 2>/dev/null; do
        sleep 1
      done
      exec "$@"
    ' grail-camera "$port" "$SIM_PYTHON" -m gear_sonic.scripts.run_camera_viewer \
    --camera-host localhost --camera-port "$port"
done

echo '五个终端已打开。Decoder 就绪后，在其窗口按 ]，再按 Enter 开启 streaming。'
echo '退出：仿真、Policy、Decoder 窗口分别按 Ctrl+C；相机窗口按 Q。'
