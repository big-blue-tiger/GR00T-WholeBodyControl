#!/usr/bin/env bash
# Launch the five terminals from docs/source/tutorials/grail_carry.md.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

OBJECT_KEY=${OBJECT_KEY:-pickup_table__soap_dispenser_0__001}
DEVICE=${DEVICE:-cpu}
PREP_PYTHON=/home/tide/miniconda3/envs/grail/bin/python
SIM_PYTHON=/home/tide/miniconda3/envs/gr00t-sim/bin/python
DATASET_DIR=/home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_slowturn_300

"$PREP_PYTHON" -m gear_sonic.scripts.prepare_grail_assets \
  --dataset-dir "$DATASET_DIR" \
  --object-key "$OBJECT_KEY" \
  --cache-dir /tmp/sonic-grail-assets

gnome-terminal --window --title="GRAIL - Simulation" --working-directory="$ROOT" -- \
  "$SIM_PYTHON" -m gear_sonic.scripts.run_sim_loop \
  --env-name grail \
  --dataset-dir "$DATASET_DIR" \
  --initial-pose-file /home/tide/robot/GRAIL/imports/SONIC/models/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl \
  --scene-config gear_sonic/config/grail/apple_2.yaml \
  --object-key "$OBJECT_KEY" \
  --grail-cache-dir /tmp/sonic-grail-assets --grail-seed 42 \
  --enable-image-publish --render-backend both --isaac-headless \
  --grail-log /tmp/grail-episodes.jsonl --verbose-timing

gnome-terminal --window --title="GRAIL - Policy" --working-directory="$ROOT" -- \
  "$SIM_PYTHON" -m gear_sonic.scripts.run_grail_policy \
  --checkpoint /home/tide/robot/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/1020robocasa_carry_table-20261026_164704/model_step_005000.pt \
  --device "$DEVICE" --threads 1

gnome-terminal --window --title="GRAIL - Decoder (press ] then Enter)" \
  --working-directory="$ROOT/gear_sonic_deploy" -- bash -ec '
    source /opt/ros/jazzy/setup.bash
    export LD_LIBRARY_PATH="/home/tide/robot/TensorRT-10.13.2.6/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
    ' grail-camera "$port" "$SIM_PYTHON" gear_sonic/scripts/run_camera_viewer.py \
    --camera-host localhost --camera-port "$port"
done

echo '五个终端已打开。Decoder 就绪后，在其窗口按 ]，再按 Enter 开启 streaming。'
echo '退出：仿真、Policy、Decoder 窗口分别按 Ctrl+C；相机窗口按 Q。'
