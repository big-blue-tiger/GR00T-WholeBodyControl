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
DATASET_DIR=${DATASET_DIR:-gear_sonic/data/grail/1600obj}
DEFAULT_INITIAL_POSE=gear_sonic/config/grail/carry_initial_pose.json
cache_args=()
mode_args=(--grail-cycle-objects)
episode_log=outputs/grail-episodes.jsonl
if [[ ${1:-} == --recovery ]]; then
  mode_args=(--grail-recovery --grail-recovery-checkpoint "$CHECKPOINT")
  episode_log=outputs/grail-recovery-episodes.jsonl
  DEFAULT_INITIAL_POSE=gear_sonic/data/grail/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl
fi
INITIAL_POSE=${INITIAL_POSE:-$DEFAULT_INITIAL_POSE}
simulation_log=${episode_log%-episodes.jsonl}-simulation.log
if [[ -n ${GRAIL_CACHE_DIR:-} ]]; then
  cache_args=(--grail-cache-dir "$GRAIL_CACHE_DIR")
fi

"$SIM_PYTHON" -c 'import sys; from gear_sonic.utils.grail.assets import load_assets; load_assets(*sys.argv[1:])' \
  "$DATASET_DIR" "$OBJECT_KEY" ${GRAIL_CACHE_DIR:+"$GRAIL_CACHE_DIR"}
mkdir -p outputs

gnome-terminal --window --title="GRAIL - Simulation" --working-directory="$ROOT" -- \
  bash -o pipefail -c '
    log=$1
    shift
    "$@" 2>&1 | {
      trap "" INT
      while IFS= read -r line || [[ -n $line ]]; do
        printf "%s\n" "$line" >&3
        if [[ $line == "[CarryRewardValue | "*"] "* ]]; then
          printf "\r\033[K%s" "${line#*] }"
        fi
      done
    } 3>>"$log"
  ' grail-simulation "$simulation_log" "$SIM_PYTHON" -u -m gear_sonic.scripts.run_sim_loop \
  --env-name grail \
  --dataset-dir "$DATASET_DIR" \
  --initial-pose-file "$INITIAL_POSE" \
  --scene-config gear_sonic/config/grail/apple_2.yaml \
  --object-key "$OBJECT_KEY" \
  "${cache_args[@]}" --grail-seed "${GRAIL_SEED:-42}" "${mode_args[@]}" \
  --grail-reward-value-checkpoint "$REWARD_VALUE_CHECKPOINT" \
  --enable-image-publish --render-backend both --isaac-headless \
  --grail-log "$episode_log" --verbose-timing

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
echo 'GRAIL - Simulation 终端刷新 Reward、Value 和 TD Advantage。'
printf '仿真日志：%s\n' "$simulation_log"
echo '退出：仿真、Policy、Decoder 窗口分别按 Ctrl+C；相机窗口按 Q。'
