#!/usr/bin/env bash
# Launch GRAIL or a headless GR00T rollout from docs/source/tutorials/grail_carry.md.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

if [[ ${POLICY_BACKEND:-grail} == gr00t ]]; then
  SIM_PYTHON=${SIM_PYTHON:-$(command -v python)}
  GR00T_REPO=${GR00T_REPO:-"$ROOT/../Isaac-GR00T"}
  MODEL_DIR=${MODEL_DIR:-"$GR00T_REPO/outputs/throw_trash_into_bin_n17_fullpass"}
  GR00T_PORT=${GR00T_PORT:-5550}

  if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    exec "$SIM_PYTHON" -m gear_sonic.scripts.run_gr00t_rollout --help
  fi

  output=${OUTPUT_DIR:-"outputs/gr00t_rollouts/$(date +%Y%m%d_%H%M%S)-$$"}
  export_dir=${EXPORT_DIR:-"$output/dataset"}
  mkdir -p -- "$(dirname -- "$output")"
  mkdir -- "$output"
  if [[ -e $export_dir || -L $export_dir ]]; then
    printf 'Export destination already exists: %s\n' "$export_dir" >&2
    exit 1
  fi
  printf 'Starting simulation with GR00T at 127.0.0.1:%s. Output %s\n' "$GR00T_PORT" "$output"

  object_args=()
  cache_args=()
  if [[ -n ${OBJECT_KEY:-} ]]; then
    object_args=(--object-key "$OBJECT_KEY")
  fi
  if [[ -n ${GRAIL_CACHE_DIR:-} ]]; then
    cache_args=(--cache-dir "$GRAIL_CACHE_DIR")
  fi
  rollout_pid=
  stopping=0
  stop_rollout() {
    wait_interrupted=1
    if (( ! stopping )); then
      stopping=1
      if [[ -n $rollout_pid ]]; then
        kill -TERM "$rollout_pid" 2>/dev/null || true
      fi
    fi
  }
  trap stop_rollout INT TERM

  (
    # The launcher handles terminal Ctrl+C and forwards a single stop signal.
    trap '' INT
    exec "$SIM_PYTHON" -u -m gear_sonic.scripts.run_gr00t_rollout \
      --dataset-dir "${DATASET_DIR:-gear_sonic/data/grail/100obj}" \
      --initial-pose "${INITIAL_POSE:-gear_sonic/config/grail/carry_initial_pose.json}" \
      --seed "${GRAIL_SEED:-42}" "${object_args[@]}" "${cache_args[@]}" \
      "$@" --output "$output" --model-dir "$MODEL_DIR" --port "$GR00T_PORT"
  ) &
  rollout_pid=$!
  if (( stopping )); then
    kill -TERM "$rollout_pid" 2>/dev/null || true
  fi

  # A signal interrupts Bash's wait before the rollout has finished shutting down.
  wait_interrupted=1
  while (( wait_interrupted )); do
    wait_interrupted=0
    status=0
    wait "$rollout_pid" || status=$?
  done
  trap - INT TERM

  shopt -s nullglob
  completed_episodes=("$output"/traces/episode_*/episode.json)
  if (( ${#completed_episodes[@]} )); then
    printf 'Exporting completed episodes to %s\n' "$export_dir"
    "$SIM_PYTHON" -u -m gear_sonic.scripts.grail_offline export \
      --traces "$output/traces" --renders "$output/renders" --output "$export_dir" --labels all
  fi
  exit "$status"
fi

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
