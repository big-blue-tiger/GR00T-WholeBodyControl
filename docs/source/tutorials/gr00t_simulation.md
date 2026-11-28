# GR00T 红桶任务采集与在线推理

在 `yuguanchao_gr00t_sim1` 容器里运行 GR00T 和仿真。本地可通过 SSH 查看异步预览。首次使用先准备 `gr00t-sim`、`gr00t-n17`、红桶场景、`100obj` 物体资产和离线 decoder。[环境与资产配置](../getting_started/grail_environment.md)

(gr00t-sync-collect)=
## 3. GR00T 离线同步推理、采集与评分

(gr00t-policy-server)=
### 启动策略服务

在本机登录容器。策略服务和采集终端分别进入同一个容器。

```bash
ssh -t -p 9980 yuguanchao@zz.irmv.top \
  'docker exec -it -w /home/GR00T-WholeBodyControl yuguanchao_gr00t_sim1 bash'
```

在策略终端启动阶段 2 的模型。看到 `Server ready` 后保持终端运行。`MODEL_DIR` 可指向已完成的训练目录或具体的 `checkpoint-*` 目录。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-n17
MODEL_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass

python -u -m gear_sonic.scripts.run_gr00t_server \
  --gr00t-repo /home/Isaac-GR00T --model-dir "$MODEL_DIR" --device cuda:0 --port 5550 --compile
```

### 同步采集回合

在同一容器另开终端。同步采集每 10 个控制步暂停仿真，等模型返回新动作块，再继续执行。以下四路采集合计跑 500 个回合后停止，也可以按 **Ctrl+C** 提前停止。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-sim
MODEL_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass
RUN_DIR="outputs/gr00t_rollouts/$(date +%Y%m%d_%H%M%S)-$$"

POLICY_BACKEND=gr00t MODEL_DIR="$MODEL_DIR" OUTPUT_DIR="$RUN_DIR" \
  bash gear_sonic/scripts/run_grail_carry.sh --cycle-objects --inference-mode sync --num-envs 4
```

两个终端使用相同的模型路径。先试采一回合时，去掉 `--cycle-objects --num-envs 4` 即可运行默认物体的单回合。

`"$RUN_DIR/traces/"` 保存状态、动作、关节命令和标签，`"$RUN_DIR/renders/"` 保存 50 fps 第一人称录像与帧索引。仿真结束后，启动脚本将已完成的成功和失败回合导出到 `"$RUN_DIR/dataset/"`。`rollout_summary.json` 记录成功率和物体到释放目标的最近距离。

### 离线评分与处理

沿用采集终端的 `RUN_DIR`。在新终端处理已有运行时，把它设为对应的运行目录。评分使用原始轨迹留下的状态和关节命令，输出逐步分数及首次接管建议。

```bash
python -m gear_sonic.scripts.score_grail_rollouts \
  --traces "$RUN_DIR/traces" \
  --output "$RUN_DIR/gr00t_reward_value_scores" --workers 8 --resume

python -m gear_sonic.scripts.visualize_grail_scores \
  --traces "$RUN_DIR/traces" --value-scores "$RUN_DIR/gr00t_reward_value_scores" \
  --output "$RUN_DIR/gr00t_reward_value_plots"
```

TD Advantage 的 50 步滑动均值连续 5 步小于等于 `-0.35` 时，记录首次接管点。发出释放请求或物体到达目标 0.2 m 内后停止生成建议。接管点在 `episodes.jsonl`，可视化目录含评分曲线和接管帧截图。

评分还会把近目标释放后的 TD Advantage 置零。物体最后一次离手前距目标不超过 0.3 m 时，从确认离手帧往前数 3 帧开始置零，连续 3 帧失去接触时的第 3 帧用于确认离手。

同步和异步运行使用同一种导出格式。已有轨迹和录像需要手动补导时，指定尚未创建的输出目录。

```bash
python -m gear_sonic.scripts.grail_offline export \
  --traces "$RUN_DIR/traces" --renders "$RUN_DIR/renders" \
  --output "$RUN_DIR/dataset_reexport" --labels all
```

评分完成后，进入[阶段 4 接管与加权训练](grail_offline.md#gr00t-recovery-train)。

(gr00t-async-inference)=
## 5. GR00T 在线异步推理

按[启动策略服务](#gr00t-policy-server)启动要评估的模型。要切换模型时，先结束现有服务，再以新的 `MODEL_DIR` 启动。以下使用阶段 4 的加权训练结果。仿真继续执行已有动作时，后台向策略服务请求新动作块，并实时发布 H.264 第一人称画面。

### 启动在线仿真

在仿真终端使用和推理服务相同的模型路径。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-sim
MODEL_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_recovery_weighted

POLICY_BACKEND=gr00t MODEL_DIR="$MODEL_DIR" \
  bash gear_sonic/scripts/run_grail_carry.sh --cycle-objects --inference-mode async
```

回合会持续轮换物体。在仿真终端按 **Ctrl+C** 结束，已完成的回合会自动导出到本次运行目录的 `dataset/`。

### 从本机看实时画面

仿真显示 `H.264 preview listening on 5555` 后，在本机建立 SSH 隧道并保持运行。

```bash
TARGET=yuguanchao@zz.irmv.top
CONTAINER_IP=$(ssh -p 9980 "$TARGET" \
  "docker inspect -f '{{(index .NetworkSettings.Networks \"bridge\").IPAddress}}' yuguanchao_gr00t_sim1")
test -n "$CONTAINER_IP" && ssh -p 9980 -N -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:15555:$CONTAINER_IP:5555" "$TARGET"
```

另开本机终端，从 WholeBodyControl 仓库根目录打开查看器。按 **Q** 退出查看器。

```bash
conda activate gr00t-sim
python -m gear_sonic.scripts.run_camera_viewer \
  --h264 --camera-host 127.0.0.1 --camera-port 15555
```
