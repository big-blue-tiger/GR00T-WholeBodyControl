# GRAIL G1 实时抓取搬运

首次按 [GRAIL 环境配置](../getting_started/grail_environment.md)安装。每个启动终端加载同一套[运行环境](../getting_started/grail_environment.md#运行环境)，从仓库根目录执行命令。

## 一键启动

在 GNOME 桌面终端执行。

```bash
DEVICE=cuda:0 bash gear_sonic/scripts/run_grail_carry.sh

DEVICE=cuda:0 bash gear_sonic/scripts/run_grail_carry_peach_bottle.sh
```

Decoder 就绪后，在其窗口按 **`]`**，再按 **Enter** 开启 streaming。退出时，仿真、策略和 decoder 终端按 **Ctrl+C**，相机窗口按 **Q**。

Isaac 相机默认使用分层 3DGS 渲染。同一帧用相同的头部相机位姿并行渲染 NuRec 背景与机器人、任务物体前景。`apple_2.yaml` 的 `isaac_scene.occlusion_proxy` 指定 `gear_sonic/data/scene/deploy/scene_combined.usda` 中的桌面深度网格，让桌面遮挡前景时保留扫描颜色。设 `show_colliders: true` 可切换到单次混合渲染，查看对齐调试网格。

搬运从 `OBJECT_KEY` 指定的物体开始。每次回合结束、机器人摔倒或在 MuJoCo 窗口按 **Backspace** 重置时，按物体标识的字典序切到下一件，末尾回到开头。轮换使用具备 USD、BPS 和 MuJoCo 预转换缓存的物体。`outputs/grail-episodes.jsonl` 中的 `object_key` 记录当前物体。

两个一键启动命令的 `GRAIL - Simulation` 终端在运行中约每 0.1 秒原地刷新一行，只显示 `Reward`、`Value` 和 `TD Advantage` 的标签与数值。其他仿真输出写入 `outputs/grail-simulation.log`（搬运）或 `outputs/grail-recovery-simulation.log`（恢复）。MuJoCo 主窗口不显示叠字。

| 环境变量 | 说明 |
|---|---|
| `DEVICE` | 顶层策略推理设备，默认 `cpu`，可设为 `cuda:0` |
| `CHECKPOINT` | 策略权重文件或目录。目录按修改时间选择最新的 `.pt` |
| `REWARD_VALUE_CHECKPOINT` | 实时奖励和价值预测权重，默认使用仓库随附的 `last.pt` |
| `DATASET_DIR` / `OBJECT_KEY` | 默认数据集为 `gear_sonic/data/grail/1600obj`，起始物体默认为 `pickup_table__soap_dispenser_0__001` |
| `GRAIL_CACHE_DIR` | 覆盖 MuJoCo 预转换目录。`1600obj` 默认使用其下的 `object_obj/` |
| `INITIAL_POSE` | 机器人初始姿态文件。常规搬运默认使用训练首帧 `gear_sonic/config/grail/carry_initial_pose.json`。出生位置以本回合物体为圆心，水平半径为 1.85 米。以垂直桌边、水平向外的方向为 0°，默认角度为 −45° 到 +45°，面向桌面看，负角在右侧，正角在左侧 |
| `SIM_PYTHON` | 覆盖 Python 解释器，默认使用当前环境的 `python` |

自定义权重须支持 884 维输入，并与其 `config.yaml` 放在同一目录。切换物体前先取得[原始资产和预转换结果](../getting_started/grail_environment.md#批量准备其他物体)。

## GR00T 红桶任务

使用服务器上的 `yuguanchao_gr00t_sim1` 时，按 [服务器启动与本地监视](gr00t_simulation.md) 操作。

先在 `gr00t-n17` 中[启动 GR00T 服务](gr00t_simulation.md#1-启动-gr00t-推理)。看到 `Server ready` 后，在同一容器的另一个终端从仓库根目录启动仿真。

```bash
conda activate gr00t-sim
POLICY_BACKEND=gr00t bash gear_sonic/scripts/run_grail_carry.sh --cycle-objects
```

`--cycle-objects` 按字典序轮换 `100obj` 中已准备的物体，每回合换一个，末尾回到开头，直到按 **Ctrl+C**。可为至少两个物体分别传入 `--object-key`，按指定顺序轮换。每回合默认运行 25 秒仿真时间，可用 `--duration` 调整。有限采集时省略 `--cycle-objects`，默认跑 `pickup_table__alcohol_10__001` 一回合，可用 `--episodes-per-object` 设置每个物体的回合数。初始姿态、转身描述和红桶目标沿用[离线采集](grail_offline.md#1-采集轨迹)。MuJoCo 推进物理，Isaac 在 headless 模式使用同一套分层 3DGS 渲染，合成单路 `ego_view`。启动脚本管理 SONIC decoder。

| 环境变量 | 说明 |
|---|---|
| `GR00T_REPO` | Isaac-GR00T 仓库，默认 `../Isaac-GR00T` |
| `MODEL_DIR` | 用于记录推理权重路径，应与手动启动的服务一致。默认是 GR00T 仓库下的 `outputs/throw_trash_into_bin_n17_fullpass` |
| `SIM_PYTHON` | 仿真解释器，默认使用当前环境的 `python` |
| `OUTPUT_DIR` | 新输出目录，默认 `outputs/gr00t_rollouts/<启动时间>-<PID>` |
| `EXPORT_DIR` | LeRobot 数据集的导出目录，默认 `$OUTPUT_DIR/dataset`，目标目录须尚未创建 |

| 通道 | 数据 |
|---|---|
| 容器回环 `127.0.0.1:5550` | ZMQ `REQ/REP` 发送合成后的原始 RGB、同帧状态和任务文本，在同一请求中返回 40 步动作块。用 `GR00T_PORT` 改端口 |
| 容器内 `decoder.sock` | 每 20 ms 仿真时间送出当前动作，由 SONIC 返回当前步的控制指令 |
| 容器 `5555` | 异步模式通过 ZMQ `PUB/SUB` 发布录像编码产生的 H.264 包。用 `--preview-port` 改端口 |

Isaac 每个控制步合成一帧，录像和 GR00T 观测共用这路 RGB。默认异步推理，控制循环按当前步与观测步的差值从动作块取动作。动作耗尽后最多保持末动作 10 步，再冻结仿真并复用当前画面请求动作。加 `--inference-mode sync` 后，每 10 步采集观测并等待新动作块。两种模式都按仿真时间保存 50 fps 录像，异步模式还复用编码包发布预览。实际仿真速度取决于每个控制步的渲染和执行耗时。

### 通过 SSH 查看

在本地解析 Docker bridge 地址并建立隧道。将 `TARGET`、SSH 端口和容器名替换为部署环境中的值。

```bash
TARGET=user@server
CONTAINER_IP=$(ssh -p 9980 "$TARGET" \
  "docker inspect -f '{{(index .NetworkSettings.Networks \"bridge\").IPAddress}}' yuguanchao_gr00t_sim1")
test -n "$CONTAINER_IP" && ssh -p 9980 -N -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:15555:$CONTAINER_IP:5555" "$TARGET"
```

另开本地终端，从仓库根目录运行查看器。H.264 模式需要 PyAV 和带窗口支持的 OpenCV。

```bash
python -m gear_sonic.scripts.run_camera_viewer \
  --h264 --camera-host 127.0.0.1 --camera-port 15555
```

**Q** 关闭本地查看器。启动终端按 **Ctrl+C** 结束采集，已完成的回合会自动导出到 `EXPORT_DIR`，省略时使用本次运行目录的 `dataset/`。[输出和手动导出](grail_offline.md#导出-gr00t-在线回合)

采集完成后，可使用保存的状态和关节命令进行 [TD Advantage 离线评分](grail_offline.md#离线评估接管时刻)。

## 抓握失败恢复测试

```bash
bash gear_sonic/scripts/run_grail_recovery.sh
```

沿用同一策略、场景、奖励与价值监视器、decoder 和相机。物体留在桌上，每轮在物体周围及上方随机生成张开或闭合的右手姿态。手物碰撞表面间距为 **5–25 cm**，碰撞或不可达的候选会重新采样。每 **5 秒仿真时间**重置，姿态求解和 decoder 同步等待不计入。

Decoder 启动后仍按 **`]`**、**Enter**。可用 `GRAIL_SEED` 复现采样序列，默认 `42`。每轮的开合状态、手物间距和相对位置写入 `outputs/grail-recovery-episodes.jsonl`。

## 分终端启动

### 终端 1 · 仿真

```bash
mkdir -p outputs
python -m gear_sonic.scripts.run_sim_loop \
  --env-name grail \
  --dataset-dir gear_sonic/data/grail/1600obj \
  --object-key pickup_table__soap_dispenser_0__001 \
  --grail-cycle-objects \
  --initial-pose-file gear_sonic/config/grail/carry_initial_pose.json \
  --scene-config gear_sonic/config/grail/apple_2.yaml \
  --grail-reward-value-checkpoint gear_sonic_deploy/policy/grail_grasp_policy/1119robocasa_carry_table-randomized-20261119_081005/last.pt \
  --enable-image-publish --render-backend both --isaac-headless \
  --grail-log outputs/grail-episodes.jsonl
```

| 参数 | 说明 |
|---|---|
| `--dataset-dir` / `--object-key` | 原始资产目录与物体标识 |
| `--grail-cycle-objects` | 每轮重置切到下一个已准备好的物体 |
| `--grail-cache-dir` | 覆盖预转换目录，与转换命令的 `--cache-dir` 对应 |
| `--initial-pose-file` / `--scene-config` | 初始姿态和场景配置 |
| `--grail-seed` | 随机种子，默认 `42` |
| `--enable-image-publish` | 发布相机图像 |
| `--render-backend` | `both` 启用双渲染，`isaac` 或 `mujoco` 启用单渲染。仅用 MuJoCo 发布图像时加 `--enable-offscreen` |
| `--isaac-headless` / `--no-enable-onscreen` | 分别隐藏 Isaac 预览和 MuJoCo 窗口 |
| `--render-fps` | 相机渲染目标帧率，默认 `30` |
| `--grail-log` / `--verbose-timing` | 任务日志路径和详细计时开关 |
| `--grail-reward-value-checkpoint` | 启用实时奖励和价值预测并输出到仿真终端，须指定含 `CarryRewardValue` 的权重 |

MuJoCo 窗口按 **Backspace** 重置场景。

### 终端 2 · 顶层策略

```bash
python -m gear_sonic.scripts.run_grail_policy --device cpu --threads 1
```

| 参数 | 说明 |
|---|---|
| `--checkpoint` | 覆盖默认权重，可传入 `.pt` 文件或目录 |
| `--device` / `--threads` | 推理设备和 PyTorch 线程数，默认 `cpu` 和 `1` |
| `--no-release-override` | 关闭强制松手和松手后的 latent 锁定 |

### 终端 3 · C++ decoder

```bash
cd gear_sonic_deploy
./target/release/g1_deploy_onnx_ref lo \
  policy/grail_model/model_decoder_grail.onnx reference/example \
  --obs-config policy/release/observation_config.yaml \
  --policy-precision 32 \
  --input-type zmq --zmq-conflate --disable-crc-check --output-type zmq
```

`lo` 为本地回环接口，后两个位置参数为 ONNX 模型和参考数据目录。`--obs-config` 选择观测配置，`--policy-precision 32` 使用 FP32，输入和输出均使用 ZMQ。

Decoder 就绪后，在其窗口按 **`]`**，再按 **Enter** 开启 streaming。终端显示 `ZMQ STREAMING MODE: ENABLED` 后进入运行状态。

### 终端 4、5 · 相机

分别在两个终端运行。

```bash
python -m gear_sonic.scripts.run_camera_viewer --camera-host localhost --camera-port 5555
```

```bash
python -m gear_sonic.scripts.run_camera_viewer --camera-host localhost --camera-port 5559
```

双渲染时 `5555` 为 Isaac 合成画面、`5559` 为 MuJoCo 对比画面。单渲染使用 `5555`。`--fps` 设置录像帧率，`--output-path` 设置录像目录。按 **R** 开始或停止录像，按 **Q** 退出。
