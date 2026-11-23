# GRAIL G1 实时抓取搬运

首次按 [GRAIL 环境配置](../getting_started/grail_environment.md)安装。每个启动终端加载同一套[运行环境](../getting_started/grail_environment.md#运行环境)，从仓库根目录执行命令。

## 一键启动

在 GNOME 桌面终端执行。

```bash
DEVICE=cuda:0 bash gear_sonic/scripts/run_grail_carry.sh
```

Decoder 就绪后，在其窗口按 **`]`**，再按 **Enter** 开启 streaming。退出时，仿真、策略和 decoder 终端按 **Ctrl+C**，相机窗口按 **Q**。

搬运从 `OBJECT_KEY` 指定的物体开始。每次回合结束、机器人摔倒或在 MuJoCo 窗口按 **Backspace** 重置时，按物体标识的字典序切到下一件，末尾回到开头。轮换使用具备 USD、BPS 和 MuJoCo 预转换缓存的物体。`outputs/grail-episodes.jsonl` 中的 `object_key` 记录当前物体。

两个一键启动命令的 `GRAIL - Simulation` 终端在运行中约每 0.1 秒原地刷新一行，只显示 `Reward`、`Value` 和 `TD Advantage` 的标签与数值。其他仿真输出写入 `outputs/grail-simulation.log`（搬运）或 `outputs/grail-recovery-simulation.log`（恢复）。MuJoCo 主窗口不显示叠字。

| 环境变量 | 说明 |
|---|---|
| `DEVICE` | 顶层策略推理设备，默认 `cpu`，可设为 `cuda:0` |
| `CHECKPOINT` | 策略权重文件或目录。目录按修改时间选择最新的 `.pt` |
| `REWARD_VALUE_CHECKPOINT` | 实时奖励和价值预测权重，默认使用仓库随附的 `last.pt` |
| `DATASET_DIR` / `OBJECT_KEY` | 默认数据集为 `gear_sonic/data/grail/1600obj`，起始物体默认为 `pickup_table__soap_dispenser_0__001` |
| `GRAIL_CACHE_DIR` | 覆盖 MuJoCo 预转换目录。`1600obj` 默认使用其下的 `object_obj/` |
| `INITIAL_POSE` | 机器人初始姿态文件 |
| `SIM_PYTHON` | 覆盖 Python 解释器，默认使用当前环境的 `python` |

自定义权重须支持 884 维输入，并与其 `config.yaml` 放在同一目录。切换物体前先取得[原始资产和预转换结果](../getting_started/grail_environment.md#批量准备其他物体)。

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
  --initial-pose-file gear_sonic/data/grail/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl \
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

双渲染时 `5555` 为 Isaac、`5559` 为 MuJoCo。单渲染使用 `5555`。`--fps` 设置录像帧率，`--output-path` 设置录像目录。按 **R** 开始或停止录像，按 **Q** 退出。
