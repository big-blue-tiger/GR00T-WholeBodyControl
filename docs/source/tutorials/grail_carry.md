# GRAIL G1 抓取搬运启动

以下命令在仓库根目录执行；仿真与策略使用 `gr00t-sim`，资产准备使用 `grail`。

## 一键启动

在 GNOME 桌面终端执行，自动准备资产并打开仿真、策略、decoder 和两路相机终端：

```bash
CHECKPOINT=gear_sonic_deploy/policy/grail_grasp_policy/1109robocasa_carry_table-randomized-20261108_164252/model_step_009000.pt \
  bash gear_sonic/scripts/run_grail_carry.sh
```

更换物体或策略推理设备：

```bash
OBJECT_KEY=pickup_table__apple_2__002 DEVICE=cuda:0 \
  CHECKPOINT=gear_sonic_deploy/policy/grail_grasp_policy/1109robocasa_carry_table-randomized-20261108_164252/model_step_009000.pt \
  bash gear_sonic/scripts/run_grail_carry.sh
```

| 环境变量 | 说明 |
|---|---|
| `OBJECT_KEY` | 物体标识，默认 `pickup_table__soap_dispenser_0__001` |
| `DEVICE` | 顶层策略推理设备，默认 `cpu`，可设为 `cuda:0` |
| `CHECKPOINT` | 策略权重文件或目录。示例直接指定 `model_step_009000.pt`；传入目录时按修改时间选择最新完整 `.pt` |

Decoder 就绪后，在其窗口按 **`]`**，再按 **Enter** 开启 streaming。
退出时，仿真、策略和 decoder 终端分别按 **Ctrl+C**，相机窗口按 **Q**。

一键启动和分终端启动都默认使用 `gear_sonic/config/grail/apple_2.yaml`。Isaac 从 `gear_sonic/data/scene/deploy/scene_combined_3dgs.usda` 和同目录的 `splats.nurec` 加载 3DGS。MuJoCo 用一块薄盒模拟桌面，顶面高度为 `0.76935 m`，前沿与画面和策略观测中的长桌沿对齐。

## 分终端启动

### 终端 1：准备资产并启动仿真

```bash
(
  set -e
  cd /home/tide/robot/GR00T-WholeBodyControl
  OBJECT_KEY=pickup_table__soap_dispenser_0__001

  /home/tide/miniconda3/envs/grail/bin/python -m gear_sonic.scripts.prepare_grail_assets \
    --dataset-dir /home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_slowturn_300 \
    --object-key "$OBJECT_KEY" --cache-dir /tmp/sonic-grail-assets

  /home/tide/miniconda3/envs/gr00t-sim/bin/python -m gear_sonic.scripts.run_sim_loop \
    --env-name grail \
    --dataset-dir /home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_slowturn_300 \
    --initial-pose-file /home/tide/robot/GRAIL/imports/SONIC/models/init_g1_state/pickup_table__alcohol_1__000_first_frame.pkl \
    --scene-config gear_sonic/config/grail/apple_2.yaml \
    --object-key "$OBJECT_KEY" \
    --grail-cache-dir /tmp/sonic-grail-assets --grail-seed 42 \
    --enable-image-publish --render-backend both --isaac-headless \
    --grail-log /tmp/grail-episodes.jsonl --verbose-timing
)
```

| 参数 | 说明 |
|---|---|
| `--dataset-dir` / `--object-key` | 资产目录与物体标识，准备和仿真时需保持一致 |
| `--cache-dir` / `--grail-cache-dir` | 资产准备和仿真使用的同一缓存目录 |
| `--initial-pose-file` | 机器人初始姿态文件 |
| `--scene-config` | 默认 `gear_sonic/config/grail/apple_2.yaml`，使用 3DGS 背景和拟合桌面碰撞 |
| `--grail-seed` | 随机种子，默认 `42` |
| `--enable-image-publish` | 发布相机图像 |
| `--render-backend` | `both` 为双渲染，`isaac` 或 `mujoco` 为单渲染 |
| `--isaac-headless` | 隐藏 Isaac 窗口；去掉后显示预览 |
| `--enable-offscreen` | 仅使用 MuJoCo 相机时追加此参数，并去掉 `--isaac-headless` |
| `--no-enable-onscreen` | 隐藏 MuJoCo 交互窗口 |
| `--render-fps` | 相机渲染目标帧率，默认 `30` |
| `--grail-log` / `--verbose-timing` | 任务日志路径与详细计时开关 |

### 终端 2：启动顶层策略

```bash
cd /home/tide/robot/GR00T-WholeBodyControl
/home/tide/miniconda3/envs/gr00t-sim/bin/python -m gear_sonic.scripts.run_grail_policy \
  --checkpoint gear_sonic_deploy/policy/grail_grasp_policy/1109robocasa_carry_table-randomized-20261108_164252/model_step_009000.pt \
  --device cpu --threads 1
```

`--checkpoint` 接受权重文件或目录。示例直接指定 `model_step_009000.pt`，启动时会打印实际加载的路径。传入目录时，按修改时间选取最新完整 `.pt`，忽略隐藏下载临时文件。`--device` 默认为 `cpu`，可改为 `cuda:0`；`--threads` 默认 `1`。追加 `--no-release-override` 可同时关闭强制松手和松手后的 latent 锁定。

### 新版 GRAIL 观测与 sparse command

实时入口和离线采集共用同一套观测、规划和接触状态代码，对齐 GRAIL 的 `robocasa_carry_table` 训练配置及 `reference_free.enabled=True` 推理入口。仅支持新版 `805 + 79 = 884` 维输入 checkpoint，输出为 64 维 latent 和左右手各 1 维 primitive；latent 按 checkpoint 的统计量还原后进行 FSQ。

本体观测为从旧到新的 5 帧历史，每帧按以下顺序拼接为 161 维：43 维相对默认关节位置、43 维关节速度、3 维 pelvis 坐标系重力方向、3 维本体角速度、3 维本体线速度、66 维上一拍 meta action。关节按场景配置的 Isaac Lab 名称顺序映射，首帧复制填满历史，重置时上一拍 action 清零。

79 维任务观测严格按 GRAIL `StudentPrivilegedCfg` 的声明顺序拼接，区间使用 Python 左闭右开下标：

| 区间 | 观测 | 定义 |
|---|---|---|
| `[0:10]` | `object_bps` | 物体 10 维 BPS |
| `[10:22]` | `table_edge_line_b` | 固定取桌子局部 `-Y` 边，切向为局部 `+X`。将该边延长为无限直线，分别编码 pelvis 和手坐标系的最近点与单位切向 `[p_B,d_B,p_H,d_H]` |
| `[22:25]` | `object_pos_b` | 物体原点在 pelvis 坐标系的位置 |
| `[25:31]` | `object_ori_b_6d` | 相对旋转矩阵前两列，按行展开 |
| `[31:40]` | `hand_object_transform_6d` | 物体在手坐标系的 3 维位置与 6 维旋转 |
| `[40:41]` | `hand_object_contact_count` | 配置的右手 8 个接触 link 中，任一 link 的法向合力模长 `> 0.01 N` 时为 1，否则为 0 |
| `[41:59]` | `target_object_poses_6d` | 最终目标分别相对当前 pelvis/手的位姿，各 9 维 |
| `[59:77]` | `target_object_carry_poses_6d` | carry 目标分别相对当前 pelvis/手的位姿，各 9 维 |
| `[77:79]` | sparse command | `[dx, dyaw]`，单位为米、弧度 |

手坐标系原点为右手 thumb_2、index_1、middle_1 三个 link 原点的平均位置，方向取 index_1。物体与目标均使用实时物理状态转换到坐标系，四元数约定为 `wxyz`。

拾取前及拾取后的 2 秒等待期启用桌沿和 carry 目标，最终目标置零。从播放起始拍开始，桌沿和 carry 目标置零，启用最终目标。机器人根节点与物体的实时 XY 距离达到 `1.5 m` 时，两个目标均置零。后续丢失接触不回退阶段。机器人和物体重置高度补偿均为 `0.001 m`，目标不加该补偿。

按物体与各 link 或桌面的配对在世界坐标系内累加法向力后取模。各手部 link 的模长之和 `>= 0.1 N` 且桌面的模长 `< 0.1 N` 连续 3 拍后确认拾取，再等待 2 秒开始固定 25 Hz 规划时钟。`dx = min(||base_goal_xy - pelvis_xy||₂, 0.15)` 每拍在线计算；`dyaw` 在规划转弯帧内取完整的最短有符号转角，其余帧为零。命令只在拾取后且当前有手接触时有效，连续 3 拍失联后持续归零；不除以控制周期、不使用旧的前瞻轨迹投影。规划走完后仍按当前位置计算 `dx`。

走完规划并保持 1.2 秒后，独立的 `release_requested` 请求将两个手部 action 设为 `-1`。实时策略入口同时锁定松手请求前最后一拍已发送的 64 维 latent 和 FSQ token，此后不再执行策略前向推理，持续向 decoder 发送固定 token 并保持双手打开，直至新 episode 重置后恢复策略更新。上一拍 action 历史也使用固定 latent；decoder 仍根据实时本体观测运行，关节目标不会被锁死。若策略进程重启后首包已请求松手，则先推理一次作为锁定值。

该请求不进入策略观测，也不额外屏蔽 sparse command；旧版桌角、8 维接触力观测和 `drop_button` 已删除。

示例仍使用本机已有的 `pickup_table_slowturn_300` 物体资产。训练命令中的 `pickup_table_slowturn_1600_no_release` 未下载到本机；可通过资产准备及仿真的 `--dataset-dir` 同步切换。物理推理不读取机器人参考 motion。

### 终端 3：启动 C++ decoder

```bash
cd /home/tide/robot/GR00T-WholeBodyControl/gear_sonic_deploy
source /opt/ros/jazzy/setup.bash
export LD_LIBRARY_PATH="/home/tide/robot/TensorRT-10.13.2.6/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

./target/release/g1_deploy_onnx_ref lo \
  policy/grail_model/model_decoder_grail.onnx reference/example \
  --obs-config policy/release/observation_config.yaml \
  --policy-precision 32 \
  --input-type zmq --zmq-conflate --disable-crc-check --output-type zmq
```

`lo` 指定本地网络接口；后两个位置参数为 decoder 模型与参考数据目录。`--obs-config` 指定观测配置，`--policy-precision 32` 使用 FP32，输入输出均使用 ZMQ。

就绪后按 **`]`**，再按 **Enter**，确认出现 `ZMQ STREAMING MODE: ENABLED`。

### 终端 4、5：查看相机

MuJoCo 和 Isaac 共用 `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml` 里的 `head_camera`。第一人称视角按 G1 Rev 1.0 的 `d435_joint` 安装位姿设置，光轴向前下倾 47.6°。离线数据生成也使用这套 640×480 D435i 彩色投影，内参来自公开的 G1 标定样本。[Humanoid-Everyday 的 RealSense RGB 标定](https://github.com/physical-superintelligence-lab/Humanoid-Everyday#camera-intrinsics-realsense-d435)

以下两条命令分别在独立终端执行：

```bash
/home/tide/miniconda3/envs/gr00t-sim/bin/python \
  gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
```

```bash
/home/tide/miniconda3/envs/gr00t-sim/bin/python \
  gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5559
```

`--camera-host` 指定图像服务器；双渲染时 `5555` 为 Isaac、`5559` 为 MuJoCo，单渲染时使用 `5555`。`--fps` 指定录像帧率，默认 `30`；`--output-path` 指定录像目录，默认 `camera_recordings/`。

相机窗口按 **R** 开始或停止录像，按 **Q** 关闭；MuJoCo 交互窗口按 **Backspace** 重置场景。
