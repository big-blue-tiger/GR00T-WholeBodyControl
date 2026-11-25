# GR00T 红桶任务服务器启动

GR00T、MuJoCo、Isaac Sim 和 SONIC decoder 都运行在 `yuguanchao_gr00t_sim1` 容器内。本机通过 SSH 查看 Isaac 的第一人称画面。

启动前确认容器可用 GPU，并已同步代码、红桶场景和 `100obj` 物体资产。默认场景的 `gear_sonic/data/scene/deploy/` 需要有 `scene_combined_3dgs.usda`、`splats.nurec` 和桌面遮挡网格所在的 `scene_combined.usda`。容器内需有 `gr00t-sim`、`gr00t-n17` 及已编译的 `grail_offline_decoder`。[GRAIL 环境与资产准备](../getting_started/grail_environment.md)

## 1. 启动 GR00T 推理

从本机终端进入服务器容器。

```bash
ssh -t -p 9980 yuguanchao@zz.irmv.top \
  'docker exec -it -w /home/GR00T-WholeBodyControl yuguanchao_gr00t_sim1 bash'
```

在容器内运行。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-n17
export CUDA_VISIBLE_DEVICES=0

python -u -m gear_sonic.scripts.run_gr00t_server \
  --gr00t-repo /home/Isaac-GR00T \
  --model-dir /home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass \
  --device cuda:0 --port 5550
```

GR00T 在容器回环 `127.0.0.1:5550` 接收合成后的原始 RGB、同帧状态和任务描述，并返回动作块。看到 `Server ready` 后保持此终端运行。

## 2. 一键启动仿真

另开一个终端，用上面的 SSH 命令进入同一容器，再运行。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-sim
export CUDA_VISIBLE_DEVICES=0

POLICY_BACKEND=gr00t \
MODEL_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass \
  bash gear_sonic/scripts/run_grail_carry.sh --cycle-objects
```

脚本启动 headless 仿真和 SONIC decoder，并连接已启动的 GR00T 服务。每回合默认运行 25 秒仿真时间，结束后按物体标识的字典序切换到 `100obj` 中的下一个已准备物体，全部跑完后从头继续。需要指定轮换范围时，可为至少两个物体分别传入 `--object-key`，按指定顺序轮换。

同步和异步推理共用 Isaac 的分层 3DGS 相机。每个控制步用同一头部相机位姿并行渲染 NuRec 背景与机器人、任务物体前景。桌面深度网格用于遮挡前景，合成保留扫描桌面的颜色。GR00T 请求和本地录像取自同一路 `ego_view`。

默认异步模式发布合成画面的 H.264 预览。终端显示 `H.264 preview listening on 5555` 后可以连接。切换物体时保持 SSH 隧道和查看器打开，新回合开始后会继续显示。

### 同步推理

沿用第 1 步启动的 GR00T 服务，在仿真终端加上 `--inference-mode sync`。

```bash
POLICY_BACKEND=gr00t \
MODEL_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass \
  bash gear_sonic/scripts/run_grail_carry.sh --cycle-objects --inference-mode sync
```

仿真在第 `0、10、20…` 个控制步将合成 RGB 和同帧状态发给 GR00T，暂停物理并等待返回 40 帧动作。每次从第 1 帧开始执行，最多执行 10 帧，然后重新观测和推理。同步模式可直接按第 5 步导出。每个完整回合仍保存 50 fps 本地录像、帧索引和轨迹，产物与异步模式使用相同结构。

## 3. 建立 SSH 隧道

另开一个本机终端，执行并保持连接。

```bash
TARGET=yuguanchao@zz.irmv.top
CONTAINER_IP=$(ssh -p 9980 "$TARGET" \
  "docker inspect -f '{{(index .NetworkSettings.Networks \"bridge\").IPAddress}}' yuguanchao_gr00t_sim1")
test -n "$CONTAINER_IP" && ssh -p 9980 -N -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:15555:$CONTAINER_IP:5555" "$TARGET"
```

## 4. 查看第一人称画面

再开一个本机终端运行查看器。

```bash
source /home/tide/miniconda3/etc/profile.d/conda.sh
conda activate gr00t-sim
cd /home/tide/robot/GR00T-WholeBodyControl
python -m gear_sonic.scripts.run_camera_viewer \
  --h264 --camera-host 127.0.0.1 --camera-port 15555
```

按 **Q** 关闭查看器。

## 5. 导出录制数据

仿真期间写入原始轨迹和 50 fps H.264 合成画面录像，异步预览复用同一份视频编码。在仿真终端按 **Ctrl+C** 停止轮换。启动脚本会等仿真进程退出，再用 GRAIL 的离线导出器将已完成的成功和失败回合写成 LeRobot v2.1。

终端显示 `[Export] ... episodes -> .../dataset` 后，使用服务器仓库中的 `outputs/gr00t_rollouts/<运行目录>/dataset/`。其中 `data/`、`videos/` 和 `meta/` 的结构、字段和模态与 GRAIL 红桶数据集一致。旧运行可从原始轨迹和录像补导。[输出与手动导出](grail_offline.md#导出-gr00t-在线回合)
