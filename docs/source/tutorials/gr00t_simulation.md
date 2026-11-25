# GR00T 红桶任务服务器启动

GR00T、MuJoCo、Isaac Sim 和 SONIC decoder 都运行在 `yuguanchao_gr00t_sim1` 容器内。本机通过 SSH 查看 Isaac 的第一人称画面。

启动前确认容器可用 GPU，并已同步代码、红桶场景和 `100obj` 物体资产。容器内需有 `gr00t-sim`、`gr00t-n17` 及已编译的 `grail_offline_decoder`。[GRAIL 环境与资产准备](../getting_started/grail_environment.md)

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

GR00T 在容器回环 `127.0.0.1:5550` 接收原始 RGB 并异步返回动作块。看到 `Server ready` 后保持此终端运行。

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

终端显示 `H.264 preview listening on 5555` 后可以连接预览。切换物体时保持 SSH 隧道和查看器打开，新回合开始后会继续显示。

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

仿真期间写入原始轨迹和 50 fps H.264 录像，预览复用同一份视频编码。在仿真终端按 **Ctrl+C** 停止轮换。启动脚本会等仿真进程退出，再用 GRAIL 的离线导出器将已完成的成功和失败回合写成 LeRobot v2.1。

终端显示 `[Export] ... episodes -> .../dataset` 后，使用服务器仓库中的 `outputs/gr00t_rollouts/<运行目录>/dataset/`。其中 `data/`、`videos/` 和 `meta/` 的结构、字段和模态与 GRAIL 红桶数据集一致。旧运行可从原始轨迹和录像补导。[输出与手动导出](grail_offline.md#导出-gr00t-在线回合)
