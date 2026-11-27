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

#### 四路同步采集

在同一条命令末尾加 `--num-envs 4`，四路共用第 1 步的 GR00T 服务。各路独立渲染、推进物理和保存回合。GR00T 逐条处理请求，物体和仿真种子按全局回合编号分配。动作采样仍由 GR00T 完成，请求顺序变化时，同一回合的轨迹可能不同。有限采集同样支持四路，省略 `--cycle-objects` 并用 `--object-key` 和 `--episodes-per-object` 指定采集范围。

```bash
POLICY_BACKEND=gr00t \
MODEL_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass \
EXPORT_DIR=outputs/gr00t_datasets/throw_trash_into_bin_500 \
  bash gear_sonic/scripts/run_grail_carry.sh --cycle-objects --inference-mode sync --num-envs 4
```

四路循环采集合计完成 **500 个回合**后自动停止，每路采集 125 个。也可在启动终端按 **Ctrl+C** 提前停止。已完成的回合统一导出到 `EXPORT_DIR`，省略时使用本次运行目录下的 `dataset/`。相对路径以仓库根目录为准，导出目标必须是尚未创建的目录。GR00T 服务继续运行，各路 decoder 日志位于本次运行目录的 `workers/<编号>/decoder.log`。

自动导出时，`EXPORT_DIR/rollout_summary.json` 记录总成功率、各回合最近距离的均值和第 25、50、75 百分位数，以及每回合的最近距离和成功状态。最近距离取物体原点与释放目标点在整个回合中的最小三维距离，严格小于 `0.2 m` 记为成功。提前停止时，统计包含已完成并导出的回合。

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

终端显示 `[Export] ... episodes -> <导出目录>` 后，从该目录读取数据集。其中 `data/`、`videos/` 和 `meta/` 的结构、字段和模态与 GRAIL 红桶数据集一致。旧运行可从原始轨迹和录像补导。[输出与手动导出](grail_offline.md#导出-gr00t-在线回合)

## 6. 离线接管与轨迹版本

先对运行目录下的原始轨迹评分。在有原运行目录的机器上，使用 `gr00t-sim` 环境运行。

```bash
python -m gear_sonic.scripts.grail_offline recovery \
  --run-dir outputs/gr00t_rollouts/20261125_191637-18053 \
  --output-dir outputs/gr00t_rollouts/20261125_191637-18053/traces_recovery \
  --num-envs 4 --cpu-workers 4
```

默认读取运行目录下的 `gr00t_reward_value_scores/episodes.jsonl`，可用 `--value-scores` 指定其他评分目录。上例中的 `--output-dir` 与默认的 `<run-dir>/traces_recovery/` 一致，可以省略。可用 `--episode 0` 配合单独的输出目录验证单个回合。有接管建议的回合从评分指定的状态边界切换到 GRAIL，运行到原回合终点。其余回合复制到输出目录。[离线评分命令](grail_offline.md#离线评估接管时刻)

中断后，用原来的输入、评分和模型设置加上 `--resume`。命令跳过已经完整写入的回合，从头生成中途停止的回合。

```bash
python -m gear_sonic.scripts.grail_offline recovery \
  --run-dir outputs/gr00t_rollouts/20261125_191637-18053 \
  --resume-from outputs/gr00t_rollouts/20261125_191637-18053/.traces_recovery-tuuxnwvs --num-envs 4 --cpu-workers 4
```

首次运行时，输出目录旁的 `.<输出目录名>.partial/` 保留已完成的回合，全部完成后发布到输出目录。`--resume` 也可以继续补齐已经发布的输出目录。旧版进程停止后若留下与输出目录同级的 `.traces_recovery-*` 临时目录，可用 `--resume-from <临时目录>` 接续。命令会将旧临时目录移到新版的续跑位置，之后再次中断时可用 `--resume`。

**升级提醒。** 同步服务器代码后，重新编译 `grail_offline_decoder`，使它能接收接管前的历史。新版 GR00T 每个状态边界保存 MuJoCo 积分状态 `mujoco_state`，每个动作步保存 decoder 原始输出 `decoder_action`。接管时直接恢复保存的积分状态，包括机器人和物体的位姿、速度、mocap、仿真时钟及求解器热启动数据。[编译命令](../getting_started/grail_environment.md)

旧轨迹使用保存的 `command_q`、`command_dq`、`command_kp`、`command_kd` 和 `command_tau` 回放，在每个控制边界对齐原始记录，恢复 MuJoCo 接触求解热启动和任务状态。接管前最后一步在对齐前比较回放结果与记录。只要 `qpos` 的最大分量误差超过 `1e-4`，或 `qvel` 的最大分量误差超过 `1e-2`，命令就会报出回合和步号并停止发布结果。

接管时，仿真沿用原场景、物体和释放目标。若任务状态已经确认掉落，GRAIL 从当前物理状态重新进入抓取阶段。首拍沿用上一步 GR00T 的关节命令，随后在 10 个控制周期内平滑切换到 GRAIL。
