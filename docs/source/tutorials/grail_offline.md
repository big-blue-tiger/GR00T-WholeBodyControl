# GRAIL 离线采集、渲染与导出

首次按 [GRAIL 环境配置](../getting_started/grail_environment.md)安装。每个启动终端加载同一套[运行环境](../getting_started/grail_environment.md#运行环境)，从仓库根目录执行命令。

## 1. 采集轨迹

```bash
python -m gear_sonic.scripts.grail_offline collect \
  --output outputs/throw_trash_into_bin \
  --dataset-dir gear_sonic/data/grail/100obj \
  --num-envs 3 --episodes-per-object 5 --cpu-workers 3
```

默认按物体名顺序采集 `100obj` 中的全部 100 个物体，每个物体 5 条，共 500 条。机器人在以本回合物体为圆心的桌外侧圆弧上采样，水平半径为 1.85 米。以垂直桌边、水平向外的方向为 0°，默认角度为 −45° 到 +45°，面向桌面看，负角在右侧，正角在左侧。释放目标位于扫描场景中红色垃圾桶的桶口中心上方约 0.35 米，世界坐标约为 `(0.015, 4.137, 0.850)` 米。采集记录每次策略执行的轨迹，并在保存时自动标注。

每条轨迹的 `episode.json` 会保存自然语言任务描述。物体名来自资产名，例如 `pickup_table__apple_0__001` 对应 `apple`。生成的转身命令中，正 yaw 表示向左后方转身，负 yaw 表示向右后方转身，零则省略转身描述。导出会沿用每条轨迹的描述，例如 `Pick up the apple from the table, turn around to the left, and release it above the red bin.`

| 参数 | 说明 |
|---|---|
| `--output` | 轨迹输出目录，须为新目录或没有已采集轨迹的目录 |
| `--num-envs` | 并行环境数，默认 `1`。总轨迹数须至少等于并行环境数 |
| `--episodes-per-object` | 每个物体的采集条数，默认 `5`；`--num-episodes` 是同一参数的别名 |
| `--cpu-workers` | CPU 仿真进程数，默认 `4` |
| `--duration` / `--seed` | 每条轨迹的时长（秒）和随机种子，默认 `25` 和 `42` |
| `--device` / `--threads` | 顶层策略推理设备和 PyTorch 线程数，默认 `cuda:0` 和 `1` |
| `--task` | 自定义任务描述。可用 `{object}` 插入物体名，用 `{turn}` 插入转身短语（含逗号） |
| `--checkpoint` | 覆盖默认权重，可传入 `.pt` 文件或目录。自定义权重须支持 884 维输入，并与其 `config.yaml` 同目录 |
| `--dataset-dir` / `--object-key` | 默认数据集为 `gear_sonic/data/grail/100obj`。可重复传入 `--object-key` 来采集指定物体 |
| `--cache-dir` | 覆盖 MuJoCo 预转换目录。`100obj` 使用其下的 `object_obj/` |
| `--initial-pose` / `--scene-config` | 默认初始姿态与实时搬运共用 `gear_sonic/config/grail/carry_initial_pose.json`。默认释放点由场景中的 `offline_release_position_scene` 设置 |
| `--no-release-override` | 关闭强制松手和松手后的 latent 锁定 |

切换物体前先取得[原始资产和预转换结果](../getting_started/grail_environment.md#批量准备其他物体)。跨机器移动轨迹用于渲染时，保留轨迹与资产之间的相对目录布局。

## 2. 渲染图像

Isaac Sim 根据每条轨迹引用的场景读取物体 USD，默认位于 `gear_sonic/data/grail/100obj/object_usd/`。默认红桶场景与 GR00T 在线推理共用分层 3DGS 渲染。每帧用轨迹中同一时刻的头部相机位姿并行渲染 NuRec 背景与机器人、任务物体前景，再合成 640×480 的 `ego_view`。桌面深度网格遮挡前景，桌面颜色来自 NuRec 背景。

默认场景的 `gear_sonic/data/scene/deploy/` 需要同时包含 `scene_combined_3dgs.usda`、`splats.nurec` 和 `scene_combined.usda`。使用该扫描场景的旧轨迹可以直接重新渲染，已有渲染目录可用 `--resume` 更新。采集时可在场景配置里设 `show_colliders: true`，让回放使用单次混合渲染显示对齐网格。

```bash
python -m gear_sonic.scripts.grail_offline render \
  --traces outputs/gr00t_rollouts/20261125_191637-18053/traces_recovery --output outputs/gr00t_rollouts/20261125_191637-18053/traces_recovery_render \
  --render-envs 4 --encode-workers 4 --resume
```

| 参数 | 说明 |
|---|---|
| `--traces` / `--output` | 轨迹目录和渲染输出目录。每条轨迹生成 50 fps 的 `ego_view.mp4` |
| `--render-envs` / `--encode-workers` | 回放任务并发上限和视频编码进程数，默认 `1` 和 `4`。启用分层渲染时，每个回放任务使用两个 Isaac 进程 |
| `--resume` | 跳过轨迹、渲染配置和渲染实现匹配的完整结果，继续渲染其余轨迹 |
| `--overwrite` | 覆盖已有渲染结果，与 `--resume` 互斥 |

## 3. 标注与导出

自动标注使用世界坐标，计算物体原点与本回合释放目标点在整条轨迹上的最小三维欧氏距离（包括首帧和末帧）。最小距离严格小于 `0.2 m` 就标为 `success`，其余标为 `failure`。`episode.json` 保存 `label` 和 `min_target_distance_m`。这个标准衡量物体是否到达释放目标附近。

已有轨迹可以批量补标，无需渲染。`auto-label` 只处理 `unlabeled` 轨迹。

```bash
python -m gear_sonic.scripts.grail_offline auto-label \
  --traces outputs/throw_trash_into_bin
```

如果视频确认第 0 条符合任务要求，可以用 `label` 手动将它标为 `success`。

```bash
python -m gear_sonic.scripts.grail_offline label \
  --traces outputs/throw_trash_into_bin --episode 0 --label success

python -m gear_sonic.scripts.grail_offline export \
  --traces outputs/throw_trash_into_bin --renders outputs/throw_trash_into_bin_render \
  --output outputs/throw_trash_into_bin_dataset --labels success
```

| 参数 | 说明 |
|---|---|
| `label --episode` / `--label` | 从 `0` 开始的轨迹编号，标签可选 `success`、`failure` 或 `unlabeled` |
| `export --traces` / `--renders` | 轨迹目录和对应的渲染目录 |
| `export --output` | GR00T N1.7 数据集输出目录，须为新目录 |
| `export --labels` | 默认导出 `success`。可用逗号分隔标签，或用 `all` 导出全部轨迹 |
| `export --value-scores` | 可选。将对应轨迹的逐步 TD Advantage 写入 LeRobot 数据集 |

### 导出成功的 Recovery 回合并加权训练

对拼接后的 `traces_recovery/` 重新评分。每条动作使用该动作执行后的转移分数。导出沿用 LeRobot v2.1，只增加一个 `float32` 列 `td_advantage`。

指定 `--value-scores` 时，每个导出的回合还会在数据集的 `debug/advantage/` 生成一张 PNG，文件名与 `data/` 中的回合编号一致。图中显示逐帧 TD Advantage、从当前帧开始的 40 帧加权 TD Advantage，以及训练 loss 权重。窗口不足 40 帧的尾部只显示逐帧分数。

```bash
run_dir=outputs/gr00t_rollouts/20261125_191637-18053

python -m gear_sonic.scripts.grail_offline auto-label \
  --traces "$run_dir/traces_recovery"

python -m gear_sonic.scripts.score_grail_rollouts \
  --traces "$run_dir/traces_recovery" \
  --output "$run_dir/traces_recovery_scores" --workers 8 --resume


python -m gear_sonic.scripts.grail_offline export \
  --traces "$run_dir/traces_recovery" \
  --renders "$run_dir/traces_recovery_render" \
  --value-scores "$run_dir/traces_recovery_scores" \
  --output "outputs/gr00t_dataset_recovery_td_advantage" --labels success
```

在训练容器里将 `RECOVERY_DATASET` 指向导出的数据集目录，启用 Isaac-GR00T 的 `--advantage-weighted-sft`。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-n17
cd /home/Isaac-GR00T

export HF_HUB_CACHE=/home/Isaac-GR00T/models/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1 GROOT_HF_LOCAL_FIRST=1 NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0
RECOVERY_DATASET=/home/Isaac-GR00T/dataset/throw_trash_into_bin_recovery_td_advantage

USE_WANDB=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=2 MAX_STEPS=25583 SAVE_STEPS=2000 \
SHARD_SIZE=1024 EPISODE_SAMPLING_RATE=1.0 NUM_SHARDS_PER_EPOCH=338 \
bash examples/finetune.sh \
  --base-model-path /home/Isaac-GR00T/models/GR00T-N1.7-3B \
  --dataset-path "$RECOVERY_DATASET" \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_recovery_weighted \
  -- --gradient-accumulation-steps 2 --advantage-weighted-sft
```

训练在同一 40 帧动作窗口内，将 TD Advantage 逐帧乘以从 2 线性降到 0 的系数后取均值 `x`。整个 chunk 的权重为 `min(exp(6.578815 * x), 1)`。权重乘到现有动作 loss 上，分母仍是有效动作元素数。省略训练开关时运行普通 SFT。

## 导出 GR00T 在线回合

[GR00T 红桶任务](grail_carry.md#gr00t-红桶任务)的同步和异步模式都会保存每个控制步的第一人称合成画面与实际执行的动作。每次运行目录中的 `traces/` 含状态、动作和回合标签，`renders/` 含 50 fps MP4 和帧索引。仿真进程退出后，启动脚本将已完成的成功和失败回合统一导出为 LeRobot v2.1，保存到 `dataset/`。每条动作对应一帧与一个状态转移，25 秒完整回合有 1,250 条动作和 1,251 个状态边界。

旧运行可以在 `gr00t-sim` 环境中补导。将 `RUN_DIR` 指向运行目录，再执行以下命令。

```bash
python -m gear_sonic.scripts.grail_offline export \
  --traces "$RUN_DIR/traces" --renders "$RUN_DIR/renders" \
  --output "$RUN_DIR/dataset" --labels all
```

只需成功回合时，使用 `--labels success`，并把 `--output` 指向新目录，例如 `"$RUN_DIR/dataset_success"`。

## 离线评估接管时刻

评分读取原始回合中的状态边界、场景、任务和实际执行的关节命令。LeRobot 导出数据的 `meta/source_episodes.json` 将训练集索引映射回原始回合。TD Advantage 的 1 秒滑动均值连续 5 步小于等于 `-0.35` 时记录接管建议。任务计划发出释放请求，或物体首次进入释放目标 0.2 米内后，停止生成新的建议。

```bash
python -m gear_sonic.scripts.score_grail_rollouts \
  --traces "outputs/gr00t_rollouts/20261124_184257-6603/traces" \
  --output outputs/gr00t_rollouts/20261124_184257-6603/gr00t_reward_value_scores
```

通过重复传入 `--episode` 可只评分指定回合。输入需要 `episode.json`、`trace.npz` 及引用的 `scene.json`、`model.mjb`。`trace.npz` 必须含 `command_q`、`command_dq`、`command_kp`、`command_kd` 和 `command_tau`。逐步分数以 NPZ 保存，`episodes.jsonl` 记录原始回合和分数文件的路径、评分权重、折扣系数及首次接管建议。

`--workers` 设置并行评分的 CPU 进程数，默认 `1`。每完成一个回合就更新 `episodes.jsonl`。中断后用原来的输入、输出和 checkpoint 加 `--resume`，已登记且校验通过的回合会跳过，缺少索引或分数文件的回合会重新评分。若旧版评分还在运行，先等它退出并写出索引，再使用 `--resume`。

用 `--batch` 将整个数据集的分数保存到以来源命名的子目录。

```bash
run_dir="outputs/gr00t_rollouts/20261125_191637-18053"

python -m gear_sonic.scripts.score_grail_rollouts \
  --traces "$run_dir/traces" \
  --output "$run_dir/gr00t_reward_value_scores"
```

例如 `throw_trash_into_bin/episode_000000/trace.npz` 的分数保存为 `outputs/gr00t_reward_value_scores/throw_trash_into_bin/episode_000000.npz`。`--traces` 可接收多个目录。输出根目录中的 `episodes.jsonl` 先按输入目录顺序排列，再按原始回合编号升序排列。

### 可视化单次运行的打分结果

完成评分后，在同一个 Python 环境中运行：

```bash
run_dir="outputs/gr00t_rollouts/20261125_191637-18053"

python -m gear_sonic.scripts.visualize_grail_scores \
  --traces "$run_dir/traces" \
  --value-scores "$run_dir/gr00t_reward_value_scores" \
  --output "$run_dir/gr00t_reward_value_plots"
```

评分目录应包含该次运行的 `episodes.jsonl` 和 NPZ 文件，截图读取与 `--traces` 同级的 `renders/`。每回合生成一张曲线图，分别显示 Value 和 TD Advantage 的原始曲线及 50 步滑动均值，并标出评分记录中的接管时刻。有接管建议的回合还会按录像时间戳截取第一人称画面，保存为 `episode_XXXXXX_recovery.png`。加 `--episode 0` 可先查看一个回合，或重复传入 `--episode` 选择多个回合。

## 校验渲染结果

```bash
python -m gear_sonic.scripts.validate_grail_render \
  --traces outputs/throw_trash_into_bin --renders outputs/throw_trash_into_bin_render \
  --report outputs/throw_trash_into_bin_render/validation.json
```

`--traces` 和 `--renders` 指定核对目录，`--report` 指定报告路径。校验会解码合成录像，核对 640×480 分辨率、50 fps、逐帧时间戳以及帧索引与原始轨迹的对应关系。

## 测速

```bash
python -m gear_sonic.scripts.grail_offline benchmark \
  --output outputs/my_grail_benchmark \
  --dataset-dir gear_sonic/data/grail/1600obj \
  --object-key pickup_table__soap_dispenser_0__001 \
  --num-envs 3 --episodes-per-object 3 --cpu-workers 3

python -m gear_sonic.scripts.benchmark_grail_render \
  --traces outputs/my_grail_benchmark --output outputs/my_grail_render_benchmark \
  --episodes 3 --trials 1 --render-envs 3 --encode-workers 4
```

采集测速使用 `collect` 的参数。渲染测速复用 `render` 的分层渲染和编码流程。`--episodes` 指定每轮使用的轨迹数，`--trials` 指定重复次数，其他参数与 `render` 相同。使用新输出目录。


```bash
rsync -avhP --no-owner --no-group -e 'ssh -p 9980' \
  --include='/gear_sonic_deploy/policy/***' \
  --exclude={'/.git/','/.agents/','/.codex/','/.aws/','/.venv*/','.env','/outputs/'} \
  --exclude={'__pycache__/','*.py[cod]','.cache/','.pytest_cache/','.mypy_cache/','.ruff_cache/','*.log'} \
  --exclude={'/gear_sonic_deploy/target/','/gear_sonic_deploy/build*/'} \
  --exclude='/gear_sonic/data/grail/1600obj/' \
  /home/tide/robot/GR00T-WholeBodyControl/ \
  yuguanchao@zz.irmv.top:/home/yuguanchao/data0/GR00T-WholeBodyControl/
```
