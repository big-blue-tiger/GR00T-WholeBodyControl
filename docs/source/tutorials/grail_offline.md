# GRAIL 数据生成与 GR00T 训练

示范采集、渲染、评分和接管使用 `gr00t-sim`，从 WholeBodyControl 仓库根目录运行。提前准备 `100obj` 物体资产、红桶扫描场景和 `grail_offline_decoder`。[仿真环境与资产配置](../getting_started/grail_environment.md)

红桶场景的 `scene_combined_3dgs.usda`、`splats.nurec` 和 `scene_combined.usda` 放在 `gear_sonic/data/scene/deploy/`。

(grail-demo-train)=
## 2. GRAIL 生成示范并训练 GR00T

### 采集、渲染、导出

`collect` 从初始状态运行 GRAIL，默认对 `100obj` 中每个物体采集 5 条轨迹。`render` 为录下的状态生成 50 fps 第一人称视频。`export` 将成功轨迹和对应视频写成 LeRobot v2.1。

```bash
conda activate gr00t-sim
DEMO_TRACES=outputs/throw_trash_into_bin
DEMO_RENDERS=outputs/throw_trash_into_bin_render
DEMO_DATASET=outputs/throw_trash_into_bin_dataset

python -m gear_sonic.scripts.grail_offline collect \
  --output "$DEMO_TRACES" --num-envs 3 --cpu-workers 3

python -m gear_sonic.scripts.grail_offline render \
  --traces "$DEMO_TRACES" --output "$DEMO_RENDERS" --render-envs 2 --resume

python -m gear_sonic.scripts.grail_offline export \
  --traces "$DEMO_TRACES" --renders "$DEMO_RENDERS" \
  --output "$DEMO_DATASET" --labels success
```

物体到释放目标的整条轨迹最小距离小于 0.2 m 时标为成功。采集时自动标注，已有未标注轨迹可用 `python -m gear_sonic.scripts.grail_offline auto-label --traces "$DEMO_TRACES"` 补标。导出目录须为新目录，训练读取其中的 `data/`、`videos/` 和 `meta/`。

已有示范数据时，可以从下方的训练步骤开始。另开一轮采集时，将这三个输出变量设为新路径，并在训练命令中使用新的数据集路径。

(gr00t-training-env)=
### 从基础模型训练 GR00T

在训练容器中准备 `gr00t-n17`、Isaac-GR00T 和本地模型缓存。在同一容器中可以直接读取 WholeBodyControl 的导出目录。数据存放在其他位置时，改 `--dataset-path` 即可。

以下环境设置用于本阶段和阶段 4。单卡容器也应准备与 PyTorch 匹配的 CUDA Toolkit，供已安装的 DeepSpeed 初始化。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t-n17
cd /home/Isaac-GR00T

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_CACHE=/home/Isaac-GR00T/models/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1 GROOT_HF_LOCAL_FIRST=1 NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0
```

使用 `UNITREE_G1_SONIC` 对示范执行普通行为克隆（SFT）。以下步数和批大小是训练示例，输出目录用于后续推理及接管数据微调。

```bash
USE_WANDB=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=2 MAX_STEPS=25583 SAVE_STEPS=2000 \
EPISODE_SAMPLING_RATE=1.0 NUM_SHARDS_PER_EPOCH=338 \
bash examples/finetune.sh \
  --base-model-path /home/Isaac-GR00T/models/GR00T-N1.7-3B \
  --dataset-path /home/GR00T-WholeBodyControl/outputs/throw_trash_into_bin_dataset \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass \
  -- --gradient-accumulation-steps 2
```

训练完成后，用该模型进行[阶段 3 同步采集与评分](gr00t_simulation.md#gr00t-sync-collect)。

(gr00t-recovery-train)=
## 4. GRAIL 接管、重渲染与 GR00T 加权训练

### 依据评分接管

从 WholeBodyControl 仓库根目录运行。将 `RUN_DIR` 指向阶段 3 已评分的运行目录。`recovery` 读取原始 `traces/` 和 `gr00t_reward_value_scores/episodes.jsonl`，保留接管点前的 GR00T 轨迹，并让 GRAIL 续跑至原回合结束。没有接管建议的回合原样复制。

```bash
conda activate gr00t-sim
RUN_DIR=outputs/gr00t_rollouts/20261125_191637-18053

python -m gear_sonic.scripts.grail_offline recovery \
  --run-dir "$RUN_DIR" --num-envs 4 --cpu-workers 4 --resume
```

输出为 `"$RUN_DIR/traces_recovery"`。新版轨迹直接恢复 MuJoCo 积分状态，旧版轨迹回放已记录的关节命令。GRAIL 在 10 个控制周期内平滑接管，已确认掉落时从当前状态重新抓取。接管需要能读取历史输入的 decoder。更新旧部署后按[环境配置](../getting_started/grail_environment.md)重新编译。

### 重渲染、重评分与导出

沿用同一个 `RUN_DIR`。接管后的轨迹需要自己的录像和 TD Advantage 分数。`--resume` 可继续未完成的渲染和评分。

```bash
python -m gear_sonic.scripts.grail_offline render \
  --traces "$RUN_DIR/traces_recovery" \
  --output "$RUN_DIR/traces_recovery_render" --render-envs 4 --resume

python -m gear_sonic.scripts.score_grail_rollouts \
  --traces "$RUN_DIR/traces_recovery" \
  --output "$RUN_DIR/traces_recovery_scores" --workers 24 --resume

python -m gear_sonic.scripts.grail_offline export \
  --traces "$RUN_DIR/traces_recovery" --renders "$RUN_DIR/traces_recovery_render" \
  --value-scores "$RUN_DIR/traces_recovery_scores" \
  --output outputs/gr00t_dataset_recovery_td_advantage --labels success
```

成功标签用于筛选导出，原样复制的成功回合也会纳入。`td_advantage` 写入 Parquet，按动作执行后的转移对齐。`debug/advantage/` 的 PNG 显示逐帧分数、40 帧加权分数和训练权重。`meta/source_episodes.json` 可用来查找导出编号对应的原始回合。

重新导出时选用新目录，并更新下方的 `RECOVERY_DATASET`。

### 用接管数据做 Advantage-weighted BC

先执行[训练环境设置](#gr00t-training-env)。从阶段 2 的 `fullpass` 里选步数最大的 checkpoint。加权 BC 使用 `--advantage-weighted-sft`，把同一 40 帧动作块内的 TD Advantage 转为动作 loss 权重。下例先跑加权 BC，成功退出后再从同一 checkpoint 跑普通 SFT 对照。

```bash
RECOVERY_DATASET=/home/GR00T-WholeBodyControl/outputs/gr00t_dataset_recovery_td_advantage
FULLPASS_DIR=/home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_fullpass
BASE_CHECKPOINT=$(find "$FULLPASS_DIR" -maxdepth 1 -regextype posix-extended \
  -type d -regex '.*/checkpoint-[0-9]+' | sort -V | tail -n 1)
printf '起始模型 %s\n' "${BASE_CHECKPOINT:-未找到 checkpoint}"

test -n "$BASE_CHECKPOINT" &&
USE_WANDB=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=2 MAX_STEPS=60000 SAVE_STEPS=3000 \
EPISODE_SAMPLING_RATE=1.0 NUM_SHARDS_PER_EPOCH=338 \
bash examples/finetune.sh \
  --base-model-path "$BASE_CHECKPOINT" --dataset-path "$RECOVERY_DATASET" \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_recovery_weighted \
  -- --gradient-accumulation-steps 2 --advantage-weighted-sft &&

USE_WANDB=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=2 MAX_STEPS=60000 SAVE_STEPS=3000 \
EPISODE_SAMPLING_RATE=1.0 NUM_SHARDS_PER_EPOCH=338 \
bash examples/finetune.sh \
  --base-model-path "$BASE_CHECKPOINT" --dataset-path "$RECOVERY_DATASET" \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_recovery_sft \
  -- --gradient-accumulation-steps 2

USE_WANDB=0 NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=2 MAX_STEPS=60000 SAVE_STEPS=3000 \
EPISODE_SAMPLING_RATE=1.0 NUM_SHARDS_PER_EPOCH=338 \
bash examples/finetune.sh \
  --base-model-path "$BASE_CHECKPOINT" --dataset-path "/home/GR00T-WholeBodyControl/outputs/gr00t_dataset_grail_matched_recovery" \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /home/Isaac-GR00T/outputs/throw_trash_into_bin_n17_success_data_sft \
  -- --gradient-accumulation-steps 2
  
```

用加权训练结果进入[阶段 5 在线异步推理](gr00t_simulation.md#gr00t-async-inference)。
