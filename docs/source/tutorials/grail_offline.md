# GRAIL 离线采集与 Isaac 批量渲染

在仓库根目录执行，依次采集轨迹、渲染图像、标注并导出 GR00T N1.7 数据集。

## 安装

```bash
bash install_scripts/install_grail_offline.sh
conda activate gr00t-sim
```

安装脚本默认使用 `/home/tide/miniconda3/envs/gr00t-sim/bin/python`，可通过环境变量 `PYTHON` 指定解释器。

## 采集轨迹

```bash
python -m gear_sonic.scripts.grail_offline collect \
  --output outputs/my_grail_run \
  --checkpoint gear_sonic_deploy/policy/grail_grasp_policy/1109robocasa_carry_table-randomized-20261108_164252/model_step_009000.pt \
  --num-envs 100 --num-episodes 200 --cpu-workers 4 \
  --task "Pick up the object from the table, carry it, and release it."
```

| 参数 | 说明 |
|---|---|
| `--output` | 轨迹输出目录，必填 |
| `--num-envs` | 并行环境数，默认 `1` |
| `--num-episodes` | 总轨迹数，默认 `max(2, num_envs)` |
| `--cpu-workers` | CPU 仿真进程数，默认 `4` |
| `--duration` | 每条轨迹的仿真时长（秒），默认 `25` |
| `--seed` | 随机种子，默认 `42` |
| `--device` / `--threads` | 顶层策略推理设备与 PyTorch 线程数，默认 `cuda:0` / `1` |
| `--task` | 导出数据集使用的任务描述 |
| `--checkpoint` | 策略权重文件或目录。示例直接指定 `model_step_009000.pt`；传入目录时按修改时间选择最新完整 `.pt` |
| `--initial-pose` | 机器人初始姿态文件 |
| `--dataset-dir` / `--object-key` | GRAIL 资产目录与物体标识，默认物体为 `pickup_table__soap_dispenser_0__001` |
| `--scene-config` | 场景配置，默认 `gear_sonic/config/grail/apple_2.yaml` |
| `--cache-dir` | 资产缓存目录，默认 `outputs/grail_asset_cache` |
| `--no-release-override` | 关闭默认启用的松手动作旁路 |

采集与实时入口共用新版 GRAIL 的观测及 reference-free sparse command，完整逐项定义见 [观测对齐说明](grail_carry.md#新版-grail-观测与-sparse-command)。仅支持 `805 + 79 = 884` 维 checkpoint：`policy_proprio` 保存 `[N,5,161]`，`policy_task` 保存 `[N,79]`；旧的桌角、8 维接触力和 `drop_button` 不再作为策略输入。`scene.json` 记录实际选中的权重文件。

拾取稳定 3 拍并等待 2 秒后开始播放；每拍按当前 pelvis 到固定 base 目标的 XY 距离计算 `dx`，上限 `0.15 m`，转弯帧内 `dyaw` 取规划的完整转角。阶段切换、接触门控与实时入口相同。最后松手旁路保留，通过独立的 `release_requested` 字段控制并写入轨迹；`--no-release-override` 仅关闭手部强制打开，不修改策略观测或 sparse command。

默认资产目录仍为本机已有的 `pickup_table_slowturn_300`。若要换用训练使用的 `pickup_table_slowturn_1600_no_release`，下载后通过 `--dataset-dir` 指定；采集不读取机器人参考 motion。

## 渲染图像

```bash
python -m gear_sonic.scripts.grail_offline render \
  --traces outputs/my_grail_run --output outputs/my_grail_render \
  --render-envs 3 --encode-workers 4
```

| 参数 | 说明 |
|---|---|
| `--traces` | 已采集的轨迹目录，必填 |
| `--output` | 渲染输出目录，必填 |
| `--render-envs` | 独立 Isaac 渲染进程的并发上限，正整数，默认 `1` |
| `--encode-workers` | 视频编码进程数，默认 `4` |
| `--resume` | 续跑：跳过源轨迹和配置匹配的完整结果，重新生成其余结果 |
| `--overwrite` | 覆盖已有渲染结果，与 `--resume` 互斥 |

第一人称 RGB 视频使用与实时推理相同的 D435i 模拟配置。新采集把相机参数保存到 `model.mjb`。用旧默认相机采集的轨迹可以直接重渲染，旧视频可用 `--resume` 或 `--overwrite` 更新。[实时推理教程](grail_carry.md)

输出为 640×480、50 FPS 的 `ego_view.mp4`。视频逐帧对应 50 Hz 的策略轨迹，实际渲染不限速。采集环境数与渲染并发数独立设置。

## 标注与导出

```bash
python -m gear_sonic.scripts.grail_offline label \
  --traces outputs/my_grail_run --episode 0 --label success

python -m gear_sonic.scripts.grail_offline export \
  --traces outputs/my_grail_run --renders outputs/my_grail_render \
  --output outputs/my_grail_dataset --labels success
```

| 参数 | 说明 |
|---|---|
| `label --episode` | 要标注的轨迹编号，从 `0` 开始 |
| `label --label` | `success`、`failure` 或 `unlabeled` |
| `export --traces` / `--renders` | 轨迹目录与对应的渲染目录，必填 |
| `export --output` | 数据集输出目录，必须为新目录 |
| `export --labels` | 默认仅导出 `success`；可用逗号分隔多个标签，`all` 包含未标注轨迹 |

## 校验渲染结果

```bash
python -m gear_sonic.scripts.validate_grail_render \
  --traces outputs/my_grail_run --renders outputs/my_grail_render \
  --report outputs/my_grail_render/validation.json
```

`--traces` 和 `--renders` 指定待核对的轨迹与视频目录，`--report` 指定检查报告路径。检查包括视频帧数、分辨率、时间戳及轨迹对齐。

## 测速

采集测速，参数与 `collect` 相同，也会保存轨迹：

```bash
python -m gear_sonic.scripts.grail_offline benchmark \
  --output outputs/my_grail_benchmark \
  --checkpoint gear_sonic_deploy/policy/grail_grasp_policy/1109robocasa_carry_table-randomized-20261108_164252/model_step_009000.pt \
  --num-envs 3 --num-episodes 3 --cpu-workers 3
```

渲染测速，使用前 3 条轨迹、3 个独立 Isaac 进程：

```bash
python -m gear_sonic.scripts.benchmark_grail_render \
  --traces outputs/my_grail_benchmark --output outputs/my_grail_render_benchmark \
  --episodes 3 --trials 1 --render-envs 3 --encode-workers 4
```

`--episodes` 指定每轮使用的前 N 条轨迹，默认 `100`；`--trials` 指定重复轮数，默认 `3`；其余参数与 `render` 相同。`--output` 必须为新目录。

`benchmark.json` 记录各轮和中位吞吐：`median_rendering_fps` 包含 Isaac 启动和退出开销，`median_pipeline_fps` 统计启动后的渲染、读回和编码流水线。结果为所有渲染进程的聚合帧率；每轮视频位于 `trial_00/episode_*/ego_view.mp4`，后续轮次依次编号。
