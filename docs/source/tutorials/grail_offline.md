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

Isaac Sim 根据每条轨迹引用的场景配置读取对应的物体 USD。默认资产位于 `gear_sonic/data/grail/100obj/object_usd/`。

```bash
python -m gear_sonic.scripts.grail_offline render \
  --traces outputs/throw_trash_into_bin --output outputs/throw_trash_into_bin_render \
  --render-envs 2 --encode-workers 4
```

| 参数 | 说明 |
|---|---|
| `--traces` / `--output` | 轨迹目录和渲染输出目录。每条轨迹生成 `ego_view.mp4` |
| `--render-envs` / `--encode-workers` | Isaac 渲染进程并发上限和视频编码进程数，默认 `1` 和 `4` |
| `--resume` | 跳过轨迹和配置匹配的完整结果，继续渲染其余轨迹 |
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
  --output outputs/my_grail_dataset --labels success
```

| 参数 | 说明 |
|---|---|
| `label --episode` / `--label` | 从 `0` 开始的轨迹编号，标签可选 `success`、`failure` 或 `unlabeled` |
| `export --traces` / `--renders` | 轨迹目录和对应的渲染目录 |
| `export --output` | GR00T N1.7 数据集输出目录，须为新目录 |
| `export --labels` | 默认导出 `success`。可用逗号分隔标签，或用 `all` 导出全部轨迹 |

## 校验渲染结果

```bash
python -m gear_sonic.scripts.validate_grail_render \
  --traces outputs/my_grail_run --renders outputs/my_grail_render \
  --report outputs/my_grail_render/validation.json
```

`--traces` 和 `--renders` 指定核对目录，`--report` 指定报告路径。

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

采集测速使用 `collect` 的参数。渲染测速中，`--episodes` 指定每轮使用的轨迹数，`--trials` 指定重复次数，其他参数与 `render` 相同。使用新输出目录。
