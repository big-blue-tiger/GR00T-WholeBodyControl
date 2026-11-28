# GRAIL 与 GR00T 红桶任务流程

| 阶段 | 工作 | 产物 |
| --- | --- | --- |
| 1 | [GRAIL 策略异步推理测试](#grail-async-test) | 实时画面和任务日志 |
| 2 | [GRAIL 生成示范并训练 GR00T](grail_offline.md#grail-demo-train) | LeRobot 示范数据和第一版 GR00T 模型 |
| 3 | [GR00T 离线同步推理、采集与评分](gr00t_simulation.md#gr00t-sync-collect) | 原始轨迹、视频、导出数据和接管建议 |
| 4 | [GRAIL 接管、重渲染与 GR00T 加权训练](grail_offline.md#gr00t-recovery-train) | 带 TD Advantage 的 LeRobot 数据和加权 BC 模型 |
| 5 | [GR00T 在线异步推理](gr00t_simulation.md#gr00t-async-inference) | 实时预览和回合录像 |

(grail-async-test)=
## 1. GRAIL 策略异步推理测试

配置 `gr00t-sim` 和实时 SONIC decoder。在 GNOME 桌面终端进入仓库根目录，加载运行环境后启动。[GRAIL 环境配置](../getting_started/grail_environment.md)

```bash
conda activate gr00t-sim
POLICY_BACKEND=grail DEVICE=cuda:0 bash gear_sonic/scripts/run_grail_carry.sh
```

仿真和 GRAIL 策略在独立进程中交换观测与动作。Decoder 就绪后，在它的窗口按 **`]`**，再按 **Enter** 开始。Isaac 和 MuJoCo 相机窗口显示执行画面，仿真窗口显示 Reward、Value 和 TD Advantage。

默认从 `1600obj` 的皂液瓶开始，回合结束后轮换物体。用 `OBJECT_KEY` 选起始物体，用 `CHECKPOINT` 指定 GRAIL 权重。水蜜桃瓶示例可用 `run_grail_carry_peach_bottle.sh` 启动。桌面遮挡由 `gear_sonic/config/grail/apple_2.yaml` 的场景配置控制。

回合写入 `outputs/grail-episodes.jsonl`，仿真日志写入 `outputs/grail-simulation.log`。仿真、Policy 和 Decoder 窗口按 **Ctrl+C** 退出，相机窗口按 **Q**。

### 抓握失败恢复测试

```bash
POLICY_BACKEND=grail DEVICE=cuda:0 bash gear_sonic/scripts/run_grail_recovery.sh
```

启动方式相同。该测试在桌上物体附近随机放置手部，每 5 秒仿真时间重置。结果写入 `outputs/grail-recovery-episodes.jsonl`。从 GR00T 轨迹指定步接管的命令见[阶段 4](grail_offline.md#gr00t-recovery-train)。

验证 GRAIL 策略后，进入[阶段 2 生成训练数据](grail_offline.md#grail-demo-train)。

## 同步项目

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