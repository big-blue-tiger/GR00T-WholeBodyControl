# GRAIL G1 抓取搬运启动

MuJoCo 物理 + PyTorch 顶层策略 + C++ GRAIL SONIC decoder。按顺序打开以下终端。
仿真与策略使用 `gr00t-sim`，资产准备使用 `grail`；命令已指定 Python 绝对路径。

## 一键启动

在本机 GNOME 桌面的终端执行：

```bash
bash gear_sonic/scripts/run_grail_carry.sh
```

脚本先准备资产，再自动打开仿真、顶层策略、C++ decoder 和两路相机的五个终端。
相机终端会等待渲染端口就绪，再打开查看窗口。
低延迟渲染默认生效，保留 640×480、完整 3DGS 背景及原有材质和抗锯齿设置。
更新代码后需退出旧进程并重新执行脚本，使渲染器和查看器加载修改。
decoder 就绪后，在其窗口按 **`]`**，再按 **Enter** 开启 streaming。
退出时，前三个窗口分别按 **Ctrl+C**，相机窗口按 **Q**。

可通过环境变量更换物体或推理设备：

```bash
OBJECT_KEY=pickup_table__apple_2__002 DEVICE=cuda:0 bash gear_sonic/scripts/run_grail_carry.sh
```

以下为对应的逐终端启动命令。

## 终端 1：准备资产并启动仿真

整段复制执行。已有有效资产缓存会复用；`/tmp` 缓存丢失后重新运行即可。
默认同时启动 Isaac 和 MuJoCo 相机渲染，MuJoCo 仍是唯一物理仿真。
`--object-key` 指定加载物体的 USD、BPS 和资产缓存；资产准备与仿真命令使用相同的 key。
`--scene-config` 控制桌子、物体初始旋转、接触、机器人和渲染参数，不绑定物体名称。
换物体时修改下面的 `OBJECT_KEY` 即可，仍可使用 `apple_2.yaml`，无需复制或修改 YAML。
`--initial-pose-file` 仅初始化机器人；物体采用场景配置中的默认旋转，位置沿用每次重置时的随机采样。
Isaac 默认加载本地 `gear_sonic/data/scene/3dgs/839951.usdz` 茶室，
使用参考图中的灰色茶几；地面和桌面碰撞仍由 MuJoCo 提供。

```bash
(
  set -e
  cd /home/tide/robot/GR00T-WholeBodyControl
  OBJECT_KEY=pickup_table__soap_dispenser_0__001

  /home/tide/miniconda3/envs/grail/bin/python -m gear_sonic.scripts.prepare_grail_assets \
    --dataset-dir /home/tide/robot/GRAIL/data/hf_dataset/data_update/data/pickup_table_slowturn_300 \
    --object-key "$OBJECT_KEY" \
    --cache-dir /tmp/sonic-grail-assets

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

## 终端 2：启动顶层策略

```bash
cd /home/tide/robot/GR00T-WholeBodyControl
/home/tide/miniconda3/envs/gr00t-sim/bin/python -m gear_sonic.scripts.run_grail_policy \
  --checkpoint /home/tide/robot/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/1020robocasa_carry_table-20261026_164704/model_step_005000.pt \
  --device cpu --threads 1
```

如需 GPU 推理，将 `--device cpu` 改为 `--device cuda:0`。

临时松手旁路默认启用：任务观测中的 `drop >= 0.5` 时，强制将 policy 输出的
左右手动作设为 `-1`，选择 checkpoint 中张开的 `pos_0` 手指姿态；身体 token 保持原样。
无需修改上面的启动命令；追加 `--no-release-override` 可关闭旁路，恢复模型原始输出。
手指仍经过现有 C++ 平滑和 PD 控制，不会瞬间改变实际关节位置。

## 终端 3：启动 C++ GRAIL decoder

使用 `gear_sonic_deploy/policy/grail_model/model_decoder_grail.onnx`，同目录为配套
TensorRT 缓存。保留原有 observation config；顶层已提供 token，无需加载 encoder。

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

C++ 就绪后，在**终端 3**按 **`]`** 启动控制，再按 **Enter** 开启 streaming。
确认出现 `ZMQ STREAMING MODE: ENABLED`。三个进程就绪前，仿真保持静态等待。
不要同时运行另一个占用 5556 端口的 teleop 发送器。

## 抓握接触参数

- 场景 XML：`gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml`。
- 机器人/手指碰撞 XML：同目录 `g1_29dof_with_hand.xml`；带 `contype="0" conaffinity="0"` 的 geom 仅用于显示。
- 物体及手—物体接触对由 `gear_sonic/utils/grail/scene.py` 动态生成，调参入口为
  `gear_sonic/config/grail/apple_2.yaml` 的 `grasp_contact`。

当前防滑参数：`condim=6`、`friction=[2,2,0.01,0.001,0.001]`、
`solref=[0.01,1]`、`solimp=[0.999,0.9999,0.001,0.5,2]`。
只覆盖 16 组手—物体接触；原脚底、桌面摩擦及手指力矩上限保持不变。
修改后重启终端 1 即生效，无需重新转换资产或编译 C++；设 `enabled: false` 可恢复原接触参数。
固定夹紧力的实际物体网格测试中，每侧 0.7 N 时 5 秒下滑量从 103 mm 降至 5.2 mm，
张开夹爪后正常落下；该夹具测试不代表完整策略抓取验收。

## 终端 4：查看 Isaac 相机窗口

机器人使用已纳入 Git LFS 的本地 `gear_sonic/data/robots/g1/g1_43dof_s3.usda`。
更新资产或渲染代码后需重启终端 1，使渲染进程重新加载；仅重开查看窗口不会更新模型。

搬运物体的 Isaac 外观直接引用数据集 `object_usd/<object-key>.usd`，
沿用 GRAIL `eval_agent_trl.py` 的原始 mesh、UV、OmniPBR 材质和贴图加载方式。
原始 USD 路径随场景传入独立渲染进程，贴图相对路径由 USD 解析；
不再从 MuJoCo 像素重建简化的物体材质。MuJoCo 继续使用准备好的 OBJ/纹理缓存。
旧流程直接将 MuJoCo 编译时已翻转 V 轴的 UV 用于 USD，导致苹果表面出现错位杂块；
直接引用原始 USD 后保留其原始 UV，避免这次错误转换。
两路渲染共享 MuJoCo 物体位姿，Isaac 中物体的刚体和碰撞均禁用。
光照、曝光和渲染器不同仍可能造成明暗差异。
此修复不需要重新转换已有有效缓存，重启终端 1 即可生效。

```bash
cd /home/tide/robot/GR00T-WholeBodyControl
/home/tide/miniconda3/envs/gr00t-sim/bin/python \
  gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
```

## 终端 5：查看 MuJoCo 相机窗口

```bash
cd /home/tide/robot/GR00T-WholeBodyControl
/home/tide/miniconda3/envs/gr00t-sim/bin/python \
  gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5559
```

相机窗口按 **R** 开始/停止录像，按 **Q** 关闭查看窗口。
MuJoCo 交互窗口随终端 1 自动打开，可按 **Backspace** 重置场景。

## 渲染选项

Isaac（终端 4）和 MuJoCo（终端 5）的第一人称画面统一为 **640 × 480**，
垂直视场角从 45° 扩大为 **75°**，对应水平视场角约 **91.3°**。
两路共用 `head_camera` 参数，Isaac 从 MuJoCo 模型读取视场角并匹配投影。
后续在 `gear_sonic/config/grail/apple_2.yaml` 中修改同一个配置即可同时调整两路：

```yaml
ego_camera:
  vertical_fov_deg: 75.0
```

修改后重启终端 1，使两个渲染器重新加载相机参数；仅重开查看窗口不会生效。

`gear_sonic/config/grail/apple_2.yaml` 中的 `isaac_scene` 控制 3DGS：

```yaml
isaac_scene:
  enabled: true
  asset_path: gear_sonic/data/scene/3dgs/839951.usdz
  translation: [4.16, 2.15, 0.342]
  quaternion_wxyz: [0.0, 0.0, 0.7071067811865476, -0.7071067811865476]
  show_colliders: false
```

将 **`show_colliders` 改为 `true`** 后重启终端 1，Isaac 会显示原地面和
2 × 0.6 × 0.04 米桌面的实体表面，方便检查与 3DGS 的重合位置；设为 `false`
隐藏这两个表面的渲染。两种模式下碰撞始终有效，机器人和所选物体继续显示。
MuJoCo 窗口及其相机继续显示碰撞桌面。`enabled: false` 关闭 3DGS 并恢复
Isaac 原地面、桌面的显示；桌高和初始位置仍由各自的配置字段控制。

场景只作刚体旋转和平移，比例保持 **1**，USDZ 内部原有变换保留。
该茶几的长边沿世界 X 轴，面向第二张参考图中空地的一侧为 **+Y**：
碰撞桌沿长边居中，外沿 `y=0.3` 对齐茶几靠空地的长边。
由于茶几比碰撞桌宽，优先对齐该边，碰撞桌在宽度方向不与茶几同心。
茶几桌面约为 `z=0.747` 米，地面约为 `z=0`；
`table.top_height_range: [0.747, 0.747]` 固定每次重置的桌高，避免与静态视觉错位。
机器人初始化沿用 GRAIL `reference_free` 的采样逻辑：每次重置沿桌子长边方向均匀随机
偏移 ±1 米，沿桌子朝空地的方向距桌子中心固定 1.5 米，并面向桌子中心。
`robot` 中不设置 `initial_position_xy`，以保留该随机化；设置此字段会覆盖随机 XY。
初始高度、身体/手指姿态仍读取原 pose 文件及高度偏移配置，不额外随机化。
`--grail-seed 42` 固定随机序列，每轮重置仍会重新采样。物体、搬运目标的采样及 policy/decoder 逻辑保持原样。
这些数值按本地 USDZ 测量，3DGS 表面本身存在少量重建误差。

在终端 1 的仿真命令中调整：

| 需求 | 参数 |
|---|---|
| 双渲染（默认） | `--render-backend both --isaac-headless`；Isaac 图像 5555，MuJoCo 图像 5559 |
| 仅 MuJoCo 相机 | 将上述两个参数替换为 `--render-backend mujoco --enable-offscreen`；图像 5555 |
| 仅 Isaac 相机 | 将 `both` 改为 `isaac`；图像 5555 |
| 隐藏 MuJoCo 交互窗口 | 追加 `--no-enable-onscreen` |
| 显示 Isaac 自带预览 | 去掉 `--isaac-headless` |

## 日志与退出

- 任务日志：`/tmp/grail-episodes.jsonl`。
- 每轮 25 秒仿真时间，或跌倒后重置；MuJoCo 窗口按 Backspace 可手动重置。
- 结束时分别在终端 1、2、3 按 **Ctrl+C**；相机窗口按 **Q** 关闭。

decoder 已通过数值对照，完整抓取、搬运、松手动作链尚未验收。
