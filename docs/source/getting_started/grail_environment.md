# GRAIL Conda 环境配置

[实时抓取搬运](../tutorials/grail_carry.md)和[离线采集、渲染及导出](../tutorials/grail_offline.md)共用 `gr00t-sim`。以下命令以本机的 Ubuntu 24.04 x86_64、Python 3.11、Isaac Sim 5.0.0.0、MuJoCo 3.17.0、NumPy 1.26.0 和 PyTorch 2.7.0（CUDA 12.6 wheel）为基准。C++ decoder 使用系统的 CUDA Toolkit 13.0、TensorRT 10.13.2.6 和 ONNX Runtime 1.16.3。两种流程都需要可用的 NVIDIA GPU 和驱动。实时一键启动另需桌面会话和 `gnome-terminal`。

## 拉取仓库和资产

克隆协作仓库后，在仓库根目录执行以下命令。Git LFS 会下载示例 checkpoint、decoder ONNX、机器人模型、3DGS 场景和默认物体资产。

```bash
git lfs install
git lfs pull
```

默认物体为 `pickup_table__soap_dispenser_0__001`。实时和离线入口直接从 `gear_sonic/data/grail/prepared_assets/pickup_table_slowturn_300/` 读取预转换结果。Isaac 仍引用同一物体的原始 USD 和贴图，因此分发时应同时保留原始资产和预转换目录。

## 创建 gr00t-sim

安装好 Conda 后执行。已有 `gr00t-sim` 的协作者可直接激活，再按下列 Python 依赖安装命令核对。Isaac 安装包及解压过程需要数十 GB 空间。

```bash
conda create -n gr00t-sim --override-channels -c conda-forge \
  python=3.11 pip 'cyclonedds=0.10.5'
conda activate gr00t-sim

python -m pip install 'isaacsim[all,extscache]==5.0.0.0' \
  --extra-index-url https://pypi.nvidia.com
CYCLONEDDS_HOME="$CONDA_PREFIX" python -m pip install \
  -r install_scripts/requirements_grail_sim.txt \
  -e external_dependencies/unitree_sdk2_python
python -m pip install --no-deps --force-reinstall 'opencv-python==4.11.0.86'
```

先后两次安装保留了本机的版本组合。Isaac 元数据要求 `websockets==12.0`，MuJoCo 3.17.0 要求 13 或以上，配置沿用本机的 17.4，`pip check` 会报告这条冲突。最后一步恢复桌面相机窗口使用的 OpenCV。以下教程从仓库根目录用 `python -m` 加载源码，使用 Isaac 提供的 NumPy 1.26.0。

阅读并接受 NVIDIA Omniverse EULA 后，在运行实时启动、离线渲染的终端执行以下命令，使新建的 Kit 子进程可以启动。 [NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## 编译 decoder

安装 CUDA Toolkit 13.0 以及对应的 NVIDIA TensorRT 10.13.2.6 C++ SDK。按[部署安装说明](installation_deploy.md#native-development-recommended)安装编译器、CMake、ONNX Runtime C++、ZMQ 和其他系统依赖。实时相机需要 `gnome-terminal`，离线视频校验和解码需要系统 FFmpeg。

在仓库根目录设置 SDK 的安装位置。示例假设 TensorRT 解压在用户目录、CUDA 和 ONNX Runtime 安装在系统默认位置，按实际位置修改。两份教程中的 ZMQ decoder 可使用默认关闭 ROS 2 的构建。

```bash
export TensorRT_ROOT="$HOME/TensorRT-10.13.2.6"
export CUDAToolkit_ROOT=/usr/local/cuda
export onnxruntime_ROOT=/opt/onnxruntime
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$CUDAToolkit_ROOT/lib64:$onnxruntime_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S gear_sonic_deploy -B gear_sonic_deploy/build -DCMAKE_BUILD_TYPE=Release
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref grail_offline_decoder -j4
```

如需执行离线渲染和数据集导出，继续运行 `bash install_scripts/install_grail_offline.sh`。脚本安装 PyAV、LeRobot 和导出依赖，并编译离线 decoder。新终端启动时需要重新激活 `gr00t-sim` 并加载上述运行环境变量。

## 检查环境

```bash
nvidia-smi
python -c 'import cv2, mujoco, numpy, pinocchio, torch; from unitree_sdk2py.core import channel; print(numpy.__version__, mujoco.__version__, torch.__version__, torch.cuda.is_available())'
python -m gear_sonic.scripts.run_sim_loop --help
python -m gear_sonic.scripts.grail_offline collect --help
ldd gear_sonic_deploy/target/release/g1_deploy_onnx_ref
ldd gear_sonic_deploy/target/release/grail_offline_decoder
```

确认 PyTorch 的 CUDA 检查为 `True`，且 `ldd` 输出没有 `not found`。首次渲染还会初始化 Kit 缓存。

## 批量准备其他物体

本节供资产维护者使用。协作者已有预转换文件时，可以直接运行两份教程。原始目录需包含 `object_usd/`、原始贴图和 `bps/`。转换器沿用本机 `grail` 环境中使用的 USD、NumPy 和 trimesh 版本，不需要加载 Isaac。

```bash
conda create -n grail --override-channels -c conda-forge python=3.11 pip
conda run -n grail python -m pip install \
  'usd-core==26.5' 'numpy==1.26.4' 'scipy==1.15.3' 'trimesh==4.5.1'
conda run -n grail --no-capture-output python -m gear_sonic.scripts.prepare_grail_assets \
  --dataset-dir gear_sonic/data/grail/1600obj --all
```

转换结果写入与 `1600obj/` 同级的 `prepared_assets/1600obj/`，按物体存放 MuJoCo 网格、贴图、BPS 和碰撞质量属性。再次执行可复用已有结果。只准备一个物体时，把 `--all` 换成 `--object-key <物体标识>`。使用 `--cache-dir` 可以指定其他输出目录。

本机的 `1600obj/` 共含 679 个物体，预转换结果约 8.1 GiB。完整 `1600obj/` 和 `prepared_assets/1600obj/` 作为配套资产单独分发。收到资产后把这两个目录放回上述位置，即可通过 `--dataset-dir gear_sonic/data/grail/1600obj` 或实时脚本的 `DATASET_DIR` 使用。保留 `object_usd/` 下的相对贴图路径。仓库默认物体的预转换目录已允许纳入 Git，OBJ 和 PNG 使用 Git LFS。
