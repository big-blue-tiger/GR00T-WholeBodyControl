# GRAIL Conda 环境配置

实时推理、离线采集、渲染和 USD 转换共用 `gr00t-sim`。从仓库根目录按顺序执行以下步骤。

## 系统依赖

- Ubuntu 24.04 x86_64、Conda、Git LFS、支持 RTX 渲染的 NVIDIA GPU 和驱动。
- CUDA Toolkit 13.0、TensorRT 10.13.2.6（CUDA 13.0 版）、ONNX Runtime C++ 1.16.3 和编译依赖。[系统依赖安装](installation_deploy.md#native-development-recommended)
- 实时一键启动需要 GNOME 桌面和 `gnome-terminal`。离线视频处理需要 FFmpeg。

## 拉取资产并创建环境

已有 `gr00t-sim` 时可直接复用。Isaac Sim 安装和解压需要预留数十 GB 磁盘空间。

```bash
git lfs install
git lfs pull

conda create -n gr00t-sim --override-channels -c conda-forge \
  python=3.11 pip 'cyclonedds=0.10.5'
```

## 运行环境

安装前和每次打开新的启动终端时执行。按实际安装位置修改 SDK 路径，阅读并接受 EULA 后再设置 `OMNI_KIT_ACCEPT_EULA`。[NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)

```bash
conda activate gr00t-sim
export TensorRT_ROOT="$HOME/TensorRT-10.13.2.6"
export CUDAToolkit_ROOT=/usr/local/cuda
export onnxruntime_ROOT=/opt/onnxruntime
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$CUDAToolkit_ROOT/lib64:$onnxruntime_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMNI_KIT_ACCEPT_EULA=YES
```

## 安装并编译

```bash
bash install_scripts/install_grail.sh
```

脚本读取 `install_scripts/requirements_grail.txt`，安装依赖并编译实时和离线 decoder。转换用的 OpenUSD 安装在环境内的专用目录，Isaac 渲染使用 Kit 自带的 USD 绑定。`pip check` 会报告已知的 `websockets` 版本冲突。

安装完成后，按[实时推理](../tutorials/grail_carry.md)或[离线采集](../tutorials/grail_offline.md)启动。

## 批量准备其他物体

仓库提供默认物体及其预转换文件，大文件由 Git LFS 下载。使用新物体时，把包含 `object_usd/`、贴图和 `bps/` 的数据集放到 `gear_sonic/data/grail/` 下，然后在 `gr00t-sim` 中转换。

```bash
python -m gear_sonic.scripts.prepare_grail_assets \
  --dataset-dir gear_sonic/data/grail/1600obj --all
```

| 参数 | 说明 |
|---|---|
| `--dataset-dir` | 原始数据集目录 |
| `--all` | 转换数据集内所有物体 |
| `--object-key` | 仅转换指定物体，与 `--all` 互斥 |
| `--cache-dir` | 覆盖输出目录。默认写入 `gear_sonic/data/grail/prepared_assets/<数据集目录名>/` |

分发额外物体时一起提供原始数据集和对应的预转换目录，并保留贴图的相对路径。
