# GRAIL SONIC decoder

此目录供 MuJoCo GRAIL 演示的独立 C++ 控制进程使用：

- `model_decoder_grail.onnx`：GRAIL `sonic_manipulation_base/last.pt` 中的 `g1_dyn`。
- `policy_model_decoder_grail.trt`：本机 TensorRT 10.13.2.6 的配套 FP32 缓存。

输入为 `obs_dict[1,994]`，包含量化后 64 维 token 和 930 维本体历史；
输出为 `action[1,29]`。FSQ 在独立顶层策略执行。

按 [GRAIL 启动教程](../../../docs/source/tutorials/grail_carry.md) 依次启动仿真、
`run_grail_policy` 和 C++ decoder。待 C++ 就绪后按 `]`，再按 Enter 开启 ZMQ streaming。

已核对本目录文件与导出产物哈希一致，未覆盖文件：

```text
5dad733040f98c212d8a4d34982f0f4dca6ab08de4a128b2d265c6c1150cbab3  model_decoder_grail.onnx
8401b162e449ba971d88c20e3af1015a8be5744d9021db9c9015947bfa0e56f4  policy_model_decoder_grail.trt
```

原 `../release/` 模型与缓存保留。ONNX/TRT 权重由 Git 忽略，需要单独保存。
已从本目录直接通过真实 C++ PolicyEngine 复测 296 组输入：
相对 GRAIL 的最大绝对误差为 `5.170703e-6`，CUDA graph 与普通推理完全一致。
