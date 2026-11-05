---
title: 运行说明
description: 已发布结果的运行条件、审计口径和比较边界。
sidebar:
  label: 运行概览
  order: 0
---

每个公开运行在 `results/runs/` 有一条记录，完成独立审计的运行另有 `evidence/audits/` 下的报告。这里补充影响结果解读的背景。

## Dense 基线（`dense_full`）

23 层完整因果注意力，FFN 宽度 6144，参数量 1.356B。训练只有一段，累计计算量比 SSSL 锚点高 **0.048%**，每步与凭证一致。独立最终 BPB 在验证集为 0.6645、训练集为 0.6666。维护者决定跳过预登记的 pilot，唯一一次正式运行从随机初始化开始。该运行已通过基线验收，排行榜中的 Δ 均以它为参照。

## 残差连接（`residual_standard`、`attnres`）

两个模型使用相同的 Dense 骨干，各自独立训练一段。Standard Residual 的累计算力偏差为 **+0.041%**，AttnRes 为 **+0.160%**，原运行目录均已通过只读审计。分析 AttnRes 的作用时，应与 Standard Residual 配对比较。[mHC](./mhc/) 分四段完成训练，最终 CORE ×100 为 35.3278，续训记录保留了对数域 Sinkhorn 的切换。[HC](./hc/) 分四段完成训练，源码未变，最终 CORE ×100 为 35.0943，CORE 排名第二；它的运行说明比较了整个残差族。

## 线性注意力

- [GSA + QK-norm](./gsa_qknorm/) 使用 64 个槽位和逐头 QK 归一化，最终 CORE ×100 为 28.1720。

- **KDA** 采用重新启动后的第二个版本。首次尝试在第 2,340 步中止，不计入本次运行。三段训练的边界为 0、21,681、32,000、46,875，来源一致，累计算力偏差 **+0.429%**。
- **ELU-Norm**（`elu_qknorm`）在 ELU+1 特征映射前归一化 q/k。16 个 rank 分两段完成，累计偏差 **−0.431%**。
- **GDN** 使用 32 个 rank 一段完成，累计偏差 **+0.317%**。
- **RetNet** 在第 25,736 步恢复一次，累计偏差 **−0.906%**。

这些续训段的数值在舍入精度内一致；逐位一致性仅适用于经过对应验证的路径。

## DeltaNet 和 GLA：冻结的重放核算

两者都完成完整协议。独立最终 BPB 分别为 0.6760 和 0.6663，CORE 分别为 28.34 和 31.11。冻结源码各文件的公开哈希见 `evidence/audits/deltanet-gla-source-manifest.json`。算子级账本包含反向重放，理论锚点为每 token 4.71661506444785 GFLOPs。算力带 ² 标记，限同口径比较；BPB 和 CORE 按标准方式排名。

## 其他注意力结构

MQA、GQA、MLA-QK 和 MLA-NoPE-QK 均来自尚未移植到当前主线的冻结源码。累计审计已复算各任务 CORE、中心化及汇总值。MLA 运行还记录了确定性模式。MQA 在第 5,000 和 35,000 步续训，GQA 在第 5,000 步续训，两个 MLA 变体均在单段内完成。

## 混合结构（DeltaNet–MHA、GDN–MHA、GDN–SWA、KDA–MLA-NoPE、MoBA–MHA）

五者使用冻结源码，[DeltaNet–MHA](./deltanet_hybrid/)、GDN–SWA、[KDA–MLA-NoPE](./kda_hybrid/) 和 [MoBA–MHA](./moba_hybrid/) 另有一项启动环境补丁未留存源码差异。只读审计通过，累计偏差分别为 **+0.063%**（DeltaNet–MHA，单段完成，最终 CORE ×100 为 29.9012）、**+0.672%**（GDN–MHA）、**−0.542%**（GDN–SWA）、**+0.426%**（KDA–MLA-NoPE，单段去除值嵌入训练，最终 CORE ×100 为 31.5418）和 **−0.266%**（MoBA–MHA，单段完成，最终 CORE ×100 为 32.7422）。DeltaNet–MHA、KDA–MLA-NoPE 与 MoBA–MHA 提供了 8 个 rank 下 12 步新进程续训一致性验收记录；因缺少独立的 512 步 pilot 记录，最终验收仍待完成。在本基准的 2,048-token 上下文中，SWA 窗口覆盖所有因果位置，所以 GDN–SWA 在这里实际是 1:1 的 GDN 与完整注意力混合。

## 循环与稀疏结构

UT-ACT、Fixed Recursive、Huginn、Ouro、DeepLoop、DSA、QSA、NSA、Seer 和 MoBA 均记录了完整训练、全部计划评测和训练器的累计审计。在对原运行目录完成独立只读审计前，记录中的 `complete` 保持为 `false`。稀疏方法训练使用 kernel 后端，CORE 评测始终走参考实现。

## 历史原生模型

SSSL、MoR、LoopFormer 和 Circle Loop 是项目首批运行，采用当时的原生实现。算力依据维护者核验的最终日志，使用矩阵 FLOPs 估算（¹）。若要纳入严格的等算力比较，仍需独立的完整计算成本凭证。

## 待发布的方法

观测站中待发布结果的模型为 GDN–QSA。
