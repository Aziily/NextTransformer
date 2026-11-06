---
title: 排行榜
description: 同一参考协议下完成的正式训练，以及各方法与 Dense 基线的差值。
---

[交互式排行榜](https://interactive-active-ai.github.io/NextTransformer/#leaderboard)支持回放训练过程、查看学习曲线，以及展开全部 CORE 任务。下表来自 `results/` 的运行记录，与 `python -m nexttransformer.export_run --summary` 使用相同的数据。

验证 BPB 取第 46,875 次更新的定期测量；CORE 显示为原分数乘以 100。Δ 表示与同一检查点的完整注意力 **Dense Baseline** 相比的差值。BPB 越低越好，CORE 越高越好。各行按 CORE 排序。竞技场参赛作品（自动研究）只按 CORE 排名，验证 BPB 只显示、不排名。

<!-- generated:leaderboard -->
| 模型 | 模型族 | 验证 BPB ↓ | Δ BPB | CORE ↑ | Δ CORE | 训练 GFLOPs/token |
|---|---|---:|---:|---:|---:|---:|
| **[Dense Baseline](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/dense_full.json)** | 基线 | 0.6672 | — | 33.10 | — | 4.675 |
| [AttnRes-Canon](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/attnres_full_canon_abcd.json) | 残差 | 0.6605 | -0.0067 | 35.99 | +2.90 | 4.695 |
| [Manifold-Constrained Hyper-Connections · mHC](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mhc.json) | 残差 | 0.6617 | -0.0055 | 35.33 | +2.23 | 4.674 |
| [Hyper-Connections · HC](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/hc.json) | 残差 | 0.6644 | -0.0028 | 35.09 | +2.00 | 4.698 |
| [Block Attention Residuals · AttnRes](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/attnres.json) | 残差 | 0.6630 | -0.0042 | 34.00 | +0.90 | 4.680 |
| [SSSL](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/sssl.json) | 其他 | 0.6671 | ±0.0000 | 33.78 | +0.69 | 4.775¹ |
| [Native Sparse Attention · NSA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/nsa.json) | 稀疏 | 0.6678 | +0.0006 | 33.78 | +0.69 | 4.639 |
| [Hybrid GDN + SWA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gdn_hybrid_swa.json) | 混合 | 0.6656 | -0.0015 | 33.66 | +0.57 | 4.647 |
| [Hybrid GDN + MHA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gdn_hybrid.json) | 混合 | 0.6647 | -0.0025 | 33.37 | +0.28 | 4.704 |
| [Hybrid MoBA + MHA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/moba_hybrid.json) | 混合 | 0.6702 | +0.0031 | 32.74 | -0.35 | 4.660 |
| [Grouped-Query Attention · GQA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gqa.json) | 其他 | 0.6741 | +0.0069 | 32.62 | -0.47 | 4.687 |
| [Multi-head Latent Attention · MLA + QK-norm](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mla_qknorm.json) | 其他 | 0.6759 | +0.0087 | 32.53 | -0.56 | 4.670 |
| [Fixed Recursive](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/fixed_recursive.json) | 循环 | 0.6784 | +0.0112 | 32.45 | -0.65 | 4.663 |
| [LoopFormer-Fixed2](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/loopformer.json) | 循环 | 0.6792 | +0.0121 | 32.03 | -1.06 | 4.776¹ |
| [Multi-Query Attention · MQA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mqa.json) | 其他 | 0.6769 | +0.0098 | 31.96 | -1.13 | 4.664 |
| [Standard Residual](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/residual_standard.json) | 残差 | 0.6740 | +0.0068 | 31.95 | -1.14 | 4.674 |
| [DeepSeek Sparse Attention · DSA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/dsa.json) | 稀疏 | 0.6679 | +0.0007 | 31.77 | -1.33 | 4.651 |
| [Hybrid KDA + MLA-NoPE](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/kda_hybrid.json) | 混合 | 0.6738 | +0.0066 | 31.54 | -1.55 | 4.692 |
| [Gated Linear Attention · GLA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gla.json) | 线性 | 0.6690 | +0.0018 | 31.11 | -1.99 | 4.701² |
| [Kimi Delta Attention · KDA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/kda.json) | 线性 | 0.6671 | ±0.0000 | 31.06 | -2.03 | 4.692 |
| [Mixture of Block Attention · MoBA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/moba.json) | 稀疏 | 0.6740 | +0.0068 | 31.04 | -2.05 | 4.674 |
| [SeerAttention-style](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/seer.json) | 稀疏 | 0.6686 | +0.0014 | 30.81 | -2.28 | 4.660 |
| [Gated DeltaNet · GDN](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gdn.json) | 线性 | 0.6700 | +0.0028 | 30.55 | -2.55 | 4.687 |
| [Mixture-of-Recursions · MoR](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mor.json) | 循环 | 0.6838 | +0.0166 | 30.37 | -2.73 | 4.958¹ |
| [Universal Transformer · ACT](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/ut_act.json) | 循环 | 0.6894 | +0.0222 | 30.07 | -3.03 | 4.660 |
| [Hybrid DeltaNet + MHA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/deltanet_hybrid.json) | 混合 | 0.6731 | +0.0059 | 29.90 | -3.19 | 4.675 |
| [Qwen Sparse Attention · QSA](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/qsa.json) | 稀疏 | 0.6708 | +0.0036 | 29.82 | -3.27 | 4.628 |
| [RetNet · Retention](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/retnet.json) | 线性 | 0.6825 | +0.0153 | 29.36 | -3.73 | 4.630 |
| [Circle Loop](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/circle.json) | 循环 | 0.6912 | +0.0240 | 28.97 | -4.12 | 4.977¹ |
| [DeltaNet](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/deltanet.json) | 线性 | 0.6787 | +0.0115 | 28.34 | -4.76 | 4.685² |
| [Gated Slot Attention · GSA + QK-norm](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/gsa_qknorm.json) | 线性 | 0.6852 | +0.0181 | 28.17 | -4.92 | 4.689 |
| [Ouro · LoopLM](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/ouro.json) | 循环 | 0.7075 | +0.0403 | 27.70 | -5.40 | 4.658 |
| [MLA-NoPE + QK-norm](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/mla_nope_qknorm.json) | 其他 | 0.6856 | +0.0184 | 27.65 | -5.44 | 4.686 |
| [Huginn · Recurrent-depth](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/huginn.json) | 循环 | 0.7036 | +0.0365 | 27.26 | -5.84 | 4.649 |
| [DeepLoop](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/deeploop.json) | 循环 | 0.7172 | +0.0500 | 25.45 | -7.64 | 4.661 |
| [ELU-Norm](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/elu_qknorm.json) | 线性 | 0.6998 | +0.0326 | 23.04 | -10.05 | 4.652 |
<!-- /generated:leaderboard -->

所有运行均为单种子，所用机器和 rank 数可能不同。排名展示观测结果，不代表差异具有统计显著性。算力仅在相同标记下可直接比较：

- 无标记：算子级账本。
- ¹：早期原生实现采用的矩阵 FLOPs 估算。
- ²：包括反向重放的算子级账本，与当次运行的冻结源码对应。

循环与稀疏方法报告训练器的累计审计，独立的运行目录审计仍在进行。两项混合架构结果已完成元数据审计，pilot 与续训验收尚待完成。各运行的具体情况见[运行说明](../runs/)。
