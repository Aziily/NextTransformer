---
title: 模型库
description: 按模型族浏览已注册的架构、论文和结果状态。
---

当前共有 37 张模型卡。各方法在同一骨干上校准至相同训练算力。在[观测站](https://interactive-active-ai.github.io/NextTransformer/#atlas)选择模型可查看结构图、曲线和模型卡。“已注册”表示已有配置与模型卡，尚无公开运行结果。

| 模型族 | 方法 | 论文 | 结果 |
|---|---|---|---|
| 基线 | Dense Baseline (`dense_full`) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 已发布 |
| 循环 | Universal Transformer · ACT (`ut_act`) | [Universal Transformers](https://arxiv.org/abs/1807.03819) | 已发布 |
| 循环 | Fixed Recursive (`fixed_recursive`) | Fixed Recursive · shared-core control baseline | 已发布 |
| 循环 | Huginn · Recurrent-depth (`huginn`) | [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) | 已发布 |
| 循环 | Ouro · LoopLM (`ouro`) | [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | 已发布 |
| 循环 | DeepLoop (`deeploop`) | [DeepLoop · v2](https://arxiv.org/abs/2607.13491v2) | 已发布 |
| 循环 | Mixture-of-Recursions · MoR (`mor`) | [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524) | 已发布 |
| 循环 | LoopFormer-Fixed2 (`loopformer`) | [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451) | 已发布 |
| 循环 | Circle Loop (`circle`) | Circle Loop · method specification | 已发布 |
| 稀疏 | DeepSeek Sparse Attention · DSA (`dsa`) | [DeepSeek-V3.2-Exp · DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/DeepSeek_V3_2.pdf) | 已发布 |
| 稀疏 | Qwen Sparse Attention · QSA (`qsa`) | [Qwen3.8-Next technical report · §2.1.2 QSA](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | 已发布 |
| 稀疏 | Native Sparse Attention · NSA (`nsa`) | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) | 已发布 |
| 稀疏 | SeerAttention-style (`seer`) | [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](https://arxiv.org/abs/2410.13276) | 已发布 |
| 稀疏 | Mixture of Block Attention · MoBA (`moba`) | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | 已发布 |
| 线性 | DeltaNet (`deltanet`) | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | 已发布 |
| 线性 | Gated Slot Attention · GSA + QK-norm (`gsa_qknorm`) | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146) | 已发布 |
| 线性 | Gated Linear Attention · GLA (`gla`) | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | 已发布 |
| 线性 | Kimi Delta Attention · KDA (`kda`) | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | 已发布 |
| 线性 | RetNet · Retention (`retnet`) | [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) | 已发布 |
| 线性 | ELU-Norm (`elu_qknorm`) | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) | 已发布 |
| 线性 | Gated DeltaNet · GDN (`gdn`) | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | 已发布 |
| 残差 | Standard Residual (`residual_standard`) | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | 已发布 |
| 残差 | Block Attention Residuals · AttnRes (`attnres`) | [Attention Residuals](https://arxiv.org/abs/2603.15031) | 已发布 |
| 残差 | Hyper-Connections · HC (`hc`) | [Hyper-Connections](https://arxiv.org/abs/2409.19606) | 已发布 |
| 残差 | Manifold-Constrained Hyper-Connections · mHC (`mhc`) | [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | 已发布 |
| 残差 | AttnRes-Canon (`attnres_full_canon_abcd`) | [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://arxiv.org/abs/2512.17351) | 已发布 |
| 混合 | Hybrid GDN + MHA (`gdn_hybrid`) | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | 已发布 |
| 混合 | Hybrid GDN + SWA (`gdn_hybrid_swa`) | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | 已发布 |
| 混合 | Hybrid DeltaNet + MHA (`deltanet_hybrid`) | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | 已发布 |
| 混合 | Hybrid KDA + MLA-NoPE (`kda_hybrid`) | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | 已发布 |
| 混合 | Hybrid GDN + QSA (`qsa_hybrid`) | [Qwen3.8-Next technical report · §2.1.2 QSA](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | 待发布 |
| 混合 | Hybrid MoBA + MHA (`moba_hybrid`) | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | 已发布 |
| 其他 | SSSL (`sssl`) | [nanochat · GPT d24 implementation](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) | 已发布 |
| 其他 | Multi-Query Attention · MQA (`mqa`) | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | 已发布 |
| 其他 | Grouped-Query Attention · GQA (`gqa`) | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | 已发布 |
| 其他 | Multi-head Latent Attention · MLA + QK-norm (`mla_qknorm`) | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | 已发布 |
| 其他 | MLA-NoPE + QK-norm (`mla_nope_qknorm`) | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | 已发布 |

训练配置见[模型配置](../configurations/)。新增架构请参见[编写插件](../../guides/plugins/)。
