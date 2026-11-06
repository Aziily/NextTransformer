---
title: Model zoo
description: Every method in NextTransformer, by family.
---

NextTransformer has 37 model cards in seven families. Every method runs in the same shell and is calibrated to
the same training compute. Open a method in the [observatory](https://interactive-active-ai.github.io/NextTransformer/#atlas) for its architecture diagram, curves and
model card; "registered" methods have a card and a configuration but no published run.

| Family | Method | Paper | Result |
|---|---|---|---|
| Baseline | Dense Baseline (`dense_full`) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | published |
| Looped | Universal Transformer · ACT (`ut_act`) | [Universal Transformers](https://arxiv.org/abs/1807.03819) | published |
| Looped | Fixed Recursive (`fixed_recursive`) | Fixed Recursive · shared-core control baseline | published |
| Looped | Huginn · Recurrent-depth (`huginn`) | [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) | published |
| Looped | Ouro · LoopLM (`ouro`) | [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | published |
| Looped | DeepLoop (`deeploop`) | [DeepLoop · v2](https://arxiv.org/abs/2607.13491v2) | published |
| Looped | Mixture-of-Recursions · MoR (`mor`) | [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524) | published |
| Looped | LoopFormer-Fixed2 (`loopformer`) | [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451) | published |
| Looped | Circle Loop (`circle`) | Circle Loop · method specification | published |
| Sparse | DeepSeek Sparse Attention · DSA (`dsa`) | [DeepSeek-V3.2-Exp · DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/DeepSeek_V3_2.pdf) | published |
| Sparse | Qwen Sparse Attention · QSA (`qsa`) | [Qwen3.8-Next technical report · §2.1.2 QSA](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | published |
| Sparse | Native Sparse Attention · NSA (`nsa`) | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) | published |
| Sparse | SeerAttention-style (`seer`) | [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](https://arxiv.org/abs/2410.13276) | published |
| Sparse | Mixture of Block Attention · MoBA (`moba`) | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | published |
| Linear | DeltaNet (`deltanet`) | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | published |
| Linear | Gated Slot Attention · GSA + QK-norm (`gsa_qknorm`) | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146) | published |
| Linear | Gated Linear Attention · GLA (`gla`) | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | published |
| Linear | Kimi Delta Attention · KDA (`kda`) | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | published |
| Linear | RetNet · Retention (`retnet`) | [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) | published |
| Linear | ELU-Norm (`elu_qknorm`) | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) | published |
| Linear | Gated DeltaNet · GDN (`gdn`) | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | published |
| Residual | Standard Residual (`residual_standard`) | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | published |
| Residual | Block Attention Residuals · AttnRes (`attnres`) | [Attention Residuals](https://arxiv.org/abs/2603.15031) | published |
| Residual | Hyper-Connections · HC (`hc`) | [Hyper-Connections](https://arxiv.org/abs/2409.19606) | published |
| Residual | Manifold-Constrained Hyper-Connections · mHC (`mhc`) | [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | published |
| Residual | AttnRes-Canon (`attnres_full_canon_abcd`) | [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://arxiv.org/abs/2512.17351) | published |
| Hybrid | Hybrid GDN + MHA (`gdn_hybrid`) | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | published |
| Hybrid | Hybrid GDN + SWA (`gdn_hybrid_swa`) | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | published |
| Hybrid | Hybrid DeltaNet + MHA (`deltanet_hybrid`) | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | published |
| Hybrid | Hybrid KDA + MLA-NoPE (`kda_hybrid`) | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | published |
| Hybrid | Hybrid GDN + QSA (`qsa_hybrid`) | [Qwen3.8-Next technical report · §2.1.2 QSA](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | coming soon |
| Hybrid | Hybrid MoBA + MHA (`moba_hybrid`) | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | published |
| Others | SSSL (`sssl`) | [nanochat · GPT d24 implementation](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) | published |
| Others | Multi-Query Attention · MQA (`mqa`) | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | published |
| Others | Grouped-Query Attention · GQA (`gqa`) | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | published |
| Others | Multi-head Latent Attention · MLA + QK-norm (`mla_qknorm`) | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | published |
| Others | MLA-NoPE + QK-norm (`mla_nope_qknorm`) | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | published |

See the [configuration reference](../configurations/) for training configurations. Add your own as a [plugin](../../guides/plugins/).
