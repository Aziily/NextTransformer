---
title: Hybrid DeltaNet + MHA
description: 22 层 DeltaNet + 2 层 MHA 全局混合架构的单段训练，以及与独立 DeltaNet 和 Dense 的对比。
---

Hybrid DeltaNet + MHA（`deltanet_hybrid`）在种子 42 下完成第 46,875 步（49.152B tokens）训练。终点周期验证 BPB 为 **0.673079**，独立终验 BPB 为 **0.670325**（训练集为 0.672545），CORE ×100 为 **29.9012**。24 层模型在 1-based 第 2 层与第 13 层放置完整因果 MHA，其余 22 层使用 DeltaNet；经校准后 FFN 宽度为 6400，总参数量为 **1,403,810,314**（含 12 个 1-based 偶数层上的交替值嵌入 VE）。

| 运行 | 层数 | FFN | 参数量 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | 相对 Dense Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline（`dense_full`） | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| DeltaNet（`deltanet`） | 25 | 6144 | 1,463,690,268 | 0.678675 | 0.675966 | 28.3357 | +0.01150 / -4.76 |
| Hybrid DeltaNet + MHA（`deltanet_hybrid`） | 24 | 6400 | 1,403,810,314 | 0.673079 | 0.670325 | 29.9012 | +0.00591 / -3.19 |

与独立 DeltaNet 相比，仅在第 2 层和第 13 层插入两层完整因果 MHA（占 24 层的 8.3%），周期验证 BPB 即降低 0.00560（弥合了独立 DeltaNet 与 Dense 基线之间 48.7% 的验证 BPB 差距），独立终验 BPB 降低 0.00564，CORE ×100 提高 1.57 分（弥合了 32.9% 的 CORE 差距）。在全部 22 个 CORE 子任务中，该混合架构有 16 个任务高于独立 DeltaNet，提升最显著的包括 COPA（+8.00 中心化分）、CommonsenseQA（+6.96）、BIG-bench Operators（+6.19）、BoolQ（+5.79）、BIG-bench Dyck Languages（+4.60）、CoQA（+3.56）和 BIG-bench Repeat Copy Logic（+3.12），而在 Winograd（−6.59）、AGIEval LSAT-AR（−2.17）和 WinoGrande（−2.05）上较低。

正式运行在单段内连续完成，采用 8 个 rank、每卡 microbatch 8、梯度累积 8 次（父提交 `fab8c032`，含启动环境补丁 `08668a33`，科学指纹 `e1b5a94c`）。DeltaNet 层使用 FLA `chunk64` 后端（`chunk_delta_rule`），两层 MHA 使用禁用 FlashAttention 的确定性 SDPA。

累计训练计算量为 **4.675326 GFLOPs/token**，高于 SSSL 锚点 0.063%，每个记录步均与成本凭证一致，无预算异常。只读审计覆盖全部 189 次周期验证、六次完整 CORE-22 评测和 94 个已校验检查点。训练全程梯度范数平稳且为有限值：第 500 步之后的中位数为 0.25，最大值为 0.94。

8 个 rank 下 12 步目标全局批量的 smoke 与新进程续训一致性验收已通过（`weights_equal`、`losses_equal`、`all_rank_states_equal`）。独立的 512 步 pilot 与启动补丁源码差异未留存归档，最终验收仍待维护者复核。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/deltanet_hybrid.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/deltanet_hybrid.json)
