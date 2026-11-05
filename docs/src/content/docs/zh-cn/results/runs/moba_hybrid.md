---
title: Hybrid MoBA + MHA
description: 22 层 MoBA + 3 层 MHA 混合架构的单段训练，以及与独立 MoBA 和 Dense 的对比。
---

Hybrid MoBA + MHA（`moba_hybrid`）在种子 42 下完成第 46,875 步（49.152B tokens）训练。终点周期验证 BPB 为 **0.670240**，独立终验 BPB 为 **0.667620**（训练集为 0.669544），CORE ×100 为 **32.7422**。25 层模型在前 22 层使用 MoBA（块大小 128、包含当前块在内的 256-token 预算），最后 3 层（第 23–25 层）使用完整因果 MHA；经校准后 FFN 宽度为 6016，总参数量为 **1,452,935,068**（含 13 个 1-based 奇数层上的交替值嵌入 VE）。

| 运行 | 层数 | FFN | 参数量 | 周期验证 BPB | 独立终验 BPB | CORE ×100 | 相对 Dense Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline（`dense_full`） | 23 | 6144 | 1,355,810,568 | 0.667172 | 0.664460 | 33.0952 | — |
| MoBA（`moba`） | 25 | 6144 | 1,462,765,468 | 0.674019 | 0.671396 | 31.0429 | +0.00685 / -2.05 |
| Hybrid MoBA + MHA（`moba_hybrid`） | 25 | 6016 | 1,452,935,068 | 0.670240 | 0.667620 | 32.7422 | +0.00307 / -0.35 |

与独立 MoBA 相比，将 25 层中的最后 3 层（占总层数的 12%）替换为完整因果 MHA 后，周期验证 BPB 降低 0.00378（弥合了独立 MoBA 与 Dense 基线之间 55.2% 的验证 BPB 差距），独立终验 BPB 降低 0.00378，CORE ×100 提高 1.70 分（弥合了 82.8% 的 CORE 差距，距 `dense_full` 仅差 0.35 分）。在全部 22 个 CORE 子任务中，该混合架构在 19 个任务上高于或持平独立 MoBA，提升最显著的包括 CommonsenseQA（+19.96 中心化分）、BoolQ（+11.03）、Winograd（+4.40）、BIG-bench CS Algorithms（+2.95）、WinoGrande（+2.84）、ARC Challenge（+1.71）、OpenBookQA（+1.07）和 SQuAD（+1.06），仅在 COPA（−10.00）和 PIQA（−1.52）等少数任务上较低。

正式运行在单段内连续完成，采用 8 个 rank、每卡 microbatch 8、梯度累积 8 次（父提交 `fab8c032`，含启动环境补丁 `08668a33`，科学指纹 `e1b5a94c`）。MoBA 层在训练中使用本地块注意力 `kernel` 后端（CORE 评测始终走参考实现路径），最后三层 MHA 使用禁用 FlashAttention 的确定性 SDPA。

累计训练计算量为 **4.659989 GFLOPs/token**，低于 SSSL 锚点 0.266%，每个记录步均与成本凭证一致，无预算异常。只读审计覆盖全部 189 次周期验证、六次完整 CORE-22 评测和 94 个已校验检查点。训练全程梯度范数平稳且为有限值：第 500 步之后的中位数为 0.33，最大值为 1.04。

8 个 rank 下 12 步目标全局批量的 smoke 与新进程续训一致性验收已通过（`weights_equal`、`losses_equal`、`all_rank_states_equal`）。独立的 512 步 pilot 与启动补丁源码差异未留存归档，最终验收仍待维护者复核。

[运行数据](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/moba_hybrid.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/moba_hybrid.json)
