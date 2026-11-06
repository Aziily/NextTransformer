---
title: AttnRes-Canon
description: Arena v1 参赛方法：Full AttnRes + Canon 层 A–D，在 32 个 rank 上分两段正式训练，以及与残差族的比较。
---

AttnRes-Canon（`attnres_full_canon_abcd`）即 Full AttnRes 加上 Canon 层 A–D。它是一个 [arena v1 参赛方法](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/attnres-canon.json)：由 LLM agent 用 NextT-Gym 在 dev 集上发现并选出，人工参与等级为 L1。它以种子 42 训练到第 46,875 步（49.152B tokens），CORE ×100 为 **35.9921**。agent 读过 `results/runs/` 中已发布的结果；选型只用 dev 集上的数值，从未使用验证分片或 CORE-22。

协议规定的验证集测量结果如下：

- 终点周期验证 BPB 为 **0.660520**；
- 独立终验 BPB 为 **0.657754**，训练集上为 0.659834。

模型有 23 层，FFN 宽度为 6144，共 **1,357,154,568 个参数**，由 23 个深度组上的 Full AttnRes 加 138 个 Canon 层组成。

下表中，Δ 为相对 Dense 的 CORE ×100 分差 / 周期验证 BPB 差：

| 运行 | FFN | CORE ×100 | 周期验证 BPB | 独立终验 BPB | 相对 Dense 的 Δ |
|---|---:|---:|---:|---:|---:|
| Dense（`dense_full`） | 6144 | 33.0952 | 0.667172 | 0.664460 | — |
| Standard Residual | 6144 | 31.9513 | 0.673959 | 0.671214 | −1.14 / +0.00679 |
| Block AttnRes（`attnres`） | 6144 | 33.9971 | 0.662963 | 0.660217 | +0.90 / −0.00421 |
| HC | 6144 | 35.0943 | 0.664400 | 0.661683 | +2.00 / −0.00277 |
| mHC | 6016 | 35.3278 | 0.661665 | 0.658908 | +2.23 / −0.00551 |
| AttnRes-Canon | 6144 | 35.9921 | 0.660520 | 0.657754 | +2.90 / −0.00665 |

最接近的内置方法是 Block AttnRes。与它相比，CORE 高 2.00 分；22 个 CORE 任务中有 18 个持平或更好：

- 提升最多：BoolQ（中心化后 +28.41 分）、SQuAD（+11.59）、COPA（+6.00）；
- 落后最多：Winograd（−5.86）、BIG-bench CS Algorithms（−5.83）、AGI Eval LSAT-AR（−4.89）。

这 2.00 分中有 1.29 分来自 BoolQ 一项；在其余 21 个任务上，平均高 0.74 分。与 mHC 相比，CORE 高 0.66 分；不计 BoolQ，在其余 21 个任务上平均高 0.47 分。

周期验证 BPB 比 Block AttnRes 低 0.00244，比 mHC 低 0.00115。选型时训练到第 2,000 步，NextT-Gym 在 dev 集上测得与 Block AttnRes 的差距为 −0.00380 BPB。以上均为单种子观测，不构成统计意义上的排名。

正式训练使用预注册的配置，训练源码为 fork 上的 commit `bf54f0d`，代码包 SHA-256 为 `c9d3f98c…`。训练在 32 个 rank 上进行（4 个节点，每个节点 8 张 GPU），每卡 microbatch 16，不做梯度累积，分两段完成：

| 纳入曲线的更新 | 恢复自 | 训练源码 |
|---|---|---|
| 1–39,000 | — | `bf54f0d` |
| 39,001–46,875 | 第 39,000 步检查点 | `bf54f0d` |

第一段按计划在集群 37 小时的作业时限处停止，此时已记录到第 39,500 步。第 39,500 步的检查点不完整，所以第二段从第 39,000 步恢复，并重跑了第 39,001–39,500 步。训练器把被重放的事件另行归档，曲线采用第二段的记录。两段的模型、协议、数据、tokenizer、配置、成本凭证和源码哈希完全相同，都以正式模式训练。两次启动都没有失败，披露卡记为两次正式启动。

累计训练计算量为 **4.695110 GFLOPs/token**，比 SSSL 锚点高 0.486%。每个记录步都与成本凭证一致，没有预算异常。只读审计覆盖全部 189 次验证、六次完整 CORE-22 评测和最终检查点。

训练耗时 143,842 秒，即每步 3.07 秒；评测耗时 9,794 秒。梯度范数始终为有限值，第 500 步之后的中位数为 0.48，最大值为 2.49。

配置、fork commit 和成本凭证预注册在 2026-11-03 06:36 UTC 开出的 [PR #10](https://github.com/Interactive-Active-AI/NextTransformer/pull/10) 中。这个 PR 四分钟后应操作者要求关闭，正式训练于 11:00 UTC 从 fork 启动，`bf54f0d` 至今仍是该 PR 的 head。arena 从 2026-11-04 起改用 proposal issue 做预注册。[Proposal issue #30](https://github.com/Interactive-Active-AI/NextTransformer/issues/30) 于正式训练之后的 2026-11-06 提交，记录了同一个 commit、配置和成本凭证。

启动前，先在 dev 集上做了一次有界的四节点冒烟检查。检查共 32 个 rank、80 步，在第 40 步中断后恢复，结果如下：

- 第 1 步的 loss 与 8 卡运行完全一致；
- 恢复后的运行与不中断的运行相比，第 41–80 步平均 \|Δloss\| 为 0.030，在舍入误差范围内一致，但不是逐位一致。

这次检查没有使用验证分片，也没有使用 CORE。

Canon 层用的是没有融合 kernel 的 eager PyTorch depthwise 卷积。因此每步耗时反映的是这一实现，而不是方法本身。

[运行记录](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/runs/attnres_full_canon_abcd.json) · [累计审计](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/evidence/audits/attnres_full_canon_abcd.json) · [披露卡](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/results/arena/attnres-canon.json) · [最终检查点和训练日志](https://huggingface.co/GY2233/nextt-arena-attnres-full-canon-abcd)
