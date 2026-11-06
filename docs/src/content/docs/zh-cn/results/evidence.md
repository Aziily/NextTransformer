---
title: 算力凭证与运行审计
description: 配置准入依据的算力凭证，以及正式运行结束后的累计审计。
---

## 算力凭证（`evidence/costs/`）

每份凭证对应一个模型配置。它在完整模型尺寸下记录前向、反向、辅助损失、重计算和优化器纳入统计的算子，按各训练阶段的步数加权，并记录该方法所用源码的指纹。

| 凭证集 | 用途 |
|---|---|
| `plugin-interface-20261031` | 当前凭证集，内置方法的配置指向这里的对应凭证 |
| `arena-v1-stage3-20261103` | arena v1 参赛方法 AttnRes-Canon（`attnres_full_canon_abcd`）的凭证及其 SSSL 锚点；它的配置指向这里。`summary.json` 中还列有未注册的 Block AttnRes 变体 |
| `mhc-logsinkhorn-20261028` | mHC 改用对数域 Sinkhorn-Knopp 后的凭证集，计数与当前一致 |
| `hybrid-kernels-20261026-release` | 已发布混合架构运行所用冻结源码的凭证集 |
| `layout-20261024` | 软件包重组前的凭证集，模型哈希和计数与之后的版本一致 |

每个凭证集都含有 SSSL 锚点（`dense.json`）。`python -m nexttransformer.protocol.fingerprint` 检查已发布方法是否仍与凭证一致。慢速测试会使用完整模型尺寸，从头复算其中的代表性凭证。

## 运行审计（`evidence/audits/`）

在本地运行目录执行 `python -m nexttransformer.audit_run` 可生成累计审计。[GSA + QK-norm](../runs/gsa_qknorm/) 等冻结版本也可随结果发布贡献者提供的审计。检查范围包括算力账本、每个训练与验证事件、全部 CORE 事件、独立的最终 BPB、最终检查点，以及每段训练的来源。报告记录结论、测量值、指纹和数据布局，不包含主机名、机器路径或设备名称。

已发布的审计和凭证保留原件。方法变更后生成新凭证；正式运行沿用其准入时的凭证和源码指纹。
