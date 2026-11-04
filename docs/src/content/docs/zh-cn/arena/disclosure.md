---
title: 研究披露卡
description: 竞技场条目 results/arena/<entry>.json 的字段和填写要求。
---

每个竞技场条目提交一份 `results/arena/<entry>.json`。[从模板开始](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/docs/public/arena/card.template.json)，使用 `python -m nexttransformer.arena --check` 验证；CI 也会执行同一检查。

| 字段 | 填写内容 |
|---|---|
| `id`、`name` | 条目 ID（与文件名一致）及显示名称 |
| `methodId` | 注册方法，对应模型卡 `results/methods/<methodId>.json` |
| `runIds` | `results/runs/` 下该方法的运行记录，`status` 为 `complete` 时必填 |
| `arenaProtocol` | `arena-v1` |
| `status`、`statusNote` | `admitted`、`complete`、`legacy` 或 `retracted`；后两种须说明原因 |
| `submitted` | 在提案 issue 中预注册 fork 提交和配置的日期，格式 `YYYY-MM-DD` |
| `verification` | `self-reported`、`reproduced` 或 `maintainer` |
| `researcher.people` | 人类参与者的 `name`、可选的 `handle`，以及 `role` |
| `researcher.llms` | 每个 LLM 的 `name`、`version`（未知填 `null`）和 `role` |
| `researcher.framework` | 自动研究框架的 `name`（未使用填 `none`）、`version`、`url`、`note` |
| `researcher.involvement`、`involvementNote` | 人类参与程度 `L0`–`L3`，以及简要依据 |
| `research.start`、`research.end` | 研究起止日期，格式 `YYYY-MM-DD` |
| `research.baseCommit` | 研究开始时的仓库提交 |
| `research.candidatesTrained` | 完成过超出冒烟测试规模的训练的不同候选数 |
| `research.formalAttempts`、`formalNote` | 正式启动总数（包含失败尝试），以及每次启动的结果 |
| `research.gpuHours`、`hardware`、`computeNote` | 研究和正式训练的 GPU 小时数、使用的算力和计量方式；未计量填 `null` |
| `research.internet` | 是否使用互联网或检索文献 |
| `research.selectionSignal` | 筛选候选用到的信号：`train-loss`、`dev-bpb`、`validation-bpb`、`core` 或 `other` |
| `novelty.closestPrior`、`novelty.statement` | 最接近的先前方法，以及本方法与它的区别 |
| `artifacts` | HTTPS 链接：`checkpointUrl`、`logsUrl`（完成条目必填）、`traceUrl`、`historyUrl`、`codeUrl` |

`validation-bpb` 与 `core` 仅供历史条目如实披露。`admitted` 或 `complete` 条目不能使用这两项信号选择候选。
