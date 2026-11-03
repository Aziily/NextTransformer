---
title: 路线图
description: NextTransformer 接下来的计划。
---

## 2026 年 11 月 14 日前：agent 原生研究

- 全面支持 agent 自主研究的 **Arena v2**：一个带 MCP 工具的研究 gym，可以提交、校准、训练候选并在 dev 划分上打分，验证集和 CORE 保持封存。
- 每次尝试都有**计量的算力和研究轨迹**，研究成本自动披露。
- **自我改进的研究循环**：agent 不仅改进它找到的东西，也改进它的搜索方式。
- **按研究者排名的排行榜**：哪个 LLM 和框架、以多少研究成本、发现了什么。

## 2026 年 11 月：扩展规模

- 把参考训练配方扩展到 **400B tokens** 和**更大的模型**，保持同样的协议纪律：锁定的数据、对齐的算力和成本账本。

## 即将发布的结果

- **Hybrid：** QSA 和 MoBA 的 hybrid。
- **审计：** looped 和 sparse 运行的独立运行目录审计；已发布 hybrid 的 pilot 和续训验收证据。

它们的模型卡片已经在观测站中，标注为“Results coming soon”。每个结果都通过与其他贡献相同的[结果 pull request](../guides/submit-a-result/)发布。

## 计划中

- **在 Hugging Face Hub 上发布检查点和分词器。**
- 把 Others 的实现（MQA、GQA、MLA）移植到当前代码。

有想法或想参与？欢迎开一个 [discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions)。
