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

### 已确定的计划范围

- 主预训练阶段采用 **4,096-token 序列**。
- 设置 **200B 和 400B tokens 两档训练预算**，按现有分词器统计实际用于训练的 tokens。
- 以 **ClimbMix 为主要数据集**，沿用现有的 **32,768 词表分词器**。
- 每档预算均包含 **Dense 和稀疏 MoE 模型**。
- 评测继续保留 **validation BPB 和 CORE-22**。

以上是下一阶段训练配方的计划范围，将制定独立版本的协议，锁定数据并明确计算成本。现有的 49.152B-token、2K benchmark 继续作为已发布结果的参考协议。

### 征集社区建议

[Discussion #22 中的 scaling 提案](https://github.com/Interactive-Active-AI/NextTransformer/discussions/22)列出了候选模型配置，重点征集以下建议：

- **数据组成：** 4K 预训练是否需要补充 ClimbMix 之外的数据，推荐哪些数据集，以及长文档的混合或采样方式。
- **Dense／MoE 配置：** 模型规模、深度和宽度、专家数量与路由，以及是否增加等算力对照。候选参数量和模型形状仍在讨论中。
- **CORE 之外的评测：** 初步考虑 **HELMET 的 RAG 任务**和 **RULER**，用于基座模型的 2K／4K 对照评测。最终 benchmark 选择以及后续是否扩展更长上下文仍待讨论。

社区反馈将用于确定优先开展的小规模消融实验，并在完整训练前敲定配方。

## 即将发布的结果与证据

- **Hybrid 结果：** GDN + QSA 和 MoBA + MHA。模型卡片已经在观测站中，标注为“Results coming soon”；每个结果都通过与其他贡献相同的[结果 pull request](../guides/submit-a-result/)发布。
- **审计：** looped 和 sparse 运行的独立运行目录审计；已发布 hybrid 的 pilot 和续训验收证据。

## 计划中

- **在 Hugging Face Hub 上发布检查点和分词器。**
- 把 Others 的实现（MQA、GQA、MLA）移植到当前代码。

有想法或想参与？欢迎加入 [scaling 讨论](https://github.com/Interactive-Active-AI/NextTransformer/discussions/22)，或另开一个 [discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions)。
