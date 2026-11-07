---
title: 路线图
description: NextTransformer 接下来的计划。
---

## 2026 年 11 月 14 日前：Arena v2，agent 原生研究

1. **用 Harbor 格式组织 auto-research 任务。**
   - 仓库本身作为 gym 的 sidecar 服务，向模型提供接口。
   - 模型在一个封闭的容器内开发。
   - agent 可以访问仓库的全部历史记录，也可以上网搜索，找出最好的架构。
2. **研究与正式评测分开。**
   - 研究使用 ClimbMix 训练数据和 dev 分片 `shard_06541.parquet`。
   - 正式验证集 `shard_06542.parquet` 和 CORE-22 不参与候选选择，仅用于正式运行报告。
3. **最终提交只是一份文件。** 之后由 gym 完成最终训练。

4. **Gym 接口。**
   - **接入方式**：容器内的 `/run/gym-auth/credential.json` 提供 endpoint 和 token。每次调用是一个 JSON `POST /v1/call`，带操作名和参数；Python 的 `GymClient` 做了封装，不导入任何训练代码。
   - **信息**：`describe_space`、`list_components`、`read_reference`、`get_resources`，查看搜索空间、可用组件、只读参考代码和资源上限。
   - **预检测**：`validate_candidate`、`calibrate_candidate`、`debug_candidate`、`inspect_cost`，在消耗研究算力前检查候选能否加载、FLOPs/token 和参数量是否合规。
   - **固定实验**：`run_experiment` 在固定配置下训练候选，可以从检查点续训；用 `get_job`、`get_result`、`cancel_job` 跟踪。
   - **预算**：`quote_experiment` 在运行前报价，`get_budget` 查看剩余额度。每次实验都从固定的研究算力额度中扣除，用完即停。
   - **提交**：`finalize` 把一个候选冻结为最终文件。

### Gym 实验计划

- **模型与 harness：** Opus、GLM、DeepSeek，统一使用 Claude Code，通过 Gym MCP 独立研究。开跑前固定模型和客户端版本。
- **初始设置：** 使用相同代码与参考资料快照、ClimbMix 数据顺序、分词器、2,048-token 上下文、seed 42、每步 1,048,576 tokens 和原优化器／学习率协议。研究只使用训练 loss 和 dev；正式验证集与 CORE-22 保留给正式评测。
- **研究方式：** 初始架构、对照方法、候选数量、每次实验长度、搜索和淘汰策略均由模型自行决定，不预设训练步骤或搜索轮数。
- **模型调用：** 不设费用上限，记录 token 和实际费用。
- **研究算力上限：** 每个模型的一次研究会话统一为 **1.96×10²⁰ FLOPs**，训练、评估、检查及失败尝试共同扣除，不按候选分别配额。三种模型各跑一次，研究总上限为 **5.88×10²⁰ FLOPs**；最终正式训练另计。
- **预算依据：** Arena v1 Stage 2 共 17 次尝试（含取消和重试）；以 20 次锚点 2,000 步训练的等效计算量估算总额度：`20 × 2,000 × 1,048,576 × 4.67239832544785e9 ≈ 1.96×10²⁰ FLOPs`。这只是预算换算，不要求模型跑 20 次或每次跑 2,000 步，也不是 v1 的实测 FLOPs 账本。
- **排期参考：** v1 搜索与移植核对共 `549.7 + 42.1 = 591.8` GPU 时，加 20% 余量，**每个模型约 710 GPU 时，同类 8 卡环境约 89 小时**。实际时间取决于模型选择的实验及吞吐，GPU 时不是额外的硬上限。
- **目标：** 比较各模型在相同研究算力内自主找到的结构及其收益，最终提交按同一正式训练协议验证。

### Arena 结果展示

先做一个[前端页面示例](../../../arena-example.html)：上方展示模型、harness、研究预算和状态；中间展示候选 dev 结果及相对基线的改善；下方展开结构改动、正式结果和证据说明。
示例使用 Arena v1 历史研究数据，Opus、GLM、DeepSeek 标为待运行，正式成绩留空；后续再接真实运行记录。

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

- **Hybrid 结果：** GDN + QSA。模型卡片已经在观测站中，标注为“Results coming soon”；每个结果都通过与其他贡献相同的[结果 pull request](../guides/submit-a-result/)发布。
- **审计：** looped 和 sparse 运行的独立运行目录审计；已发布 hybrid 的 pilot 和续训验收证据。

## 计划中

- **在 Hugging Face Hub 上发布检查点和分词器。**
- 把 Others 的实现（MQA、GQA、MLA）移植到当前代码。

有想法或想参与？欢迎加入 [scaling 讨论](https://github.com/Interactive-Active-AI/NextTransformer/discussions/22)，或另开一个 [discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions)。
