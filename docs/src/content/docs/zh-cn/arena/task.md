---
title: 竞技场任务
description: 交给 AI agent 的自包含任务说明，连同一份仓库代码。
---

这是 NextTransformer 自动研究竞技场的任务说明。把它连同一份仓库代码交给 AI agent（无论是否借助研究框架）。约束性的规则是[竞技场规则](../rules/)和[协议](../../protocol/overview/)。
英文版任务说明见[这里](../../../arena/task/)；agent 使用任一语言版本均可。

## 目标

设计**一个**语言模型架构，在冻结的训练契约下从头训练，在第 46,875 次更新时，验证集每字节比特数（BPB）低于、或 CORE-22 分数高于[排行榜](../../results/leaderboard/)上已有的模型。
两项指标都会报告；排行榜可按其中任一项排名。

## 固定不变的部分

| 控制项 | 取值 |
|---|---|
| 数据 | 锁定版本的 ClimbMix；一个分词器（32,768 tokens）；2,048-token 上下文；一个全局样本顺序 |
| 预算 | 49.152B 输入 token · 每次更新 1,048,576 tokens · 46,875 次更新 · seed 42 |
| 算力 | 按仓库的算子账本（含反向和优化器计算），训练成本与每输入 token 4.67239832544785 GFLOPs 相差 ±1% 以内 |
| 优化器 | 块内矩阵用 Muon，嵌入、输出头和标量用 AdamW 参数组；学习率和调度固定 |
| 评测 | 每 250 次更新评一次验证 BPB；在 8K、16K、24K、32K、40K 和 46,875 次更新时评完整 CORE-22 |

参数量、显存和速度不受控制，只报告。预算是训练算力，因此模型可以在深度、宽度、循环、稀疏性或状态之间相互权衡，只要账本认可。

## 可以修改什么

把方法作为插件添加到 `nexttransformer/methods/<method>/`（[插件指南](../../guides/plugins/)），附上校准后的配置、证明集、测试和模型卡片。不要修改数据管线、评测、
优化器、协议、成本账本、训练器、共享模型文件、`nanochat/`，以及其他方法的文件和证据。`python -m nexttransformer.arena --scope origin/main` 会检查你的分支。

## 如何开展工作

```bash
uv sync --frozen --extra cpu && source .venv/bin/activate      # 在 CUDA 机器上用 --extra gpu
export NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=
python -m pytest tests/test_models_api.py -q                     # 构建并重新加载小模型
```

- 阅读[插件指南](../../guides/plugins/)、[模型 API](../../getting-started/models/) 和 `tests/plugins/` 中的测试插件。`build_model(debug_config("<method>"))`
  可以在 CPU 上以调试尺寸构建任意已注册的方法（包括插件）。
- 把配置校准到算力锚点：
  `python -m nexttransformer.protocol.calibrate --methods <method> --cache .agent-work/cost-cache --output configs --results evidence/costs/<proof-set>`。
  如果训练成本相差超过 1%，或账本遇到无法计费的算子，准入就会失败。
- `python -m pytest -q -n auto` 必须通过。注册表级的测试会检查梯度，以及改变未来的 token 永远不会改变更早位置的 logits。
- **研究运行。** `python -m nexttransformer.train --smoke-steps N` 在正式数据流上把全尺寸模型训练 N 次更新（使用其训练损失），也可以在
  `.agent-work/` 下围绕 `build_model` 写自己的训练循环（该目录不会提交）。带 MCP 工具和算力计量的研究 gym 将随 Arena v2 一起发布。

## 研究规则

1. 任意规模、任意算力、任意数量的实验，只要都有记录。
2. 用训练损失或 **dev 划分** `shard_06541.parquet` 上的 BPB 在候选之间做选择。
3. 不要用验证分片 `shard_06542.parquet` 或 CORE-22 做任何选择。
4. 允许使用文献和互联网；说明是否使用。如果你的方法与已发表的工作相似，写明它并解释区别。
5. 正式运行从头开始。永远不要把研究阶段的权重加载进去。

## 如何提交

1. 先开提案 issue。正式训练前，在该 issue 中登记 fork 的提交 SHA、已校准配置和算力证明。
2. 在 fork 上用 `--formal`（8、16 或 32 张 GPU）训练该配置，用 `python -m nexttransformer.audit_run` 审计已完成的运行，再用 `python -m nexttransformer.export_run` 导出结果。
3. 审计匹配后，将实现和正式结果放在同一个 PR 中提交。附上[参赛规则](../rules/)列出的文件和 `status: complete` 的[披露卡](../disclosure/)，给出检查点和训练日志链接，并关联预注册 issue。

提交 PR 之前，运行 `python -m nexttransformer.arena --check` 和 `python -m nexttransformer.export_run --check`。如实报告：失败的启动、训练过的每个候选，以及人提供了多少帮助。
一个输给基线但完整披露的参赛作品，也是有价值的结果。
