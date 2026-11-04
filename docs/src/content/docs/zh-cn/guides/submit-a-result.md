---
title: 提交结果
description: 审计完成的正式运行、导出记录，并提交 pull request。
---

一个结果是已准入配置的一次完成的**正式**运行：46,875 次更新、全部 189 次验证事件、全部 6 次 CORE-22 事件，每个分段都用 `--formal` 训练。
贡献新方法时，将下列结果材料与方法实现放在同一个 PR 中提交。

## 1. 审计运行目录

```bash
python -m nexttransformer.audit_run runs/<method> --method <method>
```

审计会重新读取训练器写出的所有内容，并对照协议检查：账本（更新次数、token 数、计费算力）、训练事件、189 次验证事件及其文件、6
次完整 CORE 事件及其中心化和汇总、独立最终 BPB、最终检查点，以及每个分段的来源（指纹一致、正式模式、登记过的批量布局）。它会写出
`evidence/audits/<method>.json`；只要有任何缺失就以非零状态退出，文件中会列出问题。审计记录数据布局，但永远不记录主机名、路径或设备。

## 2. 导出记录

```bash
python -m nexttransformer.export_run runs/<method> --method <method> --name "<display name>" \
    --audit evidence/audits/<method>.json
python -m nexttransformer.export_run --check
```

`export_run` 写出 `results/runs/<method>.json`。它会拒绝：验证或 CORE 时间表不完整、CORE 各任务无法复现汇总值、累计审计不匹配、存在研究模式的分段、分词器不同或协议不同。
运行需要来源说明时（例如由冻结的源码版本训练），加上 `--note "..."`。

## 3. 撰写运行说明

添加 `docs/src/content/docs/results/runs/<method>.md`：运行了什么、有无续训、失败的尝试，以及读者应该知道的局限。保持事实性、简短。
如果方法还没有模型卡片，添加 `results/methods/<method>.json`；如果是已登记的方法、其卡片带有 `"status": "pending"`，删除该字段。

## 4. 刷新生成的表格

```bash
python docs/scripts/generate.py
```

它会更新文档中的排行榜和 README 中的快照表格；如果模型卡片有变化，也会更新模型库。这些内容过期时 CI 会失败。

## 5. 提交 pull request

提交这些文件并开 pull request。对只含结果的 pull request，CI 只运行数据检查和站点构建，通常一到两分钟。维护者审阅审计和说明；合并之后网站会自动更新。

永远不要修改已发布的记录或审计。同一方法的第二次运行使用 `--file <name>` 和它自己的审计。
