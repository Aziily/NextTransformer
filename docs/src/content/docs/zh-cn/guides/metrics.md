---
title: 训练指标
description: 训练输出文件、记录频率和可选的实验追踪设置。
---

审计与导出直接读取运行目录内的 JSON 文件，离线也可使用。

| 文件 | 内容 |
|---|---|
| `metrics.jsonl` | 每行一个事件，例如 `train`、`validation`、`core`、`final_bpb` 或 `budget_anomaly` |
| `provenance.json`、`resume_<step>.json` | 每段训练的配置、指纹、软件与批量布局 |
| `run_status.json`、`budget.json` | 运行阶段、最终状态与累计算力账本 |
| `val_<step>.json`、`core_<step>.json`、`core_final.json`、`bpb_final.json` | 计划评测点上的测量结果 |
| `checkpoint_<step>/`、`latest.json` | 每 5,000 次更新和训练结束时的检查点 |

`train` 事件记录优化器步数、输入 tokens、损失与辅助损失、梯度范数、学习率乘数、累计理论 FLOPs 及相对准入凭证的偏差。方法还可记录采样深度、路由统计或稀疏选择等诊断量。日志默认每 10 次更新写入，并记录首尾两次更新；用 `--log-every` 调整间隔。

## 可选的实验追踪

Weights & Biases 默认关闭（`--wandb-mode disabled`）。在 `offline` 或 `online` 模式下，由 rank 0 记录数值、实验配置和运行来源，不上传模型、数据、源码或控制台文本。`--wandb-metrics core` 记录主要指标，`full` 记录全部数值字段。指标名按事件分组，例如 `train/loss`、`validation/val_bpb`、`core/core_metric` 和 `final_bpb/val_bpb`。在线凭据来自 `WANDB_API_KEY` 或挂载的 netrc 文件。

## 本地监控训练

`launch/run_monitor.py` 在本机提供运行目录的实时视图。它在每次请求时重新读取上表中的文件，从不写入运行目录、也不获取运行锁，并且不导入 torch 或训练器，因此可以与训练同时运行，也可以在能读到同一批文件的其他机器上运行。

```bash
python launch/run_monitor.py /path/to/runtime/runs            # 打开 http://127.0.0.1:8765
python launch/run_monitor.py /path/to/runtime/runs --once site # 生成静态页面而不启动服务
```

每个参数可以是一个运行目录，也可以是包含多个运行目录的文件夹。总览页列出每个运行的状态、进度、最新损失、吞吐和预计剩余时间，并可勾选多个运行在同一页对比。运行页展示相对该运行自身验证与 CORE 计划的进度、训练曲线、与同一方法已发布曲线并列的评测结果、吞吐、显存、相对容差的算力账本，以及健康检查：非有限值、账本偏差、预算异常、损失或梯度范数尖峰、更新变慢，以及停止写入事件的运行。

训练器只为其峰值表中的 GPU 报告 MFU；其他 GPU 可用 `--peak-tflops` 传入每卡稠密 BF16 峰值。服务默认只监听 `127.0.0.1`。访问远程机器请使用 SSH 隧道（`ssh -L 8765:127.0.0.1:8765 HOST`），不要使用 `--host 0.0.0.0`：页面会显示运行路径和配置。剩余时间估计包含更新、验证和已测得的 CORE 事件，不包含检查点。
