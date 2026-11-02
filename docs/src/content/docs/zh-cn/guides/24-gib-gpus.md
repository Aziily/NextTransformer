---
title: 在 24 GiB GPU 上训练
description: 八张 24 GiB Ampere GPU 跑一次正式训练的成本、时间去向，以及有效的设置。
---

八张 24 GiB Ampere GPU（sm86，BF16、FP32 累加稠密峰值 71 TFLOPS）可以对除两个以外的全部已注册配置跑通全上下文前向和反向（见
[方法覆盖](#24-gib-ampere-上的方法覆盖)）。以下数字来自 `configs/baseline/dense_full.json` 的有界研究模式运行：单节点，
NVLink 两两相连（0-1、2-3、4-5、6-7），对与对之间走 PCIe，PyTorch 2.9.1、CUDA 12.8、NCCL 2.27.5。它们是估算，不是正式运行。

## 按正式更新形状测量

一次正式更新是 1,048,576 个 token：micro-batch 1 时每个 rank 累积 64 个 micro-batch，micro-batch 2 时累积 32 个。DDP
每次更新只同步一次，所以每次更新只有一个 micro-batch 的冒烟运行会高估通信：

| Micro-batch × 累积 | 每次更新 token | 每次更新 | Tokens/s |
|---|---:|---:|---:|
| 1 × 1 | 16,384 | 1.35 s | 12,137 |
| 1 × 8 | 131,072 | 2.97 s | 44,111 |
| 1 × 64（正式） | 1,048,576 | 16.12 s | 65,032 |
| 2 × 32（正式） | 1,048,576 | 14.54 s | 72,135 |

用 `launch/mapping_trials.py --smoke-accumulation 64`（或 `--micro-batch 2 --smoke-accumulation 32`）测量；除非用
`--rolling-every` 指定，它只写最后一个检查点。

## 一次正式运行的估算成本

| 部分 | Micro-batch 1 | Micro-batch 2，eval batch 8 |
|---|---:|---:|
| 46,875 次更新 | 210.0 h | 189.3 h |
| 189 次验证（每次 4,194 万 token） | 10.8 h | 8.6 h |
| 6 次 CORE-22（每次约 14.4 分钟） | 1.4 h | 1.4 h |
| 每 500 次更新一个检查点，共 94 个（9.2 GB，每个约 18 s） | 0.5 h | 0.5 h |
| **合计** | **222.8 h（9.3 天）** | **199.9 h（8.3 天）** |

按审计的每输入 token 4.6747 GFLOPs，micro-batch 2 每卡约 42 TFLOPS，是 71 TFLOPS 稠密峰值的
59%（micro-batch 1 为 54%）。

## 时间去向

一次 micro-batch 1 正式更新的 Nsight Systems 记录（被记录 rank 上 16.07 s）：

| 部分 | 时间 | 占比 |
|---|---:|---:|
| 计算 kernel | 14.51 s | 90% |
| NCCL kernel | 1.26 s，其中 1.03 s 未被重叠 | 8% |
| GPU 空闲 | 0.53 s | 3% |

- GEMM（CUTLASS 和 cuBLAS BF16）约 8.5 s；FlashAttention 前向和反向约 1.3 s。
- 未融合的逐元素 kernel 约 3.9 s：残差加法、标量乘、BF16 转换和拷贝。每个 micro-batch 约 3,000 个 kernel（每次更新
  195,000 个），但 GPU 只空闲 3%，所以 CPU 启动开销和 CUDA graph 不是瓶颈。融合这些 kernel 会有收益，但会改变舍入。
- 最后一个 micro-batch 的梯度 all-reduce 有 0.71 s 没有被反向隐藏；优化器步骤 0.41 s，主要是参数 all-gather（0.39 s），
  Muon 和 AdamW 计算约 0.08 s。
- GPU 功耗 325-345 W（上限 350 W），频率约 1.9 GHz，低于 2.1 GHz 的最高频率。

## 设置

| 设置 | 效果 | 说明 |
|---|---|---|
| `--micro-batch 2` | 训练吞吐 +11% | 需要 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；峰值显存 23.4 / 23.6 GiB，不加此设置第 3 次更新会显存不足 |
| `--eval-batch 8` | 验证 207 s → 165 s | 行和 token 不变；BPB 差 5e-7 |
| `NCCL_PROTO=Simple`、`NCCL_ALGO=allreduce:tree` | 无可测变化 | `NCCL_ALGO=Tree` 会失败：NCCL 没有 Tree all-gather |
| GPU 顺序、`--numa-binding node` | 无变化或更慢 | |

Micro-batch 和评测 batch 是[训练协议](../../protocol/training/)允许的每卡布局选择；它们改变归约顺序，不改变更新。请随运行记录。

## 24 GiB Ampere 上的方法覆盖

全上下文前向和反向（`nexttransformer.checks.validate --mode target --sequence-length 2048`）对所有已注册配置通过，以下除外：

| 方法 | 问题 | 处理 |
|---|---|---|
| kda、kda_no_ve、kda_hybrid、retnet、gdn、gdn_hybrid、gdn_hybrid_swa、deltanet_hybrid、qsa_hybrid | `No module named 'fla'` | 安装 `requirements/linear.txt`（fla-core 0.5.2）后九个全部通过 |
| dsa、moba、moba_hybrid、nsa、qsa、seer 使用 `--sparse-backend kernel` | FlexAttention 模板需要 114,688 字节共享内存，sm86 只有 101,376 | 使用默认的 reference 后端 |
| gla、deltanet | micro-batch 1 显存不足 | 没有省显存的 kernel 时无法在 24 GiB 上运行 |

该检查不包含优化器状态和 DDP bucket：`dense_full` 在检查中峰值 9.5 GiB，训练中 17.7 GiB。检查中峰值超过约 14 GiB 的方法
（elu_linear、elu_norm、elu_qknorm）在正式运行前需要先跑一次有界训练冒烟。
