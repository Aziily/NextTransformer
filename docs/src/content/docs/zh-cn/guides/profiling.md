---
title: 剖析训练
description: 从阶段到算子和内核，找出一次训练更新的时间花在哪里，并估算正式运行的耗时。
---

剖析器用参考训练器跑几次有界的研究更新，输出一份自包含的 HTML 报告和机器可读的 `summary.json`。它回答三个问题：一次更新的时间花在哪里，这期间每张 GPU 在做什么，以及在这台机器上完成一次正式运行需要多久。

剖析器只用于研究。它始终运行冒烟更新，拒绝 `--formal`，也不改变任何训练计算：阶段计时只在训练器已有的函数前后记录时钟和 CUDA 事件。

## 运行剖析器

在仓库根目录运行，`--site-config` 指向被忽略的本机配置：

```bash
python launch/profile_train.py \
  --site-config site.local.json \
  --config configs/baseline/dense_full.json \
  --output /path/to/runtime/runs/profile-dense-full \
  --world-size 8 \
  --micro-batch 1 \
  --warmup-steps 1 \
  --steady-steps 2 \
  --trace torch
```

默认每次更新采用所选微批次对应的正式形状：8 个 rank、微批次 1 时为 64 个微批次、1,048,576 tokens。传入 `--smoke-accumulation N` 可以跑更快、更小的更新；这时报告会提醒，每次更新只做一次的梯度通信和优化器，所占比例会高于正式训练。只有最后一次更新写检查点。

| 参数 | 作用 |
|---|---|
| `--warmup-steps` | 不计入测量的更新（编译、通信器初始化） |
| `--steady-steps` | 报告测量时间和吞吐的更新 |
| `--trace torch` | 在稳态更新之后，用 PyTorch profiler 在 `--trace-ranks`（默认 rank 0）上追踪 `--trace-steps` 次额外更新：阶段、带输入形状的算子和内核 |
| `--trace nsys` | 用 Nsight Systems 在所有 rank 上追踪同样的更新：每张 GPU 的阶段和内核（`--nsys` 是别名） |
| `--peak-tflops` | MFU 使用的每卡稠密 BF16 峰值，架构表未收录该 GPU 时使用 |
| `--smoke-accumulation` | 每次更新的微批次数（默认：正式更新） |
| `--validation-tokens` | 每次冒烟验证的 token 数（默认一个批次）。设为正式规模 41943040，正式运行估算才会包含验证时间 |

被追踪的更新排在稳态更新之后，追踪开销不会计入测得的步时。

## 阅读报告

用浏览器打开 `report.html`，不需要服务器或网络。页面跟随系统的浅色或深色主题（顶部按钮可切换），图表只用两种色相：蓝色表示花在模型 FLOPs 上的时间，橙色表示超出这部分的损失。每张图都有表格视图。要比较多次剖析，可以把它们渲染成一个站点：每次剖析一个页面，再加一个索引页，在同一时间刻度上对比它们的更新：

```bash
python -m launch.profile_report runs/profile-a/summary.json runs/profile-b/summary.json --output runs/profiles-site
```

剖析页面的各部分从上到下依次是：

- **What to look at first** 列出发现及其依据：更新形状与正式训练不同、通信没有被计算掩盖、GPU 空闲、数据卡顿、各 rank 在同一阶段耗时不一致、大量极短内核、部分 GPU 利用率偏低、显存接近占满。
- **Model FLOPs utilization** 用一次更新的审计 FLOPs（配置的成本证明，即训练器计费的 FLOPs）除以 GPU 的稠密 BF16 tensor 峰值。峰值由架构、SM 数和频率推算，可用 `--peak-tflops` 覆盖。报告给出整次更新以及前向、反向、优化器各自的 MFU，并分别按规格频率和实际运行频率计算。有 trace 时，还会给出矩阵和 attention kernel 的实际速率，以及一张“MFU 损失瀑布图”：矩阵 kernel 低于峰值、其他计算 kernel（逐元素、归一化、类型转换）、未掩盖的通信、未掩盖的内存拷贝、空闲。
- **Steady state** 给出更新时间中位数、吞吐、峰值显存和更新形状。
- **Memory** 给出稳态更新中每个 rank 的峰值已分配显存、峰值保留显存和剩余余量。
- **Formal run estimate** 只在剖析采用正式更新形状时出现：训练更新、验证事件（`--validation-tokens` 至少测了正式规模的八分之一时），以及滚动检查点。CORE-22 评测不在测量范围内。
- **Where an update spends its time** 把一次稳态更新拆成数据、前向、反向、梯度范数、主机等待，以及优化器的 reduce、compute、gather 三个阶段，在每个 rank 上用 CUDA 事件计时。展开某个阶段可以看各 rank 的时间。
- **Wall time of this profile** 区分启动、预热、稳态和被追踪的更新、评测和检查点。
- **GPU activity in traced updates** 显示被追踪 GPU 的时间中，只做计算、计算与通信重叠、只做通信、内存拷贝和空闲各占多少。
- **Communication** 按类型、数据类型和发起阶段列出每种集合通信：字节数、耗时、未被计算掩盖的部分，以及算法带宽和总线带宽（口径同 nccl-tests）。
- **Scheduling and synchronization** 统计每次更新和每个 micro-batch 的 kernel 数、主机花在发射 kernel 上的时间，以及阻塞主机的同步调用；GPU 空闲拆成主机来不及、发射延迟和等待依赖三类。
- **Timeline** 为被追踪的更新、其中间一个 micro-batch，以及从最后一次反向到优化器的尾部，画出阶段、计算、通信、内存四条泳道。
- **Model components** 按模型部件（attention、MLP、embedding、head，及每一层）拆分前向和反向的 GPU 时间；反向 kernel 归到创建它的前向模块。
- **Operators and kernels** 是一棵可折叠的树：阶段、内核类别、算子（来自 PyTorch trace，带输入形状）、内核。每个内核归属发起它的阶段，所以异步集合通信会出现在发起它的位置。**Matrix shapes** 列出每种矩阵乘和 attention 形状及其 kernel 达到的速率，**Non-matrix operators** 对“其他计算 kernel”背后的算子排序。
- **GPU × time heatmap** 显示整个剖析过程中每张 GPU 的采样利用率、显存、功耗或空闲比例，并用一条色带标出各阶段。每张 GPU 的均值和峰值只统计稳态更新。

## 输出文件

```text
summary.json       报告展示的全部数据，供脚本和 agent 读取
report.html        自包含报告
metadata.json      命令、环境、版本和 Git 提交
stages/            各 rank 的阶段时间线；stages/torch/ 存放 PyTorch trace（可用 Perfetto 打开）
telemetry.jsonl    按 UUID 记录的 GPU 采样遥测
profile.log        训练器和剖析器输出
run/               有界训练的产物和 metrics.jsonl
trace.nsys-rep     Nsight Systems 报告，使用 --trace nsys 时生成
trace.sqlite       它的 SQLite 导出，报告读取该文件
```

只在更新形状和追踪方式都相同的运行之间比较稳态数值。剖析目录、trace、检查点、数据集和本机配置不要放进 Git；每次运行用新的输出目录，除非传入 `--overwrite`。
