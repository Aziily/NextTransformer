---
title: 方法来源与实现
description: 对照原论文及锁定的参考代码，说明各方法在统一骨干中保留的机制和改动。
---

这些实现研究的是同一套 nanochat 骨干内的架构差异。论文名称用来标识机制；论文中的完整模型、数据集和训练配方可能不同。下面列出对照的版本，以及会影响比较的实现选择。

## 锁定的参考版本

| 来源 | 锁定版本及用途 |
|---|---|
| nanochat | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`，文件清单见 `configs/native/upstream_sources.json` |
| [DeepLoop](https://github.com/lszshu/DeepLoop/tree/70c170866019ed956ef6a87af7752c149ff6a5e5) | `70c170866019ed956ef6a87af7752c149ff6a5e5` |
| [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | `87e509a2e5a100d221c97df52c6e8be7835f0057` |
| [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/69885871a64393807d988b27b1b5e380e8f28526) | `69885871a64393807d988b27b1b5e380e8f28526` |
| [SeerAttention](https://github.com/microsoft/SeerAttention/tree/aba03e3f2caefd0ccd21e576670aa830b748c84e) | `aba03e3f2caefd0ccd21e576670aa830b748c84e` |
| [MoBA](https://github.com/MoonshotAI/MoBA/tree/b5d58363311d3ca946f1ec444182727c15e338b5) | `b5d58363311d3ca946f1ec444182727c15e338b5` |
| [FLA](https://github.com/fla-org/flash-linear-attention/tree/c885377e65a36f612bd10f57b7d1daa55224c31b) | `c885377e65a36f612bd10f57b7d1daa55224c31b`，用于核对层实现和朴素递推 |
| [FLA GatedDeltaNet layer](https://pypi.org/project/flash-linear-attention/0.5.2/) | 版本 0.5.2 |
| [Hyper-Connections](https://arxiv.org/html/2409.19606v3) | arXiv v3，式 10–14、算法 2–3 |
| [mHC](https://arxiv.org/html/2512.24880v2) | arXiv v2，式 7–9、附录表 5（n = 4，门尺度 0.01，20 次迭代） |
| [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | arXiv v1，第 3 节、图 2； [官方仓库](https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6) |
| [lucidrains/hyper-connections](https://github.com/lucidrains/hyper-connections/blob/e89e30357d1f79945b0d3558e6721451ff68789a/hyper_connections/manifold_constrained_hyper_connections.py) | 仅用于核对 mHC 的静态偏置初始化 |

数学参考实现独立编写。CUDA 线性注意力训练调用 MIT 许可的 `fla-core==0.5.2`，版本及哈希锁定在 `requirements/linear.txt`；仓库不包含上游 CUDA 或 Triton 内核源码。

## 统一骨干

`nexttransformer/models/lm.py` 在锁定的 GPT 骨干上扩展。历史 `dense`（SSSL）沿用原始前向，校验内容包括 logits 和所有梯度。除下文列明的改动外，各方法保留不共享权重的输入/输出嵌入、值嵌入（VE）、smear、backout、ReLU² MLP、RMSNorm，以及可学习的残差和 x0 系数。VE 绑定到物理块，随块权重共享。`dense_full` 每层使用窗口 `(2048, 0)` 的参考实现，对照 `window_pattern=L` 的原生前向校验。

共享核心按展开后的位置分配 SSSL 窗口，末层固定为长窗口。ACT、Huginn 和 Ouro 用前置层作 backout，Fixed Recursive 和 DeepLoop 取展开后中点。线性方法使用各自的特征与位置规则，稀疏方法仅按自己的规则选择位置。

## 循环深度

| 方法 | 来源与保留的机制 | 在本基准中的实现 |
|---|---|---|
| `ut_act` | [Universal Transformer v3](https://arxiv.org/html/1807.03819v3)：位置、累计停止概率、剩余概率和计算惩罚 | 每轮核心计算前加入 token/深度坐标并计算门值，轮后混合输出。停止后的状态冻结，到最大循环数时消耗剩余概率。参考实现仍计算并计费所有候选，并不跳过已停止 token 的计算。 |
| `fixed_recursive` | 权重共享对照 | 前置层一次，共享核心两次，后置层一次，不增加路由门或条件输入。 |
| `huginn` | [Recurrent depth v2](https://arxiv.org/html/2502.05171v2)：采样深度、状态、输入注入和截断反传 | 每步按路由种子与步数采样一次截断的泊松–对数正态深度，最多循环 4 次，仅最后 2 次反传。初始状态噪声按种子和全局行确定。拼接状态和前置表示并投影，使用夹心式 RMSNorm。 |
| `ouro` | [Ouro v1](https://arxiv.org/html/2510.25741v1)：多深度目标、熵奖励、门控专训 | 每轮运行后置层和输出头，联合目标为退出概率加权 CE 减 0.1 倍熵。最后 1,024 步仅训练停止门，使用分离梯度的下一轮 CE 改善量。评测在累计退出概率达到 0.5 时选 logits，参考实现仍执行所有候选。 |
| `deeploop` | [DeepLoop v2](https://arxiv.org/abs/2607.13491v2)：共享深度的 α/β 稳定化 | 完整堆栈执行两遍。注意力和 MLP 子层使用 `RMSNorm(α·x + F(RMSNorm(x)))`，α = √(2N)。V/O 与两组 FFN 矩阵的 Xavier 增益 β = 1/√(8N)，Q/K 沿用原初始化。关闭原生残差/x0 混合，额外 VE 路径也乘 β。 |

Ouro 门的 logits 表示**退出**。门 t 使用 `CE(t) − CE(t+1)`，下一轮改善越大，越倾向继续。最后一轮接收剩余概率。阶段切换时保留门的 Adam 动量、更新步、数据游标及 DDP 语义。即使没有梯度，后置层和输出头仍计入算力。

## 稀疏注意力

`sparse.py` 先收集入选的 K/V，再计算主注意力；反向采用独立的一阶实现。索引器和内部教师扫描完整历史的计算同样计费。可选的 `sparse_kernels.py` 后端在选中的 128 × 128 块上使用 FlexAttention，教师用融合 Triton 内核，避免构造完整分数张量。输出、辅助损失和梯度都与参考实现对照，不预设速度收益。

| 方法 | 参考 | 选择规则 |
|---|---|---|
| `dsa` | [DeepSeek report and indexer](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | 多头 ReLU 索引器、可学习的有符号头权重、键 LayerNorm 和部分 RoPE，从因果历史选 256 个 token。主干用原生 MHA，不复现 MLA、MoE 或 FP8 索引器。 |
| `qsa` | [Qwen report §2.1.2](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) | 每 4 个键取均值，再做 RMSNorm 和部分 RoPE。4 个查询头共享一个键，对 ReLU 分数求和；选完整块并补上未满尾块。教师对头取归一化均值后做最大池化。稀疏 KL 两侧均在选中的完整块上重新归一化。 |
| `nsa` | [NSA v1](https://arxiv.org/html/2502.11089v1) | 压缩、选择和局部三个分支分别使用逐头 sigmoid 门。压缩块与选择块均为不重叠的 32-token 块，局部窗口 512，组合历史选择和当前尾部。 |
| `seer` | [SeerAttention v1](https://arxiv.org/html/2410.13276v1) | token 查询对池化的 64-token 键计算门值，以内部教师的 KL 监督。从头训练的 SeerAttention 风格变体，与原论文的池化查询 AttnGate 及训练配方有别。 |
| `moba` | [Official MoBA](https://github.com/MoonshotAI/MoBA/blob/b5d58363311d3ca946f1ec444182727c15e338b5/moba/moba_efficient.py) | 各头按池化键选择块，强制纳入的当前块也占名额。块长 128，预算 256 tokens，无教师。支持独立的 gather 实现和本地内核。 |

DSA、QSA 和 Seer 均从随机初始化开始，在总计 46,875 次更新中先做 512 次完整注意力预热。辅助 KL 每层权重为 1，跨层求和。教师和索引器输入均分离梯度，辅助梯度只更新索引器。

## 线性注意力

每个打包行开始时重置状态，行内 BOS 不触发重置。保留 Q/K/V、输出投影和原生 VE。S 表示 K × V 状态，g 为遗忘门，β 为写入强度。实现附带独立的历史展开参考计算和梯度检查。

| 方法 | 参考 | 状态更新与适配 |
|---|---|---|
| `deltanet` | [DeltaNet](https://arxiv.org/abs/2406.06484) | `S ← S + k[β(v − kᵀS)]ᵀ`，`y = qᵀS`。因果卷积加 SiLU，Q/K 做 L2 归一化，β 经 sigmoid。 |
| `gla` | [Gated Linear Attention](https://arxiv.org/abs/2312.06635) | `S ← diag(g)S + kvᵀ`。低秩遗忘门和 SiLU 输出门。 |
| `kda` | [Kimi Delta Attention](https://arxiv.org/abs/2510.26692) | delta 更新前对各特征独立衰减；使用短卷积、L2 归一化的 Q/K、β 及可学习的衰减偏置。自定义一阶反向保存一份完整状态历史，FP64 梯度检查覆盖全部输入和初始状态。 |
| `retnet` | [RetNet](https://arxiv.org/abs/2307.08621) | 固定的逐头衰减、旋转位置、retention 更新与输出门。以原生逐头 RMSNorm 替代论文的 GroupNorm。 |
| `elu_qknorm`（ELU-Norm） | [Linear Transformer](https://arxiv.org/abs/2006.16236) | 逐头 RMSNorm 和固定缩放系数 8 位于正特征映射之前，累积状态的读出再经逐头 RMSNorm。 |
| `gdn` | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | `S ← α(I − βkkᵀ)S + βkvᵀ`，delta 更新前按头施加标量衰减。输出经 RMSNorm、增益和 SiLU 门；K/V 维度 128。 |

`elu_norm` 按 NormAttention 的思路去除分母，对每个头的输出做带可学习增益的 RMSNorm。ELU-Norm（`elu_qknorm`）还在 φ **之前**对 q/k 做无参数 FP32 RMSNorm，并乘以固定的 τ = 8，取自早期 pilot 的稳定尺度。这与 [ReBased](https://arxiv.org/abs/2402.10644) 的核前归一化有关，未使用它的可学习仿射变换或二次核。

## 完整注意力基线与残差连接

| 方法 | 来源与保留的机制 | 在本基准中的实现 |
|---|---|---|
| `dense_full` | 每层完整因果 MHA | 保留 VE、smear、backout 和可学习的残差/x0 系数，只校准深度。 |
| `residual_standard` | 单流 Pre-Norm 残差对照 | 每个注意力和 FFN 子层使用 `h + F(norm(h))`。保留 VE、smear、嵌入和 ReLU² FFN，对照原生块校验 logits 与梯度。 |
| `hc` | [Hyper-Connections](https://arxiv.org/html/2409.19606v3)：动态 HC，扩展倍数 4 | 每个注意力和 FFN 子层各有映射。初始化时子层 k 读取第 `k mod 4` 路，静态 `B = 1`、`A_m = e_(k mod n)`、`A_r = I`，动态投影为零，`s_α = s_β = 0.01`。嵌入复制到四路，在最终归一化前求和，每个分支宽 1536。动态映射采用无参数 RMSNorm 和无权重衰减的辅助 AdamW 参数组。初始化时等价于 `residual_standard`。 |
| `mhc` | [mHC](https://arxiv.org/html/2512.24880v2) 式 7–9：四路，门尺度 0.01，20 次 Sinkhorn-Knopp 迭代 | 展平 4 × 1536 路表示，经无参数 RMSNorm 和 6144 × 24 投影得到读、写和残差 logits。`H_pre = σ`、`H_post = 2σ`；`H_res` 在对数域对指数值执行 20 次先列后行的归一化，避免下溢除零。投影从零开始，静态偏置沿用 HC，系数至少为 FP32。迭代后行和为 1，列和近似 1。校准后的 FFN 宽度为 6016。 |
| `attnres` | [Attention Residuals](https://arxiv.org/abs/2603.15031v1)：Block AttnRes，约 8 组 | 以注意力和 FFN 子层数计深度，各子层及最终读出使用零初始化的伪查询。键用无参数 RMSNorm。23 块划为 3、3、3、3、3、3、3、2；输入源为嵌入、已完成的组之和及当前部分和。 |

四种残差方法与 `dense_full` 共用完整 MHA 骨干。对 Standard Residual 的配对差值应与相对 Dense 的总体差值分开报告。原论文在其他规模和 MoE 骨干上的增益不能直接推算本基准结果。

## 混合架构

逐层布局保存在 `layer_mixers`，加载时校验。各变体共用同一个模型构建器、审计、校准网格、准入流程和训练器。

| 方法 | 布局及来源 | 本地适配 |
|---|---|---|
| `deltanet_hybrid` | [DeltaNet](https://arxiv.org/html/2406.06484v1) 的全局混合布局 | 按 1 起算，第 2 层和第 N/2 + 1 层为完整因果 MHA，其余为 DeltaNet。N = 24 时即第 2 和第 13 层。采用原生 MHA 而非论文的局部 MQA；训练使用 FLA `chunk_delta_rule`。 |
| `gdn_hybrid` | Qwen3-Next 的 3:1 布局（`full_attention_interval: 4`） | 三个 GDN 接一个完整注意力，不复现 Qwen 的门控注意力、256 维 16Q/2KV、25% 部分 RoPE、零中心归一化或 MoE。 |
| `gdn_hybrid_swa` | [Gated DeltaNet](https://arxiv.org/html/2412.06464v1) H1，按官方 `lit_gpt` 的块顺序排列 | 奇数层为 GDN，偶数层为窗口注意力。窗口含当前位置，共 2,048 个键。**在本基准 2,048-token 上下文中，窗口覆盖所有因果位置，实际是 1:1 的 GDN 与完整注意力。** 注意力层保留 QK 归一化与 VE。 |
| `kda_hybrid` | Kimi Linear 的 KDA，搭配注册的 MLA-NoPE | 三层 KDA 接一层完整因果 MLA-NoPE，N 可被 4 整除。查询潜变量 768，联合 KV 潜变量 256，12 个头，QK/V 维度 128，不用 RoPE、VE 表或 VE 门。Kimi 原配置不压缩查询，KV 潜维为 512，有 32 个头，KDA 门控亦不同。 |
| `qsa_hybrid` | GDN/GDN/GDN/QSA | 每第四层为 QSA，预算 256 tokens，压缩比 4。最初 512 步的稠密预热只影响 QSA 层。 |
| `moba_hybrid` | MoBA 接最后三个完整注意力层 | MoBA 块长 128、预算 256（含当前块）；N = 25 时，末三层 MHA 占 12%。无预热或辅助损失。 |

骨干的交替 VE 规则为 `gdn_hybrid` 和 `gdn_hybrid_swa` 的每个注意力层配置 VE；在 `gdn_hybrid` 中，每三个 GDN 层还有一个含 VE。VE 经投影后的 V 进入 GDN。

## 其他结构

| 方法 | 机制与对照 |
|---|---|
| `dense`（SSSL） | 24 块中 18 层用短窗口 512，6 层做全局注意力。它是历史模型和冻结算力锚点；本基准以每层完整注意力的 Dense 为基线，SSSL 单列为其他方法。 |
| `mqa`、`gqa` | 原生注意力减少 KV 头数（`n_kv_head`），沿用 RoPE 和 QK 归一化。已发布训练来自冻结源码，主线移植待完成。 |
| `mla_qknorm`、`mla_nope_qknorm` | 带 QK 归一化的多头潜在注意力，分别保留和移除旋转维度。已发布训练来自冻结源码，主线移植待完成。 |
| `cla` | [CLA v3](https://arxiv.org/abs/2405.12981) 的 CLA2：一层使用其下一层的键和值做注意力，查询、注意力和输出投影仍逐层独立。以插件实现。配对跟随外壳的值嵌入层，因此不会丢弃任何值嵌入：每个没有值嵌入的层复用下一层，最后一层保留自己的键值。共享完整的多头 K/V；论文中最好的配置是 CLA 与 MQA 或 GQA 组合。共 25 层，其中 13 层有自己的 K/V。 |

训练时显式重建 MLA 的 K/V，再调用共有注意力接口。重建后的头不再做 QK 归一化，也没有输出门。没有归一化的 MLA 在 Muon 下可能出现 logits 增长，因此 QK 归一化变体单独注册。

## 历史原生模型

SSSL、MoR、LoopFormer-Fixed2 和 Circle Loop 使用早期原生轨道（`nexttransformer/models/native/`）。它们报告矩阵 FLOPs 估算，SSSL 为 4.775 GFLOPs/token。实现、运行 ID 和曲线均有保留；进入严格的等算力比较仍需要独立的完整成本凭证。Circle Loop 的研究过程见[竞技场案例](../../arena/circle-loop/)。

## 后端与数值边界

循环扫描实现提供独立的正确性参考。FLA 分块训练中，KDA 调用 `chunk_kda(disable_recompute=True)`，RetNet 调用带逐头对数衰减的 `chunk_simple_gla`。CUDA 因果卷积使用 `causal_conv1d(activation="silu")`；CPU 采用等价的 PyTorch 运算。注册路径明确指定 `scale=1`、`FLA_DISABLE_BACKEND_DISPATCH=1`。

在部分 GPU 和 Triton 3.4.0–3.7.0 的组合上，FLA 0.5.2 限制了 GDN 反向内核（FLA issue #640）。注册环境使用 Triton 3.5.1，因此 GDN 将标量对数衰减广播到各键通道，调用已验证的 KDA 内核。输出门控 RMSNorm 使用 FLA `rms_norm_gated`，CPU 和审计使用等价的 PyTorch 实现。

KDA、RetNet、GDN 的传递状态和累加使用 FP32，操作数与分块边界状态使用 BF16。KDA 和 GDN 的三角求逆使用 TF32。已训练的强门值可能使 KDA 的整块代数在 FP32 溢出，校验这类输入时应使用 FP64 循环参考实现。当前未保证二阶导数或优化的生成缓存。所有 ELU 变体使用稳定的 FP32 特征映射，运算保持 IEEE FP32、关闭 TF32，不做重计算。
