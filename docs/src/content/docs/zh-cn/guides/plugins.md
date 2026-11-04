---
title: 添加方法（插件）
description: 以插件的形式添加新架构，无需修改共享文件。
---

新方法就是一个插件：`nexttransformer/methods/` 下的一个目录，由注册表自动发现。它不需要修改任何共享文件，因此不会改变其他方法。
成本证明记录插件自身和共享核心的源码指纹。
开始之前，请先[提交一个提案 issue](https://github.com/Interactive-Active-AI/NextTransformer/issues/new/choose)。

## 目录结构

```text
nexttransformer/methods/<name>/
  method.json            声明，读取时不导入代码
  __init__.py            hooks（方法的其他模块可以放在旁边）
configs/<family>/<name>.json                        校准后的配置，由 calibrate 写出
evidence/costs/<proof-set>/<name>.json, dense.json  成本证明和 SSSL 锚点证明，由 calibrate 写出
results/methods/<name>.json                         模型卡片
tests/test_<name>.py                                测试
```

## 声明

```json
{
 "name": "output_gate",
 "family": "others",
 "title": "Attention with a learned output gate",
 "paper": {"title": "<paper title>", "url": "<paper URL>"},
 "options": {"gate_bias": 0.0},
 "debug_options": {"gate_bias": 0.5},
 "shell": {"layers": 24, "mlp_width": 6144, "window_pattern": "L"},
 "layer_multiple": 1
}
```

| 字段 | 含义 |
|---|---|
| `name` | 小写字母、数字和 `_`；与目录名相同；不能与内置方法重名 |
| `family` | 展示用的模型族：`looped`、`sparse`、`linear`、`residual`、`hybrid` 或 `others` |
| `options` | 方法自己的参数及其默认值；用 `config.option(name)` 读取 |
| `debug_options` | 测试所用小型 CPU 配置（`debug_config`）的取值 |
| `shell` | 外壳的起始 `layers`、`mlp_width` 和 `window_pattern`（默认 24、6144、`L`） |
| `layer_multiple` | 重复单元的大小；校准和验证会让层数保持为它的倍数 |

## Hooks

在 `__init__.py` 中定义一个或多个：

| Hook | 用途 |
|---|---|
| `mixer(config, layer)` | 返回某一层的注意力模块，调用方式为 `(x, ve, cos_sin, window, kv_cache)` |
| `ffn(config, layer)` | 返回某一层的前馈模块，调用方式为 `(x)` |
| `block(block, config, layer)` | 包装或替换一个块，调用方式为 `(x, ve, cos_sin, window, kv_cache)` |
| `model` | `nexttransformer.models.lm.ReferenceLM` 的子类，用于改变前向流程的方法 |
| `frozen` | 方法不使用的外壳参数名，例如 `("backout_lambda",)` |
| `init(model)` | 在共享初始化之后初始化方法自己的参数 |
| `stages(config)` | 按顺序排列的连续训练阶段 `[(name, updates), ...]`，总和为 46,875；默认是一个 `joint` 阶段 |
| `check(config)` | 选项取值无效时抛出 `ValueError` |

插件要遵守外壳的约定：

- 每个块都保留 `attn.c_q`、`attn.c_k`、`attn.c_v`、`attn.c_proj`、`mlp.c_fc` 和 `mlp.c_proj` 这些线性层；外壳负责初始化它们，写回残差流的投影初始化为 0。
- 每个可训练参数在每次更新中都要得到梯度；不使用的外壳参数列在 `frozen` 中。
- 模块可以设置张量 `last_aux`（训练时加上的辅助损失），也可以定义 `set_step(step)`。
- 一旦有运行存在，参数名就要保持稳定：检查点按名称保存张量。

插件不能修改内置方法或其他插件、成本账本、训练器、优化器、数据或评测，也不能新增展示用的模型族。

## 示例

```python
"""Causal attention whose output passes a learned sigmoid gate (an illustration of the hooks)."""
import torch
from torch import nn

from nexttransformer.models.backbone import CausalSelfAttention, Linear


class GatedAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.gate = Linear(config.width, config.width, bias=False)
        self.bias = config.option("gate_bias")

    def forward(self, x, ve, cos_sin, window, kv_cache):
        return super().forward(x, ve, cos_sin, window, kv_cache) * torch.sigmoid(self.gate(x) + self.bias)


def mixer(config, layer):
    return GatedAttention(config, layer)


def init(model):
    for block in model.transformer.h:
        nn.init.zeros_(block.attn.gate.weight)
```

`tests/plugins/` 中的测试插件覆盖了一个 mixer、一个带选项检查的前馈网络，以及一个有两个训练阶段和一个冻结外壳参数的整模型插件。
正式发布的方法应遵循论文的精确表述，并在[方法来源说明](../../methods/sources/)中记录所有调整。

## 成本

把方法校准到共同预算，与 SSSL 锚点相差 1% 以内：

```bash
python -m nexttransformer.protocol.calibrate --methods <name> --cache .agent-work/cost-cache \
    --output configs --results evidence/costs/<name>-<date>
```

校准会搜索登记的网格（16 到 32 层，以 `layer_multiple` 为步长；然后是 3d 到 5d、128 的倍数的 FFN 宽度），写出配置和证明，并在旁边写出 SSSL 锚点证明。

- softmax 注意力请用 `nanochat.flash_attention.flash_attn.flash_attn_func` 计算；账本会精确统计其因果或窗口的边数。
- 其他计算都用账本能统计的算子表达；审计遇到未覆盖的算子会指出其名称并拒绝。
- 自定义 kernel 需要一条 CPU 审计能运行的纯 PyTorch 路径，以及证明两者计算同一函数的测试。

正式结果需要经过审计的 PyTorch 构建（2.9.1+cu128）；研究模式可以用任意构建训练。

## 提交 pull request 之前

```bash
python -m pytest tests/test_<name>.py -q && python -m pytest -q -n auto
python -m nexttransformer.protocol.fingerprint        # 每个已发布的方法仍与其证明一致
python -m nexttransformer.arena --scope origin/main   # pull request 只添加了方法自己的文件
python -m nexttransformer.checks.changes --base origin/main
launch/train.sh --config configs/<family>/<name>.json --site-config site.local.json \
    --output runs/<name>-smoke --micro-batch 8 --smoke-steps 2
```

完成上述检查后，在你的 fork 上用已校准的配置和 `--formal` 完成正式训练，按[结果提交指南](../submit-a-result/)审计并导出结果。
然后向上游提交一个 PR，关联提案 issue，并附上插件、校准配置、算力证明、测试、模型卡、来源说明及正式结果。维护者会在合并前复核算力证明。
竞技场参赛项目请遵循[竞技场规则](../../arena/rules/)。

如果要试用放在仓库之外的插件，把它们的根目录列在 `NEXTTRANSFORMER_METHOD_PATHS` 中。指纹只覆盖仓库内的插件根目录，因此这类方法只能用于研究。
