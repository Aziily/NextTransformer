"""Deterministic platform-owned plugin source for separately admitted snapshots."""

from .package import expression


def plugin_source(tree):
    expression(tree)

    def render(node):
        args = [render(arg) for arg in node["args"]]
        op = node["op"]
        if op == "input":
            return "x"
        if op == "relu":
            return f"F.relu({args[0]})"
        if op == "square":
            return f"({args[0]}).square()"
        if op == "silu":
            return f"F.silu({args[0]})"
        if op == "tanh":
            return f"({args[0]}).tanh()"
        if op == "add":
            return f"({args[0]} + {args[1]})"
        if op == "mul":
            return f"({args[0]} * {args[1]})"
        raise ValueError("Unsupported expression")

    return (
        """import torch
from torch import nn
from torch.nn import functional as F
from nexttransformer.models.backbone import GPT, Linear

class ExpressionFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.width, config.mlp_width, bias=False)
        self.c_proj = Linear(config.mlp_width, config.width, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        return self.c_proj("""
        + render(tree)
        + """)

def ffn(config, layer):
    return ExpressionFFN(config)

def init(model):
    # Plugin reset traverses embeddings as well as projections. Reproduce the
    # canonical dense-shell seed/order rather than inherit that extra RNG drift.
    torch.manual_seed(42)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.reset_parameters()
    GPT.init_weights(model)
"""
    )
