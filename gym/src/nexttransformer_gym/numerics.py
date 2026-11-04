"""Trusted lowering into the unchanged NextTransformer model shell.

Participant Python never reaches import machinery. The only variable computation
is a finite, branch-free tree of causal pointwise operations, validated twice.
"""

from .package import expression


def apply_tree(tree, x):
    import torch.nn.functional as F

    op = tree["op"]
    if op == "input":
        return x
    args = [apply_tree(arg, x) for arg in tree["args"]]
    if op == "relu":
        return F.relu(args[0])
    if op == "square":
        return args[0].square()
    if op == "silu":
        return F.silu(args[0])
    if op == "tanh":
        return args[0].tanh()
    if op == "add":
        return args[0] + args[1]
    if op == "mul":
        return args[0] * args[1]
    raise ValueError("Uncovered operation")


def model_config(candidate, profile):
    from nexttransformer.models.config import ModelConfig

    return ModelConfig(
        method="dense_full",
        width=profile["width"],
        heads=profile["heads"],
        sequence_len=profile["sequence_len"],
        vocab_size=profile["vocab_size"],
        layers=candidate["architecture"]["layers"],
        mlp_width=profile["width"] * candidate["architecture"]["mlp_ratio"],
        window_pattern="L",
        scale="formal" if profile.get("formal_dimensions") else "debug",
    )


def build(candidate, profile):
    from torch import nn
    from nexttransformer.models import build_model

    tree = expression(candidate["expression"])
    model = build_model(
        model_config(candidate, profile), profile.get("device", "cpu"), seed=42
    )

    class ExpressionMLP(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.c_fc, self.c_proj = original.c_fc, original.c_proj

        def forward(self, x):
            return self.c_proj(apply_tree(tree, self.c_fc(x)))

    for block in model.transformer.h:
        block.mlp = ExpressionMLP(block.mlp)
    return model


def inspect(candidate, profile):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from nexttransformer.protocol.compute import (
        Ledger,
        attention_rules,
        count_optimizer,
    )

    initialization = Ledger()
    ledger = Ledger()
    evaluation = Ledger()
    prediction = Ledger()
    with FakeTensorMode(allow_non_fake_inputs=True):
        with initialization:
            model = build(candidate, {**profile, "device": "cpu"})
        x = torch.zeros(
            (profile["batch_size"], profile["sequence_len"]), dtype=torch.long
        )
        with attention_rules(model, ledger, "joint"), ledger:
            loss = model(x, x)
            ledger.phase = "backward"
            loss.backward()
        count_optimizer(model, ledger)
        parameters = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad(), attention_rules(model, prediction, "joint"), prediction:
            model(x)
        with torch.no_grad(), attention_rules(model, evaluation, "joint"), evaluation:
            logits = model(x)
            losses = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                x.reshape(-1),
                reduction="none",
            )
            valid = torch.ones_like(x.reshape(-1), dtype=torch.bool)
            (losses * valid).sum()
    report = ledger.report()
    init = initialization.report()
    unknown = {
        **report["unknown"],
        **init["unknown"],
        **evaluation.report()["unknown"],
        **prediction.report()["unknown"],
    }
    return {
        "coverage": "complete" if not unknown else "incomplete",
        "unknown": unknown,
        "initialization": str(init["total"]),
        "evaluation": str(evaluation.report()["total"]),
        "prediction": str(prediction.report()["total"]),
        "forward": str(report["phases"]["forward"]),
        "backward": str(report["phases"]["backward"]),
        "optimizer": str(report["phases"]["optimizer"]),
        "parameters": parameters,
        "training_flops_per_token": report["total"] / x.numel(),
        "anchor_ratio": report["total"] / x.numel() / 4672398325.44785,
        "accounting_version": report["formula_version"],
        "scope": "fixed-shape branch-free FFN expression on trusted ReferenceLM",
    }
