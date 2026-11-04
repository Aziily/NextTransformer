"""Native model construction and NextTransformer cost inspection in a worker."""

from .native_plugin import activate, model_config, check_interface


def build(candidate, profile):
    activate(candidate)
    from nexttransformer.models import build_model

    model = build_model(
        model_config(candidate, profile), profile.get("device", "cpu"), seed=42
    )
    check_interface(model)
    return model


def inspect(candidate, profile):
    activate(candidate)
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
        "scope": "native plugin trace for the pinned configuration; not a proof of every Python branch",
    }
