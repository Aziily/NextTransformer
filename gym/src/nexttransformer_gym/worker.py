"""Trusted subprocess protocol: intent -> durable acknowledgement -> work -> charge.

A crash between intent and charge leaves the entire uncertain reservation intact.
No host paths or executable commands are accepted from the participant API.
"""

import json
import math
import sys
from pathlib import Path
from .common import GymError


def emit(value):
    print("GYM_EVENT " + json.dumps(value, allow_nan=False), flush=True)


def operation(label, amount, function):
    emit({"kind": "intent", "label": label, "amount": str(amount)})
    reply = json.loads(sys.stdin.readline())
    if reply.get("cancel"):
        raise InterruptedError("Cancellation at operation boundary")
    if reply != {"ack": label}:
        raise RuntimeError("Accounting acknowledgement mismatch")
    failure = None
    try:
        result = function()
    except GymError as error:
        # A known failure consumes this operation's conservative bound. Never
        # refund partially executed work or describe this as an exact measurement.
        failure = error
    emit({"kind": "charge", "label": label, "amount": str(amount)})
    if json.loads(sys.stdin.readline()) != {"ack": label}:
        raise RuntimeError("Charge acknowledgement mismatch")
    if failure is not None:
        raise failure
    return result


def execute(request):
    import torch
    from .numerics import build
    from .data import streams
    from nexttransformer.models.native.config import NativeTrainConfig
    from nexttransformer.training.optimizer import build_optimizer
    from nexttransformer.native.optimizer import set_schedule
    from nexttransformer.runtime.process import rng_state, restore_rng

    torch.set_num_threads(1)
    candidate, profile, quote = (
        request["candidate"],
        request["profile"],
        request["quote"],
    )
    report = quote["cost_report"]
    init, forward, backward, optimizer_cost = (
        int(report[k]) for k in ("initialization", "forward", "backward", "optimizer")
    )
    model = operation("initialization", init, lambda: build(candidate, profile))
    optimizer = build_optimizer(model, NativeTrainConfig())
    train, dev, token_bytes = streams(profile)
    step, cumulative = 0, 0
    if request.get("checkpoint"):
        saved = torch.load(
            request["checkpoint"],
            map_location=profile.get("device", "cpu"),
            weights_only=False,
        )
        if (
            saved["candidate_hash"] != candidate["hash"]
            or saved["profile_hash"] != quote["profile_hash"]
        ):
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        train.load_state_dict(saved["data"])
        restore_rng(saved["rng"])
        model.restore_routing_rng(saved["routing_rng"])
        step, cumulative = saved["step"], saved["cumulative_flops"]
    device = torch.device(profile.get("device", "cpu"))
    batch, length = profile["batch_size"], profile["sequence_len"]
    curves = []
    if quote["operation"] == "validate":
        x = (
            torch.arange(batch * length, device=device).reshape(batch, length)
            % profile["vocab_size"]
        )

        def checks():
            model.train()
            loss = model(x, x)
            loss.backward()
            finite = bool(torch.isfinite(loss)) and all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
                if p.requires_grad
            )
            optimizer.step()
            model.eval()
            with torch.no_grad():
                first = model(x)
                altered = x.clone()
                altered[:, length // 2 :] = (altered[:, length // 2 :] + 1) % profile[
                    "vocab_size"
                ]
                second = model(altered)
                causal = torch.equal(first[:, : length // 2], second[:, : length // 2])
                import io

                checkpoint = io.BytesIO()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "data": train.state_dict(),
                        "rng": rng_state(),
                        "routing_rng": model.routing_rng_state(),
                    },
                    checkpoint,
                )
                checkpoint.seek(0)
                saved = torch.load(checkpoint, weights_only=False)
                restored = build(candidate, profile)
                restored.load_state_dict(saved["model"])
                restored_optimizer = build_optimizer(restored, NativeTrainConfig())
                restored_optimizer.load_state_dict(saved["optimizer"])
                train.load_state_dict(saved["data"])
                restore_rng(saved["rng"])
                restored.restore_routing_rng(saved["routing_rng"])
                restored.eval()
                roundtrip = torch.equal(first, restored(x))
            return {
                "construction": True,
                "numerical": finite,
                "causal": causal,
                "checkpoint": roundtrip,
                "cost_coverage": report["coverage"] == "complete",
            }

        check_cost = (
            init + forward + backward + optimizer_cost + 3 * int(report["prediction"])
        )
        result = operation("validation", check_cost, checks)
        # Numerical and initialization bounds include trusted diagnostic operations
        # separately from the model-arithmetic convention; report this explicitly.
        admitted = (
            all(result.values())
            and abs(report["anchor_ratio"] - 1) <= 0.01
            and profile.get("official_compatibility_verified", False)
        )
        if (
            profile["data"]["kind"] == "synthetic_fixture"
            and profile["kind"] == "final"
        ):
            admitted = all(result.values())
        emit(
            {
                "kind": "result",
                "value": {
                    "checks": result,
                    "admitted": admitted,
                    "candidate_hash": candidate["hash"],
                    "profile_hash": quote["profile_hash"],
                    "runtime_revision": quote["runtime_revision"],
                    "accounting": "reference-aten-math-v2",
                },
            }
        )
        return
    for update in range(step, quote["target_update"]):

        def train_step():
            model.train()
            model.set_step(update)
            model.set_row_offset(update * batch)
            optimizer.zero_grad(set_to_none=True)
            set_schedule(optimizer, update, NativeTrainConfig())
            x, y = train.batch(batch, device)
            if hasattr(train, "cursor") and train.cursor["epoch"] != 1:
                raise GymError(
                    "resource_unavailable",
                    "Research horizon exhausted the approved training prefix",
                )
            loss = model(x, y)
            loss.backward()
            if not all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
                if p.requires_grad
            ):
                raise GymError("numerical_failure", "Missing or non-finite gradient")
            optimizer.step()
            return float(loss.detach())

        cost = forward + backward + optimizer_cost
        loss = operation(f"update-{update + 1}", cost, train_step)
        cumulative += cost
        curves.append({"update": update + 1, "training_loss": loss})
        emit({"kind": "progress", "step": update + 1})
    model.eval()

    def evaluate():
        nats, byte_count, tokens = 0.0, 0, 0
        sizes = torch.tensor(token_bytes, device=device)
        with torch.no_grad():
            for _ in range(profile["dev_batches"]):
                x, labels = dev.batch(batch, device)
                logits = model(x)
                losses = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                )
                valid = sizes[labels.reshape(-1)] > 0
                nats += float((losses * valid).sum())
                byte_count += int(sizes[labels.reshape(-1)][valid].sum())
                tokens += labels.numel()
        if byte_count <= 0 or not math.isfinite(nats):
            raise GymError(
                "numerical_failure",
                "Invalid trusted scoring denominator or predictions",
            )
        return {
            "dev_bpb": nats / (math.log(2) * byte_count),
            "evaluated_tokens": tokens,
        }

    metrics = operation(
        "dev-evaluation", int(report["evaluation"]) * profile["dev_batches"], evaluate
    )
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "routing_rng": model.routing_rng_state(),
        "data": train.state_dict(),
        "candidate_hash": candidate["hash"],
        "profile_hash": quote["profile_hash"],
        "step": quote["target_update"],
        "cumulative_flops": cumulative,
        "schedule": "native-fixed-46875",
        "stage": "joint",
    }
    torch.save(saved, output.with_suffix(".tmp"))
    output.with_suffix(".tmp").replace(output)
    emit(
        {
            "kind": "result",
            "value": {
                **metrics,
                "curves": curves,
                "step": quote["target_update"],
                "training_tokens": quote["target_update"] * batch * length,
                "candidate_hash": candidate["hash"],
                "profile_hash": quote["profile_hash"],
                "runtime_revision": quote["runtime_revision"],
                "data_kind": profile["data"]["kind"],
                "accounting": "reference-aten-math-v2",
                "official_compatible": False,
            },
        }
    )


def main():
    request = json.loads(sys.stdin.readline())
    if request.get("quote", {}).get("operation") != "official":
        from .native_plugin import activate

        activate(request["candidate"])
    if request["mode"] == "inspect":
        from .numerics import inspect

        emit(
            {
                "kind": "inspection",
                "value": inspect(request["candidate"], request["profile"]),
            }
        )
    else:
        try:
            if request["quote"]["operation"] == "official":
                from .official import run

                run(request)
            else:
                execute(request)
        except InterruptedError:
            emit({"kind": "cancelled"})
        except GymError as error:
            emit(
                {
                    "kind": "failed",
                    "error_code": error.code,
                    "accounting": "conservative_operation_bound",
                }
            )


if __name__ == "__main__":
    main()
