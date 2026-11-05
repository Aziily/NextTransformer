"""Platform-owned final pipeline plans and finite pre-dispatch model metering.

This adapter is only for a separately prepared, admitted, immutable NextTransformer
snapshot. The participant never supplies a path, command, calibration proof or plan.
"""

import json
import math
from fractions import Fraction
from pathlib import Path
from .common import GymError, file_hash, require


def validate_plan(plan, candidate_hash):
    required = {
        "candidate_hash",
        "snapshot",
        "manifest",
        "config",
        "resources",
        "method",
        "initialization_flops",
    }
    require(
        isinstance(plan, dict) and set(plan) == required,
        "invalid_request",
        "Incomplete official pipeline plan.",
    )
    require(
        plan["candidate_hash"] == candidate_hash,
        "incompatible_contract",
        "Official plan is not bound to the frozen candidate.",
    )
    root = Path(plan["snapshot"]).resolve()
    require(
        root.is_dir() and isinstance(plan["manifest"], dict) and bool(plan["manifest"]),
        "resource_unavailable",
        "Prepared official runtime is unavailable.",
    )
    for relative, expected in plan["manifest"].items():
        path = root / relative
        require(
            not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and path.resolve().is_relative_to(root)
            and path.is_file()
            and file_hash(path) == expected,
            "stale_quote",
            "Official snapshot differs from its reviewed manifest.",
        )
    # Cover every importable scientific file; an omitted added plugin cannot escape the manifest.
    scientific = [
        p
        for package in ("nexttransformer", "nanochat")
        for p in (root / package).rglob("*.py")
    ]
    require(
        all(p.relative_to(root).as_posix() in plan["manifest"] for p in scientific),
        "incomplete_coverage",
        "Official manifest omits executable scientific source.",
    )
    require(
        plan["config"] in plan["manifest"],
        "missing_evidence",
        "Admitted config must be pinned in the snapshot.",
    )
    config_path = root / plan["config"]
    raw = json.loads(config_path.read_text())
    proof = (config_path.parent / raw["calibration"]["cost_record"]).resolve()
    for evidence in (proof, proof.with_name("dense.json")):
        require(
            evidence.is_relative_to(root)
            and evidence.relative_to(root).as_posix() in plan["manifest"],
            "missing_evidence",
            "Cost proof and anchor must be pinned in the official manifest.",
        )
    require(
        type(plan["initialization_flops"]) is int and plan["initialization_flops"] > 0,
        "incomplete_coverage",
        "Initialization must have a positive audited bound.",
    )
    return root


class FinalMeter:
    """A conservative finite allocation, charged before each trusted model call.

    Overestimation is explicit; no successful result claims exact execution charges
    from this guard. Scientific accounting remains the trainer's audited ledger.
    """

    def __init__(self, allocation, forward_per_token, training_per_token):
        self.allocation, self.spent = allocation, 0
        self.forward_per_token, self.training_per_token = (
            forward_per_token,
            training_per_token,
        )

    def charge(self, amount):
        require(
            type(amount) is int and amount >= 0,
            "invalid_request",
            "Invalid final charge.",
        )
        require(
            self.spent + amount <= self.allocation,
            "insufficient_budget",
            "Final allocation exhausted before model dispatch.",
        )
        self.spent += amount

    def before_forward(self, model, inputs):
        tokens = inputs[0].numel()
        # Training rate includes backward and the amortized fixed optimizer update.
        rate = self.training_per_token if model.training else self.forward_per_token
        self.charge(math.ceil(rate * tokens))


def official_world(profile):
    devices = profile.get("cuda_visible_devices", [])
    require(
        isinstance(devices, list)
        and len(devices) in (8, 16, 32)
        and len(set(str(device) for device in devices)) == len(devices),
        "resource_unavailable",
        "Official single-node dispatch requires an explicit allocation of 8, 16 or 32 distinct GPUs.",
    )
    return len(devices)


def invoke(module, arguments):
    import runpy
    import sys

    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def train_rank(rank, world, request, output, forward_rate, training_rate):
    """Spawned trusted rank; the original trainer owns data, DDP and evaluation."""
    import os
    import sys
    import torch.distributed as dist

    plan = request["profile"]["official_plan"]
    root = validate_plan(plan, request["candidate"]["hash"])
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    initialize_group = dist.init_process_group

    def file_group(*args, **kwargs):
        # File rendezvous adds no host listener or port allocation race.
        kwargs.update(
            init_method=(Path(output) / "rendezvous").as_uri(),
            rank=rank,
            world_size=world,
        )
        return initialize_group(*args, **kwargs)

    dist.init_process_group = file_group
    meter = FinalMeter(
        int(request["quote"]["bound"]) // world, forward_rate, training_rate
    )
    import nexttransformer.models.lm as lm

    original = lm.build_model

    def metered_build(*args, **kwargs):
        meter.charge(plan["initialization_flops"])
        model = original(*args, **kwargs)
        model.register_forward_pre_hook(meter.before_forward)
        return model

    lm.build_model = metered_build
    try:
        invoke(
            "nexttransformer.train",
            [
                "--formal",
                "--config",
                str(root / plan["config"]),
                "--resources",
                plan["resources"],
                "--output",
                str(Path(output) / "run"),
                "--micro-batch",
                str(request["profile"]["batch_size"]),
                "--resume",
                "never",
                "--wandb-mode",
                "disabled",
            ],
        )
        (Path(output) / f"rank-{rank}.json").write_text(
            json.dumps({"spent": str(meter.spent)})
        )
    except GymError as error:
        (Path(output) / f"failure-{rank}.json").write_text(
            json.dumps({"code": error.code})
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(request):
    """Run inside the trusted worker subprocess, outside the agent/verifier API."""
    import os
    import sys
    from .worker import emit, operation

    plan, candidate, quote = (
        request["profile"]["official_plan"],
        request["candidate"],
        request["quote"],
    )
    root = validate_plan(plan, candidate["hash"])
    # Require a dedicated prepared checkout, never publish into the user's working tree.
    require(
        str(root) != request["platform_root"],
        "denied_access",
        "Official execution needs a dedicated immutable staging checkout.",
    )
    os.chdir(root)
    sys.path.insert(0, str(root))
    from nexttransformer.protocol.admission import admit_config
    from nexttransformer.protocol.spec import Protocol

    config, proof = admit_config(root / plan["config"])
    from .native_plugin import model_config

    expected = model_config(
        candidate,
        dict(
            width=1536,
            heads=12,
            sequence_len=2048,
            vocab_size=32768,
            formal_dimensions=True,
        ),
    )
    require(
        config == expected and config.method == plan["method"],
        "incompatible_contract",
        "Admitted config differs from the frozen native model config.",
    )
    prefix = "plugin/" + candidate["method"] + "/"
    plugin = root / "nexttransformer" / "methods" / candidate["method"]
    expected_files = {
        name[len(prefix) :]: source
        for name, source in candidate["files"].items()
        if name.startswith(prefix)
    }
    actual_files = {
        path.relative_to(plugin).as_posix(): path.read_bytes().decode("utf-8")
        for path in plugin.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    require(
        actual_files == expected_files,
        "incompatible_contract",
        "Staged plugin differs from the frozen source package.",
    )
    protocol = Protocol()
    training_rate = Fraction(
        max(stage["cost"]["total"] for stage in proof["stages"]),
        protocol.global_batch_tokens,
    )
    forward_rate = Fraction(
        max(stage["cost"]["phases"]["forward"] for stage in proof["stages"]),
        protocol.global_batch_tokens,
    )
    world = official_world(request["profile"])
    import torch

    require(
        torch.cuda.device_count() == world,
        "resource_unavailable",
        "Visible GPUs differ from the explicit final allocation.",
    )
    output = Path(request["final_output_root"]) / (quote["id"] + "-official")
    output.mkdir(parents=True, exist_ok=False)

    # Whole-run reservation remains unsettled until the trusted pipeline terminates.
    # An abnormal exit is reconciled conservatively by the controller.
    def pipeline():
        try:
            torch.multiprocessing.spawn(
                train_rank,
                args=(world, request, str(output), forward_rate, training_rate),
                nprocs=world,
                join=True,
            )
        except torch.multiprocessing.ProcessRaisedException:
            # spawn has joined/terminated the entire process group before settling.
            failures = [
                json.loads(path.read_text())["code"]
                for path in output.glob("failure-*.json")
            ]
            if "insufficient_budget" in failures:
                raise GymError(
                    "insufficient_budget",
                    "Final allocation exhausted before model dispatch.",
                ) from None
            raise
        spent = sum(
            int(json.loads((output / f"rank-{rank}.json").read_text())["spent"])
            for rank in range(world)
        )
        audit = root / "evidence" / "audits" / (quote["id"] + ".json")
        invoke(
            "nexttransformer.audit_run",
            [
                str(output / "run"),
                "--method",
                plan["method"],
                "--output",
                str(audit),
            ],
        )
        invoke(
            "nexttransformer.export_run",
            [
                str(output / "run"),
                "--method",
                plan["method"],
                "--name",
                plan["method"],
                "--audit",
                audit.relative_to(root).as_posix(),
                "--file",
                quote["id"],
            ],
        )
        record = json.loads(
            (root / "results" / "runs" / (quote["id"] + ".json")).read_text()
        )
        return {
            "official_compatible": True,
            "record": record,
            "allocation_guard_spent": str(spent),
            "world_size": world,
            "accounting": "conservative_final_allocation",
            "candidate_hash": candidate["hash"],
            "profile_hash": quote["profile_hash"],
            "runtime_revision": quote["runtime_revision"],
        }

    result = operation("official-pipeline", int(quote["bound"]), pipeline)
    emit({"kind": "result", "value": result})
