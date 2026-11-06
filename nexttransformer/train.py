"""Formal trainer for admitted reference configurations, one model on 8 or 16 GPUs.

The protocol is fixed by nexttransformer.models.config.Protocol: 49.152B input tokens,
1,048,576 tokens and 512 packed rows per update, 46,875 updates, seed 42, the
native Muon+AdamW schedule and BF16 compute (NANOCHAT_DTYPE, no autocast).
val_bpb runs at step 0, every 250 steps and at the final step; full CORE at
8K/16K/24K/32K/40K and 46,875; the final train/val BPB is recorded separately.
Each update charges the audited theory cost of the stage or depth actually
executed (BudgetState); evaluation time and forward FLOPs are kept separately.
Data always follows the frozen eight-stream layout: with 16 ranks (two nodes)
each stream runs on two ranks that keep alternate halves of its rows, so the
global row order and row ids are the 8-rank run's (protocol.share_batches).

Segments are idempotent: `--resume auto` continues from the newest complete
checkpoint, moves incomplete checkpoint directories aside, never starts a new
run over an existing one, and refuses changed source, configuration or
software. `--max-wall-hours` ends a segment at a step boundary after a
checkpoint, so a cluster time limit loses at most the step in progress.
`--smoke-steps` runs are labelled smoke and can never become formal results.

Research mode is the default: a configuration whose cost evidence is stale (changed source,
another PyTorch build) or outside the shared budget still trains, with the findings recorded,
and the run cannot be exported as a result. `--formal` admits the configuration and refuses
every finding; results need it in every segment, and a run keeps its mode on resume.
"""
from nexttransformer.runtime.site import add_runtime_arguments, bootstrap_runtime


def rank_kernel_caches(environ):
    """Give every local rank its own Triton and Inductor cache directory.

    A process benchmarks each kernel configuration it finds no cached choice for. Ranks sharing
    one cache keep only the last writer's choice, so a later process could run another
    configuration than the rank used before: after a four-node resume, one rank's Muon update
    (an Inductor reduction) changed. run.sh archives both caches per node, so every rank of
    every segment reuses its own earlier choices.
    """
    if "LOCAL_RANK" in environ:
        for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            if environ.get(key): environ[key] = f"{environ[key].rstrip('/')}/rank{environ['LOCAL_RANK']}"


if __name__ == "__main__":
    import os
    bootstrap_runtime()
    rank_kernel_caches(os.environ)

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.common import COMPUTE_DTYPE
from nexttransformer.runtime.files import atomic_json
from nexttransformer.runtime.process import (append_event, initialize, reduce_sum, restore_rng, rewind_events, rng_state,
                                         source_manifest, synchronize)
from nexttransformer.runtime.tracking import RunTracker, metric_payload
from nexttransformer.runtime.tracking_auth import preflight
from nexttransformer.runtime.checkpoint import load_checkpoint, save_checkpoint
from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nexttransformer.native.optimizer import set_schedule
from nanochat.tokenizer import RustBPETokenizer

from nexttransformer.protocol.calibrate import science_hash
from nexttransformer.protocol.fingerprint import method_source
from nexttransformer.models.config import CHUNK_BACKENDS, LINEAR, SPARSE
from nexttransformer.protocol.spec import Protocol, TOKENIZER_SHA256
from nexttransformer.models.looped import huginn_depth
from nexttransformer.models.lm import SPARSE_BACKENDS, build_model
from nexttransformer.models.plugins import stage_at
from nexttransformer.models.registry import PLUGIN_METHODS
from nexttransformer.training.optimizer import build_optimizer, transition_stage
from nexttransformer.protocol.admission import (BudgetState, admit_config, data_layout, events_at, review_config,
                                                share_batches)
from nexttransformer.evaluation.formal import evaluate_bpb, evaluate_core
from nexttransformer.checks.validate import source_hash as reference_source_hash

# Dense BF16 tensor-core peaks, used only for the reported execution MFU; unknown devices report no MFU.
# Ordered so that a name is matched before any of its substrings ("H200" before "H20").
PEAK_FLOPS = (("B200", 2250e12), ("H200", 989e12), ("H100", 989e12), ("H800", 989e12), ("H20", 148e12),
              ("A100", 312e12), ("A800", 312e12), ("L40S", 362e12))


def stage_cost(proof, config, step):
    """Audited (total, forward) theory FLOPs of the update at `step`."""
    stages = proof["stages"]
    if config.method in PLUGIN_METHODS:
        name, _ = stage_at(config, step)
        stages = [s for s in stages if s["stage"] == name]
    elif config.method == "ouro":
        name = "gate_only" if step >= gate_boundary(config) else "joint"
        stages = [s for s in stages if s["stage"] == name]
    elif config.method == "huginn":
        depth = huginn_depth(config, step)
        stages = [s for s in stages if s["depth"] == depth]
    elif config.has_sparse_burnin:
        name = "burnin" if step < config.burnin_steps else "joint"
        stages = [s for s in stages if s["stage"] == name]
    if len(stages) != 1:
        raise ValueError("cost proof does not identify the executed training stage")
    cost = stages[0]["cost"]
    return cost["total"], cost["phases"]["forward"]


def gate_boundary(config):
    """First optimizer step of Ouro's gate-only stage."""
    return Protocol().steps-config.ouro_gate_steps


def software_identity(config, sparse_backend="reference"):
    """Source and kernel fingerprints that must stay fixed across segments."""
    native = source_manifest()
    nccl = torch.cuda.nccl.version() if torch.cuda.is_available() else None
    identity = dict(science_hash=science_hash(), reference_source_hash=reference_source_hash(),
                    native_source_sha256=native["sha256"], git_commit=native["commit"] or None,
                    python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    nccl=".".join(map(str, nccl)) if isinstance(nccl, tuple) else nccl,
                    code_snapshot=os.environ.get("NEXTTRANSFORMER_CODE_SNAPSHOT"))
    if config.has_sparse:
        identity.update(sparse_backend=sparse_backend)
    if config.backend in CHUNK_BACKENDS and torch.cuda.is_available():
        package = importlib.metadata.distribution("fla-core")
        identity.update(fla_core=package.version,
                        fla_record_sha256=hashlib.sha256(package.read_text("RECORD").encode()).hexdigest())
        if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
            raise RuntimeError("set FLA_DISABLE_BACKEND_DISPATCH=1 (registered Triton path)")
    if config.method not in LINEAR:
        # The native attention kernel (FA3 from the kernels cache, or the SDPA fallback)
        # changes numerics, so a resumed segment must load the same one.
        from nanochat import flash_attention
        identity.update(native_attention="fa3" if flash_attention.USE_FA3 else
                        "sdpa" if torch.backends.cuda.flash_sdp_enabled() else "sdpa-noflash")
    return identity


def peak_flops(device):
    if device.type != "cuda": return None
    name = torch.cuda.get_device_name(device)
    return next((value for key, value in PEAK_FLOPS if key in name), None)


def checkpoints(output):
    """Complete checkpoints by step, plus directories without a completion marker."""
    complete, incomplete = {}, []
    for path in Path(output).glob("checkpoint_[0-9]*"):
        if path.is_dir():
            if (path/"complete.json").exists(): complete[int(path.name.removeprefix("checkpoint_"))] = path
            else: incomplete.append(path)
    return complete, incomplete


def prepare_output(output, resume, rank, world):
    """Choose the checkpoint to continue from; rank 0 moves anything unusable aside."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    complete, incomplete = checkpoints(output)
    if resume == "never" and (complete or (output/"provenance.json").exists()):
        raise ValueError("output already contains a run; use --resume auto or a new --output")
    if resume not in ("auto", "never"):
        path = Path(resume)
        if not (path/"complete.json").exists(): raise ValueError("--resume path is not a complete checkpoint")
        if any(step > int(path.name.removeprefix("checkpoint_")) for step in complete):
            raise ValueError("output contains later checkpoints; resume from the newest or use a new --output")
        chosen = path
    else:
        chosen = complete[max(complete)] if complete else None
    if rank == 0:
        for path in incomplete:
            target = output/"incomplete_checkpoints"/f"{path.name}-{stamp}"
            target.parent.mkdir(exist_ok=True); path.rename(target)
        if chosen is None and any((output/name).exists() for name in ("provenance.json", "metrics.jsonl")):
            # Started but never reached a checkpoint: keep the record, restart at step 0.
            target = output/"abandoned"/stamp; target.mkdir(parents=True)
            for path in list(output.iterdir()):
                if path.is_file() and path.name != "run.lock": path.rename(target/path.name)
    if world > 1: dist.barrier()
    return chosen


def prune_rolling(output, keep, steps, protocol_every):
    """Delete superseded fault-recovery checkpoints; protocol checkpoints are permanent."""
    complete, _ = checkpoints(output)
    rolling = sorted(s for s in complete if s % protocol_every and s != steps)
    for step in rolling[:max(len(rolling)-keep, 0)]:  # a negative bound would delete the oldest ones
        shutil.rmtree(complete[step])


class ReferenceTracker(RunTracker):
    """Rank-zero W&B mirror of the local JSONL; one W&B run across segments."""
    def __init__(self, args, rank, config, train, state, source):
        args.wandb_metrics = "full"
        super().__init__(args, rank, config, train, state, source)
        if self.run is not None:
            self.run.config.update({"metrics_schema": "reference-v1"}, allow_val_change=True)
            self.record["metrics_schema"] = "reference-v1"

    def log(self, event):
        if self.run is None: return
        self.run.log(metric_payload(event, self.model_config, "full"))
        self.record["last_step"] = max(self.record["last_step"], event["step"])
        atomic_json(self.path, self.record)


def start_tracker(args, rank, config, train, state, source, device):
    tracker, error = None, None
    try: tracker = ReferenceTracker(args, rank, config, train, state, source)
    except Exception as exc: error = exc
    failed = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized(): dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item(): raise RuntimeError("W&B tracker initialization failed on rank 0") from error
    return tracker


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runtime_arguments(parser)
    parser.add_argument("--config", required=True, type=Path, help="configs/<family>/<method>.json")
    parser.add_argument("--formal", action="store_true",
                        help="refuse stale or out-of-budget cost evidence; required for results (default: research mode)")
    parser.add_argument("--research", dest="formal", action="store_false",
                        help="research mode, the default; overrides an earlier --formal such as a launcher's")
    parser.add_argument("--output", required=True, type=Path, help="stable persistent run directory")
    parser.add_argument("--resume", default="auto", help="auto, never, or a checkpoint directory")
    parser.add_argument("--max-wall-hours", type=float, default=0., help="end this segment after a checkpoint (0: off)")
    parser.add_argument("--rolling-every", type=int, default=500, help="fault-recovery checkpoint interval")
    parser.add_argument("--rolling-keep", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, help="BPB rows per rank and call (default: --micro-batch)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke-steps", type=int, default=0, help="bounded test; never a formal result")
    parser.add_argument("--smoke-val-tokens", type=int, default=0, help="0: one distributed evaluation batch")
    parser.add_argument("--smoke-core-per-task", type=int, default=0,
                        help="smoke CORE examples per task at the last step (0: skip, -1: all; >=32 so "
                             "10-shot tasks can sample; still smoke)")
    parser.add_argument("--smoke-accumulation", type=int, default=0, help="smoke-only microbatches per update")
    parser.add_argument("--wandb-fallback", choices=("offline", "fail"), default="offline",
                        help="if online W&B authentication fails on rank 0")
    parser.add_argument("--sparse-backend", choices=SPARSE_BACKENDS, default="reference",
                        help='sparse methods: "kernel" computes the same function on CUDA kernels; CORE always '
                             'uses the reference path, whose prompts need not be a multiple of the kernel block')
    parser.add_argument("--deterministic-attention", action="store_true",
                        help="require deterministic PyTorch kernels for SDPA; excludes FA3 and SDPA flash")
    parser.add_argument("--speedups", default="",
                        help="fused execution of shared components (nexttransformer/runtime/speedups), off by default: "
                             "\"exact\" (bit-identical to the default path; the only choice formal runs accept), "
                             "\"all\" (research only) or a comma list")
    args = parser.parse_args(argv)
    if args.speedups:
        from nexttransformer.runtime import speedups
        try: speedups.selection(args.speedups)
        except ValueError as exc: parser.error(str(exc))
        if args.formal and not speedups.formal(args.speedups):
            parser.error("formal runs accept only speedups inside \"exact\", which reproduce the default path bit for bit")
    if not args.micro_batch: parser.error("--micro-batch (or MICRO_BATCH / site micro_batch) is required")
    args.micro_batch = int(args.micro_batch)
    args.eval_batch = args.eval_batch or args.micro_batch
    if min(args.log_every, args.rolling_every) < 1 or args.rolling_keep < 0: parser.error("intervals must be positive")
    if args.smoke_accumulation and not args.smoke_steps: parser.error("--smoke-accumulation is smoke-only")
    return args


def main(argv=None):
    args = parse(argv)
    p, train_config = Protocol(), NativeTrainConfig()
    p.validate_train_config(train_config)
    smoke = args.smoke_steps > 0
    if args.deterministic_attention:
        from nanochat import flash_attention
        if flash_attention.USE_FA3: raise RuntimeError("FA3 backward is not deterministic; run without FA3")
        torch.backends.cuda.enable_flash_sdp(False)
        # Efficient SDPA also needs its deterministic backward path for MLA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.formal: (config, proof), findings = admit_config(args.config), []
    else: config, proof, findings = review_config(args.config)
    mode = "formal" if args.formal else "research"
    source = proof.get("method_source")
    source_hash = method_source(config.method, source["files"], config) if source else None
    dense_per_token = json.loads(args.config.read_text())["calibration"]["dense_flops_per_token"]
    if not args.formal and int(os.environ.get("RANK", "0")) == 0:
        print(f"research mode: {proof['flops_per_token']/1e9:.4f} GFLOPs per input token, "
              f"{proof['flops_per_token']/dense_per_token-1:+.3%} versus the shared budget; the run cannot be "
              "exported as a result (train with --formal)" + "".join(f"\n  finding: {f}" for f in findings), flush=True)

    wandb_note = None
    if int(os.environ.get("RANK", "0")) == 0 and args.wandb_mode == "online":
        try: preflight(args.wandb_mode)
        except RuntimeError as exc:
            if args.wandb_fallback == "fail": raise
            wandb_note = f"online W&B unavailable ({exc}); offline mirror, sync later"
            args.wandb_mode = "offline"; print(wandb_note, flush=True)
    rank, world, device = initialize(p.seed)
    stream_rank, stream_world, shares = data_layout(rank, world)
    if smoke and args.smoke_accumulation: accumulation = args.smoke_accumulation
    elif stream_world == p.data_parallel_size or not smoke:
        accumulation, rest = divmod(p.accumulation(args.micro_batch, stream_world), shares)
        if rest: raise ValueError("micro batch does not divide this rank's share of the global batch")
    else: accumulation = train_config.accumulation(config, world, args.micro_batch)
    step_tokens = world*accumulation*args.micro_batch*config.sequence_len
    full_batch = step_tokens == p.global_batch_tokens
    if device.type == "cuda" and not smoke and COMPUTE_DTYPE != torch.bfloat16:
        raise ValueError("formal runs use NANOCHAT_DTYPE=bfloat16")
    import fcntl
    run_lock = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        run_lock = (args.output/"run.lock").open("a")
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if world > 1: dist.barrier()
    resume = args.resume = prepare_output(args.output, args.resume, rank, world)

    record, manifest_hash = verify_resources(args.resources, args.data_dir)
    if not smoke and record["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("prepared tokenizer differs from the pinned protocol tokenizer")
    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    tokenizer = RustBPETokenizer.from_directory(args.resources/"tokenizer")
    if tokenizer.get_vocab_size() != config.vocab_size: raise ValueError("model/tokenizer vocabulary mismatch")
    data_dir = args.data_dir or args.resources/"base_data_climbmix"
    stream = BestFitStream(data_dir, tokenizer, "train", stream_rank, stream_world, config.sequence_len)
    stream_rows = shares*accumulation*args.micro_batch  # rows of each stream per update
    stop = min(p.steps, args.smoke_steps) if smoke else p.steps
    val_tokens = (args.smoke_val_tokens or args.eval_batch*world*config.sequence_len) if smoke else p.val_tokens

    if args.sparse_backend != "reference" and not config.has_sparse:
        raise ValueError("--sparse-backend applies to sparse methods only")
    if args.speedups:
        from nexttransformer.runtime import speedups
        speedups.install(args.speedups)
        if speedups.active("data") and shares == 1:  # shared streams also skip rows on the CPU
            from nexttransformer.runtime.speedups.prefetch import PrefetchedStream
            stream = PrefetchedStream(stream)
    model = build_model(config, device, seed=p.seed)
    model.set_sparse_backend(args.sparse_backend)
    # The trainable set (Ouro's gate-only stage) fixes the optimizer groups and DDP reducer.
    start_step = json.loads((resume/"complete.json").read_text())["step"] if resume is not None else 0
    model.set_training_stage(start_step)

    def wrap():
        return DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                   broadcast_buffers=False) if world > 1 else model

    wrapped = wrap()
    optimizer = build_optimizer(model, train_config)
    software = software_identity(config, args.sparse_backend)
    if args.speedups: software.update(speedups=args.speedups)
    state = dict(step=0, input_tokens=0, next_global_sequence=0, data_manifest_sha256=manifest_hash,
                 budget=BudgetState().state_dict(), theory_flops=0, training_seconds=0., evaluation_seconds=0.,
                 evaluation_forward_flops=0, smoke=smoke, mode=mode, smoke_val_steps=[], software=software,
                 model_hash=config.fingerprint, micro_batch=args.micro_batch)
    if resume is not None:
        saved = load_checkpoint(resume, model, optimizer, stream, rank, world)
        if saved["model_config"] != asdict(config) or saved["train_config"] != asdict(train_config):
            raise ValueError("resume configuration differs from checkpoint")
        previous = saved["state"]
        for key in ("data_manifest_sha256", "model_hash", "smoke"):
            if previous[key] != state[key]: raise ValueError(f"resume {key} differs from checkpoint")
        if previous.get("mode", "formal") != mode:  # checkpoints before research mode are formal
            raise ValueError(f"the run was started in {previous.get('mode', 'formal')} mode; resume it in that mode")
        # "speedups" is compared in both directions: a segment may not drop the checkpoint's selection either.
        changed = sorted(k for k in set(software) | {"speedups"} if k not in ("code_snapshot", "git_commit")
                         and previous["software"].get(k) != software.get(k))
        if changed: raise ValueError(f"resume software differs from checkpoint: {changed}")
        state, state["software"] = previous, software
        state["mode"] = mode
        if stream.rows*stream_world*config.sequence_len != state["input_tokens"]:
            raise ValueError("checkpoint token count differs from the exact data cursor")
        del saved
    # Formal and full-batch smoke updates go through BudgetState; evaluation
    # events are recorded there only for formal runs (smoke uses its own list).
    budget = BudgetState.from_state_dict(state["budget"])
    if state["micro_batch"] != args.micro_batch:
        state.setdefault("micro_batch_history", []).append([state["step"], args.micro_batch])
        state["micro_batch"] = args.micro_batch

    if rank == 0:
        if resume is not None: rewind_events(args.output, state["step"])
        name = "provenance.json" if resume is None else f"resume_{state['step']:08d}.json"
        atomic_json(args.output/name, dict(
            config=str(args.config), model=asdict(config), model_hash=config.fingerprint, protocol_hash=p.fingerprint,
            protocol=asdict(p), train=asdict(train_config), cost=dict(
                formula_version=proof["formula_version"], flops_per_token=proof["flops_per_token"],
                dense_flops_per_token=dense_per_token, relative_error=proof["flops_per_token"]/dense_per_token-1,
                total_training_flops=proof["total_training_flops"]),
            software=software, command=sys.argv, host=socket.gethostname(), world_size=world,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            compute_dtype=str(COMPUTE_DTYPE), micro_batch=args.micro_batch, accumulation=accumulation,
            eval_batch=args.eval_batch, data_manifest_sha256=manifest_hash, tokenizer_sha256=record["tokenizer_sha256"],
            parameters=sum(t.numel() for t in model.parameters()), smoke=smoke,
            admission=dict(mode=mode, findings=findings, method_source_hash=source_hash),
            resumed_from=str(resume) if resume else None,
            wandb_mode=args.wandb_mode, wandb_note=wandb_note, job_id=os.environ.get("NEXTTRANSFORMER_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
            kernel_caches={key: os.environ.get(key) for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")}))

    peak = peak_flops(device)
    rows = p.global_batch_tokens//config.sequence_len
    started = time.monotonic()
    segment_end = False
    with start_tracker(args, rank, config, train_config, state, software["reference_source_hash"], device) as tracker:
        model.train()

        def event(payload):
            if rank == 0:
                append_event(args.output, payload); tracker.log(payload); print(json.dumps(payload), flush=True)

        def validation(step, tokens=None, split="val"):
            result = evaluate_bpb(model, tokenizer, data_dir, args.resources, batch=args.eval_batch, rank=rank,
                                  world=world, device=device, tokens=tokens or val_tokens, split=split, smoke=smoke)
            _, forward = stage_cost(proof, config, min(step, p.steps-1))
            state["evaluation_seconds"] += result["seconds"]
            # Training-graph forward cost per token: an upper bound for the eval forward.
            state["evaluation_forward_flops"] += forward*result["input_tokens"]//p.global_batch_tokens
            return result

        def core(step):
            begin = time.monotonic()
            model.set_sparse_backend("reference")
            try:
                result = evaluate_core(model, tokenizer, args.resources, device,
                                       smoke_max_per_task=args.smoke_core_per_task if smoke else None)
            finally:
                model.set_sparse_backend(args.sparse_backend)
            seconds = time.monotonic()-begin
            state["evaluation_seconds"] += seconds
            return dict(result, seconds=seconds)

        def scheduled_evaluations(step):
            if smoke: due = dict(val=step in (0, stop), core=step == stop and args.smoke_core_per_task != 0, final=False)
            else: due = events_at(step)
            done = state["smoke_val_steps"] if smoke else budget.val_steps
            if due["val"] and step not in done:
                result = validation(step)
                if smoke: done.append(step)
                else: budget.record_eval("val")
                payload = dict(event="validation", step=step, val_bpb=result["bpb"], **result)
                if rank == 0: atomic_json(args.output/f"val_{step:08d}.json", payload)
                event(payload)
            if due["core"] and not due["final"] and step not in budget.core_steps:
                result = core(step)
                if not smoke: budget.record_eval("core")
                payload = dict(event="core", step=step, **result)
                if rank == 0: atomic_json(args.output/f"core_{step:08d}.json", payload)
                event(payload)

        def write_budget():
            if rank == 0:
                atomic_json(args.output/"budget.json", dict(step=state["step"], **budget.state_dict(),
                            charged_theory_flops=state["theory_flops"], dense_flops_per_token=dense_per_token,
                            theory_relative_error=state["theory_flops"]/max(state["input_tokens"], 1)/dense_per_token-1))

        def checkpoint():
            state["budget"] = budget.state_dict()
            save_checkpoint(args.output, model, optimizer, stream, config, train_config, state, rank, world)
            if rank == 0: prune_rolling(args.output, args.rolling_keep, p.steps, train_config.checkpoint_every)
            write_budget()
            if world > 1: dist.barrier()

        if state["step"] == 0: scheduled_evaluations(0)
        def warm_up():
            # DDP rebuilds its gradient buckets after a process's first backward. One discarded
            # backward on a fixed dummy batch makes every update of every segment all-reduce with
            # the same rebuilt layout: on 4+ nodes NCCL's Ring all-reduce rounds by bucket layout,
            # so a resumed segment would otherwise differ from a continuous run.
            # No data, optimizer, budget or RNG state changes.
            if world == 1: return
            rngs, routing = rng_state(), model.routing_rng_state()
            dummy = torch.zeros(args.micro_batch, config.sequence_len, dtype=torch.long, device=device)
            wrapped(dummy, dummy).backward()
            optimizer.zero_grad(set_to_none=True)
            restore_rng(rngs); model.restore_routing_rng(routing)

        warm_up()
        gate_built = config.method == "ouro" and state["step"] >= gate_boundary(config)
        window = []
        for step in range(state["step"], stop):
            if config.method == "ouro" and not gate_built and step >= gate_boundary(config):
                # Gate-only stage: rebuild the optimizer (keeping the gate's moments) and the reducer.
                optimizer, changed = transition_stage(model, optimizer, step, train_config)
                if changed and world > 1:
                    del wrapped
                    wrapped = wrap(); warm_up()
                gate_built = True
                event(dict(event="stage_transition", step=step, stage="gate_only"))
            model.set_step(step)
            synchronize(device); begin = time.perf_counter(); data_seconds = 0.
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            lr = set_schedule(optimizer, step, train_config)
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            batches = share_batches(stream, args.micro_batch, stream_rows, rank % shares, shares, device)
            for micro in range(accumulation):
                t = time.perf_counter(); x, y = next(batches); data_seconds += time.perf_counter()-t
                model.set_row_offset(step*rows+rank*(rows//world)+micro*args.micro_batch)
                with wrapped.no_sync() if world > 1 and micro < accumulation-1 else nullcontext():
                    loss = wrapped(x, y)
                    (loss/accumulation).backward()
                # Dense delegates to the native forward and leaves these unset.
                lm = model.last_lm_loss if model.last_lm_loss is not None else loss
                aux = model.last_aux_loss if model.last_aux_loss is not None else torch.zeros((), device=device)
                totals += torch.stack((lm.detach().double()/(accumulation*world), aux.detach().double()/(accumulation*world),
                                       x.new_full((), x.numel(), dtype=torch.float64)))  # filled on the device: a copy from the host would drain the queue
            # The rows of this stream that the other share keeps; afterwards every copy has the 8-rank state.
            t = time.perf_counter()
            if next(batches, None) is not None: raise RuntimeError("micro batches do not cover the stream's rows")
            data_seconds += time.perf_counter()-t
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            synchronize(device); optimizer_start = time.perf_counter()
            optimizer.step()
            synchronize(device); optimizer_seconds = time.perf_counter()-optimizer_start
            lm, aux, tokens = reduce_sum(totals).tolist()
            if int(tokens) != step_tokens: raise RuntimeError("training input-token budget drift")
            cost, _ = stage_cost(proof, config, step)
            if full_batch: budget.advance(step_tokens, cost)
            duration = time.perf_counter()-begin
            state.update(step=step+1, input_tokens=state["input_tokens"]+step_tokens,
                         next_global_sequence=(state["input_tokens"]+step_tokens)//config.sequence_len,
                         theory_flops=state["theory_flops"]+int(cost)*step_tokens//p.global_batch_tokens,
                         training_seconds=state["training_seconds"]+duration)
            window = (window+[cost])[-p.val_every:]
            rolling = sum(window)/(len(window)*p.global_batch_tokens*dense_per_token)-1
            if len(window) == p.val_every and abs(rolling) > p.routing_warning:
                event(dict(event="budget_anomaly", step=step+1, rolling_relative_error=rolling))
            if (step+1) % args.log_every == 0 or step == 0 or step+1 == stop:
                event(dict(event="train", step=step+1, input_tokens=state["input_tokens"], loss=lm+aux, lm_loss=lm,
                           aux_loss=aux, grad_norm=float(grad_norm), lr_multiplier=lr, theory_flops=state["theory_flops"],
                           theory_relative_error=state["theory_flops"]/state["input_tokens"]/dense_per_token-1,
                           step_seconds=duration, data_seconds=data_seconds, optimizer_seconds=optimizer_seconds,
                           tokens_per_second=tokens/duration, gpu_hours=world*state["training_seconds"]/3600,
                           mfu_theory=cost*step_tokens/p.global_batch_tokens/(duration*world*peak) if peak else None,
                           peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))
            scheduled_evaluations(step+1)
            protocol = (step+1) % train_config.checkpoint_every == 0 or step+1 == stop
            segment_end = bool(args.max_wall_hours) and (time.monotonic()-started)/3600 >= args.max_wall_hours \
                and step+1 < stop
            if protocol or (step+1) % args.rolling_every == 0 or segment_end: checkpoint()
            if segment_end: break

        complete = state["step"] == p.steps and not smoke
        if complete:
            # The final checkpoint precedes these evaluations; a resumed segment
            # re-runs any that did not finish, without repeating training.
            final = {split: validation(p.steps, p.final_bpb_tokens, split) for split in ("train", "val")}
            if rank == 0: atomic_json(args.output/"bpb_final.json", final)
            event(dict(event="final_bpb", step=p.steps, train_bpb=final["train"]["bpb"], val_bpb=final["val"]["bpb"]))
            if p.steps not in budget.core_steps:
                result = core(p.steps); budget.record_eval("core")
                if rank == 0: atomic_json(args.output/"core_final.json", dict(step=p.steps, **result))
                event(dict(event="core", step=p.steps, **result))
            write_budget()
        if rank == 0:
            status = "segment_complete" if segment_end else "smoke_complete" if smoke else \
                     "training_complete" if complete else "stopped"
            atomic_json(args.output/"run_status.json", dict(
                phase=status, mode=mode, step=state["step"], budget=budget.state_dict(),
                final_status="smoke" if smoke else budget.final_status(dense_per_token),
                training_seconds=state["training_seconds"], evaluation_seconds=state["evaluation_seconds"],
                evaluation_forward_flops=state["evaluation_forward_flops"], wandb_mode=args.wandb_mode,
                wandb_note=wandb_note, latest=json.loads((args.output/"latest.json").read_text())
                if (args.output/"latest.json").exists() else None))
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
