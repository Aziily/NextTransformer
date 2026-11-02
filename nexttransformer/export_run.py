"""Export a finished run directory as a results/ record, or check every record.

    python -m nexttransformer.audit_run RUN_DIR --method gdn
    python -m nexttransformer.export_run RUN_DIR --method gdn --name "Gated DeltaNet · GDN" \\
        --audit evidence/audits/gdn.json
    python -m nexttransformer.export_run --check

Export reads only the run directory (metrics.jsonl, provenance, status, CORE files). It
refuses a run unless every scheduled validation and full 22-task CORE event is present, the
per-task CORE scores reproduce the logged aggregate, the trainer's cumulative audit is
`matched`, every segment was trained with `--formal`, and the run used the pinned tokenizer and
the frozen protocol.

`--accept-baseline` records the maintainer's acceptance of a finished `dense_full` run as the
Dense baseline that every relative comparison uses. It needs the run's cumulative audit, and
`--check` allows one accepted baseline whose acceptance matches the run's provenance.
Existing records are immutable: an identical export is a no-op; a different run needs a new --file.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from nexttransformer.protocol.spec import Protocol, TOKENIZER_SHA256

REPO = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Interactive-Active-AI/NextTransformer"
GITHUB = REPO_URL+"/blob/main/"
RESULTS = REPO/"results"
P = Protocol()
VAL_STEPS, CORE_STEPS = list(P.validation_steps), list(P.core_steps)
# The historical SSSL Dense keeps method id "dense"; its model card is "sssl" (see the protocol documentation).
CARD_ALIASES = {"dense": "sssl"}
BASELINE = "dense_full"
REQUIRED = ("id", "methodId", "cohort", "name", "status", "step", "val", "core", "source", "protocol", "url")
FROZEN_COHORT = "frozen-replay-20261019"


def core_baselines():
    return {t["id"]: t["randomBaseline"] for t in json.loads((REPO/"website/data/core-tasks.json").read_text())["tasks"]}


def read_run(path):
    run = Path(path); val, core, final, last = {}, {}, None, None
    for line in (run/"metrics.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "validation": val[e["step"]] = e["val_bpb"]
        elif e["event"] == "core":
            core[e["step"]] = dict(core_metric=e["core_metric"], results=e.get("results"), centered=e.get("centered_results"),
                                   full_core=e.get("full_core"))
        elif e["event"] == "final_bpb": final = {k: v for k, v in e.items() if k != "event"}
        elif e["event"] == "train": last = e
    segments = ["provenance.json", *sorted(f.name for f in run.glob("resume_*.json"))]
    provenance = [json.loads((run/f).read_text()) for f in segments]
    return dict(run_dir=run.name, val=sorted(val.items()), core=sorted(core.items()), final_bpb=final, last=last,
                status=json.loads((run/"run_status.json").read_text()).get("final_status"),
                core_final_mtime=os.path.getmtime(run/"core_final.json"), provenance=provenance)


def check_metric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite numeric metric")


def check_core(name, step, raw, centered, aggregate, tasks):
    check_metric(aggregate, f"{name}: CORE aggregate at {step}")
    if not isinstance(raw, dict) or not isinstance(centered, dict):
        raise ValueError(f"{name}: CORE at {step} needs raw and centered task scores")
    if set(raw) != set(tasks) or set(centered) != set(tasks):
        raise ValueError(f"{name}: CORE at {step} is not the full {len(tasks)}-task evaluation")
    for task, base in tasks.items():
        check_metric(raw[task], f"{name}: {task} raw score at {step}")
        check_metric(centered[task], f"{name}: {task} centered score at {step}")
        if abs((raw[task]-base)/(1-base)-centered[task]) > 1e-7: raise ValueError(f"{name}: {task} centering at {step}")
    if abs(sum(centered.values())/len(tasks)-aggregate) > 1e-7: raise ValueError(f"{name}: CORE aggregate at {step}")


def record(raw, method, name, audit=None, commit=None, cohort="reference-v1", accept_baseline=False, file=None,
           note=None):
    tasks = core_baselines(); p = raw["provenance"][0]; last = raw["last"]
    val = [dict(step=int(s), value=v) for s, v in raw["val"]]
    if [x["step"] for x in val] != VAL_STEPS: raise ValueError(f"{method}: incomplete validation schedule")
    for point in val:
        check_metric(point["value"], f"{method}: validation BPB at {point['step']}")
    check_metric(raw["final_bpb"]["val_bpb"], f"{method}: final BPB")
    core, evaluations = [], []
    for step, c in raw["core"]:
        if not c["full_core"]: raise ValueError(f"{method}: CORE at {step} is a smoke evaluation")
        check_core(method, step, c["results"], c["centered"], c["core_metric"], tasks)
        core.append(dict(step=int(step), value=c["core_metric"]))
        evaluations.append(dict(step=int(step), raw=c["results"], centered=c["centered"]))
    if [x["step"] for x in core] != CORE_STEPS: raise ValueError(f"{method}: incomplete CORE schedule")
    if raw["status"] != "matched" or last["step"] != P.steps or last["input_tokens"] != P.total_tokens:
        raise ValueError(f"{method}: run is not complete with a matched cumulative audit")
    if any(segment.get("admission", {}).get("mode", "formal") != "formal" for segment in raw["provenance"]):
        raise ValueError(f"{method}: a segment was trained in research mode; results need --formal in every segment")
    if p.get("tokenizer_sha256") != TOKENIZER_SHA256: raise ValueError(f"{method}: tokenizer differs from the pinned tokenizer")
    if p.get("protocol_hash") != P.fingerprint: raise ValueError(f"{method}: protocol hash differs from the frozen protocol")
    commit = commit or p["software"].get("git_commit")
    if not commit: raise ValueError(f"{method}: pass --commit with the source commit of the training code")
    if accept_baseline and (method != BASELINE or not audit or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError(f"{method}: only a {BASELINE} run with its cumulative audit and full source commit can be the Dense baseline")
    path = RESULTS/"runs"/f"{file or CARD_ALIASES.get(method, method)}.json"
    published = path.relative_to(REPO).as_posix()
    run = dict(id=file or CARD_ALIASES.get(method, method), methodId=method, cohort=cohort, name=name, status="finished", step=P.steps, val=val, core=core,
               url=GITHUB+(audit or published),
               source="Run directory · cumulative audit", protocol=P.version,
               proof=dict(protocol=P.version, protocol_hash=p["protocol_hash"], src_head=commit,
                          science_hash=p["software"]["science_hash"], model_hash=p["model_hash"], tokenizer_sha256=p["tokenizer_sha256"],
                          data_manifest_sha256=p["data_manifest_sha256"], seed=P.seed, total_steps=P.steps,
                          total_input_tokens=P.total_tokens, global_batch_tokens=P.global_batch_tokens,
                          cost_proof_gflops_per_token=p["cost"]["flops_per_token"]/1e9,
                          cost_proof_vs_dense_pct=p["cost"]["relative_error"]*100,
                          native_attention=p["software"].get("native_attention"), sparse_backend=p["software"].get("sparse_backend"),
                          code_snapshot=p["software"].get("code_snapshot")),
               reportedAudit=raw["status"], lastRecordedAt=raw["core_final_mtime"], finalBpb=raw["final_bpb"]["val_bpb"],
               taskEvaluations=evaluations, complete=True,
               trainingCompute=dict(averageGflopsPerInputToken=last["theory_flops"]/P.total_tokens/1e9,
                                    trainingFlopsEstimate=last["theory_flops"], inputTokens=P.total_tokens, step=P.steps, final=True,
                                    accounting="reference-aten-math-v2",
                                    source="Cumulative audit · "+(audit or published)))
    if accept_baseline:
        run["baselineAcceptance"] = dict(status="accepted", planVersion="v2", protocolHash=p["protocol_hash"], sourceCommit=commit,
                                         reportUrl=GITHUB+audit)
    if note: run["comparisonNote"] = note
    return run, path


def check(root=RESULTS):
    """Validate every record under results/: fields, unique ids, model cards, and complete formal runs."""
    tasks = core_baselines(); problems = []; seen = set(); accepted = []
    cards = {json.loads(p.read_text())["id"] for p in (root/"methods").glob("*.json")}
    problems += [f"{p.relative_to(root).as_posix()}: results/runs/ is flat, one file per run" for p in (root/"runs").iterdir() if p.is_dir()]
    for path in sorted((root/"runs").glob("*.json")):
        r = json.loads(path.read_text()); name = path.relative_to(root).as_posix()
        missing = [k for k in REQUIRED if k not in r]
        if missing: problems.append(f"{name}: missing {missing}"); continue
        if r["id"] in seen: problems.append(f"{name}: duplicate run id {r['id']}")
        seen.add(r["id"])
        if CARD_ALIASES.get(r["methodId"], r["methodId"]) not in cards: problems.append(f"{name}: no model card results/methods/{r['methodId']}.json")
        if any(k in r for k in ("entity", "project", "trackingState")): problems.append(f"{name}: tracking-service fields")
        proof = r.get("proof") or {}
        if proof.get("tokenizer_sha256") not in (None, TOKENIZER_SHA256): problems.append(f"{name}: tokenizer differs from the pinned one")
        try:
            for kind in ("val", "core"):
                for point in r[kind]:
                    check_metric(point["value"], f"{name}: {kind} at {point['step']}")
            if "finalBpb" in r:
                check_metric(r["finalBpb"], f"{name}: final BPB")
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"{name}: invalid metrics: {error}")
            continue
        if r["status"] == "finished" and r["cohort"] in ("reference-v1", FROZEN_COHORT):
            if [x["step"] for x in r["val"]] != VAL_STEPS or [x["step"] for x in r["core"]] != CORE_STEPS:
                problems.append(f"{name}: finished run without the complete evaluation schedule")
            evaluations = r.get("taskEvaluations")
            try:
                if not isinstance(evaluations, list) or [e["step"] for e in evaluations] != CORE_STEPS:
                    raise ValueError("taskEvaluations must cover every scheduled CORE step exactly once, in order")
                aggregates = {point["step"]: point["value"] for point in r["core"]}
                for e in evaluations:
                    check_core(name, e["step"], e["raw"], e["centered"], aggregates[e["step"]], tasks)
            except (ValueError, KeyError, TypeError) as error:
                problems.append(f"{name}: invalid CORE evidence: {error}")
        if r["cohort"] == FROZEN_COHORT:
            from nexttransformer.export_frozen import check_audit
            audit_path = f"evidence/audits/{r['id']}.json"
            try:
                audit_bytes = (root.parent/audit_path).read_bytes()
                check_audit(r, json.loads(audit_bytes))
                if r.get("auditSha256") != hashlib.sha256(audit_bytes).hexdigest():
                    problems.append(f"{name}: cumulative audit hash differs")
            except (ValueError, KeyError, OSError, TypeError) as error:
                problems.append(f"{name}: frozen audit: {error}")
        acceptance = r.get("baselineAcceptance")
        if acceptance:
            accepted.append(name)
            if r["methodId"] != BASELINE or r["status"] != "finished" or not r.get("complete") or r.get("reportedAudit") != "matched":
                problems.append(f"{name}: baseline acceptance needs a complete, matched {BASELINE} run")
            if (acceptance.get("status"), acceptance.get("planVersion")) != ("accepted", "v2") \
                    or acceptance.get("protocolHash") != proof.get("protocol_hash") or acceptance.get("sourceCommit") != proof.get("src_head") \
                    or not str(acceptance.get("reportUrl", "")).startswith("https://"):
                problems.append(f"{name}: baseline acceptance differs from the run's provenance")
    if len(accepted) > 1: problems.append(f"more than one accepted Dense baseline: {accepted}")
    return problems


def ledger_mark(accounting):
    """README footnote for compute charged by a ledger other than the operator-level one; marks are not comparable."""
    if accounting.startswith("Historical matrix FLOPs"): return "¹"
    if accounting.endswith("all-backends-replay"): return "²"
    return ""


def summary(root=RESULTS):
    """Markdown table of the finished runs, for the README, with differences from the accepted Dense baseline."""
    cards = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in (root/"methods").glob("*.json")}
    records = [(path, json.loads(path.read_text())) for path in sorted((root/"runs").glob("*.json"))]
    base = next((r for _, r in records if r.get("baselineAcceptance")), None)
    rows = []
    for path, r in records:
        if r["status"] != "finished": continue
        card = cards[CARD_ALIASES.get(r["methodId"], r["methodId"])]
        bpb = r["val"][-1]["value"]  # periodic validation at the final update, as ranked on the site
        c = r.get("trainingCompute") or {}
        rows.append((r is not base, -r["core"][-1]["value"], card["name"], card["family"], bpb, r["core"][-1]["value"],
                     c.get("averageGflopsPerInputToken"), ledger_mark(c.get("accounting", "")),
                     path.relative_to(root.parent).as_posix()))
    head, rule = ["Model", "Family", "Val BPB ↓", "CORE ↑", "Training GFLOPs / token"], ["---", "---", "---:", "---:", "---:"]
    if base:
        head[3:3], rule[3:3] = ["Δ BPB"], ["---:"]
        head[5:5], rule[5:5] = ["Δ CORE"], ["---:"]
    signed = lambda value, digits: f"{value:+.{digits}f}".replace("-0."+"0"*digits, "±0."+"0"*digits).replace("+0."+"0"*digits, "±0."+"0"*digits)
    out = ["| "+" | ".join(head)+" |", "|"+"|".join(rule)+"|"]
    for other, _, name, family, bpb, core, flops, mark, record in sorted(rows):
        cells = [f"[{name}]({record})" if other else f"**[{name}]({record})**", family, f"{bpb:.4f}", f"{100*core:.2f}"]
        if base:
            cells.insert(3, signed(bpb-base["val"][-1]["value"], 4) if other else "—")
            cells.insert(5, signed(100*(core-base["core"][-1]["value"]), 2) if other else "—")
        out.append("| "+" | ".join(cells+[f"{flops:.3f}{mark}" if flops else "—"])+" |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--check", action="store_true", help="validate every record under results/ and exit")
    ap.add_argument("--summary", action="store_true", help="print the README table of finished runs and exit")
    ap.add_argument("--method"); ap.add_argument("--name")
    ap.add_argument("--audit", help="repository path of the run's cumulative audit, e.g. evidence/audits/<method>.json")
    ap.add_argument("--commit", help="source commit of the training code (default: provenance git_commit)")
    ap.add_argument("--cohort", default="reference-v1")
    ap.add_argument("--file", help="file name under results/runs/ without .json (default: the model card id)")
    ap.add_argument("--note", help="provenance note shown with the run, e.g. that it was trained from a frozen branch")
    ap.add_argument("--accept-baseline", action="store_true",
                    help=f"record the maintainer's acceptance of this {BASELINE} run as the Dense baseline (needs --audit)")
    args = ap.parse_args(argv)
    if args.summary:
        print(summary()); return
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else "results/: all records valid")
        raise SystemExit(1 if problems else 0)
    if not (args.run_dir and args.method and args.name):
        ap.error("RUN_DIR, --method and --name are required to export a run")
    run, path = record(read_run(args.run_dir), args.method, args.name, args.audit, args.commit, args.cohort,
                       args.accept_baseline, args.file, args.note)
    payload = json.dumps(run, indent=1, ensure_ascii=False, allow_nan=False)+"\n"
    try:
        with path.open("x") as out:
            out.write(payload)
    except FileExistsError:
        if json.loads(path.read_text()) == run:
            print(f"unchanged {path.relative_to(REPO)}")
            return
        raise SystemExit(f"{path.relative_to(REPO)} already exists; pass a new --file to publish another run")
    print(f"wrote {path.relative_to(REPO)}: {len(run['val'])} validations, {len(run['core'])} CORE events")


if __name__ == "__main__":
    main()
