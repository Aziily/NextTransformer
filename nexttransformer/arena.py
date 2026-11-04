"""Auto-research arena: disclosure cards, and the files an arena or method pull request may change.

    python -m nexttransformer.arena --check
    python -m nexttransformer.arena --scope origin/main

`--check` validates every card under results/arena/ against the arena rules (docs/src/content/docs/arena/rules.md):
required disclosures, status, involvement
level, selection signal, and the model card, run records, and admitted configuration the entry names. `--scope BASE`
classifies the files changed since BASE when a card is among them or a method plugin is added under
nexttransformer/methods/, and fails when such a pull request changes the training contract, the evaluation, another
method's configuration, or evidence it does not own. Neither imports torch.
"""
import argparse
import json
import re
import subprocess

from nexttransformer.export_run import CARD_ALIASES, REPO, RESULTS
from nexttransformer.models.registry import FAMILY
from nexttransformer.protocol.spec import config_path

VERSION = "arena-v1"
TRAINING_PROTOCOL = "nextt-reference-50b-v1"
# The last training shard in sorted order; a formal 49.152B stream reads from the first shard and never reaches it.
DEV_SHARD = "shard_06541.parquet"
LEVELS = ("L0", "L1", "L2", "L3")
STATUSES = ("admitted", "complete", "legacy", "retracted")
VERIFICATION = ("self-reported", "reproduced", "maintainer")
SIGNALS = ("train-loss", "dev-bpb", "validation-bpb", "core", "other")
RESERVED_SIGNALS = ("validation-bpb", "core")
FIELDS = {"card": ("id", "name", "methodId", "runIds", "arenaProtocol", "status", "statusNote", "submitted",
                   "verification", "researcher", "research", "novelty", "artifacts"),
          "researcher": ("people", "llms", "framework", "involvement", "involvementNote"),
          "research": ("start", "end", "baseCommit", "candidatesTrained", "formalAttempts", "formalNote", "gpuHours",
                       "hardware", "computeNote", "internet", "selectionSignal"),
          "novelty": ("closestPrior", "statement"),
          "artifacts": ("checkpointUrl", "logsUrl", "traceUrl", "historyUrl", "codeUrl")}
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMMIT = re.compile(r"[0-9a-f]{7,40}")

DOCS = "docs/src/content/docs/"
# Scope of an arena pull request (arena rules, "What an entry may change"). The first matching prefix decides.
BLOCKED = ("nexttransformer/models/backbone.py", "nexttransformer/protocol/", "nexttransformer/evaluation/",
           "nexttransformer/runtime/", "nexttransformer/training/", "nexttransformer/checks/", "nexttransformer/native/",
           "nexttransformer/__init__.py", "nexttransformer/arena.py", "nexttransformer/export_run.py",
           "nexttransformer/train.py", "nexttransformer/evaluate.py", "nexttransformer/prepare.py",
           "nexttransformer/generate.py", "nexttransformer/audit_run.py", "nanochat/", "launch/", "requirements/", ".github/",
           "website/", DOCS+"protocol/", DOCS+"zh-cn/protocol/", DOCS+"arena/rules.md", DOCS+"zh-cn/arena/rules.md",
           "AGENTS.md", "CONTRIBUTING.md", "pyproject.toml", "uv.lock")
FROZEN = ("results/runs/", "evidence/", DOCS+"results/runs/", DOCS+"zh-cn/results/runs/")
DOCUMENTS = (DOCS+"methods/sources.md", DOCS+"zh-cn/methods/sources.md")


def check_card(card, name):
    """Problems a card has on its own: structure, vocabulary, and disclosure rules."""
    problems = []
    for section, keys in FIELDS.items():
        part = card if section == "card" else card.get(section)
        if not isinstance(part, dict): problems.append(f"{name}: missing {section}"); continue
        missing = [k for k in keys if k not in part]
        if missing: problems.append(f"{name}: {section} missing {missing}")
    if problems: return problems
    who, how, status = card["researcher"], card["research"], card["status"]
    signals = how["selectionSignal"] if isinstance(how["selectionSignal"], list) else []
    final = status in ("admitted", "complete")
    count = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0
    links = [v for v in card["artifacts"].values() if v is not None]
    rules = [
        (card["arenaProtocol"] == VERSION, f"arenaProtocol is not {VERSION}"),
        (status in STATUSES, f"status {status!r} is not one of {STATUSES}"),
        (final or bool(card["statusNote"]), "a legacy or retracted entry needs statusNote"),
        (card["verification"] in VERIFICATION, f"verification is not one of {VERIFICATION}"),
        (isinstance(card["runIds"], list), "runIds is not a list"),
        (who["involvement"] in LEVELS, f"involvement is not one of {LEVELS}"),
        (bool(who["involvementNote"]), "involvementNote is empty"),
        (isinstance(who["people"], list) and all(p.get("name") and p.get("role") for p in who["people"]),
         "every person needs a name and a role"),
        (isinstance(who["llms"], list) and bool(who["llms"]) and all(m.get("name") and "version" in m and m.get("role")
                                                                     for m in who["llms"]),
         "researcher.llms needs at least one LLM with name, version, and role"),
        (isinstance(who["framework"], dict) and bool(who["framework"].get("name")),
         "researcher.framework needs a name ('none' if none)"),
        (all(DATE.fullmatch(str(d)) for d in (card["submitted"], how["start"], how["end"])), "dates are not YYYY-MM-DD"),
        (bool(COMMIT.fullmatch(str(how["baseCommit"]))), "baseCommit is not a commit hash"),
        (all(count(how[k]) for k in ("candidatesTrained", "formalAttempts")),
         "candidatesTrained and formalAttempts must be counts"),
        (how["gpuHours"] is None or (isinstance(how["gpuHours"], (int, float)) and how["gpuHours"] >= 0),
         "gpuHours must be a number or null"),
        (isinstance(how["internet"], bool), "research.internet must be true or false"),
        (bool(signals) and set(signals) <= set(SIGNALS), f"selectionSignal must list values from {SIGNALS}"),
        (not final or not set(signals) & set(RESERVED_SIGNALS),
         "validation BPB and CORE are reserved for the formal run and cannot select candidates"),
        (isinstance(card["novelty"]["closestPrior"], list) and bool(card["novelty"]["statement"]),
         "novelty needs closestPrior and a statement"),
        (all(str(v).startswith("https://") for v in links), "artifact links must be HTTPS"),
    ]
    if status == "complete":
        rules += [(bool(card["runIds"]), "a complete entry names its run"),
                  (bool(card["artifacts"]["checkpointUrl"] and card["artifacts"]["logsUrl"]),
                   "a complete entry links its final checkpoint and training logs")]
    return [f"{name}: {message}" for ok, message in rules if not ok]


def check(root=RESULTS):
    """Validate every card under results/arena/ and the records each one names."""
    problems, methods = [], set()
    cards = {json.loads(p.read_text())["id"] for p in (root/"methods").glob("*.json")}
    runs = {r["id"]: r for r in (json.loads(p.read_text()) for p in (root/"runs").glob("*.json"))}
    for path in sorted((root/"arena").glob("*.json")):
        name = path.relative_to(root).as_posix()
        try: card = json.loads(path.read_text())
        except json.JSONDecodeError as error: problems.append(f"{name}: {error}"); continue
        if card.get("id") != path.stem: problems.append(f"{name}: id differs from the file name")
        if card.get("methodId") in methods: problems.append(f"{name}: another card already enters {card['methodId']}")
        methods.add(card.get("methodId"))
        own = check_card(card, name)
        problems += own
        if own: continue
        method, status = card["methodId"], card["status"]
        if CARD_ALIASES.get(method, method) not in cards: problems.append(f"{name}: no model card results/methods/{method}.json")
        named = []
        for run_id in card["runIds"]:
            run = runs.get(run_id)
            if run is None: problems.append(f"{name}: no run record with id {run_id}"); continue
            if run["methodId"] != method: problems.append(f"{name}: run {run_id} belongs to {run['methodId']}")
            named.append(run)
        if status not in ("admitted", "complete"): continue
        if method not in FAMILY: problems.append(f"{name}: {method} is not registered in nexttransformer/models/registry.py"); continue
        path = config_path(method)
        if not path.exists(): problems.append(f"{name}: no admitted configuration {path.relative_to(REPO)}"); continue
        model_hash = json.loads(path.read_text()).get("model_hash")
        for run in named if status == "complete" else ():
            proof = run.get("proof") or {}
            if (run.get("protocol"), run.get("status"), run.get("complete"), run.get("reportedAudit")) != \
                    (TRAINING_PROTOCOL, "finished", True, "matched"):
                problems.append(f"{name}: run {run['id']} is not a complete, matched {TRAINING_PROTOCOL} run")
            if proof.get("model_hash") != model_hash:
                problems.append(f"{name}: run {run['id']} was not trained from the admitted configuration")
    return problems


def is_disclosure_card(path):
    """Match the same results/arena/*.json files that check() validates."""
    return re.fullmatch(r"results/arena/[^/]+\.json", path) is not None


def classify(changes, base_text=lambda path: None):
    """Verdicts for `git diff --name-status` rows: ok, review, or blocked, each with a reason.

    `base_text(path)` returns a changed file as it is at the base, to tell a calibration refresh of an
    existing configuration from a change of its model.
    """
    rows = []
    for status, path in changes:
        kind = status[0]
        if path.startswith(BLOCKED): verdict, reason = "blocked", "fixed by the training contract or the arena"
        elif is_disclosure_card(path):
            verdict, reason = ("blocked", "cards are never deleted") if kind == "D" else ("ok", "disclosure card")
        elif path.startswith(FROZEN):
            verdict, reason = ("ok", "new record or evidence") if kind == "A" else ("blocked", "published records and evidence are frozen")
        elif path.startswith("results/methods/"):
            verdict, reason = {"A": ("ok", "new model card"), "D": ("blocked", "model cards are never deleted")}.get(
                kind, ("review", "changes an existing model card"))
        elif path.startswith("configs/"):
            if kind == "A": verdict, reason = "ok", "new configuration"
            elif kind == "D": verdict, reason = "blocked", "configurations are never deleted"
            else:
                before, after = json.loads(base_text(path) or "{}"), json.loads((REPO/path).read_text())
                same = all(before.get(k) == after.get(k) for k in ("model", "model_hash", "protocol_hash"))
                verdict, reason = ("ok", "calibration refresh; model unchanged") if same else ("blocked", "changes another method's model")
        elif path.startswith("nexttransformer/methods/"):
            verdict, reason = {"A": ("ok", "new plugin file"), "D": ("blocked", "plugins are never deleted")}.get(
                kind, ("review", "changes an existing plugin"))
        elif path.startswith(("nexttransformer/models/", "tests/")):
            verdict, reason = ("ok", "new file") if kind == "A" else ("review", "changes shared code or tests")
        elif path in DOCUMENTS: verdict, reason = "ok", "documentation an entry updates"
        else: verdict, reason = "review", "outside the arena surface"
        rows.append((verdict, status, path, reason))
    return rows


def method_pull_request(changes):
    """Whether the changes add a method plugin (a new nexttransformer/methods/<name>/method.json)."""
    return any(status[0] == "A" and path.startswith("nexttransformer/methods/") and path.endswith("/method.json")
               for status, path in changes)


def scope(base):
    """Classify the files changed since the merge base with BASE; None unless a card changed or a plugin is added."""
    git = lambda *args: subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout
    merge_base = git("merge-base", base, "HEAD").strip()
    changes = [tuple(line.split("\t", 1)) for line in git("diff", "--name-status", "--no-renames", merge_base, "HEAD").splitlines()]
    if not any(is_disclosure_card(path) for _, path in changes) and not method_pull_request(changes): return None
    return classify(changes, lambda path: git("show", f"{merge_base}:{path}"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="validate every card under results/arena/")
    action.add_argument("--scope", metavar="BASE", help="classify the files an arena or method pull request changes since BASE")
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else f"{len(list((RESULTS/'arena').glob('*.json')))} arena cards are valid")
        raise SystemExit(1 if problems else 0)
    rows = scope(args.scope)
    if rows is None: print("No disclosure card changed and no method plugin added; nothing to classify."); return
    for verdict, status, path, reason in rows: print(f"{verdict:8} {status:2} {path}  ({reason})")
    raise SystemExit(1 if any(row[0] == "blocked" for row in rows) else 0)


if __name__ == "__main__":
    main()
