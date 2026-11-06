"""Arena disclosure cards and the scope of arena pull requests (arena rules)."""
import json
import shutil
import subprocess

from nexttransformer import arena

from nexttransformer.arena import check, check_card, classify
from nexttransformer.export_run import REPO, RESULTS

TEMPLATE = REPO/"docs/public/arena/card.template.json"
DOCS = "docs/src/content/docs/"


def test_published_cards_and_the_template_are_valid():
    assert check() == []
    assert check_card(json.loads(TEMPLATE.read_text()), "template") == []


def copy_results(tmp_path):
    root = tmp_path/"results"
    shutil.copytree(RESULTS, root)
    return root


def test_card_rules(tmp_path):
    root = copy_results(tmp_path)
    path = root/"arena/circle-loop.json"
    card = json.loads(path.read_text())
    card["researcher"]["involvement"] = "L4"
    card["statusNote"] = None
    path.write_text(json.dumps(card))
    problems = check(root)
    assert any("involvement is not one of" in p for p in problems)
    assert any("needs statusNote" in p for p in problems)
    card = json.loads((RESULTS/"arena/circle-loop.json").read_text())
    card["status"] = "complete"
    path.write_text(json.dumps(card))
    problems = check(root)
    assert any("reserved for the formal run" in p for p in problems)
    assert any("final checkpoint and training logs" in p for p in problems)
    card["research"]["selectionSignal"] = ["dev-bpb"]
    card["artifacts"].update(checkpointUrl="https://example.org/checkpoint", logsUrl="https://example.org/logs")
    path.write_text(json.dumps(card))
    assert any("circle is not registered" in p for p in check(root))


def test_a_complete_entry_is_the_admitted_matched_run(tmp_path):
    root = copy_results(tmp_path)
    card = json.loads(TEMPLATE.read_text())
    card.update(id="gdn-entry", methodId="gdn", runIds=["gdn"], status="complete")
    card["artifacts"].update(checkpointUrl="https://example.org/checkpoint", logsUrl="https://example.org/logs")
    (root/"arena/gdn-entry.json").write_text(json.dumps(card))
    assert check(root) == []
    run = root/"runs/gdn.json"
    record = json.loads(run.read_text())
    record["proof"]["model_hash"] = "0"*64
    record["reportedAudit"] = "budget_mismatch"
    run.write_text(json.dumps(record))
    problems = check(root)
    assert any("not trained from the admitted configuration" in p for p in problems)
    assert any("not a complete, matched" in p for p in problems)
    card.update(id="gdn-again")
    (root/"arena/gdn-again.json").write_text(json.dumps(card))
    assert any("another card already enters gdn" in p for p in check(root))


def test_scope_keeps_the_contract_and_published_evidence_fixed():
    verdicts = {path: verdict for verdict, _, path, _ in classify([
        ("A", "nexttransformer/models/ring.py"), ("M", "nexttransformer/models/registry.py"),
        ("M", "nexttransformer/models/backbone.py"), ("M", "nexttransformer/evaluation/formal.py"),
        ("M", "nanochat/dataloader.py"), ("M", DOCS+"arena/rules.md"), ("A", "results/runs/ring.json"),
        ("M", "results/runs/nsa.json"), ("A", "evidence/costs/ring/ring.json"),
        ("A", "results/arena/ring.json"), ("D", "results/arena/circle-loop.json"), ("A", "results/methods/ring.json"),
        ("A", DOCS+"results/runs/ring.md"), ("M", DOCS+"methods/sources.md"), ("M", "README.md"),
        ("M", ".github/assets/readme/results-en.svg"), ("D", ".github/assets/readme/results-zh.svg"),
        ("M", ".github/workflows/ci.yml")])}
    assert verdicts == {
        "nexttransformer/models/ring.py": "ok", "nexttransformer/models/registry.py": "review",
        "nexttransformer/models/backbone.py": "blocked", "nexttransformer/evaluation/formal.py": "blocked",
        "nanochat/dataloader.py": "blocked", DOCS+"arena/rules.md": "blocked",
        "results/runs/ring.json": "ok",
        "results/runs/nsa.json": "blocked", "evidence/costs/ring/ring.json": "ok",
        "results/arena/ring.json": "ok", "results/arena/circle-loop.json": "blocked", "results/methods/ring.json": "ok",
        DOCS+"results/runs/ring.md": "ok", DOCS+"methods/sources.md": "ok", "README.md": "review",
        ".github/assets/readme/results-en.svg": "review", ".github/assets/readme/results-zh.svg": "blocked",
        ".github/workflows/ci.yml": "blocked"}


def test_scope_allows_a_calibration_refresh_but_not_a_model_change():
    path = "configs/linear/gdn.json"
    base = json.loads((REPO/path).read_text())
    base["calibration"] = {"source_hash": "an earlier audit"}
    assert classify([("M", path)], lambda _: json.dumps(base))[0][0] == "ok"
    base["model_hash"] = "0"*64
    assert classify([("M", path)], lambda _: json.dumps(base))[0][0] == "blocked"


def test_scope_distinguishes_arena_docs_from_entry_changes(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                        *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "Base")
    git("tag", "base")
    monkeypatch.setattr(arena, "REPO", tmp_path)
    paths = ["results/arena/README.md", DOCS + "arena/rules.md"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Updated documentation")
    git("add", ".")
    git("commit", "-qm", "Document the entry workflow")
    assert arena.scope("base") is None

    (tmp_path / "results/arena/entry.json").write_text("{}")
    git("add", ".")
    git("commit", "-qm", "Add an entry")
    verdicts = {path: verdict for verdict, _, path, _ in arena.scope("base")}
    assert verdicts[paths[0]] == "review"
    assert verdicts[paths[1]] == "blocked"
    assert verdicts["results/arena/entry.json"] == "ok"
