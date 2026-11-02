"""Every published record under results/ is valid."""
import json
import shutil

import pytest

from nexttransformer.export_run import P, RESULTS, check, record


def raw_from_record(path):
    r = json.loads(path.read_text()); p = r["proof"]
    core = [(e["step"], dict(core_metric=c["value"], results=e["raw"], centered=e["centered"], full_core=True))
            for e, c in zip(r["taskEvaluations"], r["core"])]
    return dict(run_dir=r["id"], val=[(x["step"], x["value"]) for x in r["val"]], core=core, final_bpb=dict(val_bpb=r["finalBpb"]),
                last=dict(step=P.steps, input_tokens=P.total_tokens, theory_flops=r["trainingCompute"]["trainingFlopsEstimate"]),
                status="matched", core_final_mtime=r["lastRecordedAt"],
                provenance=[dict(tokenizer_sha256=p["tokenizer_sha256"], protocol_hash=p["protocol_hash"], model_hash=p["model_hash"],
                                 data_manifest_sha256=p["data_manifest_sha256"], software=dict(science_hash=p["science_hash"]),
                                 cost=dict(flops_per_token=p["cost_proof_gflops_per_token"]*1e9,
                                           relative_error=p["cost_proof_vs_dense_pct"]/100))])


def test_results_records_are_valid():
    assert check() == []


def test_export_records_a_provenance_note_only_when_given():
    raw = raw_from_record(RESULTS/"runs/dense_full.json")
    plain, _ = record(raw, "gqa", "GQA", commit="a"*40)
    noted, path = record(raw, "gqa", "GQA", commit="a"*40, note="Trained from a frozen branch.")
    assert "comparisonNote" not in plain and noted["comparisonNote"] == "Trained from a frozen branch."
    assert path == RESULTS/"runs/gqa.json" and noted["proof"]["src_head"] == "a"*40


def test_export_refuses_a_run_with_a_research_mode_segment():
    raw = raw_from_record(RESULTS/"runs/dense_full.json")
    first = raw["provenance"][0]
    formal = dict(raw, provenance=[dict(first, admission=dict(mode="formal", findings=[]))])
    assert record(formal, "gqa", "GQA", commit="a"*40)[0]["complete"]
    research = dict(raw, provenance=[*formal["provenance"], dict(first, admission=dict(mode="research", findings=[]))])
    with pytest.raises(ValueError, match="research mode"):
        record(research, "gqa", "GQA", commit="a"*40)


def test_runs_share_one_flat_directory(tmp_path):
    root = tmp_path/"results"
    shutil.copytree(RESULTS, root)
    (root/"runs/early").mkdir()
    assert any("results/runs/ is flat" in p for p in check(root))


def test_baseline_acceptance_must_match_one_complete_dense_run(tmp_path):
    root = tmp_path/"results"
    shutil.copytree(RESULTS, root)
    dense = next(p for p in root.glob("runs/*.json") if json.loads(p.read_text()).get("baselineAcceptance"))
    record = json.loads(dense.read_text())
    acceptance = dict(record["baselineAcceptance"])
    record["baselineAcceptance"]["sourceCommit"] = "0"*40
    dense.write_text(json.dumps(record))
    other = root/"runs/gdn.json"
    gdn = json.loads(other.read_text())
    gdn["baselineAcceptance"] = acceptance
    other.write_text(json.dumps(gdn))
    problems = check(root)
    assert any("differs from the run's provenance" in p for p in problems)
    assert any("needs a complete, matched dense_full" in p for p in problems)
    assert any("more than one accepted Dense baseline" in p for p in problems)



@pytest.fixture
def isolated_results(tmp_path):
    """One real reference record and its card; no dependency on archived frozen audits."""
    root = tmp_path/"results"
    for folder in ("runs", "methods"):
        (root/folder).mkdir(parents=True)
        shutil.copy(RESULTS/folder/"dense_full.json", root/folder/"dense_full.json")
    return root


def edit_dense(root, edit):
    path = root/"runs/dense_full.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("change", ["missing", "truncated", "duplicate", "reordered", "wrong-step", "missing-raw", "missing-task"])
def test_finished_run_needs_complete_core_evidence(isolated_results, change):
    def edit(r):
        if change == "missing": r.pop("taskEvaluations")
        elif change == "truncated": r["taskEvaluations"].pop()
        elif change == "duplicate": r["taskEvaluations"][1] = r["taskEvaluations"][0]
        elif change == "reordered": r["taskEvaluations"].reverse()
        elif change == "wrong-step": r["taskEvaluations"][0]["step"] += 1
        elif change == "missing-raw": r["taskEvaluations"][0].pop("raw")
        else: r["taskEvaluations"][0]["raw"].pop(next(iter(r["taskEvaluations"][0]["raw"])))
    assert check(isolated_results) == []
    edit_dense(isolated_results, edit)
    assert any("CORE evidence" in problem for problem in check(isolated_results))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5", None])
@pytest.mark.parametrize("field", ["val", "core", "finalBpb", "raw", "centered"])
def test_result_metrics_must_be_finite_numbers(isolated_results, field, value):
    def edit(r):
        if field in ("val", "core"): r[field][0]["value"] = value
        elif field == "finalBpb": r[field] = value
        else:
            scores = r["taskEvaluations"][0][field]
            scores[next(iter(scores))] = value
    edit_dense(isolated_results, edit)
    assert any("finite numeric metric" in problem for problem in check(isolated_results))


@pytest.mark.parametrize("field", ["val", "core", "final_bpb", "raw", "centered"])
def test_export_rejects_nonfinite_metrics(field):
    raw = raw_from_record(RESULTS/"runs/dense_full.json")
    if field == "val": raw["val"][0] = (raw["val"][0][0], float("nan"))
    elif field == "core": raw["core"][0][1]["core_metric"] = float("nan")
    elif field == "final_bpb": raw[field]["val_bpb"] = float("nan")
    else:
        scores = raw["core"][0][1]["results" if field == "raw" else "centered"]
        scores[next(iter(scores))] = float("nan")
    with pytest.raises(ValueError, match="finite numeric metric"):
        record(raw, "dense_full", "Dense", commit="a"*40)


@pytest.fixture
def export_cli(tmp_path, monkeypatch):
    from nexttransformer import export_run
    raw = raw_from_record(RESULTS/"runs/dense_full.json")
    tasks = export_run.core_baselines()
    root = tmp_path/"results"
    (root/"runs").mkdir(parents=True)
    monkeypatch.setattr(export_run, "REPO", tmp_path)
    monkeypatch.setattr(export_run, "RESULTS", root)
    monkeypatch.setattr(export_run, "core_baselines", lambda: tasks)
    monkeypatch.setattr(export_run, "read_run", lambda _: raw)

    def run(*extra):
        export_run.main(["run", "--method", "dense_full", "--name", "Dense", "--commit", "a"*40, *extra])
    return root, run


def test_export_repeat_is_a_noop_and_new_runs_need_a_new_file(export_cli):
    root, run = export_cli
    path = root/"runs/dense_full.json"
    run()
    original, mtime = path.read_bytes(), path.stat().st_mtime_ns
    run()
    assert path.read_bytes() == original and path.stat().st_mtime_ns == mtime
    with pytest.raises(SystemExit, match="new --file"):
        run("--name", "Another run")
    assert path.read_bytes() == original
    run("--name", "Another run", "--file", "dense_full_second")
    assert (root/"runs/dense_full_second.json").exists() and path.read_bytes() == original


def test_export_cannot_remove_baseline_acceptance(export_cli):
    root, run = export_cli
    path = root/"runs/dense_full.json"
    run("--accept-baseline", "--audit", "evidence/audits/dense_full.json")
    original = path.read_bytes()
    assert json.loads(original)["baselineAcceptance"]["status"] == "accepted"
    with pytest.raises(SystemExit, match="already exists"):
        run()
    assert path.read_bytes() == original
