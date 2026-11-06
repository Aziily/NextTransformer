"""The comparison behind `checks.speedups`: what counts as a difference between two runs."""
import json

import torch

from nexttransformer.checks import speedups


def run(path, loss=1., weight=1., seconds=1., extra_validation=False):
    (path/"checkpoint_00000002").mkdir(parents=True)
    rows = [dict(event="train", step=1, loss=loss, step_seconds=seconds), dict(event="validation", step=2, val_bpb=.5)]
    if extra_validation: rows.append(dict(event="validation", step=1, val_bpb=.7))
    (path/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    state = dict(training_seconds=seconds, software=dict(native_source_sha256=str(seconds)))
    if seconds != 1.: state["software"]["speedups"] = "exact"
    torch.save(dict(model=dict(w=torch.tensor([weight])), state=state), path/"checkpoint_00000002/model.pt")
    return path


def test_identical_runs_may_differ_in_bookkeeping(tmp_path):
    assert speedups.compare(run(tmp_path/"a"), run(tmp_path/"b", seconds=2.)) == []


def test_a_resumed_run_may_validate_once_more(tmp_path):
    assert speedups.compare(run(tmp_path/"a"), run(tmp_path/"b", extra_validation=True)) == []


def test_a_logged_value_is_a_difference(tmp_path):
    assert speedups.compare(run(tmp_path/"a"), run(tmp_path/"b", loss=2.)) == ["event ('train', 1)"]


def test_a_weight_is_a_difference(tmp_path):
    assert speedups.compare(run(tmp_path/"a"), run(tmp_path/"b", weight=2.)) == ["checkpoint model.pt/model/w"]
