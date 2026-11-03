import json
import os
import threading
import time
from urllib.request import urlopen

import pytest

from launch.run_monitor import (Monitor, find_runs, line_chart, load_run, read_events, render_compare, render_index,
                                render_run, schedule, serve, summarize, write_site)

PROTOCOL = dict(total_tokens=49_152_000_000, global_batch_tokens=1_048_576, val_every=250, val_tokens=41_943_040,
                core_steps=[8000, 16000, 24000, 32000, 40000, 46875], preflight_tolerance=0.01, cumulative_tolerance=0.02)


def train_event(step, loss=3.0, seconds=16.0, drift=0.0005, **extra):
    return dict(event="train", step=step, input_tokens=step * 1_048_576, loss=loss, lm_loss=loss, aux_loss=0.0,
                grad_norm=0.5, lr_multiplier=1.0, theory_flops=step * 4.9e15, theory_relative_error=drift,
                step_seconds=seconds, data_seconds=0.1, optimizer_seconds=0.4, tokens_per_second=1_048_576 / seconds,
                gpu_hours=8 * step * seconds / 3600, mfu_theory=None, peak_memory_bytes=20 * 2**30, **extra)


def make_run(folder, events, command=(), smoke=False, status=None, method="dense_full"):
    folder.mkdir(parents=True, exist_ok=True)
    provenance = dict(config=f"configs/baseline/{method}.json", model=dict(method=method), protocol=PROTOCOL,
                      command=["nexttransformer/train.py", *command], world_size=8, micro_batch=1, accumulation=64,
                      parameters=1_355_810_568, smoke=smoke, admission=dict(mode="research"),
                      cost=dict(flops_per_token=4.6747e9, dense_flops_per_token=4.6724e9))
    (folder / "provenance.json").write_text(json.dumps(provenance))
    (folder / "metrics.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    if status:
        (folder / "run_status.json").write_text(json.dumps(status))
    return folder


def formal_events(last, every=10, loss=lambda step: 4.0 - step / 10_000):
    events = [dict(event="validation", step=0, val_bpb=3.2, input_tokens=41_943_040, seconds=40.0, split="val")]
    for step in [1, *range(every, last + 1, every)]:
        events.append(train_event(step, loss(step)))
        if step % 250 == 0:
            events.append(dict(event="validation", step=step, val_bpb=1.2 - step / 100_000, input_tokens=41_943_040,
                               seconds=40.0, split="val"))
    return events


def test_read_events_skips_a_line_being_written(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event": "train", "step": 1}\nnot json\n{"event": "train", "step": 2}\n{"event": "tr')
    events, malformed = read_events(path)
    assert [event["step"] for event in events] == [1, 2]
    assert malformed == 1  # the broken middle line; the unfinished last line is still being written


def test_find_runs_keeps_run_folders_and_skips_their_contents(tmp_path):
    make_run(tmp_path / "a", [train_event(1)])
    make_run(tmp_path / "group" / "b", [train_event(1)])
    make_run(tmp_path / "profile" / "run", [train_event(1)])
    (tmp_path / "a" / "checkpoint_00000010").mkdir()
    make_run(tmp_path / "a" / "nested", [train_event(1)])  # inside a run: not a separate run
    (tmp_path / "empty").mkdir()
    assert sorted(find_runs([tmp_path])) == ["a", "group/b", "profile/run"]
    assert sorted(find_runs([tmp_path / "a"])) == ["a"]


def test_formal_schedule_matches_the_protocol(tmp_path):
    plan = schedule(dict(protocol=PROTOCOL, command=["train.py"]))
    assert plan["total"] == 46_875 and not plan["smoke"]
    assert len(plan["validation"]) == 189 and plan["validation"][:2] == [0, 250] and plan["validation"][-1] == 46_875
    assert plan["core"] == [8000, 16000, 24000, 32000, 40000, 46875]


def test_smoke_schedule_follows_the_command():
    plan = schedule(dict(protocol=PROTOCOL, smoke=True, command=["train.py", "--smoke-steps", "40", "--log-every", "1"]))
    assert (plan["total"], plan["validation"], plan["core"], plan["log_every"]) == (40, [0, 40], [], 1)
    plan = schedule(dict(protocol=PROTOCOL, command=["train.py", "--smoke-steps=5", "--smoke-core-per-task", "32"]))
    assert (plan["total"], plan["core"]) == (5, [5])


def test_running_formal_run_progress_and_estimate(tmp_path):
    run = make_run(tmp_path / "dense", formal_events(1000))
    now = os.path.getmtime(run / "metrics.jsonl") + 5
    summary = summarize("dense", load_run(run), now=now, results=None)
    assert summary["status"] == "running" and summary["step"] == 1000 and summary["total"] == 46_875
    assert summary["step_seconds"] == pytest.approx(16.0)
    assert summary["schedule"]["done_validation"] == [0, 250, 500, 750, 1000]
    # Updates left at 16 s plus 184 validations at 40 s; CORE has no measurement yet.
    assert summary["estimate_seconds"] == pytest.approx(45_875 * 16 + 184 * 40)
    assert "CORE" in summary["estimate_excludes"] and summary["remaining_core"] == 6
    assert summary["findings"][0]["level"] == "good"


def test_status_comes_from_run_status_unless_events_are_newer(tmp_path):
    run = make_run(tmp_path / "smoke", [train_event(1), train_event(2)], command=["--smoke-steps", "2"], smoke=True,
                   status=dict(phase="smoke_complete", step=2))
    old = time.time() - 3600
    os.utime(run / "metrics.jsonl", (old, old))
    summary = summarize("smoke", load_run(run), results=None)
    assert (summary["status"], summary["estimate_seconds"]) == ("complete", None)
    # A resumed segment appends events after the previous segment's status file.
    make_run(tmp_path / "resumed", formal_events(300), status=dict(phase="segment_complete", step=200))
    os.utime(tmp_path / "resumed" / "run_status.json", (old, old))
    summary = summarize("resumed", load_run(tmp_path / "resumed"), now=time.time(), results=None)
    assert summary["status"] == "running" and summary["step"] == 300


def test_quiet_run_is_flagged_and_core_waits_are_named(tmp_path):
    run = make_run(tmp_path / "stalled", formal_events(500))
    now = os.path.getmtime(run / "metrics.jsonl") + 3 * 3600
    summary = summarize("stalled", load_run(run), now=now, results=None)
    assert summary["status"] == "quiet"
    finding = next(f for f in summary["findings"] if f["title"].startswith("No events"))
    assert finding["level"] == "warning" and "may have stopped" in finding["detail"]
    run = make_run(tmp_path / "core", formal_events(8000, every=100))
    summary = summarize("core", load_run(run), now=os.path.getmtime(run / "metrics.jsonl") + 3 * 3600, results=None)
    assert "CORE is scheduled" in next(f for f in summary["findings"] if f["title"].startswith("No events"))["detail"]


def test_health_findings(tmp_path):
    events = formal_events(400)
    events[-1] = train_event(400, float("nan"), drift=0.025)
    events.append(dict(event="budget_anomaly", step=400, rolling_relative_error=0.04))
    run = make_run(tmp_path / "bad", events)
    summary = summarize("bad", load_run(run), now=os.path.getmtime(run / "metrics.jsonl"), results=None)
    titles = [f["title"] for f in summary["findings"]]
    assert titles[:2] == ["Non-finite training values", "Compute ledger outside the cumulative tolerance"]
    assert "1 budget anomaly event(s)" in titles and "No problems found" not in titles

    # A staged method's early updates may sit above 2%; the admitted proof's full-run figure is what an audit judges.
    staged = make_run(tmp_path / "staged", [train_event(1, drift=0.0226), train_event(10, drift=0.0226)])
    provenance = json.loads((staged / "provenance.json").read_text())
    provenance["cost"]["relative_error"] = 0.0041
    (staged / "provenance.json").write_text(json.dumps(provenance))
    summary = summarize("staged", load_run(staged), now=os.path.getmtime(staged / "metrics.jsonl"), results=None)
    titles = [f["title"] for f in summary["findings"]]
    assert "Staged compute" in titles and not any("tolerance" in title for title in titles)
    provenance["cost"]["relative_error"] = 0.015
    (staged / "provenance.json").write_text(json.dumps(provenance))
    summary = summarize("staged", load_run(staged), now=os.path.getmtime(staged / "metrics.jsonl"), results=None)
    assert summary["findings"][0]["title"] == "Compute ledger outside the admission tolerance"

    spiky = formal_events(400, loss=lambda step: 9.0 if step == 400 else 3.0)
    run = make_run(tmp_path / "spike", spiky)
    summary = summarize("spike", load_run(run), now=os.path.getmtime(run / "metrics.jsonl"), results=None)
    assert any(f["title"] == "Loss spike" and "update 400" in f["detail"] for f in summary["findings"])


def test_mfu_uses_the_reported_value_or_the_given_peak(tmp_path):
    events = [train_event(1), train_event(10), train_event(20)]
    run = make_run(tmp_path / "mfu", events)
    assert summarize("mfu", load_run(run), results=None)["mfu"] is None
    summary = summarize("mfu", load_run(run), peak_tflops=71.2, results=None)
    # 4.9 PFLOP charged per update over 16 s on 8 GPUs of 71.2 TFLOP/s.
    assert summary["mfu"] == pytest.approx(4.9e15 / (16 * 8 * 71.2e12))
    events = [train_event(1) | dict(mfu_theory=0.5), train_event(2) | dict(mfu_theory=0.42)]
    run = make_run(tmp_path / "reported", events)
    assert summarize("reported", load_run(run), peak_tflops=71.2, results=None)["mfu"] == pytest.approx(0.42)


def test_published_curve_is_found_by_method_and_alias(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "dense_full.json").write_text(json.dumps(dict(name="Dense Baseline", val=[dict(step=0, value=3.3)])))
    (results / "sssl.json").write_text(json.dumps(dict(name="SSSL", val=[dict(step=250, value=1.0)])))
    run = make_run(tmp_path / "runs" / "a", [train_event(1)])
    assert summarize("a", load_run(run), results=results)["published"] == dict(name="Dense Baseline", val=[[0, 3.3]])
    run = make_run(tmp_path / "runs" / "b", [train_event(1)], method="dense")
    assert summarize("b", load_run(run), results=results)["published"]["name"] == "SSSL"


def test_load_run_rereads_changed_files(tmp_path):
    run = make_run(tmp_path / "live", [train_event(1)])
    assert len(load_run(run)["events"]) == 1
    with (run / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(train_event(10)) + "\n")
    assert [event["step"] for event in load_run(run)["events"]] == [1, 10]


def test_line_chart_escapes_names_and_handles_empty_and_log_data():
    empty = line_chart("x", "Loss", [dict(name="a", points=[])])
    assert "No data yet." in empty and "<svg" not in empty
    chart = line_chart("x", "Loss", [dict(name="<script>", points=[[1, 3.0], [10, 2.0]]),
                                      dict(name="b", points=[[1, 0.0], [10, 2.5]])], log=True)
    assert "<script>" not in chart.split('class="chart-data"')[0] and "&lt;script&gt;" in chart
    assert chart.count('class="series') == 1 + 0  # the zero point is dropped on a log scale, leaving one point for "b"
    assert "Table view" in chart and '<div class="legend">' in chart
    payload = chart.split('class="chart-data">')[1].split("</script>")[0]
    assert "</" not in payload


def test_pages_render_for_every_state(tmp_path):
    make_run(tmp_path / "running", formal_events(1000))
    make_run(tmp_path / "smoke", [train_event(1), train_event(2)], command=["--smoke-steps", "2"], smoke=True,
             status=dict(phase="smoke_complete", step=2))
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "provenance.json").write_text(json.dumps(dict(model=dict(method="gdn"))))
    monitor = Monitor([tmp_path], results=None)
    summaries = monitor.summaries()
    assert [s["id"] for s in summaries][0] == "running"
    index = render_index(summaries, lambda i: "/run/" + i)
    assert "Training runs" in index and "/run/smoke" in index and "No events yet" in index
    for summary in summaries:
        page = render_run(summary)
        assert page.startswith("<!doctype html>") and "Health" in page
    compare = render_compare(monitor.summaries(["running", "smoke"]), lambda i: "/run/" + i)
    assert "LM loss at update 1</th>" in compare  # update 1 is the only one both runs logged
    assert "share no logged update" in render_compare(monitor.summaries(["running", "empty"]), lambda i: i)
    assert "Select runs" in render_compare([], lambda i: i)


def test_write_site_and_never_touch_run_directories(tmp_path):
    runs = tmp_path / "runs"
    run = make_run(runs / "a b", formal_events(300))
    before = {path.name: path.stat().st_mtime_ns for path in run.iterdir()}
    pages = write_site(Monitor([runs], results=None), tmp_path / "site")
    assert {page.name for page in pages} == {"index.html", "compare.html", "run-a-b.html"}
    assert 'href="run-a-b.html"' in (tmp_path / "site" / "index.html").read_text()
    assert {path.name: path.stat().st_mtime_ns for path in run.iterdir()} == before
    assert not (run / "run.lock").exists()


def test_server_routes(tmp_path):
    make_run(tmp_path / "r1", formal_events(300))
    server = serve(Monitor([tmp_path], results=None), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "r1" in urlopen(base + "/").read().decode()
        assert "Language-model loss" in urlopen(base + "/run/r1").read().decode()
        assert "Matched update" in urlopen(base + "/compare?run=r1").read().decode()
        rows = json.loads(urlopen(base + "/api/runs").read())
        assert rows[0]["id"] == "r1" and "charts" not in rows[0]
        assert json.loads(urlopen(base + "/api/run/r1").read())["step"] == 300
        with pytest.raises(Exception) as error:
            urlopen(base + "/run/missing")
        assert "404" in str(error.value)
    finally:
        server.shutdown()
        server.server_close()
