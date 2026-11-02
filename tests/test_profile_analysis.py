import pytest

from launch.profile_analysis import (diagnose, hardware_peak, memory_summary, mfu_summary, phase_bands, phase_flops,
                                     project_formal, steady_interval, summarize_stages, wall_accounting)


def segment(stage, host, device=None, start=0.0):
    return {"stage": stage, "host_s": host, "device_s": device, "start_wall": start}


def update(step, start, forward, backward, optimizer, data=0.01):
    parts = [("step_setup", 0.01), ("data", data), ("forward", forward), ("backward", backward),
             ("grad_norm", 0.02), ("wait", 0.0), ("optimizer.reduce", optimizer / 2),
             ("optimizer.compute", optimizer / 4), ("optimizer.gather", optimizer / 4), ("wait", 0.0),
             ("step_tail", 0.01)]
    segments, now = [], start
    for stage, seconds in parts:
        segments.append(segment(stage, seconds, seconds, now))
        now += seconds
    return {"step": step, "profiled": False, "segments": segments}, now


def record(rank, backward=0.5):
    steps = [{"step": 0, "profiled": False, "segments": [segment("startup", 20.0, None, 0.0),
                                                           segment("validation", 2.0, 2.0, 20.0),
                                                           segment("step_tail", 1.0, 1.0, 22.0)]}]
    now = 23.0
    for step, forward in ((1, 3.0), (2, 0.25), (3, 0.3)):
        built, now = update(step, now, forward, backward, 0.4)
        steps.append(built)
    steps[-1]["segments"] += [segment("validation", 1.5, 1.5, now), segment("checkpoint", 18.0, 18.0, now + 1.5),
                              segment("step_tail", 0.1, 0.1, now + 19.5)]
    return {"rank": rank, "world": 2, "profiled_steps": [], "steps": steps}


def test_summarize_stages_uses_steady_updates_and_reports_rank_spread():
    result = summarize_stages([record(0), record(1, backward=0.6)], warmup_steps=1)

    assert result["steady_steps"] == [2, 3]
    assert result["timing"] == "cuda_events"
    stages = {row["stage"]: row for row in result["stages"]}
    assert list(stages)[:4] == ["step_setup", "data", "forward", "backward"]
    assert "validation" not in stages and "checkpoint" not in stages
    assert stages["forward"]["seconds"] == pytest.approx(0.275)
    assert stages["backward"]["rank_min"] == pytest.approx(0.5)
    assert stages["backward"]["rank_max"] == pytest.approx(0.6)
    assert stages["backward"]["slowest_rank"] == 1
    assert result["step_seconds"] == pytest.approx(0.01 + 0.01 + 0.275 + 0.55 + 0.02 + 0.4 + 0.01)
    assert sum(row["share"] for row in result["stages"]) == pytest.approx(1.0)


def test_wall_accounting_separates_startup_warmup_steady_and_events():
    phases = {row["phase"]: row["seconds"] for row in wall_accounting(record(0), warmup_steps=1)}

    assert phases["startup"] == pytest.approx(21.0)
    assert phases["validation"] == pytest.approx(3.5)
    assert phases["checkpoint"] == pytest.approx(18.0)
    assert phases["warmup"] == pytest.approx(3.0 + 0.5 + 0.45)
    assert phases["steady"] == pytest.approx(2 * 0.45 + 0.25 + 0.3 + 2 * 0.5 + 0.1)


def test_steady_interval_and_phase_bands_follow_rank_zero_wall_clock():
    timeline = record(0)
    start, end = steady_interval(timeline, warmup_steps=1)
    first = timeline["steps"][2]["segments"][0]["start_wall"]
    assert start == pytest.approx(first)
    assert end == pytest.approx(timeline["steps"][3]["segments"][10]["start_wall"] + 0.01)

    bands = phase_bands(timeline, warmup_steps=1)
    assert [band["phase"] for band in bands] == ["startup", "validation", "startup", "warmup", "steady",
                                                 "validation", "checkpoint", "steady"]
    assert all(band["end"] >= band["start"] for band in bands)


def test_project_formal_only_when_the_update_shape_matches():
    protocol = {"total_tokens": 49_152_000_000, "global_batch_tokens": 1_048_576, "sequence_len": 2048,
                "data_parallel_size": 8}
    matched = project_formal(protocol, micro_batch=1, accumulation=64, world=8, step_seconds=60.0)
    assert matched["representative"] is True
    assert matched["formal_accumulation"] == 64
    assert matched["updates"] == 46_875
    assert matched["training_hours"] == pytest.approx(46_875 * 60 / 3600)
    protocol.update(val_every=250, val_tokens=41_943_040)
    full = project_formal(protocol, micro_batch=1, accumulation=64, world=8, step_seconds=60.0,
                          validation={"seconds": 207.0, "input_tokens": 41_943_040}, checkpoint_seconds=18.0)
    assert full["validation_events"] == 189
    assert full["validation_hours"] == pytest.approx(189 * 207 / 3600)
    assert full["checkpoints"] == 94
    assert full["estimated_hours"] == pytest.approx(full["training_hours"] + full["validation_hours"]
                                                    + 94 * 18 / 3600)
    # A one-batch smoke validation is mostly fixed overhead; scaling it 2,560 times would overstate validation.
    small = project_formal(protocol, micro_batch=1, accumulation=64, world=8, step_seconds=60.0,
                           validation={"seconds": 2.0, "input_tokens": 16_384}, checkpoint_seconds=18.0)
    assert "validation_hours" not in small
    assert small["validation_events"] == 189 and small["validation_measured_tokens"] == 16_384
    smoke = project_formal(protocol, micro_batch=1, accumulation=1, world=8, step_seconds=1.4)
    assert smoke["representative"] is False
    assert "training_hours" not in smoke


def test_diagnose_flags_unrepresentative_shape_communication_idle_data_and_skew():
    stages = {"step_seconds": 1.0, "stages": [
        {"stage": "data", "seconds": 0.2, "share": 0.2, "rank_min": 0.2, "rank_max": 0.2, "slowest_rank": 0},
        {"stage": "backward", "seconds": 0.5, "share": 0.5, "rank_min": 0.4, "rank_max": 0.6, "slowest_rank": 3},
        {"stage": "optimizer.reduce", "seconds": 0.3, "share": 0.3, "rank_min": 0.3, "rank_max": 0.3,
         "slowest_rank": 0},
    ]}
    trace = {"per_gpu": [dict(gpu=0, window_s=1.0, compute_s=0.3, communication_s=0.6, overlap_s=0.1, memory_s=0.0,
                              idle_s=0.2, exposed_communication_s=0.5, kernels=10)],
             "small_kernels": {"count_share": 0.7, "time_share": 0.05}}
    findings = diagnose(stages=stages, trace=trace, projection={"representative": False, "accumulation": 1,
                                                                "formal_accumulation": 64},
                        accounting=[{"phase": "steady", "seconds": 10.0}, {"phase": "checkpoint", "seconds": 40.0}],
                        telemetry={"per_gpu": []}, peak_memory_bytes=None, memory_total_bytes=None)
    titles = [finding["id"] for finding in findings]
    assert titles[0] == "unrepresentative-update"
    for expected in ("exposed-communication", "gpu-idle", "data-stall", "rank-skew", "small-kernels",
                     "non-training-wall"):
        assert expected in titles
    assert all(finding["severity"] in ("warning", "info") for finding in findings)


def test_diagnose_flags_lagging_and_memory_pressure_from_steady_telemetry():
    telemetry = {"per_gpu": [{"index": index, "gpu_util_mean_pct": 10.0 if index == 2 else 90.0}
                             for index in range(8)]}
    findings = diagnose(stages={"step_seconds": 1.0, "stages": []}, trace=None, projection=None, accounting=[],
                        telemetry=telemetry, peak_memory_bytes=23e9, memory_total_bytes=24e9)
    ids = {finding["id"]: finding for finding in findings}
    assert "GPU 2" in ids["lagging-gpu"]["detail"]
    assert "memory-pressure" in ids


def test_hardware_peak_follows_the_architecture_not_the_product_name():
    consumer = hardware_peak(dict(name="NVIDIA GeForce Example", major=8, minor=6, sms=82, max_mhz=2100),
                             observed_mhz=1875)
    assert consumer["dense_bf16_flops"] == pytest.approx(82 * 512 * 1695e6)
    assert consumer["dense_bf16_flops"] / 1e12 == pytest.approx(71.2, abs=0.1)
    assert consumer["observed_clock_flops"] == pytest.approx(82 * 512 * 1875e6)
    assert "sm86" in consumer["basis"] and "1695 MHz" in consumer["basis"]
    datacenter = hardware_peak(dict(name="NVIDIA Example", major=8, minor=0, sms=108, max_mhz=1410))
    assert datacenter["dense_bf16_flops"] / 1e12 == pytest.approx(312, abs=0.5)
    unlisted = hardware_peak(dict(name="NVIDIA GeForce Other", major=8, minor=6, sms=70, max_mhz=2000))
    assert unlisted["dense_bf16_flops"] == pytest.approx(70 * 512 * 2000e6)
    assert "maximum SM clock" in unlisted["basis"]
    assert hardware_peak(dict(name="Unknown", major=12, minor=0, sms=10, max_mhz=1000))["dense_bf16_flops"] is None
    override = hardware_peak(dict(name="x", major=8, minor=6, sms=82, max_mhz=2100), override_tflops=50.0)
    assert override["dense_bf16_flops"] == 50e12 and override["basis"] == "--peak-tflops 50"
    assert hardware_peak(None)["dense_bf16_flops"] is None


def test_phase_flops_splits_the_audited_update_cost():
    proof = {"stages": [{"stage": "main", "optimizer_steps": 46875, "cost": {
        "phases": {"forward": 300, "backward": 600, "optimizer": 100}, "total": 1000,
        "operators": {"forward/aten.mm.default": {"matrix": 260, "theory": 260},
                      "forward/native_attention": {"matrix": 20, "theory": 21},
                      "backward/aten.mm.default": {"matrix": 520, "theory": 520},
                      "backward/native_attention": {"matrix": 40, "theory": 42},
                      "backward/aten.add.Tensor": {"matrix": 0, "theory": 3},
                      "optimizer/aten.bmm.default": {"matrix": 90, "theory": 90}}}}]}
    # Forward and backward scale with the tokens of the update; the optimizer runs once per update.
    assert phase_flops(proof, token_share=0.25) == dict(forward=75.0, backward=150.0, optimizer=100.0, total=325.0,
                                                       matrix=(280 + 560) * 0.25 + 90, attention=(21 + 42) * 0.25)
    assert phase_flops({"stages": [{}, {}]}, token_share=1) is None
    # A staged method (indexer burn-in, then joint training) uses the stage whose total the trainer charged.
    staged = {"stages": [dict(proof["stages"][0], stage="burnin", cost=dict(proof["stages"][0]["cost"], total=1100)),
                         dict(proof["stages"][0], stage="joint")]}
    assert phase_flops(staged, token_share=1, charged_total=1100)["forward"] == 300
    assert phase_flops(staged, token_share=1, charged_total=1000)["total"] == 1000
    assert phase_flops(staged, token_share=1) is None


def test_mfu_summary_reports_phases_kernels_and_the_loss_waterfall():
    stages = {"step_seconds": 10.0, "stages": [
        {"stage": "data", "seconds": 0.1}, {"stage": "forward", "seconds": 3.0}, {"stage": "backward", "seconds": 6.0},
        {"stage": "grad_norm", "seconds": 0.1}, {"stage": "optimizer.reduce", "seconds": 0.0},
        {"stage": "optimizer.compute", "seconds": 0.2}, {"stage": "optimizer.gather", "seconds": 0.6}]}
    flops = dict(forward=60e12, backward=120e12, optimizer=2e12, total=182e12, matrix=178e12, attention=8e12)
    trace = {"per_gpu": [dict(gpu=0, window_s=10.0, compute_s=8.5, communication_s=1.2, overlap_s=0.2, memory_s=0.1,
                              exposed_memory_s=0.05, idle_s=0.45, exposed_communication_s=1.0, kernels=100, gemm_s=5.0,
                              gemm_flops=170e12, attention_s=1.0, attention_flops=8e12)]}
    result = mfu_summary(flops=flops, world=1, step_seconds=10.0, stages=stages, trace=trace, peak=50e12)

    assert result["achieved_flops"] == pytest.approx(18.2e12)
    assert result["mfu"] == pytest.approx(18.2 / 50)
    phases = {row["phase"]: row for row in result["phases"]}
    assert phases["forward"]["mfu"] == pytest.approx(60 / 3 / 50)
    assert phases["backward"]["mfu"] == pytest.approx(120 / 6 / 50)
    assert phases["optimizer"]["seconds"] == pytest.approx(0.8)
    kernels = {row["kind"]: row for row in result["kernels"]}
    assert kernels["gemm"]["efficiency"] == pytest.approx(170 / 5 / 50)
    assert kernels["attention"]["tflops"] == pytest.approx(8)
    parts = {row["part"]: row["seconds"] for row in result["waterfall"]}
    assert parts["ideal"] == pytest.approx(182 / 50)
    assert parts["gemm below peak"] == pytest.approx(5.0 - 170 / 50)
    assert parts["attention below peak"] == pytest.approx(1.0 - 8 / 50)
    assert parts["other compute kernels"] == pytest.approx(8.5 - 6.0 - (182 - 178) / 50)
    assert parts["exposed communication"] == pytest.approx(1.0)
    assert parts["idle"] == pytest.approx(0.45)
    assert sum(parts.values()) == pytest.approx(10.0, abs=0.06)
    with_categories = dict(trace, categories={"gemm": 5.0, "attention": 1.0, "elementwise": 2.0, "normalization": 0.4,
                                              "communication": 1.2})
    other = next(row for row in mfu_summary(flops=flops, world=1, step_seconds=10.0, stages=stages,
                                            trace=with_categories, peak=50e12)["waterfall"]
                 if row["part"] == "other compute kernels")
    assert other["detail"] == [{"category": "elementwise", "seconds": 2.0}, {"category": "normalization", "seconds": 0.4}]
    # Attention FLOPs always come from the audit: custom attention kernels carry no shapes in the trace.
    opaque = dict(trace, per_gpu=[dict(trace["per_gpu"][0], attention_flops=0.0)])
    attention = next(row for row in mfu_summary(flops=flops, world=1, step_seconds=10.0, stages=stages, trace=opaque,
                                                peak=50e12)["kernels"] if row["kind"] == "attention")
    assert attention["tflops"] == pytest.approx(8) and attention["flops_source"] == "audit"
    without_trace = mfu_summary(flops=flops, world=1, step_seconds=10.0, stages=stages, trace=None, peak=50e12)
    assert without_trace["waterfall"] == [] and without_trace["kernels"] == []
    assert mfu_summary(flops=flops, world=1, step_seconds=10.0, stages=stages, trace=None, peak=None)["mfu"] is None


def test_memory_summary_reports_per_rank_peaks_and_headroom():
    records = [{"rank": rank, "steps": [{"step": 0, "profiled": False, "segments": []},
                                         {"step": 1, "profiled": False, "segments": [], "memory": dict(
                                             allocated_peak=10 * 2**30, reserved_peak=12 * 2**30)},
                                         {"step": 2, "profiled": False, "segments": [], "memory": dict(
                                             allocated_peak=(18 + rank) * 2**30, reserved_peak=(20 + rank) * 2**30)}]}
               for rank in range(2)]
    result = memory_summary(records, warmup_steps=1, total_bytes=24 * 2**30)
    assert [row["rank"] for row in result["per_rank"]] == [0, 1]
    assert result["per_rank"][1] == dict(rank=1, allocated_peak=19 * 2**30, reserved_peak=21 * 2**30)
    assert result["headroom_bytes"] == 3 * 2**30
    assert result["reserved_share"] == pytest.approx(21 / 24)
    assert memory_summary([], warmup_steps=1, total_bytes=None) is None


def test_diagnose_prices_a_per_micro_batch_host_sync_by_the_idle_it_causes():
    trace = {"per_gpu": [], "launch": {"micro_batches": 64},
             "syncs": [{"api": "cudaStreamSynchronize", "source": "aten::to", "stage": "backward", "calls": 64,
                        "seconds": 4.6}],
             "idle": {"total_s": 0.39, "host_s": 0.21, "launch_s": 0.01, "dependency_s": 0.17}}
    findings = diagnose(stages={"step_seconds": 16.0, "stages": []}, trace=trace, projection=None, accounting=[],
                        telemetry={"per_gpu": []}, peak_memory_bytes=None, memory_total_bytes=None)
    sync = next(finding for finding in findings if finding["id"] == "host-sync")
    assert sync["severity"] == "info"
    assert "0.21 s" in sync["detail"] and "aten::to" in sync["detail"]


def test_phase_flops_uses_a_named_stage_when_the_profile_forced_one():
    proof = {"stages": [
        {"stage": "burnin", "cost": {"phases": {"forward": 900, "backward": 900, "optimizer": 10}, "total": 1810}},
        {"stage": "joint", "cost": {"phases": {"forward": 300, "backward": 600, "optimizer": 10}, "total": 910}}]}
    assert phase_flops(proof, token_share=1, charged_total=1810, stage="joint")["forward"] == 300


def test_rank_gantt_aligns_every_rank_on_one_wall_clock():
    from launch.profile_analysis import rank_gantt

    records = []
    for rank, offset in ((0, 0.0), (1, 0.5)):
        steady = {"step": 2, "profiled": False, "segments": [
            segment("step_setup", 0.1, 0.1, 100.0 + offset), segment("forward", 1.0, 1.0, 100.1 + offset),
            segment("backward", 2.0, 2.0, 101.1 + offset), segment("optimizer.gather", 0.5, 0.5, 103.1 + offset),
            segment("validation", 5.0, 5.0, 103.6 + offset)]}
        records.append({"rank": rank, "steps": [{"step": 0, "profiled": False, "segments": []},
                                                {"step": 1, "profiled": False, "segments": []}, steady]})
    gantt = rank_gantt(records, warmup_steps=1)
    assert gantt["step"] == 2
    assert gantt["seconds"] == pytest.approx(4.1)
    assert [row["rank"] for row in gantt["rows"]] == [0, 1]
    assert gantt["rows"][1]["segments"][0] == pytest.approx([0.5, 0.1, "step_setup"])
    assert [part[2] for part in gantt["rows"][0]["segments"]] == ["step_setup", "forward", "backward", "optimizer.gather"]
    assert rank_gantt([], warmup_steps=1) is None
