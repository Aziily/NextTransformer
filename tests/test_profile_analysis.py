import gzip
import json
from pathlib import Path
import sqlite3

import pytest

from launch.profile_analysis import (analyze_nsys_sqlite, analyze_torch_traces, busy_breakdown, classify_kernel,
                                     diagnose, phase_bands, project_formal, short_kernel_name, steady_interval,
                                     summarize_stages, wall_accounting)


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


@pytest.mark.parametrize("name, category", [
    ("ncclDevKernel_AllReduce_Sum_f32_RING_LL", "communication"),
    ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn", "gemm"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)", "gemm"),
    ("flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel", "attention"),
    ("triton_red_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_2", "fused"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>>(int, T2, T3)", "elementwise"),
    ("layer_norm_grad_input_kernel_vectorized", "normalization"),
    ("multi_tensor_apply_kernel", "optimizer"),
    ("Memcpy HtoD (Pageable -> Device)", "memory"),
    ("cunn_SoftMaxForward", "softmax"),
    ("reduce_kernel", "reduction"),
    ("CatArrayBatchedCopy_alignedK_contig", "copy"),
])
def test_classify_kernel(name, category):
    assert classify_kernel(name) == category


def test_short_kernel_name_keeps_the_informative_part():
    assert short_kernel_name("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>>(int, T2, T3)") \
        == "at::native::vectorized_elementwise_kernel"
    assert short_kernel_name("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)") \
        == "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8"
    assert short_kernel_name("ncclDevKernel_AllGather_RING_LL") == "ncclDevKernel_AllGather_RING_LL"
    assert short_kernel_name("void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float*)") \
        == "at::native::vectorized_layer_norm_kernel"


def test_busy_breakdown_separates_compute_communication_overlap_and_idle():
    result = busy_breakdown([(0.0, 10.0, "compute"), (5.0, 15.0, "communication"), (18.0, 20.0, "memory")])
    assert result == pytest.approx(dict(window_s=20.0, compute_s=10.0, communication_s=10.0, overlap_s=5.0,
                                        memory_s=2.0, idle_s=3.0, exposed_communication_s=5.0))


def write_torch_trace(path: Path):
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "nt.stage:forward", "pid": 1, "tid": 1, "ts": 0, "dur": 100},
        {"ph": "X", "cat": "user_annotation", "name": "nt.stage:backward", "pid": 1, "tid": 1, "ts": 100, "dur": 200},
        {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "pid": 1, "tid": 1, "ts": 10, "dur": 20,
         "args": {"External id": 1, "Input Dims": [[1, 4], [8, 4], []]}},
        {"ph": "X", "cat": "cpu_op", "name": "aten::addmm", "pid": 1, "tid": 1, "ts": 12, "dur": 15, "args": {"External id": 2}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 1, "ts": 14, "dur": 2,
         "args": {"correlation": 10, "External id": 2}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaMemcpyAsync", "pid": 1, "tid": 1, "ts": 3, "dur": 1,
         "args": {"correlation": 13}},
        {"ph": "X", "cat": "cpu_op", "name": "autograd::engine::evaluate_function: AddmmBackward0", "pid": 1,
         "tid": 2, "ts": 150, "dur": 40, "args": {"External id": 3}},
        {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernel", "pid": 1, "tid": 2, "ts": 155, "dur": 2,
         "args": {"correlation": 11}},
        {"ph": "X", "cat": "cpu_op", "name": "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad",
         "pid": 1, "tid": 2, "ts": 195, "dur": 10, "args": {"External id": 5}},
        {"ph": "X", "cat": "cpu_op", "name": "nccl:all_reduce", "pid": 1, "tid": 2, "ts": 198, "dur": 5,
         "args": {"External id": 4}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 2, "ts": 200, "dur": 1,
         "args": {"correlation": 12}},
        {"ph": "X", "cat": "kernel", "name": "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn", "pid": 0,
         "tid": 7, "ts": 20, "dur": 50, "args": {"correlation": 10, "device": 0, "stream": 7}},
        {"ph": "X", "cat": "kernel", "name": "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nt", "pid": 0,
         "tid": 7, "ts": 160, "dur": 30, "args": {"correlation": 11, "device": 0, "stream": 7}},
        {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL", "pid": 0, "tid": 20,
         "ts": 170, "dur": 60, "args": {"correlation": 12, "device": 0, "stream": 20}},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Pageable -> Device)", "pid": 0, "tid": 7,
         "ts": 5, "dur": 5, "args": {"correlation": 13, "device": 0, "stream": 7}},
    ]
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)


def find(nodes, name):
    return next(node for node in nodes if node["name"] == name)


def test_analyze_torch_traces_builds_stage_category_operator_kernel_tree(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    write_torch_trace(trace)
    result = analyze_torch_traces([trace], steps=1)

    assert result["source"] == "torch"
    assert result["per_gpu"][0] == pytest.approx(dict(
        gpu=0, window_s=225e-6, compute_s=80e-6, communication_s=60e-6, overlap_s=20e-6, memory_s=5e-6,
        idle_s=100e-6, exposed_communication_s=40e-6, kernels=4), rel=1e-6)
    forward = find(result["tree"], "forward")
    assert forward["seconds"] == pytest.approx(55e-6)
    gemm = find(forward["children"], "gemm")
    operator = gemm["children"][0]
    assert operator["name"] == "aten::linear [1, 4] · [8, 4]"
    assert operator["children"][0]["name"] == "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn"
    backward = find(result["tree"], "backward")
    assert find(backward["children"], "communication")["children"][0]["name"] == "nccl:all_reduce"
    assert find(backward["children"], "gemm")["children"][0]["name"] == "AddmmBackward0"
    assert result["top_kernels"][0]["name"] == "ncclDevKernel_AllReduce_Sum_f32_RING_LL"


def write_nsys_sqlite(path: Path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, streamId INT, correlationId INT,
            globalPid INT, demangledName INT, shortName INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT, globalTid INT, correlationId INT, nameId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, deviceId INT, streamId INT, correlationId INT,
            globalPid INT, bytes INT, copyKind INT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, eventType INT, text TEXT, textId INT, globalTid INT);
    """)
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "ampere_bf16_s16816gemm_nn"), (2, "ncclDevKernel_AllGather_RING_LL"),
                                                         (3, "nt.stage:optimizer.gather")])
    kernels, runtime, nvtx = [], [], []
    for gpu, pid in ((0, 100), (1, 101)):
        main, worker = (pid << 24) | pid, (pid << 24) | (pid + 1)
        nvtx.append((1_000, 2_000, 59, "nt.stage:backward", None, main))
        nvtx.append((2_000, 3_000, 59, None, 3, main))
        runtime.append((1_100, 1_110, worker, 1, 0))
        runtime.append((2_100, 2_110, main, 2, 0))
        kernels.append((1_200, 1_700 + 100 * gpu, gpu, 7, 1, pid << 24, 1, 1))
        kernels.append((2_200, 2_600, gpu, 20, 2, pid << 24, 2, 2))
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?)", kernels)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", runtime)
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)", nvtx)
    db.commit()
    db.close()


def test_analyze_nsys_sqlite_attributes_kernels_of_every_rank_to_stages(tmp_path: Path):
    path = tmp_path / "trace.sqlite"
    write_nsys_sqlite(path)
    result = analyze_nsys_sqlite(path, steps=1)

    assert result["source"] == "nsys"
    assert [row["gpu"] for row in result["per_gpu"]] == [0, 1]
    assert result["per_gpu"][1]["compute_s"] == pytest.approx(600e-9)
    assert result["per_gpu"][0]["communication_s"] == pytest.approx(400e-9)
    backward = find(result["tree"], "backward")
    assert backward["seconds"] == pytest.approx((500e-9 + 600e-9) / 2)
    gather = find(result["tree"], "optimizer.gather")
    assert find(gather["children"], "communication")["children"][0]["name"] == "ncclDevKernel_AllGather_RING_LL"


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
