import gzip
import json
from pathlib import Path
import sqlite3

import pytest

from launch.profile_trace import (analyze_nsys_sqlite, analyze_torch_traces, attention_flops, busbw_factor,
                                  busy_breakdown, classify_kernel, component_of, matmul_flops, short_kernel_name)


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


def test_busy_breakdown_separates_compute_communication_overlap_memory_and_idle():
    result = busy_breakdown([(0.0, 10.0, "compute"), (5.0, 15.0, "communication"), (14.0, 20.0, "memory")])
    assert result == pytest.approx(dict(window_s=20.0, compute_s=10.0, communication_s=10.0, overlap_s=5.0,
                                        memory_s=6.0, exposed_memory_s=5.0, idle_s=0.0, exposed_communication_s=5.0))


def test_matmul_and_attention_flops_follow_the_audit_convention():
    assert matmul_flops("aten::mm", [[2048, 1536], [1536, 6144]]) == (2 * 2048 * 1536 * 6144, "2048×1536 · 1536×6144")
    assert matmul_flops("aten::addmm", [[6144], [2048, 1536], [1536, 6144]])[0] == 2 * 2048 * 1536 * 6144
    assert matmul_flops("aten::bmm", [[3, 64, 32], [3, 32, 16]]) == (2 * 3 * 64 * 32 * 16, "3×64×32 · 3×32×16")
    assert matmul_flops("aten::mm", [[]]) == (None, None)
    forward = ["", "", "", "", "", "2048", "2048", "0.", "True", "False", "0.088"]
    flops, shape = attention_flops("aten::_flash_attention_forward", [[1, 2048, 12, 128], [1, 2048, 12, 128]], forward)
    assert flops == 2 * 1 * 12 * 2048 * 2048 * 128
    assert shape == "B1 T2048 S2048 H12 D128 causal"
    backward = [""] * 11 + ["True"]
    dims = [[1, 2048, 12, 128]] * 5
    assert attention_flops("aten::_flash_attention_backward", dims, backward)[0] == 2 * flops
    full = forward[:8] + ["False"] + forward[9:]
    assert attention_flops("aten::_flash_attention_forward", [[1, 2048, 12, 128], [1, 2048, 12, 128]], full)[0] == 2 * flops


def test_component_of_names_block_children_and_numbered_leaves():
    assert component_of("transformer.h.3.attn") == ("attn", 3)
    assert component_of("value_embeds.1") == ("value_embeds", 1)
    assert component_of("transformer.wte") == ("wte", None)
    assert component_of("lm_head") == ("lm_head", None)


def test_busbw_factor_follows_nccl_tests():
    assert busbw_factor("allreduce", 8) == pytest.approx(1.75)
    assert busbw_factor("_allgather_base", 8) == pytest.approx(0.875)
    assert busbw_factor("ReduceScatter", 4) == pytest.approx(0.75)
    assert busbw_factor("Broadcast", 8) == 1.0
    assert busbw_factor("allreduce", None) is None


def X(cat, name, ts, dur, pid=1, tid=1, **args):
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": ts, "dur": dur, "args": args}


MM = {"Input Dims": [[2048, 1536], [1536, 1536]], "Input type": ["c10::BFloat16", "c10::BFloat16"]}
ATTENTION = {"Input Dims": [[1, 2048, 12, 128], [1, 2048, 12, 128], [1, 2048, 12, 128]],
             "Input type": ["c10::BFloat16"] * 3,
             "Concrete Inputs": ["", "", "", "", "", "2048", "2048", "0.", "True", "False", "0.088"]}


def torch_trace(path: Path):
    events = [
        X("user_annotation", "nt.stage:forward", 0, 100),
        X("user_annotation", "nt.stage:backward", 100, 200),
        X("user_annotation", "nt.stage:optimizer.gather", 300, 100),
        X("user_annotation", "nt.module:transformer.h.0.attn", 5, 35),
        X("user_annotation", "nt.module:transformer.h.0.mlp", 40, 50),
        X("cpu_op", "aten::linear", 10, 20, **{"Sequence number": 7, **MM}),
        X("cpu_op", "aten::mm", 12, 16, **{"Sequence number": 8, **MM}),
        X("cuda_runtime", "cudaLaunchKernel", 14, 2, correlation=10),
        X("kernel", "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn", 20, 50, pid=0, tid=7,
          correlation=10, device=0, stream=7),
        X("cpu_op", "aten::_flash_attention_forward", 32, 6, **{"Sequence number": 9, **ATTENTION}),
        X("cuda_runtime", "cudaLaunchKernel", 33, 1, correlation=14),
        X("kernel", "pytorch_flash::flash_fwd_kernel", 72, 20, pid=0, tid=7, correlation=14, device=0, stream=7),
        X("cpu_op", "aten::add", 92, 4, **{"Sequence number": 10}),
        X("cuda_runtime", "cudaLaunchKernel", 93, 1, correlation=16),
        X("kernel", "void at::native::vectorized_elementwise_kernel<4>()", 95, 5, pid=0, tid=7, correlation=16,
          device=0, stream=7),
        X("cpu_op", "autograd::engine::evaluate_function: MmBackward0", 150, 40, tid=2, **{"Sequence number": 8}),
        X("cpu_op", "aten::mm", 152, 36, tid=2, **MM),
        X("cuda_driver", "cuLaunchKernel", 155, 2, tid=2, correlation=11),
        X("kernel", "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nt", 160, 30, pid=0, tid=7,
          correlation=11, device=0, stream=7),
        X("cpu_op", "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad", 195, 10, tid=2),
        X("cpu_op", "nccl:all_reduce", 198, 5, tid=2),
        X("cuda_runtime", "cudaLaunchKernel", 200, 1, tid=2, correlation=12),
        X("kernel", "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)", 170, 60, pid=0, tid=20,
          correlation=12, device=0,
          stream=20, **{"Collective name": "allreduce", "dtype": "Float", "In msg nelems": 1_000_000,
                        "Out msg nelems": 1_000_000, "Group size": 8}),
        X("cuda_runtime", "cudaMemcpyAsync", 3, 1, correlation=13),
        X("gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 5, 5, pid=0, tid=7, correlation=13, device=0, stream=7),
        X("cpu_op", "aten::item", 245, 40),
        X("cuda_runtime", "cudaStreamSynchronize", 250, 30),
        X("cpu_op", "c10d::_allgather_base_", 303, 5),
        X("cuda_runtime", "cudaLaunchKernel", 305, 1, correlation=15),
        X("kernel", "ncclDevKernel_AllGather_RING_LL", 310, 40, pid=0, tid=20, correlation=15, device=0, stream=20,
          **{"Collective name": "_allgather_base", "dtype": "BFloat16", "In msg nelems": 1000,
             "Out msg nelems": 8000, "Group size": 8}),
    ]
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)


def find(nodes, name):
    return next(node for node in nodes if node["name"] == name)


def test_analyze_torch_traces_attributes_every_kernel(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    result = analyze_torch_traces([trace], steps=1)

    assert result["source"] == "torch"
    gpu = result["per_gpu"][0]
    assert gpu == pytest.approx(dict(
        gpu=0, window_s=345e-6, compute_s=105e-6, communication_s=100e-6, overlap_s=20e-6, memory_s=5e-6,
        exposed_memory_s=5e-6, idle_s=155e-6, exposed_communication_s=80e-6, kernels=7,
        gemm_s=80e-6, gemm_flops=4 * 2048 * 1536 * 1536, attention_s=20e-6, attention_flops=2 * 12 * 2048 ** 2 * 128),
        rel=1e-6)
    forward = find(result["tree"], "forward")
    assert find(forward["children"], "gemm")["children"][0]["name"] == "aten::linear [2048, 1536] · [1536, 1536]"
    backward = find(result["tree"], "backward")
    assert find(backward["children"], "communication")["children"][0]["name"] == "nccl:all_reduce"


def test_torch_trace_reports_gemm_efficiency_by_shape(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    rows = analyze_torch_traces([trace], steps=1)["gemm"]

    first = rows[0]
    assert (first["kind"], first["stage"], first["op"]) == ("gemm", "forward", "aten::mm")
    assert first["shape"] == "2048×1536 · 1536×1536" and first["dtype"] == "c10::BFloat16"
    assert first["seconds"] == pytest.approx(50e-6)
    assert first["tflops"] == pytest.approx(2 * 2048 * 1536 * 1536 / 50e-6 / 1e12)
    assert first["kernel"] == "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn"
    attention = next(row for row in rows if row["kind"] == "attention")
    assert attention["shape"] == "B1 T2048 S2048 H12 D128 causal"
    assert {row["stage"] for row in rows if row["kind"] == "gemm"} == {"forward", "backward"}


def test_torch_trace_maps_forward_and_backward_kernels_to_model_components(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    components = analyze_torch_traces([trace], steps=1)["components"]

    attention = find(components, "attn")
    assert attention["forward_s"] == pytest.approx(70e-6)
    assert attention["backward_s"] == pytest.approx(30e-6)
    assert attention["flops"] == 2 * 2048 * 1536 * 1536 * 2 + 2 * 12 * 2048 ** 2 * 128
    assert attention["layers"][0]["layer"] == 0
    outside = find(components, "(outside modules)")
    assert outside["forward_s"] == pytest.approx(5e-6) and outside["backward_s"] == 0


def test_components_of_a_hybrid_model_are_named_by_mixer_class(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    events = [
        X("user_annotation", "nt.stage:forward", 0, 100),
        X("user_annotation", "nt.module:h.0.attn|LinearMixer", 0, 40),
        X("user_annotation", "nt.module:h.1.attn|SparseMixer", 50, 40),
        X("cpu_op", "aten::add", 10, 5),
        X("cuda_runtime", "cudaLaunchKernel", 11, 1, correlation=1),
        X("kernel", "void at::native::vectorized_elementwise_kernel<4>()", 20, 10, pid=0, tid=7, correlation=1, device=0),
        X("cpu_op", "aten::mul", 60, 5),
        X("cuda_runtime", "cudaLaunchKernel", 61, 1, correlation=2),
        X("kernel", "void at::native::vectorized_elementwise_kernel<4>()", 70, 20, pid=0, tid=7, correlation=2, device=0),
    ]
    with gzip.open(trace, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    names = {row["name"]: row for row in analyze_torch_traces([trace], steps=1)["components"]}
    assert names["attn (LinearMixer)"]["forward_s"] == pytest.approx(10e-6)
    assert names["attn (SparseMixer)"]["layers"][0]["layer"] == 1


def test_torch_trace_reports_collectives_with_bandwidth_and_exposed_time(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    rows = {row["collective"]: row for row in analyze_torch_traces([trace], steps=1)["communication"]}

    reduce = rows["all_reduce"]
    assert (reduce["dtype"], reduce["stage"], reduce["kernel"]) == ("Float", "backward",
                                                                     "ncclDevKernel_AllReduce_Sum_f32_RING_LL")
    assert reduce["bytes"] == 4_000_000 and reduce["calls"] == 1
    assert reduce["exposed_s"] == pytest.approx(40e-6)
    assert reduce["algbw_gbps"] == pytest.approx(4e6 / 60e-6 / 1e9)
    assert reduce["busbw_gbps"] == pytest.approx(4e6 / 60e-6 / 1e9 * 1.75)
    gather = rows["all_gather"]
    assert gather["bytes"] == 16_000 and gather["stage"] == "optimizer.gather"
    assert gather["exposed_s"] == pytest.approx(40e-6)


def test_torch_trace_reports_launches_synchronizations_and_idle_causes(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    result = analyze_torch_traces([trace], steps=1)

    assert result["launch"] == pytest.approx(dict(api_calls=6, kernels=6, memory_ops=1, host_s=8e-6, micro_batches=1,
                                                   per_micro_batch=6))
    syncs = result["syncs"]
    assert syncs == [dict(api="cudaStreamSynchronize", source="aten::item", stage="backward", calls=1,
                          seconds=pytest.approx(30e-6))]
    idle = result["idle"]
    # Gaps: 10-20 (next launched at 14), 70-72 (queued at 33), 92-95 (93), 100-160 (155), 230-310 (305).
    assert idle["total_s"] == pytest.approx(155e-6)
    assert idle["host_s"] == pytest.approx(135e-6)
    assert idle["launch_s"] == pytest.approx(18e-6)
    assert idle["dependency_s"] == pytest.approx(2e-6)
    assert idle["gaps"] == 5
    largest = idle["largest"][0]
    assert largest["seconds"] == pytest.approx(80e-6) and largest["cause"] == "host"
    assert largest["start_s"] == pytest.approx(225e-6) and largest["stage"] == "optimizer.gather"
    assert largest["next"] == "ncclDevKernel_AllGather_RING_LL"


def test_torch_trace_builds_timeline_lanes(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    windows = {window["name"]: window for window in analyze_torch_traces([trace], steps=1)["timeline"]}

    update = windows["update"]
    assert update["seconds"] == pytest.approx(345e-6)
    assert set(update["lanes"]) == {"stage", "compute", "communication", "memory"}
    labels = {run[2] for run in update["lanes"]["compute"]}
    assert {"gemm", "attention", "elementwise"} <= labels
    assert {run[2] for run in update["lanes"]["communication"]} == {"all_reduce", "all_gather"}
    assert all(0 <= run[0] < run[1] <= update["buckets"] and 0 < run[3] <= 1 for lane in update["lanes"].values()
               for run in lane)
    tail = windows["update tail"]
    assert tail["start_s"] == pytest.approx(155e-6)
    mix = update["mix"]
    assert len(mix) == update["buckets"]
    assert all(0 <= matrix <= 1 and 0 <= other <= 1 and matrix + other <= 1.0001 for matrix, other in mix)
    assert sum(matrix for matrix, _ in mix) > sum(other for _, other in mix) > 0


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
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [
        (1, "ampere_bf16_s16816gemm_nn"), (2, "ncclDevKernel_AllGather_RING_LL"), (3, "nt.stage:optimizer.gather"),
        (4, "cudaLaunchKernel_v7000"), (5, "cudaStreamSynchronize_v3020")])
    kernels, runtime, nvtx = [], [], []
    for gpu, pid in ((0, 100), (1, 101)):
        main, worker = (pid << 24) | pid, (pid << 24) | (pid + 1)
        nvtx.append((1_000, 2_000, 59, "nt.stage:backward", None, main))
        nvtx.append((2_000, 3_000, 59, None, 3, main))
        runtime.append((1_100, 1_110, worker, 1, 4))
        runtime.append((2_100, 2_110, main, 2, 4))
        runtime.append((2_700, 2_800, main, 99, 5))
        kernels.append((1_200, 1_700 + 100 * gpu, gpu, 7, 1, pid << 24, 1, 1))
        kernels.append((2_200, 2_600, gpu, 20, 2, pid << 24, 2, 2))
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?)", kernels)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", runtime)
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)", nvtx)
    db.commit()
    db.close()


def test_analyze_nsys_sqlite_attributes_kernels_of_every_rank(tmp_path: Path):
    path = tmp_path / "trace.sqlite"
    write_nsys_sqlite(path)
    result = analyze_nsys_sqlite(path, steps=1)

    assert result["source"] == "nsys"
    assert [row["gpu"] for row in result["per_gpu"]] == [0, 1]
    assert result["per_gpu"][1]["compute_s"] == pytest.approx(600e-9)
    assert result["per_gpu"][0]["communication_s"] == pytest.approx(400e-9)
    assert result["per_gpu"][0]["gemm_flops"] is None
    backward = find(result["tree"], "backward")
    assert backward["seconds"] == pytest.approx((500e-9 + 600e-9) / 2)
    gather = find(result["tree"], "optimizer.gather")
    assert find(gather["children"], "communication")["children"][0]["name"] == "ncclDevKernel_AllGather_RING_LL"
    communication = result["communication"][0]
    assert (communication["collective"], communication["kernel"], communication["bytes"]) == \
        ("all_gather", "ncclDevKernel_AllGather_RING_LL", None)
    assert result["launch"]["api_calls"] == 2
    assert result["syncs"][0]["api"] == "cudaStreamSynchronize"
    assert result["syncs"][0]["stage"] == "optimizer.gather"
    assert result["gemm"] == [] and result["components"] == []


def test_torch_trace_ranks_non_matrix_operators_and_categories(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    result = analyze_torch_traces([trace], steps=1)

    assert result["categories"]["gemm"] == pytest.approx(80e-6)
    assert result["categories"]["elementwise"] == pytest.approx(5e-6)
    other = result["other_operators"]
    assert other == [dict(stage="forward", op="aten::add", category="elementwise", calls=1, seconds=pytest.approx(5e-6))]


def test_torch_trace_builds_a_host_and_stream_pipeline(tmp_path: Path):
    trace = tmp_path / "rank_0000.json.gz"
    torch_trace(trace)
    windows = {window["name"]: window for window in analyze_torch_traces([trace], steps=1)["pipeline"]}

    update = windows["update"]
    names = [lane["name"] for lane in update["lanes"]]
    assert names[0].startswith("Host") and "Stream 7" in names and "Stream 20" in names
    stream = next(lane for lane in update["lanes"] if lane["name"] == "Stream 20")
    assert {run[2] for run in stream["runs"]} == {"all_reduce", "all_gather"}
    host = update["lanes"][0]
    assert "sync" in {run[2] for run in host["runs"]}
    assert len(update["lead"]) == update["buckets"]
    # Launch-to-start of non-communication kernels: 6, 39, 2 and 5 us; the flash kernel queued longest.
    assert max(value for value in update["lead"] if value is not None) == pytest.approx(39e-6)
