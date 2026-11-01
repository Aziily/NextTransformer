import torch

from nexttransformer.runtime import process


def test_initialize_process_group_passes_explicit_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setattr(process.dist, "init_process_group", lambda *args, **kwargs: calls.append((args, kwargs)))

    device = torch.device("cuda", 3)
    process.initialize_process_group(device, world=8)

    assert calls == [(('nccl',), {"device_id": device})]


def test_initialize_process_group_keeps_cpu_backend_simple(monkeypatch):
    calls = []
    monkeypatch.setattr(process.dist, "init_process_group", lambda *args, **kwargs: calls.append((args, kwargs)))

    process.initialize_process_group(torch.device("cpu"), world=2)

    assert calls == [(('gloo',), {})]
