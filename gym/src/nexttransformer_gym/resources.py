"""Observe the local execution node without claiming unconfigured remote capacity."""

import hashlib
import os
import socket
from pathlib import Path


def local_server_id():
    return "local-" + hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def limit_file(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def observe(server_id, profiles, busy):
    cpus = float(
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    quota = limit_file("/sys/fs/cgroup/cpu.max")
    if quota:
        amount, period = quota.split()
        if amount != "max":
            cpus = min(cpus, int(amount) / int(period))
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    maximum = limit_file("/sys/fs/cgroup/memory.max")
    if maximum and maximum != "max":
        memory = min(memory, int(maximum))
    return {
        "id": server_id,
        "backend": "local_subprocess",
        "reachable": True,
        "cpu_capacity": cpus,
        "memory_limit_bytes": memory,
        "configured_cuda_devices": sorted(
            {str(d) for p in profiles for d in p.get("cuda_visible_devices", [])}
        ),
        "research_devices": ["cpu"],
        "worker_slots": 1,
        "busy_slots": int(busy),
        "allocatable_slots": int(not busy),
        "placement": "same_node_as_gym_service",
    }
