"""Filesystem/process/network isolation for native research plugin workers.

This isolates OS resources. It does not make arbitrary in-process Python a trusted
accounting authority; adversarial accounting certification is a separate gate.
"""

import shutil
import os
import sys
from pathlib import Path
from .common import require


def command(python, platform, request):
    executable = shutil.which("bwrap")
    require(
        executable is not None,
        "resource_unavailable",
        "Native plugin workers require bubblewrap; there is no unsandboxed research fallback.",
    )
    args = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    roots = {Path(p) for p in ("/usr", "/bin", "/lib", "/lib64") if Path(p).exists()}
    roots |= {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(python).resolve().parent.parent,
        Path(__file__).resolve().parent,
        Path(platform) / "nexttransformer",
        Path(platform) / "nanochat",
    }
    if Path(python).is_symlink():
        target = Path(os.readlink(python))
        if target.is_absolute():
            roots.add(target.parent.parent)
    for path in sorted(roots, key=lambda p: (len(p.parts), str(p))):
        if any(path != parent and path.is_relative_to(parent) for parent in roots):
            continue
        args += ["--ro-bind", str(path), str(path)]
    for path in ("/etc/ld.so.cache", "/etc/localtime"):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    args += [
        "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp/cache",
    ]
    if request.get("output"):
        directory = Path(request["output"]).parent
        args += ["--bind", str(directory), str(directory)]
    if request.get("checkpoint"):
        args += ["--ro-bind", request["checkpoint"], request["checkpoint"]]
    data = request["profile"].get("data", {"kind": "synthetic_fixture"})
    if request["mode"] != "inspect" and data["kind"] != "synthetic_fixture":
        for path in [*data["train_paths"], data["dev_path"], data["tokenizer"]]:
            resolved = str(Path(path).resolve())
            args += ["--ro-bind", resolved, resolved]
    require(
        not str(request["profile"].get("device", "cpu")).startswith("cuda"),
        "resource_unavailable",
        "Research GPU plugins need a deployment-specific isolated device mapping; the supplied worker is CPU-only.",
    )
    return [*args, "--", python, "-m", "nexttransformer_gym.worker"]
