"""Materialize native plugins only in dedicated worker/staging directories.

No source rewriting, expression interpretation or changes to upstream registries.
Each subprocess loads one immutable candidate through NEXTTRANSFORMER_METHOD_PATHS.
"""

import os
import sys
import tempfile
from pathlib import Path
from .common import require
from .candidate_contract import MODEL_DEFAULTS, PROFILE_FIELDS

_active = None
_directory = None


def write_plugin(candidate, root):
    root = Path(root)
    prefix = "plugin/" + candidate["method"] + "/"
    destination = root / candidate["method"]
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in candidate["files"].items():
        if name.startswith(prefix):
            path = destination / name[len(prefix) :]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.encode("utf-8"))
    return destination


def activate(candidate):
    global _active, _directory
    if _active == candidate["hash"]:
        return
    require(
        _active is None and "nexttransformer.models.registry" not in sys.modules,
        "incompatible_contract",
        "A native candidate requires a fresh worker process.",
    )
    _directory = tempfile.TemporaryDirectory(prefix="ntgym-plugin-")
    write_plugin(candidate, _directory.name)
    os.environ["NEXTTRANSFORMER_METHOD_PATHS"] = _directory.name
    _active = candidate["hash"]


def model_config(candidate, profile):
    from nexttransformer.models.config import ModelConfig

    declaration = candidate["declaration"]
    values = {
        **MODEL_DEFAULTS,
        **declaration.get("shell", {}),
        **candidate["architecture"],
    }
    options = dict(declaration.get("options", {}))
    if not profile.get("formal_dimensions"):
        options.update(declaration.get("debug_options", {}))
    options.update(values.pop("options", {}))
    values.pop("method", None)
    scale = "formal" if profile.get("formal_dimensions") else "debug"
    require(
        values.get("scale", scale) == scale,
        "incompatible_contract",
        "Config scale must match the execution profile.",
    )
    for field in PROFILE_FIELDS:
        require(
            field not in values or values[field] == profile[field],
            "incompatible_contract",
            f"{field} must match the execution profile.",
        )
        values[field] = profile[field]
    values.update(options=options, scale=scale)
    return ModelConfig(method=candidate["method"], **values)


def check_interface(model):
    from nexttransformer.models.lm import ReferenceLM
    from nexttransformer.models.plugins import check_shell

    require(
        isinstance(model, ReferenceLM),
        "incompatible_contract",
        "Native model hooks must subclass ReferenceLM.",
    )
    check_shell(model)
    for name in (
        "forward",
        "set_step",
        "set_row_offset",
        "set_training_stage",
        "routing_rng_state",
        "restore_routing_rng",
    ):
        require(
            callable(getattr(model, name, None)),
            "incompatible_contract",
            f"Model is missing {name}.",
        )
