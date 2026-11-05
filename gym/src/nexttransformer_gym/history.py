"""Read-only, bounded research context; no Git access or source execution."""

import json
from pathlib import Path
from .common import file_hash, integer, require


def page(items, cursor="", limit=50):
    require(isinstance(cursor, str), "invalid_request", "Invalid cursor.")
    limit = integer(limit, 1, 100)
    remaining = [
        item for item in sorted(items, key=lambda x: x["id"]) if item["id"] > cursor
    ]
    selected = remaining[:limit]
    return {
        "items": selected,
        "next_cursor": selected[-1]["id"] if len(remaining) > limit else None,
    }


def research_context(platform):
    root = Path(platform).resolve()
    references = {}
    sources = sorted((root / "nexttransformer/models").rglob("*.py"))
    sources += sorted((root / "nexttransformer/methods").rglob("*.py"))
    for path in sources:
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size > 262144
        ):
            continue
        references["source:" + path.relative_to(root).as_posix()] = path.read_text()
    configurations = {}
    for path in sorted((root / "configs").glob("*/*.json")):
        if path.is_symlink() or path.stat().st_size > 1048576:
            continue
        data = json.loads(path.read_text())
        if isinstance(data.get("model"), dict):
            configurations[(path.parent.name, path.stem)] = (path, data)
    experiments = {}
    cards = {}
    for path in sorted((root / "results/methods").glob("*.json")):
        if not path.is_symlink() and path.stat().st_size <= 1048576:
            cards[path.stem] = json.loads(path.read_text())
    for path in sorted((root / "results/runs").glob("*.json")):
        if path.is_symlink() or path.stat().st_size > 1048576:
            continue
        result = json.loads(path.read_text())
        if not isinstance(result.get("step"), int) or result["step"] <= 0:
            continue
        method = result.get("methodId")
        native = str(result.get("cohort", "")).startswith("native-")
        matches = [
            (p, c)
            for (family, name), (p, c) in configurations.items()
            if name == method and (family == "native") == native
        ]
        architecture = None
        if len(matches) == 1:
            config_path, config = matches[0]
            expected = result.get("proof", {}).get("model_hash")
            actual = config.get("model_hash")
            architecture = {
                "model": config["model"],
                "config_path": config_path.relative_to(root).as_posix(),
                "config_sha256": file_hash(config_path),
                "model_hash": actual,
                "matches_recorded_model_hash": actual == expected if expected else None,
                "provenance": "current_repository_config",
            }
        experiments[path.stem] = {
            "id": path.stem,
            "scope": "published",
            "method": method,
            "name": result.get("name"),
            "cohort": result.get("cohort"),
            "status": result.get("status"),
            "step": result["step"],
            "result": result,
            "result_sha256": file_hash(path),
            "architecture": architecture,
            "method_card": cards.get(path.stem) or cards.get(method),
            "source_reference_ids": sorted(references),
            "source_provenance": "current_repository_reference_not_historical_checkout",
            "visibility": "published_historical_results_only",
        }
    return references, experiments


def summary(experiment):
    return {
        k: experiment.get(k)
        for k in ("id", "scope", "method", "name", "cohort", "status", "step")
    }
