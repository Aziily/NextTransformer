"""Discoverable input contract derived from the native ModelConfig dataclass."""

from dataclasses import fields
import math

from .common import require


MODEL_DEFAULTS = {"layers": 24, "mlp_width": 6144, "window_pattern": "L"}
PROFILE_FIELDS = ("width", "heads", "sequence_len", "vocab_size")
ROUTING_SEED = 1729
JSON_TYPES = {int: "integer", float: "number", str: "string"}


def model_fields():
    # Import only the trusted native definition, never the uploaded plugin.
    from nexttransformer.models.config import ModelConfig

    return {field.name: field for field in fields(ModelConfig)}


def model_config_schema():
    properties = {
        name: {"type": JSON_TYPES[field.type], "default": field.default}
        for name, field in model_fields().items()
        if name != "options"
    }
    properties["options"] = {
        "type": "object",
        "default": {},
        "additionalProperties": {"$ref": "#/$defs/option_value"},
        "description": "Only keys declared in this plugin's method.json options; values are checked by the native plugin.",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "$defs": {
            "option_value": {
                "anyOf": [
                    {"type": ["string", "number", "boolean"]},
                    {"type": "array", "items": {"$ref": "#/$defs/option_value"}},
                ]
            }
        },
        "description": "Partial native ModelConfig input. Defaults shown are native defaults, before Gym/shell/profile merging. Native semantic checks and profile constraints also apply.",
    }


def profile_constraints(profile):
    return {
        "fixed": {
            **{field: profile[field] for field in PROFILE_FIELDS},
            "scale": "formal" if profile.get("formal_dimensions") else "debug",
            "routing_seed": ROUTING_SEED,
        },
        "method": "must_match_proposal_and_plugin",
        "validation": "native_ModelConfig_and_plugin_check",
    }


def describe_candidate_contract():
    from .package import CONTRACT, LIMIT

    return {
        "schema_version": "gym.candidate.v1",
        "sdk_version": CONTRACT,
        "transport": "submit_candidate(package_base64, idempotency_key)",
        "encoding": "base64_zip_utf8_source_only",
        "max_package_bytes": LIMIT,
        "max_uncompressed_bytes": LIMIT,
        "max_files": 32,
        "required_files": [
            "proposal.json",
            "architecture.json",
            "plugin/<method>/method.json",
            "plugin/<method>/__init__.py",
        ],
        "model_config_file": "architecture.json",
        "model_config_schema": model_config_schema(),
        "merge_order": [
            "native_ModelConfig_defaults",
            "gym_model_defaults",
            "method.json.shell",
            "architecture.json",
        ],
        "gym_model_defaults": dict(MODEL_DEFAULTS),
        "options_merge_order": [
            "method.json.options",
            "method.json.debug_options_when_debug",
            "architecture.json.options",
        ],
        "profile_constraints": "profiles[].model_config_constraints",
        "config_only_changes": "Re-upload the same plugin with a new architecture.json, or use calibrate_candidate.grid with complete replacement architecture objects (not patches).",
        "identity": "Each candidate is immutable. Source/config changes require a new candidate and fresh training; checkpoints require exact candidate/profile/runtime identity.",
    }


def validate_model_field_types(values):
    native = model_fields()
    require(
        isinstance(values, dict) and set(values) <= native.keys(),
        "invalid_artifact",
        "architecture.json accepts native ModelConfig fields; protocol/optimizer/data overrides are not accepted.",
    )
    for name, value in values.items():
        if name == "options":
            valid = isinstance(value, dict) and all(
                option_value(item) for item in value.values()
            )
            expected = "an object of native option values"
        else:
            kind = native[name].type
            valid = (
                (type(value) is int or type(value) is float and math.isfinite(value))
                if kind is float
                else type(value) is kind
            )
            expected = JSON_TYPES[kind]
        require(
            valid,
            "invalid_artifact",
            f"ModelConfig.{name} must be {expected}.",
        )


def option_value(value):
    return (
        type(value) in (str, bool, int)
        or type(value) is float
        and math.isfinite(value)
        or isinstance(value, list)
        and all(option_value(item) for item in value)
    )
