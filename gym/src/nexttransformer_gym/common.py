"""Wire identities and stable errors shared by client and control plane."""

import hashlib
import json
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def new_id():
    return uuid.uuid4().hex


class GymError(Exception):
    def __init__(self, code, hint, phase="control", retryable=False):
        self.code, self.hint, self.phase, self.retryable = code, hint, phase, retryable
        super().__init__(hint)

    def wire(self):
        return {
            "schema_version": "gym.api.v1",
            "error": {
                "code": self.code,
                "hint": self.hint,
                "phase": self.phase,
                "retryable": self.retryable,
                "details": {},
            },
        }


def require(condition, code, hint):
    if not condition:
        raise GymError(code, hint)


def integer(value, minimum=0, maximum=None):
    require(
        type(value) is int
        and value >= minimum
        and (maximum is None or value <= maximum),
        "invalid_request",
        "Integer outside the supported range.",
    )
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
