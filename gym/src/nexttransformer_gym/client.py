"""JSON/HTTP client shared by Harbor agents and other RSI frameworks."""

import base64
import json
import time
import urllib.error
import urllib.request
from .common import GymError


class GymClient:
    def __init__(self, endpoint, token, timeout=30):
        self.endpoint, self.token, self.timeout = endpoint.rstrip("/"), token, timeout

    def call(self, operation, **params):
        request = urllib.request.Request(
            self.endpoint + "/v1/call",
            method="POST",
            data=json.dumps(
                {
                    "schema_version": "gym.api.v1",
                    "operation": operation,
                    "params": params,
                }
            ).encode(),
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            result = json.load(error)
        if "error" in result:
            error = result["error"]
            raise GymError(
                error["code"], error["hint"], error["phase"], error["retryable"]
            )
        return result["result"]

    def submit_candidate(self, package_bytes, idempotency_key):
        return self.call(
            "submit_candidate",
            package_base64=base64.b64encode(package_bytes).decode(),
            idempotency_key=idempotency_key,
        )

    def wait_result(self, job_id, timeout=600, poll_interval=0.25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.call("get_job", job_id=job_id)
            if job["state"] == "succeeded":
                return self.call("get_result", job_id=job_id)
            if job["state"] in {
                "failed",
                "cancelled",
                "budget_stopped",
                "reconciliation",
            }:
                raise GymError(
                    job.get("error_code", job["state"]),
                    "Job did not complete successfully.",
                )
            time.sleep(poll_interval)
        raise TimeoutError("Job remains active; retry reads using its existing ID.")

    def __getattr__(self, operation):
        if operation.startswith("_"):
            raise AttributeError(operation)
        return lambda **params: self.call(operation, **params)
