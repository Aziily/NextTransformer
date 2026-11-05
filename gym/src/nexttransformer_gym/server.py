"""Authenticated HTTP service with a stable host listener."""

import fcntl
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .common import GymError, require
from .backend import Scheduler


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def bind(service, state_dir, host="127.0.0.1"):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "service.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = state_dir / "listener.json"
    port = json.loads(config.read_text())["port"] if config.exists() else 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ready":true}')

        def do_POST(self):
            try:
                require(self.path == "/v1/call", "not_found", "Unknown endpoint.")
                length = int(self.headers.get("Content-Length", "0"))
                require(
                    0 < length <= 2 * 1024 * 1024
                    and not self.headers.get("Transfer-Encoding"),
                    "invalid_request",
                    "Invalid request size or transfer encoding.",
                )
                self.connection.settimeout(10)
                data = json.loads(self.rfile.read(length))
                auth = self.headers.get("Authorization", "")
                require(
                    auth.startswith("Bearer "),
                    "denied_access",
                    "Session credential required.",
                )
                require(
                    data["schema_version"] == "gym.api.v1",
                    "incompatible_contract",
                    "Unknown API version.",
                )
                value = service.call(
                    auth[7:], data["operation"], data.get("params", {})
                )
                response, status = (
                    {"schema_version": "gym.api.v1", "result": value},
                    200,
                )
            except GymError as error:
                response, status = (
                    error.wire(),
                    {"denied_access": 403, "not_found": 404, "conflict": 409}.get(
                        error.code, 422
                    ),
                )
            except (ValueError, KeyError, TypeError):
                response, status = (
                    GymError("invalid_request", "Malformed request.").wire(),
                    400,
                )
            except Exception:
                response, status = (
                    GymError(
                        "infrastructure_failure",
                        "Controller request failed.",
                        retryable=True,
                    ).wire(),
                    500,
                )
            payload = json.dumps(response, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    try:
        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        # Retain the bound socket; never probe and release a port.
        atomic_json(config, {"port": server.server_port})
        server.service_lock = lock
        return server
    except BaseException:
        lock.close()
        raise


def serve(service, state_dir, host="127.0.0.1"):
    server = bind(service, state_dir, host)
    stop = threading.Event()
    scheduler = Scheduler(service)

    def schedule():
        while not stop.is_set():
            if not scheduler.work_once():
                stop.wait(0.2)

    worker = threading.Thread(target=schedule, daemon=True)
    worker.start()
    print(json.dumps({"port": server.server_port}), flush=True)
    try:
        server.serve_forever()
    finally:
        stop.set()
        worker.join(timeout=5)
        server.server_close()
        server.service_lock.close()
