"""Native Harbor agent adapter. Provider credentials stay in the host broker.

The only container commands are fixed runtime start and bounded mailbox reads.
Model output is never interpolated into a command or imported as executable code.
"""

import asyncio
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from harbor.agents.base import BaseAgent


class GymResearchAgent(BaseAgent):
    @staticmethod
    def name():
        return "nexttransformer-gym-researcher"

    def version(self):
        return "1.0.0"

    async def setup(self, environment):
        result = await environment.exec(
            command='python -c "import nexttransformer_gym.agent_loop"', user="agent"
        )
        if result.return_code:
            raise RuntimeError("Gym client runtime is not installed in main")

    def inference(self, messages):
        # Explicit deployment inputs; never discover a user's ambient credentials.
        endpoint = os.environ["GYM_MODEL_ENDPOINT"]
        key = os.environ["GYM_MODEL_API_KEY"]
        model = os.environ["GYM_MODEL_NAME"]
        if not endpoint.startswith("https://"):
            raise ValueError("Inference broker endpoint must use HTTPS")
        body = {"model": model, "messages": messages, "max_tokens": 4096}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Inference response exceeds bounds")
        return json.loads(data)["choices"][0]["message"]["content"]

    async def run(self, instruction, environment, context):
        with tempfile.TemporaryDirectory(prefix="ntgym-broker-") as tmp:
            path = Path(tmp) / "message.json"
            path.write_text(instruction)
            await environment.upload_file(path, "/workspace/gym-instruction.txt")
            execution = asyncio.create_task(
                environment.exec(
                    command="python -m nexttransformer_gym.agent_loop",
                    user="agent",
                    timeout_sec=1800,
                )
            )
            previous = -1
            try:
                while not execution.done():
                    result = await environment.exec(
                        command="head -c 4194305 /workspace/.model/request.json",
                        user="agent",
                        timeout_sec=5,
                    )
                    if result.return_code == 0:
                        if len(result.stdout) > 4 << 20:
                            raise ValueError("Inference request exceeds bounds")
                        request = json.loads(result.stdout)
                        if type(request["id"]) is not int or request["id"] <= previous:
                            await asyncio.sleep(0.2)
                            continue
                        messages = request["messages"]
                        if not isinstance(messages, list) or any(
                            set(m) != {"role", "content"}
                            or m["role"] not in {"system", "user", "assistant"}
                            or not isinstance(m["content"], str)
                            for m in messages
                        ):
                            raise ValueError("Malformed inference request")
                        content = await asyncio.to_thread(self.inference, messages)
                        path.write_text(
                            json.dumps({"id": request["id"], "content": content})
                        )
                        await environment.upload_file(
                            path, "/workspace/.model/response.tmp"
                        )
                        await environment.exec(
                            command="mv /workspace/.model/response.tmp /workspace/.model/response.json",
                            user="agent",
                        )
                        previous = request["id"]
                    await asyncio.sleep(0.2)
                result = await execution
                if result.return_code:
                    raise RuntimeError("Mediated researcher did not complete")
                context.metadata = {
                    "execution_policy": "mediated-json-v1",
                    "model": os.environ["GYM_MODEL_NAME"],
                    "model_turns": previous + 1,
                    "inference_cost_accounting": "deferred",
                }
            finally:
                if not execution.done():
                    execution.cancel()
