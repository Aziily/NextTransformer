"""Use Harbor's environment API to observe the actual main container."""

import json


class HarborRuntime:
    def __init__(self, environment):
        self.environment = environment

    async def snapshot(self):
        result = await self.environment.exec(
            command="python /opt/gym/runtime_snapshot.py", user="root"
        )
        if result.return_code != 0:
            raise RuntimeError("Harbor could not read the agent runtime")
        return json.loads(result.stdout)
