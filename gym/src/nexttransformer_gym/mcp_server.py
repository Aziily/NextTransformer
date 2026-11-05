"""Stdio MCP projection of the authenticated Gym HTTP API.

The HTTP service owns discovery, validation, authorization and execution. This
process only translates MCP messages and never imports the training runtime.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from .client import GymClient
from .common import GymError


def create_server(client):
    server = Server("nexttransformer-gym")

    @server.list_tools()
    async def list_tools():
        try:
            space = await asyncio.to_thread(client.call, "describe_space")
            return [
                Tool(
                    name=name,
                    description=schema["description"],
                    inputSchema=schema["inputSchema"],
                    annotations=ToolAnnotations(readOnlyHint=schema["readOnly"]),
                )
                for name, schema in space["operation_schemas"].items()
            ]
        except Exception:
            # Do not expose transport URLs or credential-file contents to MCP.
            raise RuntimeError(
                "Unable to discover the Gym HTTP operation schemas."
            ) from None

    # HTTP validates arguments so its stable error envelope is preserved even
    # for malformed calls; MCP does not introduce a second validation policy.
    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        failed = False
        try:
            result = await asyncio.to_thread(client.call, name, **arguments)
        except GymError as error:
            result, failed = error.wire(), True
        except Exception:
            result = GymError(
                "transport_error",
                "Unable to reach the Gym HTTP service.",
                phase="transport",
                retryable=True,
            ).wire()
            failed = True
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result))],
            structuredContent=result if isinstance(result, dict) else None,
            isError=failed,
        )

    return server


async def serve(client):
    server = create_server(client)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credential", type=Path, default=Path("/run/gym-auth/credential.json")
    )
    args = parser.parse_args()
    try:
        credential = json.loads(args.credential.read_text())
        client = GymClient(credential["endpoint"], credential["token"])
    except Exception:
        print("Unable to load the Gym credential file.", file=sys.stderr)
        raise SystemExit(1) from None
    asyncio.run(serve(client))


if __name__ == "__main__":
    main()
