"""Create a source-only reference FFN candidate, without importing torch."""

import argparse
import json
import zipfile


def package(path, activation="square", layers=2):
    tree = {"op": "input", "args": []}
    if activation == "square":
        tree = {"op": "square", "args": [{"op": "relu", "args": [tree]}]}
    else:
        tree = {"op": activation, "args": [tree]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "proposal.json",
            json.dumps(
                {
                    "schema_version": "gym.candidate.v1",
                    "sdk_version": "reference_plugin_v1",
                    "method": "research_ffn",
                    "parent_ids": [],
                    "hypothesis": "Compare causal pointwise FFN activations.",
                }
            ),
        )
        archive.writestr(
            "architecture.json", json.dumps({"layers": layers, "mlp_ratio": 4})
        )
        archive.writestr(
            "plugin/research_ffn/method.json",
            json.dumps(
                {
                    "name": "research_ffn",
                    "family": "others",
                    "hook": "ffn",
                    "language": "ffn-expression-v1",
                }
            ),
        )
        archive.writestr(
            "plugin/research_ffn/__init__.py", "FFN_EXPRESSION = " + repr(tree) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument(
        "--activation", choices=["square", "silu", "tanh"], default="square"
    )
    args = parser.parse_args()
    package(args.path, args.activation)
