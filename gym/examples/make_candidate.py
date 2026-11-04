"""Create a native multi-file model plugin without importing its numerical code."""

import argparse
import json
import zipfile

ENTRY = """from .model import ResearchModel
from .layers import ResearchFFN

model = ResearchModel

def ffn(config, layer):
    return ResearchFFN(config)
"""
MODEL = """from nexttransformer.models.lm import ReferenceLM

class ResearchModel(ReferenceLM):
    def _readout(self, x, backout):
        # This is ordinary model code, not a Gym expression or architecture DSL.
        return super()._readout(x, backout) * self.config.option("output_scale")
"""
LAYERS = """from torch import nn
from torch.nn import functional as F
from nexttransformer.models.backbone import Linear

class ResearchFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = config.option("activation")
        self.c_fc = Linear(config.width, config.mlp_width, bias=False)
        self.c_proj = Linear(config.mlp_width, config.width, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        if self.activation == "square":
            x = F.relu(x).square()
        elif self.activation == "silu":
            x = F.silu(x)
        else:
            x = x.tanh()
        return self.c_proj(x)
"""


def package(path, activation="square", layers=2, mlp_width=128):
    files = {
        "proposal.json": json.dumps(
            {
                "schema_version": "gym.candidate.v1",
                "sdk_version": "reference_plugin_v1",
                "method": "research_model",
                "parent_ids": [],
                "hypothesis": "Compare native model configurations and plugin implementations.",
            }
        ),
        "architecture.json": json.dumps(
            {
                "layers": layers,
                "mlp_width": mlp_width,
                "window_pattern": "L",
                "options": {"activation": activation},
            }
        ),
        "plugin/research_model/method.json": json.dumps(
            {
                "name": "research_model",
                "family": "others",
                "options": {"activation": "square", "output_scale": 1.0},
                "shell": {"window_pattern": "L"},
            }
        ),
        "plugin/research_model/__init__.py": ENTRY,
        "plugin/research_model/model.py": MODEL,
        "plugin/research_model/layers.py": LAYERS,
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, source in files.items():
            archive.writestr(name, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument(
        "--activation", choices=["square", "silu", "tanh"], default="square"
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mlp-width", type=int, default=128)
    args = parser.parse_args()
    package(args.path, args.activation, args.layers, args.mlp_width)
