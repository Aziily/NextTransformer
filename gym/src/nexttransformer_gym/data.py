"""Explicit data views; research never opens a validation shard.

Reuses the upstream best-fit algorithm without changing its source. The adapter
sets the same stream fields but supplies a manifest-selected list of shards.
"""

from pathlib import Path
from .common import identity, require, file_hash


def parquet_stream(paths, tokenizer, sequence_len, split):
    from nexttransformer.runtime.data import BestFitStream

    paths = [Path(p) for p in paths]
    require(
        bool(paths) and split in {"train", "dev"},
        "invalid_request",
        "Explicit train/dev paths are required.",
    )
    names = [p.name for p in paths]
    require(
        "shard_06542.parquet" not in names,
        "denied_access",
        "Formal validation is not a research data view.",
    )
    require(
        (split == "dev" and names == ["shard_06541.parquet"])
        or (split == "train" and "shard_06541.parquet" not in names),
        "denied_access",
        "Train and dev data views overlap.",
    )
    if split == "train":
        require(
            names == [f"shard_{i:05d}.parquet" for i in range(len(names))]
            and len(names) <= 6541,
            "incompatible_contract",
            "Training view must be the pinned contiguous prefix before dev.",
        )
    stream = object.__new__(BestFitStream)
    stream.paths, stream.tokenizer, stream.split = paths, tokenizer, split
    stream.rank, stream.world = 0, 1
    stream.sequence_len, stream.buffer_size = sequence_len, 1000
    stream.tokenizer_batch_size, stream.tokenizer_threads = 128, 4
    stream.cursor = dict(pq_idx=0, rg_idx=0, document_offset=0, epoch=1)
    stream.buffer, stream.rows = [], 0
    stream._loaded_key, stream._texts = None, None
    stream.identity = identity(
        {
            "files": [(p.name, file_hash(p)) for p in paths],
            "split": split,
            "sequence_len": sequence_len,
            "packing": "bos-bestfit-v1",
        }
    )
    return stream


class FixtureStream:
    """Explicit synthetic fixture only; never labelled ClimbMix or official data."""

    def __init__(self, profile, split):
        self.profile, self.split, self.rows = profile, split, 0
        self.identity = identity({"profile": profile, "split": split})

    def batch(self, batch_size, device):
        import torch

        p = self.profile
        base = torch.arange(batch_size * (p["sequence_len"] + 1)).reshape(
            batch_size, -1
        )
        tokens = (base + self.rows + (7 if self.split == "dev" else 0)) % p[
            "vocab_size"
        ]
        self.rows += batch_size
        return tokens[:, :-1].to(device), tokens[:, 1:].to(device)

    def state_dict(self):
        return {"identity": self.identity, "rows": self.rows}

    def load_state_dict(self, state):
        require(
            state["identity"] == self.identity,
            "incompatible_checkpoint",
            "Stream identity changed.",
        )
        self.rows = state["rows"]


def streams(profile):
    if profile["data"]["kind"] == "synthetic_fixture":
        return (
            FixtureStream(profile, "train"),
            FixtureStream(profile, "dev"),
            [1] * profile["vocab_size"],
        )
    from nanochat.tokenizer import RustBPETokenizer
    import torch

    data = profile["data"]
    tokenizer = RustBPETokenizer.from_directory(Path(data["tokenizer"]))
    require(
        tokenizer.get_vocab_size() == profile["vocab_size"],
        "incompatible_contract",
        "Tokenizer vocabulary differs.",
    )
    return (
        parquet_stream(
            data["train_paths"], tokenizer, profile["sequence_len"], "train"
        ),
        parquet_stream([data["dev_path"]], tokenizer, profile["sequence_len"], "dev"),
        torch.load(data["token_bytes"], weights_only=True).tolist(),
    )
