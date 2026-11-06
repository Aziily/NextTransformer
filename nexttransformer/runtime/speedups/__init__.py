"""Opt-in execution speedups for components every architecture shares.

Nothing here changes what a method computes. Each patch in `exact` replaces an eager op sequence
with fused kernels that reproduce the same BF16 results bit for bit, so trained weights match the
reference path, and it applies only to CUDA BF16 tensors in training; everything else keeps the
original code. The shared model files stay untouched and this package is never imported by the cost
audit, so published proofs and fingerprints stay valid.

    python -m nexttransformer.train ... --speedups all      # exact, plus cudnn and merge
    python -m nexttransformer.train ... --speedups exact    # bit for bit
    python -m nexttransformer.train ... --speedups norm,mlp,attention,residual,readout,embedding

`install` refuses to run when a patched function no longer has the source it was written against.
Speedups are off unless requested. Formal runs accept only selections inside `exact`; the trainer records
the selection, and the audit and the exported result carry it.
"""
import hashlib
import inspect
import sys
import textwrap

EXACT = ("norm", "attention", "mlp", "residual", "readout", "embedding", "data", "weights", "addnorm")
# Same function, other rounding: not bit-identical, so outside `exact`. See README.md.
ALL = EXACT+("cudnn", "merge")
# Sources the patches were written against, by hash of the dedented function source.
REVIEWED = {
    "backbone.norm": "0a5b8f1719fa1e2bec21364e40102e2f1a9526a39975dce292d14085505ea976",
    "backbone.apply_rotary_emb": "c11c9baedaea4c6705ca8d0ffa817744a19dfcd2ca5f247d8c1ea5566498f762",
    "backbone.CausalSelfAttention.forward": "312cff3f00f7f8abaf6066858ec132eaa71b5609d1cadb6db2eb533004f1fa52",
    "backbone.MLP.forward": "1623a312ffb0836d43aaaf411e5117c6799a7a11a46f935e47f450244c7cc1a3",
    "backbone.GPT.forward": "493b8f80494b960f30ccfbd16c7aaf89eb9aeff769105be9a516ff3039edff17",
    "backbone.Block.forward": "6d600bf665842b773ac42c52ef493c47995020a31dafb17b24d837bf7b65ef82",
    "backbone.Linear.forward": "df0785e531d0742af7637b979957e46d87f48d4550fd97a725eb0a457d54a30f",
    "lm.ReferenceLM.set_step": "73062e356751e0ae956b0e22a1bfa3185c8677351b9a9904bbc6eba9f8024df9",
    "lm.FlexibleMLP.forward": "e94e1bf3ff2d97917184e1bc6e6b24bee0f45296fcd8270d8bc872fa66b9695d",
    "lm.ReferenceLM._block": "f92bf86582e18803fab8613a2e57098d4adca0439d3ac4ffa961798f4e5f26af",
    "lm.ReferenceLM._readout": "c1031e09bcd2d804bc669de3f1edb3d80f996e3cfb294768587197f59be1ca21",
    "lm.ReferenceLM._token_loss": "eb6e6b3e82a59821ed0660679cd0d64f335ae8c78013c02fc99cc0315c93874a",
    "sparse.SparseMixer._projections": "771071dfe8f55c5c0fa5a9d267362231fb3b64cb256ad6fe6f315611e9c8f502",
}
_installed = None
COUNTS = {}


def active(name):
    return _installed is not None and name in _installed


def _count(name):
    COUNTS[name] = COUNTS.get(name, 0)+1


def source_hash(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()


def reviewed_functions():
    from nexttransformer.models import backbone, lm, sparse
    return {"backbone.norm": backbone.norm, "backbone.apply_rotary_emb": backbone.apply_rotary_emb,
            "backbone.CausalSelfAttention.forward": backbone.CausalSelfAttention.forward,
            "backbone.MLP.forward": backbone.MLP.forward, "backbone.GPT.forward": backbone.GPT.forward,
            "backbone.Block.forward": backbone.Block.forward, "backbone.Linear.forward": backbone.Linear.forward,
            "lm.ReferenceLM.set_step": lm.ReferenceLM.set_step,
    "lm.FlexibleMLP.forward": lm.FlexibleMLP.forward, "lm.ReferenceLM._block": lm.ReferenceLM._block,
            "lm.ReferenceLM._readout": lm.ReferenceLM._readout, "lm.ReferenceLM._token_loss": lm.ReferenceLM._token_loss,
            "sparse.SparseMixer._projections": sparse.SparseMixer._projections}


def selection(spec):
    """The patch names a comma list selects, with `all` and `exact` expanded."""
    names = {name for name in spec.split(",") if name}
    if "all" in names: names = (names-{"all"}) | set(ALL)
    if "exact" in names: names = (names-{"exact"}) | set(EXACT)
    if "merge" in names: names.add("weights")  # its weight gradients leave through the update's cast
    if names-set(ALL): raise ValueError(f"unknown speedups: {sorted(names-set(ALL))}")
    return names


def formal(spec):
    """Whether a formal run accepts the selection: only patches that reproduce the default path bit for bit."""
    try: return selection(spec) <= set(EXACT)
    except ValueError: return False


def install(spec="all"):
    """Activate the named patches once per process; returns the sorted active names."""
    global _installed
    names = selection(spec)
    if _installed is not None:
        if _installed != names: raise RuntimeError("speedups are already installed with another selection")
        return sorted(names)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("speedups need CUDA")
    if torch.__version__.split("+")[0] != "2.9.1": raise RuntimeError("speedups were verified against torch 2.9.1")
    changed = [name for name, function in reviewed_functions().items() if source_hash(function) != REVIEWED[name]]
    if changed: raise RuntimeError(f"patched sources changed: {changed}; review the speedups before enabling them")
    from nexttransformer.runtime.speedups import patches
    patches.apply(names)
    _installed = names
    return sorted(names)
