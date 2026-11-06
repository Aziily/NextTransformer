"""Dense embedding gradients in ATen's accumulation order, sharing one row plan across tables.

ATen sorts the indices stably, sums each token's rows in FP32 in groups of ten, adds the group
sums in order and rounds once. Every table of a model looks up the same indices, so the sort and
the segment bookkeeping are built once per batch here. Tokens of at most ten rows, nearly all of
them, are written directly; ATen sends every token through an FP32 buffer.
"""
import torch
import triton
import triton.language as tl

GROUP = 10


@triton.jit
def _group_sum(G, PERM, COUNT, DEST, W, PARTIAL, D: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    n = tl.load(COUNT+p)
    if n == 0: return
    cols = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    mask = cols < D
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(n):
        row = tl.load(PERM+p+r)
        acc += tl.load(G+row*D+cols, mask=mask, other=0.).to(tl.float32)
    dest = tl.load(DEST+p)
    if dest >= 0:
        tl.store(W+dest*D+cols, (acc+0.).to(tl.bfloat16), mask=mask)
    else:
        tl.store(PARTIAL+(-1-dest)*D+cols, acc, mask=mask)


@triton.jit
def _heavy_sum(PARTIAL, FIRST, GROUPS, TOKEN, W, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    groups = tl.load(GROUPS+i)
    if groups == 0: return
    first = tl.load(FIRST+i)
    cols = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    mask = cols < D
    acc = tl.zeros((BLOCK,), tl.float32)
    for c in range(groups):
        acc += tl.load(PARTIAL+(first+c)*D+cols, mask=mask, other=0.)
    tl.store(W+tl.load(TOKEN+i)*D+cols, acc.to(tl.bfloat16), mask=mask)


class Plan:
    def __init__(self, indices):
        flat = indices.reshape(-1)
        n = flat.numel(); device = flat.device
        tokens, perm = torch.sort(flat, stable=True)
        position = torch.arange(n, device=device)
        first = torch.ones(n, device=device, dtype=torch.bool); first[1:] = tokens[1:] != tokens[:-1]
        last = torch.ones(n, device=device, dtype=torch.bool); last[:-1] = first[1:]
        start = torch.cummax(torch.where(first, position, 0), 0).values
        end = torch.flip(torch.cummin(torch.flip(torch.where(last, position+1, n), (0,)), 0).values, (0,))
        length = end-start
        head = (position-start) % GROUP == 0
        heavy = length > GROUP
        slot = torch.cumsum(head & heavy, 0)-1
        self.perm, self.tokens = perm, tokens
        self.count = torch.where(head, torch.minimum(end-position, torch.full_like(end, GROUP)), 0).to(torch.int32)
        self.dest = torch.where(heavy, -1-slot, tokens)
        # At most n/(GROUP+1) tokens exceed one group, so the heavy list has a static size.
        capacity = n//(GROUP+1)+1
        where = torch.nonzero_static(first & heavy, size=capacity, fill_value=-1).squeeze(1)
        valid = where >= 0
        safe = where.clamp_min(0)
        self.heavy_first = slot[safe]
        self.heavy_groups = torch.where(valid, (length[safe]+GROUP-1)//GROUP, 0).to(torch.int32)
        self.heavy_token = tokens[safe]
        self.partials = n//GROUP+capacity
        self.capacity = capacity
        self.n = n


def dense_gradient(grad, plan, rows):
    d = grad.shape[-1]
    grad = grad.contiguous()
    weight = torch.zeros(rows, d, device=grad.device, dtype=grad.dtype)
    partial = torch.empty(plan.partials, d, device=grad.device, dtype=torch.float32)
    block = 512
    grid = triton.cdiv(d, block)
    _group_sum[(plan.n, grid)](grad, plan.perm, plan.count, plan.dest, weight, partial, D=d, BLOCK=block, num_warps=4)
    _heavy_sum[(plan.capacity, grid)](partial, plan.heavy_first, plan.heavy_groups, plan.heavy_token, weight, D=d, BLOCK=block,
                                      num_warps=4)
    return weight


_cached = (None, None)


def plan_for(indices):
    """All tables of one forward share `indices`; the plan lives until the next batch."""
    global _cached
    if _cached[0] is not indices: _cached = (indices, Plan(indices))
    return _cached[1]


class Lookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.indices, ctx.rows = indices, weight.shape[0]
        return torch.nn.functional.embedding(indices, weight)

    @staticmethod
    def backward(ctx, grad):
        return dense_gradient(grad, plan_for(ctx.indices), ctx.rows), None
