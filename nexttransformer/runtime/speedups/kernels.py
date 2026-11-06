"""Triton kernels that reproduce eager PyTorch BF16 results bit for bit, with fewer memory passes.

Every kernel mirrors the rounding points of the eager op sequence it replaces. Reductions follow
the exact order of ATen's vectorized RMSNorm kernels (torch 2.9): 128 threads, four consecutive
elements per thread and round, a shuffle-down tree inside each warp, then (w0+w2)+(w1+w3). ATen
gives every row its own block, which starves rows as short as a 128-wide attention head; these
kernels keep the order and pack several rows per program.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _bf(x):
    """Round FP32 to nearest-even BF16, the boundary every eager BF16 op has.

    Inline on purpose: the compiler folds a plain truncate/extend pair away.
    """
    return tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; mov.b16 lo, 0; cvt.rn.bf16.f32 hi, $1; mov.b32 $0, {lo, hi}; }",
        "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _quad(v, L: tl.constexpr):
    """Split [R, 4L] whose element 4l+k belongs to lane l into its four [R, L] parts, k = 0..3."""
    R: tl.constexpr = v.shape[0]
    a, b = tl.split(tl.reshape(v, (R, L, 2, 2)))
    k0, k2 = tl.split(a)
    k1, k3 = tl.split(b)
    return k0, k1, k2, k3


@triton.jit
def _sq_lanes(v, s, L: tl.constexpr):
    """s[l] += v[4l]^2, ..., v[4l+3]^2 in order with fused multiply-adds, as ATen's per-thread loop."""
    k0, k1, k2, k3 = _quad(v, L)
    s = libdevice.fma(k0, k0, s)
    s = libdevice.fma(k1, k1, s)
    s = libdevice.fma(k2, k2, s)
    return libdevice.fma(k3, k3, s)


@triton.jit
def _dot_lanes(g, y, rstd, s, L: tl.constexpr):
    """s[l] += g*y*rstd over four elements in order: ATen's backward statistic."""
    g0, g1, g2, g3 = _quad(g, L)
    y0, y1, y2, y3 = _quad(y, L)
    s = libdevice.fma(g0*y0, rstd, s)
    s = libdevice.fma(g1*y1, rstd, s)
    s = libdevice.fma(g2*y2, rstd, s)
    return libdevice.fma(g3*y3, rstd, s)


@triton.jit
def _half(s, W: tl.constexpr):
    """One shuffle-down level: lane i + lane i+W/2 over the last axis of [R, G, W]."""
    R: tl.constexpr = s.shape[0]
    G: tl.constexpr = s.shape[1]
    return tl.sum(tl.reshape(s, (R, G, 2, W//2)), axis=2)


@triton.jit
def _warp_tree(s):
    """[R, G, 32] -> [R, G]: the offsets 16, 8, 4, 2, 1 of a warp shuffle-down reduction."""
    R: tl.constexpr = s.shape[0]
    G: tl.constexpr = s.shape[1]
    s = _half(s, 32); s = _half(s, 16); s = _half(s, 8); s = _half(s, 4); s = _half(s, 2)
    return tl.reshape(s, (R, G))


@triton.jit
def _block_tree(s, L: tl.constexpr):
    """[R, L] lanes -> [R]; L is 32 (one active warp) or 128 (four warps, (w0+w2)+(w1+w3))."""
    R: tl.constexpr = s.shape[0]
    if L == 32:
        return tl.reshape(_warp_tree(tl.reshape(s, (R, 1, 32))), (R,))
    else:
        w = _warp_tree(tl.reshape(s, (R, 4, 32)))
        w = tl.sum(tl.reshape(w, (R, 2, 2)), axis=1)
        return tl.sum(w, axis=1)


@triton.jit
def _rms_fwd(X, Y, RSTD, M, eps, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    s = tl.zeros((ROWS, L), tl.float32)
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        x = tl.load(X+rows[:, None].to(tl.int64)*N+cols[None, :], mask=mask, other=0.).to(tl.float32)
        s = _sq_lanes(x, s, L)
    var = tl.div_rn(_block_tree(s, L), N*1.)
    rstd = libdevice.rsqrt(var+eps)
    tl.store(RSTD+rows, rstd, mask=live)
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        tl.store(Y+offset, (rstd[:, None]*x).to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_bwd(G, X, RSTD, DX, M, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    rstd = tl.load(RSTD+rows, mask=live, other=0.)
    s = tl.zeros((ROWS, L), tl.float32)
    r = tl.broadcast_to(rstd[:, None], (ROWS, L))
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        g = tl.load(G+offset, mask=mask, other=0.).to(tl.float32)
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        s = _dot_lanes(g, x, r, s, L)
    stat = _block_tree(s, L)
    fh: tl.constexpr = N*1.
    term = tl.div_rn(1., fh)*rstd
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        g = tl.load(G+offset, mask=mask, other=0.).to(tl.float32)
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        f = fh*g-libdevice.mul_rn(x*rstd[:, None], tl.broadcast_to(stat[:, None], (ROWS, 4*L)))
        tl.store(DX+offset, (f*term[:, None]).to(tl.bfloat16), mask=mask)


def _shape(n):
    lanes = 32 if n <= 128 else 128
    return lanes, -(-n//(4*lanes))


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n = x.shape[-1]; m = x.numel()//n
        lanes, rounds = _shape(n)
        rows = 16 if n <= 128 else 2
        y = torch.empty_like(x); rstd = torch.empty(m, device=x.device, dtype=torch.float32)
        _rms_fwd[(triton.cdiv(m, rows),)](x, y, rstd, m, torch.finfo(torch.float32).eps, N=n, L=lanes, ROUNDS=rounds,
                                          ROWS=rows, num_warps=4)
        ctx.save_for_backward(x, rstd)
        return y

    @staticmethod
    def backward(ctx, g):
        x, rstd = ctx.saved_tensors
        n = x.shape[-1]; m = x.numel()//n
        lanes, rounds = _shape(n)
        rows = 16 if n <= 128 else 2
        dx = torch.empty_like(x)
        _rms_bwd[(triton.cdiv(m, rows),)](g.contiguous(), x, rstd, dx, m, N=n, L=lanes, ROUNDS=rounds, ROWS=rows, num_warps=4)
        return dx


@triton.jit
def _add_rms_fwd(X, D, S, Y, RSTD, M, eps, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    s = tl.zeros((ROWS, L), tl.float32)
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        v = _bf(tl.load(X+offset, mask=mask, other=0.).to(tl.float32)+tl.load(D+offset, mask=mask, other=0.).to(tl.float32))
        tl.store(S+offset, v.to(tl.bfloat16), mask=mask)
        s = _sq_lanes(v, s, L)
    rstd = libdevice.rsqrt(tl.div_rn(_block_tree(s, L), N*1.)+eps)
    tl.store(RSTD+rows, rstd, mask=live)
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        v = _bf(tl.load(X+offset, mask=mask, other=0.).to(tl.float32)+tl.load(D+offset, mask=mask, other=0.).to(tl.float32))
        tl.store(Y+offset, (rstd[:, None]*v).to(tl.bfloat16), mask=mask)


@triton.jit
def _norm_stat(G, X, rows, rstd, live, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr):
    """ATen's backward statistic of one row, in its order."""
    R: tl.constexpr = rows.shape[0]
    s = tl.zeros((R, L), tl.float32)
    r = tl.broadcast_to(rstd[:, None], (R, L))
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        g = tl.load(G+offset, mask=mask, other=0.).to(tl.float32)
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        s = _dot_lanes(g, x, r, s, L)
    return _block_tree(s, L)


@triton.jit
def _add_rms_bwd(G, R, X, RSTD, DX, M, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr, ROWS: tl.constexpr,
                 NORMED: tl.constexpr):
    """Norm backward plus the residual gradient, which autograd would add in a kernel of its own.

    NORMED: X is already the norm's output, x*rstd rounded to BF16.
    """
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    rstd = tl.load(RSTD+rows, mask=live, other=0.)
    unit = tl.where(live, 1., 0.) if NORMED else rstd
    stat = _norm_stat(G, X, rows, unit, live, N, L, ROUNDS)
    fh: tl.constexpr = N*1.
    term = tl.div_rn(1., fh)*rstd
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        g = tl.load(G+offset, mask=mask, other=0.).to(tl.float32)
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        f = fh*g-libdevice.mul_rn(x*unit[:, None], tl.broadcast_to(stat[:, None], (ROWS, 4*L)))
        d = _bf(f*term[:, None])+tl.load(R+offset, mask=mask, other=0.).to(tl.float32)
        tl.store(DX+offset, d.to(tl.bfloat16), mask=mask)


class AddNorm(torch.autograd.Function):
    """s = x+d and norm(s) in one pass; backward also adds the gradient arriving through s.

    `keep_output` differentiates from norm(s) instead of s, which frees s for ResidualLinear to
    overwrite, at the price of the rounding of norm(s).
    """
    keep_output = False

    @staticmethod
    def forward(ctx, x, d):
        n = x.shape[-1]; m = x.numel()//n
        lanes, rounds = _shape(n)
        s = torch.empty_like(x); y = torch.empty_like(x); rstd = torch.empty(m, device=x.device, dtype=torch.float32)
        _add_rms_fwd[(triton.cdiv(m, 2),)](x, d, s, y, rstd, m, torch.finfo(torch.float32).eps, N=n, L=lanes, ROUNDS=rounds,
                                           ROWS=2, num_warps=4)
        ctx.normed = AddNorm.keep_output
        ctx.save_for_backward(y if ctx.normed else s, rstd)
        return s, y

    @staticmethod
    def backward(ctx, gs, gy):
        s, rstd = ctx.saved_tensors
        n = s.shape[-1]; m = s.numel()//n
        lanes, rounds = _shape(n)
        dx = torch.empty_like(s)
        _add_rms_bwd[(triton.cdiv(m, 2),)](gy.contiguous(), gs.contiguous(), s, rstd, dx, m, N=n, L=lanes, ROUNDS=rounds,
                                           ROWS=2, NORMED=ctx.normed, num_warps=4)
        return dx, dx


@triton.jit
def _mix_rms_fwd(A, X, B_, X0, S, Y, RSTD, M, eps, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    a = _bf(tl.load(A)); b = _bf(tl.load(B_))
    s = tl.zeros((ROWS, L), tl.float32)
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        x0 = tl.load(X0+offset, mask=mask, other=0.).to(tl.float32)
        v = _bf(_bf(a*x)+_bf(b*x0))
        tl.store(S+offset, v.to(tl.bfloat16), mask=mask)
        s = _sq_lanes(v, s, L)
    rstd = libdevice.rsqrt(tl.div_rn(_block_tree(s, L), N*1.)+eps)
    tl.store(RSTD+rows, rstd, mask=live)
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        v = tl.load(S+offset, mask=mask, other=0.).to(tl.float32)
        tl.store(Y+offset, (rstd[:, None]*v).to(tl.bfloat16), mask=mask)


@triton.jit
def _mix_rms_bwd(G, R, A, X, B_, X0, S, RSTD, DX, DX0, PX, PX0, M, N: tl.constexpr, L: tl.constexpr, ROUNDS: tl.constexpr,
                 ROWS: tl.constexpr, PARTIAL: tl.constexpr, ACCUMULATE: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    sa = 0.; sb = 0.
    live = rows < M
    a = _bf(tl.load(A)); b = _bf(tl.load(B_))
    rstd = tl.load(RSTD+rows, mask=live, other=0.)
    stat = _norm_stat(G, S, rows, rstd, live, N, L, ROUNDS)
    fh: tl.constexpr = N*1.
    term = tl.div_rn(1., fh)*rstd
    for c in tl.static_range(ROUNDS):
        cols = c*4*L+tl.arange(0, 4*L)
        mask = live[:, None] & (cols < N)[None, :]
        offset = rows[:, None].to(tl.int64)*N+cols[None, :]
        g = tl.load(G+offset, mask=mask, other=0.).to(tl.float32)
        v = tl.load(S+offset, mask=mask, other=0.).to(tl.float32)
        f = fh*g-libdevice.mul_rn(v*rstd[:, None], tl.broadcast_to(stat[:, None], (ROWS, 4*L)))
        t = _bf(_bf(f*term[:, None])+tl.load(R+offset, mask=mask, other=0.).to(tl.float32))
        x = tl.load(X+offset, mask=mask, other=0.).to(tl.float32)
        x0 = tl.load(X0+offset, mask=mask, other=0.).to(tl.float32)
        tl.store(DX+offset, (t*a).to(tl.bfloat16), mask=mask)
        if ACCUMULATE:
            tl.store(DX0+offset, (tl.load(DX0+offset, mask=mask, other=0.).to(tl.float32)+_bf(t*b)).to(tl.bfloat16), mask=mask)
        else:
            tl.store(DX0+offset, (t*b).to(tl.bfloat16), mask=mask)
        if PARTIAL:
            sa += tl.sum(_bf(t*x)); sb += tl.sum(_bf(t*x0))
        else:
            tl.store(PX+offset, (t*x).to(tl.bfloat16), mask=mask)
            tl.store(PX0+offset, (t*x0).to(tl.bfloat16), mask=mask)
    if PARTIAL:
        tl.store(PX+tl.program_id(0), sa)
        tl.store(PX0+tl.program_id(0), sb)


class GradSum:
    """The gradient of x0 across the blocks that share it, kept here instead of in autograd's buffer."""
    __slots__ = ("total", "pending")

    def __init__(self):
        self.total, self.pending = None, 0


class MixNorm(torch.autograd.Function):
    """ResidualMix followed by the block's first norm; returns both. Same argument order as ResidualMix.

    With a GradSum, each block adds its x0 gradient in place, in the order autograd would add them,
    and the last one hands over the total.

    `partial_sums` reduces the two scalar gradients in the kernel, which saves two full-size tensors
    but leaves ATen's summation order, so those scalars can differ in the last BF16 place.
    """
    partial_sums = False

    @staticmethod
    def forward(ctx, b, x0, a, x, shared=None):
        n = x.shape[-1]; m = x.numel()//n
        lanes, rounds = _shape(n)
        ctx.shared = shared
        if shared is not None: shared.pending += 1
        s = torch.empty_like(x); y = torch.empty_like(x); rstd = torch.empty(m, device=x.device, dtype=torch.float32)
        _mix_rms_fwd[(triton.cdiv(m, 2),)](a, x, b, x0, s, y, rstd, m, torch.finfo(torch.float32).eps, N=n, L=lanes,
                                           ROUNDS=rounds, ROWS=2, num_warps=4)
        ctx.save_for_backward(b, x0, a, x, s, rstd)
        return s, y

    @staticmethod
    def backward(ctx, gs, gy):
        b, x0, a, x, s, rstd = ctx.saved_tensors
        n = x.shape[-1]; m = x.numel()//n
        lanes, rounds = _shape(n)
        shared = ctx.shared
        accumulate = shared is not None and shared.total is not None
        dx, dx0 = torch.empty_like(x), shared.total if accumulate else torch.empty_like(x)
        partial = MixNorm.partial_sums
        grid = triton.cdiv(m, 2)
        px, px0 = (torch.empty(grid, device=x.device, dtype=torch.float32) if partial else torch.empty_like(x) for _ in range(2))
        _mix_rms_bwd[(grid,)](gy.contiguous(), gs.contiguous(), a, x, b, x0, s, rstd, dx, dx0, px, px0, m, N=n,
                              L=lanes, ROUNDS=rounds, ROWS=2, PARTIAL=partial, ACCUMULATE=accumulate, num_warps=4)
        if shared is not None:
            shared.total = dx0; shared.pending -= 1
            if shared.pending: dx0 = None
        if partial: return px0.sum().bfloat16().float(), dx0, px.sum().bfloat16().float(), dx, None
        return px0.sum().float(), dx0, px.sum().float(), dx, None


@triton.jit
def _sum_paths(V, K, Q, S, Y, n, C: tl.constexpr, CH: tl.constexpr, B: tl.constexpr):
    """The input gradient of the projections, added in the order autograd receives the paths.

    S is the gate's gradient for the first CH columns; eager pads it with zeros and adds it first,
    which also turns -0 into +0.
    """
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    t = tl.load(V+i, mask=m, other=0.).to(tl.float32)
    if CH > 0:
        col = i % C
        t = _bf(tl.load(S+(i//C)*CH+col, mask=m & (col < CH), other=0.).to(tl.float32)+t)
    t = _bf(t+tl.load(K+i, mask=m, other=0.).to(tl.float32))
    tl.store(Y+i, (t+tl.load(Q+i, mask=m, other=0.).to(tl.float32)).to(tl.bfloat16), mask=m)


def weight_gradient(g, x):
    """g^T x straight out of the GEMM in FP32: no BF16 rounding and no widening pass."""
    return torch.mm(g.t(), x, out_dtype=torch.float32)


class WideLinear(torch.autograd.Function):
    """F.linear(x, cast) for the FP32 parameter `weight` that `cast` was made from."""
    @staticmethod
    def forward(ctx, x, weight, cast):
        ctx.save_for_backward(x, cast)
        return torch.nn.functional.linear(x, cast)

    @staticmethod
    def backward(ctx, g):
        x, cast = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        return g2.mm(cast).view_as(x), weight_gradient(g2, x.reshape(-1, x.shape[-1])), None


class ResidualLinear(torch.autograd.Function):
    """s+F.linear(h, cast), written over s: the add happens inside the GEMM, without a copy of s."""
    @staticmethod
    def forward(ctx, s, h, weight, cast):
        ctx.save_for_backward(h, cast)
        ctx.mark_dirty(s)
        s.view(-1, s.shape[-1]).addmm_(h.view(-1, h.shape[-1]), cast.t())
        return s

    @staticmethod
    def backward(ctx, g):
        h, cast = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        return g, g2.mm(cast).view_as(h), weight_gradient(g2, h.view(-1, h.shape[-1])), None


class Projections(torch.autograd.Function):
    """c_q(x), c_k(x), c_v(x) and optionally ve_gate(x[..., :channels]) from BF16 weights.

    The products are the ones autograd would run. With `casts`, the weights are the FP32 parameters
    and `casts` their BF16 copies: the path gradients accumulate inside the GEMMs and the weight
    gradients leave them in FP32, one rounding instead of one per path, so close but not identical.
    """
    @staticmethod
    def forward(ctx, x, wq, wk, wv, wg, channels, casts):
        ctx.channels, ctx.casts = channels, casts
        ctx.save_for_backward(x, *((wq, wk, wv, wg) if casts is None else casts))
        cq, ck, cv, cg = (wq, wk, wv, wg) if casts is None else casts
        out = tuple(torch.nn.functional.linear(x, w) for w in (cq, ck, cv))
        if cg is None: return out
        return (*out, torch.nn.functional.linear(x[..., :channels], cg))

    @staticmethod
    def backward(ctx, gq, gk, gv, gg=None):
        x, wq, wk, wv, wg = ctx.saved_tensors
        c = x.shape[-1]; x2 = x.view(-1, c); ch = ctx.channels
        gq, gk, gv = (g.reshape(-1, g.shape[-1]) for g in (gq, gk, gv))
        dgate = None
        if wg is not None:
            gg = gg.reshape(-1, gg.shape[-1]); xs = x2[:, :ch]
            small = gg.mm(wg); dgate = gg.t().mm(xs)
        dx = gv.mm(wv)
        if ctx.casts is not None:
            dx.addmm_(gk, wk).addmm_(gq, wq)
            if wg is not None: dx[:, :ch] += small
            return (dx.view_as(x), weight_gradient(gq, x2), weight_gradient(gk, x2), weight_gradient(gv, x2),
                    None if dgate is None else dgate.float(), None, None)
        total = torch.empty_like(dx); n = dx.numel()
        _sum_paths[(triton.cdiv(n, 8192),)](dx, gk.mm(wk), gq.mm(wq), small if wg is not None else dx, total, n, C=c,
                                            CH=ch if wg is not None else 0, B=8192, num_warps=8)
        return total.view_as(x), gq.t().mm(x2), gk.t().mm(x2), gv.t().mm(x2), dgate, None, None


@triton.jit
def _widen(G, Y, n, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    tl.store(Y+i, tl.load(G+i, mask=m, other=0.).to(tl.float32), mask=m)


class WeightCast(torch.autograd.Function):
    """`weight.to(bf16)` whose forward reuses a cast made earlier in the update.

    One node per use keeps eager's gradients: each use casts its own gradient to FP32, and repeated
    uses of a weight add in FP32 in the accumulator, not in BF16 at one shared cast.
    """
    @staticmethod
    def forward(ctx, weight, cast):
        ctx.dtype = weight.dtype
        return cast.view_as(cast)

    @staticmethod
    def backward(ctx, g):
        if g.dtype != torch.bfloat16 or ctx.dtype != torch.float32 or not g.is_contiguous(): return g.to(ctx.dtype), None
        y = torch.empty(g.shape, device=g.device, dtype=torch.float32)  # ATen's dtype-converting copy is 4x slower
        _widen[(triton.cdiv(g.numel(), 8192),)](g, y, g.numel(), B=8192, num_warps=8)
        return y, None


@triton.jit
def _r(x, EXACT: tl.constexpr):
    """An intermediate BF16 rounding of eager, which `fast_rope` leaves out."""
    if EXACT: return _bf(x)
    return x


@triton.jit
def _rope(x1, x2, c, s, EXACT: tl.constexpr):
    """Eager apply_rotary_emb on BF16 halves, with every intermediate rounded."""
    y1 = _r(_r(x1*c, EXACT)+_r(x2*s, EXACT), EXACT)
    y2 = _r(_r(x1*(-s), EXACT)+_r(x2*c, EXACT), EXACT)
    return y1, y2


@triton.jit
def _rope_rstd(y1, y2, eps, D: tl.constexpr):
    """rstd of a head of width D=128. Lanes 0-15 hold y1 and 16-31 y2, so offset 16 pairs y1 with y2."""
    H: tl.constexpr = D//2
    R: tl.constexpr = y1.shape[0]
    a = _sq_lanes(y1, tl.zeros((R, H//4), tl.float32), H//4)
    b = _sq_lanes(y2, tl.zeros((R, H//4), tl.float32), H//4)
    t = tl.reshape(a+b, (R, 1, 16))
    t = _half(t, 16); t = _half(t, 8); t = _half(t, 4); t = _half(t, 2)
    return libdevice.rsqrt(tl.div_rn(tl.reshape(t, (R,)), D*1.)+eps)


@triton.jit
def _rope_norm_fwd(X, C, S, Y, RSTD, M, T, HEADS, eps, scale, D: tl.constexpr, ROWS: tl.constexpr, EXACT: tl.constexpr):
    H: tl.constexpr = D//2
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    cols = tl.arange(0, H)
    position = (rows//HEADS) % T
    mask = live[:, None]
    base = rows[:, None].to(tl.int64)*D+cols[None, :]
    x1 = tl.load(X+base, mask=mask, other=0.).to(tl.float32)
    x2 = tl.load(X+base+H, mask=mask, other=0.).to(tl.float32)
    table = position[:, None]*H+cols[None, :]
    c = tl.load(C+table, mask=mask, other=0.).to(tl.float32)
    s = tl.load(S+table, mask=mask, other=0.).to(tl.float32)
    y1, y2 = _rope(x1, x2, c, s, EXACT)
    rstd = _rope_rstd(y1, y2, eps, D)
    tl.store(RSTD+rows, rstd, mask=live)
    tl.store(Y+base, (_r(rstd[:, None]*y1, EXACT)*scale).to(tl.bfloat16), mask=mask)
    tl.store(Y+base+H, (_r(rstd[:, None]*y2, EXACT)*scale).to(tl.bfloat16), mask=mask)


@triton.jit
def _rope_norm_bwd(G, X, C, S, RSTD, DX, M, T, HEADS, scale, D: tl.constexpr, ROWS: tl.constexpr, EXACT: tl.constexpr):
    H: tl.constexpr = D//2
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    cols = tl.arange(0, H)
    position = (rows//HEADS) % T
    mask = live[:, None]
    base = rows[:, None].to(tl.int64)*D+cols[None, :]
    x1 = tl.load(X+base, mask=mask, other=0.).to(tl.float32)
    x2 = tl.load(X+base+H, mask=mask, other=0.).to(tl.float32)
    table = position[:, None]*H+cols[None, :]
    c = tl.load(C+table, mask=mask, other=0.).to(tl.float32)
    s = tl.load(S+table, mask=mask, other=0.).to(tl.float32)
    y1, y2 = _rope(x1, x2, c, s, EXACT)
    rstd = tl.load(RSTD+rows, mask=live, other=0.)
    g1 = _r(tl.load(G+base, mask=mask, other=0.).to(tl.float32)*scale, EXACT)
    g2 = _r(tl.load(G+base+H, mask=mask, other=0.).to(tl.float32)*scale, EXACT)
    r = tl.broadcast_to(rstd[:, None], (ROWS, H//4))
    a = _dot_lanes(g1, y1, r, tl.zeros((ROWS, H//4), tl.float32), H//4)
    b = _dot_lanes(g2, y2, r, tl.zeros((ROWS, H//4), tl.float32), H//4)
    t = tl.reshape(a+b, (ROWS, 1, 16))
    t = _half(t, 16); t = _half(t, 8); t = _half(t, 4); t = _half(t, 2)
    stat = tl.reshape(t, (ROWS,))[:, None]
    fh: tl.constexpr = D*1.
    term = (tl.div_rn(1., fh)*rstd)[:, None]
    r2 = rstd[:, None]
    st = tl.broadcast_to(stat, (ROWS, H))
    d1 = _r((fh*g1-libdevice.mul_rn(y1*r2, st))*term, EXACT)
    d2 = _r((fh*g2-libdevice.mul_rn(y2*r2, st))*term, EXACT)
    # Eager reaches x through two zero-padded slices, whose sum turns -0 into +0.
    dx1 = _r(_r(d2*(-s), EXACT)+_r(d1*c, EXACT), EXACT)+0.
    dx2 = _r(_r(d2*c, EXACT)+_r(d1*s, EXACT), EXACT)+0.
    tl.store(DX+base, dx1.to(tl.bfloat16), mask=mask)
    tl.store(DX+base+H, dx2.to(tl.bfloat16), mask=mask)


class RopeNormScale(torch.autograd.Function):
    """norm(apply_rotary_emb(x, cos, sin))*scale for x of shape (B, T, H, D).

    `exact` keeps eager's rounding after each op; without it the kernels round once, at the end.
    """
    exact = True

    @staticmethod
    def forward(ctx, x, cos, sin, scale):
        b, t, h, d = x.shape; m = b*t*h
        y = torch.empty_like(x); rstd = torch.empty(m, device=x.device, dtype=torch.float32)
        rows = 16
        _rope_norm_fwd[(triton.cdiv(m, rows),)](x, cos, sin, y, rstd, m, t, h, torch.finfo(torch.float32).eps, scale, D=d,
                                                ROWS=rows, EXACT=RopeNormScale.exact, num_warps=4)
        ctx.save_for_backward(x, cos, sin, rstd); ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, g):
        x, cos, sin, rstd = ctx.saved_tensors
        b, t, h, d = x.shape; m = b*t*h
        dx = torch.empty_like(x)
        rows = 16
        _rope_norm_bwd[(triton.cdiv(m, rows),)](g.contiguous(), x, cos, sin, rstd, dx, m, t, h, ctx.scale, D=d, ROWS=rows,
                                                EXACT=RopeNormScale.exact, num_warps=4)
        return dx, None, None, None


@triton.jit
def _relu2_fwd(H, Y, n, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    h = tl.load(H+i, mask=m, other=0.).to(tl.float32)
    r = tl.where(h < 0., 0., h)
    tl.store(Y+i, (r*r).to(tl.bfloat16), mask=m)


@triton.jit
def _relu2_bwd(G, H, DH, n, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    h = tl.load(H+i, mask=m, other=0.).to(tl.float32)
    g = tl.load(G+i, mask=m, other=0.).to(tl.float32)
    r = tl.where(h < 0., 0., h)
    tl.store(DH+i, tl.where(r <= 0., 0., g*_bf(r*2.)).to(tl.bfloat16), mask=m)


class ReluSquare(torch.autograd.Function):
    """F.relu(h).square(): one pass each way; keeps h instead of writing relu(h)."""
    @staticmethod
    def forward(ctx, h):
        y = torch.empty_like(h); n = h.numel()
        _relu2_fwd[(triton.cdiv(n, 8192),)](h, y, n, B=8192, num_warps=8)
        ctx.save_for_backward(h)
        return y

    @staticmethod
    def backward(ctx, g):
        (h,) = ctx.saved_tensors
        dh = torch.empty_like(h); n = h.numel()
        _relu2_bwd[(triton.cdiv(n, 8192),)](g.contiguous(), h, dh, n, B=8192, num_warps=8)
        return dh


@triton.jit
def _mix_fwd(A, X, B_, X0, Y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    m = i < n
    a = _bf(tl.load(A)); b = _bf(tl.load(B_))
    x = tl.load(X+i, mask=m, other=0.).to(tl.float32)
    x0 = tl.load(X0+i, mask=m, other=0.).to(tl.float32)
    tl.store(Y+i, (_bf(a*x)+_bf(b*x0)).to(tl.bfloat16), mask=m)


@triton.jit
def _mix_bwd(G, A, X, B_, X0, DX, DX0, PX, PX0, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    m = i < n
    a = _bf(tl.load(A)); b = _bf(tl.load(B_))
    g = tl.load(G+i, mask=m, other=0.).to(tl.float32)
    x = tl.load(X+i, mask=m, other=0.).to(tl.float32)
    x0 = tl.load(X0+i, mask=m, other=0.).to(tl.float32)
    tl.store(DX+i, (g*a).to(tl.bfloat16), mask=m)
    tl.store(DX0+i, (g*b).to(tl.bfloat16), mask=m)
    tl.store(PX+i, (g*x).to(tl.bfloat16), mask=m)
    tl.store(PX0+i, (g*x0).to(tl.bfloat16), mask=m)


class ResidualMix(torch.autograd.Function):
    """b*x0 + a*x for 0-dim FP32 parameters a, b and BF16 streams.

    Arguments put x0 first so autograd delivers the two gradients in eager's order when x is x0.
    Scalar gradients go through ATen's own sum, whose reduction order is not reproduced here.
    """
    @staticmethod
    def forward(ctx, b, x0, a, x):
        y = torch.empty_like(x); n = x.numel()
        _mix_fwd[(triton.cdiv(n, 8192),)](a, x, b, x0, y, n, BLOCK=8192, num_warps=8)
        ctx.save_for_backward(b, x0, a, x)
        return y

    @staticmethod
    def backward(ctx, g):
        b, x0, a, x = ctx.saved_tensors
        n = x.numel()
        dx, dx0, px, px0 = (torch.empty_like(x) for _ in range(4))
        _mix_bwd[(triton.cdiv(n, 8192),)](g.contiguous(), a, x, b, x0, dx, dx0, px, px0, n, BLOCK=8192, num_warps=8)
        return px0.sum().float(), dx0, px.sum().float(), dx


INV_CAP = float(torch.tensor(1., dtype=torch.float32)/torch.tensor(15., dtype=torch.float32))


@triton.jit
def _cap_fwd(L, C, n, inv, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    x = tl.load(L+i, mask=m, other=0.).to(tl.float32)
    t = x*inv
    # libdevice flushes subnormals here; tanh(t) rounds to t long before that.
    tl.store(C+i, 15.*tl.where(tl.abs(t) < 1e-30, t, libdevice.tanh(t)), mask=m)


@triton.jit
def _cap_ce_bwd(O, L, T, GT, D, V, inv, B: tl.constexpr):
    """Backward of nll(log_softmax(15*tanh(L/15))) for one row and column block, in one pass."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1)*B+tl.arange(0, B)
    m = cols < V
    target = tl.load(T+row)
    g = tl.load(GT+row)
    # nll backward leaves one nonzero per row, so ATen's row sum of the gradient is exactly -g.
    total = tl.where(target == -1, 0., -g)
    o = tl.load(O+row*V+cols, mask=m, other=0.)
    grad = tl.where(cols == target, total, 0.)
    d = libdevice.fma(-libdevice.exp(o), tl.broadcast_to(total, (B,)), grad)
    x = tl.load(L+row*V+cols, mask=m, other=0.).to(tl.float32)
    z = libdevice.tanh(x*inv)
    d = (d*15.)*libdevice.fma(-z, z, 1.)
    tl.store(D+row*V+cols, (d*inv).to(tl.bfloat16), mask=m)


@triton.jit
def _cap(x, inv):
    t = x*inv
    z = libdevice.tanh(t)
    return 15.*tl.where(tl.abs(t) < 1e-30, t, z), z


@triton.jit
def _ce_fwd(L, T, TOP, LOGSUM, PICKED, inv, V: tl.constexpr, LANES: tl.constexpr):
    """ATen's log_softmax of one capped row, reading BF16 logits once and storing only its statistics.

    Order of cunn_SoftMaxForward with ILP 4 and 1024 threads: each thread adds its four consecutive
    elements per round, then a shuffle-down tree in each warp and one more across the 32 warps.
    The cap is monotone on BF16 values, so the capped maximum is the cap of the maximum.
    """
    W: tl.constexpr = 4*LANES
    row = tl.program_id(0).to(tl.int64)
    base = L+row*V
    top = tl.full((W,), float("-inf"), tl.float32)
    for c in tl.static_range(V//W):
        top = tl.maximum(top, tl.load(base+c*W+tl.arange(0, W)).to(tl.float32))
    high, _ = _cap(tl.max(top, 0), inv)
    s = tl.zeros((1, LANES), tl.float32)
    for c in tl.static_range(V//W):
        capped, _ = _cap(tl.load(base+c*W+tl.arange(0, W)).to(tl.float32), inv)
        k0, k1, k2, k3 = _quad(tl.reshape(libdevice.exp(capped-high), (1, W)), LANES)
        s = s+k0; s = s+k1; s = s+k2; s = s+k3
    warps = _warp_tree(tl.reshape(s, (1, LANES//32, 32)))
    logsum = libdevice.log(tl.sum(_warp_tree(tl.reshape(warps, (1, 1, 32)))))
    picked, _ = _cap(tl.load(base+tl.maximum(tl.load(T+row), 0)).to(tl.float32), inv)
    tl.store(TOP+row, high)
    tl.store(LOGSUM+row, logsum)
    tl.store(PICKED+row, picked-high-logsum)


@triton.jit
def _ce_bwd(L, T, GT, TOP, LOGSUM, D, V, inv, B: tl.constexpr):
    """_cap_ce_bwd with the log probability recomputed from the logits instead of read back."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1)*B+tl.arange(0, B)
    target = tl.load(T+row)
    g = tl.load(GT+row)
    total = tl.where(target == -1, 0., -g)
    capped, z = _cap(tl.load(L+row*V+cols).to(tl.float32), inv)
    o = capped-tl.load(TOP+row)-tl.load(LOGSUM+row)
    grad = tl.where(cols == target, total, 0.)
    d = libdevice.fma(-libdevice.exp(o), tl.broadcast_to(total, (B,)), grad)
    d = (d*15.)*libdevice.fma(-z, z, 1.)
    tl.store(D+row*V+cols, (d*inv).to(tl.bfloat16))


ONE_PASS_VOCAB = 32768  # ATen's plain cunn_SoftMaxForward with 1024 threads, and a whole number of rounds


class CappedCrossEntropy(torch.autograd.Function):
    """Cross entropy of 15*tanh(logits/15) from BF16 logits with ignore_index -1.

    A 32768-wide row is reduced here in ATen's order; other widths keep ATen's log_softmax with
    everything elementwise around it fused. ATen keeps the loss reduction either way. `mean` selects
    F.cross_entropy's mean over per-token losses.
    """
    @staticmethod
    def forward(ctx, logits, targets, mean=False):
        n = logits.numel(); v = logits.shape[-1]; m = n//v
        flat_targets = targets.reshape(-1)
        ctx.mean = mean
        if mean: ctx.count = (flat_targets != -1).sum().float()
        if v == ONE_PASS_VOCAB:
            top, logsum, picked = (torch.empty(m, device=logits.device, dtype=torch.float32) for _ in range(3))
            _ce_fwd[(m,)](logits, flat_targets, top, logsum, picked, INV_CAP, V=v, LANES=1024, num_warps=32, enable_fp_fusion=False)
            ctx.save_for_backward(top, logits, targets, logsum)
            # ATen still reduces the loss: one class per row gives it the same terms in the same order.
            logp, flat_targets = picked.view(m, 1), torch.where(flat_targets == -1, flat_targets, 0)
        else:
            capped = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
            _cap_fwd[(triton.cdiv(n, 8192),)](logits, capped, n, INV_CAP, B=8192, num_warps=8)
            logp = torch.log_softmax(capped.view(-1, v), -1)
            del capped
            ctx.save_for_backward(logp, logits, targets)
        if mean: return torch.nn.functional.nll_loss(logp, flat_targets, ignore_index=-1, reduction="mean")
        return torch.nn.functional.nll_loss(logp, flat_targets, ignore_index=-1, reduction="none").reshape_as(targets)

    @staticmethod
    def backward(ctx, g):
        first, logits, targets, *rest = ctx.saved_tensors
        v = logits.shape[-1]; m = logits.numel()//v
        d = torch.empty_like(logits)
        rows = (g/ctx.count).expand(m).contiguous() if ctx.mean else g.contiguous().view(-1)
        if rest:
            _ce_bwd[(m, triton.cdiv(v, 2048))](logits, targets.reshape(-1), rows, first, rest[0], d, v, INV_CAP, B=2048,
                                               num_warps=4, enable_fp_fusion=False)
        else:
            _cap_ce_bwd[(m, triton.cdiv(v, 8192))](first, logits, targets.reshape(-1), rows, d, v, INV_CAP, B=8192, num_warps=8)
        return d, None, None


@triton.jit
def _value_fwd(V, E, GATE, Y, n, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    v = tl.load(V+i, mask=m, other=0.).to(tl.float32)
    e = tl.load(E+i, mask=m, other=0.).to(tl.float32)
    g = tl.load(GATE+i//D, mask=m, other=0.).to(tl.float32)
    tl.store(Y+i, (v+_bf(g*e)).to(tl.bfloat16), mask=m)


@triton.jit
def _value_bwd(G, E, GATE, DE, P, n, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B)
    m = i < n
    d = tl.load(G+i, mask=m, other=0.).to(tl.float32)
    e = tl.load(E+i, mask=m, other=0.).to(tl.float32)
    g = tl.load(GATE+i//D, mask=m, other=0.).to(tl.float32)
    tl.store(DE+i, (d*g).to(tl.bfloat16), mask=m)
    tl.store(P+i, (d*e).to(tl.bfloat16), mask=m)


@triton.jit
def _sum128(p):
    """ATen's BF16 sum over a row of 128: 32 threads each add x[l], x[l+32], x[l+64], x[l+96], then offsets 1, 2, 4, 8, 16."""
    R: tl.constexpr = p.shape[0]
    a, b = tl.split(tl.permute(tl.reshape(p, (R, 2, 2, 32)), (0, 3, 1, 2)))
    t0, t2 = tl.split(a)
    t1, t3 = tl.split(b)
    s = ((t0+t1)+t2)+t3
    s = tl.sum(tl.reshape(s, (R, 16, 2)), axis=2)
    s = tl.sum(tl.reshape(s, (R, 8, 2)), axis=2)
    s = tl.sum(tl.reshape(s, (R, 4, 2)), axis=2)
    s = tl.sum(tl.reshape(s, (R, 2, 2)), axis=2)
    return tl.sum(s, axis=1)


@triton.jit
def _value_bwd_rows(G, E, GATE, DE, DG, M, ROWS: tl.constexpr):
    rows = tl.program_id(0)*ROWS+tl.arange(0, ROWS)
    live = rows < M
    offset = rows[:, None].to(tl.int64)*128+tl.arange(0, 128)[None, :]
    mask = live[:, None]
    d = tl.load(G+offset, mask=mask, other=0.).to(tl.float32)
    e = tl.load(E+offset, mask=mask, other=0.).to(tl.float32)
    g = tl.load(GATE+rows, mask=live, other=0.).to(tl.float32)
    tl.store(DE+offset, (d*g[:, None]).to(tl.bfloat16), mask=mask)
    tl.store(DG+rows, _sum128(_bf(d*e)).to(tl.bfloat16), mask=live)


class ValueResidual(torch.autograd.Function):
    """v+gate.unsqueeze(-1)*e for v, e of shape (B, T, H, D). Other than D = 128, ATen keeps the gate's sum over D."""
    @staticmethod
    def forward(ctx, v, gate, e):
        y = torch.empty_like(v); n = v.numel(); d = v.shape[-1]
        _value_fwd[(triton.cdiv(n, 8192),)](v, e, gate, y, n, D=d, B=8192, num_warps=8)
        ctx.save_for_backward(gate, e)
        return y

    @staticmethod
    def backward(ctx, g):
        gate, e = ctx.saved_tensors
        n = e.numel(); d = e.shape[-1]
        g = g.contiguous()
        if d == 128:
            m = n//d
            de, dgate = torch.empty_like(e), torch.empty_like(gate)
            _value_bwd_rows[(triton.cdiv(m, 16),)](g, e, gate, de, dgate, m, ROWS=16, num_warps=4)
            return g, dgate, de
        de, p = torch.empty_like(e), torch.empty_like(e)
        _value_bwd[(triton.cdiv(n, 8192),)](g, e, gate, de, p, n, D=d, B=8192, num_warps=8)
        return g, p.sum(-1), de
