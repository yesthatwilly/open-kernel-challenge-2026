# okc: problem=muse_bf16_proj
"""Router as one fused Triton hi/lo GEMM with a 3-D split-K grid at every m (hypothesis
router-triton-hilo-fused-control-splitk-all-m, the control for router-triton-hilo-fused).

Restack (hypothesis restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128): on top of the
swap-AB master, three row wins each gated to exactly the cells they won: unpooler slice m <= 8
(CUDA cluster/DSMEM, ouns_cl, behind a graph-replay gate), router m <= 8 (CUDA cluster/DSMEM,
router_cluster_ok) and pooler/unpooler 65 <= m <= 128 (NN-layout torch.mm, nn_ok). Every flag
lives in the state dict; prepare() A/B/A/B-checks the whole dispatch on two states.

The four bf16 layers go through the same cuBLAS dispatch SGLang uses on H200
(`_bf16_gemm_dispatch_impl`), untouched except the NN cells above. The fp32 router runs, at every m, as ONE Triton kernel
over a bf16 hi + lo split of its weight with fp32 accumulation (allowed by the rules at any
batch size: "the router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation").
The split is built once in prepare() from the weights alone and stored as two [256, 8192] bf16
row-major tensors; nothing derived from the inputs is kept between calls.

Swap-AB router (hypothesis router-swapab-wgmma-all-m): _router_swapab_kernel puts 64 hi / lo
weight rows on tl.dot's M side and the tokens on N (the router analogue of _bf16_swapab_kernel),
grid = (cdiv(M, BLOCK_T), 256 // 64, SPLIT_K) at every m. Each program owns one (token-block,
weight-block, K-slice) and atomically adds its fp32 K-partial, transposed, into an output zeroed
right before the launch; there is no plain-store path. Configs are static, chosen from m at
launch; tokens beyond m are masked, so any m is valid. Restacked (hypothesis
bf16-restack-m16-swapab-router-swapab-m32) gated to m <= ROUTER_SWAPAB_MAX_M = 32 only: above
that, master's tokens-on-M _router_hilo_kernel via _router_hilo_forward (master's forward,
unchanged), with _router_hilo_config and ROUTER_NUM_STAGES.

Shape-gated Triton winners (hypothesis bf16-shape-gated-triton-winners-combo): four
individually-measured Triton paths, each routed ONLY on the (layer, m, n) shapes where its own
run beat cuBLAS; every other bf16 shape keeps `_bf16_gemm_dispatch_impl`. Routing is by cfg and
shape preconditions only. Each path has its own prepare()-allocated workspace / ticket counters.
  pooler,   m <= 16:  _bf16_splitk_kernel, grid (64, 2), BLOCK_N 64, BLOCK_K 256, 4 stages
                      (from pooler-triton-splitk2-blockn64-semaphore-control-blockk256)
  qkv_gate, m <= 16:  _qkv_splitk_sem_kernel, grid (36, 4), BLOCK_N 64, BLOCK_K 128, 4 stages
                      (from qkv-gate-triton-splitk4-blockn64-semaphore)
  unpooler, n == 1024 (m <= 8): _bf16_splitk_kernel on the first 1024 rows read in place,
                      grid (16, 8), BLOCK_N 64, BLOCK_K 128, 3 stages
                      (from unpooler-slice-triton-splitk8-semaphore)
  o,        m <= 16:  _pooler_skinny_kernel, full K, grid (128,), BLOCK_N 64, BLOCK_K 256
                      (from o-triton-skinny-m32)
The split-K kernels store fp32 partials, take one gpu-scope acq_rel ticket per N-tile, and the
last-ticket CTA sums the partials in fixed order, stores bf16 once and resets its counter.

Swap-AB at 17 <= m <= 32 (hypothesis bf16-m32-swapab-wgmma-triton): all four bf16 layers run
_bf16_swapab_kernel, 64 weight rows on tl.dot's M side and the (<= 32) tokens on N, so Triton
can emit wgmma m64n32k16; pooler split-K 2, qkv_gate split-K 4 (same ticket reduce, transposed
workspace), unpooler / o full K. Every other shape keeps its path above.

Swap-AB at 1 <= m <= 16 (hypothesis bf16-m16-swapab-wgmma-extend): pooler, qkv_gate, o and the
unpooler's n == 1024 slice run the same _bf16_swapab_kernel with BLOCK_T = WS_M = 16 (wgmma
m64n16k16) and their own 16-row workspaces / tickets, routed BEFORE the m <= 16 kernels above
(which stay in the file, no longer routed for these shapes). The unpooler full width at m = 16
keeps cuBLAS; every m >= 17 shape keeps its path.
"""

import os
import time
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


ROUTER_N = 256
ROUTER_K = 8192
ROUTER_BLOCK_K = 128
ROUTER_BLOCK_W = 64     # router weight rows per program = wgmma M for one warpgroup (swap-AB)
ROUTER_NUM_STAGES = 3
ROUTER_FALLBACK_STAGES = 2  # used for a bucket whose 3-stage compile is OutOfResources (prepare)
ROUTER_NUM_WARPS = 4    # one warpgroup
ROUTER_SWAPAB_MAX_M = 32  # m <= this: swap-AB _router_forward; above: master's hilo forward
# m values pre-warmed in prepare(): one per config bucket plus the scored/tested ones, so that
# the timed call never compiles (M is not specialized and the grid's token extent is a launch
# argument, so one compile serves a whole bucket).
ROUTER_WARM_MS = (1, 8, 16, 32, 64, 128, 256)
# Gate for the pooler/unpooler NN path (own constants so it stacks cleanly with the cluster routes
# on other rows): these layers run torch.mm(x, W_t) against a transposed [K, N] copy of the
# weight made once in prepare() for NN_GATED_MIN_M <= m <= NN_GATED_MAX_M, full width only.
NN_GATED_LAYERS = ("pooler", "unpooler")
NN_GATED_MIN_M = 65
NN_GATED_MAX_M = 128
# Pooler only: widen the NN gate to 129..256 under its OWN flag (nn_p256_ok) so a self-check miss
# there cannot switch off the proven m <= 128 NN path (nn_ok).
NN_POOLER_MAX_M = 256
# Cells of the prepare()-time A/B/A/B graph self-check: (layer, n, k, m).
AB_CASES = (
    ("router", ROUTER_N, ROUTER_K, 1), ("router", ROUTER_N, ROUTER_K, 8), ("router", ROUTER_N, ROUTER_K, 128),
    ("unpooler", 1024, 4096, 1), ("unpooler", 1024, 4096, 8), ("unpooler", 8192, 4096, 128),
    ("pooler", 4096, 8192, 1), ("pooler", 4096, 8192, 8), ("pooler", 4096, 8192, 128),
    ("pooler", 4096, 8192, 256),
)
AB_ROUNDS = 2           # A, B, A, B
AB_BF16_REL_L2 = 6e-3   # bf16 output rounding is ~2e-3; the judge allows 1e-2
AB_ROUTER_REL_L2 = 5e-5  # the judge allows 1e-4


def _router_config(m: int):
    """Static swap-AB launch config from m only, for m <= ROUTER_SWAPAB_MAX_M: (BLOCK_T, SPLIT_K).

    grid = (cdiv(m, BLOCK_T), 256 // 64, SPLIT_K), always atomic_add into a zeroed output:
      m <=  16: BLOCK_T=16, SPLIT_K=32 -> (1, 4, 32) = 128 CTAs, K-slice 256
      m <=  32: BLOCK_T=32, SPLIT_K=32 -> (1, 4, 32) = 128 CTAs, K-slice 256
    m > 32 runs master's hilo path (_router_hilo_config), never this config.
    """
    if m <= 16:
        return 16, 32
    return 32, 32


def _router_hilo_config(m: int):
    """Master's tokens-on-M config (BLOCK_M, BLOCK_N, SPLIT_K) for _router_hilo_kernel; the timed
    path uses it at m > ROUTER_SWAPAB_MAX_M only (the m <= 32 rows are master's, unused)."""
    if m <= 16:
        return 16, 32, 16
    if m <= 32:
        return 32, 32, 16
    if m <= 128:
        return 64, 64, 16
    return 64, 64, 8


@triton.jit(do_not_specialize=["M"])
def _router_hilo_kernel(
    x_ptr, hi_ptr, lo_ptr, out_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    K_SLICE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[M, N] (+)= x[M, K] @ (hi + lo)[N, K]^T over this program's K-slice, fp32 accumulate.

    x: bf16 [M, K] row-major; hi, lo: bf16 [N, K] row-major; out: fp32 [M, N] row-major.
    Program (pid_m, pid_n, pid_k) covers rows of M [pid_m*BLOCK_M, +BLOCK_M), rows of N
    [pid_n*BLOCK_N, +BLOCK_N) and K [pid_k*K_SLICE, +K_SLICE), with no loop over M; its fp32
    K-partial is atomically added into the zeroed output. K_SLICE is a multiple of BLOCK_K.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    k_start = pid_k * K_SLICE

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    # Weight tiles are read directly as [BLOCK_K, BLOCK_N] (K contiguous in memory), which is
    # hi.T / lo.T for this tile, so the dot is x_tile @ w_tile without an explicit transpose.
    hi_ptrs = hi_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn
    lo_ptrs = lo_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_SLICE // BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        hi_tile = tl.load(hi_ptrs)
        lo_tile = tl.load(lo_ptrs)
        acc = tl.dot(x_tile, hi_tile, acc)
        acc = tl.dot(x_tile, lo_tile, acc)
        x_ptrs += BLOCK_K
        hi_ptrs += BLOCK_K
        lo_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None], sem="relaxed")


@triton.jit(do_not_specialize=["M"])
def _router_swapab_kernel(
    x_ptr, hi_ptr, lo_ptr, out_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    K_SLICE: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[M, N] (+)= x[M, K] @ (hi + lo)[N, K]^T over this program's K-slice, computed as
    ((hi + lo) @ x^T)^T (swap-AB), fp32 accumulate.

    x: bf16 [M, K] row-major; hi, lo: bf16 [N, K] row-major; out: fp32 [M, N] row-major.
    Program (pid_t, pid_w, pid_k) covers tokens [pid_t*BLOCK_T, +BLOCK_T) (tokens >= M load as
    zero and are never added), weight rows [pid_w*BLOCK_W, +BLOCK_W) and K [pid_k*K_SLICE,
    +K_SLICE). hi / lo tiles [BLOCK_W, BLOCK_K] (as stored) are tl.dot's LHS, the x^T tile
    [BLOCK_K, BLOCK_T] its RHS, so 64 weight rows sit on the MMA M side. The fp32 K-partial is
    atomically added, transposed, into the zeroed output. K_SLICE is a multiple of BLOCK_K.
    """
    pid_t = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_k = tl.program_id(2)
    k_start = pid_k * K_SLICE

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    hi_ptrs = hi_ptr + offs_w[:, None] * stride_wn + offs_k[None, :]   # [BLOCK_W, BLOCK_K]
    lo_ptrs = lo_ptr + offs_w[:, None] * stride_wn + offs_k[None, :]
    xt_ptrs = x_ptr + offs_t[None, :] * stride_xm + offs_k[:, None]    # [BLOCK_K, BLOCK_T] = x^T
    acc = tl.zeros((BLOCK_W, BLOCK_T), dtype=tl.float32)
    for kk in range(0, K_SLICE // BLOCK_K):
        hi_tile = tl.load(hi_ptrs)
        lo_tile = tl.load(lo_ptrs)
        xt_tile = tl.load(xt_ptrs, mask=t_mask[None, :], other=0.0)
        acc = tl.dot(hi_tile, xt_tile, acc)
        acc = tl.dot(lo_tile, xt_tile, acc)
        hi_ptrs += BLOCK_K
        lo_ptrs += BLOCK_K
        xt_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_t[None, :] * stride_om + offs_w[:, None]
    tl.atomic_add(out_ptrs, acc, mask=t_mask[None, :], sem="relaxed")


def _router_forward(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor, num_stages: int,
                    report: bool = False) -> torch.Tensor:
    """x bf16 [m, 8192] @ (hi + lo).T with fp32 accumulation -> fp32 [m, 256]. One launch plus
    the memset of the zeroed output. Branches only on shapes. report=True (prepare() only)
    prints the compiled variant's registers, spills and PTX counts."""
    m, k = x.shape
    n = hi.shape[0]
    block_t, split_k = _router_config(m)
    k_slice = k // split_k
    # Zeroed right before the launch, every call: the K-partials are atomically added into it.
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, block_t), n // ROUTER_BLOCK_W, split_k)
    compiled = _router_swapab_kernel[grid](
        x, hi, lo, out,
        m,
        x.stride(0), hi.stride(0), out.stride(0),
        K_SLICE=k_slice,
        BLOCK_W=ROUTER_BLOCK_W, BLOCK_T=block_t, BLOCK_K=ROUTER_BLOCK_K,
        num_warps=ROUTER_NUM_WARPS, num_stages=num_stages,
    )
    if report:
        _print_ptx_counts(f"[router] swapab m={m} BLOCK_T={block_t} SPLIT_K={split_k} "
                          f"grid={grid} stages={num_stages}", compiled)
    return out


def _router_hilo_forward(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """x bf16 [m, 8192] @ (hi + lo).T with fp32 accumulation -> fp32 [m, 256]. One launch plus
    the memset of the zeroed output. Branches only on shapes."""
    m, k = x.shape
    n = hi.shape[0]
    block_m, block_n, split_k = _router_hilo_config(m)
    k_slice = k // split_k
    # Zeroed right before the launch, every call: the K-partials are atomically added into it.
    out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, block_m), n // block_n, split_k)
    _router_hilo_kernel[grid](
        x, hi, lo, out,
        m,
        x.stride(0), hi.stride(0), out.stride(0),
        K_SLICE=k_slice,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=ROUTER_BLOCK_K,
        num_warps=ROUTER_NUM_WARPS, num_stages=ROUTER_NUM_STAGES,
    )
    return out


def _router_shape_ok(x: torch.Tensor, hi: torch.Tensor) -> bool:
    """The Triton path's shape preconditions (the problem's router is always [*, 8192] -> 256)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return False
    m, k = x.shape
    n = hi.shape[0]
    if m <= ROUTER_SWAPAB_MAX_M:   # swap-AB: SPLIT_K 32, 64 weight rows per program
        _block_t, split_k = _router_config(m)
        block_n = ROUTER_BLOCK_W
    else:                          # master's hilo: SPLIT_K 16 / 8
        _block_m, block_n, split_k = _router_hilo_config(m)
    return (k == hi.shape[1] and k % (ROUTER_BLOCK_K * split_k) == 0 and n % block_n == 0
            and hi.stride(1) == 1)


def _router_path(x: torch.Tensor, hi: torch.Tensor, n: int, stages: dict) -> str:
    """Which router kernel a call takes, from shapes only: "swapab" (m <= 32), "hilo" (m > 32,
    master's) or "" (the exact fp32 F.linear fallback)."""
    if not (_router_shape_ok(x, hi) and n == hi.shape[0]):
        return ""
    if x.shape[0] > ROUTER_SWAPAB_MAX_M:
        return "hilo"
    return "swapab" if _router_config(x.shape[0]) in stages else ""


def _router_prepare(state: dict) -> None:
    """Choose each swap-AB bucket's stage count (3, or 2 if the 3-stage compile is
    OutOfResources; recorded in state as a constant per bucket) and warm every m in
    ROUTER_WARM_MS on synthetic x so the timed call never compiles: m <= 32 on the swap-AB
    kernel, m > 32 (64 / 128 / 256) on master's hilo forward. Asserts that every warm m resolves
    to a kernel path, never the F.linear fallback. Weights only; nothing from a later call is
    involved."""
    hi, lo = state["router_hi"], state["router_lo"]
    k = hi.shape[1]
    stages = {}
    xs = {}
    for m in ROUTER_WARM_MS:
        x = torch.randn((m, k), device=hi.device, dtype=torch.float32).to(torch.bfloat16)
        xs[m] = x
        if not _router_shape_ok(x, hi):
            continue
        if m > ROUTER_SWAPAB_MAX_M:
            _router_hilo_forward(x, hi, lo)
            continue
        bucket = _router_config(m)
        if bucket not in stages:
            num_stages = ROUTER_NUM_STAGES
            try:
                _router_forward(x, hi, lo, num_stages, report=True)
            except _OutOfResources as e:
                print(f"[router] bucket {bucket}: {num_stages} stages OutOfResources ({e}); "
                      f"retrying at {ROUTER_FALLBACK_STAGES}", flush=True)
                num_stages = ROUTER_FALLBACK_STAGES
                _router_forward(x, hi, lo, num_stages, report=True)
            stages[bucket] = num_stages
        else:
            _router_forward(x, hi, lo, stages[bucket])
    state["router_stages"] = stages
    if tuple(hi.shape) == (ROUTER_N, ROUTER_K):
        for m, x in xs.items():
            path = _router_path(x, hi, ROUTER_N, stages)
            assert path in ("swapab", "hilo"), f"router m={m} resolves to the F.linear fallback"
            print(f"[router] m={m}: {path}", flush=True)


@triton.jit(do_not_specialize=["M"])
def _bf16_splitk_kernel(
    x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    N: tl.constexpr, WS_M: tl.constexpr, K_SLICE: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[M, N] bf16 = x[M, K] @ w[N, K]^T, split-K with an in-kernel semaphore reduce.

    Program (pid_n, pid_k) covers rows of N [pid_n*BLOCK_N, +BLOCK_N), K [pid_k*K_SLICE,
    +K_SLICE) and all M rows (M <= BLOCK_M <= WS_M; rows >= M masked). w may be the leading N
    rows of a taller row-major weight (stride_wn is its row stride). ws: fp32 [SPLIT_K, WS_M, N]
    scratch; cnt: int32 [N // BLOCK_N], 0 between launches. There is no early exit: every CTA
    stores its partial and takes exactly one ticket, so the counter always returns to 0.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_start = pid_k * K_SLICE

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    # Weight tiles read directly as [BLOCK_K, BLOCK_N] from the row-major [N, K] weight.
    w_ptrs = w_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_SLICE // BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w_tile = tl.load(w_ptrs)
        acc = tl.dot(x_tile, w_tile, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    # fp32 partial -> workspace slice pid_k, rows < M only (the reducer reads the same mask).
    ws_tile = offs_m[:, None] * N + offs_n[None, :]
    tl.store(ws_ptr + pid_k * (WS_M * N) + ws_tile, acc, mask=m_mask[:, None])
    # All threads' partial stores precede the (single, scalar) gpu-scope acq_rel ticket.
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    # Order the other threads' reloads after the ticket's acquire.
    tl.debug_barrier()
    if ticket == SPLIT_K - 1:
        total = tl.load(ws_ptr + ws_tile, mask=m_mask[:, None], other=0.0, cache_modifier=".cg")
        for s in tl.static_range(1, SPLIT_K):
            total += tl.load(ws_ptr + s * (WS_M * N) + ws_tile, mask=m_mask[:, None], other=0.0,
                             cache_modifier=".cg")
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
        tl.store(out_ptrs, total.to(tl.bfloat16), mask=m_mask[:, None])
        # Self-reset so the next launch / graph replay starts from 0.
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


# --- (1) pooler, m <= 16 (pooler-triton-splitk2-blockn64-semaphore-control-blockk256) --------
POOLER_N = 4096
POOLER_K = 8192
POOLER_MAX_M = 16   # combo: 32 -> 16 (m = 32 read 0.994x in the BLOCK_K=128 run)
POOLER_BLOCK_N = 64
POOLER_BLOCK_K = 256  # 16 K-steps per 4096 K-slice, 128 KB of weight in flight
POOLER_SPLIT_K = 2
POOLER_NUM_WARPS = 4
POOLER_NUM_STAGES = 4


def _pooler_block_m(m: int) -> int:
    return 16 if m <= 16 else 32


def _pooler_shape_ok(x: torch.Tensor, w: torch.Tensor, cfg) -> bool:
    """Preconditions of the pooler split-K path: m <= 16 and the problem's pooler shape."""
    return (cfg["m"] <= POOLER_MAX_M and cfg["n"] == POOLER_N and cfg["k"] == POOLER_K
            and x.dim() == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
            and 1 <= x.shape[0] <= POOLER_MAX_M and x.shape[1] == POOLER_K
            and tuple(w.shape) == (POOLER_N, POOLER_K) and w.stride(1) == 1)


def _pooler_forward(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """x bf16 [m <= 16, 8192] @ w.T -> bf16 [m, 4096], one launch. Branches only on shapes."""
    m, k = x.shape
    out = torch.empty((m, POOLER_N), dtype=torch.bfloat16, device=x.device)
    grid = (POOLER_N // POOLER_BLOCK_N, POOLER_SPLIT_K)
    _bf16_splitk_kernel[grid](
        x, w, out, ws, cnt,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        N=POOLER_N, WS_M=POOLER_MAX_M, K_SLICE=k // POOLER_SPLIT_K, SPLIT_K=POOLER_SPLIT_K,
        BLOCK_M=_pooler_block_m(m), BLOCK_N=POOLER_BLOCK_N, BLOCK_K=POOLER_BLOCK_K,
        num_warps=POOLER_NUM_WARPS, num_stages=POOLER_NUM_STAGES,
    )
    return out


# --- (2) qkv_gate, m <= 16 (qkv-gate-triton-splitk4-blockn64-semaphore) ----------------------
QKV_N = 2304
QKV_K = 8192
QKV_MAX_M = 16   # combo: 32 -> 16 (m = 32 read 0.858x)
QKV_BLOCK_N = 64
QKV_BLOCK_K = 128
QKV_SPLIT_K = 4
QKV_NUM_WARPS = 4
QKV_NUM_STAGES = 4


@triton.jit(do_not_specialize=["M"])
def _qkv_splitk_sem_kernel(
    x_ptr, w_ptr, ws_ptr, cnt_ptr, out_ptr,
    M,
    stride_xm, stride_wn, stride_ws_s, stride_ws_m, stride_om,
    K_SLICE: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[M, N] = x[M, K] @ w[N, K]^T in one launch, split-K with a ticket-semaphore epilogue.

    Grid (N // BLOCK_N, SPLIT_K). Program (pid_n, pid_k) streams w rows [pid_n*BLOCK_N, +BLOCK_N)
    over K [pid_k*K_SLICE, +K_SLICE) with one M-block (M <= BLOCK_M, rows >= M masked), stores its
    fp32 partial to ws[pid_k, :M, n-tile], then draws one ticket on cnt[pid_n]. The program that
    draws ticket SPLIT_K-1 (all partials of this n-tile are written by then) reloads the SPLIT_K
    partials (.cg, bypassing L1), sums them in fixed order 0..SPLIT_K-1, stores bf16 once and
    resets cnt[pid_n] to 0 for the next launch. No program waits on another, so it cannot
    deadlock whether or not the CTAs are co-resident.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_start = pid_k * K_SLICE

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    # Weight tiles read directly as [BLOCK_K, BLOCK_N] from the row-major [N, K] weight (K
    # contiguous), i.e. w.T for this tile, as in the router kernel; no repack.
    w_ptrs = w_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_SLICE // BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w_tile = tl.load(w_ptrs)
        acc = tl.dot(x_tile, w_tile, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    # fp32 partial for rows < M only; the reducer reads exactly the same (masked) rows, so every
    # value it reads was written in this launch and nothing from an earlier call can reach out.
    tile_off = offs_m[:, None] * stride_ws_m + offs_n[None, :]
    tl.store(ws_ptr + pid_k * stride_ws_s + tile_off, acc, mask=m_mask[:, None])
    # All threads' partial stores precede the (single, scalar) release atomic below.
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    # The acquire done by the atomic's issuing thread is ordered before every thread's reloads.
    tl.debug_barrier()
    if ticket == SPLIT_K - 1:
        total = tl.load(ws_ptr + tile_off, mask=m_mask[:, None], other=0.0, cache_modifier=".cg")
        for s in tl.static_range(1, SPLIT_K):
            part = tl.load(ws_ptr + s * stride_ws_s + tile_off, mask=m_mask[:, None], other=0.0,
                           cache_modifier=".cg")
            total += part
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
        tl.store(out_ptrs, total.to(tl.bfloat16), mask=m_mask[:, None])
        # Reset for the next launch (all SPLIT_K tickets of this n-tile are drawn).
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


def _qkv_forward(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """x bf16 [m, 8192] @ w[2304, 8192].T -> bf16 [m, 2304], m <= 16, ONE launch. Branches only on
    shapes; ws (fp32 [SPLIT_K, 16, 2304]) and cnt (int32 [36], all zero between launches) come
    from prepare()."""
    m, k = x.shape
    n = w.shape[0]
    block_m = 16 if m <= 16 else 32
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (n // QKV_BLOCK_N, QKV_SPLIT_K)
    _qkv_splitk_sem_kernel[grid](
        x, w, ws, cnt, out,
        m,
        x.stride(0), w.stride(0), ws.stride(0), ws.stride(1), out.stride(0),
        K_SLICE=k // QKV_SPLIT_K, SPLIT_K=QKV_SPLIT_K,
        BLOCK_M=block_m, BLOCK_N=QKV_BLOCK_N, BLOCK_K=QKV_BLOCK_K,
        num_warps=QKV_NUM_WARPS, num_stages=QKV_NUM_STAGES,
    )
    return out


def _qkv_route_ok(x: torch.Tensor, cfg: Config) -> bool:
    """qkv_gate Triton path: cfg m <= 16, n == 2304, k == 8192 and x a bf16 [m, 8192] with unit
    inner stride. Everything else (m = 32, m >= 128, the m=512 outlier test) keeps cuBLAS."""
    m = cfg["m"]
    return (1 <= m <= QKV_MAX_M and cfg["n"] == QKV_N and cfg["k"] == QKV_K
            and x.dim() == 2 and x.shape[0] == m and x.shape[1] == QKV_K
            and x.dtype == torch.bfloat16 and x.stride(1) == 1)


# --- (3) unpooler slice, n == 1024 (unpooler-slice-triton-splitk8-semaphore) -----------------
UNPOOL_N = 1024      # the sliced output width (this GPU's slice, m <= 8)
UNPOOL_K = 4096
UNPOOL_MAX_M = 16    # one BLOCK_M=16 tile; rows >= m masked
UNPOOL_BLOCK_N = 64
UNPOOL_BLOCK_K = 128
UNPOOL_SPLIT_K = 8   # K-slice 512, 4 K-steps
UNPOOL_NUM_WARPS = 4
UNPOOL_NUM_STAGES = 3
UNPOOL_WARM_MS = tuple(range(1, 9))  # combo: every m the n == 1024 slice can take


def _unpool_shape_ok(x: torch.Tensor, w: torch.Tensor, cfg) -> bool:
    """Preconditions of the unpooler-slice split-K path: n == 1024, m <= 16, and the problem's
    unpooler weight ([8192, 4096] row-major, of which only the first 1024 rows are read)."""
    return (cfg["n"] == UNPOOL_N and cfg["k"] == UNPOOL_K and cfg["m"] <= UNPOOL_MAX_M
            and x.dim() == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
            and 1 <= x.shape[0] <= UNPOOL_MAX_M and x.shape[1] == UNPOOL_K
            and w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[0] >= UNPOOL_N
            and w.shape[1] == UNPOOL_K and w.stride(1) == 1)


def _unpool_forward(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """x bf16 [m <= 16, 4096] @ w[:1024].T -> bf16 [m, 1024], one launch, w read in place
    (stride_wn = w.stride(0) = 4096, no slice copy). Branches only on shapes."""
    m, k = x.shape
    out = torch.empty((m, UNPOOL_N), dtype=torch.bfloat16, device=x.device)
    grid = (UNPOOL_N // UNPOOL_BLOCK_N, UNPOOL_SPLIT_K)
    _bf16_splitk_kernel[grid](
        x, w, out, ws, cnt,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        N=UNPOOL_N, WS_M=UNPOOL_MAX_M, K_SLICE=k // UNPOOL_SPLIT_K, SPLIT_K=UNPOOL_SPLIT_K,
        BLOCK_M=UNPOOL_MAX_M, BLOCK_N=UNPOOL_BLOCK_N, BLOCK_K=UNPOOL_BLOCK_K,
        num_warps=UNPOOL_NUM_WARPS, num_stages=UNPOOL_NUM_STAGES,
    )
    return out


# --- (4) o, m <= 16 (o-triton-skinny-m32, kernel from pooler-triton-skinny-m32) --------------
O_SKINNY_MAX_M = 16   # combo: 32 -> 16 (m = 32 read 0.907x)
O_N = 8192
O_K = 1024
O_BLOCK_N = 64    # 8192 / 64 = 128 programs: one wave on 132 SMs
O_BLOCK_K = 256   # 1024 / 256 = 4 K steps per program, all covered by the 4-stage pipeline
O_NUM_WARPS = 4
O_NUM_STAGES = 4  # 4 x (64 x 256 + BLOCK_M x 256) x 2 B: 160 KB at BLOCK_M 16


@triton.jit(do_not_specialize=["M"])
def _pooler_skinny_kernel(
    x_ptr, w_ptr, out_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[:M, n0:n0+BLOCK_N] = x[:M, :K] @ w[n0:n0+BLOCK_N, :K].T for program n0 = pid * BLOCK_N.

    One program owns BLOCK_N rows of the row-major [N, K] weight and walks the whole K in
    BLOCK_K steps, so each weight byte is read once and there is no split-K reduce. Rows of x
    at or past M load as zero and are not stored, so M need not fill BLOCK_M. The dot is bf16 x
    bf16 with an fp32 accumulator; the only bf16 rounding is at the store."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_ok = offs_m < M

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]   # [BLOCK_M, BLOCK_K]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]   # [BLOCK_N, BLOCK_K], as stored
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=row_ok[:, None], other=0.0)
        w_tile = tl.load(w_ptrs)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_ok[:, None])


def _o_skinny_applies(x: torch.Tensor, cfg: Config) -> bool:
    """Shape- and config-only guard for the Triton o path (m <= 16)."""
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    return (m <= O_SKINNY_MAX_M and n == O_N and k == O_K
            and x.dim() == 2 and x.shape[0] == m and x.shape[1] == k and x.stride(1) == 1
            and x.dtype == torch.bfloat16)


def _o_skinny_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [m, 1024] bf16 @ w [8192, 1024] bf16 .T -> [m, 8192] bf16, m <= 16."""
    m = x.shape[0]
    block_m = 16 if m <= 16 else 32
    out = torch.empty((m, O_N), dtype=torch.bfloat16, device=x.device)
    _pooler_skinny_kernel[(O_N // O_BLOCK_N,)](
        x, w, out,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        K=O_K, BLOCK_M=block_m, BLOCK_N=O_BLOCK_N, BLOCK_K=O_BLOCK_K,
        num_warps=O_NUM_WARPS, num_stages=O_NUM_STAGES,
    )
    return out


def _o_skinny_warmup(w: torch.Tensor) -> None:
    """Compile the o specialization for every m in 1..16 the timed path can route here, on
    synthetic zeros, so a compile or shared-memory failure surfaces in prepare() and no call to
    kernel_fn compiles. Outputs are dropped."""
    for m in range(1, O_SKINNY_MAX_M + 1):
        x = torch.zeros((m, O_K), dtype=torch.bfloat16, device=w.device)
        _o_skinny_gemm(x, w)


# --- (5) all four bf16 layers, 17 <= m <= 32: swap-AB (bf16-m32-swapab-wgmma-triton) ----------
# The weight is tl.dot's LHS (64 weight rows on the MMA M side) and the tokens its RHS (32 on N),
# the orientation of cuBLAS's nvjet 64x32 tiles, so that Triton can lower it to wgmma m64n32k16
# instead of the mma.sync it emits with 16/32 tokens on M.
SWAPAB_MIN_M = 17
SWAPAB_MAX_M = 32
SWAPAB_BLOCK_W = 64   # weight rows per program = wgmma M for one warpgroup
SWAPAB_BLOCK_T = 32   # token columns per program (all of them); columns >= m masked
SWAPAB_NUM_WARPS = 4  # one warpgroup
# layer -> (N, K, SPLIT_K, BLOCK_K, num_stages); static, from cfg layer only.
SWAPAB_CFG = {
    "pooler":   (4096, 8192, 2, 256, 4),  # grid (64, 2) = 128 CTAs, 48 KB/stage
    "qkv_gate": (2304, 8192, 4, 128, 4),  # grid (36, 4) = 144 CTAs, 24 KB/stage
    "unpooler": (8192, 4096, 1, 256, 4),  # grid (128, 1), full width (m > 8)
    "o":        (8192, 1024, 1, 256, 4),  # grid (128, 1), 4 K-steps, all in flight
}
SWAPAB_FALLBACK_STAGES = 3  # used for a layer whose 4-stage compile is OutOfResources

try:
    from triton.runtime.errors import OutOfResources as _OutOfResources
except Exception:  # older/newer Triton layouts
    _OutOfResources = getattr(triton.runtime, "OutOfResources", RuntimeError)


@triton.jit(do_not_specialize=["M"])
def _bf16_swapab_kernel(
    x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    N: tl.constexpr, WS_M: tl.constexpr, K_SLICE: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[M, N] bf16 = x[M, K] @ w[N, K]^T computed as (w @ x^T)^T, swap-AB.

    Program (pid_w, pid_k) owns weight rows [pid_w*BLOCK_W, +BLOCK_W) and K [pid_k*K_SLICE,
    +K_SLICE) for all M <= BLOCK_T tokens (token columns >= M load as zero and are never stored).
    acc[BLOCK_W, BLOCK_T] fp32 = w_tile[BLOCK_W, BLOCK_K] @ xt_tile[BLOCK_K, BLOCK_T]; both
    operands are K-contiguous in memory. SPLIT_K == 1: transposed bf16 store into out [M, N].
    SPLIT_K > 1: ws fp32 [SPLIT_K, WS_M, N] partials (stored transposed, tokens < M only) and
    cnt int32 [N // BLOCK_W] tickets, 0 between launches; the CTA drawing ticket SPLIT_K-1 sums
    the partials in order 0..SPLIT_K-1, stores bf16 once and resets its counter. No early exit:
    every CTA takes exactly one ticket.
    """
    pid_w = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_start = pid_k * K_SLICE

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    w_ptrs = w_ptr + offs_w[:, None] * stride_wn + offs_k[None, :]    # [BLOCK_W, BLOCK_K], as stored
    xt_ptrs = x_ptr + offs_t[None, :] * stride_xm + offs_k[:, None]   # [BLOCK_K, BLOCK_T] = x^T tile
    acc = tl.zeros((BLOCK_W, BLOCK_T), dtype=tl.float32)
    for kk in range(0, K_SLICE // BLOCK_K):
        w_tile = tl.load(w_ptrs)
        xt_tile = tl.load(xt_ptrs, mask=t_mask[None, :], other=0.0)
        acc = tl.dot(w_tile, xt_tile, acc)
        w_ptrs += BLOCK_K
        xt_ptrs += BLOCK_K

    if SPLIT_K == 1:
        out_ptrs = out_ptr + offs_t[None, :] * stride_om + offs_w[:, None]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=t_mask[None, :])
    else:
        # fp32 partial -> workspace slice pid_k, token rows < M only (the reducer reads the same mask).
        ws_tile = offs_t[None, :] * N + offs_w[:, None]
        tl.store(ws_ptr + pid_k * (WS_M * N) + ws_tile, acc, mask=t_mask[None, :])
        # All threads' partial stores precede the (single, scalar) gpu-scope acq_rel ticket.
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid_w, 1, sem="acq_rel", scope="gpu")
        # Order the other threads' reloads after the ticket's acquire.
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            total = tl.load(ws_ptr + ws_tile, mask=t_mask[None, :], other=0.0, cache_modifier=".cg")
            for s in tl.static_range(1, SPLIT_K):
                total += tl.load(ws_ptr + s * (WS_M * N) + ws_tile, mask=t_mask[None, :], other=0.0,
                                 cache_modifier=".cg")
            out_ptrs = out_ptr + offs_t[None, :] * stride_om + offs_w[:, None]
            tl.store(out_ptrs, total.to(tl.bfloat16), mask=t_mask[None, :])
            # Self-reset so the next launch / graph replay starts from 0.
            tl.atomic_xchg(cnt_ptr + pid_w, 0, sem="relaxed", scope="gpu")


def _swapab_route_ok(layer: str, x: torch.Tensor, w: torch.Tensor, cfg) -> bool:
    """Swap-AB path: a bf16 layer at 17 <= cfg m <= 32 with the problem's n/k and weight shape.
    Branches on cfg and shapes only."""
    spec = SWAPAB_CFG.get(layer)
    if spec is None:
        return False
    n, k = spec[0], spec[1]
    m = cfg["m"]
    return (SWAPAB_MIN_M <= m <= SWAPAB_MAX_M and cfg["n"] == n and cfg["k"] == k
            and x.dim() == 2 and x.shape[0] == m and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.stride(1) == 1
            and w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16
            and w.stride(1) == 1)


def _swapab_forward(layer: str, x: torch.Tensor, w: torch.Tensor, ws, cnt, num_stages: int,
                    report: bool = False) -> torch.Tensor:
    """x bf16 [17..32, k] @ w[n, k].T -> bf16 [m, n], ONE launch. ws / cnt are the layer's own
    prepare()-allocated split-K workspace and tickets (None at SPLIT_K == 1, where the kernel
    never touches them). report=True (prepare() only) prints the compiled variant's registers,
    spills and PTX MMA counts."""
    n, k, split_k, block_k, _ = SWAPAB_CFG[layer]
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if ws is None:  # SPLIT_K == 1: the pointers are unused by the compiled kernel
        ws, cnt = out, out
    grid = (n // SWAPAB_BLOCK_W, split_k)
    compiled = _bf16_swapab_kernel[grid](
        x, w, out, ws, cnt,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        N=n, WS_M=SWAPAB_MAX_M, K_SLICE=k // split_k, SPLIT_K=split_k,
        BLOCK_W=SWAPAB_BLOCK_W, BLOCK_T=SWAPAB_BLOCK_T, BLOCK_K=block_k,
        num_warps=SWAPAB_NUM_WARPS, num_stages=num_stages,
    )
    if report:
        try:
            ptx = compiled.asm["ptx"]
            print(f"[swapab] layer={layer} m={m} N={n} K={k} SPLIT_K={split_k} BLOCK_K={block_k} "
                  f"stages={num_stages}: n_regs={getattr(compiled, 'n_regs', '?')} "
                  f"n_spills={getattr(compiled, 'n_spills', '?')} "
                  f"wgmma={ptx.count('wgmma.mma_async')} mma.sync={ptx.count('mma.sync')} "
                  f"cp.async={ptx.count('cp.async')}", flush=True)
        except Exception as e:  # diagnostic only
            print(f"[swapab] layer={layer}: diagnostics unavailable ({type(e).__name__}: {e})", flush=True)
    return out


def _swapab_buffers(state: dict, layer: str):
    if layer == "pooler":
        return state["swapab_ws_pooler"], state["swapab_cnt_pooler"]
    if layer == "qkv_gate":
        return state["swapab_ws_qkv"], state["swapab_cnt_qkv"]
    return None, None


def _swapab_prepare(state: dict) -> None:
    """Allocate the swap-AB workspaces / tickets (own keys), choose each layer's stage count
    (4, or 3 if the 4-stage variant is OutOfResources; recorded in state as a constant), and warm
    every m in 17..32 for each layer on synthetic x, so the timed call never compiles."""
    dev = state["pooler"].device
    state["swapab_ws_pooler"] = torch.empty((2, SWAPAB_MAX_M, 4096), dtype=torch.float32, device=dev)
    state["swapab_cnt_pooler"] = torch.zeros(4096 // SWAPAB_BLOCK_W, dtype=torch.int32, device=dev)
    state["swapab_ws_qkv"] = torch.empty((4, SWAPAB_MAX_M, 2304), dtype=torch.float32, device=dev)
    state["swapab_cnt_qkv"] = torch.zeros(2304 // SWAPAB_BLOCK_W, dtype=torch.int32, device=dev)
    stages = {}
    for layer, (n, k, _split, _bk, num_stages) in SWAPAB_CFG.items():
        w = state[layer]
        if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16
                and w.stride(1) == 1):
            continue
        ws, cnt = _swapab_buffers(state, layer)
        x = torch.randn((SWAPAB_MAX_M, k), device=dev, dtype=torch.float32).to(torch.bfloat16)
        try:
            _swapab_forward(layer, x, w, ws, cnt, num_stages, report=True)
        except _OutOfResources as e:
            print(f"[swapab] layer={layer}: {num_stages} stages OutOfResources ({e}); "
                  f"retrying at {SWAPAB_FALLBACK_STAGES}", flush=True)
            num_stages = SWAPAB_FALLBACK_STAGES
            _swapab_forward(layer, x, w, ws, cnt, num_stages, report=True)
        stages[layer] = num_stages
        for m in range(SWAPAB_MIN_M, SWAPAB_MAX_M):
            xm = torch.randn((m, k), device=dev, dtype=torch.float32).to(torch.bfloat16)
            _swapab_forward(layer, xm, w, ws, cnt, num_stages)
    state["swapab_stages"] = stages
    # The split-K variants self-reset, but re-zero the counters after the warm-ups regardless.
    state["swapab_cnt_pooler"].zero_()
    state["swapab_cnt_qkv"].zero_()


# --- (6) pooler / qkv_gate / o at 1 <= m <= 16 and the unpooler slice: swap-AB, BLOCK_T 16 ------
# (hypothesis bf16-m16-swapab-wgmma-extend). The same _bf16_swapab_kernel as (5), with 16 token
# columns (wgmma m64n16k16) and its OWN 16-row workspaces / tickets: WS_M and BLOCK_T are passed
# as 16 here, never master's SWAPAB_MAX_M / SWAPAB_BLOCK_T (32), so the [SPLIT_K, 16, n]
# workspace is addressed with a 16-row stride. Per-layer grid / BLOCK_K / stages equal the
# incumbent m <= 16 kernels', so tl.dot orientation is the only variable.
SWAPAB16_MAX_M = 16
SWAPAB16_BLOCK_T = 16   # token columns per program; columns >= m masked
SWAPAB16_WS_M = 16      # workspace rows (== BLOCK_T)
# key -> (N, K, SPLIT_K, BLOCK_K, num_stages); static, from cfg layer (and n for the slice).
SWAPAB16_CFG = {
    "pooler":         (4096, 8192, 2, 256, 4),  # grid (64, 2)
    "qkv_gate":       (2304, 8192, 4, 128, 4),  # grid (36, 4)
    "o":              (8192, 1024, 1, 256, 4),  # grid (128, 1), full K
    # first 1024 rows of the full [8192, 4096] unpooler weight read in place (stride_wn 4096)
    "unpooler_slice": (1024, 4096, 8, 128, 3),  # grid (16, 8)
}
SWAPAB16_UNPOOL_WARM_MS = tuple(range(1, 9))  # every m the n == 1024 slice can take


def _swapab16_key(layer: str, cfg) -> str:
    """Config key from cfg only: the unpooler routes here only as its n == 1024 slice."""
    if layer == "unpooler":
        return "unpooler_slice" if cfg["n"] == 1024 else ""
    return layer if layer in ("pooler", "qkv_gate", "o") else ""


def _swapab16_route_ok(key: str, x: torch.Tensor, w: torch.Tensor, cfg) -> bool:
    """m <= 16 swap-AB path: 1 <= cfg m <= 16 with the key's n/k. pooler / qkv_gate / o need the
    full weight shape (n, k); the unpooler slice needs the full weight ([>= 1024, 4096]), of
    which only the first 1024 rows are read. Branches on cfg and shapes only."""
    spec = SWAPAB16_CFG.get(key)
    if spec is None:
        return False
    n, k = spec[0], spec[1]
    m = cfg["m"]
    if not (1 <= m <= SWAPAB16_MAX_M and cfg["n"] == n and cfg["k"] == k
            and x.dim() == 2 and x.shape[0] == m and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.stride(1) == 1
            and w.dim() == 2 and w.dtype == torch.bfloat16 and w.stride(1) == 1):
        return False
    if key == "unpooler_slice":
        return w.shape[0] >= n and w.shape[1] == k
    return tuple(w.shape) == (n, k)


def _print_ptx_counts(tag: str, compiled) -> None:
    """prepare()-only diagnostic: registers, spills and PTX MMA / cp.async counts."""
    try:
        ptx = compiled.asm["ptx"]
        print(f"{tag}: n_regs={getattr(compiled, 'n_regs', '?')} "
              f"n_spills={getattr(compiled, 'n_spills', '?')} "
              f"wgmma={ptx.count('wgmma.mma_async')} mma.sync={ptx.count('mma.sync')} "
              f"cp.async={ptx.count('cp.async')}", flush=True)
    except Exception as e:  # diagnostic only
        print(f"{tag}: diagnostics unavailable ({type(e).__name__}: {e})", flush=True)


def _swapab16_forward(key: str, x: torch.Tensor, w: torch.Tensor, ws, cnt, num_stages: int,
                      report: bool = False) -> torch.Tensor:
    """x bf16 [1..16, k] @ w[:n, k].T -> bf16 [m, n], ONE launch, BLOCK_T = WS_M = 16. w is read
    in place (for the unpooler slice, the full weight with stride_wn = 4096, first n rows only).
    ws / cnt are the key's own prepare()-allocated 16-row workspace and tickets (None at
    SPLIT_K == 1, where the kernel never touches them)."""
    n, k, split_k, block_k, _ = SWAPAB16_CFG[key]
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if ws is None:  # SPLIT_K == 1: the pointers are unused by the compiled kernel
        ws, cnt = out, out
    grid = (n // SWAPAB_BLOCK_W, split_k)
    compiled = _bf16_swapab_kernel[grid](
        x, w, out, ws, cnt,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        N=n, WS_M=SWAPAB16_WS_M, K_SLICE=k // split_k, SPLIT_K=split_k,
        BLOCK_W=SWAPAB_BLOCK_W, BLOCK_T=SWAPAB16_BLOCK_T, BLOCK_K=block_k,
        num_warps=SWAPAB_NUM_WARPS, num_stages=num_stages,
    )
    if report:
        _print_ptx_counts(f"[swapab16] {key} m={m} N={n} K={k} SPLIT_K={split_k} "
                          f"BLOCK_K={block_k} stages={num_stages}", compiled)
    return out


def _swapab16_buffers(state: dict, key: str):
    if key in ("pooler", "qkv_gate", "unpooler_slice"):
        return state[f"swapab16_ws_{key}"], state[f"swapab16_cnt_{key}"]
    return None, None


def _incumbent16_report(state: dict) -> None:
    """prepare()-only: relaunch master's incumbent m <= 16 kernels once each on synthetic x
    (same constants as their forward helpers, own buffers) and print their PTX counts, so the
    log shows incumbent mma.sync vs the new path's wgmma. Outputs are dropped."""
    dev = state["pooler"].device
    try:
        if tuple(state["pooler"].shape) == (POOLER_N, POOLER_K):
            m, w = 16, state["pooler"]
            x = torch.zeros((m, POOLER_K), dtype=torch.bfloat16, device=dev)
            out = torch.empty((m, POOLER_N), dtype=torch.bfloat16, device=dev)
            c = _bf16_splitk_kernel[(POOLER_N // POOLER_BLOCK_N, POOLER_SPLIT_K)](
                x, w, out, state["pooler_ws"], state["pooler_cnt"], m,
                x.stride(0), w.stride(0), out.stride(0),
                N=POOLER_N, WS_M=POOLER_MAX_M, K_SLICE=POOLER_K // POOLER_SPLIT_K,
                SPLIT_K=POOLER_SPLIT_K, BLOCK_M=_pooler_block_m(m), BLOCK_N=POOLER_BLOCK_N,
                BLOCK_K=POOLER_BLOCK_K, num_warps=POOLER_NUM_WARPS, num_stages=POOLER_NUM_STAGES)
            _print_ptx_counts(f"[incumbent16] pooler _bf16_splitk_kernel m={m}", c)
        if tuple(state["qkv_gate"].shape) == (QKV_N, QKV_K):
            m, w = 16, state["qkv_gate"]
            ws = state["qkv_ws"]
            x = torch.zeros((m, QKV_K), dtype=torch.bfloat16, device=dev)
            out = torch.empty((m, QKV_N), dtype=torch.bfloat16, device=dev)
            c = _qkv_splitk_sem_kernel[(QKV_N // QKV_BLOCK_N, QKV_SPLIT_K)](
                x, w, ws, state["qkv_cnt"], out, m,
                x.stride(0), w.stride(0), ws.stride(0), ws.stride(1), out.stride(0),
                K_SLICE=QKV_K // QKV_SPLIT_K, SPLIT_K=QKV_SPLIT_K,
                BLOCK_M=16, BLOCK_N=QKV_BLOCK_N, BLOCK_K=QKV_BLOCK_K,
                num_warps=QKV_NUM_WARPS, num_stages=QKV_NUM_STAGES)
            _print_ptx_counts(f"[incumbent16] qkv_gate _qkv_splitk_sem_kernel m={m}", c)
        if tuple(state["o"].shape) == (O_N, O_K):
            m, w = 16, state["o"]
            x = torch.zeros((m, O_K), dtype=torch.bfloat16, device=dev)
            out = torch.empty((m, O_N), dtype=torch.bfloat16, device=dev)
            c = _pooler_skinny_kernel[(O_N // O_BLOCK_N,)](
                x, w, out, m, x.stride(0), w.stride(0), out.stride(0),
                K=O_K, BLOCK_M=16, BLOCK_N=O_BLOCK_N, BLOCK_K=O_BLOCK_K,
                num_warps=O_NUM_WARPS, num_stages=O_NUM_STAGES)
            _print_ptx_counts(f"[incumbent16] o _pooler_skinny_kernel m={m}", c)
        w = state["unpooler"]
        if w.dim() == 2 and w.shape[0] >= UNPOOL_N and w.shape[1] == UNPOOL_K:
            m = 16
            x = torch.zeros((m, UNPOOL_K), dtype=torch.bfloat16, device=dev)
            out = torch.empty((m, UNPOOL_N), dtype=torch.bfloat16, device=dev)
            c = _bf16_splitk_kernel[(UNPOOL_N // UNPOOL_BLOCK_N, UNPOOL_SPLIT_K)](
                x, w, out, state["unpool_ws"], state["unpool_cnt"], m,
                x.stride(0), w.stride(0), out.stride(0),
                N=UNPOOL_N, WS_M=UNPOOL_MAX_M, K_SLICE=UNPOOL_K // UNPOOL_SPLIT_K,
                SPLIT_K=UNPOOL_SPLIT_K, BLOCK_M=UNPOOL_MAX_M, BLOCK_N=UNPOOL_BLOCK_N,
                BLOCK_K=UNPOOL_BLOCK_K, num_warps=UNPOOL_NUM_WARPS, num_stages=UNPOOL_NUM_STAGES)
            _print_ptx_counts(f"[incumbent16] unpooler_slice _bf16_splitk_kernel m={m}", c)
    except Exception as e:  # diagnostic only
        print(f"[incumbent16] diagnostics unavailable ({type(e).__name__}: {e})", flush=True)
    state["pooler_cnt"].zero_()
    state["qkv_cnt"].zero_()
    state["unpool_cnt"].zero_()


def _swapab16_prepare(state: dict) -> None:
    """Allocate the m <= 16 swap-AB workspaces (fp32 [SPLIT_K, 16, n]) / tickets (int32
    [n // 64], zeroed) under their own keys, choose each key's stage count (its config, or 3 if
    that is OutOfResources), warm every m in 1..16 (1..8 for the unpooler slice) on synthetic x
    so the timed call never compiles, print the PTX counts of the new and incumbent kernels, and
    re-zero all counters."""
    dev = state["pooler"].device
    for key, (n, _k, split_k, _bk, _st) in SWAPAB16_CFG.items():
        if split_k > 1:
            state[f"swapab16_ws_{key}"] = torch.empty((split_k, SWAPAB16_WS_M, n),
                                                      dtype=torch.float32, device=dev)
            state[f"swapab16_cnt_{key}"] = torch.zeros(n // SWAPAB_BLOCK_W, dtype=torch.int32,
                                                       device=dev)
    stages = {}
    for key, (n, k, _split, _bk, num_stages) in SWAPAB16_CFG.items():
        layer = "unpooler" if key == "unpooler_slice" else key
        w = state[layer]
        warm_ms = SWAPAB16_UNPOOL_WARM_MS if key == "unpooler_slice" else tuple(range(1, SWAPAB16_MAX_M + 1))
        m_top = warm_ms[-1]
        if not _swapab16_route_ok(key, torch.empty((m_top, k), dtype=torch.bfloat16, device=dev), w,
                                  {"m": m_top, "n": n, "k": k}):
            continue
        ws, cnt = _swapab16_buffers(state, key)
        x = torch.randn((m_top, k), device=dev, dtype=torch.float32).to(torch.bfloat16)
        try:
            _swapab16_forward(key, x, w, ws, cnt, num_stages, report=True)
        except _OutOfResources as e:
            print(f"[swapab16] {key}: {num_stages} stages OutOfResources ({e}); "
                  f"retrying at {SWAPAB_FALLBACK_STAGES}", flush=True)
            num_stages = SWAPAB_FALLBACK_STAGES
            _swapab16_forward(key, x, w, ws, cnt, num_stages, report=True)
        stages[key] = num_stages
        for m in warm_ms:
            xm = torch.randn((m, k), device=dev, dtype=torch.float32).to(torch.bfloat16)
            _swapab16_forward(key, xm, w, ws, cnt, num_stages)
    state["swapab16_stages"] = stages
    _incumbent16_report(state)
    # The split-K variants self-reset, but re-zero the counters after the warm-ups regardless.
    for key in SWAPAB16_CFG:
        if f"swapab16_cnt_{key}" in state:
            state[f"swapab16_cnt_{key}"].zero_()


# --- (7) unpooler n == 1024 slice, m <= 8: CUDA cluster / DSMEM split-K, gated at prepare -------
# (hypothesis restack-unpooler-slice-cluster-dsmem-m8; the kernel is ported unchanged from job
# 3f3330f7's oun_kernel, unpooler instantiation only: NT 8, no o instantiation is compiled.)
#
# Weight rows sit on the mma.sync m16n8k16 M side, the tokens on N (NT 8). A thread-block cluster of
# CL CTAs splits K: rank r owns K [r*K/CL, (r+1)*K/CL) of the cluster's ROWS output rows. Every CTA
# issues ALL its weight loads at once (NG cp.async.cg commit groups, no ring, 64 KB per CTA), the
# smem tile is XOR-swizzled (16 B chunk c of row r stored at c ^ (r & 7)) in both the cp.async
# destination and the ldmatrix addresses, the fp32 partial aliases the consumed weight buffer, and
# the CL partials are reduced through distributed shared memory in fixed rank order with one bf16
# store: no global workspace, no ticket, no atomics, no memset. There is no mbarrier and no
# spin-wait in this kernel (only cluster barriers), so nothing to bound.
#   CL 8 -> 16 clusters x 8 = 128 CTAs (1/SM), ROWS 64, K/CTA 512,  NG 4
#   CL 4 -> 32 clusters x 4 = 128 CTAs (1/SM), ROWS 32, K/CTA 1024, NG 4
# CL 8 is used only if cudaOccupancyMaxActiveClusters reports every cluster co-resident, else CL 4
# under the same rule, else master's path. All PTX is baseline sm_90 (cluster launch, mapa,
# barrier.cluster, mma.sync, cp.async): no wgmma / setmaxnreg, so no sm_90a-only guard is needed.
#
# The route is enabled only if, in prepare(), it passes the self-check AND beats the restack's own
# unpooler-slice path by OUNS_MIN_WIN on the judge's own measurement: CUDA-graph replay, >= 256 MB
# L2 flush before every replay, both paths interleaved (order swapped each rep), median of reps,
# at m = 1 and m = 8. The profile shows no kernel-time gain; the gain lives in row time minus
# kernel time, which only a graph replay includes.
OUNS_MAX_M = 8
OUNS_N = 1024
OUNS_K = 4096
OUNS_CLUSTERS_NEEDED = {8: 16, 4: 32}
OUNS_CHECK_MS = (1, 5, 8)
OUNS_GATE_MS = (1, 8)
OUNS_FLUSH_BYTES = 256 * 1024 * 1024
OUNS_GATE_REPS = 41
OUNS_MIN_WIN = 1.02
OUNS_EXT_NAME = "muse_bf16_unpool_slice_cluster_v3"

_OUNS_CPP = r"""
#include <torch/extension.h>
#include <string>
void ouns_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t cl);
int64_t ouns_occupancy(int64_t cl);
std::string ouns_attrs(int64_t cl);
"""

_OUNS_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <string>

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_pending(int pending) {
  if (pending >= 3) cp_wait<3>();
  else if (pending == 2) cp_wait<2>();
  else if (pending == 1) cp_wait<1>();
  else cp_wait<0>();
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t map_rank(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ float4 ld_remote_f4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr) : "memory");
  return v;
}

template <int NT, int ROWS, int KPER, int CL, int NG>
struct Cfg {
  static constexpr int THREADS = 256, NW = 8;
  static constexpr int T = ROWS / 16;                 // 16-row MMA tiles per cluster
  static constexpr int RG = T < NW ? T : NW;          // warp row groups
  static constexpr int KG = NW / RG;                  // warp K groups
  static constexpr int TPW = T / RG;                  // row tiles per warp
  static constexpr int NB = NT / 8;                   // n8 blocks
  static constexpr int PITCH = KPER * 2;              // smem row pitch, bytes
  static constexpr int CHG = KPER / NG / 8;           // 16 B chunks per row per commit group
  static constexpr int SG = KPER / NG / 16;           // k16 steps per commit group
  static constexpr int SPG = SG / KG;                 // k16 steps per warp per group
  static constexpr int PR = ROWS + 4;                 // partial pitch (floats): conflict-free C stores
  static constexpr int W_BYTES = ROWS * PITCH;
  static constexpr int X_BYTES = NT * PITCH;
  static constexpr int SMEM = W_BYTES + X_BYTES;
  static constexpr int RPR = ROWS / CL;               // rows each rank reduces
  static_assert(ROWS % 16 == 0 && T % RG == 0 && NW % RG == 0, "row tiling");
  static_assert(KPER % (NG * 16) == 0 && CHG % 8 == 0, "swizzle needs groups of 8-chunk multiples");
  static_assert(SG % KG == 0 && SPG % 2 == 0, "k split");
  static_assert(NT * PR * 4 <= W_BYTES, "fp32 partial must fit on the weight buffer");
  static_assert(RPR % 4 == 0 && NG <= 4, "reduce / wait");
};

// out[m, n] bf16 = x[m, KPER*CL] @ w[:n, :KPER*CL]^T. Grid = (n / ROWS) * CL CTAs, cluster dim CL.
template <int NT, int ROWS, int KPER, int CL, int NG>
__global__ void __launch_bounds__(256, 2)
ouns_kernel(const __nv_bfloat16* __restrict__ x, int64_t stride_x,
           const __nv_bfloat16* __restrict__ w, int64_t stride_w,
           __nv_bfloat16* __restrict__ out, int64_t stride_o, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using C = Cfg<NT, ROWS, KPER, CL, NG>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
  const int cid = blockIdx.x / CL;
  const int n0 = cid * ROWS;
  const int kbase = static_cast<int>(rank) * KPER;
  const uint32_t sw = smem_u32(smem);
  const uint32_t sx = sw + C::W_BYTES;
  const __nv_bfloat16* wb = w + (int64_t)n0 * stride_w + kbase;
  const __nv_bfloat16* xb = x + kbase;

  // All loads in flight at once: NG commit groups, each a K chunk of every weight row + x.
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    for (int q = tid; q < ROWS * C::CHG; q += C::THREADS) {
      const int row = q / C::CHG, c = g * C::CHG + (q % C::CHG);
      cp_async16(sw + row * C::PITCH + ((c ^ (row & 7)) << 4), wb + (int64_t)row * stride_w + c * 8, 16);
    }
    for (int q = tid; q < NT * C::CHG; q += C::THREADS) {
      const int tok = q / C::CHG, c = g * C::CHG + (q % C::CHG);
      const bool ok = tok < m;
      cp_async16(sx + tok * C::PITCH + ((c ^ (tok & 7)) << 4),
                 xb + (ok ? (int64_t)tok * stride_x : (int64_t)0) + c * 8, ok ? 16 : 0);
    }
    cp_commit();
  }

  float acc[C::TPW][C::NB][4];
#pragma unroll
  for (int i = 0; i < C::TPW; ++i)
#pragma unroll
    for (int nb = 0; nb < C::NB; ++nb)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][nb][e] = 0.f;

  const int rg = warp % C::RG, kg = warp / C::RG;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    cp_wait_pending(NG - 1 - g);
    __syncthreads();  // group g (all threads' copies) is visible
#pragma unroll
    for (int j = 0; j < C::SPG; j += 2) {
      const int kp = (g * C::SG + kg * C::SPG + j) / 2;  // k32 pair index in the CTA's K range
      uint32_t b[C::NB][4];
#pragma unroll
      for (int nb = 0; nb < C::NB; ++nb)
        ldsm_x4(b[nb], sx + (nb * 8 + (lane & 7)) * C::PITCH + (((kp * 4 + (lane >> 3)) ^ (lane & 7)) << 4));
#pragma unroll
      for (int i = 0; i < C::TPW; ++i) {
        const int arow = (rg * C::TPW + i) * 16 + (lane & 15);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t a[4];
          ldsm_x4(a, sw + arow * C::PITCH + (((((kp * 2 + h) * 2) + (lane >> 4)) ^ (lane & 7)) << 4));
#pragma unroll
          for (int nb = 0; nb < C::NB; ++nb) mma_bf16(acc[i][nb], a, b[nb][2 * h], b[nb][2 * h + 1]);
        }
      }
    }
  }
  __syncthreads();  // every warp is done with the weight buffer: the fp32 partial aliases it

  // CTA partial part[t][row] (pitch PR), K groups added in fixed order.
  float* part = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int st = 0; st < C::KG; ++st) {
    if (kg == st) {
#pragma unroll
      for (int i = 0; i < C::TPW; ++i)
#pragma unroll
        for (int nb = 0; nb < C::NB; ++nb) {
          const int row = (rg * C::TPW + i) * 16 + (lane >> 2);
          const int t = nb * 8 + (lane & 3) * 2;
          float* p0 = part + t * C::PR + row;
          float* p1 = p0 + C::PR;
          if (st == 0) {
            p0[0] = acc[i][nb][0]; p1[0] = acc[i][nb][1]; p0[8] = acc[i][nb][2]; p1[8] = acc[i][nb][3];
          } else {
            p0[0] += acc[i][nb][0]; p1[0] += acc[i][nb][1]; p0[8] += acc[i][nb][2]; p1[8] += acc[i][nb][3];
          }
        }
    }
    __syncthreads();
  }

  cluster_arrive();  // release: this CTA's partial is visible cluster-wide
  cluster_wait();    // acquire: all CL partials are visible

  const uint32_t part_s = smem_u32(part);
  constexpr int Q4 = C::RPR / 4;
  const int items = m * Q4;
  for (int idx = tid; idx < items; idx += C::THREADS) {
    const int t = idx / Q4, j = idx % Q4;
    const uint32_t off = static_cast<uint32_t>((t * C::PR + static_cast<int>(rank) * C::RPR + j * 4) * 4);
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int r = 0; r < CL; ++r) {  // fixed rank order: bit-deterministic
      const float4 v = ld_remote_f4(map_rank(part_s + off, static_cast<uint32_t>(r)));
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    __nv_bfloat162 lo = __floats2bfloat162_rn(s.x, s.y), hi = __floats2bfloat162_rn(s.z, s.w);
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&lo);
    pk.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(out + (int64_t)t * stride_o + n0 + static_cast<int>(rank) * C::RPR + j * 4) = pk;
  }

  cluster_arrive();  // nobody exits while another CTA may still read its shared memory
  cluster_wait();
#endif
}

template <int NT, int ROWS, int KPER, int CL, int NG>
struct Inst {
  using C = Cfg<NT, ROWS, KPER, CL, NG>;
  static void prep() {
    static bool done = false;
    if (!done) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(ouns_kernel<NT, ROWS, KPER, CL, NG>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
      done = true;
    }
  }
  static cudaLaunchConfig_t config(int64_t n, cudaStream_t stream, cudaLaunchAttribute* attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>((n / ROWS) * CL), 1, 1);
    cfg.blockDim = dim3(C::THREADS, 1, 1);
    cfg.dynamicSmemBytes = C::SMEM;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
  }
  static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, int64_t n, int64_t k) {
    TORCH_CHECK(k == (int64_t)KPER * CL && n % ROWS == 0, "ouns: tiling");
    prep();
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = config(n, at::cuda::getCurrentCUDAStream(), attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, ouns_kernel<NT, ROWS, KPER, CL, NG>,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int64_t)x.stride(0),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), (int64_t)w.stride(0),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0), (int)x.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  static int64_t occupancy(int64_t n) {
    prep();
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = config(n, at::cuda::getCurrentCUDAStream(), attr);
    int clusters = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, ouns_kernel<NT, ROWS, KPER, CL, NG>, &cfg));
    return clusters;
  }
  static std::string attrs() {
    prep();
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, ouns_kernel<NT, ROWS, KPER, CL, NG>));
    return "NT=" + std::to_string(NT) + " ROWS=" + std::to_string(ROWS) + " CL=" + std::to_string(CL) +
           " regs=" + std::to_string(a.numRegs) + " local(spill)=" + std::to_string(a.localSizeBytes) +
           " static_smem=" + std::to_string(a.sharedSizeBytes) + " dyn_smem=" + std::to_string(C::SMEM) +
           " max_threads=" + std::to_string(a.maxThreadsPerBlock);
  }
};

// The unpooler n == 1024 slice only (k 4096, m 1..8, NT = 8): NO o instantiation is compiled.
template <class F>
bool dispatch(int64_t cl, int64_t m, F&& f) {
  if (m < 1 || m > 8) return false;
  if (cl == 8) { f(Inst<8, 64, 512, 8, 4>{}, 1024, 4096); return true; }
  if (cl == 4) { f(Inst<8, 32, 1024, 4, 4>{}, 1024, 4096); return true; }
  return false;
}

}  // namespace

void ouns_run(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t cl) {
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.data_ptr<at::BFloat16>() != nullptr, "ouns: x");
  TORCH_CHECK(w.dim() == 2 && w.stride(1) == 1 && w.stride(0) % 8 == 0, "ouns: w");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "ouns: align");
  TORCH_CHECK(out.dim() == 2 && out.stride(1) == 1 && out.stride(0) % 4 == 0 && out.size(0) == x.size(0), "ouns: out");
  const bool ok = dispatch(cl, x.size(0), [&](auto inst, int64_t n, int64_t k) {
    using I = decltype(inst);
    TORCH_CHECK(x.size(1) == k && w.size(1) == k && w.size(0) >= n && out.size(1) == n, "ouns: shapes");
    I::launch(x, w, out, n, k);
  });
  TORCH_CHECK(ok, "ouns: unsupported (cl, m)");
}

int64_t ouns_occupancy(int64_t cl) {
  int64_t r = -1;
  dispatch(cl, 8, [&](auto inst, int64_t n, int64_t) { r = decltype(inst)::occupancy(n); });
  return r;
}

std::string ouns_attrs(int64_t cl) {
  std::string r = "unsupported";
  dispatch(cl, 8, [&](auto inst, int64_t, int64_t) { r = decltype(inst)::attrs(); });
  return r;
}

"""


def _build_ouns():
    """Compile the CUDA module once per process, at import (never during graph capture), under its
    OWN extension name. No -std flag is passed (torch's default compiled this kernel in 31 s on
    3f3330f7). Built for 9.0a only (TORCH_CUDA_ARCH_LIST set around the build and restored after).
    On any build failure the nvcc error tail is printed and master's routing is kept."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    flags = ["-O3", "-lineinfo"]
    try:
        from torch.utils.cpp_extension import load_inline
        print(f"[ouns] building {OUNS_EXT_NAME}: TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r} "
              f"(was {prev!r}), device cc {torch.cuda.get_device_capability()}, extra nvcc flags {flags}",
              flush=True)
        ext = load_inline(
            name=OUNS_EXT_NAME,
            cpp_sources=_OUNS_CPP,
            cuda_sources=_OUNS_CUDA,
            functions=["ouns_run", "ouns_occupancy", "ouns_attrs"],
            extra_cuda_cflags=flags,
            verbose=False,
        )
    except Exception as e:  # a build failure must never raise out of the import
        msg = str(e)
        print(f"[ouns] load_inline FAILED after {time.perf_counter() - t0:.1f}s ({type(e).__name__}); "
              f"nvcc error tail:\n{msg[-3000:]}\n[ouns] master path kept", flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    print(f"[ouns] load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext


_OUNS_EXT = _build_ouns() if torch.cuda.is_available() else None


def _ouns_route_ok(x: torch.Tensor, w: torch.Tensor, cfg, state: dict) -> bool:
    """The unpooler's n == 1024 slice at 1 <= m <= 8, the cluster dim chosen in prepare()
    (state["ouns_cl"] > 0 means self-check passed AND the graph-replay gate won), 16-byte-aligned
    bf16 rows. Branches on cfg, shapes, strides and a prepare() constant only."""
    if not state["ouns_cl"]:   # constant set in prepare() (0 when no module, check or gate failed)
        return False
    m = cfg["m"]
    return (1 <= m <= OUNS_MAX_M and cfg["n"] == OUNS_N and cfg["k"] == OUNS_K
            and x.dim() == 2 and x.shape[0] == m and x.shape[1] == OUNS_K
            and x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
            and w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[0] >= OUNS_N
            and w.shape[1] == OUNS_K and w.stride(1) == 1 and w.stride(0) % 8 == 0
            and w.data_ptr() % 16 == 0)


def _ouns_forward(x: torch.Tensor, w: torch.Tensor, cl: int) -> torch.Tensor:
    """x bf16 [m, 4096] @ w[:1024].T -> bf16 [m, 1024], ONE cluster launch, w read in place."""
    out = torch.empty((x.shape[0], OUNS_N), dtype=torch.bfloat16, device=x.device)
    _OUNS_EXT.ouns_run(x, w, out, cl)
    return out


def _ouns_check(cl: int, w: torch.Tensor) -> str:
    """In-process self-check of one cluster dim at m = 1, 5, 8 (plus a final m = 1): synthetic x
    against an fp64 reference, every case launched twice and required bit-identical. A flushed
    print precedes each launch and a synchronize plus a flushed print follows it, so a fault names
    its kernel. Returns "" on pass, else why."""
    n, k = OUNS_N, OUNS_K
    dev = w.device
    wd = w[:n].double()
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    cases = [(m, m >= 2) for m in OUNS_CHECK_MS] + [(1, True)]
    for m, onehot in cases:
        x = torch.randn((m, k), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
        if onehot:  # last row: one power-of-two entry, its output is a weight column exactly
            x[m - 1].zero_()
            x[m - 1, (37 * m + 5) % k] = 2.0
        ref = x.double() @ wd.T
        ref_b = ref.to(torch.bfloat16).double()
        outs = []
        for rep in (1, 2):
            print(f"[ouns] self-check launch: ouns_kernel cl={cl} m={m} run={rep} ...", flush=True)
            outs.append(_ouns_forward(x, w, cl))
            torch.cuda.synchronize()
            print(f"[ouns] self-check done:   ouns_kernel cl={cl} m={m} run={rep}", flush=True)
        if not torch.equal(outs[0], outs[1]):
            return f"m={m}: second launch differs from the first"
        o = outs[0].double()
        if not bool(torch.isfinite(o).all()):
            return f"m={m}: non-finite output"
        err = float((o - ref).norm() / ref.norm().clamp_min(1e-30))
        if not err <= 3e-3:
            return f"m={m}: rel L2 {err:.2e} (bound 3e-3)"
        if onehot:
            probe = float((o[m - 1] - ref_b[m - 1]).abs().max())
            if not probe <= 1e-4:
                return f"m={m}: one-hot row off by {probe:.2e} (bound 1e-4)"
    return ""


def _ouns_median(v):
    s = sorted(v)
    return s[len(s) // 2]


def _ouns_incumbent(state: dict):
    """The restack's own path for the unpooler slice (what kernel_fn runs without this route):
    _swapab16_forward 'unpooler_slice' when it was prepared, else the split-K8 ticket path."""
    w = state["unpooler"]
    stages = state.get("swapab16_stages", {})
    if "unpooler_slice" in stages:
        ws, cnt = _swapab16_buffers(state, "unpooler_slice")
        return "swapab16", (lambda x: _swapab16_forward("unpooler_slice", x, w, ws, cnt,
                                                       stages["unpooler_slice"]))
    return "unpool_splitk8", (lambda x: _unpool_forward(x, w, state["unpool_ws"], state["unpool_cnt"]))


def _ouns_gate(cl: int, state: dict) -> bool:
    """prepare()-only: does the cluster route beat the incumbent on the judge's measurement?
    For m in OUNS_GATE_MS: capture each path once as a CUDA graph (synthetic x, output allocated
    from the graph's pool), then OUNS_GATE_REPS times replay both with a >= 256 MB flush before
    each replay (the order alternates every rep, so drift and which-went-first cancel), time each
    replay with CUDA events and take medians. A trivial-kernel graph measured the same way gives
    the fixed replay cost, subtracted from both (the judge subtracts about 4 us). The route is
    enabled only if incumbent / cluster >= OUNS_MIN_WIN at every m. Eager cold medians are printed
    too, for continuity with 3f3330f7, but never decide. Any exception keeps master's path."""
    w = state["unpooler"]
    dev = w.device
    inc_name, inc = _ouns_incumbent(state)
    flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)
    scratch = torch.zeros(1, dtype=torch.float32, device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def replay_us(g) -> float:
        flush.zero_()
        ev[0].record()
        g.replay()
        ev[1].record()
        ev[1].synchronize()
        return ev[0].elapsed_time(ev[1]) * 1e3

    def capture(fn):
        for _ in range(3):  # warm on the current stream before capture
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    wins = []
    g_base = None
    try:
        g_base = capture(lambda: scratch.add_(1.0))
        base = _ouns_median([replay_us(g_base) for _ in range(OUNS_GATE_REPS)])
        for m in OUNS_GATE_MS:
            x = torch.randn((m, OUNS_K), device=dev, dtype=torch.float32).to(torch.bfloat16)
            fns = {"cluster": (lambda x=x: _ouns_forward(x, w, cl)), "incumbent": (lambda x=x: inc(x))}
            graphs = {name: capture(fn) for name, fn in fns.items()}
            times = {"cluster": [], "incumbent": []}
            for rep in range(OUNS_GATE_REPS):
                order = ("cluster", "incumbent") if rep % 2 == 0 else ("incumbent", "cluster")
                for name in order:
                    times[name].append(replay_us(graphs[name]))
            med = {name: _ouns_median(v) for name, v in times.items()}
            adj = {name: max(v - base, 0.25) for name, v in med.items()}
            ratio = adj["incumbent"] / adj["cluster"]
            # eager cold, continuity only (includes Python / Triton launch cost, biased to cluster)
            eager = {}
            for name, fn in fns.items():
                vals = []
                for _ in range(20):
                    flush.zero_()
                    ev[0].record()
                    fn()
                    ev[1].record()
                    ev[1].synchronize()
                    vals.append(ev[0].elapsed_time(ev[1]) * 1e3)
                eager[name] = _ouns_median(vals)
            win = ratio >= OUNS_MIN_WIN
            wins.append(win)
            print(f"[ouns] gate m={m} (graph replay, {OUNS_FLUSH_BYTES >> 20} MB flush, {OUNS_GATE_REPS} reps, "
                  f"interleaved): cluster median {med['cluster']:.2f} us, {inc_name} median "
                  f"{med['incumbent']:.2f} us, empty-graph baseline {base:.2f} us -> net "
                  f"{adj['cluster']:.2f} vs {adj['incumbent']:.2f} us, incumbent/cluster {ratio:.3f}x "
                  f"(need >= {OUNS_MIN_WIN}) -> {'WIN' if win else 'LOSS'}; eager-cold (not used) "
                  f"cluster {eager['cluster']:.2f} / {inc_name} {eager['incumbent']:.2f} us", flush=True)
            del graphs
    except Exception as e:
        print(f"[ouns] gate failed ({type(e).__name__}: {e}); master path kept", flush=True)
        return False
    finally:
        # Release the 256 MB flush buffer and the baseline graph; prepare() empties the caching
        # allocator before it returns.
        flush = None
        g_base = None
        scratch = None
        torch.cuda.synchronize()
    return len(wins) == len(OUNS_GATE_MS) and all(wins)


def _ouns_prepare(state: dict) -> None:
    """Query co-residency (cluster 8, else cluster 4, else master), self-check the chosen cluster
    dim, run the graph-replay gate, and fix state["ouns_cl"] (0 = restack's path) for the run."""
    state["ouns_cl"] = 0
    if _OUNS_EXT is None:
        print("[ouns] PATH CHOSEN: unpooler slice m<=8 -> restack path (swapab16): no CUDA module", flush=True)
        return
    w = state["unpooler"]
    if not (w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[0] >= OUNS_N
            and w.shape[1] == OUNS_K and w.stride(1) == 1 and w.stride(0) % 8 == 0
            and w.data_ptr() % 16 == 0):
        print(f"[ouns] PATH CHOSEN: unpooler slice m<=8 -> restack path: weight {tuple(w.shape)} unexpected",
              flush=True)
        return
    chosen = 0
    try:
        for cl in (8, 4):
            need = OUNS_CLUSTERS_NEEDED[cl]
            try:
                print(f"[ouns] cl={cl} {_OUNS_EXT.ouns_attrs(cl)}", flush=True)
                have = int(_OUNS_EXT.ouns_occupancy(cl))
            except Exception as e:
                print(f"[ouns] cl={cl}: attribute/occupancy query failed ({e!r})", flush=True)
                continue
            print(f"[ouns] cl={cl}: cudaOccupancyMaxActiveClusters={have}, needs {need}", flush=True)
            if have < need:
                print(f"[ouns] cl={cl}: too few co-resident clusters, trying the next", flush=True)
                continue
            try:
                why = _ouns_check(cl, w)
            except Exception as e:
                why = f"raised {e!r}"
            if why:
                print(f"[ouns] SELF-CHECK FAILED cl={cl}: {why}", flush=True)
                continue
            chosen = cl
            break
        if chosen and _ouns_gate(chosen, state):
            state["ouns_cl"] = chosen
    except Exception as e:  # never raises: any surprise leaves the restack path
        print(f"[ouns] prepare raised ({type(e).__name__}: {e}); restack path kept", flush=True)
        state["ouns_cl"] = 0
    torch.cuda.synchronize()
    for key in ("unpool_cnt", "swapab16_cnt_unpooler_slice"):
        if key in state:
            state[key].zero_()
    if state["ouns_cl"]:
        print(f"[ouns] PATH CHOSEN: unpooler slice m<=8 (n=1024) -> CUDA cluster/DSMEM split-K, cluster dim "
              f"{state['ouns_cl']}; self-check passed (m {OUNS_CHECK_MS}) and graph-replay gate won at m "
              f"{OUNS_GATE_MS} (see the [ouns] gate lines for both medians)", flush=True)
    else:
        print("[ouns] PATH CHOSEN: unpooler slice m<=8 (n=1024) -> restack path (swapab16); cluster route off "
              "(see the [ouns] lines above for why)", flush=True)


# --- (8) router, 1 <= m <= 8: hand-written CUDA split-K reduced through cluster DSMEM ---------
# (hypothesis restack-router-cluster-dsmem-m8, the router kernel of branch
# hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite / job dc5d03c4, ported gated to
# m <= 8 only.)
#
# One launch, no memset, no atomics, no global workspace (output is torch.empty). 128 CTAs = 16
# clusters of CS = 8 CTAs. Cluster c owns output columns [16c, 16c + 16) of the 256 experts, CTA
# rank r owns K [r * 1024, +1024). Swap-AB mma.sync m16n8k16 bf16 -> fp32 with the weight as the A
# operand (each 16-row A tile = 8 hi rows over the same 8 lo rows) and the <= 8 tokens as B.
# Each warp loads its own K quarter with cp.async.cg (XOR-swizzled smem), the 4 warp partials are
# summed in fixed order into a CTA partial in smem, then barrier.cluster, CTA rank r reduces
# output columns {2r, 2r + 1} over ranks 0..7 in fixed order via mapa + ld.shared::cluster and
# stores fp32 out, and a second barrier.cluster keeps every CTA's smem alive for its peers.
# No mbarrier and no spin wait exists in the kernel, so there is nothing to bound.
#
# Differences from the source: only the NT = 8 instance at CS = 8 is compiled (the CS = 4 and
# NT = 16 / 32 variants are dropped); the mode = 0 diagnostic (no DSMEM reduce, wrong result) is
# not ported; the timing diagnostics are not ported; the self-check runs at m in {1, 5, 8}. The
# occupancy gate is kept: fewer than 16 active clusters of 8 -> swap-AB router (no CS = 4
# fallback). The gain this hypothesis expects is dropping the memset node (swap-AB router =
# kernel + torch.zeros), so the m <= 8 path launches exactly one kernel and no memset.
ROUTER_CLUSTER_MAX_M = 8
ROUTER_CLUSTER_CHECK_MS = (1, 5, 8)

_RC_CPP = r"""
#include <torch/extension.h>
#include <vector>
void router_cluster(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out);
std::vector<int64_t> router_info();
"""

_RC_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int THREADS = 128;     // 4 warps, each owns a K quarter of the CTA's K range
constexpr int NWARP = 4;
constexpr int N_OUT = 256;
constexpr int K_TOT = 8192;
constexpr int NCTA = 128;
constexpr int SUB_K = 64;        // K per cp.async commit group (8 x 16 B per smem row)

template <int CS, int NT>
struct Cfg {
  static constexpr int KC = K_TOT / CS;        // K per CTA
  static constexpr int QK = KC / NWARP;        // K per warp
  static constexpr int COLS = 2 * CS;          // output columns per cluster
  static constexpr int G = COLS / 8;           // 8-column groups, one 16-row stacked hi/lo A tile each
  static constexpr int RB = KC * 2;            // smem row bytes
  static constexpr int W_ROWS = 16 * G;
  static constexpr int W_BYTES = W_ROWS * RB;  // 64 KB for both CS
  static constexpr int X_BYTES = NT * RB;
  static constexpr int P_BYTES = COLS * NT * 4;
  static constexpr int SMEM = W_BYTES + X_BYTES + P_BYTES;
  static constexpr int NTILE = NT / 8;
  static constexpr int NSUB = QK / SUB_K;
  static_assert(NCTA % CS == 0 && NSUB >= 1 && NSUB <= 8 && SUB_K * NSUB == QK, "tiling");
  static_assert(NWARP * COLS * NT * 4 <= W_BYTES, "warp partials alias the weight region");
  static_assert(SMEM <= 226 * 1024, "smem");
  static_assert(2 * NT <= THREADS, "reduce threads");
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_dyn(int n) {  // n is a compile-time constant after unrolling
  switch (n) {
    case 7: cp_wait<7>(); break;
    case 6: cp_wait<6>(); break;
    case 5: cp_wait<5>(); break;
    case 4: cp_wait<4>(); break;
    case 3: cp_wait<3>(); break;
    case 2: cp_wait<2>(); break;
    case 1: cp_wait<1>(); break;
    default: cp_wait<0>(); break;
  }
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
// D = A B + D, m16n8k16, A row-major bf16 (4 regs), B col-major bf16 (2 regs), fp32 accumulate.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cluster_sync_rel_acq() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t map_rank(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ float ld_cluster_f32(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(addr) : "memory");
  return v;
}

// out[m, 256] fp32 = x[m, 8192] bf16 @ (hi + lo)[256, 8192]^T, 1 <= m <= NT. Grid 128 CTAs, cluster
// dim (CS, 1, 1): blockIdx.x = cluster * CS + rank. Only <8, 8> is instantiated.
template <int CS, int NT>
__global__ void __launch_bounds__(THREADS, 1)
router_kernel(const __nv_bfloat16* __restrict__ x, int64_t sx,
              const __nv_bfloat16* __restrict__ hi, const __nv_bfloat16* __restrict__ lo,
              float* __restrict__ out, int m) {
  using C = Cfg<CS, NT>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x % CS;
  const int col0 = (blockIdx.x / CS) * C::COLS;
  const int kbase = rank * C::KC;
  const uint32_t ws_u = smem_u32(smem);
  const uint32_t xs_u = ws_u + C::W_BYTES;
  float* P = reinterpret_cast<float*>(smem + C::W_BYTES + C::X_BYTES);   // [COLS][NT] CTA partial
  const int kw = warp * C::QK;  // CTA-local K start of this warp's quarter

  // ---- issue every load for this warp's quarter up front, one commit group per 64 K ----
#pragma unroll
  for (int s = 0; s < C::NSUB; ++s) {
    const int k0 = kw + s * SUB_K;  // multiple of 64: chunk index k0/8 is a multiple of 8
#pragma unroll
    for (int j = 0; j < C::W_ROWS * 8 / 32; ++j) {
      const int idx = j * 32 + lane;
      const int row = idx >> 3, c = idx & 7;
      const int g = row >> 4, half = (row >> 3) & 1, i = row & 7;
      const __nv_bfloat16* src = (half ? lo : hi) + (int64_t)(col0 + g * 8 + i) * K_TOT + kbase + k0 + c * 8;
      cp_async16(ws_u + row * C::RB + ((k0 / 8 + (c ^ (row & 7))) << 4), src, 16);
    }
#pragma unroll
    for (int j = 0; j < NT * 8 / 32; ++j) {
      const int idx = j * 32 + lane;
      const int tok = idx >> 3, c = idx & 7;
      const bool ok = tok < m;
      const __nv_bfloat16* src = ok ? (x + (int64_t)tok * sx + kbase + k0 + c * 8) : x;
      cp_async16(xs_u + tok * C::RB + ((k0 / 8 + (c ^ (tok & 7))) << 4), src, ok ? 16 : 0);
    }
    cp_commit();
  }

  // ---- main loop: this warp only waits for its own copies ----
  float acc[C::G][C::NTILE][4];
#pragma unroll
  for (int g = 0; g < C::G; ++g)
#pragma unroll
    for (int t = 0; t < C::NTILE; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[g][t][e] = 0.f;

  const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;  // ldmatrix A row supplied by this lane
  const int ahalf = (lane >> 4) & 1;                    // k chunk (0: k0-7, 1: k8-15)
  const int xr = lane & 7, xq = lane >> 3;
#pragma unroll
  for (int s = 0; s < C::NSUB; ++s) {
    cp_wait_dyn(C::NSUB - 1 - s);
    __syncwarp();
#pragma unroll
    for (int kp = 0; kp < SUB_K / 32; ++kp) {
      const int cbase = (kw + s * SUB_K) / 8 + kp * 4;  // 16 B chunk index, multiple of 4
      uint32_t b[C::NTILE][4];  // b[t][0..1]: k16 step 2kp, b[t][2..3]: k16 step 2kp + 1
#pragma unroll
      for (int t = 0; t < C::NTILE; ++t)
        ldsm_x4(b[t], xs_u + (t * 8 + xr) * C::RB + (((cbase + xq) ^ xr) << 4));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int g = 0; g < C::G; ++g) {
          uint32_t a[4];
          ldsm_x4(a, ws_u + (g * 16 + arow) * C::RB + (((cbase + h * 2 + ahalf) ^ (arow & 7)) << 4));
#pragma unroll
          for (int t = 0; t < C::NTILE; ++t) mma_bf16(acc[g][t], a, b[t][2 * h], b[t][2 * h + 1]);
        }
      }
    }
  }

  // ---- warp partials (aliasing the weight region) -> CTA partial P, fixed warp order ----
  __syncthreads();  // every warp is done reading the weight / x tiles
  float* WP = reinterpret_cast<float*>(smem);  // [NWARP][COLS][NT]
#pragma unroll
  for (int g = 0; g < C::G; ++g)
#pragma unroll
    for (int t = 0; t < C::NTILE; ++t) {
      // c0, c1: hi row lane/4, tokens 2*(lane%4) + {0, 1}; c2, c3: the lo row of the same column.
      const int col = g * 8 + (lane >> 2), tk = t * 8 + (lane & 3) * 2;
      WP[(warp * C::COLS + col) * NT + tk] = acc[g][t][0] + acc[g][t][2];
      WP[(warp * C::COLS + col) * NT + tk + 1] = acc[g][t][1] + acc[g][t][3];
    }
  __syncthreads();
  for (int e = tid; e < C::COLS * NT; e += THREADS) {
    float s = WP[e];
#pragma unroll
    for (int w = 1; w < NWARP; ++w) s += WP[w * C::COLS * NT + e];
    P[e] = s;
  }
  __syncthreads();

  const int cj = tid / NT, tok = tid % NT;  // meaningful for tid < 2 * NT
  const bool mine = tid < 2 * NT && tok < m;
  const int colo = col0 + 2 * rank + cj;
  cluster_sync_rel_acq();  // every CTA's P is complete and visible cluster-wide
  if (mine) {
    const uint32_t loc = smem_u32(P + (2 * rank + cj) * NT + tok);
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < CS; ++q) s += ld_cluster_f32(map_rank(loc, q));  // fixed rank order
    out[(int64_t)tok * N_OUT + colo] = s;
  }
  cluster_sync_rel_acq();  // peers are done reading this CTA's smem before it exits
}

template <int CS, int NT>
void set_attr() {
  static bool done = false;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(router_kernel<CS, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Cfg<CS, NT>::SMEM));
    done = true;
  }
}

template <int CS, int NT>
void launch(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out) {
  set_attr<CS, NT>();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NCTA);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = Cfg<CS, NT>::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, router_kernel<CS, NT>,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int64_t>(x.stride(0)),
      reinterpret_cast<const __nv_bfloat16*>(hi.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(lo.data_ptr()),
      out.data_ptr<float>(), static_cast<int>(x.size(0))));
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // cudaGetLastError, raises
}

template <int CS, int NT>
std::vector<int64_t> info() {
  set_attr<CS, NT>();
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, router_kernel<CS, NT>));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NCTA);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = Cfg<CS, NT>::SMEM;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int nclusters = -1;
  if (cudaOccupancyMaxActiveClusters(&nclusters, router_kernel<CS, NT>, &cfg) != cudaSuccess) {
    (void)cudaGetLastError();
    nclusters = -1;
  }
  return {fa.numRegs, static_cast<int64_t>(fa.localSizeBytes), static_cast<int64_t>(fa.sharedSizeBytes),
          static_cast<int64_t>(Cfg<CS, NT>::SMEM), nclusters};
}

}  // namespace

void router_cluster(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out) {
  const int64_t m = x.size(0);
  TORCH_CHECK(x.dim() == 2 && m >= 1 && m <= 8 && x.size(1) == K_TOT, "router_cluster: x shape");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && hi.scalar_type() == at::kBFloat16 &&
              lo.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kFloat, "router_cluster: dtypes");
  TORCH_CHECK(hi.is_contiguous() && lo.is_contiguous() && hi.size(0) == N_OUT && hi.size(1) == K_TOT &&
              lo.size(0) == N_OUT && lo.size(1) == K_TOT, "router_cluster: weights");
  TORCH_CHECK(x.stride(1) == 1 && (m == 1 || x.stride(0) % 8 == 0) &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(hi.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(lo.data_ptr()) % 16 == 0, "router_cluster: x strides / alignment");
  TORCH_CHECK(out.is_contiguous() && out.size(0) == m && out.size(1) == N_OUT, "router_cluster: out");
  launch<8, 8>(x, hi, lo, out);
}

// {regs, local (spill) bytes, static smem, dynamic smem, max active clusters (-1: query failed)}
std::vector<int64_t> router_info() { return info<8, 8>(); }
"""


def _build_router_cluster():
    """Compile the CUDA module once per process, at import (never during graph capture). Nothing
    here needs sm_90a-only PTX (mma.sync, cp.async.cg, cluster launch, mapa, barrier.cluster are
    baseline sm_90), but the arch list is pinned to 9.0a around the build and restored after.
    Any failure keeps the swap-AB router path."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    flags = ["-O3", "-lineinfo"]
    print(f"[router_cluster] building: TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']} "
          f"extra_cuda_cflags={flags}", flush=True)
    try:
        from torch.utils.cpp_extension import load_inline
        ext = load_inline(
            name="muse_bf16_router_cluster_m8_v1",
            cpp_sources=_RC_CPP,
            cuda_sources=_RC_CUDA,
            functions=["router_cluster", "router_info"],
            extra_cuda_cflags=flags,
            verbose=False,
        )
    except Exception as e:
        print(f"[router_cluster] load_inline failed after {time.perf_counter() - t0:.1f}s, "
              f"swap-AB router path kept: {e!r}", flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    print(f"[router_cluster] load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext


_RC_EXT = _build_router_cluster() if torch.cuda.is_available() else None


def _router_cluster_route(x: torch.Tensor, cfg, state: dict) -> bool:
    """True when the CUDA cluster router runs for this call. Branches on cfg, shapes, strides,
    alignment and the prepare() constant only, never on input values."""
    if not state["router_cluster_ok"] or x.dim() != 2:   # constant set in prepare(); False if no module
        return False
    m = x.shape[0]
    return (1 <= m <= ROUTER_CLUSTER_MAX_M and cfg["m"] == m and cfg["n"] == ROUTER_N
            and cfg["k"] == ROUTER_K and x.shape[1] == ROUTER_K and x.dtype == torch.bfloat16
            and x.is_cuda and x.stride(1) == 1 and (m == 1 or x.stride(0) % 8 == 0)
            and x.data_ptr() % 16 == 0)


def _router_cluster_forward(x: torch.Tensor, state: dict) -> torch.Tensor:
    """x bf16 [m <= 8, 8192] @ (hi + lo).T -> fp32 [m, 256]: ONE cluster launch, no memset."""
    out = torch.empty((x.shape[0], ROUTER_N), dtype=torch.float32, device=x.device)
    _RC_EXT.router_cluster(x, state["router_hi"], state["router_lo"], out)
    return out


def _rc_check(state: dict, ms) -> str:
    """In-process self-check at the token counts ms, each launched twice, with a flushed print
    before and a synchronize + flushed print after every launch so a fault names its kernel.
    Returns "" if it passes, else why. Bounds: rel L2 <= 2e-5 against x.double() @
    router.double().T, worst row <= 1e-4, the one-hot power-of-two row within 1e-6 of the
    hi + lo reconstruction, finite, run 2 bit-identical."""
    router, hi, lo = state["router"], state["router_hi"], state["router_lo"]
    dev = router.device
    k = router.shape[1]
    rd = router.double()
    hld = hi.double() + lo.double()
    g = torch.Generator(device=dev)
    g.manual_seed(2468)
    for m in ms:
        x = torch.randn((m, k), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
        if m >= 2:  # last row: a single power-of-two entry, its output is a (hi + lo) weight column
            x[m - 1].zero_()
            x[m - 1, (37 * m + 5) % k] = 2.0 ** ((m % 5) - 2)
        ref = x.double() @ rd.T
        ref_hl = x.double() @ hld.T
        outs = []
        for run in (1, 2):
            print(f"[router_cluster] launching router_kernel<8,8> m={m} run {run}", flush=True)
            o = _router_cluster_forward(x, state)
            torch.cuda.synchronize()
            print(f"[router_cluster] router_kernel<8,8> m={m} run {run} finished OK", flush=True)
            outs.append(o)
        if not torch.equal(outs[0], outs[1]):
            return f"m={m}: second launch differs from the first"
        o = outs[0].double()
        if not bool(torch.isfinite(o).all()):
            return f"m={m}: non-finite output"
        err = float((o - ref).norm() / ref.norm().clamp_min(1e-30))
        row_ref = ref.norm(dim=1)
        row_den = torch.maximum(row_ref, 0.1 * row_ref.mean()).clamp_min(1e-30)
        row_err = float(((o - ref).norm(dim=1) / row_den).max())
        if not (err <= 2e-5 and row_err <= 1e-4):
            return f"m={m}: rel L2 {err:.2e} (bound 2e-5), worst row {row_err:.2e} (bound 1e-4)"
        if m >= 2:
            oh = float((o[m - 1] - ref_hl[m - 1]).norm() / ref_hl[m - 1].norm().clamp_min(1e-30))
            if not oh <= 1e-6:
                return f"m={m}: one-hot row off the hi+lo reconstruction by {oh:.2e} (bound 1e-6)"
    return ""


def _router_cluster_prepare(state: dict) -> None:
    """Set the constant state["router_cluster_ok"]: True iff the extension built, the router
    weight is the expected [256, 8192] fp32 CUDA tensor, cudaOccupancyMaxActiveClusters reports
    >= 16 active clusters for CS = 8 NT = 8, and the self-check at m in {1, 5, 8} passes. Derived
    from synthetic data and the weights only; never raises (any failure keeps the swap-AB router)."""
    ok = False
    state["router_cluster_ok"] = False
    router = state["router"]
    try:
        if (_RC_EXT is None or tuple(router.shape) != (ROUTER_N, ROUTER_K)
                or router.dtype != torch.float32 or not router.is_cuda):
            raise RuntimeError("extension missing or router weight shape/dtype unexpected")
        regs, spill, ssmem, dsmem, ncl = _RC_EXT.router_info()
        print(f"[router_cluster] cs=8 nt=8: regs={regs} local(spill)={spill} B static smem={ssmem} B "
              f"dynamic smem={dsmem} B max active clusters={ncl}", flush=True)
        if ncl < 16:
            print(f"[router_cluster] cs=8 skipped, max active clusters {ncl} < 16", flush=True)
        else:
            why = _rc_check(state, ROUTER_CLUSTER_CHECK_MS)
            if why:
                print(f"[router_cluster] cs=8 self-check FAILED: {why}", flush=True)
            else:
                print(f"[router_cluster] cs=8 self-check passed (m in {ROUTER_CLUSTER_CHECK_MS})", flush=True)
                ok = True
    except Exception as e:
        print(f"[router_cluster] prepare failed ({e!r}); swap-AB router path for every m", flush=True)
        ok = False
    state["router_cluster_ok"] = ok
    print("[router_cluster] PATH CHOSEN router m<=8 (n=256): "
          + ("CUDA cluster kernel router_kernel<8,8>, no memset" if ok else "swap-AB Triton router (cluster off)")
          + "; 9 <= m <= 32: swap-AB Triton router; m > 32: hilo", flush=True)


def _prepare_state(weights: Weights, flags=None) -> dict:
    """Build one state. flags None: run the router-cluster and unpooler-cluster self-checks and the
    unpooler gate (the state the judge keeps). flags a dict: the second state of the prepare()-time
    A/B self-check, which inherits the first state's three dispatch flags instead of re-deciding."""
    t_all = time.perf_counter()
    marks = []

    def tick(label, t0):
        dt = time.perf_counter() - t0
        marks.append((label, dt))
        print(f"[prepare-time] {label}: {dt:.2f}s", flush=True)
        return time.perf_counter()

    t = time.perf_counter()
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16).contiguous()
    lo = (router - hi.float()).to(torch.bfloat16).contiguous()
    state["router_hi"] = hi
    state["router_lo"] = lo
    # Lossless transposed bf16 copies [K, N] row-major (truly contiguous, not a .t() view) for
    # the gated pooler/unpooler NN path; the original [N, K] weights stay for every other path.
    for name in NN_GATED_LAYERS:
        state[name + "_t"] = state[name].t().contiguous()
    state["nn_ok"] = True   # constant; the A/B self-check may turn it off for the run
    state["nn_p256_ok"] = True   # pooler 129..256 only; own flag, own self-check cell
    t = tick("weight splits + NN copies", t)
    # Stage choice, side-by-side diagnostics and warm-up on synthetic inputs.
    _router_prepare(state)
    t = tick("router triton warm-up", t)
    # (1) Pooler split-K scratch (fp32 partials, overwritten every call) and per-N-tile tickets.
    pooler = state["pooler"]
    state["pooler_ws"] = torch.empty((POOLER_SPLIT_K, POOLER_MAX_M, POOLER_N),
                                     dtype=torch.float32, device=pooler.device)
    state["pooler_cnt"] = torch.zeros(POOLER_N // POOLER_BLOCK_N, dtype=torch.int32, device=pooler.device)
    pcfg = {"n": POOLER_N, "k": POOLER_K}
    for m in range(1, POOLER_MAX_M + 1):
        x = torch.randn((m, POOLER_K), device=pooler.device, dtype=torch.float32).to(torch.bfloat16)
        if _pooler_shape_ok(x, pooler, dict(pcfg, m=m)):
            _pooler_forward(x, pooler, state["pooler_ws"], state["pooler_cnt"])
    # (2) qkv_gate m <= 16: persistent split-K workspace and per-n-tile ticket counters (own keys).
    qkv = state["qkv_gate"]
    state["qkv_ws"] = torch.empty((QKV_SPLIT_K, QKV_MAX_M, QKV_N), dtype=torch.float32,
                                  device=qkv.device)
    state["qkv_cnt"] = torch.zeros((QKV_N // QKV_BLOCK_N,), dtype=torch.int32, device=qkv.device)
    if qkv.shape == (QKV_N, QKV_K) and qkv.dtype == torch.bfloat16:
        for m in range(1, QKV_MAX_M + 1):
            x = torch.zeros((m, QKV_K), device=qkv.device, dtype=torch.bfloat16)
            _qkv_forward(x, qkv, state["qkv_ws"], state["qkv_cnt"])
    # (3) Unpooler-slice split-K scratch and tickets; warmed on synthetic x against the real
    # (full, stride-4096) unpooler weight.
    unpooler = state["unpooler"]
    state["unpool_ws"] = torch.empty((UNPOOL_SPLIT_K, UNPOOL_MAX_M, UNPOOL_N),
                                     dtype=torch.float32, device=unpooler.device)
    state["unpool_cnt"] = torch.zeros(UNPOOL_N // UNPOOL_BLOCK_N, dtype=torch.int32, device=unpooler.device)
    ucfg = {"n": UNPOOL_N, "k": UNPOOL_K}
    for m in UNPOOL_WARM_MS:
        x = torch.randn((m, UNPOOL_K), device=unpooler.device, dtype=torch.float32).to(torch.bfloat16)
        if _unpool_shape_ok(x, unpooler, dict(ucfg, m=m)):
            _unpool_forward(x, unpooler, state["unpool_ws"], state["unpool_cnt"])
    # (4) o skinny full-K kernel (no workspace).
    o = state["o"]
    state["o_skinny"] = o.shape == (O_N, O_K)
    if state["o_skinny"]:
        _o_skinny_warmup(o)
    # The split-K kernels self-reset, but re-zero every counter after the warm-ups regardless
    # (stream-ordered).
    state["pooler_cnt"].zero_()
    state["qkv_cnt"].zero_()
    state["unpool_cnt"].zero_()
    t = tick("pooler / qkv / unpooler / o triton warm-up", t)
    # (5) Swap-AB at 17 <= m <= 32 for all four bf16 layers (own workspaces / tickets).
    _swapab_prepare(state)
    t = tick("swapab m17-32 warm-up", t)
    # (6) Swap-AB, BLOCK_T 16, at m <= 16 for pooler / qkv_gate / o and the unpooler slice.
    _swapab16_prepare(state)
    t = tick("swapab16 warm-up", t)
    if flags is None:
        # (7) Unpooler slice m <= 8: CUDA cluster + DSMEM split-K behind a graph-replay gate.
        _ouns_prepare(state)
        t = tick("ouns self-check + graph-replay gate", t)
        # (8) Router m <= 8: CUDA cluster + DSMEM split-K (constant state["router_cluster_ok"]).
        _router_cluster_prepare(state)
        t = tick("router cluster self-check", t)
    else:
        state["ouns_cl"] = flags["ouns_cl"]
        state["router_cluster_ok"] = flags["router_cluster_ok"]
        state["nn_ok"] = flags["nn_ok"]
        state["nn_p256_ok"] = flags["nn_p256_ok"]
    print(f"[prepare-time] state build ({'A' if flags is None else 'B'}) total: "
          f"{time.perf_counter() - t_all:.2f}s", flush=True)
    return state


def _ab_flags(state: dict) -> dict:
    return {"ouns_cl": state["ouns_cl"], "router_cluster_ok": state["router_cluster_ok"],
            "nn_ok": state["nn_ok"], "nn_p256_ok": state["nn_p256_ok"]}


def _ab_component(layer: str, m: int) -> str:
    """Which stacked component serves this cell ("" = master's path, nothing to switch off)."""
    if layer == "router" and m <= ROUTER_CLUSTER_MAX_M:
        return "router_cluster_ok"
    if layer == "unpooler" and m <= OUNS_MAX_M:
        return "ouns_cl"
    if layer in NN_GATED_LAYERS and NN_GATED_MIN_M <= m <= NN_GATED_MAX_M:
        return "nn_ok"
    if layer == "pooler" and NN_GATED_MAX_M < m <= NN_POOLER_MAX_M:
        return "nn_p256_ok"
    return ""


def _ab_selfcheck(weights: Weights, state_a: dict) -> None:
    """In-process self-check of the whole dispatch the way the judge runs it: a second state B is
    prepared (inheriting A's flags), kernel_fn is captured as a CUDA graph per state at m in
    {1, 8, 128} for router, unpooler and pooler, and the graphs are replayed alternately A, B, A, B
    with x refilled in place before every replay; every output is compared with an fp32 (router:
    fp64) reference. A miss, or any exception, switches off the component serving that cell in
    BOTH states (router cluster / unpooler cluster / NN layout) and falls back to master's path.
    A flushed print precedes each capture and a synchronize plus flushed print follows each replay
    so a fault names its cell. Never raises."""
    t0 = time.perf_counter()
    try:
        state_b = _prepare_state(weights, flags=_ab_flags(state_a))
    except Exception as e:
        print(f"[ab] second-state build failed ({type(e).__name__}: {e}); NN, router cluster and "
              f"unpooler cluster OFF", flush=True)
        for st in (state_a,):
            st["nn_ok"], st["router_cluster_ok"], st["ouns_cl"] = False, False, 0
            st["nn_p256_ok"] = False
        return
    states = {"A": state_a, "B": state_b}
    dev = state_a["router"].device
    g = torch.Generator(device=dev)
    g.manual_seed(97531)
    bad = {}
    wf = {}
    for layer, n, k, m in AB_CASES:
        comp = _ab_component(layer, m)
        graphs, xs, outs = {}, {}, {}
        try:
            cfg = {"m": m, "n": n, "k": k, "layer": layer}
            for tag, st in states.items():
                xs[tag] = torch.randn((m, k), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
                data = {"x": xs[tag], "config": cfg}
                print(f"[ab] capture kernel_fn {layer} m={m} n={n} state {tag} (component "
                      f"{comp or 'master path'}) ...", flush=True)
                for _ in range(2):
                    kernel_fn(data, st)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    outs[tag] = kernel_fn(data, st)
                torch.cuda.synchronize()
                graphs[tag] = gr
            if layer == "router":
                ref_w = state_a["router"].double()
            else:
                if layer not in wf:
                    wf[layer] = state_a[layer].float()
                ref_w = wf[layer][:n]
            worst = 0.0
            for rnd in range(AB_ROUNDS):
                for tag in ("A", "B"):
                    xs[tag].copy_(torch.randn((m, k), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16))
                    graphs[tag].replay()
                    torch.cuda.synchronize()
                    o = outs[tag].double() if layer == "router" else outs[tag].float()
                    ref = xs[tag].double() @ ref_w.T if layer == "router" else xs[tag].float() @ ref_w.T
                    if not bool(torch.isfinite(o).all()):
                        raise RuntimeError(f"non-finite output (state {tag}, round {rnd})")
                    err = float((o.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30))
                    worst = max(worst, err)
                    bound = AB_ROUTER_REL_L2 if layer == "router" else AB_BF16_REL_L2
                    if err > bound:
                        raise RuntimeError(f"rel L2 {err:.2e} > {bound:.0e} (state {tag}, round {rnd})")
            print(f"[ab] {layer} m={m} n={n}: A,B,A,B replays OK, worst rel L2 {worst:.2e}", flush=True)
        except Exception as e:
            print(f"[ab] MISS {layer} m={m} n={n}: {type(e).__name__}: {e}"
                  + (f" -> switching {comp} off in both states" if comp else
                     " (master path cell: nothing to switch off)"), flush=True)
            if comp:
                bad[comp] = True
        finally:
            graphs.clear()
            outs.clear()
    wf.clear()
    for comp in bad:
        for st in states.values():
            st[comp] = 0 if comp == "ouns_cl" else False
    print(f"[ab] done in {time.perf_counter() - t0:.1f}s; flags now {_ab_flags(state_a)}", flush=True)
    del states, state_b


def prepare(weights: Weights) -> dict:
    """Called once, untimed. Keep the bf16 weights as they are (contiguous), build the router's
    hi + lo bf16 split as [256, 8192] row-major (N x K), pre-compile every m-bucket of the router
    kernel on synthetic bf16 x so the timed call never compiles, decide the three dispatch flags
    (router cluster, unpooler cluster, NN layout), A/B-check the whole dispatch on two states, and
    release every gate / self-check scratch buffer before returning."""
    t0 = time.perf_counter()
    state = _prepare_state(weights)
    t = time.perf_counter()
    _ab_selfcheck(weights, state)
    t = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()   # the gate's 256 MB flush buffer and graph pools are gone by now
    # Re-print the co-residency numbers now that both cluster modules are loaded and checked.
    try:
        if _OUNS_EXT is not None:
            for cl in (8, 4):
                print(f"[ouns] final: cl={cl}: cudaOccupancyMaxActiveClusters={int(_OUNS_EXT.ouns_occupancy(cl))}, "
                      f"needs {OUNS_CLUSTERS_NEEDED[cl]}", flush=True)
        if _RC_EXT is not None:
            print(f"[router_cluster] final: {_RC_EXT.router_info()} (regs, spill B, static smem, dyn smem, "
                  f"max active clusters; needs 16)", flush=True)
    except Exception as e:
        print(f"[prepare] final occupancy query failed ({type(e).__name__}: {e})", flush=True)
    print(f"[prepare] FINAL FLAGS: ouns_cl={state['ouns_cl']} router_cluster_ok={state['router_cluster_ok']} "
          f"nn_ok={state['nn_ok']} nn_p256_ok={state['nn_p256_ok']}; prepare total {time.perf_counter() - t0:.1f}s", flush=True)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    # Shape gate (cfg layer / m / n and shape preconditions only; never on input values). Every
    # dispatch flag (ouns_cl, router_cluster_ok, nn_ok) is read from `state`.
    if layer == "unpooler" and _ouns_route_ok(x, state["unpooler"], cfg, state):
        return _ouns_forward(x, state["unpooler"], state["ouns_cl"])
    key16 = _swapab16_key(layer, cfg)
    if (key16 in state["swapab16_stages"]
            and _swapab16_route_ok(key16, x, state[layer], cfg)):
        ws, cnt = _swapab16_buffers(state, key16)
        return _swapab16_forward(key16, x, state[layer], ws, cnt, state["swapab16_stages"][key16])
    if layer == "pooler" and _pooler_shape_ok(x, state["pooler"], cfg):
        return _pooler_forward(x, state["pooler"], state["pooler_ws"], state["pooler_cnt"])
    if (layer == "qkv_gate" and state["qkv_gate"].shape == (QKV_N, QKV_K)
            and _qkv_route_ok(x, cfg)):
        return _qkv_forward(x, state["qkv_gate"], state["qkv_ws"], state["qkv_cnt"])
    if layer == "unpooler" and _unpool_shape_ok(x, state["unpooler"], cfg):
        return _unpool_forward(x, state["unpooler"], state["unpool_ws"], state["unpool_cnt"])
    if layer == "o" and state["o_skinny"] and _o_skinny_applies(x, cfg):
        return _o_skinny_gemm(x, state["o"])
    if (layer in state["swapab_stages"]
            and _swapab_route_ok(layer, x, state[layer], cfg)):
        ws, cnt = _swapab_buffers(state, layer)
        return _swapab_forward(layer, x, state[layer], ws, cnt, state["swapab_stages"][layer])
    if layer != "router":
        if (layer in NN_GATED_LAYERS and cfg["n"] == state[layer + "_t"].shape[1]
                and ((state["nn_ok"] and NN_GATED_MIN_M <= cfg["m"] <= NN_GATED_MAX_M)
                     or (layer == "pooler" and state["nn_p256_ok"]
                         and NN_GATED_MAX_M < cfg["m"] <= NN_POOLER_MAX_M))):
            # NN layout: x [m, k] @ W_t [k, n] at the full weight width (cfg-only gate).
            return torch.mm(x, state[layer + "_t"])
        # The unpooler's slice is its first n rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)
    if _router_cluster_route(x, cfg, state):   # 1 <= m <= 8, n == 256: one launch, no memset
        return _router_cluster_forward(x, state)
    hi, lo = state["router_hi"], state["router_lo"]
    path = _router_path(x, hi, cfg["n"], state["router_stages"])
    if path == "swapab":   # m <= 32
        return _router_forward(x, hi, lo, state["router_stages"][_router_config(x.shape[0])])
    if path == "hilo":     # m > 32: master's forward
        return _router_hilo_forward(x, hi, lo)
    # Shape outside the kernel's preconditions (never for this problem's router): exact fp32 path.
    return F.linear(x.float(), state["router"][: cfg["n"]])
