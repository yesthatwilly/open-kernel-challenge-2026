# okc: problem=muse_bf16_proj
"""Router as one fused Triton hi/lo GEMM with a 3-D split-K grid at every m (hypothesis
router-triton-hilo-fused-control-splitk-all-m, the control for router-triton-hilo-fused).

The four bf16 layers go through the same cuBLAS dispatch SGLang uses on H200
(`_bf16_gemm_dispatch_impl`), untouched. The fp32 router runs, at every m, as ONE Triton kernel
over a bf16 hi + lo split of its weight with fp32 accumulation (allowed by the rules at any
batch size: "the router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation").
The split is built once in prepare() from the weights alone and stored as two [256, 8192] bf16
row-major tensors; nothing derived from the inputs is kept between calls.

Grid (the one difference from the parent): grid = (cdiv(M, BLOCK_M), N // BLOCK_N, SPLIT_K)
at every m. Each program owns one (M-block, N-block, K-slice), never loops over M, and
atomically adds its fp32 K-partial into an output zeroed right before the launch; there is no
plain-store path. Configs are static, chosen from m at launch; rows beyond m are masked, so any
m is valid.

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
"""

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
ROUTER_NUM_STAGES = 3
ROUTER_NUM_WARPS = 4
# m values pre-warmed in prepare(): one per config bucket plus the scored/tested ones, so that
# the timed call never compiles (M is not specialized and the grid's M extent is a launch
# argument, so one compile serves a whole bucket).
ROUTER_WARM_MS = (1, 8, 16, 32, 64, 128, 256)


def _router_config(m: int):
    """Static launch config from m only: (BLOCK_M, BLOCK_N, SPLIT_K).

    grid = (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), always atomic_add into a zeroed output:
      m <=  16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=16 -> (1, 8, 16) = 128 CTAs, K-slice 512
      m <=  32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=16 -> (1, 8, 16) = 128 CTAs, K-slice 512
      m <= 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=16 -> (2, 4, 16) = 128 CTAs at m=128, K-slice 512
      m >  128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=8  -> (4, 4,  8) = 128 CTAs at m=256, K-slice 1024
    """
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


def _router_forward(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """x bf16 [m, 8192] @ (hi + lo).T with fp32 accumulation -> fp32 [m, 256]. One launch plus
    the memset of the zeroed output. Branches only on shapes."""
    m, k = x.shape
    n = hi.shape[0]
    block_m, block_n, split_k = _router_config(m)
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
    block_m, block_n, split_k = _router_config(m)
    return (k == hi.shape[1] and k % (ROUTER_BLOCK_K * split_k) == 0 and n % block_n == 0
            and hi.stride(1) == 1)


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


def prepare(weights: Weights) -> dict:
    """Called once, untimed. Keep the bf16 weights as they are (contiguous), build the router's
    hi + lo bf16 split as [256, 8192] row-major (N x K), and pre-compile every m-bucket of the
    router kernel on synthetic bf16 x so the timed call never compiles."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16).contiguous()
    lo = (router - hi.float()).to(torch.bfloat16).contiguous()
    state["router_hi"] = hi
    state["router_lo"] = lo
    # Warm-up on synthetic inputs (weights only; nothing from a later call is involved).
    for m in ROUTER_WARM_MS:
        x = torch.randn((m, router.shape[1]), device=router.device, dtype=torch.float32).to(torch.bfloat16)
        if _router_shape_ok(x, hi):
            _router_forward(x, hi, lo)
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
    # (5) Swap-AB at 17 <= m <= 32 for all four bf16 layers (own workspaces / tickets).
    _swapab_prepare(state)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    # Shape gate (cfg layer / m / n and shape preconditions only; never on input values).
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
        # The unpooler's slice is its first n rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)
    hi, lo = state["router_hi"], state["router_lo"]
    if _router_shape_ok(x, hi) and cfg["n"] == hi.shape[0]:
        return _router_forward(x, hi, lo)
    # Shape outside the kernel's preconditions (never for this problem's router): exact fp32 path.
    return F.linear(x.float(), state["router"][: cfg["n"]])
