# okc: problem=muse_bf16_proj
"""Router as one fused Triton hi/lo GEMM with a 3-D split-K grid at every m (hypothesis
router-triton-hilo-fused-control-splitk-all-m, the control for router-triton-hilo-fused).

Restack (hypothesis restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128): on top of the
swap-AB master, three row wins each gated to exactly the cells they won: unpooler slice m <= 8
(CUDA cluster/DSMEM, ouns_cl, behind a graph-replay gate), router m <= 8 (CUDA cluster/DSMEM,
router_cluster_ok) and pooler/unpooler 65 <= m <= 128 (NN-layout torch.mm, nn_ok). Every flag
lives in the state dict; prepare() A/B/A/B-checks the whole dispatch on two states.

Combo (hypothesis combo-rowowner-reduce-free-m1-qkv-o-plus-naive), section (9): at m == 1 the
pooler (A), qkv_gate (A2, 2 CTAs/SM) and o (A3) can run the bulk-TMA row-owner CUDA kernel on a
per-state losslessly repacked weight, and qkv_gate m == 128 (B) a cuBLASLt split-K-1 algo fixed in
prepare(); each cell has its own state flag, canary, graph-replay gate (<= 0.98x master) and
profiler kernel-name assertion, else master's path.

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

Pooler K-halves at m == 128 (hypothesis pooler-khalf-m128-fp32-partials-triton-castadd, candidate b2 only), section
(11): one strided-batched cuBLAS GEMM over the two K halves of master's NN weight with fp32 output (fp32 partials,
never bf16), then one Triton kernel (_add2_cast_kernel) adds the two fp32 partials and rounds once to bf16, the
numerics of cuBLAS's own fp32 split-K reduce. Own flag (pooler_khalf128_ok), canary, ten read-flush gate rounds
(routed by the pooled-IQR rule on rounds 1-3; 4-10 diagnostic) and a profiler assertion of exactly two launches;
any miss keeps master's NN torch.mm.

Router zero fill under PDL (hypothesis router-zero-pdl-overlap-m9-m256), per m class 9..16 / 17..32 / 33..128 / > 128
(own flags router_pdl_ok_<16|32|128|256>): torch.zeros becomes torch.empty + _router_zero_pdl_kernel (trigger at
entry), and the router kernel (swap-AB or hilo, PDL constexpr) is launched with launch_pdl=True and one
griddepcontrol.wait right before its atomic_add, so its K-loop overlaps the fill. Feature probe + PTX order check +
warm-up after the router warm-up; per class a check, a race check with a slow-primary stand-in, two read-flush gate
rounds (third arm: zero kernel, PDL off) and a kernel-name assertion; any miss keeps master. m <= 8 is untouched.

bulk12 at m == 8 (hypothesis b12-batched-escape-m8-32-race), section (12): pooler and qkv_gate at exactly m == 8 can
run b12m_tma_kernel (uc1_g1: section (10)'s 12-bit codes, x as per-call tensor-map boxes, no cluster, local mbarrier
arrives, batched escape-x gather) in its own extension, each behind its own flag, canary + fp64 + per-row checks, three
read-flush gates against today's m == 8 route (pooled ratio <= 0.95, every round <= 0.97, pooled IQR) and a profiler
kernel-name assertion; the bridge GA0 / GA1 arms and pooler m == 16 are timed as non-routable diagnostics.
"""

import os
import re
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
    # router-zero-pdl-overlap-m9-m256: the other three R-PDL cells (m=128 is above), each mapped to its own flag.
    ("router", ROUTER_N, ROUTER_K, 16), ("router", ROUTER_N, ROUTER_K, 32), ("router", ROUTER_N, ROUTER_K, 256),
    ("unpooler", 1024, 4096, 1), ("unpooler", 1024, 4096, 8), ("unpooler", 8192, 4096, 128),
    ("pooler", 4096, 8192, 1), ("pooler", 4096, 8192, 8), ("pooler", 4096, 8192, 128),
    ("pooler", 4096, 8192, 256),
    # combo-rowowner-reduce-free-m1-qkv-o-plus-naive: the new cells (pooler m=1 is above).
    ("qkv_gate", 2304, 8192, 1), ("o", 8192, 1024, 1), ("qkv_gate", 2304, 8192, 128),
    # b12-batched-escape-m8-32-race: the qkv_gate m=8 b12m cell (pooler m=8 is above).
    ("qkv_gate", 2304, 8192, 8),
    # ladder-bf1-sk2-fragment-regstream-m8-32: the bf1 cells (pooler / qkv_gate m=8 are above).
    ("pooler", 4096, 8192, 16), ("pooler", 4096, 8192, 32), ("unpooler", 8192, 4096, 16),
    ("unpooler", 8192, 4096, 32), ("qkv_gate", 2304, 8192, 16), ("qkv_gate", 2304, 8192, 32),
    ("o", 8192, 1024, 8), ("o", 8192, 1024, 16), ("o", 8192, 1024, 32),
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
    PDL: tl.constexpr,
):
    """out[M, N] (+)= x[M, K] @ (hi + lo)[N, K]^T over this program's K-slice, fp32 accumulate.

    x: bf16 [M, K] row-major; hi, lo: bf16 [N, K] row-major; out: fp32 [M, N] row-major.
    Program (pid_m, pid_n, pid_k) covers rows of M [pid_m*BLOCK_M, +BLOCK_M), rows of N
    [pid_n*BLOCK_N, +BLOCK_N) and K [pid_k*K_SLICE, +K_SLICE), with no loop over M; its fp32
    K-partial is atomically added into the zeroed output. K_SLICE is a multiple of BLOCK_K.
    PDL (launched with launch_pdl=True after _router_zero_pdl_kernel): one griddepcontrol.wait after
    the K-loop, immediately before the atomic; PDL False compiles master's kernel.
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
    if PDL:   # the zero kernel's grid has completed and its stores are visible before any atomic
        tl.extra.cuda.gdc_wait()
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None], sem="relaxed")


@triton.jit(do_not_specialize=["M"])
def _router_swapab_kernel(
    x_ptr, hi_ptr, lo_ptr, out_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    K_SLICE: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
    PDL: tl.constexpr,
):
    """out[M, N] (+)= x[M, K] @ (hi + lo)[N, K]^T over this program's K-slice, computed as
    ((hi + lo) @ x^T)^T (swap-AB), fp32 accumulate.

    x: bf16 [M, K] row-major; hi, lo: bf16 [N, K] row-major; out: fp32 [M, N] row-major.
    Program (pid_t, pid_w, pid_k) covers tokens [pid_t*BLOCK_T, +BLOCK_T) (tokens >= M load as
    zero and are never added), weight rows [pid_w*BLOCK_W, +BLOCK_W) and K [pid_k*K_SLICE,
    +K_SLICE). hi / lo tiles [BLOCK_W, BLOCK_K] (as stored) are tl.dot's LHS, the x^T tile
    [BLOCK_K, BLOCK_T] its RHS, so 64 weight rows sit on the MMA M side. The fp32 K-partial is
    atomically added, transposed, into the zeroed output. K_SLICE is a multiple of BLOCK_K.
    PDL: as _router_hilo_kernel (one griddepcontrol.wait after the K-loop, before the atomic).
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
    if PDL:   # the zero kernel's grid has completed and its stores are visible before any atomic
        tl.extra.cuda.gdc_wait()
    tl.atomic_add(out_ptrs, acc, mask=t_mask[None, :], sem="relaxed")


# --- router zero fill under PDL (hypothesis router-zero-pdl-overlap-m9-m256) ----------------------
# At m > 8 the router is a zero fill of its fp32 output plus the split-K Triton kernel that atomically adds into it.
# With a cell's flag on, the fill is _router_zero_pdl_kernel (its first statement triggers the dependent launch) and
# the router kernel is launched with launch_pdl=True and PDL=True: its whole K-loop (weight stream + MMAs) runs while
# the fill runs, and one griddepcontrol.wait right before the atomic orders every zero before every atomic. zmode
# (from the cell's state flag, kernel_fn): 0 master (torch.zeros, PDL off), 1 candidate (zero kernel + PDL),
# ROUTER_ZMODE_NOPDL = 2 the gate-only third arm (zero kernel, router PDL off, no launch_pdl), set only on prepare()'s
# gate copies of state. m <= 8 (router_kernel<8,8>, no memset) never maps to a flag.
ROUTER_PDL_MS = (16, 32, 128, 256)          # one flag per m class: 9..16, 17..32, 33..128, > 128
ROUTER_PDL_FLAGS = tuple(f"router_pdl_ok_{b}" for b in ROUTER_PDL_MS)
ROUTER_PDL_CHECK_MS = {16: (9, 16), 32: (32,), 128: (64, 128), 256: (256,)}   # correctness per flag (9: the test)
ROUTER_PDL_WARM_MS = (9, 16, 32, 64, 128, 256)   # every m > 8 in ROUTER_WARM_MS, plus the m = 9 test
ROUTER_ZMODE_NOPDL = 2
ROUTER_ZERO_BLOCK = 4096
ROUTER_ZERO_WARPS = 4
ROUTER_PDL_RACE_REPS = 32
ROUTER_PDL_SPIN_CYCLES = 20000   # stand-in primary: ~10 us of clock64 spin before its zero stores
ROUTER_PDL_ROUNDS = 2            # read-flush _cell_gate rounds; BOTH must clear the bar (the khalf rule)
ROUTER_PDL_MIN_RATIO = 1.03      # master / candidate, net, each round
ROUTER_PDL_MAX_IQR = 1.5         # cand IQR <= this x max(master IQR, ROUTER_PDL_IQR_FLOOR_US), each round
ROUTER_PDL_IQR_FLOOR_US = 0.5


@triton.jit
def _router_zero_pdl_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    """out[:numel] = 0 (fp32). First statement: trigger the dependent (router) grid's launch."""
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < numel)


@triton.jit
def _router_zero_slow_kernel(out_ptr, numel, BLOCK: tl.constexpr, SPIN: tl.constexpr):
    """prepare()-only race-check stand-in for _router_zero_pdl_kernel: the same trigger at entry, then ~SPIN clock64
    cycles of spinning before the same zero stores (a slow primary). Never launched by kernel_fn."""
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    t = t0
    while t - t0 < SPIN:
        t = tl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < numel)


def _router_zero_launch(out: torch.Tensor):
    numel = out.numel()
    return _router_zero_pdl_kernel[(triton.cdiv(numel, ROUTER_ZERO_BLOCK),)](
        out, numel, BLOCK=ROUTER_ZERO_BLOCK, num_warps=ROUTER_ZERO_WARPS)


def _router_out(m: int, n: int, dev, zmode: int) -> torch.Tensor:
    """The router's fp32 output, zeroed right before the router launch, every call: master's torch.zeros
    (zmode 0) or torch.empty + _router_zero_pdl_kernel (zmode 1 / 2)."""
    if not zmode:
        return torch.zeros((m, n), dtype=torch.float32, device=dev)
    out = torch.empty((m, n), dtype=torch.float32, device=dev)
    _router_zero_launch(out)
    return out


def _router_pdl_flag(m: int) -> str:
    """The R-PDL state flag serving m (shape only): "" at m <= 8 (the cluster path, untouched), else the first class
    in ROUTER_PDL_MS holding m (m > 256 maps to the 256 class: the same hilo config)."""
    if m <= ROUTER_CLUSTER_MAX_M:
        return ""
    for b in ROUTER_PDL_MS:
        if m <= b:
            return f"router_pdl_ok_{b}"
    return f"router_pdl_ok_{ROUTER_PDL_MS[-1]}"


def _router_zmode(m: int, state: dict) -> int:
    """0 / 1 / ROUTER_ZMODE_NOPDL from the cell's prepare()-set state flag (False / True / gate-only 2)."""
    flag = _router_pdl_flag(m)
    return int(state.get(flag, False)) if flag else 0


def _router_swapab_launch(x, hi, lo, out, num_stages: int, pdl: bool = False, launch_pdl: bool = False):
    """One _router_swapab_kernel launch into out (already zeroed / being zeroed by the PDL primary). The routed
    paths pass pdl == launch_pdl; only prepare()'s no-wait race control passes pdl False with launch_pdl True."""
    m, k = x.shape
    n = hi.shape[0]
    block_t, split_k = _router_config(m)
    grid = (triton.cdiv(m, block_t), n // ROUTER_BLOCK_W, split_k)
    extra = {"launch_pdl": True} if launch_pdl else {}
    return _router_swapab_kernel[grid](
        x, hi, lo, out,
        m,
        x.stride(0), hi.stride(0), out.stride(0),
        K_SLICE=k // split_k,
        BLOCK_W=ROUTER_BLOCK_W, BLOCK_T=block_t, BLOCK_K=ROUTER_BLOCK_K, PDL=pdl,
        num_warps=ROUTER_NUM_WARPS, num_stages=num_stages, **extra,
    )


def _router_hilo_launch(x, hi, lo, out, pdl: bool = False, launch_pdl: bool = False):
    """One _router_hilo_kernel launch into out (master's config); pdl / launch_pdl as _router_swapab_launch."""
    m, k = x.shape
    n = hi.shape[0]
    block_m, block_n, split_k = _router_hilo_config(m)
    grid = (triton.cdiv(m, block_m), n // block_n, split_k)
    extra = {"launch_pdl": True} if launch_pdl else {}
    return _router_hilo_kernel[grid](
        x, hi, lo, out,
        m,
        x.stride(0), hi.stride(0), out.stride(0),
        K_SLICE=k // split_k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=ROUTER_BLOCK_K, PDL=pdl,
        num_warps=ROUTER_NUM_WARPS, num_stages=ROUTER_NUM_STAGES, **extra,
    )


def _router_forward(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor, num_stages: int,
                    report: bool = False, zmode: int = 0) -> torch.Tensor:
    """x bf16 [m, 8192] @ (hi + lo).T with fp32 accumulation -> fp32 [m, 256]. One launch plus
    the zero fill of the output (zmode: see _router_out; 1 launches the router with PDL). Branches
    only on shapes and zmode (a state flag). report=True (prepare() only) prints the compiled
    variant's registers, spills and PTX counts."""
    m = x.shape[0]
    out = _router_out(m, hi.shape[0], x.device, zmode)
    pdl = zmode == 1
    compiled = _router_swapab_launch(x, hi, lo, out, num_stages, pdl=pdl, launch_pdl=pdl)
    if report:
        block_t, split_k = _router_config(m)
        grid = (triton.cdiv(m, block_t), hi.shape[0] // ROUTER_BLOCK_W, split_k)
        _print_ptx_counts(f"[router] swapab m={m} BLOCK_T={block_t} SPLIT_K={split_k} "
                          f"grid={grid} stages={num_stages}", compiled)
    return out


def _router_hilo_forward(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor, zmode: int = 0) -> torch.Tensor:
    """x bf16 [m, 8192] @ (hi + lo).T with fp32 accumulation -> fp32 [m, 256]. One launch plus
    the zero fill of the output (zmode as _router_forward). Branches only on shapes and zmode."""
    out = _router_out(x.shape[0], hi.shape[0], x.device, zmode)
    pdl = zmode == 1
    _router_hilo_launch(x, hi, lo, out, pdl=pdl, launch_pdl=pdl)
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


# --- R-PDL prepare (hypothesis router-zero-pdl-overlap-m9-m256) -------------------------------------
# _router_pdl_probe (both states, right after _router_prepare, before any capture): feature detection, one synced
# launch of every new variant with a PTX order check, then the warm-up of every m class the judge can hit with
# kernel_fn's exact launch kwargs. _router_pdl_prepare (state A, after the router cluster): per flag, correctness,
# race checks (kernel_fn graph + a slow-primary stand-in), two read-flush gate rounds with a third arm, a profiler
# kernel-name assertion; one PATH CHOSEN R-PDL decision line per flag.
_PTX_RED_ATOM = re.compile(r"^\s*(?:@!?%\w+\s+)?(?:red|atom)\.", re.M)


def _router_pdl_capability() -> str:
    """"" if this Triton exposes tl.extra.cuda.gdc_wait / gdc_launch_dependents, else why not. Prints the version
    and whether CUDAOptions declares launch_pdl (info; the probe launch itself is the test that it is accepted)."""
    try:
        import triton.language.extra.cuda as tlc
    except Exception as e:
        return f"triton {triton.__version__}: triton.language.extra.cuda unavailable ({type(e).__name__})"
    missing = [nm for nm in ("gdc_wait", "gdc_launch_dependents") if not hasattr(tlc, nm)]
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        opt = "launch_pdl" in getattr(CUDAOptions, "__dataclass_fields__", {})
    except Exception as e:
        opt = f"unknown ({type(e).__name__})"
    print(f"[r-pdl] triton {triton.__version__}: tl.extra.cuda gdc_wait/gdc_launch_dependents "
          f"{'present' if not missing else 'MISSING ' + str(missing)}; CUDAOptions.launch_pdl: {opt}", flush=True)
    return f"triton {triton.__version__}: tl.extra.cuda lacks {missing}" if missing else ""


def _router_pdl_ptx_why(tag: str, compiled, kind: str) -> str:
    """PTX order check of one compiled variant; "" if it holds, else why. kind "zero": exactly one
    griddepcontrol.launch_dependents, before the first st.global, and no griddepcontrol.wait. kind "router": exactly
    one griddepcontrol.wait, after the last wgmma / mma and before the first red / atom (an extra wait, e.g. one
    auto-inserted at entry for launch_pdl, fails). kind "master": no griddepcontrol at all."""
    ptx = compiled.asm["ptx"]
    nw = ptx.count("griddepcontrol.wait")
    nl = ptx.count("griddepcontrol.launch_dependents")
    i_w = ptx.find("griddepcontrol.wait")
    i_mma = max(ptx.rfind("wgmma.mma_async"), ptx.rfind("mma.sync"))
    ma = _PTX_RED_ATOM.search(ptx)
    i_atom = ma.start() if ma else -1
    i_l = ptx.find("griddepcontrol.launch_dependents")
    i_st = ptx.find("st.global")
    print(f"[r-pdl] ptx {tag}: griddepcontrol.wait x{nw} @{i_w}, launch_dependents x{nl} @{i_l}, last mma @{i_mma}, "
          f"first red/atom @{i_atom}, first st.global @{i_st}, wgmma {ptx.count('wgmma.mma_async')}", flush=True)
    if kind == "zero":
        if nl != 1 or nw != 0:
            return f"{tag}: zero kernel has launch_dependents x{nl} (want 1), wait x{nw} (want 0)"
        if i_st >= 0 and i_l > i_st:
            return f"{tag}: launch_dependents is not at the zero kernel's entry (after a st.global)"
        return ""
    if kind == "master":
        return "" if nw == 0 and nl == 0 else f"{tag}: PDL=False variant carries griddepcontrol (x{nw} wait, x{nl})"
    if nw != 1:
        return f"{tag}: griddepcontrol.wait x{nw} (want exactly 1; an auto-inserted entry wait kills the overlap)"
    if i_mma < 0 or i_atom < 0:
        return f"{tag}: no mma ({i_mma}) or no red/atom ({i_atom}) found to order the wait against"
    if not i_mma < i_w < i_atom:
        return f"{tag}: wait @{i_w} not between the last mma @{i_mma} and the first red/atom @{i_atom}"
    return ""


def _router_pdl_launch(path: str, x, state: dict, out, pdl: bool, launch_pdl: bool):
    """One router launch on the path kernel_fn takes for x (prepare()-only callers)."""
    hi, lo = state["router_hi"], state["router_lo"]
    if path == "swapab":
        return _router_swapab_launch(x, hi, lo, out, state["router_stages"][_router_config(x.shape[0])],
                                     pdl=pdl, launch_pdl=launch_pdl)
    return _router_hilo_launch(x, hi, lo, out, pdl=pdl, launch_pdl=launch_pdl)


def _router_pdl_warm(state: dict, zmodes) -> None:
    """Run every m in ROUTER_PDL_WARM_MS through the forward kernel_fn calls (exact kwargs) in each zmode, synced, so
    no timed call or capture ever compiles."""
    hi, lo = state["router_hi"], state["router_lo"]
    k = hi.shape[1]
    for m in ROUTER_PDL_WARM_MS:
        x = torch.randn((m, k), device=hi.device, dtype=torch.float32).to(torch.bfloat16)
        path = _router_path(x, hi, ROUTER_N, state["router_stages"])
        for z in zmodes:
            if path == "swapab":
                _router_forward(x, hi, lo, state["router_stages"][_router_config(m)], zmode=z)
            elif path == "hilo":
                _router_hilo_forward(x, hi, lo, zmode=z)
    torch.cuda.synchronize()


def _router_pdl_probe(state: dict) -> None:
    """State A, right after _router_prepare: sets state["router_pdl_cap"] ("" = capable). Feature detection; per
    class m in ROUTER_PDL_MS one announced, synced launch of the zero kernel + the PDL router (launch_pdl=True) and of
    master's variant, each PTX-checked (_router_pdl_ptx_why); then the warm-up of zmodes 0 / 1 / 2. Never raises."""
    t0 = time.perf_counter()
    cap = ""
    try:
        cap = _router_pdl_capability()
        hi = state["router_hi"]
        if not cap and tuple(hi.shape) != (ROUTER_N, ROUTER_K):
            cap = f"router weight {tuple(hi.shape)} unexpected"
        zero_checked = False
        for m in ROUTER_PDL_MS:
            if cap:
                break
            x = torch.randn((m, hi.shape[1]), device=hi.device, dtype=torch.float32).to(torch.bfloat16)
            path = _router_path(x, hi, ROUTER_N, state["router_stages"])
            if path not in ("swapab", "hilo"):
                cap = f"m={m}: no Triton router path"
                break
            out = torch.empty((m, ROUTER_N), dtype=torch.float32, device=hi.device)
            print(f"[r-pdl] LAUNCH probe m={m}: _router_zero_pdl_kernel + {path} PDL (launch_pdl=True) ...", flush=True)
            cz = _router_zero_launch(out)
            cr = _router_pdl_launch(path, x, state, out, pdl=True, launch_pdl=True)
            cm = _router_pdl_launch(path, x, state, torch.zeros_like(out), pdl=False, launch_pdl=False)
            torch.cuda.synchronize()
            print(f"[r-pdl] DONE   probe m={m} (synced)", flush=True)
            if not zero_checked:
                cap = _router_pdl_ptx_why("_router_zero_pdl_kernel", cz, "zero")
                zero_checked = True
            cap = cap or _router_pdl_ptx_why(f"{path} m={m} PDL", cr, "router")
            cap = cap or _router_pdl_ptx_why(f"{path} m={m} master", cm, "master")
        if not cap:
            _router_pdl_warm(state, (0, 1, ROUTER_ZMODE_NOPDL))
    except Exception as e:   # any surprise keeps master's router for every m
        cap = f"probe raised {type(e).__name__}: {str(e)[-200:]}"
    torch.cuda.synchronize()
    state["router_pdl_cap"] = cap
    print(f"[r-pdl] probe + warm-up {time.perf_counter() - t0:.2f}s: " + ("capable" if not cap else f"OFF: {cap}"),
          flush=True)


def _router_pdl_capture(fn):
    """Graph-capture fn() after 3 eager warm calls; returns (graph, the captured output)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    torch.cuda.synchronize()
    return gr, out


def _router_pdl_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    if not bool(torch.isfinite(a).all()):
        return float("inf")
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _router_pdl_race(state: dict, m: int, flag: str):
    """Race checks at class m; returns (why, info). Graphs on one x buffer, refilled before every round of
    ROUTER_PDL_RACE_REPS: master (flag off) and the candidate (flag on), both kernel_fn; the slow-primary stand-in
    (_router_zero_slow_kernel, ~10 us spin, then the PDL router launched the same way, with its wait); and the
    no-wait control (the stand-in, then master's router body launched with launch_pdl=True: prepare()-only, printed,
    never routed). Candidate and stand-in must stay within AB_ROUTER_REL_L2 of master on every replay."""
    hi = state["router_hi"]
    dev = hi.device
    cfg = {"m": m, "n": ROUTER_N, "k": ROUTER_K, "layer": "router"}
    st_on, st_off = dict(state), dict(state)
    st_on[flag], st_off[flag] = True, False
    g = torch.Generator(device=dev)
    g.manual_seed(13579 + m)
    x = torch.randn((m, ROUTER_K), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
    data = {"x": x, "config": cfg}
    path = _router_path(x, hi, ROUTER_N, state["router_stages"])

    def slow(wait: bool):
        out = torch.empty((m, ROUTER_N), dtype=torch.float32, device=dev)
        numel = out.numel()
        _router_zero_slow_kernel[(triton.cdiv(numel, ROUTER_ZERO_BLOCK),)](
            out, numel, BLOCK=ROUTER_ZERO_BLOCK, SPIN=ROUTER_PDL_SPIN_CYCLES, num_warps=ROUTER_ZERO_WARPS)
        _router_pdl_launch(path, x, state, out, pdl=wait, launch_pdl=True)
        return out

    graphs = {}
    try:
        graphs["master"] = _router_pdl_capture(lambda: kernel_fn(data, st_off))
        graphs["cand"] = _router_pdl_capture(lambda: kernel_fn(data, st_on))
        for name, wait in (("slow", True), ("nowait", False)):
            print(f"[r-pdl] LAUNCH race stand-in m={m} {name} (spin {ROUTER_PDL_SPIN_CYCLES} cycles) ...", flush=True)
            slow(wait)
            torch.cuda.synchronize()
            print(f"[r-pdl] DONE   race stand-in m={m} {name} (synced)", flush=True)
            graphs[name] = _router_pdl_capture(lambda w=wait: slow(w))
        worst = {"cand": 0.0, "slow": 0.0, "nowait": 0.0}
        nbad = {"cand": 0, "slow": 0, "nowait": 0}
        for _ in range(ROUTER_PDL_RACE_REPS):
            x.copy_(torch.randn((m, ROUTER_K), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16))
            for name in ("master", "cand", "slow", "nowait"):
                graphs[name][0].replay()
            torch.cuda.synchronize()
            ref = graphs["master"][1]
            for name in ("cand", "slow", "nowait"):
                e = _router_pdl_rel(graphs[name][1], ref)
                worst[name] = max(worst[name], e)
                nbad[name] += int(not e <= AB_ROUTER_REL_L2)
    finally:
        graphs.clear()
        torch.cuda.synchronize()
    info = (f"race {ROUTER_PDL_RACE_REPS}x vs master: cand worst {worst['cand']:.1e}, slow-primary worst "
            f"{worst['slow']:.1e}, no-wait control {nbad['nowait']}/{ROUTER_PDL_RACE_REPS} wrong "
            f"(worst {worst['nowait']:.1e}" + ("" if nbad["nowait"] else "; race NOT observable: no early launch?")
            + ")")
    print(f"[r-pdl] m={m}: {info}", flush=True)
    if nbad["cand"] or nbad["slow"]:
        return (f"race: cand {nbad['cand']}/{ROUTER_PDL_RACE_REPS}, slow-primary {nbad['slow']}/"
                f"{ROUTER_PDL_RACE_REPS} replays beyond {AB_ROUTER_REL_L2:g}"), info
    return "", info


def _router_pdl_names_why(names: list, path: str) -> str:
    """"" if one candidate call launches exactly [_router_zero_pdl_kernel, the router kernel] (no fill kernel)."""
    want = "_router_swapab_kernel" if path == "swapab" else "_router_hilo_kernel"
    if any("vectorized_elementwise" in nm or "fill" in nm.lower() for nm in names):
        return f"a fill kernel is still launched ({names})"
    if (len(names) != 2 or sum("_router_zero_pdl_kernel" in nm for nm in names) != 1
            or sum(want in nm for nm in names) != 1):
        return f"expected exactly [_router_zero_pdl_kernel, {want}], profiler saw {names}"
    return ""


def _router_pdl_prepare(state: dict) -> None:
    """State A only, after the router cluster (kernel_fn's router dispatch is complete). Per class m in
    ROUTER_PDL_MS sets state[router_pdl_ok_<m>] (False unless every step passes): _cell_check through kernel_fn at
    ROUTER_PDL_CHECK_MS (rel L2 vs fp64 <= AB_ROUTER_REL_L2, power-of-two probe row, repeat within the same
    tolerance: relaxed fp32 atomics are order-nondeterministic); _router_pdl_race; ROUTER_PDL_ROUNDS read-flush
    _cell_gate rounds (master/cand >= ROUTER_PDL_MIN_RATIO and the IQR rule in every round) with the third arm
    (zero kernel, PDL off) printed beside; then the profiler kernel-name assertion. Never raises."""
    cap = state.get("router_pdl_cap", "not probed")
    for b in ROUTER_PDL_MS:
        state[f"router_pdl_ok_{b}"] = False
    if cap:
        _decision(f"PATH CHOSEN R-PDL router m in {ROUTER_PDL_MS}: master (torch.zeros + router) for all; {cap}")
        return
    for b in ROUTER_PDL_MS:
        flag = f"router_pdl_ok_{b}"
        t0 = time.perf_counter()
        why, race_info, rounds = "", "race not run", []
        try:
            st_on = dict(state)
            st_on[flag] = True
            for m in ROUTER_PDL_CHECK_MS[b]:
                cfg = {"m": m, "n": ROUTER_N, "k": ROUTER_K, "layer": "router"}
                bad = _cell_check(f"r-pdl m={m}", lambda xx, c=cfg: kernel_fn({"x": xx, "config": c}, st_on),
                                  state["router"], m, ROUTER_N, ROUTER_K, rel_bound=AB_ROUTER_REL_L2,
                                  repeat_rel=AB_ROUTER_REL_L2)
                if bad:
                    why = f"check m={m}: {bad}"
                    break
            if not why:
                why, race_info = _router_pdl_race(state, b, flag)
            if not why:
                bar = (f"route if master/cand >= {ROUTER_PDL_MIN_RATIO} in every round and IQR <= "
                       f"{ROUTER_PDL_MAX_IQR}x master")
                for rnd in range(ROUTER_PDL_ROUNDS):
                    st = {}
                    cand_us, master_us = _cell_gate(f"r-pdl round {rnd + 1}/{ROUTER_PDL_ROUNDS}", "router", b,
                                                    ROUTER_N, ROUTER_K, flag, state, flush_mode=B12_GATE_FLUSH,
                                                    stats=st, route_txt=bar, extra={"nopdl": ROUTER_ZMODE_NOPDL})
                    st["ratio"] = master_us / cand_us
                    rounds.append(st)
                if not all(r["ratio"] >= ROUTER_PDL_MIN_RATIO for r in rounds):
                    why = f"gate: master/cand {[round(r['ratio'], 3) for r in rounds]} not all >= {ROUTER_PDL_MIN_RATIO}"
                elif not all(r["cand"]["iqr"] <= ROUTER_PDL_MAX_IQR * max(r["master"]["iqr"], ROUTER_PDL_IQR_FLOOR_US)
                             for r in rounds):
                    why = f"gate: cand IQR > {ROUTER_PDL_MAX_IQR}x master IQR in some round"
            if not why:
                xs = torch.randn((b, ROUTER_K), device=state["router"].device, dtype=torch.float32).to(torch.bfloat16)
                data = {"x": xs, "config": {"m": b, "n": ROUTER_N, "k": ROUTER_K, "layer": "router"}}
                path = _router_path(xs, state["router_hi"], ROUTER_N, state["router_stages"])
                try:
                    names = _cell_kernels(lambda: kernel_fn(data, st_on))
                    print(f"[r-pdl] m={b}: profiled kernels {names}", flush=True)
                    why = _router_pdl_names_why(names, path)
                except Exception as e:   # part of the routing contract: no profile, no route
                    why = f"profiler unavailable ({type(e).__name__}: {str(e)[-160:]})"
            if not why:
                state[flag] = True
        except Exception as e:   # any surprise keeps master's router for this class
            why = f"raised {type(e).__name__}: {str(e)[-200:]}"
            state[flag] = False
        torch.cuda.synchronize()
        r_txt = "; ".join(
            f"r{i + 1} m/c {r['ratio']:.3f} net m/c/nopdl {r['master']['net']:.2f}/{r['cand']['net']:.2f}/"
            f"{r['nopdl']['net']:.2f} IQR {r['master']['iqr']:.2f}/{r['cand']['iqr']:.2f}/{r['nopdl']['iqr']:.2f}"
            for i, r in enumerate(rounds)) or "gate not run"
        print(f"[prepare-time] r-pdl m={b}: {time.perf_counter() - t0:.2f}s", flush=True)
        _decision(f"PATH CHOSEN R-PDL router m={b} ({flag}): "
                  + ("zero kernel + PDL router" if state[flag] else "master (torch.zeros + router)")
                  + f" [{B12_GATE_FLUSH}-flush, us; {r_txt}] [{race_info}]" + ("" if state[flag] else f"; {why}"))


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
        _decision("[ouns] PATH CHOSEN: unpooler slice m<=8 -> restack path (swapab16): no CUDA module")
        return
    w = state["unpooler"]
    if not (w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[0] >= OUNS_N
            and w.shape[1] == OUNS_K and w.stride(1) == 1 and w.stride(0) % 8 == 0
            and w.data_ptr() % 16 == 0):
        _decision(f"[ouns] PATH CHOSEN: unpooler slice m<=8 -> restack path: weight {tuple(w.shape)} unexpected")
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
        _decision(f"[ouns] PATH CHOSEN: unpooler slice m<=8 (n=1024) -> CUDA cluster/DSMEM split-K, cluster dim "
              f"{state['ouns_cl']}; self-check passed (m {OUNS_CHECK_MS}) and graph-replay gate won at m "
              f"{OUNS_GATE_MS} (see the [ouns] gate lines for both medians)")
    else:
        _decision("[ouns] PATH CHOSEN: unpooler slice m<=8 (n=1024) -> restack path (swapab16); cluster route off "
              "(see the [ouns] lines above for why)")


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
    _decision("[router_cluster] PATH CHOSEN router m<=8 (n=256): "
          + ("CUDA cluster kernel router_kernel<8,8>, no memset" if ok else "swap-AB Triton router (cluster off)")
          + "; 9 <= m <= 32: swap-AB Triton router; m > 32: hilo")


# --- (9) m == 1 bulk-TMA row-owner stream (pooler A, qkv_gate A2, o A3) and qkv_gate m == 128 on a
# prepare-time cuBLASLt plan (B) (hypothesis combo-rowowner-reduce-free-m1-qkv-o-plus-naive).
#
# A / A2 / A3: the bulk_kernel of hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild
# (impl b3e80b45, job 571ea532), MODE 0 only, with the ring depth S and the min-CTAs/SM of
# __launch_bounds__ as template parameters and the grid (N / R) a launch argument. Warp 0 is the
# producer (cp.async.bulk + mbarrier full/empty ring, every wait bounded to 0.5 s, then an error
# word), warps 1-4 consume (mma.sync m16n8k16, a quarter of each chunk's K each). Each CTA owns R
# output rows over the WHOLE K: no split-K, no workspace, no ticket. prepare() repacks each weight
# losslessly (pure permutation, round trip asserted bit-exact) into Wt[cta][kblock][row][64 k] with
# the 16 B segments of each 128 B row XOR-swizzled by (row & 7), per state.
#   A  pooler   m=1: bulk_kernel<32, 8192, 8, 6, 1>, grid 128, dyn smem 123,904 B, 1 CTA/SM
#   A2 qkv_gate m=1: bulk_kernel<16, 8192, 8, 4, 2>, grid 144, dyn smem  99,072 B, 2 CTAs/SM
#                    (max-shared carveout set; the cell is disabled unless the occupancy query is 2)
#   A3 o        m=1: bulk_kernel<64, 1024, 8, 6, 1>, grid 128, dyn smem 111,616 B, 1 CTA/SM
# B: qkv_gate m == 128 runs cublasLtMatmul with an algo chosen ONCE in prepare() among
# cublasLtMatmulAlgoGetHeuristic's SPLIT-K-1 candidates (no workspace reduce node; review fix (1):
# no per-call preferred_blas_library toggle). Descriptors are built per call on the host (no
# static / global state); under graph replay none of that host code runs.
# Every cell: canary launch (flushed LAUNCH / DONE prints, synced, error word) and a correctness
# check before any capture, then a model-selector gate (graph replay of kernel_fn with the cell's
# flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline
# subtracted) that routes only at candidate / master <= 0.98, then one torch.profiler call that
# asserts the launched kernel name. One PATH CHOSEN line per cell. Flags and buffers live in state.
BULK_EXT_NAME = "muse_bf16_bulk_rowown_m1_v1"
BULK_NVCC_FLAGS = ["-O3", "-lineinfo", "-std=c++20"]
BULK_M = 1
# layer -> (source, instance id, N, K, rows per CTA R, ring stages S, min CTAs/SM, state flag)
BULK_CELLS = {
    "pooler":   ("A", 0, 4096, 8192, 32, 6, 1, "bulk_pooler_ok"),
    "qkv_gate": ("A2", 1, 2304, 8192, 16, 4, 2, "bulk_qkv_ok"),
    "o":        ("A3", 2, 8192, 1024, 64, 6, 1, "bulk_o_ok"),
}
BULK_REF_TBS = 3.95          # best effective rate the row-owner kernel reached in the judge (pooler m=1)
CELL_MAX_RATIO = 0.98        # route only if candidate / master <= this (graph replay, cold L2)
CELL_GATE_REPS = 41
CELL_FLUSH_BYTES = 256 * 1024 * 1024
CELL_FLUSH_MODES = ("write", "read")   # _cell_gate: zero_() the flush buffer, or read all of it (the judge reads)
# prepare()-only log: every routing decision line (PATH CHOSEN, bulk12 gate summaries, A/B misses), printed when made
# and re-printed together at the end of prepare(), because the judge keeps only a 16 KB log tail. Cleared at the start
# and the end of prepare(); kernel_fn never reads it.
_DECISION_LINES = []


def _decision(line: str) -> None:
    print(line, flush=True)
    _DECISION_LINES.append(line)

CELL_REL_L2 = 3e-3           # vs fp64 on random x
CELL_PROBE_REL = 1e-4        # the single power-of-two row (a weight column, exact)
LT_M = 128
LT_FLAG = "lt_qkv128_ok"
LT_EXT_NAME = "muse_bf16_lt_qkv128_percall_v1"   # not the sibling combo's name: different source
LT_QUERY_WS = (0, 1 << 20)   # heuristic workspace caps queried; only split-K-1 candidates are kept
LT_WS_BYTES = 1 << 20        # per-state workspace handed to cublasLtMatmul
LT_MAX_ALGOS = 16
LT_PICK_REPS = 15
LT_ALIGN = 256
NEW_CELL_FLAGS = ("bulk_pooler_ok", "bulk_qkv_ok", "bulk_o_ok", LT_FLAG)

_BULK_CPP = r"""
#include <torch/extension.h>
#include <vector>
void bulk_run(torch::Tensor wt, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t inst, int64_t m);
std::vector<int64_t> bulk_setup(int64_t inst);
"""

_BULK_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

#define BULK_WAIT_NS 500000000ULL   // every mbarrier wait is bounded to 0.5 s, then an error word

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Bounded wait: false after BULK_WAIT_NS (the caller sets the error word and leaves the loop).
__device__ __forceinline__ bool mbar_wait(uint32_t a, uint32_t parity) {
  unsigned long long t0 = 0;
  for (uint32_t it = 0;; ++it) {
    uint32_t done;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) return true;
    if (it == 0) t0 = gtimer();
    else if (gtimer() - t0 > BULK_WAIT_NS) return false;
  }
}
// Plain 1-D bulk copy global -> shared, completion counted on the mbarrier; no cache-hint form.
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// R: weight rows per CTA; K: reduction length; NT: padded tokens (8); S: ring stages; MINB: the
// __launch_bounds__ min-CTAs/SM. Grid = N / R (N = gridDim.x * R in the epilogue).
// err codes: 1 producer wait timeout, 2 consumer wait timeout, 3 built without sm_90.
template <int R, int K, int NT, int S, int MINB>
__global__ void __launch_bounds__(160, MINB)
bulk_kernel(const __nv_bfloat16* __restrict__ wt, const __nv_bfloat16* __restrict__ x,
            __nv_bfloat16* __restrict__ out, int* err, int m) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int CKK = 8192 / R;              // K per chunk: the weight chunk is exactly 16 KB
  constexpr int NCHUNK = K / CKK;
  constexpr int XS = CKK * 2 + 16;           // padded x row stride in smem (bank-conflict free)
  constexpr int SLOT = 16384 + NT * XS;
  constexpr int MT = R / 16;
  constexpr int NN = NT / 8;
  constexpr int KSW = CKK / 64;              // k16 steps per consumer warp per chunk
  static_assert(K % CKK == 0 && (SLOT % 128) == 0 && R * CKK * 2 == 16384, "geometry");
  static_assert(R % 16 == 0 && CKK % 64 == 0, "tiles");
  static_assert(NN == 1 || NN % 2 == 0, "NT");
  static_assert(S >= 2 && S <= 8, "full / empty barriers live in 64 B each");
  static_assert(4 * R * NT * 4 <= S * SLOT, "epilogue scratch must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  const uint32_t full0 = sbase, empty0 = sbase + 64, slot0 = sbase + 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float acc[MT][NN][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NN; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  if (warp == 0) {
    // ---- producer: whole warp, lane 0 waits and arms the barrier, lanes issue the copies ----
    const uint32_t tx = 16384u + (uint32_t)m * (CKK * 2);
    for (int j = 0; j < NCHUNK; ++j) {
      const int s = j % S;
      const uint32_t eph = ((j / S) & 1) ^ 1;
      int good = 1;
      if (lane == 0) {
        good = mbar_wait(empty0 + 8 * s, eph) ? 1 : 0;
        if (good) mbar_expect_tx(full0 + 8 * s, tx);
      }
      good = __shfl_sync(0xffffffffu, good, 0);
      if (!good) { if (lane == 0) atomicExch(err, 1); break; }
      const uint32_t wslot = slot0 + (uint32_t)s * SLOT;
      const uint32_t fb = full0 + 8 * s;
      if (lane == 0)
        bulk_g2s(wslot, wt + ((size_t)blockIdx.x * NCHUNK + j) * (size_t)(R * CKK), 16384, fb);
      if (lane < m)
        bulk_g2s(wslot + 16384 + (uint32_t)lane * XS, x + (size_t)lane * K + (size_t)j * CKK,
                 CKK * 2, fb);
    }
    __syncwarp();
  } else {
    // ---- consumers: warp cw owns k16 steps [cw*KSW, (cw+1)*KSW) of every chunk ----
    const int cw = warp - 1;
    for (int j = 0; j < NCHUNK; ++j) {
      const int s = j % S;
      const uint32_t ph = (j / S) & 1;
      if (!mbar_wait(full0 + 8 * s, ph)) { if (lane == 0) atomicExch(err, 2); break; }
      const uint32_t wslot = slot0 + (uint32_t)s * SLOT;
      const uint32_t xslot = wslot + 16384;
#pragma unroll
      for (int i = 0; i < KSW; ++i) {
        const int kk16 = cw * KSW + i;
        const int kbi = kk16 >> 2;
        const int seg = (kk16 & 3) * 2;
        uint32_t bfr[NN][2];
        if constexpr (NN == 1) {
          const int khalf = (lane >> 3) & 1;
          ldsm_x2(bfr[0][0], bfr[0][1],
                  xslot + (uint32_t)((lane & 7) * XS + kk16 * 32 + khalf * 16));
        } else {
#pragma unroll
          for (int p = 0; p < NN / 2; ++p) {
            const int mi = lane >> 3;
            const int tok = (2 * p + (mi >> 1)) * 8 + (lane & 7);
            ldsm_x4(bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1],
                    xslot + (uint32_t)(tok * XS + kk16 * 32 + (mi & 1) * 16));
          }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          uint32_t a0, a1, a2, a3;
          const int row = mt * 16 + (lane & 15);
          ldsm_x4(a0, a1, a2, a3,
                  wslot + (uint32_t)((kbi * R + row) * 128 + (((seg + (lane >> 4)) ^ (lane & 7)) * 16)));
#pragma unroll
          for (int nn = 0; nn < NN; ++nn) mma_bf16(acc[mt][nn], a0, a1, a2, a3, bfr[nn][0], bfr[nn][1]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
  }
  __syncthreads();   // every issued copy has been waited on by the consumers: the ring is quiescent
  float* scratch = reinterpret_cast<float*>(smem + 256);   // aliases the ring: [cw][R][NT]
  if (warp >= 1) {
    const int cw = warp - 1;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nn = 0; nn < NN; ++nn)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int row = mt * 16 + (lane >> 2) + (c >> 1) * 8;
          const int tok = nn * 8 + (lane & 3) * 2 + (c & 1);
          scratch[(cw * R + row) * NT + tok] = acc[mt][nn][c];
        }
  }
  __syncthreads();
  const int N = gridDim.x * R;
  for (int idx = tid; idx < R * m; idx += 160) {
    const int tok = idx / R, row = idx % R;
    const float s0 = scratch[(0 * R + row) * NT + tok], s1 = scratch[(1 * R + row) * NT + tok];
    const float s2 = scratch[(2 * R + row) * NT + tok], s3 = scratch[(3 * R + row) * NT + tok];
    out[(size_t)tok * N + (size_t)blockIdx.x * R + row] = __float2bfloat16_rn(((s0 + s1) + s2) + s3);
  }
#else
  if (threadIdx.x == 0) atomicExch(err, 3);
#endif
}

// One instance: its smem size (from the SAME template S as the kernel), launch and setup. No static
// or global state: the function attributes are set by bulk_setup() in prepare(), before any launch.
template <int R, int K, int NT, int S, int MINB>
struct BulkInst {
  static constexpr int XS = (8192 / R) * 2 + 16;
  static constexpr int SMEM = 256 + S * (16384 + NT * XS);
  static void launch(const torch::Tensor& wt, const torch::Tensor& x, const torch::Tensor& out,
                     const torch::Tensor& err, int m) {
    const int64_t n = out.size(1);
    TORCH_CHECK(n % R == 0 && x.size(1) == K && wt.numel() == n * K, "bulk_run: shape");
    TORCH_CHECK(m >= 1 && m <= NT && x.size(0) == m && out.size(0) == m, "bulk_run: m");
    bulk_kernel<R, K, NT, S, MINB><<<(unsigned)(n / R), 160, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(wt.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // {regs, local (spill) B, static smem B, dyn smem B, occupancy CTAs/SM, S, MINB, R, NT}
  static std::vector<int64_t> setup() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(bulk_kernel<R, K, NT, S, MINB>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    // Max-shared carveout only where 2 CTAs/SM must co-reside (A2); A / A3 keep the driver default,
    // as in the sibling combo, so cell A behaves the same in both builds.
    if constexpr (MINB > 1)
      C10_CUDA_CHECK(cudaFuncSetAttribute(bulk_kernel<R, K, NT, S, MINB>,
                                          cudaFuncAttributePreferredSharedMemoryCarveout,
                                          (int)cudaSharedmemCarveoutMaxShared));
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, bulk_kernel<R, K, NT, S, MINB>));
    int occ = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, bulk_kernel<R, K, NT, S, MINB>,
                                                                 160, (size_t)SMEM));
    return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)SMEM,
            (int64_t)occ, (int64_t)S, (int64_t)MINB, (int64_t)R, (int64_t)NT};
  }
};
using BulkPooler = BulkInst<32, 8192, 8, 6, 1>;   // A:  pooler   m=1, grid 128
using BulkQkv = BulkInst<16, 8192, 8, 4, 2>;      // A2: qkv_gate m=1, grid 144, 2 CTAs/SM
using BulkO = BulkInst<64, 1024, 8, 6, 1>;        // A3: o        m=1, grid 128

void bulk_run(torch::Tensor wt, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t inst, int64_t m) {
  TORCH_CHECK(wt.is_cuda() && x.is_cuda() && out.is_cuda() && err.is_cuda(), "bulk_run: cuda tensors");
  TORCH_CHECK(wt.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16 && err.scalar_type() == at::kInt, "bulk_run: dtypes");
  TORCH_CHECK(wt.is_contiguous() && x.is_contiguous() && out.is_contiguous(), "bulk_run: contiguity");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && err.numel() >= 1, "bulk_run: dims / error word");
  TORCH_CHECK(wt.get_device() == x.get_device() && out.get_device() == x.get_device() &&
              err.get_device() == x.get_device(), "bulk_run: one device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wt.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "bulk_run: 16 B alignment");
  const int mi = (int)m;
  if (inst == 0) BulkPooler::launch(wt, x, out, err, mi);
  else if (inst == 1) BulkQkv::launch(wt, x, out, err, mi);
  else if (inst == 2) BulkO::launch(wt, x, out, err, mi);
  else TORCH_CHECK(false, "bulk_run: unknown instance");
}

std::vector<int64_t> bulk_setup(int64_t inst) {
  if (inst == 0) return BulkPooler::setup();
  if (inst == 1) return BulkQkv::setup();
  if (inst == 2) return BulkO::setup();
  TORCH_CHECK(false, "bulk_setup: unknown instance");
  return {};
}
"""

_LT_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cstring>
#include <vector>

#define MUSE_LT_CHECK(expr)                                                              \
  do {                                                                                   \
    cublasStatus_t muse_s_ = (expr);                                                     \
    TORCH_CHECK(muse_s_ == CUBLAS_STATUS_SUCCESS, #expr " -> cublasStatus ",             \
                static_cast<int>(muse_s_));                                              \
  } while (0)

static_assert(sizeof(cublasLtMatmulAlgo_t) == 8 * sizeof(int64_t),
              "cublasLtMatmulAlgo_t is expected to be uint64_t[8]");

// Per-call host descriptors for out[m, n] = x[m, k] @ w[n, k]^T (column-major D[n, m] =
// op_T(W)[n, k] * x[k, m], bf16 in / out, fp32 compute and scale). Created and destroyed around one
// call: no static or global state. Under CUDA graph replay none of this host code runs.
struct MuseLtDesc {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  MuseLtDesc(int64_t n, int64_t k, int64_t m) {
    MUSE_LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    MUSE_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    MUSE_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    MUSE_LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));   // W as stored, op T
    MUSE_LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));   // x
    MUSE_LT_CHECK(cublasLtMatrixLayoutCreate(&d, CUDA_R_16BF, n, m, n));   // out
  }
  ~MuseLtDesc() {
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
  MuseLtDesc(const MuseLtDesc&) = delete;
  MuseLtDesc& operator=(const MuseLtDesc&) = delete;
};

template <typename T>
static int64_t muse_lt_cfg(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  T v{};
  size_t written = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), &written) != CUBLAS_STATUS_SUCCESS ||
      written != sizeof(v))
    return -1;
  return static_cast<int64_t>(v);
}

// Heuristic candidates: {algos int64 [count, 8] (raw cublasLtMatmulAlgo_t), info int64 [count, 8] =
// (algo_id, tile_id, stages_id, splitk_num, reduction_scheme, workspace_bytes, waves*1000, state)}.
std::vector<torch::Tensor> candidates(int64_t n, int64_t k, int64_t m, int64_t max_algos, int64_t ws_bytes) {
  MuseLtDesc d(n, k, m);
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  cublasLtMatmulPreference_t pref;
  MUSE_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const uint64_t ws = static_cast<uint64_t>(ws_bytes);
  cublasStatus_t st = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  std::vector<cublasLtMatmulHeuristicResult_t> res(static_cast<size_t>(max_algos));
  int returned = 0;
  if (st == CUBLAS_STATUS_SUCCESS)
    st = cublasLtMatmulAlgoGetHeuristic(h, d.op, d.a, d.b, d.d, d.d, pref, static_cast<int>(max_algos),
                                        res.data(), &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  MUSE_LT_CHECK(st);
  auto algos = torch::zeros({returned, 8}, torch::kInt64);
  auto info = torch::full({returned, 8}, -1, torch::kInt64);
  int64_t* ap = algos.data_ptr<int64_t>();
  int64_t* ip = info.data_ptr<int64_t>();
  for (int i = 0; i < returned; ++i) {
    const cublasLtMatmulAlgo_t& a = res[i].algo;
    std::memcpy(ap + 8 * i, &a, sizeof(a));
    ip[8 * i + 0] = muse_lt_cfg<int32_t>(a, CUBLASLT_ALGO_CONFIG_ID);
    ip[8 * i + 1] = muse_lt_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_TILE_ID);
    ip[8 * i + 2] = muse_lt_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_STAGES_ID);
    ip[8 * i + 3] = muse_lt_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    ip[8 * i + 4] = muse_lt_cfg<uint32_t>(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
    ip[8 * i + 5] = static_cast<int64_t>(res[i].workspaceSize);
    ip[8 * i + 6] = static_cast<int64_t>(res[i].wavesCount * 1000.0f);
    ip[8 * i + 7] = static_cast<int64_t>(res[i].state);
  }
  return {algos, info};
}

// out[m, n] = x[m, k] @ w[n, k]^T with the given algo on the current stream: no sync, no allocation,
// the workspace is the caller's prepare()-allocated per-state tensor (graph-capturable).
void run(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor algo_row, torch::Tensor ws) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda() && ws.is_cuda(), "cuda tensors expected");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16, "bf16 x / w / out expected");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "2-D tensors expected");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() && ws.is_contiguous(),
              "contiguous tensors expected");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k && out.size(0) == m && out.size(1) == n, "shape mismatch");
  TORCH_CHECK(!algo_row.is_cuda() && algo_row.scalar_type() == at::kLong && algo_row.numel() == 8 &&
              algo_row.is_contiguous(), "algo_row must be a CPU int64 [8]");
  MuseLtDesc d(n, k, m);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo, algo_row.data_ptr<int64_t>(), sizeof(algo));
  const float alpha = 1.0f, beta = 0.0f;
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  MUSE_LT_CHECK(cublasLtMatmul(h, d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta,
                               out.data_ptr(), d.d, out.data_ptr(), d.d, &algo, ws.data_ptr(),
                               static_cast<size_t>(ws.numel() * ws.element_size()), stream));
}
"""


def _bulk_build():
    """Compile the bulk row-owner module once per process, at import (never during capture), for
    9.0a only (TORCH_CUDA_ARCH_LIST set around the build and restored after). Prints the arch flags
    and the build seconds. Returns None (master paths) on any failure; never raises."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils import cpp_extension
        try:
            print(f"[bulk] arch flags passed to nvcc: {cpp_extension._get_cuda_arch_flags()}", flush=True)
        except Exception as e:  # diagnostic only
            print(f"[bulk] arch flags unavailable ({type(e).__name__}: {e})", flush=True)
        print(f"[bulk] building {BULK_EXT_NAME}: extra_cuda_cflags {BULK_NVCC_FLAGS}", flush=True)
        ext = cpp_extension.load_inline(
            name=BULK_EXT_NAME, cpp_sources=_BULK_CPP, cuda_sources=_BULK_CUDA,
            functions=["bulk_run", "bulk_setup"],
            extra_cuda_cflags=list(BULK_NVCC_FLAGS), verbose=False)
    except Exception as e:
        tail = "\n".join(str(e).splitlines()[-60:])
        print(f"[bulk] build FAILED after {time.perf_counter() - t0:.1f}s ({type(e).__name__}); build log "
              f"tail:\n{tail}\n[bulk] master paths kept", flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    print(f"[build-time] bulk load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext


def _lt_build():
    """Compile the cuBLASLt plan module (C++ only, with_cuda, -lcublasLt) at import. Prints the
    build seconds. Returns None (master path) on any failure; never raises."""
    t0 = time.perf_counter()
    try:
        from torch.utils.cpp_extension import load_inline
        ext = load_inline(
            name=LT_EXT_NAME, cpp_sources=_LT_CPP, functions=["candidates", "run"],
            with_cuda=True, extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"], verbose=False)
    except Exception as e:
        print(f"[lt] load_inline FAILED after {time.perf_counter() - t0:.1f}s, master path kept: {e!r}",
              flush=True)
        return None
    print(f"[build-time] lt load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext


_BULK_EXT = _bulk_build() if torch.cuda.is_available() else None
_LT_EXT = _lt_build() if torch.cuda.is_available() else None


def _bulk_repack(w: torch.Tensor, rows: int) -> torch.Tensor:
    """Lossless permutation of a bf16 [N, K] weight into Wt[cta][kblock][row][64 k], with the 8
    16-byte segments of each 128 B row swizzled by seg ^ (row & 7). Verified bit-exact."""
    n, k = w.shape
    c, kb = n // rows, k // 64
    src = w.contiguous().view(c, rows, kb, 8, 8).permute(0, 2, 1, 3, 4)  # [c, kb, row, seg, e]
    rr = torch.arange(rows, device=w.device).view(rows, 1)
    ss = torch.arange(8, device=w.device).view(1, 8) ^ (rr & 7)
    dst = src[:, :, rr, ss, :].contiguous()                                # [c, kb, row, seg', e]
    back = dst[:, :, rr, ss, :]                                            # xor is an involution
    ok = torch.equal(back.reshape(c, kb, rows, 64).permute(0, 2, 1, 3).reshape(n, k).view(torch.int16),
                     w.contiguous().view(torch.int16))
    if not ok:
        raise RuntimeError("bulk repack is not bit-exact")
    return dst.view(n, k)


def _bulk_route_ok(layer: str, x: torch.Tensor, cfg, state: dict) -> bool:
    """Bulk row-owner cell: layer in BULK_CELLS with its prepare()-set flag, cfg m == 1 and the
    cell's n / k, x a contiguous 16-byte-aligned bf16 [1, k]. cfg, shapes and state flags only."""
    spec = BULK_CELLS.get(layer)
    if spec is None or not state[spec[7]]:
        return False
    n, k = spec[2], spec[3]
    return (cfg["m"] == BULK_M and cfg["n"] == n and cfg["k"] == k
            and x.dim() == 2 and x.shape[0] == BULK_M and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0)


def _bulk_forward(layer: str, x: torch.Tensor, state: dict) -> torch.Tensor:
    """ONE launch: out[m, n] = x @ W.T, whole K per CTA, bf16 stored once, no workspace. Reads the
    state's repacked weight and error word; writes nothing to state."""
    spec = BULK_CELLS[layer]
    out = torch.empty((x.shape[0], spec[2]), dtype=torch.bfloat16, device=x.device)
    _BULK_EXT.bulk_run(state["bulk_wt_" + layer], x, out, state["bulk_err"], spec[1], x.shape[0])
    return out


def _lt_route_ok(x: torch.Tensor, cfg, state: dict) -> bool:
    """qkv_gate m == 128 on the prepare()-chosen cuBLASLt algo: flag set, cfg n / k the weight's full
    shape, x a contiguous 256-byte-aligned bf16 [128, k]. cfg, shapes and state only."""
    if not state[LT_FLAG]:
        return False
    w = state["qkv_gate"]
    return (cfg["m"] == LT_M and cfg["n"] == w.shape[0] and cfg["k"] == w.shape[1]
            and x.dim() == 2 and x.shape[0] == LT_M and x.shape[1] == w.shape[1]
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % LT_ALIGN == 0)


def _lt_forward(x: torch.Tensor, w: torch.Tensor, algo_row: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """x bf16 [m, k] @ w[n, k].T -> bf16 [m, n]: one cublasLtMatmul with a fixed algo."""
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    _LT_EXT.run(x, w, out, algo_row, ws)
    return out


def _cell_check(tag: str, fn, w: torch.Tensor, m: int, n: int, k: int, err=None, rel_bound: float = CELL_REL_L2,
                repeat_rel=None) -> str:
    """prepare()-only canary + correctness for one new path fn(x) -> [m, n] bf16. Every launch is
    announced with a flushed print and followed by a synchronize, the error word (if any) and a
    flushed print, so a fault names its kernel. Random x: rel L2 vs fp64 <= rel_bound (CELL_REL_L2), finite,
    a second launch bit-identical (repeat_rel None) or within repeat_rel of the first (R-PDL: relaxed fp32 atomics
    are order-nondeterministic); probe x (row 0 a single power-of-two entry): that row's rel error
    vs the weight column <= CELL_PROBE_REL. Returns "" on pass, else why."""
    dev = w.device
    g = torch.Generator(device=dev)
    g.manual_seed(8642 + m)
    wd = w[:n].double()
    x = torch.randn((m, k), device=dev, dtype=torch.float32, generator=g).to(torch.bfloat16)
    xp = x.clone()
    xp[0].zero_()
    col = (1234 + 7 * m) % k
    xp[0, col] = 0.5
    outs = []
    for label, xin in (("random", x), ("random-again", x), ("probe", xp)):
        print(f"[cell] LAUNCH {tag} {label} m={m} ...", flush=True)
        o = fn(xin)
        torch.cuda.synchronize()
        code = 0
        if err is not None:
            code = int(err.item())
            err.zero_()
        print(f"[cell] DONE   {tag} {label} m={m} (synced, error word {code})", flush=True)
        if code:
            return f"{label}: error word {code}"
        outs.append(o)
    if repeat_rel is None:
        if not torch.equal(outs[0], outs[1]):
            return "second launch differs from the first"
    else:
        rrel = float((outs[1].double() - outs[0].double()).norm() / outs[0].double().norm().clamp_min(1e-30))
        if not rrel <= repeat_rel:
            return f"second launch differs from the first by rel L2 {rrel:.3e} > {repeat_rel:g}"
    for o, label in ((outs[0], "random"), (outs[2], "probe")):
        if not bool(torch.isfinite(o.float()).all()):
            return f"{label}: non-finite output"
    ref = x.double() @ wd.T
    rel = float((outs[0].double() - ref).norm() / ref.norm().clamp_min(1e-30))
    pref = wd[:, col] * 0.5
    prel = float((outs[2][0].double() - pref).norm() / pref.norm().clamp_min(1e-30))
    print(f"[cell] check {tag} m={m}: rel L2 {rel:.3e} (<= {rel_bound:g}), probe row rel {prel:.3e} "
          f"(<= {CELL_PROBE_REL:g})", flush=True)
    if not rel <= rel_bound:
        return f"rel L2 {rel:.3e} > {rel_bound:g}"
    if not prel <= CELL_PROBE_REL:
        return f"probe row rel {prel:.3e} > {CELL_PROBE_REL:g}"
    return ""


def _cell_capture(fn):
    """Graph-capture fn() after 3 eager warm calls (prepare()-only; every new kernel has already run
    in its canary)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    torch.cuda.synchronize()
    return gr


def _cell_gate(tag: str, layer: str, m: int, n: int, k: int, flag: str, state: dict, flush_mode: str = "write",
               stats=None, route_txt=None, extra=None):
    """Model selector (copied from _ouns_gate): kernel_fn captured with the cell's flag on
    (candidate) and off (master's path, byte-identical routing otherwise) on shallow copies of
    state, the same synthetic x; CELL_GATE_REPS interleaved replays (order swapped every rep), a
    CELL_FLUSH_BYTES flush before each, medians minus an empty-graph baseline. flush_mode "write"
    (default, section 9's callers) zeroes the buffer; "read" reads all of it (one int32 sum, as the
    judge does before each replay and as rewrite/lib/rt.py _rt_time(flush="read")), which leaves L2
    full of clean lines instead of ~50 MB of dirty ones that the timed kernel would write back
    (DISTILL 3.2.1). Returns (candidate_us, master_us), both net. stats (optional dict, pooler K-halves cell) is
    filled with each side's raw median / IQR and the baseline (plus cand's / master's raw per-rep times, key
    "times"); route_txt (optional) replaces the printed route rule. Both default to None, which leaves every
    earlier caller's behaviour and print unchanged. extra (optional dict
    {arm name: flag value}, R-PDL's third arm) adds kernel_fn arms with the flag set to that value, interleaved in a
    rotating order with cand and master; their net us is printed and, with stats, filled in; routing still reads
    cand vs master only. None keeps the two-arm loop exactly. prepare()-only."""
    if flush_mode not in CELL_FLUSH_MODES:
        raise ValueError(f"_cell_gate: flush_mode {flush_mode!r} not in {CELL_FLUSH_MODES}")
    dev = state["router"].device
    st_c = dict(state)
    st_c[flag] = True
    st_i = dict(state)
    st_i[flag] = False
    x = torch.randn((m, k), device=dev, dtype=torch.float32).to(torch.bfloat16)
    data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
    flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)
    scratch = torch.zeros(1, dtype=torch.float32, device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    flush_i32 = flush.view(torch.int32)

    def replay_us(gr) -> float:
        if flush_mode == "read":
            flush_i32.sum()
        else:
            flush.zero_()
        ev[0].record()
        gr.replay()
        ev[1].record()
        ev[1].synchronize()
        return ev[0].elapsed_time(ev[1]) * 1e3

    g_base = _cell_capture(lambda: scratch.add_(1.0))
    base = _ouns_median([replay_us(g_base) for _ in range(CELL_GATE_REPS)])
    graphs = {"cand": _cell_capture(lambda: kernel_fn(data, st_c)),
              "master": _cell_capture(lambda: kernel_fn(data, st_i))}
    times = {"cand": [], "master": []}
    arms = ["cand", "master"]
    for name, val in (extra or {}).items():
        st_x = dict(state)
        st_x[flag] = val
        graphs[name] = _cell_capture(lambda s=st_x: kernel_fn(data, s))
        times[name] = []
        arms.append(name)
    for rep in range(CELL_GATE_REPS):
        if not extra:
            order = ("cand", "master") if rep % 2 == 0 else ("master", "cand")
        else:
            order = arms[rep % len(arms):] + arms[:rep % len(arms)]
        for name in order:
            times[name].append(replay_us(graphs[name]))
    med = {name: _ouns_median(v) for name, v in times.items()}
    adj = {name: max(v - base, 0.25) for name, v in med.items()}
    print(f"[cell] gate {tag} {layer} m={m} (graph replay, {CELL_FLUSH_BYTES >> 20} MB {flush_mode} flush, "
          f"{CELL_GATE_REPS} reps, interleaved): cand median {med['cand']:.2f} us, master median "
          f"{med['master']:.2f} us, empty-graph baseline {base:.2f} us -> net {adj['cand']:.2f} vs "
          f"{adj['master']:.2f} us, cand/master {adj['cand'] / adj['master']:.3f} ("
          + (route_txt if route_txt is not None else f"route if <= {CELL_MAX_RATIO}") + ")", flush=True)
    for name in arms[2:]:
        print(f"[cell] gate {tag} {layer} m={m}: arm {name} median {med[name]:.2f} us -> net {adj[name]:.2f} us "
              f"(cand {adj['cand']:.2f}, master {adj['master']:.2f})", flush=True)
    if stats is not None:
        for name, v in times.items():
            sv = sorted(v)
            stats[name] = {"med": med[name], "iqr": sv[(3 * len(sv)) // 4] - sv[len(sv) // 4], "net": adj[name]}
            if name in ("cand", "master"):
                stats[name]["times"] = list(v)   # raw per-rep us (khalf's pooled IQR); nothing else reads it
        stats["base"] = base
    del graphs, g_base, flush, flush_i32
    torch.cuda.synchronize()
    return adj["cand"], adj["master"]


def _cell_kernels(fn):
    """prepare()-only: the CUDA kernel names one call of fn launches, from torch.profiler (a
    PATH CHOSEN line proves the selector, not the launched kernel). Raises if the profiler fails."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    # Kineto bookkeeping rows are not launches; drop them so they cannot fail a "one kernel" assert.
    return [e.name for e in prof.events() if "CUDA" in str(getattr(e, "device_type", ""))
            and not e.name.startswith(("Activity Buffer Request", "Buffer Flush"))]


def _bulk_prepare(state: dict, flags=None) -> None:
    """A / A2 / A3. Sets state[flag] (False unless every step passes) and, per routed cell, the
    state's own repacked weight; one per-state error word. flags None (the judge's state): per cell
    query instance attributes + occupancy (A2 needs exactly 2 CTAs/SM), repack, canary +
    correctness, model-selector gate, profiler kernel-name assertion, one PATH CHOSEN line. flags a
    dict (the A/B check's second state): inherit each flag, repack only for cells that are on."""
    for spec in BULK_CELLS.values():
        state[spec[7]] = False
    if _BULK_EXT is None:
        if flags is None:
            for layer, spec in BULK_CELLS.items():
                _decision(f"PATH CHOSEN {spec[0]} {layer} m={BULK_M}: master (cand n/a us vs master n/a us; "
                      f"no bulk module)")
        return
    dev = state["router"].device
    state["bulk_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        for layer, spec in BULK_CELLS.items():
            if flags.get(spec[7]):
                try:   # a failure here turns off only this cell in this state, never the whole A/B
                    state["bulk_wt_" + layer] = _bulk_repack(state[layer], spec[4])
                    state[spec[7]] = True
                except Exception as e:
                    print(f"[bulk] second state: {spec[0]} {layer} repack raised {type(e).__name__}: {e}; "
                          f"cell off in this state", flush=True)
        return
    for layer, spec in BULK_CELLS.items():
        src, inst, n, k, rows, stages, minb, flag = spec
        t0 = time.perf_counter()
        cand_us = master_us = float("nan")
        why = ""
        try:
            w = state[layer]
            if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16
                    and w.is_contiguous()):
                why = f"weight {tuple(w.shape)} {w.dtype} unexpected"
            else:
                info = [int(v) for v in _BULK_EXT.bulk_setup(inst)]
                occ = info[4]
                print(f"[bulk] {src} {layer} bulk_kernel<{rows}, {k}, 8, {stages}, {minb}>: regs={info[0]} "
                      f"local(spill)={info[1]} B static smem={info[2]} B dyn smem={info[3]} B "
                      f"(S={info[5]} from the template), occupancy {occ} CTA/SM (needs {minb}), grid "
                      f"{n // rows}, max-shared carveout set", flush=True)
                if occ < minb:
                    why = f"occupancy {occ} CTA/SM < {minb}"
                else:
                    state["bulk_wt_" + layer] = _bulk_repack(w, rows)
                    print(f"[bulk] {src} {layer}: repack Wt{tuple(state['bulk_wt_' + layer].shape)} rows {rows}, "
                          f"lossless bf16 permutation, bit-exact round trip verified", flush=True)
                    st_on = dict(state)
                    st_on[flag] = True
                    why = _cell_check(f"{src} bulk_kernel<{rows},{k},8,{stages},{minb}>",
                                      lambda xx: _bulk_forward(layer, xx, st_on), w, BULK_M, n, k,
                                      err=state["bulk_err"])
                    if not why:
                        tg = time.perf_counter()
                        cand_us, master_us = _cell_gate(src, layer, BULK_M, n, k, flag, state)
                        proj = n * k * 2 / (BULK_REF_TBS * 1e12) * 1e6
                        print(f"[prepare-time] gate {src} {layer} m={BULK_M}: {time.perf_counter() - tg:.2f}s; "
                              f"projected {n * k * 2 / 1e6:.1f} MB / {BULK_REF_TBS} TB/s = {proj:.2f} us", flush=True)
                        code = int(state["bulk_err"].item())
                        if code:
                            why = f"error word {code} after the gate"
                        elif not cand_us <= CELL_MAX_RATIO * master_us:
                            why = f"gate: cand/master {cand_us / master_us:.3f} > {CELL_MAX_RATIO}"
                        else:
                            data = {"x": torch.randn((BULK_M, k), device=dev, dtype=torch.float32).to(torch.bfloat16),
                                    "config": {"m": BULK_M, "n": n, "k": k, "layer": layer}}
                            try:
                                names = _cell_kernels(lambda: kernel_fn(data, st_on))
                                print(f"[bulk] {src} {layer} m={BULK_M}: profiled kernels {names}", flush=True)
                                if not names:
                                    print(f"[bulk] {src} {layer}: profiler returned no CUDA kernel events; "
                                          f"route kept (kernel_fn calls bulk_run directly)", flush=True)
                                elif not (len(names) == 1 and "bulk_kernel" in names[0]):
                                    why = f"profiled kernels {names}: expected exactly one bulk_kernel"
                            except Exception as e:  # own extension, direct launch: keep the route
                                print(f"[bulk] {src} {layer}: profiler unavailable ({type(e).__name__}: {e}); "
                                      f"route kept (kernel_fn calls bulk_run directly)", flush=True)
                            if not why:
                                state[flag] = True
        except Exception as e:  # never raises: any surprise keeps master's path for this cell
            why = f"raised {type(e).__name__}: {str(e)[-300:]}"
            state[flag] = False
        if not state[flag]:
            state.pop("bulk_wt_" + layer, None)
        state["bulk_err"].zero_()
        print(f"[prepare-time] cell {src} {layer} m={BULK_M} total: {time.perf_counter() - t0:.2f}s", flush=True)
        _decision(f"PATH CHOSEN {src} {layer} m={BULK_M}: {'candidate' if state[flag] else 'master'} "
              f"(cand {cand_us:.2f} us vs master {master_us:.2f} us)"
              + ("" if state[flag] else f"; {why}"))
    torch.cuda.synchronize()


def _lt_graph_median_us(fn, flush: torch.Tensor, reps: int) -> float:
    """Median of reps graph replays of fn() after a READ flush each (prepare()-only ranking). phasec
    pc-lt-qkv128-readflush: was a write flush (zero_), which leaves dirty lines the timed kernel writes back; the
    judge reads a buffer larger than L2 before each replay (DISTILL 3.2.1)."""
    gr = _cell_capture(fn)
    ts = []
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    flush_i32 = flush.view(torch.int32)
    for _ in range(reps):
        flush_i32.sum()
        ev[0].record()
        gr.replay()
        ev[1].record()
        ev[1].synchronize()
        ts.append(ev[0].elapsed_time(ev[1]) * 1e3)
    del gr
    return _ouns_median(ts)


def _lt_prepare(state: dict, flags=None) -> None:
    """B: qkv_gate m == 128. flags None: query cublasLtMatmulAlgoGetHeuristic at workspace caps
    LT_QUERY_WS, keep split-K-1 candidates only, canary + correctness-check each eagerly, rank them
    by graph replay, gate the best against master via kernel_fn, assert by torch.profiler that the
    routed call launches exactly one kernel with no split-K / reduce in its name (else drop the
    cell), print one PATH CHOSEN line. flags a dict: inherit the flag and the algo row, allocate the
    state's own workspace."""
    state[LT_FLAG] = False
    state["lt_algo_qkv128"] = None
    state["lt_ws_qkv128"] = None
    if _LT_EXT is None:
        if flags is None:
            _decision(f"PATH CHOSEN B qkv_gate m={LT_M}: master (cand n/a us vs master n/a us; no Lt module)")
        return
    w = state["qkv_gate"]
    dev = w.device
    if flags is not None:
        if flags.get(LT_FLAG) and flags.get("_lt_algo") is not None:
            state["lt_algo_qkv128"] = flags["_lt_algo"].clone()
            state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)
            state[LT_FLAG] = True
        return
    t0 = time.perf_counter()
    cand_us = master_us = float("nan")
    why = ""
    try:
        if not (w.dim() == 2 and tuple(w.shape) == (QKV_N, QKV_K) and w.dtype == torch.bfloat16
                and w.is_contiguous()):
            raise RuntimeError(f"weight {tuple(w.shape)} unexpected")
        n, k = QKV_N, QKV_K
        ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)
        cands = {}
        for cap in LT_QUERY_WS:
            algos, info = _LT_EXT.candidates(n, k, LT_M, LT_MAX_ALGOS, cap)
            for i in range(algos.shape[0]):
                algo_id, tile, stg, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())
                desc = (f"[lt] cap={cap} cand {i:2d} algo={algo_id} tile={tile} stages={stg} splitK={splitk} "
                        f"red={red} ws={ws_need} waves={waves / 1000:.2f}")
                if st != 0 or splitk != 1 or ws_need < 0 or ws_need > LT_WS_BYTES:
                    print(f"{desc}: excluded (state {st}; only split-K 1 within {LT_WS_BYTES} B kept)", flush=True)
                    continue
                key = tuple(int(v) for v in algos[i].tolist())
                if key in cands:
                    print(f"{desc}: duplicate", flush=True)
                    continue
                cands[key] = (algos[i].clone().contiguous(), desc, f"a{algo_id}/t{tile}/s{stg}")
        flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)
        x = torch.randn((LT_M, k), device=dev, dtype=torch.float32).to(torch.bfloat16)
        best = None
        ranked = []   # (read-flush us, short label) per timed candidate, for the B-RANK decision line
        for row, desc, short in cands.values():
            bad = _cell_check(f"B cublasLt {desc[5:]}", lambda xx, row=row: _lt_forward(xx, w, row, ws),
                              w, LT_M, n, k)
            if bad:
                print(f"{desc}: excluded ({bad})", flush=True)
                continue
            us = _lt_graph_median_us(lambda row=row: _lt_forward(x, w, row, ws), flush, LT_PICK_REPS)
            print(f"{desc}: {us:.2f} us (graph replay, read flush, median of {LT_PICK_REPS})", flush=True)
            ranked.append((us, short))
            if best is None or us < best[0]:
                best = (us, row, desc)
        del flush
        _decision(f"B-RANK qkv_gate m={LT_M} split-K-1 Lt plans, read flush, median of {LT_PICK_REPS} (us): "
                  + " ".join(f"{s}:{u:.2f}" for u, s in sorted(ranked)[:10]) + f" ({len(ranked)} timed)")
        if best is None:
            why = f"no usable split-K-1 candidate ({len(cands)} queried)"
        else:
            print(f"[lt] best split-K-1 candidate: {best[2]} at {best[0]:.2f} us", flush=True)
            state["lt_algo_qkv128"] = best[1]
            state["lt_ws_qkv128"] = ws
            tg = time.perf_counter()
            cand_us, master_us = _cell_gate("B", "qkv_gate", LT_M, n, k, LT_FLAG, state, flush_mode="read")
            print(f"[prepare-time] gate B qkv_gate m={LT_M}: {time.perf_counter() - tg:.2f}s", flush=True)
            if not cand_us <= CELL_MAX_RATIO * master_us:
                why = f"gate: cand/master {cand_us / master_us:.3f} > {CELL_MAX_RATIO}"
            else:
                st_on = dict(state)
                st_on[LT_FLAG] = True
                data = {"x": x, "config": {"m": LT_M, "n": n, "k": k, "layer": "qkv_gate"}}
                try:
                    names = _cell_kernels(lambda: kernel_fn(data, st_on))
                except Exception as e:
                    names = None
                    why = f"profiler unavailable ({type(e).__name__}: {e}); kernel name unverified, cell dropped"
                if names is not None:
                    print(f"[lt] B qkv_gate m={LT_M}: profiled kernels {names}", flush=True)
                    low = [s.lower() for s in names]
                    if not (len(names) == 1 and "splitk" not in low[0] and "reduce" not in low[0]):
                        why = f"profiled kernels {names}: expected exactly one non-split-K kernel"
                    else:
                        state[LT_FLAG] = True
    except Exception as e:  # never raises: any surprise keeps master's path for this cell
        why = f"raised {type(e).__name__}: {str(e)[-300:]}"
        state[LT_FLAG] = False
    if not state[LT_FLAG]:
        state["lt_algo_qkv128"] = None
        state["lt_ws_qkv128"] = None
    torch.cuda.synchronize()
    print(f"[prepare-time] cell B qkv_gate m={LT_M} total: {time.perf_counter() - t0:.2f}s", flush=True)
    _decision(f"PATH CHOSEN B qkv_gate m={LT_M}: {'candidate' if state[LT_FLAG] else 'master'} "
          f"(read-flush pick and gate; cand {cand_us:.2f} us vs master {master_us:.2f} us)"
          + ("" if state[LT_FLAG] else f"; {why}"))


# --- (10) m == 1 row-owner stream on lossless 12-bit weights (bulk12: A12 pooler, A2_12 qkv_gate, A3_12 o) ----
# Campaign 2026-10-26-expert-bandwidth (bf16 quota, the user's 19:05Z steer). Section (9)'s row-owner shape (each
# CTA owns R output rows over the WHOLE K: no split-K, no workspace, no ticket; one producer warp streams the CTA's
# weight chunks through a cp.async.bulk + mbarrier ring) on a smaller weight stream:
#   * Codec (prepare, per weight): a bf16 is H << 8 | L with H = sign | exponent bits 7..1 and L = exponent bit 0 |
#     7 mantissa bits. L is kept raw (8 bits). H becomes a 4-bit code c = sign << 3 | (e71 - base) over the 8 values
#     of e71 = H & 0x7F that hold the most elements (16 binades): 12 bits per element instead of 16. Weights are
#     N(0, 1/k), so ~1e-4 of the elements fall outside the window: |w| < 2^-19 or so, and for o (k = 1024) also the
#     rare |w| >= 2^-3 (the audit's emulation: base 54 on every layer, at most 58 per CTA). Such an escape is
#     stored as code 0 / L 0 (it decodes to w_floor = 2^(2 base - 127)) plus an entry (row, k, true bits) in its CTA's
#     escape block. The decode is c + base + 120 * sign per byte lane (no carries), then two byte_perms per pair of
#     elements: about 2 integer ops per element. Lossless: prepare() checks the round trip on the host and on the
#     device (b12_unpack, the kernel's own decode) bit-exact before any timing.
#   * Layout: per CTA, per chunk of CKK = 8192 / R k (12 KB instead of 16 KB), per k16 step, per m16 tile: the
#     mma.sync A fragment of each lane, 8 L bytes (256 B per tile) then the 8 codes (128 B per tile). One LDS.64 +
#     one LDS.32 per lane per tile, conflict-free.
#   * x [1, K] is copied once at kernel start (with the CTA's 2 KB escape block) and stays in smem; every token column
#     of the n8 B fragment reads x row 0 (columns 1..7 are duplicates and are never stored).
#   * CW consumer warps split each chunk's k16 steps; the epilogue sums the CW partials per row in a fixed order, then
#     applies the row's escapes in k order (two fp32 FMAs each: + x * w, - x * w_floor): deterministic, and exact up
#     to fp32 accumulation like any GEMM.
# Each layer tries its instances (CTA shapes) behind section (9)'s cell machinery: occupancy query, encode, host +
# device round trip, canary + fp64 correctness (_cell_check), the model-selector graph gate (_cell_gate, which times
# kernel_fn with the cell's flag on vs off: off is whatever routes today, the raw bulk cell included), and routes the
# fastest only at candidate / incumbent <= CELL_MAX_RATIO, after a torch.profiler kernel-name assertion. Since
# b12-qkv-m1-r32-one-cta-per-sm-readflush-gate every bulk12 gate routes on a READ flush (the judge reads a buffer
# larger than L2 before each replay; the write flush flattered bulk12) with the write flush printed beside it, the
# threshold is 0.95 at m >= 8, and qkv_gate also tries the pooler's R=32 instances (72 CTAs, 1 CTA/SM) as challengers.
B12_EXT_NAME = "muse_bf16_bulk12_m1_v1"
B12_ESC_BYTES = 2048          # per-CTA escape block: uint16 row offsets [R + 1] in its first 256 B, then entries
B12_ESC_CAP = (B12_ESC_BYTES - 256) // 8   # 224 entries of (row << 16 | k, true bf16 bits) per CTA
B12_M = 1
# layer -> (source tag, N, K, the cell's home R (rows per CTA),
#           ((instance id, ring stages S, min CTAs/SM, consumer warps CW, rows per CTA R), ...), state flag)
# Each variant carries its own R (the encoding is per R). A variant whose R differs from the home R is a challenger
# (hypothesis b12-qkv-m1-r32-one-cta-per-sm-readflush-gate: qkv_gate on the pooler's R=32 instances 0 / 1, grid
# 2304 / 32 = 72 CTAs at 1 CTA/SM, two m16 tiles per k16 step): it routes only if it passes BOTH flush gates and its
# read-flush ratio is at least B12_CHALLENGER_MARGIN below the best home-R variant's (see _b12_pick).
B12_CELLS = {
    "pooler":   ("A12", 4096, 8192, 32, ((0, 8, 1, 8, 32), (1, 8, 1, 4, 32)), "b12_pooler_ok"),
    "qkv_gate": ("A2_12", 2304, 8192, 16, ((2, 6, 2, 4, 16), (3, 6, 2, 8, 16), (0, 8, 1, 8, 32), (1, 8, 1, 4, 32)),
                 "b12_qkv_ok"),
    "o":        ("A3_12", 8192, 1024, 64, ((4, 8, 1, 8, 64), (5, 8, 1, 4, 64)), "b12_o_ok"),
}
B12_INST_R = {0: 32, 1: 32, 2: 16, 3: 16, 4: 64, 5: 64}   # the R of each B12Inst (the CUDA template's first argument)
for _layer, _spec in B12_CELLS.items():
    for _v in _spec[4]:
        assert B12_INST_R[_v[0]] == _v[4], f"B12_CELLS {_layer}: instance {_v[0]} is R={B12_INST_R[_v[0]]}, not {_v[4]}"
        assert _spec[1] % _v[4] == 0 and _v[4] % 16 == 0, f"B12_CELLS {_layer}: n {_spec[1]} vs R {_v[4]}"
del _layer, _spec, _v
B12_FLAGS = tuple(spec[5] for spec in B12_CELLS.values())
# Every bulk12 gate times under a READ flush (DISTILL 3.2.1: the write flush flattered bulk12, qkv 0.88 gated vs ~1.00
# judged, o 0.87 vs 0.97). The write flush still runs beside it as a diagnostic and as the challenger's second gate.
B12_GATE_FLUSH = "read"
B12_MAX_RATIO_M8 = 0.95       # route at m >= 8 only at read-flush cand / incumbent <= this (b12x / b12r lessons)
B12_CHALLENGER_MARGIN = 0.03  # a challenger-R variant must beat the best home-R variant's read ratio by this much


def _b12_max_ratio(m: int) -> float:
    """The routing threshold on the read-flush cand / incumbent ratio: CELL_MAX_RATIO at m < 8, B12_MAX_RATIO_M8 at
    m >= 8."""
    return B12_MAX_RATIO_M8 if m >= 8 else CELL_MAX_RATIO

_B12_CPP = r"""
#include <torch/extension.h>
#include <vector>
void b12_run(torch::Tensor wc, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor err,
             int64_t inst, int64_t base);
void b12_unpack(torch::Tensor wc, torch::Tensor out, int64_t inst, int64_t base);
std::vector<int64_t> b12_setup(int64_t inst);
"""

_B12_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

#define B12_WAIT_NS 500000000ULL   // every mbarrier wait is bounded to 0.5 s, then an error word
#define B12_ESC_BYTES 2048         // per-CTA escape block (B12_ESC_BYTES in Python)
#define B12_TILE 384               // one m16 x k16 tile: 32 lanes x (8 L bytes + 4 code bytes)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Bounded wait: false after B12_WAIT_NS (the caller sets the error word and leaves the loop).
__device__ __forceinline__ bool mbar_wait(uint32_t a, uint32_t parity) {
  unsigned long long t0 = 0;
  for (uint32_t it = 0;; ++it) {
    uint32_t done;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) return true;
    if (it == 0) t0 = gtimer();
    else if (gtimer() - t0 > B12_WAIT_NS) return false;
  }
}
// Plain 1-D bulk copy global -> shared, completion counted on the mbarrier.
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void lds64(uint32_t& v0, uint32_t& v1, uint32_t a) {
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v0), "=r"(v1) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four high bytes from four codes (one per byte lane, low nibble: bit 3 the sign, bits 2..0 e71 - base):
// c + base + 120 * sign = 128 * sign + (e71 - base) + base. Every lane stays <= 255: no carry between lanes.
__device__ __forceinline__ uint32_t b12_hi4(uint32_t c4, uint32_t base4) {
  const uint32_t t = (c4 >> 3) & 0x01010101u;
  return t * 0x78u + (c4 + base4);
}
// One lane's m16n8k16 A fragment: l0 holds the low bytes of a0 (elements 0, 1) and a1 (2, 3), l1 those of a2 and
// a3; byte j of cw holds the code of l0's element j (low nibble) and of l1's element j (high nibble).
__device__ __forceinline__ void b12_frag(uint32_t l0, uint32_t l1, uint32_t cw, uint32_t base4,
                                         uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  const uint32_t h0 = b12_hi4(cw & 0x0F0F0F0Fu, base4);
  const uint32_t h1 = b12_hi4((cw >> 4) & 0x0F0F0F0Fu, base4);
  a0 = __byte_perm(l0, h0, 0x5140);
  a1 = __byte_perm(l0, h0, 0x7362);
  a2 = __byte_perm(l1, h1, 0x5140);
  a3 = __byte_perm(l1, h1, 0x7362);
}

// R: output rows per CTA; K: reduction length; S: ring stages; MINB: __launch_bounds__ min CTAs/SM; CW: consumer
// warps. Grid = N / R; block = 32 * (CW + 1); m == 1. err codes: 1 producer wait timeout, 2 consumer wait timeout,
// 3 built without sm_90.
template <int R, int K, int S, int MINB, int CW>
__global__ void __launch_bounds__(32 * (CW + 1), MINB)
b12_kernel(const uint8_t* __restrict__ wc, const uint8_t* __restrict__ esc, const __nv_bfloat16* __restrict__ x,
           __nv_bfloat16* __restrict__ out, int* err, uint32_t base) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int CKK = 8192 / R;              // K per chunk: the raw chunk would be 16 KB, this one is 12 KB
  constexpr int NCHUNK = K / CKK;
  constexpr int NS = CKK / 16;               // k16 steps per chunk
  constexpr int KSW = NS / CW;               // k16 steps per consumer warp per chunk
  constexpr int MT = R / 16;
  constexpr int CHUNK = NS * MT * B12_TILE;
  constexpr int XOFF = 256 + B12_ESC_BYTES;  // barriers, escape block, x row 0, ring
  constexpr int RING = XOFF + K * 2;
  static_assert(K % CKK == 0 && NS % CW == 0 && R % 16 == 0 && CHUNK == 12288, "geometry");
  static_assert(S >= 2 && S <= 8, "full / empty barriers live in 64 B each");
  static_assert(RING % 128 == 0 && (K * 2) % 16 == 0, "alignment");
  static_assert(CW * R * 4 <= S * CHUNK, "epilogue scratch must fit in the ring");
  static_assert(2 * (R + 1) <= 256, "row offsets must fit the escape header");
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  const uint32_t full0 = sbase, empty0 = sbase + 64, xbar = sbase + 128;
  const uint32_t escs = sbase + 256, xs = sbase + XOFF, ring = sbase + RING;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, CW); }
    mbar_init(xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float acc[MT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[a][c] = 0.f;

  if (warp == 0) {
    // ---- producer: lane 0 copies x and the escape block once, then streams the CTA's chunks ----
    if (lane == 0) {
      mbar_expect_tx(xbar, (uint32_t)(K * 2 + B12_ESC_BYTES));
      bulk_g2s(xs, x, (uint32_t)(K * 2), xbar);
      bulk_g2s(escs, esc + (size_t)blockIdx.x * B12_ESC_BYTES, B12_ESC_BYTES, xbar);
      for (int j = 0; j < NCHUNK; ++j) {
        const int s = j % S;
        const uint32_t eph = ((j / S) & 1) ^ 1;
        if (!mbar_wait(empty0 + 8 * s, eph)) { atomicExch(err, 1); break; }
        mbar_expect_tx(full0 + 8 * s, (uint32_t)CHUNK);
        bulk_g2s(ring + (uint32_t)(s * CHUNK), wc + ((size_t)blockIdx.x * NCHUNK + j) * (size_t)CHUNK,
                 (uint32_t)CHUNK, full0 + 8 * s);
      }
    }
    __syncwarp();
    // The epilogue's row threads (all of them in this warp when R <= 32) read the escape block and x: observe the
    // copies' completion here too (that phase completed long ago, so this returns at once).
    if (!mbar_wait(xbar, 0) && lane == 0) atomicExch(err, 1);
  } else {
    // ---- consumers: warp cw owns k16 steps [cw*KSW, (cw+1)*KSW) of every chunk ----
    const int cw = warp - 1;
    const uint32_t base4 = base * 0x01010101u;
    if (!mbar_wait(xbar, 0)) {
      if (lane == 0) atomicExch(err, 2);
    } else {
      for (int j = 0; j < NCHUNK; ++j) {
        const int s = j % S;
        const uint32_t ph = (j / S) & 1;
        if (!mbar_wait(full0 + 8 * s, ph)) { if (lane == 0) atomicExch(err, 2); break; }
        const uint32_t slot = ring + (uint32_t)(s * CHUNK);
#pragma unroll
        for (int i = 0; i < KSW; ++i) {
          const int kk16 = cw * KSW + i;
          // B fragment: every token column reads x row 0 (k = 2q, 2q+1 and 2q+8, 2q+9 of this k16 step)
          const uint32_t xk = xs + (uint32_t)((j * CKK + kk16 * 16 + 2 * (lane & 3)) * 2);
          const uint32_t b0 = lds32(xk), b1 = lds32(xk + 16);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const uint32_t t = slot + (uint32_t)((kk16 * MT + mt) * B12_TILE);
            uint32_t l0, l1, a0, a1, a2, a3;
            lds64(l0, l1, t + (uint32_t)lane * 8);
            const uint32_t c = lds32(t + 256 + (uint32_t)lane * 4);
            b12_frag(l0, l1, c, base4, a0, a1, a2, a3);
            mma_bf16(acc[mt], a0, a1, a2, a3, b0, b1);
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
      }
    }
  }
  __syncthreads();   // every issued copy has been waited on by the consumers: the ring is quiescent
  float* scratch = reinterpret_cast<float*>(smem + RING);   // aliases the ring: [cw][R]
  if (warp >= 1 && (lane & 3) == 0) {
    const int cw = warp - 1;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = mt * 16 + (lane >> 2);
      scratch[cw * R + row] = acc[mt][0];       // D[row][token 0]
      scratch[cw * R + row + 8] = acc[mt][2];   // D[row + 8][token 0]
    }
  }
  __syncthreads();
  if (tid < R) {
    float sum = scratch[tid];
#pragma unroll
    for (int c = 1; c < CW; ++c) sum += scratch[c * R + tid];
    const uint16_t* roff = reinterpret_cast<const uint16_t*>(smem + 256);
    const uint32_t* ent = reinterpret_cast<const uint32_t*>(smem + 256 + 256);
    const __nv_bfloat16* xsm = reinterpret_cast<const __nv_bfloat16*>(smem + XOFF);
    const float wfloor = __uint_as_float(base << 24);   // bf16 bits base << 8: what an escape decoded to
    const int e1 = (int)roff[tid + 1];
    for (int e = (int)roff[tid]; e < e1; ++e) {
      const uint32_t pos = ent[2 * e], val = ent[2 * e + 1];
      const float xv = __bfloat162float(xsm[pos & 0xFFFFu]);
      sum = fmaf(xv, __uint_as_float(val << 16), sum);
      sum = fmaf(-xv, wfloor, sum);
    }
    out[(size_t)blockIdx.x * R + tid] = __float2bfloat16_rn(sum);
  }
#else
  if (threadIdx.x == 0) atomicExch(err, 3);
#endif
}

// Device round trip of the encoding through b12_frag: out[N, K] (int16 bits) gets every decoded element at its
// (row, k). Escapes come out as w_floor's bits (base << 8); prepare() patches and compares. One warp per chunk.
template <int R, int K>
__global__ void b12_unpack_kernel(const uint8_t* __restrict__ wc, int16_t* __restrict__ out, uint32_t base) {
  constexpr int CKK = 8192 / R, NCHUNK = K / CKK, NS = CKK / 16, MT = R / 16;
  constexpr int CHUNK = NS * MT * B12_TILE;
  const int cta = blockIdx.x / NCHUNK, j = blockIdx.x % NCHUNK, lane = threadIdx.x;
  const uint32_t base4 = base * 0x01010101u;
  const int g = lane >> 2, q = lane & 3;
  for (int ks = 0; ks < NS; ++ks)
    for (int mt = 0; mt < MT; ++mt) {
      const uint8_t* t = wc + ((size_t)cta * NCHUNK + j) * CHUNK + (size_t)(ks * MT + mt) * B12_TILE;
      const uint32_t l0 = *reinterpret_cast<const uint32_t*>(t + lane * 8);
      const uint32_t l1 = *reinterpret_cast<const uint32_t*>(t + lane * 8 + 4);
      const uint32_t c = *reinterpret_cast<const uint32_t*>(t + 256 + lane * 4);
      uint32_t a[4];
      b12_frag(l0, l1, c, base4, a[0], a[1], a[2], a[3]);
      const int r0 = cta * R + mt * 16 + g, k0 = j * CKK + ks * 16 + 2 * q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = r0 + ((i & 1) ? 8 : 0), k = k0 + ((i & 2) ? 8 : 0);
        out[(size_t)r * K + k] = (int16_t)(a[i] & 0xFFFFu);
        out[(size_t)r * K + k + 1] = (int16_t)(a[i] >> 16);
      }
    }
}

template <int R, int K, int S, int MINB, int CW>
struct B12Inst {
  static constexpr int SMEM = 256 + B12_ESC_BYTES + K * 2 + S * 12288;
  static void launch(const torch::Tensor& wc, const torch::Tensor& esc, const torch::Tensor& x,
                     const torch::Tensor& out, const torch::Tensor& err, uint32_t base) {
    const int64_t n = out.size(1);
    TORCH_CHECK(n % R == 0 && x.size(1) == K && wc.numel() == n * K * 3 / 2, "b12_run: shape");
    TORCH_CHECK(esc.numel() == (n / R) * B12_ESC_BYTES, "b12_run: escape blocks");
    TORCH_CHECK(x.size(0) == 1 && out.size(0) == 1, "b12_run: m must be 1");
    b12_kernel<R, K, S, MINB, CW><<<(unsigned)(n / R), 32 * (CW + 1), SMEM, at::cuda::getCurrentCUDAStream()>>>(
        wc.data_ptr<uint8_t>(), esc.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  static void unpack(const torch::Tensor& wc, const torch::Tensor& out, uint32_t base) {
    const int64_t n = out.size(0);
    TORCH_CHECK(out.size(1) == K && n % R == 0 && wc.numel() == n * K * 3 / 2, "b12_unpack: shape");
    b12_unpack_kernel<R, K><<<(unsigned)((n / R) * (K / (8192 / R))), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        wc.data_ptr<uint8_t>(), out.data_ptr<int16_t>(), base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // {regs, local (spill) B, static smem B, dyn smem B, occupancy CTAs/SM, S, MINB, R, CW}
  static std::vector<int64_t> setup() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(b12_kernel<R, K, S, MINB, CW>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    if constexpr (MINB > 1)
      C10_CUDA_CHECK(cudaFuncSetAttribute(b12_kernel<R, K, S, MINB, CW>,
                                          cudaFuncAttributePreferredSharedMemoryCarveout,
                                          (int)cudaSharedmemCarveoutMaxShared));
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, b12_kernel<R, K, S, MINB, CW>));
    int occ = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, b12_kernel<R, K, S, MINB, CW>,
                                                                 32 * (CW + 1), (size_t)SMEM));
    return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)SMEM,
            (int64_t)occ, (int64_t)S, (int64_t)MINB, (int64_t)R, (int64_t)CW};
  }
};
using B12Pooler8 = B12Inst<32, 8192, 8, 1, 8>;   // inst 0: pooler   m=1, grid 128, 9 warps
using B12Pooler4 = B12Inst<32, 8192, 8, 1, 4>;   // inst 1: pooler   m=1, grid 128, 5 warps
using B12Qkv4 = B12Inst<16, 8192, 6, 2, 4>;      // inst 2: qkv_gate m=1, grid 144, 2 CTAs/SM x 5 warps
using B12Qkv8 = B12Inst<16, 8192, 6, 2, 8>;      // inst 3: qkv_gate m=1, grid 144, 2 CTAs/SM x 9 warps
using B12O8 = B12Inst<64, 1024, 8, 1, 8>;        // inst 4: o        m=1, grid 128, 9 warps
using B12O4 = B12Inst<64, 1024, 8, 1, 4>;        // inst 5: o        m=1, grid 128, 5 warps

static void b12_check(const torch::Tensor& wc, const torch::Tensor& esc, const torch::Tensor& x,
                      const torch::Tensor& out, const torch::Tensor& err) {
  TORCH_CHECK(wc.is_cuda() && esc.is_cuda() && x.is_cuda() && out.is_cuda() && err.is_cuda(), "b12: cuda tensors");
  TORCH_CHECK(wc.scalar_type() == at::kByte && esc.scalar_type() == at::kByte && x.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16 && err.scalar_type() == at::kInt, "b12: dtypes");
  TORCH_CHECK(wc.is_contiguous() && esc.is_contiguous() && x.is_contiguous() && out.is_contiguous(), "b12: contiguity");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && err.numel() >= 1, "b12: dims / error word");
  TORCH_CHECK(wc.get_device() == x.get_device() && esc.get_device() == x.get_device() &&
              out.get_device() == x.get_device() && err.get_device() == x.get_device(), "b12: one device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wc.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(esc.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "b12: 16 B alignment");
}

void b12_run(torch::Tensor wc, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor err,
             int64_t inst, int64_t base) {
  b12_check(wc, esc, x, out, err);
  TORCH_CHECK(base >= 1 && base <= 120, "b12_run: base");
  const uint32_t b = (uint32_t)base;
  switch (inst) {
    case 0: B12Pooler8::launch(wc, esc, x, out, err, b); break;
    case 1: B12Pooler4::launch(wc, esc, x, out, err, b); break;
    case 2: B12Qkv4::launch(wc, esc, x, out, err, b); break;
    case 3: B12Qkv8::launch(wc, esc, x, out, err, b); break;
    case 4: B12O8::launch(wc, esc, x, out, err, b); break;
    case 5: B12O4::launch(wc, esc, x, out, err, b); break;
    default: TORCH_CHECK(false, "b12_run: unknown instance");
  }
}

void b12_unpack(torch::Tensor wc, torch::Tensor out, int64_t inst, int64_t base) {
  TORCH_CHECK(wc.is_cuda() && out.is_cuda() && wc.scalar_type() == at::kByte && out.scalar_type() == at::kShort &&
              wc.is_contiguous() && out.is_contiguous() && out.dim() == 2, "b12_unpack: tensors");
  TORCH_CHECK(base >= 1 && base <= 120, "b12_unpack: base");
  const uint32_t b = (uint32_t)base;
  if (inst == 0 || inst == 1) B12Pooler8::unpack(wc, out, b);
  else if (inst == 2 || inst == 3) B12Qkv4::unpack(wc, out, b);
  else if (inst == 4 || inst == 5) B12O8::unpack(wc, out, b);
  else TORCH_CHECK(false, "b12_unpack: unknown instance");
}

std::vector<int64_t> b12_setup(int64_t inst) {
  switch (inst) {
    case 0: return B12Pooler8::setup();
    case 1: return B12Pooler4::setup();
    case 2: return B12Qkv4::setup();
    case 3: return B12Qkv8::setup();
    case 4: return B12O8::setup();
    case 5: return B12O4::setup();
    default: TORCH_CHECK(false, "b12_setup: unknown instance");
  }
  return {};
}
"""


def _b12_build():
    """Compile the bulk12 module once per process, at import (never during capture), for 9.0a only
    (TORCH_CUDA_ARCH_LIST set around the build and restored after). Prints the build seconds. Returns None
    (incumbent paths) on any failure; never raises."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils import cpp_extension
        print(f"[b12] building {B12_EXT_NAME}: extra_cuda_cflags {BULK_NVCC_FLAGS}", flush=True)
        ext = cpp_extension.load_inline(
            name=B12_EXT_NAME, cpp_sources=_B12_CPP, cuda_sources=_B12_CUDA,
            functions=["b12_run", "b12_unpack", "b12_setup"],
            extra_cuda_cflags=list(BULK_NVCC_FLAGS), verbose=False)
    except Exception as e:
        tail = "\n".join(str(e).splitlines()[-60:])
        print(f"[b12] build FAILED after {time.perf_counter() - t0:.1f}s ({type(e).__name__}); build log "
              f"tail:\n{tail}\n[b12] incumbent paths kept", flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    print(f"[build-time] b12 load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext


_B12_EXT = _b12_build() if torch.cuda.is_available() else None


def _b12_encode(w: torch.Tensor, rows: int):
    """Lossless 12-bit encoding of a bf16 [N, K] weight for the bulk12 kernel (see the section header). Returns
    (wc uint8 [N/R, NCHUNK, 12288], esc uint8 [N/R * B12_ESC_BYTES], base, escapes, max escapes per CTA). Raises if
    an escape block overflows or the host round trip is not bit-exact."""
    n, k = w.shape
    r = rows
    ctas, ckk = n // r, 8192 // r
    nch, ns, mt = k // ckk, ckk // 16, r // 16
    dev = w.device
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, lo = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    base = int(hist.unfold(0, 8, 1).sum(1).argmax())   # the 8 consecutive e71 values holding the most elements
    if not 1 <= base <= 120:
        raise RuntimeError(f"base {base} out of range")
    esc = (e71 < base) | (e71 > base + 7)
    code = torch.where(esc, torch.zeros_like(e71), (sign << 3) | (e71 - base))
    lo_enc = torch.where(esc, torch.zeros_like(lo), lo)
    # host round trip of the arithmetic the kernel does (escapes decode to base << 8, then are patched)
    dec = ((code + base + 120 * (code >> 3)) << 8) | lo_enc
    if not bool((dec[esc] == (base << 8)).all()):
        raise RuntimeError("escape slots do not decode to w_floor")
    dec = torch.where(esc, bits, dec)
    if not torch.equal(dec, bits):
        raise RuntimeError("host round trip is not bit-exact")
    # mma.sync A-fragment order per m16 x k16 tile: lane l (g = l >> 2, q = l & 3) holds rows g, g+8 and columns
    # 2q, 2q+1, 2q+8, 2q+9 as a0 = (g, 2q..2q+1), a1 = (g+8, 2q..), a2 = (g, 2q+8..), a3 = (g+8, 2q+8..)
    ln = torch.arange(32, device=dev)
    g, q = ln >> 2, ln & 3
    rows_l = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    cols_l = torch.stack([2 * q, 2 * q + 1, 2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9, 2 * q + 8, 2 * q + 9], 1)

    def frag(t):   # [N, K] -> [ctas, nch, ns, mt, 32 lanes, 8 elements]
        t6 = t.view(ctas, mt, 16, nch, ns, 16).permute(0, 3, 4, 1, 2, 5)
        return t6[..., rows_l, cols_l]

    lf = frag(lo_enc).to(torch.uint8).reshape(ctas, nch, ns, mt, 256)
    cf = frag(code)
    cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).to(torch.uint8).reshape(ctas, nch, ns, mt, 128)
    wc = torch.cat([lf, cwd], -1).reshape(ctas, nch, ns * mt * 384).contiguous()
    # escape blocks: uint16 row offsets [R + 1] (bytes 0..255), then (row << 16 | k, bits) entries from byte 256
    idx = esc.nonzero()   # row-major: sorted by (row, k)
    n_esc = int(idx.shape[0])
    blocks = torch.zeros((ctas, B12_ESC_BYTES), dtype=torch.uint8)
    per_cta = torch.bincount((idx[:, 0] // r).cpu(), minlength=ctas) if n_esc else torch.zeros(ctas, dtype=torch.int64)
    max_cta = int(per_cta.max()) if ctas else 0
    if max_cta > B12_ESC_CAP:
        raise RuntimeError(f"escape block overflow: {max_cta} > {B12_ESC_CAP} in one CTA")
    if n_esc:
        rr, kk = idx[:, 0].cpu(), idx[:, 1].cpu()
        vals = bits[idx[:, 0], idx[:, 1]].cpu()
        cta_of, rloc = rr // r, rr % r
        starts = torch.cumsum(per_cta, 0) - per_cta
        for c in torch.nonzero(per_cta).flatten().tolist():
            s0, cnt = int(starts[c]), int(per_cta[c])
            rl = rloc[s0:s0 + cnt]
            roff = torch.zeros(r + 1, dtype=torch.int64)
            roff[1:] = torch.cumsum(torch.bincount(rl, minlength=r), 0)
            blocks[c, :2 * (r + 1)] = roff.to(torch.int16).view(torch.uint8)   # little-endian uint16
            ent = torch.stack([(rl << 16) | kk[s0:s0 + cnt], vals[s0:s0 + cnt].to(torch.int64)], 1).to(torch.int32)
            blocks[c, 256:256 + 8 * cnt] = ent.reshape(-1).view(torch.uint8)
    esc_dev = blocks.reshape(-1).to(dev).contiguous()
    return wc, esc_dev, base, n_esc, max_cta


def _b12_route_ok(layer: str, x: torch.Tensor, cfg, state: dict) -> bool:
    """bulk12 cell: layer in B12_CELLS with its prepare()-set flag, cfg m == 1 and the cell's n / k, x a contiguous
    16-byte-aligned bf16 [1, k]. cfg, shapes and state flags only."""
    spec = B12_CELLS.get(layer)
    if spec is None or not state.get(spec[5], False):
        return False
    n, k = spec[1], spec[2]
    return (cfg["m"] == B12_M and cfg["n"] == n and cfg["k"] == k
            and x.dim() == 2 and x.shape[0] == B12_M and x.shape[1] == k
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0)


def _b12_forward(layer: str, x: torch.Tensor, state: dict) -> torch.Tensor:
    """ONE launch: out[1, n] = x @ W.T from the state's 12-bit weight, whole K per CTA, bf16 stored once."""
    out = torch.empty((1, B12_CELLS[layer][1]), dtype=torch.bfloat16, device=x.device)
    _B12_EXT.b12_run(state["b12_wc_" + layer], state["b12_esc_" + layer], x, out, state["b12_err"],
                     state["b12_inst_" + layer], state["b12_base_" + layer])
    return out


def _b12_encode_checked(layer: str, w: torch.Tensor, rows: int, inst: int, verbose: bool):
    """Encode w at R = rows and check the round trip on the device through the kernel's own decode (b12_unpack with
    an instance of that R): every non-escape element bit-exact, every escape slot at w_floor's bits. Returns
    (wc, esc, base). Raises on any mismatch."""
    if B12_INST_R[inst] != rows:
        raise RuntimeError(f"instance {inst} is R={B12_INST_R[inst]}, encoding is R={rows}")
    wc, esc, base, n_esc, max_cta = _b12_encode(w, rows)
    dec = torch.empty(tuple(w.shape), dtype=torch.int16, device=w.device)
    _B12_EXT.b12_unpack(wc, dec, inst, base)
    torch.cuda.synchronize()
    bits = w.contiguous().view(torch.int16)
    hi = (bits.to(torch.int32) >> 8) & 0x7F
    escm = (hi < base) | (hi > base + 7)
    if not torch.equal(dec[~escm], bits[~escm]):
        raise RuntimeError("device round trip (b12_unpack) differs from the weight")
    if not bool((dec[escm].to(torch.int32) == (base << 8)).all()):
        raise RuntimeError("device escape slots do not decode to w_floor")
    if verbose:
        el = w.numel()
        print(f"[b12] {layer} R={rows}: base {base} (e71 window {base}..{base + 7}, binades {2 * base - 127}.."
              f"{2 * base + 15 - 127}), escapes {n_esc} ({n_esc / el:.2e} of {el}), max {max_cta} per CTA (cap "
              f"{B12_ESC_CAP}); stream {wc.numel() / 1e6:.2f} MB + escape blocks {esc.numel() / 1e6:.3f} MB vs "
              f"raw {el * 2 / 1e6:.2f} MB (x{(wc.numel() + esc.numel()) / (el * 2):.4f}); host + device round "
              f"trips bit-exact", flush=True)
    return wc, esc, base


def _b12_load(state: dict, layer: str, enc, inst: int) -> None:
    """Put one encoding (wc, esc, base) and the instance that runs it into the state's buffers for this cell."""
    wc, esc, base = enc
    state["b12_wc_" + layer], state["b12_esc_" + layer] = wc, esc
    state["b12_base_" + layer], state["b12_inst_" + layer] = base, inst


def _b12_encode_into(state: dict, layer: str, w: torch.Tensor, inst: int, verbose: bool) -> None:
    """Encode w at the instance's own R (B12_INST_R), check both round trips, load it into the state's buffers."""
    _b12_load(state, layer, _b12_encode_checked(layer, w, B12_INST_R[inst], inst, verbose), inst)


def _b12_prepare(state: dict, flags=None) -> None:
    """A12 / A2_12 / A3_12. Sets state[flag] (False unless every step passes) and, per routed cell, the state's own
    encoded weight; one per-state error word. flags None (the judge's state): per cell and instance query attributes
    + occupancy (needs MINB CTAs/SM), encode once per distinct R + host / device round trips, the variant's buffers
    loaded into state, canary + correctness, model-selector gates against kernel_fn with the flag off (today's route,
    the raw bulk cell included) under a READ flush (routing) and a write flush (diagnostic; a challenger-R variant
    must pass both); _b12_pick chooses, the winner routes at read ratio <= _b12_max_ratio(m) after a torch.profiler
    kernel-name assertion and only its buffers stay; one compact gate line per variant and one PATH CHOSEN line per
    cell (both re-printed at the end of prepare). flags a dict (the A/B check's second state): inherit each flag and
    instance, encode at that instance's R (B12_INST_R) only for cells that are on."""
    for spec in B12_CELLS.values():
        state[spec[5]] = False
    if _B12_EXT is None:
        if flags is None:
            for layer, spec in B12_CELLS.items():
                _decision(f"PATH CHOSEN {spec[0]} {layer} m={B12_M}: incumbent (cand n/a us vs incumbent n/a us; "
                      f"no b12 module)")
        return
    dev = state["router"].device
    state["b12_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        insts = flags.get("_b12_inst", {})
        for layer, spec in B12_CELLS.items():
            if flags.get(spec[5]) and insts.get(layer) is not None:
                try:   # a failure here turns off only this cell in this state, never the whole A/B
                    _b12_encode_into(state, layer, state[layer], insts[layer], verbose=False)
                    state[spec[5]] = True
                except Exception as e:
                    print(f"[b12] second state: {spec[0]} {layer} encode raised {type(e).__name__}: {e}; "
                          f"cell off in this state", flush=True)
        return
    for layer, spec in B12_CELLS.items():
        src, n, k, home_r, variants, flag = spec
        maxr = _b12_max_ratio(B12_M)
        t0 = time.perf_counter()
        results = []   # one dict per gated variant: inst, R, label, read / write (cand_us, incumbent_us), rr, wr
        choice = None
        why = ""
        try:
            w = state[layer]
            if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16 and w.is_contiguous()):
                why = f"weight {tuple(w.shape)} {w.dtype} unexpected"
            else:
                st_on = dict(state)
                st_on[flag] = True
                encs = {}   # R -> (wc, esc, base), or None if that R's encoding failed; one encoding per distinct R
                faulted = False
                for inst, stages, minb, cw, rows in variants:
                    label = f"b12_kernel<{rows},{k},{stages},{minb},{cw}>"
                    info = [int(v) for v in _B12_EXT.b12_setup(inst)]
                    print(f"[b12] {src} {layer} {label} (inst {inst}): regs={info[0]} local(spill)={info[1]} B static "
                          f"smem={info[2]} B dyn smem={info[3]} B, occupancy {info[4]} CTA/SM (needs {minb}), grid "
                          f"{n // rows} x {32 * (cw + 1)} threads", flush=True)
                    if info[4] < minb:
                        why = f"{label}: occupancy {info[4]} CTA/SM < {minb}"
                        _decision(f"[b12] gate {src} {layer} m={B12_M} {label}: skipped, {why}")
                        continue
                    if rows not in encs:
                        try:
                            encs[rows] = _b12_encode_checked(layer, w, rows, inst, verbose=True)
                        except Exception as e:   # this R only: the other R's variants still run
                            encs[rows] = None
                            print(f"[b12] {src} {layer} R={rows}: encode raised {type(e).__name__}: "
                                  f"{str(e)[-300:]}", flush=True)
                    if encs[rows] is None:
                        why = f"{label}: R={rows} encoding failed"
                        _decision(f"[b12] gate {src} {layer} m={B12_M} {label}: skipped, {why}")
                        continue
                    # The variant's own buffers go into the state before its canary, check and gates.
                    _b12_load(state, layer, encs[rows], inst)
                    st_on.update({key: state[key] for key in ("b12_wc_" + layer, "b12_esc_" + layer,
                                                              "b12_base_" + layer, "b12_inst_" + layer, "b12_err")})
                    bad = _cell_check(f"{src} {layer} {label} inst {inst}", lambda xx: _b12_forward(layer, xx, st_on),
                                      w, B12_M, n, k, err=state["b12_err"])
                    if bad:
                        why = f"{label}: {bad}"
                        _decision(f"[b12] gate {src} {layer} m={B12_M} {label}: canary/check failed, {bad}")
                        continue
                    res = {"inst": inst, "R": rows, "label": label}
                    tg = time.perf_counter()
                    for mode in (B12_GATE_FLUSH, "write"):
                        res[mode] = _cell_gate(f"{src} {label}", layer, B12_M, n, k, flag, state, flush_mode=mode)
                    code = int(state["b12_err"].item())
                    res["rr"] = res["read"][0] / res["read"][1]
                    res["wr"] = res["write"][0] / res["write"][1]
                    c_us = res["read"][0]
                    print(f"[prepare-time] gates {src} {label}: {time.perf_counter() - tg:.2f}s; read-flush cand "
                          f"{c_us:.2f} us (raw-equivalent {n * k * 2 / (c_us * 1e-6) / 1e12:.2f} TB/s, stream "
                          f"{n * k * 1.5 / (c_us * 1e-6) / 1e12:.2f} TB/s) vs incumbent {res['read'][1]:.2f} us",
                          flush=True)
                    _decision(f"[b12] gate {src} {layer} m={B12_M} {label} R={rows} grid {n // rows}: read "
                              f"{res['rr']:.3f} ({res['read'][0]:.2f} / {res['read'][1]:.2f} us) write {res['wr']:.3f} "
                              f"({res['write'][0]:.2f} / {res['write'][1]:.2f} us)"
                              + (f"; error word {code}" if code else ""))
                    if code:   # a fault in any instance keeps the incumbent for the whole cell
                        why = f"{label}: error word {code} after the gates"
                        faulted = True
                        break
                    results.append(res)
                if not faulted and results:
                    choice, why = _b12_pick(src, layer, results, home_r, maxr)
                if choice is not None:
                    _b12_load(state, layer, encs[choice["R"]], choice["inst"])
                    st_on.update({key: state[key] for key in ("b12_wc_" + layer, "b12_esc_" + layer,
                                                              "b12_base_" + layer, "b12_inst_" + layer)})
                    data = {"x": torch.randn((B12_M, k), device=dev, dtype=torch.float32).to(torch.bfloat16),
                            "config": {"m": B12_M, "n": n, "k": k, "layer": layer}}
                    try:
                        names = _cell_kernels(lambda: kernel_fn(data, st_on))
                        print(f"[b12] {src} {layer} m={B12_M}: profiled kernels {names}", flush=True)
                        if names and not (len(names) == 1 and "b12_kernel" in names[0]):
                            why = f"profiled kernels {names}: expected exactly one b12_kernel"
                    except Exception as e:  # own extension, direct launch: keep the route
                        print(f"[b12] {src} {layer}: profiler unavailable ({type(e).__name__}: {e}); route kept "
                              f"(kernel_fn calls b12_run directly)", flush=True)
                    if not why:
                        state[flag] = True
                encs.clear()   # only the winner's buffers stay (in state)
        except Exception as e:  # never raises: any surprise keeps the incumbent path for this cell
            why = f"raised {type(e).__name__}: {str(e)[-300:]}"
            state[flag] = False
        if not state[flag]:
            for key in ("b12_wc_", "b12_esc_", "b12_base_", "b12_inst_"):
                state.pop(key + layer, None)
        state["b12_err"].zero_()
        shown = choice if choice is not None else (min(results, key=lambda r: r["rr"]) if results else None)
        if shown is None:
            r_txt = "read n/a, write n/a"
        else:
            r_txt = (f"{shown['label']} read {shown['rr']:.3f} ({shown['read'][0]:.2f} vs {shown['read'][1]:.2f} us), "
                     f"write {shown['wr']:.3f} ({shown['write'][0]:.2f} vs {shown['write'][1]:.2f} us)")
        print(f"[prepare-time] cell {src} {layer} m={B12_M} total: {time.perf_counter() - t0:.2f}s", flush=True)
        _decision(f"PATH CHOSEN {src} {layer} m={B12_M}: "
                  + (f"candidate {choice['label']} R={choice['R']} (inst {choice['inst']})" if state[flag]
                     else "incumbent")
                  + f" [{'chosen' if state[flag] else 'best-read'} {r_txt}; route at read <= {maxr}]"
                  + ("" if state[flag] else f"; {why}"))
    torch.cuda.synchronize()


def _b12_pick(src: str, layer: str, results: list, home_r: int, maxr: float):
    """Routing rule over the gated variants of one cell (prepare()-only). Returns (choice or None, why).
    Home-R variants: the best read-flush ratio routes if it is <= maxr. A challenger (R != the cell's home R) replaces
    it only if it passes BOTH gates (read and write ratio <= maxr) and its read ratio is at least
    B12_CHALLENGER_MARGIN below the best home-R variant's read ratio; with no gated home-R variant, no challenger
    routes (the incumbent stays)."""
    home = [r for r in results if r["R"] == home_r]
    best_home = min(home, key=lambda r: r["rr"]) if home else None
    choice, why = None, ""
    if best_home is None:
        why = f"no home-R (R={home_r}) variant gated"
    elif best_home["rr"] <= maxr:
        choice = best_home
    else:
        why = f"gate: best home-R {best_home['label']} read-flush cand/incumbent {best_home['rr']:.3f} > {maxr}"
    for r in results:
        if r["R"] == home_r:
            continue
        if best_home is None:
            verdict = "not eligible (no home-R variant to compare)"
        elif not r["rr"] <= maxr:
            verdict = f"fails read gate {r['rr']:.3f} > {maxr}"
        elif not r["wr"] <= maxr:
            verdict = f"fails write gate {r['wr']:.3f} > {maxr}"
        elif not r["rr"] <= best_home["rr"] - B12_CHALLENGER_MARGIN:
            verdict = (f"read {r['rr']:.3f} not >= {B12_CHALLENGER_MARGIN} below home-R {best_home['label']} "
                       f"read {best_home['rr']:.3f}")
        else:
            verdict = f"eligible (read {r['rr']:.3f} vs home-R {best_home['rr']:.3f})"
            if choice is None or choice["R"] == home_r or r["rr"] < choice["rr"]:
                choice, why = r, ""
        _decision(f"[b12] challenger {src} {layer} {r['label']} R={r['R']}: {verdict}")
    return choice, why


NEW_CELL_FLAGS = NEW_CELL_FLAGS + B12_FLAGS   # section (10)'s cells join the A/B check and the FINAL FLAGS line


# --- (12) m == 8 bulk12 on TMA boxes with a batched escape gather (hypothesis b12-batched-escape-m8-32-race) -----
# Section (10)'s 12-bit row-owner stream at exactly m == 8 (pooler, qkv_gate), from the H100 lane (takemyjob
# hopper/b12tma-1029 d95798e, hopper/b12dens-1029 6d80717; research/h100-b12tma-1029.md, h100-b12dens-1029.md):
#   * uc1_g1 (b12m_tma_kernel): x as 2D tensor-map boxes (64 k x NT token rows, SWIZZLE_128B) into the same ring slot
#     as the chunk's 12 KB of codes, NO cluster, one local mbarrier arrive per consumer warp (no arrive.release.cluster
#     / MEMBAR.ALL.GPU: the b12mc defect), and the escapes' x gathered with eight independent loads per batch (GA1)
#     instead of the judged serial per-element loads (GA0). The tensor map is encoded on the host from THIS call's
#     x.data_ptr() inside b12m_run on every call (a host-only cuTensorMapEncodeTiled, capture-safe, no sync), passed
#     by value as a __grid_constant__ parameter, never cached. The H100 pre-check at the judge's real escape density
#     projected pooler m8 0.939 and qkv m8 0.906 of master's incumbent (pooler m16 1.03: diag only here).
#   * bridge (b12m_br_kernel, a7da6fa9 / 2980f57a's rotated b12x, x as 16 B cp.async pieces): NON-routable diag arms
#     GA0 (the judged code) and GA1 (batched gather) at pooler m8 / qkv m8; GA0 - GA1 is the gather's cost on the judge.
# Codes: master's m=1 encoding is reused when the state holds it at the same R (pooler R=32, qkv_gate R=16), else the
# cell encodes its own through _b12_encode_checked (host + device round trip, bit-exact) under b12m_* state keys.
# Routing (pooler m8, qkv_gate m8 only, exact shape: cfg m == 8 and x.shape[0] == 8): canary + fp64 rel L2 + probe row
# + per-row bound per arm (_cell_check + _b12m_rows), B12M_ROUNDS read-flush _cell_gate rounds of kernel_fn with the
# cell flag on vs off (master's incumbent path, like for like; the diag arms ride along as extra arms), one write-flush
# round beside them (diagnostic); route only if the pooled read-flush median cand / incumbent <= B12_MAX_RATIO_M8, every
# round <= B12M_ROUND_MAX, pooled cand IQR <= B12M_POOLED_IQR_MAX x max(pooled incumbent IQR, floor), then a
# torch.profiler assertion of exactly one b12m_tma_kernel. Own extension (B12M_EXT_NAME): the m=1 unit is untouched.
# The error word is prepare()-only (kernel_fn passes it, never reads it). A <= 2 KB [b12m] summary prints last.
B12M_EXT_NAME = "muse_bf16_b12m_m8_tma_v1"
B12M_ROUNDS = 3                 # read-flush _cell_gate rounds (routing)
B12M_ROUND_MAX = 0.97           # every round's cand / incumbent must be <= this
B12M_POOLED_IQR_MAX = 2.0       # pooled cand IQR <= this x max(pooled incumbent IQR, floor) (the khalf POOLED rule)
B12M_IQR_FLOOR_US = 0.5
B12M_ROW_REL = 5 * CELL_REL_L2  # any single row's rel L2 vs fp64 (the judge allows 5x its 1% bound)
B12M_START_DEADLINE_S = 150.0   # no b12m cell later than this after state A's build began (prepare has 300 s)
B12M_SUMMARY_BYTES = 2000
# arm id (the CUDA source's instance list) -> (name, kind, layer, R, NT, S, MINB, CW, GA)
B12M_ARMS = {
    0: ("uc1_g1", "tma", "pooler", 32, 8, 6, 2, 8, 1),
    1: ("uc1_g1", "tma", "qkv_gate", 16, 8, 4, 2, 4, 1),
    2: ("uc1_s8_g1", "tma", "pooler", 32, 16, 8, 1, 8, 1),
    3: ("br_GA0", "br", "pooler", 32, 8, 8, 1, 8, 0),
    4: ("br_GA1", "br", "pooler", 32, 8, 8, 1, 8, 1),
    5: ("br_GA0", "br", "qkv_gate", 16, 8, 4, 2, 4, 0),
    6: ("br_GA1", "br", "qkv_gate", 16, 8, 4, 2, 4, 1),
}
# (layer, m) -> (tag, n, k, R, candidate arm, {diag label: arm}, state flag, routable)
B12M_CELLS = {
    ("pooler", 8): ("P8", 4096, 8192, 32, 0, {"GA0": 3, "GA1": 4}, "b12m_pooler8_ok", True),
    ("qkv_gate", 8): ("Q8", 2304, 8192, 16, 1, {"GA0": 5, "GA1": 6}, "b12m_qkv8_ok", True),
    ("pooler", 16): ("P16", 4096, 8192, 32, 2, {}, "b12m_pooler16_diag", False),   # timed, never routed
}
B12M_FLAGS = tuple(spec[6] for spec in B12M_CELLS.values() if spec[7])
B12M_HOME_INST = {32: 0, 16: 2}   # an m=1 b12 instance of each R: its b12_unpack is the device round trip
for _cell, _spec in B12M_CELLS.items():
    for _a in (_spec[4],) + tuple(_spec[5].values()):
        assert B12M_ARMS[_a][2] == _cell[0] and B12M_ARMS[_a][3] == _spec[3] and B12M_ARMS[_a][4] == _cell[1], _cell
del _cell, _spec, _a
_B12M_SUM = []                    # [b12m] summary lines (state A's prepare), printed last by prepare()
_B12M_BUILD = {"s": None, "ok": False}

_B12M_CPP = r"""
#include <torch/extension.h>
#include <vector>
void b12m_run(torch::Tensor wc, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor err,
              int64_t arm, int64_t base);
std::vector<int64_t> b12m_setup(int64_t arm);
"""

_B12M_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#define B12M_WAIT_NS 500000000ULL   // every mbarrier wait is bounded to 0.5 s, then an error word
#define B12M_ESC_BYTES 2048         // per-CTA escape block (B12_ESC_BYTES in Python)
#define B12M_TILE 384               // one m16 x k16 tile: 32 lanes x (8 L bytes + 4 code bytes)

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Bounded wait (CTA scope): false after B12M_WAIT_NS (the caller sets the error word and leaves its loop).
__device__ __forceinline__ bool mbar_wait(uint32_t a, uint32_t parity) {
  unsigned long long t0 = 0;
  for (uint32_t it = 0;; ++it) {
    uint32_t done;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) return true;
    if (it == 0) t0 = gtimer();
    else if (gtimer() - t0 > B12M_WAIT_NS) return false;
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// One 2D tensor-map box into this CTA's smem (c0 = k, c1 = token row), bytes counted on bar. No cluster launch: the
// CTA is its own 1-CTA cluster, so its shared::cta address is a valid shared::cluster address.
__device__ __forceinline__ void tma_2d(uint32_t dst, uint64_t map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void lds64(uint32_t& v0, uint32_t& v1, uint32_t a) {
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v0), "=r"(v1) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
// Bridge only: 16 B global -> shared through the LSU, completion tracked by cp_async_arrive_noinc.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Section (10)'s decode, unchanged: four high bytes from four codes, then one lane's m16n8k16 A fragment.
__device__ __forceinline__ uint32_t b12_hi4(uint32_t c4, uint32_t base4) {
  const uint32_t t = (c4 >> 3) & 0x01010101u;
  return t * 0x78u + (c4 + base4);
}
__device__ __forceinline__ void b12_frag(uint32_t l0, uint32_t l1, uint32_t cw, uint32_t base4,
                                         uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  const uint32_t h0 = b12_hi4(cw & 0x0F0F0F0Fu, base4);
  const uint32_t h1 = b12_hi4((cw >> 4) & 0x0F0F0F0Fu, base4);
  a0 = __byte_perm(l0, h0, 0x5140);
  a1 = __byte_perm(l0, h0, 0x7362);
  a2 = __byte_perm(l1, h1, 0x5140);
  a3 = __byte_perm(l1, h1, 0x7362);
}

// smem below the ring: barriers (256 B), the escape block, then EX = fp32 x values of the escapes, token-major with an
// odd stride (ESC_CAP + 1). The TMA kernel rounds the ring up to 1024 B (SWIZZLE_128B boxes); the bridge keeps 128 B.
constexpr int b12m_esx() { return (B12M_ESC_BYTES - 256) / 8 + 1; }
constexpr int b12m_ring_tma(int nt) { return (256 + B12M_ESC_BYTES + b12m_esx() * nt * 4 + 1023) / 1024 * 1024; }
constexpr int b12m_ring_br(int nt) { return (256 + B12M_ESC_BYTES + b12m_esx() * nt * 4 + 127) / 128 * 128; }

// Escape-x gather by the producer warp: exs[t][e] = x[t][k_e] for this CTA's escapes. BATCH=false is the judged b12x
// code (GA0: per-element branches expose every load one after another); BATCH=true (GA1) computes eight clamped
// offsets, issues eight independent loads, then stores.
template <int R, int K, bool BATCH>
__device__ __forceinline__ void esc_gather(const unsigned char* smem, float* exs, const __nv_bfloat16* __restrict__ x,
                                           int m, int lane) {
  constexpr int ESX = b12m_esx();
  const uint16_t* roff = reinterpret_cast<const uint16_t*>(smem + 256);
  const uint32_t* ent = reinterpret_cast<const uint32_t*>(smem + 256 + 256);
  const int total = (int)roff[R] * m;
  if constexpr (!BATCH) {
    for (int p0 = 0; p0 < total; p0 += 32 * 8) {
      float v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int p = p0 + u * 32 + lane;
        v[u] = 0.f;
        if (p < total) {
          const int e = p / m, t = p - (p / m) * m;
          v[u] = __bfloat162float(x[(size_t)t * K + (ent[2 * e] & 0xFFFFu)]);
        }
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int p = p0 + u * 32 + lane;
        if (p < total) exs[(p - (p / m) * m) * ESX + p / m] = v[u];
      }
    }
  } else {
    const unsigned short* xs = reinterpret_cast<const unsigned short*>(x);
    for (int p0 = 0; p0 < total; p0 += 32 * 8) {
      int off[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int p = p0 + u * 32 + lane;
        const int pc = p < total ? p : 0;
        const int e = pc / m, t = pc - e * m;
        off[u] = t * K + (int)(ent[2 * e] & 0xFFFFu);
      }
      unsigned short v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = __ldg(xs + off[u]);
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int p = p0 + u * 32 + lane;
        if (p < total) exs[(p - (p / m) * m) * ESX + p / m] = __uint_as_float((uint32_t)v[u] << 16);
      }
    }
  }
}

// Shared epilogue: consumer warps park their fp32 accumulators in the (drained) ring, then every thread sums the CW
// partials per (token, row) in a fixed order and applies the row's escapes in k order (+ x * w, - x * w_floor).
template <int R, int CW, int NT, int MT, int NN, int RING>
__device__ __forceinline__ void b12m_epilogue(unsigned char* smem, const float (&acc)[MT][NN][4], const float* exs,
                                              __nv_bfloat16* __restrict__ out, uint32_t base, int m, int tid,
                                              int warp, int lane) {
  constexpr int ESX = b12m_esx();
  constexpr int RP = R + 4;
  float* scratch = reinterpret_cast<float*>(smem + RING);
  if (warp >= 1) {
    const int cw = warp - 1;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nn = 0; nn < NN; ++nn)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int row = mt * 16 + (lane >> 2) + (c >> 1) * 8;
          const int tok = nn * 8 + (lane & 3) * 2 + (c & 1);
          scratch[(cw * NT + tok) * RP + row] = acc[mt][nn][c];
        }
  }
  __syncthreads();
  const int N = gridDim.x * R;
  const uint16_t* roff = reinterpret_cast<const uint16_t*>(smem + 256);
  const uint32_t* ent = reinterpret_cast<const uint32_t*>(smem + 256 + 256);
  const float wfloor = __uint_as_float(base << 24);
  for (int idx = tid; idx < R * m; idx += 32 * (CW + 1)) {
    const int tok = idx / R, row = idx % R;
    float sum = scratch[tok * RP + row];
#pragma unroll
    for (int c = 1; c < CW; ++c) sum += scratch[(c * NT + tok) * RP + row];
    const int e1 = (int)roff[row + 1];
    for (int e = (int)roff[row]; e < e1; ++e) {
      const float xv = exs[tok * ESX + e];
      sum = fmaf(xv, __uint_as_float(ent[2 * e + 1] << 16), sum);
      sum = fmaf(-xv, wfloor, sum);
    }
    out[(size_t)tok * N + (size_t)blockIdx.x * R + row] = __float2bfloat16_rn(sum);
  }
}

// uc1 (hopper/b12tma-1029 d95798e b12t_kernel at CL=1): slot s = [WB code bytes][NBOX boxes of NT rows x 128 B,
// SWIZZLE_128B: 16 B chunk c of token row t at c ^ (t & 7)]. full_s: count 1 (the producer's arrive.expect_tx of the
// whole slot); empty_s: count CW, one LOCAL arrive per consumer warp. Chunk rotation jc = (j + blockIdx.x / 4) % NCHUNK.
// m == NT (the tensor map's rows). err: 1 producer wait timeout, 2 consumer wait timeout, 3 built without sm_90,
// 4 dynamic smem not 1024 B aligned.
template <int R, int K, int S, int MINB, int CW, int NT, int GA>
__global__ void __launch_bounds__(32 * (CW + 1), MINB)
b12m_tma_kernel(const __grid_constant__ CUtensorMap tmap, const uint8_t* __restrict__ wc,
                const uint8_t* __restrict__ esc, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                int* err, uint32_t base, int m) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int NN = NT / 8;
  constexpr int CKK = 8192 / R;
  constexpr int NCHUNK = K / CKK;
  constexpr int NS = CKK / 16;
  constexpr int KSW = NS / CW;
  constexpr int MT = R / 16;
  constexpr int WB = NS * MT * B12M_TILE;
  constexpr int NBOX = CKK / 64;
  constexpr int BOXB = NT * 128;
  constexpr int SLOT = WB + NBOX * BOXB;
  constexpr int EXOFF = 256 + B12M_ESC_BYTES;
  constexpr int RING = b12m_ring_tma(NT);
  static_assert(K % CKK == 0 && NS % CW == 0 && R % 16 == 0 && WB == 12288 && CKK % 64 == 0, "geometry");
  static_assert(NBOX * BOXB == NT * CKK * 2, "x bytes per chunk");
  static_assert(S >= 2 && S <= 8, "full / empty barriers live in 64 B each");
  static_assert(RING % 1024 == 0 && SLOT % 1024 == 0 && WB % 1024 == 0 && BOXB % 1024 == 0, "SWIZZLE_128B align");
  static_assert(CW * NT * (R + 4) * 4 <= S * SLOT, "epilogue scratch must fit in the ring");
  static_assert(2 * (R + 1) <= 256, "row offsets must fit the escape header");
  static_assert(NT == 8 || NT == 16, "NT");
  static_assert(EXOFF + b12m_esx() * NT * 4 <= RING, "EX below the ring");
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (sbase & 1023u) {
    if (tid == 0) atomicExch(err, 4);
    return;
  }
  const uint32_t full0 = sbase, empty0 = sbase + 64, xbar = sbase + 128;
  const uint32_t escs = sbase + 256, ring = sbase + RING;
  const int cid = (int)blockIdx.x / 4;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, CW);
    }
    mbar_init(xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float acc[MT][NN][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NN; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  float* exs = reinterpret_cast<float*>(smem + EXOFF);
  if (warp == 0) {
    // Producer warp: lane 0 streams every chunk (codes + x boxes), then the warp gathers the escapes' x (GA1 batched).
    if (lane == 0) {
      mbar_expect_tx(xbar, (uint32_t)B12M_ESC_BYTES);
      bulk_g2s(escs, esc + (size_t)blockIdx.x * B12M_ESC_BYTES, B12M_ESC_BYTES, xbar);
      const uint64_t map = reinterpret_cast<uint64_t>(&tmap);
      for (int j = 0; j < NCHUNK; ++j) {
        const int s = j % S;
        const uint32_t eph = ((j / S) & 1) ^ 1;
        const int jc = (j + cid) % NCHUNK;
        if (!mbar_wait(empty0 + 8 * s, eph)) {
          atomicExch(err, 1);
          break;
        }
        const uint32_t fb = full0 + 8 * s;
        const uint32_t slot = ring + (uint32_t)(s * SLOT);
        mbar_expect_tx(fb, (uint32_t)SLOT);
        bulk_g2s(slot, wc + ((size_t)blockIdx.x * NCHUNK + jc) * (size_t)WB, (uint32_t)WB, fb);
#pragma unroll
        for (int b = 0; b < NBOX; ++b) tma_2d(slot + (uint32_t)(WB + b * BOXB), map, jc * CKK + 64 * b, 0, fb);
      }
    }
    __syncwarp();
    if (mbar_wait(xbar, 0)) esc_gather<R, K, GA != 0>(smem, exs, x, m, lane);
    else if (lane == 0) atomicExch(err, 1);
    __syncwarp();
  } else {
    const int cw = warp - 1;
    const uint32_t base4 = base * 0x01010101u;
    for (int j = 0; j < NCHUNK; ++j) {
      const int s = j % S;
      const uint32_t ph = (j / S) & 1;
      if (!mbar_wait(full0 + 8 * s, ph)) {
        if (lane == 0) atomicExch(err, 2);
        break;
      }
      const uint32_t slot = ring + (uint32_t)(s * SLOT);
      const uint32_t xslot = slot + WB;
#pragma unroll
      for (int i = 0; i < KSW; ++i) {
        const int kk16 = cw * KSW + i;
        const uint32_t xb = xslot + (uint32_t)((kk16 >> 2) * BOXB);
        const int c0 = (kk16 & 3) * 2;
        uint32_t bfr[NN][2];
        if constexpr (NN == 1) {
          const int tok = lane & 7, kh = (lane >> 3) & 1;
          ldsm_x2(bfr[0][0], bfr[0][1], xb + (uint32_t)(tok * 128 + (((c0 + kh) ^ (tok & 7)) << 4)));
        } else {
#pragma unroll
          for (int p = 0; p < NN / 2; ++p) {
            const int mi = lane >> 3;
            const int tok = (2 * p + (mi >> 1)) * 8 + (lane & 7), kh = mi & 1;
            ldsm_x4(bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1],
                    xb + (uint32_t)(tok * 128 + (((c0 + kh) ^ (tok & 7)) << 4)));
          }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint32_t t = slot + (uint32_t)((kk16 * MT + mt) * B12M_TILE);
          uint32_t l0, l1, a0, a1, a2, a3;
          lds64(l0, l1, t + (uint32_t)lane * 8);
          const uint32_t c = lds32(t + 256 + (uint32_t)lane * 4);
          b12_frag(l0, l1, c, base4, a0, a1, a2, a3);
#pragma unroll
          for (int nn = 0; nn < NN; ++nn) mma_bf16(acc[mt][nn], a0, a1, a2, a3, bfr[nn][0], bfr[nn][1]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
  }
  __syncthreads();
  b12m_epilogue<R, CW, NT, MT, NN, RING>(smem, acc, exs, out, base, m, tid, warp, lane);
#else
  if (threadIdx.x == 0) atomicExch(err, 3);
#endif
}

// Bridge (a7da6fa9 / 2980f57a rotated b12x: x as 16 B cp.async pieces, jc = (j + blockIdx.x) % NCHUNK, no cluster).
// GA0 is the judged code; GA1 swaps in the batched gather. Diag arms only. 2 <= m <= NT.
template <int R, int K, int S, int MINB, int CW, int NT, int GA>
__global__ void __launch_bounds__(32 * (CW + 1), MINB)
b12m_br_kernel(const uint8_t* __restrict__ wc, const uint8_t* __restrict__ esc, const __nv_bfloat16* __restrict__ x,
               __nv_bfloat16* __restrict__ out, int* err, uint32_t base, int m) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int NN = NT / 8;
  constexpr int CKK = 8192 / R;
  constexpr int NCHUNK = K / CKK;
  constexpr int NS = CKK / 16;
  constexpr int KSW = NS / CW;
  constexpr int MT = R / 16;
  constexpr int WB = NS * MT * B12M_TILE;
  constexpr int XS = CKK * 2 + 16;
  constexpr int SLOT = WB + NT * XS;
  constexpr int EXOFF = 256 + B12M_ESC_BYTES;
  constexpr int RING = b12m_ring_br(NT);
  constexpr int PPR = CKK / 8;
  static_assert(K % CKK == 0 && NS % CW == 0 && R % 16 == 0 && WB == 12288, "geometry");
  static_assert(S >= 2 && S <= 8, "full / empty barriers live in 64 B each");
  static_assert(RING % 128 == 0 && SLOT % 128 == 0 && XS % 16 == 0, "alignment");
  static_assert(CW * NT * (R + 4) * 4 <= S * SLOT, "epilogue scratch must fit in the ring");
  static_assert(2 * (R + 1) <= 256, "row offsets must fit the escape header");
  static_assert(NT == 8 || NT == 16, "NT");
  static_assert(EXOFF + b12m_esx() * NT * 4 <= RING, "EX below the ring");
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  const uint32_t full0 = sbase, empty0 = sbase + 64, xbar = sbase + 128;
  const uint32_t escs = sbase + 256, ring = sbase + RING;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1 + 32);
      mbar_init(empty0 + 8 * s, CW);
    }
    mbar_init(xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float acc[MT][NN][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NN; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  float* exs = reinterpret_cast<float*>(smem + EXOFF);
  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(xbar, (uint32_t)B12M_ESC_BYTES);
      bulk_g2s(escs, esc + (size_t)blockIdx.x * B12M_ESC_BYTES, B12M_ESC_BYTES, xbar);
    }
    for (int j = 0; j < NCHUNK; ++j) {
      const int s = j % S;
      const uint32_t eph = ((j / S) & 1) ^ 1;
      const int jc = (j + (int)blockIdx.x) % NCHUNK;
      int good = 1;
      if (lane == 0) {
        good = mbar_wait(empty0 + 8 * s, eph) ? 1 : 0;
        if (good) mbar_expect_tx(full0 + 8 * s, (uint32_t)WB);
      }
      good = __shfl_sync(0xffffffffu, good, 0);
      __syncwarp();   // every lane's cp.async into the slot after lane 0's empty-barrier wait
      if (!good) {
        if (lane == 0) atomicExch(err, 1);
        break;
      }
      const uint32_t slot = ring + (uint32_t)(s * SLOT);
      if (lane == 0)
        bulk_g2s(slot, wc + ((size_t)blockIdx.x * NCHUNK + jc) * (size_t)WB, (uint32_t)WB, full0 + 8 * s);
      for (int p = lane; p < m * PPR; p += 32) {
        const int t = p / PPR, q = p - (p / PPR) * PPR;
        cp_async16(slot + (uint32_t)(WB + t * XS + q * 16), x + (size_t)t * K + (size_t)jc * CKK + (size_t)q * 8);
      }
      cp_async_arrive_noinc(full0 + 8 * s);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");   // none in flight when the ring becomes epilogue scratch
    __syncwarp();
    if (mbar_wait(xbar, 0)) esc_gather<R, K, GA != 0>(smem, exs, x, m, lane);
    else if (lane == 0) atomicExch(err, 1);
    __syncwarp();
  } else {
    const int cw = warp - 1;
    const uint32_t base4 = base * 0x01010101u;
    for (int j = 0; j < NCHUNK; ++j) {
      const int s = j % S;
      const uint32_t ph = (j / S) & 1;
      if (!mbar_wait(full0 + 8 * s, ph)) {
        if (lane == 0) atomicExch(err, 2);
        break;
      }
      const uint32_t slot = ring + (uint32_t)(s * SLOT);
      const uint32_t xslot = slot + WB;
#pragma unroll
      for (int i = 0; i < KSW; ++i) {
        const int kk16 = cw * KSW + i;
        uint32_t bfr[NN][2];
        if constexpr (NN == 1) {
          ldsm_x2(bfr[0][0], bfr[0][1], xslot + (uint32_t)((lane & 7) * XS + kk16 * 32 + ((lane >> 3) & 1) * 16));
        } else {
#pragma unroll
          for (int p = 0; p < NN / 2; ++p) {
            const int mi = lane >> 3;
            const int tok = (2 * p + (mi >> 1)) * 8 + (lane & 7);
            ldsm_x4(bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1],
                    xslot + (uint32_t)(tok * XS + kk16 * 32 + (mi & 1) * 16));
          }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint32_t t = slot + (uint32_t)((kk16 * MT + mt) * B12M_TILE);
          uint32_t l0, l1, a0, a1, a2, a3;
          lds64(l0, l1, t + (uint32_t)lane * 8);
          const uint32_t c = lds32(t + 256 + (uint32_t)lane * 4);
          b12_frag(l0, l1, c, base4, a0, a1, a2, a3);
#pragma unroll
          for (int nn = 0; nn < NN; ++nn) mma_bf16(acc[mt][nn], a0, a1, a2, a3, bfr[nn][0], bfr[nn][1]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
  }
  __syncthreads();
  b12m_epilogue<R, CW, NT, MT, NN, RING>(smem, acc, exs, out, base, m, tid, warp, lane);
#else
  if (threadIdx.x == 0) atomicExch(err, 3);
#endif
}

typedef CUresult (*B12mEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// The driver's cuTensorMapEncodeTiled through the runtime (no -lcuda). Resolved in b12m_setup (prepare()), before any
// capture; the cached value is a driver function pointer, nothing derived from inputs.
B12mEncodeFn b12m_encode_fn() {
  static B12mEncodeFn cached = nullptr;
  if (cached != nullptr) return cached;
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
  const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
#else
  const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
#endif
  if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || fn == nullptr) {
    (void)cudaGetLastError();
    return nullptr;
  }
  cached = reinterpret_cast<B12mEncodeFn>(fn);
  return cached;
}

// Host-only, every call: a stack-local 2D map over THIS call's x [m, K] bf16 (row stride x.stride(0) * 2 B), box
// {64 k, NT rows}, SWIZZLE_128B, L2 promotion 128 B, no OOB fill; passed by value as the __grid_constant__ parameter.
CUtensorMap b12m_xmap(const torch::Tensor& x, int nt) {
  const B12mEncodeFn enc = b12m_encode_fn();
  TORCH_CHECK(enc != nullptr, "b12m: cuTensorMapEncodeTiled entry point missing");
  TORCH_CHECK(x.stride(1) == 1 && (x.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "b12m: x must be row-contiguous, 16 B aligned, row stride a multiple of 16 B");
  CUtensorMap mp;
  std::memset(&mp, 0, sizeof(mp));
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(x.size(1)), static_cast<cuuint64_t>(x.size(0))};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(x.stride(0)) * 2};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(nt)};
  const cuuint32_t es[2] = {1, 1};
  const CUresult rc = enc(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "b12m: cuTensorMapEncodeTiled rc ", static_cast<int>(rc));
  return mp;
}

template <int R, int K, int S, int MINB, int CW, int NT, int GA>
struct TmaInst {
  static constexpr int SMEM = b12m_ring_tma(NT) + S * (12288 + NT * (8192 / R) * 2);
  static void prep() {
    static bool done = false;
    if (!done) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(b12m_tma_kernel<R, K, S, MINB, CW, NT, GA>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      if constexpr (MINB > 1)
        C10_CUDA_CHECK(cudaFuncSetAttribute(b12m_tma_kernel<R, K, S, MINB, CW, NT, GA>,
                                            cudaFuncAttributePreferredSharedMemoryCarveout,
                                            (int)cudaSharedmemCarveoutMaxShared));
      done = true;
    }
  }
  static void launch(const torch::Tensor& wc, const torch::Tensor& esc, const torch::Tensor& x,
                     const torch::Tensor& out, const torch::Tensor& err, uint32_t base) {
    const int64_t n = out.size(1), m = x.size(0);
    TORCH_CHECK(n % R == 0 && x.size(1) == K && wc.numel() == n * K * 3 / 2, "b12m_run (tma): shape");
    TORCH_CHECK(esc.numel() == (n / R) * B12M_ESC_BYTES, "b12m_run (tma): escape blocks");
    TORCH_CHECK(m == NT && out.size(0) == m, "b12m_run (tma): m must equal NT");
    const CUtensorMap mp = b12m_xmap(x, NT);
    prep();
    b12m_tma_kernel<R, K, S, MINB, CW, NT, GA><<<(unsigned)(n / R), 32 * (CW + 1), SMEM,
                                                 at::cuda::getCurrentCUDAStream()>>>(
        mp, wc.data_ptr<uint8_t>(), esc.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), base, static_cast<int>(m));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  static std::vector<int64_t> setup() {
    TORCH_CHECK(b12m_encode_fn() != nullptr, "b12m: cuTensorMapEncodeTiled entry point missing");
    prep();
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, b12m_tma_kernel<R, K, S, MINB, CW, NT, GA>));
    int occ = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, b12m_tma_kernel<R, K, S, MINB, CW, NT, GA>,
                                                                 32 * (CW + 1), (size_t)SMEM));
    return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)SMEM, (int64_t)occ};
  }
};

template <int R, int K, int S, int MINB, int CW, int NT, int GA>
struct BrInst {
  static constexpr int XS = (8192 / R) * 2 + 16;
  static constexpr int SMEM = b12m_ring_br(NT) + S * (12288 + NT * XS);
  static void prep() {
    static bool done = false;
    if (!done) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(b12m_br_kernel<R, K, S, MINB, CW, NT, GA>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      if constexpr (MINB > 1)
        C10_CUDA_CHECK(cudaFuncSetAttribute(b12m_br_kernel<R, K, S, MINB, CW, NT, GA>,
                                            cudaFuncAttributePreferredSharedMemoryCarveout,
                                            (int)cudaSharedmemCarveoutMaxShared));
      done = true;
    }
  }
  static void launch(const torch::Tensor& wc, const torch::Tensor& esc, const torch::Tensor& x,
                     const torch::Tensor& out, const torch::Tensor& err, uint32_t base) {
    const int64_t n = out.size(1), m = x.size(0);
    TORCH_CHECK(n % R == 0 && x.size(1) == K && wc.numel() == n * K * 3 / 2, "b12m_run (bridge): shape");
    TORCH_CHECK(esc.numel() == (n / R) * B12M_ESC_BYTES, "b12m_run (bridge): escape blocks");
    TORCH_CHECK(m >= 2 && m <= NT && out.size(0) == m && x.is_contiguous(), "b12m_run (bridge): m must be 2..NT");
    prep();
    b12m_br_kernel<R, K, S, MINB, CW, NT, GA><<<(unsigned)(n / R), 32 * (CW + 1), SMEM,
                                                at::cuda::getCurrentCUDAStream()>>>(
        wc.data_ptr<uint8_t>(), esc.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), base, static_cast<int>(m));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  static std::vector<int64_t> setup() {
    prep();
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, b12m_br_kernel<R, K, S, MINB, CW, NT, GA>));
    int occ = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, b12m_br_kernel<R, K, S, MINB, CW, NT, GA>,
                                                                 32 * (CW + 1), (size_t)SMEM));
    return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)SMEM, (int64_t)occ};
  }
};

// Arm ids: B12M_ARMS in Python.
using Arm0 = TmaInst<32, 8192, 6, 2, 8, 8, 1>;    // pooler   m=8:  uc1_g1, 2 CTAs/SM (routable)
using Arm1 = TmaInst<16, 8192, 4, 2, 4, 8, 1>;    // qkv_gate m=8:  uc1_g1, 2 CTAs/SM (routable)
using Arm2 = TmaInst<32, 8192, 8, 1, 8, 16, 1>;   // pooler   m=16: uc1_s8_g1, 1 CTA/SM (diag)
using Arm3 = BrInst<32, 8192, 8, 1, 8, 8, 0>;     // pooler   m=8:  bridge GA0 (the judged b12x, diag)
using Arm4 = BrInst<32, 8192, 8, 1, 8, 8, 1>;     // pooler   m=8:  bridge GA1 (diag)
using Arm5 = BrInst<16, 8192, 4, 2, 4, 8, 0>;     // qkv_gate m=8:  bridge GA0 (a7da6fa9 B12XQkv4, diag)
using Arm6 = BrInst<16, 8192, 4, 2, 4, 8, 1>;     // qkv_gate m=8:  bridge GA1 (diag)

void b12m_check(const torch::Tensor& wc, const torch::Tensor& esc, const torch::Tensor& x, const torch::Tensor& out,
                const torch::Tensor& err) {
  TORCH_CHECK(wc.is_cuda() && esc.is_cuda() && x.is_cuda() && out.is_cuda() && err.is_cuda(), "b12m: cuda tensors");
  TORCH_CHECK(wc.scalar_type() == at::kByte && esc.scalar_type() == at::kByte && x.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16 && err.scalar_type() == at::kInt, "b12m: dtypes");
  TORCH_CHECK(wc.is_contiguous() && esc.is_contiguous() && out.is_contiguous(), "b12m: contiguity");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && err.numel() >= 1, "b12m: dims / error word");
  TORCH_CHECK(wc.get_device() == x.get_device() && esc.get_device() == x.get_device() &&
              out.get_device() == x.get_device() && err.get_device() == x.get_device(), "b12m: one device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wc.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(esc.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "b12m: 16 B alignment");
}

}  // namespace

void b12m_run(torch::Tensor wc, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor err,
              int64_t arm, int64_t base) {
  b12m_check(wc, esc, x, out, err);
  TORCH_CHECK(base >= 1 && base <= 120, "b12m_run: base");
  const uint32_t b = (uint32_t)base;
  switch (arm) {
    case 0: Arm0::launch(wc, esc, x, out, err, b); break;
    case 1: Arm1::launch(wc, esc, x, out, err, b); break;
    case 2: Arm2::launch(wc, esc, x, out, err, b); break;
    case 3: Arm3::launch(wc, esc, x, out, err, b); break;
    case 4: Arm4::launch(wc, esc, x, out, err, b); break;
    case 5: Arm5::launch(wc, esc, x, out, err, b); break;
    case 6: Arm6::launch(wc, esc, x, out, err, b); break;
    default: TORCH_CHECK(false, "b12m_run: unknown arm");
  }
}

// {regs, local (spill) B, static smem B, dyn smem B, occupancy CTAs/SM}
std::vector<int64_t> b12m_setup(int64_t arm) {
  switch (arm) {
    case 0: return Arm0::setup();
    case 1: return Arm1::setup();
    case 2: return Arm2::setup();
    case 3: return Arm3::setup();
    case 4: return Arm4::setup();
    case 5: return Arm5::setup();
    case 6: return Arm6::setup();
    default: TORCH_CHECK(false, "b12m_setup: unknown arm");
  }
  return {};
}
"""


def _b12m_build():
    """Compile the b12m module once per process, at import (never during capture), for 9.0a only (TORCH_CUDA_ARCH_LIST
    set around the build and restored after), in its OWN extension (the m=1 bulk12 unit is not rebuilt). Records and
    prints the build seconds. Returns None (incumbent paths) on any failure; never raises."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils import cpp_extension
        print(f"[b12m] building {B12M_EXT_NAME}: TORCH_CUDA_ARCH_LIST=9.0a, extra_cuda_cflags {BULK_NVCC_FLAGS}",
              flush=True)
        ext = cpp_extension.load_inline(
            name=B12M_EXT_NAME, cpp_sources=_B12M_CPP, cuda_sources=_B12M_CUDA,
            functions=["b12m_run", "b12m_setup"], extra_cuda_cflags=list(BULK_NVCC_FLAGS), verbose=False)
    except Exception as e:
        tail = "\n".join(str(e).splitlines()[-60:])
        _B12M_BUILD["s"] = time.perf_counter() - t0
        print(f"[b12m] build FAILED after {_B12M_BUILD['s']:.1f}s ({type(e).__name__}); build log tail:\n{tail}\n"
              f"[b12m] incumbent paths kept", flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    _B12M_BUILD["s"] = time.perf_counter() - t0
    _B12M_BUILD["ok"] = True
    print(f"[build-time] b12m load_inline compiled in {_B12M_BUILD['s']:.1f}s", flush=True)
    return ext


_B12M_EXT = _b12m_build() if torch.cuda.is_available() else None


def _b12m_route(layer: str, x: torch.Tensor, cfg, state: dict):
    """The b12m arm serving this call, or None. Exact shape only: (layer, cfg m) in B12M_CELLS, x [m, k] with
    x.shape[0] == cfg m, bf16, contiguous, 16 B aligned, row stride a multiple of 16 B; the cell's state flag True
    (its candidate arm) or, in prepare()'s shallow gate copies only, an int arm id (a diag arm). cfg, shapes and
    state only, never input values."""
    m = cfg["m"]
    spec = B12M_CELLS.get((layer, m))
    if spec is None:
        return None
    v = state.get(spec[6], False)
    if v is True:
        arm = spec[4]
    elif type(v) is int:   # a _cell_gate extra arm (prepare()-only state copy); bool is not int here
        arm = v
    else:
        return None
    if _B12M_EXT is None or ("b12m_wc_" + layer) not in state:
        return None
    if not (cfg["n"] == spec[1] and cfg["k"] == spec[2] and x.dim() == 2 and x.shape[0] == m
            and x.shape[1] == spec[2] and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.data_ptr() % 16 == 0 and (x.stride(0) * 2) % 16 == 0):
        return None
    return arm


def _b12m_forward(layer: str, x: torch.Tensor, state: dict, arm: int) -> torch.Tensor:
    """ONE launch: out[m, n] = x @ W.T from the state's 12-bit weight (the tensor map over x is encoded inside
    b12m_run on every call). The error word is passed, never read here."""
    out = torch.empty((x.shape[0], B12M_CELLS[(layer, x.shape[0])][1]), dtype=torch.bfloat16, device=x.device)
    _B12M_EXT.b12m_run(state["b12m_wc_" + layer], state["b12m_esc_" + layer], x, out, state["b12m_err"], arm,
                       state["b12m_base_" + layer])
    return out


def _b12m_codes(state: dict, layer: str, rows: int, verbose: bool) -> str:
    """Put the layer's 12-bit codes at R = rows into state["b12m_{wc,esc,base}_" + layer]: section (10)'s m=1 buffers
    when the state holds them at the same R (no copy), else this cell's own encoding through _b12_encode_checked
    (host + device round trip, bit-exact). Returns one compact info string (source, R, base, escapes total / max per
    CTA, extra MB). Raises on any encode / round-trip failure."""
    inst = state.get("b12_inst_" + layer)
    if inst is not None and B12_INST_R.get(inst) == rows and ("b12_wc_" + layer) in state:
        wc, esc, base = state["b12_wc_" + layer], state["b12_esc_" + layer], state["b12_base_" + layer]
        src, extra = f"m=1 state inst {inst}", 0.0
    else:
        wc, esc, base = _b12_encode_checked(layer, state[layer], rows, B12M_HOME_INST[rows], verbose)
        src, extra = "own encode", (wc.numel() + esc.numel()) / 1e6
    ctas = state[layer].shape[0] // rows
    cnt = (esc.view(ctas, B12_ESC_BYTES)[:, 2 * rows:2 * rows + 2].contiguous().view(torch.int16).to(torch.int32)
           & 0xFFFF)
    state["b12m_wc_" + layer], state["b12m_esc_" + layer], state["b12m_base_" + layer] = wc, esc, base
    return (f"{layer} R{rows} {src} base {base} esc {int(cnt.sum())} max {int(cnt.max())}/CTA "
            f"+{extra:.1f}MB")


def _b12m_rows(fn, w: torch.Tensor, m: int, n: int, k: int, err: torch.Tensor) -> tuple:
    """prepare()-only per-row check of one arm (after _cell_check): a fresh random x, every row's rel L2 vs fp64 <=
    B12M_ROW_REL, finite, error word 0. Returns (why or "", worst row rel)."""
    g = torch.Generator(device=w.device)
    g.manual_seed(5150 + m)
    x = torch.randn((m, k), device=w.device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    print(f"[b12m] LAUNCH per-row check m={m} ...", flush=True)
    o = fn(x)
    torch.cuda.synchronize()
    code = int(err.item())
    err.zero_()
    print(f"[b12m] DONE   per-row check m={m} (synced, error word {code})", flush=True)
    if code:
        return f"per-row: error word {code}", float("nan")
    od = o.double()
    if not bool(torch.isfinite(od).all()):
        return "per-row: non-finite output", float("nan")
    ref = x.double() @ w[:n].double().T
    rows = ((od - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30))
    worst = float(rows.max())
    if not worst <= B12M_ROW_REL:
        return f"per-row: worst row rel L2 {worst:.3e} > {B12M_ROW_REL:g}", worst
    return "", worst


def _b12m_iqr(v) -> float:
    sv = sorted(v)
    return sv[(3 * len(sv)) // 4] - sv[len(sv) // 4]


def _b12m_rule(rounds: list) -> tuple:
    """Routing rule on B12M_ROUNDS read-flush _cell_gate rounds (stats dicts): (route, why, pooled ratio, pooled cand
    IQR, pooled incumbent IQR). Pooled ratio = (median of every round's raw cand reps - median baseline) / (same for
    the incumbent), net floors as _cell_gate's; route iff pooled <= B12_MAX_RATIO_M8, every round's net ratio <=
    B12M_ROUND_MAX, and the pooled cand IQR <= B12M_POOLED_IQR_MAX x max(pooled incumbent IQR, B12M_IQR_FLOOR_US)."""
    base = _ouns_median([r["base"] for r in rounds])
    pc = [t for r in rounds for t in r["cand"]["times"]]
    pm = [t for r in rounds for t in r["master"]["times"]]
    pooled = max(_ouns_median(pc) - base, 0.25) / max(_ouns_median(pm) - base, 0.25)
    ic, im = _b12m_iqr(pc), _b12m_iqr(pm)
    ratios = [r["cand"]["net"] / r["master"]["net"] for r in rounds]
    if not pooled <= B12_MAX_RATIO_M8:
        return False, f"pooled {pooled:.3f} > {B12_MAX_RATIO_M8}", pooled, ic, im
    if not max(ratios) <= B12M_ROUND_MAX:
        return False, f"a round {max(ratios):.3f} > {B12M_ROUND_MAX}", pooled, ic, im
    if not ic <= B12M_POOLED_IQR_MAX * max(im, B12M_IQR_FLOOR_US):
        return False, (f"pooled IQR {ic:.2f} > {B12M_POOLED_IQR_MAX}x max({im:.2f}, {B12M_IQR_FLOOR_US})"), pooled, ic, im
    return True, "", pooled, ic, im


def _b12m_prepare(state: dict, t_start: float, flags=None) -> None:
    """Section (12). Sets every B12M_CELLS flag (False unless every step passes) and, per routed cell's layer, the
    state's codes. flags None (the judge's state, A): per layer the codes (_b12m_codes), per arm setup / occupancy
    (needs MINB CTAs/SM), _cell_check + _b12m_rows, then B12M_ROUNDS read-flush _cell_gate rounds of kernel_fn with the
    cell flag on vs off (diag arms as extra arms) plus one write-flush round, _b12m_rule, and (routable cells only) a
    torch.profiler assertion of exactly one b12m_tma_kernel. Fills _B12M_SUM and one PATH CHOSEN decision per cell.
    flags a dict (the A/B check's second state): codes only, for each routable flag that is on. Never raises."""
    for spec in B12M_CELLS.values():
        state[spec[6]] = False
    if _B12M_EXT is None:
        if flags is None:
            _B12M_SUM.append("[b12m] no module (build failed or no GPU): incumbent everywhere")
            for (layer, m), spec in B12M_CELLS.items():
                _decision(f"PATH CHOSEN b12m {layer} m={m}: incumbent (no b12m module)")
        return
    dev = state["router"].device
    state["b12m_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        for (layer, m), spec in B12M_CELLS.items():
            if spec[7] and flags.get(spec[6]):
                try:   # a failure here turns off only this cell in this state, never the whole A/B
                    if ("b12m_wc_" + layer) not in state:
                        _b12m_codes(state, layer, spec[3], verbose=False)
                    state[spec[6]] = True
                except Exception as e:
                    print(f"[b12m] second state: {layer} m={m} codes raised {type(e).__name__}: {e}; cell off in "
                          f"this state", flush=True)
        return
    _B12M_SUM.clear()
    codes = {}      # layer -> info string, or None if that layer's codes failed
    arm_txt = {}    # arm -> compact setup / check text
    for (layer, m), spec in B12M_CELLS.items():
        tag, n, k, rows, cand, extra, flag, routable = spec
        t0 = time.perf_counter()
        why = ""
        rounds, wst, rule, ga = [], None, None, ""
        try:
            if t0 - t_start > B12M_START_DEADLINE_S:
                raise RuntimeError(f"skipped: deadline ({t0 - t_start:.0f}s into the state build > "
                                   f"{B12M_START_DEADLINE_S:.0f}s)")
            w = state[layer]
            if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16 and w.is_contiguous()):
                raise RuntimeError(f"weight {tuple(w.shape)} {w.dtype} unexpected")
            if layer not in codes:
                try:
                    codes[layer] = _b12m_codes(state, layer, rows, verbose=True)
                except Exception as e:
                    codes[layer] = None
                    print(f"[b12m] {layer} R={rows}: codes raised {type(e).__name__}: {str(e)[-300:]}", flush=True)
            if codes[layer] is None:
                raise RuntimeError(f"{layer} R={rows} codes failed")
            ok_extra = {}
            for a in (cand,) + tuple(extra.values()):
                name, kind, _l, _r, nt, stages, minb, cw, gamode = B12M_ARMS[a]
                label = f"{name}<R{rows},S{stages},MINB{minb},CW{cw},NT{nt}>"
                info = [int(v) for v in _B12M_EXT.b12m_setup(a)]
                print(f"[b12m] {tag} arm {a} {label}: regs={info[0]} local(spill)={info[1]} B static smem={info[2]} B "
                      f"dyn smem={info[3]} B, occupancy {info[4]} CTA/SM (needs {minb}), grid {n // rows} x "
                      f"{32 * (cw + 1)} threads", flush=True)
                txt = f"{name} r{info[0]} s{info[1]} {info[3] >> 10}K {info[4]}/SM"
                bad = ""
                if info[4] < minb:
                    bad = f"occupancy {info[4]} < {minb}"
                else:
                    st_a = dict(state)
                    st_a[flag] = True if a == cand else a
                    fn = (lambda xx, s=st_a: kernel_fn({"x": xx, "config": {"m": m, "n": n, "k": k, "layer": layer}}, s))
                    if _b12m_route(layer, torch.empty((m, k), dtype=torch.bfloat16, device=dev),
                                   {"m": m, "n": n, "k": k, "layer": layer}, st_a) != a:
                        bad = "route check failed on a synthetic x"
                    else:
                        bad = _cell_check(f"b12m {tag} {label}", fn, w, m, n, k, err=state["b12m_err"])
                    if not bad:
                        bad, worst = _b12m_rows(fn, w, m, n, k, state["b12m_err"])
                        txt += f" row{worst:.1e}"
                arm_txt[a] = txt + (f" FAIL({bad[:60]})" if bad else " ok")
                if bad:
                    if a == cand:
                        raise RuntimeError(f"{name}: {bad}")
                    print(f"[b12m] {tag} diag arm {name} dropped: {bad}", flush=True)
                else:
                    if a != cand:
                        ok_extra[[lab for lab, v in extra.items() if v == a][0]] = a
            route_txt = (f"route if pooled <= {B12_MAX_RATIO_M8}, every round <= {B12M_ROUND_MAX}, pooled IQR"
                         if routable else "diag, never routes")
            for rnd in range(B12M_ROUNDS):
                st = {}
                _cell_gate(f"b12m {tag} r{rnd + 1}", layer, m, n, k, flag, state, flush_mode="read", stats=st,
                           route_txt=route_txt, extra=ok_extra or None)
                rounds.append(st)
            wst = {}
            _cell_gate(f"b12m {tag} write", layer, m, n, k, flag, state, flush_mode="write", stats=wst,
                       route_txt="diagnostic")
            code = int(state["b12m_err"].item())
            state["b12m_err"].zero_()
            if code:
                raise RuntimeError(f"error word {code} after the gates")
            rule = _b12m_rule(rounds)
            if "GA0" in ok_extra and "GA1" in ok_extra:
                g0 = _ouns_median([r["GA0"]["net"] for r in rounds])
                g1 = _ouns_median([r["GA1"]["net"] for r in rounds])
                ga = f"; GA0 {g0:.2f} GA1 {g1:.2f} GA0-GA1 {g0 - g1:.2f}us"
            if not routable:
                why = "diag cell (never routes)"
            elif not rule[0]:
                why = rule[1]
            else:
                data = {"x": torch.randn((m, k), device=dev, dtype=torch.float32).to(torch.bfloat16),
                        "config": {"m": m, "n": n, "k": k, "layer": layer}}
                st_on = dict(state)
                st_on[flag] = True
                try:
                    names = _cell_kernels(lambda: kernel_fn(data, st_on))
                    print(f"[b12m] {tag} {layer} m={m}: profiled kernels {names}", flush=True)
                    if not (len(names) == 1 and "b12m_tma_kernel" in names[0]):
                        why = f"profiled kernels {names}: expected exactly one b12m_tma_kernel"
                except Exception as e:   # the assertion is part of the routing contract: no profile, no route
                    why = f"profiler unavailable ({type(e).__name__}: {str(e)[-120:]})"
            if not why:
                state[flag] = True
        except Exception as e:   # never raises: any surprise keeps the incumbent for this cell
            why = f"{type(e).__name__}: {str(e)[-200:]}"
            state[flag] = False
        state["b12m_err"].zero_()
        dt = time.perf_counter() - t0
        print(f"[prepare-time] b12m cell {tag} {layer} m={m} total: {dt:.2f}s", flush=True)
        r_txt = " ".join(f"r{i + 1} {r['master']['net']:.2f}/{r['cand']['net']:.2f}="
                         f"{r['cand']['net'] / r['master']['net']:.3f} iqr{r['master']['iqr']:.2f}/"
                         f"{r['cand']['iqr']:.2f}" for i, r in enumerate(rounds))
        if wst:
            r_txt += f" | write {wst['master']['net']:.2f}/{wst['cand']['net']:.2f}=" \
                     f"{wst['cand']['net'] / wst['master']['net']:.3f}"
        if rule is not None:
            r_txt += f" | pooled {rule[2]:.3f} iqr{rule[4]:.2f}/{rule[3]:.2f}"
        verdict = "ROUTE" if state[flag] else "NO-ROUTE"
        _B12M_SUM.append(f"[b12m] {tag} {B12M_ARMS[cand][0]} inc/arm us: {r_txt or 'not timed'}{ga} -> {verdict}"
                         + ("" if state[flag] else f" ({why[:110]})") + f" {dt:.1f}s")
        _decision(f"PATH CHOSEN b12m {layer} m={m}: "
                  + (f"candidate {B12M_ARMS[cand][0]} (b12m_tma_kernel)" if state[flag] else "incumbent")
                  + (f" [pooled {rule[2]:.3f}]" if rule is not None else "")
                  + ("" if state[flag] else f"; {why[:160]}"))
    for layer in {spec_layer for spec_layer, _m in B12M_CELLS}:
        if not any(state[spec[6]] for (lyr, _m), spec in B12M_CELLS.items() if lyr == layer):
            for key in ("b12m_wc_", "b12m_esc_", "b12m_base_"):
                state.pop(key + layer, None)
    _B12M_SUM.insert(0, "[b12m] codes: " + " | ".join(v if v else f"{lyr} FAILED" for lyr, v in codes.items()))
    _B12M_SUM.insert(1, "[b12m] arms (regs spill dynsmem occ worst-row): "
                     + " | ".join(f"{a}:{t}" for a, t in sorted(arm_txt.items())))
    _B12M_SUM.append("[b12m] PATH CHOSEN " + " | ".join(
        f"{layer} m={m}: " + (B12M_ARMS[spec[4]][0] if state[spec[6]] else "incumbent")
        for (layer, m), spec in B12M_CELLS.items()))
    torch.cuda.synchronize()


def _b12m_summary(prepare_s: float) -> None:
    """The <= B12M_SUMMARY_BYTES [b12m] block printed at the very end of prepare() (state A's lines)."""
    b = _B12M_BUILD["s"]
    head = (f"[b12m] SUMMARY build {'n/a' if b is None else f'{b:.1f}s'} ({B12M_EXT_NAME}, "
            f"{'ok' if _B12M_BUILD['ok'] else 'FAILED'}); prepare {prepare_s:.1f}s; route at pooled <= "
            f"{B12_MAX_RATIO_M8}, rounds <= {B12M_ROUND_MAX}; rounds as inc/arm=ratio")
    lines = [head] + list(_B12M_SUM)
    out, used = [], 0
    for line in lines:
        room = B12M_SUMMARY_BYTES - used - 1
        if room <= 20:
            break
        line = line if len(line) <= room else line[:room - 3] + "..."
        out.append(line)
        used += len(line) + 1
    print("\n".join(out), flush=True)


NEW_CELL_FLAGS = NEW_CELL_FLAGS + B12M_FLAGS   # section (12)'s routable cells join the A/B check and FINAL FLAGS


# (11) Pooler K-halves at m == 128 with fp32 partials (hypothesis pooler-khalf-m128-fp32-partials-triton-castadd,
# candidate (b2) only). Master runs this cell as torch.mm(x, pooler_t): a cuBLAS split-K nvjet GEMM plus a separate
# cublasLt splitKreduce node. Here the two K halves run as ONE strided-batched cuBLAS GEMM over master's NN weight
# with fp32 output (bf16 x / W, fp32 accumulation, fp32 partials [2, 128, 4096]), then ONE Triton kernel adds the two
# fp32 partials in fp32 and rounds once to bf16: the numerics of cuBLAS's / SGLang's own fp32 split-K reduce. No bf16
# partials anywhere (DISTILL 3.4). Own flag, canary, KHALF_ROUNDS read-flush gate rounds (POOLED rule on rounds 1-3,
# the rest a diagnostic) and a profiler kernel-name assertion; any miss keeps master's NN torch.mm. Pooler m=256,
# every m <= 32 path and every other layer are untouched.
KHALF_M = 128
KHALF_FLAG = "pooler_khalf128_ok"
KHALF_HALF = POOLER_K // 2
KHALF_ADD_BLOCK = 2048           # fp32 elements per program: 128 threads x 16 (4 x 16-byte loads per partial)
KHALF_ADD_WARPS = 4
KHALF_ROUNDS = 10                # independent read-flush _cell_gate rounds: 1..KHALF_DECIDE_ROUNDS route, all diagnose
KHALF_DECIDE_ROUNDS = 3          # the POOLED rule reads rounds 1-3 only; rounds 4-10 are the rule diagnostic
KHALF_FLOOR_RATIO = 1.01         # master / candidate, net of the empty-graph baseline, in EVERY decide round
KHALF_MIN_RATIO = 1.03           # the MEDIAN decide-round ratio (OLD rule: each of rounds 1-2)
KHALF_MAX_IQR = 1.5              # per-round cand IQR <= this x max(master IQR, floor): OLD/PARENT diagnostic only
KHALF_POOLED_IQR_MAX = 2.0       # pooled (123-rep) cand IQR <= this x max(pooled master IQR, floor): routes
KHALF_IQR_FLOOR_US = 0.5
KHALF_GATE_FLUSH = "read"        # the judge reads a buffer larger than L2 before each replay (DISTILL 3.2.1)
KHALF_START_DEADLINE_S = 170.0   # no khalf phase later than this after state A's build began (prepare has 300 s)


@triton.jit
def _add2_cast_kernel(p_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    """out[i] = bf16(p[i] + p[numel + i]): the two fp32 K-half partials of the pooler K-halves cell, added once in
    fp32 and rounded once (round-to-nearest-even) to bf16. p is the contiguous [2, m, n] fp32 bmm output."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < numel
    a = tl.load(p_ptr + offs, mask=mask, other=0.0)
    b = tl.load(p_ptr + numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (a + b).to(tl.bfloat16), mask=mask)


def _add2_cast(p: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    numel = out.numel()
    _add2_cast_kernel[(triton.cdiv(numel, KHALF_ADD_BLOCK),)](p, out, numel, BLOCK=KHALF_ADD_BLOCK,
                                                             num_warps=KHALF_ADD_WARPS)
    return out


def _khalf_forward(x: torch.Tensor, state: dict) -> torch.Tensor:
    """Pooler at m == 128 as two K halves: xh is a strided view of x (no copy), wh = pooler_t.view(2, 4096, 4096)
    (master's lossless NN copy, no copy); one batched cuBLAS GEMM with fp32 output gives fp32 partials
    [2, m, 4096]; _add2_cast_kernel sums them in fp32 and stores bf16 once into a fresh output."""
    m = x.shape[0]
    xh = x.view(m, 2, KHALF_HALF).permute(1, 0, 2)
    wh = state["pooler_t"].view(2, KHALF_HALF, POOLER_N)
    p = torch.bmm(xh, wh, out_dtype=torch.float32)
    if not p.is_contiguous():   # layout metadata only (never true for a fresh bmm output); never input values
        p = p.contiguous()
    out = torch.empty((m, POOLER_N), dtype=torch.bfloat16, device=x.device)
    return _add2_cast(p, out)


def _khalf_route_ok(x: torch.Tensor, cfg, state: dict) -> bool:
    """pooler at cfg m == KHALF_M with its own flag on: cfg, shapes and state only, never input values."""
    if cfg["m"] != KHALF_M or not state.get(KHALF_FLAG):
        return False
    wt = state["pooler_t"]
    return (cfg["n"] == POOLER_N and cfg["k"] == POOLER_K
            and tuple(wt.shape) == (POOLER_K, POOLER_N) and wt.is_contiguous()
            and x.dim() == 2 and x.shape[0] == KHALF_M and x.shape[1] == POOLER_K
            and x.dtype == torch.bfloat16 and x.is_contiguous())


def _khalf_names_ok(names: list) -> str:
    """"" if the profiled launches of one candidate call are exactly two: a cuBLAS GEMM with no split-K in its name
    and _add2_cast_kernel; else why not (a copy of xh / wh, a split-K reduce node, a third launch, ...)."""
    low = [nm.lower() for nm in names]
    if len(low) != 2:
        return f"expected exactly 2 launches (GEMM + _add2_cast_kernel), profiler saw {len(low)}"
    adds = [nm for nm in low if "_add2_cast_kernel" in nm]
    gemms = [nm for nm in low if "_add2_cast_kernel" not in nm]
    if len(adds) != 1:
        return "no single _add2_cast_kernel launch"
    g = gemms[0]
    if not any(t in g for t in ("nvjet", "gemm", "cutlass", "sm90", "xmma")):
        return f"second launch is not GEMM-like ({names})"
    if "splitk" in g or "reduce" in g:
        return f"the GEMM is a split-K variant ({g[:120]})"
    if any(t in g for t in ("memcpy", "copy", "contiguous")):
        return "a copy kernel was launched (xh / wh not consumed in place)"
    return ""


def _khalf_add_canary(dev) -> str:
    """Named, synced canary of the new Triton kernel alone, before any timing: fp32 partials at the real cell
    shape (with magnitudes spread over many binades), compared bit for bit with torch's fp32 add rounded once to
    bf16. Returns "" on pass, else why."""
    g = torch.Generator(device=dev)
    g.manual_seed(24680)
    numel = KHALF_M * POOLER_N
    p = torch.randn((2, KHALF_M, POOLER_N), device=dev, dtype=torch.float32, generator=g)
    p *= torch.exp2(torch.randint(-20, 21, (2, KHALF_M, POOLER_N), device=dev, generator=g).float())
    out = torch.empty((KHALF_M, POOLER_N), dtype=torch.bfloat16, device=dev)
    print(f"[khalf] LAUNCH canary _add2_cast_kernel numel={numel} grid {triton.cdiv(numel, KHALF_ADD_BLOCK)} "
          f"BLOCK {KHALF_ADD_BLOCK} warps {KHALF_ADD_WARPS} ...", flush=True)
    _add2_cast(p, out)
    torch.cuda.synchronize()
    print("[khalf] DONE   canary _add2_cast_kernel (synced)", flush=True)
    ref = (p[0] + p[1]).to(torch.bfloat16)
    if not torch.equal(out, ref):
        nbad = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
        return f"_add2_cast_kernel differs from torch fp32 add + one bf16 rounding in {nbad} of {numel} elements"
    return ""


def _khalf_iqr(v) -> float:
    """_cell_gate's quartile convention (sorted[3n // 4] - sorted[n // 4]) on any list of per-rep us."""
    sv = sorted(v)
    return sv[(3 * len(sv)) // 4] - sv[len(sv) // 4]


def _khalf_round_iqr_ok(r: dict) -> bool:
    """The per-round (41-rep) IQR clause of the OLD and PARENT rules."""
    return r["cand"]["iqr"] <= KHALF_MAX_IQR * max(r["master"]["iqr"], KHALF_IQR_FLOOR_US)


def _khalf_rules(rs: list) -> dict:
    """OLD / PARENT / POOLED verdicts on one triple of _cell_gate rounds (each with "ratio" = master/cand and the
    stats dict's cand/master "iqr" and raw "times"): {rule: (route, clause, why)} with clause "" on a route, else
    the first failing one (ratio / floor / median / iqr), plus "med" (median ratio) and "piqr" (pooled master, cand
    IQR in us). OLD: rounds 1-2 of the triple both >= KHALF_MIN_RATIO and the per-round IQR clause. PARENT: every
    ratio >= KHALF_FLOOR_RATIO, median >= KHALF_MIN_RATIO, the per-round IQR clause in all 3. POOLED (the one that
    routes): the same floor and median, and cand IQR <= KHALF_POOLED_IQR_MAX x max(master IQR, floor) on the pooled
    raw per-rep times of the triple."""
    ratios = [r["ratio"] for r in rs]
    med = sorted(ratios)[len(ratios) // 2]
    pm = _khalf_iqr([t for r in rs for t in r["master"]["times"]])
    pc = _khalf_iqr([t for r in rs for t in r["cand"]["times"]])
    lo = min(range(len(rs)), key=lambda i: ratios[i])
    if ratios[lo] < KHALF_FLOOR_RATIO:
        common = ("floor", f"r{lo + 1} master/cand {ratios[lo]:.3f} < floor {KHALF_FLOOR_RATIO}")
    elif med < KHALF_MIN_RATIO:
        common = ("median", f"median master/cand {med:.3f} < {KHALF_MIN_RATIO}")
    else:
        common = None
    out = {"med": med, "piqr": (pm, pc)}
    if not all(r["ratio"] >= KHALF_MIN_RATIO for r in rs[:2]):
        out["OLD"] = (False, "ratio", f"rounds 1-2 master/cand not both >= {KHALF_MIN_RATIO}")
    elif not all(_khalf_round_iqr_ok(r) for r in rs[:2]):
        out["OLD"] = (False, "iqr", f"per-round cand IQR > {KHALF_MAX_IQR}x master in round 1 or 2")
    else:
        out["OLD"] = (True, "", "")
    if common:
        out["PARENT"] = (False,) + common
    elif not all(_khalf_round_iqr_ok(r) for r in rs):
        out["PARENT"] = (False, "iqr", f"per-round cand IQR > {KHALF_MAX_IQR}x master in some round")
    else:
        out["PARENT"] = (True, "", "")
    if common:
        out["POOLED"] = (False,) + common
    elif pc > KHALF_POOLED_IQR_MAX * max(pm, KHALF_IQR_FLOOR_US):
        out["POOLED"] = (False, "iqr", f"pooled cand IQR {pc:.2f} > {KHALF_POOLED_IQR_MAX}x max(pooled master IQR "
                                       f"{pm:.2f}, {KHALF_IQR_FLOOR_US}) us")
    else:
        out["POOLED"] = (True, "", "")
    return out


def _khalf_short(v: tuple) -> str:
    """One rule verdict for the compact K-DIAG line: "route" or "miss(<first failing clause>)"."""
    return "route" if v[0] else f"miss({v[1]})"


def _khalf_prepare(state: dict, t_start: float) -> None:
    """State A only. Sets state[KHALF_FLAG]; every surprise keeps master's NN torch.mm. Steps: weight layout check,
    a synced bmm(out_dtype=float32) probe, the add kernel's own canary (bit-exact), _cell_check through kernel_fn
    (canary + rel L2 vs fp64 + probe row + repeat bit-identical), KHALF_ROUNDS read-flush _cell_gate rounds against
    master's actually-routed path (flag off), routed by the POOLED rule on rounds 1..KHALF_DECIDE_ROUNDS only
    (_khalf_rules; later rounds are the OLD/PARENT/POOLED diagnostic and never route), then one torch.profiler
    assertion. One PATH CHOSEN decision line with rounds 1-3's ratios, per-round and pooled IQRs, the median and the
    deciding rule, and one K-DIAG line. Never raises."""
    state[KHALF_FLAG] = False
    t0 = time.perf_counter()
    why = ""
    rounds = []
    rules = None
    diag_err = ""
    try:
        if t0 - t_start > KHALF_START_DEADLINE_S:
            why = f"skipped: deadline ({t0 - t_start:.0f}s into the state build > {KHALF_START_DEADLINE_S:.0f}s)"
        w, wt = state["pooler"], state["pooler_t"]
        if not why and not (tuple(w.shape) == (POOLER_N, POOLER_K) and w.dtype == torch.bfloat16
                            and tuple(wt.shape) == (POOLER_K, POOLER_N) and wt.dtype == torch.bfloat16
                            and wt.is_contiguous()):
            why = f"weight layout unexpected (pooler {tuple(w.shape)}, pooler_t {tuple(wt.shape)})"
        dev = w.device
        cfg = {"m": KHALF_M, "n": POOLER_N, "k": POOLER_K, "layer": "pooler"}
        st_on = dict(state)
        st_on[KHALF_FLAG] = True
        if not why:
            xs = torch.randn((KHALF_M, POOLER_K), device=dev, dtype=torch.float32).to(torch.bfloat16)
            print("[khalf] LAUNCH probe torch.bmm(out_dtype=torch.float32) m=128 ...", flush=True)
            try:
                pp = torch.bmm(xs.view(KHALF_M, 2, KHALF_HALF).permute(1, 0, 2), wt.view(2, KHALF_HALF, POOLER_N),
                               out_dtype=torch.float32)
                torch.cuda.synchronize()
                print(f"[khalf] DONE   probe torch.bmm(out_dtype=torch.float32) (synced): {tuple(pp.shape)} "
                      f"{pp.dtype} contiguous={pp.is_contiguous()}", flush=True)
                if not (tuple(pp.shape) == (2, KHALF_M, POOLER_N) and pp.dtype == torch.float32):
                    why = f"bmm out_dtype probe gave {tuple(pp.shape)} {pp.dtype}"
                del pp
            except Exception as e:
                why = f"torch.bmm(out_dtype=torch.float32) rejected ({type(e).__name__}: {str(e)[-160:]})"
        if not why:
            why = _khalf_add_canary(dev)
        if not why and _khalf_route_ok(xs, cfg, st_on) is not True:
            why = "route check failed on a synthetic m=128 x"
        if not why:
            bad = _cell_check("khalf-b2 fp32 partials", lambda xx: kernel_fn({"x": xx, "config": cfg}, st_on),
                              w, KHALF_M, POOLER_N, POOLER_K)
            if bad:
                why = f"canary/check: {bad}"
        if not why:
            bar = (f"POOLED rule on r1-{KHALF_DECIDE_ROUNDS}: floor {KHALF_FLOOR_RATIO}, median >= {KHALF_MIN_RATIO}, "
                   f"pooled IQR <= {KHALF_POOLED_IQR_MAX}x; later rounds diagnostic")
            for rnd in range(KHALF_ROUNDS):
                st = {}
                try:
                    cand_us, master_us = _cell_gate(f"khalf-b2 round {rnd + 1}/{KHALF_ROUNDS}", "pooler", KHALF_M,
                                                    POOLER_N, POOLER_K, KHALF_FLAG, state,
                                                    flush_mode=KHALF_GATE_FLUSH, stats=st, route_txt=bar)
                except Exception as e:   # a diagnostic round's failure only shortens the diagnostic
                    if rnd < KHALF_DECIDE_ROUNDS:
                        raise
                    diag_err = f"; diagnostic stopped at round {rnd + 1} ({type(e).__name__}: {str(e)[-120:]})"
                    break
                st["ratio"] = master_us / cand_us
                rounds.append(st)
            rules = _khalf_rules(rounds[:KHALF_DECIDE_ROUNDS])
            if not rules["POOLED"][0]:
                why = (f"gate (POOLED rule, rounds 1-{KHALF_DECIDE_ROUNDS}): {rules['POOLED'][2]} "
                       f"(not a measured loss of the row unless a ratio < 1)")
        if not why:
            data = {"x": xs, "config": cfg}
            try:
                names = _cell_kernels(lambda: kernel_fn(data, st_on))
                print(f"[khalf] m={KHALF_M}: profiled kernels {names}", flush=True)
                why = _khalf_names_ok(names)
            except Exception as e:   # the assertion is part of the routing contract: no profile, no route
                why = f"profiler unavailable ({type(e).__name__}: {str(e)[-160:]})"
            try:
                mnames = _cell_kernels(lambda: kernel_fn(data, state))
                print(f"[khalf] m={KHALF_M}: master (flag off, actually routed) kernels {mnames}", flush=True)
            except Exception as e:
                print(f"[khalf] master profile unavailable ({type(e).__name__})", flush=True)
        if not why:
            state[KHALF_FLAG] = True
    except Exception as e:  # never raises: any surprise keeps master's NN path
        why = f"raised {type(e).__name__}: {str(e)[-200:]}"
        state[KHALF_FLAG] = False
    torch.cuda.synchronize()
    r_txt = "; ".join(
        f"r{i + 1} master/cand {r['ratio']:.3f} (net {r['master']['net']:.2f} / {r['cand']['net']:.2f} us; "
        f"med {r['master']['med']:.2f} / {r['cand']['med']:.2f} us, IQR {r['master']['iqr']:.2f} / "
        f"{r['cand']['iqr']:.2f}, base {r['base']:.2f})"
        for i, r in enumerate(rounds[:KHALF_DECIDE_ROUNDS])) or "gate not run"
    if rules is not None:
        pm, pc = rules["piqr"]
        n_rep = sum(len(r["cand"]["times"]) for r in rounds[:KHALF_DECIDE_ROUNDS])
        r_txt += (f"; median {rules['med']:.3f}, pooled IQR master/cand {pm:.2f} / {pc:.2f} us ({n_rep} reps each); "
                  f"decided by POOLED: {_khalf_short(rules['POOLED'])} (OLD would {_khalf_short(rules['OLD'])}, "
                  f"PARENT would {_khalf_short(rules['PARENT'])})")
    print(f"[prepare-time] khalf pooler m={KHALF_M} total: {time.perf_counter() - t0:.2f}s", flush=True)
    _decision(f"PATH CHOSEN K pooler m={KHALF_M}: "
              + ("candidate b2 (bmm fp32 partials + _add2_cast_kernel)" if state[KHALF_FLAG] else "master (NN torch.mm)")
              + f" [{KHALF_GATE_FLUSH}-flush gate, {r_txt}]" + ("" if state[KHALF_FLAG] else f"; {why}"))
    if rules is not None:   # the retired control's diagnostic: never routes, one compact line
        try:
            nr = len(rounds)
            trip = [(a, rounds[a:a + KHALF_DECIDE_ROUNDS]) for a in range(0, nr - KHALF_DECIDE_ROUNDS + 1,
                                                                            KHALF_DECIDE_ROUNDS)]
            t_txt = "; ".join(
                f"{a + 1}-{a + len(t)} " + " ".join(f"{k} {_khalf_short(v[k])}" for k in ("OLD", "PARENT", "POOLED"))
                for a, t in trip for v in (_khalf_rules(t),))
            _decision(f"K-DIAG pooler m={KHALF_M} rounds 1-{nr}: ratios "
                      + " ".join(f"{r['ratio']:.3f}" for r in rounds)
                      + f"; per-round IQR ({KHALF_MAX_IQR}x) fail "
                      + f"{sum(not _khalf_round_iqr_ok(r) for r in rounds)}/{nr}, ratio < {KHALF_MIN_RATIO} "
                      + f"{sum(r['ratio'] < KHALF_MIN_RATIO for r in rounds)}/{nr}; " + t_txt + diag_err)
        except Exception as e:   # diagnostic only: never raises
            print(f"[khalf] K-DIAG unavailable ({type(e).__name__}: {str(e)[-120:]})", flush=True)


NEW_CELL_FLAGS = NEW_CELL_FLAGS + (KHALF_FLAG,)   # section (11)'s cell joins the A/B check and the FINAL FLAGS line
NEW_CELL_FLAGS = NEW_CELL_FLAGS + ROUTER_PDL_FLAGS   # R-PDL cells join the A/B check and the FINAL FLAGS line


# ---------------------------------------------------------------------------------------------------------------------
# (13) BF1 (campaign 2026-10-30-public-ladder rung BF1; one mechanism; user-requested): the skinny swap-AB GEMM at
# m = 8..32 on a fragment-ordered, stage-major BF16 weight stream, intra-CTA warp split-K, no cross-CTA reduction.
# Design re-implemented from github.com/alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_bf16_proj.py (sk2 route),
# uncompressed BF16 only (no 12-bit code; that is rung BF2); no code copied. prepare() permutes each raced layer's
# [N, K] weight losslessly into mma.m16n8k16 B-fragment order (8 rows x 64 k = one 1024 B tile unit, lane l's
# k-steps 0-1 at byte 16 l and 2-3 at 512 + 16 l), laid out [stage][CTA][warp][unit][n8 tile] so a CTA's stage is
# one contiguous block. One CTA owns R rows over all of K; warps split rows (NWR) and, inside each stage, K (WSK);
# the K-warps meet in a shared-memory FP32 reduction (intra-CTA only: bf16 CLAIMS:42/:57 refuted the cross-CTA
# split-K / stream-K forms, this is not those). Tokens sit on the MMA M side (x by ldmatrix from a shared-memory
# ring), weight rows on N. A producer warp moves each stage into a shared-memory ring: the CTA's weight block with
# ONE 1-D cp.async.bulk and x with ONE 3-D TMA box {64 k, 16|32 rows, units} (SWIZZLE_128B, rows >= m zero-filled
# out of bounds), both on the stage's mbarrier (complete_tx). (WREG True, consumers ld.global.nc their fragments
# straight into registers, is kept in the template but lost 10-34% on the H100 and is not instantiated.) Each (layer, m range) cell is
# raced in prepare() against the incumbent kernel_fn route (b12m / swap-AB Triton / cuBLAS) and routes only if it
# is faster; the incumbent stays the fallback.
BF1_EXT_NAME = "muse_bf16_bf1_sk_v1"
BF1_RING_BUDGET = 200 * 1024     # ring bytes per CTA (1 CTA/SM)
BF1_MAX_STAGES = 8
# layer -> (N, K, NT, NWR, WSK, UPW, WREG): one stream geometry per layer (every m of a layer reads the same bytes)
BF1_GEOMS = {
    "pooler": (4096, 8192, 2, 2, 2, 4, False),     # R 32, 128 CTAs, 512 k / stage (32 KB weights)
    "unpooler": (8192, 4096, 4, 2, 2, 2, False),   # R 64, 128 CTAs, 256 k / stage (32 KB weights)
    "qkv_gate": (2304, 8192, 3, 1, 4, 2, False),   # R 24, 96 CTAs (NEG decode #8), 512 k / stage (24 KB weights)
    "o": (8192, 1024, 4, 2, 2, 2, False),          # R 64, 128 CTAs, 256 k / stage (32 KB weights)
}
# (layer, m_lo, m_hi) -> (raced m, state flag); the kernel serves any m in [m_lo, m_hi] (MT = 1 if m_hi <= 16 else 2)
BF1_CELLS = {
    ("pooler", 8, 8): (8, "bf1_pooler8_ok"), ("pooler", 9, 16): (16, "bf1_pooler16_ok"),
    ("pooler", 17, 32): (32, "bf1_pooler32_ok"),
    ("unpooler", 9, 16): (16, "bf1_unpooler16_ok"), ("unpooler", 17, 32): (32, "bf1_unpooler32_ok"),
    ("qkv_gate", 8, 8): (8, "bf1_qkv8_ok"), ("qkv_gate", 9, 16): (16, "bf1_qkv16_ok"),
    ("qkv_gate", 17, 32): (32, "bf1_qkv32_ok"),
    ("o", 8, 8): (8, "bf1_o8_ok"), ("o", 9, 16): (16, "bf1_o16_ok"), ("o", 17, 32): (32, "bf1_o32_ok"),
}
BF1_FLAGS = tuple(spec[1] for spec in BF1_CELLS.values())
BF1_INSTS = tuple(sorted({(layer, 1 if hi <= 16 else 2) for (layer, lo, hi) in BF1_CELLS}))   # cfg id = index
BF1_ROUNDS = 3                  # read-flush _cell_gate rounds (routing)
BF1_POOLED_MAX = 0.97           # pooled cand / incumbent must be <= this
BF1_ROUND_MAX = 0.99            # and every round's
BF1_POOLED_IQR_MAX = 3.0        # pooled cand IQR <= this x max(pooled incumbent IQR, floor) (b12m rule, 2.0 -> 3.0: H100 pooler m32 0.929 x3 rounds was refused at 1.09 vs 1.0)
BF1_IQR_FLOOR_US = 0.5
BF1_START_DEADLINE_S = 200.0    # no bf1 cell later than this after state A's build began (prepare has 300 s)
_BF1_SUM = []                   # [bf1] summary lines (state A), printed last by prepare()
_BF1_BUILD = {"s": None, "ok": False}


def _bf1_geom(layer: str, mt: int):
    """(R, UPS, KST, NW, XBY, WBY, SST, NSTAGE, RED, smem, grid) of one instance."""
    N, K, NT, NWR, WSK, UPW, WREG = BF1_GEOMS[layer]
    R = NWR * NT * 8
    UPS = WSK * UPW
    KST = UPS * 64
    NW = NWR * WSK
    XBY = UPS * 16 * mt * 128         # one 3-D TMA box of x per stage: [unit][token row][128 B], SWIZZLE_128B
    WBY = NW * UPW * NT * 1024
    SST = XBY + (0 if WREG else WBY)
    NSTAGE = max(2, min(BF1_MAX_STAGES, K // KST, BF1_RING_BUDGET // SST))
    RP = R + ((40 - (R % 32)) % 32)
    RED = WSK * 16 * mt * RP * 4 if WSK > 1 else 0
    return R, UPS, KST, NW, XBY, WBY, SST, NSTAGE, RED, 2048 + max(NSTAGE * SST, RED), N // R


def _bf1_repack(w: torch.Tensor, layer: str) -> torch.Tensor:
    """Lossless permutation of w [N, K] bf16 into the layer's stage-major fragment stream (flat bf16, N*K elements):
    tile units [n8][ku][jh][lane = 4 g + t4][jl][h][e] with k = 64 ku + 16 (2 jh + jl) + 8 h + 2 t4 + e and row
    8 n8 + g; then n8 = (cta, wr, nt), ku = (stage, wk, uu) -> [stage][cta][wr][wk][uu][nt][512]."""
    N, K, NT, NWR, WSK, UPW, WREG = BF1_GEOMS[layer]
    R, UPS, KST = _bf1_geom(layer, 1)[:3]
    if not (tuple(w.shape) == (N, K) and w.dtype == torch.bfloat16 and N % R == 0 and K % KST == 0):
        raise RuntimeError(f"bf1: {layer} weight {tuple(w.shape)} {w.dtype} does not fit the geometry")
    t = w.reshape(N // 8, 8, K // 64, 2, 2, 2, 4, 2).permute(0, 2, 3, 1, 6, 4, 5, 7).reshape(N // 8, K // 64, 512)
    t = t.reshape(N // R, NWR, NT, K // KST, WSK, UPW, 512).permute(3, 0, 1, 4, 5, 2, 6)
    return t.contiguous().view(-1)


_BF1_CPP = r"""
#include <torch/extension.h>
void bf1_run(torch::Tensor ws, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t cfg, int64_t dg);
std::vector<int64_t> bf1_info(int64_t cfg);
"""

_BF1_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#define BF1_WAIT_NS 500000000ULL   // bounded mbarrier waits (0.5 s), then the error word

namespace {

__device__ __forceinline__ void bf1_mbar_init(uint32_t a, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void bf1_expect(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void bf1_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ bool bf1_wait(uint32_t a, uint32_t parity) {
  unsigned long long t0 = 0;
  for (uint32_t it = 0;; ++it) {
    uint32_t done;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) return true;
    if ((it & 255) == 255) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      if (t0 == 0) t0 = t;
      else if (t - t0 > BF1_WAIT_NS) return false;
    }
  }
}
__device__ __forceinline__ void bf1_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bf1_tma3(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void bf1_ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void bf1_lds16(uint32_t a, uint32_t* r) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a)
               : "memory");
}
__device__ __forceinline__ uint4 bf1_ldg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void bf1_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA = R = NWR*NT*8 weight rows over all of K. Stage = UPS = WSK*UPW 64-k units. Warp w = wr*WSK + wk owns rows
// (wr*NT .. wr*NT+NT-1) n8 tiles and units wk*UPW .. wk*UPW+UPW-1 of every stage. Warp NW = producer.
template <int MT, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool WREG>
__global__ void __launch_bounds__((NWR * WSK + 1) * 32, 1)
bf1_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
           int* __restrict__ err, int M, int N, int K, int dg) {
  constexpr int NW = NWR * WSK, NC = NW * 32, R = NWR * NT * 8, UPS = WSK * UPW, KST = UPS * 64, TOKR = 16 * MT;
  constexpr int XBY = UPS * TOKR * 128;                    // x box [unit][token row][128 B], SWIZZLE_128B
  constexpr int WBY = NW * UPW * NT * 1024;
  constexpr int SST = XBY + (WREG ? 0 : WBY);
  constexpr int RP = R + ((40 - (R % 32)) % 32);           // = 8 (mod 32): conflict-free 64-bit partial stores
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  unsigned char* smem = smem_raw + (((sraw + 1023u) & ~1023u) - sraw);   // 1024 B aligned (SWIZZLE_128B boxes)
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t bars = sb;                                // full[NSTAGE], empty[NSTAGE]
  const uint32_t ring = sb + 1024;
  unsigned char* ringp = smem + 1024;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  const int nst = K / KST;
  const size_t wstride = (size_t)gridDim.x * WBY;          // one stage of every CTA
  const uint8_t* wcta = Ws + (size_t)cta * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { bf1_mbar_init(bars + 8 * i, 1); bf1_mbar_init(bars + 8 * (NSTAGE + i), NW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == NW) {   // producer warp
    for (int st = 0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE && !bf1_wait(bars + 8 * (NSTAGE + slot), ((st / NSTAGE) - 1) & 1)) {
        if (lane == 0) atomicOr(err, 1);
        break;
      }
      const uint32_t s0 = ring + slot * SST;
      if (lane == 0) bf1_expect(bars + 8 * slot, ((dg & 1) ? 0u : (uint32_t)XBY) + (WREG ? 0u : (uint32_t)WBY));
      __syncwarp();
      if (!WREG && lane == 0) bf1_bulk(s0 + XBY, wcta + (size_t)st * wstride, WBY, bars + 8 * slot);
      // x: ONE 3-D box {64 k, TOKR rows, UPS units} per stage; rows >= M are out of bounds -> zero-filled
      if (!(dg & 1) && lane == 0) bf1_tma3(s0, &xmap, 0, 0, st * UPS, bars + 8 * slot);
    }
    return;
  }

  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int xr = (lane & 7) + ((lane >> 3) & 1) * 8, xc = lane >> 4;   // ldmatrix: this lane's row / 16 B chunk
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

  const size_t woff = (size_t)(warp * UPW) * NT * 1024 + lane * 16;   // this lane's bytes inside a CTA stage
  uint4 wa[WREG ? UPW : 1][WREG ? NT : 1][2];
  if constexpr (WREG) {
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const uint8_t* p = wcta + woff + (uu * NT + nt) * 1024;
        wa[uu][nt][0] = bf1_ldg16(p);
        wa[uu][nt][1] = bf1_ldg16(p + 512);
      }
  }
  bool ok = true;
  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    uint4 wb[WREG ? UPW : 1][WREG ? NT : 1][2];
    if constexpr (WREG) {
      if (st + 1 < nst) {
        const uint8_t* base = wcta + (size_t)(st + 1) * wstride + woff;
#pragma unroll
        for (int uu = 0; uu < UPW; ++uu)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            wb[uu][nt][0] = bf1_ldg16(base + (uu * NT + nt) * 1024);
            wb[uu][nt][1] = bf1_ldg16(base + (uu * NT + nt) * 1024 + 512);
          }
      }
    }
    if (!bf1_wait(bars + 8 * slot, (st / NSTAGE) & 1)) { ok = false; break; }
    const uint32_t s0 = ring + slot * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      uint32_t b[NT][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (WREG) {
          b[nt][0] = wa[uu][nt][0].x; b[nt][1] = wa[uu][nt][0].y; b[nt][2] = wa[uu][nt][0].z; b[nt][3] = wa[uu][nt][0].w;
          b[nt][4] = wa[uu][nt][1].x; b[nt][5] = wa[uu][nt][1].y; b[nt][6] = wa[uu][nt][1].z; b[nt][7] = wa[uu][nt][1].w;
        } else {
          const uint32_t a0 = s0 + XBY + (uint32_t)woff + (uu * NT + nt) * 1024;
          bf1_lds16(a0, &b[nt][0]);
          bf1_lds16(a0 + 512, &b[nt][4]);
        }
      }
      uint32_t a[4][MT][4];
      if (dg & 2) {   // diagnostic: weights only (no x fragments, no MMA)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          acc[0][nt][0] += __uint_as_float((b[nt][0] ^ b[nt][1] ^ b[nt][2] ^ b[nt][3] ^ b[nt][4] ^ b[nt][5] ^ b[nt][6] ^ b[nt][7]) & 0x007fffffu);
        if (uu == UPW - 1) { __syncwarp(); if (lane == 0) bf1_arrive(bars + 8 * (NSTAGE + slot)); }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          bf1_ldsm4(s0 + (ui * TOKR + mt * 16 + xr) * 128 + (((2 * j + xc) ^ (xr & 7)) << 4), a[j][mt][0], a[j][mt][1], a[j][mt][2],
                    a[j][mt][3]);
      if (uu == UPW - 1) {   // every shared-memory read of the stage is in registers: hand the slot back
        __syncwarp();
        if (lane == 0) bf1_arrive(bars + 8 * (NSTAGE + slot));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            bf1_mma(acc[mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
    }
    if constexpr (WREG) {
#pragma unroll
      for (int uu = 0; uu < UPW; ++uu)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { wa[uu][nt][0] = wb[uu][nt][0]; wa[uu][nt][1] = wb[uu][nt][1]; }
    }
  }
  if (!ok && lane == 0) atomicOr(err, 2);

  if constexpr (WSK == 1) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = cta * R + (wr * NT + nt) * 8 + 2 * t4;
        const int t0 = mt * 16 + g;
        if (t0 < M)
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)t0 * N + row) = __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
        if (t0 + 8 < M)
          *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(t0 + 8) * N + row) = __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
      }
  } else {
    float* red = reinterpret_cast<float*>(ringp);   // the ring is idle: every copy landed (all stages consumed)
    asm volatile("bar.sync 1, %0;" :: "r"(NC) : "memory");
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int t0 = mt * 16 + g;
        float* r0 = red + (wk * TOKR + t0) * RP + row;
        *reinterpret_cast<float2*>(r0) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      }
    asm volatile("bar.sync 1, %0;" :: "r"(NC) : "memory");
    constexpr int R2 = R / 2;
    for (int idx = tid; idx < M * R2; idx += NC) {
      const int tt = idx / R2, row = (idx - tt * R2) * 2;
      float2 v = *reinterpret_cast<const float2*>(red + tt * RP + row);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float2 p = *reinterpret_cast<const float2*>(red + (w * TOKR + tt) * RP + row);
        v.x += p.x; v.y += p.y;
      }
      *reinterpret_cast<__nv_bfloat162*>(out + (size_t)tt * N + cta * R + row) = __floats2bfloat162_rn(v.x, v.y);
    }
  }
}

typedef CUresult (*Bf1EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// cuTensorMapEncodeTiled through the runtime (no -lcuda); the cached value is a driver function pointer.
Bf1EncodeFn bf1_encode_fn() {
  static Bf1EncodeFn cached = nullptr;
  if (cached != nullptr) return cached;
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
  const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
#else
  const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
#endif
  if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || fn == nullptr) {
    (void)cudaGetLastError();
    return nullptr;
  }
  cached = reinterpret_cast<Bf1EncodeFn>(fn);
  return cached;
}

// Host-only, every call: a stack-local 3-D map over THIS call's x [M, K] bf16 as {64 k, M rows, K/64 units}
// (strides K*2 B, 128 B), box {64, TOKR, UPS}, SWIZZLE_128B, OOB rows zero-filled; passed as the __grid_constant__.
CUtensorMap bf1_xmap(const torch::Tensor& x, int tokr, int ups) {
  const Bf1EncodeFn enc = bf1_encode_fn();
  TORCH_CHECK(enc != nullptr, "bf1: cuTensorMapEncodeTiled entry point missing");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) == x.size(1) && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "bf1: x must be contiguous and 16 B aligned");
  CUtensorMap mp;
  std::memset(&mp, 0, sizeof(mp));
  const cuuint64_t dims[3] = {64, static_cast<cuuint64_t>(x.size(0)), static_cast<cuuint64_t>(x.size(1) / 64)};
  const cuuint64_t strides[2] = {static_cast<cuuint64_t>(x.size(1)) * 2, 128};
  const cuuint32_t box[3] = {64, static_cast<cuuint32_t>(tokr), static_cast<cuuint32_t>(ups)};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult rc = enc(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, x.data_ptr(), dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "bf1: cuTensorMapEncodeTiled rc ", static_cast<int>(rc));
  return mp;
}

template <int MT, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool WREG>
struct Bf1Inst {
  static constexpr int NW = NWR * WSK, R = NWR * NT * 8, UPS = WSK * UPW, KST = UPS * 64, TOKR = 16 * MT;
  static constexpr int XBY = UPS * TOKR * 128, WBY = NW * UPW * NT * 1024;
  static constexpr int SST = XBY + (WREG ? 0 : WBY), RP = R + ((40 - (R % 32)) % 32);
  static constexpr int RED = WSK > 1 ? WSK * TOKR * RP * 4 : 0;
  static constexpr int SMEM = 2048 + (NSTAGE * SST > RED ? NSTAGE * SST : RED);
  static void set_attr() {
    static bool done = false;
    if (!done) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(bf1_kernel<MT, NT, NWR, WSK, UPW, NSTAGE, WREG>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      done = true;
    }
  }
  static void run(const torch::Tensor& ws, const torch::Tensor& x, torch::Tensor& out, torch::Tensor& err, int dg) {
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
    TORCH_CHECK(M >= 1 && M <= TOKR && N % R == 0 && K % KST == 0, "bf1: shape outside the instance");
    TORCH_CHECK(ws.numel() * ws.element_size() == (int64_t)N * K * 2, "bf1: weight stream size");
    set_attr();
    const CUtensorMap xm = bf1_xmap(x, TOKR, UPS);
    bf1_kernel<MT, NT, NWR, WSK, UPW, NSTAGE, WREG><<<N / R, (NW + 1) * 32, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        xm, reinterpret_cast<const uint8_t*>(ws.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), err.data_ptr<int>(), M, N, K, dg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  static std::vector<int64_t> info() {
    set_attr();
    cudaFuncAttributes fa;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, bf1_kernel<MT, NT, NWR, WSK, UPW, NSTAGE, WREG>));
    int occ = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, bf1_kernel<MT, NT, NWR, WSK, UPW, NSTAGE, WREG>,
                                                                 (NW + 1) * 32, SMEM));
    return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SMEM, (int64_t)occ, (int64_t)NSTAGE};
  }
};

}  // namespace

void bf1_run(torch::Tensor ws, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t cfg, int64_t dg) {
  @@BF1_RUN@@
  TORCH_CHECK(false, "bf1: no config ", cfg);
}

std::vector<int64_t> bf1_info(int64_t cfg) {
  @@BF1_INFO@@
  TORCH_CHECK(false, "bf1: no config ", cfg);
}
"""


def _bf1_source() -> str:
    runs, infos = [], []
    for ci, (layer, mt) in enumerate(BF1_INSTS):
        N, K, NT, NWR, WSK, UPW, WREG = BF1_GEOMS[layer]
        inst = f"Bf1Inst<{mt}, {NT}, {NWR}, {WSK}, {UPW}, {_bf1_geom(layer, mt)[7]}, {str(WREG).lower()}>"
        runs.append(f"if (cfg == {ci}) {{ {inst}::run(ws, x, out, err, (int)dg); return; }}")
        infos.append(f"if (cfg == {ci}) return {inst}::info();")
    return _BF1_CUDA.replace("@@BF1_RUN@@", "\n  ".join(runs)).replace("@@BF1_INFO@@", "\n  ".join(infos))


def _bf1_build():
    """Compile the bf1 module once per process, at import (never during capture), for 9.0a only, in its OWN
    extension. Returns None (incumbent paths) on any failure; never raises."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils import cpp_extension
        ext = cpp_extension.load_inline(
            name=BF1_EXT_NAME, cpp_sources=_BF1_CPP, cuda_sources=_bf1_source(),
            functions=["bf1_run", "bf1_info"], extra_cuda_cflags=list(BULK_NVCC_FLAGS), verbose=False)
    except Exception as e:
        tail = "\n".join(str(e).splitlines()[-40:])
        _BF1_BUILD["s"] = time.perf_counter() - t0
        print(f"[bf1] build FAILED after {_BF1_BUILD['s']:.1f}s ({type(e).__name__}); log tail:\n{tail}\n"
              f"[bf1] incumbent paths kept", flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    _BF1_BUILD["s"] = time.perf_counter() - t0
    _BF1_BUILD["ok"] = True
    print(f"[build-time] bf1 load_inline compiled in {_BF1_BUILD['s']:.1f}s", flush=True)
    return ext


_BF1_EXT = _bf1_build() if torch.cuda.is_available() else None


def _bf1_cell(layer: str, m: int):
    for key in BF1_CELLS:
        if key[0] == layer and key[1] <= m <= key[2]:
            return key
    return None


def _bf1_route(layer: str, x: torch.Tensor, cfg, state: dict):
    """The bf1 cfg id serving this call, or None: (layer, cfg m) inside a BF1_CELLS range whose state flag is True,
    the layer's stream in state, n / k the geometry's, x [m, k] bf16 contiguous and 16 B aligned. cfg, shapes and
    state only, never input values."""
    key = _bf1_cell(layer, cfg["m"])
    if key is None or state.get(BF1_CELLS[key][1]) is not True or _BF1_EXT is None or ("bf1_ws_" + layer) not in state:
        return None
    N, K = BF1_GEOMS[layer][:2]
    m = cfg["m"]
    if not (cfg["n"] == N and cfg["k"] == K and x.dim() == 2 and x.shape[0] == m and x.shape[1] == K
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0):
        return None
    return BF1_INSTS.index((layer, 1 if key[2] <= 16 else 2))


def _bf1_forward(layer: str, x: torch.Tensor, state: dict, ci: int) -> torch.Tensor:
    """ONE launch: out[m, N] = x @ W.T from the state's fragment stream. The error word is passed, never read here."""
    out = torch.empty((x.shape[0], BF1_GEOMS[layer][0]), dtype=torch.bfloat16, device=x.device)
    _BF1_EXT.bf1_run(state["bf1_ws_" + layer], x, out, state["bf1_err"], ci, 0)
    return out


def _bf1_rule(rounds: list) -> tuple:
    """(route, why, pooled, pooled cand IQR, pooled incumbent IQR) on BF1_ROUNDS read-flush _cell_gate rounds."""
    base = _ouns_median([r["base"] for r in rounds])
    pc = [t for r in rounds for t in r["cand"]["times"]]
    pm = [t for r in rounds for t in r["master"]["times"]]
    pooled = max(_ouns_median(pc) - base, 0.25) / max(_ouns_median(pm) - base, 0.25)
    ic, im = _b12m_iqr(pc), _b12m_iqr(pm)
    ratios = [r["cand"]["net"] / r["master"]["net"] for r in rounds]
    if not pooled <= BF1_POOLED_MAX:
        return False, f"pooled {pooled:.3f} > {BF1_POOLED_MAX}", pooled, ic, im
    if not max(ratios) <= BF1_ROUND_MAX:
        return False, f"a round {max(ratios):.3f} > {BF1_ROUND_MAX}", pooled, ic, im
    if not ic <= BF1_POOLED_IQR_MAX * max(im, BF1_IQR_FLOOR_US):
        return False, f"pooled IQR {ic:.2f} > {BF1_POOLED_IQR_MAX}x max({im:.2f}, {BF1_IQR_FLOOR_US})", pooled, ic, im
    return True, "", pooled, ic, im


def _bf1_prepare(state: dict, t_start: float, flags=None) -> None:
    """Section (13). Sets every BF1_CELLS flag (False unless every step passes) and, per layer with a routed cell, the
    state's stream state["bf1_ws_" + layer]. flags None (state A, the judge's): per layer the stream; per instance
    regs / spill / occupancy (>= 1 CTA/SM); per cell _cell_check at m_lo, m_hi and the raced m plus a per-row check,
    BF1_ROUNDS read-flush _cell_gate rounds of kernel_fn with the flag on vs off, _bf1_rule, and a torch.profiler
    assertion of exactly one bf1_kernel. flags a dict (the A/B check's second state): streams for flagged layers only.
    Never raises."""
    for spec in BF1_CELLS.values():
        state[spec[1]] = False
    if _BF1_EXT is None:
        if flags is None:
            _BF1_SUM.append("[bf1] no module (build failed or no GPU): incumbent everywhere")
        return
    dev = state["router"].device
    state["bf1_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        for key, (rm, flag) in BF1_CELLS.items():
            if flags.get(flag):
                try:
                    if ("bf1_ws_" + key[0]) not in state:
                        state["bf1_ws_" + key[0]] = _bf1_repack(state[key[0]], key[0])
                    state[flag] = True
                except Exception as e:
                    print(f"[bf1] second state: {key} stream raised {type(e).__name__}: {e}; cell off", flush=True)
        return
    _BF1_SUM.clear()
    inst_ok = {}
    for ci, (layer, mt) in enumerate(BF1_INSTS):
        try:
            info = [int(v) for v in _BF1_EXT.bf1_info(ci)]
            g = _bf1_geom(layer, mt)
            print(f"[bf1] inst {ci} {layer} MT{mt} {BF1_GEOMS[layer][2:]}: regs={info[0]} spill={info[1]} B "
                  f"smem={info[2]} B occ={info[3]} CTA/SM NSTAGE={info[4]} grid {g[10]} x {(g[3] + 1) * 32}", flush=True)
            inst_ok[ci] = "" if (info[3] >= 1 and info[1] == 0) else f"occ {info[3]} spill {info[1]}"
        except Exception as e:
            inst_ok[ci] = f"info raised {type(e).__name__}"
    for key, (rm, flag) in BF1_CELLS.items():
        layer, lo, hi = key
        t0 = time.perf_counter()
        why, rounds, rule = "", [], None
        try:
            if t0 - t_start > BF1_START_DEADLINE_S:
                raise RuntimeError(f"skipped: deadline ({t0 - t_start:.0f}s > {BF1_START_DEADLINE_S:.0f}s)")
            ci = BF1_INSTS.index((layer, 1 if hi <= 16 else 2))
            if inst_ok.get(ci):
                raise RuntimeError(f"instance {ci}: {inst_ok[ci]}")
            N, K = BF1_GEOMS[layer][:2]
            w = state[layer]
            if ("bf1_ws_" + layer) not in state:
                t1 = time.perf_counter()
                state["bf1_ws_" + layer] = _bf1_repack(w, layer)
                torch.cuda.synchronize()
                print(f"[bf1] {layer} stream repacked in {time.perf_counter() - t1:.2f}s "
                      f"({state['bf1_ws_' + layer].numel() * 2 / 1e6:.1f} MB)", flush=True)
            st_a = dict(state)
            st_a[flag] = True
            for m in sorted({lo, hi, rm}):
                fn = (lambda xx, s=st_a, m=m: kernel_fn({"x": xx, "config": {"m": m, "n": N, "k": K, "layer": layer}}, s))
                if _bf1_route(layer, torch.empty((m, K), dtype=torch.bfloat16, device=dev),
                              {"m": m, "n": N, "k": K, "layer": layer}, st_a) != ci:
                    raise RuntimeError(f"route check failed at m={m}")
                bad = _cell_check(f"bf1 {layer} m={m}", fn, w, m, N, K, err=state["bf1_err"])
                if not bad:
                    bad = _b12m_rows(fn, w, m, N, K, state["bf1_err"])[0]
                if bad:
                    raise RuntimeError(f"m={m}: {bad}")
            for rnd in range(BF1_ROUNDS):
                st = {}
                _cell_gate(f"bf1 {layer} {lo}-{hi} r{rnd + 1}", layer, rm, N, K, flag, state, flush_mode="read",
                           stats=st, route_txt=f"route if pooled <= {BF1_POOLED_MAX}, rounds <= {BF1_ROUND_MAX}, IQR")
                rounds.append(st)
            code = int(state["bf1_err"].item())
            state["bf1_err"].zero_()
            if code:
                raise RuntimeError(f"error word {code} after the gates")
            rule = _bf1_rule(rounds)
            if not rule[0]:
                why = rule[1]
            else:
                data = {"x": torch.randn((rm, K), device=dev, dtype=torch.float32).to(torch.bfloat16),
                        "config": {"m": rm, "n": N, "k": K, "layer": layer}}
                try:
                    names = _cell_kernels(lambda: kernel_fn(data, st_a))
                    if not (len(names) == 1 and "bf1_kernel" in names[0]):
                        why = f"profiled kernels {names}: expected exactly one bf1_kernel"
                except Exception as e:
                    why = f"profiler unavailable ({type(e).__name__}: {str(e)[-120:]})"
            if not why:
                state[flag] = True
        except Exception as e:   # never raises: any surprise keeps the incumbent for this cell
            why = f"{type(e).__name__}: {str(e)[-200:]}"
            state[flag] = False
        state["bf1_err"].zero_()
        dt = time.perf_counter() - t0
        r_txt = " ".join(f"r{i + 1} {r['master']['net']:.2f}/{r['cand']['net']:.2f}="
                         f"{r['cand']['net'] / r['master']['net']:.3f}" for i, r in enumerate(rounds))
        if rule is not None:
            r_txt += f" | pooled {rule[2]:.3f} iqr{rule[4]:.2f}/{rule[3]:.2f}"
        _BF1_SUM.append(f"[bf1] {layer} m{lo}-{hi}@{rm} inc/bf1 us: {r_txt or 'not timed'} -> "
                        + ("ROUTE" if state[flag] else f"NO-ROUTE ({why[:100]})") + f" {dt:.1f}s")
        _decision(f"PATH CHOSEN bf1 {layer} m={lo}..{hi}: " + ("bf1_kernel" if state[flag] else "incumbent")
                  + (f" [pooled {rule[2]:.3f}]" if rule is not None else "") + ("" if state[flag] else f"; {why[:160]}"))
    for layer in BF1_GEOMS:
        if not any(state[flag] for (lyr, lo, hi), (rm, flag) in BF1_CELLS.items() if lyr == layer):
            state.pop("bf1_ws_" + layer, None)
    torch.cuda.synchronize()


def _bf1_summary(prepare_s: float) -> None:
    b = _BF1_BUILD["s"]
    print(f"[bf1] SUMMARY build {'n/a' if b is None else f'{b:.1f}s'} ({BF1_EXT_NAME}, "
          f"{'ok' if _BF1_BUILD['ok'] else 'FAILED'}); prepare {prepare_s:.1f}s; route at pooled <= {BF1_POOLED_MAX}, "
          f"rounds <= {BF1_ROUND_MAX}; rounds as inc/bf1=ratio", flush=True)
    for line in _BF1_SUM:
        print(line[:300], flush=True)


NEW_CELL_FLAGS = NEW_CELL_FLAGS + BF1_FLAGS   # section (13)'s cells join the A/B check and the FINAL FLAGS line


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
    state[KHALF_FLAG] = False    # pooler m == 128 K-halves: off until state A's selector (or B's inherit) says so
    for name in ROUTER_PDL_FLAGS:   # router m > 8 zero kernel + PDL: off until state A's selector (or B's inherit)
        state[name] = False
    t = tick("weight splits + NN copies", t)
    # Stage choice, side-by-side diagnostics and warm-up on synthetic inputs.
    _router_prepare(state)
    t = tick("router triton warm-up", t)
    if flags is None:
        # R-PDL: feature probe, PTX order checks and the warm-up of every m class in zmodes 0 / 1 / 2 (state A),
        # before any capture.
        _router_pdl_probe(state)
        t = tick("r-pdl probe + warm-up", t)
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
        # (8b) Router 9 <= m: zero kernel + PDL router per m class (own flags), behind check + race + gate + profile.
        _router_pdl_prepare(state)
        t = tick("r-pdl cells (check + race + 2 read-flush gates + profile)", t)
        # (9) m == 1 bulk row-owner cells A / A2 / A3 and qkv_gate m == 128 cuBLASLt plan B, each
        # behind its own canary, gate and kernel-name assertion (after master's own work).
        _bulk_prepare(state)
        t = tick("bulk cells A/A2/A3 (setup + repack + canary + gate + profile)", t)
        # (10) m == 1 bulk12 cells A12 / A2_12 / A3_12 (12-bit weights), gated against today's route.
        _b12_prepare(state)
        t = tick("bulk12 cells A12/A2_12/A3_12 (setup + encode + round trips + canary + gate + profile)", t)
        # (12) m == 8 bulk12 on TMA boxes + batched escape gather (own flags), raced against today's m == 8 route.
        _b12m_prepare(state, t_all)
        t = tick("b12m cells P8/Q8 + P16 diag (codes + checks + 3 read-flush rounds + write + profile)", t)
        _lt_prepare(state)
        t = tick("lt cell B (heuristic + canary + gate + profile)", t)
        # (11) pooler m == 128 K-halves, fp32 partials + Triton add-cast (own flag), after every other cell.
        _khalf_prepare(state, t_all)
        t = tick("pooler k-halves cell K (probe + canary + check + 2 read-flush gates + profile)", t)
        # (13) BF1: fragment-ordered BF16 stream at m = 8..32 (own flags), raced against every incumbent above.
        _bf1_prepare(state, t_all)
        t = tick("bf1 cells (repack + checks + 3 read-flush rounds + profile)", t)
    else:
        state["ouns_cl"] = flags["ouns_cl"]
        state["router_cluster_ok"] = flags["router_cluster_ok"]
        state["nn_ok"] = flags["nn_ok"]
        state["nn_p256_ok"] = flags["nn_p256_ok"]
        state[KHALF_FLAG] = bool(flags.get(KHALF_FLAG, False))   # no per-state buffers (pooler_t is per state)
        for name in ROUTER_PDL_FLAGS:   # no per-state buffers; warm this state's routed kwargs before any capture
            state[name] = bool(flags.get(name, False))
        if any(state[name] for name in ROUTER_PDL_FLAGS):
            _router_pdl_warm(state, (0, 1))
        # Second state: its own repacked weights / error word / Lt workspace; flags inherited.
        _bulk_prepare(state, flags=flags)
        _b12_prepare(state, flags=flags)
        _b12m_prepare(state, t_all, flags=flags)
        _lt_prepare(state, flags=flags)
        _bf1_prepare(state, t_all, flags=flags)
        t = tick("bulk / lt / bf1 per-state buffers (flags inherited)", t)
    print(f"[prepare-time] state build ({'A' if flags is None else 'B'}) total: "
          f"{time.perf_counter() - t_all:.2f}s", flush=True)
    return state


def _ab_flags(state: dict) -> dict:
    flags = {"ouns_cl": state["ouns_cl"], "router_cluster_ok": state["router_cluster_ok"],
             "nn_ok": state["nn_ok"], "nn_p256_ok": state["nn_p256_ok"]}
    for name in NEW_CELL_FLAGS:
        flags[name] = state[name]
    return flags


def _ab_component(layer: str, m: int, st=None) -> str:
    """Which stacked component serves this cell ("" = master's path, nothing to switch off). Each
    new cell maps to its OWN flag, so a miss there cannot switch off nn_ok or any other cell. A
    bulk12 cell that is on in st serves its m == 1 cell ahead of the raw bulk cell."""
    bf1_key = _bf1_cell(layer, m)
    if st is not None and bf1_key is not None and st.get(BF1_CELLS[bf1_key][1]) is True:
        return BF1_CELLS[bf1_key][1]   # section (13): a miss turns off only this bf1 cell
    if st is not None and layer == "pooler" and m == KHALF_M and st.get(KHALF_FLAG):
        return KHALF_FLAG   # checked first: a miss turns off only this cell, never nn_ok (unpooler m=128 shares it)
    if st is not None and layer in B12_CELLS and m == B12_M and st.get(B12_CELLS[layer][5]):
        return B12_CELLS[layer][5]
    b12m_spec = B12M_CELLS.get((layer, m))
    if st is not None and b12m_spec is not None and b12m_spec[7] and st.get(b12m_spec[6]) is True:
        return b12m_spec[6]   # section (12): a miss turns off only this m == 8 cell
    if layer in BULK_CELLS and m == BULK_M:
        return BULK_CELLS[layer][7]
    if layer == "qkv_gate" and m == LT_M:
        return LT_FLAG
    if layer == "router" and m <= ROUTER_CLUSTER_MAX_M:
        return "router_cluster_ok"
    if st is not None and layer == "router" and _router_pdl_flag(m) and st.get(_router_pdl_flag(m)):
        return _router_pdl_flag(m)   # m > 8 R-PDL cell: a miss turns off only that class's flag
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
        fl = _ab_flags(state_a)
        fl["_lt_algo"] = state_a["lt_algo_qkv128"]   # the B cell's prepare()-chosen algo row (CPU)
        fl["_b12_inst"] = {layer: state_a.get("b12_inst_" + layer) for layer in B12_CELLS}
        state_b = _prepare_state(weights, flags=fl)
    except Exception as e:
        print(f"[ab] second-state build failed ({type(e).__name__}: {e}); NN, router cluster, "
              f"unpooler cluster, bulk and Lt cells OFF", flush=True)
        for st in (state_a,):
            st["nn_ok"], st["router_cluster_ok"], st["ouns_cl"] = False, False, 0
            st["nn_p256_ok"] = False
            for name in NEW_CELL_FLAGS:
                st[name] = False
        return
    states = {"A": state_a, "B": state_b}
    dev = state_a["router"].device
    g = torch.Generator(device=dev)
    g.manual_seed(97531)
    bad = {}
    wf = {}
    for layer, n, k, m in AB_CASES:
        comp = _ab_component(layer, m, state_a)
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
            _decision(f"[ab] MISS {layer} m={m} n={n}: {type(e).__name__}: {e}"
                      + (f" -> switching {comp} off in both states" if comp else
                         " (master path cell: nothing to switch off)"))
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
    _DECISION_LINES.clear()
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
          f"nn_ok={state['nn_ok']} nn_p256_ok={state['nn_p256_ok']} "
          + " ".join(f"{name}={state[name]}" for name in NEW_CELL_FLAGS)
          + f"; prepare total {time.perf_counter() - t0:.1f}s", flush=True)
    # The judge keeps a 16 KB log tail: every decision line again, last, so none is lost to the earlier noise.
    print(f"[prepare] DECISIONS ({len(_DECISION_LINES)} lines, re-printed for the log tail):", flush=True)
    for line in _DECISION_LINES:
        print(f"  {line}", flush=True)
    _DECISION_LINES.clear()
    _b12m_summary(time.perf_counter() - t0)   # section (12)'s <= 2 KB block, the very last lines of prepare()
    _bf1_summary(time.perf_counter() - t0)    # section (13)'s block, after it
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    # Shape gate (cfg layer / m / n and shape preconditions only; never on input values). Every
    # dispatch flag (ouns_cl, router_cluster_ok, nn_ok) is read from `state`.
    if layer == "unpooler" and _ouns_route_ok(x, state["unpooler"], cfg, state):
        return _ouns_forward(x, state["unpooler"], state["ouns_cl"])
    bf1_ci = _bf1_route(layer, x, cfg, state)   # section (13): pooler / unpooler / qkv_gate / o, m = 8..32, per cell flag
    if bf1_ci is not None:
        return _bf1_forward(layer, x, state, bf1_ci)
    if _b12_route_ok(layer, x, cfg, state):    # pooler / qkv_gate / o at m == 1 on 12-bit weights
        return _b12_forward(layer, x, state)
    b12m_arm = _b12m_route(layer, x, cfg, state)   # pooler / qkv_gate at exactly m == 8 on 12-bit weights (TMA)
    if b12m_arm is not None:
        return _b12m_forward(layer, x, state, b12m_arm)
    if _bulk_route_ok(layer, x, cfg, state):   # pooler / qkv_gate / o at m == 1, per-cell flag
        return _bulk_forward(layer, x, state)
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
        if layer == "pooler" and _khalf_route_ok(x, cfg, state):   # m == 128: bmm fp32 K-halves + add-cast
            return _khalf_forward(x, state)
        if (layer in NN_GATED_LAYERS and cfg["n"] == state[layer + "_t"].shape[1]
                and ((state["nn_ok"] and NN_GATED_MIN_M <= cfg["m"] <= NN_GATED_MAX_M)
                     or (layer == "pooler" and state["nn_p256_ok"]
                         and NN_GATED_MAX_M < cfg["m"] <= NN_POOLER_MAX_M))):
            # NN layout: x [m, k] @ W_t [k, n] at the full weight width (cfg-only gate).
            return torch.mm(x, state[layer + "_t"])
        if layer == "qkv_gate" and _lt_route_ok(x, cfg, state):   # m == 128, prepare()-fixed Lt algo
            return _lt_forward(x, state["qkv_gate"], state["lt_algo_qkv128"], state["lt_ws_qkv128"])
        # The unpooler's slice is its first n rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)
    if _router_cluster_route(x, cfg, state):   # 1 <= m <= 8, n == 256: one launch, no memset
        return _router_cluster_forward(x, state)
    hi, lo = state["router_hi"], state["router_lo"]
    path = _router_path(x, hi, cfg["n"], state["router_stages"])
    zmode = _router_zmode(x.shape[0], state)   # m > 8: the class's R-PDL flag (shape + state only)
    if path == "swapab":   # m <= 32
        return _router_forward(x, hi, lo, state["router_stages"][_router_config(x.shape[0])], zmode=zmode)
    if path == "hilo":     # m > 32: master's forward
        return _router_hilo_forward(x, hi, lo, zmode=zmode)
    # Shape outside the kernel's preconditions (never for this problem's router): exact fp32 path.
    return F.linear(x.float(), state["router"][: cfg["n"]])
