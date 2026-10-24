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
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    if layer != "router":
        # The unpooler's slice is its first n rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)
    hi, lo = state["router_hi"], state["router_lo"]
    if _router_shape_ok(x, hi) and cfg["n"] == hi.shape[0]:
        return _router_forward(x, hi, lo)
    # Shape outside the kernel's preconditions (never for this problem's router): exact fp32 path.
    return F.linear(x.float(), state["router"][: cfg["n"]])
