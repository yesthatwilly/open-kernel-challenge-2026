# okc: problem=muse_prefill_layer
"""SGLang's decoder layer pipeline (baseline.py, MuseSpark1xDecoderLayer.forward at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f) ported module for module at the prefill shapes, with two
deliberate changes (hypothesis sglang-pipeline-parity):

  1. The local-expert id remap. SGLang's path runs three small kernels on the int32 [T, 8] routing
     ids: (topk_ids - local_start), % 256, then a gather through the expert_map table (-1 for
     experts on other GPUs). Here it is one Triton elementwise kernel: r = (ids - local_start) & 255,
     local_ids = r if r < 32 else -1. local_start is read on the device. The expert_map table is gone.
  2. The two tokens <= 8 branches (the FP32 F.linear router with the FP32 router input written by
     the attention boundary, and the sliced unpooler + zero pad) are removed: every prefill case has
     q_len >= 200. The router always runs the BF16 hi/lo pair with FP32 accumulation, and the
     unpooler always uses the whole weight. kernel_fn asserts tokens > 8 (a shape check).

Everything else is SGLang's pipeline as baseline.py composes it:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT)
  router                           two BF16 GEMMs on a hi/lo split of the weight
                                   (MuseSpark1xRouter above 8 tokens); on a side stream,
                                   overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two grouped W4A16 GEMMs: SGLang's
                                   moe_wna16_marlin_gemm up to 4096 tokens; above that (hypotheses
                                   experts-hybrid-marlin-turn-triton-first-turn,
                                   experts-triton-w4a16-inkernel-fp4 and -swapab-reglhs) a Triton
                                   fused_moe_kernel, computed swap-AB, that dequantizes the
                                   NVFP4 tiles (nibble-reordered in prepare) in registers as the
                                   dot's LHS, in place of moe_wna16_marlin_gemm), silu_and_mul, Triton
                                   post-expert norm weighted sum; then the all-reduce (identity)
  unpooler                         the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang (as baseline.py does):
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity."""

from typing import TypedDict

import copy
import hashlib
import os
import re
import time
import types

import nvfp4
import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale: 1 in the scored cases, 1/64 in some tests
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# SGLang's MuseSpark1xRouter and sliced unpooler switch paths at this many tokens; every prefill
# case is above it, so only the above-8 paths are kept.
SMALL_BATCH_MAX_TOKENS = 8
# Expert path by shape (hypothesis experts-hybrid-marlin-turn-triton-first-turn): up to this many
# tokens (turn_1 753, turn_4 3012, every test) SGLang's two Marlin GEMMs; above it (the first turn's
# 16384) the Triton NVFP4 grouped GEMM. Not a measured crossover: T in (4096, 16384) is unscored and
# untuned.
MARLIN_MAX_TOKENS = 4096
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
# Grouped NVFP4 GEMM (swap-AB) tile configs, chosen from the shape (tokens) only. About T of the
# T * 8 routes land on this GPU's 32 experts, so an expert gets ~T/32 rows: the token tile BLOCK_T
# (the MMA N side) covers one expert in one tile at the turn shapes (~24 rows at turn_1, ~94 at
# turn_4) and four tiles at the first turn's ~512. BLOCK_W is the MMA M side (rows of the expert's
# N). moe_align_block_size pads to BLOCK_T. Entries are (max tokens, config); the last one has no
# bound. Above 4096 tokens (first turn) BLOCK_T 256 feeds each in-register dequantized W tile to twice
# the MMA columns (wgmma m64n256 per warpgroup); its two [64, 256] activation slabs per K step leave
# room for 2 smem stages, not 3.
# BLOCK_K is fixed by the nibble order prepare() stores (EXPERT_BLOCK_K): each byte of a 128-k chunk
# holds k0 + i (low nibble) and k0 + 64 + i (high nibble).
EXPERT_BLOCK_K = 128
EXPERT_CONFIGS = (
    (1024, dict(BLOCK_T=64, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=8, num_warps=4, num_stages=3)),
    (4096, dict(BLOCK_T=128, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=8, num_warps=8, num_stages=3)),
    (None, dict(BLOCK_T=256, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=4, num_warps=8, num_stages=2)),
)
SF_BLOCK = tl.constexpr(16)  # nvfp4.BLOCK: k per e4m3 block scale
# The kernel's W tile is E2M1 * block scale * 2^-7 (exact in BF16); the epilogue multiplies by
# global scale * 2^7. 2^119 moves the e = 0 subnormal placement (E2M1 * 2^-126) to that range.
SCALE_BIAS = tl.constexpr(2.0 ** 119)
EPILOGUE_BIAS = tl.constexpr(2.0 ** 7)
# Write zeros for the m-blocks of non-local routes (expert -1), as SGLang's Triton guard does. Only
# the second GEMM's output reaches the weighted sum (the first GEMM's -1 rows only feed the
# second GEMM's -1 blocks, which never read A), so only that GEMM pays for it.
ZERO_NONLOCAL_W2 = True
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0
REMAP_BLOCK = 1024


@triton.jit
def _local_expert_ids_kernel(ids_ptr, start_ptr, out_ptr, n, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    """out = (ids - local_start) mod 256 where that is < LOCAL, else -1 (SGLang's expert_map on
    this EP rank). int32 in, int32 out; & 255 is the non-negative mod 256 in two's complement."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    ids = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    start = tl.load(start_ptr).to(tl.int32)
    r = (ids - start) & 255
    out = tl.where(r < LOCAL, r, -1).to(tl.int32)
    tl.store(out_ptr + offs, out, mask=mask)


def _local_expert_ids(topk_ids, local_start):
    ids = topk_ids.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    n = ids.numel()
    _local_expert_ids_kernel[(triton.cdiv(n, REMAP_BLOCK),)](
        ids, local_start, out, n, LOCAL=LOCAL_EXPERTS, BLOCK=REMAP_BLOCK, num_warps=4)
    return out


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


@triton.jit
def _e2m1_bits_to_bf16(bits):
    """int32 BF16 bit patterns -> bfloat16 (a bitcast)."""
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["num_valid_tokens", "EM"])
def _moe_nvfp4_gemm_kernel(
    a_ptr, b_ptr, sf_ptr, gs_ptr, c_ptr, sorted_ids_ptr, expert_ids_ptr, n_padded_ptr,
    num_valid_tokens, EM,
    N: tl.constexpr, K: tl.constexpr, TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_T: tl.constexpr,
    ZERO_NONLOCAL: tl.constexpr,
):
    """SGLang's fused_moe_kernel (fused_moe_triton/fused_moe.py) in its W4A16 form, computed
    transposed (swap-AB): acc[n, tokens] += W[n, k] @ A[tokens, k].T, so the weight tile,
    dequantized from the stored NVFP4 codes in registers, is the dot's LEFT operand (the one sm_90
    wgmma reads from registers) and only the bf16 activation tile goes through shared memory.

      W codes   uint8 [E, N, K/2], N-major, nibble-reordered by prepare(): in each BLOCK_K chunk at
                k0, byte i holds k0 + i in the low nibble and k0 + BLOCK_K/2 + i in the high one
      W scales  float8_e4m3fn [E, N, K/16], one per 16 k (original order)
      W global  float32 [E], applied once to the fp32 accumulator in the epilogue

    A is bf16 [M, K] (row = sorted id // TOP_K), C is bf16 [num_valid_tokens, N] (row = sorted
    id). No routed-weight multiply. All strides follow from N and K (contiguous tensors)."""
    HALF: tl.constexpr = BLOCK_K // 2
    GROUPS_HALF: tl.constexpr = HALF // SF_BLOCK
    pid = tl.program_id(axis=0)
    num_pid_t = tl.cdiv(EM, BLOCK_T)
    num_pid_w: tl.constexpr = N // BLOCK_W
    # Token tiles fastest within a group, so the token tiles of one (expert, weight tile) run
    # adjacent and the weight tile is re-read from L2; the group bounds the activation rows in flight.
    num_pid_in_group = GROUP_SIZE_T * num_pid_w
    group_id = pid // num_pid_in_group
    first_pid_t = group_id * GROUP_SIZE_T
    group_size_t = min(num_pid_t - first_pid_t, GROUP_SIZE_T)
    pid_t = first_pid_t + ((pid % num_pid_in_group) % group_size_t)
    pid_w = (pid % num_pid_in_group) // group_size_t

    # The grid covers the static bound of sorted_ids; blocks past the real padded count exit here.
    num_tokens_post_padded = tl.load(n_padded_ptr)
    if pid_t * BLOCK_T >= num_tokens_post_padded:
        return

    offs_token_id = pid_t * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs_token = tl.load(sorted_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    # [BLOCK_W, BLOCK_T]: the transposed output tile.
    c_ptrs = c_ptr + offs_token[None, :] * N + offs_w[:, None]

    off_expert = tl.load(expert_ids_ptr + pid_t).to(tl.int64)
    if off_expert == -1:
        # Routes to experts on other GPUs.
        if ZERO_NONLOCAL:
            tl.store(c_ptrs, tl.zeros((BLOCK_W, BLOCK_T), dtype=c_ptr.dtype.element_ty),
                     mask=token_mask[None, :])
        return

    offs_h = tl.arange(0, HALF)
    # [HALF, BLOCK_T] activation slab, K-contiguous down the rows: k0 + i for the low half; the high
    # half is the same pointers + HALF.
    a_ptrs = a_ptr + (offs_token[None, :] // TOP_K) * K + offs_h[:, None]
    # [BLOCK_W, HALF] code bytes of one BLOCK_K chunk.
    b_ptrs = b_ptr + off_expert * (N * (K // 2)) + offs_w[:, None] * (K // 2) + offs_h[None, :]
    # [BLOCK_W, BLOCK_K // 16] compact scale tile: the chunk's 8 block scales per row, contiguous.
    s_ptrs = (sf_ptr + off_expert * (N * (K // SF_BLOCK)) + offs_w[:, None] * (K // SF_BLOCK)
              + tl.arange(0, BLOCK_K // SF_BLOCK)[None, :])

    acc = tl.zeros((BLOCK_W, BLOCK_T), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        a_lo = tl.load(a_ptrs, mask=token_mask[None, :], other=0.0)
        a_hi = tl.load(a_ptrs + HALF, mask=token_mask[None, :], other=0.0)
        b = tl.load(b_ptrs).to(tl.int32)
        s = tl.load(s_ptrs)
        # Block scale * 2^119 (exact: e4m3 has 4 significant bits and 448 * 2^119 < bf16 max), once
        # per scale on the compact tile, then split into the low half's groups 0..3 and the high
        # half's 4..7 and broadcast to 16 k each (group g covers half-index i with i // 16 == g).
        sb = (s.to(tl.float32) * SCALE_BIAS).to(tl.bfloat16)
        sb_lo, sb_hi = tl.split(tl.permute(tl.reshape(sb, (BLOCK_W, 2, GROUPS_HALF)), (0, 2, 1)))
        sb_lo = tl.reshape(tl.broadcast_to(sb_lo[:, :, None], (BLOCK_W, GROUPS_HALF, SF_BLOCK)), (BLOCK_W, HALF))
        sb_hi = tl.reshape(tl.broadcast_to(sb_hi[:, :, None], (BLOCK_W, GROUPS_HALF, SF_BLOCK)), (BLOCK_W, HALF))
        # E2M1 -> BF16 by bit placement: the 3 magnitude bits go to bf16 bits 8..6 (exponent low
        # bits and the top mantissa bit), the sign to bit 15. That is E2M1 * 2^-126 exactly,
        # e = 0 codes landing on bf16 subnormals (0.5 -> 2^-127). The product E2M1 * scale * 2^-7
        # has at most 6 significant bits, so it is exact in BF16.
        w_lo = _e2m1_bits_to_bf16(((b & 0x07) << 6) | ((b & 0x08) << 12)) * sb_lo  # k0 + i
        w_hi = _e2m1_bits_to_bf16(((b & 0x70) << 2) | ((b & 0x80) << 8)) * sb_hi   # k0 + HALF + i
        acc = tl.dot(w_lo, a_lo, acc)
        acc = tl.dot(w_hi, a_hi, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K // 2
        s_ptrs += BLOCK_K // SF_BLOCK

    # The per-expert global scale (and the 2^7 the W tile left out), once per output.
    gs = tl.load(gs_ptr + off_expert) * EPILOGUE_BIAS
    tl.store(c_ptrs, (acc * gs).to(c_ptr.dtype.element_ty), mask=token_mask[None, :])


def _expert_config(tokens):
    for bound, cfg in EXPERT_CONFIGS:
        if bound is None or tokens <= bound:
            return cfg
    raise AssertionError("unreachable")


def _nvfp4_moe_gemm(a, w, sf, gs, out, sorted_ids, expert_ids, n_padded, top_k, num_valid, cfg, zero_nonlocal,
                    report=None):
    """out[sorted id] = a[sorted id // top_k] @ dequant(w[expert of its block]).T, on the
    token blocks moe_align_block_size laid out (cfg's BLOCK_T). w is the nibble-reordered bank."""
    n, k = w.shape[1], w.shape[2] * 2
    assert a.shape[1] == k and out.shape[1] == n and sf.shape[2] * nvfp4.BLOCK == k
    assert cfg["BLOCK_K"] == EXPERT_BLOCK_K and nvfp4.BLOCK == SF_BLOCK.value
    assert n % cfg["BLOCK_W"] == 0 and k % cfg["BLOCK_K"] == 0
    em = sorted_ids.shape[0]
    grid = (triton.cdiv(em, cfg["BLOCK_T"]) * (n // cfg["BLOCK_W"]),)
    compiled = _moe_nvfp4_gemm_kernel[grid](
        a, w, sf, gs, out, sorted_ids, expert_ids, n_padded, num_valid, em,
        N=n, K=k, TOP_K=top_k, BLOCK_T=cfg["BLOCK_T"], BLOCK_W=cfg["BLOCK_W"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_SIZE_T=cfg["GROUP_SIZE_T"], ZERO_NONLOCAL=zero_nonlocal,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if report is not None:  # prepare() only: register / spill / smem use of this variant, for the log
        print(f"[expert gemm] {report} BLOCK_T={cfg['BLOCK_T']} BLOCK_W={cfg['BLOCK_W']} "
              f"warps={cfg['num_warps']} stages={cfg['num_stages']}: "
              f"n_regs={getattr(compiled, 'n_regs', '?')} n_spills={getattr(compiled, 'n_spills', '?')} "
              f"shared={getattr(compiled, 'shared', getattr(getattr(compiled, 'metadata', None), 'shared', '?'))}",
              flush=True)
    return out


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _reorder_nibbles(packed):
    """Lossless repack of uint8 [E, N, K/2] E2M1 codes (element 2i low nibble, 2i + 1 high) into
    the kernel's order: in each EXPERT_BLOCK_K chunk at k0, byte i holds k0 + i (low nibble) and
    k0 + EXPERT_BLOCK_K/2 + i (high nibble). The same codes, permuted; a new tensor (the input is
    not modified). Each expert is checked to invert exactly."""
    experts, n, kh = packed.shape
    k = kh * 2
    assert k % EXPERT_BLOCK_K == 0
    chunks, half = k // EXPERT_BLOCK_K, EXPERT_BLOCK_K // 2
    out = torch.empty_like(packed)
    for i in range(experts):
        p = packed[i]
        codes = torch.stack((p & 0x0F, p >> 4), dim=-1).reshape(n, chunks, 2, half)  # k order
        q = codes[:, :, 0, :] | (codes[:, :, 1, :] << 4)
        out[i] = q.reshape(n, kh)
        back = torch.stack((q & 0x0F, q >> 4), dim=-2).reshape(n, kh, 2)
        assert torch.equal(back[..., 0] | (back[..., 1] << 4), p), f"nibble reorder not lossless (expert {i})"
    return out


def _expert_bank(weights):
    """The local experts: E2M1 codes nibble-reordered for the kernel (new tensors, the weights are
    untouched), e4m3 block scales and fp32 global scales as stored, contiguous."""
    bank = {name: weights[name].contiguous() for name in ("w13_sf", "w13_gs", "w2_sf", "w2_gs")}
    for name in ("w13", "w2"):
        bank[name] = _reorder_nibbles(weights[name].contiguous())
    return bank


# The prepare() check covers at least one full 128 (rows) x 128 (k) tile: one whole BLOCK_K chunk,
# i.e. both nibble halves and all eight scale groups, every N row. A config's check uses
# max(CHECK_ROWS, BLOCK_T) rows, so the valid region is whole token blocks at every BLOCK_T.
CHECK_ROWS = 128
CHECK_K = EXPERT_BLOCK_K


def _check_expert_gemms(bank, weights, zero_w2=ZERO_NONLOCAL_W2):
    """Untimed. Runs every kernel variant kernel_fn can launch (so each is compiled before capture)
    on a synthetic case built from the weights alone, and asserts its in-kernel dequant of the
    reordered bank equals nvfp4.dequantize of the ORIGINAL packing. Row r of A is one-hot at
    k0 + r % 128, so output row r is column k0 + r % 128 of the dequantized expert, through the
    real pointer arithmetic, K loop, nibble order, scale indexing and transposed epilogue. One -1
    block and one block past the padded count must leave their rows untouched (or zero, for the -1
    block when zero-storing)."""
    device = bank["w13"].device
    e = LOCAL_EXPERTS - 1  # the largest expert offsets
    for name, top_k, zero_nonlocal in (("w13", TOP_K, False), ("w2", 1, zero_w2)):
        w, sf, gs = bank[name], bank[name + "_sf"], bank[name + "_gs"]
        orig = weights[name]
        n, k = w.shape[1], w.shape[2] * 2
        k0 = k // 2 + CHECK_K  # a chunk in the middle of the K loop
        assert k0 % EXPERT_BLOCK_K == 0
        ref = nvfp4.dequantize(orig[e, :, k0 // 2:(k0 + CHECK_K) // 2], sf[e, :, k0 // 16:(k0 + CHECK_K) // 16], gs[e])
        codes = nvfp4.unpack_e2m1(orig[e, :, k0 // 2:(k0 + CHECK_K) // 2])
        for _, cfg in EXPERT_CONFIGS:
            bt = cfg["BLOCK_T"]
            check_rows = max(CHECK_ROWS, bt)        # whole token blocks of valid rows
            assert check_rows % bt == 0
            rows = torch.arange(check_rows, device=device)
            want = ref[:, rows % CHECK_K].t()       # [check_rows, n]
            want_codes = codes[:, rows % CHECK_K].t()
            total = check_rows + 2 * bt             # + one -1 block + one block past n_padded
            x = torch.zeros((total, k), dtype=torch.bfloat16, device=device)
            x[rows, k0 + rows % CHECK_K] = 1.0
            sorted_ids = (torch.arange(total, device=device, dtype=torch.int32) * top_k).contiguous()
            expert_ids = torch.full((triton.cdiv(total, bt),), e, dtype=torch.int32, device=device)
            expert_ids[check_rows // bt] = -1
            expert_ids[check_rows // bt + 1] = 0    # valid expert, but past n_padded: must not run
            n_padded = torch.tensor([check_rows + bt], dtype=torch.int32, device=device)
            out = torch.full((total * top_k, n), float("nan"), dtype=torch.bfloat16, device=device)
            _nvfp4_moe_gemm(x, w, sf, gs, out, sorted_ids, expert_ids, n_padded, top_k, total * top_k, cfg,
                            zero_nonlocal, report=name)
            torch.cuda.synchronize()
            got = out[0:check_rows * top_k:top_k].float()
            bad = ((got - want).abs() > want.abs() * 2.0 ** -7) | got.isnan()
            if bool(bad.any()):
                halves = bool((want_codes[bad].abs() == 0.5).all())
                raise AssertionError(
                    f"NVFP4 in-kernel dequant mismatch: {name} BLOCK_T={bt}, {int(bad.sum())} of {bad.numel()} "
                    f"values, max err {float((got - want).abs().max())}"
                    + (" (only E2M1 +-0.5 codes: bf16 subnormals flushed to zero)" if halves else ""))
            nonlocal_rows = out[check_rows * top_k:(check_rows + bt) * top_k:top_k].float()
            if zero_nonlocal:
                assert bool((nonlocal_rows == 0).all()), f"{name} BLOCK_T={bt}: -1 block not zeroed"
            else:
                assert bool(nonlocal_rows.isnan().all()), f"{name} BLOCK_T={bt}: -1 block was written"
            past = out[(check_rows + bt) * top_k::top_k].float()
            assert bool(past.isnan().all()), f"{name} BLOCK_T={bt}: block past n_padded was written"


# prepare()'s end-to-end expert check: a synthetic case above MARLIN_MAX_TOKENS (so _experts_triton
# runs the first turn's BLOCK_T 256 config and _experts_marlin its block_size_m 64), compared under the
# judge's metric with a bound (1%) below the tightest reference tolerance (3%), and the judge's row
# rule (5x the bound, near-zero rows against a tenth of the average row).
E2E_CHECK_TOKENS = 8192
E2E_CHECK_TOL = 2e-2  # under the judge's tightest 3%; Triton-vs-Marlin is two bf16 pipelines, not a reference
E2E_ROW_FACTOR = 5.0


def _check_experts_e2e(state, device):
    """Untimed. One call each of _experts_triton (state['bank']) and _experts_marlin (state['marlin'])
    on synthetic data built here (random bf16 rows, random distinct top-8 experts of 256 mapped to this
    GPU's 32 as kernel_fn maps them, random normalized weights): the same inputs through both banks
    must agree. Covers both banks' wiring and any scale corruption between them."""
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    t = E2E_CHECK_TOKENS
    x = torch.randn((t, D_EXPERT), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    ids = torch.rand((t, N_EXPERTS), generator=gen, device=device).topk(TOP_K, dim=1).indices.to(torch.int32)
    local_ids = torch.where(ids < LOCAL_EXPERTS, ids, torch.full_like(ids, -1)).contiguous()
    topk_weights = torch.rand((t, TOP_K), generator=gen, device=device, dtype=torch.float32) + 0.1
    topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).contiguous()
    got = _experts_triton(x, topk_weights, local_ids, state).float()
    want = _experts_marlin(x, topk_weights, local_ids, state).float()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(got).all()) and bool(torch.isfinite(want).all()), "expert e2e check: NaN/Inf"
    err = float((got - want).norm() / want.norm().clamp_min(1e-30))
    row_ref = want.norm(dim=-1)
    row_err = (got - want).norm(dim=-1) / torch.maximum(row_ref, 0.1 * row_ref.mean()).clamp_min(1e-30)
    worst = float(row_err.max())
    print(f"[expert e2e] T={t} triton vs marlin: rel L2 {err:.3e}, worst row {worst:.3e} "
          f"({int((local_ids >= 0).sum())} local routes)", flush=True)
    assert err <= E2E_CHECK_TOL, f"expert e2e check: triton vs marlin rel L2 {err:.3e} > {E2E_CHECK_TOL:g}"
    assert worst <= E2E_ROW_FACTOR * E2E_CHECK_TOL, (
        f"expert e2e check: worst row {worst:.3e} > {E2E_ROW_FACTOR:g} x {E2E_CHECK_TOL:g}")


def _weighted_sum_skips_remote(state, device):
    """Untimed, prepare() only (hypothesis w2-nonlocal-zero-store-off-first-turn). True when
    fused_post_expert_norm_weighted_sum never reads the y rows of remote slots: its output is
    bitwise equal and finite whether those rows hold NaN or zero. Then the w2 zero-store of -1 token
    tiles is dead work and can be turned off."""
    try:
        gen = torch.Generator(device=device).manual_seed(20261024)
        t = 4096
        ids = torch.randint(0, N_EXPERTS, (t, TOP_K), generator=gen, device=device, dtype=torch.int32)
        local_ids = torch.where(ids < LOCAL_EXPERTS, ids, torch.full_like(ids, -1)).contiguous()
        w = torch.rand((t, TOP_K), generator=gen, device=device, dtype=torch.float32)
        w = (w / w.sum(dim=1, keepdim=True)).contiguous()
        y = torch.randn((t, TOP_K, D_EXPERT), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
        remote = local_ids < 0
        outs = []
        for fill in (float("nan"), 0.0):
            yy = y.clone()
            yy[remote] = fill
            outs.append(fused_post_expert_norm_weighted_sum(
                yy, w, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
                slot_expert_ids=local_ids))
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(outs[0].float()).all()) and torch.equal(outs[0], outs[1])
        print(f"[w2 zero-store] {'OFF (weighted sum skips remote slots)' if ok else 'ON: weighted sum reads remote rows'}",
              flush=True)
        return ok
    except Exception as e:
        print(f"[w2 zero-store] ON: probe raised {e!r}", flush=True)
        return False


def prepare(weights):
    t_prepare0 = time.perf_counter()
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        # router-hilo-one-gemm-n512: hi | lo side by side, bf16 [8192, 512], so one GEMM gives both halves.
        "router_hilo_t": torch.cat((hi.t(), lo.t()), dim=1).contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    # The Triton bank first (its scales alias the weights), and its check: compiles every
    # expert-GEMM variant and checks its dequant against nvfp4.dequantize.
    state["bank"] = _expert_bank(weights)
    state["zero_nonlocal_w2"] = not _weighted_sum_skips_remote(state, device)
    _check_expert_gemms(state["bank"], weights, state["zero_nonlocal_w2"])
    torch.cuda.synchronize()
    scale_names = ("w13_sf", "w13_gs", "w2_sf", "w2_gs")
    scales_before = {name: state["bank"][name].clone() for name in scale_names}
    # Then the Marlin bank, from copies, so SGLang's repack cannot touch the weights or the Triton bank.
    state["marlin"] = _marlin_bank(
        {name: weights[name].clone() for name in ("w13", "w2", "w13_sf", "w2_sf", "w13_gs", "w2_gs")})
    torch.cuda.synchronize()
    for name in scale_names:
        assert torch.equal(state["bank"][name], scales_before[name]), f"{name} changed by the Marlin repack"
        assert torch.equal(weights[name], scales_before[name]), f"weights[{name}] changed by the Marlin repack"
    del scales_before
    torch.cuda.empty_cache()
    _align_par_setup(state, device)
    _silu_sorted_setup(state, device)
    _check_experts_e2e(state, device)
    torch.cuda.synchronize()
    # Last: the CUDA dequant-producer path (build, canary, self-check, turn_4 gate); a failure keeps Marlin.
    print(f"[prepare] max_memory_allocated before ssdq {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB",
          flush=True)
    _ssdq_setup(state, weights, device, t_prepare0)
    torch.cuda.synchronize()
    # Then the register-A wgmma path (rewrite stage 1): its gates read what _experts dispatches (ssdq or Marlin)
    # with rega off; it prints the final PATH CHOSEN lines last.
    _rega_setup(state, weights, device, t_prepare0)
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, local_ids, state):
    """Shape gate. For T <= MARLIN_MAX_TOKENS, in this order: the register-A wgmma kernel (REGA_MIN_TOKENS < T)
    when prepare()'s gate for the T's class ('low' T <= REGA_LOW_MAX, 'high' above) turned it ON; else the
    dequant-producer SS-wgmma kernel for SSDQ_MIN_TOKENS < T when its turn_4 gate turned it ON; else Marlin.
    Above MARLIN_MAX_TOKENS: the register-A wgmma kernel for T <= REGA_FT_MAX when prepare()'s first-turn
    gate ('ft') turned it ON, else (and always above REGA_FT_MAX) the Triton NVFP4 GEMM. Reads the token count
    (a shape) and prepare-time flags, never tensor values."""
    tokens = x.shape[0]
    if tokens <= MARLIN_MAX_TOKENS:
        if tokens > REGA_MIN_TOKENS and state.get("rega_ok", {}).get("low" if tokens <= REGA_LOW_MAX else "high", False):
            return _experts_rega(x, topk_weights, local_ids, state)
        if tokens > SSDQ_MIN_TOKENS and state.get("ssdq_ok", False):
            return _experts_ssdq(x, topk_weights, local_ids, state)
        return _experts_marlin(x, topk_weights, local_ids, state)
    if tokens <= REGA_FT_MAX and state.get("rega_ok", {}).get("ft", False):
        return _experts_rega(x, topk_weights, local_ids, state)
    return _experts_triton(x, topk_weights, local_ids, state)


# On the Marlin path (T <= MARLIN_MAX_TOKENS) _experts_marlin replaces moe_align_block_size with
# _align_parallel_kernel, ported from muse_moe_w4a16 master via the decode layer (hypothesis
# moe-align-local-routes-parallel-port): one launch, and Marlin's block list holds the 32 local
# experts' routes only, so no m-blocks are spent on remote routes. Gated by a prepare() self-check.
ALIGN_PAR_CHUNK = 512
ALIGN_PAR_WARPS = 8
ALIGN_PAR_BLK = 64
ALIGN_PAR_TAG = "[align-local]"


@triton.jit(do_not_specialize=["local_start", "local_count", "n", "cap", "n_blocks"])
def _align_parallel_kernel(topk_ids_ptr, local_ids_ptr, sorted_ids_ptr, block_experts_ptr, n_padded_ptr,
                           local_start, local_count, n, cap, n_blocks, CHUNK: tl.constexpr,
                           PAD_BLOCK: tl.constexpr, BLK: tl.constexpr, BSM: tl.constexpr,
                           NUM_E: tl.constexpr):
    """_align_small_kernel's outputs for any n, grid (local_count + 1,).

    Every program streams the n route ids in CHUNK chunks and counts all local experts, so each
    knows every segment start. Program e < local_count then places expert e's routes in ascending
    route order (segment start + routes placed by earlier chunks + rank in the chunk), its padding
    slots (sentinel n) and its block_experts entries. Program local_count writes local_ids, the
    sentinel into [total, cap), -1 into block_experts past the used blocks, and n_padded. The
    store sets are disjoint across programs, so no atomics or ordering are needed; no program
    reads what another writes (local ids are recomputed from topk_ids)."""
    pid = tl.program_id(0)
    offs0 = tl.arange(0, CHUNK)
    experts = tl.arange(0, NUM_E)

    counts = tl.zeros([NUM_E], dtype=tl.int32)
    for c in range(0, n, CHUNK):
        offs = c + offs0
        mask = offs < n
        ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        local = ids - local_start
        local = tl.where(mask & (local >= 0) & (local < local_count), local, -1)
        counts += tl.sum((local[:, None] == experts[None, :]).to(tl.int32), axis=0)
    padded = (counts + BSM - 1) // BSM * BSM
    seg_end = tl.cumsum(padded, axis=0)
    seg_start = seg_end - padded
    total = tl.sum(padded, axis=0)

    if pid < local_count:
        sel = experts == pid
        my_count = tl.sum(tl.where(sel, counts, 0), axis=0)
        my_padded = tl.sum(tl.where(sel, padded, 0), axis=0)
        my_start = tl.sum(tl.where(sel, seg_start, 0), axis=0)
        # This expert's routes, ascending route index.
        base = my_start
        for c in range(0, n, CHUNK):
            offs = c + offs0
            mask = offs < n
            ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
            local = ids - local_start
            match = (mask & (local == pid)).to(tl.int32)
            rank = tl.cumsum(match, axis=0) - 1
            tl.store(sorted_ids_ptr + base + rank, offs, mask=match > 0)
            base += tl.sum(match, axis=0)
        # Its padding slots: fewer than BSM, over a power-of-2 tile (BSM 48 is not one).
        j = tl.arange(0, PAD_BLOCK)
        tl.store(sorted_ids_ptr + my_start + my_count + j, tl.zeros([PAD_BLOCK], dtype=tl.int32) + n,
                 mask=j < my_padded - my_count)
        # Its m-blocks [my_start / BSM, (my_start + my_padded) / BSM); a runtime loop, since one
        # expert can own up to cdiv(M, BSM) blocks.
        b0 = my_start // BSM
        nb = my_padded // BSM
        kb0 = tl.arange(0, BLK)
        for k in range(0, nb, BLK):
            kb = k + kb0
            tl.store(block_experts_ptr + b0 + kb, tl.zeros([BLK], dtype=tl.int32) + pid, mask=kb < nb)
    else:
        for c in range(0, n, CHUNK):
            offs = c + offs0
            mask = offs < n
            ids = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
            local = ids - local_start
            local = tl.where((local >= 0) & (local < local_count), local, -1)
            tl.store(local_ids_ptr + offs, local, mask=mask)
        # Every slot past the used length.
        for s in range(0, cap, CHUNK):
            slots = s + offs0
            tl.store(sorted_ids_ptr + slots, tl.zeros([CHUNK], dtype=tl.int32) + n,
                     mask=(slots >= total) & (slots < cap))
        used_blocks = total // BSM
        kb0 = tl.arange(0, BLK)
        for k in range(0, n_blocks, BLK):
            kb = k + kb0
            tl.store(block_experts_ptr + kb, tl.zeros([BLK], dtype=tl.int32) - 1,
                     mask=(kb >= used_blocks) & (kb < n_blocks))
        tl.store(n_padded_ptr, total)


def _align_parallel(topk_ids, local_start, local_count, block_size_m):
    """Marlin's (sorted_ids, block_experts, n_padded) and local_ids [M, 8] int32 for M > 32, in one launch of
    _align_parallel_kernel over local_count + 1 programs. n, cap and n_blocks are runtime
    arguments, so only block_size_m selects a compiled variant."""
    topk_ids = topk_ids.contiguous()
    device = topk_ids.device
    n = topk_ids.numel()
    cap = n + local_count * (block_size_m - 1)
    n_blocks = triton.cdiv(cap, block_size_m)
    local_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)
    sorted_ids = torch.empty((cap,), dtype=torch.int32, device=device)
    block_experts = torch.empty((n_blocks,), dtype=torch.int32, device=device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=device)
    _align_parallel_kernel[(local_count + 1,)](
        topk_ids, local_ids, sorted_ids, block_experts, n_padded, local_start, local_count, n, cap,
        n_blocks, CHUNK=ALIGN_PAR_CHUNK, PAD_BLOCK=triton.next_power_of_2(block_size_m),
        BLK=ALIGN_PAR_BLK, BSM=block_size_m, NUM_E=triton.next_power_of_2(local_count),
        num_warps=ALIGN_PAR_WARPS)
    return local_ids, (sorted_ids, block_experts, n_padded)

def _align_par_ref_view(sorted_ids, block_experts, n_padded, n, bsm):
    """Per local expert: (sorted routes, block count) from an align's three outputs."""
    sorted_ids = sorted_ids.cpu()
    block_experts = block_experts.cpu()
    used = min(int(n_padded.cpu()[0]) // bsm, block_experts.numel())
    out = {}
    for e in range(LOCAL_EXPERTS):
        blocks = (block_experts[:used] == e).nonzero().flatten().tolist()
        routes = []
        for b in blocks:
            seg = sorted_ids[b * bsm:(b + 1) * bsm]
            routes += seg[seg < n].tolist()
        out[e] = (sorted(routes), len(blocks))
    return out


def _align_par_setup(state, device):
    """Untimed, prepare() only: compile _align_parallel_kernel for every block_size_m the Marlin path
    can pick, then compare it per local expert with SGLang's moe_align_block_size(local_ids, bsm,
    N_EXPERTS) on synthetic routings (local_ids 0-31, -1 remote). Remote routes, padding and capacity
    differences are ignored. Sets state["align_par_ok"]; any failure keeps master's align."""
    state["align_par_ok"] = False
    cases = mismatches = 0
    try:
        for bsm in BLOCK_SIZE_M_CANDIDATES:
            _align_parallel(torch.zeros((16, TOP_K), dtype=torch.int32, device=device), 0, LOCAL_EXPERTS, bsm)
        torch.cuda.synchronize()
        gen = torch.Generator(device=device).manual_seed(20261024)
        for tokens in (9, 200, 753, 3012, 4096):
            for kind in ("random", "skewed", "all_remote"):
                ids = torch.randint(0, N_EXPERTS, (tokens, TOP_K), generator=gen, device=device, dtype=torch.int32)
                if kind == "skewed":
                    ids[:, :4] = torch.tensor([0, 31, 5, 17], dtype=torch.int32, device=device)
                elif kind == "all_remote":
                    ids = ids % (N_EXPERTS - LOCAL_EXPERTS) + LOCAL_EXPERTS
                local_ids = torch.where(ids < LOCAL_EXPERTS, ids, torch.full_like(ids, -1)).contiguous()
                for bsm in BLOCK_SIZE_M_CANDIDATES:
                    if tokens * TOP_K / LOCAL_EXPERTS / bsm < 0.9:
                        break
                r_sorted, r_blocks, r_pad = moe_align_block_size(local_ids, bsm, N_EXPERTS)
                n_local, (n_sorted, n_blocks, n_pad) = _align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm)
                cases += 1
                bad = not torch.equal(n_local, local_ids)
                if not bad:
                    bad = (_align_par_ref_view(r_sorted, r_blocks, r_pad, ids.numel(), bsm)
                           != _align_par_ref_view(n_sorted, n_blocks, n_pad, ids.numel(), bsm))
                if bad:
                    mismatches += 1
                    print(f"{ALIGN_PAR_TAG}   mismatch T={tokens} {kind} bsm={bsm}", flush=True)
        torch.cuda.synchronize()
        state["align_par_ok"] = mismatches == 0
    except Exception as e:
        print(f"{ALIGN_PAR_TAG} setup raised {e!r}", flush=True)
        mismatches = max(mismatches, 1)
        state["align_par_ok"] = False
    print(f"{ALIGN_PAR_TAG} {'ON' if state['align_par_ok'] else 'OFF'} cases={cases} mismatches={mismatches}"
          f" (PATH CHOSEN = {'parallel align' if state['align_par_ok'] else 'MASTER align'} on the Marlin path)",
          flush=True)
    _align_par_ft_setup(state, device)


ALIGN_PAR_FT_TOKENS = (8192, 16384)


def _align_par_ft_setup(state, device):
    """Untimed, prepare() only (experts-turn-rega-s4-first-turn-n128): the same per-expert comparison with
    moe_align_block_size at the first-turn sizes, at Marlin's block size for the T and at REGA_BSM (the rega
    path's block), on random, skewed and local-saturated (every route on a local expert: the most blocks)
    routings. Sets state["align_par_ft_ok"] only, so a miss here turns off the rega first-turn class and never
    master's Marlin-path align."""
    state["align_par_ft_ok"] = False
    cases = mismatches = 0
    try:
        gen = torch.Generator(device=device).manual_seed(20261025)
        for tokens in ALIGN_PAR_FT_TOKENS:
            for kind in ("random", "skewed", "local"):
                if kind == "local":
                    ids = torch.rand((tokens, LOCAL_EXPERTS), generator=gen, device=device).topk(
                        TOP_K, dim=1).indices.to(torch.int32)
                else:
                    ids = torch.randint(0, N_EXPERTS, (tokens, TOP_K), generator=gen, device=device,
                                        dtype=torch.int32)
                    if kind == "skewed":
                        ids[:, :4] = torch.tensor([0, 31, 5, 17], dtype=torch.int32, device=device)
                local_ids = torch.where(ids < LOCAL_EXPERTS, ids, torch.full_like(ids, -1)).contiguous()
                for bsm in BLOCK_SIZE_M_CANDIDATES:
                    if tokens * TOP_K / LOCAL_EXPERTS / bsm < 0.9:
                        break
                for b in sorted({bsm, REGA_BSM}):
                    r_sorted, r_blocks, r_pad = moe_align_block_size(local_ids, b, N_EXPERTS)
                    n_local, (n_sorted, n_blocks, n_pad) = _align_parallel(local_ids, 0, LOCAL_EXPERTS, b)
                    cases += 1
                    bad = not torch.equal(n_local, local_ids)
                    if not bad:
                        bad = (_align_par_ref_view(r_sorted, r_blocks, r_pad, ids.numel(), b)
                               != _align_par_ref_view(n_sorted, n_blocks, n_pad, ids.numel(), b))
                    if bad:
                        mismatches += 1
                        print(f"{ALIGN_PAR_TAG}   mismatch T={tokens} {kind} bsm={b}", flush=True)
        torch.cuda.synchronize()
        state["align_par_ft_ok"] = cases > 0 and mismatches == 0
    except Exception as e:
        print(f"{ALIGN_PAR_TAG} first-turn cases raised {e!r}", flush=True)
        mismatches = max(mismatches, 1)
        state["align_par_ft_ok"] = False
    print(f"{ALIGN_PAR_TAG} first-turn T {ALIGN_PAR_FT_TOKENS}: {'ok' if state['align_par_ft_ok'] else 'MISS'} "
          f"cases={cases} mismatches={mismatches} (gates only the rega first-turn class)", flush=True)
# Marlin-path silu-mul over the aligned local rows only (hypothesis silu-mul-triton-local-rows-marlin-turn):
# SGLang's act_and_mul_kernel launches over all T*8 slots; this walks moe_align_block_size's sorted_ids
# up to n_padded and computes silu(gate) * up for slots whose local id is >= 0 (remote and padding rows
# are skipped, as today their cache2 rows are left unwritten and never read).
SILU_SORTED_BLOCK_R = 16
SILU_SORTED_BLOCK_N = 512


@triton.jit(do_not_specialize=["num_slots"])
def _silu_mul_sorted_kernel(cache1_ptr, cache2_ptr, sorted_ptr, n_padded_ptr, local_ids_ptr, num_slots,
                            SIZE_N: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    n_padded = tl.load(n_padded_ptr)
    r0 = tl.program_id(0) * BLOCK_R
    if r0 < n_padded:
        rows = r0 + tl.arange(0, BLOCK_R)
        sid = tl.load(sorted_ptr + rows, mask=rows < n_padded, other=num_slots)
        valid = sid < num_slots
        lid = tl.load(local_ids_ptr + sid, mask=valid, other=-1)
        valid = valid & (lid >= 0)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        s64 = sid.to(tl.int64)
        m = valid[:, None]
        src = cache1_ptr + s64[:, None] * (2 * SIZE_N) + cols[None, :]
        g = tl.load(src, mask=m, other=0.0).to(tl.float32)
        u = tl.load(src + SIZE_N, mask=m, other=0.0).to(tl.float32)
        y = g / (1.0 + tl.exp(-g)) * u
        tl.store(cache2_ptr + s64[:, None] * SIZE_N + cols[None, :], y.to(tl.bfloat16), mask=m)


def _silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids, size_n):
    num_slots = cache2.shape[0]
    grid = (triton.cdiv(sorted_ids.shape[0], SILU_SORTED_BLOCK_R), size_n // SILU_SORTED_BLOCK_N)
    _silu_mul_sorted_kernel[grid](cache1, cache2, sorted_ids, n_padded, local_ids, num_slots, SIZE_N=size_n,
                                  BLOCK_R=SILU_SORTED_BLOCK_R, BLOCK_N=SILU_SORTED_BLOCK_N, num_warps=4)


def _silu_sorted_setup(state, device):
    """Untimed, prepare() only: compile _silu_mul_sorted_kernel and compare it with SGLang's
    silu_and_mul on the local rows of synthetic caches at T=753 and T=3012 (Marlin block 64), within
    one bf16 rounding (the exp may differ from SGLang's expf in the last fp32 bits). Sets
    state["silu_sorted_ok"]; any failure keeps silu_and_mul."""
    state["silu_sorted_ok"] = False
    try:
        size_n = state["marlin"]["size_n"]
        if size_n % SILU_SORTED_BLOCK_N:
            raise ValueError(f"size_n {size_n} not a multiple of {SILU_SORTED_BLOCK_N}")
        gen = torch.Generator(device=device).manual_seed(20261024)
        worst = 0.0
        for tokens in (753, 3012):
            ids = torch.randint(0, N_EXPERTS, (tokens, TOP_K), generator=gen, device=device, dtype=torch.int32)
            local_ids = torch.where(ids < LOCAL_EXPERTS, ids, torch.full_like(ids, -1)).contiguous()
            for bsm in BLOCK_SIZE_M_CANDIDATES:
                if tokens * TOP_K / LOCAL_EXPERTS / bsm < 0.9:
                    break
            sorted_ids, _, n_padded = moe_align_block_size(local_ids, bsm, N_EXPERTS)
            cache1 = (torch.randn((tokens * TOP_K, 2 * size_n), generator=gen, device=device) * 3).to(torch.bfloat16)
            ref = torch.zeros((tokens * TOP_K, size_n), dtype=torch.bfloat16, device=device)
            got = torch.zeros_like(ref)
            silu_and_mul(cache1, ref, expert_ids=local_ids.view(-1), expert_step=1)
            _silu_mul_sorted(cache1, got, sorted_ids, n_padded, local_ids.view(-1), size_n)
            torch.cuda.synchronize()
            loc = local_ids.view(-1) >= 0
            r, g = ref[loc].float(), got[loc].float()
            if not bool(torch.isfinite(g).all()):
                raise AssertionError(f"non-finite output at T={tokens}")
            err = ((g - r).abs() / (r.abs() + 1e-3)).max().item()
            worst = max(worst, err)
            if err > 2.0 ** -7:
                raise AssertionError(f"T={tokens} max rel err {err:.3e} > 2^-7")
        state["silu_sorted_ok"] = True
        print(f"[silu-sorted] ON max rel err {worst:.2e} (PATH CHOSEN = sorted silu on the Marlin path)", flush=True)
    except Exception as e:
        print(f"[silu-sorted] OFF: {e!r} (PATH CHOSEN = MASTER silu_and_mul)", flush=True)


def _experts_marlin(x, topk_weights, local_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize, on ids
    already mapped to this GPU's experts (0-31, -1 elsewhere)."""
    bank = state["marlin"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    if state.get("align_par_ok", False):
        # moe-align-local-routes-parallel-port: local routes only, one launch (prepare() self-checked).
        _, (sorted_ids, block_experts, n_padded) = _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    if state.get("silu_sorted_ok", False):
        _silu_mul_sorted(cache1.view(-1, gemm1_n), cache2, sorted_ids, n_padded, local_ids.view(-1), size_n)
    else:
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _experts_triton(x, topk_weights, local_ids, state):
    """MuseSpark1xFp4Experts' expert outputs and post-expert-norm finalize, on ids already mapped
    to this GPU's experts (0-31, -1 elsewhere), with the two Marlin GEMMs replaced by the Triton
    grouped GEMM that dequantizes the stored NVFP4 weights in registers."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    gemm1_n = bank["w13"].shape[1]
    size_n = gemm1_n // 2
    num_slots = num_tokens * TOP_K
    topk_weights = topk_weights.to(torch.float32).contiguous()
    cfg = _expert_config(num_tokens)
    # moe-align-32-experts-first-turn: the ids are already 0-31 or -1, so align over the 32 local experts
    # (muse_moe_w4a16 master does the same): 32 counters and numel + 33*(BLOCK_T-1) capacity, not 257.
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, cfg["BLOCK_T"], LOCAL_EXPERTS)
    cache2 = torch.empty((num_slots, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_slots * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_slots * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_slots * size_k].view(-1, size_k)
    _nvfp4_moe_gemm(x.contiguous(), bank["w13"], bank["w13_sf"], bank["w13_gs"], cache1, sorted_ids, block_experts,
                    n_padded, TOP_K, num_slots, cfg, False)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = _nvfp4_moe_gemm(cache2, bank["w2"], bank["w2_sf"], bank["w2_gs"], cache3, sorted_ids, block_experts,
                        n_padded, 1, num_slots, cfg, state["zero_nonlocal_w2"]).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


# ---------------------------------------------------------------------------------------------
# CUDA dequant-producer SS-wgmma expert path for turn_4 (hypothesis
# experts-turn4-cuda-dequant-producer-ss-wgmma, rewrite stage 1). 1024 < T <= MARLIN_MAX_TOKENS.
#
# Two launches of one hand-written Hopper kernel (load_inline, sm_90a only) replace the two Marlin
# GEMMs; master's parallel align (here at block 128) and the post-expert weighted sum are kept. The
# weights stay NVFP4 in global memory (the nibble-reordered bank prepare() already holds); the only
# bf16 copy of a weight tile lives in shared memory for one pipeline stage. CTA = 3 warpgroups, one
# 128-route x 256-column tile per CTA, BLOCK_K 64, 4 stages of (A 16 KB + B 32 KB):
#   WG0 (producer, 128 threads): prefetches the next 128-k chunk of packed E2M1 codes and e4m3
#       scales into registers, dequantizes it (exact, the nibble -> bf16 bit trick of the Triton
#       kernel) into the 128B-swizzled B stage with st.shared, fence.proxy.async, mbarrier arrive.
#       Thread 0 also issues the A tile TMA (the activation rows are pre-gathered into sorted order
#       by ssdq_gather, so A is a plain 2-D TMA box, no cp.async gather).
#   WG1, WG2 (consumers): SS wgmma m64n256k16 (both operands from shared memory: no register-A
#       hazard), 64 routes each, fp32 accumulators, global scale in the epilogue.
# No setmaxnreg: the review's 232 + 160 split needs 79,872 registers per SM (> 65,536); all three
# warpgroups run at the 168-register launch-bounds cap instead.
# w13's epilogue writes bf16 rows in sorted-row space (c1[row, 8192]); a Triton kernel does silu*up
# row-to-row (c2[row, 4096]); w2 reads c2 by TMA directly and scatters rows to y[sorted id].
# ---------------------------------------------------------------------------------------------
SSDQ_TAG = "[ssdq]"
SSDQ_BM = 128
SSDQ_MIN_TOKENS = 1024        # the path serves SSDQ_MIN_TOKENS < T <= MARLIN_MAX_TOKENS
SSDQ_GATE_TOKENS = 3012       # turn_4: the shape the gate times
SSDQ_GATE_RATIO = 0.95        # ON only when new / Marlin pair <= this
SSDQ_ALTERNATIONS = 9
SSDQ_BUDGET_S = 170.0         # skip the whole setup if prepare() has used more than this already
SSDQ_EXT_NAME = "muse_prefill_ssdq_v1"
SSDQ_GENCODE = "-gencode=arch=compute_90a,code=sm_90a"
SSDQ_CUDA_CFLAGS = ["-O3", "-std=c++20", "-lineinfo", SSDQ_GENCODE]
SSDQ_ERR_TEXT = {0: "ok", 1: "mbarrier wait timed out", 2: "non-sm_90a image ran (wgmma compiled out)"}
SSDQ_SILU_BLOCK_R = 16
SSDQ_SILU_BLOCK_N = 512

_SSDQ_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> ssdq_info();
void ssdq_gather(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor block_experts, int64_t num_routes,
                 int64_t n_experts, torch::Tensor xs);
int64_t ssdq_gemm(int64_t mode, torch::Tensor a, torch::Tensor w, torch::Tensor sf, torch::Tensor gs,
                  torch::Tensor sorted_ids, torch::Tensor block_experts, int64_t num_routes, torch::Tensor out,
                  torch::Tensor err, torch::Tensor prof, int64_t use_prof);
"""

_SSDQ_HEAD = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int THREADS = 384;
constexpr int STAGES = 4;
constexpr int BM = 128;                       // routes per CTA tile
constexpr int BN = 256;                       // output columns per CTA tile
constexpr int BKT = 64;                       // k per pipeline stage (one 128 B swizzled row)
constexpr int A_BYTES = BM * BKT * 2;         // 16384
constexpr int B_BYTES = BN * BKT * 2;         // 32768
constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
constexpr int SMEM_BYTES = 1024 + STAGES * STAGE_BYTES;
static_assert(A_BYTES % 1024 == 0 && STAGE_BYTES % 1024 == 0, "stage alignment");
static_assert(SMEM_BYTES + 4096 <= 227 * 1024, "shared memory budget");

struct __align__(64) Maps { CUtensorMap m; };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n\t.reg .pred P;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, P;\n\t}\n"
               : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok != 0;
}
// Bounded wait: after 2^26 cycles without the phase, record error word 1 and stop waiting (dead),
// so a broken pipeline ends in a failed self-check instead of a hang.
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity, int* err, bool& dead) {
  if (dead || mbar_try(a, parity)) return;
  const long long t0 = clock64();
  while (!mbar_try(a, parity)) {
    if (clock64() - t0 > (1ll << 26)) {
      dead = true;
      atomicExch(err, 1);
      return;
    }
  }
}
__device__ __forceinline__ void tma_2d(uint32_t dst, uint64_t map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// Generic-proxy shared writes (the dequantized B tile) must be made visible to the async proxy
// (wgmma) before the mbarrier arrive that publishes them.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// wgmma exists only for sm_90a: every such body is compiled only under __CUDA_ARCH_FEAT_SM90_ALL, so
// a stray plain compute_90 pass still assembles; there ssdq_kernel only sets error word 2.
__device__ __forceinline__ void wgmma_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
#endif
}
__device__ __forceinline__ void fence_acc(float (&d)[128]) {
#pragma unroll
  for (int i = 0; i < 128; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
// K-major SWIZZLE_128B shared-memory matrix descriptor: start address >> 4 (bits 0-13), LBO 1
// (unused for swizzled K-major), SBO 1024 B between 8-row core groups (bits 32-45), layout type 1 =
// 128B swizzle (bits 62-63), base offset 0 (every tile base is 1024-aligned; the k16 advance of +32 B
// stays inside the first 128 B row).
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}
"""

_SSDQ_BODY = r"""
__device__ __forceinline__ uint32_t bf16x2_bcast(float v) {
  __nv_bfloat162 b = __float2bfloat162_rn(v);
  return *reinterpret_cast<uint32_t*>(&b);
}
// a * b (+ -0), bf16x2, round to nearest; subnormal inputs are kept (the sibling kernels' trick).
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
// One e4m3 byte -> float (hardware cvt, as the sibling kernels), times 2^119, as a bf16x2 broadcast.
// 448 * 2^119 < the bf16 maximum and e4m3 has 4 significant bits, so the bf16 value is exact.
__device__ __forceinline__ uint32_t sf_bf16x2(uint32_t byte) {
  const unsigned short in = static_cast<unsigned short>(byte & 0xFFu);
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"(in));
  const float f = __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xFFFFu)));
  return bf16x2_bcast(f * __int_as_float(0x7B000000));
}
// p holds two E2M1 codes, at bits 0-3 and 16-19. Each code goes to bf16 bits: 3 magnitude bits to
// bits 8..6 (exponent low bits and the top mantissa bit), sign to bit 15: that is e2m1 * 2^-126
// exactly (e = 0 codes on bf16 subnormals); times scale * 2^119 gives e2m1 * scale * 2^-7, at most 6
// significant bits, exact in bf16. The epilogue multiplies by global scale * 2^7.
__device__ __forceinline__ uint32_t dq_pair(uint32_t p, uint32_t sb2) {
  const uint32_t v = ((p & 0x00070007u) << 6) | ((p & 0x00080008u) << 12);
  return mul_bf16x2(v, sb2);
}

__global__ void __launch_bounds__(THREADS, 1)
ssdq_kernel(const __grid_constant__ Maps maps,
            const uint8_t* __restrict__ w,         // [E][N][K/2], nibble-reordered per 128-k chunk
            const uint8_t* __restrict__ sf,        // e4m3 [E][N][K/16]
            const float* __restrict__ gs,          // fp32 [E] global scales
            const int* __restrict__ sorted_ids,    // block-128 align of the local ids
            const int* __restrict__ block_experts,
            int n_experts, int num_routes, int N, int K, int mode,
            __nv_bfloat16* __restrict__ out, int out_ld, int* __restrict__ err, long long* __restrict__ prof) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[2 * STAGES];
  const int e = __ldg(block_experts + blockIdx.y);
  if (e < 0 || e >= n_experts) return;  // uniform over the CTA: no barrier has been touched yet
  // Launch marker: err[0] is the error word, err[1 + mode] counts CTAs that started real work.
  if (threadIdx.x == 0) atomicAdd(err + 1 + mode, 1);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  const uint32_t full0 = smem_u32(&bars[0]), empty0 = smem_u32(&bars[STAGES]);
  const int n0 = blockIdx.x * BN;   // first output column of this tile
  const int m0 = blockIdx.y * BM;   // first sorted row of this tile
  const int KT = K / BKT;           // pipeline steps
  const bool do_prof = prof != nullptr;
  long long* pr = prof + 4 * mode;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + 8 * s, 129);   // 128 producer threads + the TMA thread's expect_tx arrive
      mbar_init(empty0 + 8 * s, 8);    // one arrive per consumer warp
    }
    fence_barrier_init();
  }
  __syncthreads();
  bool dead = false;

  if (warp < 4) {
    // ---------------- producer warpgroup ----------------
    const int p = tid;                 // 0..127
    const int q = p & 3;               // 16-byte piece of a row's 64-byte chunk
    const int rs = p >> 2;             // 0..31: rows rs + 32 * it
    const uint8_t* wbase = w + (static_cast<size_t>(e) * N + n0) * (K / 2);
    const uint8_t* sbase = sf + (static_cast<size_t>(e) * N + n0) * (K / 16);
    const int nchunks = K / 128;
    long long wait_cyc = 0;
    const long long t_begin = (do_prof && p == 0) ? clock64() : 0;
    uint4 cur[8], nxt[8];
    uint2 cs[8], ns[8];
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int row = it * 32 + rs;
      cur[it] = __ldg(reinterpret_cast<const uint4*>(wbase + static_cast<size_t>(row) * (K / 2) + q * 16));
      cs[it] = __ldg(reinterpret_cast<const uint2*>(sbase + static_cast<size_t>(row) * (K / 16)));
    }
    for (int c = 0; c < nchunks; ++c) {
      if (c + 1 < nchunks) {
#pragma unroll
        for (int it = 0; it < 8; ++it) {
          const int row = it * 32 + rs;
          nxt[it] = __ldg(reinterpret_cast<const uint4*>(wbase + static_cast<size_t>(row) * (K / 2) + (c + 1) * 64 + q * 16));
          ns[it] = __ldg(reinterpret_cast<const uint2*>(sbase + static_cast<size_t>(row) * (K / 16) + (c + 1) * 8));
        }
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int kt = 2 * c + half;
        const int s = kt % STAGES;
        const long long tw = do_prof ? clock64() : 0;
        mbar_wait(empty0 + 8 * s, ((kt / STAGES) & 1) ^ 1, err, dead);
        if (do_prof) wait_cyc += clock64() - tw;
        uint8_t* stage = smem + s * STAGE_BYTES;
        if (p == 0) {
          mbar_expect_tx(full0 + 8 * s, A_BYTES);
          tma_2d(smem_u32(stage), reinterpret_cast<uint64_t>(&maps.m), kt * BKT, m0, full0 + 8 * s);
        }
        uint8_t* bt = stage + A_BYTES;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
          const int row = it * 32 + rs;
          const uint32_t sc = half ? cs[it].y : cs[it].x;          // 4 scale bytes: groups of this half
          const uint32_t sb2 = sf_bf16x2((sc >> (8 * q)) & 0xFFu);  // this thread's 16 k = one scale group
          const uint32_t wv[4] = {cur[it].x, cur[it].y, cur[it].z, cur[it].w};
          uint32_t o[8];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            // bytes (b0, b1) and (b2, b3) of the word -> codes at bits 0-7 and 16-23; the high nibbles
            // (k + 64) for the second half.
            const uint32_t pa = __byte_perm(wv[i], 0u, 0x4140) >> (4 * half);
            const uint32_t pb = __byte_perm(wv[i], 0u, 0x4342) >> (4 * half);
            o[2 * i] = dq_pair(pa, sb2);
            o[2 * i + 1] = dq_pair(pb, sb2);
          }
          uint8_t* rowp = bt + row * 128;
          const int sw = row & 7;
          *reinterpret_cast<uint4*>(rowp + (((2 * q) ^ sw) << 4)) = make_uint4(o[0], o[1], o[2], o[3]);
          *reinterpret_cast<uint4*>(rowp + (((2 * q + 1) ^ sw) << 4)) = make_uint4(o[4], o[5], o[6], o[7]);
        }
        fence_proxy_async();
        mbar_arrive(full0 + 8 * s);
        if (half == 1) {
#pragma unroll
          for (int it = 0; it < 8; ++it) { cur[it] = nxt[it]; cs[it] = ns[it]; }
        }
      }
    }
    if (do_prof && p == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(pr + 0), static_cast<unsigned long long>(wait_cyc));
      atomicAdd(reinterpret_cast<unsigned long long*>(pr + 2), 1ull);
      atomicAdd(reinterpret_cast<unsigned long long*>(pr + 3), static_cast<unsigned long long>(clock64() - t_begin));
    }
  } else {
    // ---------------- consumer warpgroups ----------------
    const int ctid = tid - 128, wg = ctid >> 7, cw = (ctid >> 5) & 3;
    float acc[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) acc[i] = 0.f;
    long long wait_cyc = 0;
    for (int kt = 0; kt < KT; ++kt) {
      const int s = kt % STAGES;
      const long long tw = do_prof ? clock64() : 0;
      mbar_wait(full0 + 8 * s, (kt / STAGES) & 1, err, dead);
      if (do_prof) wait_cyc += clock64() - tw;
      const uint32_t a = smem_u32(smem + s * STAGE_BYTES) + wg * (64 * 128);
      const uint32_t b = smem_u32(smem + s * STAGE_BYTES + A_BYTES);
      fence_acc(acc);
      wgmma_fence();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint64_t da = make_desc(a + k * 32), db = make_desc(b + k * 32);
@MMA_CALL@
      }
      wgmma_commit();
      wgmma_wait<1>();  // the previous stage's group has retired: its buffers can be refilled
      fence_acc(acc);
      if (kt > 0 && lane == 0) mbar_arrive(empty0 + 8 * ((kt - 1) % STAGES));
    }
    wgmma_wait<0>();
    fence_acc(acc);
    if (lane == 0) mbar_arrive(empty0 + 8 * ((KT - 1) % STAGES));
    if (do_prof && (ctid & 127) == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(pr + 1), static_cast<unsigned long long>(wait_cyc));
    }

    // Epilogue straight from registers: acc[4j + 2h + l] = row (16 cw + lane / 4 + 8 h) of this
    // warpgroup's 64, column 8 j + 2 (lane % 4) + l of the 256.
    const float gsc = __ldg(gs + e) * 128.0f;   // the 2^7 the dequantized tile left out
    const int rbase = m0 + wg * 64 + cw * 16 + (lane >> 2);
    int sid0 = 0, sid1 = 0;
    if (mode == 1) {
      sid0 = __ldg(sorted_ids + rbase);
      sid1 = __ldg(sorted_ids + rbase + 8);
    }
    const bool ok0 = mode == 0 || (sid0 >= 0 && sid0 < num_routes);
    const bool ok1 = mode == 0 || (sid1 >= 0 && sid1 < num_routes);
    const size_t r0 = mode == 0 ? static_cast<size_t>(rbase) : static_cast<size_t>(sid0);
    const size_t r1 = mode == 0 ? static_cast<size_t>(rbase + 8) : static_cast<size_t>(sid1);
    __nv_bfloat16* o0 = out + r0 * out_ld + n0 + 2 * (lane & 3);
    __nv_bfloat16* o1 = out + r1 * out_ld + n0 + 2 * (lane & 3);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      if (ok0) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[4 * j] * gsc, acc[4 * j + 1] * gsc);
        *reinterpret_cast<__nv_bfloat162*>(o0 + 8 * j) = v;
      }
      if (ok1) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[4 * j + 2] * gsc, acc[4 * j + 3] * gsc);
        *reinterpret_cast<__nv_bfloat162*>(o1 + 8 * j) = v;
      }
    }
  }
#else
  // Not the sm_90a image: no wgmma here. Flag it (error word 2) and do nothing else.
  if (threadIdx.x == 0) atomicExch(err, 2);
#endif
}

// xs[slot] = x[sorted_ids[slot] / 8] for the slots of local blocks (sentinel slots: zeros); slots of
// blocks past the used count (expert -1) are skipped. One CTA per slot.
__global__ void __launch_bounds__(256)
ssdq_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ sorted_ids,
                   const int* __restrict__ block_experts, int num_routes, int H, int n_experts,
                   __nv_bfloat16* __restrict__ xs) {
  const int slot = blockIdx.x;
  const int be = __ldg(block_experts + (slot / BM));
  if (be < 0 || be >= n_experts) return;
  const int sid = __ldg(sorted_ids + slot);
  uint4* dst = reinterpret_cast<uint4*>(xs + static_cast<size_t>(slot) * H);
  const int nv = H / 8;
  if (sid >= 0 && sid < num_routes) {
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(sid >> 3) * H);
    for (int v = threadIdx.x; v < nv; v += blockDim.x) dst[v] = __ldg(src + v);
  } else {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int v = threadIdx.x; v < nv; v += blockDim.x) dst[v] = z;
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFn get_encode() {
  static EncodeTiledFn fn_cached = nullptr;
  if (fn_cached != nullptr) return fn_cached;
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
  fn_cached = reinterpret_cast<EncodeTiledFn>(fn);
  return fn_cached;
}

}  // namespace

// regs, local bytes, static smem, dynamic smem, CTAs per SM, cudaFuncSetAttribute rc, binaryVersion,
// ptxVersion, attribute calls ok.
std::vector<int64_t> ssdq_info() {
  std::vector<int64_t> r;
  const void* fn = reinterpret_cast<const void*>(ssdq_kernel);
  const cudaError_t e1 = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, fn);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ssdq_kernel, THREADS, SMEM_BYTES);
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(static_cast<int64_t>(a.sharedSizeBytes));
  r.push_back(SMEM_BYTES);
  r.push_back(occ);
  r.push_back(static_cast<int64_t>(e1));
  r.push_back(a.binaryVersion);
  r.push_back(a.ptxVersion);
  r.push_back(ok ? 1 : 0);
  return r;
}

void ssdq_gather(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor block_experts, int64_t num_routes,
                 int64_t n_experts, torch::Tensor xs) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && xs.is_contiguous() && sorted_ids.is_contiguous() &&
              block_experts.is_contiguous() && x.scalar_type() == at::kBFloat16 && xs.scalar_type() == at::kBFloat16 &&
              sorted_ids.scalar_type() == at::kInt && block_experts.scalar_type() == at::kInt, "ssdq_gather: layout");
  TORCH_CHECK(x.size(1) == xs.size(1) && x.size(1) % 8 == 0 && xs.size(0) == block_experts.numel() * BM &&
              sorted_ids.numel() <= xs.size(0), "ssdq_gather: sizes");
  const int slots = static_cast<int>(block_experts.numel() * BM);
  if (slots == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  ssdq_gather_kernel<<<slots, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const int*>(sorted_ids.data_ptr()), reinterpret_cast<const int*>(block_experts.data_ptr()),
      static_cast<int>(num_routes), static_cast<int>(x.size(1)), static_cast<int>(n_experts),
      reinterpret_cast<__nv_bfloat16*>(xs.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// mode 0: w13, out = c1 in sorted-row space [rows, N]; mode 1: w2, out = y [num_routes, N] scattered
// to the sorted ids. Returns 0, or the CUresult + 1000 of a failed tensor-map encode (1 if the entry
// point is missing).
int64_t ssdq_gemm(int64_t mode, torch::Tensor a, torch::Tensor w, torch::Tensor sf, torch::Tensor gs,
                  torch::Tensor sorted_ids, torch::Tensor block_experts, int64_t num_routes, torch::Tensor out,
                  torch::Tensor err, torch::Tensor prof, int64_t use_prof) {
  TORCH_CHECK(mode == 0 || mode == 1, "ssdq_gemm: mode");
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && w.is_contiguous() && sf.is_contiguous() && gs.is_contiguous() &&
              sorted_ids.is_contiguous() && block_experts.is_contiguous() && out.is_contiguous(), "ssdq_gemm: contiguous");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kByte &&
              sf.scalar_type() == at::kFloat8_e4m3fn && gs.scalar_type() == at::kFloat &&
              sorted_ids.scalar_type() == at::kInt && block_experts.scalar_type() == at::kInt &&
              err.scalar_type() == at::kInt && prof.scalar_type() == at::kLong, "ssdq_gemm: dtypes");
  const int64_t N = w.size(1), K = w.size(2) * 2;
  TORCH_CHECK(N % BN == 0 && K % 128 == 0 && a.size(1) == K && sf.size(2) * 16 == K && out.size(1) == N,
              "ssdq_gemm: shapes");
  const int64_t nb = block_experts.numel();
  TORCH_CHECK(a.size(0) == nb * BM && sorted_ids.numel() <= a.size(0), "ssdq_gemm: rows");
  TORCH_CHECK(mode == 0 ? out.size(0) == a.size(0) : out.size(0) == num_routes, "ssdq_gemm: out rows");
  if (nb == 0) return 0;
  EncodeTiledFn enc = get_encode();
  if (enc == nullptr) return 1;
  Maps maps;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(a.size(0))};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(K) * 2};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(BM)};
  const cuuint32_t es[2] = {1, 1};
  const CUresult rc = enc(&maps.m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.data_ptr(), dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) return 1000 + static_cast<int64_t>(rc);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(N / BN), static_cast<unsigned>(nb));
  ssdq_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
      maps, reinterpret_cast<const uint8_t*>(w.data_ptr()), reinterpret_cast<const uint8_t*>(sf.data_ptr()),
      reinterpret_cast<const float*>(gs.data_ptr()), reinterpret_cast<const int*>(sorted_ids.data_ptr()),
      reinterpret_cast<const int*>(block_experts.data_ptr()), static_cast<int>(gs.numel()),
      static_cast<int>(num_routes), static_cast<int>(N), static_cast<int>(K), static_cast<int>(mode),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(out.size(1)), reinterpret_cast<int*>(err.data_ptr()),
      use_prof ? reinterpret_cast<long long*>(prof.data_ptr()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return 0;
}
"""


def _ssdq_mma_call():
    """The wgmma m64n256k16 SS call (bf16 x bf16 -> fp32, both operands K-major from shared memory
    descriptors, scale-d = 1, scales 1, no transpose), as inline asm over the 128 accumulators."""
    d = ", ".join(f"%{i}" for i in range(128))
    outs = ", ".join(f'"+f"(acc[{i}])' for i in range(128))
    return "\n".join([
        "#if defined(__CUDA_ARCH_FEAT_SM90_ALL)",
        '        asm volatile("{\\n\\t.reg .pred p;\\n\\t"',
        '                     "setp.ne.b32 p, %130, 0;\\n\\t"',
        '                     "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "',
        f'                     "{{{d}}}, "',
        '                     "%128, %129, p, 1, 1, 0, 0;\\n\\t}\\n"',
        f"                     : {outs}",
        '                     : "l"(da), "l"(db), "r"(1));',
        "#endif",
    ])


def _ssdq_src():
    return _SSDQ_HEAD + _SSDQ_BODY.replace("@MMA_CALL@", _ssdq_mma_call())


@triton.jit
def _silu_rows_kernel(c1_ptr, c2_ptr, n_padded_ptr, SIZE_N: tl.constexpr, BLOCK_R: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    """c2[r] = silu(c1[r, :SIZE_N]) * c1[r, SIZE_N:] for every sorted row r < n_padded (a multiple of the
    128-row block, so no row mask); the padded rows hold zeros and stay zeros."""
    r0 = tl.program_id(0) * BLOCK_R
    if r0 < tl.load(n_padded_ptr):
        rows = (r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        src = c1_ptr + rows[:, None] * (2 * SIZE_N) + cols[None, :]
        g = tl.load(src).to(tl.float32)
        u = tl.load(src + SIZE_N).to(tl.float32)
        y = g / (1.0 + tl.exp(-g)) * u
        tl.store(c2_ptr + rows[:, None] * SIZE_N + cols[None, :], y.to(tl.bfloat16))


def _silu_rows(c1, c2, n_padded, size_n):
    grid = (triton.cdiv(c1.shape[0], SSDQ_SILU_BLOCK_R), size_n // SSDQ_SILU_BLOCK_N)
    _silu_rows_kernel[grid](c1, c2, n_padded, SIZE_N=size_n, BLOCK_R=SSDQ_SILU_BLOCK_R,
                            BLOCK_N=SSDQ_SILU_BLOCK_N, num_warps=4)


def _experts_ssdq(x, topk_weights, local_ids, state, tracer=None, use_prof=False):
    """The experts on the dequant-producer SS-wgmma kernel, on ids already mapped to this GPU's experts
    (0-31, -1 elsewhere). align (block 128, local routes only) -> gather rows into sorted order ->
    w13 -> silu*up (row to row) -> w2 (scatter to the route slot) -> master's weighted sum. tracer, only
    passed by prepare(), is called around every launch (it syncs and prints there)."""
    ext, bank = state["ssdq_ext"], state["bank"]
    num_tokens, size_k = x.shape
    gemm1_n = bank["w13"].shape[1]
    size_n = gemm1_n // 2
    num_slots = num_tokens * TOP_K
    topk_weights = topk_weights.to(torch.float32).contiguous()
    err, prof = state["ssdq_err"], state["ssdq_prof"]
    if tracer:
        tracer("align")
    _, (sorted_ids, block_experts, n_padded) = _align_parallel(local_ids, 0, LOCAL_EXPERTS, SSDQ_BM)
    if tracer:
        tracer("align done", sorted_ids, n_padded)
    rows = block_experts.shape[0] * SSDQ_BM
    xs = torch.empty((rows, size_k), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("ssdq_gather")
    ext.ssdq_gather(x.contiguous(), sorted_ids, block_experts, num_slots, LOCAL_EXPERTS, xs)
    if tracer:
        tracer("ssdq_gather done", xs, n_padded)
    c1 = torch.empty((rows, gemm1_n), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("ssdq_gemm w13")
    rc = ext.ssdq_gemm(0, xs, bank["w13"], bank["w13_sf"], bank["w13_gs"], sorted_ids, block_experts, num_slots,
                       c1, err, prof, 1 if use_prof else 0)
    if tracer:
        tracer("ssdq_gemm w13 done", c1, n_padded)
        tracer.rc("w13", rc)
    c2 = torch.empty((rows, size_n), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("silu_rows")
    _silu_rows(c1, c2, n_padded, size_n)
    if tracer:
        tracer("silu_rows done", c2, n_padded)
    # The weighted sum skips remote slots when prepare() proved it (state['zero_nonlocal_w2'] False);
    # otherwise remote rows must read as zeros.
    if state["zero_nonlocal_w2"]:
        y = torch.zeros((num_slots, size_k), device=x.device, dtype=x.dtype)
    else:
        y = torch.empty((num_slots, size_k), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("ssdq_gemm w2")
    rc = ext.ssdq_gemm(1, c2, bank["w2"], bank["w2_sf"], bank["w2_gs"], sorted_ids, block_experts, num_slots,
                       y, err, prof, 1 if use_prof else 0)
    if tracer:
        tracer("ssdq_gemm w2 done", y, None, local_ids)
        tracer.rc("w2", rc)
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, TOP_K, size_k), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=local_ids)


_SSDQ_TEMPLATED_ACCESSOR = re.compile(
    r"(?:\.|->)\s*(?:\w*data_ptr|size|stride|item|sizes|strides|accessor|packed_accessor\w*|data|to|toType)\s*<")


def _ssdq_source_audit():
    """Cheap pre-build audit of the C++/CUDA source. libtorch exports Tensor::data_ptr<int64_t> (long) but
    not <long long>, so a templated accessor can link and then fail at import with an undefined symbol
    (the ssdq stage-1 failure). Returns a list of 'line N: text' offenders (empty = clean)."""
    bad = []
    for name, src in (("cpp", _SSDQ_CPP), ("cuda", _ssdq_src())):
        for n, line in enumerate(src.splitlines(), 1):
            if "data_ptr<" in line or "data_ptr <" in line or _SSDQ_TEMPLATED_ACCESSOR.search(line):
                bad.append(f"{name} line {n}: {line.strip()[:120]}")
    return bad


def _ssdq_build():
    """Compile the module (prepare only, never during capture): (ext or None, error text). sm_90a only:
    cpp_extension's JIT path appends -gencode flags from TORCH_CUDA_ARCH_LIST (else from the device, a
    plain compute_90 pass that cannot assemble wgmma), so the list is set to 9.0a for the build and
    restored after; the wgmma body is also behind __CUDA_ARCH_FEAT_SM90_ALL."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    ext, err_text = None, ""
    try:
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load_inline
        print(f"{SSDQ_TAG} torch {torch.__version__} cuda {torch.version.cuda}; device capability "
              f"{torch.cuda.get_device_capability()}", flush=True)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            arch_flags = cpp_extension._get_cuda_arch_flags()
        except Exception as e:  # this torch does not know the 'a' suffix: default list + the guards
            print(f"{SSDQ_TAG} TORCH_CUDA_ARCH_LIST=9.0a rejected ({e!r}); default list, error word 2 decides",
                  flush=True)
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
            arch_flags = [f"<unknown>"]
        print(f"{SSDQ_TAG} build arch flags: TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r} -> "
              f"{arch_flags}; extra_cuda_cflags {SSDQ_CUDA_CFLAGS}", flush=True)
        ext = load_inline(name=SSDQ_EXT_NAME, cpp_sources=_SSDQ_CPP, cuda_sources=_ssdq_src(),
                          functions=["ssdq_info", "ssdq_gather", "ssdq_gemm"],
                          extra_cuda_cflags=list(SSDQ_CUDA_CFLAGS), verbose=False)
    except Exception as e:
        err_text = str(e)
        ext = None
        print(f"{SSDQ_TAG} load_inline failed after {time.perf_counter() - t0:.1f}s: {err_text[-400:]!r}",
              flush=True)
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    if ext is not None:
        print(f"{SSDQ_TAG} load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext, err_text


class _SsdqTracer:
    """prepare()-only: announces each launch with a flushed print BEFORE it, then synchronizes and
    prints AFTER it (so a fault names its kernel), checks the encode return codes, and names the first
    non-finite tensor."""

    def __init__(self, tag):
        self.tag = tag
        self.bad = None
        self.rc_bad = None

    def __call__(self, name, tensor=None, rows=None, sel=None):
        if tensor is None:
            print(f"{SSDQ_TAG} [{self.tag}] launching {name}", flush=True)
            return
        torch.cuda.synchronize()
        print(f"{SSDQ_TAG} [{self.tag}] {name}", flush=True)
        if self.bad is not None or tensor.dtype not in (torch.bfloat16, torch.float32):
            return
        if sel is not None:
            view = tensor[(sel.view(-1) >= 0)]
        elif rows is not None:
            view = tensor[:int(rows.item())]
        else:
            view = tensor
        if not bool(torch.isfinite(view.float()).all()):
            self.bad = name
            print(f"{SSDQ_TAG} [{self.tag}] FIRST NON-FINITE TENSOR after: {name}", flush=True)

    def rc(self, name, rc):
        if rc != 0 and self.rc_bad is None:
            self.rc_bad = f"{name} tensor-map encode rc {rc}"


def _ssdq_case(tokens, kind, gen, device):
    """Synthetic expert-input case: x bf16 [T, 4096], mapped local ids [T, 8], normalized weights."""
    x = torch.randn((tokens, D_EXPERT), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    ids = torch.rand((tokens, N_EXPERTS), generator=gen, device=device).topk(TOP_K, dim=1).indices.to(torch.int32)
    if kind == "skewed":
        # four experts get every token: many 128-route blocks per expert (second, third ... token tile)
        ids[:, :4] = torch.tensor([0, 31, 5, 17], dtype=torch.int32, device=device)
    elif kind == "one":
        # one local expert gets every token's first route; the other routes are remote
        ids[:, 0] = 7
        ids[:, 1:] = ids[:, 1:] % (N_EXPERTS - LOCAL_EXPERTS) + LOCAL_EXPERTS
    local_ids = torch.where(ids < LOCAL_EXPERTS, ids, torch.full_like(ids, -1)).contiguous()
    tw = torch.rand((tokens, TOP_K), generator=gen, device=device, dtype=torch.float32) + 0.1
    tw = (tw / tw.sum(dim=1, keepdim=True)).contiguous()
    return x, tw, local_ids


def _ssdq_reference_y(x, local_ids, weights, cache):
    """fp32 reference of the two expert GEMMs and the silu*up, on dequantized ORIGINAL-order weights:
    y [T * 8, 4096] bf16 (intermediates rounded to bf16 where the pipelines store bf16), zeros for
    remote routes."""
    tokens = x.shape[0]
    flat = local_ids.view(-1)
    y = torch.zeros((tokens * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=x.device)
    for e in range(LOCAL_EXPERTS):
        idx = (flat == e).nonzero().flatten()
        if idx.numel() == 0:
            continue
        if e not in cache:
            cache[e] = (nvfp4.dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]),
                        nvfp4.dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]))
        w13, w2 = cache[e]
        h = (x[idx // TOP_K].float() @ w13.t()).to(torch.bfloat16).float()
        g, u = h[:, :D_EXPERT], h[:, D_EXPERT:]
        a = (g / (1.0 + torch.exp(-g)) * u).to(torch.bfloat16).float()
        y[idx] = (a @ w2.t()).to(torch.bfloat16)
    return y


def _ssdq_metrics(out, ref):
    """The judge's relative L2 and worst row (a near-zero row is measured against a tenth of the average
    row), or inf when out is not finite."""
    out, ref = out.float(), ref.float()
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf")
    err = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
    row_ref = ref.norm(dim=-1)
    row_err = (out - ref).norm(dim=-1) / torch.maximum(row_ref, 0.1 * row_ref.mean()).clamp_min(1e-30)
    return err, float(row_err.max())


def _ssdq_graph_time(fn, device):
    """Capture fn once (after a warm-up on a side stream); returns a replay closure."""
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


SSDQ_REPLAY_TOKENS = 4096     # a judge test shape inside the path's range
SSDQ_REPLAY_KINDS = ("random", "skewed", "one", "random")


def _ssdq_replay_check(state, weights, cache, device, gen):
    """prepare() only. The judge prepares two states (setup copies: 2) and rotates them across
    repetitions with refilled inputs. Emulate it: a second state with its own bank copy, error word
    and counters; one CUDA graph of _experts_ssdq per state over that state's own static inputs; then
    rounds that refill a state's inputs with a fresh routing, replay its graph and check it against the
    fp32 reference, alternating A, B / B, A. A launch input shared across states (a module global, a
    tensor map or buffer frozen for the other state) shows up as a mismatch. Returns None or why."""
    st2 = dict(state)
    st2["bank"] = {k: v.clone() for k, v in state["bank"].items()}
    st2["ssdq_err"] = torch.zeros_like(state["ssdq_err"])
    st2["ssdq_prof"] = torch.zeros_like(state["ssdq_prof"])
    states = (state, st2)
    t = SSDQ_REPLAY_TOKENS
    bufs, graphs, holders = [], [], []
    try:
        for st in states:
            b = _ssdq_case(t, "random", gen, device)
            holder = {}

            def fn(st=st, b=b, holder=holder):
                holder["out"] = _experts_ssdq(b[0], b[1], b[2], st)

            graphs.append(_ssdq_graph_time(fn, device))
            bufs.append(b)
            holders.append(holder)
        for r, kind in enumerate(SSDQ_REPLAY_KINDS):
            for i in ((0, 1) if r % 2 == 0 else (1, 0)):
                x, tw, lids = _ssdq_case(t, kind if i == 0 else SSDQ_REPLAY_KINDS[-1 - r], gen, device)
                for dst, src in zip(bufs[i], (x, tw, lids)):
                    dst.copy_(src)
                states[i]["ssdq_err"].zero_()
                graphs[i].replay()
                torch.cuda.synchronize()
                got = holders[i]["out"].clone()
                ref_y = _ssdq_reference_y(x, lids, weights, cache)
                ref = fused_post_expert_norm_weighted_sum(
                    ref_y.view(t, TOP_K, D_EXPERT), tw, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
                    weight_first=True, slot_expert_ids=lids)
                mar = _experts_marlin(x, tw, lids, state)
                torch.cuda.synchronize()
                word = int(states[i]["ssdq_err"][0].item())
                marks = states[i]["ssdq_err"].tolist()
                if marks[1] <= 0 or marks[2] <= 0:  # this replay did not launch the ssdq kernels
                    word = word or 99
                e_new, w_new = _ssdq_metrics(got, ref)
                e_mar, _ = _ssdq_metrics(mar, ref)
                bound = max(2.0 * e_mar, 8e-3)
                print(f"{SSDQ_TAG} replay round {r} state {'AB'[i]} T={t}: ssdq rel L2 {e_new:.3e} worst row "
                      f"{w_new:.3e}; bound {bound:.3e}; error word {word}; launch marker w13 {marks[1]} "
                      f"w2 {marks[2]}", flush=True)
                if word != 0 or not (e_new <= bound and w_new <= 5.0 * bound):
                    return f"round {r} state {'AB'[i]} rel L2 {e_new:.3e} worst row {w_new:.3e} word {word}"
        print(f"{SSDQ_TAG} two-state replay check ok ({len(SSDQ_REPLAY_KINDS)} rounds x 2 states)", flush=True)
        return None
    finally:
        del graphs, holders, bufs, st2, states
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _ssdq_gate(state, device):
    """Interleaved CUDA-graph replays with a cold-L2 flush of the whole expert block (align, GEMMs,
    silu, weighted sum) at the scored turn_4 T: Marlin pair vs the new path, SSDQ_ALTERNATIONS
    alternations, order swapped each time. Returns (marlin_us, ssdq_us), medians."""
    gen = torch.Generator(device=device).manual_seed(20261025)
    x, tw, local_ids = _ssdq_case(SSDQ_GATE_TOKENS, "random", gen, device)
    g_m = _ssdq_graph_time(lambda: _experts_marlin(x, tw, local_ids, state), device)
    g_n = _ssdq_graph_time(lambda: _experts_ssdq(x, tw, local_ids, state), device)
    flush = torch.empty(256 * 2 ** 20 // 4, dtype=torch.float32, device=device)  # > the 50 MB L2
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def once(graph):
        flush.zero_()
        ev[0].record()
        graph.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    t_m, t_n = [], []
    once(g_m)
    once(g_n)
    for i in range(SSDQ_ALTERNATIONS):
        if i % 2 == 0:
            t_m.append(once(g_m))
            t_n.append(once(g_n))
        else:
            t_n.append(once(g_n))
            t_m.append(once(g_m))
    t_m.sort()
    t_n.sort()
    return t_m[len(t_m) // 2], t_n[len(t_n) // 2]


def _ssdq_setup(state, weights, device, t_start):
    """prepare() only. Build, canary, self-check against an fp32 reference (calibrated on Marlin's own
    error), producer/consumer-wait counters, and the cold-L2 graph gate at turn_4. Sets
    state['ssdq_ok']; every failure prints one OFF line and leaves Marlin as master has it."""
    state["ssdq_ok"] = False
    why = None
    phases = []
    t_phase = [time.perf_counter()]
    t_pre = t_phase[0] - t_start

    def phase(name):
        now = time.perf_counter()
        phases.append((name, now - t_phase[0]))
        t_phase[0] = now
        print(f"{SSDQ_TAG} phase {name}: {phases[-1][1]:.1f}s; prepare total {now - t_start:.1f}s; "
              f"budget remaining {SSDQ_BUDGET_S - (now - t_start):.1f}s of {SSDQ_BUDGET_S:.0f}s", flush=True)

    print(f"{SSDQ_TAG} prepare used {t_pre:.1f}s before this setup (budget {SSDQ_BUDGET_S:.0f}s, "
          f"remaining {SSDQ_BUDGET_S - t_pre:.1f}s)", flush=True)
    try:
        if not state.get("align_par_ok", False):
            why = "master's parallel align is off (the path needs it)"
        elif time.perf_counter() - t_start > SSDQ_BUDGET_S:
            why = f"prepare already used {time.perf_counter() - t_start:.0f}s > {SSDQ_BUDGET_S:.0f}s budget"
        if why:
            print(f"{SSDQ_TAG} OFF: {why} (PATH CHOSEN = marlin at every T)", flush=True)
            return
        offenders = _ssdq_source_audit()
        if offenders:
            print(f"{SSDQ_TAG} OFF: source audit found templated accessors (long/long long trap): "
                  f"{offenders[:4]} (PATH CHOSEN = marlin at every T)", flush=True)
            return
        print(f"{SSDQ_TAG} source audit ok: no data_ptr<..> or templated size/item/data/to accessor", flush=True)
        phase("audit")
        ext, err_text = _ssdq_build()
        phase("build")
        if ext is None:
            print(f"{SSDQ_TAG} OFF: build failed (PATH CHOSEN = marlin at every T)", flush=True)
            return
        info = ext.ssdq_info()
        print(f"{SSDQ_TAG} kernel ssdq_kernel: regs={info[0]} local_bytes={info[1]} static_smem={info[2]} "
              f"dyn_smem={info[3]} ctas_per_sm={info[4]} setattr_rc={info[5]} binaryVersion={info[6]} "
              f"ptxVersion={info[7]} attr_ok={info[8]}", flush=True)
        if not info[8] or info[4] < 1:
            print(f"{SSDQ_TAG} OFF: cannot launch (attributes/occupancy) (PATH CHOSEN = marlin)", flush=True)
            return
        state["ssdq_ext"] = ext
        state["ssdq_err"] = torch.zeros(3, dtype=torch.int32, device=device)
        state["ssdq_prof"] = torch.zeros(8, dtype=torch.int64, device=device)
        gen = torch.Generator(device=device).manual_seed(20261025)

        # Canary: one tiny launch of every kernel, each followed by a sync and a print; then the error
        # word, the encode return codes and a finiteness check that names the first bad tensor.
        x, tw, lids = _ssdq_case(130, "random", gen, device)
        tracer = _SsdqTracer("canary")
        out = _experts_ssdq(x, tw, lids, state, tracer=tracer)
        torch.cuda.synchronize()
        word = int(state["ssdq_err"][0].item())
        marks = state["ssdq_err"].tolist()
        print(f"{SSDQ_TAG} canary launch marker: w13 CTAs started {marks[1]}, w2 CTAs started {marks[2]}", flush=True)
        if marks[1] <= 0 or marks[2] <= 0:
            print(f"{SSDQ_TAG} OFF: canary kernels did not run (launch marker w13 {marks[1]}, w2 {marks[2]}) "
                  f"(PATH CHOSEN = marlin)", flush=True)
            return
        if word != 0 or tracer.rc_bad or tracer.bad or not bool(torch.isfinite(out.float()).all()):
            print(f"{SSDQ_TAG} OFF: canary failed: error word {word} ({SSDQ_ERR_TEXT.get(word, '?')}), "
                  f"{tracer.rc_bad}, first non-finite tensor after {tracer.bad} (PATH CHOSEN = marlin)", flush=True)
            return
        print(f"{SSDQ_TAG} canary ok: error word 0, every tensor finite", flush=True)
        phase("canary")

        # Self-check: forced routings (random, skewed to four experts, one expert) at T=1506, 3012 and
        # 4096 (a judge test shape), vs an fp32 dequant reference; bound = max(2 x Marlin's own error, 8e-3).
        cache = {}
        cases = ((1506, "random"), (3012, "random"), (4096, "skewed"), (1100, "one"), (4096, "random"))
        for tokens, kind in cases:
            x, tw, lids = _ssdq_case(tokens, kind, gen, device)
            state["ssdq_err"].zero_()
            ref_y = _ssdq_reference_y(x, lids, weights, cache)
            ref = fused_post_expert_norm_weighted_sum(
                ref_y.view(tokens, TOP_K, D_EXPERT), tw, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
                weight_first=True, slot_expert_ids=lids)
            tracer = _SsdqTracer(f"T={tokens} {kind}")
            new = _experts_ssdq(x, tw, lids, state, tracer=tracer)
            torch.cuda.synchronize()
            mar = _experts_marlin(x, tw, lids, state)
            torch.cuda.synchronize()
            word = int(state["ssdq_err"][0].item())
            marks = state["ssdq_err"].tolist()
            e_new, w_new = _ssdq_metrics(new, ref)
            e_mar, w_mar = _ssdq_metrics(mar, ref)
            bound = max(2.0 * e_mar, 8e-3)
            print(f"{SSDQ_TAG} self-check T={tokens} {kind}: ssdq rel L2 {e_new:.3e} worst row {w_new:.3e}; "
                  f"marlin rel L2 {e_mar:.3e} worst row {w_mar:.3e}; bound {bound:.3e}; error word {word}; "
                  f"{int((lids >= 0).sum())} local routes; launch marker w13 {marks[1]} w2 {marks[2]}", flush=True)
            if marks[1] <= 0 or marks[2] <= 0:
                word = word or 99
            if word != 0 or tracer.rc_bad or tracer.bad or not (e_new <= bound and w_new <= 5.0 * bound):
                print(f"{SSDQ_TAG} OFF: self-check failed at T={tokens} {kind} (error word {word}, "
                      f"{tracer.rc_bad}, non-finite after {tracer.bad}) (PATH CHOSEN = marlin)", flush=True)
                del cache
                torch.cuda.empty_cache()
                return
        phase("self-check")
        replay_why = _ssdq_replay_check(state, weights, cache, device, gen)
        del cache
        torch.cuda.empty_cache()
        if replay_why:
            print(f"{SSDQ_TAG} OFF: two-state replay check failed: {replay_why} (PATH CHOSEN = marlin)", flush=True)
            return

        phase("replay-check")
        # One profiled launch: producer wait vs consumer wait per CTA (a dequant-bound producer shows a
        # small producer wait and a large consumer wait).
        gen = torch.Generator(device=device).manual_seed(20261025)
        x, tw, lids = _ssdq_case(SSDQ_GATE_TOKENS, "random", gen, device)
        state["ssdq_prof"].zero_()
        _experts_ssdq(x, tw, lids, state, use_prof=True)
        torch.cuda.synchronize()
        prof = state["ssdq_prof"].tolist()
        for mode, name in ((0, "w13"), (1, "w2")):
            wait_p, wait_c, ctas, cyc = prof[4 * mode:4 * mode + 4]
            if ctas:
                print(f"{SSDQ_TAG} counters {name} T={SSDQ_GATE_TOKENS}: {ctas} CTAs, per CTA {cyc / ctas:.0f} cycles, "
                      f"producer waiting on empty stages {wait_p / ctas:.0f} cycles, consumer waiting on full "
                      f"stages {wait_c / (2 * ctas):.0f} cycles per warpgroup "
                      f"(producer-bound if the consumer wait is large and the producer wait small)", flush=True)

        phase("counters")
        # Gate: cold-L2 CUDA-graph replays, whole expert block, new vs Marlin pair, turn_4 T.
        t_m, t_n = _ssdq_gate(state, device)
        phase("gate")
        ratio = t_n / t_m
        ok = ratio <= SSDQ_GATE_RATIO
        state["ssdq_ok"] = bool(ok)
        print(f"{SSDQ_TAG} GATE T={SSDQ_GATE_TOKENS} cold-L2 graph medians over {SSDQ_ALTERNATIONS} alternations: "
              f"marlin pair {t_m:.1f} us, ssdq {t_n:.1f} us, ratio {ratio:.3f} (ON at <= {SSDQ_GATE_RATIO})", flush=True)
        print(f"{SSDQ_TAG} PATH CHOSEN T in ({SSDQ_MIN_TOKENS}, {MARLIN_MAX_TOKENS}]: "
              f"{'ssdq (faster: ' + format(t_n, '.1f') + ' < ' + format(t_m, '.1f') + ' us)' if ok else f'marlin (ssdq ratio {ratio:.3f} not <= {SSDQ_GATE_RATIO})'}; "
              f"all other T: unchanged", flush=True)
        if not ok:
            print(f"{SSDQ_TAG} OFF: gate ratio {ratio:.3f} > {SSDQ_GATE_RATIO}", flush=True)
        assert (not ok) or t_n < t_m
    except Exception as e:  # never raises into prepare(); a device fault would have poisoned the context
        state["ssdq_ok"] = False
        print(f"{SSDQ_TAG} OFF: setup raised {e!r} (PATH CHOSEN = marlin)", flush=True)
    finally:
        # Last lines of prepare(): the archived log keeps the tail.
        total = time.perf_counter() - t_start
        print(f"{SSDQ_TAG} prepare per-phase seconds: before-ssdq {t_pre:.1f}, "
              + ", ".join(f"{n} {d:.1f}" for n, d in phases)
              + f"; total {total:.1f}s; budget remaining {SSDQ_BUDGET_S - total:.1f}s of {SSDQ_BUDGET_S:.0f}s",
              flush=True)
        ssdq_on = bool(state.get("ssdq_ok"))
        print(f"{SSDQ_TAG} PATH CHOSEN T <= {SSDQ_MIN_TOKENS}: marlin/unchanged; "
              f"T in ({SSDQ_MIN_TOKENS}, {MARLIN_MAX_TOKENS}]: {'ssdq' if ssdq_on else 'marlin'} "
              f"(gate ratio constant SSDQ_GATE_RATIO = {SSDQ_GATE_RATIO}); T > {MARLIN_MAX_TOKENS}: triton/unchanged",
              flush=True)


# ---------------------------------------------------------------------------------------------
# experts-turn-rega-s4-port-t4096 (rewrite stage 1, replaces the Marlin pair at 200 < T <= 4096 where it
# wins its gate): the register-A swap-AB wgmma NVFP4 grouped GEMM of muse_moe_w4a16's s5 copy-safe build
# (git show hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced:muse_moe_w4a16/submission.py;
# patch: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced.patch).
#
# Per expert the weight tile (128 rows x 64 k: codes + e4m3 scales, a lossless retiling of the stored NVFP4
# bytes made once in prepare) is one bulk copy; each consumer warp dequantizes its 16 rows IN REGISTERS
# straight into the bf16 A fragments of an m64nNk16 register-A wgmma (swap-AB: the weight rows are the MMA's
# M, the routed tokens its N: 16/32/64/128 tokens per expert chunk, a 2-D TMA box of the gathered rows is
# B). The w13 epilogue computes silu(gate) * up in fp32 from the fp32 accumulators and rounds to bf16 once
# (no stored BF16 copy of any weight, no intermediate bf16 rounding); w2 scatters bf16 rows to the route
# slot of master's y [T * 8, 4096] and master's post-expert norm weighted sum finishes.
#
# Copy safety (the judge prepares two states and rotates them, copying tensors): the TMA maps are encoded
# in C++ on EVERY call from the scratch tensor passed in that call and handed to the kernel as a
# __grid_constant__ argument; nothing at file scope holds a pointer, a map or a buffer; the scratch (xs, hs)
# is allocated per call (graph pool, like master's ssdq path); the only per-state device data are the
# retiled bank and the int32[3] error word + launch markers (cloned with the state). There is no pointer
# equality gate and no "live state" check in the timed call: kernel_fn dispatches on the token count and on
# the prepare-time flags state['rega_ok'][class] only.
#
# Dispatch order in _experts for T <= MARLIN_MAX_TOKENS: rega if state['rega_ok'][class]; else ssdq if
# T > SSDQ_MIN_TOKENS and state['ssdq_ok']; else Marlin. Classes: 'low' T <= REGA_LOW_MAX (gated at T = 753
# against whatever _experts dispatches there: master's whole Marlin block), 'high' T in (REGA_LOW_MAX,
# MARLIN_MAX_TOKENS] (gated at T = 3012 against rega-off dispatch, ssdq and Marlin in one interleaved loop).
# ---------------------------------------------------------------------------------------------

REGA_TAG = "[rega]"
REGA_BSM = 16
REGA_MIN_TOKENS = 200          # the path serves REGA_MIN_TOKENS < T <= MARLIN_MAX_TOKENS
REGA_LOW_MAX = 1024            # class 'low': T <= this; class 'high': above it
REGA_GATE_TOKENS = {"low": 753, "high": 3012}
REGA_GATE_RATIO = 0.95         # ON only when rega <= this x the fastest alternative (min of ssdq, Marlin, dispatch)
REGA_ALTERNATIONS = 9          # >= 7 interleaved alternations
REGA_BUDGET_S = 200.0          # no setup if prepare() has already used this many seconds
REGA_COPY_DEADLINE_S = 255.0   # the copy / replay check must start before this many seconds into prepare()
REGA_BUCKETS = (16, 32, 64, 128)
REGA_THREADS = 384
REGA_STAGES = 4
REGA_BK = 128
REGA_REC = 4608
REGA_STAGE_BYTES = 2 * REGA_REC + 2 * 128 * 128
REGA_EPI_BYTES = {True: 128 * (64 * 2 + 16), False: 128 * (128 * 2 + 16)}
REGA_SMEM = {g: 1024 + REGA_STAGES * REGA_STAGE_BYTES + REGA_EPI_BYTES[g] for g in (True, False)}
REGA_STATIC_SMEM_BOUND = 16384  # tables + barriers (the C++ static_assert uses the same bound)
REGA_SMEM_LIMIT = 227 * 1024
REGA_EXT_NAME = "muse_prefill_rega_v2"  # v2: rega_kernel<G, BIG> (first-turn capacity class)
REGA_GENCODE = "-gencode=arch=compute_90a,code=sm_90a"
REGA_CUDA_CFLAGS = ["-O3", "-std=c++20", "-lineinfo", REGA_GENCODE]
REGA_MAGIC = 0x57474d4d        # block 0 of each kernel writes it to err[1] (w13) / err[2] (w2)
REGA_ERR_TEXT = {0: "ok", 1: "mbarrier wait timed out", 2: "non-sm_90a image ran (wgmma compiled out)",
                 3: "block or chunk table overflow", 4: "an expert owns two runs of blocks (align not sorted)"}
# experts-turn-rega-s4-first-turn-n128 (rewrite stage 2): class 'ft' = MARLIN_MAX_TOKENS < T <= REGA_FT_MAX, gated
# at T = 16384 (soft routing) against master's Triton block (the rega-off dispatch). The kernel's first-turn
# capacity class (rega_kernel<G, true>) holds the worst case at REGA_FT_MAX (every route local); T > REGA_FT_MAX
# always stays on Triton (shape-based).
REGA_FT_MAX = 16384
REGA_FT_GATE_TOKENS = 16384
REGA_MAXBLK = 2304             # small class block table (C++ MAXBLK); larger tables launch the first-turn class
REGA_MAXBLK_FT = 8224          # C++ MAXBLK_FT
REGA_MAXC_FT = 1088            # C++ MAXC_FT
REGA_FT_START_DEADLINE_S = 150.0   # no ft checks if prepare() has already used this many seconds
# (4500, local): 36480 slots = 2280 blocks, the largest stage-1-class table above 4096 (class boundary)
REGA_FT_SELF_CHECKS = ((4500, "local"), (8192, "random"), (8192, "local"), (16384, "random"), (16384, "skewed"),
                       (16384, "one"), (16384, "local"))
REGA_FT_REPLAY_PLAN = ((8192, ("random", "local")), (16384, ("random", "skewed", "one", "local")))
REGA_STATE_KEYS = ("rega_ext", "rega_bank", "rega_err", "rega_grid", "post_expert_w", "zero_nonlocal_w2",
                   "align_par_ok")

_REGA_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> rega_init();
void rega_gather(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
               int64_t num_routes, int64_t n_experts, torch::Tensor xs);
void rega_gemm(int64_t which, torch::Tensor bsrc, torch::Tensor wbank,
             torch::Tensor gs, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
             int64_t num_routes, int64_t n_tiles, int64_t K, torch::Tensor out, int64_t grid, torch::Tensor err);
"""

_REGA_HEAD = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int THREADS = 384;
constexpr int STAGES = 4;
constexpr int BK = 128;                          // k per pipeline stage (two 64-k records)
constexpr int REC = 4608;                        // record: 4096 B codes + 512 B scales
constexpr int REC_CODES = 4096;
constexpr int WST = 2 * REC;                     // 9216 B of weight per stage
constexpr int SLAB_HALF = 128 * 128;             // one SWIZZLE_128B box: up to 128 rows x 64 bf16
constexpr int STAGE_BYTES = WST + 2 * SLAB_HALF; // 41984
constexpr int MAXBLK = 2304;                     // block-table capacity (T = 4096: 2078 blocks of 16)
constexpr int MAXC = 512;                        // chunk-table capacity (<= 8 blocks per chunk)
// first-turn capacity class (T <= 16384): worst case 131072 local routes + 32 x 15 padding = 8222 blocks of 16
// and <= 8222 / 8 + 32 = 1059 chunks. Its block-expert table is read from global (__ldg), only the chunk
// tables live in shared memory (2 x 1088 x 4 B).
constexpr int MAXBLK_FT = 8224;
constexpr int MAXC_FT = 1088;
constexpr int MAXE = 32;                         // local experts
constexpr int PITCH13 = 64 * 2 + 16;             // staging row pitch, w13 (64 bf16 + pad)
constexpr int PITCH2 = 128 * 2 + 16;             // staging row pitch, w2 (128 bf16 + pad)
static_assert(WST % 1024 == 0 && STAGE_BYTES % 1024 == 0, "stage alignment");
template <bool G>
struct Smem {
  static constexpr int EPI = 128 * (G ? PITCH13 : PITCH2);
  static constexpr int BYTES = 1024 + STAGES * STAGE_BYTES + EPI;
};
static_assert(Smem<false>::BYTES + 16384 <= 227 * 1024, "shared memory budget");
// Capacity class: BIG = false is the T <= 4096 build (tables and smem exactly as stage 1); BIG = true serves
// 4096 < T <= 16384 with the block-expert table in global memory.
template <bool BIG>
struct Cap {
  static constexpr int BLK = BIG ? MAXBLK_FT : MAXBLK;
  static constexpr int C = BIG ? MAXC_FT : MAXC;
  static constexpr int SBE = BIG ? 1 : MAXBLK;   // shared block-expert table entries
};
// static tables + barriers / per-expert arrays stay under the 16384 B static bound in both classes
static_assert((Cap<false>::SBE + 2 * Cap<false>::C) * 4 + 1024 <= 16384, "static smem, small class");
static_assert((Cap<true>::SBE + 2 * Cap<true>::C) * 4 + 1024 <= 16384, "static smem, first-turn class");
// Block-expert table views: the small class indexes its shared copy (a plain pointer, as stage 1); the large
// class reads global memory through the read-only path.
struct GTab {
  const int* p;
  __device__ __forceinline__ int operator[](int i) const { return __ldg(p + i); }
};
template <bool BIG>
struct BeSel;
template <>
struct BeSel<false> {
  using T = const int*;
  static __device__ __forceinline__ T make(const int* s, const int* g) { return s; }
};
template <>
struct BeSel<true> {
  using T = GTab;
  static __device__ __forceinline__ T make(const int* s, const int* g) { return GTab{g}; }
};

struct __align__(64) Maps { CUtensorMap m[4]; };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n\t.reg .pred P;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, P;\n\t}\n"
               : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok != 0;
}
// Bounded wait: after 2^26 cycles without the phase, record an error word and stop waiting
// (dead), so a broken pipeline ends in a failed self-check instead of a hang.
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity, int* err, bool& dead) {
  if (dead || mbar_try(a, parity)) return;
  const long long t0 = clock64();
  while (!mbar_try(a, parity)) {
    if (clock64() - t0 > (1ll << 26)) {  // ~34 ms, >1000x any real stage wait
      dead = true;
      atomicExch(err, 1);
      return;
    }
  }
}
// Plain 1-D bulk copy (no createpolicy / .L2::cache_hint operand: that helper family never ran on
// the judge and is the prime suspect of the swap-AB sibling's illegal instruction 715).
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, uint64_t map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// wgmma / setmaxnreg exist only for sm_90a: every such body is compiled only when
// __CUDA_ARCH_FEAT_SM90_ALL is defined, so a stray plain compute_90 pass still assembles; in that
// pass rega_kernel only sets error word 2 and returns (the path is then turned off in setup).
__device__ __forceinline__ void wgmma_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
#endif
}
template <int NA>
__device__ __forceinline__ void fence_acc(float (&d)[NA]) {
#pragma unroll
  for (int i = 0; i < NA; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
__device__ __forceinline__ void named_bar(int id, int count) {
  asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void stsm_x2_t(uint32_t a, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" :: "r"(a), "r"(r0), "r"(r1) : "memory");
}
__device__ __forceinline__ void stsm_x4_t(uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
               :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
// K-major SWIZZLE_128B shared-memory matrix descriptor: start address >> 4 (bits 0-13), LBO 1
// (unused for swizzled K-major), SBO 1024 B between 8-row core groups (bits 32-45), layout type
// 1 = 128B swizzle (bits 62-63), base offset 0 (every box base is 1024-aligned; the k16 advance
// of +32 B stays inside the first 128 B, so address bits 7-9 stay 0).
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}

// Dequant, verbatim from the swap-AB sibling (cuda-nvfp4-swapab-mma-persistent-small-m).
// Four E4M3 bytes -> two f16x2: lo from bytes 0,1 (byte 0 in the low half), hi from bytes 2,3.
__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t bf16x2_bcast(float v) {
  __nv_bfloat162 b = __float2bfloat162_rn(v);
  return *reinterpret_cast<uint32_t*>(&b);
}
// a * b (+ -0), bf16x2, round to nearest; subnormal inputs are kept.
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
// 8 E2M1 codes -> the 4 bf16x2 A registers, exact: (q & 0x80008000) | ((q & 0x70007000) >> 6) is
// e2m1 * 2^-126 in each half (nibbles 3 - i and 7 - i); sg / sg8 are the row g / g + 8 scales
// times 2^119, so each product is e2m1 * sf * 2^-7 (at most 6 significant bits, exact in bf16).
__device__ __forceinline__ void dequant8(uint32_t q, uint32_t sg, uint32_t sg8, uint32_t (&a)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t v = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    a[i] = mul_bf16x2(v, (i & 1) ? sg8 : sg);
    q <<= 4;
  }
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&b);
}
// silu(gate) * up entirely in fp32 from the fp32 accumulators; the caller rounds the product to bf16
// ONCE (pack_bf16). No intermediate bf16 rounding of gate or up.
__device__ __forceinline__ float silu_mul(float hg, float hu) {
  return hg / (1.f + expf(-hg)) * hu;
}

template <int NB>
struct Wg;
"""

_REGA_BODY = r"""
struct Item {
  int j, e, slot0, nbk, bucket, multi;
};
// Item it of the (expert, tile, chunk) order. Expert group = chunks [f, f + k) of one expert
// (f = gfirst[e], k = gcount[e]), items [f * T, (f + k) * T); it / T always lands inside its own group.
// be: the block-expert table view (shared pointer in the small class, GTab in the first-turn class).
template <class BE>
__device__ __forceinline__ Item decode_item(int it, int T, BE be, const int* cblk, const int* cnb,
                                            const int* gfirst, const int* gcount) {
  const int cg = it / T;
  const int ge = be[cblk[cg]];
  const int f = gfirst[ge];
  const int k = gcount[ge] > 0 ? gcount[ge] : 1;
  const int local = it - f * T;
  const int c = f + local % k;
  Item m;
  m.j = local / k;
  const int b0 = cblk[c];
  m.slot0 = b0 * 16;
  m.nbk = cnb[c];
  m.e = be[b0];
  m.bucket = m.nbk <= 1 ? 0 : (m.nbk <= 2 ? 1 : (m.nbk <= 4 ? 2 : 3));
  m.multi = k > 1 ? 1 : 0;
  return m;
}

// Dequantize one pipeline stage (two 64-k records) of this warp's 16 weight rows into the 32 bf16x2
// register-A fragments (8 k16 steps x 4 registers).
__device__ __forceinline__ void dq_stage(const uint8_t* st, int cw, int lane, uint32_t (&a)[8][4]) {
  const float P119 = __int_as_float(0x7B000000);  // 2^119
  const int g = lane >> 2;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const uint4 q = *reinterpret_cast<const uint4*>(st + r * REC + cw * 512 + lane * 16);
    const uint2 sc = *reinterpret_cast<const uint2*>(st + r * REC + REC_CODES + cw * 64 + g * 8);
    uint32_t hs[4];
    e4m3x4_to_f16x2x2(sc.x, hs[0], hs[1]);
    e4m3x4_to_f16x2x2(sc.y, hs[2], hs[3]);
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 fs = __half22float2(*reinterpret_cast<const __half2*>(&hs[k]));
      dequant8(qw[k], bf16x2_bcast(fs.x * P119), bf16x2_bcast(fs.y * P119), a[r * 4 + k]);
    }
  }
}

// wgmma_wait<0> right after the commit, BEFORE the next k-step's dequant rewrites the register-A
// fragments, so no wgmma group is ever in flight while a[] is written (s3 variant A, the only variant
// kept in s4: the s3 bucket canary passed it in every N bucket).
template <int NB, bool G>
__device__ __forceinline__ void consume_item(const Item& m, uint8_t* smem, uint8_t* ep, uint32_t full0,
                                             uint32_t empty0, int& n, int KS, int cw, int lane, int ctid,
                                             const float* __restrict__ gs, const int* __restrict__ sorted_ids,
                                             int num_routes, __nv_bfloat16* __restrict__ out, int out_ld,
                                             int* err, bool& dead) {
  constexpr int NA = NB / 2;
  float acc[NA];
#pragma unroll
  for (int i = 0; i < NA; ++i) acc[i] = 0.f;
  uint32_t a0[8][4];
  for (int kb = 0; kb < KS; ++kb) {
    const int s = n % STAGES;
    mbar_wait(full0 + 8 * s, (n / STAGES) & 1, err, dead);
    const uint8_t* st = smem + s * STAGE_BYTES;
    dq_stage(st, cw, lane, a0);
    const uint32_t slab = smem_u32(st + WST);
    fence_acc<NA>(acc);
    wgmma_fence();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
      for (int k = 0; k < 4; ++k) Wg<NB>::mma(acc, a0[r * 4 + k], make_desc(slab + r * SLAB_HALF + k * 32));
    }
    wgmma_commit();
    wgmma_wait<0>();  // group retired: a0[] may be rewritten and this stage released at once
    fence_acc<NA>(acc);
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    ++n;
  }
  wgmma_wait<0>();  // as s3 variant A: nothing in flight before the epilogue reads acc
  fence_acc<NA>(acc);

  // Epilogue: stage the bf16 tile through shared memory (stmatrix .trans: token rows), then
  // coalesced 16-byte row stores. Barrier 1 = the 256 consumer threads.
  const int nvalid = m.nbk * 16;
  const float gsc = __ldg(gs + m.e);
  named_bar(1, 256);  // every consumer is done reading the previous item's staging tile
  if (G) {
    // acc[4i + 0/1]: gate (fragment row g) for tokens 8i + 2t4, +1; acc[4i + 2/3]: up (row g + 8).
#pragma unroll
    for (int i = 0; i < NB / 8; i += 2) {
      const uint32_t r0 = pack_bf16(silu_mul(acc[4 * i] * gsc, acc[4 * i + 2] * gsc),
                                    silu_mul(acc[4 * i + 1] * gsc, acc[4 * i + 3] * gsc));
      const uint32_t r1 = pack_bf16(silu_mul(acc[4 * i + 4] * gsc, acc[4 * i + 6] * gsc),
                                    silu_mul(acc[4 * i + 5] * gsc, acc[4 * i + 7] * gsc));
      const int mi = (lane >> 3) & 1, rr = lane & 7;
      const int tok = 8 * (i + mi) + rr;
      stsm_x2_t(smem_u32(ep + tok * PITCH13 + cw * 16), r0, r1);
    }
  } else {
#pragma unroll
    for (int i = 0; i < NB / 8; i += 2) {
      const uint32_t r0 = pack_bf16(acc[4 * i] * gsc, acc[4 * i + 1] * gsc);
      const uint32_t r1 = pack_bf16(acc[4 * i + 2] * gsc, acc[4 * i + 3] * gsc);
      const uint32_t r2 = pack_bf16(acc[4 * i + 4] * gsc, acc[4 * i + 5] * gsc);
      const uint32_t r3 = pack_bf16(acc[4 * i + 6] * gsc, acc[4 * i + 7] * gsc);
      const int mi = lane >> 3, rr = lane & 7;
      const int tok = 8 * (i + (mi >> 1)) + rr;
      const int col = 16 * cw + 8 * (mi & 1);
      stsm_x4_t(smem_u32(ep + tok * PITCH2 + col * 2), r0, r1, r2, r3);
    }
  }
  named_bar(1, 256);
  if (G) {
    // h_sorted[slot0 + t, 64 j .. 64 j + 64): 128 B per token row.
    for (int v = ctid; v < nvalid * 8; v += 256) {
      const int t = v >> 3, c = v & 7;
      const uint4 val = *reinterpret_cast<const uint4*>(ep + t * PITCH13 + c * 16);
      *reinterpret_cast<uint4*>(out + static_cast<size_t>(m.slot0 + t) * out_ld + m.j * 64 + c * 8) = val;
    }
  } else {
    // cache3[route, 128 j .. 128 j + 128): 256 B per token row, real routes only.
    for (int v = ctid; v < nvalid * 16; v += 256) {
      const int t = v >> 4, c = v & 15;
      const int sid = __ldg(sorted_ids + m.slot0 + t);
      if (sid >= 0 && sid < num_routes) {
        const uint4 val = *reinterpret_cast<const uint4*>(ep + t * PITCH2 + c * 16);
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(sid) * out_ld + m.j * 128 + c * 8) = val;
      }
    }
  }
}

template <bool G, bool BIG>
__global__ void __launch_bounds__(THREADS, 1)
rega_kernel(const __grid_constant__ Maps maps,
          const uint8_t* __restrict__ wbank,        // [E][T][K/64][REC]
          const float* __restrict__ gs,             // fp32 [E] global scale * 2^7
          const int* __restrict__ sorted_ids,       // block-16 align of the local ids
          const int* __restrict__ block_experts,
          const int* __restrict__ n_padded,
          int blocks_len, int num_routes, int n_tiles, int K, int n_experts,
          __nv_bfloat16* __restrict__ out, int out_ld, int* __restrict__ err) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  // Launch marker: block 0 writes a magic into err[1] (w13 instantiation) or err[2] (w2); err is int[3], one per
  // prepared state. prepare reads both back after every graph replay to prove THIS sm_90a image ran. err[0] is
  // the error word: 1 mbarrier timeout, 2 non-sm_90a image, 3 a table overflowed, 4 an expert owns two runs.
  if (blockIdx.x == 0 && threadIdx.x == 0) err[G ? 1 : 2] = 0x57474d4d;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[2 * STAGES];
  constexpr int CBLK = Cap<BIG>::BLK, CC = Cap<BIG>::C;
  __shared__ int s_be[Cap<BIG>::SBE], s_cblk[CC], s_cnb[CC];
  __shared__ int s_first[MAXE], s_gfirst[MAXE], s_gcount[MAXE];
  __shared__ int s_wsum[THREADS / 32];
  __shared__ int s_bad;  // set when an expert owns two runs: the chunk groups would not be contiguous
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* ep = smem + STAGES * STAGE_BYTES;
  const uint32_t full0 = smem_u32(&bars[0]), empty0 = smem_u32(&bars[STAGES]);
  // block-expert table view: s_be (small class) or block_experts in global via __ldg (first-turn class)
  const typename BeSel<BIG>::T be = BeSel<BIG>::make(s_be, block_experts);

  int nb = __ldg(n_padded) / 16;
  const int nb_wanted = nb;
  nb = nb < blocks_len ? nb : blocks_len;
  nb = nb < CBLK ? nb : CBLK;
  if (tid == 0 && nb_wanted > nb) atomicExch(err, 3);  // never truncate silently
  if constexpr (!BIG) {
    for (int b = tid; b < nb; b += THREADS) s_be[b] = __ldg(block_experts + b);
  }
  if (tid < MAXE) {
    s_first[tid] = -1;
    s_gfirst[tid] = 0;
    s_gcount[tid] = 0;
  }
  if (tid == 0) {
    s_bad = 0;
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 8);  // one arrive per consumer warp
    }
    fence_barrier_init();
  }
  __syncthreads();

  // Run starts. The align sorts blocks by expert, so every local expert owns ONE run of blocks; a block starts
  // its expert's run when its predecessor belongs to another expert. A second run of the same expert (an align
  // that is not sorted) loses the CAS and sets error word 4: never a silent mis-chunking. Any expert id outside
  // [0, n_experts) is not local.
  for (int b = tid; b < nb; b += THREADS) {
    const int e = be[b];
    if (e >= 0 && e < n_experts && e < MAXE && (b == 0 || be[b - 1] != e)) {
      if (atomicCAS(&s_first[e], -1, b) != -1) {
        atomicExch(err, 4);
        s_bad = 1;
      }
    }
  }
  __syncthreads();

  // Chunks: a block starts a chunk when its expert is local and it is 0 mod 8 into the expert's run. Rounds of
  // THREADS blocks; each round numbers its chunk starts after the earlier rounds' (ballot + per-warp counts).
  int base = 0;
  for (int r0 = 0; r0 < nb; r0 += THREADS) {
    const int b = r0 + tid;
    int flag = 0, cn = 0;
    if (b < nb) {
      const int e = be[b];
      if (e >= 0 && e < n_experts && e < MAXE && s_first[e] >= 0 && ((b - s_first[e]) & 7) == 0) {
        flag = 1;
        cn = 1;
        while (cn < 8 && b + cn < nb && be[b + cn] == e) ++cn;
      }
    }
    const unsigned ball = __ballot_sync(0xffffffffu, flag);
    if (lane == 0) s_wsum[warp] = __popc(ball);
    __syncthreads();
    int tot = 0;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) tot += s_wsum[w];
    if (flag) {
      int off = 0;
      for (int w = 0; w < warp; ++w) off += s_wsum[w];
      const int idx = base + off + __popc(ball & ((1u << lane) - 1u));
      if (idx < CC) {
        s_cblk[idx] = b;
        s_cnb[idx] = cn;
      }
    }
    base += tot;
    __syncthreads();
  }
  if (tid == 0 && base > CC) atomicExch(err, 3);
  const int nc = base < CC ? base : CC;
  // Per-expert chunk group: first chunk and chunk count (chunks of one expert are consecutive).
  for (int c = tid; c < nc; c += THREADS) {
    const int e = be[s_cblk[c]];
    atomicAdd(&s_gcount[e], 1);
    if (c == 0 || be[s_cblk[c - 1]] != e) s_gfirst[e] = c;
  }
  __syncthreads();

  // Two runs of one expert would make decode_item index outside its chunk group (an out-of-bounds table read
  // and a wild weight / output address): do no work at all then; error word 4 already turns the path off.
  const int total = s_bad ? 0 : nc * n_tiles;
  const int KS = K / BK;
  bool dead = false;
  if (warp < 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (tid == 0) {
      const int KC = K / 64;
      int n = 0;
      for (int it = blockIdx.x; it < total; it += gridDim.x) {
        const Item m = decode_item(it, n_tiles, be, s_cblk, s_cnb, s_gfirst, s_gcount);
        const uint8_t* wsrc = wbank + (static_cast<size_t>(m.e) * n_tiles + m.j) * static_cast<size_t>(KC) * REC;
        const uint64_t map = reinterpret_cast<uint64_t>(&maps.m[m.bucket]);
        const uint32_t tx = WST + 2 * (16 << m.bucket) * 128;
        for (int kb = 0; kb < KS; ++kb, ++n) {
          const int s = n % STAGES;
          mbar_wait(empty0 + 8 * s, ((n / STAGES) & 1) ^ 1, err, dead);
          const uint32_t st = smem_u32(smem + s * STAGE_BYTES);
          const uint32_t fb = full0 + 8 * s;
          mbar_expect_tx(fb, tx);
          bulk_g2s(st, wsrc + static_cast<size_t>(kb) * WST, WST, fb);
          tma_2d(st + WST, map, kb * BK, m.slot0, fb);
          tma_2d(st + WST + SLAB_HALF, map, kb * BK + 64, m.slot0, fb);
        }
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const int ctid = tid - 128, cw = ctid >> 5;
    int n = 0;
    for (int it = blockIdx.x; it < total; it += gridDim.x) {
      const Item m = decode_item(it, n_tiles, be, s_cblk, s_cnb, s_gfirst, s_gcount);
#define WG_CONSUME(NB_)                                                                                      \
  consume_item<NB_, G>(m, smem, ep, full0, empty0, n, KS, cw, lane, ctid, gs, sorted_ids, num_routes, out,  \
                       out_ld, err, dead)
      switch (m.bucket) {
        case 0: WG_CONSUME(16); break;
        case 1: WG_CONSUME(32); break;
        case 2: WG_CONSUME(64); break;
        default: WG_CONSUME(128); break;
      }
#undef WG_CONSUME
    }
  }
#else
  // Not the sm_90a image: no wgmma here. Flag it (error word 2) and do nothing else.
  if (threadIdx.x == 0) atomicExch(err, 2);
#endif
}

// xs[slot] = x[sorted_ids[slot] / 8] for the slots of local blocks (sentinel slots: zeros);
// slots past n_padded and slots of SGLang's remote (-1) blocks are skipped.
__global__ void __launch_bounds__(256)
rega_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ sorted_ids,
                 const int* __restrict__ block_experts, const int* __restrict__ n_padded, int num_routes, int H,
                 int n_experts, __nv_bfloat16* __restrict__ xs) {
  const int slot = blockIdx.x;
  if (slot >= __ldg(n_padded)) return;
  const int be = __ldg(block_experts + slot / 16);
  if (be < 0 || be >= n_experts) return;
  const int sid = __ldg(sorted_ids + slot);
  uint4* dst = reinterpret_cast<uint4*>(xs + static_cast<size_t>(slot) * H);
  const int nv = H / 8;
  if (sid >= 0 && sid < num_routes) {
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(sid >> 3) * H);
    for (int v = threadIdx.x; v < nv; v += blockDim.x) dst[v] = __ldg(src + v);
  } else {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int v = threadIdx.x; v < nv; v += blockDim.x) dst[v] = z;
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// Nothing lives at file scope: every rega_gemm call resolves the driver entry point and encodes its own tensor
// maps into a stack-local Maps from the tensors passed in THAT call (a __grid_constant__ kernel argument).
EncodeTiledFn get_encode() {
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
  return reinterpret_cast<EncodeTiledFn>(fn);
}

template <bool G, bool BIG>
void info_one(std::vector<int64_t>& r) {
  const void* fn = reinterpret_cast<const void*>(rega_kernel<G, BIG>);
  const cudaError_t e1 = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<G>::BYTES);
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, fn);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, THREADS, Smem<G>::BYTES);
  if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) (void)cudaGetLastError();
  r.push_back((G ? 1 : 0) + (BIG ? 2 : 0));
  r.push_back(e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess ? 1 : 0);
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(static_cast<int64_t>(a.sharedSizeBytes));
  r.push_back(Smem<G>::BYTES);
  r.push_back(occ);
  r.push_back(static_cast<int64_t>(e1));
  r.push_back(a.binaryVersion);
  r.push_back(a.ptxVersion);
}

}  // namespace

// Per instantiation (w13 then w2 of the small class, then w13 then w2 of the first-turn class): gated + 2 x
// first-turn class, attribute calls ok, numRegs, localSizeBytes, static
// smem, dynamic smem, occupancy (CTAs per SM), cudaFuncSetAttribute's error code, binaryVersion,
// ptxVersion (both report 90 for sm_90 and sm_90a: logged only; error word 2 is the real proof).
std::vector<int64_t> rega_init() {
  std::vector<int64_t> r;
  info_one<true, false>(r);
  info_one<false, false>(r);
  info_one<true, true>(r);
  info_one<false, true>(r);
  return r;
}

void rega_gather(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
               int64_t num_routes, int64_t n_experts, torch::Tensor xs) {
  TORCH_CHECK(x.is_contiguous() && xs.is_contiguous() && sorted_ids.is_contiguous() && block_experts.is_contiguous() &&
              x.scalar_type() == at::kBFloat16 && xs.scalar_type() == at::kBFloat16 &&
              sorted_ids.scalar_type() == at::kInt && block_experts.scalar_type() == at::kInt &&
              n_padded.scalar_type() == at::kInt, "rega_gather: layout");
  TORCH_CHECK(x.size(1) == xs.size(1) && x.size(1) % 8 == 0 && sorted_ids.numel() <= xs.size(0) &&
              block_experts.numel() * 16 >= sorted_ids.numel(), "rega_gather: sizes");
  const int slots = static_cast<int>(sorted_ids.numel());
  if (slots == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  rega_gather_kernel<<<slots, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<int*>(sorted_ids.data_ptr()), reinterpret_cast<int*>(block_experts.data_ptr()),
      reinterpret_cast<int*>(n_padded.data_ptr()), static_cast<int>(num_routes), static_cast<int>(x.size(1)),
      static_cast<int>(n_experts), reinterpret_cast<__nv_bfloat16*>(xs.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rega_gemm(int64_t which, torch::Tensor bsrc, torch::Tensor wbank,
             torch::Tensor gs, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
             int64_t num_routes, int64_t n_tiles, int64_t K, torch::Tensor out, int64_t grid, torch::Tensor err) {
  TORCH_CHECK(which == 0 || which == 1, "rega_gemm: which");
  TORCH_CHECK(bsrc.is_cuda() && bsrc.is_contiguous() && bsrc.dim() == 2 && bsrc.scalar_type() == at::kBFloat16 &&
              bsrc.size(0) > 0 && bsrc.size(1) > 0 && bsrc.size(1) == K,
              "rega_gemm: B-source scratch layout (cuda, contiguous, 2-D bf16, width K)");
  TORCH_CHECK(wbank.is_contiguous() && gs.is_contiguous() && sorted_ids.is_contiguous() &&
              block_experts.is_contiguous() && out.is_contiguous() && wbank.scalar_type() == at::kByte &&
              gs.scalar_type() == at::kFloat && sorted_ids.scalar_type() == at::kInt &&
              block_experts.scalar_type() == at::kInt && n_padded.scalar_type() == at::kInt &&
              out.scalar_type() == at::kBFloat16 && err.scalar_type() == at::kInt && err.numel() >= 3,
              "rega_gemm: layout");
  TORCH_CHECK(K % BK == 0 && block_experts.numel() <= MAXBLK_FT && gs.numel() <= MAXE && sorted_ids.numel() <= bsrc.size(0),
              "rega_gemm: sizes");
  TORCH_CHECK(wbank.numel() == gs.numel() * n_tiles * (K / 64) * REC, "rega_gemm: weight bank size");
  TORCH_CHECK(out.size(1) == (which == 0 ? n_tiles * 64 : n_tiles * 128), "rega_gemm: output width");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks_len = static_cast<int>(block_experts.numel());
  // Capacity class from the table length (a shape: T * 8 + 32 * 15 slots / 16): the stage-1 instantiation
  // for every table it can hold (all T <= 4096), the first-turn one above.
  const bool big = blocks_len > MAXBLK;
  // s5: encode this launch's four tensor maps (N buckets 16/32/64/128) over the bsrc passed in THIS call,
  // into a stack-local Maps passed by value as the __grid_constant__ kernel parameter (frozen into the
  // graph node at capture). Host-only driver work: no sync, no allocation, nothing persisted.
  EncodeTiledFn enc = get_encode();
  TORCH_CHECK(enc != nullptr, "rega_gemm: cuTensorMapEncodeTiled entry point missing");
  Maps mp;
  std::memset(&mp, 0, sizeof(Maps));
  {
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(bsrc.size(1)), static_cast<cuuint64_t>(bsrc.size(0))};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(bsrc.size(1)) * 2};
    const cuuint32_t es[2] = {1, 1};
    void* base = bsrc.data_ptr();
    for (int b = 0; b < 4; ++b) {
      const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(16 << b)};
      const CUresult rc = enc(&mp.m[b], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box, es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TORCH_CHECK(rc == CUDA_SUCCESS, "rega_gemm: tensor map ", b, " rc ", static_cast<int>(rc));
    }
  }
#define WG_LAUNCH(G_, B_)                                                                                     \
  rega_kernel<G_, B_><<<static_cast<unsigned>(grid), THREADS, Smem<G_>::BYTES, stream>>>(                          \
      mp, reinterpret_cast<uint8_t*>(wbank.data_ptr()), reinterpret_cast<float*>(gs.data_ptr()), reinterpret_cast<int*>(sorted_ids.data_ptr()),             \
      reinterpret_cast<int*>(block_experts.data_ptr()), reinterpret_cast<int*>(n_padded.data_ptr()), blocks_len, static_cast<int>(num_routes),      \
      static_cast<int>(n_tiles), static_cast<int>(K), static_cast<int>(gs.numel()),                           \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(out.size(1)), reinterpret_cast<int*>(err.data_ptr()))
  if (which == 0) {
    if (big) WG_LAUNCH(true, true); else WG_LAUNCH(true, false);
  } else {
    if (big) WG_LAUNCH(false, true); else WG_LAUNCH(false, false);
  }
#undef WG_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _rega_wgmma_src():
    """The four wgmma m64nNk16 RS wrappers (A = 4 bf16x2 registers, B = smem descriptor, fp32 D,
    scale-d = 1, scales a/b = 1, B not transposed: K-major), generated per N."""
    out = []
    for nb in REGA_BUCKETS:
        na = nb // 2
        d = ", ".join(f"%{i}" for i in range(na))
        lines = [
            "template <>",
            f"struct Wg<{nb}> {{",
            f"  static __device__ __forceinline__ void mma(float (&d)[{na}], const uint32_t (&a)[4], uint64_t desc) {{",
            "#if defined(__CUDA_ARCH_FEAT_SM90_ALL)",
            '    asm volatile("{\\n\\t.reg .pred p;\\n\\t"',
            f'                 "setp.ne.b32 p, %{na + 5}, 0;\\n\\t"',
            f'                 "wgmma.mma_async.sync.aligned.m64n{nb}k16.f32.bf16.bf16 "',
            f'                 "{{{d}}}, "',
            f'                 "{{%{na}, %{na + 1}, %{na + 2}, %{na + 3}}}, %{na + 4}, p, 1, 1, 0;\\n\\t}}\\n"',
            "                 : " + ", ".join(f'"+f"(d[{i}])' for i in range(na)),
            '                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));',
            "#endif",
            "  }",
            "};",
        ]
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def _rega_src():
    return _REGA_HEAD + _rega_wgmma_src() + _REGA_BODY


# Build-failure fallback (pre-flight audit): if muse_prefill_rega_v2 does not compile or load, prepare() builds
# master's exact stage-1 extension (muse_prefill_rega_v1) instead, so the low / high classes master ships survive
# and 'ft' stays OFF. Its source is derived from the v2 source by reverting this stage's edits, and is used
# only when its sha256 equals master's (else no fallback: everything OFF, as before).
REGA_EXT_NAME_V1 = "muse_prefill_rega_v1"
REGA_V1_SHA256 = "244ca8b9220a6adc820f3ef7eec3717c5ce1f2be7caf238158008807fb534a0f"
_REGA_V1_EDITS = (  # (part, text in v2, master's text)
    ("head",
     ('// first-turn capacity class (T <= 16384): worst case 131072 local routes + 32 x 15 padding = 8222 blocks of 16\n'
      '// and <= 8222 / 8 + 32 = 1059 chunks. Its block-expert table is read from global (__ldg), only the chunk\n'
      '// tables live in shared memory (2 x 1088 x 4 B).\n'
      'constexpr int MAXBLK_FT = 8224;\n'
      'constexpr int MAXC_FT = 1088;\n'),
     ""),
    ("head",
     ('// Capacity class: BIG = false is the T <= 4096 build (tables and smem exactly as stage 1); BIG = true serves\n'
      '// 4096 < T <= 16384 with the block-expert table in global memory.\n'
      'template <bool BIG>\n'
      'struct Cap {\n'
      '  static constexpr int BLK = BIG ? MAXBLK_FT : MAXBLK;\n'
      '  static constexpr int C = BIG ? MAXC_FT : MAXC;\n'
      '  static constexpr int SBE = BIG ? 1 : MAXBLK;   // shared block-expert table entries\n'
      '};\n'
      '// static tables + barriers / per-expert arrays stay under the 16384 B static bound in both classes\n'
      'static_assert((Cap<false>::SBE + 2 * Cap<false>::C) * 4 + 1024 <= 16384, "static smem, small class");\n'
      'static_assert((Cap<true>::SBE + 2 * Cap<true>::C) * 4 + 1024 <= 16384, "static smem, first-turn class");\n'
      '// Block-expert table views: the small class indexes its shared copy (a plain pointer, as stage 1); the large\n'
      '// class reads global memory through the read-only path.\n'
      'struct GTab {\n'
      '  const int* p;\n'
      '  __device__ __forceinline__ int operator[](int i) const { return __ldg(p + i); }\n'
      '};\n'
      'template <bool BIG>\n'
      'struct BeSel;\n'
      'template <>\n'
      'struct BeSel<false> {\n'
      '  using T = const int*;\n'
      '  static __device__ __forceinline__ T make(const int* s, const int* g) { return s; }\n'
      '};\n'
      'template <>\n'
      'struct BeSel<true> {\n'
      '  using T = GTab;\n'
      '  static __device__ __forceinline__ T make(const int* s, const int* g) { return GTab{g}; }\n'
      '};\n'),
     ""),
    ("body",
     ('// be: the block-expert table view (shared pointer in the small class, GTab in the first-turn class).\n'
      'template <class BE>\n'
      '__device__ __forceinline__ Item decode_item(int it, int T, BE be, const int* cblk, const int* cnb,\n'),
     ('__device__ __forceinline__ Item decode_item(int it, int T, const int* be, const int* cblk, const int* cnb,\n')),
    ("body",
     ('  }\n'
      '}\n'
      '\n'
      'template <bool G, bool BIG>\n'),
     ('  }\n'
      '}\n'
      '\n'
      'template <bool G>\n')),
    ("body",
     ('  constexpr int CBLK = Cap<BIG>::BLK, CC = Cap<BIG>::C;\n'
      '  __shared__ int s_be[Cap<BIG>::SBE], s_cblk[CC], s_cnb[CC];\n'),
     ('  __shared__ int s_be[MAXBLK], s_cblk[MAXC], s_cnb[MAXC];\n')),
    ("body",
     ('  // block-expert table view: s_be (small class) or block_experts in global via __ldg (first-turn class)\n'
      '  const typename BeSel<BIG>::T be = BeSel<BIG>::make(s_be, block_experts);\n'),
     ""),
    ("body",
     ('  nb = nb < CBLK ? nb : CBLK;\n'),
     ('  nb = nb < MAXBLK ? nb : MAXBLK;\n')),
    ("body",
     ('  if constexpr (!BIG) {\n'
      '    for (int b = tid; b < nb; b += THREADS) s_be[b] = __ldg(block_experts + b);\n'
      '  }\n'),
     ('  for (int b = tid; b < nb; b += THREADS) s_be[b] = __ldg(block_experts + b);\n')),
    ("body",
     ('    const int e = be[b];\n'
      '    if (e >= 0 && e < n_experts && e < MAXE && (b == 0 || be[b - 1] != e)) {\n'),
     ('    const int e = s_be[b];\n'
      '    if (e >= 0 && e < n_experts && e < MAXE && (b == 0 || s_be[b - 1] != e)) {\n')),
    ("body",
     ('      const int e = be[b];\n'),
     ('      const int e = s_be[b];\n')),
    ("body",
     ('        while (cn < 8 && b + cn < nb && be[b + cn] == e) ++cn;\n'),
     ('        while (cn < 8 && b + cn < nb && s_be[b + cn] == e) ++cn;\n')),
    ("body",
     ('      if (idx < CC) {\n'),
     ('      if (idx < MAXC) {\n')),
    ("body",
     ('  if (tid == 0 && base > CC) atomicExch(err, 3);\n'
      '  const int nc = base < CC ? base : CC;\n'),
     ('  if (tid == 0 && base > MAXC) atomicExch(err, 3);\n'
      '  const int nc = base < MAXC ? base : MAXC;\n')),
    ("body",
     ('    const int e = be[s_cblk[c]];\n'),
     ('    const int e = s_be[s_cblk[c]];\n')),
    ("body",
     ('    if (c == 0 || be[s_cblk[c - 1]] != e) s_gfirst[e] = c;\n'),
     ('    if (c == 0 || s_be[s_cblk[c - 1]] != e) s_gfirst[e] = c;\n')),
    ("body",
     ('        const Item m = decode_item(it, n_tiles, be, s_cblk, s_cnb, s_gfirst, s_gcount);\n'),
     ('        const Item m = decode_item(it, n_tiles, s_be, s_cblk, s_cnb, s_gfirst, s_gcount);\n')),
    ("body",
     ('    for (int it = blockIdx.x; it < total; it += gridDim.x) {\n'
      '      const Item m = decode_item(it, n_tiles, be, s_cblk, s_cnb, s_gfirst, s_gcount);\n'),
     ('    for (int it = blockIdx.x; it < total; it += gridDim.x) {\n'
      '      const Item m = decode_item(it, n_tiles, s_be, s_cblk, s_cnb, s_gfirst, s_gcount);\n')),
    ("body",
     ('  return reinterpret_cast<EncodeTiledFn>(fn);\n'
      '}\n'
      '\n'
      'template <bool G, bool BIG>\n'),
     ('  return reinterpret_cast<EncodeTiledFn>(fn);\n'
      '}\n'
      '\n'
      'template <bool G>\n')),
    ("body",
     ('  const void* fn = reinterpret_cast<const void*>(rega_kernel<G, BIG>);\n'),
     ('  const void* fn = reinterpret_cast<const void*>(rega_kernel<G>);\n')),
    ("body",
     ('  r.push_back((G ? 1 : 0) + (BIG ? 2 : 0));\n'),
     ('  r.push_back(G ? 1 : 0);\n')),
    ("body",
     ('// Per instantiation (w13 then w2 of the small class, then w13 then w2 of the first-turn class): gated + 2 x\n'
      '// first-turn class, attribute calls ok, numRegs, localSizeBytes, static\n'),
     ('// Per instantiation (w13 then w2): gated, attribute calls ok, numRegs, localSizeBytes, static\n')),
    ("body",
     ('  info_one<true, false>(r);\n'
      '  info_one<false, false>(r);\n'
      '  info_one<true, true>(r);\n'
      '  info_one<false, true>(r);\n'),
     ('  info_one<true>(r);\n'
      '  info_one<false>(r);\n')),
    ("body",
     ('  TORCH_CHECK(K % BK == 0 && block_experts.numel() <= MAXBLK_FT && gs.numel() <= MAXE && sorted_ids.numel() <= bsrc.size(0),\n'),
     ('  TORCH_CHECK(K % BK == 0 && block_experts.numel() <= MAXBLK && gs.numel() <= MAXE && sorted_ids.numel() <= bsrc.size(0),\n')),
    ("body",
     ('  // Capacity class from the table length (a shape: T * 8 + 32 * 15 slots / 16): the stage-1 instantiation\n'
      '  // for every table it can hold (all T <= 4096), the first-turn one above.\n'
      '  const bool big = blocks_len > MAXBLK;\n'),
     ""),
    ("body",
     ('#define WG_LAUNCH(G_, B_)                                                                                     \\\n'
      '  rega_kernel<G_, B_><<<static_cast<unsigned>(grid), THREADS, Smem<G_>::BYTES, stream>>>(                          \\\n'),
     ('#define WG_LAUNCH(G_)                                                                                         \\\n'
      '  rega_kernel<G_><<<static_cast<unsigned>(grid), THREADS, Smem<G_>::BYTES, stream>>>(                          \\\n')),
    ("body",
     ('  if (which == 0) {\n'
      '    if (big) WG_LAUNCH(true, true); else WG_LAUNCH(true, false);\n'
      '  } else {\n'
      '    if (big) WG_LAUNCH(false, true); else WG_LAUNCH(false, false);\n'
      '  }\n'),
     ('  if (which == 0) WG_LAUNCH(true); else WG_LAUNCH(false);\n')),
)


def _rega_src_v1():
    """Master's exact rega CUDA source (stage 1), or None when the reverted text does not hash to master's."""
    parts = {"head": _REGA_HEAD, "body": _REGA_BODY}
    for part, new, old in _REGA_V1_EDITS:
        if parts[part].count(new) != 1:
            return None
        parts[part] = parts[part].replace(new, old)
    src = parts["head"] + _rega_wgmma_src() + parts["body"]
    return src if hashlib.sha256(src.encode()).hexdigest() == REGA_V1_SHA256 else None


def _rega_build(name=REGA_EXT_NAME, src=None):
    """Compile the module (prepare only, never during capture): (ext or None, error text). sm_90a only:
    cpp_extension's JIT path appends -gencode flags from TORCH_CUDA_ARCH_LIST (else from the device, a plain
    compute_90 pass that cannot assemble wgmma), so the list is set to 9.0a for the build and restored after;
    every wgmma / setmaxnreg body is also behind __CUDA_ARCH_FEAT_SM90_ALL with an error-word-2 #else."""
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    ext, err_text = None, ""
    try:
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load_inline
        print(f"{REGA_TAG} torch {torch.__version__} cuda {torch.version.cuda}; device capability "
              f"{torch.cuda.get_device_capability()}", flush=True)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            arch_flags = cpp_extension._get_cuda_arch_flags()
        except Exception as e:  # this torch does not know the 'a' suffix: default list + the guards
            print(f"{REGA_TAG} TORCH_CUDA_ARCH_LIST=9.0a rejected ({e!r}); default list, error word 2 decides",
                  flush=True)
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
            arch_flags = ["<unknown>"]
        print(f"{REGA_TAG} build arch flags: TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r} -> "
              f"{arch_flags}; extra_cuda_cflags {REGA_CUDA_CFLAGS}", flush=True)
        ext = load_inline(name=name, cpp_sources=_REGA_CPP, cuda_sources=src if src is not None else _rega_src(),
                          functions=["rega_init", "rega_gather", "rega_gemm"],
                          extra_cuda_cflags=list(REGA_CUDA_CFLAGS), verbose=False)
    except Exception as e:
        err_text = str(e)
        ext = None
        print(f"{REGA_TAG} load_inline failed after {time.perf_counter() - t0:.1f}s: {err_text[-400:]!r}",
              flush=True)
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    if ext is not None:
        print(f"{REGA_TAG} load_inline compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext, err_text


def _rega_source_audit():
    """Pre-build audit: no templated Tensor accessor (libtorch exports data_ptr<int64_t> but not
    <long long>: the ssdq stage-1 failure), same scan as _ssdq_source_audit over the rega sources."""
    bad = []
    for name, src in (("cpp", _REGA_CPP), ("cuda", _rega_src())):
        for n, line in enumerate(src.splitlines(), 1):
            if "data_ptr<" in line or "data_ptr <" in line or _SSDQ_TEMPLATED_ACCESSOR.search(line):
                bad.append(f"{name} line {n}: {line.strip()[:120]}")
    return bad


def _rega_nibbles(packed):
    """uint8 [..., n] -> uint8 [..., 2n]: element 2i is the low nibble, as nvfp4 packs."""
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)


def _rega_pack_nibbles(nib):
    """Inverse of _rega_nibbles."""
    nib = nib.reshape(*nib.shape[:-1], -1, 2)
    return nib[..., 0] | (nib[..., 1] << 4)


def _rega_code_dims(rows, k, gated):
    """Source view and permutation of one expert's [rows, k] nibbles. Source dims: rh (gate/up
    half for w13, row half of the warp's 16 for w2), j (128-row tile), w (warp), g (lane / 4), kc
    (64-k block), s (k16 step), kh (k 8-15 half), t (lane % 4), kl (k parity). Target: j kc w g t
    s kl kh rh, then kh and rh are reversed, so the nibble in the lane's word s is 4 kl + 2 (1 -
    kh) + (1 - rh): register pair i = rh + 2 kh at nibbles 3 - i (kl = 0) and 7 - i (kl = 1)."""
    kc = k // 64
    if gated:  # row = rh * I + 64 j + 8 w + g
        return (2, rows // 128, 8, 8, kc, 4, 2, 4, 2), (1, 4, 2, 3, 7, 5, 8, 6, 0)
    # row = 128 j + 16 w + 8 rh + g
    return (rows // 128, 8, 2, 8, kc, 4, 2, 4, 2), (0, 4, 1, 3, 7, 5, 8, 6, 2)


def _rega_scale_dims(rows, kg, gated):
    """Source view and permutation of one expert's [rows, k / 16] scale bytes. Target: j kc w g s
    rh, i.e. per lane group g 8 bytes, byte 2 s + rh = row g + 8 rh at k16 step s."""
    kc = kg // 4
    if gated:
        return (2, rows // 128, 8, 8, kc, 4), (1, 4, 2, 3, 5, 0)
    return (rows // 128, 8, 2, 8, kc, 4), (0, 4, 1, 3, 5, 2)


def _rega_inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _rega_tile_expert(packed, sf, gated):
    """One expert's uint8 [rows, k / 2] codes and float8_e4m3fn [rows, k / 16] scales -> the
    kernel's records (flat uint8, [tiles][k / 64][4096 codes + 512 scales]), asserting that
    un-permuting both parts gives the source bytes back exactly."""
    rows, k = packed.shape[0], packed.shape[1] * 2
    tiles, kc = rows // 128, k // 64
    shape, perm = _rega_code_dims(rows, k, gated)
    codes = _rega_pack_nibbles(_rega_nibbles(packed).view(shape).permute(perm).flip(-1, -2).reshape(-1, 8)).reshape(
        tiles, kc, 4096)
    src_sf = sf.contiguous().view(torch.uint8)
    s_shape, s_perm = _rega_scale_dims(rows, k // 16, gated)
    scales = src_sf.view(s_shape).permute(s_perm).reshape(tiles, kc, 512)
    rec = torch.cat((codes, scales), dim=-1).contiguous()
    back_c = _rega_nibbles(rec[..., :4096].reshape(-1)).view([shape[p] for p in perm]).flip(-1, -2).permute(
        _rega_inverse(perm)).reshape(rows, k)
    assert torch.equal(_rega_pack_nibbles(back_c), packed), "code retiling is not lossless"
    back_s = rec[..., 4096:].reshape(-1).view([s_shape[p] for p in s_perm]).permute(_rega_inverse(s_perm)).reshape(
        rows, k // 16)
    assert torch.equal(back_s, src_sf), "scale retiling is not lossless"
    return rec.reshape(-1)


def _rega_bank(weights, start, count):
    """The wgmma path's lossless retiling of experts [start, start + count): codes and E4M3 scales
    stay NVFP4 bytes, global scales stay fp32 with the dequant bias 2^7 folded in (exact)."""
    w13, w2 = weights["w13"], weights["w2"]
    rec13 = w13[0].numel() + weights["w13_sf"][0].numel()
    rec2 = w2[0].numel() + weights["w2_sf"][0].numel()
    q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)
    q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)
    for i in range(count):
        e = start + i
        q13[i] = _rega_tile_expert(w13[e].contiguous(), weights["w13_sf"][e], True)
        q2[i] = _rega_tile_expert(w2[e].contiguous(), weights["w2_sf"][e], False)
    sl = slice(start, start + count)
    return {
        "q13": q13.reshape(-1), "gs13": (weights["w13_gs"][sl].float() * 128.0).contiguous(),
        "q2": q2.reshape(-1), "gs2": (weights["w2_gs"][sl].float() * 128.0).contiguous(),
    }


def _rega_audit_cpp(src):
    """Static scan of the CUDA/C++ source for state that outlives a call: mutable file / namespace
    scope variables, static locals, and device-side globals (__device__ / __constant__ / __managed__
    variables). Returns a list of findings (empty = none). __shared__ is per launch and not state."""
    findings = []
    txt = re.sub(r"(?m)^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", "", src)  # preprocessor lines and their continuations
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//[^\n]*", "", txt)
    txt = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', txt)
    for m in re.finditer(r"(?m)^\s*(__device__|__constant__|__managed__)\s+(?!__forceinline__|void\b|inline\b|static\b)"
                         r"[^;(){}]*?\w+\s*(\[[^\]]*\])?\s*(=[^;]*)?;", txt):
        findings.append("device global: " + " ".join(m.group(0).split())[:100])
    for m in re.finditer(r"\bstatic\b(?!\s+(?:constexpr|const|__device__|__forceinline__|inline)\b)[^;{]*", txt):
        findings.append("static: " + " ".join(m.group(0).split())[:100])
    stack, cur = [], ""
    for ch in txt:
        if ch == "{":
            if re.search(r"=\s*$", cur):  # brace initializer: part of the statement
                stack.append("init")
                cur += ch
            else:
                stack.append("ns" if re.search(r"\bnamespace\b[^{};]*$", cur) else "blk")
                cur = ""
        elif ch == "}":
            was = stack.pop() if stack else "blk"
            cur = cur + ch if was == "init" else ""
        elif ch == ";":
            stmt = cur.strip()
            cur = ""
            if "blk" in stack or not stmt:
                continue
            if (re.match(r"(typedef|using|static_assert|extern|template)\b", stmt) or "constexpr" in stmt
                    or "(" in stmt):
                continue
            findings.append("file-scope variable: " + " ".join(stmt.split())[:100])
        else:
            cur += ch
    return findings


def _rega_audit_globals():
    """prepare-time audit: (a) the load_inline source for mutable file-scope / static / device globals,
    (b) this Python module for tensors, graphs or containers of tensors at module scope. Returns the
    findings; any finding turns the path off."""
    findings = _rega_audit_cpp(_REGA_CPP + "\n" + _rega_src())
    n_cpp = len(findings)
    for name, val in list(globals().items()):
        if isinstance(val, (torch.Tensor, torch.cuda.CUDAGraph)):
            findings.append(f"python global {name}: {type(val).__name__}")
        elif isinstance(val, (dict, list, tuple, set)):
            items = val.values() if isinstance(val, dict) else val
            if any(isinstance(v, (torch.Tensor, torch.cuda.CUDAGraph)) for v in items):
                findings.append(f"python global {name}: container holding a tensor / graph")
    print(f"{REGA_TAG} global/static audit: CUDA/C++ source {n_cpp} mutable file-scope / static / device "
          f"globals, Python module {len(findings) - n_cpp} tensor-holding globals; findings "
          f"{findings if findings else 'none'}; per-state data: rega_bank, rega_err; per call: xs, hs, tensor maps",
          flush=True)
    return findings


def _experts_rega(x, topk_weights, local_ids, state, tracer=None):
    """The experts on the register-A wgmma kernel, on ids already mapped to this GPU's experts (0-31, -1
    elsewhere). align (block 16, local routes only, master's parallel align) -> gather the routed rows into
    sorted order -> w13 + silu*up (fp32, one bf16 rounding) into hs -> w2 scattered to the route slot of y ->
    master's weighted sum. Everything a launch reads comes from `state` (bank, error word) and this call's
    tensors (the tensor maps are encoded from xs / hs in C++ on every call). tracer, only passed by
    prepare(), announces each launch before it and syncs + prints after it."""
    ext, bank = state["rega_ext"], state["rega_bank"]
    err, grid = state["rega_err"], state["rega_grid"]
    num_tokens, size_k = x.shape
    num_slots = num_tokens * TOP_K
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if tracer:
        tracer("before", "align")
    _, (sorted_ids, block_experts, n_padded) = _align_parallel(local_ids, 0, LOCAL_EXPERTS, REGA_BSM)
    if tracer:
        tracer("after", "align")
    rows = block_experts.shape[0] * REGA_BSM
    xs = torch.empty((rows, size_k), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("before", "rega_gather")
    ext.rega_gather(x.contiguous(), sorted_ids, block_experts, n_padded, num_slots, LOCAL_EXPERTS, xs)
    if tracer:
        tracer("after", "rega_gather")
    hs = torch.empty((rows, D_EXPERT), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("before", "rega_kernel<w13+silu>")
    ext.rega_gemm(0, xs, bank["q13"], bank["gs13"], sorted_ids, block_experts, n_padded, num_slots,
                  D_EXPERT // 64, size_k, hs, grid, err)
    if tracer:
        tracer("after", "rega_kernel<w13+silu>")
    # The weighted sum skips remote slots when prepare() proved it (state['zero_nonlocal_w2'] False);
    # otherwise remote rows must read as zeros.
    if state["zero_nonlocal_w2"]:
        y = torch.zeros((num_slots, size_k), device=x.device, dtype=x.dtype)
    else:
        y = torch.empty((num_slots, size_k), device=x.device, dtype=x.dtype)
    if tracer:
        tracer("before", "rega_kernel<w2>")
    ext.rega_gemm(1, hs, bank["q2"], bank["gs2"], sorted_ids, block_experts, n_padded, num_slots,
                  size_k // 128, D_EXPERT, y, grid, err)
    if tracer:
        tracer("after", "rega_kernel<w2>")
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, TOP_K, size_k), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=local_ids)


class _RegaTracer:
    """prepare()-only: a flushed print BEFORE each launch, then a device sync and a flushed print AFTER it,
    so a fault names its kernel. Never passed by kernel_fn, never used under capture."""

    def __init__(self, tag):
        self.tag = tag
        self.t0 = 0.0

    def __call__(self, when, name):
        if when == "before":
            print(f"{REGA_TAG} [{self.tag}] launching {name}", flush=True)
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            return
        torch.cuda.synchronize()
        # eager host-timed launch + kernel (us): a ranking aid, not the judged number
        print(f"{REGA_TAG} [{self.tag}] {name} done ({(time.perf_counter() - self.t0) * 1e6:.0f} us eager)",
              flush=True)


def _rega_reference_y(x, local_ids, weights, cache):
    """fp32 reference of the two expert GEMMs and the silu*up on the dequantized original-order weights
    (nvfp4.dequantize): the gate / up matmul stays fp32 end to end, silu*up is fp32 and rounded to bf16 once
    (the kernel stores it as bf16), y [T * 8, 4096] bf16, zeros for remote routes."""
    tokens = x.shape[0]
    flat = local_ids.view(-1)
    y = torch.zeros((tokens * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=x.device)
    for e in range(LOCAL_EXPERTS):
        idx = (flat == e).nonzero().flatten()
        if idx.numel() == 0:
            continue
        if e not in cache:
            cache[e] = (nvfp4.dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]),
                        nvfp4.dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]))
        w13, w2 = cache[e]
        h = x[idx // TOP_K].float() @ w13.t()
        g, u = h[:, :D_EXPERT], h[:, D_EXPERT:]
        a = (g / (1.0 + torch.exp(-g)) * u).to(torch.bfloat16).float()
        y[idx] = (a @ w2.t()).to(torch.bfloat16)
    return y


def _rega_reference_out(x, tw, lids, state, weights, cache):
    tokens = x.shape[0]
    ref_y = _rega_reference_y(x, lids, weights, cache)
    return fused_post_expert_norm_weighted_sum(
        ref_y.view(tokens, TOP_K, D_EXPERT), tw, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=lids)


def _rega_marks(err):
    """(error word, w13 launch marker, w2 launch marker) read back from an error tensor (prepare only)."""
    word, m13, m2 = [int(v) for v in err.tolist()]
    return word, m13, m2


def _rega_launched(m13, m2):
    return m13 == REGA_MAGIC and m2 == REGA_MAGIC


def _rega_copy_tree(obj):
    """The judge's copy, modelled: a recursive walk that clones every tensor and shares everything else
    (the ext module, ints, strings), rebuilding dicts / lists / tuples."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _rega_copy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rega_copy_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_rega_copy_tree(v) for v in obj)
    return obj


REGA_REPLAY_TOKENS = (753, 3012, 4096)
REGA_REPLAY_KINDS = ("random", "skewed", "one")
REGA_REPLAY_SEQ = ("A", "B", "C", "A", "B")


def _rega_case(tokens, kind, gen, device):
    """_ssdq_case, plus kind 'local': every token's top 8 drawn from the 32 local experts only (all 8 routes
    local), which forces the most blocks and chunks the align can produce at this T."""
    if kind != "local":
        return _ssdq_case(tokens, kind, gen, device)
    x = torch.randn((tokens, D_EXPERT), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    local_ids = torch.rand((tokens, LOCAL_EXPERTS), generator=gen, device=device).topk(
        TOP_K, dim=1).indices.to(torch.int32).contiguous()
    tw = torch.rand((tokens, TOP_K), generator=gen, device=device, dtype=torch.float32) + 0.1
    tw = (tw / tw.sum(dim=1, keepdim=True)).contiguous()
    return x, tw, local_ids


def _rega_incumbent(tokens):
    """(name, fn) of master's expert block at this T without rega: Marlin up to MARLIN_MAX_TOKENS, else Triton.
    The self-check bound is calibrated on its error against the fp32 reference."""
    if tokens <= MARLIN_MAX_TOKENS:
        return "marlin", _experts_marlin
    return "triton", _experts_triton


def _rega_tiles(lids):
    """prepare() only: the rega align's table sizes for these ids: (blocks table length, used blocks, chunks,
    max chunks of one expert). Work items: chunks x 64 (w13) and chunks x 32 (w2)."""
    _, (_, block_experts, n_padded) = _align_parallel(lids, 0, LOCAL_EXPERTS, REGA_BSM)
    used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())
    be = block_experts[:used].cpu().long()
    counts = torch.bincount(be[(be >= 0) & (be < LOCAL_EXPERTS)], minlength=LOCAL_EXPERTS)
    chunks = (counts + 7) // 8
    return block_experts.numel(), used, int(chunks.sum()), int(chunks.max())


def _rega_replay_check(state, weights, cache, device, gen, plan=None):
    """prepare() only. The judge prepares TWO states (setup copies: 2), copies tensors, and rotates them across
    repetitions with refilled inputs. Three states over the pieces _experts_rega reads: A = the prepared
    state, B = copy.deepcopy (the ext module shared through the memo), C = the recursive tensor-cloning walk
    after a 2 MB pad allocation. Per T, one CUDA graph of _experts_rega per state over that state's own static
    inputs; per routing, replays A, B, C, A, B, each after REFILLING that state's inputs in place with a fresh
    case, each compared with the fp32 reference (bound: max(2 x Marlin's own error, 8e-3), worst row 5x) and
    each required to leave BOTH launch markers and error word 0 in that state's own error tensor (this is the
    LAUNCHED = rega assertion). plan: ((T, kinds), ...), default REGA_REPLAY_TOKENS x REGA_REPLAY_KINDS; above
    MARLIN_MAX_TOKENS the bound is calibrated on Triton (master's block there) instead of Marlin. Graphs are freed
    after every T and the peak allocation printed. Returns (None or why, summary line)."""
    if plan is None:
        plan = tuple((t, REGA_REPLAY_KINDS) for t in REGA_REPLAY_TOKENS)
    mini = {k: state[k] for k in REGA_STATE_KEYS}
    states = {"A": mini}
    pads = []
    for label in ("B", "C"):
        pads.append(torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device=device))
        if label == "B":
            st = copy.deepcopy(mini, {id(mini["rega_ext"]): mini["rega_ext"]})
        else:
            st = _rega_copy_tree(mini)
        torch.cuda.synchronize()
        states[label] = st
        same = st["rega_bank"]["q13"].data_ptr() == mini["rega_bank"]["q13"].data_ptr() or \
            st["rega_err"].data_ptr() == mini["rega_err"].data_ptr()
        print(f"{REGA_TAG} copy {label}: bank 0x{st['rega_bank']['q13'].data_ptr():x} vs A "
              f"0x{mini['rega_bank']['q13'].data_ptr():x}, err 0x{st['rega_err'].data_ptr():x} vs A "
              f"0x{mini['rega_err'].data_ptr():x} (shared: {same})", flush=True)
        if same:
            return f"copy {label} shares state A's tensors", ""
    n = n_launched = 0
    worst_new = worst_mar = 0.0
    for t, kinds in plan:
        inc_name, inc_fn = _rega_incumbent(t)
        graphs, bufs, holders = {}, {}, {}
        try:
            for name, st in states.items():
                b = _rega_case(t, "random", gen, device)
                holder = {}

                def fn(st=st, b=b, holder=holder):
                    holder["out"] = _experts_rega(b[0], b[1], b[2], st)

                graphs[name] = _ssdq_graph_time(fn, device)
                bufs[name], holders[name] = b, holder
            for kind in kinds:
                for name in REGA_REPLAY_SEQ:
                    st = states[name]
                    x, tw, lids = _rega_case(t, kind, gen, device)
                    for dst, src in zip(bufs[name], (x, tw, lids)):
                        dst.copy_(src)
                    st["rega_err"].zero_()
                    graphs[name].replay()
                    torch.cuda.synchronize()
                    got = holders[name]["out"].clone()
                    word, m13, m2 = _rega_marks(st["rega_err"])
                    ref = _rega_reference_out(x, tw, lids, state, weights, cache)
                    mar = inc_fn(x, tw, lids, state)
                    torch.cuda.synchronize()
                    e_new, w_new = _ssdq_metrics(got, ref)
                    e_mar, _ = _ssdq_metrics(mar, ref)
                    bound = max(2.0 * e_mar, 8e-3)
                    launched = _rega_launched(m13, m2)
                    n += 1
                    n_launched += int(launched)
                    worst_new, worst_mar = max(worst_new, e_new), max(worst_mar, e_mar)
                    ok = word == 0 and launched and e_new <= bound and w_new <= 5.0 * bound
                    print(f"{REGA_TAG} replay T={t} {kind} state {name}: rega rel L2 {e_new:.3e} worst row "
                          f"{w_new:.3e}; {inc_name} {e_mar:.3e}; bound {bound:.3e}; error word {word}; launch markers "
                          f"w13 {'yes' if m13 == REGA_MAGIC else 'NO'} w2 {'yes' if m2 == REGA_MAGIC else 'NO'}"
                          f"{'' if ok else '  <-- MISS'}", flush=True)
                    if not ok:
                        return (f"T={t} {kind} state {name}: rel L2 {e_new:.3e} worst row {w_new:.3e} bound "
                                f"{bound:.3e} word {word} ({REGA_ERR_TEXT.get(word, '?')}) markers {m13:#x}/{m2:#x}",
                                "")
        finally:
            del graphs, bufs, holders
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if t > MARLIN_MAX_TOKENS:
                print(f"{REGA_TAG} replay T={t}: graphs freed; max_memory_allocated "
                      f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)
    del states, pads
    torch.cuda.empty_cache()
    return None, (f"{n} refilled graph replays over states A, B (deepcopy), C (tensor clone) at "
                  f"{[(t, k) for t, k in plan]}; LAUNCHED = rega on {n_launched}/{n} (both launch "
                  f"markers); worst rel L2 {worst_new:.2e} (incumbent {worst_mar:.2e})")


def _rega_gate(state, device, tokens):
    """Interleaved CUDA-graph replays with a >= 256 MB cold-L2 flush of the WHOLE expert block (align, GEMMs,
    silu, weighted sum) at one scored T on a random (soft) routing: rega against every alternative, each in
    the same loop, REGA_ALTERNATIONS alternations, the order rotated and flipped each time, an empty-graph
    baseline subtracted from every median. Alternatives: whatever _experts dispatches with rega off (the
    incumbent: ssdq when ON at T > SSDQ_MIN_TOKENS, else Marlin), ssdq itself when it exists and is ON, and
    Marlin's block. Above MARLIN_MAX_TOKENS (class 'ft') the only alternative is the dispatch: master's Triton
    block (ssdq and Marlin do not serve those T). Returns {name: median us}."""
    gen = torch.Generator(device=device).manual_seed(20261025)
    x, tw, lids = _ssdq_case(tokens, "random", gen, device)
    fns = {"rega": lambda: _experts_rega(x, tw, lids, state),
           "dispatch": lambda: _experts(x, tw, lids, state)}  # rega flags are False while gating
    if tokens <= MARLIN_MAX_TOKENS:
        if tokens > SSDQ_MIN_TOKENS and state.get("ssdq_ok", False):
            fns["ssdq"] = lambda: _experts_ssdq(x, tw, lids, state)
        fns["marlin"] = lambda: _experts_marlin(x, tw, lids, state)
    graphs = {name: _ssdq_graph_time(fn, device) for name, fn in fns.items()}
    tiny = torch.zeros(16, dtype=torch.float32, device=device)
    graphs["empty"] = _ssdq_graph_time(lambda: tiny.zero_(), device)
    flush = torch.empty(256 * 2 ** 20 // 4, dtype=torch.float32, device=device)  # > the 50 MB L2
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def once(graph):
        flush.zero_()
        ev[0].record()
        graph.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    names = list(graphs)
    samples = {name: [] for name in names}
    for name in names:
        once(graphs[name])  # warm-up replay
    for i in range(REGA_ALTERNATIONS):
        order = names[i % len(names):] + names[:i % len(names)]
        if i % 2:
            order = order[::-1]
        for name in order:
            samples[name].append(once(graphs[name]))
    med = {name: sorted(v)[len(v) // 2] for name, v in samples.items()}
    base = med.pop("empty")
    return {name: max(v - base, 0.0) for name, v in med.items()}


def _rega_ft_setup(state, weights, device, t_start, phase, big_ok):
    """prepare() only, after the low / high gates (the kernel has passed canary, self-check and the copy / replay
    check at T <= 4096). Class 'ft' (MARLIN_MAX_TOKENS < T <= REGA_FT_MAX): the capacity arithmetic, the
    first-turn instantiations' attributes, zero_nonlocal_w2 False, the first-turn align check, an eager
    self-check at T 8192 / 16384 (random, skewed, one-expert, local-saturated: the most blocks and chunks) against
    the fp32 reference with the bound calibrated on Triton's own error, the A/B/C copy / replay check at 8192 and
    16384, then the cold-L2 gate at T = REGA_FT_GATE_TOKENS (soft routing) against master's Triton block (the
    rega-off dispatch). Sets only state['rega_ok']['ft']; returns (gate line or None, OFF reason or None). Any
    failure leaves the low / high classes and master's Triton path as they were."""
    state["rega_ok"]["ft"] = False
    zero = bool(state["zero_nonlocal_w2"])
    worst_slots = REGA_FT_MAX * TOP_K + LOCAL_EXPERTS * (REGA_BSM - 1)
    worst_blocks = -(-worst_slots // REGA_BSM)
    worst_chunks = worst_blocks // 8 + LOCAL_EXPERTS
    print(f"{REGA_TAG} ft capacity: worst case at T={REGA_FT_MAX} (every route local): {worst_blocks} blocks "
          f"(table {REGA_MAXBLK_FT}), <= {worst_chunks} chunks (table {REGA_MAXC_FT}); tables of <= {REGA_MAXBLK} "
          f"blocks launch the stage-1 instantiation; zero_nonlocal_w2 {zero}", flush=True)
    if worst_blocks > REGA_MAXBLK_FT or worst_chunks > REGA_MAXC_FT:
        return None, "first-turn tables smaller than the worst case"
    if not big_ok:
        return None, "first-turn instantiation attribute / occupancy / register-pool check failed"
    if zero:
        return None, "zero_nonlocal_w2 is True (rega would memset a 1.07 GB y every call)"
    if not state.get("align_par_ft_ok", False):
        return None, "parallel align missed at first-turn sizes"
    used = time.perf_counter() - t_start
    if used > REGA_FT_START_DEADLINE_S:
        return None, f"prepare already used {used:.0f}s > {REGA_FT_START_DEADLINE_S:.0f}s ft start deadline"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gen = torch.Generator(device=device).manual_seed(20261026)
    cache = {}
    try:
        for tokens, kind in REGA_FT_SELF_CHECKS:
            x, tw, lids = _rega_case(tokens, kind, gen, device)
            tab_len, n_used, chunks, max_chunks = _rega_tiles(lids)
            inc_name, inc_fn = _rega_incumbent(tokens)
            state["rega_err"].zero_()
            ref = _rega_reference_out(x, tw, lids, state, weights, cache)
            new = _experts_rega(x, tw, lids, state, tracer=_RegaTracer(f"ft T={tokens} {kind}"))
            torch.cuda.synchronize()
            inc = inc_fn(x, tw, lids, state)
            torch.cuda.synchronize()
            word, m13, m2 = _rega_marks(state["rega_err"])
            e_new, w_new = _ssdq_metrics(new, ref)
            e_inc, w_inc = _ssdq_metrics(inc, ref)
            bound = max(2.0 * e_inc, 8e-3)
            ok = word == 0 and _rega_launched(m13, m2) and e_new <= bound and w_new <= 5.0 * bound
            print(f"{REGA_TAG} ft self-check T={tokens} {kind}: {int((lids >= 0).sum())} local routes; block table "
                  f"{tab_len} ({'first-turn' if tab_len > REGA_MAXBLK else 'stage-1'} class), {n_used} used, "
                  f"{chunks} chunks (max {max_chunks} per expert; items w13 {chunks * (D_EXPERT // 64)}, w2 "
                  f"{chunks * (D_EXPERT // 128)}); rega rel L2 {e_new:.3e} worst row {w_new:.3e}; {inc_name} "
                  f"{e_inc:.3e} / {w_inc:.3e}; bound {bound:.3e}; error word {word} "
                  f"({REGA_ERR_TEXT.get(word, '?')}); launch markers w13 {m13:#x} w2 {m2:#x}"
                  f"{'' if ok else '  <-- MISS'}", flush=True)
            del x, tw, lids, ref, new, inc
            if not ok:
                return None, (f"self-check failed at T={tokens} {kind} (error word {word} "
                              f"{REGA_ERR_TEXT.get(word, '?')}, rel L2 {e_new:.3e}, worst row {w_new:.3e}, bound "
                              f"{bound:.3e}, markers {m13:#x}/{m2:#x})")
        torch.cuda.empty_cache()
        print(f"{REGA_TAG} ft self-check done; max_memory_allocated {torch.cuda.max_memory_allocated(device) / 2**30:.2f}"
              f" GiB", flush=True)
        phase("ft-self-check")
        why, line = _rega_replay_check(state, weights, cache, device, gen, plan=REGA_FT_REPLAY_PLAN)
        if why:
            return None, f"copy / replay check failed: {why}"
        print(f"{REGA_TAG} LAUNCHED = rega (ft): {line}", flush=True)
        phase("ft-copy-check")
    finally:
        cache.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    used = time.perf_counter() - t_start
    if used > REGA_BUDGET_S:
        return None, f"gate skipped: prepare already used {used:.0f}s > {REGA_BUDGET_S:.0f}s budget"
    meds = _rega_gate(state, device, REGA_FT_GATE_TOKENS)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    ratio = meds["rega"] / max(meds["dispatch"], 1e-9)
    state["rega_ok"]["ft"] = bool(ratio <= REGA_GATE_RATIO)
    gate_line = (f"{REGA_TAG} GATE ft T={REGA_FT_GATE_TOKENS} soft cold-L2 graph medians over {REGA_ALTERNATIONS} "
                 f"alternations (empty-graph baseline subtracted): rega {meds['rega']:.1f} us; dispatch (master's "
                 f"Triton block) {meds['dispatch']:.1f} us; ratio {ratio:.3f} (ON at <= {REGA_GATE_RATIO}); "
                 f"zero_nonlocal_w2 {zero}; max_memory_allocated "
                 f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB: "
                 f"{'ON' if state['rega_ok']['ft'] else 'OFF'}")
    print(gate_line, flush=True)
    phase("ft-gate")
    return gate_line, None


def _rega_setup(state, weights, device, t_start):
    """prepare() only, after master's ssdq setup. Build, retile, canary, self-check against an fp32
    reference (bound calibrated on Marlin's own error), the two-state copy / replay check with the launch
    markers (LAUNCHED asserted), then one cold-L2 gate per T class. Sets state['rega_ok'] = {'low': bool,
    'high': bool}; every failure prints one OFF line and leaves master's dispatch as it was. The gate
    medians and the PATH CHOSEN lines are printed LAST (the archived log keeps the tail). Then the first-turn
    class 'ft' (_rega_ft_setup), whose failures turn off 'ft' only."""
    state["rega_ok"] = {"low": False, "high": False, "ft": False}
    ft_line, ft_off = None, None
    v1_fallback = False
    phases = []
    t_phase = [time.perf_counter()]
    t_pre = t_phase[0] - t_start
    off = []
    gate_lines = []
    gate_meds = {}

    def phase(name):
        now = time.perf_counter()
        phases.append((name, now - t_phase[0]))
        t_phase[0] = now
        print(f"{REGA_TAG} phase {name}: {phases[-1][1]:.1f}s; prepare total {now - t_start:.1f}s; budget remaining "
              f"{REGA_BUDGET_S - (now - t_start):.1f}s of {REGA_BUDGET_S:.0f}s", flush=True)

    print(f"{REGA_TAG} prepare used {t_pre:.1f}s before this setup (budget {REGA_BUDGET_S:.0f}s)", flush=True)
    try:
        if not state.get("align_par_ok", False):
            off.append("master's parallel align is off (the path needs it)")
        elif time.perf_counter() - t_start > REGA_BUDGET_S:
            off.append(f"prepare already used {time.perf_counter() - t_start:.0f}s > {REGA_BUDGET_S:.0f}s budget")
        else:
            cap = torch.cuda.get_device_capability(device)
            if tuple(cap) != (9, 0):
                off.append(f"device capability {cap} is not (9, 0)")
            shapes = {k: tuple(weights[k].shape) for k in ("w13", "w13_sf", "w2", "w2_sf")}
            want = {"w13": (LOCAL_EXPERTS, 2 * D_EXPERT, D_EXPERT // 2), "w13_sf": (LOCAL_EXPERTS, 2 * D_EXPERT, D_EXPERT // 16),
                    "w2": (LOCAL_EXPERTS, D_EXPERT, D_EXPERT // 2), "w2_sf": (LOCAL_EXPERTS, D_EXPERT, D_EXPERT // 16)}
            if shapes != want:
                off.append(f"unsupported weight shapes {shapes}")
            for gated in (True, False):
                need = REGA_SMEM[gated] + REGA_STATIC_SMEM_BOUND
                if need > REGA_SMEM_LIMIT:
                    off.append(f"smem {need} B > {REGA_SMEM_LIMIT} B")
        if off:
            return
        offenders = _rega_source_audit()
        if offenders:
            off.append(f"source audit found templated accessors: {offenders[:4]}")
            return
        print(f"{REGA_TAG} source audit ok: no data_ptr<..> or templated size/item/data/to accessor; smem w13 "
              f"{REGA_SMEM[True]} B, w2 {REGA_SMEM[False]} B dynamic (+ <= {REGA_STATIC_SMEM_BOUND} B static) <= "
              f"{REGA_SMEM_LIMIT} B", flush=True)
        phase("audit")
        ext, _ = _rega_build()
        phase("build")
        if ext is None:
            # v2 did not compile / load: build master's exact stage-1 extension so low / high keep master's
            # behaviour; the first-turn class stays OFF (that build has no first-turn instantiation).
            src_v1 = _rega_src_v1()
            print(f"{REGA_TAG} v2 build FAILED; fallback: master's {REGA_EXT_NAME_V1} "
                  f"({'source sha256 matches master' if src_v1 else 'source does NOT match master: no fallback'})",
                  flush=True)
            if src_v1 is not None:
                ext, _ = _rega_build(REGA_EXT_NAME_V1, src_v1)
                phase("build-v1-fallback")
            if ext is None:
                off.append("build failed")
                return
            v1_fallback = True
        info = ext.rega_init()
        # per capacity class (False: stage-1 tables, True: first-turn); the v1 fallback has no first-turn class
        attrs_ok = {False: True, True: not v1_fallback}
        for i in range(0, len(info), 10):
            flag, a_ok, regs, local_bytes, static_smem, dyn_smem, occ, e1, bin_v, ptx_v = info[i:i + 10]
            gated, big = bool(flag & 1), bool(flag & 2)
            print(f"{REGA_TAG} kernel {'w13+silu' if gated else 'w2'} {'first-turn' if big else 'stage-1'} class: "
                  f"attrs ok {bool(a_ok)} (SetAttribute rc {e1}), "
                  f"regs {regs}, local bytes {local_bytes}{' (SPILL)' if local_bytes > 0 else ''}, static smem "
                  f"{static_smem} B, dynamic smem {dyn_smem} B, CTAs/SM {occ}, binaryVersion {bin_v}, ptxVersion "
                  f"{ptx_v}", flush=True)
            # setmaxnreg.inc blocks forever unless the CTA's register pool covers 40 x 128 + 232 x 256
            attrs_ok[big] = (attrs_ok[big] and bool(a_ok) and occ >= 1 and static_smem + dyn_smem <= REGA_SMEM_LIMIT
                             and regs * REGA_THREADS >= 40 * 128 + 232 * 256 and bin_v == 90)
        if not attrs_ok[False]:
            off.append("cudaFuncSetAttribute / cudaFuncGetAttributes / occupancy / register-pool check failed")
            return
        findings = _rega_audit_globals()
        if findings:
            off.append(f"module/static state found: {findings}")
            return
        t0 = time.perf_counter()
        state["rega_ext"] = ext
        state["rega_err"] = torch.zeros(3, dtype=torch.int32, device=device)
        state["rega_grid"] = int(torch.cuda.get_device_properties(device).multi_processor_count)
        state["rega_bank"] = _rega_bank(weights, 0, LOCAL_EXPERTS)
        torch.cuda.synchronize()
        print(f"{REGA_TAG} retiled {LOCAL_EXPERTS} experts losslessly (round trips exact) in "
              f"{time.perf_counter() - t0:.1f}s; grid {state['rega_grid']} persistent CTAs x {REGA_THREADS} threads, "
              f"BK {REGA_BK}, {REGA_STAGES} stages, N buckets {REGA_BUCKETS}", flush=True)
        phase("retile")
        gen = torch.Generator(device=device).manual_seed(20261025)

        # Canary: one launch of every kernel at a tiny T, each with its before / after print.
        x, tw, lids = _ssdq_case(300, "random", gen, device)
        out = _experts_rega(x, tw, lids, state, tracer=_RegaTracer("canary"))
        torch.cuda.synchronize()
        word, m13, m2 = _rega_marks(state["rega_err"])
        print(f"{REGA_TAG} canary: error word {word} ({REGA_ERR_TEXT.get(word, '?')}), launch markers w13 "
              f"{m13:#x} w2 {m2:#x}", flush=True)
        if word != 0 or not _rega_launched(m13, m2) or not bool(torch.isfinite(out.float()).all()):
            off.append(f"canary failed: error word {word}, markers {m13:#x}/{m2:#x}")
            return
        phase("canary")

        # Self-check: every N bucket (T=300 ~9 routes/expert: 16; 753 ~24: 32; 1506 ~47: 64; 3012 ~94: 128),
        # skewed (4 hot experts: multi-chunk groups) and one-expert routings, eager, vs the fp32 reference;
        # bound = max(2 x Marlin's own error, 8e-3), worst row 5x.
        cache = {}
        cases = ((300, "random"), (753, "random"), (1506, "random"), (3012, "random"), (1100, "one"),
                 (4096, "skewed"), (4096, "random"))
        for tokens, kind in cases:
            x, tw, lids = _ssdq_case(tokens, kind, gen, device)
            state["rega_err"].zero_()
            ref = _rega_reference_out(x, tw, lids, state, weights, cache)
            new = _experts_rega(x, tw, lids, state, tracer=_RegaTracer(f"T={tokens} {kind}"))
            torch.cuda.synchronize()
            mar = _experts_marlin(x, tw, lids, state)
            torch.cuda.synchronize()
            word, m13, m2 = _rega_marks(state["rega_err"])
            e_new, w_new = _ssdq_metrics(new, ref)
            e_mar, w_mar = _ssdq_metrics(mar, ref)
            bound = max(2.0 * e_mar, 8e-3)
            print(f"{REGA_TAG} self-check T={tokens} {kind}: rega rel L2 {e_new:.3e} worst row {w_new:.3e}; marlin "
                  f"{e_mar:.3e} / {w_mar:.3e}; bound {bound:.3e}; error word {word}; {int((lids >= 0).sum())} "
                  f"local routes; launch markers w13 {m13:#x} w2 {m2:#x}", flush=True)
            if word != 0 or not _rega_launched(m13, m2) or not (e_new <= bound and w_new <= 5.0 * bound):
                off.append(f"self-check failed at T={tokens} {kind} (error word {word} "
                           f"{REGA_ERR_TEXT.get(word, '?')}, rel L2 {e_new:.3e}, worst row {w_new:.3e}, bound "
                           f"{bound:.3e}, markers {m13:#x}/{m2:#x})")
                del cache
                return
        phase("self-check")
        if time.perf_counter() - t_start > REGA_COPY_DEADLINE_S:
            off.append(f"copy check skipped: prepare already used {time.perf_counter() - t_start:.0f}s")
            del cache
            return
        why, line = _rega_replay_check(state, weights, cache, device, gen)
        del cache
        torch.cuda.empty_cache()
        if why:
            off.append(f"copy / replay check failed: {why}")
            return
        print(f"{REGA_TAG} LAUNCHED = rega: {line}", flush=True)
        phase("copy-check")

        # Gates, one per T class. rega flags are False here, so `dispatch` is exactly what _experts runs today.
        for cls in ("low", "high"):
            tokens = REGA_GATE_TOKENS[cls]
            meds = _rega_gate(state, device, tokens)
            gate_meds[cls] = meds
            alts = {k: v for k, v in meds.items() if k != "rega"}
            best_name = min(alts, key=alts.get)
            ratio = meds["rega"] / max(alts[best_name], 1e-9)
            state["rega_ok"][cls] = bool(ratio <= REGA_GATE_RATIO)
            gate_lines.append(
                f"{REGA_TAG} GATE {cls} T={tokens} cold-L2 graph medians over {REGA_ALTERNATIONS} alternations "
                f"(empty-graph baseline subtracted): rega {meds['rega']:.1f} us; "
                + "; ".join(f"{k} {v:.1f} us" for k, v in alts.items())
                + f"; fastest alternative {best_name}; ratio rega/{best_name} {ratio:.3f} (ON at <= "
                  f"{REGA_GATE_RATIO}): {'ON' if state['rega_ok'][cls] else 'OFF'}")
            print(gate_lines[-1], flush=True)
        phase("gate")
        # First-turn class: its own checks and gate; a failure there turns off 'ft' only.
        try:
            if v1_fallback:
                ft_line, ft_off = None, f"v2 build failed; running master's {REGA_EXT_NAME_V1} (no first-turn class)"
            else:
                ft_line, ft_off = _rega_ft_setup(state, weights, device, t_start, phase, attrs_ok[True])
        except Exception as e:
            state["rega_ok"]["ft"] = False
            ft_line, ft_off = None, f"ft setup raised {e!r}"[:300]
        if ft_off:
            state["rega_ok"]["ft"] = False
    except Exception as e:  # never raises into prepare(); a device fault would have poisoned the context
        state["rega_ok"] = {"low": False, "high": False, "ft": False}
        off.append(f"setup raised {e!r}"[:300])
    finally:
        total = time.perf_counter() - t_start
        if off:
            state["rega_ok"] = {"low": False, "high": False, "ft": False}
            print(f"{REGA_TAG} OFF: {'; '.join(off)}"[:900], flush=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Last lines of prepare(): the archived log keeps the tail.
        print(f"{REGA_TAG} prepare per-phase seconds: before-rega {t_pre:.1f}, "
              + ", ".join(f"{n} {d:.1f}" for n, d in phases)
              + f"; total {total:.1f}s; budget {REGA_BUDGET_S:.0f}s", flush=True)
        for line in gate_lines:
            print(line, flush=True)
        if ft_line:
            print(ft_line, flush=True)
        if ft_off and not off:
            print(f"{REGA_TAG} OFF ft: {ft_off} (T in ({MARLIN_MAX_TOKENS}, {REGA_FT_MAX}] stays on triton)"[:900],
                  flush=True)
        ok = state["rega_ok"]
        ssdq_on = bool(state.get("ssdq_ok"))
        print(f"{REGA_TAG} PATH CHOSEN T in ({REGA_MIN_TOKENS}, {REGA_LOW_MAX}]: "
              f"{'rega' if ok['low'] else 'marlin (master unchanged)'}; T in ({REGA_LOW_MAX}, {MARLIN_MAX_TOKENS}]: "
              f"{'rega' if ok['high'] else ('ssdq' if ssdq_on else 'marlin')} (dispatch order: rega if ON, else ssdq if "
              f"ON, else marlin; ssdq {'ON' if ssdq_on else 'OFF'}); T <= {REGA_MIN_TOKENS}: marlin; T > "
              f"{MARLIN_MAX_TOKENS}: next line", flush=True)
        print(f"{REGA_TAG} PATH CHOSEN T in ({MARLIN_MAX_TOKENS}, {REGA_FT_MAX}]: "
              f"{'rega' if ok.get('ft', False) else 'triton'}; T > {REGA_FT_MAX}: triton", flush=True)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    # The tokens <= 8 router and unpooler paths are not ported (prefill q_len is 200-16384).
    assert tokens > SMALL_BATCH_MAX_TOKENS, f"muse_prefill_layer port needs tokens > 8, got {tokens}"
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    ffn_input, stream, _ = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        both = torch.mm(ffn_input, state["router_hilo_t"], out_dtype=torch.float32)
        logits = both[:, :N_EXPERTS] + both[:, N_EXPERTS:]
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: SGLang's expert_map on that EP rank, in one kernel.
    local_ids = _local_expert_ids(topk_ids, data["local_start"])
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
