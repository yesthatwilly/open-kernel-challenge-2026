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
  experts                          moe_align_block_size, two grouped W4A16 GEMMs (hypotheses
                                   experts-triton-w4a16-inkernel-fp4 and -swapab-reglhs: a Triton
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

import nvfp4
import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
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


def _check_expert_gemms(bank, weights):
    """Untimed. Runs every kernel variant kernel_fn can launch (so each is compiled before capture)
    on a synthetic case built from the weights alone, and asserts its in-kernel dequant of the
    reordered bank equals nvfp4.dequantize of the ORIGINAL packing. Row r of A is one-hot at
    k0 + r % 128, so output row r is column k0 + r % 128 of the dequantized expert, through the
    real pointer arithmetic, K loop, nibble order, scale indexing and transposed epilogue. One -1
    block and one block past the padded count must leave their rows untouched (or zero, for the -1
    block when zero-storing)."""
    device = bank["w13"].device
    e = LOCAL_EXPERTS - 1  # the largest expert offsets
    for name, top_k, zero_nonlocal in (("w13", TOP_K, False), ("w2", 1, ZERO_NONLOCAL_W2)):
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


def prepare(weights):
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
        "bank": _expert_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    # Compiles every expert-GEMM variant and checks its dequant against nvfp4.dequantize.
    _check_expert_gemms(state["bank"], weights)
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, local_ids, state):
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
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, cfg["BLOCK_T"], N_EXPERTS)
    cache2 = torch.empty((num_slots, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_slots * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_slots * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_slots * size_k].view(-1, size_k)
    _nvfp4_moe_gemm(x.contiguous(), bank["w13"], bank["w13_sf"], bank["w13_gs"], cache1, sorted_ids, block_experts,
                    n_padded, TOP_K, num_slots, cfg, False)
    silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = _nvfp4_moe_gemm(cache2, bank["w2"], bank["w2_sf"], bank["w2_gs"], cache3, sorted_ids, block_experts,
                        n_padded, 1, num_slots, cfg, ZERO_NONLOCAL_W2).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


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
        logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
        logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
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
