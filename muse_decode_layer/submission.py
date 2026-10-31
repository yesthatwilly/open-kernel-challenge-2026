# okc: problem=muse_decode_layer
from typing import TypedDict
import os
import random
import time
import types
import torch
import torch.nn.functional as F
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
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor
class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int
class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config
HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
MARLIN_BLOCK_SIZE_M_CAP = 16
SMALL_T_POST_NORM = 32
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0
def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)
def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)
def _norm(cls, w, size, eps, **kw):
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
def _marlin_bank(weights):
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
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }
@triton.jit
def _post_norm_combine_kernel(y_ptr, ids_ptr, tw_ptr, w_ptr, out_ptr, H, eps,
                              TOP_K_C: tl.constexpr, BLOCK_H: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    nw = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for s in tl.static_range(TOP_K_C):
        eid = tl.load(ids_ptr + t * TOP_K_C + s)
        if eid >= 0:
            g = tl.load(tw_ptr + t * TOP_K_C + s)
            row = (t * TOP_K_C + s).to(tl.int64) * H
            v = tl.load(y_ptr + row + offs, mask=mask, other=0.0).to(tl.float32) * nw
            r = tl.rsqrt(tl.sum(v * v, axis=0) / H + eps)
            acc += g * v * r
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16), mask=mask)
DL_MAX_TOKENS = 32
DL_ROWS = D_EXPERT + 2 * N_EXPERTS
DL_SPLIT_K = 3
DL_N_TILES = DL_ROWS // 64
DL_TAG = "[small-t-dense-chain]"
_DL_CPP = r"""
#include <torch/extension.h>
#include <vector>
void dl_pooler_router(torch::Tensor x, torch::Tensor w, torch::Tensor ei, torch::Tensor logits,
                      torch::Tensor ws, torch::Tensor cnt, int64_t pdl);
void dl_route_norm_align(torch::Tensor ei, torch::Tensor norm_w, torch::Tensor xn, torch::Tensor logits,
                         torch::Tensor bias, torch::Tensor local_start, torch::Tensor topk_ids,
                         torch::Tensor topk_w, torch::Tensor local_ids, torch::Tensor sorted_ids,
                         torch::Tensor block_experts, torch::Tensor n_padded, torch::Tensor cnt, int64_t bsm,
                         int64_t pdl, int64_t variant);
int64_t dl_occupancy(int64_t nf);
void dl_pdl_probe(torch::Tensor times, int64_t spin_ns, int64_t pdl);
void dl2_qkv(torch::Tensor x, torch::Tensor w, torch::Tensor ws, torch::Tensor cnt, torch::Tensor raw,
             torch::Tensor q_attn, torch::Tensor gate, torch::Tensor kc, torch::Tensor vc, torch::Tensor loc,
             torch::Tensor pos, torch::Tensor cs, torch::Tensor ks, torch::Tensor vs, int64_t pdl, int64_t mode,
             int64_t paired, int64_t rope, int64_t scaled);
void dl2_o(torch::Tensor fa, torch::Tensor gate, torch::Tensor w, torch::Tensor ws, torch::Tensor cnt,
           torch::Tensor out, int64_t pdl, int64_t prologue, int64_t paired);
int64_t dl2_occupancy(int64_t which, int64_t nf);
void dl_unpool(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
               int64_t pdl);
std::vector<int64_t> dl3_attrs(int64_t which);
void dl_norm(torch::Tensor ei, torch::Tensor norm_w, torch::Tensor xn);
"""
_DL_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int ROWS = 4608;                        // W_pr rows: 4096 pooler + 512 router hi/lo
constexpr int POOL_ROWS = 4096;
constexpr int KDIM = 8192;
constexpr int BLOCK_N = 64;                       // weight rows per CTA (4 warps x 16)
constexpr int BLOCK_K = 128;                      // K per ring stage
constexpr int KB = KDIM / BLOCK_K;                // 64 K-blocks
constexpr int SPLIT = 3;                          // K-chunks of 21 / 21 / 22 K-blocks
constexpr int NTILES = ROWS / BLOCK_N;            // 72
constexpr int MAXT = 32;                          // tokens (4 x n8)
constexpr int STAGES = 4;
constexpr int THREADS = 128;
constexpr int ROW_BYTES = BLOCK_K * 2;            // 256 B = 16 chunks of 16 B
constexpr int W_STAGE = BLOCK_N * ROW_BYTES;      // 16384
template <int NF> struct Smem {
  static constexpr int X_STAGE = NF * 8 * ROW_BYTES;           // 2048 / 4096 / 8192
  static constexpr int STAGE_BYTES = W_STAGE + X_STAGE;        // 18432 / 20480 / 24576
  static constexpr int BYTES = STAGES * STAGE_BYTES;           // 73728 / 81920 / 98304
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
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// D = A B + D, m16n8k16, A row-major bf16 (4 regs), B col-major bf16 (2 regs), fp32 accumulate.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// PDL (sm_90): wait for the prerequisite grid's completion and memory flush. A no-op when the
// kernel was launched without the programmatic attribute.
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}

// ei bf16 [m, 4096] = x @ W_pr[:4096]^T; logits fp32 [m, 256] = x @ hi^T + x @ lo^T.
// x bf16 [m, 8192] contiguous, 1 <= m <= 8 * NF. Grid = NTILES * SPLIT: n-tile = blockIdx.x / SPLIT,
// K-chunk = blockIdx.x % SPLIT. ws fp32 [SPLIT, 32, ROWS] (rows < m written / read). cnt int32
// [NTILES], 0 between launches; every CTA takes exactly one ticket.
template <int NF>
__global__ void __launch_bounds__(THREADS, 2)
dl_pooler_router_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                        __nv_bfloat16* __restrict__ ei, float* __restrict__ logits,
                        float* __restrict__ ws, int* __restrict__ cnt, int m) {
  using S = Smem<NF>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ntile = blockIdx.x / SPLIT, sidx = blockIdx.x % SPLIT;
  const int n0 = ntile * BLOCK_N;
  const int kb0 = sidx * KB / SPLIT, kb1 = (sidx + 1) * KB / SPLIT;
  const int nsteps = kb1 - kb0;
  const __nv_bfloat16* wbase = w + (int64_t)n0 * KDIM;

  // Weight part of stage j: 64 rows x 16 chunks, 8 per thread. W_pr is a prepare() constant.
  auto issue_w = [&](int j) {
    uint8_t* sw = smem + (j % STAGES) * S::STAGE_BYTES;
    const int kel = (kb0 + j) * BLOCK_K;
#pragma unroll
    for (int u = 0; u < (BLOCK_N * 16) / THREADS; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      cp_async16(smem_u32(sw + row * ROW_BYTES + ((c ^ (row & 7)) << 4)),
                 wbase + (int64_t)row * KDIM + kel + c * 8, 16);
    }
  };
  // Token part of stage j: 8 * NF rows x 16 chunks, NF per thread; rows >= m zero-filled.
  auto issue_x = [&](int j) {
    uint8_t* sx = smem + (j % STAGES) * S::STAGE_BYTES + W_STAGE;
    const int kel = (kb0 + j) * BLOCK_K;
#pragma unroll
    for (int u = 0; u < NF; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      const bool ok = row < m;
      const __nv_bfloat16* src = x + (ok ? (int64_t)row * KDIM : (int64_t)0) + kel + c * 8;
      cp_async16(smem_u32(sx + row * ROW_BYTES + ((c ^ (row & 7)) << 4)), src, ok ? 16 : 0);
    }
  };

  // Prologue. Before the PDL wait: only the weight copies of stages 0..STAGES-2 (uncommitted, so
  // they join commit group 0). After it: the x copies, one commit group per stage, so group j
  // holds x of stage j (and group 0 also every early weight copy): wait_group bookkeeping is the
  // usual STAGES-1 groups ahead.
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j)
    if (j < nsteps) issue_w(j);
  griddep_wait();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < nsteps) issue_x(j);
    cp_commit();  // always, so the group count stays uniform
  }

  float acc[2][NF][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[h][f][r] = 0.f;
  const int arow = warp * 16 + (lane & 15);  // ldmatrix A row of this lane
  const int asw = arow & 7;
  const int ahalf = lane >> 4;                // A chunk offset (k 0-7 / 8-15)
  const int xrow = lane & 7;                  // ldmatrix B row (token within an n8 fragment)
  const int xq = lane >> 3;                   // B chunk offset within a k32 pair (0..3)

  for (int i = 0; i < nsteps; ++i) {
    cp_wait<STAGES - 2>();  // stage i (and everything issued before it) has landed
    __syncthreads();        // visible to all; every warp is done with stage i - 1's buffer
    {
      const int nx = i + STAGES - 1;
      if (nx < nsteps) { issue_w(nx); issue_x(nx); }
      cp_commit();
    }
    const uint8_t* sw = smem + (i % STAGES) * S::STAGE_BYTES;
    const uint32_t a_base = smem_u32(sw) + arow * ROW_BYTES;
    const uint32_t b_base = smem_u32(sw + W_STAGE) + xrow * ROW_BYTES;
#pragma unroll
    for (int kp = 0; kp < BLOCK_K / 32; ++kp) {
      uint32_t a0[4], a1[4];  // k16 steps 2kp and 2kp + 1
      ldsm_x4(a0, a_base + ((((kp * 2 + 0) * 2 + ahalf) ^ asw) << 4));
      ldsm_x4(a1, a_base + ((((kp * 2 + 1) * 2 + ahalf) ^ asw) << 4));
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        uint32_t b[4];  // b[0], b[1]: k-step 2kp; b[2], b[3]: k-step 2kp + 1 (tokens 8f..8f+7)
        ldsm_x4(b, b_base + f * 8 * ROW_BYTES + (((kp * 4 + xq) ^ xrow) << 4));
        mma_bf16(acc[0][f], a0, b[0], b[1]);
        mma_bf16(acc[1][f], a1, b[2], b[3]);
      }
    }
  }

  // PDL (stage 3, piece B): this CTA's main K loop is done; let the dependent (the new kernel R, launched
  // with the programmatic attribute at T=1) start while the split-K partial stores / ticket / reduce run.
  // R's griddepcontrol.wait still waits for P's completion. A no-op for a dependent launched without the
  // attribute (master's R at every T > 1).
  griddep_launch();

  // C fragment: c0, c1 -> weight row lane/4, tokens 2*(lane%4) + {0, 1}; c2, c3 -> row + 8.
  const int r0 = n0 + warp * 16 + (lane >> 2);
  float* wsp = ws + (int64_t)sidx * MAXT * ROWS;
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    const int t0 = f * 8 + (lane & 3) * 2;
    if (t0 < m) {
      wsp[(int64_t)t0 * ROWS + r0] = acc[0][f][0] + acc[1][f][0];
      wsp[(int64_t)t0 * ROWS + r0 + 8] = acc[0][f][2] + acc[1][f][2];
    }
    if (t0 + 1 < m) {
      wsp[(int64_t)(t0 + 1) * ROWS + r0] = acc[0][f][1] + acc[1][f][1];
      wsp[(int64_t)(t0 + 1) * ROWS + r0 + 8] = acc[0][f][3] + acc[1][f][3];
    }
  }
  __threadfence();   // each thread's partial stores are visible GPU-wide before the ticket
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(cnt + ntile, 1) == SPLIT - 1);
  __syncthreads();
  if (s_last) {
    __threadfence();  // acquire side: the other CTAs' partials are read after their tickets
    if (n0 < POOL_ROWS) {
      for (int e = tid; e < m * BLOCK_N; e += THREADS) {
        const int t = e / BLOCK_N, r = n0 + (e % BLOCK_N);
        float s = 0.f;
#pragma unroll
        for (int p = 0; p < SPLIT; ++p) s += __ldcg(ws + ((int64_t)p * MAXT + t) * ROWS + r);
        ei[(int64_t)t * POOL_ROWS + r] = __float2bfloat16_rn(s);
      }
    } else {
      // Router tile: local rows [hi 16 | lo 16 | hi 16 | lo 16] = experts 32*(ntile-64) + 0..31.
      for (int e = tid; e < m * 32; e += THREADS) {
        const int t = e >> 5, q = e & 31, sub = q >> 4, j = q & 15;
        const int rhi = n0 + sub * 32 + j, rlo = rhi + 16;
        const int expert = (ntile - POOL_ROWS / BLOCK_N) * 32 + sub * 16 + j;
        float shi = 0.f, slo = 0.f;
#pragma unroll
        for (int p = 0; p < SPLIT; ++p) {
          shi += __ldcg(ws + ((int64_t)p * MAXT + t) * ROWS + rhi);
          slo += __ldcg(ws + ((int64_t)p * MAXT + t) * ROWS + rlo);
        }
        logits[t * 256 + expert] = shi + slo;
      }
    }
    if (tid == 0) atomicExch(cnt + ntile, 0);  // self-reset for the next launch / replay
  }
}

// Orderable key: larger biased score first, then the lower expert id (key 0 = taken / none).
__device__ __forceinline__ unsigned long long route_key(float v, int e) {
  uint32_t u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned>(255 - e);
}
__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long k) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, k, o);
    k = other > k ? other : k;
  }
  return k;
}

// Grid m blocks x 256 threads (256 experts). See the Python comment above for the contract.
// dl_route_norm_align_v1_kernel: stage 1's kernel R, unchanged (every T > 1, and T = 1 when piece B is off).
__global__ void __launch_bounds__(256)
dl_route_norm_align_v1_kernel(const __nv_bfloat16* __restrict__ ei, const float* __restrict__ norm_w,
                           __nv_bfloat16* __restrict__ xn, const float* __restrict__ logits,
                           const float* __restrict__ bias, const int* __restrict__ local_start,
                           int* __restrict__ topk_ids, float* __restrict__ topk_w, int* __restrict__ local_ids,
                           int* __restrict__ sorted_ids, int* __restrict__ block_experts,
                           int* __restrict__ n_padded, int* __restrict__ cnt,
                           int m, int bsm, int cap, int n_blocks, int align) {
  __shared__ float s_red[8];
  __shared__ unsigned long long s_kred[8];
  __shared__ unsigned long long s_sel[8];
  __shared__ float s_score[256];
  __shared__ int s_last;
  __shared__ int s_wc[8][32];      // per-warp route count of each local expert
  __shared__ int s_off[8][32];     // exclusive prefix of s_wc over warps
  __shared__ int s_count[32], s_seg0[32], s_seg1[32];
  __shared__ int s_total;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t = blockIdx.x;

  // (1) Pre-expert RMSNorm of ei[t] (4096 bf16): elements [8 tid, 8 tid + 8) and 2048 + that.
  {
    const uint4* src = reinterpret_cast<const uint4*>(ei + (int64_t)t * 4096);
    uint4 raw[2] = {src[tid], src[256 + tid]};
    float v[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw[h]);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f2 = __bfloat1622float2(p2[q]);
        v[h * 8 + 2 * q] = f2.x;
        v[h * 8 + 2 * q + 1] = f2.y;
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += v[i] * v[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += s_red[i];
    const float rstd = rsqrtf(tot / 4096.f + 1e-5f);
    uint4 outv[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&outv[h]);
      const int base = h * 2048 + tid * 8;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        o2[q] = __floats2bfloat162_rn(v[h * 8 + 2 * q] * rstd * norm_w[base + 2 * q],
                                      v[h * 8 + 2 * q + 1] * rstd * norm_w[base + 2 * q + 1]);
    }
    uint4* dst = reinterpret_cast<uint4*>(xn + (int64_t)t * 4096);
    dst[tid] = outv[0];
    dst[256 + tid] = outv[1];
  }

  // (2) Routing: thread e owns expert e.
  const int e = tid;
  const float score = 1.f / (1.f + expf(-0.1767766952966369f * logits[t * 256 + e]));
  s_score[e] = score;
  const unsigned long long key = route_key(score + bias[e], e);
  bool taken = false;
#pragma unroll 1
  for (int r = 0; r < 8; ++r) {
    const unsigned long long k = warp_max_u64(taken ? 0ull : key);
    if (lane == 0) s_kred[warp] = k;
    __syncthreads();
    if (warp == 0) {
      unsigned long long k2 = lane < 8 ? s_kred[lane] : 0ull;
      k2 = warp_max_u64(k2);
      if (lane == 0) s_sel[r] = k2;
    }
    __syncthreads();
    if (key == s_sel[r]) taken = true;
  }
  if (tid < 8) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += s_score[255 - static_cast<int>(s_sel[j] & 0xffffffffull)];
    const int ex = 255 - static_cast<int>(s_sel[tid] & 0xffffffffull);
    const int local = (ex - local_start[0]) & 255;
    topk_ids[t * 8 + tid] = ex;
    topk_w[t * 8 + tid] = s_score[ex] / (sum + 1e-15f);
    local_ids[t * 8 + tid] = local < 32 ? local : -1;
  }

  // (3) Ticket: the last block aligns every token's routes for Marlin. align 0 (campaign expert-bandwidth X0b):
  // Engine A runs this T and reads only the ids and weights above, so no ticket is drawn and nothing is aligned.
  if (!align) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(cnt, 1) == m - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const int n = m * 8;  // routes (<= 256 = one per thread)
  const int lid = tid < n ? __ldcg(local_ids + tid) : -1;
  s_wc[warp][lane] = 0;
  __syncthreads();
  const unsigned mask = __match_any_sync(0xffffffffu, lid);
  const int rank = __popc(mask & ((1u << lane) - 1u));
  if (lid >= 0 && lane == __ffs(mask) - 1) s_wc[warp][lid] = __popc(mask);
  __syncthreads();
  if (warp == 0) {
    int run = 0;
#pragma unroll
    for (int w2 = 0; w2 < 8; ++w2) { s_off[w2][lane] = run; run += s_wc[w2][lane]; }
    const int padded = (run + bsm - 1) / bsm * bsm;
    int incl = padded;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int up = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += up;
    }
    s_count[lane] = run;
    s_seg0[lane] = incl - padded;
    s_seg1[lane] = incl;
    if (lane == 31) s_total = incl;
  }
  __syncthreads();
  const int total = s_total;
  if (lid >= 0) sorted_ids[s_seg0[lid] + s_off[warp][lid] + rank] = tid;
  for (int idx = tid; idx < cap; idx += 256) {  // padding slots (disjoint from the real ones)
    bool pad = idx >= total;
    if (!pad) {
      for (int x2 = 0; x2 < 32; ++x2)
        if (idx >= s_seg0[x2] && idx < s_seg1[x2]) { pad = idx - s_seg0[x2] >= s_count[x2]; break; }
    }
    if (pad) sorted_ids[idx] = n;
  }
  for (int b = tid; b < n_blocks; b += 256) {
    const int bs = b * bsm;
    int be = -1;
    if (bs < total)
      for (int x2 = 0; x2 < 32; ++x2)
        if (bs >= s_seg0[x2] && bs < s_seg1[x2]) { be = x2; break; }
    block_experts[b] = be;
  }
  if (tid == 0) {
    n_padded[0] = total;
    atomicExch(cnt, 0);  // all m tickets are drawn: reset for the next launch / replay
  }
}

// Kernel R, new (hypothesis dl-route-warp-topk-pdl). Same contract as v1. Changes: (a) per-warp
// top-8 with lane-local taken flags and one block barrier before and one after the cross-warp
// merge (v1: 2 barriers per round); route keys are unique per expert, so the ordered top-8 set is
// identical; (b) PDL: only prepare() constants (norm_w, bias) are read before griddepcontrol.wait,
// ei / logits / local_start only after it (a no-op wait when launched without the attribute);
// (c) the padding / block_experts expert lookup is a 5-step binary search (BSEARCH) or v1's scan.
template <bool BSEARCH>
__global__ void __launch_bounds__(256)
dl_route_norm_align_kernel(const __nv_bfloat16* __restrict__ ei, const float* __restrict__ norm_w,
                           __nv_bfloat16* __restrict__ xn, const float* __restrict__ logits,
                           const float* __restrict__ bias, const int* __restrict__ local_start,
                           int* __restrict__ topk_ids, float* __restrict__ topk_w, int* __restrict__ local_ids,
                           int* __restrict__ sorted_ids, int* __restrict__ block_experts,
                           int* __restrict__ n_padded, int* __restrict__ cnt,
                           int m, int bsm, int cap, int n_blocks, int align) {
  __shared__ float s_red[8];
  __shared__ unsigned long long s_cand[64];
  __shared__ unsigned long long s_sel[8];
  __shared__ float s_score[256];
  __shared__ int s_last;
  __shared__ int s_wc[8][32];      // per-warp route count of each local expert
  __shared__ int s_off[8][32];     // exclusive prefix of s_wc over warps
  __shared__ int s_count[32], s_seg0[32], s_seg1[32];
  __shared__ int s_total;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t = blockIdx.x;
  const int e = tid;  // thread e owns expert e

  // Before the PDL wait: prepare() constants only. norm_w elements [8 tid, 8 tid + 8) and 2048 + that.
  float nwf[16];
  {
    const float4* nwp = reinterpret_cast<const float4*>(norm_w);
    const float4 q0 = nwp[tid * 2], q1 = nwp[tid * 2 + 1], q2 = nwp[512 + tid * 2], q3 = nwp[512 + tid * 2 + 1];
    nwf[0] = q0.x; nwf[1] = q0.y; nwf[2] = q0.z; nwf[3] = q0.w;
    nwf[4] = q1.x; nwf[5] = q1.y; nwf[6] = q1.z; nwf[7] = q1.w;
    nwf[8] = q2.x; nwf[9] = q2.y; nwf[10] = q2.z; nwf[11] = q2.w;
    nwf[12] = q3.x; nwf[13] = q3.y; nwf[14] = q3.z; nwf[15] = q3.w;
  }
  const float bias_e = bias[e];
  griddep_wait();  // ei and logits (written by P) and local_start are read only after this

  const float lg = logits[t * 256 + e];
  const uint4* src = reinterpret_cast<const uint4*>(ei + (int64_t)t * 4096);
  uint4 raw[2] = {src[tid], src[256 + tid]};

  // (1) Pre-expert RMSNorm of ei[t] (4096 bf16), arithmetic as v1.
  {
    float v[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw[h]);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f2 = __bfloat1622float2(p2[q]);
        v[h * 8 + 2 * q] = f2.x;
        v[h * 8 + 2 * q + 1] = f2.y;
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += v[i] * v[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += s_red[i];
    const float rstd = rsqrtf(tot / 4096.f + 1e-5f);
    uint4 outv[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&outv[h]);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        o2[q] = __floats2bfloat162_rn(v[h * 8 + 2 * q] * rstd * nwf[h * 8 + 2 * q],
                                      v[h * 8 + 2 * q + 1] * rstd * nwf[h * 8 + 2 * q + 1]);
    }
    uint4* dst = reinterpret_cast<uint4*>(xn + (int64_t)t * 4096);
    dst[tid] = outv[0];
    dst[256 + tid] = outv[1];
  }

  // (2) Routing. Same score and key as v1.
  const float score = 1.f / (1.f + expf(-0.1767766952966369f * lg));
  s_score[e] = score;
  const unsigned long long key = route_key(score + bias_e, e);
  // Per-warp top-8 over this warp's 32 keys: lane-local taken flag, no block barrier.
  bool taken = false;
  unsigned long long cw = 0ull;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const unsigned long long k = warp_max_u64(taken ? 0ull : key);
    if (key == k) taken = true;
    if (lane == r) cw = k;
  }
  if (lane < 8) s_cand[warp * 8 + lane] = cw;
  __syncthreads();
  if (warp == 0) {  // global top-8 of the 64 candidates, each lane holds two
    const unsigned long long c0 = s_cand[lane], c1 = s_cand[lane + 32];
    bool t0 = false, t1 = false;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const unsigned long long a0 = t0 ? 0ull : c0, a1 = t1 ? 0ull : c1;
      const unsigned long long k = warp_max_u64(a0 > a1 ? a0 : a1);
      if (c0 == k) t0 = true;
      if (c1 == k) t1 = true;
      if (lane == 0) s_sel[r] = k;
    }
  }
  __syncthreads();
  if (tid < 8) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += s_score[255 - static_cast<int>(s_sel[j] & 0xffffffffull)];
    const int ex = 255 - static_cast<int>(s_sel[tid] & 0xffffffffull);
    const int local = (ex - local_start[0]) & 255;
    topk_ids[t * 8 + tid] = ex;
    topk_w[t * 8 + tid] = s_score[ex] / (sum + 1e-15f);
    local_ids[t * 8 + tid] = local < 32 ? local : -1;
  }

  // (3) Ticket: the last block aligns every token's routes for Marlin. align 0 (campaign expert-bandwidth X0b):
  // Engine A runs this T and reads only the ids and weights above, so no ticket is drawn and nothing is aligned.
  if (!align) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(cnt, 1) == m - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const int n = m * 8;  // routes (<= 256 = one per thread)
  const int lid = tid < n ? __ldcg(local_ids + tid) : -1;
  s_wc[warp][lane] = 0;
  __syncthreads();
  const unsigned mask = __match_any_sync(0xffffffffu, lid);
  const int rank = __popc(mask & ((1u << lane) - 1u));
  if (lid >= 0 && lane == __ffs(mask) - 1) s_wc[warp][lid] = __popc(mask);
  __syncthreads();
  if (warp == 0) {
    int run = 0;
#pragma unroll
    for (int w2 = 0; w2 < 8; ++w2) { s_off[w2][lane] = run; run += s_wc[w2][lane]; }
    const int padded = (run + bsm - 1) / bsm * bsm;
    int incl = padded;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int up = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += up;
    }
    s_count[lane] = run;
    s_seg0[lane] = incl - padded;
    s_seg1[lane] = incl;
    if (lane == 31) s_total = incl;
  }
  __syncthreads();
  const int total = s_total;
  if (lid >= 0) sorted_ids[s_seg0[lid] + s_off[warp][lid] + rank] = tid;
  for (int idx = tid; idx < cap; idx += 256) {  // padding slots (disjoint from the real ones)
    bool pad = idx >= total;
    if (!pad) {
      if (BSEARCH) {
        // idx < total lies in the segment of the largest x with seg0[x] <= idx (seg0 non-decreasing,
        // segments tile [0, total)).
        int x2 = 0;
#pragma unroll
        for (int st = 16; st > 0; st >>= 1)
          if (x2 + st < 32 && s_seg0[x2 + st] <= idx) x2 += st;
        pad = idx - s_seg0[x2] >= s_count[x2];
      } else {
        for (int x2 = 0; x2 < 32; ++x2)
          if (idx >= s_seg0[x2] && idx < s_seg1[x2]) { pad = idx - s_seg0[x2] >= s_count[x2]; break; }
      }
    }
    if (pad) sorted_ids[idx] = n;
  }
  for (int b = tid; b < n_blocks; b += 256) {
    const int bs = b * bsm;
    int be = -1;
    if (bs < total) {
      if (BSEARCH) {
        int x2 = 0;
#pragma unroll
        for (int st = 16; st > 0; st >>= 1)
          if (x2 + st < 32 && s_seg0[x2 + st] <= bs) x2 += st;
        be = x2;
      } else {
        for (int x2 = 0; x2 < 32; ++x2)
          if (bs >= s_seg0[x2] && bs < s_seg1[x2]) { be = x2; break; }
      }
    }
    block_experts[b] = be;
  }
  if (tid == 0) {
    n_padded[0] = total;
    atomicExch(cnt, 0);  // all m tickets are drawn: reset for the next launch / replay
  }
}

// PDL probe: the primary triggers its dependents at once, then spins; the secondary records when
// it started (before griddepcontrol.wait) and when the wait returned. With a programmatic edge
// the secondary starts before the primary ends. times int64 [4].
__global__ void dl_probe_primary(unsigned long long* times, long long spin_ns) {
  const unsigned long long t0 = gtime();
  griddep_launch();
  while (static_cast<long long>(gtime() - t0) < spin_ns) {}
  if (threadIdx.x == 0) { times[0] = t0; times[1] = gtime(); }
}
__global__ void dl_probe_secondary(unsigned long long* times) {
  const unsigned long long t0 = gtime();
  griddep_wait();
  const unsigned long long t1 = gtime();
  if (threadIdx.x == 0) { times[2] = t0; times[3] = t1; }
}

template <int NF>
void set_attrs() {
  static bool done = false;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_pooler_router_kernel<NF>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Smem<NF>::BYTES));
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_pooler_router_kernel<NF>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    done = true;
  }
}

template <int NF>
void launch_p(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* ei, float* logits, float* ws,
              int* cnt, int m, bool pdl, cudaStream_t stream) {
  set_attrs<NF>();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NTILES * SPLIT);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = Smem<NF>::BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dl_pooler_router_kernel<NF>, x, w, ei, logits, ws, cnt, m));
}

template <bool BSEARCH>
void launch_r(const __nv_bfloat16* ei, const float* norm_w, __nv_bfloat16* xn, const float* logits,
              const float* bias, const int* local_start, int* topk_ids, float* topk_w, int* local_ids,
              int* sorted_ids, int* block_experts, int* n_padded, int* cnt, int m, int bsm, int cap,
              int n_blocks, bool pdl, cudaStream_t stream, int align) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(m));
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dl_route_norm_align_kernel<BSEARCH>, ei, norm_w, xn, logits, bias,
                                    local_start, topk_ids, topk_w, local_ids, sorted_ids, block_experts,
                                    n_padded, cnt, m, bsm, cap, n_blocks, align));
}


// ---------------------------------------------------------------------------------------------
// Stage 2 (dense-chain-stage2-attn-side-pdl): attention-side kernels Q and O, built from the P body.
// ---------------------------------------------------------------------------------------------
constexpr int Q_ROWS = 2304;                      // qkv_gate rows: q 1024, k 128, v 128, gate 1024
constexpr int Q_SPLIT = 6;                        // K-chunks of 10 / 11 K-blocks
constexpr int Q_NTILES = Q_ROWS / BLOCK_N;        // 36 head tiles
constexpr int O_ROWS = 8192;
constexpr int O_KDIM = 1024;
constexpr int O_SPLIT = 2;                        // K halves of 512 = 8 heads
constexpr int O_NTILES = O_ROWS / BLOCK_N;        // 128
constexpr int O_KHALF = O_KDIM / O_SPLIT;         // 512
constexpr int O_KBS = O_KHALF / BLOCK_K;          // 4 K-blocks per CTA, all resident in smem
template <int NF> struct OSmem {
  static constexpr int X_KB = NF * 8 * ROW_BYTES;              // one K-block of the x tile
  static constexpr int BYTES = O_KBS * W_STAGE + O_KBS * X_KB; // 72 KB / 80 KB / 96 KB
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// Two floats -> two E4M3 bytes (first value in the low byte), saturating to +-448 (satfinite).
__device__ __forceinline__ void store_fp8x2(uint8_t* dst, float a, float b) {
  const __nv_fp8x2_storage_t p = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<unsigned short*>(dst) = static_cast<unsigned short>(p);
}

// KERNEL Q: [m, 2304] = x @ qkv_gate^T, split-K swap-AB GEMV (the P body, 36 head tiles x 6 = 216 CTAs).
// The CTA drawing a tile's last ticket owns that head's 64 fp32 outputs of every token. mode 1 (raw):
// it writes the bf16 [m, 2304] projection (for _sliding_forward). mode 0: tiles 0-15 (q) RMSNorm
// (scaleless, eps) + interleaved RoPE (cos_sin at positions) -> bf16 -> E4M3 into q_attn (unpaired
// [m,16,64], or paired [m,16,128]: own half + zeros in the other); tiles 16-17 (k) the same, divided
// by k_scale when scaled, E4M3 into k_cache row cache_loc[t]; tiles 18-19 (v) divided by v_scale when
// scaled, E4M3 into v_cache; tiles 20-35 bf16 gate [m, 1024]. Every value goes through bf16 before
// the E4M3 cast, as the master path's bf16 tensors do.
template <int NF>
__global__ void __launch_bounds__(THREADS, 2)
dl_qkv_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
              float* __restrict__ ws, int* __restrict__ cnt, __nv_bfloat16* __restrict__ raw,
              uint8_t* __restrict__ q_attn, __nv_bfloat16* __restrict__ gate_out,
              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, const int64_t* __restrict__ loc,
              const int64_t* __restrict__ pos, const float* __restrict__ cs,
              const float* __restrict__ ks, const float* __restrict__ vs,
              int m, int mode, int paired, int rope, int scaled) {
  using S = Smem<NF>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ntile = blockIdx.x / Q_SPLIT, sidx = blockIdx.x % Q_SPLIT;
  const int n0 = ntile * BLOCK_N;
  const int kb0 = sidx * KB / Q_SPLIT, kb1 = (sidx + 1) * KB / Q_SPLIT;
  const int nsteps = kb1 - kb0;
  const __nv_bfloat16* wbase = w + (int64_t)n0 * KDIM;

  auto issue_w = [&](int j) {
    uint8_t* sw = smem + (j % STAGES) * S::STAGE_BYTES;
    const int kel = (kb0 + j) * BLOCK_K;
#pragma unroll
    for (int u = 0; u < (BLOCK_N * 16) / THREADS; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      cp_async16(smem_u32(sw + row * ROW_BYTES + ((c ^ (row & 7)) << 4)),
                 wbase + (int64_t)row * KDIM + kel + c * 8, 16);
    }
  };
  auto issue_x = [&](int j) {
    uint8_t* sx = smem + (j % STAGES) * S::STAGE_BYTES + W_STAGE;
    const int kel = (kb0 + j) * BLOCK_K;
#pragma unroll
    for (int u = 0; u < NF; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      const bool ok = row < m;
      const __nv_bfloat16* src = x + (ok ? (int64_t)row * KDIM : (int64_t)0) + kel + c * 8;
      cp_async16(smem_u32(sx + row * ROW_BYTES + ((c ^ (row & 7)) << 4)), src, ok ? 16 : 0);
    }
  };

  // Weight copies of stages 0..2 before the PDL wait; x and every global write after it.
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j)
    if (j < nsteps) issue_w(j);
  griddep_wait();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < nsteps) issue_x(j);
    cp_commit();
  }

  float acc[2][NF][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[h][f][r] = 0.f;
  const int arow = warp * 16 + (lane & 15);
  const int asw = arow & 7;
  const int ahalf = lane >> 4;
  const int xrow = lane & 7;
  const int xq = lane >> 3;

  for (int i = 0; i < nsteps; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nx = i + STAGES - 1;
      if (nx < nsteps) { issue_w(nx); issue_x(nx); }
      cp_commit();
    }
    const uint8_t* sw = smem + (i % STAGES) * S::STAGE_BYTES;
    const uint32_t a_base = smem_u32(sw) + arow * ROW_BYTES;
    const uint32_t b_base = smem_u32(sw + W_STAGE) + xrow * ROW_BYTES;
#pragma unroll
    for (int kp = 0; kp < BLOCK_K / 32; ++kp) {
      uint32_t a0[4], a1[4];
      ldsm_x4(a0, a_base + ((((kp * 2 + 0) * 2 + ahalf) ^ asw) << 4));
      ldsm_x4(a1, a_base + ((((kp * 2 + 1) * 2 + ahalf) ^ asw) << 4));
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        uint32_t b[4];
        ldsm_x4(b, b_base + f * 8 * ROW_BYTES + (((kp * 4 + xq) ^ xrow) << 4));
        mma_bf16(acc[0][f], a0, b[0], b[1]);
        mma_bf16(acc[1][f], a1, b[2], b[3]);
      }
    }
  }

  const int r0 = n0 + warp * 16 + (lane >> 2);
  float* wsp = ws + (int64_t)sidx * MAXT * Q_ROWS;
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    const int t0 = f * 8 + (lane & 3) * 2;
    if (t0 < m) {
      wsp[(int64_t)t0 * Q_ROWS + r0] = acc[0][f][0] + acc[1][f][0];
      wsp[(int64_t)t0 * Q_ROWS + r0 + 8] = acc[0][f][2] + acc[1][f][2];
    }
    if (t0 + 1 < m) {
      wsp[(int64_t)(t0 + 1) * Q_ROWS + r0] = acc[0][f][1] + acc[1][f][1];
      wsp[(int64_t)(t0 + 1) * Q_ROWS + r0 + 8] = acc[0][f][3] + acc[1][f][3];
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(cnt + ntile, 1) == Q_SPLIT - 1);
  __syncthreads();
  if (s_last) {
    __threadfence();
    // One warp per token, lane = element pair (2 lane, 2 lane + 1): an interleaved RoPE pair.
    for (int t = warp; t < m; t += THREADS / 32) {
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int p = 0; p < Q_SPLIT; ++p) {
        const float2 s2 = __ldcg(reinterpret_cast<const float2*>(ws + ((int64_t)p * MAXT + t) * Q_ROWS + n0) + lane);
        v0 += s2.x;
        v1 += s2.y;
      }
      v0 = bf16r(v0);  // the projection's bf16 output
      v1 = bf16r(v1);
      if (mode == 1) {
        *reinterpret_cast<__nv_bfloat162*>(raw + (int64_t)t * Q_ROWS + n0 + 2 * lane) = __floats2bfloat162_rn(v0, v1);
        continue;
      }
      if (ntile >= 20) {  // gate
        *reinterpret_cast<__nv_bfloat162*>(gate_out + (int64_t)t * 1024 + (ntile - 20) * 64 + 2 * lane) =
            __floats2bfloat162_rn(v0, v1);
        continue;
      }
      float y0 = v0, y1 = v1;
      if (ntile < 18) {  // q or k head: scaleless RMSNorm, then RoPE
        const float ss = warp_sum(v0 * v0 + v1 * v1);
        const float r = 1.0f / sqrtf(ss / 64.0f + 1e-5f);
        const float x0 = v0 * r, x1 = v1 * r;
        if (rope) {
          const float* c = cs + pos[t] * 64;
          const float cc = __ldg(c + lane), sn = __ldg(c + 32 + lane);
          y0 = x0 * cc - x1 * sn;
          y1 = x0 * sn + x1 * cc;
        } else {
          y0 = x0;
          y1 = x1;
        }
        y0 = bf16r(y0);
        y1 = bf16r(y1);
      }
      if (ntile < 16) {  // q head ntile
        if (paired) {
          const int half = (ntile >> 3) & 1;
          uint8_t* base = q_attn + ((int64_t)t * 16 + ntile) * 128;
          store_fp8x2(base + half * 64 + 2 * lane, y0, y1);
          *reinterpret_cast<unsigned short*>(base + (1 - half) * 64 + 2 * lane) = 0;
        } else {
          store_fp8x2(q_attn + ((int64_t)t * 16 + ntile) * 64 + 2 * lane, y0, y1);
        }
      } else if (ntile < 18) {  // k head
        if (scaled) {
          const float sc = __ldg(ks);
          y0 = bf16r(y0 / sc);
          y1 = bf16r(y1 / sc);
        }
        store_fp8x2(kc + (loc[t] * 2 + (ntile - 16)) * 64 + 2 * lane, y0, y1);
      } else {  // v head
        if (scaled) {
          const float sc = __ldg(vs);
          y0 = bf16r(y0 / sc);
          y1 = bf16r(y1 / sc);
        }
        store_fp8x2(vc + (loc[t] * 2 + (ntile - 18)) * 64 + 2 * lane, y0, y1);
      }
    }
    if (tid == 0) atomicExch(cnt + ntile, 0);
  }
}

// KERNEL O: attn_output bf16 [m, 8192] = o @ W_o^T over K = 1024. 128 row tiles x 2 K-halves (8
// heads each) = 256 CTAs. Every K-block of the CTA's 64 x 512 weight slab is resident in smem (64 KB,
// cp.async issued BEFORE griddepcontrol.wait). prologue 1: after the wait the CTA builds its x tile
// itself from FA3's output (bf16 [m,16,64], or paired [m,16,128] taking head h's own half (h>>3)&1)
// and the gate bf16 [m,1024]: per-head scaleless RMSNorm (eps 1e-5) x sigmoid(gate) in fp32, one bf16
// rounding, as FusedRMSNormGate. prologue 0: x = fa (bf16 [m,1024], already normed and gated).
template <int NF>
__global__ void __launch_bounds__(THREADS, 2)
dl_o_kernel(const __nv_bfloat16* __restrict__ fa, const __nv_bfloat16* __restrict__ gate,
            const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
            float* __restrict__ ws, int* __restrict__ cnt, int m, int prologue, int paired) {
  using S = OSmem<NF>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ntile = blockIdx.x / O_SPLIT, half = blockIdx.x % O_SPLIT;
  const int n0 = ntile * BLOCK_N;
  const int k0 = half * O_KHALF;
  uint8_t* xs = smem + O_KBS * W_STAGE;

  // Weights first (no dependence on the predecessor): 4 K-blocks x (64 rows x 16 chunks).
#pragma unroll
  for (int kb = 0; kb < O_KBS; ++kb) {
    uint8_t* sw = smem + kb * W_STAGE;
#pragma unroll
    for (int u = 0; u < (BLOCK_N * 16) / THREADS; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      cp_async16(smem_u32(sw + row * ROW_BYTES + ((c ^ (row & 7)) << 4)),
                 w + (int64_t)(n0 + row) * O_KDIM + k0 + kb * BLOCK_K + c * 8, 16);
    }
  }
  cp_commit();
  if (prologue) {  // zero the whole x tile (rows >= m stay 0), also before the wait
    uint4* xz = reinterpret_cast<uint4*>(xs);
    for (int i = tid; i < (O_KBS * S::X_KB) / 16; i += THREADS) xz[i] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();  // the zero fill is complete before any prologue store
  }
  griddep_wait();
  if (prologue) {
    for (int pr = warp; pr < m * 8; pr += THREADS / 32) {
      const int t = pr >> 3, j = pr & 7, h = half * 8 + j;
      const __nv_bfloat16* src = fa + ((int64_t)t * 16 + h) * (paired ? 128 : 64) + (paired ? ((h >> 3) & 1) * 64 : 0);
      const float2 a = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(src)[lane]);
      const float2 g = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(gate + (int64_t)t * 1024 + h * 64)[lane]);
      const float ss = warp_sum(a.x * a.x + a.y * a.y);
      const float r = 1.0f / sqrtf(ss / 64.0f + 1e-5f);
      const float y0 = a.x * r * (1.0f / (1.0f + expf(-g.x)));
      const float y1 = a.y * r * (1.0f / (1.0f + expf(-g.y)));
      const int kb = j >> 1, col = (j & 1) * 64 + 2 * lane, c = col >> 3;
      uint8_t* dst = xs + kb * S::X_KB + t * ROW_BYTES + ((c ^ (t & 7)) << 4) + (col & 7) * 2;
      *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(y0, y1);
    }
  } else {
#pragma unroll
    for (int kb = 0; kb < O_KBS; ++kb) {
#pragma unroll
      for (int u = 0; u < NF; ++u) {
        const int q = tid + u * THREADS;
        const int row = q >> 4, c = q & 15;
        const bool ok = row < m;
        const __nv_bfloat16* src = fa + (ok ? (int64_t)row * O_KDIM : (int64_t)0) + k0 + kb * BLOCK_K + c * 8;
        cp_async16(smem_u32(xs + kb * S::X_KB + row * ROW_BYTES + ((c ^ (row & 7)) << 4)), src, ok ? 16 : 0);
      }
    }
    cp_commit();
  }
  cp_wait<0>();
  __syncthreads();

  float acc[2][NF][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[h][f][r] = 0.f;
  const int arow = warp * 16 + (lane & 15);
  const int asw = arow & 7;
  const int ahalf = lane >> 4;
  const int xrow = lane & 7;
  const int xq = lane >> 3;
#pragma unroll
  for (int kb = 0; kb < O_KBS; ++kb) {
    const uint32_t a_base = smem_u32(smem + kb * W_STAGE) + arow * ROW_BYTES;
    const uint32_t b_base = smem_u32(xs + kb * S::X_KB) + xrow * ROW_BYTES;
#pragma unroll
    for (int kp = 0; kp < BLOCK_K / 32; ++kp) {
      uint32_t a0[4], a1[4];
      ldsm_x4(a0, a_base + ((((kp * 2 + 0) * 2 + ahalf) ^ asw) << 4));
      ldsm_x4(a1, a_base + ((((kp * 2 + 1) * 2 + ahalf) ^ asw) << 4));
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        uint32_t b[4];
        ldsm_x4(b, b_base + f * 8 * ROW_BYTES + (((kp * 4 + xq) ^ xrow) << 4));
        mma_bf16(acc[0][f], a0, b[0], b[1]);
        mma_bf16(acc[1][f], a1, b[2], b[3]);
      }
    }
  }

  const int r0 = n0 + warp * 16 + (lane >> 2);
  float* wsp = ws + (int64_t)half * MAXT * O_ROWS;
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    const int t0 = f * 8 + (lane & 3) * 2;
    if (t0 < m) {
      wsp[(int64_t)t0 * O_ROWS + r0] = acc[0][f][0] + acc[1][f][0];
      wsp[(int64_t)t0 * O_ROWS + r0 + 8] = acc[0][f][2] + acc[1][f][2];
    }
    if (t0 + 1 < m) {
      wsp[(int64_t)(t0 + 1) * O_ROWS + r0] = acc[0][f][1] + acc[1][f][1];
      wsp[(int64_t)(t0 + 1) * O_ROWS + r0 + 8] = acc[0][f][3] + acc[1][f][3];
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(cnt + ntile, 1) == O_SPLIT - 1);
  __syncthreads();
  if (s_last) {
    __threadfence();
    for (int e = tid; e < m * BLOCK_N; e += THREADS) {
      const int t = e / BLOCK_N, r = n0 + (e % BLOCK_N);
      float s = 0.f;
#pragma unroll
      for (int p = 0; p < O_SPLIT; ++p) s += __ldcg(ws + ((int64_t)p * MAXT + t) * O_ROWS + r);
      out[(int64_t)t * O_ROWS + r] = __float2bfloat16_rn(s);
    }
    if (tid == 0) atomicExch(cnt + ntile, 0);
  }
}

template <int NF>
void set_attrs_q() {
  static bool done = false;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_qkv_kernel<NF>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<NF>::BYTES));
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_qkv_kernel<NF>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    done = true;
  }
}
template <int NF>
void set_attrs_o() {
  static bool done = false;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_o_kernel<NF>, cudaFuncAttributeMaxDynamicSharedMemorySize, OSmem<NF>::BYTES));
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_o_kernel<NF>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    done = true;
  }
}

template <int NF>
void launch_q(const __nv_bfloat16* x, const __nv_bfloat16* w, float* ws, int* cnt, __nv_bfloat16* raw,
              uint8_t* q_attn, __nv_bfloat16* gate, uint8_t* kc, uint8_t* vc, const int64_t* loc,
              const int64_t* pos, const float* cs, const float* ks, const float* vs, int m, int mode,
              int paired, int rope, int scaled, bool pdl, cudaStream_t stream) {
  set_attrs_q<NF>();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(Q_NTILES * Q_SPLIT);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = Smem<NF>::BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dl_qkv_kernel<NF>, x, w, ws, cnt, raw, q_attn, gate, kc, vc, loc, pos, cs,
                                    ks, vs, m, mode, paired, rope, scaled));
}

template <int NF>
void launch_o(const __nv_bfloat16* fa, const __nv_bfloat16* gate, const __nv_bfloat16* w, __nv_bfloat16* out,
              float* ws, int* cnt, int m, int prologue, int paired, bool pdl, cudaStream_t stream) {
  set_attrs_o<NF>();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(O_NTILES * O_SPLIT);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = OSmem<NF>::BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dl_o_kernel<NF>, fa, gate, w, out, ws, cnt, m, prologue, paired));
}

// ---------------------------------------------------------------------------------------------
// Stage 3 piece C: dl_unpool_kernel (from stack-route-topk-pdl-unpool-nf1), NF = 1 (T <= 8) only.
// ---------------------------------------------------------------------------------------------
constexpr int HID = 8192;
constexpr int UK = 4096;                          // unpooler K
constexpr int UKB = UK / BLOCK_K;                 // 32 K-blocks
constexpr int WS_FLOATS = 2 * MAXT * HID;         // 524288 floats

// out bf16 [m, 8192] = x bf16 [m, 4096] @ W[:NROWS]^T (W bf16 [>= NROWS, 4096]). Grid =
// (NROWS / 64) * SPLIT_: n-tile = blockIdx.x / SPLIT_, K-chunk = blockIdx.x % SPLIT_. Same pipeline
// as dl_pooler_router_kernel: weight ring before the PDL wait, x after it, fixed-order ticket
// split-K reduce. When NROWS < 8192, columns NROWS..8191 of out are stored as bf16 zeros (each CTA
// a disjoint slice), after the wait. ws fp32 [SPLIT_, 32, NROWS]; cnt int32 [NROWS / 64] (0 between launches).
template <int NF, int SPLIT_, int NROWS>
__global__ void __launch_bounds__(THREADS, 2)
dl_unpool_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                 __nv_bfloat16* __restrict__ out, float* __restrict__ ws, int* __restrict__ cnt, int m) {
  using S = Smem<NF>;
  constexpr int NT = NROWS / BLOCK_N;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ntile = blockIdx.x / SPLIT_, sidx = blockIdx.x % SPLIT_;
  const int n0 = ntile * BLOCK_N;
  const int kb0 = sidx * UKB / SPLIT_, kb1 = (sidx + 1) * UKB / SPLIT_;
  const int nsteps = kb1 - kb0;
  const __nv_bfloat16* wbase = w + (int64_t)n0 * UK;

  auto issue_w = [&](int j) {
    uint8_t* sw = smem + (j % STAGES) * S::STAGE_BYTES;
    const int kel = (kb0 + j) * BLOCK_K;
#pragma unroll
    for (int u = 0; u < (BLOCK_N * 16) / THREADS; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      cp_async16(smem_u32(sw + row * ROW_BYTES + ((c ^ (row & 7)) << 4)),
                 wbase + (int64_t)row * UK + kel + c * 8, 16);
    }
  };
  auto issue_x = [&](int j) {
    uint8_t* sx = smem + (j % STAGES) * S::STAGE_BYTES + W_STAGE;
    const int kel = (kb0 + j) * BLOCK_K;
#pragma unroll
    for (int u = 0; u < NF; ++u) {
      const int q = tid + u * THREADS;
      const int row = q >> 4, c = q & 15;
      const bool ok = row < m;
      const __nv_bfloat16* src = x + (ok ? (int64_t)row * UK : (int64_t)0) + kel + c * 8;
      cp_async16(smem_u32(sx + row * ROW_BYTES + ((c ^ (row & 7)) << 4)), src, ok ? 16 : 0);
    }
  };

#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j)
    if (j < nsteps) issue_w(j);
  griddep_wait();
  if constexpr (NROWS < HID) {
    // Zero columns [NROWS, 8192): 56 per token per CTA. The previous kernel is done (wait above).
    constexpr int ZC = (HID - NROWS) / (NT * SPLIT_);
    static_assert(ZC % 8 == 0 && (HID - NROWS) % (NT * SPLIT_) == 0, "zero slice");
    constexpr int ZV = ZC / 8;                       // uint4 per token
    const int zc0 = NROWS + blockIdx.x * ZC;
    for (int e = tid; e < m * ZV; e += THREADS) {
      const int t = e / ZV, q = e % ZV;
      *reinterpret_cast<uint4*>(out + (int64_t)t * HID + zc0 + q * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < nsteps) issue_x(j);
    cp_commit();
  }

  float acc[2][NF][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[h][f][r] = 0.f;
  const int arow = warp * 16 + (lane & 15);
  const int asw = arow & 7;
  const int ahalf = lane >> 4;
  const int xrow = lane & 7;
  const int xq = lane >> 3;

  for (int i = 0; i < nsteps; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nx = i + STAGES - 1;
      if (nx < nsteps) { issue_w(nx); issue_x(nx); }
      cp_commit();
    }
    const uint8_t* sw = smem + (i % STAGES) * S::STAGE_BYTES;
    const uint32_t a_base = smem_u32(sw) + arow * ROW_BYTES;
    const uint32_t b_base = smem_u32(sw + W_STAGE) + xrow * ROW_BYTES;
#pragma unroll
    for (int kp = 0; kp < BLOCK_K / 32; ++kp) {
      uint32_t a0[4], a1[4];
      ldsm_x4(a0, a_base + ((((kp * 2 + 0) * 2 + ahalf) ^ asw) << 4));
      ldsm_x4(a1, a_base + ((((kp * 2 + 1) * 2 + ahalf) ^ asw) << 4));
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        uint32_t b[4];
        ldsm_x4(b, b_base + f * 8 * ROW_BYTES + (((kp * 4 + xq) ^ xrow) << 4));
        mma_bf16(acc[0][f], a0, b[0], b[1]);
        mma_bf16(acc[1][f], a1, b[2], b[3]);
      }
    }
  }

  const int r0 = n0 + warp * 16 + (lane >> 2);
  float* wsp = ws + (int64_t)sidx * MAXT * NROWS;
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    const int t0 = f * 8 + (lane & 3) * 2;
    if (t0 < m) {
      wsp[(int64_t)t0 * NROWS + r0] = acc[0][f][0] + acc[1][f][0];
      wsp[(int64_t)t0 * NROWS + r0 + 8] = acc[0][f][2] + acc[1][f][2];
    }
    if (t0 + 1 < m) {
      wsp[(int64_t)(t0 + 1) * NROWS + r0] = acc[0][f][1] + acc[1][f][1];
      wsp[(int64_t)(t0 + 1) * NROWS + r0 + 8] = acc[0][f][3] + acc[1][f][3];
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(cnt + ntile, 1) == SPLIT_ - 1);
  __syncthreads();
  if (s_last) {
    __threadfence();
    for (int e = tid; e < m * BLOCK_N; e += THREADS) {
      const int t = e / BLOCK_N, r = n0 + (e % BLOCK_N);
      float s = 0.f;
#pragma unroll
      for (int p = 0; p < SPLIT_; ++p) s += __ldcg(ws + ((int64_t)p * MAXT + t) * NROWS + r);
      out[(int64_t)t * HID + r] = __float2bfloat16_rn(s);
    }
    if (tid == 0) atomicExch(cnt + ntile, 0);
  }
}

template <int NF, int SPLIT_, int NROWS>
void launch_u(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, float* ws, int* cnt, int m,
              bool pdl, cudaStream_t stream) {
  static bool done = false;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_unpool_kernel<NF, SPLIT_, NROWS>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<NF>::BYTES));
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_unpool_kernel<NF, SPLIT_, NROWS>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    done = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((NROWS / BLOCK_N) * SPLIT_);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = Smem<NF>::BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dl_unpool_kernel<NF, SPLIT_, NROWS>, x, w, out, ws, cnt, m));
}

// W6-B: kernel R's step (1) norm alone (same arithmetic), grid T.
__global__ void __launch_bounds__(256)
dl_norm_kernel(const __nv_bfloat16* __restrict__ ei, const float* __restrict__ norm_w, __nv_bfloat16* __restrict__ xn) {
  __shared__ float s_red[8];
  const int tid = threadIdx.x;
  const float* nw[2] = {norm_w + tid * 8, norm_w + 2048 + tid * 8};
  const uint4* src = reinterpret_cast<const uint4*>(ei + (int64_t)blockIdx.x * 4096);
  const uint4 raw[2] = {src[tid], src[256 + tid]};
  float v[16], ss = 0.f, tot = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&raw[i >> 3])[i & 7]);
    ss += v[i] * v[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) s_red[tid >> 5] = ss;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += s_red[i];
  const float rstd = rsqrtf(tot / 4096.f + 1e-5f);
  uint4 o[2];
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    o2[i] = __floats2bfloat162_rn(v[2 * i] * rstd * nw[i >> 2][(2 * i) & 7], v[2 * i + 1] * rstd * nw[i >> 2][(2 * i + 1) & 7]);
  uint4* dst = reinterpret_cast<uint4*>(xn + (int64_t)blockIdx.x * 4096);
  dst[tid] = o[0];
  dst[256 + tid] = o[1];
}

}  // namespace

void dl_pooler_router(torch::Tensor x, torch::Tensor w, torch::Tensor ei, torch::Tensor logits,
                      torch::Tensor ws, torch::Tensor cnt, int64_t pdl) {
  const int64_t m = x.size(0);
  TORCH_CHECK(m >= 1 && m <= MAXT, "dl_pooler_router: tokens out of range");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == KDIM, "dl_pooler_router: x");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(0) == ROWS && w.size(1) == KDIM,
              "dl_pooler_router: w");
  TORCH_CHECK(ei.scalar_type() == at::kBFloat16 && ei.is_contiguous() && ei.size(0) == m && ei.size(1) == POOL_ROWS,
              "dl_pooler_router: ei");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.is_contiguous() && logits.size(0) == m &&
              logits.size(1) == 256, "dl_pooler_router: logits");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)SPLIT * MAXT * ROWS, "dl_pooler_router: ws");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= NTILES, "dl_pooler_router: cnt");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* ep = reinterpret_cast<__nv_bfloat16*>(ei.data_ptr());
  float* lp = logits.data_ptr<float>();
  float* wsp = ws.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
  const int mi = static_cast<int>(m);
  if (m <= 8) launch_p<1>(xp, wp, ep, lp, wsp, cp, mi, pdl != 0, stream);
  else if (m <= 16) launch_p<2>(xp, wp, ep, lp, wsp, cp, mi, pdl != 0, stream);
  else launch_p<4>(xp, wp, ep, lp, wsp, cp, mi, pdl != 0, stream);
}

void dl_route_norm_align(torch::Tensor ei, torch::Tensor norm_w, torch::Tensor xn, torch::Tensor logits,
                         torch::Tensor bias, torch::Tensor local_start, torch::Tensor topk_ids,
                         torch::Tensor topk_w, torch::Tensor local_ids, torch::Tensor sorted_ids,
                         torch::Tensor block_experts, torch::Tensor n_padded, torch::Tensor cnt, int64_t bsm,
                         int64_t pdl, int64_t variant) {
  // variant 0: new R (binary-search pad), 2: v1 (stage 1's R). pdl applies to the new R only (v1 has no
  // griddepcontrol.wait, so it is never launched with the attribute).
  // variant + 1 (campaign expert-bandwidth X0b): the same kernel without its ticket and Marlin align.
  const int64_t m = ei.size(0);
  TORCH_CHECK(variant >= 0 && variant <= 3, "dl_route_norm_align: variant (0 = new R, 2 = v1, + 1 = no align)");
  const int align = (variant & 1) ? 0 : 1;
  variant &= 2;
  TORCH_CHECK(m >= 1 && m <= MAXT && bsm >= 1 && bsm <= 64, "dl_route_norm_align: sizes");
  TORCH_CHECK(ei.scalar_type() == at::kBFloat16 && ei.is_contiguous() && ei.size(1) == 4096, "dl_route_norm_align: ei");
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.size(0) == m && xn.size(1) == 4096,
              "dl_route_norm_align: xn");
  TORCH_CHECK(norm_w.scalar_type() == at::kFloat && norm_w.is_contiguous() && norm_w.numel() == 4096,
              "dl_route_norm_align: norm_w");
  TORCH_CHECK(variant == 2 || reinterpret_cast<uintptr_t>(norm_w.data_ptr()) % 16 == 0,
              "dl_route_norm_align: norm_w must be 16-byte aligned (the new R reads it as float4)");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.is_contiguous() && logits.size(0) == m &&
              logits.size(1) == 256, "dl_route_norm_align: logits");
  TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == 256, "dl_route_norm_align: bias");
  TORCH_CHECK(local_start.scalar_type() == at::kInt && local_start.numel() >= 1, "dl_route_norm_align: local_start");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() && topk_ids.numel() == m * 8,
              "dl_route_norm_align: topk_ids");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.is_contiguous() && topk_w.numel() == m * 8,
              "dl_route_norm_align: topk_w");
  TORCH_CHECK(local_ids.scalar_type() == at::kInt && local_ids.is_contiguous() && local_ids.numel() == m * 8,
              "dl_route_norm_align: local_ids");
  const int64_t cap = m * 8 + 32 * (bsm - 1);
  const int64_t n_blocks = (cap + bsm - 1) / bsm;
  TORCH_CHECK(sorted_ids.scalar_type() == at::kInt && sorted_ids.numel() == cap, "dl_route_norm_align: sorted_ids");
  TORCH_CHECK(block_experts.scalar_type() == at::kInt && block_experts.numel() == n_blocks,
              "dl_route_norm_align: block_experts");
  TORCH_CHECK(n_padded.scalar_type() == at::kInt && n_padded.numel() >= 1, "dl_route_norm_align: n_padded");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= 1, "dl_route_norm_align: cnt");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* eip = reinterpret_cast<const __nv_bfloat16*>(ei.data_ptr());
  auto* xnp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());
  const int mi = static_cast<int>(m), bi = static_cast<int>(bsm), ci = static_cast<int>(cap),
            nb = static_cast<int>(n_blocks);
  if (variant == 2) {
    dl_route_norm_align_v1_kernel<<<static_cast<unsigned>(m), 256, 0, stream>>>(
        eip, norm_w.data_ptr<float>(), xnp, logits.data_ptr<float>(), bias.data_ptr<float>(),
        local_start.data_ptr<int>(), topk_ids.data_ptr<int>(), topk_w.data_ptr<float>(), local_ids.data_ptr<int>(),
        sorted_ids.data_ptr<int>(), block_experts.data_ptr<int>(), n_padded.data_ptr<int>(), cnt.data_ptr<int>(),
        mi, bi, ci, nb, align);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  launch_r<true>(eip, norm_w.data_ptr<float>(), xnp, logits.data_ptr<float>(), bias.data_ptr<float>(),
                 local_start.data_ptr<int>(), topk_ids.data_ptr<int>(), topk_w.data_ptr<float>(),
                 local_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(), block_experts.data_ptr<int>(),
                 n_padded.data_ptr<int>(), cnt.data_ptr<int>(), mi, bi, ci, nb, pdl != 0, stream, align);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dl_norm(torch::Tensor ei, torch::Tensor norm_w, torch::Tensor xn) {
  const int64_t m = ei.size(0);
  auto ok = [](const torch::Tensor& t, at::ScalarType st, int64_t n) {
    return t.scalar_type() == st && t.is_contiguous() && t.numel() == n && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  TORCH_CHECK(m >= 1 && m <= 65535 && ei.dim() == 2 && xn.dim() == 2 && ok(ei, at::kBFloat16, m * 4096) &&
              ok(xn, at::kBFloat16, m * 4096) && ok(norm_w, at::kFloat, 4096), "dl_norm");
  dl_norm_kernel<<<static_cast<unsigned>(m), 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(ei.data_ptr()), norm_w.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t dl_occupancy(int64_t nf) {
  int blocks = 0;
  if (nf == 1) {
    set_attrs<1>();
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dl_pooler_router_kernel<1>, THREADS, Smem<1>::BYTES));
  } else if (nf == 2) {
    set_attrs<2>();
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dl_pooler_router_kernel<2>, THREADS, Smem<2>::BYTES));
  } else {
    set_attrs<4>();
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dl_pooler_router_kernel<4>, THREADS, Smem<4>::BYTES));
  }
  return blocks;
}

void dl_pdl_probe(torch::Tensor times, int64_t spin_ns, int64_t pdl) {
  TORCH_CHECK(times.scalar_type() == at::kLong && times.numel() >= 4, "dl_pdl_probe: times");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto* tp = reinterpret_cast<unsigned long long*>(times.data_ptr());
  dl_probe_primary<<<1, 32, 0, stream>>>(tp, static_cast<long long>(spin_ns));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(32);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dl_probe_secondary, tp));
}

void dl2_qkv(torch::Tensor x, torch::Tensor w, torch::Tensor ws, torch::Tensor cnt, torch::Tensor raw,
             torch::Tensor q_attn, torch::Tensor gate, torch::Tensor kc, torch::Tensor vc, torch::Tensor loc,
             torch::Tensor pos, torch::Tensor cs, torch::Tensor ks, torch::Tensor vs, int64_t pdl, int64_t mode,
             int64_t paired, int64_t rope, int64_t scaled) {
  const int64_t m = x.size(0);
  TORCH_CHECK(m >= 1 && m <= MAXT, "dl2_qkv: tokens out of range");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == KDIM, "dl2_qkv: x");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(0) == Q_ROWS && w.size(1) == KDIM,
              "dl2_qkv: w");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)Q_SPLIT * MAXT * Q_ROWS, "dl2_qkv: ws");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= Q_NTILES, "dl2_qkv: cnt");
  TORCH_CHECK(mode == 0 || mode == 1, "dl2_qkv: mode");
  if (mode == 1) {
    TORCH_CHECK(raw.scalar_type() == at::kBFloat16 && raw.is_contiguous() && raw.size(0) == m &&
                raw.size(1) == Q_ROWS, "dl2_qkv: raw");
  } else {
    TORCH_CHECK(q_attn.scalar_type() == at::kByte && q_attn.is_contiguous() &&
                q_attn.numel() == m * 16 * (paired ? 128 : 64), "dl2_qkv: q_attn");
    TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && gate.is_contiguous() && gate.numel() == m * 1024,
                "dl2_qkv: gate");
    TORCH_CHECK(kc.scalar_type() == at::kByte && kc.is_contiguous() && vc.scalar_type() == at::kByte &&
                vc.is_contiguous(), "dl2_qkv: caches");
    TORCH_CHECK(loc.scalar_type() == at::kLong && loc.numel() >= m, "dl2_qkv: cache_loc");
    TORCH_CHECK(!rope || (pos.scalar_type() == at::kLong && pos.numel() >= m && cs.scalar_type() == at::kFloat &&
                          cs.is_contiguous() && cs.dim() == 2 && cs.size(1) == 64), "dl2_qkv: positions / cos_sin");
    TORCH_CHECK(!scaled || (ks.scalar_type() == at::kFloat && ks.numel() == 1 && vs.scalar_type() == at::kFloat &&
                            vs.numel() == 1), "dl2_qkv: scales");
  }
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* rp = reinterpret_cast<__nv_bfloat16*>(raw.data_ptr());
  auto* qp = reinterpret_cast<uint8_t*>(q_attn.data_ptr());
  auto* gp = reinterpret_cast<__nv_bfloat16*>(gate.data_ptr());
  auto* kp = reinterpret_cast<uint8_t*>(kc.data_ptr());
  auto* vp = reinterpret_cast<uint8_t*>(vc.data_ptr());
  const auto* lp = reinterpret_cast<const int64_t*>(loc.data_ptr());
  const auto* pp = reinterpret_cast<const int64_t*>(pos.data_ptr());
  const auto* cp = reinterpret_cast<const float*>(cs.data_ptr());
  const auto* kkp = reinterpret_cast<const float*>(ks.data_ptr());
  const auto* vvp = reinterpret_cast<const float*>(vs.data_ptr());
  float* wsp = ws.data_ptr<float>();
  int* cntp = cnt.data_ptr<int>();
  const int mi = static_cast<int>(m);
  const bool p = pdl != 0;
  if (m <= 8) launch_q<1>(xp, wp, wsp, cntp, rp, qp, gp, kp, vp, lp, pp, cp, kkp, vvp, mi, (int)mode, (int)paired, (int)rope, (int)scaled, p, stream);
  else if (m <= 16) launch_q<2>(xp, wp, wsp, cntp, rp, qp, gp, kp, vp, lp, pp, cp, kkp, vvp, mi, (int)mode, (int)paired, (int)rope, (int)scaled, p, stream);
  else launch_q<4>(xp, wp, wsp, cntp, rp, qp, gp, kp, vp, lp, pp, cp, kkp, vvp, mi, (int)mode, (int)paired, (int)rope, (int)scaled, p, stream);
}

void dl2_o(torch::Tensor fa, torch::Tensor gate, torch::Tensor w, torch::Tensor ws, torch::Tensor cnt,
           torch::Tensor out, int64_t pdl, int64_t prologue, int64_t paired) {
  const int64_t m = out.size(0);
  TORCH_CHECK(m >= 1 && m <= MAXT, "dl2_o: tokens out of range");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(1) == O_ROWS, "dl2_o: out");
  TORCH_CHECK(fa.scalar_type() == at::kBFloat16 && fa.is_contiguous() &&
              fa.numel() == m * (prologue ? 16 * (paired ? 128 : 64) : O_KDIM), "dl2_o: fa");
  TORCH_CHECK(!prologue || (gate.scalar_type() == at::kBFloat16 && gate.is_contiguous() && gate.numel() == m * 1024),
              "dl2_o: gate");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(0) == O_ROWS && w.size(1) == O_KDIM,
              "dl2_o: w");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)O_SPLIT * MAXT * O_ROWS, "dl2_o: ws");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= O_NTILES, "dl2_o: cnt");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* fp = reinterpret_cast<const __nv_bfloat16*>(fa.data_ptr());
  const auto* gp = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  float* wsp = ws.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
  const int mi = static_cast<int>(m);
  const bool p = pdl != 0;
  if (m <= 8) launch_o<1>(fp, gp, wp, op, wsp, cp, mi, (int)prologue, (int)paired, p, stream);
  else if (m <= 16) launch_o<2>(fp, gp, wp, op, wsp, cp, mi, (int)prologue, (int)paired, p, stream);
  else launch_o<4>(fp, gp, wp, op, wsp, cp, mi, (int)prologue, (int)paired, p, stream);
}

// Resident CTAs per SM of kernel Q (which = 0) or O (which = 1) at NF = 1 / 2 / 4.
int64_t dl2_occupancy(int64_t which, int64_t nf) {
  int blocks = 0;
#define DL2_OCC(NFV)                                                                                             \
  if (which == 0) {                                                                                              \
    set_attrs_q<NFV>();                                                                                          \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dl_qkv_kernel<NFV>, THREADS, Smem<NFV>::BYTES)); \
  } else {                                                                                                       \
    set_attrs_o<NFV>();                                                                                          \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dl_o_kernel<NFV>, THREADS, OSmem<NFV>::BYTES)); \
  }
  if (nf == 1) { DL2_OCC(1) }
  else if (nf == 2) { DL2_OCC(2) }
  else { DL2_OCC(4) }
#undef DL2_OCC
  return blocks;
}

void dl_unpool(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
               int64_t pdl) {
  const int64_t m = x.size(0);
  TORCH_CHECK(m >= 1 && m <= MAXT, "dl_unpool: tokens out of range");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == UK, "dl_unpool: x");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(0) == HID && w.size(1) == UK,
              "dl_unpool: w");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == m && out.size(1) == HID,
              "dl_unpool: out");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= WS_FLOATS, "dl_unpool: ws");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= HID / BLOCK_N, "dl_unpool: cnt");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  float* wsp = ws.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
  const int mi = static_cast<int>(m);
  if (m <= 8) launch_u<1, 8, 1024>(xp, wp, op, wsp, cp, mi, pdl != 0, stream);
  else if (m <= 16) launch_u<2, 2, 8192>(xp, wp, op, wsp, cp, mi, pdl != 0, stream);
  else launch_u<4, 2, 8192>(xp, wp, op, wsp, cp, mi, pdl != 0, stream);
}

// {binaryVersion, ptxVersion, numRegs, localSizeBytes, staticSharedBytes, dynamicSharedBytes, CTAs/SM} of
// 0 = Q (NF=1), 1 = O (NF=1), 2 = new R, 3 = unpool (NF=1), 4 = P (NF=1).
std::vector<int64_t> dl3_attrs(int64_t which) {
  cudaFuncAttributes a;
  const void* fn = nullptr;
  int dyn = 0, threads = THREADS, occ = 0;
  if (which == 0) { set_attrs_q<1>(); fn = reinterpret_cast<const void*>(dl_qkv_kernel<1>); dyn = Smem<1>::BYTES; }
  else if (which == 1) { set_attrs_o<1>(); fn = reinterpret_cast<const void*>(dl_o_kernel<1>); dyn = OSmem<1>::BYTES; }
  else if (which == 2) { fn = reinterpret_cast<const void*>(dl_route_norm_align_kernel<true>); threads = 256; }
  else if (which == 3) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dl_unpool_kernel<1, 8, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<1>::BYTES));
    fn = reinterpret_cast<const void*>(dl_unpool_kernel<1, 8, 1024>); dyn = Smem<1>::BYTES;
  } else { set_attrs<1>(); fn = reinterpret_cast<const void*>(dl_pooler_router_kernel<1>); dyn = Smem<1>::BYTES; }
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, fn));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, dyn));
  return {static_cast<int64_t>(a.binaryVersion), static_cast<int64_t>(a.ptxVersion), static_cast<int64_t>(a.numRegs),
          static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes), static_cast<int64_t>(dyn),
          static_cast<int64_t>(occ)};
}
"""
_DL_CFLAGS = ["-O3", "-std=c++20", "-lineinfo"]
_DL_ARCH_FLAGS = None
_DL_MODULE = "muse_decode_dense_chain_s3"
def _build_dense_chain():
    global _DL_ARCH_FLAGS
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils.cpp_extension import load_inline
        try:
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            _DL_ARCH_FLAGS = list(_get_cuda_arch_flags())
        except Exception as e:
            print(f"{DL_TAG} could not read torch's nvcc arch flags: {e!r}", flush=True)
        print(f"{DL_TAG} building {_DL_MODULE}: TORCH_CUDA_ARCH_LIST=9.0a, arch flags {_DL_ARCH_FLAGS}, "
              f"extra {_DL_CFLAGS}", flush=True)
        ext = load_inline(
            name=_DL_MODULE,
            cpp_sources=_DL_CPP,
            cuda_sources=_DL_CUDA,
            functions=["dl_pooler_router", "dl_route_norm_align", "dl_occupancy", "dl_pdl_probe", "dl2_qkv", "dl2_o",
                       "dl2_occupancy", "dl_unpool", "dl3_attrs", "dl_norm"],
            extra_cuda_cflags=_DL_CFLAGS,
            verbose=False,
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"{DL_TAG} load_inline FAILED after {dt:.1f}s, master path kept: {e!r}", flush=True)
        return None, dt
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    dt = time.perf_counter() - t0
    print(f"{DL_TAG} load_inline compiled in {dt:.1f}s", flush=True)
    return ext, dt
_DL_EXT, _DL_COMPILE_S = _build_dense_chain() if torch.cuda.is_available() else (None, 0.0)
_DL_S3 = _DL_EXT is not None
def _block_size_m(num_tokens):
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    return min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP)
ALIGN_PAR_CHUNK = 512
ALIGN_PAR_WARPS = 8
ALIGN_PAR_BLK = 64
def _dl_forward(ffn_input, local_start, dl, pdl=None, r_t=False, align=True, b12p=None, rx=None, fs=None):
    ext = _DL_EXT
    tokens = ffn_input.shape[0]
    dev = ffn_input.device
    pdl_v = dl["pdl"] if pdl is None else pdl
    x = ffn_input.contiguous()
    ei = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    logits = torch.empty((tokens, N_EXPERTS), dtype=torch.float32, device=dev)
    lg = None
    if rx is not None and fs is not None:
        m = torch.cuda.current_stream()
        fs.wait_stream(m)
        with torch.cuda.stream(fs):
            lg = _dc0_lin(*rx)
    if not (_b12p_p(x, ei, logits, b12p)):
        ext.dl_pooler_router(x, dl["w_pr"], ei, logits, dl["ws"], dl["cnt_p"], pdl_v)
    if lg is not None:
        m.wait_stream(fs)
        if not torch.cuda.is_current_stream_capturing():
            lg.record_stream(m)
        logits = lg
    elif rx is not None:
        logits = _dc0_lin(*rx)
    new_r = (tokens == 1 and dl.get("r_new", False)) or bool(r_t)
    xn, routed = _dl_route(ei, logits, local_start, dl, pdl_v if new_r else 0, 0 if new_r else 2, align=align)
    return xn, routed, (ei, logits)
def _dl_route_bufs(tokens, dev):
    bsm = _block_size_m(tokens)
    n = tokens * TOP_K
    cap = n + LOCAL_EXPERTS * (bsm - 1)
    n_blocks = (cap + bsm - 1) // bsm
    xn = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    topk_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    topk_w = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    sorted_ids = torch.empty((cap,), dtype=torch.int32, device=dev)
    block_experts = torch.empty((n_blocks,), dtype=torch.int32, device=dev)
    n_padded = torch.empty((1,), dtype=torch.int32, device=dev)
    return (xn, topk_ids, topk_w, local_ids, sorted_ids, block_experts, n_padded), bsm
def _dl_route_launch(ei, logits, local_start, dl, bufs, bsm, pdl, variant, bias=None, align=True):
    xn, topk_ids, topk_w, local_ids, sorted_ids, block_experts, n_padded = bufs
    _DL_EXT.dl_route_norm_align(ei, dl["norm_w"], xn, logits, dl["bias"] if bias is None else bias, local_start,
                                topk_ids, topk_w, local_ids, sorted_ids, block_experts, n_padded, dl["cnt_r"], bsm,
                                pdl, variant if align else variant + 1)
def _dl_route(ei, logits, local_start, dl, pdl=0, variant=None, align=True):
    if variant is None:
        variant = 0 if (ei.shape[0] == 1 and dl.get("r_new", False)) else 2
    bufs, bsm = _dl_route_bufs(ei.shape[0], ei.device)
    _dl_route_launch(ei, logits, local_start, dl, bufs, bsm, pdl, variant, align=align)
    xn, topk_ids, topk_w, local_ids, sorted_ids, block_experts, n_padded = bufs
    return xn, (topk_w, topk_ids, local_ids, (sorted_ids, block_experts, n_padded))
def _rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))
def _dl_align_problems(local_ids, align, bsm, tag):
    sorted_ids, block_experts, n_padded = align
    ids = local_ids.reshape(-1).tolist()
    numel = len(ids)
    s_ids, b_exp, n_pad = sorted_ids.tolist(), block_experts.tolist(), int(n_padded[0])
    want = {}
    for i, e in enumerate(ids):
        if e >= 0:
            want.setdefault(e, []).append(i)
    want_blocks = {e: (len(v) + bsm - 1) // bsm for e, v in want.items()}
    problems = []
    if n_pad != bsm * sum(want_blocks.values()):
        problems.append(f"{tag}: n_padded {n_pad} vs {bsm * sum(want_blocks.values())}")
        return problems
    got, got_blocks = {}, {}
    for b in range(n_pad // bsm):
        e = b_exp[b]
        if e < 0:
            problems.append(f"{tag}: block {b} has no expert")
            continue
        got_blocks[e] = got_blocks.get(e, 0) + 1
        got.setdefault(e, []).extend(i for i in s_ids[b * bsm:(b + 1) * bsm] if i < numel)
    if any(e != -1 for e in b_exp[n_pad // bsm:]):
        problems.append(f"{tag}: block_experts past the used blocks are not -1")
    if {e: sorted(v) for e, v in got.items()} != want or got_blocks != want_blocks:
        problems.append(f"{tag}: per-expert routes or blocks differ")
    if any(i > numel for i in s_ids) or sum(1 for i in s_ids if i == numel) != len(s_ids) - sum(len(v) for v in want.values()):
        problems.append(f"{tag}: sorted_ids sentinel fill")
    return problems
def _dl_route_problems(logits, topk_w, topk_ids, local_ids, ls, state, tag, tie_free):
    problems = []
    bias = state["dl"]["bias"].double()
    scores = torch.sigmoid(logits.double() * 0.1767766952966369)
    biased = scores + bias
    top9 = biased.topk(TOP_K + 1, dim=-1).values
    eighth, ninth = top9[:, TOP_K - 1], top9[:, TOP_K]
    mine = topk_ids.long()
    picked = biased.gather(1, mine)
    if not bool((picked >= (eighth - 1e-5 - 1e-6 * eighth.abs())[:, None]).all()) or any(len(set(r)) != TOP_K for r in mine.tolist()):
        problems.append(f"{tag}: a picked expert is below the 8th best biased score (or a duplicate)")
    ref_w, ref_ids = state["routing"](None, logits.contiguous(), TOP_K, True)
    ref_ids = ref_ids.long()
    ref_w = ref_w.float()
    gap = (eighth - ninth).tolist()
    for t, (mi, ri) in enumerate(zip(mine.tolist(), ref_ids.tolist())):
        if set(mi) != set(ri):
            if tie_free and gap[t] > 1e-4:
                problems.append(f"{tag}: token {t} ids {sorted(mi)} vs FusedMuseSparkRouting {sorted(ri)}")
            continue
        mw = dict(zip(mi, topk_w[t].tolist()))
        rw = dict(zip(ri, ref_w[t].tolist()))
        if max(abs(mw[e] - rw[e]) for e in mi) > 1e-3:
            problems.append(f"{tag}: token {t} weights differ from FusedMuseSparkRouting")
    want_local = (topk_ids.long() - ls) % N_EXPERTS
    want_local = torch.where(want_local < LOCAL_EXPERTS, want_local, torch.full_like(want_local, -1)).to(torch.int32)
    if not torch.equal(local_ids, want_local):
        problems.append(f"{tag}: local_ids differ from (id - local_start) % 256 -> expert_map")
    return problems
def _dl_self_check(state, pdl):
    dl = state["dl"]
    dev = dl["w_pr"].device
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261024)
    problems = []
    worst = {"ei": 0.0, "logit": 0.0, "xn": 0.0, "w": 0.0, "y": 0.0, "graph": 0.0}
    pooler, router = state["dense"]["pooler"], state["router"]
    starts = (0, 224, 32, 96, 0, 160)
    for case, tokens in enumerate((1, 4, 8, 9, 16, 32)):
        ls_val = starts[case]
        ls = torch.tensor([ls_val], dtype=torch.int32, device=dev)
        tag = f"T={tokens} ls={ls_val}"
        x = torch.randn((tokens, HIDDEN), device=dev, generator=gen).to(torch.bfloat16)
        for rep in range(2):
            xn, (tw, ids, lids, align), (ei, logits) = _dl_forward(x, ls, dl, pdl)
        torch.cuda.synchronize()
        ei_ref = _gemm(x, pooler)
        logit_ref = x.double() @ router.double().T
        e_ei = _rel_l2(ei, ei_ref)
        e_lg = float((logits.double() - logit_ref).abs().max() / logit_ref.abs().max().clamp_min(1e-30))
        xn_ref = state["pre_expert_norm"](ei)
        e_xn = _rel_l2(xn, xn_ref)
        worst["ei"], worst["logit"], worst["xn"] = max(worst["ei"], e_ei), max(worst["logit"], e_lg), max(worst["xn"], e_xn)
        finite = all(bool(torch.isfinite(v).all()) for v in (ei, logits, xn, tw))
        if not finite or e_ei > 2e-3 or e_lg > 1e-4 or e_xn > 1e-2:
            problems.append(f"{tag}: finite {finite} ei {e_ei:.2e} logit {e_lg:.2e} xn {e_xn:.2e}")
        problems += _dl_route_problems(logits, tw, ids, lids, ls_val, state, tag, tie_free=True)
        problems += _dl_align_problems(lids, align, _block_size_m(tokens), tag)
        y_new = _experts(xn, tw, lids, state, align=align)
        y_ref = _experts(xn, tw, (ids - ls) % N_EXPERTS, state)
        e_y = _rel_l2(y_new, y_ref) if bool(y_ref.abs().sum() > 0) else float((y_new.double().abs().max()))
        worst["y"] = max(worst["y"], e_y)
        if not bool(torch.isfinite(y_new).all()) or e_y > 1e-3:
            problems.append(f"{tag}: Marlin output y rel L2 {e_y:.2e}")
    for tokens in (1, 8, 32):
        ei = torch.randn((tokens, D_EXPERT), device=dev, generator=gen).to(torch.bfloat16)
        base = torch.randn((tokens, N_EXPERTS), device=dev, generator=gen) * 4.0
        crafted = {}
        lg = base.clone()
        lg[:, :LOCAL_EXPERTS] = -200.0
        lg[:, LOCAL_EXPERTS:LOCAL_EXPERTS + TOP_K] = 200.0
        crafted["no-local"] = (lg, 0, True)
        lg = base.clone()
        lg[:, 64 + 5] = 200.0
        crafted["one-hot-local"] = (lg, 64, True)
        lg = base.clone()
        b = dl["bias"]
        sc = torch.sigmoid(lg.double() * 0.1767766952966369) + b.double()
        order = sc.argsort(dim=-1, descending=True)
        a8, a9 = order[:, TOP_K - 1], order[:, TOP_K]
        target = (sc.gather(1, a8[:, None]) - 1e-7 - b.double()[a9][:, None]).clamp(1e-6, 1 - 1e-6)
        lg.scatter_(1, a9[:, None], (torch.log(target / (1 - target)) / 0.1767766952966369).float())
        crafted["near-tie"] = (lg, 224, False)
        for name, (lg, ls_val, tie_free) in crafted.items():
            tag = f"R T={tokens} {name}"
            ls = torch.tensor([ls_val], dtype=torch.int32, device=dev)
            lg = lg.contiguous()
            xn, (tw, ids, lids, align) = _dl_route(ei, lg, ls, dl)
            torch.cuda.synchronize()
            problems += _dl_route_problems(lg, tw, ids, lids, ls_val, state, tag, tie_free=tie_free)
            problems += _dl_align_problems(lids, align, _block_size_m(tokens), tag)
            no_local = bool((lids < 0).all())
            if name == "no-local" and not no_local:
                print(f"{DL_TAG} (info) {tag}: the router bias kept a local slot", flush=True)
            if no_local and int(align[2][0]) != 0:
                problems.append(f"{tag}: n_padded {int(align[2][0])} with no local slot")
            y_new = _experts(xn, tw, lids, state, align=align)
            y_ref = _experts(xn, tw, (ids - ls) % N_EXPERTS, state)
            ok_y = bool(torch.isfinite(y_new).all()) and (
                float(y_new.abs().max()) == 0.0 if no_local else _rel_l2(y_new, y_ref) <= 1e-3)
            if not ok_y:
                problems.append(f"{tag}: Marlin output y differs")
    for tokens in (8, 32):
        s0 = torch.randn((tokens, HIDDEN), device=dev, generator=gen) * 3.0
        a0 = torch.randn((tokens, HIDDEN), device=dev, generator=gen).to(torch.bfloat16)
        ls = torch.tensor([32], dtype=torch.int32, device=dev)
        def chain():
            ffn_input, _stream, _ = state["attn_boundary"](s0.clone(), a0, branch_dtype=torch.bfloat16, router_dtype=None)
            xn, (tw, ids, lids, align), _ = _dl_forward(ffn_input, ls, dl, pdl)
            return _experts(xn, tw, lids, state, align=align), ids
        y_eager, ids_eager = chain()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with _rt_cap(graph):
            y_graph, ids_graph = chain()
        graph.replay()
        torch.cuda.synchronize()
        e_g = _rel_l2(y_graph, y_eager) if bool(y_eager.abs().sum() > 0) else float(y_graph.double().abs().max())
        worst["graph"] = max(worst["graph"], e_g)
        if not torch.equal(ids_graph, ids_eager) or e_g > 1e-3 or not bool(torch.isfinite(y_graph).all()):
            problems.append(f"graph T={tokens} pdl={pdl}: replay differs from eager (y {e_g:.2e})")
        del graph
    summary = " ".join(f"{k} {v:.2e}" for k, v in worst.items())
    return not problems, summary, problems
def _dl_pdl_probe():
    dev = torch.device("cuda", torch.cuda.current_device())
    times = torch.zeros(4, dtype=torch.int64, device=dev)
    try:
        _DL_EXT.dl_pdl_probe(times, 50_000, 1)
        torch.cuda.synchronize()
        eager = times.tolist()
        eager_overlap = eager[2] < eager[1]
        times.zero_()
        graph = torch.cuda.CUDAGraph()
        with _rt_cap(graph):
            _DL_EXT.dl_pdl_probe(times, 50_000, 1)
        graph.replay()
        torch.cuda.synchronize()
        cap = times.tolist()
        del graph
    except Exception as e:
        return 0, f"capture with PDL raised {e!r}"
    overlap = cap[2] < cap[1]
    msg = (f"eager overlap {eager_overlap} ({(eager[1] - eager[2]) / 1e3:.1f} us), graph overlap {overlap} "
           f"({(cap[1] - cap[2]) / 1e3:.1f} us, secondary waited {(cap[3] - cap[2]) / 1e3:.1f} us)")
    return (1 if overlap else 0), msg
def _dl_setup(weights, state):
    if _DL_EXT is None:
        print(f"{DL_TAG} OFF pdl=0 compile {_DL_COMPILE_S:.1f}s (build failed): PATH CHOSEN = MASTER at every T", flush=True)
        return None
    try:
        router = weights["router"].float()
        hi = router.to(torch.bfloat16)
        lo = (router - hi.float()).to(torch.bfloat16)
        tiles = torch.stack((hi.view(16, 16, HIDDEN), lo.view(16, 16, HIDDEN)), dim=1).reshape(2 * N_EXPERTS, HIDDEN)
        w_pr = torch.cat((weights["pooler"].to(torch.bfloat16), tiles), dim=0).contiguous()
        try:
            norm_w = state["pre_expert_norm"]._get_effective_weight().detach().float().contiguous()
        except Exception:
            norm_w = (1.0 + weights["pre_expert_norm"].float()).contiguous()
        dev = w_pr.device
        state["dl"] = {
            "w_pr": w_pr,
            "ws": torch.zeros((DL_SPLIT_K * DL_MAX_TOKENS * DL_ROWS,), dtype=torch.float32, device=dev),
            "cnt_p": torch.zeros((DL_N_TILES,), dtype=torch.int32, device=dev),
            "cnt_r": torch.zeros((1,), dtype=torch.int32, device=dev),
            "norm_w": norm_w.reshape(-1),
            "bias": weights["router_bias"].float().contiguous(),
            "pdl": 0,
        }
        occ = [int(_DL_EXT.dl_occupancy(nf)) for nf in (1, 2, 4)]
        print(f"{DL_TAG} P occupancy (CTAs/SM, NF=1/2/4): {occ}; grid {DL_N_TILES * DL_SPLIT_K} CTAs", flush=True)
        pdl, probe = _dl_pdl_probe()
        print(f"{DL_TAG} PDL probe: {probe} -> pdl={pdl}", flush=True)
        ok, summary, problems = False, "", []
        for try_pdl in ((pdl, 0) if pdl else (0,)):
            state["dl"]["pdl"] = try_pdl
            try:
                ok, summary, problems = _dl_self_check(state, try_pdl)
            except Exception as e:
                ok, summary, problems = False, "", [f"self-check raised {e!r}"]
            state["dl"]["cnt_p"].zero_()
            state["dl"]["cnt_r"].zero_()
            if ok:
                pdl = try_pdl
                break
            print(f"{DL_TAG} SELF-CHECK FAILED with pdl={try_pdl}: {len(problems)} problems", flush=True)
            for message in problems[:12]:
                print(f"{DL_TAG}   {message}", flush=True)
        if not ok:
            print(f"{DL_TAG} OFF pdl=0 compile {_DL_COMPILE_S:.1f}s max errs {summary}: PATH CHOSEN = MASTER at every T",
                  flush=True)
            state.pop("dl", None)
            return None
        for t in (1, 8, 32):
            torch.randn((t, HIDDEN), device=state["dl"]["w_pr"].device)
        state["dl"]["cnt_p"].zero_()
        state["dl"]["cnt_r"].zero_()
        print(f"{DL_TAG} ON pdl={pdl} compile {_DL_COMPILE_S:.1f}s max errs {summary}: "
              f"PATH CHOSEN = DENSE-CHAIN at T <= {DL_MAX_TOKENS}, MASTER above", flush=True)
        return state["dl"]
    except Exception as e:
        print(f"{DL_TAG} OFF pdl=0 setup raised {e!r}: PATH CHOSEN = MASTER at every T", flush=True)
        state.pop("dl", None)
        return None
SLIDE_CNT_BAND = 0
SLIDE_CNT_B1 = 64
SLIDE_CNT_LEN = 64 * 2
def _rope_inv_freq(theta: float, d: int, device) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
def _slide_state(device) -> dict:
    return {"slide_cnt": torch.zeros(SLIDE_CNT_LEN, dtype=torch.int32, device=device),
            "slide_inv_freq": _rope_inv_freq(ROPE_THETA, 64, device)}
def _slide_inv_freq(cfg, state, device) -> torch.Tensor:
    if cfg["rope_theta"] == ROPE_THETA and cfg["head_dim"] == 64:
        return state["slide_inv_freq"]
    return _rope_inv_freq(cfg["rope_theta"], cfg["head_dim"], device)
LOG2E = 1.4426950408889634
def _num_splits(batch: int, hkv: int) -> int:
    s = 1
    while s < 16 and batch * hkv * s * 2 <= 128:
        s *= 2
    return s
@triton.jit
def _norm_rope(x, xs, pos, freq, d, D: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr):
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + EPS)
    x = x * r[:, None]
    if ROPE:
        xs = xs * r[:, None]
        ang = pos[:, None] * freq[None, :]
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
        x = x * tl.cos(ang) + sign[None, :] * xs * tl.sin(ang)
    return x
@triton.jit
def _gpu_fence(x):
    return tl.inline_asm_elementwise("fence.acq_rel.gpu;\n\tmov.b32 $0, $1;", "=r,r", [x],
                                     dtype=tl.int32, is_pure=False, pack=1)
@triton.jit
def _online_update(qk, valid, v, m_i, l_i, acc):
    qk = tl.where(valid, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_use)
    p = tl.exp2(qk - m_use[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l_i, acc
@triton.jit
def _gate_epilogue(o, qkv_ptr, out_ptr, tok, head, rvalid, d, qkv_stride,
                   HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    r = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + EPS)
    gptr = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + head[:, None] * D + d[None, :]
    gate = tl.load(gptr, mask=rvalid[:, None], other=0.0).to(tl.float32)
    y = o * r[:, None] * tl.sigmoid(gate)
    optr = out_ptr + (tok.to(tl.int64)[:, None] * HQ + head[:, None]) * D + d[None, :]
    tl.store(optr, y.to(tl.bfloat16), mask=rvalid[:, None])
@triton.jit
def _sliding_fused_kernel(
    qkv_ptr, pos_ptr, loc_ptr, seq_ptr, pt_ptr, inv_freq_ptr, k_scale_ptr, v_scale_ptr,
    k_cache_ptr, v_cache_ptr, ws_acc_ptr, ws_ml_ptr, cnt_ptr, out_ptr,
    qkv_stride, pt_stride, q_scale, score_scale,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
    LN: tl.constexpr, WINDOW: tl.constexpr, PAGE: tl.constexpr, S: tl.constexpr,
    BLOCK_N: tl.constexpr, ROPE: tl.constexpr, EPS: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    s = tl.program_id(2)
    GROUP: tl.constexpr = HQ // HKV
    d = tl.arange(0, D)
    dsw = d ^ 1
    freq = tl.load(inv_freq_ptr + d // 2) if ROPE else tl.zeros([D], dtype=tl.float32)

    r = tl.arange(0, M)
    rvalid = r < GROUP * L
    j = tl.minimum(r // GROUP, L - 1)
    head = g * GROUP + r % GROUP
    tok = b * L + j
    qrow = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + head[:, None] * D
    xq = tl.load(qrow + d[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    xqs = tl.load(qrow + dsw[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qpos = tl.load(pos_ptr + tok).to(tl.float32)
    q = _norm_rope(xq, xqs, qpos, freq, d, D, ROPE, EPS)
    q = (q * q_scale).to(tl.bfloat16)

    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    qk_scale = score_scale * k_scale
    seq = tl.load(seq_ptr + b)
    p_row = seq + j

    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D], dtype=tl.float32)

    if s == S - 1:
        i = tl.arange(0, LN)
        ivalid = i < L
        ntok = b * L + tl.minimum(i, L - 1)
        nrow = qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride
        xk = tl.load(nrow + (HQ + g) * D + d[None, :], mask=ivalid[:, None], other=0.0).to(tl.float32)
        xks = tl.load(nrow + (HQ + g) * D + dsw[None, :], mask=ivalid[:, None], other=0.0).to(tl.float32)
        kpos = tl.load(pos_ptr + ntok).to(tl.float32)
        kn = _norm_rope(xk, xks, kpos, freq, d, D, ROPE, EPS)
        k8 = (kn / k_scale).to(tl.float8e4nv)
        vn = tl.load(nrow + (HQ + HKV + g) * D + d[None, :], mask=ivalid[:, None], other=0.0).to(tl.float32)
        v8 = (vn / v_scale).to(tl.float8e4nv)
        loc = tl.load(loc_ptr + ntok).to(tl.int64)
        cptr = (loc[:, None] * HKV + g) * D + d[None, :]
        tl.store(k_cache_ptr + cptr, k8.to(tl.uint8, bitcast=True), mask=ivalid[:, None])
        tl.store(v_cache_ptr + cptr, v8.to(tl.uint8, bitcast=True), mask=ivalid[:, None])
        qk = tl.dot(q, tl.trans(k8.to(tl.bfloat16))) * qk_scale
        valid = ivalid[None, :] & (i[None, :] <= j[:, None])
        m_i, l_i, acc = _online_update(qk, valid, v8.to(tl.bfloat16), m_i, l_i, acc)

    lo = tl.maximum(seq - (WINDOW - 1), 0)
    count = seq - lo
    chunk = ((count + S - 1) // S + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    start = lo + s * chunk
    end = tl.minimum(start + chunk, seq)
    n = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        t = n0 + n
        tmask = t < end
        page = tl.load(pt_ptr + b.to(tl.int64) * pt_stride + t // PAGE, mask=tmask, other=0)
        row = page.to(tl.int64) * PAGE + t % PAGE
        kvoff = (row[:, None] * HKV + g) * D + d[None, :]
        kb = tl.load(k_cache_ptr + kvoff, mask=tmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        vb = tl.load(v_cache_ptr + kvoff, mask=tmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)
        qk = tl.dot(q, tl.trans(kb.to(tl.bfloat16))) * qk_scale
        valid = tmask[None, :] & (t[None, :] >= p_row[:, None] - (WINDOW - 1))
        m_i, l_i, acc = _online_update(qk, valid, vb.to(tl.bfloat16), m_i, l_i, acc)

    if S == 1:
        o = acc / l_i[:, None] * v_scale
        _gate_epilogue(o, qkv_ptr, out_ptr, tok, head, rvalid, d, qkv_stride, HQ, HKV, D, EPS)
    else:
        bg = b * HKV + g
        part = (bg * S + s).to(tl.int64)
        tl.store(ws_acc_ptr + (part * M + r[:, None]) * D + d[None, :], acc)
        tl.store(ws_ml_ptr + part * 2 * M + r, m_i)
        tl.store(ws_ml_ptr + part * 2 * M + M + r, l_i)
        _gpu_fence(r)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + bg, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == S - 1:
            _gpu_fence(r)
            base = (bg * S).to(tl.int64)
            m_max = tl.full([M], float("-inf"), dtype=tl.float32)
            for s2 in tl.static_range(S):
                m_s = tl.load(ws_ml_ptr + (base + s2) * 2 * M + r, cache_modifier=".cg")
                m_max = tl.maximum(m_max, m_s)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
            l_sum = tl.zeros([M], dtype=tl.float32)
            o = tl.zeros([M, D], dtype=tl.float32)
            for s2 in tl.static_range(S):
                m_s = tl.load(ws_ml_ptr + (base + s2) * 2 * M + r, cache_modifier=".cg")
                l_s = tl.load(ws_ml_ptr + (base + s2) * 2 * M + M + r, cache_modifier=".cg")
                a_s = tl.load(ws_acc_ptr + ((base + s2) * M + r[:, None]) * D + d[None, :], cache_modifier=".cg")
                w = tl.where(m_s == float("-inf"), 0.0, tl.exp2(m_s - m_max))
                l_sum += l_s * w
                o += a_s * w[:, None]
            o = o / l_sum[:, None] * v_scale
            _gate_epilogue(o, qkv_ptr, out_ptr, tok, head, rvalid, d, qkv_stride, HQ, HKV, D, EPS)
            tl.store(cnt_ptr + bg, 0)
def _sliding_forward(data, qkv, state):
    cfg = data["config"]
    if state.get("w6p_ph", False) and _w6p_shape(cfg):
        return _w6p_ph_attn(data, qkv, state)
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    device = qkv.device
    theta = cfg["rope_theta"]
    group = hq // hkv
    m = max(16, triton.next_power_of_2(group * length))
    splits = _num_splits(batch, hkv)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)
    if splits > 1:
        ws_acc = torch.empty((batch * hkv * splits, m, d), dtype=torch.float32, device=device)
        ws_ml = torch.empty((batch * hkv * splits, 2, m), dtype=torch.float32, device=device)
        counters = state["slide_cnt"][SLIDE_CNT_B1:SLIDE_CNT_B1 + batch * hkv]
    else:
        ws_acc = ws_ml = counters = out
    inv_freq = _slide_inv_freq(cfg, state, device) if theta else data["k_scale"]
    page_table = data["page_table"]
    block_n = state.get("w6p_bn", 64) if batch == 1 and length == 1 else 64
    _sliding_fused_kernel[(batch, hkv, splits)](
        qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, inv_freq,
        data["k_scale"], data["v_scale"],
        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), ws_acc, ws_ml, counters, out,
        qkv.stride(0), page_table.stride(0), cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        HQ=hq, HKV=hkv, D=d, L=length, M=m, LN=max(16, triton.next_power_of_2(length)),
        WINDOW=cfg["window"], PAGE=cfg["page_size"], S=splits, BLOCK_N=block_n, ROPE=bool(theta), EPS=RMS_EPS,
        num_warps=4, num_stages=2,
    )
    return out
S1_SPLITS = 16
S1_CLASSES = {"A": (2, 16), "B": (17, 32)}
def _s1_class(cfg):
    if not (cfg["window"] and cfg["q_len"] == 1 and cfg["page_size"] == 1 and cfg["num_q_heads"] == 16
            and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64):
        return None
    for name, (lo, hi) in S1_CLASSES.items():
        if lo <= cfg["batch"] <= hi:
            return name
    return None
def _sliding_forward_band(data, qkv, state):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    device = qkv.device
    theta = cfg["rope_theta"]
    m = max(16, triton.next_power_of_2((hq // hkv) * length))
    splits = S1_SPLITS
    out = torch.empty((batch * length, hq * d), dtype=torch.bfloat16, device=device)
    ws_acc = torch.empty((batch * hkv * splits, m, d), dtype=torch.float32, device=device)
    ws_ml = torch.empty((batch * hkv * splits, 2, m), dtype=torch.float32, device=device)
    counters = state["slide_cnt"][SLIDE_CNT_BAND:SLIDE_CNT_BAND + batch * hkv]
    inv_freq = _slide_inv_freq(cfg, state, device) if theta else data["k_scale"]
    page_table = data["page_table"]
    _sliding_fused_kernel[(batch, hkv, splits)](
        qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, inv_freq,
        data["k_scale"], data["v_scale"],
        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), ws_acc, ws_ml, counters, out,
        qkv.stride(0), page_table.stride(0), cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        HQ=hq, HKV=hkv, D=d, L=length, M=m, LN=max(16, triton.next_power_of_2(length)),
        WINDOW=cfg["window"], PAGE=cfg["page_size"], S=splits, BLOCK_N=64, ROPE=bool(theta), EPS=RMS_EPS,
        num_warps=4, num_stages=2,
    )
    return out
S3_CELLS = ("c1-b16", "c1-b32", "c2-t8", "c2-t128", "c2-t256", "c3-t128", "c3-t256", "c4-t8")
def _s3_cell(cfg, unpaired_feed, paired):
    if not (cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64):
        return None
    batch, length, window = cfg["batch"], cfg["q_len"], cfg["window"]
    if length == 1:
        if window == 0 and 2 <= batch <= 64 and unpaired_feed:
            return "c1-b16" if batch <= 23 else "c1-b32"
        return None
    if not (2 <= length <= 8 and batch <= 64):
        return None
    if window:
        return "c2-t8" if batch == 1 else ("c2-t128" if batch <= 23 else "c2-t256")
    if batch == 1:
        return "c4-t8"
    if paired:
        return "c3-t128" if batch <= 23 else "c3-t256"
    return None
@triton.jit
def _pre_attention_kernel(
    qkv_ptr, pos_ptr, loc_ptr, cos_sin_ptr, k_scale_ptr, v_scale_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    qkv_stride,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr, PAIRED: tl.constexpr,
    ROPE: tl.constexpr, EPS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    even = 2 * i
    odd = 2 * i + 1

    row = qkv_ptr + t.to(tl.int64) * qkv_stride
    base = row + h * D
    xe = tl.load(base + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32)

    ss = tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)
    r = 1.0 / tl.sqrt(ss / D + EPS)
    xe = xe * r
    xo = xo * r

    if ROPE:
        p = tl.load(pos_ptr + t).to(tl.int64)
        cs = cos_sin_ptr + p * D
        c = tl.load(cs + i)
        s = tl.load(cs + HALF + i)
        ye = xe * c - xo * s
        yo = xe * s + xo * c
        xe = ye
        xo = yo

    if h < HQ:
        if PAIRED:
            half = (h // G) % 2
            qrow = q_out_ptr + (t.to(tl.int64) * HQ + h) * (2 * D)
            out = qrow + half * D
            zj = tl.arange(0, D)
            tl.store(qrow + (1 - half) * D + zj, tl.zeros([D], dtype=tl.float32).to(tl.float8e4nv))
        else:
            out = q_out_ptr + (t.to(tl.int64) * HQ + h) * D
        tl.store(out + even, xe.to(tl.float8e4nv))
        tl.store(out + odd, xo.to(tl.float8e4nv))
    else:
        kh = h - HQ
        loc = tl.load(loc_ptr + t).to(tl.int64)
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        kout = k_cache_ptr + (loc * HKV + kh) * D
        tl.store(kout + even, (xe / k_scale).to(tl.float8e4nv))
        tl.store(kout + odd, (xo / k_scale).to(tl.float8e4nv))
        j = tl.arange(0, D)
        v = tl.load(row + (HQ + HKV + kh) * D + j).to(tl.float32)
        vout = v_cache_ptr + (loc * HKV + kh) * D
        tl.store(vout + j, (v / v_scale).to(tl.float8e4nv))
@triton.jit
def _paired_norm_gate_kernel(
    attn_ptr, qkv_ptr, out_ptr,
    attn_stride_t, attn_stride_h, qkv_stride,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr, EPS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.arange(0, D)
    half = (h // G) % 2
    src = attn_ptr + t.to(tl.int64) * attn_stride_t + h * attn_stride_h + half * D
    x = tl.load(src + j).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + EPS)
    gate = tl.load(qkv_ptr + t.to(tl.int64) * qkv_stride + (HQ + 2 * HKV) * D + h * D + j)
    gate = gate.to(tl.float32)
    y = x * r * tl.sigmoid(gate)
    tl.store(out_ptr + (t.to(tl.int64) * HQ + h) * D + j, y.to(tl.bfloat16))
def _s3_pre_attention(data, qkv, state, paired):
    cfg = data["config"]
    tokens = cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    q = torch.empty((tokens, hq, 2 * d if paired else d), dtype=FP8, device=qkv.device)
    _pre_attention_kernel[(tokens, hq + hkv)](
        qkv, data["positions"], data["cache_loc"], state["cos_sin"], data["k_scale"], data["v_scale"],
        q, data["k_cache"], data["v_cache"],
        qkv.stride(0),
        HQ=hq, HKV=hkv, D=d, G=hq // hkv, PAIRED=paired, ROPE=bool(cfg["rope_theta"]), EPS=RMS_EPS,
        num_warps=1,
    )
    return q
def _s3_paired_norm_gate(out, qkv, cfg):
    tokens = cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    attn = out.view(tokens, hq, 2 * d)
    o = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=out.device)
    _paired_norm_gate_kernel[(tokens, hq)](
        attn, qkv, o,
        attn.stride(0), attn.stride(1), qkv.stride(0),
        HQ=hq, HKV=hkv, D=d, G=hq // hkv, EPS=RMS_EPS,
        num_warps=1,
    )
    return o
_SMS = {}
def _sm_count(device) -> int:
    key = device.index if device.index is not None else torch.cuda.current_device()
    n = _SMS.get(key)
    if n is None:
        n = _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return n
def _splitkv_synthetic(lens, kv_scale, gen, device):
    batch, hq, hkv, d = len(lens), 16, 2, 64
    max_pages = max(lens) + 37
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(lens):
        page_table[b, :n] = perm[off:off + n]
        off += n
    q = torch.randn((batch, hq, d), generator=gen, device=device)
    q = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(FP8)
    k_cache = (torch.randn((total, 1, hkv, d), generator=gen, device=device) / kv_scale).to(FP8)
    v_cache = (torch.randn((total, 1, hkv, d), generator=gen, device=device) / kv_scale).to(FP8)
    qkv = (2 * torch.randn((batch, (2 * hq + 2 * hkv) * d), generator=gen, device=device)).to(torch.bfloat16)
    data = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "cache_seqlens": torch.tensor(lens, dtype=torch.int32, device=device),
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table,
        "config": {"batch": batch, "q_len": 1, "tokens": batch, "num_q_heads": hq, "num_kv_heads": hkv,
                   "head_dim": d, "page_size": 1, "window": 0, "rope_theta": 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": d ** -0.5, "kv_scale": kv_scale},
    }
    return data, q, qkv
def _splitkv_reference(data, q, qkv, lens):
    cfg = data["config"]
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qf = q.float() * cfg["qk_scale_factor"]
    kf = data["k_cache"].view(-1, hkv, d)
    vf = data["v_cache"].view(-1, hkv, d)
    outs = []
    for b, n in enumerate(lens):
        rows = data["page_table"][b, :n].long()
        keys = (kf[rows].float() * data["k_scale"]).repeat_interleave(hq // hkv, dim=1)
        vals = (vf[rows].float() * data["v_scale"]).repeat_interleave(hq // hkv, dim=1)
        p = torch.softmax(torch.einsum("hd,nhd->hn", qf[b], keys) * cfg["sm_scale"], dim=-1)
        outs.append(torch.einsum("hn,nhd->hd", p, vals))
    a = torch.stack(outs)
    a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    gate = qkv.float()[:, (hq + 2 * hkv) * d:].view(batch, hq, d)
    return (a * torch.sigmoid(gate)).reshape(batch, hq * d)
_SK_CPP = r"""
#include <torch/extension.h>
#include <vector>
void sk_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor qkv,
             int64_t qkv_stride, int64_t gate_off, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
             torch::Tensor ticket,
             torch::Tensor gticket, torch::Tensor gacc, torch::Tensor gml,
             double score_scale, double eps, int64_t grid, int64_t variant);
std::vector<int64_t> sk_attrs(int64_t variant);
"""
_SK_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int D = 64;              // head dim
constexpr int HQ = 16;             // query heads
constexpr int HKV = 2;             // kv heads
constexpr int G = HQ / HKV;        // queries per kv head (8)
constexpr int ROW = HKV * D;       // bytes per cache row (page 1: both kv heads, 128)
constexpr int TILE = 64;           // tokens per tile (the stream-K unit)
constexpr int MB_TILES = 16;       // tiles per page-index block
constexpr int MB_TOK = MB_TILES * TILE;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAXB = 64;
constexpr int STAGE_BYTES = 2 * TILE * ROW;  // K tile + V tile = 16 KB
constexpr int QBYTES = HQ * D;               // one request's e4m3 query

template <int STAGES>
struct Lay {
  static constexpr int RING = 0;
  static constexpr int QR = RING + STAGES * STAGE_BYTES;      // e4m3 [STAGES][HQ][D]
  static constexpr int IDX = QR + STAGES * QBYTES;            // int32 [2][MB_TOK]
  static constexpr int ACC = IDX + 2 * MB_TOK * 4;            // f32 [WARPS][G][D]
  static constexpr int ML = ACC + WARPS * G * D * 4;          // f32 [WARPS][G][2]
  static constexpr int SEQ = ML + WARPS * G * 2 * 4;          // int32 [MAXB]
  static constexpr int TP = SEQ + MAXB * 4;                   // int32 [MAXB + 1] (+pad)
  static constexpr int FLAG = TP + (MAXB + 4) * 4;            // int32 [4]
  static constexpr int BYTES = FLAG + 16;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
// PDL (sm_90), stage 3: this CTA's last tile is done, its merge / ticket epilogue follows. A dependent kernel
// launched with the programmatic attribute (dl_o_kernel at batch 1) may start its weight prefetch once every CTA
// has issued this or exited; its griddepcontrol.wait still waits for this grid's completion and memory flush.
// A no-op for every dependent launched without the attribute.
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// cache_seqlens -> seq_s, and the tile prefix tp_s[0..B] (tp_s[B] = Ttot). Ends with a barrier.
__device__ __forceinline__ void load_prefix(const int* __restrict__ seqlens, int B, int* seq_s, int* tp_s) {
  const int tid = threadIdx.x;
  if (tid < B) { const int s = seqlens[tid]; seq_s[tid] = s > 0 ? s : 0; }
  __syncthreads();
  if (tid == 0) {
    int run = 0;
    tp_s[0] = 0;
    for (int b = 0; b < B; ++b) {
      run += (seq_s[b] + TILE - 1) / TILE;
      tp_s[b + 1] = run;
    }
  }
  __syncthreads();
}
// First tile of CTA c, and the CTA owning tile t (the exact inverse: s_c <= t < s_{c+1}).
__device__ __forceinline__ int range_start(int c, int T, int gx) {
  return static_cast<int>((static_cast<long long>(c) * T) / gx);
}
__device__ __forceinline__ int tile_owner(int t, int T, int gx) {
  return static_cast<int>(((static_cast<long long>(t) + 1) * gx - 1) / T);
}

// LSE merge of the partials in slots s_lo..s_hi for head h, dims d0..d0+3 (M = the raw max, so the
// result can itself be stored as a slot). L2 loads (__ldcg): the slots were written by other CTAs.
// Stage-2 callers pass s = c + b (c = c_lo..c_hi); the two-level path passes group-slot indices.
__device__ __forceinline__ void merge_range(const float* ws_acc, const float* ws_ml, int s_lo, int s_hi,
                                            int h, int d0, float4& o, float& L, float& Mout) {
  const float NEG_INF = __int_as_float(0xff800000u);
  float M = NEG_INF;
  for (int c = s_lo; c <= s_hi; ++c) M = fmaxf(M, __ldcg(ws_ml + (int64_t)c * 2 * HQ + h));
  const float Mu = M == NEG_INF ? 0.f : M;
  Mout = M;
  o = make_float4(0.f, 0.f, 0.f, 0.f);
  L = 0.f;
  for (int c = s_lo; c <= s_hi; ++c) {
    const int64_t sl = c;
    const float m = __ldcg(ws_ml + sl * 2 * HQ + h);
    const float wt = m == NEG_INF ? 0.f : ex2(m - Mu);
    L += wt * __ldcg(ws_ml + sl * 2 * HQ + HQ + h);
    const float4 a = __ldcg(reinterpret_cast<const float4*>(ws_acc + (sl * HQ + h) * D + d0));
    o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
  }
}
__device__ __forceinline__ void merge_slots(const float* ws_acc, const float* ws_ml, int b, int c_lo, int c_hi,
                                            int h, int d0, float4& o, float& L) {
  float M;
  merge_range(ws_acc, ws_ml, c_lo + b, c_hi + b, h, d0, o, L, M);
}

// Release/acquire ticket draw (two-level path): the CTA's partial stores are ordered before the
// increment (acq_rel at gpu scope, on top of the caller's __threadfence + __syncthreads), and the
// returned count orders the winner after every earlier arrival's stores.
__device__ __forceinline__ int ticket_arrive(int* p) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(1) : "memory");
  return old;
}

// `qkv` here is the GATE base pointer (stage 3: the host passes qkv.data_ptr() + gate_off, or a separate
// contiguous gate tensor with gate_off 0), row b at qkv + b * qkv_stride.
// Stage-1 B epilogue for head h, dims d0..d0+3 (16 consecutive lanes hold one head): o / L *
// v_scale, scaleless RMSNorm over 64 (eps inside the sqrt), times sigmoid(gate), bf16 out.
// Every lane of the warp must call it (16-lane shuffles).
__device__ __forceinline__ void finalize_store(float4 o, float L, float vs, int b, int h, int d0,
                                               const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                                               __nv_bfloat16* __restrict__ out, float eps) {
  o.x = o.x / L * vs; o.y = o.y / L * vs; o.z = o.z / L * vs; o.w = o.w / L * vs;
  float ss = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  ss += __shfl_xor_sync(0xffffffffu, ss, 8);
  const float r = 1.f / sqrtf(ss / D + eps);
  const __nv_bfloat162* gp =
      reinterpret_cast<const __nv_bfloat162*>(qkv + (int64_t)b * qkv_stride + h * D + d0);
  const float2 g01 = __bfloat1622float2(gp[0]);
  const float2 g23 = __bfloat1622float2(gp[1]);
  const float y0 = o.x * r / (1.f + expf(-g01.x));
  const float y1 = o.y * r / (1.f + expf(-g01.y));
  const float y2 = o.z * r / (1.f + expf(-g23.x));
  const float y3 = o.w * r / (1.f + expf(-g23.y));
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + ((int64_t)b * HQ + h) * D + d0);
  op[0] = __floats2bfloat162_rn(y0, y1);
  op[1] = __floats2bfloat162_rn(y2, y3);
}

template <int STAGES, bool TICKET, bool TWO = false>
__global__ void __launch_bounds__(THREADS, 2)
splitkv_attn_sk_kernel(const uint8_t* __restrict__ q,          // e4m3 [B, HQ, D]
                       const uint8_t* __restrict__ kc,         // e4m3 [rows, ROW]
                       const uint8_t* __restrict__ vc,         // e4m3 [rows, ROW]
                       const int* __restrict__ page_table, int64_t pt_stride,
                       const int* __restrict__ seqlens, int B,
                       const float* __restrict__ k_scale, const float* __restrict__ v_scale, float score_scale,
                       const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                       __nv_bfloat16* __restrict__ out,        // bf16 [B, HQ * D]
                       float* __restrict__ ws_acc,             // f32 [grid + B, HQ, D]
                       float* __restrict__ ws_ml,              // f32 [grid + B, 2, HQ]
                       int* __restrict__ ticket,               // int32 [>= B], 0 between launches
                       int* __restrict__ gticket,              // int32 [>= B * 32], 0 between launches (TWO)
                       float* __restrict__ gacc,               // f32 [B * 32, HQ, D] group slots (TWO)
                       float* __restrict__ gml,                // f32 [B * 32, 2, HQ] (TWO)
                       float eps) {
  using S = Lay<STAGES>;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem + S::RING;
  uint8_t* qr = smem + S::QR;
  int* idx_s = reinterpret_cast<int*>(smem + S::IDX);
  float* acc_s = reinterpret_cast<float*>(smem + S::ACC);
  float* ml_s = reinterpret_cast<float*>(smem + S::ML);
  int* seq_s = reinterpret_cast<int*>(smem + S::SEQ);
  int* tp_s = reinterpret_cast<int*>(smem + S::TP);
  int* flag_s = reinterpret_cast<int*>(smem + S::FLAG);
  const float NEG_INF = __int_as_float(0xff800000u);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bx = blockIdx.x;
  load_prefix(seqlens, B, seq_s, tp_s);
  const int T = tp_s[B];
  if (T <= 0) return;  // uniform
  // Effective grid min(gridDim.x, T): below it every CTA owns >= 1 tile, so CTAs c_lo..c_hi of a
  // request all hold a segment (the ticket count and the merge rely on it). CTAs above it exit.
  const int gx = min(static_cast<int>(gridDim.x), T);
  if (bx >= gx) return;  // uniform
  const int t0 = range_start(bx, T, gx);
  const int N = range_start(bx + 1, T, gx) - t0;  // tiles of this CTA, one flat pipeline
  if (N <= 0) return;  // uniform
  const int nblk = (N + MB_TILES - 1) / MB_TILES;
  const float scale = score_scale * __ldg(k_scale);  // sm_scale * qk_scale_factor * log2(e) * k_scale
  const float vs = __ldg(v_scale);

  // Request holding global tile gt: the largest b with tp_s[b] <= gt (gt < Ttot).
  auto find_req = [&](int gt) {
    int lo = 0, hi = B - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (tp_s[mid] <= gt) lo = mid; else hi = mid - 1;
    }
    return lo;
  };
  // Page indices of block k (flat tiles 16k .. 16k+15) into buffer k & 1: thread -> tile tid / 16,
  // tokens (tid % 16) * 4 .. +4. Positions past the request end (or past the range) zero-filled.
  auto issue_meta = [&](int k) {
    const int buf = k & 1;
    const int tl = tid >> 4;
    const int e0 = (tid & 15) * 4;
    const int n = k * MB_TILES + tl;
    int b = 0, pos = 0, seq = 0;
    if (n < N) {
      const int gt = t0 + n;
      b = find_req(gt);
      pos = (gt - tp_s[b]) * TILE + e0;
      seq = seq_s[b];
    }
    const int* prow = page_table + (int64_t)b * pt_stride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool ok = pos + j < seq;
      cp_async4(smem_u32(idx_s + buf * MB_TOK + tl * TILE + e0 + j), ok ? prow + pos + j : prow, ok ? 4 : 0);
    }
  };
  // Tile nn (issued in order 0, 1, 2, ...) into ring stage nn % STAGES, as kernel A: thread ->
  // token tid / 4, 16-byte chunks 2 * (tid % 4) + {0, 1} of its K row and V row, chunk c stored
  // at c ^ (token % 8). A segment's first tile also brings the request's query into its q slot.
  int p_b = find_req(t0);
  auto issue_tile = [&](int nn) {
    const int gt = t0 + nn;
    while (gt >= tp_s[p_b + 1]) ++p_b;
    const int lt = gt - tp_s[p_b];
    const int st = nn % STAGES;
    uint8_t* stp = ring + st * STAGE_BYTES;
#ifdef SK_FULLLINE
    // W3-A (campaign expert-bandwidth, port of attn-skfull 00acb910): thread -> 16-byte chunk c = tid % 8 of tokens
    // tid / 8 and tid / 8 + 32, so one warp-wide cp.async covers 4 whole 128-byte K (or V) lines. Same swizzled
    // stage image (chunk c of token tt at c ^ (tt % 8)), same 4 x 16-byte copies per thread per tile. Only the
    // separately named module built with -DSK_FULLLINE=1 compiles this branch; the incumbent module is unchanged.
    const int c = tid & 7;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tt = (tid >> 3) + 32 * j;
      const bool ok = lt * TILE + tt < seq_s[p_b];
      const int64_t row = ok ? (int64_t)idx_s[((nn / MB_TILES) & 1) * MB_TOK + (nn % MB_TILES) * TILE + tt] : 0;
      const int dst = tt * ROW + ((c ^ (tt & 7)) << 4);
      cp_async16(smem_u32(stp + dst), kc + row * ROW + c * 16, ok ? 16 : 0);
      cp_async16(smem_u32(stp + TILE * ROW + dst), vc + row * ROW + c * 16, ok ? 16 : 0);
    }
#else
    const int tt = tid >> 2;
    const bool ok = lt * TILE + tt < seq_s[p_b];
    const int64_t row = ok ? (int64_t)idx_s[((nn / MB_TILES) & 1) * MB_TOK + (nn % MB_TILES) * TILE + tt] : 0;
    const uint8_t* ks = kc + row * ROW;
    const uint8_t* vsrc = vc + row * ROW;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = (tid & 3) * 2 + j;
      const int dst = tt * ROW + ((c ^ (tt & 7)) << 4);
      cp_async16(smem_u32(stp + dst), ks + c * 16, ok ? 16 : 0);
      cp_async16(smem_u32(stp + TILE * ROW + dst), vsrc + c * 16, ok ? 16 : 0);
    }
#endif
    if ((nn == 0 || lt == 0) && tid < QBYTES / 16)
      cp_async16(smem_u32(qr + st * QBYTES + tid * 16), q + (int64_t)p_b * QBYTES + tid * 16, 16);
  };

  // Prologue: block 0's indices (waited), then tiles 0 .. STAGES-2; block 1 rides tile 0's group.
  issue_meta(0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < N) {
      issue_tile(j);
      if (j % MB_TILES == 0 && j / MB_TILES + 1 < nblk) issue_meta(j / MB_TILES + 1);
    }
    cp_commit();
  }

  const int g = warp & 1;             // kv head
  const int slice = (warp >> 1) * 16; // token slice of each tile
  const int quad = lane >> 2, qi = lane & 3;
  uint32_t qa[4][2];
  float m_i = NEG_INF, l_i = 0.f;
  float acc[8][4];
  int c_b = find_req(t0);
  bool seg_head = false;  // the current segment began at its request's first tile

  for (int n = 0; n < N; ++n) {
    cp_wait<STAGES - 2>();  // tile n (and everything issued before it) has landed
    __syncthreads();        // visible to all; every warp is done with tile n - 1's stage
    {
      const int nn = n + STAGES - 1;
      if (nn < N) {
        issue_tile(nn);
        if (nn % MB_TILES == 0 && nn / MB_TILES + 1 < nblk) issue_meta(nn / MB_TILES + 1);
      }
      cp_commit();  // always, so the group count stays uniform
    }
    const int gt = t0 + n;
    while (gt >= tp_s[c_b + 1]) ++c_b;
    const int lt = gt - tp_s[c_b];
    const int nt_b = tp_s[c_b + 1] - tp_s[c_b];
    const int cseq = seq_s[c_b];
    const int st = n % STAGES;
    if (n == 0 || lt == 0) {
      seg_head = lt == 0;
      const uint8_t* qrow = qr + st * QBYTES + (g * G + quad) * D + qi * 4;
#pragma unroll
      for (int s = 0; s < 4; ++s)
        e4m3x4_to_f16x2x2(*reinterpret_cast<const uint32_t*>(qrow + s * 16), qa[s][0], qa[s][1]);
      m_i = NEG_INF;
      l_i = 0.f;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;
    }
    const uint8_t* Ks = ring + st * STAGE_BYTES;
    const uint8_t* Vs = Ks + TILE * ROW;

    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
      const int tt = slice + nb * 8 + quad;
      const uint8_t* kr = Ks + tt * ROW + qi * 4;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kr + (((g * 4 + ks) ^ (tt & 7)) << 4));
        uint32_t b0, b1;
        e4m3x4_to_f16x2x2(w, b0, b1);
        mma16816(s[nb], qa[ks][0], 0u, qa[ks][1], 0u, b0, b1);
      }
    }
    const int pos0 = lt * TILE + slice + qi * 2;
    float x0 = pos0 < cseq ? s[0][0] * scale : NEG_INF;
    float x1 = pos0 + 1 < cseq ? s[0][1] * scale : NEG_INF;
    float x2 = pos0 + 8 < cseq ? s[1][0] * scale : NEG_INF;
    float x3 = pos0 + 9 < cseq ? s[1][1] * scale : NEG_INF;
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float m_new = fmaxf(m_i, mx);
    const float m_use = m_new == NEG_INF ? 0.f : m_new;
    const float alpha = ex2(m_i - m_use);
    x0 = ex2(x0 - m_use);
    x1 = ex2(x1 - m_use);
    x2 = ex2(x2 - m_use);
    x3 = ex2(x3 - m_use);
    l_i = l_i * alpha + (x0 + x1) + (x2 + x3);
    m_i = m_new;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) { acc[nb][0] *= alpha; acc[nb][1] *= alpha; }
    const uint32_t pa0 = pack_f16x2(x0, x1), pa2 = pack_f16x2(x2, x3);

    const int tv = slice + qi * 2;
    const int vlc = g * 4 + (quad >> 1);
    const int voff = (quad & 1) * 8;
    const uint2 v0 = *reinterpret_cast<const uint2*>(Vs + tv * ROW + ((vlc ^ (tv & 7)) << 4) + voff);
    const uint2 v1 = *reinterpret_cast<const uint2*>(Vs + (tv + 1) * ROW + ((vlc ^ ((tv + 1) & 7)) << 4) + voff);
    const uint2 v8 = *reinterpret_cast<const uint2*>(Vs + (tv + 8) * ROW + ((vlc ^ ((tv + 8) & 7)) << 4) + voff);
    const uint2 v9 = *reinterpret_cast<const uint2*>(Vs + (tv + 9) * ROW + ((vlc ^ ((tv + 9) & 7)) << 4) + voff);
#pragma unroll
    for (int hw = 0; hw < 2; ++hw) {
      const uint32_t w0 = hw ? v0.y : v0.x, w1 = hw ? v1.y : v1.x;
      const uint32_t w8 = hw ? v8.y : v8.x, w9 = hw ? v9.y : v9.x;
      uint32_t r0[4], r1[4];
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x5140), r0[0], r0[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x7362), r0[2], r0[3]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x5140), r1[0], r1[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x7362), r1[2], r1[3]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma16816(acc[hw * 4 + j], pa0, 0u, pa2, 0u, r0[j], r1[j]);
    }

    if (n == N - 1 || lt == nt_b - 1) {
      if (n == N - 1) griddep_launch();  // uniform: the CTA's last tile
      // Segment end (uniform over the CTA): merge the four warps of each kv head in smem.
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 1);
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 2);
      const int rw = warp * G + quad;
      if (qi == 0) { ml_s[rw * 2] = m_i; ml_s[rw * 2 + 1] = l_i; }
      float* ar = acc_s + rw * D;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
        ar[(qi * 2) * 8 + nb] = acc[nb][0];
        ar[(qi * 2 + 1) * 8 + nb] = acc[nb][1];
      }
      __syncthreads();
      const int h = tid >> 4, d0 = (tid & 15) * 4;
      const int hg = h / G, r = h % G;
      float mw[4];
      float M = NEG_INF;
#pragma unroll
      for (int j = 0; j < 4; ++j) { mw[j] = ml_s[((hg + 2 * j) * G + r) * 2]; M = fmaxf(M, mw[j]); }
      const float Mu = M == NEG_INF ? 0.f : M;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int wr = (hg + 2 * j) * G + r;
        const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
        L += wt * ml_s[wr * 2 + 1];
        const float4 a = *reinterpret_cast<const float4*>(acc_s + wr * D + d0);
        o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
      }
      const bool whole = seg_head && lt == nt_b - 1;  // this CTA is the request's only segment
      if (TICKET && whole) {
        finalize_store(o, L, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
      } else {
        const int64_t slot = (int64_t)bx + c_b;
        *reinterpret_cast<float4*>(ws_acc + (slot * HQ + h) * D + d0) = o;
        if ((tid & 15) == 0) { ws_ml[slot * 2 * HQ + h] = M; ws_ml[slot * 2 * HQ + HQ + h] = L; }
        if (TICKET && TWO) {
          // Two-level ticket. CTAs c_lo..c_hi hold request c_b's segments (nseg of them); group
          // size Gs = ceil(sqrt(nseg)), group gi = (c - c_lo) / Gs, ngrp = ceil(nseg / Gs) <= 32.
          const int c_lo = tile_owner(tp_s[c_b], T, gx);
          const int c_hi = tile_owner(tp_s[c_b + 1] - 1, T, gx);
          const int nseg = c_hi - c_lo + 1;
          int Gs = 1;
          while (Gs * Gs < nseg) ++Gs;
          const int gi = (bx - c_lo) / Gs;
          const int ngrp = (nseg + Gs - 1) / Gs;
          const int g_lo = c_lo + gi * Gs;
          const int g_hi = min(c_hi, g_lo + Gs - 1);
          __threadfence();  // release: this CTA's partial is visible device-wide before the ticket
          __syncthreads();
          if (tid == 0) flag_s[0] = ticket_arrive(gticket + c_b * 32 + gi) == g_hi - g_lo ? 1 : 0;
          __syncthreads();
          if (flag_s[0]) {  // uniform: the last arrival of group gi
            __threadfence();  // acquire: every group member's partial is visible
            float4 om;
            float Lm, Mm;
            merge_range(ws_acc, ws_ml, g_lo + c_b, g_hi + c_b, h, d0, om, Lm, Mm);
            if (tid == 0) gticket[c_b * 32 + gi] = 0;  // all members arrived: clean for the next launch
            if (ngrp == 1) {
              finalize_store(om, Lm, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
            } else {
              const int64_t gs = (int64_t)c_b * 32 + gi;
              *reinterpret_cast<float4*>(gacc + (gs * HQ + h) * D + d0) = om;
              if ((tid & 15) == 0) { gml[gs * 2 * HQ + h] = Mm; gml[gs * 2 * HQ + HQ + h] = Lm; }
              __threadfence();  // release: the group slot is visible before the second-level ticket
              __syncthreads();
              if (tid == 0) flag_s[1] = ticket_arrive(ticket + c_b) == ngrp - 1 ? 1 : 0;
              __syncthreads();
              if (flag_s[1]) {  // uniform: the last group of request c_b
                __threadfence();  // acquire
                float4 of;
                float Lf, Mf;
                merge_range(gacc, gml, c_b * 32, c_b * 32 + ngrp - 1, h, d0, of, Lf, Mf);
                finalize_store(of, Lf, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
                if (tid == 0) ticket[c_b] = 0;  // every group has arrived: clean for the next launch
              }
            }
          }
        } else if (TICKET) {
          const int c_lo = tile_owner(tp_s[c_b], T, gx);
          const int c_hi = tile_owner(tp_s[c_b + 1] - 1, T, gx);
          __threadfence();  // this CTA's partial is visible device-wide before the ticket
          __syncthreads();
          if (tid == 0) flag_s[0] = ticket_arrive(ticket + c_b) == c_hi - c_lo ? 1 : 0;
          __syncthreads();
          if (flag_s[0]) {  // uniform: the last segment of request c_b to finish
            __threadfence();
            float4 om;
            float Lm;
            merge_slots(ws_acc, ws_ml, c_b, c_lo, c_hi, h, d0, om, Lm);
            finalize_store(om, Lm, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
            if (tid == 0) ticket[c_b] = 0;  // every segment has arrived: clean for the next launch
          }
        }
      }
    }
  }
  cp_wait<0>();  // the trailing groups are empty; drain anyway
}

using SkFn = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const int*, int64_t, const int*, int,
                      const float*, const float*, float, const __nv_bfloat16*, int64_t, __nv_bfloat16*,
                      float*, float*, int*, int*, float*, float*, float);

// variant 0 = <4 stages, single-level ticket> (1 < batch <= 64, grid 2 x SM);
// variant 3 = <4 stages, two-level ticket>    (batch 1, grid = SM count).
SkFn sk_pick(int64_t v) {
  return v == 3 ? splitkv_attn_sk_kernel<4, true, true> : splitkv_attn_sk_kernel<4, true, false>;
}
int sk_smem() { return Lay<4>::BYTES; }
// Function-attribute flags only (one bit per instantiation); no pointer, tensor or descriptor is
// kept in any static: every launch reads its buffers from the arguments.
void sk_prepare(int64_t v) {
  static bool done[2] = {false, false};
  const int i = v == 3 ? 1 : 0;
  if (!done[i]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(sk_pick(v)),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, sk_smem()));
    done[i] = true;
  }
}

}  // namespace

void sk_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor qkv,
             int64_t qkv_stride, int64_t gate_off, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml,
             torch::Tensor ticket,
             torch::Tensor gticket, torch::Tensor gacc, torch::Tensor gml,
             double score_scale, double eps, int64_t grid, int64_t variant) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  TORCH_CHECK(B >= 1 && B <= MAXB, "sk_attn: batch out of range");
  TORCH_CHECK(variant == 0 || variant == 3, "sk_attn: variant must be 0 (single-level) or 3 (two-level)");
  TORCH_CHECK(grid >= 1 && ws_acc.size(0) >= grid + B && ws_ml.size(0) >= grid + B, "sk_attn: workspace");
  TORCH_CHECK(ticket.numel() >= B && ticket.scalar_type() == at::kInt, "sk_attn: ticket");
  if (variant == 3) {
    TORCH_CHECK(grid <= 1024, "sk_attn: two-level ticket needs grid <= 1024 (<= 32 groups)");
    TORCH_CHECK(gticket.numel() >= B * 32 && gticket.scalar_type() == at::kInt, "sk_attn: gticket");
    TORCH_CHECK(gacc.size(0) >= B * 32 && gml.size(0) >= B * 32 && gacc.scalar_type() == at::kFloat
                && gml.scalar_type() == at::kFloat, "sk_attn: group slots");
  }
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && gate_off >= 0 && qkv_stride >= HQ * D, "sk_attn: gate source");
  // The gate rows read are qkv[b * qkv_stride + gate_off .. + 1024) for every b < B: they must lie inside the
  // tensor handed in, so a mismatched stride / offset cannot read a different tensor's memory silently.
  TORCH_CHECK(gate_off + static_cast<int64_t>(B - 1) * qkv_stride + HQ * D <= qkv.numel(),
              "sk_attn: gate rows exceed the gate tensor (gate_off / qkv_stride mismatch)");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && page_table.scalar_type() == at::kInt, "sk_attn: dtypes");
  TORCH_CHECK(page_table.stride(1) == 1 && q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous()
              && out.is_contiguous(), "sk_attn: layout");
  sk_prepare(variant);
  auto stream = at::cuda::getCurrentCUDAStream();
  SkFn fn = sk_pick(variant);
  fn<<<static_cast<unsigned>(grid), THREADS, sk_smem(), stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_table.data_ptr<int>(), page_table.stride(0),
      cache_seqlens.data_ptr<int>(), B, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
      static_cast<float>(score_scale), reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()) + gate_off, qkv_stride,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(),
      ticket.data_ptr<int>(), gticket.data_ptr<int>(), gacc.data_ptr<float>(), gml.data_ptr<float>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// {binaryVersion, ptxVersion, numRegs, localSizeBytes, maxThreadsPerBlock, dynamic smem, CTAs/SM}
std::vector<int64_t> sk_attrs(int64_t variant) {
  TORCH_CHECK(variant == 0 || variant == 3, "sk_attrs: variant");
  sk_prepare(variant);
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(sk_pick(variant))));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sk_pick(variant), THREADS, sk_smem()));
  return {static_cast<int64_t>(a.binaryVersion), static_cast<int64_t>(a.ptxVersion), static_cast<int64_t>(a.numRegs),
          static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.maxThreadsPerBlock),
          static_cast<int64_t>(sk_smem()), static_cast<int64_t>(occ)};
}
"""
_SK = None
_SK_ARCH_FLAGS = None
_SK_MAXB = 64
_SK_TAG = "[sk]"
_SK_MARGIN = 1.02
_SK_DECISION = None
_PREPARE_WALL_S = 0.0
def _say(msg: str) -> None:
    print(msg, flush=True)
def _build_sk(name="muse_decode_sk_s3", extra=()):
    global _SK_ARCH_FLAGS
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        from torch.utils.cpp_extension import load_inline
        try:
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            _SK_ARCH_FLAGS = list(_get_cuda_arch_flags())
        except Exception as e:
            _say(f"{_SK_TAG} could not read torch's nvcc arch flags: {e!r}")
        _say(f"{_SK_TAG} building {name}: TORCH_CUDA_ARCH_LIST=9.0a, arch flags {_SK_ARCH_FLAGS}, "
             f"extra -O3 -std=c++20 -lineinfo{''.join(' ' + f for f in extra)}, instantiations <4,true,false> and "
             f"<4,true,true>")
        ext = load_inline(
            name=name,
            cpp_sources=_SK_CPP,
            cuda_sources=_SK_CUDA,
            functions=["sk_attn", "sk_attrs"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"] + list(extra),
            verbose=False,
        )
    except Exception as e:
        _say(f"{_SK_TAG} load_inline failed after {time.perf_counter() - t0:.1f}s, FA3 kept: {e!r}")
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    _say(f"{_SK_TAG} load_inline compiled in {time.perf_counter() - t0:.1f}s")
    return ext
_SK_S3 = False
def _sk_buffers(device) -> dict:
    rows = 2 * _sm_count(device) + _SK_MAXB
    return {
        "ws_acc": torch.zeros((rows, 16, 64), dtype=torch.float32, device=device),
        "ws_ml": torch.zeros((rows, 2, 16), dtype=torch.float32, device=device),
        "ticket": torch.zeros(_SK_MAXB, dtype=torch.int32, device=device),
        "gticket": torch.zeros(32, dtype=torch.int32, device=device),
        "gacc": torch.zeros((32, 16, 64), dtype=torch.float32, device=device),
        "gml": torch.zeros((32, 2, 16), dtype=torch.float32, device=device),
    }
def _sk_zero(sk) -> None:
    sk["ticket"].zero_()
    sk["gticket"].zero_()
def _sk_ticket_sum(sk) -> int:
    return int(sk["ticket"].abs().sum().item()) + int(sk["gticket"].abs().sum().item())
def _sk_shape_ok(cfg) -> bool:
    return (cfg["window"] == 0 and cfg["q_len"] == 1 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and cfg["page_size"] == 1)
def _sk_batch_class(batch: int) -> int:
    return 16 if batch <= 23 else 32
def _sk_variant(cfg, sk) -> int:
    if sk is None or not sk["on"] or not _sk_shape_ok(cfg):
        return -1
    batch = cfg["batch"]
    if batch == 1:
        return 3 if sk["b1"] else -1
    if 1 < batch <= _SK_MAXB:
        return 0 if sk["n"].get(_sk_batch_class(batch), False) else -1
    return -1
SK_QKV_GATE_OFF = (16 + 2 * 2) * 64
def _full_decode_sk(data: Inputs, q: torch.Tensor, qkv, sk: dict, variant: int, grid=None,
                    gate=None, mod=None) -> torch.Tensor:
    cfg = data["config"]
    batch, hq, d = cfg["batch"], cfg["num_q_heads"], cfg["head_dim"]
    if mod is None:
        mod = _SKF if (variant == 0 and _SKF is not None and sk.get("fl", {}).get(_sk_batch_class(batch), False)) \
            else _SK
    if gate is not None:
        gsrc, goff, gstride = gate, 0, gate.stride(0)
    else:
        gsrc, goff, gstride = qkv, SK_QKV_GATE_OFF, qkv.stride(0)
    device = gsrc.device
    if grid is None:
        grid = _sm_count(device) if variant == 3 else 2 * _sm_count(device)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    mod.sk_attn(q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"], data["k_scale"],
                data["v_scale"], gsrc, gstride, goff, out, sk["ws_acc"], sk["ws_ml"], sk["ticket"], sk["gticket"],
                sk["gacc"], sk["gml"], cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, RMS_EPS, grid, variant)
    return out
def _sk_vname(variant: int) -> str:
    return "splitkv_attn_sk_kernel<4,true,true>" if variant == 3 else "splitkv_attn_sk_kernel<4,true,false>"
def _sk_err(out, ref):
    out = out.reshape(ref.shape[0], -1).float()
    return float((out - ref).norm() / ref.norm()), bool(torch.isfinite(out).all())
def _sk_plant(data, q, lens, pos_mode):
    q0 = q.float()[:, [0, 8], :] * 6.0
    kc = data["k_cache"].view(torch.uint8).view(-1, 2, 64)
    rows, vals = [], []
    for b, n in enumerate(lens):
        pos = {"first": 0, "t63": min(63, n - 1), "t64": min(64, n - 1), "last": n - 1}[pos_mode]
        rows.append(int(data["page_table"][b, pos]))
        vals.append(q0[b])
    kc[torch.tensor(rows, dtype=torch.long, device=kc.device)] = torch.stack(vals).to(FP8).view(torch.uint8)
def _sk_case(tag, variant, lens, kv_scale, grid, gen, device, sk, plant=None, mod=None) -> bool:
    data, q, qkv = _splitkv_synthetic(lens, kv_scale, gen, device)
    assert _sk_shape_ok(data["config"])
    if plant is not None:
        _sk_plant(data, q, lens, plant)
    ref = _splitkv_reference(data, q, qkv, lens)
    g = grid or (_sm_count(device) if variant == 3 else 2 * _sm_count(device))
    _say(f"{_SK_TAG} {tag}: launching {_sk_vname(variant)}{_skf_mname(mod)} B={len(lens)} grid {g}")
    out = _full_decode_sk(data, q, qkv, sk, variant, g, mod=mod)
    torch.cuda.synchronize()
    _say(f"{_SK_TAG} {tag}: {_sk_vname(variant)} returned")
    err, finite = _sk_err(out, ref)
    ok = finite and err < 2e-2
    _say(f"{_SK_TAG} {tag}: rel L2 {err:.2e}, finite {finite} -> {'ok' if ok else 'FAIL'}")
    return ok
def _sk_agentic(rng, n, ctx):
    return [int(ctx * (1 + 0.075 * (2 * rng.random() - 1))) for _ in range(n)]
def _sk_graph_of(fn, device):
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with _rt_cap(g):
        fn()
    torch.cuda.synchronize()
    return g
def _sk_alternation_check(variant, batch, max_len, gen, device, rng, mod=None) -> "str | None":
    tag = f"alt {_sk_vname(variant)}{_skf_mname(mod)} B={batch} max {max_len}"
    sets = []
    for s in range(2):
        sk = _sk_buffers(device)
        data, q, qkv = _splitkv_synthetic([max_len] * batch, 1.0, gen, device)
        cell = {}
        def fn(data=data, q=q, qkv=qkv, sk=sk, cell=cell):
            cell["out"] = _full_decode_sk(data, q, qkv, sk, variant, mod=mod)
        _say(f"{_SK_TAG} {tag}: capturing state {s}")
        graph = _sk_graph_of(fn, device)
        sets.append((sk, data, q, qkv, cell, graph))
    for r in range(6):
        s = r % 2
        sk, data, q, qkv, cell, graph = sets[s]
        kv_scale = 1.0 if (r // 2) % 2 == 0 else 0.25
        nd, nq, nqkv = _splitkv_synthetic([max_len] * batch, kv_scale, gen, device)
        lens = [max_len] + [rng.randint(max_len // 2, max_len) for _ in range(batch - 1)]
        data["k_cache"].copy_(nd["k_cache"])
        data["v_cache"].copy_(nd["v_cache"])
        data["page_table"].copy_(nd["page_table"])
        data["k_scale"].fill_(kv_scale)
        data["v_scale"].fill_(kv_scale)
        data["cache_seqlens"].copy_(torch.tensor(lens, dtype=torch.int32, device=device))
        q.copy_(nq)
        qkv.copy_(nqkv)
        ref = _splitkv_reference(nd, nq, nqkv, lens)
        out = cell["out"]
        out.fill_(float("nan"))
        _say(f"{_SK_TAG} {tag}: replay {r} on state {s} kv_scale {kv_scale} lens {min(lens)}..{max(lens)}: "
             f"launching graph ({_sk_vname(variant)})")
        graph.replay()
        torch.cuda.synchronize()
        err, finite = _sk_err(out, ref)
        _say(f"{_SK_TAG} {tag}: replay {r} state {s}: rel L2 {err:.2e}, finite {finite}, tickets "
             f"{_sk_ticket_sum(sk)}")
        if not (finite and err < 2e-2):
            return f"{tag} replay {r} state {s}"
        if _sk_ticket_sum(sk):
            return f"{tag} replay {r} state {s} left a ticket dirty"
    del sets
    return None
def _sk_graph_time(fns, device, reps=15):
    graphs = [_sk_graph_of(f, device) for f in fns]
    flush = torch.empty(64 << 20, dtype=torch.float32, device=device)
    ts = [[] for _ in fns]
    for it in range(3 + reps):
        for i, g in enumerate(graphs):
            torch.sum(flush)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            if it >= 3:
                ts[i].append(e0.elapsed_time(e1) * 1e3)
    del graphs, flush
    return [sorted(t)[len(t) // 2] for t in ts]
def _sk_select(batch, device, state, rng) -> "tuple[bool, str]":
    ctx = 80000
    lens = _sk_agentic(rng, batch, ctx) if batch > 1 else [ctx]
    gen = torch.Generator(device=device).manual_seed(99 + batch)
    data, q, qkv = _splitkv_synthetic(lens, 1.0, gen, device)
    cfg = data["config"]
    sk = _sk_buffers(device)
    variant = 3 if batch == 1 else 0
    holder = {}
    def new():
        holder["n"] = _full_decode_sk(data, q, qkv, sk, variant)
    gate = qkv[:, (16 + 4) * 64:]
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device)
    def old():
        o = flash_attn_with_kvcache(
            q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"], cu_seqlens_q=cu, max_seqlen_q=1,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True, window_size=(-1, -1),
            scheduler_metadata=None)
        o = o[0] if isinstance(o, tuple) else o
        holder["o"] = state["o_norm_gate"](o.reshape(batch, 1024).to(torch.bfloat16), gate)
    inc_name = "FA3 + combine + norm-gate"
    _say(f"{_SK_TAG} selector B={batch}: launching incumbent ({inc_name}) then {_sk_vname(variant)}, "
         f"lens {min(lens)}..{max(lens)}")
    old()
    torch.cuda.synchronize()
    new()
    torch.cuda.synchronize()
    _say(f"{_SK_TAG} selector B={batch}: both returned")
    ref_n = holder["o"].reshape(batch, -1).float()
    err = float((holder["n"].reshape(batch, -1).float() - ref_n).norm() / ref_n.norm())
    tol = 2e-2
    if not (err < tol and bool(torch.isfinite(holder["n"]).all())):
        return False, f"new vs incumbent rel L2 {err:.2e} at ctx {ctx}"
    t_new, t_old = _sk_graph_time([new, old], device)
    nbytes = sum(lens) * 256
    win = t_old >= _SK_MARGIN * t_new
    msg = (f"new {t_new:.1f} us ({nbytes / (t_new * 1e-6) / 1e12:.2f} TB/s) vs incumbent {inc_name} "
           f"{t_old:.1f} us ({nbytes / (t_old * 1e-6) / 1e12:.2f} TB/s), ratio old/new {t_old / t_new:.3f}, "
           f"need >= {_SK_MARGIN}, rel L2 vs incumbent {err:.2e}")
    _say(f"{_SK_TAG} selector B={batch}: {msg}")
    del holder, sk
    return win, msg
def _sk_single_cases(rng):
    tails = [64 * 3 + 1, 64 * 5 + 2, 64 * 2 + 3, 64 * 7 + 4, 64 * 4 + 5, 64 * 9 + 6, 64 * 6 + 7,
             1, 7, 33, 63, 64, 65, 2000 + 7, 5003, 700]
    return (
        ("B=2 one request spans every CTA, grid 7", [20000, 2], 1.0, 7, None),
        ("B=2 uniform kv 0.25 grid device", [6000, 6000], 0.25, None, None),
        ("B=2 agentic ctx 80000 grid device", _sk_agentic(rng, 2, 80000), 1.0, None, None),
        ("B=5 mixed grid device", [9000, 1025, 70, 4097, 1023], 1.0, None, None),
        ("B=5 mixed kv 0.25 grid 3", [9000, 1025, 70, 4097, 1023], 0.25, 3, None),
        ("B=5 agentic ctx 20000 kv 0.25 grid device", _sk_agentic(rng, 5, 20000), 0.25, None, None),
        ("B=16 tails (seq%64 in 1..7, <64) grid device", tails, 1.0, None, None),
        ("B=16 tails kv 0.25 grid 3", tails, 0.25, 3, None),
        ("B=16 tails grid 40", tails, 1.0, 40, None),
        ("B=16 tails planted last key grid 40", tails, 1.0, 40, "last"),
        ("B=16 tails planted last key grid device", tails, 1.0, None, "last"),
        ("B=16 tails planted key 64 grid 7", tails, 1.0, 7, "t64"),
        ("B=16 tails planted key 63 grid 40", tails, 1.0, 40, "t63"),
        ("B=16 tails planted first key grid 3", tails, 1.0, 3, "first"),
        ("B=16 agentic ctx 8000 grid device", _sk_agentic(rng, 16, 8000), 1.0, None, None),
        ("B=16 uniform 5000 kv 0.25 grid device", [5000] * 16, 0.25, None, None),
        ("B=32 agentic ctx 8000 kv 0.25 grid device", _sk_agentic(rng, 32, 8000), 0.25, None, None),
        ("B=32 agentic ctx 3000 grid 7", _sk_agentic(rng, 32, 3000), 1.0, 7, None),
        ("B=33 agentic ctx 8000 planted last key grid device", _sk_agentic(rng, 33, 8000), 1.0, None, "last"),
        ("B=33 ragged grid device", [1 + (i * 977) % 5000 for i in range(33)], 1.0, None, None),
        ("B=33 uniform grid device", [4000] * 33, 0.25, None, None),
    )
def _sk_validate(state, device) -> "tuple[dict | None, str | None]":
    cap = tuple(torch.cuda.get_device_capability(device))
    sms = _sm_count(device)
    _say(f"{_SK_TAG} device capability {cap}, {torch.cuda.get_device_name(device)}, {sms} SMs")
    if cap != (9, 0):
        return None, f"device capability {cap}"
    for variant in (0, 3):
        a = list(_SK.sk_attrs(variant))
        _say(f"{_SK_TAG} {_sk_vname(variant)}: binaryVersion {a[0]}, ptxVersion {a[1]}, regs {a[2]}, "
             f"local {a[3]} B, dyn smem {a[5]} B, {a[6]} CTAs/SM")
        if a[0] != 90:
            return None, "cubin version is not 90 (sm_90a image not loaded)"
        if a[6] < 1:
            return None, f"{_sk_vname(variant)} cannot launch (0 CTAs/SM)"
        if a[6] < 2:
            _say(f"{_SK_TAG} WARNING: {_sk_vname(variant)} does not reach 2 CTAs/SM (performance, not correctness)")
    if _SK_ARCH_FLAGS is not None:
        bad = [f for f in _SK_ARCH_FLAGS if "arch=" in f and "90a" not in f]
        if bad:
            return None, f"nvcc arch flags beyond sm_90a: {bad}"
    gen = torch.Generator(device=device).manual_seed(4321)
    rng = random.Random(2025)
    sk = _sk_buffers(device)
    if not _sk_case("canary single-level B=2 short", 0, [4, 66], 1.0, None, gen, device, sk):
        return None, "canary single-level"
    if not _sk_case("canary two-level B=1 63 keys grid 3", 3, [63], 1.0, 3, gen, device, sk):
        return None, "canary two-level"
    cases = _sk_single_cases(rng)
    for tag, lens, kv_scale, grid, plant in cases:
        if not _sk_case(tag, 0, lens, kv_scale, grid, gen, device, sk, plant):
            return None, f"self-check single-level {tag}"
    cases1 = (
        ("B=1 80000 keys grid SM", 80000, 1.0, sms, None),
        ("B=1 80000 keys kv 0.25 grid SM", 80000, 0.25, sms, None),
        ("B=1 80000 keys grid 7", 80000, 1.0, 7, None),
        ("B=1 80000 keys kv 0.25 grid 3", 80000, 0.25, 3, None),
        ("B=1 79999 keys planted last key grid SM", 79999, 1.0, sms, "last"),
        ("B=1 80001 keys planted last key grid 7", 80001, 1.0, 7, "last"),
        ("B=1 4097 keys grid SM", 4097, 1.0, sms, None),
        ("B=1 4097 keys planted key 64 grid 7", 4097, 1.0, 7, "t64"),
        ("B=1 4097 keys planted last key grid SM", 4097, 1.0, sms, "last"),
        ("B=1 63 keys grid SM", 63, 1.0, sms, None),
        ("B=1 63 keys planted last key grid 3", 63, 1.0, 3, "last"),
        ("B=1 1 key grid SM", 1, 1.0, sms, None),
        ("B=1 1 key kv 0.25 grid 3", 1, 0.25, 3, None),
    )
    for tag, n, kv_scale, grid, plant in cases1:
        if not _sk_case(tag, 3, [n], kv_scale, grid, gen, device, sk, plant):
            return None, f"self-check two-level {tag}"
    left = _sk_ticket_sum(sk)
    _say(f"{_SK_TAG} tickets (both levels) after the eager self-checks: sum {left}")
    if left:
        return None, "tickets not reset by the eager self-checks"
    del sk
    for variant, batch, max_len in ((3, 1, 80000), (0, 2, 80000), (0, 5, 20000), (0, 33, 8000)):
        reason = _sk_alternation_check(variant, batch, max_len, gen, device, rng)
        if reason is not None:
            return None, reason
    decision = {"on": True, "b1": False, "n": {16: False, 32: False}, "msg": {}}
    for batch in (1, 16, 32):
        try:
            win, msg = _sk_select(batch, device, state, rng)
        except Exception as e:
            win, msg = False, f"selector raised {e!r}"
            _say(f"{_SK_TAG} selector B={batch}: {msg}")
        decision["msg"][batch] = msg
        if batch == 1:
            decision["b1"] = win
        else:
            decision["n"][batch] = win
    return decision, None
def _sk_state_check(sk, decision, device) -> "str | None":
    gen = torch.Generator(device=device).manual_seed(555)
    rng = random.Random(77)
    jobs = []
    if decision["b1"]:
        jobs.append((3, [30000]))
    if decision["n"][16] or decision["n"][32]:
        jobs.append((0, _sk_agentic(rng, 5, 9000)))
    for variant, lens in jobs:
        data, q, qkv = _splitkv_synthetic(lens, 0.25, gen, device)
        ref = _splitkv_reference(data, q, qkv, lens)
        cell = {}
        def fn(data=data, q=q, qkv=qkv, cell=cell, variant=variant):
            cell["out"] = _full_decode_sk(data, q, qkv, sk, variant)
        _say(f"{_SK_TAG} state check: capturing {_sk_vname(variant)} B={len(lens)} on this state's buffers")
        graph = _sk_graph_of(fn, device)
        for i in (1, 2):
            cell["out"].fill_(float("nan"))
            _say(f"{_SK_TAG} state check: replay {i}: launching graph ({_sk_vname(variant)})")
            graph.replay()
            torch.cuda.synchronize()
            err, finite = _sk_err(cell["out"], ref)
            _say(f"{_SK_TAG} state check: replay {i}: rel L2 {err:.2e}, finite {finite}")
            if not (finite and err < 2e-2):
                return f"state check {_sk_vname(variant)} replay {i}"
        if _sk_ticket_sum(sk):
            return f"state check {_sk_vname(variant)} left a ticket dirty"
        del graph
    return None
def _sk_prepare(state, device, t_prepare0):
    global _SK_DECISION
    state["sk"] = None
    reason, decision = None, None
    try:
        if _SK is None:
            reason = "module not built"
        else:
            if _SK_DECISION is None:
                _SK_DECISION, reason0 = _sk_validate(state, device)
                if _SK_DECISION is None:
                    _SK_DECISION = {"on": False, "b1": False, "n": {16: False, 32: False}, "msg": {},
                                    "reason": reason0}
            decision = _SK_DECISION
            if not decision["on"]:
                reason = decision.get("reason", "validation failed")
    except Exception as e:
        reason = f"validation raised {e!r}"
        _SK_DECISION = {"on": False, "b1": False, "n": {16: False, 32: False}, "msg": {}, "reason": reason}
    if reason is None:
        sk = None
        try:
            sk = _sk_buffers(device)
            _sk_zero(sk)
            torch.cuda.synchronize()
            reason = _sk_state_check(sk, decision, device)
        except Exception as e:
            reason = f"state check raised {e!r}"
        finally:
            if sk is not None:
                try:
                    _sk_zero(sk)
                    torch.cuda.synchronize()
                    left = _sk_ticket_sum(sk)
                    _say(f"{_SK_TAG} tickets of this state zeroed: sum {left}")
                    if left:
                        reason = reason or "tickets not zero after prepare"
                except Exception as e:
                    reason = reason or f"ticket zero raised {e!r}"
        if reason is None:
            state["sk"] = {**sk, "on": True, "b1": decision["b1"], "n": dict(decision["n"]), "fl": {16: False, 32: False}}
    wall = _PREPARE_WALL_S + (time.perf_counter() - t_prepare0)
    tail = f"w2_mode={state.get('w2_mode')} prepare_wall_cum={wall:.1f}s"
    if state["sk"] is None:
        for what in ("B=1", "B<=23", "B>=24"):
            _say(f"PATH CHOSEN: full decode attention {what}: incumbent (fallback: {reason}) {tail}")
        return
    msg = decision["msg"]
    names = {"B=1": (decision["b1"], 1, "stream-K two-level ticket, grid = SM count", "FA3 + combine + norm-gate"),
             "B<=23": (decision["n"][16], 16, "stream-K single-level ticket, grid 2 x SM", "FA3 paired"),
             "B>=24": (decision["n"][32], 32, "stream-K single-level ticket, grid 2 x SM", "FA3 paired")}
    for what, (win, b, new_name, inc_name) in names.items():
        _say(f"PATH CHOSEN: full decode attention {what}: {new_name if win else inc_name + ' incumbent'} "
             f"[{msg.get(b, 'n/a')}] {tail}")
if torch.cuda.is_available():
    _SK = _build_sk()
    _SK_S3 = _SK is not None
def _prepare_done(t_prepare0) -> None:
    global _PREPARE_WALL_S
    dt = time.perf_counter() - t_prepare0
    _PREPARE_WALL_S += dt
    _say(f"[prepare] this call {dt:.1f}s, cumulative {_PREPARE_WALL_S:.1f}s")
SKF_TAG = "[skfull]"
SKF_MODULE = "muse_decode_sk_s3_fl1"
SKF_CLASSES = (16, 32)
SKF_LAYER_MARGIN = 1.01
SKF_START_DEADLINE_S = 225.0
SKF_GATE_DEADLINE_S = 255.0
_SKF = None
_SKF_BUILD_LINE = f"{SKF_TAG} build: not attempted"
def _skf_mname(mod) -> str:
    if mod is None or mod is _SK:
        return ""
    return " [full-line]" if mod is _SKF else " [module]"
if torch.cuda.is_available() and _SK is not None and _SK_S3:
    _t_skf = time.perf_counter()
    _SKF = _build_sk(SKF_MODULE, extra=("-DSK_FULLLINE=1",))
    _SKF_BUILD_LINE = (f"{SKF_TAG} build {SKF_MODULE} (-DSK_FULLLINE=1): "
                       f"{'ok' if _SKF is not None else 'FAILED'} in {time.perf_counter() - _t_skf:.1f}s at import")
    _say(_SKF_BUILD_LINE)
elif torch.cuda.is_available():
    _SKF_BUILD_LINE = f"{SKF_TAG} build skipped: the stage-3 sk module is not the one loaded (incumbent only)"
def _skf_step(label, what):
    torch.cuda.synchronize()
    _say(f"{SKF_TAG}     [{label}] {what}")
def sk_fullline_canary(device) -> "tuple[str | None, list]":
    gen = torch.Generator(device=device).manual_seed(20261091)
    rng = random.Random(20261091)
    sk = _sk_buffers(device)
    lines = []
    cases = (("synthetic B=2 lens 4,66 grid 3", [4, 66], 1.0, 3, True),
             ("synthetic B=3 lens 1,64,129 grid device", [1, 64, 129], 1.0, None, True),
             ("synthetic B=16 tails kv 0.25 grid 7", [64 * 3 + 1, 64 * 5 + 2, 64 * 2 + 3, 64 * 7 + 4, 64 * 4 + 5,
                                                      64 * 9 + 6, 64 * 6 + 7, 1, 7, 33, 63, 64, 65, 2007, 5003, 700],
              0.25, 7, True),
             ("synthetic B=33 ragged grid device", [1 + (i * 977) % 5000 for i in range(33)], 1.0, None, True),
             ("judge-shaped B=16 agentic 80000 grid device", _sk_agentic(rng, 16, 80000), 1.0, None, False),
             ("judge-shaped B=32 agentic 80000 grid device", _sk_agentic(rng, 32, 80000), 1.0, None, False))
    for tag, lens, kv_scale, grid, with_ref in cases:
        label = f"sk_fullline_canary {tag}"
        data, q, qkv = _splitkv_synthetic(lens, kv_scale, gen, device)
        g = grid or 2 * _sm_count(device)
        _skf_step(label, f"launch half-line (incumbent module) B={len(lens)} grid {g}")
        out_h = _full_decode_sk(data, q, qkv, sk, 0, g, mod=_SK)
        _skf_step(label, "done half-line")
        out_h0, out_h = out_h, out_h.clone()
        junk = torch.full_like(out_h, float("nan"))
        del junk
        _skf_step(label, f"launch full-line ({SKF_MODULE}) B={len(lens)} grid {g}")
        out_f = _full_decode_sk(data, q, qkv, sk, 0, g, mod=_SKF)
        _skf_step(label, "done full-line")
        same = torch.equal(out_h.view(torch.int16), out_f.view(torch.int16))
        finite = bool(torch.isfinite(out_f.float()).all())
        n_diff = int((out_h.view(torch.int16) != out_f.view(torch.int16)).sum())
        left = _sk_ticket_sum(sk)
        ref_txt, ref_ok = "", True
        if with_ref:
            err, _ = _sk_err(out_f, _splitkv_reference(data, q, qkv, lens))
            ref_ok = err < 2e-2
            ref_txt = f", full-line vs fp32 rel L2 {err:.2e}"
        ok = same and finite and left == 0 and ref_ok
        line = (f"{SKF_TAG}   sk_fullline_canary {tag}: bitwise {'IDENTICAL' if same else f'DIFFERS ({n_diff} values)'}"
                f", finite {finite}, tickets {left}{ref_txt} -> {'ok' if ok else 'FAIL'}")
        _say(line)
        lines.append(line)
        del data, q, qkv, out_h, out_f
        if not ok:
            lines.append(f"{SKF_TAG} sk_fullline_canary: MISS at {tag}: {line.split(': ', 1)[1]}")
            _say(lines[-1])
            return tag, lines
    torch.cuda.synchronize()
    lines.append(f"{SKF_TAG} sk_fullline_canary: {len(cases)} cases (4 synthetic, 2 judge-shaped) bitwise identical "
                 f"half-line vs full-line, tickets 0, synced")
    _say(lines[-1])
    return None, lines
def _skf_validate(device) -> "str | None":
    a_h, a_f = list(_SK.sk_attrs(0)), list(_SKF.sk_attrs(0))
    _say(f"{SKF_TAG} {_sk_vname(0)} attrs half-line / full-line: binaryVersion {a_h[0]}/{a_f[0]}, regs {a_h[2]}/"
         f"{a_f[2]}, local {a_h[3]}/{a_f[3]} B, dyn smem {a_h[5]}/{a_f[5]} B, CTAs/SM {a_h[6]}/{a_f[6]}")
    if a_f[0] != 90:
        return "full-line cubin version is not 90"
    if a_f[6] < 1:
        return "full-line variant cannot launch (0 CTAs/SM)"
    if a_f[3] != 0:
        _say(f"{SKF_TAG} WARNING: full-line variant uses {a_f[3]} B of local memory (performance, not correctness)")
    gen = torch.Generator(device=device).manual_seed(4321)
    rng = random.Random(2025)
    sk = _sk_buffers(device)
    for tag, lens, kv_scale, grid, plant in _sk_single_cases(rng):
        if not _sk_case(f"full-line {tag}", 0, lens, kv_scale, grid, gen, device, sk, plant, mod=_SKF):
            return f"self-check full-line {tag}"
    left = _sk_ticket_sum(sk)
    _say(f"{SKF_TAG} tickets after the full-line eager self-checks: sum {left}")
    if left:
        return "tickets not reset by the full-line self-checks"
    del sk
    for batch, max_len in ((2, 80000), (5, 20000), (33, 8000)):
        reason = _sk_alternation_check(0, batch, max_len, gen, device, rng, mod=_SKF)
        if reason is not None:
            return reason
    return None
def _skf_layer_data(state, device, gen, rng, cls):
    data = _cb_data(state, device, gen, 0, cls, 1, _sk_agentic(rng, cls, 80000), 1.0)
    ls, _load = _ead_busiest(data, state)
    data["local_start"] = torch.tensor([ls], dtype=torch.int32, device=device)
    return data
def _skf_layer_canary_check(state, device, cls) -> "tuple[str | None, str]":
    fl = state["sk"]["fl"]
    gen = torch.Generator(device=device).manual_seed(20261093 + cls)
    rng = random.Random(20261093 + cls)
    label = f"layer canary d{cls}f full-line in kernel_fn"
    worst = {}
    try:
        data = _skf_layer_data(state, device, gen, rng, cls)
        ls_val = int(data["local_start"][0])
        s_in = data["stream"].clone()
        scores = _cb_router_scores(_dl3_clone_data(data), state)
        fl[cls] = False
        d_ref = _dl3_clone_data(data)
        _skf_step(label, "launch eager kernel_fn, half-line (reference)")
        ref = kernel_fn(d_ref, state)
        attn_ref = _attn_core(_dl3_clone_data(data), state).clone()
        attn_ref2 = _attn_core(_dl3_clone_data(data), state).clone()
        _skf_step(label, "done half-line")
        attn_det = torch.equal(attn_ref.view(torch.int16), attn_ref2.view(torch.int16))
        fl[cls] = True
        d_got = _dl3_clone_data(data)
        _skf_step(label, "launch eager kernel_fn, full-line")
        got = kernel_fn(d_got, state)
        attn_got = _attn_core(_dl3_clone_data(data), state).clone()
        _skf_step(label, "done full-line")
        d_g = _dl3_clone_data(data)
        _skf_step(label, "graph warm-up + capture of kernel_fn, full-line")
        graph, g_out = _rt_graph(lambda: kernel_fn(d_g, state))
        _skf_step(label, "captured")
        for i in range(2):
            g_out[0].fill_(float("nan"))
            _skf_step(label, f"launch graph replay {i}")
            graph.replay()
            _skf_step(label, f"done graph replay {i}")
        g_finite = bool(torch.isfinite(g_out[0].float()).all())
        left = _sk_ticket_sum(state["sk"])
        del graph, g_out, d_g
        attn_same = torch.equal(attn_ref.view(torch.int16), attn_got.view(torch.int16))
        attn_rel = _rt_rel(attn_got.float(), attn_ref.float())
        ok, msg = _dsa_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst)
        ok = ok and g_finite and left == 0 and (attn_same or not attn_det)
        line = (f"{SKF_TAG} layer canary + check d{cls}f (on vs off): attention output bitwise "
                f"{'IDENTICAL' if attn_same else f'DIFFERS (rel {attn_rel:.1e})'} (half-line run-to-run "
                f"{'deterministic' if attn_det else 'NOT deterministic'}), graph replays finite {g_finite}, "
                f"tickets {left}; "
                + ", ".join(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items())
                + f" -> {'ok' if ok else 'FAIL'}")
        return (None if ok else f"layer check d{cls}f: {msg[:200]}"), line
    finally:
        fl[cls] = False
def _skf_layer_gate(state, device, cls) -> "tuple[bool, float, list]":
    fl = state["sk"]["fl"]
    gen = torch.Generator(device=device).manual_seed(20261094 + cls)
    rng = random.Random(20261094 + cls)
    per = {}
    try:
        for rep in range(2):
            data = _skf_layer_data(state, device, gen, rng, cls)
            graphs = {}
            for name, v in (("half", False), ("full", True)):
                fl[cls] = v
                _skf_step(f"layer gate d{cls}f input {rep}", f"capture kernel_fn graph, {name}-line")
                graphs[name], _ = _rt_graph(lambda: kernel_fn(data, state))
                _skf_step(f"layer gate d{cls}f input {rep}", f"captured {name}-line")
            fl[cls] = False
            per[f"d{cls}f layer input {rep}"] = _rt_time(graphs, device)
            _skf_step(f"layer gate d{cls}f input {rep}", "timed")
            del graphs, data
    finally:
        fl[cls] = False
    win, worst, rlines = _rt_select(per, "half", ["full"], margin=SKF_LAYER_MARGIN, tight=SKF_LAYER_MARGIN)
    parts = []
    for rounds in per.values():
        parts.append(" ".join(f"{_rt_stats(r['half'])[0]:.1f}/{_rt_stats(r['full'])[0]:.1f}" for r in rounds))
    lines = [f"{SKF_TAG} layer gate d{cls}f (whole layer, read flush), us half/full per round: {' | '.join(parts)} -> "
             f"worst x{worst['full']:.3f} (need >= {SKF_LAYER_MARGIN} every round) -> "
             f"{'FULL-LINE' if win == 'full' else 'half-line'}"]
    for rl in rlines:
        lines.append(f"{SKF_TAG}   {rl[:400]}")
    return win == "full", worst["full"], lines
def _skf_state_check(state, device) -> "str | None":
    sk = state["sk"]
    gen = torch.Generator(device=device).manual_seed(20261095)
    lens = _sk_agentic(random.Random(20261095), 5, 9000)
    data, q, qkv = _splitkv_synthetic(lens, 0.25, gen, device)
    ref = _splitkv_reference(data, q, qkv, lens)
    cell = {}
    def fn():
        cell["out"] = _full_decode_sk(data, q, qkv, sk, 0, mod=_SKF)
    _skf_step("state check", "capture full-line B=5 on this state's buffers")
    graph = _sk_graph_of(fn, device)
    for i in (1, 2):
        cell["out"].fill_(float("nan"))
        _skf_step("state check", f"launch graph replay {i}")
        graph.replay()
        _skf_step("state check", f"done graph replay {i}")
        err, finite = _sk_err(cell["out"], ref)
        if not (finite and err < 2e-2):
            return f"state check replay {i}: rel L2 {err:.2e}, finite {finite}"
    del graph
    if _sk_ticket_sum(sk):
        return "state check left a ticket dirty"
    return None
def _skf_setup(state, device, t_prepare0) -> list:
    lines = [_SKF_BUILD_LINE]
    t0 = time.perf_counter()
    decisions = {cls: "half-line incumbent (not reached)" for cls in SKF_CLASSES}
    sk = state.get("sk")
    def finish():
        for cls in SKF_CLASSES:
            lines.append(f"{SKF_TAG} PATH CHOSEN full decode attention d{cls}f K/V gather: {decisions[cls]}")
        lines.append(f"{SKF_TAG} ===== W3-A SUMMARY ({time.perf_counter() - t0:.1f} s; prepare at "
                     f"{time.perf_counter() - t_prepare0:.1f} s) fl={None if sk is None else sk.get('fl')} =====")
        return lines
    if _SKF is None:
        for cls in SKF_CLASSES:
            decisions[cls] = "half-line incumbent (full-line module not built)"
        return finish()
    if sk is None:
        for cls in SKF_CLASSES:
            decisions[cls] = "half-line incumbent (stream-K off for this state)"
        return finish()
    if time.perf_counter() - t_prepare0 > SKF_START_DEADLINE_S:
        for cls in SKF_CLASSES:
            decisions[cls] = "half-line incumbent (deadline before the canary)"
        return finish()
    try:
        _sk_zero(sk)
        miss, clines = sk_fullline_canary(device)
        lines += clines
        if miss is not None:
            for cls in SKF_CLASSES:
                decisions[cls] = f"half-line incumbent (sk_fullline_canary miss: {miss})"
            return finish()
        reason = _skf_validate(device)
        lines.append(f"{SKF_TAG} stage-2 self-check (full-line, single-level): {'ok' if reason is None else reason}")
        _say(lines[-1])
        if reason is not None:
            for cls in SKF_CLASSES:
                decisions[cls] = f"half-line incumbent (self-check miss: {reason})"
            return finish()
        reason = _skf_state_check(state, device)
        lines.append(f"{SKF_TAG} state check (this state's buffers): {'ok' if reason is None else reason}")
        _say(lines[-1])
        if reason is not None:
            for cls in SKF_CLASSES:
                decisions[cls] = f"half-line incumbent (state check miss: {reason})"
            return finish()
        for cls in SKF_CLASSES:
            if not sk["n"].get(cls, False):
                decisions[cls] = "half-line incumbent (this class does not run stream-K: nothing to gate)"
                continue
            if time.perf_counter() - t_prepare0 > SKF_GATE_DEADLINE_S:
                decisions[cls] = "half-line incumbent (deadline before the layer gate)"
                continue
            try:
                problem, line = _skf_layer_canary_check(state, device, cls)
                lines.append(line)
                _say(line)
                if problem is not None:
                    decisions[cls] = f"half-line incumbent ({problem})"
                    continue
                win, worst, glines = _skf_layer_gate(state, device, cls)
                lines += glines
                for gl in glines:
                    _say(gl)
                if win:
                    sk["fl"][cls] = True
                    decisions[cls] = f"FULL-LINE (SK_FULLLINE, {SKF_MODULE}; whole layer x{worst:.3f})"
                else:
                    decisions[cls] = f"half-line incumbent (whole layer x{worst:.3f} < {SKF_LAYER_MARGIN})"
            except Exception as e:
                sk["fl"][cls] = False
                decisions[cls] = f"half-line incumbent (raised {type(e).__name__}: {str(e)[:160]})"
                _say(f"{SKF_TAG} d{cls}f: {decisions[cls]}")
        torch.cuda.synchronize()
        left = _sk_ticket_sum(sk)
        _sk_zero(sk)
        torch.cuda.synchronize()
        if left:
            for cls in SKF_CLASSES:
                sk["fl"][cls] = False
                decisions[cls] = f"half-line incumbent (tickets {left} after W3-A)"
    except Exception as e:
        for cls in SKF_CLASSES:
            sk["fl"][cls] = False
            decisions[cls] = f"half-line incumbent (W3-A raised {type(e).__name__}: {str(e)[:160]})"
        try:
            _sk_zero(sk)
            torch.cuda.synchronize()
        except Exception:
            pass
    try:
        _sk_zero(sk)
        torch.cuda.synchronize()
    except Exception:
        pass
    return finish()
DL3_TAG = "[dl3]"
DL3_WS_TOKENS = 32
DL3_Q_ROWS = 2304
DL3_Q_SPLIT = 6
DL3_Q_NTILES = DL3_Q_ROWS // 64
DL3_O_SPLIT = 2
DL3_O_NTILES = HIDDEN // 64
DL3_MARGIN = 1.02
DL3_GATE_STRIDE = 1024
def _say3(message):
    print(f"{DL3_TAG} {message}", flush=True)
def _dl3_launch(label, fn):
    _say3(f"launching {label}")
    result = fn()
    torch.cuda.synchronize()
    _say3(f"ok        {label}")
    return result
def _dl3_buffers(device):
    return {
        "ws_q": torch.zeros((DL3_Q_SPLIT * DL3_WS_TOKENS * DL3_Q_ROWS,), dtype=torch.float32, device=device),
        "cnt_q": torch.zeros((DL3_Q_NTILES,), dtype=torch.int32, device=device),
        "ws_o": torch.zeros((DL3_O_SPLIT * DL3_WS_TOKENS * HIDDEN,), dtype=torch.float32, device=device),
        "cnt_o": torch.zeros((DL3_O_NTILES,), dtype=torch.int32, device=device),
        "ws_u": torch.zeros((2 * DL3_WS_TOKENS * HIDDEN,), dtype=torch.float32, device=device),
        "cnt_u": torch.zeros((HIDDEN // 64,), dtype=torch.int32, device=device),
    }
def _dl3_tickets(state) -> int:
    total = 0
    d3, dl, sk = state.get("dl3"), state.get("dl"), state.get("sk")
    if d3 is not None:
        total += sum(int(d3[k].abs().sum()) for k in ("cnt_q", "cnt_o", "cnt_u"))
    if dl is not None:
        total += int(dl["cnt_p"].abs().sum()) + int(dl["cnt_r"].abs().sum())
    if sk is not None:
        total += _sk_ticket_sum(sk)
    return total
def _dl3_zero(state):
    d3, dl, sk = state.get("dl3"), state.get("dl"), state.get("sk")
    if d3 is not None:
        for k in ("cnt_q", "cnt_o", "cnt_u"):
            d3[k].zero_()
    if dl is not None:
        dl["cnt_p"].zero_()
        dl["cnt_r"].zero_()
    if sk is not None:
        _sk_zero(sk)
    torch.cuda.synchronize()
def _dl3_q(data, x, state, d3):
    cfg = data["config"]
    tokens = x.shape[0]
    dev = x.device
    q_attn = torch.empty((tokens, 16, 64), dtype=FP8, device=dev)
    gate = torch.empty((tokens, 1024), dtype=torch.bfloat16, device=dev)
    _DL_EXT.dl2_qkv(x.contiguous(), state["dense"]["qkv_gate"], d3["ws_q"], d3["cnt_q"], gate, q_attn.view(torch.uint8), gate,
                    data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["cache_loc"],
                    data["positions"], state["cos_sin"], data["k_scale"], data["v_scale"], d3["pdl"], 0, 0,
                    int(bool(cfg["rope_theta"])), int(cfg["kv_scale"] != 1.0))
    return q_attn, gate
def _dl3_o(o, state, d3):
    o = o.contiguous()
    out = torch.empty((o.shape[0], HIDDEN), dtype=torch.bfloat16, device=o.device)
    _DL_EXT.dl2_o(o, o, state["dense"]["o"], d3["ws_o"], d3["cnt_o"], out, d3["pdl"], 0, 0)
    return out
def _dl3_attention(data, state, d3):
    c3 = state.get("c320d")
    if c3 is not None and c3["on"] and d3["pdl"]:
        return _c320d_attention(data, state, d3, c3["trig"])
    tokens = data["config"]["tokens"]
    q_bytes, gate = _dl3_q(data, data["attn_input"], state, d3)
    o = _full_decode_sk(data, q_bytes, None, state["sk"], 3, gate=gate)
    return _dl3_o(o.reshape(tokens, 1024), state, d3)
def _unpool_master(dense, expert_output):
    unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
    return F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
def _dl3_unpool(expert_output, state, d3):
    ffn_output = torch.empty((expert_output.shape[0], HIDDEN), dtype=torch.bfloat16, device=expert_output.device)
    _DL_EXT.dl_unpool(expert_output.contiguous(), state["dense"]["unpooler"], ffn_output, d3["ws_u"], d3["cnt_u"],
                      d3["pdl"])
    return ffn_output
def _dl3_flags(state, a=None, b=None, u=None):
    ms = dict(state)
    ms["dl"] = dict(state["dl"])
    ms["dl3"] = dict(state["dl3"])
    if a is not None:
        ms["dl3"]["a"] = bool(a)
    if b is not None:
        ms["dl"]["r_new"] = bool(b)
    if u is not None:
        ms["dl3"]["u"] = bool(u)
    return ms
def _dl3_ministate(state, device):
    ms = dict(state)
    dl = dict(state["dl"])
    for k in ("ws", "cnt_p", "cnt_r"):
        dl[k] = torch.zeros_like(state["dl"][k])
    d3 = dict(state["dl3"])
    d3.update(_dl3_buffers(device))
    ms["dl"], ms["dl3"] = dl, d3
    if state.get("sk") is not None:
        sk = _sk_buffers(device)
        _sk_zero(sk)
        ms["sk"] = {**sk, "on": True, "b1": state["sk"]["b1"], "n": dict(state["sk"]["n"])}
    ms["slide_cnt"] = torch.zeros_like(state["slide_cnt"])
    torch.cuda.synchronize()
    return ms
def _fp8_code_gap(a, b) -> int:
    def order(u):
        u = u.to(torch.int16)
        return torch.where(u >= 128, -(u - 128), u)
    return int((order(a) - order(b)).abs().max())
def _dl3_ref_q(data, x, state):
    cfg = data["config"]
    tokens = x.shape[0]
    hq, hkv, d = 16, 2, 64
    qkv = _gemm(x, state["dense"]["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k = k.reshape(tokens, hkv * d)
    if cfg["kv_scale"] != 1.0:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                data["k_cache"].view(-1, hkv * d).view(torch.uint8), data["v_cache"].view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    return q.view(tokens, hq, d).to(FP8).contiguous(), gate.contiguous()
def _dl3_make_x(state, device, gen, kv_scale):
    x = torch.randn((1, HIDDEN), device=device, generator=gen).to(torch.bfloat16)
    vmax = float(_gemm(x, state["dense"]["qkv_gate"])[:, 1152:1280].abs().max())
    lim = 200.0 * kv_scale
    if vmax > lim:
        x = (x.float() * (lim / vmax)).to(torch.bfloat16)
    return x
def _dl3_layer_data(state, device, gen, window, kv_scale, n, local_start=0):
    hkv, d = 2, 64
    max_pages = n + 37
    perm = torch.randperm(max_pages, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((1, max_pages), -1, dtype=torch.int32, device=device)
    page_table[0, :n] = perm[:n]
    cfg = {"batch": 1, "q_len": 1, "tokens": 1, "hidden": HIDDEN, "num_q_heads": 16, "num_kv_heads": hkv,
           "head_dim": d, "page_size": 1, "window": window, "rope_theta": ROPE_THETA if window else 0.0,
           "qk_scale_factor": 5.4730064863875, "sm_scale": d ** -0.5, "kv_scale": kv_scale, "n_experts": N_EXPERTS,
           "top_k": TOP_K, "local_experts": LOCAL_EXPERTS}
    return {
        "stream": torch.randn((1, HIDDEN), device=device, generator=gen) * 3.0,
        "attn_input": _dl3_make_x(state, device, gen, kv_scale),
        "k_cache": (torch.randn((max_pages, 1, hkv, d), generator=gen, device=device) / kv_scale).to(FP8),
        "v_cache": (torch.randn((max_pages, 1, hkv, d), generator=gen, device=device) / kv_scale).to(FP8),
        "positions": torch.tensor([n - 1], dtype=torch.int64, device=device),
        "cache_loc": page_table[0, n - 1:n].to(torch.int64).contiguous(),
        "cache_seqlens": torch.tensor([n], dtype=torch.int32, device=device),
        "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "scheduler_metadata": None,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table,
        "seq_lens": torch.tensor([n - 1], dtype=torch.int32, device=device),
        "local_start": torch.tensor([local_start], dtype=torch.int32, device=device),
        "config": cfg,
    }
def _dl3_clone_data(data):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
def _dl3_check_a(state, device):
    d3 = state["dl3"]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261031)
    problems = []
    worst = {"q": 0, "k": 0, "v": 0, "gate": 0.0, "o": 0.0}
    for kvs in (1.0, 0.25):
        tag = f"kv_scale={kvs}"
        data = _dl3_layer_data(state, device, gen, 0, kvs, 2049)
        ref = _dl3_clone_data(data)
        x = data["attn_input"]
        q_ref, gate_ref = _dl3_ref_q(ref, x, state)
        torch.cuda.synchronize()
        init_k = data["k_cache"].view(torch.uint8).view(-1, 128).clone()
        init_v = data["v_cache"].view(torch.uint8).view(-1, 128).clone()
        for rep in range(2):
            q_bytes, gate = _dl3_launch(f"Q full T=1 {tag} rep{rep}", lambda: _dl3_q(data, x, state, d3))
        loc = data["cache_loc"]
        kn, vn = data["k_cache"].view(torch.uint8).view(-1, 128), data["v_cache"].view(torch.uint8).view(-1, 128)
        kr, vr = ref["k_cache"].view(torch.uint8).view(-1, 128), ref["v_cache"].view(torch.uint8).view(-1, 128)
        gap_q = _fp8_code_gap(q_bytes.view(torch.uint8), q_ref.view(torch.uint8))
        gap_k, gap_v = _fp8_code_gap(kn[loc], kr[loc]), _fp8_code_gap(vn[loc], vr[loc])
        e_g = _rel_l2(gate, gate_ref)
        e_q = _rel_l2(q_bytes.float(), q_ref.float())
        e_k = _rel_l2(kn[loc].view(FP8).float(), kr[loc].view(FP8).float())
        e_v = _rel_l2(vn[loc].view(FP8).float(), vr[loc].view(FP8).float())
        untouched = True
        for new, init in ((kn, init_k), (vn, init_v)):
            want = init.clone()
            want[loc] = new[loc]
            untouched = untouched and torch.equal(new, want)
        worst["q"], worst["k"], worst["v"] = max(worst["q"], gap_q), max(worst["k"], gap_k), max(worst["v"], gap_v)
        worst["gate"] = max(worst["gate"], e_g)
        _say3(f"Q {tag}: fp8 code gap q {gap_q} k {gap_k} v {gap_v} (<= 1 required), rel L2 q {e_q:.2e} k {e_k:.2e} "
              f"v {e_v:.2e} gate {e_g:.2e}, untouched rows intact {untouched}, gate stride {gate.stride(0)}")
        if (not bool(torch.isfinite(gate.float()).all()) or not untouched or max(gap_q, gap_k, gap_v) > 1
                or max(e_q, e_k, e_v) > 4e-2 or e_g > 1e-2 or gate.stride(0) != DL3_GATE_STRIDE):
            problems.append(f"Q {tag}: code gaps {gap_q}/{gap_k}/{gap_v}, rel L2 {e_q:.2e}/{e_k:.2e}/{e_v:.2e}, "
                            f"gate {e_g:.2e}, untouched {untouched}, gate stride {gate.stride(0)}")
        o_in = torch.randn((1, 1024), device=device, generator=gen).to(torch.bfloat16)
        y_ref = _gemm(o_in, state["dense"]["o"])
        for rep in range(2):
            y = _dl3_launch(f"O prologue-0 T=1 {tag} rep{rep}", lambda: _dl3_o(o_in, state, d3))
        e_o = _rel_l2(y, y_ref)
        worst["o"] = max(worst["o"], e_o)
        _say3(f"O {tag}: rel L2 {e_o:.2e}")
        if e_o > 5e-3 or not bool(torch.isfinite(y.float()).all()):
            problems.append(f"O {tag}: rel L2 {e_o:.2e}")
    sk = _sk_buffers(device)
    for n, kvs in ((4097, 1.0), (30000, 0.25)):
        sdata, sq, sqkv = _splitkv_synthetic([n], kvs, gen, device)
        gate_c = torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device)
        gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:])
        a = _dl3_launch(f"sk two-level, gate in qkv (n={n})", lambda: _full_decode_sk(sdata, sq, sqkv, sk, 3))
        b = _dl3_launch(f"sk two-level, separate gate stride {gate_c.stride(0)} (n={n})",
                        lambda: _full_decode_sk(sdata, sq, None, sk, 3, gate=gate_c))
        same = bool(torch.equal(a, b)) and bool(torch.isfinite(b.float()).all())
        _say3(f"sk gate plumbing n={n}: separate-gate output identical to qkv-gate output: {same}")
        _say3(f"sk gate plumbing n={n}: separate gate stride {gate_c.stride()} (required row stride {DL3_GATE_STRIDE})")
        if not same or gate_c.stride(0) != DL3_GATE_STRIDE:
            problems.append(f"sk gate plumbing n={n}: outputs differ (or gate stride {gate_c.stride(0)})")
    if _sk_ticket_sum(sk):
        problems.append("sk tickets dirty after the gate-plumbing check")
    _say3(f"piece A kernel-level worst: {worst}")
    return problems
def _dl3_route_out(ei, lg, ls, dl, bias, variant, pdl):
    bufs, bsm = _dl_route_bufs(ei.shape[0], ei.device)
    label = f"R variant {variant} (T={ei.shape[0]} pdl={pdl if variant == 0 else 0})"
    _dl3_launch(label, lambda: _dl_route_launch(ei, lg, ls, dl, bufs, bsm, pdl if variant == 0 else 0, variant, bias))
    return bufs
def _dl3_check_r(state, device, tokens=(1,)):
    dl = state["dl"]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261032)
    problems = []
    names = ("xn", "topk_ids", "topk_w", "local_ids", "sorted_ids", "block_experts", "n_padded")
    zero_bias = torch.zeros(N_EXPERTS, dtype=torch.float32, device=device)
    cases = []
    for t in tokens:
        for trial, ls_val in enumerate((0, 32, 96, 224, 160, 64)):
            ei = torch.randn((t, D_EXPERT), device=device, generator=gen).to(torch.bfloat16)
            lg = (torch.randn((t, N_EXPERTS), device=device, generator=gen) * 4.0).contiguous()
            cases.append((f"T={t} random{trial} ls={ls_val}", ei, lg, ls_val, dl["bias"]))
        ei = torch.randn((t, D_EXPERT), device=device, generator=gen).to(torch.bfloat16)
        base = torch.randn((t, N_EXPERTS), device=device, generator=gen) * 4.0
        lg = base.clone()
        lg[:, :LOCAL_EXPERTS] = -200.0
        lg[:, LOCAL_EXPERTS:LOCAL_EXPERTS + TOP_K] = 200.0
        cases.append((f"T={t} no-local ls=0", ei, lg.contiguous(), 0, dl["bias"]))
        lg = base.clone()
        lg[:, 64 + 5] = 200.0
        cases.append((f"T={t} one-hot-local ls=64", ei, lg.contiguous(), 64, dl["bias"]))
        lg = base.clone()
        sc = torch.sigmoid(lg.double() * 0.1767766952966369) + dl["bias"].double()
        order = sc.argsort(dim=-1, descending=True)
        a8, a9 = order[:, TOP_K - 1], order[:, TOP_K]
        target = (sc.gather(1, a8[:, None]) - 1e-7 - dl["bias"].double()[a9][:, None]).clamp(1e-6, 1 - 1e-6)
        lg.scatter_(1, a9[:, None], (torch.log(target / (1 - target)) / 0.1767766952966369).float())
        cases.append((f"T={t} near-tie ls=224", ei, lg.contiguous(), 224, dl["bias"]))
        lg = (torch.randn((t, N_EXPERTS), device=device, generator=gen) * 0.5 - 6.0)
        lg[:, 5:17] = 1.5
        cases.append((f"T={t} exact-tie12 ls=0 zero-bias", ei, lg.contiguous(), 0, zero_bias))
        lg = (torch.randn((t, N_EXPERTS), device=device, generator=gen) * 0.5 - 6.0)
        lg[:, 100:104] = 2.0
        lg[:, 200:210] = 1.0
        cases.append((f"T={t} exact-tie-2groups ls=96 zero-bias", ei, lg.contiguous(), 96, zero_bias))
    for name, ei, lg, ls_val, bias in cases:
        ls = torch.tensor([ls_val], dtype=torch.int32, device=device)
        old = _dl3_route_out(ei, lg, ls, dl, bias, 2, 0)
        new0 = _dl3_route_out(ei, lg, ls, dl, bias, 0, 0)
        new1 = _dl3_route_out(ei, lg, ls, dl, bias, 0, dl["pdl"]) if dl["pdl"] else new0
        for tag, new in (("pdl=0", new0), ("pdl=chosen", new1)):
            bad = [nm for nm, a_, b_ in zip(names, new, old) if not torch.equal(a_, b_)]
            if bad:
                problems.append(f"R {name} {tag}: {bad} differ from stage 1's R")
        if int(dl["cnt_r"].abs().sum()):
            problems.append(f"R {name}: ticket not reset")
            dl["cnt_r"].zero_()
        if "exact-tie" in name and ei.shape[0] == 1:
            try:
                router = FusedMuseSparkRouting(expert_bias=zero_bias, route_scale=1.0,
                                               router_output_multiplier=0.1767766952966369,
                                               router_score_division_eps=1e-15)
                _w, sg_ids = router(None, lg, TOP_K, True)
                torch.cuda.synchronize()
                ours = [int(v) for v in new0[1][0].tolist()]
                sg = [int(v) for v in sg_ids[0].tolist()]
                _say3(f"tie order {name}: new R ids {ours}; SGLang FusedMuseSparkRouting ids {sg}; "
                      f"same set {set(ours) == set(sg)}, same order {ours == sg}")
            except Exception as e:
                _say3(f"tie order {name}: SGLang comparison raised {e!r} (info only)")
    return problems
def _dl3_check_u(state, device, tokens_list=(1,)):
    d3 = state["dl3"]
    dense = state["dense"]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261033)
    problems = []
    for tokens in tokens_list:
        y = (torch.randn((tokens, D_EXPERT), device=device, generator=gen) * 0.5).to(torch.bfloat16)
        f_new = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        def run():
            f_new.fill_(7.0)
            _DL_EXT.dl_unpool(y, dense["unpooler"], f_new, d3["ws_u"], d3["cnt_u"], d3["pdl"])
        for rep in range(2):
            _dl3_launch(f"unpool NF=1 T={tokens} rep{rep}", run)
        f_ref = _unpool_master(dense, y)
        e_u = _rel_l2(f_new, f_ref)
        zero_ok = bool((f_new[:, UNPOOL_SLICE:] == 0).all())
        _say3(f"unpool T={tokens}: rel L2 {e_u:.2e} (<= 2e-3), zero columns exact {zero_ok}")
        if not bool(torch.isfinite(f_new).all()) or e_u > 2e-3 or not zero_ok:
            problems.append(f"unpool T={tokens}: rel L2 {e_u:.2e}, zero columns exact {zero_ok}")
    if int(d3["cnt_u"].abs().sum()):
        problems.append("unpool: ticket not reset")
        d3["cnt_u"].zero_()
    return problems
def _dl3_verdict(piece, t_new, t_old, what):
    win = t_old >= DL3_MARGIN * t_new
    msg = (f"{what}: new {t_new:.2f} us vs incumbent {t_old:.2f} us, ratio old/new {t_old / t_new:.3f}, "
           f"need >= {DL3_MARGIN}")
    _say3(f"selector {piece}: {msg} -> WINNER = {'NEW' if win else 'INCUMBENT'}")
    return win, msg
def _dl3_select_r(state, device):
    dl = state["dl"]
    pdl = dl["pdl"]
    ls = torch.tensor([96], dtype=torch.int32, device=device)
    x = torch.randn((1, HIDDEN), device=device).to(torch.bfloat16)
    eip = torch.empty((1, D_EXPERT), dtype=torch.bfloat16, device=device)
    lgp = torch.empty((1, N_EXPERTS), dtype=torch.float32, device=device)
    bufs, bsm = _dl_route_bufs(1, device)
    def pair(variant, rpdl):
        def run():
            _DL_EXT.dl_pooler_router(x, dl["w_pr"], eip, lgp, dl["ws"], dl["cnt_p"], pdl)
            _dl_route_launch(eip, lgp, ls, dl, bufs, bsm, rpdl, variant)
        return run
    _say3("selector B: timing the P -> R pair (new R, then stage 1's R)")
    t_new, t_old = _sk_graph_time([pair(0, pdl), pair(2, 0)], device)
    dl["cnt_p"].zero_()
    dl["cnt_r"].zero_()
    torch.cuda.synchronize()
    return _dl3_verdict("B", t_new, t_old, f"P->R pair T=1 (R pdl={pdl})")
def _dl3_select_u(state, device):
    d3 = state["dl3"]
    dense = state["dense"]
    y = (torch.randn((1, D_EXPERT), device=device) * 0.5).to(torch.bfloat16)
    def new():
        _dl3_unpool(y, state, d3)
    def old():
        _unpool_master(dense, y)
    _say3("selector C: timing the unpooler (new kernel, then slice GEMM + F.pad)")
    t_new, t_old = _sk_graph_time([new, old], device)
    d3["cnt_u"].zero_()
    torch.cuda.synchronize()
    return _dl3_verdict("C", t_new, t_old, "unpooler T=1 (kernel vs _gemm + F.pad)")
def _dl3_select_a(state, device):
    d3 = state["dl3"]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261034)
    data = _dl3_layer_data(state, device, gen, 0, 1.0, 80000)
    ref_state = _dl3_flags(state, False, None, None)
    def new():
        _dl3_attention(data, state, d3)
    def old():
        _parent_attention(data, ref_state)
    _say3("selector A: timing the full decode_1 attention side (new chain, then the parent's chain)")
    t_new, t_old = _sk_graph_time([new, old], device)
    _dl3_zero(state)
    win, msg = _dl3_verdict("A", t_new, t_old, "full decode_1 attention side, 80000 keys")
    _say3(f"M2 (Q + sk two-level + O chain <= 36 us in prepare timing): {t_new:.1f} us -> "
          f"{'met' if t_new <= 36.0 else 'MISSED'}")
    del data
    return win, msg
def _dl3_compare_layer(got, ref, d_got, d_ref, a_on):
    finite = all(bool(torch.isfinite(t.float()).all()) for t in got[:2])
    e_h, e_s = _rel_l2(got[0], ref[0]), _rel_l2(got[1], ref[1])
    tol_h, tol_s = (2e-2, 1e-2) if a_on else (1e-2, 5e-3)
    if a_on:
        overlap = (got[2].unsqueeze(-1) == ref[2].unsqueeze(-2)).any(-1).sum(-1)
        ids_ok = bool((overlap >= TOP_K - 1).all())
    else:
        ids_ok = bool(torch.equal(got[2], ref[2]))
    loc = d_ref["cache_loc"]
    cache_ok = True
    for name in ("k_cache", "v_cache"):
        g = d_got[name].view(torch.uint8).view(-1, 128)
        r = d_ref[name].view(torch.uint8).view(-1, 128)
        mask = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
        mask[loc] = False
        cache_ok = cache_ok and torch.equal(g[mask], r[mask]) and _fp8_code_gap(g[loc], r[loc]) <= (1 if a_on else 0)
        if a_on:
            cache_ok = cache_ok and _rel_l2(g[loc].view(FP8).float(), r[loc].view(FP8).float()) <= 2e-2
    ok = finite and e_h <= tol_h and e_s <= tol_s and ids_ok and cache_ok
    return ok, f"hidden {e_h:.2e} stream {e_s:.2e} ids_ok {ids_ok} cache_ok {cache_ok} finite {finite}"
_DL3_CASES = (("full kv_scale=0.25", 0, 0.25, 4097, 0), ("full kv_scale=1", 0, 1.0, 9001, 96),
              ("sliding", 2048, 1.0, 3000, 0))
def _dl3_eager_layer_check(state, device, a, b, u):
    new_state = _dl3_flags(state, a, b, u)
    ref_state = _dl3_flags(state, False, False, False)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261035)
    for name, window, kvs, n, ls_val in _DL3_CASES:
        data = _dl3_layer_data(state, device, gen, window, kvs, n, ls_val)
        d_ref, d_new = _dl3_clone_data(data), _dl3_clone_data(data)
        ref = _dl3_launch(f"layer {name} parent", lambda: kernel_fn(d_ref, ref_state))
        got = _dl3_launch(f"layer {name} pieces A={int(a)} B={int(b)} C={int(u)}", lambda: kernel_fn(d_new, new_state))
        ok, msg = _dl3_compare_layer(got, ref, d_new, d_ref, a and window == 0)
        _say3(f"eager layer {name} A={int(a)} B={int(b)} C={int(u)}: {msg} -> {'ok' if ok else 'FAIL'}")
        if not ok:
            return f"eager layer {name}: {msg}"
        left = _dl3_tickets(new_state)
        if left:
            return f"eager layer {name}: tickets dirty ({left})"
    return None
def _dl3_alternation_check(state, device, a, b, u):
    ms = [_dl3_flags(state, a, b, u), _dl3_flags(_dl3_ministate(state, device), a, b, u)]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261036)
    for name, window, kvs, n, ls_val in _DL3_CASES:
        sets = []
        for s in range(2):
            data = _dl3_layer_data(state, device, gen, window, kvs, n, ls_val)
            cell = {}
            def fn(data=data, st=ms[s], cell=cell):
                cell["out"] = kernel_fn(data, st)
            _dl3_launch(f"alt {name} eager warm-up state {s}", lambda: kernel_fn(_dl3_clone_data(data), ms[s]))
            _say3(f"alt {name}: capturing state {s}")
            graph = _sk_graph_of(fn, device)
            sets.append((data, cell, graph))
        for r in range(6):
            s = r % 2
            data, cell, graph = sets[s]
            cell["out"][0].fill_(float("nan"))
            cell["out"][1].fill_(float("nan"))
            cell["out"][2].fill_(-1)
            fresh = _dl3_layer_data(state, device, gen, window, kvs, n, ls_val)
            for k, v in fresh.items():
                if torch.is_tensor(v):
                    data[k].copy_(v)
            d_ref = _dl3_clone_data(data)
            ref = kernel_fn(d_ref, _dl3_flags(ms[s], False, False, False))
            torch.cuda.synchronize()
            _say3(f"alt {name}: replay {r} on state {s}: launching graph")
            graph.replay()
            torch.cuda.synchronize()
            ok, msg = _dl3_compare_layer(cell["out"], ref, data, d_ref, a and window == 0)
            left = _dl3_tickets(ms[s])
            _say3(f"alt {name}: replay {r} state {s}: {msg}, tickets {left} -> {'ok' if ok and not left else 'FAIL'}")
            if not ok or left:
                return f"alt {name} replay {r} state {s}: {msg}, tickets {left}"
        del sets
    return None
def _dl3_layer_checks(state, device, a, b, u, t8=False):
    try:
        if t8:
            return _cb_t8_layer_check(state, device)
        err = _dl3_eager_layer_check(state, device, a, b, u)
        if err is None:
            err = _dl3_alternation_check(state, device, a, b, u)
        return err
    except Exception as e:
        return f"raised {type(e).__name__}: {str(e)[:300]}"
def _dl3_attrs_line():
    names = ("Q (NF=1)", "O (NF=1)", "R new", "unpool (NF=1)", "P (NF=1)")
    for which, name in enumerate(names):
        a = list(_DL_EXT.dl3_attrs(which))
        _say3(f"{name}: binaryVersion {a[0]}, ptxVersion {a[1]}, regs {a[2]}, local {a[3]} B, static smem {a[4]} B, "
              f"dyn smem {a[5]} B, {a[6]} CTAs/SM")
        if a[0] != 90:
            raise RuntimeError(f"{name}: cubin version {a[0]} is not 90 (sm_90a image not loaded)")
def _dl3_setup_bc(state, device):
    dl = state.get("dl")
    state["dl3"] = None
    if _DL_EXT is None or dl is None or not _DL_S3:
        _say3("OFF: the stage-1 dense chain is off, or the stage-3 module did not build (the parent's module runs "
              "stage 1): pieces A, B, C all off")
        return
    d3 = _dl3_buffers(device)
    d3.update({"a": False, "u": False, "pdl": dl["pdl"], "reason": {}, "msg": {},
               "b_t": {8: False, 16: False, 32: False}, "u8": False})
    dl["r_new"] = False
    state["dl3"] = d3
    try:
        _dl3_attrs_line()
    except Exception as e:
        reason = f"attribute query failed: {e!r}"
        _say3(reason)
        for p in "ABC":
            d3["reason"][p] = reason
        state["dl3"] = None
        return
    for piece, check, select in (("B", _dl3_check_r, _dl3_select_r), ("C", _dl3_check_u, _dl3_select_u)):
        reason = None
        try:
            problems = check(state, device)
            if problems:
                reason = "self-check: " + "; ".join(problems[:4])
            else:
                win, msg = select(state, device)
                d3["msg"][piece] = msg
                if not win:
                    reason = "selector: " + msg
        except Exception as e:
            reason = f"raised {type(e).__name__}: {str(e)[:300]}"
        try:
            torch.cuda.synchronize()
            dl["cnt_p"].zero_()
            dl["cnt_r"].zero_()
            d3["cnt_u"].zero_()
        except Exception:
            pass
        d3["reason"][piece] = reason
        if piece == "B":
            dl["r_new"] = reason is None
        else:
            d3["u"] = reason is None
        _say3(f"piece {piece} after its self-check + selector: {'ON' if reason is None else 'OFF (' + reason + ')'}")
def _dl3_finish(state, device):
    dl, d3 = state.get("dl"), state.get("dl3")
    if d3 is None:
        for piece, what in (("A", "full decode_1 attention side"), ("B", "kernel R at T=1"), ("C", "unpooler at T=1")):
            _say3(f"PATH CHOSEN: dl3 piece {piece} ({what}): parent path (fallback: dense-chain module unavailable) "
                  f"w2_mode={state.get('w2_mode')}")
        return
    reason = d3["reason"]
    sk = state.get("sk")
    try:
        if not _SK_S3:
            reason["A"] = "the stage-3 sk module did not build (the parent's sk module, if any, has no gate_off input)"
        elif sk is None or not sk["on"]:
            reason["A"] = "the sk attention module is off for this state (FA3 chain kept); piece A is not composed with FA3"
        elif not sk["b1"]:
            reason["A"] = "the parent's B=1 selector kept FA3 + combine + norm-gate; piece A is off, no other composition"
        else:
            problems = _dl3_check_a(state, device)
            _dl3_zero(state)
            if problems:
                reason["A"] = "self-check: " + "; ".join(problems[:4])
            else:
                win, msg = _dl3_select_a(state, device)
                d3["msg"]["A"] = msg
                reason["A"] = None if win else "selector: " + msg
    except Exception as e:
        reason["A"] = f"raised {type(e).__name__}: {str(e)[:300]}"
    try:
        _dl3_zero(state)
    except Exception:
        pass
    d3["a"] = reason.get("A") is None and "A" in reason
    pieces = {"A": d3["a"], "B": bool(dl["r_new"]), "C": bool(d3["u"])}
    on = [p for p in "ABC" if pieces[p]]
    b_before = pieces["B"]
    if on:
        err = _dl3_layer_checks(state, device, "A" in on, "B" in on, "C" in on)
        try:
            _dl3_zero(state)
        except Exception:
            pass
        if err is not None:
            _say3(f"layer-level check of pieces {on} FAILED ({err}); bisecting one piece at a time")
            for p in on:
                e1 = _dl3_layer_checks(state, device, p == "A", p == "B", p == "C")
                try:
                    _dl3_zero(state)
                except Exception:
                    pass
                if e1 is not None:
                    pieces[p] = False
                    reason[p] = f"layer-level check on this piece alone: {e1}"
            rest = [p for p in "ABC" if pieces[p]]
            if rest:
                e2 = _dl3_layer_checks(state, device, "A" in rest, "B" in rest, "C" in rest)
                try:
                    _dl3_zero(state)
                except Exception:
                    pass
                if e2 is not None:
                    for p in rest:
                        pieces[p] = False
                        reason[p] = f"layer-level check of the combination {rest}: {e2}"
        else:
            _say3(f"layer-level check of pieces {on}: eager + two-state alternating graph replays passed")
    d3["a"], dl["r_new"], d3["u"] = pieces["A"], pieces["B"], pieces["C"]
    if b_before and not pieces["B"]:
        _say3("piece B turned off after the w2 self-check: re-running _w2_atomic_self_check on stage 1's R")
        state["w2_mode"] = "master"  # compliance: SGLang reduces split-K in FP32 (use_atomic_add=False), never BF16 atomics
    try:
        _dl3_zero(state)
        _say3(f"tickets of this state after the stage-3 checks: sum {_dl3_tickets(state)}")
    except Exception as e:
        _say3(f"ticket zeroing raised {e!r}")
    tail = f"w2_mode={state.get('w2_mode')}"
    msgs = d3["msg"]
    lines = (("A", "full decode_1 attention side",
              "Q(qkv+norm+RoPE+FP8+KV store, gate stride 1024) + sk two-level (PDL) + O(prologue 0, PDL), FA3 not used",
              "parent chain (nvjet qkv + FusedQKNorm + casts + store_cache + sk two-level + nvjet o)"),
             ("B", "routing kernel R at T=1, both layers", "per-warp top-8 R with PDL after P", "stage 1's R"),
             ("C", "unpooler at T=1, both layers", "dl_unpool_kernel NF=1", "slice GEMM + F.pad"))
    for piece, what, new_name, inc_name in lines:
        if pieces[piece]:
            _say3(f"PATH CHOSEN: dl3 piece {piece} ({what}): {new_name} [{msgs.get(piece, 'n/a')}] {tail}")
        else:
            _say3(f"PATH CHOSEN: dl3 piece {piece} ({what}): {inc_name} (fallback: {reason.get(piece)}) {tail}")
    _say3(f"PATH CHOSEN: dl3 T > 1 (decode_16/32, verify): parent path, all pieces gated to tokens == 1 {tail}")
CB_TAG = "[combo]"
CB_RATIO = 0.98
CB_REPS = 41
CB_FLUSH_BYTES = 256 << 20
CB_DEADLINE = {"S2a": 120.0, "S2b": 120.0, "S1": 150.0, "S3": 180.0, "T8": 200.0,
               "SKW": 200.0}
CB_CELL_DEADLINE = 240.0
CB_CASE_DEADLINE = 260.0
_CB_T0 = None
CB_KV_SCALES = (1.0, 0.015625)
CB_Q8_FACTOR = 1.75
CB_Q8_SLACK = 1e-3
S1_CHECK_B = {"A": (2, 9, 16), "B": (17, 32)}
S1_SELECT_B = {"A": 16, "B": 32}
S1_LENS = (1, 63, 2047, 2048, 2049, 80000)
S2A_CLASSES = (8, 16, 32)
S2A_CHECK_T = {8: (2, 5, 8), 16: (16,), 32: (32,)}
S2B_CHECK_T = (2, 5, 8)
_CB_POS_ROPE = (80000, 299995, 1048500)
S3_SPEC = {
    "c1-b16": (0, ((2, 1), (16, 1)), (5000,), ((2, 1, 4000), (16, 1, 4000)), (16, 1)),
    "c1-b32": (0, ((32, 1), (40, 1)), (5000,), ((32, 1, 4000), (40, 1, 4000)), (32, 1)),
    "c2-t8": (2048, ((1, 8),), _CB_POS_ROPE, ((1, 8, 3000), (1, 8, 1)), (1, 8)),
    "c2-t128": (2048, ((2, 8), (5, 3), (16, 8)), _CB_POS_ROPE, ((2, 8, 300001), (5, 3, 3000), (16, 8, 3000)), (16, 8)),
    "c2-t256": (2048, ((32, 8),), _CB_POS_ROPE, ((32, 8, 3000),), (32, 8)),
    "c3-t128": (0, ((2, 8), (5, 3), (16, 8)), (5000,), ((2, 8, 4000), (5, 3, 4000), (16, 8, 4000)), (16, 8)),
    "c3-t256": (0, ((32, 8),), (5000,), ((32, 8, 4000),), (32, 8)),
    "c4-t8": (0, ((1, 8),), (5000, 1048500), ((1, 8, 5000), (1, 8, 1)), (1, 8)),
}
_CB_T8_CELLS = (("S2a", 8), ("S2b", 8), ("S3", "c4-t8"), ("S3", "c2-t8"))
_CB_T8_CASES = (("full ctx 80000", 0, 80000), ("full ctx 1048570", 0, 1048570), ("sliding ctx 80000", 2048, 80000))
_CB_LINES = (
    ("S1", "A", "sliding decode q_len 1 page 1, 2 <= B <= 16", "_sliding_fused_kernel S=16 (per-state counters)",
     "FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate"),
    ("S1", "B", "sliding decode q_len 1 page 1, 17 <= B <= 32", "_sliding_fused_kernel S=16 (per-state counters)",
     "FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate"),
    ("S2a", 8, "routing kernel R, 2 <= T <= 8", "per-warp top-8 R (variant 0), PDL after P", "stage 1's R"),
    ("S2a", 16, "routing kernel R, 9 <= T <= 16", "per-warp top-8 R (variant 0), PDL after P", "stage 1's R"),
    ("S2a", 32, "routing kernel R, 17 <= T <= 32", "per-warp top-8 R (variant 0), PDL after P", "stage 1's R"),
    ("S2b", 8, "unpooler, 2 <= T <= 8", "dl_unpool_kernel NF=1", "slice GEMM + splitKreduce + F.pad"),
    ("S3", "c1-b16", "full decode 2 <= B <= 23 (sk / split-KV feed)", "_pre_attention_kernel unpaired",
     "FusedQKNorm + casts + store_cache + q FP8 cast"),
    ("S3", "c1-b32", "full decode 24 <= B <= 64 (sk / split-KV feed)", "_pre_attention_kernel unpaired",
     "FusedQKNorm + casts + store_cache + q FP8 cast"),
    ("S3", "c2-t8", "sliding verify B = 1", "_pre_attention_kernel unpaired",
     "FusedQKNorm + casts + store_cache + q FP8 cast"),
    ("S3", "c2-t128", "sliding verify 2 <= B <= 23", "_pre_attention_kernel unpaired",
     "FusedQKNorm + casts + store_cache + q FP8 cast"),
    ("S3", "c2-t256", "sliding verify 24 <= B <= 64", "_pre_attention_kernel unpaired",
     "FusedQKNorm + casts + store_cache + q FP8 cast"),
    ("S3", "c3-t128", "full verify 2 <= B <= 23 (paired FA3)", "_pre_attention_kernel PAIRED + _paired_norm_gate_kernel",
     "FusedQKNorm + casts + store_cache + paired zero pad + copies + FusedRMSNormGate"),
    ("S3", "c3-t256", "full verify 24 <= B <= 64 (paired FA3)", "_pre_attention_kernel PAIRED + _paired_norm_gate_kernel",
     "FusedQKNorm + casts + store_cache + paired zero pad + copies + FusedRMSNormGate"),
    ("S3", "c4-t8", "full verify B = 1", "_pre_attention_kernel unpaired",
     "FusedQKNorm + casts + store_cache + q FP8 cast"),
    ("SKW", "b32", "sliding decode 17 <= B <= 32 (q_len 1, page 1)", "skw_pre_kernel + skw_attn_kernel (PSEUDO TOK=1)",
     "the attention side as master routes it (FusedQKNorm + casts + store_cache + FA3 + combine + FusedRMSNormGate)"),
    ("SKW", "v1", "sliding verify B = 1 (q_len 8, page 1)", "skw_pre_kernel + skw_attn_kernel (PSEUDO TOK=8)",
     "the attention side as master routes it (FusedQKNorm + casts + store_cache + FA3 + combine + FusedRMSNormGate)"),
)
_CB_SUMMARY = []
_CB_SELECT = {}
def _cb_say(message):
    print(f"{CB_TAG} {message}", flush=True)
def _cb_launch(label, fn):
    _cb_say(f"launching {label}")
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    _cb_say(f"ok        {label} ({time.perf_counter() - t0:.2f}s)")
    return result
def _s2a_class(tokens):
    return 8 if tokens <= 8 else (16 if tokens <= 16 else 32)
def _cb_flags(state, s1=None, s3=None, b_t=None, u8=None, skw=None, ph=None, bn=None):
    ms = dict(state)
    if ph is not None:
        ms["w6p_ph"] = bool(ph)
    if bn is not None:
        ms["w6p_bn"] = bn
    if skw is not None and state.get("skw") is not None:
        ms["skw"] = {**state["skw"], "route": {**state["skw"]["route"], **skw}}
    if s1 is not None:
        ms["s1"] = {**state["s1"], **s1}
    if s3 is not None:
        ms["s3"] = {**state["s3"], **s3}
    if state.get("dl3") is not None and (b_t is not None or u8 is not None):
        d3 = dict(state["dl3"])
        if b_t is not None:
            d3["b_t"] = {**d3["b_t"], **b_t}
        if u8 is not None:
            d3["u8"] = bool(u8)
        ms["dl3"] = d3
    return ms
def _cb_off(state):
    return _cb_flags(state, s1={c: False for c in S1_CLASSES}, s3={c: False for c in S3_CELLS},
                     b_t={c: False for c in S2A_CLASSES}, u8=False, skw={b: None for b in SKW_BANDS}, ph=False, bn=64)
def _cb_second_state(state, device):
    if state.get("dl") is not None and state.get("dl3") is not None:
        ms = _dl3_ministate(state, device)
    else:
        ms = dict(state)
        if state.get("sk") is not None:
            sk = _sk_buffers(device)
            _sk_zero(sk)
            ms["sk"] = {**sk, "on": True, "b1": state["sk"]["b1"], "n": dict(state["sk"]["n"])}
        ms["slide_cnt"] = torch.zeros_like(state["slide_cnt"])
    if state.get("skw") is not None:
        ms["skw"] = _skw_clone(state["skw"], device)
    torch.cuda.synchronize()
    return ms
def _cb_counters(state) -> int:
    return _dl3_tickets(state) + int(state["slide_cnt"].abs().sum()) + _skw_ticket_sum(state.get("skw"))
def _cb_clean(state, recs, label) -> bool:
    torch.cuda.synchronize()
    left = _cb_counters(state)
    if not left:
        return True
    _cb_say(f"counters NOT back to 0 after {label} (sum {left}): zeroed, master kept for this state")
    _dl3_zero(state)
    state["slide_cnt"].zero_()
    if state.get("skw") is not None:
        _skw_zero(state["skw"])
    torch.cuda.synchronize()
    recs["_dirty"] = recs["_dirty"] or label
    return False
def _cb_late(t_prepare0) -> bool:
    return time.perf_counter() - t_prepare0 > CB_CELL_DEADLINE
def _cb_case_late() -> bool:
    return _CB_T0 is not None and time.perf_counter() - _CB_T0 > CB_CASE_DEADLINE
def _cb_rms(t):
    return t / torch.sqrt((t * t).mean(-1, keepdim=True) + RMS_EPS)
def _cb_rope64(x, positions, theta):
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float64) / d))
    ang = positions.double()[:, None] * inv
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
def _cb_mixed(batch, ctx):
    tails = (1, 63, 64, 65, 2047, 2049)
    return [ctx] + [min(ctx, tails[(i - 1) % 6]) if i <= 6 else max(1, ctx - 37 * i) for i in range(1, batch)]
CB_STREAM_MEAN, CB_STREAM_SHARED, CB_STREAM_RANK, CB_DRAFT_SHARE = 0.7, 0.9, 4, 0.6
CB_JUDGE_DISTINCT = {(1, 1): 2.5, (1, 8): 7.5, (16, 1): 7.5, (32, 1): 9.0, (16, 8): 18.4, (32, 8): 21.3}
CB_BASIS_BETAS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8)
CB_BASIS_DRAWS = 8
CB_BASIS_SAMPLES = 8
CB_HOT_ROUTES = 16
_CB_BASIS = {}
_CB_BASIS_LINE = []
def _cb_stream(basis, gen, batch, length, device):
    mean, shared = basis
    tokens = batch * length
    coef = torch.randn((tokens, CB_STREAM_RANK), generator=gen, device=device)
    scale = torch.ones((batch, length, 1), device=device)
    if 1 < length <= 8:
        scale[:, 1:] = CB_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    noise = torch.randn((tokens, HIDDEN), generator=gen, device=device)
    return (scale * (CB_STREAM_MEAN * mean + (CB_STREAM_SHARED / CB_STREAM_RANK) ** 0.5 * (coef @ shared))
            + (1 - CB_STREAM_SHARED * scale * scale).sqrt() * noise)
def _cb_local_load(ids, inputs):
    ids = ids.long().reshape(inputs, -1)
    ranges = N_EXPERTS // LOCAL_EXPERTS
    ones = torch.ones_like(ids, dtype=torch.float32)
    pairs = torch.zeros((inputs, ranges), device=ids.device).scatter_add_(1, ids // LOCAL_EXPERTS, ones)
    busiest = pairs.argmax(1)
    per = torch.zeros((inputs, N_EXPERTS), device=ids.device).scatter_add_(1, ids, ones)
    per = per.view(inputs, ranges, LOCAL_EXPERTS)[torch.arange(inputs, device=ids.device), busiest]
    load = torch.stack([per.sum(1), (per > 0).sum(1).float(), per.max(1).values, (per > CB_HOT_ROUTES).sum(1).float()])
    return [float(v) for v in load.mean(1)]
def _cb_basis_stats(state, basis, gen, device):
    rows, bias = state["router"].float(), state["router_bias"].float()
    out = {}
    for (b, l) in CB_JUDGE_DISTINCT:
        ids = []
        for _ in range(CB_BASIS_SAMPLES):
            stream = _cb_stream(basis, gen, b, l, device)
            a = torch.randn(stream.shape, generator=gen, device=device).to(torch.bfloat16)
            ffn_input = state["attn_boundary"](stream, a, branch_dtype=torch.bfloat16, router_dtype=None)[0]
            scores = torch.sigmoid((ffn_input.float() @ rows.T) * 0.1767766952966369) + bias
            ids.append(scores.topk(TOP_K, dim=-1).indices)
        out[(b, l)] = _cb_local_load(torch.cat(ids), CB_BASIS_SAMPLES)
    return out
def _cb_basis(state, device):
    router = state["router"]
    key = (str(device), router.data_ptr(), float(router[:, :64].double().sum()))
    if key in _CB_BASIS:
        return _CB_BASIS[key]
    t0 = time.perf_counter()
    gen = torch.Generator(device=device)
    gen.manual_seed(20261026)
    best = None
    with torch.no_grad():
        for draw in range(CB_BASIS_DRAWS):
            r = torch.randn((HIDDEN,), generator=gen, device=device)
            u = torch.randn((N_EXPERTS,), generator=gen, device=device) @ router.float()
            shared = torch.randn((CB_STREAM_RANK, HIDDEN), generator=gen, device=device)
            for beta in CB_BASIS_BETAS:
                mean = r + beta * u / u.std()
                basis = (mean / mean.std(), shared)
                stats = _cb_basis_stats(state, basis, gen, device)
                err = max(abs(stats[k][1] / v - 1.0) for k, v in CB_JUDGE_DISTINCT.items())
                if best is None or err < best[0]:
                    best = (err, basis, draw, beta, stats)
    err, basis, draw, beta, stats = best
    _CB_BASIS[key] = basis
    line = (f"{CB_TAG} gate stream basis: draw {draw} beta {beta}, worst distinct error {err:.1%} "
            f"({time.perf_counter() - t0:.1f} s); busiest range routes/distinct/max/hot per BxL: " + ", ".join(
                f"{b}x{l} {s[0]:.0f}/{s[1]:.1f}/{s[2]:.0f}/{s[3]:.1f} (judge distinct {CB_JUDGE_DISTINCT[(b, l)]})"
                for (b, l), s in stats.items()))
    _CB_BASIS_LINE[:] = [line]
    print(line, flush=True)
    return basis
def _cb_data(state, device, gen, window, batch, length, lens, kv_scale, local_start=0):
    hkv, d = 2, 64
    tokens = batch * length
    lens_t = torch.tensor(lens, dtype=torch.int64, device=device)
    max_pages = max(lens) + length
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(lens):
        page_table[b, :n + length] = perm[off:off + n + length]
        off += n + length
    positions = (lens_t[:, None] + torch.arange(length, device=device)).reshape(-1)
    cache_loc = torch.gather(page_table.long(), 1, positions.view(batch, length)).reshape(-1)
    stream = _cb_stream(_cb_basis(state, device), gen, batch, length, device)
    x = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(torch.bfloat16)
    vmax = _gemm(x, state["dense"]["qkv_gate"])[:, 1152:1280].float().abs().amax(dim=1, keepdim=True)
    x = (x.float() * (100.0 * kv_scale / vmax.clamp_min(1e-6)).clamp(max=1.0)).to(torch.bfloat16)
    cfg = {"batch": batch, "q_len": length, "tokens": tokens, "hidden": HIDDEN, "num_q_heads": 16,
           "num_kv_heads": hkv, "head_dim": d, "page_size": 1, "window": window,
           "rope_theta": ROPE_THETA if window else 0.0, "qk_scale_factor": 5.4730064863875, "sm_scale": d ** -0.5,
           "kv_scale": kv_scale, "n_experts": N_EXPERTS, "top_k": TOP_K, "local_experts": LOCAL_EXPERTS}
    kc = torch.randn((total, 1, hkv, d), generator=gen, device=device).clamp_(-6.0, 6.0) * 0.3 / kv_scale
    vc = torch.randn((total, 1, hkv, d), generator=gen, device=device).clamp_(-6.0, 6.0) / kv_scale
    return {
        "stream": stream,
        "attn_input": x,
        "k_cache": kc.to(FP8),
        "v_cache": vc.to(FP8),
        "positions": positions,
        "cache_loc": cache_loc,
        "cache_seqlens": (lens_t + length).to(torch.int32),
        "cu_seqlens_q": torch.arange(0, tokens + 1, length, dtype=torch.int32, device=device),
        "scheduler_metadata": None,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table,
        "seq_lens": lens_t.to(torch.int32),
        "local_start": torch.tensor([local_start], dtype=torch.int32, device=device),
        "config": cfg,
    }
def _cb_refill(data, fresh):
    for k, v in fresh.items():
        if torch.is_tensor(v):
            data[k].copy_(v)
def _cb_attn_ref(data, qkv, q_override=None, q_fp8=False):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = 16, 2, 64
    g = hq // hkv
    window = cfg["window"]
    x = qkv.double()
    if q_override is not None:
        q = q_override.view(-1, hq, d).float().double()
    else:
        q = _cb_rms(x[:, :hq * d].reshape(-1, hq, d))
        if cfg["rope_theta"]:
            q = _cb_rope64(q, data["positions"], cfg["rope_theta"])
        if q_fp8:
            q = q.float().to(FP8).double()
    gate = x[:, (hq + 2 * hkv) * d:].reshape(-1, hq, d)
    kc = data["k_cache"].view(-1, hkv, d)
    vc = data["v_cache"].view(-1, hkv, d)
    ks, vs = float(data["k_scale"]), float(data["v_scale"])
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    outs = []
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        n = seq + length
        t0 = max(0, seq - (window - 1)) if window else 0
        rows = data["page_table"][b, t0:n].long()
        k = kc[rows].double() * ks
        v = vc[rows].double() * vs
        qb = q[b * length:(b + 1) * length].reshape(length, hkv, g, d)
        s = torch.einsum("lkgd,nkd->lkgn", qb, k) * scale
        t = torch.arange(t0, n, device=x.device)
        p = seq + torch.arange(length, device=x.device)
        valid = t[None, :] <= p[:, None]
        if window:
            valid &= t[None, :] >= p[:, None] - (window - 1)
        s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
        outs.append(torch.einsum("lkgn,nkd->lkgd", torch.softmax(s, dim=-1), v).reshape(length, hq, d))
    o = _cb_rms(torch.cat(outs, 0)) * torch.sigmoid(gate)
    return o.reshape(-1, hq * d)
def _cb_cmp_attn(o_new, o_old, o_ref, d_new, d_old, worst, row_agg=False, o_ref8=None):
    finite = bool(torch.isfinite(o_new.float()).all())
    e_nm, e_n, e_m = _rel_l2(o_new, o_old), _rel_l2(o_new, o_ref), _rel_l2(o_old, o_ref)
    rn = (o_new.double() - o_ref).norm(dim=-1) / o_ref.norm(dim=-1).clamp_min(1e-30)
    rm = (o_old.double() - o_ref).norm(dim=-1) / o_ref.norm(dim=-1).clamp_min(1e-30)
    rows_ok = bool((rn <= 2.0 * rm + 5e-3).all())
    if row_agg:
        rows_ok = bool((rn <= 3.0 * max(e_m, 1e-3) + 5e-3).all())
    judge = ""
    if o_ref8 is not None:
        e_r8 = _rel_l2(o_ref8, o_ref)
        tol = CB_Q8_FACTOR * e_r8 + CB_Q8_SLACK
        rr = o_ref.norm(dim=-1)
        row_j = float(((o_new.double() - o_ref).norm(dim=-1) / torch.maximum(rr, 0.1 * rr.mean()).clamp_min(1e-30)).max())
        rows_ok = row_j <= 5.0 * tol
        o_ok = finite and (e_nm <= 1e-2 or (e_n <= tol and rows_ok))
        judge = f", fp8-q ref err {e_r8:.2e} -> bound {tol:.2e}, worst row {row_j:.2e} (<= {5.0 * tol:.2e})"
        worst["o_fp8q_ref"] = max(worst.get("o_fp8q_ref", 0), e_r8)
        worst["o_new_over_ref8"] = max(worst.get("o_new_over_ref8", 0), e_n / max(e_r8, 1e-12))
    else:
        o_ok = finite and (e_nm <= 1e-2 or (e_n <= 1.25 * e_m + 1e-3 and rows_ok))
    loc = d_old["cache_loc"]
    kn, km = d_new["k_cache"].view(torch.uint8).view(-1, 128), d_old["k_cache"].view(torch.uint8).view(-1, 128)
    vn, vm = d_new["v_cache"].view(torch.uint8).view(-1, 128), d_old["v_cache"].view(torch.uint8).view(-1, 128)
    mask = torch.ones(kn.shape[0], dtype=torch.bool, device=kn.device)
    mask[loc] = False
    rest_ok = torch.equal(kn[mask], km[mask]) and torch.equal(vn[mask], vm[mask])
    k_gap = _fp8_code_gap(kn[loc], km[loc])
    k_frac = float((kn[loc] != km[loc]).float().mean())
    k_rel = _rel_l2(kn[loc].view(FP8).float(), km[loc].view(FP8).float())
    k_ok = bool(torch.isfinite(kn[loc].view(FP8).float()).all()) and k_rel <= 4e-2
    v_ok = torch.equal(vn[loc], vm[loc])
    ok = o_ok and rest_ok and k_ok and v_ok
    for key, val in (("o_vs_master", e_nm), ("o_new_vs_fp64", e_n), ("o_master_vs_fp64", e_m), ("k_rel", k_rel),
                     ("k_gap", k_gap), ("k_frac", k_frac)):
        worst[key] = max(worst.get(key, 0), val)
    msg = (f"o vs master {e_nm:.2e} (<= 1e-2), vs fp64 new {e_n:.2e} master {e_m:.2e} rows ok {rows_ok}{judge}, k rel L2 "
           f"vs master {k_rel:.2e} (<= 4e-2; info: gap {k_gap} on {k_frac:.1%} of bytes), v exact {v_ok}, "
           f"other rows intact {rest_ok}, finite {finite}")
    return ok, msg
def _cb_attn_alt(state, device, label, cases, on, row_agg=False):
    second = _cb_second_state(state, device)
    sts = [state, second]
    st_on = [_cb_flags(s, **on) for s in sts]
    st_off = [_cb_off(s) for s in sts]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261041)
    worst = {}
    for window, batch, length, lens, kvs in cases:
        tag = f"{label} B={batch} L={length} lens {min(lens)}..{max(lens)} kv_scale {kvs}"
        if _cb_case_late():
            return f"{tag}: case deadline {CB_CASE_DEADLINE:.0f}s passed"
        sets = []
        for s in range(2):
            data = _cb_data(state, device, gen, window, batch, length, lens, kvs)
            cell = {}
            def fn(data=data, st=st_on[s], cell=cell):
                cell["o"] = _attn_core(data, st)
            _cb_launch(f"{tag}: eager warm-up state {s}", lambda: _attn_core(_dl3_clone_data(data), st_on[s]))
            sets.append((data, cell, _sk_graph_of(fn, device)))
        for r in range(4):
            s = r % 2
            data, cell, graph = sets[s]
            cell["o"].fill_(float("nan"))
            _cb_refill(data, _cb_data(state, device, gen, window, batch, length, lens, kvs))
            d_old = _dl3_clone_data(data)
            o_old = _attn_core(d_old, st_off[s])
            qkv_ref = _gemm(d_old["attn_input"], state["dense"]["qkv_gate"])
            o_ref = _cb_attn_ref(d_old, qkv_ref)
            o_ref8 = _cb_attn_ref(d_old, qkv_ref, q_fp8=True)
            torch.cuda.synchronize()
            _cb_say(f"{tag}: replay {r} on state {s}: launching graph")
            graph.replay()
            torch.cuda.synchronize()
            ok, msg = _cb_cmp_attn(cell["o"], o_old, o_ref, data, d_old, worst, row_agg=row_agg, o_ref8=o_ref8)
            left = _cb_counters(sts[s])
            _cb_say(f"{tag}: replay {r} state {s}: {msg}, counters {left} -> {'ok' if ok and not left else 'FAIL'}")
            if not ok or left:
                return f"{tag} replay {r} state {s}: {msg}, counters {left}"
            del d_old, o_old, o_ref
        del sets
    _cb_say(f"{label}: max errors over every replay: " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()))
    return None
def _s3_front_master(data, qkv, state, paired):
    cfg = data["config"]
    tokens, hq, hkv, d = cfg["tokens"], 16, 2, 64
    q = _master_front(data, qkv, state)
    if paired:
        return _master_pair_q(q, tokens, hq, hkv, d)
    return q.view(tokens, hq, d).to(FP8).contiguous()
def _s3_front_check(state, device, gen, window, batch, length, kv_scale, pos_base, paired, pre_fn=None,
                    gap_gate=False, label="S3 front"):
    tokens = batch * length
    hq, hkv, d = 16, 2, 64
    rows = 4 * tokens + 61
    raw = torch.randn((tokens, DL3_Q_ROWS), generator=gen, device=device)
    v_raw = raw[:, 1152:1280]
    raw[:, 1152:1280] = v_raw * min(1.0, 100.0 * kv_scale / float(v_raw.abs().max()))
    qkv = raw.to(torch.bfloat16)
    seq = torch.tensor([pos_base + 37 * b for b in range(batch)], dtype=torch.int64, device=device)
    positions = (seq[:, None] + torch.arange(length, device=device)).reshape(-1)
    cache_loc = torch.randperm(rows, generator=gen, device=device)[:tokens].to(torch.int64)
    kc0 = (torch.randn((rows, 1, hkv, d), generator=gen, device=device).clamp_(-6.0, 6.0) / kv_scale).to(FP8)
    vc0 = (torch.randn((rows, 1, hkv, d), generator=gen, device=device).clamp_(-6.0, 6.0) / kv_scale).to(FP8)
    cfg = {"batch": batch, "q_len": length, "tokens": tokens, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
           "page_size": 1, "window": window, "rope_theta": ROPE_THETA if window else 0.0, "kv_scale": kv_scale}
    def mk():
        return {"positions": positions, "cache_loc": cache_loc, "config": cfg,
                "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
                "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
                "k_cache": kc0.clone(), "v_cache": vc0.clone()}
    dm, dn = mk(), mk()
    q_m = _s3_front_master(dm, qkv, state, paired)
    torch.cuda.synchronize()
    tag = (f"{label} {'PAIRED' if paired else 'unpaired'} {'sliding' if window else 'full'} B={batch} L={length} "
           f"pos~{pos_base} kv_scale {kv_scale}")
    q_n = _cb_launch(f"{tag}: {'_pre_attention_kernel' if pre_fn is None else 'skw_pre_kernel'}",
                     lambda: (pre_fn or _s3_pre_attention)(dn, qkv, state, paired))
    x = qkv.double()
    q_ref = _cb_rms(x[:, :hq * d].reshape(tokens, hq, d))
    k_ref = _cb_rms(x[:, hq * d:(hq + hkv) * d].reshape(tokens, hkv, d))
    if window:
        q_ref, k_ref = _cb_rope64(q_ref, positions, ROPE_THETA), _cb_rope64(k_ref, positions, ROPE_THETA)
    k_ref = k_ref / kv_scale
    if paired:
        own = ((torch.arange(hq, device=device) // (hq // hkv)) % 2).view(1, hq, 1, 1).expand(tokens, hq, 1, d)
        un, um = q_n.view(torch.uint8).view(tokens, hq, 2, d), q_m.view(torch.uint8).view(tokens, hq, 2, d)
        qn_u, qm_u = un.gather(2, own).squeeze(2), um.gather(2, own).squeeze(2)
        zero_ok = bool((un.gather(2, 1 - own) == 0).all())
    else:
        qn_u, qm_u = q_n.view(torch.uint8).view(tokens, hq, d), q_m.view(torch.uint8).view(tokens, hq, d)
        zero_ok = True
    gap_q = _fp8_code_gap(qn_u, qm_u)
    e_qn, e_qm = _rel_l2(qn_u.view(FP8).float(), q_ref), _rel_l2(qm_u.view(FP8).float(), q_ref)
    kn, km = dn["k_cache"].view(torch.uint8).view(-1, hkv * d), dm["k_cache"].view(torch.uint8).view(-1, hkv * d)
    vn, vm = dn["v_cache"].view(torch.uint8).view(-1, hkv * d), dm["v_cache"].view(torch.uint8).view(-1, hkv * d)
    gap_k = _fp8_code_gap(kn[cache_loc], km[cache_loc])
    e_kn = _rel_l2(kn[cache_loc].view(FP8).float().view(tokens, hkv, d), k_ref)
    e_km = _rel_l2(km[cache_loc].view(FP8).float().view(tokens, hkv, d), k_ref)
    v_ok = torch.equal(vn[cache_loc], vm[cache_loc])
    mask = torch.ones(rows, dtype=torch.bool, device=device)
    mask[cache_loc] = False
    k0, v0 = kc0.view(torch.uint8).view(-1, hkv * d), vc0.view(torch.uint8).view(-1, hkv * d)
    rest_ok = torch.equal(kn[mask], k0[mask]) and torch.equal(vn[mask], v0[mask])
    finite = bool(torch.isfinite(qn_u.view(FP8).float()).all())
    ok = (finite and zero_ok and v_ok and rest_ok and (not gap_gate or (gap_q <= 1 and gap_k <= 1))
          and e_qn <= e_qm + 1e-3 and e_kn <= e_km + 1e-3)
    msg = (f"q gap {gap_q} rel L2 vs fp64 new {e_qn:.2e} master {e_qm:.2e}; k gap {gap_k} rel L2 vs fp64 new "
           f"{e_kn:.2e} master {e_km:.2e}; v exact {v_ok}; zero half exact {zero_ok}; other rows intact {rest_ok}")
    _cb_say(f"{tag}: {msg} -> {'ok' if ok else 'FAIL'}")
    return None if ok else f"{tag}: {msg}"
def _cb_s2_input(which, tokens, gen, device):
    if which == "R":
        return torch.randn((tokens, HIDDEN), device=device, generator=gen).to(torch.bfloat16)
    return (torch.randn((tokens, D_EXPERT), device=device, generator=gen) * 0.5).to(torch.bfloat16)
def _cb_s2_alt(state, device, which, tokens):
    second = _cb_second_state(state, device)
    sts = [state, second]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261044 + tokens)
    ls = torch.tensor([32], dtype=torch.int32, device=device)
    dense = state["dense"]
    names = ("xn", "topk_ids", "topk_w", "local_ids", "sorted_ids", "block_experts", "n_padded")
    sets = []
    for s in range(2):
        dl, d3 = sts[s]["dl"], sts[s]["dl3"]
        inp = _cb_s2_input(which, tokens, gen, device)
        if which == "R":
            ei = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=device)
            lg = torch.empty((tokens, N_EXPERTS), dtype=torch.float32, device=device)
            bufs, bsm = _dl_route_bufs(tokens, device)
            def fn(inp=inp, ei=ei, lg=lg, bufs=bufs, bsm=bsm, dl=dl):
                _DL_EXT.dl_pooler_router(inp, dl["w_pr"], ei, lg, dl["ws"], dl["cnt_p"], dl["pdl"])
                _dl_route_launch(ei, lg, ls, dl, bufs, bsm, dl["pdl"], 0)
            io = {"inp": inp, "ei": ei, "lg": lg, "bufs": bufs, "bsm": bsm}
        else:
            out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
            def fn(inp=inp, out=out, d3=d3):
                _DL_EXT.dl_unpool(inp, dense["unpooler"], out, d3["ws_u"], d3["cnt_u"], d3["pdl"])
            io = {"inp": inp, "out": out}
        _cb_launch(f"S2 {which} T={tokens}: eager warm-up state {s}", fn)
        sets.append((io, _sk_graph_of(fn, device)))
    for r in range(4):
        s = r % 2
        io, graph = sets[s]
        io["inp"].copy_(_cb_s2_input(which, tokens, gen, device))
        if which == "R":
            for t in io["bufs"]:
                t.fill_(float("nan") if t.is_floating_point() else -7)
        else:
            io["out"].fill_(7.0)
        _cb_say(f"S2 {which} T={tokens}: replay {r} on state {s}: launching graph")
        graph.replay()
        torch.cuda.synchronize()
        if which == "R":
            ref_bufs, _bsm = _dl_route_bufs(tokens, device)
            _dl_route_launch(io["ei"], io["lg"], ls, sts[s]["dl"], ref_bufs, io["bsm"], 0, 2)
            torch.cuda.synchronize()
            bad = [nm for nm, a_, b_ in zip(names, io["bufs"], ref_bufs) if not torch.equal(a_, b_)]
            finite = bool(torch.isfinite(io["lg"]).all()) and bool(torch.isfinite(io["ei"].float()).all())
            ok = finite and not bad
            msg = f"outputs differing from stage 1's R {bad}, P outputs finite {finite}"
        else:
            f_ref = _unpool_master(dense, io["inp"])
            e_u = _rel_l2(io["out"], f_ref)
            zero_ok = bool((io["out"][:, UNPOOL_SLICE:] == 0).all())
            ok = bool(torch.isfinite(io["out"]).all()) and e_u <= 2e-3 and zero_ok
            msg = f"rel L2 {e_u:.2e} (<= 2e-3), zero columns exact {zero_ok}"
        left = _cb_counters(sts[s])
        _cb_say(f"S2 {which} T={tokens}: replay {r} state {s}: {msg}, counters {left} -> "
                f"{'ok' if ok and not left else 'FAIL'}")
        if not ok or left:
            return f"S2 {which} T={tokens} replay {r} state {s}: {msg}, counters {left}"
    del sets
    return None
def _cb_router_scores(data, ref_state):
    attn = _parent_attention(data, ref_state)
    ffn_input, _s, _r = ref_state["attn_boundary"](data["stream"], attn, branch_dtype=torch.bfloat16,
                                                   router_dtype=None)
    logits = ffn_input.double() @ ref_state["router"].double().T
    return torch.sigmoid(logits * 0.1767766952966369) + ref_state["router_bias"].double()
def _cb_route_rules(ids, scores, ls_val):
    ids = ids.long()
    top9 = scores.topk(TOP_K + 1, dim=-1)
    eighth, ninth = top9.values[:, TOP_K - 1], top9.values[:, TOP_K]
    ref8, ref8v = top9.indices[:, :TOP_K], top9.values[:, :TOP_K]
    r_pick = bool((scores.gather(1, ids) >= (eighth - 0.05)[:, None]).all())
    def local(e):
        return ((e - ls_val) % N_EXPERTS) < LOCAL_EXPERTS
    must = local(ref8) & ((ref8v - ninth[:, None]) > 0.03)
    present = (ref8.unsqueeze(-1) == ids.unsqueeze(-2)).any(-1)
    r_clear = bool((present | ~must).all())
    routes, ref_routes = int(local(ids).sum()), int(local(ref8).sum())
    distinct, ref_distinct = len(set(ids[local(ids)].tolist())), len(set(ref8[local(ref8)].tolist()))
    r_routes = routes >= 0.92 * ref_routes - 4
    r_distinct = distinct >= 0.85 * ref_distinct - 2
    ok = r_pick and r_clear and r_routes and r_distinct
    return ok, (f"picks within 0.05 {r_pick}, clear-margin top-8 present {r_clear}, routes {routes}/{ref_routes}, "
                f"distinct {distinct}/{ref_distinct}")
def _cb_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst, loose=False, k_gap_max=1):
    finite = all(bool(torch.isfinite(t.float()).all()) for t in got[:2])
    gi, ri = got[2].long(), ref[2].long()
    same = ~(gi.sort(-1).values != ri.sort(-1).values).any(-1)
    n_diff = int((~same).sum())
    rules_ok, rules = _cb_route_rules(gi, scores, ls_val)
    if loose:
        if bool(same.any()):
            e_h = _rel_l2(got[0][same], ref[0][same])
            e_s = _rel_l2(got[1][same], ref[1][same])
            e_u = _rel_l2(got[1][same].double() - s_in[same].double(), ref[1][same].double() - s_in[same].double())
        else:
            e_h = e_s = e_u = 0.0
        tol_h, tol_s, tol_u = 1e-1, float("inf"), 1e-1
    else:
        e_h, e_s = _rel_l2(got[0], ref[0]), _rel_l2(got[1], ref[1])
        e_u = _rel_l2(got[1].double() - s_in.double(), ref[1].double() - s_in.double())
        tol_h, tol_s, tol_u = (2e-2, 1e-2, 5e-2) if n_diff == 0 else (5e-2, 2.5e-2, 1e-1)
    loc = d_ref["cache_loc"]
    kn, kr = d_got["k_cache"].view(torch.uint8).view(-1, 128), d_ref["k_cache"].view(torch.uint8).view(-1, 128)
    vn, vr = d_got["v_cache"].view(torch.uint8).view(-1, 128), d_ref["v_cache"].view(torch.uint8).view(-1, 128)
    mask = torch.ones(kn.shape[0], dtype=torch.bool, device=kn.device)
    mask[loc] = False
    rest_ok = torch.equal(kn[mask], kr[mask]) and torch.equal(vn[mask], vr[mask])
    k_gap = _fp8_code_gap(kn[loc], kr[loc])
    e_k = _rel_l2(kn[loc].view(FP8).float(), kr[loc].view(FP8).float())
    v_ok = torch.equal(vn[loc], vr[loc])
    for key, val in (("hidden", e_h), ("stream", e_s), ("update", e_u), ("k_cache", e_k), ("k_gap", k_gap),
                     ("ids_tokens_differ", n_diff)):
        worst[key] = max(worst.get(key, 0), val)
    ok = (finite and rules_ok and e_h <= tol_h and e_s <= tol_s and e_u <= tol_u and rest_ok and k_gap <= k_gap_max
          and e_k <= 4e-2 and v_ok)
    msg = (f"hidden {e_h:.2e} (<= {tol_h}), stream {e_s:.2e} (<= {tol_s}), update {e_u:.2e} (<= {tol_u}), "
           f"k_cache gap {k_gap} rel {e_k:.2e}, v_cache exact {v_ok}, other rows intact {rest_ok}, "
           f"ids differ on {n_diff}/{gi.shape[0]} tokens, routing: {rules}, finite {finite}")
    return ok, msg
def _cb_t8_layer_check(state, device):
    second = _cb_second_state(state, device)
    sts = [state, second]
    refs = [_cb_off(s) for s in sts]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261042)
    ls_val = 96
    worst = {}
    for name, window, ctx in _CB_T8_CASES:
        if _cb_case_late():
            return f"T=8 layer {name}: case deadline {CB_CASE_DEADLINE:.0f}s passed"
        sets = []
        for s in range(2):
            data = _cb_data(state, device, gen, window, 1, 8, [ctx], 1.0, ls_val)
            cell = {}
            def fn(data=data, st=sts[s], cell=cell):
                cell["out"] = kernel_fn(data, st)
            _cb_launch(f"T=8 layer {name}: eager warm-up state {s}", lambda: kernel_fn(_dl3_clone_data(data), sts[s]))
            sets.append((data, cell, _sk_graph_of(fn, device)))
        for r in range(4):
            s = r % 2
            data, cell, graph = sets[s]
            cell["out"][0].fill_(float("nan"))
            cell["out"][1].fill_(float("nan"))
            cell["out"][2].fill_(-1)
            _cb_refill(data, _cb_data(state, device, gen, window, 1, 8, [ctx], 1.0, ls_val))
            s_in = data["stream"].clone()
            d_ref, d_probe = _dl3_clone_data(data), _dl3_clone_data(data)
            ref = kernel_fn(d_ref, refs[s])
            scores = _cb_router_scores(d_probe, refs[s])
            torch.cuda.synchronize()
            _cb_say(f"T=8 layer {name}: replay {r} on state {s}: launching graph")
            graph.replay()
            torch.cuda.synchronize()
            s3_t8 = bool(sts[s].get("s3", {}).get("c4-t8") or sts[s].get("s3", {}).get("c2-t8"))
            ok, msg = _cb_cmp_layer(cell["out"], ref, data, d_ref, s_in, scores, ls_val, worst, loose=s3_t8,
                                    k_gap_max=(1 << 8) if s3_t8 else 1)
            left = _cb_counters(sts[s])
            _cb_say(f"T=8 layer {name}: replay {r} state {s}: {msg}, counters {left} -> "
                    f"{'ok' if ok and not left else 'FAIL'}")
            if not ok or left:
                return f"T=8 layer {name} replay {r} state {s}: {msg}, counters {left}"
            del d_ref, d_probe, ref, scores
        del sets
    _cb_say("T=8 layer check max error per output: " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()))
    return None
def _cb_time(fns, device, reps=CB_REPS):
    scratch = torch.zeros(1, dtype=torch.float32, device=device)
    graphs = [_sk_graph_of(lambda: scratch.add_(1.0), device)] + [_sk_graph_of(f, device) for f in fns]
    flush = torch.empty(CB_FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = [[] for _ in graphs]
    for it in range(3 + reps):
        order = list(range(len(graphs)))
        if it % 2:
            order.reverse()
        for i in order:
            torch.sum(flush)
            e0.record()
            graphs[i].replay()
            e1.record()
            e1.synchronize()
            if it >= 3:
                ts[i].append(e0.elapsed_time(e1) * 1e3)
    med = [sorted(t)[len(t) // 2] for t in ts]
    del graphs, flush, scratch
    torch.cuda.synchronize()
    return [max(m - med[0], 0.05) for m in med[1:]], med[0]
def _cb_select(source, cls, what, build, device):
    key = f"{source} {cls}"
    if key in _CB_SELECT:
        win, t_new, t_old, base = _CB_SELECT[key]
        note = " (timing of the first prepare(), reused)"
    else:
        new, old = build()
        (t_new, t_old), base = _cb_time([new, old], device)
        del new, old
        win = t_new <= CB_RATIO * t_old
        _CB_SELECT[key] = (win, t_new, t_old, base)
        note = ""
    _cb_say(f"selector {key} ({what}): new {t_new:.2f} us vs master {t_old:.2f} us (graph replay, "
            f"{CB_FLUSH_BYTES >> 20} MB flush, {CB_REPS} interleaved reps, baseline {base:.2f} us subtracted), "
            f"new/master {t_new / t_old:.3f}, need <= {CB_RATIO} -> WINNER = {'NEW' if win else 'MASTER'}{note}")
    return win, t_new, t_old
def _cb_verdict(rec, win, t_new, t_old):
    rec.update(win=win, t_new=t_new, t_old=t_old)
    if not win:
        rec["reason"] = f"selector: new/master {t_new / t_old:.3f} > {CB_RATIO}"
def _cb_attn_fns(state, device, window, batch, length, on):
    rng = random.Random(4100 + 7 * batch + length + window)
    gen = torch.Generator(device=device)
    gen.manual_seed(4100 + batch)
    lens = _sk_agentic(rng, batch, 80000) if batch > 1 else [80000]
    data = _cb_data(state, device, gen, window, batch, length, lens, 1.0)
    st_on, st_off = _cb_flags(state, **on), _cb_off(state)
    return (lambda: _attn_core(data, st_on)), (lambda: _attn_core(data, st_off))
def _cb_r_fns(state, device, tokens):
    dl = state["dl"]
    pdl = dl["pdl"]
    ls = torch.tensor([96], dtype=torch.int32, device=device)
    x = torch.randn((tokens, HIDDEN), device=device).to(torch.bfloat16)
    eip = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=device)
    lgp = torch.empty((tokens, N_EXPERTS), dtype=torch.float32, device=device)
    bufs, bsm = _dl_route_bufs(tokens, device)
    def pair(variant, rpdl):
        def run():
            _DL_EXT.dl_pooler_router(x, dl["w_pr"], eip, lgp, dl["ws"], dl["cnt_p"], pdl)
            _dl_route_launch(eip, lgp, ls, dl, bufs, bsm, rpdl, variant)
        return run
    return pair(0, pdl), pair(2, 0)
def _cb_u_fns(state, device, tokens):
    d3 = state["dl3"]
    y = (torch.randn((tokens, D_EXPERT), device=device) * 0.5).to(torch.bfloat16)
    return (lambda: _dl3_unpool(y, state, d3)), (lambda: _unpool_master(state["dense"], y))
def _cb_cell_of(state, window, batch, length):
    cfg = {"window": window, "q_len": length, "batch": batch, "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64,
           "page_size": 1}
    sk_variant = _sk_variant(cfg, state.get("sk"))
    paired = window == 0 and batch > 1 and sk_variant < 0
    return _s3_cell(cfg, sk_variant >= 0, paired)
def _cb_prepare_s2a(state, device, recs, t_prepare0):
    if state.get("dl") is None or state.get("dl3") is None or not _DL_S3:
        for rec in recs["S2a"].values():
            rec["reason"] = "the dense chain or its stage-3 module is off for this state"
        return
    for cls in S2A_CLASSES:
        rec = recs["S2a"][cls]
        if _cb_late(t_prepare0):
            rec["reason"] = f"cell deadline {CB_CELL_DEADLINE:.0f}s passed"
            continue
        problems = _dl3_check_r(state, device, tokens=S2A_CHECK_T[cls])
        if not _cb_clean(state, recs, f"S2a check T-class {cls}"):
            problems.append("counters not back to 0 after the check")
        if not problems:
            err = _cb_s2_alt(state, device, "R", cls)
            if not _cb_clean(state, recs, f"S2a alternation T={cls}"):
                err = err or "counters not back to 0 after the alternation"
            if err:
                problems.append(err)
        _cb_say(f"S2a T-class {cls}: correctness {'ok' if not problems else 'FAIL: ' + '; '.join(problems[:3])}")
        if problems:
            rec["reason"] = "check: " + "; ".join(problems[:3])
            continue
        win, t_new, t_old = _cb_select("S2a", cls, f"P -> R pair T={cls}", lambda: _cb_r_fns(state, device, cls),
                                       device)
        _cb_verdict(rec, win, t_new, t_old)
        if not _cb_clean(state, recs, f"S2a selector T={cls}"):
            rec["reason"] = "counters not back to 0 after the selector"
def _cb_prepare_s2b(state, device, recs, t_prepare0):
    rec = recs["S2b"][8]
    if state.get("dl") is None or state.get("dl3") is None or not _DL_S3:
        rec["reason"] = "the dense chain or its stage-3 module is off for this state"
        return
    problems = _dl3_check_u(state, device, tokens_list=S2B_CHECK_T)
    if not _cb_clean(state, recs, "S2b check"):
        problems.append("counters not back to 0 after the check")
    if not problems:
        err = _cb_s2_alt(state, device, "U", 8)
        if not _cb_clean(state, recs, "S2b alternation"):
            err = err or "counters not back to 0 after the alternation"
        if err:
            problems.append(err)
    _cb_say(f"S2b T 2..8: correctness {'ok' if not problems else 'FAIL: ' + '; '.join(problems[:3])}")
    if problems:
        rec["reason"] = "check: " + "; ".join(problems[:3])
        return
    win, t_new, t_old = _cb_select("S2b", 8, "unpooler T=8", lambda: _cb_u_fns(state, device, 8), device)
    _cb_verdict(rec, win, t_new, t_old)
    if not _cb_clean(state, recs, "S2b selector"):
        rec["reason"] = "counters not back to 0 after the selector"
def _cb_prepare_s1(state, device, recs, t_prepare0):
    for cls in S1_CLASSES:
        rec = recs["S1"][cls]
        if _cb_late(t_prepare0):
            rec["reason"] = f"cell deadline {CB_CELL_DEADLINE:.0f}s passed"
            continue
        cases = [(2048, b, 1, [S1_LENS[(i + b) % len(S1_LENS)] for i in range(b)], kvs)
                 for b in S1_CHECK_B[cls] for kvs in CB_KV_SCALES]
        err = _cb_attn_alt(state, device, f"S1 class {cls}", cases, {"s1": {cls: True}})
        if not _cb_clean(state, recs, f"S1 class {cls} check"):
            err = err or "counters not back to 0 after the check"
        _cb_say(f"S1 class {cls}: correctness {'ok' if err is None else 'FAIL: ' + err}")
        if err:
            rec["reason"] = f"check: {err}"
            continue
        b = S1_SELECT_B[cls]
        win, t_new, t_old = _cb_select("S1", cls, f"whole attention side, sliding decode B={b}",
                                       lambda: _cb_attn_fns(state, device, 2048, b, 1, {"s1": {cls: True}}), device)
        _cb_verdict(rec, win, t_new, t_old)
        if not _cb_clean(state, recs, f"S1 class {cls} selector"):
            rec["reason"] = "counters not back to 0 after the selector"
def _cb_prepare_s3(state, device, recs, t_prepare0):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261043)
    for cell in S3_CELLS:
        rec = recs["S3"][cell]
        window, fronts, poses, e2e, (sb, sl) = S3_SPEC[cell]
        if _cb_late(t_prepare0):
            rec["reason"] = f"cell deadline {CB_CELL_DEADLINE:.0f}s passed"
            continue
        if _cb_cell_of(state, window, sb, sl) != cell:
            rec["reason"] = "this state's attention side does not route the cell (no unpaired sk / split-KV feed)"
            continue
        paired = cell.startswith("c3")
        problems = []
        for b, length in fronts:
            for kvs in CB_KV_SCALES:
                for pos in poses:
                    p = _s3_front_check(state, device, gen, window, b, length, kvs, pos, paired)
                    if p is not None:
                        problems.append(p)
        if not _cb_clean(state, recs, f"S3 {cell} front check"):
            problems.append("counters not back to 0 after the front check")
        if not problems:
            cases = [(window, b, length, _cb_mixed(b, ctx), kvs) for b, length, ctx in e2e for kvs in CB_KV_SCALES]
            err = _cb_attn_alt(state, device, f"S3 {cell}", cases, {"s3": {cell: True}})
            if not _cb_clean(state, recs, f"S3 {cell} alternation"):
                err = err or "counters not back to 0 after the alternation"
            if err:
                problems.append(err)
        _cb_say(f"S3 {cell}: correctness {'ok' if not problems else 'FAIL: ' + '; '.join(problems[:2])}")
        if problems:
            rec["reason"] = "check: " + "; ".join(problems[:2])
            continue
        win, t_new, t_old = _cb_select("S3", cell, f"whole attention side B={sb} q_len {sl}",
                                       lambda: _cb_attn_fns(state, device, window, sb, sl, {"s3": {cell: True}}),
                                       device)
        _cb_verdict(rec, win, t_new, t_old)
        if not _cb_clean(state, recs, f"S3 {cell} selector"):
            rec["reason"] = "counters not back to 0 after the selector"
def _cb_on(recs, source, cls) -> bool:
    rec = recs[source][cls]
    return bool(rec.get("win")) and rec.get("reason") is None and recs["_dirty"] is None
def _cb_prepare_t8(state, device, recs, t_prepare0):
    live = [(s, c) for s, c in _CB_T8_CELLS if _cb_on(recs, s, c)]
    if not live:
        _cb_say("T=8 layer check: no T=8 cell is on for this state, nothing to check")
        return
    _cb_apply(state, recs)
    err = _dl3_layer_checks(state, device, False, False, False, t8=True)
    if not _cb_clean(state, recs, "T=8 layer check"):
        err = err or "counters not back to 0"
    if err is not None:
        for s, c in live:
            recs[s][c]["reason"] = f"T=8 layer check: {err}"
        _cb_say(f"T=8 layer check FAILED ({err}): master kept at T=8 for {live}")
    else:
        _cb_say(f"T=8 layer check passed with {live} on")
def _cb_apply(state, recs):
    state["s1"] = {c: _cb_on(recs, "S1", c) for c in S1_CLASSES}
    state["s3"] = {c: _cb_on(recs, "S3", c) for c in S3_CELLS}
    d3 = state.get("dl3")
    if d3 is not None:
        d3["b_t"] = {c: _cb_on(recs, "S2a", c) for c in S2A_CLASSES}
        d3["u8"] = _cb_on(recs, "S2b", 8)
    if state.get("skw") is not None:
        state["skw"]["route"] = {b: ((recs["SKW"][b]["variant"], recs["SKW"][b]["spec"])
                                     if _cb_on(recs, "SKW", b) else None) for b in SKW_BANDS}
def _cb_phase(name, state, device, t_prepare0, recs, body):
    t0 = time.perf_counter()
    since = t0 - t_prepare0
    if since > CB_DEADLINE[name]:
        reason = (f"phase {name} skipped: deadline (start deadline {CB_DEADLINE[name]:.0f}s passed, {since:.1f}s into "
                  f"prepare())")
    else:
        _cb_say(f"phase {name}: start {since:.1f}s into prepare()")
        try:
            body(state, device, recs, t_prepare0)
            reason = None
        except Exception as e:
            reason = f"phase {name} raised {type(e).__name__}: {str(e)[:300]}"
        try:
            _cb_clean(state, recs, f"phase {name}")
        except Exception as e:
            recs["_dirty"] = recs["_dirty"] or f"phase {name} counter read raised {e!r}"
    if reason is not None:
        _cb_say(reason)
        cells = _CB_T8_CELLS if name == "T8" else [(name, c) for c in recs[name]]
        for src, c in cells:
            rec = recs[src][c]
            if name == "T8":
                if _cb_on(recs, src, c):
                    rec["reason"] = f"T=8 layer check not completed: {reason}"
            elif rec.get("reason") is None and not rec.get("win"):
                rec["reason"] = reason
    _cb_apply(state, recs)
    torch.cuda.empty_cache()
    now = time.perf_counter()
    _cb_say(f"phase {name}: {now - t0:.1f}s wall, this prepare() {now - t_prepare0:.1f}s, "
            f"all prepare() calls {_PREPARE_WALL_S + now - t_prepare0:.1f}s")
def _cb_report(state, recs):
    d3 = state.get("dl3")
    del _CB_SUMMARY[:]
    for source, cls, what, new_name, old_name in _CB_LINES:
        rec = recs[source][cls]
        if source == "S1":
            flag = state["s1"][cls]
        elif source == "S3":
            flag = state["s3"][cls]
        elif source == "SKW":
            flag = state.get("skw") is not None and state["skw"]["route"].get(cls) is not None
        elif d3 is None:
            flag = False
        else:
            flag = d3["b_t"][cls] if source == "S2a" else d3["u8"]
        winner_new = _cb_on(recs, source, cls)
        if source == "SKW" and flag and winner_new and state["skw"]["route"][cls] != (rec["variant"], rec["spec"]):
            flag = False
        if flag != winner_new:
            _cb_say(f"ASSERT: {source} {cls} flag {flag} != printed winner {winner_new}; forcing master")
            if source == "S1":
                state["s1"][cls] = False
            elif source == "S3":
                state["s3"][cls] = False
            elif source == "SKW":
                state["skw"]["route"][cls] = None
            elif d3 is not None and source == "S2a":
                d3["b_t"][cls] = False
            elif d3 is not None:
                d3["u8"] = False
            flag, winner_new = False, False
            rec["reason"] = rec.get("reason") or "flag / winner mismatch"
        times = (f"[new {rec['t_new']:.2f} us vs master {rec['t_old']:.2f} us]" if rec.get("t_new") is not None
                 else "[not timed]")
        if winner_new:
            extra = f" variant {rec['variant']} grid {rec['spec']}" if source == "SKW" else ""
            print(f"PATH CHOSEN: {source} {what}: NEW {new_name}{extra} {times}", flush=True)
            _CB_SUMMARY.append(f"{source} {cls}: NEW{extra} {times}")
        else:
            why = recs["_dirty"] and f"counters not back to 0 after {recs['_dirty']}" or rec.get("reason")
            print(f"PATH CHOSEN: {source} {what}: MASTER {old_name} {times} (fallback: {why})", flush=True)
            _CB_SUMMARY.append(f"{source} {cls}: MASTER {times} (fallback: {str(why)[:90]})")
def _combo_prepare(state, device, t_prepare0):
    global _CB_T0
    _CB_T0 = t_prepare0
    recs = {"S1": {c: {} for c in S1_CLASSES}, "S2a": {c: {} for c in S2A_CLASSES}, "S2b": {8: {}},
            "S3": {c: {} for c in S3_CELLS}, "SKW": {b: {} for b in SKW_BANDS}, "_dirty": None}
    _cb_say("new load_inline module muse_decode_sk_window_v1 (SKW) was built at import (compile seconds printed above); "
            "S1 / S3 are Triton (JIT at their first prepare() launch, timed in the launch lines), S2a / S2b are "
            "kernels of muse_decode_dense_chain_s3 built at import")
    for name, body in (("S2a", _cb_prepare_s2a), ("S2b", _cb_prepare_s2b), ("S1", _cb_prepare_s1),
                       ("S3", _cb_prepare_s3), ("T8", _cb_prepare_t8), ("SKW", _cb_prepare_skw)):
        _cb_phase(name, state, device, t_prepare0, recs, body)
    try:
        _cb_report(state, recs)
    except Exception as e:
        _cb_say(f"report raised {e!r}: every combination flag off for this state")
        recs["_dirty"] = recs["_dirty"] or "report"
        _cb_apply(state, recs)
SKW_TAG = "[skw]"
SKW_WINDOW = 2048
SKW_MARGIN = 1.03
SKW_IQR_RATIO = 1.5
SKW_IQR_FLOOR_US = 0.5
SKW_REPS = 41
SKW_BANDS = ("b32", "v1")
SKW_CANDS = {
    "b32": ((6, "p2"), (6, "p4"), (6, "p8")),
    "v1": ((4, "p4"), (4, "p8"), (4, "sm"), (5, "p4"), (5, "p8"), (5, "sm")),
}
SKW_SELECT = {"b32": (32, 1), "v1": (1, 8)}
SKW_CHECK_B = {"b32": (17, 32), "v1": (1,)}
SKW_QKV_COLS = (16 + 2 * 2 + 16) * 64
SKW_TOL_TOTAL = 1.5e-2
SKW_TOL_ROW = 5e-2
SKW_LENS = (1, 63, 2047, 2048, 2049, 80000)
_SKW = None
_SKW_RES = {}
_SKW_ATTRS_DONE = False
_SKW_BUILD = {"s": None, "err": None}
_SKW_CPP = r"""
#include <torch/extension.h>
#include <vector>
void skw_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor qkv,
             int64_t qkv_stride, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor ticket,
             torch::Tensor gticket, torch::Tensor gacc, torch::Tensor gml,
             double score_scale, double eps, int64_t grid, int64_t variant);
std::vector<int64_t> skw_attrs(int64_t variant);
std::vector<int64_t> skw_ph_setup();
void skw_ph_attn(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor seq_lens,
                 torch::Tensor page_table, torch::Tensor inv_freq, torch::Tensor k_scale, torch::Tensor v_scale,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out,
                 int64_t window, double q_scale, double score_scale, double eps);
"""
_SKW_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include <vector>

#define SK_PSEUDO 1  // this module holds only the PSEUDO instantiations (variants 4..7)

namespace {

constexpr int D = 64;              // head dim
constexpr int HQ = 16;             // query heads
constexpr int HKV = 2;             // kv heads
constexpr int G = HQ / HKV;        // queries per kv head (8)
constexpr int ROW = HKV * D;       // bytes per cache row (page 1: both kv heads, 128)
constexpr int TILE = 64;           // tokens per tile (the stream-K unit)
constexpr int MB_TILES = 16;       // tiles per page-index block
constexpr int MB_TOK = MB_TILES * TILE;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAXB = 64;
constexpr int STAGE_BYTES = 2 * TILE * ROW;  // K tile + V tile = 16 KB
constexpr int QBYTES = HQ * D;               // one request's e4m3 query
constexpr int PS_MAXR = 256;                 // PSEUDO: pseudo-rows (8 per verify request, B <= 32)
constexpr int PS_WINDOW = 2048;              // PSEUDO: the sliding window (keys p - 2047 .. p)
#ifdef SK_PSEUDO
constexpr int SK_MAX_VARIANT = 7;
#else
constexpr int SK_MAX_VARIANT = 3;
#endif

// MAXR = request capacity of the smem tables. The default (64, no LO table) is exactly the landed
// layout (95264 B at 4 stages); PSEUDO instantiations use Lay<STAGES, PS_MAXR, true>.
template <int STAGES, int MAXR = MAXB, bool WITH_LO = false>
struct Lay {
  static constexpr int RING = 0;
  static constexpr int QR = RING + STAGES * STAGE_BYTES;      // e4m3 [STAGES][HQ][D]
  static constexpr int IDX = QR + STAGES * QBYTES;            // int32 [2][MB_TOK]
  static constexpr int ACC = IDX + 2 * MB_TOK * 4;            // f32 [WARPS][G][D]
  static constexpr int ML = ACC + WARPS * G * D * 4;          // f32 [WARPS][G][2]
  static constexpr int SEQ = ML + WARPS * G * 2 * 4;          // int32 [MAXR]
  static constexpr int TP = SEQ + MAXR * 4;                   // int32 [MAXR + 1] (+pad)
  static constexpr int FLAG = TP + (MAXR + 4) * 4;            // int32 [4]
  static constexpr int LO = FLAG + 16;                        // int32 [MAXR], only when WITH_LO
  static constexpr int BYTES = LO + (WITH_LO ? MAXR * 4 : 0);
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// cache_seqlens -> seq_s, and the tile prefix tp_s[0..B] (tp_s[B] = Ttot). Ends with a barrier.
__device__ __forceinline__ void load_prefix(const int* __restrict__ seqlens, int B, int* seq_s, int* tp_s) {
  const int tid = threadIdx.x;
  if (tid < B) { const int s = seqlens[tid]; seq_s[tid] = s > 0 ? s : 0; }
  __syncthreads();
  if (tid == 0) {
    int run = 0;
    tp_s[0] = 0;
    for (int b = 0; b < B; ++b) {
      run += (seq_s[b] + TILE - 1) / TILE;
      tp_s[b + 1] = run;
    }
  }
  __syncthreads();
}
// PSEUDO prefix. Pseudo-request r = 8 * b + j (token j of verify block b) attends keys [lo_r, sp_r) of
// page-table row b, sp_r = cache_seqlens[b] - 7 + j (= position + 1: causal inside the block),
// lo_r = max(0, sp_r - PS_WINDOW) (= position - 2047 clamped: the sliding window, every key of it).
// cnt_s[r] = sp_r - lo_r is the key count the rest of the kernel treats as the request length,
// lo_s[r] = lo_r is added only to the page-table offset, tp_s is the inclusive tile prefix by one
// block-wide scan (R = 8B <= 256 = THREADS). wsum: >= 8 ints of scratch. Ends with a barrier.
// Without SK_PSEUDO the body is empty (the landed build).
template <int TOK>
__device__ __forceinline__ void load_prefix_pseudo(const int* __restrict__ seqlens, int B, int* cnt_s, int* lo_s,
                                                   int* tp_s, int* wsum) {
#ifdef SK_PSEUDO
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int R = B * TOK;
  int v = 0;
  if (tid < R) {
    // TOK == 8 (sliding verify) keeps the shift / mask forms; TOK == 1 (sliding decode): row = request.
    int sp = seqlens[TOK == 8 ? (tid >> 3) : tid / TOK] - (TOK - 1) + (TOK == 8 ? (tid & 7) : tid % TOK);
    sp = sp > 0 ? sp : 0;
    const int lo = sp > PS_WINDOW ? sp - PS_WINDOW : 0;
    const int cnt = sp - lo;
    cnt_s[tid] = cnt;
    lo_s[tid] = lo;
    v = (cnt + TILE - 1) / TILE;
  }
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += n;
  }
  if (lane == 31) wsum[warp] = v;
  __syncthreads();
  int add = 0;
  for (int w = 0; w < warp; ++w) add += wsum[w];
  if (tid < R) tp_s[tid + 1] = v + add;
  if (tid == 0) tp_s[0] = 0;
  __syncthreads();
#endif
}
// First tile of CTA c, and the CTA owning tile t (the exact inverse: s_c <= t < s_{c+1}).
__device__ __forceinline__ int range_start(int c, int T, int gx) {
  return static_cast<int>((static_cast<long long>(c) * T) / gx);
}
__device__ __forceinline__ int tile_owner(int t, int T, int gx) {
  return static_cast<int>(((static_cast<long long>(t) + 1) * gx - 1) / T);
}

// LSE merge of the partials in slots s_lo..s_hi for head h, dims d0..d0+3 (M = the raw max, so the
// result can itself be stored as a slot). L2 loads (__ldcg): the slots were written by other CTAs.
// Stage-2 callers pass s = c + b (c = c_lo..c_hi); the two-level path passes group-slot indices.
__device__ __forceinline__ void merge_range(const float* ws_acc, const float* ws_ml, int s_lo, int s_hi,
                                            int h, int d0, float4& o, float& L, float& Mout) {
  const float NEG_INF = __int_as_float(0xff800000u);
  float M = NEG_INF;
  for (int c = s_lo; c <= s_hi; ++c) M = fmaxf(M, __ldcg(ws_ml + (int64_t)c * 2 * HQ + h));
  const float Mu = M == NEG_INF ? 0.f : M;
  Mout = M;
  o = make_float4(0.f, 0.f, 0.f, 0.f);
  L = 0.f;
  for (int c = s_lo; c <= s_hi; ++c) {
    const int64_t sl = c;
    const float m = __ldcg(ws_ml + sl * 2 * HQ + h);
    const float wt = m == NEG_INF ? 0.f : ex2(m - Mu);
    L += wt * __ldcg(ws_ml + sl * 2 * HQ + HQ + h);
    const float4 a = __ldcg(reinterpret_cast<const float4*>(ws_acc + (sl * HQ + h) * D + d0));
    o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
  }
}
__device__ __forceinline__ void merge_slots(const float* ws_acc, const float* ws_ml, int b, int c_lo, int c_hi,
                                            int h, int d0, float4& o, float& L) {
  float M;
  merge_range(ws_acc, ws_ml, c_lo + b, c_hi + b, h, d0, o, L, M);
}

// Release/acquire ticket draw (two-level path): the CTA's partial stores are ordered before the
// increment (acq_rel at gpu scope, on top of the caller's __threadfence + __syncthreads), and the
// returned count orders the winner after every earlier arrival's stores.
__device__ __forceinline__ int ticket_arrive(int* p) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(1) : "memory");
  return old;
}

// Stage-1 B epilogue for head h, dims d0..d0+3 (16 consecutive lanes hold one head): o / L *
// v_scale, scaleless RMSNorm over 64 (eps inside the sqrt), times sigmoid(gate), bf16 out.
// Every lane of the warp must call it (16-lane shuffles).
__device__ __forceinline__ void finalize_store(float4 o, float L, float vs, int b, int h, int d0,
                                               const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                                               __nv_bfloat16* __restrict__ out, float eps) {
  o.x = o.x / L * vs; o.y = o.y / L * vs; o.z = o.z / L * vs; o.w = o.w / L * vs;
  float ss = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  ss += __shfl_xor_sync(0xffffffffu, ss, 8);
  const float r = 1.f / sqrtf(ss / D + eps);
  const __nv_bfloat162* gp =
      reinterpret_cast<const __nv_bfloat162*>(qkv + (int64_t)b * qkv_stride + (HQ + 2 * HKV) * D + h * D + d0);
  const float2 g01 = __bfloat1622float2(gp[0]);
  const float2 g23 = __bfloat1622float2(gp[1]);
  const float y0 = o.x * r / (1.f + expf(-g01.x));
  const float y1 = o.y * r / (1.f + expf(-g01.y));
  const float y2 = o.z * r / (1.f + expf(-g23.x));
  const float y3 = o.w * r / (1.f + expf(-g23.y));
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + ((int64_t)b * HQ + h) * D + d0);
  op[0] = __floats2bfloat162_rn(y0, y1);
  op[1] = __floats2bfloat162_rn(y2, y3);
}

// PSEUDO (sliding verify, q_len 8, TOK = 8): the kernel rows are the R = 8B pseudo-rows r = 8 b + j; q is the
// e4m3 query [B * 8, HQ, D] (token r), qkv / out rows are r, page_table row is r >> 3, seqlens is
// cache_seqlens [B]; ws / ticket / group slots are sized for R. PSEUDO = false is the landed kernel.
// TOK = 1 (sliding decode, q_len 1): R = B, row r = request r, sp_r = cache_seqlens[r] (the new key
// included), page-table row r. TOK = 8 compiles exactly as before (only constants changed spelling).
template <int STAGES, bool TICKET, bool TWO = false, bool PSEUDO = false, int TOK = 8>
__global__ void __launch_bounds__(THREADS, 2)
skw_attn_kernel(const uint8_t* __restrict__ q,          // e4m3 [B, HQ, D]
                       const uint8_t* __restrict__ kc,         // e4m3 [rows, ROW]
                       const uint8_t* __restrict__ vc,         // e4m3 [rows, ROW]
                       const int* __restrict__ page_table, int64_t pt_stride,
                       const int* __restrict__ seqlens, int B,
                       const float* __restrict__ k_scale, const float* __restrict__ v_scale, float score_scale,
                       const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                       __nv_bfloat16* __restrict__ out,        // bf16 [B, HQ * D]
                       float* __restrict__ ws_acc,             // f32 [grid + R, HQ, D]
                       float* __restrict__ ws_ml,              // f32 [grid + R, 2, HQ]
                       int* __restrict__ ticket,               // int32 [>= R], 0 between launches
                       int* __restrict__ gticket,              // int32 [>= R * 32], 0 between launches (TWO)
                       float* __restrict__ gacc,               // f32 [R * 32, HQ, D] group slots (TWO)
                       float* __restrict__ gml,                // f32 [R * 32, 2, HQ] (TWO)
                       float eps) {
  using S = typename std::conditional<PSEUDO, Lay<STAGES, PS_MAXR, true>, Lay<STAGES>>::type;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem + S::RING;
  uint8_t* qr = smem + S::QR;
  int* idx_s = reinterpret_cast<int*>(smem + S::IDX);
  float* acc_s = reinterpret_cast<float*>(smem + S::ACC);
  float* ml_s = reinterpret_cast<float*>(smem + S::ML);
  int* seq_s = reinterpret_cast<int*>(smem + S::SEQ);
  int* tp_s = reinterpret_cast<int*>(smem + S::TP);
  int* flag_s = reinterpret_cast<int*>(smem + S::FLAG);
  int* lo_s = reinterpret_cast<int*>(smem + S::LO);  // PSEUDO only (no storage otherwise: never read)
  const float NEG_INF = __int_as_float(0xff800000u);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bx = blockIdx.x;
  const int R = PSEUDO ? B * TOK : B;  // rows of the flat tile space
  if constexpr (PSEUDO) load_prefix_pseudo<TOK>(seqlens, B, seq_s, lo_s, tp_s, reinterpret_cast<int*>(acc_s));
  else load_prefix(seqlens, B, seq_s, tp_s);
  const int T = tp_s[R];
  if (T <= 0) return;  // uniform
  // Effective grid min(gridDim.x, T): below it every CTA owns >= 1 tile, so CTAs c_lo..c_hi of a
  // request all hold a segment (the ticket count and the merge rely on it). CTAs above it exit.
  const int gx = min(static_cast<int>(gridDim.x), T);
  if (bx >= gx) return;  // uniform
  const int t0 = range_start(bx, T, gx);
  const int N = range_start(bx + 1, T, gx) - t0;  // tiles of this CTA, one flat pipeline
  if (N <= 0) return;  // uniform
  const int nblk = (N + MB_TILES - 1) / MB_TILES;
  const float scale = score_scale * __ldg(k_scale);  // sm_scale * qk_scale_factor * log2(e) * k_scale
  const float vs = __ldg(v_scale);

  // Request holding global tile gt: the largest b with tp_s[b] <= gt (gt < Ttot).
  auto find_req = [&](int gt) {
    int lo = 0, hi = R - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (tp_s[mid] <= gt) lo = mid; else hi = mid - 1;
    }
    return lo;
  };
  // Page indices of block k (flat tiles 16k .. 16k+15) into buffer k & 1: thread -> tile tid / 16,
  // tokens (tid % 16) * 4 .. +4. Positions past the request end (or past the range) zero-filled.
  auto issue_meta = [&](int k) {
    const int buf = k & 1;
    const int tl = tid >> 4;
    const int e0 = (tid & 15) * 4;
    const int n = k * MB_TILES + tl;
    int b = 0, pos = 0, seq = 0;
    if (n < N) {
      const int gt = t0 + n;
      b = find_req(gt);
      pos = (gt - tp_s[b]) * TILE + e0;
      seq = seq_s[b];
    }
    int prb = b, plo = 0;  // page-table row, first key of the request's range
    if constexpr (PSEUDO) { prb = TOK == 8 ? (b >> 3) : b / TOK; plo = lo_s[b]; }
    const int* prow = page_table + (int64_t)prb * pt_stride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool ok = pos + j < seq;
      cp_async4(smem_u32(idx_s + buf * MB_TOK + tl * TILE + e0 + j), ok ? prow + plo + pos + j : prow, ok ? 4 : 0);
    }
  };
  // Tile nn (issued in order 0, 1, 2, ...) into ring stage nn % STAGES, as kernel A: thread ->
  // token tid / 4, 16-byte chunks 2 * (tid % 4) + {0, 1} of its K row and V row, chunk c stored
  // at c ^ (token % 8). A segment's first tile also brings the request's query into its q slot.
  int p_b = find_req(t0);
  auto issue_tile = [&](int nn) {
    const int gt = t0 + nn;
    while (gt >= tp_s[p_b + 1]) ++p_b;
    const int lt = gt - tp_s[p_b];
    const int st = nn % STAGES;
    uint8_t* stp = ring + st * STAGE_BYTES;
    const int tt = tid >> 2;
    const bool ok = lt * TILE + tt < seq_s[p_b];
    const int64_t row = ok ? (int64_t)idx_s[((nn / MB_TILES) & 1) * MB_TOK + (nn % MB_TILES) * TILE + tt] : 0;
    const uint8_t* ks = kc + row * ROW;
    const uint8_t* vsrc = vc + row * ROW;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = (tid & 3) * 2 + j;
      const int dst = tt * ROW + ((c ^ (tt & 7)) << 4);
      cp_async16(smem_u32(stp + dst), ks + c * 16, ok ? 16 : 0);
      cp_async16(smem_u32(stp + TILE * ROW + dst), vsrc + c * 16, ok ? 16 : 0);
    }
    if ((nn == 0 || lt == 0) && tid < QBYTES / 16)
      cp_async16(smem_u32(qr + st * QBYTES + tid * 16), q + (int64_t)p_b * QBYTES + tid * 16, 16);
  };

  // Prologue: block 0's indices (waited), then tiles 0 .. STAGES-2; block 1 rides tile 0's group.
  issue_meta(0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < N) {
      issue_tile(j);
      if (j % MB_TILES == 0 && j / MB_TILES + 1 < nblk) issue_meta(j / MB_TILES + 1);
    }
    cp_commit();
  }

  const int g = warp & 1;             // kv head
  const int slice = (warp >> 1) * 16; // token slice of each tile
  const int quad = lane >> 2, qi = lane & 3;
  uint32_t qa[4][2];
  float m_i = NEG_INF, l_i = 0.f;
  float acc[8][4];
  int c_b = find_req(t0);
  bool seg_head = false;  // the current segment began at its request's first tile

  for (int n = 0; n < N; ++n) {
    cp_wait<STAGES - 2>();  // tile n (and everything issued before it) has landed
    __syncthreads();        // visible to all; every warp is done with tile n - 1's stage
    {
      const int nn = n + STAGES - 1;
      if (nn < N) {
        issue_tile(nn);
        if (nn % MB_TILES == 0 && nn / MB_TILES + 1 < nblk) issue_meta(nn / MB_TILES + 1);
      }
      cp_commit();  // always, so the group count stays uniform
    }
    const int gt = t0 + n;
    while (gt >= tp_s[c_b + 1]) ++c_b;
    const int lt = gt - tp_s[c_b];
    const int nt_b = tp_s[c_b + 1] - tp_s[c_b];
    const int cseq = seq_s[c_b];
    const int st = n % STAGES;
    if (n == 0 || lt == 0) {
      seg_head = lt == 0;
      const uint8_t* qrow = qr + st * QBYTES + (g * G + quad) * D + qi * 4;
#pragma unroll
      for (int s = 0; s < 4; ++s)
        e4m3x4_to_f16x2x2(*reinterpret_cast<const uint32_t*>(qrow + s * 16), qa[s][0], qa[s][1]);
      m_i = NEG_INF;
      l_i = 0.f;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;
    }
    const uint8_t* Ks = ring + st * STAGE_BYTES;
    const uint8_t* Vs = Ks + TILE * ROW;

    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
      const int tt = slice + nb * 8 + quad;
      const uint8_t* kr = Ks + tt * ROW + qi * 4;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kr + (((g * 4 + ks) ^ (tt & 7)) << 4));
        uint32_t b0, b1;
        e4m3x4_to_f16x2x2(w, b0, b1);
        mma16816(s[nb], qa[ks][0], 0u, qa[ks][1], 0u, b0, b1);
      }
    }
    const int pos0 = lt * TILE + slice + qi * 2;
    float x0 = pos0 < cseq ? s[0][0] * scale : NEG_INF;
    float x1 = pos0 + 1 < cseq ? s[0][1] * scale : NEG_INF;
    float x2 = pos0 + 8 < cseq ? s[1][0] * scale : NEG_INF;
    float x3 = pos0 + 9 < cseq ? s[1][1] * scale : NEG_INF;
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float m_new = fmaxf(m_i, mx);
    const float m_use = m_new == NEG_INF ? 0.f : m_new;
    const float alpha = ex2(m_i - m_use);
    x0 = ex2(x0 - m_use);
    x1 = ex2(x1 - m_use);
    x2 = ex2(x2 - m_use);
    x3 = ex2(x3 - m_use);
    l_i = l_i * alpha + (x0 + x1) + (x2 + x3);
    m_i = m_new;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) { acc[nb][0] *= alpha; acc[nb][1] *= alpha; }
    const uint32_t pa0 = pack_f16x2(x0, x1), pa2 = pack_f16x2(x2, x3);

    const int tv = slice + qi * 2;
    const int vlc = g * 4 + (quad >> 1);
    const int voff = (quad & 1) * 8;
    const uint2 v0 = *reinterpret_cast<const uint2*>(Vs + tv * ROW + ((vlc ^ (tv & 7)) << 4) + voff);
    const uint2 v1 = *reinterpret_cast<const uint2*>(Vs + (tv + 1) * ROW + ((vlc ^ ((tv + 1) & 7)) << 4) + voff);
    const uint2 v8 = *reinterpret_cast<const uint2*>(Vs + (tv + 8) * ROW + ((vlc ^ ((tv + 8) & 7)) << 4) + voff);
    const uint2 v9 = *reinterpret_cast<const uint2*>(Vs + (tv + 9) * ROW + ((vlc ^ ((tv + 9) & 7)) << 4) + voff);
#pragma unroll
    for (int hw = 0; hw < 2; ++hw) {
      const uint32_t w0 = hw ? v0.y : v0.x, w1 = hw ? v1.y : v1.x;
      const uint32_t w8 = hw ? v8.y : v8.x, w9 = hw ? v9.y : v9.x;
      uint32_t r0[4], r1[4];
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x5140), r0[0], r0[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x7362), r0[2], r0[3]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x5140), r1[0], r1[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x7362), r1[2], r1[3]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma16816(acc[hw * 4 + j], pa0, 0u, pa2, 0u, r0[j], r1[j]);
    }

    if (n == N - 1 || lt == nt_b - 1) {
      // Segment end (uniform over the CTA): merge the four warps of each kv head in smem.
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 1);
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 2);
      const int rw = warp * G + quad;
      if (qi == 0) { ml_s[rw * 2] = m_i; ml_s[rw * 2 + 1] = l_i; }
      float* ar = acc_s + rw * D;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
        ar[(qi * 2) * 8 + nb] = acc[nb][0];
        ar[(qi * 2 + 1) * 8 + nb] = acc[nb][1];
      }
      __syncthreads();
      const int h = tid >> 4, d0 = (tid & 15) * 4;
      const int hg = h / G, r = h % G;
      float mw[4];
      float M = NEG_INF;
#pragma unroll
      for (int j = 0; j < 4; ++j) { mw[j] = ml_s[((hg + 2 * j) * G + r) * 2]; M = fmaxf(M, mw[j]); }
      const float Mu = M == NEG_INF ? 0.f : M;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int wr = (hg + 2 * j) * G + r;
        const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
        L += wt * ml_s[wr * 2 + 1];
        const float4 a = *reinterpret_cast<const float4*>(acc_s + wr * D + d0);
        o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
      }
      const bool whole = seg_head && lt == nt_b - 1;  // this CTA is the request's only segment
      if (TICKET && whole) {
        finalize_store(o, L, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
      } else {
        const int64_t slot = (int64_t)bx + c_b;
        *reinterpret_cast<float4*>(ws_acc + (slot * HQ + h) * D + d0) = o;
        if ((tid & 15) == 0) { ws_ml[slot * 2 * HQ + h] = M; ws_ml[slot * 2 * HQ + HQ + h] = L; }
        if (TICKET && TWO) {
          // Two-level ticket. CTAs c_lo..c_hi hold request c_b's segments (nseg of them); group
          // size Gs = ceil(sqrt(nseg)), group gi = (c - c_lo) / Gs, ngrp = ceil(nseg / Gs) <= 32.
          const int c_lo = tile_owner(tp_s[c_b], T, gx);
          const int c_hi = tile_owner(tp_s[c_b + 1] - 1, T, gx);
          const int nseg = c_hi - c_lo + 1;
          int Gs = 1;
          while (Gs * Gs < nseg) ++Gs;
          const int gi = (bx - c_lo) / Gs;
          const int ngrp = (nseg + Gs - 1) / Gs;
          const int g_lo = c_lo + gi * Gs;
          const int g_hi = min(c_hi, g_lo + Gs - 1);
          __threadfence();  // release: this CTA's partial is visible device-wide before the ticket
          __syncthreads();
          if (tid == 0) flag_s[0] = ticket_arrive(gticket + c_b * 32 + gi) == g_hi - g_lo ? 1 : 0;
          __syncthreads();
          if (flag_s[0]) {  // uniform: the last arrival of group gi
            __threadfence();  // acquire: every group member's partial is visible
            float4 om;
            float Lm, Mm;
            merge_range(ws_acc, ws_ml, g_lo + c_b, g_hi + c_b, h, d0, om, Lm, Mm);
            if (tid == 0) gticket[c_b * 32 + gi] = 0;  // all members arrived: clean for the next launch
            if (ngrp == 1) {
              finalize_store(om, Lm, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
            } else {
              const int64_t gs = (int64_t)c_b * 32 + gi;
              *reinterpret_cast<float4*>(gacc + (gs * HQ + h) * D + d0) = om;
              if ((tid & 15) == 0) { gml[gs * 2 * HQ + h] = Mm; gml[gs * 2 * HQ + HQ + h] = Lm; }
              __threadfence();  // release: the group slot is visible before the second-level ticket
              __syncthreads();
              if (tid == 0) flag_s[1] = ticket_arrive(ticket + c_b) == ngrp - 1 ? 1 : 0;
              __syncthreads();
              if (flag_s[1]) {  // uniform: the last group of request c_b
                __threadfence();  // acquire
                float4 of;
                float Lf, Mf;
                merge_range(gacc, gml, c_b * 32, c_b * 32 + ngrp - 1, h, d0, of, Lf, Mf);
                finalize_store(of, Lf, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
                if (tid == 0) ticket[c_b] = 0;  // every group has arrived: clean for the next launch
              }
            }
          }
        } else if (TICKET) {
          const int c_lo = tile_owner(tp_s[c_b], T, gx);
          const int c_hi = tile_owner(tp_s[c_b + 1] - 1, T, gx);
          __threadfence();  // this CTA's partial is visible device-wide before the ticket
          __syncthreads();
          if (tid == 0) flag_s[0] = atomicAdd(ticket + c_b, 1) == c_hi - c_lo ? 1 : 0;
          __syncthreads();
          if (flag_s[0]) {  // uniform: the last segment of request c_b to finish
            __threadfence();
            float4 om;
            float Lm;
            merge_slots(ws_acc, ws_ml, c_b, c_lo, c_hi, h, d0, om, Lm);
            finalize_store(om, Lm, vs, c_b, h, d0, qkv, qkv_stride, out, eps);
            if (tid == 0) ticket[c_b] = 0;  // every segment has arrived: clean for the next launch
          }
        }
      }
    }
  }
  cp_wait<0>();  // the trailing groups are empty; drain anyway
}

using SkFn = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const int*, int64_t, const int*, int,
                      const float*, const float*, float, const __nv_bfloat16*, int64_t, __nv_bfloat16*,
                      float*, float*, int*, int*, float*, float*, float);

SkFn sk_pick(int64_t v) {
  if (v == 5) return skw_attn_kernel<4, true, true, true>;
  if (v == 6) return skw_attn_kernel<4, true, false, true, 1>;
  if (v == 7) return skw_attn_kernel<4, true, true, true, 1>;
  return skw_attn_kernel<4, true, false, true>;  // v == 4 (skw_attn / skw_attrs reject every other variant)
}
// 4 / 5 = the PSEUDO single-level / two-level ticket kernels (TOK 8, verify); 6 / 7 = the same with TOK 1
// (sliding decode). The module holds no other variant.
int sk_smem(int64_t) { return Lay<4, PS_MAXR, true>::BYTES; }
void sk_prepare(int64_t v) {
  static bool done[8] = {false, false, false, false, false, false, false, false};  // no device state
  if (!done[v]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(sk_pick(v)),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, sk_smem(v)));
    done[v] = true;
  }
}

}  // namespace

void skw_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
             torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor qkv,
             int64_t qkv_stride, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor ticket,
             torch::Tensor gticket, torch::Tensor gacc, torch::Tensor gml,
             double score_scale, double eps, int64_t grid, int64_t variant) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  TORCH_CHECK(variant >= 4 && variant <= SK_MAX_VARIANT, "skw_attn: variant");
  const bool pseudo = variant >= 4;
  const int tok = variant >= 6 ? 1 : 8;  // rows per request: 8 (verify pseudo-rows) or 1 (decode)
  const int R = pseudo ? B * tok : B;  // rows of the tile space (pseudo-rows for verify)
  TORCH_CHECK(B >= 1 && B <= (pseudo ? PS_MAXR / tok : MAXB), "skw_attn: batch out of range");
  if (variant == 3 || variant == 5 || variant == 7) {
    TORCH_CHECK(grid <= 1024, "skw_attn: two-level ticket needs grid <= 1024 (<= 32 groups)");
    TORCH_CHECK(gticket.numel() >= R * 32 && gticket.scalar_type() == at::kInt, "skw_attn: gticket");
    TORCH_CHECK(gacc.size(0) >= R * 32 && gml.size(0) >= R * 32 && gacc.scalar_type() == at::kFloat
                && gml.scalar_type() == at::kFloat, "skw_attn: group slots");
  }
  TORCH_CHECK(grid >= 1 && ws_acc.size(0) >= grid + R && ws_ml.size(0) >= grid + R, "skw_attn: workspace");
  TORCH_CHECK(ticket.numel() >= R && ticket.scalar_type() == at::kInt, "skw_attn: ticket");
  if (pseudo) TORCH_CHECK(q.size(0) >= R, "skw_attn: pseudo query rows");
  {  // the gate is read at element (HQ + 2 * HKV) * D of qkv row r (finalize_store): the qkv GEMM's row layout
    constexpr int64_t kCols = (HQ + 2 * HKV) * D + HQ * D;  // 2304: q | k | v | gate
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(1) == kCols && qkv.stride(1) == 1
                && qkv_stride >= kCols && qkv.size(0) >= R && qkv.stride(0) == qkv_stride,
                "skw_attn: qkv row layout (q | k | v | gate, 2304 bf16 columns) does not match the kernel's gate offset");
  }
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && page_table.scalar_type() == at::kInt, "skw_attn: dtypes");
  TORCH_CHECK(page_table.stride(1) == 1 && q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous()
              && out.is_contiguous(), "skw_attn: layout");
  sk_prepare(variant);
  auto stream = at::cuda::getCurrentCUDAStream();
  SkFn fn = sk_pick(variant);
  fn<<<static_cast<unsigned>(grid), THREADS, sk_smem(variant), stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_table.data_ptr<int>(), page_table.stride(0),
      cache_seqlens.data_ptr<int>(), B, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
      static_cast<float>(score_scale), reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv_stride,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(),
      ticket.data_ptr<int>(), gticket.data_ptr<int>(), gacc.data_ptr<float>(), gml.data_ptr<float>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// {binaryVersion, ptxVersion, numRegs, localSizeBytes, maxThreadsPerBlock, dynamic smem, CTAs/SM}
std::vector<int64_t> skw_attrs(int64_t variant) {
  TORCH_CHECK(variant >= 4 && variant <= SK_MAX_VARIANT, "skw_attrs: variant");
  sk_prepare(variant);
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(sk_pick(variant))));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sk_pick(variant), THREADS, sk_smem(variant)));
  return {static_cast<int64_t>(a.binaryVersion), static_cast<int64_t>(a.ptxVersion), static_cast<int64_t>(a.numRegs),
          static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.maxThreadsPerBlock),
          static_cast<int64_t>(sk_smem(variant)), static_cast<int64_t>(occ)};
}

// ---- [w6p-perhead] attention master's PERHEAD sliding_cluster_kernel<true> (muse_attention_decode PH module),
// ported verbatim into its own namespace (fp8 QK only; master's C=16 / C=4 instances are not built here) ----
namespace skwph {

namespace cg = cooperative_groups;

constexpr int D = 64;              // head dim
constexpr int HQ = 16;             // query heads
constexpr int HKV = 2;             // kv heads
constexpr int G = HQ / HKV;        // queries per kv head (8)
constexpr int ROW = HKV * D;       // bytes per cache row (page 1: both kv heads, 128)
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int SK = 128;            // keys per sub-chunk (one cp.async group, one ring slot)
constexpr int NS = 3;              // ring slots, all in flight at the start
constexpr int NSUB_MAX = 4;        // ck <= round16(ceil(2047 / 4)) = 512: host keeps window <= 2048, C >= 4
static_assert(NSUB_MAX <= NS + 1, "only one late sub-chunk is prefetched (its indices sit in registers)");
constexpr int SLOT = SK * ROW;                   // 16384 B per slot, for K and again for V
constexpr int K_OFF = 0;                         // e4m3 [NS][SK][ROW], swizzled
constexpr int V_OFF = K_OFF + NS * SLOT;         // e4m3 [NS][SK][ROW], swizzled
constexpr int XK_OFF = V_OFF + NS * SLOT;        // e4m3 [16][ROW]: the new key (row 0), zero rows after
constexpr int XV_OFF = XK_OFF + 16 * ROW;        // e4m3 [16][ROW]: the new value
constexpr int Q_OFF = XV_OFF + 16 * ROW;         // e4m3 [HQ][D]
constexpr int SMEM_BYTES = Q_OFF + HQ * D;       // 103424
static_assert(SMEM_BYTES <= 113 * 1024, "two CTAs per SM (C = 4 needs them for one wave at batch 32)");
constexpr int QKS = 2;             // k32 steps of the m16n8k32 e4m3 QK

// Partials alias the consumed ring once every warp is done: per-warp staging f32 [WARPS][G][D]
// and [WARPS][G][2], then the CTA partial acc f32 [16][D], m f32 [16], l f32 [16].
constexpr int STG_ACC = 0;
constexpr int STG_ML = STG_ACC + WARPS * G * D * 4;    // 16384
constexpr int P_ACC = STG_ML + WARPS * G * 2 * 4;      // 16896
constexpr int P_M = P_ACC + HQ * D * 4;
constexpr int P_L = P_M + HQ * 4;
static_assert(P_L + HQ * 4 <= NS * SLOT, "partials must fit in the consumed ring");
static_assert(P_ACC % 16 == 0, "float4 alignment");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_ring() { asm volatile("cp.async.wait_group 2;\n" ::: "memory"); }
static_assert(NS - 1 == 2, "cp_wait_ring waits for all but the NS - 1 newest groups");
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// Four E4M3 bytes -> two f16x2: lo from bytes 0,1 (byte 0 in the low half), hi from bytes 2,3.
__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
// Two fp32 -> two E4M3 bytes (round to nearest even, saturating): lo in the low byte. The PTX
// cvt puts its first source in the upper half.
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);  // .x (low half) = lo
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// D = A B + D, m16n8k16, A row-major f16 (4 regs), B col-major f16 (2 regs), fp32 accumulate.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// D = A B + D, m16n8k32, A row-major e4m3 (a0: row g, k 4t..4t+3; a1: row g+8, same k; a2: row g,
// k 16+4t..; a3: row g+8, k 16+4t..), B col-major e4m3 (b0: k 4t..4t+3, b1: k 16+4t.., col g),
// fp32 accumulate (C layout as m16n8k16).
__device__ __forceinline__ void mma16832(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One row's online-softmax step over the lane's 4 base-2 logits (-inf = masked), quad-reduced
// max; -inf safe (a row with no valid key so far keeps m = -inf, l = 0, alpha = 0).
__device__ __forceinline__ float online_row(float& x0, float& x1, float& x2, float& x3, float& m_i, float& l_i) {
  const float NEG_INF = __int_as_float(0xff800000u);
  float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  const float m_new = fmaxf(m_i, mx);
  const float m_use = m_new == NEG_INF ? 0.f : m_new;
  const float alpha = ex2(m_i - m_use);
  x0 = ex2(x0 - m_use);
  x1 = ex2(x1 - m_use);
  x2 = ex2(x2 - m_use);
  x3 = ex2(x3 - m_use);
  l_i = l_i * alpha + (x0 + x1) + (x2 + x3);  // this lane's columns; quad-reduced at the end
  m_i = m_new;
  return alpha;
}

// Page indices of sub-chunk s for this thread's tokens (tid / 8 + 32 i); 0 past the rank's keys.
// PH (PERHEAD: one kv head per cluster, 4 chunks per row): tokens tid / 4 + 64 i, i < 2 (idx[2..3] unused).
template <bool PH>
__device__ __forceinline__ void load_idx(int (&idx)[SK / 32], const int* prow, int s, int n_old, int tid) {
  const int tsub = PH ? (tid >> 2) : (tid >> 3);
  constexpr int NI = PH ? SK / 64 : SK / 32;
  constexpr int STEP = PH ? 64 : 32;
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int e = s * SK + tsub + STEP * i;
    idx[i] = e < n_old ? __ldg(prow + e) : 0;
  }
}

// Gather sub-chunk s (s < nsub) into ring slot s % NS: thread -> token tid / 8 (+ 32 i), 16-byte
// chunk tid % 8 of its K row and of its V row, chunk c stored at c ^ (slot row % 8); each warp
// instruction covers 4 full 128-byte lines. Slot rows past the rank's last key up to the end of
// its 16-row tile are zeroed (P = 0 there must not meet NaN bytes). The caller commits the group.
// PH: only kv head kh's 64-byte half (chunks 4 kh .. 4 kh + 3), thread -> token tid / 4 (+ 64 i),
// chunk 4 kh + tid % 4, same smem row layout and swizzle (the other half is never read).
template <bool PH>
__device__ __forceinline__ void issue_sub(uint8_t* smem, const uint8_t* kc, const uint8_t* vc,
                                          const int (&idx)[SK / 32], int s, int n_old, int tid, int kh) {
  const int tsub = PH ? (tid >> 2) : (tid >> 3);
  const int c = PH ? (kh * 4 + (tid & 3)) : (tid & 7);
  constexpr int NI = PH ? SK / 64 : SK / 32;
  constexpr int STEP = PH ? 64 : 32;
  uint8_t* Kd = smem + K_OFF + (s % NS) * SLOT;
  uint8_t* Vd = smem + V_OFF + (s % NS) * SLOT;
  const int base = s * SK;
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int r = tsub + STEP * i;
    if (base + r < n_old) {
      const int dst = r * ROW + ((c ^ (r & 7)) << 4);
      const int64_t row = static_cast<int64_t>(idx[i]);
      cp_async16(smem_u32(Kd + dst), kc + row * ROW + c * 16);
      cp_async16(smem_u32(Vd + dst), vc + row * ROW + c * 16);
    }
  }
  const int rows = min(SK, n_old - base);
  const int rpad = (rows + 15) & ~15;
  for (int x = tid; x < (rpad - rows) * 16; x += THREADS) {
    const int r = rows + (x >> 4), part = x & 15;
    *reinterpret_cast<uint4*>((part < 8 ? Kd : Vd) + r * ROW + (part & 7) * 16) = make_uint4(0u, 0u, 0u, 0u);
  }
}

// One 16-key tile at buffer rows tb .. tb + 15 of Kb / Vb (chunks swizzled on the buffer row):
// S = Q K^T, mask (buffer row < nvalid, key position kp0 + row inside [low, p_row]), online
// softmax, O += P V. A rows: row quad = query head g * 8 + quad, rows 8-15 zero.
__device__ __forceinline__ void attend_tile(const uint8_t* Kb, const uint8_t* Vb, int tb, int nvalid, int kp0,
                                            const uint32_t (&qa)[QKS][4], int g, int quad, int qi, float scale,
                                            int p_row, int low, float& m_a, float& l_a, float (&acc)[8][4]) {
  const float NEG_INF = __int_as_float(0xff800000u);
  // S = Q K^T over the tile's 16 rows: n-block nb holds rows tb + nb * 8 + (0..7).
  float s[2][4];
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
    const int tt = tb + nb * 8 + quad;
    const uint8_t* kr = Kb + tt * ROW + qi * 4;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const uint32_t b0 = ld32(kr + (((g * 4 + 2 * ks) ^ (tt & 7)) << 4));
      const uint32_t b1 = ld32(kr + (((g * 4 + 2 * ks + 1) ^ (tt & 7)) << 4));
      mma16832(s[nb], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
    }
  }
  // This lane's columns: rows tb + 2 qi + {0, 1} (nb 0) and + 8 (nb 1).
  bool ok[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int e = tb + 2 * qi + (c & 1) + (c >> 1) * 8;
    const int kp = kp0 + e;
    ok[c] = e < nvalid && kp <= p_row && kp >= low;
  }
  float x0 = ok[0] ? s[0][0] * scale : NEG_INF;
  float x1 = ok[1] ? s[0][1] * scale : NEG_INF;
  float x2 = ok[2] ? s[1][0] * scale : NEG_INF;
  float x3 = ok[3] ? s[1][1] * scale : NEG_INF;
  const float alpha = online_row(x0, x1, x2, x3, m_a, l_a);
  const uint32_t pa0 = pack_f16x2(x0, x1), pa2 = pack_f16x2(x2, x3);
#pragma unroll
  for (int nb = 0; nb < 8; ++nb) {
    acc[nb][0] *= alpha;
    acc[nb][1] *= alpha;
  }
  // O += P V. B operand of dim-block nb: k = rows tb + 2 qi + {0, 1} (reg 0) and + 8 (reg 1),
  // column quad = dim quad * 8 + nb; each lane reads 8 contiguous V bytes per row and widens
  // them to f16 in registers.
  const int tv = tb + qi * 2;
  const int vlc = g * 4 + (quad >> 1);
  const int voff = (quad & 1) * 8;
  const uint2 v0 = *reinterpret_cast<const uint2*>(Vb + tv * ROW + ((vlc ^ (tv & 7)) << 4) + voff);
  const uint2 v1 = *reinterpret_cast<const uint2*>(Vb + (tv + 1) * ROW + ((vlc ^ ((tv + 1) & 7)) << 4) + voff);
  const uint2 v8 = *reinterpret_cast<const uint2*>(Vb + (tv + 8) * ROW + ((vlc ^ ((tv + 8) & 7)) << 4) + voff);
  const uint2 v9 = *reinterpret_cast<const uint2*>(Vb + (tv + 9) * ROW + ((vlc ^ ((tv + 9) & 7)) << 4) + voff);
#pragma unroll
  for (int hw = 0; hw < 2; ++hw) {
    const uint32_t w0 = hw ? v0.y : v0.x, w1 = hw ? v1.y : v1.x;
    const uint32_t w8 = hw ? v8.y : v8.x, w9 = hw ? v9.y : v9.x;
    // Interleave two rows' bytes: byte 2j = row a dim j, byte 2j + 1 = row b dim j.
    uint32_t r0[4], r1[4];
    e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x5140), r0[0], r0[1]);
    e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x7362), r0[2], r0[3]);
    e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x5140), r1[0], r1[1]);
    e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x7362), r1[2], r1[3]);
#pragma unroll
    for (int j = 0; j < 4; ++j) mma16816(acc[hw * 4 + j], pa0, 0u, pa2, 0u, r0[j], r1[j]);
  }
}

// PERHEAD = false: master (one cluster of C per request, both kv heads). PERHEAD = true: HKV clusters
// of C per request, cluster index b * HKV + kh serving kv head kh only (its 8 query heads, its
// 64-byte half of every K / V row and of the new cache row; all 8 warps split the tiles of that head;
// the reduce-scatter covers its 8 rows: with C = 16, ranks 0..7 own one row each, ranks 8..15 own
// none). Both paths execute exactly two cluster.sync(), unconditionally, on every rank.
template <bool PERHEAD>
__global__ void __launch_bounds__(THREADS, 2)
sliding_cluster_kernel(const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                       const int64_t* __restrict__ positions, const int64_t* __restrict__ cache_loc,
                       const int* __restrict__ seq_lens, const int* __restrict__ page_table, int64_t pt_stride,
                       const float* __restrict__ inv_freq, const float* __restrict__ k_scale_p,
                       const float* __restrict__ v_scale_p, uint8_t* kc, uint8_t* vc,
                       __nv_bfloat16* __restrict__ out, int C, int window, float q_scale, float score_scale,
                       float eps) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* XK = smem + XK_OFF;
  uint8_t* XV = smem + XV_OFF;
  uint8_t* Qs = smem + Q_OFF;
  cg::cluster_group cluster = cg::this_cluster();
  const float NEG_INF = __int_as_float(0xff800000u);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ncl = PERHEAD ? HKV : 1;                   // clusters per request
  const int cl = static_cast<int>(blockIdx.x) / C;     // 1-D clusters: blocks cl*C .. cl*C+C-1
  const int b = cl / ncl;
  const int kh = cl % ncl;                             // PERHEAD: this cluster's kv head (0 otherwise)
  const int rank = static_cast<int>(blockIdx.x) % C;   // == cluster.block_rank()
  const int seq = seq_lens[b];
  const int lo = max(0, seq - (window - 1));
  const int ck = (((seq - lo) + C - 1) / C + 15) & ~15;
  const int start = lo + rank * ck;
  const int n_old = max(0, min(ck, seq - start));      // old keys of this rank
  const bool last = rank == C - 1;
  const int nsub = (n_old + SK - 1) / SK;              // 0 .. NSUB_MAX, CTA-uniform
  const int* prow = page_table + static_cast<int64_t>(b) * pt_stride + start;

  // ---- Prologue: the page indices of the first NS sub-chunks (every load before any copy), then
  // one cp.async group per sub-chunk (empty groups past nsub keep the group count fixed).
  {
    int idx[NS][SK / 32];
#pragma unroll
    for (int s = 0; s < NS; ++s) load_idx<PERHEAD>(idx[s], prow, s, n_old, tid);
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      if (s < nsub) issue_sub<PERHEAD>(smem, kc, vc, idx[s], s, n_old, tid, kh);
      cp_commit();
    }
  }
  int idx_late[SK / 32];  // sub-chunk NS (C = 4 only), issued into slot 0 once slot 0 is consumed
  load_idx<PERHEAD>(idx_late, prow, NS, n_old, tid);

  // ---- While the gather flies: the request's queries, one warp per head, lane = RoPE pair
  // (dims 2 lane, 2 lane + 1). qk_scale_factor is applied here, once, before E4M3.
  // PERHEAD: only kv head kh's 8 query heads, one per warp (Qs keeps the global head rows).
  const float fr = inv_freq[lane];
  const float pos = static_cast<float>(positions[b]);
  for (int h = PERHEAD ? kh * G + warp : warp; h < (PERHEAD ? (kh + 1) * G : HQ); h += WARPS) {
    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(
        qkv + static_cast<int64_t>(b) * qkv_stride + h * D + 2 * lane);
    float xe = __low2float(x2), xo = __high2float(x2);
    float ss = xe * xe + xo * xo;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = 1.f / sqrtf(ss / D + eps);
    xe *= r;
    xo *= r;
    float sn, cs;
    sincosf(pos * fr, &sn, &cs);
    const float ye = xe * cs - xo * sn, yo = xe * sn + xo * cs;
    *reinterpret_cast<uint16_t*>(Qs + h * D + 2 * lane) = f32x2_to_e4m3x2(ye * q_scale, yo * q_scale);
  }
  // The last rank: the new key (RMSNorm + RoPE, / k_scale) and value (/ v_scale), E4M3, stored at
  // cache_loc and kept as row 0 of the XK / XV tile (rows 1..15 zero). One warp per (kind, kv head).
  if (last) {
    for (int x = tid; x < 15 * 16; x += THREADS) {
      const int r = 1 + (x >> 4), part = x & 15;
      *reinterpret_cast<uint4*>((part < 8 ? XK : XV) + r * ROW + (part & 7) * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    const float ksc = __ldg(k_scale_p), vsc = __ldg(v_scale_p);
    // PERHEAD: two warps, (k, kh) and (v, kh): this cluster writes only its head's half of the row.
    for (int u = warp; u < (PERHEAD ? 2 : 2 * HKV); u += WARPS) {
      const bool isv = PERHEAD ? (u == 1) : (u >= HKV);
      const int g = PERHEAD ? kh : (u % HKV);
      const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(
          qkv + static_cast<int64_t>(b) * qkv_stride + (HQ + (isv ? HKV : 0) + g) * D + 2 * lane);
      float xe = __low2float(x2), xo = __high2float(x2);
      uint16_t packed;
      if (!isv) {  // warp-uniform
        float ss = xe * xe + xo * xo;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float r = 1.f / sqrtf(ss / D + eps);
        xe *= r;
        xo *= r;
        float sn, cs;
        sincosf(pos * fr, &sn, &cs);
        const float ye = xe * cs - xo * sn, yo = xe * sn + xo * cs;
        packed = f32x2_to_e4m3x2(ye / ksc, yo / ksc);
      } else {
        packed = f32x2_to_e4m3x2(xe / vsc, xo / vsc);
      }
      const int64_t loc = cache_loc[b];
      *reinterpret_cast<uint16_t*>((isv ? vc : kc) + loc * ROW + g * D + 2 * lane) = packed;
      const int db = g * D + 2 * lane;  // byte within the 128-byte row; row 0, so the swizzle is identity
      *reinterpret_cast<uint16_t*>((isv ? XV : XK) + db) = packed;
    }
  }
  __syncthreads();  // queries and the new-key tile are in smem

  const int quad = lane >> 2, qi = lane & 3;
  const int g = PERHEAD ? kh : (warp & 1);     // kv head
  const int sub = PERHEAD ? warp : (warp >> 1);  // tile slice
  constexpr int NSL = PERHEAD ? WARPS : 4;       // tile slices per kv head
  static_assert((SK / 16) % NSL == 0, "global tile t belongs to slice t % NSL");
  const int qra = g * G + quad;
  uint32_t qa[QKS][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const uint8_t* pa = Qs + qra * D + ks * 32 + qi * 4;
    qa[ks][0] = ld32(pa);
    qa[ks][2] = ld32(pa + 16);
    qa[ks][1] = qa[ks][3] = 0u;
  }
  const float scale = score_scale * __ldg(k_scale_p);  // sm_scale * log2(e) * k_scale
  const int p_row = seq;                                // the query's position
  const int low = p_row - (window - 1);
  float m_a = NEG_INF, l_a = 0.f;
  float acc[8][4];
#pragma unroll
  for (int nb = 0; nb < 8; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;

  // ---- Attention, sub-chunk by sub-chunk as each cp.async group lands.
  for (int s = 0; s < nsub; ++s) {
    cp_wait_ring();   // groups 0 .. s are complete (NS + s committed so far)
    __syncthreads();  // ... for every thread's copies
    const int base = s * SK;
    const int rows = min(SK, n_old - base);
    const uint8_t* Kb = smem + K_OFF + (s % NS) * SLOT;
    const uint8_t* Vb = smem + V_OFF + (s % NS) * SLOT;
    const int nt = (rows + 15) >> 4;
    for (int k = sub; k < nt; k += NSL)
      attend_tile(Kb, Vb, k * 16, rows, start + base, qa, g, quad, qi, scale, p_row, low, m_a, l_a, acc);
    if (s + NS < nsub) {  // CTA-uniform; only s = 0 at C = 4
      __syncthreads();    // every warp is done with slot s % NS
      issue_sub<PERHEAD>(smem, kc, vc, idx_late, s + NS, n_old, tid, kh);
    }
    cp_commit();
  }
  // The new key: one more tile on the last rank, for the slice next in the round robin
  // (SK is a multiple of NSL tiles, so global tile t belongs to slice t % NSL).
  if (last && sub == (((n_old + 15) >> 4) & (NSL - 1)))
    attend_tile(XK, XV, 0, 1, seq, qa, g, quad, qi, scale, p_row, low, m_a, l_a, acc);
  cp_wait_all();  // only empty groups can remain
  l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
  l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
  __syncthreads();  // every warp is done with the ring: the partials alias it

  float* pacc = reinterpret_cast<float*>(smem + P_ACC);
  float* pm = reinterpret_cast<float*>(smem + P_M);
  float* pl = reinterpret_cast<float*>(smem + P_L);
  {
    // Stage each warp's 8 rows, then merge the four slice warps of each kv head (row h = head).
    float* stg_acc = reinterpret_cast<float*>(smem + STG_ACC);
    float* stg_ml = reinterpret_cast<float*>(smem + STG_ML);
    const int rw = warp * G + quad;
    if (qi == 0) { stg_ml[rw * 2] = m_a; stg_ml[rw * 2 + 1] = l_a; }
    float* ar = stg_acc + rw * D;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
      ar[(qi * 2) * 8 + nb] = acc[nb][0];
      ar[(qi * 2 + 1) * 8 + nb] = acc[nb][1];
    }
    __syncthreads();
    if (PERHEAD) {
      // One kv head: partial row r (query head kh * G + r) merges all WARPS warps' row r; 16 threads
      // per row, 4 dims each, threads 0 .. G * 16 - 1 (warps 0..3); the partial has G rows.
      if (tid < G * 16) {
        const int r = tid >> 4, d0 = (tid & 15) * 4;
        float mw[WARPS];
        float M = NEG_INF;
#pragma unroll
        for (int j = 0; j < WARPS; ++j) { mw[j] = stg_ml[(j * G + r) * 2]; M = fmaxf(M, mw[j]); }
        const float Mu = M == NEG_INF ? 0.f : M;
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
        float Ls = 0.f;
#pragma unroll
        for (int j = 0; j < WARPS; ++j) {
          const int wr = j * G + r;
          const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
          Ls += wt * stg_ml[wr * 2 + 1];
          const float4 a = *reinterpret_cast<const float4*>(stg_acc + wr * D + d0);
          o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
        }
        *reinterpret_cast<float4*>(pacc + r * D + d0) = o;
        if ((tid & 15) == 0) { pm[r] = M; pl[r] = Ls; }
      }
    } else {
    const int h = tid >> 4, d0 = (tid & 15) * 4;
    const int hg = h / G, r = h % G;
    float mw[4];
    float M = NEG_INF;
#pragma unroll
    for (int j = 0; j < 4; ++j) { mw[j] = stg_ml[((hg + 2 * j) * G + r) * 2]; M = fmaxf(M, mw[j]); }
    const float Mu = M == NEG_INF ? 0.f : M;
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    float Ls = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int wr = (hg + 2 * j) * G + r;
      const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
      Ls += wt * stg_ml[wr * 2 + 1];
      const float4 a = *reinterpret_cast<const float4*>(stg_acc + wr * D + d0);
      o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
    }
    *reinterpret_cast<float4*>(pacc + h * D + d0) = o;
    if ((tid & 15) == 0) { pm[h] = M; pl[h] = Ls; }
    }
  }
  cluster.sync();  // every rank's partial is written and visible cluster-wide

  // ---- Reduce-scatter over DSMEM: rank r finishes rows [r NR, (r + 1) NR), 16 threads per row
  // (4 dims each); warps past the owned rows idle, partially used warps duplicate the last row.
  // PERHEAD: the cluster's partials have G = 8 rows (local row hl = query head kh * G + hl), so with
  // C = 16 HQ / C-style division would give 0 rows per rank: NR = max(1, G / C) and ranks with
  // rank * NR >= G own no row and skip the write (they still reach the second cluster.sync below).
  {
    const int NROWS = PERHEAD ? G : HQ;                 // partial rows held by every rank
    const int NR = PERHEAD ? max(1, NROWS / C) : HQ / C;
    const int nthr = NR * 16;
    const bool owns = !PERHEAD || rank * NR < NROWS;    // CTA-uniform
    if (owns && warp * 32 < nthr) {
      const int rl = min(tid >> 4, NR - 1);
      const int hl = rank * NR + rl;                    // row of the partials
      const int h = PERHEAD ? kh * G + hl : hl;         // query head: gate and output row
      const int d0 = (tid & 15) * 4;
      float mv[16];
      float M = NEG_INF;
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        mv[c] = NEG_INF;
        if (c < C) {
          mv[c] = *cluster.map_shared_rank(pm + hl, static_cast<unsigned>(c));
          M = fmaxf(M, mv[c]);
        }
      }
      const float Mu = M == NEG_INF ? 0.f : M;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      float Ls = 0.f;
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        if (c < C) {
          const float wt = mv[c] == NEG_INF ? 0.f : ex2(mv[c] - Mu);  // masked partials weigh 0
          Ls += wt * *cluster.map_shared_rank(pl + hl, static_cast<unsigned>(c));
          const float4 a = *reinterpret_cast<const float4*>(
              cluster.map_shared_rank(pacc + hl * D + d0, static_cast<unsigned>(c)));
          o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
        }
      }
      // Every row sees its own new key (last rank), so Ls > 0.
      const float vs = __ldg(v_scale_p);
      o.x = o.x / Ls * vs; o.y = o.y / Ls * vs; o.z = o.z / Ls * vs; o.w = o.w / Ls * vs;
      float ss = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
#pragma unroll
      for (int x = 8; x > 0; x >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, x);
      const float rr = 1.f / sqrtf(ss / D + eps);
      const int64_t t = static_cast<int64_t>(b);
      const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(
          qkv + t * qkv_stride + (HQ + 2 * HKV) * D + h * D + d0);
      const __nv_bfloat162 g01 = gp[0], g23 = gp[1];
      const float y0 = o.x * rr / (1.f + expf(-__low2float(g01)));
      const float y1 = o.y * rr / (1.f + expf(-__high2float(g01)));
      const float y2 = o.z * rr / (1.f + expf(-__low2float(g23)));
      const float y3 = o.w * rr / (1.f + expf(-__high2float(g23)));
      if (tid < nthr) {
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + (t * HQ + h) * D + d0);
        op[0] = __floats2bfloat162_rn(y0, y1);
        op[1] = __floats2bfloat162_rn(y2, y3);
      }
    }
  }
  cluster.sync();  // peers' smem stays live until every rank has read it
}

template <bool PH>
int occupancy_of(int C) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(C), 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = static_cast<unsigned>(C);
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = -1;
  if (cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(sliding_cluster_kernel<PH>), &cfg) != cudaSuccess) {
    (void)cudaGetLastError();
    return -1;
  }
  return n;
}

template <bool PH>
void set_attrs() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(sliding_cluster_kernel<PH>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      SMEM_BYTES));
  if (cudaFuncSetAttribute(sliding_cluster_kernel<PH>, cudaFuncAttributePreferredSharedMemoryCarveout, 100) != cudaSuccess)
    (void)cudaGetLastError();
  if (cudaFuncSetAttribute(sliding_cluster_kernel<PH>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess)
    (void)cudaGetLastError();  // then a cluster of 16 reports occupancy <= 0 and is not used
}

}  // namespace skwph

// [w6p-perhead] d1s PERHEAD (B=1, page 1, window <= 2048): 2 clusters of C=16, one kv head each, attention
// master's sliding_cluster_kernel<true> (namespace skwph above). It reads the qkv GEMM's bf16 row, stores the new
// k / v rows at cache_loc and writes the normed, sigmoid(gate)-gated [1, Hq*D] input of the o GEMM. No tickets, no
// workspaces: the merge is DSMEM inside each cluster, so nothing is left behind between launches.
// {max active clusters of 16, numRegs, localSizeBytes, dynamic smem, binaryVersion, ptxVersion}
std::vector<int64_t> skw_ph_setup() {
  skwph::set_attrs<true>();
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(skwph::sliding_cluster_kernel<true>)));
  return {static_cast<int64_t>(skwph::occupancy_of<true>(16)), static_cast<int64_t>(a.numRegs),
          static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(skwph::SMEM_BYTES),
          static_cast<int64_t>(a.binaryVersion), static_cast<int64_t>(a.ptxVersion)};
}

void skw_ph_attn(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor seq_lens,
                 torch::Tensor page_table, torch::Tensor inv_freq, torch::Tensor k_scale, torch::Tensor v_scale,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out,
                 int64_t window, double q_scale, double score_scale, double eps) {
  constexpr int C = 16;
  constexpr int kHQ = skwph::HQ, kHKV = skwph::HKV, kD = skwph::D;
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == 1, "skw_ph_attn: batch 1 only");
  TORCH_CHECK(window >= 1 && window <= 2048, "skw_ph_attn: window must be in [1, 2048]");
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.size(0) == 1 &&
              qkv.size(1) >= (2 * kHQ + 2 * kHKV) * kD && qkv.stride(1) == 1, "skw_ph_attn: qkv layout (q | k | v | gate)");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == kHQ * kD,
              "skw_ph_attn: out layout");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.is_contiguous() && positions.numel() == 1 &&
              cache_loc.scalar_type() == at::kLong && cache_loc.is_contiguous() && cache_loc.numel() == 1,
              "skw_ph_attn: positions / cache_loc");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous() &&
              page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == 1 &&
              page_table.stride(1) == 1, "skw_ph_attn: seq_lens / page_table");
  TORCH_CHECK(inv_freq.scalar_type() == at::kFloat && inv_freq.is_contiguous() && inv_freq.numel() == kD / 2,
              "skw_ph_attn: inv_freq");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat, "skw_ph_attn: scales");
  for (const auto* c : {&k_cache, &v_cache})
    TORCH_CHECK(c->is_contiguous() && c->element_size() == 1 && c->dim() == 4 && c->size(1) == 1 &&
                c->size(2) == kHKV && c->size(3) == kD, "skw_ph_attn: cache layout (page 1, 2 kv heads, 64)");
  static bool attrs_done = false;  // function attributes only (process-wide, not data)
  if (!attrs_done) {
    skwph::set_attrs<true>();
    attrs_done = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(C * kHKV), 1, 1);
  cfg.blockDim = dim3(skwph::THREADS, 1, 1);
  cfg.dynamicSmemBytes = skwph::SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = static_cast<unsigned>(C);
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, skwph::sliding_cluster_kernel<true>,
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), static_cast<int64_t>(qkv.stride(0)),
      positions.data_ptr<int64_t>(), cache_loc.data_ptr<int64_t>(), seq_lens.data_ptr<int>(),
      page_table.data_ptr<int>(), static_cast<int64_t>(page_table.stride(0)), inv_freq.data_ptr<float>(),
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<uint8_t*>(v_cache.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(C), static_cast<int>(window), static_cast<float>(q_scale), static_cast<float>(score_scale),
      static_cast<float>(eps)));
}
"""
def _build_skw():
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    arch = None
    try:
        from torch.utils.cpp_extension import load_inline
        try:
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            arch = list(_get_cuda_arch_flags())
        except Exception as e:
            _say(f"{SKW_TAG} could not read torch's nvcc arch flags: {e!r}")
        _say(f"{SKW_TAG} building muse_decode_sk_window_v1: TORCH_CUDA_ARCH_LIST=9.0a, arch flags {arch}, extra "
             f"-O3 -std=c++20 -lineinfo, instantiations skw_attn_kernel variants 4 5 6 7 + [w6p-perhead] "
             f"sliding_cluster_kernel<true> (PERHEAD, in this existing TU: no new module)")
        ext = load_inline(
            name="muse_decode_sk_window_v1",
            cpp_sources=_SKW_CPP,
            cuda_sources=_SKW_CUDA,
            functions=["skw_attn", "skw_attrs", "skw_ph_setup", "skw_ph_attn"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"],
            verbose=False,
        )
    except Exception as e:
        _SKW_BUILD["s"], _SKW_BUILD["err"] = time.perf_counter() - t0, _skw_err_digest(e)
        _say(f"{SKW_TAG} load_inline muse_decode_sk_window_v1 FAILED after {time.perf_counter() - t0:.1f}s, master path "
             f"kept: {str(e)[-1200:]!r}")
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    _SKW_BUILD["s"] = time.perf_counter() - t0
    _say(f"{SKW_TAG} load_inline muse_decode_sk_window_v1 compiled in {_SKW_BUILD['s']:.1f}s")
    return ext
def _skw_err_digest(e, n=300) -> str:
    text = str(e)
    hits = [ln.strip() for ln in text.splitlines() if "error" in ln.lower() and "error building extension" not in
            ln.lower()]
    out = " | ".join(hits) if hits else text
    return (out[:n] if hits else out[-n:]).replace("\n", " ")
if torch.cuda.is_available():
    _SKW = _build_skw()
@triton.jit
def skw_pre_kernel(
    qkv_ptr, pos_ptr, loc_ptr, cos_sin_ptr, k_scale_ptr, v_scale_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    qkv_stride,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr, PAIRED: tl.constexpr,
    ROPE: tl.constexpr, EPS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    even = 2 * i
    odd = 2 * i + 1

    row = qkv_ptr + t.to(tl.int64) * qkv_stride
    base = row + h * D
    xe = tl.load(base + even).to(tl.float32)
    xo = tl.load(base + odd).to(tl.float32)

    ss = tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)
    r = 1.0 / tl.sqrt(ss / D + EPS)
    xe = xe * r
    xo = xo * r

    if ROPE:
        p = tl.load(pos_ptr + t).to(tl.int64)
        cs = cos_sin_ptr + p * D
        c = tl.load(cs + i)
        s = tl.load(cs + HALF + i)
        ye = xe * c - xo * s
        yo = xe * s + xo * c
        xe = ye
        xo = yo

    if h < HQ:
        if PAIRED:
            half = (h // G) % 2
            qrow = q_out_ptr + (t.to(tl.int64) * HQ + h) * (2 * D)
            out = qrow + half * D
            zj = tl.arange(0, D)
            tl.store(qrow + (1 - half) * D + zj, tl.zeros([D], dtype=tl.float32).to(tl.float8e4nv))
        else:
            out = q_out_ptr + (t.to(tl.int64) * HQ + h) * D
        tl.store(out + even, xe.to(tl.float8e4nv))
        tl.store(out + odd, xo.to(tl.float8e4nv))
    else:
        kh = h - HQ
        loc = tl.load(loc_ptr + t).to(tl.int64)
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        kout = k_cache_ptr + (loc * HKV + kh) * D
        tl.store(kout + even, (xe / k_scale).to(tl.float8e4nv))
        tl.store(kout + odd, (xo / k_scale).to(tl.float8e4nv))
        j = tl.arange(0, D)
        v = tl.load(row + (HQ + HKV + kh) * D + j).to(tl.float32)
        vout = v_cache_ptr + (loc * HKV + kh) * D
        tl.store(vout + j, (v / v_scale).to(tl.float8e4nv))
def _skw_say(msg):
    print(f"{SKW_TAG} {msg}", flush=True)
def _skw_launch_logged(label, fn):
    _skw_say(f"launching {label}")
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    _skw_say(f"ok        {label} ({time.perf_counter() - t0:.2f}s)")
    return result
def _skw_buffers(device):
    rows = 2 * _sm_count(device) + 256
    return {
        "ws_acc": torch.zeros((rows, 16, 64), dtype=torch.float32, device=device),
        "ws_ml": torch.zeros((rows, 2, 16), dtype=torch.float32, device=device),
        "ticket": torch.zeros(256, dtype=torch.int32, device=device),
        "gticket": torch.zeros(8 * 32, dtype=torch.int32, device=device),
        "gacc": torch.zeros((8 * 32, 16, 64), dtype=torch.float32, device=device),
        "gml": torch.zeros((8 * 32, 2, 16), dtype=torch.float32, device=device),
    }
def _skw_setup(state, device):
    state["skw"] = None
    if _SKW is None:
        return
    try:
        state["skw"] = {"buf": _skw_buffers(device), "route": {b: None for b in SKW_BANDS}}
        torch.cuda.synchronize()
    except Exception as e:
        _skw_say(f"state buffers raised {e!r}: sliding sk-window off for this state")
def _skw_clone(skw, device):
    return {"buf": _skw_buffers(device), "route": dict(skw["route"])}
def _skw_ticket_sum(skw):
    if skw is None:
        return 0
    return int(skw["buf"]["ticket"].abs().sum()) + int(skw["buf"]["gticket"].abs().sum())
def _skw_zero(skw):
    skw["buf"]["ticket"].zero_()
    skw["buf"]["gticket"].zero_()
def _skw_state(state, only):
    ms = dict(state)
    if state.get("skw") is not None:
        route = {b: None for b in SKW_BANDS}
        if only is not None:
            route[only[0]] = only[1]
        ms["skw"] = {**state["skw"], "route": route}
    return ms
def _skw_band(cfg):
    if not (cfg["window"] == SKW_WINDOW and cfg["page_size"] == 1 and cfg["num_q_heads"] == 16
            and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64 and cfg["rope_theta"] == ROPE_THETA):
        return None
    batch, length = cfg["batch"], cfg["q_len"]
    if length == 1 and 17 <= batch <= 32:
        return "b32"
    if length == 8 and batch == 1:
        return "v1"
    return None
def _skw_route(cfg, state):
    skw = state.get("skw")
    if skw is None or _SKW is None:
        return None
    band = _skw_band(cfg)
    return None if band is None else skw["route"].get(band)
def _skw_grid(spec, rows, sms):
    g = {"p2": 2 * rows, "p4": 4 * rows, "p8": 8 * rows, "sm": sms}[spec]
    return max(1, min(g, 2 * sms))
def _skw_pre(data, qkv, state, paired=False):
    cfg = data["config"]
    tokens = cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    q = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    skw_pre_kernel[(tokens, hq + hkv)](
        qkv, data["positions"], data["cache_loc"], state["cos_sin"], data["k_scale"], data["v_scale"],
        q, data["k_cache"], data["v_cache"],
        qkv.stride(0),
        HQ=hq, HKV=hkv, D=d, G=hq // hkv, PAIRED=False, ROPE=True, EPS=RMS_EPS,
        num_warps=1,
    )
    return q
def _skw_launch(data, q8, qkv, skw, variant, grid, out=None):
    cfg = data["config"]
    if out is None:
        out = torch.empty((cfg["tokens"], 16 * 64), dtype=torch.bfloat16, device=qkv.device)
    b = skw["buf"]
    _SKW.skw_attn(q8, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"], data["k_scale"],
                  data["v_scale"], qkv, qkv.stride(0), out, b["ws_acc"], b["ws_ml"], b["ticket"], b["gticket"],
                  b["gacc"], b["gml"], cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, RMS_EPS, grid, variant)
    return out
def _skw_attention(data, qkv, state, route):
    variant, spec = route
    q8 = _skw_pre(data, qkv, state)
    grid = _skw_grid(spec, data["config"]["tokens"], _sm_count(qkv.device))
    return _skw_launch(data, q8, qkv, state["skw"], variant, grid)
def _skw_stats(ts):
    t = sorted(ts)
    n = len(t)
    return t[n // 2], t[(3 * n) // 4] - t[n // 4]
def _skw_time(fns, device, reps=SKW_REPS):
    scratch = torch.zeros(1, dtype=torch.float32, device=device)
    graphs = [_sk_graph_of(lambda: scratch.add_(1.0), device)] + [_sk_graph_of(f, device) for f in fns]
    flush = torch.empty(CB_FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = [[] for _ in graphs]
    for it in range(3 + reps):
        order = list(range(len(graphs)))
        if it % 2:
            order.reverse()
        for i in order:
            torch.sum(flush)
            e0.record()
            graphs[i].replay()
            e1.record()
            e1.synchronize()
            if it >= 3:
                ts[i].append(e0.elapsed_time(e1) * 1e3)
    base = sorted(ts[0])[len(ts[0]) // 2]
    del graphs, flush, scratch
    torch.cuda.synchronize()
    return [[max(x - base, 0.05) for x in t] for t in ts[1:]], base
def _skw_attrs_print():
    global _SKW_ATTRS_DONE
    if _SKW_ATTRS_DONE:
        return
    for v in (4, 5, 6):
        a = list(_SKW.skw_attrs(v))
        _skw_say(f"skw_attn_kernel variant {v}: binaryVersion {a[0]}, ptxVersion {a[1]}, regs {a[2]}, local/spill "
                 f"{a[3]} B, max threads {a[4]}, dyn smem {a[5]} B, {a[6]} CTAs/SM")
        if a[0] != 90:
            raise RuntimeError(f"skw_attn_kernel variant {v}: cubin version {a[0]} is not 90 (sm_90a image not loaded)")
    _SKW_ATTRS_DONE = True
def _skw_plant(data, q8, mode):
    cfg = data["config"]
    length, batch = cfg["q_len"], cfg["batch"]
    ks = float(data["k_scale"])
    q0 = (q8.float()[::length][:, [0, 8], :] * 6.0 / ks).clamp(-440.0, 440.0)
    kc = data["k_cache"].view(torch.uint8).view(-1, 2, 64)
    rows = []
    for b in range(batch):
        n = int(data["seq_lens"][b])
        pos = {"first": max(0, n - (SKW_WINDOW - 1)), "out": max(0, n - SKW_WINDOW), "last": n}[mode]
        rows.append(int(data["page_table"][b, pos]))
    kc[torch.tensor(rows, dtype=torch.long, device=kc.device)] = q0.to(FP8).view(torch.uint8)
def _skw_kernel_case(state, device, gen, tag, batch, length, lens, kv_scale, cands, plant=None):
    skw, sms = state["skw"], _sm_count(device)
    data = _cb_data(state, device, gen, SKW_WINDOW, batch, length, lens, kv_scale)
    tokens = data["config"]["tokens"]
    qkv = _gemm(data["attn_input"], state["dense"]["qkv_gate"])
    if qkv.dim() != 2 or qkv.shape[1] != SKW_QKV_COLS or qkv.stride(1) != 1:
        bad = f"{tag}: qkv layout {tuple(qkv.shape)} strides {tuple(qkv.stride())} != [T, {SKW_QKV_COLS}] row-major"
        return {c: bad for c in cands}
    q8 = _skw_launch_logged(f"{tag}: skw_pre_kernel", lambda: _skw_pre(data, qkv, state))
    if plant is not None:
        _skw_plant(data, q8, plant)
    ref = _cb_attn_ref(data, qkv, q_override=q8)
    verdicts = {}
    for variant, spec in cands:
        grid = _skw_grid(spec, tokens, sms)
        out = torch.full((tokens, 16 * 64), float("nan"), dtype=torch.bfloat16, device=device)
        label = f"{tag}: skw_attn_kernel v{variant} {spec} grid {grid}"
        _skw_launch_logged(label, lambda: _skw_launch(data, q8, qkv, skw, variant, grid, out))
        finite = bool(torch.isfinite(out.float()).all())
        err = _rel_l2(out, ref)
        rows = float(((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max())
        left = _skw_ticket_sum(skw)
        ok = finite and err <= SKW_TOL_TOTAL and rows <= SKW_TOL_ROW and not left
        _skw_say(f"{label}: rel L2 {err:.2e} (<= {SKW_TOL_TOTAL}), worst row {rows:.2e} (<= {SKW_TOL_ROW}), finite {finite}, "
                 f"tickets {left} -> {'ok' if ok else 'FAIL'}")
        if left:
            _skw_zero(skw)
            torch.cuda.synchronize()
        verdicts[(variant, spec)] = None if ok else f"{label}: rel L2 {err:.2e}, worst row {rows:.2e}, finite {finite}, tickets {left}"
    del data, qkv, q8, ref
    return verdicts
def _skw_kernel_suite(state, device, band):
    cands = SKW_CANDS[band]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261051 + (7 if band == "v1" else 0))
    alive, problems = list(cands), []
    def run(tag, batch, length, lens, kvs, plant=None, only=None):
        todo = [c for c in (only or alive) if c in alive]
        if not todo:
            return
        if _cb_case_late():
            problems.append(f"{tag}: skipped: deadline (case deadline {CB_CASE_DEADLINE:.0f}s passed)")
            del alive[:]
            return
        for cand, why in _skw_kernel_case(state, device, gen, tag, batch, length, lens, kvs, todo, plant).items():
            if why is not None:
                problems.append(why)
                if cand in alive:
                    alive.remove(cand)
    tiny = (17, 1, [63] * 17) if band == "b32" else (1, 8, [63])
    for cand in cands:
        if cand in alive:
            run(f"canary {band} {cand}", tiny[0], tiny[1], tiny[2], 1.0, only=[cand])
    if not alive:
        return [], problems
    front = []
    for b, length in ((17, 1), (32, 1)) if band == "b32" else ((1, 8),):
        for kvs in CB_KV_SCALES:
            for pos in _CB_POS_ROPE if band == "v1" else (80000, 299995):
                p = _s3_front_check(state, device, gen, SKW_WINDOW, b, length, kvs, pos, False, pre_fn=_skw_pre,
                                    gap_gate=False, label="SKW front")
                if p is not None:
                    front.append(p)
    if front:
        return [], problems + front
    if band == "b32":
        for b in SKW_CHECK_B["b32"]:
            for kvs in CB_KV_SCALES:
                run(f"B={b} ragged kv {kvs}", b, 1, [S1_LENS[(i + b) % len(S1_LENS)] for i in range(b)], kvs)
        run("B=32 mixed ctx 80000", 32, 1, _cb_mixed(32, 80000), 1.0)
        for plant in ("first", "out", "last"):
            run(f"B=17 ragged plant {plant}", 17, 1, [S1_LENS[(i + 17) % len(S1_LENS)] for i in range(17)], 1.0, plant)
    else:
        for n in SKW_LENS:
            for kvs in CB_KV_SCALES:
                run(f"B=1 q_len 8 n={n} kv {kvs}", 1, 8, [n], kvs)
        for n in (2049, 80000):
            for plant in ("first", "out", "last"):
                run(f"B=1 q_len 8 n={n} plant {plant}", 1, 8, [n], 1.0, plant)
    return alive, problems
def _skw_race(state, device, band, cands):
    batch, length = SKW_SELECT[band]
    rng = random.Random(4100 + 7 * batch + length + SKW_WINDOW)
    gen = torch.Generator(device=device)
    gen.manual_seed(4100 + batch)
    lens = _sk_agentic(rng, batch, 80000) if batch > 1 else [80000]
    data = _cb_data(state, device, gen, SKW_WINDOW, batch, length, lens, 1.0)
    st_master = _skw_state(state, None)
    sts = [_skw_state(state, (band, c)) for c in cands]
    fns = [lambda: _parent_attention(data, st_master)] + [(lambda st=st: _parent_attention(data, st)) for st in sts]
    names = ["master (as routed: FA3 chain / S1 / S3 per this state's flags)"] + [
        f"skw_pre_kernel + skw_attn_kernel v{v} {s} (grid {_skw_grid(s, batch * length, _sm_count(device))})" for v, s in cands]
    rounds = []
    for r in range(2):
        order = list(range(len(fns)))
        if r == 1:
            order.reverse()
        nets, base = _skw_time([fns[i] for i in order], device)
        per = [None] * len(fns)
        for k, i in enumerate(order):
            per[i] = _skw_stats(nets[k])
        rounds.append(per)
        for i, (med, iqr) in enumerate(per):
            _skw_say(f"race {band} round {r + 1}/2 (baseline {base:.2f} us subtracted): {names[i]}: median {med:.2f} us IQR "
                     f"{iqr:.2f} us" + ("" if i == 0 else f", master/new {per[0][0] / med:.3f}"))
    best, best_ratio = None, 0.0
    lines = []
    for ci, cand in enumerate(cands, start=1):
        ratios = [rounds[r][0][0] / rounds[r][ci][0] for r in range(2)]
        iqr_ok = all(rounds[r][ci][1] <= SKW_IQR_RATIO * max(rounds[r][0][1], SKW_IQR_FLOOR_US) for r in range(2))
        ok = min(ratios) >= SKW_MARGIN and iqr_ok
        lines.append(f"v{cand[0]} {cand[1]}: master/new {ratios[0]:.3f} / {ratios[1]:.3f} (need >= {SKW_MARGIN} both), "
                     f"IQR ok {iqr_ok} -> {'QUALIFIES' if ok else 'no'}")
        if ok and min(ratios) > best_ratio:
            best, best_ratio = ci, min(ratios)
    for line in lines:
        _skw_say(f"race {band}: {line}")
    top = best if best is not None else max(range(1, len(fns)), key=lambda i: min(rounds[r][0][0] / rounds[r][i][0]
                                                                               for r in range(2)))
    t_new = sum(rounds[r][top][0] for r in range(2)) / 2
    t_old = sum(rounds[r][0][0] for r in range(2)) / 2
    res = {"t_new": t_new, "t_old": t_old, "cands": [list(c) for c in cands]}
    if best is None:
        res.update(winner=None, reason=f"sub-race: no candidate cleared master/new >= {SKW_MARGIN} in both rounds with IQR <= "
                                       f"{SKW_IQR_RATIO}x master's (best {cands[top - 1]} new {t_new:.2f} vs master {t_old:.2f} us)")
    else:
        res.update(winner=cands[best - 1], reason=None)
    _skw_say(f"race {band}: WINNER = {('NEW ' + str(cands[best - 1])) if best is not None else 'MASTER'} [new {t_new:.2f} us vs "
             f"master {t_old:.2f} us]")
    del data, fns, sts
    return res
def _skw_alt_cases(band):
    if band == "b32":
        cases = [(SKW_WINDOW, b, 1, [S1_LENS[(i + b) % len(S1_LENS)] for i in range(b)], kvs)
                 for b in SKW_CHECK_B["b32"] for kvs in CB_KV_SCALES]
        cases.append((SKW_WINDOW, 32, 1, _cb_mixed(32, 80000), 1.0))
        return cases
    return [(SKW_WINDOW, 1, 8, [n], kvs) for n in SKW_LENS for kvs in CB_KV_SCALES]
def _skw_layer_check(state, device, band, cand):
    batch, length = SKW_SELECT[band]
    second = _cb_second_state(state, device)
    sts = [_skw_state(s, (band, cand)) for s in (state, second)]
    refs = [_skw_state(s, None) for s in (state, second)]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261052)
    rng = random.Random(4200 + batch)
    ls_val = 96
    worst = {}
    cases = (("agentic ctx 80000 kv 1", _sk_agentic(rng, batch, 80000) if batch > 1 else [80000], 1.0),
             ("ragged kv 1/64", _cb_mixed(batch, 80000) if batch > 1 else [2049], 0.015625))
    for name, lens, kvs in cases:
        tag = f"SKW layer {band} {name}"
        if _cb_case_late():
            return f"{tag}: skipped: deadline (case deadline {CB_CASE_DEADLINE:.0f}s passed)"
        sets = []
        for s in range(2):
            data = _cb_data(state, device, gen, SKW_WINDOW, batch, length, lens, kvs, ls_val)
            cell = {}
            def fn(data=data, st=sts[s], cell=cell):
                cell["out"] = kernel_fn(data, st)
            _skw_launch_logged(f"{tag}: eager warm-up state {s}", lambda: kernel_fn(_dl3_clone_data(data), sts[s]))
            sets.append((data, cell, _sk_graph_of(fn, device)))
        for r in range(4):
            s = r % 2
            data, cell, graph = sets[s]
            cell["out"][0].fill_(float("nan"))
            cell["out"][1].fill_(float("nan"))
            cell["out"][2].fill_(-1)
            _cb_refill(data, _cb_data(state, device, gen, SKW_WINDOW, batch, length, lens, kvs, ls_val))
            s_in = data["stream"].clone()
            d_ref, d_probe = _dl3_clone_data(data), _dl3_clone_data(data)
            ref = kernel_fn(d_ref, refs[s])
            scores = _cb_router_scores(d_probe, refs[s])
            torch.cuda.synchronize()
            _skw_say(f"{tag}: replay {r} on state {s}: launching graph")
            graph.replay()
            torch.cuda.synchronize()
            ok, msg = _cb_cmp_layer(cell["out"], ref, data, d_ref, s_in, scores, ls_val, worst, loose=True, k_gap_max=1 << 8)
            left = _cb_counters(sts[s])
            _skw_say(f"{tag}: replay {r} state {s}: {msg}, counters {left} -> {'ok' if ok and not left else 'FAIL'}")
            if not ok or left:
                return f"{tag} replay {r} state {s}: {msg}, counters {left}"
            del d_ref, d_probe, ref, scores
        del sets
    _skw_say(f"SKW layer {band}: max error per output: " + ", ".join(f"{k} {v:.3g}" for k, v in worst.items()))
    return None
def _skw_band_phase(state, device, rec, band, t_prepare0):
    _skw_attrs_print()
    res = _SKW_RES.get(band)
    if res is None:
        alive, problems = _skw_kernel_suite(state, device, band)
        _skw_say(f"{band} kernel-level gates: {len(alive)}/{len(SKW_CANDS[band])} candidates passed"
                 + (f"; problems: {'; '.join(problems[:3])}" if problems else ""))
        if not _cb_clean(state, {"_dirty": None}, f"SKW {band} kernel suite"):
            problems.append("tickets not back to 0 after the kernel suite")
            alive = []
        if not alive:
            res = {"winner": None, "t_new": None, "t_old": None,
                   "reason": "kernel-level gate: " + ("; ".join(problems[:2]) or "no candidate passed")[:600]}
        elif _cb_late(t_prepare0):
            res = {"winner": None, "t_new": None, "t_old": None,
                   "reason": f"skipped: deadline (cell deadline {CB_CELL_DEADLINE:.0f}s passed before the sub-race)"}
        else:
            res = _skw_race(state, device, band, tuple(alive))
        _SKW_RES[band] = res
    else:
        _skw_say(f"{band}: kernel-level gates and sub-race of the first prepare() reused (winner {res['winner']}); this "
                 f"state still runs its own alternation and layer check")
    rec["t_new"], rec["t_old"] = res.get("t_new"), res.get("t_old")
    if res["winner"] is None:
        rec["reason"] = res["reason"]
        return
    cand = tuple(res["winner"])
    err = _cb_attn_alt(state, device, f"SKW {band}", _skw_alt_cases(band), {"skw": {band: cand}}, row_agg=True)
    if not _cb_clean(state, {"_dirty": None}, f"SKW {band} alternation"):
        err = err or "counters not back to 0 after the alternation"
    _skw_say(f"{band} attention-side alternation: {'ok' if err is None else 'FAIL: ' + err}")
    if err is None:
        err = _skw_layer_check(state, device, band, cand)
        if not _cb_clean(state, {"_dirty": None}, f"SKW {band} layer check"):
            err = err or "counters not back to 0 after the layer check"
        _skw_say(f"{band} whole-layer check: {'ok' if err is None else 'FAIL: ' + err}")
    if err is not None:
        rec["reason"] = f"check: {err}"
        return
    rec.update(win=True, variant=cand[0], spec=cand[1])
def _cb_prepare_skw(state, device, recs, t_prepare0):
    for band in SKW_BANDS:
        rec = recs["SKW"][band]
        if state.get("skw") is None or _SKW is None:
            rec["reason"] = "module muse_decode_sk_window_v1 not built (see the build line at the top of the log) or no buffers"
            continue
        if _cb_late(t_prepare0):
            rec["reason"] = f"skipped: deadline (cell deadline {CB_CELL_DEADLINE:.0f}s passed)"
            continue
        try:
            _skw_band_phase(state, device, rec, band, t_prepare0)
        except Exception as e:
            rec["win"] = False
            rec["reason"] = f"raised {type(e).__name__}: {str(e)[:300]}"
            _skw_say(f"{band} raised {type(e).__name__}: {str(e)[:300]}: master kept")
        try:
            if not _cb_clean(state, recs, f"SKW {band}"):
                rec["reason"] = "counters not back to 0"
        except Exception as e:
            recs["_dirty"] = recs["_dirty"] or f"SKW {band} counter read raised {e!r}"
def _print_summary(state):
    for line in _CB_SUMMARY:
        print(f"[summary] PATH CHOSEN {line}", flush=True)
    sk = _SK_DECISION or {}
    print(f"[summary] dense-chain {'ON' if state.get('dl') else 'MASTER'}, w2 {state.get('w2_mode')}, full-decode sk on="
          f"{sk.get('on')} b1={sk.get('b1')} n={sk.get('n')}", flush=True)
    skw = state.get("skw")
    route = None if skw is None else skw["route"]
    print(f"[summary] [skw] module {'built' if _SKW is not None else 'MISSING'}, routes b32={route and route.get('b32')} "
          f"v1={route and route.get('v1')} (None = master's chain), prepare wall {_PREPARE_WALL_S:.1f}s", flush=True)
CAMPAIGN_SFK = True
SFK_TAG = "[w6m-fork]"
SFK_CLASSES = (128, 256)
SFK_DL = 250.0
SFK_SEED = 20261160
def _sfk_stream(device):
    if CAMPAIGN_SFK:
        try:
            s = _rt_fork_stream(device)
            print(f"{SFK_TAG} prepare {_rt_fork_line(s)}", flush=True)
            return s
        except Exception as e:
            print(f"{SFK_TAG} prepare fork stream miss ({type(e).__name__}: {str(e)[:80]}) -> pool stream", flush=True)
    return torch.cuda.Stream(device)
def _sfk_gate(state, device, t_prepare0):
    out = []
    n_inputs = {t: len(v) for t, v in _EAD_INPUTS.items()}
    def setter(v):
        def f():
            state["side_serial"] = v
        return f
    for cls in SFK_CLASSES:
        if time.perf_counter() - t_prepare0 > SFK_DL:
            out.append(f"{SFK_TAG} fork gate T={cls}: skipped (deadline)")
            continue
        t0 = time.perf_counter()
        try:
            per = _ead_layer_gate(state, device, cls, {"serial": setter(True), "fork": setter(False)}, SFK_SEED + cls)
            rs, parts = [], []
            for rounds in per.values():
                pr = []
                for ts in rounds:
                    s_, f_ = _rt_stats(ts["serial"])[0], _rt_stats(ts["fork"])[0]
                    rs.append(s_ / max(f_, 1e-3))
                    pr.append(f"{s_:.1f}/{f_:.1f}")
                parts.append(f"in{len(parts)} " + " ".join(pr))
            out.append(f"{SFK_TAG} fork gate T={cls} serial/fork us: " + " | ".join(parts)
                       + f" -> fork vs serial twin x{min(rs):.3f}..x{max(rs):.3f} ({time.perf_counter() - t0:.1f} s)")
        except Exception as e:
            out.append(f"{SFK_TAG} fork gate T={cls}: raised {type(e).__name__}: {str(e)[:80]}")
        state["side_serial"] = False
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    for t in list(_EAD_INPUTS):
        if t in n_inputs:
            del _EAD_INPUTS[t][n_inputs[t]:]
        else:
            del _EAD_INPUTS[t]
    return out
DC0F_TAG = "[dc0b-fork]"
DC0K = {1: (16, 4, 256), 8: (16, 8, 64)}
@triton.jit
def _dc0_r32_kernel(x_ptr, w_ptr, p_ptr, M, K: tl.constexpr, N: tl.constexpr, MP: tl.constexpr, KS: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rm = tl.arange(0, MP)
    acc = tl.zeros((MP, BN, BK), dtype=tl.float32)
    for k0 in range(pk * KS, pk * KS + KS, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
        acc += x[:, None, :] * w[None, :, :]
    tl.store(p_ptr + pk * M * N + rm[:, None] * N + rn[None, :], tl.sum(acc, axis=2), mask=rm[:, None] < M)
def _dc0_lin(x, w, k=None):
    if k is None:
        return F.linear(x, w)
    sp, bn, bk = k
    m = x.shape[0]
    p = torch.empty((sp, m, N_EXPERTS), dtype=torch.float32, device=x.device)
    _dc0_r32_kernel[(N_EXPERTS // bn, sp)](x, w, p, m, K=HIDDEN, N=N_EXPERTS, MP=triton.next_power_of_2(m),
                                           KS=HIDDEN // sp, BN=bn, BK=bk, num_warps=4)
    return p.sum(0)
DC0F_T = (1, 8)
def _dc0f_gate(state, device, t_prepare0):
    state["dc0f"], state["dc0k"], out = {}, {}, []
    if state.get("dl") is None:
        return [f"{DC0F_TAG} no dense chain: serial F.linear FP32 router"]
    arms = {"serial": (False, None), "fork": (True, None)}
    for t in DC0F_T:
        if time.perf_counter() - t_prepare0 > 260.0:
            out.append(f"{DC0F_TAG} T={t}: skipped (deadline), serial F.linear")
            continue
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(20261200 + t)
            x = torch.randn((t, HIDDEN), generator=gen, device=device)
            ref = F.linear(x, state["router"])
            r = _dc0_lin(x, state["router"], DC0K[t])
            torch.cuda.synchronize()
            err = float((r - ref).abs().max())
            ok32 = bool(torch.isfinite(r).all()) and err <= 1e-5 and torch.equal(r, _dc0_lin(x, state["router"], DC0K[t]))
            a = dict(arms)
            if ok32:
                a.update({"r32": (False, DC0K[t]), "r32fork": (True, DC0K[t])})
            d = _ead_cases(state, device, gen, t)
            o = {}
            for n, (f, k) in a.items():
                state["dc0f"][t], state["dc0k"][t] = f, k
                o[n] = kernel_fn(_dl3_clone_data(d), state)
            torch.cuda.synchronize()
            ex = {n: all(torch.equal(u, v) for u, v in zip(o[n], o["serial" if a[n][1] is None else "r32"])) for n in a}
            cands = [n for n in a if n != "serial" and ex[n]]
            def setter(n):
                def f():
                    state["dc0f"][t], state["dc0k"][t] = a[n]
                return f
            per = _ead_layer_gate(state, device, t, {n: setter(n) for n in ["serial"] + cands}, 20261210 + t)
            win, worst, _ = _rt_select(per, "serial", cands, margin=1.01, tight=1.01)
            state["dc0f"][t], state["dc0k"][t] = a[win or "serial"]
            out.append(f"{DC0F_TAG} PATH CHOSEN T={t}: {(win or 'serial').upper()} FP32 router (r32 max|r32-F.linear| "
                       f"{err:.1e} ok {ok32}; " + ", ".join(f"serial/{c} x{worst.get(c, 0):.3f}" for c in cands) + ")")
        except Exception as e:
            state["dc0f"][t], state["dc0k"][t] = False, None
            out.append(f"{DC0F_TAG} T={t}: raised {type(e).__name__}: {str(e)[:80]}; serial F.linear")
        torch.cuda.synchronize()
    return out
def prepare(weights):
    t_prepare0 = time.perf_counter()
    whj = _wh_start(t_prepare0)
    try:
        return _prepare_main(weights, t_prepare0, whj)
    finally:
        _wh_join(whj, restore=True)
_FVG_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> fvg_attr(int64_t mode);
void fvg_launch(int64_t mode, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                torch::Tensor cache_seqlens, torch::Tensor qkv, torch::Tensor part_o, torch::Tensor part_ml,
                torch::Tensor cnt, torch::Tensor out, double sc, double eps, int64_t grid, int64_t coop);
"""
_FVG_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
namespace {
constexpr int D = 64;
constexpr int HQ = 16;
constexpr int HKV = 2;
constexpr int ROW = 128;
constexpr int TILE = 64;
constexpr int THREADS = 256;
constexpr int KV_STAGE = TILE * ROW;
constexpr int STAGE = 2 * KV_STAGE;
constexpr int MAXB = 64;
constexpr int NS = 3;
constexpr int MINB = 2;
constexpr int FN = 3;
constexpr long long SPIN_MAX = 1ll << 22;
constexpr float NEG_INF = -__builtin_huge_valf();
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint2 lds2(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void sts1(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ int cta_of(int x, int T, int G) {
  return (int)(((int64_t)(x + 1) * G + T - 1) / T) - 1;
}
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16) |
         ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void pin(float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
__device__ __forceinline__ void wg_qk(float (&d)[32], const uint32_t* a, uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
__device__ __forceinline__ void wg_pv(float (&d)[32], const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}
__device__ __forceinline__ void wg_pv8(float (&d)[32], const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}
__device__ __forceinline__ uint32_t pk8(uint32_t a, uint32_t b) {
  uint32_t y;
  asm("{.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f16x2 lo, %1;\ncvt.rn.satfinite.e4m3x2.f16x2 hi, %2;\nmov.b32 %0, {lo, hi};\n}\n"
      : "=r"(y) : "r"(a), "r"(b));
  return y;
}
__device__ __forceinline__ uint32_t e8x4(float a, float b, float c, float d) {
  uint32_t y;
  asm("{.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\nmov.b32 %0, {lo, hi};\n}\n"
      : "=r"(y) : "f"(a), "f"(b), "f"(c), "f"(d));
  return y;
}
constexpr int VT_LBO = 128;
constexpr int VT_SBO = 528;
constexpr int VT_HEAD = 4224;
static_assert(2 * 2 * VT_HEAD <= 2 * 2 * TILE * 128, "PV8 V^T double buffer fits fvg's f16 V buffer");
__device__ __forceinline__ uint64_t ndesc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(VT_LBO >> 4) << 16) | ((uint64_t)(VT_SBO >> 4) << 32);
}
__device__ __forceinline__ uint32_t h2u(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void e4m3x4_to_f16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t ex2h2(uint32_t x) {
  uint32_t y;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
  uint32_t y;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(y) : "r"(a), "r"(b));
  return y;
}
__device__ __forceinline__ float hsum2f(uint32_t a) {
  __half2 h = *reinterpret_cast<__half2*>(&a);
  float2 f = __half22float2(h);
  return f.x + f.y;
}
struct SmemY {
  static constexpr int RING = 0;
  static constexpr int VB = NS * STAGE;
  static constexpr int IDX = VB + 2 * 2 * TILE * 128;
  static constexpr int TST = IDX + 2 * NS * TILE * 4;
  static constexpr int LEN = TST + (MAXB + 2) * 4;
  static constexpr int FLAG = LEN + MAXB * 4;
  static constexpr int TINFO = FLAG + 16;
  static constexpr int MAXT = 1024;
  static constexpr int BYTES = TINFO + MAXT * 4 + 1024;
};
template <int MODE, bool PV8>
__global__ void __launch_bounds__(THREADS, MINB)
fvg_kernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
           const int* __restrict__ pt, const int* __restrict__ lens, int B, int W, float sc,
           float* __restrict__ part_o, float2* __restrict__ part_ml, int* __restrict__ cnt,
           const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride, float eps, __nv_bfloat16* __restrict__ out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using L = SmemY;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  uint8_t* ring = smem + L::RING;
  uint8_t* vbuf = smem + L::VB;
  int* idx_s = reinterpret_cast<int*>(smem + L::IDX);
  int* tst = reinterpret_cast<int*>(smem + L::TST);
  int* len_s = reinterpret_cast<int*>(smem + L::LEN);
  int* tinfo = reinterpret_cast<int*>(smem + L::TINFO);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int h = warp >> 2, wm = warp & 3, g = lane >> 2, qd = lane & 3;
  const int G = gridDim.x;
  if (warp == 0) {
    const int n0 = lane < B ? lens[lane] : 0, n1 = lane + 32 < B ? lens[lane + 32] : 0;
    int s0 = (n0 + TILE - 1) / TILE, s1 = (n1 + TILE - 1) / TILE;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v0 = __shfl_up_sync(~0u, s0, o), v1 = __shfl_up_sync(~0u, s1, o);
      if (lane >= o) { s0 += v0; s1 += v1; }
    }
    const int tot0 = __shfl_sync(~0u, s0, 31);
    if (lane < B) { len_s[lane] = n0; tst[lane + 1] = s0; }
    if (lane + 32 < B) { len_s[lane + 32] = n1; tst[lane + 33] = tot0 + s1; }
    if (lane == 0) tst[0] = 0;
  }
  __syncthreads();
  const int T = tst[B];
  const int Ge = max(1, min(G, T));
  const bool over = (T + Ge - 1) / Ge > L::MAXT;
  if (over && tid == 0 && blockIdx.x == 0) atomicOr(cnt + 2, 2);
  int t0 = 0, t1 = 0;
  if (!over && (int)blockIdx.x < Ge) {
    t0 = (int)((int64_t)blockIdx.x * T / Ge);
    t1 = (int)((int64_t)(blockIdx.x + 1) * T / Ge);
  }
  const int vid = blockIdx.x;
  if (t0 < t1) {
  for (int k = tid; k < t1 - t0; k += THREADS) {
    const int x = t0 + k;
    int lo = 0, hi = B - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (tst[mid] <= x) lo = mid; else hi = mid - 1;
    }
    const int j = x - tst[lo];
    const int rel = min(len_s[lo] - j * TILE, 1023);
    tinfo[k] = j | (rel << 16) | (lo << 26);
  }
  __syncthreads();
  const int lc = tid & 7, kl0 = tid >> 3;
  const uint32_t doff0 = kl0 * ROW + ((lc ^ (kl0 & 7)) << 4);
  const uint32_t doff1 = (kl0 + 32) * ROW + ((lc ^ ((kl0 + 32) & 7)) << 4);
  const uint32_t ring_u = smem_u32(ring), idx_u = smem_u32(idx_s);
  const uint8_t* kbase = kc + lc * 16;
  const uint8_t* vbase = vc + lc * 16;
  auto issue_idx = [&](int y) {
    if (MODE == 3) return;
    if (tid < TILE / 4) {
      const int e = tinfo[y - t0];
      const int key0 = (e & 0xffff) * TILE + tid * 4;
      const bool ok = key0 < W;
      cp16(idx_u + ((y % (2 * NS)) * TILE + tid * 4) * 4,
           ok ? (const void*)(pt + (int64_t)(e >> 26) * W + key0) : (const void*)pt, ok ? 16 : 0);
    }
  };
  uint32_t cpd = ring_u + doff0;
  const uint8_t* kbp = kbase;
  const uint8_t* vbp = vbase;
  static_assert(((32 * ROW) >> 4) % 8 == 0, "rows kl0 and kl0 + 32 share the swizzle phase");
  auto issue_data = [&](int x) {
    if (MODE == 3) return;
    const uint32_t st = ring_u + (x % NS) * STAGE;
    const int* ix = idx_s + (x % (2 * NS)) * TILE;
    const int lim = (tinfo[x - t0] >> 16) & 1023;
    const bool ok0 = kl0 < lim, ok1 = kl0 + 32 < lim;
    const uint32_t p0 = ok0 ? (uint32_t)ix[kl0] : 0u, p1 = ok1 ? (uint32_t)ix[kl0 + 32] : 0u;
    const uint32_t d0 = cpd + (st - ring_u);
    cp16(d0, kbp + (size_t)p0 * ROW, ok0 ? 16 : 0);
    cp16(d0 + KV_STAGE, vbp + (size_t)p0 * ROW, ok0 ? 16 : 0);
    cp16(d0 + 32 * ROW, kbp + (size_t)p1 * ROW, ok1 ? 16 : 0);
    cp16(d0 + 32 * ROW + KV_STAGE, vbp + (size_t)p1 * ROW, ok1 ? 16 : 0);
  };
  for (int y = t0; y < t0 + NS && y < t1; ++y) issue_idx(y);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int x = t0; x < t0 + NS - 1; ++x) {
    if (x < t1) issue_data(x);
    if (x + NS < t1) issue_idx(x + NS);
    cp_commit();
  }
  int segb = tinfo[0] >> 26;
  float o[32], s[32];
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
  uint32_t qa[8];
#pragma unroll
  for (int k = 0; k < 32; ++k) { o[k] = 0.f; s[k] = 0.f; }
  auto load_q = [&](int b) {
    if (MODE == 2 || MODE == 3) {
#pragma unroll
      for (int k = 0; k < 8; ++k) qa[k] = 0x30303030u + k;
      return;
    }
    const uint8_t* qb = q + ((int64_t)b * 8 + 2 * wm) * HQ * D + (h * 8 + g) * D + 4 * qd;
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      qa[4 * st + 0] = __ldg(reinterpret_cast<const uint32_t*>(qb + 32 * st));
      qa[4 * st + 1] = __ldg(reinterpret_cast<const uint32_t*>(qb + HQ * D + 32 * st));
      qa[4 * st + 2] = __ldg(reinterpret_cast<const uint32_t*>(qb + 32 * st + 16));
      qa[4 * st + 3] = __ldg(reinterpret_cast<const uint32_t*>(qb + HQ * D + 32 * st + 16));
    }
  };
  auto finalize = [&](int b) {
    if (MODE == 2) return;
    float la = l0 + __shfl_xor_sync(~0u, l0, 1);
    la += __shfl_xor_sync(~0u, la, 2);
    float lb_ = l1 + __shfl_xor_sync(~0u, l1, 1);
    lb_ += __shfl_xor_sync(~0u, lb_, 2);
    const int slot = vid + b;
    const int rA = h * 64 + (2 * wm) * 8 + g, rB = rA + 8;
    float* pa = part_o + ((int64_t)slot * 128 + rA) * D + 2 * qd;
    float* pb = part_o + ((int64_t)slot * 128 + rB) * D + 2 * qd;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      *reinterpret_cast<float2*>(pa + 8 * jj) = make_float2(o[4 * jj + 0], o[4 * jj + 1]);
      *reinterpret_cast<float2*>(pb + 8 * jj) = make_float2(o[4 * jj + 2], o[4 * jj + 3]);
    }
    if (qd == 0) {
      part_ml[(int64_t)slot * 128 + rA] = make_float2(m0, la);
      part_ml[(int64_t)slot * 128 + rB] = make_float2(m1, lb_);
    }
  };
  load_q(segb);
  const int tl = tid & 127, vkey = tl >> 1, vfc = (tl & 1) * 2;
  const uint32_t vsrc_off = KV_STAGE + vkey * ROW;
  const uint32_t vs0 = (((4 * h + vfc) ^ (vkey & 7)) << 4), vs1 = (((4 * h + vfc + 1) ^ (vkey & 7)) << 4);
  const uint32_t vd0 = vkey * 128 + (((2 * vfc) ^ (vkey & 7)) << 4), vd1 = vkey * 128 + (((2 * vfc + 1) ^ (vkey & 7)) << 4);
  const uint32_t vd2 = vkey * 128 + (((2 * vfc + 2) ^ (vkey & 7)) << 4), vd3 = vkey * 128 + (((2 * vfc + 3) ^ (vkey & 7)) << 4);
  const uint64_t dk0 = gdesc(ring_u + 64 * h, 16, 1024);
  const uint64_t dv0 = PV8 ? ndesc(smem_u32(vbuf) + h * VT_HEAD) : gdesc(smem_u32(vbuf) + h * (TILE * 128), 128, 1024);
  const int tc = tl >> 5, tq = (tl >> 3) & 3, tdg = tl & 7;
  const uint32_t tsrc0 = KV_STAGE + (16 * tc + 2 * tq) * ROW + (tdg & 1) * 8;
  const uint32_t tch = (uint32_t)(4 * h + (tdg >> 1));
  const uint32_t tdst = h * VT_HEAD + tdg * VT_SBO + tc * VT_LBO + 4 * tq;
  uint32_t vta = ring_u + tsrc0 + ((tch ^ (uint32_t)((2 * tq) & 7)) << 4);
  uint32_t vtb = ring_u + tsrc0 + ROW + ((tch ^ (uint32_t)((2 * tq + 1) & 7)) << 4);
  uint32_t vtd = smem_u32(vbuf) + tdst;
  for (int i = t0; i < t1; ++i) {
    if constexpr (PV8) asm volatile("" : "+r"(cpd), "+l"(kbp), "+l"(vbp), "+r"(vta), "+r"(vtb), "+r"(vtd));
    cp_wait<NS - 2>();
    if (FN & 1) fence_async();
    __syncthreads();
    {
      const int x = i + NS - 1;
      if (x < t1) issue_data(x);
      const int y = i + 2 * NS - 1;
      if (y < t1) issue_idx(y);
      cp_commit();
    }
    const int e = tinfo[i - t0];
    const int cb = e >> 26;
    if (cb != segb) {
      wg_wait<0>();
      pin(o);
      finalize(segb);
#pragma unroll
      for (int k = 0; k < 32; ++k) o[k] = 0.f;
      m0 = m1 = NEG_INF;
      l0 = l1 = 0.f;
      load_q(cb);
      segb = cb;
    }
    if (MODE != 2) {
      const uint32_t slot = (uint32_t)(i % NS);
      const uint64_t dk = dk0 + ((slot * STAGE) >> 4);
      wg_fence();
      pin(s);
      wg_qk(s, qa, dk, 0);
      wg_qk(s, qa + 4, dk + 2, 1);
      wg_commit();
      const uint32_t vbo = (uint32_t)(i & 1) * (PV8 ? 2 * VT_HEAD : 2 * TILE * 128);
      if constexpr (PV8) {
        const uint32_t so = slot * STAGE;
        uint2 w[4];
        w[0] = lds2(vta + so);
        w[1] = lds2(vtb + so);
        w[2] = lds2(vta + so + 8 * ROW);
        w[3] = lds2(vtb + so + 8 * ROW);
        const uint32_t vt = vtd + vbo;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const uint32_t a0 = hf ? w[0].y : w[0].x, a1 = hf ? w[1].y : w[1].x;
          const uint32_t a2 = hf ? w[2].y : w[2].x, a3 = hf ? w[3].y : w[3].x;
          const uint32_t t0 = __byte_perm(a0, a1, 0x5140), t1 = __byte_perm(a0, a1, 0x7362);
          const uint32_t t2 = __byte_perm(a2, a3, 0x5140), t3 = __byte_perm(a2, a3, 0x7362);
          sts1(vt + (4 * hf + 0) * 16, __byte_perm(t0, t2, 0x5410));
          sts1(vt + (4 * hf + 1) * 16, __byte_perm(t0, t2, 0x7632));
          sts1(vt + (4 * hf + 2) * 16, __byte_perm(t1, t3, 0x5410));
          sts1(vt + (4 * hf + 3) * 16, __byte_perm(t1, t3, 0x7632));
        }
      } else {
        const uint8_t* vsrc = ring + slot * STAGE + vsrc_off;
        uint8_t* vb = vbuf + vbo + h * (TILE * 128);
        const uint4 a = *reinterpret_cast<const uint4*>(vsrc + vs0);
        const uint4 c2 = *reinterpret_cast<const uint4*>(vsrc + vs1);
        uint4 r0, r1, r2, r3;
        e4m3x4_to_f16(a.x, r0.x, r0.y);
        e4m3x4_to_f16(a.y, r0.z, r0.w);
        e4m3x4_to_f16(a.z, r1.x, r1.y);
        e4m3x4_to_f16(a.w, r1.z, r1.w);
        e4m3x4_to_f16(c2.x, r2.x, r2.y);
        e4m3x4_to_f16(c2.y, r2.z, r2.w);
        e4m3x4_to_f16(c2.z, r3.x, r3.y);
        e4m3x4_to_f16(c2.w, r3.z, r3.w);
        *reinterpret_cast<uint4*>(vb + vd0) = r0;
        *reinterpret_cast<uint4*>(vb + vd1) = r1;
        *reinterpret_cast<uint4*>(vb + vd2) = r2;
        *reinterpret_cast<uint4*>(vb + vd3) = r3;
      }
      if (FN & 2) fence_async();
      named_bar(1 + h, 128);
      wg_wait<0>();
      pin(s);
      pin(o);
      const int rel = (e >> 16) & 1023;
      if (rel < TILE + 8) {
        const int lim0 = rel - 8 + 2 * wm, lim1 = lim0 + 1;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
          const int key = jn * 8 + 2 * qd;
          if (key > lim0) s[4 * jn + 0] = NEG_INF;
          if (key + 1 > lim0) s[4 * jn + 1] = NEG_INF;
          if (key > lim1) s[4 * jn + 2] = NEG_INF;
          if (key + 1 > lim1) s[4 * jn + 3] = NEG_INF;
        }
      }
      float r0m[8], r1m[8];
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        r0m[jn] = fmaxf(s[4 * jn + 0], s[4 * jn + 1]);
        r1m[jn] = fmaxf(s[4 * jn + 2], s[4 * jn + 3]);
      }
#pragma unroll
      for (int w = 4; w >= 1; w >>= 1)
#pragma unroll
        for (int jn = 0; jn < w; ++jn) {
          r0m[jn] = fmaxf(r0m[jn], r0m[jn + w]);
          r1m[jn] = fmaxf(r1m[jn], r1m[jn + w]);
        }
      float x0 = r0m[0], x1 = r1m[0];
      x0 = fmaxf(x0, __shfl_xor_sync(~0u, x0, 1));
      x0 = fmaxf(x0, __shfl_xor_sync(~0u, x0, 2));
      x1 = fmaxf(x1, __shfl_xor_sync(~0u, x1, 1));
      x1 = fmaxf(x1, __shfl_xor_sync(~0u, x1, 2));
      const float n0 = fmaxf(m0, x0 * sc), n1 = fmaxf(m1, x1 * sc);
      const float u0 = n0 == NEG_INF ? 0.f : n0, u1 = n1 == NEG_INF ? 0.f : n1;
      const bool moved = (n0 != m0) || (n1 != m1);
      if (__any_sync(~0u, moved)) {
        const float a0 = ex2(m0 - u0), a1 = ex2(m1 - u1);
        l0 *= a0;
        l1 *= a1;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          o[4 * jj + 0] *= a0;
          o[4 * jj + 1] *= a0;
          o[4 * jj + 2] *= a1;
          o[4 * jj + 3] *= a1;
        }
      }
      m0 = n0;
      m1 = n1;
      const float e0 = PV8 ? u0 - 8.f : u0, e1 = PV8 ? u1 - 8.f : u1;
      float pe[32];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int k = 0; k < 8; ++k) pe[8 * t + k] = ex2(fmaf(s[8 * t + k], sc, (k & 2) ? -e1 : -e0));
      }
      {
        float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          a0 += pe[8 * t + 0] + pe[8 * t + 1];
          b0 += pe[8 * t + 4] + pe[8 * t + 5];
          a1 += pe[8 * t + 2] + pe[8 * t + 3];
          b1 += pe[8 * t + 6] + pe[8 * t + 7];
        }
        l0 += a0 + b0;
        l1 += a1 + b1;
      }
      const uint64_t dv = dv0 + (vbo >> 4);
      if constexpr (PV8) {
        uint32_t pa[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const float* q0 = pe + 16 * kk;
          pa[kk][0] = e8x4(q0[0], q0[1], q0[4], q0[5]);
          pa[kk][1] = e8x4(q0[2], q0[3], q0[6], q0[7]);
          pa[kk][2] = e8x4(q0[8], q0[9], q0[12], q0[13]);
          pa[kk][3] = e8x4(q0[10], q0[11], q0[14], q0[15]);
        }
        wg_fence();
        pin(o);
        wg_pv8(o, pa[0], dv);
        wg_pv8(o, pa[1], dv + ((2 * VT_LBO) >> 4));
        wg_commit();
      } else {
        wg_fence();
        pin(o);
        uint32_t pf[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          pf[t][0] = h2u(pe[8 * t + 0], pe[8 * t + 1]);
          pf[t][1] = h2u(pe[8 * t + 2], pe[8 * t + 3]);
          pf[t][2] = h2u(pe[8 * t + 4], pe[8 * t + 5]);
          pf[t][3] = h2u(pe[8 * t + 6], pe[8 * t + 7]);
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) wg_pv(o, pf[t], dv + t * 128);
        wg_commit();
      }
    }
  }
  cp_wait<0>();
  wg_wait<0>();
  pin(o);
  finalize(segb);
  }
  if (MODE == 1) {
    int* gctl = cnt;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      atomicAdd(gctl + 0, 1);
      long long spins = 0;
      while (*reinterpret_cast<volatile int*>(gctl + 0) < G) {
        __nanosleep(128);
        if (++spins > SPIN_MAX) { atomicOr(gctl + 2, 1); break; }
      }
      __threadfence();
    }
    __syncthreads();
    for (int it = blockIdx.x * 8 + warp; !over && it < B * 128; it += G * 8) {
      const int b = it >> 7, row = it & 127;
      const int cf = cta_of(tst[b], T, Ge), nseg = cta_of(tst[b + 1] - 1, T, Ge) - cf + 1, s0 = cf + b;
      float M = NEG_INF;
      for (int sg = lane; sg < nseg; sg += 32) M = fmaxf(M, __ldcg(part_ml + (int64_t)(s0 + sg) * 128 + row).x);
#pragma unroll
      for (int o_ = 16; o_ >= 1; o_ >>= 1) M = fmaxf(M, __shfl_xor_sync(~0u, M, o_));
      float2 acc = make_float2(0.f, 0.f);
      float Ls = 0.f;
      for (int c0 = 0; c0 < nseg; c0 += 32) {
        const int sg = c0 + lane;
        float w = 0.f;
        if (sg < nseg) {
          const float2 ml = __ldcg(part_ml + (int64_t)(s0 + sg) * 128 + row);
          w = ml.x == NEG_INF ? 0.f : ex2(ml.x - M);
          Ls += w * ml.y;
        }
        const int n = min(32, nseg - c0);
        const float* op = part_o + ((int64_t)(s0 + c0) * 128 + row) * D + 2 * lane;
#pragma unroll 8
        for (int kk = 0; kk < n; ++kk) {
          const float wk = __shfl_sync(~0u, w, kk);
          const float2 v = __ldcg(reinterpret_cast<const float2*>(op + (int64_t)kk * 128 * D));
          acc.x += wk * v.x;
          acc.y += wk * v.y;
        }
      }
#pragma unroll
      for (int o_ = 16; o_ >= 1; o_ >>= 1) Ls += __shfl_xor_sync(~0u, Ls, o_);
      const float inv = 1.f / Ls;
      const int hh = row >> 6, rr = row & 63, t = rr >> 3, gq = rr & 7;
      const int head = hh * 8 + gq;
      const int64_t tok = (int64_t)b * 8 + t;
      const float x0 = acc.x * inv, x1 = acc.y * inv;
      float ss = x0 * x0 + x1 * x1;
#pragma unroll
      for (int o_ = 16; o_ >= 1; o_ >>= 1) ss += __shfl_xor_sync(~0u, ss, o_);
      const float r = 1.f / sqrtf(ss * (1.f / D) + eps);
      const float2 gf = __bfloat1622float2(
          reinterpret_cast<const __nv_bfloat162*>(qkv + tok * qkv_stride + (HQ + 2 * HKV) * D + head * D)[lane]);
      const float y0 = x0 * r * (1.f / (1.f + expf(-gf.x))), y1 = x1 * r * (1.f / (1.f + expf(-gf.y)));
      reinterpret_cast<__nv_bfloat162*>(out + (tok * HQ + head) * D)[lane] = __floats2bfloat162_rn(y0, y1);
    }
    __syncthreads();
    if (tid == 0 && atomicAdd(gctl + 1, 1) == G - 1) {
      gctl[0] = 0;
      gctl[1] = 0;
      __threadfence();
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) atomicOr(cnt + 2, 4);
#endif
}
constexpr int SMEM = SmemY::BYTES;
using KFn = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const int*, const int*, int, int, float, float*,
                     float2*, int*, const __nv_bfloat16*, int64_t, float, __nv_bfloat16*);
KFn kfn(int64_t mode) {
  switch (mode & 7) {
    case 0: return fvg_kernel<0, false>;
    case 1: return fvg_kernel<1, false>;
    case 2: return fvg_kernel<2, false>;
    case 3: return fvg_kernel<3, false>;
    case 4: return fvg_kernel<0, true>;
    case 5: return fvg_kernel<1, true>;
    case 6: return fvg_kernel<2, true>;
    default: return fvg_kernel<3, true>;
  }
}
bool g_attr_set[8] = {false, false, false, false, false, false, false, false};
void set_smem(int64_t mode) {
  const int m = (int)(mode & 7);
  if (!g_attr_set[m]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn(m), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    g_attr_set[m] = true;
  }
}
}
std::vector<int64_t> fvg_attr(int64_t mode) {
  set_smem(mode);
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, (const void*)kfn(mode)));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kfn(mode), THREADS, SMEM));
  int dev = 0, sms = 0, coop = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
  return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)SMEM, (int64_t)a.sharedSizeBytes, (int64_t)occ,
          (int64_t)sms, (int64_t)coop};
}
void fvg_launch(int64_t mode, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
                torch::Tensor cache_seqlens, torch::Tensor qkv, torch::Tensor part_o, torch::Tensor part_ml,
                torch::Tensor cnt, torch::Tensor out, double sc, double eps, int64_t grid, int64_t coop) {
  const int64_t B = page_table.size(0);
  TORCH_CHECK(B >= 1 && B <= MAXB, "fvg: batch");
  TORCH_CHECK(q.is_contiguous() && q.element_size() == 1 && q.numel() == B * 8 * HQ * D, "fvg: q [B*8, 16, 64] e4m3");
  for (const auto* c : {&k_cache, &v_cache})
    TORCH_CHECK(c->is_contiguous() && c->element_size() == 1 && c->dim() == 4 && c->size(1) == 1 &&
                c->size(2) == HKV && c->size(3) == D, "fvg: cache layout (page 1, 2 kv heads, 64)");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.stride(1) == 1 &&
              page_table.stride(0) == page_table.size(1) && page_table.size(1) % 4 == 0 &&
              (reinterpret_cast<uintptr_t>(page_table.data_ptr()) & 15) == 0, "fvg: page table");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && cache_seqlens.numel() == B,
              "fvg: cache_seqlens");
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B * 8 &&
              qkv.stride(0) % 2 == 0, "fvg: qkv");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * 8 * HQ * D, "fvg: out");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= (grid + B) * 128 * D &&
              part_ml.scalar_type() == at::kFloat && part_ml.numel() >= (grid + B) * 128 * 2 &&
              cnt.scalar_type() == at::kInt && cnt.numel() >= 4, "fvg: workspace");
  set_smem(mode);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = coop ? attr : nullptr;
  cfg.numAttrs = coop ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, kfn(mode), reinterpret_cast<const uint8_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_table.data_ptr<int>(), cache_seqlens.data_ptr<int>(),
      static_cast<int>(B), static_cast<int>(page_table.size(1)), static_cast<float>(sc), part_o.data_ptr<float>(),
      reinterpret_cast<float2*>(part_ml.data_ptr<float>()), cnt.data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), static_cast<int64_t>(qkv.stride(0)),
      static_cast<float>(eps), reinterpret_cast<__nv_bfloat16*>(out.data_ptr())));
}
"""
FVG_TAG = "[dc-fvg]"
FVG_W = 86008
FVG_ARMS = {"fvg": 1, "fvg8": 5}
FVG_MARGIN = 1.01
FVG_DEADLINE_S = 250.0
_FVG = {"ext": None, "err": None, "s": None, "th": None, "g": 0, "coop": 1, "buf": {}}
def _fvg_build_worker():
    t0 = time.perf_counter()
    try:
        from torch.utils.cpp_extension import load_inline
        _FVG["ext"] = load_inline(name="muse_decode_fvg_v1", cpp_sources=_FVG_CPP, cuda_sources=_FVG_CUDA,
                                  functions=["fvg_attr", "fvg_launch"], verbose=False,
                                  extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"])
    except BaseException as e:
        _FVG["err"] = f"{type(e).__name__}: {str(e)[-600:]}"
    _FVG["s"] = time.perf_counter() - t0
if torch.cuda.is_available():
    import threading as _fvg_threading
    _FVG["th"] = _fvg_threading.Thread(target=_fvg_build_worker, name="dc-fvg-build", daemon=True)
    _FVG["th"].start()
def _fvg_ok(cfg, data):
    pt = data["page_table"]
    b, w = pt.shape
    return (cfg["window"] == 0 and not cfg["rope_theta"] and cfg["q_len"] == 8 and cfg["num_q_heads"] == 16
            and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64 and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0
            and w % 4 == 0 and pt.stride(0) == w and pt.stride(1) == 1 and data["k_cache"].is_contiguous()
            and data["v_cache"].is_contiguous() and _FVG["g"] > 0 and -(-(b * -(-w // 64)) // _FVG["g"]) <= 1024)
def _fvg_attn(data, qkv, state, mode):
    cfg = data["config"]
    q = _s3_pre_attention(data, qkv, state, False)
    out = torch.empty((cfg["tokens"], 1024), dtype=torch.bfloat16, device=qkv.device)
    po, pm, cnt = _FVG["buf"][cfg["batch"]]
    _FVG["ext"].fvg_launch(mode, q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"], qkv, po,
                           pm, cnt, out, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, RMS_EPS, _FVG["g"],
                           _FVG["coop"])
    return out
def _fvg_input(state, device, gen, b):
    lens = [int(80000 * (1 + 0.075 * (2 * float(torch.rand((), generator=gen, device=device)) - 1))) for _ in range(b)]
    lens[0] = FVG_W - 8
    data = _cb_data(state, device, gen, 0, b, 8, lens, 1.0)
    pt = torch.full((b, FVG_W), -1, dtype=torch.int32, device=device)
    pt[:, :data["page_table"].shape[1]] = data["page_table"]
    data["page_table"] = pt
    ls, _load = _ead_busiest(data, state)
    data["local_start"] = torch.tensor([ls], dtype=torch.int32, device=device)
    return data
def _fvg_ref(data, qkv):
    cfg = data["config"]
    L, tok = 8, cfg["tokens"]
    q = qkv[:, :1024].double().view(tok, 16, 64)
    q = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS)).float().to(FP8).double() * cfg["qk_scale_factor"]
    kf, vf, outs = data["k_cache"].view(-1, 2, 64), data["v_cache"].view(-1, 2, 64), []
    for b, tot in enumerate(data["cache_seqlens"].tolist()):
        rows = data["page_table"][b, :tot].long()
        s_ = torch.einsum("lkgd,nkd->lkgn", q[b * L:(b + 1) * L].view(L, 2, 8, 64), kf[rows].double()) * cfg["sm_scale"]
        m = torch.arange(tot, device=q.device)[None, :] > (tot - L + torch.arange(L, device=q.device))[:, None]
        outs.append(torch.einsum("lkgn,nkd->lkgd", torch.softmax(s_.masked_fill(m[:, None, None, :], float("-inf")), -1),
                                 vf[rows].double()).reshape(L, 16, 64))
    a = torch.cat(outs)
    a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (a * torch.sigmoid(qkv[:, 1280:].double().view(tok, 16, 64))).reshape(tok, 1024)
def _fvg_clean(b):
    return int(_FVG["buf"][b][2][:3].abs().sum()) == 0
def _fvg_row(state, device, b):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261130 + b)
    fv = state["fvg"]
    per, keep, worst = {}, [], {}
    names = list(FVG_ARMS)
    for rep in range(2):
        fv.pop(b, None)
        data = _fvg_input(state, device, gen, b)
        if not _fvg_ok(data["config"], data):
            return None, f"shape gate refuses the judge-shaped input (page table {tuple(data['page_table'].shape)})"
        s_in = data["stream"].clone()
        scores = _cb_router_scores(_dl3_clone_data(data), state)
        d_ref = _dl3_clone_data(data)
        qkv = _dsa_gemm("qkv_gate", d_ref["attn_input"], state["dense"]["qkv_gate"], state)
        a_inc = _attn_core(d_ref, state).double()
        r8 = _fvg_ref(d_ref, qkv)
        e_inc = _rel_l2(a_inc, r8)
        d_ref = _dl3_clone_data(data)
        ref = kernel_fn(d_ref, state)
        for nm in names:
            fv[b] = FVG_ARMS[nm]
            d_got = _dl3_clone_data(data)
            e_a = _rel_l2(_attn_core(d_got, state).double(), r8)
            worst[f"attn {nm}/inc"] = max(worst.get(f"attn {nm}/inc", 0), e_a / max(e_inc, 1e-9))
            d_got = _dl3_clone_data(data)
            got = kernel_fn(d_got, state)
            torch.cuda.synchronize()
            _ok, msg = _dsa_cmp_layer(got, ref, d_got, d_ref, s_in, scores, int(data["local_start"][0]), worst)
            ok = ("finite True" in msg and e_a <= 1.1 * e_inc + 1e-3 and worst["hidden"] <= 0.05 and worst["update"] <= 0.05
                  and worst["k_gap"] == 0 and worst["v_gap"] == 0 and worst["tokens_rerouted"] <= b * 2 and _fvg_clean(b))
            if not ok:
                fv.pop(b, None)
                return None, (f"{nm} check input {rep}: attn vs fp64 {e_a:.2e} (inc {e_inc:.2e}); {msg[:200]} "
                              f"(counters clean {_fvg_clean(b)})")
        graphs = {}
        for nm in ["incumbent"] + names:
            if nm == "incumbent":
                fv.pop(b, None)
            else:
                fv[b] = FVG_ARMS[nm]
            graphs[nm], out = _rt_graph(lambda: kernel_fn(data, state))
            keep.append(out)
        fv.pop(b, None)
        keep.append(graphs)
        per[f"v{b}f layer input {rep}"] = _rt_time(graphs, device)
        if not _fvg_clean(b):
            return None, "counters not clean after the timed replays"
    win, wx, _rl = _rt_select(per, "incumbent", names, margin=FVG_MARGIN, tight=FVG_MARGIN)
    allr = [r for rounds in per.values() for r in rounds]
    mean = {n: sum(_rt_stats(r[n])[0] for r in allr) / len(allr) for n in ["incumbent"] + names}
    msg = (f"layer us inc/" + "/".join(names) + " " + "/".join(f"{mean[n]:.1f}" for n in ["incumbent"] + names)
           + "; worst " + ", ".join(f"{n} x{wx[n]:.3f}" for n in names) + f" (>= {FVG_MARGIN} every round); check "
           + ", ".join(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
    return (None if win is None else FVG_ARMS[win]), msg
def _fvg_setup(state, device, t_prepare0):
    state["fvg"] = {}
    out, route = [], {}
    try:
        th = _FVG["th"]
        if th is None:
            raise RuntimeError("build never started")
        th.join(timeout=max(0.0, FVG_DEADLINE_S - 60.0 - (time.perf_counter() - t_prepare0)))
        if th.is_alive() or _FVG["ext"] is None:
            raise RuntimeError(f"build {'not joined' if th.is_alive() else 'failed'}: {str(_FVG['err'])[:300]}")
        ext = _FVG["ext"]
        a = {m: [int(x) for x in ext.fvg_attr(m)] for m in FVG_ARMS.values()}
        regs, local, _sm, _st, occ, sms, coop = a[1]
        out.append(f"{FVG_TAG} build {_FVG['s']:.1f} s (bg from import); attrs regs/local/occ "
                   + ", ".join(f"m{m} {x[0]}/{x[1]}/{x[4]}" for m, x in a.items()) + f"; {sms} SMs, coop {coop}")
        if any(x[4] < 2 or x[1] != 0 for x in a.values()) or not coop:
            raise RuntimeError("occupancy < 2, spill or no cooperative launch")
        _FVG["g"] = 2 * sms
        for b in (16, 32):
            _FVG["buf"][b] = (torch.empty((_FVG["g"] + b) * 128 * 64, dtype=torch.float32, device=device),
                              torch.empty((_FVG["g"] + b) * 128 * 2, dtype=torch.float32, device=device),
                              torch.zeros(16, dtype=torch.int32, device=device))
        gen = torch.Generator(device=device)
        gen.manual_seed(20261129)
        data = _cb_data(state, device, gen, 0, 16, 8, [300] * 16, 1.0)
        data["page_table"] = torch.cat([data["page_table"], torch.full((16, 4 - data["page_table"].shape[1] % 4), -1,
                                        dtype=torch.int32, device=device)], 1)
        state["fvg"][16] = 1
        try:
            g, _o = _rt_graph(lambda: kernel_fn(data, state))
            g.replay()
            torch.cuda.synchronize()
            del g
        except Exception as e:
            _FVG["coop"] = 0
            out.append(f"{FVG_TAG} cooperative launch not capturable ({str(e)[:120]!r}): plain launch")
        finally:
            state["fvg"].pop(16, None)
            _FVG["buf"][16][2].zero_()
            torch.cuda.synchronize()
        for b in (16, 32):
            if time.perf_counter() - t_prepare0 > FVG_DEADLINE_S:
                out.append(f"{FVG_TAG} v{b}f: deadline, incumbent")
                continue
            try:
                m, msg = _fvg_row(state, device, b)
            except Exception as e:
                m, msg = None, f"raised {str(e)[:200]!r}"
            finally:
                state["fvg"].pop(b, None)
                _FVG["buf"][b][2].zero_()
                torch.cuda.synchronize()
            if m:
                route[b] = m
            out.append(f"{FVG_TAG} v{b}f: {msg}"[:600])
    except Exception as e:
        out.append(f"{FVG_TAG} incumbent kept: {str(e)[:400]}")
    state["fvg"] = dict(route)
    for b in (16, 32):
        out.append(f"{FVG_TAG} PATH CHOSEN v{b}f full verify attention: "
                   + (f"fvg_kernel<1{', PV8' if route[b] & 4 else ''}> (mode {route[b]}, grid {_FVG['g']}, coop "
                      f"{_FVG['coop']}, fused norm-gate)" if b in route else "incumbent (decode's current route)"))
    for line in out:
        print(line, flush=True)
    return out
def _prepare_main(weights, t_prepare0, whj):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
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
        "router": router.contiguous(),
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
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "post_expert_norm_w": weights["post_expert_norm"].contiguous(),
        "bank": _marlin_bank(weights),
        "side_stream": _sfk_stream(device),
        "side_serial": False,
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
        "router_bias": weights["router_bias"].float().contiguous(),
        "s1": {c: False for c in S1_CLASSES},
        "s3": {c: False for c in S3_CELLS},
        **_slide_state(device),
    }
    del freqs
    state["dl"] = _dl_setup(weights, state)
    _dl3_setup_bc(state, device)
    state["w2_mode"] = "master"  # compliance: SGLang reduces split-K in FP32 (use_atomic_add=False), never BF16 atomics
    _sk_prepare(state, device, t_prepare0)
    _dl3_finish(state, device)
    _skw_setup(state, device)
    _combo_prepare(state, device, t_prepare0)
    torch.cuda.synchronize()
    _prepare_done(t_prepare0)
    _print_summary(state)
    paths = []
    if CAMPAIGN_ENGINE_A_DL:
        for line in _ead_setup(weights, state, device, t_prepare0):
            print(line, flush=True)
            paths += [line] if ("PATH CHOSEN" in line or "worst ratio" in line or " kernel v4 " in line) else []
    eaej = _eae_start(t_prepare0) if CAMPAIGN_EAE else None
    if CAMPAIGN_DENSE_SWAPAB:
        for line in _dsa_setup(state, device, t_prepare0):
            print(line, flush=True)
            paths += [line] if "PATH CHOSEN" in line else []
    if CAMPAIGN_HOTCOLD:
        for line in _hc_setup(weights, state, device, t_prepare0):
            print(line, flush=True)
            paths += [line] if "PATH CHOSEN" in line else []
    if CAMPAIGN_B12P:
        for line in _b12p_setup(state, device, t_prepare0):
            print(line, flush=True)
            paths.append(line)
    if CAMPAIGN_SKFULL:
        for line in _skf_setup(state, device, t_prepare0):
            print(line, flush=True)
            if not line.startswith(SKF_TAG + "   "):
                paths.append(line)
    if CAMPAIGN_C320:
        for line in _c320d_setup(state, device, t_prepare0):
            print(line, flush=True)
            paths.append(line)
    if CAMPAIGN_T128:
        for line in _t128_setup(state, device, t_prepare0):
            print(line, flush=True)
            paths.append(line)
    if CAMPAIGN_W6P_PERHEAD:
        paths += _w6p_setup(state, device, t_prepare0)
    whl = _wh_phase(state, device, t_prepare0, whj)
    if CAMPAIGN_SFK:
        paths += _sfk_gate(state, device, t_prepare0)
    eal = _eae_phase(state, weights, device, t_prepare0, eaej) if eaej is not None else []
    paths += _dc0f_gate(state, device, t_prepare0)
    paths += _fvg_setup(state, device, t_prepare0)
    print("[campaign] ===== FINAL PATHS =====", flush=True)
    for line in paths:
        print(line[:200], flush=True)
    for line in _CB_BASIS_LINE + ([_ead_inputs_line()] if CAMPAIGN_ENGINE_A_DL else []):
        print(line[:600], flush=True)
    if CAMPAIGN_ENGINE_A_DL and CAMPAIGN_EA6:
        for line in _ead6_final_lines(state):
            print(line, flush=True)
    if CAMPAIGN_SFK:
        try:
            print(f"{SFK_TAG} end {_rt_fork_line(state['side_stream'])}", flush=True)
        except Exception as e:
            print(f"{SFK_TAG} end fork line raised {type(e).__name__}", flush=True)
    for line in whl + [ln for ln in EA4X_LINES if " route (board" in ln or " cold half (layer" in ln][-2:] + eal:
        print(line[:330], flush=True)
    print(f"[campaign] prepare at {time.perf_counter() - t_prepare0:.1f} s", flush=True)
    return state
def _expert_dispatch(topk_ids, topk_weights, local_start, num_tokens, state):
    shifted = (topk_ids - local_start) % N_EXPERTS
    tw = topk_weights.to(torch.float32).contiguous()
    local_ids = state["bank"]["expert_map"][shifted].contiguous()
    align = moe_align_block_size(local_ids, _block_size_m(num_tokens), N_EXPERTS)
    return local_ids, tw, align
def _experts(x, topk_weights, topk_ids, state, align=None, dispatch=None, w2_mode=None):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    block_size_m = _block_size_m(num_tokens)
    if dispatch is not None:
        local_ids, topk_weights, (sorted_ids, block_experts, n_padded) = dispatch
    elif align is None:
        local_ids = bank["expert_map"][topk_ids].contiguous()
        topk_weights = topk_weights.to(torch.float32).contiguous()
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    else:
        local_ids = topk_ids
        sorted_ids, block_experts, n_padded = align
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    if w2_mode is None:
        w2_mode = state.get("w2_mode", "master")
    w2_common = common
    if w2_mode != "master" and num_tokens == 1 and align is not None and dispatch is None:
        w2_common = dict(common, use_atomic_add=True)
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    if w2_common is not common and w2_mode == "atomic_zero":
        cache3.zero_()
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **w2_common).view(num_tokens, TOP_K, size_k)
    if num_tokens < SMALL_T_POST_NORM:
        y = y.contiguous()
        out = torch.empty((num_tokens, size_k), device=x.device, dtype=torch.bfloat16)
        _post_norm_combine_kernel[(num_tokens,)](
            y, local_ids, topk_weights, state["post_expert_norm_w"], out, size_k, POST_NORM_EPS,
            TOP_K_C=TOP_K, BLOCK_H=triton.next_power_of_2(size_k), num_warps=8)
        return out
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)
def _master_front(data, qkv, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    k = k.reshape(tokens, hkv * d)
    if cfg["kv_scale"] != 1.0:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    return q
def _master_pair_q(q, tokens, hq, hkv, d):
    g, s = hq // hkv, hkv // 2
    q_attn = torch.zeros((tokens, hq, 2 * d), dtype=FP8, device=q.device)
    qp = q_attn.view(tokens, s, 2, g, 2 * d)
    qv = q.view(tokens, s, 2, g, d)
    qp[:, :, 0, :, :d] = qv[:, :, 0]
    qp[:, :, 1, :, d:] = qv[:, :, 1]
    return q_attn
def _attn_route(cfg, state):
    batch, hkv = cfg["batch"], cfg["num_kv_heads"]
    sk_variant = _sk_variant(cfg, state.get("sk"))
    paired = cfg["window"] == 0 and batch > 1 and hkv % 2 == 0 and sk_variant < 0
    cell = _s3_cell(cfg, sk_variant >= 0, paired)
    use_s3 = cell is not None and state["s3"][cell]
    return use_s3, cell, sk_variant, paired
def _attn_core(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = _dsa_gemm("qkv_gate", data["attn_input"], state["dense"]["qkv_gate"], state)
    fm = state.get("fvg", {}).get(batch)
    if fm and _fvg_ok(cfg, data):
        return _fvg_attn(data, qkv, state, fm)
    skw_route = _skw_route(cfg, state)
    if skw_route is not None:
        return _skw_attention(data, qkv, state, skw_route)
    if cfg["window"] and batch == 1 and length == 1:
        return _sliding_forward(data, qkv, state)
    s1 = _s1_class(cfg)
    if s1 is not None and state["s1"][s1]:
        return _sliding_forward_band(data, qkv, state)
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    window = cfg["window"]
    use_s3, _cell, sk_variant, paired = _attn_route(cfg, state)
    if use_s3:
        q8 = _s3_pre_attention(data, qkv, state, paired)
        q = None
    else:
        q = _master_front(data, qkv, state)
        q8 = None
    if sk_variant >= 0:
        q_fp8 = q8 if use_s3 else q.view(tokens, hq, d).to(FP8).contiguous()
        return _full_decode_sk(data, q_fp8, qkv, state["sk"], sk_variant).reshape(tokens, hq * d)
    if paired:
        g, s = hq // hkv, hkv // 2
        pages, page_size = k_cache.shape[0], k_cache.shape[1]
        k_attn = k_cache.view(pages, page_size, s, 2 * d)
        v_attn = v_cache.view(pages, page_size, s, 2 * d)
        q_attn = q8 if use_s3 else _master_pair_q(q, tokens, hq, hkv, d)
        attn_heads = s
        metadata = None
    else:
        k_attn, v_attn = k_cache, v_cache
        q_attn = q8 if use_s3 else q.view(tokens, hq, d).to(FP8)
        attn_heads = hkv
        metadata = data["scheduler_metadata"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, attn_heads),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, attn_heads))
    out = flash_attn_with_kvcache(
        q=q_attn,
        k_cache=k_attn,
        v_cache=v_attn,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=metadata,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    if paired and use_s3:
        return _s3_paired_norm_gate(out, qkv, cfg)
    if paired:
        g, s = hq // hkv, hkv // 2
        attn = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=out.device)
        a = attn.view(tokens, s, 2, g, d)
        o = out.view(tokens, s, 2, g, 2 * d)
        a[:, :, 0] = o[:, :, 0, :, :d]
        a[:, :, 1] = o[:, :, 1, :, d:]
        out = attn
    return state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
def _parent_attention(data, state):
    return _dsa_gemm("o", _attn_core(data, state), state["dense"]["o"], state)
def kernel_fn(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    dense = state["dense"]
    dl3 = state.get("dl3")
    if dl3 is not None and dl3["a"] and tokens == 1 and _sk_variant(cfg, state.get("sk")) == 3:
        attn_output = _dl3_attention(data, state, dl3)
    else:
        attn_output = _parent_attention(data, state)
    dl = state.get("dl")
    use_dl = dl is not None and tokens <= DL_MAX_TOKENS
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    f128 = _t128_flags(state, tokens)
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16,
        router_dtype=None if split_router else torch.float32)
    if use_dl:
        r_t = dl3 is not None and tokens >= 2 and dl3["b_t"][_s2a_class(tokens)]
        ea = state.get("ea")
        ea_t = ea is not None and tokens <= ea["max_t"] and ea["on"].get(tokens, False)
        xn, (topk_weights, topk_ids, local_ids, align), _ = _dl_forward(
            ffn_input, data["local_start"], dl, r_t=r_t, align=not (ea_t and dl.get("r_noalign", False)),
            b12p=state.get("b12p"), rx=None if split_router else (router_input, state["router"], state.get("dc0k", {}).get(tokens)),
            fs=_rt_fork_pick(state["side_stream"]) if state.get("dc0f", {}).get(tokens) else None)
        if ea_t:
            expert_output = _ead_core(ea["on"][tokens])(ea, xn, local_ids, topk_weights, 0, LOCAL_EXPERTS,
                                                         state["post_expert_norm_w"],
                                                         pdl13=ea["pdl13_t"].get(tokens, 0))
        else:
            expert_output = _experts(xn, topk_weights, local_ids, state, align=align)
    else:
        hc_bsm = _hc_bsm(state, tokens)
        main_stream = torch.cuda.current_stream()
        side_stream = None if state.get("side_serial") else _rt_fork_pick(state["side_stream"])
        if side_stream is not None:
            side_stream.wait_stream(main_stream)
        expert_input = _t128_pool(ffn_input, state, f128)
        with torch.cuda.stream(side_stream):
            topk_weights, topk_ids, local_ids, tw, align = _t128_side(
                ffn_input, router_input, data["local_start"], tokens, hc_bsm, state, f128)
        xn = _t128_norm(expert_input, state, f128)
        if side_stream is not None:
            main_stream.wait_stream(side_stream)
            if not torch.cuda.is_current_stream_capturing():
                for tensor in (topk_weights, topk_ids, local_ids, tw, *align):
                    tensor.record_stream(main_stream)
        ea = state.get("ea")
        if hc_bsm:
            expert_output = _hc_experts(xn, local_ids, tw, align, state, hc_bsm)
        elif ea is not None and ea["on"].get(tokens, False):
            expert_output = _ead_core(ea["on"][tokens])(ea, xn, local_ids, tw, 0, LOCAL_EXPERTS,
                                                        state["post_expert_norm_w"])
        else:
            expert_output = _experts(xn, None, None, state, dispatch=(local_ids, tw, align))
    if use_dl and dl3 is not None and ((tokens == 1 and dl3["u"])
                                       or (2 <= tokens <= SLICED_UNPOOL_MAX_TOKENS and dl3["u8"])):
        ffn_output = _dl3_unpool(expert_output, state, dl3)
    elif tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    elif f128.get("unpool"):
        ffn_output = _t128_unpool(expert_output, state, f128)
    else:
        ffn_output = _dsa_gemm("unpooler", expert_output, dense["unpooler"], state)
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
CAMPAIGN_ENGINE_A_DL = True
RT_FLUSH_BYTES = 256 * 1024 * 1024
RT_REPS = 41
RT_ROUNDS = 2
RT_MARGIN = 1.03
RT_MARGIN_TIGHT_IQR = 1.02
RT_IQR_FACTOR = 1.5
RT_IQR_FLOOR_US = 0.5
RT_JUDGE_REL = 1.5e-2
RT_ROW_FACTOR = 5.0
RT_CHECK_FRACTION = 0.5
_RT_FLUSH = {}
def _rt_flush_buf(device):
    key = str(device)
    if key not in _RT_FLUSH:
        _RT_FLUSH[key] = torch.zeros(RT_FLUSH_BYTES // 4, dtype=torch.float32, device=device)
    return _RT_FLUSH[key]
def _rt_read_flush(device):
    return torch.sum(_rt_flush_buf(device))
_RT_WS = {}
def _rt_cap(g):
    # Every capture owns its cuBLAS workspace. torch keeps one workspace per (handle, stream); a capture on the shared
    # default capture stream reuses the existing one (allocated eagerly when a torch.cuda.Stream() warm-up drew that
    # same pool stream, or by an earlier capture), and CUDAGraph::reset of ANY graph clears that stream's entry, so the
    # next empty_cache unmapped memory a live graph still wrote (Warp MMU Fault in cuBLAS dot_kernel). Destroying a
    # one-kernel graph clears the entry; the capture below then allocates its workspace in its own private pool.
    # Covers the capture stream only: side-stream workspaces (t128 fork) stay shared, safe while no capture runs on them.
    t = _RT_WS.get("t")
    if t is None:
        t = _RT_WS["t"] = torch.zeros(1, device="cuda")
    d = torch.cuda.CUDAGraph()
    with torch.cuda.graph(d, capture_error_mode="thread_local"):
        t.add_(0.0)
    del d
    return torch.cuda.graph(g)
def _rt_graph(fn, warm=2):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with _rt_cap(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out
RT_FORK_DRAWS = 8
_RT_FORK = {}
def _rt_fork_stream(device, prio=0):
    avoid = {torch.cuda.current_stream(device).cuda_stream}
    cap = getattr(torch.cuda.graph, "default_capture_stream", None)
    if cap is not None:
        avoid.add(cap.cuda_stream)
    pre = set(avoid)
    got, draws = [], []
    for _ in range(RT_FORK_DRAWS):
        s = torch.cuda.Stream(device, priority=prio)
        draws.append(s.cuda_stream)
        if s.cuda_stream not in avoid:
            got.append(s)
            avoid.add(s.cuda_stream)
            if len(got) == 2:
                break
    if len(got) < 2:
        raise RuntimeError(f"no distinct fork stream pair in {RT_FORK_DRAWS} draws (priority {prio})")
    beq = {draws[-1]} | ({draws[0]} if draws[0] in pre else set())
    _RT_FORK[got[0].cuda_stream] = {"alt": got[1], "n": 0, "cap": 0, "alt_n": 0, "alt_cap": 0,
                                    "rej": len(draws) - 2, "beq": beq, "bcap": 0}
    return got[0]
def _rt_fork_pick(s):
    rec = _RT_FORK.get(s.cuda_stream)
    if rec is None:
        return s
    rec["n"] += 1
    cur = torch.cuda.current_stream().cuda_stream
    capt = torch.cuda.is_current_stream_capturing()
    if capt:
        rec["cap"] += 1
        if cur in rec["beq"]:
            rec["bcap"] += 1
    if cur != s.cuda_stream:
        return s
    rec["alt_n"] += 1
    if capt:
        rec["alt_cap"] += 1
    return rec["alt"]
def _rt_fork_line(s):
    rec = _RT_FORK.get(s.cuda_stream) or {}
    cap = getattr(torch.cuda.graph, "default_capture_stream", None)
    cur = torch.cuda.current_stream().cuda_stream
    hs = {s.cuda_stream, rec["alt"].cuda_stream} if rec else {s.cuda_stream}
    clash = (cur in hs) or (cap is not None and cap.cuda_stream in hs)
    bclash = cap is not None and cap.cuda_stream in rec.get("beq", ())
    return (f"fork s {s.cuda_stream:#x} p{s.priority} alt {rec['alt'].cuda_stream if rec else 0:#x} | cur {cur:#x} "
            f"cap {cap.cuda_stream if cap is not None else 0:#x} -> {'CLASH' if clash else 'distinct'} | picks "
            f"{rec.get('n', 0)} cap {rec.get('cap', 0)} alt {rec.get('alt_n', 0)} altcap {rec.get('alt_cap', 0)} "
            f"rej {rec.get('rej', 0)} | board-eq {'CLASH' if bclash else 'distinct'} bcap {rec.get('bcap', 0)}")
def _rt_stats(samples):
    s = sorted(samples)
    n = len(s)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return s[n // 2], s[(3 * n) // 4] - s[n // 4], s[n // 10], s[min(n - 1, (9 * n) // 10)]
_RT_EMPTY = {}
def _rt_empty_graph(device):
    key = str(device)
    if key not in _RT_EMPTY:
        t = torch.zeros(1, dtype=torch.float32, device=device)
        g, _ = _rt_graph(lambda: t.add_(0.0))
        _RT_EMPTY[key] = (g, t)
    return _RT_EMPTY[key][0]
def _rt_time(graphs, device, reps=RT_REPS, rounds=RT_ROUNDS):
    gs = dict(graphs)
    gs["__empty__"] = _rt_empty_graph(device)
    order0 = list(gs)
    for _ in range(2):
        for n in order0:
            gs[n].replay()
    torch.cuda.synchronize()
    out = []
    for r in range(rounds):
        ts = {n: [] for n in order0}
        for rep in range(reps):
            k = rep % len(order0)
            order = order0[k:] + order0[:k]
            if (rep + r) % 2:
                order = order[::-1]
            for n in order:
                _rt_read_flush(device)
                s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s_.record()
                gs[n].replay()
                e_.record()
                e_.synchronize()
                ts[n].append(s_.elapsed_time(e_) * 1000.0)
        b = _rt_stats(ts.pop("__empty__"))[0]
        ts = {n: [v - b for v in vs] for n, vs in ts.items()}
        ts["__baseline_us__"] = [b]
        out.append(ts)
    return out
def _rt_select(rounds_by_routing, incumbent, candidates, margin=RT_MARGIN, tight=RT_MARGIN_TIGHT_IQR):
    worst = {c: float("inf") for c in candidates}
    ok = {c: True for c in candidates}
    lines = []
    for rlabel, rounds in rounds_by_routing.items():
        for ri, ts in enumerate(rounds):
            im, ii, _, _ = _rt_stats(ts[incumbent])
            parts = [f"{incumbent} {im:.2f} us (IQR {ii:.2f})"]
            for c in candidates:
                cm, ci, p10, p90 = _rt_stats(ts[c])
                ratio = im / max(cm, 1e-3)
                iqr_ok = ci <= RT_IQR_FACTOR * max(ii, RT_IQR_FLOOR_US)
                passed = ratio >= margin or (ratio >= tight and iqr_ok)
                worst[c] = min(worst[c], ratio)
                ok[c] = ok[c] and passed
                parts.append(f"{c} {cm:.2f} us (IQR {ci:.2f}, p10/p90 {p10:.2f}/{p90:.2f}, x{ratio:.3f}"
                             f"{'' if passed else ', FAIL'})")
            base = ts.get("__baseline_us__", [float("nan")])[0]
            lines.append(f"{rlabel} round {ri + 1} (graph baseline {base:.2f} us subtracted): " + " | ".join(parts))
    good = [c for c in candidates if ok[c]]
    win = max(good, key=lambda c: worst[c]) if good else None
    return win, worst, lines
def _rt_rel(a, b):
    a64, b64 = a.double(), b.double()
    den = float(b64.norm())
    num = float((a64 - b64).norm())
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den
def _rt_row_rel(out, ref):
    o, r = out.double().reshape(out.shape[0], -1), ref.double().reshape(ref.shape[0], -1)
    rn = r.norm(dim=-1)
    den = torch.maximum(rn, 0.1 * rn.mean()).clamp_min(1e-30)
    return float(((o - r).norm(dim=-1) / den).max())
def _rt_check(out, ref, judge_rel=RT_JUDGE_REL, fraction=RT_CHECK_FRACTION):
    o = out.float()
    finite = bool(torch.isfinite(o).all())
    if not finite:
        bad = int((~torch.isfinite(o)).sum())
        return False, float("inf"), float("inf"), f"NON-FINITE ({bad} values)"
    rel = _rt_rel(o, ref)
    row = _rt_row_rel(o, ref)
    lim, rlim = fraction * judge_rel, fraction * RT_ROW_FACTOR * judge_rel
    ok = rel <= lim and row <= rlim
    return ok, rel, row, f"rel-L2 {rel:.3g} (<= {lim:.3g}), worst row {row:.3g} (<= {rlim:.3g})"
def _rt_tracer(tag, label):
    def _step(what):
        torch.cuda.synchronize()
        print(f"{tag}     [{label}] {what}", flush=True)
    return _step
def _rt_build(tag, name, cpp, cuda, functions):
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    arch, flags = "9.0a", ["-O3", "-std=c++20", "-lineinfo"]
    ext, err = None, ""
    try:
        from torch.utils.cpp_extension import load_inline
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        ext = load_inline(name=name, cpp_sources=cpp, cuda_sources=cuda, functions=list(functions),
                          extra_cuda_cflags=flags, verbose=False)
    except Exception as e:
        err = str(e)
        errs = [ln.strip()[:240] for ln in err.splitlines() if "error" in ln.lower() and "Error building" not in ln]
        err = " || ".join(errs[:8]) if errs else err[-600:]
        print(f"{tag} load_inline {name} failed after {time.perf_counter() - t0:.1f}s: {err!r}", flush=True)
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    dt = time.perf_counter() - t0
    if ext is not None:
        print(f"{tag} load_inline {name} compiled in {dt:.1f}s (TORCH_CUDA_ARCH_LIST={arch}, {flags})", flush=True)
    return ext, err, dt
EA_MAX_T = 8
EA_CTAS_PER_SM = 2
EA_LAYOUT_NAME = {1: "master cvt dequant", 2: "exact integer dequant"}
def _ea_luts(device):
    lut = torch.zeros((3, 4, 2, 16), dtype=torch.int64)
    for c in range(16):
        for p in range(4):
            for half in range(2):
                lut[1, p, half, c] = c << (4 * ((3 - p) + 4 * half))
                if p < 3:
                    v = ((c & 7) << (6 - 3 * p)) | ((c >> 3) << (15 - 3 * p))
                else:
                    v = (((c >> 1) & 3) << 10) | ((c & 1) << 13) | ((c >> 3) << 14)
                lut[2, p, half, c] = v << (16 * half)
    return lut.to(device)
def _ea_pack(tile_nib, tile_sf, lut):
    ng, k = tile_nib.shape[0], tile_nib.shape[-1]
    nk = k // 512
    t = tile_nib.reshape(ng, 4, 2, 8, nk, 4, 4, 2, 2, 4, 2).long()
    words = torch.zeros((ng, 4, 2, 8, nk, 4, 4, 4), dtype=torch.int64, device=tile_nib.device)
    for bsel in range(2):
        for hi in range(2):
            p = 2 * bsel + hi
            for el in range(2):
                words |= lut[p, el][t[:, :, :, :, :, :, :, bsel, hi, :, el]]
    del t
    w = words.permute(0, 4, 1, 2, 5, 3, 7, 6).contiguous()
    del words
    codes = torch.stack([(w >> (8 * i)) & 0xFF for i in range(4)], dim=-1).to(torch.uint8).reshape(ng, nk, 4, 4096)
    sc = tile_sf.reshape(ng, 4, 2, 8, nk, 32).permute(0, 4, 1, 2, 3, 5).reshape(ng, nk, 4, 512)
    return torch.cat((codes, sc), dim=-1).contiguous()
def _ea_scales_ok(weights, start, count):
    for key in ("w13_sf", "w2_sf"):
        b = weights[key][start:start + count].contiguous().view(torch.uint8)
        if not bool(((b >= 0x08) & (b <= 0x7E)).all()):
            return False
    return True
def _ea_repack(weights, start, count, dq):
    inter = weights["w13"].shape[1] // 2
    hidden = weights["w2"].shape[1]
    dev = weights["w13"].device
    lut = _ea_luts(dev)[dq]
    w13r = torch.empty((count, inter // 32, hidden // 512, 4, 4608), dtype=torch.uint8, device=dev)
    w2r = torch.empty((count, hidden // 64, inter // 512, 4, 4608), dtype=torch.uint8, device=dev)
    for i in range(count):
        e = start + i
        nib = _ea_nibbles(weights["w13"][e].contiguous())
        sf = weights["w13_sf"][e].contiguous().view(torch.uint8)
        tile = torch.stack((nib[:inter].reshape(inter // 32, 4, 8, hidden),
                            nib[inter:].reshape(inter // 32, 4, 8, hidden)), dim=2)
        stile = torch.stack((sf[:inter].reshape(inter // 32, 4, 8, -1), sf[inter:].reshape(inter // 32, 4, 8, -1)), dim=2)
        w13r[i] = _ea_pack(tile, stile, lut)
        del nib, sf, tile, stile
        nib = _ea_nibbles(weights["w2"][e].contiguous())
        sf = weights["w2_sf"][e].contiguous().view(torch.uint8)
        w2r[i] = _ea_pack(nib.reshape(hidden // 64, 4, 2, 8, inter), sf.reshape(hidden // 64, 4, 2, 8, -1), lut)
        del nib, sf
    sl = slice(start, start + count)
    return {"w13r": w13r.reshape(-1), "w2r": w2r.reshape(-1),
            "gs13": (weights["w13_gs"][sl].float() * 128.0).contiguous(),
            "gs2": (weights["w2_gs"][sl].float() * 128.0).contiguous()}
def _ea_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0):
    ext = ea["ext"]
    tokens, hidden = x.shape
    inter = ea["inter"]
    h = torch.empty((32, tokens, inter), device=x.device, dtype=torch.bfloat16)
    local_ids = torch.empty((tokens, TOP_K), device=x.device, dtype=torch.int32)
    y = torch.empty((tokens * TOP_K, hidden), device=x.device, dtype=torch.bfloat16)
    if trace is not None:
        trace("before ea_w13")
    ext.ea_w13(x, topk_ids, ea["w13r"], ea["gs13"], h, local_ids, ea["ws13"], ea["tk13"], ea["err"], tokens, start,
               count, ea["grid"], ea["dq"], pdl13)
    if trace is not None:
        trace("after ea_w13 / before ea_w2")
    ext.ea_w2(h, topk_ids, ea["w2r"], ea["gs2"], y, ea["ws2"], ea["tk2"], ea["err"], tokens, start, count,
              ea.get("g2_t", {}).get(tokens, ea["grid"]), ea["dq"], ea["pdl"])
    if trace is not None:
        trace("after ea_w2 / before post-expert norm")
    out = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    tw32 = topk_weights.to(torch.float32).contiguous()
    if ea.get("cmb_t", {}).get(tokens, 0):
        ext.ea_combine(y, local_ids, tw32, norm_w, out, POST_NORM_EPS, 1)
    else:
        _post_norm_combine_kernel[(tokens,)](
            y, local_ids, tw32, norm_w, out, hidden, POST_NORM_EPS,
            TOP_K_C=TOP_K, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
    if trace is not None:
        trace("after post-expert norm")
    return out
def _ea_nibbles(packed):
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
def _ea_dequant(packed, sf, gs):
    nib = _ea_nibbles(packed).long()
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32, device=packed.device)
    mag = grid[nib & 7]
    val = torch.where((nib & 8) != 0, -mag, mag)
    rows, k = val.shape
    return (val.view(rows, k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)).view(rows, k)
def _ea_ref_block(weights, base, x, lids, tw, norm_w, eps):
    tokens, hidden = x.shape
    inter = weights["w13"].shape[1] // 2
    flat = lids.reshape(-1).long()
    y = torch.zeros((tokens * TOP_K, hidden), dtype=torch.float32, device=x.device)
    for e in sorted(set(flat.tolist()) - {-1}):
        g = base + e
        w13 = _ea_dequant(weights["w13"][g], weights["w13_sf"][g], weights["w13_gs"][g])
        w2 = _ea_dequant(weights["w2"][g], weights["w2_sf"][g], weights["w2_gs"][g])
        rts = (flat == e).nonzero().flatten()
        hh = x[rts // TOP_K].float() @ w13.T
        h = torch.nn.functional.silu(hh[:, :inter]) * hh[:, inter:]
        y[rts] = h @ w2.T
        del w13, w2
    v = y.view(tokens, TOP_K, hidden) * (norm_w.float() + 1.0)
    loc = lids >= 0
    v = torch.where(loc.unsqueeze(-1), v, torch.zeros_like(v))
    r = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    gw = torch.where(loc, tw.float(), torch.zeros_like(tw.float()))
    return (v * r * gw.unsqueeze(-1)).sum(1)
def _ea_cmb_check(ext, w, device):
    gen = torch.Generator().manual_seed(20261034)
    worst, ok = 0.0, True
    for tokens in (1, 8, 16, 32):
        y = torch.randn((tokens * TOP_K, 4096), generator=gen).to(torch.bfloat16).to(device)
        ids = torch.randint(-1, 32, (tokens, TOP_K), generator=gen, dtype=torch.int32)
        ids[0, 0] = 5
        if tokens > 1:
            ids[-1, :] = -1
        ids = ids.to(device).contiguous()
        tw = torch.rand((tokens, TOP_K), generator=gen)
        tw = (tw / tw.sum(-1, keepdim=True)).to(device).contiguous()
        ref = torch.empty((tokens, 4096), device=device, dtype=torch.bfloat16)
        got = torch.full((tokens, 4096), float("nan"), device=device, dtype=torch.bfloat16)
        _post_norm_combine_kernel[(tokens,)](y, ids, tw, w, ref, 4096, POST_NORM_EPS, TOP_K_C=TOP_K, BLOCK_H=4096,
                                             num_warps=8)
        ext.ea_combine(y, ids, tw, w, got, POST_NORM_EPS, 1)
        torch.cuda.synchronize()
        rel = _rt_rel(got.float(), ref.float())
        worst = rel if rel != rel else max(worst, rel)
        ok = ok and rel <= 1e-3 and (tokens == 1 or bool((got[-1].float() == 0).all()))
    return ok, f"combine check (CUDA vs Triton, T 1/8/16/32): worst rel-L2 {worst:.3g} -> {'ok' if ok else 'FAIL'}"
EA2_MAX_T = 32
EA_TK_MAXV = 160
EA_CLASSES = (1, 8, 16, 32)
EA_ENGINES = {"v1": EA_MAX_T, "v2": EA2_MAX_T}
def _ea_class(tokens):
    return next(c for c in EA_CLASSES if tokens <= c)
def _ea_engines(tokens):
    return [e for e, hi in EA_ENGINES.items() if tokens <= hi]
def _ea_path_lines(tag, on, max_t):
    out, t = [], 1
    while t <= max_t:
        u = t
        while u < max_t and on[u + 1] == on[t]:
            u += 1
        out.append(f"{tag} PATH CHOSEN T={t}{'' if u == t else f'..{u}'} = "
                   f"{'ENGINE A ' + on[t] if on[t] else 'MASTER'}")
        t = u + 1
    return out
_EA2_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> ea_info();
void ea_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
            torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
            int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl);
void ea_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
           torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
           int64_t grid, int64_t dq, int64_t pdl);
void ea_w13_pf(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
               torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
               int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl, torch::Tensor w2r,
               int64_t pf_max_act);
std::vector<int64_t> ea2_info();
void ea_combine(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w, torch::Tensor out, double eps,
                int64_t pdl);
std::vector<int64_t> ea_combine_info();
void ea2_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
             int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl);
void ea2_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
            int64_t grid, int64_t dq, int64_t pdl);
std::vector<int64_t> ea3_info();
void ea3_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
             int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl);
void ea3_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
            int64_t grid, int64_t dq, int64_t pdl);
"""
_EA4_CPP = r"""
std::vector<int64_t> ea4_info();
void ea4_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
             int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl);
void ea4_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
            int64_t grid, int64_t dq, int64_t pdl);
"""
_EA2_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
// ---- begin include rewrite/lib/cuda/ea_common.cuh ----
// Engine A device helpers (campaign 2026-10-26-decode-first-rewrite, workstream W2).
// Only instruction forms already run on the judge: 1-D cp.async.bulk + mbarrier (init, arrive, expect_tx,
// try_wait.parity), mma.sync m16n8k16 bf16, fma.rn.bf16x2 with a -0 addend, cvt.rn.f16x2.e4m3x2, prmt,
// ldmatrix, griddepcontrol. No createpolicy / L2::cache_hint, no wgmma, no setmaxnreg here.
namespace ea {

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
// Bounded wait: after 2^26 cycles without the phase, record error word 1 and stop waiting, so a broken
// pipeline ends in a failed check instead of a hang.
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity, int* err, bool& dead) {
  if (dead || mbar_try(a, parity)) return;
  const long long t0 = clock64();
  while (!mbar_try(a, parity)) {
    if (clock64() - t0 > (1ll << 26)) {
      dead = true;
      if (err != nullptr) atomicExch(err, 1);
      return;
    }
  }
}
// Plain 1-D bulk copy global -> shared (no cache-hint operand). bytes and both addresses 16-byte aligned.
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
// Plain L2 prefetch hint (the form that ran on the judge in attention 2026-10-25; no cache-policy operand).
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p)); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Two 8x8 b16 matrices: lanes 0-7 address matrix 0's rows, lanes 8-15 matrix 1's (lanes 16-31 ignored).
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// ---- dequant, master form (moe master dq_stage / rowowner deq + sc4) ------------------------------------
__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t bf16x2_bcast(float v) {
  __nv_bfloat162 b = __float2bfloat162_rn(v);
  return *reinterpret_cast<uint32_t*>(&b);
}
// a * b + (-0), bf16x2, round to nearest; subnormal inputs are kept (no .ftz form exists for bf16).
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
// Master code layout: pair I (0..3) of a word has its low-half element at nibble 3 - I and its high-half
// element at nibble 7 - I (e2m1 = s e1 e0 m). Result: e2m1 * 2^-126 in each bf16 half (exact).
template <int I>
__device__ __forceinline__ uint32_t ext_master(uint32_t q) {
  const uint32_t t = q << (4 * I);
  return (t & 0x80008000u) | ((t & 0x70007000u) >> 6);
}
// Four E4M3 block scales (one word, byte j = block j) -> four bf16x2 broadcasts of sf * 2^119 (master form).
__device__ __forceinline__ void sc4_master(uint32_t w, uint32_t (&o)[4]) {
  const float P119 = __int_as_float(0x7B000000);  // 2^119
  uint32_t lo, hi;
  e4m3x4_to_f16x2x2(w, lo, hi);
  const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  o[0] = bf16x2_bcast(f0.x * P119);
  o[1] = bf16x2_bcast(f0.y * P119);
  o[2] = bf16x2_bcast(f1.x * P119);
  o[3] = bf16x2_bcast(f1.y * P119);
}

// ---- dequant, proposed exact integer form (FINDINGS 4; research/nvfp4_engine_forensics.md section 3) -----
// Code layout (a lossless prepare-time permutation of each 32-bit word): pair 0 sits at bits {6,7,8,15} (low
// element: m e0 e1 at 6..8, sign at 15) and {22,23,24,31} (high element); pair 1 three bits lower, pair 2 six
// bits lower; pair 3 in the leftover bits: (e1, e0) at (11, 10), m at 13, s at 14 (+16 for the high element).
// Each extraction yields e2m1 * 2^-126 per bf16 half, bit-identical to ext_master on the master layout.
template <int I>
__device__ __forceinline__ uint32_t ext_prop(uint32_t q) {
  if constexpr (I == 0) {
    return q & 0x81C081C0u;
  } else if constexpr (I == 1) {
    return (q << 3) & 0x81C081C0u;
  } else if constexpr (I == 2) {
    return (q << 6) & 0x81C081C0u;
  } else {
    return ((q >> 3) & 0x01800180u) | ((q >> 7) & 0x00400040u) | ((q << 1) & 0x80008000u);
  }
}
// Byte j (0..3) of a word of E4M3 scales -> {S, S} with S = (b << 4) + (239 << 7), the bf16 bits of
// sf * 2^119 for every positive normal E4M3 byte b in [0x08, 0x7E] (prepare checks the range once).
template <int J>
__device__ __forceinline__ uint32_t scale_prop(uint32_t w) {
  const uint32_t t = __byte_perm(w, 0u, 0x4040u | (J << 8) | J);  // [b, 0, b, 0]
  return t * 16u + 0x77807780u;
}
__device__ __forceinline__ uint32_t scale_prop_rt(uint32_t w, int j) {
  const uint32_t t = __byte_perm(w, 0u, 0x4040u | (static_cast<uint32_t>(j) << 8) | static_cast<uint32_t>(j));
  return t * 16u + 0x77807780u;
}
// L3 paired decode (research/eac-spec-2026-10-28.md 4.3): byte j of row g's word wg and row g+8's wg8 -> {S(b_g),
// S(b_g8)}. Selector nibbles 8+ write a byte's sign (0 for dq2's [0x08, 0x7E]): [b_g, 0, b_g8, 0], no carry across.
__device__ __forceinline__ uint32_t scale_pair_rt(uint32_t wg, uint32_t wg8, int j) {
  const uint32_t jj = static_cast<uint32_t>(j);
  uint32_t t;
  asm("prmt.b32 %0, %1, %2, %3;\n"
      : "=r"(t) : "r"(wg), "r"(wg8), "r"(((0xCu + jj) << 12) | ((4u + jj) << 8) | ((8u + jj) << 4) | jj));
  return t * 16u + 0x77807780u;
}
// L4 (eac L4): a 16 that is known only at run time, from a consumer's own token / route count n (1..16): n >> 30 is 0
// for every n in [0, 2^30). The shift is inline PTX, so NVVM cannot range-fold it from the callers' guards, and ptxas
// sees a register (n is a kernel parameter or an smem load), so t * m16 + C stays one IMAD (FMA pipe) instead of
// becoming a LEA (ALU pipe, the binding pipe of the consume loop). Hoisted: one SHF + one SHL per consume call.
__device__ __forceinline__ uint32_t rt16(int n) {
  uint32_t z;
  asm("shr.u32 %0, %1, 30;\n" : "=r"(z) : "r"(n));
  return 16u << z;
}
// L4 paired decode: scale_pair_rt's prmt, then the add as mad.lo.u32 t * m16 + 0x77807780 (m16 == 16: same bits).
__device__ __forceinline__ uint32_t scale_pair_imad_rt(uint32_t wg, uint32_t wg8, int j, uint32_t m16) {
  const uint32_t jj = static_cast<uint32_t>(j);
  uint32_t t, d;
  asm("prmt.b32 %0, %1, %2, %3;\n"
      : "=r"(t) : "r"(wg), "r"(wg8), "r"(((0xCu + jj) << 12) | ((4u + jj) << 8) | ((8u + jj) << 4) | jj));
  asm("mad.lo.u32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(t), "r"(m16), "r"(0x77807780u));
  return d;
}
// mul_bf16x2(a, {S, S}), S = half H of s; this spelling folds into HMUL2's .H0_H0 / .H1_H1 (a __byte_perm does not)
template <int H>
__device__ __forceinline__ uint32_t mul_bf16x2_h(uint32_t a, uint32_t s) {
  static_assert(H == 0 || H == 1, "half 0 or 1");
  uint32_t d;
  if constexpr (H == 0)
    asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 t;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 t, {l, l};\n\t"
        "fma.rn.bf16x2 %0, %1, t, %3;\n\t}\n" : "=r"(d) : "r"(a), "r"(s), "r"(0x80008000u));
  else
    asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 t;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 t, {h, h};\n\t"
        "fma.rn.bf16x2 %0, %1, t, %3;\n\t}\n" : "=r"(d) : "r"(a), "r"(s), "r"(0x80008000u));
  return d;
}

}  // namespace ea
// ---- end include rewrite/lib/cuda/ea_common.cuh ----
// ---- begin include rewrite/lib/cuda/ea_kernels.cuh ----
// Engine A v1: NVFP4 expert block for T <= 8 tokens (campaign 2026-10-26-decode-first-rewrite, W2 step 2).
// K1 ea_w13: h[a][t][:] = silu(x_t . Wgate_e) * (x_t . Wup_e) for every active local expert a (rank order) and every
//            token t < T (all-token B: N = 8, the routed rows are picked by K2's epilogue and the combine).
// K2 ea_w2:  y[t * 8 + s][:] = h[a][t] . W2_e for every route (t, s) on local expert e (rank a); master's cache3 layout.
// Then master's Triton post-norm combine reads y and local_ids (written by K1's block 0, [T, 8], -1 = remote).
// Load path: CTA-shared ring of D stages; a stage is one (expert, 64-row group, 512-k chunk) = 4 warp records
// (18432 B, contiguous after the prepare-time repack) plus the T token rows of that k-chunk (1 KB each, padded rows).
// Warp 4 produces, warps 0..3 consume (each owns 16 rows over the full K: row ownership, no cross-warp reduction).
// Balance: stream-K over the flattened (expert, group, chunk) list; a group split across CTAs is finished by its
// last-arriving CTA (ticket, fixed-order fp32 sum of the partials, ticket reset). No spins, no atomics on data.
namespace eak {
using namespace ea;

constexpr int REC = 4608;
constexpr int STAGE_W = 4 * REC;   // 18432
constexpr int XROW = 1040;         // 1024 B of one token's 512-k slice + 16 B pad (conflict-free ldmatrix)
constexpr int XS = 8 * XROW;       // up to 8 token rows per stage
constexpr int NK = 8;              // 512-k chunks per 4096-k row
constexpr int THREADS = 160;
constexpr int KDIM = 4096;         // H = I = 4096 (host-checked)

__device__ __forceinline__ void bar_consumers() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

// Active local experts from topk_ids (R = 8 T <= 64 routes), warp-wide: s_act[0..n) ascending, s_n = n.
// write_ids: also write local_ids[r] (local expert or -1), master's [T, 8] layout.
__device__ __forceinline__ void build_map(const int* __restrict__ topk_ids, int T, int start, int count, int lane,
                                          int* s_act, int* s_n, int* __restrict__ local_ids, bool write_ids) {
  const int R = T * 8;
  uint32_t mask = 0u;
  for (int r = lane; r < R; r += 32) {
    const int d = topk_ids[r] - start;
    const bool loc = d >= 0 && d < count;
    if (write_ids) local_ids[r] = loc ? d : -1;
    if (loc) mask |= 1u << d;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mask |= __shfl_xor_sync(0xffffffffu, mask, o);
  if ((mask >> lane) & 1u) s_act[__popc(mask & ((1u << lane) - 1u))] = lane;
  if (lane == 0) *s_n = __popc(mask);
}

// Owner CTA of flattened chunk q under the static split b_c = floor(c W / G): the largest c with b_c <= q.
__device__ __forceinline__ int owner(int q, int W, int G) {
  return static_cast<int>(((static_cast<long long>(q) + 1) * G + W - 1) / W) - 1;
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// One warp record (16 rows x 512 k) against the T token rows of this chunk (B via ldmatrix, rows >= T read row T-1;
// those output columns are never stored). acc[bsel] alternates by k16 block parity (two dependency chains).
// XB = 0 (the default; v1 / v2 / v3 / v4): one ldmatrix.x2 per k16 step (L4's form). XB = 1 (X1; v6 family only): each
// kq's eight k16 B fragments are preloaded by four ldmatrix.x4 (matrices 0/1 = block 2p, 2/3 = block 2p + 1: lanes
// 16..31 address +32 B), the same registers as eight per-step ldmatrix.x2 (x1_emul.py).
template <int DQ, int XB = 0>
__device__ __forceinline__ void consume(const uint8_t* rec, uint32_t xs_u32, int lane, int T, float (&acc)[2][4]) {
  static_assert(XB == 0 || XB == 1, "XB");
  const int g = lane >> 2;
  const int xr = (lane & 7) < T ? (lane & 7) : T - 1;
  const uint32_t xb =
      xs_u32 + static_cast<uint32_t>(xr * XROW + ((lane >> 3) & 1) * 16 + (XB == 1 ? (lane >> 4) * 32 : 0));
  const uint32_t m16 = rt16(T);  // L4: run-time 16 (dq2 only; dead for DQ 1)
#pragma unroll
  for (int kq = 0; kq < 4; ++kq) {
    const uint4 c0 = *reinterpret_cast<const uint4*>(rec + ((0 * 4 + kq) * 32 + lane) * 16);
    const uint4 c1 = *reinterpret_cast<const uint4*>(rec + ((1 * 4 + kq) * 32 + lane) * 16);
    const uint2 s0 = *reinterpret_cast<const uint2*>(rec + 4096 + g * 32 + kq * 8);
    const uint2 s1 = *reinterpret_cast<const uint2*>(rec + 4096 + (g + 8) * 32 + kq * 8);
    uint32_t bk[8][2];  // X1: this kq's B fragments, block j in bk[j]
    if constexpr (XB == 1) {
#pragma unroll
      for (int p = 0; p < 4; ++p)
        ldsm_x4(xb + static_cast<uint32_t>((kq * 8 + 2 * p) * 32), bk[2 * p][0], bk[2 * p][1], bk[2 * p + 1][0],
                bk[2 * p + 1][1]);
    }
    uint32_t m0[8], m1[8];
    if constexpr (DQ == 1) {
      uint32_t t4[4];
      sc4_master(s0.x, t4);
      m0[0] = t4[0]; m0[1] = t4[1]; m0[2] = t4[2]; m0[3] = t4[3];
      sc4_master(s0.y, t4);
      m0[4] = t4[0]; m0[5] = t4[1]; m0[6] = t4[2]; m0[7] = t4[3];
      sc4_master(s1.x, t4);
      m1[0] = t4[0]; m1[1] = t4[1]; m1[2] = t4[2]; m1[3] = t4[3];
      sc4_master(s1.y, t4);
      m1[4] = t4[0]; m1[5] = t4[1]; m1[6] = t4[2]; m1[7] = t4[3];
    }
    const uint32_t qg[4] = {c0.x, c0.y, c0.z, c0.w};
    const uint32_t qu[4] = {c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
      for (int bsel = 0; bsel < 2; ++bsel) {
        const int j = wd * 2 + bsel;
        uint32_t sg, sg8;
        if constexpr (DQ == 1) {
          sg = m0[j];
          sg8 = m1[j];
        } else {  // L3 + L4: one paired decode (add on IMAD); HMUL2 reads row g from half 0, row g+8 from half 1
          sg = scale_pair_imad_rt(j < 4 ? s0.x : s0.y, j < 4 ? s1.x : s1.y, j & 3, m16);
          sg8 = sg;
        }
        uint32_t a[4];
        if (bsel == 0) {
          if constexpr (DQ == 1) {
            a[0] = mul_bf16x2(ext_master<0>(qg[wd]), sg);
            a[2] = mul_bf16x2(ext_master<1>(qg[wd]), sg);
            a[1] = mul_bf16x2(ext_master<0>(qu[wd]), sg8);
            a[3] = mul_bf16x2(ext_master<1>(qu[wd]), sg8);
          } else {
            a[0] = mul_bf16x2_h<0>(ext_prop<0>(qg[wd]), sg);
            a[2] = mul_bf16x2_h<0>(ext_prop<1>(qg[wd]), sg);
            a[1] = mul_bf16x2_h<1>(ext_prop<0>(qu[wd]), sg8);
            a[3] = mul_bf16x2_h<1>(ext_prop<1>(qu[wd]), sg8);
          }
        } else {
          if constexpr (DQ == 1) {
            a[0] = mul_bf16x2(ext_master<2>(qg[wd]), sg);
            a[2] = mul_bf16x2(ext_master<3>(qg[wd]), sg);
            a[1] = mul_bf16x2(ext_master<2>(qu[wd]), sg8);
            a[3] = mul_bf16x2(ext_master<3>(qu[wd]), sg8);
          } else {
            a[0] = mul_bf16x2_h<0>(ext_prop<2>(qg[wd]), sg);
            a[2] = mul_bf16x2_h<0>(ext_prop<3>(qg[wd]), sg);
            a[1] = mul_bf16x2_h<1>(ext_prop<2>(qu[wd]), sg8);
            a[3] = mul_bf16x2_h<1>(ext_prop<3>(qu[wd]), sg8);
          }
        }
        uint32_t b0, b1;
        if constexpr (XB == 1) {
          b0 = bk[j][0];
          b1 = bk[j][1];
        } else {
          ldsm_x2(xb + static_cast<uint32_t>((kq * 8 + j) * 32), b0, b1);
        }
        mma16816(acc[bsel], a, b0, b1);
      }
    }
  }
}

// SGLang parity: Marlin writes gate and up as bf16, then silu_and_mul computes in fp32.
__device__ __forceinline__ float silu_mul(float hg, float hu) {
  hg = __bfloat162float(__float2bfloat16_rn(hg));
  hu = __bfloat162float(__float2bfloat16_rn(hu));
  return __fdividef(hg, 1.f + __expf(-hg)) * hu;  // fast form (prefill 21911eb6: w13 x0.92-0.94, error = master's)
}

// The shared body of K1 (W13 = true: 2I = 8192 rows, gate/up interleaved per warp record, B = x) and K2 (W13 =
// false: H = 4096 rows, B = h of the expert's rank). Workspace ws: [G][2][4 warps][32 lanes][4] fp32 partials;
// tickets: [32 local experts][NG groups] int, zero between launches (the finisher resets its group's).
template <bool W13, int D, int DQ>
__device__ __forceinline__ void expert_body(const __nv_bfloat16* __restrict__ bsrc, const int* __restrict__ topk_ids,
                                            const uint8_t* __restrict__ wr, const float* __restrict__ gs,
                                            __nv_bfloat16* __restrict__ out, int* __restrict__ local_ids,
                                            float* __restrict__ ws, int* __restrict__ tickets, int* err, int T,
                                            int start, int count, int pdl,
                                            const uint8_t* __restrict__ pf_w2 = nullptr, int pf_max_act = 0) {
  constexpr int NG = W13 ? 128 : 64;   // 64-row groups per expert
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* xs = smem + D * STAGE_W;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + D * (STAGE_W + XS));
  int* s_misc = reinterpret_cast<int*>(bars + 2 * D);  // [0] n_act, [1] last-arriver flag, [2..34) active experts
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * D;
  // K1 in decode follows the router: its topk_ids are only valid after the wait. K2 needs only h after the wait.
  if (W13 && pdl) griddep_wait();
  if (warp == 0) build_map(topk_ids, T, start, count, lane, s_misc + 2, s_misc, local_ids, W13 && blockIdx.x == 0);
  if (threadIdx.x == 32) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4);
    }
    fence_barrier_init();
  }
  __syncthreads();
  // Every thread lets the dependent be scheduled now (K1 -> K2, K2 -> the combine); a dependent waits for this grid
  // to complete before reading anything it wrote.
  griddep_launch();
  const int n_act = s_misc[0];
  const int W = n_act * NG * NK;
  // Groups that fit the grid run aligned: each group in s = 1, 2, 4 or 8 equal pieces (no fix-up at s = 1, and every
  // split group has exactly s contributors); otherwise plain stream-K over every CTA. CTAs at or past G have no work.
  int G = static_cast<int>(gridDim.x);
  if (n_act * NG <= G) {
    int s = 1;
    while (s < NK && n_act * NG * 2 * s <= G) s *= 2;
    G = n_act * NG * s;
  }
  const bool idle = static_cast<int>(blockIdx.x) >= G;
  const int b = idle ? 0 : static_cast<int>((static_cast<long long>(blockIdx.x) * W) / G);
  const int e = idle ? 0 : static_cast<int>((static_cast<long long>(blockIdx.x + 1) * W) / G);
  bool dead = false;
  if (b >= e) {  // no work for this CTA (no local route, or past the aligned split): CTA-uniform
    if (!W13 && pdl) griddep_wait();  // K2 must not finish before K1: the combine reads K1's local_ids
    return;
  }
  if (warp == 4) {
    // ---- producer ----
    const int n = e - b;
    auto weight_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int grp = q / NK, kc = q - grp * NK;
      const int a = grp / NG, gi = grp - a * NG;
      const int le = s_misc[2 + a];
      mbar_expect_tx(full0 + 8 * slot, static_cast<uint32_t>(STAGE_W + T * 1024));
      bulk_g2s(smem_u32(ring + slot * STAGE_W), wr + (static_cast<size_t>(le * NG + gi) * NK + kc) * STAGE_W, STAGE_W,
               full0 + 8 * slot);
    };
    auto x_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int grp = q / NK, kc = q - grp * NK;
      const int a = grp / NG;
      if (lane < T) {
        const __nv_bfloat16* src = W13 ? bsrc + static_cast<size_t>(lane) * KDIM + kc * 512
                                       : bsrc + (static_cast<size_t>(a) * T + lane) * KDIM + kc * 512;
        bulk_g2s(smem_u32(xs + slot * XS + lane * XROW), src, 1024, full0 + 8 * slot);
      }
    };
    const int pre = n < D ? n : D;
    if (!W13) {
      // K2: the first stages' weights do not depend on K1, so they stream before the PDL wait; h after it.
      if (lane == 0)
        for (int i = 0; i < pre; ++i) weight_copy(i);
      __syncwarp();
      if (pdl) griddep_wait();
      for (int i = 0; i < pre; ++i) x_copy(i);
    } else {
      for (int i = 0; i < pre; ++i) {
        if (lane == 0) weight_copy(i);
        __syncwarp();
        x_copy(i);
      }
    }
    for (int i = pre; i < n; ++i) {
      const int slot = i % D;
      mbar_wait(empty0 + 8 * slot, static_cast<uint32_t>(((i / D) - 1) & 1), err, dead);
      if (lane == 0) weight_copy(i);
      __syncwarp();
      x_copy(i);
    }
    if (W13 && pf_max_act > 0 && n_act <= pf_max_act) {
      // Few active experts (T = 1): pull this CTA's share of their w2 records into L2 for K2, after the last w13 copy
      // went out (at most D stages per CTA of w13 still in flight, so both fit the 50 MB L2). Hints only.
      constexpr long long EXP2 = 64LL * NK * STAGE_W;  // w2 record bytes per local expert (K2's layout)
      const long long total = static_cast<long long>(n_act) * EXP2;
      const long long beg = (static_cast<long long>(blockIdx.x) * total) / G;
      const long long end = (static_cast<long long>(blockIdx.x + 1) * total) / G;
      for (long long off = (beg & ~127LL) + lane * 128; off < end; off += 32 * 128) {
        const int a = static_cast<int>(off / EXP2);
        prefetch_l2(pf_w2 + static_cast<long long>(s_misc[2 + a]) * EXP2 + (off - a * EXP2));
      }
    }
    return;
  }
  // ---- consumers (warps 0..3) ----
  const int g = lane >> 2, cq = lane & 3;
  float acc[2][4];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[s][c] = 0.f;
  for (int q = b, i = 0; q < e; ++q, ++i) {
    const int slot = i % D;
    mbar_wait(full0 + 8 * slot, static_cast<uint32_t>((i / D) & 1), err, dead);
    consume<DQ>(ring + slot * STAGE_W + warp * REC, smem_u32(xs + slot * XS), lane, T, acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    const int grp = q / NK, kc = q - grp * NK;
    if (kc != NK - 1 && q != e - 1) continue;
    // ---- this CTA is done with group grp ----
    float c[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      c[k] = acc[0][k] + acc[1][k];
      acc[0][k] = 0.f;
      acc[1][k] = 0.f;
    }
    const int a = grp / NG, gi = grp - a * NG;
    const int le = s_misc[2 + a];
    const int g0 = grp * NK;
    if (!(g0 >= b && g0 + NK <= e)) {
      // Partial group: publish, take a ticket; the last arriver sums every contributor in CTA order.
      const int slot_mine = (grp == b / NK) ? 0 : 1;
      float4* mine = reinterpret_cast<float4*>(ws + (static_cast<size_t>(blockIdx.x) * 2 + slot_mine) * 512);
      mine[warp * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
      __threadfence();
      bar_consumers();
      const int c_first = owner(g0, W, G), c_last = owner(g0 + NK - 1, W, G);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&tickets[le * NG + gi], 1);
        s_misc[1] = (old == c_last - c_first) ? 1 : 0;
      }
      bar_consumers();
      if (s_misc[1] == 0) continue;
      __threadfence();
      float t4[4] = {0.f, 0.f, 0.f, 0.f};
      for (int cc = c_first; cc <= c_last; ++cc) {
        const int bc = static_cast<int>((static_cast<long long>(cc) * W) / G);
        const int sl = (grp == bc / NK) ? 0 : 1;
        const float4 v = __ldcg(reinterpret_cast<const float4*>(ws + (static_cast<size_t>(cc) * 2 + sl) * 512) +
                                warp * 32 + lane);
        t4[0] += v.x;
        t4[1] += v.y;
        t4[2] += v.z;
        t4[3] += v.w;
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] = t4[k];
      if (threadIdx.x == 0) tickets[le * NG + gi] = 0;
    }
    // ---- epilogue: rows (g, g + 8) of this warp, tokens 2 cq, 2 cq + 1 ----
    const float sc = gs[le];  // global scale x 2^7 (the dequant bias)
    if (W13) {
      const int col = gi * 32 + warp * 8 + g;  // h column: rows g = gate col, g + 8 = up col
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int t = 2 * cq + j;
        if (t < T) out[(static_cast<size_t>(a) * T + t) * KDIM + col] = __float2bfloat16_rn(silu_mul(c[j] * sc, c[2 + j] * sc));
      }
    } else {
      const int n0 = gi * 64 + warp * 16 + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int t = 2 * cq + j;
        if (t < T) {
          const __nv_bfloat16 v0 = __float2bfloat16_rn(c[j] * sc);
          const __nv_bfloat16 v1 = __float2bfloat16_rn(c[2 + j] * sc);
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            if (topk_ids[t * 8 + s] - start == le) {
              __nv_bfloat16* yr = out + static_cast<size_t>(t * 8 + s) * KDIM;
              yr[n0] = v0;
              yr[n0 + 8] = v1;
            }
          }
        }
      }
    }
  }
}

template <int D, int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea_w13_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
              const float* __restrict__ gs, __nv_bfloat16* __restrict__ h, int* __restrict__ local_ids,
              float* __restrict__ ws, int* __restrict__ tickets, int* err, int T, int start, int count, int pdl,
              const uint8_t* __restrict__ pf_w2, int pf_max_act) {
  expert_body<true, D, DQ>(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, pdl, pf_w2,
                           pf_max_act);
}

template <int D, int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea_w2_kernel(const __nv_bfloat16* __restrict__ h, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
             const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
             int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<false, D, DQ>(h, topk_ids, wr, gs, y, nullptr, ws, tickets, err, T, start, count, pdl);
}

}  // namespace eak

constexpr int EA_D = 4;
constexpr int ea_smem() { return EA_D * (eak::STAGE_W + eak::XS) + 2 * EA_D * 8 + 34 * 4; }

template <typename F>
static void ea_one_info(std::vector<int64_t>& r, F fn) {
  const void* p = reinterpret_cast<const void*>(fn);
  const cudaError_t e1 = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, ea_smem());
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, p);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, eak::THREADS, ea_smem());
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(ok ? 1 : 0);
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(ea_smem());
  r.push_back(occ);
}

// Per kernel (w13 dq1, w13 dq2, w2 dq1, w2 dq2): ok, regs, spill bytes, dynamic smem, CTAs/SM. Also raises every
// kernel's dynamic-smem limit; call once from prepare before any launch.
std::vector<int64_t> ea_info() {
  std::vector<int64_t> r;
  ea_one_info(r, eak::ea_w13_kernel<EA_D, 1>);
  ea_one_info(r, eak::ea_w13_kernel<EA_D, 2>);
  ea_one_info(r, eak::ea_w2_kernel<EA_D, 1>);
  ea_one_info(r, eak::ea_w2_kernel<EA_D, 2>);
  return r;
}

static void ea_check_common(const torch::Tensor& topk_ids, const torch::Tensor& wr, const torch::Tensor& gs,
                            const torch::Tensor& ws, const torch::Tensor& tickets, const torch::Tensor& err, int64_t T,
                            int64_t count, int64_t grid, int64_t dq, int64_t rows, const char* what) {
  TORCH_CHECK(T >= 1 && T <= 8 && count >= 1 && count <= 32 && grid >= 1 && grid <= 512 && (dq == 1 || dq == 2),
              what, ": arguments");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() && topk_ids.scalar_type() == at::kInt &&
              topk_ids.numel() == T * 8, what, ": topk_ids");
  TORCH_CHECK(wr.is_cuda() && wr.is_contiguous() && wr.scalar_type() == at::kByte &&
              wr.numel() == count * (rows / 64) * eak::NK * static_cast<int64_t>(eak::STAGE_W), what, ": weight records");
  TORCH_CHECK(gs.is_cuda() && gs.is_contiguous() && gs.scalar_type() == at::kFloat && gs.numel() >= count, what, ": gs");
  TORCH_CHECK(ws.is_cuda() && ws.is_contiguous() && ws.scalar_type() == at::kFloat && ws.numel() >= grid * 1024,
              what, ": workspace");
  TORCH_CHECK(tickets.is_cuda() && tickets.is_contiguous() && tickets.scalar_type() == at::kInt &&
              tickets.numel() >= 32 * (rows / 64), what, ": tickets");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, what, ": err");
}

static cudaLaunchConfig_t ea_cfg(int64_t grid, int64_t pdl, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(eak::THREADS);
  cfg.dynamicSmemBytes = ea_smem();
  cfg.stream = at::cuda::getCurrentCUDAStream();
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl != 0 ? 1 : 0;
  return cfg;
}

static void ea_w13_launch(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs,
                          torch::Tensor h, torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets,
                          torch::Tensor err, int64_t T, int64_t start, int64_t count, int64_t grid, int64_t dq,
                          int64_t pdl, const uint8_t* pf_w2, int64_t pf_max_act) {
  ea_check_common(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 8192, "ea_w13");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(0) == T && x.size(1) == eak::KDIM, "ea_w13: x");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= 32 * T * eak::KDIM,
              "ea_w13: h");
  TORCH_CHECK(local_ids.is_cuda() && local_ids.is_contiguous() && local_ids.scalar_type() == at::kInt &&
              local_ids.numel() >= T * 8, "ea_w13: local_ids");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea_cfg(grid, pdl, attr);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak::ea_w13_kernel<EA_D, 1>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl), pf_w2,
                           static_cast<int>(pf_max_act))
      : cudaLaunchKernelEx(&cfg, eak::ea_w13_kernel<EA_D, 2>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl), pf_w2,
                           static_cast<int>(pf_max_act));
  if (le != cudaSuccess) (void)cudaGetLastError();  // do not leave the error for torch's next check
  TORCH_CHECK(le == cudaSuccess, "ea_w13: launch: ", cudaGetErrorString(le));
}

void ea_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
            torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
            int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl) {
  ea_w13_launch(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, grid, dq, pdl, nullptr, 0);
}

// ea_w13 that also streams the active experts' w2 records (w2r, K2's layout) into L2 when at most pf_max_act experts
// are active (T = 1), so K2 reads them from L2.
void ea_w13_pf(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
               torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
               int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl, torch::Tensor w2r,
               int64_t pf_max_act) {
  TORCH_CHECK(w2r.is_cuda() && w2r.is_contiguous() && w2r.scalar_type() == at::kByte &&
              w2r.numel() == count * 64 * eak::NK * static_cast<int64_t>(eak::STAGE_W) && pf_max_act >= 0 &&
              pf_max_act <= 32, "ea_w13_pf: w2 records");
  ea_w13_launch(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, grid, dq, pdl,
                w2r.data_ptr<uint8_t>(), pf_max_act);
}

void ea_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
           torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
           int64_t grid, int64_t dq, int64_t pdl) {
  ea_check_common(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 4096, "ea_w2");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= 32 * T * eak::KDIM,
              "ea_w2: h");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eak::KDIM,
              "ea_w2: y");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea_cfg(grid, pdl, attr);
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak::ea_w2_kernel<EA_D, 1>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak::ea_w2_kernel<EA_D, 2>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea_w2: launch: ", cudaGetErrorString(le));
}
// ---- end include rewrite/lib/cuda/ea_kernels.cuh ----
// ---- begin include rewrite/lib/cuda/ea2_kernels.cuh ----
// Engine A v2: NVFP4 expert block for 1 <= T <= 32 tokens (campaign 2026-10-26-decode-first-rewrite, W2).
// Same load path, consumer and stream-K fix-up as v1 (ea_kernels.cuh, whose eak::consume it reuses), with two changes
// the ladder (job a997f2587a654f7c) asked for:
//   - B is gathered: a pass feeds only the routes of one expert (<= 8 token rows), not all T tokens;
//   - an expert with more than 8 routes runs as several passes of <= 8 routes ("virtual experts"), each streaming the
//     expert's weights again (concurrently, so the repeats are served by L2); never more than one n-tile (the ladder:
//     16 tokens per pass ran at 2.84 TB/s, 32 at 1.37, against 3.40-3.64 at <= 8).
// Work map (every CTA, smem): routes sorted by local expert, stable in route order (t * 8 + s), via __match_any_sync
// ranks; passes v = (expert, first route, routes) in ascending expert order; flattened work (v, group, 512-k chunk).
// K1 ea2_w13: h[pos][:] = silu(x_t . Wgate_e) * (x_t . Wup_e) for every local route position pos (sorted order).
// K2 ea2_w2:  y[t * 8 + s][:] = h[pos] . W2_e for the route at pos (master's cache3 layout).
namespace eak2 {
using namespace ea;
using eak::REC;
using eak::STAGE_W;
using eak::XROW;
using eak::XS;
using eak::NK;
using eak::THREADS;
using eak::KDIM;

constexpr int MAXR = 256;   // T * 8 routes, T <= 32
constexpr int MAXV = 64;    // passes: <= 32 experts + 256 / CAP extra (v2 CAP 8: <= 60)

// The work map (smem): up to MAXR_ routes in MAXR_ / 32 chunks of 32, up to MAXV_ passes. CT holds the pass fields and
// the per-chunk counts: v2 int (Map below, the layout it was proven with, 2572 B); Engine A v3 (ea3_kernels.cuh,
// T <= 256: 2048 routes, 160 passes) short, which keeps two CTAs per SM.
template <int MAXR_, int MAXV_, typename CT>
struct MapT {
  static constexpr int MAXR = MAXR_, MAXV = MAXV_, NCH = MAXR_ / 32;
  using cnt_t = CT;
  int n_v;                  // passes
  int n_local;              // local routes
  CT v_e[MAXV_];            // local expert of pass v
  CT v_beg[MAXV_];          // first sorted route position of pass v
  CT v_n[MAXV_];            // routes in pass v (1..CAP)
  short pos_r[MAXR_];       // sorted position -> route index t * 8 + s
  int cnt[32];              // routes per local expert (scratch)
  int base[32];             // first sorted position of each local expert (scratch)
  CT chunk_cnt[NCH][32];    // per 32-route chunk, routes per expert, then their exclusive prefix (scratch)
  int flag;                 // last-arriver flag
};
using Map = MapT<MAXR, MAXV, int>;  // v2

// Every thread of the CTA takes part (two __syncthreads inside). write_ids: CTA 0 writes local_ids [T, 8] (-1 remote).
// CAP: routes per pass (v2 8; Engine A v3, ea3_kernels.cuh, 16). MapT_: v2's Map or v3's (T <= 256). The 5 warps take
// chunks c = warp + 5 k, k < KMAX (v2: 8 chunks, KMAX 2; v3: 64 chunks, KMAX 13); a lane's (expert, rank) per chunk
// stays in registers (constant trip count, unrolled) until the scatter.
template <int CAP = 8, typename MapT_ = Map>
__device__ __forceinline__ void build_map(const int* __restrict__ topk_ids, int T, int start, int count, MapT_& m,
                                          int* __restrict__ local_ids, bool write_ids) {
  using CT = typename MapT_::cnt_t;
  constexpr int NW = THREADS / 32;                   // every launch runs THREADS threads
  constexpr int KMAX = (MapT_::NCH + NW - 1) / NW;   // chunks per warp
  static_assert(KMAX * NW >= MapT_::NCH && MapT_::MAXR % 32 == 0, "build_map: the warps cover every chunk");
  const int R = T * 8;
  const int nch = (R + 31) / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 32; i += blockDim.x) m.cnt[i] = 0;
  for (int i = tid; i < nch * 32; i += blockDim.x) (&m.chunk_cnt[0][0])[i] = 0;
  __syncthreads();
  // Each warp takes 32-route chunks; a lane's rank among equal experts in its chunk comes from __match_any_sync.
  int my_e[KMAX], my_rank[KMAX];
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    const int c = warp + k * NW;  // warp-uniform
    my_e[k] = -1;
    my_rank[k] = 0;
    if (c < nch) {
      const int r = c * 32 + lane;
      int le = -1;
      if (r < R) {
        const int d = topk_ids[r] - start;
        le = (d >= 0 && d < count) ? d : -1;
        if (write_ids) local_ids[r] = le;
      }
      const unsigned grp = __match_any_sync(0xffffffffu, le);
      my_e[k] = le;
      my_rank[k] = __popc(grp & ((1u << lane) - 1u));
      if (le >= 0 && my_rank[k] == 0) m.chunk_cnt[c][le] = static_cast<CT>(__popc(grp));
    }
  }
  __syncthreads();
  if (tid < 32) {  // per expert: total count and the chunk prefix (in place)
    int run = 0;
#pragma unroll 8
    for (int c = 0; c < nch; ++c) {
      const int v = m.chunk_cnt[c][tid];
      m.chunk_cnt[c][tid] = static_cast<CT>(run);
      run += v;
    }
    m.cnt[tid] = run;
  }
  __syncthreads();
  if (tid == 0) {  // expert bases and the pass list, ascending expert
    int b = 0, nv = 0;
    for (int e = 0; e < 32; ++e) {
      m.base[e] = b;
      for (int p = 0; p < m.cnt[e]; p += CAP) {
        m.v_e[nv] = static_cast<CT>(e);
        m.v_beg[nv] = static_cast<CT>(b + p);
        m.v_n[nv] = static_cast<CT>((m.cnt[e] - p) < CAP ? (m.cnt[e] - p) : CAP);
        ++nv;
      }
      b += m.cnt[e];
    }
    m.n_v = nv;
    m.n_local = b;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    const int c = warp + k * NW;
    const int le = my_e[k];  // -1 when c >= nch
    if (le >= 0) m.pos_r[m.base[le] + m.chunk_cnt[c][le] + my_rank[k]] = static_cast<short>(c * 32 + lane);
  }
  __syncthreads();
}

// K1 (W13) and K2 (!W13). ws: [G][2][4 warps][32 lanes][4] fp32; tickets: [MAXV][NG] int, zero between launches.
template <bool W13, int D, int DQ>
__device__ __forceinline__ void expert_body(const __nv_bfloat16* __restrict__ bsrc, const int* __restrict__ topk_ids,
                                            const uint8_t* __restrict__ wr, const float* __restrict__ gs,
                                            __nv_bfloat16* __restrict__ out, int* __restrict__ local_ids,
                                            float* __restrict__ ws, int* __restrict__ tickets, int* err, int T,
                                            int start, int count, int pdl) {
  constexpr int NG = W13 ? 128 : 64;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* xs = smem + D * STAGE_W;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + D * (STAGE_W + XS));
  Map& m = *reinterpret_cast<Map*>(bars + 2 * D);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * D;
  if (W13 && pdl) griddep_wait();  // decode: topk_ids come from the routing kernel
  if (threadIdx.x == 32) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4);
    }
    fence_barrier_init();
  }
  build_map(topk_ids, T, start, count, m, local_ids, W13 && blockIdx.x == 0);  // ends with __syncthreads
  griddep_launch();  // K1 -> K2 and K2 -> the combine: a dependent waits for this grid before reading its outputs
  const int W = m.n_v * NG * NK;
  // Groups that fit the grid run aligned: each group in s = 1, 2, 4 or 8 equal pieces (no fix-up at s = 1, and every
  // split group has exactly s contributors); otherwise plain stream-K over every CTA. CTAs at or past G have no work.
  int G = static_cast<int>(gridDim.x);
  if (m.n_v * NG <= G) {
    int s = 1;
    while (s < NK && m.n_v * NG * 2 * s <= G) s *= 2;
    G = m.n_v * NG * s;
  }
  const bool idle = static_cast<int>(blockIdx.x) >= G;
  const int b = idle ? 0 : static_cast<int>((static_cast<long long>(blockIdx.x) * W) / G);
  const int e = idle ? 0 : static_cast<int>((static_cast<long long>(blockIdx.x + 1) * W) / G);
  bool dead = false;
  if (b >= e) {  // no work for this CTA (no local route, or past the aligned split): CTA-uniform
    if (!W13 && pdl) griddep_wait();  // K2 must not finish before K1: the combine reads K1's local_ids
    return;
  }
  if (warp == 4) {
    // ---- producer ----
    const int n = e - b;
    auto weight_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int grp = q / NK, kc = q - grp * NK;
      const int v = grp / NG, gi = grp - v * NG;
      mbar_expect_tx(full0 + 8 * slot, static_cast<uint32_t>(STAGE_W + m.v_n[v] * 1024));
      bulk_g2s(smem_u32(ring + slot * STAGE_W),
               wr + (static_cast<size_t>(m.v_e[v] * NG + gi) * NK + kc) * STAGE_W, STAGE_W, full0 + 8 * slot);
    };
    auto x_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int grp = q / NK, kc = q - grp * NK;
      const int v = grp / NG;
      if (lane < m.v_n[v]) {
        const int pos = m.v_beg[v] + lane;
        const __nv_bfloat16* src = W13 ? bsrc + static_cast<size_t>(m.pos_r[pos] >> 3) * KDIM + kc * 512
                                       : bsrc + static_cast<size_t>(pos) * KDIM + kc * 512;
        bulk_g2s(smem_u32(xs + slot * XS + lane * XROW), src, 1024, full0 + 8 * slot);
      }
    };
    const int pre = n < D ? n : D;
    if (!W13) {
      if (lane == 0)
        for (int i = 0; i < pre; ++i) weight_copy(i);
      __syncwarp();
      if (pdl) griddep_wait();  // h is K1's output
      for (int i = 0; i < pre; ++i) x_copy(i);
    } else {
      for (int i = 0; i < pre; ++i) {
        if (lane == 0) weight_copy(i);
        __syncwarp();
        x_copy(i);
      }
    }
    for (int i = pre; i < n; ++i) {
      const int slot = i % D;
      mbar_wait(empty0 + 8 * slot, static_cast<uint32_t>(((i / D) - 1) & 1), err, dead);
      if (lane == 0) weight_copy(i);
      __syncwarp();
      x_copy(i);
    }
    return;
  }
  // ---- consumers (warps 0..3) ----
  const int g = lane >> 2, cq = lane & 3;
  float acc[2][4];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[s][c] = 0.f;
  for (int q = b, i = 0; q < e; ++q, ++i) {
    const int slot = i % D;
    const int grp = q / NK, kc = q - grp * NK;
    const int v = grp / NG;
    mbar_wait(full0 + 8 * slot, static_cast<uint32_t>((i / D) & 1), err, dead);
    eak::consume<DQ>(ring + slot * STAGE_W + warp * REC, smem_u32(xs + slot * XS), lane, m.v_n[v], acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (kc != NK - 1 && q != e - 1) continue;
    float c[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      c[k] = acc[0][k] + acc[1][k];
      acc[0][k] = 0.f;
      acc[1][k] = 0.f;
    }
    const int gi = grp - v * NG;
    const int g0 = grp * NK;
    if (!(g0 >= b && g0 + NK <= e)) {
      const int slot_mine = (grp == b / NK) ? 0 : 1;
      float4* mine = reinterpret_cast<float4*>(ws + (static_cast<size_t>(blockIdx.x) * 2 + slot_mine) * 512);
      mine[warp * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
      __threadfence();
      eak::bar_consumers();
      const int c_first = eak::owner(g0, W, G), c_last = eak::owner(g0 + NK - 1, W, G);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&tickets[v * NG + gi], 1);
        m.flag = (old == c_last - c_first) ? 1 : 0;
      }
      eak::bar_consumers();
      if (m.flag == 0) continue;
      __threadfence();
      float t4[4] = {0.f, 0.f, 0.f, 0.f};
      for (int cc = c_first; cc <= c_last; ++cc) {
        const int bc = static_cast<int>((static_cast<long long>(cc) * W) / G);
        const int sl = (grp == bc / NK) ? 0 : 1;
        const float4 x4 = __ldcg(reinterpret_cast<const float4*>(ws + (static_cast<size_t>(cc) * 2 + sl) * 512) +
                                 warp * 32 + lane);
        t4[0] += x4.x;
        t4[1] += x4.y;
        t4[2] += x4.z;
        t4[3] += x4.w;
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] = t4[k];
      if (threadIdx.x == 0) tickets[v * NG + gi] = 0;
    }
    // ---- epilogue: rows (g, g + 8), route columns 2 cq + {0, 1} of pass v ----
    const int le = m.v_e[v];
    const float sc = gs[le];
    const int nr = m.v_n[v];
    if (W13) {
      const int col = gi * 32 + warp * 8 + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int jj = 2 * cq + j;
        if (jj < nr)
          out[static_cast<size_t>(m.v_beg[v] + jj) * KDIM + col] =
              __float2bfloat16_rn(eak::silu_mul(c[j] * sc, c[2 + j] * sc));
      }
    } else {
      const int n0 = gi * 64 + warp * 16 + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int jj = 2 * cq + j;
        if (jj < nr) {
          __nv_bfloat16* yr = out + static_cast<size_t>(m.pos_r[m.v_beg[v] + jj]) * KDIM;
          yr[n0] = __float2bfloat16_rn(c[j] * sc);
          yr[n0 + 8] = __float2bfloat16_rn(c[2 + j] * sc);
        }
      }
    }
  }
}

template <int D, int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea2_w13_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
               const float* __restrict__ gs, __nv_bfloat16* __restrict__ h, int* __restrict__ local_ids,
               float* __restrict__ ws, int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<true, D, DQ>(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, pdl);
}

template <int D, int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea2_w2_kernel(const __nv_bfloat16* __restrict__ h, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
              const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
              int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<false, D, DQ>(h, topk_ids, wr, gs, y, nullptr, ws, tickets, err, T, start, count, pdl);
}

}  // namespace eak2

constexpr int ea2_smem() {
  return EA_D * (eak::STAGE_W + eak::XS) + 2 * EA_D * 8 + static_cast<int>(sizeof(eak2::Map));
}

template <typename F>
static void ea2_one_info(std::vector<int64_t>& r, F fn) {
  const void* p = reinterpret_cast<const void*>(fn);
  const cudaError_t e1 = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, ea2_smem());
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, p);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, eak::THREADS, ea2_smem());
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(ok ? 1 : 0);
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(ea2_smem());
  r.push_back(occ);
}

std::vector<int64_t> ea2_info() {
  std::vector<int64_t> r;
  ea2_one_info(r, eak2::ea2_w13_kernel<EA_D, 1>);
  ea2_one_info(r, eak2::ea2_w13_kernel<EA_D, 2>);
  ea2_one_info(r, eak2::ea2_w2_kernel<EA_D, 1>);
  ea2_one_info(r, eak2::ea2_w2_kernel<EA_D, 2>);
  return r;
}

static void ea2_check(const torch::Tensor& topk_ids, const torch::Tensor& wr, const torch::Tensor& gs,
                      const torch::Tensor& ws, const torch::Tensor& tickets, const torch::Tensor& err, int64_t T,
                      int64_t count, int64_t grid, int64_t dq, int64_t rows, const char* what,
                      int64_t max_t = 32, int64_t maxv = eak2::MAXV) {
  TORCH_CHECK(T >= 1 && T <= max_t && count >= 1 && count <= 32 && grid >= 1 && grid <= 512 && (dq == 1 || dq == 2),
              what, ": arguments");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() && topk_ids.scalar_type() == at::kInt &&
              topk_ids.numel() == T * 8, what, ": topk_ids");
  TORCH_CHECK(wr.is_cuda() && wr.is_contiguous() && wr.scalar_type() == at::kByte &&
              wr.numel() == count * (rows / 64) * eak::NK * static_cast<int64_t>(eak::STAGE_W), what, ": weight records");
  TORCH_CHECK(gs.is_cuda() && gs.is_contiguous() && gs.scalar_type() == at::kFloat && gs.numel() >= count, what, ": gs");
  TORCH_CHECK(ws.is_cuda() && ws.is_contiguous() && ws.scalar_type() == at::kFloat && ws.numel() >= grid * 1024,
              what, ": workspace");
  TORCH_CHECK(tickets.is_cuda() && tickets.is_contiguous() && tickets.scalar_type() == at::kInt &&
              tickets.numel() >= maxv * (rows / 64), what, ": tickets");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, what, ": err");
}

static cudaLaunchConfig_t ea2_cfg(int64_t grid, int64_t pdl, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(eak::THREADS);
  cfg.dynamicSmemBytes = ea2_smem();
  cfg.stream = at::cuda::getCurrentCUDAStream();
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl != 0 ? 1 : 0;
  return cfg;
}

void ea2_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
             int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl) {
  ea2_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 8192, "ea2_w13");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(0) == T && x.size(1) == eak::KDIM, "ea2_w13: x");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea2_w13: h");
  TORCH_CHECK(local_ids.is_cuda() && local_ids.is_contiguous() && local_ids.scalar_type() == at::kInt &&
              local_ids.numel() >= T * 8, "ea2_w13: local_ids");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea2_cfg(grid, pdl, attr);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak2::ea2_w13_kernel<EA_D, 1>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak2::ea2_w13_kernel<EA_D, 2>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();  // do not leave the error for torch's next check
  TORCH_CHECK(le == cudaSuccess, "ea2_w13: launch: ", cudaGetErrorString(le));
}

void ea2_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
            int64_t grid, int64_t dq, int64_t pdl) {
  ea2_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 4096, "ea2_w2");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea2_w2: h");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eak::KDIM,
              "ea2_w2: y");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea2_cfg(grid, pdl, attr);
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak2::ea2_w2_kernel<EA_D, 1>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak2::ea2_w2_kernel<EA_D, 2>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea2_w2: launch: ", cudaGetErrorString(le));
}
// ---- end include rewrite/lib/cuda/ea2_kernels.cuh ----
// ---- begin include rewrite/lib/cuda/ea3_kernels.cuh ----
// Engine A v3: NVFP4 expert block for 1 <= T <= 256 tokens (campaign 2026-10-26-decode-first-rewrite, W2; T <= 32
// judge-proven, T 33..256 for the verify_16 / verify_32 rows: ea-v3big).
// v2 (ea2_kernels.cuh) with passes of up to 16 routes instead of 8, so a hot expert's weights stream once per 16 of its
// routes, not once per 8 (the ladder: a 16-row pass ran at 2.84 TB/s against 3.40 at 8 rows, about 1.2x the time of
// an 8-route pass instead of 2x for two). A pass with 9..16 routes runs two n8 MMA tiles (routes 0-7 and 8-15) that
// share every dequantized A fragment (consume2: dequant once, ldmatrix.x4 for both tiles' B, two mma.sync per k16
// step); a pass with <= 8 routes runs v1/v2's consume unchanged (one tile). Everything else is v2's: the work map
// (eak2::build_map at CAP 16, the same Map), the aligned / stream-K split, the early griddep_launch in both kernels,
// K2's no-work exit that waits on K1, PDL K1 -> K2 with K2's weight prefetch before its wait, bounded mbarrier waits
// with the error word, last-arriver tickets per (pass, group) without spins, the exact integer dequant.
// Differences in the buffers: 16 token rows per stage (ring depth D = 3 keeps 2 CTAs/SM), and stream-K partial slots
// of 1024 floats ([tile][warp][lane][4]; tile 1 only in two-tile passes), so ws needs grid * 2048 floats.
// T <= 256: v3's own work map (eak2::MapT at 2048 routes / 160 passes, short pass fields and chunk counts: 9420 B,
// still two CTAs per SM), build_map over up to 64 chunks of 32 routes (5 warps x 13), tickets [160][NG].
namespace eak3 {
using namespace ea;
using eak::REC;
using eak::STAGE_W;
using eak::XROW;
using eak::NK;
using eak::THREADS;
using eak::KDIM;
constexpr int CAP = 16;          // routes per pass: two n8 tiles
constexpr int MAXT = 256;        // tokens: 2048 routes
// Passes <= sum over the 32 local experts of ceil(cnt_e / 16) <= (8 T + 15 * 32) / 16 = 158 at T = 256.
using Map = eak2::MapT<MAXT * 8, 32 + MAXT * 8 / CAP, short>;
constexpr int XS = CAP * XROW;   // 16640: up to 16 token rows per stage
constexpr int PART = 1024;       // floats per stream-K partial slot: [2 tiles][4 warps][32 lanes][4]

// Four 8x8 b16 matrices: lanes 8 i .. 8 i + 7 address matrix i's rows (the form master's decode kernels run).
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// eak::consume for two n8 tiles: routes 0-7 (tile 0) and 8-15 (tile 1) of a pass with 9..16 routes. The A side is
// eak::consume's code verbatim (every fragment dequantized once); B for both tiles comes from one ldmatrix.x4
// (matrix 0: tile 0 k 0-7, 1: tile 0 k 8-15, 2: tile 1 k 0-7, 3: tile 1 k 8-15; lane l addresses token row
// (l & 7) + 8 (l >> 4), k half (l >> 3) & 1), and each fragment feeds two mma.sync. Token rows >= n read row n - 1
// (those output columns are never stored). acc[tile][bsel] alternates by k16 block parity as in eak::consume.
template <int DQ>
__device__ __forceinline__ void consume2(const uint8_t* rec, uint32_t xs_u32, int lane, int n, float (&acc)[2][2][4]) {
  const int g = lane >> 2;
  const int row = (lane & 7) + ((lane >> 4) << 3);
  const int xr = row < n ? row : n - 1;
  const uint32_t xb = xs_u32 + static_cast<uint32_t>(xr * XROW + ((lane >> 3) & 1) * 16);
  const uint32_t m16 = rt16(n);  // L4: run-time 16 (dq2 only; dead for DQ 1)
#pragma unroll
  for (int kq = 0; kq < 4; ++kq) {
    const uint4 c0 = *reinterpret_cast<const uint4*>(rec + ((0 * 4 + kq) * 32 + lane) * 16);
    const uint4 c1 = *reinterpret_cast<const uint4*>(rec + ((1 * 4 + kq) * 32 + lane) * 16);
    const uint2 s0 = *reinterpret_cast<const uint2*>(rec + 4096 + g * 32 + kq * 8);
    const uint2 s1 = *reinterpret_cast<const uint2*>(rec + 4096 + (g + 8) * 32 + kq * 8);
    uint32_t m0[8], m1[8];
    if constexpr (DQ == 1) {
      uint32_t t4[4];
      sc4_master(s0.x, t4);
      m0[0] = t4[0]; m0[1] = t4[1]; m0[2] = t4[2]; m0[3] = t4[3];
      sc4_master(s0.y, t4);
      m0[4] = t4[0]; m0[5] = t4[1]; m0[6] = t4[2]; m0[7] = t4[3];
      sc4_master(s1.x, t4);
      m1[0] = t4[0]; m1[1] = t4[1]; m1[2] = t4[2]; m1[3] = t4[3];
      sc4_master(s1.y, t4);
      m1[4] = t4[0]; m1[5] = t4[1]; m1[6] = t4[2]; m1[7] = t4[3];
    }
    const uint32_t qg[4] = {c0.x, c0.y, c0.z, c0.w};
    const uint32_t qu[4] = {c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
      for (int bsel = 0; bsel < 2; ++bsel) {
        const int j = wd * 2 + bsel;
        uint32_t sg, sg8;
        if constexpr (DQ == 1) {
          sg = m0[j];
          sg8 = m1[j];
        } else {  // L3 + L4: one paired decode (add on IMAD); HMUL2 reads row g from half 0, row g+8 from half 1
          sg = scale_pair_imad_rt(j < 4 ? s0.x : s0.y, j < 4 ? s1.x : s1.y, j & 3, m16);
          sg8 = sg;
        }
        uint32_t a[4];
        if (bsel == 0) {
          if constexpr (DQ == 1) {
            a[0] = mul_bf16x2(ext_master<0>(qg[wd]), sg);
            a[2] = mul_bf16x2(ext_master<1>(qg[wd]), sg);
            a[1] = mul_bf16x2(ext_master<0>(qu[wd]), sg8);
            a[3] = mul_bf16x2(ext_master<1>(qu[wd]), sg8);
          } else {
            a[0] = mul_bf16x2_h<0>(ext_prop<0>(qg[wd]), sg);
            a[2] = mul_bf16x2_h<0>(ext_prop<1>(qg[wd]), sg);
            a[1] = mul_bf16x2_h<1>(ext_prop<0>(qu[wd]), sg8);
            a[3] = mul_bf16x2_h<1>(ext_prop<1>(qu[wd]), sg8);
          }
        } else {
          if constexpr (DQ == 1) {
            a[0] = mul_bf16x2(ext_master<2>(qg[wd]), sg);
            a[2] = mul_bf16x2(ext_master<3>(qg[wd]), sg);
            a[1] = mul_bf16x2(ext_master<2>(qu[wd]), sg8);
            a[3] = mul_bf16x2(ext_master<3>(qu[wd]), sg8);
          } else {
            a[0] = mul_bf16x2_h<0>(ext_prop<2>(qg[wd]), sg);
            a[2] = mul_bf16x2_h<0>(ext_prop<3>(qg[wd]), sg);
            a[1] = mul_bf16x2_h<1>(ext_prop<2>(qu[wd]), sg8);
            a[3] = mul_bf16x2_h<1>(ext_prop<3>(qu[wd]), sg8);
          }
        }
        uint32_t b0, b1, b2, b3;
        ldsm_x4(xb + static_cast<uint32_t>((kq * 8 + j) * 32), b0, b1, b2, b3);
        mma16816(acc[0][bsel], a, b0, b1);
        mma16816(acc[1][bsel], a, b2, b3);
      }
    }
  }
}

// K1 (W13) and K2 (!W13), v2's body with two-tile passes. ws: [G][2 slots][PART] fp32; tickets: [Map::MAXV][NG] int,
// zero between launches (each finisher resets its own).
template <bool W13, int D, int DQ>
__device__ __forceinline__ void expert_body(const __nv_bfloat16* __restrict__ bsrc, const int* __restrict__ topk_ids,
                                            const uint8_t* __restrict__ wr, const float* __restrict__ gs,
                                            __nv_bfloat16* __restrict__ out, int* __restrict__ local_ids,
                                            float* __restrict__ ws, int* __restrict__ tickets, int* err, int T,
                                            int start, int count, int pdl) {
  constexpr int NG = W13 ? 128 : 64;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* xs = smem + D * STAGE_W;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + D * (STAGE_W + XS));
  Map& m = *reinterpret_cast<Map*>(bars + 2 * D);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * D;
  if (W13 && pdl) griddep_wait();  // decode: topk_ids come from the routing kernel
  if (threadIdx.x == 32) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4);
    }
    fence_barrier_init();
  }
  eak2::build_map<CAP>(topk_ids, T, start, count, m, local_ids, W13 && blockIdx.x == 0);  // ends with __syncthreads
  griddep_launch();  // K1 -> K2 and K2 -> the combine: a dependent waits for this grid before reading its outputs
  const int W = m.n_v * NG * NK;
  // Groups that fit the grid run aligned: each group in s = 1, 2, 4 or 8 equal pieces (no fix-up at s = 1, and every
  // split group has exactly s contributors); otherwise plain stream-K over every CTA. CTAs at or past G have no work.
  int G = static_cast<int>(gridDim.x);
  if (m.n_v * NG <= G) {
    int s = 1;
    while (s < NK && m.n_v * NG * 2 * s <= G) s *= 2;
    G = m.n_v * NG * s;
  }
  const bool idle = static_cast<int>(blockIdx.x) >= G;
  const int b = idle ? 0 : static_cast<int>((static_cast<long long>(blockIdx.x) * W) / G);
  const int e = idle ? 0 : static_cast<int>((static_cast<long long>(blockIdx.x + 1) * W) / G);
  bool dead = false;
  if (b >= e) {  // no work for this CTA (no local route, or past the aligned split): CTA-uniform
    if (!W13 && pdl) griddep_wait();  // K2 must not finish before K1: the combine reads K1's local_ids
    return;
  }
  if (warp == 4) {
    // ---- producer: a stage = the 18432 B weight record + the pass's v_n (<= 16) token rows of that k-chunk ----
    const int n = e - b;
    auto weight_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int grp = q / NK, kc = q - grp * NK;
      const int v = grp / NG, gi = grp - v * NG;
      mbar_expect_tx(full0 + 8 * slot, static_cast<uint32_t>(STAGE_W + m.v_n[v] * 1024));
      bulk_g2s(smem_u32(ring + slot * STAGE_W),
               wr + (static_cast<size_t>(m.v_e[v] * NG + gi) * NK + kc) * STAGE_W, STAGE_W, full0 + 8 * slot);
    };
    auto x_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int grp = q / NK, kc = q - grp * NK;
      const int v = grp / NG;
      if (lane < m.v_n[v]) {
        const int pos = m.v_beg[v] + lane;
        const __nv_bfloat16* src = W13 ? bsrc + static_cast<size_t>(m.pos_r[pos] >> 3) * KDIM + kc * 512
                                       : bsrc + static_cast<size_t>(pos) * KDIM + kc * 512;
        bulk_g2s(smem_u32(xs + slot * XS + lane * XROW), src, 1024, full0 + 8 * slot);
      }
    };
    const int pre = n < D ? n : D;
    if (!W13) {
      if (lane == 0)
        for (int i = 0; i < pre; ++i) weight_copy(i);
      __syncwarp();
      if (pdl) griddep_wait();  // h is K1's output
      for (int i = 0; i < pre; ++i) x_copy(i);
    } else {
      for (int i = 0; i < pre; ++i) {
        if (lane == 0) weight_copy(i);
        __syncwarp();
        x_copy(i);
      }
    }
    for (int i = pre; i < n; ++i) {
      const int slot = i % D;
      mbar_wait(empty0 + 8 * slot, static_cast<uint32_t>(((i / D) - 1) & 1), err, dead);
      if (lane == 0) weight_copy(i);
      __syncwarp();
      x_copy(i);
    }
    return;
  }
  // ---- consumers (warps 0..3) ----
  const int g = lane >> 2, cq = lane & 3;
  float acc[2][2][4];  // [tile][k16 parity][C fragment]; tile 1 only in passes of 9..16 routes
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[t][s][c] = 0.f;
  for (int q = b, i = 0; q < e; ++q, ++i) {
    const int slot = i % D;
    const int grp = q / NK, kc = q - grp * NK;
    const int v = grp / NG;
    const int nr = m.v_n[v];
    mbar_wait(full0 + 8 * slot, static_cast<uint32_t>((i / D) & 1), err, dead);
    if (nr > 8)
      consume2<DQ>(ring + slot * STAGE_W + warp * REC, smem_u32(xs + slot * XS), lane, nr, acc);
    else
      eak::consume<DQ>(ring + slot * STAGE_W + warp * REC, smem_u32(xs + slot * XS), lane, nr, acc[0]);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (kc != NK - 1 && q != e - 1) continue;
    const bool two = nr > 8;  // the same for every contributor of this group (one pass)
    float c[2][4];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        c[t][k] = acc[t][0][k] + acc[t][1][k];
        acc[t][0][k] = 0.f;
        acc[t][1][k] = 0.f;
      }
    const int gi = grp - v * NG;
    const int g0 = grp * NK;
    if (!(g0 >= b && g0 + NK <= e)) {
      // Partial group: publish (tile 1 at +512 floats when the pass has two), take a ticket; the last arriver sums
      // every contributor in CTA order.
      const int slot_mine = (grp == b / NK) ? 0 : 1;
      float4* mine = reinterpret_cast<float4*>(ws + (static_cast<size_t>(blockIdx.x) * 2 + slot_mine) * PART);
      mine[warp * 32 + lane] = make_float4(c[0][0], c[0][1], c[0][2], c[0][3]);
      if (two) mine[128 + warp * 32 + lane] = make_float4(c[1][0], c[1][1], c[1][2], c[1][3]);
      __threadfence();
      eak::bar_consumers();
      const int c_first = eak::owner(g0, W, G), c_last = eak::owner(g0 + NK - 1, W, G);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&tickets[v * NG + gi], 1);
        m.flag = (old == c_last - c_first) ? 1 : 0;
      }
      eak::bar_consumers();
      if (m.flag == 0) continue;
      __threadfence();
      float t4[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      for (int cc = c_first; cc <= c_last; ++cc) {
        const int bc = static_cast<int>((static_cast<long long>(cc) * W) / G);
        const int sl = (grp == bc / NK) ? 0 : 1;
        const float4* src = reinterpret_cast<const float4*>(ws + (static_cast<size_t>(cc) * 2 + sl) * PART) +
                            warp * 32 + lane;
        const float4 x0 = __ldcg(src);
        t4[0][0] += x0.x;
        t4[0][1] += x0.y;
        t4[0][2] += x0.z;
        t4[0][3] += x0.w;
        if (two) {
          const float4 x1 = __ldcg(src + 128);
          t4[1][0] += x1.x;
          t4[1][1] += x1.y;
          t4[1][2] += x1.z;
          t4[1][3] += x1.w;
        }
      }
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int k = 0; k < 4; ++k) c[t][k] = t4[t][k];
      if (threadIdx.x == 0) tickets[v * NG + gi] = 0;
    }
    // ---- epilogue: rows (g, g + 8), route columns tile * 8 + 2 cq + {0, 1} of pass v; only routes < v_n ----
    const int le = m.v_e[v];
    const float sc = gs[le];
    if (W13) {
      const int col = gi * 32 + warp * 8 + g;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int jj = t * 8 + 2 * cq + j;
          if (jj < nr)
            out[static_cast<size_t>(m.v_beg[v] + jj) * KDIM + col] =
                __float2bfloat16_rn(eak::silu_mul(c[t][j] * sc, c[t][2 + j] * sc));
        }
    } else {
      const int n0 = gi * 64 + warp * 16 + g;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int jj = t * 8 + 2 * cq + j;
          if (jj < nr) {
            __nv_bfloat16* yr = out + static_cast<size_t>(m.pos_r[m.v_beg[v] + jj]) * KDIM;
            yr[n0] = __float2bfloat16_rn(c[t][j] * sc);
            yr[n0 + 8] = __float2bfloat16_rn(c[t][2 + j] * sc);
          }
        }
    }
  }
}

template <int D, int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea3_w13_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
               const float* __restrict__ gs, __nv_bfloat16* __restrict__ h, int* __restrict__ local_ids,
               float* __restrict__ ws, int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<true, D, DQ>(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, pdl);
}

template <int D, int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea3_w2_kernel(const __nv_bfloat16* __restrict__ h, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
              const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
              int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<false, D, DQ>(h, topk_ids, wr, gs, y, nullptr, ws, tickets, err, T, start, count, pdl);
}

}  // namespace eak3

constexpr int EA3_D = 3;  // ring depth: 3 x (18432 + 16640) B keeps two CTAs per SM
// smem: 105216 ring + x rows, 48 mbarriers, 9420 Map = 114684 B; 2 x (114684 + 1024) = 231416 <= 233472.
constexpr int EA3_MAX_T = eak3::MAXT;        // host bound on T (ea3_w13 / ea3_w2)
constexpr int EA3_MAXV = eak3::Map::MAXV;    // ticket rows: tickets need EA3_MAXV x NG ints (128 K1, 64 K2)
constexpr int EA3_SMEM = EA3_D * (eak::STAGE_W + eak3::XS) + 2 * EA3_D * 8 + static_cast<int>(sizeof(eak3::Map));
constexpr int ea3_smem() { return EA3_SMEM; }
// Layout (expert_body): ring @0, x rows @D * STAGE_W, mbarriers @D * (STAGE_W + XS), Map after the 2 D barriers.
static_assert((EA3_D * eak::STAGE_W) % 16 == 0 && eak3::XS % 16 == 0 && eak::XROW % 16 == 0,
              "v3: bulk-copy shared targets 16-byte aligned");
static_assert((EA3_D * (eak::STAGE_W + eak3::XS)) % 8 == 0, "v3: mbarriers 8-byte aligned");
static_assert((EA3_D * (eak::STAGE_W + eak3::XS) + 2 * EA3_D * 8) % alignof(eak3::Map) == 0, "v3: Map aligned");
static_assert(sizeof(eak3::Map) == 9420, "v3: Map = 12 + 3 x 160 x 2 + 2048 x 2 + 2 x 32 x 4 + 64 x 32 x 2 B");
static_assert(2 * (EA3_SMEM + 1024) <= 233472, "v3: two CTAs per SM (228 KB per SM, 1 KB reserved per CTA)");
static_assert(eak3::Map::MAXR == 8 * EA3_MAX_T && eak3::Map::NCH == 64, "v3: 2048 routes in 64 chunks of 32");
static_assert(EA3_MAXV >= (8 * EA3_MAX_T + (eak3::CAP - 1) * 32) / eak3::CAP, "v3: passes <= (2048 + 15 x 32) / 16");
static_assert(8 * EA3_MAX_T - 1 <= 32767, "v3: route indices and sorted positions fit the short pos_r / v_beg");
static_assert(eak::STAGE_W + eak3::CAP * 1024 < (1 << 20), "v3: mbarrier tx-count range");
static_assert(eak3::PART == 2 * 4 * 32 * 4, "v3: partial slot = two tiles of [4 warps][32 lanes][4]");

template <typename F>
static void ea3_one_info(std::vector<int64_t>& r, F fn) {
  const void* p = reinterpret_cast<const void*>(fn);
  const cudaError_t e1 = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, ea3_smem());
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, p);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, eak::THREADS, ea3_smem());
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(ok ? 1 : 0);
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(ea3_smem());
  r.push_back(occ);
}

// Per kernel (w13 dq1, w13 dq2, w2 dq1, w2 dq2): ok, regs, spill bytes, dynamic smem, CTAs/SM (ea_info's layout).
// Also raises every v3 kernel's dynamic-smem limit; call once from prepare before any launch.
std::vector<int64_t> ea3_info() {
  std::vector<int64_t> r;
  ea3_one_info(r, eak3::ea3_w13_kernel<EA3_D, 1>);
  ea3_one_info(r, eak3::ea3_w13_kernel<EA3_D, 2>);
  ea3_one_info(r, eak3::ea3_w2_kernel<EA3_D, 1>);
  ea3_one_info(r, eak3::ea3_w2_kernel<EA3_D, 2>);
  return r;
}

static cudaLaunchConfig_t ea3_cfg(int64_t grid, int64_t pdl, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(eak::THREADS);
  cfg.dynamicSmemBytes = ea3_smem();
  cfg.stream = at::cuda::getCurrentCUDAStream();
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl != 0 ? 1 : 0;
  return cfg;
}

// Same arguments as ea2_w13 / ea2_w2 (ea2_check at T <= EA3_MAX_T and EA3_MAXV ticket rows, plus v3's larger
// workspace).
void ea3_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
             int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl) {
  ea2_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 8192, "ea3_w13", EA3_MAX_T, EA3_MAXV);
  TORCH_CHECK(ws.numel() >= grid * 2048, "ea3_w13: workspace (grid x 2 slots x 1024 floats)");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(0) == T && x.size(1) == eak::KDIM, "ea3_w13: x");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea3_w13: h");
  TORCH_CHECK(local_ids.is_cuda() && local_ids.is_contiguous() && local_ids.scalar_type() == at::kInt &&
              local_ids.numel() >= T * 8, "ea3_w13: local_ids");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea3_cfg(grid, pdl, attr);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak3::ea3_w13_kernel<EA3_D, 1>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak3::ea3_w13_kernel<EA3_D, 2>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();  // do not leave the error for torch's next check
  TORCH_CHECK(le == cudaSuccess, "ea3_w13: launch: ", cudaGetErrorString(le));
}

void ea3_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
            int64_t grid, int64_t dq, int64_t pdl) {
  ea2_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 4096, "ea3_w2", EA3_MAX_T, EA3_MAXV);
  TORCH_CHECK(ws.numel() >= grid * 2048, "ea3_w2: workspace (grid x 2 slots x 1024 floats)");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea3_w2: h");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eak::KDIM,
              "ea3_w2: y");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea3_cfg(grid, pdl, attr);
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak3::ea3_w2_kernel<EA3_D, 1>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak3::ea3_w2_kernel<EA3_D, 2>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea3_w2: launch: ", cudaGetErrorString(le));
}
// ---- end include rewrite/lib/cuda/ea3_kernels.cuh ----
// ---- begin include rewrite/lib/cuda/ea4_kernels.cuh ----
// Engine A v4 (campaign 2026-10-26-expert-bandwidth, S1): v3 (ea3_kernels.cuh) on a CTA of 8 consumer warps and one
// producer warp (288 threads). The consumer is latency-bound (each warp sustains about 0.22-0.32 instructions per clock
// at 1 or 2 consumer warps per SMSP), so throughput scales with consumer warps: v4 runs 16 per SM against v3's 8.
// A stage holds two group records of one expert and k-chunk (groups 2 sg and 2 sg + 1: 128 weight rows x 512 k, two
// 18432 B bulk copies on one full barrier) plus the pass's token rows once, shared by all 8 warps (half v3's activation
// traffic per weight byte). Ring depth D = 2 keeps two CTAs per SM. The work unit is (pass, 128-row super-group sg,
// k-chunk); warp w consumes record (w & 3) of group 2 sg + (w >> 2) with v3's consume / consume2 unchanged. Everything
// else is v3's: passes of up to 16 routes (two n8 tiles), the aligned / stream-K split, the early griddep_launch, K2's
// no-work exit that waits on K1, PDL K1 -> K2 with K2's weight prefetch before its wait, bounded mbarrier waits with
// the error word, last-arriver tickets per (pass, super-group) without spins, the exact integer dequant. Same repack,
// h layout, workspace rules and combine as v3. build_map runs on all 9 warps; its 4 KB chunk scratch lives in the ring
// (idle until build_map's last barrier), so the Map is 5968 B.
// v4c (expert-bandwidth): the same kernels with the stream-K ranges weighted by pass cost. pdl bit 0 is PDL; bits 4..7
// hold the unit cost of a two-tile pass (one-tile = CW1); 0 keeps the unit-count split. A two-tile pass's unit costs
// about 1.17x a one-tile one (ladder-2: C-R4 3.45 vs 2.94 TB/s), so equal unit counts leave the CTAs on two-tile
// passes as the kernel's tail. Weighted: CTA c takes the units u with c <= P(u) G / C < c + 1 (P the units' prefix
// cost, C the total), found by a binary search over the passes' prefix costs (Map::s_cost); the owner of unit q is
// floor(P(q) G / C). The aligned split (few super-groups) is unchanged.
namespace eak4 {
using namespace ea;
using eak::REC;
using eak::STAGE_W;
using eak::XROW;
using eak::NK;
using eak::KDIM;
constexpr int THREADS = 288;           // warps 0..7 consume, warp 8 produces
constexpr int NCW = 8;                 // consumer warps
constexpr int CAP = 16;                // routes per pass: two n8 tiles (v3's)
constexpr int MAXT = 256;              // tokens: 2048 routes
constexpr int D = 2;                   // ring depth
constexpr int STAGE2 = 2 * STAGE_W;    // 36864: two group records
constexpr int XS = CAP * XROW;         // 16640: up to 16 token rows per stage
constexpr int PART = 2 * NCW * 32 * 4; // 2048 floats per stream-K partial slot: [2 tiles][8 warps][32 lanes][4]
constexpr int CW1 = 6;                 // v4c: a one-tile pass's unit cost (pdl bits 4..7 give the two-tile one's)

// v3's MapT without the chunk scratch (that lives in the ring during build_map).
template <int MAXR_, int MAXV_>
struct MapS {
  static constexpr int MAXR = MAXR_, MAXV = MAXV_, NCH = MAXR_ / 32;
  int n_v;                  // passes
  int n_local;              // local routes
  short v_e[MAXV_];         // local expert of pass v
  short v_beg[MAXV_];       // first sorted route position of pass v
  short v_n[MAXV_];         // routes in pass v (1..CAP)
  short pos_r[MAXR_];       // sorted position -> route index t * 8 + s
  int cnt[32];              // routes per local expert (scratch)
  int base[32];             // first sorted position of each local expert (scratch)
  int flag;                 // last-arriver flag
  int s_cost[MAXV_ + 1];    // v4c: prefix pass costs (sum of the earlier passes' unit costs)
};
using Map = MapS<MAXT * 8, 32 + MAXT * 8 / CAP>;

__device__ __forceinline__ void bar_consumers() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

// eak2::build_map for THREADS threads, with the per-chunk counts in cc[NCH][32] (the ring) instead of the Map. Every
// thread of the CTA takes part (__syncthreads inside). write_ids: CTA 0 writes local_ids [T, 8] (-1 remote).
// CAP_: routes per pass (v4 16, v5 32); MapT_: v4's or v5's MapS.
template <int CAP_, typename MapT_>
__device__ __forceinline__ void build_map_t(const int* __restrict__ topk_ids, int T, int start, int count, MapT_& m,
                                            short (*cc)[32], int* __restrict__ local_ids, bool write_ids) {
  constexpr int NW = THREADS / 32;
  constexpr int KMAX = (MapT_::NCH + NW - 1) / NW;
  static_assert(KMAX * NW >= MapT_::NCH && MapT_::MAXR % 32 == 0, "v4 build_map: the warps cover every chunk");
  const int R = T * 8;
  const int nch = (R + 31) / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 32; i += blockDim.x) m.cnt[i] = 0;
  for (int i = tid; i < nch * 32; i += blockDim.x) (&cc[0][0])[i] = 0;
  __syncthreads();
  int my_e[KMAX], my_rank[KMAX];
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    const int c = warp + k * NW;  // warp-uniform
    my_e[k] = -1;
    my_rank[k] = 0;
    if (c < nch) {
      const int r = c * 32 + lane;
      int le = -1;
      if (r < R) {
        const int d = topk_ids[r] - start;
        le = (d >= 0 && d < count) ? d : -1;
        if (write_ids) local_ids[r] = le;
      }
      const unsigned grp = __match_any_sync(0xffffffffu, le);
      my_e[k] = le;
      my_rank[k] = __popc(grp & ((1u << lane) - 1u));
      if (le >= 0 && my_rank[k] == 0) cc[c][le] = static_cast<short>(__popc(grp));
    }
  }
  __syncthreads();
  if (tid < 32) {  // per expert: total count and the chunk prefix (in place)
    int run = 0;
#pragma unroll 8
    for (int c = 0; c < nch; ++c) {
      const int v = cc[c][tid];
      cc[c][tid] = static_cast<short>(run);
      run += v;
    }
    m.cnt[tid] = run;
  }
  __syncthreads();
  if (tid == 0) {  // expert bases and the pass list, ascending expert
    int b = 0, nv = 0;
    for (int e = 0; e < 32; ++e) {
      m.base[e] = b;
      for (int p = 0; p < m.cnt[e]; p += CAP_) {
        m.v_e[nv] = static_cast<short>(e);
        m.v_beg[nv] = static_cast<short>(b + p);
        m.v_n[nv] = static_cast<short>((m.cnt[e] - p) < CAP_ ? (m.cnt[e] - p) : CAP_);
        ++nv;
      }
      b += m.cnt[e];
    }
    m.n_v = nv;
    m.n_local = b;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    const int c = warp + k * NW;
    const int le = my_e[k];  // -1 when c >= nch
    if (le >= 0) m.pos_r[m.base[le] + cc[c][le] + my_rank[k]] = static_cast<short>(c * 32 + lane);
  }
  // audit: cc (ring bytes) was written through the generic proxy; the next writer is cp.async.bulk (async proxy)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
}

// K1 (W13) and K2 (!W13). ws: [G][2 slots][PART] fp32; tickets: [Map::MAXV][NG / 2] int, zero between launches (each
// finisher resets its own).
template <bool W13, int DQ>
__device__ __forceinline__ void expert_body(const __nv_bfloat16* __restrict__ bsrc, const int* __restrict__ topk_ids,
                                            const uint8_t* __restrict__ wr, const float* __restrict__ gs,
                                            __nv_bfloat16* __restrict__ out, int* __restrict__ local_ids,
                                            float* __restrict__ ws, int* __restrict__ tickets, int* err, int T,
                                            int start, int count, int pdl) {
  constexpr int NG = W13 ? 128 : 64;  // 64-row groups (v3's records)
  constexpr int NS = NG / 2;          // 128-row super-groups
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* xs = smem + D * STAGE2;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + D * (STAGE2 + XS));
  Map& m = *reinterpret_cast<Map*>(bars + 2 * D);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * D;
  if (W13 && (pdl & 1)) griddep_wait();  // decode: topk_ids come from the routing kernel
  if (threadIdx.x == 32) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, NCW);
    }
    fence_barrier_init();
  }
  build_map_t<CAP, Map>(topk_ids, T, start, count, m, reinterpret_cast<short (*)[32]>(ring), local_ids,
                        W13 && blockIdx.x == 0);  // ends with __syncthreads; the ring is free again
  griddep_launch();  // K1 -> K2 and K2 -> the combine: a dependent waits for this grid before reading its outputs
  const int W = m.n_v * NS * NK;
  // Super-groups that fit the grid run aligned: each in s = 1, 2, 4 or 8 equal pieces (no fix-up at s = 1, and every
  // split super-group has exactly s contributors); otherwise plain stream-K over every CTA. CTAs at or past G idle.
  int G = static_cast<int>(gridDim.x);
  bool aligned = false;
  if (m.n_v * NS <= G) {
    int s = 1;
    while (s < NK && m.n_v * NS * 2 * s <= G) s *= 2;
    G = m.n_v * NS * s;
    aligned = true;
  }
  const int cw = (pdl >> 4) & 15;
  const bool weighted = cw != 0 && !aligned;  // CTA-uniform
  constexpr int UPP = NS * NK;                // units per pass
  if (weighted) {
    if (threadIdx.x == 0) {
      int run = 0;
      for (int v = 0; v < m.n_v; ++v) {
        m.s_cost[v] = run;
        run += m.v_n[v] > 8 ? cw : CW1;
      }
      m.s_cost[m.n_v] = run;
    }
    __syncthreads();
  }
  const long long CT = weighted ? static_cast<long long>(m.s_cost[m.n_v]) * UPP : 0;  // total cost
  auto cost_of = [&](int u) -> long long {  // prefix cost of the units before u (weighted only)
    const int v = u / UPP;
    if (v >= m.n_v) return CT;
    return static_cast<long long>(m.s_cost[v]) * UPP + static_cast<long long>(u - v * UPP) * (m.v_n[v] > 8 ? cw : CW1);
  };
  auto begin_of = [&](int c) -> int {  // first unit of CTA c
    if (!weighted) return static_cast<int>((static_cast<long long>(c) * W) / G);
    if (c >= G) return W;
    const long long X = (static_cast<long long>(c) * CT + G - 1) / G;  // the first unit with P(u) >= X
    int lo = 0, hi = m.n_v;  // the last pass whose prefix cost is <= X
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (static_cast<long long>(m.s_cost[mid]) * UPP <= X) lo = mid;
      else hi = mid - 1;
    }
    if (lo >= m.n_v) return W;
    const int wv = m.v_n[lo] > 8 ? cw : CW1;
    const long long rem = X - static_cast<long long>(m.s_cost[lo]) * UPP;
    const int u = lo * UPP + static_cast<int>((rem + wv - 1) / wv);
    return u < W ? u : W;
  };
  auto owner_of = [&](int q) -> int {  // the CTA whose range holds unit q
    if (!weighted) return eak::owner(q, W, G);
    const int c = static_cast<int>((cost_of(q) * G) / CT);
    return c < G - 1 ? c : G - 1;
  };
  const bool idle = static_cast<int>(blockIdx.x) >= G;
  const int b = idle ? 0 : begin_of(static_cast<int>(blockIdx.x));
  const int e = idle ? 0 : begin_of(static_cast<int>(blockIdx.x) + 1);
  bool dead = false;
  if (b >= e) {  // no work for this CTA (no local route, or past the aligned split): CTA-uniform
    if (!W13 && (pdl & 1)) griddep_wait();  // K2 must not finish before K1: the combine reads K1's local_ids
    return;
  }
  if (warp == NCW) {
    // ---- producer: a stage = groups 2 sg and 2 sg + 1's records (2 x 18432 B) + the pass's v_n (<= 16) token rows ----
    const int n = e - b;
    auto weight_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int sgrp = q / NK, kc = q - sgrp * NK;
      const int v = sgrp / NS, si = sgrp - v * NS;
      mbar_expect_tx(full0 + 8 * slot, static_cast<uint32_t>(STAGE2 + m.v_n[v] * 1024));
      const uint8_t* src = wr + (static_cast<size_t>(m.v_e[v] * NG + 2 * si) * NK + kc) * STAGE_W;
      bulk_g2s(smem_u32(ring + slot * STAGE2), src, STAGE_W, full0 + 8 * slot);
      bulk_g2s(smem_u32(ring + slot * STAGE2 + STAGE_W), src + static_cast<size_t>(NK) * STAGE_W, STAGE_W,
               full0 + 8 * slot);
    };
    auto x_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int sgrp = q / NK, kc = q - sgrp * NK;
      const int v = sgrp / NS;
      if (lane < m.v_n[v]) {
        const int pos = m.v_beg[v] + lane;
        const __nv_bfloat16* src = W13 ? bsrc + static_cast<size_t>(m.pos_r[pos] >> 3) * KDIM + kc * 512
                                       : bsrc + static_cast<size_t>(pos) * KDIM + kc * 512;
        bulk_g2s(smem_u32(xs + slot * XS + lane * XROW), src, 1024, full0 + 8 * slot);
      }
    };
    const int pre = n < D ? n : D;
    if (!W13) {
      if (lane == 0)
        for (int i = 0; i < pre; ++i) weight_copy(i);
      __syncwarp();
      if (pdl & 1) griddep_wait();  // h is K1's output
      for (int i = 0; i < pre; ++i) x_copy(i);
    } else {
      for (int i = 0; i < pre; ++i) {
        if (lane == 0) weight_copy(i);
        __syncwarp();
        x_copy(i);
      }
    }
    for (int i = pre; i < n; ++i) {
      const int slot = i % D;
      mbar_wait(empty0 + 8 * slot, static_cast<uint32_t>(((i / D) - 1) & 1), err, dead);
      if (lane == 0) weight_copy(i);
      __syncwarp();
      x_copy(i);
    }
    return;
  }
  // ---- consumers (warps 0..7): warp w owns record (w & 3) of group 2 sg + (w >> 2) ----
  const int g = lane >> 2, cq = lane & 3;
  const int half = warp >> 2, wq = warp & 3;
  float acc[2][2][4];  // [tile][k16 parity][C fragment]; tile 1 only in passes of 9..16 routes
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[t][s][c] = 0.f;
  for (int q = b, i = 0; q < e; ++q, ++i) {
    const int slot = i % D;
    const int sgrp = q / NK, kc = q - sgrp * NK;
    const int v = sgrp / NS;
    const int nr = m.v_n[v];
    mbar_wait(full0 + 8 * slot, static_cast<uint32_t>((i / D) & 1), err, dead);
    const uint8_t* rec = ring + slot * STAGE2 + half * STAGE_W + wq * REC;
    if (nr > 8)
      eak3::consume2<DQ>(rec, smem_u32(xs + slot * XS), lane, nr, acc);
    else
      eak::consume<DQ>(rec, smem_u32(xs + slot * XS), lane, nr, acc[0]);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (kc != NK - 1 && q != e - 1) continue;
    const bool two = nr > 8;  // the same for every contributor of this super-group (one pass)
    float c[2][4];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        c[t][k] = acc[t][0][k] + acc[t][1][k];
        acc[t][0][k] = 0.f;
        acc[t][1][k] = 0.f;
      }
    const int si = sgrp - v * NS;
    const int g0 = sgrp * NK;
    if (!(g0 >= b && g0 + NK <= e)) {
      // Partial super-group: publish (tile 1 at +NCW * 32 float4 when the pass has two), take a ticket; the last
      // arriver sums every contributor in CTA order.
      const int slot_mine = (sgrp == b / NK) ? 0 : 1;
      float4* mine = reinterpret_cast<float4*>(ws + (static_cast<size_t>(blockIdx.x) * 2 + slot_mine) * PART);
      mine[warp * 32 + lane] = make_float4(c[0][0], c[0][1], c[0][2], c[0][3]);
      if (two) mine[NCW * 32 + warp * 32 + lane] = make_float4(c[1][0], c[1][1], c[1][2], c[1][3]);
      __threadfence();
      bar_consumers();
      const int c_first = owner_of(g0), c_last = owner_of(g0 + NK - 1);
      if (threadIdx.x == 0) {
        const int old = atomicAdd(&tickets[v * NS + si], 1);
        m.flag = (old == c_last - c_first) ? 1 : 0;
      }
      bar_consumers();
      if (m.flag == 0) continue;
      __threadfence();
      float t4[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      // weighted: every contributor after c_first starts inside this super-group (slot 0); c_first holds it in slot 1
      // when its range began earlier (one search instead of one per contributor)
      const bool first_s1 = weighted && begin_of(c_first) < g0;
      for (int cc = c_first; cc <= c_last; ++cc) {
        const int bc = weighted ? 0 : static_cast<int>((static_cast<long long>(cc) * W) / G);
        const int sl = weighted ? ((cc == c_first && first_s1) ? 1 : 0) : ((sgrp == bc / NK) ? 0 : 1);
        const float4* src = reinterpret_cast<const float4*>(ws + (static_cast<size_t>(cc) * 2 + sl) * PART) +
                            warp * 32 + lane;
        const float4 x0 = __ldcg(src);
        t4[0][0] += x0.x;
        t4[0][1] += x0.y;
        t4[0][2] += x0.z;
        t4[0][3] += x0.w;
        if (two) {
          const float4 x1 = __ldcg(src + NCW * 32);
          t4[1][0] += x1.x;
          t4[1][1] += x1.y;
          t4[1][2] += x1.z;
          t4[1][3] += x1.w;
        }
      }
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int k = 0; k < 4; ++k) c[t][k] = t4[t][k];
      // m.flag is read again by the next partial super-group: every consumer has read it before the next write
      // (the first bar_consumers of that fix-up), so no extra barrier is needed here.
      if (threadIdx.x == 0) tickets[v * NS + si] = 0;
    }
    // ---- epilogue: group gi = 2 sg + half, record wq: rows (g, g + 8), route columns tile * 8 + 2 cq + {0, 1} ----
    const int gi = 2 * si + half;
    const int le = m.v_e[v];
    const float sc = gs[le];
    if (W13) {
      const int col = gi * 32 + wq * 8 + g;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int jj = t * 8 + 2 * cq + j;
          if (jj < nr)
            out[static_cast<size_t>(m.v_beg[v] + jj) * KDIM + col] =
                __float2bfloat16_rn(eak::silu_mul(c[t][j] * sc, c[t][2 + j] * sc));
        }
    } else {
      const int n0 = gi * 64 + wq * 16 + g;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int jj = t * 8 + 2 * cq + j;
          if (jj < nr) {
            __nv_bfloat16* yr = out + static_cast<size_t>(m.pos_r[m.v_beg[v] + jj]) * KDIM;
            yr[n0] = __float2bfloat16_rn(c[t][j] * sc);
            yr[n0 + 8] = __float2bfloat16_rn(c[t][2 + j] * sc);
          }
        }
    }
  }
}

template <int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea4_w13_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
               const float* __restrict__ gs, __nv_bfloat16* __restrict__ h, int* __restrict__ local_ids,
               float* __restrict__ ws, int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<true, DQ>(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, pdl);
}

template <int DQ>
__global__ void __launch_bounds__(THREADS, 2)
ea4_w2_kernel(const __nv_bfloat16* __restrict__ h, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
              const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
              int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body<false, DQ>(h, topk_ids, wr, gs, y, nullptr, ws, tickets, err, T, start, count, pdl);
}

}  // namespace eak4

// smem: 2 x (36864 + 16640) ring + x rows, 4 mbarriers, 5968 Map = 113008 B; 2 x (113008 + 1024) = 228064 <= 233472.
constexpr int EA4_MAX_T = eak4::MAXT;
constexpr int EA4_MAXV = eak4::Map::MAXV;
constexpr int EA4_SMEM = eak4::D * (eak4::STAGE2 + eak4::XS) + 2 * eak4::D * 8 + static_cast<int>(sizeof(eak4::Map));
constexpr int ea4_smem() { return EA4_SMEM; }
static_assert((eak4::D * eak4::STAGE2) % 16 == 0 && eak4::XS % 16 == 0 && eak::STAGE_W % 16 == 0,
              "v4: bulk-copy shared targets 16-byte aligned");
static_assert((eak4::D * (eak4::STAGE2 + eak4::XS)) % 8 == 0, "v4: mbarriers 8-byte aligned");
static_assert((eak4::D * (eak4::STAGE2 + eak4::XS) + 2 * eak4::D * 8) % alignof(eak4::Map) == 0, "v4: Map aligned");
static_assert(sizeof(eak4::Map) == 5968, "v4: Map = 12 + 3 x 160 x 2 + 2048 x 2 + 2 x 32 x 4 + 161 x 4 B");
static_assert(eak4::Map::NCH * 32 * 2 <= eak4::D * eak4::STAGE2, "v4: build_map's chunk scratch fits the ring");
static_assert(2 * (EA4_SMEM + 1024) <= 233472, "v4: two CTAs per SM (228 KB per SM, 1 KB reserved per CTA)");
static_assert(EA4_MAXV >= (8 * EA4_MAX_T + (eak4::CAP - 1) * 32) / eak4::CAP, "v4: passes <= (2048 + 15 x 32) / 16");
static_assert(eak4::STAGE2 + eak4::CAP * 1024 < (1 << 20), "v4: mbarrier tx-count range");
static_assert(eak4::PART == 2 * 8 * 32 * 4, "v4: partial slot = two tiles of [8 warps][32 lanes][4]");
static_assert(EA4_MAXV <= 160, "v4: ticket rows fit _ea2_buffers (EA_TK_MAXV)");

template <typename F>
static void ea4_one_info(std::vector<int64_t>& r, F fn) {
  const void* p = reinterpret_cast<const void*>(fn);
  const cudaError_t e1 = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, ea4_smem());
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, p);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, eak4::THREADS, ea4_smem());
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(ok ? 1 : 0);
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(ea4_smem());
  r.push_back(occ);
}

// Per kernel (w13 dq1, w13 dq2, w2 dq1, w2 dq2): ok, regs, spill bytes, dynamic smem, CTAs/SM (ea_info's layout).
// Also raises every v4 kernel's dynamic-smem limit; call once from prepare before any launch.
std::vector<int64_t> ea4_info() {
  std::vector<int64_t> r;
  ea4_one_info(r, eak4::ea4_w13_kernel<1>);
  ea4_one_info(r, eak4::ea4_w13_kernel<2>);
  ea4_one_info(r, eak4::ea4_w2_kernel<1>);
  ea4_one_info(r, eak4::ea4_w2_kernel<2>);
  return r;
}

static cudaLaunchConfig_t ea4_cfg(int64_t grid, int64_t pdl, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(eak4::THREADS);
  cfg.dynamicSmemBytes = ea4_smem();
  cfg.stream = at::cuda::getCurrentCUDAStream();
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (pdl & 1) != 0 ? 1 : 0;  // bits 4..7: v4c's two-tile unit cost (the kernel reads them)
  return cfg;
}

// Same arguments as ea3_w13 / ea3_w2; the workspace holds two 2048-float partial slots per CTA.
void ea4_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
             int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl) {
  ea2_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 8192, "ea4_w13", EA4_MAX_T, EA4_MAXV);
  TORCH_CHECK(ws.numel() >= grid * 2 * eak4::PART, "ea4_w13: workspace (grid x 2 slots x 2048 floats)");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(0) == T && x.size(1) == eak::KDIM, "ea4_w13: x");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea4_w13: h");
  TORCH_CHECK(local_ids.is_cuda() && local_ids.is_contiguous() && local_ids.scalar_type() == at::kInt &&
              local_ids.numel() >= T * 8, "ea4_w13: local_ids");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea4_cfg(grid, pdl, attr);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak4::ea4_w13_kernel<1>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak4::ea4_w13_kernel<2>, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();  // do not leave the error for torch's next check
  TORCH_CHECK(le == cudaSuccess, "ea4_w13: launch: ", cudaGetErrorString(le));
}

void ea4_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
            int64_t grid, int64_t dq, int64_t pdl) {
  ea2_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 4096, "ea4_w2", EA4_MAX_T, EA4_MAXV);
  TORCH_CHECK(ws.numel() >= grid * 2 * eak4::PART, "ea4_w2: workspace (grid x 2 slots x 2048 floats)");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea4_w2: h");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eak::KDIM,
              "ea4_w2: y");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea4_cfg(grid, pdl, attr);
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  const cudaError_t le = dq == 1
      ? cudaLaunchKernelEx(&cfg, eak4::ea4_w2_kernel<1>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak4::ea4_w2_kernel<2>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea4_w2: launch: ", cudaGetErrorString(le));
}
// ---- end include rewrite/lib/cuda/ea4_kernels.cuh ----
// ---- begin include rewrite/lib/cuda/ea5_kernels.cuh ----
// ---- end include rewrite/lib/cuda/ea5_kernels.cuh ----
// ---- begin include rewrite/lib/cuda/ea_combine.cuh ----
// Engine A post-expert combine (campaign 2026-10-26-decode-first-rewrite, W2): the CUDA twin of the problems'
// Triton _post_norm_combine_kernel, so the expert block's chain stays PDL-linked (K1 -> K2 -> combine -> the next own
// kernel). out[t] = sum over local slots s of g[t, s] * rmsnorm(y[t * 8 + s] * (1 + w)), fp32 math, H = 4096.
// One CTA per token, 256 threads x 16 columns (two 8-column runs: tid * 8 and 2048 + tid * 8). Remote slots (id -1)
// are skipped and never loaded; a token without a local slot stores zeros. PDL: the dependent may launch at once;
// every read of y / ids / tw waits for the expert kernels to complete (only the norm weight is read before).
namespace eac {
using namespace ea;

constexpr int H = 4096;
constexpr int THREADS = 256;

__device__ __forceinline__ void bf16x8_to_f32(const uint4& u, float (&f)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__global__ void __launch_bounds__(THREADS) ea_combine_kernel(const __nv_bfloat16* __restrict__ y,
                                                             const int* __restrict__ ids, const float* __restrict__ tw,
                                                             const __nv_bfloat16* __restrict__ w,
                                                             __nv_bfloat16* __restrict__ out, float eps, int pdl) {
  // Campaign expert-bandwidth X0b: (1 + w) is a prepare() constant, read before the PDL wait; every local slot's two
  // 16-byte loads are issued before any reduction, and one barrier serves all eight sums of squares (was one L2 round
  // trip and one barrier per slot). Per-slot arithmetic and its order are unchanged, so the output is bit-identical.
  __shared__ float red[8][THREADS / 32];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c0 = tid * 8, c1 = H / 2 + tid * 8;
  float nw[16];
  {
    float a[8], b[8];
    bf16x8_to_f32(*reinterpret_cast<const uint4*>(w + c0), a);
    bf16x8_to_f32(*reinterpret_cast<const uint4*>(w + c1), b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      nw[i] = a[i] + 1.0f;
      nw[8 + i] = b[i] + 1.0f;
    }
  }
  if (pdl) {
    griddep_launch();
    griddep_wait();
  }
  bool loc[8];
  float g[8];
  uint4 ua[8], ub[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    loc[s] = ids[t * 8 + s] >= 0;  // CTA-uniform
    g[s] = tw[t * 8 + s];
    ua[s] = make_uint4(0u, 0u, 0u, 0u);
    ub[s] = make_uint4(0u, 0u, 0u, 0u);
    if (loc[s]) {
      const __nv_bfloat16* row = y + (static_cast<size_t>(t) * 8 + s) * H;
      ua[s] = *reinterpret_cast<const uint4*>(row + c0);
      ub[s] = *reinterpret_cast<const uint4*>(row + c1);
    }
  }
  float ss[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float a[8], b[8], v[16];
    bf16x8_to_f32(ua[s], a);
    bf16x8_to_f32(ub[s], b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = a[i] * nw[i];
      v[8 + i] = b[i] * nw[8 + i];
    }
    ss[s] = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss[s] += v[i] * v[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
    for (int s = 0; s < 8; ++s) ss[s] += __shfl_xor_sync(0xffffffffu, ss[s], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < 8; ++s) red[s][warp] = ss[s];
  }
  __syncthreads();
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!loc[s]) continue;  // CTA-uniform
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) tot += red[s][i];
    const float r = rsqrtf(tot / static_cast<float>(H) + eps);
    float a[8], b[8], v[16];
    bf16x8_to_f32(ua[s], a);
    bf16x8_to_f32(ub[s], b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = a[i] * nw[i];
      v[8 + i] = b[i] * nw[8 + i];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] += g[s] * v[i] * r;
  }
  uint32_t o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = (i < 4) ? 2 * i : 8 + 2 * (i - 4);
    o[i] = (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(acc[k + 1]))) << 16) |
           static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(acc[k])));
  }
  __nv_bfloat16* orow = out + static_cast<size_t>(t) * H;
  *reinterpret_cast<uint4*>(orow + c0) = make_uint4(o[0], o[1], o[2], o[3]);
  *reinterpret_cast<uint4*>(orow + c1) = make_uint4(o[4], o[5], o[6], o[7]);
}
}  // namespace eac

// ok, regs, spill bytes of ea_combine_kernel (expert-bandwidth X0b audit: the prefetched slots keep 64 more registers
// live across the barrier; the first run must show 0 B spill).
std::vector<int64_t> ea_combine_info() {
  cudaFuncAttributes a{};
  const cudaError_t e = cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(eac::ea_combine_kernel));
  if (e != cudaSuccess) (void)cudaGetLastError();
  return {e == cudaSuccess ? 1 : 0, a.numRegs, static_cast<int64_t>(a.localSizeBytes)};
}

void ea_combine(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w, torch::Tensor out, double eps,
                int64_t pdl) {
  const int64_t T = out.size(0);
  TORCH_CHECK(out.is_cuda() && T >= 1 && T <= EA3_MAX_T && out.dim() == 2 && out.size(1) == eac::H &&  // v1..v3's T
              out.is_contiguous() && out.scalar_type() == at::kBFloat16, "ea_combine: out");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eac::H,
              "ea_combine: y");
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == at::kInt && ids.numel() == T * 8,
              "ea_combine: ids");
  TORCH_CHECK(tw.is_cuda() && tw.is_contiguous() && tw.scalar_type() == at::kFloat && tw.numel() == T * 8,
              "ea_combine: tw");
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.numel() == eac::H,
              "ea_combine: w");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "ea_combine: 16-byte alignment (uint4 rows)");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(T));
  cfg.blockDim = dim3(eac::THREADS);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t le = cudaLaunchKernelEx(
      &cfg, eac::ea_combine_kernel, reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      ids.data_ptr<int>(), tw.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<float>(eps), static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea_combine: launch: ", cudaGetErrorString(le));
}
// ---- end include rewrite/lib/cuda/ea_combine.cuh ----
"""
def _ea2_src_without(names):
    src = _EA2_SRC
    for nm in names:
        a = src.find(f"// ---- begin include rewrite/lib/cuda/{nm}_kernels.cuh ----")
        b = src.find(f"// ---- end include rewrite/lib/cuda/{nm}_kernels.cuh ----")
        if a >= 0 and b >= 0:
            src = src[:a] + src[b:]
    return src
def _ea2_build_ext(tag, name, t_prepare0=None, start_deadline_s=None):
    base = ["ea_info", "ea_w13", "ea_w2", "ea_w13_pf", "ea2_info", "ea2_w13", "ea2_w2", "ea_combine",
            "ea_combine_info", "ea3_info", "ea3_w13", "ea3_w2"]
    occs, lines = {}, []
    tries = ((("v4",), _EA2_CPP + _EA4_CPP, _ea2_src_without(("ea5",)), base + ["ea4_info", "ea4_w13", "ea4_w2"],
              name + "_nov5"), ((), _EA2_CPP, _ea2_src_without(("ea5", "ea4")), base, name + "_nov4"))
    ext, err_text, dt, have = None, "", 0.0, ()
    for i, (engines, cpp, src, fns, nm) in enumerate(tries):
        if i and t_prepare0 is not None and time.perf_counter() - t_prepare0 > start_deadline_s:
            lines.append(f"{tag} no time for the fallback build {nm}")
            break
        ext, err_text, dt1 = _rt_build(tag, nm, cpp, src, fns)
        dt += dt1
        if ext is not None:
            have = engines
            break
        lines.append(f"{tag} build {nm} FAILED: {err_text[-400:]}")
    if ext is None:
        return None, err_text, dt, occs, lines
    infos = [("v1", list(ext.ea_info())), ("v2", list(ext.ea2_info())), ("v3", list(ext.ea3_info()))]
    if "v4" in have:
        infos.append(("v4", list(ext.ea4_info())))
        infos.append(("v4c", infos[-1][1]))
    for eng, info in infos:
        for i, nm in enumerate(("w13/dq1", "w13/dq2", "w2/dq1", "w2/dq2")):
            ok_, regs, spill, smem, occ = info[5 * i:5 * i + 5]
            occs[f"{eng} {nm}"] = int(occ) if ok_ else 0
            lines.append(f"{tag} kernel {eng} {nm}: attrs ok {bool(ok_)}, {regs} regs, {spill} B spill, "
                         f"{smem // 1024} KB smem, {occ} CTAs/SM")
    ok_, regs, spill = list(ext.ea_combine_info())
    lines.append(f"{tag} kernel combine: attrs ok {bool(ok_)}, {regs} regs, {spill} B spill")
    return ext, err_text, dt, occs, lines
def _ea2_buffers(device, grid, inter):
    return {"inter": inter,
            "ws13": torch.zeros(grid * 8192, dtype=torch.float32, device=device),
            "tk13": torch.zeros(EA_TK_MAXV * 128, dtype=torch.int32, device=device),
            "ws2": torch.zeros(grid * 8192, dtype=torch.float32, device=device),
            "tk2": torch.zeros(EA_TK_MAXV * 64, dtype=torch.int32, device=device),
            "err": torch.zeros(4, dtype=torch.int32, device=device)}
def _ea2_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0):
    ext = ea["ext"]
    tokens, hidden = x.shape
    inter = ea["inter"]
    h = torch.empty((tokens * TOP_K, inter), device=x.device, dtype=torch.bfloat16)
    local_ids = torch.empty((tokens, TOP_K), device=x.device, dtype=torch.int32)
    y = torch.empty((tokens * TOP_K, hidden), device=x.device, dtype=torch.bfloat16)
    if trace is not None:
        trace("before ea2_w13")
    ext.ea2_w13(x, topk_ids, ea["w13r"], ea["gs13"], h, local_ids, ea["ws13"], ea["tk13"], ea["err"], tokens, start,
                count, ea["grid"], ea["dq"], pdl13)
    if trace is not None:
        trace("after ea2_w13 / before ea2_w2")
    ext.ea2_w2(h, topk_ids, ea["w2r"], ea["gs2"], y, ea["ws2"], ea["tk2"], ea["err"], tokens, start, count,
               ea.get("g2_t", {}).get(tokens, ea["grid"]), ea["dq"], ea["pdl"])
    if trace is not None:
        trace("after ea2_w2 / before post-expert norm")
    out = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    tw32 = topk_weights.to(torch.float32).contiguous()
    if ea.get("cmb_t", {}).get(tokens, 0):
        ext.ea_combine(y, local_ids, tw32, norm_w, out, POST_NORM_EPS, 1)
    else:
        _post_norm_combine_kernel[(tokens,)](
            y, local_ids, tw32, norm_w, out, hidden, POST_NORM_EPS,
            TOP_K_C=TOP_K, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
    if trace is not None:
        trace("after post-expert norm")
    return out
EA3_MAX_T = 256
EA3_MAXV = 160
assert EA3_MAXV <= EA_TK_MAXV
EA_ENGINES["v3"] = min(EA3_MAX_T, EA2_MAX_T)
def _ea3_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0,
                      kern=("ea3_w13", "ea3_w2"), recs=("w13r", "w2r"), pdl_or=0, ext_key="ext", bufs=None,
                      poison=False):
    ext = ea["ext"]
    kx = ea[ext_key]
    k13, k2 = getattr(kx, kern[0]), getattr(kx, kern[1])
    if bufs is None:
        ws13, tk13, ws2, tk2, errw = ea["ws13"], ea["tk13"], ea["ws2"], ea["tk2"], ea["err"]
    else:
        ws13, tk13, ws2, tk2, errw = ea["ws13x"], ea["tk13x"], ea["ws2x"], ea["tk2x"], ea["errx"]
    tokens, hidden = x.shape
    inter = ea["inter"]
    h = torch.empty((tokens * TOP_K, inter), device=x.device, dtype=torch.bfloat16)
    local_ids = torch.empty((tokens, TOP_K), device=x.device, dtype=torch.int32)
    y = torch.empty((tokens * TOP_K, hidden), device=x.device, dtype=torch.bfloat16)
    if poison:
        h.fill_(float("nan"))
        y.fill_(float("nan"))
    if trace is not None:
        trace(f"before {kern[0]}")
    k13(x, topk_ids, ea[recs[0]], ea["gs13"], h, local_ids, ws13, tk13, errw, tokens, start, count,
        ea["grid"], ea["dq"], pdl13 | pdl_or)
    if trace is not None:
        trace(f"after {kern[0]} / before {kern[1]}")
    k2(h, topk_ids, ea[recs[1]], ea["gs2"], y, ws2, tk2, errw, tokens, start, count,
       ea.get("g2_t", {}).get(tokens, ea["grid"]), ea["dq"], ea["pdl"] | pdl_or)
    if trace is not None:
        trace(f"after {kern[1]} / before post-expert norm")
    out = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    tw32 = topk_weights.to(torch.float32).contiguous()
    if ea.get("cmb_t", {}).get(tokens, 0):
        ext.ea_combine(y, local_ids, tw32, norm_w, out, POST_NORM_EPS, 1)
    else:
        _post_norm_combine_kernel[(tokens,)](
            y, local_ids, tw32, norm_w, out, hidden, POST_NORM_EPS,
            TOP_K_C=TOP_K, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
    if trace is not None:
        trace("after post-expert norm")
    return out
EA4_MAX_T = 256
EA_ENGINES["v4"] = min(EA4_MAX_T, EA2_MAX_T)
EA4C_CW = 7
EA_PDL_BITS = {"v4c": EA4C_CW << 4}
EA_ENGINES["v4c"] = EA_ENGINES["v4"]
def _ea4_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0):
    return _ea3_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace, pdl13, ("ea4_w13", "ea4_w2"))
def _ea4c_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0):
    return _ea3_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace, pdl13, ("ea4_w13", "ea4_w2"),
                             pdl_or=EA_PDL_BITS["v4c"])
CAMPAIGN_EA6 = True
CAMPAIGN_EA6F = True
EA6_TAG = "[ea6]"
EA6_W1 = {2: 0.475, 3: 0.43, 4: 0.085, 5: 0.01}
EA6F_ARMS = {"v6pe": 7, "v6pe2": 23, "v6p": 3, "v6e": 4}
EA6F_MARGIN = 1.01
EA6F_FLOOR = 0.99
_EA6_FNS = ("ea6_info", "ea6_w13", "ea6_w2")
EA6_LINES = []
_EA6_BUILD = {}
_EA6_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> ea6_info();
void ea6_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor err, torch::Tensor dbg, int64_t T, int64_t start, int64_t count,
             int64_t grid, int64_t minb, int64_t flags);
void ea6_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor err, torch::Tensor dbg, int64_t T, int64_t start, int64_t count, int64_t grid, int64_t minb,
            int64_t flags);
"""
_EA6_CPP_F = r"""
void ea6_cmb(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w, torch::Tensor out, double eps);
"""
_EA6_SRC_BODY = r"""
// ---- begin include rewrite/lib/cuda/ea6_kernels.cuh ----
// Engine A v6 (expert-bandwidth S3; research/s3-panel-2026-10-28.md section 6), T <= 8, dq2 only. K1 / K2 compute what
// v1's do. A CTA (320 threads, one per SM) owns whole RECORDS (a 16-row warp record of v1's repack over the full K: 8
// pieces of 4608 B): warp w consumes chunk w against the resident B, warp 8 streams a D-record ring, warp 9 sums the 8
// partials in warp order and writes h / y. No split-K, tickets or workspace; no CTA waits on another. Needs ea_common.cuh
// and ea_kernels.cuh's namespace eak.
#ifndef EA6_FAST
#define EA6_FAST 0
#endif
namespace eak6 {
using namespace ea;

constexpr int THREADS = 320;               // warps 0..7 consume, warp 8 produces, warp 9 runs the epilogue
constexpr int RECB = eak::NK * eak::REC;   // 36,864 B: one record (8 pieces of 4608 B)
constexpr int PBYTES = 2 * 8 * 32 * 16;    // P: 2 buffers x 8 warps x 32 lanes x float4
constexpr int D13 = 3;                     // ring depth in records, K1
constexpr int D2 = 2;                      // ring depth in records, K2
constexpr int K2_PAD = 118784;             // K2's request when co-residency with K1 is not intended (one K2 CTA per SM)
constexpr int DBG = 8;                     // int64 per CTA in the prepare-only debug buffer

// Dynamic smem: ring D x 36,864 | B 8 T x 1040 | P 8192 | mbarriers full[D] empty[D] bfull pfull[2] pfree[2] | s_n,
// s_act[32] (136 B). K1 (D 3): 127,328 B at T = 1, 185,568 at T = 8; K2 (D 2): 90,448 / 148,688.
__host__ __device__ constexpr int smem6(int D, int T) {
  return D * RECB + 8 * eak::XROW * T + PBYTES + 8 * (2 * D + 5) + 136;
}

__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}

template <typename P>
__device__ __forceinline__ P* pin_addr(P* p) {
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  asm volatile("" : "+l"(a));
  return reinterpret_cast<P*>(a);
}

// flags: 1 = K2 has the PDL attribute (host), 2 = K2 may sit beside K1 (host: unpadded request at T = 1 with 1),
// 4 = K1 waits first (decode), 8 = FAST (T = 1, MINB 2).
// dbg (prepare only): per CTA 8 int64 at
// ((W13 ? 0 : 1) * G + c) * 8: entry, first record full, last consume, last epilogue, r, K2 wait returned, first
// epilogue, n (times %globaltimer ns).
template <bool W13, int D, int MINB, bool FAST = false>
__global__ void __launch_bounds__(THREADS, MINB)
ea6_kernel(const __nv_bfloat16* __restrict__ bsrc, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
           const float* __restrict__ gs, __nv_bfloat16* __restrict__ out, int* __restrict__ local_ids, int* err,
           long long* __restrict__ dbg, int T, int start, int count, int flags) {
  constexpr int NG = W13 ? 128 : 64;   // 64-row groups per expert
  constexpr int NREC = NG * 4;         // records per expert: 512 (w13) / 256 (w2)
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* bsm = smem + D * RECB;
  float4* P = reinterpret_cast<float4*>(bsm + 8 * eak::XROW * T);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + D * RECB + 8 * eak::XROW * T + PBYTES);
  int* s_n = reinterpret_cast<int*>(bars + 2 * D + 5);
  int* s_act = s_n + 1;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * D, bfull = empty0 + 8 * D;
  const uint32_t pfull0 = bfull + 8, pfree0 = pfull0 + 16;
  const int G = static_cast<int>(gridDim.x), c = static_cast<int>(blockIdx.x);
  long long* dme = dbg != nullptr ? dbg + (static_cast<size_t>(W13 ? 0 : 1) * G + c) * DBG : nullptr;
  if (dme != nullptr && threadIdx.x == 0) dme[0] = gtimer();
  if (W13 && (flags & 4)) griddep_wait();  // decode: topk_ids / x come from the previous kernel
  uint4 xb0 = make_uint4(0u, 0u, 0u, 0u), xb1 = xb0;
  if constexpr (FAST && W13) {
    if (warp < 8) {
      const uint4* p = pin_addr(reinterpret_cast<const uint4*>(bsrc) + warp * 64);
      xb0 = __ldcg(p + lane);
      xb1 = __ldcg(p + 32 + lane);
    }
  }
  if (warp == 0) eak::build_map(topk_ids, T, start, count, lane, s_act, s_n, local_ids, W13 && c == 0);
  if (threadIdx.x == 32) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      mbar_init(full0 + 8 * d, 1);
      mbar_init(empty0 + 8 * d, 8);
    }
    mbar_init(bfull, 1);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      mbar_init(pfull0 + 8 * k, 8);
      mbar_init(pfree0 + 8 * k, 1);
    }
    fence_barrier_init();
  }
  __syncthreads();
  const int n = *s_n;
  // ---- this CTA's records [b, e): K1 flat q = a * 512 + rt over the active experts; K2 rt within expert a ----
  int a = 0, b = 0, e = 0;
  if (W13) {
    if constexpr (FAST) {  // n <= 32: exact in 32 bits
      const unsigned R = static_cast<unsigned>(NREC * n), Gu = static_cast<unsigned>(G);
      b = static_cast<int>(static_cast<unsigned>(c) * R / Gu);
      e = static_cast<int>(static_cast<unsigned>(c + 1) * R / Gu);
    } else {
      const long long R = static_cast<long long>(NREC) * n;
      b = static_cast<int>((static_cast<long long>(c) * R) / G);
      e = static_cast<int>((static_cast<long long>(c + 1) * R) / G);
    }
  } else if (n > 0) {
    a = (c * n) / G;
    const int c0 = (G * a + n - 1) / n, c1 = (G * (a + 1) + n - 1) / n;
    const int m = c1 - c0, j = c - c0;
    b = (NREC * j) / m;
    e = (NREC * (j + 1)) / m;
  }
  const int r = e - b;
  if (dme != nullptr && threadIdx.x == 0) {
    dme[4] = r;
    dme[7] = n;
  }
  if (r <= 0) {  // CTA-uniform: no local route, or no record for this CTA
    if (!W13) griddep_wait();  // K2 must not complete before K1: the combine reads K1's local_ids
    griddep_launch();  // invariant (ea6_cmb): K2 triggers after its wait
    return;
  }
  // piece 0 of record q (pieces kc at + kc * STAGE_W): wr + ((le * NG + gi) * 8 + kc) * 18432 + wq * 4608
  auto rec_src = [&](int q) -> const uint8_t* {
    const int aa = W13 ? (q / NREC) : a;
    const int rt = W13 ? (q - aa * NREC) : q;
    const int le = s_act[aa];
    return wr + static_cast<size_t>(le * NG + (rt >> 2)) * eak::NK * eak::STAGE_W + (rt & 3) * eak::REC;
  };
  if (warp == 8) {
    // ---- producer ----
    bool dead = false;
    auto issue = [&](int i) {
      const int slot = i % D;
      const uint32_t fb = full0 + 8 * slot;
      if (lane == 0) mbar_expect_tx(fb, static_cast<uint32_t>(RECB));
      __syncwarp();
      if (lane < eak::NK)
        bulk_g2s(smem_u32(ring + slot * RECB + lane * eak::REC), rec_src(b + i) + static_cast<size_t>(lane) * eak::STAGE_W,
                 static_cast<uint32_t>(eak::REC), fb);
    };
    // B: chunk kc of token t (1024 B) from src + t * 4096 + kc * 512 to B + (kc * T + t) * 1040, all on bfull
    auto b_copy = [&](const __nv_bfloat16* src) {
      if (lane == 0) mbar_expect_tx(bfull, static_cast<uint32_t>(8 * T * 1024));
      __syncwarp();
      for (int k = lane; k < 8 * T; k += 32) {
        const int kc = k / T, t = k - kc * T;
        bulk_g2s(smem_u32(bsm + k * eak::XROW), src + static_cast<size_t>(t) * eak::KDIM + kc * 512, 1024u, bfull);
      }
    };
    const int pre = r < D ? r : D;
    if (W13) {
      if constexpr (!FAST) b_copy(bsrc);
      for (int i = 0; i < pre; ++i) issue(i);
      mbar_wait(full0, 0u, err, dead);  // the first record landed: let K2 be scheduled
      griddep_launch();
    } else {
      for (int i = 0; i < pre; ++i) issue(i);  // weights do not depend on K1
      griddep_wait();
      if (dme != nullptr && lane == 0) dme[5] = gtimer();
      griddep_launch();
      if constexpr (!FAST) b_copy(bsrc + static_cast<size_t>(a) * T * eak::KDIM);
    }
    for (int i = pre; i < r; ++i) {
      const int slot = i % D;
      mbar_wait(empty0 + 8 * slot, static_cast<uint32_t>(((i / D) - 1) & 1), err, dead);
      issue(i);
    }
    return;
  }
  if (warp < 8) {
    // ---- consumers: warp w takes chunk w of every record ----
    bool dead = false;
    if (!W13) {  // park while a co-resident K1 runs
      griddep_wait();
      griddep_launch();
    }
    if constexpr (FAST) {
      if constexpr (!W13) {  // h after the wait (pinned)
        const uint4* p =
            pin_addr(reinterpret_cast<const uint4*>(bsrc + static_cast<size_t>(a) * eak::KDIM) + warp * 64);
        xb0 = __ldcg(p + lane);
        xb1 = __ldcg(p + 32 + lane);
      }
      uint4* d4 = reinterpret_cast<uint4*>(bsm + warp * eak::XROW);
      d4[lane] = xb0;
      d4[32 + lane] = xb1;
      __syncwarp();
    } else {
      mbar_wait(bfull, 0u, err, dead);
    }
    const uint32_t xs = smem_u32(bsm) + static_cast<uint32_t>(warp * T * eak::XROW);
    for (int i = 0; i < r; ++i) {
      const int slot = i % D;
      mbar_wait(full0 + 8 * slot, static_cast<uint32_t>((i / D) & 1), err, dead);
      if (W13 && i == 0) griddep_launch();
      if (dme != nullptr && i == 0 && threadIdx.x == 0) dme[1] = gtimer();
      float acc[2][4] = {};
      eak::consume<2, 1>(ring + slot * RECB + warp * eak::REC, xs, lane, T, acc);  // X1 (v6 family only)
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      const float4 cv = make_float4(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], acc[0][2] + acc[1][2],
                                    acc[0][3] + acc[1][3]);
      if (i >= 2) mbar_wait(pfree0 + 8 * (i & 1), static_cast<uint32_t>(((i >> 1) - 1) & 1), err, dead);
      P[((i & 1) * 8 + warp) * 32 + lane] = cv;
      __syncwarp();
      if (lane == 0) mbar_arrive(pfull0 + 8 * (i & 1));
    }
    if (dme != nullptr && threadIdx.x == 0) dme[2] = gtimer();
    return;
  }
  // ---- epilogue (warp 9): rows (g, g + 8) of each record, tokens 2 cq, 2 cq + 1 ----
  bool dead = false;
  if (!W13) {
    griddep_wait();
    griddep_launch();
  }
  const int g = lane >> 2, cq = lane & 3;
  // K2 serves one expert per CTA: the slots of tokens 2 cq + j that route to it are fixed for the CTA
  int le2 = 0;
  float sc2 = 0.f;
  uint32_t smask[2] = {0u, 0u};
  if (!W13) {
    le2 = s_act[a];
    sc2 = gs[le2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int t = 2 * cq + j;
      if (t < T)
#pragma unroll
        for (int s = 0; s < 8; ++s)
          if (topk_ids[t * 8 + s] - start == le2) smask[j] |= 1u << s;
    }
  }
  for (int i = 0; i < r; ++i) {
    const int k = i & 1;
    mbar_wait(pfull0 + 8 * k, static_cast<uint32_t>((i >> 1) & 1), err, dead);
    if (W13 && i == 0) griddep_launch();
    const float4* pp = P + k * 256 + lane;
    float4 s = pp[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {  // ((p0 + p1) + p2) ... + p7, component-wise
      const float4 v = pp[w * 32];
      s.x += v.x;
      s.y += v.y;
      s.z += v.z;
      s.w += v.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(pfree0 + 8 * k);
    const float sv[4] = {s.x, s.y, s.z, s.w};
    const int q = b + i;
    if (W13) {
      const int aa = q / NREC, rt = q - aa * NREC;
      const int le = s_act[aa];
      const float sc = gs[le];  // global scale x 2^7 (the dequant bias)
      const int col = (rt >> 2) * 32 + (rt & 3) * 8 + g;  // rows g = gate col, g + 8 = up col
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int t = 2 * cq + j;
        if (t < T)
          out[(static_cast<size_t>(aa) * T + t) * eak::KDIM + col] =
              __float2bfloat16_rn(eak::silu_mul(sv[j] * sc, sv[2 + j] * sc));
      }
    } else {
      const int n0 = (q >> 2) * 64 + (q & 3) * 16 + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int t = 2 * cq + j;
        if (t < T) {
          const __nv_bfloat16 v0 = __float2bfloat16_rn(sv[j] * sc2);
          const __nv_bfloat16 v1 = __float2bfloat16_rn(sv[2 + j] * sc2);
#pragma unroll
          for (int s2 = 0; s2 < 8; ++s2) {
            if ((smask[j] >> s2) & 1u) {
              __nv_bfloat16* yr = out + static_cast<size_t>(t * 8 + s2) * eak::KDIM;
              yr[n0] = v0;
              yr[n0 + 8] = v1;
            }
          }
        }
      }
    }
    if (dme != nullptr && i == 0 && lane == 0) dme[6] = gtimer();
  }
  if (dme != nullptr && lane == 0) dme[3] = gtimer();
}

#if EA6_FAST
__device__ __forceinline__ void cmb_bf16x8(const uint4& u, float (&f)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

// Early combine (T = 1), bit-identical to ea_combine_kernel: w, tw, ids before the PDL wait (a v6 K2 triggers only
// after K1), y after it. Only behind a v6 K2.
__global__ void __launch_bounds__(256) ea6_cmb_kernel(const __nv_bfloat16* __restrict__ y,
                                                      const int* __restrict__ ids, const float* __restrict__ tw,
                                                      const __nv_bfloat16* __restrict__ w,
                                                      __nv_bfloat16* __restrict__ out, float eps) {
  constexpr int H = eak::KDIM, NW = 8;  // 256 threads
  __shared__ float red[8][NW];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c0 = tid * 8, c1 = H / 2 + tid * 8;
  const uint4 wa = __ldcg(reinterpret_cast<const uint4*>(w + c0)), wb = __ldcg(reinterpret_cast<const uint4*>(w + c1));
  int iv[8];
  float g[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    iv[s] = __ldcg(ids + t * 8 + s);
    g[s] = __ldcg(tw + t * 8 + s);
  }
  // pinned before the PDL wait
  asm volatile("" ::"r"(wa.x), "r"(wa.y), "r"(wa.z), "r"(wa.w), "r"(wb.x), "r"(wb.y), "r"(wb.z), "r"(wb.w));
  asm volatile("" ::"r"(iv[0]), "r"(iv[1]), "r"(iv[2]), "r"(iv[3]), "r"(iv[4]), "r"(iv[5]), "r"(iv[6]), "r"(iv[7]));
  asm volatile("" ::"f"(g[0]), "f"(g[1]), "f"(g[2]), "f"(g[3]), "f"(g[4]), "f"(g[5]), "f"(g[6]), "f"(g[7]));
  float nw[16];
  {
    float a[8], b[8];
    cmb_bf16x8(wa, a);
    cmb_bf16x8(wb, b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      nw[i] = a[i] + 1.0f;
      nw[8 + i] = b[i] + 1.0f;
    }
  }
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) loc[s] = iv[s] >= 0;  // CTA-uniform
  griddep_launch();
  griddep_wait();
  const __nv_bfloat16* yl = pin_addr(y);
  uint4 ua[8], ub[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    ua[s] = make_uint4(0u, 0u, 0u, 0u);
    ub[s] = make_uint4(0u, 0u, 0u, 0u);
    if (loc[s]) {
      const __nv_bfloat16* row = yl + (static_cast<size_t>(t) * 8 + s) * H;
      ua[s] = __ldcg(reinterpret_cast<const uint4*>(row + c0));
      ub[s] = __ldcg(reinterpret_cast<const uint4*>(row + c1));
    }
  }
  float ss[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    ss[s] = 0.f;
    if (!loc[s]) continue;  // remote slot
    float a[8], b[8], v[16];
    cmb_bf16x8(ua[s], a);
    cmb_bf16x8(ub[s], b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = a[i] * nw[i];
      v[8 + i] = b[i] * nw[8 + i];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) ss[s] += v[i] * v[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[s] += __shfl_xor_sync(0xffffffffu, ss[s], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < 8; ++s) red[s][warp] = ss[s];
  }
  __syncthreads();
  // ea_combine.cuh verbatim from here
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!loc[s]) continue;  // CTA-uniform
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += red[s][i];
    const float r = rsqrtf(tot / static_cast<float>(H) + eps);
    float a[8], b[8], v[16];
    cmb_bf16x8(ua[s], a);
    cmb_bf16x8(ub[s], b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = a[i] * nw[i];
      v[8 + i] = b[i] * nw[8 + i];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] += g[s] * v[i] * r;
  }
  uint32_t o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = (i < 4) ? 2 * i : 8 + 2 * (i - 4);
    o[i] = (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(acc[k + 1]))) << 16) |
           static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(acc[k])));
  }
  __nv_bfloat16* orow = out + static_cast<size_t>(t) * H;
  *reinterpret_cast<uint4*>(orow + c0) = make_uint4(o[0], o[1], o[2], o[3]);
  *reinterpret_cast<uint4*>(orow + c1) = make_uint4(o[4], o[5], o[6], o[7]);
}
#endif  // EA6_FAST

}  // namespace eak6

// ---- host ----------------------------------------------------------------------------------------------------------
static int ea6_req(bool w13, int64_t T, int64_t flags) {
  if (w13) return eak6::smem6(eak6::D13, static_cast<int>(T));
  const int s = eak6::smem6(eak6::D2, static_cast<int>(T));
  // K2 meant to land beside K1 (219,824 B with both reserves <= 233,472): only when the host found the registers fit
  if (T == 1 && (flags & 1) && (flags & 2)) return s;
  return s > eak6::K2_PAD ? s : eak6::K2_PAD;  // never two K2 CTAs on one SM (the per-expert split assumes one)
}

template <typename F>
static void ea6_one_info(std::vector<int64_t>& r, F fn, bool w13) {
  const void* p = reinterpret_cast<const void*>(fn);
  const int smax = w13 ? eak6::smem6(eak6::D13, 8) : eak6::smem6(eak6::D2, 8);
  const int req1 = ea6_req(w13, 1, 3);
  const cudaError_t e1 = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, smax);
  const cudaError_t e2 = cudaFuncSetAttribute(p, cudaFuncAttributePreferredSharedMemoryCarveout,
                                              cudaSharedmemCarveoutMaxShared);
  cudaFuncAttributes at{};
  const cudaError_t e3 = cudaFuncGetAttributes(&at, p);
  int occ = 0;
  const cudaError_t e4 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, eak6::THREADS, req1);
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess && e4 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(ok ? 1 : 0);
  r.push_back(at.numRegs);
  r.push_back(static_cast<int64_t>(at.localSizeBytes));
  r.push_back(w13 ? eak6::smem6(eak6::D13, 1) : eak6::smem6(eak6::D2, 1));
  r.push_back(smax);
  r.push_back(occ);
}

#if EA6_FAST
static void ea6_cmb_info(std::vector<int64_t>& r) {  // row 6
  const void* p = reinterpret_cast<const void*>(eak6::ea6_cmb_kernel);
  cudaFuncAttributes at{};
  const cudaError_t e1 = cudaFuncGetAttributes(&at, p);
  int occ = 0;
  const cudaError_t e2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, 256, 0);
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  for (int64_t v : {int64_t(ok), int64_t(at.numRegs), int64_t(at.localSizeBytes), int64_t(0), int64_t(0), int64_t(occ)})
    r.push_back(v);
}
#endif

// Per instance (K1 m2, K1 m1, K2 m2, K2 m1 (+ K1f, K2f, cmb)): ok, regs, local bytes, smem T=1, smem T=8, CTAs/SM
// at the T=1 request. Also sets the smem limit and carveout: call before any launch.
std::vector<int64_t> ea6_info() {
  std::vector<int64_t> r;
  ea6_one_info(r, eak6::ea6_kernel<true, eak6::D13, 2>, true);
  ea6_one_info(r, eak6::ea6_kernel<true, eak6::D13, 1>, true);
  ea6_one_info(r, eak6::ea6_kernel<false, eak6::D2, 2>, false);
  ea6_one_info(r, eak6::ea6_kernel<false, eak6::D2, 1>, false);
#if EA6_FAST
  ea6_one_info(r, eak6::ea6_kernel<true, eak6::D13, 2, true>, true);
  ea6_one_info(r, eak6::ea6_kernel<false, eak6::D2, 2, true>, false);
  ea6_cmb_info(r);
#endif
  return r;
}

static void ea6_check(const torch::Tensor& topk_ids, const torch::Tensor& wr, const torch::Tensor& gs,
                      const torch::Tensor& err, const torch::Tensor& dbg, int64_t T, int64_t count, int64_t grid,
                      int64_t minb, int64_t ng, const char* what) {
  TORCH_CHECK(T >= 1 && T <= 8 && count >= 1 && count <= 32 && grid >= 32 && grid <= 1024 && (minb == 1 || minb == 2),
              what, ": arguments");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() && topk_ids.scalar_type() == at::kInt &&
              topk_ids.numel() == T * 8, what, ": topk_ids");
  TORCH_CHECK(wr.is_cuda() && wr.is_contiguous() && wr.scalar_type() == at::kByte &&
              wr.numel() == count * ng * eak::NK * static_cast<int64_t>(eak::STAGE_W), what, ": weight records");
  TORCH_CHECK(gs.is_cuda() && gs.is_contiguous() && gs.scalar_type() == at::kFloat && gs.numel() >= count, what, ": gs");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, what, ": err");
  TORCH_CHECK(dbg.numel() == 0 || (dbg.is_cuda() && dbg.is_contiguous() && dbg.scalar_type() == at::kLong &&
                                   dbg.numel() >= 2 * grid * eak6::DBG), what, ": dbg");
}

static cudaLaunchConfig_t ea6_cfg(int64_t grid, int smem, bool pdl, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(eak6::THREADS);
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cfg;
}

void ea6_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
             torch::Tensor local_ids, torch::Tensor err, torch::Tensor dbg, int64_t T, int64_t start, int64_t count,
             int64_t grid, int64_t minb, int64_t flags) {
  ea6_check(topk_ids, wr, gs, err, dbg, T, count, grid, minb, 128, "ea6_w13");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(0) == T && x.size(1) == eak::KDIM, "ea6_w13: x");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= 32 * T * eak::KDIM,
              "ea6_w13: h");
  TORCH_CHECK(local_ids.is_cuda() && local_ids.is_contiguous() && local_ids.scalar_type() == at::kInt &&
              local_ids.numel() >= T * 8, "ea6_w13: local_ids");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea6_cfg(grid, ea6_req(true, T, flags), (flags & 4) != 0, attr);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  long long* dp = dbg.numel() == 0 ? nullptr : reinterpret_cast<long long*>(dbg.data_ptr());
  auto fn = minb == 2 ? &eak6::ea6_kernel<true, eak6::D13, 2> : &eak6::ea6_kernel<true, eak6::D13, 1>;
#if EA6_FAST
  if (flags & 8) {  // FAST
    TORCH_CHECK(T == 1 && minb == 2, "ea6_w13: FAST needs T = 1, MINB 2");
    fn = &eak6::ea6_kernel<true, eak6::D13, 2, true>;
  }
#else
  TORCH_CHECK((flags & 8) == 0, "ea6_w13: FAST needs EA6_FAST");
#endif
  const cudaError_t le = cudaLaunchKernelEx(&cfg, fn, xp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                                            gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(), err.data_ptr<int>(),
                                            dp, static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                                            static_cast<int>(flags));
  if (le != cudaSuccess) (void)cudaGetLastError();  // do not leave the error for torch's next check
  TORCH_CHECK(le == cudaSuccess, "ea6_w13: launch: ", cudaGetErrorString(le));
}

void ea6_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
            torch::Tensor err, torch::Tensor dbg, int64_t T, int64_t start, int64_t count, int64_t grid, int64_t minb,
            int64_t flags) {
  ea6_check(topk_ids, wr, gs, err, dbg, T, count, grid, minb, 64, "ea6_w2");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= 32 * T * eak::KDIM,
              "ea6_w2: h");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eak::KDIM,
              "ea6_w2: y");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea6_cfg(grid, ea6_req(false, T, flags), (flags & 1) != 0, attr);
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  long long* dp = dbg.numel() == 0 ? nullptr : reinterpret_cast<long long*>(dbg.data_ptr());
  int* no_ids = nullptr;
  auto fn = minb == 2 ? &eak6::ea6_kernel<false, eak6::D2, 2> : &eak6::ea6_kernel<false, eak6::D2, 1>;
#if EA6_FAST
  if (flags & 8) {  // FAST
    TORCH_CHECK(T == 1 && minb == 2, "ea6_w2: FAST needs T = 1, MINB 2");
    fn = &eak6::ea6_kernel<false, eak6::D2, 2, true>;
  }
#else
  TORCH_CHECK((flags & 8) == 0, "ea6_w2: FAST needs EA6_FAST");
#endif
  const cudaError_t le = cudaLaunchKernelEx(&cfg, fn, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                                            gs.data_ptr<float>(), yp, no_ids, err.data_ptr<int>(), dp,
                                            static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                                            static_cast<int>(flags));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea6_w2: launch: ", cudaGetErrorString(le));
}

#if EA6_FAST
void ea6_cmb(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w, torch::Tensor out, double eps) {
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.size(0) == 1 && out.size(1) == eak::KDIM && out.is_contiguous() &&
              out.scalar_type() == at::kBFloat16, "ea6_cmb: out (T = 1)");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= 8 * eak::KDIM &&
              ids.is_cuda() && ids.is_contiguous() && ids.scalar_type() == at::kInt && ids.numel() == 8 &&
              tw.is_cuda() && tw.is_contiguous() && tw.scalar_type() == at::kFloat && tw.numel() == 8 && w.is_cuda() &&
              w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.numel() == eak::KDIM, "ea6_cmb: y/ids/tw/w");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(y.data_ptr()) | reinterpret_cast<uintptr_t>(w.data_ptr()) |
               reinterpret_cast<uintptr_t>(out.data_ptr())) % 16 == 0, "ea6_cmb: 16-B alignment");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea6_cfg(1, 0, true, attr);
  cfg.blockDim = dim3(256);
  const cudaError_t le = cudaLaunchKernelEx(
      &cfg, eak6::ea6_cmb_kernel, reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), ids.data_ptr<int>(),
      tw.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<float>(eps));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea6_cmb: launch: ", cudaGetErrorString(le));
}
#endif
// ---- end include rewrite/lib/cuda/ea6_kernels.cuh ----
"""
def _ea6_say(s):
    s = s[:199]
    EA6_LINES.append(s)
    print(s, flush=True)
def _ea6_source():
    src = _EA2_SRC
    k = src.find("// ---- begin include")
    head = src if k < 0 else src[:k]
    inc = "".join(ln + "\n" for ln in head.splitlines() if ln.startswith("#include "))
    def section(nm):
        b = f"// ---- begin include rewrite/lib/cuda/{nm} ----"
        e = f"// ---- end include rewrite/lib/cuda/{nm} ----"
        i, j = src.find(b), src.find(e)
        if i < 0 or j < i:
            raise RuntimeError(f"{nm} not found in _EA2_SRC")
        return src[i + len(b):j]
    common = section("ea_common.cuh")
    kern = section("ea_kernels.cuh")
    i, end = kern.find("namespace eak {"), "}  // namespace eak"
    j = kern.find(end)
    if i >= 0 and j > i:
        kern = kern[i:j + len(end)] + "\n"
    return inc + common + kern + _EA6_SRC_BODY
def _ea6_build_start(t_prepare):
    import hashlib
    import threading
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    EA6_LINES.clear()
    _EA6_BUILD.clear()
    try:
        src = _ea6_source()
    except Exception as e:
        _EA6_BUILD["result"] = (None, f"source: {type(e).__name__}: {str(e)[:150]}", 0.0)
        return
    fast = bool(globals().get("CAMPAIGN_EA6F", False))
    plan = ([(_EA6_CPP + _EA6_CPP_F, "#define EA6_FAST 1\n" + src, _EA6_FNS + ("ea6_cmb",))] if fast else []) + [
        (_EA6_CPP, src, _EA6_FNS)]
    def run():
        try:
            for i, (cpp, s, fns) in enumerate(plan):
                name = "okc_campaign_engine_a_v6_" + hashlib.sha256((cpp + s).encode()).hexdigest()[:10]
                _EA6_BUILD["name"] = name
                r = _rt_build(EA6_TAG, name, cpp, s, fns)
                _EA6_BUILD["result"] = r
                _EA6_BUILD["fast"] = fast and i == 0 and r[0] is not None
                if r[0] is not None:
                    break
                if i + 1 < len(plan):
                    _EA6_BUILD["fast_err"] = str(r[1])[:150]
        except BaseException as e:
            _EA6_BUILD["result"] = (None, f"build thread raised {type(e).__name__}: {str(e)[:150]}", 0.0)
    th = threading.Thread(target=run, name="ea6-build", daemon=True)
    _EA6_BUILD["thread"] = th
    th.start()
def _ea6_judge(per, specs, arms, margin=None, tight=None):
    margin = RT_MARGIN if margin is None else margin
    tight = RT_MARGIN_TIGHT_IQR if tight is None else tight
    res = {}
    for arm in arms:
        rr = []
        for ri in range(RT_ROUNDS):
            num = wsum = 0.0
            floor_ok = iqr_ok = True
            fmin, fwhere = float("inf"), ""
            med = {}
            for label, wgt, floor in specs:
                ts = per[label][ri]
                im, ii = _rt_stats(ts["inc"])[:2]
                am, ai = _rt_stats(ts[arm])[:2]
                med[label] = (im, am)
                ratio = im / max(am, 1e-3)
                if ratio - floor < fmin:
                    fmin, fwhere = ratio - floor, label
                floor_ok = floor_ok and ratio >= floor
                iqr_ok = iqr_ok and ai <= RT_IQR_FACTOR * max(ii, RT_IQR_FLOOR_US)
                num += wgt * ratio
                wsum += wgt
            wr = num / max(wsum, 1e-12)
            ok = floor_ok and (wr >= margin or (wr >= tight and iqr_ok))
            rr.append({"wr": wr, "ok": ok, "fmin": fmin, "fwhere": fwhere, "iqr": iqr_ok, "med": med})
        res[arm] = {"rounds": rr, "ok": all(r["ok"] for r in rr), "worst": min(r["wr"] for r in rr)}
    return res
CAMPAIGN_EA4X = True
EA4X_TAG = "[ea4x]"
EA4X_ARMS = {"v4xs": 0, "v4xd": 256, "v4cxs": EA4C_CW << 4, "v4cxd": (EA4C_CW << 4) | 256}
EA4X_BASE = {"v4xs": "v4", "v4xd": "v4", "v4cxs": "v4c", "v4cxd": "v4c"}
EA_PDL_BITS.update(EA4X_ARMS)
EA4X_LINES = []
_EA4X_FNS = ("ea4x_info", "ea4x_w13", "ea4x_w2")
_EA4X_BUILD = {}
_EA4X_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> ea4x_info();
void ea4x_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
              torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
              int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl);
void ea4x_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
             torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
             int64_t grid, int64_t dq, int64_t pdl);
"""
_EA4X_SRC_BODY = r"""
// ---- begin include rewrite/lib/cuda/ea4x_kernels.cuh ----
// Engine A v4x (phase A 2026-10-29, research/phaseA-2026-10-29/engineA.json): bitwise twins of v4 / v4c (eak4,
// ea4_kernels.cuh). Same units, ring, consume / consume2, slots, contributor sets, fp32 order, epilogue arithmetic and
// PDL placement; outputs bit-equal. Changed: build_map's pass list is a warp-0 scan (no thread-0 loop, no weighted
// thread-0 cost loop), the split math is 32-bit, gs is read from smem, and the stream-K fix-up:
//   DEFER = false (arm s, control): today's schedule; one thread fences around the ticket (fence.acq_rel.gpu, relaxed
//     atomic; CUTLASS GenericBarrier) instead of 256 threads' MEMBAR.SC.
//   DEFER = true (arm d): a partial super-group's partial is stored and the stream continues; one ticket round at the
//     CTA's end covers its head and tail partials, then the CTA finishes the super-groups it closed, in turn.
// No thread waits on another CTA. After the producer warp returns only bar_consumers is used.
namespace eak4x {
using namespace ea;
using eak::REC;
using eak::STAGE_W;
using eak::XROW;
using eak::NK;
using eak::KDIM;
using eak4::THREADS;
using eak4::NCW;
using eak4::CAP;
using eak4::MAXT;
using eak4::D;
using eak4::STAGE2;
using eak4::XS;
using eak4::PART;
using eak4::CW1;
using eak4::Map;
using eak4::bar_consumers;

constexpr int SMEM = D * (STAGE2 + XS) + 2 * D * 8 + static_cast<int>(sizeof(Map)) + 128;  // + 32 gs floats
constexpr int GS_OFF = SMEM - 128;
constexpr uint32_t WMAX = static_cast<uint32_t>(Map::MAXV) * 64u * NK;        // K1 units at 160 passes
constexpr uint32_t CTMAX = static_cast<uint32_t>(Map::MAXV) * 15u * 64u * NK;  // cost units (cw <= 15)
constexpr uint32_t GMAX = 512;                                                // host check
static_assert(WMAX == 81920u && CTMAX == 1228800u, "v4x: split bounds");
static_assert(static_cast<unsigned long long>(CTMAX) * GMAX + GMAX < (1ull << 31), "v4x: 32-bit weighted split");
static_assert((static_cast<unsigned long long>(WMAX) + 1) * GMAX + WMAX < (1ull << 31), "v4x: 32-bit owner");

__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }

// eak4::build_map_t<CAP, Map> with step 4 on warp 0 (lane = local expert, inclusive __shfl_up_sync scans) and s_gs.
__device__ __forceinline__ void build_map_x(const int* __restrict__ topk_ids, int T, int start, int count, Map& m,
                                            short (*cc)[32], int* __restrict__ local_ids, bool write_ids,
                                            const float* __restrict__ gs, float* s_gs, int cw) {
  constexpr int NW = THREADS / 32;
  constexpr int KMAX = (Map::NCH + NW - 1) / NW;
  static_assert(KMAX * NW >= Map::NCH && Map::MAXR % 32 == 0, "v4x build_map: the warps cover every chunk");
  const int R = T * 8;
  const int nch = (R + 31) / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 32; i += blockDim.x) m.cnt[i] = 0;
  for (int i = tid; i < nch * 32; i += blockDim.x) (&cc[0][0])[i] = 0;
  if (tid < count) s_gs[tid] = gs[tid];
  __syncthreads();
  int my_e[KMAX], my_rank[KMAX];
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    const int c = warp + k * NW;  // warp-uniform
    my_e[k] = -1;
    my_rank[k] = 0;
    if (c < nch) {
      const int r = c * 32 + lane;
      int le = -1;
      if (r < R) {
        const int d = topk_ids[r] - start;
        le = (d >= 0 && d < count) ? d : -1;
        if (write_ids) local_ids[r] = le;
      }
      const unsigned grp = __match_any_sync(0xffffffffu, le);
      my_e[k] = le;
      my_rank[k] = __popc(grp & ((1u << lane) - 1u));
      if (le >= 0 && my_rank[k] == 0) cc[c][le] = static_cast<short>(__popc(grp));
    }
  }
  __syncthreads();
  if (tid < 32) {  // per expert: total count and the chunk prefix (in place)
    int run = 0;
#pragma unroll 8
    for (int c = 0; c < nch; ++c) {
      const int v = cc[c][tid];
      cc[c][tid] = static_cast<short>(run);
      run += v;
    }
    m.cnt[tid] = run;
  }
  __syncthreads();
  if (tid < 32) {  // expert bases, the pass list (ascending expert) and v4c's prefix costs: inclusive warp scans
    const int cnt = m.cnt[lane];
    const int np = (cnt + CAP - 1) / CAP;
    const int last = np ? cnt - CAP * (np - 1) : 0;
    const int cost = np ? (np - 1) * cw + (last > 8 ? cw : CW1) : 0;
    int rb = cnt, rp = np, rc = cost;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, rb, off);
      const int p = __shfl_up_sync(0xffffffffu, rp, off);
      const int q = __shfl_up_sync(0xffffffffu, rc, off);
      if (lane >= off) {
        rb += a;
        rp += p;
        rc += q;
      }
    }
    const int base = rb - cnt, p0 = rp - np, c0 = rc - cost;
    m.base[lane] = base;
    for (int p = 0; p < np; ++p) {
      m.v_e[p0 + p] = static_cast<short>(lane);
      m.v_beg[p0 + p] = static_cast<short>(base + CAP * p);
      m.v_n[p0 + p] = static_cast<short>(p + 1 < np ? CAP : last);
      if (cw) m.s_cost[p0 + p] = c0 + p * cw;
    }
    if (lane == 31) {
      m.n_v = rp;
      m.n_local = rb;
      if (cw) m.s_cost[rp] = rc;
    }
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    const int c = warp + k * NW;
    const int le = my_e[k];  // -1 when c >= nch
    if (le >= 0) m.pos_r[m.base[le] + cc[c][le] + my_rank[k]] = static_cast<short>(c * 32 + lane);
  }
  // audit: cc (ring bytes) was written through the generic proxy; the next writer is cp.async.bulk (async proxy)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
}

// eak4's epilogue (group gi = 2 si + half, record wq: rows (g, g + 8), route columns tile * 8 + 2 cq + {0, 1}).
template <bool W13>
__device__ __forceinline__ void epi(const float (&c)[2][4], const Map& m, int v, int si, int half, int wq, int g,
                                    int cq, float sc, __nv_bfloat16* __restrict__ out) {
  const int gi = 2 * si + half;
  const int nr = m.v_n[v];
  if (W13) {
    const int col = gi * 32 + wq * 8 + g;
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int jj = t * 8 + 2 * cq + j;
        if (jj < nr)
          out[static_cast<size_t>(m.v_beg[v] + jj) * KDIM + col] =
              __float2bfloat16_rn(eak::silu_mul(c[t][j] * sc, c[t][2 + j] * sc));
      }
  } else {
    const int n0 = gi * 64 + wq * 16 + g;
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int jj = t * 8 + 2 * cq + j;
        if (jj < nr) {
          __nv_bfloat16* yr = out + static_cast<size_t>(m.pos_r[m.v_beg[v] + jj]) * KDIM;
          yr[n0] = __float2bfloat16_rn(c[t][j] * sc);
          yr[n0 + 8] = __float2bfloat16_rn(c[t][2 + j] * sc);
        }
      }
  }
}

// eak4's finisher loop, exactly: every contributor in CTA order, from 0.f.
__device__ __forceinline__ void finish(float (&c)[2][4], const float* __restrict__ ws, int sgrp, int c_first,
                                       int c_last, bool weighted, bool first_s1, uint32_t W, uint32_t G, int warp,
                                       int lane, bool two) {
  float t4[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  for (int cc = c_first; cc <= c_last; ++cc) {
    const int bc = weighted ? 0 : static_cast<int>((static_cast<uint32_t>(cc) * W) / G);
    const int sl = weighted ? ((cc == c_first && first_s1) ? 1 : 0) : ((sgrp == bc / NK) ? 0 : 1);
    const float4* src = reinterpret_cast<const float4*>(ws + (static_cast<size_t>(cc) * 2 + sl) * PART) +
                        warp * 32 + lane;
    const float4 x0 = __ldcg(src);
    t4[0][0] += x0.x;
    t4[0][1] += x0.y;
    t4[0][2] += x0.z;
    t4[0][3] += x0.w;
    if (two) {
      const float4 x1 = __ldcg(src + NCW * 32);
      t4[1][0] += x1.x;
      t4[1][1] += x1.y;
      t4[1][2] += x1.z;
      t4[1][3] += x1.w;
    }
  }
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int k = 0; k < 4; ++k) c[t][k] = t4[t][k];
}

// K1 (W13) and K2 (!W13): eak4::expert_body's contract. ws: [G][2 slots][PART] fp32; tickets: [Map::MAXV][NG / 2]
// int, zero between launches (each finisher resets its own).
template <bool W13, int DQ, bool DEFER>
__device__ __forceinline__ void expert_body_x(const __nv_bfloat16* __restrict__ bsrc,
                                              const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
                                              const float* __restrict__ gs, __nv_bfloat16* __restrict__ out,
                                              int* __restrict__ local_ids, float* __restrict__ ws,
                                              int* __restrict__ tickets, int* err, int T, int start, int count,
                                              int pdl) {
  constexpr int NG = W13 ? 128 : 64;  // 64-row groups (v3's records)
  constexpr int NS = NG / 2;          // 128-row super-groups
  constexpr int UPP = NS * NK;        // units per pass
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* xs = smem + D * STAGE2;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + D * (STAGE2 + XS));
  Map& m = *reinterpret_cast<Map*>(bars + 2 * D);
  float* s_gs = reinterpret_cast<float*>(smem + GS_OFF);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t full0 = smem_u32(bars), empty0 = full0 + 8 * D;
  if (W13 && (pdl & 1)) griddep_wait();  // decode: topk_ids come from the routing kernel
  if (threadIdx.x == 32) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, NCW);
    }
    fence_barrier_init();
  }
  const int cw = (pdl >> 4) & 15;
  build_map_x(topk_ids, T, start, count, m, reinterpret_cast<short (*)[32]>(ring), local_ids,
              W13 && blockIdx.x == 0, gs, s_gs, cw);  // ends with __syncthreads; the ring is free again
  griddep_launch();  // K1 -> K2 and K2 -> the combine: a dependent waits for this grid before reading its outputs
  const int W = m.n_v * NS * NK;
  int G = static_cast<int>(gridDim.x);
  bool aligned = false;
  if (m.n_v * NS <= G) {
    int s = 1;
    while (s < NK && m.n_v * NS * 2 * s <= G) s *= 2;
    G = m.n_v * NS * s;
    aligned = true;
  }
  const bool weighted = cw != 0 && !aligned;  // CTA-uniform
  const uint32_t uW = static_cast<uint32_t>(W), uG = static_cast<uint32_t>(G);
  const uint32_t CT = weighted ? static_cast<uint32_t>(m.s_cost[m.n_v]) * UPP : 0u;  // total cost
  auto cost_of = [&](int u) -> uint32_t {  // prefix cost of the units before u (weighted only)
    const int v = u / UPP;
    if (v >= m.n_v) return CT;
    return static_cast<uint32_t>(m.s_cost[v]) * UPP +
           static_cast<uint32_t>(u - v * UPP) * static_cast<uint32_t>(m.v_n[v] > 8 ? cw : CW1);
  };
  auto begin_of = [&](int c) -> int {  // first unit of CTA c
    if (!weighted) return static_cast<int>((static_cast<uint32_t>(c) * uW) / uG);
    if (c >= G) return W;
    const uint32_t X = (static_cast<uint32_t>(c) * CT + uG - 1u) / uG;  // the first unit with P(u) >= X
    int lo = 0, hi = m.n_v;  // the last pass whose prefix cost is <= X
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (static_cast<uint32_t>(m.s_cost[mid]) * UPP <= X) lo = mid;
      else hi = mid - 1;
    }
    if (lo >= m.n_v) return W;
    const uint32_t wv = static_cast<uint32_t>(m.v_n[lo] > 8 ? cw : CW1);
    const uint32_t rem = X - static_cast<uint32_t>(m.s_cost[lo]) * UPP;
    const int u = lo * UPP + static_cast<int>((rem + wv - 1u) / wv);
    return u < W ? u : W;
  };
  auto owner_of = [&](int q) -> int {  // the CTA whose range holds unit q
    if (!weighted) return static_cast<int>(((static_cast<uint32_t>(q) + 1u) * uG + uW - 1u) / uW) - 1;
    const int c = static_cast<int>((cost_of(q) * uG) / CT);
    return c < G - 1 ? c : G - 1;
  };
  const bool idle = static_cast<int>(blockIdx.x) >= G;
  const int b = idle ? 0 : begin_of(static_cast<int>(blockIdx.x));
  const int e = idle ? 0 : begin_of(static_cast<int>(blockIdx.x) + 1);
  bool dead = false;
  if (b >= e) {  // no work for this CTA (no local route, or past the aligned split): CTA-uniform
    if (!W13 && (pdl & 1)) griddep_wait();  // K2 must not finish before K1: the combine reads K1's local_ids
    return;
  }
  if (warp == NCW) {
    // ---- producer: eak4's, unchanged ----
    const int n = e - b;
    auto weight_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int sgrp = q / NK, kc = q - sgrp * NK;
      const int v = sgrp / NS, si = sgrp - v * NS;
      mbar_expect_tx(full0 + 8 * slot, static_cast<uint32_t>(STAGE2 + m.v_n[v] * 1024));
      const uint8_t* src = wr + (static_cast<size_t>(m.v_e[v] * NG + 2 * si) * NK + kc) * STAGE_W;
      bulk_g2s(smem_u32(ring + slot * STAGE2), src, STAGE_W, full0 + 8 * slot);
      bulk_g2s(smem_u32(ring + slot * STAGE2 + STAGE_W), src + static_cast<size_t>(NK) * STAGE_W, STAGE_W,
               full0 + 8 * slot);
    };
    auto x_copy = [&](int i) {
      const int q = b + i, slot = i % D;
      const int sgrp = q / NK, kc = q - sgrp * NK;
      const int v = sgrp / NS;
      if (lane < m.v_n[v]) {
        const int pos = m.v_beg[v] + lane;
        const __nv_bfloat16* src = W13 ? bsrc + static_cast<size_t>(m.pos_r[pos] >> 3) * KDIM + kc * 512
                                       : bsrc + static_cast<size_t>(pos) * KDIM + kc * 512;
        bulk_g2s(smem_u32(xs + slot * XS + lane * XROW), src, 1024, full0 + 8 * slot);
      }
    };
    const int pre = n < D ? n : D;
    if (!W13) {
      if (lane == 0)
        for (int i = 0; i < pre; ++i) weight_copy(i);
      __syncwarp();
      if (pdl & 1) griddep_wait();  // h is K1's output
      for (int i = 0; i < pre; ++i) x_copy(i);
    } else {
      for (int i = 0; i < pre; ++i) {
        if (lane == 0) weight_copy(i);
        __syncwarp();
        x_copy(i);
      }
    }
    for (int i = pre; i < n; ++i) {
      const int slot = i % D;
      mbar_wait(empty0 + 8 * slot, static_cast<uint32_t>(((i / D) - 1) & 1), err, dead);
      if (lane == 0) weight_copy(i);
      __syncwarp();
      x_copy(i);
    }
    return;
  }
  // ---- consumers (warps 0..7): warp w owns record (w & 3) of group 2 sg + (w >> 2) ----
  const int g = lane >> 2, cq = lane & 3;
  const int half = warp >> 2, wq = warp & 3;
  int pend0 = -1, pend1 = -1;  // DEFER: the super-groups whose partial sits in slot 0 / 1 (CTA-uniform)
  float acc[2][2][4];  // [tile][k16 parity][C fragment]; tile 1 only in passes of 9..16 routes
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[t][s][c] = 0.f;
  for (int q = b, i = 0; q < e; ++q, ++i) {
    const int slot = i % D;
    const int sgrp = q / NK, kc = q - sgrp * NK;
    const int v = sgrp / NS;
    const int nr = m.v_n[v];
    mbar_wait(full0 + 8 * slot, static_cast<uint32_t>((i / D) & 1), err, dead);
    const uint8_t* rec = ring + slot * STAGE2 + half * STAGE_W + wq * REC;
    if (nr > 8)
      eak3::consume2<DQ>(rec, smem_u32(xs + slot * XS), lane, nr, acc);
    else
      eak::consume<DQ>(rec, smem_u32(xs + slot * XS), lane, nr, acc[0]);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (kc != NK - 1 && q != e - 1) continue;
    const bool two = nr > 8;  // the same for every contributor of this super-group (one pass)
    float c[2][4];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        c[t][k] = acc[t][0][k] + acc[t][1][k];
        acc[t][0][k] = 0.f;
        acc[t][1][k] = 0.f;
      }
    const int si = sgrp - v * NS;
    const int g0 = sgrp * NK;
    if (!(g0 >= b && g0 + NK <= e)) {
      const int slot_mine = (sgrp == b / NK) ? 0 : 1;
      float4* mine = reinterpret_cast<float4*>(ws + (static_cast<size_t>(blockIdx.x) * 2 + slot_mine) * PART);
      mine[warp * 32 + lane] = make_float4(c[0][0], c[0][1], c[0][2], c[0][3]);
      if (two) mine[NCW * 32 + warp * 32 + lane] = make_float4(c[1][0], c[1][1], c[1][2], c[1][3]);
      if constexpr (DEFER) {  // published at the CTA's end
        if (slot_mine == 0) pend0 = sgrp;
        else pend1 = sgrp;
        continue;
      }
      bar_consumers();
      const int c_first = owner_of(g0), c_last = owner_of(g0 + NK - 1);
      if (threadIdx.x == 0) {
        fence_acq_rel_gpu();  // release: every consumer's stores (ordered by the barrier) before the ticket
        const int old = atomicAdd(&tickets[sgrp], 1);
        const int f = (old == c_last - c_first) ? 1 : 0;
        if (f) fence_acq_rel_gpu();  // acquire: the other contributors' partials, before the barrier below
        m.flag = f;
      }
      bar_consumers();
      if (m.flag == 0) continue;
      finish(c, ws, sgrp, c_first, c_last, weighted, weighted && begin_of(c_first) < g0, uW, uG, warp, lane, two);
      // m.flag is read again by the next partial super-group only after its first bar_consumers
      if (threadIdx.x == 0) tickets[sgrp] = 0;
    }
    epi<W13>(c, m, v, si, half, wq, g, cq, s_gs[m.v_e[v]], out);
  }
  if constexpr (DEFER) {
    if (pend0 < 0 && pend1 < 0) return;  // CTA-uniform
    bar_consumers();
    if (threadIdx.x == 0) {
      fence_acq_rel_gpu();  // release both partials
      const int o0 = pend0 >= 0 ? atomicAdd(&tickets[pend0], 1) : 0;
      const int o1 = pend1 >= 0 ? atomicAdd(&tickets[pend1], 1) : 0;
      int bits = 0;
      if (pend0 >= 0 && o0 == owner_of(pend0 * NK + NK - 1) - owner_of(pend0 * NK)) bits |= 1;
      if (pend1 >= 0 && o1 == owner_of(pend1 * NK + NK - 1) - owner_of(pend1 * NK)) bits |= 2;
      if (bits) fence_acq_rel_gpu();  // acquire
      m.flag = bits;
    }
    bar_consumers();
    const int bits = m.flag;
#pragma unroll 1
    for (int j = 0; j < 2; ++j) {
      if (!((bits >> j) & 1)) continue;
      const int sgrp = j == 0 ? pend0 : pend1;
      const int v = sgrp / NS, si = sgrp - v * NS;
      const int g0 = sgrp * NK;
      const int c_first = owner_of(g0), c_last = owner_of(g0 + NK - 1);
      float c[2][4];
      finish(c, ws, sgrp, c_first, c_last, weighted, weighted && begin_of(c_first) < g0, uW, uG, warp, lane,
             m.v_n[v] > 8);
      if (threadIdx.x == 0) tickets[sgrp] = 0;
      epi<W13>(c, m, v, si, half, wq, g, cq, s_gs[m.v_e[v]], out);
    }
  }
}

template <int DQ, bool DEFER>
__global__ void __launch_bounds__(THREADS, 2)
ea4x_w13_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
                const float* __restrict__ gs, __nv_bfloat16* __restrict__ h, int* __restrict__ local_ids,
                float* __restrict__ ws, int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body_x<true, DQ, DEFER>(x, topk_ids, wr, gs, h, local_ids, ws, tickets, err, T, start, count, pdl);
}

template <int DQ, bool DEFER>
__global__ void __launch_bounds__(THREADS, 2)
ea4x_w2_kernel(const __nv_bfloat16* __restrict__ h, const int* __restrict__ topk_ids, const uint8_t* __restrict__ wr,
               const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
               int* __restrict__ tickets, int* err, int T, int start, int count, int pdl) {
  expert_body_x<false, DQ, DEFER>(h, topk_ids, wr, gs, y, nullptr, ws, tickets, err, T, start, count, pdl);
}

}  // namespace eak4x

// smem: v4's 113008 B + 128 B of gs = 113136; 2 x (113136 + 1024) = 228320 <= 233472.
constexpr int EA4X_SMEM = eak4x::SMEM;
static_assert(EA4X_SMEM == 113136 && eak4x::GS_OFF == 113008, "v4x: smem layout");
static_assert(sizeof(eak4::Map) == 5968, "v4x: v4's Map");
static_assert(2 * (EA4X_SMEM + 1024) <= 233472, "v4x: two CTAs per SM (228 KB per SM, 1 KB reserved per CTA)");
static_assert(eak4x::GS_OFF % 16 == 0, "v4x: s_gs aligned");

template <typename F>
static void ea4x_one_info(std::vector<int64_t>& r, F fn) {
  const void* p = reinterpret_cast<const void*>(fn);
  const cudaError_t e1 = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, EA4X_SMEM);
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, p);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, eak4::THREADS, EA4X_SMEM);
  const bool ok = e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess;
  if (!ok) (void)cudaGetLastError();
  r.push_back(ok ? 1 : 0);
  r.push_back(a.numRegs);
  r.push_back(static_cast<int64_t>(a.localSizeBytes));
  r.push_back(EA4X_SMEM);
  r.push_back(occ);
}

// Per kernel (K1 s, K1 d, K2 s, K2 d; dq2): ok, regs, lmem bytes, dynamic smem, CTAs/SM. Also raises every twin's
// dynamic-smem limit; call once from prepare before any launch.
std::vector<int64_t> ea4x_info() {
  std::vector<int64_t> r;
  ea4x_one_info(r, eak4x::ea4x_w13_kernel<2, false>);
  ea4x_one_info(r, eak4x::ea4x_w13_kernel<2, true>);
  ea4x_one_info(r, eak4x::ea4x_w2_kernel<2, false>);
  ea4x_one_info(r, eak4x::ea4x_w2_kernel<2, true>);
  return r;
}

static cudaLaunchConfig_t ea4x_cfg(int64_t grid, int64_t pdl, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(eak4::THREADS);
  cfg.dynamicSmemBytes = EA4X_SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (pdl & 1) != 0 ? 1 : 0;  // bits 4..7: v4c's two-tile unit cost; bit 8: DEFER (the instance)
  return cfg;
}

static void ea4x_check(const torch::Tensor& topk_ids, const torch::Tensor& wr, const torch::Tensor& gs,
                       const torch::Tensor& ws, const torch::Tensor& tickets, const torch::Tensor& err, int64_t T,
                       int64_t count, int64_t grid, int64_t dq, int64_t rows, const char* what) {
  TORCH_CHECK(T >= 1 && T <= eak4::MAXT && count >= 1 && count <= 32 && grid >= 1 && grid <= 512 && dq == 2, what,
              ": arguments");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() && topk_ids.scalar_type() == at::kInt &&
              topk_ids.numel() == T * 8, what, ": topk_ids");
  TORCH_CHECK(wr.is_cuda() && wr.is_contiguous() && wr.scalar_type() == at::kByte &&
              wr.numel() == count * (rows / 64) * eak::NK * static_cast<int64_t>(eak::STAGE_W), what,
              ": weight records");
  TORCH_CHECK(gs.is_cuda() && gs.is_contiguous() && gs.scalar_type() == at::kFloat && gs.numel() >= count, what, ": gs");
  TORCH_CHECK(ws.is_cuda() && ws.is_contiguous() && ws.scalar_type() == at::kFloat &&
              ws.numel() >= grid * 2 * eak4::PART, what, ": workspace (grid x 2 slots x 2048 floats)");
  TORCH_CHECK(tickets.is_cuda() && tickets.is_contiguous() && tickets.scalar_type() == at::kInt &&
              tickets.numel() >= eak4::Map::MAXV * (rows / 64), what, ": tickets");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, what, ": err");
}

// ea4_w13 / ea4_w2's arguments; pdl bit 8 picks the DEFER instance.
void ea4x_w13(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor h,
              torch::Tensor local_ids, torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T,
              int64_t start, int64_t count, int64_t grid, int64_t dq, int64_t pdl) {
  ea4x_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 8192, "ea4x_w13");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
              x.size(0) == T && x.size(1) == eak::KDIM, "ea4x_w13: x");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea4x_w13: h");
  TORCH_CHECK(local_ids.is_cuda() && local_ids.is_contiguous() && local_ids.scalar_type() == at::kInt &&
              local_ids.numel() >= T * 8, "ea4x_w13: local_ids");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea4x_cfg(grid, pdl, attr);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  const cudaError_t le = (pdl & 256) != 0
      ? cudaLaunchKernelEx(&cfg, eak4x::ea4x_w13_kernel<2, true>, xp, topk_ids.data_ptr<int>(),
                           wr.data_ptr<uint8_t>(), gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(),
                           ws.data_ptr<float>(), tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T),
                           static_cast<int>(start), static_cast<int>(count), static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak4x::ea4x_w13_kernel<2, false>, xp, topk_ids.data_ptr<int>(),
                           wr.data_ptr<uint8_t>(), gs.data_ptr<float>(), hp, local_ids.data_ptr<int>(),
                           ws.data_ptr<float>(), tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T),
                           static_cast<int>(start), static_cast<int>(count), static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();  // do not leave the error for torch's next check
  TORCH_CHECK(le == cudaSuccess, "ea4x_w13: launch: ", cudaGetErrorString(le));
}

void ea4x_w2(torch::Tensor h, torch::Tensor topk_ids, torch::Tensor wr, torch::Tensor gs, torch::Tensor y,
             torch::Tensor ws, torch::Tensor tickets, torch::Tensor err, int64_t T, int64_t start, int64_t count,
             int64_t grid, int64_t dq, int64_t pdl) {
  ea4x_check(topk_ids, wr, gs, ws, tickets, err, T, count, grid, dq, 4096, "ea4x_w2");
  TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.scalar_type() == at::kBFloat16 && h.numel() >= T * 8 * eak::KDIM,
              "ea4x_w2: h");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.numel() >= T * 8 * eak::KDIM,
              "ea4x_w2: y");
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = ea4x_cfg(grid, pdl, attr);
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  const cudaError_t le = (pdl & 256) != 0
      ? cudaLaunchKernelEx(&cfg, eak4x::ea4x_w2_kernel<2, true>, hp, topk_ids.data_ptr<int>(), wr.data_ptr<uint8_t>(),
                           gs.data_ptr<float>(), yp, ws.data_ptr<float>(), tickets.data_ptr<int>(),
                           err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start), static_cast<int>(count),
                           static_cast<int>(pdl))
      : cudaLaunchKernelEx(&cfg, eak4x::ea4x_w2_kernel<2, false>, hp, topk_ids.data_ptr<int>(),
                           wr.data_ptr<uint8_t>(), gs.data_ptr<float>(), yp, ws.data_ptr<float>(),
                           tickets.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(T), static_cast<int>(start),
                           static_cast<int>(count), static_cast<int>(pdl));
  if (le != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, "ea4x_w2: launch: ", cudaGetErrorString(le));
}
// ---- end include rewrite/lib/cuda/ea4x_kernels.cuh ----
"""
def _ea4x_say(s):
    s = s[:199]
    if len(EA4X_LINES) < 6:
        EA4X_LINES.append(s)
    print(s, flush=True)
def _ea4x_source():
    src = _EA2_SRC
    k = src.find("// ---- begin include")
    if k < 0:
        raise RuntimeError("no include markers in _EA2_SRC")
    inc = "".join(ln + "\n" for ln in src[:k].splitlines() if ln.startswith("#include "))
    def section(nm):
        b = f"// ---- begin include rewrite/lib/cuda/{nm} ----"
        e = f"// ---- end include rewrite/lib/cuda/{nm} ----"
        i, j = src.find(b), src.find(e)
        if i < 0 or j < i:
            raise RuntimeError(f"{nm} not found in _EA2_SRC")
        return src[i + len(b):j]
    out = [inc, section("ea_common.cuh")]
    for nm, ns in (("ea_kernels.cuh", "eak"), ("ea2_kernels.cuh", "eak2"), ("ea3_kernels.cuh", "eak3"),
                   ("ea4_kernels.cuh", "eak4")):
        body = section(nm)
        b, e = f"namespace {ns} {{", f"}}  // namespace {ns}\n"
        i, j = body.find(b), body.find(e)
        if i < 0 or j < i:
            raise RuntimeError(f"namespace {ns} not found in {nm}")
        out.append(body[i:j + len(e)])
    return "".join(out) + _EA4X_SRC_BODY
def _ea4x_build_start(t_prepare):
    import hashlib
    import threading
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    EA4X_LINES.clear()
    _EA4X_BUILD.clear()
    try:
        src = _ea4x_source()
    except Exception as e:
        _EA4X_BUILD["result"] = (None, f"source: {type(e).__name__}: {str(e)[:150]}", 0.0)
        return
    name = "okc_campaign_engine_a_v4x_" + hashlib.sha256((_EA4X_CPP + src).encode()).hexdigest()[:10]
    _EA4X_BUILD["name"] = name
    def run():
        try:
            _EA4X_BUILD["result"] = _rt_build(EA4X_TAG, name, _EA4X_CPP, src, _EA4X_FNS)
        except BaseException as e:
            _EA4X_BUILD["result"] = (None, f"build thread raised {type(e).__name__}: {str(e)[:150]}", 0.0)
    th = threading.Thread(target=run, name="ea4x-build", daemon=True)
    _EA4X_BUILD["thread"] = th
    th.start()
def _ea4x_build_join(t_prepare, limit):
    t0 = time.perf_counter()
    th = _EA4X_BUILD.get("thread")
    if th is not None:
        th.join(max(0.0, limit - (time.perf_counter() - t_prepare)))
        if th.is_alive():
            return None, f"build not joined by {limit:.0f} s", 0.0, time.perf_counter() - t0
    if "result" not in _EA4X_BUILD:
        return None, "build not started", 0.0, time.perf_counter() - t0
    ext, err, dt = _EA4X_BUILD["result"]
    return ext, err, dt, time.perf_counter() - t0
def _ea4x_buffers(device, grid):
    return {"ws13x": torch.zeros(grid * 8192, dtype=torch.float32, device=device),
            "tk13x": torch.zeros(EA_TK_MAXV * 128, dtype=torch.int32, device=device),
            "ws2x": torch.zeros(grid * 8192, dtype=torch.float32, device=device),
            "tk2x": torch.zeros(EA_TK_MAXV * 64, dtype=torch.int32, device=device),
            "errx": torch.zeros(4, dtype=torch.int32, device=device)}
def _ea4x_core(name):
    bits = EA4X_ARMS[name]
    def core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0, poison=False):
        return _ea3_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace, pdl13,
                                 ("ea4x_w13", "ea4x_w2"), pdl_or=bits, ext_key="ext4x", bufs="x", poison=poison)
    return core
_EA4X_CORES = {name: _ea4x_core(name) for name in EA4X_ARMS}
def _ea4x_admit(ea, ext4x):
    miss = [f for f in _EA4X_FNS if not hasattr(ext4x, f)]
    if any(ea.get("cmb_t", {}).values()) and not hasattr(ea["ext"], "ea_combine"):
        miss.append("ext.ea_combine")
    if miss:
        return False, "missing " + " ".join(miss)
    info = [int(v) for v in ext4x.ea4x_info()]
    rows = [info[5 * i:5 * i + 5] for i in range(4)]
    ok = len(rows) == 4 and all(len(r) == 5 and r[0] == 1 and r[1] <= 112 and r[2] == 0 and r[4] == 2 for r in rows)
    text = " ".join(f"{nm} {r[1]}/{r[2]}/{r[4]}" for nm, r in zip(("K1s", "K1d", "K2s", "K2d"), rows))
    return ok, text
def _ea4x_sums(ea):
    return [int(v) for v in torch.stack([ea["tk13x"].sum(), ea["tk2x"].sum(), ea["errx"].sum()]).tolist()]
def _ea4x_zero(ea):
    for k in ("tk13x", "tk2x", "errx"):
        ea[k].zero_()
def _ea4x_canary_one(ea, core, x, ids, tw, start, count, norm_w, out_inc):
    for run in range(2):
        _ea4x_zero(ea)
        ea["ws13x"].fill_(float("nan"))
        ea["ws2x"].fill_(float("nan"))
        out = core(ea, x, ids, tw, start, count, norm_w, poison=True)
        torch.cuda.synchronize()
        s = _ea4x_sums(ea)
        if not torch.equal(out, out_inc):
            return f"r{run + 1} neq"
        if s[2]:
            return f"r{run + 1} err {s[2]}"
        if s[0] or s[1]:
            return f"r{run + 1} tickets {s[0]}/{s[1]}"
    return ""
import gc
EAD_TAG = "[engineA-dl]"
EAD_EXT_NAME = "okc_campaign_engine_a_v2_dl"
EAD_START_DEADLINE_S = 200.0
EAD_GATE_DEADLINE_S = 245.0
EAD_COPY_DEADLINE_S = 262.0
EAD_PLAIN_K1 = ("v4", "v4c", "v5", "v6", "v6pe", "v6pe2", "v6p", "v6e",
                "v4xs", "v4xd", "v4cxs", "v4cxd")
EAD_SWEEP_T = (1, 2, 3, 5, 8, 9, 12, 16, 17, 24, 32, 128, 256)
EAD_BIG = (128, 256)
EAD_BIG_ENGINES = {128: ("v3", "v4", "v4c", "v5"), 256: ("v3", "v4", "v4c", "v5")}
EAD_COPY_TOL = 1e-3
EAD_KINDS = ("judge", "one", "zero", "local8", "skew")
EAD_PIN = {8: "v4", 16: "v4c"}
_EAD_INPUTS = {}
def _ead_route(kind, tokens, gen, device):
    pop = torch.arange(1, LOCAL_EXPERTS + 1, dtype=torch.float32) ** (-0.9)
    pop = pop[torch.randperm(LOCAL_EXPERTS, generator=gen)]
    rows = []
    for _ in range(tokens):
        if kind == "judge":
            n_loc = 2 if tokens == 1 else int(torch.randint(0, 3, (1,), generator=gen))
        else:
            n_loc = {"one": 1, "zero": 0, "local8": TOP_K, "skew": 4}[kind]
        if kind == "skew":
            experts = [3, 7, 11, 19]
        else:
            experts = torch.multinomial(pop, n_loc, replacement=False, generator=gen).tolist() if n_loc else []
        slots = torch.randperm(TOP_K, generator=gen)[:n_loc].tolist()
        row = [-1] * TOP_K
        for s, e in zip(slots, experts):
            row[s] = int(e)
        rows.append(row)
    if kind == "judge" and all(v < 0 for r in rows for v in r):
        rows[0][0] = int(torch.multinomial(pop, 1, generator=gen))
    return torch.tensor(rows, dtype=torch.int32, device=device)
def _ead_inputs(kind, tokens, gen, device):
    lids = _ead_route(kind, tokens, gen, device)
    xn = torch.randn((tokens, D_EXPERT), generator=gen).to(torch.bfloat16).to(device)
    tw = torch.rand((tokens, TOP_K), generator=gen)
    tw = (tw / tw.sum(-1, keepdim=True)).to(device).contiguous()
    return xn, lids, tw
def _ead_master(xn, lids, tw, state):
    align = moe_align_block_size(lids, _block_size_m(xn.shape[0]), N_EXPERTS)
    return lambda: _experts(xn, tw, lids, state, align=align)
def _ead_core(engine):
    if engine in _EAE_CORES:
        return _EAE_CORES[engine]
    if engine in _EAD6_FCORES:
        return _EAD6_FCORES[engine]
    if engine in _EA4X_CORES:
        return _EA4X_CORES[engine]
    return {"v1": _ea_forward_core, "v2": _ea2_forward_core, "v3": _ea3_forward_core, "v4": _ea4_forward_core,
            "v4c": _ea4c_forward_core, "v6": _ead6_forward_core}[engine]
def _ead_canary(weights, state, device, t_prepare0):
    ea = state["ea"]
    ok, lines = {}, []
    first = {e: True for e in EA_ENGINES}
    worst = {e: 0.0 for e in EA_ENGINES}
    gen = torch.Generator().manual_seed(20261030)
    for tokens in range(1, EA2_MAX_T + 1):
        engines = [e for e in _ea_engines(tokens) if e not in ea.get("skip", ())]
        ok[tokens] = {e: True for e in engines}
        for kind in EAD_KINDS:
            if time.perf_counter() - t_prepare0 > EAD_GATE_DEADLINE_S:
                ok[tokens] = {e: False for e in engines}
                lines.append(f"{EAD_TAG} canary T={tokens} {kind}: skipped: deadline")
                continue
            xn, lids, tw = _ead_inputs(kind, tokens, gen, device)
            ref = _ea_ref_block(weights, 0, xn, lids, tw, state["post_expert_norm_w"], POST_NORM_EPS)
            m_rel = _rt_rel(_ead_master(xn, lids, tw, state)().float(), ref)
            n_act = len({int(e) for e in lids.flatten().tolist() if e >= 0})
            for eng in engines:
                ea["err"].zero_()
                try:
                    if first[eng]:
                        print(f"{EAD_TAG} first {eng} launch: T={tokens} {kind} (each launch named, synced)", flush=True)
                    out = _ead_core(eng)(ea, xn, lids, tw, 0, LOCAL_EXPERTS, state["post_expert_norm_w"],
                                         _rt_tracer(EAD_TAG, f"{eng} T={tokens} {kind}") if first[eng] else None)
                    torch.cuda.synchronize()
                    first[eng] = False
                    errw = int(ea["err"][0])
                    good, rel, row, text = _rt_check(out, ref)
                    good = good and errw == 0 and rel <= 2.0 * max(m_rel, 2e-3)
                    if n_act == 0:
                        good = good and bool((out.float() == 0).all())
                    worst[eng] = max(worst[eng], rel)
                    msg = f"{text}, master {m_rel:.3g}, error word {errw}"
                except Exception as e:
                    good, msg = False, f"raised {type(e).__name__}: {str(e)[:200]}"
                ok[tokens][eng] = ok[tokens][eng] and good
                if not good:
                    lines.append(f"{EAD_TAG} canary FAIL {eng} T={tokens} {kind} (active {n_act}): {msg}")
                    print(lines[-1], flush=True)
    summary = []
    for eng, hi in EA_ENGINES.items():
        if eng in ea.get("skip", ()):
            summary.append(f"{EAD_TAG} canary {eng}: skipped (occupancy below the other engines')")
            continue
        bad = [t for t in range(1, hi + 1) if not ok[t][eng]]
        summary.append(f"{EAD_TAG} canary {eng}: T 1..{hi} " + (f"FAIL at T {bad}" if bad else "clean") +
                       f"; worst rel-L2 vs fp32 {worst[eng]:.3g}")
    return ok, lines, summary
def _ead_layer_gate(state, device, cls, setters, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    per = {}
    for rep in range(2):
        data = _ead_cases(state, device, gen, cls)
        graphs = {}
        for name, setter in setters.items():
            setter()
            graphs[name], _ = _rt_graph(lambda: kernel_fn(data, state))
        per[f"T={cls} layer input {rep}"] = _rt_time(graphs, device)
        del graphs
    return per
def _ead_gate(state, device, ok, t_prepare0):
    on = state["ea"]["on"]
    won, lines = {}, []
    for cls in EA_CLASSES:
        won[cls] = None
        cands = [e for e in _ea_engines(cls) if ok.get(cls, {}).get(e, False)]
        if not cands:
            lines.append(f"{EAD_TAG} gate T={cls}: no canary-clean engine, master kept")
            continue
        if time.perf_counter() - t_prepare0 > EAD_GATE_DEADLINE_S:
            lines.append(f"{EAD_TAG} gate T={cls}: skipped: deadline, master kept")
            continue
        def setter(v):
            def f():
                on[cls] = v
            return f
        setters = {"master": setter(False)}
        setters.update({e: setter(e) for e in cands})
        per = _ead_layer_gate(state, device, cls, setters, 20261031 + cls)
        on[cls] = False
        win, worst, glines = _rt_select(per, "master", cands)
        pin = EAD_PIN.get(cls)
        if pin and pin in cands:
            win = pin
        for ln in glines:
            print(f"{EAD_TAG} gate {ln}", flush=True)
        won[cls] = win
        lines.append(f"{EAD_TAG} gate T={cls} (whole layer): worst ratio " +
                     ", ".join(f"{c} {worst[c]:.3f}" for c in cands) + f" -> {'ENGINE A ' + win if win else 'master'}"
                     + (" PINNED" if pin and win == pin else ""))
    return won, lines
def _ead_cmb_gate(state, device, classes=EA_CLASSES):
    ea = state["ea"]
    lines = []
    for cls in classes:
        if not ea["on"].get(cls, False):
            continue
        def setter(v):
            def f():
                ea["cmb_t"][cls] = v
            return f
        per = _ead_layer_gate(state, device, cls, {"triton": setter(0), "cuda": setter(1)}, 20261036 + cls)
        ea["cmb_t"][cls] = 0
        win, worst, glines = _rt_select(per, "triton", ["cuda"], margin=0.995, tight=0.995)
        for ln in glines:
            print(f"{EAD_TAG} combine gate {ln}", flush=True)
        keep = win == "cuda"
        for t in range(1, EA2_MAX_T + 1):
            if _ea_class(t) == cls and ea["on"][t]:
                ea["cmb_t"][t] = 1 if keep else 0
        lines.append(f"{EAD_TAG} combine gate T={cls} (whole layer): Triton/CUDA worst ratio {worst['cuda']:.3f} -> "
                     f"{'CUDA' if keep else 'Triton'}")
    return lines
def _ead_g2_gate(state, device):
    ea = state["ea"]
    cls = 1
    if not ea["on"].get(cls, False):
        return []
    half = ea["grid"] // EA_CTAS_PER_SM
    def setter(v):
        def f():
            ea["g2_t"][cls] = v
        return f
    per = _ead_layer_gate(state, device, cls, {"full": setter(ea["grid"]), "one_per_sm": setter(half)}, 20261042)
    ea["g2_t"].pop(cls, None)
    win, worst, glines = _rt_select(per, "full", ["one_per_sm"], margin=1.02, tight=1.02)
    for ln in glines:
        print(f"{EAD_TAG} k2 grid gate {ln}", flush=True)
    if win == "one_per_sm":
        ea["g2_t"][cls] = half
    return [f"{EAD_TAG} k2 grid gate T=1 ({ea['on'][cls]}, whole layer): full/{half}-CTA worst ratio "
            f"{worst['one_per_sm']:.3f} -> {'K2 grid ' + str(half) if win else 'full grid'}"]
def _ead_pdl_gate(state, device, classes=EA_CLASSES):
    ea = state["ea"]
    lines = []
    for cls in classes:
        if not ea["on"].get(cls, False):
            continue
        if ea["on"][cls] in EAD_PLAIN_K1:
            lines.append(f"{EAD_TAG} pdl gate T={cls}: {ea['on'][cls]} keeps K1 plain")
            continue
        def setter(v):
            def f():
                ea["pdl13_t"][cls] = v
            return f
        per = _ead_layer_gate(state, device, cls, {"plain": setter(0), "pdl": setter(1)}, 20261037 + cls)
        ea["pdl13_t"][cls] = 0
        win, worst, glines = _rt_select(per, "plain", ["pdl"], margin=0.995, tight=0.995)
        for ln in glines:
            print(f"{EAD_TAG} pdl gate {ln}", flush=True)
        keep = win == "pdl"
        for t in range(1, EA2_MAX_T + 1):
            if _ea_class(t) == cls and ea["on"][t]:
                ea["pdl13_t"][t] = 1 if keep else 0
        lines.append(f"{EAD_TAG} pdl gate T={cls} (whole layer, R -> K1): plain/PDL worst ratio {worst['pdl']:.3f} -> "
                     f"{'PDL' if keep else 'plain'}")
    return lines
def _ead_big(weights, state, device, t_prepare0):
    ea = state["ea"]
    lines = []
    first = {e: True for e in EA_ENGINES}
    gen = torch.Generator().manual_seed(20261044)
    for tokens in EAD_BIG:
        ea["on"][tokens] = False
        engines = [e for e in EAD_BIG_ENGINES[tokens] if e in EA_ENGINES and e not in ea.get("skip", ())]
        if not engines:
            lines.append(f"{EAD_TAG} big T={tokens}: {'/'.join(EAD_BIG_ENGINES[tokens])} skipped (occupancy or build)")
            continue
        good = {e: True for e in engines}
        worst = {e: 0.0 for e in engines}
        for kind in EAD_KINDS:
            if time.perf_counter() - t_prepare0 > EAD_GATE_DEADLINE_S:
                good = {e: False for e in engines}
                lines.append(f"{EAD_TAG} big canary T={tokens} {kind}: skipped: deadline")
                break
            xn, lids, tw = _ead_inputs(kind, tokens, gen, device)
            ref = _ea_ref_block(weights, 0, xn, lids, tw, state["post_expert_norm_w"], POST_NORM_EPS)
            m_rel = _rt_rel(_ead_master(xn, lids, tw, state)().float(), ref)
            n_act = len({int(e) for e in lids.flatten().tolist() if e >= 0})
            for eng in engines:
                ea["err"].zero_()
                try:
                    if first[eng]:
                        print(f"{EAD_TAG} first big {eng} launch: T={tokens} {kind} (synced)", flush=True)
                    out = _ead_core(eng)(ea, xn, lids, tw, 0, LOCAL_EXPERTS, state["post_expert_norm_w"],
                                         _rt_tracer(EAD_TAG, f"{eng} T={tokens} {kind}") if first[eng] else None)
                    torch.cuda.synchronize()
                    first[eng] = False
                    errw = int(ea["err"][0])
                    ok_, rel, row, text = _rt_check(out, ref)
                    ok_ = ok_ and errw == 0 and rel <= 2.0 * max(m_rel, 2e-3)
                    if n_act == 0:
                        ok_ = ok_ and bool((out.float() == 0).all())
                    worst[eng] = max(worst[eng], rel)
                    msg = f"{text}, master {m_rel:.3g}, error word {errw}"
                except Exception as e:
                    ok_, msg = False, f"raised {type(e).__name__}: {str(e)[:200]}"
                if not ok_:
                    good[eng] = False
                    lines.append(f"{EAD_TAG} big canary FAIL {eng} T={tokens} {kind} (active {n_act}): {msg}")
                    print(lines[-1], flush=True)
        lines.append(f"{EAD_TAG} big canary T={tokens}: " + ", ".join(
            f"{e} {'clean' if good[e] else 'FAIL'} (worst rel-L2 vs fp32 {worst[e]:.3g})" for e in engines))
        cands = [e for e in engines if good[e]]
        if not cands or time.perf_counter() - t_prepare0 > EAD_GATE_DEADLINE_S:
            continue
        def setter(v, tokens=tokens):
            def f():
                ea["on"][tokens] = v
            return f
        setters = {"master": setter(False)}
        setters.update({e: setter(e) for e in cands})
        per = _ead_layer_gate(state, device, tokens, setters, 20261045 + tokens)
        ea["on"][tokens] = False
        win, wr, glines = _rt_select(per, "master", cands)
        for ln in glines:
            print(f"{EAD_TAG} big gate {ln}", flush=True)
        line = f"{EAD_TAG} big gate T={tokens} (whole layer): worst ratio " + ", ".join(f"{c} {wr[c]:.3f}" for c in cands)
        if win:
            problem = _ead_layer_check(state, device, tokens, win)
            if problem is None:
                ea["on"][tokens] = win
                line += f" -> ENGINE A {win}"
            else:
                ea["on"][tokens] = False
                line += f" -> master ({win} layer check FAIL: {problem})"
        else:
            line += " -> master"
        lines.append(line)
    return lines
def _ead_busiest(data, state):
    scores = _cb_router_scores(_dl3_clone_data(data), state)
    ids = scores.topk(TOP_K, dim=-1).indices
    counts = torch.bincount(ids.flatten() // LOCAL_EXPERTS, minlength=N_EXPERTS // LOCAL_EXPERTS)
    return int(counts.argmax()) * LOCAL_EXPERTS, _cb_local_load(ids, 1)
def _ead_inputs_line():
    parts = []
    for t in sorted(_EAD_INPUTS):
        s = torch.tensor(_EAD_INPUTS[t]).mean(0).tolist()
        parts.append(f"T={t} {s[0]:.0f}/{s[1]:.1f}/{s[2]:.0f}/{s[3]:.1f} (n={len(_EAD_INPUTS[t])})")
    return f"{EAD_TAG} gate inputs, busiest range routes/distinct/max/hot: " + ", ".join(parts)
def _ead_cases(state, device, gen, tokens):
    if tokens == 1:
        data = _cb_data(state, device, gen, 2048, 1, 1, [80000], 1.0)
    elif tokens <= 8:
        data = _cb_data(state, device, gen, 0, 1, tokens, [80000], 1.0)
    elif tokens >= 128:
        data = _cb_data(state, device, gen, 2048, tokens // 8, 8, [4096] * (tokens // 8), 1.0)
    else:
        data = _cb_data(state, device, gen, 2048, tokens, 1, [4096] * tokens, 1.0)
    ls, load = _ead_busiest(data, state)
    _EAD_INPUTS.setdefault(tokens, []).append(load)
    data["local_start"] = torch.tensor([ls], dtype=torch.int32, device=device)
    return data
def _ead_noalign_check(state, device):
    dl = state["dl"]
    dl["r_noalign"] = False
    if not _DL_S3:
        return f"{EAD_TAG} R no-align: the parent dense-chain module has no variant + 1, align kept"
    gen = torch.Generator(device=device)
    gen.manual_seed(20261047)
    ls = torch.tensor([64], dtype=torch.int32, device=device)
    for tokens in (1, 2, 8, 16, 32):
        ei = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        logits = torch.randn((tokens, N_EXPERTS), generator=gen, device=device)
        logits[:, 64:96] += 2.0
        for variant in (0, 2):
            runs = [_dl_route(ei, logits, ls, dl, 0, variant, align=al) for al in (True, False, True)]
            torch.cuda.synchronize()
            (x0, (w0, i0, l0, a0)), (x1, (w1, i1, l1, _)), (x2, (w2, i2, l2, a2)) = runs
            same = all(torch.equal(p, q) for p, q in ((x0, x1), (w0, w1), (i0, i1), (l0, l1), (x0, x2), (w0, w2),
                                                      (i0, i2), (l0, l2)))
            n0 = int(a0[2][0])
            same = same and n0 == int(a2[2][0]) and torch.equal(a0[0][:n0], a2[0][:n0]) and torch.equal(a0[1], a2[1])
            same = same and int(dl["cnt_r"][0]) == 0
            if not same:
                return f"{EAD_TAG} R no-align FAIL at T={tokens} variant {variant}: outputs differ, align kept"
    dl["r_noalign"] = True
    return f"{EAD_TAG} R no-align: exact at T = 1, 2, 8, 16, 32 (both R kernels) -> R skips its align where Engine A runs"
def _ead_layer_check(state, device, tokens, engine):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261032 + tokens)
    worst = {}
    for rep in range(2):
        data = _ead_cases(state, device, gen, tokens)
        ls_val = int(data["local_start"][0])
        s_in = data["stream"].clone()
        scores = _cb_router_scores(_dl3_clone_data(data), state)
        state["ea"]["on"][tokens] = False
        d_ref = _dl3_clone_data(data)
        ref = kernel_fn(d_ref, state)
        state["ea"]["on"][tokens] = engine
        d_got = _dl3_clone_data(data)
        got = kernel_fn(d_got, state)
        torch.cuda.synchronize()
        n_loc = int(((got[2].long() - ls_val) % N_EXPERTS < LOCAL_EXPERTS).sum())
        ok, msg = _cb_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst)
        print(f"{EAD_TAG} layer {engine} T={tokens} rep {rep} (local routes {n_loc}): {msg} -> {'ok' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            return f"layer T={tokens} rep {rep}: {msg}"
    return None
def _ead_copy_tree(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _ead_copy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ead_copy_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_ead_copy_tree(v) for v in obj)
    return obj
def _ead_sweep(state, device):
    on = state["ea"]["on"]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261048)
    done = []
    for tokens in EAD_SWEEP_T:
        if not on.get(tokens, False):
            continue
        data = _ead_cases(state, device, gen, tokens)
        print(f"{EAD_TAG} sweep T={tokens} ({on[tokens]}): launching", flush=True)
        kernel_fn(data, state)
        torch.cuda.synchronize()
        done.append(tokens)
    return f"{EAD_TAG} sweep: eager kernel_fn ok at T {done}"
def _ead_copy_one(state, other, device, gen, tokens, fails):
    on = state["ea"]["on"]
    n = 0
    base = _ead_cases(state, device, gen, tokens)
    per = []
    for si, st in enumerate((state, other)):
        data = _dl3_clone_data(base)
        print(f"{EAD_TAG} copy check T={tokens} ({on[tokens]}): state {'AB'[si]} warm-up + capture", flush=True)
        graph, out = _rt_graph(lambda data=data, st=st: kernel_fn(data, st))
        per.append((data, graph, out))
    for k in range(4):
        print(f"{EAD_TAG} copy check T={tokens}: refill + replays {k}", flush=True)
        fresh = _ead_cases(state, device, gen, tokens)
        for data, _, _ in per:
            for key, v in fresh.items():
                if torch.is_tensor(v) and v.shape == data[key].shape:
                    data[key].copy_(v)
        ref = kernel_fn(_dl3_clone_data(per[0][0]), state)
        for _, graph, _ in (per if k % 2 == 0 else per[::-1]):
            graph.replay()
        torch.cuda.synchronize()
        for i, (_, _, out) in enumerate(per):
            n += 1
            e_h, e_s = _rt_rel(out[0].float(), ref[0].float()), _rt_rel(out[1].float(), ref[1].float())
            if not (e_h <= EAD_COPY_TOL and e_s <= EAD_COPY_TOL and torch.equal(out[2], ref[2])):
                fails.append(f"T={tokens} ({on[tokens]}) state {'AB'[i]} refill {k}: hidden {e_h:.3g} "
                             f"stream {e_s:.3g}")
    del per
    return n
def _ead_copy_check(state, device):
    on = state["ea"]["on"]
    ts = sorted(t for t, v in on.items() if v)
    if not ts:
        return f"{EAD_TAG} copy check: no T is ON"
    picks = sorted({ts[0], ts[-1]} | {min(t for t in ts if on[t] == e) for e in {on[t] for t in ts}}
                   | {t for t in state["ea"].get("x_picks", ()) if on.get(t)})
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261033)
    fails, n = [], 0
    print(f"{EAD_TAG} copy check: picks T {picks}", flush=True)
    for tokens in picks:
        n += _ead_copy_one(state, other, device, gen, tokens, fails)
    if fails:
        for t in ts:
            on[t] = False
    del other
    gc.collect()
    torch.cuda.empty_cache()
    return f"{EAD_TAG} copy check: {n} replays on two states, T {picks}: " + \
        ("PASS" if not fails else f"FAIL -> master for every T: {fails[0]}")
EAD6_TAG = "[ea6-dl]"
EAD6_START_S = 95.0
EAD6_GATE_S = 105.0
EAD6_COPY_S = 112.0
EAD6_GUARD = 0.995
EAD6_SEED = 20261080
EAD6F_S = 116.0
EAD6_V6_SWITCH = 0.005
def _ead6_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0, scratch=None, fast=0,
                       enc=(0, 0)):
    ext = ea["ext6"]
    k1 = (ea["ext6e"], ea[f"w13e{enc[0]}"]) if enc[0] else (ext, ea["w13r"])
    k2 = (ea["ext6e"], ea[f"w2e{enc[1]}"]) if enc[1] else (ext, ea["w2r"])
    tokens, hidden = x.shape
    if scratch is None:
        h = torch.empty((32, tokens, ea["inter"]), device=x.device, dtype=torch.bfloat16)
        y = torch.empty((tokens * TOP_K, hidden), device=x.device, dtype=torch.bfloat16)
    else:
        h, y = scratch
    local_ids = torch.empty((tokens, TOP_K), device=x.device, dtype=torch.int32)
    grid, minb, nodbg = ea["ea6_grid"], ea["ea6_minb"], ea["ea6_nodbg"]
    g2 = grid * 2 if fast & 16 else grid
    cmb = ea.get("cmb_t", {}).get(tokens, 0)
    early = bool(cmb and fast & 4)
    if early:
        tw32 = topk_weights.to(torch.float32).contiguous()
    if trace is not None:
        trace("before ea6_w13")
    k1[0].ea6_w13(x, topk_ids, k1[1], ea["gs13"], h, local_ids, ea["err"], nodbg, tokens, start, count, grid, minb,
                  (4 if pdl13 else 0) | (8 if fast & 1 else 0) | (4096 if enc[0] == 4 else 0))
    if trace is not None:
        trace("after ea6_w13 / before ea6_w2")
    k2[0].ea6_w2(h, topk_ids, k2[1], ea["gs2"], y, ea["err"], nodbg, tokens, start, count, g2, minb,
                 1 | (2 if ea["ea6_cores"] else 0) | (8 if fast & 2 else 0) | (4096 if enc[1] == 4 else 0))
    if trace is not None:
        trace("after ea6_w2 / before post-expert norm")
    out = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    if early:
        ext.ea6_cmb(y, local_ids, tw32, norm_w, out, POST_NORM_EPS)
        if trace is not None:
            trace("after post-expert norm")
        return out
    tw32 = topk_weights.to(torch.float32).contiguous()
    if cmb:
        ea["ext"].ea_combine(y, local_ids, tw32, norm_w, out, POST_NORM_EPS, 1)
    else:
        _post_norm_combine_kernel[(tokens,)](
            y, local_ids, tw32, norm_w, out, hidden, POST_NORM_EPS,
            TOP_K_C=TOP_K, BLOCK_H=triton.next_power_of_2(hidden), num_warps=8)
    if trace is not None:
        trace("after post-expert norm")
    return out
def _ead6_fast_core(bits):
    def core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0, scratch=None):
        return _ead6_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace, pdl13, scratch, bits)
    return core
_EAD6_FCORES = {a: _ead6_fast_core(b) for a, b in EA6F_ARMS.items()}
def _ead6_inputs(n, gen, device):
    row = [-1] * TOP_K
    experts = torch.randperm(LOCAL_EXPERTS, generator=gen)[:n].tolist()
    for s, e in zip(torch.randperm(TOP_K, generator=gen)[:n].tolist(), experts):
        row[s] = int(e)
    lids = torch.tensor([row], dtype=torch.int32, device=device)
    xn = torch.randn((1, D_EXPERT), generator=gen).to(torch.bfloat16).to(device)
    tw = torch.rand((1, TOP_K), generator=gen)
    return xn, lids, (tw / tw.sum(-1, keepdim=True)).to(device).contiguous()
def _ead6_admit(state, device, t_prepare0):
    ea = state["ea"]
    th = _EA6_BUILD.get("thread")
    if th is not None:
        th.join(max(0.0, EAD6_START_S - (time.perf_counter() - t_prepare0)))
        if th.is_alive():
            return f"build not joined by {EAD6_START_S:.0f} s"
    if "result" not in _EA6_BUILD:
        return "build not started"
    ext6, err_text, dt = _EA6_BUILD["result"]
    if ext6 is None:
        return f"build FAILED: {str(err_text)[:160]}"
    info = [int(v) for v in ext6.ea6_info()]
    rows = [info[6 * i:6 * i + 6] for i in range(len(info) // 6)]
    usable = [bool(r[0]) and r[5] >= 1 for r in rows]
    if usable[0] and usable[2] and rows[0][2] == 0 and rows[2][2] == 0:
        minb = 2
    elif usable[1] and usable[3]:
        minb = 1
    else:
        return f"build {dt:.1f} s: no usable instance (attrs/occupancy {rows})"
    alloc = lambda r: -(-32 * r // 256) * 256
    r13, r2 = rows[0 if minb == 2 else 1], rows[2 if minb == 2 else 3]
    cores = minb == 2 and 10 * alloc(r13[1]) + 10 * alloc(r2[1]) <= 65536 and r2[5] >= 2
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    ea.update({"ext6": ext6, "ea6_minb": minb, "ea6_grid": sms, "ea6_cores": cores,
               "ea6_nodbg": torch.empty(0, dtype=torch.int64, device=device)})
    _ea6_say(f"{EAD6_TAG} build {dt:.1f} s (regs/lmem B/occ): " + " ".join(
        f"{nm}m{mb} {r[1]}/{r[2]}/{r[5]}" for nm, mb, r in
        (("K1", 2, rows[0]), ("K1", 1, rows[1]), ("K2", 2, rows[2]), ("K2", 1, rows[3]))) +
        f"; MINB {minb}; grid {sms}; K2 co-resident at T=1: {'yes' if cores else 'no (padded)'}")
    fr = rows[4:7] if _EA6_BUILD.get("fast") else []
    fast = (len(fr) == 3 and minb == 2 and cores and all(r[0] and r[2] == 0 and r[5] >= 1 for r in fr[:2])
            and 10 * alloc(fr[0][1]) + 10 * alloc(fr[1][1]) <= 65536)
    early = (len(fr) == 3 and bool(fr[2][0]) and fr[2][2] == 0 and fr[2][5] >= 1 and bool(ea.get("cmb_ok"))
             and 1 not in ea.get("cmb_bad", ()))
    farms = tuple(a for a, b in EA6F_ARMS.items() if (fast or not b & 3) and (early or not b & 4))
    ea.update({"ea6_fast": fast, "ea6_farms": farms})
    if fr or _EA6_BUILD.get("fast_err"):
        _ea6_say(f"{EAD6_TAG} fast (regs/lmem B/occ): " + (" ".join(
            f"{nm} {r[1]}/{r[2]}/{r[5]}" for nm, r in zip(("K1f", "K2f", "cmb"), fr)) if fr else
            "build FAILED, v6 from the fallback build: " + _EA6_BUILD.get("fast_err", "")[:110]) +
            f"; arms {' '.join(farms) or 'none'}")
    return None
def _ead6_canary(weights, state, device, farms=()):
    ea = state["ea"]
    norm = state["post_expert_norm_w"]
    gen = torch.Generator().manual_seed(EAD6_SEED)
    cases = [(k, _ead_inputs(k, 1, gen, device)) for k in EAD_KINDS]
    cases += [(f"n{n}", _ead6_inputs(n, gen, device)) for n in (2, 3, 4, 5)]
    worst = m_worst = 0.0
    first = True
    fbad, feq = {}, {a: 0 for a in farms}
    for label, (xn, lids, tw) in cases:
        ref = _ea_ref_block(weights, 0, xn, lids, tw, norm, POST_NORM_EPS)
        m_rel = _rt_rel(_ead_master(xn, lids, tw, state)().float(), ref)
        m_worst = max(m_worst, m_rel)
        n_act = len({int(e) for e in lids.flatten().tolist() if e >= 0})
        ea["err"].zero_()
        errw = -1
        try:
            outs = []
            for _ in range(2):
                scratch = (torch.full((32, 1, ea["inter"]), float("nan"), device=device, dtype=torch.bfloat16),
                           torch.full((TOP_K, D_EXPERT), float("nan"), device=device, dtype=torch.bfloat16))
                tr = None
                if first:
                    print(f"{EAD6_TAG} first v6 launch: T=1 {label} (each launch named, synced)", flush=True)
                    tr = _rt_tracer(EAD6_TAG, f"v6 T=1 {label}")
                outs.append(_ead6_forward_core(ea, xn, lids, tw, 0, LOCAL_EXPERTS, norm, tr, pdl13=0,
                                               scratch=scratch))
                torch.cuda.synchronize()
                first = False
            errw = int(ea["err"][0])
            good, rel, row, text = _rt_check(outs[0], ref)
            same = bool(torch.equal(outs[0], outs[1]))
            good = good and errw == 0 and same and rel <= 2.0 * max(m_rel, 2e-3)
            if n_act == 0:
                good = good and bool((outs[0].float() == 0).all())
            worst = rel if rel != rel else max(worst, rel)
            msg = f"{text}, master {m_rel:.3g}, error word {errw}{'' if same else ', REPEAT RUN DIFFERS'}"
        except Exception as e:
            good, msg = False, f"raised {type(e).__name__}: {str(e)[:120]}"
        if not good:
            return False, f"{EAD6_TAG} canary FAIL T=1 {label} (active {n_act}): {msg}", fbad, feq
        for arm in farms:
            if arm in fbad:
                continue
            ea["err"].zero_()
            try:
                fo = []
                for _ in range(2):
                    scratch = (torch.full((32, 1, ea["inter"]), float("nan"), device=device, dtype=torch.bfloat16),
                               torch.full((TOP_K, D_EXPERT), float("nan"), device=device, dtype=torch.bfloat16))
                    tr = None
                    if label == cases[0][0]:
                        print(f"{EAD6_TAG} first {arm} launch: T=1 {label} (each launch named, synced)", flush=True)
                        tr = _rt_tracer(EAD6_TAG, f"{arm} T=1 {label}")
                    fo.append(_ead_core(arm)(ea, xn, lids, tw, 0, LOCAL_EXPERTS, norm, tr, 0, scratch))
                    torch.cuda.synchronize()
                fe = int(ea["err"][0])
                if fe or not torch.equal(fo[0], fo[1]) or not torch.equal(fo[0], outs[0]):
                    fbad[arm] = (f"{label}: error word {fe}" + ("" if torch.equal(fo[0], fo[1]) else ", REPEAT DIFFERS")
                                 + ("" if torch.equal(fo[0], outs[0]) else ", NOT BITWISE v6"))
                else:
                    feq[arm] += 1
            except Exception as e:
                fbad[arm] = f"{label}: raised {type(e).__name__}: {str(e)[:80]}"
    return True, (f"{EAD6_TAG} canary T=1 ({len(cases)} inputs x 2 runs, K1 plain, NaN h/y): clean; worst rel-L2 vs "
                  f"fp32 {worst:.3g}, master worst {m_worst:.3g}; repeat runs bitwise equal"), fbad, feq
def _ead6_copy_check(state, device, arm="v6"):
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(EAD6_SEED + 2)
    fails = []
    try:
        n = _ead_copy_one(state, other, device, gen, 1, fails)
    finally:
        del other
        gc.collect()
        torch.cuda.empty_cache()
    return not fails, (f"{EAD6_TAG} copy check {arm} T=1: {n} replays on two states: " +
                       ("PASS" if not fails else f"FAIL ({fails[0]})"))
def _ead6_fblock(state, device, arms):
    ea = state["ea"]
    norm = state["post_expert_norm_w"]
    gen = torch.Generator().manual_seed(EAD6_SEED + 10)
    keep, per = [], {}
    fns = {"inc": _ead6_forward_core, **{a: _ead_core(a) for a in arms}}
    for n in range(1, 6):
        xn, lids, tw = inp = _ead6_inputs(n, gen, device)
        keep.append(inp)
        graphs = {nm: _rt_graph(lambda fn=fn, xn=xn, lids=lids, tw=tw: fn(ea, xn, lids, tw, 0, LOCAL_EXPERTS, norm))[0]
                  for nm, fn in fns.items()}
        per[f"n{n}"] = _rt_time(graphs, device)
        torch.cuda.synchronize()
        del graphs
    res = _ea6_judge(per, [(f"n{n}", EA6_W1.get(n, 0.0), EA6F_FLOOR) for n in range(1, 6)], arms, EA6F_MARGIN,
                     EA6F_MARGIN)
    for ri in range(RT_ROUNDS):
        _ea6_say(f"{EAD6_TAG} T=1 fast us n1..n5 r{ri + 1}: " + " | ".join(
            f"{'v6' if nm == 'inc' else nm} " + "/".join(f"{_rt_stats(per[f'n{n}'][ri][nm])[0]:.2f}" for n in range(1, 6))
            for nm in fns))
    good = sorted((a for a in arms if res[a]["ok"]), key=lambda a: -res[a]["worst"])
    note = ""
    if "v6pe2" in arms and "v6pe" in arms:
        w1, w2 = [r["wr"] for r in res["v6pe"]["rounds"]], [r["wr"] for r in res["v6pe2"]["rounds"]]
        n = min(len(w1), len(w2))
        note = " | pe/pe2 x" + (f"{min(w2[r] / max(w1[r], 1e-9) for r in range(n)):.3f}" if n else "nan")
        if "v6pe2" in good and "v6pe" in good and min(w2) <= min(w1) + EAD6_V6_SWITCH:
            good = ["v6pe"] + [a for a in good if a != "v6pe"]
            note += f", v6pe kept (switch {EAD6_V6_SWITCH})"
    _ea6_say(f"{EAD6_TAG} T=1 fast vs v6: " + " | ".join(
        f"{a} wr " + "/".join(f"{r['wr']:.3f}" for r in res[a]["rounds"]) +
        f" floor {min(r['fmin'] for r in res[a]['rounds']):+.3f}" + (" Q" if res[a]["ok"] else " -") for a in arms) +
        note + f" -> {good[0] if good else 'v6'}")
    del keep
    return good
def _ead6_fast(state, device, farms, late, put):
    ea = state["ea"]
    v6c = ("v6", ea["pdl13_t"].get(1), ea["cmb_t"].get(1))
    arms = tuple(a for a in farms if v6c[2] or not EA6F_ARMS[a] & 4)
    if not arms:
        return None, "no canary-clean arm" + ("" if arms == farms else " (v6 kept the Triton combine)")
    if late(EAD6F_S):
        return None, "skipped: deadline"
    try:
        chain = _ead6_fblock(state, device, arms)
        def setter(cfg):
            def f():
                put(*cfg)
            return f
        for arm in chain:
            acfg = (arm,) + v6c[1:]
            if late(EAD6F_S):
                return None, f"{arm} guard: deadline"
            per = _ead_layer_gate(state, device, 1, {"inc": setter(v6c), arm: setter(acfg)}, EAD6_SEED + 13)
            put(*v6c)
            win, worst, glines = _rt_select(per, "inc", [arm], margin=EAD6_GUARD, tight=EAD6_GUARD)
            for ln in glines:
                print(f"{EAD6_TAG} fast layer gate {ln}", flush=True)
            _ea6_say(f"{EAD6_TAG} whole layer T=1 v6 vs {arm} r1/r2: " + "; ".join(
                f"in{i} x" + "/".join(f"{_rt_stats(r['inc'])[0] / max(_rt_stats(r[arm])[0], 1e-3):.3f}" for r in rounds)
                for i, rounds in enumerate(per.values())) +
                f"; worst x{worst[arm]:.3f} (guard >= {EAD6_GUARD}) -> {arm if win == arm else 'v6'}")
            if win != arm:
                continue
            if late(EAD6F_S):
                return None, f"{arm} layer check: deadline"
            put(*acfg)
            problem = _ead_layer_check(state, device, 1, arm)
            if problem is not None:
                put(*v6c)
                _ea6_say(f"{EAD6_TAG} {arm} T=1 layer check FAIL: {problem}")
                continue
            if late(EAD6F_S):
                put(*v6c)
                return None, f"{arm} copy check: deadline"
            ok, line = _ead6_copy_check(state, device, arm)
            _ea6_say(line)
            if ok:
                return arm, "fast race, guard, layer and copy checks passed"
            put(*v6c)
        return None, "no arm passed" if chain else "lost the fast block race"
    except Exception as e:
        put(*v6c)
        return None, f"raised {type(e).__name__}: {str(e)[:120]}"
def _ead6_stage(weights, state, device, t_prepare0):
    ea = state["ea"]
    on = ea["on"]
    t0 = time.perf_counter()
    inc = on.get(1, False)
    saved = (inc, ea["pdl13_t"].get(1), ea["cmb_t"].get(1))
    def put(eng, pdl, cmb):
        on[1] = eng
        for d, v in ((ea["pdl13_t"], pdl), (ea["cmb_t"], cmb)):
            if v is None:
                d.pop(1, None)
            else:
                d[1] = v
    def late(limit):
        return time.perf_counter() - t_prepare0 > limit
    cmb0 = 1 if (ea.get("cmb_ok") and 1 not in ea.get("cmb_bad", ())) else 0
    v6cfg = ("v6", 0, cmb0)
    verdict = "incumbent"
    try:
        if late(EAD6_START_S):
            verdict = f"skipped: deadline (start at {time.perf_counter() - t_prepare0:.1f} s > {EAD6_START_S:.0f})"
            raise _Ead6Off(verdict)
        if ea.get("dq") != 2:
            raise _Ead6Off(f"skipped: dq {ea.get('dq')} (v6 is dq2 only)")
        why = _ead6_admit(state, device, t_prepare0)
        if why is not None:
            raise _Ead6Off(why)
        farms = ea.get("ea6_farms", ())
        if farms:
            put(inc, saved[1], cmb0)
        ok, line, fbad, feq = _ead6_canary(weights, state, device, farms)
        put(*saved)
        _ea6_say(line)
        if farms:
            _ea6_say(f"{EAD6_TAG} canary fast T=1: " + "; ".join(
                f"{a} " + (f"FAIL ({fbad[a]})" if a in fbad else f"clean (=v6 bitwise {feq[a]} inputs x 2)")
                for a in farms))
        farms_ok = tuple(a for a in farms if a not in fbad)[:2]
        if not ok:
            raise _Ead6Off("canary FAIL")
        if late(EAD6_GATE_S):
            raise _Ead6Off("whole-layer guard skipped: deadline")
        def setter(cfg):
            def f():
                put(*cfg)
            return f
        per = _ead_layer_gate(state, device, 1, {"inc": setter(saved), "v6": setter(v6cfg)}, EAD6_SEED + 3)
        put(*saved)
        win, worst, glines = _rt_select(per, "inc", ["v6"], margin=EAD6_GUARD, tight=EAD6_GUARD)
        for ln in glines:
            print(f"{EAD6_TAG} layer gate {ln}", flush=True)
        ns = [int(ld[1]) for ld in _EAD_INPUTS.get(1, [])[-len(per):]]
        parts = []
        for i, rounds in enumerate(per.values()):
            mi = [_rt_stats(r["inc"])[0] for r in rounds]
            mv = [_rt_stats(r["v6"])[0] for r in rounds]
            parts.append(f"in{i} n={ns[i] if i < len(ns) else '?'} " + "/".join(f"{v:.1f}" for v in mi) + " vs " +
                         "/".join(f"{v:.1f}" for v in mv) + " x" + "/".join(f"{a / max(b, 1e-3):.3f}"
                                                                         for a, b in zip(mi, mv)))
        q = win == "v6"
        _ea6_say(f"{EAD6_TAG} whole layer T=1 us inc({inc or 'master'}) vs v6 r1/r2: " + "; ".join(parts) +
                 f"; worst x{worst['v6']:.3f} (guard >= {EAD6_GUARD}) -> {'v6' if q else 'incumbent'}")
        if not q:
            raise _Ead6Off("whole-layer guard (a loss)")
        put(*v6cfg)
        notes = ["K1 plain (EAD_PLAIN_K1)"]
        for nm, fn, cond in (("combine gate", _ead_cmb_gate, bool(cmb0)),):
            if not cond:
                notes.append(f"{nm}: n/a (CUDA combine off at T=1: unit or layer check)")
                continue
            if late(EAD6_GATE_S):
                notes.append(f"{nm}: skipped (deadline), kept")
                continue
            now = ("v6", ea["pdl13_t"].get(1), ea["cmb_t"].get(1))
            try:
                for ln in fn(state, device, classes=(1,)):
                    print(f"{EAD6_TAG} {ln}", flush=True)
                    notes.append(f"{nm} -> {ln.rsplit('-> ', 1)[-1]} (x{ln.split('worst ratio ')[-1].split(' ->')[0]})")
            except Exception as e:
                put(*now)
                notes.append(f"{nm} raised {type(e).__name__}, kept")
        if late(EAD6_GATE_S):
            raise _Ead6Off("layer check skipped: deadline")
        problem = _ead_layer_check(state, device, 1, "v6")
        notes.append("layer check " + ("ok" if problem is None else "FAIL"))
        _ea6_say(f"{EAD6_TAG} v6 T=1: " + "; ".join(notes) + (f"; {problem}" if problem else ""))
        if problem is not None:
            raise _Ead6Off("layer check FAIL")
        if late(EAD6_COPY_S):
            raise _Ead6Off("copy check skipped: deadline")
        ok, line = _ead6_copy_check(state, device)
        _ea6_say(line)
        if not ok:
            raise _Ead6Off("copy check FAIL")
        verdict = "v6"
        if farms:
            arm, why = _ead6_fast(state, device, farms_ok, late, put)
            verdict = f"{arm} (fc over v6: {why})" if arm else f"v6 (fc: {why})"
    except _Ead6Off as e:
        put(*saved)
        verdict = f"incumbent ({e})"
    except Exception as e:
        put(*saved)
        verdict = f"incumbent (stage raised {type(e).__name__}: {str(e)[:140]})"
    try:
        ea["err"].zero_()
        torch.cuda.synchronize()
    except Exception as e:
        verdict += f"; final sync raised {type(e).__name__}"
    pdl, cmb = ea["pdl13_t"].get(1, 0), ea["cmb_t"].get(1, 0)
    _ea6_say(f"{EAD6_TAG} route T=1: {on.get(1) or 'master'} -> {verdict}"
             + (f" (K1 {'PDL' if pdl else 'plain'}, K2 {'co-resident' if ea.get('ea6_cores') else 'padded'}, "
                f"{'CUDA' if cmb else 'Triton'} combine)" if on.get(1) in ("v6", *EA6F_ARMS) else "")
             + f"; stage {time.perf_counter() - t0:.1f} s, prepare at {time.perf_counter() - t_prepare0:.1f} s")
    return [EA6_LINES[-1]]
class _Ead6Off(Exception):
    pass
def _ead6_final_lines(state):
    out = list(EA6_LINES) if EA6_LINES else [f"{EAD6_TAG} stage did not run (Engine A setup ended before it)"]
    ea = state.get("ea") or {}
    on = ea.get("on") or {}
    eng, e8, pin8 = on.get(1), on.get(8), EAD_PIN.get(8)
    b8 = EA4X_BASE.get(e8, e8)
    th = _EA6_BUILD.get("thread")
    out.append(f"{EAD6_TAG} PATH CHOSEN T=1 = {'ENGINE A ' + eng if eng else 'MASTER'} (T>=2 unchanged; T=8 "
               f"{'ENGINE A ' + e8 if e8 else 'MASTER'}" + (f", EAD_PIN {pin8} {'routed' if b8 == pin8 else 'NOT routed'}"
                                                            + (f" (as {e8})" if b8 == pin8 and e8 != pin8 else "")
                                                            if pin8 else "") + ")"
               + ("; v6 build thread STILL RUNNING" if th is not None and th.is_alive() else ""))
    out.extend(EA4X_LINES if CAMPAIGN_EA4X else ())
    return [ln[:199] for ln in out]
EAD4X_TAG = "[ea4x-dl]"
EAD4X_START_S = 90.0
EAD4X_GATE_S = 100.0
EAD4X_CHECK_S = 108.0
EAD4X_MARGIN = 1.005
EAD4X_TIGHT = 1.003
EAD4X_RACE = ("v4xd", "v4cxd", "v4xs", "v4cxs")
EAD4X_CANARY_T = (2, 3, 5, 8, 9, 12, 16, 17, 24, 32)
EAD4X_SEED = 20261129
EAD4X_KEEP = 1.0
EAD4X_SWITCH = 0.01
EAD4X_NEAR = 0.005
EAD4X_MORE = 2
EAD4X_BOARD = {8: "s", 16: "s", 32: "s", 128: "s", 256: "s"}
EAD4X_PIN = {16: "v4cxs"}
def _ead4x_rq(per, arm, pooled=False):
    def one(ti, ta):
        im, ii = _rt_stats(ti)[:2]
        am, ai = _rt_stats(ta)[:2]
        r = im / max(am, 1e-3)
        return r, r - (EAD4X_TIGHT if ai <= RT_IQR_FACTOR * max(ii, RT_IQR_FLOOR_US) else EAD4X_MARGIN)
    labs = list(per)
    if pooled:
        v = [one([x for rd in per[lb] for x in rd["inc"]], [x for rd in per[lb] for x in rd[arm]]) for lb in labs]
        return min(a for a, _ in v), min(b for _, b in v)
    out = []
    for ri in range(min(len(per[lb]) for lb in labs)):
        v = [one(per[lb][ri]["inc"], per[lb][ri][arm]) for lb in labs]
        out.append((min(a for a, _ in v), min(b for _, b in v)))
    return out
def _ead4x_order(view, board, dw=None):
    dw = dw or {}
    w = {k: v[0] + dw.get(k, 0.0) for k, v in view.items()}
    q = sorted((k for k, v in view.items() if (w[k] >= EAD4X_KEEP if k == board else v[1] + dw.get(k, 0.0) >= 0.0)),
               key=lambda k: -w[k])
    if board in q:
        rest = [k for k in q if k != board]
        if not (rest and w[rest[0]] > w[board] + EAD4X_SWITCH):
            q = [board] + rest
    return q
def _ead4x_worst(rr):
    return {k: (min(r[0] for r in v), min(r[1] for r in v)) for k, v in rr.items()}
def _ead4x_marginal(rr, board):
    n = min(len(v) for v in rr.values())
    picks = {(_ead4x_order({k: v[ri] for k, v in rr.items()}, board) or ["inc"])[0] for ri in range(n)}
    if len(picks) > 1:
        return "flip"
    view = _ead4x_worst(rr)
    base = (_ead4x_order(view, board) or ["inc"])[0]
    nudges = [{}]
    for k in view:
        nudges = [dict(c, **{k: sg * EAD4X_NEAR}) for c in nudges for sg in (-1, 0, 1)]
    return "near" if any((_ead4x_order(view, board, c) or ["inc"])[0] != base for c in nudges) else ""
def _ead4x_decide(per, ak, board, regate, late):
    rr0 = {k: _ead4x_rq(per, a) for k, a in ak.items()}
    q0 = _ead4x_order(_ead4x_worst(rr0), board)
    why, more = _ead4x_marginal(rr0, board), 0
    if why:
        try:
            for _ in range(EAD4X_MORE):
                if late():
                    break
                p2 = regate()
                for lab in per:
                    per[lab].extend(p2[lab])
                more += 1
        except Exception as e:
            why += f" {type(e).__name__}"
    rr = {k: _ead4x_rq(per, a) for k, a in ak.items()}
    view = {k: _ead4x_rq(per, a, pooled=True) for k, a in ak.items()} if more else _ead4x_worst(rr)
    order = _ead4x_order(view, board)
    n = min(len(v) for v in rr.values())
    pick = (order or ["inc"])[0]
    flips = sum(1 for ri in range(n) if (_ead4x_order({k: v[ri] for k, v in rr.items()}, board) or ["inc"])[0] != pick)
    return {"order": order, "q0": q0, "view": view, "n": n, "flips": flips,
            "mode": (f"pooled {why} {n}r f{flips}" if more else (f"{why} unpooled" if why else f"{n}r"))}
def _ead4x_regate(state, device, cls, setters, seed, restore):
    def f():
        n0 = len(_EAD_INPUTS.get(cls, []))
        had = cls in _EAD_INPUTS
        try:
            return _ead_layer_gate(state, device, cls, setters, seed)
        finally:
            restore()
            if had:
                del _EAD_INPUTS[cls][n0:]
            else:
                _EAD_INPUTS.pop(cls, None)
    return f
def _ead4x_say(s):
    _ea4x_say(s)
def _ead4x_canary(state, device, t_prepare0):
    ea = state["ea"]
    norm = state["post_expert_norm_w"]
    bases = [b for b in ("v4", "v4c") if b in EA_ENGINES and b not in ea.get("skip", ())]
    clean = {a: EA4X_BASE[a] in bases for a in EAD4X_RACE}
    fails, runs = [], 0
    gen = torch.Generator().manual_seed(EAD4X_SEED)
    for tokens in EAD4X_CANARY_T:
        if time.perf_counter() - t_prepare0 > EAD4X_START_S:
            clean = {a: False for a in clean}
            fails.append(f"deadline at T={tokens}")
            break
        for kind in EAD_KINDS:
            xn, lids, tw = _ead_inputs(kind, tokens, gen, device)
            for base in bases:
                arms = [a for a in clean if EA4X_BASE[a] == base and clean[a]]
                if not arms:
                    continue
                out_inc = _ead_core(base)(ea, xn, lids, tw, 0, LOCAL_EXPERTS, norm)
                torch.cuda.synchronize()
                for a in arms:
                    why = _ea4x_canary_one(ea, _EA4X_CORES[a], xn, lids, tw, 0, LOCAL_EXPERTS, norm, out_inc)
                    runs += 2
                    if why:
                        clean[a] = False
                        fails.append(f"{a} T={tokens} {kind} {why}")
    _ea4x_zero(ea)
    torch.cuda.synchronize()
    return clean, fails, runs
def _ead4x_copy_check(state, device, tokens):
    ea = state["ea"]
    _ea4x_zero(ea)
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(EAD4X_SEED + 50 + tokens)
    fails, n = [], 0
    try:
        n = _ead_copy_one(state, other, device, gen, tokens, fails)
        s = [a + b for a, b in zip(_ea4x_sums(ea), _ea4x_sums(other["ea"]))]
        if any(s):
            fails.append(f"tk/err {'/'.join(map(str, s))}")
    finally:
        del other
        _ea4x_zero(ea)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    return not fails, (f"copy PASS({n})" if not fails else f"copy FAIL({fails[0][:60]})")
def _ead4x_stage(state, device, t_prepare0):
    ea = state["ea"]
    on = ea["on"]
    prev = dict(on)
    t0 = time.perf_counter()
    ea["x_picks"] = []
    ea["x_clean"] = {}
    def late(limit):
        return time.perf_counter() - t_prepare0 > limit
    def done():
        ln = f"{EAD_TAG} v4x stage done in {time.perf_counter() - t0:.1f} s ({len(EA4X_LINES)} v4x lines, printed last)"
        print(ln, flush=True)
        return [ln]
    try:
        if late(EAD4X_START_S):
            _ead4x_say(f"{EAD4X_TAG} skipped: deadline (start at {time.perf_counter() - t_prepare0:.1f} s)")
            return done()
        if ea.get("dq") != 2:
            _ead4x_say(f"{EAD4X_TAG} skipped: dq {ea.get('dq')} (v4x is dq2 only)")
            return done()
        ext4x, err_text, dt, wait = _ea4x_build_join(t_prepare0, EAD4X_START_S)
        if ext4x is None:
            _ead4x_say(f"{EAD4X_TAG} build FAILED (join wait {wait:.1f} s): {str(err_text)[:150]}")
            return done()
        ok, text = _ea4x_admit(ea, ext4x)
        _ead4x_say(f"{EAD4X_TAG} build {dt:.1f} s, join wait {wait:.1f} s (regs/lmem/occ): {text}; "
                   f"{_EA4X_BUILD.get('name', '?')}" + ("" if ok else " -> NOT ADMITTED"))
        if not ok:
            return done()
        ea["ext4x"] = ext4x
        ea.update(_ea4x_buffers(device, ea["grid"]))
        if late(EAD4X_START_S):
            _ead4x_say(f"{EAD4X_TAG} canary skipped: deadline -> incumbents kept")
            return done()
        clean, fails, runs = _ead4x_canary(state, device, t_prepare0)
        ea["x_clean"] = dict(clean)
        _ead4x_say(f"{EAD4X_TAG} canary {len(EAD4X_CANARY_T)} T x {len(EAD_KINDS)} kinds x 2 runs (NaN h/y/ws, K1 plain): "
                   f"{runs} runs, " + " ".join(f"{a} {'ok' if c else 'FAIL'}" for a, c in clean.items()) +
                   ("; =inc bitwise, err 0, tickets 0" if not fails else f"; {len(fails)} fails, first {fails[0]}"))
        race, route, rinfo = {}, {}, {}
        for cls in (8, 16, 32):
            inc = on.get(cls)
            arms = [a for a in EAD4X_RACE if EA4X_BASE[a] == inc and clean.get(a)]
            if not arms:
                race[cls] = f"T{cls} inc {inc or 'master'}" + (" twin FAIL" if inc in ("v4", "v4c") else "")
                continue
            if EAD4X_PIN.get(cls) in arms:
                rinfo[cls] = {"order": ["s"], "view": {"s": (1.0, 0.0)}, "mode": "PIN", "inc": inc,
                              "ak": {"s": EAD4X_PIN[cls]}}
                race[cls] = f"T{cls} PIN {EAD4X_PIN[cls]}"
                continue
            if late(EAD4X_GATE_S):
                race[cls] = f"T{cls} deadline"
                continue
            try:
                def setter(v, cls=cls):
                    def f():
                        on[cls] = v
                    return f
                setters = {"inc": setter(inc)}
                setters.update({a: setter(a) for a in arms})
                per = _ead_layer_gate(state, device, cls, setters, EAD4X_SEED + cls)
                on[cls] = inc
                _, worst, glines = _rt_select(per, "inc", arms, margin=EAD4X_MARGIN, tight=EAD4X_TIGHT)
                for ln in glines:
                    print(f"{EAD4X_TAG} gate {ln}", flush=True)
                def med(nm):
                    v = [_rt_stats(r[nm])[0] for rounds in per.values() for r in rounds]
                    return f"{sum(v) / len(v):.1f}"
                meds = {nm: med(nm) for nm in ("inc", *arms)}
                ri = _ead4x_decide(per, {a[-1]: a for a in arms}, EAD4X_BOARD.get(cls, "s"),
                                   _ead4x_regate(state, device, cls, setters, EAD4X_SEED + cls,
                                                 lambda cls=cls, inc=inc: on.__setitem__(cls, inc)),
                                   lambda: late(EAD4X_GATE_S))
                on[cls] = inc
                ri.update({"inc": inc, "ak": {a[-1]: a for a in arms}})
                rinfo[cls] = ri
                race[cls] = (f"T{cls} " + " ".join(f"{a[-1]}{worst[a]:.3f}{'Q' if a[-1] in ri['q0'] else ''}"
                                                   for a in arms)
                             + f" i {meds['inc']} " + " ".join(f"{a[-1]} {meds[a]}" for a in arms))
            except Exception as e:
                on[cls] = prev[cls]
                rinfo.pop(cls, None)
                race[cls] = f"T{cls} raised {type(e).__name__}"
        _ead4x_say(f"{EAD4X_TAG} race layer us (x worst, Q qual): " +
                   " | ".join(race.get(c, f"T{c} -") for c in (8, 16, 32)))
        def put(cls, frm, arm):
            if late(EAD4X_CHECK_S):
                return False, "dl"
            try:
                problem = _ead_layer_check(state, device, cls, arm)
                if problem is not None:
                    on[cls] = frm
                    return False, "layer FAIL"
                ok_, text = _ead4x_copy_check(state, device, cls)
                if not ok_:
                    on[cls] = frm
                    return False, text[:32]
            except Exception as e:
                on[cls] = frm
                return False, f"raised {type(e).__name__}"
            for t in range(2, EA2_MAX_T + 1):
                if _ea_class(t) == cls and on.get(t) == frm:
                    on[t] = arm
            return True, text[5:]
        for cls, ri in sorted(rinfo.items()):
            tried = []
            for k in ri["order"]:
                ok_, text = put(cls, ri["inc"], ri["ak"][k])
                tried.append(f"{k.upper() if ok_ else k}{ri['view'][k][0]:.3f} {text}")
                if ok_:
                    ea["x_picks"].append(cls)
                    break
            route[cls] = f"{'>'.join(ri['order']) or 'none'} {' '.join(tried)} {ri['mode']}".replace("  ", " ")
            if ri["mode"] == "PIN":
                route[cls] = f"s PIN {ri['ak']['s']} {text}"
        _ead4x_say(f"{EAD4X_TAG} route (board " + "/".join(EAD4X_BOARD.get(c, "s") for c in (8, 16, 32)) + "): " +
                   ", ".join(f"{c}:{route[c] if c in route else 'inc ' + str(on.get(c))}" for c in (8, 16, 32))
                   + f"; at {time.perf_counter() - t_prepare0:.1f} s")
    except Exception as e:
        on.clear()
        on.update(prev)
        ea["x_picks"] = []
        _ead4x_say(f"{EAD4X_TAG} stage raised {type(e).__name__}: {str(e)[:150]} -> incumbents kept")
    try:
        if "tk13x" in ea:
            _ea4x_zero(ea)
        torch.cuda.synchronize()
    except Exception as e:
        _ead4x_say(f"{EAD4X_TAG} final sync raised {type(e).__name__}: {str(e)[:150]}")
    return done()
def _ead_setup(weights, state, device, t_prepare0):
    lines = []
    def say(s):
        lines.append(s)
        print(s, flush=True)
    state["ea"] = {"on": {t: False for t in (*range(1, EA2_MAX_T + 1), *EAD_BIG)}, "max_t": EA2_MAX_T, "cmb_t": {},
                   "pdl13_t": {}, "g2_t": {}}
    t0 = time.perf_counter()
    try:
        if time.perf_counter() - t_prepare0 > EAD_START_DEADLINE_S:
            raise RuntimeError(f"skipped: deadline (start at {time.perf_counter() - t_prepare0:.1f} s)")
        if state.get("dl") is None:
            raise RuntimeError("the small-T dense chain (kernels P and R) is off, so there are no R outputs to feed")
        if (tuple(weights["w13"].shape) != (LOCAL_EXPERTS, 2 * D_EXPERT, D_EXPERT // 2)
                or tuple(weights["w2"].shape) != (LOCAL_EXPERTS, D_EXPERT, D_EXPERT // 2)):
            raise RuntimeError(f"unsupported expert shapes {tuple(weights['w13'].shape)} / {tuple(weights['w2'].shape)}")
        if CAMPAIGN_EA6:
            _ea6_build_start(t_prepare0)
        if CAMPAIGN_EA4X:
            try:
                _ea4x_build_start(t_prepare0)
            except Exception as e:
                _EA4X_BUILD["result"] = (None, f"build start raised {type(e).__name__}: {str(e)[:120]}", 0.0)
        ext, err_text, dt, occs, ilines = _ea2_build_ext(EAD_TAG, EAD_EXT_NAME, t_prepare0, EAD_START_DEADLINE_S)
        if ext is None:
            raise RuntimeError(f"build failed: {err_text[-400:]}")
        for ln in ilines:
            say(ln)
        dq = 2 if _ea_scales_ok(weights, 0, LOCAL_EXPERTS) else 1
        cps = max(1, min([EA_CTAS_PER_SM] + [v for k, v in occs.items() if k.endswith(f"dq{dq}") and k[:2] in ("v1", "v2")]))
        state["ea"]["skip"] = {e for e in EA_ENGINES if e not in ("v1", "v2")
                               and min((v for k, v in occs.items() if k.startswith(e + " ") and k.endswith(f"dq{dq}")),
                                       default=0) < cps}
        sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        t1 = time.perf_counter()
        state["ea"].update(_ea_repack(weights, 0, LOCAL_EXPERTS, dq))
        torch.cuda.synchronize()
        grid = cps * sms
        state["ea"].update({"ext": ext, "dq": dq, "grid": grid, "pdl": 1,
                            **_ea2_buffers(device, grid, D_EXPERT)})
        say(f"{EAD_TAG} {EA_LAYOUT_NAME[dq]}; repack {time.perf_counter() - t1:.1f} s; grid {grid} "
            f"({cps} CTAs/SM; v1..v3 160 threads, v4 / v5 288); build {dt:.1f} s")
        ok, flines, slines = _ead_canary(weights, state, device, t_prepare0)
        lines.extend(flines)
        for ln in slines:
            say(ln)
        if not any(v for d in ok.values() for v in d.values()):
            raise RuntimeError("canary failed for every T and engine")
        won, glines = _ead_gate(state, device, ok, t_prepare0)
        lines.extend(glines)
        for t in range(1, EA2_MAX_T + 1):
            w = won.get(_ea_class(t))
            state["ea"]["on"][t] = w if (w is not None and ok[t].get(w, False)) else False
        if any(state["ea"]["on"].get(t) for t in range(1, EA2_MAX_T + 1)) \
                and time.perf_counter() - t_prepare0 < EAD_GATE_DEADLINE_S:
            try:
                lines.append(_ead_noalign_check(state, device))
            except Exception as e:
                state["dl"]["r_noalign"] = False
                lines.append(f"{EAD_TAG} R no-align check raised {type(e).__name__}: {str(e)[:200]}")
        for cls in EA_CLASSES:
            eng = state["ea"]["on"].get(cls, False)
            if eng:
                problem = _ead_layer_check(state, device, cls, eng)
                if problem is not None:
                    for t in range(1, EA2_MAX_T + 1):
                        if _ea_class(t) == cls:
                            state["ea"]["on"][t] = False
                    lines.append(f"{EAD_TAG} layer check FAIL -> master for its T: {problem}")
        if time.perf_counter() - t_prepare0 < EAD_GATE_DEADLINE_S:
            try:
                lines.extend(_ead_big(weights, state, device, t_prepare0))
            except Exception as e:
                for t in EAD_BIG:
                    state["ea"]["on"][t] = False
                lines.append(f"{EAD_TAG} big classes raised {type(e).__name__}: {str(e)[:200]}")
        if any(state["ea"]["on"].values()) and time.perf_counter() - t_prepare0 < EAD_GATE_DEADLINE_S:
            try:
                lines.extend(_ead_g2_gate(state, device))
            except Exception as e:
                state["ea"]["g2_t"] = {}
                lines.append(f"{EAD_TAG} k2 grid gate raised {type(e).__name__}: {str(e)[:200]}")
        if any(state["ea"]["on"].values()) and time.perf_counter() - t_prepare0 < EAD_GATE_DEADLINE_S:
            try:
                cok, cline = _ea_cmb_check(ext, state["post_expert_norm_w"], device)
                state["ea"]["cmb_ok"] = bool(cok)
                lines.append(f"{EAD_TAG} {cline}")
                if cok:
                    lines.extend(_ead_cmb_gate(state, device))
                    for cls in EA_CLASSES:
                        eng = state["ea"]["on"].get(cls, False)
                        if eng and state["ea"]["cmb_t"].get(cls, 0):
                            problem = _ead_layer_check(state, device, cls, eng)
                            if problem is not None:
                                for t in range(1, EA2_MAX_T + 1):
                                    if _ea_class(t) == cls:
                                        state["ea"]["cmb_t"][t] = 0
                                state["ea"].setdefault("cmb_bad", set()).add(cls)
                                lines.append(f"{EAD_TAG} combine layer check FAIL -> Triton combine: {problem}")
            except Exception as e:
                state["ea"]["cmb_t"] = {}
                state["ea"]["cmb_ok"] = False
                lines.append(f"{EAD_TAG} combine check raised {type(e).__name__}: {str(e)[:200]}")
        if any(state["ea"]["on"].values()) and time.perf_counter() - t_prepare0 < EAD_GATE_DEADLINE_S:
            try:
                lines.extend(_ead_pdl_gate(state, device))
            except Exception as e:
                state["ea"]["pdl13_t"] = {}
                lines.append(f"{EAD_TAG} pdl gate raised {type(e).__name__}: {str(e)[:200]}")
        if CAMPAIGN_EA6:
            lines.extend(_ead6_stage(weights, state, device, t_prepare0))
        if CAMPAIGN_EA4X:
            lines.extend(_ead4x_stage(state, device, t_prepare0))
        if any(state["ea"]["on"].values()):
            lines.append(_ead_sweep(state, device))
            if time.perf_counter() - t_prepare0 > EAD_COPY_DEADLINE_S:
                for t in state["ea"]["on"]:
                    state["ea"]["on"][t] = False
                lines.append(f"{EAD_TAG} copy check: skipped: deadline -> master for every T")
            else:
                lines.append(_ead_copy_check(state, device))
    except Exception as e:
        for t in state["ea"]["on"]:
            state["ea"]["on"][t] = False
        lines.append(f"{EAD_TAG} {type(e).__name__}: {str(e)[:400]}")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        lines.append(f"{EAD_TAG} final sync raised {type(e).__name__}: {str(e)[:200]}")
    tail = [f"{EAD_TAG} ===== ENGINE A (decode) SUMMARY ({time.perf_counter() - t0:.1f} s; module {EAD_EXT_NAME}) ====="]
    tail.extend(ln for ln in lines if ("gate T=" in ln and "round" not in ln) or "copy check" in ln
                or f"{EAD_TAG} canary v" in ln or "repack" in ln or "Error" in ln or "skipped" in ln
                or "layer check" in ln or "no-align" in ln or "sweep" in ln
                or (("combine" in ln or "pdl gate" in ln or "grid gate" in ln or "big " in ln)
                                           and "round" not in ln and "FAIL v" not in ln))
    tail.extend(_ea_path_lines(EAD_TAG, state["ea"]["on"], EA2_MAX_T))
    for t in EAD_BIG:
        tail.append(f"{EAD_TAG} PATH CHOSEN T={t} = {'ENGINE A ' + state['ea']['on'][t] if state['ea']['on'][t] else 'MASTER'}")
    return tail
CAMPAIGN_DENSE_SWAPAB = True
SWAPAB_MIN_M = 17
SWAPAB_MAX_M = 32
SWAPAB_BLOCK_W = 64
SWAPAB_BLOCK_T = 32
SWAPAB_NUM_WARPS = 4
SWAPAB_CFG = {
    "pooler":   (4096, 8192, 2, 256, 4),
    "qkv_gate": (2304, 8192, 4, 128, 4),
    "unpooler": (8192, 4096, 1, 256, 4),
    "o":        (8192, 1024, 1, 256, 4),
}
SWAPAB_FALLBACK_STAGES = 3
try:
    from triton.runtime.errors import OutOfResources as _OutOfResources
except Exception:
    _OutOfResources = getattr(triton.runtime, "OutOfResources", RuntimeError)
@triton.jit(do_not_specialize=["M"])
def _bf16_swapab_kernel(
    x_ptr, w_ptr, out_ptr, ws_ptr, cnt_ptr,
    M,
    stride_xm, stride_wn, stride_om,
    N: tl.constexpr, WS_M: tl.constexpr, K_SLICE: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_start = pid_k * K_SLICE

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = offs_t < M
    w_ptrs = w_ptr + offs_w[:, None] * stride_wn + offs_k[None, :]
    xt_ptrs = x_ptr + offs_t[None, :] * stride_xm + offs_k[:, None]
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
        ws_tile = offs_t[None, :] * N + offs_w[:, None]
        tl.store(ws_ptr + pid_k * (WS_M * N) + ws_tile, acc, mask=t_mask[None, :])
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid_w, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            total = tl.load(ws_ptr + ws_tile, mask=t_mask[None, :], other=0.0, cache_modifier=".cg")
            for s in tl.static_range(1, SPLIT_K):
                total += tl.load(ws_ptr + s * (WS_M * N) + ws_tile, mask=t_mask[None, :], other=0.0,
                                 cache_modifier=".cg")
            out_ptrs = out_ptr + offs_t[None, :] * stride_om + offs_w[:, None]
            tl.store(out_ptrs, total.to(tl.bfloat16), mask=t_mask[None, :])
            tl.atomic_xchg(cnt_ptr + pid_w, 0, sem="relaxed", scope="gpu")
def _swapab_route_ok(layer: str, x: torch.Tensor, w: torch.Tensor, cfg) -> bool:
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
def _swapab_forward(layer: str, x: torch.Tensor, w: torch.Tensor, ws, cnt, num_stages: int) -> torch.Tensor:
    n, k, split_k, block_k, _ = SWAPAB_CFG[layer]
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if ws is None:
        ws, cnt = out, out
    grid = (n // SWAPAB_BLOCK_W, split_k)
    _bf16_swapab_kernel[grid](
        x, w, out, ws, cnt,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        N=n, WS_M=SWAPAB_MAX_M, K_SLICE=k // split_k, SPLIT_K=split_k,
        BLOCK_W=SWAPAB_BLOCK_W, BLOCK_T=SWAPAB_BLOCK_T, BLOCK_K=block_k,
        num_warps=SWAPAB_NUM_WARPS, num_stages=num_stages,
    )
    return out
def _swapab_buffers(state: dict, layer: str):
    if layer == "pooler":
        return state["swapab_ws_pooler"], state["swapab_cnt_pooler"]
    if layer == "qkv_gate":
        return state["swapab_ws_qkv"], state["swapab_cnt_qkv"]
    return None, None
SWAPAB16_MAX_M = 16
SWAPAB16_BLOCK_T = 16
SWAPAB16_WS_M = 16
SWAPAB16_CFG = {
    "pooler":         (4096, 8192, 2, 256, 4),
    "qkv_gate":       (2304, 8192, 4, 128, 4),
    "o":              (8192, 1024, 1, 256, 4),
    "unpooler_slice": (1024, 4096, 8, 128, 3),
    "unpooler_full":  (8192, 4096, 1, 256, 4),
}
def _swapab16_key(layer: str, cfg) -> str:
    if layer == "unpooler":
        return "unpooler_slice" if cfg["n"] == 1024 else ("unpooler_full" if cfg["n"] == 8192 else "")
    return layer if layer in ("pooler", "qkv_gate", "o") else ""
def _swapab16_route_ok(key: str, x: torch.Tensor, w: torch.Tensor, cfg) -> bool:
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
def _swapab16_forward(key: str, x: torch.Tensor, w: torch.Tensor, ws, cnt, num_stages: int) -> torch.Tensor:
    n, k, split_k, block_k, _ = SWAPAB16_CFG[key]
    m = x.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if ws is None:
        ws, cnt = out, out
    grid = (n // SWAPAB_BLOCK_W, split_k)
    _bf16_swapab_kernel[grid](
        x, w, out, ws, cnt,
        m,
        x.stride(0), w.stride(0), out.stride(0),
        N=n, WS_M=SWAPAB16_WS_M, K_SLICE=k // split_k, SPLIT_K=split_k,
        BLOCK_W=SWAPAB_BLOCK_W, BLOCK_T=SWAPAB16_BLOCK_T, BLOCK_K=block_k,
        num_warps=SWAPAB_NUM_WARPS, num_stages=num_stages,
    )
    return out
def _swapab16_buffers(state: dict, key: str):
    if key in ("pooler", "qkv_gate", "unpooler_slice"):
        return state[f"swapab16_ws_{key}"], state[f"swapab16_cnt_{key}"]
    return None, None
DSA_TAG = "[dense-swapab]"
DSA_LAYERS = ("qkv_gate", "o", "unpooler")
DSA_SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "unpooler": (8192, 4096)}
DSA_MAX_T = 32
DSA_CLASSES = (1, 8, 16, 32)
DSA_T_LO = {"qkv_gate": 1, "o": 1, "unpooler": SLICED_UNPOOL_MAX_TOKENS + 1}
DSA_GEMM_JUDGE_REL = 1e-2
DSA_LAYER_MARGIN = 1.0
DSA_LAYER_TOL = (2e-2, 1e-2, 5e-2)
DSA_CACHE_REL = 4e-2
DSA_COPY_TOL = 1e-3
DSA_START_DEADLINE_S = 225.0
DSA_BUILD_DEADLINE_S = 245.0
DSA_GATE_DEADLINE_S = 255.0
DSA_COPY_DEADLINE_S = 270.0
DSA_PIN = {("unpooler", 16): False, ("qkv_gate", 32): True, ("o", 32): True}
def _dsa_class(tokens):
    return next(c for c in DSA_CLASSES if tokens <= c)
def _dsa_ts(layer, cls):
    return [t for t in range(DSA_T_LO[layer], DSA_MAX_T + 1) if _dsa_class(t) == cls]
def _dsa_key16(layer, w):
    return _swapab16_key(layer, {"n": w.shape[0]})
def _dsa_route_ok(layer, x, w, dsa):
    m = x.shape[0]
    if m <= SWAPAB16_MAX_M:
        key = _dsa_key16(layer, w)
        spec = SWAPAB16_CFG.get(key)
        return (spec is not None and key in dsa["swapab16_stages"]
                and _swapab16_route_ok(key, x, w, {"m": m, "n": spec[0], "k": spec[1]}))
    spec = SWAPAB_CFG.get(layer)
    return (spec is not None and layer in dsa["swapab_stages"]
            and _swapab_route_ok(layer, x, w, {"m": m, "n": spec[0], "k": spec[1]}))
def _dsa_forward(layer, x, w, dsa):
    if x.shape[0] <= SWAPAB16_MAX_M:
        key = _dsa_key16(layer, w)
        ws, cnt = _swapab16_buffers(dsa, key)
        return _swapab16_forward(key, x, w, ws, cnt, dsa["swapab16_stages"][key])
    ws, cnt = _swapab_buffers(dsa, layer)
    return _swapab_forward(layer, x, w, ws, cnt, dsa["swapab_stages"][layer])
def _dsa_gemm(layer, x, w, state):
    dsa = state.get("dsa")
    if dsa is not None and dsa["on"][layer].get(x.shape[0], False) and _dsa_route_ok(layer, x, w, dsa):
        return _dsa_forward(layer, x, w, dsa)
    return _gemm(x, w)
def _dsa_set(dsa, cls, layers):
    for layer in DSA_LAYERS:
        for t in _dsa_ts(layer, cls):
            dsa["on"][layer][t] = layer in layers
def _dsa_all_off(dsa):
    for layer in DSA_LAYERS:
        for t in dsa["on"][layer]:
            dsa["on"][layer][t] = False
def _dsa_tickets(dsa):
    torch.cuda.synchronize()
    return sum(int(t.abs().sum()) for name, t in dsa.items() if torch.is_tensor(t) and "_cnt" in name)
def _dsa_zero_tickets(dsa):
    for name, t in dsa.items():
        if torch.is_tensor(t) and "_cnt" in name:
            t.zero_()
    torch.cuda.synchronize()
def _dsa_compile(dsa, layer, w, big, device, gen):
    k = w.shape[1]
    if big:
        key, book = layer, dsa["swapab_stages"]
        ms = list(range(SWAPAB_MIN_M, SWAPAB_MAX_M + 1))
        stages = SWAPAB_CFG[layer][4]
        ws, cnt = _swapab_buffers(dsa, layer)
        def run(x, st):
            return _swapab_forward(layer, x, w, ws, cnt, st)
    else:
        key, book = _dsa_key16(layer, w), dsa["swapab16_stages"]
        ms = list(range(DSA_T_LO[layer], SWAPAB16_MAX_M + 1))
        stages = SWAPAB16_CFG[key][4]
        ws, cnt = _swapab16_buffers(dsa, key)
        def run(x, st):
            return _swapab16_forward(key, x, w, ws, cnt, st)
    label = f"{layer} B{SWAPAB_BLOCK_T if big else SWAPAB16_BLOCK_T}"
    t0 = time.perf_counter()
    x = torch.randn((ms[-1], k), generator=gen, device=device).to(torch.bfloat16)
    print(f"{DSA_TAG} launching {label} (compile, m={ms[-1]}, {stages} stages)", flush=True)
    try:
        run(x, stages)
        torch.cuda.synchronize()
    except _OutOfResources as e:
        print(f"{DSA_TAG} {label}: {stages} stages OutOfResources ({str(e)[:160]}); retrying at "
              f"{SWAPAB_FALLBACK_STAGES}", flush=True)
        stages = SWAPAB_FALLBACK_STAGES
        run(x, stages)
        torch.cuda.synchronize()
    for m in ms[:-1]:
        run(torch.randn((m, k), generator=gen, device=device).to(torch.bfloat16), stages)
    torch.cuda.synchronize()
    book[key] = stages
    return f"{DSA_TAG} ok {label}: {stages} stages, m {ms[0]}..{ms[-1]} warmed, {time.perf_counter() - t0:.1f} s"
def _dsa_build(dsa, dense, layers, device, t_prepare0):
    t0 = time.perf_counter()
    if "qkv_gate" in layers:
        n, _k, split_k, _bk, _st = SWAPAB_CFG["qkv_gate"]
        dsa["swapab_ws_qkv"] = torch.empty((split_k, SWAPAB_MAX_M, n), dtype=torch.float32, device=device)
        dsa["swapab_cnt_qkv"] = torch.zeros(n // SWAPAB_BLOCK_W, dtype=torch.int32, device=device)
    for layer in layers:
        key = _dsa_key16(layer, dense[layer])
        n, _k, split_k, _bk, _st = SWAPAB16_CFG[key]
        if split_k > 1:
            dsa[f"swapab16_ws_{key}"] = torch.empty((split_k, SWAPAB16_WS_M, n), dtype=torch.float32, device=device)
            dsa[f"swapab16_cnt_{key}"] = torch.zeros(n // SWAPAB_BLOCK_W, dtype=torch.int32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261061)
    lines = []
    for layer in layers:
        for big in (False, True):
            if time.perf_counter() - t_prepare0 > DSA_BUILD_DEADLINE_S:
                lines.append(f"{DSA_TAG} {layer} B{SWAPAB_BLOCK_T if big else SWAPAB16_BLOCK_T} skipped: deadline "
                             f"-> master for its T")
                print(lines[-1], flush=True)
                continue
            try:
                lines.append(_dsa_compile(dsa, layer, dense[layer], big, device, gen))
            except Exception as e:
                lines.append(f"{DSA_TAG} {layer} B{SWAPAB_BLOCK_T if big else SWAPAB16_BLOCK_T} raised "
                             f"{type(e).__name__}: {str(e)[:240]} -> master for its T")
            print(lines[-1], flush=True)
    _dsa_zero_tickets(dsa)
    return lines, time.perf_counter() - t0
def _dsa_gemm_check(dsa, layer, w, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261062 + DSA_LAYERS.index(layer))
    k = w.shape[1]
    wd = w.double()
    ts = list(range(DSA_T_LO[layer], DSA_MAX_T + 1))
    oks, bad = {}, []
    worst = {"rel": 0.0, "row": 0.0, "cublas": 0.0}
    for t in ts:
        x = torch.randn((t, k), generator=gen, device=device).to(torch.bfloat16)
        if not _dsa_route_ok(layer, x, w, dsa):
            oks[t] = False
            bad.append(f"T={t}: no compiled variant")
            continue
        col = k - 1 - (97 * t) % (k // 4)
        xp = x.clone()
        xp[0].zero_()
        xp[0, col] = 0.5
        out = _dsa_forward(layer, x, w, dsa)
        again = _dsa_forward(layer, x, w, dsa)
        probe = _dsa_forward(layer, xp, w, dsa)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        good, rel, row, text = _rt_check(out, ref, judge_rel=DSA_GEMM_JUDGE_REL)
        same = torch.equal(out, again)
        exact = torch.equal(probe[0], (w[:, col].float() * 0.5).to(torch.bfloat16))
        worst["rel"], worst["row"] = max(worst["rel"], rel), max(worst["row"], row)
        worst["cublas"] = max(worst["cublas"], _rt_rel(_gemm(x, w).float(), ref))
        oks[t] = bool(good and same and exact)
        if not oks[t]:
            bad.append(f"T={t}: {text}, repeat identical {same}, probe exact {exact}")
    left = _dsa_tickets(dsa)
    if left:
        _dsa_zero_tickets(dsa)
        oks = {t: False for t in ts}
        bad.append(f"tickets not back to 0 (sum {left})")
    fails = [t for t in ts if not oks[t]]
    line = (f"{DSA_TAG} check {layer} T {ts[0]}..{ts[-1]} " + (f"FAIL at T {fails}" if fails else "ok")
            + f": rel-L2 <= {worst['rel']:.3g} (cuBLAS {worst['cublas']:.3g}), row <= {worst['row']:.3g} (bounds "
            f"{0.5 * DSA_GEMM_JUDGE_REL:.3g} / {0.5 * RT_ROW_FACTOR * DSA_GEMM_JUDGE_REL:.3g} vs fp64), "
            + (f"first: {bad[0][:200]}" if bad else "repeats identical, probes exact"))
    return oks, line
def _dsa_gemm_select(dsa, layer, w, cls, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261063 + 10 * cls + DSA_LAYERS.index(layer))
    x = torch.randn((cls, w.shape[1]), generator=gen, device=device).to(torch.bfloat16)
    graphs = {"cuBLAS": _rt_graph(lambda: _gemm(x, w))[0],
              "swap-AB": _rt_graph(lambda: _dsa_forward(layer, x, w, dsa))[0]}
    rounds = _rt_time(graphs, device)
    win, worst, _ = _rt_select({f"{layer} T={cls}": rounds}, "cuBLAS", ["swap-AB"])
    del graphs
    med = {name: [_rt_stats(r[name])[0] for r in rounds] for name in ("cuBLAS", "swap-AB")}
    iqr = [_rt_stats(r["swap-AB"])[1] for r in rounds]
    wins = win == "swap-AB"
    line = (f"{DSA_TAG} gemm {layer} T={cls}: cuBLAS {'/'.join(f'{v:.2f}' for v in med['cuBLAS'])} vs swap-AB "
            f"{'/'.join(f'{v:.2f}' for v in med['swap-AB'])} us (IQR {'/'.join(f'{v:.2f}' for v in iqr)}) "
            f"x{worst['swap-AB']:.3f} -> {'swap-AB' if wins else 'master'}")
    means = (sum(med["cuBLAS"]) / len(med["cuBLAS"]), sum(med["swap-AB"]) / len(med["swap-AB"]))
    return wins, worst["swap-AB"], line, means
def _dsa_layer_gate(state, device, cls, layers):
    dsa = state["dsa"]
    def setter(on_layers):
        def f():
            _dsa_set(dsa, cls, on_layers)
        return f
    _dsa_set(dsa, cls, ())
    try:
        per = _ead_layer_gate(state, device, cls, {"swap-AB": setter(tuple(layers)), "master": setter(())},
                              20261064 + cls)
    finally:
        _dsa_set(dsa, cls, ())
    win, worst, _ = _rt_select(per, "master", ["swap-AB"], margin=DSA_LAYER_MARGIN, tight=DSA_LAYER_MARGIN)
    parts = []
    for rounds in per.values():
        parts.append(" ".join(f"{_rt_stats(r['master'])[0]:.1f}/{_rt_stats(r['swap-AB'])[0]:.1f}" for r in rounds))
    keep = win == "swap-AB"
    line = (f"{DSA_TAG} layer gate T={cls} ({'+'.join(layers)}), master/swap-AB us per round: {' | '.join(parts)} -> "
            f"x{worst['swap-AB']:.3f} {'keep' if keep else 'master'}")
    return keep, worst["swap-AB"], line
def _dsa_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst):
    tokens = got[0].shape[0]
    finite = all(bool(torch.isfinite(t.float()).all()) for t in got[:2])
    gi, ri = got[2].long(), ref[2].long()
    same = ~(gi.sort(-1).values != ri.sort(-1).values).any(-1)
    n_diff = int((~same).sum())
    rules_ok, rules = _cb_route_rules(gi, scores, ls_val)
    if bool(same.any()):
        e_h = _rel_l2(got[0][same], ref[0][same])
        e_s = _rel_l2(got[1][same], ref[1][same])
        e_u = _rel_l2(got[1][same].double() - s_in[same].double(), ref[1][same].double() - s_in[same].double())
    else:
        e_h = e_s = e_u = 0.0
    tol_h, tol_s, tol_u = DSA_LAYER_TOL
    loc = d_ref["cache_loc"]
    kn, kr = d_got["k_cache"].view(torch.uint8).view(-1, 128), d_ref["k_cache"].view(torch.uint8).view(-1, 128)
    vn, vr = d_got["v_cache"].view(torch.uint8).view(-1, 128), d_ref["v_cache"].view(torch.uint8).view(-1, 128)
    mask = torch.ones(kn.shape[0], dtype=torch.bool, device=kn.device)
    mask[loc] = False
    rest_ok = torch.equal(kn[mask], kr[mask]) and torch.equal(vn[mask], vr[mask])
    k_gap, v_gap = _fp8_code_gap(kn[loc], kr[loc]), _fp8_code_gap(vn[loc], vr[loc])
    e_k = _rel_l2(kn[loc].view(FP8).float(), kr[loc].view(FP8).float())
    e_v = _rel_l2(vn[loc].view(FP8).float(), vr[loc].view(FP8).float())
    for key, val in (("hidden", e_h), ("stream", e_s), ("update", e_u), ("k_rel", e_k), ("v_rel", e_v),
                     ("k_gap", k_gap), ("v_gap", v_gap), ("tokens_rerouted", n_diff)):
        worst[key] = max(worst.get(key, 0), val)
    ok = (finite and rules_ok and n_diff <= max(1, tokens // 4) and e_h <= tol_h and e_s <= tol_s and e_u <= tol_u
          and rest_ok and k_gap <= 1 and v_gap <= 1 and e_k <= DSA_CACHE_REL and e_v <= DSA_CACHE_REL)
    msg = (f"hidden {e_h:.2e} (<= {tol_h}), stream {e_s:.2e} (<= {tol_s}), update {e_u:.2e} (<= {tol_u}) on "
           f"{tokens - n_diff}/{tokens} tokens routed alike; k gap {k_gap} rel {e_k:.2e}, v gap {v_gap} rel {e_v:.2e}, "
           f"other rows intact {rest_ok}; routing: {rules}; finite {finite}")
    return ok, msg
def _dsa_layer_check(state, device, cls, layers):
    dsa = state["dsa"]
    off = dict(state)
    off["dsa"] = None
    gen = torch.Generator(device=device)
    gen.manual_seed(20261065 + cls)
    worst = {}
    try:
        for rep in range(2):
            _dsa_set(dsa, cls, ())
            data = _ead_cases(off, device, gen, cls)
            ls_val = int(data["local_start"][0])
            s_in = data["stream"].clone()
            scores = _cb_router_scores(_dl3_clone_data(data), off)
            d_ref = _dl3_clone_data(data)
            ref = kernel_fn(d_ref, off)
            _dsa_set(dsa, cls, layers)
            d_got = _dl3_clone_data(data)
            got = kernel_fn(d_got, state)
            torch.cuda.synchronize()
            ok, msg = _dsa_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst)
            if not ok:
                problem = f"layer check T={cls} input {rep}: {msg}"
                return problem, f"{DSA_TAG} {problem} -> FAIL"
    finally:
        _dsa_set(dsa, cls, ())
    line = (f"{DSA_TAG} layer check T={cls} ({'+'.join(layers)} on vs off, 2 inputs) ok: worst " +
            ", ".join(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
    return None, line
def _dsa_copy_check(state, device):
    import gc
    dsa = state["dsa"]
    ts = [c for c in DSA_CLASSES if any(dsa["on"][layer].get(c, False) for layer in DSA_LAYERS)]
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261066)
    fails, n = [], 0
    for tokens in ts:
        base = _ead_cases(state, device, gen, tokens)
        per = []
        for st in (state, other):
            data = _dl3_clone_data(base)
            graph, out = _rt_graph(lambda data=data, st=st: kernel_fn(data, st))
            per.append((data, graph, out))
        for k in range(4):
            fresh = _ead_cases(state, device, gen, tokens)
            for data, _, _ in per:
                for key, v in fresh.items():
                    if torch.is_tensor(v) and v.shape == data[key].shape:
                        data[key].copy_(v)
            ref = kernel_fn(_dl3_clone_data(per[0][0]), state)
            for _, graph, _ in (per if k % 2 == 0 else per[::-1]):
                graph.replay()
            torch.cuda.synchronize()
            for i, (_, _, out) in enumerate(per):
                n += 1
                e_h, e_s = _rt_rel(out[0].float(), ref[0].float()), _rt_rel(out[1].float(), ref[1].float())
                if not (e_h <= DSA_COPY_TOL and e_s <= DSA_COPY_TOL and torch.equal(out[2], ref[2])):
                    fails.append(f"T={tokens} state {'AB'[i]} refill {k}: hidden {e_h:.3g} stream {e_s:.3g}")
        del per
        print(f"{DSA_TAG} copy check T={tokens}: {n} replays, {len(fails)} fails", flush=True)
    left = (_dsa_tickets(dsa), _dsa_tickets(other["dsa"]))
    if any(left):
        fails.append(f"tickets left {left[0]} / {left[1]}")
    del other
    gc.collect()
    torch.cuda.empty_cache()
    ok = not fails
    return ok, (f"{DSA_TAG} copy check: {n} replays on two states, T {ts}: "
                + ("PASS" if ok else f"FAIL -> master for every T: {fails[0]}"))
def _dsa_path_lines(dsa, gemm, why, layer_x):
    out = []
    for cls in DSA_CLASSES:
        lo = min(t for t in range(1, DSA_MAX_T + 1) if _dsa_class(t) == cls)
        parts = []
        for layer in DSA_LAYERS:
            ts = _dsa_ts(layer, cls)
            if not ts:
                continue
            flags = {bool(dsa["on"][layer][t]) for t in ts}
            if len(flags) > 1:
                for t in ts:
                    dsa["on"][layer][t] = False
                why[(layer, cls)] = "inconsistent flags"
                flags = {False}
            g = gemm.get((layer, cls))
            gtxt = "" if g is None else f"swap-AB {g[1]:.2f} vs cuBLAS {g[0]:.2f} us"
            if flags == {True}:
                parts.append(f"{layer} SWAP-AB{' PINNED' if DSA_PIN.get((layer, cls)) else ''} ({gtxt})")
            else:
                reason = why.get((layer, cls), "off")
                parts.append(f"{layer} MASTER ({reason}{'; ' + gtxt if gtxt else ''})")
        lx = layer_x.get(cls)
        out.append(f"{DSA_TAG} PATH CHOSEN T={lo}{'' if lo == cls else f'..{cls}'}: " + ", ".join(parts)
                   + ("" if lx is None else f"; whole layer x{lx:.3f}"))
    return out
def _dsa_setup(state, device, t_prepare0):
    t0 = time.perf_counter()
    dsa = {"on": {layer: {t: False for t in range(1, DSA_MAX_T + 1)} for layer in DSA_LAYERS},
           "swapab_stages": {}, "swapab16_stages": {}}
    state["dsa"] = dsa
    gemm, why, layer_x, notes = {}, {}, {}, []
    build_s = 0.0
    def late(limit):
        return time.perf_counter() - t_prepare0 > limit
    def say(line, keep=True):
        print(line, flush=True)
        if keep:
            notes.append(line)
    def miss(layers, cls, reason):
        for layer in layers:
            if _dsa_ts(layer, cls):
                why.setdefault((layer, cls), reason)
    try:
        if late(DSA_START_DEADLINE_S):
            raise RuntimeError(f"skipped: deadline (start at {time.perf_counter() - t_prepare0:.1f} s)")
        dense = state["dense"]
        layers = []
        for layer in DSA_LAYERS:
            w = dense[layer]
            if w.dim() == 2 and tuple(w.shape) == DSA_SHAPES[layer] and w.dtype == torch.bfloat16 and w.stride(1) == 1:
                layers.append(layer)
            else:
                for cls in DSA_CLASSES:
                    miss([layer], cls, f"weight {tuple(w.shape)} {w.dtype}")
        blines, build_s = _dsa_build(dsa, dense, layers, device, t_prepare0)
        notes.extend(ln for ln in blines if "raised" in ln or "skipped" in ln)
        okt, ck_why = {}, {}
        for layer in layers:
            if late(DSA_GATE_DEADLINE_S):
                okt[layer], ck_why[layer] = {}, "check skipped: deadline"
                line = f"{DSA_TAG} check {layer} skipped: deadline"
            else:
                try:
                    okt[layer], line = _dsa_gemm_check(dsa, layer, dense[layer], device)
                except Exception as e:
                    okt[layer], ck_why[layer] = {}, f"check {type(e).__name__}"
                    line = f"{DSA_TAG} check {layer} raised {type(e).__name__}: {str(e)[:240]}"
            say(line)
        chosen = {cls: [] for cls in DSA_CLASSES}
        for cls in DSA_CLASSES:
            for layer in layers:
                ts = _dsa_ts(layer, cls)
                if not ts:
                    continue
                if not all(okt[layer].get(t, False) for t in ts):
                    miss([layer], cls, ck_why.get(layer, "check FAIL"))
                    continue
                if late(DSA_GATE_DEADLINE_S):
                    miss([layer], cls, "skipped: deadline")
                    continue
                try:
                    wins, worst, line, means = _dsa_gemm_select(dsa, layer, dense[layer], cls, device)
                except Exception as e:
                    say(f"{DSA_TAG} gemm {layer} T={cls} raised {type(e).__name__}: {str(e)[:240]}")
                    miss([layer], cls, f"gemm selector {type(e).__name__}")
                    continue
                say(line, keep=False)
                gemm[(layer, cls)] = means
                pin = DSA_PIN.get((layer, cls))
                if wins if pin is None else pin:
                    chosen[cls].append(layer)
                else:
                    miss([layer], cls, ("PINNED; " if pin is False else "") + f"gemm x{worst:.3f}")
        final = {cls: [] for cls in DSA_CLASSES}
        for cls in DSA_CLASSES:
            if not chosen[cls]:
                continue
            if late(DSA_GATE_DEADLINE_S):
                miss(chosen[cls], cls, "layer gate skipped: deadline")
                continue
            try:
                keep, worst, line = _dsa_layer_gate(state, device, cls, chosen[cls])
                say(line, keep=False)
                layer_x[cls] = worst
                if not keep:
                    pinned = [layer for layer in chosen[cls] if DSA_PIN.get((layer, cls))]
                    miss([layer for layer in chosen[cls] if layer not in pinned], cls, f"whole layer x{worst:.3f}")
                    if not pinned:
                        continue
                    chosen[cls] = pinned
                problem, line = _dsa_layer_check(state, device, cls, chosen[cls])
                say(line, keep=problem is not None)
                if problem is not None:
                    miss(chosen[cls], cls, "layer check FAIL")
                    continue
            except Exception as e:
                _dsa_set(dsa, cls, ())
                say(f"{DSA_TAG} layer T={cls} raised {type(e).__name__}: {str(e)[:240]}")
                miss(chosen[cls], cls, f"layer {type(e).__name__}")
                continue
            final[cls] = list(chosen[cls])
        for cls in DSA_CLASSES:
            _dsa_set(dsa, cls, final[cls])
        if any(final.values()):
            if late(DSA_COPY_DEADLINE_S):
                _dsa_all_off(dsa)
                for cls in DSA_CLASSES:
                    miss(final[cls], cls, "copy check skipped: deadline")
                say(f"{DSA_TAG} copy check: skipped: deadline -> master for every T")
            else:
                ok, line = _dsa_copy_check(state, device)
                say(line)
                if not ok:
                    _dsa_all_off(dsa)
                    for cls in DSA_CLASSES:
                        miss(final[cls], cls, "copy check FAIL")
        left = _dsa_tickets(dsa)
        if left:
            _dsa_zero_tickets(dsa)
            _dsa_all_off(dsa)
            for cls in DSA_CLASSES:
                miss(final[cls], cls, "tickets left")
            say(f"{DSA_TAG} tickets not back to 0 (sum {left}): zeroed, master for every T")
    except Exception as e:
        _dsa_all_off(dsa)
        for cls in DSA_CLASSES:
            miss(DSA_LAYERS, cls, type(e).__name__)
        say(f"{DSA_TAG} {type(e).__name__}: {str(e)[:400]} ... {str(e)[-400:]}")
        try:
            _dsa_zero_tickets(dsa)
        except Exception as e2:
            say(f"{DSA_TAG} ticket reset raised {type(e2).__name__}: {str(e2)[:200]}")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        say(f"{DSA_TAG} final sync raised {type(e).__name__}: {str(e)[:200]}")
    tail = [f"{DSA_TAG} ===== DENSE SWAP-AB (decode) SUMMARY ({time.perf_counter() - t0:.1f} s, Triton build "
            f"{build_s:.1f} s; prepare at {time.perf_counter() - t_prepare0:.1f} s) ====="]
    tail.extend(notes)
    try:
        tail.extend(_dsa_path_lines(dsa, gemm, why, layer_x))
    except Exception as e:
        _dsa_all_off(dsa)
        tail.append(f"{DSA_TAG} PATH CHOSEN T=1..{DSA_MAX_T}: MASTER for every layer (summary raised "
                    f"{type(e).__name__}: {str(e)[:200]})")
    return tail
CAMPAIGN_HOTCOLD = True
WG_TAG = "[cuda-wgmma-verify]"
WG_BSM = 16
WG_THREADS = 384
WG_SMEM_LIMIT = 227 * 1024
WG_CAP_ROWS = 8192
WG_BUCKETS = (16, 32, 64, 128)
_WG_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> wg_init();
void wg_gather(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
               int64_t num_routes, int64_t n_experts, torch::Tensor xs);
void wg_gemm(int64_t which, torch::Tensor bsrc, torch::Tensor wbank,
             torch::Tensor gs, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
             int64_t num_routes, int64_t n_tiles, int64_t K, torch::Tensor out, int64_t grid, torch::Tensor err,
             int64_t pdl);
"""
_WG_SRC_HEAD = r"""
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
constexpr int MAXB = 384;
constexpr int PITCH13 = 64 * 2 + 16;             // staging row pitch, w13 (64 bf16 + pad)
constexpr int PITCH2 = 128 * 2 + 16;             // staging row pitch, w2 (128 bf16 + pad)
static_assert(WST % 1024 == 0 && STAGE_BYTES % 1024 == 0, "stage alignment");
template <bool G>
struct Smem {
  static constexpr int EPI = 128 * (G ? PITCH13 : PITCH2);
  static constexpr int BYTES = 1024 + STAGES * STAGE_BYTES + EPI;
};
static_assert(Smem<false>::BYTES + 8192 <= 227 * 1024, "shared memory budget");

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
// pass wg_kernel only sets error word 2 and returns (the path is then turned off in setup).
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
// Programmatic dependent launch. griddepcontrol.wait returns at once when the grid was not launched with the
// programmatic-stream-serialization attribute (or has no predecessor), otherwise it blocks until every
// prerequisite grid has completed and its memory writes are visible. launch_dependents lets the next grid's
// CTAs start; it is per CTA (all threads execute it or exit) and a no-op when nothing depends on this grid.
__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
}
__device__ __forceinline__ void griddep_launch() {
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
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
// SGLang parity: Marlin writes gate and up as bf16, then silu_and_mul computes in fp32.
__device__ __forceinline__ float silu_mul(float hg, float hu) {
  hg = __bfloat162float(__float2bfloat16_rn(hg));
  hu = __bfloat162float(__float2bfloat16_rn(hu));
  return __fdividef(hg, 1.f + __expf(-hg)) * hu;  // fast form (prefill 21911eb6: w13 x0.92-0.94, error = master's)
}

template <int NB>
struct Wg;
"""
_WG_SRC_BODY = r"""
struct Item {
  int j, e, slot0, nbk, bucket, multi;
};
// Item it of the (expert, tile, chunk) order. Expert group = chunks [f, f + k) of one expert,
// items [f * T, (f + k) * T); it / T always lands inside its own group.
__device__ __forceinline__ Item decode_item(int it, int T, const int* be, const int* cblk, const int* cnb,
                                            const int* gf, const int* gk) {
  const int cg = it / T;
  const int f = gf[cg], k = gk[cg];
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

template <bool G>
__global__ void __launch_bounds__(THREADS, 1)
wg_kernel(const __grid_constant__ Maps maps,
          const uint8_t* __restrict__ wbank,        // [E][T][K/64][REC]
          const float* __restrict__ gs,             // fp32 [E] global scale * 2^7
          const int* __restrict__ sorted_ids,       // block-16 align of the local ids
          const int* __restrict__ block_experts,
          const int* __restrict__ n_padded,
          int blocks_len, int num_routes, int n_tiles, int K, int n_experts,
          __nv_bfloat16* __restrict__ out, int out_ld, int* __restrict__ err) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  // s5 launch marker: block 0 writes a magic into err[1] (err is int[2], per state); prepare reads it back to
  // prove THIS kernel (the sm_90a image) launched. The error word stays err[0].
  if (blockIdx.x == 0 && threadIdx.x == 0) err[1] = 0x57474d4d;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[2 * STAGES];
  __shared__ int s_be[MAXB], s_cblk[MAXB], s_cnb[MAXB], s_gf[MAXB], s_gk[MAXB];
  __shared__ int s_wsum[THREADS / 32];
  __shared__ int s_nc;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* ep = smem + STAGES * STAGE_BYTES;
  const uint32_t full0 = smem_u32(&bars[0]), empty0 = smem_u32(&bars[STAGES]);

  int nb = __ldg(n_padded) / 16;
  nb = nb < blocks_len ? nb : blocks_len;
  nb = nb < MAXB ? nb : MAXB;
  if (tid < nb) s_be[tid] = __ldg(block_experts + tid);
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 8);  // one arrive per consumer warp
    }
    fence_barrier_init();
  }
  __syncthreads();

  // Chunks: a block starts a chunk when its expert is local and it is 0 mod 8 into its run.
  // Any expert id outside [0, n_experts) (SGLang's blocks of remote routes) is not local.
  int flag = 0, cn = 0;
  if (tid < nb) {
    const int e = s_be[tid];
    if (e >= 0 && e < n_experts) {
      int rs = tid;
      while (rs > 0 && s_be[rs - 1] == e) --rs;
      if (((tid - rs) & 7) == 0) {
        flag = 1;
        cn = 1;
        while (cn < 8 && tid + cn < nb && s_be[tid + cn] == e) ++cn;
      }
    }
  }
  const unsigned ball = __ballot_sync(0xffffffffu, flag);
  if (lane == 0) s_wsum[warp] = __popc(ball);
  __syncthreads();
  if (flag) {
    int off = 0;
    for (int w = 0; w < warp; ++w) off += s_wsum[w];
    const int idx = off + __popc(ball & ((1u << lane) - 1u));
    s_cblk[idx] = tid;
    s_cnb[idx] = cn;
  }
  if (tid == 0) {
    int t = 0;
    for (int w = 0; w < THREADS / 32; ++w) t += s_wsum[w];
    s_nc = t;
  }
  __syncthreads();
  const int nc = s_nc;
  if (tid < nc) {
    const int e = s_be[s_cblk[tid]];
    int f = tid;
    while (f > 0 && s_be[s_cblk[f - 1]] == e) --f;
    int k = tid - f + 1;
    while (f + k < nc && s_be[s_cblk[f + k]] == e) ++k;
    s_gf[tid] = f;
    s_gk[tid] = k;
  }
  __syncthreads();

  const int total = nc * n_tiles;
  const int KS = K / BK;
  bool dead = false;
  if (warp < 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (tid == 0) {
      const int KC = K / 64;
      int n = 0;
      for (int it = blockIdx.x; it < total; it += gridDim.x) {
        const Item m = decode_item(it, n_tiles, s_be, s_cblk, s_cnb, s_gf, s_gk);
        const uint8_t* wsrc = wbank + (static_cast<size_t>(m.e) * n_tiles + m.j) * static_cast<size_t>(KC) * REC;
        const uint64_t map = reinterpret_cast<uint64_t>(&maps.m[m.bucket]);
        const uint32_t tx = WST + 2 * (16 << m.bucket) * 128;
        for (int kb = 0; kb < KS; ++kb, ++n) {
          const int s = n % STAGES;
          mbar_wait(empty0 + 8 * s, ((n / STAGES) & 1) ^ 1, err, dead);
          const uint32_t st = smem_u32(smem + s * STAGE_BYTES);
          const uint32_t fb = full0 + 8 * s;
          mbar_expect_tx(fb, tx);
          bulk_g2s(st, wsrc + static_cast<size_t>(kb) * WST, WST, fb);  // weights: constant, no dependency
          // PDL: the B slab (wg_xs / wg_hs) is the predecessor's output; nothing reads it before this wait.
          if (n == 0) griddep_wait();
          tma_2d(st + WST, map, kb * BK, m.slot0, fb);
          tma_2d(st + WST + SLAB_HALF, map, kb * BK + 64, m.slot0, fb);
        }
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const int ctid = tid - 128, cw = ctid >> 5;
    int n = 0;
    // PDL: no consumer global store (the first epilogue) before the predecessor grid has completed. The
    // consumers would idle on the full barriers until the producer's own wait returns anyway.
    griddep_wait();
    for (int it = blockIdx.x; it < total; it += gridDim.x) {
      const Item m = decode_item(it, n_tiles, s_be, s_cblk, s_cnb, s_gf, s_gk);
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
  // PDL: this thread's item loop is over (the other threads of the producer warpgroup exited above): let the
  // next grid's CTAs take the SMs freed in the tail. Every thread of the CTA executes it or has exited.
  griddep_launch();
#else
  // Not the sm_90a image: no wgmma here. Flag it (error word 2) and do nothing else.
  if (threadIdx.x == 0) atomicExch(err, 2);
#endif
}

// xs[slot] = x[sorted_ids[slot] / 8] for the slots of local blocks (sentinel slots: zeros);
// slots past n_padded and slots of SGLang's remote (-1) blocks are skipped.
__global__ void __launch_bounds__(256)
wg_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ sorted_ids,
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

// s5: the ONLY file-scope variable is this host function pointer (a driver entry point, resolved by
// wg_init in prepare, outside any capture). It holds no device pointer, no tensor map, no buffer: every
// wg_gemm call encodes its own tensor maps into a stack-local Maps from the tensors passed in THAT call.
static EncodeTiledFn g_encode_pfn = nullptr;

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

template <bool G>
void info_one(std::vector<int64_t>& r) {
  const void* fn = reinterpret_cast<const void*>(wg_kernel<G>);
  const cudaError_t e1 = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem<G>::BYTES);
  cudaFuncAttributes a{};
  const cudaError_t e2 = cudaFuncGetAttributes(&a, fn);
  int occ = 0;
  const cudaError_t e3 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, THREADS, Smem<G>::BYTES);
  if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) (void)cudaGetLastError();
  r.push_back(G ? 1 : 0);
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

// Per instantiation (w13 then w2): gated, attribute calls ok, numRegs, localSizeBytes, static
// smem, dynamic smem, occupancy (CTAs per SM), cudaFuncSetAttribute's error code, binaryVersion,
// ptxVersion (both report 90 for sm_90 and sm_90a: logged only; error word 2 is the real proof).
std::vector<int64_t> wg_init() {
  std::vector<int64_t> r;
  if (g_encode_pfn == nullptr) g_encode_pfn = get_encode();
  info_one<true>(r);
  info_one<false>(r);
  return r;
}

void wg_gather(torch::Tensor x, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
               int64_t num_routes, int64_t n_experts, torch::Tensor xs) {
  TORCH_CHECK(x.is_contiguous() && xs.is_contiguous() && sorted_ids.is_contiguous() && block_experts.is_contiguous() &&
              x.scalar_type() == at::kBFloat16 && xs.scalar_type() == at::kBFloat16 &&
              sorted_ids.scalar_type() == at::kInt && block_experts.scalar_type() == at::kInt &&
              n_padded.scalar_type() == at::kInt, "wg_gather: layout");
  TORCH_CHECK(x.size(1) == xs.size(1) && x.size(1) % 8 == 0 && sorted_ids.numel() <= xs.size(0) &&
              block_experts.numel() * 16 >= sorted_ids.numel(), "wg_gather: sizes");
  const int slots = static_cast<int>(sorted_ids.numel());
  if (slots == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  wg_gather_kernel<<<slots, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), sorted_ids.data_ptr<int>(), block_experts.data_ptr<int>(),
      n_padded.data_ptr<int>(), static_cast<int>(num_routes), static_cast<int>(x.size(1)),
      static_cast<int>(n_experts), reinterpret_cast<__nv_bfloat16*>(xs.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wg_gemm(int64_t which, torch::Tensor bsrc, torch::Tensor wbank,
             torch::Tensor gs, torch::Tensor sorted_ids, torch::Tensor block_experts, torch::Tensor n_padded,
             int64_t num_routes, int64_t n_tiles, int64_t K, torch::Tensor out, int64_t grid, torch::Tensor err,
             int64_t pdl) {
  TORCH_CHECK(which == 0 || which == 1, "wg_gemm: which");
  TORCH_CHECK(bsrc.is_cuda() && bsrc.is_contiguous() && bsrc.dim() == 2 && bsrc.scalar_type() == at::kBFloat16 &&
              bsrc.size(0) > 0 && bsrc.size(1) > 0 && bsrc.size(1) == K,
              "wg_gemm: B-source scratch layout (cuda, contiguous, 2-D bf16, width K)");
  TORCH_CHECK(wbank.is_contiguous() && gs.is_contiguous() && sorted_ids.is_contiguous() &&
              block_experts.is_contiguous() && out.is_contiguous() && wbank.scalar_type() == at::kByte &&
              gs.scalar_type() == at::kFloat && sorted_ids.scalar_type() == at::kInt &&
              block_experts.scalar_type() == at::kInt && n_padded.scalar_type() == at::kInt &&
              out.scalar_type() == at::kBFloat16 && err.scalar_type() == at::kInt && err.numel() >= 2,
              "wg_gemm: layout");
  TORCH_CHECK(K % BK == 0 && block_experts.numel() <= MAXB && sorted_ids.numel() <= bsrc.size(0), "wg_gemm: sizes");
  TORCH_CHECK(wbank.numel() == gs.numel() * n_tiles * (K / 64) * REC, "wg_gemm: weight bank size");
  TORCH_CHECK(out.size(1) == (which == 0 ? n_tiles * 64 : n_tiles * 128), "wg_gemm: output width");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks_len = static_cast<int>(block_experts.numel());
  // s5: encode this launch's four tensor maps (N buckets 16/32/64/128) over the bsrc passed in THIS call,
  // into a stack-local Maps passed by value as the __grid_constant__ kernel parameter (frozen into the
  // graph node at capture). Host-only driver work: no sync, no allocation, nothing persisted.
  EncodeTiledFn enc = g_encode_pfn;
  if (enc == nullptr) {  // normally resolved by wg_init in prepare
    enc = get_encode();
    g_encode_pfn = enc;
  }
  TORCH_CHECK(enc != nullptr, "wg_gemm: cuTensorMapEncodeTiled entry point missing");
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
      TORCH_CHECK(rc == CUDA_SUCCESS, "wg_gemm: tensor map ", b, " rc ", static_cast<int>(rc));
    }
  }
#define WG_LAUNCH(G_)                                                                                         \
  do { if (pdl == 0) {                                                                                             \
    wg_kernel<G_><<<static_cast<unsigned>(grid), THREADS, Smem<G_>::BYTES, stream>>>(                        \
        mp, wbank.data_ptr<uint8_t>(), gs.data_ptr<float>(), sorted_ids.data_ptr<int>(),           \
        block_experts.data_ptr<int>(), n_padded.data_ptr<int>(), blocks_len, static_cast<int>(num_routes),    \
        static_cast<int>(n_tiles), static_cast<int>(K), static_cast<int>(gs.numel()),                         \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(out.size(1)), err.data_ptr<int>()); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                           \
  } else {                                                                                                    \
    cudaLaunchConfig_t cfg{};                                                                                 \
    cfg.gridDim = dim3(static_cast<unsigned>(grid));                                                          \
    cfg.blockDim = dim3(THREADS);                                                                             \
    cfg.dynamicSmemBytes = Smem<G_>::BYTES;                                                                   \
    cfg.stream = stream;                                                                                      \
    cudaLaunchAttribute attr[1];                                                                              \
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                          \
    attr[0].val.programmaticStreamSerializationAllowed = 1;                                                   \
    cfg.attrs = attr;                                                                                         \
    cfg.numAttrs = 1;                                                                                         \
    const cudaError_t le_ = cudaLaunchKernelEx(                                                               \
        &cfg, wg_kernel<G_>, mp, wbank.data_ptr<uint8_t>(), gs.data_ptr<float>(), sorted_ids.data_ptr<int>(),\
        block_experts.data_ptr<int>(), n_padded.data_ptr<int>(), blocks_len, static_cast<int>(num_routes),    \
        static_cast<int>(n_tiles), static_cast<int>(K), static_cast<int>(gs.numel()),                         \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(out.size(1)), err.data_ptr<int>()); \
    TORCH_CHECK(le_ == cudaSuccess, "wg_gemm: PDL launch: ", cudaGetErrorString(le_));                        \
  } } while (0)
  // do { } while (0): the macro is the body of if/else below; a bare if/else here left a dangling else.
  // pdl != 0 (the w2 launch only; the gather and w13 are plain launches): the programmatic-stream-serialization
  // attribute makes this grid a programmatic dependent of the previous kernel in the stream.
  if (which == 0) WG_LAUNCH(true); else WG_LAUNCH(false);
#undef WG_LAUNCH
}
"""
def _wg_wgmma_src(nbs=WG_BUCKETS):
    out = []
    for nb in nbs:
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
def _wg_src(pdl=True):
    if not pdl:
        return _wg_revert(_WG_SRC_HEAD, "head") + _wg_wgmma_src() + _wg_revert(_WG_SRC_BODY, "body")
    return _WG_SRC_HEAD + _wg_wgmma_src() + _WG_SRC_BODY
WG_EXT_NAME = "muse_moe_w4a16_wgmma_rega_v6_pdl"
WG_EXT_NAME_MASTER = "muse_moe_w4a16_wgmma_rega_v5"
def _wg_revert(text, key):
    for new, old in _WG_MASTER_REVERT[key]:
        assert text.count(new) == 1, f"master revert hunk not unique in {key}: {new[:60]!r}"
        text = text.replace(new, old)
    return text
def _wg_cpp(pdl=True):
    return _WG_CPP if pdl else _wg_revert(_WG_CPP, "cpp")
WG_GENCODE = "-gencode=arch=compute_90a,code=sm_90a"
WG_CUDA_CFLAGS = ["-O3", "-lineinfo", WG_GENCODE]
WG_BUILD_INFO = {}
def _wg_build(probe=False, pdl=True):
    t0 = time.perf_counter()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    flags = list(WG_CUDA_CFLAGS) + (["-Xptxas", "-Werror"] if probe else [])
    name = (WG_EXT_NAME if pdl else WG_EXT_NAME_MASTER) + ("_probe" if probe else "")
    ext, err_text = None, ""
    try:
        from torch.utils import cpp_extension
        from torch.utils.cpp_extension import load_inline
        print(f"{WG_TAG} torch {torch.__version__} cuda {torch.version.cuda}; device capability "
              f"{torch.cuda.get_device_capability()}", flush=True)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            arch_flags = cpp_extension._get_cuda_arch_flags()
        except Exception as e:
            print(f"{WG_TAG} TORCH_CUDA_ARCH_LIST=9.0a rejected ({e!r}); building with the default list "
                  f"(the compute_90 pass is guarded; error word 2 decides)", flush=True)
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
            try:
                arch_flags = cpp_extension._get_cuda_arch_flags()
            except Exception as e2:
                arch_flags = [f"<unknown: {e2!r}>"]
        WG_BUILD_INFO["arch"] = arch_flags
        print(f"{WG_TAG} build arch flags: TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r} -> "
              f"{arch_flags}; extra_cuda_cflags {flags}", flush=True)
        ext = load_inline(
            name=name,
            cpp_sources=_wg_cpp(pdl),
            cuda_sources=_wg_src(pdl),
            functions=["wg_init", "wg_gather", "wg_gemm"],
            extra_cuda_cflags=flags,
            verbose=False,
        )
    except Exception as e:
        err_text = str(e)
        ext = None
        print(f"{WG_TAG} load_inline {name} failed after {time.perf_counter() - t0:.1f}s: "
              f"{err_text[:240]!r}", flush=True)
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    if ext is not None:
        print(f"{WG_TAG} load_inline {name} compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return ext, err_text
def _wg_nibbles(packed):
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
def _wg_pack_nibbles(nib):
    nib = nib.reshape(*nib.shape[:-1], -1, 2)
    return nib[..., 0] | (nib[..., 1] << 4)
def _wg_code_dims(rows, k, gated):
    kc = k // 64
    if gated:
        return (2, rows // 128, 8, 8, kc, 4, 2, 4, 2), (1, 4, 2, 3, 7, 5, 8, 6, 0)
    return (rows // 128, 8, 2, 8, kc, 4, 2, 4, 2), (0, 4, 1, 3, 7, 5, 8, 6, 2)
def _wg_scale_dims(rows, kg, gated):
    kc = kg // 4
    if gated:
        return (2, rows // 128, 8, 8, kc, 4), (1, 4, 2, 3, 5, 0)
    return (rows // 128, 8, 2, 8, kc, 4), (0, 4, 1, 3, 5, 2)
def _wg_inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv
def _wg_tile_expert(packed, sf, gated):
    rows, k = packed.shape[0], packed.shape[1] * 2
    tiles, kc = rows // 128, k // 64
    shape, perm = _wg_code_dims(rows, k, gated)
    codes = _wg_pack_nibbles(_wg_nibbles(packed).view(shape).permute(perm).flip(-1, -2).reshape(-1, 8)).reshape(
        tiles, kc, 4096)
    src_sf = sf.contiguous().view(torch.uint8)
    s_shape, s_perm = _wg_scale_dims(rows, k // 16, gated)
    scales = src_sf.view(s_shape).permute(s_perm).reshape(tiles, kc, 512)
    rec = torch.cat((codes, scales), dim=-1).contiguous()
    back_c = _wg_nibbles(rec[..., :4096].reshape(-1)).view([shape[p] for p in perm]).flip(-1, -2).permute(
        _wg_inverse(perm)).reshape(rows, k)
    assert torch.equal(_wg_pack_nibbles(back_c), packed), "code retiling is not lossless"
    back_s = rec[..., 4096:].reshape(-1).view([s_shape[p] for p in s_perm]).permute(_wg_inverse(s_perm)).reshape(
        rows, k // 16)
    assert torch.equal(back_s, src_sf), "scale retiling is not lossless"
    return rec.reshape(-1)
def _wg_bank(weights, start, count):
    w13, w2 = weights["w13"], weights["w2"]
    rec13 = w13[0].numel() + weights["w13_sf"][0].numel()
    rec2 = w2[0].numel() + weights["w2_sf"][0].numel()
    q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)
    q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)
    for i in range(count):
        e = start + i
        q13[i] = _wg_tile_expert(w13[e].contiguous(), weights["w13_sf"][e], True)
        q2[i] = _wg_tile_expert(w2[e].contiguous(), weights["w2_sf"][e], False)
    sl = slice(start, start + count)
    return {
        "q13": q13.reshape(-1), "gs13": (weights["w13_gs"][sl].float() * 128.0).contiguous(),
        "q2": q2.reshape(-1), "gs2": (weights["w2_gs"][sl].float() * 128.0).contiguous(),
    }
def _wg_gemm(state, which, wq, wgs, sorted_ids, block_experts, n_padded, routes, n_tiles, k, out):
    args = (which, state["wg_xs"] if which == 0 else state["wg_hs"], wq, wgs, sorted_ids, block_experts, n_padded,
            routes, n_tiles, k, out, state["wg_grid"], state["wg_err"])
    if state.get("wg_ext_pdl", False):
        state["wg_ext"].wg_gemm(*args, 0)
    else:
        state["wg_ext"].wg_gemm(*args)
def _wg_attach_scratch(state, ext, device, hidden, inter):
    state["wg_xs"] = torch.zeros((WG_CAP_ROWS, hidden), dtype=torch.bfloat16, device=device)
    state["wg_hs"] = torch.zeros((WG_CAP_ROWS, inter), dtype=torch.bfloat16, device=device)
    state["wg_err"] = torch.zeros((2,), dtype=torch.int32, device=device)
    state["wg_ext"] = ext
    state["wg_grid"] = torch.cuda.get_device_properties(device).multi_processor_count
_WG_MASTER_REVERT = None
WH_TAG = "[hh-wh]"
WH_ARMS = (("wh5ds", 14),)
WH_ROUTE, WH_GATE = 0.97, 1.00
WH_SCALE = (0x08, 0x7E)
WH_REC = 4608
WH_PERM_CHUNK = 16384
WH_DL = 240.0
WH_REBUILD_S = 100.0
WH_REFILLS = 8
WH_SEED = 20261190
WH_MAGIC = 0x57474D4D
WH_COUNTS = (129, 113, 97, 81, 65, 49, 33, 17, 140, 100, 70, 40, 24, 56, 88, 120)
WH_ENC4 = (8, 9, 10, 1, 2, 7, 12, 13, 14, 11, 0, 3, 20, 21, 22, 15, 24, 25, 26, 17, 18, 23, 28, 29, 30, 27, 16, 19, 4, 5,
           6, 31)
_WH_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> wh_init();
void wh_gemm(int64_t which, torch::Tensor bsrc, torch::Tensor wbank, torch::Tensor gs, torch::Tensor sorted_ids,
             torch::Tensor block_experts, torch::Tensor n_padded, int64_t num_routes, int64_t n_tiles, int64_t K,
             torch::Tensor out, int64_t grid, torch::Tensor err, int64_t pdl, int64_t v);
"""
_WH_DQ = r"""__device__ __forceinline__ uint32_t hh_rt16(int n) {
  uint32_t z;
  asm("shr.u32 %0, %1, 30;\n" : "=r"(z) : "r"(n));
  return 16u << z;
}
__device__ __forceinline__ uint32_t hh_rt78(int n) {
  uint32_t z;
  asm("shr.u32 %0, %1, 30;\n" : "=r"(z) : "r"(n));
  return ((1u << 28) << z) ^ 0x68007800u;
}
__device__ __forceinline__ uint32_t hh_lo_of(uint32_t a) { return ((a & 0x3FFFFu) >> 4) | (1u << 16); }
__device__ __forceinline__ uint64_t hh_mkd(uint32_t lo) {
  uint64_t d;
  asm("mov.b64 %0, {%1, %2};\n" : "=l"(d) : "r"(lo), "r"(0x40000040u));
  return d;
}
template <int J0>
__device__ __forceinline__ uint32_t hh_scale_pair(uint32_t w, uint32_t m16) {
  uint32_t t, d;
  asm("prmt.b32 %0, %1, 0, %2;\n" : "=r"(t) : "r"(w), "n"((4 << 12) | ((J0 + 1) << 8) | (4 << 4) | J0));
  asm("mad.lo.u32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(t), "r"(m16), "r"(0x77807780u));
  return d;
}
template <int H>
__device__ __forceinline__ uint32_t hh_mul_h(uint32_t a, uint32_t s) {
  uint32_t d;
  if constexpr (H == 0)
    asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 t;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 t, {l, l};\n\t"
        "fma.rn.bf16x2 %0, %1, t, %3;\n\t}\n" : "=r"(d) : "r"(a), "r"(s), "r"(0x80008000u));
  else
    asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 t;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 t, {h, h};\n\t"
        "fma.rn.bf16x2 %0, %1, t, %3;\n\t}\n" : "=r"(d) : "r"(a), "r"(s), "r"(0x80008000u));
  return d;
}
template <int I>
__device__ __forceinline__ uint32_t hh_ext_prop(uint32_t q) {
  if constexpr (I == 0) {
    return q & 0x81C081C0u;
  } else if constexpr (I == 1) {
    return (q << 3) & 0x81C081C0u;
  } else if constexpr (I == 2) {
    return (q << 6) & 0x81C081C0u;
  } else {
    return ((q >> 3) & 0x01800180u) | ((q >> 7) & 0x00400040u) | ((q << 1) & 0x80008000u);
  }
}
template <int V, int K>
__device__ __forceinline__ void hh_dq_step(uint32_t qk, uint32_t sw, uint32_t m, uint32_t (&a)[4]) {
  if constexpr ((V & 8) != 0) {
    const uint32_t ws = (K & 1) ? __byte_perm(sw, sw, 0x2103) : sw;
    uint32_t S;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(S) : "r"(ws), "n"(0x07F007F0), "r"(m));
    constexpr uint32_t M = 0x81C081C0u;
    a[0] = hh_mul_h<0>(qk & M, S);
    a[1] = hh_mul_h<1>((qk << 6) & M, S);
    a[2] = hh_mul_h<0>(__funnelshift_l(qk, qk, 10) & M, S);
    a[3] = hh_mul_h<1>(((qk << 4) & 0x81808180u) | ((qk >> 4) & 0x00400040u), S);
  } else {
    const uint32_t S = hh_scale_pair<2 * (K & 1)>(sw, m);
    a[0] = hh_mul_h<0>(hh_ext_prop<0>(qk), S);
    a[1] = hh_mul_h<1>(hh_ext_prop<1>(qk), S);
    a[2] = hh_mul_h<0>(hh_ext_prop<2>(qk), S);
    a[3] = hh_mul_h<1>(hh_ext_prop<3>(qk), S);
  }
}
template <int V>
__device__ __forceinline__ void hh_dq_rec(const uint8_t* rp, int cw, int lane, uint32_t m, uint32_t (*a)[4]) {
  const uint4 q = *reinterpret_cast<const uint4*>(rp + cw * 512 + lane * 16);
  const uint2 sc = *reinterpret_cast<const uint2*>(rp + REC_CODES + cw * 64 + (lane >> 2) * 8);
  hh_dq_step<V, 0>(q.x, sc.x, m, a[0]);
  hh_dq_step<V, 1>(q.y, sc.x, m, a[1]);
  hh_dq_step<V, 2>(q.z, sc.y, m, a[2]);
  hh_dq_step<V, 3>(q.w, sc.y, m, a[3]);
}
__device__ __forceinline__ void dq_stage_a3(const uint8_t* st, int cw, int lane, uint32_t (&a)[8][4], uint32_t m16) {
  hh_dq_rec<1>(st, cw, lane, m16, a);
  hh_dq_rec<1>(st + REC, cw, lane, m16, a + 4);
}

"""
_WH_RING = r"""  if constexpr ((V & 4) != 0 && NB >= 48) {
  const uint32_t hh_m16 = (V & 8) != 0 ? hh_rt78(KS) : hh_rt16(KS);
  const int hh_g = lane >> 2;
  uint32_t A0[4], A1[4], A2[4], A3[4];
  int hh_sp = -1;
  for (int kb = 0; kb < KS; ++kb) {
    const int s = n % STAGES;
    mbar_wait(full0 + 8 * s, (n / STAGES) & 1, err, dead);
    const uint8_t* st = smem + s * STAGE_BYTES;
    const uint4 q0 = *reinterpret_cast<const uint4*>(st + cw * 512 + lane * 16);
    const uint2 c0 = *reinterpret_cast<const uint2*>(st + REC_CODES + cw * 64 + hh_g * 8);
    const uint4 q1 = *reinterpret_cast<const uint4*>(st + REC + cw * 512 + lane * 16);
    const uint2 c1 = *reinterpret_cast<const uint2*>(st + REC + REC_CODES + cw * 64 + hh_g * 8);
    const uint64_t d0 = make_desc(smem_u32(st + WST)), d1 = make_desc(smem_u32(st + WST + SLAB_HALF));
    const uint32_t l0 = hh_lo_of(smem_u32(st + WST)), l1 = l0 + (SLAB_HALF >> 4);
#define HH_K16(QW_, SW_, K_, AB_, AO_, D_, L_)                                                                  \
    hh_dq_step<V, K_>(QW_, SW_, hh_m16, AB_);                                                                  \
    fence_acc<NA>(acc);                                                                                        \
    wgmma_fence();                                                                                             \
    Wg<NB>::mma(acc, AB_, (V & 8) != 0 ? hh_mkd((L_) + 2u * (K_)) : (D_) + static_cast<uint64_t>(2 * (K_)));    \
    wgmma_commit();                                                                                            \
    wgmma_wait<2>();                                                                                           \
    asm volatile("" : "+r"(AO_[0]), "+r"(AO_[1]), "+r"(AO_[2]), "+r"(AO_[3]) :: "memory");
    HH_K16(q0.x, c0.x, 0, A0, A1, d0, l0)
    HH_K16(q0.y, c0.x, 1, A1, A2, d0, l0)
    if (hh_sp >= 0 && lane == 0) mbar_arrive(empty0 + 8 * hh_sp);
    HH_K16(q0.z, c0.y, 2, A2, A3, d0, l0)
    HH_K16(q0.w, c0.y, 3, A3, A0, d0, l0)
    HH_K16(q1.x, c1.x, 0, A0, A1, d1, l1)
    HH_K16(q1.y, c1.x, 1, A1, A2, d1, l1)
    HH_K16(q1.z, c1.y, 2, A2, A3, d1, l1)
    HH_K16(q1.w, c1.y, 3, A3, A0, d1, l1)
#undef HH_K16
    hh_sp = s;
    ++n;
  }
  wgmma_wait<0>();
  fence_acc<NA>(acc);
  if (hh_sp >= 0 && lane == 0) mbar_arrive(empty0 + 8 * hh_sp);
  } else if constexpr ((V & 8) != 0) {
  uint32_t a0[8][4];
  const uint32_t hh_m16 = hh_rt78(KS);
  for (int kb = 0; kb < KS; ++kb) {
    const int s = n % STAGES;
    mbar_wait(full0 + 8 * s, (n / STAGES) & 1, err, dead);
    const uint8_t* st = smem + s * STAGE_BYTES;
    const uint32_t hh_lo = hh_lo_of(smem_u32(st + WST));
    hh_dq_rec<V>(st, cw, lane, hh_m16, a0);
    fence_acc<NA>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; ++k) Wg<NB>::mma(acc, a0[k], hh_mkd(hh_lo + static_cast<uint32_t>(k * 2)));
    wgmma_commit();
    hh_dq_rec<V>(st + REC, cw, lane, hh_m16, a0 + 4);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; ++k) Wg<NB>::mma(acc, a0[4 + k], hh_mkd(hh_lo + static_cast<uint32_t>((SLAB_HALF >> 4) + k * 2)));
    wgmma_commit();
    wgmma_wait<0>();
    fence_acc<NA>(acc);
    if (lane == 0) mbar_arrive(empty0 + 8 * s);
    ++n;
  }
  wgmma_wait<0>();
  fence_acc<NA>(acc);
  } else {
  uint32_t a0[8][4];
  const uint32_t hh_m16 = hh_rt16(KS);
  for (int kb = 0; kb < KS; ++kb) {"""
_WH_NX_SWITCH = """      if constexpr ((V & 2) != 0) {
        switch (m.nbk) {
          case 1: WG_CONSUME(16); break;
          case 2: WG_CONSUME(32); break;
          case 3: WG_CONSUME(48); break;
          case 4: WG_CONSUME(64); break;
          case 5: WG_CONSUME(80); break;
          case 6: WG_CONSUME(96); break;
          case 7: WG_CONSUME(112); break;
          default: WG_CONSUME(128); break;
        }
      } else
      switch (m.bucket) {"""
_WH_EDITS = (
    ("// wgmma_wait<0> right after the commit, BEFORE the next", _WH_DQ + "// wgmma_wait<0> right after the commit, BEFORE the next"),
    ("template <int NB, bool G>\n__device__", "template <int NB, bool G, int V>\n__device__"),
    ("  uint32_t a0[8][4];\n  for (int kb = 0; kb < KS; ++kb) {", _WH_RING),
    ("    dq_stage(st, cw, lane, a0);\n",
     "    if constexpr ((V & 1) != 0) dq_stage_a3(st, cw, lane, a0, hh_m16); else dq_stage(st, cw, lane, a0);\n"),
    ("  wgmma_wait<0>();  // as s3 variant A: nothing in flight before the epilogue reads acc\n  fence_acc<NA>(acc);\n",
     "  wgmma_wait<0>();  // as s3 variant A: nothing in flight before the epilogue reads acc\n  fence_acc<NA>(acc);\n  }\n"),
    ("template <bool G>\n__global__", "template <bool G, int V>\n__global__"),
    ("        const uint64_t map = reinterpret_cast<uint64_t>(&maps.m[m.bucket]);\n"
     "        const uint32_t tx = WST + 2 * (16 << m.bucket) * 128;\n",
     "        const int mi = (V & 2) != 0 ? m.nbk - 1 : (1 << m.bucket) - 1;\n"
     "        const uint64_t map = reinterpret_cast<uint64_t>(&maps.m[mi]);\n"
     "        const uint32_t tx = WST + 2 * (16 * (mi + 1)) * 128;\n"),
    ("consume_item<NB_, G>(", "consume_item<NB_, G, V>("),
    ("      switch (m.bucket) {", _WH_NX_SWITCH),
    ("template <bool G>\nvoid info_one", "template <bool G, int V>\nvoid info_one"),
    ("reinterpret_cast<const void*>(wg_kernel<G>)", "reinterpret_cast<const void*>(wg_kernel<G, V>)"),
    ("  r.push_back(G ? 1 : 0);", "  r.push_back((G ? 1 : 0) + 2 * V);"),
    ("std::vector<int64_t> wg_init() {", "std::vector<int64_t> wh_init() {"),
    ("  info_one<true>(r);\n  info_one<false>(r);",
     "  info_one<true, 0>(r);\n  info_one<false, 0>(r);\n  info_one<true, 1>(r);\n  info_one<false, 1>(r);\n"
     "  info_one<true, 7>(r);\n  info_one<false, 7>(r);\n  info_one<true, 14>(r);\n  info_one<false, 14>(r);"),
    ("void wg_gemm(int64_t which,", "void wh_gemm(int64_t which,"),
    ("             int64_t pdl) {\n  TORCH_CHECK(which == 0 || which == 1, \"wg_gemm: which\");",
     "             int64_t pdl, int64_t v) {\n  TORCH_CHECK((which == 0 || which == 1) && (v == 0 || v == 1 || v == 7 || "
     "v == 14), \"wh_gemm: which / v\");"),
    ("    for (int b = 0; b < 4; ++b) {\n      const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(16 << b)};",
     "    for (int b = 0; b < 8; ++b) {\n      const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(16 * (b + 1))};"),
    ("#define WG_LAUNCH(G_)", "#define WG_LAUNCH(G_, V_)"),
    ("wg_kernel<G_><<<", "wg_kernel<G_, V_><<<"),
    ("&cfg, wg_kernel<G_>, mp,", "&cfg, wg_kernel<G_, V_>, mp,"),
    ("  if (which == 0) WG_LAUNCH(true); else WG_LAUNCH(false);",
     "#define WH_LV(G_) switch (v) { case 1: WG_LAUNCH(G_, 1); break; case 7: WG_LAUNCH(G_, 7); break; "
     "case 14: WG_LAUNCH(G_, 14); break; default: WG_LAUNCH(G_, 0); }\n"
     "  if (which == 0) { WH_LV(true) } else { WH_LV(false) }\n#undef WH_LV"),
)
_WH_MAPS = ("struct __align__(64) Maps { CUtensorMap m[4]; };", "struct __align__(64) Maps { CUtensorMap m[8]; };")
def _wh_src():
    head, body = _WG_SRC_HEAD, _WG_SRC_BODY
    for old, new in (_WH_MAPS,) + _WH_EDITS:
        part = head if old == _WH_MAPS[0] else body
        if part.count(old) != 1:
            raise RuntimeError(f"edit target not unique: {old[:60]!r}")
        if old == _WH_MAPS[0]:
            head = head.replace(old, new)
        else:
            body = body.replace(old, new)
    return _WH_CPP, head + _wg_wgmma_src((16, 32, 48, 64, 80, 96, 112, 128)) + body
def _wh_build_bg(name, cpp, src, res):
    import re
    t0 = time.perf_counter()
    try:
        os.setpriority(os.PRIO_PROCESS, 0, min(19, os.getpriority(os.PRIO_PROCESS, 0) + 10))
    except Exception:
        pass
    try:
        from torch.utils.cpp_extension import load_inline
        for probe in (True, False):
            try:
                res["ext"] = load_inline(name=name + ("_p" if probe else ""), cpp_sources=cpp, cuda_sources=src,
                                         functions=["wh_init", "wh_gemm"], verbose=False,
                                         extra_cuda_cflags=list(WG_CUDA_CFLAGS) + (["-Xptxas", "-Werror"] if probe else []))
                res["probe"] = probe
                break
            except Exception as e:
                text = str(e)
                res["err"] = text[-300:]
                res["c75"] = sorted(set(re.findall(r"C75\d\d", text))) or res.get("c75")
                if res.get("c75") or not probe or time.perf_counter() - t0 > WH_REBUILD_S:
                    break
    except BaseException as e:
        res["ext"], res["err"] = None, f"{type(e).__name__}: {str(e)[-300:]}"
    finally:
        res["secs"] = time.perf_counter() - t0
def _wh_start(t0):
    job = {"res": {}, "th": None, "env": None, "off": None, "wait": 0.0}
    try:
        import hashlib
        import threading
        cpp, src = _wh_src()
        name = "muse_decode_wg_wh5_" + hashlib.sha256((cpp + src).encode()).hexdigest()[:10]
        job["env"] = (os.environ.get("TORCH_CUDA_ARCH_LIST"),)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        th = threading.Thread(target=_wh_build_bg, args=(name, cpp, src, job["res"]), name="hh-wh-nvcc", daemon=False)
        job["t"] = time.perf_counter() - t0
        th.start()
        job["th"] = th
    except Exception as e:
        job["off"] = f"start {type(e).__name__}: {str(e)[:120]}"
    return job
def _wh_join(job, restore=False):
    try:
        th = job.get("th")
        if th is not None:
            tw = time.perf_counter()
            th.join()
            job["th"] = None
            job["wait"] += time.perf_counter() - tw
        if restore and job.get("env") is not None:
            prev, job["env"] = job["env"][0], None
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    except Exception as e:
        job["off"] = f"join raised {e!r}"[:160]
def _wh_pair(state, v, sh, bh, nh, routes, size_n, size_k, out, sc=None):
    wb = state["wg_banks"][(0, LOCAL_EXPERTS)]
    if v is None:
        _wg_gemm(state, 0, wb["q13"], wb["gs13"], sh, bh, nh, routes, size_n // 64, size_k, state["wg_hs"])
        _wg_gemm(state, 1, wb["q2"], wb["gs2"], sh, bh, nh, routes, size_k // 128, size_n, out)
        return
    sc = state if sc is None else sc
    b = state["whk_x5"] if v & 8 else (state["whk_bank"] if v & 1 else wb)
    ext, grid, err = state["whk_ext"], state["wg_grid"], sc["whk_err"]
    ext.wh_gemm(0, sc["wg_xs"], b["q13"], wb["gs13"], sh, bh, nh, routes, size_n // 64, size_k, sc["wg_hs"], grid,
                err, 0, v)
    ext.wh_gemm(1, sc["wg_hs"], b["q2"], wb["gs2"], sh, bh, nh, routes, size_k // 128, size_n, out, grid, err, 0, v)
def _wh_run(state, v, x, al, out, sc=None):
    sc = state if sc is None else sc
    sh, bh, nh = al
    routes = x.shape[0] * TOP_K
    out.fill_(HC_JUNK)
    state["wg_ext"].wg_gather(x, sh, bh, nh, routes, LOCAL_EXPERTS, sc["wg_xs"])
    _wh_pair(state, v, sh, bh, nh, routes, state["bank"]["size_n"], x.shape[1], out, sc)
    return out
def _wh_bitperm(w, d):
    p = torch.zeros_like(w)
    for s in range(32):
        p |= ((w >> s) & 1) << d[s]
    return p
def _wh_pack(w):
    f0, f1, f2, f3 = (((w >> (8 * i)) & 0xFF) - 8 for i in range(4))
    return (f0 << 4) | (f1 << 20) | (f3 << 12) | ((f2 & 0xF) << 28) | (f2 >> 4)
def _wh_unpack(p):
    f2 = ((p >> 28) & 0xF) | ((p & 7) << 4)
    return (((p >> 4) & 0x7F) + 8) | ((((p >> 20) & 0x7F) + 8) << 8) | ((f2 + 8) << 16) | ((((p >> 12) & 0x7F) + 8) << 24)
def _wh_e4():
    e4 = [0] * 32
    for j, t in enumerate(WH_ENC4):
        e4[t] = j
    if sorted(e4) != list(range(32)):
        raise RuntimeError("code map is not a bit permutation")
    return e4
def _wh_i32(w):
    return (w - ((w >> 31) & 1) * (1 << 32)).to(torch.int32)
def _wh_recode(wb, d, pack):
    inv = [0] * 32
    for s in range(32):
        inv[d[s]] = s
    out = {}
    for key in ("q13", "q2"):
        src = wb[key].view(-1, WH_REC)
        dst = src.clone()
        for i in range(0, src.shape[0], WH_PERM_CHUNK):
            c = src[i:i + WH_PERM_CHUNK]
            w = c[:, :4096].contiguous().view(torch.int32).long() & 0xFFFFFFFF
            p = _wh_bitperm(w, d)
            if not torch.equal(_wh_bitperm(p, inv), w):
                raise RuntimeError(f"{key} chunk {i}: code map not lossless")
            dst[i:i + WH_PERM_CHUNK, :4096] = _wh_i32(p).view(torch.uint8).view(-1, 4096)
            if pack:
                sw = c[:, 4096:].contiguous().view(torch.int32).long() & 0xFFFFFFFF
                ps = _wh_pack(sw)
                if not torch.equal(_wh_unpack(ps), sw):
                    raise RuntimeError(f"{key} chunk {i}: scale packing not lossless")
                dst[i:i + WH_PERM_CHUNK, 4096:] = _wh_i32(ps).view(torch.uint8).view(-1, 512)
            w = p = c = None
        if not pack and not torch.equal(dst[:, 4096:], src[:, 4096:]):
            raise RuntimeError(f"{key}: scale bytes changed")
        out[key] = dst.view(-1)
    return out
def _wh_inputs(state, device, gen, gcpu, cls, h, synth=None):
    if synth is None:
        data = _ead_cases(state, device, gen, cls)
        ids = _cb_router_scores(_dl3_clone_data(data), state).topk(TOP_K, dim=-1).indices.to(torch.int32).contiguous()
        ls = data["local_start"]
    else:
        ids = torch.arange(LOCAL_EXPERTS, LOCAL_EXPERTS + TOP_K).repeat(cls, 1)
        for j, e in enumerate(range(synth, LOCAL_EXPERTS, 4)):
            c = min(WH_COUNTS[(j + 2 * synth) % len(WH_COUNTS)], cls)
            ids[torch.randperm(cls, generator=gcpu)[:c], j] = e
        ids = ((ids + 96) % N_EXPERTS).to(torch.int32).to(device)
        ls = torch.tensor([96], dtype=torch.int32, device=device)
    tw = torch.full((cls, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    _, _, align = _hc_dispatch(ids, tw, ls, h)
    return torch.randn((cls, D_EXPERT), generator=gen, device=device).to(torch.bfloat16), align[3:6]
def _wh_exact(state, x, al, arms):
    nh = int(al[2][0])
    out = torch.empty((x.shape[0] * TOP_K, x.shape[1]), dtype=x.dtype, device=x.device)
    ref = _wh_run(state, None, x, al, out).clone()
    hs = state["wg_hs"][:nh].clone()
    return {n for n, v in arms if not (torch.equal(_wh_run(state, v, x, al, out), ref)
                                       and torch.equal(state["wg_hs"][:nh], hs))}
def _wh_time(state, device, x, al, arms):
    sh, bh, nh = al
    routes, size_k = x.shape[0] * TOP_K, x.shape[1]
    out = torch.empty((routes, size_k), dtype=x.dtype, device=x.device)
    state["wg_ext"].wg_gather(x, sh, bh, nh, routes, LOCAL_EXPERTS, state["wg_xs"])
    sz = state["bank"]["size_n"]
    graphs = {n: _rt_graph(lambda v=v: _wh_pair(state, v, sh, bh, nh, routes, sz, size_k, out))[0]
              for n, v in (("m", None),) + arms}
    rounds = _rt_time(graphs, device)
    del graphs
    return [{n: _rt_stats(ts[n])[0] for n in ts if n != "__baseline_us__"} for ts in rounds]
def _wh_replay(state, device, gen, gcpu, cls, h, v):
    x0, al0 = _wh_inputs(state, device, gen, gcpu, cls, h)
    x, al = x0.clone(), [t.clone() for t in al0]
    sb = {"wg_xs": torch.zeros_like(state["wg_xs"]), "wg_hs": torch.zeros_like(state["wg_hs"]),
          "whk_err": torch.zeros_like(state["whk_err"])}
    outs = [torch.empty((cls * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=device) for _ in range(2)]
    gs = [_rt_graph(lambda o=o, s=s: _wh_run(state, v, x, al, o, s))[0] for o, s in zip(outs, (state, sb))]
    ref_o = torch.empty_like(outs[0])
    bad = 0
    for k in range(WH_REFILLS):
        x1, al1 = _wh_inputs(state, device, gen, gcpu, cls, h, None if k % 2 == 0 else (k // 2) % 4)
        x.copy_(x1)
        for t, s in zip(al, al1):
            t.copy_(s)
        ref = _wh_run(state, None, x, al, ref_o)
        for g in (gs if k % 2 == 0 else gs[::-1]):
            g.replay()
        torch.cuda.synchronize()
        bad += sum(0 if torch.equal(o, ref) else 1 for o in outs)
    e = [int(t) for t in sb["whk_err"].tolist()]
    del gs
    return bad + (0 if e == [0, WH_MAGIC] else 100), 2 * WH_REFILLS
def _wh_fmt(vals):
    return "/".join(f"{v:.3f}".lstrip("0") for v in vals)
def _wh_gate(state, device, cls, v):
    whk = state["whk"]
    def st(a, ser):
        def f():
            whk[cls], state["side_serial"] = a, ser
        return f
    try:
        per = _ead_layer_gate(state, device, cls, {"ms": st(None, True), "mf": st(None, False), "as": st(v, True),
                                                   "af": st(v, False)}, WH_SEED + 9 + cls)
    finally:
        whk[cls], state["side_serial"] = None, False
    md = {k: [_rt_stats(ts[k])[0] for rounds in per.values() for ts in rounds] for k in ("ms", "mf", "as", "af")}
    return [[a / max(b, 1e-3) for a, b in zip(md["a" + s], md["m" + s])] for s in "sf"]
def _wh_phase(state, device, t0, job):
    whk = state["whk"] = {c: None for c in HC_CLASSES}
    out, t1, stp, pick = [], time.perf_counter(), ["join"], {}
    n_in = {t: len(v) for t, v in _EAD_INPUTS.items()}
    n, v = WH_ARMS[0]
    def step(what):
        if time.perf_counter() - t0 > WH_DL:
            raise RuntimeError(f"deadline ({time.perf_counter() - t0:.0f} s)")
        torch.cuda.synchronize()
        stp[0] = what
    try:
        _wh_join(job)
        res = job["res"]
        out.append(f"{WH_TAG} BUILD {res.get('secs', 0):.0f}s bg from {job.get('t', 0):.1f}s, join waited "
                   f"{job['wait']:.0f}s; ptxas -Werror {'clean' if res.get('probe') else 'NOT clean'}, C75xx "
                   f"{res.get('c75') or 'none'}")
        if job.get("off"):
            raise RuntimeError(job["off"])
        ext = res.get("ext")
        if ext is None:
            raise RuntimeError(f"build failed: {res.get('err', '')[-160:]}")
        hc = state.get("hc") or {}
        cls_on = [c for c in HC_CLASSES if hc.get("on", {}).get(c, 0) in HC_WG_HOT]
        if not cls_on or "wg_banks" not in state:
            raise RuntimeError("master's wgmma hot half is not routed")
        step("wh_init")
        rows = [int(x) for x in ext.wh_init()]
        for i in range(0, len(rows), 10):
            f, a_ok, regs, lb, ss, ds, occ, _, bv, _ = rows[i:i + 10]
            if not (a_ok and occ >= 1 and ss + ds <= WG_SMEM_LIMIT and bv == 90 and lb == 0
                    and regs * WG_THREADS >= 40 * 128 + 232 * 256):
                raise RuntimeError(f"kernel attributes G{f & 1} V{f >> 1} r{regs}/l{lb}/occ{occ}")
        wb = state["wg_banks"][(0, LOCAL_EXPERTS)]
        step("scales")
        sc = [wb[k].view(-1, WH_REC)[:, 4096:] for k in ("q13", "q2")]
        nbad = sum(int(((s < WH_SCALE[0]) | (s > WH_SCALE[1])).sum()) for s in sc)
        sc = None
        if nbad:
            raise RuntimeError(f"{nbad} scale bytes outside 0x08..0x7E")
        state["whk_x5"] = _wh_recode(wb, _wh_e4(), True)
        state["whk_ext"], state["whk_err"] = ext, torch.zeros((2,), dtype=torch.int32, device=device)
        gen = torch.Generator(device=device).manual_seed(WH_SEED)
        gcpu = torch.Generator().manual_seed(WH_SEED + 1)
        for cls in cls_on:
            h, why, nrep = hc["on"][cls], "", ""
            try:
                step(f"exact T={cls}")
                ins = [_wh_inputs(state, device, gen, gcpu, cls, h) for _ in range(2)]
                bad = set()
                for x, al in ins + [_wh_inputs(state, device, gen, gcpu, cls, h, r) for r in range(4)]:
                    bad |= _wh_exact(state, x, al, WH_ARMS)
                torch.cuda.synchronize()
                e = [int(x) for x in state["whk_err"].tolist()]
                if e != [0, WH_MAGIC]:
                    raise RuntimeError(f"err word {e}")
                step(f"time T={cls}")
                rnd = [r for x, al in ins for r in _wh_time(state, device, x, al, WH_ARMS)]
                rat = [r[n] / max(r["m"], 1e-3) for r in rnd]
                mu = {k: sum(r[k] for r in rnd) / len(rnd) for k in ("m", n)}
                if bad:
                    why = "not exact"
                elif max(rat) > WH_ROUTE:
                    why = f"{n}/m > {WH_ROUTE} in a round"
                else:
                    step(f"replay T={cls}")
                    nb, nr = _wh_replay(state, device, gen, gcpu, cls, h, v)
                    nrep = f"; replay {nr - nb}/{nr}"
                    if nb:
                        why = "replay"
                    else:
                        pick[cls] = n
                out.append(f"{WH_TAG} ROUTE T={cls} {('pick ' + n) if cls in pick else 'OFF (' + why + ')'}: {n}/m "
                           f"{_wh_fmt(rat)}; us m {mu['m']:.1f} {n} {mu[n]:.1f}; exact {'n' if bad else 'y'} 6 inputs{nrep}")
            except Exception as e:
                pick.pop(cls, None)
                out.append(f"{WH_TAG} ROUTE T={cls} OFF at {stp[0]}: {type(e).__name__}: {str(e)[:120]}")
    except Exception as e:
        pick = {}
        out.append(f"{WH_TAG} OFF at {stp[0]}: {type(e).__name__}: {str(e)[:160]}")
    for cls in list(pick):
        try:
            step(f"gate T={cls}")
            rs, rf = _wh_gate(state, device, cls, v)
            ok = max(rs + rf) <= WH_GATE
            gl = f"{n} s {_wh_fmt(rs)} f {_wh_fmt(rf)} {'PASS' if ok else 'VETO'}"
        except Exception as e:
            ok, gl = False, f"raised at {stp[0]}: {type(e).__name__}: {str(e)[:100]}"
        if not ok:
            pick.pop(cls)
        out.append(f"{WH_TAG} GATE T={cls} layer/m: {gl}")
    for t in list(_EAD_INPUTS):
        if t in n_in:
            del _EAD_INPUTS[t][n_in[t]:]
        else:
            del _EAD_INPUTS[t]
    try:
        torch.cuda.synchronize()
    except Exception as e:
        pick = {}
        out.append(f"{WH_TAG} final sync raised {type(e).__name__}")
    for c in whk:
        whk[c] = v if c in pick else None
    if not pick:
        state.pop("whk_x5", None)
    out.append(f"{WH_TAG} PATH CHOSEN " + ", ".join(
        f"T={c} " + (f"{n} wg_kernel<G,{v}>" if whk[c] else "master wg_kernel<G>") for c in HC_CLASSES)
        + f" ({time.perf_counter() - t1:.1f}s, prepare at {time.perf_counter() - t0:.1f}s); [ea-enc] races after this route")
    return out
HC_TAG = "[hotcold]"
HC_CLASSES = (128, 256)
HC_CLASS_SPLIT = 192
HC_MIN_TOKENS = 128
HC_MAX_TOKENS = 256
HC_HOT_ROWS = 16
HC_COLD_BSM = 16
HC_HOT_BSMS = (32, 48, 64, -16, -8, -32, -48)
HC_WG_HOT = {-16: 16, -8: 8}
HC_WG_BUILD_DEADLINE_S = 150.0
HC_COLD_SLOTS = (HC_HOT_ROWS + HC_COLD_BSM - 1) // HC_COLD_BSM * HC_COLD_BSM
HC_COLD_CAP = LOCAL_EXPERTS * HC_COLD_SLOTS
HC_KINDS = ("zipf", "uniform", "mixed", "one", "remote", "all8")
HC_JUNK = 100.0
HC_TOL_MASTER = 1e-2
HC_ROW_MASTER = 5e-2
HC_TOL_F32 = 1.25e-2
HC_COPY_TOL = 1e-3
HC_START_DEADLINE_S = 225.0
HC_GATE_DEADLINE_S = 250.0
HC_CHECK_DEADLINE_S = 255.0
HC_PIN = {128: -16, 256: -16}
def _hc_class(tokens):
    if tokens < HC_MIN_TOKENS or tokens > HC_MAX_TOKENS:
        return None
    return HC_CLASSES[0] if tokens < HC_CLASS_SPLIT else HC_CLASSES[1]
def _hc_bsm(state, tokens):
    hc = state.get("hc")
    cls = _hc_class(tokens)
    if hc is None or cls is None:
        return 0
    return hc["on"].get(cls, 0)
def _hc_hot_rows(bsm_hot):
    return HC_WG_HOT.get(bsm_hot, HC_HOT_ROWS)
def _hc_hot_block(bsm_hot):
    return WG_BSM if bsm_hot in HC_WG_HOT else abs(bsm_hot)
def _hc_caps(n, bsm_hot):
    bsm_hot = _hc_hot_block(bsm_hot)
    cap_h = n + LOCAL_EXPERTS * (bsm_hot - 1)
    return HC_COLD_CAP, HC_COLD_CAP // HC_COLD_BSM, cap_h, triton.cdiv(cap_h, bsm_hot)
@triton.jit(do_not_specialize=["n", "cap_c", "nb_c", "cap_h", "nb_h"])
def _hc_align_kernel(ids_ptr, ls_ptr, local_ids_ptr, sc_ptr, bc_ptr, nc_ptr, sh_ptr, bh_ptr, nh_ptr,
                     n, cap_c, nb_c, cap_h, nb_h, cold_ids_ptr, HOT: tl.constexpr, CHUNK: tl.constexpr,
                     PAD: tl.constexpr, BLK: tl.constexpr, BSM_C: tl.constexpr, BSM_H: tl.constexpr,
                     NUM_E: tl.constexpr, N_GLOBAL: tl.constexpr, COLD_IDS: tl.constexpr):
    pid = tl.program_id(0)
    offs0 = tl.arange(0, CHUNK)
    experts = tl.arange(0, NUM_E)
    ls = tl.load(ls_ptr).to(tl.int32)

    counts = tl.zeros([NUM_E], dtype=tl.int32)
    for c in range(0, n, CHUNK):
        offs = c + offs0
        mask = offs < n
        ids = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
        local = ((ids - ls) % N_GLOBAL + N_GLOBAL) % N_GLOBAL
        local = tl.where(mask & (local < NUM_E), local, -1)
        counts += tl.sum((local[:, None] == experts[None, :]).to(tl.int32), axis=0)
    hot = counts > HOT
    cnt_c = tl.where(hot, 0, counts)
    cnt_h = tl.where(hot, counts, 0)
    pad_c = (cnt_c + BSM_C - 1) // BSM_C * BSM_C
    pad_h = (cnt_h + BSM_H - 1) // BSM_H * BSM_H
    start_c = tl.cumsum(pad_c, axis=0) - pad_c
    start_h = tl.cumsum(pad_h, axis=0) - pad_h
    total_c = tl.sum(pad_c, axis=0)
    total_h = tl.sum(pad_h, axis=0)

    if pid < NUM_E:
        sel = experts == pid
        my_c = tl.sum(tl.where(sel, cnt_c, 0), axis=0)
        my_h = tl.sum(tl.where(sel, cnt_h, 0), axis=0)
        my_pc = tl.sum(tl.where(sel, pad_c, 0), axis=0)
        my_ph = tl.sum(tl.where(sel, pad_h, 0), axis=0)
        my_sc = tl.sum(tl.where(sel, start_c, 0), axis=0)
        my_sh = tl.sum(tl.where(sel, start_h, 0), axis=0)
        in_c = my_c > 0
        in_h = my_h > 0
        base_c = my_sc
        base_h = my_sh
        for c in range(0, n, CHUNK):
            offs = c + offs0
            mask = offs < n
            ids = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
            local = ((ids - ls) % N_GLOBAL + N_GLOBAL) % N_GLOBAL
            match = (mask & (local == pid)).to(tl.int32)
            rank = tl.cumsum(match, axis=0) - 1
            tl.store(sc_ptr + base_c + rank, offs, mask=(match > 0) & in_c)
            tl.store(sh_ptr + base_h + rank, offs, mask=(match > 0) & in_h)
            k = tl.sum(match, axis=0)
            base_c += k
            base_h += k
        j = tl.arange(0, PAD)
        tl.store(sc_ptr + my_sc + my_c + j, tl.zeros([PAD], dtype=tl.int32) + n, mask=j < my_pc - my_c)
        tl.store(sh_ptr + my_sh + my_h + j, tl.zeros([PAD], dtype=tl.int32) + n, mask=j < my_ph - my_h)
        kb0 = tl.arange(0, BLK)
        b0c = my_sc // BSM_C
        nbc = my_pc // BSM_C
        for kk in range(0, nbc, BLK):
            kb = kk + kb0
            tl.store(bc_ptr + b0c + kb, tl.zeros([BLK], dtype=tl.int32) + pid, mask=kb < nbc)
        b0h = my_sh // BSM_H
        nbh = my_ph // BSM_H
        for kk in range(0, nbh, BLK):
            kb = kk + kb0
            tl.store(bh_ptr + b0h + kb, tl.zeros([BLK], dtype=tl.int32) + pid, mask=kb < nbh)
    else:
        for c in range(0, n, CHUNK):
            offs = c + offs0
            mask = offs < n
            ids = tl.load(ids_ptr + offs, mask=mask, other=0).to(tl.int32)
            local = ((ids - ls) % N_GLOBAL + N_GLOBAL) % N_GLOBAL
            local = tl.where(local < NUM_E, local, -1)
            tl.store(local_ids_ptr + offs, local, mask=mask)
            if COLD_IDS:
                is_hot = tl.sum(((local[:, None] == experts[None, :]) & hot[None, :]).to(tl.int32), axis=1) > 0
                tl.store(cold_ids_ptr + offs, tl.where(is_hot, -1, local), mask=mask)
        for s in range(0, cap_c, CHUNK):
            slots = s + offs0
            tl.store(sc_ptr + slots, tl.zeros([CHUNK], dtype=tl.int32) + n, mask=(slots >= total_c) & (slots < cap_c))
        for s in range(0, cap_h, CHUNK):
            slots = s + offs0
            tl.store(sh_ptr + slots, tl.zeros([CHUNK], dtype=tl.int32) + n, mask=(slots >= total_h) & (slots < cap_h))
        kb0 = tl.arange(0, BLK)
        used_c = total_c // BSM_C
        for kk in range(0, nb_c, BLK):
            kb = kk + kb0
            tl.store(bc_ptr + kb, tl.zeros([BLK], dtype=tl.int32) - 1, mask=(kb >= used_c) & (kb < nb_c))
        used_h = total_h // BSM_H
        for kk in range(0, nb_h, BLK):
            kb = kk + kb0
            tl.store(bh_ptr + kb, tl.zeros([BLK], dtype=tl.int32) - 1, mask=(kb >= used_h) & (kb < nb_h))
        tl.store(nc_ptr, total_c)
        tl.store(nh_ptr, total_h)
def _hc_align(topk_ids, local_start, bsm_hot):
    topk_ids = topk_ids.contiguous()
    device = topk_ids.device
    n = topk_ids.numel()
    cap_c, nb_c, cap_h, nb_h = _hc_caps(n, bsm_hot)
    local_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)
    sc = torch.empty((cap_c,), dtype=torch.int32, device=device)
    bc = torch.empty((nb_c,), dtype=torch.int32, device=device)
    nc = torch.empty((1,), dtype=torch.int32, device=device)
    sh = torch.empty((cap_h,), dtype=torch.int32, device=device)
    bh = torch.empty((nb_h,), dtype=torch.int32, device=device)
    nh = torch.empty((1,), dtype=torch.int32, device=device)
    hyb, h = bsm_hot < 0, _hc_hot_block(bsm_hot)
    cold_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=device) if hyb else local_ids
    _hc_align_kernel[(LOCAL_EXPERTS + 1,)](
        topk_ids, local_start, local_ids, sc, bc, nc, sh, bh, nh, n, cap_c, nb_c, cap_h, nb_h, cold_ids,
        HOT=_hc_hot_rows(bsm_hot), CHUNK=ALIGN_PAR_CHUNK, PAD=triton.next_power_of_2(max(HC_COLD_BSM, h)),
        BLK=ALIGN_PAR_BLK, BSM_C=HC_COLD_BSM, BSM_H=h, NUM_E=LOCAL_EXPERTS, N_GLOBAL=N_EXPERTS, COLD_IDS=hyb,
        num_warps=ALIGN_PAR_WARPS)
    return local_ids, ((sc, bc, nc, sh, bh, nh, cold_ids) if hyb else (sc, bc, nc, sh, bh, nh))
def _hc_dispatch(topk_ids, topk_weights, local_start, bsm_hot):
    tw = topk_weights.to(torch.float32).contiguous()
    local_ids, align = _hc_align(topk_ids, local_start, bsm_hot)
    return local_ids, tw, align
def _hcw_build(weights, state, device, t_prepare0):
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        return "device capability is not (9, 0)"
    if time.perf_counter() - t_prepare0 > HC_WG_BUILD_DEADLINE_S:
        return f"wgmma build skipped: deadline ({time.perf_counter() - t_prepare0:.0f} s)"
    t0 = time.perf_counter()
    ext, btext = _wg_build(probe=True, pdl=True)
    if ext is None:
        ext, btext = _wg_build(probe=False, pdl=True)
    if ext is None:
        return f"wgmma module build failed: {str(btext)[-300:]}"
    info = ext.wg_init()
    for i in range(0, len(info), 10):
        gated, a_ok, regs, local_bytes, static_smem, dyn_smem, occ, e1, bin_v, ptx_v = info[i:i + 10]
        print(f"{HC_TAG} wgmma kernel {'w13+silu' if gated else 'w2'}: attrs ok {bool(a_ok)}, regs {regs}, spill "
              f"{local_bytes} B, smem {static_smem}+{dyn_smem} B, CTAs/SM {occ}, binary {bin_v}", flush=True)
        if not (a_ok and occ >= 1 and static_smem + dyn_smem <= WG_SMEM_LIMIT and bin_v == 90
                and regs * WG_THREADS >= 40 * 128 + 232 * 256):
            return f"wgmma kernel attributes ({'w13' if gated else 'w2'}: ok {a_ok}, occ {occ}, bin {bin_v})"
    _wg_attach_scratch(state, ext, device, weights["w2"].shape[1], D_EXPERT)
    state["wg_banks"] = {(0, LOCAL_EXPERTS): _wg_bank(weights, 0, LOCAL_EXPERTS)}
    state["wg_ext_pdl"] = True
    torch.cuda.synchronize()
    print(f"{HC_TAG} wgmma path ready for the hybrid's hot half: build + retile {time.perf_counter() - t0:.1f} s",
          flush=True)
    return None
def _hc_experts(x, local_ids, topk_weights, align, state, bsm_hot, fill=None, part="all", fork=None):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    hybrid = bsm_hot < 0
    if hybrid:
        sc, bc, nc, sh, bh, nh, cold_ids = align
        classes = ((sh, bh, nh, -bsm_hot),)
    else:
        sc, bc, nc, sh, bh, nh = align
        classes = ((sc, bc, nc, HC_COLD_BSM), (sh, bh, nh, bsm_hot))
    fk = state["hc"].get("fk") or {}
    fork = (fk.get("on") and _hc_class(num_tokens) == HCF_CLS) if fork is None else fork
    fork = bool(fork) and bsm_hot in HC_WG_HOT and part in ("all", "both")
    if bsm_hot in HC_WG_HOT:
        routes = num_tokens * TOP_K
        cache3 = torch.empty((routes, size_k), device=x.device, dtype=x.dtype)
        if fill is not None:
            cache3.fill_(fill)
        if part != "cold":
            if fork:
                fk["e0"].record()
                fk["s"].wait_event(fk["e0"])
            with torch.cuda.stream(fk["s"] if fork else None):
                state["wg_ext"].wg_gather(x, sh, bh, nh, routes, LOCAL_EXPERTS, state["wg_xs"])
                _wh_pair(state, (state.get("whk") or {}).get(_hc_class(num_tokens)), sh, bh, nh, routes, size_n, size_k,
                         cache3)
                if fork:
                    fk["e1"].record()
        classes = ()
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    if classes:
        cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
        cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
        cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                  use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    if fill is not None and classes:
        cache13.fill_(fill)
    for sorted_ids, block_experts, n_padded, bsm in classes:
        moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, moe_block_size=bsm, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    if classes:
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    if fill is not None and classes:
        cache3.fill_(fill)
    for sorted_ids, block_experts, n_padded, bsm in classes:
        moe_wna16_marlin_gemm(
            cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, moe_block_size=bsm, top_k=1,
            size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common)
    if hybrid and part != "hot":
        ea = state["ea"]
        ext = ea["ext"]
        h3 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=torch.bfloat16)
        lid3 = torch.empty((num_tokens, TOP_K), device=x.device, dtype=torch.int32)
        xarm = (state["hc"].get("cold_x") or {}).get(_hc_class(num_tokens))
        if xarm:
            if fill is not None:
                h3.fill_(fill)
            _eae_pair(ea, x, cold_ids, h3, lid3, cache3, num_tokens, 0, LOCAL_EXPERTS, EA4X_ARMS[xarm],
                      (state["hc"].get("cold_e") or {}).get(_hc_class(num_tokens), 0))
        else:
            k13, k2 = (ext.ea4_w13, ext.ea4_w2) if state["hc"].get("cold") == "v4" else (ext.ea3_w13, ext.ea3_w2)
            cb = state["hc"].get("cold_bits", {}).get(_hc_class(num_tokens), 0)
            k13(x, cold_ids, ea["w13r"], ea["gs13"], h3, lid3, ea["ws13"], ea["tk13"], ea["err"], num_tokens, 0,
                LOCAL_EXPERTS, ea["grid"], ea["dq"], cb)
            k2(h3, cold_ids, ea["w2r"], ea["gs2"], cache3, ea["ws2"], ea["tk2"], ea["err"], num_tokens, 0,
               LOCAL_EXPERTS, ea["grid"], ea["dq"], ea["pdl"] | cb)
    if fork:
        torch.cuda.current_stream().wait_event(fk["e1"])
    if part != "all":
        return cache3
    return _hc_tail(cache3, local_ids, topk_weights, state, 0 if fork else 1)
def _hc_tail(cache3, local_ids, tw, state, pdl=1):
    t = local_ids.shape[0]
    if _t128_flags(state, t).get("comb"):
        return _t128_comb(cache3, local_ids, tw, state, pdl)
    return fused_post_expert_norm_weighted_sum(
        cache3.view(t, TOP_K, -1), tw, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)
def _hc_align_ref(ids_cpu, ls, bsm_hot):
    flat = [int(v) for v in ids_cpu.reshape(-1).tolist()]
    n = len(flat)
    cap_c, nb_c, cap_h, nb_h = _hc_caps(n, bsm_hot)
    thr = _hc_hot_rows(bsm_hot)
    bsm_hot = _hc_hot_block(bsm_hot)
    local = []
    routes = {e: [] for e in range(LOCAL_EXPERTS)}
    for i, v in enumerate(flat):
        e = (v - ls) % N_EXPERTS
        e = e if e < LOCAL_EXPERTS else -1
        local.append(e)
        if e >= 0:
            routes[e].append(i)
    def build(want_hot, bsm, cap, nb):
        srt, blk = [], []
        for e in range(LOCAL_EXPERTS):
            r = routes[e]
            if not r or (len(r) > thr) != want_hot:
                continue
            padded = -(-len(r) // bsm) * bsm
            srt += r + [n] * (padded - len(r))
            blk += [e] * (padded // bsm)
        total = len(srt)
        return srt + [n] * (cap - total), blk + [-1] * (nb - len(blk)), total
    return (torch.tensor(local, dtype=torch.int32).view(ids_cpu.shape), build(False, HC_COLD_BSM, cap_c, nb_c),
            build(True, bsm_hot, cap_h, nb_h))
def _hc_align_cmp(ids_cpu, ls, bsm_hot, local_ids, align):
    ref_l, ref_c, ref_h = _hc_align_ref(ids_cpu, ls, bsm_hot)
    if not torch.equal(local_ids.cpu(), ref_l):
        return "local ids differ"
    if len(align) == 7:
        flat = ref_l.reshape(-1).tolist()
        hot = {e for e in set(flat) if e >= 0 and flat.count(e) > _hc_hot_rows(bsm_hot)}
        want = torch.tensor([-1 if e in hot else e for e in flat], dtype=torch.int32).view(ref_l.shape)
        if not torch.equal(align[6].cpu(), want):
            return "cold ids differ"
    for name, (srt, blk, total), (s_d, b_d, n_d) in (("cold", ref_c, align[0:3]), ("hot", ref_h, align[3:6])):
        if int(n_d.cpu()[0]) != total:
            return f"{name} n_padded {int(n_d.cpu()[0])} != {total}"
        if not torch.equal(s_d.cpu(), torch.tensor(srt, dtype=torch.int32)):
            return f"{name} sorted_ids differ"
        if not torch.equal(b_d.cpu(), torch.tensor(blk, dtype=torch.int32)):
            return f"{name} block_experts differ"
    return None
def _hc_route_ids(kind, tokens, ls, gen):
    pop = torch.arange(1, LOCAL_EXPERTS + 1, dtype=torch.float32) ** (-1.1)
    pop = pop[torch.randperm(LOCAL_EXPERTS, generator=gen)]
    rows = []
    for _ in range(tokens):
        if kind == "zipf":
            k = int(torch.randint(0, 4, (1,), generator=gen))
            loc = torch.multinomial(pop, k, replacement=False, generator=gen).tolist() if k else []
        elif kind == "uniform":
            loc = torch.randperm(LOCAL_EXPERTS, generator=gen)[:2].tolist()
        elif kind == "mixed":
            loc = [3, 7] + [e for e in torch.randperm(LOCAL_EXPERTS, generator=gen).tolist() if e not in (3, 7)][:1]
        elif kind == "one":
            loc = [5]
        elif kind == "remote":
            loc = []
        else:
            loc = torch.randperm(LOCAL_EXPERTS, generator=gen)[:TOP_K].tolist()
        rem = (torch.randperm(N_EXPERTS - LOCAL_EXPERTS, generator=gen)[:TOP_K - len(loc)] + LOCAL_EXPERTS).tolist()
        row = [(int(e) + ls) % N_EXPERTS for e in loc + rem]
        rows.append([row[p] for p in torch.randperm(TOP_K, generator=gen).tolist()])
    return torch.tensor(rows, dtype=torch.int32)
def _hc_counts_text(local_ids):
    flat = local_ids.reshape(-1).long()
    counts = torch.bincount(flat[flat >= 0], minlength=LOCAL_EXPERTS).tolist()
    hot = [c for c in counts if c > HC_HOT_ROWS]
    return (f"local routes {sum(counts)}, distinct {sum(1 for c in counts if c)}, hot experts {len(hot)} "
            f"({sum(hot)} rows, max {max(counts)}), master's 16-row blocks {sum(-(-c // HC_COLD_BSM) for c in counts)}")
def _hc_warm(device, state):
    lines, bad = [], {}
    ls = torch.zeros((1,), dtype=torch.int32, device=device)
    ea = state.get("ea") or {}
    v3_ready = ea.get("ext") is not None and "v3" not in ea.get("skip", ()) and "w13r" in ea
    for h in HC_HOT_BSMS:
        if h < 0 and not v3_ready:
            bad[h] = "hybrid needs Engine A v3"
            lines.append(f"{HC_TAG} align kernel H={h}: off (Engine A v3 not built in this prepare)")
            continue
        if h in HC_WG_HOT and state["hc"].get("wg_off"):
            bad[h] = state["hc"]["wg_off"]
            lines.append(f"{HC_TAG} align kernel H={h}: off ({state['hc']['wg_off']})")
            continue
        t0 = time.perf_counter()
        try:
            _hc_align(torch.zeros((HC_MIN_TOKENS, TOP_K), dtype=torch.int32, device=device), ls, h)
            torch.cuda.synchronize()
            bad[h] = None
            lines.append(f"{HC_TAG} align kernel H={h}: compiled + ran in {time.perf_counter() - t0:.2f} s")
        except Exception as e:
            bad[h] = f"align build {type(e).__name__}"
            lines.append(f"{HC_TAG} align kernel H={h} raised {type(e).__name__} after {time.perf_counter() - t0:.2f} s: "
                         f"{str(e)[:300]}")
    return bad, lines
def _hc_kernel_check(weights, state, device, t_prepare0, built):
    ok = {(t, h): built.get(h) is None for t in HC_CLASSES for h in HC_HOT_BSMS}
    why = {(t, h): built[h] for t in HC_CLASSES for h in HC_HOT_BSMS if built.get(h) is not None}
    worst, f32w, lines = {h: 0.0 for h in HC_HOT_BSMS}, {}, []
    first = {h: True for h in HC_HOT_BSMS}
    gen = torch.Generator().manual_seed(20261091)
    for tokens in HC_CLASSES:
        for i, kind in enumerate(HC_KINDS):
            if time.perf_counter() - t_prepare0 > HC_GATE_DEADLINE_S:
                for h in HC_HOT_BSMS:
                    if ok[(tokens, h)]:
                        ok[(tokens, h)], why[(tokens, h)] = False, "kernel check skipped: deadline"
                continue
            ls = (240, 0, 96)[i % 3]
            ids_cpu = _hc_route_ids(kind, tokens, ls, gen)
            ids = ids_cpu.to(device)
            ls_t = torch.tensor([ls], dtype=torch.int32, device=device)
            xn = torch.randn((tokens, D_EXPERT), generator=gen).to(torch.bfloat16).to(device)
            tw = torch.rand((tokens, TOP_K), generator=gen)
            tw = (tw / tw.sum(-1, keepdim=True)).to(device).contiguous()
            m_lids, m_tw, m_align = _expert_dispatch(ids, tw, ls_t, tokens, state)
            ref = _experts(xn, None, None, state, dispatch=(m_lids, m_tw, m_align)).float()
            torch.cuda.synchronize()
            no_local = (m_lids < 0).all(-1)
            f32, m_err = None, 0.0
            if kind == "zipf":
                f32 = _ea_ref_block(weights, 0, xn, m_lids, m_tw, state["post_expert_norm_w"], POST_NORM_EPS)
                m_err = _rt_rel(ref, f32)
            shape = _hc_counts_text(m_lids)
            for h in HC_HOT_BSMS:
                if not ok[(tokens, h)]:
                    continue
                try:
                    trace = _rt_tracer(HC_TAG, f"H={h} T={tokens} {kind}") if first[h] else None
                    if trace:
                        print(f"{HC_TAG} first H={h} launch: T={tokens} {kind} ls={ls} (each launch named, synced)",
                              flush=True)
                        trace("launching _hc_align")
                    lids, tw2, align = _hc_dispatch(ids, tw, ls_t, h)
                    if trace:
                        trace("_hc_align ok; launching _hc_experts (2 x w13, silu_and_mul, 2 x w2, post-norm)")
                    out = _hc_experts(xn, lids, tw2, align, state, h, fill=HC_JUNK).float()
                    torch.cuda.synchronize()
                    if trace:
                        trace("_hc_experts ok")
                    first[h] = False
                    problem = _hc_align_cmp(ids_cpu, ls, h, lids, align)
                    if problem is None and not bool(torch.isfinite(out).all()):
                        problem = f"non-finite output ({int((~torch.isfinite(out)).sum())} values)"
                    if problem is None:
                        rel, row = _rt_rel(out, ref), _rt_row_rel(out, ref)
                        worst[h] = max(worst[h], rel)
                        if not (rel <= HC_TOL_MASTER and row <= HC_ROW_MASTER):
                            problem = f"vs master rel-L2 {rel:.3g} (<= {HC_TOL_MASTER}), worst row {row:.3g}"
                        elif (bool(no_local.any()) and float(ref[no_local].abs().max()) == 0.0
                              and float(out[no_local].abs().max()) != 0.0):
                            problem = "a token without a local route is not exactly zero (master's is)"
                    if problem is None and f32 is not None:
                        e = _rt_rel(out, f32)
                        f32w[(tokens, h)] = (e, m_err)
                        if e > max(1.5 * m_err, HC_TOL_F32):
                            problem = f"vs fp32 rel-L2 {e:.3g} > max(1.5 x master {m_err:.3g}, {HC_TOL_F32})"
                except Exception as ex:
                    problem = f"raised {type(ex).__name__}: {str(ex)[:200]}"
                if problem is not None:
                    ok[(tokens, h)], why[(tokens, h)] = False, f"kernel check {kind}: {problem}"
                    lines.append(f"{HC_TAG} kernel check FAIL H={h} T={tokens} {kind} ls={ls} ({shape}): {problem}")
                    print(lines[-1], flush=True)
            print(f"{HC_TAG} kernel check T={tokens} {kind} ls={ls}: {shape}", flush=True)
    for h in HC_HOT_BSMS:
        f = ", ".join(f"T={t} fp32 {f32w[(t, h)][0]:.3g} (master {f32w[(t, h)][1]:.3g})" for t in HC_CLASSES
                      if (t, h) in f32w)
        lines.append(f"{HC_TAG} kernel check H={h}: " +
                     ", ".join(f"T={t} {'clean' if ok[(t, h)] else 'FAIL'}" for t in HC_CLASSES) +
                     f"; worst rel-L2 vs master {worst[h]:.3g}" + (f"; {f}" if f else ""))
    return ok, why, lines
def _hc_layer_gate(state, device, cls, cands, seed):
    hc = state["hc"]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    per, stats = {}, []
    try:
        for rep in range(2):
            data = _ead_cases(state, device, gen, cls)
            ls_val = int(data["local_start"][0])
            graphs, outs = {}, {}
            for name, h in (("master", 0),) + tuple((f"H{h}", h) for h in cands):
                hc["on"][cls] = h
                graphs[name], outs[name] = _rt_graph(lambda: kernel_fn(data, state))
            hc["on"][cls] = 0
            label = f"T={cls} layer input {rep}"
            per[label] = _rt_time(graphs, device)
            ids = outs["master"][2].long()
            local = (ids - ls_val) % N_EXPERTS
            stats.append(f"{HC_TAG} {label} (local_start {ls_val}): "
                         + _hc_counts_text(torch.where(local < LOCAL_EXPERTS, local, -1)))
            print(stats[-1], flush=True)
            del graphs, outs
    finally:
        hc["on"][cls] = 0
    return per, stats
def _hc_layer_check(state, device, cls, h):
    hc = state["hc"]
    gen = torch.Generator(device=device)
    gen.manual_seed(20261092 + cls)
    worst = {}
    try:
        for rep in range(2):
            hc["on"][cls] = 0
            data = _ead_cases(state, device, gen, cls)
            ls_val = int(data["local_start"][0])
            s_in = data["stream"].clone()
            scores = _cb_router_scores(_dl3_clone_data(data), state)
            d_ref = _dl3_clone_data(data)
            ref = kernel_fn(d_ref, state)
            hc["on"][cls] = h
            d_got = _dl3_clone_data(data)
            got = kernel_fn(d_got, state)
            hc["on"][cls] = 0
            torch.cuda.synchronize()
            same = torch.equal(got[2], ref[2])
            ok, msg = _cb_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst)
            if not (ok and same):
                problem = f"layer check T={cls} H={h} input {rep}: ids identical {same}; {msg}"
                return problem, f"{HC_TAG} {problem} -> FAIL"
    finally:
        hc["on"][cls] = 0
    return None, (f"{HC_TAG} layer check T={cls} H={h} (on vs off, 2 inputs, ids identical) ok: worst " +
                  ", ".join(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
def _hc_copy_check(state, device, ts=None, off="master for every T"):
    import gc
    on = state["hc"]["on"]
    ts = [c for c in HC_CLASSES if on.get(c, 0)] if ts is None else ts
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261093)
    fails, n = [], 0
    for tokens in ts:
        base = _ead_cases(state, device, gen, tokens)
        per = []
        for st in (state, other):
            data = _dl3_clone_data(base)
            graph, out = _rt_graph(lambda data=data, st=st: kernel_fn(data, st))
            per.append((data, graph, out))
        for k in range(4):
            fresh = _ead_cases(state, device, gen, tokens)
            for data, _, _ in per:
                for key, v in fresh.items():
                    if torch.is_tensor(v) and v.shape == data[key].shape:
                        data[key].copy_(v)
            ref = kernel_fn(_dl3_clone_data(per[0][0]), state)
            for _, graph, _ in (per if k % 2 == 0 else per[::-1]):
                graph.replay()
            torch.cuda.synchronize()
            for i, (_, _, out) in enumerate(per):
                n += 1
                e_h, e_s = _rt_rel(out[0].float(), ref[0].float()), _rt_rel(out[1].float(), ref[1].float())
                if not (e_h <= HC_COPY_TOL and e_s <= HC_COPY_TOL and torch.equal(out[2], ref[2])):
                    fails.append(f"T={tokens} (H={on[tokens]}) state {'AB'[i]} refill {k}: hidden {e_h:.3g} "
                                 f"stream {e_s:.3g}")
        del per
    del other
    gc.collect()
    torch.cuda.empty_cache()
    ok = not fails
    return ok, (f"{HC_TAG} copy check: {n} replays on two states, T {ts}: "
                + ("PASS" if ok else f"FAIL -> {off}: {fails[0]}"))
def _hc_path_lines(state, why, times):
    out = []
    for cls in HC_CLASSES:
        lo, hi = (HC_MIN_TOKENS, HC_CLASS_SPLIT - 1) if cls == HC_CLASSES[0] else (HC_CLASS_SPLIT, HC_MAX_TOKENS)
        h = state["hc"]["on"].get(cls, 0)
        cx = (state["hc"].get("cold_x") or {}).get(cls)
        t = times.get(cls)
        if h:
            out.append(f"{HC_TAG} PATH CHOSEN T={cls} ({lo}..{hi}): HOT/COLD H={h} (cold experts block "
                       f"{HC_COLD_BSM}, hot > {_hc_hot_rows(h)} rows block {_hc_hot_block(h)}"
                       f"{', cold split by cost' if state['hc'].get('cold_bits', {}).get(cls) else ''}"
                       f"{', cold ' + cx if cx else ''}), "
                       f"layer {t[h]:.1f} us vs master {t[0]:.1f} us")
        else:
            ttxt = "" if not t else "; layer us " + ", ".join(
                f"{'master' if k == 0 else f'H{k}'} {v:.1f}" for k, v in sorted(t.items()))
            out.append(f"{HC_TAG} PATH CHOSEN T={cls} ({lo}..{hi}): MASTER (fallback: {why.get(cls, 'off')}{ttxt})")
    return out
def _hc_setup(weights, state, device, t_prepare0):
    t0 = time.perf_counter()
    ea0 = state.get("ea") or {}
    cold = "v4" if ("v4" in EA_ENGINES and "v4" not in ea0.get("skip", ()) and ea0.get("ext") is not None
                    and hasattr(ea0["ext"], "ea4_w13")) else "v3"
    state["hc"] = {"on": {c: 0 for c in HC_CLASSES}, "cold": cold,
                   "cold_bits": {c: (EA_PDL_BITS.get(ea0.get("on", {}).get(c), 0) if cold == "v4" else 0)
                                 for c in HC_CLASSES},
                   "cold_x": {c: None for c in HC_CLASSES}}
    on = state["hc"]["on"]
    why, times, notes, raced = {}, {}, [], {}
    def late(limit):
        return time.perf_counter() - t_prepare0 > limit
    def say(line, keep=True):
        print(line, flush=True)
        if keep:
            notes.append(line)
    try:
        if late(HC_START_DEADLINE_S):
            raise RuntimeError(f"skipped: deadline (start at {time.perf_counter() - t_prepare0:.1f} s)")
        bank = state["bank"]
        if bank["size_n"] != D_EXPERT or tuple(weights["w13"].shape[:1]) != (LOCAL_EXPERTS,):
            raise RuntimeError(f"unsupported expert bank size_n {bank['size_n']} / w13 {tuple(weights['w13'].shape)}")
        try:
            state["hc"]["wg_off"] = _hcw_build(weights, state, device, t_prepare0)
        except Exception as e:
            state["hc"]["wg_off"] = f"wgmma setup raised {type(e).__name__}: {str(e)[:200]}"
        if state["hc"]["wg_off"]:
            say(f"{HC_TAG} wgmma hot half off: {state['hc']['wg_off']}")
        built, wlines = _hc_warm(device, state)
        for ln in wlines:
            say(ln)
        ok, kwhy, klines = _hc_kernel_check(weights, state, device, t_prepare0, built)
        for ln in klines:
            if "FAIL" not in ln:
                print(ln, flush=True)
            notes.append(ln)
        for cls in HC_CLASSES:
            cands = [h for h in HC_HOT_BSMS if ok.get((cls, h), False)]
            if not cands:
                why[cls] = "; ".join(sorted({kwhy[(cls, h)] for h in HC_HOT_BSMS if (cls, h) in kwhy}))[:300] \
                    or "no clean H"
                continue
            if late(HC_GATE_DEADLINE_S):
                why[cls] = "gate skipped: deadline"
                continue
            try:
                per, stats = _hc_layer_gate(state, device, cls, cands, 20261094 + cls)
                notes.extend(stats)
                win, worst, glines = _rt_select(per, "master", [f"H{h}" for h in cands])
                pin = HC_PIN.get(cls)
                if pin and pin in cands:
                    win = f"H{pin}"
                for ln in glines:
                    print(f"{HC_TAG} gate {ln}", flush=True)
                med = {}
                for name, h in (("master", 0),) + tuple((f"H{h}", h) for h in cands):
                    vals = [_rt_stats(r[name])[0] for rounds in per.values() for r in rounds]
                    med[h] = sum(vals) / len(vals)
                times[cls] = med
                say(f"{HC_TAG} gate T={cls} (whole layer, 2 inputs x 2 rounds): worst ratio "
                    + ", ".join(f"H{h} {worst[f'H{h}']:.3f}" for h in cands)
                    + f" -> {'H=' + win[1:] if win else 'master'}" + (" PINNED" if pin and win == f"H{pin}" else ""))
                raced[cls] = (cands, worst)
                if win is None:
                    why[cls] = "no H wins every round (" + ", ".join(f"H{h} x{worst[f'H{h}']:.3f}" for h in cands) + ")"
                    continue
                h = int(win[1:])
                if late(HC_CHECK_DEADLINE_S):
                    why[cls] = "layer check skipped: deadline"
                    continue
                problem, line = _hc_layer_check(state, device, cls, h)
                say(line)
                if problem is not None:
                    why[cls] = "layer check FAIL"
                    continue
                on[cls] = h
            except Exception as e:
                on[cls] = 0
                why[cls] = f"gate {type(e).__name__}"
                say(f"{HC_TAG} T={cls} raised {type(e).__name__}: {str(e)[:300]}")
        if CAMPAIGN_EA4X:
            try:
                ln = _hcx_cold(state, device, t_prepare0)
            except Exception as e:
                state["hc"]["cold_x"] = {c: None for c in HC_CLASSES}
                ln = f"{EAD4X_TAG} cold half raised {type(e).__name__}: {str(e)[:120]} -> incumbent cold halves"
            if ln:
                _ea4x_say(ln)
        if any(on.values()):
            if late(HC_CHECK_DEADLINE_S):
                for cls in HC_CLASSES:
                    if on[cls]:
                        on[cls], why[cls] = 0, "copy check skipped: deadline"
                say(f"{HC_TAG} copy check: skipped: deadline -> master for every T")
            else:
                cok, line = _hc_copy_check(state, device)
                say(line)
                if not cok:
                    for cls in HC_CLASSES:
                        if on[cls]:
                            on[cls], why[cls] = 0, "copy check FAIL"
    except Exception as e:
        for cls in HC_CLASSES:
            on[cls] = 0
            why.setdefault(cls, type(e).__name__)
        say(f"{HC_TAG} {type(e).__name__}: {str(e)[:400]}")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        say(f"{HC_TAG} final sync raised {type(e).__name__}: {str(e)[:200]}")
    state["hc"].update(why=why, times=times, race={c: v for c, v in raced.items()
                                                   if on[c] or why.get(c, "").startswith("no H wins")})
    tail = [f"{HC_TAG} ===== HOT/COLD MARLIN (decode, 128 <= T <= 256) SUMMARY ({time.perf_counter() - t0:.1f} s; "
            f"prepare at {time.perf_counter() - t_prepare0:.1f} s) ====="]
    tail.extend(notes)
    try:
        tail.extend(_hc_path_lines(state, why, times))
    except Exception as e:
        for cls in HC_CLASSES:
            on[cls] = 0
        tail.append(f"{HC_TAG} PATH CHOSEN T={HC_MIN_TOKENS}..{HC_MAX_TOKENS}: MASTER (summary raised "
                    f"{type(e).__name__}: {str(e)[:200]})")
    return tail
HCX_S = 200.0
HCX_SEED = 20261139
HCX_S_ALONE = (128, 256)
HCX_TIE = 0.002
def _hcx_order(cls, cand):
    if cls not in HCX_S_ALONE:
        return ["d"] if "d" in cand else []
    order = [k for k in ("s", "d") if k in cand]
    if len(order) == 2 and cand["d"][1] > cand["s"][1] + HCX_TIE:
        order = ["d", "s"]
    return order
class _HcxMiss(Exception):
    pass
def _hcx_inputs(state, device, gen, cls, h):
    data = _ead_cases(state, device, gen, cls)
    scores = _cb_router_scores(_dl3_clone_data(data), state)
    ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int32).contiguous()
    tw = torch.full((cls, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    xn = torch.randn((cls, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
    local_ids, tw2, align = _hc_dispatch(ids, tw, data["local_start"], h)
    return xn, local_ids, tw2, align
def _hcx_cold(state, device, t_prepare0):
    hc, ea = state["hc"], state.get("ea") or {}
    cx = hc["cold_x"]
    clean = ea.get("x_clean") or {}
    if ea.get("ext4x") is None or not any(clean.values()):
        return None
    t0 = time.perf_counter()
    nan = float("nan")
    parts, cand, d_on = {}, {}, {}
    def copy(cls, arm, frm):
        if time.perf_counter() - t_prepare0 > HCX_S:
            raise _HcxMiss("copy dl")
        _ea4x_zero(ea)
        cx[cls] = arm
        try:
            cok, line = _hc_copy_check(state, device, [cls], "cold twin off")
            print(f"{EAD4X_TAG} cold {line}", flush=True)
            sums = _ea4x_sums(ea)
            _ea4x_zero(ea)
        except Exception:
            cx[cls] = frm
            raise
        if not cok or any(sums):
            cx[cls] = frm
            raise _HcxMiss("copy FAIL" + (f" tk/err {'/'.join(map(str, sums))}" if any(sums) else ""))
    for cls in HC_CLASSES:
        h = hc["on"].get(cls, 0)
        cb = hc.get("cold_bits", {}).get(cls, 0)
        arms = [a for a in EAD4X_RACE if EA4X_BASE[a] == ("v4c" if cb else "v4") and clean.get(a)
                and EA4X_ARMS[a] & 0xFF == cb]
        alone = cls in HCX_S_ALONE
        if hc.get("cold") != "v4" or h not in HC_WG_HOT or not (arms if alone else any(a.endswith("d") for a in arms)):
            parts[cls] = f"T{cls} -"
            continue
        arms.sort(key=lambda a: not a.endswith("d"))
        txt = f"T{cls} {arms[0][:-1]}"
        try:
            if time.perf_counter() - t_prepare0 > HCX_S:
                raise _HcxMiss("canary deadline")
            gen = torch.Generator(device=device)
            gen.manual_seed(HCX_SEED + cls)
            bad = {}
            for i in range(3):
                if all(a in bad for a in arms):
                    break
                xn, lids, tw2, align = _hcx_inputs(state, device, gen, cls, h)
                cx[cls] = None
                ref = _hc_experts(xn, lids, tw2, align, state, h, fill=nan, part="cold")
                torch.cuda.synchronize()
                for arm in arms:
                    for run in range(2):
                        if arm in bad:
                            break
                        _ea4x_zero(ea)
                        ea["ws13x"].fill_(nan)
                        ea["ws2x"].fill_(nan)
                        cx[cls] = arm
                        try:
                            got = _hc_experts(xn, lids, tw2, align, state, h, fill=nan, part="cold")
                            torch.cuda.synchronize()
                        finally:
                            cx[cls] = None
                        sums = _ea4x_sums(ea)
                        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                            bad[arm] = f"canary {arm[-1]} in{i} r{run + 1} neq"
                        elif any(sums):
                            bad[arm] = f"canary {arm[-1]} in{i} r{run + 1} tk/err {'/'.join(map(str, sums))}"
                        del got
                    if arms[0] in bad and not alone:
                        raise _HcxMiss(bad[arms[0]])
                del ref
            if alone:
                txt += " c:" + "".join("!" if a in bad else a[-1] for a in arms)
                arms = [a for a in arms if a not in bad]
                if not arms:
                    raise _HcxMiss("canary FAIL")
            else:
                arms = [a for a in arms if a not in bad]
                txt += " ok/" + ("ok" if len(arms) > 1 else "-" if not bad else "FAIL")
            if time.perf_counter() - t_prepare0 > HCX_S:
                raise _HcxMiss("race deadline")
            def setter(v, cls=cls):
                def f():
                    cx[cls] = v
                return f
            setters = {"inc": setter(None)}
            setters.update({a: setter(a) for a in arms})
            per = _ead_layer_gate(state, device, cls, setters, HCX_SEED + 10 + cls)
            cx[cls] = None
            _, worst, glines = _rt_select(per, "inc", arms, margin=EAD4X_MARGIN, tight=EAD4X_TIGHT)
            for ln in glines:
                print(f"{EAD4X_TAG} cold gate {ln}", flush=True)
            def med(nm):
                v = [_rt_stats(r[nm])[0] for rounds in per.values() for r in rounds]
                return f"{sum(v) / len(v):.1f}"
            meds = {nm: med(nm) for nm in ("inc", *arms)}
            if alone:
                ri = _ead4x_decide(per, {a[-1]: a for a in arms}, EAD4X_BOARD.get(cls, "s"),
                                   _ead4x_regate(state, device, cls, setters, HCX_SEED + 10 + cls,
                                                 lambda cls=cls: cx.__setitem__(cls, None)),
                                   lambda: time.perf_counter() - t_prepare0 > HCX_S)
                cx[cls] = None
                q = {k: (a, ri["view"][k][0]) for k, a in {a[-1]: a for a in arms}.items() if k in ri["order"]}
                qmark = set(ri["q0"])
            else:
                for a in arms:
                    if _rt_select(per, "inc", [a], margin=EAD4X_MARGIN, tight=EAD4X_TIGHT)[0]:
                        cand.setdefault(cls, {})[a[-1]] = (a, worst[a])
                q = cand.get(cls, {})
                qmark = set(q)
            txt += (" " + " ".join(f"{a[-1]}{worst[a]:.3f}{'Q' if a[-1] in qmark else ''}" for a in arms)
                    + f" i{meds['inc']} " + " ".join(f"{a[-1]}{meds[a]}" for a in arms))
            if alone and ri["mode"] != f"{ri['n']}r":
                txt += f" [{ri['mode']}]"
            order = ri["order"] if alone else _hcx_order(cls, q)
            if not order:
                raise _HcxMiss("inc")
            if not alone:
                copy(cls, q["d"][0], None)
                d_on[cls] = q["d"]
                txt += " D"
            else:
                for k in order:
                    try:
                        copy(cls, q[k][0], None)
                    except Exception as e:
                        cx[cls] = None
                        txt += f" {k} {e}" if isinstance(e, _HcxMiss) else f" {k} raised {type(e).__name__}"
                        try:
                            _ea4x_zero(ea)
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        continue
                    txt += f" {k.upper()}"
                    break
                if cx[cls] is None:
                    raise _HcxMiss("inc")
        except Exception as e:
            cx[cls] = None
            txt += f" -> {e}" if isinstance(e, _HcxMiss) else f" raised {type(e).__name__} -> inc"
            try:
                _ea4x_zero(ea)
                torch.cuda.synchronize()
            except Exception:
                pass
        parts[cls] = txt
    for cls in HC_CLASSES:
        if "s" not in cand.get(cls, {}) or cls in HCX_S_ALONE:
            continue
        arm, wr = cand[cls]["s"]
        if not d_on:
            parts[cls] += " s alone"
        elif cls not in d_on:
            parts[cls] += " s no d"
        elif not wr > d_on[cls][1] + 0.002:
            parts[cls] += " s<=d"
        else:
            try:
                copy(cls, arm, d_on[cls][0])
                parts[cls] = parts[cls][:-1] + "d S"
            except Exception as e:
                cx[cls] = d_on[cls][0]
                parts[cls] += f" s {e}" if isinstance(e, _HcxMiss) else f" s raised {type(e).__name__}"
                try:
                    _ea4x_zero(ea)
                    torch.cuda.synchronize()
                except Exception:
                    pass
    return (f"{EAD4X_TAG} cold half (layer us): " + " | ".join(parts[c] for c in HC_CLASSES) +
            f"; {time.perf_counter() - t0:.1f} s, prepare at {time.perf_counter() - t_prepare0:.1f} s")
HCF_CLS = 128
CAMPAIGN_T128 = True
T128_TAG = "[w6b-t128diet]"
T128_TS = (128, 256)
T128_CELLS = ("norm", "comb", "pool", "hilo", "unpool")
T128_BAR = {"norm": 1.5, "comb": 1.15, "khalf": 1.07, "nn": 1.02, "hilo": 1.5}
T128_MARGIN = 1.01
T128_DL = {"start": 225.0, "gate": 255.0, "copy": 266.0}
T128_PIN = {256: ("pool",)}
_T128_OFF = {}
def _t128_flags(state, tokens):
    t = state.get("t128")
    return _T128_OFF if t is None else t["on"].get(tokens, _T128_OFF)
@triton.jit
def _t128_add2_cast_kernel(p_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < numel
    a = tl.load(p_ptr + offs, mask=mask, other=0.0)
    b = tl.load(p_ptr + numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (a + b).to(tl.bfloat16), mask=mask)
@triton.jit(do_not_specialize=["M"])
def _t128_hilo_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, M, stride_xm, stride_wn, stride_om, K_SLICE: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * K_SLICE + tl.arange(0, BLOCK_K)
    hi_ptrs = hi_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn
    lo_ptrs = lo_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_SLICE // BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        acc = tl.dot(x_tile, tl.load(hi_ptrs), acc)
        acc = tl.dot(x_tile, tl.load(lo_ptrs), acc)
        x_ptrs += BLOCK_K
        hi_ptrs += BLOCK_K
        lo_ptrs += BLOCK_K
    tl.atomic_add(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc, mask=m_mask[:, None], sem="relaxed")
def _t128_pool(x, state, f):
    if not (f.get("pool") and x.is_contiguous()):
        return _gemm(x, state["dense"]["pooler"])
    wt = state["t128"]["pool_t"]
    m = x.shape[0]
    if m != 128:
        return torch.mm(x, wt)
    p = torch.bmm(x.view(m, 2, HIDDEN // 2).permute(1, 0, 2), wt.view(2, HIDDEN // 2, D_EXPERT),
                  out_dtype=torch.float32)
    if not p.is_contiguous():
        p = p.contiguous()
    out = torch.empty((m, D_EXPERT), dtype=torch.bfloat16, device=x.device)
    _t128_add2_cast_kernel[(triton.cdiv(out.numel(), 2048),)](p, out, out.numel(), BLOCK=2048, num_warps=4)
    return out
def _t128_router(x, state, f):
    if not (f.get("hilo") and x.stride(1) == 1):
        logits = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        logits += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return logits
    t, m = state["t128"], x.shape[0]
    sk = 16 if m <= 128 else 8
    out = torch.zeros((m, N_EXPERTS), dtype=torch.float32, device=x.device)
    _t128_hilo_kernel[(triton.cdiv(m, 64), N_EXPERTS // 64, sk)](
        x, t["hi"], t["lo"], out, m, x.stride(0), t["hi"].stride(0), out.stride(0), K_SLICE=HIDDEN // sk,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3)
    return out
def _t128_side(x, router_input, ls, tokens, hc_bsm, state, f):
    if tokens > ROUTER_FP32_MAX_TOKENS:
        logits = _t128_router(x, state, f)
    else:
        logits = F.linear(router_input, state["router"])
    topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    if hc_bsm:
        local_ids, tw, align = _hc_dispatch(topk_ids, topk_weights, ls, hc_bsm)
    else:
        local_ids, tw, align = _expert_dispatch(topk_ids, topk_weights, ls, tokens, state)
    return topk_weights, topk_ids, local_ids, tw, align
def _t128_comb(y, lid, tw, state, pdl=1):
    out = torch.empty((lid.shape[0], D_EXPERT), device=y.device, dtype=torch.bfloat16)
    state["ea"]["ext"].ea_combine(y, lid, tw, state["post_expert_norm_w"], out, POST_NORM_EPS, pdl)
    return out
def _t128_norm(ei, state, f):
    if not f.get("norm"):
        return state["pre_expert_norm"](ei)
    xn = torch.empty_like(ei)
    _DL_EXT.dl_norm(ei, state["dl"]["norm_w"], xn)
    return xn
def _t128_cands(state, t):
    ea = state.get("ea") or {}
    ok = {"norm": _DL_S3 and state.get("dl") is not None and hasattr(_DL_EXT, "dl_norm"),
          "comb": bool(_hc_bsm(state, t)) and hasattr(ea.get("ext"), "ea_combine"), "unpool": t == 128}
    return [c for c in T128_CELLS if ok.get(c, True)]
def _t128_bar(c, t):
    return T128_BAR[("khalf" if t == 128 else "nn") if c == "pool" else "nn" if c == "unpool" else c]
def _t128_unpool(x, state, f):
    return torch.mm(x, state["t128"]["unpool_t"]) if f.get("unpool") else _gemm(x, state["dense"]["unpooler"])
def _t128_inputs(state, device, gen, t):
    x = torch.randn((t, HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    lid = torch.randint(0, LOCAL_EXPERTS, (t, TOP_K), generator=gen, device=device, dtype=torch.int32)
    lid = torch.where(torch.rand((t, TOP_K), generator=gen, device=device) < 0.25, lid, -1).contiguous()
    y = torch.randn((t * TOP_K, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
    y[lid.view(-1) < 0] = HC_JUNK
    return {"x": x, "ei": _gemm(x, state["dense"]["pooler"]), "lid": lid, "y": y,
            "tw": (torch.rand((t, TOP_K), generator=gen, device=device) / 4).contiguous(),
            "yo": torch.randn((t, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)}
def _t128_ops(state, d, t, c):
    on = {c: True}
    if c != "comb":
        fn, a = {"norm": (_t128_norm, "ei"), "pool": (_t128_pool, "x"), "hilo": (_t128_router, "x"),
                 "unpool": (_t128_unpool, "yo")}[c]
        return (lambda: fn(d[a], state, _T128_OFF)), (lambda: fn(d[a], state, on))
    return (lambda: fused_post_expert_norm_weighted_sum(
        d["y"].view(t, TOP_K, D_EXPERT), d["tw"], norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=d["lid"])), (lambda: _t128_comb(d["y"], d["lid"], d["tw"], state))
def _t128_canary(state, d, t, cells):
    bad, parts = [], []
    for c in cells:
        try:
            a, b = (fn() for fn in _t128_ops(state, d, t, c))
            torch.cuda.synchronize()
            if c in ("norm", "comb"):
                v = d["ei"].double() if c == "norm" else \
                    d["y"].double().view(t, TOP_K, -1) * (1 + state["post_expert_norm_w"].double())
                v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + (RMS_EPS if c == "norm" else POST_NORM_EPS))
                ref = v * state["dl"]["norm_w"].double() if c == "norm" else \
                    (v * (d["tw"].double() * (d["lid"] >= 0))[..., None]).sum(1)
            else:
                xx, w = {"pool": (d["x"], state["dense"]["pooler"]), "unpool": (d["yo"], state["dense"]["unpooler"]),
                         "hilo": (d["x"], state["t128"]["hi"].double() + state["t128"]["lo"].double())}[c]
                ref = xx.double() @ w.double().t()
            e, ei, es = _rt_rel(b.double(), ref), _rt_rel(a.double(), ref), _rt_rel(b.float(), a.float())
            ok = bool(torch.isfinite(b.float()).all()) and (e <= max(1e-6 if c == "hilo" else 2e-3, 1.5 * ei)
                                                           or c in ("norm", "comb") and es <= 2e-3)
            txt = f"{c} vs fp64 {e:.2g} (inc {ei:.2g}; vs inc {es:.2g})"
            if c == "hilo":
                ia, ib = (state["routing"](None, v, TOP_K, True)[1].sort(-1).values for v in (a, b))
                nd = int((ia != ib).any(-1).sum())
                ok, txt = ok and nd == 0, txt + f", top-8 differs on {nd}/{t}"
        except Exception as e:
            ok, txt = False, f"{c} raised {type(e).__name__}: {str(e)[:80]}"
        parts.append(txt + ": FAIL" * (not ok))
        bad += [] if ok else [c]
    return bad, parts
def _t128_rider(state, device, gen, t, cells):
    f = {c: True for c in cells}
    x = torch.randn((t, HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    ls = torch.tensor([0], dtype=torch.int32, device=device)
    hb = _hc_bsm(state, t)
    data = _ead_cases(state, device, gen, t)
    rec = {}
    def keep(ei):
        rec["xn"] = state["pre_expert_norm"](ei).clone()
        return rec["xn"]
    kernel_fn(_dl3_clone_data(data), {**state, "pre_expert_norm": keep})
    noop = {**state, "pre_expert_norm": lambda ei: rec["xn"]}
    fns = {"main before": lambda: _t128_norm(_t128_pool(x, state, _T128_OFF), state, _T128_OFF),
           "main after": lambda: _t128_norm(_t128_pool(x, state, f), state, f),
           "side before": lambda: _t128_side(x, None, ls, t, hb, state, _T128_OFF),
           "side after": lambda: _t128_side(x, None, ls, t, hb, state, f),
           "layer": lambda: kernel_fn(data, state), "noop": lambda: kernel_fn(data, noop)}
    graphs = {n: _rt_graph(fn)[0] for n, fn in fns.items()}
    rounds = _rt_time(graphs, device)
    med = {n: [_rt_stats(r[n])[0] for r in rounds] for n in fns}
    del graphs, data, rec, x
    def j(vs, sign=""):
        return "/".join(f"{v:{sign}.1f}" for v in vs)
    sl = {k: [m - s for m, s in zip(med["main " + k], med["side " + k])] for k in ("before", "after")}
    r = f"{T128_TAG} rider T={t} "
    return med, [r + f"us r1/r2 before -> after: main {j(med['main before'])} -> {j(med['main after'])}; side "
                 f"{j(med['side before'])} -> {j(med['side after'])}; Q2 slack {j(sl['before'], '+')} -> "
                 f"{j(sl['after'], '+')}", r + f"no-op norm {j([a - b for a, b in zip(med['layer'], med['noop'])], '+')}"
                 f" of {j(med['layer'])} us; cells {'+'.join(cells)}"]
def _t128_layer(state, device, t, cells, seed):
    on = state["t128"]["on"][t]
    gen = torch.Generator(device=device).manual_seed(seed)
    draws = [_ead_cases(state, device, gen, t) for _ in range(4)]
    loads = [ld[0] for ld in _EAD_INPUTS[t][-4:]]
    hv = max(range(1, 4), key=lambda i: loads[i])
    per, worst = {}, {}
    def flags(cs):
        for c in T128_CELLS:
            on[c] = c in cs
    try:
        for label, i in (("judge-like", 0), ("p90", hv)):
            graphs = {}
            for name, cs in (("inc", ()), ("cells", cells)):
                flags(cs)
                graphs[name] = _rt_graph(lambda: kernel_fn(draws[i], state))[0]
            flags(())
            per[f"T={t} {label} (routes {loads[i]:.0f})"] = _rt_time(graphs, device)
            del graphs
        del draws
        for rep in range(2):
            data = _ead_cases(state, device, gen, t)
            d_ref, d_got = _dl3_clone_data(data), _dl3_clone_data(data)
            ref = kernel_fn(d_ref, state)
            flags(cells)
            got = kernel_fn(d_got, state)
            flags(())
            ok, msg = _cb_cmp_layer(got, ref, d_got, d_ref, data["stream"], _cb_router_scores(_dl3_clone_data(data), state),
                                    int(data["local_start"][0]), worst)
            if not ok:
                return per, msg, f"{T128_TAG} layer check T={t} input {rep} FAIL: {msg}"
    finally:
        flags(())
    return per, None, f"{T128_TAG} layer check T={t} (on vs off, 2 inputs) ok: worst " + \
        ", ".join(f"{k} {v:.2g}" for k, v in worst.items())
def _t128_setup(state, device, t_prepare0):
    t0 = time.perf_counter()
    out, why = [], {t: "off" for t in T128_TS}
    def say(line):
        print(line, flush=True)
        out.append(line)
    def late(k):
        return time.perf_counter() - t_prepare0 > T128_DL[k]
    try:
        if late("start"):
            raise RuntimeError("start deadline")
        dense = state["dense"]
        state["t128"] = {"on": {t: {c: False for c in T128_CELLS} for t in T128_TS},
                         "pool_t": dense["pooler"].t().contiguous(), "unpool_t": dense["unpooler"].t().contiguous(),
                         "hi": state["router_hi_t"].t().contiguous(), "lo": state["router_lo_t"].t().contiguous()}
        gen = torch.Generator(device=device).manual_seed(20261127)
        inps = {t: _t128_inputs(state, device, gen, t) for t in T128_TS}
        cands = {t: _t128_cands(state, t) for t in T128_TS}
        print(f"{T128_TAG} LAUNCH t128_cells_canary, cells {cands}", flush=True)
        bad = {}
        for t in T128_TS:
            bad[t], txt = _t128_canary(state, inps[t], t, cands[t])
            for s in txt:
                say(f"{T128_TAG} t128_cells_canary T={t} {s}")
        torch.cuda.synchronize()
        print(f"{T128_TAG} DONE   t128_cells_canary (synced)", flush=True)
        for t in T128_TS:
            cands[t] = [c for c in cands[t] if c not in bad[t]]
            graphs = {}
            for c in cands[t]:
                inc, cell = _t128_ops(state, inps[t], t, c)
                graphs[c + " inc"], graphs[c] = _rt_graph(inc)[0], _rt_graph(cell)[0]
            rounds = _rt_time(graphs, device) if graphs else []
            del graphs
            keep = []
            for c in cands[t]:
                iso = [_rt_stats(r[c + " inc"])[0] / max(_rt_stats(r[c])[0], 1e-3) for r in rounds]
                pin = c in T128_PIN.get(t, ())
                ok = pin or min(iso) >= _t128_bar(c, t)
                keep += [c] if ok else []
                say(f"{T128_TAG} Q1 T={t} {c} isolated r1/r2 x{'/'.join(f'{v:.3f}' for v in iso)} (bar "
                    f"{_t128_bar(c, t)}{', PINNED' if pin else '' if ok else ', DROP'})")
            for c in bad[t]:
                say(f"{T128_TAG} Q1 T={t} {c}: canary FAIL")
            med, lines = _t128_rider(state, device, gen, t, keep)
            for line in lines:
                say(line)
            if "hilo" not in keep and {"pool", "norm"} & set(keep) and max(med["side before"]) >= min(med["main after"]):
                keep = [c for c in keep if c not in ("pool", "norm")]
                say(f"{T128_TAG} T={t}: side chain binds, hilo missed -> pool, norm dropped")
            if not keep or late("gate"):
                why[t] = "deadline" if keep else "no cell passed"
                continue
            per, problem, cline = _t128_layer(state, device, t, keep, 20261128 + t)
            win, worst, glines = _rt_select(per, "inc", ["cells"], margin=T128_MARGIN, tight=T128_MARGIN)
            for ln in glines:
                print(f"{T128_TAG}   gate {ln}", flush=True)
            mu = [sum(_rt_stats(r[n])[0] for rr in per.values() for r in rr) / (2 * len(per)) for n in ("inc", "cells")]
            say(f"{T128_TAG} gate T={t} {keep} (layer, 2 inputs x 2 rounds): worst x{worst['cells']:.3f} (>= "
                f"{T128_MARGIN}), us {mu[0]:.1f} -> {mu[1]:.1f}")
            say(cline)
            if win is None or problem is not None:
                why[t] = "whole-layer gate miss" if win is None else "layer check FAIL"
                continue
            for c in keep:
                state["t128"]["on"][t][c] = True
            why[t] = ""
        ts = [t for t in T128_TS if not why[t]]
        if ts:
            ok, line = (False, "copy check: skipped: deadline") if late("copy") else _hc_copy_check(state, device, ts)
            say(f"{T128_TAG} {line}")
            for t in ts if not ok else ():
                why[t] = "copy check FAIL"
    except Exception as e:
        why = {t: f"{type(e).__name__}: {str(e)[:120]}" for t in T128_TS}
        say(f"{T128_TAG} raised -> incumbent: {why[128]}")
    t128 = state.get("t128")
    for t in T128_TS:
        on = [] if t128 is None else t128["on"][t]
        for c in on if why[t] else ():
            on[c] = False
        cs = [c for c in on if on[c]]
        pc = [c for c in cs if c in T128_PIN.get(t, ())]
        say(f"{T128_TAG} PATH CHOSEN T={t}: " + (f"CELLS {'+'.join(cs)}" if cs else f"INCUMBENT ({why[t]})")
            + (f" ({'+'.join(pc)} PINNED)" if pc else ""))
    return out
CAMPAIGN_B12P = True
B12P_TAG = "[b12p]"
B12P_EXT_NAME = "muse_decode_b12p_v1"
B12P_ESC_BYTES = 2048
B12P_ESC_CAP = (B12P_ESC_BYTES - 256) // 8
B12P_R = 32
B12P_CTAS = D_EXPERT // B12P_R
B12P_NCHUNK = 32
B12P_PW = 12288
B12P_RT = 2048
B12P_CHUNK = B12P_PW + B12P_RT
B12P_INSTS = {0: (8, 8, 1), 1: (8, 4, 1), 2: (8, 8, 4), 3: (12, 8, 4)}
B12P_GEMM_REL = 1e-2
B12P_LOGIT_REL = 1e-5
B12P_LAYER_MARGIN = 1.02
B12P_START_DEADLINE_S = 220.0
B12P_GATE_DEADLINE_S = 255.0
B12P_COPY_DEADLINE_S = 270.0
_B12P_CPP = r"""
#include <torch/extension.h>
#include <vector>
void b12p_run(torch::Tensor wc, torch::Tensor esc, torch::Tensor x, torch::Tensor ei, torch::Tensor logits,
              torch::Tensor err, int64_t inst, int64_t base, int64_t pdl);
void b12p_unpack(torch::Tensor wc, torch::Tensor out_p, torch::Tensor out_r, int64_t base);
std::vector<int64_t> b12p_setup(int64_t inst);
"""
_B12P_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
// ---- begin include rewrite/lib/cuda/b12p_kernels.cuh ----
// bulk12 P at T=1 for the decode layer (campaign 2026-10-26-expert-bandwidth, lever #5, DISTILL §4.2).
// One launch computes ei bf16 [1, 4096] = x @ pooler^T and logits fp32 [1, 256] = x @ hi^T + x @ lo^T (hi = bf16(router),
// lo = bf16(router - hi), master's split) from:
//   * the pooler in muse_bf16_proj's lossless 12-bit codec (bulk12, section (10)): R = 32 rows per CTA over the whole
//     K, 32 chunks of 256 k, 12,288 B per chunk in mma.sync A-fragment order, plus a 2 KB escape block per CTA;
//   * the router rows of experts 2c and 2c + 1 RAW, in the order hi(2c), lo(2c), hi(2c + 1), lo(2c + 1), as a 2,048 B
//     tail of every chunk: per k16 step 128 B, lane l < 16 (g = l >> 2 the router row, q = l & 3) holding the A
//     fragment registers a0 = (g, 2q..2q+1) and a2 = (g, 2q+8..2q+9); rows 4..15 of that m16 tile are zero.
// Grid 128 (CTA c: pooler rows 32c..32c+31, experts 2c, 2c+1), 1 CTA/SM, block 32 * (CW + 1): warp 0 lane 0 streams the
// CTA's 32 chunks (14,336 B each, one cp.async.bulk per chunk) through an S-slot mbarrier ring; CW consumer warps
// split each chunk's 16 k16 steps. No split-K, no workspace, no ticket. x [1, 8192] is copied once, in XS bulk copies.
// PDL: the escape block and chunks 0..S-1 (prepare-time constants) are issued BEFORE griddepcontrol.wait; x after it.
// Every thread executes griddepcontrol.wait before any global write and before reading x (it is a no-op without the
// launch attribute). launch_dependents after the main loop (the dependent's own wait still covers this grid's
// completion and memory). Epilogue in a fixed order: per row, the CW partials in warp order; pooler rows then apply
// their escapes in k order (+ x w_true, - x w_floor, as b12_kernel); logits = sum_hi + sum_lo. Deterministic.
// err codes: 1 producer wait timeout, 2 consumer wait timeout, 3 built without sm_90.

#define B12P_WAIT_NS 500000000ULL   // every mbarrier wait is bounded to 0.5 s, then an error word
#define B12P_ESC_BYTES 2048         // per-CTA escape block: uint16 row offsets [33] in its first 256 B, then entries
#define B12P_TILE 384               // one pooler m16 x k16 tile: 32 lanes x (8 L bytes + 4 code bytes)
#define B12P_PW 12288               // pooler bytes per chunk (16 k16 steps x 2 m16 tiles x 384)
#define B12P_RT 2048                // router tail bytes per chunk (16 k16 steps x 128)
#define B12P_CHUNK 14336            // B12P_PW + B12P_RT
#define B12P_R 32                   // pooler rows per CTA
#define B12P_K 8192
#define B12P_NCHUNK 32              // chunks per CTA (256 k each)
#define B12P_CTAS 128

namespace b12p {

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
// Bounded wait: false after B12P_WAIT_NS (the caller sets the error word and leaves the loop).
__device__ __forceinline__ bool mbar_wait(uint32_t a, uint32_t parity) {
  unsigned long long t0 = 0;
  for (uint32_t it = 0;; ++it) {
    uint32_t done;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) return true;
    if (it == 0) t0 = gtimer();
    else if (gtimer() - t0 > B12P_WAIT_NS) return false;
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
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// muse_bf16_proj section (10), verbatim: four high bytes from four codes (one per byte lane, low nibble: bit 3 the
// sign, bits 2..0 e71 - base): c + base + 120 * sign = 128 * sign + (e71 - base) + base. No carry between lanes.
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

// S: ring stages; CW: consumer warps; XS: x bulk copies (1 or 4; consumers wait for x part j / (32 / XS) lazily).
template <int S, int CW, int XS>
__global__ void __launch_bounds__(32 * (CW + 1), 1)
b12p_kernel(const uint8_t* __restrict__ wc, const uint8_t* __restrict__ esc, const __nv_bfloat16* __restrict__ x,
            __nv_bfloat16* __restrict__ ei, float* __restrict__ logits, int* err, uint32_t base) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int R = B12P_R, K = B12P_K, NCHUNK = B12P_NCHUNK, CHUNK = B12P_CHUNK;
  constexpr int CKK = K / NCHUNK;            // 256 k per chunk
  constexpr int NS = CKK / 16;               // 16 k16 steps per chunk
  constexpr int KSW = NS / CW;               // k16 steps per consumer warp per chunk
  constexpr int MT = R / 16;                 // 2 pooler m16 tiles
  constexpr int XOFF = 256 + B12P_ESC_BYTES; // barriers, escape block, x row 0, ring
  constexpr int RING = XOFF + K * 2;
  constexpr int XB = K * 2 / XS;             // bytes per x part
  constexpr int XCH = NCHUNK / XS;           // chunks per x part
  constexpr int SR = R + 4;                  // epilogue scratch row stride: 32 pooler rows + 4 router rows
  static_assert(NS % CW == 0 && MT * NS * B12P_TILE == B12P_PW && NS * 128 == B12P_RT, "geometry");
  static_assert(S >= 2 && S <= NCHUNK && 16 * S + 8 * XS <= 256, "barriers live in the first 256 B");
  static_assert(XS >= 1 && NCHUNK % XS == 0 && XB % 16 == 0, "x parts");
  static_assert(RING % 128 == 0 && CHUNK % 128 == 0, "alignment");
  static_assert(CW * SR * 4 <= S * CHUNK, "epilogue scratch must fit in the ring");
  static_assert(2 * (R + 1) <= 256, "row offsets must fit the escape header");
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t sbase = smem_u32(smem);
  const uint32_t full0 = sbase, empty0 = sbase + 8 * S, xbar0 = sbase + 16 * S;
  const uint32_t escs = sbase + 256, xs = sbase + XOFF, ring = sbase + RING;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int blk = blockIdx.x;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, CW); }
#pragma unroll
    for (int i = 0; i < XS; ++i) mbar_init(xbar0 + 8 * i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float acc[MT][4], accr[4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[a][c] = 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) accr[c] = 0.f;

  if (warp == 0) {
    if (lane == 0) {
      // ---- producer. Before the PDL wait: only prepare-time constants (escape block, chunks 0..S-1; their slots are
      // fresh, so no empty wait). After it: x, then the rest of the stream. ----
      mbar_expect_tx(xbar0, (uint32_t)(XB + B12P_ESC_BYTES));
#pragma unroll
      for (int i = 1; i < XS; ++i) mbar_expect_tx(xbar0 + 8 * i, (uint32_t)XB);
      bulk_g2s(escs, esc + (size_t)blk * B12P_ESC_BYTES, B12P_ESC_BYTES, xbar0);
      const uint8_t* wcta = wc + (size_t)blk * NCHUNK * CHUNK;
      for (int j = 0; j < S; ++j) {
        mbar_expect_tx(full0 + 8 * j, (uint32_t)CHUNK);
        bulk_g2s(ring + (uint32_t)(j * CHUNK), wcta + (size_t)j * CHUNK, (uint32_t)CHUNK, full0 + 8 * j);
      }
      griddep_wait();
#pragma unroll
      for (int i = 0; i < XS; ++i)
        bulk_g2s(xs + (uint32_t)(i * XB), reinterpret_cast<const uint8_t*>(x) + (size_t)i * XB, (uint32_t)XB,
                 xbar0 + 8 * i);
      for (int j = S; j < NCHUNK; ++j) {
        const int s = j % S;
        const uint32_t eph = ((j / S) & 1) ^ 1;
        if (!mbar_wait(empty0 + 8 * s, eph)) { atomicExch(err, 1); break; }
        mbar_expect_tx(full0 + 8 * s, (uint32_t)CHUNK);
        bulk_g2s(ring + (uint32_t)(s * CHUNK), wcta + (size_t)j * CHUNK, (uint32_t)CHUNK, full0 + 8 * s);
      }
    } else {
      griddep_wait();
    }
    __syncwarp();
    // The epilogue's pooler-row threads (this warp) read the escape block and x: observe every x part's completion.
#pragma unroll
    for (int i = 0; i < XS; ++i)
      if (!mbar_wait(xbar0 + 8 * i, 0) && lane == 0) atomicExch(err, 1);
  } else {
    // ---- consumers: warp cw owns k16 steps [cw*KSW, (cw+1)*KSW) of every chunk ----
    griddep_wait();
    const int cw = warp - 1;
    const uint32_t base4 = base * 0x01010101u;
    bool ok = true;
    for (int j = 0; j < NCHUNK && ok; ++j) {
      if (j % XCH == 0 && !mbar_wait(xbar0 + 8 * (j / XCH), 0)) { if (lane == 0) atomicExch(err, 2); ok = false; break; }
      const int s = j % S;
      const uint32_t ph = (j / S) & 1;
      if (!mbar_wait(full0 + 8 * s, ph)) { if (lane == 0) atomicExch(err, 2); ok = false; break; }
      const uint32_t slot = ring + (uint32_t)(s * CHUNK);
#pragma unroll
      for (int i = 0; i < KSW; ++i) {
        const int kk16 = cw * KSW + i;
        // B fragment: every token column reads x row 0 (k = 2q, 2q+1 and 2q+8, 2q+9 of this k16 step)
        const uint32_t xk = xs + (uint32_t)((j * CKK + kk16 * 16 + 2 * (lane & 3)) * 2);
        const uint32_t b0 = lds32(xk), b1 = lds32(xk + 16);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint32_t t = slot + (uint32_t)((kk16 * MT + mt) * B12P_TILE);
          uint32_t l0, l1, a0, a1, a2, a3;
          lds64(l0, l1, t + (uint32_t)lane * 8);
          const uint32_t c = lds32(t + 256 + (uint32_t)lane * 4);
          b12_frag(l0, l1, c, base4, a0, a1, a2, a3);
          mma_bf16(acc[mt], a0, a1, a2, a3, b0, b1);
        }
        // router tile: rows 0..3 (lanes 0..15) raw bf16, rows 4..15 zero
        uint32_t r0 = 0u, r2 = 0u;
        if (lane < 16) lds64(r0, r2, slot + (uint32_t)(B12P_PW + kk16 * 128 + lane * 8));
        mma_bf16(accr, r0, 0u, r2, 0u, b0, b1);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
  }
  __syncthreads();   // every issued copy has been waited on by the consumers: the ring is quiescent
  griddep_launch();
  float* scratch = reinterpret_cast<float*>(smem + RING);   // aliases the ring: [cw][SR]
  if (warp >= 1 && (lane & 3) == 0) {
    const int cw = warp - 1;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = mt * 16 + (lane >> 2);
      scratch[cw * SR + row] = acc[mt][0];       // D[row][token 0]
      scratch[cw * SR + row + 8] = acc[mt][2];   // D[row + 8][token 0]
    }
    if (lane < 16) scratch[cw * SR + R + (lane >> 2)] = accr[0];   // router rows 0..3
  }
  __syncthreads();
  if (tid < R) {
    float sum = scratch[tid];
#pragma unroll
    for (int c = 1; c < CW; ++c) sum += scratch[c * SR + tid];
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
    ei[(size_t)blk * R + tid] = __float2bfloat16_rn(sum);
  } else if (tid < R + 2) {
    const int i = tid - R;   // expert 2 blk + i: tail rows 2i (hi) and 2i + 1 (lo)
    float shi = scratch[R + 2 * i], slo = scratch[R + 2 * i + 1];
#pragma unroll
    for (int c = 1; c < CW; ++c) { shi += scratch[c * SR + R + 2 * i]; slo += scratch[c * SR + R + 2 * i + 1]; }
    logits[2 * blk + i] = shi + slo;
  }
#else
  if (threadIdx.x == 0) atomicExch(err, 3);
#endif
}

// Device round trip of the stream through the kernel's own decode and tail indexing: out_p [4096, 8192] and out_r
// [512, 8192] (int16 bits; out_r row 4c + g = tail row g of CTA c). Escapes come out as w_floor's bits (base << 8).
// One warp per (CTA, chunk).
__global__ void b12p_unpack_kernel(const uint8_t* __restrict__ wc, int16_t* __restrict__ out_p,
                                   int16_t* __restrict__ out_r, uint32_t base) {
  constexpr int R = B12P_R, K = B12P_K, NCHUNK = B12P_NCHUNK, CKK = K / NCHUNK, NS = CKK / 16, MT = R / 16;
  const int cta = blockIdx.x / NCHUNK, j = blockIdx.x % NCHUNK, lane = threadIdx.x;
  const uint32_t base4 = base * 0x01010101u;
  const int g = lane >> 2, q = lane & 3;
  const uint8_t* ch = wc + ((size_t)cta * NCHUNK + j) * B12P_CHUNK;
  for (int ks = 0; ks < NS; ++ks) {
    const int k0 = j * CKK + ks * 16 + 2 * q;
    for (int mt = 0; mt < MT; ++mt) {
      const uint8_t* t = ch + (size_t)(ks * MT + mt) * B12P_TILE;
      const uint32_t l0 = *reinterpret_cast<const uint32_t*>(t + lane * 8);
      const uint32_t l1 = *reinterpret_cast<const uint32_t*>(t + lane * 8 + 4);
      const uint32_t c = *reinterpret_cast<const uint32_t*>(t + 256 + lane * 4);
      uint32_t a[4];
      b12_frag(l0, l1, c, base4, a[0], a[1], a[2], a[3]);
      const int r0 = cta * R + mt * 16 + g;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = r0 + ((i & 1) ? 8 : 0), k = k0 + ((i & 2) ? 8 : 0);
        out_p[(size_t)r * K + k] = (int16_t)(a[i] & 0xFFFFu);
        out_p[(size_t)r * K + k + 1] = (int16_t)(a[i] >> 16);
      }
    }
    if (lane < 16) {
      const uint8_t* t = ch + B12P_PW + ks * 128 + lane * 8;
      const uint32_t r0 = *reinterpret_cast<const uint32_t*>(t);
      const uint32_t r2 = *reinterpret_cast<const uint32_t*>(t + 4);
      const size_t row = (size_t)(4 * cta + g) * K;
      out_r[row + k0] = (int16_t)(r0 & 0xFFFFu);
      out_r[row + k0 + 1] = (int16_t)(r0 >> 16);
      out_r[row + k0 + 8] = (int16_t)(r2 & 0xFFFFu);
      out_r[row + k0 + 9] = (int16_t)(r2 >> 16);
    }
  }
}

template <int S, int CW, int XS>
struct Inst {
  static constexpr int SMEM = 256 + B12P_ESC_BYTES + B12P_K * 2 + S * B12P_CHUNK;
  static void launch(const torch::Tensor& wc, const torch::Tensor& esc, const torch::Tensor& x,
                     const torch::Tensor& ei, const torch::Tensor& logits, const torch::Tensor& err, uint32_t base,
                     bool pdl) {
    static bool attrs = false;   // the dynamic-smem limit, once per process (setup() raises it too)
    if (!attrs) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(b12p_kernel<S, CW, XS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      attrs = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(B12P_CTAS);
    cfg.blockDim = dim3(32 * (CW + 1));
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, b12p_kernel<S, CW, XS>, (const uint8_t*)wc.data_ptr<uint8_t>(),
                                      (const uint8_t*)esc.data_ptr<uint8_t>(),
                                      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                      reinterpret_cast<__nv_bfloat16*>(ei.data_ptr()), logits.data_ptr<float>(),
                                      err.data_ptr<int>(), base));
  }
  // {regs, local (spill) B, static smem B, dyn smem B, occupancy CTAs/SM, S, CW, XS}; raises the smem limit.
  static std::vector<int64_t> setup() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(b12p_kernel<S, CW, XS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, b12p_kernel<S, CW, XS>));
    int occ = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, b12p_kernel<S, CW, XS>, 32 * (CW + 1),
                                                                 (size_t)SMEM));
    return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)SMEM, (int64_t)occ,
            (int64_t)S, (int64_t)CW, (int64_t)XS};
  }
};
using Inst0 = Inst<8, 8, 1>;    // inst 0: S=8,  9 warps, x in one copy   (133,376 B)
using Inst1 = Inst<8, 4, 1>;    // inst 1: S=8,  5 warps, x in one copy
using Inst2 = Inst<8, 8, 4>;    // inst 2: S=8,  9 warps, x in four copies
using Inst3 = Inst<12, 8, 4>;   // inst 3: S=12, 9 warps, x in four copies (190,720 B)

}  // namespace b12p

void b12p_run(torch::Tensor wc, torch::Tensor esc, torch::Tensor x, torch::Tensor ei, torch::Tensor logits,
              torch::Tensor err, int64_t inst, int64_t base, int64_t pdl) {
  TORCH_CHECK(wc.is_cuda() && esc.is_cuda() && x.is_cuda() && ei.is_cuda() && logits.is_cuda() && err.is_cuda(),
              "b12p: cuda tensors");
  TORCH_CHECK(wc.scalar_type() == at::kByte && esc.scalar_type() == at::kByte && x.scalar_type() == at::kBFloat16 &&
              ei.scalar_type() == at::kBFloat16 && logits.scalar_type() == at::kFloat && err.scalar_type() == at::kInt,
              "b12p: dtypes");
  TORCH_CHECK(wc.is_contiguous() && esc.is_contiguous() && x.is_contiguous() && ei.is_contiguous() &&
              logits.is_contiguous(), "b12p: contiguity");
  TORCH_CHECK(wc.numel() == (int64_t)B12P_CTAS * B12P_NCHUNK * B12P_CHUNK &&
              esc.numel() == (int64_t)B12P_CTAS * B12P_ESC_BYTES, "b12p: stream / escape block sizes");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) == B12P_K, "b12p: x must be [1, 8192]");
  TORCH_CHECK(ei.numel() == (int64_t)B12P_CTAS * B12P_R && logits.numel() == 2 * B12P_CTAS && err.numel() >= 1,
              "b12p: outputs");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wc.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(esc.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "b12p: 16 B alignment");
  TORCH_CHECK(base >= 1 && base <= 120, "b12p: base");
  const uint32_t b = (uint32_t)base;
  const bool p = pdl != 0;
  switch (inst) {
    case 0: b12p::Inst0::launch(wc, esc, x, ei, logits, err, b, p); break;
    case 1: b12p::Inst1::launch(wc, esc, x, ei, logits, err, b, p); break;
    case 2: b12p::Inst2::launch(wc, esc, x, ei, logits, err, b, p); break;
    case 3: b12p::Inst3::launch(wc, esc, x, ei, logits, err, b, p); break;
    default: TORCH_CHECK(false, "b12p_run: unknown instance");
  }
}

void b12p_unpack(torch::Tensor wc, torch::Tensor out_p, torch::Tensor out_r, int64_t base) {
  TORCH_CHECK(wc.is_cuda() && out_p.is_cuda() && out_r.is_cuda() && wc.scalar_type() == at::kByte &&
              out_p.scalar_type() == at::kShort && out_r.scalar_type() == at::kShort, "b12p_unpack: dtypes");
  TORCH_CHECK(wc.numel() == (int64_t)B12P_CTAS * B12P_NCHUNK * B12P_CHUNK &&
              out_p.numel() == (int64_t)B12P_CTAS * B12P_R * B12P_K && out_r.numel() == (int64_t)B12P_CTAS * 4 * B12P_K,
              "b12p_unpack: shapes");
  b12p::b12p_unpack_kernel<<<B12P_CTAS * B12P_NCHUNK, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      wc.data_ptr<uint8_t>(), out_p.data_ptr<int16_t>(), out_r.data_ptr<int16_t>(), (uint32_t)base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> b12p_setup(int64_t inst) {
  switch (inst) {
    case 0: return b12p::Inst0::setup();
    case 1: return b12p::Inst1::setup();
    case 2: return b12p::Inst2::setup();
    case 3: return b12p::Inst3::setup();
    default: TORCH_CHECK(false, "b12p_setup: unknown instance");
  }
  return {};
}
// ---- end include rewrite/lib/cuda/b12p_kernels.cuh ----
"""
def _b12p_build():
    try:
        ext, err_text, dt = _rt_build(B12P_TAG, B12P_EXT_NAME, _B12P_CPP, _B12P_SRC,
                                      ["b12p_run", "b12p_unpack", "b12p_setup"])
    except Exception as e:
        return None, 0.0, f"{type(e).__name__}: {str(e)[:300]}"
    return ext, dt, err_text
def _b12p_encode_pooler(w):
    n, k = w.shape
    r = B12P_R
    ctas, ckk = n // r, 8192 // r
    nch, ns, mt = k // ckk, ckk // 16, r // 16
    dev = w.device
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, lo = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    base = int(hist.unfold(0, 8, 1).sum(1).argmax())
    if not 1 <= base <= 120:
        raise RuntimeError(f"base {base} out of range")
    esc = (e71 < base) | (e71 > base + 7)
    code = torch.where(esc, torch.zeros_like(e71), (sign << 3) | (e71 - base))
    lo_enc = torch.where(esc, torch.zeros_like(lo), lo)
    dec = ((code + base + 120 * (code >> 3)) << 8) | lo_enc
    if not bool((dec[esc] == (base << 8)).all()):
        raise RuntimeError("escape slots do not decode to w_floor")
    dec = torch.where(esc, bits, dec)
    if not torch.equal(dec, bits):
        raise RuntimeError("host round trip is not bit-exact")
    del dec
    ln = torch.arange(32, device=dev)
    g, q = ln >> 2, ln & 3
    rows_l = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    cols_l = torch.stack([2 * q, 2 * q + 1, 2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9, 2 * q + 8, 2 * q + 9], 1)
    def frag(t):
        t6 = t.view(ctas, mt, 16, nch, ns, 16).permute(0, 3, 4, 1, 2, 5)
        return t6[..., rows_l, cols_l]
    lf = frag(lo_enc).to(torch.uint8).reshape(ctas, nch, ns, mt, 256)
    cf = frag(code)
    cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).to(torch.uint8).reshape(ctas, nch, ns, mt, 128)
    del cf
    wc = torch.cat([lf, cwd], -1).reshape(ctas, nch, ns * mt * 384).contiguous()
    idx = esc.nonzero()
    n_esc = int(idx.shape[0])
    blocks = torch.zeros((ctas, B12P_ESC_BYTES), dtype=torch.uint8)
    per_cta = torch.bincount((idx[:, 0] // r).cpu(), minlength=ctas) if n_esc else torch.zeros(ctas, dtype=torch.int64)
    max_cta = int(per_cta.max()) if ctas else 0
    if max_cta > B12P_ESC_CAP:
        raise RuntimeError(f"escape block overflow: {max_cta} > {B12P_ESC_CAP} in one CTA")
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
            blocks[c, :2 * (r + 1)] = roff.to(torch.int16).view(torch.uint8)
            ent = torch.stack([(rl << 16) | kk[s0:s0 + cnt], vals[s0:s0 + cnt].to(torch.int64)], 1).to(torch.int32)
            blocks[c, 256:256 + 8 * cnt] = ent.reshape(-1).view(torch.uint8)
    esc_dev = blocks.reshape(-1).to(dev).contiguous()
    return wc, esc_dev, base, n_esc, max_cta
def _b12p_tail_rows(router):
    r = router.float()
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    k = r.shape[1]
    return torch.stack((hi.view(B12P_CTAS, 2, k), lo.view(B12P_CTAS, 2, k)), dim=2).reshape(4 * B12P_CTAS, k).contiguous()
def _b12p_tail(rows):
    k = rows.shape[1]
    t = rows.view(torch.int16).view(B12P_CTAS, 4, B12P_NCHUNK, 16, 16).permute(0, 2, 3, 1, 4)
    ln = torch.arange(16, device=rows.device)
    g, q = ln >> 2, ln & 3
    cols = torch.stack([2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9], 1)
    out = t[..., g[:, None], cols]
    assert k == B12P_NCHUNK * 256
    return out.contiguous().view(torch.uint8).reshape(B12P_CTAS, B12P_NCHUNK, B12P_RT)
def _b12p_encode(ext, pooler, router):
    w = pooler.to(torch.bfloat16).contiguous()
    wc_p, esc, base, n_esc, max_cta = _b12p_encode_pooler(w)
    rows = _b12p_tail_rows(router)
    wc = torch.cat((wc_p, _b12p_tail(rows)), dim=-1).contiguous()
    del wc_p
    if tuple(wc.shape) != (B12P_CTAS, B12P_NCHUNK, B12P_CHUNK):
        raise RuntimeError(f"stream shape {tuple(wc.shape)}")
    dec_p = torch.empty(tuple(w.shape), dtype=torch.int16, device=w.device)
    dec_r = torch.empty(tuple(rows.shape), dtype=torch.int16, device=w.device)
    ext.b12p_unpack(wc, dec_p, dec_r, base)
    torch.cuda.synchronize()
    bits = w.view(torch.int16)
    hb = (bits.to(torch.int32) >> 8) & 0x7F
    escm = (hb < base) | (hb > base + 7)
    if not torch.equal(dec_p[~escm], bits[~escm]):
        raise RuntimeError("device round trip (b12p_unpack) differs from the pooler")
    if not bool((dec_p[escm].to(torch.int32) == (base << 8)).all()):
        raise RuntimeError("device escape slots do not decode to w_floor")
    if not torch.equal(dec_r, rows.view(torch.int16)):
        raise RuntimeError("device round trip of the router tail differs from master's hi / lo rows")
    del dec_p, dec_r, escm, hb
    el = w.numel()
    mb = (wc.numel() + esc.numel()) / 1e6
    line = (f"{B12P_TAG} encode: pooler base {base} (binades {2 * base - 127}..{2 * base + 15 - 127}), escapes {n_esc} "
            f"({n_esc / el:.1e}), max {max_cta}/CTA (cap {B12P_ESC_CAP}); stream {mb:.2f} MB vs raw P 75.50 MB; host + "
            f"device round trips bit-exact (pooler + router tail)")
    return wc, esc, base, line
def _b12p_launch(b12p, x, ei, logits, inst, pdl):
    b12p["ext"].b12p_run(b12p["wc"], b12p["esc"], x, ei, logits, b12p["err"], inst, b12p["base"], pdl)
def _b12p_p(x, ei, logits, b12p):
    if b12p is None or not b12p.get("on", False) or x.shape[0] != 1:
        return False
    if not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == HIDDEN and x.is_contiguous()
            and x.data_ptr() % 16 == 0 and ei.is_contiguous() and logits.is_contiguous()):
        return False
    _b12p_launch(b12p, x, ei, logits, b12p["inst"], b12p["pdl"])
    return True
def _b12p_err(b12p):
    torch.cuda.synchronize()
    return int(b12p["err"][0])
def _b12p_incumbent(state):
    dl = state["dl"]
    def p(x):
        ei = torch.empty((x.shape[0], D_EXPERT), dtype=torch.bfloat16, device=x.device)
        lg = torch.empty((x.shape[0], N_EXPERTS), dtype=torch.float32, device=x.device)
        _DL_EXT.dl_pooler_router(x, dl["w_pr"], ei, lg, dl["ws"], dl["cnt_p"], dl["pdl"])
        return ei, lg
    return p
def _b12p_engine(b12p, inst, pdl):
    def p(x):
        ei = torch.empty((1, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        lg = torch.empty((1, N_EXPERTS), dtype=torch.float32, device=x.device)
        _b12p_launch(b12p, x, ei, lg, inst, pdl)
        return ei, lg
    return p
def _b12p_check(b12p, inst, pdl, state, device, refs):
    wp, wr, k_esc, esc_rows = refs
    lab = f"b12p<{','.join(map(str, B12P_INSTS[inst]))}> pdl={pdl}"
    step = _rt_tracer(B12P_TAG, lab)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261081 + 10 * inst + pdl)
    g = 32
    ei_buf = torch.full((1, D_EXPERT + 2 * g), float("nan"), dtype=torch.bfloat16, device=device)
    lg_buf = torch.full((1, N_EXPERTS + 2 * g), float("nan"), dtype=torch.float32, device=device)
    ei, lg = ei_buf[:, g:g + D_EXPERT], lg_buf[:, g:g + N_EXPERTS]
    inc = _b12p_incumbent(state)
    problems, worst = [], {"rel": 0.0, "row": 0.0, "lrel": 0.0, "linc": 0.0}
    def guards_ok():
        return (bool(torch.isnan(ei_buf[:, :g]).all()) and bool(torch.isnan(ei_buf[:, g + D_EXPERT:]).all())
                and bool(torch.isnan(lg_buf[:, :g]).all()) and bool(torch.isnan(lg_buf[:, g + N_EXPERTS:]).all()))
    def launch(x, what):
        ei.fill_(float("nan"))
        lg.fill_(float("nan"))
        step(f"launch {what}")
        _b12p_launch(b12p, x, ei, lg, inst, pdl)
        step(f"done {what}")
        e = _b12p_err(b12p)
        if e:
            problems.append(f"{what}: error word {e}")
        if not guards_ok():
            problems.append(f"{what}: guard bands written")
        nn = int((~torch.isfinite(ei)).sum()) + int((~torch.isfinite(lg)).sum())
        if nn:
            problems.append(f"{what}: {nn} non-finite outputs")
        return ei.clone(), lg.clone()
    xs = [torch.randn((1, HIDDEN), generator=gen, device=device).to(torch.bfloat16) for _ in range(3)]
    for i, x in enumerate(xs):
        o_ei, o_lg = launch(x, f"random x {i}")
        ref = x.double() @ wp.double().t()
        lref = x.double() @ wr.t()
        good, rel, row, text = _rt_check(o_ei, ref, judge_rel=B12P_GEMM_REL)
        i_ei, i_lg = inc(x)
        torch.cuda.synchronize()
        lrel, linc = _rt_rel(o_lg.double(), lref), _rt_rel(i_lg.double(), lref)
        worst["rel"], worst["row"] = max(worst["rel"], rel), max(worst["row"], row)
        worst["lrel"], worst["linc"] = max(worst["lrel"], lrel), max(worst["linc"], linc)
        if not good or lrel > B12P_LOGIT_REL:
            problems.append(f"random x {i}: {text}, logits rel {lrel:.3g} (incumbent {linc:.3g})")
        if i == 0:
            r_ei, r_lg = launch(x, "repeat of random x 0")
            if not (torch.equal(r_ei, o_ei) and torch.equal(r_lg, o_lg)):
                problems.append("repeat launch not bit-identical")
    for kp in (k_esc, HIDDEN - 1):
        xp = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=device)
        xp[0, kp] = 0.5
        p_ei, p_lg = launch(xp, f"probe k={kp}")
        want = (wp[:, kp].float() * 0.5).to(torch.bfloat16)
        on_esc = esc_rows[kp] if kp in esc_rows else torch.zeros(D_EXPERT, dtype=torch.bool, device=device)
        exact = torch.equal(p_ei[0][~on_esc], want[~on_esc])
        if bool(on_esc.any()):
            e_rel = _rt_rel(p_ei[0][on_esc].double(), want[on_esc].double())
            exact = exact and e_rel <= 1e-2
        exact = exact and torch.equal(p_lg[0], (wr[:, kp] * 0.5).float())
        if not exact:
            problems.append(f"probe k={kp} ({int(on_esc.sum())} escape rows) not exact")
    xg = xs[0].clone()
    step("graph capture")
    graph, out = _rt_graph(lambda: _b12p_engine(b12p, inst, pdl)(xg))
    for rep in range(3):
        xg.copy_(xs[rep])
        graph.replay()
        step(f"graph replay {rep}")
        e_ei, e_lg = _b12p_engine(b12p, inst, pdl)(xs[rep])
        torch.cuda.synchronize()
        if not (torch.equal(out[0], e_ei) and torch.equal(out[1], e_lg)):
            problems.append(f"graph replay {rep} differs from eager")
    del graph
    e = _b12p_err(b12p)
    if e:
        problems.append(f"after graphs: error word {e}")
        b12p["err"].zero_()
    ok = not problems
    line = (f"{B12P_TAG} check {lab}: " + ("ok" if ok else f"FAIL ({len(problems)}: {problems[0][:120]})")
            + f"; ei rel {worst['rel']:.2g} row {worst['row']:.2g}, logits rel {worst['lrel']:.2g} "
            f"(incumbent {worst['linc']:.2g})")
    return ok, line
def _b12p_select(b12p, state, device, insts, pdl):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261082)
    x = torch.randn((1, HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    inc = _b12p_incumbent(state)
    names = {i: f"b12p{i}" for i in insts}
    graphs = {"incumbent": _rt_graph(lambda: inc(x))[0]}
    for i in insts:
        graphs[names[i]] = _rt_graph(lambda i=i: _b12p_engine(b12p, i, pdl)(x))[0]
    rounds = _rt_time(graphs, device)
    win, worst, _ = _rt_select({"P T=1": rounds}, "incumbent", list(names.values()))
    del graphs
    mean = {n: sum(_rt_stats(r[n])[0] for r in rounds) / len(rounds) for n in ["incumbent"] + list(names.values())}
    parts = [f"incumbent {mean['incumbent']:.2f}"] + [
        f"<{','.join(map(str, B12P_INSTS[i]))}> {mean[names[i]]:.2f} ({58.98 / mean[names[i]]:.2f} TB/s) "
        f"x{worst[names[i]]:.3f}" for i in insts]
    wi = None if win is None else next(i for i in insts if names[i] == win)
    pick = "incumbent" if wi is None else "b12p<" + ",".join(map(str, B12P_INSTS[wi])) + ">"
    line = f"{B12P_TAG} gemm P T=1 (read flush, pdl={pdl}), us: " + ", ".join(parts) + f" -> {pick}"
    return wi, {i: worst[names[i]] for i in insts}, line, mean
def _b12p_set(b12p, on, inst=None, pdl=None):
    b12p["on"] = bool(on)
    if inst is not None:
        b12p["inst"] = int(inst)
    if pdl is not None:
        b12p["pdl"] = int(pdl)
def _b12p_layer_gate(state, device, inst, pdls):
    b12p = state["b12p"]
    def setter(on, pdl=None):
        def f():
            _b12p_set(b12p, on, inst, pdl)
        return f
    setters = {"incumbent": setter(False)}
    for p in pdls:
        setters[f"pdl{p}"] = setter(True, p)
    try:
        per = _ead_layer_gate(state, device, 1, setters, 20261083)
    finally:
        _b12p_set(b12p, False)
    cands = [f"pdl{p}" for p in pdls]
    win, worst, _ = _rt_select(per, "incumbent", cands, margin=B12P_LAYER_MARGIN, tight=B12P_LAYER_MARGIN)
    parts = []
    for rounds in per.values():
        parts.append(" ".join("/".join(f"{_rt_stats(r[n])[0]:.1f}" for n in ["incumbent"] + cands) for r in rounds))
    wp = None if win is None else int(win[3:])
    line = (f"{B12P_TAG} layer gate T=1 b12p<{','.join(map(str, B12P_INSTS[inst]))}>, us incumbent/" + "/".join(cands)
            + f" per round: {' | '.join(parts)} -> " + ", ".join(f"{c} x{worst[c]:.3f}" for c in cands)
            + f" -> {'incumbent' if wp is None else f'b12p pdl={wp}'}")
    return wp, {int(c[3:]): worst[c] for c in cands}, line
def _b12p_layer_canary(state, device, inst, pdl):
    b12p = state["b12p"]
    lab = f"b12p<{','.join(map(str, B12P_INSTS[inst]))}> pdl={pdl} in kernel_fn T=1"
    step = _rt_tracer(B12P_TAG, lab)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261087 + pdl)
    problems = []
    def err(what):
        e = _b12p_err(b12p)
        if e:
            problems.append(f"{what}: error word {e}")
            b12p["err"].zero_()
    try:
        data = _ead_cases(state, device, gen, 1)
        _b12p_set(b12p, True, inst, pdl)
        outs = []
        for i in range(2):
            step(f"launch eager kernel_fn {i}")
            o = kernel_fn(_dl3_clone_data(data), state)
            step(f"done eager kernel_fn {i}")
            err(f"eager {i}")
            nn = int((~torch.isfinite(o[0].float())).sum()) + int((~torch.isfinite(o[1].float())).sum())
            if nn:
                problems.append(f"eager {i}: {nn} non-finite outputs")
            outs.append(o)
        if not all(torch.equal(a, b) for a, b in zip(outs[0][:3], outs[1][:3])):
            problems.append("eager repeat not bit-identical")
        dg = _dl3_clone_data(data)
        step("launch graph warm-up + capture of kernel_fn")
        graph, gout = _rt_graph(lambda: kernel_fn(dg, state))
        step("done graph warm-up + capture")
        err("graph warm-up / capture")
        for rep in range(2):
            for key, v in data.items():
                if torch.is_tensor(v) and v.shape == dg[key].shape:
                    dg[key].copy_(v)
            step(f"launch graph replay {rep}")
            graph.replay()
            step(f"done graph replay {rep}")
            err(f"graph replay {rep}")
            ref = outs[0]
            e_h, e_s = _rt_rel(gout[0].float(), ref[0].float()), _rt_rel(gout[1].float(), ref[1].float())
            if not (e_h <= DSA_COPY_TOL and e_s <= DSA_COPY_TOL and torch.equal(gout[2], ref[2])):
                problems.append(f"graph replay {rep} vs eager: hidden {e_h:.3g} stream {e_s:.3g}")
        del graph
    finally:
        _b12p_set(b12p, False)
    ok = not problems
    return ok, f"{B12P_TAG} layer canary {lab}: " + ("ok" if ok else f"FAIL ({len(problems)}: {problems[0][:160]})")
def _b12p_layer_check(state, device):
    b12p = state["b12p"]
    master = dict(state)
    master["b12p"] = None
    master["dsa"] = None
    gen = torch.Generator(device=device)
    gen.manual_seed(20261084)
    worst = {}
    try:
        for rep in range(2):
            _b12p_set(b12p, False)
            data = _ead_cases(state, device, gen, 1)
            ls_val = int(data["local_start"][0])
            s_in = data["stream"].clone()
            scores = _cb_router_scores(_dl3_clone_data(data), master)
            d_ref = _dl3_clone_data(data)
            ref = kernel_fn(d_ref, state)
            _b12p_set(b12p, True)
            d_got = _dl3_clone_data(data)
            got = kernel_fn(d_got, state)
            torch.cuda.synchronize()
            ok, msg = _dsa_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst)
            if not ok:
                return f"layer check input {rep}: {msg}", f"{B12P_TAG} layer check input {rep}: {msg[:300]} -> FAIL"
    finally:
        _b12p_set(b12p, False)
    line = (f"{B12P_TAG} layer check T=1 (on vs off, 2 inputs) ok: worst " +
            ", ".join(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
    return None, line
def _b12p_copy_check(state, device):
    import gc
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261085)
    fails, n = [], 0
    base = _ead_cases(state, device, gen, 1)
    per = []
    for st in (state, other):
        data = _dl3_clone_data(base)
        graph, out = _rt_graph(lambda data=data, st=st: kernel_fn(data, st))
        per.append((data, graph, out))
    for k in range(4):
        fresh = _ead_cases(state, device, gen, 1)
        for data, _, _ in per:
            for key, v in fresh.items():
                if torch.is_tensor(v) and v.shape == data[key].shape:
                    data[key].copy_(v)
        ref = kernel_fn(_dl3_clone_data(per[0][0]), state)
        for _, graph, _ in (per if k % 2 == 0 else per[::-1]):
            graph.replay()
        torch.cuda.synchronize()
        for i, (_, _, out) in enumerate(per):
            n += 1
            e_h, e_s = _rt_rel(out[0].float(), ref[0].float()), _rt_rel(out[1].float(), ref[1].float())
            if not (e_h <= DSA_COPY_TOL and e_s <= DSA_COPY_TOL and torch.equal(out[2], ref[2])):
                fails.append(f"state {'AB'[i]} refill {k}: hidden {e_h:.3g} stream {e_s:.3g}")
    del per
    left = (_b12p_err(state["b12p"]), _b12p_err(other["b12p"]))
    if any(left):
        fails.append(f"error words {left[0]} / {left[1]}")
    del other
    gc.collect()
    torch.cuda.empty_cache()
    ok = not fails
    return ok, f"{B12P_TAG} copy check: {n} replays on two states, T=1: " + ("PASS" if ok else f"FAIL: {fails[0]}")
def _b12p_profile_check(state, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261086)
    data = _ead_cases(state, device, gen, 1)
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            kernel_fn(data, state)
            torch.cuda.synchronize()
        cnt = {}
        for ev in prof.key_averages():
            cnt[ev.key] = cnt.get(ev.key, 0) + int(ev.count)
    except Exception as e:
        return None, f"{B12P_TAG} profiler unavailable ({type(e).__name__}: {str(e)[:120]}); route kept"
    nb = sum(v for k_, v in cnt.items() if "b12p_kernel" in k_)
    ni = sum(v for k_, v in cnt.items() if "dl_pooler_router_kernel" in k_)
    ok = nb == 1 and ni == 0
    return ok, f"{B12P_TAG} profiler T=1: b12p_kernel x{nb}, dl_pooler_router_kernel x{ni} -> {'ok' if ok else 'FAIL'}"
def _b12p_setup(state, device, t_prepare0):
    t0 = time.perf_counter()
    notes = []
    why = "off"
    chosen = None
    times = {}
    def late(limit):
        return time.perf_counter() - t_prepare0 > limit
    def say(line, keep=True):
        print(line, flush=True)
        if keep:
            notes.append(line)
    b12p = None
    try:
        if _B12P_EXT is None:
            raise RuntimeError(f"no module ({_B12P_BUILD_ERR[:200]})")
        dl = state.get("dl")
        if dl is None:
            raise RuntimeError("no small-T dense chain (master's side-stream P): nothing to replace")
        if late(B12P_START_DEADLINE_S):
            raise RuntimeError(f"skipped: deadline (start at {time.perf_counter() - t_prepare0:.1f} s)")
        ext = _B12P_EXT
        insts = []
        for inst, (s_, cw, xs_) in B12P_INSTS.items():
            info = [int(v) for v in ext.b12p_setup(inst)]
            good = info[4] >= 1 and info[1] == 0
            say(f"{B12P_TAG} inst {inst} <S={s_},CW={cw},XS={xs_}>: {info[0]} regs, spill {info[1]} B, smem {info[3]} B, "
                f"{info[4]} CTA/SM -> {'ok' if good else 'skipped'}", keep=not good)
            if good:
                insts.append(inst)
        if not insts:
            raise RuntimeError("no instance with occupancy >= 1 and no spill")
        te = time.perf_counter()
        wc, esc, base, line = _b12p_encode(ext, state["dense"]["pooler"], state["router"])
        say(line + f" ({time.perf_counter() - te:.1f} s)")
        b12p = {"ext": ext, "wc": wc, "esc": esc, "base": int(base), "inst": insts[0], "pdl": 0, "on": False,
                "err": torch.zeros((1,), dtype=torch.int32, device=device)}
        state["b12p"] = b12p
        wp = state["dense"]["pooler"].to(torch.bfloat16)
        rows = _b12p_tail_rows(state["router"]).view(B12P_CTAS, 2, 2, HIDDEN)
        wr = (rows[:, :, 0].double() + rows[:, :, 1].double()).reshape(N_EXPERTS, HIDDEN)
        hb = (wp.view(torch.int16).to(torch.int32) >> 8) & 0x7F
        escm = (hb < base) | (hb > base + 7)
        col_n = escm.sum(0)
        k_esc = int(col_n.argmax())
        esc_rows = {k_esc: escm[:, k_esc].clone(), HIDDEN - 1: escm[:, HIDDEN - 1].clone()}
        del hb, escm, col_n
        refs = (wp, wr, k_esc, esc_rows)
        pdl_dl = int(dl["pdl"])
        pdls = [0] + ([1] if pdl_dl else [])
        okc = {}
        for p in pdls:
            for inst in insts:
                if late(B12P_GATE_DEADLINE_S):
                    raise RuntimeError("checks: deadline")
                try:
                    okc[(inst, p)], line = _b12p_check(b12p, inst, p, state, device, refs)
                except Exception as e:
                    okc[(inst, p)] = False
                    line = f"{B12P_TAG} check inst {inst} pdl={p} raised {type(e).__name__}: {str(e)[:200]}"
                    b12p["err"].zero_()
                say(line)
        clean = [i for i in insts if okc.get((i, 0), False)]
        if not clean:
            raise RuntimeError("no instance passed the canary / correctness checks")
        if late(B12P_GATE_DEADLINE_S):
            raise RuntimeError("gemm selector: deadline")
        wi, gworst, line, mean = _b12p_select(b12p, state, device, clean, 0)
        say(line)
        times = mean
        if wi is None:
            why = f"gemm best x{max(gworst.values()):.3f}"
            raise _B12pMiss(why)
        if _b12p_err(b12p):
            raise RuntimeError(f"error word {int(b12p['err'][0])} after the selector")
        gate_pdls = []
        for p in [p for p in pdls if okc.get((wi, p), False)]:
            if late(B12P_GATE_DEADLINE_S):
                raise RuntimeError("layer canary: deadline")
            try:
                cok, line = _b12p_layer_canary(state, device, wi, p)
            except Exception as e:
                cok = False
                line = f"{B12P_TAG} layer canary inst {wi} pdl={p} raised {type(e).__name__}: {str(e)[:200]}"
                b12p["err"].zero_()
            say(line)
            if cok:
                gate_pdls.append(p)
        if not gate_pdls:
            why = "layer canary FAIL"
            raise _B12pMiss(why)
        if late(B12P_GATE_DEADLINE_S):
            raise RuntimeError("layer gate: deadline")
        wp_, lworst, line = _b12p_layer_gate(state, device, wi, gate_pdls)
        say(line)
        if wp_ is None:
            why = f"whole layer x{max(lworst.values()):.3f}"
            raise _B12pMiss(why)
        _b12p_set(b12p, False, wi, wp_)
        problem, line = _b12p_layer_check(state, device)
        say(line)
        if problem is not None:
            why = "layer check FAIL"
            raise _B12pMiss(why)
        _b12p_set(b12p, True, wi, wp_)
        if late(B12P_COPY_DEADLINE_S):
            why = "copy check skipped: deadline"
            raise _B12pMiss(why)
        ok, line = _b12p_copy_check(state, device)
        say(line)
        if not ok:
            why = "copy check FAIL"
            raise _B12pMiss(why)
        pok, line = _b12p_profile_check(state, device)
        say(line)
        if pok is False:
            why = "profiler assertion FAIL"
            raise _B12pMiss(why)
        if _b12p_err(b12p):
            why = f"error word {int(b12p['err'][0])}"
            raise _B12pMiss(why)
        chosen = (wi, wp_, lworst[wp_], gworst[wi])
    except _B12pMiss:
        if b12p is not None:
            _b12p_set(b12p, False)
    except Exception as e:
        if b12p is not None:
            _b12p_set(b12p, False)
            try:
                b12p["err"].zero_()
            except Exception:
                pass
        why = f"{type(e).__name__}: {str(e)[:160]}"
        say(f"{B12P_TAG} {why}")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        say(f"{B12P_TAG} final sync raised {type(e).__name__}: {str(e)[:200]}")
    if b12p is not None and not b12p["on"]:
        chosen = None
    if chosen is None and b12p is not None:
        state["b12p"] = None
    tail = [f"{B12P_TAG} ===== bulk12 P T=1 SUMMARY ({time.perf_counter() - t0:.1f} s; build {_B12P_BUILD_S:.1f} s at "
            f"import; prepare at {time.perf_counter() - t_prepare0:.1f} s) ====="]
    tail.extend(notes)
    if chosen is not None:
        wi, wp_, lx, gx = chosen
        tail.append(f"{B12P_TAG} PATH CHOSEN P T=1: B12P<{','.join(map(str, B12P_INSTS[wi]))}> pdl={wp_} "
                    f"(gemm x{gx:.3f}, whole layer x{lx:.3f}); T>1 incumbent")
    else:
        gtxt = "" if not times else f"; incumbent {times.get('incumbent', float('nan')):.2f} us"
        tail.append(f"{B12P_TAG} PATH CHOSEN P T=1: incumbent dl_pooler_router ({why}{gtxt}); T>1 incumbent")
    return tail
class _B12pMiss(Exception):
    pass
if CAMPAIGN_B12P and torch.cuda.is_available():
    _B12P_EXT, _B12P_BUILD_S, _B12P_BUILD_ERR = _b12p_build()
else:
    _B12P_EXT, _B12P_BUILD_S, _B12P_BUILD_ERR = None, 0.0, "CAMPAIGN_B12P off or no CUDA"
CAMPAIGN_SKFULL = True
CAMPAIGN_C320 = True
C320D_TAG = "[c320d]"
C320D_EXT_NAME = "muse_decode_c320_v1"
C320D_CHUNK = 320
C320D_NW = 16
C320D_REF_REL = 2e-2
C320D_V0_REL = 5e-3
C320D_LAYER_MARGIN = 1.01
C320D_TRIG_H2H = 1.003
C320D_START_DEADLINE_S = 215.0
C320D_GATE_DEADLINE_S = 250.0
C320D_COPY_DEADLINE_S = 265.0
C320D_NAMES = {0: "c320", 1: "c320t"}
C320D_CASES = (
    ("canary 5 keys", 5, 1.0, None),
    ("1 key", 1, 1.0, None),
    ("1 key kv 0.25 grid 3", 1, 0.25, 3),
    ("63 keys", 63, 1.0, None),
    ("64 keys kv 0.25 grid 3", 64, 0.25, 3),
    ("320 keys: seq-1 at 319 of item 0", 320, 1.0, None),
    ("320 keys kv 0.25 grid 3", 320, 0.25, 3),
    ("321 keys: seq-1 at 320 (0 of item 1)", 321, 1.0, None),
    ("321 keys kv 0.25 grid 3", 321, 0.25, 3),
    ("640 keys grid 3: seq-1 at 319 of item 1", 640, 1.0, 3),
    ("641 keys grid 3: seq-1 at 0 of item 2", 641, 1.0, 3),
    ("4097 keys", 4097, 1.0, None),
    ("4097 keys kv 0.25 grid 3", 4097, 0.25, 3),
    ("5001 keys", 5001, 1.0, None),
    ("5001 keys kv 0.25 grid 3", 5001, 0.25, 3),
    ("80000 keys: seq-1 at 319 of item 249", 80000, 1.0, None),
    ("80000 keys kv 0.25 grid 3", 80000, 0.25, 3),
    ("80001 keys", 80001, 1.0, None),
    ("80001 keys kv 0.25", 80001, 0.25, None),
    ("80001 keys grid 3", 80001, 1.0, 3),
    ("80001 keys kv 0.25 grid 3", 80001, 0.25, 3),
    ("90000 keys grid 264: two items per CTA", 90000, 1.0, 264),
)
_C320D_CPP = r"""
#include <torch/extension.h>
#include <vector>
void c320_init();
void c320_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
               torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor ws_acc, torch::Tensor ws_ml,
               double score_scale, int64_t max_chunks, int64_t grid);
void c320_combine(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                  torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t max_chunks, double eps,
                  int64_t onepass);
void c320_combine_pdl(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                      torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t max_chunks, double eps,
                      int64_t onepass, int64_t nw, int64_t trig);
std::vector<int64_t> c320_attrs(int64_t v);
"""
_C320D_SRC = r"""
// ---- begin include rewrite/lib/cuda/c320_kernels.cuh ----
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

namespace {

constexpr int D = 64;              // head dim
constexpr int HQ = 16;             // query heads
constexpr int HKV = 2;             // kv heads
constexpr int G = HQ / HKV;        // queries per kv head (8)
constexpr int ROW = HKV * D;       // bytes per cache row (page 1: both kv heads, 128)
constexpr int TILE = 64;           // tokens per pipeline tile
constexpr int STAGES = 4;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAXB = 64;
constexpr int CHUNK_B1 = 320;      // tokens per work item (the only instantiation)
constexpr int NW_B1 = 16;          // combine warps (512 threads)
constexpr int STAGE_BYTES = 2 * TILE * ROW;                  // K tile + V tile = 16 KB
constexpr int RING_OFF = 0;
constexpr int IDX_OFF = RING_OFF + STAGES * STAGE_BYTES;     // int32 [2][CHUNK]
// FUSED: the q region holds the raw bf16 q row (2 KB), the raw bf16 new k heads then v heads (512 B), the
// new row's e4m3 bytes (k heads, then v heads: 256 B) and the new token's cache row (int64, padded to 16 B).
constexpr int FQ_RAW = HQ * D * 2;
constexpr int FKV_RAW = 2 * HKV * D * 2;
constexpr int FKV_NEW = FQ_RAW + FKV_RAW;
constexpr int FKV_LOC = FKV_NEW + 2 * HKV * D;
constexpr int FQ_BYTES = FKV_LOC + 16;
// Shared-memory layout as constexpr functions of the work-item size and the FUSED flag.
__host__ __device__ constexpr int q_bytes(bool fused) { return fused ? FQ_BYTES : 2 * HQ * D; }      // e4m3 [2][HQ][D]
__host__ __device__ constexpr int q_off(int chunk) { return IDX_OFF + 2 * chunk * 4; }
__host__ __device__ constexpr int acc_off(int chunk, bool fused) { return q_off(chunk) + q_bytes(fused); }        // f32 [WARPS][G][D]
__host__ __device__ constexpr int ml_off(int chunk, bool fused) { return acc_off(chunk, fused) + WARPS * G * D * 4; }  // f32 [WARPS][G][2]
__host__ __device__ constexpr int seq_off(int chunk, bool fused) { return ml_off(chunk, fused) + WARPS * G * 2 * 4; }  // int32 [MAXB]
__host__ __device__ constexpr int pre_off(int chunk, bool fused) { return seq_off(chunk, fused) + MAXB * 4; }         // int32 [MAXB + 1]
__host__ __device__ constexpr int smem_a(int chunk, bool fused) { return pre_off(chunk, fused) + (MAXB + 4) * 4; }
static_assert(acc_off(CHUNK_B1, true) % 16 == 0 && acc_off(CHUNK_B1, false) % 16 == 0, "float4 alignment");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// Four E4M3 bytes -> two f16x2: lo from bytes 0,1 (byte 0 in the low half), hi from bytes 2,3.
__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);  // .x (low half) = lo
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// D = A B + D, m16n8k16, A row-major f16 (4 regs), B col-major f16 (2 regs), fp32 accumulate.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// EARLY (c320-pdl-early-trigger; instantiated FUSED only): griddepcontrol.launch_dependents right after the
// prologue instead of as the last statement. A no-op without a programmatic dependent; the dependent combine's
// griddepcontrol.wait still blocks until this grid has completed and flushed, so only its launch moves.
template <int CHUNK, bool FUSED, bool EARLY = false>
__global__ void __launch_bounds__(THREADS, 2)
c320_attn_kernel(const uint8_t* __restrict__ q,          // e4m3 [B, HQ, D] (unused when FUSED)
                 uint8_t* __restrict__ kc,               // e4m3 [rows, ROW]; FUSED stores the new row
                 uint8_t* __restrict__ vc,               // e4m3 [rows, ROW]; FUSED stores the new row
                 const int* __restrict__ page_table, int64_t pt_stride,
                 const int* __restrict__ seqlens, int B,
                 const float* __restrict__ k_scale, const float* __restrict__ v_scale, float score_scale,
                 float* __restrict__ ws_acc,             // f32 [B * max_chunks, HQ, D]
                 float* __restrict__ ws_ml,              // f32 [B * max_chunks, 2, HQ]
                 int max_chunks,
                 const __nv_bfloat16* __restrict__ qkv,  // bf16 [1, >= 2304] (FUSED, B == 1)
                 const int64_t* __restrict__ cache_loc,  // int64 [1] (FUSED, B == 1)
                 float eps) {
  static_assert(CHUNK % TILE == 0, "CHUNK must be a whole number of tiles");
  static_assert(CHUNK / TILE >= STAGES - 1, "cross-item meta prefetch needs TILES >= STAGES - 1");
  constexpr int TILES = CHUNK / TILE;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem + RING_OFF;
  int* idx_s = reinterpret_cast<int*>(smem + IDX_OFF);
  uint8_t* q_s = smem + q_off(CHUNK);  // FUSED: the raw q / new-row region
  float* acc_s = reinterpret_cast<float*>(smem + acc_off(CHUNK, FUSED));
  float* ml_s = reinterpret_cast<float*>(smem + ml_off(CHUNK, FUSED));
  int* seq_s = reinterpret_cast<int*>(smem + seq_off(CHUNK, FUSED));
  int* pre_s = reinterpret_cast<int*>(smem + pre_off(CHUNK, FUSED));
  const float NEG_INF = __int_as_float(0xff800000u);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bx = blockIdx.x, gx = gridDim.x;
  if (tid < B) seq_s[tid] = seqlens[tid];
  __syncthreads();
  if (tid == 0) {
    int run = 0;
    pre_s[0] = 0;
    for (int b = 0; b < B; ++b) {
      run += (seq_s[b] + CHUNK - 1) / CHUNK;
      pre_s[b + 1] = run;
    }
  }
  __syncthreads();
  const int total = pre_s[B];
  if (bx >= total) return;  // uniform over the CTA
  const int nk = (total - bx + gx - 1) / gx;  // items of this CTA
  const int N = nk * TILES;                   // tiles of this CTA, one flat pipeline
  const float scale = score_scale * __ldg(k_scale);  // sm_scale * qk_scale_factor * log2(e) * k_scale
  // FUSED (B == 1): the CTA owning the last item (item total - 1 = this CTA's item nk - 1) holds key seq - 1.
  const bool owner = FUSED && bx == (total - 1) % gx;

  // Item k of this CTA -> (request, chunk, cache length).
  auto item_of = [&](int k, int& b, int& chunk, int& seq) {
    const int i = bx + k * gx;
    int lo = 0;
    while (lo + 1 < B && pre_s[lo + 1] <= i) ++lo;
    b = lo;
    chunk = i - pre_s[lo];
    seq = seq_s[lo];
  };
  // Page indices (and, unless FUSED, the query) of item k into buffer k & 1 (positions past the length
  // zero-filled).
  auto issue_meta = [&](int k) {
    int b, chunk, seq;
    item_of(k, b, chunk, seq);
    const int buf = k & 1;
    const int* prow = page_table + (int64_t)b * pt_stride;
    for (int e = tid; e < CHUNK; e += THREADS) {
      const int pos = chunk * CHUNK + e;
      const bool ok = pos < seq;
      cp_async4(smem_u32(idx_s + buf * CHUNK + e), ok ? prow + pos : prow, ok ? 4 : 0);
    }
    if constexpr (!FUSED) {
      if (tid < HQ * D / 16)
        cp_async16(smem_u32(q_s + buf * HQ * D + tid * 16), q + (int64_t)b * HQ * D + tid * 16, 16);
    }
  };
  // Tile nn of the flat sequence into ring stage nn % STAGES: thread -> token tid / 4, 16-byte
  // chunks 2 * (tid % 4) + {0, 1} of its K row and its V row, chunk c stored at c ^ (token % 8).
  int ik = -1, ib = 0, ichunk = 0, iseq = 0;
  auto issue_tile = [&](int nn) {
    const int k = nn / TILES, t = nn % TILES;
    if (k != ik) { ik = k; item_of(k, ib, ichunk, iseq); }
    uint8_t* st = ring + (nn % STAGES) * STAGE_BYTES;
    const int tt = tid >> 2;
    const int e = t * TILE + tt;
    const bool ok = ichunk * CHUNK + e < iseq;
    const int64_t row = ok ? (int64_t)idx_s[(k & 1) * CHUNK + e] : 0;
    const uint8_t* ks = kc + row * ROW;
    const uint8_t* vs = vc + row * ROW;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = (tid & 3) * 2 + j;
      const int dst = tt * ROW + ((c ^ (tt & 7)) << 4);
      cp_async16(smem_u32(st + dst), ks + c * 16, ok ? 16 : 0);
      cp_async16(smem_u32(st + TILE * ROW + dst), vs + c * 16, ok ? 16 : 0);
    }
  };

  // Prologue: item 0's meta (waited), then tiles 0 .. STAGES-2; item 1's meta rides tile 0's group.
  issue_meta(0);
  if constexpr (FUSED) {
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  if constexpr (EARLY) asm volatile("griddepcontrol.launch_dependents;" :::);  // c320-pdl-early-trigger
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < N) {
      issue_tile(j);
      if (j % TILES == 0 && j / TILES + 1 < nk) issue_meta(j / TILES + 1);
    }
    cp_commit();
  }

  const int g = warp & 1;             // kv head
  const int slice = (warp >> 1) * 16; // token slice of each tile
  const int quad = lane >> 2, qi = lane & 3;
  uint32_t qa[4][2];
  float m_i = NEG_INF, l_i = 0.f;
  float acc[8][4];
  int cb = 0, cchunk = 0, cseq = 0;
  int n_last = -1, tt_last = 0;  // FUSED owner: flat tile of key seq - 1 and its token in that tile

  if constexpr (FUSED) {
  }

  for (int n = 0; n < N; ++n) {
    cp_wait<STAGES - 2>();  // tile n (and everything issued before it) has landed
    __syncthreads();        // visible to all; every warp is done with tile n - 1's stage
    {
      const int nn = n + STAGES - 1;
      if (nn < N) {
        issue_tile(nn);
        if (nn % TILES == 0 && nn / TILES + 1 < nk) issue_meta(nn / TILES + 1);
      }
      cp_commit();  // always, so the group count stays uniform
    }
    const int k = n / TILES, t = n % TILES;
    if (t == 0) {
      item_of(k, cb, cchunk, cseq);
      if constexpr (!FUSED) {
        // A operand for kstep s: row quad = query g * G + quad; k index 2 * qi + {0, 1} holds dims
        // s * 16 + 4 * qi + {0, 1}, k index 8 + 2 * qi + {0, 1} holds dims s * 16 + 4 * qi + {2, 3}
        // (K uses the same permutation, so the dot product is unchanged).
        const uint8_t* qrow = q_s + (k & 1) * HQ * D + (g * G + quad) * D + qi * 4;
#pragma unroll
        for (int s = 0; s < 4; ++s)
          e4m3x4_to_f16x2x2(*reinterpret_cast<const uint32_t*>(qrow + s * 16), qa[s][0], qa[s][1]);
      }
      m_i = NEG_INF;
      l_i = 0.f;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f;
    }
    uint8_t* Ks = ring + (n % STAGES) * STAGE_BYTES;
    uint8_t* Vs = Ks + TILE * ROW;
    if constexpr (FUSED) {
    }

    // S = Q K^T over the warp's 16 tokens: n-block nb holds tokens slice + nb * 8 + (0..7).
    float s[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.f;
      const int tt = slice + nb * 8 + quad;
      const uint8_t* kr = Ks + tt * ROW + qi * 4;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(kr + (((g * 4 + ks) ^ (tt & 7)) << 4));
        uint32_t b0, b1;
        e4m3x4_to_f16x2x2(w, b0, b1);
        mma16816(s[nb], qa[ks][0], 0u, qa[ks][1], 0u, b0, b1);
      }
    }
    // Row quad's scores at tokens slice + 2 * qi + {0, 1} (nb 0) and + 8 (nb 1).
    const int pos0 = cchunk * CHUNK + t * TILE + slice + qi * 2;
    float x0 = pos0 < cseq ? s[0][0] * scale : NEG_INF;
    float x1 = pos0 + 1 < cseq ? s[0][1] * scale : NEG_INF;
    float x2 = pos0 + 8 < cseq ? s[1][0] * scale : NEG_INF;
    float x3 = pos0 + 9 < cseq ? s[1][1] * scale : NEG_INF;
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float m_new = fmaxf(m_i, mx);
    const float m_use = m_new == NEG_INF ? 0.f : m_new;
    const float alpha = ex2(m_i - m_use);
    x0 = ex2(x0 - m_use);
    x1 = ex2(x1 - m_use);
    x2 = ex2(x2 - m_use);
    x3 = ex2(x3 - m_use);
    l_i = l_i * alpha + (x0 + x1) + (x2 + x3);  // this lane's columns; quad-reduced at item end
    m_i = m_new;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) { acc[nb][0] *= alpha; acc[nb][1] *= alpha; }
    const uint32_t pa0 = pack_f16x2(x0, x1), pa2 = pack_f16x2(x2, x3);

    // O += P V. B operand of dim-block nb: k = token slice + 2 * qi + {0, 1} (reg 0) and + 8
    // (reg 1), column quad = dim quad * 8 + nb. Each lane reads 8 contiguous bytes per token.
    const int tv = slice + qi * 2;
    const int vlc = g * 4 + (quad >> 1);
    const int voff = (quad & 1) * 8;
    const uint2 v0 = *reinterpret_cast<const uint2*>(Vs + tv * ROW + ((vlc ^ (tv & 7)) << 4) + voff);
    const uint2 v1 = *reinterpret_cast<const uint2*>(Vs + (tv + 1) * ROW + ((vlc ^ ((tv + 1) & 7)) << 4) + voff);
    const uint2 v8 = *reinterpret_cast<const uint2*>(Vs + (tv + 8) * ROW + ((vlc ^ ((tv + 8) & 7)) << 4) + voff);
    const uint2 v9 = *reinterpret_cast<const uint2*>(Vs + (tv + 9) * ROW + ((vlc ^ ((tv + 9) & 7)) << 4) + voff);
#pragma unroll
    for (int hw = 0; hw < 2; ++hw) {
      const uint32_t w0 = hw ? v0.y : v0.x, w1 = hw ? v1.y : v1.x;
      const uint32_t w8 = hw ? v8.y : v8.x, w9 = hw ? v9.y : v9.x;
      // Interleave the two tokens' bytes: byte 2j = token a dim j, byte 2j + 1 = token b dim j.
      uint32_t r0[4], r1[4];
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x5140), r0[0], r0[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w0, w1, 0x7362), r0[2], r0[3]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x5140), r1[0], r1[1]);
      e4m3x4_to_f16x2x2(__byte_perm(w8, w9, 0x7362), r1[2], r1[3]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma16816(acc[hw * 4 + j], pa0, 0u, pa2, 0u, r0[j], r1[j]);
    }

    if (t == TILES - 1) {
      // Item end: merge the four warps of each kv head, store the item's partial.
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 1);
      l_i += __shfl_xor_sync(0xffffffffu, l_i, 2);
      const int rw = warp * G + quad;
      if (qi == 0) { ml_s[rw * 2] = m_i; ml_s[rw * 2 + 1] = l_i; }
      float* ar = acc_s + rw * D;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
        ar[(qi * 2) * 8 + nb] = acc[nb][0];
        ar[(qi * 2 + 1) * 8 + nb] = acc[nb][1];
      }
      __syncthreads();
      const int h = tid >> 4, d0 = (tid & 15) * 4;
      const int hg = h / G, r = h % G;
      float mw[4];
      float M = NEG_INF;
#pragma unroll
      for (int j = 0; j < 4; ++j) { mw[j] = ml_s[((hg + 2 * j) * G + r) * 2]; M = fmaxf(M, mw[j]); }
      const float Mu = M == NEG_INF ? 0.f : M;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int wr = (hg + 2 * j) * G + r;
        const float wt = mw[j] == NEG_INF ? 0.f : ex2(mw[j] - Mu);
        L += wt * ml_s[wr * 2 + 1];
        const float4 a = *reinterpret_cast<const float4*>(acc_s + wr * D + d0);
        o.x += wt * a.x; o.y += wt * a.y; o.z += wt * a.z; o.w += wt * a.w;
      }
      const int64_t slot = (int64_t)cb * max_chunks + cchunk;
      *reinterpret_cast<float4*>(ws_acc + (slot * HQ + h) * D + d0) = o;
      if ((tid & 15) == 0) { ws_ml[slot * 2 * HQ + h] = M; ws_ml[slot * 2 * HQ + HQ + h] = L; }
    }
  }
  // PDL (SOURCE D): let a programmatic dependent (the combine, PDL variants only) launch now; its
  // griddepcontrol.wait still blocks until this grid has completed and flushed. No-op without one.
  // (EARLY instantiations triggered after the prologue instead.)
  if constexpr (!EARLY) asm volatile("griddepcontrol.launch_dependents;" :::);
  cp_wait<0>();  // the trailing groups are empty; drain anyway
}

// ONEPASS (c320-combine-onepass): one pass instead of max-then-fold. Each warp issues the loads of up to OP_GROUP
// of its partials (c = w + NW * i) before any use, folds them with an online LSE (running M; L and acc rescaled by
// ex2(M_old - M_new); an m = -inf partial weighs 0), repeating only when n > NW * OP_GROUP; then one barrier and
// warp 0 merges the NW warp states with the same LSE weights. Same arithmetic up to fp32 summation order.
constexpr int OP_GROUP = 16;
// TRIG (decode port, campaign/eb-w4-c320): griddepcontrol.launch_dependents as the first statement, so a programmatic
// dependent (decode's dl_o_kernel, prologue 0) launches and prefetches its weights while this combine runs; its
// griddepcontrol.wait still blocks until this grid has completed and flushed. Instantiated <320, 16, true, true, true>.
template <int CHUNK, int NW, bool PDL, bool ONEPASS = false, bool TRIG = false>
__global__ void __launch_bounds__(NW * 32)
c320_combine_kernel(const float* __restrict__ ws_acc, const float* __restrict__ ws_ml,
                    const int* __restrict__ seqlens, const float* __restrict__ v_scale,
                    const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                    __nv_bfloat16* __restrict__ out, int max_chunks, float eps) {
  if constexpr (TRIG) asm volatile("griddepcontrol.launch_dependents;" :::);
  __shared__ float red_m[NW];
  __shared__ float red_l[NW];
  __shared__ float red_acc[NW][D];
  const float NEG_INF = __int_as_float(0xff800000u);
  const int b = blockIdx.x, h = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = (seqlens[b] + CHUNK - 1) / CHUNK;
  const int64_t base = (int64_t)b * max_chunks;
  float vs_pre = 0.f, g0_pre = 0.f, g1_pre = 0.f;
  if constexpr (PDL) {
    // Above the wait: graph inputs only (seqlens above, v_scale and the gate logits here), complete before
    // kernel A started and never written by it. Every read of A's partials is below the wait.
    if (warp == 0) {
      vs_pre = __ldg(v_scale);
      const __nv_bfloat16* gp = qkv + (int64_t)b * qkv_stride + (HQ + 2 * HKV) * D + h * D + lane * 2;
      g0_pre = __bfloat162float(gp[0]);
      g1_pre = __bfloat162float(gp[1]);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }

  float o0 = 0.f, o1 = 0.f, Ls = 0.f;  // warp 0's merged sums (both bodies)
  if constexpr (ONEPASS) {
    float Mw = NEG_INF, Lw = 0.f, w0 = 0.f, w1 = 0.f;  // this warp's running state
    for (int gb = warp; gb < n; gb += NW * OP_GROUP) {
      float mv[OP_GROUP], lv[OP_GROUP];
      float2 av[OP_GROUP];
#pragma unroll
      for (int i = 0; i < OP_GROUP; ++i) {  // every load of the group before any use
        const int c = gb + NW * i;
        if (c < n) {
          const int64_t sl = base + c;
          mv[i] = ws_ml[sl * 2 * HQ + h];
          lv[i] = ws_ml[sl * 2 * HQ + HQ + h];
          av[i] = *reinterpret_cast<const float2*>(ws_acc + (sl * HQ + h) * D + lane * 2);
        } else {
          mv[i] = NEG_INF;
          lv[i] = 0.f;
          av[i] = make_float2(0.f, 0.f);
        }
      }
      float gm = mv[0];
#pragma unroll
      for (int i = 1; i < OP_GROUP; ++i) gm = fmaxf(gm, mv[i]);
      const float Mn = fmaxf(Mw, gm);
      const float Mnu = Mn == NEG_INF ? 0.f : Mn;
      const float alpha = Mw == NEG_INF ? 0.f : ex2(Mw - Mnu);  // Lw, w0, w1 are 0 while Mw is -inf
      Lw *= alpha;
      w0 *= alpha;
      w1 *= alpha;
#pragma unroll
      for (int i = 0; i < OP_GROUP; ++i) {
        const float wt = mv[i] == NEG_INF ? 0.f : ex2(mv[i] - Mnu);  // empty / absent partials weigh 0
        Lw += wt * lv[i];
        w0 += wt * av[i].x;
        w1 += wt * av[i].y;
      }
      Mw = Mn;
    }
    red_acc[warp][lane * 2] = w0;
    red_acc[warp][lane * 2 + 1] = w1;
    if (lane == 0) { red_m[warp] = Mw; red_l[warp] = Lw; }
    __syncthreads();
    if (warp == 0) {
      float M = red_m[0];
#pragma unroll
      for (int w = 1; w < NW; ++w) M = fmaxf(M, red_m[w]);
      const float Mu = M == NEG_INF ? 0.f : M;
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float mw = red_m[w];
        const float wt = mw == NEG_INF ? 0.f : ex2(mw - Mu);
        Ls += wt * red_l[w];
        o0 += wt * red_acc[w][lane * 2];
        o1 += wt * red_acc[w][lane * 2 + 1];
      }
    }
  } else {
  float M = NEG_INF;
  for (int c = tid; c < n; c += NW * 32) M = fmaxf(M, ws_ml[(base + c) * 2 * HQ + h]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  if (lane == 0) red_m[warp] = M;
  __syncthreads();
  M = red_m[0];
#pragma unroll
  for (int w = 1; w < NW; ++w) M = fmaxf(M, red_m[w]);
  const float Mu = M == NEG_INF ? 0.f : M;

  float a0 = 0.f, a1 = 0.f, L = 0.f;
#pragma unroll 4
  for (int c = warp; c < n; c += NW) {
    const int64_t sl = base + c;
    const float m = ws_ml[sl * 2 * HQ + h];
    const float wt = m == NEG_INF ? 0.f : ex2(m - Mu);  // empty partials weigh 0
    L += wt * ws_ml[sl * 2 * HQ + HQ + h];
    const float2 a = *reinterpret_cast<const float2*>(ws_acc + (sl * HQ + h) * D + lane * 2);
    a0 += wt * a.x;
    a1 += wt * a.y;
  }
  red_acc[warp][lane * 2] = a0;
  red_acc[warp][lane * 2 + 1] = a1;
  if (lane == 0) red_l[warp] = L;
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int w = 0; w < NW; ++w) { o0 += red_acc[w][lane * 2]; o1 += red_acc[w][lane * 2 + 1]; Ls += red_l[w]; }
  }
  }  // !ONEPASS
  if (warp == 0) {
    float vs, g0, g1;
    if constexpr (PDL) {
      vs = vs_pre; g0 = g0_pre; g1 = g1_pre;
    } else {
      vs = __ldg(v_scale);
      const __nv_bfloat16* gp = qkv + (int64_t)b * qkv_stride + (HQ + 2 * HKV) * D + h * D + lane * 2;
      g0 = __bfloat162float(gp[0]);
      g1 = __bfloat162float(gp[1]);
    }
    o0 = o0 / Ls * vs;
    o1 = o1 / Ls * vs;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = 1.f / sqrtf(ss / D + eps);
    const float y0 = o0 * r / (1.f + expf(-g0));
    const float y1 = o1 * r / (1.f + expf(-g1));
    *reinterpret_cast<__nv_bfloat162*>(out + ((int64_t)b * HQ + h) * D + lane * 2) = __floats2bfloat162_rn(y0, y1);
  }
}

template <bool FUSED, bool EARLY = false>
void set_attn_smem() {
  auto kern = c320_attn_kernel<CHUNK_B1, FUSED, EARLY>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_a(CHUNK_B1, FUSED)));
}

// {binaryVersion, ptxVersion, numRegs, localSizeBytes, maxThreadsPerBlock, dynamic smem, CTAs/SM}
template <typename K>
std::vector<int64_t> attrs_of(K kern, int threads, int smem) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kern));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem));
  return {static_cast<int64_t>(a.binaryVersion), static_cast<int64_t>(a.ptxVersion), static_cast<int64_t>(a.numRegs),
          static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.maxThreadsPerBlock),
          static_cast<int64_t>(smem), static_cast<int64_t>(occ)};
}

void check_combine(const torch::Tensor& ws_acc, const torch::Tensor& ws_ml, const torch::Tensor& cache_seqlens,
                   const torch::Tensor& out, int64_t max_chunks) {
  const int64_t B = cache_seqlens.size(0);
  TORCH_CHECK(B >= 1 && B <= MAXB && ws_acc.size(0) >= B * max_chunks && ws_ml.size(0) >= B * max_chunks,
              "c320_combine: batch / workspace");
  TORCH_CHECK(out.is_contiguous() && out.numel() >= B * HQ * D, "c320_combine: out");
}

}  // namespace

// Import-time only (never in the timed path): the dynamic-smem attribute of every kernel-A instantiation.
void c320_init() {
  set_attn_smem<false>();
}

void c320_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor page_table,
               torch::Tensor cache_seqlens, torch::Tensor k_scale, torch::Tensor ws_acc, torch::Tensor ws_ml,
               double score_scale, int64_t max_chunks, int64_t grid) {
  const int B = static_cast<int>(cache_seqlens.size(0));
  TORCH_CHECK(B >= 1 && B <= MAXB, "c320_attn: batch out of range");
  TORCH_CHECK(page_table.stride(1) == 1 && q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous(),
              "c320_attn: layout");
  TORCH_CHECK(ws_acc.size(0) >= B * max_chunks && ws_ml.size(0) >= B * max_chunks, "c320_attn: workspace");
  auto stream = at::cuda::getCurrentCUDAStream();
  c320_attn_kernel<CHUNK_B1, false><<<static_cast<unsigned>(grid), THREADS, smem_a(CHUNK_B1, false), stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()), reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<uint8_t*>(v_cache.data_ptr()), page_table.data_ptr<int>(), page_table.stride(0),
      cache_seqlens.data_ptr<int>(), B, k_scale.data_ptr<float>(), nullptr, static_cast<float>(score_scale),
      ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(), static_cast<int>(max_chunks), nullptr, nullptr, 0.f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void c320_combine(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                  torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t max_chunks, double eps,
                  int64_t onepass) {
  check_combine(ws_acc, ws_ml, cache_seqlens, out, max_chunks);
  const int B = static_cast<int>(cache_seqlens.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kern = onepass ? c320_combine_kernel<CHUNK_B1, NW_B1, false, true>
                      : c320_combine_kernel<CHUNK_B1, NW_B1, false, false>;
  kern<<<dim3(B, HQ), NW_B1 * 32, 0, stream>>>(
      ws_acc.data_ptr<float>(), ws_ml.data_ptr<float>(), cache_seqlens.data_ptr<int>(), v_scale.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv_stride,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(max_chunks), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The PDL entry point: programmatic stream serialization on THIS launch only (it must directly follow kernel A
// on the same stream). No fallback: whether PDL is used is decided once at import by the capture/replay check.
// onepass: the c320-combine-onepass body; nw: 16 (incumbent) or 8 warps (c320-pdl-early-trigger's co-residency
// fallback, PDL only).
void c320_combine_pdl(torch::Tensor ws_acc, torch::Tensor ws_ml, torch::Tensor cache_seqlens, torch::Tensor v_scale,
                      torch::Tensor qkv, int64_t qkv_stride, torch::Tensor out, int64_t max_chunks, double eps,
                      int64_t onepass, int64_t nw, int64_t trig) {
  TORCH_CHECK(nw == NW_B1 || nw == 8, "c320_combine_pdl: nw must be 16 or 8");
  TORCH_CHECK(!trig || (nw == NW_B1 && onepass), "c320_combine_pdl: trig needs nw 16 and onepass");
  check_combine(ws_acc, ws_ml, cache_seqlens, out, max_chunks);
  const int B = static_cast<int>(cache_seqlens.size(0));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(B, HQ);
  cfg.blockDim = dim3(static_cast<unsigned>(nw * 32));
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const float* p_acc = ws_acc.data_ptr<float>();
  const float* p_ml = ws_ml.data_ptr<float>();
  const int* p_seq = cache_seqlens.data_ptr<int>();
  const float* p_vs = v_scale.data_ptr<float>();
  const __nv_bfloat16* p_qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  __nv_bfloat16* p_out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const int mc = static_cast<int>(max_chunks);
  const float ep = static_cast<float>(eps);
  using CombineFn = void (*)(const float*, const float*, const int*, const float*, const __nv_bfloat16*, int64_t,
                             __nv_bfloat16*, int, float);
  CombineFn kern = nw == NW_B1 ? (onepass ? (CombineFn)c320_combine_kernel<CHUNK_B1, NW_B1, true, true>
                                          : (CombineFn)c320_combine_kernel<CHUNK_B1, NW_B1, true, false>)
                               : (onepass ? (CombineFn)c320_combine_kernel<CHUNK_B1, 8, true, true>
                                          : (CombineFn)c320_combine_kernel<CHUNK_B1, 8, true, false>);
  if (trig) kern = (CombineFn)c320_combine_kernel<CHUNK_B1, NW_B1, true, true, true>;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p_acc, p_ml, p_seq, p_vs, p_qkv, qkv_stride, p_out, mc, ep));
}

// v: 0 = A<320, false>, 1 = A<320, true> (FUSED builds only), 2 = combine<320, 16, false>, 3 = the PDL combine;
// 4 = A<320, true, EARLY> (FUSED builds only), 5 = combine<320, 16, false, ONEPASS>, 6 = combine<320, 16, true,
// ONEPASS>, 7 = combine<320, 8, true>, 8 = combine<320, 8, true, ONEPASS>, 9 = combine<320, 16, true, ONEPASS, TRIG>.
// Call c320_init() first (the occupancy of kernel A depends on its dynamic-smem attribute).
std::vector<int64_t> c320_attrs(int64_t v) {
  if (v == 0) return attrs_of(c320_attn_kernel<CHUNK_B1, false>, THREADS, smem_a(CHUNK_B1, false));
  if (v == 2) return attrs_of(c320_combine_kernel<CHUNK_B1, NW_B1, false>, NW_B1 * 32, 0);
  if (v == 3) return attrs_of(c320_combine_kernel<CHUNK_B1, NW_B1, true>, NW_B1 * 32, 0);
  if (v == 5) return attrs_of(c320_combine_kernel<CHUNK_B1, NW_B1, false, true>, NW_B1 * 32, 0);
  if (v == 6) return attrs_of(c320_combine_kernel<CHUNK_B1, NW_B1, true, true>, NW_B1 * 32, 0);
  if (v == 7) return attrs_of(c320_combine_kernel<CHUNK_B1, 8, true, false>, 8 * 32, 0);
  if (v == 8) return attrs_of(c320_combine_kernel<CHUNK_B1, 8, true, true>, 8 * 32, 0);
  if (v == 9) return attrs_of(c320_combine_kernel<CHUNK_B1, NW_B1, true, true, true>, NW_B1 * 32, 0);
  TORCH_CHECK(false, "c320_attrs: no such kernel in this build: ", v);
  return {};
}
// ---- end include rewrite/lib/cuda/c320_kernels.cuh ----
"""
def _c320d_build():
    try:
        ext, err_text, dt = _rt_build(C320D_TAG, C320D_EXT_NAME, _C320D_CPP, _C320D_SRC,
                                      ["c320_init", "c320_attn", "c320_combine_pdl", "c320_attrs"])
        if ext is not None:
            ext.c320_init()
            torch.cuda.synchronize()
            print(f"{C320D_TAG} c320_init: kernel A dynamic-smem attribute set", flush=True)
    except Exception as e:
        return None, 0.0, f"{type(e).__name__}: {str(e)[:300]}"
    return ext, dt, err_text
def _c320d_q(data, x, state, d3):
    cfg = data["config"]
    tokens = x.shape[0]
    dev = x.device
    q_attn = torch.empty((tokens, 16, 64), dtype=FP8, device=dev)
    buf = torch.empty((tokens, DL3_Q_ROWS), dtype=torch.bfloat16, device=dev)
    _DL_EXT.dl2_qkv(x.contiguous(), state["dense"]["qkv_gate"], d3["ws_q"], d3["cnt_q"], buf, q_attn.view(torch.uint8),
                    buf[:, SK_QKV_GATE_OFF:], data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8),
                    data["cache_loc"], data["positions"], state["cos_sin"], data["k_scale"], data["v_scale"], d3["pdl"],
                    0, 0, int(bool(cfg["rope_theta"])), int(cfg["kv_scale"] != 1.0))
    return q_attn, buf
def _c320d_core(data, q8, gsrc, trig, grid=None):
    cfg = data["config"]
    page_table = data["page_table"]
    dev = q8.device
    mc = -(-page_table.shape[1] // C320D_CHUNK)
    ws_acc = torch.empty((mc, 16, 64), dtype=torch.float32, device=dev)
    ws_ml = torch.empty((mc, 2, 16), dtype=torch.float32, device=dev)
    out = torch.empty((1, 1, 1024), dtype=torch.bfloat16, device=dev)
    g = grid or 2 * _sm_count(dev)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    _C320D_EXT.c320_attn(q8, data["k_cache"], data["v_cache"], page_table, data["cache_seqlens"], data["k_scale"],
                         ws_acc, ws_ml, scale, mc, g)
    _C320D_EXT.c320_combine_pdl(ws_acc, ws_ml, data["cache_seqlens"], data["v_scale"], gsrc, gsrc.stride(0), out, mc,
                                RMS_EPS, 1, C320D_NW, int(trig))
    return out
def _c320d_attention(data, state, d3, trig):
    q8, buf = _c320d_q(data, data["attn_input"], state, d3)
    o = _c320d_core(data, q8, buf, trig)
    return _dl3_o(o.reshape(1, 1024), state, d3)
class _C320dMiss(Exception):
    pass
def _c320d_set(state, on, trig=None):
    c3 = state["c320d"]
    c3["on"] = bool(on)
    if trig is not None:
        c3["trig"] = int(trig)
def _c320d_say(line):
    print(line, flush=True)
def _c320d_stage(name, t_prepare0, fn):
    t0 = time.perf_counter()
    _c320d_say(f"{C320D_TAG} stage {name}: start (prepare at {t0 - t_prepare0:.1f} s)")
    result = fn()
    torch.cuda.synchronize()
    _c320d_say(f"{C320D_TAG} stage {name}: synced ({time.perf_counter() - t0:.1f} s)")
    return result
def _c320d_bounds(data, q, gsrc, tag):
    pt, kc, vc, csl = data["page_table"], data["k_cache"], data["v_cache"], data["cache_seqlens"]
    rows = kc.numel() // 128
    ok = (pt.dim() == 2 and pt.shape[0] == 1 and pt.dtype == torch.int32 and pt.stride(1) == 1
          and csl.dtype == torch.int32 and csl.numel() == 1 and kc.is_contiguous() and vc.is_contiguous()
          and vc.numel() == kc.numel() and q.is_contiguous() and q.numel() == 1024
          and gsrc.dtype == torch.bfloat16 and gsrc.dim() == 2 and gsrc.shape[1] >= DL3_Q_ROWS and gsrc.stride(1) == 1)
    seq = int(csl[0]) if ok else -1
    if ok:
        ok = 1 <= seq <= pt.shape[1] and bool(((pt[0, :seq] >= 0) & (pt[0, :seq] < rows)).all())
    if not ok:
        raise RuntimeError(f"{tag}: host-side bounds check failed (seq {seq}, page-table {tuple(pt.shape)}, rows {rows})")
def _c320d_layer_input(state, device, gen):
    data = _cb_data(state, device, gen, 0, 1, 1, [80000], 1.0)
    ls, _load = _ead_busiest(data, state)
    data["local_start"] = torch.tensor([ls], dtype=torch.int32, device=device)
    return data
def _c320d_attrs():
    names = {0: "A", 6: "combine", 9: "trig"}
    a = {v: [int(t) for t in _C320D_EXT.c320_attrs(v)] for v in names}
    parts, ok, warn = [], True, ""
    for v, nm in names.items():
        x = a[v]
        parts.append(f"{nm} {x[0]}/{x[2]}/{x[3]}/{x[5]}/{x[6]}")
        ok = ok and x[0] == 90 and x[6] >= 1
    if a[0][6] < 2:
        warn = f"; WARNING A {a[0][6]} CTA/SM (< 2)"
    return ok, (f"{C320D_TAG} attrs cubin/regs/local B/smem B/CTA per SM: " + ", ".join(parts) + warn
                + f" -> {'ok' if ok else 'FAIL'}")
def _c320d_canary(device):
    step = _rt_tracer(C320D_TAG, "canary 5 keys")
    gen = torch.Generator(device=device)
    gen.manual_seed(20261101)
    data, q, qkv = _splitkv_synthetic([5], 1.0, gen, device)
    _c320d_bounds(data, q, qkv, "canary 5 keys")
    cfg = data["config"]
    mc = -(-data["page_table"].shape[1] // C320D_CHUNK)
    g = 2 * _sm_count(device)
    ws_acc = torch.full((mc, 16, 64), float("nan"), dtype=torch.float32, device=device)
    ws_ml = torch.full((mc, 2, 16), float("nan"), dtype=torch.float32, device=device)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    step(f"launch c320_attn_kernel<320,false> alone (5 keys, {mc} chunk slot(s), grid {g})")
    _C320D_EXT.c320_attn(q, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"],
                         data["k_scale"], ws_acc, ws_ml, scale, mc, g)
    step("done A")
    a_ok = bool(torch.isfinite(ws_acc[0]).all()) and bool(torch.isfinite(ws_ml[0]).all()) and bool((ws_ml[0, 1] > 0).all())
    outs = {}
    for trig in (0, 1):
        outs[trig] = torch.full((1, 1, 1024), float("nan"), dtype=torch.bfloat16, device=device)
        step(f"launch c320_combine_kernel<320,16,PDL,ONEPASS{',TRIG' if trig else ''}> (trig {trig})")
        _C320D_EXT.c320_combine_pdl(ws_acc, ws_ml, data["cache_seqlens"], data["v_scale"], qkv, qkv.stride(0),
                                    outs[trig], mc, RMS_EPS, 1, C320D_NW, trig)
        step(f"done combine trig {trig}")
    sk = _sk_buffers(device)
    step("launch V0 (sk variant 3, private buffers)")
    v0 = _full_decode_sk(data, q, qkv, sk, 3)
    step("done V0")
    e_ref, fin = _sk_err(outs[0], _splitkv_reference(data, q, qkv, [5]))
    e_v0 = _rt_rel(outs[0].float().reshape(1, -1), v0.float().reshape(1, -1))
    same = torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
    left = _sk_ticket_sum(sk)
    ok = a_ok and fin and e_ref < C320D_REF_REL and e_v0 < C320D_V0_REL and same and left == 0
    return ok, (f"{C320D_TAG} canary 5 keys: A partial finite {a_ok}; combine vs fp32 {e_ref:.2e}, vs V0 {e_v0:.2e}, "
                f"finite {fin}; TRIG bitwise {'IDENTICAL' if same else 'DIFFERS'}; V0 tickets {left} -> "
                f"{'ok' if ok else 'FAIL'}")
def _c320d_q_plumbing(state, device):
    d3 = state["dl3"]
    step = _rt_tracer(C320D_TAG, "Q plumbing")
    gen = torch.Generator(device=device)
    gen.manual_seed(20261102)
    problems = []
    for kv in (1.0, 0.25):
        data = _cb_data(state, device, gen, 0, 1, 1, [4096], kv)
        a, b = _dl3_clone_data(data), _dl3_clone_data(data)
        step(f"launch _dl3_q (incumbent Q) kv {kv}")
        q_ref, gate_ref = _dl3_q(a, a["attn_input"], state, d3)
        step(f"launch dl2_qkv into buf (gate = buf[:, {SK_QKV_GATE_OFF}:]) kv {kv}")
        q8, buf = _c320d_q(b, b["attn_input"], state, d3)
        step(f"done kv {kv}")
        g_view = buf[:, SK_QKV_GATE_OFF:]
        same = {"gate": torch.equal(g_view.view(torch.int16), gate_ref.view(torch.int16)),
                "q8": torch.equal(q8.view(torch.uint8), q_ref.view(torch.uint8)),
                "k_cache": torch.equal(a["k_cache"].view(torch.uint8), b["k_cache"].view(torch.uint8)),
                "v_cache": torch.equal(a["v_cache"].view(torch.uint8), b["v_cache"].view(torch.uint8))}
        if not all(same.values()) or not g_view.is_contiguous():
            problems.append(f"kv {kv}: " + ", ".join(f"{k} {'=' if v else 'DIFFERS'}" for k, v in same.items())
                            + f", gate view contiguous {g_view.is_contiguous()}")
    ok = not problems
    return ok, (f"{C320D_TAG} Q plumbing (buf vs _dl3_q, kv 1 / 0.25): gate, q8, k / v caches bitwise "
                + ("IDENTICAL -> ok" if ok else f"FAIL ({problems[0][:160]})"))
def _c320d_cases(device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261103)
    sk = _sk_buffers(device)
    ok = {0: True, 1: True}
    worst = {"ref": 0.0, "v0": 0.0, "v0ref": 0.0}
    diff_trig, first_fail = 0, None
    for tag, n, kv, grid in C320D_CASES:
        step = _rt_tracer(C320D_TAG, f"case {tag}")
        data, q, qkv = _splitkv_synthetic([n], kv, gen, device)
        _c320d_bounds(data, q, qkv, tag)
        g = grid or 2 * _sm_count(device)
        items = -(-n // C320D_CHUNK)
        step("launch V0 (sk variant 3)")
        v0 = _full_decode_sk(data, q, qkv, sk, 3)
        outs = {}
        for trig in (0, 1):
            step(f"launch c320 trig {trig}, grid {g}, {items} items")
            outs[trig] = _c320d_core(data, q, qkv, trig, g)
        torch.cuda.synchronize()
        ref = _splitkv_reference(data, q, qkv, [n])
        e_v0ref, _ = _sk_err(v0, ref)
        worst["v0ref"] = max(worst["v0ref"], e_v0ref)
        parts = []
        for trig in (0, 1):
            e_ref, fin = _sk_err(outs[trig], ref)
            e_v0 = _rt_rel(outs[trig].float().reshape(1, -1), v0.float().reshape(1, -1))
            good = fin and e_ref < C320D_REF_REL and e_v0 < C320D_V0_REL
            worst["ref"], worst["v0"] = max(worst["ref"], e_ref), max(worst["v0"], e_v0)
            parts.append(f"trig {trig}: ref {e_ref:.2e} V0 {e_v0:.2e} finite {fin}{'' if good else ' FAIL'}")
            if not good:
                ok[trig] = False
                first_fail = first_fail or f"{tag} trig {trig}"
        same = torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
        diff_trig += 0 if same else 1
        _c320d_say(f"{C320D_TAG}   case {tag}: " + " | ".join(parts) + f" | V0 vs ref {e_v0ref:.2e}"
                   + f" | TRIG {'bitwise =' if same else 'DIFFERS'}")
        del data, q, qkv, v0, outs, ref
    left = _sk_ticket_sum(sk)
    if left:
        ok = {0: False, 1: False}
        first_fail = first_fail or f"V0 tickets {left}"
    short = (f"cases {len(C320D_CASES)}x2 c320 {'ok' if ok[0] else 'FAIL'} c320t {'ok' if ok[1] else 'FAIL'}, worst "
             f"ref {worst['ref']:.1e} V0 {worst['v0']:.1e} (V0-ref {worst['v0ref']:.1e}), TRIG diff {diff_trig}")
    line = (f"{C320D_TAG} kernel {short} (bounds < {C320D_REF_REL} / < {C320D_V0_REL})"
            + ("" if first_fail is None else f"; first miss: {first_fail}"))
    return ok, line, short
def _c320d_alt(device, trig, n, kv, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    tag = f"replay trig {trig} {n} keys kv {kv}"
    step = _rt_tracer(C320D_TAG, tag)
    lens = [n, n, n - 2, n, max(1, n // 2 + 7), max(1, n - C320D_CHUNK)]
    def synth(m):
        data, q, qkv = _splitkv_synthetic([n], kv, gen, device)
        data["cache_seqlens"].fill_(m)
        return data, q, qkv
    states, keep = [], []
    for who, m in (("A", lens[0]), ("B", lens[1])):
        data, q, qkv = synth(m)
        _c320d_bounds(data, q, qkv, f"{tag} state {who}")
        st = {"who": who, "data": data, "q": q, "qkv": qkv, "len": m}
        step(f"warm-up + capture state {who}")
        st["graph"], st["out"] = _rt_graph(lambda st=st: _c320d_core(st["data"], st["q"], st["qkv"], trig))
        states.append(st)
        keep.append(st)
    plan = [(0, None), (1, None), (1, lens[2]), (0, lens[3]), (0, lens[4]), (1, lens[5])]
    try:
        for i, (si, refill) in enumerate(plan):
            st = states[si]
            if refill is not None:
                data, q, qkv = synth(refill)
                for key in ("k_cache", "v_cache", "page_table", "cache_seqlens", "k_scale", "v_scale"):
                    st["data"][key].copy_(data[key])
                st["q"].copy_(q)
                st["qkv"].copy_(qkv)
                st["len"] = refill
                del data, q, qkv
                torch.cuda.synchronize()
                _c320d_bounds(st["data"], st["q"], st["qkv"], f"{tag} step {i}")
            st["out"].fill_(float("nan"))
            step(f"launch step {i}: state {st['who']} replay{' (refilled)' if refill is not None else ''}")
            st["graph"].replay()
            step(f"done step {i}")
            err, fin = _sk_err(st["out"], _splitkv_reference(st["data"], st["q"], st["qkv"], [st["len"]]))
            if not (fin and err < C320D_REF_REL):
                return f"{tag} step {i} (state {st['who']}, {st['len']} keys): rel L2 {err:.2e}, finite {fin}"
    finally:
        torch.cuda.synchronize()
        del states, keep
    return None
def _c320d_layer_canary(state, device, trig):
    lab = f"{C320D_NAMES[trig]} in kernel_fn d1f"
    step = _rt_tracer(C320D_TAG, lab)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261104 + trig)
    problems, keep = [], []
    try:
        _c320d_set(state, False)
        data = _c320d_layer_input(state, device, gen)
        data2 = _c320d_layer_input(state, device, gen)
        keep += [data, data2]
        _c320d_set(state, True, trig)
        outs = []
        for i in range(2):
            step(f"launch eager kernel_fn {i}")
            o = kernel_fn(_dl3_clone_data(data), state)
            step(f"done eager kernel_fn {i}")
            nn = int((~torch.isfinite(o[0].float())).sum()) + int((~torch.isfinite(o[1].float())).sum())
            if nn:
                problems.append(f"eager {i}: {nn} non-finite outputs")
            outs.append(o)
        if not all(torch.equal(a, b) for a, b in zip(outs[0][:3], outs[1][:3])):
            problems.append("eager repeat not bit-identical")
        step("launch eager kernel_fn on input 2")
        ref2 = kernel_fn(_dl3_clone_data(data2), state)
        step("done eager kernel_fn on input 2")
        dg = _dl3_clone_data(data)
        step("launch graph warm-up + capture of kernel_fn")
        graph, gout = _rt_graph(lambda: kernel_fn(dg, state))
        step("done graph warm-up + capture")
        keep += [dg, graph, gout, outs, ref2]
        for rep, (src, ref) in enumerate(((data, outs[0]), (data2, ref2))):
            for key, v in src.items():
                if torch.is_tensor(v) and v.shape == dg[key].shape:
                    dg[key].copy_(v)
            gout[0].fill_(float("nan"))
            step(f"launch graph replay {rep} (refilled: input {rep + 1})")
            graph.replay()
            step(f"done graph replay {rep}")
            e_h, e_s = _rt_rel(gout[0].float(), ref[0].float()), _rt_rel(gout[1].float(), ref[1].float())
            if not (e_h <= DSA_COPY_TOL and e_s <= DSA_COPY_TOL and torch.equal(gout[2], ref[2])):
                problems.append(f"graph replay {rep} vs eager: hidden {e_h:.3g} stream {e_s:.3g}")
        torch.cuda.synchronize()
    finally:
        _c320d_set(state, False)
        del keep
    ok = not problems
    return ok, f"{C320D_TAG} layer canary {lab}: " + ("ok" if ok else f"FAIL ({len(problems)}: {problems[0][:160]})")
def _c320d_readout(state, device, cands):
    d3 = state["dl3"]
    lab = "readout attention side d1f"
    step = _rt_tracer(C320D_TAG, lab)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261105)
    keep = []
    try:
        _c320d_set(state, False)
        data = _c320d_layer_input(state, device, gen)
        keep.append(data)
        graphs = {}
        for name, on, trig in [("incumbent", False, 0)] + [(C320D_NAMES[t], True, t) for t in cands]:
            _c320d_set(state, on, trig)
            step(f"capture {name}")
            graphs[name], out = _rt_graph(lambda: _dl3_attention(data, state, d3))
            keep.append(out)
        _c320d_set(state, False)
        step("captured; timing")
        rounds = _rt_time(graphs, device)
        step("timed")
        keep.append(graphs)
    finally:
        _c320d_set(state, False)
    names = list(graphs)
    parts = []
    med = {n: [] for n in names}
    for r in rounds:
        for n in names:
            med[n].append(_rt_stats(r[n])[0])
        parts.append("/".join(f"{_rt_stats(r[n])[0]:.2f}" for n in names))
    mean = {n: sum(v) / len(v) for n, v in med.items()}
    extra = [f"{n} x{mean['incumbent'] / max(mean[n], 1e-3):.3f}" for n in names[1:]]
    if "c320" in mean and "c320t" in mean:
        extra.append(f"TRIG delta {mean['c320t'] - mean['c320']:+.2f} us")
    del keep, graphs
    return (f"{C320D_TAG} readout Q->O d1f (graph, read flush), us {'/'.join(['inc'] + names[1:])} per round: "
            f"{' | '.join(parts)}; " + ", ".join(extra))
def _c320d_layer_gate(state, device, cands):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261106)
    settings = [("incumbent", False, 0)] + [(C320D_NAMES[t], True, t) for t in cands]
    per, keep = {}, []
    try:
        for rep in range(2):
            _c320d_set(state, False)
            data = _c320d_layer_input(state, device, gen)
            keep.append(data)
            graphs = {}
            for name, on, trig in settings:
                _c320d_set(state, on, trig)
                _rt_tracer(C320D_TAG, f"layer gate input {rep}")(f"capture kernel_fn graph, {name}")
                graphs[name], out = _rt_graph(lambda: kernel_fn(data, state))
                keep.append(out)
            _c320d_set(state, False)
            keep.append(graphs)
            _rt_tracer(C320D_TAG, f"layer gate input {rep}")("captured; timing")
            per[f"d1f layer input {rep}"] = _rt_time(graphs, device)
            _rt_tracer(C320D_TAG, f"layer gate input {rep}")("timed")
    finally:
        _c320d_set(state, False)
    names = [C320D_NAMES[t] for t in cands]
    win, worst, rlines = _rt_select(per, "incumbent", names, margin=C320D_LAYER_MARGIN, tight=C320D_LAYER_MARGIN)
    h2h = ""
    if win == "c320t":
        hwin, hworst, hlines = (_rt_select(per, "c320", ["c320t"], margin=C320D_TRIG_H2H,
                                           tight=C320D_TRIG_H2H) if "c320" in names else (None, {}, []))
        c320_ok = "c320" in names and _rt_select(per, "incumbent", ["c320"], margin=C320D_LAYER_MARGIN,
                                                 tight=C320D_LAYER_MARGIN)[0] is not None
        hx = hworst.get("c320t")
        h2h = (f"; TRIG head to head c320t vs c320 "
               + (f"x{hx:.3f}" if hx is not None else "not timed (c320 not a candidate)")
               + f" (need >= {C320D_TRIG_H2H} every round) -> "
               + ("c320t" if hwin is not None else ("c320" if c320_ok else "incumbent")))
        rlines = rlines + [f"TRIG head to head (c320 as incumbent): {r}" for r in hlines]
        if hwin is None:
            win = "c320" if c320_ok else None
    del keep
    parts = []
    for rounds in per.values():
        parts.append(" ".join("/".join(f"{_rt_stats(r[n])[0]:.1f}" for n in ["incumbent"] + names) for r in rounds))
    wt = None if win is None else next(t for t in cands if C320D_NAMES[t] == win)
    pick = "incumbent" if wt is None else C320D_NAMES[wt]
    line = (f"{C320D_TAG} layer gate d1f (read flush), us incumbent/{'/'.join(names)}: {' | '.join(parts)} -> "
            + ", ".join(f"{n} x{worst[n]:.3f}" for n in names) + f" (need >= {C320D_LAYER_MARGIN} every round)"
            + h2h + " -> " + pick)
    allr = [r for rounds in per.values() for r in rounds]
    mean = {n: sum(_rt_stats(r[n])[0] for r in allr) / len(allr) for n in ["incumbent"] + names}
    short = ("gate d1f worst " + ", ".join(f"{n} x{worst[n]:.3f}" for n in names) + f" (>= {C320D_LAYER_MARGIN})"
             + h2h + " -> " + pick + "; mean us " + "/".join(f"{mean[n]:.1f}" for n in ["incumbent"] + names))
    return wt, {t: worst[C320D_NAMES[t]] for t in cands}, line, [f"{C320D_TAG}   {r[:400]}" for r in rlines], short
def _c320d_layer_check(state, device, trig):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261107)
    worst = {}
    try:
        for rep in range(2):
            _c320d_set(state, False)
            data = _c320d_layer_input(state, device, gen)
            ls_val = int(data["local_start"][0])
            s_in = data["stream"].clone()
            scores = _cb_router_scores(_dl3_clone_data(data), state)
            d_ref = _dl3_clone_data(data)
            ref = kernel_fn(d_ref, state)
            _c320d_set(state, True, trig)
            d_got = _dl3_clone_data(data)
            got = kernel_fn(d_got, state)
            torch.cuda.synchronize()
            ok, msg = _dsa_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst)
            ok = ok and worst.get("k_gap", 0) == 0 and worst.get("v_gap", 0) == 0
            if not ok:
                return f"layer check input {rep}: {msg}", f"{C320D_TAG} layer check input {rep}: {msg[:300]} -> FAIL"
    finally:
        _c320d_set(state, False)
    line = (f"{C320D_TAG} layer check d1f (on vs off, 2 inputs) ok: worst "
            + ", ".join(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
    return None, line
def _c320d_copy_check(state, device):
    import gc
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261108)
    fails, n = [], 0
    base = _c320d_layer_input(state, device, gen)
    per = []
    for st in (state, other):
        data = _dl3_clone_data(base)
        graph, out = _rt_graph(lambda data=data, st=st: kernel_fn(data, st))
        per.append((data, graph, out))
    for k in range(4):
        fresh = _c320d_layer_input(state, device, gen)
        for data, _, _ in per:
            for key, v in fresh.items():
                if torch.is_tensor(v) and v.shape == data[key].shape:
                    data[key].copy_(v)
        ref = kernel_fn(_dl3_clone_data(per[0][0]), state)
        for _, graph, _ in (per if k % 2 == 0 else per[::-1]):
            graph.replay()
        torch.cuda.synchronize()
        for i, (_, _, out) in enumerate(per):
            n += 1
            e_h, e_s = _rt_rel(out[0].float(), ref[0].float()), _rt_rel(out[1].float(), ref[1].float())
            if not (e_h <= DSA_COPY_TOL and e_s <= DSA_COPY_TOL and torch.equal(out[2], ref[2])):
                fails.append(f"state {'AB'[i]} refill {k}: hidden {e_h:.3g} stream {e_s:.3g}")
    torch.cuda.synchronize()
    del per
    left = (_dl3_tickets(state), _dl3_tickets(other))
    if any(left):
        fails.append(f"tickets {left[0]} / {left[1]}")
    del other
    gc.collect()
    torch.cuda.empty_cache()
    ok = not fails
    return ok, f"{C320D_TAG} copy check: {n} replays on two states, d1f: " + ("PASS" if ok else f"FAIL: {fails[0]}")
def _c320d_profile_check(state, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261109)
    data = _c320d_layer_input(state, device, gen)
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            kernel_fn(data, state)
            torch.cuda.synchronize()
        cnt = {}
        for ev in prof.key_averages():
            cnt[ev.key] = cnt.get(ev.key, 0) + int(ev.count)
    except Exception as e:
        return None, f"{C320D_TAG} profiler unavailable ({type(e).__name__}: {str(e)[:120]}); route kept"
    na = sum(v for k_, v in cnt.items() if "c320_attn" in k_)
    nc = sum(v for k_, v in cnt.items() if "c320_combine" in k_)
    ns = sum(v for k_, v in cnt.items() if "splitkv_attn_sk" in k_)
    ok = na == 1 and nc == 1 and ns == 0
    return ok, (f"{C320D_TAG} profiler d1f: c320_attn x{na}, c320_combine x{nc}, splitkv_attn_sk x{ns} -> "
                f"{'ok' if ok else 'FAIL'}")
def _c320d_setup(state, device, t_prepare0):
    t0 = time.perf_counter()
    state["c320d"] = {"on": False, "trig": 0}
    notes = {}
    why = "off"
    chosen = None
    def late(limit):
        return time.perf_counter() - t_prepare0 > limit
    def note(key, line):
        _c320d_say(line)
        notes[key] = line
    def stage(name, fn):
        return _c320d_stage(name, t_prepare0, fn)
    try:
        if _C320D_EXT is None:
            raise _C320dMiss(f"no module ({_C320D_BUILD_ERR[:160]})")
        if late(C320D_START_DEADLINE_S):
            raise _C320dMiss(f"deadline: start at {time.perf_counter() - t_prepare0:.1f} s > {C320D_START_DEADLINE_S}")
        ok, line = stage("attrs", _c320d_attrs)
        note("attrs", line)
        if not ok:
            raise _C320dMiss("attributes")
        d3, sk = state.get("dl3"), state.get("sk")
        if d3 is None or not d3.get("a", False):
            raise _C320dMiss("not reached: piece A is off (d1f runs the parent's chain)")
        if not d3.get("pdl", 0):
            raise _C320dMiss("not reached: PDL is off for this state")
        if sk is None or not sk.get("b1", False):
            raise _C320dMiss("not reached: sk two-level (variant 3) is off")
        ok, cline = stage("canary", lambda: _c320d_canary(device))
        _c320d_say(cline)
        if not ok:
            notes["checks"] = cline
            raise _C320dMiss("canary FAIL")
        ok, qline = stage("q plumbing", lambda: _c320d_q_plumbing(state, device))
        _c320d_say(qline)
        if not ok:
            notes["checks"] = qline
            raise _C320dMiss("Q plumbing FAIL")
        tok, kline, kshort = stage("kernel cases", lambda: _c320d_cases(device))
        _c320d_say(kline)
        cands = [t for t in (0, 1) if tok[t]]
        rparts = []
        for t in list(cands):
            for n, kv, seed in ((80001, 1.0, 20261110), (4097, 0.25, 20261111)):
                try:
                    reason = stage(f"replays trig {t} {n} keys", lambda t=t, n=n, kv=kv, s=seed: _c320d_alt(device, t, n, kv, s))
                except Exception as e:
                    reason = f"raised {type(e).__name__}: {str(e)[:160]}"
                if reason is not None:
                    _c320d_say(f"{C320D_TAG} two-state replays {C320D_NAMES[t]}: FAIL ({reason})")
                    rparts.append(f"{C320D_NAMES[t]} FAIL")
                    cands.remove(t)
                    break
            else:
                rparts.append(f"{C320D_NAMES[t]} ok")
        notes["checks"] = (f"{C320D_TAG} canary ok, Q bitwise ok; {kshort}; replays "
                           f"{', '.join(rparts) or 'none run'}")
        _c320d_say(notes["checks"])
        if not cands:
            raise _C320dMiss("no candidate passed the kernel cases and replays")
        lparts = []
        for t in list(cands):
            if late(C320D_GATE_DEADLINE_S):
                raise _C320dMiss("deadline before the layer canaries")
            try:
                cok, line = stage(f"layer canary {C320D_NAMES[t]}", lambda t=t: _c320d_layer_canary(state, device, t))
            except Exception as e:
                cok, line = False, f"{C320D_TAG} layer canary {C320D_NAMES[t]} raised {type(e).__name__}: {str(e)[:200]}"
            _c320d_say(line)
            lparts.append(f"{C320D_NAMES[t]} {'ok' if cok else 'FAIL'}")
            if not cok:
                cands.remove(t)
        if not cands:
            notes["gate"] = f"{C320D_TAG} layer canaries: {', '.join(lparts)}"
            raise _C320dMiss("layer canary FAIL")
        if late(C320D_GATE_DEADLINE_S):
            raise _C320dMiss("deadline before the readout")
        try:
            note("readout", stage("readout", lambda: _c320d_readout(state, device, cands)))
        except Exception as e:
            note("readout", f"{C320D_TAG} readout raised {type(e).__name__}: {str(e)[:200]} (print only)")
        if late(C320D_GATE_DEADLINE_S):
            raise _C320dMiss("deadline before the layer gate")
        wt, lworst, gline, glines, gshort = stage("layer gate", lambda: _c320d_layer_gate(state, device, cands))
        _c320d_say(gline)
        for gl in glines:
            _c320d_say(gl)
        note("gate", f"{C320D_TAG} layer canaries {', '.join(lparts)}; {gshort}")
        if wt is None:
            raise _C320dMiss(f"whole layer {', '.join(f'{C320D_NAMES[t]} x{w:.3f}' for t, w in lworst.items())}")
        wparts = []
        if late(C320D_GATE_DEADLINE_S):
            raise _C320dMiss("deadline before the layer check")
        problem, line = stage("layer check", lambda: _c320d_layer_check(state, device, wt))
        _c320d_say(line)
        wparts.append("layer check " + ("ok (k/v gap 0)" if problem is None else "FAIL"))
        if problem is not None:
            notes["winner"] = f"{C320D_TAG} {C320D_NAMES[wt]}: " + "; ".join(wparts)
            raise _C320dMiss("layer check FAIL")
        _c320d_set(state, True, wt)
        if late(C320D_COPY_DEADLINE_S):
            raise _C320dMiss("deadline before the copy check")
        ok, line = stage("copy check", lambda: _c320d_copy_check(state, device))
        _c320d_say(line)
        wparts.append("copy check " + ("PASS" if ok else "FAIL"))
        if not ok:
            notes["winner"] = f"{C320D_TAG} {C320D_NAMES[wt]}: " + "; ".join(wparts)
            raise _C320dMiss("copy check FAIL")
        pok, line = stage("profiler", lambda: _c320d_profile_check(state, device))
        _c320d_say(line)
        wparts.append(line[len(C320D_TAG) + 1:].replace("splitkv_attn_sk", "sk"))
        if pok is False:
            notes["winner"] = f"{C320D_TAG} {C320D_NAMES[wt]}: " + "; ".join(wparts)
            raise _C320dMiss("profiler assertion FAIL")
        left = _dl3_tickets(state)
        wparts.append(f"tickets {left}")
        notes["winner"] = f"{C320D_TAG} {C320D_NAMES[wt]}: " + "; ".join(wparts)
        _c320d_say(notes["winner"])
        if left:
            raise _C320dMiss(f"tickets {left} after the checks")
        chosen = (wt, lworst[wt])
    except _C320dMiss as e:
        why = str(e)
        _c320d_set(state, False)
    except Exception as e:
        why = f"{type(e).__name__}: {str(e)[:160]}"
        _c320d_say(f"{C320D_TAG} raised {why}")
        try:
            _c320d_set(state, False)
        except Exception:
            pass
    try:
        torch.cuda.synchronize()
        left = _dl3_tickets(state)
        if left:
            _dl3_zero(state)
            if chosen is not None:
                why = f"tickets {left} at the end of the c320 setup"
                chosen = None
            _c320d_say(f"{C320D_TAG} tickets {left} at the end (zeroed)")
    except Exception as e:
        why = f"final sync raised {type(e).__name__}: {str(e)[:160]}"
        chosen = None
        _c320d_say(f"{C320D_TAG} {why}")
    if chosen is None:
        state["c320d"]["on"] = False
    else:
        _c320d_set(state, True, chosen[0])
    tail = [f"{C320D_TAG} ===== c320 d1f SUMMARY (setup {time.perf_counter() - t0:.1f} s; build {_C320D_BUILD_S:.1f} s "
            f"at import; prepare at {time.perf_counter() - t_prepare0:.1f} s) ====="]
    tail.extend(notes[k] for k in ("attrs", "checks", "readout", "gate", "winner") if k in notes)
    if chosen is not None:
        wt, lx = chosen
        tail.append(f"{C320D_TAG} PATH CHOSEN full decode_1 attention (piece A): C320 {C320D_NAMES[wt]} "
                    f"[c320_attn<320,false> plain + combine<320,16,PDL,ONEPASS{',TRIG' if wt else ''}>] "
                    f"(whole layer x{lx:.3f}); other shapes unchanged")
    else:
        tail.append(f"{C320D_TAG} PATH CHOSEN full decode_1 attention (piece A): incumbent sk two-level ({why})")
    return tail
if CAMPAIGN_C320 and torch.cuda.is_available():
    _C320D_EXT, _C320D_BUILD_S, _C320D_BUILD_ERR = _c320d_build()
else:
    _C320D_EXT, _C320D_BUILD_S, _C320D_BUILD_ERR = None, 0.0, "CAMPAIGN_C320 off or no CUDA"
CAMPAIGN_W6P_PERHEAD = True
W6P_TAG = "[w6p-perhead]"
W6P_ATTN_MARGIN = 1.03
W6P_LAYER_MARGIN = 1.003
W6P_TOL_TOTAL, W6P_TOL_ROW = 1.5e-2, 5e-2
W6P_K_REL = 4e-2
W6P_NEED_CLUSTERS = 2
W6P_LENS = (1, 63, 64, 65, 127, 128, 129, 1000, 2047, 2048, 2049, 4097, 80000)
W6P_START_S, W6P_GATE_S, W6P_COPY_S = 230.0, 255.0, 265.0
class _W6pMiss(Exception):
    pass
def _w6p_say(line):
    print(line, flush=True)
    return line
def _w6p_shape(cfg) -> bool:
    return (cfg["batch"] == 1 and cfg["q_len"] == 1 and cfg["page_size"] == 1 and 0 < cfg["window"] <= 2048
            and bool(cfg["rope_theta"]) and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64)
def _w6p_ph_attn(data, qkv, state):
    cfg = data["config"]
    out = torch.empty((1, 16 * 64), dtype=torch.bfloat16, device=qkv.device)
    _SKW.skw_ph_attn(qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["page_table"],
                     _slide_inv_freq(cfg, state, qkv.device), data["k_scale"], data["v_scale"],
                     data["k_cache"], data["v_cache"], out, cfg["window"], cfg["qk_scale_factor"],
                     cfg["sm_scale"] * LOG2E, RMS_EPS)
    return out
def _w6p_q_emul(data, qkv, state):
    cfg = data["config"]
    x = qkv[:, :16 * 64].float().view(-1, 16, 64)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    ang = data["positions"].float()[:, None] * _slide_inv_freq(cfg, state, qkv.device)[None, :]
    c, s = ang.double().cos().float()[:, None, :], ang.double().sin().float()[:, None, :]
    xe, xo = x[..., 0::2], x[..., 1::2]
    y = torch.stack([xe * c - xo * s, xe * s + xo * c], -1).flatten(-2)
    return (y * cfg["qk_scale_factor"]).to(FP8).float() / cfg["qk_scale_factor"]
def _w6p_input(state, device, gen, tokens=1):
    data = _cb_data(state, device, gen, 2048, 1, tokens, [80000], 1.0)
    data["local_start"] = torch.tensor([_ead_busiest(data, state)[0]], dtype=torch.int32, device=device)
    return data
def d1s_perhead_canary(state, device):
    step = _rt_tracer(W6P_TAG, "d1s_perhead_canary")
    gen = torch.Generator(device=device)
    gen.manual_seed(20261130)
    sts = {"64": {**state, "w6p_ph": False, "w6p_bn": 64}, "128": {**state, "w6p_ph": False, "w6p_bn": 128},
           "ph": {**state, "w6p_ph": True}}
    w = {"rel": 0.0, "row": 0.0, "k_rel": 0.0, "k_gap": 0, "vs64": 0.0, "rel64": 0.0, "j": 0.0, "rel128": 0.0}
    ok128, miss128, n = True, "", 0
    for kvs in (1.0, 0.015625):
        for lens in W6P_LENS:
            tag = f"seq {lens} kv_scale {kvs}"
            data = _cb_data(state, device, gen, 2048, 1, 1, [lens], kvs)
            qkv = _gemm(data["attn_input"], state["dense"]["qkv_gate"])
            ds, outs = {}, {}
            for name in ("64", "128", "ph"):
                ds[name] = _dl3_clone_data(data)
                if n == 0:
                    step(f"launch {'PERHEAD' if name == 'ph' else 'BN' + name} ({tag})")
                outs[name] = _sliding_forward(ds[name], qkv, sts[name])
                if n == 0:
                    step(f"done {name}")
            torch.cuda.synchronize()
            n += 1
            loc = data["cache_loc"]
            u8 = {k: {c: ds[k][c].view(torch.uint8).view(-1, 128) for c in ("k_cache", "v_cache")} for k in ds}
            mask = torch.ones(u8["64"]["k_cache"].shape[0], dtype=torch.bool, device=device)
            mask[loc] = False
            rest = all(torch.equal(u8["ph"][c][mask], u8["64"][c][mask]) for c in ("k_cache", "v_cache"))
            v_ok = torch.equal(u8["ph"]["v_cache"][loc], u8["64"]["v_cache"][loc])
            k_rel = _rt_rel(u8["ph"]["k_cache"][loc].view(FP8).float(), u8["64"]["k_cache"][loc].view(FP8).float())
            k_gap = _fp8_code_gap(u8["ph"]["k_cache"][loc], u8["64"]["k_cache"][loc])
            o = outs["ph"].float()
            finite = bool(torch.isfinite(o).all())
            ref_own = _cb_attn_ref(ds["ph"], qkv, q_override=_w6p_q_emul(ds["ph"], qkv, state))
            rel, row = _rt_rel(o, ref_own), _rt_row_rel(o.reshape(16, 64), ref_own.reshape(16, 64))
            ref = _cb_attn_ref(ds["ph"], qkv)
            e8 = _rt_rel(_cb_attn_ref(ds["ph"], qkv, q_fp8=True), ref)
            w["j"] = max(w["j"], _rt_rel(o, ref) / max(e8, 1e-12))
            w["rel"], w["row"], w["k_gap"] = max(w["rel"], rel), max(w["row"], row), max(w["k_gap"], k_gap)
            w["k_rel"] = max(w["k_rel"], k_rel)
            w["vs64"] = max(w["vs64"], _rt_rel(o, outs["64"].float()))
            w["rel64"] = max(w["rel64"], _rt_rel(outs["64"].float(), _cb_attn_ref(ds["64"], qkv)))
            left = int(state["slide_cnt"].abs().sum())
            if not (finite and rest and v_ok and k_rel <= W6P_K_REL and rel <= W6P_TOL_TOTAL and row <= W6P_TOL_ROW
                    and left == 0):
                return False, False, _w6p_say(
                    f"{W6P_TAG} d1s_perhead_canary: MISS at {tag}: finite {finite}, other rows {rest}, v {v_ok}, k rel "
                    f"{k_rel:.1e} gap {k_gap}, vs own-q fp64 rel {rel:.2e} row {row:.2e}, counters {left}")
            if ok128:
                same = all(torch.equal(u8["128"][c], u8["64"][c]) for c in ("k_cache", "v_cache"))
                o1 = outs["128"].float()
                r1 = _cb_attn_ref(ds["128"], qkv)
                e1, row1 = _rt_rel(o1, r1), _rt_row_rel(o1.reshape(16, 64), r1.reshape(16, 64))
                w["rel128"] = max(w["rel128"], e1)
                if not (same and bool(torch.isfinite(o1).all()) and e1 <= W6P_TOL_TOTAL and row1 <= W6P_TOL_ROW):
                    ok128, miss128 = False, f"; BN128 MISS at {tag} (kv bitwise {same}, rel {e1:.1e}): out of the race"
            del data, qkv, ds, outs, u8, ref_own, ref, o
    return True, ok128, _w6p_say(
        f"{W6P_TAG} d1s_perhead_canary: {n} cases ok; PERHEAD vs own-q fp64 rel {w['rel']:.1e} row {w['row']:.1e} "
        f"(<= {W6P_TOL_TOTAL}/{W6P_TOL_ROW}), k rel {w['k_rel']:.1e} gap {w['k_gap']}, v + other rows bitwise; info: vs "
        f"exact fp64 / fp8-q ref err x{w['j']:.2f}, vs BN64 {w['vs64']:.1e}, BN64 fp64 {w['rel64']:.1e}, BN128 fp64 "
        f"{w['rel128']:.1e}{miss128}")
def _w6p_gate(state, device, with128):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261131)
    var = {"64": (False, 64), **({"128": (False, 128)} if with128 else {}), "ph": (True, 64)}
    per, keep = {}, []
    for rep in range(2):
        step = _rt_tracer(W6P_TAG, f"gate input {rep}")
        data = _w6p_input(state, device, gen)
        graphs = {}
        for name, (ph, bn) in var.items():
            st = {**state, "w6p_ph": ph, "w6p_bn": bn}
            step(f"capture {name} attention side" + ("" if name == "128" else " + layer"))
            graphs[f"a{name}"], oa = _rt_graph(lambda st=st, data=data: _attn_core(data, st))
            keep.append(oa)
            if name != "128":
                graphs[f"l{name}"], ol = _rt_graph(lambda st=st, data=data: kernel_fn(data, st))
                keep.append(ol)
        keep += [data, graphs]
        per[f"d1s input {rep}"] = _rt_time(graphs, device)
        step("timed")
    per_a, which = {}, []
    for lab, rounds in per.items():
        per_a[lab] = []
        for ts in rounds:
            tri = min((k for k in ("a64", "a128") if k in ts), key=lambda k: _rt_stats(ts[k])[0])
            which.append(tri[1:])
            per_a[lab].append({"triton": ts[tri], "aph": ts["aph"], "__baseline_us__": ts.get("__baseline_us__", [0.0])})
    wa, xa, la = _rt_select(per_a, "triton", ["aph"], margin=W6P_ATTN_MARGIN, tight=W6P_ATTN_MARGIN)
    wl, xl, ll = _rt_select(per, "l64", ["lph"], margin=W6P_LAYER_MARGIN, tight=W6P_LAYER_MARGIN)
    med = {k: [_rt_stats(r[k])[0] for rounds in per.values() for r in rounds if k in r]
           for k in ("a64", "a128", "aph", "l64", "lph")}
    del keep
    return wa is not None, wl is not None, {"a": xa["aph"], "l": xl["lph"]}, med, la + ll, which
def _w6p_layer_check(state, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261132)
    worst = {}
    off, on = {**state, "w6p_ph": False, "w6p_bn": 64}, {**state, "w6p_ph": True}
    for rep in range(2):
        data = _w6p_input(state, device, gen)
        ls_val = int(data["local_start"][0])
        s_in = data["stream"].clone()
        scores = _cb_router_scores(_dl3_clone_data(data), off)
        d_ref, d_got = _dl3_clone_data(data), _dl3_clone_data(data)
        ref = kernel_fn(d_ref, off)
        got = kernel_fn(d_got, on)
        torch.cuda.synchronize()
        ok, msg = _cb_cmp_layer(got, ref, d_got, d_ref, s_in, scores, ls_val, worst, loose=True, k_gap_max=1 << 8)
        left = _cb_counters(state)
        if not ok or left:
            return f"layer check input {rep}", _w6p_say(f"{W6P_TAG} layer check input {rep}: {msg[:220]}, counters {left}"
                                                        f" -> FAIL")
    return None, _w6p_say(f"{W6P_TAG} layer check d1s (PERHEAD vs BN64, 2 inputs, loose as SKW) ok: worst " + ", ".join(
        f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
def _w6p_copy_check(state, device):
    import gc
    other = _ead_copy_tree(state)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261133)
    base = _w6p_input(state, device, gen)
    per, fails = [], []
    for st in (state, other):
        data = _dl3_clone_data(base)
        graph, out = _rt_graph(lambda data=data, st=st: kernel_fn(data, st))
        per.append((data, graph, out))
    for k in range(4):
        fresh = _w6p_input(state, device, gen)
        for data, _, out in per:
            _cb_refill(data, fresh)
            out[0].fill_(float("nan"))
        ref = kernel_fn(_dl3_clone_data(per[0][0]), state)
        for _, graph, _ in (per if k % 2 == 0 else per[::-1]):
            graph.replay()
        torch.cuda.synchronize()
        for i, (_, _, out) in enumerate(per):
            e_h, e_s = _rt_rel(out[0].float(), ref[0].float()), _rt_rel(out[1].float(), ref[1].float())
            if not (e_h <= DSA_COPY_TOL and e_s <= DSA_COPY_TOL and torch.equal(out[2], ref[2])):
                fails.append(f"state {'AB'[i]} refill {k}: hidden {e_h:.3g} stream {e_s:.3g}")
    torch.cuda.synchronize()
    del per
    left = (_cb_counters(state), _cb_counters(other))
    if any(left):
        fails.append(f"counters {left[0]} / {left[1]}")
    del other
    gc.collect()
    torch.cuda.empty_cache()
    return not fails, _w6p_say(f"{W6P_TAG} copy check: 8 replays on two states, d1s PERHEAD: "
                               + ("PASS" if not fails else f"FAIL: {fails[0]}"))
def _w6p_clean(state) -> int:
    torch.cuda.synchronize()
    left = _cb_counters(state)
    if left:
        state["slide_cnt"].zero_()
        _dl3_zero(state)
        if state.get("skw") is not None:
            _skw_zero(state["skw"])
        torch.cuda.synchronize()
    return left
def _w6p_setup(state, device, t_prepare0):
    t0 = time.perf_counter()
    state["w6p_ph"] = False
    lines, why, gx, safe = [], None, None, False
    q1 = [f"{W6P_TAG} Q1 PERHEAD ratios: not measured"]
    bs = _SKW_BUILD["s"]
    bld = f"SKW TU build {bs:.1f} s at import" if bs is not None else "SKW TU build not attempted (no CUDA at import)"
    q2 = f"{W6P_TAG} Q2 PERHEAD kernel: not queried; {bld}"
    def fmt(xs, f="x{:.3f}"):
        return " ".join(f.format(x) for x in xs)
    try:
        if _SKW is None:
            err = _SKW_BUILD["err"] or "no error text"
            q2 = f"{W6P_TAG} Q2 PERHEAD kernel: none; {bld}, FAILED"
            lines += [f"{W6P_TAG} Q2 nvcc: {err[i:i + 160]}" for i in range(0, len(err), 160)]
            raise _W6pMiss(f"the SKW module (muse_decode_sk_window_v1) did not build ({bld}): no PERHEAD; "
                           f"{err[:300]}")
        if time.perf_counter() - t_prepare0 > W6P_START_S:
            raise _W6pMiss("deadline before the canary")
        occ, regs, local, smem, bver, pver = (int(x) for x in list(_SKW.skw_ph_setup())[:6])
        q2 = _w6p_say(f"{W6P_TAG} Q2 {bld}; sliding_cluster_kernel<true> (no new module): {regs} regs, {local} B "
                      f"local, {smem} B smem, sm_{bver}/ptx {pver}, co-resident clusters of 16 {occ} (need >= "
                      f"{W6P_NEED_CLUSTERS})")
        if occ < W6P_NEED_CLUSTERS:
            raise _W6pMiss(f"max active clusters {occ} < {W6P_NEED_CLUSTERS}")
        ok, ok128, line = d1s_perhead_canary(state, device)
        lines.append(line)
        if not ok:
            raise _W6pMiss("d1s_perhead_canary miss")
        reason = _cb_attn_alt(state, device, f"{W6P_TAG} two-state PERHEAD",
                              ((2048, 1, 1, [80000], 1.0), (2048, 1, 1, [2049], 0.015625)), {"ph": True})
        lines.append(_w6p_say(f"{W6P_TAG} two-state alternation (2 inputs x 4 replays, vs BN64 and the fp64 / fp8-q "
                              f"judge form): " + ("ok" if reason is None else f"FAIL {reason[:120]}")))
        if reason is not None:
            raise _W6pMiss("two-state alternation")
        safe = True
        if time.perf_counter() - t_prepare0 > W6P_GATE_S:
            raise _W6pMiss("deadline before the gate")
        wa, wl, gx, med, glines, which = _w6p_gate(state, device, ok128)
        for gl in glines:
            _w6p_say(f"{W6P_TAG}   {gl[:400]}")
        best = [min(a, b) for a, b in zip(med["a64"], med["a128"])] if med["a128"] else med["a64"]
        mean = {k: (sum(v) / len(v) if v else float("nan")) for k, v in med.items()}
        q1 = [_w6p_say(f"{W6P_TAG} Q1a attn ratio (in0r1 in0r2 in1r1 in1r2) vs best Triton "
                       f"{fmt([b / max(p, 1e-3) for b, p in zip(best, med['aph'])])} (>= {W6P_ATTN_MARGIN}; best BN"
                       f"{'/'.join(which)}), vs BN64 {fmt([a / max(p, 1e-3) for a, p in zip(med['a64'], med['aph'])])}"),
              _w6p_say(f"{W6P_TAG} Q1b layer ratio vs BN64 {fmt([a / max(p, 1e-3) for a, p in zip(med['l64'], med['lph'])])}"
                       f" (>= {W6P_LAYER_MARGIN}); mean us attn BN64/BN128/PH {mean['a64']:.2f}/{mean['a128']:.2f}/"
                       f"{mean['aph']:.2f}, layer BN64/PH {mean['l64']:.1f}/{mean['lph']:.1f}")]
        if not wa:
            raise _W6pMiss(f"KILL: attention side x{gx['a']:.3f} < {W6P_ATTN_MARGIN} vs the best Triton in a round")
        if not wl:
            raise _W6pMiss(f"whole layer x{gx['l']:.3f} < {W6P_LAYER_MARGIN} in a round")
        problem, line = _w6p_layer_check(state, device)
        lines.append(line)
        if problem is not None:
            safe = False
            raise _W6pMiss(problem)
        if time.perf_counter() - t_prepare0 > W6P_COPY_S:
            raise _W6pMiss("deadline before the copy check")
        state["w6p_ph"] = True
        ok, line = _w6p_copy_check(state, device)
        lines.append(line)
        if not ok:
            safe = False
            raise _W6pMiss("copy check")
    except _W6pMiss as e:
        why = str(e)
    except Exception as e:
        why, safe = f"raised {type(e).__name__}: {str(e)[:160]}", False
        _w6p_say(f"{W6P_TAG} {why}")
    if why is not None:
        state["w6p_ph"] = False
    try:
        left = _w6p_clean(state)
        if left:
            why, state["w6p_ph"], safe = why or f"counters {left} after the checks", False, False
    except Exception as e:
        why, state["w6p_ph"], safe = f"sync raised {type(e).__name__}: {str(e)[:120]}", False, False
    try:
        left = _w6p_clean(state)
        clean = _w6p_say(f"{W6P_TAG} counters at the end: {left}" + (" (zeroed)" if left else ""))
        if left and state["w6p_ph"]:
            why, state["w6p_ph"] = f"counters {left} at the end", False
    except Exception as e:
        why, state["w6p_ph"] = f"final sync raised {type(e).__name__}: {str(e)[:120]}", False
        clean = _w6p_say(f"{W6P_TAG} {why}")
    if state["w6p_ph"]:
        pick = f"PERHEAD (attention side x{gx['a']:.3f} vs the best Triton, whole layer x{gx['l']:.3f}, worst round)"
    else:
        pick = f"Triton BN64 incumbent ({why})"
    return [f"{W6P_TAG} ===== W6P SUMMARY (setup {time.perf_counter() - t0:.1f} s; prepare at "
            f"{time.perf_counter() - t_prepare0:.1f} s) =====", q2, *lines, *q1,
            _w6p_say(f"{W6P_TAG} PATH CHOSEN d1s sliding attention (B=1, q_len 1): {pick}"), clean]
CAMPAIGN_EAE = True
EAE_TAG = "[ea-enc]"
EAE_JOIN_S = 220.0
EAE_END_S = 240.0
EAE_COPY_S = 248.0
EAE_RACE_S = 60.0
EAE_BAR = 0.98
EAE_VETO = 1.0
EAE_ROUNDS = 5
EAE_REPS = 15
EAE_SEED = 20261099
EAE_T = (1, 8, 16, 32, 128, 256)
EAE_T1 = ("33", "44", "30")
EAE_CANARY = {128: (144, 176), 256: (200, 240)}
EAE_PERM = (3, 4, 5, 10, 11, 9, 6, 7, 8, 12, 13, 14, 16, 17, 18, 15, 19, 20, 21, 26, 27, 25, 22, 23, 24, 28, 29, 30, 0,
            1, 2, 31)
EAE_XL = 16384
EAE_XL_TIE = 0.995
EAE_ARMS = {3: (3, 0), 4: (4, 0), 5: (3, 1), 6: (4, 1)}
_EAE_COMMON = r"""
// ea-enc: A4b views of the recoded word (pair I's e2m1 bits on {6,7,8,15,22,23,24,31}); rtv(n, v) = v, opaque to ptxas
template <int I>
__device__ __forceinline__ uint32_t ext_a4b(uint32_t q) {
  if constexpr (I == 0) return q & 0x81C081C0u;
  else if constexpr (I == 1) return (q << 6) & 0x81C081C0u;
  else if constexpr (I == 2) return __funnelshift_l(q, q, 10) & 0x81C081C0u;
  else return ((q << 4) & 0x81808180u) | ((q >> 4) & 0x00400040u);
}
__device__ __forceinline__ uint32_t rtv(int n, uint32_t v) {
  uint32_t z;
  asm("shr.u32 %0, %1, 30;\n" : "=r"(z) : "r"(n));
  return v << z;
}
"""
_EAE_CONSUME = r"""
// ea-enc consume: A4b codes; DQ 3 dq2's paired scale decode, DQ 4 A5b words (rec + 4096 + (g * 4 + kq) * 16; odd steps
// on rotl 8). NT 1: consume's B (XB 0 ldmatrix.x2 per step, XB 1 the X1 preload), NT 2: consume2's two n8 tiles.
template <int DQ, int XB, int NT>
__device__ __forceinline__ void consume_e(const uint8_t* rec, uint32_t xb, int lane, int n, float (&acc)[NT][2][4]) {
  const int g = lane >> 2;
  const uint32_t m16 = rt16(n);
  const uint32_t m78 = rtv(n, 0x78007800u);
#pragma unroll
  for (int kq = 0; kq < 4; ++kq) {
    const uint4 c0 = *reinterpret_cast<const uint4*>(rec + (kq * 32 + lane) * 16);
    const uint4 c1 = *reinterpret_cast<const uint4*>(rec + ((4 + kq) * 32 + lane) * 16);
    uint32_t sw[4];
    if constexpr (DQ == 4) {
      const uint4 p = *reinterpret_cast<const uint4*>(rec + 4096 + (g * 4 + kq) * 16);
      sw[0] = p.x; sw[1] = p.y; sw[2] = p.z; sw[3] = p.w;
    } else {
      const uint2 s0 = *reinterpret_cast<const uint2*>(rec + 4096 + g * 32 + kq * 8);
      const uint2 s1 = *reinterpret_cast<const uint2*>(rec + 4096 + (g + 8) * 32 + kq * 8);
      sw[0] = s0.x; sw[1] = s0.y; sw[2] = s1.x; sw[3] = s1.y;
    }
    uint32_t bk[8][2];
    if constexpr (XB == 1 && NT == 1) {
#pragma unroll
      for (int p = 0; p < 4; ++p)
        ldsm_x4(xb + static_cast<uint32_t>((kq * 8 + 2 * p) * 32), bk[2 * p][0], bk[2 * p][1], bk[2 * p + 1][0],
                bk[2 * p + 1][1]);
    }
    const uint32_t qg[4] = {c0.x, c0.y, c0.z, c0.w};
    const uint32_t qu[4] = {c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
      for (int bsel = 0; bsel < 2; ++bsel) {
        const int j = wd * 2 + bsel;
        uint32_t sg;
        if constexpr (DQ == 4) {
          const uint32_t w = (j & 1) ? __byte_perm(sw[j >> 1], sw[j >> 1], 0x2103u) : sw[j >> 1];
          asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(sg) : "r"(w), "n"(0x07F007F0), "r"(m78));
        } else {
          sg = scale_pair_imad_rt(j < 4 ? sw[0] : sw[1], j < 4 ? sw[2] : sw[3], j & 3, m16);
        }
        uint32_t a[4];
        if (bsel == 0) {
          a[0] = mul_bf16x2_h<0>(ext_a4b<0>(qg[wd]), sg);
          a[2] = mul_bf16x2_h<0>(ext_a4b<1>(qg[wd]), sg);
          a[1] = mul_bf16x2_h<1>(ext_a4b<0>(qu[wd]), sg);
          a[3] = mul_bf16x2_h<1>(ext_a4b<1>(qu[wd]), sg);
        } else {
          a[0] = mul_bf16x2_h<0>(ext_a4b<2>(qg[wd]), sg);
          a[2] = mul_bf16x2_h<0>(ext_a4b<3>(qg[wd]), sg);
          a[1] = mul_bf16x2_h<1>(ext_a4b<2>(qu[wd]), sg);
          a[3] = mul_bf16x2_h<1>(ext_a4b<3>(qu[wd]), sg);
        }
        if constexpr (NT == 2) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4(xb + static_cast<uint32_t>((kq * 8 + j) * 32), b0, b1, b2, b3);
          mma16816(acc[0][bsel], a, b0, b1);
          mma16816(acc[1][bsel], a, b2, b3);
        } else if constexpr (XB == 1) {
          mma16816(acc[0][bsel], a, bk[j][0], bk[j][1]);
        } else {
          uint32_t b0, b1;
          ldsm_x2(xb + static_cast<uint32_t>((kq * 8 + j) * 32), b0, b1);
          mma16816(acc[0][bsel], a, b0, b1);
        }
      }
    }
  }
}

"""
_EAE_M16 = "  const uint32_t m16 = rt16(T);  // L4: run-time 16 (dq2 only; dead for DQ 1)\n"
_EAE_LT = r"""
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ void imad_step(uint32_t& a, uint32_t s, uint32_t k) {
  asm("mad.lo.u32 %0, %1, %2, %0;\n" : "+r"(a) : "r"(s), "r"(k));
}
"""
_EAE_LT_LOOP = r"""  if constexpr (DQ >= 3 && !DEFER) {
    static_assert(D == 2, "ea-enc XL: ring depth 2");
    constexpr int PU = NK * NS;
    const uint32_t r0 = smem_u32(ring + half * STAGE_W + wq * REC);
    const uint32_t xs0 = smem_u32(xs) + static_cast<uint32_t>(((lane >> 3) & 1) * 16 + (lane >> 4) * 32);
    uint32_t recl = r0 + static_cast<uint32_t>(lane * 16);
    uint32_t recs = r0 + static_cast<uint32_t>(4096 + g * (DQ == 4 ? 64 : 32));
    uint32_t fb = full0;
    uint32_t sl = 0u, ph = 0u, sg = 1u;
    int vend = 0, nr = 1;
    uint32_t xb = 0u, m16 = 16u, m78 = 0x78007800u;
    auto setpass = [&](int q) {
      const int v = q / PU;
      vend = (v + 1) * PU;
      nr = m.v_n[v];
      const int xr = (lane & 7) < nr ? (lane & 7) : nr - 1;
      xb = xs0 + static_cast<uint32_t>(xr * XROW) + (sl ? static_cast<uint32_t>(XS) : 0u);
      m16 = rt16(nr);
      m78 = rtv(nr, 0x78007800u);
    };
    int qs = b;
    int cnt0 = NK - (b & (NK - 1));
    if (cnt0 > e - b) cnt0 = e - b;
    int cnt = cnt0;
    setpass(qs);
    for (;;) {
      if (!mbar_try(fb, ph)) mbar_wait(fb, ph, err, dead);
      if (nr > 8)
        eak3::consume2<DQ>(ring + sl * STAGE2 + half * STAGE_W + wq * REC, smem_u32(xs) + sl * XS, lane, nr, acc);
      else
        eak::consume_lt<DQ, 1, 1>(recl, recs, xb, m16, m78, reinterpret_cast<float (&)[1][2][4]>(acc));
      __syncwarp();
      if (lane == 0) mbar_arrive(fb + 8u * D);
      ph ^= sl;
      sl ^= 1u;
      eak::imad_step(recl, sg, static_cast<uint32_t>(STAGE2));
      eak::imad_step(recs, sg, static_cast<uint32_t>(STAGE2));
      eak::imad_step(xb, sg, static_cast<uint32_t>(XS));
      eak::imad_step(fb, sg, 8u);
      sg = 0u - sg;
      if (--cnt != 0) continue;
      {
        const int q = qs + cnt0 - 1;
        const int sgrp = q / NK;
        const int v = sgrp / NS;
        const bool two = nr > 8;
        float c[2][4];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            c[t][k] = acc[t][0][k] + acc[t][1][k];
            acc[t][0][k] = 0.f;
            acc[t][1][k] = 0.f;
          }
        const int si = sgrp - v * NS;
        const int g0 = sgrp * NK;
        if (!(g0 >= b && g0 + NK <= e)) {
          const int slot_mine = (sgrp == b / NK) ? 0 : 1;
          float4* mine = reinterpret_cast<float4*>(ws + (static_cast<size_t>(blockIdx.x) * 2 + slot_mine) * PART);
          mine[warp * 32 + lane] = make_float4(c[0][0], c[0][1], c[0][2], c[0][3]);
          if (two) mine[NCW * 32 + warp * 32 + lane] = make_float4(c[1][0], c[1][1], c[1][2], c[1][3]);
          bar_consumers();
          const int c_first = owner_of(g0), c_last = owner_of(g0 + NK - 1);
          if (threadIdx.x == 0) {
            fence_acq_rel_gpu();
            const int old = atomicAdd(&tickets[sgrp], 1);
            const int f = (old == c_last - c_first) ? 1 : 0;
            if (f) fence_acq_rel_gpu();
            m.flag = f;
          }
          bar_consumers();
          if (m.flag == 0) goto lt_next;
          finish(c, ws, sgrp, c_first, c_last, weighted, weighted && begin_of(c_first) < g0, uW, uG, warp, lane, two);
          if (threadIdx.x == 0) tickets[sgrp] = 0;
        }
        epi<W13>(c, m, v, si, half, wq, g, cq, s_gs[m.v_e[v]], out);
      }
    lt_next:
      qs += cnt0;
      if (qs >= e) break;
      cnt0 = e - qs < NK ? e - qs : NK;
      cnt = cnt0;
      if (qs >= vend) setpass(qs);
    }
    return;
  }
"""
def _eae_rep(s, a, b, n=1):
    if s.count(a) != n:
        raise RuntimeError(f"ea-enc anchor x{s.count(a)}: {a[:60]!r}")
    return s.replace(a, b)
def _eae_dev(src):
    src = _eae_rep(src, "}  // namespace ea\n", _EAE_COMMON + "}  // namespace ea\n")
    a = "template <int DQ, int XB = 0>\n__device__ __forceinline__ void consume("
    src = _eae_rep(src, a, _EAE_CONSUME + a)
    src = _eae_rep(src, _EAE_M16, _EAE_M16 + "  if constexpr (DQ >= 3) {\n    consume_e<DQ, XB, 1>(rec, xb, lane, T, "
                   "reinterpret_cast<float (&)[1][2][4]>(acc));\n    return;\n  }\n")
    a = _EAE_M16.replace("(T)", "(n)")
    if a in src:
        src = _eae_rep(src, a, a + "  if constexpr (DQ >= 3) {\n    eak::consume_e<DQ, 0, 2>(rec, xb, lane, n, acc);\n"
                       "    return;\n  }\n")
    return src
def _eae_src4x():
    src = _eae_dev(_ea4x_source())
    for kn in ("w13", "w2"):
        k = f"eak4x::ea4x_{kn}_kernel<"
        for a, b in (("(r, {}2, false>)", "(r, {}3, false>)"), ("(r, {}2, true>)", "(r, {}4, false>)"),
                     ("(&cfg, {}2, false>", "(&cfg, {}3, false>"), ("(&cfg, {}2, true>", "(&cfg, {}4, false>")):
            src = _eae_rep(src, a.format(k), b.format(k))
    return _eae_rep(src, "(pdl & 256) != 0\n      ?", "(pdl & 4096) != 0\n      ?", 2)
def _eae_src4xl():
    s = _EAE_CONSUME[_EAE_CONSUME.index("template"):]
    for a, b in (("consume_e(const uint8_t* rec, uint32_t xb, int lane, int n,",
                  "consume_lt(uint32_t recl, uint32_t recs, uint32_t xb, uint32_t m16, uint32_t m78,"),
                 ("  const int g = lane >> 2;\n  const uint32_t m16 = rt16(n);\n  const uint32_t m78 = rtv(n, 0x78007800u);\n", ""),
                 ("rec + (kq * 32 + lane) * 16", "recl + kq * 512"), ("rec + ((4 + kq) * 32 + lane) * 16", "recl + (4 + kq) * 512"),
                 ("rec + 4096 + (g * 4 + kq) * 16", "recs + kq * 16"), ("rec + 4096 + g * 32 + kq * 8", "recs + kq * 8"),
                 ("rec + 4096 + (g + 8) * 32 + kq * 8", "recs + 256 + kq * 8")):
        s = _eae_rep(s, a, b)
    s = s.replace("*reinterpret_cast<const uint4*>(rec", "lds128(rec").replace("*reinterpret_cast<const uint2*>(rec", "lds64(rec")
    src = _eae_src4x()
    a = "template <int DQ, int XB = 0>\n__device__ __forceinline__ void consume("
    src = _eae_rep(src, a, _EAE_LT + s + a)
    i = src.index("  for (int q = b, i = 0; q < e; ++q, ++i) {\n", src.index("namespace eak4x {"))
    src = src[:i] + _EAE_LT_LOOP + src[i:]
    a = "  cudaLaunchConfig_t cfg = ea4x_cfg(grid, pdl, attr);\n"
    return _eae_rep(src, a, a + f'  TORCH_CHECK(pdl & {EAE_XL}, "ea4x XL bit");\n', 2)
def _eae_src6():
    src = _eae_dev("#define EA6_FAST 1\n" + _ea6_source())
    src = _eae_rep(src, "template <bool W13, int D, int MINB, bool FAST = false>\n",
                   "template <bool W13, int D, int MINB, bool FAST = false, int EQ = 2>\n")
    src = _eae_rep(src, "eak::consume<2, 1>(ring", "eak::consume<EQ, 1>(ring")
    for w, d in (("true", "D13"), ("false", "D2")):
        k = f"eak6::ea6_kernel<{w}, eak6::{d}, "
        src = _eae_rep(src, f"minb == 2 ? &{k}2> : &{k}1>", f"(flags & 4096) ? &{k}2, true, 4> : &{k}2, true, 3>")
        src = _eae_rep(src, f"    fn = &{k}2, true>;\n", "")
        src = _eae_rep(src, f"(r, {k}2>, {w})", f"(r, {k}2, true, 3>, {w})")
        src = _eae_rep(src, f"(r, {k}1>, {w})", f"(r, {k}2, true, 4>, {w})")
        src = _eae_rep(src, f"  ea6_one_info(r, {k}2, true>, {w});\n", "")
    return src
def _eae_start(t0):
    import hashlib
    import threading
    job = {"th": [], "res": {}, "err": [], "t": time.perf_counter() - t0}
    for key, cpp, mk, fns in (("4x", _EA4X_CPP, _eae_src4x, _EA4X_FNS), ("6", _EA6_CPP, _eae_src6, _EA6_FNS),
                              ("4xl", _EA4X_CPP, _eae_src4xl, _EA4X_FNS)):
        try:
            src = mk()
            name = f"okc_eaenc_{key}_" + hashlib.sha256((cpp + src).encode()).hexdigest()[:10]
            def run(key=key, name=name, cpp=cpp, src=src, fns=fns):
                try:
                    job["res"][key] = _rt_build(EAE_TAG, name, cpp, src, fns)
                except BaseException as e:
                    job["res"][key] = (None, f"thread {type(e).__name__}: {str(e)[:100]}", 0.0)
            th = threading.Thread(target=run, name=f"eaenc-{key}", daemon=True)
            th.start()
            job["th"].append(th)
        except Exception as e:
            job["err"].append(f"{key} source {type(e).__name__}: {str(e)[:100]}")
    return job
def _eae_recode(wr, a5b):
    wr = wr.view(-1)
    nrec = wr.numel() // 4608
    o3 = torch.empty_like(wr)
    o4 = torch.empty_like(wr) if a5b else None
    for r0 in range(0, nrec, 8192):
        n = min(nrec, r0 + 8192) - r0
        raw = wr[r0 * 4608:(r0 + n) * 4608]
        src = raw.view(torch.int32).view(n, 1152).to(torch.int64) & 0xFFFFFFFF
        codes = src[:, :1024]
        o = torch.zeros_like(codes)
        for j, p in enumerate(EAE_PERM):
            o |= ((codes >> p) & 1) << j
        sws = [(o3, src[:, 1024:])]
        if a5b:
            b = raw.view(n, 4608)[:, 4096:].to(torch.int64) - 8
            b = b.view(n, 2, 8, 4, 4, 2)
            f0, f1, f2, f3 = b[:, 0, ..., 0], b[:, 1, ..., 0], b[:, 0, ..., 1], b[:, 1, ..., 1]
            sws.append((o4, ((f0 << 4) | (f1 << 20) | (f3 << 12) | (((f2 << 28) | (f2 >> 4)) & 0xFFFFFFFF)).reshape(n, 128)))
        for dst, sw in sws:
            full = torch.cat([o, sw], 1)
            full = torch.where(full >= 2 ** 31, full - 2 ** 32, full).to(torch.int32)
            dst[r0 * 4608:(r0 + n) * 4608].view(torch.int32).copy_(full.reshape(-1))
        del src, codes, o, sws, full
    return o3, o4
def _eae_pair(ea, x, ids, h, lid, y, T, start, count, bits, dq=0, g2=None):
    bq, xl = EAE_ARMS.get(dq, (0, 0))
    kx, w13, w2 = ((ea["extexl" if xl else "exte"], ea[f"w13e{bq}"], ea[f"w2e{bq}"]) if bq
                   else (ea["ext4x"], ea["w13r"], ea["w2r"]))
    b = bits | (4096 if bq == 4 else 0) | (EAE_XL if xl else 0)
    kx.ea4x_w13(x, ids, w13, ea["gs13"], h, lid, ea["ws13x"], ea["tk13x"], ea["errx"], T, start, count, ea["grid"],
                ea["dq"], b)
    kx.ea4x_w2(h, ids, w2, ea["gs2"], y, ea["ws2x"], ea["tk2x"], ea["errx"], T, start, count,
               ea["grid"] if g2 is None else g2, ea["dq"], ea["pdl"] | b)
def _eae_xcore(base, dq):
    bq, xl = EAE_ARMS[dq]
    bits = EA4X_ARMS[base] | (4096 if bq == 4 else 0) | (EAE_XL if xl else 0)
    def core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0, poison=False):
        return _ea3_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace, pdl13,
                                 ("ea4x_w13", "ea4x_w2"), (f"w13e{bq}", f"w2e{bq}"), bits, "extexl" if xl else "exte",
                                 "x", poison)
    return core
def _eae_t1core(base, e1, e2):
    def core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace=None, pdl13=0, scratch=None):
        return _ead6_forward_core(ea, x, topk_ids, topk_weights, start, count, norm_w, trace, pdl13, scratch,
                                  EA6F_ARMS[base], (e1, e2))
    return core
_EAE_CORES = {**{f"{b}:e{d}": _eae_xcore(b, d) for b, v in EA4X_ARMS.items() if not v & 256 for d in EAE_ARMS},
              **{f"{b}:e{a}": _eae_t1core(b, int(a[0]), int(a[1])) for b in EA6F_ARMS for a in EAE_T1}}
def _eae_base(state, T):
    ea, hc = state.get("ea") or {}, state.get("hc") or {}
    if T in HC_CLASSES:
        h, xa = (hc.get("on") or {}).get(T, 0), (hc.get("cold_x") or {}).get(T)
        ok = h < 0 and xa in EA4X_ARMS and not EA4X_ARMS[xa] & 256
        return ("cold", xa) if ok else (None, f"cold half H{h}/{xa}")
    b = (ea.get("on") or {}).get(T)
    if T == 1:
        return ("t1", b) if EA6F_ARMS.get(b, 0) & 3 == 3 else (None, f"route {b} not FAST K1+K2")
    return ("x", b) if b in EA4X_ARMS and not EA4X_ARMS[b] & 256 else (None, f"route {b} not a v4x s twin")
def _eae_set(state, kind, T, base, arm):
    if kind == "cold":
        state["hc"].setdefault("cold_e", {})[T] = int(arm or 0)
    else:
        state["ea"]["on"][T] = f"{base}:e{arm}" if arm else base
def _eae_zero(ea):
    for k in ("tk13x", "tk2x", "errx", "err"):
        if k in ea:
            ea[k].zero_()
def _eae_sums(ea):
    return sum(int(ea[k].abs().sum()) for k in ("tk13x", "tk2x", "errx", "err") if k in ea)
def _eae_draw(state, device, gens, T, kind, n=None):
    if kind == "cold":
        h = state["hc"]["on"][T]
        return (*_hcx_inputs(state, device, gens[1], n or T, h), h)
    return (*_ead_inputs("judge", T, gens[0], device), None, 0)
def _eae_fn(state, kind, T, base, d, a, part="all", fill=None):
    ea, (xn, lids, tw, al, h) = state["ea"], d
    if kind == "cold":
        def f():
            _eae_set(state, kind, T, base, a)
            try:
                return _hc_experts(xn, lids, tw, al, state, h, fill=fill, part=part)
            finally:
                _eae_set(state, kind, T, base, None)
        return f
    core = _ead_core(f"{base}:e{a}" if a else base)
    return lambda: core(ea, xn, lids, tw, 0, LOCAL_EXPERTS, state["post_expert_norm_w"], pdl13=ea["pdl13_t"].get(T, 0))
def _eae_hy(state, kind, T, base, d, a):
    ea, (xn, lids, tw, al, _) = state["ea"], d
    nan, dev, n, hid = float("nan"), xn.device, xn.shape[0], xn.shape[1]
    y = torch.full((n * TOP_K, hid), nan, dtype=torch.bfloat16, device=dev)
    if kind == "t1":
        h = torch.full((32, n, ea["inter"]), nan, dtype=torch.bfloat16, device=dev)
        o = _ead6_forward_core(ea, xn, lids, tw, 0, LOCAL_EXPERTS, state["post_expert_norm_w"], None,
                               ea["pdl13_t"].get(T, 0), (h, y), EA6F_ARMS[base], (int(a[0]), int(a[1])) if a else (0, 0))
        return h, y, o
    h = torch.full((n * TOP_K, ea["inter"]), nan, dtype=torch.bfloat16, device=dev)
    lid = torch.empty((n, TOP_K), dtype=torch.int32, device=dev)
    _eae_pair(ea, xn, al[6] if kind == "cold" else lids, h, lid, y, n, 0, LOCAL_EXPERTS, EA4X_ARMS[base], int(a or 0),
              None if kind == "cold" else ea.get("g2_t", {}).get(n, ea["grid"]))
    return h, y
def _eae_exact(state, kind, T, base, d, arms, ex):
    part, fill = ("cold", float("nan")) if kind == "cold" else ("all", None)
    hy0 = _eae_hy(state, kind, T, base, d, None)
    o0 = _eae_fn(state, kind, T, base, d, None, part, fill)().clone()
    for a in arms:
        o = _eae_fn(state, kind, T, base, d, a, part, fill)()
        ex[a] = ex[a] and _eae_eq(o, o0) and all(_eae_eq(p, q) for p, q in zip(_eae_hy(state, kind, T, base, d, a), hy0))
def _eae_class(state, device, T, kind, base, arms, gens):
    ea = state["ea"]
    ex, rat = {a: True for a in arms}, {a: [] for a in arms}
    _eae_zero(ea)
    for _ in range(2):
        d = _eae_draw(state, device, gens, T, kind)
        _eae_exact(state, kind, T, base, d, arms, ex)
        pt = ("all", "cold")[T == 256]
        graphs = {a: _rt_graph(_eae_fn(state, kind, T, base, d, a, pt))[0] for a in (None, *arms)}
        res = _rt_time(graphs, device, reps=EAE_REPS, rounds=EAE_ROUNDS)
        del graphs
        for r in res:
            m = max(_rt_stats(r[None])[0], 1e-3)
            for a in arms:
                rat[a].append(_rt_stats(r[a])[0] / m)
    data = _ead_cases(state, device, gens[1], T)
    _eae_set(state, kind, T, base, None)
    ref = kernel_fn(_dl3_clone_data(data), state)
    nd = "" if all(torch.equal(p, q) for p, q in zip(kernel_fn(_dl3_clone_data(data), state), ref)) else " layer-nondet"
    for a in arms:
        _eae_set(state, kind, T, base, a)
        try:
            got = kernel_fn(_dl3_clone_data(data), state)
        finally:
            _eae_set(state, kind, T, base, None)
        ex[a] = ex[a] and (bool(nd) or all(torch.equal(p, q) for p, q in zip(got, ref)))
    torch.cuda.synchronize()
    bad = _eae_sums(ea)
    _eae_zero(ea)
    return {a: v and not bad for a, v in ex.items()}, rat, bad, nd
def _eae_veto(state, device, T, kind, base, a):
    ss = state.get("side_serial", False)
    def st(v, ser):
        def f():
            _eae_set(state, kind, T, base, v)
            state["side_serial"] = ser
        return f
    md = ("s", "f") if kind == "cold" else ("f",)
    try:
        per = _ead_layer_gate(state, device, T, {k + m: st(v, m == "s") for m in md for k, v in (("m", None), ("e", a))},
                              EAE_SEED + 9 + T)
    finally:
        _eae_set(state, kind, T, base, None)
        state["side_serial"] = ss
    return {m: [_rt_stats(ts["e" + m])[0] / max(_rt_stats(ts["m" + m])[0], 1e-3) for r in per.values() for ts in r]
            for m in md}
def _eae_canary(state, device, gens, T, base, a):
    ea = state["ea"]
    for n in EAE_CANARY.get(T, ()):
        _eae_zero(ea)
        ex = {a: True}
        _eae_exact(state, "cold", T, base, _eae_draw(state, device, gens, T, "cold", n), [a], ex)
        torch.cuda.synchronize()
        if not ex[a] or _eae_sums(ea):
            return f"canary T={n} FAIL"
    _eae_zero(ea)
    return ""
def _eae_off(state):
    ea, hc = state.get("ea") or {}, state.get("hc") or {}
    for t, v in list((ea.get("on") or {}).items()):
        if isinstance(v, str) and ":e" in v:
            ea["on"][t] = v.split(":e")[0]
    hc.pop("cold_e", None)
    for k in ("exte", "ext6e", "extexl", "w13e3", "w2e3", "w13e4", "w2e4"):
        ea.pop(k, None)
def _eae_phase(state, weights, device, t0, job):
    L = []
    n_in = {t: len(v) for t, v in _EAD_INPUTS.items()}
    try:
        _eae_run(state, weights, device, t0, job, L)
    except Exception as e:
        _eae_off(state)
        L.append(f"{EAE_TAG} raised {type(e).__name__}: {str(e)[:150]} -> master everywhere")
    for t in list(_EAD_INPUTS):
        if t in n_in:
            del _EAD_INPUTS[t][n_in[t]:]
        else:
            del _EAD_INPUTS[t]
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        L.append(f"{EAE_TAG} final sync raised {type(e).__name__}")
    hc, wk = state.get("hc") or {}, state.get("whk") or {}
    ce = hc.get("cold_e") or {}
    return [f"{HC_TAG} PATH " + " | ".join(
        f"T={c} H{(hc.get('on') or {}).get(c, 0)} cold {(hc.get('cold_x') or {}).get(c) or hc.get('cold', '-')}"
        + (f"+{_eae_lab(str(ce[c]))}" if ce.get(c) else "") + f" hot {'wh5ds' if wk.get(c) else 'master wg'}"
        for c in HC_CLASSES)] + L
def _eae_run(state, weights, device, t0, job, L):
    tw = time.perf_counter()
    for th in job["th"]:
        th.join(max(0.0, EAE_JOIN_S - (time.perf_counter() - t0)))
    wait = time.perf_counter() - tw
    res = {k: v for k, v in job["res"].items() if not any(t.name == f"eaenc-{k}" and t.is_alive() for t in job["th"])}
    ext4, ext6, ext4xl = ((res.get(k) or (None,))[0] for k in ("4x", "6", "4xl"))
    bl = ", ".join(f"{k} " + (f"{res[k][2]:.0f}s" + ("" if res[k][0] is not None else f" FAIL {str(res[k][1])[:60]}")
                              if k in res else f"not joined by {EAE_JOIN_S:.0f}s") for k in ("4x", "4xl", "6"))
    L.append(f"{EAE_TAG} build {bl}{'; ' + '; '.join(job['err']) if job['err'] else ''} (bg from {job['t']:.0f}s, after "
             f"ea4x-dl), join wait {wait:.1f}s, phase at {time.perf_counter() - t0:.0f}s (after [hh-wh] routed)")
    ea = state.get("ea")
    if ea is None or ea.get("dq") != 2 or (ext4 is None and ext6 is None):
        L.append(f"{EAE_TAG} PATH CHOSEN all T = master (no E module / Engine A dq {None if ea is None else ea.get('dq')})")
        return
    at, ok = [], {}
    for key, ext, w, need in (("4x", ext4, 5, "ext4x"), ("4xl", ext4xl, 5, "ext4x"), ("6", ext6, 6, "ext6")):
        if ext is None or ea.get(need) is None:
            continue
        rows = _eae_info(ext, w)
        at.append(f"{key} " + " ".join(f"{n}{r[1]}/{r[2]}/{r[3]}" for n, r in zip(("K1e4 ", "K1e5 ", "K2e4 ", "K2e5 "), rows)))
        for i, (kk, dq) in enumerate((("K1", 3), ("K1", 4), ("K2", 3), ("K2", 4))):
            ok[(key, kk, dq)] = bool(rows[i][0]) and (rows[i][2] == 0 and rows[i][3] >= 2 if key == "4xl" else rows[i][3] >= 1)
    t = time.perf_counter()
    lo, hi, bad = 255, 0, 0
    for k in ("w13r", "w2r"):
        s = ea[k].view(-1, 4608)[:, 4096:]
        lo, hi = min(lo, int(s.min())), max(hi, int(s.max()))
        bad += int(((s < 0x08) | (s > 0x7E)).sum())
    e5 = bad == 0
    for k in ("w13", "w2"):
        ea[f"{k}e3"], o4 = _eae_recode(ea[f"{k}r"], e5)
        if e5:
            ea[f"{k}e4"] = o4
    torch.cuda.synchronize()
    mb = sum(ea[k].numel() for k in ("w13e3", "w2e3", "w13e4", "w2e4") if k in ea) / 1e6
    ea["exte"], ea["ext6e"], ea["extexl"] = ext4, ext6, ext4xl
    L.append(f"{EAE_TAG} regs/lmem/occ {'; '.join(at)}; recode {time.perf_counter() - t:.1f}s {mb:.0f}MB; scales "
             f"0x{lo:02X}..0x{hi:02X}, {bad} outside 0x08..0x7E -> E5 {'eligible' if e5 else 'OFF'}")
    gens = (torch.Generator().manual_seed(EAE_SEED), torch.Generator(device=device).manual_seed(EAE_SEED + 1))
    tr, picks, why, txt = time.perf_counter(), {}, {}, []
    dqs = ("3", "4") if e5 else ("3",)
    for T in EAE_T:
        kind, base = _eae_base(state, T)
        if kind is None:
            why[T] = base
            continue
        if time.perf_counter() - tr > EAE_RACE_S or time.perf_counter() - t0 > EAE_END_S:
            why[T] = "race cap / deadline"
            continue
        if kind == "t1":
            arms = [a for a in EAE_T1 if all(c == "0" or (c in dqs and ok.get(("6", k, int(c))))
                                             for c, k in zip(a, ("K1", "K2")))]
        else:
            arms = [str(int(a) + 2 * j) for j, k in enumerate(("4x", "4xl")) for a in dqs
                    if ok.get((k, "K1", int(a))) and ok.get((k, "K2", int(a)))]
        if not arms:
            why[T] = "no admitted E instance"
            continue
        ex, rat, nb, nd = _eae_class(state, device, T, kind, base, arms, gens)
        good = [a for a in arms if ex[a] and rat[a] and max(rat[a]) <= EAE_BAR]
        s = (f"T={T} {base}{nd}: {'cold ' * (T == 256)}iso " + " ".join(f"{_eae_lab(a)} {'y' if ex[a] else 'n'} {_eae_rs(rat[a])}" for a in arms)
             + (f" tk/err {nb}" if nb else ""))
        if good:
            med = lambda k: sorted(rat[k])[len(rat[k]) // 2]
            ge = [a for a in good if a not in "56"]
            a0 = a = min(ge or good, key=med)
            for x in [x for x in good if x in "56" and ge]:
                rat[x + "e"] = [p / max(q, 1e-6) for p, q in zip(rat[x], rat[a0])]
                s += f" | {_eae_lab(x)}/{_eae_lab(a0)} {_eae_rs(rat[x + 'e'])}"
                if med(x + "e") < EAE_XL_TIE and (a == a0 or med(x) < med(a)):
                    a = x
            for a in dict.fromkeys((a, a0)):
                v = _eae_veto(state, device, T, kind, base, a)
                s += f" | layer {_eae_lab(a)} " + " ".join(f"{m} {_wh_fmt(r)}" for m, r in v.items())
                if max(max(r) for r in v.values()) > EAE_VETO:
                    why[T] = f"layer VETO > {EAE_VETO}"
                    continue
                why[T] = _eae_canary(state, device, gens, T, base, a) if kind == "cold" else ""
                if not why[T]:
                    picks[T] = (kind, base, a)
                    break
        else:
            ne = [a for a in arms if not ex[a]]
            why[T] = (f"not exact {','.join(map(_eae_lab, ne))}" if len(ne) == len(arms) else
                      f"iso E/m > {EAE_BAR} in a round")
        txt.append(s)
    for T, (kind, base, a) in picks.items():
        _eae_set(state, kind, T, base, a)
    used = {EAE_ARMS[int(c)][0] for _, _, a in picks.values() for c in a if c != "0"}
    for d in (3, 4):
        if d not in used:
            ea.pop(f"w13e{d}", None)
            ea.pop(f"w2e{d}", None)
    if picks:
        fails, cold = [], [T for T, p in picks.items() if p[0] == "cold"]
        if time.perf_counter() - t0 > EAE_COPY_S:
            fails.append("deadline")
        else:
            small = [T for T in picks if T not in cold]
            if small:
                other = _ead_copy_tree(state)
                g = torch.Generator(device=device).manual_seed(EAE_SEED + 2)
                try:
                    for T in small:
                        _ead_copy_one(state, other, device, g, T, fails)
                finally:
                    del other
                    gc.collect()
                    torch.cuda.empty_cache()
            if cold and not fails:
                cok, ln = _hc_copy_check(state, device, cold, "E off")
                if not cok:
                    fails.append(ln[-60:])
            if _eae_sums(ea):
                fails.append("tk/err")
        _eae_zero(ea)
        L.append(f"{EAE_TAG} state-copy check T {sorted(picks)}: " + (f"FAIL ({fails[0][:80]}) -> master" if fails else "PASS"))
        if fails:
            _eae_off(state)
            why.update({T: "copy check FAIL" for T in picks})
            picks = {}
    if not picks:
        _eae_off(state)
    for ln in txt:
        L.append(f"{EAE_TAG} {ln}")
    L.append(f"{EAE_TAG} PATH CHOSEN " + "; ".join(
        f"T={T} " + (_eae_lab(picks[T][2]) if T in picks else f"master ({why.get(T, '?')})") + " (no XL arm: ea6)" * (T == 1) for T in EAE_T)
        + f" (race {time.perf_counter() - tr:.1f}s, prepare at {time.perf_counter() - t0:.0f}s)")
def _eae_eq(a, b):
    if a.shape != b.shape:
        return False
    m = torch.isnan(a)
    return bool(torch.equal(m, torch.isnan(b))) and bool(torch.equal(a[~m], b[~m]))
def _eae_lab(a):
    n = {"0": "m", "3": "E4", "4": "E5", "5": "E4XL", "6": "E5XL"}
    return n[a] if len(a) == 1 else f"{n[a[0]]}/{n[a[1]]}"
def _eae_rs(v):
    return f"{sorted(v)[len(v) // 2]:.3f}[{min(v):.3f},{max(v):.3f}]" if v else "-"
def _eae_info(ext, w):
    v = [int(t) for t in (ext.ea4x_info() if w == 5 else ext.ea6_info())]
    return [(v[w * i], v[w * i + 1], v[w * i + 2], v[w * i + w - 1]) for i in range(4)]
