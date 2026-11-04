# okc: problem=muse_bf16_proj
# Adopted from alygg77@6cae49a kernels/muse_bf16_proj.py (Apache-2.0; MIT/third-party notices kept below and in third_party/okc-muse-spark-kernels/NOTICE), unmodified except this line. yesthatwilly 2026-10-31 + BF16FIX: sk2 / skinny ring-slot WAR fix: fence.proxy.async.shared::cta before every hand-back of a TMA / bulk-filled ring slot read with generic loads (sk2 consumers, escape owner, skinny; ordering only); 3to1 sk2fix BF16FIX, judge 10-31. + SWEEP-bf16 P1F: unpooler m128 static cuBLASLt plan nn(18,4,2) (128x64 coopB 1x1) on the existing W^T copy. + BFX (trx-study 11-03): router m 2..16 routertail_small_kernel (4-owner cluster reduction, ruwwww); router m 32 rr_kernel<32,16,4,1,2,1> route (ruwwww route choice, existing Nathen ROUTER_CUDA). + PLAIN: every embedded compressed CUDA payload is now plain readable source text (string literals); compiled code byte-identical. pp-nonce PLAIN-r1
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions (routertail_small_kernel, verbatim; the router m=32 rr_kernel route choice) adapted from
# github.com/ruwwww/open-kernel-challenge @c035bf9 (muse_bf16_proj), modified; OKC submissions are licensed under
# the Apache License 2.0 (llms.txt), and that attribution is retained here.
# ---- Third-party notice: Nathen Fernandes routes are derived from
# https://github.com/NathenFernandes/muse-bf16-proj, used under the MIT License below:
# MIT License
#
# Copyright (c) 2026 Nathen Fernandes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---- end of third-party notice
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

Math
----
``out[m, n] = x[m, k] @ W[n, k]^T`` for the five projection matrices of one
model (``qkv_gate``, ``o``, ``pooler``, ``unpooler``, ``router``). The router's
FP32 weights run as a BF16 hi + lo pair concatenated along K with FP32
accumulation, as SGLang does above 8 tokens.

Lossless 12-bit weights (BF16 projections)
-------------------------------------------
Each BF16 weight is stored as its low byte (exponent bit 0 + 7 mantissa bits,
raw) and a 4-bit code for its high byte (sign + exponent bits 7..1, as an
offset into an 8-value window chosen per matrix, the window covering the most
weights). The kernel rebuilds the exact BF16 bits in registers (a few integer
ops per weight). The rare weights whose exponent falls outside the window
(tiny magnitudes) are exact-corrected: the kernel adds ``x[:, k] * (w -
w_decoded)`` for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.
``prepare()`` rejects a packing whose escape share exceeds
``1 / MAX_ESCAPE_DENOMINATOR`` or whose correction is not exact in FP32, and
falls back to uncompressed fragments.

Why the kernels look the way they do
------------------------------------
Weights dominate: for m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 fragment order (prepared once), and
multiplied against x (staged in shared memory) on BF16 tensor cores with FP32
accumulation. Split-K across warps (shared-memory reduction) and across CTAs
(FP32 partials + a self-resetting semaphore; the last CTA of a row block
reduces) keeps ~128 CTAs streaming on every shape.

Routes (chosen per layer and token count m by ``_static_choice`` / ``_autotune``)
---------------------------------------------------------------------------------
* ``sk2`` -- m <= 32, swapped operands, no cross-CTA reduction (the production
  skinny path on sm_90). ``skinny`` (TMA-streamed tensor-core GEMV) is the
  fallback / pre-sk2 path, kept for non-sm_90 and as the autotuner's
  alternative.
* ``cgemv`` / ``sgemv`` / ``rgemv`` -- m == 1 scalar GEMV specialists.
* ``wide`` / ``gemm`` / ``gq`` / ``gqx`` -- m 33..256 TMA + wgmma GEMMs
  (``wide`` is the BF16 route; the router uses ``wide``/``gemm``/``rk``).
* ``rk`` -- router-only split-K GEMM (order-dependent FP32 adds, inside the
  router tolerance).
* ``cutlass`` -- CUTLASS pingpong GEMM, pooler m == 256 only, built in the
  background during ``prepare()``; a failed build just drops the route.
* ``torch`` / ``torchT`` -- cuBLAS fallbacks (``F.linear`` / ``mm`` on a
  K-major copy). Any custom path that fails ``_selftest`` is disabled in
  favour of cuBLAS.

Design of this file
--------------------
* One Python file; native CUDA only, compiled once by
  ``torch.utils.cpp_extension.load_inline`` (``_get_ext`` / ``_build_cutlass``).
* The CUDA sources are string constants, kept verbatim: the kernels, tile
  sizes, and launch configs in them are the tuned behaviour and are not
  touched by this refactor. Python only assembles dispatch tables into them
  (``@@PLACEHOLDER@@``) and prepares their inputs.
* ``prepare(weights)`` builds every weight layout once (compressed streams,
  escape tables, workspaces) into ``state``; ``kernel_fn(data, state)`` runs
  one projection. Fresh outputs are allocated on every call; nothing is
  cached across calls except the compiled extensions and the per-shape
  autotune choice in ``state``.
* CUDA-graph capturable: the timed paths perform no host synchronisation
  (``_selftest`` / ``_autotune`` synchronise only during ``prepare()``).
* Config tables live next to the kernel they configure. Constants that more
  than one helper needs live in the "Shared constants" section below.
"""

import importlib.util
import os
import sys
import threading
from typing import TypedDict

import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import load_inline

# Debug only, from the environment: one random escape warp per CTA stalls this
# many ns at a random stage; ring waits trap after 2 s.
_SK2_DELAY = int(os.environ.get("SK2_DELAY_ESC", "0"))


# ---------------------------------------------------------------------------
# Shared constants (tuning numbers used by more than one helper)
# ---------------------------------------------------------------------------
# Lossless 12-bit packing: an exponent window of EXPONENT_WINDOW consecutive
# exponent values, picked per matrix to cover the most weights.
EXPONENT_WINDOW = 8
EXPONENT_VALUES = 128  # distinct values of BF16 exponent bits 7..1
# Packing is rejected when more than this share of weights escapes the window.
MAX_ESCAPE_DENOMINATOR = 200
# base4 packs one window base byte into each byte of a uint32 (see decode8).
BASE4_REPLICATE = 0x01010101
# Stage-count clamps for the TMA skinny kernel's pipeline.
TMA_MIN_STAGES = 2
TMA_MAX_STAGES = 8
# sm_90 / sm_80 shared-memory budgets (bytes) for that pipeline.
SM90_SMEM_BUDGET_BYTES = 112 * 1024
SM80_SMEM_BUDGET_BYTES = 40 * 1024
# Stage-count clamps for the sk2 kernel's ring.
SK2_MIN_STAGES = 2
SK2_MAX_STAGES = 16
# Failure seeds keep selftest / autotune reproducible across runs.
SELFTEST_SEED = 1234
AUTOTUNE_SEED = 4321
AUTOTUNE_REPS = 24
# Autotune replays from a cold L2: this read sweep is larger than L2.
L2_FLUSH_ROWS = 65_536  # 65_536 x 1024 float32 = 256 MiB
# sk2 ring sizing: a config name ending in _rNNN caps the ring at NNN KiB,
# otherwise the default applies; the ring also never exceeds the smem budget
# left after the rest of the kernel's shared-memory layout.
SK2_DEFAULT_RING_CAP_KB = 200
SK2_RING_SMEM_BUDGET_BYTES = 232_448



class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# ---------------------------------------------------------------------------
# CUDA sources. These string constants are the kernels, verbatim: tile sizes,
# launch configs, and numerics in them are the tuned behaviour. Each kernel's
# config table and dispatch builder live next to its source below; @@...@@
# placeholders are filled by the builder for that route.
# ---------------------------------------------------------------------------
CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
@@BODY@@
"""

MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}

"""

WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


def wgmma_rs_defs(bns):
    """Generate wgmma_bn (register-sourced A) overloads for m64nBNk16 bf16."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // [WARFIX] reads done before the refill
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      while (*((volatile int*)(counters + rb)) - target < 0) __nanosleep(20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    while (true) {
      int v = *((volatile int*)(counters + rb));
      if (v - target >= 0) break;
      __nanosleep(32);
    }
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


SK2_SRC = r"""

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#define SK2_DIAG 0
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // [WARFIX] reads done before the refill
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const uint32_t s0 = sbase + slot * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // [WARFIX] reads done before the refill
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // [WARFIX] reads done before the refill
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    return;
  }
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapA, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    if constexpr (HLW) {   // router: BF16 hi and lo tiles [64 rows x 64 k] by TMA (128B swizzle) from [hi | lo]
      tma_load_2d_ef(st + B_BYTES, &mapA, full0 + 8 * slot, (u0 + i) * 64, rb * 64);
      tma_load_2d_ef(st + B_BYTES + 8192, &mapA, full0 + 8 * slot, K + (u0 + i) * 64, rb * 64);
    } else {
      bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
    }
  };
  const bool producer = tid == 0;
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    if constexpr (HLW) asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
      // router: SS-wgmma on the TMA'd hi and lo tiles (no register-sourced operands: the wgmma pipeline stays full)
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + j * 32), make_desc_sw128(st + j * 32));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + 8192 + j * 32), make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }  // !HLW
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  named_bar(1, NT);
  if (tid == 0) {
    // release: every thread's partial (ordered before the CTA barrier) is visible before the count; acquire on the
    // spin makes the other CTAs' partials visible to this CTA after the barrier below
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(counters + zh * nrb + rb) : "memory");
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      int v;
      do {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(counters + zh * nrb + rb) : "memory");
      } while (v - target < 0);
    } else if (s_last) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB, mA;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  mA = mB;
  if (HLW && major >= 9) mA = make_map_2d(Wst.data_ptr(), (uint64_t)Wst.size(0), (uint64_t)Wst.size(1), 64);   // [hi | lo]
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, mA, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void rgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M);
void sgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t N);
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);
void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk(torch::Tensor Wc, torch::Tensor x, torch::Tensor out, int64_t S, torch::Tensor zc);
"""

# Weight rows per layer (the full matrices; some routes run on a prefix, e.g.
# the unpooler slice at m <= 8).
WEIGHT_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
COMPRESSED_LAYERS = {"qkv_gate", "o", "pooler", "unpooler"}
# At small m the unpooler runs on its first rows only (the slice SGLang scores).
UNPOOLER_SLICE_ROWS = 1024

_ext = None


# ---------------------------------------------------------------------------
# Skinny (TMA) route config
# ---------------------------------------------------------------------------
TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def _tma_nstage(nw, comp, sm90, ks, hl=False):
    """Pipeline stages for a skinny config: as many as fit the smem budget."""
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = SM90_SMEM_BUDGET_BYTES if sm90 else SM80_SMEM_BUDGET_BYTES
    return max(TMA_MIN_STAGES, min(TMA_MAX_STAGES, budget // (nw * chunk)))


# alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
# against the default config and cuBLAS on every scored shape
TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def _tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(TMA_ALT.items())]
    return items


def _tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in _tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], _mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = _tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and mtk <= 1) else WEIGHT_ROWS[layer]
        wgm = bool((sm90 or WGM_SM80_TEST) and comp and USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    return TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(lines)).replace("@@TMA_NAME@@", "skinny_tma_a")


GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256




def _gemm_src(sm90):
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in GEMM_CONFIGS})
    return (GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 8),
}
USE_GQ = True

GQX_SRC = r"""
// ============================================================================ gqx (m 33..256): BF16 weights, big tiles
// CTA = BM = 64 NWG weight rows x BN tokens over 1/CL of K; a cluster of CL CTAs covers all of K for one row block.
// Ring of TMA'd W [BM x 64] and x [BN x 64] tiles (128B swizzle) fed by a producer warp; warpgroup w runs SS-wgmma
// m64nBN on rows [64w, 64w + 64). Reduction: after every rank's main loop (cluster barrier: every ring is free), each
// rank owns BN/CL tokens; every thread st.async's the other ranks' token blocks of its accumulator fragment into the
// owner's ring space ([src rank][thread][4 JR floats]) and sums the CL-1 incoming copies of its own block.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gqx_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NTH = NWG * 128;
  constexpr int NB = BN / 8;                 // 8-token blocks of the tile; rank r owns blocks [r NB / CL, (r+1) NB / CL)
  constexpr int JM = (NB + CL - 1) / CL;     // most blocks a rank owns
  static_assert(CL <= NB, "gqx: every rank owns at least one token block");
  static_assert(((NB + CL - 1) / CL) * 8 * BM * 2 <= NSTAGE * ST, "gqx: the BF16 staging tile must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, rbar = bars + 16 * NSTAGE;
  // CL > 1: this rank's accumulator fragments [NB][NTH] float4, outside the ring (written right after the main loop)
  float4* frag = reinterpret_cast<float4*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  float* recv = reinterpret_cast<float*>(gbase);   // [CL][NTH][4 JM] (the ring, once every rank is done with it)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int U = K >> 6;
  const int u0 = ((int)crank * U) / CL, u1 = (((int)crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(rbar, 1);

    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();

  if (warp == NWG * 4) {   // producer: refills; then joins the cluster barrier
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {   // the consumers' two cluster barriers (fragments written, reads done)
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  if constexpr (CL > 1) {
    // 1) fragments out right away (own area: no need to wait for the other ranks' main loops), 2) one cluster
    // barrier, 3) the owner of each token block sums the other ranks' copies over DSMEM, stages BF16 [token][64 rows]
    // in its own (free) ring and 4) writes each token's 64 rows as 128 contiguous bytes; 5) exit barrier
#pragma unroll
    for (int j = 0; j < NB; ++j) frag[j * NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t fl = (uint32_t)__cvta_generic_to_shared(frag);
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);   // [owned tokens][BM rows]
    const int jlo = ((int)crank * NB) / CL, jhi = (((int)crank + 1) * NB) / CL;
    const int rl = wg * 64 + (warp & 3) * 16 + g;   // this thread's first row within the tile
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      if (r != (int)crank) continue;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        if (j < (r * NB) / CL || j >= ((r + 1) * NB) / CL) continue;
        float4 pv[CL];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(fl + (uint32_t)((j * NTH + tid) * 16)), "r"(r2));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                       : "=f"(pv[r2].x), "=f"(pv[r2].y), "=f"(pv[r2].z), "=f"(pv[r2].w) : "r"(ra) : "memory");
        }
        float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          v0 += pv[r2].x; v1 += pv[r2].y; v2 += pv[r2].z; v3 += pv[r2].w;
        }
        const int tl = 8 * (j - (r * NB) / CL) + 2 * t4;   // token within the owned slice
        stg[tl * BM + rl] = __float2bfloat16_rn(v0);
        stg[tl * BM + rl + 8] = __float2bfloat16_rn(v2);
        stg[(tl + 1) * BM + rl] = __float2bfloat16_rn(v1);
        stg[(tl + 1) * BM + rl + 8] = __float2bfloat16_rn(v3);
      }
    }
    named_bar(1, NTH);
    {
      constexpr int CH = BM / 8;                 // 16-byte chunks per token row
      const int nt = 8 * (jhi - jlo);
      for (int idx = tid; idx < nt * CH; idx += NTH) {
        const int tt = idx / CH, c = idx - tt * CH;
        const int tok = 8 * jlo + tt;
        if (tok < M)
          *reinterpret_cast<uint4*>(out + (size_t)tok * N + rb * BM + c * 8) =
              *reinterpret_cast<const uint4*>(stg + tt * BM + c * 8);
      }
    }
    // nobody leaves while another rank may still read its fragments
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gqx(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 + (CL > 1 ? (size_t)(BN / 8) * NWG * 128 * 16 : 0);
  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(N / BM * CL);
  lc.blockDim = dim3(NWG * 128 + 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CL;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  lc.attrs = at_;
  lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
}

void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}
"""

GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}
USE_GQX = True
TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


def _gqx_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gqx_bn(m):
    return 128 if m <= 128 else 256


def _run_gq_family(x, layer, m, n, k, state, configs, bn_of_m, method):
    """Run a gq/gqx cluster GEMM: same call shape, different config table and
    entry point (gq = plain BF16 weights, gqx = big tiles)."""
    bn = bn_of_m(m)
    nwg, cl, ns = configs[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(_get_ext(), method)(state["raw"][layer], x, out, m, n, k, bn, nwg, cl, ns)
    return out


def _run_gqx(x, layer, m, n, k, state):
    return _run_gq_family(x, layer, m, n, k, state, GQX_CONFIGS, _gqx_bn, "gqx")



RGEMV_SRC = r"""
// ============================================================================ router GEMV (m = 1): FP32 weights
// read directly (8.4 MB, no hi/lo split), ROWS rows per CTA, 256 threads; thread i owns k = 8i + 2048j (j < 4): every
// weight and x load is issued before the first FMA; warp butterfly then a fixed-order cross-warp sum (deterministic).
template <int ROWS, int MMAX>
__global__ void __launch_bounds__(256) rgemv_kernel(const float* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int M, int N) {
  constexpr int K = 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  float4 w[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4* p = reinterpret_cast<const float4*>(W + (size_t)(r0 + r) * K + j * 2048 + tid * 8);
      w[r][2 * j] = __ldg(p); w[r][2 * j + 1] = __ldg(p + 1);
    }
  float acc[ROWS][MMAX];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int t = 0; t < MMAX; ++t) {
    if (t >= M) break;
    uint4 xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + j * 2048 + tid * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float4 a = w[r][2 * j], b = w[r][2 * j + 1];
        float v = acc[r][t];
        v = fmaf(a.x, xf[0], v); v = fmaf(a.y, xf[1], v); v = fmaf(a.z, xf[2], v); v = fmaf(a.w, xf[3], v);
        v = fmaf(b.x, xf[4], v); v = fmaf(b.y, xf[5], v); v = fmaf(b.z, xf[6], v); v = fmaf(b.w, xf[7], v);
        acc[r][t] = v;
      }
    }
  }
  __shared__ float red[ROWS][MMAX][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * MMAX) {
    const int r = tid / MMAX, t = tid % MMAX;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[r][t][q];
      out[(size_t)t * N + r0 + r] = v;
    }
  }
}


// ============================================================================ cgemv (m = 1): 12-bit compressed GEMV
// Lossless planes, row-major: L [N][K] low bytes, C [N][K/2] 4-bit codes (sign | exponent offset from base), byte j of
// each 4-byte group g holding codes 8g + j and 8g + j + 4. Exponents outside the 8-value window decode wrong and are
// corrected exactly by a dedicated escape warp from a per-CTA table (entry 0 = count; ((row << 16) | k, delta bits)).
// Every weight and x load is issued before the first FMA; fixed-order reductions (deterministic).
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BF16 GEMV (the unpooler slice at m = 1: W[:1024] bf16 [1024, 4096]): 4 rows per CTA, 512 threads, thread i owns
// k = 8i (one 16 B load per row and one of x), fixed-order reductions.
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH) sgemv_kernel(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    __nv_bfloat16* __restrict__ out) {
  constexpr int PER = K / NTH / 8;
  constexpr int NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  uint4 w[ROWS][PER], xv[PER];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) w[r][j] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(r0 + r) * K + j * (NTH * 8) + tid * 8));
#pragma unroll
  for (int j = 0; j < PER; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat162* hx = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
    float xf[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(hx[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&w[r][j]);
      float v = acc[r];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 a = __bfloat1622float2(hw[e]); v = fmaf(a.x, xf[2 * e], v); v = fmaf(a.y, xf[2 * e + 1], v); }
      acc[r] = v;
    }
  }
  __shared__ float red[ROWS][NWARP];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void sgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t N) {
  TORCH_CHECK(W.size(1) == 4096 && x.size(0) == 1 && N % 4 == 0 && N <= W.size(0), "sgemv: m = 1, K = 4096");
  sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M) {
  TORCH_CHECK(W.size(1) == 8192 && M == 1, "rgemv: K = 8192, m = 1");
  rgemv_kernel<1, 1><<<(int)W.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), (int)M,
      (int)W.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _gq_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def _run_gq(x, layer, m, n, k, state):
    return _run_gq_family(x, layer, m, n, k, state, GQ_CONFIGS, _gq_bn, "gq")



WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    # router: BF16 hi|lo tiles (plain), 64-row blocks
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def _wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def _wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(WIDE_CONFIGS.items()):
        hl = layer == "router"
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


RK_SRC = r"""
// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#define RK_QS 6
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rk_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
static PFN_cuTensorMapEncodeTiled_v12000 rk_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr, "rk: cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap rk_map(const void* ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc) {
  const int M = x.size(0);
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && M >= 1 && M <= 256, "rk: shape");
  const int ntt = (M + 63) / 64;
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk: grid ", grid, " exceeds the SM count ", nsm, " (every CTA must be resident: one per SM)");
  const CUtensorMap mw = rk_map(Wc.data_ptr(), 256, 16384), mx = rk_map(x.data_ptr(), M, 8192);
  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""


def _rk_src(sm90):
    if sm90:
        return RK_SRC


# ---------------------------------------------------------------------------
# Extension build: everything above is assembled into one CUDA translation
# ---------------------------------------------------------------------------
def _get_ext():
    """Compile everything as one CUDA translation unit with load_inline.

    no_implicit_headers: the CUDA file includes only the ATen headers it needs
    (torch/types.h alone costs ~45 s of nvcc time).
    """
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        skinny = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", WG_HELPERS if sm90 else "")
        rs_defs = wgmma_rs_defs([8, 16, 32, 64, 128]) if sm90 else ""
        # Order matters only in that each route's helpers precede its kernel;
        # every route is self-contained apart from the shared helpers above.
        cuda_src = (skinny + rs_defs + _tma_srcs(sm90) + MAP_SRC + _gemm_src(sm90)
                    + _wide_src(sm90) + _sk2_src(sm90) + _gq_src(sm90) + _gqx_src(sm90)
                    + RGEMV_SRC + _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


# ---------------------------------------------------------------------------------------------- CUTLASS route (m 129..256)
# CUTLASS 3.x sm90 TMA warp-specialized PINGPONG GEMM, persistent tile scheduler, swap-AB (D^T = W x^T: A = W [N, K]
# row-major, B = x [M, K] as a K x M column-major matrix, D column-major [N, M] = out [M, N] row-major), BF16 in/out,
# FP32 accumulate, tile 128 (W rows) x 64 (tokens) x 64, cluster 1x1. Headers: CUTLASS bundled with flashinfer (in the
# SGLang image). Sweep (pod/cutsweep.py, 54 configs x 8 shapes): it beats cuBLAS only for pooler m=256 (31.3 vs 34.5 us
# raw); a missing header or a failed build just leaves the route out.
CUTLASS_LAYERS = {"pooler"}   # layers that get the CUTLASS candidate (scored m=256 only)
CUTLASS_BUILD_TIMEOUT = 170   # seconds prepare waits for the CUTLASS build (main build ~25 s; prepare's limit is 300 s)
CUTLASS_SRC = r"""
/***************************************************************************************************
 * The GEMM below instantiates NVIDIA CUTLASS 3.x/4.x templates (header-only, compiled from the CUTLASS headers that
 * ship in this image's FlashInfer package; nothing prebuilt is linked). Its type composition follows CUTLASS's Hopper
 * GEMM examples (examples/48_hopper_warp_specialized_gemm, 49_hopper_gemm_with_collective_builder), so it carries
 * CUTLASS's license notice:
 *
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = cutlass::bfloat16_t; using LC = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, 8, EC, LC, 8,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
int64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  return (int64_t)Gemm::get_workspace_size(args);
}
void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  Gemm gemm;
  TORCH_CHECK((size_t)ws.numel() >= Gemm::get_workspace_size(args), "cut_gemm: workspace");
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cut_gemm: cannot implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), st) == cutlass::Status::kSuccess, "cut_gemm: init");
  TORCH_CHECK(gemm.run(st) == cutlass::Status::kSuccess, "cut_gemm: run");
}
"""
_cut_ext = None
_cut_err = None


def _build_cutlass():
    """Build the CUTLASS extension (sm_90 only); on any failure record it and leave the route out."""
    global _cut_ext, _cut_err
    try:
        major, minor = torch.cuda.get_device_capability()
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("flashinfer (CUTLASS headers) not installed")
        root = os.path.join(list(spec.submodule_search_locations)[0], "data", "cutlass")
        inc = [os.path.join(root, "include"), os.path.join(root, "tools", "util", "include")]
        if not all(os.path.isdir(d) for d in inc):
            raise RuntimeError("CUTLASS headers missing under " + root)
        _cut_ext = load_inline(
            name=f"okc_bf16proj_v141_cutlass_{major}{minor}a",
            cpp_sources="#include <torch/extension.h>\nint64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out);\n"
                        "void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws);",
            cuda_sources=CUTLASS_SRC, functions=["cut_ws", "cut_gemm"], extra_include_paths=inc,
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-DNDEBUG"],
            verbose=False)
    except Exception as e:   # noqa: BLE001 - any failure only drops this optional route
        _cut_ext, _cut_err = None, f"{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Weight preparation (host side, runs in prepare()): fragment orders and the
# lossless 12-bit packing. The packing helpers share _exponent_window /
# _escapes; the per-kernel stream/table builders stay separate because their
# layouts and orderings differ (per-CTA streams vs k-ordered escape tables).
# ---------------------------------------------------------------------------
def _exponent_window(e71):
    """Best EXPONENT_WINDOW-wide window of BF16 exponent bits 7..1 for a matrix.

    Returns (base, offset): ``base`` is the window's lowest exponent, and
    ``offset`` is each weight's exponent clamped into [0, EXPONENT_WINDOW).
    The window is chosen to cover the most weights, so the fewest weights
    need exact correction as escapes.
    """
    hist = torch.bincount(e71.flatten(), minlength=EXPONENT_VALUES)
    win = torch.stack(
        [hist[b:b + EXPONENT_WINDOW].sum() for b in range(EXPONENT_VALUES - EXPONENT_WINDOW + 1)]
    )
    base = int(win.argmax())
    return base, (e71 - base).clamp(0, EXPONENT_WINDOW - 1)


def _escapes(e71, base):
    """Weights whose exponent falls outside [base, base + EXPONENT_WINDOW)."""
    return (e71 < base) | (e71 > base + EXPONENT_WINDOW - 1)


def pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    base, o = _exponent_window(e71)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = _escapes(e71, base)
    n_esc = int(esc.sum())
    if n_esc > n * k // MAX_ESCAPE_DENOMINATOR:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    # fragment order for low bytes and codes: [rg, s, lane, 8]
    lo_f = pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * BASE4_REPLICATE,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


# ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', UNPOOLER_SLICE_ROWS, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
    # okc_tune (14173ac4) retimed winners, Oct 25 12:30 (search-harness ratios vs production in brackets); pair.py both
    # orders: pooler16 -0.43, unpooler16 -0.60, o16 -0.15, qkv8 -0.12, qkv16 -0.08 us (pooler32 r160 / o32 r88: no gain)
    ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),      # tokr16 [0.973 vs 1.0]
    ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # tokr16 [0.961 vs 0.969]
    ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # tokr8 [0.969 vs 0.979]
    ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # tokr16 [0.985 vs 0.994]
    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
]
SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o16t_r68',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler16t_r192',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8t_r144',
    ('qkv_gate', 16): 'qkv16t_r200',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16t_r72',
    ('unpooler', 32): 'unpooler32_r144',
}
SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
USE_SK2 = True
WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def _sk2_name(layer, m):
    return SK2_PICK[(layer, _sk2_tokr(m))]


def _sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def _sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap_kb = SK2_DEFAULT_RING_CAP_KB
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap_kb = int(tail[1])
    ring = min(cap_kb * 1024, SK2_RING_SMEM_BUDGET_BYTES - extra)
    nstage = max(SK2_MIN_STAGES, min(SK2_MAX_STAGES, nst, ring // sst))
    return hl, nstage


def _sk2_src(sm90):
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = _sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    return SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))


def sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    base, o = _exponent_window(e71)
    code = (sign << 3) | o
    esc = _escapes(e71, base)
    if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()   # the original BF16 weights
    return {"packed": packed, "base4": base * BASE4_REPLICATE, "rows": rows, "cols": cols, "wbits": wbits}


CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
# (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)


def cg_compress(w, rows_per_cta):
    """Row-major 12-bit planes for cgemv (lossless; escapes corrected exactly from a per-CTA table)."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    base, o = _exponent_window(e71)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = _escapes(e71, base)
    if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)          # row-major order
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    delta = (wtrue - dec).contiguous()
    if not bool(((dec + delta) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)   # the original BF16 weights; the kernel forms the correction
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * BASE4_REPLICATE}


def sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


SK2_BALANCE = True   # v135: deal 4-row groups to sk2 row blocks so every block carries ~the same number of escapes
SK2_BALANCE_LAYERS = {"qkv_gate"}   # v136: only where it measured a gain (qkv m=8/16 -0.5 us; o/unpooler m=32 lost)


def sk2_balance_perm(w, n_rows, R):
    """Row order for R-row sk2 blocks over the first n_rows rows of w: rows sorted by escape count and snake-dealt, so
    every block holds ~the same number of escapes (the escape warps' work sets the CTA tail). perm[p] = original row."""
    wl = w[:n_rows]
    bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    cnt = _escapes(e71, base).sum(1)
    # 4-row groups are dealt (keeps 4-column output groups contiguous: vector stores in the epilogue)
    G4 = 4
    tcnt = cnt.view(n_rows // G4, G4).sum(1)
    order = torch.argsort(tcnt, descending=True, stable=True)
    TB = R // G4                                   # groups per block
    nb = n_rows // R
    i = torch.arange(n_rows // G4, device=w.device)
    rnd, pos = i // nb, i % nb
    blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
    tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
    tperm[blk * TB + rnd] = order
    return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


def sk2_table_add_perm(tab, pmax, perm, R):
    """Append each CTA's row map (R int32 output columns) after its escapes; entry 0 .y = the map's int2 offset."""
    ncta = tab.shape[0]
    pm2 = pmax + R // 2
    pm2 += pm2 % 2
    out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
    out[:, :pmax] = tab
    out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
    out[:, 0, 1] = pmax
    return out.contiguous(), pm2


def sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, the escaped weight's
    original BF16 bits); the kernel forms the exact correction (w - decode(w)) itself."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.to(torch.int32)
    return tab.contiguous(), pmax


def sk2_prepare(weights, state):
    state["sk2"] = {}
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            perm = None
            if SK2_BALANCE and CL == 1 and layer in SK2_BALANCE_LAYERS:
                pkey = (layer, n_rows, R)
                bal = state.setdefault("sk2_bal", {})
                if pkey not in bal:
                    pr = sk2_balance_perm(w, n_rows, R)
                    bal[pkey] = (pr, sk2_compress(w[:n_rows][pr].contiguous()))
                perm, comp = bal[pkey]
                if comp is None:
                    perm = None
            if perm is None:
                comp = state.setdefault("sk2_comp", {}).get(layer)
                if comp is None:
                    comp = sk2_compress(w.contiguous())
                    state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW, perm is not None)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                tab, pmax = sk2_table(comp, n_rows, R)
                if perm is not None:
                    tab, pmax = sk2_table_add_perm(tab, pmax, perm, R)
                shared[lkey] = (sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax)
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)
    state.pop("sk2_bal", None)


def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def _mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


# ---------------------------------------------------------------------------
# prepare(): build every weight layout once into `state`. The CUTLASS
# extension builds on a background thread while the weights are compressed.
# ---------------------------------------------------------------------------
def _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg):
    """Ensure one skinny decomposition's prepared inputs exist in ``state``:
    the per-CTA weight stream, its split-K counter, and (compressed only) its
    per-CTA escape table. Idempotent: shared decompositions are built once."""
    key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
    if key not in state["streams"]:
        state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
    ckey = (name, splits, rgc, wsn, ks, n_cfg)
    if ckey not in state["cnt"]:
        state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
    if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
        state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)


def prepare(weights):
    _get_ext()  # compile now, so a build failure surfaces in prepare(), not mid-run
    # the CUTLASS extension builds in the background (nvcc is a subprocess) while the weights are compressed; started
    # after the main build so the two load_inline calls never overlap
    cut_thread = None
    if CUTLASS_LAYERS and torch.cuda.get_device_capability()[0] >= 9 and _cut_ext is None and _cut_err is None:
        cut_thread = threading.Thread(target=_build_cutlass, daemon=True)
        cut_thread.start()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
    old_paths = not (sm90_ and USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    # split-K partials: kernels run one at a time, so every layer shares one workspace
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * WEIGHT_ROWS[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * WEIGHT_ROWS[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            _get_small_router_ext()
            state["router_small"] = pack_small_router(w)
            assert state["router_small"].is_contiguous() and state["router_small"].data_ptr() % 128 == 0
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_f32"] = w.float().contiguous()   # the router GEMV (m = 1) reads the FP32 weights as given
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256), rk
            state["rk_zc"] = torch.zeros(8, dtype=torch.int32, device=w.device)   # rk: grid counters, one per token-tile count
            # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
            src = torch.stack([pack_fragments(hi), pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            del t
            for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            if name in CG_CONFIGS:
                wc = w[:UNPOOLER_SLICE_ROWS] if name == "unpooler" else w   # the unpooler at m = 1 is the slice
                cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                if cg is not None:
                    state.setdefault("cg", {})[name] = cg
            if name in TORCHT_LAYERS and sm90_:
                state.setdefault("rawT", {})[name] = w.t().contiguous()
            c = pack_compressed(w) if name in COMPRESSED_LAYERS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mtk <= 1) else n
            _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mm <= 8) else n
                _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
        for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
            if lname != name or not comp or not (old_paths or WIDE_BF16):   # unused on sm_90 (cuBLAS / gq / gqx)
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count   # rk: one resident CTA per SM
    state["old_paths"] = old_paths
    sk2_prepare(weights, state)
    if cut_thread is not None:
        cut_thread.join(timeout=CUTLASS_BUILD_TIMEOUT)   # ~70-80 s on an H200 pod; never let it threaten prepare's limit
        if cut_thread.is_alive():
            print(f"[okc] CUTLASS route off: build still running after {CUTLASS_BUILD_TIMEOUT} s", file=sys.stderr)
    if _cut_ext is not None and (cut_thread is None or not cut_thread.is_alive()):
        # Persistent scheduler workspace: a few KB at most, sized with headroom.
        state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
        state["cut_layers"] = set(CUTLASS_LAYERS) & set(state["raw"])
    elif _cut_err is not None:
        print(f"[okc] CUTLASS route off: {_cut_err}", file=sys.stderr)
    _selftest(weights, state)
    _autotune(weights, state)
    torch.cuda.empty_cache()
    return state


# ---------------------------------------------------------------------------
# Routing, selftest, autotune. The static route maps (layer, m) to a family;
# _candidates adds the autotuner alternatives; _autotune times them as CUDA
# graph replays from a cold L2 and keeps the fastest per (layer, m) in
# state["choice"]. An untuned m in _nearby_choice takes its bucket's choice.
# ---------------------------------------------------------------------------
# Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
# "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def _route(layer, m, sm90):
    for mmax, r in (ROUTE_ROUTER if layer == "router" else ROUTE_BF16):
        if m <= mmax:
            return r
    return "torch"


USE_RK = True   # v141: router m 33..256 candidate 'rk' (order-dependent FP32 adds; stress compares it by tolerance)


def _route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, _sk2_name(layer, m), _sk2_tokr(m))
    if route == "skinny":
        return (layer, route, _mt_key(m))
    if route == "wide":
        return (layer, route) + _wide_key(m)
    if route == "rk":
        return (layer, route, (m + 63) // 64)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def _selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(SELFTEST_SEED)
    for layer, w in weights.items():
        n, k = w.shape
        for m in SELFTEST_M:
            nn = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in _candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else _route_key(layer, m, ch[0]))
                try:
                    out = _run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError as e:  # launch failure -> fall back
                    err = float("inf")
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                if not err < tol:
                    state["disabled"].add(key)
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


# ---------------------------------------------------------------------------
# Running one projection: each _run_* allocates a fresh output and launches
# one route. Nothing is cached across calls.
# ---------------------------------------------------------------------------
def _run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = _mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = _get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, WEIGHT_ROWS[layer] // 16, ks, dg)
    else:
        # Uncompressed: no escapes, so the patch table is the dummy and
        # pmax/base4 are 0 (the kernel skips exact correction).
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, WEIGHT_ROWS[layer] // 16, ks, dg)
    return out


def _run_wide(x, layer, m, n, k, state):
    tok, nz = _wide_key(m)
    nwg, splits, _ = WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    wst = state["router_cat"] if layer == "router" else state["streams"][(layer, 0, rgc, 1, 4, n)]
    _get_ext().wide(wst, x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def _run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    _get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def _run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def _run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind == "skinny":
        return _run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return _run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return _run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return _run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return _run_gq(x, layer, m, n, k, state)
    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
    if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
        cg = state["cg"][layer]
        rows, nth = CG_CONFIGS[layer]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
        return out
    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().sgemv(state["raw"][layer], x, out, n)
        return out
    if kind == "rgemv":   # router, m = 1: FP32 weights straight from memory (one row per CTA)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rgemv(state["router_f32"], x, out, m)
        return out
    if kind == "cutlass":   # m 129..256: CUTLASS pingpong 128x64 (see CUTLASS_SRC)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _cut_ext.cut_gemm(state["raw"][layer], x, out, state["cut_ws"])
        return out
    if kind == "rk":   # router m 33..256: dedicated split-K GEMM with order-dependent FP32 adds (see RK_SRC)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rk(state["router_cat"], x, out, _rk_splits(m, state["nsm"]), state["rk_zc"])
        return out
    if kind == "torchT":   # cuBLAS on a K-major copy of W: a different (faster) kernel choice for unpooler m=128
        return torch.mm(x, state["rawT"][layer])
    return _run_torch(x, layer, m, n, k, state)


RK_SPLITS = {1: 16, 2: 12, 3: 10, 4: 8}   # token tiles of 64 -> K splits (grid 4 x ntt x S <= 128; pod/rk.py sweep)


def _rk_splits(m, nsm=132):
    """K splits for m tokens: the measured table, capped so the grid (4 x ntt x S, one CTA per SM, all resident: the
    in-kernel clear waits for every CTA) fits the GPU's SMs."""
    ntt = (m + 63) // 64
    return max(1, min(RK_SPLITS[ntt], nsm // (4 * ntt)))


def _static_choice(layer, m, state):
    route = _route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (_sk2_name(layer, m) in state.get("sk2", {})
                              and _route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not WIDE_BF16:
            route = "torch"
    if route != "torch" and _route_key(layer, m, route) in state.get("disabled", ()):
        # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and _route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = _mt_key(m)
        return ("skinny", TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def _candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [_static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and _route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
            and _route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gq_bn(m)) in GQ_CONFIGS
            and _route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
    if layer == "router" and m == 1 and "router_f32" in state:
        c.insert(0, ("rgemv",))
    if (USE_RK and layer == "router" and 32 < m <= 256 and state["sm90"] and "rk_zc" in state and state.get("nsm", 0) >= 16
            and _route_key(layer, m, "rk") not in state["disabled"]):
        c.insert(0, ("rk",))
    if layer == "unpooler" and m == 1:
        c.insert(0, ("sgemv",))
    if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
        c.insert(0, ("cgemv",))
    if layer in TORCHT_LAYERS and 32 < m <= 256 and layer in state.get("rawT", {}):
        c.append(("torchT",))
    if (128 < m <= 256 and layer in state.get("cut_layers", ()) and _cut_ext is not None
            and _route_key(layer, m, "cutlass") not in state["disabled"]):
        c.append(("cutlass",))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


AUTOTUNE = True
TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def _autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(L2_FLUSH_ROWS, 1024, dtype=torch.float32, device=dev)  # read-only sweep
    sink = torch.empty(flush.shape[1], dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(AUTOTUNE_SEED)
    reps = AUTOTUNE_REPS
    for m in TUNE_M:
        for layer, w in weights.items():
            n = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = _candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                _run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    _run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
            # cuBLAS): other m of the bucket take the best other candidate
            alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]
            if alt:
                state.setdefault("choice_alt", {})[(layer, m)] = cands[min(alt, key=lambda i: med[i])]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()
    # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
    for layer, m, tl in state.get("tune_log", []):
        ch = state["choice"][(layer, m)]
        desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                        for (name, t), c in zip(tl, _candidates(layer, m, state)))
        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
        print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
    print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


def _nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    if tm == 256 and m != 256 and layer != "router":
        # an untuned m in 129..255 runs SGLang's own call (F.linear): the m=256 picks (CUTLASS, gqx, torchT) were
        # only timed at m=256 and some lose to cuBLAS at partial tiles (pooler CUTLASS m=129 +14 us, unpooler
        # torchT m=240 +11%)
        return ("torch",)
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "cutlass" and m != tm:
        ch = state.get("choice_alt", {}).get((layer, tm))
        if ch is None:
            return None
    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    return ch


# ---------------------------------------------------------------------------
# Entry point: the harness calls prepare(weights) once, then kernel_fn per call.
# ---------------------------------------------------------------------------
def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or _nearby_choice(layer, m, state) or _static_choice(layer, m, state)
    out = _run(choice, x, layer, m, n, k, state)
    return out


import ctypes as C

class _KPointer(C.c_void_p):
    def __deepcopy__(self, memo):
        return self


_KP, _KI, _KSZ = _KPointer, C.c_int, C.c_size_t


import threading as _KThreading

_KTLS = _KThreading.local()
_KCUDA = None


def _k_ctx_bind(x):
    if getattr(_KTLS, "device", None) == x.device.index:
        return
    global _KCUDA
    if _KCUDA is None:
        cu = C.CDLL("libcuda.so.1")
        for name, args in (("cuCtxGetCurrent", [C.POINTER(C.c_void_p)]),
                           ("cuDeviceGet", [C.POINTER(C.c_int), C.c_int]),
                           ("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int]),
                           ("cuCtxSetCurrent", [C.c_void_p])):
            fn = getattr(cu, name)
            fn.argtypes, fn.restype = args, C.c_int
        _KCUDA = cu
    ctx = C.c_void_p()
    _KCUDA.cuCtxGetCurrent(C.byref(ctx))
    if not ctx.value:
        dev, primary = C.c_int(), C.c_void_p()
        index = x.device.index or 0
        if _KCUDA.cuDeviceGet(C.byref(dev), index) == 0 and _KCUDA.cuDevicePrimaryCtxRetain(C.byref(primary), dev) == 0:
            _KCUDA.cuCtxSetCurrent(primary)
    _KTLS.device = x.device.index


class _KAlgo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class _KHeuristic(C.Structure):
    _fields_ = [("algo", _KAlgo), ("workspaceSize", C.c_size_t), ("state", C.c_int),
                ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]


class _KLargeLt:
    def __init__(self):
        self.lib = C.CDLL("libcublasLt.so.13")
        signatures = {
            "cublasLtCreate": [C.POINTER(_KP)],
            "cublasLtMatmulDescCreate": [C.POINTER(_KP), _KI, _KI],
            "cublasLtMatmulDescSetAttribute": [_KP, _KI, _KP, _KSZ],
            "cublasLtMatrixLayoutCreate": [C.POINTER(_KP), _KI, C.c_uint64, C.c_uint64, C.c_int64],
            "cublasLtMatrixLayoutSetAttribute": [_KP, _KI, _KP, _KSZ],
            "cublasLtMatmul": [_KP] * 12 + [C.POINTER(_KAlgo), _KP, _KSZ, _KP],
            "cublasLtMatmulAlgoInit": [_KP, _KI, _KI, _KI, _KI, _KI, _KI, _KI, C.POINTER(_KAlgo)],
            "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(_KAlgo), _KI, _KP, _KSZ],
            "cublasLtMatmulAlgoCheck": [_KP, _KP, _KP, _KP, _KP, _KP, C.POINTER(_KAlgo), C.POINTER(_KHeuristic)],
        }
        for name, argtypes in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, _KI
        self.handle = _KP()
        self.call("cublasLtCreate", C.byref(self.handle))
        self.desc = {}
        for mode, transpose in (("tn", 1), ("nn", 0)):
            desc = _KP()
            self.call("cublasLtMatmulDescCreate", C.byref(desc), 68, 0)
            op = _KI(transpose)
            self.call("cublasLtMatmulDescSetAttribute", desc, 3, C.byref(op), C.sizeof(op))
            self.desc["nn" if mode == "rownn" else mode] = desc
        self.workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
        self.required_workspace = 0
        self.alpha, self.beta = C.c_float(1), C.c_float(0)

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        clone.lib = self.lib
        clone.handle = self.handle
        clone.desc = dict(self.desc)
        clone.workspace = self.workspace.clone()
        clone.required_workspace = self.required_workspace
        clone.alpha, clone.beta = self.alpha, self.beta
        return clone

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} status {status}")

    def layouts(self, mode, m, n, k, batch):
        specs = ((k, n // batch, k, k * n // batch), (k, m, k, 0), (n // batch, m, n, n // batch))
        if mode == "nn":
            specs = ((n // batch, k, n, n // batch),) + specs[1:]
        elif mode == "rownn":
            specs = ((m, k, k, 0), (k, n, n, 0), (m, n, n, 0))
        layouts = []
        for rows, cols, ld, stride in specs:
            desc = _KP()
            self.call("cublasLtMatrixLayoutCreate", C.byref(desc), 14, rows, cols, ld)
            if mode == "rownn":
                order = _KI(1)
                self.call("cublasLtMatrixLayoutSetAttribute", desc, 1, C.byref(order), C.sizeof(order))
            count, offset = _KI(batch), C.c_int64(stride)
            self.call("cublasLtMatrixLayoutSetAttribute", desc, 5, C.byref(count), C.sizeof(count))
            self.call("cublasLtMatrixLayoutSetAttribute", desc, 6, C.byref(offset), C.sizeof(offset))
            layouts.append(desc)
        return layouts

    def plan(self, mode, m, n, k, cfg):
        batch = cfg["batch"]
        layouts = self.layouts(mode, m, n, k, batch)
        if "values" in cfg:
            algo = _KAlgo((C.c_uint64 * 8)(*cfg["values"]))
        else:
            algo = _KAlgo()
            self.call("cublasLtMatmulAlgoInit", self.handle, 68, 0, 14, 14, 14, 14, 66, C.byref(algo))
            attrs = (("tile", 1, C.c_uint32), ("splitk", 2, C.c_int32), ("red", 3, C.c_uint32),
                     ("swz", 4, C.c_uint32), ("custom", 5, C.c_uint32), ("stages", 6, C.c_uint32),
                     ("cluster", 8, C.c_uint16))
            for name, attr, ctype in attrs:
                value = ctype(cfg[name])
                status = self.lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(value), C.sizeof(value))
                if status:
                    raise RuntimeError(f"large-proj config {name} status {status}")
        result = _KHeuristic()
        status = self.lib.cublasLtMatmulAlgoCheck(self.handle, self.desc["nn" if mode == "rownn" else mode], *layouts, layouts[2],
                                                  C.byref(algo), C.byref(result))
        if status:
            raise RuntimeError(f"large-proj AlgoCheck status {status}")
        self.required_workspace = max(self.required_workspace, int(result.workspaceSize))
        return mode, layouts, algo

    def run(self, x, weight, n, plan):
        _k_ctx_bind(x)
        mode, layouts, algo = plan
        out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
        a, b, c = layouts
        ap = _KP(x.data_ptr() if mode == "rownn" else weight.data_ptr())
        bp = _KP(weight.data_ptr() if mode == "rownn" else x.data_ptr())
        self.call("cublasLtMatmul", self.handle, self.desc["nn" if mode == "rownn" else mode], C.byref(self.alpha), ap, a,
                  bp, b, C.byref(self.beta), _KP(out.data_ptr()), c, _KP(out.data_ptr()), c,
                  C.byref(algo), _KP(self.workspace.data_ptr()), self.workspace.numel(),
                  _KP(torch.cuda.current_stream().cuda_stream))
        return out


def _kserial(batch, values):
    return {"batch": batch, "values": values}


def _kplan(tile, custom, cluster):
    return {"batch": 1, "tile": tile, "stages": 35, "splitk": 1, "red": 0, "swz": 0,
            "custom": custom, "cluster": cluster}


_KLARGE_CONFIGS = {
    ("qkv_gate", 256): ("tn", _kplan(124, 0, 2)),
    ("pooler", 256): ("rownn", _kplan(18, 0, 2)),
    ("pooler", 128): ("nn", _kplan(15, 4, 2)),
    ("o", 256): ("tn", _kplan(20, 6, 2)),
    ("unpooler", 128): ("nn", _kplan(18, 4, 2)),  # [SWEEP-bf16 P1F] RBX/BFR P1 plan, static (judge 3/3 reads)
    ("unpooler", 256): ("rownn", _kplan(20, 4, 2)),
}
_KLARGE_TRANSPOSED = {("unpooler", 256), ("pooler", 256), ("pooler", 128), ("unpooler", 128)}
_klarge_base_prepare = prepare
_klarge_base_kernel = kernel_fn


def prepare(weights):
    state = _klarge_base_prepare(weights)
    state["klarge_rawT"] = {
        "unpooler": state["rawT"]["unpooler"],
        "pooler": state["raw"]["pooler"].T.contiguous(),
    }
    state["klarge_lt"] = _KLargeLt()
    state["klarge_plans"] = {}
    for key, (mode, cfg) in _KLARGE_CONFIGS.items():
        layer, m = key
        n, k = state["raw"][layer].shape
        state["klarge_plans"][key] = state["klarge_lt"].plan(mode, m, n, k, cfg)
    state["klarge_lt"].workspace = torch.empty(state["klarge_lt"].required_workspace, device="cuda", dtype=torch.uint8)
    return state


def kernel_fn(data, state):
    x = data["x"]
    key = (data["config"]["layer"], x.shape[0])
    plan = state["klarge_plans"].get(key)
    if plan is not None:
        weight = state["klarge_rawT"][key[0]] if key in _KLARGE_TRANSPOSED else state["raw"][key[0]]
        return state["klarge_lt"].run(x, weight, data["config"]["n"], plan)
    return _klarge_base_kernel(data, state)




# Independent exact escape corrections: per-warp scratch avoids atomics.
ESC_WARPS = 4
_ESC_EXT = None
_ORIGINAL_RUN_SK2 = _run_sk2
_ORIGINAL_PREPARE = prepare

def _escape_source():
    source = _sk2_src(True)
    source = source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)', str(ESC_WARPS))
    source = source.replace('NW + (COMP ? NESC : 0)', 'NW')
    source = source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");', '')
    start = source.index('      int pe = 1;')
    end = source.index('\n    }\n    mbar_arrive(cbar);', start)
    source = source[:start] + r"""
      for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
        const int2 e = ptab_s[pe];
        const int kk = e.x & 0xFFFF;
        if (lane < M && kk >= gs0 * KST && kk < (gs0 + nst) * KST) {
          const float v = __bfloat162float(x[(size_t)lane * K + kk]);
          cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * v;
        }
      }
""" + source[end:]
    # Build only SK2; unchanged helper definitions precede it.
    shared = CUDA_SRC.replace('@@MBAR@@', MBAR_HELPERS).replace('@@BODY@@', '')
    return shared + MAP_SRC + source

def _escape_ext():
    global _ESC_EXT
    if _ESC_EXT is None:
        import hashlib
        src = _escape_source()
        _ESC_EXT = load_inline(
            name='proj_escape_' + hashlib.md5(src.encode()).hexdigest()[:12],
            cpp_sources='#include <torch/extension.h>\n' + next(line for line in CPP_SRC.splitlines() if line.startswith('void sk2(')) + '\n' + next(line for line in CPP_SRC.splitlines() if 'int64_t M, int64_t N, int64_t K, int64_t cfg' in line),
            cuda_sources=src, functions=['sk2'],
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90', '-Xptxas=-v'],
            no_implicit_headers=True, verbose=False)
    return _ESC_EXT

def prepare(weights):
    _escape_ext()
    state = _ORIGINAL_PREPARE(weights)
    for ci, stream, tab, pmax, b4, n_rows in state['sk2'].values():
        assert stream.data_ptr() % 128 == 0
        assert tab.data_ptr() % 16 == 0
        assert pmax == 0 or pmax * 8 % 16 == 0
    return state

def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    if layer not in ('qkv_gate', 'unpooler') or not 2 <= m <= 8:
        return _ORIGINAL_RUN_SK2(x, layer, m, n, k, state, dg, dbg)
    assert x.is_contiguous() and x.data_ptr() % 128 == 0 and k * 2 % 128 == 0
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state['sk2'][name]
    out = torch.empty((m,n), dtype=torch.bfloat16, device=x.device)
    _escape_ext().sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,_sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
    return out


# H200 backend choices measured across paired runs; weight-dependent selftests
# and route availability retain the base's correctness fallbacks.
_PINNED_LARGE = {
    ('qkv_gate',128): 'gqx', ('pooler',128): 'gq',
    ('o',128): 'torchT', ('unpooler',128): 'torchT', ('router',128): 'rk',
    ('qkv_gate',256): 'torch', ('pooler',256): 'cutlass',
    ('o',256): 'torchT', ('unpooler',256): 'torchT', ('router',256): 'rk',
}
def _autotune(weights,state):
    state['choice']={};state['choice_alt']={}
    for m in TUNE_M:
        for layer in weights:
            want=('rgemv' if layer=='router' else 'cgemv') if m==1 else ('rsmall' if layer=='router' and m<=16 else 'sk2') if m<=32 else _PINNED_LARGE[(layer,m)]
            cands=_candidates(layer,m,state)
            choice=next((c for c in cands if c[0]==want),cands[0])
            state['choice'][(layer,m)]=choice
            state['choice_alt'][(layer,m)]=next((c for c in cands if c[0]=='torch'),('torch',))
    print('[proj] pinned choices:',state['choice'],file=sys.stderr)

# Small-router kernel adapted from NathenFernandes/muse-bf16-proj (MIT), S=16, WK=8; kernel body: ruwwww
# routertail_small_kernel verbatim (Apache-2.0, github.com/ruwwww/open-kernel-challenge @c035bf9) [BFX R16].
SMALL_ROUTER_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void routertail_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void routertail_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t routertail_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void routertail_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 routertail_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}

template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void routertail_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }

}

template<int MT,int S,int WK>
int routertail_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=routertail_small_kernel<MT,S,WK>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int routertail_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return routertail_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return routertail_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return routertail_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

"""
SMALL_ROUTER_CPP = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'
_small_router_ext = None


def _get_small_router_ext():
    global _small_router_ext
    if _small_router_ext is None:
        _small_router_ext = load_inline(
            "ruw_routertail_small_s16w8_v1", SMALL_ROUTER_CPP, cuda_sources=SMALL_ROUTER_SRC,
            functions=["routertail_small"], extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
    return _small_router_ext


def pack_small_router(w, s=16, wk=8):
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()




_RS_RUN = _run
_RS_CANDIDATES = _candidates
_RS_KERNEL = kernel_fn

def _run(choice,x,layer,m,n,k,state):
    if choice[0] != 'rsmall':
        return _RS_RUN(choice,x,layer,m,n,k,state)
    assert x.is_contiguous() and x.data_ptr()%16 == 0 and k==8192
    out=torch.empty((m,256),dtype=torch.float32,device=x.device)
    rc=_get_small_router_ext().routertail_small(state['router_small'].data_ptr(),x.data_ptr(),out.data_ptr(),m,16,8,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'small router launch failed: {rc}')
    return out

def _candidates(layer,m,state):
    choices=_RS_CANDIDATES(layer,m,state)
    if layer=='router' and 1<m<=16 and 'router_small' in state and _route_key(layer,m,'rsmall') not in state.get('disabled',()):
        choices.insert(0,('rsmall',))
    return choices

def kernel_fn(data,state):
    c=data['config'];layer,m=c['layer'],c['m']
    if layer=='router' and 1<m<=16 and 'router_small' in state and _route_key(layer,m,'rsmall') not in state.get('disabled',()):
        return _run(('rsmall',),data['x'],layer,m,c['n'],c['k'],state)
    return _RS_KERNEL(data,state)



# _QBIG_PAYLOAD: payload removed (decoded inline)


# _qbig_zsrc: removed (plain-source rule)


_QBIG_CUDA = '\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#define QD __device__ __forceinline__\nQD uint32_t q_addr(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}\nQD void q_init(uint32_t p,int n) {asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}\nQD void q_expect(uint32_t p,int n) {asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(p),"r"(n):"memory");}\nQD void q_wait(uint32_t p,int phase) {\n  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(p),"r"(phase):"memory");\n}\nQD void q_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}\nQD uint32_t q_map(uint32_t p,int rank) {uint32_t d;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(d):"r"(p),"r"(rank));return d;}\nQD void q_push(uint32_t p,float4 v,uint32_t b) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");\n}\nQD uint64_t q_desc(uint32_t p) {return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}\nQD uint64_t q_time() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}\nQD void q_tma(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k),"r"(r),"r"(b),"l"(policy):"memory");\n}\nQD void q_tma3(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {\n  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k/64),"r"(r),"r"(b),"l"(policy):"memory");\n}\n@@MMA@@\ntemplate<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>\n__global__ __launch_bounds__((PIPE&2)?256:128) void qb_kernel(const __grid_constant__ CUtensorMap tw,\n    const __grid_constant__ CUtensorMap tx,__nv_bfloat16* out,int m,unsigned long long* stamps) {\n  constexpr int KT=(PIPE&32)?2:1, LR=(2304+G-1)/G, WB=NR*128, XB=TM*128, SB=KT*(WB+XB);\n  constexpr int NV=NR*TM/4, OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);\n  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;\n  extern __shared__ __align__(1024) unsigned char sm[];\n  const int tid=threadIdx.x, lane=tid&31, rank=blockIdx.x%S, group=blockIdx.x/S;\n  const int tok=blockIdx.y*TM, row=group*2304/G, rows=(group+1)*2304/G-row;\n  const int kb=(128/KT)*rank/S, nkb=(128/KT)*(rank+1)/S-kb, own=RED>=4?((NV/128+S-1-rank)/S)*128:min(OWN,NV-rank*OWN);\n  const uint32_t base=q_addr(sm), bars=base+ST*SB, empty=bars+ST*8, rb=base+RB;\n  uint64_t begin=0, stream_end=0;\n  if constexpr(PROFILE) begin=q_time();\n  if constexpr(LR<NR) {\n    for(int i=tid;i<ST*(NR-LR)*32;i+=blockDim.x) {\n      int stage=i/((NR-LR)*32), j=i%((NR-LR)*32);\n      ((uint32_t*)(sm+stage*SB+LR*128))[j]=0;\n    }\n    asm volatile("fence.proxy.async.shared::cta;":::"memory");\n    __syncthreads();\n  }\n  uint64_t pw,px;\n  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));\n  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));\n  auto tma=[&](uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {\n    if constexpr(PIPE&2048)q_tma3(p,tm,k,r,b,policy);else q_tma(p,tm,k,r,b,policy);\n  };\n  if constexpr(PIPE&4096)asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pw));\n  auto load=[&](int i) {\n    const int stage=i%ST;\n    q_expect(bars+stage*8,KT*(LR*128+XB));\n    if constexpr(KT==2)tma(base+stage*SB,&tw,0,(group*128+(kb+i)*KT)*NR,bars+stage*8,pw);\n    else tma(base+stage*SB,&tw,(kb+i)*64,group*LR,bars+stage*8,pw);\n#pragma unroll\n    for(int q=0;q<KT;q++)tma(base+stage*SB+KT*WB+q*XB,&tx,((kb+i)*KT+q)*64,tok,bars+stage*8,px);\n  };\n  if(tid==((PIPE&2)?128:0)) {\n#pragma unroll\n    for(int i=0;i<ST;i++) {q_init(bars+i*8,1);if constexpr(PIPE&2)q_init(empty+i*8,4);}\n    if constexpr(S>1) q_init(rb,1);\n    asm volatile("fence.proxy.async.shared::cta;":::"memory");\n    if constexpr(S>1 && RED<4) {q_expect(rb,(S-1)*own*16);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n#pragma unroll\n    for(int i=0;i<ST;i++) if(i<nkb)load(i);\n  }\n  __syncthreads();\n  if constexpr(S>1 && RED>=4) {\n    if(tid==((PIPE&2)?128:0)) {\n      q_expect(rb,(S-1)*own*16);\n      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n    }\n    __syncthreads();\n  }\n  if constexpr(S>1 && RED!=3) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");\n  if constexpr(PIPE&2) {\n    if(tid>=128) {\n      if(tid==128) for(int i=ST;i<nkb;i++) {q_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n      if constexpr(S>1) {\n        if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");\n        asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n        if constexpr(RED==1 || RED==3) q_sync();\n      }\n      return;\n    }\n  }\n  float a[(TM/64)*(NR/2)];\n#pragma unroll\n  for(int h=0;h<TM/64;h++)\n#pragma unroll\n    for(int j=0;j<NR/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(a[h*(NR/2)+j]));\nif constexpr(PIPE&64) asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#pragma unroll 1\n  for(int i=0;i<nkb;i++) {\n    const int stage=i%ST;\n    q_wait(bars+stage*8,(i/ST)&1);\n#pragma unroll\n    for(int h=0;h<TM/64;h++)\n#pragma unroll\n      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");\n    if constexpr(!(PIPE&64)) asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#pragma unroll\n    for(int k=0;k<4*KT;k++) {\n      if constexpr(PIPE&8) {\n        uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);\n        @@SWAP@@\n      } else {\n#pragma unroll\n        for(int h=0;h<TM/64;h++) {\n          uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+h*8192+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);\n          @@CALL@@\n        }\n      }\n    }\n    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(((PIPE>>8)&7)?((PIPE>>8)&7):1):"memory");\n    else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#pragma unroll\n    for(int h=0;h<TM/64;h++)\n#pragma unroll\n      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");\n    int done=i-((PIPE&1)?(((PIPE>>8)&7)?((PIPE>>8)&7):1):0);\n    if(done>=0 && done+ST<nkb) {\n      if constexpr(PIPE&2) {if(lane==0)asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(empty+(done%ST)*8):"memory");}\n      else if(tid==0)load(done+ST);\n    }\n  }\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n  if constexpr(PROFILE) stream_end=q_time();\n  auto emit=[&](int v,float4 z) {\n    int t=v%128, j=(v/128)%(NR/8), h=(v/128)/(NR/8);\n    int rr=h*64+(t/32)*16+(t%32)/4, cc=j*8+(t%4)*2;\n    float vv[4]={z.x,z.y,z.z,z.w};\n#pragma unroll\n    for(int c=0;c<4;c++) {\n      int tr,wc;\n      if constexpr(PIPE&8) {tr=tok+(v/128)*8+(t%4)*2+(c&1);wc=(t/32)*16+(t%32)/4+(c/2)*8;}\n      else {tr=tok+rr+(c/2)*8;wc=cc+(c&1);}\n      if(tr<m && wc<rows)out[tr*2304+row+wc]=__float2bfloat16_rn(vv[c]);\n    }\n  };\n  if constexpr(S==1) {\n#pragma unroll\n    for(int h=0;h<TM/64;h++)\n#pragma unroll\n      for(int j=0;j<NR/8;j++)emit((h*(NR/8)+j)*128+tid,make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]));\n  } else if constexpr(RED>=4) {\n    static_assert(NR==64 && (PIPE&8));\n    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n#pragma unroll\n    for(int j=0;j<TM/8;j++) {\n      int dst=j%S,local=(j/S)*128+tid,slot=rank-(rank>dst);\n      if(dst!=rank) q_push(q_map(base+RECV+(slot*OWN+local)*16,dst),\n        make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]),q_map(rb,dst));\n    }\n    q_wait(rb,0);\n#pragma unroll\n    for(int j=0;j<TM/8;j++) if(j%S==rank) {\n      float4 sum=make_float4(0,0,0,0);\n#pragma unroll\n      for(int r=0;r<S;r++) {\n        float4 z=r==rank?make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]):\n          *(float4*)(sm+RECV+((r-(r>rank))*OWN+(j/S)*128+tid)*16);\n        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;\n      }\n      if constexpr(RED==4) emit(j*128+tid,sum);\n      else {\n        a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;\n      }\n    }\n    if constexpr(RED==5) {\n      asm volatile("bar.sync 1,128;":::"memory");\n#pragma unroll\n      for(int j=0;j<TM/8;j++) if(j%S==rank) {\n        __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]),hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n        int rr=(tid/32)*16+((tid%16)/8)*8,cc=8*j+tid%8;\n        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::\n          "r"(base+2*(cc*72+rr)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");\n      }\n      asm volatile("bar.sync 1,128;":::"memory");\n      for(int q=tid;q<(own/128)*64;q+=128) {\n        int token=(rank+(q/64)*S)*8+(q/8)%8,rr=(q%8)*8;\n        if(tok+token<m) *(uint4*)(out+(tok+token)*2304+row+rr)=*(uint4*)((__nv_bfloat16*)sm+token*72+rr);\n      }\n    }\n  } else {\n    if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");\n    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n#pragma unroll\n    for(int h=0;h<TM/64;h++)\n#pragma unroll\n      for(int j=0;j<NR/8;j++) {\n        int v=(h*(NR/8)+j)*128+tid,dst=v/OWN,local=v%OWN;\n        float4 z=make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]);\n        uint32_t p=base+RECV+(rank*OWN+local)*16;\n        if(dst==rank)*(float4*)(sm+RECV+(rank*OWN+local)*16)=z;\n        else q_push(q_map(p,dst),z,q_map(rb,dst));\n      }\n    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;":::"memory");else __syncthreads();\n    q_wait(rb,0);\n    for(int t=tid;t<own;t+=128) {\n      float4 sum=make_float4(0,0,0,0);\n#pragma unroll\n      for(int r=0;r<S;r++) {\n        float4 z=*(float4*)(sm+RECV+(r*OWN+t)*16);\n        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;\n      }\n      emit(rank*OWN+t,sum);\n    }\n    if constexpr(RED==1 || RED==3) q_sync();\n  }\n  if constexpr(PROFILE) if(lane==0) {\n    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));\n    int idx=((blockIdx.y*gridDim.x+blockIdx.x)*4+tid/32)*4;\n    stamps[idx]=smid;stamps[idx+1]=begin;stamps[idx+2]=stream_end;stamps[idx+3]=q_time();\n  }\n}\nusing QEncode=decltype(&cuTensorMapEncodeTiled);\nint q_tensor_map(CUtensorMap* tm,int64_t p,int inner,int rows,int box,bool map3) {\n  static QEncode encode=nullptr;\n  if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);if(!encode)return -100;}\n  if(map3) {\n    uint64_t dims[3]={64,(uint64_t)inner/64,(uint64_t)rows},stride[2]={128,(uint64_t)inner*2};\n    uint32_t box3[3]={64,1,(uint32_t)box},es[3]={1,1,1};\n    return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)p,dims,stride,box3,es,CU_TENSOR_MAP_INTERLEAVE_NONE,\n      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n  }\n  uint64_t dims[2]={(uint64_t)inner,(uint64_t)rows},stride[1]={(uint64_t)inner*2};uint32_t b[2]={64,(uint32_t)box},es[2]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,\n    CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\ntemplate<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>\nint qb_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {\n  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);\n  constexpr int KT=(PIPE&32)?2:1;\n  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?(RED>=4?S-1:S)*OWN*16:0);\n  auto fn=qb_kernel<NR,G,TM,S,ST,PIPE,RED,PROFILE>;\n  static bool ready=false;\n  if(!ready){cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return -(int)e;\n    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(e!=cudaSuccess)return -(int)e;ready=true;}\n  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(G*S,(m+TM-1)/TM);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={S,1,1};if(S>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  if(query==1){int v=0;cudaError_t e=S>1?cudaOccupancyMaxActiveClusters(&v,fn,&cfg):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,fn,cfg.blockDim.x,bytes);return e==cudaSuccess?v:-(int)e;}\n  CUtensorMap tw,tx;int e=q_tensor_map(&tw,w,KT==2?64:8192,KT==2?G*128*NR:G*LR,KT==2?KT*NR:LR,PIPE&2048);if(e)return e;e=q_tensor_map(&tx,x,8192,m,TM,PIPE&2048);if(e)return e;\n  return int(cudaLaunchKernelEx(&cfg,fn,tw,tx,(__nv_bfloat16*)out,m,(unsigned long long*)stats));\n}\nint qb_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {\n  switch(id){@@CASES@@ default:return -99;}\n}\n'
_QBIG_CONFIG = (64, 36, 128, 3, 8, 539, 5)
_QBIG_EXT = None
_qbig_parent_prepare = prepare
_qbig_parent_kernel = kernel_fn


def _qbig_mma(n):
    regs = ','.join(f'%{i}' for i in range(n // 2))
    outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
    return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'


def _qbig_ext():
    global _QBIG_EXT
    if _QBIG_EXT is None:
        import hashlib
        configs = [_QBIG_CONFIG]
        ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
        source = _QBIG_CUDA.replace('@@MMA@@', '\n'.join(_qbig_mma(n) for n in ns))
        source = source.replace('@@CALL@@', ' '.join(f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns))
        source = source.replace('@@SWAP@@', ' '.join(f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns))
        source = source.replace('@@CASES@@', '\n'.join(
            f"case {i}: return qb_launch_t<{','.join(map(str, c))},false>(w,x,out,m,stream,query,stats);" for i, c in enumerate(configs)))
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        namespace = 'qb_' + digest
        begin = source.index('#define QD')
        args = 'int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats'
        source = source[:begin] + f'namespace {namespace} {{\n' + source[begin:] + f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}'
        cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
        _QBIG_EXT = load_inline('qbig_' + hashlib.sha256(source.encode()).hexdigest()[:16], cpp,
            cuda_sources=source, functions=['qb_launch'], no_implicit_headers=True,
            with_pytorch_error_handling=False, verbose=False,
            extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])
    return _QBIG_EXT


def prepare(weights):
    state = _qbig_parent_prepare(weights)
    state["qbig"] = {"ext": _qbig_ext(), "weight": state["raw"]["qkv_gate"]}
    return state


def kernel_fn(data, state):
    x = data["x"]
    if data["config"]["layer"] == "qkv_gate" and x.shape[0] == 128 and x.is_contiguous():
        out = torch.empty((128, 2304), dtype=torch.bfloat16, device=x.device)
        rc = state["qbig"]["ext"].qb_launch(0, state["qbig"]["weight"].data_ptr(), x.data_ptr(), out.data_ptr(),
                                           128, torch.cuda.current_stream().cuda_stream, 0, 0)
        if rc:
            raise RuntimeError(f"qbig launch failed: {rc}")
        return out
    return _qbig_parent_kernel(data, state)




# Portions Copyright vjeux, Apache License 2.0; independent accumulator experiments by OKC proj inventor.
import torch
from torch.utils.cpp_extension import load_inline
_IV_CG_SRC='#include <ATen/core/Tensor.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {\n  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;\n  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);\n  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);\n}\n__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {\n  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);\n}\ntemplate <int ROWS, int NTH, int K, int NACC>\n__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,\n    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,\n    __nv_bfloat16* __restrict__ out, uint32_t base4) {\n  constexpr int PER = K / (16 * NTH);\n  constexpr int NWARP = NTH / 32;\n  static_assert(PER >= 1, "cgemv: K >= 16 NTH");\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int r0 = blockIdx.x * ROWS;\n  __shared__ float red[ROWS][NWARP + 1];\n  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row\n    const int2* tb = etab + (size_t)blockIdx.x * pmax;\n    int2 mine = make_int2(0, 0);\n    if (lane < pmax) mine = __ldg(&tb[lane]);\n    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);\n    float ec[ROWS];\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;\n    for (int i0 = 0; i0 <= cnt; i0 += 32) {\n      const int i = i0 + lane;\n      int2 en = mine;\n      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);\n      if (i >= 1 && i <= cnt) {\n        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);\n        const int rl = en.x >> 16;\n#pragma unroll\n        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;\n      }\n    }\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) {\n      float v = ec[r];\n#pragma unroll\n      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n      if (lane == 0) red[r][NWARP] = v;\n    }\n    __syncthreads();\n    return;\n  }\n  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n    for (int j = 0; j < PER; ++j) {\n      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;\n      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));\n      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));\n    }\n#pragma unroll\n  for (int j = 0; j < PER; ++j) {\n    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;\n    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));\n    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));\n  }\n  float acc[ROWS][NACC];\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) for(int a=0;a<NACC;++a) acc[r][a] = 0.f;\n#pragma unroll\n  for (int j = 0; j < PER; ++j) {\n    float xf[16];\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }\n    }\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) {\n      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};\n      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};\n      float v[NACC]; for(int a=0;a<NACC;++a) v[a]=acc[r][a];\n#pragma unroll\n      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]\n        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3\n        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7\n        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];\n        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);\n        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);\n        const float* xg = xf + 8 * g;\n        v[0%NACC] = fmaf(__uint_as_float(p01 << 16), xg[0], v[0%NACC]); v[1%NACC] = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v[1%NACC]);\n        v[2%NACC] = fmaf(__uint_as_float(p23 << 16), xg[2], v[2%NACC]); v[3%NACC] = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v[3%NACC]);\n        v[4%NACC] = fmaf(__uint_as_float(p45 << 16), xg[4], v[4%NACC]); v[5%NACC] = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v[5%NACC]);\n        v[6%NACC] = fmaf(__uint_as_float(p67 << 16), xg[6], v[6%NACC]); v[7%NACC] = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v[7%NACC]);\n      }\n      for(int a=0;a<NACC;++a) acc[r][a]=v[a];\n    }\n  }\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) {\n    float v=0.f; for(int a=0;a<NACC;++a) v+=acc[r][a];\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n    if (lane == 0) red[r][warp] = v;\n  }\n  __syncthreads();\n  if (tid < ROWS) {\n    float v = 0.f;\n#pragma unroll\n    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];\n    out[r0 + tid] = __float2bfloat16_rn(v);\n  }\n}\n\nvoid invoke(at::Tensor L,at::Tensor C,at::Tensor x,at::Tensor tab,at::Tensor out,int64_t pmax,int64_t b4,int64_t cfg){\n auto st=at::cuda::getCurrentCUDAStream();int N=out.size(1);\n auto lp=L.data_ptr<uint8_t>();auto cp=C.data_ptr<uint8_t>();auto xp=(const __nv_bfloat16*)x.data_ptr();auto tp=(const int2*)tab.data_ptr<int>();auto op=(__nv_bfloat16*)out.data_ptr();\n @DISPATCH@\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'

_IV_CG_CONFIGS=[(8192,2,512,1),(8192,2,512,2),(4096,2,256,1),(4096,2,128,4)]
_iv_cg_ext=None

def _iv_cg_build():
 global _iv_cg_ext
 if _iv_cg_ext is None:
  lines=[]
  for i,(k,r,t,a) in enumerate(_IV_CG_CONFIGS):
   lines.append(f'if(cfg=={i}) cgemv_kernel<{r},{t},{k},{a}><<<N/{r},{t}+32,0,st>>>(lp,cp,xp,tp,pmax,op,b4);')
  _iv_cg_ext=load_inline(name='inv_proj_cg_candidate_v005',cpp_sources='#include <torch/extension.h>\nvoid invoke(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',cuda_sources=_IV_CG_SRC.replace('@DISPATCH@','\n'.join(lines)),functions=['invoke'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],no_implicit_headers=True)
 return _iv_cg_ext

def _iv_cg_run(c,x,n,i):
 y=torch.empty((1,n),device=x.device,dtype=torch.bfloat16)
 _iv_cg_build().invoke(c['L'],c['C'],x,c['tab'],y,c['pmax'],c['base4'],i)
 return y

# Portions Copyright vjeux, Apache License 2.0; normalized router codec scale tuning.
import torch
from torch.utils.cpp_extension import load_inline
_IV_R24_SRC='#include <ATen/core/Tensor.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {\n  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;\n  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);\n  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);\n}\n__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {\n  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);\n}\ntemplate <int ROWS, int NTH, int K, int NACC, int SHIFT>\n__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,\n    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,\n    float* __restrict__ out, uint32_t base4) {\n  constexpr float INV_SCALE = 1.f / float(1<<SHIFT);\n  constexpr int PER = K / (16 * NTH);\n  constexpr int NWARP = NTH / 32;\n  static_assert(PER >= 1, "cgemv: K >= 16 NTH");\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int r0 = blockIdx.x * ROWS;\n  __shared__ float red[ROWS][NWARP + 1];\n  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row\n    const int2* tb = etab + (size_t)blockIdx.x * pmax;\n    int2 mine = make_int2(0, 0);\n    if (lane < pmax) mine = __ldg(&tb[lane]);\n    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);\n    float ec[ROWS];\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;\n    for (int i0 = 0; i0 <= cnt; i0 += 32) {\n      const int i = i0 + lane;\n      int2 en = mine;\n      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);\n      if (i >= 1 && i <= cnt) {\n        const float c = __bfloat162float(x[en.x & 8191]) * esc_delta((uint32_t)en.y, base4 & 0xFFu) * ((en.x & 16383) >= 8192 ? INV_SCALE : 1.f);\n        const int rl = en.x >> 16;\n#pragma unroll\n        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;\n      }\n    }\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) {\n      float v = ec[r];\n#pragma unroll\n      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n      if (lane == 0) red[r][NWARP] = v;\n    }\n    __syncthreads();\n    return;\n  }\n  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n    for (int j = 0; j < PER; ++j) {\n      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;\n      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));\n      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));\n    }\n#pragma unroll\n  for (int j = 0; j < PER; ++j) {\n    const __nv_bfloat16* xp = x + ((j * 16 * NTH + 16 * tid) % 8192);\n    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));\n    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));\n  }\n  float acc[ROWS][NACC];\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) for(int a=0;a<NACC;++a) acc[r][a] = 0.f;\n#pragma unroll\n  for (int j = 0; j < PER; ++j) {\n    float xf[16];\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }\n    }\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) {\n      const float xs=((j*16*NTH+16*tid)>=8192)?INV_SCALE:1.f;\n      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};\n      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};\n      float v[NACC]; for(int a=0;a<NACC;++a) v[a]=acc[r][a];\n#pragma unroll\n      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]\n        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3\n        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7\n        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];\n        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);\n        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);\n        const float* xg = xf + 8 * g;\n        v[0%NACC] = fmaf(__uint_as_float(p01 << 16), (xg[0]*xs), v[0%NACC]); v[1%NACC] = fmaf(__uint_as_float(p01 & 0xFFFF0000u), (xg[1]*xs), v[1%NACC]);\n        v[2%NACC] = fmaf(__uint_as_float(p23 << 16), (xg[2]*xs), v[2%NACC]); v[3%NACC] = fmaf(__uint_as_float(p23 & 0xFFFF0000u), (xg[3]*xs), v[3%NACC]);\n        v[4%NACC] = fmaf(__uint_as_float(p45 << 16), (xg[4]*xs), v[4%NACC]); v[5%NACC] = fmaf(__uint_as_float(p45 & 0xFFFF0000u), (xg[5]*xs), v[5%NACC]);\n        v[6%NACC] = fmaf(__uint_as_float(p67 << 16), (xg[6]*xs), v[6%NACC]); v[7%NACC] = fmaf(__uint_as_float(p67 & 0xFFFF0000u), (xg[7]*xs), v[7%NACC]);\n      }\n      for(int a=0;a<NACC;++a) acc[r][a]=v[a];\n    }\n  }\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) {\n    float v=0.f; for(int a=0;a<NACC;++a) v+=acc[r][a];\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n    if (lane == 0) red[r][warp] = v;\n  }\n  __syncthreads();\n  if (tid < ROWS) {\n    float v = 0.f;\n#pragma unroll\n    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];\n    out[r0 + tid] = v;\n  }\n}\n\nvoid invoke(at::Tensor L,at::Tensor C,at::Tensor x,at::Tensor tab,at::Tensor out,int64_t pmax,int64_t b4,int64_t cfg){\n auto st=at::cuda::getCurrentCUDAStream();int N=out.size(1);\n auto lp=L.data_ptr<uint8_t>();auto cp=C.data_ptr<uint8_t>();auto xp=(const __nv_bfloat16*)x.data_ptr();auto tp=(const int2*)tab.data_ptr<int>();auto op=(float*)out.data_ptr();\n @DISPATCH@\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
_IV_R24_CONFIGS=[(16384,2,512,4,8),(16384,2,512,8,9),(16384,2,512,8,10),(16384,1,512,2,7)]
_iv_r24_ext=None

def _iv_r24_build():
 global _iv_r24_ext
 if _iv_r24_ext is None:
  lines=[]
  for i,(k,r,t,a,s) in enumerate(_IV_R24_CONFIGS):
   lines.append(f'if(cfg=={i}) cgemv_kernel<{r},{t},{k},{a},{s}><<<N/{r},{t}+32,0,st>>>(lp,cp,xp,tp,pmax,op,b4);')
  _iv_r24_ext=load_inline(name='inv_proj_router24_candidate_v004',cpp_sources='#include <torch/extension.h>\nvoid invoke(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',cuda_sources=_IV_R24_SRC.replace('@DISPATCH@','\n'.join(lines)),functions=['invoke'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],no_implicit_headers=True)
 return _iv_r24_ext

def _iv_r24_run(c,x,n,i):
 y=torch.empty((1,n),device=x.device,dtype=torch.float32)
 _iv_r24_build().invoke(c['L'],c['C'],x,c['tab'],y,c['pmax'],c['base4'],i)
 return y



_iv_base_prepare=prepare
_iv_base_kernel=kernel_fn

def _iv_choose(funcs,x,w,dev,tol):
    truth=x.double()@w.double().T
    usable=[]
    for j,fn in enumerate(funcs):
        out=fn()
        err=((out.double()-truth).norm()/truth.norm()).item()
        if err<tol:usable.append(j)
    if not usable:return 0
    graphs={}
    for j in usable:
        gr=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):funcs[j]()
        graphs[j]=gr
    flush=torch.ones((65536,1024),device=dev,dtype=torch.float32)
    sink=torch.empty(1024,device=dev,dtype=torch.float32)
    events=[]
    for rep in range(96):
        order=usable if rep%2==0 else list(reversed(usable))
        for j in order:
            torch.sum(flush,dim=0,out=sink)
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[j].replay();b.record();events.append((j,a,b))
    torch.cuda.synchronize()
    times={j:[] for j in usable}
    for j,a,b in events:times[j].append(a.elapsed_time(b))
    medians={j:sorted(ts)[len(ts)//2] for j,ts in times.items()}
    return min(medians,key=medians.get)

def prepare(weights):
    state=_iv_base_prepare(weights)
    _iv_cg_build()
    router=weights['router'];dev=router.device
    hi=router.to(torch.bfloat16);lo=(router-hi.float()).to(torch.bfloat16)
    variants=[]
    for i,(_,rows,_,_,shift) in enumerate(_IV_R24_CONFIGS):
        scale=2**shift
        scaled=(lo.float()*scale).to(torch.bfloat16)
        if not bool((scaled.float()/scale==lo.float()).all()):continue
        c=cg_compress(torch.cat([hi,scaled],dim=1).contiguous(),rows)
        if c is not None:variants.append((i,c))
    gen=torch.Generator(device=dev).manual_seed(31919)
    if variants:
        _iv_r24_build()
        x=torch.randn((1,8192),generator=gen,device=dev,dtype=torch.bfloat16)
        data={'x':x,'config':{'m':1,'n':256,'k':8192,'layer':'router'}}
        funcs=[lambda:_iv_base_kernel(data,state)]
        funcs.extend(lambda i=i,c=c:_iv_r24_run(c,x,256,i) for i,c in variants)
        best=_iv_choose(funcs,x,router,dev,1e-4)
        if best:state['iv_router24']=variants[best-1]
    state['iv_cg_choice']={}
    for layer,indices in [('qkv_gate',(0,1)),('unpooler',(2,3))]:
        c=state.get('cg',{}).get(layer)
        if c is None:continue
        w=weights[layer][:1024] if layer=='unpooler' else weights[layer]
        n,k=w.shape
        x=torch.randn((1,k),generator=gen,device=dev,dtype=torch.bfloat16)
        data={'x':x,'config':{'m':1,'n':n,'k':k,'layer':layer}}
        funcs=[lambda:_iv_base_kernel(data,state)]
        funcs.extend(lambda i=i:_iv_cg_run(c,x,n,i) for i in indices)
        best=_iv_choose(funcs,x,w,dev,1e-2)
        if best:state['iv_cg_choice'][layer]=indices[best-1]
    return state

def kernel_fn(data,state):
    layer=data['config']['layer'];m=data['x'].shape[0]
    if m==1:
        if layer=='router' and 'iv_router24' in state:
            i,c=state['iv_router24']
            return _iv_r24_run(c,data['x'],data['config']['n'],i)
        if layer in state.get('iv_cg_choice',{}):
            return _iv_cg_run(state['cg'][layer],data['x'],data['config']['n'],state['iv_cg_choice'][layer])
    return _iv_base_kernel(data,state)


import hashlib as _oc_hashlib


# _oc_zsrc: removed (plain-source rule)


_OC_OLAYER_CUDA = '\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define OD __device__ __forceinline__\nOD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\nOD void ol_wait(uint32_t b,int ph) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");\n}\nOD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nOD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nOD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {\n  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");\n}\n@@MMA@@\n\ntemplate<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(\n    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,\n    const uint8_t* w,__nv_bfloat16* out,int m) {\n  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;\n  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);\n  extern __shared__ __align__(1024) uint8_t sm[];\n  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;\n  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;\n  uint64_t pw,px;\n  if constexpr(HINT) {\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));\n    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));\n  } else {\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;\n  }\n  auto load=[&](int i) {\n    int s=i%ST;\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n    if constexpr(PACK==1) {\n      const uint8_t* src=w+((long)nt*IT+i)*AB;\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");\n    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);\n    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);\n    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);\n  };\n  if(tid==((PIPE&2)?CT:0)) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    #pragma unroll\n    for(int i=0;i<ST && i<IT;i++) load(i);\n  }\n  __syncthreads();\n  if((PIPE&2) && tid>=CT) {\n    if(tid==CT) {\n      #pragma unroll 1\n      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n    }\n  } else {\n    float a[BN/2];\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");\n    #pragma unroll 1\n    for(int i=0;i<IT;i++) {\n      int s=i%ST;\n      ol_wait(full+s*8,(i/ST)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n      #pragma unroll\n      for(int kk=0;kk<BK/16;kk++) {\n        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);\n        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);\n        @@CALL@@\n      }\n      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\n      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\n      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n      int done=i-((PIPE&1)?1:0);\n      if(done>=0 && done+ST<IT && tid%128==0) {\n        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);\n        else load(done+ST);\n      }\n    }\n    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");\n    #pragma unroll\n    for(int j=0;j<BN/8;j++) {\n      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);\n      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);\n      uint32_t d=base+2*(cb*72+rb);\n      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");\n    }\n  }\n  __syncthreads();\n  for(int q=tid;q<T*(R/8);q+=TH) {\n    int token=q/(R/8),row=(q%(R/8))*8;\n    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);\n  }\n}\n\nusing OLEncode=decltype(&cuTensorMapEncodeTiled);\nint ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {\n  static OLEncode encode=nullptr;\n  if(!encode) {\n    cudaDriverEntryPointQueryResult q;\n    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);\n    if(rc!=cudaSuccess || !encode) return -100;\n  }\n  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};\n  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\ntemplate<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\nint ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n  constexpr int T=BN*TW,R=PACK==2?32:64;\n  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;\n  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;\n  static bool configured=false;\n  if(!configured) {\n    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;\n    configured=true;\n  }\n  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}\n  CUtensorMap tw,tx;\n  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;\n  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;\n  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);\n  return int(cudaGetLastError());\n}\nint ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n  switch(id) {@@CASES@@ default:return -99;}\n}\n'
_OC_RL_CUDA = '\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define OD __device__ __forceinline__\nOD uint32_t rl_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\nOD void rl_wait(uint32_t b,int ph) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");\n}\nOD void rl_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nOD uint64_t rl_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nOD void rl_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {\n  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");\n}\n@@MMA@@\n\ntemplate<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void rl_early_projection(\n    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,\n    const uint8_t* w,__nv_bfloat16* out,int m) {\n  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;\n  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);\n  extern __shared__ __align__(1024) uint8_t sm[];\n  uint32_t base=rl_addr(sm),full=base+ST*SB,empty=full+ST*8;\n  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;\n  uint64_t pw,px;\n  if constexpr(HINT) {\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));\n    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));\n  } else {\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;\n  }\n  auto load=[&](int i) {\n    int s=i%ST;\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n    if constexpr(PACK==1) {\n      const uint8_t* src=w+((long)nt*IT+i)*AB;\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");\n    } else if constexpr(PACK==2) rl_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);\n    else rl_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);\n    rl_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);\n  };\n  if(tid==((PIPE&2)?CT:0)) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  if constexpr(PIPE&4) __syncthreads();\n  if(tid==((PIPE&2)?CT:0)) {\n    #pragma unroll\n    for(int i=0;i<ST && i<IT;i++) load(i);\n  }\n  if constexpr(!(PIPE&4)) __syncthreads();\n  if((PIPE&2) && tid>=CT) {\n    if(tid==CT) {\n      #pragma unroll 1\n      for(int i=ST;i<IT;i++) {rl_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n    }\n  } else {\n    float a[BN/2];\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");\n    #pragma unroll 1\n    for(int i=0;i<IT;i++) {\n      int s=i%ST;\n      rl_wait(full+s*8,(i/ST)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n      #pragma unroll\n      for(int kk=0;kk<BK/16;kk++) {\n        uint64_t aw=rl_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);\n        uint64_t ax=rl_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);\n        @@CALL@@\n      }\n      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\n      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\n      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n      #pragma unroll\n      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n      int done=i-((PIPE&1)?1:0);\n      if(done>=0 && done+ST<IT && tid%128==0) {\n        if constexpr(PIPE&2) rl_arrive(empty+(done%ST)*8);\n        else load(done+ST);\n      }\n    }\n    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n    #pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");\n    #pragma unroll\n    for(int j=0;j<BN/8;j++) {\n      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);\n      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);\n      uint32_t d=base+2*(cb*72+rb);\n      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");\n    }\n  }\n  __syncthreads();\n  for(int q=tid;q<T*(R/8);q+=TH) {\n    int token=q/(R/8),row=(q%(R/8))*8;\n    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);\n  }\n}\n\nusing OLEncode=decltype(&cuTensorMapEncodeTiled);\nint rl_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {\n  static OLEncode encode=nullptr;\n  if(!encode) {\n    cudaDriverEntryPointQueryResult q;\n    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);\n    if(rc!=cudaSuccess || !encode) return -100;\n  }\n  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};\n  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\ntemplate<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\nint rl_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n  constexpr int T=BN*TW,R=PACK==2?32:64;\n  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;\n  auto fn=rl_early_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;\n  static bool configured=false;\n  if(!configured) {\n    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;\n    configured=true;\n  }\n  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}\n  CUtensorMap tw,tx;\n  int rc=rl_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;\n  rc=rl_map(&tx,(void*)x,m,T,BK);if(rc)return rc;\n  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);\n  return int(cudaGetLastError());\n}\nint rl_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n  switch(id) {@@CASES@@ default:return -99;}\n}\n'
_OC_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'
_OC_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def _oc_olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join(f'%{i}' for i in range(n // 2))
        outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
        }}''')
    cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
    calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
    return _OC_OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)


def _oc_build_olayer(configs):
    for cfg in configs:
        bn,bk,st,pipe,pack,hint,*groups = cfg
        tw = groups[0] if groups else 1
        if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = _oc_olayer_source(configs)
    name = 'olayer_' + _oc_hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name,_OC_OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True,with_pytorch_error_handling=False)


def _oc_pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
    tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
    return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()




def _oc_rl_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
    return _OC_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def _oc_build_rl(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported rl configuration: {cfg}')
    tt_v10_tune_source = _oc_rl_source(configs)
    name = 'rl_' + _oc_hashlib.sha256(tt_v10_tune_source.encode()).hexdigest()[:12]
    return load_inline(name, _OC_RL_CPP, cuda_sources=tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)



def _oc_rl_pack(w: torch.Tensor, rows: int) -> torch.Tensor:
    n, k = w.shape
    col = torch.arange(64, device=w.device)[None, :] ^ (torch.arange(rows, device=w.device)[:, None] & 7) * 8
    return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()



_OC_OL_CONFIG = (128, 128, 4, 3, 1, 1)
_OC_RL_CONFIG = (256, 64, 5, 3, 1, 1, 1)
_oc_parent_prepare = prepare
_oc_parent_kernel = kernel_fn


def prepare(weights):
    state = _oc_parent_prepare(weights)
    state["oc_ol"] = {"ext": _oc_build_olayer([_OC_OL_CONFIG]), "weight": _oc_pack_olayer(state["raw"]["o"])}
    state["oc_rl"] = {"ext": _oc_build_rl([_OC_RL_CONFIG]), "weight": _oc_rl_pack(state["raw"]["o"], 64)}
    return state


def kernel_fn(data, state):
    x = data["x"]
    if data["config"]["layer"] == "o" and x.is_contiguous():
        m = x.shape[0]
        if m == 128:
            out = torch.empty((128, 8192), device=x.device, dtype=torch.bfloat16)
            rc = state["oc_ol"]["ext"].ol_launch(0, x.data_ptr(), state["oc_ol"]["weight"].data_ptr(),
                                                out.data_ptr(), 128, torch.cuda.current_stream().cuda_stream, 0)
            if rc:
                raise RuntimeError(f"custom olayer launch failed: {rc}")
            return out
        if m == 256:
            out = torch.empty((256, 8192), device=x.device, dtype=torch.bfloat16)
            rc = state["oc_rl"]["ext"].rl_launch(0, x.data_ptr(), state["oc_rl"]["weight"].data_ptr(),
                                                out.data_ptr(), 256, torch.cuda.current_stream().cuda_stream, 0)
            if rc:
                raise RuntimeError(f"custom rl launch failed: {rc}")
            return out
    return _oc_parent_kernel(data, state)


import types as _KTypes


class _KSharedModule:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __deepcopy__(self, memo):
        return self


def _kshare_modules(value):
    if isinstance(value, _KTypes.ModuleType):
        return _KSharedModule(value)
    if isinstance(value, dict):
        return {key: _kshare_modules(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_kshare_modules(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_kshare_modules(item) for item in value)
    return value


_oc_shared_prepare = prepare


def prepare(weights):
    return _kshare_modules(_oc_shared_prepare(weights))


_U_CFG=('u32_r96', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1)
_U_NATIVE=None
_U_RUN=_run_sk2
_U_PREPARE=prepare

def _u_ext():
 global _U_NATIVE
 if _U_NATIVE is None:
  cfg=_U_CFG;name,layer,n,R,NT,NWR,WSK,UPW,NACC,CL=cfg
  _,ns=_sk2_geom(cfg,32)
  call=f'launch_sk2<2,32,{NT},{NWR},{WSK},{UPW},{ns},true,false,{NACC},{CL},__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;'
  src=CUDA_SRC.replace('@@MBAR@@',MBAR_HELPERS).replace('@@BODY@@','')+MAP_SRC+SK2_SRC.replace('@@SK2_DISPATCH@@',call)
  import hashlib
  cpp='#include <torch/extension.h>\n'+next(l for l in CPP_SRC.splitlines() if l.startswith('void sk2('))+'\n'+next(l for l in CPP_SRC.splitlines() if 'int64_t M, int64_t N, int64_t K, int64_t cfg' in l)
  _U_NATIVE=load_inline('proj_u32geom_'+hashlib.md5(src.encode()).hexdigest()[:12],cpp,cuda_sources=src,functions=['sk2'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90','-Xptxas=-v'],verbose=False)
 return _U_NATIVE

def prepare(weights):
 _u_ext()
 state=_U_PREPARE(weights)
 state['u32']=_u_pack(weights['unpooler'])
 return state

def _u_pack(w):
 name,layer,n,R,NT,NWR,WSK,UPW,NACC,CL=_U_CFG
 c=sk2_compress(w)
 if c is None:return None
 tab,pmax=sk2_table(c,n,R)
 return dict(w=sk2_stream(c['packed'],n,R,NT,NWR,WSK,UPW),tab=tab,pmax=pmax,b4=c['base4'])

def _run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='unpooler' or m!=32 or state.get('u32') is None:return _U_RUN(x,layer,m,n,k,state,dg,dbg)
 c=state['u32'];out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 _u_ext().sk2(c['w'],x,out,c['tab'],c['pmax'],c['b4'],m,n,k,0,32,dg,state['dummy'] if dbg is None else dbg)
 return out


_ship_parent_kernel = kernel_fn


def kernel_fn(data, state):
    if data["config"]["layer"] == "router" and data["x"].shape[0] == 1:
        return _run(("rgemv",), data["x"], "router", 1, data["config"]["n"], data["config"]["k"], state)
    return _ship_parent_kernel(data, state)


# Ensure driver API calls on every route have a current CUDA context in a fresh thread.
# The helper is credited to matvey; it is cached per thread/device.
_ship_context_parent_kernel = kernel_fn
def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    return _ship_context_parent_kernel(data, state)


# Nathen Fernandes (MIT): literal pinned router routes at m <= 32.
import hashlib


# _zsrc: removed (plain-source rule)


def _mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


ROUTER_CUDA = '\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#define RD __device__ __forceinline__\nRD uint32_t rr_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\nRD void rr_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }\nRD void rr_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }\nRD void rr_wait(uint32_t b, int ph) {\n  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");\n}\nRD void rr_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }\nRD uint32_t rr_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }\nRD void rr_push(uint32_t d, float4 v, uint32_t b) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");\n}\nRD float4 rr_pull(uint32_t d) {\n  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;\n}\nRD void rr_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }\nRD uint64_t rr_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }\nRD void rr_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");\n}\n@@MMA@@\n\ntemplate<int BN,int S,int ST,int SW,int RED,int PIPE>\n__global__ __launch_bounds__((PIPE&2)?256:128) void rr_kernel(const __grid_constant__ CUtensorMap tw,\n                                               const __grid_constant__ CUtensorMap tx, float* out, int m) {\n  constexpr int NR=SW?64:BN, TM=SW?BN:64;\n  constexpr int WB=NR*128, XB=TM*128, SB=WB*2+XB;\n  constexpr int NF=64*BN, NV=NF/4, OWN=NV/S;\n  extern __shared__ __align__(1024) unsigned char sm[];\n  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;\n  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;\n  const uint32_t base=rr_smem(sm), bars=base+ST*SB, empty=bars+ST*8, rb=(empty+((PIPE&2)?ST*8:0)+15)&~15u;\n  const uint32_t recv=rb+16;\n  const int kb=rank*(128/S);\n  auto load=[&](int i) {\n    const int st=i%ST;\n    const uint32_t b=bars+st*8, d=base+st*SB;\n    rr_expect(b,SB);\n    rr_tma(d,&tw,0,((row/NR)*128+kb+i)*NR*2,b);\n    rr_tma(d+2*WB,&tx,(kb+i)*64,tok,b);\n  };\n  if(tid==((PIPE&2)?128:0)) {\n#pragma unroll\n    for(int i=0;i<ST;i++) { rr_init(bars+i*8,1); if constexpr(PIPE&2) rr_init(empty+i*8,4); }\n    asm volatile("fence.proxy.async.shared::cta;" :::"memory");\n    if constexpr(PIPE&4) {\n    if constexpr(RED>=1) {\n      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);\n      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");\n    }\n    }\n#pragma unroll\n    for(int i=0;i<ST && i<128/S;i++) load(i);\n    if constexpr(!(PIPE&4)) {\n    if constexpr(RED>=1) {\n      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);\n      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");\n    }\n    }\n  }\n  __syncthreads();\n  if constexpr(RED>=1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");\n  if constexpr(PIPE&2) {\n    if(tid>=128) {\n      if(tid==128) {\n        for(int i=ST;i<128/S;i++) {\n          rr_wait(empty+(i%ST)*8,((i/ST)-1)&1);\n          load(i);\n        }\n      }\n      if constexpr(RED>=1) asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n      if constexpr(RED!=2) rr_sync();\n      return;\n    }\n  }\n  float acc[BN/2];\n#pragma unroll\n  for(int j=0;j<BN/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(acc[j]));\n  int prev=0;\n#pragma unroll 1\n  for(int i=0;i<128/S;i++) {\n    const int st=i%ST;\n    rr_wait(bars+st*8,(i/ST)&1);\n    const uint32_t w=base+st*SB, x=w+2*WB;\n#pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");\n    asm volatile("wgmma.fence.sync.aligned;" :::"memory");\n#pragma unroll\n    for(int q=0;q<4;q++) {\n      const uint64_t dx=rr_desc(x+q*32),dh=rr_desc(w+q*32),dl=rr_desc(w+WB+q*32);\n      if constexpr(SW) { @@CALL_H@@; @@CALL_L@@; }\n      else { @@CALL_NH@@; @@CALL_NL@@; }\n    }\n    asm volatile("wgmma.commit_group.sync.aligned;" :::"memory");\n    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" :::"memory");\n    else asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");\n#pragma unroll\n    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");\n    if constexpr(PIPE&2) {\n      if constexpr(PIPE&1) {\n        if(i>0 && prev+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+(prev%ST)*8):"memory");\n      } else {\n        if(i+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+st*8):"memory");\n      }\n    } else if constexpr(PIPE&1) {\n      if(i>0 && prev+ST<128/S && tid==0) load(prev+ST);\n    } else {\n      if(i+ST<128/S && tid==0) load(i+ST);\n    }\n    prev=i;\n  }\n  asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");\n  auto emit=[&](int v,float4 z) {\n    const int t=v%128, j=v/128, ln=t&31, wr=t>>5;\n    const int ar=wr*16+(ln>>2),bc=j*8+(ln&3)*2;\n    const float vv[4]={z.x,z.y,z.z,z.w};\n#pragma unroll\n    for(int c=0;c<4;c++) {\n      const int a=ar+(c/2)*8, b=bc+(c&1);\n      const int tr=tok+(SW?b:a), wc=row+(SW?a:b);\n      if(tr<m) out[tr*256+wc]=vv[c];\n    }\n  };\n  if constexpr(S==1) {\n#pragma unroll\n    for(int j=0;j<BN/8;j++) emit(j*128+tid,make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]));\n  } else if constexpr(RED==0) {\n    float4* part=(float4*)sm;\n#pragma unroll\n    for(int j=0;j<BN/8;j++) part[j*128+tid]=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);\n    rr_sync();\n    for(int t=tid;t<OWN;t+=128) {\n      const int v=rank*OWN+t;\n      float4 sum=make_float4(0,0,0,0);\n#pragma unroll\n      for(int r=0;r<S;r++) rr_add(sum,rr_pull(rr_map(base+v*16,r)));\n      emit(v,sum);\n    }\n    rr_sync();\n  } else {\n    asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n#pragma unroll\n    for(int j=0;j<BN/8;j++) {\n      const int v=j*128+tid, dst=v/OWN, local=v%OWN;\n      const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);\n      const uint32_t addr=recv+(rank*OWN+local)*16;\n      if(dst==rank) *(float4*)(sm+(addr-base))=val;\n      else rr_push(rr_map(addr,dst),val,rr_map(rb,dst));\n    }\n    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;" :::"memory"); else __syncthreads();\n    rr_wait(rb,0);\n    for(int t=tid;t<OWN;t+=128) {\n      float4 sum=make_float4(0,0,0,0);\n#pragma unroll\n      for(int r=0;r<S;r++) rr_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));\n      emit(rank*OWN+t,sum);\n    }\n    if constexpr(RED!=2) rr_sync();\n  }\n}\n\ntypedef CUresult (*RR_Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,\n const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);\nstatic int rr_map_host(CUtensorMap* tm,int64_t p,int inner,int outer,int box) {\n  static RR_Encode enc=nullptr;\n  if(!enc) { cudaDriverEntryPointQueryResult q; if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&enc,12000,cudaEnableDefault,&q)!=cudaSuccess||!enc) return -100; }\n  cuuint64_t dims[2]={(uint64_t)inner,(uint64_t)outer},stride[1]={(uint64_t)inner*2};\n  cuuint32_t sz[2]={64,(uint32_t)box},es[2]={1,1};\n  return (int)enc(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,sz,es,CU_TENSOR_MAP_INTERLEAVE_NONE,\n                 CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n}\n\ntemplate<int BN,int S,int ST,int SW,int RED,int PIPE>\nint rr_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n  constexpr int NR=SW?64:BN, TM=SW?BN:64;\n  constexpr int bytes=ST*(2*NR+TM)*128+ST*8*((PIPE&2)?2:1)+48+(RED>=1?64*BN*4:0);\n  auto fn=rr_kernel<BN,S,ST,SW,RED,PIPE>;\n  static bool configured=false;\n  if(!configured) {\n    cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);\n    if(e!=cudaSuccess) return -(int)e;\n    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);\n    if(e!=cudaSuccess)return -(int)e;\n    configured=true;\n  }\n  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n  cudaLaunchConfig_t cfg={};\n  cfg.gridDim=dim3(256/NR*S,(m+TM-1)/TM,1);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n  cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={S,1,1};\n  if(S>1){cfg.attrs=attr;cfg.numAttrs=1;}\n  if(query==1){int n=0;cudaError_t e=cudaOccupancyMaxActiveClusters(&n,fn,&cfg);return e==cudaSuccess?n:-(int)e;}\n  CUtensorMap tw,tx;\n  int e=rr_map_host(&tw,w,64,256*128*2,NR*2);if(e)return e;\n  e=rr_map_host(&tx,x,8192,m,TM);if(e)return e;\n  cudaError_t err=cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m);\n  return err==cudaSuccess?0:(int)err;\n}\nint rr_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n switch(id){@@CASES@@ default:return -99;}\n}\n@@SCALAR@@\n'

ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def build_router(configs):
    src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join(_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rr_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
    name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch','rr_scalar','rr_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rr_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rr_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]


def prepare_router(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = prepare_candidates(weights, ROUTER_CONFIGS)
    st['small'] = {(16,8):pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return run_candidate(x,st,-1)
    if m <= 16:
        return run_small_candidate(x,st)
    return run_candidate(x,st,0 if m <= 32 else 1 if m <= 128 else 2)


ROUTER_SCALAR = '\ntemplate<int M, int NT, int ST>\n__global__ void rr_scalar_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n                                float* __restrict__ y, int m) {\n  constexpr int K=8192;\n  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n  float acc[M][4]={};\n  float4 ww[ST];\n  uint2 xx[ST][M];\n  auto ld=[&](int slot,int i) {\n    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n    #pragma unroll\n    for(int t=0;t<M;t++) {\n      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n  #pragma unroll 1\n  for(int base=tid;base<K/4;base+=ST*NT) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      float4 v=ww[s];\n      uint2 a[M];\n      #pragma unroll\n      for(int t=0;t<M;t++) a[t]=xx[s][t];\n      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n      #pragma unroll\n      for(int t=0;t<M;t++) if(t<m) {\n        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n      }\n    }\n  }\n  __shared__ float sums[M][NT/32];\n  #pragma unroll\n  for(int t=0;t<M;t++) if(t<m) {\n    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n    #pragma unroll\n    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n    if(lane==0) sums[t][warp]=v;\n  }\n  __syncthreads();\n  if(warp==0) {\n    #pragma unroll\n    for(int t=0;t<M;t++) if(t<m) {\n      float v=lane<NT/32?sums[t][lane]:0.f;\n      #pragma unroll\n      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n      if(lane==0) y[t*256+row]=v;\n    }\n  }\n}\n\nint rr_scalar(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n  auto kern=rr_scalar_kernel<1,512,2>;\n  if(query==2) {\n    cudaFuncAttributes fa;\n    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n  }\n  if(m!=1) return -5;\n  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n  return cudaGetLastError()==cudaSuccess?0:-8;\n}\n'

ROUTER_SMALL = '\ntemplate<int MT,int S,int WK>\n__global__ __launch_bounds__(WK*32) void rr_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {\n  constexpr int KU=128/S/WK, NI=MT*32;\n  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;\n  const int gw=(rank*16+row/16)*WK+warp;\n  extern __shared__ __align__(16) unsigned char sm[];\n  float4* recv=(float4*)sm;\n  float4* part=recv+S*NI;\n  const uint32_t rb=rr_smem(sm+(S+WK)*NI*16);\n  int4 w[KU][8];uint4 xx[KU][MT][2];\n#pragma unroll\n  for(int k=0;k<KU;k++) {\n#pragma unroll\n    for(int j=0;j<8;j++) {\n      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;\n      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));\n    }\n#pragma unroll\n    for(int n=0;n<MT;n++) {\n      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;\n#pragma unroll\n      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));\n    }\n  }\n  if(rank==0 && tid==0) {rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}\n  if(rank==0 && warp==0) __syncwarp();\n  if(rank==0 && tid==0) {rr_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}\n  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");\n  float c[2][MT][4]={};\n#pragma unroll\n  for(int k=0;k<KU;k++) {\n#pragma unroll\n    for(int n=0;n<MT;n++) {\n      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};\n#pragma unroll\n      for(int q=0;q<4;q++) {\n#pragma unroll\n        for(int h=0;h<2;h++) {\n          const int4 a=w[k][q+h*4];\n          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));\n        }\n      }\n    }\n  }\n#pragma unroll\n  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);\n  __syncthreads();\n  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n  for(int v=tid;v<NI;v+=WK*32) {\n    float4 sum=make_float4(0,0,0,0);\n#pragma unroll\n    for(int k=0;k<WK;k++) rr_add(sum,part[k*NI+v]);\n    if(rank==0)recv[v]=sum;\n    else rr_push(rr_map(rr_smem(recv+rank*NI+v),0),sum,rr_map(rb,0));\n  }\n  if(rank==0) {\n    __syncthreads();rr_wait(rb,0);\n    for(int v=tid;v<NI;v+=WK*32) {\n      float4 sum=make_float4(0,0,0,0);\n#pragma unroll\n      for(int s=0;s<S;s++) rr_add(sum,recv[s*NI+v]);\n      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);\n      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}\n      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}\n    }\n  }\n}\n\ntemplate<int MT,int S,int WK>\nint rr_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n  auto fn=rr_small_kernel<MT,S,WK>;\n  constexpr int bytes=(S+WK)*MT*512+16;\n  static bool configured=false;\n  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}\n  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;\n  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);\n}\nint rr_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {\n#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return rr_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return rr_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return rr_small_t<4,S,W>(w,x,out,m,stream,query);}\nRSM(8,8) RSM(16,8) RSM(16,4)\nreturn -99;\n}\n\n'


def pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rr_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out


class _NRShared(dict):
    def __deepcopy__(self, memo):
        import copy as _nr_copy
        out = _NRShared()
        memo[id(self)] = out
        for key, value in self.items():
            out[key] = value if key == 'ext' else _nr_copy.deepcopy(value, memo)
        return out


_nr_parent_prepare = prepare
_nr_parent_kernel = kernel_fn


def prepare(weights):
    return _nr_parent_prepare(weights)


def kernel_fn(data, state):
    return _nr_parent_kernel(data, state)


V18S_CUDA_TEMPLATE = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define SM3D __device__ __forceinline__\n\nSM3D int4 v18s_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSM3D uint4 v18s_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSM3D uint32_t v18s_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nSM3D uint32_t v18s_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nSM3D void v18s_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nSM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = v18s_prmt(T0, T1, cm);\n  const uint32_t hhi = v18s_prmt(T0, T1, cm >> 16);\n  a[0] = v18s_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = v18s_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = v18s_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = v18s_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\n__device__ __noinline__ void v18s_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nSM3D void v18s_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) v18s_never(p);\n}\n\nSM3D uint32_t v18s_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nSM3D uint32_t v18s_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nSM3D uint32_t v18s_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nSM3D void v18s_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nSM3D void v18s_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nSM3D void v18s_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nSM3D void v18s_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nSM3D void v18s_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nSM3D void v18s_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nSM3D bool v18s_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nSM3D uint64_t v18s_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a wrong result instead of a hang.\nSM3D void v18s_wait_bar(uint32_t bar) {\n  if (v18s_mbar_try(bar)) return;\n  const uint64_t t0 = v18s_gtime();\n  while (!v18s_mbar_try(bar))\n    if (v18s_gtime() - t0 > 20000000ull) __trap();\n}\n\nSM3D void v18s_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nSM3D float4 v18s_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nSM3D void v18s_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nSM3D void v18s_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct Sm3Fmt;\ntemplate <> struct Sm3Fmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct Sm3Fmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct Sm3Fmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct Sm3Fmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nSM3D void v18s_decode(const int4 (&w)[Sm3Fmt<FMT>::NL], uint32_t (&A)[Sm3Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < Sm3Fmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < Sm3Fmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      v18s_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      v18s_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      v18s_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      v18s_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nSM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nSM3D void v18s_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) v18s_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n__global__ void __launch_bounds__(NTMAX) v18s_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = Sm3Fmt<FMT>::NL, NA = Sm3Fmt<FMT>::NA;\n  constexpr bool C12 = FMT == 1 || FMT == 3;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);\n    const uint32_t bar = v18s_smem_u32(&s_bar);\n    uint32_t rank = 0, my_e = 0xFFFFFFFFu;\n    if constexpr (!(OPT & 1)) {\n    if constexpr (PUSH) {\n      rank = v18s_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        v18s_mbar_init(bar, 1);\n        v18s_mbar_expect(bar, parts * RT * TV * 128u);\n        v18s_fence_init();\n      }\n      v18s_cl_arrive();\n    }\n\n    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);\n\n    }\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO && !(OPT & 2)) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = v18s_ldx(xp);\n          xv[slot][n][1] = v18s_ldx(xp + 8);\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    if constexpr (!(OPT & 4)) v18s_order_point(m, outp);\n    if constexpr (OPT & 1) {\n    if constexpr (PUSH) {\n      rank = v18s_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        v18s_mbar_init(bar, 1);\n        v18s_mbar_expect(bar, parts * RT * TV * 128u);\n        v18s_fence_init();\n      }\n      v18s_cl_arrive();\n    }\n\n    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);\n\n    }\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);\n        } else {\n          uint32_t b[MT][8];\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else if constexpr (OPT & 2) {\n              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;\n              uint4 u = v18s_ldx(xp), v = v18s_ldx(xp + 8);\n              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;\n              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;\n            } else {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) v18s_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          if (si + ST < KU) load(s, si + ST);\n          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n                } else {\n                  v18s_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                  if constexpr (HILO) v18s_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                }\n              }\n        }\n      }\n    }\n\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          v18s_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) v18s_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                v18s_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          if constexpr (OPT & 8) { emit_full(idx, v); return; }\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem);\n          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)\n              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });\n          } else {\n            float4* csum = part + (size_t)NW * RT * MT * 32;\n            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) v18s_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2) {\n            v18s_cl_wait();\n            if (t < TV) {\n              const uint32_t rbar = v18s_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = n * 4 + t;\n                  if (tn < TV) {\n                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            v18s_wait_bar(bar);\n            v18s_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if (n * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            v18s_cl_wait();\n            const uint32_t rbar = v18s_mapa(bar, 0);\n            v18s_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            v18s_wait_bar(bar);\n            v18s_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int v18s_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\nstatic int v18s_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = v18s_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = v18s_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[1];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    lc.numAttrs = 1;\n  }\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\n@@CASES@@\n    default: return -99;\n  }\n}\n\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\ntemplate<int M, int NT, int ST>\n__global__ void v18s_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n                                float* __restrict__ y, int m) {\n  constexpr int K=8192;\n  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n  float acc[M][4]={};\n  float4 ww[ST];\n  uint2 xx[ST][M];\n  auto ld=[&](int slot,int i) {\n    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n    #pragma unroll\n    for(int t=0;t<M;t++) {\n      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n  #pragma unroll 1\n  for(int base=tid;base<K/4;base+=ST*NT) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      float4 v=ww[s];\n      uint2 a[M];\n      #pragma unroll\n      for(int t=0;t<M;t++) a[t]=xx[s][t];\n      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n      #pragma unroll\n      for(int t=0;t<M;t++) if(t<m) {\n        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n      }\n    }\n  }\n  __shared__ float sums[M][NT/32];\n  #pragma unroll\n  for(int t=0;t<M;t++) if(t<m) {\n    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n    #pragma unroll\n    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n    if(lane==0) sums[t][warp]=v;\n  }\n  __syncthreads();\n  if(warp==0) {\n    #pragma unroll\n    for(int t=0;t<M;t++) if(t<m) {\n      float v=lane<NT/32?sums[t][lane]:0.f;\n      #pragma unroll\n      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n      if(lane==0) y[t*256+row]=v;\n    }\n  }\n}\n\nint v18s_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n  auto kern=v18s_router_scalar<1,512,2>;\n  if(query==2) {\n    cudaFuncAttributes fa;\n    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n  }\n  if(m!=1) return -5;\n  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n  return cudaGetLastError()==cudaSuccess?0:-8;\n}\n'

V18S_CPP = r"""
int v18s_router_launch(long,long,long,int,long,int);
#include <pybind11/pybind11.h>
#include <stdint.h>
int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
V18S_ESC_CAP = 32


def v18s_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst + (0,) * (15 - len(inst))))
        cases.append(f"    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return V18S_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def v18s_build(instances, name="v18s_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, V18S_CPP + extra_cpp, cuda_sources=v18s_cuda_source(instances) + extra_cuda,
                       functions=["v18s_launch", "v18s_router_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def v18s_units(w):
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def v18s_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def v18s_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def v18s_bf16_bytes(v):
    r, c = v.shape[:2]
    words = v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def v18s_c12_bytes(v):
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = v18s_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def v18s_c12_unpack(data, base, esc):
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def v18s_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def v18s_pack_format(w, fmt):
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [v18s_units(hi), v18s_units(lo)]
    else:
        mats = [v18s_units(w)]
    if fmt in (0, 2):
        data = torch.cat([v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = v18s_c12_bytes(v)
        if not torch.equal(v18s_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(v18s_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def v18s_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def v18s_stream(pk, rt, wr, wk, ku, s):
    data = pk["data"]
    R, C, B = data.shape
    geo = v18s_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, V18S_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > V18S_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = v18s_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    if sp["fmt"] == 4:
        return ext.v18s_router_launch(sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                     torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18s_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


def v18s_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def v18s_bf16_twin(inst):
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def v18s_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = v18s_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = v18s_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = v18s_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = v18s_verify_onehot(ext, inst_id[inst], sp, v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


_VO_CONFIG = ((1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8, 2, 1, 6), (1, 8, 2, 1))
_vo_parent_prepare = prepare
_vo_parent_kernel = kernel_fn


def _vo_prepare_o(weights):
    instances = [_VO_CONFIG[0], v18s_bf16_twin(_VO_CONFIG[0])]
    ext = v18s_build(instances, 'v18_small_selected')
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    plan = v18s_plan(ext, ids, packs, streams, weights['o'], 'o', ('m', 8), _VO_CONFIG[0], _VO_CONFIG[1])
    if plan is None:
        plan = v18s_plan(ext, ids, packs, streams, weights['o'], 'o', ('m', 8), instances[1][0], _VO_CONFIG[1])
    return _NRShared(ext=ext, plan=plan)


def prepare(weights):
    state = _vo_parent_prepare(weights)
    state['v18o'] = _vo_prepare_o(weights)
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    if data['config']['layer'] == 'o' and data['x'].shape[0] == 8:
        plan = state['v18o']['plan']
        x = data['x']
        out = torch.empty((8, 8192), device=x.device, dtype=torch.bfloat16)
        rc = v18s_launch_raw(state['v18o']['ext'], plan['inst_id'], plan, x, out, 8192, plan['cfg'])
        if rc:
            raise RuntimeError(f'v18 selected launch failed: {rc}')
        return out
    return _vo_parent_kernel(data, state)


"""H200 small-m table refinements. Unclaimed shapes return None."""
import torch
from torch.utils.cpp_extension import load_inline
_NP_SMALLM2_CUDA_TEMPLATE = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define SMALLM2D __device__ __forceinline__\n\nSMALLM2D int4 smallm2_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSMALLM2D uint4 smallm2_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSMALLM2D uint32_t smallm2_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nSMALLM2D uint32_t smallm2_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nSMALLM2D void smallm2_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nSMALLM2D void smallm2_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = smallm2_prmt(T0, T1, cm);\n  const uint32_t hhi = smallm2_prmt(T0, T1, cm >> 16);\n  a[0] = smallm2_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = smallm2_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = smallm2_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = smallm2_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\n__device__ __noinline__ void smallm2_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nSMALLM2D void smallm2_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) smallm2_never(p);\n}\n\nSMALLM2D uint32_t smallm2_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nSMALLM2D uint32_t smallm2_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nSMALLM2D uint32_t smallm2_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nSMALLM2D void smallm2_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nSMALLM2D void smallm2_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nSMALLM2D void smallm2_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nSMALLM2D void smallm2_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nSMALLM2D void smallm2_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nSMALLM2D void smallm2_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nSMALLM2D bool smallm2_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nSMALLM2D uint64_t smallm2_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.\nSMALLM2D void smallm2_wait_bar(uint32_t bar) {\n  if (smallm2_mbar_try(bar)) return;\n  const uint64_t t0 = smallm2_gtime();\n  while (!smallm2_mbar_try(bar))\n    if (smallm2_gtime() - t0 > 20000000ull) __trap();\n}\n\nSMALLM2D void smallm2_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nSMALLM2D float4 smallm2_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nSMALLM2D void smallm2_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nSMALLM2D void smallm2_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct Smallm2Fmt;\ntemplate <> struct Smallm2Fmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct Smallm2Fmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct Smallm2Fmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct Smallm2Fmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nSMALLM2D void smallm2_decode(const int4 (&w)[Smallm2Fmt<FMT>::NL], uint32_t (&A)[Smallm2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < Smallm2Fmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < Smallm2Fmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      smallm2_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      smallm2_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      smallm2_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      smallm2_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nSMALLM2D void smallm2_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nSMALLM2D void smallm2_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) smallm2_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n__global__ void __launch_bounds__(NTMAX) smallm2_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = Smallm2Fmt<FMT>::NL, NA = Smallm2Fmt<FMT>::NA;\n  constexpr bool C12 = FMT == 1 || FMT == 3;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);\n    const uint32_t bar = smallm2_smem_u32(&s_bar);\n    uint32_t rank = 0, my_e = 0xFFFFFFFFu;\n    if constexpr (!(OPT & 1)) {\n    if constexpr (PUSH) {\n      rank = smallm2_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        smallm2_mbar_init(bar, 1);\n        smallm2_mbar_expect(bar, parts * RT * TV * 128u);\n        smallm2_fence_init();\n      }\n      smallm2_cl_arrive();\n    }\n\n    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);\n\n    }\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) w[slot][j][l] = smallm2_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO && !(OPT & 2)) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = smallm2_ldx(xp);\n          xv[slot][n][1] = smallm2_ldx(xp + 8);\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n    if constexpr (OPT & 1) {\n    if constexpr (PUSH) {\n      rank = smallm2_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        smallm2_mbar_init(bar, 1);\n        smallm2_mbar_expect(bar, parts * RT * TV * 128u);\n        smallm2_fence_init();\n      }\n      smallm2_cl_arrive();\n    }\n\n    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);\n\n    }\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n        } else {\n          uint32_t b[MT][8];\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else if constexpr (OPT & 2) {\n              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;\n              uint4 u = smallm2_ldx(xp), v = smallm2_ldx(xp + 8);\n              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;\n              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;\n            } else {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) smallm2_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          if (si + ST < KU) load(s, si + ST);\n          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) smallm2_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n                } else {\n                  smallm2_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                  if constexpr (HILO) smallm2_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                }\n              }\n        }\n      }\n    }\n\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          smallm2_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) smallm2_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                smallm2_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          if constexpr (OPT & 8) { emit_full(idx, v); return; }\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem);\n          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)\n              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });\n          } else {\n            float4* csum = part + (size_t)NW * RT * MT * 32;\n            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) smallm2_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2) {\n            smallm2_cl_wait();\n            if (t < TV) {\n              const uint32_t rbar = smallm2_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = n * 4 + t;\n                  if (tn < TV) {\n                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            smallm2_wait_bar(bar);\n            smallm2_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if (n * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            smallm2_cl_wait();\n            const uint32_t rbar = smallm2_mapa(bar, 0);\n            smallm2_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            smallm2_wait_bar(bar);\n            smallm2_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int smallm2_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\nstatic int smallm2_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = smallm2_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = smallm2_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[1];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    lc.numAttrs = 1;\n  }\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\n@@CASES@@\n    default: return -99;\n  }\n}\n\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\ntemplate<int M, int NT, int ST>\n__global__ void smallm2_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n                                float* __restrict__ y, int m) {\n  constexpr int K=8192;\n  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n  float acc[M][4]={};\n  float4 ww[ST];\n  uint2 xx[ST][M];\n  auto ld=[&](int slot,int i) {\n    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n    #pragma unroll\n    for(int t=0;t<M;t++) {\n      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n  #pragma unroll 1\n  for(int base=tid;base<K/4;base+=ST*NT) {\n    #pragma unroll\n    for(int s=0;s<ST;s++) {\n      float4 v=ww[s];\n      uint2 a[M];\n      #pragma unroll\n      for(int t=0;t<M;t++) a[t]=xx[s][t];\n      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n      #pragma unroll\n      for(int t=0;t<M;t++) if(t<m) {\n        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n      }\n    }\n  }\n  __shared__ float sums[M][NT/32];\n  #pragma unroll\n  for(int t=0;t<M;t++) if(t<m) {\n    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n    #pragma unroll\n    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n    if(lane==0) sums[t][warp]=v;\n  }\n  __syncthreads();\n  if(warp==0) {\n    #pragma unroll\n    for(int t=0;t<M;t++) if(t<m) {\n      float v=lane<NT/32?sums[t][lane]:0.f;\n      #pragma unroll\n      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n      if(lane==0) y[t*256+row]=v;\n    }\n  }\n}\n\nint smallm2_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n  auto kern=smallm2_router_scalar<1,512,2>;\n  if(query==2) {\n    cudaFuncAttributes fa;\n    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n  }\n  if(m!=1) return -5;\n  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n  return cudaGetLastError()==cudaSuccess?0:-8;\n}\n'
_NP_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_NP_SMALLM2_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_NP_SMALLM2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_NP_SMALLM2_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_NP_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_NP_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_NP_SMALLM2_ESC_CAP = 32

def _NP_smallm2_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _NP_smallm2_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _NP_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _NP_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, _NP_SMALLM2_CPP + extra_cpp, cuda_sources=_NP_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _NP_smallm2_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _NP_smallm2_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NP_smallm2_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def _NP_smallm2_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NP_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NP_smallm2_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NP_SMALLM2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NP_smallm2_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NP_smallm2_c12_unpack(data, base, esc):
    """Inverse of smallm2_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NP_SMALLM2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NP_smallm2_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _NP_smallm2_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [_NP_smallm2_units(hi), _NP_smallm2_units(lo)]
    else:
        mats = [_NP_smallm2_units(w)]
    if fmt in (0, 2):
        data = torch.cat([_NP_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _NP_smallm2_c12_bytes(v)
        if not torch.equal(_NP_smallm2_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_NP_smallm2_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _NP_smallm2_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _NP_smallm2_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _NP_smallm2_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, _NP_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _NP_SMALLM2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _NP_smallm2_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _NP_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def _NP_smallm2_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _NP_smallm2_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _NP_smallm2_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _NP_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _NP_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _NP_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if _NP_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _NP_smallm2_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def _NP_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _NP_smallm2_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _NP_smallm2_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _NP_smallm2_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _NP_smallm2_verify_onehot(ext, inst_id[inst], sp, _NP_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _NP_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
_NP_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def _NP_prepare_smallm2(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in _NP_SMALLM2_ROUTES.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, _NP_smallm2_bf16_twin(inst)))))
    ext = _NP_smallm2_build(instances, 'smallm2_final_v3')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = _NP_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = _NP_smallm2_plan(ext, ids, packs, streams, w, layer, key, _NP_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

def _NP_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != _NP_SMALLM2_LAYER_K[layer]:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = _NP_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
_NP_SMALLM2_ROUTES = {key: value for key, value in _NP_SMALLM2_ROUTES.items() if key == ('pooler', 8)}


"""Lossless BF16 small-token projection routes validated on H200."""
import torch
from torch.utils.cpp_extension import load_inline
_NU_SMTRACE_CUDA_TEMPLATE = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define SMD __device__ __forceinline__\n\nSMD int4 smtrace_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSMD uint4 smtrace_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nSMD uint32_t smtrace_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nSMD uint32_t smtrace_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nSMD void smtrace_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nSMD void smtrace_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  const uint32_t cm = C & 0x77777777u;\n  const uint32_t hlo = smtrace_prmt(T0, T1, cm);\n  const uint32_t hhi = smtrace_prmt(T0, T1, cm >> 16);\n  a[0] = smtrace_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n  a[1] = smtrace_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n  a[2] = smtrace_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n  a[3] = smtrace_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n}\n\n__device__ __noinline__ void smtrace_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nSMD void smtrace_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) smtrace_never(p);\n}\n\nSMD uint32_t smtrace_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nSMD uint32_t smtrace_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nSMD uint32_t smtrace_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nSMD void smtrace_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nSMD void smtrace_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nSMD void smtrace_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nSMD void smtrace_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nSMD void smtrace_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nSMD void smtrace_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nSMD bool smtrace_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nSMD uint64_t smtrace_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.\nSMD void smtrace_wait_bar(uint32_t bar) {\n  if (smtrace_mbar_try(bar)) return;\n  const uint64_t t0 = smtrace_gtime();\n  while (!smtrace_mbar_try(bar))\n    if (smtrace_gtime() - t0 > 20000000ull) __trap();\n}\n\nSMD void smtrace_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nSMD float4 smtrace_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nSMD void smtrace_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nSMD void smtrace_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct SmFmt;\ntemplate <> struct SmFmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct SmFmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct SmFmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct SmFmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nSMD void smtrace_decode(const int4 (&w)[SmFmt<FMT>::NL], uint32_t (&A)[SmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < SmFmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < SmFmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      smtrace_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      smtrace_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      smtrace_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      smtrace_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nSMD void smtrace_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nSMD void smtrace_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) smtrace_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>\n__global__ void __launch_bounds__(NTMAX) smtrace_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = SmFmt<FMT>::NL, NA = SmFmt<FMT>::NA;\n  constexpr bool C12 = FMT == 1 || FMT == 3;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = min(4 * MT, (m + 1) >> 1);\n    const uint32_t bar = smtrace_smem_u32(&s_bar);\n    uint32_t rank = 0;\n    if constexpr (PUSH) {\n      rank = smtrace_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        smtrace_mbar_init(bar, 1);\n        smtrace_fence_init();\n      }\n      if (rank == 0) {\n        __syncthreads();\n        if(tid==0) {\n          const uint32_t parts=RED==2?(uint32_t)((S-1)*NW):(uint32_t)((S-1)*WR);\n          smtrace_mbar_expect(bar,parts*RT*TV*128u);\n        }\n      }\n      smtrace_cl_arrive();\n    }\n\n    uint32_t my_e = 0xFFFFFFFFu;\n    if constexpr (C12 && !RO) my_e = smtrace_lde(esc + (size_t)gw * 32 + lane);\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) w[slot][j][l] = smtrace_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = smtrace_ldx(xp);\n          xv[slot][n][1] = smtrace_ldx(xp + 8);\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    smtrace_order_point(m, outp);\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          smtrace_order_point(m, outp);\n        } else {\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) smtrace_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          uint32_t b[MT][8];\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          if (si + ST < KU) load(s, si + ST);\n          smtrace_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) smtrace_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n                } else {\n                  smtrace_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                  if constexpr (HILO) smtrace_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                }\n              }\n        }\n      }\n    }\n\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          smtrace_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) smtrace_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                smtrace_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem);\n          const int NI = WR * RT * MT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            smtrace_reduce(part, NI, WK, NI, tid, nt, emit_full);\n          } else {\n            float4* csum = part + (size_t)NW * RT * MT * 32;\n            smtrace_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) smtrace_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2) {\n            smtrace_cl_wait();\n            if (t < TV) {\n              const uint32_t rbar = smtrace_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = n * 4 + t;\n                  if (tn < TV) {\n                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            smtrace_wait_bar(bar);\n            smtrace_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if (n * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            smtrace_cl_wait();\n            const uint32_t rbar = smtrace_mapa(bar, 0);\n            smtrace_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            smtrace_wait_bar(bar);\n            smtrace_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int smtrace_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>\nstatic int smtrace_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = smtrace_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS>;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = std::min(4 * MT, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = smtrace_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[1];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    lc.numAttrs = 1;\n  }\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\n@@CASES@@\n    default: return -99;\n  }\n}\n'
_NU_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_NU_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_NU_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_NU_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_NU_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_NU_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_NU_SMTRACE_ESC_CAP = 32

def _NU_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _NU_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _NU_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _NU_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, _NU_SMTRACE_CPP + extra_cpp, cuda_sources=_NU_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _NU_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _NU__smtrace_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NU__smtrace_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def _NU_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NU__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NU_smtrace_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NU_SMTRACE_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NU__smtrace_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NU_smtrace_c12_unpack(data, base, esc):
    """Inverse of smtrace_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NU_SMTRACE_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NU_smtrace_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _NU_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [_NU_smtrace_units(hi), _NU_smtrace_units(lo)]
    else:
        mats = [_NU_smtrace_units(w)]
    if fmt in (0, 2):
        data = torch.cat([_NU_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _NU_smtrace_c12_bytes(v)
        if not torch.equal(_NU_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_NU_smtrace_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _NU_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _NU_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _NU_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, _NU_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _NU_SMTRACE_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _NU__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _NU_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
_NU_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_NU_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_NU_SMTRACE_MAX_M = 32
_NU_SMTRACE_TABLE_M.update({('router', 8): _NU_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): _NU_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): _NU_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): _NU_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): _NU_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): _NU_SMTRACE_TABLE_MT['unpooler', 1]})
_NU_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_NU_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_NU_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_NU_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _NU_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _NU_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _NU_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _NU_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _NU_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _NU_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if _NU_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _NU_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def _NU_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _NU_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _NU_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _NU_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _NU_smtrace_verify_onehot(ext, inst_id[inst], sp, _NU_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _NU_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _NU_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(_NU_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(_NU_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _NU_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _NU_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _NU_smtrace_bf16_twin(inst):
            instances.append(_NU_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _NU_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _NU_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _NU_smtrace_bf16_twin(inst):
            plan = _NU_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _NU_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _NU_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= _NU_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _NU_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != _NU_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    if _NU_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_NU_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def _NU_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in _NU_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in _NU_SMTRACE_ROUTES.items() if key[0] in usable}
    return _NU_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

def _NU_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in _NU_SMTRACE_ROUTES:
        return None
    return _NU_smtrace_run_core(x, layer, n, st)


# Nathen Fernandes (MIT): measured selective O/pooler routes.
_nathen_m8_parent_prepare = prepare
_nathen_m8_parent_kernel = kernel_fn


def prepare(weights):
    state = _nathen_m8_parent_prepare(weights)
    state['nathen_pooler8'] = _kshare_modules(_NP_prepare_smallm2({'pooler': weights['pooler']}))
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    x = data['x']
    layer = data['config']['layer']
    if x.shape[0] == 8 and layer == 'pooler':
        out = _NP_run_smallm2(x, layer, data['config']['n'], state['nathen_pooler8'])
        if out is not None:
            return out
    return _nathen_m8_parent_kernel(data, state)


'Experimental lossless nine-instruction BF16 decode. No route is promoted. See FINDINGS.txt.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline












































_NDD_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}











_NDD_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def _NDD_dd_cap2_b_cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0






_NDD_CPSSM_CUDA_TEMPLATE = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n\n#define CPSSD __device__ __forceinline__\n\nCPSSD int4 dd_cap2_b_cpssm_ldw(const int4* p) {\n  int4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nCPSSD uint4 dd_cap2_b_cpssm_ldx(const void* p) {\n  uint4 v;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n\nCPSSD uint32_t dd_cap2_b_cpssm_lde(const uint32_t* p) {\n  uint32_t v;\n  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n\nCPSSD uint32_t dd_cap2_b_cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n  return d;\n}\n\nCPSSD void dd_cap2_b_cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\nCPSSD void dd_cap2_b_cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n  uint32_t rotated;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n  const uint32_t low_high = (C & 0x87878787u) + T0;\n  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n  a[0] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x5140u);\n  a[1] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x7362u);\n  a[2] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x5140u);\n  a[3] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x7362u);\n}\n\n__device__ __noinline__ void dd_cap2_b_cpssm_never(void* p) { asm volatile("trap;"); }\n\n// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\nCPSSD void dd_cap2_b_cpssm_order_point(int m, void* p) {\n  uint32_t t;\n  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n  if ((t & (uint32_t)m) == 0x7fffffffu) dd_cap2_b_cpssm_never(p);\n}\n\nCPSSD uint32_t dd_cap2_b_cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\nCPSSD uint32_t dd_cap2_b_cpssm_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n\nCPSSD uint32_t dd_cap2_b_cpssm_mapa(uint32_t a, uint32_t rank) {\n  uint32_t d;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n  return d;\n}\n\nCPSSD void dd_cap2_b_cpssm_mbar_init(uint32_t bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n}\n\nCPSSD void dd_cap2_b_cpssm_mbar_expect(uint32_t bar, uint32_t tx) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n}\n\nCPSSD void dd_cap2_b_cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\nCPSSD void dd_cap2_b_cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\nCPSSD void dd_cap2_b_cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n\nCPSSD void dd_cap2_b_cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n}\n\nCPSSD bool dd_cap2_b_cpssm_mbar_try(uint32_t bar) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n  return ok != 0;\n}\n\nCPSSD uint64_t dd_cap2_b_cpssm_gtime() {\n  uint64_t t;\n  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n  return t;\n}\n\n// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.\nCPSSD void dd_cap2_b_cpssm_wait_bar(uint32_t bar) {\n  if (dd_cap2_b_cpssm_mbar_try(bar)) return;\n  const uint64_t t0 = dd_cap2_b_cpssm_gtime();\n  while (!dd_cap2_b_cpssm_mbar_try(bar))\n    if (dd_cap2_b_cpssm_gtime() - t0 > 20000000ull) __trap();\n}\n\nCPSSD void dd_cap2_b_cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n\nCPSSD float4 dd_cap2_b_cpssm_shfl_xor4(const float4& v, int o) {\n  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n}\n\nCPSSD void dd_cap2_b_cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\nCPSSD void dd_cap2_b_cpssm_store(float* p, float v) { *p = v; }\n\ntemplate <int FMT> struct CpsSmFmt;\ntemplate <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };\ntemplate <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };\ntemplate <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };\ntemplate <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };\n\ntemplate <int FMT>\nCPSSD void dd_cap2_b_cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n                   uint32_t U0, uint32_t U1) {\n  if constexpr (FMT == 0 || FMT == 2) {\n#pragma unroll\n    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {\n      A[4 * l + 0] = (uint32_t)w[l].x;\n      A[4 * l + 1] = (uint32_t)w[l].y;\n      A[4 * l + 2] = (uint32_t)w[l].z;\n      A[4 * l + 3] = (uint32_t)w[l].w;\n    }\n  } else {\n#pragma unroll\n    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {\n      const int4* u = w + 3 * h;\n      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n      dd_cap2_b_cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n      dd_cap2_b_cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n      dd_cap2_b_cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n      dd_cap2_b_cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n    }\n  }\n}\n\n// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte\'s exponent bits.\ntemplate <int NR>\nCPSSD void dd_cap2_b_cpssm_patch(uint32_t (&A)[NR], uint32_t e) {\n  const uint32_t reg = (e >> 8) & 31u;\n  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n#pragma unroll\n  for (int r = 0; r < NR; ++r)\n    if (reg == (uint32_t)r) A[r] ^= xr;\n}\n\n// Adjacent lanes read adjacent float4 slots, with each lane summing that slot\'s split-K partials.\ntemplate <class Emit>\nCPSSD void dd_cap2_b_cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n  for (int idx = tid; idx < NI; idx += nt) {\n    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n    for (int p = 0; p < P; ++p) dd_cap2_b_cpssm_add4(v, src[(size_t)p * PSTR + idx]);\n    emit(idx, v);\n  }\n}\n\n// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n// straight-line code block at the end, 128 (with 4) no output stores.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>\n__global__ void __launch_bounds__(NTMAX) dd_cap2_b_cpssm_gemv(\n    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;\n  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;\n  constexpr bool C12 = FMT == 1 || FMT == 3;\n  constexpr bool HILO = FMT >= 2;\n  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n  OutT* out = reinterpret_cast<OutT*>(outp);\n  extern __shared__ __align__(16) unsigned char smem[];\n  __shared__ __align__(8) uint64_t s_bar;\n  const int rawbytes = CP * WR * WK * RT * NL * 512;\n  if constexpr (EMPTY) {\n    return;\n  } else {\n    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n    const int S = CS ? CS : gridDim.y;\n    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n    const int G = GX * S * NW;\n    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n    const int rt0 = (blockIdx.x * WR + wr) * RT;\n    const int c0 = (blockIdx.y * WK + wk) * KU;\n    const int TV = min(4 * MT, (m + 1) >> 1);\n    const uint32_t bar = dd_cap2_b_cpssm_smem_u32(&s_bar);\n    uint32_t rank = 0;\n    if constexpr (PUSH) {\n      rank = dd_cap2_b_cpssm_rank();\n      if (rank == 0 && tid == 0) {\n        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n        dd_cap2_b_cpssm_mbar_init(bar, 1);\n        dd_cap2_b_cpssm_mbar_expect(bar, parts * RT * TV * 128u);\n        dd_cap2_b_cpssm_fence_init();\n      }\n      dd_cap2_b_cpssm_cl_arrive();\n    }\n\n    uint32_t my_e = 0xFFFFFFFFu;\n    if constexpr (C12 && !RO) my_e = dd_cap2_b_cpssm_lde(esc + (size_t)gw * 32 + lane);\n\n    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n    const size_t sstride = (size_t)G * (RT * NL * 32);\n    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n\n    int4* rawring = reinterpret_cast<int4*>(smem);\n    auto stage_weights = [&](int slot, int si) {\n      if constexpr (CP > 0) {\n        if (si < KU) {\n#pragma unroll\n          for (int j=0; j<RT; ++j)\n#pragma unroll\n            for (int l=0; l<NL; ++l) {\n              const uint32_t dst=dd_cap2_b_cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);\n              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;\n              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");\n            }\n        }\n        asm volatile("cp.async.commit_group;" ::: "memory");\n      }\n    };\n#pragma unroll\n    for (int q=0; q<CP; ++q) stage_weights(q,q);\n    int4 w[ST][RT][NL];\n    uint4 xv[ST][MT][2];\n    auto load = [&](int slot, int s) {\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = dd_cap2_b_cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n      if constexpr (!NOX && !RO) {\n#pragma unroll\n        for (int n = 0; n < MT; ++n) {\n          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n          xv[slot][n][0] = dd_cap2_b_cpssm_ldx(xp);\n          xv[slot][n][1] = dd_cap2_b_cpssm_ldx(xp + 8);\n        }\n      }\n    };\n#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n    dd_cap2_b_cpssm_order_point(m, outp);\n\n    float acc[NA][RT][MT][4];\n#pragma unroll\n    for (int h = 0; h < NA; ++h)\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n    uint32_t fold = 0;\n    uint32_t nxt = 0xFFFFFFFFu;\n    int ep = 0;\n    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n\n#pragma unroll 1\n    for (int i = 0; i < KU; i += ST) {\n#pragma unroll\n      for (int s = 0; s < ST; ++s) {\n        const int si = i + s;\n        if (ST == 3 && si >= KU) break;\n        if constexpr (RO) {\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int l = 0; l < NL; ++l)\n              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n          if (si + ST < KU) load(s, si + ST);\n          dd_cap2_b_cpssm_order_point(m, outp);\n        } else {\n          if constexpr (CP > 0) {\n            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");\n#pragma unroll\n            for (int j=0; j<RT; ++j)\n#pragma unroll\n              for (int l=0; l<NL; ++l) {\n                const uint32_t src=dd_cap2_b_cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);\n                int4& v=w[s][j][l];\n                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");\n              }\n          }\n          uint32_t A[RT][NA * 16];\n#pragma unroll\n          for (int j = 0; j < RT; ++j) dd_cap2_b_cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n          if constexpr (CP > 0) stage_weights(si%CP,si+CP);\n          uint32_t b[MT][8];\n#pragma unroll\n          for (int n = 0; n < MT; ++n) {\n            if constexpr (NOX) {\n#pragma unroll\n              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n            } else {\n              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n            }\n          }\n          if (si + ST < KU) load(s, si + ST);\n          dd_cap2_b_cpssm_order_point(m, outp);\n          if constexpr (C12) {\n#pragma unroll\n            for (int j = 0; j < RT; ++j) {\n              const uint32_t useq = (uint32_t)(si * RT + j);\n              if ((nxt >> 18) == useq) {\n                do {\n                  if (lane == (int)((nxt >> 13) & 31u)) dd_cap2_b_cpssm_patch<NA * 16>(A[j], nxt);\n                  ++ep;\n                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n                } while ((nxt >> 18) == useq);\n              }\n            }\n          }\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n#pragma unroll\n              for (int q = 0; q < 4; ++q) {\n                if constexpr (NOMMA) {\n#pragma unroll\n                  for (int h = 0; h < NA; ++h)\n                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n                            A[j][16 * h + 4 * q + 3];\n                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n                } else {\n                  dd_cap2_b_cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                  if constexpr (HILO) dd_cap2_b_cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n                }\n              }\n        }\n      }\n    }\n\n    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");\n    if constexpr (RO) {\n      if constexpr (RO_CODE) {\n        if (warp == 0) {\n          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n          const uint32_t k = (uint32_t)m | 1u;\n#pragma unroll\n          for (int q = 0; q < 64; ++q)\n            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n          fold = a ^ b ^ c ^ d;\n        }\n      }\n      if constexpr (RO_STORE) {\n        if ((lane & 3) == 0)\n          dd_cap2_b_cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n      } else {\n        if (fold == 0x9e3779b9u) dd_cap2_b_cpssm_store(out, 1.f);\n      }\n      return;\n    } else {\n      float r[RT][MT][4];\n#pragma unroll\n      for (int j = 0; j < RT; ++j)\n#pragma unroll\n        for (int n = 0; n < MT; ++n)\n#pragma unroll\n          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n      if constexpr (NOMMA) {\n        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n      }\n\n      const int RB = WR * RT * 16;\n      const int row0 = blockIdx.x * RB;\n      if (NORED || (RED == 0 && WK == 1)) {\n#pragma unroll\n        for (int j = 0; j < RT; ++j)\n#pragma unroll\n          for (int n = 0; n < MT; ++n)\n#pragma unroll\n            for (int c = 0; c < 4; ++c) {\n              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n                dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);\n            }\n        return;\n      }\n      if constexpr (!NORED) {\n        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n        auto emit_full = [&](int idx, float4 v) {\n          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        auto emit_compact = [&](int idx, float4 v) {\n          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n          const float vv[4] = {v.x, v.y, v.z, v.w};\n#pragma unroll\n          for (int c = 0; c < 4; ++c) {\n            const int tok = 2 * t + (c & 1);\n            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);\n          }\n        };\n        if constexpr (RED == 0 || RED == 1) {\n          float4* part = reinterpret_cast<float4*>(smem + rawbytes);\n          const int NI = WR * RT * MT * 32;\n#pragma unroll\n          for (int j = 0; j < RT; ++j)\n#pragma unroll\n            for (int n = 0; n < MT; ++n)\n              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n          __syncthreads();\n          if constexpr (RED == 0) {\n            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);\n          } else {\n            float4* csum = part + (size_t)NW * RT * MT * 32;\n            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n            cl.sync();\n            const int rk = (int)cl.block_rank();\n            const int per = (NI + S - 1) / S;\n            const int e1 = min(NI, (rk + 1) * per);\n            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n              float4 v = *cl.map_shared_rank(csum + idx, 0);\n              for (int q = 1; q < S; ++q) dd_cap2_b_cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));\n              emit_full(idx, v);\n            }\n            cl.sync();\n          }\n        } else {\n          const int NIc = WR * RT * TV * 8;\n          const int t = lane & 3, g = lane >> 2;\n          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);\n          // The leader\'s own partials use plain shared stores; only remote CTAs push with st.async.\n          if constexpr (RED == 2) {\n            dd_cap2_b_cpssm_cl_wait();\n            if (t < TV) {\n              const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n) {\n                  const int tn = n * 4 + t;\n                  if (tn < TV) {\n                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;\n                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n                    if (rank == 0) *dst = v;\n                    else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);\n                  }\n                }\n            }\n            if (rank != 0) return;\n            __syncthreads();\n            dd_cap2_b_cpssm_wait_bar(bar);\n            dd_cap2_b_cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n          } else {\n            float4* part = recv + (size_t)S * NIc;\n            if (t < TV) {\n#pragma unroll\n              for (int j = 0; j < RT; ++j)\n#pragma unroll\n                for (int n = 0; n < MT; ++n)\n                  if (n * 4 + t < TV)\n                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =\n                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n            }\n            __syncthreads();\n            dd_cap2_b_cpssm_cl_wait();\n            const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);\n            dd_cap2_b_cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n              float4* dst = recv + (size_t)rank * NIc + idx;\n              if (rank == 0) *dst = v;\n              else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);\n            });\n            if (rank != 0) return;\n            __syncthreads();\n            dd_cap2_b_cpssm_wait_bar(bar);\n            dd_cap2_b_cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n          }\n        }\n      }\n    }\n  }\n}\n\nstatic int dd_cap2_b_cpssm_smem_optin() {\n  static int v = -1;\n  if (v < 0) {\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n  }\n  return v;\n}\n\n// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\ntemplate <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>\nstatic int dd_cap2_b_cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  auto kern = dd_cap2_b_cpssm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;\n  const int policy = query / 100;\n  query %= 100;\n  if (query == 2) {\n    cudaFuncAttributes fa;\n    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n  }\n  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n  if (N % RB) return -2;\n  if ((int64_t)S * WK * KU * 64 != K) return -3;\n  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n  if (m < 1 || m > MT * 8) return -5;\n  const bool cl = RED != 0;\n  if (!cl && S != 1) return -9;\n  if (cl && (S < 2 || S > 16)) return -9;\n  const int TV = std::min(4 * MT, (m + 1) / 2);\n  size_t sm = 0;\n  if (!(DIAG & 28)) {\n    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n  }\n  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;\n  static int max_dyn = -1;\n  if (max_dyn < 0) {\n    cudaFuncAttributes fa;\n    cudaFuncGetAttributes(&fa, kern);\n    max_dyn = dd_cap2_b_cpssm_smem_optin() - (int)fa.sharedSizeBytes;\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  }\n  if ((int64_t)sm > max_dyn) return -6;\n  cudaLaunchConfig_t lc = {};\n  cudaLaunchAttribute at[2];\n  lc.gridDim = dim3(N / RB, S, 1);\n  lc.blockDim = dim3(nt, 1, 1);\n  lc.dynamicSmemBytes = sm;\n  lc.stream = (cudaStream_t)stream;\n  if (cl) {\n    at[0].id = cudaLaunchAttributeClusterDimension;\n    at[0].val.clusterDim.x = 1;\n    at[0].val.clusterDim.y = S;\n    at[0].val.clusterDim.z = 1;\n    lc.attrs = at;\n    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;\n    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);\n    lc.numAttrs = 2;\n  }\n  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }\n  if (query == 4) return (int)sm;\n  if (query == 1) {\n    int v = 0;\n    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n    cudaGetLastError();\n    return e == cudaSuccess ? v : -11;\n  }\n  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n                     (uint32_t)((uint64_t)u01 >> 32));\n  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n}\n\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n  switch (inst) {\n@@CASES@@\n    default: return -99;\n  }\n}\n'
_NDD_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_NDD_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_NDD_CPSSM_ESC_CAP = 32

def _NDD_dd_cap2_b_cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _NDD_dd_cap2_b_cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NDD_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NDD_dd_cap2_b_cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NDD_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NDD_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NDD_dd_cap2_b_cpssm_table(base):
    return base * 16843009

def _NDD_dd_cap2_b_cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [_NDD_dd_cap2_b_cpssm_units(hi), _NDD_dd_cap2_b_cpssm_units(lo)]
    else:
        mats = [_NDD_dd_cap2_b_cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([_NDD_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _NDD_dd_cap2_b_cpssm_c12_bytes(v)
        if not torch.equal(_NDD_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_NDD_dd_cap2_b_cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _NDD_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _NDD_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _NDD_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, _NDD_CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _NDD_CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _NDD_cpstage___to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _NDD_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
_NDD_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_NDD_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_NDD_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_NDD_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_NDD_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_NDD_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _NDD_dd_cap2_b_cpssm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _NDD_dd_cap2_b_cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _NDD_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _NDD_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _NDD_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if _NDD_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _NDD_dd_cap2_b_cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def _NDD_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _NDD_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _NDD_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _NDD_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _NDD_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, _NDD_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _NDD_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _NDD_cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NDD_cpstage___to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
_NDD_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def _NDD_cps_build(family, instances, depth):
    is32 = False
    prefix = 'dd_cap2_b_cpssm_'
    template = _NDD_CPSSM_CUDA_TEMPLATE
    cpp = _NDD_CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _NDD_prepare_cps(weights, depth=None, layers=_NDD_CPS_LAYERS, buckets=(2, 4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
        for bucket, selected_depth in ((2, 2), (4, 1)):
            if bucket in buckets and selected_layers:
                part = _NDD_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0, 1, 2, 3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm',):
        table = _NDD_CPSSM_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and (layer in _NDD_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and ((layer, bucket) in _NDD_CPS32_TABLE_MT):
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0, rows, _NDD_CPSSM_LAYER_K[layer], *cfg)
            entries.append(((layer, 'mt', bucket), (inst, cfg)))
        if include_policy and family == 'sm':
            for (layer, m), (inst, cfg) in _NDD_CPSSM_TABLE_M.items():
                if layer not in _NDD_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m, rows, _NDD_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'm', m), (inst, cfg)))
        if not entries:
            continue
        twin = _NDD_dd_cap2_b_cpssm_bf16_twin
        instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
        ext = _NDD_cps_build(family, instances, depth)
        ids = {inst: i for i, inst in enumerate(instances)}
        plan_fn = _NDD_dd_cap2_b_cpssm_plan
        packs, streams = ({}, {})
        for key, (inst, cfg) in entries:
            sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
            if sp is not None:
                st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
    return st

def _NDD_cps_launch(sp, x, out, n, policy=0, query=0):
    fn = _NDD_dd_cap2_b_cpssm_launch_raw
    return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def _NDD_run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _NDD_dd_cap2_b_cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != _NDD_CPS32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    rc = _NDD_cps_launch(sp, x, out, n, st.get('policy', 0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def _NDD_tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
_NDD_tt_mw_CUDA_PREFIX = '// dd_cap2_b_\n' + '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {\n  uint32_t rotated;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n  const uint32_t low_high = (C & 0x87878787u) + T0;\n  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n  a[0] = prmt(L0, low_high, 0x5140u);\n  a[1] = prmt(L0, low_high, 0x7362u);\n  a[2] = prmt(L1, high_high, 0x5140u);\n  a[3] = prmt(L1, high_high, 0x7362u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
_NDD_tt_mw_CUDA_CORE = '\n\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\n__global__ __launch_bounds__(128*WK,CAP) void m32w_gemm(\n    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,\n    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {\n  extern __shared__ __align__(1024) uint8_t sm[];\n  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;\n  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;\n  uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;\n  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;\n  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;\n  uint64_t pw,px;\n  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));\n  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  auto loadx=[&](int i) {\n    if(tid%128==0 && i<nkb) {\n      int s=i%Q;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n      tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);\n    }\n  };\n  #pragma unroll\n  for(int s=0;s<Q;s++)loadx(s);\n  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;\n  if(C12 && warp<RW) {\n    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;\n    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);\n  }\n  uint4 raw[PF][NL]={};\n  auto load=[&](int slot,int i) {\n    if(i<nkb && warp<RW) {\n      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;\n      #pragma unroll\n      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);\n    }\n  };\n  #pragma unroll\n  for(int p=0;p<PF;p++)load(p,p);\n  float acc[BN/2];\n  #pragma unroll\n  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));\n  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {\n    #pragma unroll\n    for(int p=0;p<PF;p++) {\n      int i=ib+p;if(i>=nkb)break;\n      uint32_t a[16];\n      if constexpr(C12) {\n        dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);\n        dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);\n        dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);\n        dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);\n      } else {\n        #pragma unroll\n        for(int l=0;l<4;l++) {\n          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;\n        }\n      }\n      load(p,i+PF);\n      if constexpr(C12) {\n        while((nxt>>18)==uint32_t(i)) {\n          if(lane==int((nxt>>13)&31u))patch(a,nxt);\n          ++ep;\n          uint32_t ev=ep<32?e0:e1;\n          nxt=__shfl_sync(0xffffffffu,ev,ep&31);\n          if(ep>=64)nxt=0xffffffffu;\n        }\n      }\n      int s=i%Q;\n      waitbar(full+s*8,(i/Q)&1);\n      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n      if constexpr(BN==16)mma16(acc,a,desc(base+s*SB));\n      if constexpr(BN==32)mma32(acc,a,desc(base+s*SB));\n      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n      #pragma unroll\n      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");\n      loadx(i+Q);\n    }\n  }\n  __syncthreads();\n  if(tid%128==0) {\n    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");\n  }\n  __syncthreads();\n  int row=warp*16+lane/4,col=2*(lane%4);\n  #pragma unroll\n  for(int j=0;j<BN/8;j++) {\n    #pragma unroll\n    for(int h=0;h<2;h++) {\n      if(warp>=RW)continue;\n      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];\n      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];\n    }\n  }\n  __syncthreads();\n  if(split>1)cluster_sync();\n  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {\n    int token=q/(R/4),row=(q%(R/4))*4;\n    if(mt*BN+token>=m)continue;\n    float4 sum=make_float4(0,0,0,0);\n    for(int pp=0;pp<split*WK;pp++) {\n      int s=pp/WK;\n      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;\n      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));\n      float4 v;\n      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");\n      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;\n    }\n    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);\n    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);\n  }\n  __syncthreads();\n  if(split>1)cluster_sync();\n}\n\n'
_NDD_tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
_NDD_tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
_NDD_tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def _NDD_tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return _NDD_tt_mw_CUDA_PREFIX + '\n'.join((_NDD_tt_mw_mma(n) for n in (16, 32, 64))) + _NDD_tt_mw_CUDA_CORE + _NDD_tt_mw_CUDA_MAP + _NDD_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def _NDD_tt_mw_build(instances):
    src = _NDD_tt_mw_cuda_source(instances)
    name = 'dd_cap2_b_m32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, _NDD_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])

def _NDD_tt_mw__to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NDD_tt_mw__to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def _NDD_tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NDD_tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NDD_tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NDD_tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NDD_tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NDD_tt_mw_sm_table(base):
    return base * 16843009

def _NDD_tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _NDD_tt_mw_sm_c12_bytes(v)
        if not torch.equal(_NDD_tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = _NDD_tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = _NDD_tt_mw__to_i32(entry)
    else:
        data = _NDD_tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def _NDD_tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def _NDD_tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if _NDD_tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
_NDD_tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def _NDD_tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = _NDD_tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = _NDD_tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = _NDD_tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not _NDD_tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not _NDD_tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def _NDD_tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    rc = _NDD_tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out
import torch
from torch.utils.cpp_extension import load_inline





















import hashlib
import torch
from torch.utils.cpp_extension import load_inline
import torch
from torch.utils.cpp_extension import load_inline














import hashlib
import torch
from torch.utils.cpp_extension import load_inline


































dd_DECODEDIET_ROUTES = (('pooler',16),('unpooler',16))

def dd_prepare_decodediet(weights) -> dict:
    routes = dd_DECODEDIET_ROUTES
    state = {'routes': frozenset(routes)}
    if not routes:
        return state
    if any((m == 16 and layer in ('pooler', 'unpooler') for layer, m in routes)):
        layers = tuple((layer for layer, m in routes if m == 16 and layer in ('pooler', 'unpooler')))
        state['cps'] = _NDD_prepare_cps(weights, layers=layers, buckets=(2,))
    return state

def dd_run_decodediet(x, layer, n, state) -> torch.Tensor | None:
    m = x.shape[0] if x.ndim == 2 else 0
    if (layer, m) not in state.get('routes', ()):
        return None
    return _NDD_run_cps(x, layer, n, state['cps'])


# Nathen Fernandes (MIT): literal pooler-m16 decodediet route.
dd_DECODEDIET_ROUTES = (('pooler', 16),)
_nathen_pool16_parent_prepare = prepare
_nathen_pool16_parent_kernel = kernel_fn


def prepare(weights):
    state = _nathen_pool16_parent_prepare(weights)
    state['nathen_pool16'] = _kshare_modules(dd_prepare_decodediet({'pooler': weights['pooler']}))
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    x = data['x']
    if data['config']['layer'] == 'pooler' and x.shape[0] == 16:
        out = dd_run_decodediet(x, 'pooler', data['config']['n'], state['nathen_pool16'])
        if out is not None:
            return out
    return _nathen_pool16_parent_kernel(data, state)


# [BFX R32] ruwwww (Apache-2.0, github.com/ruwwww/open-kernel-challenge @c035bf9) route choice: router m == 32 on
# rr_kernel<32,16,4,1,2,1> (ROUTER_CONFIGS[0]; ROUTER_CUDA above, byte-identical to ruwwww's judged build, Nathen MIT).
# FP32 router weights as BF16 hi + lo K tiles, FP32 wgmma accumulation; 16-CTA K split, fixed-order owner sum.
# prepare builds and checks it on synthetic data only (FP64 reference, tol 2e-5); any failure keeps the sk2 route.
_rr32_parent_prepare = prepare
_rr32_parent_kernel = kernel_fn


def prepare(weights):
    state = _rr32_parent_prepare(weights)
    w = weights.get('router'); st = {'enabled': False}
    try:
        if (w is not None and tuple(w.shape) == (256, 8192) and w.dtype == torch.float32 and w.is_cuda
                and torch.cuda.get_device_capability(w.device) == (9, 0)):
            cand = prepare_candidates(weights, ROUTER_CONFIGS)
            g = torch.Generator(device=w.device); g.manual_seed(4242)
            x = torch.randn((32, 8192), generator=g, device=w.device).to(torch.bfloat16)
            out = run_candidate(x, cand, 0).double(); torch.cuda.synchronize(w.device)
            ref = x.double() @ w.double().T
            err = ((out - ref).norm() / ref.norm()).item()
            if err < 2e-5:
                cand['enabled'] = True; st = cand
            else:
                print(f'[okc] rr32 router m=32: rel err {err:.3e} -> disabled', file=sys.stderr)
    except Exception as e:  # build / launch failure -> keep the master route
        print(f'[okc] rr32 router m=32 disabled: {e}', file=sys.stderr)
        st = {'enabled': False}
    state['rr32'] = _kshare_modules(st)
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    x = data['x']; c = data['config']; st = state.get('rr32', {})
    if (c['layer'] == 'router' and st.get('enabled') and c['n'] == 256 and x.ndim == 2 and x.shape[0] == 32
            and x.shape[1] == 8192 and x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
            and x.data_ptr() % 16 == 0 and x.device == st['weight'].device):
        return run_candidate(x, st, 0)
    return _rr32_parent_kernel(data, state)
