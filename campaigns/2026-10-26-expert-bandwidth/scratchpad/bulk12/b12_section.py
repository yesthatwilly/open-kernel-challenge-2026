# --- (10) m == 1 row-owner stream on lossless 12-bit weights (bulk12: A12 pooler, A2_12 qkv_gate, A3_12 o) ----
# Campaign 2026-10-26-expert-bandwidth (bf16 quota, the user's 19:05Z steer). Section (9)'s row-owner shape (each
# CTA owns R output rows over the WHOLE K: no split-K, no workspace, no ticket; one producer warp streams the CTA's
# weight chunks through a cp.async.bulk + mbarrier ring) on a smaller weight stream:
#   * Codec (prepare, per weight): a bf16 is H << 8 | L with H = sign | exponent bits 7..1 and L = exponent bit 0 |
#     7 mantissa bits. L is kept raw (8 bits). H becomes a 4-bit code c = sign << 3 | (e71 - base) over the 8 values
#     of e71 = H & 0x7F that hold the most elements (16 binades): 12 bits per element instead of 16. Weights are
#     N(0, 1/k), so an element outside the window is one with |w| < 2^-19 or so (~1e-4 of them). Such an escape is
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
# fastest only at candidate / incumbent <= CELL_MAX_RATIO, after a torch.profiler kernel-name assertion.
B12_EXT_NAME = "muse_bf16_bulk12_m1_v1"
B12_ESC_BYTES = 2048          # per-CTA escape block: uint16 row offsets [R + 1] in its first 256 B, then entries
B12_ESC_CAP = (B12_ESC_BYTES - 256) // 8   # 224 entries of (row << 16 | k, true bf16 bits) per CTA
B12_M = 1
# layer -> (source tag, N, K, rows per CTA R, ((instance id, ring stages S, min CTAs/SM, consumer warps CW), ...),
#           state flag)
B12_CELLS = {
    "pooler":   ("A12", 4096, 8192, 32, ((0, 8, 1, 8), (1, 8, 1, 4)), "b12_pooler_ok"),
    "qkv_gate": ("A2_12", 2304, 8192, 16, ((2, 6, 2, 4), (3, 6, 2, 8)), "b12_qkv_ok"),
    "o":        ("A3_12", 8192, 1024, 64, ((4, 8, 1, 8), (5, 8, 1, 4)), "b12_o_ok"),
}
B12_FLAGS = tuple(spec[5] for spec in B12_CELLS.values())

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
            ent = torch.stack([(rl << 16) | kk[s0:s0 + cnt], vals[s0:s0 + cnt]], 1).to(torch.int32)
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


def _b12_encode_into(state: dict, layer: str, w: torch.Tensor, rows: int, inst: int, verbose: bool) -> None:
    """Encode w into the state's buffers for this cell and check the round trip on the device through the kernel's
    own decode (b12_unpack): every non-escape element bit-exact, every escape slot at w_floor's bits. Raises on any
    mismatch."""
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
    state["b12_wc_" + layer], state["b12_esc_" + layer] = wc, esc
    state["b12_base_" + layer], state["b12_inst_" + layer] = base, inst
    if verbose:
        el = w.numel()
        print(f"[b12] {layer}: base {base} (e71 window {base}..{base + 7}, binades {2 * base - 127}.."
              f"{2 * base + 15 - 127}), escapes {n_esc} ({n_esc / el:.2e} of {el}), max {max_cta} per CTA (cap "
              f"{B12_ESC_CAP}); stream {wc.numel() / 1e6:.2f} MB + escape blocks {esc.numel() / 1e6:.3f} MB vs "
              f"raw {el * 2 / 1e6:.2f} MB (x{(wc.numel() + esc.numel()) / (el * 2):.4f}); host + device round "
              f"trips bit-exact", flush=True)


def _b12_prepare(state: dict, flags=None) -> None:
    """A12 / A2_12 / A3_12. Sets state[flag] (False unless every step passes) and, per routed cell, the state's own
    encoded weight; one per-state error word. flags None (the judge's state): per cell and instance query attributes
    + occupancy (needs MINB CTAs/SM), encode + host / device round trips, canary + correctness, model-selector gate
    against kernel_fn with the flag off (today's route, the raw bulk cell included); the fastest passing instance
    routes at <= CELL_MAX_RATIO after a torch.profiler kernel-name assertion; one PATH CHOSEN line per cell. flags a
    dict (the A/B check's second state): inherit each flag and instance, encode only for cells that are on."""
    for spec in B12_CELLS.values():
        state[spec[5]] = False
    if _B12_EXT is None:
        if flags is None:
            for layer, spec in B12_CELLS.items():
                print(f"PATH CHOSEN {spec[0]} {layer} m={B12_M}: incumbent (cand n/a us vs incumbent n/a us; "
                      f"no b12 module)", flush=True)
        return
    dev = state["router"].device
    state["b12_err"] = torch.zeros(1, dtype=torch.int32, device=dev)
    if flags is not None:
        insts = flags.get("_b12_inst", {})
        for layer, spec in B12_CELLS.items():
            if flags.get(spec[5]) and insts.get(layer) is not None:
                try:   # a failure here turns off only this cell in this state, never the whole A/B
                    _b12_encode_into(state, layer, state[layer], spec[3], insts[layer], verbose=False)
                    state[spec[5]] = True
                except Exception as e:
                    print(f"[b12] second state: {spec[0]} {layer} encode raised {type(e).__name__}: {e}; "
                          f"cell off in this state", flush=True)
        return
    for layer, spec in B12_CELLS.items():
        src, n, k, rows, variants, flag = spec
        t0 = time.perf_counter()
        best = None   # (cand_us, incumbent_us, inst, label)
        why = ""
        try:
            w = state[layer]
            if not (w.dim() == 2 and tuple(w.shape) == (n, k) and w.dtype == torch.bfloat16 and w.is_contiguous()):
                why = f"weight {tuple(w.shape)} {w.dtype} unexpected"
            else:
                st_on = dict(state)
                st_on[flag] = True
                encoded = False
                for inst, stages, minb, cw in variants:
                    label = f"b12_kernel<{rows},{k},{stages},{minb},{cw}>"
                    info = [int(v) for v in _B12_EXT.b12_setup(inst)]
                    print(f"[b12] {src} {layer} {label}: regs={info[0]} local(spill)={info[1]} B static smem="
                          f"{info[2]} B dyn smem={info[3]} B, occupancy {info[4]} CTA/SM (needs {minb}), grid "
                          f"{n // rows} x {32 * (cw + 1)} threads", flush=True)
                    if info[4] < minb:
                        why = f"{label}: occupancy {info[4]} CTA/SM < {minb}"
                        continue
                    if not encoded:
                        _b12_encode_into(state, layer, w, rows, inst, verbose=True)
                        encoded = True
                    state["b12_inst_" + layer] = inst
                    st_on.update({key: state[key] for key in ("b12_wc_" + layer, "b12_esc_" + layer,
                                                              "b12_base_" + layer, "b12_inst_" + layer, "b12_err")})
                    bad = _cell_check(f"{src} {label}", lambda xx: _b12_forward(layer, xx, st_on), w, B12_M, n, k,
                                      err=state["b12_err"])
                    if bad:
                        why = f"{label}: {bad}"
                        continue
                    tg = time.perf_counter()
                    cand_us, inc_us = _cell_gate(f"{src} {label}", layer, B12_M, n, k, flag, state)
                    code = int(state["b12_err"].item())
                    print(f"[prepare-time] gate {src} {label}: {time.perf_counter() - tg:.2f}s; cand {cand_us:.2f} us "
                          f"(raw-equivalent {n * k * 2 / (cand_us * 1e-6) / 1e12:.2f} TB/s, stream "
                          f"{n * k * 1.5 / (cand_us * 1e-6) / 1e12:.2f} TB/s) vs incumbent {inc_us:.2f} us", flush=True)
                    if code:
                        why = f"{label}: error word {code} after the gate"
                        break
                    if best is None or cand_us < best[0]:
                        best = (cand_us, inc_us, inst, label)
                if best is not None:
                    cand_us, inc_us, inst, label = best
                    state["b12_inst_" + layer] = inst
                    st_on["b12_inst_" + layer] = inst
                    if not cand_us <= CELL_MAX_RATIO * inc_us:
                        why = f"gate: best {label} cand/incumbent {cand_us / inc_us:.3f} > {CELL_MAX_RATIO}"
                    else:
                        why = ""
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
        except Exception as e:  # never raises: any surprise keeps the incumbent path for this cell
            why = f"raised {type(e).__name__}: {str(e)[-300:]}"
            state[flag] = False
        if not state[flag]:
            for key in ("b12_wc_", "b12_esc_", "b12_base_", "b12_inst_"):
                state.pop(key + layer, None)
        state["b12_err"].zero_()
        c_txt = "n/a" if best is None else f"{best[0]:.2f}"
        i_txt = "n/a" if best is None else f"{best[1]:.2f}"
        print(f"[prepare-time] cell {src} {layer} m={B12_M} total: {time.perf_counter() - t0:.2f}s", flush=True)
        print(f"PATH CHOSEN {src} {layer} m={B12_M}: {'candidate ' + best[3] if state[flag] else 'incumbent'} "
              f"(cand {c_txt} us vs incumbent {i_txt} us)" + ("" if state[flag] else f"; {why}"), flush=True)
    torch.cuda.synchronize()


NEW_CELL_FLAGS = NEW_CELL_FLAGS + B12_FLAGS   # section (10)'s cells join the A/B check and the FINAL FLAGS line
