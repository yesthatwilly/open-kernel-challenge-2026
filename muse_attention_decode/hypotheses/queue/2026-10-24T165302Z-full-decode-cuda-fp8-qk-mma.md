---
problem: muse_attention_decode
slug: full-decode-cuda-fp8-qk-mma
created: 2026-10-24T165302Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel
  (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the
  native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of
  widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row
  quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1
  = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers
  (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 +
  quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at
  dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR
  (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and
  the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4
  mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick
  and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 =
  row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened
  exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte
  for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached
  build is not reused; keep the import-time self-check and its thresholds as they are (it now
  validates the fp8 QK path).
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: kernel A 111.4 us at 2.95 TB/s vs the 63.2 us floor; per lane per tile it issues 16 cvt.f16x2.e4m3x2 for K+Q-side work plus 16 for V and 8+16 mma; dropping the K widening halves the cvt count and halves QK mma issue (the 8 zero rows waste half of either op). The review of stage 1 estimated ~40 us of cvt ALU at decode_16 that must hide under the gather; if A is partly issue-bound this is ~5-10% on A (100-106 us), shape ~113-118 us
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: kernel A 245.5 us at 2.67 TB/s, lower per-byte efficiency than decode_16, which points at issue/ALU pressure rather than pure DRAM; target 225-235 us
  - shapes: "*layer=sliding* *load=verify_* *layer=full load=decode_1*"
    direction: flat
    why: outside the CUDA decode path; untouched
risk: medium
depends_on: full-decode-cuda-gather-splitkv
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# full-decode-cuda-fp8-qk-mma

## Change

Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1 = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 + quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4 mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 = row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached build is not reused; keep the import-time self-check and its thresholds as they are (it now validates the fp8 QK path).

## Expected

- `*layer=full load=decode_16*` faster: kernel A 111.4 us at 2.95 TB/s vs the 63.2 us floor; per lane per tile it issues 16 cvt.f16x2.e4m3x2 for K+Q-side work plus 16 for V and 8+16 mma; dropping the K widening halves the cvt count and halves QK mma issue (the 8 zero rows waste half of either op). The review of stage 1 estimated ~40 us of cvt ALU at decode_16 that must hide under the gather; if A is partly issue-bound this is ~5-10% on A (100-106 us), shape ~113-118 us
- `*layer=full load=decode_32*` faster: kernel A 245.5 us at 2.67 TB/s, lower per-byte efficiency than decode_16, which points at issue/ALU pressure rather than pure DRAM; target 225-235 us
- `*layer=sliding* *load=verify_* *layer=full load=decode_1*` flat: outside the CUDA decode path; untouched

## Risk

Single attributable change on the measured kernel: read kernel A time in top_kernels at decode_16/32 against 111.4/245.5 us. If flat, kernel A is DRAM/TLB-bound on the random 128-byte rows, not issue-bound, and further ALU tuning of the decode kernel should stop (next levers would be bytes in flight: stages / CTAs per SM). Numerics: e4m3 x e4m3 products are exact in fp32 and the f16 path also widened them exactly, so only the tensor-core accumulation order/precision differs (Hopper fp8 MMA accumulates with reduced internal precision; for 64-term logits that is ~1e-4 relative, far inside the 3% bound; the self-check rel L2 should stay ~1.7e-3). PTX m16n8k32 e4m3 needs sm_89+, the build already uses -arch=sm_90a. Overlaps in source with full-verify-cuda-streamk-splitkv (which adds new kernels to the same CUDA string and uses the same fp8 QK op in its own kernel); whichever lands second rebases with a textual merge, shapes are disjoint.
