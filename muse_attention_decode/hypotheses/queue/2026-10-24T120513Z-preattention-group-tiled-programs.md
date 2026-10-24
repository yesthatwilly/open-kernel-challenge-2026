---
problem: muse_attention_decode
slug: preattention-group-tiled-programs
created: 2026-10-24T120513Z
base: master
area: pre-attention norm+rope
change: >
  Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv)
  = 18 programs per token, 4608 at verify_32) to one program per (token, kv group): grid (tokens,
  Hkv), num_warps=4. Program (t, g) loads the G = Hq/Hkv = 8 query heads of group g as one [G, D]
  tile (rows h = g*G + r, columns even/odd halves as now, or a [G, D] tile plus a pair-swapped
  copy for RoPE as _norm_rope in the sliding-single-launch patch does), plus key head g and value
  head g as [1, D] rows; it applies the same scaleless RMSNorm per row (sum over axis 1, eps 1e-5,
  fp32), the same interleaved RoPE from positions[t] and the cached inv_freq when ROPE, writes the
  8 FP8 query rows (in the PAIRED layout, query h writes its half (h // G) % 2 of its 128-byte
  padded row and FP8 zeros into the other half, every byte written each call, exactly as now), and
  stores the FP8 key and value rows divided by k_scale/v_scale at cache_loc[t]. Arithmetic,
  rounding points and output layouts are identical to master; only the decomposition into programs
  changes. kernel_fn, FA3 and both epilogues are unchanged. Mechanism: the kernel moves under 1.2
  MB even at verify_32 (0.25 us at 4.8 TB/s) yet takes 4.2 us there, 2.7-2.8 us at verify_16 and
  ~2 us at decode shapes: it is bound by issuing thousands of single-warp CTAs each doing a
  dependent load-reduce-store chain, so 9x fewer, 4x wider programs should cut it toward ~1-1.5
  us.
expected:
  - shapes: "*load=verify_32*"
    direction: faster
    why: "pre-attention 4.2 us -> ~1.5 us: ~9% on sliding verify_32 (28.9 us), ~1% full"
  - shapes: "*load=verify_16*"
    direction: faster
    why: "pre-attention 2.8 us -> ~1.3 us: ~6% on sliding verify_16"
  - shapes: "*load=decode_32*"
    direction: faster
    why: "pre-attention 2.2-2.3 us -> ~1.2 us: ~4% sliding"
  - shapes: "*load=decode_16*"
    direction: faster
    why: pre-attention 1.8-2.0 us -> ~1.2 us, small
  - shapes: "*load=*_1 *"
    direction: same
    why: "1-8 tokens: 18-144 programs already fit one wave; latency per program may rise slightly with 8 rows each, expected within drift"
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# preattention-group-tiled-programs

## Change

Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (token, kv group): grid (tokens, Hkv), num_warps=4. Program (t, g) loads the G = Hq/Hkv = 8 query heads of group g as one [G, D] tile (rows h = g*G + r, columns even/odd halves as now, or a [G, D] tile plus a pair-swapped copy for RoPE as _norm_rope in the sliding-single-launch patch does), plus key head g and value head g as [1, D] rows; it applies the same scaleless RMSNorm per row (sum over axis 1, eps 1e-5, fp32), the same interleaved RoPE from positions[t] and the cached inv_freq when ROPE, writes the 8 FP8 query rows (in the PAIRED layout, query h writes its half (h // G) % 2 of its 128-byte padded row and FP8 zeros into the other half, every byte written each call, exactly as now), and stores the FP8 key and value rows divided by k_scale/v_scale at cache_loc[t]. Arithmetic, rounding points and output layouts are identical to master; only the decomposition into programs changes. kernel_fn, FA3 and both epilogues are unchanged. Mechanism: the kernel moves under 1.2 MB even at verify_32 (0.25 us at 4.8 TB/s) yet takes 4.2 us there, 2.7-2.8 us at verify_16 and ~2 us at decode shapes: it is bound by issuing thousands of single-warp CTAs each doing a dependent load-reduce-store chain, so 9x fewer, 4x wider programs should cut it toward ~1-1.5 us.

## Expected

- `*load=verify_32*` faster: pre-attention 4.2 us -> ~1.5 us: ~9% on sliding verify_32 (28.9 us), ~1% full
- `*load=verify_16*` faster: pre-attention 2.8 us -> ~1.3 us: ~6% on sliding verify_16
- `*load=decode_32*` faster: pre-attention 2.2-2.3 us -> ~1.2 us: ~4% sliding
- `*load=decode_16*` faster: pre-attention 1.8-2.0 us -> ~1.2 us, small
- `*load=*_1 *` same: 1-8 tokens: 18-144 programs already fit one wave; latency per program may rise slightly with 8 rows each, expected within drift

## Risk

Low correctness risk (same math, same stores); the reviewer should check that the paired zero-half store and the value store remain complete for every byte and that the key program per group still writes to cache_loc with the page_size-64 test layout ((loc * HKV + kh) * D is unchanged). Perf risk at batch-1 shapes: 2 programs of 4 warps per token instead of 18 of 1 may lengthen the critical path by a few hundred ns; if so, a companion with grid (tokens, Hkv, 2) (4 query heads per program) is the fallback. Independent of fa3-scheduler-metadata-side-stream except that a shorter pre-attention kernel leaves less time to hide the side-stream prepare kernel under; they should be measured separately.
