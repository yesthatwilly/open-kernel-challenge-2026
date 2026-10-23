---
problem: muse_moe_w4a16
slug: triton-w13-grouped-gemv-small-m
created: 2026-10-23T225501Z
base: master
area: w13 marlin
change: >
  For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path
  untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with two Triton launches that
  write the same cache1 [M*8, 2I] bf16 in flat slot order (row = token*8 + slot), so silu_and_mul,
  the w2 Marlin GEMM and the post-norm stay exactly as in parity. (a) compact_routes: grid (32,),
  program e loads the M*8 <= 64 mapped local_ids, mask = ids == e, cnt[e] = sum(mask), and writes
  the matching flat slot indices to slot_list[e, cumsum(mask)-1] with a masked store (cnt[e] <= M
  <= 8 because a token routes to an expert at most once). (b) w13_gemv: grid (32, 2I/BLOCK_N);
  program (e, n) loads cnt[e] and returns immediately if 0 (untouched experts cost one load), else
  loads up to BLOCK_M=16 slots, gathers x[slot//8] rows masked by arange < cnt into a [16,
  BLOCK_K] bf16 tile per K step, and dequantizes the expert's w13 tile on the fly: the uint8 tile
  [BLOCK_N, BLOCK_K/2] is split into nibbles, E2M1 magnitude from code&7 via the
  (0,.5,1,1.5,2,3,4,6) grid (arithmetic: e=c>>1, m=c&1, e==0 ? 0.5*m : (1+0.5*m)*2^(e-1)), sign
  from bit 3, times the fp8 e4m3 block scale (.to(float32), one per 16 elements); the e2m1*e4m3
  product has <= 7 significant bits so the cast to bf16 for tl.dot is exact; accumulate in fp32
  and multiply by the per-expert global scale w13_gs[e] once after the K loop; store rows at
  cache1[slot, n-tile] masked by arange < cnt. prepare() keeps, beside the Marlin bank, a second
  lossless copy of the 32 local experts' raw w13/w13_sf/w13_gs tensors (contiguous, same dtype),
  optionally nibble-regrouped within each row so the low nibbles of byte j hold element j and the
  high nibbles element j + H/2 (a K permutation applied identically to the scale index; or use
  tl.interleave instead). Marlin's moe_align_block_size still runs for the w2 GEMM in this
  hypothesis. Attribution: any change is confined to bs=1 and bs=8 and to the w13 GEMM; bs=16..256
  are untouched controls.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "SGLang is 77 us against an 11.4 us floor at bs=1 (1-2 touched experts); Marlin MoE at M=1 with 8-row blocks is far below bandwidth, while a grouped GEMV reading each touched expert's 19 MB of w13 once and skipping the other 30 experts runs near HBM speed"
  - shapes: bs=8 *
    direction: faster
    why: "bs=8: 151 us vs 37.3 us floor; same mechanism, cnt[e] <= 8 fits one BLOCK_M=16 tile so no m-loop"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens=16 > 8 stays on the parity Marlin path"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: sglang-marlin-path-parity
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# triton-w13-grouped-gemv-small-m

## Change

For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with two Triton launches that write the same cache1 [M*8, 2I] bf16 in flat slot order (row = token*8 + slot), so silu_and_mul, the w2 Marlin GEMM and the post-norm stay exactly as in parity. (a) compact_routes: grid (32,), program e loads the M*8 <= 64 mapped local_ids, mask = ids == e, cnt[e] = sum(mask), and writes the matching flat slot indices to slot_list[e, cumsum(mask)-1] with a masked store (cnt[e] <= M <= 8 because a token routes to an expert at most once). (b) w13_gemv: grid (32, 2I/BLOCK_N); program (e, n) loads cnt[e] and returns immediately if 0 (untouched experts cost one load), else loads up to BLOCK_M=16 slots, gathers x[slot//8] rows masked by arange < cnt into a [16, BLOCK_K] bf16 tile per K step, and dequantizes the expert's w13 tile on the fly: the uint8 tile [BLOCK_N, BLOCK_K/2] is split into nibbles, E2M1 magnitude from code&7 via the (0,.5,1,1.5,2,3,4,6) grid (arithmetic: e=c>>1, m=c&1, e==0 ? 0.5*m : (1+0.5*m)*2^(e-1)), sign from bit 3, times the fp8 e4m3 block scale (.to(float32), one per 16 elements); the e2m1*e4m3 product has <= 7 significant bits so the cast to bf16 for tl.dot is exact; accumulate in fp32 and multiply by the per-expert global scale w13_gs[e] once after the K loop; store rows at cache1[slot, n-tile] masked by arange < cnt. prepare() keeps, beside the Marlin bank, a second lossless copy of the 32 local experts' raw w13/w13_sf/w13_gs tensors (contiguous, same dtype), optionally nibble-regrouped within each row so the low nibbles of byte j hold element j and the high nibbles element j + H/2 (a K permutation applied identically to the scale index; or use tl.interleave instead). Marlin's moe_align_block_size still runs for the w2 GEMM in this hypothesis. Attribution: any change is confined to bs=1 and bs=8 and to the w13 GEMM; bs=16..256 are untouched controls.

## Expected

- `bs=1 *` faster: SGLang is 77 us against an 11.4 us floor at bs=1 (1-2 touched experts); Marlin MoE at M=1 with 8-row blocks is far below bandwidth, while a grouped GEMV reading each touched expert's 19 MB of w13 once and skipping the other 30 experts runs near HBM speed
- `bs=8 *` faster: bs=8: 151 us vs 37.3 us floor; same mechanism, cnt[e] <= 8 fits one BLOCK_M=16 tile so no m-loop
- `bs=16 *` flat: control: num_tokens=16 > 8 stays on the parity Marlin path
- `bs=128 *` flat: control: untouched

## Risk

Risks: Triton compile time counts toward the 15 min run limit (two small kernels, two shapes: fine); the nibble/scale indexing must match nvfp4.py (low nibble = element 2i) or the bs=1 test fails immediately; the extra 'compact_routes' launch adds ~3 us, which the bs=1 gain must exceed; early-exit programs still cost grid scheduling for 32 x (8192/BLOCK_N) programs. Correctness: fp32 accumulation and exact bf16 dequant leave more margin than Marlin's path. Rules: the raw-layout copy is the same NVFP4 bytes, a lossless permutation at most; x stays bf16; no data-derived caching (slot_list/cnt are rebuilt each call inside the timed graph); branching on num_tokens is a shape branch. Nearby shapes (bs 4) take the same path, bs 33 the Marlin path, so final-review reruns see no cliff.
