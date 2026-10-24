---
problem: muse_decode_layer
slug: full-verify-cuda-splitkv-stage2
created: 2026-10-24T165224Z
base: master
area: FA3 call
change: >
  Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify
  blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16,
  296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build
  on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page
  64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched
  so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work
  items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query:
  master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as
  f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv
  heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query
  tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA
  merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert
  each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per
  warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t
  only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never
  dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @
  V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64]
  and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine:
  splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that
  request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16
  into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged
  lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else
  changes.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: "incumbent paired FA3 153.3 us + combine + pad/unpad glue (~6.8 us x3) + norm-gate ~170 us of 604 us; the gather is the same as decode_16 where stage 1 kernel A ran ~111 us (attention), plus ~41 GFLOP of mma.sync that overlaps it: target <=130 us A+B, ~30-40 us off, +5-7% on the shape, ~+1.5-2% on the verify_16 load"
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: "incumbent FA3 296.3 us + glue ~315 us of 933 us; target <=265 us A+B: ~40-50 us off, +4-6% on the shape, ~+1.5% on the load"
  - shapes: "*layer=full load=decode_*"
    direction: flat
    why: stage 1 kernels unchanged; control
  - shapes: "*load=verify_1 *"
    direction: flat
    why: batch 1 outside the gate
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
risk: high
depends_on: full-decode-cuda-splitkv-port
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32
review: required
status: queued
---

# full-verify-cuda-splitkv-stage2

## Change

Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16, 296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page 64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query: master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @ V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64] and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine: splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16 into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else changes.

## Expected

- `*layer=full load=verify_16 *` faster: incumbent paired FA3 153.3 us + combine + pad/unpad glue (~6.8 us x3) + norm-gate ~170 us of 604 us; the gather is the same as decode_16 where stage 1 kernel A ran ~111 us (attention), plus ~41 GFLOP of mma.sync that overlaps it: target <=130 us A+B, ~30-40 us off, +5-7% on the shape, ~+1.5-2% on the verify_16 load
- `*layer=full load=verify_32 *` faster: incumbent FA3 296.3 us + glue ~315 us of 933 us; target <=265 us A+B: ~40-50 us off, +4-6% on the shape, ~+1.5% on the load
- `*layer=full load=decode_*` flat: stage 1 kernels unchanged; control
- `*load=verify_1 *` flat: batch 1 outside the gate
- `*layer=sliding*` flat: untouched

## Risk

High: new causal-in-block masking and a 4x larger query tile; mma.sync compute (~41/82 GFLOP at verify_16/32) is no longer negligible against the gather, so if A exceeds ~150 us at verify_16 the fix is fp8 conversion once per stage in smem, not more splits. Run only after stage 1 lands or is judged on its replaced kernels. Not attempted in muse_attention_decode yet; a win here is portable back there.

## Milestones

Rewrite stage 2 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
