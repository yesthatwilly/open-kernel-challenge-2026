---
problem: muse_attention_decode
slug: full-decode-cuda-gather-splitkv
created: 2026-10-24T140133Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  Stage 1 of a rewrite of full-attention decode, built on master
  (sliding-fused-triton-decode1-gate). Gate (config/shape only): cfg window == 0 and q_len == 1
  and batch > 1 and num_q_heads == 16 and num_kv_heads == 2 and head_dim == 64 and page_size == 1;
  every other shape (all sliding, full decode_1, every verify, heads dp, page 64) keeps master
  code byte for byte. On the gated path run master _pre_attention_kernel with PAIRED=False (FP8 q
  [T,16,64], cache store unchanged), then two new CUDA kernels compiled once per process with
  torch.utils.cpp_extension.load_inline (-O3, -arch=sm_90a, compiled on the first eager call,
  never during capture). KERNEL A (split-KV gather attention): a persistent grid of exactly 2*132
  CTAs, 256 threads (8 warps), launched with the current torch stream. Work items are fixed
  C=1024-token chunks of each request: every CTA reads cache_seqlens[0..B-1] (B<=64) into shared
  memory and computes the per-request chunk prefix sum itself, then walks items i = blockIdx.x,
  blockIdx.x+gridDim.x, ... over the flat (b, chunk) list, so ragged agentic lengths balance
  across SMs instead of FA3 splitting each request the same way. Per item: (1) load the item 1024
  int32 page_table entries (row = page_table[b, t]) into shared memory in one coalesced pass, so
  no K/V address waits on a dependent index load inside the loop; (2) stream the item as 16 tiles
  of 64 tokens, each token one contiguous 128-byte K row (both KV heads, k_cache viewed [rows,128]
  uint8) and one 128-byte V row, with cp.async.cg 16-byte copies (8 lanes per row) into a 4-stage
  shared-memory ring (16 KB per stage, 64 KB per CTA, 2 CTAs/SM = 128 KB of gathers in flight per
  SM), commit/wait_group pipelining so tile n+3 is in flight while tile n is computed; (3) compute
  with mma.sync m16n8k16 f16 (fp32 accumulate): warp w owns kv head g = w % 2 and token slice (w /
  2) * 16 .. +16 of each tile; A = that head 8 query rows (master FP8 q up-converted to f16, rows
  8-15 zero), B = the slice K converted fp8 e4m3 -> f16 in registers (cvt.rn.f16x2.e4m3x2), scores
  times sm_scale*qk_scale_factor*k_scale*log2e, online softmax in the exp2 domain with fp32 m/l
  per warp (-inf safe: m = -inf uses 0), P rounded to f16 and P @ V_slice with mma.sync (V
  fragment built from the converted shared V tile), each warp keeping its own running (m, l, acc[8
  heads x 64]) across tiles; tokens t >= cache_seqlens[b] are masked to -inf (never dereferenced:
  their page index is not loaded and the copy is zero-filled via cp.async src-size 0). At item end
  the four warps of each kv head merge in shared memory and the CTA writes one fp32 partial acc[16
  heads,64] plus m[16], l[16] for (b, chunk) into workspace sized B * ceil(page_table.shape[1] /
  C) (allocated per call from shapes). KERNEL B (combine + epilogue): grid (B, 16), one warp per
  (request, query head): log-sum-exp merge of that request n_chunks partials (n_chunks recomputed
  from cache_seqlens in-kernel; empty partials weight 0), times v_scale, then master epilogue math
  (scaleless RMSNorm over 64, eps 1e-5 inside the sqrt, times sigmoid(gate) from qkv, fp32) stored
  bf16 into the [B,1,1024] output. scheduler_metadata is ignored on this path. Launches on
  decode_16/32 full drop from five (pre-attn, prepare_varlen, FA3 main, FA3 combine, norm-gate) to
  three (pre-attn, A, B). k_scale/v_scale are read from device pointers (no .item()). No
  triton.autotune, no host sync, grid and workspace from shapes only.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: incumbent 156.8 us (FA3 main 150.4 + combine 4.4 + prepare 3.0 + norm-gate 1.5) against a 63.2 us floor, ~328 MB of random 128-byte K and V rows at 2.1 TB/s; index preload + 128 KB/SM of cp.async gathers + balanced chunks target <=100 us main (>=3.3 TB/s), shape ~105 us; ~1.4x on the decode_16 load (full is 72% of it)
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: incumbent 298.5 us (FA3 main 292.0) against 126.4 us floor; target <=195 us main, shape ~200 us; full is 81% of the decode_32 load, ~1.45x on it
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: batch 1 is outside the gate; master unpaired FA3 path
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 outside the gate (stage 2 extends to verify)
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched control
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "full-layer decode at batch>1: FA3 flash_fwd main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel"
review: required
status: queued
---

# full-decode-cuda-gather-splitkv

## Change

Stage 1 of a rewrite of full-attention decode, built on master (sliding-fused-triton-decode1-gate). Gate (config/shape only): cfg window == 0 and q_len == 1 and batch > 1 and num_q_heads == 16 and num_kv_heads == 2 and head_dim == 64 and page_size == 1; every other shape (all sliding, full decode_1, every verify, heads dp, page 64) keeps master code byte for byte. On the gated path run master _pre_attention_kernel with PAIRED=False (FP8 q [T,16,64], cache store unchanged), then two new CUDA kernels compiled once per process with torch.utils.cpp_extension.load_inline (-O3, -arch=sm_90a, compiled on the first eager call, never during capture). KERNEL A (split-KV gather attention): a persistent grid of exactly 2*132 CTAs, 256 threads (8 warps), launched with the current torch stream. Work items are fixed C=1024-token chunks of each request: every CTA reads cache_seqlens[0..B-1] (B<=64) into shared memory and computes the per-request chunk prefix sum itself, then walks items i = blockIdx.x, blockIdx.x+gridDim.x, ... over the flat (b, chunk) list, so ragged agentic lengths balance across SMs instead of FA3 splitting each request the same way. Per item: (1) load the item 1024 int32 page_table entries (row = page_table[b, t]) into shared memory in one coalesced pass, so no K/V address waits on a dependent index load inside the loop; (2) stream the item as 16 tiles of 64 tokens, each token one contiguous 128-byte K row (both KV heads, k_cache viewed [rows,128] uint8) and one 128-byte V row, with cp.async.cg 16-byte copies (8 lanes per row) into a 4-stage shared-memory ring (16 KB per stage, 64 KB per CTA, 2 CTAs/SM = 128 KB of gathers in flight per SM), commit/wait_group pipelining so tile n+3 is in flight while tile n is computed; (3) compute with mma.sync m16n8k16 f16 (fp32 accumulate): warp w owns kv head g = w % 2 and token slice (w / 2) * 16 .. +16 of each tile; A = that head 8 query rows (master FP8 q up-converted to f16, rows 8-15 zero), B = the slice K converted fp8 e4m3 -> f16 in registers (cvt.rn.f16x2.e4m3x2), scores times sm_scale*qk_scale_factor*k_scale*log2e, online softmax in the exp2 domain with fp32 m/l per warp (-inf safe: m = -inf uses 0), P rounded to f16 and P @ V_slice with mma.sync (V fragment built from the converted shared V tile), each warp keeping its own running (m, l, acc[8 heads x 64]) across tiles; tokens t >= cache_seqlens[b] are masked to -inf (never dereferenced: their page index is not loaded and the copy is zero-filled via cp.async src-size 0). At item end the four warps of each kv head merge in shared memory and the CTA writes one fp32 partial acc[16 heads,64] plus m[16], l[16] for (b, chunk) into workspace sized B * ceil(page_table.shape[1] / C) (allocated per call from shapes). KERNEL B (combine + epilogue): grid (B, 16), one warp per (request, query head): log-sum-exp merge of that request n_chunks partials (n_chunks recomputed from cache_seqlens in-kernel; empty partials weight 0), times v_scale, then master epilogue math (scaleless RMSNorm over 64, eps 1e-5 inside the sqrt, times sigmoid(gate) from qkv, fp32) stored bf16 into the [B,1,1024] output. scheduler_metadata is ignored on this path. Launches on decode_16/32 full drop from five (pre-attn, prepare_varlen, FA3 main, FA3 combine, norm-gate) to three (pre-attn, A, B). k_scale/v_scale are read from device pointers (no .item()). No triton.autotune, no host sync, grid and workspace from shapes only.

## Expected

- `*layer=full load=decode_16*` faster: incumbent 156.8 us (FA3 main 150.4 + combine 4.4 + prepare 3.0 + norm-gate 1.5) against a 63.2 us floor, ~328 MB of random 128-byte K and V rows at 2.1 TB/s; index preload + 128 KB/SM of cp.async gathers + balanced chunks target <=100 us main (>=3.3 TB/s), shape ~105 us; ~1.4x on the decode_16 load (full is 72% of it)
- `*layer=full load=decode_32*` faster: incumbent 298.5 us (FA3 main 292.0) against 126.4 us floor; target <=195 us main, shape ~200 us; full is 81% of the decode_32 load, ~1.45x on it
- `*layer=full load=decode_1*` flat: batch 1 is outside the gate; master unpaired FA3 path
- `*load=verify_*` flat: q_len 8 outside the gate (stage 2 extends to verify)
- `*layer=sliding*` flat: untouched control

## Risk

Milestones on the replaced kernels GPU time (profile rows A+B vs FA3 main+combine+prepare+paired norm-gate): M1 all 16 tests pass (the batch 4 ctx 80000 full spread mixed dist coherent page 1 test runs this path with ragged lengths) and A+B below the incumbent at decode_16 and decode_32; M2 A <= 110 us at decode_16 and <= 215 us at decode_32 (>=3 TB/s); M3 A <= 85 / 165 us (>=3.8 TB/s). Report achieved KV bandwidth = sum(cache_seqlens)*256 B / A time. Why this should beat both FA3 (2.1 TB/s) and the two Triton tries (decode_layer triton-flash-decode-full-paired-kv 1.6 TB/s, stage 2 slower still): those keep the page-table load on the critical path of every tile and hold at most ~32 KB of gathers per CTA; this design takes the index load out of the loop entirely, holds 4 tiles of K and V in flight per CTA, and balances ragged requests. Risks: load_inline compile time (~60-90 s of the 15 min budget, no prepare on this problem so it happens on the first kernel_fn call; keep the CUDA source minimal, one translation unit); mma.sync fragment layouts for the fp8->f16 converted V operand (implementer may instead ldmatrix.trans the converted f16 V tile staged in shared memory); losing FA3 PDL overlap costs ~1-2 us. If M2 fails, stage 2 tunes C, stages and CTAs/SM before extending to verify (q_len 8, 64 rows per kv head, where fp8 mma m16n8k32 would be needed).

## Milestones

Rewrite stage 1 of `full-layer decode at batch>1: FA3 flash_fwd main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
