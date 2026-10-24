---
problem: muse_attention_decode
slug: full-verify-cuda-streamk-splitkv
created: 2026-10-24T165242Z
base: master
area: FA3 call (full-layer verify gather)
change: >
  Stage 2 of the full-attention rewrite, built on master (stage 1 full-decode-cuda-gather-splitkv
  landed). Extend the CUDA split-KV path to full-layer VERIFY (q_len 8, causal inside the block)
  at 1 < batch <= 64, tp8 (16 q heads, 2 kv heads), head_dim 64, page 1. Gate (config only):
  _splitkv_verify_gate(cfg) = _EXT is not None and window == 0 and q_len == 8 and 1 < batch <= 64
  and hq == 16 and hkv == 2 and head_dim == 64 and page_size == 1; the paired predicate becomes
  paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv and not splitkv_verify, so
  master _pre_attention_kernel runs with PAIRED=False (FP8 q [B*8, 16, 64]) and kernel_fn returns
  _full_verify_splitkv(...). Decode kernels A/B and every other shape stay byte for byte. Two NEW
  kernels in the same load_inline module (rename it muse_full_decode_splitkv_v2 so no stale cache
  is reused). KERNEL A8 splitkv_verify_kernel: persistent grid 2 x SM count, 256 threads,
  __launch_bounds__(256, 2). Work split is STREAM-K over 64-token tiles, not 1024-token chunks:
  every CTA computes in smem tiles_b = ceil(cache_seqlens[b] / 64) and its prefix TP[b] (B <= 64),
  Ttot = TP[B]; CTA c owns the flat tile range [c*Ttot/grid, (c+1)*Ttot/grid) (int64 math), which
  may span several requests; it emits one partial per request segment it touches, at slot c + b
  (unique because ranges are monotone; slots <= grid + B), so the workspace (ws_acc f32 [grid + B,
  8, 16, 64], ws_ml f32 [grid + B, 2, 128], allocated per call from grid and batch) is ~10 MB
  instead of the ~84 MB that 1024-chunk partials of 8x16 rows would cost at verify_32. Reuse stage
  1 machinery: page indices for the next 16 tiles of the CTA range prefetched into a
  double-buffered smem block with 4-byte cp.async (zero-filled past each request end), 4-stage
  cp.async.cg ring of 64 tokens x (128 B K + 128 B V) with the token-XOR swizzle, the pipeline
  running across segment boundaries. The request FP8 query block (8 tokens x 16 heads x 64 = 8 KB)
  is fetched into a double-buffered smem slot when a segment starts. Compute, the q_len 8 queries
  on M: rows of kv head g are its 8 heads x 8 block tokens = 64 rows = 4 m-tiles of 16; warp w
  owns kv head g = w & 1 and m-tile mt = w >> 1, rows i < 8 = (head g*8 + 2*mt, token i) and rows
  i >= 8 = (head g*8 + 2*mt + 1, token i - 8), so both rows a lane holds in the C fragment belong
  to the same block token (quad) and share one causal limit; each warp runs all 64 tokens of the
  tile for its 16 rows, so no cross-warp merge is needed. QK^T with
  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32: A = the FP8 query bytes straight from smem
  (8 regs per lane, loaded once per segment, no conversion), B = raw FP8 K bytes of kv head g from
  the ring (b0 = token quad, dims 4*qi..+3 of the k-step, b1 = +16; one 32-bit read each), 8
  n-blocks x 2 k-steps = 16 mma per tile per warp, no K cvt. P@V stays f16 (finer than FA3 e4m3
  P): after the tile lands, all 256 threads convert its V half-tiles once (cvt.rn.f16x2.e4m3x2)
  into a 16 KB f16 smem tile (64 tokens x 128 dims, swizzled for ldmatrix), one extra
  __syncthreads, then each warp loads B fragments with ldmatrix.x4.trans and runs m16n8k16 (P f16
  from the fp32 exp2 scores as A): 8 dim-blocks x 4 token k-steps = 32 mma per tile per warp.
  Converting V once per CTA instead of per warp avoids a 4x redundant cvt (~150 us of ALU at
  verify_32). Online softmax in the exp2 domain exactly as stage 1 (scale =
  sm_scale*qk_scale_factor*log2e*k_scale from the device pointer, -inf safe). Masking: key t of
  request b is valid for block token j iff t <= cache_seqlens[b] - 8 + j (causal, includes the
  block own new keys already stored by pre-attention) and t < cache_seqlens[b]; only the request
  last tile needs the per-row test. At segment end each warp stores its 16 rows (acc unnormalized,
  m, l quad-reduced) to the slot. KERNEL B8 splitkv_verify_combine: grid (B*8, 16), 128 threads
  per (request, token, head): recompute TP and Ttot from cache_seqlens, find the CTA range
  c_lo..c_hi overlapping [TP[b], TP[b+1]), log-sum-exp merge of slots c + b, times v_scale, then
  the stage 1 epilogue (scaleless RMSNorm over 64, eps inside the sqrt, times sigmoid(gate) from
  qkv row b*8 + j), bf16 into [B, 8, 1024]. Extend the import-time self-check with a q_len 8
  synthetic case (lens [9000, 1025, 70, 4097, 1023, 8] incl. a request whose length is only the
  block, kv_scale 1 and 0.25, the device grid and a 3-CTA grid so CTAs span several requests)
  against an FP32 causal reference; on failure disable only the verify path. No autotune, no host
  sync, grid and workspace from shapes and the device only; scheduler_metadata (None on verify) is
  ignored.
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: incumbent 162.8 us shape, kernels FA3 main 150.7 + combine 6.4 + prepare 3.1 + paired norm-gate 2.5 against a 63.4 us floor; the same ~328 MB gather that stage 1 kernel A streams in 111.4 us at decode_16 (2.95 TB/s), now with 64 rows per kv head (fp8 QK mma, once-per-CTA V conversion) so the extra math hides under the gather; target A <= 120 us, B <= 4 us, shape ~128 us; full is 71% of the verify_16 load, ~1.18x on it
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: incumbent 301.9 us shape (FA3 main 287.4) against 126.8 us floor; stage 1 decode_32 kernel A is 245.5 us on the same bytes; target A <= 255 us, shape ~262 us; full is 78% of the verify_32 load, ~1.12x on it
  - shapes: "*layer=full load=verify_1*"
    direction: flat
    why: batch 1 outside the gate (unpaired FA3)
  - shapes: "*load=decode_*"
    direction: flat
    why: decode kernels A/B untouched (module renamed only)
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched control
risk: high
depends_on: full-decode-cuda-gather-splitkv
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "full-layer decode at batch>1: FA3 flash_fwd main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel"
review: required
status: queued
---

# full-verify-cuda-streamk-splitkv

## Change

Stage 2 of the full-attention rewrite, built on master (stage 1 full-decode-cuda-gather-splitkv landed). Extend the CUDA split-KV path to full-layer VERIFY (q_len 8, causal inside the block) at 1 < batch <= 64, tp8 (16 q heads, 2 kv heads), head_dim 64, page 1. Gate (config only): _splitkv_verify_gate(cfg) = _EXT is not None and window == 0 and q_len == 8 and 1 < batch <= 64 and hq == 16 and hkv == 2 and head_dim == 64 and page_size == 1; the paired predicate becomes paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv and not splitkv_verify, so master _pre_attention_kernel runs with PAIRED=False (FP8 q [B*8, 16, 64]) and kernel_fn returns _full_verify_splitkv(...). Decode kernels A/B and every other shape stay byte for byte. Two NEW kernels in the same load_inline module (rename it muse_full_decode_splitkv_v2 so no stale cache is reused). KERNEL A8 splitkv_verify_kernel: persistent grid 2 x SM count, 256 threads, __launch_bounds__(256, 2). Work split is STREAM-K over 64-token tiles, not 1024-token chunks: every CTA computes in smem tiles_b = ceil(cache_seqlens[b] / 64) and its prefix TP[b] (B <= 64), Ttot = TP[B]; CTA c owns the flat tile range [c*Ttot/grid, (c+1)*Ttot/grid) (int64 math), which may span several requests; it emits one partial per request segment it touches, at slot c + b (unique because ranges are monotone; slots <= grid + B), so the workspace (ws_acc f32 [grid + B, 8, 16, 64], ws_ml f32 [grid + B, 2, 128], allocated per call from grid and batch) is ~10 MB instead of the ~84 MB that 1024-chunk partials of 8x16 rows would cost at verify_32. Reuse stage 1 machinery: page indices for the next 16 tiles of the CTA range prefetched into a double-buffered smem block with 4-byte cp.async (zero-filled past each request end), 4-stage cp.async.cg ring of 64 tokens x (128 B K + 128 B V) with the token-XOR swizzle, the pipeline running across segment boundaries. The request FP8 query block (8 tokens x 16 heads x 64 = 8 KB) is fetched into a double-buffered smem slot when a segment starts. Compute, the q_len 8 queries on M: rows of kv head g are its 8 heads x 8 block tokens = 64 rows = 4 m-tiles of 16; warp w owns kv head g = w & 1 and m-tile mt = w >> 1, rows i < 8 = (head g*8 + 2*mt, token i) and rows i >= 8 = (head g*8 + 2*mt + 1, token i - 8), so both rows a lane holds in the C fragment belong to the same block token (quad) and share one causal limit; each warp runs all 64 tokens of the tile for its 16 rows, so no cross-warp merge is needed. QK^T with mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32: A = the FP8 query bytes straight from smem (8 regs per lane, loaded once per segment, no conversion), B = raw FP8 K bytes of kv head g from the ring (b0 = token quad, dims 4*qi..+3 of the k-step, b1 = +16; one 32-bit read each), 8 n-blocks x 2 k-steps = 16 mma per tile per warp, no K cvt. P@V stays f16 (finer than FA3 e4m3 P): after the tile lands, all 256 threads convert its V half-tiles once (cvt.rn.f16x2.e4m3x2) into a 16 KB f16 smem tile (64 tokens x 128 dims, swizzled for ldmatrix), one extra __syncthreads, then each warp loads B fragments with ldmatrix.x4.trans and runs m16n8k16 (P f16 from the fp32 exp2 scores as A): 8 dim-blocks x 4 token k-steps = 32 mma per tile per warp. Converting V once per CTA instead of per warp avoids a 4x redundant cvt (~150 us of ALU at verify_32). Online softmax in the exp2 domain exactly as stage 1 (scale = sm_scale*qk_scale_factor*log2e*k_scale from the device pointer, -inf safe). Masking: key t of request b is valid for block token j iff t <= cache_seqlens[b] - 8 + j (causal, includes the block own new keys already stored by pre-attention) and t < cache_seqlens[b]; only the request last tile needs the per-row test. At segment end each warp stores its 16 rows (acc unnormalized, m, l quad-reduced) to the slot. KERNEL B8 splitkv_verify_combine: grid (B*8, 16), 128 threads per (request, token, head): recompute TP and Ttot from cache_seqlens, find the CTA range c_lo..c_hi overlapping [TP[b], TP[b+1]), log-sum-exp merge of slots c + b, times v_scale, then the stage 1 epilogue (scaleless RMSNorm over 64, eps inside the sqrt, times sigmoid(gate) from qkv row b*8 + j), bf16 into [B, 8, 1024]. Extend the import-time self-check with a q_len 8 synthetic case (lens [9000, 1025, 70, 4097, 1023, 8] incl. a request whose length is only the block, kv_scale 1 and 0.25, the device grid and a 3-CTA grid so CTAs span several requests) against an FP32 causal reference; on failure disable only the verify path. No autotune, no host sync, grid and workspace from shapes and the device only; scheduler_metadata (None on verify) is ignored.

## Expected

- `*layer=full load=verify_16*` faster: incumbent 162.8 us shape, kernels FA3 main 150.7 + combine 6.4 + prepare 3.1 + paired norm-gate 2.5 against a 63.4 us floor; the same ~328 MB gather that stage 1 kernel A streams in 111.4 us at decode_16 (2.95 TB/s), now with 64 rows per kv head (fp8 QK mma, once-per-CTA V conversion) so the extra math hides under the gather; target A <= 120 us, B <= 4 us, shape ~128 us; full is 71% of the verify_16 load, ~1.18x on it
- `*layer=full load=verify_32*` faster: incumbent 301.9 us shape (FA3 main 287.4) against 126.8 us floor; stage 1 decode_32 kernel A is 245.5 us on the same bytes; target A <= 255 us, shape ~262 us; full is 78% of the verify_32 load, ~1.12x on it
- `*layer=full load=verify_1*` flat: batch 1 outside the gate (unpaired FA3)
- `*load=decode_*` flat: decode kernels A/B untouched (module renamed only)
- `*layer=sliding*` flat: untouched control

## Risk

Milestones on the replaced kernels (A8 + B8 vs FA3 main + prepare + combine + _paired_norm_gate_kernel): M1 all 16 tests pass (batch 2 ctx 100000 full q_len 8 page 1 runs this path, ~1563 tiles per request; everything else on verify is page 64 or dp and stays on FA3) and A8+B8 below the incumbent at verify_16/32; M2 A8 <= 1.15x stage 1 decode A on the same batch (<= 128 / <= 282 us); M3 A8 <= 120 / 255 us. Report achieved KV bandwidth = sum(cache_seqlens)*256 B / A8 time, and A8 registers/spills from the build log (-Xptxas -v is fine). Compute check: verify_32 is ~84 GFLOP (16 rows x 8 tokens x 64 x 2 x 2 per key) split half fp8 QK, half f16 PV; at mma.sync rates this is ~100-140 us of tensor time that must overlap the ~245 us gather, which is why the 2-CTA/SM occupancy, fp8 QK (no K cvt) and the single cooperative V conversion matter. Risks: m16n8k32 e4m3 fragment layout (sm_89+ PTX; FA3 uses fp8 the same way, accumulation precision is ample for 64-term dots); register pressure (acc 32 + q 8 + S 32 + V/ldmatrix regs; spills would show in the time; if so split S into two halves of 32 tokens); smem ~64 KB ring + 16 KB f16 V + 16 KB q + 8 KB idx + seqlens = ~105 KB per CTA, 2 CTAs/SM fit in 227 KB; stream-K segment bookkeeping (a CTA range can start mid-request and end mid-request; empty ranges at small Ttot must still write nothing and B8 must give empty slots weight 0). Losing FA3 PDL overlap ~1-2 us. If M2 fails, stage 3 tunes stages / CTAs per SM / P precision before porting stream-K back to decode.

## Milestones

Rewrite stage 2 of `full-layer decode at batch>1: FA3 flash_fwd main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
