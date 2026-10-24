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
review: proceed
status: running
review_at: 2026-10-24T165728Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T165728Z
base_sha: 27b43dcd8b752ab9c2454f7e20fb949c059b9213
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv
worktree: .worktrees/muse_attention_decode/full-verify-cuda-streamk-splitkv
mode: null
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

## Adversarial review

Reviewed 2026-10-24T165728Z by an adversarial subagent (model opus).

**Verdict: proceed.** The stage-2 rewrite is sound and fits the hardware. Stream-K over 64-token tiles balances the agentic spread, FP8 QK mma avoids converting K, and f16 PV is finer than FA3, so it should beat the paired-FA3 verify kernels (150.7/287.4 us). The 120/255 us targets are tight because ex2 and shared-memory traffic grow about 6x over stage 1. Proceed as a post-reset check (risk high, never a scored submit), with the correctness and fallback fixes below built in.

### Justification
The mechanism is plausible. FINDINGS shows full verify_16/32 dominated by FA3 main: 150.7 us of 162.8 at verify_16 and 287.4 of 301.9 at verify_32, against floors of 63.4/126.8 us. Stage 1's kernel A streams the same bytes in 111.4/245.5 us (about 2.95/2.7 TB/s), so a gather-bound verify kernel at <=1.15x of stage 1 would clear M2, and anything under FA3 clears M1. The expected directions match TASK: full layers are about 71%/78% of the verify_16/32 load time (16 x 162.8 vs 46 x 22.8; 16 x 301.9 vs 46 x 29.3). The predicted ~1.18x/1.12x on those two loads gives about +4-5% on the headline geomean. verify_1 (batch 1), every decode shape and every sliding shape are gated out, so 'flat' there is right. Weakest link: q_len 8 makes the kernel no longer purely gather-bound. Per key it does 128 rows of QK+PV (about 84 GFLOP at verify_32). It also does 128 ex2: about 328M MUFU ops at verify_32, roughly 88 us at 16 SFU/clk/SM, and about 44 us at verify_16. Shared memory is the second ceiling. Each warp reads its head's whole K half (4x redundant across the 4 warps of a head) and ldmatrix-loads the whole f16 V half (8 KB per warp, 64 KB per tile per CTA), on top of the convert pass (8 KB read, 16 KB write) and the cp.async fill. That is about 130 KB of shared-memory traffic per 16 KB of global, about 1000+ clk per tile at 128 B/clk. With 2 CTAs/SM this caps at roughly 3.5 TB/s equivalent, close to the 2.95 TB/s the targets assume. So 120/255 us needs MUFU, the mma.sync issue (16 k32 + 32 k16 per warp-tile) and smem all to overlap the gather almost perfectly. M1 and beating FA3 are likely; M3 is optimistic. The headline 'shape ~128 us' at verify_16 should be read as a stretch. Other design notes. (a) The single f16 V buffer works with one extra barrier only because the top-of-loop barrier already guarantees PV(n-1) is done. QK(n) could be issued before the convert barrier to hide it. (b) Registers at __launch_bounds__(256,2) give a 128-reg cap: acc 32 + S 32 + q A-fragments 8 + P 16 + ldmatrix 4-8 + addressing. This is likely 110-128, so the build log must show 0 spills. The spec's fallback of splitting S into 2x32 tokens is the right lever. (c) The m16n8k32 e4m3 mma needs PTX ISA >= 8.4 for sm_90a. A build failure here takes down the whole renamed module, including stage 1's decode kernels (see rules_flags/open_questions).

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv (stage 1, same load_inline module): the rename to _v2 rebuilds decode kernels A/B. If the new verify kernels fail to compile (e.g. e4m3 mma unsupported by the toolchain), load_inline fails as a unit and _EXT=None, which silently reverts decode_16/32 too and shows as a regression on those shapes that the verify change did not cause. If the import-time self-check is not split per path, a verify failure would also disable decode. Read the log tail ('[splitkv] load_inline failed' / 'SELF-CHECK FAILED') before attributing any shape.
- paired-heads-launch-cleanup and paired-heads-launch-cleanup-ab-tune-batch-gate: the incumbent being replaced at verify_16/32 is the PAIRED FA3 call plus _paired_norm_gate_kernel. The change flips the pre-attention kernel to PAIRED=False for these shapes, so the pre-attention kernel's q store changes (64- vs 128-wide FP8 rows, half the q bytes written). That is a small part of the measured shape delta (_pre_attention_kernel is 2.7/4.1 us there), not part of A8+B8. Judge the milestone on A8+B8 vs FA3 main+prepare+combine+paired-norm-gate from top_kernels, as the spec says, not on shape time alone.
- fused-preattention-triton: A8 attends to the 8 new keys that the pre-attention kernel stores in the same stream just before it. Ordering is plain stream order, so there is no confound beyond the PAIRED layout change above. The causal limit t <= cache_seqlens-8+j relies on cache_seqlens = seq_lens + q_len, as reference.generate_input builds it.

### Interference and A/B needs
No companion is warranted. This is a rewrite judged on the replaced kernels' own time against the incumbent and the floor, and the gate is shape-disjoint (only full verify at 1<batch<=64, tp8, page 1 changes). The per-shape top_kernels rows therefore attribute the effect without an A/B. An ab-revert of the head pairing would not be informative: pairing is simply bypassed on the gated shapes, so reverting it changes only the incumbent, which the same run measures through SGLang's paired ratio anyway. A control isolating fp8-QK vs f16-QK or per-CTA vs per-warp V conversion would be a stage-3 tuning question, not something needed to read this stage. The one interpretive hazard is the shared-module build/self-check coupling with stage 1, which is a log check, not an extra run.

### Open questions
- Build isolation: put A8/B8 in a second load_inline module (e.g. muse_full_verify_splitkv_v1) instead of renaming the stage-1 module. A toolchain problem with mma m16n8k32 e4m3 would then cost only the verify path, not the landed decode gain. At minimum, catch the build failure and retry building the stage-1 source alone. Watch the compile time too: stage 1 alone took 41 s inside the 15-minute budget.
- Self-check strength: add several grids (e.g. 3, 7, 61, device), so that segments starting and ending mid-request, single-tile segments and fully masked rows in a last partial tile are all hit. Also add one lens case with cache_seqlens % 64 in 1..7. Keep the 2e-2 rel-L2 bound and the finiteness test. Make clear whether `lens` means seq_lens or cache_seqlens; the 8-token request should have seq_lens 0, cache_seqlens 8. The synthetic builder must also produce 8 positions/cache_loc per request and cu_seqlens_q with stride 8.
- B8 locating c_lo..c_hi: what exact inverse of floor(c*Ttot/grid) is used? Recommend a shared device helper plus a +/-1 correction loop on exact start(c) comparisons.
- Report from the build log: A8 registers and spills at __launch_bounds__(256,2); and from the profile: achieved KV bandwidth (sum(cache_seqlens)*256 B / A8 time) at verify_16/32 vs stage 1's 2.95/2.7 TB/s. If A8 is clearly above 1.15x stage 1 decode A with no spills, the next stage should first try moving QK before the V-convert barrier and cutting the 4x redundant smem reads, before stream-K is ported back to decode.
- Dispatch: risk is high, so per the standing instruction this goes out as a check after the reset, not as a scored submit.

