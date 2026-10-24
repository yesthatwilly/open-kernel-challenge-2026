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
review: proceed
status: running
review_at: 2026-10-24T140601Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T140601Z
base_sha: 0e3d04d3405a65d8916f21e7fac2e822929939e7
base_blob: 387dcb6a0e6137b74f7f65b048dce5a3fcbe2008
branch: hyp/muse_attention_decode/full-decode-cuda-gather-splitkv
worktree: .worktrees/muse_attention_decode/full-decode-cuda-gather-splitkv
mode: null
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

## Adversarial review

Reviewed 2026-10-24T140601Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite, aimed at the one place left with real headroom: full-layer decode at batch 16/32, which is 72-81% of those loads and runs at 2.1 TB/s against a 4.8 TB/s floor. load_inline is explicitly allowed. The weakest link is the 3.3 TB/s target on uniformly random 128-byte rows, and stage 1's milestone will measure it directly. Fix three robustness gaps before claiming: when the kernel compiles, a batch<=64 gate, and PAIRED=False on the gated path.

### Justification
Profile support is strong. At decode_16 the full layer takes 156.8 us (FA3 main 150.4, combine 4.4, prepare 3.0) against a 63.2 us floor. At decode_32 it takes 298.5 us (main 292.0) against 126.4 us. Both are ~2.4x off the floor and these are the largest absolute rows in the problem. The byte count checks out: 16 x ~80K tokens x (128 B K + 128 B V) = ~328 MB, 2.1 TB/s at 156 us. reference.generate_input builds the page table from a torch.randperm over batch*max_pages, so every token's K row and V row is an independent random 128-byte line spread over ~350 MB. The kernel design itself is reasonable for H200. Kernel A has 264 persistent CTAs with 64 KB of cp.async ring each, ~128 KB in flight per SM; Little's law needs only ~20 KB/SM to sustain 3.3 TB/s at ~800 ns latency, so in-flight depth is ample. The page-index preload removes the dependent load, and 1024-token chunks over a flat (b, chunk) list balance the ragged lengths (~1280 items at decode_16 = 4.85 waves, ~97% balanced). Compute is small: mma.sync with 8 of 16 rows wasted is still far below the tensor-core roof. The fp8->f16 conversion of all 328 M K+V elements is ~164 M cvt.e4m3x2 instructions. At ~16/clk/SM that is ~40 us of ALU, which must overlap the memory stream, so it is not free at a 100 us target. Weakest link: whether FA3's 2.1 TB/s is limited by latency and in-flight depth (which this design fixes) or by HBM efficiency on uniformly random 128-byte lines plus TLB reach over a ~350 MB randomly scattered cache (which no kernel fixes). FA3 sm90 already pipelines its paged K/V loads, so the claim that FA3 'keeps the page-table load on the critical path of every tile' is unverified. Both earlier Triton gather kernels topped out at 1.6 TB/s, so there is no in-repo evidence of anything above 2.1 TB/s on this access pattern. The M1/M2 bandwidth readout (sum(cache_seqlens)*256 B / A time) is the right test. The per-load arithmetic is overstated. At the stated shape targets (105 us and 200 us), decode_16 goes from 952+2509=3461 to 952+1680=2632 (1.315x, not ~1.4x), and decode_32 from 1145+4776=5921 to 1145+3200=4345 (1.363x, not ~1.45x). That would still be ~1.10x on the 6-load geomean, about 1.42x headline. Directions are all consistent with TASK.md: decode_1, verify and sliding are outside the gate and byte-identical, so flat.

### Confounders on the current kernel
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / -ab-tune-batch-gate: these build the incumbent this rewrite replaces on the gated shapes (paired FA3 plus _paired_norm_gate_kernel). They do not confound: the new kernel keeps the same one-128-byte-K-row-per-token gather, so the measured delta is purely replacing paired FA3 with kernels A and B. One implementation hazard: the paired predicate must become (window==0 and hkv%2==0 and batch>1 and not gated). Otherwise the pre-attention kernel writes the padded [T,16,128] q and kernel A reads it as [T,16,64]. Full verify_16/32 must stay paired.
- fused-preattention-triton: the gated path reuses _pre_attention_kernel unchanged with PAIRED=False, the same launch master uses on unpaired paths. It is a constant ~1.3-2 us with no interaction, except that the unpaired q write is slightly smaller than the paired one, a sub-0.5 us effect.
- sliding-fused-triton-decode1-gate: disjoint (sliding batch-1 only). No shared launches or memory, and cold L2 on every replay removes cache interaction.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite, judged on kernels A+B's own profile time against the 156.8/298.5 us incumbent and the 63.2/126.4 us floors. Those times are directly readable from top_kernels, because load_inline kernels appear under their own names. Reverting pairing (ab-revert) would only compare against a slower unpaired FA3, which answers nothing new. A control that splits off the index preload or the chunk balancing would need a second full CUDA implementation; that is stage-2 tuning (C, stage count, CTAs/SM), not a disambiguator. With only 2 attention checks left today, spend one on this stage and keep one for the stage-2 fix.

### Open questions
- Is FA3's 2.1 TB/s latency-bound or limited by DRAM efficiency on random rows? If kernel A also stalls near 2.0-2.5 TB/s with a deep ring and balanced chunks, random 128-byte HBM3e efficiency plus TLB reach over the ~350 MB randomly permuted cache is the ceiling, and the rewrite should stop. Report achieved bandwidth and, if the profile allows, sector and L2-miss ratios.
- Conversion cost: should QK use the native FP8 mma.sync m16n8k32 e4m3 path (sm_89+), with q already FP8, so K needs no cvt? That roughly halves conversion ALU. V still needs widening, or an FP8 P@V with P rounded to e4m3 as FA3 does.
- Only one test exercises the gated path: batch 4, ctx 80000, spread mixed, coherent, page 1, with lengths up to ~4x ctx, so ~313 chunks per request and a larger workspace. kv_scale != 1 never reaches kernel A in tests. Is a local CPU or emulated check of the scale path worth adding before stage 2?
- Kernel B: one warp per (request, head) serially merging ~80 partials (decode_16) to ~85 (decode_32), with C=1024, reads ~5-11 MB in total. Check that it stays at or below FA3 combine plus norm-gate (~6 us). If not, use two-level reduction or larger C in stage 2.
- Losing FA3's PDL overlap between pre-attention, main and combine costs ~1-2 us per shape. Consider cudaLaunchAttributeProgrammaticStreamSerialization on A and B in a later stage.

