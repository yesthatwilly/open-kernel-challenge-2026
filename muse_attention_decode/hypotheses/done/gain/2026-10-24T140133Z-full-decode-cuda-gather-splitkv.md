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
status: done
review_at: 2026-10-24T140601Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T140601Z
base_sha: 0e3d04d3405a65d8916f21e7fac2e822929939e7
base_blob: 387dcb6a0e6137b74f7f65b048dce5a3fcbe2008
branch: hyp/muse_attention_decode/full-decode-cuda-gather-splitkv
worktree: .worktrees/muse_attention_decode/full-decode-cuda-gather-splitkv
mode: check
implemented_at: 2026-10-24T141722Z
impl_commit: 02d7023484fe904ef0028896853d8689cd3e9c73
submission_sha256: 1645e5cccd6de02aa4bfe75e1b33677bc2f4d435f8cc175db3f41148a9cfa6af
norm_sha256: f04a8ab8b2046f0895a5ff82df943dd7e1dc7a2efd5a027132457485546a6f7d
submission_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
lint_warnings: 0
attempt_file: 2026-10-24T141722Z-check-full-decode-cuda-gather-splitkv.md
prepared_at: 2026-10-24T141722Z
job: 82085a4e82154393
dispatched_at: 2026-10-24T141733Z
outcome: gain
verdict_detail: gain
speedup: 1.3467624733171553
speedup_ci: [1.3265610362173978, 1.3669639104169125]
harvested_at: 2026-10-24T144809Z
best_at_harvest: 1.2930643047593426
promotion: merged
promotion_commit: 3693fb89b6ce898de437c29d4859b90c33277309
promotion_attempt: 2026-10-24T163458Z-submit-full-decode-cuda-gather-splitkv.md
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


## Implementation notes

Worktree `.worktrees/muse_attention_decode/full-decode-cuda-gather-splitkv`, file `muse_attention_decode/submission.py` only (+596/-2). Not compiled or run here (no GPU/torch/nvcc on this node); self-reviewed twice for indexing, ragged-length masking, key coverage, softmax numerics and the FP8 descale.

- **Gate** `_splitkv_gate(cfg)` (config only): `_EXT is not None and window == 0 and q_len == 1 and 1 < batch <= 64 and hq == 16 and hkv == 2 and head_dim == 64 and page_size == 1`. Review fix (1): `paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv`, so the master `_pre_attention_kernel` runs with PAIRED=False on the gated path (q FP8 [T,16,64]); right after it, `kernel_fn` returns `_full_decode_splitkv(...)`. Every other shape runs master code unchanged (full verify_16/32 stay paired).
- **Kernel A** `splitkv_attn_kernel`: follows the spec. Grid = 2 x `multi_processor_count`, read from the device (review fix (2)); 256 threads, `__launch_bounds__(256, 2)`; 93,200 B of dynamic smem (64 KB ring, 2x4 KB page-index buffers, 2x1 KB q buffers, 16 KB merge, seqlens/prefix). The in-kernel chunk prefix sum is computed per CTA; items are request-major and strided by grid. Implementation details the spec left open:
  - The tile pipeline runs across item boundaries without draining. Item k+1's 1024 page indices (4-byte `cp.async.ca`, zero-filled past the length) and its query (1 KB) ride in tile (k,0)'s commit group, double-buffered by k & 1. One commit group per tile and `wait_group 2`, with a barrier before each issue, so the hazards are covered. I traced each reuse distance in the code.
  - 16-byte chunks of each 128-byte smem row are XOR-swizzled by `token & 7`. K fragment reads (32-bit) are conflict-free; V reads (64-bit) are 2-way, the minimum for 256 B.
  - QK^T: the k (dim) order is permuted identically for the Q A-fragment and the K B-fragment, so each lane reads 4 contiguous K bytes: k = 2qi + j maps to dim 4qi + j, and k = 8 + 2qi + j maps to dim 4qi + 2 + j. One `cvt.rn.f16x2.e4m3x2` pair per word.
  - P@V: V is the B operand. The output column n of dim-block nb is physical dim n*8 + nb, so each lane reads 8 contiguous bytes per token (4 tokens) and interleaves token pairs with `__byte_perm` (0x5140 / 0x7362) before the cvt. The merge writes the columns back in physical order. This replaces the ldmatrix.trans option the risk section mentions.
  - P is f16 (finer than FA3's e4m3 P), l is summed from the fp32 p, and the scale is sm_scale*qk_scale_factor*log2e*k_scale (k_scale read on device). Masked positions give -inf, and the m = -inf case uses 0, as master `_online_update` does. Warps whose slice is fully masked for a whole item (short last chunks) carry m = -inf and get weight 0 in the merge.
  - The workspace is `ws_acc` f32 [B*max_chunks,16,64] and `ws_ml` f32 [B*max_chunks,2,16], with max_chunks = ceil(page_table.shape[1]/1024). Both are allocated per call from shapes.
- **Kernel B** `splitkv_combine_kernel`: grid (B,16). Deviation: 128 threads (4 warps) per (request, head) instead of one warp, following the review's open question on the serial ~80-partial chain. Pass 1 takes the block max of m over the chunks. Pass 2 has each warp accumulate the exp2-weighted acc/l of every 4th chunk (independent loads, unroll 4), followed by an smem reduce. Warp 0 then divides by l, multiplies by v_scale (device pointer), applies RMSNorm (eps inside the sqrt, fp32) and multiplies by sigmoid(gate) from qkv, storing bf16x2 into [B,1,1024]. scheduler_metadata is ignored on this path.
- **Compile** (review fix (3)): `load_inline(name="muse_full_decode_splitkv_v1", extra_cuda_cflags=["-O3","-arch=sm_90a"])` runs at module import, never inside kernel_fn or capture. It prints `[splitkv] load_inline compiled in X s`. `cudaFuncSetAttribute` for the 93 KB smem runs on the first launch, which is the eager self-check. If the build fails, the error is printed and `_EXT = None`, so the master path is used everywhere.
- **Self-check** (review fix (4)): runs once, eagerly, at import and only on synthetic data, so the timed path has no host sync. Inputs: lengths [9000,1025,70,4097,1023] (partial chunks, fully masked warps, an exact 1024 chunk), scattered pages, odd page-table stride. Three configurations: kv_scale 1 with the device grid, kv_scale 0.25 with a 3-CTA grid, and kv_scale 1 with a 3-CTA grid. The 3-CTA grid gives several items per CTA, which exercises the cross-item pipeline and the meta double-buffering. Each run's `kernel_fn` output is compared to an FP32 torch reference (FP8 queries, the cache as written), and rel L2 is printed. If any error is >= 2e-2, the output is non-finite, or the check raises, it prints `SELF-CHECK FAILED`, disables the path (`_EXT = None`) and the run continues on master. **Read the log tail**: a flat result plus that line means the kernel is wrong, not slow. The self-check runs against an FP32 reference rather than FA3, because FA3's e4m3 P would blur a ~1e-3 comparison.
- **Uncertainties:** (a) The code has never been compiled. Likely first-compile issues: lambda captures in device code, `"n"` asm constraints from a template parameter, the `-arch=sm_90a` interplay with torch's own arch flags (torch skips adding its own when a flag containing "arch" is present). (b) Register pressure under `(256,2)` (≤128 regs: acc 32, scores 8, q 8, V words 8, ...). Spills would show in the kernel time. (c) Only one judge test reaches this path (batch 4, ctx 80000, spread mixed, coherent, page 1). kv_scale != 1 is covered only by the self-check. (d) Losing FA3's PDL overlap (~1-2 us) remains, as the review noted.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/full-decode-cuda-gather-splitkv diff`:

- [x] CUDA-graph capturable: kernel_fn and everything it reaches has no `.item()`/`.cpu()`/`synchronize()`/tensor prints. Grid = 2 x SM count (device property), and the workspace sizes come from `cfg["batch"]` and `page_table.shape[1]`. Kernels launch on `at::cuda::getCurrentCUDAStream()`. k_scale/v_scale/cache_seqlens are read only on device. The only branch is the config predicate `_splitkv_gate`. The host syncs (`float(...)`, `bool(...)`) live only in the import-time self-check on synthetic data.
- [x] Nothing derived from `data` is cached across calls. `_SMS` caches the device SM count and `_EXT` the compiled module; the workspaces are allocated per call.
- [x] Weights: n/a. This problem has no weights or prepare().
- [x] Activations BF16 where SGLang keeps BF16; KV cache stays FP8 (read raw, widened exactly to f16); queries are master's FP8 E4M3, no coarser; P is f16 (finer than FA3's e4m3); accumulation and the epilogue are fp32; the output is bf16.
- [x] Attention sees every key: chunks cover [0, cache_seqlens[b]) completely (n_chunks = ceil(seq/1024), 16 tiles x 64 tokens each), and only positions >= cache_seqlens[b] are masked. There is no top-k, sparse, block-skip or weight-based skipping, and empty partials get weight 0 only when they hold no valid key.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. `load_inline` (nvcc) is the rules' stated way to write native CUDA. The self-check builds its own synthetic reference and does not import reference.py.
- [x] All work happens inside the timed call; no output caching or replay. The import-time self-check output is discarded, and every timed call computes from its own inputs.

okc_lint output:

```
CLEAN: submission.py
```

`python3 -I -m py_compile`: OK (worktree `__pycache__` removed).

## Result

- job `82085a4e82154393` (check), status done, passed True, tests 16/16
- speedup 1.3468x, 95% CI [1.3266, 1.3670], score 2796.5584686387806 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.2930643047593426, "ci": [1.2736683401879525, 1.3211607975952195], "job": "295727fc68e14093"}; base result: muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-gate.result.json
- **outcome: gain** (gain)
- files: `muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json`, `muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.patch`; branch `hyp/muse_attention_decode/full-decode-cuda-gather-splitkv`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.12 | 13.88 | 0.9551 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 26.99 | 27.93 | 0.9809 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 20.66 | 20.84 | 0.9939 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 156.82 | 123.02 | 1.2767 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.89 | 25.2 | 0.9889 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 298.53 | 257.75 | 1.1616 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.01 | 17.63 | 0.991 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 29.67 | 30.29 | 0.9963 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.59 | 22.77 | 0.9901 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.06 | 162.83 | 1.0042 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.61 | 29.35 | 0.9812 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.16 | 301.91 | 1.0058 | flat |

gained on full decode_16, decode_32; lost on sliding decode_1. Expected hit: 4/5. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `full-layer decode at batch>1: FA3 flash_fwd main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-decode-cuda-gather-splitkv --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.
- candidate for promotion; stack further changes with `depends_on` once it lands.

