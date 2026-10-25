---
problem: muse_decode_layer
slug: full-decode-splitkv-kv-bw-3p5
created: 2026-10-25T001251Z
base: master
area: "FA3 call (full decode, batch > 1): splitkv_attn_kernel bytes in flight"
change: >
  Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on
  the restacked code from full-decode-splitkv-restack-on-pdl. Kernel A (splitkv_attn_kernel) is
  93-96% of the replaced time and reaches only 2.9 TB/s at B=16 (112.0 us) and 2.6 TB/s at B=32
  (249.5 us) vs a KV floor of ~63/126 us; it has STAGES=4 x 16 KB ring at 2 CTAs/SM (~96 KB in
  flight per SM) over a persistent 2xSM grid of CHUNK=1024-token items, and at B=32 the
  agentic-spread tail leaves SMs idle. Change only kernel A and its host grid/chunk constants: (a)
  deeper ring: STAGES 6 if per-CTA smem (ring + idx [2][CHUNK] + q + acc/ml) stays <= ~113 KB so 2
  CTAs/SM hold, else STAGES 5; keep the cp.async.cg 16-byte form and the wait_group<STAGES-2>
  logic; (b) smaller chunk where the tail dominates: make CHUNK a template parameter with
  instantiations 1024 and 512 and pick 512 when total KV tokens / (2 x SMs) is below ~3 chunks of
  1024 or batch >= 24 (host-side from config and cache_seqlens-free sizes only: decide from batch,
  no host sync), with the combine's split count sized accordingly (S per sequence doubles; combine
  reads ws by item exactly as now); (c) issue the next item's page-index prefetch two items ahead
  instead of one. Combine kernel, gate (_splitkv_gate) and everything else unchanged. Self-check:
  rerun the existing _splitkv_self_check cases (B=5 and B=33, kv_scale 1.0/0.25, device grid and
  grid 3) for every (STAGES, CHUNK) instantiation used; any failure falls back to the stage-1
  configuration (STAGES 4, CHUNK 1024), not to FA3. Print [splitkv-s2] stages/chunk/occupancy and
  kernel-A CUDA-event time + effective TB/s at B=16 and 32 for old and new configs.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "kernel A 112.0 us at 2.9 TB/s -> ~90-95 us at >=3.5 TB/s (M3 <=95 us, floor ~63): ~17-22 us off the restacked shape"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "kernel A 249.5 us at 2.6 TB/s -> ~185-190 us at >=3.5 TB/s with the 512 chunk draining the agentic tail (M3 <=189 us, floor ~126): ~55-60 us off"
  - shapes: "*layer=sliding *"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=verify_*"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: "batch 1 outside gate: control"
risk: medium
depends_on: full-decode-splitkv-restack-on-pdl
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: paired FA3 flash_fwd + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full decode, 1 < batch <= 64 (now splitkv_attn_kernel + splitkv_combine_kernel)
review: proceed
status: queued
review_at: 2026-10-25T001534Z
review_companions: null
---

# full-decode-splitkv-kv-bw-3p5

## Change

Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on the restacked code from full-decode-splitkv-restack-on-pdl. Kernel A (splitkv_attn_kernel) is 93-96% of the replaced time and reaches only 2.9 TB/s at B=16 (112.0 us) and 2.6 TB/s at B=32 (249.5 us) vs a KV floor of ~63/126 us; it has STAGES=4 x 16 KB ring at 2 CTAs/SM (~96 KB in flight per SM) over a persistent 2xSM grid of CHUNK=1024-token items, and at B=32 the agentic-spread tail leaves SMs idle. Change only kernel A and its host grid/chunk constants: (a) deeper ring: STAGES 6 if per-CTA smem (ring + idx [2][CHUNK] + q + acc/ml) stays <= ~113 KB so 2 CTAs/SM hold, else STAGES 5; keep the cp.async.cg 16-byte form and the wait_group<STAGES-2> logic; (b) smaller chunk where the tail dominates: make CHUNK a template parameter with instantiations 1024 and 512 and pick 512 when total KV tokens / (2 x SMs) is below ~3 chunks of 1024 or batch >= 24 (host-side from config and cache_seqlens-free sizes only: decide from batch, no host sync), with the combine's split count sized accordingly (S per sequence doubles; combine reads ws by item exactly as now); (c) issue the next item's page-index prefetch two items ahead instead of one. Combine kernel, gate (_splitkv_gate) and everything else unchanged. Self-check: rerun the existing _splitkv_self_check cases (B=5 and B=33, kv_scale 1.0/0.25, device grid and grid 3) for every (STAGES, CHUNK) instantiation used; any failure falls back to the stage-1 configuration (STAGES 4, CHUNK 1024), not to FA3. Print [splitkv-s2] stages/chunk/occupancy and kernel-A CUDA-event time + effective TB/s at B=16 and 32 for old and new configs.

## Expected

- `*layer=full load=decode_16 *` faster: kernel A 112.0 us at 2.9 TB/s -> ~90-95 us at >=3.5 TB/s (M3 <=95 us, floor ~63): ~17-22 us off the restacked shape
- `*layer=full load=decode_32 *` faster: kernel A 249.5 us at 2.6 TB/s -> ~185-190 us at >=3.5 TB/s with the 512 chunk draining the agentic tail (M3 <=189 us, floor ~126): ~55-60 us off
- `*layer=sliding *` flat: outside gate: control
- `*load=verify_*` flat: outside gate: control
- `*load=decode_1 *` flat: batch 1 outside gate: control

## Risk

Medium: 6 stages may drop residency to 1 CTA/SM (print occupancy; the fallback rule picks 5). A 512 chunk doubles combine work (8-9 us now); if combine grows past ~12 us, keep 1024 at B=16. Must not start until full-decode-splitkv-restack-on-pdl lands (it needs a hand merge); build on whatever that run measured.

## Milestones

Rewrite stage 2 of `paired FA3 flash_fwd + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full decode, 1 < batch <= 64 (now splitkv_attn_kernel + splitkv_combine_kernel)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T001534Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a stage-2 rewrite with no companions. Two points need fixing at build time: STAGES 6 cannot hold 2 CTAs/SM, so the deeper-ring branch resolves to STAGES 5; and the batch-only CHUNK rule picks 512 at B=32, where tail quantization is already about 3%, rather than at B=16, where it is about 15%. Before any CHUNK=512 run, the in-run per-config kernel-A timings should decide which sub-change is kept.

### Justification
The target is real. Kernel A is 93-96% of the replaced time (112.0 us at B=16, 249.5 us at B=32 in job e789032128df4291) against KV floors of about 63 and 126 us, which is 1.9-2.05x the floor. The ~95/189 us milestone (3.5 TB/s on a 128-B-row page-1 gather on H200) is physically reachable. The three mechanisms are much weaker than the target. (a) Shared-memory arithmetic from the stage-1 constants: ring 4x16 KB = 65536, idx 2x1024x4 = 8192, q 2048, acc_s 16384, ml 512, seq/pre 528, about 93.2 KB in total. STAGES 6 is about 126 KB, above the 113 KB limit, so 2 CTAs/SM cannot hold and the rule always lands on STAGES 5 (about 109.6 KB; 2x(107+1 KB) = 216 KB fits within 228 KB). Bytes in flight per SM rise from 3x16x2 = 96 KB to 4x16x2 = 128 KB (+33%). But 96 KB x 132 SMs is already about 12.7 MB in flight chip-wide, well above HBM bandwidth x latency even at 2 us, so pure latency-bound starvation is not demonstrated. The 2.6-2.9 TB/s may instead come from the per-tile __syncthreads across 8 warps, the item-end merge barrier, cp.async issue rate (four 16-B cp.async per thread per tile plus the index gather), or gather sector efficiency. (b) The 'agentic tail at B=32' story contradicts the numbers. Items are fixed 1024-token chunks dealt round-robin over a 264-CTA grid, so imbalance is just ceil/mean of items per CTA. B=32 has about 2.36M tokens, about 2300 items, about 8.7 per CTA: max/mean about 9/8.7, roughly 3% loss. B=16 has about 1.18M tokens, about 1150 items, about 4.4 per CTA: max/mean about 5/4.4, roughly 15% loss. B=16 has the worse tail yet the higher bandwidth (2.9 vs 2.6 TB/s), so the B=32 shortfall is something else. Candidates: the combine/workspace, L2 thrash from 32 page-table rows, or per-item overhead. The batch-only rule (512 when batch >= 24) applies the smaller chunk where it helps least and skips the shape where it helps most, and it doubles item-end merges, ws partial writes (about 4.6K x 4 KB, about 19 MB extra at B=32) and combine reads (combine already 8-9 us). (c) Prefetching the next item's page indices two items ahead buys nothing: with one-ahead issue at tile 0 there are already 16 tiles (8 at CHUNK 512) of lead. It also silently requires idx_s to grow to [3][CHUNK] (+4 KB, putting STAGES 5 at about 113.7 KB, which still fits 2/SM but only just), or k&1 buffer reuse races. Weakest link: no profile evidence shows kernel A is limited by bytes in flight or by tail; the M3 gain (17-22 us at B=16, 55-60 us at B=32) is asserted from bandwidth targets rather than derived from a measured bottleneck. Expected directions are otherwise consistent with TASK.md: only full decode_16/32 (weight 16 each) are gated, and every other shape is a valid control.

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl (running, restack: conflicts, not landed): this is the actual base. Kernel A's stage-1 times (112.0/249.5 us) were measured on the pre-dense-chain base. If the hand-merged restack moves kernel A (stream, PDL ordering, workspace alloc), the 'old config' reference changes. Use the restack run's own profile and the in-run old-vs-new CUDA-event print as the reference, not the e789 numbers.
- small-t-launch-minimal-dense-chain-pdl (gain, master): T<=32 includes decode_16/32. Its PDL chain wraps the qkv/o GEMMs around attention. If kernel A or the combine sits inside a griddepcontrol chain, launch overlap changes, and the shape delta can move without kernel A changing. Judge on kernel A's own us in profile top_kernels, not the shape median.
- fa3-hdim128-head-pairing-batched (neutral, master): it defines the incumbent being replaced (paired FA3 + pad/unpad) on exactly these shapes. That affects only the incumbent baseline for M2, not this stage's delta, which is splitkv stage 1 vs stage 2.
- stack-align-par-sidestream-sliding1 / marlin-block-m-cap-16 (master): MoE-side, after attention. Not causal for kernel A, but Marlin swings ±20-40 us between single profiles on byte-identical paths (REWRITE_READ caveat). A shape-level delta of ±20-40 us at full decode_16/32 is uninterpretable from the headline alone.

### Interference and A/B needs
No companions. This is kind: rewrite stage 2, judged on kernel A's own us against the incumbent and the floor, and none of the landed hypotheses touches kernel A's code path, so an ab-revert would not disambiguate anything. The real attribution risk is internal: three bundled sub-changes (STAGES 5, CHUNK 512 at batch >= 24, two-ahead index prefetch) could cancel each other. That is best resolved in-run at zero check cost. The change already prints kernel-A CUDA-event time for old vs new configs at B=16/32; extend that print to cover each sub-change alone: (S4,C1024), (S5,C1024), (S4,C512), (S5,C512), and two-ahead on/off at the chosen config, at B=16 and B=32 on the self-check synthetic lengths at realistic about 74K average context. Then pick the per-batch winner on device from those timings, as long as the choice comes from batch only. A separate okc control run would cost a check to learn what a 50 ms prepare-time sweep already shows. Caveat: prepare-time timings run with warm L2 and synthetic page tables, so they rank configs but do not reproduce the judge's cold-L2 numbers.

### Open questions
- Why is B=32 kernel-A bandwidth (2.6 TB/s) lower than B=16 (2.9 TB/s) when B=32's round-robin tail quantization is better (about 3% vs about 15%)? If a per-batch cost such as the item_of prefix walk, page-table L2 footprint or ws writes is responsible, neither a deeper ring nor CHUNK 512 addresses it.
- Is CHUNK 512 better applied at B=16 (items/CTA 4.4 -> 8.7, max/mean 1.15 -> about 1.03) than at B=32? The batch-only rule currently has it backwards relative to the quantization argument.
- Does the two-ahead index prefetch get a third idx buffer ([3][CHUNK])? Without one it races with the k&1 double buffer. With one, STAGES 5 + CHUNK 1024 is about 113.7 KB per CTA. That still fits 2 CTAs/SM (about 229 KB including the 1 KB per-block reserve is borderline against the 228 KB SM limit), so verify cudaOccupancyMaxActiveBlocksPerMultiprocessor == 2 and print it.
- STAGES 6 cannot fit 2 CTAs/SM with the current layout (about 126 KB). Should it be dropped from the build, or should acc_s (16 KB) be aliased onto a ring stage at item end to make 6 stages fit?
- Is full-decode-splitkv-restack-on-pdl's measured kernel-A time close to the stage-1 112.0/249.5 us? If the restack itself moved kernel A, re-derive the expected savings before reading this run.

