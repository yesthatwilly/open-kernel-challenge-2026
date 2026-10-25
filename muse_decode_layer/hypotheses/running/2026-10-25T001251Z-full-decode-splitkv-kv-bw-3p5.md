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
status: running
review_at: 2026-10-25T001534Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T002203Z
base_sha: b2f95e63a6e8ad3cff7947760e69e448399e295d
base_blob: c9067b8201f2480c64910c4a71098df7ce616f73
branch: hyp/muse_decode_layer/full-decode-splitkv-kv-bw-3p5
worktree: .worktrees/muse_decode_layer/full-decode-splitkv-kv-bw-3p5
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T003019Z
impl_commit: 2e50a1c882d8266acaa8403578ff0edf3d7df2ab
submission_sha256: 2994e7e8a781f2149779a53aca4cac8f7065ca815b203b72cc71acb3cbae8ccd
norm_sha256: 10e4d35a9616c4cacabaf3c1578711425d88a6788b025816b233298546a3e1a2
submission_blob: 320bc577c8ea6a341804b40ed6228e4cbc6757e2
lint_warnings: 36
attempt_file: 2026-10-25T003019Z-check-full-decode-splitkv-kv-bw-3p5.md
prepared_at: 2026-10-25T003019Z
job: 6af3b0e3311943ee
dispatched_at: 2026-10-25T003026Z
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


## Implementation notes

Built on the worktree base b2f95e63a6 (the hand-merged full-decode-splitkv-restack-on-pdl). Only the split-KV section of `muse_decode_layer/submission.py` changed: kernel A, the host launcher, the combine's chunk argument, the self-check, and prepare-time selection and timing. The gate (`_splitkv_gate`), the kernel_fn call site and the combine's math are unchanged.

- **Kernel A is templated as `splitkv_attn_kernel<STAGES, CHUNK, AHEAD>`** with a `KCfg` smem layout struct. Six instantiations are dispatched by a variant id: 0 = S4/C1024/A1 (stage 1, byte-identical logic), 1 = S5/C1024/A1, 2 = S5/C1024/A2, 3 = S4/C512/A1, 4 = S5/C512/A1, 5 = S5/C512/A2. The extension was renamed `muse_full_decode_splitkv_v2` and gained `splitkv_variant_info(v)`, which returns `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and the smem bytes.
- **(a) Ring depth: STAGES 5, not 6 (review fix).** S6 would need ~126 KB/CTA and drop to 1 CTA/SM. S5 is 109,584 B at C1024/A1 and 114,704 B with the third meta buffer. The cp.async.cg 16-B form and `wait_group<STAGES-2>` are unchanged.
- **(b) CHUNK 512 by batch only, applied to the small-batch class (review fix; the review overrides the spec's rule).** The spec's rule (512 when batch >= 24) put the small chunk on B=32, where the round-robin tail loss is ~3%, and skipped B=16, where it is ~15%. The rule is now CHUNK 512 when batch <= 23 and 1024 above it. The combine takes the chunk as a runtime argument (n = ceil(len/chunk)), and `max_chunks` and the workspace are sized per chunk.
- **(c) Two-ahead index/query prefetch (AHEAD=2), kept per the spec with the third buffer the review flagged.** It uses `[NBUF=3][CHUNK]` idx and q buffers indexed `k % NBUF`, so there is no k&1 race. The prologue waits for items 0..AHEAD-1, and meta(k+AHEAD) rides item k's first tile and reuses item k-1's buffer. `static_assert`s enforce `AHEAD*TILES >= STAGES-1` and `TILES > STAGES-1`.
- **Selection (prepare, batch only):**
  - Small batch prefers 5 (S5/C512/A2), then 4, then stage 1.
  - Large batch prefers 2 (S5/C1024/A2), then 1, then stage 1.
  - A variant qualifies only if it passes the self-check and has occupancy >= 2.
  - If the stage-1 variant fails its self-check, the path is OFF (FA3). Any other variant's failure falls back to stage 1, not to FA3.
  - The log prints `[splitkv-s2] PATH CHOSEN = ...`.
- **Self-check:** the existing 4 cases (B=5 at kv_scale 1/0.25 on the device grid and grid 3, plus B=33) run for every variant with that variant forced. Each launch is announced before, synced, and printed after.
- **In-run timings (review fix):** `_splitkv_s2_timing` prints every self-checked variant's kernel-A CUDA-event time and TB/s at B=16 and B=32 on synthetic lengths with a ~74K mean, plus the combine time at each chunk. This covers S4/S5 × C1024/C512 and two-ahead on/off, so the kept sub-change can be decided from the log. The timing only informs; it does not choose the variant (no autotuning). Caveat: warm L2 and a synthetic page table.
- **Not done:** the review's acc_s-aliasing idea to make S6 fit (it is a new idea, outside scope). No wgmma or setmaxnreg is used, so `TORCH_CUDA_ARCH_LIST` is untouched. The existing `-arch=sm_90a -lineinfo` flags are kept. There are no mbarriers; the only waits are cp.async wait_group, which are bounded.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/full-decode-splitkv-kv-bw-3p5 diff`:
- [x] CUDA-graph capturable. kernel_fn's path adds only a dict lookup on `batch` (config) and a `torch.empty` sized from shapes. The syncs and `.elapsed_time` calls are in prepare-only self-check and timing code.
- [x] Nothing derived from `data` is cached. `_SPLITKV_SEL` is set once in prepare from synthetic checks and the device occupancy.
- [x] Weights are untouched by this change.
- [x] Activations and precision are unchanged: the query is FP8 E4M3, the KV stays FP8, the combine output is bf16, and accumulation is fp32.
- [x] Attention sees every key: all chunks cover [0, len) and only positions >= len are masked.
- [x] No timing patches, harness access, module edits, reference import, network, or processes. There is no subprocess, multiprocessing, fork or spawn.
- [x] All work happens inside the timed call; nothing is cached or replayed.
- [x] Fallbacks never raise. Build failure or a stage-1 self-check failure gives FA3; a new-variant failure gives stage 1. The timing is wrapped in try/except.

okc_lint output:
```
(host-sync warnings, all "outside kernel_fn's call graph; fine in prepare()")
warn  dtype        L2348/L2350: torch.randint(..., dtype=torch.uint8) -- synthetic FP8 cache bytes for prepare-time timing only
OK with warnings: submission.py
```
`python3 -I -m py_compile` passed. `__pycache__` was removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, by reading the code; no nvcc or GPU was available. **Verdict: dispatch-with-fixes.**

Fixes:
- `attn_launch`: hoisted `KCfg<S, CK, A>::SMEM` into `constexpr int smem_bytes` before the launch, so no template-id sits inside `<<< >>>`. This is defensive and costs nothing at runtime. A parse error there would cost the whole module and fall back to FA3.

Verified:
- **Pipeline.** A Python emulation (`AHEAD*TILES >= STAGES-1`, `TILES > STAGES-1`) covered all 6 variants at nk 1..14. In each case, item k's meta lands before its first tile issue and before the t==0 q read. The `k % NBUF` buffer is overwritten only after item k-1's last tile was issued and its q was read, with a `__syncthreads` in between. The `wait_group<STAGES-2>` group count stays uniform.
- **Shared memory and occupancy.** smem is 114,704 B for S5/C1024/A2: 2×(114,704+1,024) = 231,456 ≤ 233,472, so 2 CTAs/SM. All offsets are 16-B aligned for cp.async16 and the float4 reads. `__launch_bounds__(256,2)` is unchanged.
- **Dispatch.** The macro dispatch, `C10_CUDA_CHECK` around template-ids (the commas are paren-protected), the per-instantiation `static attr_set`, the anonymous-namespace split and the pybind return of `std::vector<int64_t>` all read correctly.
- **Combine.** The chunk comes from the same variant tuple as kernel A's CHUNK, and `max_chunks` and ws are sized per chunk.
- **No new memory accesses out of bounds and no spin waits.**
- **Fallbacks.** A build error goes to FA3 with PATH CHOSEN printed. A stage-1 self-check miss goes to FA3. A new variant's miss, or occupancy < 2, goes to stage 1. The final `[splitkv-s2] PATH CHOSEN` line names the variant per batch class.
- **prepare()'s timing sweep.** It only logs: it reads `_SPLITKV_SEL` after the selection is final, and it resets `_SPLITKV_FORCE` in a `finally`. kernel_fn depends on nothing from the sweep. There is no subprocess, multiprocessing or fork.
- **kernel_fn.** It adds only dict lookups on `batch` and allocations sized from shapes. There is no host sync, and nothing derived from data is cached.

Scope note: the CHUNK rule is inverted from the spec, to 512 at batch <= 23, following the adversarial review. B=16 gets C512 and B=32 gets C1024.

Residual risk: the 6-way template has never been compiled. A build error lands on FA3, not stage 1, and a device fault in a new variant's self-check would poison the CUDA context rather than fall back. No out-of-bounds path was found by reading.
