---
problem: muse_moe_w4a16
slug: silu-mul-triton-row-tiled-gated
created: 2026-10-25T145952Z
base: master
area: silu-mul
change: >
  Built on master (9080a71, submission_blob 60dfe35d); silu-mul only. Replace the one call
  silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1) in
  _forward with a Triton kernel, per T class, only where a prepare-time gate shows it wins. Marlin
  w13/w2 (flags, block_size_m, the w2 atomic mode), the aligns and the post-norm tail are master
  byte for byte. MECHANISM: master's profile (job 7a0d900a1dd34666) has SGLang's
  act_and_mul_kernel at 1.8 / 2.1 / 2.4 / 2.2 / 3.7 / 5.7 us at bs 1 / 16 / 32 / 8 / 128 / 256. It
  launches one CTA per route row (T*8 CTAs, 2048 at bs=256) and the remote-route CTAs (local id
  -1, most of them on this EP8 shard) exit after one id load, so at verify most CTAs do nothing
  yet each costs a scheduling slot. The local rows of cache1 were just written by the w13 Marlin,
  so they sit in L2 (at most about 16 MB of local gate+up rows at bs=256). A kernel with ROWS
  route rows per program and a BLOCK_N column tile cuts the CTA count 4-16x and loads gate and up
  as wide vectors. Target: 5.7 -> about 3 us at bs=256 and 3.7 -> about 2.3 us at bs=128. CHANGE:
  (1) Add _silu_mul_local_kernel(h_ptr, ids_ptr, out_ptr, n_rows, I, ROWS: constexpr, BLOCK_N:
  constexpr), grid (cdiv(n_rows, ROWS), cdiv(I, BLOCK_N)). Program (p, q) loads ids =
  local_ids[p*ROWS + r] for r < ROWS (masked by row < n_rows) and sets valid = ids >= 0. It loads
  gate = h[row, q*BLOCK_N + j] and up = h[row, I + q*BLOCK_N + j] with mask valid[:, None] & (col
  < I), so remote rows' uninitialised cache1 is never read. It computes in fp32 out = gate / (1 +
  exp(-gate)) * up, the same formula as SGLang's silu (x / (1 + expf(-x)), times up in fp32, one
  round to bf16), and stores bf16 to out[row, col] under the same mask, so remote rows of cache2
  stay untouched as now. The w2 Marlin reads only rows named by sorted_ids, which are local. (2)
  Classes by shape only: t1 (T=1), t8 (2..8), t16 (9..16), t32 (17..32), t128 (33..191), t256
  (192..512); T > 512 keeps silu_and_mul. Candidates per class: (ROWS, BLOCK_N, num_warps) in {(1,
  1024, 4), (2, 1024, 4), (4, 512, 4), (8, 512, 8), (8, 1024, 8)}. (3) prepare(), a new step
  _silu2_setup(state, device) run after _w2_atomic_self_check (it does not affect that check). For
  each class's representative T (1, 8, 16, 32, 128, 256): build a real cache1 by running master's
  align + w13 Marlin on synthetic skewed routing (3 of 8 routes local, as _w2_atomic_self_check
  does) and on an all-local case. Then check each candidate: on local rows, bf16 outputs must
  differ from silu_and_mul's by at most 1 bf16 ulp per element and rel-L2 <= 1e-3, and all must be
  finite. The end-to-end check is rel-L2 <= 5e-3 of _forward with the candidate vs master's
  _forward. Time it as the bare kernel in a CUDA graph replayed WARM (no L2 flush, because in the
  pipeline cache1 has just been written by w13 and is L2-resident), 41 interleaved replays each of
  candidate and silu_and_mul, empty-graph baseline subtracted, medians. Enable the fastest
  candidate for a class only if silu_and_mul_median / candidate_median >= 1.05. Assert the stored
  choice is the printed winner. Store state['silu2'] = {class: (ROWS, BLOCK_N, num_warps) or
  None}, plain ints only (the judge copies state). Compile every enabled (class, config) in
  prepare(), so nothing JIT-compiles in kernel_fn. (4) _forward: if state['silu2'].get(cls) is
  set, launch _silu_mul_local_kernel on (cache1, local_ids.view(-1), cache2, T*8, size_n) with the
  stored config; else the master call. No host sync, no prints, no try/except, nothing cached from
  inputs. (5) Print at the very end of prepare() one line per class: [silu2] PATH CHOSEN class=<c>
  T=<T> = triton (ROWS, BLOCK_N, warps) | master, sgl <x> us, triton <y> us, ratio <r>, max ulp
  <u>, e2e rel <e>.
expected:
  - shapes: bs=256 load=verify_32 sharding=ep8
    direction: up
    why: act_and_mul_kernel 5.7 us (2048 one-row CTAs, most exiting on a remote id) -> about 3 us with 8 rows x 512-1024 columns per program; about -2.5 us of 459 us, +0.5% on the row. Read the _silu_mul_local_kernel row against 5.7 us.
  - shapes: bs=128 load=verify_16 sharding=ep8
    direction: up
    why: act_and_mul 3.7 us -> about 2.3 us; +0.4% on a 315 us row.
  - shapes: bs=32 load=decode_32 sharding=ep8
    direction: up
    why: act_and_mul 2.4 us -> about 1.6 us; +0.4% on the weakest-ratio load, if the gate enables t32.
  - shapes: bs=1 load=decode_1 sharding=ep8
    direction: flat
    why: "1.8 us is near a single launch's floor; the gate is expected to keep SGLang's kernel here. If it enables, each 0.5 us saved is +1.1% on this 45 us row."
  - shapes: bs=16 load=decode_16 sharding=ep8
    direction: flat
    why: 2.1 us, small headroom; gate decides.
  - shapes: bs=8 load=verify_1 sharding=ep8
    direction: flat
    why: 2.2 us, small headroom; gate decides.
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T150432Z
review_companions: null
---

# silu-mul-triton-row-tiled-gated

## Change

Built on master (9080a71, submission_blob 60dfe35d); silu-mul only. Replace the one call silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1) in _forward with a Triton kernel, per T class, only where a prepare-time gate shows it wins. Marlin w13/w2 (flags, block_size_m, the w2 atomic mode), the aligns and the post-norm tail are master byte for byte. MECHANISM: master's profile (job 7a0d900a1dd34666) has SGLang's act_and_mul_kernel at 1.8 / 2.1 / 2.4 / 2.2 / 3.7 / 5.7 us at bs 1 / 16 / 32 / 8 / 128 / 256. It launches one CTA per route row (T*8 CTAs, 2048 at bs=256) and the remote-route CTAs (local id -1, most of them on this EP8 shard) exit after one id load, so at verify most CTAs do nothing yet each costs a scheduling slot. The local rows of cache1 were just written by the w13 Marlin, so they sit in L2 (at most about 16 MB of local gate+up rows at bs=256). A kernel with ROWS route rows per program and a BLOCK_N column tile cuts the CTA count 4-16x and loads gate and up as wide vectors. Target: 5.7 -> about 3 us at bs=256 and 3.7 -> about 2.3 us at bs=128. CHANGE: (1) Add _silu_mul_local_kernel(h_ptr, ids_ptr, out_ptr, n_rows, I, ROWS: constexpr, BLOCK_N: constexpr), grid (cdiv(n_rows, ROWS), cdiv(I, BLOCK_N)). Program (p, q) loads ids = local_ids[p*ROWS + r] for r < ROWS (masked by row < n_rows) and sets valid = ids >= 0. It loads gate = h[row, q*BLOCK_N + j] and up = h[row, I + q*BLOCK_N + j] with mask valid[:, None] & (col < I), so remote rows' uninitialised cache1 is never read. It computes in fp32 out = gate / (1 + exp(-gate)) * up, the same formula as SGLang's silu (x / (1 + expf(-x)), times up in fp32, one round to bf16), and stores bf16 to out[row, col] under the same mask, so remote rows of cache2 stay untouched as now. The w2 Marlin reads only rows named by sorted_ids, which are local. (2) Classes by shape only: t1 (T=1), t8 (2..8), t16 (9..16), t32 (17..32), t128 (33..191), t256 (192..512); T > 512 keeps silu_and_mul. Candidates per class: (ROWS, BLOCK_N, num_warps) in {(1, 1024, 4), (2, 1024, 4), (4, 512, 4), (8, 512, 8), (8, 1024, 8)}. (3) prepare(), a new step _silu2_setup(state, device) run after _w2_atomic_self_check (it does not affect that check). For each class's representative T (1, 8, 16, 32, 128, 256): build a real cache1 by running master's align + w13 Marlin on synthetic skewed routing (3 of 8 routes local, as _w2_atomic_self_check does) and on an all-local case. Then check each candidate: on local rows, bf16 outputs must differ from silu_and_mul's by at most 1 bf16 ulp per element and rel-L2 <= 1e-3, and all must be finite. The end-to-end check is rel-L2 <= 5e-3 of _forward with the candidate vs master's _forward. Time it as the bare kernel in a CUDA graph replayed WARM (no L2 flush, because in the pipeline cache1 has just been written by w13 and is L2-resident), 41 interleaved replays each of candidate and silu_and_mul, empty-graph baseline subtracted, medians. Enable the fastest candidate for a class only if silu_and_mul_median / candidate_median >= 1.05. Assert the stored choice is the printed winner. Store state['silu2'] = {class: (ROWS, BLOCK_N, num_warps) or None}, plain ints only (the judge copies state). Compile every enabled (class, config) in prepare(), so nothing JIT-compiles in kernel_fn. (4) _forward: if state['silu2'].get(cls) is set, launch _silu_mul_local_kernel on (cache1, local_ids.view(-1), cache2, T*8, size_n) with the stored config; else the master call. No host sync, no prints, no try/except, nothing cached from inputs. (5) Print at the very end of prepare() one line per class: [silu2] PATH CHOSEN class=<c> T=<T> = triton (ROWS, BLOCK_N, warps) | master, sgl <x> us, triton <y> us, ratio <r>, max ulp <u>, e2e rel <e>.

## Expected

- `bs=256 load=verify_32 sharding=ep8` up: act_and_mul_kernel 5.7 us (2048 one-row CTAs, most exiting on a remote id) -> about 3 us with 8 rows x 512-1024 columns per program; about -2.5 us of 459 us, +0.5% on the row. Read the _silu_mul_local_kernel row against 5.7 us.
- `bs=128 load=verify_16 sharding=ep8` up: act_and_mul 3.7 us -> about 2.3 us; +0.4% on a 315 us row.
- `bs=32 load=decode_32 sharding=ep8` up: act_and_mul 2.4 us -> about 1.6 us; +0.4% on the weakest-ratio load, if the gate enables t32.
- `bs=1 load=decode_1 sharding=ep8` flat: 1.8 us is near a single launch's floor; the gate is expected to keep SGLang's kernel here. If it enables, each 0.5 us saved is +1.1% on this 45 us row.
- `bs=16 load=decode_16 sharding=ep8` flat: 2.1 us, small headroom; gate decides.
- `bs=8 load=verify_1 sharding=ep8` flat: 2.2 us, small headroom; gate decides.

## Risk

Risk low: per-class gate on numerics (1 bf16 ulp vs act_and_mul, e2e rel-L2 5e-3) and a >= 1.05x kernel-time win, else SGLang's silu_and_mul, so a failed gate measures master. The expected effect is 0.4-0.5% per touched row, under the +-1.5% headline resolution; judge it on the profile's _silu_mul_local_kernel rows vs act_and_mul's 1.8-5.7 us and the paired ratios. Caveat for the reviewer: the gate times warm-L2 (cache1 is just-written in the pipeline); if the judge's profile shows the Triton row slower than act_and_mul's, the warm assumption was wrong and the class should be turned off. The prefill lesson that fusing silu into the w13 epilogue lost 1-3% does not apply: this keeps a standalone kernel and changes only its launch geometry. Not a duplicate: the queued silu-mul hypotheses (triton-w13-silu-epilogue, swapab-w13-silu-epilogue-verify) are epilogue fusions on unlanded GEMMs.

## Adversarial review

Reviewed 2026-10-25T150432Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion: the change is a single swapped kernel behind a per-class gate, so the profile row (_silu_mul_local_kernel vs act_and_mul_kernel 1.8-5.7 us) and the [silu2] print line make one run interpretable on their own. Its expected value is low, though, and the gate as specified (warm-L2 bare-kernel replay) is the step most likely to mislead. The one prior Triton replacement of act_and_mul in this repo (prefill silu-mul-triton-local-rows-marlin-turn) ran 2x SLOWER on the judge (14.2 -> 29.4 us, 52.9 -> 86.0 us), so make the gate time it the way the judge does.

### Justification
Numbers check out: master's submit profile 7a0d900a1dd34666 (archive/w2-marlin-atomic-add-small-m) has act_and_mul_kernel at 1.8/2.1/2.4/2.2/3.7/5.7 us for bs 1/16/32/8/128/256, and a2744bc496b84790 repeats them within 0.1-0.3 us (1.5/2.2/2.4/2.0/4.0/5.6), so the kernel row can be read to well under the claimed deltas. The mechanism is plausible only at the verify shapes. At bs=256, T*8 = 2048 one-row CTAs is a real scheduling cost. The local gate+up rows (skewed routing makes this GPU the busiest, so plausibly 3/8 or more of the routes are local, about 12-16 MB) were just written by the w13 Marlin and can sit in the 50 MB L2. 5.7 -> ~3 us is therefore reachable but not guaranteed. At bs <= 32, 1.8-2.4 us is mostly launch and tail latency, and a Triton launch is no cheaper, so 'flat' (gate keeps master) is the honest expectation and the hypothesis says so. The expected shapes match TASK.md: every load weighs 60, the deltas are about 0.4-0.5% per touched row (below the +-1.5% headline CI), and the hypothesis tells the reader to judge on the kernel row, which is right. WEAKEST LINK: the go/no-go rests on a warm-L2, bare-kernel, 41-replay graph microbenchmark of a ~2-6 us kernel. (a) Per-replay event timing of one ~2 us kernel with an empty-graph baseline subtracted is close to the timer's resolution, so a 1.05x threshold can be crossed by noise. (b) Warm replay keeps cache1 AND cache2 in L2 and leaves out the w13 Marlin's eviction pressure: it streams up to ~512 MB of NVFP4 w13 weights through L2 right before silu, so 'just written, therefore resident' is an assumption. (c) Repo precedent: prefill's silu-mul-triton-local-rows-marlin-turn (archive b2259479e57e4196) was reviewed with an expected 14.2 -> ~5-6 us and measured 29.4 us on the judge (2x slower). That differs in kind (a sorted-id gather, BLOCK_R 16 x 512 at 4 warps), but it shows SGLang's act_and_mul is not an easy target for a 2-D tiled Triton elementwise kernel. Sharp note for the builder: time the gate in a graph that contains the w13 Marlin followed by silu (candidate vs silu_and_mul), with an L2-flush buffer read before each replay as the judge does, and subtract a w13-only graph. Or at least put K=20 back-to-back silu launches in each replayed graph and divide, so that per-launch timer noise does not decide the 1.05x gate. Second note: the hypothesis does not mention the prefill regression. It should be added to the risk text so a slower row is read as the known failure mode and not as the warm-L2 caveat alone.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m: cache3 aliases cache13, so w2's output overwrites cache1's storage after silu. Ordering on one stream makes this safe, but _silu2_setup's end-to-end check must call _forward with state['w2_mode'] (the mode actually shipped at T<=32), not 'master', or the check does not cover the path that runs. The atomic mode does not change silu's own time, so it does not confound the kernel row.
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: these produce local_ids, which the new kernel uses as its validity mask (ids >= 0). SGLang's act_and_mul consumes the same tensor today, so the -1 contract is already exercised, and neither confounds the timing. The Triton kernel indexes rows by route slot, not sorted_ids, so the block_size_m pick (marlin-block-m-two-smaller-on-postnorm) does not touch it at all.
- Not landed but relevant to value: BEST.json (1.3727x) is the unpromoted wgmma rewrite cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate. Its submission_blob field reads 60dfe35d, which is master's Marlin-only blob, so the record looks mislabeled. That rewrite fuses silu into its w13 at 128<=T<=512, so the only shapes this hypothesis expects to move (t128, t256) are the ones the rewrite line would make dead code. On master's 1.3445x lineage, a scored submit of this change will land near 1.345-1.35, below the board best, and will not move the board.

### Interference and A/B needs
No companion. Both a gain and a loss can be read without one. The profile replaces the act_and_mul_kernel row with _silu_mul_local_kernel at the same shapes, between the same Marlin producer and consumer. The [silu2] PATH CHOSEN line says per class whether the gate enabled it, with the gate's own sgl/triton microbench numbers to compare against the judge row. Classes the gate leaves on master are an in-run control for drift. No landed hypothesis changes what silu reads or how long it takes (see confounders), so an ab-revert would measure nothing new. A control that isolates a sub-change has nothing to isolate either, because there is only one kernel. Since every extra run is a scored submit today, a companion is clearly not worth it. If the judge row comes out slower than act_and_mul, the follow-up is a stacked hypothesis that changes the gate's timing method or turns the class off, not an A/B.

### Open questions
- What fraction of the T*8 route rows is local under the judge's skewed routing at bs=128/256? The 'most CTAs exit immediately' premise and the 12-16 MB L2-residency estimate both depend on it. The gate could print the local-row count of its skewed synthetic case for comparison.
- Why was prefill's Triton silu 2x slower than act_and_mul (gather through sorted_ids, 4 warps x 16x512 tile, or cold-L2 behaviour)? If the cause was cold L2 rather than the gather, the warm-L2 gate here will pass candidates that lose on the judge.
- Is a scored submit worth it for a <=0.5%-per-row change on master's 1.3445x lineage, when the board best (1.3727x) comes from the unlanded wgmma rewrite that already replaces silu at 128<=T<=512? If the wgmma line is promoted, only the decode classes of this change survive, and they are expected to stay flat.
- BEST.json records submission_blob 60dfe35d (master's Marlin-only file) for the wgmma s5 job 4e4a0e26c85944f2. Is the churn index writing the wrong blob, and does that mislead promotion or base selection?

