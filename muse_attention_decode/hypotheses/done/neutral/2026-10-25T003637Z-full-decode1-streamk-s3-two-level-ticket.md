---
problem: muse_attention_decode
slug: full-decode1-streamk-s3-two-level-ticket
created: 2026-10-25T003637Z
base: master
area: full-layer attention (batch-1 decode, splitkv stream-K kernel)
change: >
  Stage 3 of the full-decode rewrite, built on the stage-2 branch
  hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel:
  stream-K over 64-token tiles with an in-kernel per-request ticket combine + norm-gate), whatever
  its score. Full decode_1 is 27.3 us (FA3 main 14.6 + FA3 combine 7.1 + pre-attn 1.3 + norm-gate)
  against a 4.3 us floor and weight 16 in the decode_1 load (42% of that load); stage-1
  chunk320/chunk640 gated batch 1 onto splitkv and gained on full decode_1 but paid a separate
  combine over many partials. Change: (1) widen the gate to 1 <= batch <= 64 (config only), so
  full decode_1 runs master _pre_attention_kernel (PAIRED=False, as master batch 1 already does) +
  the stage-2 kernel. (2) At batch == 1 only (config-only), grid = SM count (one CTA per SM, ~9.5
  tiles each for 80000 keys) instead of 2 x SM count, passed through the existing grid argument.
  (3) Two-level ticket for a request covered by many CTAs: CTAs whose ranges intersect request b
  are grouped by c // G with G = ceil(sqrt(nseg_b)) (from TP and grid only); each CTA writes its
  fp32 partial (acc[16,64], m[16], l[16]), __threadfence(), atomicAdd on group ticket gticket[b*32
  + g]; the group last arrival LSE-merges its group partials into a group slot, __threadfence(),
  atomicAdd on ticket[b]; the last group LSE-merges the <= G group slots, applies v_scale, the
  stage-2 norm-gate epilogue and stores bf16, and resets both its tickets to 0 for graph replay.
  Batch > 1 keeps the stage-2 single-level ticket (nseg_b <= ~17 there). gticket is a module-level
  int32[64*32] zero buffer allocated at import. BUILD/SAFETY exactly as stage 2
  (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary +
  self-check, each kernel followed by torch.cuda.synchronize() and a print; never a child
  process); add batch-1 self-check cases at 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids
  3/7/SM count, and two consecutive replays to prove both ticket levels reset. Fallback on
  failure: batch 1 returns to master FA3 (PATH CHOSEN: full decode_1 FA3 (fallback: <reason>));
  success prints PATH CHOSEN: full decode_1 streamk two-level. Every other shape keeps the stage-2
  code. First measurement is an okc check, never a scored submit.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: replaced FA3 main 14.6 + combine 7.1 + norm-gate (~23 us of a 27.3 us shape), floor 4.3. 20.5 MB at the stage-1 2.7-2.9 TB/s is ~7.5 us + two merges ~2 us. M1 tests pass and shape < 27.3; M2 shape <= 16 us; M3 <= 12 us
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32*"
    direction: flat
    why: batch > 1 keeps stage-2 grid and single-level ticket
  - shapes: "*layer=sliding* *load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: full-decode-splitkv-s2-streamk-ticket-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1
review: proceed
status: done
review_at: 2026-10-25T014055Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T014056Z
base_sha: f291b49c49670595d0615e749294f69bc15de1a4
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket
worktree: .worktrees/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T014724Z
impl_commit: 47f0b132ce44ef1abe3c08993414df386327d33a
submission_sha256: 1d33819472efb0da5bd5f4f8f44536ac210807337e7989cb15ceaa9a873403b7
norm_sha256: ade28e32d49b1eccb1dcbf82f145d75155772bb1b14dacea06cf396daa644120
submission_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
lint_warnings: 13
attempt_file: 2026-10-25T014724Z-check-full-decode1-streamk-s3-two-level-ticket.md
prepared_at: 2026-10-25T014724Z
job: 2a3241780c7b4c46
dispatched_at: 2026-10-25T014732Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4321773019894537
speedup_ci: [1.4106946424596118, 1.4554282820505813]
harvested_at: 2026-10-25T034127Z
best_at_harvest: 1.3948738389144513
promotion: merged
hand_landed: true
promotion_commit: c3c1b01b18837f889ac11a15d8f0daf06d356d98
promotion_attempt: 2026-10-25T041933Z-submit-full-decode1-streamk-s3-two-level-ticket.md
---

# full-decode1-streamk-s3-two-level-ticket

## Change

Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: stream-K over 64-token tiles with an in-kernel per-request ticket combine + norm-gate), whatever its score. Full decode_1 is 27.3 us (FA3 main 14.6 + FA3 combine 7.1 + pre-attn 1.3 + norm-gate) against a 4.3 us floor and weight 16 in the decode_1 load (42% of that load); stage-1 chunk320/chunk640 gated batch 1 onto splitkv and gained on full decode_1 but paid a separate combine over many partials. Change: (1) widen the gate to 1 <= batch <= 64 (config only), so full decode_1 runs master _pre_attention_kernel (PAIRED=False, as master batch 1 already does) + the stage-2 kernel. (2) At batch == 1 only (config-only), grid = SM count (one CTA per SM, ~9.5 tiles each for 80000 keys) instead of 2 x SM count, passed through the existing grid argument. (3) Two-level ticket for a request covered by many CTAs: CTAs whose ranges intersect request b are grouped by c // G with G = ceil(sqrt(nseg_b)) (from TP and grid only); each CTA writes its fp32 partial (acc[16,64], m[16], l[16]), __threadfence(), atomicAdd on group ticket gticket[b*32 + g]; the group last arrival LSE-merges its group partials into a group slot, __threadfence(), atomicAdd on ticket[b]; the last group LSE-merges the <= G group slots, applies v_scale, the stage-2 norm-gate epilogue and stores bf16, and resets both its tickets to 0 for graph replay. Batch > 1 keeps the stage-2 single-level ticket (nseg_b <= ~17 there). gticket is a module-level int32[64*32] zero buffer allocated at import. BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary + self-check, each kernel followed by torch.cuda.synchronize() and a print; never a child process); add batch-1 self-check cases at 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids 3/7/SM count, and two consecutive replays to prove both ticket levels reset. Fallback on failure: batch 1 returns to master FA3 (PATH CHOSEN: full decode_1 FA3 (fallback: <reason>)); success prints PATH CHOSEN: full decode_1 streamk two-level. Every other shape keeps the stage-2 code. First measurement is an okc check, never a scored submit.

## Expected

- `*layer=full load=decode_1 *` faster: replaced FA3 main 14.6 + combine 7.1 + norm-gate (~23 us of a 27.3 us shape), floor 4.3. 20.5 MB at the stage-1 2.7-2.9 TB/s is ~7.5 us + two merges ~2 us. M1 tests pass and shape < 27.3; M2 shape <= 16 us; M3 <= 12 us
- `*layer=full load=decode_16* *layer=full load=decode_32*` flat: batch > 1 keeps stage-2 grid and single-level ticket
- `*layer=sliding* *load=verify_*` flat: outside the gate

## Risk

Risk: a single CTA LSE-merging 132 partials would be ~1 MB serial read, so the two-level merge is the design point; if M2 is missed, the per-phase eager diagnostic should name whether the gather or the merge dominates.

## Milestones

Rewrite stage 3 of `FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T014055Z by an adversarial subagent (model opus).

**Verdict: proceed.** Targets the largest remaining full-layer gap (decode_1 27.2 us vs floor 4.3, FA3 main+combine+norm-gate ~22.8 us) with a kernel that stage 2 already proved correct for one request spanning every CTA. The design is sound and M2 (<=16 us) is plausible from the chunk320 evidence (A 11.0 + combine 4.5). The fixed choices, grid = SM count and two-level ticket, are untested and should be bracketed by in-process eager diagnostics, not by a separate okc run.

### Justification
Stage 2 (job e592b1f5d3db4f83, 1.4171 [1.396,1.443] vs best 1.3949 [1.374,1.416], unproven-vs-best only because the CIs overlap) made a real, attributable kernel gain: full decode_16 shape 122.8 -> 110.9 us (A+B 120.0 -> splitkv_attn_sk_kernel 108.9), full decode_32 241.9 -> 226.8 us (A+B 254.3 -> 226.8); every other shape within ~1 us of best. M1 met on both; M2 narrowly missed (108.9 vs <=108, 226.8 vs <=222). The in-process diagnostics split it: stream-K work split alone gave 1.10x/1.14x on kernel A (117.5 -> 106.9, 245.3 -> 215.1 us eager), while the ticket fold saved only 1.05x/1.03x net: the ticket kernel is ~5.8/6.2 us slower than the no-ticket kernel, i.e. the serialized last-arriver merge tail costs ~6 us against the 11.7 us separate combine it replaces. A: stage 2's in-kernel ticket path already passed the self-check for 'B=2 one request spans every CTA' at grids 7 and 264, so widening the gate to batch 1 reuses a correct kernel. Prior batch-1 evidence: full-decode1-cuda-splitkv-chunk320 ran stage-1 A in 11.0 us (20.5 MB, ~1.86 TB/s) + combine 4.5 + pre-attn 1.4 = 19.5 us shape, vs 27.3 on FA3. It lost only through its unrelated chunk change at decode_32, which this gate avoids (batch > 1 keeps the stage-2 code). Stream-K over 1250 64-token tiles removes chunk quantization, and folding combine + norm-gate removes ~8 us of launches and kernels. A realistic target is attention ~8-11 us (ramp-dominated at only ~9.5 tiles per CTA) + merge ~1-2 us + pre-attn 1.4, giving 11-14 us: M2 likely, M3 (<=12) possible. Expected directions are consistent: decode_1 full weight 16; decode_16/32 should be byte-identical to stage 2 (the same grid and single-level ticket at batch > 1); sliding and verify are gated out. Merge cost: with 132 partials of 16x64 fp32 (+m,l), single-level is ~0.55 MB read serially by one CTA after the last arrival. At B=16 (~17 slots per request) the stage-2 diagnostics already attribute ~6 us of tail to the ticket path, so a two-level merge at nseg 132 is the right design point. WEAKEST LINK: grid = SM count at batch 1. It halves the CTAs (and the bytes in flight) vs the 2-CTA/SM residency the kernel was tuned for (113 regs, 95 KB smem, 2 CTAs/SM), and each CTA's fixed prologue (prefix load, meta prefetch, ring fill, q fetch) dominates a ~9.5-tile range. Grid 264 halves the per-CTA work instead, which may hide the ramp better, at the cost of 264 partials (G=17). Nothing in the record says which wins; the hypothesis fixes one config with no measurement. A secondary weak point: the 2.7-2.9 TB/s bandwidth figure comes from 328-655 MB runs. A 20.5 MB kernel lasts ~7 us, so ramp-up and the tail matter, and the 7.5 us estimate is optimistic.

### Confounders on the current kernel
- full-decode-splitkv-s2-streamk-ticket-inproc (parent, not on master): decode_16/32 should reproduce stage 2 exactly (110.9/226.8). The headline vs BEST mixes stage 2's own gain into this run, so judge decode_1 by top_kernels: the target is the v2 kernel plus pre-attn replacing FA3 main 14.7 + FlashAttnFwdCombine 7.0 + norm_gate 1.1.
- paired-heads-launch-cleanup-ab-tune-batch-gate: batch 1 already takes the unpaired q layout (paired requires batch > 1), and 'and not splitkv' keeps it unpaired once the gate widens. Consistent, no confound; but if the gate widening is written into _splitkv_gate, check that the pairing predicate and the q allocation still agree at batch 1.
- fused-preattention-triton: the batch-1 pre-attn (1.4 us, unpaired) is kept as is; it neighbours the kernel only in launch order.
- full-decode-cuda-gather-splitkv (v1) and the earlier chunk320/chunk640 attempts: same-area prior results (neutral). They are the evidence base, not confounders, because v1 is not on the batch-1 path here.
- sliding-fused-triton-decode1-gate / sliding-deep-splits: sliding-only, gated out.

### Interference and A/B needs
No okc companion is warranted. Both sub-decisions (grid SM vs 2xSM, single- vs two-level ticket) are config or template switches of one kernel, and they are best bracketed in-process on the same GPU, as stage 2 did with its diagnostics. Ask the builder to print eager cold-L2 timings at synthetic B=1 x 80000 for: (a) the stage-2 kernel as is at grid 2xSM (single-level, the widen-gate-only control), (b) grid SM single-level, (c) grid SM two-level (the timed path), (d) grid 2xSM two-level, alongside master FA3 main + combine + norm_gate at the same input. That makes a decode_1 gain or loss attributable to widening, grid or merge without spending a check. Also print the two-level path's kernel time at B=16 as a diagnostic only (it is not on the timed path), which answers whether the ~6 us stage-2 tail would shrink at batch > 1. Rank by prepare timings only (LESSONS: they undercount the judge by 20-25%).

### Open questions
- Grid SM vs 2xSM at batch 1: which wins once ramp is included? Make the diagnostic decide it (print all four variants), or make the grid a second import-time selection from the printed timing (LESSONS 2026-10-25: pick variants from the printed timing, or assert the static choice is the fastest).
- Does a 256-thread parallel single-level merge (each thread owns 4 of 1024 outputs and streams 132 slots with independent loads) already reach ~1-2 us, which would make the second ticket level unnecessary complexity?
- Will the fused pre-attn of the sibling stage 3 (full-decode-splitkv-s3-fused-preattn) be extended to batch 1 when the two stage-3 branches are merged? 1.4 us is ~10% of the target decode_1 shape.
- Block-scheduler placement: with grid == SM count and 2 CTAs/SM residency, check that the CTAs actually spread 1 per SM (it is usually breadth-first, but not guaranteed); the eager diagnostic's grid-SM bandwidth will show it.


## Implementation notes

All in `<worktree>/muse_attention_decode/submission.py` (stage 2 base f291b49; nothing could be compiled or run on this node).

- Gate: new shape-only `_shape_gate` (1 <= batch <= 64); `_splitkv_gate` = `_EXT` and shape and (batch > 1 or `_B1_PATH`). `_B1_PATH` is set only after the batch-1 validation passes, so a failure sends batch 1 to master FA3 (FA3 + norm-gate, unpaired as before; pairing still needs batch > 1, so the q layout and branch agree).
- Kernel: `splitkv_attn_sk_kernel<STAGES, TICKET, TWO>`; new variant 3 = `<4, true, true>`. Args added: `gticket` (int32[B*32]), `gacc`/`gml` (group slots). Variants 0-2 are unchanged in behaviour (batch > 1 keeps the stage-2 single-level ticket, grid 2xSM). `merge_slots` now wraps a generic `merge_range` that also returns the raw max.
- Two-level ticket (effective gx = min(gridDim.x, T) as in stage 2; CTAs above gx exit): for a non-whole segment, nseg = c_hi - c_lo + 1, Gs = ceil(sqrt(nseg)) (integer loop, from TP and grid only), group gi = (c - c_lo) / Gs (relative to c_lo rather than the spec's absolute c // G: same size bound, groups stay full-aligned), ngrp = ceil(nseg / Gs) <= 32. Each CTA stores its partial, `__threadfence`, `__syncthreads`, tid 0 `atom.add.acq_rel.gpu` on `gticket[b*32+gi]`. The group's last arrival (all threads `__threadfence` = acquire) LSE-merges its slots into group slot b*32+gi, resets its group ticket to 0, `__threadfence` + `__syncthreads`, tid 0 `atom.add.acq_rel.gpu` on `ticket[b]`; the last group merges the ngrp group slots, runs the stage-2 `finalize_store` (v_scale, norm-gate, bf16) and resets `ticket[b]` to 0. If ngrp == 1 the group merger finalizes directly. No CTA waits on another. `gticket` is a module-level zero buffer made at import (`_sk_gticket`, per device).
- Config at batch 1 (fixed by shape, not by timing): variant 3, grid = SM count; batch > 1 unchanged.
- Review fixes applied: (1) in-process eager cold-L2 timing of the four variants at B=1 x 80000 printed at import (`_sk_b1_diagnostics`, `[sk3-diag]`): (a) single-level grid 2xSM (control), (b) single-level grid SM, (c) two-level grid SM (timed), (d) two-level grid 2xSM, with rel L2 vs the FP32 reference, TB/s, a WARNING line if (c) is not fastest (path stays shape-fixed), master FA3 main+combine and norm-gate at the same input, and the two-level vs single-level kernel at B=16 as diagnostic only. (2) The review's open question on a parallel single-level merge was not pursued (the spec fixes the two-level design). Where review and spec differ nothing conflicts; the review's "rank by prepare timings only" is followed.
- Safety: module renamed `muse_full_decode_splitkv_v3` (TORCH_CUDA_ARCH_LIST=9.0a, -O3 -lineinfo, arch flags printed, binaryVersion == 90 gate, same plain cp.async.cg). Batch-1 validation `_sk_validate_b1` (after stage-2 validation, each kernel announced before and synced + printed after): 12 cases, B=1 at 80000, 4097, 63 and 1 keys, kv scale 1 and 0.25, grids 3/7/SM/2xSM, plus B=16 two-level; then two consecutive CUDA-graph replays (output NaN-poisoned) at 80000 and 4097 keys, and a check that both ticket levels sum to 0. Failure leaves batch 1 on FA3.
- PATH CHOSEN lines: `full decode_1 streamk two-level` or `full decode_1 FA3 (fallback: <reason>)`; the stage-2 lines for batch > 1 are kept.

## Rules check

- [x] CUDA-graph capturable: kernel_fn has no sync/.item()/prints; the new allocations (group slots) are shape-only; the tickets reset in-kernel and the graph replays are verified in the import self-check.
- [x] Nothing derived from `data` is cached: only the zero ticket buffers (device-only) persist.
- [x] n/a: no weights in this problem.
- [x] Activations BF16 as before; KV FP8; queries FP8 E4M3 (unchanged from stage 2).
- [x] Attention sees every key: tile ranges cover all keys of the request, no skipping (the git diff only changes the merge tree and the gate).
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (no subprocess/multiprocessing/os.fork anywhere; diff checked).
- [x] All work in the timed call; no output caching or replay.
- [x] Effective grid gx = min(gridDim.x, tiles) used by the group arithmetic; CTAs without work exit cleanly.

okc_lint: `OK with warnings: submission.py` (only host-sync warnings, all in import/prepare-time self-check code outside kernel_fn). `python3 -I -m py_compile` ok, __pycache__ removed. Not compiled or run (no nvcc/GPU here).

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only review plus a Python emulation; nothing compiled).

**Verdict: dispatch-with-fixes.**

- Fix: `ticket_arrive` inline PTX rewritten from `atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;` to the documented qualifier order `atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;` with the increment in a register. Reason: the variant-3 template lives in the same module as stage 2, so a ptxas rejection of the non-canonical form would fail the whole v3 build and drop batch > 1 back to stage 1 too. Semantics unchanged.
- Checked, no change needed: the group arithmetic relative to c_lo. A Python emulation (20k random cases, B=1 and B=2..64, grids 1..1024, shuffled arrival order, plus B=1 at 1/63/4097/80001/1048577 keys over grids 1/3/7/132/264/1024) confirms every partial is merged exactly once, each group and the request finalize exactly once, ngrp <= 32, leaders are well defined for nseg < Gs, and both ticket levels end at 0. Release/acquire: the writer does fence, sync, then an acq_rel atomic, and the reader syncs, fences, then does __ldcg loads at both levels, the standard threadFenceReduction pattern. Both levels are reset in-kernel by their last arrival. The routing for batch > 1 is identical to stage 2, because _splitkv_gate is unchanged when batch > 1 and pairing still needs batch > 1. Batch 1 stays on FA3 unless _sk_validate_b1 passes. The diagnostics re-zero both ticket levels in finally. The gate checks binaryVersion == 90, not ptxVersion. kernel_fn has no host sync, no data caching and no child process, and its allocations depend only on shapes. Every key is attended, KV stays FP8, and ROPE is off on full layers.

Residual risk: variant 3 is untested on hardware. A ptxas or register-allocation problem in the shared v3 module would also cost stage 2 for batch > 1, though it falls back cleanly to v1 with a PATH CHOSEN line. Import time grows with 12 extra cases, 2 replays and the B=16x80000 diagnostics.

## Result

- job `2a3241780c7b4c46` (check), status done, passed True, tests 16/16
- speedup 1.4322x, 95% CI [1.4107, 1.4554], score 2629.7721621255905 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.patch`; branch `hyp/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.09 | 13.15 | 1.0286 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.19 | 22.6 | 1.2023 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.19 | 15.33 | 0.9998 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 110.91 | 111.14 | 0.9953 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.29 | 24.33 | 1.0135 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.81 | 226.42 | 0.9996 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 16.83 | 17.78 | 0.9407 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 29.84 | 30.51 | 0.9845 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.72 | 22.68 | 1.0097 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.42 | 162.8 | 1.0095 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.87 | 28.34 | 1.0127 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.06 | 302.2 | 1.0065 | flat |

gained on full decode_1; lost on sliding verify_1. Expected hit: 2/3. Noise ±1.6%.

## Follow-ups

- rewrite stage 3 of `FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-decode1-streamk-s3-two-level-ticket --stage 4`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-streamk-s3-two-level-ticket`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

