---
problem: muse_attention_decode
slug: full-verify-fa3-num-splits-band-gate
created: 2026-10-25T153027Z
base: master
area: FA3 call (full verify split count)
change: >
  Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 main 14.9
  + combine 7.3 + prepare 2.8 + norm-gate 1.1 us, shape 30.5) and B=16/32 paired (main 154.5/293.1
  + combine 5.9/7.7 + prepare 3.0/3.6, profile 252abb7f11ec48c3). At B=1 FA3 splits one request's
  80K keys over ~all SMs and the combine (7.3 us, half the main kernel) merges every split's [8 x
  16 x 64] fp32 partial; fewer, longer splits trade main-kernel parallelism for a cheaper combine.
  num_splits is a cap under dynamic split (LESSONS), so lowering it is effective. CHANGE: in
  kernel_fn's FA3 branch for full verify only (window == 0, q_len 8; config-only) pass
  num_splits=_FV_SPLITS.get(band, 0), band 1 for B=1, 16 for 2 <= B <= 24 (paired), 32 for 25 <= B
  <= 64 (paired); everything else keeps the default. _FV_SPLITS is an import-time frozen table
  chosen in prepare per band at batch x 80000 (agentic lengths for 16/32): time master's step
  after pre-attention (B=1: FA3 + combine + rms_norm_gate; B=16/32: paired FA3 + combine +
  _paired_norm_gate_kernel) for num_splits in {0 (auto), 16, 24, 32, 48, 64, 96} at B=1 and {0, 4,
  6, 8, 12, 16} at B=16/32, interleaved cold-L2 graph replays (256 MB flush, empty graph
  subtracted, median of 31); print every number; route the fastest only at >= 1.03x vs auto and
  output rel L2 < 1e-2 vs auto, else keep auto. Self-check the chosen settings on the full-verify
  tests' shapes at page 1 (B 2 ctx 100000, B 4 ctx 6000 edges, kv 0.25) against master's output.
  Disjoint from full-verify-wgmma-ws-s1-producer-split (that one replaces the B=16/32 FA3 only
  where its gate routes; this one is measured against the FA3 path itself). RULES (every path):
  CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in
  kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32
  ticket arrays, zeroed once and returned to 0 by every launch, are fine).
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: combine 7.3 us is ~half of FA3 main 14.9; fewer splits should cut 2-4 us; shape 30.5, floor 4.3
  - shapes: "*layer=full load=verify_16 *"
    direction: flat
    why: combine 5.9 us of 163; small possible gain, gate keeps auto otherwise
  - shapes: "*layer=full load=verify_32 *"
    direction: flat
    why: combine 7.7 us of 302; small possible gain
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: fa3-num-splits-full-verify (expired, unrun; it raised splits, this one caps below the dynamic pick)
review: proceed
status: running
review_at: 2026-10-25T212855Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T212903Z
base_sha: 2922874607ac8f3aee67594fba45a5337e29365e
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/full-verify-fa3-num-splits-band-gate
worktree: .worktrees/muse_attention_decode/full-verify-fa3-num-splits-band-gate
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T213737Z
impl_commit: 4871dfb7df1f51c65c28413d229c1fa70d59253d
submission_sha256: 66068e3c96f6dffe814729ba85437b0e71df2cfa3a70cf9da2f9d143d75f34f1
norm_sha256: 5ddd85291d12665f2ebab53159a422d2c031c6f2f47aa8b0f9e01f8ec40d5c19
submission_blob: f171866b1c89fc143bbb35f711723ef60d39a715
lint_warnings: 65
attempt_file: 2026-10-26T012536Z-check-full-verify-fa3-num-splits-band-gate.md
prepared_at: 2026-10-26T012536Z
---

# full-verify-fa3-num-splits-band-gate

## Change

Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 main 14.9 + combine 7.3 + prepare 2.8 + norm-gate 1.1 us, shape 30.5) and B=16/32 paired (main 154.5/293.1 + combine 5.9/7.7 + prepare 3.0/3.6, profile 252abb7f11ec48c3). At B=1 FA3 splits one request's 80K keys over ~all SMs and the combine (7.3 us, half the main kernel) merges every split's [8 x 16 x 64] fp32 partial; fewer, longer splits trade main-kernel parallelism for a cheaper combine. num_splits is a cap under dynamic split (LESSONS), so lowering it is effective. CHANGE: in kernel_fn's FA3 branch for full verify only (window == 0, q_len 8; config-only) pass num_splits=_FV_SPLITS.get(band, 0), band 1 for B=1, 16 for 2 <= B <= 24 (paired), 32 for 25 <= B <= 64 (paired); everything else keeps the default. _FV_SPLITS is an import-time frozen table chosen in prepare per band at batch x 80000 (agentic lengths for 16/32): time master's step after pre-attention (B=1: FA3 + combine + rms_norm_gate; B=16/32: paired FA3 + combine + _paired_norm_gate_kernel) for num_splits in {0 (auto), 16, 24, 32, 48, 64, 96} at B=1 and {0, 4, 6, 8, 12, 16} at B=16/32, interleaved cold-L2 graph replays (256 MB flush, empty graph subtracted, median of 31); print every number; route the fastest only at >= 1.03x vs auto and output rel L2 < 1e-2 vs auto, else keep auto. Self-check the chosen settings on the full-verify tests' shapes at page 1 (B 2 ctx 100000, B 4 ctx 6000 edges, kv 0.25) against master's output. Disjoint from full-verify-wgmma-ws-s1-producer-split (that one replaces the B=16/32 FA3 only where its gate routes; this one is measured against the FA3 path itself). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=full load=verify_1 *` faster: combine 7.3 us is ~half of FA3 main 14.9; fewer splits should cut 2-4 us; shape 30.5, floor 4.3
- `*layer=full load=verify_16 *` flat: combine 5.9 us of 163; small possible gain, gate keeps auto otherwise
- `*layer=full load=verify_32 *` flat: combine 7.7 us of 302; small possible gain
- `*layer=sliding *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Full verify_1 is 45% of the verify_1 load (16 x 30.5 of 1095 us). Low cost: config + prepare timing only.

## Adversarial review

Reviewed 2026-10-25T212855Z by an adversarial subagent (model opus).

**Verdict: proceed.** A cheap, config-only, self-measuring stopgap that DIAGNOSIS action 5 names for full verify_1's 7.2 us combine. It caps FA3's split count below the dynamic pick, which is the only direction LESSONS says works. Proceed as a single check with no companions, because the prepare sweep already carries its own auto-vs-chosen control. Fix the missing prior and use the two-round selector rule.

### Justification
Mechanism: reference.py passes scheduler_metadata=None on every verify block, and master's FA3 call for full verify (B=1 unpaired, B>1 paired) passes metadata=None or the judge's None, so FA3 runs prepare_varlen_num_blocks and splits dynamically. The board profile confirms this: prepare 2.8 us at verify_1, 3.1 us at verify_16. Under varlen dynamic split num_splits is a cap (LESSONS 'Attention'), so a value below the heuristic does change the schedule. Values at or above auto (64/96 at B=1, 16 at B=16/32) should time the same as auto, which gives a free internal noise control. Profile support: full verify_1 is 30.3 us (FA3 main 15.2, combine 7.2, prepare 2.8, plus pre-attn and norm-gate) against a 4.3 us floor. At B=1 with pack-GQA there are only 2 m-tiles (1 request x 2 kv heads x 64 packed rows), so FA3 fans out to about 64 splits. The combine is then latency-dominated: about 2 MB of fp32 partials should take under 1 us at bandwidth, yet it takes 7.2 us. Fewer splits can plausibly cut 2-4 us of combine, but every halving also halves the CTAs pulling 20.5 MB through a page-1 gather that today runs at only about 1.35 TB/s. Weakest link: the trade is not monotone and the main kernel may slow by more than the combine saves. The prepare sweep with a >=1.03x gate turns that loss into 'keep auto' (flat), so the run is cheap either way. Expected directions are consistent with TASK. Full verify_1 has weight 16: 2-4 us there is 32-64 us on the verify_1 load (~1095 us), +3-6% on that load and about +0.5-1% headline, which is inside the headline CI. Read the result per shape (judge paired-ratio medians on full verify_1), not from the headline. At verify_16/32 the combine is 6.4/6.7 us of 162/302 us. Reaching 1.03x on the step would need about 5/9 us of saving, so 'flat via auto' is the right expectation, and a lower cap there mostly removes CTAs from a gather-bound kernel. Prior: the frontmatter has NO prior field, and there are obvious earlier attempts. (1) muse_attention_decode/hypotheses/done/expired/2026-10-23T225858Z-fa3-num-splits-full-verify.md, held as moot and expired unrun. It tried to RAISE splits to fill SMs, which the cap semantics make impossible. (2) muse_decode_layer done/expired fa3-full-single-split-probe (num_splits=1 cap, held and unrun). Neither was ever measured. This hypothesis caps BELOW the dynamic pick and chooses the value by measurement, so the difference is real and the mechanism has not been tried. It is not on either Do-not-try list, and DIAGNOSIS action 5 explicitly names this slug as the stopgap. I do not hold on the missing prior, but the builder must add prior: fa3-num-splits-full-verify (expired, unrun; tried to raise splits, while this one lowers them under the cap).

### Confounders on the current kernel
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup + paired-heads-launch-cleanup-ab-tune-batch-gate: these fix the tile geometry the split choice depends on (B=1 unpaired, 2 kv heads x 1 m-tile; B>1 paired, 1 super-head). The chosen caps only hold for that geometry. The sweep is measured on master's own geometry, so it is self-consistent, but the table would be stale if pairing at B=1 ever changed.
- fused-preattention-triton (+ paired q layout): sets the q tensor layout FA3 sees (64- vs 128-wide). It does not move with num_splits, so it is no confounder for the attributed delta. Note only that the sweep must time the step AFTER pre-attention, as specified.
- Not landed but relevant: full-verify-wgmma-ws-s1-producer-split / full-verify-wgmma-s6-split-ring-2cta would replace paired FA3 at B=16/32 where they gate. If one lands first, the B=16/32 bands of this table become dead code. That causes no confounding, because the gate keeps those shapes on whatever routes first, but the merge needs the band check ordered after the wgmma gate.

### Interference and A/B needs
No companion is warranted. The change is a single keyword argument, and its value is chosen by an in-prepare A/B against auto (num_splits=0) on master's exact step at each band, with every number printed. That log is itself the control: a flat result reads as 'auto kept' (or as a chosen cap below 1.03x), and a gain or loss on full verify_1 is attributable to the printed cap with no landed hypothesis in between. An ab-revert of pairing would only re-measure the B=1 unpaired geometry, which already is unpaired. An ab-tune of the cap set duplicates what the sweep already does. The only real ambiguity risk is selector noise, which is addressed in open_questions (two interleaved rounds), not by a companion run.

### Open questions
- Add the missing prior field: fa3-num-splits-full-verify (expired, unrun; it raised splits, this one caps below them).
- Gate the table on the scored geometry (heads tp8 / hkv 2, head_dim 64, page_size 1) as well as window/q_len/batch. Otherwise the B=1 cap also applies to the 'batch 1 ctx 1048576 q_len 8' test (page 64) and the dp-heads verify tests, and the B=2..24 band cap applies to 'batch 2 ctx 100000'. That is correct but unmeasured. Keep everything off the scored geometry on auto.
- Should the B=1 candidate set include 8, and an explicit 1 (no combine launch at all), as diagnostic rows only? Printing them would tell whether the combine or the main kernel's MLP sets the curve. 1 is almost certainly slower (2 CTAs), so keep it unroutable.
- Confirm that at B=1 FA3 actually skips or shrinks FlashAttnFwdCombine when the cap lowers the dynamic split. Read it from the run's profile: combine us at the chosen cap against 7.2 us.
- Can it be stacked with sliding-verify-fa3-num-splits-band-gate? It uses the same mechanism on disjoint shapes, so if both gain, a restack is a clean merge.


## Implementation notes

Edited only `muse_attention_decode/submission.py` (+249 lines, no deletions; no new load_inline module).
- `kernel_fn`: the FA3 call gets `num_splits=_fv_cap(cfg, metadata)`. `_fv_cap` returns 0 (auto) unless the frozen table `_FV_SPLITS` (MappingProxyType, ints only) has the band AND the shape is the scored geometry (window 0, q_len 8, page_size 1, hq 16 / hkv 2 (tp8), head_dim 64) AND `scheduler_metadata` is None (a judge-supplied metadata keeps auto). Bands: B=1 -> 1, 2..24 -> 16, 25..64 -> 32.
- `_fv_setup()` runs last in the import chain (its PATH CHOSEN lines close the prepare log). Synthetic tensors only (`_ps_synthetic`, window 0, no RoPE, lens [80000] at B=1, 80000 + random 2000..80000 at B=16/32, kv scale 1). Candidates: B=1 {16,24,32,48,64,96}, B=16/32 {4,6,8,12,16}, plus auto (0) in every sweep. Timed step = master's step after pre-attention (FA3 + combine + rms_norm_gate at B=1, paired FA3 + combine + `_paired_norm_gate_kernel` at B>1), cold L2 (256 MB), empty graph subtracted, median of 31, two interleaved rounds, every median/IQR printed.
- Gate (per band): chosen cap = lowest worst-round median among routable caps; it must, in BOTH rounds, give auto/cap >= 1.03x, have its Q3 below auto's median, and have IQR <= 1.5x auto's. Otherwise the band stays auto. Then the cap is checked against auto's output (rel L2 < 1e-2, finite) on B=2 ctx 100000 kv 0.25 and B=4 edge lengths at kv 0.25 (band 16), B=1 ctx 1000 kv 0.25 and 80000 (band 1), B=26 mixed (band 32); a miss drops the band to auto.
- Diagnostic rows at B=1: caps 8 and 1 are timed and printed, never routable (deviation note: 8 is also diagnostic-only, following the review's wording).
- Phase budget: skipped (printed `skipped: deadline`) if > 230 s since module import or > 45 s into the phase. Any raise leaves the band on auto.
- `prior:` frontmatter added.

Review fixes applied: prior field; geometry gate; two interleaved rounds with IQR bound; diagnostic rows 8 and 1 (1 never routed). Open question on whether FA3 shrinks/skips the combine at a lower cap: answer from the run's profile (combine us at the chosen cap vs 7.2). Stacking with sliding-verify-fa3-num-splits-band-gate is a disjoint-shape merge (different gate, separate table name `_FV_SPLITS`).

## Rules check

- [x] CUDA-graph capturable: kernel_fn only adds a Python-int kwarg read from a frozen table; no sync/alloc on values; sweep is import-time only.
- [x] Nothing derived from `data` is cached: table built from synthetic tensors; no device pointers or buffers kept (sweep tensors deleted).
- [x] n/a: weights untouched.
- [x] Activations/KV/queries unchanged (FP8 KV, e4m3 q, bf16 out).
- [x] Attention sees every key: only num_splits changes, same math (cap verified vs auto rel L2 < 1e-2).
- [x] No timing patches, harness frames, module edits, reference import, subprocesses or network (timing is in prepare only on synthetic data).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only host-sync warns in prepare-time code, outside kernel_fn's call graph). py_compile passed; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus). **Verdict: dispatch-with-fixes.** Rules OK.

Fixes (all in the worktree submission.py; okc_lint OK with only prepare-time host-sync warnings; py_compile passed; __pycache__ removed):
1. **Gate that always failed (import deadline).** `_fv_setup` skipped every band once more than 230 s had passed since module import. Master's import already runs four ~41 s load_inline builds plus the ps/skF/sk-diag/slide/sd gates before this phase (job 252abb7f's whole run was ~470 s), so on the judge every band would always print `skipped: deadline` and the path would never run. The deadline is now `_FV_IMPORT_DEADLINE_S = 480`, the convention other import-phase hypotheses use under the 15-minute cap. The 45 s per-phase limit is kept.
2. **Fallback isolation for the FA3 kwarg.** kernel_fn passed `num_splits=_fv_cap(...)` on every FA3 call, sliding verify included. If the judge's wrapper ever rejected the kwarg, every FA3 shape would fail, not fall back. kernel_fn now passes `**_fv_kw(cfg, metadata)`, which is `{}` unless a band cap is frozen. With an empty table the call is byte-identical to master's, and a rejected kwarg surfaces only in the sweep, where it falls back to auto.
3. **Capture-failure recovery (concern 1).** `torch.cuda.graph.__exit__` always calls `capture_end`, so capture always ends. But if `capture_end` itself raises (an invalidated capture), the stream context is never exited and the capture stream stays current. The new `_fv_recover(stream0)` runs in both except handlers (sweep and self-check). It restores the import-time stream, checks `is_current_stream_capturing()` is False and syncs. If it cannot recover, the table is cleared and the sweep stops with a PATH CHOSEN line. c10's CUDA check clears the non-sticky capture error, so no stale error reaches the next launch. A sticky fault (illegal address) is unrecoverable by any design.
4. **IQR gate false negatives.** The rule "cap IQR <= 1.5x auto IQR" rejected any cap whenever auto's IQR came out 0 at event-timer granularity. This could happen on the ~20 us B=1 step. The rule is now `1.5x auto IQR + 0.5 us`.

Concern answers:
- (2) `_fv_cap` and `_fv_kw` are pure cfg/dict lookups, with only an identity test on `scheduler_metadata`. I emulated them over 11 configs.
- (3) This sgl_kernel's `flash_attn_with_kvcache` has `num_splits=0` as its default: 0 means FA3's heuristic. Under the varlen dynamic split (cache_seqlens is always passed), an explicit value is a device-side cap, min(dynamic, static) (LESSONS; fa3-num-splits-full-no-metadata review). reference.py passes `num_splits=0` to the judge's `get_scheduler_metadata`, but no prior judge run has passed it to `flash_attn_with_kvcache`. The sweep's eager warm-up is therefore the first real test, and fix 2 makes that safe.
- (4) The table applies only at window 0, q_len 8, page 1, hq 16/hkv 2, head_dim 64, metadata None, B<=64. The 1M-ctx test (page 64), the dp tests and decode stay on auto. The banded B=2..24 cap does reach the `batch 2 ctx 100000 q_len 8 page 1` test, which is self-checked in prepare at exactly that shape. This follows the spec's banding, so I kept it.

Residual risk: on the judge, a cap below the dynamic pick may simply not win 1.03x in both rounds, giving a flat "auto kept" run. Untested on the judge: the num_splits kwarg itself, and the sweep's wall-clock fit after master's ~400 s import.
