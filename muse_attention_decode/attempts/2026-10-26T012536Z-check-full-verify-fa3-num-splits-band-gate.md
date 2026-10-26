# muse_attention_decode · 2026-10-26T012536Z · check · full-verify-fa3-num-splits-band-gate

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T153027Z-full-verify-fa3-num-splits-band-gate.md` (branch `hyp/muse_attention_decode/full-verify-fa3-num-splits-band-gate`).

- **Job:** `849f2733f8ea4888` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `4871dfb7`

## Goal
Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 main 14.9 + combine 7.3 + prepare 2.8 + norm-gate 1.1 us, shape 30.5) and B=16/32 paired (main 154.5/293.1 + combine 5.9/7.7 + prepare 3.0/3.6, profile 252abb7f11ec48c3). At B=1 FA3 splits one request's 80K keys over ~all SMs and the combine (7.3 us, half the main kernel) merges every split's [8 x 16 x 64] fp32 partial; fewer, longer splits trade main-kernel parallelism for a cheaper combine. num_splits is a cap under dynamic split (LESSONS), so lowering it is effective. CHANGE: in kernel_fn's FA3 branch for full verify only (window == 0, q_len 8; config-only) pass num_splits=_FV_SPLITS.get(band, 0), band 1 for B=1, 16 for 2 <= B <= 24 (paired), 32 for 25 <= B <= 64 (paired); everything else keeps the default. _FV_SPLITS is an import-time frozen table chosen in prepare per band at batch x 80000 (agentic lengths for 16/32): time master's step after pre-attention (B=1: FA3 + combine + rms_norm_gate; B=16/32: paired FA3 + combine + _paired_norm_gate_kernel) for num_splits in {0 (auto), 16, 24, 32, 48, 64, 96} at B=1 and {0, 4, 6, 8, 12, 16} at B=16/32, interleaved cold-L2 graph replays (256 MB flush, empty graph subtracted, median of 31); print every number; route the fastest only at >= 1.03x vs auto and output rel L2 < 1e-2 vs auto, else keep auto. Self-check the chosen settings on the full-verify tests' shapes at page 1 (B 2 ctx 100000, B 4 ctx 6000 edges, kv 0.25) against master's output. Disjoint from full-verify-wgmma-ws-s1-producer-split (that one replaces the B=16/32 FA3 only where its gate routes; this one is measured against the FA3 path itself). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## What changed, and where
Edited only `muse_attention_decode/submission.py` (+249 lines, no deletions; no new load_inline module).
- `kernel_fn`: the FA3 call gets `num_splits=_fv_cap(cfg, metadata)`. `_fv_cap` returns 0 (auto) unless the frozen table `_FV_SPLITS` (MappingProxyType, ints only) has the band AND the shape is the scored geometry (window 0, q_len 8, page_size 1, hq 16 / hkv 2 (tp8), head_dim 64) AND `scheduler_metadata` is None (a judge-supplied metadata keeps auto). Bands: B=1 -> 1, 2..24 -> 16, 25..64 -> 32.
- `_fv_setup()` runs last in the import chain (its PATH CHOSEN lines close the prepare log). Synthetic tensors only (`_ps_synthetic`, window 0, no RoPE, lens [80000] at B=1, 80000 + random 2000..80000 at B=16/32, kv scale 1). Candidates: B=1 {16,24,32,48,64,96}, B=16/32 {4,6,8,12,16}, plus auto (0) in every sweep. Timed step = master's step after pre-attention (FA3 + combine + rms_norm_gate at B=1, paired FA3 + combine + `_paired_norm_gate_kernel` at B>1), cold L2 (256 MB), empty graph subtracted, median of 31, two interleaved rounds, every median/IQR printed.
- Gate (per band): chosen cap = lowest worst-round median among routable caps; it must, in BOTH rounds, give auto/cap >= 1.03x, have its Q3 below auto's median, and have IQR <= 1.5x auto's. Otherwise the band stays auto. Then the cap is checked against auto's output (rel L2 < 1e-2, finite) on B=2 ctx 100000 kv 0.25 and B=4 edge lengths at kv 0.25 (band 16), B=1 ctx 1000 kv 0.25 and 80000 (band 1), B=26 mixed (band 32); a miss drops the band to auto.
- Diagnostic rows at B=1: caps 8 and 1 are timed and printed, never routable (deviation note: 8 is also diagnostic-only, following the review's wording).
- Phase budget: skipped (printed `skipped: deadline`) if > 230 s since module import or > 45 s into the phase. Any raise leaves the band on auto.
- `prior:` frontmatter added.

Review fixes applied: prior field; geometry gate; two interleaved rounds with IQR bound; diagnostic rows 8 and 1 (1 never routed). Open question on whether FA3 shrinks/skips the combine at a lower cap: answer from the run's profile (combine us at the chosen cap vs 7.2). Stacking with sliding-verify-fa3-num-splits-band-gate is a disjoint-shape merge (different gate, separate table name `_FV_SPLITS`).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn only adds a Python-int kwarg read from a frozen table; no sync/alloc on values; sweep is import-time only.
- [x] Nothing derived from `data` is cached: table built from synthetic tensors; no device pointers or buffers kept (sweep tensors deleted).
- [x] n/a: weights untouched.
- [x] Activations/KV/queries unchanged (FP8 KV, e4m3 q, bf16 out).
- [x] Attention sees every key: only num_splits changes, same math (cap verified vs auto rel L2 < 1e-2).
- [x] No timing patches, harness frames, module edits, reference import, subprocesses or network (timing is in prepare only on synthetic data).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only host-sync warns in prepare-time code, outside kernel_fn's call graph). py_compile passed; __pycache__ removed.

## Risks / what could fail
medium
Full verify_1 is 45% of the verify_1 load (16 x 30.5 of 1095 us). Low cost: config + prepare timing only.

## Result (after the run)
- job `849f2733f8ea4888` (check), status done, passed True, tests 16/16
- speedup 1.5216x, 95% CI [1.4988, 1.5506], score 2475.2470041896886 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-fa3-num-splits-band-gate.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-fa3-num-splits-band-gate.patch`; branch `hyp/muse_attention_decode/full-verify-fa3-num-splits-band-gate`
