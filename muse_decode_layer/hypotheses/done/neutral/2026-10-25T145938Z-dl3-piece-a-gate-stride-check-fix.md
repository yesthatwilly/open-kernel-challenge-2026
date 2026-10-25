---
problem: muse_decode_layer
slug: dl3-piece-a-gate-stride-check-fix
created: 2026-10-25T145938Z
base: master
area: full decode_1 attention side (dl3 piece A enable)
change: >
  Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel
  Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + kernel O dl2_o, replacing nvjet
  qkv + FusedQKNorm + FP8 casts + store_cache + nvjet o at full decode_1) never ran in the landed
  job 34a9be06918d448b: its log prints 'sk gate plumbing n=4097: separate-gate output identical to
  qkv-gate output: True' (and the same for n=30000) yet PATH CHOSEN says 'piece A ... (fallback:
  self-check: sk gate plumbing n=4097: outputs differ (or gate stride 2304))'. The outputs
  matched; the failure is the stride clause. In _dl3_check_a the test gate is built as gate_c =
  sqkv[:, SK_QKV_GATE_OFF:].contiguous() from a ONE-row sqkv [1, 2304]; PyTorch treats a size-1
  leading dim as contiguous whatever its stride, so .contiguous() returns the view unchanged with
  stride(0) == 2304, and 'gate_c.stride(0) != DL3_GATE_STRIDE' (1024) is always true. The fix, in
  _dl3_check_a only: build the test gate as a fresh tensor with real [rows, 1024] strides, gate_c
  = torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device) followed by
  gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:]), so the separate-gate launch uses row stride 1024
  exactly as dl2_qkv's live gate does (that live gate already printed 'gate stride 1024' in the Q
  check). Keep the stride clause and the torch.equal comparison as they are. Nothing else changes:
  kernel_fn, the CUDA modules, piece A's gate (tokens == 1, full layer, sk variant 3, state flag),
  the layer-level eager and alternating-state checks (_dl3_layer_checks with A=1), and selector A
  (new chain must beat the parent chain by DL3_MARGIN 1.02 or the incumbent stays) all run
  unmodified and still decide whether A goes live. Print one extra line after the plumbing check
  with gate_c.stride() so the log shows the corrected stride. If any later check or selector A
  fails, the run measures master exactly (fallback per piece).
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "piece A replaces the attention side around the sk kernel at full decode_1 (nvjet qkv 128x8 splitK 12.4 us + splitKreduce + FusedQKNorm + 2-3 cast/store nodes + nvjet o 64x8 6.8 us) with Q (8.3 us measured in s2) and O (6.5 us); s2 measured custom Q+O 18.6 vs 27.3 us and full decode_1 -3.3% (z 5.8) on an older base; expect about -4 to -8 us of 130 us. Read the dl2_qkv / dl2_o kernel rows and the absence of the nvjet qkv/o rows in the full decode_1 profile, and the 'PATH CHOSEN: dl3 piece A' line"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: "control: piece A is gated to full layers; sliding decode_1 keeps _sliding_fused_kernel and master's GEMMs, byte-identical path"
  - shapes: "*load=decode_16 *"
    direction: flat
    why: "control: T > 1 untouched"
  - shapes: "*load=decode_32 *"
    direction: flat
    why: "control: T > 1 untouched"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "control: T = 8 untouched"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "control: untouched"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T150343Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T150343Z
base_sha: 2f268b5931c7c758c5178f3bb152d80688ca9d95
base_blob: 07cda658e21f2eb25a9dd1f50ac5202ad5985831
branch: hyp/muse_decode_layer/dl3-piece-a-gate-stride-check-fix
worktree: .worktrees/muse_decode_layer/dl3-piece-a-gate-stride-check-fix
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T151022Z
impl_commit: b7cda2f98490edaa87165022bdbfbb2ba1cb1e7c
submission_sha256: 7c05556dc4f176d98eaee0a84a2cb72a9c42ae167af50f3291020bc809abf797
norm_sha256: 0ecc710dd1b881b74d8fdef9019a3ab78a795894ae7614342aed797ae968c552
submission_blob: 4bb57fd292a7f0fd5a80bd88dc773008b490bf8c
lint_warnings: 57
attempt_file: 2026-10-25T151022Z-submit-dl3-piece-a-gate-stride-check-fix.md
prepared_at: 2026-10-25T151022Z
job: 933d3f825a044d36
dispatched_at: 2026-10-25T151030Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3079569800910293
speedup_ci: [1.288337625389664, 1.3275763347923946]
harvested_at: 2026-10-25T153250Z
best_at_harvest: 1.3017519911209459
promotion: null
---

# dl3-piece-a-gate-stride-check-fix

## Change

Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + kernel O dl2_o, replacing nvjet qkv + FusedQKNorm + FP8 casts + store_cache + nvjet o at full decode_1) never ran in the landed job 34a9be06918d448b: its log prints 'sk gate plumbing n=4097: separate-gate output identical to qkv-gate output: True' (and the same for n=30000) yet PATH CHOSEN says 'piece A ... (fallback: self-check: sk gate plumbing n=4097: outputs differ (or gate stride 2304))'. The outputs matched; the failure is the stride clause. In _dl3_check_a the test gate is built as gate_c = sqkv[:, SK_QKV_GATE_OFF:].contiguous() from a ONE-row sqkv [1, 2304]; PyTorch treats a size-1 leading dim as contiguous whatever its stride, so .contiguous() returns the view unchanged with stride(0) == 2304, and 'gate_c.stride(0) != DL3_GATE_STRIDE' (1024) is always true. The fix, in _dl3_check_a only: build the test gate as a fresh tensor with real [rows, 1024] strides, gate_c = torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device) followed by gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:]), so the separate-gate launch uses row stride 1024 exactly as dl2_qkv's live gate does (that live gate already printed 'gate stride 1024' in the Q check). Keep the stride clause and the torch.equal comparison as they are. Nothing else changes: kernel_fn, the CUDA modules, piece A's gate (tokens == 1, full layer, sk variant 3, state flag), the layer-level eager and alternating-state checks (_dl3_layer_checks with A=1), and selector A (new chain must beat the parent chain by DL3_MARGIN 1.02 or the incumbent stays) all run unmodified and still decide whether A goes live. Print one extra line after the plumbing check with gate_c.stride() so the log shows the corrected stride. If any later check or selector A fails, the run measures master exactly (fallback per piece).

## Expected

- `*layer=full load=decode_1 *` faster: piece A replaces the attention side around the sk kernel at full decode_1 (nvjet qkv 128x8 splitK 12.4 us + splitKreduce + FusedQKNorm + 2-3 cast/store nodes + nvjet o 64x8 6.8 us) with Q (8.3 us measured in s2) and O (6.5 us); s2 measured custom Q+O 18.6 vs 27.3 us and full decode_1 -3.3% (z 5.8) on an older base; expect about -4 to -8 us of 130 us. Read the dl2_qkv / dl2_o kernel rows and the absence of the nvjet qkv/o rows in the full decode_1 profile, and the 'PATH CHOSEN: dl3 piece A' line
- `*layer=sliding load=decode_1 *` flat: control: piece A is gated to full layers; sliding decode_1 keeps _sliding_fused_kernel and master's GEMMs, byte-identical path
- `*load=decode_16 *` flat: control: T > 1 untouched
- `*load=decode_32 *` flat: control: T > 1 untouched
- `*load=verify_1 *` flat: control: T = 8 untouched
- `*load=verify_16 *` flat: control: untouched
- `*load=verify_32 *` flat: control: untouched

## Risk

Risk: this turns on code that has never run in the judge's timed path. Its kernels passed their own checks in the landed job (Q code gaps 0/1/0, O rel L2 1e-5, sk gate output identical), but the A=1 layer-level checks and selector A have not run yet. Both still gate it, and a miss falls back to master. A scored submit that fails a test costs one submit and not the board. Upside is about 1.4% on the decode_1 load (one weight-16 shape), about 0.25% on the headline, so judge it on the full decode_1 row and the PATH CHOSEN line, not the headline. Reviewer: confirm that _dl3_layer_checks with A=1 exercise kv_scale 0.25 and 1.0, the alternating two-state graph replays and the tickets (cnt_q, cnt_o, sk) reset. Also confirm that kernel_fn's piece-A gate reads only config and the state flag.

## Adversarial review

Reviewed 2026-10-25T150343Z by an adversarial subagent (model opus).

**Verdict: proceed.** The diagnosis checks out against master and the landed job's log: piece A was turned off by a test artifact, not by its kernels. The one-line fix to the check makes the stride clause pass for real and leaves every downstream gate (layer-level eager and alternating checks with A=1, selector A at 2%) in charge. The full decode_1 row, its profile rows and the PATH CHOSEN line make a single scored run readable without a companion.

### Justification
Verified in submission.py (master d5595769), lines 5276-5284: gate_c = sqkv[:, SK_QKV_GATE_OFF:].contiguous() on a [1, 2304] sqkv. A size-1 leading dim counts as contiguous, so stride(0) stays 2304 and the clause gate_c.stride(0) != DL3_GATE_STRIDE (1024) always trips. Log 34a9be06918d448b.log lines 83/88 print 'identical ... True', and line 143 prints the fallback naming 'gate stride 2304' for both n values. Q's live gate already reports stride 1024 (log lines 63/73), and the Q/O kernel checks passed: code gaps 0/1/0, O rel L2 1e-5. Mechanism: at full decode_1 (130.0 us vs SoL 44.2) the attention side around splitkv_attn_sk (19.8 us) still runs nvjet qkv splitK plus reduce, FusedQKNorm, FP8 casts, store_cache and nvjet o, about 27 us by s2's measurement. Replacing that with Q (8.3) and O (6.5) saving 4-8 us is plausible, and s2 measured -3.3% (z 5.8) for the same Q/O kernels on an older base. All other expected rows are correct as flat: kernel_fn line 6060 gates A on dl3['a'], tokens == 1 and _sk_variant == 3, which needs batch 1 and a full layer, so sliding decode_1 and every T>1 shape are untouched. Weakest link: the -4 to -8 us estimate carries over s2's FA3-era chain, but sk two-level already folded the norm-gate into its epilogue, and PDL in the parent may already hide part of the small-kernel launch gaps, so the real saving could sit at the low end (~3-4 us, ~3%). Selector A (80000 keys, needs 1.02x on a ~47 us chain) should still pass easily. A second, minor point: at B=1 the gate stride argument is never used past row 0, so the old and new checks test the same plumbing (offset 0 vs offset 1280). The fix is correct but it does not add test coverage.

### Confounders on the current kernel
- dense-chain-s3-t1-q-sk2-o-route-unpool (the base): pieces B and C are live. Turning A on reruns _dl3_layer_checks on {A,B,C} together. If that combined check fails, bisection can turn B or C off, and that would move BOTH decode_1 rows (B and C apply to sliding and full) for reasons unrelated to A. Read the 'layer-level check of pieces' and the per-piece PATH CHOSEN lines before attributing any decode_1 movement. A sliding decode_1 regression together with a B/C fallback means bisection, not A.
- full-decode-sk-ticket-restack-b1-b64 / full-decode-splitkv-restack-on-pdl lineage (sk['b1'] decision): piece A only composes with the sk two-level variant 3. If the parent's B=1 selector picks FA3 in this run (run-to-run selector noise), A is off by construction and the run measures master. That is visible in the A fallback reason.
- w2-marlin-atomic-add-t1-port: no numeric interaction with A, but if bisection turns B off, _w2_atomic_self_check reruns and w2_mode could change on both decode_1 rows. Check the w2_mode tail on the PATH CHOSEN lines.
- small-t-launch-minimal-dense-chain-pdl: A's O output feeds the attn_boundary and P with PDL. A's Q rounds q to FP8 slightly differently, and _dl3_compare_layer allows one near-tied expert swap. Marlin cost is unchanged by routing ids at T=1 (8 experts either way), so this does not confound timing.

### Interference and A/B needs
No companion is warranted. The change is gated to one shape (full decode_1), and the run attributes itself: the PATH CHOSEN line says whether A went live and with which selector ratio, and the profile shows whether dl2_qkv/dl2_o replaced the nvjet qkv/o, FusedQKNorm and cast/store rows. All five other loads and sliding decode_1 act as built-in controls. If A is live and full decode_1 drops ~3-6% with the controls flat, that is a clean gain. If A is live and the row is flat or slower, that is a clean negative on the Q/O chain on the sk base. If A falls back, the run measures master plus the check fix, and the log says why. An ab-revert of the base would not make sense, because A exists only on the base, and any companion would cost a scored submit for no interpretive gain.

### Open questions
- Alternation coverage for A: _dl3_alternation_check runs only _DL3_CASES[0] (full kv_scale=0.25, n=4097) and sliding. The full kv_scale=1.0 / local_start=96 case is covered eagerly only, not under two-state graph replay. Tickets cnt_q/cnt_o/sk are checked as 0 after every replay, which answers the hypothesis's ticket-reset question.
- Is s3's dl2_qkv (which now also does RoPE, the FP8 casts and the KV store) the same kernel s2 ran through the judge, or an extension? If it is an extension, the judge-pass evidence from s2 is weaker for the KV-store half (judge bound: new cache entries within 8%; the internal check allows a 1-code FP8 gap).
- If the run passes and full decode_1 improves by less than ~2 us, check whether the prepare-time selector A ratio (printed with the M2 line) overstates the in-graph gain because the parent's chain already overlaps with PDL in the full layer.


## Implementation notes

Base 2f268b59 has the same submission.py blob as master 9716bda4, so no merge was needed. Single change in `_dl3_check_a` (muse_decode_layer/submission.py, 4 insertions, 1 deletion): the test gate is now `torch.empty((sqkv.shape[0], 1024), dtype=sqkv.dtype, device=device)` plus `gate_c.copy_(sqkv[:, SK_QKV_GATE_OFF:])`, replacing `.contiguous()`. The stride clause and torch.equal comparison are unchanged. Added one print after the plumbing line with `gate_c.stride()`. The per-piece `PATH CHOSEN: dl3 piece A/B/C` lines already exist in `_dl3_layer_checks` (one per piece, each with w2_mode tail and the fallback reason), so nothing was added there. Review items: all confounder and open-question items are reading guidance for the harvest (check the B/C PATH CHOSEN lines and w2_mode before attributing decode_1 movement); no build action needed. Deviations: none.

## Rules check

- [x] CUDA-graph capturable: change is prepare-only (self-check); kernel_fn untouched.
- [x] Nothing derived from `data` is cached across calls: n/a, no state change.
- [x] Weights: unchanged.
- [x] Activations/KV/queries precision: unchanged.
- [x] Attention sees every key: unchanged.
- [x] No timing patches, harness I/O, module edits, reference import, child processes: checked in the git diff.
- [x] All work inside the timed call: unchanged.

okc_lint: OK with warnings (only host-sync warnings in prepare(), pre-existing). py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent, before a SCORED submit with no checks left. **Verdict: dispatch-with-fixes.**

The stride fix is correct: a fresh `torch.empty((rows, 1024))` has stride (1024, 1), so the clause passes, and the `torch.equal` test is unchanged. Scope matches the spec. kernel_fn's piece-A gate reads only config and state flags (tokens == 1, `_sk_variant == 3`, which needs page 1, a full layer and batch 1). Judge test 11 and the scored full decode_1 shape take A; the page-64 `batch 1, ctx 3000, full` test does not. Every failure path in `_dl3_finish` and `_dl3_layer_checks` is caught and lands on the parent path with a PATH CHOSEN line.

The audit was told to be strict about what goes live once the gate passes, so it tightened the A=1 layer gate (prepare-only, all inside `_dl3_compare_layer` / `_dl3_alternation_check`):

1. **The A-on stream tolerance went from 2e-2 to 1e-2, and hidden from 5e-2 to 2e-2.** The synthetic stream has rms ~3, so it dilutes the error: 2e-2 on the full stream allowed a residual-update error of about 7% against the parent. Judge test 11 in 34a9be06918d448b had the parent's update at 5.55% of a 0.148 bound and its hidden at 3.23% of 0.086. 1e-2 caps the added update error at about 3.5%, which keeps a worst-case linear sum well inside the bound. A's expected deviation is far smaller: kernel-level q gap 0, O 1e-5, sk output identical. B/C alone already show about 1.7e-3, so the gate should not trip on legitimate output.
2. **A-on cache check gained a rel-L2 bound of 2e-2 on the new K/V rows** against the parent, in addition to the code gap of at most 1. A one-code shift on every element passes the gap test but is about 9% L2, over the judge's 8%.
3. **The alternating two-state graph check now runs all three `_DL3_CASES`.** That adds full kv_scale=1, n=9001, local_start=96. Judge K_SCALE defaults to 1.0, so the scored full decode_1 shape and test 11 run at kv_scale=1, and before this they were covered only eagerly. Cost: one more pair of captures and 6 replays, under a second of prepare time.

An observation, not fixed: in 34a9be06918d448b, alt full kv_scale=0.25 replays 2-5 compared at exactly 0.00e+00, while replays 0-1 and every sliding replay were about 1.7e-3. The most likely cause is that no local experts were routed, so C's numeric difference vanished. It is not structural aliasing, since replays 0-1 differ. Read the new alt full kv_scale=1 lines in the log.

okc_lint: OK with warnings (only the pre-existing host-sync warnings in prepare()). py_compile OK, __pycache__ removed.

Residual risk: A's Q/O kernels meet judge data for the first time in the timed path. The checks bound their deviation from the parent to about 3.5% of the update. That is inside the judge's headroom but rests on synthetic inputs (randn x, non-soft KV), and the worst case is a failed scored run, not a wrong board entry.

## Result

- job `933d3f825a044d36` (submit), status done, passed True, tests 13/13
- speedup 1.3080x, 95% CI [1.2883, 1.3276], score 17294.37614869084 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.3017519911209459, "ci": [1.2822257112541318, 1.32127827098776], "job": "34a9be06918d448b"}; base result: muse_decode_layer/hypotheses/done/neutral/dense-chain-s3-t1-q-sk2-o-route-unpool.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.result.json`, `muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.patch`; branch `hyp/muse_decode_layer/dl3-piece-a-gate-stride-check-fix`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 117.81 | 118.05 | 0.9787 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 130.03 | 122.2 | 1.0338 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 209.22 | 195.64 | 1.0081 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 298.39 | 296.66 | 1.0016 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 236.43 | 239.18 | 1.0043 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 441.9 | 430.66 | 1.0059 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 189.87 | 180.39 | 1.0039 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 199.07 | 199.99 | 0.9971 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 427.37 | 427.64 | 1.0119 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 571.83 | 570.07 | 1.0008 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 600.4 | 606.99 | 1.0155 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 881.43 | 869.34 | 1.0138 | flat |

gained on full decode_1. Expected hit: 7/7. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: dl3-piece-a-gate-stride-check-fix`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

