---
problem: muse_bf16_proj
slug: unpooler-k-halves-batched-gemm-m65-m128
created: 2026-10-25T153141Z
base: master
area: unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM)
change: >
  Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192)
  and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=130 and m=1000 tests stay on
  master. WHY: master runs torch.mm on the NN weight here, nvjet 128x64 NNT, 20.4 us kernel /
  18.95 row against a 14.64 floor (about 3.5 TB/s). That grid is 128 CTAs of 64 weight columns x
  full k=4096, so each CTA re-reads the whole 128 x 4096 x token tile (1 MB) from L2. x traffic is
  then about 128 MB against 67 MB of weight. This is the 'x K-slice re-read about 2x the weight
  bytes' that REWRITE_READ_2026-10-25 named as the m=128 mainloop limiter. CHANGE: xh = x.view(m,
  2, 2048).permute(1, 0, 2) (strided, no copy); wh = state['unpooler_t'].view(2, 2048, 8192) (the
  existing NN repack, contiguous halves); p = torch.bmm(xh, wh) -> [2, m, 8192]; out =
  torch.add(p[0], p[1]). With n=8192 and k=2048 per batch, cuBLAS can use 128x128 tiles (64
  N-tiles x 2 batches = 128 CTAs), which halves the x bytes per weight byte. A TN-view candidate
  on state['unpooler'] goes into the same selector. Precision as the pooler halves (exact
  single-nonzero rows, two bf16 roundings). Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Every
  other layer and m stays byte-identical, including the ouns slice kernel and the swap-AB paths.
expected:
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "NN nvjet 20.4 kernel / 18.95 row, floor 14.64; halving x L2 traffic should give a kernel of ~16.5-17.5 plus an add of ~1.0: row ~17.5-18.3"
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T203723Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T203723Z
base_sha: 2178a1d62536d4e2c94af82772ab6316f61d8598
base_blob: ad8d7254a64a52db2a8f56ec0ad262748153a2e8
branch: hyp/muse_bf16_proj/unpooler-k-halves-batched-gemm-m65-m128
worktree: .worktrees/muse_bf16_proj/unpooler-k-halves-batched-gemm-m65-m128
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T204107Z
impl_commit: 03252b00cd6f9077e723eff868944d344746d121
submission_sha256: 893edf99de9bc7dceb9a5198428741f4a5b81449e347e4a3e7e270e0feee4d17
norm_sha256: 1ddd2b82957d3794fddce4dd750dbca0debc391cc7e4274f303c0e1330218ffb
submission_blob: 572599cf4876a6353e8acc95486e0bb0be34586c
lint_warnings: 39
attempt_file: 2026-10-25T204107Z-submit-unpooler-k-halves-batched-gemm-m65-m128.md
prepared_at: 2026-10-25T204107Z
job: e82feba3dd8d4efb
dispatched_at: 2026-10-25T204115Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.245209797855405
speedup_ci: [1.226531650887574, 1.263887944823236]
harvested_at: 2026-10-25T204640Z
best_at_harvest: 1.2566457115954903
promotion: null
---

# unpooler-k-halves-batched-gemm-m65-m128

## Change

Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192) and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=130 and m=1000 tests stay on master. WHY: master runs torch.mm on the NN weight here, nvjet 128x64 NNT, 20.4 us kernel / 18.95 row against a 14.64 floor (about 3.5 TB/s). That grid is 128 CTAs of 64 weight columns x full k=4096, so each CTA re-reads the whole 128 x 4096 x token tile (1 MB) from L2. x traffic is then about 128 MB against 67 MB of weight. This is the 'x K-slice re-read about 2x the weight bytes' that REWRITE_READ_2026-10-25 named as the m=128 mainloop limiter. CHANGE: xh = x.view(m, 2, 2048).permute(1, 0, 2) (strided, no copy); wh = state['unpooler_t'].view(2, 2048, 8192) (the existing NN repack, contiguous halves); p = torch.bmm(xh, wh) -> [2, m, 8192]; out = torch.add(p[0], p[1]). With n=8192 and k=2048 per batch, cuBLAS can use 128x128 tiles (64 N-tiles x 2 batches = 128 CTAs), which halves the x bytes per weight byte. A TN-view candidate on state['unpooler'] goes into the same selector. Precision as the pooler halves (exact single-nonzero rows, two bf16 roundings). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Every other layer and m stays byte-identical, including the ouns slice kernel and the swap-AB paths.

## Expected

- `layer=unpooler load=verify_16 m=128` faster: NN nvjet 20.4 kernel / 18.95 row, floor 14.64; halving x L2 traffic should give a kernel of ~16.5-17.5 plus an add of ~1.0: row ~17.5-18.3
- `layer=* load=decode_* m=*` unchanged: untouched

## Risk

Weakest of the four primaries (best case about 1.4 us on one row, about +0.3% headline). Its value is diagnostic: it tests the x-re-read limiter that bf16-verify16-cutlass-m128-onetile-splitk-rewrite is built to remove. If cuBLAS keeps 128x64 tiles for the halves, the add makes it lose, and the selector should keep master.

## Adversarial review

Reviewed 2026-10-25T203723Z by an adversarial subagent (model opus).

**Verdict: proceed.** Real, untried variant of pooler-k-halves-batched-gemm-m65-m256 (own rows 1.063/1.123 paired; selector and profiler machinery worked under a scored submit). Weaker case: the add costs ~2 us (not 1), so the batched GEMM must reach <= ~17.3 us vs 19.3-19.4 today (~11%) from a tile change alone. The prepare-time selector makes a loss safe and readable.

### Justification
master unpooler verify_16 m=128: torch.mm(x, unpooler_t) under nn_ok, nvjet 128x64 NNT 19.3-19.4 us, row 18.9-19.15, floor 14.64. 128 CTAs of 128x64 over k=4096 pull ~128 MB of x via L2 vs 67 MB weight; 2-batch k=2048 128x128 halves it; L2 limiter plausible, not shown. The add moves 4 MB read / 2 MB write (pooler m=256 analogue 2.1 us); realistic row 18.5-19.3. Sibling at m=128 picked 128x64 (bmm 20.1 vs 19.4 us); at m=256 128x128. Weakest link: an ~11% GEMM speedup from a tile change on the shape where cuBLAS prefers 128x64. Prior (missing field): pooler-k-halves-batched-gemm-m65-m256; difference is real (tile-deficit fix vs x-L2 re-read on a matched tile).

### Confounders on the current kernel
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (landed): owns unpooler_t and shared nn_ok (pooler m=128 and unpooler m=128); give the path its own flag (unpooler_khalf_ok) in NEW_CELL_FLAGS checked before nn_ok in _ab_component; a miss falls back to master's NN mm.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (landed): follow the NEW_CELL_FLAGS/_ab_flags/B-state pattern; select once on state A.
- sglang-gemm-dispatch-parity: fallback if nn_ok off; record nn_ok in the attempt file.
- pooler-nn-layout-widen-m256: insert the route without reordering the nn_p256_ok condition.

### Interference and A/B needs
No companion: the selector times master's exact path vs both candidates (N view, T view) with cold L2 and names the kernel; log medians+IQR, bmm-only median, kernel names, xh.stride(), final flags; 'refuted at prepare-time' if master kept.

### Open questions
- Port _khalf_forward/_khalf_select/relaxed _khalf_names_ok from the sibling branch/patch.
- Stack the sibling pooler k-halves later (separate hypothesis).
- Correct expected row to ~18.5-19.3; break-even bmm <= ~17.3 us.
- Does cuBLAS pick 128x128 for (b=2, m=128, n=8192, k=2048)? If 128x64, follow up with a pinned cuBLASLt algo.
- Add prior: pooler-k-halves-batched-gemm-m65-m256.
- Overlaps queued bf16-verify16-cutlass-m128-onetile-splitk-rewrite (re-baseline).


## Implementation notes

Only `muse_bf16_proj/submission.py` in the worktree changed (+197/-4). Ported `_khalf_forward`, `_khalf_kernel_names`, relaxed `_khalf_names_ok` and `_khalf_select` from the sibling pooler patch, retargeted at the unpooler (n 8192, k 4096, halves of 2048; candidate N = `unpooler_t.view(2, 2048, 8192)`, candidate T = TN view of `state['unpooler']`, strides (2048, 1, 4096)).
- Own flag `unpooler_khalf_ok` (+ per-m dict `state['unpooler_khalf']`) added to NEW_CELL_FLAGS; `_ab_flags` also carries the dict; state B inherits both; `_ab_component` gained a `state` arg and checks the flag BEFORE the nn_ok range, so a miss falls back to master's NN `torch.mm` and never turns nn_ok off. nn_p256_ok condition untouched.
- Selector runs once on state A (step 10 in `_prepare_state`, after `_lt_prepare`): master exact path vs N vs T, synthetic x, 256 MB flush, 41 interleaved reps, empty-graph baseline, >= 1.02x to switch, reject > 6e-3 rel L2. Prints medians+IQR, bmm-only medians, kernel names, xh.stride(), PATH CHOSEN; FINAL FLAGS line also prints unpooler_khalf. Any exception keeps master. No exact kernel count; only copy/contiguous kernels or splitKreduce disqualify; empty events print "name check skipped".
- Scope: exact m == 128 only (UNPOOLER_KHALF_MS = (128,)), per the task, narrower than the md's 65..128.
- Deviation: kernel_fn also checks cheap host-side shape preconditions (x.dim()==2, shape[0]==cfg m, shape[1]==4096, is_contiguous) so the `view` can never raise; no print/try/state mutation there. Profiler failure inside the selector returns [] (check skipped) instead of aborting.

## Rules check

- [x] CUDA-graph capturable: kernel_fn path is view/permute/bmm/add only, no sync or value-dependent control flow (checked in diff).
- [x] Nothing derived from `data` is cached across calls: the selector uses synthetic x in prepare(); state holds only weight views/choice.
- [x] Weights: no new repack; reuses the existing lossless unpooler_t and views of the original weight, BF16 stays BF16.
- [x] Activations BF16: bmm outputs bf16, add in bf16 (two roundings, as the pooler halves; verified at prepare vs 6e-3 rel L2).
- [x] Attention sees every key: n/a: projection problem, no attention.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (diff adds only torch calls and torch.profiler).
- [x] All work inside the timed call; no output caching or replay.
- [x] Marker `# okc: problem=muse_bf16_proj` kept; no triton.autotune added; no host syncs in kernel_fn.

Lint (`okc_lint.py`): `OK with warnings` (0 errors; warnings are the existing and new prepare()-only synchronize/torch.empty uint8 flush buffers, outside kernel_fn). `py_compile` clean, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only; no GPU). **Verdict: dispatch** (no fixes).
- Gates: the selector compares master NN mm vs N/T bmm+add on synthetic x with the A/B self-check bound (6e-3; two bf16 roundings is about 2-3e-3). The relaxed name check rejects only copy/contiguous or splitKreduce kernels, and an empty profile skips the check. It cannot always fail. The A/B unpooler m=128 cell (n=8192) exercises kernel_fn on the new path in states A and B, and B inherits the flag and the per-m dict.
- Fallback: in _ab_component, unpooler_khalf_ok is checked before the nn_ok range, so a miss switches off only that flag. kernel_fn then falls through to master NN torch.mm, and nn_ok (pooler m=128) is untouched. A selector exception, an invalid candidate, or a ratio below 1.02x keeps master and prints PATH CHOSEN. A failure building state B turns off every NEW_CELL_FLAGS entry, including this one.
- kernel_fn: the added checks (x.dim, x.shape, is_contiguous, a cfg m dict lookup) read only metadata and are constant per config. They do no sync or allocation, and graph capture is not affected. The path is view/permute/bmm/add. No earlier route (ouns m<=8, bulk m=1, swapab16 n=1024, unpool m<=16, swapab 17..32) catches unpooler m=128 n=8192. m=130 and m=1000 stay on master. The judge x is contiguous.
- Correctness and rules: single-nonzero probe rows are exact (the other half is exactly 0). The weights are views of the existing BF16 unpooler/unpooler_t with no new repack, and nothing derived from data is cached. Lint is OK with warnings (prepare-only syncs) and py_compile is clean.
- Residual risk: cuBLAS may keep 128x64 tiles for the halves (as the pooler sibling saw at m=128). The selector then keeps master and the row is neutral (refuted at prepare time).

## Result

- job `e82feba3dd8d4efb` (submit), status done, passed True, tests 9/9
- speedup 1.2452x, 95% CI [1.2265, 1.2639], score 3879.5068978094864 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/unpooler-k-halves-batched-gemm-m65-m128.result.json`, `muse_bf16_proj/hypotheses/done/neutral/unpooler-k-halves-batched-gemm-m65-m128.patch`; branch `hyp/muse_bf16_proj/unpooler-k-halves-batched-gemm-m65-m128`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.54 | 11.16 | 0.9612 | lost |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 5.76 | 1.0037 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 17.72 | 17.88 | 1.0007 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.01 | 4.94 | 1.024 | flat |
| layer=router load=decode_1 m=1 | 62 | 5.17 | 5.03 | 1.0235 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.64 | 13.3 | 0.9574 | lost |
| layer=o load=decode_16 m=16 | 62 | 6.21 | 6.19 | 0.9847 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.46 | 19.44 | 1.0037 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.82 | 18.49 | 1.0004 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.51 | 5.61 | 0.9599 | lost |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.71 | 14.34 | 0.9613 | lost |
| layer=o load=decode_32 m=32 | 62 | 6.47 | 6.76 | 0.9828 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.86 | 20.15 | 0.9946 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.49 | 18.53 | 1.0108 | flat |
| layer=router load=decode_32 m=32 | 62 | 5.95 | 6.15 | 0.9797 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.35 | 13.14 | 0.965 | lost |
| layer=o load=verify_1 m=8 | 62 | 5.9 | 6.31 | 0.9711 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.31 | 19.15 | 1.0123 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 5.54 | 5.38 | 1.0327 | gained |
| layer=router load=verify_1 m=8 | 62 | 5.49 | 5.2 | 1.0507 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 15.55 | 16.14 | 0.9853 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.68 | 7.32 | 1.0059 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 22.85 | 22.97 | 0.954 | lost |
| layer=unpooler load=verify_16 m=128 | 62 | 18.88 | 18.81 | 0.9893 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.61 | 7.56 | 1.0088 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.46 | 19.54 | 0.9849 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.44 | 9.42 | 1.0153 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 27.98 | 30.05 | 0.9704 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.18 | 24.4 | 1.0128 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.72 | 9.43 | 1.0292 | flat |

gained on unpooler verify_1; router verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; router decode_16; pooler verify_16. Expected hit: 0/2. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: unpooler-k-halves-batched-gemm-m65-m128`) that adds a further change in the same area is the only way to learn more.

