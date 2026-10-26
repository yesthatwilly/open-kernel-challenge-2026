---
problem: muse_bf16_proj
slug: pooler-k-halves-fp32-partials-restack-m128-m256
created: 2026-10-25T211856Z
base: master
area: pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)
change: >
  DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch
  hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256
  (done/neutral/pooler-k-halves-batched-gemm-m65-m256.patch: _khalf_forward, _khalf_kernel_names,
  _khalf_names_ok, _khalf_select, flag pooler_khalf_ok + dict state['pooler_khalf']) for layer ==
  'pooler' at cfg m in {128, 256} only. Today master runs pooler m=128 as nvjet 128x64 splitK NNT
  20.8 + cublasLt splitKreduce 3.0 us (row 22.85) and m=256 as nvjet 128x64 NNT 28.8 us (row 27.98
  in 741072dc, 29.9-30.05 in today's two other runs). The khalf path: xh = x.view(m, 2,
  4096).permute(1, 0, 2) (strided, no copy), wh = state['pooler_t'].view(2, 4096, 4096) (master's
  existing lossless NN repack, no new weight bytes), p = torch.bmm(xh, wh) -> [2, m, 4096], out =
  p[0] + p[1]. Offer the selector two candidates per m: (a) bf16 partials exactly as ran
  (ce808e5a: nvjet 128x128 NNT 24.7 + add 2.1 at m=256; 20.1 + 1.5 at m=128); (b) fp32 partials: p
  = torch.bmm(xh, wh, out_dtype=torch.float32), then out = torch.empty(m, 4096, bf16);
  torch.add(p[0], p[1], out=out) (one elementwise kernel, fp32 add, one bf16 rounding, the same
  numerics as cuBLAS's own split-K + splitKreduce). If torch.bmm rejects out_dtype on this image,
  print it and offer (a) only. Prefer (b) on a tie within 1%; route (a) only if it beats (b) by >=
  1% (bf16 partials passed the tests already, rel L2 2.45e-3 vs 1.66e-3 master, well under the 1%
  bound). Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph
  replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved
  reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved
  rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two
  picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted
  1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The
  pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the
  chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both
  medians, both IQRs and the kernels. Wire it the way the source branch did: own flag in
  NEW_CELL_FLAGS, inherited by state B via _ab_flags, an AB_CASES entry for pooler m=128 and
  m=256, the kernel_fn branch placed before master's NN torch.mm branch, cfg/shape/flag-only gate,
  no print/try/sync, fresh output. Everything else in master is untouched, m=512/1000 tests stay
  on master.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: K-halves bmm 24.7+2.1 (fp32 add similar) vs nvjet NNT 28.8 kernel; judged 29.94->26.75 on the old base
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: removes the 3.0 us splitKreduce node; judged 23.2->21.43 on the old base
  - shapes: layer=pooler load=decode_* m=*
    direction: flat
    why: m<=32 paths untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "pooler-k-halves-batched-gemm-m65-m256: same mechanism, now rebased onto the combo master 36c26199 (bulk row-owner, Lt cell B), with an fp32-partials candidate added and the hardened two-round 1.03 gate; the bf16-partials form already won pooler verify_16 23.2->21.43 and verify_32 29.94->26.75 on the 67b46a57 base"
review: proceed
status: done
review_at: 2026-10-25T212522Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T212546Z
base_sha: c1d4c033dc83025d0a9fec3357e4d7f493a6c1b0
base_blob: ad8d7254a64a52db2a8f56ec0ad262748153a2e8
branch: hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256
worktree: .worktrees/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T010513Z
impl_commit: 659ff4783582d8103e09cf89dd5fae13e57f96cf
submission_sha256: d451b3ff16827922c05d638b1d478a8b1b7a760dfd5c24280686db285b98cb8b
norm_sha256: 478b2209429ec85fd18bb83e264144195eb06c7769c7a6c9c4e2999fb92989da
submission_blob: ff7580f8edf36c612459db87204b32d412842b5b
lint_warnings: 36
attempt_file: 2026-10-26T012539Z-check-pooler-k-halves-fp32-partials-restack-m128-m256.md
prepared_at: 2026-10-26T012539Z
job: de95f81782374124
dispatched_at: 2026-10-26T012616Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2682668011504035
speedup_ci: [1.2492427991331474, 1.2872908031676593]
harvested_at: 2026-10-26T015638Z
best_at_harvest: 1.2566457115954903
promotion: null
confirm_queued: pooler-k-halves-fp32-partials-restack-m128-m256-cf
---

# pooler-k-halves-fp32-partials-restack-m128-m256

## Change

DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256 (done/neutral/pooler-k-halves-batched-gemm-m65-m256.patch: _khalf_forward, _khalf_kernel_names, _khalf_names_ok, _khalf_select, flag pooler_khalf_ok + dict state['pooler_khalf']) for layer == 'pooler' at cfg m in {128, 256} only. Today master runs pooler m=128 as nvjet 128x64 splitK NNT 20.8 + cublasLt splitKreduce 3.0 us (row 22.85) and m=256 as nvjet 128x64 NNT 28.8 us (row 27.98 in 741072dc, 29.9-30.05 in today's two other runs). The khalf path: xh = x.view(m, 2, 4096).permute(1, 0, 2) (strided, no copy), wh = state['pooler_t'].view(2, 4096, 4096) (master's existing lossless NN repack, no new weight bytes), p = torch.bmm(xh, wh) -> [2, m, 4096], out = p[0] + p[1]. Offer the selector two candidates per m: (a) bf16 partials exactly as ran (ce808e5a: nvjet 128x128 NNT 24.7 + add 2.1 at m=256; 20.1 + 1.5 at m=128); (b) fp32 partials: p = torch.bmm(xh, wh, out_dtype=torch.float32), then out = torch.empty(m, 4096, bf16); torch.add(p[0], p[1], out=out) (one elementwise kernel, fp32 add, one bf16 rounding, the same numerics as cuBLAS's own split-K + splitKreduce). If torch.bmm rejects out_dtype on this image, print it and offer (a) only. Prefer (b) on a tie within 1%; route (a) only if it beats (b) by >= 1% (bf16 partials passed the tests already, rel L2 2.45e-3 vs 1.66e-3 master, well under the 1% bound). Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. Wire it the way the source branch did: own flag in NEW_CELL_FLAGS, inherited by state B via _ab_flags, an AB_CASES entry for pooler m=128 and m=256, the kernel_fn branch placed before master's NN torch.mm branch, cfg/shape/flag-only gate, no print/try/sync, fresh output. Everything else in master is untouched, m=512/1000 tests stay on master.

## Expected

- `layer=pooler load=verify_32 m=256` faster: K-halves bmm 24.7+2.1 (fp32 add similar) vs nvjet NNT 28.8 kernel; judged 29.94->26.75 on the old base
- `layer=pooler load=verify_16 m=128` faster: removes the 3.0 us splitKreduce node; judged 23.2->21.43 on the old base
- `layer=pooler load=decode_* m=*` flat: m<=32 paths untouched

## Risk

Effect: -1.2 to -3 us at 0.184 %/us on verify_32 and -1.4 us at 0.230 %/us on verify_16, about +0.5-0.9% headline: below the ~1.24% one-run margin, so it is expected to land neutral alone and pays inside combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16. Low fault risk (cuBLAS only). Rules: bf16 weights unchanged (master's lossless NN repack), fp32 only in internal partials.

## Adversarial review

Reviewed 2026-10-25T212522Z by an adversarial subagent (model opus).

**Verdict: proceed.** This restacks a measured but never-landed row win (pooler K-halves bmm: judged 23.2->21.43 at m=128 and 29.94->26.75 at m=256 in ce808e5a) onto master 36c26199, adds an fp32-partials candidate and uses the hardened gate. One check reads cleanly from the two pooler rows' paired ratios plus the printed selector and kernel names, so no companion is needed. The main risk is that the hardened gate is a false negative at m=256, and that the fp32 candidate loses there on extra bytes.

### Justification
A. The mechanism is measured, not conjectured. In ce808e5a, candidate N (the NN halves, view of pooler_t) ran nvjet 128x128 NNT 24.7 us + add 2.1 us at m=256, against master's nvjet 128x64 NNT 28.8. At m=128 it ran nvjet 128x64 NNT 20.1 + add 1.5, against master's splitK NNT 20.8 + splitKreduce 3.0 (FINDINGS: master 741072dc rows 22.85 / 27.98). The floors are 14.6 / 17.4 us, so both rows are still 1.3-1.6x off their floors and the direction holds. The expected shapes match TASK.md: only the two verify pooler rows move, and decode m<=32 is routed earlier in kernel_fn and stays untouched. No test runs pooler at m=128/256, and m=512/1000 stay on master. The prior field is accurate. pooler-k-halves-batched-gemm-m65-m256 is the same mechanism, and FINDINGS' auto-generated 'Do not try' lists it as neutral/within-ci. But its rows won and it was never landed (DIAGNOSIS #7, Action 1), so this is a restack, not a retry; the differences (new base, fp32 candidate, harder gate) are real. The run is still expected to come out neutral on the headline: +0.5-0.9% against a margin of at least 1.5%. Its value is per-row confirmation on the current base and acting as the depends_on parent for combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16. WEAKEST LINK: the new gate against the measured margins. In ce808e5a's own prepare log the single-round master/cand was 1.069 at m=128 but only 1.037 at m=256 (IQR 0.90 vs 1.34), while the judge gave about 1.12 at m=256. The prepare gate under-predicts this cell. Two independent rounds that must both be >= 1.03 will plausibly drop m=256 to master by chance, which would be a gate false negative, not a refutation. Second-weakest: candidate (b) at m=256 is not free. Its fp32 partials double the GEMM epilogue write (8 MB vs 4 MB) and the add's read (8 MB fp32 vs 4 MB bf16). A mixed-dtype torch.add(out=bf16) also leaves PyTorch's vectorized_elementwise path for the slower dynamic-cast/unrolled one. I estimate about +1.5-2.5 us over (a), which would put (b) near parity with master at m=256 (about 27-29 vs 28.8). The fp32 C type may also change cuBLAS's tile heuristic for the bmm, away from the 128x128 that made (a) win. (b) is likelier to pay at m=128, where the incumbent already runs a fp32 splitKreduce.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256 (landed): it owns nn_p256_ok and the incumbent at m=256 (NN torch.mm on pooler_t), and candidate N reuses pooler_t. The gate's 'flag off' arm must run master's NN path, so nn_p256_ok must already be True when the khalf gate runs inside _prepare_state. Otherwise the baseline silently becomes dispatch TNT (about 30-31 us) and the gate is inflated.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (landed): it owns nn_ok, which is shared with unpooler m=128. Master's _ab_component(layer, m) has no state argument and maps pooler m=128 to nn_ok, and pooler 128/256 are already AB_CASES entries. Unless _ab_component returns pooler_khalf_ok first for the cells that were routed (the prior patch added a state parameter to do this), a self-check miss on the khalf path turns off nn_ok as well, and the resulting unpooler verify_16 regression would be misread as this hypothesis's effect. Do not add duplicate AB_CASES entries; remap the component instead.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (landed, the current base): it adds the cuBLASLt qkv m=128 cell (own workspace and plan) and bulk m=1 cells, so qkv_gate shares the verify_16 load with the pooler but not its graph. The judge times each shape in its own graph with a cold L2, so I expect no direct interaction. The one effect is that its _cell_gate/_ab_flags machinery is what the new flag must plug into (NEW_CELL_FLAGS and second-state inheritance).
- sglang-gemm-dispatch-parity (landed): it remains the fallback when nn_ok or nn_p256_ok is off. If FINAL FLAGS shows either NN flag False, the comparison baseline has shifted and attribution against 741072dc is void.

### Interference and A/B needs
No companion. (i) An ab-revert of either NN-layout hypothesis would recreate a stale incumbent, and candidate N needs pooler_t anyway. (ii) There is nothing to tune: two halves is the conjecture, and the bf16-vs-fp32 question is answered in-process by the selector, which times master, (a) and (b) on the same synthetic x in the same run. (iii) The bf16-partials control already exists as ce808e5a, the same mechanism judged on the previous base, with kernel names recorded. A gain or a loss is therefore readable from one run if the build prints the following for both m: both rounds' medians and IQRs for master, (a) and (b); the kernel names for (a) and (b), even when not routed; which candidate was routed; and FINAL FLAGS (nn_ok, nn_p256_ok, pooler_khalf_ok). Judge rows by the per-shape paired speedup against SGLang, not by absolute microseconds. Master's own pooler m=256 row varied 27.98-30.05 us across today's runs (about 7%), so a us-delta against one prior run is not reliable. If the gate keeps master at m=256, log it as 'gate false negative / not measured', not as a refutation, since the judge already measured about 1.12x there.

### Open questions
- Does torch.bmm(xh, wh, out_dtype=torch.float32) exist and accept the strided xh (strides 4096, 8192, 1) on torch 2.14.1 without inserting a contiguous() copy? Print xh.stride() and assert no copy kernel in the profile, as the prior run did.
- Will the profiler name assertion accept candidate (b)'s add kernel? A mixed-dtype add may launch unrolled_elementwise_kernel rather than vectorized_elementwise_kernel, and a name filter copied from the prior patch could falsely reject (b).
- Does a fp32 C type change cuBLAS's bmm tile choice at m=256 away from 128x128 (the source of the win)? Print (b)'s GEMM kernel name.
- Master's _cell_gate returns medians only. The hardened IQR <= 1.5x criterion needs an extended gate that also returns IQRs; the prior run's N at m=128 had IQR 0.96 against master 0.70 (1.37x), close to the limit.
- The single-round m=256 margin was 1.037 while the judge measured about 1.12. Should m=256 keep the 1.03 two-round bar (risk: a false negative that makes the run a no-op on that row), or should the gate record both rounds so the orchestrator can decide? At minimum, the outcome must distinguish 'gate rejected' from 'row lost'.
- The queued combo depends_on this branch. If the gate routes only m=128 here, does the combo re-gate m=256 itself, or inherit the master route?


## Implementation notes

Edited only `<worktree>/muse_bf16_proj/submission.py` (base blob ad8d7254, +279/-15).

What changed:
- New constants `KHALF_*` and per-m flags `pooler_khalf128_ok` / `pooler_khalf256_ok` (appended to `NEW_CELL_FLAGS`, so `_ab_flags` hands them to state B); per-state dict `state['pooler_khalf_kind']` ({m: 'a'|'b'}), also inherited by state B.
- `_khalf_forward`: xh = x.view(m,2,4096).permute(1,0,2) (strided), wh = pooler_t.view(2,4096,4096) (master's existing NN repack, no new bytes). Kind a: bf16 bmm + add. Kind b: `torch.bmm(..., out_dtype=torch.float32)`, then `torch.add(p[0], p[1], out=<fresh bf16>)`.
- `_khalf_route_ok` (cfg/shape/flag only) and a kernel_fn branch placed immediately before master's NN torch.mm branch, under `layer != "router"`; no print/try/sync.
- `_khalf_prepare` (state A only, after `_lt_prepare`, own start deadline 170 s, prints `skipped: deadline`) and `_khalf_cell` per m. Per candidate: `_cell_check` through kernel_fn (rel L2 vs fp64, power-of-two probe row, bit-identical rerun), then `_cell_kernels` profiler assertion (needs a GEMM kernel; rejects copy/memcpy and splitKreduce; elementwise names are only printed, so a mixed-dtype add cannot falsely reject b). Then 2 independent interleaved rounds (41 reps, 256 MB flush, own empty-graph baseline each) of master's ACTUALLY ROUTED path (kernel_fn with the flag off, so nn_ok / nn_p256_ok already True) against each valid candidate. Route only if EVERY round has master/cand >= 1.03 AND cand IQR <= 1.5x master IQR. Prefer b; route a only if it passes and its min round ratio >= 1.01x b's (or b fails). Prints xh.stride(), master kernels, both candidates' kernels (even if unrouted or excluded), every round's medians/IQRs, and one PATH CHOSEN line per m that says "gate rejected, not a measured loss of the row" vs a candidate.
- `_ab_component(layer, m, state)` now returns the K-halves flag FIRST for a routed pooler 128/256 cell, so a miss turns off only that flag (master NN path serves it), never `nn_ok` (shared with unpooler m=128). No duplicate AB_CASES entries.
- `_path_print` helper: PATH CHOSEN / gate lines of the existing cells (ouns, router cluster, bulk A/A2/A3, Lt B) and the new khalf cells now also go to `_PATH_LINES` (host strings), and prepare() repeats them as `[summary]` lines AFTER FINAL FLAGS, as the last lines of prepare (long lines cut at 260 chars, last 40 kept). This is a print-only change to the old cells.

Review fixes applied: (1) gate baseline is master's routed flag-off path with nn flags True (khalf runs inside `_prepare_state`, before the A/B check); (2) `_ab_component` remap, no duplicate AB_CASES; (3) hardened two-round 1.03 + IQR 1.5x gate (extended gate returns IQRs, written as `_khalf_cell` since `_cell_gate` returns medians only); (4) FINAL FLAGS prints nn_ok, nn_p256_ok and both khalf flags plus kinds; (5) profiler assertion does not match elementwise names. Open questions answered in code: out_dtype and xh strides are probed and printed (b excluded if bmm rejects it; copy kernel in profile excludes a candidate); b's GEMM kernel name is printed; m=256 keeps the 1.03 two-round bar, but the PATH CHOSEN line separates 'gate rejected' from a lost row, and both rounds are printed so the orchestrator can decide; m=128 and m=256 are gated and flagged independently, and the combo parent inherits whatever each cell routed (flags are per cell, per state).

Deviations: none from the review. Compile/run risks are listed in the report (bmm out_dtype availability, bf16 candidate a rel L2 2.45e-3 against the 3e-3 CELL_REL_L2 canary bound).

## Rules check

- [x] CUDA-graph capturable: kernel_fn branch is torch.bmm/torch.add/torch.empty on shapes and flags only; no .item()/.cpu()/sync/print/try (all syncs, prints and try blocks are in prepare()).
- [x] Nothing derived from `data` is cached: only the kind choice from synthetic prepare-time x, stored per state; kernel_fn reads x and cfg only.
- [x] Weights: no new weight bytes; reuses master's lossless bf16 NN repack pooler_t (view only). fp32 only in internal partials.
- [x] Activations BF16 where SGLang keeps BF16: x and the output are bf16; the output of both kinds is bf16 (b: one rounding from an fp32 add).
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only torch ops, torch.profiler in prepare and prints.
- [x] All work happens inside the timed call; no output caching or replay (fresh output tensor per call).
- [x] Review required fixes applied (see Implementation notes).

okc_lint: `OK with warnings: muse_bf16_proj/submission.py` (only pre-existing-style host-sync/dtype warnings in prepare(); no errors). `python3 -I -m py_compile` passed; __pycache__ deleted.

## Pre-flight audit

Audited 2026-10-25 (opus) by reading, plus a numpy emulation of the numerics. **Verdict: dispatch** (no fixes).

- Candidate b fallback: the out_dtype probe in `_khalf_prepare` sits inside its own try and only prints. If torch.bmm rejects out_dtype, b raises inside `_cell_check` within `_khalf_cell`'s per-candidate try. b is then printed as EXCLUDED and never becomes a kind, and a runs alone. Every outer layer (the per-m try and the whole-function try) falls back to master with a PATH CHOSEN line. The parity base already uses torch.mm(out_dtype=torch.float32) on this image, so bmm's overload is very likely present.
- Landed cells: with `_path_print(` mapped back to `print(`, the only differences from master are the new khalf code, the NEW_CELL_FLAGS extension, `_ab_component`'s state arg and its khalf-first check, the B-state inherit, the FINAL FLAGS/summary prints and the kernel_fn branch. No routing or flag logic changed in ouns, router cluster, bulk A/A2/A3 or Lt B. The new kernel_fn branch fires only for pooler with a khalf flag set, and it sits before master's NN torch.mm. No earlier branch takes pooler at m=128/256.
- Numerics (numpy emulation, m=128/256): rel L2 is 1.66e-3 for master, 2.45e-3 for a and 1.66e-3 for b. The canary bound is 3e-3 and the A/B bound is 6e-3. Probe rows are exact for both kinds, so no gate is always-fail.
- Rules: the kernel_fn branch reads only cfg, shapes and flags, with no sync, print or try, and it returns a fresh output. pooler_t is master's lossless bf16 NN repack. fp32 appears only in the internal partials.

Residual risk: the hardened gate (two rounds at 1.03, IQR <= 1.5x) may keep master at m=256, which is a gate false negative and is printed as one. Candidate b is excluded if cuBLAS picks a split-K plan for the fp32 bmm.

## Result

- job `de95f81782374124` (check), status done, passed True, tests 9/9
- speedup 1.2683x, 95% CI [1.2492, 1.2873], score 3808.9777289905714 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.patch`; branch `hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.54 | 10.92 | 0.9965 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 6.11 | 1.0129 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 17.72 | 17.81 | 1.0033 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.01 | 4.96 | 1.0347 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.17 | 4.87 | 1.0404 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.64 | 12.99 | 1.0707 | gained |
| layer=o load=decode_16 m=16 | 62 | 6.21 | 5.9 | 1.028 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.46 | 19.52 | 0.9996 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.82 | 18.4 | 1.0 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.51 | 5.5 | 0.9994 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.71 | 13.63 | 1.0455 | gained |
| layer=o load=decode_32 m=32 | 62 | 6.47 | 6.35 | 0.9949 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.86 | 19.98 | 0.9943 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.49 | 18.72 | 1.0064 | flat |
| layer=router load=decode_32 m=32 | 62 | 5.95 | 6.31 | 0.956 | lost |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.35 | 12.74 | 0.9963 | flat |
| layer=o load=verify_1 m=8 | 62 | 5.9 | 6.12 | 0.9874 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.31 | 19.27 | 1.0058 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 5.54 | 5.27 | 1.012 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.49 | 5.49 | 0.9995 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 15.55 | 15.99 | 1.0147 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.68 | 7.54 | 1.0044 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 22.85 | 21.36 | 1.0677 | gained |
| layer=unpooler load=verify_16 m=128 | 62 | 18.88 | 18.54 | 1.0086 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.61 | 7.54 | 1.0138 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.46 | 19.7 | 0.9918 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.44 | 9.23 | 0.9998 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 27.98 | 28.59 | 0.9876 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.18 | 24.54 | 1.0085 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.72 | 9.24 | 1.0382 | gained |

gained on unpooler decode_1; router decode_1, verify_32; qkv_gate decode_16, decode_32; pooler verify_16; lost on router decode_32. Expected hit: 2/3. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

