---
problem: muse_bf16_proj
slug: router-pdl-plus-khalf-pooled-iqr-gate
created: 2026-10-27T143409Z
base: master
area: router m=9..256 PDL (restack) + pooler m=128 khalf gate rule
change: >
  Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all
  4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the
  read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean)
  on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128,
  whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6
  and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held)
  showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master
  IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to
  _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >=
  1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across
  the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)).
  To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict
  (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs,
  the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the
  DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.
expected:
  - shapes: layer=router load=decode_16 m=16
    direction: faster
    why: "router PDL routed at m=16 in a20f8048 (gate x1.040/1.091); judged row 5.27 vs 5.48-5.73 us"
  - shapes: layer=router load=decode_32 m=32
    direction: flat
    why: "router PDL routed at m=32 (gate x1.086/1.101) but the judged row was flat in a20f8048 (5.96 vs 5.94-6.10 us)"
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "router PDL routed at m=128 (gate x1.031/1.059); judged row 7.06 vs 7.39-7.95 us (likely)"
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "router PDL routed at m=256 (gate x1.057/1.041); judged row 9.06 inside master's spread (marginal)"
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "khalf routes on most runs under the pooled rule (b2 won every one of 8 recorded rounds at x1.025-1.054): keeps the ~-1.2 us judged pooler m=128 row"
  - shapes: "*"
    direction: flat
    why: every other cell unchanged
risk: low
depends_on: router-zero-pdl-overlap-m9-m256
priority: 1
kind: primary
controls_for: null
prior: "router-zero-pdl-overlap-m9-m256: same router code, which routed everywhere but lost the headline to a khalf gate miss in the same run; the khalf part is the pooled-IQR rule the khalf-gate-three-rounds-pooled adversary prescribed (that parent was held because its rule kept the per-round IQR test)"
review: proceed
status: running
review_at: 2026-10-27T143812Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-27T143812Z
base_sha: 15558ffa690a1e3aa20c39c6efdd22a6ec73a644
base_blob: d51c745a0e43d20ebf2cbb3d582f3f8ead72f039
branch: hyp/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate
worktree: .worktrees/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate
mode: check
restack_of: router-zero-pdl-overlap-m9-m256
stale_base: null
implemented_at: 2026-10-27T144321Z
impl_commit: 76bb1188e0eea0d083961bdaf0feab23ff0883c5
submission_sha256: 6339277e6610988dbddc6a41e89e5613779a671e4ac82e84c22d1d35138e6af5
norm_sha256: 0f9b16f298a27e49e7feaea77fe2a8176c9bd51dd7aef5669cf3c0d69818c843
submission_blob: ada0503dc81049c9bfc984ad24101c09d992438f
lint_warnings: 54
attempt_file: 2026-10-27T144321Z-check-router-pdl-plus-khalf-pooled-iqr-gate.md
prepared_at: 2026-10-27T144321Z
job: 023f4260e86e423e
dispatched_at: 2026-10-27T144336Z
---

# router-pdl-plus-khalf-pooled-iqr-gate

## Change

Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all 4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean) on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128, whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6 and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held) showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >= 1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)). To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs, the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.

## Expected

- `layer=router` m=16/32/128/256 only (corrected 2026-10-27 at implementation; `layer=router *` was too broad: m=1 and m=8 take the untouched router_cluster path and cannot move): faster at m=16 (clear) and m=128 (likely), m=256 marginal, m=32 expected flat. The judged router gain in a20f8048 was ~0.6-1.0 us summed over the four rows (m=32 flat), ~0.2-0.3% headline, NOT the gate's -0.2 to -0.45 us per row (x1.03-1.10).
- `layer=pooler load=verify_16 m=128` faster: khalf routes on most runs under the pooled rule (b2 won every one of 8 recorded rounds at x1.025-1.054): keeps the ~-1.2 us judged pooler m=128 row
- `*` flat: every other cell unchanged

## Risk

Headline effect ~+0.3-0.6% combined (router ~0.3% + khalf ~0.27% when it would otherwise miss): near the single-run readability line, so read the R-PDL and K PATH CHOSEN lines. The 1.2735 board run had khalf OFF, so a run with both ON can plausibly beat it; a check that beats the board gets a scored run.

## Adversarial review

Reviewed 2026-10-27T143812Z by an adversarial subagent (model opus).

**Verdict: proceed.** A reasonable bid to beat the board: it restacks the router PDL change (routed in all four classes) and adds the pooled-IQR khalf rule; the parent patch applies to the same base blob and breaks no rule. One run cannot read the headline (~+0.5% vs CI ~1.5%): judge on the R-PDL and K PATH CHOSEN lines and the judged rows. The khalf half is the same routing rule the queued control khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr implements: fold that control's diagnostic in here or run them in sequence.

### Justification
Router PDL is real: gates passed in all four classes, the nopdl arm equals master, the race checks are clean and the no-wait control was wrong 32/32 times. Judged rows only partly agree vs three master-lineage runs: m=16 5.27 vs 5.48-5.73 (~-0.36 us, clear); m=128 7.06 vs 7.39-7.95 (likely); m=32 5.96 vs 5.94-6.10 (flat despite ~0.4 us at the gate); m=256 9.06 inside master's spread: ~0.6-1.0 us summed, ~0.2-0.3% headline. Khalf: the IQR clause binds (>= 5 of 10 logged rounds failed it), but a20f8048 r1 was 1.027 (< 1.03), so that miss was ratio and IQR. Weakest link: the candidate is genuinely wider (summed IQRs cand/master ~1.42 vs a 1.5 bar), so a 123-rep pooled test at 1.5x is near a coin flip (~55-70%). Median-of-3 >= 1.03 passes ~90%. 'layer=router *' is too broad: m=1/8 take the untouched cluster path.

### Confounders on the current kernel
- 8f233e88 (read-flush gates + khalf b2): R-PDL classes are themselves marginal (m=128 r1 1.031): read FINAL FLAGS before any row.
- pooler-nn-layout-widen-m256 / restack-three-cell-wins: the khalf master side's IQR (0.45-0.90 us) and the 0.5 us floor set the bar.
- router hilo / swapab designs are preconditions; the flat judged m=32 row suggests less of the fill is hidden there.
- router_kernel<8,8> at m<=8: m=1/8 rows cannot move.
- bulk12 m=1: whole-run drift 1.7-3.6% exceeds this effect; headline comparisons with bc147c5b mean nothing.

### Interference and A/B needs
No new companion. Fold the queued control's diagnostic into this build: KHALF_ROUNDS=10, route on POOLED over rounds 1-3, print the per-round IQR-fail and ratio<1.03 fractions plus OLD/PARENT/POOLED over triples 1-3, 4-6, 7-9 (~0.7 s); then retire the control.

### Open questions
- Re-base the pooled KHALF_MAX_IQR: the candidate's true IQR is ~1.42x master's vs a 1.5x bar; a pooled 2.0x bar or p90(cand) < median(master) guards bimodality without failing ~40% of real wins.
- Why did the judged m=32 router row not move? If repeated, decide whether router_pdl_ok_32 should route.
- Give R-PDL the same median-of-3 rule so its classes do not flicker.
- Store raw times in the parent's modified _cell_gate (extra-arm version) for cand and master only; keep R-PDL callers' prints and stats keys.
- A scored run's FINAL FLAGS must show pooler_khalf128_ok and all four router_pdl_ok flags.


## Implementation notes

Worktree `.worktrees/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate`, `muse_bf16_proj/submission.py` only. The parent
router-zero-pdl-overlap-m9-m256 patch (claim's restack) is staged and untouched; this change is the unstaged diff on top
(`git diff`: 1 file, +119 / -24). Nothing in kernel_fn or any kernel changed; every edit is prepare()-time gate logic.

- `_cell_gate` (the parent's `extra=` version): when `stats` is passed, cand's and master's entries also get
  `"times"` = the raw per-rep us list (`list(v)`). Extra arms (R-PDL's `nopdl`) get no `times`. Return value, prints and
  every existing stats key unchanged, so the R-PDL callers behave byte-identically (they only read ratio / net / iqr).
  Docstring updated.
- Constants: `KHALF_ROUNDS` 2 -> 10; new `KHALF_DECIDE_ROUNDS = 3`, `KHALF_FLOOR_RATIO = 1.01`,
  `KHALF_POOLED_IQR_MAX = 2.0`; `KHALF_MIN_RATIO = 1.03` (now the median bar, and OLD's per-round bar) and
  `KHALF_MAX_IQR = 1.5` (now OLD/PARENT diagnostic only) kept; `KHALF_IQR_FLOOR_US = 0.5` unchanged.
- New helpers before `_khalf_prepare`: `_khalf_iqr` (same quartile convention as `_cell_gate`), `_khalf_round_iqr_ok`
  (per-round 1.5x clause), `_khalf_rules(triple)` -> OLD / PARENT / POOLED verdicts `(route, clause, why)` plus median and
  pooled IQRs, `_khalf_short` (route / miss(clause)).
- `_khalf_prepare` gate: 10 read-flush `_cell_gate` rounds; routing = POOLED on rounds 1-3 only (every ratio >= 1.01,
  median >= 1.03, pooled-123-rep cand IQR <= 2.0x max(pooled master IQR, 0.5 us)). A failure in rounds 1-3 raises as
  before (no route); a failure in a diagnostic round 4-10 only truncates the diagnostic (noted in K-DIAG). Profiler
  assertion, candidate, canary, `_cell_check`, A/B copy check and the 170 s start deadline unchanged.
- PATH CHOSEN K line (still a `_decision`, so in the DECISIONS re-print): rounds 1-3 per-round detail (ratio, net,
  med, per-round IQRs, base; rounds 4-10 no longer listed there), then `median`, `pooled IQR master/cand ... (123 reps
  each)`, `decided by POOLED: route|miss(clause)` and what OLD / PARENT would have said on rounds 1-3; on a miss the
  reason names the failing clause with numbers.
- New `K-DIAG pooler m=128` `_decision` line (also re-printed): all 10 ratios, per-round 1.5x IQR-fail count /10, ratio
  < 1.03 count /10, then OLD / PARENT / POOLED verdicts for triples 1-3, 4-6, 7-9. Wrapped so it can never raise.
- Module docstring and section (11) comment updated (ten rounds, pooled rule on 1-3).
- Offline sanity check (no GPU): `_khalf_rules` / `_khalf_short` exercised on synthetic rounds via AST extraction;
  route, floor miss, median miss and pooled-IQR miss cases all came out as intended.

Deviations from the `change` paragraph, all per the orchestrator's binding decisions: pooled IQR bar is 2.0x
(`KHALF_POOLED_IQR_MAX`), not 1.5x; 10 rounds run (decision still on rounds 1-3 only) for the folded-in diagnostic of
the retired control. `expected` corrected: router rows m=16/32/128/256 only (m=1/8 take the untouched cluster path),
with the judged a20f8048 gain (~0.6-1.0 us summed) instead of the gate's per-row figure. Judgement call: the m=32 router
row is predicted `flat` (judged flat in a20f8048 despite the gate); flip it to `faster` if the orchestrator prefers the
gate's reading. Cost: ~8 more gate rounds, roughly +0.7-1 s of prepare, well inside the khalf deadline.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate diff` (this change) and the
staged parent patch it sits on.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints added to kernel_fn; the diff touches
      only prepare()-time gate code (`_cell_gate` stats, `_khalf_*` helpers, `_khalf_prepare`); kernel_fn's khalf branch
      still dispatches on cfg m and the state flag only.
- [x] Nothing derived from `data` is cached across calls: the stored raw times are prepare()-time gate timings on
      synthetic x, kept in a local stats dict, never in state.
- [x] Weights: no change to repacking; pooler stays BF16, router hi+lo split unchanged (parent's code).
- [x] Activations BF16 where SGLang keeps BF16: n/a to this diff (no numerics touched); khalf's fp32 partials + one
      bf16 rounding unchanged.
- [x] Attention sees every key: n/a: dense projections problem, no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the gate
      only times its own CUDA graphs with torch events in prepare().
- [x] All work happens inside the timed call; no output caching or replay: routing flags only; outputs computed per call.
- [x] No triton.autotune added; first line `# okc: problem=muse_bf16_proj` kept.
- [x] `python3 -I -m py_compile` passes; __pycache__ removed from the worktree.

okc_lint.py output (rc 0):

```
warn  host-sync    L555: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L583: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L594: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L604: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L608: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L651: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L660: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L668: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L751: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1846: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1892: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1901: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1907: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1911: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1940: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1961: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2006: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2406: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3029: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3032: code = int(err.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3066: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3070: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3099: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L3113: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3153: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3162: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3165: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3231: code = int(state["bulk_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3262: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3275: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3300: state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L3311: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L3316: algo_id, tile, stg, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3322: key = tuple(int(v) for v in algos[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3327: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L3374: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3856: lf = frag(lo_enc).to(torch.uint8).reshape(ctas, nch, ns, mt, 256)  (allowed only where SGLang does the same)
warn  dtype        L3858: cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).to(torch.uint8).reshape(ctas, nch, ns, mt, 128)  (allowed only where SGLang does the same)
warn  dtype        L3863: blocks = torch.zeros((ctas, B12_ESC_BYTES), dtype=torch.uint8)  (allowed only where SGLang does the same)
warn  host-sync    L3864: per_cta = torch.bincount((idx[:, 0] // r).cpu(), minlength=ctas) if n_esc else torch.zeros(ctas, dtype=torch.int64)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3869: rr, kk = idx[:, 0].cpu(), idx[:, 1].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3870: vals = bits[idx[:, 0], idx[:, 1]].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3873: for c in torch.nonzero(per_cta).flatten().tolist():  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3914: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4026: code = int(state["b12_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4081: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4224: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4228: nbad = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4323: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4378: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4617: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4621: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4634: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4674: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate/muse_bf16_proj/submission.py
```
