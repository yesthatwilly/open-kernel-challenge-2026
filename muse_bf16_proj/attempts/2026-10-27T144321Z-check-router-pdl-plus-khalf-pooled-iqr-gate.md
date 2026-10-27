# muse_bf16_proj · 2026-10-27T144321Z · check · router-pdl-plus-khalf-pooled-iqr-gate

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-27T143409Z-router-pdl-plus-khalf-pooled-iqr-gate.md` (branch `hyp/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate`).

- **Job:** `023f4260e86e423e` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `76bb1188`

## Goal
Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all 4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean) on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128, whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6 and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held) showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >= 1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)). To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs, the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
low
Headline effect ~+0.3-0.6% combined (router ~0.3% + khalf ~0.27% when it would otherwise miss): near the single-run readability line, so read the R-PDL and K PATH CHOSEN lines. The 1.2735 board run had khalf OFF, so a run with both ON can plausibly beat it; a check that beats the board gets a scored run.

## Result (after the run)
- job `023f4260e86e423e` (check), status done, passed True, tests 9/9
- speedup 1.2723x, 95% CI [1.2532, 1.2914], score 3797.001348390161 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2739626767393788, "ci": [1.254853236588288, 1.2930721168904693], "job": "254c83ae55834807"}; base result: SGLang (no base result)
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-pdl-plus-khalf-pooled-iqr-gate.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-pdl-plus-khalf-pooled-iqr-gate.patch`; branch `hyp/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate`
