# muse_prefill_layer · 2026-10-27T141037Z · check · experts-rega-w2-early-loads-low-class

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-27T140120Z-experts-rega-w2-early-loads-low-class.md` (branch `hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class`).

- **Job:** `59eb665f162a4ac9` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `52169f17`

## Goal
Restack of experts-rega-w2-epilogue-prefetch-bulk-store (neutral, check 665c32dd; patch in muse_prefill_layer/hypotheses/done/neutral/) on current master, with three changes to its gate and dispatch only. (1) Keep arm (a) (early gsc / sorted_ids loads, the sid table in the staging tile's pad column, master's 16-B scatter loop reading it) and DROP arm (b) (the cp.async.bulk row stores): b was slower than master in every class and mode (x1.001-1.046), consistent with TMA contention against the producer. Removing b also removes its PTX from the fsilu module (fewer instantiations, less build time). (2) The LOW class (T=753) becomes routable: in 665c32dd arm a measured x0.975/0.978 cold and x0.975/0.981 warm on the judge-like pop:1 routing and x0.976/0.975 cold, x0.971/0.972 warm on random (w2 164.8 -> 161.0 us), all inside the <= 0.99 bar, exactness torch.equal on all four routings; the review had called low 'HBM-bound, flat' and made it print-only, which the measurement contradicts. Gate low exactly as the parent gated high / ft (arm/master <= 0.99 every round, cold+warm, pop+random, then the two-state replay check), with rega_low_variant == 0 precedence kept. (3) High and ft: keep gating arm a (it measured x0.991-0.998, a near miss); route only if it clears the same bar. Everything else (fsilu module host, exactness checks, deadlines, compact PATH CHOSEN lines last) as in the parent.

## What changed, and where
Implemented 2026-10-27T140918Z in the worktree's `muse_prefill_layer/submission.py` on top of the restacked parent
(experts-rega-w2-epilogue-prefetch-bulk-store, staged). Diff vs master 564d4453: 1 file, +420 / -22 (vs the staged
parent: +163 / -258). Not committed, not run.

- **Arm b removed entirely.** `REGA_W2ST_EDITS` keeps only arm a's three kernel edits (early `gsc_w2` / `sid_r` loads
  before the k loop under `if constexpr (!G)`; `gsc` from `gsc_w2` on w2 plus the sid write into the staging tile's
  pad column after the first epilogue barrier; master's 16-B scatter loop reading the sid from that table). Dropped:
  the `SB` template parameter on `consume_item` / `rega_kernel` / `info_one`, the `fence.proxy.async`,
  `cp.async.bulk` store + `commit_group`, both `wait_group` asm, the two SB=true `info_one` rows, the which<=2
  `TORCH_CHECK` and the 3-arg `WG_LAUNCH` / which=2 launch. No template switch is needed any more: the fsilu module's
  own w2 (rega_gemm which 1) IS arm a, so the module is back to master's four `rega_kernel` instantiations
  (w13 s1/ft + w2a s1/ft) and arm a's kernel name equals master's (only PATH CHOSEN attributes it, as the review noted).
- `REGA_FSILU_EXT_NAME` -> `muse_prefill_rega_fsilu_w2a` (new name: never served from the parent's two-arm build).
- `_rega_fsilu_attrs`: back to 4 rows (`seen != 4`); fsilu's ON reads the w13 rows only (as the parent), the w2
  rows (`w2a`, also local 0) return `{big: ok}` for the w2st gate.
- `_experts_rega`: `elif w2st: fsilu_ext.rega_gemm(1, ...)`; `rega_w2st_which` removed from state (the per-class
  `rega_w2st_ok` bool is the only switch). `not variant` precedence unchanged: rega_lo's V64/V32 always win at low.
- **Gate (`_rega_w2st_class`)**: every class routable (`REGA_W2ST_ROUTABLE` removed). Exactness unchanged in
  substance (sentinel-filled y, torch.equal master's on random, skewed, one and fsilu's judge-like pop routing, error
  word 0 + w2 marker for both launches). Timing master vs a in the same 2 rounds x 11 alternations, cold + warm, on
  pop and random; pass = a/master <= 0.99 in all 8 ratios. **Low only (review note 1):** when that passes, rega-lo's
  guard routing is re-derived with the same helpers and seed (`_rega_pop_counts(753)` + `_rega_pop_pick(...,
  REGA_LO_GUARD_ROUTES=1350)`, i.e. the alpha ~2.2 / ~1356 routes rega-lo uses) and timed the same way; it must be
  <= `REGA_W2ST_GUARD_RATIO` 1.00 in all 4 ratios (cold + warm, both rounds). Then the two-state replay check with arm
  a ON (`REGA_FSILU_REPLAY_PLAN[cls]`, keys `REGA_FSILU_STATE_KEYS + rega_w2st_ok`).
- **Print (review note 2)**: the w2st phase's first line prints fsilu's build seconds and the w2a attr rows. Each
  `[rega-w2st] PATH CHOSEN <cls>:` line carries `which 0/1 ok True/False`, `rega_low_variant v` (low) and
  `fsilu=ON/OFF`; the final `PATH CHOSEN: rega_w2st_ok {...}` line is the last prepare() line. The low GATE line
  shows pop+random gm/max and the guard max with pass/MISS.
- Deadlines: unchanged (`REGA_W2ST_STEP_DEADLINE_S` = 265 s; every step, including the guard timing, goes through
  `late()`).

Deviations / judgement calls:
- **Class order changed to `("low", "high", "ft")`** (fsilu's order) from the parent's `("high", "ft", "low")`: the
  parent put low last only because it was print-only; low now carries the expected gain, so it is gated first and
  cannot be starved by the deadline after the costlier ft class.
- The guard routing is timed only when the pop+random gate already passed (saves ~one timing call per class where it
  cannot change the verdict); it is not added to the exactness list (spec item 5 lists random/skewed/one/pop).
- The guard alpha is recomputed in w2st (cheap: 3 draws x 21 alphas of routing generation, no kernel timing) rather
  than plumbed out of `_rega_lo_run`, to keep the change inside the w2st area; same seed 20261028 and helpers, so the
  same alpha as rega-lo's guard line.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-w2-early-loads-low-class diff HEAD`:

- [x] CUDA-graph capturable: the only kernel_fn-path change (`_experts_rega`) is a dict lookup on prepare-time
      state and the class from `num_tokens` (shape), then one launch with the same arguments as master's; every
      `.cpu()` / `synchronize()` / `torch.equal` added is in prepare()-only w2st / fsilu helpers.
- [x] Nothing derived from `data` is cached: state gains only `rega_w2st_ok` (plain bools per class) and the module.
- [x] Weights: untouched (same `bank["q2"]` / `gs2` NVFP4 bank as master's w2).
- [x] Activations: bf16 y with master's values, addresses and rounding (bit-exactness asserted in the gate); KV cache
      / query precision untouched.
- [x] Attention: n/a: no attention code touched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the
      nvcc build is the existing fsilu `load_inline` path; no new imports.
- [x] All work inside the timed call; no output caching or replay: arm a recomputes y every call; the prepare-time
      gate only picks which module's w2 launches.
- [x] First line `# okc: problem=muse_prefill_layer` kept; no triton.autotune added.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py
--problem muse_prefill_layer`), exit 0:

```
107 x warn host-sync   (all "outside kernel_fn's call graph; fine in prepare()")
  5 x warn dtype       L3470, L3471, L3761, L4308, L6794 torch.uint8 buffers (pre-existing master lines)
OK with warnings: muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` OK; `__pycache__` removed.

## Risks / what could fail
low
Headline ~0.1-0.2%: below single-run readability; worth a check only because the parent measured it directly (x0.971-0.981 in all 8 low-class rounds) and it removes the slower arm b from the module. Read it on the [rega-w2st] low GATE / PATH CHOSEN lines and the turn_1 w2 profile row.

## Result (after the run)
- job `59eb665f162a4ac9` (check), status done, passed True, tests 5/5
- speedup 1.4986x, 95% CI [1.4761, 1.5211], score 258507.82036541268 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.488912488847924, "ci": [1.466578801515205, 1.5112461761806426], "job": "ec3bd01859574bf1"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-rega-w13-fast-silu-epilogue.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class.patch`; branch `hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class`
