# muse_prefill_layer · 2026-10-29T235116Z · check · experts-rega-low-2cta-a5b

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-29T233056Z-experts-rega-low-2cta-a5b.md` (branch `hyp/muse_prefill_layer/experts-rega-low-2cta-a5b`).

- **Job:** `074a0e02f4e94efd` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `6fad5b2c`

## Goal
Add a low-class-only arm to master's rega dq module: the H100-validated SC-2CTA tile from takemyjob hopper/rgx4 290f097 (research/h100-rega4-1029.md): master's A5b+D+SC k loop in a 2-CTAs-per-SM tile (grid 264), launch registers 80 with setmaxnreg 32 producer / 104 consumer, 3-stage ring, 64-row slab, chunks capped at N64 (at low, 28 of 32 chunks are N16/N32). Bit-exact vs master. Route it ONLY at the low class through master's existing per-class rega race: pick it if bit-exact on every draw (torch.equal, replay) and <= 0.97 vs the routed A5b+D+SC path in every round (both L2 modes, pop and random routings); high and ft keep A5b+D+SC. Build it as extra instantiations in the existing background dq build, keeping the 233 s build gate (retire nothing else; count instantiations). Print it in the [rega-dq] summary (RATIOS/PAIR/ROUTE for the new arm at low) inside the 16 KB tail.

## What changed, and where
**Mechanism:** the rega dq module gains arm S2 (id 7, SC-2CTA): A5b+D+SC's k loop and master's epilogues in `rega_kernel<G, false, 5, 1>`. It runs 2 CTAs/SM at 80 regs, with setmaxnreg 32/104, a 3-stage ring, a 64-row slab, chunks of at most 4 blocks and a grid of 2 x SMs. It is raced and routed at the LOW class (turn_1 rows) only, against the routed A5b+D+SC. High and ft keep A5b.

Source: takemyjob hopper/rgx4 4dc233f arm 2 (`RGX_STAGES 3, RGX_SLABN 64, RGX_MINB 2, RGX_CHB 4, RGX_PREG 32, RGX_CREG 104`). Base: master's dq module (blob 7569df17). Only `<worktree>/muse_prefill_layer/submission.py` changed.

- **CUDA (dq module only, built from master's strings by asserted edits; master's own modules are untouched):**
  - `_REGA_DQ_HELPERS` gains `Geo<S2>` / `SmemS<G, S2>`. `Geo<0>` is master's geometry exactly, with static_asserts that SmemS<.,0> == Smem<.>; S2 = 0 code compiles to master's constants. `Geo<1>` is rgx4 arm 2's geometry. Static asserts check the setmaxnreg pool (32x128 + 104x256 == 80x384), the register file, the smem (87040 / 95232 B, rgx4's numbers; 2 CTAs + static + reserve <= 228 KB) and the chunk table (`S2_MAXBLK = 4 (MAXC - MAXE)` = 1920; T = 1024 worst case 542 blocks -> <= 167 chunks of 512).
  - `dq_batch_d` takes the slab-half size as a template argument (default SLAB_HALF).
  - `consume_item<NB, G, DQ, S2>`: the MODE 5 / MODE 2 / A5b+D+SC loops read `Geo<S2>` (ring depth, stage bytes, slab half). With S2 = 1, the MODE 2/5 sink uses the stage area, because S2's 64-row staging tile is smaller than 128 x 128 B. A static_assert allows S2 only with DQ 5 and NB <= 64. The epilogue is master's dq-module epilogue unchanged (fast-silu w13, w2st arm a early gs/sid loads and the pad-column route ids). The non-local zero store is `_experts_rega`'s `y` zeros, which is the same dispatch.
  - `rega_kernel<G, BIG, DQ, S2>`: `__launch_bounds__(THREADS, S2 ? 2 : 1)`. Barriers, ep, ring, TMA second half, chunk cut (`& (CHB-1)`, `cn < CHB`) and the setmaxnreg immediates (`"n"(GK::PREG/CREG)`) come from `Geo<S2>`. With S2 = 1 the N128 bucket is never instantiated; reaching it sets error word 3 (bounded producer timeout).
  - `info_one<.., S2>` sets S2's own dynamic smem attribute in `rega_init`, before any launch, and reports flag arm id 7.
  - Host `rega_gemm`: accepts dq 7, checks `grid > 0`, and launches `rega_kernel<G, false, 5, 1>` at **2 x the SM-count grid** with `SmemS<G,1>::BYTES`. `TORCH_CHECK(dq != 7 || (!big && blocks_len <= S2_MAXBLK))`.
  - Instantiations: 12 (unchanged) + S2 w13/w2 = **14**. The ptxas probe (`REGA_DQ_PTXAS_PROBE`) instantiates **only S2's 2** (`REGA_DQ_PROBE_INST` 4 -> 2), so est = 40 x 16/4 = 160 s and need = 233 s are unchanged. `REGA_DQ_EST_4INST_S` 40 and LIMIT 270 are kept. The probe parser matches `Li5ELi1E`.
- **Python:**
  - Hang safety (`_rega_dq_attrs`): an S2 row is good only if `numRegs == 80`, `numRegs x 384 == 30720` (the pool), `localSizeBytes == 0`, CTAs/SM == 2, smem <= limit, sm_90 image and not BIG. Otherwise `arm_off[7]` is set and S2 is never launched.
  - The phase races S2 only at `cls == "low"`, only with A5b raced (A5b's bank, stored as `banks[7]`, the same tensors, no copy) and only after the attrs pass. It prints `raced [...]; rega_low_variant v`. The low class is still skipped entirely when `rega_low_variant != 0` (master's rule; v is printed in the SUMMARY head and the ROUTE low line).
  - `_rega_dq_class` (low with S2): calibrates master's guard `_rega_pop_pick(_rega_pop_counts(753), REGA_LO_GUARD_ROUTES=1350)`. That routing is an extra exactness routing, which also prints the chunk histogram N16/32/64/128 at cap 8 -> chunks at cap 4, and an extra timed routing.
  - `_rega_dq_s2_gate` (in `_rega_dq_route_pick`): S2 passes only if all of these hold:
    - A5b passes its own gate.
    - hs and y are torch.equal to master's dispatched fsilu w13 + w2st w2 on every exactness routing (random, skewed, one, pop, guard).
    - The pair MODE 0 is <= 0.97 x A5b AND x master AND x DQ0 in every round and L2 mode on pop and random.
    - The guard pair is <= 1.00 x A5b in every round and L2 mode.

    Route order is S2 > A5b > A3. A route still requires master's two-state replay check with S2 ON (`_rega_dq_route`, unchanged). `_experts_rega` accepts `rega_dq_ok['low'] == 7` (bank key 7).
  - Prints:
    - attrs (in-phase: all 14; summary: `attrs 14 inst: 12 DQ0/A3/A5b all ok r168 l0 o1; S2 r80/80 l0/0 o2/2 ok`), the `ptxas probe S2` line, and `need 233s (14+2 inst)` in SUMMARY.
    - RATIOS low gains S2. PAIR low shows S2 vs DQ0/m/A5b ('b') for pop warm/cold and random, plus the guard (routes, histogram, warm/cold).
    - ROUTE low carries S2's gate terms, exactness `eqN/N`, `ctab<=167/512 at T=1024` and `rega_low_variant`.
    - To fit within the 3000-char cap, the high/ft RATIOS lines are reduced to MODE 0 warm and the high/ft VERDICT lines to their head. `_rega_dq_cap` trims non-low lines first. A synthetic-data run of the summary code measured 2888 chars, uncapped.
- **Review fixes applied:** all four MUSTs (build gate 14+2, guard as a must-pass <= 1.00 round, the print set, rega_low_variant). The review's open question on hard-pinning: S2 is a separate id raced and routable only at low, so a noisy high/ft race cannot route it.
- **Deviations:**
  - The spec's "grid 264" is implemented as 2 x the SM count, as the review asks.
  - S2 also requires A5b to pass its own gate (the "routed A5b+D+SC" reference).
  - The canary's D-descriptor check still covers SLAB_HALF only. S2's 8192-B half uses the same no-carry argument (slab + 2 x 8192 < 2^18), and the torch.equal exactness covers it.
- **Not validated on a GPU or nvcc here** (no toolchain locally; nodejudge has no prefill problem). Biggest risks:
  - (a) nvcc compile of the template edits. The dq module turns OFF on a build error, and master runs.
  - (b) S2 attrs != r80/o2: S2 is then OFF and never launched.
  - (c) The guard round on the judge: the node measured 0.946-0.957.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-low-2cta-a5b diff` (1 file, +336/-88).

- [x] CUDA-graph capturable: kernel_fn's path changes only `_experts_rega`'s `dq in (3, 5, 7)` bank lookup (a prepare-time int). The S2 launch is the same host call (grid doubled from an int in C++, no sync, no allocation). The new `.cpu()` / bincount (`_rega_dq_hist`) and `_rega_pop_counts` are prepare-only (lint: outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls: state gains only prepare-time ints and the existing A5b bank copy under key 7 (made from the weights in prepare).
- [x] Weights: repacked losslessly in `prepare()` only. S2 reads A5b's ENC4 + packed-scale copy, which is unchanged and checked by its inverse per chunk. NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: same bf16 hs / y and fp32 accumulation. The k order per output element is unchanged, so S2 is bit-exact by construction and gated on torch.equal.
- [x] n/a (attention untouched): the diff touches only the rega dq module and the experts dispatch.
- [x] No timing patches, harness I/O, module edits, reference import, subprocesses or network. The only build is the existing load_inline background thread; no new process, fork or spawn.
- [x] All work happens inside the timed call: S2 launches in `_experts_rega` per call, with no output caching or replay.
- [x] `# okc: problem=muse_prefill_layer` first-line marker kept. No triton.autotune. sm_90a PTX (setmaxnreg / wgmma) stays under `__CUDA_ARCH_FEAT_SM90_ALL`, with master's error-word-2 `#else` and arch 9.0a set and restored by master's code.
- [x] Fails safe: build error, attr miss, exactness miss, gate miss, replay failure or deadline all leave low on A5b / master (`PATH`/ROUTE lines printed). Every mbarrier wait remains bounded.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <submission> --problem muse_prefill_layer`):
```
OK with warnings: .worktrees/muse_prefill_layer/experts-rega-low-2cta-a5b/muse_prefill_layer/submission.py
(warnings only: host-sync / dtype in prepare-only code outside kernel_fn's call graph, as on master; 0 errors)
```
`python3 -I -m py_compile`: ok (`__pycache__` removed).

## Risks / what could fail
medium
Node->judge transfer for this tile family held within 0.01-0.02 (A3: node low 0.84-0.89 vs judge 0.854; A5b: node HBM-like low .961-.972 vs judge .943). Low class = turn_1 rows (sliding share 21.7%, full 11.7%), rega 48.6% of turn_1 sliding.

## Result (after the run)
- job `074a0e02f4e94efd` (check), status done, passed True, tests 5/5
- speedup 1.5678x, 95% CI [1.5442, 1.5913], score 247105.74348961783 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.5721154841318719, "ci": [1.5485337518698938, 1.5956972163938499], "job": "9f93484606cd4f9e"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-rega-a5b-desc-splitcommit.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-low-2cta-a5b.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-low-2cta-a5b.patch`; branch `hyp/muse_prefill_layer/experts-rega-low-2cta-a5b`
