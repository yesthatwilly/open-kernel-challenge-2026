---
problem: muse_prefill_layer
slug: experts-rega-low-2cta-a5b
created: 2026-10-29T233056Z
base: master
mint_blob: "7569df17e62ecfb6ecf85d12e2033c6611fcaade"
best_at_mint: 1.5721
area: "experts rega tile at the LOW class (turn_1): occupancy / tile shape"
change: >
  Add a low-class-only arm to master's rega dq module: the H100-validated SC-2CTA tile from
  takemyjob hopper/rgx4 290f097 (research/h100-rega4-1029.md): master's A5b+D+SC k loop in a
  2-CTAs-per-SM tile (grid 264), launch registers 80 with setmaxnreg 32 producer / 104 consumer,
  3-stage ring, 64-row slab, chunks capped at N64 (at low, 28 of 32 chunks are N16/N32). Bit-exact
  vs master. Route it ONLY at the low class through master's existing per-class rega race: pick it
  if bit-exact on every draw (torch.equal, replay) and <= 0.97 vs the routed A5b+D+SC path in
  every round (both L2 modes, pop and random routings); high and ft keep A5b+D+SC. Build it as
  extra instantiations in the existing background dq build, keeping the 233 s build gate (retire
  nothing else; count instantiations). Print it in the [rega-dq] summary (RATIOS/PAIR/ROUTE for
  the new arm at low) inside the 16 KB tail.
expected:
  - shapes: load=turn_1*
    direction: faster
    why: "node pair vs A5b+D+SC at low: L2 0.896, HBM 0.938, random 0.887; ncu eligible warps 0.73 -> 1.41, SM cycles -13%"
  - shapes: load=turn_4*
    direction: same
    why: high keeps A5b+D+SC (2-CTA loses x1.13 there)
  - shapes: load=first_turn*
    direction: same
    why: ft keeps A5b+D+SC (x1.23)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-a5b-desc-splitcommit: same k loop (A5b+D+SC, landed); this changes only the low-class tile occupancy (2 CTAs/SM, smaller slab/ring) to double eligible warps, which the earlier K-pair four-consumer attempt could not at 232 regs"
target: check
value_gain: 1.6
value_p: 0.55
value_ev: 0.88
value_cost: 350k tokens, 1 judge run
evidence: >
  H100 rgx4 (takemyjob hopper/rgx4 290f097; research/h100-rega4-1029.md, GPU 5): SC-2CTA pair vs
  A5b+D+SC at low L2 0.896, HBM 0.938, random 0.887 (high 1.129, ft 1.23); bit-exact, no spill, 2
  CTAs/SM at 80 regs; ncu low w13 eligible warps/scheduler 0.73 -> 1.41, issue active 52 -> 63%,
  SM cycles -13%; prefill board 1.5721 (9f934846)
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T233429Z
review_companions: null
review_best: 1.5721154841318719
review_blob: 7569df17e62ecfb6ecf85d12e2033c6611fcaade
stage: implemented
claimed_at: 2026-10-29T233706Z
base_sha: 6364d94aa7863fd527376ac09288d0238158f8d7
base_blob: 7569df17e62ecfb6ecf85d12e2033c6611fcaade
branch: hyp/muse_prefill_layer/experts-rega-low-2cta-a5b
worktree: .worktrees/muse_prefill_layer/experts-rega-low-2cta-a5b
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T235116Z
impl_commit: 6fad5b2c9aefca8da37896521c7a7b48cc6409ae
submission_sha256: 0ef9947dcde6f19ee8b59661cd7c0f2578ba96d2528c1cdf31a56a38f3c84258
norm_sha256: 1ff9e0b91583cd42208107e250252bab9eab8765819dcac8b8071adf7d89f0ea
submission_blob: 0112a434215de60e5629552bc7b8ccaff363f437
lint_warnings: 125
attempt_file: 2026-10-29T235116Z-check-experts-rega-low-2cta-a5b.md
prepared_at: 2026-10-29T235116Z
---

# experts-rega-low-2cta-a5b

## Change

Add a low-class-only arm to master's rega dq module: the H100-validated SC-2CTA tile from takemyjob hopper/rgx4 290f097 (research/h100-rega4-1029.md): master's A5b+D+SC k loop in a 2-CTAs-per-SM tile (grid 264), launch registers 80 with setmaxnreg 32 producer / 104 consumer, 3-stage ring, 64-row slab, chunks capped at N64 (at low, 28 of 32 chunks are N16/N32). Bit-exact vs master. Route it ONLY at the low class through master's existing per-class rega race: pick it if bit-exact on every draw (torch.equal, replay) and <= 0.97 vs the routed A5b+D+SC path in every round (both L2 modes, pop and random routings); high and ft keep A5b+D+SC. Build it as extra instantiations in the existing background dq build, keeping the 233 s build gate (retire nothing else; count instantiations). Print it in the [rega-dq] summary (RATIOS/PAIR/ROUTE for the new arm at low) inside the 16 KB tail.

## Expected

- `load=turn_1*` faster: node pair vs A5b+D+SC at low: L2 0.896, HBM 0.938, random 0.887; ncu eligible warps 0.73 -> 1.41, SM cycles -13%
- `load=turn_4*` same: high keeps A5b+D+SC (2-CTA loses x1.13 there)
- `load=first_turn*` same: ft keeps A5b+D+SC (x1.23)

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 rgx4 (takemyjob hopper/rgx4 290f097; research/h100-rega4-1029.md, GPU 5): SC-2CTA pair vs A5b+D+SC at low L2 0.896, HBM 0.938, random 0.887 (high 1.129, ft 1.23); bit-exact, no spill, 2 CTAs/SM at 80 regs; ncu low w13 eligible warps/scheduler 0.73 -> 1.41, issue active 52 -> 63%, SM cycles -13%; prefill board 1.5721 (9f934846)
- **Gain if it works:** 1.6% headline; **p** 0.55; **EV** 0.88% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.6% < 2.5%; EV 0.88% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge run.

## Risk

Node->judge transfer for this tile family held within 0.01-0.02 (A3: node low 0.84-0.89 vs judge 0.854; A5b: node HBM-like low .961-.972 vs judge .943). Low class = turn_1 rows (sliding share 21.7%, full 11.7%), rega 48.6% of turn_1 sliding.

## Adversarial review

Reviewed 2026-10-29T233429Z by an adversarial subagent (model opus).

**Verdict: proceed.** E: fresh (board 1.5721, blob 7569df17 unchanged). CLAIMS.md:55's reopen ('a different low-class kernel, not V64/V32') is met only in form. V64 died on the judge's skewed ~1350-route turn_1 load (guard 1.030-1.037, pinned 1.4776 vs 1.4897), and this arm keeps V64's N64 chunk cap. Proceed only if the gate adds master's REGA_LO guard routing and the build gate stays at need 233 s, which works if the ptxas probe compiles the new arm in place of A5b's.

### Justification
A: The mechanism is plausible and was measured on the same SM type. At rgx4 (290f097) low l2, 2 CTAs/SM doubles eligible warps per scheduler (0.73 -> 1.41), raises issue-active from 52% to 63% and cuts SM cycles 13%. The pair falls 6-11% vs A5b+D+SC: l2 0.896, hbm 0.938, random 0.887. The run was bit-exact with 0 spill, and the chb4s3 control isolates the +3-12% cost of the cap and ring. This fits the [lead] in CLAIMS.md:71 (super-serial stages, TLP levers reopen). Value is about 1.0-1.6% headline: turn_1 carries ~33% of the weight, rega is ~49% of turn_1 sliding, and the cut is ~8%. That is below the bar, but the user asked for it. WEAKEST LINK: the routing load. The node evidence and master's dq race both run at the low class's 'pop:1 711r' (judge log: 'RATIOS low pop:1 711r'), about the random load. Master's own constant REGA_LO_GUARD_ROUTES = 1350 says the judge's turn_1 has ~1314-1359 local routes, about 1.8x that. Under the N64 cap, every expert above 64 routes has its N128 chunk dequantized twice. That is exactly why SC-2CTA loses 13-24% at high (44 of 54 chunks are N128), and it is the same tile family as V64. V64 won on node (occ2 0.887) but lost on the guard routing (1.030-1.037) and lost when pinned on the judge (CLAIMS.md:55). The hypothesis's 0.01-0.02 transfer figure comes from A3/A5b, which change instruction cost. That figure says nothing about an occupancy and chunk-shape change whose cost scales with route skew. PRIOR: the prior field names A5b and K-pair but leaves out the direct ancestors: experts-rega-low-class-two-ctas-per-sm (FINDINGS:226, neutral 1.4173), combo-b-normgather-two-cta-low-regate (CLAIMS.md:46, 'never routes on the judge'), and lo-pin-v64-fsilu (CLAIMS.md:55, refuted). The difference is real: an A5b+D+SC k loop, setmaxnreg 32/104 and a 3-stage ring, against V64's older dequant. So this is not a hold, but the prior is incomplete. BUILD GATE: need = used + 1.3 x 40 x (INST + PROBE_INST)/4 + 25 <= 270. Today 12 + 4 gives need 233.1 (the judge log shows 'from 0.1s (est 160s)'; the actual build took 46.9 s). The new arm needs only BIG=false x G{w13,w2} = 2 instantiations. 14 + 4 gives need 259 (passes only while used <= 11 s). 14 + 6, with the probe also covering the new arm, gives 285 and fails. A new template parameter crossed with DQ 0/3/5 or BIG also gives 285+ and fails. The fix that keeps 233 s: point the 4-inst probe at the new arm's 2 instantiations plus nothing else. A5b's probe is already proven clean on the judge (regs 168, C7510 0, C7512 0), so INST 14 + PROBE 2 = 16 gives need 233. LOW ROUTE = turn_1 exactly: the TASK shapes are T 753 / 3012 / 16384, and low is 200 < T <= REGA_LOW_MAX 1024, so only turn_1 rows (sliding and full) route. The unmeasured 753-3012 crossover does not matter for the judge. EXACTNESS: in swap-AB the 64-row slab and N64 cap split the route (N) side only. Each output element keeps master's k order (same BK, records, A5b dequant, split commit), and wgmma's per-element fp32 accumulation does not depend on N. rgx4 showed bit-exact on all 70 EXACT lines including random. It must also match master's epilogue (fast-silu w13, w2st arm a early gs/sid loads, non-local zero store), and the gate must compare hs and y against master's routed dispatch, not just DQ0.

### Confounders on the current kernel
- experts-rega-a5b-desc-splitcommit (landed, the reference): the new arm reuses its k loop and the A4b/A5b repacked bank. The race must compare against the routed A5b MODE 0 (rega_dq_ok low = 5), not DQ0 or A3, or a 'win' is just A5b's own gain counted again.
- experts-rega-w13-fast-silu-epilogue + w2 early gs/sid loads (arm a, CLAIMS.md:54): these are master's epilogues at low. If the 2-CTA arm drops either, exactness vs master fails, or a timing difference gets wrongly credited to occupancy. Port both, with the 104-register consumer budget checked (fast silu adds live regs in the w13 epilogue).
- align-two-phase-all-t-rega + norm-gather (combo-a): these produce the block/chunk tables and the gathered slab. The '% 4' chunking roughly doubles the chunk count at low. The chunk-table capacity (err code 3 'block or chunk table overflow') must be checked at worst-case T=1024 (8192 routes + 32x15 padding), not just at the drawn routing.
- combo-b rega-lo module (REGA_LO_SKIP): _experts_rega has 'dq = 0 if variant else ...'. With the LO phase skipped, variant is 0, so the new dq code reaches dispatch. If anything re-enables LO, the new arm is silently bypassed. Print the variant in the ROUTE line.

### Interference and A/B needs
No companion run is needed. The per-class in-prepare race against the routed A5b+D+SC (bit-exact on hs/y, replay, both L2 modes, pop and random) makes one check readable: either ROUTE low says the new arm, and the turn_1 rows carry the change, or it says A5b, and the check is master. High and ft cannot move, and turn_4/first_turn act as in-run controls. The confound that would make a gain or loss ambiguous is the routing load (711 vs ~1350 local routes). That belongs in the gate (a guard round), not in a second judge run. A revert of fast-silu/w2st or A5b tells us nothing here: they define the baseline the arm must match bit-exactly. Optional and outside this check: before spending the check, one H100 rgx4 rerun of arms 1 vs 2 at T=753 with the guard routing (_rega_pop_pick to ~1350 routes) would settle the V64 failure mode for free. That needs the user's go (Hopper lane closed 10-29 14:54Z).

### Open questions
- MUST (gate): add master's REGA_LO guard routing (_rega_pop_pick(counts, REGA_LO_GUARD_ROUTES=1350) at T=753) as a required round for the new arm, at <= 0.97 vs routed A5b (or at least <= 1.00, REGA_LO_GUARD_RATIO). Print its PAIR, chunk histogram (N16/32/64/128 counts) and local-route count. Without it the gate repeats V64's 711-route blind spot.
- MUST (build gate): exactly 2 new instantiations (G w13/w2, BIG=false). Keep REGA_DQ_EST_4INST_S = 40 and LIMIT 270. Retarget the ptxas probe to the new arm's 2 instantiations (REGA_DQ_PROBE_INST 4 -> 2) so INST 14 + PROBE 2 = 16 -> need 233 s, unchanged. If the probe stays at A5b's 4, need is 259 s and passes only while the build starts before 11 s. Never 14 + 6 (285 s, OFF: budget).
- MUST (prints, inside REGA_DQ_SUMMARY_MAX and the 16 KB tail): [rega-dq] attrs '14 inst ... A5b r168 l0 o1; S2 r80 l0 o2'. Probe line for the new arm: regs, spill, C7510/C7512 (RS2 hit C7512 at 104 regs; SC must show 0). SUMMARY with est/need/build seconds. RATIOS/PAIR low for the new arm vs DQ0, vs m and vs A5b (pop warm/cold, random, guard). ROUTE low naming the chosen arm with every gate term. Exactness counts (torch.equal hs and y vs master's routed path, replay). Chunk-table size vs capacity. rega_low_variant. If the summary exceeds 3000 chars, shorten the high/ft RATIOS lines rather than truncate the low lines.
- Does the rgx4 synthetic bank's chunk histogram at low (pop 16/12/2/2) reflect the judge's turn_1? If the judge's route count is really ~1350, expect several N128 chunks, and the 0.896 will not hold.
- Should the per-class code be a new rega_dq_ok value (e.g. 6) dispatched only for cls == 'low', with high/ft hard-pinned to 5 regardless of the race? That way a noisy high/ft race cannot route a 1.13-1.24x arm.


## Implementation notes

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

## Rules check

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
