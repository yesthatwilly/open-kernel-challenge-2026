# muse_prefill_layer · 2026-10-25T163415Z · submit · combo-b-normgather-two-cta-low-regate

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T161233Z-combo-b-normgather-two-cta-low-regate.md` (branch `hyp/muse_prefill_layer/combo-b-normgather-two-cta-low-regate`).

- **Job:** `73b40cd13fdf4bb0` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `25f55bbc`

## Goal
COMBINATION B (conjectural: re-gate the two-CTA/SM low class on master's own gate routing, stacked on A). Start from the same merge as combination A (/home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py: master 1.4587 + pre-expert-norm-fused-rega-gather bd60e2db) and port the low-class two-CTAs-per-SM kernels from experts-rega-low-class-two-ctas-per-sm, branch hyp/muse_prefill_layer/experts-rega-low-class-two-ctas-per-sm, impl commit dfb755d3 (check 1a77da43b3304b07). PORT, from git show dfb755d3:muse_prefill_layer/submission.py: _REGA_LO_CPP, _REGA_LO_SRC (rega_lo_kernel<G, NMAX, STG, OCC>, rega_init_lo, rega_gemm_lo), REGA_LO_VARIANTS restricted to V64 = (64, 3, OCC 2) and V32 = (32, 4, OCC 2) (drop the V64-occ1 diag to save nvcc time), REGA_LO_REGS_OCC2 80, REGA_LO_POOL, REGA_LO_MAXBLK, REGA_LO_STATIC_BOUND, _rega_pop_case, _rega_pop_alpha, _rega_lo_tiles, _rega_lo_gate, the per-candidate attribute check (numRegs == 80, local 0, 2 CTAs/SM, binaryVersion 90, static <= 12288) and per-candidate self-checks (T=300/753 random/skewed/one, 753 pop, 1024 local-saturated; A/B/C copy-replay with refilled inputs). DO NOT port its module layout: dfb755d3 renamed master's extension to muse_prefill_rega_v3 (master v2 source + LO). Here master's muse_prefill_rega_v2 build, its v1 fallback and its sha checks stay byte-identical, and the LO source compiles as a SEPARATE extension muse_prefill_rega_lo containing only _REGA_HEAD + _rega_wgmma_src() + the namespace part of _REGA_BODY needed by the LO template + _REGA_LO_SRC (the pattern experts-rega-producer-scale-expand-all-t used for muse_prefill_rega_se: master templates never instantiated, only the 4 LO instantiations). Same REGA_CUDA_CFLAGS, TORCH_CUDA_ARCH_LIST=9.0a around load_inline and restored. DISPATCH: _experts_rega (as merged with norm-gather) reads state['rega_low_variant'] (0 master, 1 V64, 2 V32) and state['rega_low_grid'] only when T <= REGA_LOW_MAX and state['rega_ok']['low']; it calls state['rega_lo_ext'].rega_gemm_lo(which, variant, ...) for w13 (on the xs that the fused norm-gather or master's gather produced) and w2, else master's rega_gemm. The align tables come from master's align2 path (bit-identical to _align_parallel's), so the LO decode_item / chunk-cut logic consumes them unchanged; assert that in the self-check by comparing _rega_lo_tiles on align2 vs _align_parallel tables. ROUTING: T in (200,1024] -> V64 or V32 if its gate passes, else master rega; every other T unchanged from A (norm-gather ON where A has it). GATE (the conjectural change): dfb755d3 kept master because it took the WORSE of two routings at <= 0.95 (V64: random 0.887, pop:2 0.994). Master's own low-class rega gate decides on the random routing ('random (soft)'). New rule, per candidate, after master's setup and after norm-gather's setup, on the norm-gather-ON expert block (so the gate sees the same prologue the judge runs): cold-L2 graph medians, >= 9 interleaved alternations, empty-graph baseline subtracted, at T=753 on (a) random routing: ON only at <= 0.95 x master, AND (b) the pop routing calibrated to 1.8x random local routes: guard <= 1.00 x master (never slower under skew). Pick the lower random-routing median of passing candidates. PRINT-ONLY diagnostic: also time V64 and master on pop routings calibrated to 1.2x and 1.5x local routes and print the three ratios as '[rega-lo] SKEW CURVE 1.0x r0 1.2x r1 1.5x r2 1.8x r3', plus routes, touched experts and max chunks per expert at each. Print w13 / w2 kernel-alone us and TB/s per candidate, and '[rega-lo] PATH CHOSEN low: master|V64|V32' as the last prepare() line, beside norm-gather's and fa3c's PATH CHOSEN lines. PREPARE BUDGET: master ~124.8 s, fa3c 1.1 s, norm-gather 0.7 s -> ~125.5 s; LO nvcc + load ~30-35 s (4 instantiations; scale-expand's 4 took 39.0 s) -> ~161 s; attribute checks + self-checks ~1.5 s; pop-alpha calibration and gates (random, pop 1.8x, plus diag 1.2x/1.5x) ~1.5 s -> ~164 s. Own deadlines: REGA_LO_START_DEADLINE_S = 185 s (no LO build after this), REGA_LO_GATE_DEADLINE_S = 250 s (no candidate step, and no switch away from master, after this). REGA_FT_START_DEADLINE_S 150 s and REGA_COPY_DEADLINE_S 255 s do not move; the LO phase starts only after _rega_setup, _fa3c_setup and _norm_gather_setup have returned. A build failure, attribute mismatch, self-check miss or deadline leaves rega_low_variant 0 (master), and A's paths are unaffected. RULES: kernel_fn adds one int lookup and one grid int keyed by the T class; no syncs, prints, try/except or input-sized allocation. setmaxnreg.inc hangs on a short pool, so no LO candidate is launched before its attribute check passes. No module, static or device globals (run _rega_audit_cpp on the LO source); TMA maps encoded per call from that call's tensors; the judge's two state copies each get their own ints, no data_ptr equality. NVFP4 weights stay NVFP4 (register dequant only).

## What changed, and where
Base: worktree at 848b2c2e. Master (455705cd) differs from it only in this md, so no merge was needed. The worktree's submission.py starts from `hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py`, which is combination A's exact code. The diff against that file:

- **Master's rega module is untouched.** `_REGA_CPP`, `_REGA_HEAD`, `_REGA_BODY`, `_rega_build`, `_rega_setup` and the v1 fallback are unchanged. This was checked offline: sha256(`_REGA_CPP + _rega_src()`) = b7fbe28b..., the same v2 hash dfb755d3 recorded, and `_rega_src_v1()` still derives the source that hashes to `REGA_V1_SHA256`. prepare() prints both checks again (`_rega_lo_identity`), and LO stays OFF if v2 does not match.
- **New section (before kernel_fn): the separate `muse_prefill_rega_lo` extension.**
  - Source: `_REGA_HEAD + _rega_wgmma_src() + the namespace part of _REGA_BODY + "}  // namespace" + _REGA_LO_SRC`. `_REGA_LO_SRC` reopens the anonymous namespace. This is the same split pattern as `muse_prefill_rega_se`.
  - `_REGA_LO_SRC` is dfb755d3's `rega_lo_kernel<G, NMAX, STG, OCC>`, `consume_item_lo`, `info_lo`, `rega_init_lo` and `rega_gemm_lo`, unchanged except that the V64-occ1 instantiations are dropped. That leaves 4 instantiations, and `rega_gemm_lo` accepts variant 1..2 only.
  - Same `REGA_CUDA_CFLAGS`. `TORCH_CUDA_ARCH_LIST=9.0a` is set around load_inline, restored afterwards, and the arch flags are printed (`_rega_lo_build`).
  - Pre-build audit (`_rega_lo_audit`): the templated-accessor scan plus `_rega_audit_cpp` over the LO cpp and cuda sources. A finding turns LO OFF.
- **Constants.** `REGA_LO_VARIANTS = {1: V64 (64,3,2), 2: V32 (32,4,2)}`, `REGA_LO_REGS_OCC2 80`, `REGA_LO_POOL`, `REGA_LO_MAXBLK 1024`, `REGA_LO_STATIC_BOUND 12288`, `REGA_LO_START_DEADLINE_S 185`, `REGA_LO_GATE_DEADLINE_S 250`, and `REGA_LO_STATE_KEYS = REGA_STATE_KEYS + (rega_lo_ext, rega_low_variant, rega_low_grid)`. `REGA_FT_START_DEADLINE_S` and `REGA_COPY_DEADLINE_S` are unchanged.
- **Dispatch (`_experts_rega`).** At T <= REGA_LOW_MAX it reads the int `state.get('rega_low_variant', 0)` and the grid int `state.get('rega_low_grid', grid)`. A nonzero variant calls `state['rega_lo_ext'].rega_gemm_lo(which, variant, ...)` for w13, on the xs that the fused norm-gather or master's gather produced, and for w2. Everything else is unchanged. kernel_fn itself is unchanged: no prints, mutable state or try/except.
- **Small edits to master helpers, prepare-only, behaviour identical for master's calls.**
  - `_rega_case` accepts `pop:<alpha>`.
  - `_rega_replay_check` takes `keys=` (default `REGA_STATE_KEYS`). Its deepcopy memo shares every module-typed value, i.e. rega_ext and rega_lo_ext. Before, it shared rega_ext only.
- **prepare().** `_rega_lo_phase` runs after `_rega_setup`, `_fa3c_setup` and `_norm_gather_setup` have returned. Its summary lines print after the align2 and fa3c echoes, so `[rega-lo] PATH CHOSEN low: master|V64|V32 (...)` is the last prepare() line. The line before it echoes A's flags (rega_ok, norm_gather_ok), because the LO self-check output may push them out of the 16 KB log tail.
- **Phase order.** Identity check, then audit, then build, then `_rega_lo_attrs` (numRegs == 80, local 0, 2 CTAs/SM, binaryVersion 90, static <= 12288, geometry tag; checked before ANY launch). Then pop calibration, then per-candidate `_rega_lo_selfcheck`, then the gate, then the print-only SKEW CURVE. Any failure keeps variant 0. The phase never raises. If the diagnostic leaves an error word, the except path also resets to master.

### Review fixes applied (binding review + caller's numbered list)
1. **LO on master's two-phase align tables through the real path.**
   - Every self-check runs `_experts_rega` on `dict(state)` with the candidate selected, so `_align_dispatch` uses align2 wherever master gated it ON.
   - `_rega_lo_align_check` prints whether the align2 tables equal `_align_parallel`'s bit for bit (print-only), and requires `_rega_lo_tiles` equality at chunk sizes 8, 4 and 2.
   - Each case also prints the rel L2 of LO against master's rega on the same tables.
   - With norm-gather ON for the low class, prenorm=True cases are added: 753 random, 753 guard pop, 1024 local. So are `_ng_replay_check` on the LO state and the A/B/C replay check (`keys=REGA_LO_STATE_KEYS`, refilled inputs, 300 x {random, skewed, one}, 753 x {random, skewed, one, pop}).
   - The gate's w13-alone and w2-alone graphs also use align2 tables.
2. **Attribute check before every new launch.** The canary-equivalent self-checks, the gates and the SKEW CURVE all run only after `_rega_lo_attrs` passes. The SKEW CURVE candidate (V64, else the first passing one) must also have passed every self-check and replay check.
3. **Pop alpha grid refined and pick by guard median.**
   - `REGA_LO_POP_ALPHAS` = 1.0..3.0 in 0.1 steps (2.0-2.4 included), 3 draws each. The guard alpha is the one whose calibrated count is closest to `REGA_LO_GUARD_ROUTES = 1350`.
   - The guard line prints the achieved routes, the ratio to random and the whole calibration table.
   - The winner is the passing candidate with the lowest guard-routing block median, not the lowest random median.
4. **'LO SKIPPED'.** Every time-based skip says so explicitly: the start deadline 185 s, the self-check or replay deadline 250 s, the gate deadline, and a pass that arrives after the deadline (no switch). The phase then also prints `[rega-lo] LO SKIPPED (time): ... identical to combination A`.
5. **Gate as specified.**
   - Cold-L2 graph medians, `REGA_LO_ALTERNATIONS = 11` (>= 9), a 256 MB flush, empty-graph baseline subtracted.
   - Measured on the norm-gather-ON block (`prenorm` = `norm_gather_ok['low']`), at T=753.
   - ON only at random <= 0.95 AND guard <= 1.00. The winner's medians are asserted.
   - `LOW GATE` lines carry the block us/ratio, w13 and w2 kernel-alone us and TB/s, chunks, touched experts and max routes/expert.
   - The verdict line comes next, then `[rega-lo] SKEW CURVE 1.0x r0 1.2x r1 1.5x r2 1.8x r3 (...)`, with routes, touched experts, max routes/expert and max chunks/expert per point. The 1.2x and 1.5x points are print-only; the 1.0x and 1.8x points reuse the gate measurements.
   - `[rega-lo] PATH CHOSEN low: ...` is last.

### Deviations
- **Pick rule.** The spec's "pick the lower random-routing median" is replaced by the review's guard-median pick (caller item 3).
- **Guard target.** The guard targets an absolute 1350 local routes, not 1.8x of the draw. This follows the review's open question; 1350 ≈ 1.79x of 752.
- **Alternations.** 11 instead of 9, for a little less noise (still >= 9).
- **Diagnostic placement.** The SKEW CURVE 1.2x/1.5x points run after the decision, so they cannot delay or affect it.
- **Error word in the gate.** A nonzero error word after gate replays raises, which keeps master.
- **Open questions not built.** Routing synthetic h through the judge's router weights, and a forced-V64 control, are new ideas, so they are not implemented.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable. kernel_fn is unchanged; `_experts_rega` adds two prepare-set int lookups and a shape test (T <= REGA_LOW_MAX). There is no sync, no `.item()`/`.cpu()`, and no allocation or branch on input values. All new syncs are prepare-only (lint: outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached. The LO module keeps no globals (audited with `_rega_audit_cpp`). TMA maps are encoded per call from that call's tensors. Each state copy carries its own ints and shares the ext module.
- [x] Weights. Same `rega_bank` (lossless NVFP4 retile from master). The LO kernels dequantize in registers. NVFP4 stays NVFP4.
- [x] Activations. bf16 xs/hs/y as master's rega. KV cache and attention are untouched.
- [x] n/a: attention is untouched (the change is in the expert GEMMs only). Every key is still attended (FA3 / fa3c as in A).
- [x] No timing patches, harness access, module edits, reference import, subprocess or network. load_inline compiles in-process; there is no child process.
- [x] All work happens inside the timed call; no output caching or replay. The gate and checks are prepare-only.

okc_lint: `OK with warnings` (host-sync warnings only, all outside kernel_fn's call graph in prepare(); uint8 dtype warnings are master's existing bank/pad/fa3c buffers). py_compile OK; __pycache__ removed. An offline check confirmed: v2 sha256 = b7fbe28b (matches), the v1 fallback derives, the LO source passes `_rega_lo_audit` with 0 findings, braces and parens balance, and no undefined globals (symtable scan).

## Risks / what could fail
medium

## Result (after the run)
- job `73b40cd13fdf4bb0` (submit), status done, passed True, tests 5/5
- speedup 1.4677x, 95% CI [1.4457, 1.4898], score 263944.1735733959 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4604901800994607, "ci": [1.4385828273979688, 1.4823975328009524], "job": "9bfc488c36c84924"}; base result: muse_prefill_layer/hypotheses/done/neutral/fa3-full-contiguous-kv-triton-gather-gated.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/combo-b-normgather-two-cta-low-regate.result.json`, `muse_prefill_layer/hypotheses/done/neutral/combo-b-normgather-two-cta-low-regate.patch`; branch `hyp/muse_prefill_layer/combo-b-normgather-two-cta-low-regate`
