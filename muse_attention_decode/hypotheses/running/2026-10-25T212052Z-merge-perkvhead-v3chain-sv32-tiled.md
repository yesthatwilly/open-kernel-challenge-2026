---
problem: muse_attention_decode
slug: merge-perkvhead-v3chain-sv32-tiled
created: 2026-10-25T212052Z
base: master
area: full decode_1 split-KV launch chain (merge onto per-kvhead master, with SV32 tiled pre-attention)
change: >
  DIAGNOSIS action 3(i), the PENDING_USER hand-merge. Start from branch
  hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled (built on master 5d528bf3 =
  per-kvhead; job a78f29aeab9b47cb, 1.5347: sliding decode_1 11.05 us, sliding verify_32 28.02 us,
  but full decode_1 19.71 us because it carries c320 V1 = pre-attn + c320 attn + c320 combine,
  three nodes). Replace ONLY that full decode_1 piece with the chain from
  combo-b1-chain-c320-fused-pre-pdl (branch
  hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl, job 8713665f, the current best:
  full decode_1 17.39 us = c320_attn_kernel<320,true> 11.4 + c320_combine_kernel<320,16> 4.2, i.e.
  V3 = FUSED pre-attention prologue in kernel A + combine, with V4 = V3 + PDL A->combine also
  built). Port combo-b1's muse_full_decode_c320 module source (variants V0..V4, FUSED and PDL host
  entry points as separate functions, no mutable toggle) and its host code verbatim: GATE A
  correctness for every variant incl. the new k/v cache row comparison against
  _pre_attention_kernel and the two-state alternating replay at 80000 and 4097 keys, the
  five-variant interleaved pass (41 reps) and the head-to-head re-time of the pass-1 best against
  V0 and V1 (31 reps), routing the best only if best/V0 <= 0.98, and its 'PATH CHOSEN: full
  decode_1 [chain]: V<k>' line. Delete the merge branch's V1-only c320 module and host code so
  exactly ONE c320 module is built (combo-b1's is a superset; the import clock is 222 s across 4
  modules on master and must not gain a 6th build). Keep per-kvhead's sliding decode_1 cluster
  path (2 clusters x C=16) and the SV32 tiled pre-attention with its GATE B exactness and timing
  gate exactly as on the merge branch. Every other class is master byte for byte. Phase order:
  master's own setup first, then the c320 chain (own deadline 120 s), then SV32 tiled (own
  deadline), each falling back to master with one PATH CHOSEN line on failure or overrun, and the
  480 s import start rule kept. No new CUDA: every kernel here already ran on the judge (8713665f,
  a78f29ae). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no
  timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may
  hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting
  int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo,
  in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM
  and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2
  graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path
  WINS by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "vs the merge run 19.71 us (c320 V1): V3 measured 17.39 us in 8713665f (0.791 of master two-level 22.4); about -2.3 us at ~0.33%/us"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "vs the best 8713665f (12.1 us, pre-per-kvhead base): per-kvhead measured 11.05 us in a78f29ae"
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "vs the best 29.0 us: SV32 tiled pre-attention measured 28.02 us in a78f29ae (0.964 of master)"
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_1 *"
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "merge-perkvhead-c320-sv32-tiled: that merge ported c320 V1 (three nodes, 19.71 us, lost full decode_1 to the best's 17.39) and this one ports combo-b1-chain-c320-fused-pre-pdl's V3/V4 chain (fused pre-attention prologue + PDL) instead, keeping per-kvhead and SV32 unchanged"
review: proceed
status: running
review_at: 2026-10-25T212855Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T212904Z
base_sha: b27495d07f7afbf34627adb964d4beee91b40864
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/merge-perkvhead-v3chain-sv32-tiled
worktree: .worktrees/muse_attention_decode/merge-perkvhead-v3chain-sv32-tiled
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T213549Z
impl_commit: 906f1bd409fc936d0460fe2c1373fc74c4d444e0
submission_sha256: 0c361e8ac5f80705c503d9813b85cff96fb415246d13f940bc8148817bae9129
norm_sha256: a8ca565fd226f89ad0371386c77c57a5c0ad4efb2e3e729262f38bc79141e003
submission_blob: 219c08d9b932ec2cd4ba9f53ab3386d7b72348b7
lint_warnings: 79
---

# merge-perkvhead-v3chain-sv32-tiled

## Change

DIAGNOSIS action 3(i), the PENDING_USER hand-merge. Start from branch hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled (built on master 5d528bf3 = per-kvhead; job a78f29aeab9b47cb, 1.5347: sliding decode_1 11.05 us, sliding verify_32 28.02 us, but full decode_1 19.71 us because it carries c320 V1 = pre-attn + c320 attn + c320 combine, three nodes). Replace ONLY that full decode_1 piece with the chain from combo-b1-chain-c320-fused-pre-pdl (branch hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl, job 8713665f, the current best: full decode_1 17.39 us = c320_attn_kernel<320,true> 11.4 + c320_combine_kernel<320,16> 4.2, i.e. V3 = FUSED pre-attention prologue in kernel A + combine, with V4 = V3 + PDL A->combine also built). Port combo-b1's muse_full_decode_c320 module source (variants V0..V4, FUSED and PDL host entry points as separate functions, no mutable toggle) and its host code verbatim: GATE A correctness for every variant incl. the new k/v cache row comparison against _pre_attention_kernel and the two-state alternating replay at 80000 and 4097 keys, the five-variant interleaved pass (41 reps) and the head-to-head re-time of the pass-1 best against V0 and V1 (31 reps), routing the best only if best/V0 <= 0.98, and its 'PATH CHOSEN: full decode_1 [chain]: V<k>' line. Delete the merge branch's V1-only c320 module and host code so exactly ONE c320 module is built (combo-b1's is a superset; the import clock is 222 s across 4 modules on master and must not gain a 6th build). Keep per-kvhead's sliding decode_1 cluster path (2 clusters x C=16) and the SV32 tiled pre-attention with its GATE B exactness and timing gate exactly as on the merge branch. Every other class is master byte for byte. Phase order: master's own setup first, then the c320 chain (own deadline 120 s), then SV32 tiled (own deadline), each falling back to master with one PATH CHOSEN line on failure or overrun, and the 480 s import start rule kept. No new CUDA: every kernel here already ran on the judge (8713665f, a78f29ae). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_1 *` faster: vs the merge run 19.71 us (c320 V1): V3 measured 17.39 us in 8713665f (0.791 of master two-level 22.4); about -2.3 us at ~0.33%/us
- `*layer=sliding load=decode_1 *` faster: vs the best 8713665f (12.1 us, pre-per-kvhead base): per-kvhead measured 11.05 us in a78f29ae
- `*layer=sliding load=verify_32 *` faster: vs the best 29.0 us: SV32 tiled pre-attention measured 28.02 us in a78f29ae (0.964 of master)
- `*layer=full load=decode_16 *` flat: untouched
- `*load=verify_1 *` flat: untouched

## Risk

Expected headline about 1.545 (merge 1.5347 plus ~-2.3 us on full decode_1), i.e. +1.0-1.3% over the best 1.5295: at or just under the 1.24% one-run margin, so it is the base that sliding-verify1-pseudo-fused-preattn-rope stacks on (depends_on) for the scored submit. Risk is low: every kernel already ran on the judge; the risk is the 3-way hand merge and the import clock (one c320 module only).

## Adversarial review

Reviewed 2026-10-25T212855Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a justified, low-novelty port: it swaps the merge branch's c320 V1 (19.71 us at full decode_1) for combo-b1's gated V0..V4 chain, which routed V3 at 0.791 in 8713665f and scored 17.39 us. Every row is already measured, so no companion is needed. The predicted ~1.545 is only about +1.0% over the 1.5295 best, inside run noise, so read the result row by row and from the PATH CHOSEN lines, not from the headline.

### Justification
A. The mechanism is measured, not inferred. In 8713665f the chain gate printed 'V3 (A+C) 18.98 us vs V0 24.00 us head to head, ratio 0.791; pass 1: V0 24.00, V1 21.41, V2 21.18, V3 18.91, V4 18.91', and the judged full decode_1 row was 17.39 us (c320_attn_kernel<320,true> 11.4 + combine 4.2). The merge run a78f29ae carried only V1 (its gate printed c320 21.6 vs two-level 24.0, ratio 0.900) and scored 19.71 us. Expect full decode_1 near 17.4-17.6 us, about -2.3 us x weight 16. The kernel is still about 4x its 4.3 us SoL, so this is a launch-chain gain, not a floor claim. The other two rows are the merge's own measured wins and are kept unchanged: sliding decode_1 11.05 (best 12.1, master per-kvhead 10.87) and SV32 28.02 (best 29.01). Arithmetic: 1.5347 x (1 + about 0.75% from the decode_1 load) gives about 1.545, matching the hypothesis. The expected directions and baselines are consistent: full decode_1 is faster than the merge but about flat against the best, which already had V3. sliding decode_1 and verify_32 are faster than the best. Everything else is flat. The prior is accurate and not a duplicate: merge-perkvhead-c320-sv32-tiled ported only V1, this ports V3/V4, and combo-b1 is not on the do-not-try list (DIAGNOSIS action 3(i) names exactly this merge). Weakest link: the headline. The whole expected gain over the best (~+1.0-1.3%) sits at or below the ~1.24% one-run margin and the ±1.5% per-row noise, so a single check can confirm the stack only through the three rows (FD1 ~17.4, SD1 ~11.0, SV32 ~28.0) and the PATH CHOSEN lines. The second weak link is port fidelity. The two branches use different route mechanisms (merge: `_C320_ROUTE` bool + `_IMPORT_T0` + shared `_cx_*` helpers that GATE B also uses; combo-b1: `_C320_FN` + `IMPORT_T0` as the module's first statement + its own `_c320_*` phases). Deleting the merge's c320 code must not delete the `_cx_clock`/`_cx_may_start`/`_cx_interleaved`/`_cx_phase_b` helpers that SV32's GATE B needs.

### Confounders on the current kernel
- sliding-decode1-cluster-per-kvhead (master 5d528bf3): disjoint row (SD1), but it is the reason the merge beats combo-b1 at SD1. Its `_slide_init` / `_sd_setup_body` route.pop(1) and the `_SD_ROUTE` freeze must stay byte-identical. If SD1 reads ~12 us or the '[slide] PATH CHOSEN: sliding B=1 PERHEAD ON' line is missing, that is a merge error, not a result.
- full-decode1-streamk-s3-two-level-ticket (V0 comparator at FD1): unchanged on 5d528bf3 vs combo-b1's base 7e12891 (both gates printed V0 24.00 us), so the 0.791 ratio carries over unconfounded.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (master skF FUSED_PRE at 1<B<=64): a different fused prologue, a different kernel and module, and no runtime interaction at B=1. The name collision is a risk: the slug's 'v3chain' (combo-b1 variant V3) is not master's 'v3 fused pre-attn'. The builder must port combo-b1's c320 FUSED code, not copy master's skF B>1 prologue layout.
- fused-preattention-triton / paired-heads-launch-cleanup (master _pre_attention_kernel): V3 reproduces its unpaired B=1 full-layer semantics (RMSNorm, no RoPE, e4m3, k/v scale) inside kernel A and writes the cache row itself. Any mismatch shows up as a GATE A stored-row failure, which sends the class to V0/V1 rather than causing a timing confound. The SV32 tiled kernel replaces this kernel's launch only at SV32 and is self-attributed by GATE B.

### Interference and A/B needs
No companion is warranted. Every piece is shape-gated to a different row (FD1: combo-b1 chain; SD1: master per-kvhead; SV32: tiled pre-attention). Each piece prints its own in-run A/B (the chain prints V0..V4 numbers plus a head-to-head; GATE B prints pre and step times), and there are judged references for every row: master 4b608df5 (FD1 22.70, SD1 10.87, SV32 28.83), merge a78f29ae (19.71 / 11.05 / 28.02) and combo-b1 8713665f (17.39 / 12.1 / 29.01). A gain or a loss is therefore attributable row by row in one run. An ab-revert of per-kvhead would reproduce 8713665f. A control with V1 only would reproduce a78f29ae. Neither tells us anything new, and both spend a check that DIAGNOSIS wants moved to the verify wgmma line.

### Open questions
- New refusal rule vs verbatim port: the brief adds 'refuse a variant with local bytes > 0', which combo-b1 never applied. The 8713665f log prints no regs/local-bytes line for the c320 instantiations, so it is unknown whether A<320,FUSED=true> (it builds q MMA fragments in registers) spills. If it does, this rule silently drops V3/V4 and the run degrades to the merge's V1 (~19.7 us), which wastes the check. Either print local bytes but do not refuse V3/V4 on it (they already passed on the judge), or have the builder check ptxas -v output if any nvcc is reachable.
- V3 and V4 tied in pass 1 (18.91 vs 18.91 us), so the routed variant can be either. Readout: name the routed variant from the PATH CHOSEN chain line. Either is a success at FD1 ~17.4-17.6 us. A FD1 row near 19.5-19.7 with V3/V4 routed would mean the merged base breaks the chain gain. A row near 22.7 means GATE A fell back.
- Gate standard vs today's LESSON (two interleaved rounds >= 1.03 with IQR): the chain gate (0.791) clears any standard. The SV32 GATE B routed at step ratio 0.975 (1.026x single round) in a78f29ae, which is below the new 1.03 bar. It has gone live and gained on the judge in three runs (b9d062aa 27.92, 52cbfaeb 27.42, a78f29ae 28.02 vs ~28.8-29.0). Keep it exactly as on the merge branch, as the brief says. Do not retrofit the 1.03 rule here, or SV32 will likely fall back and the run loses its verify_32 row.
- Import clock: merge ended its phases at 222.8 s and combo-b1's chain finished at 179 s on a base without the PERHEAD build. combo-b1's module (FUSED + PDL variants, with a FUSED-fail rebuild path) is larger than the V1-only module. Expect the chain phase to end around 250-280 s, well within the 480 s start rule. Confirm only one muse_full_decode_c320 load_inline appears in the log; two modules with the same name and different sources would be a build-cache hazard.
- Submit gating: at ~1.545 predicted vs 1.5295 best, a single check will not prove a new best. Treat it as the stacking base for sliding-verify1-pseudo-fused-preattn-rope (DIAGNOSIS action 3(ii)), and promote it to a scored submit only together with that stacked change, or if the check rows all land as predicted.


## Implementation notes

Built in `<worktree>/muse_attention_decode/submission.py` (7513 lines; +1672/-80 vs master blob 6d6afa47). Constructed from
the merge branch `hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled` (a78f29ae) with its V1-only piece A swapped for
the chain of `combo-b1-chain-c320-fused-pre-pdl` (8713665f), both sources taken from their branches.

- kernel_fn: the merge's `_C320_ROUTE`/`_full_decode_c320` line is replaced by combo-b1's
  `if _C320_FN is not None and _c320_gate(cfg): return _C320_FN(data)` (same position after the sliding branch);
  `_C320_FN` / `_c320_gate` defined above kernel_fn as in combo-b1. The SV32 tiled launch in kernel_fn and
  `_pre_attention_tiled_kernel` are the merge branch's, byte for byte. Master's per-kvhead sliding route is untouched.
- Piece A: combo-b1's block ported verbatim (constants, `_C320_CPP`, `_C320_CUDA` byte-identical per diff, `_build_c320(fused)`,
  `_c320_run`, V1..V4 entry points, `_c320_rows` stored-row check vs `_pre_attention_kernel`, `_c320_bounds`, 21 `_C320_CASES`,
  `_c320_alt` two-state A B B A A B replay at 80000 and 4097 keys, `_c320_timed`, `_c320_gate_run` 41 reps + 31-rep head-to-head,
  route only if best/V0 <= 0.98, `_c320_setup_body`, `_c320_setup`, the 'PATH CHOSEN: full decode_1 [chain]: V<k>' line).
  Called at the end of import inside try/except, after master's whole setup and before piece B.
- Deleted from the merge branch: its V1-only `_C320_CPP/_C320_CUDA`, `_build_c320`, `_c320_ready`, `_c320_launch`,
  `_full_decode_c320`, `_c320_case`, `_c320_gate_a`, `_cx_inc`, `_cx_phase_a`, `_C320_ROUTE/_C320_ARCH_FLAGS/_C320_BUILD_DEADLINE/_GATE_A_DEADLINE`.
  Kept: `_cx_clock`, `_cx_may_start`, `_cx_interleaved`, `_cx_phase_b` and all `_tiled_*` (GATE B), diff-verified identical.
- Only one c320 `load_inline` call site exists (module name `muse_full_decode_c320f`; the non-FUSED rebuild `muse_full_decode_c320`
  runs only if the FUSED build fails, as in combo-b1, so at most one of them is ever built). No 6th module in the normal case.
- `IMPORT_T0 = _IMPORT_T0` alias so combo-b1's `_c320_clock` shares the merge's import clock (started at module top).

Review / override fixes applied:
1. No two-round 1.03 rule on GATE B or the chain gate; both gates exactly as on their source branches.
2. c320 regs/local bytes printed by combo-b1's `[c320] <kernel>: ... regs N, local B B` lines (via `c320_attrs`); V3/V4 are NOT refused on local bytes > 0.
3. Exactly one c320 module; shared `_cx_*` helpers kept.
4. `TORCH_CUDA_ARCH_LIST=9.0a` is set only around `load_inline` in `_build_c320` and restored in `finally` (combo-b1 code).
5. GATE A's stored-row check (`_c320_rows`, V3/V4 <= 3 of 128 bytes differ, rel L2 < 2e-2) ported with the FUSED kernel.
Deviations: none beyond the alias and call-site wiring. Biggest risk: the three-way hand merge at import (clock ~250-280 s expected), and V3/V4 register spills being unknown (printed, not gated).
Not run (no GPU on this node).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only `_C320_FN(data)` (torch.empty workspaces from page-table width, no sync/.item()); all `.item()`/synchronize are in import-time checks (lint: warnings only, outside kernel_fn).
- [x] Nothing derived from `data` cached across calls: `_C320_FN` is a frozen function chosen at import; workspaces are per-call; no device pointers/TMA maps held in globals.
- [x] Weights: n/a: attention problem, no weights.
- [x] KV FP8, q at most FP8 E4M3 (FUSED kernel rounds q to e4m3 in registers, as master's kernel), fp32 softmax/merge, bf16 out.
- [x] Attention sees every key (all 320-token items over seq keys incl. the new row); no skipping.
- [x] No timing patches/harness edits/subprocesses/network in code (load_inline's own ninja as in the other modules); timing only at import.
- [x] All work inside the timed call; no output caching or replay (checked against `git diff`).

okc_lint: `OK with warnings` (only host-sync warnings outside kernel_fn's call graph, and one dtype warning at L7264 in the ported synthetic-input helper path); `py_compile` OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus), read-only plus diffs; no GPU on this node.

**Verdict: dispatch.** No fixes.

- Port fidelity, checked with `git show` + diff. Master blob at base b27495d0 = 6d6afa47 = the merge branch's base, so the merge branch is the right starting point. The c320 block (`_C320_CHUNK` .. end of `_c320_setup`, 1166 lines) is byte-identical to `hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl`. Its `_C320_FN`/`_c320_gate` preamble is also byte-identical. Every top-level def that both the merge branch and this file have is byte-identical to `hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled`, except for three groups. `_build_c320`, `_c320_case` and `_c320_gate` now equal combo-b1's versions. `kernel_fn` changes in one line: the c320 dispatch. `_sd_setup_body` differs only in the trailing header comment. That leaves `_cx_clock`, `_cx_may_start`, `_cx_interleaved`, `_cx_phase_b`, all `_tiled_*`, `_sd_*` and `_slide_*` identical to the merge branch.
- The c320 block calls these master helpers: `_pre_attention_kernel`, `_sk_prepare_q`, `_splitkv_reference`, `_splitkv_synthetic`, `_alt_sets`, `_graph_of`, `_rel_l2`, `_rezero_tickets`, `_sm_count`, `_say` and `Inputs`. Each one is byte-identical between combo-b1's base and this file. The constants FP8, LOG2E, RMS_EPS, ROW_BYTES and _ALT_KEYS are also identical. An AST scan found no undefined names.
- (1) No leftovers. `_C320_ROUTE`, `_full_decode_c320`, `_c320_ready`, `_c320_launch`, `_c320_gate_a`, `_cx_phase_a`, `_cx_inc`, `_C320_BUILD_DEADLINE`, `_GATE_A_DEADLINE`, `_C320_ARCH_FLAGS` and `c320_prepare` appear nowhere in the file.
- (2) Phase order is master setup, then `_sd_setup`, then `_c320_setup`, then `_cx_phase_b`. Each phase sits in its own try/except with a PATH CHOSEN fallback line. The chain keeps combo-b1's start gates on the import clock: build 120 s, checks 60 s, gate 60 s, all against the 480 s budget. `IMPORT_T0` is an alias of the module-top `_IMPORT_T0`, so both phases use one clock. GATE B starts only if clock + 40 <= 480, and prints a PATH CHOSEN line even when it is skipped.
- (3) The `_cx_*` helpers that GATE B needs are present and identical to the merge branch.
- (4) There is one `load_inline` call site. `muse_full_decode_c320f` is built normally. `muse_full_decode_c320` is built only if the FUSED build fails.
- (5) kernel_fn dispatches on `_C320_FN is not None and _c320_gate(cfg)`. That is config-only: batch, rope_theta and `_shape_gate`. `_C320_FN` is frozen at import. `_c320_run` allocates workspaces only from `page_table.shape[1]`, with no sync and no data-dependent control flow.
- okc_lint: OK with warnings. The host-sync warnings are all outside kernel_fn, plus one ported dtype warning in a GATE B synthetic helper. py_compile: OK. __pycache__ removed.

Residual risk: the import clock. The chain's phases can take up to 240 s on top of the setup that runs before it, so a slow build could push GATE B past its 440 s start gate. That would lose the SV32 row without failing the run. V3/V4 local bytes are printed but not gated.
