# muse_attention_decode · 2026-10-26T012517Z · check · merge-perkvhead-v3chain-sv32-tiled

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T212052Z-merge-perkvhead-v3chain-sv32-tiled.md` (branch `hyp/muse_attention_decode/merge-perkvhead-v3chain-sv32-tiled`).

- **Job:** `b4b20425eaff4942` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `906f1bd4`

## Goal
DIAGNOSIS action 3(i), the PENDING_USER hand-merge. Start from branch hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled (built on master 5d528bf3 = per-kvhead; job a78f29aeab9b47cb, 1.5347: sliding decode_1 11.05 us, sliding verify_32 28.02 us, but full decode_1 19.71 us because it carries c320 V1 = pre-attn + c320 attn + c320 combine, three nodes). Replace ONLY that full decode_1 piece with the chain from combo-b1-chain-c320-fused-pre-pdl (branch hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl, job 8713665f, the current best: full decode_1 17.39 us = c320_attn_kernel<320,true> 11.4 + c320_combine_kernel<320,16> 4.2, i.e. V3 = FUSED pre-attention prologue in kernel A + combine, with V4 = V3 + PDL A->combine also built). Port combo-b1's muse_full_decode_c320 module source (variants V0..V4, FUSED and PDL host entry points as separate functions, no mutable toggle) and its host code verbatim: GATE A correctness for every variant incl. the new k/v cache row comparison against _pre_attention_kernel and the two-state alternating replay at 80000 and 4097 keys, the five-variant interleaved pass (41 reps) and the head-to-head re-time of the pass-1 best against V0 and V1 (31 reps), routing the best only if best/V0 <= 0.98, and its 'PATH CHOSEN: full decode_1 [chain]: V<k>' line. Delete the merge branch's V1-only c320 module and host code so exactly ONE c320 module is built (combo-b1's is a superset; the import clock is 222 s across 4 modules on master and must not gain a 6th build). Keep per-kvhead's sliding decode_1 cluster path (2 clusters x C=16) and the SV32 tiled pre-attention with its GATE B exactness and timing gate exactly as on the merge branch. Every other class is master byte for byte. Phase order: master's own setup first, then the c320 chain (own deadline 120 s), then SV32 tiled (own deadline), each falling back to master with one PATH CHOSEN line on failure or overrun, and the 480 s import start rule kept. No new CUDA: every kernel here already ran on the judge (8713665f, a78f29ae). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
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

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only `_C320_FN(data)` (torch.empty workspaces from page-table width, no sync/.item()); all `.item()`/synchronize are in import-time checks (lint: warnings only, outside kernel_fn).
- [x] Nothing derived from `data` cached across calls: `_C320_FN` is a frozen function chosen at import; workspaces are per-call; no device pointers/TMA maps held in globals.
- [x] Weights: n/a: attention problem, no weights.
- [x] KV FP8, q at most FP8 E4M3 (FUSED kernel rounds q to e4m3 in registers, as master's kernel), fp32 softmax/merge, bf16 out.
- [x] Attention sees every key (all 320-token items over seq keys incl. the new row); no skipping.
- [x] No timing patches/harness edits/subprocesses/network in code (load_inline's own ninja as in the other modules); timing only at import.
- [x] All work inside the timed call; no output caching or replay (checked against `git diff`).

okc_lint: `OK with warnings` (only host-sync warnings outside kernel_fn's call graph, and one dtype warning at L7264 in the ported synthetic-input helper path); `py_compile` OK; __pycache__ removed.

## Risks / what could fail
low
Expected headline about 1.545 (merge 1.5347 plus ~-2.3 us on full decode_1), i.e. +1.0-1.3% over the best 1.5295: at or just under the 1.24% one-run margin, so it is the base that sliding-verify1-pseudo-fused-preattn-rope stacks on (depends_on) for the scored submit. Risk is low: every kernel already ran on the judge; the risk is the 3-way hand merge and the import clock (one c320 module only).

## Result (after the run)
- job `b4b20425eaff4942` (check), status done, passed True, tests 16/16
- speedup 1.5360x, 95% CI [1.5130, 1.5652], score 2451.9750915976238 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/merge-perkvhead-v3chain-sv32-tiled.result.json`, `muse_attention_decode/hypotheses/done/neutral/merge-perkvhead-v3chain-sv32-tiled.patch`; branch `hyp/muse_attention_decode/merge-perkvhead-v3chain-sv32-tiled`
