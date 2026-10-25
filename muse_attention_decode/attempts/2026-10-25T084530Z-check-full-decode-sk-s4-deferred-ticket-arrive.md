# muse_attention_decode · 2026-10-25T084530Z · check · full-decode-sk-s4-deferred-ticket-arrive

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T054448Z-full-decode-sk-s4-deferred-ticket-arrive.md` (branch `hyp/muse_attention_decode/full-decode-sk-s4-deferred-ticket-arrive`).

- **Job:** `7f10d2d820cd49ee` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `a94b74ed`

## Goal
Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs 106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.09 TB/s), and 221.5 vs 218.2 us at B=32 ([sk-diag] in c0f1945a108c4aa2 and f519261b7c2b435f logs): the in-kernel combine costs ~6 us at B=16 against the no-ticket kernel, before the separate combine it saves. Where: at every segment end inside the tile loop the CTA does a __threadfence + 2 __syncthreads + atomicAdd round trip (and the last arrival a serial merge_slots of ~16 slots) while its next segment's tiles wait. Change (new template flag DEFER on the single-level ticket path, B>1 only; DEFER=false byte-identical): inside the loop a finished non-whole segment only stores its partial (ws_acc/ws_ml slot bx + c_b) and records c_b in a small smem list (<= 4 entries; at grid 264 a CTA holds <= 3 segments for B<=64, assert it); the loop continues streaming with no fence or atomic. After the last tile, ONE __threadfence + __syncthreads, then for each recorded request draw the ticket (atomicAdd == c_hi - c_lo) and the last arrival merges + finalize_store + resets the ticket, exactly as today. Whole segments still finalize in place. Correctness argument: the ticket draw still follows the release fence of every partial it counts. Variant matrix in prepare: DEFER on/off x grid 2xSM/SM at B=16, B=32, B=64; route DEFER only if it beats master's variant 0 by >= 1.02x in interleaved graph replay; B=1 two-level unchanged. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
Worktree base 0561d7b is current for submission.py (no master commit touched it since; no merge needed). Edited only `muse_attention_decode/submission.py` (+292/-14).

What changed
- `splitkv_attn_sk_kernel<STAGES, TICKET, TWO, DEFER, CNT, SKIPM>` (new defaults false). DEFER (variant 4) on the single-level ticket path: a finished non-whole segment stores its partial (slot bx + c_b) and records c_b in `flag_s[2 + ndef]` (the unused FLAG ints; no smem growth, so occupancy is unchanged), no fence or atomic in the loop. After `cp_wait<0>`: ONE `__threadfence` + `__syncthreads`, then per recorded request: ticket atomicAdd (== c_hi - c_lo), last arriver does `__threadfence` (acquire) then merge_slots + finalize_store + ticket reset, `__syncthreads` before the next draw rewrites `flag_s[0]`. Whole segments finalize in place as before.
- Variant 5 = DEFER + CNT (self-check/diag only): a CTA that wins two end merges bumps `gticket[2047]`; host zeroes that slot before and after every read. Variant 6 = SKIPM (diag only, never routed): last arriver resets the ticket but skips merge + finalize.
- Host: variants 0..6 in sk_pick/sk_prepare/sk_attn/sk_attrs; `_SK_ROUTE` (batch -> (variant, grid multiple)), written only in prepare, read by config (batch) in kernel_fn, default (0, 2) = master. B=1 two-level `<4,true,true>` and `<4,true>`/`<4,false>`/`<5,true>` are unchanged template code (every new branch is constexpr-false for them).
- Prepare: `_sk_defer_validate` (attrs, 13 cases x variants 4 and 5, double-last-arriver counts, two-state alternating graph replays at B=16/32/64 for grid 2xSM and SM, tickets zero), `_sk_defer_matrix` (DEFER on/off x grid 2xSM/SM at B=16/32/64 x 80000, one graph per cell, interleaved rotating replays, 256 MB flush, empty-graph baseline subtracted, median of 20; routes a band to DEFER only if best DEFER cell >= 1.02x faster than master variant 0; PATH CHOSEN prints both numbers). Any validation miss leaves `_SK_ROUTE` empty (master) with a PATH CHOSEN line naming the failing case.
- `_sk_diagnostics`: new `[sk-diag]` row "merge skipped" (variant 6) at B=16 and 32 plus a printed gap split (timed - skipped = end-merge tail; skipped - no-ticket A = in-loop fence + ticket cost).

Review fixes applied
1. Size the deferred list on non-whole segments (<= 2, not total segments), no trap/assert: list capacity 2; a hypothetical third would take the immediate-arrival path. No `assert`/`__trap`.
2. Single post-loop `__threadfence` after all partial stores and before every atomicAdd; acquire `__threadfence` before merge_slots: done.
3. Diag-only [sk-diag] row with merge+finalize skipped, never routed (variant 6): done.
4. Double-last-arriver count in the self-check: printed per case and at B=16/32/64 x 80000.
5. B=1 two-level instantiation byte-identical: template defaults unchanged, new code constexpr-dead for it (cannot verify SASS here: no GPU/nvcc).
6. Open questions: the diag row answers the merge-tail question (printed in the prepare log); measurement asymmetry in the no-ticket number remains the existing eager diag (unchanged); the matrix itself is graph-replay interleaved.

Deviations: deferred list in the existing `flag_s` smem ints and a register count rather than a new smem array (same capacity, no layout change). Whole-segment and non-routed-batch behavior unchanged.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn has no sync/alloc on input values; routing is a dict lookup on `config["batch"]`; two-state graph captures are in the self-check.
- [x] Nothing derived from `data` cached across calls: `_SK_ROUTE` is keyed by batch, set in prepare only; tickets are the existing self-resetting arrays; no device pointers in globals.
- [x] Weights n/a: attention only, no weights.
- [x] KV FP8, q E4M3, fp32 merges, bf16 out: math untouched.
- [x] Attention sees every key: tile ranges unchanged; only arrival timing moved.
- [x] No timing patches, harness access, subprocesses or network (grep for subprocess/fork/multiprocessing: none).
- [x] All work inside the timed call; no output caching (SKIPM variant 6 is diag only and never routed).

okc_lint: `OK with warnings: muse_attention_decode/submission.py` (only host-sync warnings in prepare-side code, "fine in prepare()"). `py_compile` passes; __pycache__ removed.

## Risks / what could fail
medium
Small (~0.3-0.5% headline) but attributable on two weight-16 rows via the kernel row. Composes with v3 fused-preattn later (both are template flags on the same kernel). Risk: deferring arrivals lengthens the time the last request waits for its earliest segments' owners; the gate decides.

## Result (after the run)
- job `7f10d2d820cd49ee` (check), status done, passed False, tests 0/16
- first failing: {"batch": 1, "ctx": 10, "heads": "tp8", "layer": "sliding", "seed": 6893170448766904337}: AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1

- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json
- **outcome: failed** (test:{"batch": 1, "ctx": 10, "heads": "tp8", "layer": "sliding", "seed": 6893170448766904337})
- files: `muse_attention_decode/hypotheses/done/failed/full-decode-sk-s4-deferred-ticket-arrive.result.json`, `muse_attention_decode/hypotheses/done/failed/full-decode-sk-s4-deferred-ticket-arrive.patch`; branch `hyp/muse_attention_decode/full-decode-sk-s4-deferred-ticket-arrive`
