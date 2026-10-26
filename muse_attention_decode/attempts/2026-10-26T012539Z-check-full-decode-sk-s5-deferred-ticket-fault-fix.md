# muse_attention_decode · 2026-10-26T012539Z · check · full-decode-sk-s5-deferred-ticket-fault-fix

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T212052Z-full-decode-sk-s5-deferred-ticket-fault-fix.md` (branch `hyp/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix`).

- **Job:** `f1d6d493c24244f4` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `1f1606dd`

## Goal
DIAGNOSIS action 4: re-measure the deferred ticket arrive, which has never been measured because it faulted. Re-apply done/failed/full-decode-sk-s4-deferred-ticket-arrive.patch (built on c3c1b01) onto CURRENT master 5d528bf3 (the stream-K kernel splitkv_attn_sk_kernel and its sk_pick/sk_prepare are unchanged there; resolve by hand). Keep its mechanism exactly: template flag DEFER on the single-level ticket path (B > 1): a finished non-whole segment stores its partial and records c_b in flag_s[2 + ndef] (ndef < 2, flag_s is int32[4] in Lay), the loop keeps streaming, and after the last tile ONE __threadfence + __syncthreads, then each recorded request draws its ticket and the last arrival merges + finalize_store + resets. Job 7f10d2d8 failed 0/16 with an illegal address that surfaced inside the DEFER matrix (cuCtxSynchronize) and then the sk-diag; the sticky error poisoned every test. Fix the surface, not the design: (1) drop the CNT (variant 5, writes gticket[2047], which is also the two-level group slot c_b*32+gi for c_b=63) and SKIPM (variant 6) diagnostic instantiations; build only DEFER as variant 4. (2) Device guard in the DEFER epilogue: if cb is outside [0, B) or bx outside [c_lo, c_hi], set the error word and skip the draw instead of touching memory. (3) Host asserts before every DEFER launch: ws_acc/ws_ml rows >= grid + B, ticket numel >= B, and the matrix allocates its workspaces per cell from that cell's grid and B (assert, do not assume the self-check's sizes). (4) Matrix only at B=16 and B=32 (the scored bands) x grid 2xSM / SM; B 33..64 stays master variant 0. (5) Order: master's own sk-diag first, unchanged; the DEFER validate + matrix last; '[sk5] launching v4 B=.. grid=..' / 'returned' with torch.cuda.synchronize() around every DEFER launch, so a fault names its launch in the 15 KB log tail. (6) A canary per matrix cell (same B and grid, lengths 1..130) before its 80K timing. Route a band only if DEFER beats variant 0 by >= 1.02x in the interleaved graph replay and passes the fp32 self-check and the two-state alternating replay; print both numbers. First measurement is an okc check, never a scored submit. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
Edited only `.worktrees/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix/muse_attention_decode/submission.py` (+477/-23). Not run (no GPU on this node).

**CUDA (module source `_SK_CUDA`)**
- `splitkv_attn_sk_kernel` gets two new trailing template parameters after TOK: `DEFER` (6th) and `SKIPM` (7th), with `static_assert(!(DEFER && (PSEUDO || TWO)))` and an assert that SKIPM excludes PSEUDO/TWO/DEFER. Both are constexpr-dead for variants 0-7, so those instantiations compile as before.
- Variant 8 = DEFER `<4, true, false, false, 8, true>`, variant 9 = SKIPM `<4, true, false, false, 8, false, true>` (diag only). Both are picked OUTSIDE `#ifdef SK_PSEUDO`, so the non-PSEUDO fallback build has them. `SK_MAX_VARIANT = 9` in both builds; `sk_variant_ok()` keeps 4-7 invalid in the non-PSEUDO build; `sk_is_pseudo(v) = 4..7` replaces `variant >= 4` in `sk_attn` and `sk_smem`, and `tok` is 1 only for 6/7. 8 and 9 are excluded from the pseudo / tok / B<=32 predicates and use `Lay<4>` smem. `done[]` is `SK_MAX_VARIANT + 1` = 10 entries (the review said 9, but variant 9 SKIPM needs index 9). `_SK_LABELS` extended to ten entries.
- DEFER mechanism as in s4 (record c_b in `flag_s[2+ndef]`, `ndef < 2`, one fence + sync after the last tile, then the ticket draws). Device guard: a recorded request outside [0, B) or a CTA outside [c_lo, c_hi] does `atomicExch(ticket + 64, 1)` and skips the draw (no memory touched).
- Error word: slot 64 of the existing static `_sk_ticket` int32 buffer (grown 64 -> 128 ints; slots >= 64 are used by nothing else, unlike gticket[2047], which is a live two-level slot). It is read only in prepare (`_s5_err_word`); `_rezero_tickets` clears it, and every ticket-sum check also trips on it.
- Host asserts in `sk_attn` for variant 8 (2 <= B <= 64, ticket numel > 64 and >= B, ws rows >= grid + B with the right shape) and Python asserts (`_s5_assert_defer`) before every matrix DEFER launch; each matrix cell allocates its own workspaces from its grid and B.

**Prepare (Python)**
- Order: master's sk-diag first (unchanged except the added SKIPM variant-9 diag at B=16/32, one announced + synced eager launch first, then timed, with the fence-vs-merge gap split printed), then `_s5_phase()` LAST: `_s5_validate` (attrs of all sk variants incl. 4-7 local-byte warnings and the variant-0 identity line; 10 eager cases at B=2/16/32 + B=24x80000, no B=64; two-state alternating replays at B=16 for both grids, replaying pre-attn + variant 8 directly so skF cannot intercept) -> `_s5_band(16, route=True)` -> `_s5_band(32, route=False)` -> `_s5_probe_b64` (one eager variant-0 launch at B=64 x 80000, sync + print, after the matrix).
- Every DEFER launch prints `[sk5] launching v8 B=.. grid=..` and `returned (error word ..)` with `torch.cuda.synchronize()` around it. A canary per cell (lengths 1..130, same B and grid) runs before that cell's 80K timing.
- Gate (B=16 only): master's routed path (variant 0, `_SPLITKV_GRID or 2*SM`, flag-off) vs DEFER at 2xSM and SM; two interleaved rounds (rotation reversed in round 2), cold L2 (256 MB flush), empty graph subtracted, median/IQR of 20 printed per cell per round. Route only if BOTH rounds >= 1.03x AND the cell's IQR <= 1.5x master's in both rounds AND the band is not owned by skF (`_fused_gate`). B=32: numbers only, never routes. Route lands in `_SK_ROUTE[16] = (8, mult)`, read in `_full_decode_sk` by batch only; a miss leaves master's exact (variant 0, 2xSM) path.
- Deadlines: the phase starts only below 235 s of import clock, each band/probe below 275 s (`skipped: deadline` printed). All PATH CHOSEN/GATE lines of the phase are repeated as the last lines of prepare (`[sk5] SUMMARY`).

**Deviations / notes**
- `done[]` grown to 10, not 9 (see above). The review's variant-4-7 "keeps master's regs/local" check is a printed comparison: master's pseudo-variant reg counts are not recorded anywhere in the repo, so the `[sk5]` attrs lines must be compared by eye with the `[ps]` lines of master's log; variant 0 is checked against the landed 121/0/95264/2.
- No `_ab_flags`/NEW_CELL_FLAGS pattern exists in this problem, so there is no per-state flag; a miss falls back by construction.
- Biggest risk: variant 8 compile/regs (DEFER adds live state; a spill refuses it, < 2 CTAs/SM is only warned) and the unexplained s4 fault, which may not be DEFER-specific.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn change is one dict `.get` by batch and the same launch; no sync/.item()/alloc on values in kernel_fn (all syncs/.item() are prepare-only, lint: host-sync warnings only outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: `_SK_ROUTE` is keyed by config batch, written only in prepare; the ticket buffers are self-resetting int32 (error word is a device-only int, read in prepare).
- [x] Weights: n/a: no weight handling touched.
- [x] Activations / KV / queries: KV FP8, e4m3 queries, fp32 softmax/merges, bf16 out, unchanged (DEFER only reorders when the ticket draw happens).
- [x] Attention sees every key: tile ranges and merges unchanged; DEFER merges exactly the same slots.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: timing only in prepare, no child processes.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only `host-sync` warnings for prepare-only synchronize()/.item() outside kernel_fn's call graph); `python3 -I -m py_compile` passes; `__pycache__` deleted.

## Risks / what could fail
medium
Weighted effect ~+0.6-0.7% headline: below the 1.24% margin, so it can only land stacked (with the merge pair or a verify win); judge this check on the splitkv_attn_sk_kernel rows at decode_16/32 and the printed DEFER gate.

## Result (after the run)
- job `f1d6d493c24244f4` (check), status done, passed True, tests 16/16
- speedup 1.5162x, 95% CI [1.4934, 1.5389], score 2484.075876135329 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode-sk-s5-deferred-ticket-fault-fix.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode-sk-s5-deferred-ticket-fault-fix.patch`; branch `hyp/muse_attention_decode/full-decode-sk-s5-deferred-ticket-fault-fix`
