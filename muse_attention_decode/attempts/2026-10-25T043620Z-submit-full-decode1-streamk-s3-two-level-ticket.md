# muse_attention_decode · 2026-10-25T043620Z · submit · full-decode1-streamk-s3-two-level-ticket

Hypothesis: `muse_attention_decode/hypotheses/done/neutral/2026-10-25T003637Z-full-decode1-streamk-s3-two-level-ticket.md` (branch `hyp/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket`).

- **Job:** `341623fa53d94cef` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `dd46532`

## Goal
Promote check-proven gain full-decode1-streamk-s3-two-level-ticket (1.4322x [1.4107,1.4554], check job 2a3241780c7b4c46) to the board.

Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: stream-K over 64-token tiles with an in-kernel per-request ticket combine + norm-gate), whatever its score. Full decode_1 is 27.3 us (FA3 main 14.6 + FA3 combine 7.1 + pre-attn 1.3 + norm-gate) against a 4.3 us floor and weight 16 in the decode_1 load (42% of that load); stage-1 chunk320/chunk640 gated batch 1 onto splitkv and gained on full decode_1 but paid a separate combine over many partials. Change: (1) widen the gate to 1 <= batch <= 64 (config only), so full decode_1 runs master _pre_attention_kernel (PAIRED=False, as master batch 1 already does) + the stage-2 kernel. (2) At batch == 1 only (config-only), grid = SM count (one CTA per SM, ~9.5 tiles each for 80000 keys) instead of 2 x SM count, passed through the existing grid argument. (3) Two-level ticket for a request covered by many CTAs: CTAs whose ranges intersect request b are grouped by c // G with G = ceil(sqrt(nseg_b)) (from TP and grid only); each CTA writes its fp32 partial (acc[16,64], m[16], l[16]), __threadfence(), atomicAdd on group ticket gticket[b*32 + g]; the group last arrival LSE-merges its group partials into a group slot, __threadfence(), atomicAdd on ticket[b]; the last group LSE-merges the <= G group slots, applies v_scale, the stage-2 norm-gate epilogue and stores bf16, and resets both its tickets to 0 for graph replay. Batch > 1 keeps the stage-2 single-level ticket (nseg_b <= ~17 there). gticket is a module-level int32[64*32] zero buffer allocated at import. BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary + self-check, each kernel followed by torch.cuda.synchronize() and a print; never a child process); add batch-1 self-check cases at 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids 3/7/SM count, and two consecutive replays to prove both ticket levels reset. Fallback on failure: batch 1 returns to master FA3 (PATH CHOSEN: full decode_1 FA3 (fallback: <reason>)); success prints PATH CHOSEN: full decode_1 streamk two-level. Every other shape keeps the stage-2 code. First measurement is an okc check, never a scored submit.

## What changed, and where
All in `<worktree>/muse_attention_decode/submission.py` (stage 2 base f291b49; nothing could be compiled or run on this node).

- Gate: new shape-only `_shape_gate` (1 <= batch <= 64); `_splitkv_gate` = `_EXT` and shape and (batch > 1 or `_B1_PATH`). `_B1_PATH` is set only after the batch-1 validation passes, so a failure sends batch 1 to master FA3 (FA3 + norm-gate, unpaired as before; pairing still needs batch > 1, so the q layout and branch agree).
- Kernel: `splitkv_attn_sk_kernel<STAGES, TICKET, TWO>`; new variant 3 = `<4, true, true>`. Args added: `gticket` (int32[B*32]), `gacc`/`gml` (group slots). Variants 0-2 are unchanged in behaviour (batch > 1 keeps the stage-2 single-level ticket, grid 2xSM). `merge_slots` now wraps a generic `merge_range` that also returns the raw max.
- Two-level ticket (effective gx = min(gridDim.x, T) as in stage 2; CTAs above gx exit): for a non-whole segment, nseg = c_hi - c_lo + 1, Gs = ceil(sqrt(nseg)) (integer loop, from TP and grid only), group gi = (c - c_lo) / Gs (relative to c_lo rather than the spec's absolute c // G: same size bound, groups stay full-aligned), ngrp = ceil(nseg / Gs) <= 32. Each CTA stores its partial, `__threadfence`, `__syncthreads`, tid 0 `atom.add.acq_rel.gpu` on `gticket[b*32+gi]`. The group's last arrival (all threads `__threadfence` = acquire) LSE-merges its slots into group slot b*32+gi, resets its group ticket to 0, `__threadfence` + `__syncthreads`, tid 0 `atom.add.acq_rel.gpu` on `ticket[b]`; the last group merges the ngrp group slots, runs the stage-2 `finalize_store` (v_scale, norm-gate, bf16) and resets `ticket[b]` to 0. If ngrp == 1 the group merger finalizes directly. No CTA waits on another. `gticket` is a module-level zero buffer made at import (`_sk_gticket`, per device).
- Config at batch 1 (fixed by shape, not by timing): variant 3, grid = SM count; batch > 1 unchanged.
- Review fixes applied: (1) in-process eager cold-L2 timing of the four variants at B=1 x 80000 printed at import (`_sk_b1_diagnostics`, `[sk3-diag]`): (a) single-level grid 2xSM (control), (b) single-level grid SM, (c) two-level grid SM (timed), (d) two-level grid 2xSM, with rel L2 vs the FP32 reference, TB/s, a WARNING line if (c) is not fastest (path stays shape-fixed), master FA3 main+combine and norm-gate at the same input, and the two-level vs single-level kernel at B=16 as diagnostic only. (2) The review's open question on a parallel single-level merge was not pursued (the spec fixes the two-level design). Where review and spec differ nothing conflicts; the review's "rank by prepare timings only" is followed.
- Safety: module renamed `muse_full_decode_splitkv_v3` (TORCH_CUDA_ARCH_LIST=9.0a, -O3 -lineinfo, arch flags printed, binaryVersion == 90 gate, same plain cp.async.cg). Batch-1 validation `_sk_validate_b1` (after stage-2 validation, each kernel announced before and synced + printed after): 12 cases, B=1 at 80000, 4097, 63 and 1 keys, kv scale 1 and 0.25, grids 3/7/SM/2xSM, plus B=16 two-level; then two consecutive CUDA-graph replays (output NaN-poisoned) at 80000 and 4097 keys, and a check that both ticket levels sum to 0. Failure leaves batch 1 on FA3.
- PATH CHOSEN lines: `full decode_1 streamk two-level` or `full decode_1 FA3 (fallback: <reason>)`; the stage-2 lines for batch > 1 are kept.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn has no sync/.item()/prints; the new allocations (group slots) are shape-only; the tickets reset in-kernel and the graph replays are verified in the import self-check.
- [x] Nothing derived from `data` is cached: only the zero ticket buffers (device-only) persist.
- [x] n/a: no weights in this problem.
- [x] Activations BF16 as before; KV FP8; queries FP8 E4M3 (unchanged from stage 2).
- [x] Attention sees every key: tile ranges cover all keys of the request, no skipping (the git diff only changes the merge tree and the gate).
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (no subprocess/multiprocessing/os.fork anywhere; diff checked).
- [x] All work in the timed call; no output caching or replay.
- [x] Effective grid gx = min(gridDim.x, tiles) used by the group arithmetic; CTAs without work exit cleanly.

okc_lint: `OK with warnings: submission.py` (only host-sync warnings, all in import/prepare-time self-check code outside kernel_fn). `python3 -I -m py_compile` ok, __pycache__ removed. Not compiled or run (no nvcc/GPU here).

## Risks / what could fail
medium

## Result (after the run)
Promotion passed: 1.4441x [1.4225,1.4658], board: {"rank": 348, "team": "yesthatwilly", "speedup": 1.4441, "first": false, "ahead": 347, "to_first": 2.0149}
