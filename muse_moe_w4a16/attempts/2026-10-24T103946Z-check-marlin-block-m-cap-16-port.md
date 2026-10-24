# muse_moe_w4a16 · 2026-10-24T103946Z · check · marlin-block-m-cap-16-port

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T100217Z-marlin-block-m-cap-16-port.md` (branch `hyp/muse_moe_w4a16/marlin-block-m-cap-16-port`).

- **Job:** `1c33392914594026` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `b848620`

## Goal
Port muse_decode_layer marlin-block-m-cap-16 (landed gain 1.062x, job 0730135cdcac43b7; patch in muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch) to kernel_fn in muse_moe_w4a16/submission.py. Add a module constant MARLIN_BLOCK_SIZE_M_CAP = 16 beside BLOCK_SIZE_M_CANDIDATES, and immediately after SGLang unchanged selection loop (first candidate in (8, 16, 32, 48, 64) with num_tokens*8/bank["count"]/block_size_m < 0.9, loop exhausting at 64) add the single line block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP). Master picks, verified from the rule with bank count 32: bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48, bs=256/257 -> 64 (not 32 at bs=128). After the cap: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 16 (4 -> 1); every bs<=33 is unchanged (8 and 16 are already <= 16) and is the in-run control. bs=32 is deliberately left at 16 rather than 8: 8 and 16 both run thread_m_blocks=1 (8 only switches to the m_block_size_8 special path), the only evidence on that step is next-larger bs=1/8/16 rows at 2-3% (inside noise once extra m-blocks for 9-16-route experts are paid), decode_layer kept 16 at T=32 when it gained, and marlin-block-size-m-next-smaller already tests bs=32 16 -> 8 from master. The capped value feeds moe_align_block_size (still over the bank 32 experts, and moe_align_single_token at bs=1 which is unaffected) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2), as it must. Nothing else changes: no input-dependent branch, no sync, the branch is on num_tokens only. Mechanism (as measured, not as decode first argued): per-m-block Marlin cost grows with thread_m_blocks (next-larger here: 48 -> 64 at bs=128 took Marlin 495 -> 662 us; 16 -> 32 at bs=32 lost 21%), while SGLang T*8/32 rule overcounts routes per local expert 8x under EP8 (true ~4-8 routes/expert at bs=128, ~8-16 at bs=256 on the busiest GPU), so block 16 adds an extra m-block (another weight pass) only for a few hot experts.

## What changed, and where
Implemented inline by churnsubs (a one-line change): `MARLIN_BLOCK_SIZE_M_CAP = 16` added beside
`BLOCK_SIZE_M_CANDIDATES`, and `block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP)` right after
SGLang's unchanged selection loop in `kernel_fn`. The capped value feeds `moe_align_block_size` (still
over the bank's 32 experts), `moe_align_single_token` and both Marlin calls' `moe_block_size`. Picks:
bs<=33 unchanged (8/16), bs=128 48->16, bs=256/257 64->16. No deviation from the spec.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-m-cap-16-port diff` (+4 lines: one constant, one line in kernel_fn, comments).

- [x] CUDA-graph capturable: a Python `min` on an int chosen from `num_tokens` (a shape); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: prepare() and the Marlin repack untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged; KV cache / query items n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention; alignment still covers every local route.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

okc_lint output:

```
warn  host-sync    L113: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/marlin-block-m-cap-16-port/muse_moe_w4a16/submission.py
```

## Risks / what could fail
low
Risk: the decode_layer evidence was measured with 256-expert align and dist=soft routing; here the align is over 32 experts and routing is skewed so this GPU is the busiest, so hot experts may hold more than 16 local routes at bs=256 and pay 2+ m-blocks (each a full pass over that expert 16 MB w13 / 8 MB w2 slice, partly from L2). If bs=256 is flat or loses while bs=128 gains, read it as hot-expert re-streaming and prefer the verify-two-smaller pick (32) at bs=256. Correctness unaffected (same fp32 reduce, same math); the bs=257 and skewed bs=128 tests run the new picks. Sweep: with marlin-block-size-m-next-smaller (one step down: bs=128 -> 32, bs=256 -> 48, bs=32 -> 8) and marlin-block-size-m-verify-two-smaller (two steps down: bs=128 -> 16, bs=256 -> 32), this run (cap 16: bs=256 -> 16) completes a 3-point sweep at bs=256 (48/32/16) against SGLang 64. At bs=128 this pick (16) coincides with verify-two-smaller; that overlap is intended for the direct port but means only one of the two needs to spend a check on bs=128, and the bs=128 rows of whichever runs second are a reproducibility read, not new information. All three edit the same line from master, so at most one lands; the others become stacked re-runs or are closed by this result. Run this one first if checks are limited: it is the replicated design.

## Result (after the run)
- job `1c33392914594026` (check), status done, passed True, tests 5/5
- speedup 1.0950x, 95% CI [1.0786, 1.1114], score 12281.463398511893 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.1297159216968498, "ci": [1.112770182871397, 1.1466616605223023], "job": "bacd1f502f4e46fa"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/marlin-block-m-cap-16-port.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/marlin-block-m-cap-16-port.patch`; branch `hyp/muse_moe_w4a16/marlin-block-m-cap-16-port`
