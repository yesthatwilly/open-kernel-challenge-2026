# muse_attention_decode · 2026-10-24T184626Z · check · sliding-fused-hoist-first-tile

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-sliding-fused-hoist-first-tile.md` (branch `hyp/muse_attention_decode/sliding-fused-hoist-first-tile`).

- **Job:** `ead13d58c6c843e9` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `06db73a`

## Goal
In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so the KV round trip overlaps the q load, RMSNorm and RoPE. Move the computation of lo, count, chunk, start, end (it needs only seq = seq_lens[b], already loaded from seq_ptr) to the top of the kernel, right after `d`/`freq`, then load tile 0: t0 = start + n, tmask0 = t0 < end, page0 = tl.load(pt_ptr + b * pt_stride + t0 // PAGE, mask=tmask0, other=0), and kb0 / vb0 with the same kvoff formula as the loop. Then the existing q load + _norm_rope + the `if s == S - 1` new-key block run as now (its stores go to cache_loc rows, which are positions >= seq and never in [lo, seq - 1], so the hoisted loads cannot read them). After it, process kb0/vb0 with the same qk, valid mask (t0 >= p_row - (WINDOW - 1)) and _online_update call as the loop body, then run the loop from start + BLOCK_N. The partial store, ticket and combine are unchanged; verify (q_len 8, not routed here today) is unaffected.

## What changed, and where
In `_sliding_fused_kernel` (worktree muse_attention_decode/submission.py): moved `seq = tl.load(seq_ptr + b)` (per the review it was NOT at the top in master), lo/count/chunk/start/end and `n` to right after `freq`, then issued tile 0's page-index load and raw uint8 K/V gathers (kb0/vb0, same kvoff formula, mask `t0 < end`) before the q load. After the `s == S - 1` new-key block, tile 0 is processed under `if start < end:` with the loop body's exact qk, `valid = tmask0 & (t0 >= p_row - (WINDOW - 1))` and `_online_update`; the loop now runs `range(start + BLOCK_N, end, BLOCK_N)`. Deviation: the `start < end` guard (equivalent to the loop not running) avoids a spurious update on empty chunks. kb0/vb0 are kept as uint8 and bitcast to e4m3 at use. Spill risk from live kb0/vb0 is unmeasured (no GPU here); check n_regs/n_spills.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only tl.load/dot code; no host syncs, allocations or value-dependent host control flow.
- [x] Nothing derived from `data` is cached across calls: diff adds no state.
- [x] Weights: n/a, no weight or prepare() change.
- [x] Activations BF16, KV cache FP8, q bf16: unchanged (diff does not touch dtypes).
- [x] Attention sees every key: tile 0 uses the loop's window mask; loop covers the remaining tiles; no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network in the diff.
- [x] All work inside the timed call; no caching or replay.
- [x] No triton.autotune added.

Lint: `CLEAN: muse_attention_decode/submission.py`

## Risks / what could fail
medium
Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches). Mechanism-overlapping with sliding-decode-blockn128-single-tile; if both run they combine trivially (at BLOCK_N 128 the hoisted tile is the whole chunk).

## Result (after the run)
- job `ead13d58c6c843e9` (check), status done, passed True, tests 16/16
- speedup 1.3449x, 95% CI [1.3248, 1.3651], score 2800.3508046768407 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/sliding-fused-hoist-first-tile.result.json`, `muse_attention_decode/hypotheses/done/regression/sliding-fused-hoist-first-tile.patch`; branch `hyp/muse_attention_decode/sliding-fused-hoist-first-tile`
