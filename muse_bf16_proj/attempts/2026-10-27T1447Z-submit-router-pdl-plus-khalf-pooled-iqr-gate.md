# muse_bf16_proj · 2026-10-27T1447Z · submit · router-pdl-plus-khalf-pooled-iqr-gate

3:1 rhythm ("use a scoring submission any time there is a gain to be secured"): scored run of the exact bytes of check
023f4260e86e423e (branch hyp/muse_bf16_proj/router-pdl-plus-khalf-pooled-iqr-gate, submission sha256 6339277e6610988d...).

- **Job:** `254c83ae55834807` (`okc status <job>`)
- **Base:** board 1.2735 (bc147c5b; that run had pooler_khalf128_ok=False). Check 023f4260 scored 1.2723.

## Why (a borderline call, recorded)
Every cell routed in the check (FINAL FLAGS):
- pooler_khalf128_ok=True, under the new POOLED rule: rounds 1-3 x1.044 / x1.027 / x1.038, median 1.038. The 10-round
  diagnostic shows per-round IQR fail 0/10 and ratio < 1.03 in 1/10.
- router_pdl_ok_16 / 32 / 128 / 256 = True (x1.033-1.104).
The headline (1.2723) is just under the board, which was a khalf-OFF draw. bf16's run-to-run drift is 1.7-3.6%, far
above this stack's ~0.5%. These bytes have never been scored, bf16 has 9 scored runs left, and the board keeps the best.

## Rules check (each must be true before the run)
- [x] Weights lossless and unchanged.
- [x] fp32 accumulation (khalf fp32 partials + one add-cast; the router hi+lo fp32 with the same relaxed atomics).
- [x] Activations bf16.
- [x] CUDA-graph capturable: PDL edges are captured; routes frozen in prepare; no host sync in kernel_fn.
- [x] Nothing derived from `data` is cached across calls.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Result (after the run)
- 2026-10-27T1452Z: **passed, speedup 1.2740** (CI 1.255-1.293; score 3791.9 us). The new board (was 1.2735, bc147c5b); rank 388. FINAL FLAGS: pooler_khalf128_ok=True, router_pdl_ok 16/128/256 True, 32 False. Landed on master by hand.
