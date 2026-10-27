# muse_attention_decode · 2026-10-27T1453Z · submit · sk-onepass-verify1-timing-fix

3:1 rhythm ("use a scoring submission any time there is a gain to be secured"): scored run of the exact bytes of check
bfb0457c9ae84322 (branch hyp/muse_attention_decode/sk-onepass-verify1-timing-fix, impl 58f460d3, submission sha256 3b29647036aa1a23...).

- **Job:** `aa9fc86078f543fb` (`okc status <job>`)
- **Base:** board 1.6340 (4e386647). Check bfb0457c scored 1.6219.

## Why (a borderline call, recorded)
The one-pass ticket merge routed at sliding verify B=1:
- x1.026 / x1.024, 14.9 -> 14.6 us on the pre-attn + kernel pair; incumbent and twin timing-data rel L2 1.64e-03.
- The merge canary is exact, the twins compile without spill, and the two-state replay check is ok.
- Every other cell keeps the incumbent.
The headline (1.6219) is under the 1.6340 board, which was a high draw (attention checks today: 1.59-1.63). The gain is
~0.3 us on a weight-46 row (0.68%/us, ~0.2% headline), far under run-to-run spread. These bytes have never been scored,
attention has 6 scored runs left, and the board keeps the best.

## Rules check (each must be true before the run)
- [x] No weights involved (attention); KV FP8; queries no coarser than FP8 E4M3; every key attended.
- [x] fp32 LSE merge (the same math up to summation order; the canary checks M exact, L / o ~1e-7).
- [x] Activations bf16.
- [x] CUDA-graph capturable: routes frozen at import by config / batch band; no host sync in kernel_fn.
- [x] Nothing derived from `data` is cached across calls.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Result (after the run)

- 2026-10-27T1503Z: **1.6183x** (ci [1.5940463532676648, 1.6425959883925683]), passed. Below the 1.6340 board, which stays (scored run kept by the board is the best).
- Pooled with check 1.6219, the onepass bytes average ~1.620: about 0.9% under the board's high draw, inside today's 1.59-1.63 attention spread. The verify B=1 gain (14.9 -> 14.6 us, ~2% of one row) is too small to move the headline; not landed. The code stays on its hypothesis branch for stacking.
