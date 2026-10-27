# muse_decode_layer · 2026-10-27T1302Z · submit · w6g-hcgate

3:1 rhythm: scored run of the exact bytes of check 41975d1ce6334756 (campaign/eb-w6g-hcgate c62b501c, sha256 d6e5d459b808a9e5..., 975,200 bytes). The last decode scored run of 10-27.

- **Job:** `c1daf58bf192471e` (`okc status <job>`)
- **Base:** board 1.5697 (4a7a49d0ae82440f). Check 41975d1c scored 1.5749.

## Why
The check beat the board.
- The W6-G confirm race promoted H-16+comb at T=128 where the cells-off race said master: x1.023 vs master+cells
  (layer 350.5 -> 336.3 us). v16 rows now match the board's hybrid draw (x1.000 / x0.997 vs 4a7a49d0).
- PERHEAD d1s x1.036 and c320t d1f x1.038 on top. Fast silu everywhere, engines' canaries ok.
- The stack is now: W6-B + PERHEAD + c320t + fast silu + the T=128 hybrid made reliable.

## Rules check (each must be true before the run)
- [x] Weights lossless and unchanged.
- [x] fp32 accumulation; silu in fp32 (fast intrinsics); PERHEAD fp32 softmax and merge.
- [x] Activations bf16; KV FP8; queries <= FP8 E4M3; every key attended.
- [x] CUDA-graph capturable; every route frozen in prepare per state.
- [x] Nothing derived from `data` is cached across calls.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.
- [x] Size under 1,000,000 bytes (975,200).

## Result (after the run)
- 2026-10-27T1312Z: **passed, speedup 1.5920** (CI 1.568-1.616; 14208.6 us vs SGLang). The new board (was 1.5697, 4a7a49d0);
  rank 323. Landed on master.
  - T=128 H-16+comb (cells-off race H-16, kept). PERHEAD x1.126 / x1.032; c320t x1.044.
