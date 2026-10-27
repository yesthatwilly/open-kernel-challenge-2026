# muse_moe_w4a16 · 2026-10-27T0947Z · submit · w6m-fork

3:1 rhythm, phase A (moe is decode's expert-block testbed): scored run of the exact bytes of check a80dda1ab8c14f48
(campaign/eb-w6m-fork 4e8a8be9, sha256 ab46fbb722c2dd83..., 641,473 bytes).

- **Job:** `2a7b5d2030894c4a` (`okc status <job>`)
- **Base:** board 1.7076 (cf937dd3656c4193, eacf0722 bytes). Check a80dda1a scored 1.6978.

## Why (a borderline call, recorded)
The check headline is under the board, but the fork routed at M=256:
- **Fork vs serial:** H-16f vs serial H-16 x1.024-1.060 in every round, and x1.107 vs master; gate mean 329.1 -> 316.2 us.
- **Canaries:** all four bitwise, with buffers poisoned before each replay.
- **Rows vs cf937dd3, each against its own SGLang time:** verify 32 (M=256) **x1.014**. The other rows read x0.98-0.99;
  that run's SGLang was 1.5-7% slower on every row, so this run sat slow overall.
- **Expected:** a real one-row gain worth about +0.2-0.6% headline.
- **Not a re-roll:** these bytes have never been scored, moe has 9 scored runs left, and the board keeps the best.

## Rules check (each must be true before the run)
- [x] Weights lossless and unchanged (same NVFP4 / W4A16 kernels; the fork only changes stream placement).
- [x] fp32 accumulation unchanged; the expert output and cache3 are bitwise equal to serial (canary).
- [x] Activations bf16.
- [x] CUDA-graph capturable: stream and events are made in prepare; the fork / join is recorded as graph edges.
- [x] Nothing derived from `data` is cached across calls.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Result (after the run)
- 2026-10-27T0954Z: **passed, speedup 1.7187** (CI 1.693-1.745). The new board (was 1.7076, cf937dd3); rank 406 (was 415).
  Landed on master.
  - M=256 ran H-16f (FORK), race mean 328.6 -> 316.1 us.
  - Rows vs cf937dd3 (rowcmp_moe): verify 32 **x1.023**, verify 16 x1.016; the rest x0.995-1.004.
