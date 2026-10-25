# muse_prefill_layer · 2026-10-25T004119Z · check · experts-triton-persistent-grid-first-turn

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T184832Z-experts-triton-persistent-grid-first-turn.md` (branch `hyp/muse_prefill_layer/experts-triton-persistent-grid-first-turn`).

- **Job:** `5d9ddbbaaffd4599` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `f82ccce`

## Goal
Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (pointer setup, K loop, in-register dequant, tl.dot, epilogue), EXPERT_CONFIGS, moe_align_block_size, silu_and_mul, the weighted sum, ZERO_NONLOCAL_W2 and the Marlin path at T <= 4096 all stay as they are. MECHANISM: today the grid is cdiv(EM, BLOCK_T) * (N // BLOCK_W) with EM = sorted_ids.shape[0], the static capacity of moe_align_block_size(local_ids, 256, 256) (about 131072 + 257*255 slots at T=16384, ~768 token tiles). Only ~80 token tiles hold local routes; ~448 are -1 tiles (the remote routes, which SGLang's align buckets into blocks of their own: w13 returns, w2 zero-stores a 256x128 tile) and ~240 lie past n_padded (return). A BLOCK_T 256 CTA holds ~145 KB of smem and 256 threads at ~255 regs, so one CTA fits per SM, and ~44K dead w13 CTAs and ~22K w2 CTAs run one after another on each SM at ~1 us each (two dependent global loads, then exit): an estimated 0.4-0.6 ms of the 8.1-8.9 ms expert row. CHANGE: make the kernel persistent. Launcher: grid = (min(NUM_SMS, cdiv(EM, BLOCK_T) * (N // BLOCK_W)),) with NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count (a host int, read once in prepare() or at import; it is a hardware constant, not input-derived). Kernel: load n_padded once at the top; num_pid_t = tl.cdiv(n_padded, BLOCK_T) (runtime: only tiles below n_padded, which includes the -1 tiles); num_tiles = num_pid_t * num_pid_w; then `for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):` and inside it compute group_id, first_pid_t, group_size_t, pid_t, pid_w from `tile` with exactly master's GROUP_SIZE_T formulas (using the runtime num_pid_t); load off_expert; `if off_expert == -1:` master's ZERO_NONLOCAL store (unchanged), `else:` master's body unchanged. Triton has no `return`/`continue` inside a loop, so use the if/else; the inner K loop stays the innermost loop so Triton still pipelines it with the config's num_stages. Keep do_not_specialize on num_valid_tokens and EM. _check_expert_gemms runs unchanged (its -1 block and past-n_padded block assertions cover the new loop) and its log line adds `persistent grid=<g>`; if any variant now spills (n_spills > 0 where master had 0), print it so the run can be read.

## What changed, and where
Worktree `muse_prefill_layer/submission.py` only (+224/-47). Not run (no GPU on this node).
- Per-tile math moved verbatim into `@triton.jit _nvfp4_live_tile`; master's `_moe_nvfp4_gemm_kernel` is otherwise unchanged (early loads, return-on-dead, same swizzle) and calls it, so it stays the A/B baseline.
- New `_moe_nvfp4_gemm_persistent_kernel` + `_persistent_tile_coords` (master's GROUP_SIZE_T swizzle with runtime `num_pid_t = cdiv(n_padded, BLOCK_T)`, group size clamped to >= 1 for the masked prefetch of a past-the-end tile). Per tile: `expert_ids` only (next tile's id prefetched at loop top, loop-carried), `sorted_ids`/`offs_token` loaded inside the live branch (and inside the -1 branch only when ZERO_NONLOCAL). if/else, no return in the loop.
- Launcher `_nvfp4_moe_gemm(..., persistent=False)`: grid = min(`_num_sms(device)`, cdiv(EM, BLOCK_T) * N/BLOCK_W); `_num_sms` is a host hardware constant read per launch (not cached, not input-derived).
- `_check_expert_gemms` now runs master and persistent variants (w13/w2 each), keeps the -1 and past-n_padded assertions; the persistent case uses live token blocks so live tiles >= 2*NUM_SMS (loop wraps). Log line says `persistent grid=<g>` or `non-persistent` with n_regs/n_spills/shared for both variants.
- `_bench_persistent` (prepare, untimed): T=16384 random distinct top-8 of 256 -> 32 local, `moe_align_block_size(local_ids, 256, 32)`, CUDA-graph replays of both kernels for w13 and w2, 7 alternations (order flipped each round), 256 MiB flush write before every replay (cold L2), prints per-GEMM medians and ratio and a `PATH CHOSEN` line. `state["persistent"][w13|w2]` is True only at ratio <= 0.98; any exception -> non-persistent. `_experts_triton` passes it per GEMM.
- Deviation: the gate is per GEMM, not joint. Extra memory in prepare: ~4 GiB of synthetic buffers, freed before returning.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only dict reads of bools and a launcher host call (`get_device_properties`); no .item/.cpu/sync, no data-dependent control flow (diff checked; `.cpu()`/sync only in prepare).
- [x] Nothing derived from `data` is cached across calls: `state["persistent"]` comes from synthetic prepare() data; SM count is hardware.
- [x] Weights: unchanged; NVFP4 stays NVFP4 (same in-register dequant).
- [x] Activations BF16 / KV FP8 / queries: n/a, no attention or dtype change; the flush buffer is a uint8 scratch (lint warn, not an activation).
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: none added (no child process).
- [x] All work inside the timed call; no output caching or replay: GEMMs run every call; the benchmark outputs are scratch, in prepare only.

okc_lint: `OK with warnings` (host-sync warns all in prepare(), one dtype warn for the uint8 flush buffer, none in kernel_fn). `py_compile` ok, `__pycache__` removed.

## Risks / what could fail
medium
Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) replaces this kernel at first_turn with a persistent CUDA kernel that skips the same dead items; this is the cheap Triton-side hedge and it measures the dead-CTA cost that design assumes away. If the CUDA kernel lands and wins, this is moot. Risks: dead CTAs may retire faster than ~1 us (then ~1-2%, judge on the kernel row); the persistent loop may raise register pressure or change the pipeliner's schedule (read n_regs/n_spills in the log); tail imbalance with 132 programs over ~5120 w13 tiles is under one tile per program.

## Result (after the run)
- job `5d9ddbbaaffd4599` (check), status done, passed True, tests 5/5
- speedup 1.0726x, 95% CI [1.0565, 1.0887], score 361194.4412226105 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-persistent-grid-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-persistent-grid-first-turn.patch`; branch `hyp/muse_prefill_layer/experts-triton-persistent-grid-first-turn`
