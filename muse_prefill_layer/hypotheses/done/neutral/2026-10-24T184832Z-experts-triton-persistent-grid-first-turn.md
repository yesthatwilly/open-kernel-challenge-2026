---
problem: muse_prefill_layer
slug: experts-triton-persistent-grid-first-turn
created: 2026-10-24T184832Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)"
change: >
  Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change;
  the per-tile math (pointer setup, K loop, in-register dequant, tl.dot, epilogue),
  EXPERT_CONFIGS, moe_align_block_size, silu_and_mul, the weighted sum, ZERO_NONLOCAL_W2 and the
  Marlin path at T <= 4096 all stay as they are. MECHANISM: today the grid is cdiv(EM, BLOCK_T) *
  (N // BLOCK_W) with EM = sorted_ids.shape[0], the static capacity of
  moe_align_block_size(local_ids, 256, 256) (about 131072 + 257*255 slots at T=16384, ~768 token
  tiles). Only ~80 token tiles hold local routes; ~448 are -1 tiles (the remote routes, which
  SGLang's align buckets into blocks of their own: w13 returns, w2 zero-stores a 256x128 tile) and
  ~240 lie past n_padded (return). A BLOCK_T 256 CTA holds ~145 KB of smem and 256 threads at ~255
  regs, so one CTA fits per SM, and ~44K dead w13 CTAs and ~22K w2 CTAs run one after another on
  each SM at ~1 us each (two dependent global loads, then exit): an estimated 0.4-0.6 ms of the
  8.1-8.9 ms expert row. CHANGE: make the kernel persistent. Launcher: grid = (min(NUM_SMS,
  cdiv(EM, BLOCK_T) * (N // BLOCK_W)),) with NUM_SMS =
  torch.cuda.get_device_properties(device).multi_processor_count (a host int, read once in
  prepare() or at import; it is a hardware constant, not input-derived). Kernel: load n_padded
  once at the top; num_pid_t = tl.cdiv(n_padded, BLOCK_T) (runtime: only tiles below n_padded,
  which includes the -1 tiles); num_tiles = num_pid_t * num_pid_w; then `for tile in
  range(tl.program_id(0), num_tiles, tl.num_programs(0)):` and inside it compute group_id,
  first_pid_t, group_size_t, pid_t, pid_w from `tile` with exactly master's GROUP_SIZE_T formulas
  (using the runtime num_pid_t); load off_expert; `if off_expert == -1:` master's ZERO_NONLOCAL
  store (unchanged), `else:` master's body unchanged. Triton has no `return`/`continue` inside a
  loop, so use the if/else; the inner K loop stays the innermost loop so Triton still pipelines it
  with the config's num_stages. Keep do_not_specialize on num_valid_tokens and EM.
  _check_expert_gemms runs unchanged (its -1 block and past-n_padded block assertions cover the
  new loop) and its log line adds `persistent grid=<g>`; if any variant now spills (n_spills > 0
  where master had 0), print it so the run can be read.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel x2 (8149-8873 us on the four first_turn shapes, 37-58% of each) loses its ~66K dead CTAs: target 7.6-8.0 ms, i.e. -3 to -5% on first_turn shapes (sliding ctx32768, weight 138, is the heaviest row of the problem); ~+1-1.5% on the headline geomean"
  - shapes: "*load=turn_*"
    direction: neutral
    why: T <= 4096 runs Marlin; the Triton kernel only runs in prepare checks there
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T003615Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T003624Z
base_sha: d310c3558b4da1934c0587b4005cb3c14093af5f
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-triton-persistent-grid-first-turn
worktree: .worktrees/muse_prefill_layer/experts-triton-persistent-grid-first-turn
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T004119Z
impl_commit: f82cccec98bb61df139cb881efa9287d51bb212f
submission_sha256: a5897dd8c1546dc182ed1e23c3545fd869d6fef6bb57c0c9d5f1a3523f8f8c7d
norm_sha256: 273f451c9c54497e1a3c42621dcd160319cc524f79e8df40078df98c6496f942
submission_blob: 680ae9105b7863c930da7c8759029431bf2d4680
lint_warnings: 22
attempt_file: 2026-10-25T004119Z-check-experts-triton-persistent-grid-first-turn.md
prepared_at: 2026-10-25T004119Z
job: 5d9ddbbaaffd4599
dispatched_at: 2026-10-25T004152Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.0725632949628818
speedup_ci: [1.0564748455384385, 1.088651744387325]
harvested_at: 2026-10-25T034139Z
best_at_harvest: 1.074386115190733
promotion: null
---

# experts-triton-persistent-grid-first-turn

## Change

Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (pointer setup, K loop, in-register dequant, tl.dot, epilogue), EXPERT_CONFIGS, moe_align_block_size, silu_and_mul, the weighted sum, ZERO_NONLOCAL_W2 and the Marlin path at T <= 4096 all stay as they are. MECHANISM: today the grid is cdiv(EM, BLOCK_T) * (N // BLOCK_W) with EM = sorted_ids.shape[0], the static capacity of moe_align_block_size(local_ids, 256, 256) (about 131072 + 257*255 slots at T=16384, ~768 token tiles). Only ~80 token tiles hold local routes; ~448 are -1 tiles (the remote routes, which SGLang's align buckets into blocks of their own: w13 returns, w2 zero-stores a 256x128 tile) and ~240 lie past n_padded (return). A BLOCK_T 256 CTA holds ~145 KB of smem and 256 threads at ~255 regs, so one CTA fits per SM, and ~44K dead w13 CTAs and ~22K w2 CTAs run one after another on each SM at ~1 us each (two dependent global loads, then exit): an estimated 0.4-0.6 ms of the 8.1-8.9 ms expert row. CHANGE: make the kernel persistent. Launcher: grid = (min(NUM_SMS, cdiv(EM, BLOCK_T) * (N // BLOCK_W)),) with NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count (a host int, read once in prepare() or at import; it is a hardware constant, not input-derived). Kernel: load n_padded once at the top; num_pid_t = tl.cdiv(n_padded, BLOCK_T) (runtime: only tiles below n_padded, which includes the -1 tiles); num_tiles = num_pid_t * num_pid_w; then `for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):` and inside it compute group_id, first_pid_t, group_size_t, pid_t, pid_w from `tile` with exactly master's GROUP_SIZE_T formulas (using the runtime num_pid_t); load off_expert; `if off_expert == -1:` master's ZERO_NONLOCAL store (unchanged), `else:` master's body unchanged. Triton has no `return`/`continue` inside a loop, so use the if/else; the inner K loop stays the innermost loop so Triton still pipelines it with the config's num_stages. Keep do_not_specialize on num_valid_tokens and EM. _check_expert_gemms runs unchanged (its -1 block and past-n_padded block assertions cover the new loop) and its log line adds `persistent grid=<g>`; if any variant now spills (n_spills > 0 where master had 0), print it so the run can be read.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel x2 (8149-8873 us on the four first_turn shapes, 37-58% of each) loses its ~66K dead CTAs: target 7.6-8.0 ms, i.e. -3 to -5% on first_turn shapes (sliding ctx32768, weight 138, is the heaviest row of the problem); ~+1-1.5% on the headline geomean
- `*load=turn_*` neutral: T <= 4096 runs Marlin; the Triton kernel only runs in prepare checks there

## Risk

Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) replaces this kernel at first_turn with a persistent CUDA kernel that skips the same dead items; this is the cheap Triton-side hedge and it measures the dead-CTA cost that design assumes away. If the CUDA kernel lands and wins, this is moot. Risks: dead CTAs may retire faster than ~1 us (then ~1-2%, judge on the kernel row); the persistent loop may raise register pressure or change the pipeliner's schedule (read n_regs/n_spills in the log); tail imbalance with 132 programs over ~5120 w13 tiles is under one tile per program.

## Adversarial review

Reviewed 2026-10-25T003615Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is real, but it is smaller than the hypothesis says now that master aligns over 32 experts: about 45K dead CTAs, not 66K. The 3-5% kernel-row target is also smaller than this row's run-to-run noise (7463-9174 us for near-identical code). Run it, but with an in-process paired graph-timed A/B of master's kernel against the persistent kernel at T=16384 logged from prepare(). Without that, the single result cannot be read.

### Justification
A. The mechanism is plausible: at BLOCK_T 256 the w13 variant (254-255 regs x 256 threads, ~140 KB smem) fits one CTA per SM, so dead CTAs retire one after another per SM. But the hypothesis was written against 635cfee (align over 257 buckets). On master (stack-tonight-five) moe_align_block_size runs over LOCAL_EXPERTS=32: ~545 token tiles at T=16384, ~448 of them the -1 bucket, ~80 local, ~17 past n_padded. That leaves ~465 dead token tiles: ~29.8K dead w13 CTAs and ~14.9K dead w2 CTAs (zero-store OFF on master, so pure returns), ~45K total, ~340 per SM. At ~1 us each that is ~0.3 ms, 3-4% of the 8.1-9.2 ms row, so -3 to -5% on first_turn and ~+1% headline is the upper bound. The persistent loop does not make a dead tile free: each still pays a dependent load of expert_ids[pid_t], and master's body loads sorted_ids before the -1 branch, so the real saving may be about half. Calibration: moe-align-32-experts-first-turn removed ~21K dead CTAs and the four first_turn kernel rows did not move outside noise. Kernel-row noise across same-code runs is ~+-10%. Build notes: (1) load expert_ids first and move the sorted_ids/offs_token load inside the live branch; better, prefetch the next tile's expert id at the loop top. (2) w13 BLOCK_T 256 already spills 4 at 254 regs; log n_regs/n_spills for both variants vs master. (3) Tail imbalance 2-3%, acceptable. (4) GROUP_SIZE_T swizzle with runtime num_pid_t and a strided loop keeps L2 reuse.

### Confounders on the current kernel
- moe-align-32-experts-first-turn (in stack-tonight-five): already removed ~224 past-n_padded tiles, so the prize is ~45K dead CTAs, not 66K. Arithmetic; no A/B needed.
- w2-nonlocal-zero-store-off-first-turn (in stack-tonight-five): w2 -1 tiles are pure returns, so the loop saves only their launch cost; read the '[w2 zero-store]' log line.
- experts-triton-w4a16-first-turn-blockt256 / experts-hybrid-marlin-turn-triton-first-turn: BLOCK_T 256 forces one CTA per SM, which is the premise; master's config, fixed, not a confounder.
- router-hilo-one-gemm-n512, moe-align-local-routes-parallel-port, silu-mul-triton-local-rows-marlin-turn: separate kernels; no confound.

### Interference and A/B needs
No companion makes this readable; the ambiguity is kernel-row noise (~+-10%). Fix it in the run itself: in prepare() (untimed), build realistic T=16384 routing (random distinct top-8 of 256 mapped to 32 local, as _check_experts_e2e does), align with moe_align_block_size(local_ids, 256, LOCAL_EXPERTS), then time CUDA-graph replays of master's non-persistent kernel and the persistent kernel, w13 and w2 each, interleaved over >=5 alternations with cold L2, and log per-GEMM medians and their ratio. That paired ratio is the readout; the judged headline is a sanity check. Optionally gate the persistent path ON only at ratio <= 0.98 (shape-class decision from synthetic data). Overlap: experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc (running) replaces this kernel at first_turn; first-turn-moe-dead-work-stack and experts-triton-turn4-persistent-local-align (queued) contain this change, so do not count them as independent evidence.

### Open questions
- Where does sgl-kernel's align place the -1 bucket? If contiguous, a follow-up could loop over live tiles only.
- Does the persistent loop change Triton's inner-K pipelining or add spills to w13 BLOCK_T 256?
- Should the next-tile expert_id prefetch be part of this change?
- If the CUDA wgmma remeasure wins at first_turn, this becomes moot.


## Implementation notes

Worktree `muse_prefill_layer/submission.py` only (+224/-47). Not run (no GPU on this node).
- Per-tile math moved verbatim into `@triton.jit _nvfp4_live_tile`; master's `_moe_nvfp4_gemm_kernel` is otherwise unchanged (early loads, return-on-dead, same swizzle) and calls it, so it stays the A/B baseline.
- New `_moe_nvfp4_gemm_persistent_kernel` + `_persistent_tile_coords` (master's GROUP_SIZE_T swizzle with runtime `num_pid_t = cdiv(n_padded, BLOCK_T)`, group size clamped to >= 1 for the masked prefetch of a past-the-end tile). Per tile: `expert_ids` only (next tile's id prefetched at loop top, loop-carried), `sorted_ids`/`offs_token` loaded inside the live branch (and inside the -1 branch only when ZERO_NONLOCAL). if/else, no return in the loop.
- Launcher `_nvfp4_moe_gemm(..., persistent=False)`: grid = min(`_num_sms(device)`, cdiv(EM, BLOCK_T) * N/BLOCK_W); `_num_sms` is a host hardware constant read per launch (not cached, not input-derived).
- `_check_expert_gemms` now runs master and persistent variants (w13/w2 each), keeps the -1 and past-n_padded assertions; the persistent case uses live token blocks so live tiles >= 2*NUM_SMS (loop wraps). Log line says `persistent grid=<g>` or `non-persistent` with n_regs/n_spills/shared for both variants.
- `_bench_persistent` (prepare, untimed): T=16384 random distinct top-8 of 256 -> 32 local, `moe_align_block_size(local_ids, 256, 32)`, CUDA-graph replays of both kernels for w13 and w2, 7 alternations (order flipped each round), 256 MiB flush write before every replay (cold L2), prints per-GEMM medians and ratio and a `PATH CHOSEN` line. `state["persistent"][w13|w2]` is True only at ratio <= 0.98; any exception -> non-persistent. `_experts_triton` passes it per GEMM.
- Deviation: the gate is per GEMM, not joint. Extra memory in prepare: ~4 GiB of synthetic buffers, freed before returning.

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only dict reads of bools and a launcher host call (`get_device_properties`); no .item/.cpu/sync, no data-dependent control flow (diff checked; `.cpu()`/sync only in prepare).
- [x] Nothing derived from `data` is cached across calls: `state["persistent"]` comes from synthetic prepare() data; SM count is hardware.
- [x] Weights: unchanged; NVFP4 stays NVFP4 (same in-register dequant).
- [x] Activations BF16 / KV FP8 / queries: n/a, no attention or dtype change; the flush buffer is a uint8 scratch (lint warn, not an activation).
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: none added (no child process).
- [x] All work inside the timed call; no output caching or replay: GEMMs run every call; the benchmark outputs are scratch, in prepare only.

okc_lint: `OK with warnings` (host-sync warns all in prepare(), one dtype warn for the uint8 flush buffer, none in kernel_fn). `py_compile` ok, `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 (opus). **Verdict: dispatch-with-fixes.**

Checked and fine: the baseline kernel inlines `_nvfp4_live_tile` with the same loads, early return and swizzle as master, so it does the same work (`HALF`/`GROUPS_HALF` are now dead locals there). Turn_1, turn_4 and the tests run Marlin (T <= 4096), so they are untouched. In the persistent loop, an emulation over num_pid_t 0..39, num_pid_w {1,2,3,32,64}, G {1,4,8} and nprog {1,7,132} shows the swizzle with runtime `num_pid_t = cdiv(n_padded, BLOCK_T)` is a bijection onto the live tiles and the strided loop visits each tile exactly once. The `max(...,1)` clamp only affects masked past-the-end prefetch ids. The prefetched `off_expert` always matches the current tile. Both if/else branches yield matching types. `n_padded` is read once on device. kernel_fn adds only dict reads and a host `get_device_properties`: no sync, graph-capturable. The gate uses synthetic data only, is keyed on the shape class (T > 4096 -> BLOCK_T 256), and adds only seconds and about 4 GiB to prepare. NVFP4 and the dequant are unchanged. No child process.

Fixes:
1. `prepare()` called `_check_experts_e2e` (which runs `_experts_triton`, which reads `state["persistent"]`) before `state["persistent"]` was set. That is a KeyError on every run. Now `state["persistent"]` starts as non-persistent before the e2e check.
2. The persistent variants ran inside `_check_expert_gemms`, so a build error or dequant mismatch failed prepare instead of falling back. They now run with `persistent=True` inside `_bench_persistent`'s try, so any failure prints `PATH CHOSEN: non-persistent`. Master's variants still assert as before.
3. When the gate picks a persistent GEMM, `_check_experts_e2e` runs again on the chosen path against Marlin. If it fails, the path goes back to non-persistent with a loud `PATH CHOSEN` line.

Residual risk: the gain (~0.3 ms upper bound) may land under the 0.98 gate. Then the run is neutral and the readout is the `[persistent A/B]` ratio line. Also, the nested loop may change the pipelining or spills of w13 at BLOCK_T 256 (see the `[expert gemm] ... persistent` n_regs/n_spills lines).

okc_lint after fixes: `OK with warnings` (host-sync in prepare only); `py_compile` ok; `__pycache__` removed.

## Result

- job `5d9ddbbaaffd4599` (check), status done, passed True, tests 5/5
- speedup 1.0726x, 95% CI [1.0565, 1.0887], score 361194.4412226105 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-persistent-grid-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-persistent-grid-first-turn.patch`; branch `hyp/muse_prefill_layer/experts-triton-persistent-grid-first-turn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1101.29 | 1020.62 | 1.0012 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1529.55 | 1371.56 | 1.003 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3318.71 | 3306.49 | 0.9994 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4849.27 | 4836.84 | 0.9987 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14705.32 | 14193.46 | 1.016 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14264.64 | 13901.86 | 1.0044 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 17746.69 | 18142.13 | 1.0024 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21733.75 | 21746.19 | 1.0043 | flat |

no shape moved outside noise. Expected hit: 0/2. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-persistent-grid-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

