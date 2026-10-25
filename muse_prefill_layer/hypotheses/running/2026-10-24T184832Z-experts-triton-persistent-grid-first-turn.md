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
status: running
review_at: 2026-10-25T003615Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T003624Z
base_sha: d310c3558b4da1934c0587b4005cb3c14093af5f
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-triton-persistent-grid-first-turn
worktree: .worktrees/muse_prefill_layer/experts-triton-persistent-grid-first-turn
mode: null
restack_of: null
stale_base: null
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

