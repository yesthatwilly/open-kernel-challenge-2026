---
problem: muse_prefill_layer
slug: first-turn-moe-dead-work-stack
created: 2026-10-24T184911Z
base: master
area: MoE dispatch + silu-mul + Triton NVFP4 grid at T > 4096 (first_turn dead-work removal, combined)
change: >
  Built on master (635cfee). Touches only the T > 4096 path (_experts_triton and the Triton
  kernel's launch); _experts_marlin (T <= 4096) is byte-identical. Three parts, all aimed at work
  the first_turn expert block does on rows that are not this GPU's: (1) PERSISTENT GRID, exactly
  as hypothesis experts-triton-persistent-grid-first-turn specifies (grid = min(NUM_SMS, static
  tiles); loop `for tile in range(pid, num_tiles, num_programs)` with num_tiles = cdiv(n_padded,
  BLOCK_T) * (N // BLOCK_W) read on device; master's GROUP_SIZE_T mapping on `tile`; if/else
  instead of return). (2) LOCAL-ROUTES ALIGN in _experts_triton only, exactly as hypothesis
  moe-align-local-routes-parallel-port specifies (copy muse_moe_w4a16's
  _align_parallel_kernel/_align_parallel, local_start read from data['local_start'] on device,
  local_count 32, block = cfg['BLOCK_T'] = 256): it replaces _local_expert_ids +
  moe_align_block_size on this path, so there are no -1 blocks below n_padded and w2 writes
  nothing for remote slots (ZERO_NONLOCAL_W2 then never fires; leave the flag as is). kernel_fn
  must pass topk_ids and data['local_start'] down so this path gets them; the Marlin path keeps
  master's remap and align. (3) SORTED SILU: replace silu_and_mul(cache1, cache2,
  expert_ids=local_ids.view(-1), expert_step=1) in _experts_triton with a new Triton kernel
  _silu_mul_sorted_kernel(c1, c2, sorted_ids, n_padded, num_slots, N=4096, BLOCK_R=16,
  BLOCK_N=512), grid (cdiv(sorted_ids.shape[0], BLOCK_R), N // BLOCK_N), num_warps 4: load
  n_padded; if pid0*BLOCK_R >= n_padded return; rows = pid0*BLOCK_R + arange; s = load(sorted_ids
  + rows, mask=rows < n_padded, other=num_slots); valid = s < num_slots (with part 2 every valid s
  is a local slot); cols = pid1*BLOCK_N + arange; g = c1[s, cols], u = c1[s, N + cols] (bf16), out
  = (g_f32 / (1 + exp(-g_f32))) * u_f32 rounded once to bf16, store c2[s, cols] under valid. It
  reads ~T local rows (16384 x 24 KB) instead of launching over all T*8 slots. prepare() gates,
  each one log line, any failure turns the whole stack off for the run (master's _experts_triton
  runs; log '[ft-stack] OFF: <reason>', else '[ft-stack] ON'): the align self-check vs
  moe_align_block_size per-expert route sets at T=8192 and 16384; the NaN-poison check
  (fused_post_expert_norm_weighted_sum with NaN in remote slots equals the zero-filled call); the
  silu kernel vs silu_and_mul on the local rows of a synthetic cache1 (max diff <= 1 bf16 ulp);
  _check_expert_gemms (now also exercising the persistent loop) and _check_experts_e2e adapted to
  the new align (Triton stack vs Marlin, rel L2 <= 2e-2).
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Removes, per first_turn shape: ~66K dead Triton CTAs (est. 0.3-0.6 ms), the w2 zero-store of ~940 MB (~0.2 ms), act_and_mul 284-290 us -> ~90 us, count_and_sort + moe_align ~137 us -> ~60 us: ~0.8-1.1 ms of 14.3-22.3 ms, i.e. -5 to -8% on the first_turn shapes, ~+2-2.5% on the headline geomean"
  - shapes: "*load=turn_*"
    direction: neutral
    why: T <= 4096 path is byte-identical
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-25T014119Z
review_companions: null
hold_reason: "adversarial review: Hold: every part is already measured or is in flight. Part 1 is experts-triton-persistent-grid-first-turn verbatim (running now). Parts 2 and 3 ran on their own"
---

# first-turn-moe-dead-work-stack

## Change

Built on master (635cfee). Touches only the T > 4096 path (_experts_triton and the Triton kernel's launch); _experts_marlin (T <= 4096) is byte-identical. Three parts, all aimed at work the first_turn expert block does on rows that are not this GPU's: (1) PERSISTENT GRID, exactly as hypothesis experts-triton-persistent-grid-first-turn specifies (grid = min(NUM_SMS, static tiles); loop `for tile in range(pid, num_tiles, num_programs)` with num_tiles = cdiv(n_padded, BLOCK_T) * (N // BLOCK_W) read on device; master's GROUP_SIZE_T mapping on `tile`; if/else instead of return). (2) LOCAL-ROUTES ALIGN in _experts_triton only, exactly as hypothesis moe-align-local-routes-parallel-port specifies (copy muse_moe_w4a16's _align_parallel_kernel/_align_parallel, local_start read from data['local_start'] on device, local_count 32, block = cfg['BLOCK_T'] = 256): it replaces _local_expert_ids + moe_align_block_size on this path, so there are no -1 blocks below n_padded and w2 writes nothing for remote slots (ZERO_NONLOCAL_W2 then never fires; leave the flag as is). kernel_fn must pass topk_ids and data['local_start'] down so this path gets them; the Marlin path keeps master's remap and align. (3) SORTED SILU: replace silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) in _experts_triton with a new Triton kernel _silu_mul_sorted_kernel(c1, c2, sorted_ids, n_padded, num_slots, N=4096, BLOCK_R=16, BLOCK_N=512), grid (cdiv(sorted_ids.shape[0], BLOCK_R), N // BLOCK_N), num_warps 4: load n_padded; if pid0*BLOCK_R >= n_padded return; rows = pid0*BLOCK_R + arange; s = load(sorted_ids + rows, mask=rows < n_padded, other=num_slots); valid = s < num_slots (with part 2 every valid s is a local slot); cols = pid1*BLOCK_N + arange; g = c1[s, cols], u = c1[s, N + cols] (bf16), out = (g_f32 / (1 + exp(-g_f32))) * u_f32 rounded once to bf16, store c2[s, cols] under valid. It reads ~T local rows (16384 x 24 KB) instead of launching over all T*8 slots. prepare() gates, each one log line, any failure turns the whole stack off for the run (master's _experts_triton runs; log '[ft-stack] OFF: <reason>', else '[ft-stack] ON'): the align self-check vs moe_align_block_size per-expert route sets at T=8192 and 16384; the NaN-poison check (fused_post_expert_norm_weighted_sum with NaN in remote slots equals the zero-filled call); the silu kernel vs silu_and_mul on the local rows of a synthetic cache1 (max diff <= 1 bf16 ulp); _check_expert_gemms (now also exercising the persistent loop) and _check_experts_e2e adapted to the new align (Triton stack vs Marlin, rel L2 <= 2e-2).

## Expected

- `*load=first_turn*` faster: Removes, per first_turn shape: ~66K dead Triton CTAs (est. 0.3-0.6 ms), the w2 zero-store of ~940 MB (~0.2 ms), act_and_mul 284-290 us -> ~90 us, count_and_sort + moe_align ~137 us -> ~60 us: ~0.8-1.1 ms of 14.3-22.3 ms, i.e. -5 to -8% on the first_turn shapes, ~+2-2.5% on the headline geomean
- `*load=turn_*` neutral: T <= 4096 path is byte-identical

## Risk

Bold combined run of experts-triton-persistent-grid-first-turn, moe-align-local-routes-parallel-port (its first_turn half) and a sorted-row silu at first_turn; each part's effect is readable on its own kernel row (_moe_nvfp4_gemm_kernel, the align rows, act_and_mul vs _silu_mul_sorted_kernel). The silu part is the held silu-mul-triton-local-rows-marlin-turn mechanism folded into a larger change, as its hold asked, but at T > 4096 only (where act_and_mul is 285 us, not 14-53). Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) replaces the Triton GEMM at first_turn and plans to fuse silu in its stage 4; parts 2-3 compose with it, part 1 is moot if it lands. Risk high only because three parts must all pass their gates; each falls back to master.

## Adversarial review

Reviewed 2026-10-25T014119Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold: every part is already measured or is in flight. Part 1 is experts-triton-persistent-grid-first-turn verbatim (running now). Parts 2 and 3 ran on their own as align-local-parallel-first-turn, silu-mul-sorted-first-turn and stack-tonight-six-prefill (all flat). Two of the four claimed savings are already on master. Wait for the persistent-grid result, then re-mint only if it lands.

### Justification
The stated base (635cfee) is pre stack-tonight-five, and the expected list is stale against master a81d19c. (a) 'w2 zero-store of ~940 MB (~0.2 ms)' is gone: master's prepare logs '[w2 zero-store] OFF (weighted sum skips remote slots)' (jobs c4d14d62, 502f2eab). (b) 'count_and_sort + moe_align ~137 us' is replaced on master by moe_align_block_size(local_ids, 256, LOCAL_EXPERTS=32) (moe-align-32-experts-first-turn). (c) 'act_and_mul 284-290 -> ~90 us' is the sorted silu. It ran at first_turn as silu-mul-sorted-first-turn (job 9528eed5) and was flat (first_turn rows 0.979-0.990, within noise). (d) The local-routes align at first_turn ran as align-local-parallel-first-turn (c2f3e9d8, first_turn 0.977-0.996, flat) and again inside stack-tonight-six-prefill (5cb6e68a, flat). That leaves (e), the ~66K dead CTAs (est. 0.3-0.6 ms), as the only live lever. It is exactly the running experts-triton-persistent-grid-first-turn. On master the -1 tiles and the tiles past n_padded already exit early, so the lever is the ~1 us per dead CTA serialized at 1 CTA/SM, not wasted math. WEAKEST LINK: the -5 to -8% first_turn estimate sums four savings, and two of them no longer exist on master. The other two have already been measured as below the ~±1.5-6% first_turn noise (REWRITE_READ: ±6% on the Triton row).

### Confounders on the current kernel
- stack-tonight-five-prefill: it already contains the w2 zero-store OFF and the 32-expert align at first_turn, plus the Marlin-path local align and sorted silu. A stack built on 635cfee would re-add or conflict with these, and its expected savings double-count them.
- experts-hybrid-marlin-turn-triton-first-turn / experts-triton-w4a16-first-turn-blockt256: these set BLOCK_T 256 at 1 CTA/SM (140 KB smem), and that is what makes dead CTAs costly (serialized). The persistent-grid part is only worth anything because of this config, so a change of first_turn config changes its value.
- Running experts-triton-persistent-grid-first-turn (part 1 verbatim) and running experts-triton-turn4-persistent-local-align-control-no-persi (its _align_parallel swap in _experts_triton also covers T=16384, i.e. part 2 at first_turn): a stack result now could not be separated from either.

### Interference and A/B needs
No companion fixes this. The problem is not attribution inside one run: the run would re-measure parts already measured flat and duplicate a running check. Decision tree after experts-triton-persistent-grid-first-turn harvests. If it is a gain and lands: the only new question this stack can answer is whether the local align (no -1 tiles below n_padded) plus the sorted silu pay on top of the persistent loop, where each -1 tile becomes one in-loop iteration (an expert-id load plus a branch) instead of an exiting CTA. Re-mint that as a small restack with depends_on the persistent grid, base = new master, and expected limited to the -1 tile iterations and act_and_mul. It must drop the w2 zero-store and count_and_sort claims. Also read the turn_4 control's first_turn rows first, since they already measure the local align at T=16384 with the non-persistent grid. If it is flat or a loss: retire this stack. Parts 2 and 3 are measured flat, and 'identical code is never re-run'.

### Open questions
- Does SGLang's moe_align_block_size with num_experts=32 still emit -1 blocks for remote ids on master? The persistent-grid hypothesis's tile counts (~448 -1 tiles, EM from a 257-expert capacity) were computed for the old 256-expert align. The dead-CTA count at master is smaller (EM = numel + 33*255), so read the real tile counts from the persistent-grid log before re-minting.
- The stack's silu kernel omits master's local_ids>=0 guard (master's _silu_mul_sorted_kernel also loads local_ids). That guard is correct only if no -1 rows remain in sorted_ids, so any re-mint must keep master's version.

