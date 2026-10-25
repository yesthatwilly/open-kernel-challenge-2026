---
problem: muse_prefill_layer
slug: experts-triton-first-turn-per-gemm-config-pick
created: 2026-10-25T042410Z
base: master
area: "Triton NVFP4 engine at T > 4096 (first_turn): separate tile config for w13 and w2, picked by prepare timing"
change: >
  Built on master (a81d19c). Today both _nvfp4_moe_gemm calls in _experts_triton use the single T
  > 4096 EXPERT_CONFIGS entry (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4, 8 warps, 2 stages),
  although w13 is N=8192 and w2 is N=4096 (half the weight tiles, so half as many CTAs per token
  block and a different L2 reuse pattern), and that config was only ever measured on w13+w2
  together. Change only the config choice, no kernel code: in prepare(), for w13 and for w2
  separately, compile and time a fixed candidate list on a synthetic T=16384 soft routing through
  master's own align (moe_align_block_size over the 32 local experts) and master's
  zero_nonlocal_w2 flag: (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4/8/16, 8 warps, 2 stages),
  (BLOCK_T 256, BLOCK_W 64, GROUP_SIZE_T 8, 4 warps, 3 stages), (BLOCK_T 128, BLOCK_W 128,
  GROUP_SIZE_T 8, 8 warps, 3 stages). Timing: CUDA-graph replays with a cold-L2 flush, 7
  interleaved alternations against master's config; reject any candidate with n_spills > 4 or a
  rel L2 vs master's output above 1e-3. Pick a candidate only at <= 0.97x master's time for that
  GEMM; print the table and PATH CHOSEN per GEMM, asserting the pick is the fastest printed.
  kernel_fn passes state['cfg_w13'] / state['cfg_w2'] (a shape-only branch); the align still pads
  to the w13 BLOCK_T, so a w2 candidate with a different BLOCK_T is only allowed if it divides it
  (128 into 256: two w2 token tiles per aligned block, the sorted_ids layout is unchanged). T <=
  4096 untouched.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel 8.1-9.2 ms -> -3 to -8% if a pick wins (w2 is the likelier mover: 1580 us of the 4640 us measured in the persistent A/B); first_turn shapes -2 to -5%"
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: Marlin path
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: Marlin path
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-25T043311Z
review_companions: null
hold_reason: "adversarial review: Weak mechanism and a candidate list made mostly of known losers or configs that will not compile; the likely outcome is PATH CHOSEN = master for both GEMMs, i.e"
outcome: expired
verdict_detail: "retired: engine runs on no scored shape (rega replaced Triton/Marlin/ssdq; prefill DIAGNOSIS 2026-10-25)"
expired_at: 2026-10-25T211931Z
---

# experts-triton-first-turn-per-gemm-config-pick

## Change

Built on master (a81d19c). Today both _nvfp4_moe_gemm calls in _experts_triton use the single T > 4096 EXPERT_CONFIGS entry (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4, 8 warps, 2 stages), although w13 is N=8192 and w2 is N=4096 (half the weight tiles, so half as many CTAs per token block and a different L2 reuse pattern), and that config was only ever measured on w13+w2 together. Change only the config choice, no kernel code: in prepare(), for w13 and for w2 separately, compile and time a fixed candidate list on a synthetic T=16384 soft routing through master's own align (moe_align_block_size over the 32 local experts) and master's zero_nonlocal_w2 flag: (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4/8/16, 8 warps, 2 stages), (BLOCK_T 256, BLOCK_W 64, GROUP_SIZE_T 8, 4 warps, 3 stages), (BLOCK_T 128, BLOCK_W 128, GROUP_SIZE_T 8, 8 warps, 3 stages). Timing: CUDA-graph replays with a cold-L2 flush, 7 interleaved alternations against master's config; reject any candidate with n_spills > 4 or a rel L2 vs master's output above 1e-3. Pick a candidate only at <= 0.97x master's time for that GEMM; print the table and PATH CHOSEN per GEMM, asserting the pick is the fastest printed. kernel_fn passes state['cfg_w13'] / state['cfg_w2'] (a shape-only branch); the align still pads to the w13 BLOCK_T, so a w2 candidate with a different BLOCK_T is only allowed if it divides it (128 into 256: two w2 token tiles per aligned block, the sorted_ids layout is unchanged). T <= 4096 untouched.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel 8.1-9.2 ms -> -3 to -8% if a pick wins (w2 is the likelier mover: 1580 us of the 4640 us measured in the persistent A/B); first_turn shapes -2 to -5%
- `*load=turn_1*` unchanged: Marlin path
- `*load=turn_4*` unchanged: Marlin path

## Risk

Kill: PATH CHOSEN keeps master's config for both GEMMs (then the shipped code is master and the run is a re-measure; read the table only). Prepare timings undercount the judge by 20-25% (LESSONS), so rank, do not predict. Does not touch the K-loop (the running unmasked-loads hypothesis does).

## Adversarial review

Reviewed 2026-10-25T043311Z by an adversarial subagent (model opus).

**Verdict: hold.** Weak mechanism and a candidate list made mostly of known losers or configs that will not compile; the likely outcome is PATH CHOSEN = master for both GEMMs, i.e. a check that re-measures master. Fold it into the next first_turn Triton check as a prepare-only table instead of spending a check on it.

### Justification
Profile: first_turn _moe_nvfp4_gemm_kernel 8.08-9.17 ms, w13 3059 / w2 1580 us in the persistent A/B. Knobs: (a) GROUP_SIZE_T 4/8/16 - with ~512 routes per expert there are only ~2 token tiles of 256 per expert, so the weight tile of an expert is reused by at most 2 adjacent token tiles regardless of group size; groups >2 only change which experts share the L2 window, a ~1% effect. (b) BLOCK_T 256/BLOCK_W 64/4 warps/3 stages - the master comment says BLOCK_T 256 leaves room for 2 smem stages not 3 (two [64,256] bf16 activation slabs = 64 KB/stage), and a [64,256] fp32 accumulator over 128 threads is 128 regs + dequant temporaries, so it either fails to compile (OutOfResources) or spills >4 and is rejected. (c) BLOCK_T 128/BLOCK_W 128/3 stages is effectively stage 2's first_turn config, measured 13-18% slower than BLOCK_T 256 (REWRITE_READ #5). REWRITE_READ also concluded the Triton tuning line is done ('Further gains on this kernel need the CUDA wgmma engine') and the persistent-grid scheduling A/B lost 3-5%. Weakest link: no candidate has a mechanism that plausibly clears the 0.97x pick threshold, and even a 5% w2 win is ~80 us on a 14-22 ms shape (<0.6%), unreadable at the shape level.

### Confounders on the current kernel
- experts-triton-w4a16-first-turn-blockt256: established the current single config; candidate (c) re-tests its predecessor.
- In flight experts-triton-first-turn-unmasked-loads-l2-hints (aeda8b5a523642b1) edits the same K-loop and adds L2 eviction hints, which directly change what GROUP_SIZE_T and num_stages trade off; a config ranking taken on master's K-loop may invert on that one.

### Interference and A/B needs
If revived, no companion is needed (prepare A/B per GEMM is self-attributing), but it should be built on top of the unmasked-loads result, not beside it.

### Open questions
- If revived after unmasked-loads is harvested: replace the list with configs that can win - w2-only GROUP_SIZE_T 1/2 (match the ~2 token tiles per expert), num_warps 4 vs 8 at BLOCK_T 256/BLOCK_W 128/2 stages, and catch compile OutOfResources as a rejection; drive the synthetic routing with the judge's soft distribution (prepare totals 4.6 ms vs 8.1 ms judged suggest the synthetic case is not representative).

