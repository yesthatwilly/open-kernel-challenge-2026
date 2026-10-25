---
problem: muse_moe_w4a16
slug: marlin-block-m-8-verify16
created: 2026-10-25T054545Z
base: master
area: MoE align/dispatch + block_size_m pick
change: >
  In _forward on master 9080a71, for 128 <= num_tokens <= 255 only (verify_16 bs=128 and the
  skewed bs=128 test), set block_size_m = 8 instead of master's two-smaller pick of 16; num_tokens
  >= 256 keeps 32 and T < 128 keeps SGLang's pick. Add a constant VERIFY16_BLOCK_SIZE_M = 8 and
  branch on the shape only. The same block size feeds _align_parallel (BSM=8, PAD_BLOCK=8) and the
  moe_block_size kwarg of both Marlin calls. prepare() must warm the _align_parallel BSM=8 variant
  (master warms one variant per block_size_m it can see above 32 tokens; add 8) and
  _align_self_check must cover (128, 8) against moe_align_block_size(local_ids, 8, 32) per expert
  route sets and block counts (capacity-only differences allowed, as master's check already logs).
  Mechanism: at bs=128 the measured trend is monotonic in block size: 48 -> 462 us (SGLang pick),
  32 -> 382 us (next-smaller), 16 -> 340-344 us (two-smaller / cap-16), so Marlin's per-m-block
  cost is still falling at 16; with ~32 routes per local expert on average, 8-row blocks pad less
  on the cold experts. Hard rules: CUDA-graph capturable; kernel_fn does no host sync
  (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except
  and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in
  registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes.
  The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may
  hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=128 *
    direction: faster
    why: continues the measured 48 -> 32 -> 16 trend (462 -> 382 -> 340 us); Marlin pair 306 us vs floor 101 us
  - shapes: bs=256 *
    direction: flat
    why: "control: keeps 32 (cap-16 at bs=256 was flat, 465 vs 468)"
  - shapes: bs=32 *
    direction: flat
    why: "control: T<128 untouched"
  - shapes: bs=1 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T054818Z
review_companions: null
---

# marlin-block-m-8-verify16

## Change

In _forward on master 9080a71, for 128 <= num_tokens <= 255 only (verify_16 bs=128 and the skewed bs=128 test), set block_size_m = 8 instead of master's two-smaller pick of 16; num_tokens >= 256 keeps 32 and T < 128 keeps SGLang's pick. Add a constant VERIFY16_BLOCK_SIZE_M = 8 and branch on the shape only. The same block size feeds _align_parallel (BSM=8, PAD_BLOCK=8) and the moe_block_size kwarg of both Marlin calls. prepare() must warm the _align_parallel BSM=8 variant (master warms one variant per block_size_m it can see above 32 tokens; add 8) and _align_self_check must cover (128, 8) against moe_align_block_size(local_ids, 8, 32) per expert route sets and block counts (capacity-only differences allowed, as master's check already logs). Mechanism: at bs=128 the measured trend is monotonic in block size: 48 -> 462 us (SGLang pick), 32 -> 382 us (next-smaller), 16 -> 340-344 us (two-smaller / cap-16), so Marlin's per-m-block cost is still falling at 16; with ~32 routes per local expert on average, 8-row blocks pad less on the cold experts. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=128 *` faster: continues the measured 48 -> 32 -> 16 trend (462 -> 382 -> 340 us); Marlin pair 306 us vs floor 101 us
- `bs=256 *` flat: control: keeps 32 (cap-16 at bs=256 was flat, 465 vs 468)
- `bs=32 *` flat: control: T<128 untouched
- `bs=1 *` flat: control

## Risk

Risk: at bs=32 a uniform 8 (vs 16) was flat-to-worse (215 vs 208), so the trend may stop at 16 for hot experts that now need 2x the m-blocks. Independent of the wgmma line: if wgmma lands at B128 this becomes moot there.

## Adversarial review

Reviewed 2026-10-25T054818Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean one-shape test of block_size_m 16 -> 8 at 128 <= T <= 255. bs=256/32/16/8/1 act as in-run controls, so a gain or a loss can be read without a companion. Mechanistically, though, it is more likely flat or slower than faster at bs=128, so expect a closing negative rather than a win.

### Justification
The bs=128 trend is real: SGLang's pick of 48 ran 462 us, 32 ran 382 us and 16 ran 340 us. master profile: verify_16 is 317.8 us, Marlin is 305.9 us (95.7%) and the floor is 101.3 us, so Marlin is the only lever at this shape. The weakest link is extrapolating that trend below 16. Earlier hypotheses tie the gains to Marlin's thread_m_blocks falling (48 -> 16 took it from 3 to 1). It cannot drop below 1. At 8, Marlin takes its m_block_size_8 path, which still issues 16-row MMAs with half the rows wasted, and each m-block still streams that expert's weight tile for its n-slice. So per-block cost falls only a little, while a hot expert gets twice as many m-blocks. Routings are skewed so that this GPU is the busiest, so hot experts with 60-128 routes carry most of the Marlin time. The ~3.5 padding rows saved per cold expert are worth much less than the extra weight reads on the hot ones. The direct evidence agrees: at bs=32 (about 8 routes per expert on average, where 8 should fit best) next-smaller (16 -> 8) went from 208.15 to 215.21 us, 3.4% slower. That shape had a lower share of hot-expert routes than bs=128. The expected directions are internally consistent: only bs=128 changes, and bs=256 keeps 32 (64 -> 32 two-smaller). One load in a six-load geomean moves the headline by at most about (1+x)^(1/6), so a 5% gain at bs=128 is roughly +0.8% headline, near the CI half-width of about 1.5%. Read it per shape (the verify_16 timed median against paired SGLang), not by headline.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: not a confound but the direct baseline. This change overrides its bs=128 pick (16 -> 8) and leaves bs=256 at its 32. Reverting it would just mean SGLang's 48, which is already measured.
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: bs=128 runs _align_parallel_kernel, which is about 7 us at bs=256. With BSM=8, n_blocks roughly doubles (cap = 1024 + 32*7 -> 156 blocks vs 1024 + 32*15 -> 94), so its block_experts loop and sentinel fill grow slightly and PAD_BLOCK becomes 8. That is a possible 1-2 us shift inside the same timed shape. It is too small to flip the sign of a Marlin change of ~10+ us, but it matters if the result is within about 1%. Check the _align_parallel_kernel row in the profile.
- w2-marlin-atomic-add-small-m: no interaction. It gates at T <= 32, and bs=128 keeps use_atomic_add=False / fp32 reduce.
- post-expert-norm-triton-small-m: no interaction. bs=128 keeps SGLang's fused_post_expert_norm_weighted_sum.
- sglang-marlin-path-parity: shared Marlin path and workspace (max_blocks_per_sm=4). With more m-blocks at 8, the lock workspace must cover the larger block count; Marlin sizes its locks per n-tile/SM, so this should be fine. The 5 tests (skewed bs=128) will show it.

### Interference and A/B needs
No companion is warranted. Only one shape's pick changes, and every other scored shape is byte-identical to master, so those rows are the in-run drift control. A gain at bs=128 can only come from the 16 -> 8 pick, plus a negligible _align_parallel BSM=8 shift that the profile separates. A loss is just as unambiguous: Marlin's per-block cost at 8 does not offset the doubled hot-expert blocks. An ab-revert of two-smaller would reproduce the measured 48 point, and an ab-tune to 32 is already measured (382 us), so both would duplicate known data. With moe at 5 checks left and 5 queued, a companion would buy nothing.

### Open questions
- Does the bs=128 profile show Marlin's w13 and w2 kernels separately? A loss confined to w2 (K=size_n, smaller weights per expert) vs w13 would say whether the doubled-block weight re-read is the cost.
- If this comes back flat or slower, the block-m axis at bs=128 is exhausted (48/32/16/8 all measured). Record it in FINDINGS as do-not-retry for uniform block 8 at T >= 128 and leave the shape to the wgmma/swapab rewrites. A per-expert two-class split (like marlin-two-class-block-m-decode32 at bs=32: 8 for cold experts, 16 for hot ones) is the only block-m variant left with a mechanism.
- Priority: given the bs=32 evidence and the thread_m_blocks floor, consider dispatching this after the higher-expected-value queued moe items. It is cheap and interpretable but has a low prior.

