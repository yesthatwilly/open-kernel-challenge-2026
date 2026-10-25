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
review: required
status: queued
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
