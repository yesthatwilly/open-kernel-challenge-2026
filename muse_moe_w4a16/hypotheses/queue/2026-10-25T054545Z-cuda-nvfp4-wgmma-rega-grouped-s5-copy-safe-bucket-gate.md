---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
created: 2026-10-25T054545Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  The shippable twin of cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: identical code (same s4
  base rebased onto master 9080a71, same per-call cuTensorMapEncodeTiled from the bsrc pointer
  inside wg_gemm, same removal of wg_maps/wg_mptr/wg_mptr_py and _wg_state_live, same deepcopy +
  tensor-clone three-state refill-replay self-check, same profiler LAUNCHED assertion), plus a
  per-T-bucket selector so the path can only ship where it wins. Buckets: B128 = 128<=T<=255
  (verify_16, skewed-128 test) and B256 = 256<=T<=512 (verify_32, bs=257 test). For each bucket,
  prepare times master's full Marlin path against the full wgmma path (gather + w13+silu + w2 +
  the shared align/combine) at T=128 and T=256 with the LESSONS-173 selector: both captured as
  CUDA graphs, a >=256 MB L2 flush before every replay, 41 interleaved A/B reps, an empty-graph
  baseline subtracted, under the judge-like skewed routing calibrated so prepare's Marlin pair is
  ~0.8x the judged rows (~245 us at T=128, ~370 us at T=256; print local route counts) plus one
  random routing. A bucket is enabled only if wgmma's median wins by >= 1.02x on BOTH routings;
  print both medians and the ratio, and assert the enabled side is the faster printed one.
  kernel_fn branches on T against two prepare-time booleans state['wg_on_128'] and
  state['wg_on_256'] (shape + flag only). Hard rules: CUDA-graph capturable; kernel_fn does no
  host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no
  try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4
  (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child
  processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static
  global may hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=128 *
    direction: faster
    why: only if B128 selector enables wgmma (>=1.02x on judge-like routing); otherwise flat (master)
  - shapes: bs=256 *
    direction: faster
    why: only if B256 selector enables wgmma; s4 prepare all-local ratio 0.55x is the most favourable bucket
  - shapes: bs=1 *
    direction: flat
    why: "control: T<128 is master"
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate

## Change

The shippable twin of cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: identical code (same s4 base rebased onto master 9080a71, same per-call cuTensorMapEncodeTiled from the bsrc pointer inside wg_gemm, same removal of wg_maps/wg_mptr/wg_mptr_py and _wg_state_live, same deepcopy + tensor-clone three-state refill-replay self-check, same profiler LAUNCHED assertion), plus a per-T-bucket selector so the path can only ship where it wins. Buckets: B128 = 128<=T<=255 (verify_16, skewed-128 test) and B256 = 256<=T<=512 (verify_32, bs=257 test). For each bucket, prepare times master's full Marlin path against the full wgmma path (gather + w13+silu + w2 + the shared align/combine) at T=128 and T=256 with the LESSONS-173 selector: both captured as CUDA graphs, a >=256 MB L2 flush before every replay, 41 interleaved A/B reps, an empty-graph baseline subtracted, under the judge-like skewed routing calibrated so prepare's Marlin pair is ~0.8x the judged rows (~245 us at T=128, ~370 us at T=256; print local route counts) plus one random routing. A bucket is enabled only if wgmma's median wins by >= 1.02x on BOTH routings; print both medians and the ratio, and assert the enabled side is the faster printed one. kernel_fn branches on T against two prepare-time booleans state['wg_on_128'] and state['wg_on_256'] (shape + flag only). Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=128 *` faster: only if B128 selector enables wgmma (>=1.02x on judge-like routing); otherwise flat (master)
- `bs=256 *` faster: only if B256 selector enables wgmma; s4 prepare all-local ratio 0.55x is the most favourable bucket
- `bs=1 *` flat: control: T<128 is master
- `bs=32 *` flat: control

## Risk

Dispatch in parallel with cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced (same round): the forced twin gives the per-shape kernel rows whatever the selector decides, this one is the land candidate. If both buckets come out enabled, the two runs are a replication of the same judged path, which is still useful given +-20% Marlin profile noise. Risk: the prepare routing still mis-forecasts the judge (LESSONS bf16 cuBLASLt pick); the 1.02x margin and two routings limit that.

## Milestones

Rewrite stage 5 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
