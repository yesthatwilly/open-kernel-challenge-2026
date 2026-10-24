---
problem: muse_prefill_layer
slug: experts-hybrid-marlin-turn-triton-first-turn
created: 2026-10-24T124504Z
base: master
area: "w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above)"
change: >
  Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3
  experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in
  muse_prefill_layer/submission.py change; kernel_fn is byte-identical between parity and master
  and stays so. (1) Restore the parity Marlin path verbatim from sglang-pipeline-parity (commit
  6e616b8a05, muse_prefill_layer/submission.py): its imports (types, scalar_types from
  sgl_kernel.scalar_type, moe_wna16_marlin_gemm, marlin_make_workspace,
  prepare_moe_nvfp4_layer_for_marlin), BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64),
  _marlin_bank(weights) unchanged, and the parity _experts body renamed _experts_marlin
  (block_size_m loop breaking at num_tokens*TOP_K/LOCAL_EXPERTS/block_size_m < 0.9,
  moe_align_block_size at that block size, the two moe_wna16_marlin_gemm calls with the same
  common kwargs: mul_topk_weights=False, is_ep=True, float4_e2m1f, is_k_full=True,
  use_atomic_add=False, use_fp32_reduce=True, workspace max_blocks_per_sm=4, then silu_and_mul and
  fused_post_expert_norm_weighted_sum exactly as parity). (2) Rename master's _experts body to
  _experts_triton, unchanged (EXPERT_CONFIGS, _nvfp4_moe_gemm, ZERO_NONLOCAL_W2 as on master). (3)
  New _experts(x, topk_weights, local_ids, state) with MARLIN_MAX_TOKENS = 4096: if x.shape[0] <=
  MARLIN_MAX_TOKENS call _experts_marlin, else _experts_triton. The gate reads only the token
  count (a shape, equal to config tokens), never tensor values: turn_1 T=753, turn_4 T=3012 and
  every test (largest T=4096) go to Marlin; the four first_turn shapes (T=16384) go to Triton with
  the BLOCK_T 256 config. (4) prepare() builds both layouts: state["marlin"] =
  _marlin_bank(weights) beside state["bank"] = _expert_bank(weights), and still runs
  _check_expert_gemms on the Triton bank (leave EXPERT_CONFIGS intact so the two now-unreachable
  small-T configs are still compiled and checked; no runtime effect). Weights stay NVFP4 in both
  layouts, each kernel dequantizes in registers; no BF16 copy, no autotune, no host sync in
  kernel_fn, nothing derived from inputs cached. Memory cost of the second bank: Marlin-repacked
  codes 768 MiB (w13 512 + w2 256) plus permuted e4m3 scales 96 MiB plus the small workspace,
  about 0.86 GiB, on top of master's nibble-reordered codes (0.75 GiB), cos_sin (512 MiB) and the
  original weights; activations are unchanged (largest is cache13 at T=16384, 2 GiB, Triton path).
  Under 1 GiB extra against an H200's 141 GB, so headroom is not a concern; prepare time grows by
  SGLang's repack (seconds, within 300 s). Print torch.cuda.max_memory_allocated() at the end of
  prepare() so the run log shows the ~0.9 GiB growth.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "w13+w2 return from the Triton kernel (847 us sliding / 729 us full) to Marlin (~730 / 681 us in the parity profile): 50-120 us off a 1.08-1.43 ms shape, master 0.93-0.98x of SGLang back to ~1.0x"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "Triton 2275 / 2318 us vs Marlin ~2250 / 2200 us: 1-5% on the shape, back to parity with SGLang"
  - shapes: "*load=first_turn*"
    direction: neutral
    why: T=16384 > 4096 keeps stage 3 Triton BLOCK_T 256 (w13+w2 8.0-8.7 ms vs Marlin 10.2-12.0 ms) unchanged
risk: low
depends_on: experts-triton-w4a16-first-turn-blockt256
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: required
status: queued
---

# experts-hybrid-marlin-turn-triton-first-turn

## Change

Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3 experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in muse_prefill_layer/submission.py change; kernel_fn is byte-identical between parity and master and stays so. (1) Restore the parity Marlin path verbatim from sglang-pipeline-parity (commit 6e616b8a05, muse_prefill_layer/submission.py): its imports (types, scalar_types from sgl_kernel.scalar_type, moe_wna16_marlin_gemm, marlin_make_workspace, prepare_moe_nvfp4_layer_for_marlin), BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64), _marlin_bank(weights) unchanged, and the parity _experts body renamed _experts_marlin (block_size_m loop breaking at num_tokens*TOP_K/LOCAL_EXPERTS/block_size_m < 0.9, moe_align_block_size at that block size, the two moe_wna16_marlin_gemm calls with the same common kwargs: mul_topk_weights=False, is_ep=True, float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, workspace max_blocks_per_sm=4, then silu_and_mul and fused_post_expert_norm_weighted_sum exactly as parity). (2) Rename master's _experts body to _experts_triton, unchanged (EXPERT_CONFIGS, _nvfp4_moe_gemm, ZERO_NONLOCAL_W2 as on master). (3) New _experts(x, topk_weights, local_ids, state) with MARLIN_MAX_TOKENS = 4096: if x.shape[0] <= MARLIN_MAX_TOKENS call _experts_marlin, else _experts_triton. The gate reads only the token count (a shape, equal to config tokens), never tensor values: turn_1 T=753, turn_4 T=3012 and every test (largest T=4096) go to Marlin; the four first_turn shapes (T=16384) go to Triton with the BLOCK_T 256 config. (4) prepare() builds both layouts: state["marlin"] = _marlin_bank(weights) beside state["bank"] = _expert_bank(weights), and still runs _check_expert_gemms on the Triton bank (leave EXPERT_CONFIGS intact so the two now-unreachable small-T configs are still compiled and checked; no runtime effect). Weights stay NVFP4 in both layouts, each kernel dequantizes in registers; no BF16 copy, no autotune, no host sync in kernel_fn, nothing derived from inputs cached. Memory cost of the second bank: Marlin-repacked codes 768 MiB (w13 512 + w2 256) plus permuted e4m3 scales 96 MiB plus the small workspace, about 0.86 GiB, on top of master's nibble-reordered codes (0.75 GiB), cos_sin (512 MiB) and the original weights; activations are unchanged (largest is cache13 at T=16384, 2 GiB, Triton path). Under 1 GiB extra against an H200's 141 GB, so headroom is not a concern; prepare time grows by SGLang's repack (seconds, within 300 s). Print torch.cuda.max_memory_allocated() at the end of prepare() so the run log shows the ~0.9 GiB growth.

## Expected

- `*load=turn_1*` faster: w13+w2 return from the Triton kernel (847 us sliding / 729 us full) to Marlin (~730 / 681 us in the parity profile): 50-120 us off a 1.08-1.43 ms shape, master 0.93-0.98x of SGLang back to ~1.0x
- `*load=turn_4*` faster: Triton 2275 / 2318 us vs Marlin ~2250 / 2200 us: 1-5% on the shape, back to parity with SGLang
- `*load=first_turn*` neutral: T=16384 > 4096 keeps stage 3 Triton BLOCK_T 256 (w13+w2 8.0-8.7 ms vs Marlin 10.2-12.0 ms) unchanged

## Risk

Low risk: both halves have passed every test before (parity ran Marlin at all T, master runs Triton at all T); only the shape gate and holding both banks are new. Judge the stage on the replaced kernels per shape: moe_wna16_marlin at turn_1/turn_4 should match the parity profile within ~2%, the Triton kernel at first_turn should match stage 3 within ~1%. If turn shapes do not return to ~1.0x of SGLang, look for a confound outside the experts, not the gate. The 4096 bound comes from the measured crossover (Triton loses at T=3012, wins at T=16384); T between 4096 and 16384 is unscored and not tuned here. Expected overall: the turn_1 and turn_4 loads move from ~0.95x to ~1.0x, first_turn keeps its gain, so roughly 1.023x -> 1.04x.

## Milestones

Rewrite stage 4 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
