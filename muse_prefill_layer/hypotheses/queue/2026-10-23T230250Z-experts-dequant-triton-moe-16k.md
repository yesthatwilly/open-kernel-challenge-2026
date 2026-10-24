---
problem: muse_prefill_layer
slug: experts-dequant-triton-moe-16k
created: 2026-10-23T230250Z
base: master
area: w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens)
change: >
  Shape-gated replacement of the two Marlin W4A16 GEMMs for tokens >= DEQUANT_MIN_TOKENS = 4096 (a
  module constant), i.e. only the four first-turn shapes (T=16384) and the q_len 4096 test; below
  the gate _experts stays exactly the parity Marlin path. prepare() keeps the raw NVFP4 tensors
  from weights (w13 uint8 [32, 8192, 2048], w13_sf e4m3 [32, 8192, 256], w13_gs fp32 [32], w2 [32,
  4096, 2048], w2_sf, w2_gs) contiguous beside the Marlin bank; no BF16 copy is stored in state.
  On the gated path, inside the timed call every call: (a) torch.empty two BF16 buffers w13_bf16
  [32, 8192, 4096] and w2_bf16 [32, 4096, 4096] and fill them with one Triton dequant kernel (grid
  over expert x row-block): per packed byte the low nibble is element 2i and the high nibble 2i+1
  (nvfp4.unpack_e2m1), magnitude from the E2M1 grid (0, .5, 1, 1.5, 2, 3, 4, 6) with bit 3 the
  sign, times sf[e, n, k//16] (e4m3 read as fp32) times gs[e], rounded once to bf16, the same
  arithmetic as nvfp4.dequantize; assert equality to nvfp4.dequantize on one expert row-slice in
  prepare() (untimed). (b) moe_align_block_size(local_ids, 128, N_EXPERTS) for BLOCK_SIZE_M=128
  (the parity path keeps its 64). (c) two calls of
  sglang.srt.layers.moe.fused_moe_triton.fused_moe.invoke_fused_moe_kernel exactly as
  fused_experts_impl uses them for plain BF16 weights: A=x [T, 4096] bf16, B=w13_bf16, C=cache1
  [T*8, 8192], mul_routed_weight=False, top_k=8, config {BLOCK_SIZE_M: 128, BLOCK_SIZE_N: 128,
  BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 8, num_stages: 4}, compute_type=tl.bfloat16, every
  quant flag False, no scales; then the parity silu_and_mul(cache1, cache2, expert_ids=local_ids,
  expert_step=1); then the second call with A=cache2, B=w2_bf16, C=cache3 [T*8, 4096], top_k=1.
  The kernel's off_experts == -1 guard skips the non-local blocks that Marlin skips via is_ep. (d)
  the parity fused_post_expert_norm_weighted_sum on cache3.view(T, 8, 4096). If
  invoke_fused_moe_kernel's keyword names differ at SGLang a55ab7d, adapt the names only, never
  the math.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: at 4096 rows per local expert Marlin re-streams each expert 4-bit weight set once per 64-row m-block (64 passes, dequant in the MMA loop); the BF16 path pays one 0.77 GB read + 3.2 GB write + 3.2 GB read (~1.5 ms) and then a tensor-core GEMM at 128-row tiles with L2-swizzled reuse on 1.65 TFLOP of work; expected several ms off 17-26 ms, most visible on ctx=0 and sliding where attention is small
  - shapes: "*load=turn_1*"
    direction: flat
    why: "below the gate: Marlin path unchanged (control)"
  - shapes: "*load=turn_4*"
    direction: flat
    why: "below the gate: Marlin path unchanged (control)"
risk: high
depends_on: sglang-pipeline-parity
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# experts-dequant-triton-moe-16k

## Change

Shape-gated replacement of the two Marlin W4A16 GEMMs for tokens >= DEQUANT_MIN_TOKENS = 4096 (a module constant), i.e. only the four first-turn shapes (T=16384) and the q_len 4096 test; below the gate _experts stays exactly the parity Marlin path. prepare() keeps the raw NVFP4 tensors from weights (w13 uint8 [32, 8192, 2048], w13_sf e4m3 [32, 8192, 256], w13_gs fp32 [32], w2 [32, 4096, 2048], w2_sf, w2_gs) contiguous beside the Marlin bank; no BF16 copy is stored in state. On the gated path, inside the timed call every call: (a) torch.empty two BF16 buffers w13_bf16 [32, 8192, 4096] and w2_bf16 [32, 4096, 4096] and fill them with one Triton dequant kernel (grid over expert x row-block): per packed byte the low nibble is element 2i and the high nibble 2i+1 (nvfp4.unpack_e2m1), magnitude from the E2M1 grid (0, .5, 1, 1.5, 2, 3, 4, 6) with bit 3 the sign, times sf[e, n, k//16] (e4m3 read as fp32) times gs[e], rounded once to bf16, the same arithmetic as nvfp4.dequantize; assert equality to nvfp4.dequantize on one expert row-slice in prepare() (untimed). (b) moe_align_block_size(local_ids, 128, N_EXPERTS) for BLOCK_SIZE_M=128 (the parity path keeps its 64). (c) two calls of sglang.srt.layers.moe.fused_moe_triton.fused_moe.invoke_fused_moe_kernel exactly as fused_experts_impl uses them for plain BF16 weights: A=x [T, 4096] bf16, B=w13_bf16, C=cache1 [T*8, 8192], mul_routed_weight=False, top_k=8, config {BLOCK_SIZE_M: 128, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 8, num_stages: 4}, compute_type=tl.bfloat16, every quant flag False, no scales; then the parity silu_and_mul(cache1, cache2, expert_ids=local_ids, expert_step=1); then the second call with A=cache2, B=w2_bf16, C=cache3 [T*8, 4096], top_k=1. The kernel's off_experts == -1 guard skips the non-local blocks that Marlin skips via is_ep. (d) the parity fused_post_expert_norm_weighted_sum on cache3.view(T, 8, 4096). If invoke_fused_moe_kernel's keyword names differ at SGLang a55ab7d, adapt the names only, never the math.

## Expected

- `*load=first_turn*` faster: at 4096 rows per local expert Marlin re-streams each expert 4-bit weight set once per 64-row m-block (64 passes, dequant in the MMA loop); the BF16 path pays one 0.77 GB read + 3.2 GB write + 3.2 GB read (~1.5 ms) and then a tensor-core GEMM at 128-row tiles with L2-swizzled reuse on 1.65 TFLOP of work; expected several ms off 17-26 ms, most visible on ctx=0 and sliding where attention is small
- `*load=turn_1*` flat: below the gate: Marlin path unchanged (control)
- `*load=turn_4*` flat: below the gate: Marlin path unchanged (control)

## Risk

High risk. Correctness: a nibble/scale layout error shows up as a failed test at q_len 4096 (that test exercises the gate), so it fails before timing; the assert in prepare() against nvfp4.dequantize catches it earlier. API: invoke_fused_moe_kernel's signature and its -1 expert guard at a55ab7d are assumed from SGLang's fused_experts_impl; a signature mismatch costs one failed check run. Memory: 3.2 GB + 1 GB BF16 scratch plus the 2 GB cache13 per call, graph-pooled, fine on 141 GB. Rules: state holds only NVFP4 (Marlin repack and raw); the BF16 copy is produced inside the timed call from NVFP4 and never persists, which TASK.md names as allowed (dequantizing inside your kernels is fine); activations stay BF16 (A operands and outputs bf16, fp32 accumulate) as in Marlin; no host sync, allocation sizes depend only on config. The result attributes to the expert GEMM engine alone; the dequant cost is a known fixed ~1.5 ms that a dependent hypothesis can hide.
