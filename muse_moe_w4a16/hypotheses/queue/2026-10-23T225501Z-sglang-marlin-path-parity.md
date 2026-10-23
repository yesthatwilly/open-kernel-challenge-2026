---
problem: muse_moe_w4a16
slug: sglang-marlin-path-parity
created: 2026-10-23T225501Z
base: master
area: MoE align/dispatch
change: >
  Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call,
  x @ w13.T over all tokens, masked weighted sum) with SGLang's Marlin pipeline from baseline.py:
  prepare() builds the Marlin bank with prepare_moe_nvfp4_layer_for_marlin over experts
  [local_start, local_start+local_count), the expert_map (-1 for remote experts),
  marlin_make_workspace(device, max_blocks_per_sm=4) and norm_w = (post_norm_w + 1) in bf16;
  kernel_fn runs local_ids = expert_map[topk_ids], the BLOCK_SIZE_M_CANDIDATES 0.9-ratio pick,
  moe_align_block_size, moe_wna16_marlin_gemm for w13 (top_k=8, size_m=num_tokens, size_n=2I),
  silu_and_mul(expert_ids=local_ids), moe_wna16_marlin_gemm for w2 (top_k=1, size_m=num_tokens*8)
  and fused_post_expert_norm_weighted_sum(weight_first=True, slot_expert_ids=local_ids), with
  is_ep=True, use_fp32_reduce=True, use_atomic_add=False, exactly as baseline.py. ONE structural
  change so this is not a verbatim port: moe_align_block_size is called with
  num_experts=bank['count'] (32, the local bank) instead of N_EXPERTS=256. The ids it receives are
  already mapped to [-1] U [0,32) by expert_map, so no id is >= 32 and the only out-of-range value
  is -1, which the baseline already relies on the align kernel to skip at num_experts=256.
  Effects: the align kernel's per-expert cumsum shrinks 256 -> 32 entries, sorted_ids shrinks from
  M*8 + 256*(block_size_m-1) to M*8 + 32*(block_size_m-1) entries (at bs=256, block 64: 18176 ->
  4064, which drops under SGLang's 4096 fuse_sorted_ids_padding threshold and removes the separate
  sorted_ids.fill_ launch), and Marlin's block_experts array shrinks accordingly. Attribution: the
  whole move from 0.006x to ~1.0x is the Marlin pipeline; anything above ~1.0x (expected a few us
  per shape, largest in relative terms at bs=1) is the 32-expert align. Fallback if the align
  wrapper refuses num_experts < 256 or a test fails on it: keep 256 and instead differ from
  baseline.py by hoisting the per-call block_size_m pick and the `common` kwargs into a bs-keyed
  table built in prepare() (host-only, zero GPU effect), and say so in the attempt file so the
  result reads as pure parity.
expected:
  - shapes: "*"
    direction: faster
    why: "starter dequantizes all 32 experts to BF16 every call (index/elementwise kernels are 80% of 37 ms); SGLang's Marlin W4A16 path reads NVFP4 once and only for routed blocks: ~0.006x -> ~1.0x everywhere"
  - shapes: bs=1 *
    direction: faster
    why: "parity plus the 32-expert align: at 77 us total, a shorter align cumsum and smaller sorted_ids are a visible fraction"
  - shapes: bs=256 *
    direction: faster
    why: "parity plus: sorted_ids 18176 -> 4064 entries drops under the 4096 fused-padding threshold, removing the sorted_ids.fill_ launch that bs=256 alone pays"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# sglang-marlin-path-parity

## Change

Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's Marlin pipeline from baseline.py: prepare() builds the Marlin bank with prepare_moe_nvfp4_layer_for_marlin over experts [local_start, local_start+local_count), the expert_map (-1 for remote experts), marlin_make_workspace(device, max_blocks_per_sm=4) and norm_w = (post_norm_w + 1) in bf16; kernel_fn runs local_ids = expert_map[topk_ids], the BLOCK_SIZE_M_CANDIDATES 0.9-ratio pick, moe_align_block_size, moe_wna16_marlin_gemm for w13 (top_k=8, size_m=num_tokens, size_n=2I), silu_and_mul(expert_ids=local_ids), moe_wna16_marlin_gemm for w2 (top_k=1, size_m=num_tokens*8) and fused_post_expert_norm_weighted_sum(weight_first=True, slot_expert_ids=local_ids), with is_ep=True, use_fp32_reduce=True, use_atomic_add=False, exactly as baseline.py. ONE structural change so this is not a verbatim port: moe_align_block_size is called with num_experts=bank['count'] (32, the local bank) instead of N_EXPERTS=256. The ids it receives are already mapped to [-1] U [0,32) by expert_map, so no id is >= 32 and the only out-of-range value is -1, which the baseline already relies on the align kernel to skip at num_experts=256. Effects: the align kernel's per-expert cumsum shrinks 256 -> 32 entries, sorted_ids shrinks from M*8 + 256*(block_size_m-1) to M*8 + 32*(block_size_m-1) entries (at bs=256, block 64: 18176 -> 4064, which drops under SGLang's 4096 fuse_sorted_ids_padding threshold and removes the separate sorted_ids.fill_ launch), and Marlin's block_experts array shrinks accordingly. Attribution: the whole move from 0.006x to ~1.0x is the Marlin pipeline; anything above ~1.0x (expected a few us per shape, largest in relative terms at bs=1) is the 32-expert align. Fallback if the align wrapper refuses num_experts < 256 or a test fails on it: keep 256 and instead differ from baseline.py by hoisting the per-call block_size_m pick and the `common` kwargs into a bs-keyed table built in prepare() (host-only, zero GPU effect), and say so in the attempt file so the result reads as pure parity.

## Expected

- `*` faster: starter dequantizes all 32 experts to BF16 every call (index/elementwise kernels are 80% of 37 ms); SGLang's Marlin W4A16 path reads NVFP4 once and only for routed blocks: ~0.006x -> ~1.0x everywhere
- `bs=1 *` faster: parity plus the 32-expert align: at 77 us total, a shorter align cumsum and smaller sorted_ids are a visible fraction
- `bs=256 *` faster: parity plus: sorted_ids 18176 -> 4064 entries drops under the 4096 fused-padding threshold, removing the sorted_ids.fill_ launch that bs=256 alone pays

## Risk

Parity base for every other muse_moe_w4a16 hypothesis. Risks: (1) sgl_kernel's moe_align_block_size must skip -1 ids at num_experts=32 exactly as it does at 256 (same code path, same -1 values; the five pre-timing tests, incl. bs=1 and dist=skewed, catch any miscount); (2) the SGLang wrapper may have a num_experts-dependent branch (older versions switched kernels at >=224 experts) so the 32 path could be a different, possibly slower align kernel: then the result is still parity +/- a few us and the attempt file must record which branch ran; (3) correctness is Marlin's own (BF16 activations, fp32 reduce), the same the judge's SGLang side passes. Rules: NVFP4 stays NVFP4 (Marlin repack is lossless), activations bf16 where SGLang keeps them, nothing derived from data is cached, no host syncs (block_size_m branches on num_tokens, a shape). Commit-step dedup: the num_experts change alters the stripped AST, so this does not collide with baseline.py.
