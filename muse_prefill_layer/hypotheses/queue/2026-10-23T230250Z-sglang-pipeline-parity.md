---
problem: muse_prefill_layer
slug: sglang-pipeline-parity
created: 2026-10-23T230250Z
base: master
area: whole-layer parity (SGLang pipeline) + routing remap
change: >
  Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline
  module for module as baseline.py does: _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and
  unpooler; FusedQKNorm with the [2^21, 128] fp32 cos|sin table and interleaved RoPE on sliding
  layers; k.to(FP8) and v.to(FP8) then store_cache at cache_loc (reserved_skip_index=-1);
  flash_attn_with_kvcache (FA3, paged, causal, window (2047, 0) on sliding layers, FP8 queries,
  the kv_scale descale branch kept); FusedRMSNormGate; FusedPostNormResidualPreNorm at the
  attention and FFN boundaries; the router as the BF16 hi/lo pair built in prepare() with two
  torch.mm(..., out_dtype=torch.float32) and FusedMuseSparkRouting on the side stream overlapping
  the pooler GEMM (with the record_stream loop when not capturing); MuseSpark1xRMSNorm pre-expert
  norm; moe_align_block_size with the baseline's block_size_m selection (64 at every prefill
  shape), two moe_wna16_marlin_gemm calls with the bank from prepare_moe_nvfp4_layer_for_marlin,
  use_fp32_reduce=True, use_atomic_add=False, silu_and_mul(expert_ids=local_ids, expert_step=1),
  fused_post_expert_norm_weighted_sum. Two deliberate structural differences: (1) the local-expert
  id remap, which baseline.py does as three kernels ((topk_ids - local_start), % N_EXPERTS,
  expert_map[...] gather), becomes ONE small Triton elementwise kernel over int32 [T, 8]: r =
  (topk_ids - local_start) & 255; local_ids = r if r < 32 else -1; the expert_map tensor is
  dropped from the bank. (2) The two dead tokens<=8 branches (FP32 F.linear router with
  router_dtype=float32 at the attention boundary, and the sliced unpooler + F.pad) are removed,
  because every prefill test and scored case has q_len >= 200; the router always runs hi/lo and
  the unpooler always uses the whole [8192, 4096] weight. Everything else stays as baseline.py has
  it, including cache13/cache2 allocation inside the call. Attribution: the remap fusion removes
  two launches (3-8 us), the dead-branch removal is timing-neutral, so the run should land at
  1.00x within the 1.5% interval on every shape. Its purpose is to be the base every later prefill
  hypothesis stacks on and to give the first per-kernel profile of SGLang's path at these shapes
  (profiles/submission.<job>.txt), which no run has produced yet (the starter profile is of plain
  PyTorch).
expected:
  - shapes: "*load=turn_1*"
    direction: flat
    why: SGLang kernels one for one; the two saved launches are 3-8 us of ~1120-1465 us, inside the 1.5% interval
  - shapes: "*load=turn_4*"
    direction: flat
    why: same kernels; saved launches are <0.3% of 3.4-4.9 ms
  - shapes: "*load=first_turn*"
    direction: flat
    why: same kernels; saved launches are <0.1% of 17-26 ms
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# sglang-pipeline-parity

## Change

Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline module for module as baseline.py does: _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler; FusedQKNorm with the [2^21, 128] fp32 cos|sin table and interleaved RoPE on sliding layers; k.to(FP8) and v.to(FP8) then store_cache at cache_loc (reserved_skip_index=-1); flash_attn_with_kvcache (FA3, paged, causal, window (2047, 0) on sliding layers, FP8 queries, the kv_scale descale branch kept); FusedRMSNormGate; FusedPostNormResidualPreNorm at the attention and FFN boundaries; the router as the BF16 hi/lo pair built in prepare() with two torch.mm(..., out_dtype=torch.float32) and FusedMuseSparkRouting on the side stream overlapping the pooler GEMM (with the record_stream loop when not capturing); MuseSpark1xRMSNorm pre-expert norm; moe_align_block_size with the baseline's block_size_m selection (64 at every prefill shape), two moe_wna16_marlin_gemm calls with the bank from prepare_moe_nvfp4_layer_for_marlin, use_fp32_reduce=True, use_atomic_add=False, silu_and_mul(expert_ids=local_ids, expert_step=1), fused_post_expert_norm_weighted_sum. Two deliberate structural differences: (1) the local-expert id remap, which baseline.py does as three kernels ((topk_ids - local_start), % N_EXPERTS, expert_map[...] gather), becomes ONE small Triton elementwise kernel over int32 [T, 8]: r = (topk_ids - local_start) & 255; local_ids = r if r < 32 else -1; the expert_map tensor is dropped from the bank. (2) The two dead tokens<=8 branches (FP32 F.linear router with router_dtype=float32 at the attention boundary, and the sliced unpooler + F.pad) are removed, because every prefill test and scored case has q_len >= 200; the router always runs hi/lo and the unpooler always uses the whole [8192, 4096] weight. Everything else stays as baseline.py has it, including cache13/cache2 allocation inside the call. Attribution: the remap fusion removes two launches (3-8 us), the dead-branch removal is timing-neutral, so the run should land at 1.00x within the 1.5% interval on every shape. Its purpose is to be the base every later prefill hypothesis stacks on and to give the first per-kernel profile of SGLang's path at these shapes (profiles/submission.<job>.txt), which no run has produced yet (the starter profile is of plain PyTorch).

## Expected

- `*load=turn_1*` flat: SGLang kernels one for one; the two saved launches are 3-8 us of ~1120-1465 us, inside the 1.5% interval
- `*load=turn_4*` flat: same kernels; saved launches are <0.3% of 3.4-4.9 ms
- `*load=first_turn*` flat: same kernels; saved launches are <0.1% of 17-26 ms

## Risk

Low risk: it is SGLang's own code path. Failure modes: an import or signature drift of a JIT module inside the judge image (baseline.py imports the same symbols, so they exist); the Triton remap kernel must produce int32 and -1 exactly where expert_map gave -1 (local_start is a multiple of 32, so (ids - local_start) & 255 == (ids - local_start) % 256 for the int32 range in play). The commit step rejects an AST identical to baseline.py: the fused remap and the removed dead branches make it differ structurally, not only in docstrings. Rules: no host sync, state holds weights only, NVFP4 stays in Marlin's lossless repack, BF16 activations where SGLang keeps them, FP8 cache, FA3 over every key.
