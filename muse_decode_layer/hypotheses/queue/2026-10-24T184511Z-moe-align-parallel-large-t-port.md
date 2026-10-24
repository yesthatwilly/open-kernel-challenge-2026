---
problem: muse_decode_layer
slug: moe-align-parallel-large-t-port
created: 2026-10-24T184511Z
base: master
area: MoE align/dispatch (T>32)
change: >
  Port muse_moe_w4a16 align-parallel-per-expert-verify (patch
  muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check
  45eacf4ca2e849d2: SGLang align 10.6 / 11.4 us -> _align_parallel_kernel 4.0 / 6.9 us at bs=128 /
  256, verify_16 / verify_32 +3-4% there) to decode _experts for num_tokens > 32 only. Copy
  verbatim from the patch: ALIGN_PAR_CHUNK, ALIGN_PAR_WARPS, ALIGN_PAR_BLK, _align_parallel_kernel
  and _align_parallel. In _experts, when num_tokens > 32: replace the two master statements
  `local_ids = bank["expert_map"][topk_ids].contiguous()` and `sorted_ids, block_experts, n_padded
  = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` with `local_ids, (sorted_ids,
  block_experts, n_padded) = _align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)` (decode
  hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are
  0..31: local_start 0, count 32; the kernel writes local_ids itself, -1 for remote slots, exactly
  the expert_map semantics). block_size_m stays master pick with the cap (16 at T=128 and T=256).
  Everything downstream (cache allocation, both Marlin calls, silu_and_mul with
  expert_ids=local_ids, fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids) is
  unchanged. prepare(): compile one variant per block_size_m it can see above 32 tokens (8, 16) on
  dummy ids, then a self-check on synthetic skewed routings at T in 33, 40, 57, 128, 256, 257: per
  local expert, the set of routes in sorted_ids and its block count must equal
  moe_align_block_size(expert_map[ids], bsm, N_EXPERTS) and local_ids must be equal; ignore
  capacity and padding-count differences (LESSONS: never gate on a check that counts the capacity
  line, it always differs). On a real mismatch set state["align_par_ok"] = False and keep master;
  print "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". T <= 32 untouched.
expected:
  - shapes: "*load=verify_16 *"
    direction: faster
    why: "SGLang align (~10.6 us in moe at bs=128) plus the expert_map gather launch -> one ~4 us launch: -6..-8 us of 439 / 601 us, +1-1.5%"
  - shapes: "*load=verify_32 *"
    direction: faster
    why: "~11.4 us + gather -> ~6.9 us: -5..-6 us of 623 / 906 us, +0.6-0.9%"
  - shapes: "*load=decode_*"
    direction: flat
    why: T <= 32 untouched
  - shapes: "*load=verify_1 *"
    direction: flat
    why: T=8 untouched
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# moe-align-parallel-large-t-port

## Change

Port muse_moe_w4a16 align-parallel-per-expert-verify (patch muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check 45eacf4ca2e849d2: SGLang align 10.6 / 11.4 us -> _align_parallel_kernel 4.0 / 6.9 us at bs=128 / 256, verify_16 / verify_32 +3-4% there) to decode _experts for num_tokens > 32 only. Copy verbatim from the patch: ALIGN_PAR_CHUNK, ALIGN_PAR_WARPS, ALIGN_PAR_BLK, _align_parallel_kernel and _align_parallel. In _experts, when num_tokens > 32: replace the two master statements `local_ids = bank["expert_map"][topk_ids].contiguous()` and `sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` with `local_ids, (sorted_ids, block_experts, n_padded) = _align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)` (decode hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are 0..31: local_start 0, count 32; the kernel writes local_ids itself, -1 for remote slots, exactly the expert_map semantics). block_size_m stays master pick with the cap (16 at T=128 and T=256). Everything downstream (cache allocation, both Marlin calls, silu_and_mul with expert_ids=local_ids, fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids) is unchanged. prepare(): compile one variant per block_size_m it can see above 32 tokens (8, 16) on dummy ids, then a self-check on synthetic skewed routings at T in 33, 40, 57, 128, 256, 257: per local expert, the set of routes in sorted_ids and its block count must equal moe_align_block_size(expert_map[ids], bsm, N_EXPERTS) and local_ids must be equal; ignore capacity and padding-count differences (LESSONS: never gate on a check that counts the capacity line, it always differs). On a real mismatch set state["align_par_ok"] = False and keep master; print "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". T <= 32 untouched.

## Expected

- `*load=verify_16 *` faster: SGLang align (~10.6 us in moe at bs=128) plus the expert_map gather launch -> one ~4 us launch: -6..-8 us of 439 / 601 us, +1-1.5%
- `*load=verify_32 *` faster: ~11.4 us + gather -> ~6.9 us: -5..-6 us of 623 / 906 us, +0.6-0.9%
- `*load=decode_*` flat: T <= 32 untouched
- `*load=verify_1 *` flat: T=8 untouched

## Risk

A proven per-shape win in moe (measured on the align kernel row), ported unchanged; the headline effect (~0.3%) is under resolution, so judge on the align rows at verify_16/32. OVERLAP: moe-chain-cuda-swapab-expert-once-large-t (in judge) replaces the align at T>32 with its own route-list kernel when its path is ON; if it lands ON this is moot, if it falls back to Marlin this is the cheap T>32 fallback. Run it only after that harvest.
