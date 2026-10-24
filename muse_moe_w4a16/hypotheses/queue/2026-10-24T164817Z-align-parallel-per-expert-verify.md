---
problem: muse_moe_w4a16
slug: align-parallel-per-expert-verify
created: 2026-10-24T164817Z
base: master
area: MoE align/dispatch
change: >
  For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when
  expert_map is not None, replace master's else-branch (topk_ids.to(int32), the
  expert_map[local_ids] gather = at::native index_elementwise 3.2 us, the at::native
  unrolled_elementwise 2.8 us, and moe_align_block_size's moe_align_block_size_kernel +
  count_and_sort_expert_tokens_kernel: ~10.5 us of kernels and 4 graph nodes in master's
  bs=128/256 profile) with ONE launch of a new multi-program Triton kernel _align_parallel_kernel,
  grid (NUM_E + 1,) = 33 programs, num_warps 4. Every program loops over the n = M*8 flat route
  ids in chunks of CHUNK = 1024 (tl.arange(0, CHUNK), masked), maps each id to local = id -
  local_start if 0 <= local < local_count else -1 (as _align_small_kernel does), and accumulates
  per-expert counts for all 32 local experts (one-hot [CHUNK, 32] summed over axis 0). Program e <
  32 then computes padded = ceil(count/BSM)*BSM for all experts, its own exclusive seg_start
  (cumsum of padded, minus its own), and in a second pass over the chunks stores route index i at
  sorted_ids[seg_start + running_rank + in-chunk rank] for every route with local == e (in-chunk
  rank = tl.cumsum of the match mask minus 1; running_rank += the chunk's match count). It stores
  the sentinel n into its padding slots [seg_start+count, seg_start+padded) using tl.arange(0,
  PAD_BLOCK) with PAD_BLOCK = next_power_of_2(BSM), masked to padded-count (never tl.arange(0,
  BSM): BSM 48 occurs at M 116..127 and is what broke triton-map-align-verify-chunked), and writes
  block_experts[b] = e for b in [seg_start/BSM, (seg_start+padded)/BSM). Program 32 writes
  local_ids for all n routes (int32, same values as master), the sentinel n into every slot in
  [total, cap), -1 into block_experts[b] for b in [total/BSM, n_blocks), and n_padded = total.
  Capacities exactly as master's 32-expert moe_align_block_size allocates: cap = n + 32*(BSM-1),
  n_blocks = cdiv(cap, BSM). The store sets are disjoint across programs, so no atomics or
  ordering are needed. BSM is a constexpr; n, cap, n_blocks, local_start, local_count are
  do_not_specialize. The block_size_m pick is unchanged (two-smaller at >= 128). In prepare(),
  compile the kernel for BSM in (16, 32, 48, 64) on dummy ids and extend _align_self_check with T
  in (33, 57, 120, 128, 256, 257) using the same per-expert-set / block-count / sentinel / layout
  checks against moe_align_block_size(ref_local, bsm, 32). num_tokens <= 32 keeps
  _align_small_kernel; Marlin, silu_and_mul and post-norm are unchanged.
expected:
  - shapes: bs=128 *
    direction: faster
    why: 4 align-path launches (~10.5 us of kernels plus their graph nodes) become one ~3-4 us 33-CTA launch; the single-program chunked kernel (16.5 us) was already flat-to-better here (paired 1.403 vs 1.375), so the removed node overhead is real and a fast kernel should net ~7-12 us of 333 us
  - shapes: bs=256 *
    direction: faster
    why: same 4 launches (~11.5 us) replaced; the single-program chunked kernel took 29.2 us and lost only ~1%, so a 33-program kernel at ~4-5 us should net ~8-15 us of 470 us
  - shapes: bs=32 *
    direction: flat
    why: "control: _align_small path untouched"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# align-parallel-per-expert-verify

## Change

For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when expert_map is not None, replace master's else-branch (topk_ids.to(int32), the expert_map[local_ids] gather = at::native index_elementwise 3.2 us, the at::native unrolled_elementwise 2.8 us, and moe_align_block_size's moe_align_block_size_kernel + count_and_sort_expert_tokens_kernel: ~10.5 us of kernels and 4 graph nodes in master's bs=128/256 profile) with ONE launch of a new multi-program Triton kernel _align_parallel_kernel, grid (NUM_E + 1,) = 33 programs, num_warps 4. Every program loops over the n = M*8 flat route ids in chunks of CHUNK = 1024 (tl.arange(0, CHUNK), masked), maps each id to local = id - local_start if 0 <= local < local_count else -1 (as _align_small_kernel does), and accumulates per-expert counts for all 32 local experts (one-hot [CHUNK, 32] summed over axis 0). Program e < 32 then computes padded = ceil(count/BSM)*BSM for all experts, its own exclusive seg_start (cumsum of padded, minus its own), and in a second pass over the chunks stores route index i at sorted_ids[seg_start + running_rank + in-chunk rank] for every route with local == e (in-chunk rank = tl.cumsum of the match mask minus 1; running_rank += the chunk's match count). It stores the sentinel n into its padding slots [seg_start+count, seg_start+padded) using tl.arange(0, PAD_BLOCK) with PAD_BLOCK = next_power_of_2(BSM), masked to padded-count (never tl.arange(0, BSM): BSM 48 occurs at M 116..127 and is what broke triton-map-align-verify-chunked), and writes block_experts[b] = e for b in [seg_start/BSM, (seg_start+padded)/BSM). Program 32 writes local_ids for all n routes (int32, same values as master), the sentinel n into every slot in [total, cap), -1 into block_experts[b] for b in [total/BSM, n_blocks), and n_padded = total. Capacities exactly as master's 32-expert moe_align_block_size allocates: cap = n + 32*(BSM-1), n_blocks = cdiv(cap, BSM). The store sets are disjoint across programs, so no atomics or ordering are needed. BSM is a constexpr; n, cap, n_blocks, local_start, local_count are do_not_specialize. The block_size_m pick is unchanged (two-smaller at >= 128). In prepare(), compile the kernel for BSM in (16, 32, 48, 64) on dummy ids and extend _align_self_check with T in (33, 57, 120, 128, 256, 257) using the same per-expert-set / block-count / sentinel / layout checks against moe_align_block_size(ref_local, bsm, 32). num_tokens <= 32 keeps _align_small_kernel; Marlin, silu_and_mul and post-norm are unchanged.

## Expected

- `bs=128 *` faster: 4 align-path launches (~10.5 us of kernels plus their graph nodes) become one ~3-4 us 33-CTA launch; the single-program chunked kernel (16.5 us) was already flat-to-better here (paired 1.403 vs 1.375), so the removed node overhead is real and a fast kernel should net ~7-12 us of 333 us
- `bs=256 *` faster: same 4 launches (~11.5 us) replaced; the single-program chunked kernel took 29.2 us and lost only ~1%, so a 33-program kernel at ~4-5 us should net ~8-15 us of 470 us
- `bs=32 *` flat: control: _align_small path untouched
- `bs=1 *` flat: control: untouched

## Risk

Not a rerun of triton-map-align-verify-chunked(-pow2-fix) (neutral 1.311x): that kernel was ONE program streaming every route serially (16.5 us at bs=128, 29.2 us at bs=256, slower than the 4 kernels it replaced), yet both verify shapes held, so the launches it removed were worth roughly its extra kernel time. This design parallelises over experts so the kernel itself should be ~3-5 us and the node saving shows. Judge on the bs=128/256 paired ratios and on the new kernel row vs the four removed rows (top_kernels). Risks: each program re-reads all n ids (8 KB at bs=256, L2-resident after the first program) and builds a [1024, 32] one-hot per chunk; if the kernel exceeds ~6 us the gain vanishes. Must pass bs=33 (BSM 16), bs=257 (BSM 32) and skewed-128 tests; BSM 48 must never reach tl.arange.
