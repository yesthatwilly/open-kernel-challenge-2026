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
review: proceed
status: queued
review_at: 2026-10-24T165146Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T165146Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, rules-clean redesign of the chunked align. It parallelises over experts and removes the serial single-CTA bottleneck that made _align_chunked_kernel 16.5/29.2 us. The realistic gain is 1.5-2.5% per verify shape and about 0.5% on the headline, so the run will almost certainly be classified as neutral. Judge it on the new kernel row against the four removed rows in top_kernels, and fix the [1024,32] one-hot register blow-up before building.

### Justification
Mechanism: at bs=128/256, master's else-branch runs at::native index_elementwise (3.2/3.3 us), the align wrapper's unrolled_elementwise fill (~2.8 us), moe_align_block_size_kernel (~2.7 us) and count_and_sort (~1.8-2.6 us). That is about 10.5/11.5 us against 333/470 us of shape time. Marlin is 94-95% of each verify shape (316/490 us vs floors of 101/109 us), so this area is capped at ~3.1%/~2.4% per shape even for a free kernel. With a 3-5 us kernel and ~1 us of saved graph-node cost (LESSONS: ~0.3 us per single Triton launch node), expect ~6-8 us, i.e. ~2% at bs=128 and ~1.5% at bs=256. Over the 6 equally weighted loads that is ~0.5-0.6% on the geomean, inside the +-1.5% CI and the 1.5-2.5% per-shape drift. The expected shapes and directions are right: faster at bs=128/256, flat at bs<=32, where _align_small is byte-identical and serves as the in-run drift control. The design parallelism is plausible. Each of the 33 programs reads only 2-8 KB of ids, and pass 2 needs only a 1-D match mask and cumsum, unlike the chunked kernel's [CHUNK,32] cumsum. Weakest link 1 (the argument): the claim that 'the removed node overhead is real' rests on the chunked run's paired ratios, bs=128 1.403 vs 1.375 (+2.0%) and bs=256 1.301 vs 1.314 (-1.0%). Both are inside drift, and the Marlin row itself moved 316->337 and 490->462 us between those runs. So the hidden ~6 us node saving the expected section leans on is unproven; budget ~1 us. Weakest link 2 (the build): CHUNK=1024 with num_warps=4 makes the [1024,32] int32 one-hot 256 live values per thread per temporary, 4x the chunked kernel's 64. That will likely spill and could push the kernel past the ~6 us break-even. Use CHUNK=512 with num_warps=8, or CHUNK=256 with num_warps=4 (64 per thread), or tl.histogram over remote-remapped ids if this Triton version supports a mask. Also: each program's own block_experts range can span up to cdiv(M, BSM) blocks (8 at bs=128 BSM 16 and at bs=256 BSM 32; more at skewed-128), so it needs a loop or a power-of-2 arange sized for that bound. The spec leaves this unstated. Extract seg_start[e] and count[e] with a where+sum over the expert axis, and cast the bool match mask to int32 before tl.cumsum.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: sets BSM=16 at bs=128 and 32 at bs=256/257. That fixes cap, n_blocks, the padding width and each expert's block count, so it shifts the new kernel's own time slightly (well under 1 us). It does not change what Marlin receives, so it does not confound attribution of the removed align rows. Without it, BSM would be 48/64, and PAD_BLOCK=64 would matter more.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: same area and same design family, but gated at <=32 tokens. Those shapes are this run's byte-identical control, not a confounder. Latent hazard: _align_small_kernel still uses tl.arange(0, BSM) unpadded, which is safe only because its BSM is <=16.
- post-expert-norm-triton-small-m: no timing confound at the target shapes. bs>=32 still calls fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids, and silu_and_mul uses expert_ids=local_ids. Program 32's local_ids must be exactly -1 for remote routes and int32 contiguous, as on master. This is a correctness dependency, not an attribution one.
- sglang-marlin-path-parity: provides the base pipeline and the 32-expert capacity contract (cap = n + 32*(BSM-1), n_blocks = cdiv(cap, BSM), sentinel n). It does not interact beyond the sizes the kernel must reproduce.

### Interference and A/B needs
No companion is warranted. Nothing landed sits between the change and its measurement. The four removed rows and the one added row are directly visible in top_kernels, and Triton kernel times reproduce within ~1% between runs, so the kernel-level comparison is interpretable from this single run. The headline and per-shape ratios are dominated by Marlin drift, which no A/B companion fixes. An ab-revert of block-m-two-smaller would change only BSM and the padding cost and could not flip the reading. A CHUNK/num_warps ab-tune is a follow-up to choose from this run's kernel row, not a disambiguator. Decision rule for the harvest: merge by hand if the log shows the self-check with 0 mismatches for T in {33..257}, _align_parallel_kernel at or below (sum of removed rows - ~2 us) at both verify shapes, and the bs=128/256 paired ratios at or above master's, even if classify says neutral. If the kernel row exceeds ~8 us at bs=256, gate it to bs<=128 or shrink CHUNK in a follow-up.

### Open questions
- The new kernel's own time at bs=256 (n=2048, 2 chunks x 2 passes per program at CHUNK=1024). Check the compile log or register report for spills at [1024,32] int32 with 4 warps. If the build keeps CHUNK=1024, consider num_warps=8.
- skewed-128: the busiest expert can receive up to M routes (128, so 8 blocks at BSM 16). Confirm the per-program block_experts and route-store loops are bounded by a runtime loop, not a fixed small arange.
- Does this image's Triton support tl.histogram with a mask? It would replace the one-hot entirely in pass 1 (map remote routes to bin 32 of a 64-bin histogram if there is no mask).
- Prepare-time compile: four BSM variants plus 6 more self-check T values. Expected seconds, well inside 300 s. Confirm every BSM variant (16/32/48/64) compiled before trusting the timings, given the chunked kernel's arange history.
- With moe at 2 checks left today, the expected classify outcome is neutral (about 0.5% headline). Is the second check better kept for a follow-up tune of this kernel than for a held item?

