---
problem: muse_attention_decode
slug: splitkv-combine-wide-mlp
created: 2026-10-24T183848Z
base: master
area: split-KV combine
change: >
  Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight;
  grid (B, 16) and the math stay. Today each (request, head) block has 4 warps; warp w walks
  partials w, w+4, ... (~20 per warp at 79 partials), each iteration loading m, then l and a
  float2 of acc, so the loop is a chain of L2 round trips. New: 8 warps (256 threads,
  __launch_bounds__(256), red_m/red_l/red_acc sized [8], loop 1 strides by 256). Loop 2: warp w
  handles partials c = w + 8 * i; process them in groups of 4: first load the 4 m values and 4 l
  values and the 4 float2 acc values into registers (masked by c < n, m = -inf when out of range),
  then compute the 4 weights and accumulate, so up to 12 independent loads are in flight per lane.
  The final warp-0 merge sums 8 entries; the epilogue (v_scale, scaleless RMSNorm with eps inside
  the sqrt, sigmoid(gate), bf16 store) is unchanged. Host launch uses 256 threads. Rename the
  module muse_full_decode_splitkv_v2.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: "splitkv_combine_kernel costs 8.2 us at decode_16 and 8.4 us at decode_32 to merge ~5-10 MB of partials that sit in L2: it is latency-bound. 2x warps and 4-deep batched loads should bring it to ~3-4 us, -4% on the decode_16 shape, -2% on decode_32"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: these shapes do not run the combine
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T184058Z
review_companions: null
---

# splitkv-combine-wide-mlp

## Change

Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight; grid (B, 16) and the math stay. Today each (request, head) block has 4 warps; warp w walks partials w, w+4, ... (~20 per warp at 79 partials), each iteration loading m, then l and a float2 of acc, so the loop is a chain of L2 round trips. New: 8 warps (256 threads, __launch_bounds__(256), red_m/red_l/red_acc sized [8], loop 1 strides by 256). Loop 2: warp w handles partials c = w + 8 * i; process them in groups of 4: first load the 4 m values and 4 l values and the 4 float2 acc values into registers (masked by c < n, m = -inf when out of range), then compute the 4 weights and accumulate, so up to 12 independent loads are in flight per lane. The final warp-0 merge sums 8 entries; the epilogue (v_scale, scaleless RMSNorm with eps inside the sqrt, sigmoid(gate), bf16 store) is unchanged. Host launch uses 256 threads. Rename the module muse_full_decode_splitkv_v2.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: splitkv_combine_kernel costs 8.2 us at decode_16 and 8.4 us at decode_32 to merge ~5-10 MB of partials that sit in L2: it is latency-bound. 2x warps and 4-deep batched loads should bring it to ~3-4 us, -4% on the decode_16 shape, -2% on decode_32
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: these shapes do not run the combine

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs).

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: proceed.** A small, contained latency fix to splitkv_combine_kernel. It is plausible: 8.2 us at decode_16 vs 8.4 us at decode_32 for twice the partials points to latency, not bandwidth. The headline effect (~0.5%) will be inside the CI, so it must be judged on the profile's splitkv_combine_kernel row, and no companion is needed.

### Justification
Mechanism is plausible. FINDINGS puts splitkv_combine_kernel at 8.2 us (decode_16, 6.7% of 122.8 us) and 8.4 us (decode_32, 3.3% of 241.9 us). Twice the partials (16x79 vs 32x79 slots, ~5.2 vs ~10.4 MB of fp32 acc) cost almost the same time, so the kernel is bound by a chain of round trips, not bytes. At L2 or even HBM bandwidth the bytes take 0.5-3 us. The occupancy argument holds: 256 or 512 blocks x 4 warps leaves ~8-15 warps per SM on 132 SMs, and 8 warps doubles that. The shape directions are right. Full decode_1 is excluded by the 1 < batch gate, and sliding and verify never run the combine. Magnitude: a ~4 us saving is ~3.3% of the full decode_16 shape, but with weights (sliding x46, full x16) it is ~2.4% of the decode_16 load and ~1.2% of decode_32. That is ~0.6% on the 6-load geomean, well inside the >=1.5% CI. Weakest link: the 'chain of L2 round trips' premise is partly overstated. The existing loop already has `#pragma unroll 4` with __restrict__ pointers, and its loads depend only on c, so nvcc may already hoist several iterations' loads. The remaining critical path also has fixed serial round trips this change does not touch: seqlens[b], then the separate M-max pass, __syncthreads, the batched loop, __syncthreads, then the gate load from qkv and the store. So 8.2 -> 3-4 us is optimistic; 5-6 us is more likely. Two cheap additions would make the target more reachable: (1) issue the gate load (gp[0], gp[1]) at kernel start, before the M pass; (2) fold the M pass into loop 2 with online rescaling, so one pass over ws_ml is removed. A second caveat: the partials may not be L2-resident, because kernel A streams ~330-660 MB of KV through L2 with cp.async.cg while it writes them. That makes the 'in L2' premise uncertain, but it strengthens the case for more loads in flight.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: it defines the combine kernel and CHUNK=1024, which sets n~79 partials per request. The benefit scales with n, so any later change to the chunk size (queued splitkv-chunk2048-batch32) would shrink this gain. That is a tuning interaction, not an attribution confounder: on master's lineage the measured delta is cleanly this change.
- sliding-deep-splits-on-cuda-splitkv (current best): disjoint path (sliding _sliding_fused_kernel), no shared kernel or launch. Its only relevance is the noise floor: FINDINGS notes full decode_32 medians drifting ~6% between identical runs, which is larger than this hypothesis's expected shape-level gain.

### Interference and A/B needs
No companion is warranted. The change touches one kernel, splitkv_combine_kernel, which runs only on full decode at 1 < B <= 64. The profile reports that kernel's own time per shape, so both a gain and a loss can be read directly from the top_kernels row (8.2 / 8.4 us baseline) without the headline. No landed hypothesis shares this launch, so an ab-revert would revert the kernel itself. A control splitting '8 warps' from '4-deep batched loads' would attribute between sub-changes, but the land/drop decision does not depend on that. Note the overlap with in-flight and queued work: full-cuda-onelaunch-warpspec-bulk-gather (running) replaces this kernel outright. splitkv-pdl-attn-to-combine and fused-full-combine-normgate touch the same launch boundary or epilogue. Rebase or hold this one if the rewrite lands first.

### Open questions
- Has SASS for the current loop been checked? If `#pragma unroll 4` already batches the loads, most of the expected gain must come from the 8-warp occupancy increase alone.
- Are the partials actually L2-resident when the combine runs, given kernel A streams hundreds of MB of KV through L2 with cp.async.cg? (ncu lts hit rate on the combine would answer it.)
- Should the gate prefetch and a single-pass online-rescale merge (removing the separate M pass) be included? They attack the remaining fixed round trips and would make the 3-4 us target more reachable.
- If full-cuda-onelaunch-warpspec-bulk-gather lands and self-checks before this runs, this hypothesis is moot for its shapes; hold it then rather than spend a check.

