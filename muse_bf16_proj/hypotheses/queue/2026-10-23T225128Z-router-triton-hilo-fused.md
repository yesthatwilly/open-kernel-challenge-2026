---
problem: muse_bf16_proj
slug: router-triton-hilo-fused
created: 2026-10-23T225128Z
base: master
area: router GEMM
change: >
  Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two
  torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch at every m. prepare() keeps
  the hi/lo split but stores router_hi and router_lo as [256, 8192] bf16 row-major (N x K, not
  transposed). The kernel: grid = (N/BLOCK_N, SPLIT_K); each program loads an x tile [BLOCK_M,
  BLOCK_K] of bf16 (BLOCK_M=16 for m<=16 with masked rows, else 32/64/128 chosen from m at launch;
  M is looped in BLOCK_M chunks for m=256), loads the hi and lo weight tiles [BLOCK_N, BLOCK_K]
  for its K-slice, and accumulates acc += tl.dot(x, hi.T) + tl.dot(x, lo.T) in fp32 over its K
  range (BLOCK_K=128, num_stages=3); with SPLIT_K>1 it tl.atomic_add's the fp32 partial into an
  output that kernel_fn zeroes with torch.zeros((m, 256), fp32) just before the launch (SPLIT_K=8
  and BLOCK_N=64 for m<=32 gives 32 programs x 8 = 256 CTAs; SPLIT_K=1 with a plain store for
  m>=128). The output is the fp32 [m, 256] tensor, dtype and shape as before. Kernels moved: at
  m<=8 the dot_kernel + x.float() cast + reduce_1Block (m=1) or the two xmma f32 SIMT GEMMs + cast
  (m=8) become one launch (plus one memset); at m>8 the 2 nvjet + 2 splitKreduce + 1 add become
  one launch (plus one memset). The weight read stays 8.4 MB (hi + lo bf16) which at 4.8 TB/s is
  the 1.8 us floor.
expected:
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: "m=8: 26.6 us fp32 SIMT path becomes one tensor-core kernel over 8.4 MB; target under 8 us"
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: "m=1: 3 launches (gemv dot + cast + reduce, 10.4 us) become 1 kernel + memset"
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: 5 launches (2 nvjet splitK + 2 splitKreduce + add, 14.8 us) become 1 kernel + memset
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same 5-to-1 launch reduction at m=32
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: "m=128: same, SPLIT_K=1, 2x 128x256x8192 bf16 dots are ~1 us of tensor-core work"
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: "m=256: 17.2 us today; one kernel reading 8.4 MB weights + 4 MB x"
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: bf16 layers untouched; control
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-23T230544Z
review_companions: [router-triton-hilo-fused-control-splitk-all-m]
---

# router-triton-hilo-fused

## Change

Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch at every m. prepare() keeps the hi/lo split but stores router_hi and router_lo as [256, 8192] bf16 row-major (N x K, not transposed). The kernel: grid = (N/BLOCK_N, SPLIT_K); each program loads an x tile [BLOCK_M, BLOCK_K] of bf16 (BLOCK_M=16 for m<=16 with masked rows, else 32/64/128 chosen from m at launch; M is looped in BLOCK_M chunks for m=256), loads the hi and lo weight tiles [BLOCK_N, BLOCK_K] for its K-slice, and accumulates acc += tl.dot(x, hi.T) + tl.dot(x, lo.T) in fp32 over its K range (BLOCK_K=128, num_stages=3); with SPLIT_K>1 it tl.atomic_add's the fp32 partial into an output that kernel_fn zeroes with torch.zeros((m, 256), fp32) just before the launch (SPLIT_K=8 and BLOCK_N=64 for m<=32 gives 32 programs x 8 = 256 CTAs; SPLIT_K=1 with a plain store for m>=128). The output is the fp32 [m, 256] tensor, dtype and shape as before. Kernels moved: at m<=8 the dot_kernel + x.float() cast + reduce_1Block (m=1) or the two xmma f32 SIMT GEMMs + cast (m=8) become one launch (plus one memset); at m>8 the 2 nvjet + 2 splitKreduce + 1 add become one launch (plus one memset). The weight read stays 8.4 MB (hi + lo bf16) which at 4.8 TB/s is the 1.8 us floor.

## Expected

- `layer=router load=verify_1 *` faster: m=8: 26.6 us fp32 SIMT path becomes one tensor-core kernel over 8.4 MB; target under 8 us
- `layer=router load=decode_1 *` faster: m=1: 3 launches (gemv dot + cast + reduce, 10.4 us) become 1 kernel + memset
- `layer=router load=decode_16 *` faster: 5 launches (2 nvjet splitK + 2 splitKreduce + add, 14.8 us) become 1 kernel + memset
- `layer=router load=decode_32 *` faster: same 5-to-1 launch reduction at m=32
- `layer=router load=verify_16 *` faster: m=128: same, SPLIT_K=1, 2x 128x256x8192 bf16 dots are ~1 us of tensor-core work
- `layer=router load=verify_32 *` faster: m=256: 17.2 us today; one kernel reading 8.4 MB weights + 4 MB x
- `*layer=qkv_gate*` flat: bf16 layers untouched; control

## Risk

Risks: (1) probe rows at m=1 now use hi/lo (allowed by the rules at any m) where the residual ~1.5e-5 relative must stay under the 1e-3 probe bound and the 1e-4 L2 bound; it should by ~10x, but m=1 has no averaging. (2) tl.atomic_add makes the fp32 sum order nondeterministic across replays; fp32 rounding ~1e-7 is far inside 1e-4. (3) Triton compile of 4-6 (m-bucket) specializations must fit the 10-minute run limit; well under. (4) Graph capture: torch.zeros inside kernel_fn is a fixed-shape allocation plus a memset node, allowed (shapes and config are fixed); no value-dependent branching. (5) The memset + atomics could be replaced by a SPLIT_K=1 full-K loop if the atomic path is slow at m<=32; note it as the ab-tune if the result is only partly faster. (6) If tl.dot with BLOCK_M=16 masked rows is slow at m=1, an FMA fallback (x[:, None, :] * w broadcast + tl.sum) is the alternative; keep that out of this run.

## Adversarial review

Reviewed 2026-10-23T230544Z by an adversarial Fable subagent (model fable).

**Verdict: proceed-with-companions.** The mechanism (one tensor-core hi/lo kernel at every m, replacing 3-5 launches) is sound and the router is the one place in this problem sitting 5-10x above its floor, but the change paragraph's grid is wrong: at m<=32 it launches 32 CTAs (not 256; 256/64 x 8 = 32) and at m>=128 it launches SPLIT_K=1 with M looped inside the program, i.e. 4-8 CTAs streaming 8.4 MB of weights plus the whole x, which will almost certainly be slower than the parent's cuBLAS splitK at verify_16/verify_32. A loss on those rows would then be uninterpretable (bad grid vs. Triton-vs-cuBLAS), so one control with a properly filled grid is warranted.

### Justification
Plausible on H200 and well-targeted. Router after the parent (job 017804b34fed40b3): m=1 10.4 us (dot_kernel 4.1 + x.float() cast 2.0 + reduce 1.6 + gaps), m=8 26.1 us (two sm80_xmma f32 SIMT GEMMs 9.3+8.2 + cast 2.3), m=16 14.5, m=32 15.1, m=128 16.6, m=256 17.8 us (nvjet splitK 8-10.5 + splitKreduce 3.2-3.8 + add ~1.0 + gaps); floors 1.8-2.7 us. Every router shape is >=5x above its floor and the gap is launch count and SIMT fp32, not bytes: note the fp32 router weight is also 8.4 MB, identical to hi+lo bf16, so the m=1 win can only come from fewer launches and better memory-level parallelism, not from fewer bytes (the hypothesis's '1.8 us floor' line is correct but is not a mechanism). Shape weights are 62 everywhere so the load-level arithmetic holds: verify_1 load ~75 us with router 26 us, decode_16 ~75 us with router 14.5, so a router at ~6-8 us is worth 8-25% on each load, ~1.05-1.08x geomean overall. The expected directions are consistent with TASK.md (all six router rows faster, qkv_gate control flat). Weakest link: the CTA count. grid=(N/BLOCK_N, SPLIT_K)=(4,8)=32 programs, not 256 as the text claims; with num_stages=3 x 32 KB/stage that is ~3 MB in flight, borderline for 4.8 TB/s at ~0.7 us HBM latency and sensitive to Triton actually keeping 3 stages outstanding with the mma.sync fallback that BLOCK_M=16 forces on Hopper (wgmma needs M=64). Worse, 'SPLIT_K=1 with a plain store for m>=128' and 'M looped in BLOCK_M chunks for m=256' mean 4 CTAs (BLOCK_N=64) each reading 2.1 MB of weight plus the full 4.2 MB x; at ~100 GB/s per CTA that is 20+ us, slower than the parent's 16.6/17.8 us. cuBLAS chooses splitK at every router shape for exactly this reason (N=256 is narrow, K=8192 is long; parallelism must come from K). Secondary: the memset node for torch.zeros costs ~1-2 us of graph time per call, which eats a third of the budget at m<=32; a split-0-stores/others-add scheme is a race, so the memset is the honest price unless SPLIT_K=1 with an M-block grid is used, which is only viable at m>=128.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (only landed item; promotion pending): no interaction confound. This hypothesis replaces the whole router branch, keeps the parent's prepare() hi/lo split (re-laid out), and leaves the four bf16 layers on _bf16_gemm_dispatch_impl, which on H200 resolves to the same cuBLAS kernels as x @ w.T, so it is inert for this change. The one hazard is the attribution base, not the code: BEST.json still points at the starter (0.902x, job 366189f0f1114da1) because promotion failed; if the harvester attributes this run against BEST.result.json instead of the parent's result.json, router decode_16/32 and verify_16/32 will show as 'gained' (1.3-2.7x) purely from the parent's cuBLAS hi/lo path. Attribution must be against job 017804b34fed40b3's per-shape medians.

### Interference and A/B needs
ab-revert is inapplicable: reverting the parent's patch would restore the starter's fp32 path, a different base, and this change overwrites the router branch anyway. ab-tune of the parent (ROUTER_FP32_MAX_TOKENS) is already being run as router-hilo-at-m8 in this tick, and it serves as the de-facto cuBLAS-hi/lo control for the m=8 row: without it an m=8 gain here could not separate 'left the fp32 SIMT path' from 'Triton beats cuBLAS hi/lo'; with it, it can. The m=16/32/128/256 rows are already on cuBLAS hi/lo in the parent, so they isolate Triton-vs-cuBLAS directly from one run. A gain is therefore interpretable. A loss, or the likely mixed result (small-m faster, m>=128 slower), is not: the primary's grid under-fills the GPU (32 CTAs at m<=32, 4-8 CTAs at m>=128, with x re-read per CTA at m=256), so a slower verify_16/verify_32 row would be indistinguishable from 'a fused Triton hi/lo kernel cannot beat cuBLAS splitK here', and the do-not-retry list would gain a false entry. One control with the same kernel and a grid that puts M-blocks in the grid and sizes SPLIT_K so every launch has >=128 CTAs disambiguates that. No second companion: an m=1 cuBLAS hi/lo control (ROUTER_FP32_MAX_TOKENS=0) would be 2 gemv + 2 reduce + add, five launches, and cannot inform the adopt/reject decision at m=1, which is simply 'is Triton faster than the parent's 10.4 us'. If the dispatcher is allowed to amend the primary's change paragraph, fold the control's grid into the primary and drop the companion; the cost of the wrong grid is a wasted check slot and a misleading FINDINGS entry.

### Companions queued
- `router-triton-hilo-fused-control-splitk-all-m`: Separates 'a fused Triton hi/lo kernel cannot beat cuBLAS splitK + reduce + add on this shape' from 'the primary's grid under-fills the GPU (32 CTAs at m<=32, 4-8 at m>=128)'. If the primary loses at verify_16/verify_32 and this control gains there, the primary's tiling is the cause and the Triton route stays alive; if both lose, the Triton route for the router at large m goes on the do-not-retry list with evidence. If the primary gains everywhere, this control is the tuning data for the follow-up.

### Open questions
- Fix the primary's CTA arithmetic before building: (256/64) x 8 = 32 programs, not 256. If the dispatcher may amend the change paragraph, adopt the control's 3-D grid in the primary and drop the companion.
- Attribution base: BEST.json still names the starter (job 366189f0f1114da1, 0.902x) because the parent's promotion failed. Harvest must compare this run's per-shape medians against the parent's result.json (job 017804b34fed40b3), or every router row m>8 will be credited to this change.
- Memset cost: at m<=32 the torch.zeros memset node is ~1-2 us of a ~6-8 us target. Measure it from the profile (a 'memset' or 'fill' entry) so the follow-up can decide between keeping SPLIT_K atomics and a two-kernel split-K + tiny reduce.
- BLOCK_M=16 forces Triton onto mma.sync rather than wgmma on Hopper; the kernel is memory-bound so this should not matter, but if m=1/m=8 land well above 5 us while m>=32 land near 4 us, that is the signature to look for.
- Interaction with router-hilo-at-m8 (same tick): both are gains if either works; promotion order does not matter because this change overwrites the whole router branch, but the m=8 row of router-hilo-at-m8 is the cuBLAS hi/lo comparison point for this run's verify_1 row and should be cited in the harvest.
- router-hilo-addmm-epilogue becomes moot if this lands; it should be deprioritized rather than run in parallel.
- Nearby-shape reruns: the final review runs unlisted m values. Confirm the launch config selection covers every m (cdiv on M, masking), not just the six scored m values; a crash there is a disqualification risk, not a speed one.

