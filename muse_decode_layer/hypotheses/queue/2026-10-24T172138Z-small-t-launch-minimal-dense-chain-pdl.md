---
problem: muse_decode_layer
slug: small-t-launch-minimal-dense-chain-pdl
created: 2026-10-24T172138Z
base: master
area: whole-layer fusion
change: >
  WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per
  layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.7 us, the dense GEMMs ~55 us
  against a 28.8 us weight floor (qkv 37.7 + o 16.8 + router 8.4 + pooler 67.1 + unpooler slice
  8.4 MB), FA3 ~14 us, and ~20 small kernels (norms, casts, store, routing, map, align, pad,
  splitKreduce) about 40 us, ~2 us each. LESSONS says fusing two or three of them is flat (each
  ~1.5%, inside drift), which is why it has to be done as one attributable rewrite that removes
  most of them at once, while also lifting the weight streams to >=3.8 TB/s (Triton reached 3.73
  TB/s, LESSONS) and hiding each stream ramp behind the predecessor with programmatic dependent
  launch (PDL): weights are constants from prepare, so a CTA may issue its first ring stages of
  weight before griddepcontrol.wait without caching anything derived from inputs. STAGES: (1) this
  file, the FFN pre-expert chain; (2) attention-side chain: one qkv split-K GEMV whose last CTA
  per 64-row head tile (ticket semaphore, LESSONS) does QK RMSNorm, RoPE on sliding, k/v FP8 cast
  + cache store at cache_loc, q FP8 cast (paired-padded layout when master pairs), gate bf16; and
  an o GEMV whose prologue does the per-head RMSNorm x sigmoid(gate) (and paired unpad) on the
  [T,1024] attention output; (3) unpooler GEMV writing the zero columns itself (no F.pad), PDL on
  every custom kernel, then folding the two boundaries at T<=8. End state ~12-13 nodes at T<=32.
  STAGE-1 SCOPE, gate tokens <= 32 from config (T>32 keeps master byte for byte). PREPARE: build
  W_pr bf16 [4608, 8192] = pooler rows 0..4095, then the router as 16 tiles of [16 hi rows of
  experts 16i..16i+15 ; their 16 lo rows] (hi = router.to(bf16), lo = (router -
  hi.float()).to(bf16): the rules allow hi+lo at any batch, so T=1 also leaves the fp32 gemv).
  Compile one load_inline TU (-O3 -arch=sm_90a; the bf16 problem branch
  bf16-small-m-cuda-gemv-rewrite has a gemv_m8_kernel swap-AB body to start from). KERNEL P
  dl_pooler_router_gemv: swap-AB mma.sync m16n8k16, weight rows on M, tokens on N (n = 8/16/32 by
  T, x zero-padded rows masked), row tiles of 32 x SPLIT_K 2 = 288 CTAs, 4-stage cp.async ring of
  32 rows x 512 k (128 KB in flight per CTA), fixed-order ticket-semaphore split-K reduce;
  epilogue: rows < 4096 -> expert_input bf16 [T,4096]; router tiles -> logits fp32 [T,256] =
  acc_hi + acc_lo in fp32 inside the tile. Launched from C++ with cudaLaunchKernelEx and
  cudaLaunchAttributeProgrammaticStreamSerialization on the current torch stream; before
  griddepcontrol.wait it issues ring stages 0-2 of its weight, then waits, then reads ffn_input.
  The side stream and its wait_stream/record_stream calls go away at T<=32. KERNEL R
  dl_route_norm_align (Triton or CUDA, grid T programs): program t computes the pre-expert RMSNorm
  of expert_input[t] x (1+w) (eps 1e-5, fp32 math, bf16 out), and the routing exactly as
  FusedMuseSparkRouting does (read its source: sigmoid(0.1767766952966369 x logit) + bias for
  selection, top-8, weights from the unbiased scores renormalized, route_scale 1, eps 1e-15)
  writing topk_ids int32 (global, the returned experts), topk_weights fp32 and local_ids ((id -
  local_start) mod 256 mapped through expert_map, -1 when remote); then an acq_rel atomic ticket;
  the last program builds Marlin sorted_ids / block_experts / n_padded for master block_size_m
  exactly as the current align does, and resets the ticket. Marlin w13, silu, w2 and the post-norm
  combine are untouched. FALLBACK + SELF-CHECK: in prepare run T in (1, 4, 8, 9, 16, 32) synthetic
  ffn_input through both master path and the new path: expert_input rel L2 <= 2e-3, logits max abs
  err vs fp32 router <= 1e-4 x max|logit|, topk ids equal on tie-free data,
  sorted_ids/block_experts/n_padded producing identical Marlin output y, and the PDL launch
  captured and replayed once inside a CUDA graph; any failure (or capture error with PDL, then
  retry without PDL and log pdl=0) sets the flag off and keeps master. Print one line
  [small-t-dense-chain] ON/OFF pdl=0/1 compile s, max errs. MILESTONES from the profile at T<=32
  shapes: rows dl_pooler_router_gemv and dl_route_norm_align present; the nvjet pooler row, its
  splitKreduce, the router dot/mm rows, FusedMuseSparkRouting, vectorized_layer_norm (F.rms_norm)
  and the align/map elementwise rows gone (kernel count down by >=8 per profile); M1 P <= master
  pooler+router+reduce rows (decode_1 ~25-30 us); M2 P <= 20 us at T=1 and <= 22 us at T=32 (75.5
  MB at >=3.8 TB/s), R <= 4 us; M3 decode_1 sliding <= 130 us.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "incumbent 145.6 / 156.2 us (floor 40.0 / 44.2): ~10 nodes -> 2 and the pooler+router stream at >=3.8 TB/s: ~15-20 us off (+10-14% on the shape)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "incumbent 204.8 / 215.5 us (floor 62.3 / 66.4): same nodes at T=8, ~15-18 us off (+7-9%)"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "incumbent 230.3 / 385.5 us (floor 76.7 / 137.5): ~12-18 us off (+5-7% sliding)"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "incumbent 269.7 / 557.1 us (floor 84.5 / 208.1): pooler nvjet 64x32 + splitK rows and the side-stream router, ~12-18 us off (+5-6% sliding). Headline stage 1 ~+4-6% (1.166x -> ~1.22-1.23x); all three stages target ~35-50 us off every T<=32 shape (+12-18% headline)"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "T=128 outside the gate: control"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "T=256 outside the gate: control"
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "T<=32 dense chain: stage 1 = pooler GEMM (nvjet + splitKreduce), router (fp32 dot gemv at T=1, bf16 hi/lo mm pair + add at T>=2) and its side-stream fork/join, FusedMuseSparkRouting, local-id modulo + expert_map gather + contiguous + topk_weights cast, moe align, pre-expert F.rms_norm; stage 2 = qkv GEMM + FusedQKNorm + k/v/q FP8 casts + store_cache + paired pad, o GEMM + FusedRMSNormGate + unpad; stage 3 = unpooler + F.pad and PDL across the chain"
review: proceed
status: queued
review_at: 2026-10-24T172646Z
review_companions: null
---

# small-t-launch-minimal-dense-chain-pdl

## Change

WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.7 us, the dense GEMMs ~55 us against a 28.8 us weight floor (qkv 37.7 + o 16.8 + router 8.4 + pooler 67.1 + unpooler slice 8.4 MB), FA3 ~14 us, and ~20 small kernels (norms, casts, store, routing, map, align, pad, splitKreduce) about 40 us, ~2 us each. LESSONS says fusing two or three of them is flat (each ~1.5%, inside drift), which is why it has to be done as one attributable rewrite that removes most of them at once, while also lifting the weight streams to >=3.8 TB/s (Triton reached 3.73 TB/s, LESSONS) and hiding each stream ramp behind the predecessor with programmatic dependent launch (PDL): weights are constants from prepare, so a CTA may issue its first ring stages of weight before griddepcontrol.wait without caching anything derived from inputs. STAGES: (1) this file, the FFN pre-expert chain; (2) attention-side chain: one qkv split-K GEMV whose last CTA per 64-row head tile (ticket semaphore, LESSONS) does QK RMSNorm, RoPE on sliding, k/v FP8 cast + cache store at cache_loc, q FP8 cast (paired-padded layout when master pairs), gate bf16; and an o GEMV whose prologue does the per-head RMSNorm x sigmoid(gate) (and paired unpad) on the [T,1024] attention output; (3) unpooler GEMV writing the zero columns itself (no F.pad), PDL on every custom kernel, then folding the two boundaries at T<=8. End state ~12-13 nodes at T<=32. STAGE-1 SCOPE, gate tokens <= 32 from config (T>32 keeps master byte for byte). PREPARE: build W_pr bf16 [4608, 8192] = pooler rows 0..4095, then the router as 16 tiles of [16 hi rows of experts 16i..16i+15 ; their 16 lo rows] (hi = router.to(bf16), lo = (router - hi.float()).to(bf16): the rules allow hi+lo at any batch, so T=1 also leaves the fp32 gemv). Compile one load_inline TU (-O3 -arch=sm_90a; the bf16 problem branch bf16-small-m-cuda-gemv-rewrite has a gemv_m8_kernel swap-AB body to start from). KERNEL P dl_pooler_router_gemv: swap-AB mma.sync m16n8k16, weight rows on M, tokens on N (n = 8/16/32 by T, x zero-padded rows masked), row tiles of 32 x SPLIT_K 2 = 288 CTAs, 4-stage cp.async ring of 32 rows x 512 k (128 KB in flight per CTA), fixed-order ticket-semaphore split-K reduce; epilogue: rows < 4096 -> expert_input bf16 [T,4096]; router tiles -> logits fp32 [T,256] = acc_hi + acc_lo in fp32 inside the tile. Launched from C++ with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization on the current torch stream; before griddepcontrol.wait it issues ring stages 0-2 of its weight, then waits, then reads ffn_input. The side stream and its wait_stream/record_stream calls go away at T<=32. KERNEL R dl_route_norm_align (Triton or CUDA, grid T programs): program t computes the pre-expert RMSNorm of expert_input[t] x (1+w) (eps 1e-5, fp32 math, bf16 out), and the routing exactly as FusedMuseSparkRouting does (read its source: sigmoid(0.1767766952966369 x logit) + bias for selection, top-8, weights from the unbiased scores renormalized, route_scale 1, eps 1e-15) writing topk_ids int32 (global, the returned experts), topk_weights fp32 and local_ids ((id - local_start) mod 256 mapped through expert_map, -1 when remote); then an acq_rel atomic ticket; the last program builds Marlin sorted_ids / block_experts / n_padded for master block_size_m exactly as the current align does, and resets the ticket. Marlin w13, silu, w2 and the post-norm combine are untouched. FALLBACK + SELF-CHECK: in prepare run T in (1, 4, 8, 9, 16, 32) synthetic ffn_input through both master path and the new path: expert_input rel L2 <= 2e-3, logits max abs err vs fp32 router <= 1e-4 x max|logit|, topk ids equal on tie-free data, sorted_ids/block_experts/n_padded producing identical Marlin output y, and the PDL launch captured and replayed once inside a CUDA graph; any failure (or capture error with PDL, then retry without PDL and log pdl=0) sets the flag off and keeps master. Print one line [small-t-dense-chain] ON/OFF pdl=0/1 compile s, max errs. MILESTONES from the profile at T<=32 shapes: rows dl_pooler_router_gemv and dl_route_norm_align present; the nvjet pooler row, its splitKreduce, the router dot/mm rows, FusedMuseSparkRouting, vectorized_layer_norm (F.rms_norm) and the align/map elementwise rows gone (kernel count down by >=8 per profile); M1 P <= master pooler+router+reduce rows (decode_1 ~25-30 us); M2 P <= 20 us at T=1 and <= 22 us at T=32 (75.5 MB at >=3.8 TB/s), R <= 4 us; M3 decode_1 sliding <= 130 us.

## Expected

- `*load=decode_1 *` faster: incumbent 145.6 / 156.2 us (floor 40.0 / 44.2): ~10 nodes -> 2 and the pooler+router stream at >=3.8 TB/s: ~15-20 us off (+10-14% on the shape)
- `*load=verify_1 *` faster: incumbent 204.8 / 215.5 us (floor 62.3 / 66.4): same nodes at T=8, ~15-18 us off (+7-9%)
- `*load=decode_16 *` faster: incumbent 230.3 / 385.5 us (floor 76.7 / 137.5): ~12-18 us off (+5-7% sliding)
- `*load=decode_32 *` faster: incumbent 269.7 / 557.1 us (floor 84.5 / 208.1): pooler nvjet 64x32 + splitK rows and the side-stream router, ~12-18 us off (+5-6% sliding). Headline stage 1 ~+4-6% (1.166x -> ~1.22-1.23x); all three stages target ~35-50 us off every T<=32 shape (+12-18% headline)
- `*load=verify_16 *` flat: T=128 outside the gate: control
- `*load=verify_32 *` flat: T=256 outside the gate: control

## Risk

Biggest risk: PDL may not survive the judge CUDA-graph capture or give little overlap because the preceding SGLang JIT boundary kernel never triggers launch_dependents (then the gain is only the node count and the faster stream, ~10 us); routing must match FusedMuseSparkRouting semantics or tests fail on route checks. Overlap: subsumes the running bf16-shape-gated-triton-gemms-small-t pooler part, queued moe-align-small-t-restack-on-master and held router-triton-splitk-side-stream; build on master at claim time and judge on the replaced rows. Independent of moe-chain-cuda-swapab-expert-once-large-t (disjoint T), so both can stack by 3-way merge.

## Milestones

Rewrite stage 1 of `T<=32 dense chain: stage 1 = pooler GEMM (nvjet + splitKreduce), router (fp32 dot gemv at T=1, bf16 hi/lo mm pair + add at T>=2) and its side-stream fork/join, FusedMuseSparkRouting, local-id modulo + expert_map gather + contiguous + topk_weights cast, moe align, pre-expert F.rms_norm; stage 2 = qkv GEMM + FusedQKNorm + k/v/q FP8 casts + store_cache + paired pad, o GEMM + FusedRMSNormGate + unpad; stage 3 = unpooler + F.pad and PDL across the chain`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T172646Z by an adversarial subagent (model opus).

**Verdict: proceed.** The rewrite's target can be reached, and the rules allow it, including PDL weight prefetch before griddepcontrol.wait and the hi+lo router at T=1. But the stage-1 gain is overstated: at T=1 the pooler already streams at about 3.2 TB/s and the router is hidden on the side stream, so most of the saving must come from removing nodes, which LESSONS records as weak. The 288-CTA x 128 KB grid runs in 2.2 waves at 1 CTA/SM, and that is the biggest risk to M2.

### Justification
Mechanism: at T<=32 the dense pre-expert chain is a serial run of bandwidth- and latency-bound launches, and decode_1 sits at 145.6 us against a 40 us floor. Collapsing pooler + splitKreduce + router + routing + mod + expert_map gather + contiguous + cast + align + F.rms_norm into 2 kernels is plausible on H200, and the profile rows confirm the replaced kernels exist and cost something. At decode_1: vectorized_layer_norm (F.rms_norm) is 5.9 us, splitKreduce 5.0 us over 3 calls, the fp32 dot_kernel router 5.0 us, _moe_align_small_numel 3.6 us at verify_1, and vectorized_elementwise 2.8 us over 3 calls. The expected directions match TASK.md: T<=32 shapes faster, and verify_16/32 (T=128/256) flat because they are outside the config gate, which is a good built-in control. Corrections to the arithmetic: (1) The decode_1 row nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT, 32.6 us, is 2 calls (qkv 37.7 MB + pooler 67.1 MB = 105 MB, about 3.2 TB/s), not the pooler alone. The pooler is about 21 us plus its share of splitKreduce. P moving 75.5 MB at 3.8 TB/s is 19.9 us, so the stream-bandwidth gain at T=1 is only about 2-4 us. (2) The router (8.4 MB) today runs concurrently on the side stream. Folding it into P puts its bytes on the critical path, so the side-stream removal is mostly a node-count win, not a byte win. (3) Removing small nodes was measured neutral twice on this exact chain (expert-id-map-single-kernel 1.037x, moe-align-single-kernel-small-t 1.043x, both within CI). The realistic stage-1 saving is about 8-14 us on decode_1 and verify_1 and about 6-12 us at T=16/32: +5-9% on those shapes and about +2-4% headline, not +4-6%. A 2-4% headline gain is still readable from per-shape rows and replaced-kernel time, which is how a rewrite is judged. Weakest link: the P design's own bandwidth. 4608 rows / 32 = 144 row tiles x SPLIT_K 2 = 288 CTAs. Each holds 128 KB of smem, so only 1 CTA/SM fits (228 KB/SM), and the grid runs in waves of 132/132/24: the last wave streams with 18% of the SMs, and two wave ramps are exposed. M2 (<=20 us at T=1) is unlikely as written. Fix before building: either (a) a persistent or stream-K grid of 132 CTAs walking (row tile, k-chunk) units, or (b) a 64-96 KB ring (e.g. 4 stages x 32 rows x 256 k = 64 KB) so 2-3 CTAs/SM keep all 288 resident in one wave. Little's law needs only about 5-8 MB in flight chip-wide at 4.8 TB/s (264 x 64 KB = 17 MB is plenty); the 3.73 TB/s LESSONS figure came from a Triton grid shaped differently. A second measurement caveat: the profiler charges a PDL-launched kernel's row with its wait on the predecessor ('an overlapping kernel's time includes any wait for the one before it'), so P's top_kernels row will read high when pdl=1. Time P in prepare() with CUDA events, with and without PDL, and print both, so M1/M2 are judged on P's stream time.

### Confounders on the current kernel
- No landed hypotheses are listed for master, so no landed patch confounds this run. Master's in-file history does touch this region: sglang-pipeline-parity-router-hilo-m8 (ROUTER_FP32_MAX_TOKENS=1, router_dtype=None at T>=2), marlin-block-m-cap-16 and post-expert-norm-triton-small-t (SMALL_T_POST_NORM=32). R must reproduce their contracts exactly: local_ids -1 for remote slots (the _post_norm_combine_kernel skips on eid<0 and never loads those y rows), topk_weights fp32 contiguous, and block_size_m from BLOCK_SIZE_M_CANDIDATES then capped at 16 (8 at T=1/8, 16 at T=16/32). Moving T=1 to hi+lo also means attn_boundary at T=1 must be asked for router_dtype=None (the fp32 router_input write goes away), which is a small extra saving, not a confounder.
- In-flight bf16-shape-gated-triton-gemms-small-t (running, pooler at m<=16) and queued moe-align-small-t-restack-on-master replace the same rows. If either lands before this is claimed, the incumbent rows change (Triton pooler instead of nvjet+splitKreduce; one align launch). Judge M1 against whatever master is at claim time, not against the numbers quoted in the hypothesis.

### Interference and A/B needs
No companion is warranted. Nothing has landed on master's lineage that this change could interact with. The T>32 shapes are byte-identical and serve as an in-run drift control. The attribution question that matters (how much is P's stream, how much is node removal, how much is PDL) is answered more cheaply inside the run than with a second check: (1) the profile rows show P and R time against the vanished rows; (2) prepare() can time P with PDL on and off using CUDA events on synthetic input and print both, alongside the existing [small-t-dense-chain] line. A pdl=0 control run would only help if the prepare-time numbers are inconclusive, and then it is a cheap follow-up (flip the PDL launch attribute only). The brief also asks for a rewrite's follow-up to be its next stage, not a companion.

### Open questions
- Which grid will P actually use: 288 x 128 KB at 1 CTA/SM (2.2 waves), a persistent/stream-K 132-CTA grid, or a smaller ring at 2-3 CTAs/SM? Pick before building; M2 depends on it.
- Does the PDL edge actually form in the captured graph? Is P's predecessor at T<=32 a kernel node on the main stream once the side-stream fork/join is gone? Check the edge type in the prepare self-check, not just capture success.
- Does the profile row for P include its PDL wait? Print prepare-time CUDA-event timings of P with pdl=0 and pdl=1 so M1/M2 are judged on the kernel's own stream time.
- R's last-program align must match moe_align_block_size's sentinel, padding and -1 handling exactly at block_size_m 8 and 16, and the buffer sizes Marlin expects. Is comparing Marlin output y on synthetic skewed routing (including all-remote tokens and a single hot expert) enough coverage? Add a T where no slot is local.
- Tie-break order in top-8 must match FusedMuseSparkRouting, or at least stay inside the judge's 0.05/0.03 routing tolerances. Equality is checked only on tie-free data; one near-tie synthetic case would give confidence.
- If bf16-shape-gated-triton-gemms-small-t lands first, does this still beat the Triton pooler row it installs (likely about 3.7 TB/s)? If not, stage 1's win is mostly node removal, and the headline expectation should be revised down.

