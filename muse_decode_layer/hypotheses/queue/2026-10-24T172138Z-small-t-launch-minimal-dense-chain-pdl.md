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
review: required
status: queued
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
