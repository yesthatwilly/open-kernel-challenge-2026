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
status: running
review_at: 2026-10-24T172646Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T172706Z
base_sha: 2b6cb6edd61405f6d01cf0aa39cd6e34d56fe812
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/small-t-launch-minimal-dense-chain-pdl
worktree: .worktrees/muse_decode_layer/small-t-launch-minimal-dense-chain-pdl
mode: null
implemented_at: 2026-10-24T174208Z
impl_commit: 0ad20c672ef6e2e895e059ac05d69499299bea15
submission_sha256: 0ff3803200d8a6a96cbe1712a67a1d0183eed13257d3140cfb49ce667f40b22c
norm_sha256: 8494bbd10fed49ceb26daf4e42a46e5dbe87b5f22f9c916f4d5e295d1e7050ac
submission_blob: 70b7563d2c5873c3eac4517889657268679f2098
lint_warnings: 18
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


## Implementation notes

Worktree file: `.worktrees/muse_decode_layer/small-t-launch-minimal-dense-chain-pdl/muse_decode_layer/submission.py` (uncommitted; diff stat 1 file, +989 / -29). Only this file changed.

What changed:
- New section "Small-T dense chain" (between `_post_norm_combine_kernel` and `prepare`): one `load_inline` TU `muse_decode_dense_chain_v1` (`-O3 -arch=sm_90a`, compiled at import, the precedent attention module pattern) with:
  - **Kernel P `dl_pooler_router_kernel<NF>`** (NF = 1/2/4 n8 token fragments for T <= 8/16/32). The swap-AB mma.sync m16n8k16 body and the XOR swizzle, ldmatrix and fragment mapping are taken from the audited `gemv_m8_kernel` (bf16-small-m-cuda-gemv-rewrite) and extended to NF fragments. W_pr bf16 [4608, 8192] is built in prepare(): pooler rows, then 16 tiles of [16 hi rows; their 16 lo rows]. **Grid (review fix): 64 rows x SPLIT_K 3 (K-chunks of 21/21/22 x 128) = 72 x 3 = 216 CTAs, a 4-stage ring of 16 KB weight + NF x 2 KB x per stage (72/80/96 KB), 2 CTAs/SM, so all 216 CTAs are resident in one wave**, with about 10 MB of weight in flight chip-wide. This replaces the spec's 288 CTAs x 128 KB. The fixed-order ticket split-K reduce writes expert_input bf16 (pooler tiles), or logits = sum(hi) + sum(lo) fp32 (router tiles), and resets its counter.
  - **PDL:** P is launched from C++ with `cudaLaunchKernelEx` and `cudaLaunchAttributeProgrammaticStreamSerialization` on the current torch stream. Before `griddepcontrol.wait` it issues only `cp.async` of W_pr (a prepare() constant) for ring stages 0-2, uncommitted. The x copies (ffn_input), the commit groups and every global write come after the wait (review REQUIRED).
  - **Kernel R `dl_route_norm_align_kernel`** (CUDA, grid T x 256). Per token it does the following:
    - The pre-expert RMSNorm with fp32 math, eps 1e-5 and the module's effective (1+w) weight in fp32 (`_get_effective_weight()`, falling back to 1+w).
    - sigmoid(0.1767766952966369 x logit), top-8 of score+bias with ties going to the lower id, and weights s/(sum+1e-15).
    - It writes topk_ids int32 (global), topk_weights fp32 contiguous, and local_ids `(id - local_start) & 255`, set to -1 when >= 32 (master contract).
    - Then an atomic ticket. The last block builds sorted_ids, block_experts and n_padded over the 32 local experts with `__match_any_sync` ranks. Capacity is T*8 + 32*(bsm-1), the sentinel is T*8, experts are ascending with routes ascending inside each expert, and block_experts is -1 past the used blocks. This is the layout of the Triton `_align_small` that passed the judge in moe-align-single-kernel-small-t. Finally it resets the ticket.
  - PDL probe kernels: a spinning primary that calls `griddepcontrol.launch_dependents`, and a secondary that records `%globaltimer` before and after its wait.
- `_block_size_m()` holds master's candidate loop plus the cap of 16, unchanged. `_experts()` gains `align=None`. With None it runs master's code as before; with a value it skips the expert_map gather, the cast and moe_align_block_size.
- `kernel_fn`: the gate is `use_dl = state["dl"] is not None and tokens <= 32`, which depends only on config and the prepare flag. With the gate on, attn_boundary is called with `router_dtype=None`, also at T=1 (review), then P, R and the experts run. The side stream and its wait_stream and record_stream calls are gone. With the gate off (T > 32), the code is master's, line for line, re-indented under `else`.
- `prepare()` ends with `state["dl"] = _dl_setup(weights, state)`, which does the following:
  1. Builds the constants and prints P occupancy.
  2. Runs the PDL probe. It is eager and then captured in a torch CUDA graph and replayed; pdl=1 only if the secondary started before the primary ended under capture.
  3. Runs the self-check with that pdl, and on failure retries with pdl=0.
  4. Times P with PDL off and on: 20 back-to-back launches in a graph, CUDA events, print only.
  5. Prints `[small-t-dense-chain] ON|OFF pdl=0/1 compile Xs max errs ... PATH CHOSEN = DENSE-CHAIN|MASTER`.
- Self-check (prepare only):
  - Runs T in (1, 4, 8, 9, 16, 32) with varied local_start, twice each to catch ticket reset. Bounds: expert_input rel L2 <= 2e-3 vs cuBLAS; logits max abs err <= 1e-4 x max|logit| vs fp64; R norm vs `state["pre_expert_norm"]` <= 2e-3. Routing is compared with FusedMuseSparkRouting on the same logits (id sets equal where the 8th-9th gap > 1e-4, weights within 1e-5), and local_ids are checked exactly.
  - Checks the align structure using only what Marlin consumes: per-expert route sets, block counts, n_padded and the sentinel. **It never compares capacity or n_padded with SGLang's** (this is the moe-sibling audit trap). It also compares Marlin output y between R's align and master's map plus moe_align_block_size (<= 1e-3).
  - Runs crafted-logit cases through R alone at T = 1, 8 and 32: **no local slot** (checks n_padded == 0 and y == 0 when every slot is remote), one hot local expert, and a **near tie** (the 9th placed 1e-7 below the 8th, checked against the 8th best biased score).
  - Captures the real chain (attn_boundary -> P -> R -> experts) at T = 8 and 32 in a CUDA graph and replays it against eager.

Deviations from the spec, and why:
- The grid is 216 CTAs, not 288, with a 72-96 KB ring. The review required one of these two fixes.
- R is CUDA rather than Triton. It shares the TU, and the review allowed either.
- PDL confirmation measures overlap with a globaltimer probe rather than querying graph edge types. The brief allows "or measure overlap". It also avoids `cudaGraphGetEdges` / `_v2`, whose signature differs between CUDA 12.3+ and 13 and would risk the whole TU failing to compile.
- Logits are checked against an fp64 recompute rather than torch fp32 F.linear. This is stricter and avoids TF32 ambiguity.

Uncertainties (nothing could be compiled or run here):
- PTX forms are copied from the audited gemv kernel. The new pieces are `griddepcontrol.wait` / `launch_dependents`, `%globaltimer`, `cudaLaunchKernelEx` with a template kernel, `__match_any_sync`, and the 64-bit `__shfl_xor_sync`.
- A CUDA fault in P or R inside the prepare self-check would be sticky and fail the whole run, not just fall back.
- The FusedMuseSparkRouting call contract (returns (weights, ids)) is taken from master's usage. If its semantics differ, the self-check turns the path OFF.
- At T=32, P re-reads x from L2 per CTA: about 37.7 MB against 75.5 MB of weight. This could cap P below 3.8 TB/s at T=32. BLOCK_N 128 would halve it, and is a possible stage-2 item.
- The profiler row for P includes its PDL wait. Judge M1/M2 from the `P timing` lines.
- At harvest, read the log tail first. `PATH CHOSEN = MASTER`, `SELF-CHECK FAILED` or `load_inline FAILED` means the run measured master.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/small-t-launch-minimal-dense-chain-pdl diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn or anything it reaches. `_dl_forward`/`_dl_route`/`_experts` only allocate with sizes from T and launch, and the C++ wrappers do host-side TORCH_CHECKs on shapes and dtypes only. All `.tolist()`/`synchronize()` calls are in the prepare-only self-check, probe and timing (lint: outside kernel_fn's call graph). Branching is on `tokens` (config) and the prepare flag only. local_start is read on device.
- [x] Nothing derived from `data` is cached across calls: `state["dl"]` holds W_pr, the zeroed workspace and tickets, the norm weight, the bias and the pdl flag, all built from weights in prepare(). The tickets self-reset in every launch. PDL prefetch reads only W_pr before griddepcontrol.wait.
- [x] Weights: W_pr is a lossless repack. The pooler stays BF16, and the router is FP32 -> BF16 hi + lo with FP32 accumulation, the one allowed exception, now also used at T=1, which the rules allow at any batch. NVFP4 experts are untouched (master's Marlin bank).
- [x] Activations: expert_input and the normed expert input are BF16 as in SGLang, logits FP32 and topk_weights FP32. No FP8/FP16 introduced. KV cache and queries are untouched.
- [x] n/a: attention is untouched (the change is the FFN pre-expert chain only). Every key is still attended.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. `load_inline` compiles at import, as the promoted attention precedent does (DECISIONS: the load_inline rules question is settled). reference.py was read for semantics only and is not imported.
- [x] All work happens inside the timed call: P, R and the experts run every call on fresh inputs. No output caching or replay. The prepare() self-check uses synthetic inputs only.
- [x] First line marker is `# okc: problem=muse_decode_layer` (kept).
- [x] T > 32 keeps master byte for byte (verify_16/32 control). `_block_size_m` is the same loop plus cap 16 as master's inline code.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L890: ids = local_ids.reshape(-1).tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L892: s_ids, b_exp, n_pad = sorted_ids.tolist(), block_experts.tolist(), int(n_padded[0])  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L929: if not bool((picked >= (eighth - 1e-5 - 1e-6 * eighth.abs())[:, None]).all()) or any(len(set(r)) != TOP_K for r in mine.tolist()):  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L934: gap = (eighth - ninth).tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L935: for t, (mi, ri) in enumerate(zip(mine.tolist(), ref_ids.tolist())):  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L940: mw = dict(zip(mi, topk_w[t].tolist()))  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L941: rw = dict(zip(ri, ref_w[t].tolist()))  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L968: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1014: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1041: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1046: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1064: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1065: eager = times.tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1072: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1073: cap = times.tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1101: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1107: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1221: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
exit=0
```

`python3 -I -m py_compile`: OK (the __pycache__ it left in the worktree was deleted).
