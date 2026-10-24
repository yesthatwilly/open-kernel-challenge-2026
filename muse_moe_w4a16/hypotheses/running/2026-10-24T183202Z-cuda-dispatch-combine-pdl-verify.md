---
problem: muse_moe_w4a16
slug: cuda-dispatch-combine-pdl-verify
created: 2026-10-24T183202Z
base: master
area: MoE align/dispatch + post-expert norm+combine
change: >
  Dispatch + combine only, at num_tokens > 32 (verify_16 bs=128, verify_32 bs=256, and the bs=33,
  bs=257 and skewed bs=128 tests). num_tokens <= 32 runs master byte for byte (that range is owned
  by the running swap-AB/fused-bs1/two-stream runs and the queued midm single-launch rewrite), and
  both moe_wna16_marlin_gemm calls and silu_and_mul stay master's calls at every M (the running
  cuda-nvfp4-wgmma-rega-grouped-verify owns those kernels at bs>=128; this file must not touch
  them, so its rows stay attributable). WHAT IS REPLACED: master's _align_parallel_kernel
  (33-program Triton, 4.0 us at bs=128, 6.9 us at bs=256 in job 45eacf4ca2e849d2) and SGLang's
  fused_post_expert_norm_weighted_sum (_post_expert_norm_weighted_sum_kernel, 4.9 / 6.0 us), i.e.
  the whole routing-to-Marlin dispatch and the Marlin-to-output combine, with two hand-written
  CUDA kernels in ONE load_inline TU compiled in prepare() (extra_cuda_cflags -O3 -arch=sm_90a, no
  fast-math; build pattern of muse_attention_decode _build_splitkv, which compiled in the judge
  image in ~41 s; print compile seconds). WHY: both are latency-bound, not byte-bound (the align
  moves ~8 KB of ids; the combine reads only the local y rows, a few MB) so a 1-launch,
  all-loads-in-flight design should take each to ~2 us; this is the largest remaining gap not
  covered by an in-flight run (every Marlin and silu row at every M is). KERNEL A, cuda align (one
  CTA, 1024 threads, grid (1,)): args topk_ids (int32 or int64 via a template, cast inside), n =
  M*8, local_start, local_count (<= 32), bsm (runtime int; 16/32/48/64 all handled, no power-of-2
  constraint), cap = n + local_count*(bsm-1), n_blocks = cdiv(cap, bsm). Pass 1: route i -> local
  = id - local_start in [0, local_count) else -1, store local_ids[i]; routes are processed in warp
  tiles of 32 (tile w = routes [32w, 32w+32)); per tile, __match_any_sync on local gives the peer
  mask, rank_in_tile = popc(peers & lanemask_lt), and the lowest lane of each peer group writes
  cnt[w][local] (shared, int16/int32, zeroed first) = popc(peers). Requires n <= 32*MAX_TILES with
  MAX_TILES = 128 (n <= 4096, M <= 512); larger M falls back to master's _align_parallel (host
  branch on shape). After __syncthreads: warp 0 does, per expert e (lane = e), the exclusive scan
  over tiles base[w][e] and total count[e]; padded[e] = roundup(count[e], bsm); a warp inclusive
  scan (__shfl_up_sync) gives seg_start[e]; total = sum padded. Pass 2: real route i writes
  sorted_ids[seg_start[e] + base[w][e] + rank_in_tile] = i, so the order within an expert is
  ascending route index, EXACTLY master's _align_parallel order (byte-identical sorted_ids is the
  self-check target); padding slots of each expert and every slot in [total, cap) get the sentinel
  n; block_experts[b] = the expert whose segment contains b*bsm for b < total/bsm, -1 for
  total/bsm <= b < n_blocks; n_padded[0] = total. Outputs go into prepare-allocated scratch
  buffers sized for M <= 512 (local_ids [512*8], sorted_ids [512*8 + 32*63], block_experts,
  n_padded [1]) and are passed to Marlin as exact-size views [:cap], [:n_blocks] (slicing, not
  allocation; Marlin reads sorted_ids.size(0), so the view must have exactly master's capacity).
  KERNEL B, cuda post-norm combine (grid M, 256 threads, 16 bf16 columns per thread as two 16-byte
  vectors, H = 4096 constexpr with a host assert): prologue loads (1 + post_norm_w) in fp32 (from
  the original bf16 post_norm_w, as master's _post_norm_combine_kernel does), and the token's 8
  local ids and fp32 weights into registers; then griddepcontrol.wait
  (cudaGridDependencySynchronize) when launched with PDL; then issues ALL local-slot loads before
  any math (up to 8 slots x 2 x ld.global.nc.L1::no_allocate.v4, so ~2 KB per warp in flight;
  remote slots, local id -1, are never loaded since their y rows are uninitialised), computes the
  8 per-slot sums of squares with one warp-shuffle reduce of an 8-vector and ONE shared-memory
  round across the 8 warps (one __syncthreads), r_s = rsqrt(ss_s / H + 1e-8), out = sum_s g_s *
  v_s * nw * r_s in fp32, one bf16 16-byte store x2; a token with no local slot stores zeros. Same
  math and eps as master's _post_norm_combine_kernel. PDL: kernel B is launched from the TU's host
  wrapper with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 on
  at::cuda::getCurrentCUDAStream(), so its launch and prologue overlap the w2 Marlin tail inside
  the graph (Marlin never triggers, so the implicit trigger at its block exits applies;
  griddepcontrol.wait guarantees y is visible). PDL is a separate flag state['pdl_ok'], set only
  if the prepare() graph-capture test below passes with it; otherwise kernel B launches plainly.
  kernel_fn at num_tokens > 32 and state['cuda_glue_ok']: block_size_m exactly as master (SGLang
  pick, then two-smaller at >= 128); kernel A replaces _align_parallel; master's Marlin w13,
  silu_and_mul, Marlin w2 unchanged; kernel B replaces fused_post_expert_norm_weighted_sum and
  writes into a torch.empty((M, H), bf16) output exactly as master's small-M branch allocates its
  out (the returned tensor; no other allocation added, no .item()/.cpu()/sync, branch on shape and
  prepare-time flags only). SELF-CHECK in prepare() (untimed, fixed torch.Generator, synthetic
  inputs only, nothing derived from any call's data): (1) align: M in (33, 57, 120, 128, 256, 257,
  512) x every bsm kernel_fn can pick at that M (incl. the two-smaller pick) x local_start in (0,
  32, 224) x {random, skewed, all-remote, all-one-expert, all-local} (the same generators as
  master's _align_self_check): require torch.equal of local_ids, sorted_ids[:cap],
  block_experts[:n_blocks] and n_padded against master's _align_parallel on the same ids. Do NOT
  gate on master's _align_self_check return or its 'capacity vs SGLang' line (LESSONS: it always
  reports capacity mismatches); compare to _align_parallel directly. (2) combine: M in (33, 128,
  256, 257) x the same routings, y = random bf16 [M, 8, H] with every REMOTE row filled with NaN
  (proves remote rows are never read), real bank post_norm_w: compare kernel B against an fp32
  torch reference of the formula (per-row rel L2 <= 4e-3, global <= 2e-3, no NaN/Inf) and report
  its rel L2 against fused_post_expert_norm_weighted_sum on the same y with remote rows zeroed
  (info). (3) end to end: M in (128, 256, 257) random + skewed, real weights, synthetic x: the
  full new kernel_fn path vs master's path, per-row rel L2 <= 1e-2. (4) capture each of (1)-(3)'s
  kernel launches inside torch.cuda.graph and replay 3 times, with PDL on; if the PDL capture or
  replay raises or mismatches, retry without PDL and set pdl_ok False. FALLBACK: any load_inline
  exception, kernel launch error, mismatch or capture error sets state['cuda_glue_ok'] = False and
  every M runs master byte for byte. LOGGING, loud in both outcomes: '[cuda-dispatch-combine]
  build OK/FAILED (<s> s)', 'SELF-CHECK PASSED/FAILED' with the failing case and max rel L2,
  numRegs and localSizeBytes of both kernels (cudaFuncGetAttributes), and 'PATH CHOSEN: cuda
  align+combine (pdl on|off) for num_tokens > 32' or 'PATH CHOSEN: master (reason)'. STAGES: stage
  1 is this file; stage 2 (if M3 holds) folds kernel B's sum of squares into whichever w2 GEMM
  rewrite lands, or extends kernel A+B to bs<=32 if the midm rewrite is dropped; whichever GEMM
  rewrite lands at bs>=128 can take kernel A's outputs unchanged (same format as
  moe_align_block_size). Scope: two small kernels, ~250 lines of CUDA + ~150 lines of Python;
  about 1-1.5 h.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "verify_32 (454.3 us median in 45eacf4ca2e849d2): replaced rows _align_parallel_kernel 6.9 us + _post_expert_norm_weighted_sum_kernel 6.0 us = 12.9 us. Floor for the two: align moves ~8 KB of ids (launch-bound, ~1.5 us), combine reads the local y rows plus 2 MB of output (~1.5-3.5 us at 4.8 TB/s depending on local routes): ~3-5 us. Targets: align <= 2.0 us, combine <= 3.0 us (M3), ~8 us off the shape (~1.8%), plus up to ~1 us of graph gap from PDL"
  - shapes: bs=128 *
    direction: faster
    why: "verify_16 (318.3 us median): replaced rows _align_parallel_kernel 4.0 us + _post_expert_norm_weighted_sum_kernel 4.9 us = 8.9 us vs a ~2.5-4 us floor; targets align <= 1.8 us, combine <= 2.5 us (M3), ~4.5-5 us off the shape (~1.5%)"
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens <= 32 runs master byte for byte"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens <= 32 runs master byte for byte"
  - shapes: bs=8 *
    direction: flat
    why: "control: num_tokens <= 32 runs master byte for byte"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: _align_parallel_kernel + fused_post_expert_norm_weighted_sum at num_tokens > 32
review: proceed
status: running
review_at: 2026-10-24T183415Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T183415Z
base_sha: b17562ed4f8acdd2a6e2e98f542acc2bba35a2b4
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/cuda-dispatch-combine-pdl-verify
worktree: .worktrees/muse_moe_w4a16/cuda-dispatch-combine-pdl-verify
mode: null
---

# cuda-dispatch-combine-pdl-verify

## Change

Dispatch + combine only, at num_tokens > 32 (verify_16 bs=128, verify_32 bs=256, and the bs=33, bs=257 and skewed bs=128 tests). num_tokens <= 32 runs master byte for byte (that range is owned by the running swap-AB/fused-bs1/two-stream runs and the queued midm single-launch rewrite), and both moe_wna16_marlin_gemm calls and silu_and_mul stay master's calls at every M (the running cuda-nvfp4-wgmma-rega-grouped-verify owns those kernels at bs>=128; this file must not touch them, so its rows stay attributable). WHAT IS REPLACED: master's _align_parallel_kernel (33-program Triton, 4.0 us at bs=128, 6.9 us at bs=256 in job 45eacf4ca2e849d2) and SGLang's fused_post_expert_norm_weighted_sum (_post_expert_norm_weighted_sum_kernel, 4.9 / 6.0 us), i.e. the whole routing-to-Marlin dispatch and the Marlin-to-output combine, with two hand-written CUDA kernels in ONE load_inline TU compiled in prepare() (extra_cuda_cflags -O3 -arch=sm_90a, no fast-math; build pattern of muse_attention_decode _build_splitkv, which compiled in the judge image in ~41 s; print compile seconds). WHY: both are latency-bound, not byte-bound (the align moves ~8 KB of ids; the combine reads only the local y rows, a few MB) so a 1-launch, all-loads-in-flight design should take each to ~2 us; this is the largest remaining gap not covered by an in-flight run (every Marlin and silu row at every M is). KERNEL A, cuda align (one CTA, 1024 threads, grid (1,)): args topk_ids (int32 or int64 via a template, cast inside), n = M*8, local_start, local_count (<= 32), bsm (runtime int; 16/32/48/64 all handled, no power-of-2 constraint), cap = n + local_count*(bsm-1), n_blocks = cdiv(cap, bsm). Pass 1: route i -> local = id - local_start in [0, local_count) else -1, store local_ids[i]; routes are processed in warp tiles of 32 (tile w = routes [32w, 32w+32)); per tile, __match_any_sync on local gives the peer mask, rank_in_tile = popc(peers & lanemask_lt), and the lowest lane of each peer group writes cnt[w][local] (shared, int16/int32, zeroed first) = popc(peers). Requires n <= 32*MAX_TILES with MAX_TILES = 128 (n <= 4096, M <= 512); larger M falls back to master's _align_parallel (host branch on shape). After __syncthreads: warp 0 does, per expert e (lane = e), the exclusive scan over tiles base[w][e] and total count[e]; padded[e] = roundup(count[e], bsm); a warp inclusive scan (__shfl_up_sync) gives seg_start[e]; total = sum padded. Pass 2: real route i writes sorted_ids[seg_start[e] + base[w][e] + rank_in_tile] = i, so the order within an expert is ascending route index, EXACTLY master's _align_parallel order (byte-identical sorted_ids is the self-check target); padding slots of each expert and every slot in [total, cap) get the sentinel n; block_experts[b] = the expert whose segment contains b*bsm for b < total/bsm, -1 for total/bsm <= b < n_blocks; n_padded[0] = total. Outputs go into prepare-allocated scratch buffers sized for M <= 512 (local_ids [512*8], sorted_ids [512*8 + 32*63], block_experts, n_padded [1]) and are passed to Marlin as exact-size views [:cap], [:n_blocks] (slicing, not allocation; Marlin reads sorted_ids.size(0), so the view must have exactly master's capacity). KERNEL B, cuda post-norm combine (grid M, 256 threads, 16 bf16 columns per thread as two 16-byte vectors, H = 4096 constexpr with a host assert): prologue loads (1 + post_norm_w) in fp32 (from the original bf16 post_norm_w, as master's _post_norm_combine_kernel does), and the token's 8 local ids and fp32 weights into registers; then griddepcontrol.wait (cudaGridDependencySynchronize) when launched with PDL; then issues ALL local-slot loads before any math (up to 8 slots x 2 x ld.global.nc.L1::no_allocate.v4, so ~2 KB per warp in flight; remote slots, local id -1, are never loaded since their y rows are uninitialised), computes the 8 per-slot sums of squares with one warp-shuffle reduce of an 8-vector and ONE shared-memory round across the 8 warps (one __syncthreads), r_s = rsqrt(ss_s / H + 1e-8), out = sum_s g_s * v_s * nw * r_s in fp32, one bf16 16-byte store x2; a token with no local slot stores zeros. Same math and eps as master's _post_norm_combine_kernel. PDL: kernel B is launched from the TU's host wrapper with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 on at::cuda::getCurrentCUDAStream(), so its launch and prologue overlap the w2 Marlin tail inside the graph (Marlin never triggers, so the implicit trigger at its block exits applies; griddepcontrol.wait guarantees y is visible). PDL is a separate flag state['pdl_ok'], set only if the prepare() graph-capture test below passes with it; otherwise kernel B launches plainly. kernel_fn at num_tokens > 32 and state['cuda_glue_ok']: block_size_m exactly as master (SGLang pick, then two-smaller at >= 128); kernel A replaces _align_parallel; master's Marlin w13, silu_and_mul, Marlin w2 unchanged; kernel B replaces fused_post_expert_norm_weighted_sum and writes into a torch.empty((M, H), bf16) output exactly as master's small-M branch allocates its out (the returned tensor; no other allocation added, no .item()/.cpu()/sync, branch on shape and prepare-time flags only). SELF-CHECK in prepare() (untimed, fixed torch.Generator, synthetic inputs only, nothing derived from any call's data): (1) align: M in (33, 57, 120, 128, 256, 257, 512) x every bsm kernel_fn can pick at that M (incl. the two-smaller pick) x local_start in (0, 32, 224) x {random, skewed, all-remote, all-one-expert, all-local} (the same generators as master's _align_self_check): require torch.equal of local_ids, sorted_ids[:cap], block_experts[:n_blocks] and n_padded against master's _align_parallel on the same ids. Do NOT gate on master's _align_self_check return or its 'capacity vs SGLang' line (LESSONS: it always reports capacity mismatches); compare to _align_parallel directly. (2) combine: M in (33, 128, 256, 257) x the same routings, y = random bf16 [M, 8, H] with every REMOTE row filled with NaN (proves remote rows are never read), real bank post_norm_w: compare kernel B against an fp32 torch reference of the formula (per-row rel L2 <= 4e-3, global <= 2e-3, no NaN/Inf) and report its rel L2 against fused_post_expert_norm_weighted_sum on the same y with remote rows zeroed (info). (3) end to end: M in (128, 256, 257) random + skewed, real weights, synthetic x: the full new kernel_fn path vs master's path, per-row rel L2 <= 1e-2. (4) capture each of (1)-(3)'s kernel launches inside torch.cuda.graph and replay 3 times, with PDL on; if the PDL capture or replay raises or mismatches, retry without PDL and set pdl_ok False. FALLBACK: any load_inline exception, kernel launch error, mismatch or capture error sets state['cuda_glue_ok'] = False and every M runs master byte for byte. LOGGING, loud in both outcomes: '[cuda-dispatch-combine] build OK/FAILED (<s> s)', 'SELF-CHECK PASSED/FAILED' with the failing case and max rel L2, numRegs and localSizeBytes of both kernels (cudaFuncGetAttributes), and 'PATH CHOSEN: cuda align+combine (pdl on|off) for num_tokens > 32' or 'PATH CHOSEN: master (reason)'. STAGES: stage 1 is this file; stage 2 (if M3 holds) folds kernel B's sum of squares into whichever w2 GEMM rewrite lands, or extends kernel A+B to bs<=32 if the midm rewrite is dropped; whichever GEMM rewrite lands at bs>=128 can take kernel A's outputs unchanged (same format as moe_align_block_size). Scope: two small kernels, ~250 lines of CUDA + ~150 lines of Python; about 1-1.5 h.

## Expected

- `bs=256 *` faster: verify_32 (454.3 us median in 45eacf4ca2e849d2): replaced rows _align_parallel_kernel 6.9 us + _post_expert_norm_weighted_sum_kernel 6.0 us = 12.9 us. Floor for the two: align moves ~8 KB of ids (launch-bound, ~1.5 us), combine reads the local y rows plus 2 MB of output (~1.5-3.5 us at 4.8 TB/s depending on local routes): ~3-5 us. Targets: align <= 2.0 us, combine <= 3.0 us (M3), ~8 us off the shape (~1.8%), plus up to ~1 us of graph gap from PDL
- `bs=128 *` faster: verify_16 (318.3 us median): replaced rows _align_parallel_kernel 4.0 us + _post_expert_norm_weighted_sum_kernel 4.9 us = 8.9 us vs a ~2.5-4 us floor; targets align <= 1.8 us, combine <= 2.5 us (M3), ~4.5-5 us off the shape (~1.5%)
- `bs=32 *` flat: control: num_tokens <= 32 runs master byte for byte
- `bs=16 *` flat: control: num_tokens <= 32 runs master byte for byte
- `bs=8 *` flat: control: num_tokens <= 32 runs master byte for byte
- `bs=1 *` flat: control: untouched

## Risk

Small absolute stakes: the replaced rows are 8.9 us (bs=128) and 12.9 us (bs=256), so the headline effect is ~+0.5-0.7% (2 of 6 loads at ~1.5-1.8%), below the 1.5% CI: judge ONLY on the replaced kernels' top_kernels rows and the per-shape paired ratios, and read the PATH CHOSEN line first. This is still the largest gap left untouched by the four in-flight moe runs (every Marlin and silu row at every M is owned by them) and the queued midm rewrite (bs 2-32 glue). Attribution: the in-flight wgmma-rega run replaces Marlin+silu at bs>=128 but keeps master's align/post-norm, so the two touch disjoint kernels on the same shapes and can stack by 3-way merge. Risks: (a) PDL in stream capture needs CUDA 12.3+ (judge torch 2.14.1+cu130, so expected fine), guarded by its own flag and capture test; (b) Marlin may read sorted_ids.size(0), so scratch views must have exactly master's capacity, which the torch.equal self-check covers; (c) the combine's fp32 (1+w) differs from SGLang's bf16-rounded norm_w by design, as master's small-M combine already does and passes; (d) post-norm-triton-all-m (one Triton program per token at every M) was neutral, so the gain must come from issuing all slot loads before the reduction and one barrier per token, not from the launch alone; if the combine row does not drop below 4 us, that is the finding.

## Milestones

Rewrite stage 1 of `_align_parallel_kernel + fused_post_expert_norm_weighted_sum at num_tokens > 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T183415Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound, rules-clean stage-1 rewrite of the two latency-bound glue kernels at bs>32 (8.9 us at bs=128, 12.9 us at bs=256 in 45eacf4ca2e849d2) whose milestones can be read from the per-kernel profile rows; no companion is needed, but the replaced kernels' times should be logged directly, because a 2 us kernel can drop out of the profile's top-N list.

### Justification
Plausible on H100. _align_parallel_kernel is 33 Triton programs. Each one streams all n ids and builds a [512,32] one-hot per chunk, so it is compute/latency bound at 4.0/6.9 us for a job that moves ~8-16 KB. A single 1024-thread CTA using __match_any_sync ranks, a per-tile count table in shared memory (<=128x32) and one warp-0 scan should finish in ~1.5-2.5 us. The serial warp-0 scan over up to 64-65 tiles per expert is the longest part, at roughly 1 us of shared-memory round trips. The combine (_post_expert_norm_weighted_sum_kernel, 4.9/6.0 us in the parent profile, 6.8 us at bs=256 in the current-best profile) reads local y rows that Marlin w2 just wrote and that are mostly still in L2, then writes 1-2 MB, so its floor is ~1.5-3 us. With M CTAs x 256 threads, all slot loads issued before the reduction and one barrier, the 2.5-3 us target is reachable. Holding up to 8 slots x 32 B per thread as packed bf16 is ~64 registers, which is fine; converting everything to fp32 up front would double that. The localSizeBytes log will catch any spills. The expected directions match TASK.md. Only bs=128 and bs=256 change, each by ~1.5-1.8% of the shape, so ~0.5% of the headline, which is inside the +-1.5% CI. The hypothesis says this itself and correctly asks to be judged on the replaced rows. The bs<=32 shapes are untouched controls (bs=32 still uses master's SGLang combine and _align_small). Weakest link 1, readability: the profile lists only the top ~3-4 kernels per shape. At bs=128, act_and_mul is 3.8 us, so a kernel A that hits its 1.8 us target will drop out of the visible rows. M2/M3 for kernel A then cannot be confirmed from top_kernels, only inferred from its absence. The build should name the kernels distinctly, and the prepare() log should print a CUDA-event median for kernel A vs _align_parallel and kernel B vs fused_post_expert_norm_weighted_sum at M=128/256 (untimed, synthetic). Weakest link 2: the PDL gain is unmeasurable (<=1 us). Marlin never calls launch_dependents, so the implicit trigger only fires at grid exit and the overlap is at most launch latency plus kernel B's prologue. Treat PDL as free if the capture test passes, not as part of the claim. Prior evidence against the combine: post-norm-triton-all-m (one Triton program per token at every M) was within-ci. The idea that issuing all loads first and using a single barrier beats SGLang's kernel is the real bet, and the hypothesis names 'combine row not below 4 us' as the finding if it fails.

### Confounders on the current kernel
- align-parallel-per-expert-verify: this is the incumbent being replaced (4.0/6.9 us), and kernel A must reproduce its sorted_ids byte for byte. Its numbers are the comparison, not a confound: the profile row comparison is direct and nothing else in the launch sequence changes.
- marlin-block-m-two-smaller-on-postnorm: sets bsm to 16 (bs=128) and 32 (bs=256/257), which fixes cap and n_blocks for kernel A and the padding fill. Kernel A's cost barely depends on bsm, and Marlin's time is unchanged because sorted_ids are identical, so no confound. The self-check covers every bsm kernel_fn can pick, including two-smaller.
- post-expert-norm-triton-small-m: introduced the fp32 (1+w) combine that kernel B copies, and the remote-row-never-read convention, which relies on Marlin w2 leaving remote y rows unwritten. It only applies below 32 tokens, so it does not interact at bs>32. Its sibling post-norm-triton-all-m (neutral at large M) is the prior that kernel B has to beat.
- sglang-marlin-path-parity / triton-map-align-on-block-m-restack: these define the Marlin calls and the bs<=32 path, which this change leaves byte-identical. No interaction with the replaced kernels beyond what is listed above.

### Interference and A/B needs
No companion is warranted. The change replaces two specific kernels on two shapes and leaves every other launch identical (same sorted_ids, so Marlin does identical work), so the per-kernel profile rows attribute kernel A and kernel B separately within one run. Reverting a landed hypothesis would not make a gain or loss more readable: the only landed change that shapes kernel A's inputs is the bsm pick, and kernel A's cost is nearly bsm-independent. A PDL-off control would isolate <=1 us, which is below any per-shape resolution, so it is not worth a check. The pdl_ok log line already tells you which path ran. The one real gap, kernel A falling out of the top-N profile rows, is fixed better by prepare()-time event timing in the log than by an extra run.

### Open questions
- Does the profile's top_kernels list show enough rows per shape to see kernel A once it drops below act_and_mul (3.8 us at bs=128)? If not, log prepare()-time CUDA-event medians for A vs _align_parallel and B vs fused_post_expert_norm_weighted_sum at M=128/256, so M2/M3 can be read.
- Is moe_wna16_marlin_gemm (w2, use_atomic_add=False, use_fp32_reduce=True) exactly one kernel launch in the captured graph, so that PDL's preceding kernel is the Marlin GEMM itself?
- Does the judge's torch 2.14.1+cu130 stream capture keep the programmatic-dependency edge from cudaLaunchKernelEx (it should on CUDA >= 12.3), or does it silently drop it to a full dependency? The capture test checks correctness only, not that overlap actually happens.
- If the in-flight cuda-nvfp4-wgmma-rega-grouped-verify lands first, the kernel before kernel B changes. Re-check the PDL assumptions (trigger and number of launches) on restack.

