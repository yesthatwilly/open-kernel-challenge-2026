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
status: done
review_at: 2026-10-24T183415Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T183415Z
base_sha: b17562ed4f8acdd2a6e2e98f542acc2bba35a2b4
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/cuda-dispatch-combine-pdl-verify
worktree: .worktrees/muse_moe_w4a16/cuda-dispatch-combine-pdl-verify
mode: submit
implemented_at: 2026-10-24T184624Z
impl_commit: 38270a8e28e9d5fe4920187220679aa4b35ab074
submission_sha256: 7c7485b51531cb6fab31bedfb3970593214e591f0da8f35d9f6ca078199a504d
norm_sha256: b26f9d20e2ac256a87e1d24b2de921b62310f7690a72d7f03c787fdd42a80a2e
submission_blob: 005e9bed2e9971ae7413b3fe0c398f7241911804
lint_warnings: 19
attempt_file: 2026-10-24T184624Z-submit-cuda-dispatch-combine-pdl-verify.md
prepared_at: 2026-10-24T184624Z
job: c41860068fe4432f
dispatched_at: 2026-10-24T184641Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3282656846073913
speedup_ci: [1.3083416993382804, 1.348189669876502]
harvested_at: 2026-10-24T232005Z
best_at_harvest: 1.3278240305297715
promotion: null
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


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/cuda-dispatch-combine-pdl-verify`, file `muse_moe_w4a16/submission.py` only (+733/-2, uncommitted).

- New block after `_align_self_check`: one load_inline TU (`muse_moe_cuda_dispatch_combine_v1`, `-O3 -arch=sm_90a`, no fast-math, build/fallback scaffolding copied from master's muse_attention_decode `_build_splitkv` and the PDL `cudaLaunchKernelEx` pattern from muse_decode_layer small-t-launch-minimal-dense-chain-pdl). It is built in prepare() and the build time is printed as `[cuda-dispatch-combine] build OK/FAILED (<s> s)`.
  - `cdc_align_cuda_kernel<IdT>` (kernel A): grid (1,), 1024 threads, int32/int64 ids. The spec's pass 1 uses `__match_any_sync` ranks per 32-route tile and a `cnt[128][32]` table in shared memory. Deviation: the spec has warp 0 scan serially over 128 tiles. That is replaced by a 3-step parallel scan: 32 tile-groups of 4 tiles each, then a warp-per-expert `__shfl_up` scan over the groups, then per-tile bases. Same outputs, far fewer dependent shared-memory round trips. Warp 0 then scans the padded segments. Pass 2 writes the real routes, then the per-expert padding, the `[total, cap)` sentinel, `block_experts` (the expert is found by counting `seg_end <= b*bsm`), and `n_padded`.
  - `cdc_postnorm_combine_cuda_kernel` (kernel B): grid M, 256 threads, 2 x 16 B vectors per thread at cols `[8t, 8t+8)` and `[2048+8t, ...)`. The prologue loads (1+w) fp32, the ids and the weights, then runs `griddepcontrol.wait`. After the wait, every local-slot load is issued with `ld.global.nc.L1::no_allocate.v4`. The 8-vector is reduced with warp shuffles, followed by one shared round and one `__syncthreads`. The kernel computes `rsqrtf(ss/H + 1e-8)` and does the fp32 accumulation. Remote slots are never loaded. A token with no local slot gets zeros. The host wrapper uses `cudaLaunchKernelEx` with programmatic stream serialization when `pdl=1`.
  - Distinct kernel names (`cdc_align_cuda_kernel`, `cdc_postnorm_combine_cuda_kernel`) are used so the profile rows can be attributed (reviewer note).
- `_cdc_align`: writes into prepare()-allocated scratch (`local_ids[4096]`, `sorted_ids[4096+32*63]`, `block_experts[764]`, `n_padded[1]`) and returns exact-size views `[:n]`, `[:cap]` and `[:n_blocks]`. Every element of each view is written by every launch.
- kernel_fn: `use_cdc` is true when the expert_map exists, `state['cuda_glue_ok']` is set, 32 < M <= 512, H == 4096, count <= 32, and the topk_ids dtype is int32/int64 (shape, dtype and flags only). Under `use_cdc`, kernel A replaces `_align_parallel`, and kernel B with `pdl_ok` replaces `fused_post_expert_norm_weighted_sum`, writing into `torch.empty((M, H), bf16)`. Marlin w13, silu_and_mul and Marlin w2 are untouched. With `use_cdc` false the code is master's code, byte for byte, which covers M <= 32 and every fallback.
- prepare(): master's state dict, then `_cdc_setup` (wrapped in try, never raises). The setup sequence is:
  1. print numRegs/localSizeBytes/smem for both kernels;
  2. (1) align: M in (33, 57, 120, 128, 256, 257, 512) x {SGLang bsm, two-smaller bsm at >= 128} x local_start (0, 32, 224) x 5 routings x ids {int32, int64}. Each case uses `torch.equal` of all four outputs against master's `_align_parallel`, with the scratch prefilled with -7 so stale data is caught;
  3. (2) combine at pdl=0: M in (33, 128, 256, 257) x 5 routings, remote y rows NaN, compared against an fp32 torch reference (per-row 4e-3, global 2e-3, finite). It also prints the rel L2 against SGLang's fused combine as info;
  4. (3) e2e eager: M in (128, 256, 257) x {random, skewed}, real weights, comparing the full new kernel_fn against master's kernel_fn (per-row 1e-2);
  5. (4) graph: the combine check eagerly with PDL, then the e2e check captured in `torch.cuda.graph` and replayed 3x, all at pdl=1. If that fails or raises, the same graph e2e runs at pdl=0. The pdl=1 graph run is the one that sets `pdl_ok`.
  
  After the checks it prints `SELF-CHECK PASSED/FAILED: <case, rel L2>` and per-launch CUDA-event medians (graph of 20, warm L2) of A vs `_align_parallel` and B vs `fused_post_expert_norm_weighted_sum` at M=128/256. The reviewer asked for these so the rows stay readable even if they drop out of the top-N. The setup ends with loud `===== PATH CHOSEN: ... =====` lines: cuda align+combine (pdl on|off) for 32 < M <= 512, or master (reason), plus master for M <= 32 and M > 512.
- Deviations: (a) the parallel 3-step scan described above, not the warp-0 serial scan; (b) step (4) captures the end-to-end path, which holds both new kernels after the real Marlin predecessor, rather than each check's launches separately; (c) kernel B reads (1+w) from the bf16 post_norm_w directly, as the spec says.
- Uncertainties: nvcc is not on this node, so the CUDA was not compiled here. If the build fails, the run falls back to master with a loud FAILED line. Whether PDL forms a real programmatic edge after Marlin in capture is unverified, since the capture test checks correctness only.

## Rules check

- [x] CUDA-graph capturable: kernel_fn and `_cdc_align` have no `.item()`/`.cpu()`/`synchronize()`/tensor prints. Branches use only shape, dtype and prepare-time flags. The only allocation in the new path is `out`, which is master's own small-M pattern.
- [x] Nothing derived from `data` is cached across calls: the scratch views are fully overwritten by every kernel A launch before they are read, so no earlier call's data is ever read. The self-check fills them with -7 to prove this.
- [x] Weights: unchanged. Master's Marlin repack is kept. post_norm_w stays bf16 and is read as bf16 with +1 in fp32, as master's `_post_norm_combine_kernel` does.
- [x] Activations BF16 where SGLang keeps BF16: y is read as bf16 and out is bf16. The fp32 math is internal, as in SGLang's combine.
- [x] Attention sees every key: n/a, no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. load_inline is the same compile pattern as the landed CUDA builds, and the lint reports no process/network/harness findings.
- [x] All work happens inside the timed call; no output caching or replay. prepare() runs only untimed synthetic self-checks and timing.

Lint (`okc_lint.py --problem muse_moe_w4a16`): `OK with warnings`. There are 0 errors and 20 `host-sync` warns, all outside kernel_fn's call graph: master's `_align_self_check` (L338-366) and the new prepare()-only `_cdc_routings`/checks/timing (L834-1182). `py_compile` OK and `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-24 by an Opus subagent. I read the code and emulated parts of it in numpy. There is no nvcc or GPU here, so nothing was compiled or run.

**Verdict: dispatch.** No fixes were made, and the worktree file is unchanged.

- **Compile:** I read the TU line by line. It includes torch/extension.h, ATen/cuda/CUDAContext.h, c10/cuda/CUDAException.h and cuda_bf16.h. It uses the following, all valid on sm_90a:
  - `__match_any_sync`;
  - `%lanemask_lt` inline PTX;
  - `ld.global.nc.L1::no_allocate.v4.u32`;
  - `griddepcontrol.wait`;
  - the `cudaLaunchKernelEx` C++ template, whose argument coercion makes `int*` → `const int*` legal, with `attr.val.programmaticStreamSerializationAllowed`;
  - `C10_CUDA_CHECK` / `C10_CUDA_KERNEL_LAUNCH_CHECK`.

  The bf16 code uses intrinsics only (`__float2bfloat16_rn`, `__bfloat16_as_ushort`), so torch's `__CUDA_NO_BFLOAT16_*` flags do not bite. `-arch=sm_90a` in extra_cuda_cflags suppresses torch's own gencode list, the same as the splitkv build that compiled in the judge. The cpp declarations match the definitions, and the pybind arities match every Python call site: cdc_align takes 8 args and cdc_combine takes 7.
- **Build failure path:** a `load_inline` exception is caught in `_cdc_build`. It prints `build FAILED (<s> s)` with the nvcc output in the repr, then `SELF-CHECK FAILED: build failed`, then `PATH CHOSEN: master (build failed)`, and kernel_fn runs master's code byte for byte. `_cdc_setup` is also wrapped in a try in prepare(), which has its own loud master line.
- **Gates:**
  - Kernel A: a numpy emulation of the match/scan/scatter algorithm (M 33-512, bsm 16/32/48/64, local_start 0/32/224, random/one-expert/all-local) is byte-identical to `_align_parallel` in every case. The torch.equal gate can pass.
  - Combine gate: global ≤ 2e-3 vs fp32 ref. The only error source is the final bf16 RNE rounding, and the emulation gives a global rel L2 of 1.66e-3 and per-row ≤ 1.75e-3 (gate 4e-3) at every M and routing. It passes, though with only ~17% margin on the global bound.
  - The e2e 1e-2 per-row bound vs master (bf16 1+w in SGLang's combine) is estimated at ~3-4e-3.
  - All-remote rows give ref = 0 and out = 0, which is handled.
- **Memory safety:** I checked indexing for every shape up to M = 512 (n ≤ 4096 = 128 tiles). The sorted_ids index stays below total ≤ cap, block_experts stays below n_blocks, and the y index stays below M·8·H. Scratch sizes are 4096 / 6112 / 764 / 1. I found no OOB, and an OOB would matter, because a sticky fault in the self-check would kill the run rather than fall back.
- **Rules:**
  - kernel_fn branches on shape, dtype and prepare-time flags only, with no syncs.
  - The scratch views are fully rewritten before they are read on every call.
  - out is master's allocation pattern.
  - Weights are unchanged, and activations are bf16 with fp32 internals.
  - Remote y rows are never loaded.

  Lint is OK with warnings: host-sync warnings in prepare() only. py_compile is OK.
- **PDL:** local_ids and tw are read before the wait. Both are produced by kernels that finished before Marlin w2 started, so this is safe. y is read only after `griddepcontrol.wait`. A capture failure at pdl=1 retries at pdl=0, and the result is logged.

Residual risk: the CUDA has never been compiled, so a toolchain-specific compile error (e.g. cudaLaunchKernelEx template coercion or an asm constraint) would cost the run's gain, but it lands on master with a loud FAILED line rather than failing the run.

## Result

- job `c41860068fe4432f` (submit), status done, passed True, tests 5/5
- speedup 1.3283x, 95% CI [1.3083, 1.3482], score 10124.856913679214 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch`; branch `hyp/muse_moe_w4a16/cuda-dispatch-combine-pdl-verify`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.82 | 50.36 | 1.0045 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 148.37 | 145.72 | 1.0005 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 199.88 | 200.12 | 0.9987 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.74 | 106.42 | 1.0007 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 318.26 | 319.29 | 1.0101 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 454.28 | 457.11 | 1.0007 | flat |

no shape moved outside noise. Expected hit: 4/6. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `_align_parallel_kernel + fused_post_expert_norm_weighted_sum at num_tokens > 32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_moe_w4a16 "<what to fix or tune>" --kind rewrite --depends-on cuda-dispatch-combine-pdl-verify --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: cuda-dispatch-combine-pdl-verify`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

