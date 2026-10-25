---
problem: muse_attention_decode
slug: sliding-cluster-s2-decode-c4-pdl
created: 2026-10-25T001451Z
base: master
area: sliding-layer attention (decode, cluster kernel)
change: >
  Stage 2 of the sliding-layer cluster rewrite, built on the stage-1 code (branch
  hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch, impl_commit 593859d, rebased
  onto current master; stage 1 regressed so it is not on master). Per REWRITE_READ_2026-10-25
  section 3, three changes to sliding_cluster_kernel, decode only; rename the module
  muse_sliding_cluster_v2. (1) VERIFY OFF: _slide_gate requires q_len == 1; do not instantiate the
  L = 8 template at all (shorter build), so every sliding verify shape runs master's FA3 path byte
  for byte. (2) DECODE_32 ONE WAVE: config-only cluster size _slide_clusters(B) = 16 at B == 1, 8
  at 2 <= B <= 16, 4 at 17 <= B <= 32; B > 32 leaves the cluster path (master). At C = 4 each rank
  owns ck = round_up(ceil(count/4), 16) ~ 528 keys; with (3)'s two half-buffers that is <= 2 x 264
  x 272 B ~ 144 KB smem, 1 CTA/SM, 128 CTAs at B = 32. At import, call
  cudaOccupancyMaxActiveClusters for each (C, smem) instantiation and print it; a C whose max is
  below the largest batch it serves (C=4 needs >= 32, C=8 >= 16) is disabled for those batches
  (fall back to master Triton for them) with a printed reason, so no gated shape can run two
  waves. (3) PDL + SPLIT GATHER: launch with cudaLaunchKernelEx adding
  cudaLaunchAttributeProgrammaticStreamSerialization = 1 beside the cluster-dim attribute
  (graph-capturable). In-kernel, BEFORE griddepcontrol.wait do only input-independent work: smem
  carve-up, the cluster/DSMEM barrier init (barrier.cluster arrive/wait), per-lane RoPE inv_freq
  constants; then griddepcontrol.wait; then the page-table cp.async, and the chunk gather issued
  as TWO cp.async groups (first half keys, second half keys) into two smem buffers; q
  norm/RoPE/fp8 and (last rank) the new k/v build run while both fly; cp.async.wait_group 1 ->
  compute the first half, wait_group 0 -> compute the second half; issue
  griddepcontrol.launch_dependents right after the gather is issued (safe: dependents still wait
  for this grid's completion). Merge across ranks over DSMEM and the fused norm-gate epilogue as
  stage 1. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only), -lineinfo,
  print ptxas regs/smem/local; the first launch and the self-check (stage 1's L=1 cases at B in
  {1,2,16,17,32}, window 2048, lens < window and < 8 + window, kv_scale 1 and 0.25, cache rows
  compared to the reference store, graph replay) run in a CHILD PROCESS. Any failure keeps master
  for that C (all of it if the build fails), printing 'PATH CHOSEN: sliding master (fallback:
  <reason>)'; success prints 'PATH CHOSEN: sliding cluster v2 C=<C> B=<range>' per batch class.
  Full layers and page-64 tests keep master.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "master Triton _sliding_fused_kernel 22.1 us, stage-1 cluster 22.2 (two waves: 32 clusters at C=8 vs 30 co-resident), floor 3.29. M1: tests pass, one wave (printed max clusters >= 32 at C=4). M2: < 22 us (else DEAD per the read). M3 (read's target): <= 12 us"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "master 12.9 us, stage 1 13.0, floor 1.65. M2: < 12.9. M3 (read's target): <= 10 us via PDL prologue overlap + split gather"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "master 10.4 us, stage 1 10.0, floor 0.11; dependency-chain bound. Milestone: <= 10.0 (no regression), PDL may take ~1 us"
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: L=8 gated off; master FA3 (stage 1 lost 0.48-0.69x there)
  - shapes: "*layer=full*"
    direction: flat
    why: untouched
risk: high
depends_on: sliding-cuda-cluster-dsmem-onelaunch
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "sliding layer decode at tp8/page 1, 1<=batch<=32: master Triton _sliding_fused_kernel (stage-1 sliding_cluster_kernel<1>)"
review: required
status: queued
---

# sliding-cluster-s2-decode-c4-pdl

## Change

Stage 2 of the sliding-layer cluster rewrite, built on the stage-1 code (branch hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch, impl_commit 593859d, rebased onto current master; stage 1 regressed so it is not on master). Per REWRITE_READ_2026-10-25 section 3, three changes to sliding_cluster_kernel, decode only; rename the module muse_sliding_cluster_v2. (1) VERIFY OFF: _slide_gate requires q_len == 1; do not instantiate the L = 8 template at all (shorter build), so every sliding verify shape runs master's FA3 path byte for byte. (2) DECODE_32 ONE WAVE: config-only cluster size _slide_clusters(B) = 16 at B == 1, 8 at 2 <= B <= 16, 4 at 17 <= B <= 32; B > 32 leaves the cluster path (master). At C = 4 each rank owns ck = round_up(ceil(count/4), 16) ~ 528 keys; with (3)'s two half-buffers that is <= 2 x 264 x 272 B ~ 144 KB smem, 1 CTA/SM, 128 CTAs at B = 32. At import, call cudaOccupancyMaxActiveClusters for each (C, smem) instantiation and print it; a C whose max is below the largest batch it serves (C=4 needs >= 32, C=8 >= 16) is disabled for those batches (fall back to master Triton for them) with a printed reason, so no gated shape can run two waves. (3) PDL + SPLIT GATHER: launch with cudaLaunchKernelEx adding cudaLaunchAttributeProgrammaticStreamSerialization = 1 beside the cluster-dim attribute (graph-capturable). In-kernel, BEFORE griddepcontrol.wait do only input-independent work: smem carve-up, the cluster/DSMEM barrier init (barrier.cluster arrive/wait), per-lane RoPE inv_freq constants; then griddepcontrol.wait; then the page-table cp.async, and the chunk gather issued as TWO cp.async groups (first half keys, second half keys) into two smem buffers; q norm/RoPE/fp8 and (last rank) the new k/v build run while both fly; cp.async.wait_group 1 -> compute the first half, wait_group 0 -> compute the second half; issue griddepcontrol.launch_dependents right after the gather is issued (safe: dependents still wait for this grid's completion). Merge across ranks over DSMEM and the fused norm-gate epilogue as stage 1. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only), -lineinfo, print ptxas regs/smem/local; the first launch and the self-check (stage 1's L=1 cases at B in {1,2,16,17,32}, window 2048, lens < window and < 8 + window, kv_scale 1 and 0.25, cache rows compared to the reference store, graph replay) run in a CHILD PROCESS. Any failure keeps master for that C (all of it if the build fails), printing 'PATH CHOSEN: sliding master (fallback: <reason>)'; success prints 'PATH CHOSEN: sliding cluster v2 C=<C> B=<range>' per batch class. Full layers and page-64 tests keep master.

## Expected

- `*layer=sliding load=decode_32*` faster: master Triton _sliding_fused_kernel 22.1 us, stage-1 cluster 22.2 (two waves: 32 clusters at C=8 vs 30 co-resident), floor 3.29. M1: tests pass, one wave (printed max clusters >= 32 at C=4). M2: < 22 us (else DEAD per the read). M3 (read's target): <= 12 us
- `*layer=sliding load=decode_16*` faster: master 12.9 us, stage 1 13.0, floor 1.65. M2: < 12.9. M3 (read's target): <= 10 us via PDL prologue overlap + split gather
- `*layer=sliding load=decode_1 *` faster: master 10.4 us, stage 1 10.0, floor 0.11; dependency-chain bound. Milestone: <= 10.0 (no regression), PDL may take ~1 us
- `*layer=sliding load=verify_*` flat: L=8 gated off; master FA3 (stage 1 lost 0.48-0.69x there)
- `*layer=full*` flat: untouched

## Risk

Source: REWRITE_READ_2026-10-25 section 3 (PROMISING narrowly, decode only). Kill rule from the read: if decode_32 cannot beat 22 us at C=4 the line is DEAD; L=8 returns only with a clock64 per-phase timing build. Overlaps: sliding verify is owned by sliding-verify-cuda-streamk-window (queued, stacks on full-verify stage 2 A8/B8) - this hypothesis gates verify OFF so the two are shape-disjoint. fused-sliding-preattn-into-window-kernel and fused-sliding-normgate-epilogue (hold, Triton/FA3 era) are already realised inside the cluster kernel for decode. stream-fork-preattention (hold) does not touch this path. Attribution: three changes in one run; read decode_32 vs the printed one-wave cluster count (C=4) and decode_16/decode_1 vs PDL+split gather. PDL gains depend on the preceding graph node; if decode_16 does not move, rerun with the PDL attribute off as an ab-tune companion. Risk: C=4 1-CTA/SM co-residency on H100 GPCs may give < 32 clusters; the import check then disables C=4 and decode_32 reads flat (master), which is a valid DEAD signal, not a bug.

## Milestones

Rewrite stage 2 of `sliding layer decode at tp8/page 1, 1<=batch<=32: master Triton _sliding_fused_kernel (stage-1 sliding_cluster_kernel<1>)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
