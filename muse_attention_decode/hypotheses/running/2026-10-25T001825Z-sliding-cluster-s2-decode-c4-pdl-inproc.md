---
problem: muse_attention_decode
slug: sliding-cluster-s2-decode-c4-pdl-inproc
created: 2026-10-25T001825Z
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
  compared to the reference store, graph replay) run IN-PROCESS (never in a child process: the
  rules statically reject submissions that start processes and okc_lint flags multiprocessing
  (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log
  torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only
  compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for
  sliding_cluster_kernel; anything else turns the path OFF), then a tiny synthetic canary launch
  of sliding_cluster_kernel per enabled C (synthetic B=1, short lengths), each followed by
  torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any
  failure keeps master for that C (all of it if the build fails), printing 'PATH CHOSEN: sliding
  master (fallback: <reason>)'; success prints 'PATH CHOSEN: sliding cluster v2 C=<C> B=<range>'
  per batch class. Full layers and page-64 tests keep master. The first measurement of this
  hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered
  in-process).
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
review: proceed
status: running
review_at: 2026-10-25T002143Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T002143Z
base_sha: 593859d3ce482d60c7bb0ec1fb4e4101e7b1ce31
base_blob: bd96c94a12cf87fdcdac119d1e0508f009c4120c
branch: hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc
worktree: .worktrees/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc
mode: null
restack_of: null
stale_base: true
---

# sliding-cluster-s2-decode-c4-pdl-inproc

## Change

Stage 2 of the sliding-layer cluster rewrite, built on the stage-1 code (branch hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch, impl_commit 593859d, rebased onto current master; stage 1 regressed so it is not on master). Per REWRITE_READ_2026-10-25 section 3, three changes to sliding_cluster_kernel, decode only; rename the module muse_sliding_cluster_v2. (1) VERIFY OFF: _slide_gate requires q_len == 1; do not instantiate the L = 8 template at all (shorter build), so every sliding verify shape runs master's FA3 path byte for byte. (2) DECODE_32 ONE WAVE: config-only cluster size _slide_clusters(B) = 16 at B == 1, 8 at 2 <= B <= 16, 4 at 17 <= B <= 32; B > 32 leaves the cluster path (master). At C = 4 each rank owns ck = round_up(ceil(count/4), 16) ~ 528 keys; with (3)'s two half-buffers that is <= 2 x 264 x 272 B ~ 144 KB smem, 1 CTA/SM, 128 CTAs at B = 32. At import, call cudaOccupancyMaxActiveClusters for each (C, smem) instantiation and print it; a C whose max is below the largest batch it serves (C=4 needs >= 32, C=8 >= 16) is disabled for those batches (fall back to master Triton for them) with a printed reason, so no gated shape can run two waves. (3) PDL + SPLIT GATHER: launch with cudaLaunchKernelEx adding cudaLaunchAttributeProgrammaticStreamSerialization = 1 beside the cluster-dim attribute (graph-capturable). In-kernel, BEFORE griddepcontrol.wait do only input-independent work: smem carve-up, the cluster/DSMEM barrier init (barrier.cluster arrive/wait), per-lane RoPE inv_freq constants; then griddepcontrol.wait; then the page-table cp.async, and the chunk gather issued as TWO cp.async groups (first half keys, second half keys) into two smem buffers; q norm/RoPE/fp8 and (last rank) the new k/v build run while both fly; cp.async.wait_group 1 -> compute the first half, wait_group 0 -> compute the second half; issue griddepcontrol.launch_dependents right after the gather is issued (safe: dependents still wait for this grid's completion). Merge across ranks over DSMEM and the fused norm-gate epilogue as stage 1. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only), -lineinfo, print ptxas regs/smem/local; the first launch and the self-check (stage 1's L=1 cases at B in {1,2,16,17,32}, window 2048, lens < window and < 8 + window, kv_scale 1 and 0.25, cache rows compared to the reference store, graph replay) run IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for sliding_cluster_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of sliding_cluster_kernel per enabled C (synthetic B=1, short lengths), each followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any failure keeps master for that C (all of it if the build fails), printing 'PATH CHOSEN: sliding master (fallback: <reason>)'; success prints 'PATH CHOSEN: sliding cluster v2 C=<C> B=<range>' per batch class. Full layers and page-64 tests keep master.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*layer=sliding load=decode_32*` faster: master Triton _sliding_fused_kernel 22.1 us, stage-1 cluster 22.2 (two waves: 32 clusters at C=8 vs 30 co-resident), floor 3.29. M1: tests pass, one wave (printed max clusters >= 32 at C=4). M2: < 22 us (else DEAD per the read). M3 (read's target): <= 12 us
- `*layer=sliding load=decode_16*` faster: master 12.9 us, stage 1 13.0, floor 1.65. M2: < 12.9. M3 (read's target): <= 10 us via PDL prologue overlap + split gather
- `*layer=sliding load=decode_1 *` faster: master 10.4 us, stage 1 10.0, floor 0.11; dependency-chain bound. Milestone: <= 10.0 (no regression), PDL may take ~1 us
- `*layer=sliding load=verify_*` flat: L=8 gated off; master FA3 (stage 1 lost 0.48-0.69x there)
- `*layer=full*` flat: untouched

## Risk

supersedes sliding-cluster-s2-decode-c4-pdl, whose change requires a child-process self-check

## Milestones

Rewrite stage 2 of `sliding layer decode at tp8/page 1, 1<=batch<=32: master Triton _sliding_fused_kernel (stage-1 sliding_cluster_kernel<1>)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002143Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage 2 with no rules problem, but as specified it may hit its own fallback on decode_32 and its PDL part does nothing. The spec's ~144 KB smem at C=4 forces 1 CTA/SM. That gives the same per-GPC slot count as stage 1's C=8 at 2 CTAs/SM, which measured 30 max active clusters, below the 32 needed, so decode_32 would likely fall back to master. Size C=4 for 2 CTAs/SM (<= ~113 KB), and expect no gain from PDL, because the cluster kernel is the only kernel node in the sliding graph.

### Justification
Mechanism. FINDINGS and REWRITE_READ section 3 support stage 2 for decode only. On master, sliding decode_1/16/32 is one Triton _sliding_fused_kernel at 10.4/12.9/22.1 us against floors of 0.11/1.65/3.29 us. Stage 1's cluster kernel measured 10.0/13.0/22.2 us. decode_32 did not scale from decode_16 because of two waves: 32 clusters of C=8 against 30 co-resident. Bytes are small: 32 x 2048 x 2 heads x 128 B = 16.8 MB, about 3.5 us. So a one-wave decode_32 at <= 12 us is physically plausible, and so is decode_16 <= 10 us from a split gather that overlaps compute with the second half of the gather. Gating verify off is right, since stage 1 lost 0.48-0.69x there. The expected directions match the TASK.md weights: sliding shapes weigh 46 each in all three decode loads.

WEAKEST LINK 1, the occupancy arithmetic of the one-wave fix. The spec sizes C=4 at '2 x 264 x 272 B ~ 144 KB smem, 1 CTA/SM'. A cluster must sit inside one GPC. With 1 CTA/SM, max active C=4 clusters is sum over GPCs of floor(SMs_g/4). Stage 1 ran C=8 at 2 CTAs/SM (the brief required fitting 2/SM), which gives floor(2*SMs_g/8). That is the same number, and it was measured at 30. So cudaOccupancyMaxActiveClusters for C=4 at 144 KB will very likely return about 30, not >= 32. The spec's own guard then disables C=4 for B=17..32, and decode_32, the stage's main target with the 'DEAD if not < 22 us' milestone, silently runs master. The check would teach nothing about decode_32. Fix in the build: keep the stage-1 aliasing (fp32 partials over consumed K, V widened per fragment in registers) and store only the raw fp8 K/V. That is 128 B/key x 528 keys ~ 66 KB plus q/partials, about 70-90 KB, which fits 2 CTAs/SM and should allow about 60 clusters. The two 'half-buffers' are just halves of the same 528 keys, so the split gather needs no extra smem. Also, as a fallback before reverting to master, try the non-portable size or C=2.

WEAKEST LINK 2, PDL is inert on the timed shapes. Stage 1's profile shows sliding_cluster_kernel as the only kernel on every sliding shape: pre-attention and norm-gate are fused into it. So inside the captured graph it has no upstream kernel node. Programmatic edges do not cross graph launches or the judge's L2-flush read. griddepcontrol.wait returns at once, and launch_dependents has no dependents. The decode_16 M3 claim ('via PDL prologue overlap') and the decode_1 '~1 us from PDL' should be credited to the split gather alone, and decode_1 should be expected flat. It is harmless to keep, since it is graph-capturable and a no-op, but do not read a decode_16 gain as a PDL result.

Minor: decode_1 is labelled 'faster' but its milestone is 'no regression'. At B=1 the bound is the dependent chain (index -> gather -> compute -> DSMEM merge), and none of the three changes shortens that chain, so flat is the honest expectation. Reaching 1.5x of the floor is out of reach at every shape, but rewrites are judged on the incumbent and the trajectory, and the decode_16/32 targets are reachable.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: the incumbent being replaced (Triton _sliding_fused_kernel at 16 splits on batched decode) and also the fallback target. If the C=4 guard trips, decode_32 runs this kernel unchanged, and a flat decode_32 would wrongly read as 'C=4 did not help'. Read the printed max-clusters and PATH CHOSEN lines before interpreting decode_32.
- sliding-fused-triton-decode1-gate: the B=1 incumbent (10.4 us). The cluster kernel at C=16 already matched it in stage 1 (10.0), so a decode_1 delta within about 0.5 us is noise, not this stage's change.
- fused-preattention-triton: on the sliding cluster path the cluster kernel does its own q/k norm, RoPE, fp8 and cache store, while verify (now gated off) keeps master's _pre_attention_kernel plus FA3. There is no timing confound, but the cache-row self-check must compare against this landed kernel's store convention.
- full-decode-cuda-gather-splitkv: a separate load_inline module compiled in the same process (~41 s), plus ~45 s for this module. That is fine for the 15-minute budget, but a build failure here must not disturb the splitkv module (keep the modules separate, as stage 1 did). No timing interaction: full shapes are untouched.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite, judged on the replaced kernel's own time per shape from profile top_kernels, and the profile plus the import-time log (max active clusters per C, PATH CHOSEN per batch class, ptxas regs/smem) identify which path ran on each sliding decode shape. No landed hypothesis is reverted or tuned by this change. The verify shapes go back to master's FA3 byte for byte, and full layers are untouched. A PDL-off control would measure a no-op (no upstream kernel node in the graph). A C=8-vs-C=4 A/B is also not needed, because stage 1 already measured C=8 at decode_32 (22.2 us, two waves). The real risk to interpretability is the occupancy guard falling back on decode_32, and that is fixed in the build (size for 2 CTAs/SM), not by an extra run.

### Open questions
- What does cudaOccupancyMaxActiveClusters return for C=4 at the real smem size? If it is about 30 (the same as stage 1's C=8 at 2 CTAs/SM), decode_32 falls back. Does the build fit C=4 at 2 CTAs/SM (<= ~113 KB smem, <= 128 regs at the CTA size)?
- Is anything other than the cluster kernel in the captured sliding-decode graph (an output memset, a counter zeroing)? If not, PDL has nothing to overlap, and decode_16/decode_1 gains must be credited to the split gather.
- At C=4, does one CTA per 528-key chunk issue enough bytes in flight (about 135 KB per SM) to approach per-SM bandwidth, or is a deeper cp.async pipeline (more than 2 groups) needed in stage 3?
- decode_1 at C=16 stays chain-bound at ~10 us. Which phase (page-table index load, gather, DSMEM merge, epilogue) dominates? Add clock64 per-phase stamps in the self-check so stage 3 can target it.

