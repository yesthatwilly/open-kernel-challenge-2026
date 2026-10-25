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
stage: dispatched
claimed_at: 2026-10-25T002143Z
base_sha: 593859d3ce482d60c7bb0ec1fb4e4101e7b1ce31
base_blob: bd96c94a12cf87fdcdac119d1e0508f009c4120c
branch: hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc
worktree: .worktrees/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T003616Z
impl_commit: 938dfebce29caa2acd412dd8ae1f27f930042496
submission_sha256: 50aaf8535c92809fb540f38da73f9dbdb68e5909b91acb8569a383a0e99b879e
norm_sha256: 51e427419bb99b1fec184ef72795f13ffdf01d534fdc62af30f34b12b84c8017
submission_blob: 01024fb7e27a1343414dfbd5fca8c0e5f5449225
lint_warnings: 3
attempt_file: 2026-10-25T003616Z-check-sliding-cluster-s2-decode-c4-pdl-inproc.md
prepared_at: 2026-10-25T003616Z
job: 1990ce53e9d34073
dispatched_at: 2026-10-25T003624Z
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


## Implementation notes

Built on the stage-1 worktree (593859d). Only `muse_attention_decode/submission.py` was edited.

- **Base drift (stale_base).** The worktree predates master's `sliding-deep-splits-on-cuda-splitkv` (0a56d22). I hand-applied master's two hunks so every fallback is master byte for byte: `kernel_fn` sends all sliding decode to `_sliding_forward`, and `_sliding_forward` uses `splits = 16 if length == 1`. Rebasing the branch is still a churnsubs step.
- **(1) Verify off.** `_slide_gate` now requires `q_len == 1`. The kernel is no longer a template: L = 8 is removed, so there are no L = 8 instantiations, and verify keeps master FA3.
- **(2) Cluster size.** `_slide_clusters(B)` is 16 at B = 1 (8 if C = 16 is off), 8 at 2..16, 4 at 17..32, and 0 (master Triton) above 32. Module renamed `muse_sliding_cluster_v2`.
- **Review fix: smem sized for 2 CTAs/SM.** The review's 66 KB figure counts K only. A 512-key C = 4 chunk needs K + V = 512 x 256 B = 128 KB, which cannot fit 2 CTAs/SM, so I did not use two half-buffers. The gather goes through a ring instead:
  - 3 slots of SK = 128 keys, K + V = 96 KB, plus a separate 16-row new-key tile (4 KB) and q (1 KB).
  - Total 103,424 B dynamic smem, with `static_assert <= 113 KB`, `__launch_bounds__(256, 2)` and the smem carveout set to max shared.
  - Each 128-key sub-chunk is one cp.async group. The first 3 are issued at once. The 4th (C = 4 only) goes into slot 0 right after slot 0 is computed; its page indices are prefetched into registers.
  - Compute runs per sub-chunk after `cp.async.wait_group 2`.
  - The spec's "two halves" is the C = 8 case (2 x 128 keys). C = 4 has 4 sub-chunks and C = 16 has 1.
  - Partials alias the consumed ring, as in stage 1.
- **Occupancy guard.** At import, `cudaOccupancyMaxActiveClusters` is printed for C = 4/8/16. A C below what its largest batch needs (C=4 >= 32, C=8 >= 16, C=16 >= 1) is disabled with a printed reason, and its batches run master.
- **PDL: off (review fix).** There is no programmatic-serialization attribute and no griddepcontrol, because the cluster kernel is the only node in the sliding graph. Any decode_16 or decode_1 change should be credited to the split gather and the C choice.
- **Build and safety.**
  - `TORCH_CUDA_ARCH_LIST=9.0a` is set around `load_inline` and restored afterwards, with `-O3 -lineinfo`. `-arch` was dropped so torch emits only the compute_90a/sm_90a gencode, and the arch flags are printed.
  - The kernel uses no sm_90a-only PTX (it uses mma.sync, cluster/DSMEM and plain `cp.async.cg`), so the `__CUDA_ARCH_FEAT_SM90_ALL` guard does not apply.
  - The path turns off unless capability == (9, 0) and `cudaFuncGetAttributes` binaryVersion == 90. Regs, local bytes, smem, binaryVersion and ptxVersion are printed.
  - There are no mbarrier or spin waits, so no wait bounds were needed.
- **Import-time checks, in-process, per C.**
  1. A canary launch (B = 1, 40 keys).
  2. The self-check against the FP32 reference at B = 1 (C16), B = 2 and 16 (C8), B = 17 and 32 (C4). Lengths are below, at and past the window (1 ... 5007), with planted window-edge keys and kv_scale 1 and 0.25.
  3. New cache rows compared with master `_pre_attention_kernel` on cloned caches.
  4. Graph capture plus replay.

  Every launch is preceded by a flushed print and followed by `torch.cuda.synchronize()` and a print.
- **Fallback.** A failed canary or self-check retries that C with the f16-QK build. Failed occupancy does not retry. Any failure leaves master for that C's batches. `_slide_init` is wrapped so it never raises.
- **Log lines.** `[slide] PATH CHOSEN: sliding cluster v2 C=<C> <range>` or `PATH CHOSEN: sliding master (fallback: <reason>) <range>` per batch class.
- **Not done.** The clock64 per-phase stamps from the review's open question.

## Rules check

- [x] CUDA-graph capturable: `kernel_fn` adds only `torch.empty` and one `cudaLaunchKernelEx` (cluster attribute only). The C choice comes from `config["batch"]` and import-time flags. `synchronize` appears only in the import-time canary and self-check.
- [x] Nothing derived from `data` is cached across calls. `_SLIDE_NOTED` holds config keys only, and `_inv_freq` is keyed on theta.
- [x] Weights: n/a. This problem has no weights, and the cache store convention is unchanged.
- [x] Activations BF16 out, KV cache FP8 E4M3 (new rows / k_scale, v_scale, satfinite), queries rounded once to E4M3, fp32 softmax and accumulation.
- [x] Attention sees every key in the window. Each rank's chunk is gathered whole and every tile is masked only by window and causal position. No skipping.
- [x] No timing patches, harness access, reference import, network or process creation (no subprocess, multiprocessing, fork or spawn). `os` is used only for `TORCH_CUDA_ARCH_LIST`.
- [x] All work happens inside the timed call, with no output caching or replay. The self-check runs on synthetic data at import.
- [x] First line `# okc: problem=muse_attention_decode` kept. Diff checked with `git diff`: only `submission.py` changed.

okc_lint:
```
warn  host-sync    L1890: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1931: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1958: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```
py_compile OK; __pycache__ removed. Not committed; no okc run. The first measurement must be an `okc check`.

## Pre-flight audit

Audited 2026-10-24 (opus), by reading plus a Python emulation of the rank partition and the cp.async ring schedule.

**Verdict: dispatch-with-fixes.** The first measurement must still be an `okc check`.

Fix:
- `kernel_fn`: `_slide_forward` is wrapped in `try/except RuntimeError`. A host-side launch error (C10_CUDA_CHECK on cudaLaunchKernelEx) now prints a `PATH CHOSEN: sliding master (fallback: C=<c> launch raised ...)` line once per config and falls through to master `_sliding_forward`, instead of failing the run. The fallback is stateless: no module state is written from kernel_fn. okc_lint OK (only the 3 known prepare-time synchronize warnings), py_compile OK, __pycache__ removed.

Verified with no change needed:
- **Fallbacks match master byte for byte.** `diff` against master's submission.py shows only the docstring, `import os`, the kernel_fn dispatch hunk and the appended sliding block. `_sliding_forward` (splits = 16 at L = 1) and the FA3 and split-KV paths are identical.
- **Every key is attended.** Emulated over C in {4, 8, 16} and seq in 0..90000: the C ranks cover [lo, seq) exactly once. The new key at position seq is attended once, on the last rank, by both kv-head warps of one slice. The mask is `low = seq - 2047 <= kp <= seq`, which matches reference.py (`start = p - window + 1`, `p = seq_lens`, positions = seq_lens).
- **Ring.** ck <= 512, so nsub <= 4/2/1 at C = 4/8/16. Group accounting: 3 prologue commits plus one commit per iteration, so `wait_group 2` at iteration s guarantees groups 0..s. The late sub-chunk 3 is in group 3. The WAR on slot 0 is covered by `__syncthreads` before reissue. Pad rows are zeroed with generic stores before the next barrier. Partials alias the ring only after `wait_group 0` and `__syncthreads`.
- **DSMEM.** One `cluster.sync` after the partials are written and before any remote read, and one before exit.
- **Fragments.** m16n8k32 e4m3 A/B, the C layout, the P->A f16 layout, the V `byte_perm` widening and the staging dim mapping all check out by hand. Ls > 0 always (the own key).
- **Precision.** KV stays FP8. q is rounded once to E4M3 after `qk_scale_factor` (|q| <= ~44, normal range, so it is the same precision class as the reference's rounding before the scale). P and V are f16, with fp32 softmax and accumulation. Output is BF16.
- **Rules.** No child process, subprocess or multiprocessing. Nothing derived from data is cached (`_SLIDE_NOTED` is config keys, `_inv_freq` is keyed by theta, same as master). kernel_fn does `torch.empty` plus one cluster `cudaLaunchKernelEx` on the current stream, with no syncs, so it is graph-capturable and the self-check captures and replays it.
- **Gates.** The self-check's `_pre_attention_kernel` call matches master's signature. Its synthetic shapes (B in {1, 2, 16, 17, 32}, lens up to 5007, edge keys, kv_scale 1 and 0.25) match the gated scored shapes. The 2e-2 tolerance is realistic. 103,424 B smem with `launch_bounds(256, 2)` fits 2 CTAs/SM.

Interpretation:
- PDL is off, as the review advised: the cluster kernel is the only node in the sliding graph. The slug's "pdl" is a misnomer. Credit any decode_16 or decode_1 change to the split gather and the choice of C, not to PDL.
- Before reading decode_32, read the printed `cudaOccupancyMaxActiveClusters C=4` value and the PATH CHOSEN lines. If C = 4 was disabled, decode_32 ran master.

Residual risk: never compiled. A ptxas or `TORCH_CUDA_ARCH_LIST=9.0a` failure, or a C=4 occupancy below 32, silently lands those batches on master. Only the log lines distinguish that from "no gain".
