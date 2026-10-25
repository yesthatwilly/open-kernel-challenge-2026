# muse_attention_decode · 2026-10-25T003616Z · check · sliding-cluster-s2-decode-c4-pdl-inproc

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T001825Z-sliding-cluster-s2-decode-c4-pdl-inproc.md` (branch `hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc`).

- **Job:** `1990ce53e9d34073` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `938dfeb`

## Goal
Stage 2 of the sliding-layer cluster rewrite, built on the stage-1 code (branch hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch, impl_commit 593859d, rebased onto current master; stage 1 regressed so it is not on master). Per REWRITE_READ_2026-10-25 section 3, three changes to sliding_cluster_kernel, decode only; rename the module muse_sliding_cluster_v2. (1) VERIFY OFF: _slide_gate requires q_len == 1; do not instantiate the L = 8 template at all (shorter build), so every sliding verify shape runs master's FA3 path byte for byte. (2) DECODE_32 ONE WAVE: config-only cluster size _slide_clusters(B) = 16 at B == 1, 8 at 2 <= B <= 16, 4 at 17 <= B <= 32; B > 32 leaves the cluster path (master). At C = 4 each rank owns ck = round_up(ceil(count/4), 16) ~ 528 keys; with (3)'s two half-buffers that is <= 2 x 264 x 272 B ~ 144 KB smem, 1 CTA/SM, 128 CTAs at B = 32. At import, call cudaOccupancyMaxActiveClusters for each (C, smem) instantiation and print it; a C whose max is below the largest batch it serves (C=4 needs >= 32, C=8 >= 16) is disabled for those batches (fall back to master Triton for them) with a printed reason, so no gated shape can run two waves. (3) PDL + SPLIT GATHER: launch with cudaLaunchKernelEx adding cudaLaunchAttributeProgrammaticStreamSerialization = 1 beside the cluster-dim attribute (graph-capturable). In-kernel, BEFORE griddepcontrol.wait do only input-independent work: smem carve-up, the cluster/DSMEM barrier init (barrier.cluster arrive/wait), per-lane RoPE inv_freq constants; then griddepcontrol.wait; then the page-table cp.async, and the chunk gather issued as TWO cp.async groups (first half keys, second half keys) into two smem buffers; q norm/RoPE/fp8 and (last rank) the new k/v build run while both fly; cp.async.wait_group 1 -> compute the first half, wait_group 0 -> compute the second half; issue griddepcontrol.launch_dependents right after the gather is issued (safe: dependents still wait for this grid's completion). Merge across ranks over DSMEM and the fused norm-gate epilogue as stage 1. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only), -lineinfo, print ptxas regs/smem/local; the first launch and the self-check (stage 1's L=1 cases at B in {1,2,16,17,32}, window 2048, lens < window and < 8 + window, kv_scale 1 and 0.25, cache rows compared to the reference store, graph replay) run IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for sliding_cluster_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of sliding_cluster_kernel per enabled C (synthetic B=1, short lengths), each followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any failure keeps master for that C (all of it if the build fails), printing 'PATH CHOSEN: sliding master (fallback: <reason>)'; success prints 'PATH CHOSEN: sliding cluster v2 C=<C> B=<range>' per batch class. Full layers and page-64 tests keep master. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
high
supersedes sliding-cluster-s2-decode-c4-pdl, whose change requires a child-process self-check

## Result (after the run)
- job `1990ce53e9d34073` (check), status done, passed True, tests 16/16
- speedup 1.4001x, 95% CI [1.3791, 1.4258], score 2689.969288390816 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/regression/sliding-cuda-cluster-dsmem-onelaunch.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch`; branch `hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc`
