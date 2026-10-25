---
problem: muse_bf16_proj
slug: restack-router-cluster-dsmem-m8
created: 2026-10-25T042040Z
base: master
area: router GEMM (m<=8)
change: >
  Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x;
  check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch
  hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs,
  swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its
  CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the
  build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing
  any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile
  only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path =
  _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 ->
  return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the
  cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8}
  only (same bounds as the source branch). Everything else byte-identical to the restack. At
  harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the
  restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found
  nine runs that were really master).
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: restack row 5.38 = swapab 3.1 + memset 0.8 + node; cluster read 5.24 on master (-8.7%, z -16)
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: restack row 5.47; cluster read 5.08 on master (-14%, z -31)
  - shapes: layer=router load=decode_16 m=16
    direction: flat
    why: keeps restack swapab
  - shapes: layer=router load=decode_32 m=32
    direction: flat
    why: keeps restack swapab
  - shapes: layer=router load=verify_* m=12*
    direction: flat
    why: m=128 keeps hilo
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: keeps hilo
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T042320Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T042320Z
base_sha: 459e84861e1a2c35ffa8148cd1fe09f5ead28416
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/restack-router-cluster-dsmem-m8
worktree: .worktrees/muse_bf16_proj/restack-router-cluster-dsmem-m8
mode: null
restack_of: null
stale_base: null
---

# restack-router-cluster-dsmem-m8

## Change

Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs, swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path = _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 -> return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8} only (same bounds as the source branch). Everything else byte-identical to the restack. At harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found nine runs that were really master).

## Expected

- `layer=router load=decode_1 m=1` faster: restack row 5.38 = swapab 3.1 + memset 0.8 + node; cluster read 5.24 on master (-8.7%, z -16)
- `layer=router load=verify_1 m=8` faster: restack row 5.47; cluster read 5.08 on master (-14%, z -31)
- `layer=router load=decode_16 m=16` flat: keeps restack swapab
- `layer=router load=decode_32 m=32` flat: keeps restack swapab
- `layer=router load=verify_* m=12*` flat: m=128 keeps hilo
- `layer=router load=verify_32 m=256` flat: keeps hilo
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Kill: router decode_1 and verify_1 rows not both below the restack's 1e90292c rows (5.38/5.47) with the cluster kernel in the profile. Expected headline ~1.224 (03:41Z substitution). Risk: the load_inline build adds ~40 s and an illegal-instruction fault would poison the context; the kernel already ran clean on the judge, so run as a check.

## Adversarial review

Reviewed 2026-10-25T042320Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean m<=8 router-only shape gate onto the landed restack (b99b1dd), and the per-shape router rows can be read without a companion. The m=1 claim is overstated: -8.7% is measured against old master (5.73 us), and against the restack's 5.38 us the cluster's 5.24 us is only -2.6%, close to the 1.5% noise. The headline cannot move outside its CI, so judge the run on the router decode_1 and verify_1 rows only.

### Justification
The mechanism holds up, but it is not the one the hypothesis states. In dc5d03c4 the cluster kernel itself is SLOWER than the restack's Triton kernel. router_kernel<8,8> profiles at 3.6 us (m=1) and 4.3 us (m=8), against _router_swapab_kernel at 3.1 / 3.0 us plus a 0.7-0.8 us memset node. The row still drops (5.24 vs 5.38, 5.08 vs 5.47) because a single graph node with torch.empty replaces the memset-plus-kernel pair, and each node costs about 1 us in a cold-L2 replay. So the gain comes from removing a node, not from a faster stream. That is plausible on H200, and the m=8 row is real (-0.39 us, per-shape speedup 5.18 vs 4.80, err ~20 ns). The two runs being compared were built on different bases (dc5d03c4 on 8b9ad43, the restack on b99b1dd), but SGLang's baselines agree within 1% (10432 vs 10368 ns, 26320 vs 26240 ns), so the cross-run row comparison is acceptable. Weakest link: the decode_1 m=1 expectation. The quoted -8.7% / z -16 is against old master's 5.73 us; against the restack it is 5.24 vs 5.38 = -2.6%, barely above the 1.5% per-shape noise threshold, so a re-measure could easily land flat. Second weakness: headline impact. Router decode_1 and verify_1 are ~5.4 us of ~48-49 us loads (weight 62 each), so 0.14 + 0.39 us is about +0.3% and +0.8% on two of six loads, about +0.18% geomean. Expect roughly 1.2237 against a CI of +/-1.5%. The headline will read within-ci whatever happens, and promotion has to rest on the router rows. Both rows remain 2.8-2.9x off the 1.75-1.78 us floor. The expected directions (router m=1/m=8 faster, everything else flat) are consistent with the code path: only router m in 1..8 changes.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (landed b99b1dd): it is the incumbent being displaced at m<=8 (_router_swapab_kernel plus the torch.zeros memset), so the change is defined relative to it. That makes it the comparison, not a confound. If its swap-AB router were ever made memset-free, most of this hypothesis's margin would disappear, because the cluster kernel's own time is higher (3.6/4.3 us vs 3.1/3.0 us). Judge against the restack's rows 5.38/5.47 (job 1e90292c7b954d96), not against master-era numbers.
- router-triton-hilo-fused-control-splitk-all-m, router-hilo-at-m8, sglang-gemm-dispatch-parity: superseded at m<=8 by the restack. They only set the hi/lo [256, 8192] bf16 row-major layout that the cluster kernel reads, and that layout is unchanged in b99b1dd, so there is no interaction.
- bf16-m32-swapab-wgmma-triton, bf16-shape-gated-triton-winners-combo: these change other layers only. Per-shape graphs are timed separately, so the only interaction is about 40 s more prepare/compile time inside the 10-minute budget. The non-router losses in dc5d03c4 (qkv/o 5-9% slower) came from its 8b9ad43 base lacking the restack, NOT from the cluster kernel. Here those rows must come back flat, and if they do not, the run did not build on the restack.

### Interference and A/B needs
No companions are needed. The change is a pure shape gate (router, 1 <= m <= 8, n == 256) inside the restack's kernel_fn, and the profile names a different kernel on exactly the two target rows. A gain or loss on router decode_1/verify_1 is therefore attributable to the swap of swapab+memset for cluster, and every other row is a built-in null control: if any non-router row moves beyond noise, the build is wrong, not the hypothesis. Reverting the restack (ab-revert) would only re-measure dc5d03c4. The informative follow-up is a different hypothesis, not a companion: a memset-free swap-AB router at m<=8 (e.g. SPLIT_K reduction in-kernel or a single-CTA-per-column variant with torch.empty) to test whether node removal alone captures the gain without a load_inline CUDA kernel. That would not be needed to read this run.

### Open questions
- Port faithfully: the source used state['router_cluster'] (NT bucket -> cluster size, 0 = fallback) plus a cudaOccupancyMaxActiveClusters gate (>=16 for CS=8). The hypothesis says 'router_cluster_ok' and 'compile only NT=8'. Keep the occupancy check and fall back to the restack's swapab when CS=8 NT=8 reports fewer than 16 active clusters. If the CS=4 variant is dropped, say so in the implementation notes.
- Insert the hook after _router_prepare in prepare() so router_stages exists, and have the self-check compare against x.double() @ router.double().T, the same bounds as the source (rel L2 <= 2e-5, worst row <= 1e-4, one-hot exact vs hi+lo, run 2 bit-identical). m in {1, 5, 8} matches the gate. The m=9 outlier test stays on swapab.
- Kill or promote on verify_1 m=8 above all (expected -7%). Treat decode_1 m=1 as a success only if its per-shape speedup beats the restack's 1.925 by more than 1.5% (above ~1.954). A 5.30-5.38 us row is a tie, not a loss.
- At harvest, confirm in the profile: router_kernel<8,8> on router decode_1/verify_1 with no vectorized_elementwise (memset) node; _router_swapab_kernel on router decode_16/decode_32; _router_hilo_kernel on m=128/256; _bf16_swapab_kernel on the restack's bf16 rows. Also confirm qkv_gate decode_1 is near 12.2 us and not 13.5 us; 13.5 would mean the build was on pre-restack master, as in the nine 03:41Z mis-builds.
- Why is the cluster kernel 3.6-4.3 us when swapab runs at 3.0-3.1 us? Cluster launch and scheduling cost and the DSMEM double barrier.cluster are likely causes. If this lands, the next stage should attack the kernel's own time, because the node saving has already been banked.

