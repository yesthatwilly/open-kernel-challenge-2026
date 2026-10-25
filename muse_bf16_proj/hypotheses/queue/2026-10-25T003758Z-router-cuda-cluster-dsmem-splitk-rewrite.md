---
problem: muse_bf16_proj
slug: router-cuda-cluster-dsmem-splitk-rewrite
created: 2026-10-25T003758Z
base: master
area: router GEMM
change: >
  Built on master (8b9ad43, blob a8bb42a); the restack is not landed and is not needed. WHY:
  router rows at m<=32 are 5.4-6.2 us against a 1.8-1.9 us floor (8.4 MB of hi+lo bf16). The
  profile splits that into the kernel (3.0-3.6 us) plus a memset node (0.7-0.8 us) from
  torch.zeros, which master needs because its split-K partials are atomically added; the rest is
  per-node cost. Tuning the Triton router has stopped paying
  (router-splitk32-small-m-all-loads-in-flight, router-splitk-semaphore-no-memset and its stages3
  control: all neutral; the global-memory semaphore lost on router decode rows). STAGE 1: a
  hand-written CUDA kernel, one launch, no memset, no atomics, no global workspace: split-K
  reduced through thread-block-cluster distributed shared memory. Compile at import with
  torch.utils.cpp_extension.load_inline (reuse the bf16 gemv/lossless scaffold: print compile
  seconds, per-route in-process self-check, print-and-fall-back to master; set
  TORCH_CUDA_ARCH_LIST=9.0a around the build and pass -lineinfo, though nothing here needs sm_90a:
  mma.sync, cp.async.cg, cluster launch, mapa/ld.shared::cluster and barrier.cluster are all
  baseline sm_90). Decomposition: 16 clusters x 8 CTAs = 128 CTAs (cudaLaunchKernelEx with
  cudaLaunchAttributeClusterDimension 8, portable size). Cluster c owns output columns [16c,
  16c+16) of the 256 experts; CTA rank r in the cluster owns K [1024r, 1024r+1024). Swap-AB
  mma.sync m16n8k16: the 16 hi rows and the 16 lo rows of the slice are the A operand (two MMAs
  per k16 step into one fp32 accumulator), x^T is B with tokens on N (NT = 8, 16 or 32 compiled as
  template instances; columns >= m are zero-filled, never stored). Each CTA issues ALL its loads
  up front: 16 x 1024 x 2 arrays x 2 B = 64 KB of weight plus m x 1024 x 2 B of x via plain
  cp.async.cg 16 B into smem (one commit group per 256-K chunk so MMAs start on chunk 0), 4 warps
  each taking a 256-K quarter and reducing the 4 warp partials in smem. Then each CTA writes its
  [16, NT] fp32 partial to its own smem, barrier.cluster.arrive.release / wait.acquire, and CTA
  rank r reduces output columns {2r, 2r+1} over the 8 ranks in fixed rank order 0..7 via mapa +
  ld.shared::cluster (deterministic), and stores fp32 out[m, 16c+2r..] with plain stores; a second
  cluster barrier before exit keeps peer smem alive. Output allocated with torch.empty. ROUTING:
  kernel_fn router branch at 1 <= m <= 32 when state router_cluster_ok (a constant set in prepare)
  and shapes match; m > 32 (64/128/256) keeps master byte for byte. The router tests m=8 and m=9
  dist outliers go through it. SELF-CHECK in prepare at m in {1, 8, 9, 16, 32}, run twice: rel L2
  <= 2e-5 vs x.double() @ router.double().T, one-hot power-of-two probe rows within 1e-6 relative
  to the hi+lo reconstruction, finite, run 2 bit-identical; synchronize and print after each
  launch. DIAGNOSTICS (prints, try/except): regs/spills/smem via cudaFuncGetAttributes,
  cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush a > L2 buffer first) event times vs
  master _router_forward on the same synthetic x. Stage 2 (later): extend to m=128/256 with
  NT=64/128 tiles or a 2-cluster-per-column split. Run as a check.
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: "incumbent row 5.4 us = kernel 3.1 + memset 0.8 + node cost; floor 1.8. One node, no memset, 64 KB per CTA all in flight: M2 row <= 5.0, M3 row <= 4.0"
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: incumbent row 5.5 (kernel 3.0 + memset 0.7); floor 1.8; M3 <= 4.0
  - shapes: layer=router load=decode_16 m=16
    direction: faster
    why: incumbent row 5.5 (kernel 3.4 + 0.7); floor 1.8; M3 <= 4.2
  - shapes: layer=router load=decode_32 m=32
    direction: faster
    why: incumbent row 6.2 (kernel 3.6 + 0.7); floor 1.9; M3 <= 4.5
  - shapes: layer=router load=verify_16 m=128
    direction: flat
    why: m > 32 keeps master
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: m > 32 keeps master
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
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "router at m<=32: torch.zeros memset node (vectorized_elementwise 0.7-0.8 us) + _router_hilo_kernel 3-D split-K with fp32 atomic_add (3.0-3.6 us)"
review: proceed
status: queued
review_at: 2026-10-25T004221Z
review_companions: null
---

# router-cuda-cluster-dsmem-splitk-rewrite

## Change

Built on master (8b9ad43, blob a8bb42a); the restack is not landed and is not needed. WHY: router rows at m<=32 are 5.4-6.2 us against a 1.8-1.9 us floor (8.4 MB of hi+lo bf16). The profile splits that into the kernel (3.0-3.6 us) plus a memset node (0.7-0.8 us) from torch.zeros, which master needs because its split-K partials are atomically added; the rest is per-node cost. Tuning the Triton router has stopped paying (router-splitk32-small-m-all-loads-in-flight, router-splitk-semaphore-no-memset and its stages3 control: all neutral; the global-memory semaphore lost on router decode rows). STAGE 1: a hand-written CUDA kernel, one launch, no memset, no atomics, no global workspace: split-K reduced through thread-block-cluster distributed shared memory. Compile at import with torch.utils.cpp_extension.load_inline (reuse the bf16 gemv/lossless scaffold: print compile seconds, per-route in-process self-check, print-and-fall-back to master; set TORCH_CUDA_ARCH_LIST=9.0a around the build and pass -lineinfo, though nothing here needs sm_90a: mma.sync, cp.async.cg, cluster launch, mapa/ld.shared::cluster and barrier.cluster are all baseline sm_90). Decomposition: 16 clusters x 8 CTAs = 128 CTAs (cudaLaunchKernelEx with cudaLaunchAttributeClusterDimension 8, portable size). Cluster c owns output columns [16c, 16c+16) of the 256 experts; CTA rank r in the cluster owns K [1024r, 1024r+1024). Swap-AB mma.sync m16n8k16: the 16 hi rows and the 16 lo rows of the slice are the A operand (two MMAs per k16 step into one fp32 accumulator), x^T is B with tokens on N (NT = 8, 16 or 32 compiled as template instances; columns >= m are zero-filled, never stored). Each CTA issues ALL its loads up front: 16 x 1024 x 2 arrays x 2 B = 64 KB of weight plus m x 1024 x 2 B of x via plain cp.async.cg 16 B into smem (one commit group per 256-K chunk so MMAs start on chunk 0), 4 warps each taking a 256-K quarter and reducing the 4 warp partials in smem. Then each CTA writes its [16, NT] fp32 partial to its own smem, barrier.cluster.arrive.release / wait.acquire, and CTA rank r reduces output columns {2r, 2r+1} over the 8 ranks in fixed rank order 0..7 via mapa + ld.shared::cluster (deterministic), and stores fp32 out[m, 16c+2r..] with plain stores; a second cluster barrier before exit keeps peer smem alive. Output allocated with torch.empty. ROUTING: kernel_fn router branch at 1 <= m <= 32 when state router_cluster_ok (a constant set in prepare) and shapes match; m > 32 (64/128/256) keeps master byte for byte. The router tests m=8 and m=9 dist outliers go through it. SELF-CHECK in prepare at m in {1, 8, 9, 16, 32}, run twice: rel L2 <= 2e-5 vs x.double() @ router.double().T, one-hot power-of-two probe rows within 1e-6 relative to the hi+lo reconstruction, finite, run 2 bit-identical; synchronize and print after each launch. DIAGNOSTICS (prints, try/except): regs/spills/smem via cudaFuncGetAttributes, cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush a > L2 buffer first) event times vs master _router_forward on the same synthetic x. Stage 2 (later): extend to m=128/256 with NT=64/128 tiles or a 2-cluster-per-column split. Run as a check.

## Expected

- `layer=router load=decode_1 m=1` faster: incumbent row 5.4 us = kernel 3.1 + memset 0.8 + node cost; floor 1.8. One node, no memset, 64 KB per CTA all in flight: M2 row <= 5.0, M3 row <= 4.0
- `layer=router load=verify_1 m=8` faster: incumbent row 5.5 (kernel 3.0 + memset 0.7); floor 1.8; M3 <= 4.0
- `layer=router load=decode_16 m=16` faster: incumbent row 5.5 (kernel 3.4 + 0.7); floor 1.8; M3 <= 4.2
- `layer=router load=decode_32 m=32` faster: incumbent row 6.2 (kernel 3.6 + 0.7); floor 1.9; M3 <= 4.5
- `layer=router load=verify_16 m=128` flat: m > 32 keeps master
- `layer=router load=verify_32 m=256` flat: m > 32 keeps master
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Biggest risk: a first-launch fault in a hand-written cluster kernel poisons the CUDA context (LESSONS 2026-10-25), so the self-check prints around each launch and the first measurement is an okc check. Second: cluster scheduling needs 8 co-resident CTAs on one GPC; 128 CTAs at 64-80 KB smem should fit 16 clusters, but print cudaOccupancyMaxActiveClusters and fall back if it is < 16. Third: the precision bar is the router 0.01%; hi+lo with fp32 accumulation and a fixed-order fp32 reduction matches what master already passes. Headline if M3: router saves ~1.5 us on 4 shapes x 62 weight, about +2% on decode_1/16/32 and verify_1 loads.

## Milestones

Rewrite stage 1 of `router at m<=32: torch.zeros memset node (vectorized_elementwise 0.7-0.8 us) + _router_hilo_kernel 3-D split-K with fp32 atomic_add (3.0-3.6 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T004221Z by an adversarial subagent (model opus).

**Verdict: proceed.** Design fits sm_90 and the correctness plan is sound, but the quoted incumbent numbers are the restack's, not master's, and M3 (row <=4.0 us) is very unlikely because a graph node still costs ~1.9 us on top of the kernel. Run as a check, judge against master's router rows, and confirm from the log that the cluster path ran.

### Justification
Removes the torch.zeros memset node and fp32 atomics, reducing split-K through cluster DSMEM in one launch; plain sm_90 (mma.sync, cp.async.cg, cluster launch, mapa, barrier.cluster), no wgmma. Problem 1: master's base run 1602a4e947e64090 has _router_hilo_kernel 3.2/3.5/4.1/3.2 us plus 0.7 us memset, rows 5.73/6.03/6.55/5.91 for m=1/16/32/8; the quoted 5.4/5.5/6.2/5.5 are the restack's. Problem 2: row ~= kernel + 1.9 us, so M3 needs a ~2.1 us kernel (4 TB/s cold), above anything measured here (~3.5 TB/s); realistic stage 1 is kernel 2.7-3.2, rows 4.7-5.1, saving 0.7-1.2 us/shape. Recalibrate M3. Problem 3: at NT=32 x is 64 KB/CTA plus 64 KB weight plus partials, >128 KB, so 1 CTA/SM and an explicit MaxDynamicSharedMemorySize; 16 clusters of 8 co-residency depends on GPCs. m=32 re-reads x 16x (~8 MB L2->SM), most likely to lag.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: the incumbent at m<=32; judge vs _router_hilo_kernel 3.2/3.5/4.1/3.2 us.
- bf16-restack-m16-swapab-router-swapab-m32 (pending): would make the incumbent _router_swapab_kernel 3.1/3.4/3.6/3.0; a master-relative gain at m=32 overstates the restack-relative gain by ~0.5 us; compare vs both (job 1e90292c7b954d96).
- bf16-m32-swapab-wgmma-triton, bf16-shape-gated-triton-winners-combo: other layers; ~40-50 s more load_inline in prepare, within budget.
- router-hilo-at-m8, sglang-gemm-dispatch-parity: superseded.

### Interference and A/B needs
No companions. Clean shape gate; the profile names the new kernel on the four rows; in-process warm/cold-L2 event times vs _router_forward give a same-run A/B.

### Open questions
- Does cudaOccupancyMaxActiveClusters(cluster=8) report >=16 at the NT=32 footprint? Also compile a cluster-4 variant, or load each warp's x quarter into registers so NT=32 fits 2 CTAs/SM; print which path ran per m.
- Recalibrate milestones against master (rows 5.73/6.03/6.55/5.91, kernels 3.2/3.5/4.1/3.2), M3 as kernel + 1.9 us.
- Print kernel time with and without the DSMEM reduce to split mainloop from tail.
- Stage 2 (m=128/256) needs a K-tiled x pipeline.
- At harvest confirm the build line, occupancy print, PATH CHOSEN, and no memset node before reading numbers.

