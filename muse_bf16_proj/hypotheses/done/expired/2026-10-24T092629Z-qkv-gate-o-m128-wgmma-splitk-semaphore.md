---
problem: muse_bf16_proj
slug: qkv-gate-o-m128-wgmma-splitk-semaphore
created: 2026-10-24T092629Z
base: master
area: qkv GEMM + o GEMM (m=128)
change: >
  Built on master (dba4ffd; router untouched). For layer in {qkv_gate, o} and cfg m == 128 only
  (verify_16; 65 <= m <= 128 may share the path with rows >= m masked), replace
  _bf16_gemm_dispatch_impl (qkv_gate: nvjet_sm90_tst_96x64_64x10_4x2_splitK 12.9 us +
  cublasLt::splitKreduce_kernel 3.1 us, two graph nodes, 17.0 us shape; o:
  nvjet_sm90_tst_128x64_64x8_1x2 5.3 us, 7.0 us shape) with ONE Triton launch of a tensor-core
  split-K kernel _bf16_tc_splitk_kernel: BLOCK_M=128 (all of m), BLOCK_N=128, BLOCK_K=64,
  num_warps=8 (two warpgroups -> wgmma m64n128), num_stages=4. qkv_gate (n=2304 = 18 x 128,
  k=8192): SPLIT_K=8 -> grid (18, 8) = 144 CTAs, K-slice 1024 (16 K-steps), each CTA 256 KB weight
  from HBM + 256 KB x from L2. o (n=8192, k=1024): SPLIT_K=2 -> grid (64, 2) = 128 CTAs, K-slice
  512 (8 K-steps). Weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [N, K]
  weight (master router transposed addressing, no repack); fp32 acc via tl.dot(x_tile, w_tile,
  acc). Reduce in the same kernel with the ticket-semaphore epilogue copied verbatim from branch
  hyp/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore (masked fp32 partial store into a
  prepare()-allocated workspace state[layer + "_ws128"] = torch.empty((SPLIT_K, 128, n), fp32);
  tl.debug_barrier(); one gpu-scope acq_rel atomic_add ticket per N-tile on state[layer +
  "_cnt128"] = torch.zeros(n // 128, int32); the CTA drawing SPLIT_K-1 reloads the partials with
  .cg, sums in fixed order, casts once to bf16, masked-stores into out = torch.empty((m, n),
  bf16), resets its counter with atomic_xchg; add a tl.debug_barrier() after the ticket). This
  removes cuBLAS splitKreduce node and its graph gap at qkv_gate and fills 144 SMs with 128x128
  wgmma tiles instead of 96x64. Pre-compile both specializations in prepare() on synthetic x at
  m=128 and re-zero the counters after warm-up. All other layers and m keep their master path
  (qkv_gate/o at m=256 are compute-bound and stay cuBLAS; m <= 32 stays cuBLAS, where qkv-gate
  split-K4 was neutral).
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "17.0 us vs floor 8.4 (37.7 MB; 4.8 GFLOP): 8.6 us gap, second largest on the board. cuBLAS spends 3.1 us + a graph gap in splitKreduce and runs 96x64 tiles; one launch of 128x128 wgmma tiles over 144 CTAs streaming 256 KB each plus an in-kernel reduce of 9.4 MB L2-resident partials. Target <= 12 us"
  - shapes: layer=o load=verify_16 m=128
    direction: faster
    why: "7.0 us vs floor 4.0 (16.8 MB, 2.1 GFLOP); cuBLAS 128x64 tile over full K=1024 at 128 CTAs reads 256 KB x per 128 KB of weight. SPLIT_K=2 at 128x128 halves that ratio; target <= 6 us (small: o is 9% of the load)"
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: flat
    why: "control: stays cuBLAS"
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: done
hold_reason: same m=128 wgmma split-K design as pooler-unpooler-m128-wgmma-splitk-semaphore (dispatching now); its reviewer asked to wait for that harvest; re-queue by hand if that kernel reaches its target
outcome: expired
verdict_detail: "retired: on the bf16 DIAGNOSIS Do-not-try list"
expired_at: 2026-10-25T211946Z
---

# qkv-gate-o-m128-wgmma-splitk-semaphore

## Change

Built on master (dba4ffd; router untouched). For layer in {qkv_gate, o} and cfg m == 128 only (verify_16; 65 <= m <= 128 may share the path with rows >= m masked), replace _bf16_gemm_dispatch_impl (qkv_gate: nvjet_sm90_tst_96x64_64x10_4x2_splitK 12.9 us + cublasLt::splitKreduce_kernel 3.1 us, two graph nodes, 17.0 us shape; o: nvjet_sm90_tst_128x64_64x8_1x2 5.3 us, 7.0 us shape) with ONE Triton launch of a tensor-core split-K kernel _bf16_tc_splitk_kernel: BLOCK_M=128 (all of m), BLOCK_N=128, BLOCK_K=64, num_warps=8 (two warpgroups -> wgmma m64n128), num_stages=4. qkv_gate (n=2304 = 18 x 128, k=8192): SPLIT_K=8 -> grid (18, 8) = 144 CTAs, K-slice 1024 (16 K-steps), each CTA 256 KB weight from HBM + 256 KB x from L2. o (n=8192, k=1024): SPLIT_K=2 -> grid (64, 2) = 128 CTAs, K-slice 512 (8 K-steps). Weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [N, K] weight (master router transposed addressing, no repack); fp32 acc via tl.dot(x_tile, w_tile, acc). Reduce in the same kernel with the ticket-semaphore epilogue copied verbatim from branch hyp/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore (masked fp32 partial store into a prepare()-allocated workspace state[layer + "_ws128"] = torch.empty((SPLIT_K, 128, n), fp32); tl.debug_barrier(); one gpu-scope acq_rel atomic_add ticket per N-tile on state[layer + "_cnt128"] = torch.zeros(n // 128, int32); the CTA drawing SPLIT_K-1 reloads the partials with .cg, sums in fixed order, casts once to bf16, masked-stores into out = torch.empty((m, n), bf16), resets its counter with atomic_xchg; add a tl.debug_barrier() after the ticket). This removes cuBLAS splitKreduce node and its graph gap at qkv_gate and fills 144 SMs with 128x128 wgmma tiles instead of 96x64. Pre-compile both specializations in prepare() on synthetic x at m=128 and re-zero the counters after warm-up. All other layers and m keep their master path (qkv_gate/o at m=256 are compute-bound and stay cuBLAS; m <= 32 stays cuBLAS, where qkv-gate split-K4 was neutral).

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: 17.0 us vs floor 8.4 (37.7 MB; 4.8 GFLOP): 8.6 us gap, second largest on the board. cuBLAS spends 3.1 us + a graph gap in splitKreduce and runs 96x64 tiles; one launch of 128x128 wgmma tiles over 144 CTAs streaming 256 KB each plus an in-kernel reduce of 9.4 MB L2-resident partials. Target <= 12 us
- `layer=o load=verify_16 m=128` faster: 7.0 us vs floor 4.0 (16.8 MB, 2.1 GFLOP); cuBLAS 128x64 tile over full K=1024 at 128 CTAs reads 256 KB x per 128 KB of weight. SPLIT_K=2 at 128x128 halves that ratio; target <= 6 us (small: o is 9% of the load)
- `layer=qkv_gate load=verify_32 m=256` flat: control: stays cuBLAS
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Medium risk. (1) wgmma lowering with K-contiguous [BLOCK_K, BLOCK_N] tiles must happen (check PTX for wgmma.mma_async); mma.sync would make the 33.5 MFLOP-per-CTA qkv tile compute-limited near cuBLAS. (2) SPLIT_K=8 partials are 9.4 MB per call (64 KB fp32 per CTA); they should stay in L2, but the last-CTA reduce reads 8 x 64 KB serially, ~1 us tail. If qkv gains less than the splitKreduce node it removed, SPLIT_K=4 with BLOCK_N=64 (also 144 CTAs, 4.7 MB partials) is the follow-up retune. (3) Correctness: fp32 accumulate and one bf16 rounding, as cuBLAS split-K; the m=512 qkv_gate and m=33 o tests stay on cuBLAS. Attribution: only the two verify_16 rows run new code; the readout is the kernel us against nvjet+splitKreduce 16.0 us and nvjet 5.3 us.
