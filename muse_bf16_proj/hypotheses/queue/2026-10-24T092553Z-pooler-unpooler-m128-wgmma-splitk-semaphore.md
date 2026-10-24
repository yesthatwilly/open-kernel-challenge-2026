---
problem: muse_bf16_proj
slug: pooler-unpooler-m128-wgmma-splitk-semaphore
created: 2026-10-24T092553Z
base: master
area: pooler GEMM + unpooler GEMM (m=128)
change: >
  Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For
  layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 65 <= m <= 128 may take the
  same path, rows >= m masked), replace _bf16_gemm_dispatch_impl (pooler:
  nvjet_sm90_tst_64x64_64x13_2x1 at 25.2 us, a 64x64 tile with no split-K, 2 x 64 = 128 CTAs each
  re-reading half of x; unpooler: nvjet_sm90_tst_128x64_64x8_1x2 at 20.0 us) with ONE Triton
  launch of a new tensor-core split-K kernel _bf16_tc_splitk_kernel: tile BLOCK_M=128 (all of m),
  BLOCK_N=128, BLOCK_K=64, num_warps=8 (two warpgroups, so Triton emits wgmma m64n128 per
  warpgroup), num_stages=4 (32 KB per stage: 16 KB x + 16 KB weight, 128 KB smem). Grid (N // 128,
  SPLIT_K): pooler (n=4096, k=8192) SPLIT_K=4 -> 32 x 4 = 128 CTAs, K-slice 2048 (32 K-steps);
  unpooler (n=8192, k=4096) SPLIT_K=2 -> 64 x 2 = 128 CTAs, K-slice 2048. Every CTA therefore
  moves 512 KB of weight from HBM and 512 KB of x from L2 (cuBLAS pooler moves 512 KB weight + 1
  MB x per CTA), and does 67 MFLOP. Weight tiles read as [BLOCK_K, BLOCK_N] straight from the
  row-major [N, K] weight with master router direct transposed addressing (offs_k[:, None] +
  offs_n[None, :] * K; no repack); acc = tl.dot(x_tile, w_tile, acc) fp32. Reduce: reuse verbatim
  the ticket-semaphore epilogue from branch
  hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (masked fp32 partial store to a
  prepare()-allocated workspace state[layer + "_ws"] = torch.empty((SPLIT_K, 128, n), fp32),
  tl.debug_barrier(), one gpu-scope acq_rel tl.atomic_add ticket per N-tile on state[layer +
  "_cnt"] = torch.zeros(n // 128, int32), the CTA drawing SPLIT_K-1 reloads all partials with
  cache_modifier=".cg", sums in fixed order 0..SPLIT_K-1, casts once to bf16, masked-stores into
  out = torch.empty((m, n), bf16), and resets its counter with tl.atomic_xchg; add
  tl.debug_barrier() after the ticket as the review of that parent suggested). Pre-compile both
  layer specializations in prepare() on synthetic x at m=128 and re-zero the counters after the
  warm-up. Every other layer/m keeps its master path (pooler/unpooler at m <= 32 and m = 256 stay
  cuBLAS: m=256 is compute-bound, 17.2 GFLOP, where cuBLAS already runs ~580 TFLOP/s).
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "25.2 us vs floor 14.6 (67.1 MB; 8.6 GFLOP = 8.7 us at 989): the largest single gap on the board (10.6 us). cuBLAS picks a 64x64 tile without split-K (2.7 TB/s); a 128x128 wgmma tile over a 2048 K-slice halves x L2 traffic per weight byte and keeps 128 CTAs. Target <= 19 us (kernel us in top_kernels), ~3.7 TB/s"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "20.0 us vs floor 14.6; cuBLAS 128x64 tiles over full K=4096 read 1 MB of x per CTA. Same kernel at SPLIT_K=2: half the x traffic and twice the MMA N per weight byte; target <= 17.5 us"
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: "control: stays cuBLAS (compute-bound)"
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: "control: stays cuBLAS"
  - shapes: layer=qkv_gate *
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
status: queued
---

# pooler-unpooler-m128-wgmma-splitk-semaphore

## Change

Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 65 <= m <= 128 may take the same path, rows >= m masked), replace _bf16_gemm_dispatch_impl (pooler: nvjet_sm90_tst_64x64_64x13_2x1 at 25.2 us, a 64x64 tile with no split-K, 2 x 64 = 128 CTAs each re-reading half of x; unpooler: nvjet_sm90_tst_128x64_64x8_1x2 at 20.0 us) with ONE Triton launch of a new tensor-core split-K kernel _bf16_tc_splitk_kernel: tile BLOCK_M=128 (all of m), BLOCK_N=128, BLOCK_K=64, num_warps=8 (two warpgroups, so Triton emits wgmma m64n128 per warpgroup), num_stages=4 (32 KB per stage: 16 KB x + 16 KB weight, 128 KB smem). Grid (N // 128, SPLIT_K): pooler (n=4096, k=8192) SPLIT_K=4 -> 32 x 4 = 128 CTAs, K-slice 2048 (32 K-steps); unpooler (n=8192, k=4096) SPLIT_K=2 -> 64 x 2 = 128 CTAs, K-slice 2048. Every CTA therefore moves 512 KB of weight from HBM and 512 KB of x from L2 (cuBLAS pooler moves 512 KB weight + 1 MB x per CTA), and does 67 MFLOP. Weight tiles read as [BLOCK_K, BLOCK_N] straight from the row-major [N, K] weight with master router direct transposed addressing (offs_k[:, None] + offs_n[None, :] * K; no repack); acc = tl.dot(x_tile, w_tile, acc) fp32. Reduce: reuse verbatim the ticket-semaphore epilogue from branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (masked fp32 partial store to a prepare()-allocated workspace state[layer + "_ws"] = torch.empty((SPLIT_K, 128, n), fp32), tl.debug_barrier(), one gpu-scope acq_rel tl.atomic_add ticket per N-tile on state[layer + "_cnt"] = torch.zeros(n // 128, int32), the CTA drawing SPLIT_K-1 reloads all partials with cache_modifier=".cg", sums in fixed order 0..SPLIT_K-1, casts once to bf16, masked-stores into out = torch.empty((m, n), bf16), and resets its counter with tl.atomic_xchg; add tl.debug_barrier() after the ticket as the review of that parent suggested). Pre-compile both layer specializations in prepare() on synthetic x at m=128 and re-zero the counters after the warm-up. Every other layer/m keeps its master path (pooler/unpooler at m <= 32 and m = 256 stay cuBLAS: m=256 is compute-bound, 17.2 GFLOP, where cuBLAS already runs ~580 TFLOP/s).

## Expected

- `layer=pooler load=verify_16 m=128` faster: 25.2 us vs floor 14.6 (67.1 MB; 8.6 GFLOP = 8.7 us at 989): the largest single gap on the board (10.6 us). cuBLAS picks a 64x64 tile without split-K (2.7 TB/s); a 128x128 wgmma tile over a 2048 K-slice halves x L2 traffic per weight byte and keeps 128 CTAs. Target <= 19 us (kernel us in top_kernels), ~3.7 TB/s
- `layer=unpooler load=verify_16 m=128` faster: 20.0 us vs floor 14.6; cuBLAS 128x64 tiles over full K=4096 read 1 MB of x per CTA. Same kernel at SPLIT_K=2: half the x traffic and twice the MMA N per weight byte; target <= 17.5 us
- `layer=pooler load=verify_32 m=256` flat: control: stays cuBLAS (compute-bound)
- `layer=unpooler load=verify_32 m=256` flat: control: stays cuBLAS
- `layer=qkv_gate *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Medium risk. (1) Triton must lower BLOCK_M=128 x BLOCK_N=128 with K-contiguous [BLOCK_K, BLOCK_N] weight tiles to wgmma (MN-major B from smem is supported for bf16); if it falls back to mma.sync the kernel is compute-limited near 9-12 us of math at 128 CTAs and may only match cuBLAS. Check the PTX for wgmma.mma_async before the run if possible. (2) Workspace traffic: SPLIT_K x 64 KB fp32 partials per N-tile (8 MB per call), L2-resident; the last-CTA reduce tail is ~0.5-1 us. (3) The semaphore is the measured-correct pattern from the pooler/qkv-gate split-K runs; counter must be re-zeroed after warm-up. Attribution: only the two verify_16 rows change code; their kernel us against nvjet 25.2/20.0 us is the readout. Load verify_16 is one of six; ~8 us out of ~76 us is ~+1.8% on the headline, so read per-shape rows.
