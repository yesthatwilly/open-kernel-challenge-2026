---
problem: muse_bf16_proj
slug: qkv-swapab-w128-splitk7-one-wave-m32
created: 2026-10-25T110547Z
base: master
area: qkv GEMM (m<=32, swap-AB)
change: >
  Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128,
  check 84f17adc3f074a33, 1.2366x). WHY: at m<=32 qkv_gate (n=2304, k=8192, a 37.7 MB weight, cold
  L2 every replay) is the one DRAM weight stream still well below the bandwidth the pooler reaches
  on the same kernel family: master rows 12.38/13.14/13.93/12.45 us at m=1/16/32/8 = 3.05-2.71
  TB/s, against pooler 67.1 MB in 18.74 us = 3.58 TB/s. Both qkv entries (SWAPAB16_CFG['qkv_gate']
  and SWAPAB_CFG['qkv_gate']) launch _bf16_swapab_kernel with BLOCK_W=64, SPLIT_K=4, BLOCK_K=128,
  4 stages: grid (36, 4) = 144 CTAs on 132 SMs (12 SMs carry two CTAs, so the stream ends on a
  straggler half-wave) and only 64x128x2 B x 4 = 64 KB of weight in flight per CTA; LESSONS
  measured 64 KB in flight at ~3.3 TB/s and 128 KB at ~3.7 TB/s. CHANGE (qkv_gate only, both the
  m<=16 swapab16 route and the 17..32 swapab route; pooler, o, unpooler, router and every m>=128
  path byte-identical): (1) give _bf16_swapab_kernel a per-layer BLOCK_W and use BLOCK_W=128 for
  qkv_gate (2304/128 = 18 weight tiles; tl.dot w_tile[128, BLOCK_K] @ xt_tile[BLOCK_K, BLOCK_T],
  still swap-AB so wgmma is emitted; num_warps 4, print regs/spills/wgmma count from the compiled
  kernel's asm once in prepare, as the m32 swap-AB hypothesis did). (2) SPLIT_K=7 so the grid is
  (18, 7) = 126 CTAs, one wave, one CTA per SM. 8192/128 = 64 K-blocks do not divide by 7, so
  replace the constexpr K_SLICE with an uneven but static split: add constexpr NUM_KB = K //
  BLOCK_K and compute kb_lo = pid_k * NUM_KB // SPLIT_K, kb_hi = (pid_k + 1) * NUM_KB // SPLIT_K
  (9 or 10 K-blocks per CTA) and loop kk in range(kb_lo, kb_hi); keep the K_SLICE path unchanged
  for the other layers (e.g. NUM_KB = 0 / a constexpr EVEN flag selects the old code so their
  compiled kernels are unchanged). (3) Keep BLOCK_K=128 and 4 stages: (128 + BLOCK_T) x 128 x 2 B
  x 4 = 144 KB (BLOCK_T 16) / 160 KB (BLOCK_T 32) of smem, i.e. 128 KB of weight in flight per
  CTA, the pooler's recipe; if the 4-stage compile is OutOfResources fall back to 3 stages via the
  existing *_FALLBACK_STAGES mechanism and print it. (4) Reduce: the existing ticket semaphore
  with SPLIT_K=7, the qkv workspaces resized to (7, WS_M, 2304) fp32 and the tickets to 18 int32
  counters (N // BLOCK_W); fixed summation order 0..6, winner .cg reloads, self-reset, no memset
  node. Routing, gates and every other layer unchanged. In prepare print, per qkv route and per m
  in (1, 8, 16, 32), the cold-L2 (flush a > L2 buffer before each replay) CUDA-graph replay median
  of the new config against the old (36, 4) config on the same synthetic x (print-only, never
  routing), and assert torch.allclose against x.float() @ W.float().T at the reference tolerance.
expected:
  - shapes: layer=qkv_gate load=decode_* m=*
    direction: faster
    why: "126-CTA one-wave grid with 128 KB of weight in flight per CTA; target 3.4-3.6 TB/s: rows 12.38/13.14/13.93 -> <=11.3/11.6/12.2 us (floor 7.9-8.0)"
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: "same kernel at BLOCK_T 16: 12.45 -> <=11.3 us"
  - shapes: "layer=qkv_gate load=verify_* m=12[8] layer=qkv_gate load=verify_32 m=256"
    direction: flat
    why: m>=128 stays on cuBLAS nvjet, untouched
  - shapes: "layer=[oru]* *"
    direction: flat
    why: o, router, unpooler untouched
  - shapes: layer=pooler *
    direction: flat
    why: pooler config untouched (its constants and compiled kernel are byte-identical)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# qkv-swapab-w128-splitk7-one-wave-m32

## Change

Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, check 84f17adc3f074a33, 1.2366x). WHY: at m<=32 qkv_gate (n=2304, k=8192, a 37.7 MB weight, cold L2 every replay) is the one DRAM weight stream still well below the bandwidth the pooler reaches on the same kernel family: master rows 12.38/13.14/13.93/12.45 us at m=1/16/32/8 = 3.05-2.71 TB/s, against pooler 67.1 MB in 18.74 us = 3.58 TB/s. Both qkv entries (SWAPAB16_CFG['qkv_gate'] and SWAPAB_CFG['qkv_gate']) launch _bf16_swapab_kernel with BLOCK_W=64, SPLIT_K=4, BLOCK_K=128, 4 stages: grid (36, 4) = 144 CTAs on 132 SMs (12 SMs carry two CTAs, so the stream ends on a straggler half-wave) and only 64x128x2 B x 4 = 64 KB of weight in flight per CTA; LESSONS measured 64 KB in flight at ~3.3 TB/s and 128 KB at ~3.7 TB/s. CHANGE (qkv_gate only, both the m<=16 swapab16 route and the 17..32 swapab route; pooler, o, unpooler, router and every m>=128 path byte-identical): (1) give _bf16_swapab_kernel a per-layer BLOCK_W and use BLOCK_W=128 for qkv_gate (2304/128 = 18 weight tiles; tl.dot w_tile[128, BLOCK_K] @ xt_tile[BLOCK_K, BLOCK_T], still swap-AB so wgmma is emitted; num_warps 4, print regs/spills/wgmma count from the compiled kernel's asm once in prepare, as the m32 swap-AB hypothesis did). (2) SPLIT_K=7 so the grid is (18, 7) = 126 CTAs, one wave, one CTA per SM. 8192/128 = 64 K-blocks do not divide by 7, so replace the constexpr K_SLICE with an uneven but static split: add constexpr NUM_KB = K // BLOCK_K and compute kb_lo = pid_k * NUM_KB // SPLIT_K, kb_hi = (pid_k + 1) * NUM_KB // SPLIT_K (9 or 10 K-blocks per CTA) and loop kk in range(kb_lo, kb_hi); keep the K_SLICE path unchanged for the other layers (e.g. NUM_KB = 0 / a constexpr EVEN flag selects the old code so their compiled kernels are unchanged). (3) Keep BLOCK_K=128 and 4 stages: (128 + BLOCK_T) x 128 x 2 B x 4 = 144 KB (BLOCK_T 16) / 160 KB (BLOCK_T 32) of smem, i.e. 128 KB of weight in flight per CTA, the pooler's recipe; if the 4-stage compile is OutOfResources fall back to 3 stages via the existing *_FALLBACK_STAGES mechanism and print it. (4) Reduce: the existing ticket semaphore with SPLIT_K=7, the qkv workspaces resized to (7, WS_M, 2304) fp32 and the tickets to 18 int32 counters (N // BLOCK_W); fixed summation order 0..6, winner .cg reloads, self-reset, no memset node. Routing, gates and every other layer unchanged. In prepare print, per qkv route and per m in (1, 8, 16, 32), the cold-L2 (flush a > L2 buffer before each replay) CUDA-graph replay median of the new config against the old (36, 4) config on the same synthetic x (print-only, never routing), and assert torch.allclose against x.float() @ W.float().T at the reference tolerance.

## Expected

- `layer=qkv_gate load=decode_* m=*` faster: 126-CTA one-wave grid with 128 KB of weight in flight per CTA; target 3.4-3.6 TB/s: rows 12.38/13.14/13.93 -> <=11.3/11.6/12.2 us (floor 7.9-8.0)
- `layer=qkv_gate load=verify_1 m=8` faster: same kernel at BLOCK_T 16: 12.45 -> <=11.3 us
- `layer=qkv_gate load=verify_* m=12[8] layer=qkv_gate load=verify_32 m=256` flat: m>=128 stays on cuBLAS nvjet, untouched
- `layer=[oru]* *` flat: o, router, unpooler untouched
- `layer=pooler *` flat: pooler config untouched (its constants and compiled kernel are byte-identical)

## Risk

Biggest risk: the split-K tail. qkv-swapab-m32-splitk8-stages3 (a61778a0) lost +14% at m=32 with a flat kernel row, the loss sitting in the SPLIT_K=8 reduce/semaphore tail; here the split count drops 8 -> 7 and the tile count halves (18 winners each summing 7 partials of 128 x BLOCK_T fp32, 115 KB per winner at m=32), so a slower winner is possible: the prepare A/B print separates mainloop from tail if it lands flat. Second: uneven 9/10 K-block slices make the 10-block CTAs (8 of 126... every pid_k with an extra block) the critical path, a ~10% stretch of the per-CTA stream; still one wave instead of the current 144-CTA straggler. Third: BLOCK_W 128 at 4 warps doubles the accumulator (128 x 32 fp32 = 32 regs/thread at m=32), well below spill. Headline: qkv is on 4 of 6 loads; -1.2 to -1.9 us on a 47-60 us load is +2-4% on each, about +1.5-2.5% on the headline if all four rows land. Judge per row against the master rows above (drift +-1.5-2.5%); a row gain under ~3% needs the kernel time in top_kernels to confirm.
