---
problem: muse_bf16_proj
slug: pooler-unpooler-m128-tma-wspec-rewrite
created: 2026-10-24T140242Z
base: master
area: pooler GEMM + unpooler GEMM (m=128)
change: >
  Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler}
  at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m masked; the m=130 unpooler test
  stays cuBLAS). It starts from the regression pooler-unpooler-m128-wgmma-splitk-semaphore (branch
  hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore, _bf16_tc_splitk_kernel) and
  fixes the two costs that run's own prepare() diagnostics measured (d34fd76f9c444728.log: wgmma
  emitted, 140-154 regs, no spills; warm-L2 mainloop 26.0 us for both layers = 2.6 TB/s, reduce
  tail 6.5 us pooler / 4.3 us unpooler). Mainloop: (1) the weight tile arrives by TMA: a
  TensorDescriptor built once in prepare() from the weight alone
  (triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128]); this image's Triton
  is 3.8.0 with the tensor_descriptor API, per pooler-splitk2-tma-weight-stream's log) and w =
  w_desc.load([n0, k]) [BLOCK_N=128, BLOCK_K=128], fed as tl.dot(x_tile, tl.trans(w), acc); x
  [128, 128] stays a masked tl.load (2 MB, L2-resident). (2) BLOCK_K 128 instead of 64 (256 B
  contiguous per weight row per stage instead of 128 B), num_stages 3 (64 KB per stage, 192 KB),
  num_warps 8 (two warpgroups, wgmma m64n128 each). (3) Warp specialization: the K loop is `for k
  in tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=True)` so a producer warp
  issues TMA while the consumer warpgroups run wgmma; if Triton 3.8 rejects warp_specialize on
  sm90 at compile time in prepare(), fall back to the same loop without it and print which variant
  compiled (a constant fixed in prepare, never input-derived). Grid as before: pooler (n 4096, k
  8192) (32, SPLIT_K=4) = 128 CTAs, K-slice 2048; unpooler (8192, 4096) (64, SPLIT_K=2) = 128
  CTAs. Reduce tail: master's ticket semaphore (masked fp32 partial store into prepare()-allocated
  state[layer+'_m128_ws'] [SPLIT_K, 128, n], debug_barrier, one gpu-scope acq_rel ticket per
  N-tile, debug_barrier, atomic_xchg reset) with one change: the CTA that draws the last ticket
  does NOT store or reload its own partial; it keeps acc in registers and adds the other SPLIT_K-1
  partials (loaded .cg in index order, skipping its own pid_k), then stores bf16 once. That
  removes one 64 KB store and one 64 KB reload from the critical path (pooler reloads 3 partials
  instead of 4, unpooler 1 instead of 2). The sum order then depends on which CTA wins; still
  fp32, well inside the 1% bound. prepare(): build descriptors, allocate workspaces and zeroed
  int32 counters [n // 128], warm m in {65, 96, 128} per layer on synthetic x, re-zero counters,
  and print (try/except, diagnostics only) per layer: variant compiled, n_regs, n_spills, PTX
  counts of wgmma.mma_async and cp.async.bulk.tensor, and warm-L2 event times of the full kernel
  vs a constexpr REDUCE=False variant (as the parent did), so the log attributes any loss to
  mainloop or tail. kernel_fn gates on layer, cfg m in 65..128, cfg n/k and x bf16 contiguous;
  everything else keeps master.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "incumbent 26.0 us (2.6 TB/s) vs floor 14.6 us (67 MB; math 8.7 us at 989). Parent Triton: mainloop 26.0 warm + 6.5 tail = 29.4 us kernel. Milestones: M1 correct with printed mainloop <= 21 us warm-L2 (TMA + BLOCK_K 128 + warp specialization, ~3.3 TB/s); M2 tail <= 4 us (own partial kept in registers); M3 kernel <= 24 us, beating nvjet"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "incumbent 20.3 us vs floor 14.6. Parent: mainloop 26.0 warm + 4.3 tail. Milestones: M1 mainloop <= 18 us warm; M2 tail <= 2.5 us; M3 kernel <= 19.5 us"
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched (cuBLAS)
  - shapes: "* m=32"
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: cuBLAS nvjet at m=128 for pooler (nvjet_sm90_tst_64x64_64x13_2x1, 26.0 us) and unpooler (nvjet_sm90_tst_128x64_64x8_1x2, 20.3 us)
review: required
status: queued
---

# pooler-unpooler-m128-tma-wspec-rewrite

## Change

Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler} at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m masked; the m=130 unpooler test stays cuBLAS). It starts from the regression pooler-unpooler-m128-wgmma-splitk-semaphore (branch hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore, _bf16_tc_splitk_kernel) and fixes the two costs that run's own prepare() diagnostics measured (d34fd76f9c444728.log: wgmma emitted, 140-154 regs, no spills; warm-L2 mainloop 26.0 us for both layers = 2.6 TB/s, reduce tail 6.5 us pooler / 4.3 us unpooler). Mainloop: (1) the weight tile arrives by TMA: a TensorDescriptor built once in prepare() from the weight alone (triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128]); this image's Triton is 3.8.0 with the tensor_descriptor API, per pooler-splitk2-tma-weight-stream's log) and w = w_desc.load([n0, k]) [BLOCK_N=128, BLOCK_K=128], fed as tl.dot(x_tile, tl.trans(w), acc); x [128, 128] stays a masked tl.load (2 MB, L2-resident). (2) BLOCK_K 128 instead of 64 (256 B contiguous per weight row per stage instead of 128 B), num_stages 3 (64 KB per stage, 192 KB), num_warps 8 (two warpgroups, wgmma m64n128 each). (3) Warp specialization: the K loop is `for k in tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=True)` so a producer warp issues TMA while the consumer warpgroups run wgmma; if Triton 3.8 rejects warp_specialize on sm90 at compile time in prepare(), fall back to the same loop without it and print which variant compiled (a constant fixed in prepare, never input-derived). Grid as before: pooler (n 4096, k 8192) (32, SPLIT_K=4) = 128 CTAs, K-slice 2048; unpooler (8192, 4096) (64, SPLIT_K=2) = 128 CTAs. Reduce tail: master's ticket semaphore (masked fp32 partial store into prepare()-allocated state[layer+'_m128_ws'] [SPLIT_K, 128, n], debug_barrier, one gpu-scope acq_rel ticket per N-tile, debug_barrier, atomic_xchg reset) with one change: the CTA that draws the last ticket does NOT store or reload its own partial; it keeps acc in registers and adds the other SPLIT_K-1 partials (loaded .cg in index order, skipping its own pid_k), then stores bf16 once. That removes one 64 KB store and one 64 KB reload from the critical path (pooler reloads 3 partials instead of 4, unpooler 1 instead of 2). The sum order then depends on which CTA wins; still fp32, well inside the 1% bound. prepare(): build descriptors, allocate workspaces and zeroed int32 counters [n // 128], warm m in {65, 96, 128} per layer on synthetic x, re-zero counters, and print (try/except, diagnostics only) per layer: variant compiled, n_regs, n_spills, PTX counts of wgmma.mma_async and cp.async.bulk.tensor, and warm-L2 event times of the full kernel vs a constexpr REDUCE=False variant (as the parent did), so the log attributes any loss to mainloop or tail. kernel_fn gates on layer, cfg m in 65..128, cfg n/k and x bf16 contiguous; everything else keeps master.

## Expected

- `layer=pooler load=verify_16 m=128` faster: incumbent 26.0 us (2.6 TB/s) vs floor 14.6 us (67 MB; math 8.7 us at 989). Parent Triton: mainloop 26.0 warm + 6.5 tail = 29.4 us kernel. Milestones: M1 correct with printed mainloop <= 21 us warm-L2 (TMA + BLOCK_K 128 + warp specialization, ~3.3 TB/s); M2 tail <= 4 us (own partial kept in registers); M3 kernel <= 24 us, beating nvjet
- `layer=unpooler load=verify_16 m=128` faster: incumbent 20.3 us vs floor 14.6. Parent: mainloop 26.0 warm + 4.3 tail. Milestones: M1 mainloop <= 18 us warm; M2 tail <= 2.5 us; M3 kernel <= 19.5 us
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched
- `* m=256` flat: untouched (cuBLAS)
- `* m=32` flat: untouched

## Risk

Big-lever rewrite: the pooler m=128 row is the largest single gap on the board (11.4 us over floor), unpooler 5.7 us. The parent's diagnostics show the loss is the mainloop (26 us warm-L2, 2.6 TB/s with cp.async 128-byte row segments) plus a 4-6.5 us reduce tail, not a missing wgmma. Three mainloop changes are bundled on purpose (rewrite stage, judged on the replaced kernels' time against Milestones, not the headline); the printed variant and mainloop/tail times attribute them. Risks: Triton 3.8's warp_specialize may be Blackwell-only (fallback compiles the plain pipelined loop and says so); LESSONS says TMA added little over deep cp.async at m<=16, but there the stream was BLOCK_K 256 and BW-bound, here the shape is co-bound (128 FLOP per weight byte vs ~206 ridge) and the warps must be freed for wgmma. Overlaps bf16-cublaslt-algo-pick-in-prepare-m128-m256 on these two rows; per-shape rows decide which a later combo keeps. Stage 2 (if M1 met): stream-K over 132 SMs or BLOCK_N 256 with one warpgroup per 128 rows; also extend to qkv_gate/o at m=128 (the held qkv-gate-o-m128-wgmma-splitk-semaphore).

## Milestones

Rewrite stage 1 of `cuBLAS nvjet at m=128 for pooler (nvjet_sm90_tst_64x64_64x13_2x1, 26.0 us) and unpooler (nvjet_sm90_tst_128x64_64x8_1x2, 20.3 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
