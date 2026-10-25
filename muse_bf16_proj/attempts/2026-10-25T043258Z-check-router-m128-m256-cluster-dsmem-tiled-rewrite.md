# muse_bf16_proj · 2026-10-25T043258Z · check · router-m128-m256-cluster-dsmem-tiled-rewrite

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T042040Z-router-m128-m256-cluster-dsmem-tiled-rewrite.md` (branch `hyp/muse_bf16_proj/router-m128-m256-cluster-dsmem-tiled-rewrite`).

- **Job:** `b7b2b55d04544660` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `8dd2a8d`

## Goal
Built on master (the router at m > 32 is identical on master and the restack). Router rows at m=128/256 are 7.3/9.6 us against floors of 2.2/2.7 (3.3-3.6x, the worst ratio left at m>=128): _router_hilo_kernel 5.0/6.9 us plus a 0.7-0.8 us vectorized_elementwise memset node for the atomically-added split-K output. New CUDA kernel, one node, no memset, no atomics, reusing the router-cuda-cluster-dsmem-splitk-rewrite scaffold (load_inline, TORCH_CUDA_ARCH_LIST=9.0a, -std=c++20 replacing any -std, -lineinfo, per-route self-check, print-and-fall-back). Tiles the [m, 256] fp32 output: m=128 -> 64 token x 32 expert tiles (2 x 8 = 16 tiles); m=256 -> 64 x 64 tiles (4 x 4 = 16 tiles); each tile is one cluster of 8 CTAs splitting K=8192 into 1024-wide slices = 128 CTAs. Per CTA a 3-stage cp.async.cg ring of 128-K chunks of x [64, 128] and router_hi/router_lo [BN, 128]; mma.sync m16n8k16 with hi and lo as two MMAs into one fp32 accumulator (the same hi+lo fp32-equivalent math as master); 4 warps per CTA (2x2 warp tiles). Split-K reduce: each CTA writes its fp32 tile partial to smem, barrier.cluster release/acquire, rank r reduces rows [8r, 8r+8) of the tile over ranks 0..7 in fixed order via mapa/ld.shared::cluster and stores fp32 with plain stores (deterministic); second cluster barrier before exit. torch.empty output. Rows m in 33..64 and m > 256 keep master; m 65..256 that is not a multiple of the token tile is handled by zero-fill on load and a guarded store (covers the m=130/200 tests). Self-check at m in {65, 128, 130, 200, 256} twice: rel L2 <= 2e-5 vs fp64, one-hot power-of-two probe rows within 1e-6 of the hi+lo reconstruction, run 2 bit-identical, sync+print after each launch. Diagnostics: regs/spills/smem, cudaOccupancyMaxActiveClusters, cold-L2 event time vs master _router_hilo_forward.

## What changed, and where
Edited only `muse_bf16_proj/submission.py` in the worktree (+574/-2): new section (7) before `prepare()`, an `os`/`time` import, a docstring paragraph, a call to `_router_tiled_prepare(state)` after `_router_prepare(state)`, and a gate in kernel_fn's router `hilo` branch (m > 32 only).
- CUDA kernel `router_tiled_kernel<CS>` (load_inline at import, `TORCH_CUDA_ARCH_LIST=9.0a` around the build and restored, `-O3 -lineinfo`, flags printed). One launch, `torch.empty` output, no memset, no atomics. 64x64 output tiles, one cluster per tile, CS CTAs split K (CS=8: 1024-wide, CS=16: 512-wide non-portable cluster), grid = ceil(m/64)*4*CS. Per CTA 8 warps (4 token x 2 expert warp tiles of 16x32), 4-stage cp.async.cg ring of BLOCK_K=64 (x 8 KB + hi 8 KB + lo 8 KB per stage = 96 KB), mma.sync m16n8k16 with hi then lo into one fp32 accumulator, fp32 partial tile (padded) aliased onto the ring, barrier.cluster release/acquire, rank r reduces BM/CS rows via mapa + 16-B ld.shared::cluster.v4 in fixed rank order, plain float4 stores, second cluster barrier. Rows >= m zero-filled and never stored. No mbarrier, so no spin wait to bound.
- Per-mt choice (token-tile count 2/3/4 = m 65..128/129..192/193..256) in prepare(): candidate needs no spill, dynamic smem <= 110 KB, cudaOccupancyMaxActiveClusters >= mt*4, self-check pass (m in 65,128,129,130,192,200,256, twice, rel L2 <= 2e-5, row <= 1e-4, one-hot within 1e-6 of hi+lo, bit-identical, flushed print before and sync+print after each launch). Winner = fastest candidate at the representative m (128/192/256) by cold-L2 (256 MB flush) eager median, kept only if < 0.98x master `_router_hilo_forward` (kernel + memset); else master. Prints regs/spills/smem/max-active-clusters, L2->SM bytes per tiling vs incumbent, cold-L2 medians (full, no-DSMEM-reduce mode 0, master) at m=65/128/130/192/200/256, and `PATH CHOSEN` for m=64/65/128/129/130/192/193/200/256/257. m=64 and m>256 are routed to master by `_router_tiled_route`.
- Review fixes applied (the review wins over the spec): smem 96 KB per CTA (not 3x48 KB), BLOCK_K=64 4 stages, occupancy >= need enforced else master, L2->SM bytes printed (64x64 tiling equals the incumbent's, 25.2 / 50.3 MB by my formula, it counts x and hi+lo exactly), 8 warps, vectorized DSMEM reduce, no-reduce and master cold-L2 timings, m=64 stays master.
- Deviations: (1) the spec's 64x32/cs8 tile and the optional 128x64 instance were NOT built; the 64x32 tiling moves more L2 bytes than the incumbent (review: do not ship that), so the two instances are 64x64 with CS 8 and CS 16 (the review's "64x64 split 16 cluster 16" for m=128), picked by timing per token-tile count. (2) No `-std=c++20` override (nothing needs it). (3) No wgmma, so no `__CUDA_ARCH_FEAT_SM90_ALL` guard or error flag. (4) Worktree base is c67ea86 (the claimed base_sha), which lacks stage 1's CUDA scaffold, so the scaffold's helpers were re-embedded here.
- Uncertainties: I could not compile or run anything (no GPU/torch here). Biggest risk is a CUDA fault or compile error: a fault in the self-check poisons the context (lesson 2026-10-25); a build error falls back to master. Second risk is cs=16 non-portable cluster support / occupancy (falls back to cs=8 or master by the occupancy and timing gates).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn launches via cudaLaunchKernelEx on the current stream, no sync/.item()/print in kernel_fn; allocation is `torch.empty` of a shape-only size; branches only on m, cfg, strides, alignment and a prepare() constant (checked in the diff).
- [x] Nothing derived from `data` is cached across calls: `state["router_tiled"]` is built in prepare() from synthetic x and the weights only.
- [x] Weights: unchanged hi/lo bf16 split built in prepare(); no new weight copies.
- [x] Activations BF16 (x stays bf16, cp.async straight from it); router accumulation fp32 over hi+lo as before.
- [x] Attention: n/a, router GEMM only.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, process creation or network in the file (grep of the file for process-spawning names is empty; load_inline is the same build path stage 1 used).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Router hi+lo fp32 accumulation rules unchanged; gate is shape-based (65 <= m <= 256, router only).

okc_lint output:
```
warn  host-sync    L1426: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1448: flush = torch.empty(ROUTER_TILED_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1462: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```
py_compile OK; __pycache__ removed.

## Risks / what could fail
high
Kill: neither target row below its incumbent with the new kernel in the profile, or a spill in the build line. Risk: 2x hi/lo MMA count makes m=256 ~2.1 GFLOP; mma.sync at ~400 TFLOP/s is ~5.4 us, so M3 at m=256 may need stage 2 (wgmma). Stage-1 cluster kernel ran clean on the judge (dc5d03c4), so the cluster/DSMEM instruction set is proven.

## Result (after the run)
- job `b7b2b55d04544660` (check), status done, passed True, tests 9/9
- speedup 1.2189x, 95% CI [1.2006, 1.2372], score 3963.1608306764333 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-m128-m256-cluster-dsmem-tiled-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-m128-m256-cluster-dsmem-tiled-rewrite.patch`; branch `hyp/muse_bf16_proj/router-m128-m256-cluster-dsmem-tiled-rewrite`
