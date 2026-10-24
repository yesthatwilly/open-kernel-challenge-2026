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
review: proceed
status: running
review_at: 2026-10-24T100116Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T100116Z
base_sha: 93fae4dfde47a06219a581785dc8c4ff2e7dab3b
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore
worktree: .worktrees/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore
mode: null
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

## Adversarial review

Reviewed 2026-10-24T100116Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound single-area change that targets the two largest gaps at m=128 (pooler 25.2 vs 14.6 us floor, unpooler 20.0 vs 14.6). The incumbent nvjet times are very stable across 14 runs (pooler 24.1-25.9 us, unpooler 19.8-21.9 us), so the replaced kernel's own time is a clean readout and no companion run is needed. The main risks are whether Triton's wgmma mainloop runs near peak and how long the 64 KB-per-partial reduce tail takes, both of which can be measured for free in prepare().

### Justification
Mechanism: at m=128 the arithmetic intensity is 128 FLOP per weight byte, while the H200 ridge is about 206 (989 TFLOP/s / 4.8 TB/s). The shape is memory-bound, but only by about 1.6x (8.7 us of math vs 14 us of bytes). cuBLAS's pooler choice (64x64 tile, 2x1 cluster, 128 CTAs) reaches about 2.7 TB/s, and each CTA pulls 1 MB of x from L2 per 512 KB of weight. The proposed 128x128 tile over a 2048-wide K-slice halves the x traffic per weight byte (64 MB of L2 x reads per call instead of 128 MB) and still fills one wave of 128 CTAs. The arithmetic in the hypothesis checks out: 512 KB of weight plus 512 KB of x per CTA, 67 MFLOP per CTA, 128 KB of shared memory for 4 stages. About 64 KB of weight is in flight per SM, or about 8 MB across the GPU, well above the roughly 5 MB that Little's law needs at 4.8 TB/s. FINDINGS and the result files support the premise. Both rows are single nvjet kernels at 100% of shape time, with no splitKreduce node to remove, so the gain must come from the mainloop and not from launch savings. The unpooler argument is the same at SPLIT_K=2. The expected directions agree with TASK.md: only the two verify_16 rows change, and the verify_16 load is 1 of 6. Saving about 8 us out of about 76 us on that load gives about 1.11x on it and about +1.8% on the geomean, which is inside the 1.5% CI resolution, so the result must be read per shape and per kernel, as the hypothesis already says. WEAKEST LINK: the shape is co-bound. One wave of 128 CTAs on 132 SMs needs about 9.0 us of pure wgmma time per CTA at 100% of per-SM peak. A Triton cp.async (non-TMA) pipeline with BLOCK_K=64 typically reaches 60-75% of wgmma peak, which is 12-15 us of math, and that must overlap almost perfectly with about 16 us of weight streaming at 4 TB/s. There is little slack. Each stage also fetches only 128 B per weight row (BLOCK_K=64, rows 16 KB apart), the shortest contiguous run of any Triton stream measured here; LESSONS' best stream used BLOCK_K=256. The second weakest link is the reduce tail, which this epilogue never paid at m<=32. Each partial is now 128x128 fp32 = 64 KB, 8x the m<=32 case. The winning CTA reloads SPLIT_K partials serially: 256 KB for the pooler, 128 KB for the unpooler. At roughly 100-150 GB/s of per-SM L2 bandwidth that is about 1.7-2.5 us for the pooler and 0.9-1.3 us for the unpooler, on the critical path after the single wave. With a 16 us mainloop, the pooler's <=19 us target is reachable but not comfortable, and the unpooler's <=17.5 us target is plausible. A fallback to mma.sync would land near cuBLAS, a wash and not a disaster.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity: this change replaces that dispatch for exactly these two rows. SGLang's side uses the same cuBLAS kernel, so the paired ratio is against the incumbent and is clean. Each shape is a separate CUDA graph with cold L2, so the cuBLAS path for other shapes cannot interact. Not a confounder.
- router-triton-hilo-fused-control-splitk-all-m: this change shares only the code pattern (direct transposed [BLOCK_K, BLOCK_N] addressing) and adds Triton warm-up in prepare() (compile time comes out of the 10-minute budget, about 2 more specializations). Router shapes are timed in separate graphs and have no runtime interaction with pooler or unpooler at m=128. Not a confounder.
- router-hilo-at-m8: its router branch was replaced wholesale by the split-K router. No interaction. Not a confounder.

### Interference and A/B needs
No companion is warranted. A gain is unambiguous: the readout is the _bf16_tc_splitk_kernel time in top_kernels against nvjet times that sit in a 24.1-25.9 us band (pooler) and a 19.8-21.9 us band (unpooler) across all 14 runs, Triton reproduces within about 1%, and no landed hypothesis touches these rows. Only a loss would be ambiguous: wgmma fallback or a low-efficiency mainloop versus the reduce tail. A passing companion cannot separate those, because any run that skips or alters the reduce either fails correctness or changes the tail itself. Resolve it inside this run instead, at no cost in checks. In prepare() (untimed, 300 s budget), after warm-up: (1) print whether 'wgmma' appears in the compiled kernel's PTX (kernel.asm['ptx']) and print n_regs and n_spills. This replaces the 'check the PTX before the run' step that this GPU-less node cannot do. (2) Time, with torch.cuda.Event on synthetic x, the full kernel and a constexpr variant (REDUCE=False) that stores partials and takes the ticket but skips the winner's reload and sum. Print both times. Neither print is used by kernel_fn. The difference is the tail, and the printed log_tail attributes a loss to the mainloop or the reduce. Re-zero the counters after these timings, just as after warm-up. If the implementer prefers not to touch prepare(), proceed anyway: a loss then settles the 'Triton 128x128 wgmma at m=128' family well enough to decide the follow-up (a TMA weight stream or a larger BLOCK_K), which is the purpose of the sibling pooler-splitk2-tma-weight-stream.

### Open questions
- Sequencing with the sibling qkv-gate-o-m128-wgmma-splitk-semaphore: it uses the same _bf16_tc_splitk_kernel design, so a wgmma fallback or tail problem would sink both. Should the sibling wait for this run to harvest (this one has the larger gap) instead of spending two checks on the same design risk at once? If both land, the shared kernel name and the _ws/_cnt state keys must be merged by hand.
- Is BLOCK_K=64 forced by the 4-stage shared-memory budget? Three stages at BLOCK_K=128 (64 KB per stage, 192 KB, at the 227 KB limit before epilogue scratch) would double the contiguous run per weight row. That is a natural follow-up if the mainloop, not the tail, is the problem.
- The verbatim epilogue makes the winner reload its own partial too. Keeping the winner's acc in registers and reloading only the other SPLIT_K-1 partials would cut the pooler tail by about 25% and the unpooler's by 50%, at the cost of a summation order that depends on which CTA wins (still correct, but not bitwise reproducible). This is a candidate stacked follow-up, not for this run.
- Does Triton's epilogue convert the 128x128 fp32 accumulator through shared memory for the partial store? If so, 128 KB of pipeline buffers plus 64 KB of conversion scratch is about 192 KB. That fits, but check n_spills and occupancy in the printed diagnostics.

