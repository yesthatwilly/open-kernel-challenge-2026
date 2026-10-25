# muse_bf16_proj · 2026-10-25T111928Z · submit · qkv-swapab-w128-splitk7-one-wave-m32

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T110547Z-qkv-swapab-w128-splitk7-one-wave-m32.md` (branch `hyp/muse_bf16_proj/qkv-swapab-w128-splitk7-one-wave-m32`).

- **Job:** `c7138fe05598428e` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `d634a1ef`

## Goal
Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, check 84f17adc3f074a33, 1.2366x). WHY: at m<=32 qkv_gate (n=2304, k=8192, a 37.7 MB weight, cold L2 every replay) is the one DRAM weight stream still well below the bandwidth the pooler reaches on the same kernel family: master rows 12.38/13.14/13.93/12.45 us at m=1/16/32/8 = 3.05-2.71 TB/s, against pooler 67.1 MB in 18.74 us = 3.58 TB/s. Both qkv entries (SWAPAB16_CFG['qkv_gate'] and SWAPAB_CFG['qkv_gate']) launch _bf16_swapab_kernel with BLOCK_W=64, SPLIT_K=4, BLOCK_K=128, 4 stages: grid (36, 4) = 144 CTAs on 132 SMs (12 SMs carry two CTAs, so the stream ends on a straggler half-wave) and only 64x128x2 B x 4 = 64 KB of weight in flight per CTA; LESSONS measured 64 KB in flight at ~3.3 TB/s and 128 KB at ~3.7 TB/s. CHANGE (qkv_gate only, both the m<=16 swapab16 route and the 17..32 swapab route; pooler, o, unpooler, router and every m>=128 path byte-identical): (1) give _bf16_swapab_kernel a per-layer BLOCK_W and use BLOCK_W=128 for qkv_gate (2304/128 = 18 weight tiles; tl.dot w_tile[128, BLOCK_K] @ xt_tile[BLOCK_K, BLOCK_T], still swap-AB so wgmma is emitted; num_warps 4, print regs/spills/wgmma count from the compiled kernel's asm once in prepare, as the m32 swap-AB hypothesis did). (2) SPLIT_K=7 so the grid is (18, 7) = 126 CTAs, one wave, one CTA per SM. 8192/128 = 64 K-blocks do not divide by 7, so replace the constexpr K_SLICE with an uneven but static split: add constexpr NUM_KB = K // BLOCK_K and compute kb_lo = pid_k * NUM_KB // SPLIT_K, kb_hi = (pid_k + 1) * NUM_KB // SPLIT_K (9 or 10 K-blocks per CTA) and loop kk in range(kb_lo, kb_hi); keep the K_SLICE path unchanged for the other layers (e.g. NUM_KB = 0 / a constexpr EVEN flag selects the old code so their compiled kernels are unchanged). (3) Keep BLOCK_K=128 and 4 stages: (128 + BLOCK_T) x 128 x 2 B x 4 = 144 KB (BLOCK_T 16) / 160 KB (BLOCK_T 32) of smem, i.e. 128 KB of weight in flight per CTA, the pooler's recipe; if the 4-stage compile is OutOfResources fall back to 3 stages via the existing *_FALLBACK_STAGES mechanism and print it. (4) Reduce: the existing ticket semaphore with SPLIT_K=7, the qkv workspaces resized to (7, WS_M, 2304) fp32 and the tickets to 18 int32 counters (N // BLOCK_W); fixed summation order 0..6, winner .cg reloads, self-reset, no memset node. Routing, gates and every other layer unchanged. In prepare print, per qkv route and per m in (1, 8, 16, 32), the cold-L2 (flush a > L2 buffer before each replay) CUDA-graph replay median of the new config against the old (36, 4) config on the same synthetic x (print-only, never routing), and assert torch.allclose against x.float() @ W.float().T at the reference tolerance.

## What changed, and where
Edited only `.worktrees/muse_bf16_proj/qkv-swapab-w128-splitk7-one-wave-m32/muse_bf16_proj/submission.py` (base blob 353e3b19 == current master's blob, no merge needed; 312 insertions, 81 deletions). Nothing could be run (no GPU/Triton on this node).

What changed:
- `_bf16_swapab_kernel`: new constexprs `NUM_KB` and `MAINLOOP_ONLY`. `NUM_KB == 0` takes the original code (`k_start = pid_k*K_SLICE`, `K_SLICE // BLOCK_K` iterations), so pooler/o/unpooler-slice are traced as before. `NUM_KB > 0` uses `kb_lo = pid_k*NUM_KB//SPLIT_K`, `kb_hi = (pid_k+1)*NUM_KB//SPLIT_K` and loops `range(0, kb_hi-kb_lo)` (6 slices of 9 K-blocks, pid_k=6 gets 10). `MAINLOOP_ONLY` (A/B control only) skips the ticket/reduce.
- Per-layer `BLOCK_W`: `SWAPAB_BLOCK_W_BY_KEY = {"qkv_gate": 128}`, default 64. One shared launcher `_swapab_launch` (grid `(n // block_w, split_k)`) is used by `_swapab_forward` (17..32), `_swapab16_forward` (m<=16) and the A/B.
- `SWAPAB_CFG` / `SWAPAB16_CFG` `qkv_gate` = (2304, 8192, 7, 128, 4), grid (18, 7) = 126 CTAs. A `qkv_gate_old` entry keeps master's (36, 4) config as the fallback and the A/B "old" arm.
- Buffers: `swapab_ws_qkv` (7, 32, 2304) and `swapab16_ws_qkv_gate` (7, 16, 2304); tickets `n // block_w` = 18 on both routes (`_swapab_prepare`, `_swapab16_prepare`); separate buffers for `qkv_gate_old`. All in `state`, no globals.
- Fallback: in prepare, the new qkv config's build error or a `_qkv_selfcheck` miss deletes it from the stage dict; `state["swapab_qkv_key"]` / `state["swapab16_qkv_key"]` (constants) pick `qkv_gate` or `qkv_gate_old`; `kernel_fn` reads them. Loud `PATH CHOSEN` line per route.

Adversarial-review items:
1. Kernel shape: uneven split exactly as written; the m<=16 route switched too (grid, ticket count in `_swapab16_prepare`, BLOCK_W in `_swapab16_forward`). Done.
2. Print-only A/B (`_qkv_ab_report`, state A only): old / new / mainloop-only control per route (swapab16 m in 1,8,16; swapab with BLOCK_T 32 m in 1,8,16,32, since BLOCK_T 16 cannot hold m=32), cold-L2 (256 MB flush before each replay), graph replay medians, separate ws/cnt per variant, tickets re-zeroed after every replay, flush and A/B buffers freed (finally + empty_cache). Never routes, never raises.
3. Compile checks: `_print_ptx_counts` now prints shared and, for swap-AB variants, `wgmma_ok` (wgmma>0 and mma.sync==0) with BLOCK_T and stages in the tag, for BLOCK_T 16 and 32 and every layer; non-qkv variants print n_regs/n_spills/shared (compare to master's log; I did not add a PTX hash). Stage count shown in the tag and in PATH CHOSEN.
4. Self-check: relative L2 vs x.float() @ W.float().T, bound 1e-2 (bf16 rounding is ~3e-3), two launches per m on different x, tickets must be 0 afterwards; each launch announced with flushed prints and a sync before and after. m in (17, 24, 32) and (1, 8, 16).
5. Malformed `expected` entry: treated as the two flat entries 'layer=qkv_gate load=verify_16 m=128' and 'layer=qkv_gate load=verify_32 m=256'; frontmatter left as is. Both stay on cuBLAS nvjet.
- Review's risk-text correction (only pid_k=6 gets 10 blocks) and "do not rebalance": followed. Review's "profile kernel time is L2-contaminated, use the A/B print": noted for judging.

Deviations: (a) the brief's two-state graph self-check is not possible inside one prepare for a Triton-only path; there are no module/static globals, every pointer comes from `state`, and the A/B graph replays use per-variant buffers. (b) Fallback goes to the old qkv swap-AB config (master's behaviour) instead of a non-swap-AB path.

Biggest risks: Triton may not pipeline the dynamic-trip-count loop (NUM_KB path) at 4 stages as it does the constexpr loop, or may emit mma.sync/spill for the [128, BLOCK_T] dot; both show in the compile print and the A/B.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn only launches `_bf16_swapab_kernel` via `_swapab_launch`; syncs/`.item()` live only in prepare helpers (`_qkv_selfcheck`, `_qkv_ab_report`); routing reads constants from `state`.
- [x] Nothing derived from `data` is cached across calls (diff checked): buffers/tickets are prepare-time, key choice is a prepare constant.
- [x] Weights: unchanged, bf16 read in place; the fp32 `w.float()` copy exists only inside prepare helpers and is freed.
- [x] Activations BF16, fp32 accumulation, bf16 output: unchanged.
- [x] n/a: attention not touched.
- [x] No timing patches, harness access, subprocesses or network: none added.
- [x] All work inside the timed call; no output caching or replay (the A/B outputs are dropped).
- [x] Judge two-state: no tensors/pointers in module globals (workspaces and tickets in `state`).

okc_lint: `OK with warnings` (only host-sync warnings in prepare-only code and the uint8 flush buffer dtype warning, same class as the existing ouns flush); undef.py: []; py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Biggest risk: the split-K tail. qkv-swapab-m32-splitk8-stages3 (a61778a0) lost +14% at m=32 with a flat kernel row, the loss sitting in the SPLIT_K=8 reduce/semaphore tail; here the split count drops 8 -> 7 and the tile count halves (18 winners each summing 7 partials of 128 x BLOCK_T fp32, 115 KB per winner at m=32), so a slower winner is possible: the prepare A/B print separates mainloop from tail if it lands flat. Second: uneven 9/10 K-block slices make the 10-block CTAs (8 of 126... every pid_k with an extra block) the critical path, a ~10% stretch of the per-CTA stream; still one wave instead of the current 144-CTA straggler. Third: BLOCK_W 128 at 4 warps doubles the accumulator (128 x 32 fp32 = 32 regs/thread at m=32), well below spill. Headline: qkv is on 4 of 6 loads; -1.2 to -1.9 us on a 47-60 us load is +2-4% on each, about +1.5-2.5% on the headline if all four rows land. Judge per row against the master rows above (drift +-1.5-2.5%); a row gain under ~3% needs the kernel time in top_kernels to confirm.

## Result (after the run)
- job `c7138fe05598428e` (submit), status done, passed True, tests 9/9
- speedup 1.2229x, 95% CI [1.2045, 1.2412], score 3950.4213735231624 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2490645809967658, "ci": [1.2303286122818142, 1.267800549711717], "job": "ee4f89f2f62c42d3"}; base result: muse_bf16_proj/hypotheses/done/neutral/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/qkv-swapab-w128-splitk7-one-wave-m32.result.json`, `muse_bf16_proj/hypotheses/done/neutral/qkv-swapab-w128-splitk7-one-wave-m32.patch`; branch `hyp/muse_bf16_proj/qkv-swapab-w128-splitk7-one-wave-m32`
