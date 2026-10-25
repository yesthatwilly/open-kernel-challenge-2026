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
review: proceed
status: running
review_at: 2026-10-25T111210Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T111210Z
base_sha: 848976650adeb8a2fc5263b71136dbf192bebf90
base_blob: 353e3b19a7edd5cb879e202d78097f474bcce207
branch: hyp/muse_bf16_proj/qkv-swapab-w128-splitk7-one-wave-m32
worktree: .worktrees/muse_bf16_proj/qkv-swapab-w128-splitk7-one-wave-m32
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T111928Z
impl_commit: d634a1ef1fc94dd7c0db05612c7c16aadf874f8b
submission_sha256: 8aefb6e42b790ebe4bc8c6a2906d0029247c293c95ff1d096945e83fcd46ab60
norm_sha256: 178db6026e557087ce1fedba1a0eaf582876420f94ed134cb9e4d6b680c8d9c7
submission_blob: b3a2f3b5156edd61c70c5007840f11068f7afb95
lint_warnings: 20
attempt_file: 2026-10-25T111928Z-submit-qkv-swapab-w128-splitk7-one-wave-m32.md
prepared_at: 2026-10-25T111928Z
job: c7138fe05598428e
dispatched_at: 2026-10-25T111935Z
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

## Adversarial review

Reviewed 2026-10-25T111210Z by an adversarial subagent (model opus).

**Verdict: proceed.** This retunes only qkv_gate's own swap-AB kernel at m<=32 (BLOCK_W 128, an uneven SPLIT_K 7, a one-wave 126-CTA grid, 128 KB in flight per CTA), so it is readable on the four qkv rows and the prepare-time cold-L2 A/B. It does not duplicate qkv-swapab-m32-splitk8-stages3, which went the other way (more, smaller CTAs). No companion is worth a scored submit, but the headline alone is uninterpretable, so judge it per row plus the prepare print. The malformed flat glob should be split into two entries: 'layer=qkv_gate load=verify_16 m=128' and 'layer=qkv_gate load=verify_32 m=256'.

### Justification
Arithmetic checks out. 2304/128 = 18 tiles x 7 = 126 CTAs, one per SM. 64 K-blocks split as kb_lo = pid_k*64//7 gives 0,9,18,27,36,45,54,64: six slices of 9 and only pid_k=6 with 10. That is 18 critical-path CTAs at 320 KB vs a 299 KB mean, about +7%, not 'every pid_k with an extra block' as the risk text says. Smem: (128+16)x128x2x4 = 144 KB at BLOCK_T 16 and 160 KB at BLOCK_T 32, both under 227 KB, 1 CTA/SM. The accumulator is 128x32 fp32 / 128 threads = 32 regs at m=32, so spills are unlikely. In-flight weight per GPU rises from ~9.2 MB (120x64 KB + 12x128 KB, the 12 doubled SMs run both CTAs concurrently) to ~16 MB, the pooler's figure (128 CTAs x 64x256x2x4). The mechanism is plausible. On these shapes LESSONS measured 64 KB/CTA at ~3.3 TB/s and 128 KB/CTA at ~3.73 TB/s.

Weakest link: the size of the gain, and how it will be judged. (1) The straggler half-wave argument is weak. The 12 SMs holding two CTAs run them co-resident (96 KB smem each), not serially, and per-SM L2->SM bandwidth (~126 GB/s) is not the limit for 512 KB. The real lever is total bytes in flight vs DRAM latency, which is a smaller and less certain effect than 'one wave' suggests. (2) The targets need row-level 3.34/3.25/3.09 TB/s at m=1/16/32 (12.38->11.3, 13.14->11.6, 13.93->12.2). The pooler row reaches 3.58, so m=32 (the worst row, 2.71 TB/s) is the most likely to land and m=1/8 the least, because their rows are already at 3.0 TB/s. (3) The 'Judge... a row gain under ~3% needs the kernel time in top_kernels to confirm' rule is unsafe for qkv. The 37.7 MB weight fits H200's 50 MB L2, and the profile's qkv kernel times (9.7/10.2/11.5/9.7 us = 3.3-3.9 TB/s) sit 2.4-2.9 us under the cold rows, while the 67 MB pooler's kernel time is about equal to its row. So the profile kernel time for qkv is L2-contaminated and cannot confirm a DRAM-stream gain. Use the prepare cold-L2 graph-replay A/B (new vs (36,4)) as the confirmation, keeping in mind that LESSONS says prepare event timings undercount by 20-25% (good for ranking, not absolute). (4) The headline estimate of about +1.5% (4 of 6 loads, -1.1 to -1.7 us on 47-60 us loads) is inside the +-1.5% CI and inside the 1% check-vs-submit drift master itself showed (1.2366 check vs 1.2491 submit). Spent as a scored submit, it can only top BEST (1.2491, CI upper 1.2678) if every qkv row lands at target and drift is kind. Expected shapes and directions agree with TASK.md: all qkv rows have weight 62, the floors are 7.9/7.9/8.0/7.9 us, and m>=128 stays on cuBLAS. The decode_* m=* glob correctly covers decode_1/16/32.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (landed): introduced _bf16_swapab_kernel and the qkv (36,4) tuple this retunes. It is the base, not a confounder, but the kernel body is shared with pooler/o/unpooler. The NUM_KB/EVEN constexpr edit recompiles every caller, so 'byte-identical' for other layers must be shown, not assumed: print each non-qkv variant's n_regs/spills/shared and a hash of its PTX in prepare and diff them against master's log.
- bf16-restack-m16-swapab-router-swapab-m32 (landed, carries bf16-m16-swapab-wgmma-extend): put qkv m<=16 on the same kernel at BLOCK_T 16 with its own swapab16_ws_qkv_gate / swapab16_cnt_qkv_gate buffers. This hypothesis changes that route too, so the m<=16 grid, ticket count (currently n // SWAPAB_BLOCK_W in _swapab16_prepare) and BLOCK_W argument in _swapab16_forward must all switch to the per-layer BLOCK_W. Otherwise the reduce reads the wrong counters or runs 36 counters against 18 tiles (harmless) and the ws stride stays correct. Not a measurement confounder.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (current master): touches router m<=8, the unpooler slice m<=8 (ouns_kernel) and pooler/unpooler m=65..128. It shares no kernel, buffer or launch with the qkv m<=32 path, so it does not confound. Its rows are the right in-run drift reference.
- Run-to-run baseline drift, not a landed change, is the dominant confounder. In qkv-swapab-m32-splitk8-stages3 (a61778a0) the untouched o/router/unpooler-slice rows moved -5% to -9% alongside the qkv loss. Read each qkv row against the same run's untouched o and router rows at the same load, not against the master numbers alone.

### Interference and A/B needs
No companion. Every bf16 run left today is a scored submit, so each companion costs a real attempt, and neither candidate would disambiguate more than an in-run print does. (i) ab-revert of a landed patch makes no sense: the change retunes only its own kernel's qkv tuples, and no landed hypothesis alters the qkv m<=32 numerics or launch sequence. (ii) The one real ambiguity is internal: the change bundles BLOCK_W 128 (two wgmma m64 per k-step on one warpgroup), SPLIT_K 7 with an uneven split and a 7-way reduce, and the one-wave grid. A flat or lost row could come from the wider tile or from the reduce tail. The split-out that matters, a (36,4) BLOCK_W 64 with BLOCK_K 256 variant (128 KB in flight without changing the tile or split), costs nothing to time in prepare. Recommendation for the builder: extend the print-only cold-L2 A/B to three variants per route and m in (1,8,16,32): old (BLOCK_W 64, SPLIT_K 4, BLOCK_K 128), new (BLOCK_W 128, SPLIT_K 7, BLOCK_K 128), and a mainloop-only control (BLOCK_W 128, SPLIT_K 7, partial store with no ticket/reduce, output discarded) to price the tail. This stays print-only and never routes, and its compile cost (3 compiles per BLOCK_T) fits the 300 s prepare and 10 min run budget. That separates mainloop from tail in the same run, which is what a companion would have done.

### Open questions
- Malformed glob: the third expected entry 'layer=qkv_gate load=verify_* m=12[8] layer=qkv_gate load=verify_32 m=256' is two shape keys jammed into one string, so it matches no key. The builder should treat it as two flat entries: 'layer=qkv_gate load=verify_16 m=128' and 'layer=qkv_gate load=verify_32 m=256'. Both stay on cuBLAS nvjet and are untouched. Do not let 'load=verify_*' widen it, because verify_1 m=8 is expected faster.
- Does Triton emit wgmma for tl.dot with a [128, BLOCK_K] LHS at num_warps 4, as two m64 instructions, without falling back to mma.sync or splitting the accumulator awkwardly? The prepare print must show wgmma>0 and mma.sync==0 for both BLOCK_T 16 and 32 before trusting any timing.
- Is the 4-stage BLOCK_T 32 variant (160 KB plus the reduce's register/smem scratch) actually compiled at 4 stages, or silently at the 3-stage fallback? Print the stage count chosen per route.
- If m=32 lands but m=1/8 are flat, a per-m gate (new config at 17..32 only) is the natural follow-up, not a retry.
- The hypothesis's risk text says '8 of 126... every pid_k with an extra block'. With the stated formula only pid_k=6 (18 CTAs) gets 10 K-blocks. The builder should implement the formula as written and not try to 'balance' it differently.


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn only launches `_bf16_swapab_kernel` via `_swapab_launch`; syncs/`.item()` live only in prepare helpers (`_qkv_selfcheck`, `_qkv_ab_report`); routing reads constants from `state`.
- [x] Nothing derived from `data` is cached across calls (diff checked): buffers/tickets are prepare-time, key choice is a prepare constant.
- [x] Weights: unchanged, bf16 read in place; the fp32 `w.float()` copy exists only inside prepare helpers and is freed.
- [x] Activations BF16, fp32 accumulation, bf16 output: unchanged.
- [x] n/a: attention not touched.
- [x] No timing patches, harness access, subprocesses or network: none added.
- [x] All work inside the timed call; no output caching or replay (the A/B outputs are dropped).
- [x] Judge two-state: no tensors/pointers in module globals (workspaces and tickets in `state`).

okc_lint: `OK with warnings` (only host-sync warnings in prepare-only code and the uint8 flush buffer dtype warning, same class as the existing ouns flush); undef.py: []; py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus), by reading the diff against master blob 353e3b19; the kernel could not be run on this node. **Verdict: dispatch.** No fixes were needed.

- The `NUM_KB == 0` / `MAINLOOP_ONLY == False` branches are static constexpr ifs. Every other config splits evenly: pooler 8192/(2x256), o 1024/256, unpooler 4096/256, unpooler_slice 4096/(8x128) and qkv_gate_old 8192/(4x128). So pooler, o, unpooler and the slice trace the original `k_start = pid_k*K_SLICE`, `range(0, K_SLICE // BLOCK_K)` code. Only qkv_gate (8192 % 896 != 0) takes NUM_KB = 64, which splits as 9x6 + 10 at pid_k = 6.
- Both routes use one launcher with grid `(n // block_w, split_k)` = (18, 7). There are 18 tickets on both routes (`_swapab_prepare` and `_swapab16_prepare` use `_swapab_block_w`). The workspace stride is WS_M*N = 16 or 32 x 2304, which BLOCK_W does not affect. The old config has its own ws and cnt.
- Fallback: a build error in the new qkv config, or a self-check miss, removes it from the stage dict. The route then reads `swapab*_qkv_key` = qkv_gate_old (master's config), and a PATH CHOSEN line prints for each route. The self-check bound (rel L2 1e-2 vs fp32) and the zero-ticket check can pass, since bf16 output rounding is about 1e-3.
- kernel_fn reads only state constants: no try/except, no prints, no syncs. The data_ptr checks are pre-existing alignment checks, not pointer-equality gates. State B rebuilds and self-checks independently.
- Prepare budget: about 4 extra Triton compiles (new BLOCK_T 16 and 32, plus the two MAINLOOP_ONLY controls). The A/B is 21 graphs x 25 replays with a 256 MB memset, well under a second, against the current ~1 s per swapab warm-up stage. That fits easily in 300 s.
- Cleanup: tickets are re-zeroed after every replay and again in `finally` (both routes, both keys). The flush buffer and graphs are dropped and `empty_cache` runs. The selfcheck prints and syncs are prepare-only.
- Lint passes with warnings only (host-sync in prepare code, the existing ouns uint8 flush). undef.py reports [] and py_compile is OK.

Residual risk: Triton may pipeline the dynamic-trip-count loop or the [128, BLOCK_T] wgmma poorly, which would make the qkv rows flat or slower. That is a performance risk, not a correctness one, and the self-check catches a numerics miss.
