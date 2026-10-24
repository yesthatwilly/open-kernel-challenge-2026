---
problem: muse_bf16_proj
slug: router-triton-hilo-fused-control-splitk-all-m
created: 2026-10-23T230544Z
base: master
area: router GEMM
change: >
  Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare()
  stores router_hi and router_lo as [256, 8192] bf16 row-major; one Triton kernel for the router
  at every m; BLOCK_K=128, num_stages=3; acc += tl.dot(x_tile, tl.trans(hi_tile)) + tl.dot(x_tile,
  tl.trans(lo_tile)) in fp32; tl.atomic_add into an output kernel_fn creates with torch.zeros((m,
  256), dtype=torch.float32) immediately before the launch; bf16 layers untouched on
  _bf16_gemm_dispatch_impl; no triton.autotune; all m-buckets compiled in prepare() on synthetic
  x), with one difference: the grid is 3-D, grid = (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K) at
  every m, each program owns one (M-block, N-block, K-slice) and never loops over M, and there is
  no SPLIT_K=1 plain-store path. Static configs chosen from m at launch: m <= 16: BLOCK_M=16,
  BLOCK_N=32, SPLIT_K=16 (K-slice 512; 1 x 8 x 16 = 128 CTAs); 16 < m <= 32: BLOCK_M=32,
  BLOCK_N=32, SPLIT_K=16 (128 CTAs); 32 < m <= 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=16 (2 x 4 x 16
  = 128 CTAs at m=128); m > 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=8 (4 x 4 x 8 = 128 CTAs at
  m=256). Rows beyond m and K beyond the slice are masked, so m=7, 9 and any nearby m are valid.
  Output dtype/shape as before.
expected:
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: "128 CTAs splitting K=8192, like cuBLAS's splitK, instead of the primary's 4-8 CTAs each streaming 2.1 MB of weight plus the full x"
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: "same; x (4.2 MB) is read once per N-block across the grid rather than once per program's M loop"
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: 128 CTAs with 3 stages keep ~12 MB in flight, enough to saturate HBM for the 8.4 MB hi+lo read; one launch plus a memset replaces 3 launches (10.4 us)
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: "leaves the 26 us fp32 SIMT path; compare against router-hilo-at-m8's m=8 number to separate path change from Triton"
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: 5 launches (14.5 us) become 1 + memset with a full grid
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same at m=32
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: bf16 layers untouched; control
risk: medium
depends_on: router-triton-hilo-fused
priority: 1
kind: control
controls_for: router-triton-hilo-fused
review: proceed
status: running
review_at: 2026-10-24T024326Z
review_companions: [router-triton-hilo-fused-control-hybrid-cublas-above-m32]
stage: claimed
claimed_at: 2026-10-24T024344Z
base_sha: a392f0b6dcd3d6a023c67bc68ac2f22a8aeece5a
base_blob: 3f59d3eeb5b8a1fe83df61cbea1c119b5926bc63
branch: hyp/muse_bf16_proj/router-triton-hilo-fused-control-splitk-all-m
worktree: .worktrees/muse_bf16_proj/router-triton-hilo-fused-control-splitk-all-m
mode: null
---

# router-triton-hilo-fused-control-splitk-all-m

## Change

Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Triton kernel for the router at every m; BLOCK_K=128, num_stages=3; acc += tl.dot(x_tile, tl.trans(hi_tile)) + tl.dot(x_tile, tl.trans(lo_tile)) in fp32; tl.atomic_add into an output kernel_fn creates with torch.zeros((m, 256), dtype=torch.float32) immediately before the launch; bf16 layers untouched on _bf16_gemm_dispatch_impl; no triton.autotune; all m-buckets compiled in prepare() on synthetic x), with one difference: the grid is 3-D, grid = (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K) at every m, each program owns one (M-block, N-block, K-slice) and never loops over M, and there is no SPLIT_K=1 plain-store path. Static configs chosen from m at launch: m <= 16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=16 (K-slice 512; 1 x 8 x 16 = 128 CTAs); 16 < m <= 32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=16 (128 CTAs); 32 < m <= 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=16 (2 x 4 x 16 = 128 CTAs at m=128); m > 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=8 (4 x 4 x 8 = 128 CTAs at m=256). Rows beyond m and K beyond the slice are masked, so m=7, 9 and any nearby m are valid. Output dtype/shape as before.

## Expected

- `layer=router load=verify_16 *` faster: 128 CTAs splitting K=8192, like cuBLAS's splitK, instead of the primary's 4-8 CTAs each streaming 2.1 MB of weight plus the full x
- `layer=router load=verify_32 *` faster: same; x (4.2 MB) is read once per N-block across the grid rather than once per program's M loop
- `layer=router load=decode_1 *` faster: 128 CTAs with 3 stages keep ~12 MB in flight, enough to saturate HBM for the 8.4 MB hi+lo read; one launch plus a memset replaces 3 launches (10.4 us)
- `layer=router load=verify_1 *` faster: leaves the 26 us fp32 SIMT path; compare against router-hilo-at-m8's m=8 number to separate path change from Triton
- `layer=router load=decode_16 *` faster: 5 launches (14.5 us) become 1 + memset with a full grid
- `layer=router load=decode_32 *` faster: same at m=32
- `*layer=qkv_gate*` flat: bf16 layers untouched; control

## Risk

Separates 'a fused Triton hi/lo kernel cannot beat cuBLAS splitK + reduce + add on this shape' from 'the primary's grid under-fills the GPU (32 CTAs at m<=32, 4-8 at m>=128)'. If the primary loses at verify_16/verify_32 and this control gains there, the primary's tiling is the cause and the Triton route stays alive; if both lose, the Triton route for the router at large m goes on the do-not-retry list with evidence. If the primary gains everywhere, this control is the tuning data for the follow-up.

## Adversarial review

Reviewed 2026-10-24T024326Z by an adversarial Fable subagent (model fable).

**Verdict: proceed-with-companions.** The parent's result already explains its own regression (8 CTAs x ~60 GB/s per CTA = 49.7/98.1 us at m=128/256, while the same kernel won 1.3-3.4x at m<=32), so this run no longer disambiguates the parent; it is now the de-facto primary for 'can a filled-grid Triton split-K beat cuBLAS hi/lo (16.2/17.7 us) at m=128/256', which is worth one check slot. One companion is recommended: a hybrid (parent's Triton config at m<=32, cuBLAS hi/lo above) that turns the parent's already-measured small-m wins into a promotable kernel regardless of how the 3-D grid fares at large m.

### Justification
Mechanism is plausible and the numbers support it. From done/regression/router-triton-hilo-fused.result.json (job c2a8de5969374dff) the parent's 2-D grid gave router m=1 7.7 us (kernel 5.3 + memset 0.7; master 10.5), m=8 7.6 (master after router-hilo-at-m8: 14.2), m=16 8.0 (14.5), m=32 8.4 (15.0) with 32 CTAs/SPLIT_K=8, but m=128 49.7 us and m=256 98.1 us with 8 CTAs/SPLIT_K=1 and M looped. Per-CTA arithmetic matches: ~61-62 GB/s per CTA. The loss is per-CTA bandwidth times an 8-CTA grid, not 'Triton vs cuBLAS'. The control's 128-CTA 3-D grid should bring m>=128 back to the 5-10 us range; floors are 2.2/2.7 us and master's cuBLAS hi/lo path is 16.2/17.7 us, so ~9-10 us per shape to win on 2 of 6 loads (~4% geomean). Notes: (1) expected 'why' lines are stale against master (verify_1 is 14.2 us now); the control's marginal change at m<=32 is BLOCK_N 64->32, SPLIT_K 8->16, 32->128 CTAs, not obviously better than the parent's 5.3-6.4 us there. (2) At m=128/256 atomic traffic ~2 MB per launch and the weight is re-read once per M-block relying on L2 after the cold-L2 flush. (3) Weakest link: SPLIT_K=16 at m=128 is untuned; a result between 50/98 us and 16/18 us says 'this config loses', not 'Triton loses'.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (965b046): inert on the router path; compare per-shape against master's lineage, not BEST.json's starter.
- router-hilo-at-m8 (f9dc05d): no code interaction (its constant is deleted by this change) but it moves the comparison point for router verify_1 m=8 from 26.1 us to 14.2 us; harvest must use aa3d033c208e4511's per-shape medians.
- router-hilo-at-m8 promotion mechanics: build from master, not from the parity branch, or promotion conflicts on the docstring/ROUTER_FP32_MAX_TOKENS hunks.
- Run noise: the parent's untouched bf16 rows swung -2.2% to +4.4% in one run; router rows moving less than ~4% vs the parent are not evidence of a config difference.

### Interference and A/B needs
ab-revert/ab-tune of the landed items is inapplicable (nothing they change survives this change). A single run is interpretable per shape: m<=32 rows vs the parent's rows, m>=128 rows vs the parent and master's cuBLAS. The ambiguity is in promotion: the likely mixed outcome leaves no run with a promotable kernel even though the parent measured 2.6-6.6 us/shape wins at m<=32. One companion fixes that: the parent's kernel at m<=32 with cuBLAS hi/lo at m>32. No second companion.

### Companions queued
- `router-triton-hilo-fused-control-hybrid-cublas-above-m32`: The promotable composite of what has already been measured: if the control's m>=128 rows lose to cuBLAS, this run carries the parent's 2.6-6.6 us/shape small-m wins onto master (~1.05x over master's 1.0219x by load arithmetic). If the control wins everywhere it is superseded at the cost of one check slot. Its m<=32 rows are also the same-day baseline for the control's small-m config.

### Open questions
- Build from master (f9dc05d), not from the parity branch as the change paragraph says.
- Attribution base for router verify_1 m=8 must be router-hilo-at-m8's 14.2 us (job aa3d033c208e4511), not the parity result's 26.1 us.
- Read the control's m<=32 rows against the parent's (7.7/7.6/8.0/8.4 us), not only against master.
- From the profile, record the memset and _router_hilo_kernel time at m=128/256 separately; if the kernel is 6-9 us but the row is >12 us, the atomics epilogue or memset is the cost.
- If verify_32 is much worse than verify_16 relative to cuBLAS, swap the grid order so the M-blocks of one (N-block, K-slice) run concurrently.
- Confirm m=128 lands in the SPLIT_K=16 bucket (2x4x16).
- router-hilo-addmm-epilogue remains relevant only for m>=128 if the hybrid lands and the 3-D grid does not; deprioritize rather than run in parallel.

