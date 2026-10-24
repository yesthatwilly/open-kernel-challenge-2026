---
problem: muse_bf16_proj
slug: bf16-shape-gated-triton-winners-combo
created: 2026-10-24T112147Z
base: master
area: bf16 GEMM dispatch (shape-gated Triton routing, m<=16)
change: >
  Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched).
  Combine, behind one shape gate in kernel_fn, the four individually-neutral Triton kernels ONLY
  on the shapes where each one's own kernel beat cuBLAS in its own run (paired ratio vs SGLang > 1
  and kernel time in top_kernels below nvjet+splitKreduce); every other shape keeps
  _bf16_gemm_dispatch_impl. Code is copied verbatim from the harvested patches in
  muse_bf16_proj/hypotheses/done/neutral/: (1) pooler, cfg m <= 16 (decode_1, verify_1, decode_16;
  NOT m=32, which read 0.994x): _bf16_splitk_kernel + _pooler_forward from
  pooler-triton-splitk2-blockn64-semaphore.patch (grid (64, 2), BLOCK_N=64, BLOCK_K=128,
  SPLIT_K=2, 4 warps, 4 stages, BLOCK_M=16, ticket-semaphore last-CTA reduce, ws fp32 [2, 16,
  4096], cnt int32 [64]); change POOLER_MAX_M to 16. (2) qkv_gate, cfg m <= 16 (NOT m=32, which
  read 0.858x): the same _bf16_splitk_kernel (it is the same design as _qkv_splitk_sem_kernel in
  qkv-gate-triton-splitk4-blockn64-semaphore.patch; either kernel body is acceptable, the
  constants must match that patch: N=2304, grid (36, 4), SPLIT_K=4, BLOCK_N=64, BLOCK_K=128, 4
  warps, 4 stages, BLOCK_M=16), its own ws fp32 [4, 16, 2304] and cnt int32 [36]. (3) unpooler,
  cfg n == 1024 (the sliced case, m <= 8: decode_1, verify_1): _bf16_splitk_kernel launch from
  unpooler-slice-triton-splitk8-semaphore.patch (w = state[unpooler][:1024], stride_wn 4096, grid
  (16, 8), SPLIT_K=8, BLOCK_N=64, BLOCK_K=128, 4 warps, 3 stages, BLOCK_M=16), its own ws fp32 [8,
  16, 1024] and cnt int32 [16]. Unpooler at n == 8192 stays cuBLAS (its Triton m=16 row was only
  1.014-1.033x, m=32 lost). (4) o, cfg m <= 16 (decode_1, verify_1, decode_16; NOT m=32, which
  read 0.907x): _pooler_skinny_kernel from pooler-triton-skinny-m32.patch launched as in
  o-triton-skinny-m32.patch (full K=1024, no split, grid (8192 // 64,) = 128 programs, BLOCK_N=64,
  BLOCK_K=256, 4 warps, 4 stages, BLOCK_M=16). Keep separate ws/cnt tensors per layer (no sharing)
  so each path is byte-for-byte the measured one. prepare(): allocate all ws/cnt, warm every
  routed specialization on synthetic bf16 x for m in 1..16 (pooler, qkv_gate, o) and 1..8
  (unpooler slice), then zero_() every cnt. Route by cfg layer/m/n and the shape preconditions
  only (no data-dependent branch). The 9 tests: m=1 qkv_gate, m=7 pooler, m=5 unpooler (slice)
  take Triton paths; m=33 o, m=130 unpooler, m=512 qkv_gate, m=1000 unpooler stay cuBLAS. If
  pooler-splitk2-tma-weight-stream or pooler-triton-splitk2-blockn64-semaphore-control-blockk256
  is harvested as a gain before this is claimed, use that pooler variant in (1) and say so in the
  attempt file.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: split-K2 semaphore kernel 19.5 us vs nvjet 18.3 + splitKreduce 1.6; own run 1.071x vs SGLang, -1.0 us vs master (21.2)
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: own run 1.068x vs SGLang, 19.9 vs master 21.3 us
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: own run 1.059x vs SGLang, 20.7 vs master 21.3 us
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: split-K8 kernel 4.6 us replaces nvjet 4.0 + splitKreduce 1.5 and one graph node; own run 1.169x vs SGLang, 5.8 vs master 7.1 us
  - shapes: layer=unpooler load=verify_1 m=8
    direction: faster
    why: own run 1.148x vs SGLang, 6.3 vs master 7.0 us
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: faster
    why: split-K4 kernel 11.6 us vs nvjet 10.8 + reduce 1.7; own run 1.029x, 13.8 vs master 14.4 us (small, near drift)
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: own run 1.044x, 13.6 vs master 14.0 us
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: own run 1.024x, 13.7 vs master 13.8 us (marginal)
  - shapes: layer=o load=decode_1 m=1
    direction: faster
    why: skinny full-K kernel 4.9 us vs nvjet 5.0; own run 1.085x vs SGLang, 6.4 vs master 6.7 us
  - shapes: layer=o load=decode_16 m=16
    direction: faster
    why: own run 1.069x vs SGLang, 6.3 vs master 6.8 us
  - shapes: layer=o load=verify_1 m=8
    direction: flat
    why: own run 1.028x vs SGLang but master already 6.3 us; at most a small gain
  - shapes: layer=* load=decode_32 m=32
    direction: flat
    why: every layer stays on master at m=32 (each Triton kernel lost there)
  - shapes: layer=* load=verify_16 m=128
    direction: flat
    why: cuBLAS / master router
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: cuBLAS / master router
  - shapes: layer=unpooler load=decode_16 m=16
    direction: flat
    why: stays cuBLAS (n=8192)
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: hold
review_at: 2026-10-24T112458Z
review_companions: null
hold_reason: "reviewer: dispatch after pooler-splitk2-tma-weight-stream (4e89311eefdc4bc3) and its BLOCK_K=256 control (75d595a29a4c4b57) harvest; choose the pooler kernel by their m<=16 pooler rows, then unhold"
---

# bf16-shape-gated-triton-winners-combo

## Change

Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). Combine, behind one shape gate in kernel_fn, the four individually-neutral Triton kernels ONLY on the shapes where each one's own kernel beat cuBLAS in its own run (paired ratio vs SGLang > 1 and kernel time in top_kernels below nvjet+splitKreduce); every other shape keeps _bf16_gemm_dispatch_impl. Code is copied verbatim from the harvested patches in muse_bf16_proj/hypotheses/done/neutral/: (1) pooler, cfg m <= 16 (decode_1, verify_1, decode_16; NOT m=32, which read 0.994x): _bf16_splitk_kernel + _pooler_forward from pooler-triton-splitk2-blockn64-semaphore.patch (grid (64, 2), BLOCK_N=64, BLOCK_K=128, SPLIT_K=2, 4 warps, 4 stages, BLOCK_M=16, ticket-semaphore last-CTA reduce, ws fp32 [2, 16, 4096], cnt int32 [64]); change POOLER_MAX_M to 16. (2) qkv_gate, cfg m <= 16 (NOT m=32, which read 0.858x): the same _bf16_splitk_kernel (it is the same design as _qkv_splitk_sem_kernel in qkv-gate-triton-splitk4-blockn64-semaphore.patch; either kernel body is acceptable, the constants must match that patch: N=2304, grid (36, 4), SPLIT_K=4, BLOCK_N=64, BLOCK_K=128, 4 warps, 4 stages, BLOCK_M=16), its own ws fp32 [4, 16, 2304] and cnt int32 [36]. (3) unpooler, cfg n == 1024 (the sliced case, m <= 8: decode_1, verify_1): _bf16_splitk_kernel launch from unpooler-slice-triton-splitk8-semaphore.patch (w = state[unpooler][:1024], stride_wn 4096, grid (16, 8), SPLIT_K=8, BLOCK_N=64, BLOCK_K=128, 4 warps, 3 stages, BLOCK_M=16), its own ws fp32 [8, 16, 1024] and cnt int32 [16]. Unpooler at n == 8192 stays cuBLAS (its Triton m=16 row was only 1.014-1.033x, m=32 lost). (4) o, cfg m <= 16 (decode_1, verify_1, decode_16; NOT m=32, which read 0.907x): _pooler_skinny_kernel from pooler-triton-skinny-m32.patch launched as in o-triton-skinny-m32.patch (full K=1024, no split, grid (8192 // 64,) = 128 programs, BLOCK_N=64, BLOCK_K=256, 4 warps, 4 stages, BLOCK_M=16). Keep separate ws/cnt tensors per layer (no sharing) so each path is byte-for-byte the measured one. prepare(): allocate all ws/cnt, warm every routed specialization on synthetic bf16 x for m in 1..16 (pooler, qkv_gate, o) and 1..8 (unpooler slice), then zero_() every cnt. Route by cfg layer/m/n and the shape preconditions only (no data-dependent branch). The 9 tests: m=1 qkv_gate, m=7 pooler, m=5 unpooler (slice) take Triton paths; m=33 o, m=130 unpooler, m=512 qkv_gate, m=1000 unpooler stay cuBLAS. If pooler-splitk2-tma-weight-stream or pooler-triton-splitk2-blockn64-semaphore-control-blockk256 is harvested as a gain before this is claimed, use that pooler variant in (1) and say so in the attempt file.

## Expected

- `layer=pooler load=decode_1 m=1` faster: split-K2 semaphore kernel 19.5 us vs nvjet 18.3 + splitKreduce 1.6; own run 1.071x vs SGLang, -1.0 us vs master (21.2)
- `layer=pooler load=verify_1 m=8` faster: own run 1.068x vs SGLang, 19.9 vs master 21.3 us
- `layer=pooler load=decode_16 m=16` faster: own run 1.059x vs SGLang, 20.7 vs master 21.3 us
- `layer=unpooler load=decode_1 m=1` faster: split-K8 kernel 4.6 us replaces nvjet 4.0 + splitKreduce 1.5 and one graph node; own run 1.169x vs SGLang, 5.8 vs master 7.1 us
- `layer=unpooler load=verify_1 m=8` faster: own run 1.148x vs SGLang, 6.3 vs master 7.0 us
- `layer=qkv_gate load=decode_1 m=1` faster: split-K4 kernel 11.6 us vs nvjet 10.8 + reduce 1.7; own run 1.029x, 13.8 vs master 14.4 us (small, near drift)
- `layer=qkv_gate load=verify_1 m=8` faster: own run 1.044x, 13.6 vs master 14.0 us
- `layer=qkv_gate load=decode_16 m=16` faster: own run 1.024x, 13.7 vs master 13.8 us (marginal)
- `layer=o load=decode_1 m=1` faster: skinny full-K kernel 4.9 us vs nvjet 5.0; own run 1.085x vs SGLang, 6.4 vs master 6.7 us
- `layer=o load=decode_16 m=16` faster: own run 1.069x vs SGLang, 6.3 vs master 6.8 us
- `layer=o load=verify_1 m=8` flat: own run 1.028x vs SGLang but master already 6.3 us; at most a small gain
- `layer=* load=decode_32 m=32` flat: every layer stays on master at m=32 (each Triton kernel lost there)
- `layer=* load=verify_16 m=128` flat: cuBLAS / master router
- `layer=* load=verify_32 m=256` flat: cuBLAS / master router
- `layer=unpooler load=decode_16 m=16` flat: stays cuBLAS (n=8192)
- `layer=router *` flat: untouched

## Risk

Per-load arithmetic from the measured rows: decode 1 saves ~3.5 us of 55 (pooler 1.2, unpooler 1.3, qkv 0.6, o 0.4), verify 1 ~2.5 of 54, decode 16 ~1.5 of 66, the other three loads unchanged: geomean ~1.02x over master, ~1.17-1.18x headline (pooler split-K2 alone already read 1.166x). Each component was within the CI alone; together the summed per-shape wins should clear it. Risks: (a) the qkv_gate and o-verify_1 wins are 2-4%, near the 1.5-2.5% per-shape drift, so they may read flat; (b) three ticket-semaphore kernels in one file: each has its own counters and self-resets, and a missed reset would hang or corrupt only its own layer (the tests catch it); (b2) o-triton-skinny-m32 was measured on an older base (d2840d, pre-router-split-K), the others on dba4ffd-era bases; the o kernel is independent of the router, so this should not matter; (c) per-shape rows are the evidence: if a component reads below 1.0 vs SGLang here, the attribution names it and the next run drops it. A reviewer may want an ab-revert dropping qkv_gate (the weakest component).

## Adversarial review

Reviewed 2026-10-24T112458Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound and well-attributed: each judge shape is its own CUDA graph, so four layer-disjoint, shape-gated Triton paths should add up per load. Re-deriving from the five result.json files gives about +2.3% geomean over master (decode_1 ~6%, verify_1 ~5%, decode_16 ~3%, other loads flat), which is above the ~1.5% CI. No companion is needed because the per-shape rows already attribute each component. Dispatch it after the two in-flight pooler runs harvest, so pooler slot (1) uses the better variant and no check is spent twice.

### Justification
Mechanism: at m<=16 these are pure weight streams (pooler 67 MB, qkv_gate 37.7 MB, o 16.8 MB, unpooler slice 8.4 MB). cuBLAS runs nvjet plus a separate splitKreduce, which adds one kernel and one graph node (LESSONS: ~2-4 us per nvjet+reduce pair). The ticket-semaphore last-CTA reduce removes that node, and the full-K o kernel keeps 128 CTAs with 128-160 KB in flight. Both are plausible on H200, and the component runs confirm it. I re-checked them against master's check run (240eb/bd9f era, 1.151x). Paired ratios vs SGLang: pooler 1.071/1.068/1.059 at decode_1/verify_1/decode_16, against master 1.006/1.011/1.009 and ~0.99-1.00 in the other runs, so +5-7% on 20-21 us = 1.1-1.4 us per shape; this is the largest absolute lever and still 6 us above the 14 us floor. Unpooler slice 1.169/1.148 vs ~0.99-1.01, so ~1 us on a 7 us shape (floor 1.7). o 1.085/1.069 at decode_1/decode_16 vs ~1.00-1.01, ~0.4-0.5 us; verify_1 1.028 vs 1.020 is flat, as the hypothesis says. qkv_gate 1.029/1.044/1.024 vs a cross-run norm of ~1.00-1.01 (master's own 0.981 at decode_1 was a low draw, so the stated 14.4->13.8 overstates it), worth 0.2-0.45 us. Load sums with weights all 62: decode_1 ~55 -> ~52 us (~6%), verify_1 ~54 -> ~51.5 (~5%), decode_16 ~66 -> ~64 (~3%), decode_32/verify_16/verify_32 unchanged. Geomean ~exp((.058+.049+.030)/6) ~ 1.023, so ~1.17-1.18 for the run, against BEST 1.143 [1.126,1.160]. The expected shapes and directions in the frontmatter match this. Excluding m=32 is right: qkv 0.858, o 0.907 and pooler 0.994 there. Excluding unpooler n=8192 is right: 1.002-1.009 at m=16. Weakest link: the ~2.3% predicted gain rests on summing several 1-7% per-shape deltas, each measured in a different run against a single master run. The qkv_gate and o-verify_1 parts (2-4%) sit inside the 1.5-2.5% per-shape drift. Also, router verify_1 read 4.26-4.44x in all three dba4ffd-based component runs vs 4.61x on master (5.9-6.2 vs 5.7 us). verify_1 is the load most sensitive to the router (SGLang's router there is 26 us), so if that drop is systematic rather than drift it would take ~0.3-0.5 us off verify_1's ~2.8 us gain. The combined geomean should still clear the CI.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: no confound. Router shapes are separate graphs and the router path is untouched. Its only coupling is the shared prepare() and caching-allocator state, which could explain the repeated router verify_1 dip in the component runs; that dip is drift-sized and on a shape this change does not route.
- sglang-gemm-dispatch-parity: not a confounder but the incumbent. Every non-gated shape keeps _bf16_gemm_dispatch_impl, so a loss on a gated shape means Triton lost to nvjet. The gating rule was chosen from the same rows, so a mild winner's-curse regression toward ~1.0 is expected on the 2-4% components (qkv_gate, o verify_1).
- router-hilo-at-m8: superseded wholesale by the split-K router on master. It has no effect on any bf16-layer shape.

### Interference and A/B needs
No companion is warranted. Each judge shape is timed as its own CUDA graph with a cold L2, and the four routed paths touch disjoint (layer, m) shapes. The per-shape rows of this single run therefore attribute each component directly. A gain on pooler rows is the pooler kernel; a qkv_gate row reading <=1.0 says drop qkv_gate. Neither a gain nor a loss is ambiguous at the component level. The ab-revert the author floats (drop qkv_gate) would only replay rows this run already reports, would cost one of bf16's 4 remaining checks, and cannot resolve a 2-4% effect better than the paired rows do. The landed hypotheses act on other shapes (router) or are the incumbent dispatch, so reverting or retuning them tells nothing about these kernels. Timing: do not dispatch while pooler-splitk2-tma-weight-stream and pooler-triton-splitk2-blockn64-semaphore-control-blockk256 are in the judge; wait one harvest. Pooler is the largest component (~1.2-1.4 us per shape), so choosing its variant decides most of the headline, and re-running the combo for a better pooler later would cost a second check. Choose the variant by its m<=16 pooler rows (paired ratio and top_kernels time), not by 'harvested as a gain'. A single-layer pooler change will almost certainly classify neutral vs BEST even if its kernel is faster, so the spec's trigger as written will never fire. The router split-K32 runs in flight need no wait: router is untouched here and their result composes independently.

### Open questions
- Is the router verify_1 dip in all three dba4ffd-based component runs (4.26-4.44x vs master 4.61x) drift or systematic? If this run shows it again on an untouched shape, compare against the BEST submit run's per-shape router row before attributing any verify_1 change.
- If both in-flight pooler variants harvest, pick the one with the lower m<=16 pooler kernel time. If TMA wins only at m=32, which is excluded here, keep the split-K2 BLOCK_K=128 baseline.
- Should the qkv_gate slot go in at all? A component expected at ~+0.3 us per shape adds a third semaphore kernel and more compile time for under 0.2% geomean. Keeping it is defensible because the per-shape rows will settle it, but a follow-up should drop it if any of its three rows reads below ~1.01.

