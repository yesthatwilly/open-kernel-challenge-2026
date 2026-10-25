---
problem: muse_attention_decode
slug: epilogue-gate-l2-prefetch-cuda
created: 2026-10-25T150036Z
base: master
area: output norm-gate (gate operand prefetch in the CUDA epilogues)
change: >
  Built on current master eb1b68b (no depends_on). Mechanism: both landed CUDA kernels read the
  output gate logits (qkv row r, bf16 columns (HQ + 2*HKV)*D .. +HQ*D, i.e. 2 KB = 16 aligned
  128-byte lines per row) only in their final epilogue, after the last merge, and the judge
  flushes L2 before every replay, so that load is a cold HBM round trip (~0.7-1 us) on the
  critical path of the last CTA. The q / new-k / v loads at kernel start touch other lines of the
  same row, so the gate lines are never warm. Fix: issue L2 prefetches for exactly those gate
  lines at kernel entry, so the epilogue load hits L2. No other change; no math, layout, rounding,
  grid or route changes. Use one helper in each CUDA source: __device__ __forceinline__ void
  prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); } (plain
  prefetch.global.L2, no evict_* / createpolicy cache-hint form). (1) sliding_cluster_kernel
  (_SLIDE_CUDA; master routes sliding decode B=1 at C=16 and B=17..32 at C=4): immediately after b
  / rank are computed and before the prologue page-index loads, let NR = HQ / C; threads tid < NR
  prefetch the line qkv + b*qkv_stride + (HQ + 2*HKV)*D + (rank*NR + tid)*D (the rows this rank
  finishes in the DSMEM reduce-scatter), and thread tid == NR prefetches v_scale_p (read with
  __ldg in the same epilogue). (2) splitkv_attn_sk_kernel (_SK_CUDA, every template instantiation;
  master routes full decode B=1 two-level, full decode 1<B<=64 single-level ticket, and sliding
  verify B=1 PSEUDO): as the first statements of the kernel, before load_prefix /
  load_prefix_pseudo, with R = PSEUDO ? B*8 : B (the same R the kernel already uses), for (int i =
  blockIdx.x*THREADS + tid; i < R*16; i += gridDim.x*THREADS) prefetch_l2(qkv + (int64_t)(i >>
  4)*qkv_stride + (HQ + 2*HKV)*D + (i & 15)*64). Every CTA participates before the early returns,
  so the lines are in L2 whichever CTA wins the ticket; at most one prefetch per thread at the
  gated shapes (R*16 <= 512 lines vs grid*THREADS). Leave splitkv_attn_sk_fused_kernel (skF, not
  routed on master) and the stage-1 kernels untouched; _SKF_HEAD slices _SK_CUDA up to 'template
  <int STAGES, bool TICKET', so put the helper above that point or in both sources and confirm the
  skF build line still prints. Every prepare-time self-check, replay check and timing gate stays
  as is, so a broken build falls back exactly as master's would. In the log, confirm before
  trusting the run: the [slide] fp8qk regs/local/spill line (master: 124 regs, 32 B local, 103424
  B smem; an extra spill is a red flag), [slide-gate] C=16 ROUTE and C=4 ROUTE, PATH CHOSEN
  sliding verify B=1 pseudo grid p8, and PATH CHOSEN full decode_1 streamk two-level, all as
  master. Rules: prefetch is a hint with no data dependence, so the kernel stays graph-capturable,
  adds no host sync, allocation or input-dependent Python branch, caches nothing across calls, and
  every key is still attended with FP8 KV and E4M3 q.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "cluster C=16 kernel 10.9 us in an 11.9 us shape: the reduce-scatter's gate load moves from HBM to L2, ~0.3-0.6 us; this row is the most leveraged in the problem (46 / 907 us load, ~0.85% headline per us)"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "PSEUDO sk kernel 9.8 us: the ticket winner's finalize_store loads 8 pseudo-rows' gates cold after the merge; ~0.3-0.6 us at ~0.70% headline per us"
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "two-level ticket sk kernel 18.4 us: same finalize_store tail, ~0.3-0.5 us at ~0.29%/us"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "cluster C=4: 4 gate lines per rank; ~0.3 us of 18.1 us, small"
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: 328-655 MB of KV stream through L2 evicts most prefetched lines and only the last finalize is on the critical path; 32-64 KB of extra reads is noise
  - shapes: "*load=verify_16 * *load=verify_32 * *layer=full load=verify_1 *"
    direction: flat
    why: FA3 / Triton paths, untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T150347Z
review_companions: null
---

# epilogue-gate-l2-prefetch-cuda

## Change

Built on current master eb1b68b (no depends_on). Mechanism: both landed CUDA kernels read the output gate logits (qkv row r, bf16 columns (HQ + 2*HKV)*D .. +HQ*D, i.e. 2 KB = 16 aligned 128-byte lines per row) only in their final epilogue, after the last merge, and the judge flushes L2 before every replay, so that load is a cold HBM round trip (~0.7-1 us) on the critical path of the last CTA. The q / new-k / v loads at kernel start touch other lines of the same row, so the gate lines are never warm. Fix: issue L2 prefetches for exactly those gate lines at kernel entry, so the epilogue load hits L2. No other change; no math, layout, rounding, grid or route changes. Use one helper in each CUDA source: __device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); } (plain prefetch.global.L2, no evict_* / createpolicy cache-hint form). (1) sliding_cluster_kernel (_SLIDE_CUDA; master routes sliding decode B=1 at C=16 and B=17..32 at C=4): immediately after b / rank are computed and before the prologue page-index loads, let NR = HQ / C; threads tid < NR prefetch the line qkv + b*qkv_stride + (HQ + 2*HKV)*D + (rank*NR + tid)*D (the rows this rank finishes in the DSMEM reduce-scatter), and thread tid == NR prefetches v_scale_p (read with __ldg in the same epilogue). (2) splitkv_attn_sk_kernel (_SK_CUDA, every template instantiation; master routes full decode B=1 two-level, full decode 1<B<=64 single-level ticket, and sliding verify B=1 PSEUDO): as the first statements of the kernel, before load_prefix / load_prefix_pseudo, with R = PSEUDO ? B*8 : B (the same R the kernel already uses), for (int i = blockIdx.x*THREADS + tid; i < R*16; i += gridDim.x*THREADS) prefetch_l2(qkv + (int64_t)(i >> 4)*qkv_stride + (HQ + 2*HKV)*D + (i & 15)*64). Every CTA participates before the early returns, so the lines are in L2 whichever CTA wins the ticket; at most one prefetch per thread at the gated shapes (R*16 <= 512 lines vs grid*THREADS). Leave splitkv_attn_sk_fused_kernel (skF, not routed on master) and the stage-1 kernels untouched; _SKF_HEAD slices _SK_CUDA up to 'template <int STAGES, bool TICKET', so put the helper above that point or in both sources and confirm the skF build line still prints. Every prepare-time self-check, replay check and timing gate stays as is, so a broken build falls back exactly as master's would. In the log, confirm before trusting the run: the [slide] fp8qk regs/local/spill line (master: 124 regs, 32 B local, 103424 B smem; an extra spill is a red flag), [slide-gate] C=16 ROUTE and C=4 ROUTE, PATH CHOSEN sliding verify B=1 pseudo grid p8, and PATH CHOSEN full decode_1 streamk two-level, all as master. Rules: prefetch is a hint with no data dependence, so the kernel stays graph-capturable, adds no host sync, allocation or input-dependent Python branch, caches nothing across calls, and every key is still attended with FP8 KV and E4M3 q.

## Expected

- `*layer=sliding load=decode_1 *` faster: cluster C=16 kernel 10.9 us in an 11.9 us shape: the reduce-scatter's gate load moves from HBM to L2, ~0.3-0.6 us; this row is the most leveraged in the problem (46 / 907 us load, ~0.85% headline per us)
- `*layer=sliding load=verify_1 *` faster: PSEUDO sk kernel 9.8 us: the ticket winner's finalize_store loads 8 pseudo-rows' gates cold after the merge; ~0.3-0.6 us at ~0.70% headline per us
- `*layer=full load=decode_1 *` faster: two-level ticket sk kernel 18.4 us: same finalize_store tail, ~0.3-0.5 us at ~0.29%/us
- `*layer=sliding load=decode_32 *` faster: cluster C=4: 4 gate lines per rank; ~0.3 us of 18.1 us, small
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: 328-655 MB of KV stream through L2 evicts most prefetched lines and only the last finalize is on the critical path; 32-64 KB of extra reads is noise
- `*load=verify_16 * *load=verify_32 * *layer=full load=verify_1 *` flat: FA3 / Triton paths, untouched

## Risk

Medium risk only because it edits two landed CUDA sources (one 40 s rebuild each); the added instruction is the plain prefetch.global.L2 hint on in-bounds addresses, which cannot fault and changes no data, and every route keeps its own self-check and timing gate. If the build fails the run measures the fallbacks (Triton sliding decode, FA3 sliding verify_1) and loses ~2-5% on those rows, so the auditor must grep the build and PATH CHOSEN lines listed in the change. Expected headline ~+0.6-1.0% if the epilogue load is as exposed as the cold-L2 model says; a flat result says the compiler already overlapped it or the line was warm, and closes this knob for both kernels. Weakest link: sector-level L2 behaviour; if a gate line was already pulled in with the q row the effect is zero, which the per-row deltas will show (sliding decode_1 and sliding verify_1 are 0.05 us-resolution rows). Independent of preattention-tiled-every-verify-block (Triton pre-attention on verify) and of the queued sliding-decode-sk-s3 line (which edits splitkv_attn_sk_fused_kernel, not these two kernels).

## Adversarial review

Reviewed 2026-10-25T150347Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism holds on current master (7e12891): both CUDA kernels read the gate columns (row bytes 2560..4607) only in the epilogue, after a cold-L2 flush, and nothing earlier touches those lines. The expected gain (~+0.6-1.0% headline) is inside the +/-1.5% CI, so the run will be read from the per-row deltas. Two build-text fixes are needed before it runs: use the kernel's own R = B*TOK, and define the helper only once.

### Justification
Plausible. On master the sliding_cluster_kernel reduce-scatter (qkv + b*stride + (HQ+2HKV)*D + h*D + d0) and splitkv_attn_sk_kernel finalize_store read the bf16 gate only after the DSMEM merge or the ticket merge. TASK.md says every timed replay starts with a cold L2, and the q/k/v loads (cluster prologue, or the separate _pre_attention_kernel on the sk routes) touch only the first 2560 B of each 4608 B row. So the gate lines really are cold. The row is 36x128 B and the gate offset is 20x128 B, so each head's 64 bf16 are exactly one 128 B line and the line math in the change is right. The sk-kernel merge goes through ticket atomics and fences, so the compiler cannot hoist the gate load above it. That leaves ~0.5-0.8 us of HBM latency on the last CTA's critical path, which matters on the latency-bound rows (sliding decode_1 cluster 10.8 us of 11.9; sliding verify_1 PSEUDO 10.0 of 13.2; full decode_1 two-level 19.0 of 22.4). Full decode_16/32 stream 328-655 MB through a ~50 MB L2, so flat is right there. Correction against current master: sliding decode B=17..32 now routes through splitkv_attn_sk_kernel with PSEUDO and TOK=1 (the [sd] window route, profile shows sk kernel 14.0 us at decode_32), not cluster C=4. The decode_32 expectation still applies, but through the sk prefetch, and only if R is B*TOK. Weakest link: the sliding decode_1 cluster epilogue. Inside the post-barrier reduce-scatter block, nvcc may already have scheduled the gp[0]/gp[1] loads ahead of the 16 DSMEM map_shared_rank reads and the Ls/o merge, hiding part of the latency. Then the most leveraged row (~0.85% headline per us) gains well under 0.3 us. Checking SASS for the LDG position relative to the DSMEM loads would settle it before the run. Magnitude: the whole expected headline gain is below the CI floor, so the run is readable only per row.

### Confounders on the current kernel
- restack-sliding-decode-sk-window-on-cluster-master (landed ~15:20Z, after the hypothesis was written against eb1b68b): sliding decode B=17..32 is now chosen at prepare time between sk window (TOK=1) and the cluster C=4 incumbent, with a 1.03x head-to-head gate. The change speeds up both candidates, so it can flip that route. A decode_32 delta could then be a route flip, not the prefetch. The audit must confirm 'PATH CHOSEN: sliding decode B=17..32: [pre-attn + stream-K sk window]' as on master, a line missing from the hypothesis's log checklist.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: its gates (C=16 at B=1, C=4 cluster vs Triton, PSEUDO verify B=1 >= 1.05x vs FA3) are timed at prepare. The prefetch only helps the CUDA side, so it can only strengthen those routes, but a route that flips to CUDA where master used Triton/FA3 would confound the row. The listed PATH CHOSEN / [slide-gate] ROUTE checks cover this.
- full-decode1-streamk-s3-two-level-ticket: in the two-level ticket, the CTA that runs finalize_store is chosen at runtime. The grid-wide prefetch loop covers that because every CTA prefetches before the early returns. No interaction beyond that, and the effect does not depend on how the ticket is tuned.
- full-decode-cuda-gather-splitkv / v2 stream-K at 1<B<=64: same kernel, but the KV stream evicts the prefetched lines. Expect flat, plus 32-64 KB of extra reads. A loss there would point to perturbed code generation (registers or scheduling) in the shared template, not the mechanism.

### Interference and A/B needs
No companion. The submit is scored and the change is one hint instruction in two kernels with no route, math or grid change. The attribution comes from the per-row deltas, which the expected list already pins: sliding decode_1 (cluster only), sliding verify_1 (PSEUDO TOK=8), full decode_1 (two-level), sliding decode_32 (sk TOK=1). The full decode_16/32 rows serve as a built-in null, since the same template gets the prefetch but L2 thrashes. The only real ambiguity is a route flip, and the PATH CHOSEN log lines settle that for free. An ab-revert of a landed hypothesis would remove the very kernels being edited, and a control that splits the two kernels would cost a second scored run to separate two effects that already show up on disjoint rows. Leaving it uninterpreted costs less than a companion.

### Open questions
- Does the SASS of sliding_cluster_kernel already issue the gate LDGs before the DSMEM merge loads (cuobjdump -sass on the built module)? If so, the decode_1 row should be expected flat and the headline gain drops to ~0.3-0.5%.
- Does the extra prologue code change the [slide] fp8qk register, local or spill line (master 124 regs / 32 B local), or the [sk]/[sd]/[ps] regs lines for the PSEUDO variants? A new spill would cancel the gain on exactly the rows meant to improve.
- At full decode_1 (~20 MB of KV in 19 us), do the 16 prefetched gate lines survive in L2 until the two-level finalize? If they are partly evicted, full decode_1 reads flat while the sliding rows (0.5 MB window) still gain.
- Is the v_scale_p prefetch in the cluster kernel worth keeping? The last rank already __ldg's v_scale_p at kernel start, so it is probably warm for every rank by the epilogue. It is harmless either way.

