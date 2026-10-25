---
problem: muse_bf16_proj
slug: qkv-swapab-m32-splitk8-stages3
created: 2026-10-25T003858Z
base: master
area: qkv GEMM (m=17..32, swap-AB)
change: >
  Built on master (8b9ad43, blob a8bb42a). qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with
  SWAPAB_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages): grid (36, 4) = 144 CTAs on
  132 SMs (12 SMs get a second CTA, the rest idle half their smem), 24 KB per stage, 11.8 us
  kernel = 3.2 TB/s against an 8.0 us floor (row 14.2 at decode_32). Change only that tuple to
  (2304, 8192, 8, 128, 3): grid (36, 8) = 288 CTAs, K_SLICE 1024 (8 K-steps of 128), 3 stages = 72
  KB smem per CTA so 3 CTAs fit per SM (396 slots >= 288, one wave), roughly doubling the weight
  bytes in flight per SM. The ticket reduce is unchanged code; it now sums 8 fp32 partials of [32,
  64] per tile in fixed order 0..7. In _swapab_prepare, allocate state swapab_ws_qkv as (8,
  SWAPAB_MAX_M, 2304) fp32 (2.4 MB) instead of (4, ...); swapab_cnt_qkv is unchanged (36
  counters). Nothing else changes: other layers, m<=16 qkv path (_qkv_splitk_sem_kernel), m>=128
  cuBLAS.
expected:
  - shapes: layer=qkv_gate load=decode_32 m=32
    direction: faster
    why: swap-AB kernel 11.8 us at 3.2 TB/s, 144 CTAs (partial second wave on 12 SMs); 288 CTAs at 3/SM doubles bytes in flight; floor 8.0; target kernel ~10.5 us, row ~13
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: flat
    why: m<=16 path untouched
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: flat
    why: m<=16 path untouched
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
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
status: queued
review_at: 2026-10-25T004153Z
review_companions: null
---

# qkv-swapab-m32-splitk8-stages3

## Change

Built on master (8b9ad43, blob a8bb42a). qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages): grid (36, 4) = 144 CTAs on 132 SMs (12 SMs get a second CTA, the rest idle half their smem), 24 KB per stage, 11.8 us kernel = 3.2 TB/s against an 8.0 us floor (row 14.2 at decode_32). Change only that tuple to (2304, 8192, 8, 128, 3): grid (36, 8) = 288 CTAs, K_SLICE 1024 (8 K-steps of 128), 3 stages = 72 KB smem per CTA so 3 CTAs fit per SM (396 slots >= 288, one wave), roughly doubling the weight bytes in flight per SM. The ticket reduce is unchanged code; it now sums 8 fp32 partials of [32, 64] per tile in fixed order 0..7. In _swapab_prepare, allocate state swapab_ws_qkv as (8, SWAPAB_MAX_M, 2304) fp32 (2.4 MB) instead of (4, ...); swapab_cnt_qkv is unchanged (36 counters). Nothing else changes: other layers, m<=16 qkv path (_qkv_splitk_sem_kernel), m>=128 cuBLAS.

## Expected

- `layer=qkv_gate load=decode_32 m=32` faster: swap-AB kernel 11.8 us at 3.2 TB/s, 144 CTAs (partial second wave on 12 SMs); 288 CTAs at 3/SM doubles bytes in flight; floor 8.0; target kernel ~10.5 us, row ~13
- `layer=qkv_gate load=decode_1 m=1` flat: m<=16 path untouched
- `layer=qkv_gate load=decode_16 m=16` flat: m<=16 path untouched
- `layer=qkv_gate load=verify_1 m=8` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

One scored shape (qkv decode_32), plus the m=33 o test is unaffected. Risk: the 8-way reduce tail (8 x 8 KB reads by the last CTA) and more partial-store traffic may eat the gain; qkv weight (37.7 MB) is L2-resident so judge on the cold-L2 row, not the profile. LESSONS: Triton streams top out near 3.5-3.7 TB/s here, so the realistic ceiling is ~10.5 us kernel.

## Adversarial review

Reviewed 2026-10-25T004153Z by an adversarial subagent (model opus).

**Verdict: proceed.** Retunes one tuple (qkv_gate swap-AB at 17<=m<=32: SPLIT_K 4->8, stages 4->3) in the bf16-m32-swapab-wgmma-triton kernel. Only qkv_gate decode_32 changes, so one run is readable on that row; no companions. No overlap with the pending restack, which never touches the m=17..32 qkv tuple or swapab_ws_qkv.

### Justification
Arithmetic checks: 24 KB/stage, 3 stages = 72 KB/CTA, 3 CTAs ~219 KB < 228 KB; 16 acc regs/thread; 288 CTAs in 396 slots, one wave; 8-way workspace 2.36 MB. Row 14.2 us vs SGLang 14.6, floor 8.0, kernel 11.8 us (3.2 TB/s), cold L2. Weakest link: the mechanism. Master already keeps ~72 KB/SM in flight, and Little's law needs only ~36-75 KB/SM, so any gain is more likely load balance (144 CTAs of 256 KB on 132 SMs leave 12 SMs 79% over the mean; 288 x 128 KB leave ~24 SMs 34% over). Costs: partial write+reduce doubles (~1.2 -> 2.4 MB each way, mostly L2). Expected ~1.2 us on a 64.5 us load is ~0.3% headline, inside CI: judge on the qkv_gate decode_32 row and its top_kernels time only.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (landed): retune of its own kernel; the 11.8 us baseline is this kernel on master.
- router-triton-hilo-fused-control-splitk-all-m / sglang-gemm-dispatch-parity / router-hilo-at-m8 / bf16-shape-gated-triton-winners-combo: different layers or m ranges; none confounds.

### Interference and A/B needs
No companion. A control splitting SPLIT_K 8 from stages 3 (e.g. (8,128,4) at 2 CTAs/SM) explains why, not whether; queue only as a follow-up if flat. Restack: no scored-path overlap, a 3-way merge should be clean; if this wins, the same retune is a follow-up for the restack's m<=16 qkv tuple. Sibling o-swapab-m32-splitk2-two-ctas-per-sm edits the same SWAPAB_CFG dict, _swapab_buffers and _swapab_prepare: textual conflict when stacked.

### Open questions
- Print the compiled qkv variant's shared/n_regs in prepare(): if smem exceeds ~75 KB only 2 CTAs fit and 288 CTAs no longer fit one wave.
- BEST.json lists blob a8bb42a for the restack slug but its blob is 970b415: bookkeeping inconsistency; this builds on master a8bb42a.
- Judge on the qkv_gate decode_32 row and kernel time only.

