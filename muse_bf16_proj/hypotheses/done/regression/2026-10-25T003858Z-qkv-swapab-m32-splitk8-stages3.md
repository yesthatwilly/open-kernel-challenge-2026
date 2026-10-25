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
status: done
review_at: 2026-10-25T004153Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T004153Z
base_sha: eaddbb705ebf231e1f6510a66f7051d716c65fdd
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/qkv-swapab-m32-splitk8-stages3
worktree: .worktrees/muse_bf16_proj/qkv-swapab-m32-splitk8-stages3
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T004327Z
impl_commit: e9ec7d9b3f0860d811500ab7e61813883b89a8bf
submission_sha256: 128cc19c4135b2d8eb3e87ede2b12983edaacf8cc12b4ca3f1b5ad0d79f20e7d
norm_sha256: 3640abea7b64eccffd761e873abdb0974b9806e370121a5cc2f1c4e586ab0f1a
submission_blob: 46e00b642357edc983496010f4eef95573178aa2
lint_warnings: 0
attempt_file: 2026-10-25T004328Z-check-qkv-swapab-m32-splitk8-stages3.md
prepared_at: 2026-10-25T004328Z
job: a61778a08ec04ea6
dispatched_at: 2026-10-25T004351Z
outcome: regression
verdict_detail: regression
speedup: 1.1963185001394392
speedup_ci: [1.1783737226373476, 1.2142632776415307]
harvested_at: 2026-10-25T034131Z
best_at_harvest: 1.2215006063849365
promotion: null
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


## Implementation notes

- SWAPAB_CFG qkv_gate (2304, 8192, 4, 128, 4) -> (2304, 8192, 8, 128, 3): grid (36, 8) = 288 CTAs, K_SLICE 1024.
- _swapab_prepare: swapab_ws_qkv is sized from SWAPAB_CFG["qkv_gate"][2] (constant, = 8) -> (8, 32, 2304) fp32; swapab_cnt_qkv unchanged (36).
- Counters: prepare() already zeroes swapab_cnt_pooler / swapab_cnt_qkv after all warm-ups; kept as is.
- Per the review, the prepare()-time report in _swapab_forward now also prints compiled shared bytes, n_regs and ctas_per_sm_fit (min of 227 KB smem incl. 1 KB/CTA reserve, and 64K regs / (regs*128 threads)); ctas_per_sm_fit >= 3 means 3 CTAs/SM fit. Not runnable here (no GPU): the expected value is 3 (72 KB + pipeline, ~219 KB), unverified.
- No other deviation. Reduce kernel code untouched (loops over SPLIT_K).

## Rules check

- [x] CUDA-graph capturable: diff has no .item/.cpu/sync in kernel_fn; the diagnostic print is in prepare() only (report=True), no tensor prints.
- [x] Nothing derived from data cached: ws/cnt sized from constants; synthetic warm-up inputs only.
- [x] Weights: unchanged, BF16 stays BF16; fp32 partials are workspace only.
- [x] Activations BF16: n/a, unchanged.
- [x] Attention: n/a, no attention touched.
- [x] No timing patches, harness access, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `CLEAN: submission.py`; py_compile OK; __pycache__ removed.

## Result

- job `a61778a08ec04ea6` (check), status done, passed True, tests 9/9
- speedup 1.1963x, 95% CI [1.1784, 1.2143], score 4038.0550826865397 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/qkv-swapab-m32-splitk8-stages3.result.json`, `muse_bf16_proj/hypotheses/done/regression/qkv-swapab-m32-splitk8-stages3.patch`; branch `hyp/muse_bf16_proj/qkv-swapab-m32-splitk8-stages3`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.18 | 13.56 | 0.9252 | lost |
| layer=o load=decode_1 m=1 | 62 | 5.89 | 6.27 | 0.9305 | lost |
| layer=pooler load=decode_1 m=1 | 62 | 18.84 | 19.16 | 0.9863 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 6.15 | 6.48 | 0.949 | lost |
| layer=router load=decode_1 m=1 | 62 | 5.38 | 5.78 | 0.9209 | lost |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.15 | 13.73 | 0.9557 | lost |
| layer=o load=decode_16 m=16 | 62 | 6.23 | 6.76 | 0.9554 | lost |
| layer=pooler load=decode_16 m=16 | 62 | 19.47 | 19.92 | 0.9949 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.51 | 18.65 | 1.0022 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.47 | 5.99 | 0.9143 | lost |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.19 | 15.64 | 0.8745 | lost |
| layer=o load=decode_32 m=32 | 62 | 6.34 | 6.66 | 0.999 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.73 | 20.14 | 0.995 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.08 | 18.53 | 1.0024 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.2 | 6.58 | 0.9251 | lost |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.5 | 13.98 | 0.935 | lost |
| layer=o load=verify_1 m=8 | 62 | 6.13 | 6.29 | 0.9524 | lost |
| layer=pooler load=verify_1 m=8 | 62 | 18.89 | 19.39 | 0.9822 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.0 | 5.99 | 0.9935 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.47 | 5.69 | 0.9538 | lost |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.05 | 16.75 | 1.0085 | flat |
| layer=o load=verify_16 m=128 | 62 | 6.98 | 6.98 | 1.0117 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.0 | 24.6 | 1.0375 | gained |
| layer=unpooler load=verify_16 m=128 | 62 | 19.96 | 19.8 | 1.0099 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.28 | 7.27 | 0.9984 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.69 | 19.38 | 1.0156 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.26 | 9.39 | 1.0163 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.19 | 30.64 | 1.019 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.42 | 24.22 | 1.0135 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.6 | 9.56 | 1.0125 | flat |

gained on pooler verify_16; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32, verify_1. Expected hit: 3/8. Noise ±1.5%.

## Follow-ups

- record which shapes lost (Attribution) and do not retry this area in this form.

