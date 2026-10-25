---
problem: muse_bf16_proj
slug: o-swapab-m32-splitk2-two-ctas-per-sm
created: 2026-10-25T003858Z
base: master
area: o GEMM (m=17..32, swap-AB)
change: >
  Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with
  SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (128, 1) = 128 CTAs, each
  streaming 64 x 1024 x 2 B = 128 KB in 4 K-steps, one CTA per SM (4 x 48 KB = 192 KB smem), 4.8
  us kernel against a 3.6 us floor (row 6.3 at decode_32). With one CTA per SM and only 4 steps,
  the pipeline is mostly fill. Change the tuple to (8192, 1024, 2, 256, 2): grid (128, 2) = 256
  CTAs, K_SLICE 512 (2 K-steps, both in flight with 2 stages), 96 KB smem per CTA so 2 CTAs fit
  per SM (264 slots, one wave); each CTA streams 64 KB and the two CTAs on an SM overlap their
  fill. The existing ticket reduce handles SPLIT_K 2 (sum of 2 fp32 [32, 64] partials, fixed
  order). _swapab_buffers returns state swapab_ws_o / swapab_cnt_o for layer o; _swapab_prepare
  allocates swapab_ws_o = (2, SWAPAB_MAX_M, 8192) fp32 (2 MB) and swapab_cnt_o = zeros(128) int32
  and re-zeroes it after warm-up like the others. Nothing else changes (o at m<=16 keeps
  _pooler_skinny_kernel; m=33 o test and m>=128 keep cuBLAS).
expected:
  - shapes: layer=o load=decode_32 m=32
    direction: faster
    why: 4.8 us kernel, 1 CTA/SM with a 4-step pipeline that is mostly fill; 2 CTAs/SM with 64 KB each overlap fill; floor 3.6; target kernel ~4.2, row ~5.8
  - shapes: layer=o load=decode_1 m=1
    direction: flat
    why: m<=16 skinny path untouched
  - shapes: layer=o load=decode_16 m=16
    direction: flat
    why: untouched
  - shapes: layer=o load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=qkv_gate *
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
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T004221Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T004221Z
base_sha: 9a3cafafd2a3738a49dc5fe238ef2bf753adafd4
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/o-swapab-m32-splitk2-two-ctas-per-sm
worktree: .worktrees/muse_bf16_proj/o-swapab-m32-splitk2-two-ctas-per-sm
mode: null
restack_of: null
stale_base: null
---

# o-swapab-m32-splitk2-two-ctas-per-sm

## Change

Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (128, 1) = 128 CTAs, each streaming 64 x 1024 x 2 B = 128 KB in 4 K-steps, one CTA per SM (4 x 48 KB = 192 KB smem), 4.8 us kernel against a 3.6 us floor (row 6.3 at decode_32). With one CTA per SM and only 4 steps, the pipeline is mostly fill. Change the tuple to (8192, 1024, 2, 256, 2): grid (128, 2) = 256 CTAs, K_SLICE 512 (2 K-steps, both in flight with 2 stages), 96 KB smem per CTA so 2 CTAs fit per SM (264 slots, one wave); each CTA streams 64 KB and the two CTAs on an SM overlap their fill. The existing ticket reduce handles SPLIT_K 2 (sum of 2 fp32 [32, 64] partials, fixed order). _swapab_buffers returns state swapab_ws_o / swapab_cnt_o for layer o; _swapab_prepare allocates swapab_ws_o = (2, SWAPAB_MAX_M, 8192) fp32 (2 MB) and swapab_cnt_o = zeros(128) int32 and re-zeroes it after warm-up like the others. Nothing else changes (o at m<=16 keeps _pooler_skinny_kernel; m=33 o test and m>=128 keep cuBLAS).

## Expected

- `layer=o load=decode_32 m=32` faster: 4.8 us kernel, 1 CTA/SM with a 4-step pipeline that is mostly fill; 2 CTAs/SM with 64 KB each overlap fill; floor 3.6; target kernel ~4.2, row ~5.8
- `layer=o load=decode_1 m=1` flat: m<=16 skinny path untouched
- `layer=o load=decode_16 m=16` flat: untouched
- `layer=o load=verify_1 m=8` flat: untouched
- `layer=qkv_gate *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Counter-evidence: the CUDA gemv rewrite ran o with SPLIT_K 2 and lost at m=8 (the m-proportional ticket-reduce tail); here the reducer reads only 2 partials of 8 KB and the tail is one CTA per tile, but if the decode_32 row is not below 6.3 the tail cost is confirmed and o-unpooler-slice-cuda-cluster-dsmem-rewrite (DSMEM reduce) is the route instead. One scored shape, attributable from its row.

## Adversarial review

Reviewed 2026-10-25T004221Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, attributable one-tuple change to the o decode_32 row, worth one run. The mechanism is weaker than claimed: per-SM bytes in flight do not increase (the whole 16.8 MB is already requested in the first wave). The cited baseline (4.8 us kernel, 6.3 row) is the restack job, not master. Judge on kernel time vs master's 4.5-4.6 us.

### Justification
Incumbent _bf16_swapab_kernel grid (128,1), 4 stages x 48 KB = 128 KB/CTA in flight, the whole weight in wave 1, ~3.5-3.7 TB/s vs a documented Triton ceiling ~3.73. Proposed (8192,1024,2,256,2): 256 CTAs, 2/SM, 96 KB smem; at most 128 KB/SM in flight, same as today. Any gain is shorter per-CTA serial latency and filling 4 idle SMs, against a ticket reduce (2 x 1 MB fp32, L2) and the winner's tail. The 4.2 us target is 4.0 TB/s, above the ceiling. Weakest link: Triton num_stages=2 usually prefetches only 1 tile ahead, possibly leaving 64 KB/SM in flight (the recorded BLOCK_N=32 regression). Run-to-run spread of identical code is ~0.4 us row, ~0.3 us kernel. Read as kernel <= ~4.3 us and paired per-shape speedup vs SGLang (master 1.051-1.070), not an absolute row threshold.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (landed): the incumbent being retuned; its own o decode_32 kernel read 4.5 us.
- bf16-shape-gated-triton-winners-combo (landed): o m<=16 on _pooler_skinny_kernel, untouched.
- router-hilo / router-hilo-at-m8 / sglang-gemm-dispatch-parity: other rows.
- bf16-restack-m16-swapab-router-swapab-m32 (pending): disjoint rows and state keys; expect a trivial textual conflict after state['swapab_cnt_qkv'].zero_() in _swapab_prepare and in _swapab_buffers; BEST.json blob a8bb42a vs real 970b415 is a bookkeeping error.

### Interference and A/B needs
No companion. A gain is attributable; a loss is ambiguous between the ticket tail and num_stages=2 keeping one tile in flight; settle in-run: prepare() prints compiled.metadata.shared (<= ~113 KB for 2 CTAs/SM) and the PTX order of cp.async/commit_group vs wait_group for layer o. Follow-up if it loses that way: SPLIT_K 2 with BLOCK_K 128 and 4 stages.

### Open questions
- Does num_stages=2 with a 2-iteration K loop keep both tiles in flight? Check the PTX.
- Is compiled shared <= ~113 KB so 2 CTAs co-reside?
- Harvest baseline: master o decode_32 kernel 4.5-4.6, row 6.64-6.73.
- If it wins, the restack's SWAPAB16_CFG['o'] at m<=16 is the next target.

