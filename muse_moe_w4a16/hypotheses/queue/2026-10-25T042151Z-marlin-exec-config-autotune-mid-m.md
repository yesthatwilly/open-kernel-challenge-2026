---
problem: muse_moe_w4a16
slug: marlin-exec-config-autotune-mid-m
created: 2026-10-25T042151Z
base: master
area: Marlin launch config (w13 + w2)
change: >
  On master (9080a71), at 8 <= num_tokens <= 32 only, choose Marlin's tile/exec config per T
  instead of its built-in heuristic. In prepare(): inspect.signature(moe_wna16_marlin_gemm); if it
  accepts thread_k, thread_n and/or blocks_per_sm (the vLLM-lineage kwargs), time candidate
  configs for w13 and w2 separately at T = 8, 16, 32 on synthetic skewed routings (cold L2, CUDA
  graph replays, interleaved, >= 30 reps): (thread_k, thread_n) in {(-1,-1) default, (128,128),
  (64,256), (128,64)} x blocks_per_sm in {-1, 1, 2}; reject any config that raises or misses
  rel-L2 <= 5e-3 vs the default; store the winner per (kernel, T) as Python ints in state only if
  it beats the default p50 by >= 3% and assert the stored choice is the printed winner. kernel_fn
  passes the stored kwargs by T (a shape, not a value). If the binding lacks the kwargs, print
  PATH CHOSEN = default (the run is then a master control - reviewer may prefer to drop it).
  Everything else master byte for byte. Mechanism: at 7-13 touched experts Marlin's heuristic
  picks a single small-M config; wave quantization over 132 SMs (e.g. w13 ~7 blocks x 32 N-tiles)
  is a plausible part of its ~1/3-of-peak bandwidth at mid M.
expected:
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair ~95-114 us; a better config -5 to -15%
  - shapes: bs=16 *
    direction: faster
    why: ~170 us pair; -5 to -10%
  - shapes: bs=32 *
    direction: faster
    why: ~156-207 us pair; -5 to -10%
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T042602Z
review_companions: null
---

# marlin-exec-config-autotune-mid-m

## Change

On master (9080a71), at 8 <= num_tokens <= 32 only, choose Marlin's tile/exec config per T instead of its built-in heuristic. In prepare(): inspect.signature(moe_wna16_marlin_gemm); if it accepts thread_k, thread_n and/or blocks_per_sm (the vLLM-lineage kwargs), time candidate configs for w13 and w2 separately at T = 8, 16, 32 on synthetic skewed routings (cold L2, CUDA graph replays, interleaved, >= 30 reps): (thread_k, thread_n) in {(-1,-1) default, (128,128), (64,256), (128,64)} x blocks_per_sm in {-1, 1, 2}; reject any config that raises or misses rel-L2 <= 5e-3 vs the default; store the winner per (kernel, T) as Python ints in state only if it beats the default p50 by >= 3% and assert the stored choice is the printed winner. kernel_fn passes the stored kwargs by T (a shape, not a value). If the binding lacks the kwargs, print PATH CHOSEN = default (the run is then a master control - reviewer may prefer to drop it). Everything else master byte for byte. Mechanism: at 7-13 touched experts Marlin's heuristic picks a single small-M config; wave quantization over 132 SMs (e.g. w13 ~7 blocks x 32 N-tiles) is a plausible part of its ~1/3-of-peak bandwidth at mid M.

## Expected

- `bs=8 *` faster: Marlin pair ~95-114 us; a better config -5 to -15%
- `bs=16 *` faster: ~170 us pair; -5 to -10%
- `bs=32 *` faster: ~156-207 us pair; -5 to -10%
- `bs=1 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

Kill: binding lacks the kwargs (log says default) or no (kernel, T) wins >= 3% in prepare, or judged rows flat. Prepare timings undercount the judge by 20-25% (LESSONS); use them only to rank.

## Adversarial review

Reviewed 2026-10-25T042602Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed only once the binding is known to take the kwargs. Get that answer for free by putting an untimed print-only signature probe and config sweep table into prepare() of the next moe run already being dispatched. Spending a check to learn 'kwargs absent' or 'nothing beat 3%' just re-measures master, which was measured an hour ago (a2744bc496b84790, 1.3445x). Before building, raise the adoption margin from 3% to at least 8% (with two routings) and measure correctness against fp32, because the LESSONS cuBLASLt dead knob mostly transfers.

### Justification
Where the time goes is real. On master (a2744bc496b84790) bs=8/16/32 run at 106.0/147.7/198.7 us against floors of 37.3/52.5/70.8 us, roughly 35% of the 4.8 TB/s floor. Marlin is 92-96% of each shape. bs=32 is the weakest load at 1.034x vs SGLang. The mechanism as written is weak on three counts. (1) The arithmetic is off. I = H = 4096, so w13 has N = 8192, which is 64 n-tiles at Marlin's default thread_n = 128, not 32. More importantly, Marlin's striped partition spreads the (m-block x n-tile x k-tile) space evenly over the persistent CTAs precisely to avoid wave quantization. A different tile changes slice count, reduction depth and bytes in flight per CTA, not wave fill. (2) The candidate grid partly duplicates the default. For thread_m_blocks = 1 (block_size_m 8, and 16 at T=32), Marlin's small-batch list is (128,128,256 thr), (64,128), (128,64), and the heuristic takes the first valid one, so (128,128) is very likely the default under another name. (64,256) is a large-batch config that may be invalid or not instantiated at small M, so it would raise. In practice the sweep is (64,128) and (128,64) times blocks_per_sm in {1, 2}. blocks_per_sm = 1 or 2 means fewer CTAs, which means fewer bytes in flight. That usually hurts a bandwidth-bound small-M GEMV, the opposite of the hope. (3) Local priors are against it. Every mid-M Marlin knob tried here was flat: marlin-workspace-blocks-per-sm-8, w13 atomic add, w13 bf16 reduce, and the two-stream halves. Only block_size_m (at M >= 128) and w2 atomic add (bs=1, by removing a slow mode) moved anything. Expected direction and size: a 5% win at one shape is about +0.8% geomean, inside the ±1.5% headline CI but visible in that shape's paired ratio (err ~2 us, ~1.5%). The expected rows cite 'Marlin pair ~170 us' at bs=16 and '156-207' at bs=32. Those come from single-profile rows that are noisy: this very job shows Marlin 121.5 us at bs=32 but 176.6 us at bs=16, with profile totals 131.6 vs 184.7 us against judged 198.7 vs 147.7. Read the paired ratios, not the profile. On the LESSONS cuBLASLt transfer question, most of it transfers. bf16-cublaslt-algo-pick-in-prepare picked a config that was 6.8% faster in prepare and became 14% slower in the judge, about a 20-point swing. Prepare timings also undercount the judge by 20-25%. A 3% margin sits deep inside that swing, so the selector would mostly be choosing on noise. Two things make Marlin somewhat less hopeless than cuBLASLt. Cold-L2, graph-replay, interleaved timing is closer to the judge than the bf16 pick was. And these candidates differ structurally (CTA count, tile), so a real effect would be large, not a near-tie. That argues for a much higher bar, not for dropping the idea. Requirements: margin >= 8%; the win must reproduce on two synthetic skewed routings (different seeds, sized to the judge's 7-13 touched experts at the right per-expert counts) and in two interleaved passes; the pair (w13 + act + w2) is timed as one graph in the exact master mode; the sorted per-config table is printed and the winner asserted (LESSONS 03:41Z). Weakest link: there is no evidence that SGLang's moe_wna16_marlin_gemm (sglang.kernels.ops.moe, the b_bias-era signature) exposes thread_k/thread_n/blocks_per_sm at all. No local SGLang is installed and no archived log prints its version or signature. The vLLM-lineage kernel hard-codes -1 internally in several versions.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71): w2 at T<=32 runs use_atomic_add=True with use_fp32_reduce=True (log: PATH CHOSEN = atomic, worst rel-L2 0.00479 vs master). thread_k and blocks_per_sm change the k-slice count, so they change both the atomic chain cost and its bf16 rounding. The w2 sweep must run in master's w2_mode, not the default flags, or the ranking is of a different kernel. The proposed 5e-3 rel-L2 bound against the default config is barely above atomic's own 4.79e-3 jitter, so it will reject valid configs at random. Bound against an fp32 reference at master's own error plus a margin instead.
- marlin-block-m-two-smaller-on-postnorm / triton-map-align-on-block-m-restack: these fix block_size_m (8 at T<=16, 16 at T=32), which sets thread_m_blocks and therefore which (thread_k, thread_n) are valid. The sweep must use master's per-T block_size_m. The queued marlin-two-class-block-m-decode32 would change bs=32's block size and make a T=32 pick stale, so do not stack the two without re-sweeping.
- sglang-marlin-path-parity: the lock workspace comes from marlin_make_workspace(max_blocks_per_sm=4). Candidates of 1 and 2 stay inside it, so this is not a confounder for the proposed grid. Never sweep above 4 without a matching workspace (marlin-workspace-blocks-per-sm-8 already showed the workspace itself is neutral).
- align and post-norm landings (align-parallel-per-expert-verify, post-expert-norm-triton-small-m): they share no launches or traffic with the Marlin config, so they do not confound.

### Interference and A/B needs
No companion. The change is gated to bs=8/16/32, so bs=1/128/256 run byte-identical master launches as in-run controls, and the printed PATH CHOSEN table names exactly which (kernel, T) changed. A gain or loss on a changed shape is attributable without an ab-revert. Reverting the w2 atomic add would measure a kernel nobody ships, and an ab-tune of block_size_m belongs to marlin-two-class-block-m-decode32. The real waste is the no-op outcome, not ambiguity. If the binding lacks the kwargs, or no config clears the margin, the check re-measures master, which a2744bc496b84790 measured an hour ago. The 'master control' framing does not justify that spend: drift in this harness is already recorded (untouched rows move 2-6% between batches), and a duplicate master run adds nothing the board does not have. Mitigation, in order: (a) an untimed, print-only probe goes into the next moe run being dispatched anyway (e.g. swapab-w13-silu-epilogue-verify, queued on master). The probe does a try/except kwargs call of moe_wna16_marlin_gemm with thread_k/thread_n/blocks_per_sm on dummy inputs, prints the signature or TypeError, and prints the sorted cold-L2 sweep table. It applies nothing. (b) Mint this hypothesis for dispatch only if that table shows a config >= 8% faster at some (kernel, T) on both routings. Otherwise retire it as a dead knob with the table as evidence. If the churn cannot piggyback, run it as an okc check (never a submit) and record a kwargs-absent outcome as 'knob unavailable', not 'neutral'.

### Open questions
- Does the judge image's sglang.kernels.ops.moe.moe_wna16_marlin_gemm accept thread_k/thread_n/blocks_per_sm? A zero-cost piggyback print answers it before any check is spent.
- Which exec config does Marlin's heuristic pick at moe_block_size 8 and 16 for N=8192/K=4096 (w13) and N=4096/K=4096 (w2)? If it is (128,128) everywhere, half the proposed grid is the default under another name.
- Is Marlin at mid-M limited by tile config at all, or by block_size_m padding and per-expert weight re-reads (bs=16 with block 8 splits popular experts into several m-blocks that each stream that expert's weights)? If the latter, marlin-two-class-block-m-decode32 is the right lever and this knob will be flat.
- Should the adoption margin be calibrated from data? Running one config against itself twice in prepare gives the noise floor; the winner then has to beat the default by max(8%, 3x that floor).

