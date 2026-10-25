---
problem: muse_prefill_layer
slug: marlin-w2-atomic-add-turn
created: 2026-10-25T042410Z
base: master
area: "w2 Marlin call on the Marlin path (T <= 4096): use_atomic_add, picked per shape class in prepare"
change: >
  Built on master (a81d19c). Port of muse_moe_w4a16 w2-marlin-atomic-add-small-m (checked 1.3445
  there: bs=1 Marlin 48.8 -> 44.9 us and its slow mode 44/100 -> 10/100 replays; see
  muse_moe_w4a16/submission.py _w2_atomic_self_check and its patch in that problem's done/). Only
  the second moe_wna16_marlin_gemm call in _experts_marlin changes: use_atomic_add=True
  (use_fp32_reduce stays True) when state['w2_mode'][cls] says so, cls in {'turn1' (T <= 1024),
  'turn4' (1024 < T <= 4096)}. prepare() picks each class's mode among 'atomic' (no memset),
  'atomic_zero' (cache3.zero_() first; cache3 aliases cache13, which holds w13's output when w2
  runs, so a non-zeroing atomic path that needs a zeroed C must fail its check) and 'master', on
  synthetic soft routings at T=753 and T=3012: correctness vs master's own w2 output (rel L2 <=
  1e-3, no non-finite), then CUDA-graph cold-L2 interleaved timing of the whole _experts_marlin; a
  non-master mode is chosen only at <= 0.97x. Print PATH CHOSEN per class with both times. w13,
  align, silu and the weighted sum are untouched.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "Marlin w2 at T=753 splits K=4096 across CTAs; atomic add removes the serialized global reduce: Marlin pair 650-770 us -> -2 to -5% if chosen"
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: moe saw no change at larger M (bs 8-32); expected master pick
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: Triton path
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-25T043311Z
review_companions: null
hold_reason: "adversarial review: The ported mechanism does not apply at prefill M, and the correctness bound as written rejects the atomic modes anyway, so the expected outcome is PATH CHOSEN ="
outcome: expired
verdict_detail: "retired: engine runs on no scored shape (rega replaced Triton/Marlin/ssdq; prefill DIAGNOSIS 2026-10-25)"
expired_at: 2026-10-25T211931Z
---

# marlin-w2-atomic-add-turn

## Change

Built on master (a81d19c). Port of muse_moe_w4a16 w2-marlin-atomic-add-small-m (checked 1.3445 there: bs=1 Marlin 48.8 -> 44.9 us and its slow mode 44/100 -> 10/100 replays; see muse_moe_w4a16/submission.py _w2_atomic_self_check and its patch in that problem's done/). Only the second moe_wna16_marlin_gemm call in _experts_marlin changes: use_atomic_add=True (use_fp32_reduce stays True) when state['w2_mode'][cls] says so, cls in {'turn1' (T <= 1024), 'turn4' (1024 < T <= 4096)}. prepare() picks each class's mode among 'atomic' (no memset), 'atomic_zero' (cache3.zero_() first; cache3 aliases cache13, which holds w13's output when w2 runs, so a non-zeroing atomic path that needs a zeroed C must fail its check) and 'master', on synthetic soft routings at T=753 and T=3012: correctness vs master's own w2 output (rel L2 <= 1e-3, no non-finite), then CUDA-graph cold-L2 interleaved timing of the whole _experts_marlin; a non-master mode is chosen only at <= 0.97x. Print PATH CHOSEN per class with both times. w13, align, silu and the weighted sum are untouched.

## Expected

- `*load=turn_1*` faster: Marlin w2 at T=753 splits K=4096 across CTAs; atomic add removes the serialized global reduce: Marlin pair 650-770 us -> -2 to -5% if chosen
- `*load=turn_4*` unchanged: moe saw no change at larger M (bs 8-32); expected master pick
- `*load=first_turn*` unchanged: Triton path

## Risk

Kill: master picked for both classes, or Marlin rows within +-3% (turn_1 drift is up to 10%, so judge only by PATH CHOSEN plus runs_ns mode counts, LESSONS 03:41Z). Cheap single-flag port.

## Adversarial review

Reviewed 2026-10-25T043311Z by an adversarial subagent (model opus).

**Verdict: hold.** The ported mechanism does not apply at prefill M, and the correctness bound as written rejects the atomic modes anyway, so the expected outcome is PATH CHOSEN = master on both classes, i.e. a wasted check. Ride it on another Marlin-path check as a prepare-only probe instead.

### Justification
moe gained only at bs=1, where Marlin's grid has far fewer (m-block x n-tile) tiles than SMs and splits K across CTAs, so the fp32 global reduce with locks serialises; at bs 8-32 there it measured flat, and w13 atomic add was neutral 0.987. Here, at T=753 with _align_parallel at bsm 64, w2 has ~32 local m-blocks x 16-32 n-tiles = 512-1024 tiles on 132 SMs, so at most a stripe-boundary tile per SM is K-split and the reduce is a minor cost; at T=3012 even less. Weakest link 1: realistic gain is a few % of a ~230-250 us w2 call (~5-15 us/shape, <0.5% of the turn_1 load) - below the 0.97x pick threshold on the whole _experts_marlin, and turn_1 rows drift 7-10% so it is unreadable in the judge. Weakest link 2: the self-check bound (rel L2 <= 1e-3 vs master's w2 output) is too tight: moe's chosen atomic mode measured 4.79e-3 vs master (bf16 atomic accumulation), so both atomic candidates fail and master is picked. Weakest link 3: 'atomic_zero' adds a 49 MB (turn_1) / 197 MB (turn_4) memset of cache3, 15-65 us, larger than any plausible gain.

### Confounders on the current kernel
- stack-tonight-five-prefill: _align_parallel (local routes only) reduced the m-block count Marlin sees; that is what decides how much K-splitting happens, and it makes K-split rarer, not more common.
- experts-hybrid-marlin-turn-triton-first-turn: Marlin path scope (T <= 4096).

### Interference and A/B needs
If it runs at all, no companion: the prepare A/B prints PATH CHOSEN per class with both times, which is the only readable evidence at this effect size.

### Open questions
- Does sglang's moe_wna16_marlin_gemm zero C internally when use_atomic_add=True (moe's no-memset 'atomic' mode passed its check, which suggests it does)? If so, that internal memset is a fixed cost at prefill M.
- How many of w2's tiles are actually K-split at T=753/3012 (Marlin's par/stripe count)? Printing it would settle the mechanism without a check.

