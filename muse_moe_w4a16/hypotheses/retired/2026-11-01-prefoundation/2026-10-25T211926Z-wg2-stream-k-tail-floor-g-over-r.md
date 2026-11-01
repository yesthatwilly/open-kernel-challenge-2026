---
problem: muse_moe_w4a16
slug: wg2-stream-k-tail-floor-g-over-r
created: 2026-10-25T211926Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Work decomposition of the last wave only, in wg2_kernel at B256 on the restacked base
  (restack-occ2-wg2-b256-two-stream-halves-on-master), for both w13 and w2. Every other cell stays
  the base. MECHANISM: wg2 walks total = nc x n_tiles items over G = 2 x SMs = 264 persistent
  CTAs, so makespan is ceil(total / G) items and the last wave has only r = total mod G busy CTAs
  while G - r idle. CHANGE (reviewer corrections folded in): the last r items are split along K
  into S = min(KS, floor(G / r)) slices (KS = K / BK: 32 for w13, 16 for w2), so G - r idle CTAs
  share the tail for EVERY r, not only r <= G/2. The first q x G items run unchanged. Tail slices
  write fp32 partial tiles to a per-state workspace [r, S, tile] (allocated in prepare per state
  copy, never a process-global; LESSONS 162) and take a per-item ticket (atomicAdd on a per-state
  counter zeroed by the kernel's own prologue for the next replay); the last arriver for an item
  sums its S partials in fixed slice order (deterministic) and runs the unchanged epilogue (w13:
  silu_mul with bf16 rounding of gate/up; w2: scale + bf16). No CTA waits on another: no spin, no
  flag. Producer and consumer must derive the identical (item, k-range) sequence from the same
  function; the canary asserts this and max |stream-k - base| relative L2 within the judge bound,
  and forces r = 1, r = G/2 + 1 and r = G - 1 by synthetic routings. Before any run, compute nc,
  total, q, r, S under _wg_judge_ids and zipf offline (CPU replay of _wg_chunks) and print them in
  prepare for both kernels; if S == 1 for both under the judge routing, the path is not built
  (nothing to gain). Gate: cold-L2 graph A/B vs base on both routings, >= 1.02x on the worse.
  First measurement is an okc check (new ticket/workspace path can hang or corrupt across graph
  replays).
expected:
  - shapes: bs=256 *
    direction: faster
    why: "tail recovery of up to one item time per kernel: master per-item ~15 us at OCC1 (reviewer: w13 ~238 us, w2 ~112-119 us), larger per item at OCC2; expect -5 to -20 us of the base 358 us verify_32 = +0.2-0.9% headline, below the 1.24% margin alone: must stack with the other wg2 follow-ups"
  - shapes: bs=128 *
    direction: flat
    why: B128 stays Marlin
risk: high
depends_on: restack-occ2-wg2-b256-two-stream-halves-on-master
priority: 3
kind: primary
controls_for: null
prior: "wg-stream-k-tail-split: that held master-based file used S=2 only when r <= G/2 (a coin flip per kernel), quoted the wrong profile's times, and targeted G=132 single-CTA wg_kernel that is no longer the board path; this uses S = min(KS, floor(G/r)) for every r on wg2 (G=264), per-state workspace, printed q/r/S, and skips the build when the judge routing gives S=1. Attention's stream-K ticket combine (full-decode-splitkv-s2-streamk-ticket-inproc) gained 9-11% on its kernel; wg-pdl-gather-w13-w2-chain (flat) attacked the same tail from the dependent side"
review: hold
status: hold
review_at: 2026-10-28T145611Z
review_companions: null
hold_reason: "adversarial review: Stale and superseded: the change edits wg2_kernel at OCC2 / G=264 on the restack-occ2-wg2-b256 base, but that base never landed (it is on FINDINGS' Do-not-try l"
---

# wg2-stream-k-tail-floor-g-over-r

## Change

Work decomposition of the last wave only, in wg2_kernel at B256 on the restacked base (restack-occ2-wg2-b256-two-stream-halves-on-master), for both w13 and w2. Every other cell stays the base. MECHANISM: wg2 walks total = nc x n_tiles items over G = 2 x SMs = 264 persistent CTAs, so makespan is ceil(total / G) items and the last wave has only r = total mod G busy CTAs while G - r idle. CHANGE (reviewer corrections folded in): the last r items are split along K into S = min(KS, floor(G / r)) slices (KS = K / BK: 32 for w13, 16 for w2), so G - r idle CTAs share the tail for EVERY r, not only r <= G/2. The first q x G items run unchanged. Tail slices write fp32 partial tiles to a per-state workspace [r, S, tile] (allocated in prepare per state copy, never a process-global; LESSONS 162) and take a per-item ticket (atomicAdd on a per-state counter zeroed by the kernel's own prologue for the next replay); the last arriver for an item sums its S partials in fixed slice order (deterministic) and runs the unchanged epilogue (w13: silu_mul with bf16 rounding of gate/up; w2: scale + bf16). No CTA waits on another: no spin, no flag. Producer and consumer must derive the identical (item, k-range) sequence from the same function; the canary asserts this and max |stream-k - base| relative L2 within the judge bound, and forces r = 1, r = G/2 + 1 and r = G - 1 by synthetic routings. Before any run, compute nc, total, q, r, S under _wg_judge_ids and zipf offline (CPU replay of _wg_chunks) and print them in prepare for both kernels; if S == 1 for both under the judge routing, the path is not built (nothing to gain). Gate: cold-L2 graph A/B vs base on both routings, >= 1.02x on the worse. First measurement is an okc check (new ticket/workspace path can hang or corrupt across graph replays).

## Expected

- `bs=256 *` faster: tail recovery of up to one item time per kernel: master per-item ~15 us at OCC1 (reviewer: w13 ~238 us, w2 ~112-119 us), larger per item at OCC2; expect -5 to -20 us of the base 358 us verify_32 = +0.2-0.9% headline, below the 1.24% margin alone: must stack with the other wg2 follow-ups
- `bs=128 *` flat: B128 stays Marlin

## Risk

High risk: ticket counters and workspace must be per state copy and self-resetting across graph replays, or replays read stale partials (LESSONS 162). PDL on the w13->w2 chain overlaps part of this tail; read the wg_pdl line before attributing.

## Adversarial review

Reviewed 2026-10-28T145611Z by an adversarial subagent (model opus).

**Verdict: hold.** Stale and superseded: the change edits wg2_kernel at OCC2 / G=264 on the restack-occ2-wg2-b256 base, but that base never landed (it is on FINDINGS' Do-not-try list), master's submission.py has no wg2 kernel, and verify_32 now runs a hybrid (hot experts on the OCC1 wg_kernel, cold experts on Engine A ea4, on two concurrent streams). Even if it were re-aimed at that hybrid, the plausible headline gain is about 0.1-0.4%, well under the 1.24% bar, and the change is expensive and high-risk.

### Justification
A. The premise is a persistent wg2 grid of G = 2 x 132 = 264 CTAs whose last wave leaves G - r CTAs idle. That kernel does not exist on master (2ec8090f, board 1.8941). grep finds no wg2 symbol in muse_moe_w4a16/submission.py. The only wgmma GEMM is wg_kernel, with __launch_bounds__(THREADS, 1) and state['wg_grid'] = multi_processor_count (G=132, OCC1). The hypothesis's depends_on, restack-occ2-wg2-b256-two-stream-halves-on-master, measured neutral within-ci and sits on the Do-not-try list. The campaign's own wg2 hot half (H=-17, LOG 2026-10-27T0131Z/0136Z) was killed on both moe (x0.405 at T=128, x0.882 at T=256) and decode, and SINGLE-PASS.md lists 'wg2 re-split' as dead. Master routes bs=256 (verify_32, 304.5 us scored, floor 108.9) through _hm_forward with H=-16. Per the board profile 0b49b1aa, profile 5, wg_kernel<true> handles the hot experts in 63.6 us and wg_kernel<false> in 64.4 us, on a fork stream concurrent with ea4_w13 (141.2 us) and ea4_w2 (113.5 us). That is 400.9 us of kernel time against 307.5 us of GPU-busy time, so the two halves overlap heavily. The weakest link: 'idle tail CTAs' no longer means idle SMs. When the hot half's last wave is partial, the concurrent cold Engine A CTAs fill those SMs, so recovering the hot-half tail mostly shifts work rather than shortening the makespan. The critical path is the ea4 cold chain (~255 us), not the 128 us hot chain. The stated expectation was already only +0.2-0.9% headline on a 358 us verify_32 that no longer exists. On today's 304 us row, with the tail hidden behind the cold stream, a realistic figure is 0-5 us, i.e. under 0.3% geomean from one of six loads. The prior check is also weak. The difference from wg-stream-k-tail-split (held) was 'G=264 wg2 instead of G=132 wg_kernel', and master has gone back to the G=132 wg_kernel, now in a different role (hot-only, concurrent). The campaign also measured a ticket/last-arriver tail directly: W6-A, LOG 0728Z, cost about twice what it saved, and the lesson recorded there is 'no single-CTA tails'. Only the fixed-order fp32 partial sum differs from that design.

### Confounders on the current kernel
- campaign X2 / W6-M fork hybrid (hm, H=-16, landed 72e6bd6d / 86e4a4d9): the wgmma hot half now runs on a fork stream concurrent with the Engine A cold half, so any tail recovery on wg_kernel is absorbed by SM sharing with ea4. A measured gain or loss at bs=256 would reflect co-scheduling with the cold stream, not tail utilization.
- campaign Engine A ea4 (cold half at T=256; L3/L4/X1 decode tweaks through 2ec8090f): ea4_w13 + ea4_w2 (~255 us) is the longer stream and so sets verify_32's makespan; shortening the hot half below it does not move the row.
- wg-pdl-gather-w13-w2-chain (landed neutral): PDL on the w13->w2 boundary already overlaps w2's ramp with w13's tail; the hypothesis itself flags this, and it compounds the attribution problem.
- restack-occ2-wg2-b256-two-stream-halves-on-master (Do-not-try, never landed): the named base is absent, so the change cannot be built as written.

### Interference and A/B needs
No companions. The hypothesis cannot be built as specified because its kernel and base are absent. A companion would only make sense for a re-aimed version on the hm hot half's wg_kernel. Even then, a fork-stream-serialized control (the hot half run without the concurrent cold stream) would be needed to see any tail effect at all. That costs an extra check to resolve a gain capped well under 1.24%, which the user's bar rules out.

### Open questions
- If anyone wants to revive the tail idea, the target is the H=-16 hot half's wg_kernel at G=132. First measure how much of its last wave is actually uncovered by the concurrent ea4 cold stream (prepare-time graph timing with and without the fork). If the uncovered tail is under ~4 us at T=256, drop the idea.
- Is there evidence that the verify_32 makespan is ever bound by the hot half rather than the ea4 cold chain under the judge routing? If not, nothing in the hot half's decomposition can move the row.

