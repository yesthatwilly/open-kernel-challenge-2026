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
review: required
status: queued
---

# wg2-stream-k-tail-floor-g-over-r

## Change

Work decomposition of the last wave only, in wg2_kernel at B256 on the restacked base (restack-occ2-wg2-b256-two-stream-halves-on-master), for both w13 and w2. Every other cell stays the base. MECHANISM: wg2 walks total = nc x n_tiles items over G = 2 x SMs = 264 persistent CTAs, so makespan is ceil(total / G) items and the last wave has only r = total mod G busy CTAs while G - r idle. CHANGE (reviewer corrections folded in): the last r items are split along K into S = min(KS, floor(G / r)) slices (KS = K / BK: 32 for w13, 16 for w2), so G - r idle CTAs share the tail for EVERY r, not only r <= G/2. The first q x G items run unchanged. Tail slices write fp32 partial tiles to a per-state workspace [r, S, tile] (allocated in prepare per state copy, never a process-global; LESSONS 162) and take a per-item ticket (atomicAdd on a per-state counter zeroed by the kernel's own prologue for the next replay); the last arriver for an item sums its S partials in fixed slice order (deterministic) and runs the unchanged epilogue (w13: silu_mul with bf16 rounding of gate/up; w2: scale + bf16). No CTA waits on another: no spin, no flag. Producer and consumer must derive the identical (item, k-range) sequence from the same function; the canary asserts this and max |stream-k - base| relative L2 within the judge bound, and forces r = 1, r = G/2 + 1 and r = G - 1 by synthetic routings. Before any run, compute nc, total, q, r, S under _wg_judge_ids and zipf offline (CPU replay of _wg_chunks) and print them in prepare for both kernels; if S == 1 for both under the judge routing, the path is not built (nothing to gain). Gate: cold-L2 graph A/B vs base on both routings, >= 1.02x on the worse. First measurement is an okc check (new ticket/workspace path can hang or corrupt across graph replays).

## Expected

- `bs=256 *` faster: tail recovery of up to one item time per kernel: master per-item ~15 us at OCC1 (reviewer: w13 ~238 us, w2 ~112-119 us), larger per item at OCC2; expect -5 to -20 us of the base 358 us verify_32 = +0.2-0.9% headline, below the 1.24% margin alone: must stack with the other wg2 follow-ups
- `bs=128 *` flat: B128 stays Marlin

## Risk

High risk: ticket counters and workspace must be per state copy and self-resetting across graph replays, or replays read stale partials (LESSONS 162). PDL on the w13->w2 chain overlaps part of this tail; read the wg_pdl line before attributing.
