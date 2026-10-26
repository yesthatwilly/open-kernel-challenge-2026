---
campaign: 2026-10-26-decode-first-rewrite
status: closed
closed: 2026-10-26T15:30Z
successor: 2026-10-26-expert-bandwidth
---

# Retrospective: decode-first rewrite (2026-10-26, 04:38Z–08:26Z active)

## Results

| board | start | end | scored run |
|---|---|---|---|
| muse_decode_layer | 1.3164 (rank 448) | **1.4422** (rank 355 at close, 373 by 15:30Z) | 0d84f58ca33b424b |
| muse_moe_w4a16 | 1.402 (rank 490) | **1.5762** (rank 413 at close) | f0fc89031659499c |
| overall | rank 480 of 1163 | rank 362 of 1193 at close (380 of 1222 by 15:30Z) | |

Quota used on 10-26: decode 13 checks + 7 scored, moe 14 checks + 4 scored (including 3 churn-era moe checks).

## What worked

- **Engine A (v1/v2/v3)**: exact integer NVFP4 dequant, CTA-shared bulk ring, mma.sync swap-AB, stream-K with
  last-arriver tickets, PDL. +19% on moe T=1, T=8..32 classes x1.06-1.19; v3 (passes of up to 16 routes) won T=32 and
  T=128.
- **Aligned split** at T=1 (w13 19.8 -> 14.3 us) and the **CUDA combine** (kept in every class).
- **Dense swap-AB port** (bf16_proj's Triton GEMMs for decode qkv_gate / o at T <= 32, x1.09-1.16 per GEMM).
- **Hybrid at decode T=256** (Marlin hot experts + v3 cold) and **v3 at T=128**: verify loads 1.24 -> 1.31-1.34.
- Whole-layer gates with "keep unless slower", PATH CHOSEN lines re-printed at the end of prepare, attempt files filled
  by `rewrite/bin/dispatch.py`, archives via `rewrite/bin/archive.py`.
- Builder subagents in their own worktrees (session-scoped widening 05:47Z-09:49Z) with Opus audits ran in parallel
  with first checks: the main throughput multiplier of the day.

## What failed (measured; do not redo blindly)

- L2 prefetch of K2's w2 from K1 at T=1; K2 on one CTA per SM; R -> K1 PDL (neutral).
- Engine B v1 (wgmma RS, 3 warpgroups) and v2 (mma.sync m + wgmma w cells): correct, 1.5-1.95 TB/s; dequant and MMA
  serialized in the same warps ("OVERLAP" diagnostics: ring 4.0, no-dequant 3.5, no-MMA 2.9, together 1.76 TB/s).
- moe hybrid at T=256: 0.70-0.74x of master's wg path on the gate's zipf input.

## Mistakes and their fixes

- A gate closure freed align tensors still referenced by a captured graph (illegal address, check 485b670e): closures
  are now kept alive until replays finish.
- A layer check left T=1/T=8 switched off (check 4594d4d0 measured nothing new): the check keeps the class flag set.
- The 16 KB judge log tail dropped Engine A's summary: decode prepare re-prints every PATH CHOSEN line last.
- Discovered afterwards (expert-bandwidth research): the decode gate inputs (`_cb_data`) use an isotropic stream, so
  T=128 gates saw 23-26 distinct experts against the judge's ~18.4; moe's v3 T=128/256 gate never timed a judge-style
  draw. Fixed first in the successor campaign (X0a).

## Carry-overs to 2026-10-26-expert-bandwidth

- Expert block at T=128/256 (read each expert once) and the T <= 32 consumer's warp starvation (2 consumer warps per
  SMSP, 0.22-0.32 IPC per warp) are the two biggest remaining levers; see that campaign's FINDINGS.md.
- The code continues on branch `campaign/expert-bandwidth`, forked from this campaign's branch at ae91f55d. This
  branch is merged into master when the successor's first milestone lands.
