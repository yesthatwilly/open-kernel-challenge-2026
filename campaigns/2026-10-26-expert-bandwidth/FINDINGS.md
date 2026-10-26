---
campaign: 2026-10-26-expert-bandwidth
updated: 2026-10-26T15:30Z
---

# Findings: the evidence base for the expert-bandwidth campaign

Tags: **[M]** judge-measured, **[P]** prepare-time measured, **[I]** inference or estimate.

| source in `research/` | what it holds |
|---|---|
| `board_structure.py` + `.out.txt` | how the overall board orders teams; decode speedup -> decode rank |
| `lever_model.py` + `.out.txt` | the headline formula and component scenarios on scored run 0d84f58ca33b424b |
| `expert_engine_forensics.md` | verbatim: why v3 / Engine B are slow, designs S1 / D1 / D2 / D3, the ladder-2 rungs and verdict |
| `secondary_levers.md` | verbatim: attention / dense / glue levers ranked with the judge formula |
| `leaderboard-2026-10-26T15Z/` | `okc leaderboard --json` snapshots, gzipped |

## 1. Scoring

- **[M] The overall board is the decode board for teams with a decode run.**
  - 731 of 1222 overall rows have a decode run and sit in decode-speedup order.
  - Teams without one are slotted in by a composite of moe/bf16/attention (moe-only teams fit an Amdahl moe weight of 0.476).
  - Prefill does not count.
- **[M] Decode headline** = geomean over the 6 loads of (46·sglang_sliding + 16·sglang_full) / (46·ours_sliding + 16·ours_full). This reproduces 1.4422 exactly.
  - Loads: d1 1.608, d16 1.465, d32 1.314, v1 1.665, v16 1.307, v32 1.336.
  - A µs saved on a sliding row is worth about 2.9× one saved on a full row.
- **[M] moe headline** = plain geomean of 6 per-case speedups (equal weights).
- Decode rank today:

  | speedup | 1.46 | 1.51 | 1.56 | 1.62 | 1.69 | 1.78 | 1.86 | 1.97 |
  |---|---|---|---|---|---|---|---|---|
  | rank | 358 | 329 | 283 | 253 | 206 | 136 | 98 | 50 |

## 2. Where the time is (scored decode 0d84f58ca33b424b)

- **[M] Expert block share:**
  - v16 rows 57–75%;
  - v32 rows 49–72% (the Marlin hot half alone is 207 µs);
  - T ≤ 32 rows 26–52%.
- **[M] Rows vs the judge's speed of light:**
  - verify 16/32: 3.0–3.8×, the furthest of any row;
  - others: 2.1–2.7×.
- **[I] Scenario multipliers** (`lever_model.out.txt`):

  | scenario | multiplier |
  |---|---|
  | experts T=128/256 ×0.5 | ×1.145 |
  | experts T≤32 ×0.7 | ×1.094 |
  | both | ×1.253 |
  | full attention ×0.6 | ×1.041 |
  | dense T≥128 ×0.6 | ×1.023 |

- **[I] Research ranges** (judge formula):

  | design | decode headline |
  |---|---|
  | D1 | ~1.56 |
  | D2 | ~1.62 |
  | + S1 | 1.64–1.70 |

  - T ≤ 32 own-kernel chain: ×1.047–1.106.
  - Attention: ×1.014–1.032.
  - Dense W5: ×1.009–1.012.

## 3. Root causes of the expert block's gap

1. **[M] v3 passes cover ≤16 routes**, so hot experts are streamed and dequantized ceil(n/16) times.
   - [I] Pass/distinct ratio: moe T=128 1.17, T=256 1.54; decode T=128 ~1.3, T=256 ~1.7.
   - [M] Ladder: two-tile passes 2.84 vs 3.40 TB/s.
2. **[I] Activations are re-fetched from L2 per (64-row group, 512-k chunk):** n_local × 1.5 MB per call, i.e. 0.4–0.9× the weight bytes.
3. **[M] Consumer latency-bound.**
   - 0.22–0.32 warp-instructions per clock per warp in every ladder rung.
   - smem limits the CTA to 2/SM, so 2 consumer warps per SMSP.
   - At 1 CTA/SM, R4 collapses to 1.37 TB/s.
4. **[M] Engine B serialized dequant and MMA in the same warps** (OVERLAP 1.55×). Never tried: warp specialization, hot/cold mixing per SM, 8-warp Engine A CTAs.
5. **[M] Decode gate inputs are isotropic** (`_cb_data`).
   - T=128 gates see 23–26 distinct experts against the judge's ~18.4 (backed out of the SoL formula).
   - moe's v3 T=128/256 gate never timed a judge-style draw.
6. **[M] Profile kernel rows include PDL waits** (combine 22–41 µs in profile vs 3.1–4.3 µs real). Use graph-timed kernel-only or whole-layer gates.

## 4. Constants

- **[M] Per expert:** w13 18,874,368 B (16 MiB codes + 2 MiB scales; 1024 stages of 18,432 B); w2 9,437,184 B (512 stages); 28.31 MB total.
- **[M] Judge clock:** clocks.max.sm 1980 MHz. At 4 TB/s that is 15.3 B/clk/SM, i.e. 27.2 NVFP4 el/clk/SM including scales.
- **[I] Judge-implied distinct local experts:** decode d1 2.5, v1 7.5, d16 7.5, d32 9.0, v16 18.4, v32 21.3; moe 2.6 / 8.4 / 11.9 / 16.0 / 22.8 / 24.4.

## End game: the board row must hold the final code (2026-10-26)

- Final standings come from post-close reruns of each team's board submission (technical rules: top submissions are
  rerun several times, on held-out data too, and ranked by the medians).
- The board keeps the best-scoring run. The same decode bytes scored 1.4569 (check) and 1.4401 (scored) on 10-26, so a
  noise-high older run can hold the board row with worse code.
- Before the close: submit the final bytes. If an older run still leads, consider `okc withdraw` of the older run so
  the final code is the one rerun.

## Engine results, 10-26 (expert-bandwidth day 1)

- **v4 = v3 on 8 consumer warps** (2 group records per stage, D=2, 2 CTAs/SM).
  - Kernel-only w13 (moe, zipf) at T=8/16/32: 3.21/3.12/2.92 TB/s vs v3's 3.07/2.88/2.66, i.e. +5-10%. The research
    expected +15-25% (3.4-3.7).
  - It wins every T>=9 class and T=128 in both problems (whole layer, decode: T=16 x1.12-1.19, T=32 x1.07-1.10, T=128
    x1.09 vs master; v3 is 3-5% behind at T=16/32 in every run).
  - Decode scored 1.4956 (from 1.4466).
- **v5 = D1 (32-route passes on 256-k records) lost every gate**: kernel-only 2.39-2.90 TB/s, big gates x0.86-0.90.
  - The 4-tile mma.sync path is tensor-issue bound: about 0.25 HMMA/clk/SM against a 0.31 floor.
  - Half-size stages double the per-stage overhead with less in flight.
  - **Wide passes on mma.sync are a dead end. Hot experts need wgmma.**
- **Hot experts on moe's wgmma path + cold on v4 (hybrid H=-16)**: the wgmma path forms per-expert chunks of up to 128
  routes, so a hot expert is read about once.
  - moe T=256: x1.12-1.20 over master on every reference-style draw; row 391 -> 340 us; scored 1.6851.
  - decode: T=128 x1.055 and T=256 x1.179 over Engine A v4 alone; v32 rows -8..-11%; check 1.5097.
  - The path is shared via rewrite/lib/wg_core.py. Decode prepare grows to 108 s (the wgmma build).
  - At moe T=128 it loses (x0.97): moe's T=128 routing has few hot experts.
- **Selector policy**:
  - Earliest engine within 1% (`prefer_within`) put T=9..16 on v3 where v4 leads by 3-5%, costing d16 -4..-6%.
    Back to the best worst ratio.
  - Near-tie classes (T=2..8: v1/v2/v3/v4 within ~3%) still flip between runs. Row noise on v1 sliding is about +-5%
    even with the same engine.
- **The decode copy-check fault (ff2bd918)** (illegal address with v4 at T=2 + PDL K1, unexplained): K1 of v4/v5 now
  runs plain (the PDL gate measured no gain). It has not recurred in 3 decode runs since.
- **Board context**: decode leader 2.38, top 10 >= 2.18, rank 100 1.86, rank 200 1.70, rank 300 1.54. The gap is broad:
  our T<=32 rows run 2.4-2.6x their speed-of-light bounds, the verify rows 2.9-3.3x.
- **Bandwidth budget per T<=32 row (d32 sliding 206 us, SoL 84.5)**:
  - expert block, at about 3 TB/s;
  - P (pooler + router) at 2.75-3.5 TB/s;
  - unpooler 3.7 TB/s (graph-timed 17 us);
  - qkv_gate 3.0 TB/s;
  - sliding attention about 15 us (SoL 3.5).
  The expert block and P are the largest gaps. Ladder-2 (moe, 10-26 evening) measures v4's geometry rung by rung to
  find v4's limiter.
