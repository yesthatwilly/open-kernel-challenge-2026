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
