---
campaign: 2026-10-26-expert-bandwidth
updated: 2026-10-26T23:56Z (date -u): the evening's findings added at the pause; the judgment and plan are in DISTILL.md
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

## Engine D and the dense GEMMs (10-26 evening)

- **The T <= 32 dense GEMMs already run near the ring ceiling** (graph-timed, cold L2, f59cb301):
  - P (pooler + router, master's dl kernel): 19.6-23.8 us, 3.85 TB/s at T=1;
  - qkv_gate (swap-AB): 11.2-12.7 us;
  - o: 4.85-5.3 us;
  - unpooler at T >= 9 (cuBLAS): 16.8-17.2 us.
  Their speed-of-light headroom adds up to about 6 us per row.
- The P rows of 22-27 us in the decode profiles include PDL waits. Plan only from graph-timed kernels in isolation.
- **Engine D** (a dense bf16 skinny GEMM on v4's CTA, ring of 32 KB stages, stream-K + last-arriver fix-up) lost every
  selector: x0.923 (P, T=1) down to x0.403 (o, T=32).
  - Its fixed cost per launch (ramp, fix-up tails, T x-copies per stage) outweighs the streaming gain.
  - Killed; the code stays behind CAMPAIGN_ENGINE_D = False.


## Ladder-2: v4 is bound by its consumer's instruction stream (moe 5eafb39e, 10-26 17:50Z)

- **[M] Rung ceilings in v4's geometry** (8 consumer warps, two 18 KB records per stage, D=2, 2 CTAs/SM; graph-timed,
  read flush):

  | rung | TB/s | drop |
  |---|---|---|
  | C-R1 ring alone | 4.37 | (4-warp ring 4.46) |
  | C-R2 + exact dequant | 3.98 | -8.9% |
  | C-R3 + MMA (const B, 1 tile) | 3.75 | -5.8% |
  | C-R4 + x from smem, 8 tokens | 3.45 | -8.0% |
  | C-R4 + x from smem, 16 tokens (2 tiles) | 2.94 | -14.8% |
  | C-R1 / C-R4 at D=3, 1 CTA/SM | 4.34 / 3.44 | same as D=2 |

- **[M] Warp count barely matters.** 8 vs 4 consumer warps at the same ring work: x1.024 (8 tokens), x1.043 (16
  tokens). The kernel is neither ring- nor latency-bound: each addition costs throughput in proportion to its
  instructions per weight element. v4's real kernels (3.21 / 3.12 / 2.92 TB/s at T=8 / 16 / 32) sit ~7% under C-R4
  (maps, scatter, fix-ups).
- **[I] The two remaining consumer levers:**
  - B-fragment reuse: each warp takes a k-slice of all 128 rows, so one ldmatrix B fragment feeds 8 m-tiles, at the
    cost of one cross-warp reduction per unit. It attacks the C-R4 drop.
  - A cheaper dequant, which attacks C-R2.
- **[P] dq3, the cheaper dequant.**
  - It extracts each NVFP4 code pair with a funnel-shift rotation plus AND (two of each for pair 3): ~9 ops per 32-bit
    word instead of ~11.
  - An exhaustive search showed that four pure rotations cannot tile the word, and that 9 is the minimum. It was
    verified on the host over all 2^32 words (`scratchpad/dqcheck`).
  - The audit noted that dq3's ops are all ALU-pipe, where production dq2's left shifts run as IMAD.SHL on the FMA
    pipe. dq4 does the rotations as IMAD.WIDE.U32 by run-time powers of two, to balance the pipes.
  - All three are timed in the ladder at the reset (VERDICT-dq: switch production only at >= 5% at C-R4).
- **[I] v4c.** A two-tile pass's unit costs about 7/6 of a one-tile one, so equal unit counts leave the CTAs that
  draw two-tile passes as the kernel's tail. v4c weights the stream-K ranges by pass cost (`Map::s_cost`). It is
  raced against v4 in every gate from eacf0722; unmeasured.
- **[I] H=-8.** With a hot threshold of 8 routes, Engine A keeps only one-tile cold passes (the C-R4 8-token ceiling,
  3.45, instead of 2.94). Unmeasured (eacf0722).

## Judge-weighted decode time by area (check f59cb301, 1.5347; openclaw relay)

Shares of (46 x sliding + 16 x full) time per load. The headline moves by about share x cut.

| load | moe block | bf16 dense | attention | small ops |
|---|---|---|---|---|
| decode 1 | 35% | 44% | 14% | 7% |
| decode 16 | 53% | 27% | 17% | 4% |
| decode 32 | 53% | 22% | 23% | 3% |
| verify 1 | 59% | 29% | 8% | 4% |
| verify 16 | 62% | 18% | 12% | 1% |
| verify 32 | 60% | 18% | 14% | 2% |
| **headline** | **53%** | **26%** | **15%** | **3%** |

- About 2% sits in the profiler's "... N more" rows.
- **[I] Ceilings, from `1 / (1 - share x cut)`:**

  | scenario | multiplier |
  |---|---|
  | moe block cut to x0.6-0.7 | x1.19-1.27 |
  | dense cut 10-15% | x1.03-1.04 |
  | attention | x1.014-1.032 |
  | small ops halved | x1.017 |
  | everything outside moe | ~x1.09 |
  | everything stacked | ~x1.30 |

## bulk12: lossless 12-bit bf16 weights (bf16 board, 10-26 20:20-22:05Z)

- **Codec.** A bf16 weight is H << 8 | L:
  - H = sign | exponent bits 7..1; L = exponent bit 0 | 7 mantissa bits.
  - L stays raw. H becomes a 4-bit code c = sign << 3 | (e71 - base) over the 8 most-used e71 values; base is 54 on
    every judge weight (N(0, 1/k), a 16-binade window).
  - Escapes (|w| < 2^-19 and rare large values; 1.0-1.4e-4 of the elements, <= 58 per CTA, cap 224) decode to the
    window floor (code 0, L = 0). The epilogue corrects them exactly from a per-CTA escape block: + x·w_true - x·w_floor.
  - Decode costs 2 integer ops per element: `H4 = ((c4 >> 3) & 0x01010101) * 0x78 + (c4 + base4)`, then
    `__byte_perm`.
  - Layout: 384 B tiles (256 B L + 128 B codes) in mma.sync m16n8k16 A-fragment order.
  - Lossless for any weights; only the speed depends on the distribution.
- **[M] m=1 (row-owner kernel, x resident).** pooler 17.7 -> **14.8 us**: 50.3 MB at ~3.4-3.6 TB/s, i.e. ~4.8 TB/s
  raw-equivalent, the first bf16 row below its raw-byte floor rate. qkv 10.5 -> 10.7 (flat: 144 CTAs at 2/SM).
  o 6.3 -> 5.9. Board 1.2566 -> 1.2664 (3834ceec).
- **[M] m >= 8 lost every time:**
  - Per-row TMA bulk copies cost ~30-50 ns each, serialized: +7 / +21 / +51 us for 256 / 512 / 1024 copies per CTA
    (22ddaf4a).
  - With LSU cp.async for x, ~1 us per token row remains (0acdca48). It is L2 hot-spotting: every CTA reads the same
    x lines in lockstep. Rotating each CTA's chunk order cut qkv m=8 17.1 -> 14.5 net us (2980f57a).
  - b12r (x resident, m=2..8) judged pooler m=8 x1.016 only (6617f94d).
  - Open: x once per cluster (TMA multicast).
- **[M] Router at m >= 9:** a memset graph node is cheaper than a serialized last-arriver ticket epilogue (7e8be36a:
  x1.07-1.10 slower).

## Gates vs the judge (10-26)

- **[M] Write flush vs read flush.** bf16 `_cell_gate` flushes with `flush.zero_()` (a write); the judge reads a
  buffer larger than L2. The write leaves ~50 MB of dirty lines whose write-back the timed kernel pays, less so for a
  byte-light kernel. Gate vs judge:

  | cell | gate | judge |
  |---|---|---|
  | b12 pooler m=1 | 0.82-0.85 | 0.84 |
  | qkv m=1 | 0.88 | ~1.00 |
  | o m=1 | 0.87 | 0.97 |
  | b12x qkv m=8 | 0.976 | **x0.769** |
  | b12r pooler m=8 | 0.938 | 0.984 |

  `rewrite/lib/rt.py` (decode / moe) already reads. bf16 needs the same.
- **[M] Big T:** moe v4 T=256 gate p50 x1.12, judged row +1.3%. The judge's T=256 routing sits near our p90 draw.
- **[M] Same-bytes scored spread:** about ±1.2% (decode X0b: check 1.4569, scored 1.4401).
- **[M] Selector tie-breaks:** a 1% preference for the earlier engine cost d16 -4..-6% (a1bc1756). Use the best worst
  ratio.

## Tooling state at the pause

- Churn's BEST.json is stale for decode (1.3142, a903b9d0) and moe (1.4021, 31c27e4b): campaign runs submitted from
  worktree copies are invisible to churn. bf16's was registered by hand (3834ceec).
- `rewrite/bin/archive.py` hardcodes the old session's /tmp runs dir (`SCRATCH_RUNS`). Repoint it before use.
