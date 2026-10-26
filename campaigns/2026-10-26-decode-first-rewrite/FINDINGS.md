---
campaign: 2026-10-26-decode-first-rewrite
updated: 2026-10-26T05:00Z
---

# Findings: the evidence base for the decode-first rewrite

This is the digest that PLAN.md rests on. The full sources are in `research/`.
Tags: **[M]** judge-measured, **[P]** prepare-time measured, **[I]** inference or estimate.

| source in `research/` | what it holds |
|---|---|
| `decode_layer_read.md` | decode time budget per shape, ceilings, stranded wins, port-lag table, ROI paths, queue verdicts |
| `nvfp4_engine_forensics.md` | every NVFP4 kernel's limiter, the instruction budget, the exact cheaper dequant with proof, Engine A/B designs, the R0–R6 ladder print format |
| `dense_gemm_forensics.md` | bf16 cell table with floors, why m≥128 custom kernels lost, the one-M-tile split-K design, the decode port map |
| `board_fit.py` → `board_fit.out.txt` | how the overall board ranks teams; decode speedups needed for overall top-N |
| `roi_model.py` → `roi_model.out.txt` | decode time per component class per shape; headline under component scenarios |
| `leaderboard-2026-10-26/*.json.gz` | the six `okc leaderboard --json` snapshots used above |

## 0. User decisions (2026-10-26)

- Objective: the **overall board**. Churn is **paused**; all quota goes to the campaign and unspent checks are accepted.
- One agent writes all code. A fresh read-only Opus audit reviews each new or changed CUDA module before its first check. No other subagents.
- Keep agent fan-out at 3 or fewer concurrent agents: the user's token quota is the binding constraint, not compute.
- Plans larger than a hypothesis live in `campaigns/`.

## 1. The overall board is the decode board [M]

From `board_fit.out.txt` (632 teams that solved all four kernel boards):

| ordering formula | Spearman vs overall | adjacent-order violations |
|---|---|---|
| decode_layer rank alone | **0.9961** | **98** |
| weighted log speedups, time-share weights | 0.9936 | 310 |
| equal-weight log speedups | 0.9766 | 311 |
| mean rank over the four boards | 0.9495 | |
| moe / attention / bf16 rank alone | 0.9465 / 0.8481 / 0.6295 | |

- The least-violating weight grid puts zero weight on the other three boards.
- Holding attention, bf16 and moe at today's values, overall rank needs this decode headline:

| overall target | decode needed |
|---|---|
| top 300 | 1.450 |
| top 200 | 1.606 |
| top 100 | 1.783 |
| top 50 | 1.933 |
| top 25 | 2.026 |

- Consequence: decode_layer is the KPI. moe, bf16 and attention are testbeds with their own quota and isolated timing.

## 2. Where decode_layer's time goes [M, profile a903b9d0]

From `roi_model.out.txt`. Times in µs; shares are of GPU-busy time.

| load / layer | T | ours | SGLang | experts | dense | attention | experts / dense / attn / small |
|---|---|---|---|---|---|---|---|
| decode 1 sliding | 1 | 118.5 | 171.0 | 47.7 | 48.5 | 13.5 | 39/39/11/11% |
| decode 1 full | 1 | 124.3 | 182.1 | 47.5 | 49.2 | 20.5 | 36/38/16/10% |
| decode 16 sliding | 16 | 210.5 | 263.2 | 100.3 | 63.0 | 19.5 | 49/31/9/11% |
| decode 16 full | 16 | 290.5 | 430.7 | 79.1 | 65.4 | 112.2 | 29/24/41/5% |
| decode 32 sliding | 32 | 239.5 | 260.2 | 128.6 | 66.3 | 23.8 | 53/27/10/10% |
| decode 32 full | 32 | 446.4 | 620.6 | 119.9 | 65.7 | 224.0 | 27/15/51/6% |
| verify 1 sliding | 8 | 181.1 | 263.1 | 88.2 | 49.6 | 16.5 | 49/28/9/14% |
| verify 1 full | 8 | 189.5 | 271.3 | 88.5 | 51.1 | 29.3 | 46/26/15/13% |
| verify 16 sliding | 128 | 414.7 | 524.5 | 323.8 | 80.0 | 18.9 | 69/17/4/10% |
| verify 16 full | 128 | 593.4 | 751.9 | 277.9 | 79.1 | 161.9 | 48/14/28/10% |
| verify 32 sliding | 256 | 599.8 | 720.9 | 408.8 | 101.2 | 20.2 | 70/17/3/9% |
| verify 32 full | 256 | 866.6 | 1073.7 | 375.5 | 100.1 | 295.1 | 45/12/35/9% |

Load speedups: d1 1.449, d16 1.325, **d32 1.206 (weakest)**, v1 1.447, v16 1.266, v32 1.214. Load weights: 46 per sliding row, 16 per full row.

| scenario (exact load math) | headline | ≈ decode rank |
|---|---|---|
| experts ×0.75 | 1.502 | 300 |
| experts ×0.6 | 1.645 | 200 |
| experts ×0.5 | 1.756 | 150 |
| experts ×0.4 | 1.884 | 75 |
| dense ×0.8 (T≤32), ×0.6 (T≥128) | 1.400 | 400 |
| full-layer attention ×0.76 / ×0.75 | 1.348 | 450 |
| small ops ×0.5 | 1.387 | 400 |
| experts ×0.5 + dense | 1.913 | 75 |
| experts ×0.5 + dense + small ×0.5 | 2.071 | 25 |

- The non-MoE program alone caps at about 1.36–1.42 [I]. The experts must change.
- Per-shape error is 0.7 µs at decode_1 and 2.5–7.3 µs elsewhere. Sliding decode_16 drifts 195.6–217.2 µs on master-equivalent code, so effects under about 5% there are unresolvable.
- Board 1.3164 (eaa8cc0f, branch code) and master 1.3142 (a903b9d0) run the same scored kernels; the gap is noise (22 identical-code pairs differed by at most 1.01%).

## 3. Root causes of the plateau

### 3.1 The NVFP4 consumer, not DRAM [P/M]
- Expert shapes: H = I = 4096; w13 [8192, 4096], w2 [4096, 4096]; **28.31 MB per expert** (0.5625 B/element).
- Mean touched local experts per call, backed out of the moe SoL floors: d1 2.6, v1 8.4, d16 11.9, d32 16.0, v16 22.9, v32 24.6. Decode's own estimates are lower: d1 about 1.9, d16 4.8–5.7, d32 5.2–6.8, v1 about 5.7, v16 about 14, v32 about 16 [I].
- Every repo NVFP4 kernel uses master's `dequant8` plus a cvt scale path: 26–31 instructions per lane per 8 elements, about 3.5 issue slots per element.
  - At 4 TB/s that needs **72–93% of the INT32 ALU pipe** and 64% of the conversion pipe. Practical ceiling about 3.2–3.6 TB/s however it is scheduled.
- The wg consumer runs `wait full` → dequant → 8 wgmma → `wait_group<0>` → arrive, with both consumer warpgroups lockstepped on the same stage. About 1150–1200 clk per 9216 B stage per SM.
- Measured weight-stream rates: wg 1.8–2.2, wg2 2.1–2.5, fsi 1.6–1.9 TB/s; Marlin about 2.0–2.5 in moe and **1.16–1.92 on decode's active-expert bytes** (worst at T=1 and T=128/256).
- The rowowner ring kept only about 55 KB/SM in flight. bf16 streams reach 3.4–3.6 TB/s only at 96–128 KB/SM.

### 3.2 Measurement bias [P]
- Every prepare gate flushes L2 with `flush.zero_()`, a 256 MB **write**; the judge **reads** its flush buffer.
  - About 50 MB of dirty lines are written back during the measured kernel: +66% HBM traffic for a 75 MB kernel, +8% for 600 MB.
  - Evidence: bf16 pooler m=128 master 27.3 µs in prepare vs a 22.85 µs judge row; qkv m=256 23.4 vs 19.46.
  - Sites: moe `submission.py` ~2350 (`_wg_med_cold`), ~2641 (selector), ~655; bf16 L1488/1527/2666/2808; decode L4913/6596.
- Many "kernel-only" TB/s numbers are eager single launches. Example: bf16 pure read 2.54 TB/s in prepare vs 64 MB in 19.0 µs = 3.37 TB/s in the judge profile.
- Prepare CUDA-event timings undercount the judge by 20–25%.
- Two verdicts were measured in the wrong regime: "consumer is not the limiter" (wait<1> variant B, on a grid-starved one-expert canary) and "bytes in flight is not the limiter" (true only for the wg consumer at 1 CTA/SM).

### 3.3 Self-inflicted stranded wins [M/P]
Code exists and wins, but self-checks against master turned it off:

| cell | evidence | status |
|---|---|---|
| S1 B (Triton S=16, sliding B=17..32) | d32s 239.5 → 231.0 in 3117f2e4 [M]; prepare 43.17 vs 51.55 | fell back |
| S1 A (B=2..16) | prepare 32.74 vs 44.58 [P]; judge rows within d16s drift | fell back: o vs master 4.02e-2 > 1e-2, while vs fp64 new 1.22e-2 and master 3.99e-2 |
| SKW v1 (windowed stream-K, sliding verify_1) | 38.69 vs 48.27 µs over 2 rounds [P] | fell back: layer check with 6/32 routing ids differing on near-ties |
| SKW b32 | 43.07 vs 51.10 µs | ties S1 B |
| S3 c1-b32, c2-*, c3-*, c4-t8 | new error vs fp64 equal to or below master's | fell back: "o vs master ≤ 1e-2" or gap gate |
| S3 c1-b16 | ON in a903b9d0, OFF in eaa8cc0f | master's own gate flips run to run |

Source branch for the restack: `hyp/muse_decode_layer/sliding-sk-window-port-decode32-verify1` (a48a10b3). Logs: a903b9d0 lines 58–71; 3117f2e4 lines 52–84.

### 3.4 Split-K plumbing, not mainloops, sank dense m≥128 [P/M]
- CUTLASS DataParallel ran 6.86 TFLOP/s per SM (85–92% of peak) on the judge; its Deterministic StreamK fixup plus an `initialize_workspace` memset node added 10–32 µs per call.
- An extra graph node costs about 1.5–2.5 µs per row at these sizes.
- Largest bf16 gaps (row vs floor, µs): pooler m=256 27.98 vs 17.37; qkv m=256 19.46 vs 9.77; pooler m=128 22.85 vs 14.64; qkv m=128 15.55 vs 8.42; router m=256 9.72 vs 2.68.
- m≤32 cells sit 0.5–3 µs above a practical floor. The only 3.97 TB/s kernel (bulk m=1) is also the only one with a tile-contiguous weight repack.
- **Rules risk:** the "K-halves fp32 partials" runs (de95f817, caaa76cc) actually routed bf16 partial sums. Never promote them.

### 3.5 Port lag and divergence
- Decode still runs Marlin at every T, nvjet at T≥8, and has the stranded attention cells above.
- The attention and moe boards are branch code, not master.
- moe split-I midt (6fe2a63f) lost 0.44–0.81× after DIAGNOSIS.md was written; DIAGNOSIS action 2 is contradicted. Rega was gated OFF by decode's own selector at B128 and B256 (eaa8cc0f).

## 4. The exact cheaper dequant (Engine A/B core) [I, proven]

Full proof in `research/nvfp4_engine_forensics.md` §3.

- **Bias split:** 2^126 = code 2^−126 × scale 2^+119 × epilogue 2^+7.
- **Code:** for e2m1 c = (s, e1, e0, m), bf16 bits `s<<15 | E<<7 | m<<6` (E = 2·e1 + e0) equal v(c)·2^−126 exactly, including the subnormal E = 0 case and ±0.
- **Scale, integer-only:** for an e4m3 byte b in [0x08, 0x7E], `S = (b<<4) + (239<<7)` is the bf16 of sf·2^119. Broadcast {S, S} via `PRMT(w, RZ, 0x4040)` then `IMAD(t, 16, 0x77807780)`; no carry between halves.
  - Precondition checked once in prepare: every scale byte in [0x08, 0x7E]. The generator clamps sf to [8, 448], so it always holds; otherwise fall back to master's cvt path.
- **Product:** `fma.rn.bf16x2(B, S, −0)` = v·sf·2^−7 exactly (≤6 significant bits; magnitude in [2^−14, ≈21], all normal).
- **Epilogue:** `acc · (gs·2^7)` in fp32.
- **Code bit layout (prepare-time lossless permutation per 32-bit word):** target set P = {6,7,8,15, 22,23,24,31}.
  - pair0 = `q & 0x81C081C0`; pair1 = `(q<<3) & 0x81C081C0`; pair2 = `(q<<6) & 0x81C081C0`;
  - pair3 = `((q>>3)&0x01800180) | ((q>>7)&0x00400040) | ((q<<1)&0x80008000)` from leftover bits {10,11,13,14} (+16).
  - About 6 INT + 5 IMAD per 8 elements instead of 12–15 INT plus 3 conversions.
- Expected effect: INT-ALU per element 1.7–2.2 → 0.9–1.2; conversions → 0. At 4 TB/s the INT pipe goes from 72–93% to 38–52%.
- `dq_exact` check: bitwise equality against master's dequant over 16 codes × 119 valid scale bytes (1904 cases).

## 5. Engine designs (summary; details in the forensics report)

**Engine A (≤32 routes per expert):** K1 w13 + SiLU (builds the device work map, zeros `ss`) → K2 w2 + per-route sum of squares (ring prefetched before `griddepcontrol.wait`) → K3 post-norm combine, chained with PDL.
- CTA-shared 18,432 B 1-D `cp.async.bulk` stages, 4–6 deep, 2–3 CTAs/SM, ≥128 KB/SM in flight. 1 producer warp + 4 consumer warps.
- mma.sync m16n8k16 swap-AB: weights are A, 16 rows per warp over the full K; tokens are B, N = 8·⌈n_e/8⌉ ≤ 32; 2 accumulator sets.
- Repack: [expert][64-row group][512-k chunk][warp][16 rows × 512 k = 4096 B codes + 512 B scales]; w13 tiles interleave 8 gate + 8 up rows of the same 8 h-columns.
- Stream-K over (group, k-chunk) with ≤2 partial groups per CTA; fp32 workspace, per-group ticket; the last arriver reduces and resets. No spins.
- Expected (moe, judge): d1 46.4 → 26–29, v1 104.5 → 73–84, d16 150.9 → 101–117, d32 199.1 → 134–156 µs.

**Engine B (T = 128/256):** wgmma m64nNk16 RS with BK = 64 stages, double-buffered register A and `wait_group<1>`; N ≤ 64 at 2 CTAs/SM (setmaxnreg 32/104); N = 128/256 at 1 CTA/SM with NMAX ≥ max routes so hot experts are read and dequantized once; LPT order and a stream-K tail. Expected v16 323 → 215–245, v32 358 → 235–265 µs.

## 6. Counterinteractions and their guardrails

| interaction (evidence) | guardrail |
|---|---|
| PDL only pays between own kernels: P after SGLang's boundary +0.3 µs; own chain Q → sk → O 57.25 → 48.96 µs [P], full d1 −7.8 µs [M] | own the whole T≤32 chain before tuning launch gaps |
| Side streams contend: L2-prefetch side stream regressed d16s +21%; T>32 router (~25 µs) races the pooler (~30 µs) | no side streams in new code; at T>32 measure join slack first |
| Gates derived from shared constants (ALIGN_FUSED_MAX_TOKENS) | one bound constant per path |
| Module-global tensor maps across the judge's 2 state copies (moe g_maps bug) | per-state pointers and descriptors; 2-state alternation canary |
| Cross-CTA spins made the judge loop bimodal (fused bs1: 41–59% of replays +23 µs) | last-arriver tickets only |
| moe verdicts do not transfer to decode (rega won in moe, gated OFF in decode B128/B256) | decode selectors time decode routings |
| Kernel-vs-kernel comparisons mislead (S1 14.6 vs FA3 13.7 µs, while the FA3 chain is 25–29 µs) | selectors time the whole replaced path |
| Marlin profile rows swing ±20–40 µs between runs on identical code | judge on paired shape rows plus kernel names, not single profile rows |
| An illegal instruction in prepare poisons the context; no child processes allowed | canary, `-lineinfo`, proven instruction forms only, first run of a new module is a check |

Proven-on-judge instruction forms: 1-D `cp.async.bulk` + mbarrier, `mma.sync` bf16, `fma.rn.bf16x2` with −0, `cvt.rn.f16x2.e4m3x2`, wgmma RS, setmaxnreg, `griddepcontrol`, `ldmatrix`/`stmatrix`, `atomicAdd(float4*)`. Avoid `createpolicy` / `L2::cache_hint`.

## 7. Do not try (closed with evidence)

Triton NVFP4 GEMV; swap-AB persistent mma.sync with `__syncthreads` per stage; grid-spin fused bs1; split-I with fp32 y `red.add` (fsi midt 0.44–0.81×); rega cluster multicast (MC2 1.3–1.9× slower); ping-pong consumer warpgroups (1.18–1.53× slower); producer-side scale expansion (1.03–1.13× slower); deeper rings on the wg consumer; lossless bf16 compression; full-layer attention rewrites (sk is at 1.75× its KV floor, every verify kernel lost to FA3); DeepGEMM (the API is cuBLASLt here, the raw kernel has no split-K); CUTLASS Deterministic StreamK fixup; Marlin small-M knobs beyond bsm 8 / two-stream / hot-cold cells; align and post-norm micro-work; pairing on sliding layers; the queued `fsi-moe-midt-decode-port`.

## 8. Open questions the first checks must answer

1. The real DRAM ceiling on the judge for 18 KB 1-D bulk stages at 2–3 CTAs/SM (ladder R1).
2. The size of the write-flush penalty (ladder R0); it rescales every prepare TB/s in the repo.
3. Whether the proposed dequant is exact on the judge (`dq_exact`) and what it costs relative to R1 (R2 vs master's dequant).
4. Whether mma.sync swap-AB at N = 8..32 keeps up with the stream at full grid (R3/R4).
5. Engine A vs Marlin at T=1 on a judge-like routing (R6).
6. Whether Marlin re-streams expert weights per m-block at T=128/256 or L2 absorbs it.
7. What the +15–20 µs bench-over-kernel-sum on d16f, v16f and v32s is.
