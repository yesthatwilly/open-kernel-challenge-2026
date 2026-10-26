# Campaign log: 2026-10-26-decode-first-rewrite

Append-only. Newest entries at the bottom. Every check and scored run gets an entry: question asked, job,
rows that moved, decision. A new session resumes from the last entry.

## 2026-10-26T04:38Z: campaign approved, execution started

- The user approved PLAN.md ("execute the plan").
- Baseline: decode board 1.3164 (eaa8cc0f), master 1.3142 (a903b9d0). moe 1.4021, attention 1.5347, bf16 1.2566, prefill 1.468. Overall rank 480 of 1163.
- State at start: master d89b52b1 clean. One in-flight churn run: prefill `attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload`.
- Quota day 10-26 began at 00:00Z. Plan for today: 2 decode checks, 1 moe check, nothing scored.

## 2026-10-26T05:10Z: Phase 0 done; W0 infra built; first decode check dispatched

- Campaign docs committed on master (cd7fe010). Churn paused (11a14fc8): the harvest-only tick found nothing in flight.
- Branch `campaign/decode-first-rewrite`, worktree `.worktrees/decode-first-rewrite`, commit a1abc1d1:
  - `rewrite/build.py` generates each `<p>/submission.py` from `rewrite/src/<p>.py` by expanding `#@include`, then lints.
  - `rewrite/lib/rt.py` (W0): read flush, graph timing with an empty-graph baseline, selection rule, judge-form checks.
  - `rewrite/bin/report.py`: per-shape and per-load comparison against a base job plus the PATH CHOSEN lines.
- No nvcc or torch on this node; CUDA is validated on the judge, behind an Opus audit for each new module.
- **decode W1 check, job 9dacd0f59a174eaf.** Base = SKW branch a48a10b3 (master + SKW). Attention gate now bounds new o
  by 1.75x the fp64 FP8-query reference error (judge allows 2.5x) with the judge row form; S3 front code-gap gate off;
  routing-aware layer checks (SKW, and T=8 when an S3 T=8 cell is live); read-flush timers; SKW start 200 s.
  Question: which stranded cells (S1 A/B, S3 x7, SKW b32/v1) pass and win, and the headline effect.
- **moe ladder (W2 step 1)**: built (ea_common.cuh, ea_ladder.cuh, ea_ladder.py), audit running before its check.

## 2026-10-26T05:15Z: decode W1 check 9dacd0f59a174eaf: 1.3371 [1.3170, 1.3571], 13/13. M0 reached; scored submit sent

- vs board 1.3164: +1.57% (margin 1.24%). vs master a903b9d0 1.3142: +1.74%.
- Every stranded cell passed the judge-faithful gate and won its selector: S1 A 28.6 vs 39.5 us, S1 B 38.2 vs 45.6,
  S3 c1-b16/c1-b32/c2-t8/c2-t128/c2-t256/c3-t128/c3-t256/c4-t8 all NEW (3-18 us each), SKW b32 38.4 vs 45.7, SKW v1
  33.5 vs 38.1. S2a/S2b unchanged (NEW).
- Rows (base a903b9d0 -> run): v16 full 593.4 -> 547.0, v1 sliding 181.1 -> 177.0, d32 sliding 239.5 -> 236.2,
  v32 full 866.6 -> 851.4; d16 full 290.5 -> 302.3 (drift: same cells as master there).
- Loads: d1 1.441 (1.449), d16 1.359 (1.325), d32 1.250 (1.206), v1 1.502 (1.447), v16 1.282 (1.266), v32 1.213 (1.214).
- Decision: scored submit of the same bytes, job 69db8aeb251944c5 (PLAN §6, M0).

## 2026-10-26T05:30Z: M0 scored 1.3408 [1.3207, 1.3609]; landed on master

- Scored submit 69db8aeb251944c5: 13/13, the same 16 cells NEW. Board: decode 1.3164 -> 1.3408, rank 448 -> 434;
  overall rank 480 -> 460.
- Rows vs the check: d16 sliding 211.7 -> 198.4, v1 sliding 177.0 -> 168.2, v16 sliding 411.6 -> 428.9 (drift both ways).
- Landed: master muse_decode_layer/submission.py = branch a1abc1d1 bytes (sha256 74fd5a6b...). DECISIONS.md notes that
  BEST.json (churn-generated) still names a903b9d0.
- Next: moe ladder check once its audit returns; Engine A v1 (T <= 8) written, to be audited and checked in moe next.


## 2026-10-26T05:35Z: moe ladder check dispatched; Engine A v1 written and under audit

- Ladder audit (Opus): dispatch-with-fixes, no build/fault/hang issue. Its emulation and a g++ type check found 0
  dequant mismatches over 121,856 cases and 800,000 random words. Fixes applied: verdict links compare equal grids
  (new D3 stream rungs), error word re-checked after timing, raw times printed beside baseline-subtracted ones,
  cleanup never raises and frees the flush buffer, no duplicate log lines, 16-byte alignment guard.
- **moe ladder check, job a997f2587a654f7c** (branch b2421f48; CAMPAIGN_LADDER on, Engine A off).
- Engine A v1 (T <= 8): rewrite/lib/cuda/ea_kernels.cuh (K1 w13+silu, K2 w2 with PDL weight prefetch; CTA-shared
  18 KB bulk ring, mma.sync swap-AB, stream-K with last-arriver ticket), ea_core.py (repack, forward), ea_moe.py
  (canary, rates, selector, copy check), ea_decode.py (decode adapter; layer check; inactive). A numpy emulation of
  the repack + fragment decode: 0 mismatches over 262,144 elements for both code layouts. Opus audit running.

## 2026-10-26T05:45Z: moe ladder a997f2587a654f7c: Engine A is viable (kill bar 3.0 TB/s cleared)

Passed 5/5 (master path scored 1.3865, irrelevant). 201 MB bank, read flush, graph timing, baseline 5.70 us:

| rung | t_us (net) | TB/s net / raw |
|---|---|---|
| R1 stream, CTA ring D4/D6/D3, per-warp ring, 1-3 CTAs/SM | 43.7-44.9 | 4.48-4.61 / 3.98-4.07 |
| R2 + dequant prop (master) | 49.8 (50.5) | 4.04 (3.99) / 3.63 |
| R3 + mma.sync const B, N=8, prop (master) | 53.5 (59.4) | 3.76 (3.39) / 3.40 |
| R3 N=32 | 69.2 | 2.91 |
| R4 smem B via ldmatrix: 1 tok / 8 tok | 55.3 / 59.2 | 3.64 / 3.40 |
| R4 16 tok (2 CTAs/SM) / 32 tok (1 CTA/SM) | 70.8 / 147.1 | 2.84 / 1.37 |
| R6ref master kernel_fn T=1 local8 / T=1 zipf / T=8 / T=32 (active bytes) | 103.7 / 72.5 / 174.9 / 359.3 | 2.19 / 1.95 / 2.43 / 2.44 |

- dq_exact 0/121856. R0: the zero_ write flush inflates the 45 us stream by 8.83 us (+19.7%).
- Ring depth and occupancy do not move the pure stream; the dequant and MMA cost ~24% from R1 to R4 (8 tokens).
- Exact integer dequant: equal alone, 11% faster than master's once the MMA is in the loop.
- Decision: Engine A keeps D4 / 2 CTAs/SM / exact dequant / N = 8. T above 8 goes in passes of at most 8 routes per
  expert (repeat reads served from L2), never more than one n-tile. Gathering B per expert (n_e rows instead of all
  T) is worth about 7% at T = 8.

## 2026-10-26T05:52Z: Engine A v1 live in moe and decode (checks 7e40dd8509da4586, 485b670ef12f4909)

- User (05:47Z): continue autonomously until 2026-10-26 09:49Z (2:49 MST) using any and all checks, scored submits and
  Claude tokens; agent concurrency stays at <= 3.
- Engine A v1 Opus audit: dispatch-with-fixes. Applied to v1 and v2: K2's no-work exit now waits on K1 (else the
  combine could read K1's local_ids early on zero-local canary cases); host grid bound 512 (stream-K needs W >= G);
  cudaGetLastError() after a failed launch; the diagnostic rates call in its own try. Its name scan, run on the decode
  file, found `gc` unbound in the decode adapter's copy check (would have turned Engine A off); fixed.
- Branch df55c554: moe CAMPAIGN_LADDER off, CAMPAIGN_ENGINE_A on; decode CAMPAIGN_ENGINE_A_DL on. Checks dispatched in
  parallel with rewrite/bin/dispatch.py (fills the attempt file's job id).
- In parallel: an Opus audit of Engine A v2 (T <= 32); an Opus builder for Engine B (W3, T = 128..512) on branch
  campaign/engine-b in .worktrees/engine-b (no okc runs; I dispatch).

## 2026-10-26T06:12Z: Engine A v1 on the moe board (1.4434); decode v1 measured but not yet live

- moe check 7e40dd8509da4586: **1.4501** (master path 1.3865): decode 1 44.6 -> 40.5 us, verify 1 106.7 -> 95.2 us.
  Canary clean T 1..8 (worst rel-L2 0.00408), gates T=1 x1.19 / T=8 x1.15, copy check PASS, build 30.5 s, setup 32 s.
  Kernel-only: T=8 (19 active) w13 3.00 TB/s, w2 2.81 TB/s; T=1 (2 active) w13 1.88, w2 1.33 TB/s (latency-bound:
  ~9 us of fixed cost per kernel against 6-11 us of streaming).
- moe scored 1b3f70acd8c14900: **1.4434**, moe rank 490 -> 473 of 814.
- decode check 485b670ef12f4909: FAILED, illegal address in prepare. My gate captured master's expert block from a
  closure that owned its align tensors and dropped it after capture; the freed blocks were reused (n_padded garbage)
  and Marlin read out of bounds on replay. Fix: keep closures alive until the replays are done (7323673f).
- decode re-check 4594d4d07d114cdc: passed 13/13, 1.3488 (M0 1.3408), but Engine A was ON only at T = 2..7: the layer
  check left T=1/T=8 switched off after running (fixed b640edc4). Measured on decode routings (expert block given R's
  outputs): T=1 judge-like 40.9 -> 34.4 us (x1.19), one expert 33.3 -> 23.0 (x1.44); T=8 judge-like 110.9 -> 96.9
  (x1.14), skewed 62.5 -> 55.2 (x1.13); layer checks clean (hidden <= 2.0e-3). Expected once ON: d1 x1.05, v1 x1.07,
  headline about +2%.
- Next: unified adapters (ebac60b2: one module with v1 + v2; classes T = 1/8/16/32 choose master/v1/v2) go out in moe
  and decode as soon as the v2 audit returns. T=1 idea queued: align the stream-K split to whole groups when the
  groups fit the grid (no fix-ups at T=1 w13; 2-way at w2).

## 2026-10-26T06:32Z: Engine A v1 + v2 live in moe (scored 1.5269, rank 437) and decode (check 1.3783)

- v2 audit (Opus): dispatch, no findings. One module now holds v1 and v2; classes T = 1/8/16/32 pick master/v1/v2.
- moe check 33e25409b3e044a6: **1.5254**: decode 16 146.1 -> 127.1 us, decode 32 198.2 -> 168.0; T=1 v1 (x1.19),
  T=8..32 v2 (x1.19 / x1.17 / x1.06). Build 38.9 s (8 kernels at 2 CTAs/SM), setup 45.9 s. moe scored
  7c4d16997dfc43df: **1.5269**, rank 473 -> 437 of 815.
- decode check 903de4ed3a9c4558: **1.3783** (+2.8% over M0): d1s 118.3 -> 115.1, v1s 168.2 -> 158.6, d16s 198.4 ->
  180.5 us. T=1..8 v1, T=9..16 v2, T=32 master: v2 scored x0.698 on the gate's 4-hot-expert routing (4 passes per
  expert). Scored run c3b7440aa9394c04 sent.
- Profile insight: at the judge's T=16/32 the hot experts take 9-32 routes, so v2 re-dequantizes them per 8-route pass
  (ea2_w13 at ~2.1 TB/s vs ~3.0 with spread routes). -> v3: passes of up to 16 routes with two n8 tiles sharing each
  dequantized A fragment (ladder: 16 rows cost ~1.2x of 8, not 2x). Opus builder on branch campaign/ea-v3
  (.worktrees/ea-v3).
- Round 3 dispatched at 06:30Z (moe 223452681ddb4a22, decode 7f0f22f40f694929; git dc240ed8): aligned split (no
  fix-up at T=1 w13), K2 triggers dependents early, CUDA combine (PDL; check vs Triton + per-class gate), decode engine
  gate on whole-layer graphs of judge-like inputs. Focused Opus audit of that diff running in parallel.
- Decode T=1 anatomy (4594d4d0 profile, d1s 121.8 us of kernels): Marlin pair 42.2 (now Engine A), P pooler+router
  20.9 (75.5 MB bf16: ~3.6 TB/s, near floor), sliding attention 13.2, qkv nvjet 12.2 + splitK reduce 1.7, o 7.0,
  R 6.6, unpool 6.0, boundaries 3.6 + 3.4, combine 3.4. Remaining T=1 slack is launch gaps and PDL breaks (W4).

