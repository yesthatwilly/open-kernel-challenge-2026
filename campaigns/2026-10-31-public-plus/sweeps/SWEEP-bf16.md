# SWEEP-bf16: fresh-submission variants on master BF16FIX (muse_bf16_proj)

Design only, written 2026-11-02 ~01:00Z. Nothing ran on a GPU, nothing in the repo changed, no okc command ran.
Base: master `muse_bf16_proj/submission.py` = BF16FIX-score, sha256 `abd182ea…`, 383,506 B (master 8d0df028 lineage).
Board: 1.5657 (109f577b, scored 10-31; the pre-fence bytes). Target asked: a scored read >= 1.5814 (board +1%).

**Gate metric, in every gate below.** rho(row) = geomean over the reads of that row's `paired_ratio`
(okc status JSON, `result.records[]` with `event == "benchmark"`, field `paired_ratio`, matched on case.load + case.layer),
divided by the row's reference R0 (§1). It is **not** the median of `ratios`. Headline = geomean over the 6 loads of
Σ SGLang ns / Σ (SGLang ns / paired_ratio); this reproduces every read's okc speedup exactly (checked on 10 reads).

"Floors" here means two things: (a) the per-row gate floors from the pooled reference (§1, §4), and (b) the judge-fitted
physical floor per row from GAPMAP-1031b (t0 2.33 us + bytes / 4.57 TB/s, or FLOP / (0.85 x 989 TF)), used as the cap
on every predicted row gain.

## 0. Bottom line

1. **Only one knob has judge evidence: P1, the RBX unpooler-128 cuBLASLt plan.** In all 3 judge reads that ran it
   (BFR-r1 7d43d14a, BFR-r2 a16e53de, RBX-r2 ad738856), the v16 unpooler row read rho 1.0261 / 1.0330 / 1.0225
   (pooled **1.0272**). That row's master reference is very tight: 7 reads, sd 0.34% per read. That is **+0.13%
   headline**. The H200 in-prepare race picked it in 3 of 4 races, and in 6 of 6 race passes it read ≤ 1.000 x
   production. The race is what made RBX's P1 arm a no-op in one read. **V1 routes it by shape, with no race.**
2. **The rest of the scheduling-knob space is already mined.** The 10-31 H100 lever finder (lev-bf16 exp1-exp8)
   swept cgemv geometry, rsmall/rk split counts, sk2 cluster split-K / row blocks / ring depth, qbig/olayer/rl configs
   and 543 cuBLASLt plan checks (exp7 192, exp8 351). The judge raced the rest (BFR, RBX). Every requested direction except P1 is settled (§2).
   The riders below are real, compliant changes. Their expected judge effect is 0 to +0.02% headline, below 2-read
   resolution. **Treat V2-V5 as distinct fresh bytes at V1's level, not as separate gains.**
3. **No knob variant reaches board +1%.** Master's own level is 1.5459 (mean of 7 master-equivalent reads, sd 0.78% per
   read) or 1.5489 (pooled-row model). The board 1.5657 is a high read of master, +1.1% over its level. Clearing
   1.5814 needs about +2.1% over master. With V1's +0.13%, P(one scored read > 1.5814) = **0.3-0.6%** and
   P(read > 1.5657) = **7-11%**. Even a real +1.0% would give only 5-8% per read for the +1% target. The only bf16 path
   with that size is the 12-bit m=128 GEMM (GAPMAP-1031b candidate A, +1.5..+3.4% if it works). It is a build; its
   decider BF-DX2 was never run (the 10-31 diag workflow was stopped).

| rank | variant | knobs (one build each) | touched rows: predicted rho [range] | dH central [range] | p | EV (dH x p) | P(read > 1.5657) | P(read > 1.5814) |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | **V1 P1F** | (unpooler,128) static Lt plan nn(18,4,2) | v16 unpooler 1.027 [1.022, 1.033] | **+0.13%** [+0.11, +0.16] | 0.90 | **0.116%** | 0.07-0.11 | 0.003-0.006 |
| 2 | **V4 P1FX** | V1 + QB3 + OL1 | V1 + v16 qkv 1.000 [0.99, 1.02] + v16 o 1.010 [0.99, 1.02] | +0.15% [+0.09, +0.20] | 0.85 | ~0.124% | 0.08-0.12 | 0.004-0.007 |
| 3 | **V3 P1FOL** | V1 + OL1 | V1 + v16 o 1.010 [0.99, 1.02] | +0.15% [+0.11, +0.17] | 0.88 | ~0.124% | 0.08-0.12 | 0.004-0.007 |
| 4 | **V2 P1FQB** | V1 + QB3 | V1 + v16 qkv 1.000 [0.99, 1.02] | +0.13% [+0.10, +0.19] | 0.88 | ~0.116% | 0.07-0.11 | 0.003-0.006 |
| 5 | V5 P1FRL (H100-screen-gated) | V1 + RL7 | V1 + v32 o 1.00 [0.98, 1.04] | +0.13% [+0.09, +0.20] | 0.6 | ~0.08% | 0.07-0.11 | 0.003-0.006 |

- p = P(the variant reaches a check GO with its gain intact).
- Ranks 2-5 differ from V1 by under 0.02% in expectation. Order: V1 first (cleanest, judge-proven). Then V4 (the most
  extra content; both riders pass correctness on the H100 (exp4), and QB3 also ran clean on the judge in BFR-r1). V5 is
  sent only on an H100 screen pass.
- P(read > X) uses a normal model on log headline, sd 0.775% per read. The two numbers per cell use the two level
  estimates (1.5459 / 1.5489) times the variant's dH. Script: `sweep-bf16/table.py`.

**Recommendation (bf16 quota today: 5 scored / 10 checks).**
- V1: checks r1 + r2, then scored r3 on GO.
  - Fast path, if the main thread prefers: r1 only, then scored r3. P1 is already judge-proven on 3 reads; r1 only has
    to confirm the static route and its kernel name.
- V4 then V3/V2: 1-2 checks each, then a scored run on GO.
- That is 6 checks and 4 scored runs. P(at least one scored read > 1.5657) ≈ 25-35%; P(any > 1.5814) ≈ 1-3%.
- Hold the last scored run and 4 checks for V5 (on an H100 pass), or for BF-DX2 if someone builds it.
- Board moves from read noise do not survive the final review (median of reruns). Only P1F's +0.13% is real.

## 1. Reference pool R0, floors and elasticities

**Pool.** 10 judge reads of identical row routes, `sweep-bf16/runs/` (profile/okc JSONs) and `pool.py` (writes
`ref.json`):
- 7 master-equivalent reads, all 30 rows:
  - 109f577b (ADOPT-score), a9547928 (B0-aa1), 280a3622 (B0-aa2);
  - 754e0c9d (BF16FIX-r1), a3aa275b (BF16FIX-r2), d1f5f46c (BF16FIX-score);
  - 4b5887c7 (RBX-r1: both races picked production, so the GPU work equals master's; a CUDA graph replays no Python).
- 3 more reads, for every row they did not change:
  - 7d43d14a (BFR-r1) minus v16 qkv (3-D map arm) and v16 unpooler (Lt);
  - a16e53de (BFR-r2) minus v16 unpooler;
  - ad738856 (RBX-r2) minus v16 unpooler.
- BF16FIX's fences are inside the A/A band on every row; pre- and post-fence reads are pooled.

**Calibration** (`calib.py`, 21 master read pairs):
- 2-read G_u sd 0.58%. G_u ≥ 0.990 false-fails 4.8% of the pairs.
- A 2-read headline mean ≥ 1.5350 false-fails 4.8%.
- A per-row hard floor of R0/1.03 false-fails 14 of 21 pairs, so it is unusable. R0/1.05 false-fails 1 of 21: it is the
  untouched-row hard floor, with R0/1.03 flagged only.

**Columns.**
- "us now" = SGLang mean / R0 (judge net).
- "floor" = GAPMAP-1031b physical floor in the row's current byte format: 12b = lossless 12-bit, raw = BF16 / hi+lo.
- e_r = %H per 1% row speedup; Σ e_r = 1.
- Model headline at R0 = 1.5489.

| row | n | R0 | sd/read | R0/1.05 | us now | floor us | fmt | gap us | %H/us | e_r |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| d1 qkv_gate | 10 | 1.6121 | 1.51% | 1.5353 | 8.78 | 8.54 | 12b | 0.24 | 0.483 | 0.0424 |
| d1 o | 10 | 1.3297 | 1.76% | 1.2664 | 5.18 | 5.09 | 12b | 0.09 | 0.483 | 0.0250 |
| d1 pooler | 10 | 1.5987 | 0.76% | 1.5226 | 13.38 | 13.36 | 12b | 0.02 | 0.483 | 0.0647 |
| d1 unpooler | 10 | 1.9824 | 3.03% | 1.8880 | 3.52 | 3.71 | 12b | 0 | 0.483 | 0.0170 |
| d1 router | 10 | 2.8575 | 2.18% | 2.7214 | 3.63 | 4.18 | raw | 0 | 0.483 | 0.0175 |
| d16 qkv_gate | 10 | 1.4432 | 2.17% | 1.3745 | 9.79 | 8.61 | 12b | 1.18 | 0.350 | 0.0343 |
| d16 o | 10 | 1.2066 | 2.33% | 1.1491 | 5.59 | 5.15 | 12b | 0.44 | 0.350 | 0.0196 |
| d16 pooler | 10 | 1.5443 | 0.82% | 1.4708 | 13.92 | 13.45 | 12b | 0.47 | 0.350 | 0.0488 |
| d16 unpooler | 10 | 1.3527 | 1.20% | 1.2883 | 13.67 | 13.45 | 12b | 0.22 | 0.350 | 0.0479 |
| d16 router | 10 | 3.2162 | 3.85% | 3.0630 | 4.61 | 4.23 | raw | 0.38 | 0.350 | 0.0162 |
| d32 qkv_gate | 10 | 1.3420 | 2.85% | 1.2781 | 10.60 | 8.68 | 12b | 1.92 | 0.323 | 0.0342 |
| d32 o | 10 | 1.1555 | 3.30% | 1.1005 | 5.93 | 5.22 | 12b | 0.71 | 0.323 | 0.0191 |
| d32 pooler | 10 | 1.4582 | 1.27% | 1.3888 | 15.06 | 13.53 | 12b | 1.53 | 0.323 | 0.0486 |
| d32 unpooler | 10 | 1.2914 | 1.46% | 1.2299 | 14.52 | 13.53 | 12b | 0.99 | 0.323 | 0.0469 |
| d32 router | 10 | 2.7606 | 3.85% | 2.6291 | 5.54 | 4.29 | raw | 1.25 | 0.323 | 0.0179 |
| v1 qkv_gate | 10 | 1.5412 | 1.29% | 1.4678 | 9.24 | 8.57 | 12b | 0.67 | 0.456 | 0.0422 |
| v1 o | 10 | 1.3003 | 1.98% | 1.2384 | 5.17 | 5.12 | 12b | 0.05 | 0.456 | 0.0236 |
| v1 pooler | 10 | 1.5689 | 1.08% | 1.4942 | 13.63 | 13.40 | 12b | 0.23 | 0.456 | 0.0622 |
| v1 unpooler | 10 | 1.6755 | 3.59% | 1.5957 | 4.18 | 3.73 | 12b | 0.45 | 0.456 | 0.0191 |
| v1 router | 10 | 6.1000 | 5.28% | 5.8095 | 4.32 | 4.20 | raw | 0.12 | 0.456 | 0.0197 |
| v16 qkv_gate | 9 | 1.2224 | 1.58% | 1.1642 | 13.86 | 11.19 | raw | 2.67 | 0.252 | 0.0349 |
| v16 o | 10 | 1.0745 | 1.18% | 1.0233 | 6.94 | 6.53 | raw | 0.41 | 0.252 | 0.0175 |
| v16 pooler | 10 | 1.2414 | 3.22% | 1.1823 | 20.27 | 17.72 | raw | 2.55 | 0.252 | 0.0511 |
| **v16 unpooler** | 7 | **1.0361** | **0.34%** | 0.9868 | 19.08 | 17.72 | raw | 1.36 | 0.252 | 0.0480 |
| v16 router | 10 | 2.7046 | 2.54% | 2.5758 | 6.03 | 4.66 | raw | 1.37 | 0.252 | 0.0152 |
| v32 qkv_gate | 10 | 1.1297 | 2.22% | 1.0759 | 17.57 | 13.82 | raw | 3.75 | 0.205 | 0.0360 |
| v32 o | 10 | 1.0420 | 1.21% | 0.9924 | 9.02 | 7.44 | raw | 1.58 | 0.205 | 0.0185 |
| v32 pooler | 10 | 1.2600 | 1.01% | 1.2000 | 24.46 | 22.76 | raw | 1.70 | 0.205 | 0.0501 |
| v32 unpooler | 10 | 1.0459 | 0.55% | 0.9961 | 23.30 | 22.76 | raw | 0.54 | 0.205 | 0.0477 |
| v32 router | 10 | 2.4702 | 1.21% | 2.3526 | 7.09 | 5.15 | raw | 1.94 | 0.205 | 0.0145 |

**Where the floor gaps are, and why knobs do not reach them.**
- **v16 pooler/unpooler (2.55 / 1.36 us):** bandwidth-bound at a 67 MB read. The H200's size-matched ceiling at
  66 MiB is only 3.6-3.8 TB/s (public H200_MEASURED_FACTS §1), so these rows sit near it.
  - P1 takes 0.51 us of the unpooler gap.
  - The pooler's other plans lost the H200 race (BFR, 2 of 2 reads).
- **v16 qkv (2.67 us):** L2->SM x re-reads (108 CTAs x 0.67 MB of x per 0.35 MB of W) inside the public qbig kernel.
  - Its G/S/ST space is pinned: NR = 64 forces G ≥ 36, ST = 8 already fills 221 KB of smem, and other G/S broke (exp4).
- **v32 qkv (3.75 us):** L2-transfer-bound nvjet 72x64.
  - Every cuBLASLt alternative and every split-K / stream-K kernel lost: exp8, public NEG bf16 #1/#3, CLAIMS
    bf16-qkv256-cutlass-streamk.
- **d32 rows (1-1.9 us):** decode-issue-bound sk2.
  - Cluster split-K, larger row blocks and ring depth all lose on the H100 (exp3, exp5).
  - Public NEG decode #8: the sk2 qkv saturates near 96 CTAs.

## 2. Requested directions already settled (do not spend checks)

H100 numbers quoted in §2-§3 are **time ratios t(arm)/t(production)**, lower is faster (lev-bf16 convention). §5's
new screen uses S = t(M)/t(arm), higher is faster.

**Split counts.**
- rk router: S 8/10/14/16 at m128 and S 4-7 at m256 lose cold to production's S12 / S8 (lvbf exp2, H100 GPU 4, run
  20261031-011039). m256 cannot go above S8: all CTAs must be resident for the in-kernel clear (4 x 4 x S ≤ 132).
- sk2 cluster split-K (CL 2/3/4): qkv m32 1.17-1.54x slower, pooler m32 1.12-1.89x, unpooler 1.16x (exp3).
- qbig S/G: S2 and other ST configs fault (misaligned address), and ST 6/7 are slower (exp4).
- cuBLASLt split-K: absent from the top 60 of a 1,789-2,309-config sweep (public NEG decode #6). At m ≥ 128 split-K
  pays tile area x 4 B / ~45 GB/s per-SM L2 writes (public NEG bf16 #1).

**CTA counts.**
- sk2 row blocks: unpooler m32 R64 1.012 (exp3). The kernel saturates near 96 CTAs (public NEG decode #8).
- cgemv m=1 geometry: production ±1-2%, and the ±1-2% variants are already raced per layer in prepare (`_IV_CG_CONFIGS`;
  exp1).
- Empty-kernel launch cost is flat across cluster sizes 1-16 (1.50-1.70 us net; lvbf micro 20261031-005619).

**PDL / launch overlap.**
- Every row is one kernel node: flush, then a fixed ~4 us, then ONE kernel, then the end event (GAPMAP-1031b nsys, H100).
  There is no second node inside the timed region to overlap.
- Overlapping the judge's flush (outside the timed events) would be gaming. Not proposed.

**Tile shapes per row.**
- cuBLASLt (exp7/exp8, 192 + 351 plans):
  - unpooler m128: the P1 plan is the only one that transferred (V1).
  - pooler m128: non-coop 64x64 lost the H200 race 2 of 2 (BFR; CLAIMS node-only).
  - m256 pooler/unpooler/qkv: production is optimal or tied.
- olayer PIPE=1: warm 0.80 but cold 1.04, and the H200 race kept production (BFR).
- qbig RED=4 / 3-D map: warm 0.93 / 0.92 on the H100, cold 1.006 / 0.994.

**RBX P1 arm alone.** Yes: V1, in its forced form.
- `mk_rbx.py P1` (the race form) is H100-unvalidated (RBX audit amendment).
- Its race kept production in RBX-r1 (r = 0.9972, 1.0000), which wasted that read.
- The forced route has never read slower: 6 of 6 H200 race passes (BFR 2, RBX 4) were ≤ 1.000 x production.

**RBX R32 (rsmall<4,16,8> at d32 router):** KILL stands. The H200 race ratios were 1.000 / 1.007 and 0.989 / 0.996.

**Not proposed (needs a build or loses on paper).**
- **d32 warp-split reshapes** (sk2 WSK x UPW at fixed R / CTA count, e.g. qkv32 WSK 8 / UPW 1):
  - untested on the public sk2;
  - but `sk2_prepare` builds a weight stream for every SK2_CONFIGS geometry, so a new geometry adds a 28-50 MB 12-bit
    copy (the NEG decode #11 memory-layout risk);
  - p ≈ 0.15 x +0.1-0.2%.
- **Nathen's m32w / cps32 kernels:** present in the file but never promoted (public "No route is promoted"). The cps32
  family is not built at all (`_NDD_prepare_cps` skips `_NDD_CPS32_TABLE_MT` keys).

## 3. Variant specs

**Generator** (written and dry-run tested, CPU only):
`/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-bf16/mk_sweep_bf16.py`
(sha256 348cbc4f…).
- It asserts the input is master BF16FIX (sha256 abd182ea…).
- Every edit is anchored and must occur exactly once.
- It appends ` + SWEEP-bf16 <NAME>: <knobs>. pp-nonce <NAME>-r<n>` to line 2 and asserts lines 1 and 3-9 are unchanged.
- Nonce files differ only in line 2. Each file must compile as Python and stay under 1 MB.
- The builder copies the generator to `campaigns/2026-10-31-public-plus/files/combos/mk_sweep_bf16.py` on its branch.

```
git -C /home/user/git/musespark show master:muse_bf16_proj/submission.py > master.py   # sha256 abd182ea...
python3 -I mk_sweep_bf16.py master.py OUT --name P1F   --p1f             --nonces 1,2,3   # V1
python3 -I mk_sweep_bf16.py master.py OUT --name P1FX  --p1f --qb3 --ol1 --nonces 1,2,3   # V4
python3 -I mk_sweep_bf16.py master.py OUT --name P1FOL --p1f --ol1       --nonces 1,2,3   # V3
python3 -I mk_sweep_bf16.py master.py OUT --name P1FQB --p1f --qb3       --nonces 1,2,3   # V2
python3 -I mk_sweep_bf16.py master.py OUT --name P1FRL --p1f --rl7       --nonces 1,2,3   # V5 (screen first)
```

Dry-run sizes (`sweep-bf16/out/`; for reference only, regenerate in the worktree): P1F 383,761 B; P1FQB 383,867;
P1FOL 383,847; P1FX 383,947; P1FRL 383,835.

### V1 P1F (rank 1): one knob, v16 unpooler

**Exact edit.** Two lines in the klarge layer, at master :4017 and :4020:
```
    ("o", 256): ("tn", _kplan(20, 6, 2)),
+   ("unpooler", 128): ("nn", _kplan(18, 4, 2)),  # [SWEEP-bf16 P1F] RBX/BFR P1 plan, static (judge 3/3 reads)
-_KLARGE_TRANSPOSED = {("unpooler", 256), ("pooler", 256), ("pooler", 128)}
+_KLARGE_TRANSPOSED = {("unpooler", 256), ("pooler", 256), ("pooler", 128), ("unpooler", 128)}
```

**Mechanism.**
- The klarge prepare loop builds the plan: AlgoCheck, tile 18 = 128x64, custom 4 = coopB, cluster id 2 = 1x1, stages 35,
  splitK 1, red 0.
- It is the same plan, built by the same `_KLargeLt.plan` call with the same (m, n, k), as RBX's `_RBX_LT_ARMS` and
  BFR's Lt arm.
- The klarge `kernel_fn` then routes exactly (unpooler, 128) through it, on `state["klarge_rawT"]["unpooler"]` =
  `state["rawT"]["unpooler"]`. That is the existing K-major copy torchT already uses, so there is no new weight copy.
- Judge kernel in profile 23: `nvjet_sm90_tst_128x64_64x8_1x1_v_bz_coopB_NNN`, replacing `..._128x64_64x8_1x2_h_bz_NNT`.
- No other layer intercepts (unpooler, 128). RBX-r2 routed it through this exact chain.
- Nearby shapes (unpooler m127/129/130, and TASK test m130) keep torchT: the plan key is the exact (layer, m).

**Prediction.**
- v16 unpooler rho 1.027 [1.022, 1.033]: paired_ratio 1.064 vs R0 1.0361; net 19.08 -> 18.57 us; floor 17.72.
- Every other row is unchanged: SASS-identical modules; the only other change is one plan created in prepare.
- Headline **+0.129%** (e_r 0.0480 x ln 1.0272).
- H100 is weaker (time ratios: RBX exprbx2 0.9974 copy-pooled; exp8 0.985 cold / 0.961 warm). The judge is the
  evidence here.

**Floors (2-check).**
- v16 unpooler rho ≥ 1.015 in **each** read (paired_ratio ≥ 1.0516).
- Untouched rows: 2-read rho ≥ R0/1.05 (hard; table §1); R0/1.03 flagged only.
- G_u (e_r-weighted geomean rho of the 29 untouched rows) ≥ 0.990.
- 2-check mean headline ≥ 1.5350.

**Compliance.**
- Same raw BF16 weights: the existing W^T copy, lossless.
- `CUBLAS_COMPUTE_32F` descriptor (68), scale R_32F, BF16 in/out, splitK 1 / reduction 0. BFR expaud1 read this exact
  plan back.
- Shape-only routing; no timing, sync or caching in kernel_fn.
- The ctypes cuBLASLt path is pre-existing in master. No new import, no subprocess, no network. Notices kept.

**Builder must assert:**
- `required_workspace` and `torch.cuda.memory_allocated()` after prepare equal master's. If the coopB plan grows the
  workspace, record it: NEG decode #11 layout risk; RBX measured equal memory.
- Plan read-back: tile 18, splitK 1, red 0, custom 4, cluster 2.

### V4 P1FX (rank 2): V1 + QB3 + OL1, all on v16

**QB3 edit** (master :4300): `_QBIG_CONFIG = (64, 36, 128, 3, 8, 539, 5)` -> `(64, 36, 128, 3, 8, 2587, 5)`.
- PIPE 2587 = 539 | 2048: 3-D tensor maps for W and x (`q_tma3` / `q_tensor_map(..., map3)`). This is BFR's id2 arm.
- Same grid (108 CTAs, cluster 3), same MMA order. exp4 rel-L2 is identical to production (1.013e-04, probe rows exact).
- Judge: BFR-r1 routed it through its race. The row read rho 1.0074, vs 1.0067 for production in BFR-r2's identical
  module, so ~0 on the judge.
- H100 (exp4; time ratio vs the same-module q0 production rebuild): cold 1.004, warm 0.931.
- Class: TMA box / L2-sector pattern, in between the classes.
- Prediction: v16 qkv rho 1.000 [0.99, 1.02].

**OL1 edit** (master :4560): `_OC_OL_CONFIG = (128, 128, 4, 3, 1, 1)` -> `(128, 64, 6, 3, 1, 1)`.
- BK 64 with 6 stages: 147.6 KB smem against the 227 KB cap. `_oc_pack_olayer` is BK-agnostic (PACK 1 reads AB = 64 x BK x
  2 B per stage from the same [64 rows][16 k-blocks] tiles), so there is no new weight copy.
- H100 (exp4 c1; time ratio vs production): cold 0.989, warm 0.982. Rel-L2 is identical to production, probe rows
  exact.
- Class: tile/stage shape on an on-SM-bound row ("o m128 8.1/8.2 warm/cold", exp4 header), so it should transfer. But
  ±1% is inside the single-module placement band, so p(real) ≈ 0.5.
- Prediction: v16 o rho 1.010 [0.99, 1.02].

**Prediction (V4):** headline +0.15% [+0.09, +0.20]. Touched rows are v16 unpooler, v16 qkv and v16 o; the other 27 rows
are unchanged. QB3 and OL1 each recompile one existing module (qbig / olayer; namespace and name change with the
source); P1F adds no compile.

**Floors:**
- V1's floors;
- v16 qkv 2-read rho ≥ 0.985 (paired_ratio ≥ 1.2041);
- v16 o 2-read rho ≥ 0.985 (paired_ratio ≥ 1.0584);
- G_u over the 27 untouched rows ≥ 0.990.

**Compliance:** as V1. QB3 and OL1 only change launch geometry and TMA addressing of existing raw-BF16 packs; the
wgmma f32 accumulation is unchanged and the outputs are bitwise identical to production in exp4.

### V3 P1FOL (rank 3) and V2 P1FQB (rank 4)
- Single-rider forms of V4. Same edits, predictions and floors restricted to their rider row.
- Send them only if quota remains after V1 and V4. They are fresh bytes at V1's level.

### V5 P1FRL (rank 5, H100-screen-gated)
**RL7 edit** (master :4561): `_OC_RL_CONFIG = (256, 64, 5, 3, 1, 1, 1)` -> `(256, 64, 5, 7, 1, 1, 1)`.
- PIPE bit 4 moves the first ST TMA issues after the CTA barrier. Bit 8 is unused in the rl source, so exp4's c6 (PIPE 15)
  is the same kernel.
- H100 (exp4 c5; time ratio): cold 0.956 but warm 1.016. The ratios conflict, and the single module had no same-module production
  rebuild (c0 failed rc 1).
- v32 o is tensor-bound on the H200 (public FACTS §8), so the warm number is the closer proxy. Prediction: v32 o rho 1.00
  [0.98, 1.04].
- **Gate to build a check file:** the §5 screen shows v32 o S ≥ 1.03 cold AND ≥ 1.000 warm, copy-pooled.
  Otherwise drop it.
- Floors: V1's, plus v32 o 2-read rho ≥ 0.985 (paired_ratio ≥ 1.0264).

## 4. Judge gate template (for the auditor; pre-data)

**Metric.** rho(row) = geomean over checks of the row's `paired_ratio` (okc status `result.records[]` event=benchmark,
field paired_ratio) ÷ R0(row) from §1. It is **not** the median of `ratios`. G_u = exp(Σ_u e_r ln rho_r / Σ_u e_r) over
the untouched rows U.

**Per variant, 2 checks r1 / r2:**
- **VALID:**
  - 9/9 tests and 30/30 rows in both reads;
  - profile 23 (v16 unpooler) shows `nvjet_sm90_tst_128x64_64x8_1x1_v_bz_coopB_NNN` in both reads;
  - for QB3, profile 20 shows `qb_kernel<64, 36, 128, 3, 8, 25…`;
  - for OL1, profile 21 shows `ol_projection<128, 64, 6, 3, …`;
  - for RL7, profile 26 shows `rl_early_projection<256, 64, 5, 7, …`.
- **GO:** all of:
  - v16 unpooler rho ≥ 1.015 in each read;
  - every rider row's 2-read rho ≥ 0.985;
  - every untouched row's 2-read rho ≥ R0/1.05;
  - G_u ≥ 0.990;
  - 2-check mean headline ≥ 1.5350.
- **KILL:** any of:
  - v16 unpooler rho < 1.005 in either read (the plan did not route, or it does not transfer);
  - any test failure;
  - G_u < 0.980;
  - a rider row's 2-read rho < 0.970: KILL that rider, and V1 stands alone.
- **Otherwise HOLD:** no third check, no post-hoc subset.
- **Fast path for V1 only** (P1 already judge-proven on 3 reads): r1 VALID + v16 unpooler rho ≥ 1.015 + G_u ≥ 0.985 +
  headline ≥ 1.530 -> scored r3.

**Scored runs.**
- On GO, 1 scored run of r3.
- A valid scored read > 1.5657 lands per the standing rule ("land anything that beats the board", 11-02 00:00Z).
- A valid V1 read ≤ 1.5657: proposed default is still to land P1F as the new bf16 master. It is a judge-proven +0.13%
  on 4+ reads, with the same precedent as B4X ("valid scored run lands") and BF16FIX. This is the user's call.
- Do not land V2-V5 below the board. Their riders add no proven gain.

**Power.**
- v16 unpooler: sd 0.34% per read against a +2.7% effect, so P(GO | works) > 0.95 and P(false GO | null) ≈ 0.
- Rider rows cannot be resolved in 2 reads (2-read sd 0.8-1.1%); their floors are no-harm guards only.

## 5. Builder: H100 screen (gpunode06 GPU 1 only) and correctness

**Harness.**
- Reuse the RBX/BFR probes: takemyjob worktree `combo-bf16_proj` `prototypes/hopper-probe/probes/lvbf/` (exprbx1-3
  layout), or lev-bf16 lvbf exp4/exp8.
- hopper-probe rules: USER, no root, no Slurm allocation, node-local `/tmp/USER-hopper`, guard, GPU 1 only, never
  change node state.

**Mandatory protocol.**
- Parties: master M, a byte-identical master copy M' (the A/A arm), P1F, P1FX, P1FOL, P1FQB, P1FRL.
- Each party gets **≥ 4 byte-identical module copies** (one prepare per process; RBX's illegal-address note forbids
  multi-prepare replays).
- Shuffled load order. Capture order alternated by copy parity; RBX's first run showed a ±2% capture-order bias.
- Copy-pooled statistics: S = t(M) / t(arm), the median of paired per-rep ratios, with a 95% bootstrap CI.
- GPU-bound graph replay, host enqueued ahead, a 256 MiB read flush before every call. Warm mode (no flush) as well, for
  OL1 and RL7.
- Rows: v16 unpooler, v16 qkv, v16 o, v32 o. Controls (SASS-identical): d1 pooler, d32 qkv, v16 pooler, v32 qkv.
- Byte-identical copies differ by up to ~1 us from placement, so judge only on pooled copies.

**Pre-registered H100 gates (written before the first timing run):**
- **VALID:** A/A |S(M') - 1| ≤ 0.3% pooled on every timed row. Repeat once if not.
- **P1F:** v16 unpooler S ≥ 0.997 pooled. This is a sanity check only (RBX read S = 1.0026, i.e. time ratio 0.9974);
  the judge is the evidence.
- **QB3 / OL1 riders:** their row S ≥ 0.995 cold, and OL1 also ≥ 0.995 warm. A rider that fails is dropped from V4 and
  its single form is not built.
- **RL7:** v32 o S ≥ 1.03 cold AND ≥ 1.000 warm, else drop V5.
- **Controls:** within ±0.5%. Flag a breach, do not reject on it; it should be impossible by construction.

**Correctness for every file** (the RBX/BFR audit set):
- TASK's 9 tests;
- 30 judge rows x 2 seeds, normal and outliers;
- on each touched cell:
  - rel-L2 vs fp64;
  - a full-K one-hot sweep bitwise vs master (the known signed-zero row is excepted, as in BFR expaud2 / RBX);
  - determinism (2 runs bitwise);
  - graph replay on fresh inputs bitwise equal to eager;
  - 2000 fresh-input graph replays with 0 gross errors;
- nearby shapes bitwise equal to master: unpooler m127/129/130, qkv m127/129, o m127/129/255/257.

**Plumbing.**
- Cold prepare with an empty extension dir ≤ master + 5 s.
- `memory_allocated` after prepare equal to master's; klarge `required_workspace` recorded with and without P1F.
- Plan read-back for (unpooler, 128).
- SASS of untouched modules identical to master. Only qbig (QB3), olayer (OL1) and rl (RL7) are rebuilt.
- okc_lint: no new warnings.

## 6. Files

All under `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-bf16/`:
- `mk_sweep_bf16.py`: the generator.
- `out/`: dry-run files P1F r1-r3, P1FQB / P1FOL / P1FX r1-r2, P1FRL r1. Regenerate in the worktree before use.
- `master.py`: master bytes, abd182ea.
- `runs/`: the 10 judge JSONs used as the reference.
- `rows.py`: headline reconstruction.
- `pool.py`: writes `ref.json`, the R0 per row.
- `table.py`: §1 table and the read probabilities.
- `calib.py`: guard false-fail rates.
- `qbig.cu`, `oc_rl_cuda.cu`, `oc_olayer_cuda.cu`: decompressed public kernels, used to read the PIPE/BK semantics.

Evidence sources:
- lev-bf16 H100 sweeps:
  - `/home/user/git/.takemyjob-wt/lev-bf16/prototypes/hopper-probe/runs/20261031-{005619,010449,011039,011934,012145,012451,012844,013026}-gpunode06`
    (micro, exp1-exp8).
- RBX: `combo-bf16_proj.md` and `audit-combo-bf16_proj.md` (scratchpad).
- BFR: attempt files 2026-10-31T10290[9/10]Z.
- GAPMAP-1031b and `research/gapmap-1031b/finder-bf16.md` (master).
- Public `NEGATIVE_RESULTS.md` / `H200_MEASURED_FACTS.md` (`/home/user/git/okc-muse-spark-kernels/docs/`).
