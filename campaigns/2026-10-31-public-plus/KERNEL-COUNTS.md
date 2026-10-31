# Kernel counts: why we launch more kernels, and what transfers from the public files

Written 2026-10-31 ~03:45Z, read-only. Sources: the judge profiles inside the okc status records of our BEST jobs and of
the public controls (counted with the scratch `kcount.py`; the per-row appendix is below), and the CLAIMS lines cited.
Public-exposed document (see PLAN §5).

## Launches per row

| problem | ours | public | what the extra launches are |
|---|---|---|---|
| moe | 3 per row, 7 at v32 | 1 (`okc::moe_fused`) | w13 kernel, w2 kernel, combine; at v32 also wg<false>/wg<true> split, ms_align, wg_gather |
| decode d1s | 13 | ~8 (all PDL) | FP32 router as cuBLAS dot + reduce (+2, DC0b, decode CLAIMS:109); moe K1/K2/combine (+2); a second SGLang norm_residual boundary where they fold one into gle_o (+1) |
| attention | sliding d1/d16/d32/v1 1; full rows 2; sliding v16/v32 3 | 1 on every row | prep/combine launches on 8 rows |
| prefill t1 | ~23 (inventory; the profile shows a truncated 12 kernels / 14 launches) | ~14 | SGLang glue and a separate routing chain |
| bf16 | 1 (2 on router d16/d32/v16/v32 and pooler v16) | 1 | **not a kernel-count gap**; per-kernel efficiency (their 12-bit consumer) |

**Cost of the extras (judge profiles):**
- moe: combine 2.9–3.0 µs at d1/d16, 12.7 µs at v16, 14.1 µs at v32. At v32, ms_align 4.3 + wg_gather 3.1. The w13→w2
  boundary costs ~5.5 µs of hand-off and refill (moe CLAIMS:72).
- decode: at T ≤ 8 the layer is one serial chain. Each glue launch is 1.2–2.0 µs (their H200_MEASURED_FACTS §2;
  all three of theirs 4.0–4.8 µs). Our 5 extra launches at d1s ≈ 6–10 µs of the ~26 µs row gap.
- moe v16 is not only launch count: per-route re-dequantisation of the same weights carries most of the rest of the
  39 µs row gap; their jobs dequantise once for up to 64 tokens.

**Gap shares (moe, raw µs, ours BEST vs today's public mean):** v16 32.0%, v32 29.7%, v1 17.3%, d32 11.4%, d16 8.9%,
d1 8.1%. v16 + v32 = 62% of the moe gap.

## Why we launch more

1. **Kernel boundaries are our phase sync.** Every in-kernel sync we tried was an expensive form, so fusion read as
   refuted:
   - arch-mirror L3 skeleton: cooperative launch (~5 µs) + 2 grid barriers = 17–19 µs empty, against their whole
     24.6 µs d1 row (arch-mirror CLAIMS:135);
   - combine fused into w2 as a single last-arriver CTA: slower by 4.7 / 14.1 / 15.5 µs (moe CLAIMS:102);
   - EPF one-launch inside our 2-CTA/SM engine: x1.09 slower (moe CLAIMS:110);
   - UFS readiness waits: +6..12 µs (moe CLAIMS:124).
2. **Accretion.** One mechanism per hypothesis usually added a gated route with a race and a fallback next to the old
   one. Attention 694 KB / 23 kernels vs 155 KB / 1 template; moe 982 KB vs 326 KB.
3. **CTA shape.** Our 2-CTA/SM engine at a 96–112 register cap leaves no room to stage a second phase.
4. **Parity and compliance kernels.** SGLang norm/routing kernels kept for exactness; the DC0b FP32 router became a
   separate 2-launch GEMV (decode CLAIMS:109).
5. **Decode byte cap** (3.3 KB free) blocks consolidation without a trim.

## What transfers (primitives, with our measured costs)

| primitive | replaces | cost (ours, R8 unless noted) |
|---|---|---|
| per-job ready counters inside one launch (release/acquire, batched per job, not per tile) | the w13→w2 kernel boundary (~5.5 µs + a launch) | publish + acquire ~0.9 µs; fence ~0.13 µs per issue |
| plain launch, grid = SM count, `__launch_bounds__(N,1)`, at most one grid barrier | cooperative launch + 2 barriers | barrier 1.76 µs; cooperative launch ~5 µs avoided |
| distributed segmented final pass by all CTAs after the barrier | combine kernel or one last-arriver CTA | removes 2.9–14.1 µs per moe row |
| routing recomputed redundantly per CTA at small T; first weight loads issued before prep (FAST1, our S1) | route/align launches | 0 launches |
| deferred-norm fold into the o epilogue; FP32 router as stages inside the pooler kernel | 1.2–2.0 µs per glue launch at T ≤ 8 | also recovers the DC0 compliance cost (~1.6% decode headline) |
| 1 CTA/SM warp-specialised shape with dedicated input slots | our 2-CTA/SM engine | the reason EPF/UFS lost |

**Gate for any fusion:** build the empty skeleton in their sync form first (plain launch, ≤ 1 barrier, counters,
distributed final pass) and require its fixed cost ≤ 2–4 µs before porting consume loops. Port as a bundle with
ablation arms; judge memory-bound rows on the judge, not the H100.

**Where it pays:** moe (v16/v32 bundle: one launch, counters, distributed final, dequantise-once wide jobs, 1 CTA/SM)
first; decode (router into pooler, combine, one boundary) second; attention full and v16s/v32s rows to one launch
(≤ ~2.9%); prefill launch diet (~17–21% of its gap). Not bf16.

## Appendix: launches per row (judge profiles)

### attention-ours `4db5f1395ced4b34`
```
sliding decode 1                   row     6.43 us  kernels  1  launches   1
full decode 1                      row    15.08 us  kernels  2  launches   2
sliding decode 16                  row    10.71 us  kernels  1  launches   1
full decode 16                     row    96.00 us  kernels  2  launches   2
sliding decode 32                  row    14.81 us  kernels  1  launches   1
full decode 32                     row   175.40 us  kernels  2  launches   2
sliding verify 1 8                 row     9.41 us  kernels  1  launches   1
full verify 1 8                    row    23.67 us  kernels  2  launches   2
sliding verify 16 8                row    19.57 us  kernels  3  launches   3
full verify 16 8                   row   131.43 us  kernels  2  launches   2
sliding verify 32 8                row    25.24 us  kernels  3  launches   3
full verify 32 8                   row   240.72 us  kernels  2  launches   2
```
### moe-ours `476d3a0e9dc74d0e`
```
decode 1 1                         row    26.92 us  kernels  3  launches   3
decode 16 16                       row    99.92 us  kernels  3  launches   3
decode 32 32                       row   134.17 us  kernels  3  launches   3
verify 1 8                         row    73.03 us  kernels  3  launches   3
verify 16 128                      row   213.97 us  kernels  3  launches   3
verify 32 256                      row   253.13 us  kernels  7  launches   7
```
### moe-public `b3510dcc6553442b`
```
decode 1 1                         row    26.05 us  kernels  1  launches   1
decode 16 16                       row    91.41 us  kernels  1  launches   1
decode 32 32                       row   125.21 us  kernels  1  launches   1
verify 1 8                         row    67.45 us  kernels  1  launches   1
verify 16 128                      row   174.98 us  kernels  1  launches   1
verify 32 256                      row   212.55 us  kernels  1  launches   1
```
### bf16-ours `7753d5c7031a4e72`
```
qkv_gate decode 1 1                row    10.53 us  kernels  1  launches   1
o decode 1 1                       row     5.32 us  kernels  1  launches   1
pooler decode 1 1                  row    14.62 us  kernels  1  launches   1
unpooler decode 1 1                row     4.71 us  kernels  1  launches   1
router decode 1 1                  row     3.71 us  kernels  1  launches   1
qkv_gate decode 16 16              row    12.20 us  kernels  1  launches   1
o decode 16 16                     row     6.14 us  kernels  1  launches   1
pooler decode 16 16                row    17.97 us  kernels  1  launches   1
unpooler decode 16 16              row    17.35 us  kernels  1  launches   1
router decode 16 16                row     5.48 us  kernels  2  launches   2
qkv_gate decode 32 32              row    11.97 us  kernels  1  launches   1
o decode 32 32                     row     6.46 us  kernels  1  launches   1
pooler decode 32 32                row    18.20 us  kernels  1  launches   1
unpooler decode 32 32              row    18.40 us  kernels  1  launches   1
router decode 32 32                row     5.67 us  kernels  2  launches   2
qkv_gate verify 1 8                row    11.65 us  kernels  1  launches   1
o verify 1 8                       row     6.50 us  kernels  1  launches   1
pooler verify 1 8                  row    15.99 us  kernels  1  launches   1
unpooler verify 1 8                row     5.24 us  kernels  1  launches   1
router verify 1 8                  row     4.88 us  kernels  1  launches   1
qkv_gate verify 16 128             row    16.18 us  kernels  1  launches   1
o verify 16 128                    row     7.41 us  kernels  1  launches   1
pooler verify 16 128               row    21.48 us  kernels  2  launches   2
unpooler verify 16 128             row    19.09 us  kernels  1  launches   1
router verify 16 128               row     7.10 us  kernels  2  launches   2
qkv_gate verify 32 256             row    19.27 us  kernels  1  launches   1
o verify 32 256                    row     9.28 us  kernels  1  launches   1
pooler verify 32 256               row    27.34 us  kernels  1  launches   1
unpooler verify 32 256             row    24.05 us  kernels  1  launches   1
router verify 32 256               row     9.04 us  kernels  2  launches   2
```
### bf16-public `a95479284a204d87`
```
qkv_gate decode 1 1                row     8.66 us  kernels  1  launches   1
o decode 1 1                       row     5.00 us  kernels  1  launches   1
pooler decode 1 1                  row    13.21 us  kernels  1  launches   1
unpooler decode 1 1                row     3.47 us  kernels  1  launches   1
router decode 1 1                  row     3.66 us  kernels  1  launches   1
qkv_gate decode 16 16              row     9.82 us  kernels  1  launches   1
o decode 16 16                     row     5.81 us  kernels  1  launches   1
pooler decode 16 16                row    13.77 us  kernels  1  launches   1
unpooler decode 16 16              row    13.77 us  kernels  1  launches   1
router decode 16 16                row     4.69 us  kernels  1  launches   1
qkv_gate decode 32 32              row    10.80 us  kernels  1  launches   1
o decode 32 32                     row     6.27 us  kernels  1  launches   1
pooler decode 32 32                row    15.24 us  kernels  1  launches   1
unpooler decode 32 32              row    14.62 us  kernels  1  launches   1
router decode 32 32                row     5.63 us  kernels  1  launches   1
qkv_gate verify 1 8                row     9.19 us  kernels  1  launches   1
o verify 1 8                       row     5.38 us  kernels  1  launches   1
pooler verify 1 8                  row    13.56 us  kernels  1  launches   1
unpooler verify 1 8                row     4.22 us  kernels  1  launches   1
router verify 1 8                  row     4.66 us  kernels  1  launches   1
qkv_gate verify 16 128             row    14.11 us  kernels  1  launches   1
o verify 16 128                    row     6.58 us  kernels  1  launches   1
pooler verify 16 128               row    21.16 us  kernels  1  launches   1
unpooler verify 16 128             row    18.99 us  kernels  1  launches   1
router verify 16 128               row     5.75 us  kernels  1  launches   1
qkv_gate verify 32 256             row    17.61 us  kernels  1  launches   1
o verify 32 256                    row     8.82 us  kernels  1  launches   1
pooler verify 32 256               row    24.66 us  kernels  1  launches   1
unpooler verify 32 256             row    23.37 us  kernels  1  launches   1
router verify 32 256               row     7.09 us  kernels  1  launches   1
```
### decode-ours `dfb318ee06b842f7`
```
sliding decode 1                   row    95.55 us  kernels 12  launches  13
full decode 1                      row   102.70 us  kernels 12  launches  12
sliding decode 16                  row   164.87 us  kernels 10  launches  11
full decode 16                     row   253.63 us  kernels 11  launches  12
sliding decode 32                  row   190.53 us  kernels 10  launches  12
full decode 32                     row   360.37 us  kernels 10  launches  12
sliding verify 1 8                 row   149.24 us  kernels 12  launches  13
full verify 1 8                    row   162.77 us  kernels 12  launches  13
sliding verify 16 8                row   312.78 us  kernels 12  launches  12
full verify 16 8                   row   416.71 us  kernels 12  launches  12
sliding verify 32 8                row   401.36 us  kernels 12  launches  13
full verify 32 8                   row   615.34 us  kernels 12  launches  13
```
### prefill-ours `9f93484606cd4f9e`
```
sliding turn 1 753                 row   709.59 us  kernels 12  launches  14
full turn 1 753                    row  1065.21 us  kernels 12  launches  14
sliding turn 4 753                 row  1987.58 us  kernels 12  launches  13
full turn 4 753                    row  3534.54 us  kernels 12  launches  13
full first turn 16384              row 10596.78 us  kernels 12  launches  14
sliding first turn 16384           row 10288.80 us  kernels 12  launches  14
full first turn 16384              row 14315.00 us  kernels 12  launches  14
full first turn 16384              row 17224.81 us  kernels 12  launches  14
```
