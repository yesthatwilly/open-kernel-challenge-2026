<!-- Research report, copied verbatim from a read-only Explore subagent (Opus) run in the planning session.
     Task: decode_layer deep read for the rewrite. Finished 2026-10-26T03:39:48.892Z. Saved by campaign 2026-10-26-decode-first-rewrite. Do not edit; add notes to ../FINDINGS.md. -->

# muse_decode_layer: research report (read-only, 2026-10-26)

Tags: **[M]** judge-measured, **[P]** prepare-time measured (CUDA events or graph timing in a run log), **[I]** my inference or estimate.

## 1. Summary

- **Board and master are not the same run.**
  - Board row 1.3164 is job `eaa8cc0f18f64bfb` (submit, slug combo-measured-rows-plus-rega-align2-large-t). It is branch code and not on master.
  - Master `ec89382b` (blob `3fc72e97`) is job `a903b9d0`, 1.3142, which is what BEST.json names.
  - On the scored shapes both run the same kernels. Two differences: the board run's rega MoE path gated itself OFF in both buckets, and its S3 c1-b16 cell fell back on a flaky gate.
- **The expert block decides the KPI.** Marlin pair plus act_and_mul plus combine is this share of each load (weighted, from the board profile): decode_1 40%, decode_16 41%, decode_32 43%, verify_1 48%, verify_16 68%, verify_32 60%.
  - Marlin's effective bandwidth on the active experts' bytes (each read once) is 1.16 to 1.92 TB/s [I, from the SoL byte decomposition].
  - It is worst at T=1 (about 2 local experts, 1.24 TB/s) and at T=128/256 (1.16 to 1.40 TB/s).
- **The non-MoE program caps around 1.36 to 1.42 [I].**
  - Stranded wins, attention/bf16/moe ports and Marlin knobs give about +3.5-4%, to about 1.36.
  - An own-kernel T≤32 PDL chain with Marlin kept reaches about 1.42.
  - Overall top-200 (decode about 1.61) also needs T≤32 experts at about 2.5 TB/s read-once (about 0.5-0.65x of today's T≤32 expert block). The model gives 1.62.
  - Top-100 (1.78) additionally needs T≥128 experts at about 0.6-0.7x; top-50 (1.93) needs about 0.5x.
- **Wins are stranded by self-checks, not by kernels.** S1, S3 and SKW cells fell back on self-imposed gates even when the new kernel was closer to fp64 than master:
  - board S1 class A: o vs master 4.02e-2 > 1e-2, while vs fp64 new is 1.22e-2 and master 3.99e-2, and the per-row pairing failed;
  - SKW: the layer-vs-master check failed with 6 of 32 routing ids differing on near-ties;
  - master's own path is nondeterministic: S3 c1-b16 passed in a903b9d0 and failed in eaa8cc0f, and S1 passed in 3117f2e4 with master's own gate.

  Fixing the gates is step zero.
- **Fresh evidence postdates DIAGNOSIS.md.** It was harvested 2026-10-26T02:31Z; DIAGNOSIS was written 2026-10-25T20:32Z.
  - moe split-I midt (6fe2a63f) lost 0.44-0.81x on both routings. This contradicts DIAGNOSIS action 2, so `fsi-moe-midt-decode-port` should be killed.
  - moe NVFP4 row-owner tiny-T (c08dbd73) lost 0.884 (8 experts) and 0.966 (4 hot) on moe's gates. Decode's T=1 has about 2 local experts, where Marlin is weakest, so a decode-routing selector could flip that verdict. Unmeasured.
  - Decode's own selector turned rega OFF at both B128 and B256 (eaa8cc0f).

## 2. Time budget per scored shape

Source: board profile `a903b9d0`. Kernel times are from the profile, which is likely warm-L2 and undercounts the judge by about 20-25% (LESSONS); bench medians are the shape times. "bnd" is the two SGLang FusedPostNormResidualPreNorm launches.

| shape | w | SGLang | ours | SoL | spd | ours/SoL | experts (Marlin/act/combine) | attention chain | dense GEMMs and routing | bnd | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d1 sliding | 46 | 171.0 | 118.5 | 40.0 | 1.443 | 2.96 | 47.7 (42.5/1.8/3.4) | 13.5 `_sliding_fused_kernel` | qkv nvjet 12.1+splitKreduce 1.8; o nvjet 7.0; P 21.5; R 6.7; unpool 6.1 | 7.2 | – |
| d1 full | 16 | 182.1 | 124.3 | 44.2 | 1.465 | 2.81 | 47.5 (42.6/1.5/3.4) | sk two-level 20.5 (incl. norm-gate) | dl_qkv 13.3 (+pre-attn); dl_o 8.2; P 21.6; R 6.3; unpool 6.1 | 7.0 | – |
| d16 sliding | 46 | 263.2 | 210.5 | 76.7 | 1.250 | 2.74 | 100.3 (94.9/2.1/3.3) | FA3 13.8 + combine 3.4 + prep 2.3, plus front/gate hidden | qkv 12.1; o+unpool (2 nvjet) 26.3; P 24.6; R 6.6 | 7.4 | 9.5 hidden |
| d16 full | 16 | 430.7 | 290.5 | 137.5 | 1.483 | 2.11 | 79.1 (73.6/2.0/3.5) | sk 110.5 + `_pre_attention` 1.7 | qkv 12.3+1.9; o+unpool 26.5; P 24.7; R 6.4 | 7.5 | bench exceeds kernel sum by about 20 |
| d32 sliding | 46 | 260.2 | 239.5 | 84.5 | 1.087 | 2.83 | 128.6 (122.2/2.3/4.1) | FA3 18.6 + 2.9 + 2.3, plus front | qkv 12.6; o+unpool 27.1; P 26.6; R 7.0 | 8.0 | 10.3 |
| d32 full | 16 | 620.6 | 446.4 | 208.1 | 1.390 | 2.15 | 119.9 (113.4/2.4/4.1) | sk 224.0, plus front/q-cast about 4.2 | qkv 12.4; o+unpool 26.5; P 26.8; R 11.2 | 8.1 | 4.8 |
| v1 sliding | 46 | 263.1 | 181.1 | 62.3 | 1.453 | 2.91 | 88.2 (85.1/–/3.1) | FA3 10.0 + 4.3 + 2.2, plus front | qkv 12.8; o 7.7; P 22.5; R 6.9; unpool 6.6 | 7.3 | 11.8 |
| v1 full | 16 | 271.3 | 189.5 | 66.4 | 1.432 | 2.85 | 88.5 | FA3 18.6 + 7.4 + 3.3 | qkv 13.7; o 7.0; P 23.8; R 6.7; unpool 6.6 | 7.3 | 11.4 |
| v16 sliding | 46 | 524.5 | 414.7 | 126.6 | 1.265 | 3.28 | 323.8 (314.8/4.1/4.9) | FA3 14.7 plus hidden | 4 dense nvjet 69.3; router hi/lo 10.7 (side stream); pre-expert norm 6.2 | 9.6 | 28.9 |
| v16 full | 16 | 751.9 | 593.4 | 188.7 | 1.267 | 3.14 | 277.9 | paired FA3 155.6 + combine 6.3; pad/unpad elementwise 11.6 | nvjet 79.1; pre-expert norm 5.9 | 4.8+ | 36.5 |
| v32 sliding | 46 | 720.9 | 599.8 | 143.8 | 1.202 | 4.17 | 408.8 (397.2/6.0/5.6) | FA3 20.2 plus hidden | nvjet 101.2 (qkv 22.8, pooler 30.8, o+unpool 34.4, router 13.2 side); align 6.5; norm 6.2 | 15.7 | 26.1 |
| v32 full | 16 | 1073.7 | 866.6 | 267.2 | 1.239 | 3.24 | 375.5 | paired FA3 288.4 + 6.7; elementwise 7.0 | nvjet 100.1; align 6.3; norm 6.3 | 15.5 | 38.0 |

Load speedups: d1 1.449, d16 1.325, **d32 1.206 (weakest)**, v1 1.447, v16 1.266, v32 1.214; geomean 1.3142.

Per-shape error is 0.7 µs at decode_1 and 2.5-7.3 µs elsewhere. Sliding decode_16 drifts 195.6-217.2 µs across master-equivalent runs, so effects under about 5% there cannot be resolved.

**Active local experts [I].** Estimated as SoL bytes minus dense minus KV, at 28.3 MB per expert: d1 about 1.9; d16 5.7 (sliding) / 4.8 (full); d32 6.8/5.2; v1 about 5.7; v16 about 14; v32 about 16. `reference.py:476-555` sets local_start to the busiest 32-expert range and picks the router seed to hit a target load within 1.5%.

## 3. Ceilings

- **SoL geomean: 3.87.**
- **Practical ceilings [I]** (model in this session):

| scenario | headline |
|---|---|
| Stranded cells only (S1 A/B) | 1.324 |
| Path A, conservative (stranded, ports, Marlin knobs) | about 1.36 |
| A + own-kernel non-MoE chain at T≤32 (R, boundaries, P prefetch), Marlin kept | about 1.42 |
| + T≤32 experts read-once at 2.5 TB/s | about 1.62 |
| + T≥128 experts at 0.85 / 0.7 / 0.5x | 1.67 / 1.74 / 1.84 |
| Experts read-once at 2.0 / 2.5 / 3.0 TB/s on every shape, plus the non-MoE program | 1.68 / 1.85 / 1.97 |
| MoE-only multiplier 0.7 / 0.6 / 0.5 (nothing else) | 1.55 / 1.64 / 1.76 |

- **Uniform scaling.** Each load time must be multiplied by 0.890 (r300 1.477), 0.807 (r200 1.628), 0.730 (r100 1.80), 0.675 (r50 1.948), or 0.557 (leader 2.36).

| shape | r300 | r200 | r100 | r50 | leader |
|---|---|---|---|---|---|
| d1 s / f | 105.4 / 110.6 | 95.6 / 100.4 | 86.5 / 90.8 | 79.9 / 83.9 | 66.0 / 69.2 |
| d16 s / f | 187.3 / 258.5 | 170.0 / 234.5 | 153.7 / 212.1 | 142.0 / 196.0 | 117.2 / 161.8 |
| d32 s / f | 213.1 / 397.2 | 193.3 / 360.4 | 174.9 / 325.9 | 161.6 / 301.2 | 133.4 / 248.6 |
| v1 s / f | 161.1 / 168.6 | 146.2 / 153.0 | 132.2 / 138.4 | 122.2 / 127.9 | 100.8 / 105.5 |
| v16 s / f | 369.0 / 528.0 | 334.8 / 479.0 | 302.8 / 433.3 | 279.8 / 400.3 | 230.9 / 330.5 |
| v32 s / f | 533.7 / 771.1 | 484.2 / 699.6 | 437.9 / 632.7 | 404.6 / 584.7 | 334.0 / 482.6 |

- **What r200 means at the verify shapes:** it needs -80 to -167 µs per v16/v32 row. Non-MoE work can supply about 10-20 µs there, so Marlin must drop to about 0.75x at T≥128.
- **What the leader's 2.36 implies [I]:** about 1.65x SoL at decode_1 (66 µs) and about 2.3x SoL at verify_32 sliding. That means near-peak dense streaming and an expert kernel at about 2.5 TB/s or better at every T.
- **T=1 persistent megakernel [I].**
  - Lower bound: 191-212 MB at 3.6-4.0 TB/s is 48-59 µs, plus exposed latency (attention, top-k, about 8 grid barriers), so about 60-75 µs with experts at about 3.5 TB/s and about 72-87 µs at 2 TB/s.
  - That would take the decode_1 load to about 2.1, worth about +6.8% headline on its own.
  - A PDL chain (R2 below) gets decode_1 rows to about 84-92 µs. The megakernel adds only about +2% beyond it, at XL+ risk.

## 4. Proven wins, dead ends, open leads

**On master [M]** (from git log of submission.py):

| landing | headline | what it did |
|---|---|---|
| router-hilo-m8 | 1.0216 | router hi/lo split |
| marlin-block-m-cap-16 | 1.0621 | T=128/256 Marlin -8 to -15% |
| fa3-hdim128-head-pairing-batched | 1.0874 | full B>1 +7 to 13% |
| post-expert-norm-triton-small-t | 1.1635 | Triton combine at T<32 |
| small-t-launch-minimal-dense-chain-pdl | 1.2283 | kernels P and R at T≤32, no side stream |
| stack-align-par-sidestream-sliding1 | 1.2495 | T>32 align on the side stream; sliding B=1 fused Triton |
| full-decode-splitkv-restack-on-pdl | 1.268 | full d16 -42.7, d32 -50.1 µs |
| w2-marlin-atomic-add-t1-port | 1.2856 | d1 -2.3 / -3.5 µs |
| dense-chain-s3 | 1.3018 | sk ticket at full decode; R variant 0 at T=1; unpool at T=1 |
| dl3-piece-a | 1.3080 | full d1 130.0 → 122.2 µs |
| combo (a903b9d0) | 1.3142 | live cells: S2a R variant 0 at T 2..32 (0.95-0.96); S2b unpool at T 2..8 (11.1 → 6.5 µs) [P]; S3 c1-b16 (flaky) |

**Dead (decode-measured).**
- Triton flash-decode on full layers: 1.63-1.66 TB/s; the oversubscribed stage-2 variant is slower still (1.34-1.37 TB/s).
- CUDA split-KV on full verify: 1.70-1.79x slower than FA3.
- mma.sync swap-AB expert chain: 670 vs 320 µs at T=128.
- Kernel-A knob sweep (kv-bw-3p5): no structural gain.
- dl-p-wide-tile: slower in its own timing.
- L2 prefetch on a side stream: d16s +21%.
- dense-chain stage 2 O kernel at T>1: 40.8 µs at T=32, 0.41 TB/s.
- Standalone dl_boundary kernel: 3.9 vs 2.8 µs for SGLang's.
- Rega in decode: B128 384.8 vs 327.2 µs, B256 454.6 vs 440.0 µs on soft routing [P, eaa8cc0f log lines 82-89].
- Flat: align/expert-id/cache-store fusions, pooler autotune.

**Dead in moe (fresh).**
- split-I midt: best 0.809 (T=16, zipf), 0.588 (random) [P, 6fe2a63f].
- wgmma tiny bucket16: 46.0 vs 45.5 µs at T=1.
- bs1 persistent: bimodal.

**Stranded (code exists, gated off by self-checks).**
- **S1 B** (Triton S=16, sliding decode B=17..32): d32s 239.5 → 231.0 in 3117f2e4 [M]. Master-equivalent runs sit at 236.4-241.9, so this is likely real. Prepare timing 43.17 vs 51.55 µs.
- **S1 A** (B=2..16): prepare 32.74 vs 44.58 [P]. Judge rows conflict: 203.7 (3117f2e4) vs 215.3 (02bf4172), within d16s drift. Four of five runs with this kernel beat their same-day base. Likely -5 to -9 µs [I].
- **SKW v1** (windowed stream-K, sliding verify_1): 38.69 vs 48.27 µs over 2 rounds, IQR ok [P]. Fell back on the layer check "stream 4.40e-2 ≤ 0.025" with 6/32 ids differing.
- **SKW b32**: 43.07 vs 51.10 µs, which ties S1 B.
- **S3 cells** c1-b32, c2-*, c3-*, c4-t8: fell back on "o vs master ≤ 1e-2" or the gap gate, while new error vs fp64 was equal to or below master's.

**Open leads.** The T=1 decode-routing expert kernel (row-owner or fsi-tiny), Marlin hot/cold blocks at T=128/256, bsm 8 at T=17..32, w13 atomic at T=1, and the own-kernel tail (R ≤ 4 µs, boundary folding).

## 5. Board vs master

| | board | master / BEST.json |
|---|---|---|
| job, mode | eaa8cc0f, submit | a903b9d0, submit |
| headline | 1.3164 [1.2966, 1.3361] | 1.3142 [1.2945, 1.3339] |
| code | master + rega module; M1 OFF both buckets; align2 not implemented (patch has 2 mentions) | ec89382b, blob 3fc72e97 |
| S3 c1-b16 | OFF (flaky gate) | ON (129.18 vs 135.14 µs) |

- The 0.2-point difference is noise: 22 identical-code pairs differed by 1.01% at most.
- BEST.json is stale relative to the board; promotion was null for eaa8cc0f.
- Re-submitting master re-rolls the S1 and S3 gates, so the board's path is not reproducible until the gates are fixed.

## 6. Path map per T band (master)

**Common to all bands.**
- qkv and o: `_gemm` → cuBLAS nvjet, plus splitKreduce on qkv. The exception is full decode_1, which runs dl3 piece A (`dl_qkv_kernel<1>` → `splitkv_attn_sk_kernel<4,true,true>` → `dl_o_kernel<1>`, CUDA mma.sync, PDL chain).
- Boundaries: SGLang FusedPostNormResidualPreNorm ×2.
- Experts: SGLang `moe_wna16_marlin_gemm` ×2 + `silu_and_mul`. block_size_m is 8 at T≤16 and 16 at T≥32 (cap 16). w2 `use_atomic_add` only at T=1.
- Combine: Triton `_post_norm_combine_kernel` at T<32; SGLang Triton weighted-sum at T≥32.

**T=1 (`submission.py:7301-7365`).**
- Sliding: `_sliding_fused_kernel`, Triton (QK-norm, RoPE, FP8 store, 16-split window attention, combine, norm-gate), config-gated (`:7216`).
- Full: piece A.
- Then P (`dl_pooler_router_kernel<1>`: pooler + router hi/lo, split-K 3, PDL weight prefetch), R variant 0 (PDL after P), Marlin, combine, `dl_unpool_kernel` NF=1 (PDL). No side stream.

**T=8 (verify_1).**
- Attention: master front (FusedQKNorm/RoPE, k/v/q FP8 casts, store_cache), then FA3 + combine + `prepare_varlen`, then FusedRMSNormGate. The S3 c2-t8/c4-t8 cells fell back.
- Then P<1>, R variant 0 (S2a), Marlin bsm 8, Triton combine, dl_unpool NF=1 (S2b).

**T=16/32.**
- Sliding: master front + FA3 chain (S1 fell back).
- Full: S3 c1-b16 `_pre_attention_kernel` (flaky) or master front, then `splitkv_attn_sk_kernel<4,true,false>` stream-K single-level ticket with norm-gate (`_sk_variant`, `:4742`).
- P<2>/P<4>, R variant 0, Marlin (bsm 8 at T=16, 16 at T=32), full-width nvjet unpooler.

**T=128/256.** The `kernel_fn` else branch (`:7331-7351`).
- Side stream: router torch.mm hi + lo, fp32 add, FusedMuseSparkRouting, `_align_parallel_kernel`.
- Main stream: pooler nvjet + F.rms_norm.
- Join, then Marlin bsm 16, then the SGLang weighted-sum combine and nvjet unpooler.
- Sliding attention: front + FA3 chain. Full attention: paired FA3 (`_master_pair_q` zeros/copies, then un-pad copies) + FusedRMSNormGate.

**prepare() selectors.**
- `_dl_setup`: PDL probe and P/R self-check.
- `_dl3_setup_bc`: pieces B and C, margin 1.02.
- `_align_par_setup`, and the w2 atomic self-check (tolerance 2e-2, no timing).
- `_splitkv_prepare` and `_sk_prepare` (per class b1/16/32, margin 1.02).
- `_dl3_finish`: piece A.
- `_combo_prepare`: S2a, S2b, S1, S3, T8; CB_RATIO 0.98, 41 cold reps; the gate is `_cb_cmp_attn` at about line 6230.

Three load_inline modules are compiled at import (dense chain s3, splitkv, sk), about 41-51 s each.

## 7. Port-lag table

Shape-level deltas; headline % from the per-load model.

| component win (source) | decode today | est µs saved / shape | headline | effort |
|---|---|---|---|---|
| S1 A/B Triton S=16 (attention master kernel; decode 3117f2e4) | stranded | d32s -8.5 [M]; d16s -5..-9 [I] | +0.5 to +0.73% | S (gate fix) |
| SKW v1 sk pseudo TOK=8 (attention 13.2 shape) | stranded | v1s -9.6 [P] | +0.66% | S (on branch) |
| S3 fused pre-attn / paired norm-gate (attention master) | c2/c3/c4/c1-b32 stranded | v16f/v32f -8/-10; v16s/v32s -3; d32f/v1f -2 | about +0.36% | S |
| per-kvhead sliding cluster (attention master 4b608df5; 13.4 → 12.1) | `_sliding_fused` 13.5 | d1s -1.3 | +0.13% | M |
| c320 + b1-chain (attention board 8713665f; V3 0.791 of two-level) | sk two-level 20.5 in piece A | d1f about -4 | +0.14% | M-L |
| bf16 bulk row-owner qkv m=1 (741072dc; 10.5 vs 13.6) | nvjet + reduce | d1s -3.1 | +0.32% | M |
| bf16 swap-AB qkv/o m≤32 (-0.4..-1.4 per GEMM) | nvjet; decode b0057d2e was flat in-gate (kernel -1..-1.5) | -1..-2 per row | about +0.4% | M |
| bf16 m=128/256 cells (cuBLASLt qkv plan, pooler K-halves, Triton hilo router) | nvjet, torch.mm router | v16 -5, v32 -2 | about +0.2% | M |
| bf16 ouns unpool slice | dl_unpool is already equal | 0 | 0 | – |
| moe w13 atomic at T=1 (moe master F1: 0.943 zipf, 0.750 random) | w2 atomic only | d1 -1.5 (up to -5) | +0.2 to +0.6% | S (queued cell B) |
| moe two-stream expert halves T 8..32 (0.968-0.980; not on moe master) | single stream | d16/d32/v1 -2..-3 | about +0.55% | S-M |
| moe hoisted post-norm T=32 (3.62 → 2.98) | SGLang weighted sum | d32 -0.5 | +0.03% | S |
| moe wg2 occ2 / align2 at B256 | Marlin; align on side stream | about 0 (decode rega 1.033 soft) | about 0 | M |
| moe nvfp4 row-owner M (lost in moe) | Marlin 1.24 TB/s at T=1 | d1 -5..-15 [I] | +0.7 to +2.1% | M |

Two notes on the table:
- The full port lag (stranded plus ports plus knobs) is about +3.5-4.5%, putting decode around 1.36-1.37.
- bf16 kernels lose about 3 µs in context compared with their standalone profile (swapab qkv 12.7-13.0 vs 9.6 µs). Most of that is warm-L2 profile versus cold bench, so compare shape times, not kernel times.

## 8. Unleveraged areas

1. **Judge-faithful, deterministic gates.** Compare against an fp64 reference that uses the candidate's own routing ids, as the judge does, not against master. Bound per row by the aggregate error.
2. **Decode-specific expert routing** (about 2 local experts at T=1, 5-7 at T=8..32). moe verdicts do not transfer; rega flipped from a moe win to a decode loss. No decode selector has ever timed an alternative to Marlin at T≤32.
3. **Glue at T>32.** The bench exceeds the kernel sum by +15 to +20 µs on d16f, v16f and v32s, and 26-38 µs of small hidden kernels sit at v16/v32 (casts, pad/unpad, splitKreduce, routing, FusedQKNorm). The join slack between the side-stream router and the pooler has never been measured.
4. **Weight prefetch before `griddepcontrol.wait`.** Static weights (o, pooler, router, unpooler) can be fetched during latency-bound phases: attention, R, boundaries, combine. This is idle-DRAM time of about 30 µs at T=1.
5. **Boundary folding** into the O and U epilogues: 7-8 µs per row at T≤32, 9.6-15.7 at T≥128.
6. **The T=32 expert block uses bsm 16**, the same as SGLang, so d32s gets no MoE advantage at all (1.087x shape).

## 9. Counterinteractions (evidence)

- **PDL only pays between own kernels.**
  - P after SGLang's boundary kernel gained 0.3 µs (81ee6714 log).
  - The own-kernel chain Q → sk → O: 57.25 → 48.96 µs [P], and full d1 -7.8 µs [M].
  - Attention's b1-chain reached 0.791 of master, against 0.892 and 0.883 for its pieces.
  - Marlin and other SGLang kernels cannot be launched with PDL (no module edits), so they break any chain.
- **bf16 Triton split-K at T=8/16.** Under the old side-stream router it lost +20 µs at d16s and +15 at v1s (6076ae1f). On the no-fork master the swap-AB version was flat in-gate (b0057d2e). The side-stream interference was likely real and is now gone at T≤32.
- **Side-stream contention.** The L2-prefetch side stream regressed d16s to 254.5 µs. The T>32 router branch (about 25 µs) races the pooler (about 30 µs) for SMs.
- **Kernel-vs-kernel comparisons mislead.** S1's 14.6 µs vs FA3's 13.7 µs got the line declared dead, but the whole chain (FA3 + combine + prepare + front + gate) is 25-29 µs.
- **Selector placement matters.** Prepare-time selectors that time the whole routed path predict the judge; kernel-only rankings do not. Prepare CUDA-event timings undercount the judge by 20-25%.
- **Marlin profile rows swing ±20-40 µs** between runs on identical code (d16s 94.9 vs 105.0; v16s 314.8 vs 285.7).
- **Two state copies rotate.** Any module-global device pointer breaks (the moe g_maps bug).

## 10. ROI paths, ranked

| # | path | targets | current → target µs | headline | effort | risk | evidence | kill criterion |
|---|---|---|---|---|---|---|---|---|
| 1 | Deterministic fp64 gates; restack S1 A/B + SKW (v1 and b32 race) + S3 cells, from branch `hyp/muse_decode_layer/sliding-sk-window-port-decode32-verify1` (a48a10b3) | d16s, d32s, v1s, v16/v32 full, d32f | 210.5→~203; 239.5→~231; 181.1→~172; v16f/v32f -8/-10 | +1.5 to +2.0% | S-M | low | a903b9d0 log lines 58-71; 3117f2e4 log lines 52-84 | a check with all cells on shows d32s > 234 or v1s > 177: drop that cell |
| 2 | Decode-routing expert kernel at T=1: port moe row-owner M (branch `hyp/muse_moe_w4a16/nvfp4-rowowner-bulk-stream-tiny-t-rewrite`) and/or fsi-tiny, plus the w13-atomic cell | d1 s/f | expert block 47.7 → 32-42 | +0.7 to +2.1% | M | medium | Marlin 1.24 TB/s on about 53 MB; row-owner 2.12/1.71 TB/s kernel-only (c08dbd73) | worse-of-two routings ≥ 0.98 of Marlin, or kernel-only < 1.8 TB/s |
| 3 | Marlin knob cells: bsm 8 at T=17..32; two-stream halves T 8..32; hot/cold two-class bsm at T=128/256 (queued) | d32, d16, v1, v16, v32 | d32 -5..-10; v16/v32 -15..-60 | +1 to +2.5% | S-M | medium | cap-16 gained 8-15%; moe bsm 8 at T=128 lost 391 vs 341 (re-streaming) | selector picks MASTER in every class |
| 4 | T=1 attention/dense ports: per-kvhead cluster, c320 inside piece A, bf16 bulk qkv on the sliding T=1 path | d1 | 118.5→~114; 124.3→~120 | +0.6% | M | low-medium | 4b608df5, 8713665f, 741072dc | prepare selector below 1.02 |
| 5 | Own-kernel tail at T≤32: R ≤ 4 µs, attn_boundary folded into O, ffn_boundary into U, O scaling to T=32, pre-wait weight prefetch | all T≤32 | -6..-12 per row | +1 to +2% | L | medium | dl_boundary 3.9 vs 2.8; dl_o 40.8 µs at T=32 | piece selector fails |
| 6 | Own NVFP4 expert kernel for T=2..32 (row-owner with tokens on N, N=8/16/32) | d16, d32, v1 | 85-128 → 60-90 | +2 to +4% | L-XL | high | fsi-midt 0.47-0.81 in moe, but moe has heavier routing | kernel-only < 2.3 TB/s on decode routing |
| 7 | T>32 glue: fused pooler+router concat GEMM + routing; paired S3; cuBLASLt plans | v16, v32 | -10..-30 | +0.5 to +1% | M-L | medium | 26-38 µs hidden small kernels | join-slack measurement shows the router branch is off the critical path |
| 8 | **XL rewrite: own-kernel T≤32 PDL layer chain** (2+4+5+6 as one chain) | 4 of 6 loads | T≤32 rows to about 0.65-0.75x | to about 1.55-1.63 | XL | high | dl3 chain, attention b1-chain | expert stage below Marlin speed |
| 9 | T=1 megakernel | d1 | to 70-85 µs | about +2% over #8 | XL+ | high | none in repo | – |

Do not pursue: full-layer attention rewrites (sk is at 1.75x of its KV floor; FA3 full verify at 2.5x; every attention-problem attempt there failed), rega/wg2 at B128, and `fsi-moe-midt-decode-port`.

**Queue verdicts (18 items).**
- Kill: `moe-small-t-cuda-swapab-port`, `moe-bs1-fused-persistent-port`, `moe-chain-cuda-swapab-expert-once-fix`, `dl3-unpool-nf1-t2-t8` (already on master), `moe-rega-bucket-gate-port-t128-t512`, `combo-attn-master-sliding-race-skw-c16`, `fsi-moe-midt-decode-port` (fresh moe loss).
- Fold into path 1: `preattn-fused-t-gt1-sk-and-fa3-feeds`.
- Fold into the bf16 bundle: `bf16-m32-swapab-wgmma-port`, `swapab16-t2-t16-port-on-master`, `dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32`.
- Keep: `fsi-moe-tiny-t-decode-port` (run its kernel-only DIAG first), `marlin-bsm8-t17-t32-w13-atomic-t1`, `marlin-hot-cold-two-class-bsm-t128-t256`, `ffn-front-t128-t256-concat-route` (low priority), `sliding-cluster-c16-decode1-port` (switch to the per-kvhead variant).
- Deprioritize: `sliding-cluster-c4-s3-satfinite-on-sk-ticket` (S1 B / sk-window cover that band), `wg2-occ2-b256-decode-port` (projected about 0.99 on soft routing).

## 11. Rewrite candidates

- **R1, days 1-2 (S-M): gate fix plus restack.**
  - Build a checker that recomputes an fp64 reference with the candidate's own topk ids, using bounds derived from TASK.md tolerances, with no per-row pairing against master.
  - Restack paths 1, 3 and 4. Keep per-cell 2-round ≥1.03 cold-L2 selectors and assert each printed winner.
  - Target about 1.35-1.37, and a reproducible master path.
- **R2, days 2-9 (XL): own-kernel T≤32 chain.**
  - Stage 1: an expert kernel pair (w13+silu → h; w2 + post-expert combine via a per-token ticket). Use a cp.async.bulk/mbarrier ring, Marlin's dequant8 trick, and mma.sync tokens-on-N. Gate it at T=1 first, then T≤8, then T≤32, against Marlin on judge-like and zipf routings, and print kernel-only TB/s.
  - Stage 2: fold R (parallel align) and the boundaries into neighbouring kernels; add an O kernel that scales to T=32.
  - Stage 3: every link PDL, with static-weight prefetch before the wait. Expert weights cannot be prefetched before R completes (they depend on routing).
  - Kill after stage 1 if T=1 is below 1.8 TB/s or T=8 cannot beat 0.98x.
  - Target about 1.55-1.63.
- **R3, optional: T≥128 expert block.** Hot/cold Marlin first. Build a read-once kernel only if R2's kernel generalizes to N 64-128; otherwise stop at the knobs.

Budget and constraints:
- Run limits are 20 checks and 10 scored submits per problem per day. Make every new CUDA module's first run an `okc check`.
- Consolidate translation units: there are 3 import builds of about 45 s each today, and the 15-minute run limit includes compile time.
- Keep all buffers per state.

## 12. Key files and open questions

**Files.**
- `/home/user/git/musespark/muse_decode_layer/submission.py`: `kernel_fn` 7301, `prepare` 6932, `_attn_core` 7207, `_experts` 7094, `_dl_forward` 2601, `_block_size_m` 2413, S1 3220-3266, S3 3268-3445, `_splitkv_gate` 3887, `_sk_variant` 4742, combo constants 5974-6030, `_cb_cmp_attn` about 6230, `_cb_cmp_layer` about 6499.
- `/home/user/git/musespark/muse_decode_layer/hypotheses/archive/combo-measured-rows-slide-batched-preattn-t8-tail/a903b9d004d2452d.profile.txt` and `.log`
- `/home/user/git/musespark/muse_decode_layer/hypotheses/archive/combo-measured-rows-plus-rega-align2-large-t/eaa8cc0f18f64bfb.log`
- `/home/user/git/musespark/muse_decode_layer/hypotheses/archive/sliding-sk-window-port-decode32-verify1/3117f2e423d44cb7.log` and `.profile.txt`
- `/home/user/git/musespark/muse_moe_w4a16/hypotheses/archive/nvfp4-rowowner-bulk-stream-tiny-t-rewrite/c08dbd737e29442d.log`
- `/home/user/git/musespark/muse_moe_w4a16/hypotheses/archive/cuda-fused-split-i-moe-midt-cluster-reduce/6fe2a63f490f49a3.log`
- `/home/user/git/musespark/muse_decode_layer/reference.py` lines 476-555
- `/home/user/git/musespark/muse_decode_layer/hypotheses/DIAGNOSIS.md` (action 2 is now contradicted)
- `/home/user/git/musespark/hypotheses/LESSONS.md`

**Open questions (could not verify).**
1. Why SKW b32 only ties S1 B in decode (43.07 vs 43.17) when attention measured sk 14.0 vs Triton 25.8. The race lines were cut from the 16 KB log tail.
2. True active-expert counts per shape. Mine are inferred from SoL; the judge's router_load targets are not visible.
3. Whether Marlin re-streams expert weights per m-block at T=128/256, or L2 absorbs it. This decides the value of hot/cold.
4. Why decode's d1 Marlin pair (42.5 µs) is slower than moe's (35.4 µs) at similar load.
5. What the +15 to +20 µs bench-over-kernel-sum on d16f/v16f/v32s is: launch gaps or cold-L2 effects.
6. The judge-level effect of S1 A at d16s, which is inside that shape's ±5% drift.
