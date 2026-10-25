# Rewrite read: muse_attention_decode, kind:rewrite runs of 2026-10-24

Each rewrite is judged on the time of the kernels it **replaces**, per target shape, against the
incumbent kernels' time and the speed-of-light floor (the `sol` column), not on the headline.
All numbers are profile `top_kernels` GPU time in µs, from the recorded runs. No GPU was used for this read.

**Incumbent sources**
- Full layer: `sliding-fused-triton-decode1-gate` job 295727fc68e14093 (FA3 path, the base of
  stage 1) and `full-decode-cuda-gather-splitkv` job 82085a4e82154393 (splitkv path, the base of
  the three later rewrites).
- Sliding layer: 82085a4e82154393 for the base, and current best/master
  `sliding-deep-splits-on-cuda-splitkv` job 698d4cd3838c4a63. Master now runs the Triton
  `_sliding_fused_kernel` on decode_1/16/32 and FA3 on verify.
- `sliding-single-launch-fused-attention`: base `fused-preattention-triton` job dfee6f2939974bad.

"Replaced" means the kernels the rewrite takes out. `_pre_attention_kernel` is kept by the two
splitkv rewrites, so it is excluded from both columns for those.

---

## 1. full-decode-cuda-gather-splitkv (gain, now on master): PROMISING (foundation, stage 2 = tune)

**Path (log 82085a4e82154393 and de486093311d4bb9):** `[splitkv] load_inline compiled in 41.0s`,
and three self-checks with rel L2 of about 1.6-1.7e-3, all finite. There was no fallback. The
profile shows `splitkv_attn_kernel` + `splitkv_combine_kernel` on full decode_16/32 only. Every
other shape keeps its master kernels, as gated.

| shape (full) | incumbent replaced (FA3 main+combine+prepare+paired norm-gate) | rewrite A + B | vs incumbent | floor | x floor | A achieved BW |
|---|---|---|---|---|---|---|
| decode_16 | 150.4+4.4+3.0+1.5 = **159.3** | 111.4+7.9 = **119.3** | 1.34x faster | 63.2 | 1.89x | ~2.94 TB/s |
| decode_32 | 292.0+4.8+3.4+1.6 = **301.8** | 245.5+8.6 = **254.1** | 1.19x faster | 126.4 | 2.01x | ~2.67 TB/s |

**Milestones:** M1 is met (16/16, faster on both shapes). The spec's M2 (A <= 110 / 215 µs) is
missed: barely at decode_16 (111.4) and clearly at decode_32 (245.5). M3 (<= 1.5x floor) is not met.

**Where the time goes:** kernel A is 92-96% of each shape. The combine takes 7.9-8.6 µs, about 2x
FA3's 4.4-4.8 µs combine. Bandwidth falls from 2.94 to 2.67 TB/s between B=16 and B=32 at the same
per-CTA work. That drop points to DRAM/TLB locality or the 1024-token chunk tail (about 9.5 items
per CTA at B=32), not to issue rate.

**Verdict: PROMISING.** It is already the incumbent. Stage 2 should stay on this kernel and
change one thing at a time:
1. Fold the combine and norm-gate into kernel A with a per-request ticket (about -8 µs per
   shape, 6.5% at decode_16).
2. Switch to stream-K over 64-token tiles to remove the chunk tail.
3. Recover the decode_32 bandwidth by testing C/stages and L2-friendly request ordering.

Rewrite 5 showed that the 2-CTA/SM all-thread cp.async design is the one to keep.

---

## 2. full-verify-cuda-streamk-splitkv (regression 1.231x vs 1.349x base): PROMISING, needs a compute restructure

**Path (log a43acd1ed9b24b4e):** `[splitkv-verify] load_inline compiled in 43.6s`. A8 uses 128 regs
with 0 local bytes and 90,640 B of smem; B8 uses 35 regs. Four verify self-checks passed (rel L2
of about 1.65e-3, including grids 3/7/61). There was no fallback. The profile shows
`splitkv_verify_kernel` + `splitkv_verify_combine_kernel` on full verify_16/32, and the decode
path is unchanged (A 111.3 / 246.7).

| shape (full) | incumbent replaced (FA3 main+combine+prepare+paired norm-gate) | rewrite A8 + B8 | vs incumbent | floor | x floor | A8 BW |
|---|---|---|---|---|---|---|
| verify_16 | 150.7+6.4+3.1+2.5 = **162.7** | 227.8+13.9 = **241.7** | 1.49x slower | 63.4 | 3.81x | ~1.44 TB/s |
| verify_32 | 287.4+6.7+3.4+3.8 = **301.3** | 447.7+17.8 = **465.5** | 1.55x slower | 126.8 | 3.67x | ~1.46 TB/s |

**Milestones:** M1 correctness is met (16/16). M2 is missed: A8 is 2.04x / 1.82x stage-1 decode
A on the same bytes, against a target of <= 1.15x. M3 is missed.

**Where the time goes:** A8 is 93-95%. Time scales linearly with bytes (447.7 / 227.8 = 1.97x for
2x the KV), so the problem is not tail imbalance or a latency floor. The kernel is throughput-bound
in the per-tile loop at half the gather rate of stage 1, so the 64-rows-per-kv-head math is not
hiding under the gather. 128 regs is exactly the `(256, 2)` cap. The likely serializers are the
cooperative V widen and the extra CTA-wide `__syncthreads` per tile, plus each warp doing the whole
64-token tile: 48 mma plus softmax over 16x64. B8 is 13.9-17.8 µs, about 2x FA3 combine +
norm-gate, because it reads (grid+B) slots x 8x16x64 fp32.

The headline row "full decode_32 gained 1.14x" (257.75 -> 226.29 µs) is not supported by this
run's profile, where A is 246.7 µs, the same as base. It is a scoring artifact; do not credit it.

**Verdict: PROMISING.** The gather machinery is proven, and FA3 itself does verify at about
2.2 TB/s, so the math can hide. The fix is structural and specific. Stage 3 must:
1. Restructure the verify compute so it overlaps the gather. Each kv head's 64 rows (8 heads x 8
   tokens) map exactly onto a wgmma M=64 tile. Use one consumer warpgroup per kv head with
   `wgmma m64nNk32` e4m3 for QK, and either fp8 PV on a once-transposed V tile or f16 PV via
   wgmma from smem, so there is no per-tile CTA barrier. Or, at minimum, drop the cooperative
   V-widen + `__syncthreads` per tile and split tiles across warps.
2. Shrink B8 by moving to an in-kernel ticket combine, or by cutting the slot count.

Gate: A8 <= 1.15x stage-1 decode A (<= 128 / 282 µs).

---

## 3. sliding-cuda-cluster-dsmem-onelaunch (regression 1.221x): PROMISING for decode only; drop L=8 verify

**Path (log 6821e67ff3e749b7):** `[slide] load_inline (fp8qk) compiled in 45.4s`. L=1 uses 104 regs
and L=8 uses 98, each with 32 B of local memory (likely the full-precision `sincosf` stack). Max
active clusters are C=8 -> 30 and C=16 -> 14. All 8 self-checks pass (rel L2 about 1.6e-3, cache
rows match 1.0000), and graph replay matches. PATH CHOSEN is the CLUSTER kernel for q_len 1 and 8,
with C=16 at B=1 and C=8 at B=2..64. Page-64 tests stay on master Triton/FA3. There was no
fallback. The profile shows `sliding_cluster_kernel<L>` as the only kernel on every sliding shape.

| shape (sliding) | base incumbent (82085a4e) kernels | current master (698d4cd3) kernels | cluster kernel | vs base | vs master | floor |
|---|---|---|---|---|---|---|
| decode_1 | 10.3 (Triton fused) | 10.4 | **10.0** | 1.03x | 1.04x | 0.11 |
| decode_16 | 17.9 (5 kernels, busy 16.5) | 12.9 (Triton fused) | **13.0** | 1.38x | 0.99x | 1.65 |
| decode_32 | 21.3 (busy 19.9) | 22.1 (Triton fused) | **22.2** | 0.96x | 1.00x | 3.29 |
| verify_1 | 15.7 (busy 14.4) | 15.9 (FA3, busy 14.5) | **23.2** | 0.68x | 0.69x | 0.12 |
| verify_16 | 20.7 (busy 19.3) | 20.7 (busy 19.2) | **36.6** | 0.57x | 0.57x | 1.86 |
| verify_32 | 26.4 (busy 24.7) | 26.4 (busy 25.0) | **55.5** | 0.48x | 0.48x | 3.72 |

The attribution against the base was decode_1 1.17x, decode_16 1.36x, verify 0.45-0.62x.

**Where the time goes:**
- **decode_32:** B=32 x C=8 = 32 clusters, but only 30 clusters can be co-resident. The 256 CTAs
  run as two waves, which explains why decode_32 does not scale from decode_16 (13.0 -> 22.2 µs
  for 2x bytes, against a 3.3 µs floor).
- **verify (L=8):** 13 µs over L=1 even at B=1, where each CTA holds only ~128 keys. The extra
  time is a fixed serialized chain, not bandwidth. Candidates: the 128-row q norm/RoPE/fp8
  prologue, the per-tile V widen, the last-rank new-key build that every peer waits on at
  `barrier.cluster`, and the DSMEM reduce-scatter over 128 rows x C partials. Without a per-phase
  breakdown the split cannot be named.
- **B=1:** both this kernel and master's Triton kernel sit at about 10 µs for a 0.1 µs floor. The
  dependent chain (index -> gather -> compute -> merge) is the limit.

**Verdict: PROMISING, narrowly.** Its decode wins came against the base. They are now matched by
master's deep-splits Triton kernel, so the case for a stage 2 rests on what master cannot do.
Stage 2 must:
1. Gate verify (L=8) off, back to master FA3.
2. Fix the decode_32 two-wave problem: use C=4 at B >= 32, or cut regs/smem so 32 clusters
   co-reside.
3. Add PDL and a split gather so the prologue overlaps the previous node and the compute overlaps
   the second half of the gather.

Target: decode_32 <= 12 µs and decode_16 <= 10 µs. If decode_32 cannot beat 22 µs at C=4, the
rewrite is DEAD. L=8 should only come back with a per-phase timing build (clock64 stamps per phase
in the self-check) that names the 13 µs.

---

## 4. sliding-single-launch-fused-attention (regression 0.945x): DEAD (superseded)

**Path:** no `.log` was archived for job f389af196ddf47b1. The profile shows the new
`_sliding_fused_kernel` as the only kernel on all six sliding shapes, so the path ran everywhere
and did not fall back. Full layers ran master FA3, as gated.

| shape (sliding) | incumbent replaced (dfee6f29: FA3+combine+prepare+pre-attn+norm-gate) | rewrite kernel | vs incumbent | floor |
|---|---|---|---|---|
| decode_1 | 15.1 | **10.2** | 1.48x faster | 0.11 |
| decode_16 | 17.9 (busy 16.4) | **17.4** | 1.03x | 1.65 |
| decode_32 | 21.2 (busy 20.0) | **28.8** | 0.74x | 3.29 |
| verify_1 | 16.2 (busy 14.7) | **25.5** | 0.64x | 0.12 |
| verify_16 | 20.4 (busy 18.9) | **30.7** | 0.66x | 1.86 |
| verify_32 | 26.7 (busy 25.2) | **44.4** | 0.60x | 3.72 |

**Where the time goes:** the single Triton kernel is 100% of each shape. Verify with M=64 tl.dot
tiles is 1.6-1.7x slower. B=32 with S=2 (128 CTAs) is under-split.

**Verdict: DEAD (superseded).** Its winning part, the decode Triton kernel, was harvested to
master. `sliding-fused-triton-decode1-gate` took it for decode_1, and
`sliding-deep-splits-on-cuda-splitkv` took it for batched decode at 12.9 / 22.1 µs. The verify
half lost by 1.6-1.7x, and the CUDA cluster rewrite (section 3) took that idea further and also
lost. There is nothing left to mint as its stage 2.

---

## 5. full-cuda-onelaunch-warpspec-bulk-gather (regression 0.824x): DEAD

**Path (log 4994759f5b4b4278):** `[full_ws] load_inline compiled in 50.0s (BULK=0 and BULK=1
built)`. Four BULK=1 self-checks pass (rel L2 about 1.6e-3, new rows equal 1.0000, counters 0),
and the BULK=0 diagnostic passes. PATH CHOSEN is `full_ws_kernel<BULK=1>`, grid 132, for full
decode 1<B<=64 at page 1. Verify, page 64, B=1 and dp stay on master. There was no fallback.

The eager diagnostic at B=16 x 80000:
- BULK=1: 667.3 µs, 0.49 TB/s
- BULK=0: 185.0 µs, 1.77 TB/s
- master path: 121.0 µs, 2.71 TB/s

| shape (full) | incumbent replaced (pre-attn + splitkv A + combine) | rewrite (one kernel, BULK=1) | vs incumbent | floor | x floor | BW |
|---|---|---|---|---|---|---|
| decode_16 | 2.1+111.4+7.9 = **121.4** | **705.9** | 5.8x slower | 63.2 | 11.2x | ~0.46 TB/s |
| decode_32 | 2.3+245.5+8.6 = **256.4** | **1406.4** | 5.5x slower | 126.4 | 11.1x | ~0.47 TB/s |
| decode_16, BULK=0 (diag only) | 121.0 (eager master) | 185.0 | 1.53x slower | 63.2 | 2.9x | 1.77 TB/s |

**Where the time goes:** everything is in the one kernel. The core bet was a single producer warp
issuing one `cp.async.bulk` per 128 B row into a 1-CTA/SM, 8-stage ring. That is TMA-issue-bound
at about 0.5 TB/s, a sixth of master. Even with BULK=0 the 1-CTA/SM single-producer design is
1.53x slower than master's 2-CTA/SM all-thread cp.async. The design keeps fewer requests in flight
than master, so it cannot reach master's bandwidth. The one-launch fusion and stream-K savings
(about 10-15 µs) cannot be measured under this design.

**Verdict: DEAD** as a rewrite line. Its central mechanism, the warp-specialized per-row bulk
producer, is slower on its own target shapes in both variants. Per-row bulk copies have no
structural fix on sm_90, because there is no TMA gather for 128 B page-1 rows. The ideas worth
keeping (in-kernel ticket combine, fused pre-attention, stream-K tiles) belong on master's
splitkv kernel A as stage 2 of `full-decode-cuda-gather-splitkv` (section 1), not as a stage 2 of
this rewrite.

---

## 01:36Z harvest

Master for comparison is BEST `sliding-deep-splits-on-cuda-splitkv`, submit 698d4cd3838c4a63, 1.3949x [1.374, 1.416]. That is the result file named in BEST.json. Per-shape figures are benchmark medians ± err; z is Δ / √(err_new² + err_ref²). The err values here are 0.0-0.3 µs, so z overstates confidence: identical untouched kernels drift up to ~6% between runs (LESSONS). Every claim below is therefore also checked against the kernel rows in the profile.

### full-decode-splitkv-s2-streamk-ticket-inproc — job e592b1f5d3db4f83, check 1.4171x [1.396, 1.443]: RAN, WON on both target shapes. LAND CANDIDATE.

**What ran (log):**
- The master splitkv TU built (48.9 s) and passed its 3 self-checks. The new TU `muse_full_decode_splitkv_v2` built with `TORCH_CUDA_ARCH_LIST=9.0a`, `-gencode=arch=compute_90a,code=sm_90a`, `-O3 -lineinfo`, in 36.9 s, on an H200 with capability (9, 0) and 132 SMs.
- Kernel attributes: `splitkv_attn_sk_kernel<4 stages, ticket>` uses 121 regs, 0 B local, 95264 B dyn smem, 2 CTAs/SM. The no-ticket variant uses 113 regs, and the 5-stage variant 117 regs with 112672 B. All three are binaryVersion and ptxVersion 90.
- The canary (B=2 short) and 8 self-check cases passed. They cover B=2/16/32/64, ragged and mixed lengths, kv 0.25, and grids 3, 7 and device. rel L2 was 1.63-1.75e-3, all finite. Two graph replays passed and tickets summed to 0 after the self-check. No fault.
- `PATH CHOSEN: full decode streamk+ticket v2` (full decode, 1<B≤64, page 1). Every other shape kept master's path.
- The prepare diagnostic (eager, cold L2, median of 10):

| B | stage-1 A + combine | stream-K + ticket, 4 stages (timed path) | stream-K no ticket + combine | stream-K + ticket, 5 stages |
|---|---|---|---|---|
| 16 | 117.5 + 12.2 = 129.7 µs (2.79 TB/s) | **112.7 µs (2.91 TB/s)** | 106.9 + 11.7 = 118.6 | 116.1 |
| 32 | 245.3 + 12.5 = 257.8 µs (2.67 TB/s) | **221.3 µs (2.96 TB/s)** | 215.1 + 11.9 = 227.0 | 221.1 |

  The diagnostic splits the gain in two. Stream-K split and locality gives 1.10x at B=16 and 1.14x at B=32. Folding the combine into a ticket gives 1.05x and 1.03x. A 5th stage gives nothing. This is the first variant above 2.9 TB/s; kernel A without the combine reaches 3.05 TB/s.

**Per-shape against BEST (698d4cd3):**

| shape | master µs | new µs | Δ% | z | kernel rows (profile) |
|---|---|---|---|---|---|
| **full decode_16** | 122.8 | 110.9 | **−9.7%** | **−57.8** | splitkv A 111.8 + combine 8.2 = 120.0 → sk 108.9 (−9.3%) |
| **full decode_32** | 241.9 | 226.8 | **−6.2%** | **−41.5** | 245.9 + 8.4 = 254.3 → sk 226.8 (−10.8%) |
| full decode_1 (not routed) | 27.3 | 27.2 | −0.3% | −1.0 | FA3, same |
| sliding decode_1/16/32 (not routed) | 13.2 / 15.3 / 24.2 | 13.1 / 15.2 / 24.3 | −0.6 / −1.0 / +0.2% | −1.3 / −2.8 / +0.5 | Triton, same |
| sliding verify_1/16/32 (not routed) | 16.7 / 22.7 / 28.7 | 16.8 / 22.7 / 28.9 | +0.6 / +0.2 / +0.6% | 1.5 / 0.7 / 2.2 | FA3, same |
| full verify_1/16/32 (not routed) | 30.0 / 162.0 / 300.8 | 29.8 / 163.4 / 302.1 | −0.6 / +0.9 / +0.4% | −2.6 / 4.5 / 3.0 | FA3 152.4 vs 151.1: drift |

- Only the two target shapes moved beyond noise, and the kernel rows confirm both: −11 µs and −27 µs per call. Every control is within ±1%. The +0.9% at full verify_16 is the untouched FA3 kernel (+1.3 µs), so it is run drift.
- Loads: decode 16 rose from 1.663 to 1.791 and decode 32 from 1.520 to 1.593. The other loads are flat (decode 1 moved 1.397 → 1.381, which is sliding decode_1 drift).
- The headline CI lower bound (1.396) sits just above master's point (1.3949), so the headline gain alone is marginal. The per-shape gain is unambiguous.

**Verdict: LAND CANDIDATE** (scored submit, needs the user's approval). Possible next stages:
- The no-ticket kernel A at 3.05 TB/s shows the ticket epilogue costs ~6 µs at B=16. A cheaper fold, such as last-CTA-per-request only, is the stage-3 lever.
- Port the kernel to decode_layer's full decode once `full-decode-splitkv-restack-on-pdl` lands there.

### sliding-cluster-s2-decode-c4-pdl-inproc — job 1990ce53e9d34073, check 1.4001x [1.379, 1.426]: RAN at every B. WON at B=1 and B=32, LOST at B=16. NEXT STAGE (gate C=8 off).

**What ran (log):**
- The master splitkv TU built (50.7 s) and passed its self-checks. The new fp8qk TU built with `-gencode=arch=compute_90a,code=sm_90a` in 40.7 s on capability (9, 0).
- `sliding_cluster_kernel` uses 124 regs, **32 B local/spill**, 103424 B dyn smem/CTA, and is binaryVersion and ptxVersion 90. Max active clusters are C=4 → 62, C=8 → 30 and C=16 → 14.
- Canaries passed (finite) at C=16, C=8 and C=4. Self-checks passed for C=16 at B=1 (kv 1 and 0.25), C=8 at B=2 and B=16, and C=4 at B=17 and B=32. rel L2 was 1.59-1.67e-3, and cache k/v matched 1.0000 with rel 0. A graph capture and replay matched at each C. No fault.
- **PATH CHOSEN: C=16 at B=1, C=8 at B=2..16, C=4 at B=17..32 (fp8qk).** Master kept B>32 decode, every verify (FA3) and page 64.
- Per-case path lines confirm the scored shapes: batch=1 → C=16, batch=16 → C=8, batch=32 → C=4. The profile shows `sliding_cluster_kernel` as the only kernel on sliding decode_1, decode_16 and decode_32. **The cluster path ran at every B.**
- The log has no prepare timing or PDL line. Each sliding decode shape is a single kernel, so PDL has nothing to overlap inside the measured region.

**Per-shape against BEST (698d4cd3):**

| shape | master µs | new µs | Δ% | z | kernel rows (profile) |
|---|---|---|---|---|---|
| **sliding decode_1 (C=16)** | 13.2 | 12.5 | **−5.0%** | **−10.8** | Triton 10.4 → cluster 10.6 (kernel flat; the 0.7 µs shape gain is outside the kernel) |
| **sliding decode_16 (C=8)** | 15.3 | 16.2 | **+5.7%** | **+13.6** | 12.9 → 13.8 (+7%): LOST |
| **sliding decode_32 (C=4)** | 24.2 | 20.2 | **−16.8%** | **−48.1** | 22.1 → 18.3 (−17%) |
| full decode_1/16 (untouched) | 27.3 / 122.8 | 27.2 / 122.2 | −0.3 / −0.5% | −1.2 / −3.8 | same |
| full decode_32 (untouched) | 241.9 | 257.7 | +6.5% | +51.8 | splitkv A 245.9 vs 244.4: identical kernel, benchmark drift (the known 242 vs 256-260 bimodality) |
| sliding verify_1/16/32 (untouched) | 16.7 / 22.7 / 28.7 | 16.6 / 22.6 / 29.5 | −0.8 / −0.3 / +2.9% | −2.1 / −0.9 / 11.0 | FA3 15.0 vs 14.5 at verify_32: drift |
| full verify_1/16/32 (untouched) | 30.0 / 162.0 / 300.8 | 29.9 / 162.0 / 300.6 | ≤0.6% | ≤2.5 | same |

- The kernel targets were missed: the review set decode_16 ≤ 10 µs and decode_32 ≤ 12 µs, and the kernels ran at 13.8 and 18.3.
- The decode_32 win (−3.8 µs kernel, −4.0 µs shape) and the decode_1 win (−0.7 µs) are real. decode_16 is a real loss (+0.9 µs).
- The full decode_32 +6.5% is drift on an identical kernel. It costs this run about 0.01 of headline, so the headline understates the sliding change.
- **Not a landing candidate as-is**, because of the decode_16 loss.

**Gated estimate:** take BEST's rows, substitute the cluster rows at B=1 and B=32 only, and recompute the six load speedups. That gives about 1.411x vs 1.395, which is the next stage. Concretely: the same kernel with the C=8 band turned off (B=2..16 back to master Triton), as an `okc check`.

### Do they stack?

**Yes.** The two changes touch disjoint shapes and code paths:
- streamk: full layer, decode, 1<B≤64, page 1
- cluster: sliding layer, decode, B≤32, page 1

The streamk run left every sliding row on master, and the cluster run left every full row on master's kernels. Substituting both runs' winning rows into BEST's rows gives these estimates (load-level recompute; drift not removed):

| stack | estimated headline |
|---|---|
| streamk only | ~1.424 |
| streamk + cluster at B=1,32 only | **~1.440** |
| streamk + cluster at every B | ~1.437 |

Restack streamk with the C=8-gated cluster kernel as one check.

### Answer for the held `sliding-decode-paired-kvhead-rows`

The cluster path **ran at every B** (C=16 / C=8 / C=4, confirmed by PATH CHOSEN, the per-case path lines and the profile). It won at B=1 and B=32 and **lost at B=16**. By the hold rule ("if it lands on all batches, drop; ... lands only at one batch, run this pair"), it did **not** land on all batches, so **unhold the pair and its unpaired S=8 control**.

The pair targets B>1:
- decode_16: Triton master 12.9 µs, cluster 13.8. The cluster loses here, so the pair has the shape to itself.
- decode_32: cluster 18.3 µs vs the earlier unpaired S=8 Triton 17.3 µs. The pair and its control must beat 18.3 there to displace the cluster.

## 03:41Z harvest

References:
- BEST `sliding-deep-splits-on-cuda-splitkv`: submit 698d4cd3838c4a63, 1.3949x.
- The unlanded stream-K s2 `full-decode-splitkv-s2-streamk-ticket-inproc`: check e592b1f5d3db4f83, 1.4171x.

Notation:
- z is Δ / √(err_new² + err_ref²).
- Benchmark err here is 0.04-0.3 µs, so any |Δ| under ~2% is read against the profile kernel rows, not z.

### full-decode1-streamk-s3-two-level-ticket — job 2a3241780c7b4c46, check 1.4322x [1.411, 1.455]: RAN. LAND CANDIDATE (it carries s2).

**What ran (log):**
- Build: the `muse_full_decode_splitkv_v3` TU, sm_90a only, `-O3 -lineinfo`. The s2 kernels match the last run (ticket 119 regs, no-ticket 113, 5-stage 119). The new `<4 stages, two-level ticket>` kernel uses 121 regs, 0 B local, 95264 B smem and runs 2 CTAs/SM.
- Self-check: the canary and every s2 case passed. The B=1 two-level cases also passed: 80000 / 4097 / 63 / 1 keys, kv 1.0 and 0.25, grids 3 / 7 / SM / 2xSM, all at rel L2 ≈1.7e-3 and finite. Tickets at both levels summed to 0 afterwards. No fault.
- `PATH CHOSEN: full decode streamk+ticket v2` (1<B≤64, as in s2) + `PATH CHOSEN: full decode_1 streamk two-level`; every other shape is on master's path.
- B=1 diagnostic, cold L2:

| variant | time |
|---|---|
| single-level grid 2xSM (stage 2) | 53.9 µs |
| single-level grid SM | 34.8 µs |
| **two-level grid SM (timed)** | **23.1 µs (0.89 TB/s)** |
| two-level grid 2xSM | 24.7 µs |
| master FA3 main+combine + norm-gate | 33.2 + 5.5 = 38.7 µs |

  At B=16 the two-level kernel is slower than s2 (115.8 vs 107.0 µs), so gating it to B=1 is correct.

**Per-shape** (µs; Δ vs s2 / vs BEST):

| shape | BEST | s2 | new | vs s2 | vs BEST | kernel rows |
|---|---|---|---|---|---|---|
| **full decode_1** | 27.26 | 27.19 | **22.60** | **−16.9% z−67** | **−17.1% z−67** | FA3 14.6 + combine 7.1 + norm-gate 1.1 → sk3 18.6 (+pre 1.4) |
| **full decode_16** | 122.79 | 110.91 | 111.14 | +0.2% z+0.8 | **−9.5% z−53** | sk 110.1 (s2 108.9) |
| **full decode_32** | 241.90 | 226.81 | 226.42 | −0.2% z−1.0 | **−6.4% z−54** | sk 225.6 (s2 226.8) |
| sliding decode_1/16/32 | 13.17/15.34/24.24 | 13.09/15.19/24.29 | 13.15/15.33/24.33 | ≤+0.9% | ≤+0.4% | Triton, same |
| sliding verify_1 (untouched) | 16.72 | 16.83 | 17.78 | +5.6% z+14 | +6.3% z+16 | FA3 7.9+3.9 vs 7.6+3.9: batch drift (every attention run in this harvest reads 17.4-17.8 here) |
| full verify_1 (untouched) | 30.03 | 29.84 | 30.51 | +2.2% | +1.6% | FA3 15.4 vs 15.0: drift |
| verify_16/32 both layers | — | — | — | ≤±1.8% | ≤±1.2% | same kernels |

- Loads: decode 1 rose 1.381 → 1.513 vs s2 (1.397 vs BEST). decode 16 is 1.784 and decode 32 1.596 (s2 levels). verify 1 fell to 1.239 on untouched-FA3 drift alone.
- The headline CI lower bound (1.411) is above BEST's point (1.3949), and above s2's point (1.4171).
- Substituting this run's three full-decode rows into BEST's rows gives **≈1.440x**. The run's own 1.432 is held down by the verify_1 drift.

**Verdict: LAND CANDIDATE.** It needs a scored submit with the user's approval. It contains s2 (full decode 1<B≤64) plus the B=1 two-level ticket. Next:
- Restack the sliding cluster kernel with the C=8 band gated off (B=1 and B=32 only) on top. The earlier estimate was +0.016.
- A cheaper ticket epilogue at B=16 is still open: the no-ticket kernel A ran at 3.06 TB/s.

### full-verify-wgmma-s3-warpgroup-per-kvhead-inproc — job 2c3f7fff72bb4e1d, check 1.3801x: BUILT, FAILED SELF-CHECK, FELL BACK. FIXABLE.

- Build: sm_90a only. `splitkv_verify_wgmma_kernel<ticket>` 128 regs and `<B8>` 120 regs, both 0 B local, 2 CTAs/SM, 91744 B dyn smem.
- Canary (B=2, grid 4) and case 1 (B=6, kv 1.0, grid=device) passed at rel L2 7.9e-3 and 4.8e-3. **Case 2 (B=6, kv_scale 0.25, grid=3) failed at rel L2 1.23, finite, with both the ticket and the B8 combine.** The bug is therefore in the main wgmma kernel, not the combine.
- `PATH CHOSEN: full verify master FA3 (fallback: self-check failed)`. Every scored full verify row is FA3: verify_16 162.6 vs 162.0 and verify_32 301.9 vs 300.8. The wgmma kernel was never timed.
- Case 2 confounds two triggers: kv_scale≠1 and grid=3, where each CTA loops over several work items. Next stage: split it into (kv 0.25, grid=device) and (kv 1.0, grid=3) and print the per-item loop state.
- Other rows (sliding Triton, full decode s1) are master's. The full decode_32 +7% is the known bimodality.

### sliding-decode-paired-kvhead-rows — job 48ffbe5690cc4ab4, 1.2865x: RAN (Triton), LOST BADLY. DEAD.

| shape | BEST | new | Δ | kernel |
|---|---|---|---|---|
| sliding decode_16 | 15.34 | 30.58 | +99% z+201 | `_sliding_paired_kernel` 16.8 vs `_sliding_fused_kernel` 12.9 |
| sliding decode_32 | 24.24 | 43.15 | +78% z+146 | 28.1 vs 22.1 |

- The paired kernel itself is 1.3x slower than the fused kernel. It also adds ~11 µs per shape outside any profiled kernel (shape 30.6 vs kernel 16.8; master's gap is 2.4), which looks like host or launch overhead in the timed path.
- It also loses to the cluster kernel at B=32 (18.3 µs). This closes the paired-row line.

### sliding-verify-fa3-head-pairing — job 031731b5db294a92, 1.3475x: RAN (Python routing), LOST. DEAD.

- sliding verify_16 rose 22.67 → 29.43 (+29.8%) and verify_32 28.70 → 33.71 (+17.4%).
- In the profile, FA3 main went 10.7 → 13.2 and combine 4.1 → 7.6 µs, plus a `_paired_norm_gate_kernel` (2.3 µs).
- Pairing two heads per FA3 call doubles the rows per kv head and the split-combine work. FA3 is already efficient per head on verify.

## 04:36Z harvest

References: new master `full-decode1-streamk-s3-two-level-ticket` (check 2a3241780c7b4c46, 1.4322). All three runs below were built on older bases. None of them has the two-level ticket, so their full decode_1 row reads 27.1-27.6 µs against master's 22.6 µs (+20%, weight 16). That alone explains why their headlines sit below master. Read the changed rows only. Shape err is about 50 ns, but sliding rows drift 2-3% between batches (LESSONS), so a move under ~3% on an untouched row is drift, whatever its z.

### stack-streamk-ticket-cluster-c16-c4: job 195dcc1209b6457b, check 1.4266 [1.405, 1.454], base sliding-deep-splits (1.3949). LAND-CANDIDATE once rebased onto master.

- PATH CHOSEN: full decode uses streamk+ticket v2. The sliding paths are cluster v2 C=16 at B=1 (fp8qk), the master Triton deep-splits at B=2..16 (C=8 is gated off) and cluster v2 C=4 at B=17..32. Every verify stays on FA3.
- Build: sm_90a only, `sliding_cluster_kernel` 124 regs, 32 B local, max active clusters C4=62, C8=30, C16=14. All self-checks passed: rel L2 about 1.6e-3, cache k and v match 1.0000, graph replay matches.
- Rows against the new master:

| row | hyp µs | master µs | change |
|---|---|---|---|
| sliding decode_32 (`sliding_cluster_kernel` 18.0) | 20.05 | 24.33 | −17.6%, z −49, pr 1.180 |
| sliding decode_1 (C=16, 10.6) | 12.57 | 13.15 | −4.4%, z −11, pr 1.024 |
| full decode_16 | 111.64 | 111.14 | +0.4% (flat) |
| full decode_32 | 226.98 | 226.42 | +0.2% (flat) |

  Every other row except full decode_1 is within ±1.3%.
- The targeted rows moved, and the decode_32 gain is the largest single-row gain on this problem today. Substituting the two sliding rows into master's result gives an estimated headline of 1.449-1.476. My two estimates bracket it because I could not recover the judge's exact aggregation.
- Next: one restack on master that adds the sliding cluster C=16 (B=1) and C=4 (B=17..32) routes and keeps the two-level ticket at full B=1. Then a scored submit.

### full-decode-splitkv-s3-fused-preattn: job 825baa03798f431a, check 1.4283 [1.407, 1.450], base stage 2 (e592b1f5, 1.4171). STACK-INTO-NEXT.

- `PATH CHOSEN: full decode streamk fused-preattn v3` and `every other shape: master path`. The v3 kernel has 125 regs, 0 local and 2 CTAs/SM. The new k/v cache rows match bit for bit (rel L2 0.00e+00), other rows are unchanged, and the ticket sum is 0.
- Diagnostic: `[sk3-diag]` at B=16 gives pre-attn 6.2 + v2 111.9 = 118.2 µs vs fused v3 115.1 µs. The in-kernel q/kv build costs +3.1 µs and saves 3.1. At B=32, 227.7 vs 220.5 µs: the build costs −0.8 µs and saves 7.2.
- Rows vs stage 2: full decode_16 110.91 → 109.55 (−1.2%, z −4.5) and full decode_32 226.81 → 223.11 (−1.6%, z −8.8). Against the new master the figures are −1.4% and −1.5%. The kernel profiles show 107.4 µs (decode_16) and 227.4 µs (decode_32), and `_pre_attention_kernel` is gone from both rows.
- That is a small real gain of about 1.4-3.7 µs on two weight-16 rows, about +0.2% on the headline. It is too small to land alone. Fold it into the next attention restack together with the two-level ticket at B=1. Both modify `splitkv_attn_sk_kernel`, so v3 has to be gated to 1 < B ≤ 64.

### full-verify-wgmma-s4-multitile-fix: job 51b0d5c110f3459c, check 1.3758 (diagnosis build). FIX-AND-RERUN.

- PATH CHOSEN: full verify stays on master FA3 paired. All three variants (ticket, b8, ticket_sync) fell back on `lens_a-grid3-kv1` at rel L2 1.23e+00, finite, error word 0. No mbarrier timeout and no shape error occurred, so the wgmma kernel was never timed. The verify rows are master FA3 at ±0.6%.
- BISECT, identical for b8 and ticket_sync: `single-request multi-tile cases fail (B1-2t, B1-4t, B1-17t, B1-34t, B1-4t-kv0.25): multi-tile loop state (n >= 1), no request crossing involved`.
  - Passing cases: canary, single-tile-lens_a-device, kv0.25-grid264, lens_b-device and coherent-lens_a-device. Every one of them gives each CTA at most one tile per segment.
- DIAG (ticket_sync lens_a-grid3): the 1-tile segments inside multi-tile CTAs are GOOD (2/2), and every segment of 2 or more tiles is BAD. A bad segment has `m_bad_rows 128` (every row's running max is wrong), l_err 0.8-1.2 and o_err 3-5. B1-2t (one request, 2 tiles, no crossing, no index rotation) already fails.
- **The bug: per-segment state carried from tile lt=0 into tile lt≥1 inside the same segment.** The SYNC variant (cp_wait_all + proxy fence + `__syncthreads` around every tile) fails identically, so the bug is not a ring or mbarrier race. A 1-tile segment that starts at stage ≥1 passes, so the stage, ring and descriptor addressing are not the cause either. kv_scale, request crossing and the IDX_TILES>16 rotation are ruled out by the table.
- The row max is wrong on every row, so the error already shows up in S or its max at tile lt≥1, not only in the O/l rescale.
- Reading the s4 kernel, the only per-segment state that is NOT reloaded right before the S wgmma of a later tile is the register-sourced Q A-fragment `qa[2][4]`. It is loaded once at `n == 0 || lt == 0` and reused across loop iterations as the `wgmma ... {a regs}, desc` operand. A 1-tile segment always loads it in the same iteration, which matches the good/bad split exactly.
- Stage 5 should test that first:
  - (a) Reload `qa` every tile, unconditionally. It is 2 KB from L1/L2.
  - (b) Alternatively, stage Q in smem and use a smem-descriptor A operand.
  - (c) Add a DIAG that dumps S (pre-max) for tile 1 of B1-2t against the fp32 reference. That separates "S wrong" from "carry wrong".
  - If (a) still fails, the next suspects are the `mA`/`lA` carry (`ex2(mA - uA)`, with the Max_offset folded) and `oacc` across the `fence_acc` / `wg_fence` pairs.
  - Keep SYNC off; it is ruled out.
