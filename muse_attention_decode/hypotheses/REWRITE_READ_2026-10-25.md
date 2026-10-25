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
