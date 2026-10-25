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
