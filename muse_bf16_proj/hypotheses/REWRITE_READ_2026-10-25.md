# muse_bf16_proj: per-shape read of the 2026-10-24 kind:rewrite runs

Read 2026-10-25 from recorded runs only; no GPU was used. Sources are `done/regression/<slug>.{md,result.json}` and `archive/<slug>/<job>.{log,profile.txt,profile.json.gz}`.

**Baseline.** Every rewrite's `attribution.base_result` is `done/neutral/bf16-m32-swapab-wgmma-triton.result.json`, job `1602a4e947e64090` (check, 1.198x). In the tables below:
- "incumbent kernel" is that run's profile top_kernels.
- "row" is the judged per-shape median: cold L2, CUDA graph, about 4 us of replay overhead already subtracted.
- Floors are the run's own `speed_of_light_ns`.

**Measurement caveat found while reading.** The judge flushes L2 before every timed replay (TASK.md). For weights that fit in L2 (qkv_gate 37.7 MB, o 16.8 MB, the unpooler slice 8.4 MB), the profile's kernel time can sit well below what the judge measures:
- Triton swap-AB on qkv m=1 profiles at 9.4 us against a 12.46 us row (bf16-m16-swapab-wgmma-extend, `89fabef70bd745f7`).
- The gemv rewrite's qkv m=1 profiles at 8.8 us against a 12.83 us row.
- The base Triton qkv kernel has only a ~1.1 us gap.

So on L2-resident layers the profile kernel time is a warm-ish lower bound, and the row decides. For pooler and unpooler-full (67 MB, larger than L2) the profile and the row agree to within ~1 us.

---

## 1. pooler-unpooler-m128-tma-wspec-rewrite: DEAD

Job `61da9b91dab44bf4` (submit), 9/9 tests passed, headline 1.164x [1.146, 1.181].

**Path that ran.** The log shows `[m128_ws] triton 3.8.0; TensorDescriptor available`. Both layers selected variant `(warp_specialize=True, num_stages=3)`, so the self-check passed and nothing fell back. The profile shows `_bf16_m128_tma_ws_kernel` as the only kernel on both target rows. The compile diagnostics differed by layer:

| layer | regs | spills | wgmma | cp.async.bulk.tensor |
|---|---|---|---|---|
| pooler | 255 | **110** | 8 | 3 |
| unpooler | 218 | 0 | 8 | 3 |

**Per target shape:**

| shape | incumbent nvjet kernel | rewrite kernel | row base -> hyp | floor | rewrite / floor |
|---|---|---|---|---|---|
| pooler verify_16 m=128 | 25.7 | **33.5** (+30%) | 25.55 -> 31.69 | 14.64 | 2.29x |
| unpooler verify_16 m=128 | 20.4 | **27.6** (+35%) | 19.79 -> 25.68 | 14.64 | 1.89x |

**Where the time went.** These are warm-L2 prepare() events. Per the review, read them as a mainloop vs tail split only.

| layer | full | no_reduce (mainloop) | tail | parent (cp.async) mainloop / tail | milestone M1 / M2 |
|---|---|---|---|---|---|
| pooler | 36.85 | 27.64 | 9.21 | 26.0 / 6.5 | <=21 / <=4: both missed |
| unpooler | 30.73 | 27.43 | 3.30 | 26.0 / 4.3 | <=18 / <=2.5: both missed |

- **Mainloop.** TMA, BLOCK_K 128 and warp specialization did not move it. It got slightly worse than the parent's cp.async version (26.0 to ~27.5 us). The clean evidence is the unpooler: no spills, the same 27.4 us. The 128x128-tile / 128-CTA mainloop is stuck at ~2.4 TB/s whatever the load mechanism. That points at the tiling itself (the x K-slice re-read from L2 per CTA, ~2x the weight bytes, as the review warned; one CTA per SM), not at how the weight arrives.
- **Tail.** Keeping the winner's partial in registers trimmed the unpooler tail (4.3 to 3.3 us). On the pooler, the 110-register spill from warp specialization made it worse (6.5 to 9.2 us).
- **No milestone reached** beyond M1-correct.

**Verdict: DEAD.** Two Triton mainloops now land at 26-27.6 us warm, above nvjet's whole kernel on both rows: the parent `_bf16_tc_splitk_kernel` with cp.async, and this one with TMA and warp specialization. Triton 3.8's sm90 warp specialization compiles but spills at 8 warps. A BLOCK_N 256 or stream-K stage 2 in Triton would be a third attempt at the same ceiling. If m=128 pooler/unpooler is attacked again, do it as stage 2 of bf16-verify-streamk-cutlass-wgmma-rewrite: the same CUTLASS kernel at 65 <= m <= 128, with real producer/consumer warp specialization and register reallocation. Do not stage-2 this one.

---

## 2. bf16-small-m-cuda-gemv-rewrite: DEAD

Job `b76aca7ec1a048e7` (submit), 9/9 tests passed, headline 1.167x [1.149, 1.184].

**Path that ran.**
- `[gemv] load_inline compiled in 39.9s`, 110592 B smem per CTA, occupancy 2 CTAs/SM as designed.
- All four layers printed `self-check passed`. Grids: qkv 252 (36 x SPLIT_K 7), o 256 (128 x 2), pooler 256 (64 x 4), unpooler slice 256 (16 x 16).
- The profile shows `gemv_m8_kernel` alone on all eight target rows. There was no fallback.

**Per target shape.** Incumbent = the base run's Triton m<=8 kernels. "Triton swap-AB m16 row" is the competing Triton path from bf16-m16-swapab-wgmma-extend (`89fabef70bd745f7`), the alternative for the same shapes.

| shape | incumbent kernel | gemv kernel (profile) | row base -> hyp | Triton swap-AB m16 row | floor | hyp row / floor |
|---|---|---|---|---|---|---|
| qkv_gate m=1 | 11.8 | 8.8 | 12.91 -> 12.83 (flat) | 12.46 | 7.87 | 1.63x |
| qkv_gate m=8 | 12.0 | 11.2 | 13.18 -> 15.28 (-16%) | 12.67 | 7.90 | 1.93x |
| o m=1 | 4.6 | 5.2 | 6.32 -> 7.85 (-24%) | 6.27 | 3.50 | 2.24x |
| o m=8 | 5.2 | 6.8 | 6.32 -> 9.52 (-51%) | 6.08 | 3.53 | 2.70x |
| pooler m=1 | 19.0 | 18.9 | 19.00 -> 19.29 (flat) | 18.95 | 13.99 | 1.38x |
| pooler m=8 | 19.3 | 21.3 | 19.14 -> 20.57 (-7%) | 18.79 | 14.02 | 1.47x |
| unpooler slice m=1 | 4.5 | 4.5 | 5.94 -> 6.20 (-4%) | 5.46 | 1.75 | 3.54x |
| unpooler slice m=8 | 4.6 | 6.2 | 6.13 -> 7.55 (-23%) | 5.84 | 1.76 | 4.29x |

Milestone M2 (at or below the incumbent) was met on no judged row; qkv m=1 and pooler m=1 tie. The only profile-level win, qkv m=1 at 8.8 us, does not show in the cold-L2 row (see the caveat).

**Where the time went.**
- **m=8 always costs more than m=1:** qkv +2.4, o +1.6, pooler +2.4, unpooler +1.7 us in the profile, and +1.4 to +2.5 us in the rows. In a bandwidth-bound n8 MMA kernel the token count should be free. The m-proportional cost is in the per-row partial store and the serialized ticket-reduce tail. This is most likely: the unpooler slice uses SPLIT_K 16, and o uses 2 with only 512 K per item.
- **o and the unpooler slice are latency-bound.** Each CTA streams only 64 KB (o) or 32 KB (slice) of weight, so the 6-stage ring never fills. The prepare warm-L2 eager prints show the two lowest effective rates: o 1.97 TB/s, unpooler 1.0 TB/s.
- **On pooler (the only pure-DRAM row) the kernel ties Triton at ~3.5 TB/s.** Doubling the bytes in flight did not lift the ceiling. This confirms the review's "weakest link": the gap to the floor is fixed per-launch cost, not bytes in flight.

**Verdict: DEAD.** No target row beats the incumbent, and the newer Triton swap-AB m16 path beats it on every row (qkv m=8 by 2.6 us, o m=8 by 3.4 us). A stage 2 that fixed the reduce tail and the o/slice splits could at best tie Triton swap-AB on pooler and qkv m=1. Two things stay reusable: the `load_inline` plus per-layer self-check and fallback scaffold (40 s compile), and the ticket-reduce skeleton. bf16-lossless-exp4 already reused them.

---

## 3. bf16-lossless-exp4-weight-stream-rewrite: PROMISING (conditional, with a kill gate)

Job `542f2b169c934706` (submit), 9/9 tests passed, headline **0.707x** [0.697, 0.718].

**Path that ran.**
- `[lossless] load_inline compiled in 52.2s`.
- Kernel instantiations: `_lossless_swapab_kernel<8/16/32>` at 98/112/115 regs, 0 spills, 115.7/132.1/164.9 KB smem (8 stages), **1 CTA/SM**.
- Codec, per layer: 12.17 bits/element, ratio 0.7605-0.7609, escape groups 0.27-0.31%, round trip bit-exact through the kernel decoder.
- Compressed sizes: qkv 28.7 MB, o 12.8, pooler 51.1, unpooler 51.1.
- `PATH CHOSEN ... lossless; self-check passed` for all five routes: qkv, o, pooler, unpooler full, unpooler slice.
- The profile shows the lossless kernel alone on every m<=32 row. There was no fallback. M1 is met.

**Per target shape.** Times in us; compressed floor = raw floor x 0.761.

| shape | incumbent kernel | lossless kernel | row base -> hyp | raw / compressed floor | hyp kernel / incumbent |
|---|---|---|---|---|---|
| pooler m=1 | 19.0 | 40.6 | 19.00 -> 41.49 | 14.0 / 10.7 | 2.14x |
| pooler m=8 | 19.3 | 41.0 | 19.14 -> 42.11 | 14.0 / 10.7 | 2.12x |
| pooler m=16 | 19.5 | 45.2 | 19.85 -> 44.82 | 14.1 / 10.7 | 2.32x |
| pooler m=32 | 20.5 | 51.4 | 20.00 -> 51.05 | 14.1 / 10.8 | 2.51x |
| unpooler-full m=16 | 18.5 (nvjet) | 43.4 | 18.61 -> 41.58 | 14.1 / 10.7 | 2.35x |
| unpooler-full m=32 | 18.9 | 45.1 | 18.48 -> 44.83 | 14.1 / 10.8 | 2.39x |
| qkv_gate m=1 | 11.8 | 22.5 | 12.91 -> 30.19 | 7.9 / 6.0 | 1.91x |
| qkv_gate m=8 | 12.0 | 23.9 | 13.18 -> 31.59 | 7.9 / 6.0 | 1.99x |
| qkv_gate m=16 | 12.3 | 26.4 | 13.39 -> 33.86 | 7.9 / 6.0 | 2.15x |
| qkv_gate m=32 | 12.1 | 32.4 | 13.69 -> 38.70 | 8.0 / 6.1 | 2.68x |
| o m=1 / 8 / 16 / 32 | 4.6 / 5.2 / 4.8 / 4.5 | 10.8 / 10.9 / 11.4 / 12.7 | ~6.5 -> 15.3-16.6 | 3.5 / 2.7 | 2.1-2.8x |
| unpooler slice m=1 / 8 | 4.5 / 4.6 | 7.8 / 8.7 | 5.94 / 6.13 -> 10.98 / 11.49 | 1.75 / 1.3 | 1.7-1.9x |

M2 and M3 were missed everywhere. The kernel is 1.9-2.7x slower than the incumbent.

**Where the time went.**
- **Stream rate.** The prepare warm-L2 prints give 1.05-1.19 TB/s compressed at NT=8, which is 1.4-1.56 TB/s raw-equivalent. The incumbent Triton streams run ~3.5 TB/s raw. To tie on pooler, the kernel needs ~2.7 TB/s compressed; for the M3 target (15.5 us) it needs ~3.3 TB/s.
- **Not integer-ALU throughput.** A rough count: about 7 elements/clk/SM decoded at ~2.5 int ops per element is ~18 int ops/clk/SM, well under the 64/clk INT32 rate.
- **Most likely cause: issue/latency, with only 4 warps per SM.** Each SM runs one 128-thread CTA, so each SMSP has a single warp. That warp serially does cp.async waits, the integer dequant, ldmatrix of x, and dependent mma.sync chains. Nothing hides one stage's dequant behind another warp's loads. (The gemv rewrite, with no dequant, ran 8 warps/SM and reached ~3.5 TB/s.)
- **Time grows with token count:** pooler 40.6 -> 45.2 -> 51.4 us and qkv 22.5 -> 26.4 -> 32.4 us for NT 8/16/32. The per-n-fragment MMA and B-fragment work on top of the dequant is serialized in the same 4 warps. This also suggests the decoded A fragment is not amortized well across NT/8 fragments, which should be checked in the code.
- **o and the unpooler slice are latency-bound.** These are 12.8 MB and 6.4 MB streams on 128 CTAs. They cannot beat the incumbent even at full rate, because the incumbent's fixed cost dominates.

**Why PROMISING and not DEAD.** The risky parts all worked on the judge on the first try: the lossless codec (0.761 ratio, bit-exact, ~0.3% escapes on the judge's weights), the 52 s compile, routing and self-check. The loss is a kernel-occupancy problem, not a property of the idea. It is also the only lever that can put m<=32 pooler/unpooler/qkv rows below their raw-byte floors. The other approaches here (Triton tuning, the CUDA gemv) tie at ~3.5 TB/s raw.

**Stage 2 must fix these:**
1. **Occupancy.** Reach 8-16 warps per SM. Options: a 3-4-stage ring (~13 KB compressed per stage) so 2-3 CTAs fit per SM, or 8 warps per CTA with 2 warps per 16-row tile splitting the K-stage. Print `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and the achieved warps/SM.
2. **Dequant once per k16 A fragment.** Decode each fragment once and reuse it across all NT/8 B fragments. Keep decoded fragments for stage s+1 in registers while stage s's MMAs issue, i.e. software-pipeline the dequant against the MMAs.
3. **Narrow the scope to the shapes that can win:** pooler and unpooler-full at m 1-32, and qkv_gate. Route o and the unpooler slice back to master; they are latency-bound and do not pay.
4. **Print a cold-L2 timing in prepare** (flush a buffer larger than L2 before the event), so qkv's profile/row gap does not mislead.

**Kill gate.** If stage 2's pooler m=1 kernel is not <= 19 us (>= 2.7 TB/s compressed, the incumbent's time), the codec cannot be streamed fast enough with mma.sync, and the line is DEAD.

---

## 4. bf16-verify-streamk-cutlass-wgmma-rewrite (running/, implemented, not yet run): still worth its check

Do the three results change its case? They strengthen it. Run it as a check, not a submit.

**For running it:**
- **load_inline + nvcc on the judge is now proven three more times:** gemv 39.9 s, lossless 52.2 s, attention 41 s. The import-time build, the per-layer self-check and the print-and-fall-back pattern behaved as designed in all three. The remaining unknown is only CUTLASS header discovery and its 60-150 s compile. Both fit the 10-minute budget, since stream-K is not stacked with another extension.
- **The m128 rewrite (section 1) closes the hand-rolled Triton route for m>=128.** Triton 3.8's sm90 warp specialization spills and gives no mainloop gain. A CUTLASS `KernelTmaWarpSpecializedCooperative` kernel, with real producer/consumer warp specialization and register reallocation, is the remaining credible mainloop for m=128/256. Its stage 2 (65 <= m <= 128) now inherits the pooler/unpooler m=128 targets: 25.7 and 20.4 us against a 14.6 floor.
- **The m=256 incumbent rows are stable,** giving a clean yardstick. Across the base and all three rewrite runs they ranged: qkv 19.57-19.69, o 9.01-9.56, pooler 30.53-31.12, unpooler 23.89-24.74 us.

**Cautions the results add:**
- **Judge on the rows as well as top_kernels.** At m=256 the shapes are math-bound, so the warm/cold L2 gap should be small. But the stream-K barrier memset is an extra graph node. LESSONS puts the per-node cost at ~0.3-4 us, which can eat most of the o gain (M2 9.0 us). This is why o and unpooler use Heuristic decomposition.
- **cuBLAS-incumbent rows at m>=128 swing between runs.** For example, pooler m=128 nvjet was 23.9 us in the gemv run against 25.7 us in the base. A per-shape gain under ~7% on qkv or pooler needs a repeat before it counts.
- **Read the `[streamk]` log lines first.** `no CUTLASS headers`, `build FAILED` or `DISABLED` means the run measured master, not the kernel.

---

## 5. load_inline arch check (sm_90 vs sm_90a), added at the coordinator's request

The prefill read found a bug in how torch's cpp_extension builds CUDA on the judge (LESSONS, 2026-10-25). It adds its own plain `compute_90` `-gencode` from `TORCH_CUDA_ARCH_LIST` or the detected capability, even when the flags say `-arch=sm_90a`. ptxas cannot assemble wgmma or setmaxnreg for that sm_90 pass, so the build fails and the run silently measures the fallback. Each bf16 rewrite, checked against that bug:

| rewrite | load_inline? | wgmma/setmaxnreg in its CUDA? | archived log | did the new kernel run? |
|---|---|---|---|---|
| bf16-small-m-cuda-gemv-rewrite | yes, `-O3 -arch=sm_90a` | no: `mma.sync` m16n8k16, ldmatrix and cp.async are all valid on plain sm_90. The only "wgmma" matches in the patch are context lines from master's docstring. | `[gemv] load_inline compiled in 39.9s`; no ptxas or FAILED lines; `self-check passed` on all 4 layers | **Yes.** The profile shows `gemv_m8_kernel` on all 8 target rows. The read in section 2 is of the real kernel. |
| bf16-lossless-exp4-weight-stream-rewrite | yes, `-O3 -arch=sm_90a` | no: `mma.sync`, `prmt` and integer ops only | `[lossless] load_inline compiled in 52.2s`; no ptxas or FAILED lines; `PATH CHOSEN route=<all 5>: lossless; self-check passed` | **Yes.** The profile shows `_lossless_swapab_kernel<8/16/32>` on every m<=32 row. The read in section 3 is of the real kernel. |
| pooler-unpooler-m128-tma-wspec-rewrite | **no.** This is a Triton kernel; Triton compiles sm_90a itself (log: `wgmma=8 cp.async.bulk.tensor=3`). | n/a | `selected variant: (True, 3)` for both layers | **Yes.** The profile shows `_bf16_m128_tma_ws_kernel` on both m=128 rows. |

Stage 2 of the lossless rewrite (section 3) stays on mma.sync and is unaffected. If it moves to wgmma, it needs the fix below.

### bf16-verify-streamk-cutlass-wgmma-rewrite: will hit the same bug; fix before it runs

The worktree `submission.py` (L1007-1023) calls `load_inline` with `extra_cuda_cflags` containing `-gencode=arch=compute_90a,code=sm_90a`. It never sets `TORCH_CUDA_ARCH_LIST`, so torch will add its plain sm_90 gencode as it did for prefill. The prefill build had `-arch=sm_90a` and still got the extra pass, so an explicit `-gencode` in the flags gives no reason to expect torch to skip its own. Two outcomes are possible:

- **CUTLASS guards its sm90 collectives with `CUTLASS_ARCH_MMA_SM90A_ENABLED` (`__CUDA_ARCH_FEAT_SM90_ALL`).** Then the sm_90 pass may *compile* to a stub whose device code prints "Arch conditional MMA instruction used without targeting sm90a" and/or traps. The build "succeeds". If the driver loads the sm_90 image, the self-check trips a trap, and that is a sticky CUDA error the Python fallback cannot recover from. The whole run fails, not just the layer (uncertainty (d) in its notes).
- **Any unguarded wgmma, setmaxnreg or TMA asm in the instantiation.** Then the build FAILS like prefill, prints `[streamk] build FAILED`, and the check measures master.

**Change before dispatch:**
1. In `_build_streamk()`, set `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"` before each `load_inline`, and restore the old value afterwards. Keep the explicit `-gencode=arch=compute_90a,code=sm_90a`.
2. Print the arch flags nvcc actually received, for example via `torch.utils.cpp_extension._get_cuda_arch_flags()` in a try/except, so the log shows no plain `compute_90` entry.
3. Add a cheap device-side guard. One option is a `static_assert` or `#error` inside `#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)` in `_STREAMK_CUDA`, so a stray sm_90 pass fails the *build*, which falls back cleanly, instead of trapping at run time. The other option is an `#else` that writes an error word, which the self-check reads.
4. At harvest, grep the log for `build OK`/`build FAILED`, ptxas `.target 'sm_90'`, `DISABLED` and `PATH CHOSEN`, and check that the profile has a `cutlass::device_kernel<...>` row on the m=256 rows before reading any number.

With the fix applied, the section 4 conclusion stands: worth its check.

---

## 01:04Z harvest (2026-10-25)

Master for comparison is BEST `bf16-restack-m16-swapab-router-swapab-m32` (check job 1e90292c7b954d96, 1.2215x). Note that `hypotheses/BEST.result.json` holds a different, stale job (366189f0f1114da1, 0.902x), so read the file named in BEST.json `result`.

### bf16-lossless-exp4-occupancy-dequant-once-stage2 — job f1024212dd2f49f3, neutral 0.887x [0.874, 0.900]: RAN, LOST. DEAD.

**What ran (log):**
- Build OK: `[lossless] load_inline compiled in 44.5s`. There is no arch line. The master swapab kernels report wgmma=8-16, mma.sync=0, no spills.
- `_lossless_swapab_kernel<8/16/32>`: 99-117 regs, 0 B local, 115-165 KB dynamic smem (8 stages), 8 warps/CTA, **occupancy 1 CTA/SM**. The grid is 108 CTAs (qkv_gate) or 128 (pooler, unpooler), which gives 6.55-7.76 warps/SM achieved.
- Encode: 12.17 bits/element (ratio 0.761, 0.28-0.31% escape groups). The round trip through the kernel decoder is bit-exact for all three layers.
- Self-check: every launch is ok. `PATH CHOSEN route=qkv_gate|pooler|unpooler: lossless (m 1..32)`. No fault.
- Prepare timing, eager and including launch:
  - qkv m=1: 20.6 µs warm (1.40 TB/s compressed, 1.84 raw-equivalent), 30.1 µs cold
  - pooler m=1: 33.1 µs warm (1.54 TB/s), 40.3 µs cold
  - unpooler m=1: 31.3 µs warm (1.63 TB/s), 38.6 µs cold
- **`KILL-GATE pooler m=1 cold-L2 40.32 us vs gate 19.0 us: MISS`**.

**Per-shape against master.** These are profile kernel µs, lossless vs master `_bf16_swapab_kernel`:

| layer | decode_1 (m=1) | verify_1 (m=8) | decode_16 | decode_32 | shape Δ vs master |
|---|---|---|---|---|---|
| qkv_gate | 18.9 vs 9.5 | 19.2 vs 9.9 | 22.2 vs 10.1 | 25.7 vs 11.8 | +85% to +103% |
| pooler | 30.8 vs 18.6 | 31.0 vs 19.7 | 34.7 vs 19.6 | 39.6 vs 20.7 | +65% to +94% |
| unpooler | (not routed: splitk 4.4) | (splitk 4.5) | 32.7 vs 19.3 (nvjet) | 34.3 vs 18.7 | +74% / +90% |

- Stage 2 did improve on stage 1 (job 542f2b169c934706): qkv went 22.5→18.9 µs and pooler 40.6→30.8 µs at m=1. It is still 1.7-2.1x master on every routed shape.
- The m≥128 shapes are flat, within ±1%.
- **The branch is not on current master.** The o and small-m unpooler rows run `_pooler_skinny_kernel` / `_bf16_splitk_kernel`, not master's `_bf16_swapab_kernel`, so even untouched rows are 4-10% slower (o decode_1 6.5 vs 5.9 µs, router +4-10%).
- The headline therefore mixes two losses: the lossless kernel and a stale base.

**Verdict: DEAD.** The kill gate missed by 2.1x, and the kernel moves 1.4-1.6 TB/s of compressed bytes against master's ~3.6 TB/s raw. Entropy decode at 1 CTA/SM cannot make up the 24% byte saving. Do not mint stage 3.

## 03:41Z harvest

References:
- **Master run 1602a4e947e64090** (`bf16-m32-swapab-wgmma-triton`, 1.1984x) is the baseline.
- BEST.json's 1.2215 is the unlanded restack (`bf16-restack-m16-swapab-router-swapab-m32`, 1e90292c). That is used only for stacking estimates.
- The attribution files say base = restack, but the profiles show master's kernels: `_pooler_skinny_kernel` on o and `_router_hilo_kernel` on router, where the restack has `_bf16_swapab_kernel` and `_router_swapab_kernel`. **All nine runs were on master.**
- **Batch drift:** untouched small-m rows read +2% to +6% slow in every run, on identical kernels. qkv_gate decode_1 read 13.2-13.6 vs 12.91 in all nine. So the 1.17-1.20 headlines are master-level, and only a target row whose kernel changed counts.

### Wins worth stacking (per target row vs master, kernel changed)

| run | shape | master µs | new µs | Δ | z | kernel |
|---|---|---|---|---|---|---|
| router-cuda-cluster-dsmem | router decode_1 (m=1) | 5.73 | **5.24** | −8.7% | −16.3 | cluster 3.6 vs hilo 3.2 + elementwise 0.7 (one kernel instead of two) |
| | router verify_1 (m=8) | 5.91 | **5.08** | −14.1% | −30.7 | 4.3 vs 3.2 + 0.7 |
| | router decode_16 (m=16) | 6.03 | 5.87 | −2.8% | −6.0 | 4.6 vs 3.5 + 0.7; **but restack's swapab reads 5.47 here** |
| o-unpooler-slice-cluster | unpooler decode_1 (m=1) | 5.94 | **5.02** | −15.5% | −32.7 | `oun_kernel` 4.4 vs splitk 4.5 |
| | unpooler verify_1 (m=8) | 6.13 | **5.56** | −9.4% | −20.9 | 4.8 vs 4.6 |
| pooler-unpooler-m128-nn-layout | pooler verify_16 (m=128) | 25.55 | **22.98** | −10.1% | −29.3 | nvjet NN splitK 20.4 + reduce 3.0 vs TN 25.7 |
| | unpooler verify_16 (m=128) | 19.79 | 19.20 | −3.0% | −9.0 | NN 19.2 vs TN 20.4 |

The router-cluster and oun kernels also beat the restack rows at m=1 and m=8 (router 5.38/5.47 and unpooler 6.15/6.00).

Substitutions into the restack rows:

| stack | estimated headline |
|---|---|
| restack (base) | 1.2215 |
| + router cluster at m=1, 8 | 1.2242 |
| + oun unpooler at m=1, 8 | 1.2270 |
| + NN layout at m=128 | 1.2302 |
| **all three** | **≈1.238** |

The same three on master give 1.208. **Next:** after the restack lands, run one check of the three gated pieces: router cluster m≤8, oun on the unpooler only at m≤8, NN at m=128 only.

### Per-slug verdicts

- **router-cuda-cluster-dsmem-splitk-rewrite** (dc5d03c4, 1.1897):
  - Ran: cs=8 NT 8/16 at m≤16, 162-168 regs, 0 spill. Master Triton above m=16.
  - Won at m=1/8 (table) and lost to the restack's swapab at m=16 (+7.1%).
  - The m=32 row (6.85) fell back to master hilo, not the restack's m=32 swapab.
  - **STACK at m≤8 only.**
- **o-unpooler-slice-cuda-cluster-dsmem-rewrite** (3f3330f7, 1.1710):
  - Ran on both layers: cl=8, 68/56 regs, 0 spill.
  - The prepare gate appears to have used warm eager times (o m=1: cluster 8.1 vs master 14.8 µs). Cold, the cluster was slower on o (13.1 vs 12.2) and only slightly faster on the unpooler (10.3 vs 11.1).
  - The judge: O lost at m=1/8/16 (+14% / +34% / +48%; `oun_kernel` 5.2/6.0/7.6 vs `_pooler_skinny` 4.6/5.2/4.8). The unpooler won at m=1/8.
  - **STACK the unpooler half only.** Gate on cold-L2 timings.
- **o-swapab-m32-splitk2-two-ctas-per-sm** (75f09fa6, 1.1860):
  - Ran: o m=32 `_bf16_swapab_kernel` 7.3 vs 4.5 µs, shape +40.8% (z+69).
  - Split-K=2 at two CTAs/SM doubles the reduction and halves the per-CTA stream for a K=1024 GEMM that is already tiny.
  - **DEAD.**
- **qkv-swapab-m32-splitk8-stages3** (a61778a0, 1.1963):
  - Ran: qkv_gate m=32 rose 13.69 → 15.64 (+14.2%, z+27). The kernel row is flat (12.4 vs 12.1), so the loss sits outside the GEMM, in the split-K=8 reduce/semaphore tail.
  - **DEAD.**
- **unpooler-full-m16-swapab-route** (05782267, 1.1885):
  - Ran: unpooler m=16 is on `_bf16_swapab_kernel` (18.4 µs) vs master nvjet (18.5). Shape 18.48 vs 18.61 (−0.7%, z−1.9).
  - The m=16 unpooler is bandwidth-bound and cuBLAS is already at that bound.
  - **DEAD** (neutral).
- **pooler-unpooler-m128-nn-layout-gated** (0daf3faa, 1.1953):
  - Ran at m=128 (table). At m=256 the pooler stayed on the TN kernel: 31.0 vs 29.5 µs profile, +2.4%, drift.
  - **STACK at m=128.**
- **bf16-lossless-exp4-decode-only-throughput-probe** (9148142c, 1.1904): **DEAD.** The verbatim line:
  > `DEAD: best decode-only cold 29.41 us (W16xC2, 64 regs, 16 B spills; 1.74 TB/s compressed) = 1.021x yardstick 28.80 us (REVIVE needs <= 0.85)`

  - The copy-only kernel reaches 0.770x of the yardstick, so the bytes are cheap; the entropy decode alone eats the whole 24% saving.
  - Scored rows are master's (probe only).
  - Close the lossless line.
- **bf16-verify-deepgemm-bf16-gemm-nt-route** (2d522d47, 1.2027): **DeepGEMM LOADED but did NOT run in the timed path.**
  - `deep_gemm 0.2.1.post1+muse.cbe1727` imported and `bf16_gemm_nt` was found.
  - Every layer at m=128/130/200/256 passed (rel L2 1.66e-3, graph ok), and the log printed `PATH CHOSEN ... deepgemm` for all 16 (layer, m) pairs.
  - However, every m=128/256 profile row is an nvjet cuBLAS kernel with master's tile names and times: pooler 24.5 / 29.5, unpooler 20.1 / 24.5, qkv 13.5+3.0 / 19.4, o 5.6 / 8.2. No DeepGEMM kernel appears.
  - So either the route never reached the timed `kernel_fn`, or this build's `bf16_gemm_nt` dispatches to cuBLASLt. The pooler verify_32 −5.1% is the same nvjet kernel (29.5 = 29.5), i.e. drift.
  - **UNMEASURED.** Next: print the profiler kernel name from a one-call probe in prepare, and assert it is not `nvjet`, before spending a check.
- **pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite** (cd39d36e, 1.1828): **BUILD FAILED.**
  - `#error C++20 or later compatible compiler is required to use PyTorch`, then `[bulk] DISABLED: no extension; master paths`. The 1.183 is master plus batch drift.
  - The host `main.o` compile in the tail does carry `-std=c++20`. The failing step is therefore the nvcc compile of the .cu, which was given only `-gencode=arch=compute_90a,code=sm_90a`.
  - Next: add `-std=c++20` to that TU's `extra_cuda_cflags`, then rerun as a check. The design is unmeasured, not dead.

## 04:36Z harvest

References: master `bf16-restack-m16-swapab-router-swapab-m32` (check 1e90292c7b954d96, 1.2215). Shape err is 20-60 ns. Untouched small-m rows drift by ±3-8% in this batch (o verify_16 +5.8-7.9% in three runs on master's own nvjet kernel), so I judge each run on the paired ratio of its targeted rows plus the kernel name in the profile. Three of the four runs were really built on the restack: `_bf16_swapab_kernel` and `_router_swapab_kernel` appear on the untouched rows.

### restack-router-cluster-dsmem-m8: job d323ce8ef6a54fde, check 1.2199 [1.202, 1.238]. LAND-CANDIDATE (stack with the two below).

- `[router_cluster] PATH CHOSEN router m=1..8: CUDA cluster kernel router_kernel<8,8>, no memset`. m 9..32 stays on the swap-AB Triton kernel and m > 32 on hilo. Build: TORCH_CUDA_ARCH_LIST=9.0a, 162 regs, 0 spill, max active clusters 30. The self-check passed at m 1/5/8.
- Profile: router decode_1 and verify_1 show `router_kernel<8, 8>` as the only kernel. The `vectorized_elementwise` memset node is gone.
- Router decode_1 went from 5.38 to 4.98 µs (−7.5%, z −13, pr 1.078). Its per-shape speedup is 2.076, above the 1.954 bar.
- Router verify_1 went from 5.47 to 4.93 µs (−9.8%, z −19, pr 1.099).
- The other router rows are flat or within this batch's drift (decode_16 +2.8%, verify_16 +2.8%, both on unchanged kernels).
- The two targeted rows moved by more than the predicted −2.6% and −14%. Headline estimate +0.15-0.4% alone.

### restack-unpooler-slice-cluster-dsmem-m8: job e557d92969864449, check 1.2260 [1.208, 1.244]. LAND-CANDIDATE.

- The gate printed WIN at both sizes:
  - `gate m=1: cluster 10.21 vs swapab16 10.75 us, net 5.22 vs 5.76, 1.104x (need >= 1.02) -> WIN`
  - `gate m=8: 1.093x -> WIN`
- `PATH CHOSEN: unpooler slice m<=8 (n=1024) -> CUDA cluster/DSMEM split-K, cluster dim 8`. `ouns_kernel` has 56 regs, 0 spill and 72 KB dynamic smem.
- Unpooler decode_1 went from 6.15 to 5.34 µs (−13.2%, z −24, pr 1.136). Unpooler verify_1 went from 6.00 to 5.41 µs (−9.8%, z −20, pr 1.124). Both rows run `ouns_kernel`.
- The other rows are within the batch drift: o decode_32 +4.5% and o verify_16 +6.5%, both on unchanged kernels.
- This gate is a model selector: it printed both medians, used graph replay with a cold flush, and demanded a ≥1.02 margin. Headline estimate +0.3-0.8% alone.

### restack-pooler-unpooler-nn-layout-m128: job 92cd4e758a924ca0, check 1.2282 [1.210, 1.247]. LAND-CANDIDATE.

- Pooler verify_16 (m=128) went from 25.00 to 22.59 µs (−9.6%, z −25), **pr 1.117 ≥ 1.06**. The kernel is `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_splitK_NNT` plus splitKreduce.
- Unpooler verify_16 (m=128) went from 19.96 to 18.99 µs (−4.9%, z −13.5), **pr 1.047 ≥ 1.02**. The kernel is `nvjet_..._h_bz_NNT`.
- Both bars are met, with the NN layout confirmed by kernel name. The archived log tail holds no PATH CHOSEN line for the layout, so the kernel names are the evidence.
- m=256 rows are flat (pooler −0.1%, unpooler −0.5%), as expected.
- Unpooler decode_1 −11.5% is not this change: it runs the same `_bf16_swapab_kernel`, and master's 6.15 is the high reading among four runs (5.34-6.15). o verify_16 +5.8% is the batch-wide drift.
- Headline estimate +0.5-0.9% alone.

**Restack proposal (one check, then submit):** router cluster at m≤8, unpooler-slice cluster at m≤8 and the NN layout at m=128, all on master. They touch disjoint (layer, m) cells, and none of the three edits the others' gate constants. The substituted estimate is about 1.238-1.243. The CIs are ±1.5%, so promotion must rest on the six rows: router decode_1/verify_1, unpooler decode_1/verify_1, pooler/unpooler verify_16.

### pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild: job 571ea532ad7a410d, check 1.033, regression. DEAD (the line's own kill gate says so); one cell is worth salvaging.

- **It compiled:** `extra_cuda_cflags: ['-O3', '-lineinfo', '-std=c++20']`, gencode sm_90a only, load_inline in 37.3 s. The weight repack is bit-exact, and the self-checks passed at m 1..32 (rel L2 1.65e-3, error word 0).
- **Path that ran:** `PATH CHOSEN layer=pooler m 1..32: bulk-TMA row-owner stream` and `layer=unpooler m 9..32 (n=8192): bulk-TMA row-owner stream`. The profile confirms `bulk_kernel<32, 8192, NT>` and `bulk_kernel<64, 4096, NT>`.
- The base was OLD master (`_pooler_skinny`, `_router_hilo`, `_qkv_splitk_sem`). That is why qkv, o and router read +5-11% against the new master. Ignore those rows.
- **Why it was slow:**
  - The kernel's time grows linearly with m. Prepare measured pooler at 27.1, 28.1, 32.5 and 51.9 µs for m=1, 8, 16 and 32, against a pure-read time of 26.5 µs. The compute gap was +0.6, +1.6, +6.1 and +25.5 µs.
  - Unpooler showed the same pattern: +6.1 µs at m=16 and +28.2 µs at m=32.
  - The design runs one CTA per SM (160 threads, 128 CTAs, 124-200 KB smem), so per-token FMA work in the row-owner epilogue cannot be hidden behind the stream.
  - Even the pure stream reached only 2.54 TB/s cold, against the 3.8 TB/s kill gate: `KILL-GATE ... MISS -> line DEAD; route kept on`. The judge then saw pooler m=16/32 at +43% and +145%, and unpooler m=16/32 at +49% and +180%.
- **The one positive cell:** pooler decode_1 (m=1) went from 18.84 (master swapab) to 17.28 µs (−8.3%, z −24, pr 1.094), on `bulk_kernel<32,8192,8>` at 17.0 µs. Verify_1 (m=8) already loses (+5.1%).
- Optional single-cell follow-up: bulk-TMA pooler at m=1 only, on the restack. It is worth about +0.1-0.2% headline. Run it as a STACK-INTO-NEXT add-on only if the restack above has room in its check; it does not need its own check.
