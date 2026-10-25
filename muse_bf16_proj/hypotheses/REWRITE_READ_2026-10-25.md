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
