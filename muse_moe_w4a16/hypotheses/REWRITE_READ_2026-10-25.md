# muse_moe_w4a16: per-shape read of the 2026-10-24 kind:rewrite runs

These are read-only reads of recorded runs. There was no GPU here. Each rewrite is judged on its replaced kernels'
profiled time (`top_kernels` / `*.profile.txt`) against the incumbent (`attribution.base_result`) and the floor (SOL).
The headline is not the measure.

Caveats that apply to every section:
- The profile is a single call. Marlin's profiled time moves by up to about ±20% between runs on identical code.
  The wgmma run fell back to master and is effectively a master rerun: its Marlin row at bs=128 was 333.5 µs against
  316.1 µs in the base, and at bs=256 it was 403.1 µs against 490.2 µs. Treat a Marlin-row delta of less than about
  20% as noise.
- `parent_runs_ns` in the okc records is **not** a master control. It tracks each run's own submission (for example,
  split-K reads 130.9 µs parent against 128.2 µs hyp). Do not use it for attribution.
- Build check, as the coordinator asked: I grepped every archived log for `ptxas`, `not supported on .target` and
  PATH CHOSEN. Only the wgmma run hit the cpp_extension `compute_90` gencode failure. The other three CUDA builds
  compiled (41 to 51 s) and contain no sm_90a-only instructions.

| rewrite | build | path that ran on the target shapes |
|---|---|---|
| cuda-dispatch-combine-pdl-verify | OK, 41.4 s | **CUDA** align+combine with PDL, 32 < M <= 512 |
| cuda-nvfp4-swapab-mma-persistent-small-m | OK, 51.2 s | **none**: illegal instruction in the setup self-check, then prepare() crashed |
| cuda-nvfp4-wgmma-rega-grouped-verify | **FAILED** (ptxas: wgmma/setmaxnreg "not supported on .target 'sm_90'") | **MARLIN** (fallback) |
| cuda-fused-moe-bs1-persistent-w13-silu-w2 | OK, 44.2 s | **fused**, bs=1 |
| small-m-grouped-gemv-splitk-rewrite | Triton | **TRITON** split-K GEMV, M <= 16 |

---

## 1. cuda-dispatch-combine-pdl-verify: DEAD (stakes below resolution)

**Path:** `build OK (41.4 s)`. Self-check PASSED: align byte-identical in 330 cases, e2e rel-L2 3.15e-3 eager and
graph+PDL. PATH CHOSEN = cuda align+combine (pdl on) for bs=33/128/256/257. bs<=32 ran master, as designed. Combine
kernel: numRegs 160, no spill.

The replaced kernels are `_align_parallel_kernel` and `_post_expert_norm_weighted_sum_kernel`. Base is
align-parallel-per-expert-verify.

| shape | incumbent align + combine (µs) | new align + combine (µs) | sum, new vs incumbent | M3 target (A / B) | floor (sum) | scored base -> hyp |
|---|---|---|---|---|---|---|
| bs=128 | 4.0 + 4.9 = 8.9 | 3.3 + 3.7 = 7.0 | -1.9 µs (M2 yes) | 1.8 / 2.5 (no / no) | ~2.5-4 | 318.3 -> 319.3 |
| bs=256 | 6.9 + 6.0 = 12.9 | 3.9 + **6.2** = 10.1 | -2.8 µs (M2 yes on sum; the combine alone is slower) | 2.0 / 3.0 (no / no) | ~3-5 | 454.3 -> 457.1 |

Prepare-time warm-L2 event timings point the same way. At M=128, align is 3.24 vs 3.88 µs and combine 3.08 vs
3.50 µs. At M=256, align is 4.09 vs 6.34 µs and combine is **4.75 vs 3.72 µs (slower)**.

**Where the time went:** kernel A did beat the 33-program Triton align, but it is still about 2x its floor. That is
the single 1024-thread CTA's scan and launch latency. Kernel B, at 160 registers per thread with 8 slots x 2 x 16 B
held, does not beat SGLang's combine at bs=256. This repeats the earlier finding from post-norm-triton-all-m.

**Verdict: DEAD as a rewrite line.** It met M1 and met M2 on the sum, but the whole remaining headroom is about
4-6 µs per shape on 2 of 6 loads, about 0.3% of the headline. That is far inside the ±1.5% CI and inside Marlin's
run-to-run noise on the same shapes. Optional: fold kernel A (−0.7 / −3.0 µs) into a later bs>=128 rewrite as a
free part. Do not mint stage 2.

---

## 2. cuda-nvfp4-swapab-mma-persistent-small-m: FIXABLE (bug, not design)

**Path:** the build was OK (51.2 s). All 6 instantiations had attrs ok, 80-128 registers, localSizeBytes 0 and
88-136 KB smem. The retile was lossless, and the `_align_small` check at BSM 8/16/32 had 0 hard mismatches. The
**first CUDA launch** in `_cuda_self_check` failed: no per-M self-check line was printed, so it died at M=1, NT=1,
on w13+silu or w2. The error was `CUDA error: an illegal instruction was encountered` (715, sticky). Setup caught the
exception and printed `PATH CHOSEN = MARLIN`. Because the context was already poisoned, prepare()'s next
`torch.cuda.synchronize()` (submission.py:1004) re-raised. The worker died, with returncode 112 and **tests 0/0**.
None of the 5 tests failed on numerics; prepare never completed. Nothing was measured on any shape.

| shape | incumbent Marlin + silu (µs, base profile) | new | floor | M3 target |
|---|---|---|---|---|
| bs=1 | 44.6 + 1.6 | not run | 11.4 | 18 |
| bs=8 | 105.8 + 2.0 | not run | 37.3 | 56 |
| bs=16 | 116.0 + 2.2 | not run | 52.5 | 79 |
| bs=32 | 187.6 + 2.3 | not run | 70.8 | 106 |

**Two bugs:**
1. **The fallback cannot survive a device fault.** An illegal instruction or address is sticky for the process, so
   an in-process try/except "fallback" turns any kernel fault into a crashed run. Fix: run the CUDA self-check's
   launches in a spawned child process that uses the same load_inline build directory, and enable the path only if
   the child exits 0 with PASSED. Alternatively, make the very first launch a tiny canary in a child process. The
   bs=1 rewrite's "check the error flag after the first launch" does not help here, because a trap is not a flag.
2. **The illegal instruction itself.** The log cannot pin it to a line. The sibling bs=1 kernel ran cleanly on the
   judge with mma.sync m16n8k16 bf16, `cvt.rn.f16x2.e4m3x2`, `fma.rn.bf16x2` with a -0 addend, plain
   `cp.async.cg ... 16` and the same dequant bit trick, so those are cleared. This kernel's instructions that have
   **never executed on the judge** are `createpolicy.fractional.L2::evict_first` with
   `cp.async.cg.shared.global.L2::cache_hint`, `cp.async.ca ... 16, src_bytes` (zero-fill), and `ldmatrix.x4`.
   muse_decode_layer `moe-chain-cuda-swapab-expert-once-large-t` uses the same helper set and died with the same 715
   error. So the fault is in the shared helpers or the shared structure, not in this problem's indexing. It is not the
   `compute_90` gencode issue: nothing here is sm_90a-only, and the build succeeded.
   Stage-2 fix: (a) swap `cp_async16_ef` for the bs=1 kernel's proven plain `cp.async.cg` and drop `createpolicy`;
   (b) build with `-lineinfo`; (c) have the child-process canary run w13 and w2 separately at NT=1, so the log names
   the faulting instantiation. If it still traps, the next suspect is ldmatrix/zfill. Then fix the slab or zero-fill
   path, or temporarily replace ldmatrix with 32-bit `ld.shared` B-fragment loads.

**Verdict: FIXABLE.** The review audited the design, and the offline emulation matched to 1e-15. Stage 2 is "make
it run": the child-process self-check plus removal of the cache-hint path. Performance is unknown. The closest
evidence is §4: the bs=1 kernel, with the same mainloop family, reached only about 1.5 TB/s.

### Hold cuda-fused-moe-midm-single-launch-routed? **Yes, hold.**
Its mainloop and bank are this kernel's, and that kernel has never executed on the judge. Its in-process fallback has
the same sticky-fault hole. On top of that it adds grid-wide flag/spin dataflow, which §4 shows can go bimodal. Hold
it until the swap-AB stage 2 harvests with `PATH CHOSEN = CUDA` and per-shape rows that reach at least M2. When it
is released, its spec must adopt (i) the child-process self-check, (ii) the review's flag fix (every flag advances by
exactly 8 per call, or a reset by the exit ticket) with bounded spins, and (iii) whatever helper fix stage 2 lands.

---

## 3. cuda-nvfp4-wgmma-rega-grouped-verify: UNMEASURED (build)

**Path:** `load_inline failed`. ptxas ran on `...compute_90.ptx` and reported `wgmma.mma_async / wgmma.fence /
wgmma.commit_group / wgmma.wait_group / setmaxnreg.inc|dec not supported on .target 'sm_90'`. cpp_extension's JIT
path appends its own `-gencode arch=compute_90,code=sm_90` from the detected arch beside the user's `-arch=sm_90a`,
and the plain sm_90 pass cannot assemble wgmma. Setup printed `PATH CHOSEN = MARLIN for 128 <= num_tokens <= 512`.
The run measured master on every shape.

| shape | incumbent Marlin + silu (base) | this run (Marlin, fallback) | new kernel | floor | M3 target |
|---|---|---|---|---|---|
| bs=128 | 316.1 + 3.9 | 333.5 + 3.8 | **not run** | 101.3 | <= 170 |
| bs=256 | 490.2 + 6.4 | 403.1 + 5.8 | **not run** | 108.9 | <= 220 |

**Verdict: UNMEASURED (build). Fix: `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"` before `load_inline`, and guard
the wgmma/setmaxnreg bodies with `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`** (a stub or trap in the other branch), so
any stray compute_90 pass still assembles. Re-run as stage 1, not stage 2, since nothing has been learned yet. The
same latent issue applies to any future CUDA build here that uses sm_90a-only instructions. dispatch-combine,
swap-AB and bs=1 did not use any. Its Marlin rows are this report's evidence for the ±20% Marlin profile noise.

---

## 4. cuda-fused-moe-bs1-persistent-w13-silu-w2: PROMISING (stage 2 required)

**Path:** compiled in 44.2 s; 72 registers, 0 local, 158 KB smem, 8 stages, 1 block/SM, grid 132. SELF-CHECK PASSED:
50 eager runs and 10 graph replays, R in {0,1,2,3,8}, max rel-L2 vs master 6.0e-3, and against an fp32 dequant
2.9e-3 (master's is 4.9e-3). `PATH CHOSEN = fused`. The bs=1 test passed at 3.2e-3. All other shapes ran master.

The replaced kernels are `_align_small`, the Marlin pair, `act_and_mul` and `_post_norm_combine`, against
`moe_bs1_kernel`. Base is triton-map-align-on-block-m-restack.

| shape | incumbent replaced kernels (µs) | fused kernel (µs) | vs incumbent | floor | x floor | M3 target | scored base -> hyp |
|---|---|---|---|---|---|---|---|
| bs=1 | 44.6 + 3.4 + 1.6 + 1.6 = 51.2 | **37.7** | -26% (M2 yes) | 11.4 | 3.3x (~1.46 TB/s on ~55 MB) | <= 18 (no) | 48.4 -> 50.9 (flat) |

**Why the scored number is flat when the kernel row is 26% better:** the 100 timed replays are **bimodal**.
- Fused kernel: 41 replays in 40-47 µs and 59 replays in 64-79 µs. The raw median is 66.1 µs.
- Master at bs=1 in the two other runs from that day is unimodal at 48-68 µs, with a median of about 55 µs.
- The SGLang baseline in this same run is unimodal, at 76-97 µs.

So the kernel is about 15% faster than master in its fast mode, which matches the profile's 37.7 µs, and about 25%
slower in its slow mode, which is about 24 µs extra. The seed is fixed per case, so this is the kernel's pathology,
not input variance. The likely cause is the grid-wide phase-2 spin. Phase-2 CTAs spin with `ld.acquire` and
`__nanosleep(32)` on w13 chunk counters. With R≈2 there are 256 phase-1 items over 132 CTAs, so phase 2 for route 1
waits on second-wave phase-1 items. Late or unlucky CTA scheduling, or a nanosleep wake-up overshoot, adds a whole
chain stall.

**Verdict: PROMISING.** It met M1 and met M2 on the kernel row. Stage 2 must:
1. **Kill the slow mode.** During the self-check, record per-CTA `clock64`/`globaltimer` timestamps at the phase
   boundaries and the spin-iteration counts into a debug buffer. Then remove the dependence: replace `__nanosleep`
   with a tight acquire poll, and order items so each phase-2 (route, k-split) item is taken by CTAs whose phase-1
   chunks finish first. Or split K so phase 1 is exactly one wave: 128R items padded to a multiple of 132.
2. **Then push bandwidth** from about 1.5 TB/s toward 3 TB/s, the <=18 µs M3. The stated knobs are a deeper ring,
   2 m16 tiles per warp, and phase-1 K-split when R is 1-2. Dequant ALU is about 27 ops per k16 per lane, near the
   per-stage DRAM time, so overlapping it is the second lever.

Even only removing the slow mode lands bs=1 at about 42 µs, about +15% on that load.

---

## 5. small-m-grouped-gemv-splitk-rewrite: DEAD (design)

**Path:** SELF-TEST PASSED (rel-L2 1.7e-3). `PATH CHOSEN for num_tokens <= 16: TRITON split-K grouped GEMV`. Grid 264
(2 per SM). **shared=8192 B and 211 registers for both kernels**, with BLOCK_N 64, SPLIT_K 4/8, BLOCK_K 256 and 4
"stages". Base is sglang-marlin-path-parity, the older base.

| shape | incumbent Marlin pair (µs) | split-K GEMV pair (µs) | vs incumbent | floor | x floor | target | scored base -> hyp |
|---|---|---|---|---|---|---|---|
| bs=1 | 41.1 | 84.1 | 2.05x slower | 11.4 | 7.4x | <= 20 | 80.2 -> 137.4 |
| bs=8 | 107.6 | 344.7 | 3.2x slower | 37.3 | 9.2x | <= 60 | 152.0 -> 355.4 |
| bs=16 | 150.9 | 345.8 | 2.3x slower | 52.5 | 6.6x | <= 85 | 191.4 -> 491.7 |

The bs=32/128/256 controls ran Marlin and scored flat, within noise.

**Where the time went:** shared=8192 B means Triton did not software-pipeline the per-item K loop: only one
BLOCK_K tile is resident. Each item's 2-4 k-steps therefore run serially at full DRAM latency. 211 registers also cap
occupancy. bs=8 and bs=16 cost the same (345 µs), so the kernel is latency- and issue-bound, not byte-bound. The extra
fp32 split-K partial and reduce plus the at::native glue (about 25 µs at bs=1) add to that.

**Verdict: DEAD.** This was the second Triton small-M W4A16 GEMV to lose about 2x or worse to Marlin, after
triton-w13-grouped-gemv-small-m at 0.751x. The cross-item cp.async ring and the ~1.5 op/weight dequant this needs
are exactly what the CUDA swap-AB and fused-bs1 kernels were built for. Do not mint a Triton stage 2.

---

## 01:04Z harvest (2026-10-25)

Master for comparison is BEST `align-parallel-per-expert-verify`:
- submit 1947e3ec2abb4bdf: 1.3278x
- check 45eacf4ca2e849d2: 1.3249x [1.305, 1.345]. This is the result file named in BEST.json. `BEST.result.json` is a stale reference job (5e7d55323d75425e, 0.006x).

Both archived `.log` files are exactly 15022 B and start mid-line. They are the tail of the run output, and the top of the align dump is cut off.

### cuda-nvfp4-swapab-mma-persistent-small-m-inproc — job fecc63ef567b4dc2, regression 0.925x [0.911, 0.939]: RAN at T≤32, LOST 1.7-2x to Marlin. DEAD.

**What ran (log):**
- Build: `nvcc flags ['-O3', '-arch=sm_90a', '-lineinfo']`, `load_inline compiled in 42.8s`, device H200 (9, 0).
- Kernels w13+silu and w2 at NT=1/2/4: 80-128 regs, 0 B local, 90-139 KB dynamic smem.
- Grid 132. Split-K: w13 {1:4, 2:2, 4:1}, w2 {1:8, 2:4, 4:2}.
- Probes 0-3 (cp.async.cg/ca, ldmatrix.x4, cvt e4m3x2 + fma bf16x2 + mma.sync m16n8k16) all ok with the expected values.
- Canary w13/w2 ok and finite. **The illegal-instruction fault of the earlier version is gone.**
- `SELF-CHECK PASSED: max rel-L2 9.244e-03 (bound 1e-2), tickets zero True`. This is close to the bound: 8.8-9.2e-3 at every M.
- `PATH CHOSEN = CUDA for num_tokens <= 32; MARLIN above`. No gate or timing lines.

**Per-shape against master (check 45eacf4c).** Kernel µs are new `w4a16_kernel` w13 + w2 vs master Marlin:

| shape (T) | new µs | master µs | Δ | kernels new vs Marlin |
|---|---|---|---|---|
| decode_1 (1) | 68.6 | 48.8 | +40.6% (z 11) | 44.7 + 31.2 = 75.9 vs 44.7 |
| verify_1 (8) | 218.0 | 107.7 | +102% (z 25) | 130.7 + 88.4 = 219.1 vs 114.0 |
| decode_16 (16) | 267.1 | 148.4 | +80% (z 23) | 150.9 + 90.7 = 241.6 vs 176.1 |
| decode_32 (32) | 368.8 | 199.9 | +85% (z 25) | 255.2 + 160.3 = 415.5 vs 207.5 |
| verify_16/32 (Marlin) | 324.6 / 460.7 | 318.3 / 454.3 | +2.0 / +1.4% | control, flat |

The kernel lost on every shape it ran. It is slowest relative to Marlin at T=8-32 (2x). A likely cause, not verified from the profile, is that the m16 tiles are mostly padding and the weights are re-streamed per split. Its "1.143x on decode_1" in the attribution is against an old baseline, not master.

**Verdict: DEAD.** This line (with the decode-side `moe-chain-cuda-swapab-expert-once-*`) now runs correctly and is 1.7-2x slower than Marlin. Correctness is fixed; the design is the problem.

### cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc — job b56d17fda2d54be2, neutral 1.330x [1.310, 1.350]: NEVER RAN (canary non-finite). Master was re-measured.

**What ran (log):**
- Build: `TORCH_CUDA_ARCH_LIST='9.0a' -> ['-gencode=arch=compute_90a,code=sm_90a']`, `load_inline compiled in 32.8s`. The sm_90a-only fix worked.
- Kernels:
  - w13+silu: 168 regs, 0 B local
  - w2: 168 regs, **8 B local (spill)**
  - both: 187/204 KB dynamic smem + 8 KB static, 1 CTA/SM
- `cuTensorMapEncodeTiled x 8: rc 0`. Grid 132 persistent × 384 threads, BK 128, 4 stages, N buckets 16-128.
- Canary at M=128 (1 local route per token): every stage of gather, w13, w2 and norm returned, with **`device error word 0 (ok), output finite False`**. Then `wgmma path setup raised: RuntimeError('canary failed')`.
- `PATH CHOSEN = MARLIN` for every T. So this is a numerical bug, not a fault or a build failure. Likely sites are the register-A fragment layout, the TMA descriptor or swizzle, the scale application, or an uninitialised accumulator or epilogue.
- The log does not say which output went non-finite, w13's cache or y.

**Per-shape:** this is master's path. Every shape is within noise of check 45eacf4c (decode_1 −0.3%, decode_16 −1.0%, decode_32 +3.6% z 1.9, verify_1 +6.0% z 2.4, verify_16/32 +1.3/+2.0%). The 1.330 is not a gain over the 1.3287 best.

**Verdict: FIXABLE (bug).** It was never timed. Stage-next should make the canary name the first non-finite tensor (w13 output vs w2 output, and the first bad row/col), and compare one tile against the fp32 dequant reference. Remove the w2 spill (8 B local at 168 regs) too. It is worth one more check only if the per-kernel bar (≤ Marlin at T=128-512) still looks reachable.

---

## 01:36Z harvest

Master for comparison is BEST `align-parallel-per-expert-verify`: submit 1947e3ec 1.3278x; check 45eacf4c 1.3249x [1.305, 1.345] is the result file named in BEST.json. z is Δ / √(err_new² + err_ref²).

All three logs show master's own align self-check line, `tokens > 32: 135 cases, 135 mismatches, 135 capacity-only, 0 other`. That is unchanged master behaviour.

### cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc — job 7a6919420b724ee7, neutral 1.3257x [1.306, 1.368]: RAN at bs=1. The slow mode is REDUCED, NOT GONE. NEXT STAGE.

**What ran (log):**
- The build used TORCH_CUDA_ARCH_LIST=9.0a with `-gencode=arch=compute_90a,code=sm_90a`, `-O3 -lineinfo`, and compiled in 41.0 s.
- The kernel uses 72 regs (DBG 78), 0 local, 158208 B smem, 8 stages, 1 block/SM, grid 132.
- The arch probe reported `__CUDA_ARCH_FEAT_SM90_ALL` defined, and the ARCH GATE was OK (binaryVersion and ptxVersion 90/90).
- Canaries `<false>` and `<true>` (DBG) gave rel-L2 vs master 5.55e-3, 0 spin-bound hits, and counters reset.
- 9 rounds of 5 eager launches and 10 graph replays ran with R in {0,1,2,3,8}. Against an fp32 dequant reference, the fused kernel scored 2.96e-3 / 2.83e-3 and master 4.97e-3 / 4.81e-3.
- The log reports `SELF-CHECK PASSED ... max rel-L2 vs master 6.06e-03` and `PATH CHOSEN = fused for num_tokens == 1`. No fault.
- The prepare DIAG ran 100 cold-L2 graph replays interleaved with master. Fused R=2 had p50 **42.7 µs** (p10 42.2, p90 43.6, max 54.8) against master's 51.8. The run reported it as **UNIMODAL** with p90/p10 1.03. The other R values gave R=1 33.9, R=3 66.2 and R=8 141.6 µs.
- Timeline for the DBG run: phase 1 ends at ~21 µs, the phase-2 first-wait max is 18.7 µs (median 1.25), and spins reach max 38 (median 1).

**Per-shape against BEST:**

| shape | master µs | new µs | Δ% | z | note |
|---|---|---|---|---|---|
| **decode_1 (bs=1, fused)** | 48.8±0.5 | 47.7±1.3 | −2.4% | −0.8 | paired 1.613 vs 1.578 (+2.2%) |
| decode_16 / decode_32 / verify_1 (Marlin) | 148.4 / 199.9 / 107.7 | 146.8 / 199.6 / 108.0 | −1.1 / −0.2 / +0.2% | ≤0.5 | control |
| verify_16 / verify_32 (Marlin) | 318.3 / 454.3 | 327.3 / 467.2 | +2.8 / +2.8% | 2.3 / 2.4 | control, drift |

**The judge's bs=1 runs are still bimodal:**

| run | 40-45 µs | 65-80 µs | other | median |
|---|---|---|---|---|
| this run | 59/100 | 40/100 | 1 at 95+ | 47.7 (raw 43.6) |
| stage 1 | 41/100 | 59/100 | — | — |

Master is unimodal at 50-60 µs. The single profile replay caught the slow mode (61.6 µs). In the replay order the slow runs are scattered and roughly every 3rd-7th one, so this is not warm-up.

**Verdict: NEXT STAGE**, not a land. The nospin change moved the fast-mode share from 41% to 59%, but the slow mode (+23 µs) survives under the judge even though the prepare DIAG is unimodal.

The judge's timed loop differs from the DIAG: a different L2 flush, baseline interleaving, and the replay gate. Stage 3 should keep the per-CTA globaltimer debug buffer in the TIMED instantiation for a sample of replays and dump the slow ones. Fast mode everywhere is worth ~42.5 vs 48.8 µs, about +13% on the decode 1 load.

### post-norm-triton-hoisted-decode32 — job 47d186dd91974d84, neutral 1.3248x: RAN at decode_32. DEAD (below resolution).

This is Triton only, with no CUDA build. The log is the 15 KB tail of the align dump and has no PATH line for this change. The profile shows `_post_norm_combine_hoisted_kernel` at decode_32 only, so the gate held:
- decode_32: kernel 3.8 µs vs SGLang `_post_expert_norm_weighted_sum` 4.5 (−0.7 µs). Shape 199.9 → 201.5 (+0.8%, z 0.4), because Marlin's row is seed-noisy (207.5 vs 220.8).
- Every other shape is within ±1.9%, |z| ≤ 1.5.

0.7 µs per call cannot be seen in the headline. At most it is worth folding into the sibling hoisted kernel for bs<32 as one restack. It does not justify a check of its own.

### post-norm-pdl-launch-small-m — job 5202cf15936c4492, neutral 1.3222x: RAN (`[post-norm-pdl] PATH CHOSEN: pdl on (self-test passed)`). DEAD.

| shape | master µs | new µs | Δ% | z |
|---|---|---|---|---|
| decode_1 | 48.8 | 48.7 | −0.2% | −0.1 |
| decode_16 | 148.4 | 148.2 | −0.1% | −0.1 |
| decode_32 | 199.9 | 199.9 | 0.0% | 0.0 |
| verify_1 | 107.7 | 112.8 | +4.7% | +1.8 |
| verify_16 / verify_32 | 318.3 / 454.3 | 318.4 / 451.1 | +0.1 / −0.7% | ≤0.6 |

The `_post_norm_combine_kernel` row moved 3.1 → 2.8 µs at bs=1, with no shape-level effect. PDL on a ~3 µs tail kernel after Marlin buys nothing measurable.

## 03:41Z harvest

Reference: BEST `align-parallel-per-expert-verify`, using the result file named in BEST.json (check 45eacf4ca2e849d2, 1.3249x; the submit 1947e3ec2abb4bdf gave 1.3278x). z is Δ / √(err²+err²). moe err is 0.4-4 µs.

### w2-marlin-atomic-add-small-m — job a2744bc496b84790, check 1.3445x [1.324, 1.365]: RAN. LAND CANDIDATE (bs=1 only moves).

**What ran (log, 15 KB tail):**
- `[w2-atomic] PATH CHOSEN = atomic (worst rel-L2 vs master 0.00479)`.
- `atomic add enabled for w2 at T<=32: True; use_fp32_reduce=True; zeroing memset in path: False`.
- Note: `use_fp32_reduce=True` differs from the spec ("False, Marlin requires the pair"). Marlin accepted it, and the judge's 5/5 tests passed at rel L2 ≤ 6e-3.
- Pure Python kwargs change, so there is no CUDA build. No fault.

**Per-shape vs BEST (45eacf4c):**

| shape | BEST µs | new µs | Δ | z | Marlin pair (profile) |
|---|---|---|---|---|---|
| **bs=1 (decode_1)** | 48.82±0.6 | **44.88±0.65** | **−8.1%** | **−4.9** | 44.7 → 42.1 µs |
| bs=8 (verify_1) | 107.74 | 106.03 | −1.6% | −0.6 | — |
| bs=16 (decode_16) | 148.37 | 147.71 | −0.4% | −0.2 | — |
| bs=32 (decode_32) | 199.88 | 198.72 | −0.6% | −0.3 | — |
| bs=128 / 256 (control, T>32 untouched) | 318.26 / 454.28 | 317.77 / 467.80 | −0.2 / +3.0% | −0.1 / +2.5 | bs=256 Marlin 451 vs 391: that Marlin row varies 391-451 across today's runs (drift) |

- The raw timed distribution at bs=1 confirms it beyond the adjusted median. raw p50 is 47.3 µs vs 51.4-51.8 in both BEST runs, and p10 46.1 vs 50.0-50.4. The +23 µs slow mode hit 10/100 replays vs 44-45/100 in BEST.
- Every other run today sits at 48.5-49.2 µs at bs=1, so 44.9 is outside that spread.
- decode 1 load: 1.578 → 1.721. Substituting the bs=1 row alone into BEST gives **≈1.344x**; adding the flat bs≤32 rows gives 1.346.
- The headline CI lower bound (1.324) is just under BEST's submit point (1.3278), so a scored submit is the test.

**Verdict: LAND CANDIDATE** (scored submit, needs the user's approval). The predicted gains at bs=8/16/32 did not appear; only bs=1 moved.

### cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix — job 583a73418fbf4d01, check 1.3307x: BUILT, DIAGNOSED, FELL BACK TO MARLIN. FIXABLE, high value.

**What ran:**
- Build: sm_90a only, 37.6 s, 0 ptxas warnings, no wgmma serialization warning.
- w13+silu kernel: 168 regs, 0 B spill. w2 kernel: 168 regs, **8 B spill**. 1 CTA/SM each.
- `cuTensorMapEncodeTiled` ×8 returned rc 0. The device error word was 0 everywhere.
- `PATH CHOSEN = MARLIN` for every num_tokens, so the 1.3307 is master re-measured: every shape is within ±1-4% at z ≤ 2.5.

**Diagnostic summary block: which kernel, which bucket.** The bug is in the token-bucket (M-tile) instantiations of 64 and 128 rows. The weights, scales, gather and N-tiling are correct.

| kernel | N16 bucket | N32 | N64 | N128 |
|---|---|---|---|---|
| w13+silu (h vs Marlin cache2) | ok, rel 6.3e-3 | ok, 6.5e-3 | **every row non-finite** | **every row non-finite** |
| w2 fed Marlin's h (y vs Marlin) | ok, 3.4e-3 | ok, 3.5e-3 | ok, 3.5e-3 | **every row non-finite** |

- The gather is clean: 0/644 bad real rows, pad rows zero, 0 remote-slot writes. ids and coverage are True everywhere.
- In failing buckets the whole row is non-finite. The first bad element is (route 3, token 0, col 9, N-tile 0), and the "worst N-tiles" list contains every tile with the same row count. A per-N-tile or per-K error would not look like that; a per-M-tile (bucket) error would.
- The routings that put every chunk into the 16-bucket pass at every M: `random` at M=128/192/256/257, with h 6.4e-3, y 3.5e-3 and out 6.0e-3, plus the graph replay. `skewed`, `all-local` and `all-one-expert` pull in 64/128 buckets and fail. The fp32-dequant reference also reads inf on those.
- w13 breaks from 64 rows and w2 only at 128. Suspects are whatever differs between the 64/128-row instantiations and the 16/32 ones: the register-A fragment row offset past row 32/64, the second consumer warpgroup's row base, or a per-bucket smem carve-out overlapping a B stage. The smem sizes are w13 8192+187392 B and w2 8192+203776 B. This is inference: the log names the buckets, not the line.
- `fused norm skips remote rows: False` is probably downstream of the NaN rows. Re-test it after the fix.
- **Upside if fixed** (prepare, warm L2, eager): M=128 gives wgmma 7.0 + 346.3 + 162.6 = **515.9 µs vs Marlin 911.3** (0.57x). M=256 gives 728.0 vs 1258.5 (0.58x). The judge reads cold-L2 profiles 20-25% higher, and the log itself warns the time is optimistic if a race releases early.

**Verdict: FIXABLE.** Stage 3 should:
1. Run one-expert M=64 and M=128 cases per bucket, for w13 and w2 separately.
2. Dump h for the first bad row (all NaN vs a scattered inf).
3. Diff the w13 N64 instantiation against w2 N64, which passes.
4. Keep the both-kernels-pass gate.

### cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc — job cd0f6b08a6dd4165, check 1.3295x: RAN at bs=1, LOST to bimodality. DEAD as a path.

- ARCH GATE ok (sm_90a, binary and ptx 90). The canaries, 50 eager + 10 graph replays and the fp32 check all passed: fused 2.8e-3 vs master 5.1e-3 against fp32 dequant.
- `PATH CHOSEN = fused for num_tokens == 1`.
- The prepare DIAG was again unimodal: fused p50 42.0 µs (p90/p10 1.03) vs master 50.7 µs, interleaved, cold L2.
- **The judge's timed loop was bimodal again, and worse.** 53/100 replays were in the slow mode (~66 µs; p50 65.6, p10 41.9). nospin had 41/100.
- Adjusted median 50.41 vs BEST 48.82 (+3.3%, z+1.2). The profile kernel row reads 62.5 µs vs Marlin pair 44.7.
- Two stages have failed on the same mode, which the prepare instrumentation cannot reproduce. The atomic-add Marlin change now takes bs=1 to 44.9 with a 10% slow mode. Stop here.
