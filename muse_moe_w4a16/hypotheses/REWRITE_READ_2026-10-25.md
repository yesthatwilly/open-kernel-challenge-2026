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
