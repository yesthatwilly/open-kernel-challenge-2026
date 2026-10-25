# muse_bf16_proj · 2026-10-25T002149Z · check · bf16-lossless-exp4-occupancy-dequant-once-stage2

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T001220Z-bf16-lossless-exp4-occupancy-dequant-once-stage2.md` (branch `hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2`).

- **Job:** `f1024212dd2f49f3` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `39098a3`

## Goal
Build on the stage-1 code of bf16-lossless-exp4-weight-stream-rewrite (job 542f2b169c934706; codec 12.17 bits/elem, ratio 0.761, bit-exact; load_inline + per-route self-check + fallback all kept). Base is bf16 master = bf16-m32-swapab-wgmma-triton (the 1.2215 restack is NOT landed, pending the user; do not build on it). Three kernel changes plus a routing cut, all inside _lossless_swapab_kernel<NT> and its launcher: (1) OCCUPANCY: stage 1 ran one 128-thread CTA/SM (115-165 KB smem, 8 stages) = 4 warps/SM, one warp per SMSP serially doing cp.async waits, integer dequant, ldmatrix of x and dependent mma.sync chains. Reach 8-16 warps/SM: either cut the ring to 3-4 stages (~13 KB compressed/stage) so 2-3 CTAs fit per SM, or go to 8 warps/CTA with 2 warps per 16-row tile splitting each K-stage (reduce the pair in smem at the end). Keep regs <=128 so 2+ CTAs fit; print cudaOccupancyMaxActiveBlocksPerMultiprocessor and achieved warps/SM per NT in prepare. (2) DEQUANT ONCE PER k16 A FRAGMENT: decode each weight fragment exactly once into registers and reuse it for all NT/8 token (B) fragments; check the stage-1 code for re-decoding per n-fragment (time grew 40.6->45.2->51.4 us on pooler for NT 8/16/32, which says it is not amortized). Software-pipeline: decode stage s+1 fragments while stage s mma.sync issue. (3) SCOPE: route only pooler (m 1/8/16/32), unpooler-full (m 16/32) and qkv_gate (m 1/8/16/32) to the lossless kernel; route o (all m) and the unpooler slice (m 1/8) back to master's Triton path (they are latency-bound 12.8/6.4 MB streams and cannot win). Router and all m>=128 untouched. (4) In prepare, print a cold-L2 timing per route (flush a >L2 buffer before the event) next to the warm one, since qkv/o profile-vs-row gaps mislead. Stay on mma.sync (no wgmma), so the sm_90 vs sm_90a load_inline bug does not apply; if wgmma is introduced, set TORCH_CUDA_ARCH_LIST=9.0a around load_inline. Codec, weight compression at load time, and output are unchanged; results must stay bit-exact with stage 1's self-check.

## What changed, and where
All edits are in the worktree's `muse_bf16_proj/submission.py`, section (6), which holds `_LL_CUDA`, the launcher and the `_ll_*` helpers. The diff is 161+/78-. kernel_fn, the routers and the m>=128 paths are unchanged.

- **(1) Occupancy: option (b), as the review asked.** Option (a) was rejected because it raises warps/SM only if the grid grows past 132 CTAs, which adds split-K reduce traffic.
  - `THREADS` goes from 128 to 256 (8 warps per CTA). Warp w owns 16-row tile `w >> 1` and 64-K group `w & 1` of every 128-K stage, so the two warps of a tile split each K-stage.
  - After the mainloop the ring is reused for the pair reduce: K-half 1 writes its accumulators to smem (bank-conflict-free, `[...]*32 + lane`, at most 8 KB), and K-half 0 adds them in fixed order (half 0 + half 1). Then K-half 0 runs the unchanged epilogue (direct bf16 store, or ws partial plus ticket reduce).
  - The grid is unchanged: pooler 64x2=128, unpooler-full 128x1=128, qkv 36x3=108. So 8 warps/SM are actually resident on the busy SMs (1 CTA/SM, one wave).
  - The ring stays at 8 stages, and the x-tile issue loop is now guarded: at NT=8, `NT*16 = 128 < 256` threads.
  - `__launch_bounds__(256, 1)`. Deviation: the spec's "regs <=128 so 2+ CTAs fit" applies to option (a) only. Under (b) occupancy is limited by smem (1 CTA/SM), and 256 threads x up to 255 regs still fits the register file. The register count is printed.
  - prepare prints, per NT, regs, spills, smem and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (what fits). Per NT x route it also prints the **achieved** warps/SM as `min(grid, occ*SMs)/SMs*8` and the number of busy SMs, per the review's formula.
  - qkv keeps split 3 (108 CTAs, 24 SMs idle). Raising it would mean a second wave or option (a); that is left open for stage 3.
- **(2) Dequant once per k16: ALREADY IN STAGE 1. Not re-implemented.** Stage 1 already calls `decode_step` once per k16 step and reuses it for all NT/8 B fragments, so the NT growth is not caused by re-decoding. Per the review, it more likely comes from ldmatrix/MMA serialisation on one warp per SMSP and from x L2 re-reads.
  - Only the software-pipelining half was added:
    - (i) Each warp loads the NEXT stage's group from smem into registers (`Gn`) before issuing this stage's MMAs. The wait became `cp_wait<STAGES-3>` so that stage i+1 has landed, and the prologue loads stage 0 after `cp_wait<STAGES-2>`. `issue(i+7)` writes stage i-1's buffer, so it does not race with that load.
    - (ii) Inside a group, k16 step s+1's A fragment is decoded into `a[(s+1)&1]` before step s's MMAs issue.
  - The milestone line "(2) did not land if the NT growth is >=4 us" is misattributed, per the review. Any remaining NT growth points at x traffic or ldmatrix, not decode.
- **(3) Scope.**
  - `LL_CFG` drops `o` and `unpooler_slice`, and `LL_WEIGHTS` drops `o`, so o is no longer encoded.
  - `_ll_route` returns `"unpooler_slice"` for any unpooler n != 8192. With no LL_CFG entry, `_ll_route_ok` returns False, so o (all m) and the slice reach master's `_unpool_shape_ok`, `_o_skinny_applies` and `_swapab_route_ok` gates exactly as on master.
  - If a kept route's self-check fails, it falls back to master's shape-gated Triton kernels (the same gates), not straight to `_bf16_gemm_dispatch_impl`.
- **(4) Cold-L2 timing and the kill-gate reading.** `_ll_time_route` now prints, for m in 1/8/16/32:
  - a warm-L2 time (20 back-to-back launches);
  - a cold-L2 time: a 256 MB buffer (larger than the 50 MB L2) is touched with `add_(1)` before each timed launch, and the median of 15 is taken;
  - TB/s compressed and raw-equivalent for both.
  - pooler m=1 also prints `KILL-GATE pooler m=1 cold-L2 X us vs gate 19.0 us: PASS/MISS`. This is timing only and routes nothing. The judged kill gate is still the pooler m=1 **profile** kernel time, and the line says so. The flush buffer is freed at the end of prepare.
- **LESSONS compliance.**
  - No child processes.
  - The self-check syncs and prints after each launch (`launch 1 ok` / `launch 2 ok` per route and m), and after each weight's decode kernel (`decode kernel ok`), so a fault is attributed.
  - Build adds `-lineinfo`. The extension name is bumped to `muse_bf16_lossless_v2`.
  - The CUDA is still mma.sync, ldmatrix and cp.async.cg only (no wgmma or setmaxnreg), so `TORCH_CUDA_ARCH_LIST=9.0a` is not needed and is not set.
- **Bit-exactness.** The codec round trip is still checked bit-exact through the kernel decoder. GEMM outputs will differ from stage 1 in the last bits because the fp32 summation order changed (pair reduce). The self-check stays tolerance-based against fp64 (rel L2 <= 1e-3, worst row <= 5e-3, one-hot row exact, launch 1 == launch 2), as the review requires.
- **Fallback.** All master fallbacks and `PATH CHOSEN` lines are kept.
- **Uncertainties.**
  - The register count at NT=32 with the `Gn` prefetch (about 18 extra registers) is unverified. It is printed, and spills would show as local bytes.
  - Per SMSP there are now 2 warps rather than 4, since 8 warps sit across 4 SMSPs. That may still be too little latency hiding for 2.7 TB/s compressed.
  - The x L2 re-read (BLOCK_N=64) is unchanged.
  - Nothing could be compiled or run on this node.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2 diff`:

- [x] CUDA-graph capturable: kernel_fn is unchanged. `_ll_forward` makes one launch with no sync, no allocation beyond `torch.empty` for out (same as stage 1), and branches only on cfg/shape/strides/prepare constants. The new syncs and prints are in prepare-only helpers (`_ll_check_route`, `_ll_time_route`, `_lossless_prepare`).
- [x] Nothing derived from `data` is cached across calls: no new state keys except the prepare-only flush buffer, which is deleted.
- [x] Weights are repacked losslessly in prepare() only. The codec is unchanged and bit-exact round-trip gated. BF16 stays BF16 (losslessly smaller), and the router is untouched.
- [x] Activations are BF16 where SGLang keeps BF16: x is consumed as bf16, fp32 accumulation, bf16 output. n/a for KV cache and queries: not an attention problem.
- [x] n/a: attention key coverage. This problem has no attention, and nothing is skipped: every K element of every weight row is streamed.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network. The diff adds no imports. load_inline is unchanged apart from the name and `-lineinfo`, and it runs in-process with no child process of our own.
- [x] All work happens inside the timed call, with no output caching or replay: the pair reduce and the split-K reduce run inside the kernel each launch.
- [x] The first-line marker `# okc: problem=muse_bf16_proj` is kept, with no triton.autotune and no host sync in kernel_fn.

Lint output:

```
warn  dtype        L1189: sm8 = sm.to(torch.uint8)  (allowed only where SGLang does the same)
warn  dtype        L1190: c8 = c.to(torch.uint8)  (allowed only where SGLang does the same)
warn  dtype        L1202: hdr_t = hdr.view(rt, 16, kg).permute(0, 2, 1).to(torch.uint8).contiguous().view(-1)  (allowed only where SGLang does the same)
warn  host-sync    L1264: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1267: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1308: e1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1318: c1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1342: attrs = _LL_EXT.lossless_attrs().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1343: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1386: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1428: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```

The dtype warnings are the compressed payload packing to uint8 (lossless codec bytes, not a weight precision change; the same in stage 1). The host-sync warnings are all in prepare(). py_compile OK; __pycache__ removed.

## Risks / what could fail
high
Milestones (per target, judged on rows and the lossless kernel time vs incumbent): M1 = builds, self-check bit-exact on the 3 kept routes, o/slice on master, prepare prints >=8 warps/SM achieved. M-KILL = pooler m=1 kernel <=19.0 us; if pooler m=1 is still above 19 us the line is DEAD (codec cannot be streamed fast enough with mma.sync) - do not mint stage 3. M2 = at or below incumbent on each target: pooler m1/8/16/32 <=19.0/19.3/19.5/20.5; unpooler-full m16/32 <=18.5/18.9; qkv_gate rows m1/8/16/32 <=12.91/13.18/13.39/13.69. M3 = below the raw-byte floor where only compression can get: pooler and unpooler-full <=14 us (stretch 15.5 first), qkv_gate rows <=7.9. Kernel time should be ~flat in NT (8/16/32) if dequant is amortized; if it still grows >=4 us from m=1 to m=32, (2) did not land. Risks: 2-3 CTAs/SM with a short ring may expose DRAM latency again; 8-warp split-K adds an smem reduce; qkv rows are cold-L2 and the profile understates them. Incumbent numbers are from check job 1602a4e947e64090; master is still bf16-m32-swapab-wgmma-triton (restack 1.2215 not landed, pending the user). Source: muse_bf16_proj/hypotheses/REWRITE_READ_2026-10-25.md section 3.

## Result (after the run)
- job `f1024212dd2f49f3` (check), status done, passed True, tests 9/9
- speedup 0.8870x, 95% CI [0.8737, 0.9003], score 5446.480634650908 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/regression/bf16-lossless-exp4-weight-stream-rewrite.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-lossless-exp4-occupancy-dequant-once-stage2.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-lossless-exp4-occupancy-dequant-once-stage2.patch`; branch `hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2`
