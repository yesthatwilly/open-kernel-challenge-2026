# muse_bf16_proj · 2026-10-26T012539Z · check · bf16-verify16-cutlass-m128-onetile-splitk-rewrite

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T153141Z-bf16-verify16-cutlass-m128-onetile-splitk-rewrite.md` (branch `hyp/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite`).

- **Job:** `d4edd38aeed44334` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `f1adab32`

## Goal
Built on current master (blob 353e3b19). This is a new line for the m=128 band, which the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (m=129..256 only) leaves to cuBLAS. It takes its scaffold from bf16-verify-streamk-cutlass-wgmma-rewrite stage 1 (patch done/regression/bf16-verify-streamk-cutlass-wgmma-rewrite.patch): CUTLASS 4.5 header discovery from flashinfer, load_inline with TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20, the #error guard on !__CUDA_ARCH_FEAT_SM90_ALL, the one prepare-allocated workspace per state, the self-check and the prints. WHY: verify_16 is 73.2 us against a 43.8 us floor, and its four GEMM rows are all cuBLAS at 1.3-2.0x their floors (qkv 16.58/8.42, o 7.35/3.99, pooler 22.94/14.64, unpooler 18.95/14.64). cuBLAS knobs are spent there: NN layout (landed for pooler/unpooler), cublasLt algo pick (did not carry over), and DeepGEMM, which never ran. Two Triton m=128 mainloops stalled at ~26 us, about 2.4 TB/s; REWRITE_READ named the x K-slice re-read per CTA (about 2x the weight bytes) and one CTA per SM. DESIGN, stage 1: (1) TileShape 128x128x64 with M = tokens, so m <= 128 is ONE M-tile and every W byte is fetched from DRAM by exactly one CTA. Use KernelTmaWarpSpecializedCooperative (2 consumer warpgroups of 64x128, 1 TMA producer, setmaxnreg) and TN operands: x [m, k] K-major as A, W [n, k] K-major as B, the layout master already holds, so no repack. (2) ClusterShape 1x2x1 so the two N-neighbour CTAs multicast the x tile (the reused operand) by TMA, halving x L2->SM bytes. (3) DecompositionMode::SplitK with static splits for one wave of <= 132 CTAs: pooler 32 N-tiles x 4 = 128 (K 2048), unpooler 64 x 2 = 128 (K 2048), qkv 18 x 7 = 126, o 64 x 2 = 128 (K 512). Print can_implement, grid, splits, smem, workspace bytes, regs/spills and the decomposition actually applied, and assert the grid is <= 132. (4) StageCountAutoCarveout; bf16 out, fp32 accumulate. (5) Rows >= m are handled by TMA OOB zero-fill and a predicated store, never by padding x, so there is no allocation or copy in kernel_fn. Routing: qkv_gate, o, pooler and unpooler (full width, n == 8192) at 65 <= cfg m <= 128 only, gated on cfg layer/m/n and shapes. Every m <= 64 path and every m > 128 path stays byte-identical to master. Self-check per layer at m 65, 100 and 128: rel L2 against fp32 torch, finite, and run2 bit-identical; plus capture once and replay with refilled inputs while alternating two prepared states (LESSONS: the judge keeps 2 state copies; no module-global workspace or TMA descriptors). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The selector runs per layer at m=128; a failed build or self-check prints DISABLED and that layer keeps master. Run as an okc check.

## What changed, and where
Edited only `.worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite/muse_bf16_proj/submission.py` (+714/-4 vs the worktree's HEAD, master blob ad8d7254, which already is the review's rebase base). Everything outside the new code is unchanged except the small hooks listed below.

What changed:
- New load_inline module `muse_bf16_cut128_v1_c<ver>` (`_CUT_CPP`/`_CUT_CUDA`, `_cut_build`), built at import after master's own modules. Scaffold = stage-1 streamk patch: CUTLASS header discovery (flashinfer / deep_gemm / sgl_kernel / tilelang / cutlass_library, newest 3.x+ tree), `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored, `-gencode=arch=compute_90a,code=sm_90a`, `-std=c++20`, `-lineinfo`, arch flags and build seconds printed, an `#error` if a non-sm_90a device pass is ever compiled (build then fails loudly and master runs). No subprocess/fork anywhere.
- Kernel: TileShape 128x128x64, ClusterShape 1x2x1, KernelTmaWarpSpecializedCooperative + TmaWarpSpecializedCooperative epilogue, StreamKScheduler with DecompositionMode::SplitK, runtime `splits`, ReductionMode::Deterministic, StageCountAutoCarveout, TN operands (x [m,k] A, W [n,k] B, no repack), bf16 out / fp32 accumulate, rows >= m by TMA OOB zero-fill and clipped TMA store (no padding, no copy). Distinct symbols (`c128::GemmKernel`, `cut128_*`).
- Per-layer state flags `cut_qkv_ok / cut_o_ok / cut_pooler_ok / cut_unpooler_ok` (CUT_LAYERS), added to NEW_CELL_FLAGS so the A/B state B inherits them; a miss in one cell touches no other flag. Splits: qkv 7 (fallback 8, 4), o 2 (4), pooler 4 (2), unpooler 2 (4); first candidate with can_implement, split mode applied and grid <= SM count is used (grid, splits, smem, workspace, sk_tiles/sk_units, regs/spills, decomposition applied are printed).
- Routing in kernel_fn: one new branch after the swapab routes and BEFORE the nn_ok / Lt / cuBLAS block, gated on layer flag, 65 <= cfg m <= 128, cfg n/k == full weight, x contiguous bf16 [m,k] 16-B aligned. Every m <= 64 and m > 128 path is byte-identical to master; with a flag off the call falls through to master's exact route (nn_ok, Lt, cuBLAS).
- prepare(): `_cut_prepare` after `_lt_prepare` (state A). Per layer: self-check at m 65/100/128 through `_cell_check` (announced launches, sync after each, rel L2 vs fp64 <= 3e-3, finite, run2 bit-identical, one-hot probe row), then `_cut_gate` at m=128, then a profiler kernel-name assertion (exactly one cutlass device_kernel<GemmUniversal> plus memset). One workspace per state (state B allocates its own); no module-global workspace or descriptors. A/B/A/B graph check extended with an o m=128 case and `_ab_component` now takes state and blames the cut flag when it serves the cell.
- Last lines of prepare() repeat every `PATH CHOSEN CUT <layer>` line and a `[cut128] FINAL` flags line.

Review fixes applied:
- Rebase: worktree already at blob ad8d7254 (checked with git rev-parse). The qkv incumbent is the B-cell cuBLASLt single-node plan (15.5 us), not 12.7+3.1: the gate races kernel_fn flag-off, which routes through `_lt_route_ok` first, so the qkv cell must beat 15.5 us (M2 target for qkv = below 15.5). Pooler judged as cutlass kernel + Memset vs 23.8 us.
- Selector reuses master's flag-on / flag-off kernel_fn capture (`_cut_gate`, modelled on `_cell_gate`) instead of a copy of `_ouns_gate` racing plain dispatch; nn_ok keeps being master's route for pooler/unpooler.
- Two interleaved rounds, both master/cand >= 1.03x (net of empty-graph baseline), candidate IQR <= 1.5x master's; both rounds' medians and IQRs, both sides' profiled kernel names and PATH CHOSEN are printed even when master is kept.
- Open questions answered in code: can_implement / splits-applied / sk_units / grid printed per layer, qkv falls back 7 -> 8 -> 4 splits rather than disabling; fixup serialisation is not directly observable, the printed per-layer gate times decide whether stage 2 (DSMEM reduce) is needed; the kernel-name assertion requires a cutlass device_kernel<GemmUniversal> row per routed cell; rebase on whatever master holds at build time (done, no unlanded companion present in this tree).

Deviations:
- CUTLASS owns the mainloop, so its mbarrier waits are not bounded by us (cannot patch them); mitigation: the canary runs every launch announced and synced before anything is routed, and the module is never run on an unchecked shape. A hang would still run to the timeout.
- The brief's `#if __CUDA_ARCH_FEAT_SM90_ALL ... #else error flag` is realised as the scaffold's `#error` (loud build failure -> master), since CUTLASS itself is only usable on sm_90a.
- IQR rule has a 0.5 us absolute floor (event timer resolution), else a master IQR of 0 makes the rule unsatisfiable.
- The gate runs at m=128 only (the scored cell); the 65..127 band is routed by the same flag on the strength of the m 65/100/128 self-checks, untimed.
- Deadlines: build skipped if import took > 180 s; phase skipped (prints "skipped: deadline") if state A's build is > 150 s in; each layer skipped if > 200 s in.
- Nothing could be compiled or run on this node (no nvcc/GPU/CUTLASS); the CUTLASS scheduler member names (`splits`, `decomposition_mode`) are detected by SFINAE, and if absent the split mode reports not applied and the layers are DISABLED (master).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds one branch calling `cut128_gemm` (host-side TMA descriptor build, stream memset, cluster launch), a fixed-shape torch.empty, no .item()/.cpu()/sync; branches on cfg, shapes and state flags only. Same structure as stage 1, which was graph-captured by the judge.
- [x] Nothing derived from `data` is cached: workspace and splits are weight/shape constants made in prepare(); descriptors are rebuilt per call from the live pointers.
- [x] Weights: untouched (read in place as [N, K] bf16, no repack, no dtype change).
- [x] Activations BF16, output bf16 as SGLang's; fp32 accumulation.
- [x] n/a: attention rules (no attention in this problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (load_inline compile only, no child process of our own, no subprocess import).
- [x] All work inside the timed call; no output caching or replay (checked against `git -C .worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite diff`: new branch returns the fresh CUTLASS output).
- [x] Per-layer flags, own workspace per state, a miss falls back to master's exact path and never disables an unrelated flag (state B inherits via _ab_flags / flags["_cut_*"]).

okc_lint (warnings only, all in prepare()-time code; tail of output):
```
warn  host-sync    L1452: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1498: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1507: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1513: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1517: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1546: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1567: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1613: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2013: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2647: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2650: code = int(err.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2679: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2683: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2700: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2709: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2729: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2738: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2741: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2807: code = int(state["bulk_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2838: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2851: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2876: state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2887: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2892: algo_id, tile, stg, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2898: key = tuple(int(v) for v in algos[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2903: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2950: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3385: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L3394: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3427: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3604: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3791: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3795: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3808: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3847: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite/muse_bf16_proj/submission.py
```
py_compile: OK; __pycache__ removed.

## Risks / what could fail
high
Milestones (judged on the replaced kernels' own time in top_kernels, including any Memset node, and on the m=128 rows). M1: build OK, 9/9 tests, a cutlass device_kernel on the m=128 rows the selector routed, no silent fallback (grep the build-OK / DISABLED / PATH CHOSEN lines first). M2: pooler and qkv rows below the incumbent, which proves the reduce-node removal. M3: verify_16 load 73.2 -> ~62 us (+18% on the load, about +3% headline). Stage 2 if M2 is met: replace the global-workspace split-K fixup and its Memset node with an in-cluster DSMEM reduce (cluster 1x2xS), and merge this extension with the m=256 line into one load_inline module. Kill: if the pooler kernel is not under 20.2 us at M1, the CUTLASS tile is no better than nvjet at m=128; stop the line. Fallback: master through the prepare-time gate. Risks: the compile adds about 50 s, and the m=256 line builds a second module; keep both under the 10-minute run budget by compiling only the instantiations routed here. o's 8 K-steps per split may leave it slower: let the gate drop it.

## Result (after the run)
- job `d4edd38aeed44334` (check), status done, passed True, tests 9/9
- speedup 1.2423x, 95% CI [1.2237, 1.2610], score 3888.4910207152952 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-verify16-cutlass-m128-onetile-splitk-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-verify16-cutlass-m128-onetile-splitk-rewrite.patch`; branch `hyp/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite`
