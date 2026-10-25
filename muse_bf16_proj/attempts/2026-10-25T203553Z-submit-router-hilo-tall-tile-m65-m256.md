# muse_bf16_proj · 2026-10-25T203553Z · submit · router-hilo-tall-tile-m65-m256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T153141Z-router-hilo-tall-tile-m65-m256.md` (branch `hyp/muse_bf16_proj/router-hilo-tall-tile-m65-m256`).

- **Job:** `864e8094b5bc4c1b` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `61b04376`

## Goal
Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound, not DRAM bound. Master's config is (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m<=128 and (64, 64, 8) at m=256, 128 CTAs both times. Each tile re-reads hi+lo (8.4 MB) once per M-tile and x once per N-tile, which is 25.2 MB at m=128 and 50.3 MB at m=256 (router_tiled printed these bytes, job b7b2b55d04544660). The kernel time follows those bytes, not the 8.4 MB of DRAM: hilo is 5.0 us at m=128 and 7.0 us at m=256 with the same CTA count, so the extra 25 MB cost 2 us (about 12.5 TB/s from L2). CHANGE: give _router_hilo_config taller token tiles and deeper split-K at the same CTA count: m in 65..128 -> (BLOCK_M 128, BLOCK_N 64, SPLIT_K 32, BLOCK_K 64), grid 1x4x32 = 128 CTAs, 16.8 MB L2->SM (-33%). m in 129..256 -> candidate A (128, 64, 16, BLOCK_K 128), grid 2x4x16 = 128 CTAs, 33.6 MB (-33%); candidate B (256, 64, 32, BLOCK_K 64, num_warps 8), grid 1x4x32 = 128 CTAs, 25.2 MB (-50%). Add a per-config BLOCK_K and num_warps to the launch; _router_hilo_kernel's body is unchanged (two tl.dot per K-step into one fp32 acc, atomic_add into the torch.zeros output). Print n_regs, n_spills and wgmma count per compiled variant; smem per stage is 64 KB at (128, 64, BK 128), so use 3 stages at BK 128 and 4 at BK 64. Precision is unchanged (same hi+lo fp32 accumulate, only the split order differs). The router at m <= 32 (cluster kernel and swap-AB) and every other layer stay byte-identical. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Risk: twice as many fp32 atomics into L2 (1.05M vs 0.52M at m=128 and m=256, 2.1M for candidate B); the selector catches that.

## What changed, and where
Only muse_bf16_proj/submission.py in the worktree. Added `_router_hilo_cfg6(m, tall)` (full config: BM, BN, SPLIT_K, BLOCK_K, stages, warps); m 33..64 explicitly stays master (64,64,16,128,3,4); 65..128 / 129..256 use state["router_tall"][128|256] if set, else master. `_router_hilo_forward`, `_router_shape_ok`, `_router_path` take the per-config values (BLOCK_K, stages, warps, split-K), not the ROUTER_* globals. Candidates: bucket 128: (128,64,32,64) at 3 and 4 stages; bucket 256: A (128,64,16,128,3 st) and B (256,64,32,64,3 st, 8 warps). `_router_tall_prepare` (prepare only, state A): rel-L2 self-check (5e-5) at m in {65,128} / {129,130,200,256}, prints n_regs/spills/wgmma, CUDA-graph timing with 256 MB L2 flush, 41 interleaved reps, tiny-graph baseline subtracted, wins only at >= 1.02x, profiler kernel-name assert, PATH CHOSEN prints (m=33,64,65,128,129,200,256). Flush buffer and graphs are freed in `_router_tall_time`'s finally; nothing is stored in state except the constant dict. State B inherits the dict via `_ab_flags` ("router_tall") and only warms it. A/B self-check got a router m=256 cell and component "router_tall" (miss or B-build failure -> {} in both states = master config). kernel_fn: only shape + state["router_tall"] branching. Deviation: 3 vs 4 stages offered as two candidates at BK 64 (review asked to try 3) so the selector decides; extra compile time ~seconds.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn change is shape + constant dict lookup only; torch.zeros stays inside _router_hilo_forward; no syncs/timing/profiler there (all in prepare).
- [x] Nothing derived from `data` is cached across calls: state holds only a per-bucket config constant chosen on synthetic x.
- [x] Weights: unchanged (hi+lo split from prepare).
- [x] Activations BF16 as before; router FP32 hi+lo with fp32 accumulate unchanged.
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness access, subprocesses or network (torch.profiler is local, prepare only).
- [x] All work inside the timed call; no output caching or replay.
- [x] Lint: OK with warnings (only prepare-time host-sync/dtype warns, none in kernel_fn path); py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Distinct from router-m128-m256-cluster-dsmem-tiled-rewrite (CUDA mma.sync with the same 64x64 tiles and the same 50.3 MB; the gate kept master) and from router-swapab-wgmma-all-m (swap-AB, lost m=128). Neither changed the L2 byte count. Atomic contention could cancel the byte saving; the selector would then print master and the run reads flat.

## Result (after the run)
- job `864e8094b5bc4c1b` (submit), status done, passed True, tests 9/9
- speedup 1.2539x, 95% CI [1.2351, 1.2728], score 3852.468343993057 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-hilo-tall-tile-m65-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-hilo-tall-tile-m65-m256.patch`; branch `hyp/muse_bf16_proj/router-hilo-tall-tile-m65-m256`
