---
problem: muse_bf16_proj
slug: router-hilo-tall-tile-m65-m256
created: 2026-10-25T153141Z
base: master
area: router GEMM (m=65..256, hilo tile config)
change: >
  Built on current master (blob 353e3b19). Only the m > 32 router path changes:
  _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound, not DRAM
  bound. Master's config is (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m<=128 and (64, 64, 8) at
  m=256, 128 CTAs both times. Each tile re-reads hi+lo (8.4 MB) once per M-tile and x once per
  N-tile, which is 25.2 MB at m=128 and 50.3 MB at m=256 (router_tiled printed these bytes, job
  b7b2b55d04544660). The kernel time follows those bytes, not the 8.4 MB of DRAM: hilo is 5.0 us
  at m=128 and 7.0 us at m=256 with the same CTA count, so the extra 25 MB cost 2 us (about 12.5
  TB/s from L2). CHANGE: give _router_hilo_config taller token tiles and deeper split-K at the
  same CTA count: m in 65..128 -> (BLOCK_M 128, BLOCK_N 64, SPLIT_K 32, BLOCK_K 64), grid 1x4x32 =
  128 CTAs, 16.8 MB L2->SM (-33%). m in 129..256 -> candidate A (128, 64, 16, BLOCK_K 128), grid
  2x4x16 = 128 CTAs, 33.6 MB (-33%); candidate B (256, 64, 32, BLOCK_K 64, num_warps 8), grid
  1x4x32 = 128 CTAs, 25.2 MB (-50%). Add a per-config BLOCK_K and num_warps to the launch;
  _router_hilo_kernel's body is unchanged (two tl.dot per K-step into one fp32 acc, atomic_add
  into the torch.zeros output). Print n_regs, n_spills and wgmma count per compiled variant; smem
  per stage is 64 KB at (128, 64, BK 128), so use 3 stages at BK 128 and 4 at BK 64. Precision is
  unchanged (same hi+lo fp32 accumulate, only the split order differs). The router at m <= 32
  (cluster kernel and swap-AB) and every other layer stay byte-identical. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Risk:
  twice as many fp32 atomics into L2 (1.05M vs 0.52M at m=128 and m=256, 2.1M for candidate B);
  the selector catches that.
expected:
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "hilo 5.0 us + memset 0.8 / row 7.38, floor 2.21; L2->SM bytes 25.2 -> 16.8 MB: kernel ~4.0, row ~6.4"
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "hilo 7.0 us + memset 0.7 / row 9.31, floor 2.68; 50.3 -> 25-34 MB: kernel ~4.5-5.3, row ~7.0-7.6"
  - shapes: layer=router load=decode_* m=*
    direction: unchanged
    why: m <= 32 router paths untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T203138Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T203138Z
base_sha: 72279d8be856d60ae496236e976e81549d53cc2d
base_blob: ad8d7254a64a52db2a8f56ec0ad262748153a2e8
branch: hyp/muse_bf16_proj/router-hilo-tall-tile-m65-m256
worktree: .worktrees/muse_bf16_proj/router-hilo-tall-tile-m65-m256
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T203553Z
impl_commit: 61b0437658831be3502564adac79478dd44c441e
submission_sha256: 1af8e420904350eba5ffc36d54e97d46abd170e2573a0e9dcdfeeb41708ddd3d
norm_sha256: 50c0204c429a714c8c74cdeed192fa0c0515eacb68f40f5f8b804e0a5ad393a0
submission_blob: 07996bb0a21bf7d1b490334998bdc0f7517aa5f0
lint_warnings: 37
attempt_file: 2026-10-25T203553Z-submit-router-hilo-tall-tile-m65-m256.md
prepared_at: 2026-10-25T203553Z
job: 864e8094b5bc4c1b
dispatched_at: 2026-10-25T203601Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.2539493043550642
speedup_ci: [1.2351400647897384, 1.2727585439203901]
harvested_at: 2026-10-25T204223Z
best_at_harvest: 1.2566457115954903
promotion: null
---

# router-hilo-tall-tile-m65-m256

## Change

Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound, not DRAM bound. Master's config is (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m<=128 and (64, 64, 8) at m=256, 128 CTAs both times. Each tile re-reads hi+lo (8.4 MB) once per M-tile and x once per N-tile, which is 25.2 MB at m=128 and 50.3 MB at m=256 (router_tiled printed these bytes, job b7b2b55d04544660). The kernel time follows those bytes, not the 8.4 MB of DRAM: hilo is 5.0 us at m=128 and 7.0 us at m=256 with the same CTA count, so the extra 25 MB cost 2 us (about 12.5 TB/s from L2). CHANGE: give _router_hilo_config taller token tiles and deeper split-K at the same CTA count: m in 65..128 -> (BLOCK_M 128, BLOCK_N 64, SPLIT_K 32, BLOCK_K 64), grid 1x4x32 = 128 CTAs, 16.8 MB L2->SM (-33%). m in 129..256 -> candidate A (128, 64, 16, BLOCK_K 128), grid 2x4x16 = 128 CTAs, 33.6 MB (-33%); candidate B (256, 64, 32, BLOCK_K 64, num_warps 8), grid 1x4x32 = 128 CTAs, 25.2 MB (-50%). Add a per-config BLOCK_K and num_warps to the launch; _router_hilo_kernel's body is unchanged (two tl.dot per K-step into one fp32 acc, atomic_add into the torch.zeros output). Print n_regs, n_spills and wgmma count per compiled variant; smem per stage is 64 KB at (128, 64, BK 128), so use 3 stages at BK 128 and 4 at BK 64. Precision is unchanged (same hi+lo fp32 accumulate, only the split order differs). The router at m <= 32 (cluster kernel and swap-AB) and every other layer stay byte-identical. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Risk: twice as many fp32 atomics into L2 (1.05M vs 0.52M at m=128 and m=256, 2.1M for candidate B); the selector catches that.

## Expected

- `layer=router load=verify_16 m=128` faster: hilo 5.0 us + memset 0.8 / row 7.38, floor 2.21; L2->SM bytes 25.2 -> 16.8 MB: kernel ~4.0, row ~6.4
- `layer=router load=verify_32 m=256` faster: hilo 7.0 us + memset 0.7 / row 9.31, floor 2.68; 50.3 -> 25-34 MB: kernel ~4.5-5.3, row ~7.0-7.6
- `layer=router load=decode_* m=*` unchanged: m <= 32 router paths untouched

## Risk

Distinct from router-m128-m256-cluster-dsmem-tiled-rewrite (CUDA mma.sync with the same 64x64 tiles and the same 50.3 MB; the gate kept master) and from router-swapab-wgmma-all-m (swap-AB, lost m=128). Neither changed the L2 byte count. Atomic contention could cancel the byte saving; the selector would then print master and the run reads flat.

## Adversarial review

Reviewed 2026-10-25T203138Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, self-gated retile of master's _router_hilo_kernel at m=128/256; distinct from earlier router attempts, no rules issue. Mechanism weaker than stated (m=256 mixes math and L2 bytes; doubled fp32 atomics may cancel savings): expect a small win at m=128 and likely PATH CHOSEN master at m=256. The in-run selector medians and profile make the single run readable without companions.

### Justification
Per-M-tile hi+lo re-reads are real (25.2/50.3 MB at m=128/256, matches job b7b2b55d). _router_hilo_kernel 5.0/6.9 us + 0.7-0.8 us memset; rows 7.6/9.7 us vs floors 2.2/2.7. Weakest link: the L2-bound claim from 5.0 -> 7.0 us also doubles math (2.15 GFLOP ~2.2 us wgmma floor at m=256) and K-steps; implied 12.5 TB/s marginal L2 is above H100/H200 L2. SPLIT_K 32 doubles red.global (0.52M -> 1.05M); BK 64 gives 4 K-steps, all prologue/epilogue. Prior check: cluster-dsmem-tiled-rewrite kept 64x64 tiling (never tested bytes); swapab-wgmma-all-m changed operand roles. Base drift (353e3b19 -> ad8d7254) does not touch the router code. Spec gap: master's bucket m<=128 covers 33..128; keep 33..64 on (64,64,16) explicitly.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: the incumbent kernel; body/memset/epilogue unchanged.
- bf16-restack-m16-swapab-router-swapab-m32: ROUTER_SWAPAB_MAX_M=32 puts 33..64 in the hilo m<=128 bucket; keep the boundary explicit.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (master): prepare's _ab_selfcheck rebuilds state B via _ab_flags; the new selector flag must be inherited like NEW_CELL_FLAGS.
- pooler/unpooler/qkv/o and router m<=32 hypotheses: no shared traffic or numerics with router m=128/256.

### Interference and A/B needs
No companion: the change is ~0.3-0.5% headline at most, inside the CI; the run is read from the prepare() selector's same-run cold-L2 medians (master vs A vs B), the profiler kernel-name assertion + PATH CHOSEN, and the judge rows vs 7.6/9.7 us.

### Open questions
- Keep m 33..64 at (64,64,16) and print the m=64 path.
- _router_shape_ok / _router_hilo_forward must take per-config BLOCK_K, stages and warps.
- Selector flag must pass through _ab_flags into state B.
- 4-stage pipeline at 4 K-steps: try 3 stages; print n_regs/spills.
- Overlap with queued router-m128-m256-wgmma-swapab-tall-cluster-stage2 (re-baseline if this lands).


## Implementation notes

Only muse_bf16_proj/submission.py in the worktree. Added `_router_hilo_cfg6(m, tall)` (full config: BM, BN, SPLIT_K, BLOCK_K, stages, warps); m 33..64 explicitly stays master (64,64,16,128,3,4); 65..128 / 129..256 use state["router_tall"][128|256] if set, else master. `_router_hilo_forward`, `_router_shape_ok`, `_router_path` take the per-config values (BLOCK_K, stages, warps, split-K), not the ROUTER_* globals. Candidates: bucket 128: (128,64,32,64) at 3 and 4 stages; bucket 256: A (128,64,16,128,3 st) and B (256,64,32,64,3 st, 8 warps). `_router_tall_prepare` (prepare only, state A): rel-L2 self-check (5e-5) at m in {65,128} / {129,130,200,256}, prints n_regs/spills/wgmma, CUDA-graph timing with 256 MB L2 flush, 41 interleaved reps, tiny-graph baseline subtracted, wins only at >= 1.02x, profiler kernel-name assert, PATH CHOSEN prints (m=33,64,65,128,129,200,256). Flush buffer and graphs are freed in `_router_tall_time`'s finally; nothing is stored in state except the constant dict. State B inherits the dict via `_ab_flags` ("router_tall") and only warms it. A/B self-check got a router m=256 cell and component "router_tall" (miss or B-build failure -> {} in both states = master config). kernel_fn: only shape + state["router_tall"] branching. Deviation: 3 vs 4 stages offered as two candidates at BK 64 (review asked to try 3) so the selector decides; extra compile time ~seconds.

## Rules check
- [x] CUDA-graph capturable: kernel_fn change is shape + constant dict lookup only; torch.zeros stays inside _router_hilo_forward; no syncs/timing/profiler there (all in prepare).
- [x] Nothing derived from `data` is cached across calls: state holds only a per-bucket config constant chosen on synthetic x.
- [x] Weights: unchanged (hi+lo split from prepare).
- [x] Activations BF16 as before; router FP32 hi+lo with fp32 accumulate unchanged.
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness access, subprocesses or network (torch.profiler is local, prepare only).
- [x] All work inside the timed call; no output caching or replay.
- [x] Lint: OK with warnings (only prepare-time host-sync/dtype warns, none in kernel_fn path); py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus, read-only; no fixes). **Verdict: dispatch.**

- Gates: the self-check bound (AB_ROUTER_REL_L2 5e-5) is the same one master's identical kernel body passes; the split order is the only numeric change, so it can pass. K % (BLOCK_K*SPLIT_K) holds for every candidate (8192 = 64*32*4 = 128*16*4), n % 64 == 0, so _router_shape_ok accepts them.
- Fallback: each candidate's first launch (Triton compile, OutOfResources, wrong answer, non-finite) sits inside a per-candidate try and is REJECTED with a print; timing, profiler and whole-selector failures each land on {} = master config with a PATH CHOSEN line; a state-B build failure or an A/B miss at router m=128/256 resets router_tall to {} in both states. _print_ptx_counts swallows its own errors, so a diagnostic cannot reject a good config.
- Feasibility (H100, 227 KB smem/block): (128,64,BK64) 32 KB/stage -> 96 KB at 3 stages, 128 KB at 4; (128,64,BK128) 64 KB/stage -> 192 KB at 3 (+<=32 KB epilogue scratch still under the limit); (256,64,BK64,8 warps) 48 KB/stage -> 144 KB. fp32 acc is 64 regs/thread in every case. All plausible; any that is not is caught at launch.
- Correctness: kernel body unchanged; rows masked by m_mask on load and atomic; tails (m=65, 129, 130, 200) covered by the self-check; no OOB indexing for any grid, so no device-fault path was found.
- Rules: kernel_fn adds only a constant-dict lookup plus shape branching; torch.zeros stays as in master; selector, flush buffer, graphs and profiler are prepare()-only and synthetic-x only; no precision change. Lint OK (prepare-only warnings), py_compile OK.
- Scope: matches the change paragraph plus the review's asks (explicit 33..64 = master, flag via _ab_flags, router m=256 A/B cell); the extra 4-stage candidate at bucket 128 is a declared deviation, cost seconds.

Residual risk: an untested config could device-fault (not hang/raise) on first launch in prepare(), which no try can catch; masking analysis finds no such path.

## Result

- job `864e8094b5bc4c1b` (submit), status done, passed True, tests 9/9
- speedup 1.2539x, 95% CI [1.2351, 1.2728], score 3852.468343993057 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-hilo-tall-tile-m65-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-hilo-tall-tile-m65-m256.patch`; branch `hyp/muse_bf16_proj/router-hilo-tall-tile-m65-m256`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.54 | 10.75 | 1.014 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 6.07 | 0.9986 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 17.72 | 17.54 | 1.0045 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.01 | 4.72 | 1.0568 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.17 | 4.91 | 1.0555 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.64 | 12.9 | 0.987 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.21 | 5.94 | 1.022 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.46 | 19.38 | 1.0108 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.82 | 18.64 | 1.0034 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.51 | 5.86 | 0.9345 | lost |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.71 | 13.8 | 0.9932 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.47 | 6.39 | 0.9981 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.86 | 20.16 | 0.9925 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.49 | 18.43 | 1.0048 | flat |
| layer=router load=decode_32 m=32 | 62 | 5.95 | 6.05 | 0.9855 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.35 | 12.82 | 1.0066 | flat |
| layer=o load=verify_1 m=8 | 62 | 5.9 | 6.1 | 0.9924 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.31 | 18.94 | 1.0148 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 5.54 | 5.55 | 1.015 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.49 | 5.16 | 1.0637 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 15.55 | 16.03 | 0.9773 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.68 | 7.45 | 0.9799 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 22.85 | 22.78 | 0.9735 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 18.88 | 18.67 | 0.9893 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.61 | 7.5 | 1.0054 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.46 | 19.98 | 0.9836 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.44 | 9.11 | 0.99 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 27.98 | 29.88 | 0.9734 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.18 | 24.56 | 0.9971 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.72 | 9.99 | 0.9642 | lost |

gained on unpooler decode_1; router decode_1, verify_1; lost on router decode_16, verify_32. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-hilo-tall-tile-m65-m256`) that adds a further change in the same area is the only way to learn more.

