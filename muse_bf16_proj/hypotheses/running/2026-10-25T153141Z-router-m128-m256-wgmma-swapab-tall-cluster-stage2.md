---
problem: muse_bf16_proj
slug: router-m128-m256-wgmma-swapab-tall-cluster-stage2
created: 2026-10-25T153141Z
base: master
area: router GEMM (m=65..256, CUDA wgmma swap-AB cluster kernel)
change: >
  Stage 2 of router-m128-m256-cluster-dsmem-tiled-rewrite (job b7b2b55d04544660, neutral). Port
  its load_inline scaffold, self-check, cold-L2 timing table and PATH CHOSEN gate onto CURRENT
  master (blob 353e3b19), not onto its stale branch. Stage 1 compiled and self-checked, and its
  gate correctly kept master. It lost on the kernel: mma.sync over the incumbent's same 64x64
  tiles, printing the same 25.2 / 50.3 MB L2->SM operand bytes. Cold-L2 eager it ran 17.3 / 24.0
  us against master's 12.5 / 14.7 at m=128 / 256. Its no-DSMEM-reduce variant was 15.2-22.2, so
  the mainloop, not the reduce, was slow. STAGE 2 CHANGE, three parts. (1) An sm90a wgmma swap-AB
  mainloop. The A operand is weight rows from smem: 64 hi rows and the 64 lo rows of the same
  output columns, TMA-loaded. The B operand is the x tile [BLOCK_T tokens x 64 K], TMA-loaded. One
  consumer warpgroup issues wgmma m64nBLOCK_Tk16 twice per K-step (hi, then lo) into one fp32
  accumulator, so each x tile in smem feeds both halves. There is one producer warp, a 4-6 stage
  mbarrier ring, and setmaxnreg. (2) Tall token tiles to cut L2 bytes. At m=256, BLOCK_T 128 x 4
  column groups x cluster 16 split-K = 128 CTAs, with 33.6 MB L2->SM (-33%). At m=128, two
  candidates: BLOCK_T 128 x 4 x cluster 16 = 64 CTAs (16.8 MB), or BLOCK_T 64 x 2 token tiles x 4
  x 16 = 128 CTAs (25.2 MB). (3) Keep stage 1's in-cluster DSMEM split-K reduce, so each output
  tile is owned by one cluster and stored once. That deletes the torch.zeros memset node (LESSONS:
  on 5-10 us rows a graph node matters). The output is torch.empty. Build per LESSONS:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, the #error guard, -lineinfo, per-launch
  'launching'/'ok' prints in the self-check, and cudaFuncAttributeNonPortableClusterSizeAllowed
  for cluster 16 (stage 1 measured max active clusters 14 at cs=16). Keep TMA descriptors per
  state, never module-global. Self-check at m 65, 128, 129, 200 and 256 with outlier rows and
  power-of-two single-nonzero rows: rel L2 <= 1e-5 against fp64 (bound 1e-4), exact-column rows <=
  1e-3; plus a two-state alternating graph replay. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The
  selector runs per m-band (65..128, 129..256) against whatever master's m > 32 router is at
  dispatch time; if router-hilo-tall-tile-m65-m256 has landed, it is the incumbent to beat. Router
  m <= 32 and every bf16 layer stay byte-identical. Run as an okc check.
expected:
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "incumbent hilo 7.0 + memset 0.7 / row 9.31, floor 2.68: M2 kernel <= 6.0 with no memset node, M3 <= 4.5 (row ~6.5)"
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "incumbent hilo 5.0 + memset 0.8 / row 7.38, floor 2.21: M2 kernel <= 4.5 with no memset, M3 <= 3.5 (row ~5.5)"
  - shapes: layer=router load=decode_* m=*
    direction: unchanged
    why: router m <= 32 untouched
risk: high
depends_on: router-m128-m256-cluster-dsmem-tiled-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _router_hilo_kernel + torch.zeros memset at router m=65..256 (hilo 5.0 + 0.8 us at m=128, 7.0 + 0.7 us at m=256)
review: proceed
status: running
review_at: 2026-10-25T212522Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T212549Z
base_sha: 8dd2a8d59b8619aeeb15dc4ebd6d2939f2eab87e
base_blob: 095978ec904ee937a4b5b039bf2b47b911801fa6
branch: hyp/muse_bf16_proj/router-m128-m256-wgmma-swapab-tall-cluster-stage2
worktree: .worktrees/muse_bf16_proj/router-m128-m256-wgmma-swapab-tall-cluster-stage2
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-26T011617Z
impl_commit: 6bfb294589ecfc9341bcdc4eb217f7d1a206e532
submission_sha256: b2f154c5c2c58bb8fc92a8b25638a0d2f163174a1f11b6878a87704f39ffb26d
norm_sha256: 0eb8636ce0288818fc5074ad949dcd28aae5a0a8919517e00fd2037c9a0838e3
submission_blob: 6c52162510bc950d3e53196acc57c9df34525b5d
lint_warnings: 38
attempt_file: 2026-10-26T012540Z-check-router-m128-m256-wgmma-swapab-tall-cluster-stage2.md
prepared_at: 2026-10-26T012540Z
job: 38daa28c6a704810
dispatched_at: 2026-10-26T012820Z
---

# router-m128-m256-wgmma-swapab-tall-cluster-stage2

## Change

Stage 2 of router-m128-m256-cluster-dsmem-tiled-rewrite (job b7b2b55d04544660, neutral). Port its load_inline scaffold, self-check, cold-L2 timing table and PATH CHOSEN gate onto CURRENT master (blob 353e3b19), not onto its stale branch. Stage 1 compiled and self-checked, and its gate correctly kept master. It lost on the kernel: mma.sync over the incumbent's same 64x64 tiles, printing the same 25.2 / 50.3 MB L2->SM operand bytes. Cold-L2 eager it ran 17.3 / 24.0 us against master's 12.5 / 14.7 at m=128 / 256. Its no-DSMEM-reduce variant was 15.2-22.2, so the mainloop, not the reduce, was slow. STAGE 2 CHANGE, three parts. (1) An sm90a wgmma swap-AB mainloop. The A operand is weight rows from smem: 64 hi rows and the 64 lo rows of the same output columns, TMA-loaded. The B operand is the x tile [BLOCK_T tokens x 64 K], TMA-loaded. One consumer warpgroup issues wgmma m64nBLOCK_Tk16 twice per K-step (hi, then lo) into one fp32 accumulator, so each x tile in smem feeds both halves. There is one producer warp, a 4-6 stage mbarrier ring, and setmaxnreg. (2) Tall token tiles to cut L2 bytes. At m=256, BLOCK_T 128 x 4 column groups x cluster 16 split-K = 128 CTAs, with 33.6 MB L2->SM (-33%). At m=128, two candidates: BLOCK_T 128 x 4 x cluster 16 = 64 CTAs (16.8 MB), or BLOCK_T 64 x 2 token tiles x 4 x 16 = 128 CTAs (25.2 MB). (3) Keep stage 1's in-cluster DSMEM split-K reduce, so each output tile is owned by one cluster and stored once. That deletes the torch.zeros memset node (LESSONS: on 5-10 us rows a graph node matters). The output is torch.empty. Build per LESSONS: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, the #error guard, -lineinfo, per-launch 'launching'/'ok' prints in the self-check, and cudaFuncAttributeNonPortableClusterSizeAllowed for cluster 16 (stage 1 measured max active clusters 14 at cs=16). Keep TMA descriptors per state, never module-global. Self-check at m 65, 128, 129, 200 and 256 with outlier rows and power-of-two single-nonzero rows: rel L2 <= 1e-5 against fp64 (bound 1e-4), exact-column rows <= 1e-3; plus a two-state alternating graph replay. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The selector runs per m-band (65..128, 129..256) against whatever master's m > 32 router is at dispatch time; if router-hilo-tall-tile-m65-m256 has landed, it is the incumbent to beat. Router m <= 32 and every bf16 layer stay byte-identical. Run as an okc check.

## Expected

- `layer=router load=verify_32 m=256` faster: incumbent hilo 7.0 + memset 0.7 / row 9.31, floor 2.68: M2 kernel <= 6.0 with no memset node, M3 <= 4.5 (row ~6.5)
- `layer=router load=verify_16 m=128` faster: incumbent hilo 5.0 + memset 0.8 / row 7.38, floor 2.21: M2 kernel <= 4.5 with no memset, M3 <= 3.5 (row ~5.5)
- `layer=router load=decode_* m=*` unchanged: router m <= 32 untouched

## Risk

Milestones: M1 build OK, 9/9 tests, the router kernel name on both rows (or a printed gate decision keeping master, plus the cold-L2 table); M2 as above; M3 router rows within ~2.5x of floor, about +4% on the verify_32 load and +3% on verify_16. Kill: if the wgmma mainloop without the reduce is not under master's cold-L2 eager time at m=256 (14.7 us in stage 1's table), the CUDA router line is dead; leave the router on Triton. Overlaps router-hilo-tall-tile-m65-m256 (a Triton config change on the same rows). Run the primary first: if it lands, this stage's gate measures against it. Fallback: master through the gate. Risk: two prior CUDA router kernels (mma.sync) lost; this stage changes the MMA path and the L2 byte count, the two things stage 1 did not.

## Milestones

Rewrite stage 2 of `_router_hilo_kernel + torch.zeros memset at router m=65..256 (hilo 5.0 + 0.8 us at m=128, 7.0 + 0.7 us at m=256)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T212522Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a legitimate stage 2 of a neutral rewrite. It changes the two things stage 1 left alone: the MMA path (mma.sync becomes wgmma) and the L2->SM byte count (-33% at m=256). Nothing landed or held already covers it. Four things must change at build time or this check gets misread: build on the current master 36c26199 (blob ad8d7254, not 353e3b19), replace the single-round 1.02x gate with the new two-round >=1.03 + IQR gate, fix the setmaxnreg/producer-warp mismatch, and never cache a TMA descriptor for x.

### Justification
A. The mechanism is plausible. On current master (741072dc) the router rows are 7.61 us at m=128 (_router_hilo_kernel 5.0 + memset 0.8, floor 2.2) and 9.72 us at m=256 (6.9 + 0.8, floor 2.7). These are the worst row/floor ratios left at m>=128. The hypothesis quotes 7.38 and 9.31 for those rows, which is stale; use 7.61 and 9.72 as the M2 baseline. At m=256 hi+lo is 2.15 GFLOP, so math and L2->SM traffic both matter there. That makes wgmma plus taller token tiles the right two levers. The tiling arithmetic checks out. At m=256, BLOCK_T 128 gives 2 token tiles x 4 column groups x 16 split = 128 CTAs and 8 clusters of 16. That is within the 14 active clusters stage 1 measured. L2->SM traffic is x 16.8 MB + hi/lo 16.8 MB = 33.6 MB. The two m=128 candidates check out the same way. Each CTA streams about 256 KB of operands, roughly 2.3 us at about 110 GB/s per SM from L2. M2 (kernel <=6.0 us at m=256, <=4.5 us at m=128) is therefore physically reachable. M3 (<=4.5 us at m=256) is tight but not impossible. The expected directions agree with TASK: both rows carry weight 62, and the decode_* rows are untouched. The prior is correct and the difference is real. Stage 1 (b7b2b55d) used mma.sync over the incumbent's 64x64 tiles at equal bytes. Stage 1's harvest note itself recommended 'stage 2 should be wgmma + ...'. FINDINGS forbids stage-2ing 'bf16-verify-swapab-wgmma-cluster-dsmem-rewrite', but that is the bf16 qkv/pooler/unpooler layers at N=2304-8192, not the N=256 router, so this is not a repeat. router-hilo-tall-tile-m65-m256 (the Triton tall-tile primary) was harvested neutral and NOT landed: router verify_32 went 9.72 -> 9.99 after its gate mis-routed. The incumbent to beat is therefore master's (64,64,SPLIT_K 16/8) hilo kernel plus torch.zeros. Weakest link: the claim that stage 1 lost 'on mma.sync'. At m=128 the math is small, yet stage 1's mainloop with no reduce still ran 15.2-17 us against master's 12.5 eager. That gap looks like latency or pipeline trouble, not MMA throughput: at cluster 16 each CTA's K-slice is only 512, i.e. 8 K-steps of 64, so a 4-6 stage ring spends most of its life filling and draining. wgmma does not fix that. Second weakest link: with BLOCK_T 128 the DSMEM reduce tile doubles to 64x128 fp32 = 32 KB per CTA. Stage 1 already measured its reduce tail at about 1.1 us with a half-size tile, more than the 0.8 us memset the reduce removes. The reduce can eat the whole memset saving unless it is a reduce-scatter (each CTA owns 1/16 of the rows and does 16-B vector ld.shared::cluster reads from its peers). The kill criterion is also stated in eager cold-L2 terms (14.7 us), while the selector and the judge use graph replay. Restate it in graph-replay net terms: the no-reduce kernel must beat about 6.9 us at m=256.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m (landed gain): it defines the incumbent _router_hilo_kernel + torch.zeros at m>32 that this rewrite replaces. It is the comparison itself, not a confound. Its configs (64,64,SPLIT_K 16 at m<=128, 8 at m=256) are what the gate measures against, and they are unchanged on current master.
- router-hilo-at-m8 / sglang-gemm-dispatch-parity (landed): they set the hi+lo bf16 split with fp32 accumulation. The new kernel keeps that numeric contract, so there is no confound beyond a changed accumulation order (hi then lo per k16 into one accumulator), which the self-check covers.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (current master, 36c26199): it does not touch router m>32. It is listed only because the hypothesis names base blob 353e3b19, which is two landings stale. Built on the stale blob, the attribution would compare against the wrong base and drop the combo's wins (the LESSONS 03:41Z failure mode). Build on 36c26199 / blob ad8d7254 and identify the base from kernel names in the profile.
- router-hilo-tall-tile-m65-m256 (neutral, NOT landed): it does not confound, but its gate mis-route (9.72 -> 9.99) is direct evidence that this exact cell is noise-prone (±3%) under a single-round 1.02x gate.

### Interference and A/B needs
No companion is warranted. This is a rewrite, judged on the replaced kernel's own top_kernels time at m=128 and m=256 against 5.0+0.8 and 6.9+0.8 us. Nothing landed touches router m>32 except the incumbent itself, so neither an ab-revert nor an ab-tune would disambiguate anything. The in-prepare diagnostics already provide the same-run split: candidate vs master graph-replay medians, a no-DSMEM-reduce mode that separates mainloop from reduce tail, printed L2->SM bytes, and profiler kernel-name assertions. The one ambiguity that matters is a mis-routed gate, as happened in the tall-tile run. The fix for that is the new LESSONS gate, not an extra run: master's _cell_gate run twice interleaved, both rounds >=1.03 (the hypothesis says >=1.02, single round), candidate IQR <=1.5x master's. Also print the no-reduce and full-kernel medians in graph replay, not only eager, so the kill decision for stage 3 can be read from this one check even if the gate keeps master.

### Open questions
- Is the DSMEM reduce a reduce-scatter (each of the 16 CTAs owns 4 of the 64 weight rows x BLOCK_T and reads 15 peers with vector loads), or a gather onto rank 0? With a 32 KB tile per CTA, a gather will cost more than the 0.8 us memset it deletes.
- With cluster 16 the per-CTA K-slice is 512 (8 K-steps of 64). Should BLOCK_K be 128 (4 steps, 2 wgmma pairs per stage) or cluster 8 with K-slice 1024 (64 CTAs at m=256), so the ring is not all prologue? The hypothesis does not offer the selector a cluster-8 candidate.
- At m=128 the 64-CTA candidate leaves half the SMs idle. Is per-SM L2 bandwidth, rather than total bytes, the binding limit? The selector should measure both m=128 candidates; do not pick one on L2 bytes alone.
- wgmma B operand = the x tile [BLOCK_T x 64K] in K-major layout with 128B swizzle: confirm that the descriptor's LBO/SBO match the TMA box (64 bf16 = 128 B rows). A silent layout mismatch would pass small-m self-checks and only show on outlier rows, so run the outlier and power-of-two rows at every self-check m.
- Should TMA multicast of the weight tile across token tiles (m=256, 2 token tiles) be staged now or left to stage 3? It would cut weight L2->SM bytes from 16.8 MB to 8.4 MB.


## Implementation notes

Built 2026-10-26 on CURRENT master: `git checkout master -- muse_bf16_proj/submission.py` in the worktree (blob ad8d7254, as the review requires, not the stale 353e3b19 / stage-1 branch). Only `<worktree>/muse_bf16_proj/submission.py` changed (+834/-5 vs master). Stage 1's mma.sync kernel was NOT ported; its scaffold ideas (cluster launch, reduce, self-check, occupancy gate, per-launch prints) were rewritten around wgmma. Nothing was run (no GPU on this node); the CUDA source was never compiled here.

What changed (new section (10), inserted before `_prepare_state`; hooks in `NEW_CELL_FLAGS`, `AB_CASES`, `_ab_flags`, `_ab_component`, `_prepare_state`, `prepare()`, `kernel_fn`):
- `rt2_wgmma_swapab_kernel<BT, CS, ST>` (load_inline, sm_90a only): 160 threads, warps 0-3 consumer warpgroup, warp 4 lane 0 producer. Per 64-K step the producer TMA-loads hi tile [64 rows x 64 K], lo tile of the same rows and the x tile [BT x 64 K] (128B swizzle) into an ST-deep full/empty mbarrier ring; the consumer issues wgmma m64nBTk16 twice per k16 (A = hi rows then lo rows, B = x tile, both K-major smem descriptors, SBO 1024 B, LBO 1) into ONE fp32 accumulator. D[weight row][token] goes to smem P[token][68] (aliasing the consumed ring), barrier.cluster, rank r reduce-scatters tokens [r*BT/CS, +BT/CS) by 16-B `ld.shared::cluster` reads of all CS peers in fixed rank order, one fp32 store, second barrier.cluster. Output is `torch.empty` (no memset node). Instances (BT, CS, stages): (64,16,4), (128,16,3), (128,8,3); smem about 97-99 KB so 2 CTAs/SM fit.
- Candidates per band: a (m 65..128, rep m 128): (64,16)=128 CTAs, (128,16)=64 CTAs; b (m 129..256, rep m 256): (128,16)=128 CTAs, (128,8)=64 CTAs (this is the review's cluster-8 / K-slice-1024 open question, offered to the selector). L2->SM bytes are printed per tiling vs master's hilo tiles.
- Per-band state flags `rt2_a_ok` / `rt2_b_ok` (+ `rt2_cfg_a/b`, per-state error word `rt2_err`), in `NEW_CELL_FLAGS` so the A/B/A/B two-state graph check inherits and can switch them off; `AB_CASES` gained router m = 65, 200, 256 (m = 128 was already there and now maps to band a). Flag off or any miss = master's exact `_router_hilo_forward` path; no unrelated flag is touched.
- Self-check (`_rt2_check`) per instance at m = 65, 128, 129, 200, 256: random rows, x64 outlier rows, x1024 outlier entries, about one row in eight (plus the last) a single power-of-two entry; rel L2 vs fp64 router <= 1e-5, vs fp64 hi+lo <= 5e-6, worst row <= 5e-5, one-hot rows <= 1e-3 of the original column, finite, second launch bit-identical, error word 0; flushed print before and synchronize + error word + flushed print after every launch.
- Selector (`_rt2_gate`): master's ACTUALLY ROUTED path (kernel_fn, band flag off = hilo + torch.zeros) and each candidate (kernel_fn, flag on) captured as graphs on the same synthetic x, 256 MB flush before each replay, 41 reps x 2 interleaved rounds (all graphs once per rep, rotating order), empty-graph baseline subtracted. A candidate wins only if BOTH rounds reach master/cand >= 1.03 AND its IQR <= 1.5x master's in both rounds. Per round it prints master and candidate medians/IQRs and the mode-0 no-DSMEM-reduce graph-replay median (the kill-criterion read, in graph-replay terms). The winner gets one torch.profiler call asserting exactly one `rt2_wgmma_swapab_kernel` (no memset node) before the flag is set. PATH CHOSEN per band is printed even when master is kept, and repeated (with the final post-A/B flags) as the LAST lines of prepare().

Review fixes applied (where the review corrected the spec, the review won):
1. Base: built on master blob ad8d7254 (36c26199), not 353e3b19.
2. Gate: single-round 1.02x replaced by two interleaved rounds, both >= 1.03, candidate IQR <= 1.5x master's; master timed through kernel_fn with the flag off.
3. setmaxnreg/producer mismatch: removed setmaxnreg altogether. It is warpgroup-wide, a lone producer warp cannot run it legally, an unbalanced `.inc` can block without bound, and the kernel is not register-limited (about 64 accumulator regs at BT 128). Deviation from the spec's "setmaxnreg", per the review's own framing.
4. No cached x descriptor: x's tensor map is re-encoded on the host on every call from that call's pointer and stride and baked into the graph like a pointer argument; hi/lo maps are re-encoded per call from the state's own tensors; nothing in state or module globals (only a host function pointer to cuTensorMapEncodeTiled, resolved via dlsym).
5. Kill criterion restated in graph-replay terms (the printed no-reduce median is compared against master's net ~6.9 us at m=256).
Open questions answered in code: reduce-scatter, not gather (each rank owns BT/CS tokens x 64 rows, 16-B peer reads); cluster-8 candidate offered at m 129..256; both m=128 candidates timed by the selector, not picked on bytes; descriptors K-major 128B swizzle with SBO 1024 B matching the 128-B TMA box rows, and outlier plus power-of-two rows run at every self-check m; weight-tile multicast left to stage 3.

Deviations / things to know: TORCH_CUDA_ARCH_LIST=9.0a around the build, `-O3 -lineinfo -std=c++20`, arch flags printed, `#if __CUDA_ARCH_FEAT_SM90_ALL` guards wgmma with an `#else` that sets error bit 1<<30 (self-check reads it). Smem is sized for 2 CTAs/SM because at 1 CTA/SM only about 7 clusters of 16 can be co-resident (< the 8 needed); every instance must pass spill == 0, CTAs/SM >= 2 and max active clusters >= tiles x 4. Bounded waits: clock64, about 0.6 s, then the error word. The phase starts only if state-A's build is under 150 s old and stops starting steps after 70 s, printing `skipped: deadline`. Build time and regs/spills are printed. No subprocess, host sync, allocation on values or data-derived caching in kernel_fn; the only per-call host work is descriptor encoding and one `torch.empty`.

## Rules check

- [x] CUDA-graph capturable: kernel_fn's new branch does only shape/cfg/state-flag tests, `torch.empty`, host-side cuTensorMapEncodeTiled and cudaLaunchKernelEx; no `.item()`/`.cpu()`/synchronize in kernel_fn (diff checked; every synchronize/.item() is in prepare-only helpers).
- [x] Nothing derived from `data` is cached across calls: x's tensor map is rebuilt every call; routing constants come from synthetic data and the weights in prepare().
- [x] Weights: hi/lo are master's existing lossless prepare()-time router split (BF16 hi + lo, FP32 accumulation, the permitted exception); no new weight copies.
- [x] Activations BF16 where SGLang keeps BF16: x stays bf16; the router output stays fp32 as in master.
- [x] n/a: attention rules (this is the router GEMM; no attention in the diff).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: dlopen/dlsym only resolves the CUDA driver's tensor-map function; no child process anywhere in the diff.
- [x] All work happens inside the timed call; no output caching or replay (one kernel per call into a fresh `torch.empty`).

okc_lint (python3 -I okc_lint.py ... --problem muse_bf16_proj): `OK with warnings` (rc 0); the warnings are host-sync / dtype notes on prepare()-only code (synchronize, `.item()` on the error word, 256 MB uint8 flush buffers), none inside kernel_fn's call graph. `python3 -I -m py_compile` passes; `__pycache__` deleted.

## Pre-flight audit

Verdict: dispatch-with-fixes. Read-only audit (no GPU/nvcc on the node); nothing was compiled.

- Fix 1: the gate's IQR veto floored master's IQR at 0.05 us (`max(im, 0.05)`); a timer-resolution IQR could veto a real win. Now `max(im, RT2_IQR_FLOOR_US)` with RT2_IQR_FLOOR_US = 0.5. okc_lint OK with warnings (prepare-only syncs), py_compile OK, __pycache__ removed.
- Per-call x descriptor: graph-capture safe. cuTensorMapEncodeTiled is a pure host encode into a stack CUtensorMap passed by value as __grid_constant__ (baked into the kernel node like a pointer; the A/B check refills x in place, so replay stays correct). No sync, no allocation, no cached encode (only the function pointer, resolved in the first self-check launch, before any capture). Same per-call pattern as muse_prefill_layer and muse_moe_w4a16; master's own TMA is raw cp.async.bulk, no tensor maps.
- Cluster 16: cudaFuncAttributeNonPortableClusterSizeAllowed is set when CS > 8 (set_attr, before info() and every launch); candidates need spill 0, CTAs/SM >= 2 and cudaOccupancyMaxActiveClusters >= tiles x 4 before any launch; failures land on master with a PATH CHOSEN line.
- Output: torch.empty, every element written exactly once (emulated for all m 65..256, BT 64/128, CS 8/16; tail tokens guarded by m0 + tl < m, OOB x rows TMA-zero-filled).
- Emulation: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/emu_rt2_wgmma_swapab_cluster.py (128B swizzle store, descriptor reads, accumulator fragment map is a bijection, P layout, reduce-scatter, K slicing); rel L2 vs fp64(hi+lo) about 2e-7, ALL OK.
- Deadlines: master prepare is about 4.5 s (archive log 741072dc); the file has no import clock; sibling load_inline builds take 30-49 s each, so rt2's compile adds roughly 40-60 s at import, well inside 10 min. Phase start deadline 150 s is measured from state-build start (about 4.4 s in), so it will not skip.
- Residual: unverified by compile or hardware (wgmma asm, 160-thread layout, 16-cluster co-residency); any miss falls back to master.
