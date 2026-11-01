---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix
created: 2026-10-25T054545Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 6, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit. Use that run's
  judged profile rows to enable each (T bucket, kernel) cell independently: cells are B128/B256 x
  {w13+silu, w2}. Four kernel_fn paths, chosen by T and two prepare booleans per bucket (shape +
  flag only): (a) wgmma w13+silu then wgmma w2; (b) wgmma w13+silu writes h in the route-row
  layout Marlin's w2 consumes as cache2 (sorted by route index, [T*8, I] bf16) then Marlin w2 with
  master's flags and block_size_m; (c) Marlin w13 + act_and_mul into cache2, then wgmma w2 fed
  Marlin's cache2 (s2-s4 already self-checked this exact pairing as 'w2 (fed Marlin cache2)', rel
  3.5e-3); (d) master. Both GEMMs must use one align: when a Marlin kernel runs in the cell, align
  at master's block_size_m and give the wgmma kernel that sorted_ids/block_experts (wgmma consumes
  m-blocks of 16; if master's block is 32 at B256, either reuse 16 with a second align only for
  wgmma or keep (a)/(d) for that bucket; state which in the log). Static table from the s5-forced
  profile: enable a cell where its wgmma kernel row was <= 0.95x the matching Marlin row (w13 vs
  Marlin call 1 + act_and_mul; w2 vs Marlin call 2) in the same profile; then confirm each enabled
  cell in prepare with the LESSONS-173 graph-replay selector at >=1.02x on the judge-like routing,
  else drop to (d). Self-check every enabled mix with the three-state refill-replay check against
  fp32 dequant. Hard rules: CUDA-graph capturable; kernel_fn does no host sync
  (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except
  and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in
  registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes.
  The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may
  hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=128 *
    direction: faster
    why: keeps only the wgmma kernel(s) that beat their Marlin counterpart in the s5-forced judged profile
  - shapes: bs=256 *
    direction: faster
    why: same, B256 cells
  - shapes: bs=16 *
    direction: flat
    why: "control: T<128 master"
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu and/or Marlin w2 at 128<=T<=512
review: proceed
status: hold
review_at: 2026-10-25T054837Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T212857Z
base_sha: 57e77548ba31c2145d575df62752f62684b2f6ad
base_blob: 77f34503659c90a5953f91a93ba6e1a2b02e4613
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix
mode: null
restack_of: null
stale_base: true
hold_reason: "superseded (phase C 2026-10-27 2215Z): implemented, but its data-derived table routes B128 -> master and B256 -> wgmma for both GEMMs, exactly the s5 bucket-gate twin (4e4a0e26 / 7be125e9, 1.3744x); paths (b)/(c) are never selected, and the base is the stale s5 commit (master 1.7187), so a check answers no new question. Implementation left unstaged in worktree .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix (never built by nvcc)."
---

# cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix

## Change

Stage 6, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit. Use that run's judged profile rows to enable each (T bucket, kernel) cell independently: cells are B128/B256 x {w13+silu, w2}. Four kernel_fn paths, chosen by T and two prepare booleans per bucket (shape + flag only): (a) wgmma w13+silu then wgmma w2; (b) wgmma w13+silu writes h in the route-row layout Marlin's w2 consumes as cache2 (sorted by route index, [T*8, I] bf16) then Marlin w2 with master's flags and block_size_m; (c) Marlin w13 + act_and_mul into cache2, then wgmma w2 fed Marlin's cache2 (s2-s4 already self-checked this exact pairing as 'w2 (fed Marlin cache2)', rel 3.5e-3); (d) master. Both GEMMs must use one align: when a Marlin kernel runs in the cell, align at master's block_size_m and give the wgmma kernel that sorted_ids/block_experts (wgmma consumes m-blocks of 16; if master's block is 32 at B256, either reuse 16 with a second align only for wgmma or keep (a)/(d) for that bucket; state which in the log). Static table from the s5-forced profile: enable a cell where its wgmma kernel row was <= 0.95x the matching Marlin row (w13 vs Marlin call 1 + act_and_mul; w2 vs Marlin call 2) in the same profile; then confirm each enabled cell in prepare with the LESSONS-173 graph-replay selector at >=1.02x on the judge-like routing, else drop to (d). Self-check every enabled mix with the three-state refill-replay check against fp32 dequant. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=128 *` faster: keeps only the wgmma kernel(s) that beat their Marlin counterpart in the s5-forced judged profile
- `bs=256 *` faster: same, B256 cells
- `bs=16 *` flat: control: T<128 master

## Risk

Dispatch only after cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced is harvested with wgmma kernel names in the judged profile. If that run shows both kernels winning in both buckets, skip this (land the s5 gated twin instead); if both lose in both buckets, close the wgmma line. Overlaps the held cuda-nvfp4-wgmma-rega-s2-per-kernel-enable / w2-only-midm in intent; this one is grounded on judged rows and supersedes them.

## Milestones

Rewrite stage 6 of `Marlin w13+silu and/or Marlin w2 at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054837Z by an adversarial subagent (model opus).

**Verdict: proceed.** Run this rewrite stage only after s5-copy-safe-forced has been harvested and its judged profile actually contains wgmma rows with a mixed per-cell outcome. Before the s5 data exists it has nothing to stand on, and the prepare-time evidence so far makes a mixed outcome unlikely: both kernels beat Marlin by about the same margin.

### Justification
The mechanism is plausible. In the judged profiles, Marlin's two calls take 306.2 us at bs=128 and 483.2 us at bs=256 (s4, job 4e7f8be895a145ba). The floors are 101.3 and 108.9 us, so there is 3-4.5x of headroom on 2 of the 6 equally weighted loads. A per-cell enable that keeps only the winning kernel is a sound way to stage the rewrite. The expected directions (bs=128/256 faster, bs=16 flat) fit TASK.md: T<128 runs master. The weakest link is that the s6 decision table has no data behind it yet. No judged profile has ever contained a wgmma row: s1 failed to build, s2 fell back, s3 failed, and in s4 the copy-unsafe gate silently ran Marlin. The only per-kernel evidence is s4's prepare timing. At M=128 cold that was w13+silu 385.6 vs Marlin w13+act 611.6 (0.63x) and w2 185.4 vs 310.0 (0.60x); at M=256 it was 457.2 vs 852.0 (0.54x) and 230.2 vs 425.9 (0.54x). Both kernels win by nearly the same ratio, so the most likely s5 outcome is all cells win (s6 is skipped in favour of the s5 gated twin) or all cells lose (the line closes). Only the mixed case gives s6 something to do. Second weakness: the prepare timings ran an all-local routing (1024 local routes at M=128). There Marlin reads ~920 us against 306 us judged, so the routing does not look like the judge's. The LESSONS-173 selector must use the judge-like skewed routing, where per-expert M is much smaller and the two kernels' ratio may differ. Third: path (b) needs a new wgmma epilogue that scatters h into Marlin's cache2 route-row layout, and that code has never run. Path (c), Marlin cache2 into wgmma w2, has been self-checked at rel 3.5e-3. Finally, a single profile's 0.95x threshold sits inside the about ±20% run-to-run noise of a Marlin row (REWRITE_READ). The static table should only nominate cells, and the selector's ≥1.02x graph-replay confirmation should decide.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: sets master's block_size_m at T>=128 (16 at M=128, 32 at M=256 per the s4 log). It defines the incumbent Marlin cells that paths (b), (c) and (d) compare against, and it decides whether wgmma can share the Marlin align (16-row m-blocks) at B256. If it were reverted, the incumbent would be slower and the per-cell table would shift towards wgmma. That is not a confound of the s6 result, but the table is only valid for this block_size_m.
- align-parallel-per-expert-verify: master's T>32 align. Mixed paths (b) and (c) need it plus possibly a second align for wgmma at B256 (~4-7 us). That cost lands on the mixed cells and not on (a) or (d), so a mixed cell's gain must clear it.
- w2-marlin-atomic-add-small-m: T<=32 only, so it is disjoint from the 128<=T<=512 cells. No confound beyond the rebase that s4 already did.
- post-expert-norm-triton-small-m / SGLang fused_post_expert_norm at bs>=32: every path feeds the same combine. Path (a)'s wgmma w2 output layout ('fused norm skips remote rows') differs from Marlin's, so check that the combine input layout is identical across (a), (b), (c) and (d), or the combine time becomes a hidden per-path variable.

### Interference and A/B needs
No companion. This is kind: rewrite, judged per cell on the replaced kernels' own profile rows (wgmma w13+silu vs Marlin call 1 + act_and_mul; wgmma w2 vs Marlin call 2) within the same run. That attribution is internal, and no reverted or retuned landed hypothesis would sharpen it. The landed changes that touch this range (block_size_m pick, T>32 align) define the incumbent and are not alternative explanations for a per-kernel row delta. With 5 moe checks left and s5-forced still unrun, a companion would spend a check that s5 needs. The real control is the dispatch gate in the hypothesis's own Risk section: run s6 only if s5's judged rows split. If s5 shows all-win or all-lose, do not run s6.

### Open questions
- Did s5-forced log LAUNCHED = wgmma and produce judged wgmma kernel rows at bs=128 and 256? If not, s6 has no table and should not run.
- Did the per-cell judged ratios actually split across the 0.95x line? The prepare ratios (0.54-0.63 for both kernels) predict they will not. If they do not, this slug should close as superseded by the s5 gated twin or the line closure.
- At B256, does the wgmma kernel get its own 16-row align (an extra ~5-7 us launch) or reuse Marlin's 32-row align? The spec leaves this open. The cost of a second align must be inside the selector's timed graph so the ≥1.02x test sees it.
- Is the prepare selector's 'judge-like routing' the skewed, this-GPU-busiest distribution (per-expert M far below all-local)? s4's prepare timings used 1024 local routes at M=128 and overstated Marlin by about 3x against the judge.
- How are the bs=257 and skewed bs=128 tests bucketed (B256 cell vs B128 cell)? Each must hit an enabled mix that the three-state check has verified.

## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix` on base 57e77548 (the s5-forced impl commit, as the change asks; `stale_base: true`: master has since moved on to 1.7187x, campaign-eb-w6m-fork). Only `muse_moe_w4a16/submission.py` changed: 1 file, +405/-59. Nothing committed or staged.

**The static table this stage stands on.** s5-forced (job d4ad4f31d2764050) logged LAUNCHED = wgmma and its judged profile has wgmma rows: B128 w13+silu 238.4 / w2 110.9 us, B256 271.0 / 126.1 us. Deviation from "the matching Marlin row in the same profile": that profile has no Marlin row at bs=128/256 (the path was forced there), and no judged profile splits Marlin's two calls (it is one "Marlin<...> 2 calls" row). So the matching rows are the incumbent judged pair that both reviews use (s4 job 4e7f8be895a145ba: 306.2 / 483.2 us; act_and_mul 3.7 / 5.7 us), split by s5-forced's own prepare per-call cold timing at the judge-like routing (Marlin m13 : m2 = 240 : 128 us at T=128, 318 : 166 us at T=256). wgmma / Marlin: B128 w13 (vs call 1 + act_and_mul) 1.172, w2 (vs call 2) 1.041, both off; B256 w13 0.839, w2 0.761, both on. Cross-checks give the same table: master job 7a0d900a1dd34666's pair (326.6 / 436.7 us) gives 1.100 / 0.976 / 0.926 / 0.842, and s5-forced's prepare per-kernel ratios give 1.22 / 1.13 / 0.94 / 0.89. Encoded as `WG_CELL_ROWS` -> `WG_CELL_TABLE = {128: (False, False), 256: (True, True)}` (`WG_CELL_NOMINATE = 0.95`) and printed in prepare's summary.

**Consequence: read this before spending a check.** The cells do not split inside a bucket, so the nominated mixes are B128 = (d) master and B256 = (a) wgmma w13+silu + wgmma w2. Paths (b) and (c) are implemented and gated, but this table never selects them, so this run cannot measure them. The reachable behaviour is the s5 bucket-gate twin's (jobs 4e4a0e26c85944f2 / 7be125e9e1ea42e3: B128 MARLIN, B256 WGMMA, 1.3744x), minus its B128 selector run. The adversarial review's open question 2 says that in this case the slug should close as superseded by the s5 gated twin. I implemented it as briefed and leave that dispatch decision to the caller. Expected `bs=128 * faster` cannot hold under this table: bs=128 runs master, so it is flat.

**What changed, and where (submission.py only):**
- Module docstring: a Stage 6 paragraph. New constants `WG_B256_MIN_TOKENS = 256` (B128 = [128, 255], B256 = [256, 512]) and `WG_TOKEN_SHIFT = 3`.
- CUDA/C++ (load_inline source; ext name bumped `_v5` -> `_v6`):
  - `consume_item` and `wg_kernel` take an `int route_rows`. When it is set (path b), the w13+silu epilogue stores each staged h row at `cache2[sorted_ids[slot]]` instead of `h_sorted[slot]`, real routes only (sentinel and padding slots are skipped), the same pattern as w2's epilogue. With route_rows = 0, path (a) behaves exactly as before.
  - `wg_gemm`: `which` 0 and 1 are as before; the new 2 = w13+silu into route rows. The output-width check is generalised, a new check requires `out.size(0) >= num_routes` for route-row outputs, and the launch macro is now `WG_LAUNCH(G_, R_)`.
  - `wg_gather_kernel` / `wg_gather`: a new `row_shift` argument (3 = token rows of x, as before; 0 = route rows, for path (c)'s cache2 -> sorted-slot gather), with a size check. The `_WG_CPP` declaration is updated to match.
- Python:
  - `_master_bsm(T, count)`: master's block_size_m pick (shape only: 16 at 128 <= T <= 172, 32 at 173..512). `_moe_marlin` itself is untouched.
  - `_moe_wg_mix(..., wg_w13)` is paths (b) and (c). Both GEMMs use one align, master's own `_align_parallel` at block 16, which is master's block_size_m wherever a mix may run. (b): gather x -> wgmma w13+silu (which=2) into cache2's route rows -> master's Marlin w2 (master's flags, block 16). (c): master's Marlin w13 + `silu_and_mul` into cache2 -> `wg_gather(cache2, row_shift 0)` into wg_hs -> wgmma w2 -> cache3. All four paths end in the same `fused_post_expert_norm_weighted_sum` on the same [T, 8, H] route rows (the review's combine-layout confounder).
  - `kernel_fn` branches on shape and flags only. The bucket is 128 if T < 256, else 256, and it reads `state['wg_on_<b>_w13']` and `state['wg_on_<b>_w2']`. Both flags on -> (a) `_moe_wgmma`. One flag on -> (b)/(c) `_moe_wg_mix`, but only if `_master_bsm(T) == 16`. Otherwise (d) `_moe_marlin`.
  - `_wg_nominate` turns the static table into one nominated mix per bucket. A lone cell where master's block_size_m != 16 falls back to (d). Each bucket's table line states the align choice, as the change asks ("state which in the log"); at B256 it reads "master block_size_m 32 at T=256: (a)/(d) only, no second align for wgmma".
  - `_wg_copy_check(..., mixes)` is the s5 three-state refill-replay check: A, deepcopy B and clone-walk C; refilled replays A,B,C,A,B; an fp32-dequant reference with a 1.5x-Marlin bound; the launch marker. It now forces exactly the nominated mixes into every state's cell flags and runs only their buckets' M (`WG_COPY_MS_BY_BUCKET`: B128 -> 128; B256 -> 256, 257).
  - `_wg_select(..., mixes)` is the s5 bucket-gate twin's LESSONS-173 selector, with `_wg_calibrate`, `_wg_graph_of` and `_wg_ab_graphs` copied verbatim: full-path graphs, a >= 256 MB flush before every replay, 41 interleaved reps, the empty-graph baseline subtracted. It now times each nominated mix's full path against master's full path at the bucket's T. A mix is confirmed only at >= 1.02x on the calibrated judge-like Zipf routing (the random routing is printed, not gating), with error word 0, the launch marker, a finite output and rel-L2 <= 2e-2 vs Marlin on both routings.
  - Final flags = nominated, AND the path is ok (canary, diag, self-check, copy check, marker, error word), AND the selector confirmed the mix; otherwise (d). s5's print-only per-kernel cold timing (`_wg_time_vs_marlin`) is kept after the selector as per-cell evidence. The summary is renamed STAGE 6, and the per-bucket `PATH CHOSEN B128 / B256 = (x) ...` lines plus a combined line are printed last.

**Deviations and choices, with reasons:**
- Matching Marlin rows: they are not in the same profile (see above), so they are estimated from the incumbent judged pair, split by s5's prepare per-call ratio. The three estimates agree.
- B256 align: I took the change's option "keep (a)/(d) for that bucket", because master's block_size_m is 32 at every T >= 173. So no second align is added, and mixed paths exist only at 128 <= T <= 172 (one shared align at 16). A lone B128 cell at T 173..255 would run (d).
- "Confirm each enabled cell" is done per nominated mix: a bucket's enabled cells are timed together against (d), because at B256 one cell cannot run alone without a second align. A failed confirm drops the bucket to (d), as the change says, with no partial fallback.
- The selector gate is the change's single-round >= 1.02x (LESSONS line 170, "LESSONS-173"). LESSONS line 24 recommends two interleaved rounds, both >= 1.03; I did not adopt it, since it is not in the change.
- The copy check no longer forces wgmma at M = 128, because B128 is (d). Path (a) at M = 128..257 is still exercised by the unchanged s5 canaries and self-check (eager + graph).

**Checked offline (no GPU or torch on this node):**
- py_compile OK. okc_lint OK with warnings: 68, all prepare-only host-sync or uint8 flush lines; 0 errors.
- The pure-Python pieces ran offline: the table, `_master_bsm` (16 -> 32 at T=173), the nomination, and `_wg_audit_cpp` on the new C++ source, which reports no findings, so the audit will not switch the path off.
- The load_inline CUDA source and cpp declarations type-check under `g++ -fsyntax-only` against stub CUDA/torch headers (launch configs stripped), with and without `__CUDA_ARCH_FEAT_SM90_ALL`: 0 errors. The same harness passes the s5 source and rejects three injected errors (consume_item arity, a dropped kernel argument, a declaration mismatch).
- kernel_fn's dispatch, run with the paths stubbed: with the final table, T 256..512 -> (a) and everything else -> (d); with no flags, everything -> (d); a hypothetical B128 (b) or (c) runs mixed only at T 128..172.
- Not checked: nvcc and ptxas (register count, and the codegen effect of the `continue` in the epilogue store loop), and anything numeric. Paths (b) and (c) have never run on a GPU.

## Rules check

Checked against the actual `git -C .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix diff` (1 file, +405/-59):

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; no allocation or control flow on input values (shapes and `config` are fine). kernel_fn and everything it reaches (`_master_bsm`, `_moe_wg_mix`, `_moe_wgmma`, `_moe_marlin`, `_wg_gemm`, `_align_parallel`) contain no item/cpu/tolist/synchronize/print/try/assert (AST scan, and lint shows 0 errors). They branch only on T and the four prepare-time bools, and allocate only by shape. Tensor maps are host-encoded per call (s5).
- [x] Nothing derived from `data` is cached across calls. The cell flags and the table are prepare-time constants. Per-call scratch (wg_xs, wg_hs, cache2, cache3) is overwritten on every call and never read back across calls.
- [x] Weights: repacked losslessly in `prepare()` only; NVFP4 stays NVFP4, BF16 stays BF16. No new weight handling: the wgmma bank is s5's lossless retiling (round trip asserted), and the Marlin weights are SGLang's repack. Dequant stays in registers.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. h is written as bf16 into cache2 (path b) just as Marlin + act_and_mul writes it; x, cache2 and cache3 stay bf16; fp32 only in accumulators and the norm, as before. KV cache and queries: n/a (MoE problem, no attention).
- [x] Attention sees every key: n/a (MoE problem, no attention). The MoE analogue holds: every local route is computed on all four paths, and only padding and remote rows are skipped.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. None added (no new imports at all).
- [x] All work happens inside the timed call; no output caching or replay. kernel_fn issues every launch on every call. Prepare-time graphs, states and buffers are freed before return (gc + empty_cache, as in s5).
- [x] (extra) No module or static global holds device pointers, TMA maps or buffers. `_wg_audit_cpp` on the new source reports no findings, and the new Python globals are floats, bools and ints only.

okc_lint output (`python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`, exit 0):

```
warn  host-sync    L388: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L390: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L392: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L410: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L415: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L416: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1310: q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L1311: q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L1365: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1544: res["tiles"] = [(i, int(t_nf[i]), float(t_rel[i])) for i in torch.topk(score, min(3, t)).indices.tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1546: idx = bad.nonzero().flatten()[:4].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1614: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1616: print(f"{WG_TAG}     [{tag}] done {what}; device error word {int(err[0].item())}", flush=True)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1626: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1629: used = int(n_padded.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1637: chunks = _wg_chunks(block_experts[:used // WG_BSM].tolist(), count)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1650: ev["buckets_seen"] = set(int(v) for v in bucket_of.unique().tolist() if v >= 0)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1693: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1699: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1704: ev["err_word"] = int(err[0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1731: for t in torch.randperm(tokens, generator=gen)[:c].tolist():  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1736: row = per_tok[t] + [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:need].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1737: ids.append([row[i] for i in torch.randperm(TOP_K, generator=gen).tolist()])  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1752: "canary M=128": (128, [[start] + [remote[i] for i in torch.randperm(len(remote), generator=gen)[:TOP_K - 1].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1792: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1832: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1840: e.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1873: pick = torch.randperm(len(remote), generator=gen)[:TOP_K].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1882: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1912: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1937: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1939: "skewed": [local_set[:4] + [others[i] for i in torch.randperm(len(others), generator=gen)[:4].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1943: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1964: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1965: if int(state["wg_err"][0].item()):  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1966: print(f"{WG_TAG}   device error word {int(state['wg_err'][0].item())} in graph replay "  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2020: local_perm = torch.randperm(len(local_set), generator=gen).tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2078: flush = torch.empty(192 * 1024 * 1024, dtype=torch.uint8, device=device)  (allowed only where SGLang does the same)
warn  host-sync    L2106: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2192: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2196: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2208: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2217: e_.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2236: flush = torch.empty(WG_SEL_FLUSH_BYTES, dtype=torch.uint8, device=device)  (allowed only where SGLang does the same)
warn  host-sync    L2265: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2266: err_word, marker = [int(v) for v in state["wg_err"].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2296: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2304: return [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2307: return [local_set[:4] + [others[i] for i in torch.randperm(len(others), generator=gen)[:4].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2403: pads.append(torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device=device))  (allowed only where SGLang does the same)
warn  host-sync    L2408: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2448: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2452: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2463: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2464: err_word, marker = [int(v) for v in st["wg_err"].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2468: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2486: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2653: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2742: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2743: err_word = int(state["wg_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2786: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2803: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2882: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2884: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2898: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3012: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3013: rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3014: if not (torch.isfinite(out).all().item() and rel <= 5e-3):  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix/muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK; the `__pycache__` it left in the worktree was removed.
