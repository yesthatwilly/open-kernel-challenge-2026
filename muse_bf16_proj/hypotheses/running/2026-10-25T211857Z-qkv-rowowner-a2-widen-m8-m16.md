---
problem: muse_bf16_proj
slug: qkv-rowowner-a2-widen-m8-m16
created: 2026-10-25T211857Z
base: master
area: qkv GEMM (m=8/16, bulk row-owner kernel)
change: >
  DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to
  bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; decode_1 12.67 -> 10.54
  us). NT=8 is the padded token count and bulk_run already accepts 1 <= m <= NT, so (1) m == 8:
  route qkv_gate at cfg m == 8 onto the SAME compiled A2 instance (verify_1 qkv now runs
  _bf16_swapab_kernel SPLIT_K 4, 9.6 us kernel, row 12.35). (2) m == 16: add one instance
  bulk_kernel<16, 8192, 16, 3, 2> (NT=16 doubles the x slot: SLOT = 16384 + 16*1040 = 33,024 B,
  S=3 gives 99,328 B, still 2 CTAs/SM; S=4 would not fit two CTAs) for decode_16 (swapab16, 9.5 us
  kernel, row 12.64). Bump BULK_EXT_NAME to a new name since the instance set changes; keep
  -std=c++20 in the nvcc flags and the bounded mbarrier waits (BULK_WAIT_NS). In BULK_CELLS-style
  config add two cells with their own flags bulk_qkv8_ok and bulk_qkv16_ok (appended to
  NEW_CELL_FLAGS, inherited by state B, AB_CASES entries for qkv m=8 and m=16); generalise
  _bulk_route_ok from cfg m == BULK_M to the cell's own m (exact equality, cfg/shape/flag only).
  Each cell goes through master's existing canary (flushed LAUNCH/DONE, synced, error word), the
  rel-L2/probe-row correctness check, the occupancy query (disable the m=16 cell unless it reports
  2 CTAs/SM), then the gate. Gate (hardened from today's evidence): copy master's _cell_gate model
  selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41
  interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent
  interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's
  IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at
  m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged
  9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one
  torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN
  line per cell prints both medians, both IQRs and the kernels. Print per cell the kernel-only
  TB/s next to BULK_REF_TBS. The m=1 cells, all other layers and m=32 stay exactly as master.
expected:
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: one reduce-free launch at 2 CTAs/SM vs SPLIT_K 4 ticket swap-AB 9.6 us kernel; target ~11.0 us row from 12.35
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: same at NT=16; target ~11.8 from 12.64, gate keeps master if the per-token FMA cost eats it
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: flat
    why: A2 m=1 cell unchanged
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild: that row-owner lost +1 us at m=8 and +6 us at m=16 on pooler/unpooler at 1 CTA/SM; this widens master's qkv A2 instance, which runs 2 CTAs/SM and won 1.16x at m=1, against the split-K-4 swap-AB incumbent, behind a per-cell gate"
review: proceed
status: running
review_at: 2026-10-25T212522Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T212547Z
base_sha: e90069716dd467a3801b43a8166dedef6c2ed987
base_blob: ad8d7254a64a52db2a8f56ec0ad262748153a2e8
branch: hyp/muse_bf16_proj/qkv-rowowner-a2-widen-m8-m16
worktree: .worktrees/muse_bf16_proj/qkv-rowowner-a2-widen-m8-m16
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T213458Z
impl_commit: e22908f0fb014a081a5eaced88bad564f5ed6e0f
submission_sha256: 042526c721a4122d7d687eacb0cf536e5acb6c9dfca3ab43e01d8d75e805299d
norm_sha256: a8fc057b6ab26474e2c3a026c7083b8f255fe1f2a2001eba200638182b7e1c69
submission_blob: 463e7b309adf7b48cc294b2f9cff0d85dd4fb9ae
lint_warnings: 36
attempt_file: 2026-10-26T012539Z-check-qkv-rowowner-a2-widen-m8-m16.md
prepared_at: 2026-10-26T012539Z
job: 0124ea8fcaf14b3c
dispatched_at: 2026-10-26T012616Z
---

# qkv-rowowner-a2-widen-m8-m16

## Change

DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; decode_1 12.67 -> 10.54 us). NT=8 is the padded token count and bulk_run already accepts 1 <= m <= NT, so (1) m == 8: route qkv_gate at cfg m == 8 onto the SAME compiled A2 instance (verify_1 qkv now runs _bf16_swapab_kernel SPLIT_K 4, 9.6 us kernel, row 12.35). (2) m == 16: add one instance bulk_kernel<16, 8192, 16, 3, 2> (NT=16 doubles the x slot: SLOT = 16384 + 16*1040 = 33,024 B, S=3 gives 99,328 B, still 2 CTAs/SM; S=4 would not fit two CTAs) for decode_16 (swapab16, 9.5 us kernel, row 12.64). Bump BULK_EXT_NAME to a new name since the instance set changes; keep -std=c++20 in the nvcc flags and the bounded mbarrier waits (BULK_WAIT_NS). In BULK_CELLS-style config add two cells with their own flags bulk_qkv8_ok and bulk_qkv16_ok (appended to NEW_CELL_FLAGS, inherited by state B, AB_CASES entries for qkv m=8 and m=16); generalise _bulk_route_ok from cfg m == BULK_M to the cell's own m (exact equality, cfg/shape/flag only). Each cell goes through master's existing canary (flushed LAUNCH/DONE, synced, error word), the rel-L2/probe-row correctness check, the occupancy query (disable the m=16 cell unless it reports 2 CTAs/SM), then the gate. Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. Print per cell the kernel-only TB/s next to BULK_REF_TBS. The m=1 cells, all other layers and m=32 stay exactly as master.

## Expected

- `layer=qkv_gate load=verify_1 m=8` faster: one reduce-free launch at 2 CTAs/SM vs SPLIT_K 4 ticket swap-AB 9.6 us kernel; target ~11.0 us row from 12.35
- `layer=qkv_gate load=decode_16 m=16` faster: same at NT=16; target ~11.8 from 12.64, gate keeps master if the per-token FMA cost eats it
- `layer=qkv_gate load=decode_1 m=1` flat: A2 m=1 cell unchanged

## Risk

Effect if both route: -1.3 us at 0.343 %/us (verify_1) plus ~-0.8 us at 0.27 %/us (decode_16), about +0.45-0.7% headline: below the margin, must stack (combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16). New CUDA instance with mbarrier rings: fault/hang risk, so the first measurement is an okc check. The gate decides per cell, so a slow m=16 costs nothing.

## Adversarial review

Reviewed 2026-10-25T212522Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a legitimate, unduplicated follow-up to DIAGNOSIS Action 4 and LESSONS' explicit 'untested at m>=8' note. Each cell has its own gate, so one check is cheap and can be read from the PATH CHOSEN lines and the qkv verify_1 and decode_16 rows. But the stated mechanism ('2 CTAs/SM escapes the pooler's per-token loss') is weak. An x-ingress model predicts m=8 is roughly break-even and m=16 loses, so expect the gate to keep master on m=16 and possibly on m=8.

### Justification
Profile/rows (741072dc): A2 took qkv decode_1 from 12.67 to 10.54 us, a margin of about 2.1 us. The swap-AB SPLIT_K 4 incumbent is nearly flat across m: rows 12.35 at m=8, 12.64 at m=16, 13.71 at m=32. The SoL is 7.9 us. The profile's 7.4 us bulk kernel time is below the 7.87 us floor, so profiler kernel times are L2-warm (H200 L2 > 37.7 MB). Only the cold rows and the cold gate are comparable; a profiler-derived 'kernel-only TB/s' will be inflated. m=8 is the strong half. On the existing NT=8 instance, m=8 runs exactly the same mma/ldmatrix work as m=1 (padding is already 8 tokens, NN=1). The only change is 8 KB of x per 16 KB weight chunk from L2, plus a larger epilogue. WEAKEST LINK: the argument that 2 CTAs/SM separates this from the failed pooler/unpooler row-owner. (1) The grid is 144 CTAs on 132 SMs, so only 12 SMs ever host two CTAs. The other 120 run one CTA, just like the pooler's 1 CTA/SM. (2) The pooler loss tracks x ingress, not occupancy. In the row-owner design every CTA re-reads all of x, so L2->SM x traffic = (n/R)*m*K*2, which is x/W = m/R. For qkv R=16 that ratio is 0.5 at m=8 and 1.0 at m=16. For the pooler (R=32) it was 0.25, 0.5 and 1.0 at m=8, 16 and 32, where the gap over the pure read was +1, +6 and +25 us on 67 MB. Scaled to 37.7 MB, qkv would pay about +3.4 us at m=8 and about +14 us at m=16, against a 2.1 us margin. An SM-ingress estimate using LESSONS' ~6 TB/s L2->SM agrees. m=8 needs 56.6 MB (about 9.4 us), level with the incumbent. m=16 needs 75.4 MB (about 12.6 us) and loses. The MMA issued per weight byte is also identical to the pooler at m=16 (16 mma per warp per 16 KB chunk in both). The ~11.0 target at m=8 is therefore optimistic: 11.2-11.6 is more plausible, or a gate reject. The ~11.8 target at m=16 is unlikely. The 99,328 B smem arithmetic for NT=16/S=3 is correct (2 x 99,328 < 228 KB), and 3 x 16 KB of weight in flight per CTA is enough for DRAM latency. The expected directions are consistent with TASK weights: every shape is weight 62, verify_1 ~0.343 %/us, decode_16 ~0.27 %/us. Prior check: `prior` names pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild (regression) honestly. The difference (qkv's A2 instance against a split-K-4 incumbent) is real and not on the do-not-try list: DIAGNOSIS limits that entry to pooler/unpooler at 1 CTA/SM, and LESSONS marks qkv m>=8 as untested. It is not a duplicate. The queued combo 'ombo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16' builds on it rather than duplicating it.

### Confounders on the current kernel
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (A2, master): the change reuses its compiled instance, its repack (bulk_wt_qkv_gate) and its gate machinery. This is a dependency, not a confound for the m=8/16 rows, but there is a build hazard. _bulk_prepare does state.pop('bulk_wt_' + layer) when a cell is off, and BULK_CELLS, _bulk_forward and _ab_component are keyed by layer with m == BULK_M. If the m=8 or m=16 cell is rejected after A2 m=1 passed, a naive extension deletes the shared repack and breaks or un-routes decode_1. Key cells by (layer, m), share one repack per layer, and pop it only when every cell of that layer is off. The decode_1 row must stay flat; if it moves, suspect this.
- bf16-restack-m16-swapab-router-swapab-m32 / bf16-m16-swapab-wgmma-extend lineage: this set the incumbent being displaced at qkv m=8/16 (swap-AB SPLIT_K 4 with ticket, 9.5-9.6 us warm). Any measured gain is relative to that tuning. It is the right comparator, and the gate measures it directly, so it does not confound attribution.
- bf16-m32-swapab-wgmma-triton: m=32 stays on swap-AB. No interaction, because routing is exact cfg m equality.

### Interference and A/B needs
No companion is warranted. The two target cells are separate scored shapes with exact-m routing, per-cell flags, per-cell PATH CHOSEN lines that print both medians and IQRs, and a profiler kernel-name assertion. A row move on qkv verify_1 or decode_16 is attributable from the log alone, and a gate reject reads as 'master measured'. Nothing else in kernel_fn changes. Every timed replay starts with a cold L2, so neighbouring layers cannot interact. The expected headline effect (+0.45-0.7%) is inside the ±1.5% CI, so judge the run on the two rows' paired ratios and the gate prints, not the headline. An m=8-only control would add nothing that the per-cell lines don't already show.

### Open questions
- Does the gate's printed cold net time at m=8 beat 1.03 x master in both rounds? The ingress model (W 37.7 MB + x 18.9 MB from L2 at ~6 TB/s ≈ 9.4 us) says it is marginal. Print W+x ingress bytes and the implied L2->SM rate per cell from the COLD gate median, not the warm profiler time, so the result can confirm or refute the x-ingress explanation.
- If m=16 is gated off as predicted, should the m=16 instance be dropped to cut build and hang risk? A structural fix would need x multicast across a 2-4 CTA cluster (cp.async.bulk .multicast::cluster), which cuts x ingress by the cluster size, or a larger R. That is a design change, not a widen.
- With 144 CTAs on 132 SMs, does occupancy=2 actually matter? A per-SM timing or a grid-132 variant would show whether the 12 doubled-up SMs set the tail at m=8.
- Confirm that _ab_selfcheck's AB_CASES gains qkv m=8 and m=16 entries, and that _ab_component maps them to bulk_qkv8_ok and bulk_qkv16_ok, not to bulk_qkv_ok (the m=1 flag). A self-check miss at m=8 must not switch off A2 at m=1.


## Implementation notes

Edited only `.worktrees/muse_bf16_proj/qkv-rowowner-a2-widen-m8-m16/muse_bf16_proj/submission.py` (base blob ad8d7254 verified equal to the worktree HEAD; +255/-15 lines).

What changed:
- CUDA: added `BulkQkv16 = BulkInst<16, 8192, 16, 3, 2>` as instance id 3 in `bulk_run` / `bulk_setup` (SMEM 99,328 B, grid 144, MINB 2). The kernel template is unchanged, so every bounded mbarrier wait (BULK_WAIT_NS) and the err word are inherited. The m == 8 cell reuses the SAME compiled NT=8 instance (id 1). `BULK_EXT_NAME` bumped to `muse_bf16_bulk_rowown_qkvwide_v1`; `-std=c++20`, `-lineinfo` and the 9.0a arch handling are kept.
- Cells keyed by (layer, m): new `BULK_WIDE_CELLS` = {("qkv_gate", 8): flag bulk_qkv8_ok, ("qkv_gate", 16): flag bulk_qkv16_ok}. `BULK_CELLS` (m == 1) is untouched. Both flags are appended to `NEW_CELL_FLAGS`, so they reach state B and the FINAL FLAGS line. `AB_CASES` gained qkv m=8 and m=16, and `_ab_component` maps (qkv_gate, 8/16) to their OWN flags, so a miss never switches off `bulk_qkv_ok` (m == 1).
- `_bulk_route_ok` and `_bulk_forward` are generalised to exact equality on the cell's own m (cfg m, x.shape[0], n, k, flag). Nothing else in `kernel_fn` changed, and the new route sits at the existing bulk slot, ahead of swapab16.
- New `_bulk_wide_prepare` (run right after `_bulk_prepare`; also run for state B with inherited flags). Per cell: deadline check, `bulk_setup` attributes and occupancy (cell disabled unless the query reports 2 CTAs/SM, both cells), shared repack, master's `_cell_check` canary (flushed LAUNCH/DONE, synced, error word, rel-L2 and probe row), the new two-round gate, the profiler kernel-name assertion (exactly one `bulk_kernel`), then PATH CHOSEN.
- New `_cell_gate_rounds`: master's `_cell_gate` measurement (kernel_fn with the flag on vs off on shallow state copies, 256 MB flush before every replay, 41 interleaved reps, empty-graph baseline subtracted) run for 2 independent rounds. Each round re-times its baseline and starts on the opposite order. Routes only if BOTH rounds give master/cand >= 1.03 AND cand IQR <= 1.5x master IQR. The master graph is kernel_fn with the flag off, i.e. the actually routed swapab16 path.

Review fixes applied:
1. Shared-repack hazard: the repack `bulk_wt_qkv_gate` is shared by the m=1, 8 and 16 cells. The wide phase creates it if the m=1 cell popped it. It is popped after the wide loop only if no qkv flag is on, so an m=8/16 reject cannot un-route decode_1. The existing m=1 pop is unchanged and runs before any wide cell exists.
2. Per-cell own flags, `_ab_component` and AB_CASES as in the open-question list (A/B miss at m=8 cannot disable A2 at m=1).
3. Print W + x ingress bytes and the implied rate per cell from the COLD gate net median (not the warm profiler time), for each round, next to BULK_REF_TBS and master's weight-only rate (`[cell] ingress ...` lines).
4. Review wins over the spec: expect m=8 near break-even and m=16 to lose. Both are gated, and the gate keeps master.

Open questions answered:
- Keep the m=16 instance? Yes (the spec asks for it; a rejected cell costs one canary and one gate, no routing). Its build adds one template instantiation to the existing module; a build failure still returns `_BULK_EXT = None` and master paths for ALL bulk cells (same blast radius as before the change).
- Grid-132 variant / per-SM timing: not built (a design change, not a widen).

Deviations: (a) master's IQR in the 1.5x test is floored at 0.05 us (BULK_WIDE_IQR_FLOOR_US) so a timer-resolution zero cannot trivially reject. (b) The compact end-of-prepare summary (`[summary] ...`, plus a final flags line) repeats the PATH CHOSEN lines of the bulk m=1, wide and Lt B cells, via `state["path_log"]` (host strings only); the other cells' (ouns / router cluster) lines are not repeated. (c) The wide phase start is guarded by `BULK_WIDE_DEADLINE_S` = 170 s of `_prepare_state` age and prints `skipped: deadline`. (d) The profiler name check also prints a note if the template arguments do not appear verbatim in the kernel name; it does not reject on that.

Biggest risk: the NT=16 instance (NN=2 ldmatrix.x4 B-fragment path and 33 KB slots) has never run on a GPU, and nothing here could be executed (no GPU). A fault in its canary would poison the context; the first measurement is an okc check, and every launch is announced with a flushed print. If the NT=16 template fails to compile, all bulk cells (including A/A2/A3 at m=1) fall back to master.

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn; the new route is a cfg/shape/flag check plus `bulk_run`, no allocation or control flow on input values (diff checked; the `.item()`/sync calls are all in prepare-only code).
- [x] Nothing derived from `data` is cached across calls (state holds weights-only repack, flags, a host float and host strings).
- [x] Weights: the qkv repack is the existing lossless permutation, built in `prepare()` only, round trip asserted bit-exact; BF16 stays BF16.
- [x] Activations BF16 where SGLang keeps BF16; the new kernel's output is bf16, fp32 accumulation.
- [x] n/a: attention-only item; this is a dense projection.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (load_inline is the existing in-process build; no child process code added).
- [x] All work happens inside the timed call (one bulk_kernel launch for out); no output caching or replay.
- [x] Never raises: build error, canary/check miss, occupancy < 2, gate miss, deadline or profiler mismatch each land on master's exact path with only that cell's flag off.
- [x] Bounded waits on every mbarrier (kernel unchanged, BULK_WAIT_NS), distinct symbol for the new instance, build time and regs/spills printed per cell.

okc_lint: `OK with warnings` (only the usual prepare-only host-sync / dtype warnings; no errors). `python3 -I -m py_compile`: pass. `__pycache__` removed from the worktree.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent. **Verdict: dispatch-with-fixes.**

Checks done:
- **NT=16 ldmatrix path (never run on a GPU).** I emulated bulk_kernel's fragment indexing in numpy (`scratchpad/audits26/emu/bulk_emu.py`). The emulator models byte-addressed smem ring slots, the swizzled `_bulk_repack` layout, ldmatrix.x2/x4 lane-to-row semantics, the mma.m16n8k16 row.col A/B/C fragments, the scratch epilogue and the 4-warp sum. It asserts 16 B alignment and slot bounds on every smem address. The data are integers, so the fp32 sums are exact, and pad tokens hold NaN to prove they never leak. Every case was an EXACT MATCH: <16,8192,8,4> at m=1 and 8, and <16,8192,16,3> at m=16, 11 and 1. A mutation test (swapping the token-block and k-half roles of `mi`) gives MISMATCH, so the emulator does catch indexing errors. SMEM is 99,072 B (NT=8) and 99,328 B (NT=16). Every x and W ldmatrix row stays inside its slot.
- **Gates.** No gate always fails. The 1.03 two-round margin and the IQR <= 1.5x max(master IQR, 0.05 us) test can both pass. The 170 s deadline counts from the start of `_prepare_state` (master's prepare takes about 4.5 s). Routing is exact equality on cfg m, using only cfg, shape and flag.
- **Fallback.** The wide cells run after the m=1 cells are decided. Each has its own flag, also in `_ab_component`. The shared repack is popped only when no qkv flag is on. A failure in any cell lands on master and prints a PATH CHOSEN line.
- **Rules.** In kernel_fn only `_bulk_route_ok` and `_bulk_forward` change. They use cfg, shape and flag only, and allocate the same way as master's m=1 cell. Nothing derived from data is cached. The weights keep master's lossless repack. okc_lint: OK with warnings (prepare-only). py_compile passes.

Fixes:
1. **An NT=16 compile failure took down the m=1 bulk cells (category 2).** All instances are in one load_inline module. A build error in the new instance therefore returned `_BULK_EXT = None`, which sent A/A2/A3 at m=1 to cuBLAS (A2 is master's 12.67 -> 10.54 us win) along with the m=8 cell. That turns a neutral run into a decode_1 regression. The fix wraps the NT=16 `using`, launch and setup lines in `#ifndef BULK_NO_NT16`. `_bulk_build` now retries once, under its own name `<BULK_EXT_NAME>_nont16`, with `-DBULK_NO_NT16`. That rebuild has exactly master's instance set, so only A2w16 falls back: `bulk_setup(3)` raises "unknown instance", the error is caught, and the cell prints PATH CHOSEN master. The NN>=2 branch is a discarded `if constexpr` inside a template, so it is never instantiated without NT=16.

Residual risk: the NT=16 indexing emulates exactly, so what remains is a device fault, or a gate reject on timing (the ingress model predicts m=16 loses and m=8 is break-even). A double build failure unrelated to NT=16 would add about 40 s of nvcc at import.
