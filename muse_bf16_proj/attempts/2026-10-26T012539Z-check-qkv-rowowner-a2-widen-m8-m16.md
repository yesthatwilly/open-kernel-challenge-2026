# muse_bf16_proj · 2026-10-26T012539Z · check · qkv-rowowner-a2-widen-m8-m16

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T211857Z-qkv-rowowner-a2-widen-m8-m16.md` (branch `hyp/muse_bf16_proj/qkv-rowowner-a2-widen-m8-m16`).

- **Job:** `0124ea8fcaf14b3c` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `e22908f0`

## Goal
DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; decode_1 12.67 -> 10.54 us). NT=8 is the padded token count and bulk_run already accepts 1 <= m <= NT, so (1) m == 8: route qkv_gate at cfg m == 8 onto the SAME compiled A2 instance (verify_1 qkv now runs _bf16_swapab_kernel SPLIT_K 4, 9.6 us kernel, row 12.35). (2) m == 16: add one instance bulk_kernel<16, 8192, 16, 3, 2> (NT=16 doubles the x slot: SLOT = 16384 + 16*1040 = 33,024 B, S=3 gives 99,328 B, still 2 CTAs/SM; S=4 would not fit two CTAs) for decode_16 (swapab16, 9.5 us kernel, row 12.64). Bump BULK_EXT_NAME to a new name since the instance set changes; keep -std=c++20 in the nvcc flags and the bounded mbarrier waits (BULK_WAIT_NS). In BULK_CELLS-style config add two cells with their own flags bulk_qkv8_ok and bulk_qkv16_ok (appended to NEW_CELL_FLAGS, inherited by state B, AB_CASES entries for qkv m=8 and m=16); generalise _bulk_route_ok from cfg m == BULK_M to the cell's own m (exact equality, cfg/shape/flag only). Each cell goes through master's existing canary (flushed LAUNCH/DONE, synced, error word), the rel-L2/probe-row correctness check, the occupancy query (disable the m=16 cell unless it reports 2 CTAs/SM), then the gate. Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. Print per cell the kernel-only TB/s next to BULK_REF_TBS. The m=1 cells, all other layers and m=32 stay exactly as master.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
Effect if both route: -1.3 us at 0.343 %/us (verify_1) plus ~-0.8 us at 0.27 %/us (decode_16), about +0.45-0.7% headline: below the margin, must stack (combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16). New CUDA instance with mbarrier rings: fault/hang risk, so the first measurement is an okc check. The gate decides per cell, so a slow m=16 costs nothing.

## Result (after the run)
- job `0124ea8fcaf14b3c` (check), status done, passed True, tests 9/9
- speedup 1.2603x, 95% CI [1.2414, 1.2792], score 3833.178433399135 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/qkv-rowowner-a2-widen-m8-m16.result.json`, `muse_bf16_proj/hypotheses/done/neutral/qkv-rowowner-a2-widen-m8-m16.patch`; branch `hyp/muse_bf16_proj/qkv-rowowner-a2-widen-m8-m16`
