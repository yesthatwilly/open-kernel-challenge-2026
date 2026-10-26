# muse_bf16_proj · 2026-10-26T012539Z · check · pooler-k-halves-fp32-partials-restack-m128-m256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T211856Z-pooler-k-halves-fp32-partials-restack-m128-m256.md` (branch `hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256`).

- **Job:** `de95f81782374124` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `659ff478`

## Goal
DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256 (done/neutral/pooler-k-halves-batched-gemm-m65-m256.patch: _khalf_forward, _khalf_kernel_names, _khalf_names_ok, _khalf_select, flag pooler_khalf_ok + dict state['pooler_khalf']) for layer == 'pooler' at cfg m in {128, 256} only. Today master runs pooler m=128 as nvjet 128x64 splitK NNT 20.8 + cublasLt splitKreduce 3.0 us (row 22.85) and m=256 as nvjet 128x64 NNT 28.8 us (row 27.98 in 741072dc, 29.9-30.05 in today's two other runs). The khalf path: xh = x.view(m, 2, 4096).permute(1, 0, 2) (strided, no copy), wh = state['pooler_t'].view(2, 4096, 4096) (master's existing lossless NN repack, no new weight bytes), p = torch.bmm(xh, wh) -> [2, m, 4096], out = p[0] + p[1]. Offer the selector two candidates per m: (a) bf16 partials exactly as ran (ce808e5a: nvjet 128x128 NNT 24.7 + add 2.1 at m=256; 20.1 + 1.5 at m=128); (b) fp32 partials: p = torch.bmm(xh, wh, out_dtype=torch.float32), then out = torch.empty(m, 4096, bf16); torch.add(p[0], p[1], out=out) (one elementwise kernel, fp32 add, one bf16 rounding, the same numerics as cuBLAS's own split-K + splitKreduce). If torch.bmm rejects out_dtype on this image, print it and offer (a) only. Prefer (b) on a tie within 1%; route (a) only if it beats (b) by >= 1% (bf16 partials passed the tests already, rel L2 2.45e-3 vs 1.66e-3 master, well under the 1% bound). Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. Wire it the way the source branch did: own flag in NEW_CELL_FLAGS, inherited by state B via _ab_flags, an AB_CASES entry for pooler m=128 and m=256, the kernel_fn branch placed before master's NN torch.mm branch, cfg/shape/flag-only gate, no print/try/sync, fresh output. Everything else in master is untouched, m=512/1000 tests stay on master.

## What changed, and where
Edited only `<worktree>/muse_bf16_proj/submission.py` (base blob ad8d7254, +279/-15).

What changed:
- New constants `KHALF_*` and per-m flags `pooler_khalf128_ok` / `pooler_khalf256_ok` (appended to `NEW_CELL_FLAGS`, so `_ab_flags` hands them to state B); per-state dict `state['pooler_khalf_kind']` ({m: 'a'|'b'}), also inherited by state B.
- `_khalf_forward`: xh = x.view(m,2,4096).permute(1,0,2) (strided), wh = pooler_t.view(2,4096,4096) (master's existing NN repack, no new bytes). Kind a: bf16 bmm + add. Kind b: `torch.bmm(..., out_dtype=torch.float32)`, then `torch.add(p[0], p[1], out=<fresh bf16>)`.
- `_khalf_route_ok` (cfg/shape/flag only) and a kernel_fn branch placed immediately before master's NN torch.mm branch, under `layer != "router"`; no print/try/sync.
- `_khalf_prepare` (state A only, after `_lt_prepare`, own start deadline 170 s, prints `skipped: deadline`) and `_khalf_cell` per m. Per candidate: `_cell_check` through kernel_fn (rel L2 vs fp64, power-of-two probe row, bit-identical rerun), then `_cell_kernels` profiler assertion (needs a GEMM kernel; rejects copy/memcpy and splitKreduce; elementwise names are only printed, so a mixed-dtype add cannot falsely reject b). Then 2 independent interleaved rounds (41 reps, 256 MB flush, own empty-graph baseline each) of master's ACTUALLY ROUTED path (kernel_fn with the flag off, so nn_ok / nn_p256_ok already True) against each valid candidate. Route only if EVERY round has master/cand >= 1.03 AND cand IQR <= 1.5x master IQR. Prefer b; route a only if it passes and its min round ratio >= 1.01x b's (or b fails). Prints xh.stride(), master kernels, both candidates' kernels (even if unrouted or excluded), every round's medians/IQRs, and one PATH CHOSEN line per m that says "gate rejected, not a measured loss of the row" vs a candidate.
- `_ab_component(layer, m, state)` now returns the K-halves flag FIRST for a routed pooler 128/256 cell, so a miss turns off only that flag (master NN path serves it), never `nn_ok` (shared with unpooler m=128). No duplicate AB_CASES entries.
- `_path_print` helper: PATH CHOSEN / gate lines of the existing cells (ouns, router cluster, bulk A/A2/A3, Lt B) and the new khalf cells now also go to `_PATH_LINES` (host strings), and prepare() repeats them as `[summary]` lines AFTER FINAL FLAGS, as the last lines of prepare (long lines cut at 260 chars, last 40 kept). This is a print-only change to the old cells.

Review fixes applied: (1) gate baseline is master's routed flag-off path with nn flags True (khalf runs inside `_prepare_state`, before the A/B check); (2) `_ab_component` remap, no duplicate AB_CASES; (3) hardened two-round 1.03 + IQR 1.5x gate (extended gate returns IQRs, written as `_khalf_cell` since `_cell_gate` returns medians only); (4) FINAL FLAGS prints nn_ok, nn_p256_ok and both khalf flags plus kinds; (5) profiler assertion does not match elementwise names. Open questions answered in code: out_dtype and xh strides are probed and printed (b excluded if bmm rejects it; copy kernel in profile excludes a candidate); b's GEMM kernel name is printed; m=256 keeps the 1.03 two-round bar, but the PATH CHOSEN line separates 'gate rejected' from a lost row, and both rounds are printed so the orchestrator can decide; m=128 and m=256 are gated and flagged independently, and the combo parent inherits whatever each cell routed (flags are per cell, per state).

Deviations: none from the review. Compile/run risks are listed in the report (bmm out_dtype availability, bf16 candidate a rel L2 2.45e-3 against the 3e-3 CELL_REL_L2 canary bound).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn branch is torch.bmm/torch.add/torch.empty on shapes and flags only; no .item()/.cpu()/sync/print/try (all syncs, prints and try blocks are in prepare()).
- [x] Nothing derived from `data` is cached: only the kind choice from synthetic prepare-time x, stored per state; kernel_fn reads x and cfg only.
- [x] Weights: no new weight bytes; reuses master's lossless bf16 NN repack pooler_t (view only). fp32 only in internal partials.
- [x] Activations BF16 where SGLang keeps BF16: x and the output are bf16; the output of both kinds is bf16 (b: one rounding from an fp32 add).
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only torch ops, torch.profiler in prepare and prints.
- [x] All work happens inside the timed call; no output caching or replay (fresh output tensor per call).
- [x] Review required fixes applied (see Implementation notes).

okc_lint: `OK with warnings: muse_bf16_proj/submission.py` (only pre-existing-style host-sync/dtype warnings in prepare(); no errors). `python3 -I -m py_compile` passed; __pycache__ deleted.

## Risks / what could fail
low
Effect: -1.2 to -3 us at 0.184 %/us on verify_32 and -1.4 us at 0.230 %/us on verify_16, about +0.5-0.9% headline: below the ~1.24% one-run margin, so it is expected to land neutral alone and pays inside combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16. Low fault risk (cuBLAS only). Rules: bf16 weights unchanged (master's lossless NN repack), fp32 only in internal partials.

## Result (after the run)
- job `de95f81782374124` (check), status done, passed True, tests 9/9
- speedup 1.2683x, 95% CI [1.2492, 1.2873], score 3808.9777289905714 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.patch`; branch `hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256`
