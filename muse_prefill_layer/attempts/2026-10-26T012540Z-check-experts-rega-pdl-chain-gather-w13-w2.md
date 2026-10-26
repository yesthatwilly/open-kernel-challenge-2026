# muse_prefill_layer · 2026-10-26T012540Z · check · experts-rega-pdl-chain-gather-w13-w2

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T211833Z-experts-rega-pdl-chain-gather-w13-w2.md` (branch `hyp/muse_prefill_layer/experts-rega-pdl-chain-gather-w13-w2`).

- **Job:** `8e4685f303c143a8` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `6dc70581`

## Goal
DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no programmatic dependent launch (griddepcontrol appears 0 times in submission.py). Port muse_moe_w4a16's helpers (submission.py ~line 1029: pdl_wait() = griddepcontrol.wait, pdl_launch() = griddepcontrol.launch_dependents; host side cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization = 1, ~line 1524) into the rega module. rega_gemm gains an int pdl argument; when 1 it launches with the attribute (graph capture keeps the programmatic edge, as moe's PDL chain does on the judge). In rega_kernel: everything that reads only align outputs (sorted_ids, block_experts, n_padded, chunk tables in smem: produced two or more grids back, complete before the previous grid launched) and weights may run BEFORE pdl_wait: barrier init, tensormap prefetch, the table decode, and the producer's first STAGES weight bulk copies (bulk_g2s of wsrc for the first items; weight addresses come from decode_item over align tables). pdl_wait() is called by the producer before its FIRST activation tma_2d and by every consumer warp before its first epilogue store, so no read of xs/hs and no write of out happens before the previous grid's memory is visible. pdl_launch() is issued by each CTA's producer after it has issued its last item's last stage. Apply pdl=1 to w13 (primary: _norm_gather_kernel or ext.rega_gather, launched normally; implicit trigger at its exit) and to w2 (primary: w13, explicit trigger). Do not touch fused_post_expert_norm_weighted_sum (SGLang Triton, launched normally after w2). All three rega classes. GATE per class in _rega_setup: exact-compare pdl=1 vs pdl=0 (bit-identical expected), two-state replay check, cold-L2 interleaved graph timing of the whole _experts_rega call; ON at <= 0.99x (a small, low-risk win); state['rega_pdl'][cls] plain int. Print per class both medians and PATH CHOSEN.

## What changed, and where
All changes are in `muse_prefill_layer/submission.py` of the worktree (+496 lines, no deletions).

- **Separate module, master untouched.** `_REGA_HEAD`, `_REGA_BODY`, `_REGA_CPP`, the v2/v1 sha256 checks and the low-class module are unchanged. The PDL CUDA source is derived at prepare time from master's strings by exact-once text edits (`_REGA_PDL_EDITS`, `_rega_pdl_src`; any anchor miss gives None and no PDL build). It builds as `muse_prefill_rega_pdl` with distinct symbols: `rega_kernel_pdl<G,BIG>`, `rega_gemm_pdl(..., int64 pdl)`, `rega_init_pdl`, `rega_gather_pdl[_kernel]`. Offline check: master's hash still equals REGA_V2_SHA256, the derived source passes `_rega_audit_cpp`.
- **Kernel.** `pdl_wait()` / `pdl_launch()` (griddepcontrol, sm_90). With `pdl=1` the producer first issues, for the first STAGES stages, `expect_tx` (armed once, whole-stage bytes) plus the weight `bulk_g2s`, then `pdl_wait()`, then the main loop (stages n < STAGES skip their expect_tx and weight copy, issue only the two activation TMAs, so the barrier is neither double-armed nor under-counted). Every consumer thread calls `pdl_wait()` right after `setmaxnreg.inc`, before any smem read of activations and before any store to `out`. The producer calls `pdl_launch()` after its last item's last stage. Barrier init, table decode and the align-table reads stay before the wait (align precedes the normally launched gather). `pdl=0` skips every PDL instruction.
- **Host.** `rega_gemm_pdl` launches via `cudaLaunchKernelEx`; `ProgrammaticStreamSerialization=1` only when `pdl=1`, otherwise a plain launch. A launch error raises inside prepare only (caught by the phase).
- **Phase.** `_rega_pdl_phase` runs after `_rega_lo_phase` (it needs the low variant int). Its own deadlines: start 195 s ('skipped: deadline' printed), per class 255/255/235 s, replay check before REGA_COPY_DEADLINE_S. It builds the module (prints arch flags, build time), prints regs/local bytes per instantiation (`_rega_pdl_attrs`), and per routed class (`rega_ok`) runs `_rega_pdl_class`: traced eager canary; graphs of the whole `_experts_rega` call for master's ACTUALLY ROUTED path (flag off, incl. the norm-gather path and the low variant) vs PDL over the same static inputs; cold-L2 (flush before each replay) bit-for-bit compare over several routings x 3 PDL replays plus error word / launch markers; two interleaved cold-L2 rounds x 11 alternations with empty-graph baseline; the two-state (A/B/C) replay check with the flag on (`REGA_PDL_STATE_KEYS`). Both gate lines and every `PATH CHOSEN` line are printed again as the last lines of prepare().
- **Flags.** `state['rega_pdl'][cls]` plain ints, `state['rega_pdl_ext']` the module (None when no class is ON; keys are read with `.get`, so master's mini states and the older phases are unaffected). `_experts_rega` uses `_rega_pdl_flags(state, T, variant)`.

### Review fixes applied
1. Gate is 'not slower' (REGA_PDL_RATIO = 1.005, inside the user's 1.00-1.01 band) in BOTH interleaved rounds with candidate IQR <= 1.5x master's, instead of 0.99 (user instruction, matches the review's open question 1).
2. w2 is forced to pdl=0 whenever `state['zero_nonlocal_w2']` is True (memset node between w13 and w2); w13 may still chain to the gather.
3. Both launches are pdl=0 on the `rega_gemm_lo` path (variant != 0), and the low class is not PDL-gated when the variant is not master's.
4. Open question 2 (expect_tx): armed once per stage before the weight copy, activation TMAs complete the same transaction; stages n >= STAGES unchanged.
5. Open question 3 (Triton primary): not assumable from the moe precedent; the exact compare and the timing run on CUDA-graph captures of the real prenorm path, so a missing programmatic edge shows up as a flat gate (no correctness effect: pdl_wait is a no-op when there is no programmatic edge). Cannot be verified without a GPU here.
6. Open question 4 (mc2 stacking): the PDL kernel/launch edits are isolated as `_REGA_PDL_EDITS` text edits over master's strings; a later cluster kernel needs the same three insertions (prefetch pass, consumer wait, trigger) plus the programmatic attribute next to its cluster-dim attribute.

### Deviations
- No `prefetch.tensormap` (master's rega kernel has none; the maps are __grid_constant__).
- The consumers' wait is at the start of the consumer branch instead of just before the first epilogue store: they cannot do useful work before the first full barrier anyway, and it is the simpler, stricter placement.
- The mc2 coupling, review reading rules (do not read per-kernel rows: rega_kernel_pdl durations grow with PDL) are for the judge-side reading, nothing to build.
- Nothing could be compiled or run here (no nvcc/GPU/torch). Review the new CUDA as untested.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-pdl-chain-gather-w13-w2 diff` (one file, +496 / -0).

- [x] CUDA-graph capturable: kernel_fn's added code is `_rega_pdl_flags` (shape + prepare-time ints) and one `rega_gemm_pdl` launch via `cudaLaunchKernelEx` on the current stream; no `.item()`/`.cpu()`/`synchronize()`/tensor prints, no allocation or control flow on input values. All syncs and prints are in the prepare-only phase. Every wait is bounded (existing `mbar_wait`; `griddepcontrol.wait` returns when the prerequisite grid completes).
- [x] Nothing derived from `data` is cached across calls: the new state is the module (`rega_pdl_ext`) and per-class ints picked in prepare from synthetic cases; tensor maps are still encoded per call in C++.
- [x] Weights: untouched (same retiled bank, NVFP4 stays NVFP4).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries FP8: no numeric change (output bit-identical by construction and compared bit for bit in prepare).
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the new build is `load_inline` in-process like master's; no subprocess/fork.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Standing build rules: two interleaved rounds (REGA_PDL_ROUNDS), master's actually routed path timed flag-off, medians/IQRs/kernel names/PATH CHOSEN printed and repeated as the last lines of prepare; per-state flags (`rega_pdl`), a miss falls back to master's exact path and touches no other flag; build time and regs/spills printed; new symbol names; own start deadlines with 'skipped: deadline'; no module-global device state.

okc_lint: `OK with warnings` (0 errors; only the usual host-sync/dtype warnings outside kernel_fn's graph). `python3 -I -m py_compile` ok, __pycache__ deleted.

## Risks / what could fail
medium
Prior art outside prefill: muse_moe_w4a16 combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 ran a PDL w13->w2 chain on the judge and front-end/epilogue wins did not interfere (LESSONS 183); attention combo-b1-chain-c320-fused-pre-pdl showed launch-gap wins compound with kernel wins. A read of xs/hs or a write of out before griddepcontrol.wait is a silent race that random routings may not expose: the exact-compare must run under graph replay with pdl=1. First measurement is an okc check.

## Result (after the run)
- job `8e4685f303c143a8` (check), status done, passed True, tests 5/5
- speedup 1.4622x, 95% CI [1.4403, 1.4841], score 264947.92479134316 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4677493909229, "ci": [1.4457331500590564, 1.4897656317867434], "job": "73b40cd13fdf4bb0"}; base result: muse_prefill_layer/hypotheses/done/neutral/combo-b-normgather-two-cta-low-regate.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-pdl-chain-gather-w13-w2.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch`; branch `hyp/muse_prefill_layer/experts-rega-pdl-chain-gather-w13-w2`
