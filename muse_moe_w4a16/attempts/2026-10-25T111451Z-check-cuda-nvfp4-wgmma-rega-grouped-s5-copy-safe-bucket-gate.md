# muse_moe_w4a16 · 2026-10-25T111451Z · check · cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T054545Z-cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.md` (branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate`).

- **Job:** `7be125e9e1ea42e3` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `fc959d9c`

## Goal
The shippable twin of cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: identical code (same s4 base rebased onto master 9080a71, same per-call cuTensorMapEncodeTiled from the bsrc pointer inside wg_gemm, same removal of wg_maps/wg_mptr/wg_mptr_py and _wg_state_live, same deepcopy + tensor-clone three-state refill-replay self-check, same profiler LAUNCHED assertion), plus a per-T-bucket selector so the path can only ship where it wins. Buckets: B128 = 128<=T<=255 (verify_16, skewed-128 test) and B256 = 256<=T<=512 (verify_32, bs=257 test). For each bucket, prepare times master's full Marlin path against the full wgmma path (gather + w13+silu + w2 + the shared align/combine) at T=128 and T=256 with the LESSONS-173 selector: both captured as CUDA graphs, a >=256 MB L2 flush before every replay, 41 interleaved A/B reps, an empty-graph baseline subtracted, under the judge-like skewed routing calibrated so prepare's Marlin pair is ~0.8x the judged rows (~245 us at T=128, ~370 us at T=256; print local route counts) plus one random routing. A bucket is enabled only if wgmma's median wins by >= 1.02x on BOTH routings; print both medians and the ratio, and assert the enabled side is the faster printed one. kernel_fn branches on T against two prepare-time booleans state['wg_on_128'] and state['wg_on_256'] (shape + flag only). Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## What changed, and where
Base: the worktree was on s4 (928151c, stale). master's muse_moe_w4a16/submission.py is unchanged since 9080a71 (git diff 9080a71 master empty), which is the base of s5-forced, so no 3-way merge was needed. submission.py = the s5-forced submission verbatim (from branch hyp/.../s5-copy-safe-forced, same job-proven code: per-call cuTensorMapEncodeTiled, no wg_maps/_wg_state_live, three-state copy check, device launch marker, LAUNCHED line), plus only these changes (221 changed lines vs forced):
- Removed the forced "no timing gate" and print-only `_wg_time_vs_marlin`. Added `_wg_calibrate` (alpha pick so Marlin pair ~0.8x judged: 245 us T=128, 370 us T=256; same code as before), `_wg_graph_of`, `_wg_ab_graphs` (3 untimed warm-up replays, 41 interleaved reps with alternating order, 256 MB flush.zero_() before EVERY replay, events per replay, medians) and `_wg_select`.
- `_wg_select` runs after the copy check and only if the path is otherwise ok. Per bucket (T=128 -> B128, T=256 -> B256) it captures TWO full-path graphs on static inputs: `_moe_marlin` (master's align + block_size_m + w13 + silu + w2 + norm, w2_mode master) and `_moe_wgmma` (align + gather + w13+silu + w2 + norm), plus an empty-ish baseline graph (dummy add_); baseline median is subtracted from both. Routings: the calibrated Zipf routing and one random routing; local route counts, Marlin bsm, both medians, ratio and the calibrated Marlin pair vs judged 306.2/483.2 us are printed per routing and per bucket. Enabled only if Marlin/wgmma >= 1.02 on BOTH routings; also required: wg_err word 0, launch marker == WG_MAGIC, finite output and end-to-end rel-L2 vs Marlin <= 2e-2 on the same graphs. Asserts that an enabled bucket has wgmma faster on every printed routing. Any exception -> both buckets Marlin.
- state['wg_on_128'], state['wg_on_256'] (False by default, set at the end of `_wg_setup` from the selector). kernel_fn branches on shape and these flags only: 128<=T<=255 -> wg_on_128, 256<=T<=512 -> wg_on_256, else Marlin. `wgmma_ok` remains as the "path is correct" capability flag but kernel_fn no longer reads it. The copy check sets both flags True on its three states so it exercises wgmma at M=128, 256 and 257.
- Logging: `PATH CHOSEN B128 / B256 = WGMMA|MARLIN` lines and a combined PATH CHOSEN line are printed last; the diagnostic summary contains the selector lines; LAUNCHED = wgmma line retained from the copy check. Per-phase times (copy check, selector, setup) are printed. Review open question: id(state['wg_ext']) and wg_xs data_ptr are printed in prepare.
- Review items applied: (1) selector times each side as its full graph, align included, and calls master's own `_moe_marlin` so master's block_size_m is used (review "Confounders"); (2) Marlin pair printed beside judged 306.2/483.2 us; (3) per-bucket decision printed late in the log; (4) per-phase prepare timings printed. Not applied: the review's advice to dispatch only after the forced twin is harvested is a sequencing decision for the caller. Spec deviation: none besides using 256 MB flush and 3 warm-up replays.
- Expected from s5-forced judge data: B128 Marlin (wgmma 353 vs Marlin 310 us) and B256 wgmma (402 vs 457 us).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only a Python branch on shape + two prepare-time booleans; no sync/.item/allocation on values (checked in diff: selector code is prepare-only, called from `_wg_setup`).
- [x] Nothing derived from `data` cached across calls: flags come from synthetic prepare-time routings only.
- [x] Weights: unchanged from s5-forced (lossless retile in prepare, NVFP4 stays NVFP4, dequant in registers).
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] Attention: n/a (MoE problem); every route attended (no skipping), unchanged.
- [x] No timing patches, harness/IO/secrets, module edits, reference import, subprocesses or network: no subprocess/fork/spawn added; selector only uses torch events/graphs.
- [x] All work inside the timed call; no output caching or replay. No module/static globals hold device pointers (selector tensors are locals; `_wg_audit_globals` unchanged and still run; flags are in state).
- [x] Two-state judge copies: flags are plain bools in state and survive a tree-map clone; copy check (A, deepcopy B, clone C) unchanged.

okc_lint: `OK with warnings` (0 errors; only host-sync/dtype warnings outside kernel_fn's call graph, prepare-only). undef.py: `submission.py []` (no undefined names). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Dispatch in parallel with cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced (same round): the forced twin gives the per-shape kernel rows whatever the selector decides, this one is the land candidate. If both buckets come out enabled, the two runs are a replication of the same judged path, which is still useful given +-20% Marlin profile noise. Risk: the prepare routing still mis-forecasts the judge (LESSONS bf16 cuBLASLt pick); the 1.02x margin and two routings limit that.

## Result (after the run)
- job `7be125e9e1ea42e3` (check), status done, passed True, tests 5/5
- speedup 1.3744x, 95% CI [1.3538, 1.3950], score 9785.216999650855 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate`
