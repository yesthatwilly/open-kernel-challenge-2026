# muse_moe_w4a16 · 2026-10-25T051658Z · check · cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T044131Z-cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.md` (branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe`).

- **Job:** `4e7f8be895a145ba` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `928151c`

## Goal
Fix the s3 judge failure (job 3bdec0e58d684c17: verify 16 bs=128 "recheck failed on repetition 2: relative L2 error 1.211e+00"). CAUSE (s3 branch submission.py): the 8 TMA maps live in ONE process-global `Maps g_maps[2]` inside the shared load_inline module (:972). wg_set_maps encodes them over the CALLING state's wg_xs/wg_hs (:1026-1045; called from prepare at :2122-2124), and every launch passes g_maps[which] by value (:1087), which the graph then freezes. The judge prepares several copies (setup record `copies: 2`) and rotates them across repetitions. The last prepare() re-points g_maps at its own scratch, so on every other copy: the gather writes that copy's state["wg_xs"] (:1364) and w13 writes its state["wg_hs"] (:1368), but the w13 TMA reads the OTHER copy's xs and the w2 TMA reads the OTHER copy's hs. Those hold rows from whatever the other copy last ran. Reps 0/1 pass (the other copy's rows still match those inputs); rep 2 is the first replay where the other copy has run different inputs. It is not a rules violation: n_padded, chunks and buckets are read on the device (:836-889), and kernel_fn branches on shape only (:2364). The s3 self-check could not see this. It used one state, captured per case and replayed the SAME inputs 3 times (:1897-1900). FIX. Base: master 9080a71 (w2 atomic-add landed). T<128 and T>512 run master's Marlin unchanged. Port the s3 kernels with variant A (wgmma_wait<0>) ONLY in every bucket. Drop variant B, the vmask argument and the A/B canary timing. (1) Delete the g_maps/g_maps_ok globals. Add wg_make_maps(xs, hs), which returns a CPU uint8 tensor [2, sizeof(Maps)] stored in state["wg_maps"]. wg_gemm takes that tensor plus the B-source scratch tensor, memcpys row `which` into the __grid_constant__ Maps argument, and TORCH_CHECKs that the scratch data_ptr equals the one recorded when the maps were encoded (keep a per-blob int64 ptr pair in state). Each copy's graph then freezes maps over its OWN scratch. (2) Keep wg_xs/wg_hs/wg_err per state (no module-level scratch, no static device pointers anywhere). (3) Audit the module for any other static/global that holds a device pointer, and print the result. MANDATORY self-check before PATH CHOSEN=WGMMA. Any failure prints one `OFF: <reason>` line and master's Marlin runs. (a) Refill replay: for M in (128, 256, 257), allocate static x / topk_ids / topk_weights, capture kernel_fn ONCE, then for >=3 different routings (random, skewed 4-hot, all-one-expert, mixed local+remote) copy_ the new values into the SAME buffers, replay, and compare each output to eager Marlin on the same values (rel-L2 <= 1e-2, row bound 5x, no NaN/Inf). (b) Cross-copy: build a second state B with its own scratch and maps (call prepare() again, or the wgmma setup alone). Capture GA on state A and GB on state B, then run GA, GB, GA, GB, GA with a fresh routing refilled before each replay, and compare each to Marlin. (c) Print one line per replay: routing tag, rel-L2, and the map scratch ptr vs that state's buffer ptr. Keep the s3 cold timing lines (M=128/256 wgmma vs Marlin) unchanged. The first measurement is an okc check, never a submit.

## What changed, and where
File: `muse_moe_w4a16/submission.py` in the worktree (only file touched; first-line marker kept).

**Base correction.** The worktree was NOT on master 9080a71: its HEAD was 8261c04, the s3 impl commit, so its file was byte-identical to the s3 branch and lacked master's w2 atomic-add. I ported master's pieces into it instead: `W2_ATOMIC_MAX_TOKENS`, `_w2_atomic_self_check`, `_forward`, and the `w2_common` use_atomic_add branch (now a `w2_mode` parameter of `_moe_marlin`, default "master"). prepare() runs the w2 self-check BEFORE the wgmma setup so the 200 s budget counts it. T<128 and T>512 call `_moe_marlin(..., state["w2_mode"])`, i.e. master's launches. A diff against master shows the wgmma module plus these hooks only. The caller must commit on a base that is master or accept that the commit's parent is the s3 impl commit.

**Fix (spec items 1-3).**
- C++ globals `g_maps`, `g_maps_ok`, `g_map_rows` and `wg_set_maps` deleted. `wg_make_maps(xs, hs)` returns a CPU uint8 `[2, sizeof(Maps)]` blob (throws on any encode failure). `wg_gemm(which, maps, mptr, bsrc, ...)` memcpys row `which` into a local `Maps` passed by value (the `__grid_constant__` argument) and TORCH_CHECKs `mptr[which] == bsrc.data_ptr()`; `bsrc` is that state's xs (w13) or hs (w2), and the row bound now comes from `bsrc.size(0)`.
- `_wg_attach_scratch(state, ext, device, hidden, inter)` gives a state its own wg_xs, wg_hs, wg_err, wg_maps, wg_mptr (int64 CPU pair), wg_mptr_py, wg_ext, wg_grid. `_wg_gemm(state, ...)` is the one launch wrapper, so every call site reads only `state`.
- Host-side live guard in kernel_fn (`_wg_state_live`, python-int pointer compare, no sync): a state whose scratch pointers differ from the recorded pair runs master's Marlin instead of raising.
- Audit (`_wg_audit_globals`, printed in prepare): a comment/string-stripped scan of the whole load_inline source for file-scope variables, static locals and `__device__/__constant__/__managed__` globals, plus a scan of this module's Python globals for tensors/graphs/containers of them. Any finding turns the path off. The scanner was run offline on both sources: s3's source reports g_maps, g_maps_ok and g_map_rows; this file's reports none.
- Extension name bumped to `..._rega_v4` so a stale s3 .so can never be reused.

**Variant A only.** Variant B, the `vmask` kernel/host argument, the `V` template parameter, `_wg_vmask`, `WG_VARIANT_NAME` and the A/B canary timing are removed. `consume_item<NB, G>` is the s3 variant-A loop verbatim (wait<0> after each commit, trailing wait<0> before the epilogue). `_wg_bucket_canary` keeps the four per-bucket launches and judging, one variant, no cold-pair timing.

**Mandatory self-check (`_wg_replay_check`, before PATH CHOSEN).** (a) Refill replay: for M in (128, 256, 257) static x/topk_ids/topk_weights, ONE capture on state A, then 5 replays (random, skewed-4hot, all-one-expert, mixed-local-remote, random) refilled in place into the same buffers, each compared with eager Marlin on the same values (rel-L2 <= 1e-2, per-row <= 5e-2, finite, device error word 0). (b) Cross-copy: GA is captured on A BEFORE state B exists (this covers the review's open question on capture order), then B is built (`_wg_clone_state`: own scratch, maps, error word AND a cloned retiled bank), GB is captured on B and GA2 on A after B exists, and per M the replay order is GA, GB, GA, GB, GA, GA2, GB with a fresh routing and fresh x/weights refilled before every replay, each compared with Marlin. (c) One line per replay: tag, rel-L2, row, err, map scratch ptr vs that state's buffer ptr. A miss, exception or budget skip (WG_REPLAY_DEADLINE_S = 235) prints one `OFF: <reasons>` line and master's path runs. Every replay is announced with a flushed print before and printed after a sync. The final summary block carries the replay totals and worst rel-L2 so the 15 KB log tail keeps them. The s3 diag, 4 M x 5 routing self-check (with its same-input graph replays) and the cold/warm timing lines are unchanged apart from the launch wrapper.

**Review fixes applied.** Per-state blob + ptr record (spec), audit printed, capture GA before and after building B (review open question 1), second state with its own bank (review open question 2 is moot for moe, the clone makes weights distinct anyway). The review's conclusion that no other global holds a pointer was re-checked by the scanner. Where review and spec differ there was no conflict.

**Deviations.** (1) Base correction above. (2) GA2 and the cloned bank are additions beyond the spec's GA/GB sequence. (3) Setup now also prints `OFF:` for non-replay failures (canary, self-check, timing), one line total.

**Biggest risk.** Nothing here was compiled or run (no GPU/torch on this node): the new C++ (wg_make_maps, the maps/mptr wg_gemm, `Maps` by-value from a memcpy'd local) and the self-check graph captures of two states in one process are untested. A C++ compile error lands on Marlin via the build try/except.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only a host python-int pointer compare (`_wg_state_live`) and CPU-tensor reads in C++ (`maps`, `mptr`); no .item()/.cpu()/sync/alloc-on-values in kernel_fn or _moe_wgmma (git diff: syncs and .item() only inside prepare-time functions).
- [x] Nothing derived from `data` cached across calls: the maps/pointer record are derived from the state's own scratch in prepare(); the replay check uses synthetic inputs and keeps none.
- [x] Weights repacked losslessly in prepare() only (_wg_bank unchanged, asserts round trip); NVFP4 stays NVFP4; the cloned bank in the check is a bytewise copy discarded after the check.
- [x] Activations BF16 as before; KV cache: n/a (MoE problem); precision unchanged (fp32 accumulate in wgmma, bf16 staging as s3).
- [x] Attention: n/a (MoE problem).
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: grepped the file for subprocess, multiprocessing, fork, Popen, os.system, spawn: no matches (load_inline's own build is the same as s3's).
- [x] All work inside the timed call; no output caching or replay: the self-check replays only synthetic graphs inside prepare().
- [x] torch.cuda arch: TORCH_CUDA_ARCH_LIST=9.0a set and restored around load_inline (_wg_build, unchanged), wgmma/setmaxnreg PTX guarded by __CUDA_ARCH_FEAT_SM90_ALL with error-word-2 fallback, arch flags printed.
- [x] Self-check announces before and syncs/prints after each launch; plain cp.async.cg/bulk (no createpolicy); -lineinfo kept; mbarrier waits bounded with error word 1.
- [x] No module/static globals holding device pointers or maps: audit scanner (C++ and Python) finds none; per-state maps.

okc_lint output:
```
OK with warnings: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe/muse_moe_w4a16/submission.py
(plus 60 warn lines: host-sync / dtype in prepare()-only code, none in kernel_fn's call graph)
```
py_compile: ok; __pycache__ deleted.

## Risks / what could fail
medium
Stage 4 of the cuda-nvfp4-wgmma-rega-grouped rewrite. It fixes s3 (3bdec0e58d684c17): every bucket canary, the self-check and the 5 judge tests passed, then a benchmark recheck failed on the first cross-copy replay. The held moe cuda-nvfp4-wgmma-rega-midm-bucket-cap32 and -w2-only-midm hypotheses and the prefill experts-turn1-cuda-wgmma-rega-port-bucket32 port all reuse the same _wg_* module, including the global g_maps. They must rebuild on this stage's commit (per-state maps plus the refill and cross-copy self-check), not on s2 or s3.

## Result (after the run)
- job `4e7f8be895a145ba` (check), status done, passed True, tests 5/5
- speedup 1.3439x, 95% CI [1.3237, 1.3641], score 10007.111881394072 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: SGLang (no base result)
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe`
