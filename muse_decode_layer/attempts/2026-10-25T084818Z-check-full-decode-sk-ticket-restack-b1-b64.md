# muse_decode_layer · 2026-10-25T084818Z · check · full-decode-sk-ticket-restack-b1-b64

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T054108Z-full-decode-sk-ticket-restack-b1-b64.md` (branch `hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64`).

- **Job:** `9c99379067084f6e` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `67bca2f6`

## Goal
Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention-side stream-K ports that each won their rows on the previous master, in ONE muse_decode_sk TU with two template instantiations and disjoint config gates: (a) from branch hyp/muse_decode_layer/full-decode-streamk-ticket-port, splitkv_attn_sk_kernel<4,true,false> (single-level ticket, grid 2 x SM) replacing master's splitkv_attn_kernel + splitkv_combine_kernel at full layer, q_len 1, page 1, 1 < batch <= 64 (run 6d48932b: selector B=16 107.6 vs 120.6 us, B=32 222.8 vs 257.4; profile 109.7 / 227.3 us; full decode_16 306.7->296.4, decode_32 479.6->437.5); (b) from branch hyp/muse_decode_layer/full-decode1-two-level-ticket-port, splitkv_attn_sk_kernel<4,true,true> (two-level ticket, grid = SM count) replacing FA3 + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate at full decode batch == 1 (run 5e44c7ae: selector 24.8 vs 34.1 us; profile 20.1 us; full decode_1 142.7->137.7). Both read the q prepared by master's split-KV pre-attention route (extend it to B=1 as (b) did) and write the bf16 tensor the o GEMM reads; master's kernel A/B and FA3 chain stay compiled as fallbacks. Sliding layers, verify and full decode B>64 untouched. The previous ports kept the ticket buffer as a module-level import-time zero buffer: MOVE it (both ticket levels) and the fp32 partial workspace into the per-state dict, sized from config in prepare; the kernel resets tickets to 0 on exit, and the self-check must prove it across alternating states. Self-check per gate: B in {1,2,5,16,32,33} with agentic and uniform lengths, kv_scale 1 and 0.25, vs master kernels and fp32 reference; selectors at B=1,16,32. HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on input values, data-dependent control flow or per-call mutable state; gates read config only (layer, batch, q_len, window, page); nothing cached from inputs; precision per TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES: the judge prepares two state copies and rotates them, so every ticket buffer, partial workspace, device pointer and descriptor lives in the per-state dict built in prepare(), never in a module/static global (LESSONS moe g_maps). Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with refilled inputs (new q/KV/lengths each replay) and compare each to master; any miss falls back to master path for that gate and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a, -std=c++20 if cflags overridden), print regs/smem/local/binaryVersion, first launch a tiny canary + synchronize with launching/ok prints. The prepare-time selector prints CUDA-event timing new vs incumbent per gated batch and keeps the incumbent if new is slower. First measurement is an okc check.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree (+1118/-2, no diff against master 00e36e3 before the edit). Nothing was run: this node has no GPU or torch.

What changed:
- One new module `muse_decode_sk` (load_inline at import, `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored, `-O3 -std=c++20 -lineinfo`, arch flags printed). The kernel template is byte-identical in both port branches (checked by diff); it has two instantiations: variant 0 = `<4,true,false>` (grid 2 x SM) and variant 3 = `<4,true,true>` (grid = SM count). No wgmma/setmaxnreg, so no `__CUDA_ARCH_FEAT_SM90_ALL` guard is needed; the kernel uses cp.async.cg, mma.sync and cvt e4m3 only.
- kernel_fn: `_sk_variant(cfg, state["sk"])` (config plus the state's own flags). B == 1 uses variant 3, which replaces FA3 + FlashAttnFwdCombine + FusedRMSNormGate. For 1 < B <= 64 it uses variant 0 where `_splitkv_gate` holds, replacing kernels A + B. The B=1 q route is the existing qk_norm/cast block (no extra change was needed). Kernels A/B and the FA3 chain stay as fallbacks. Sliding, verify and B > 64 are untouched.
- Per state, not global: prepare() builds `state["sk"]` with ws_acc, ws_ml, ticket[64], gticket[32], gacc and gml (sized from SM count and kernel MAXB only, zeroed). The only module globals are the compiled module, arch flags, plain-bool decision dict and a wall-time float. The C++ side keeps only function-attribute flags in a static, no pointers.

Review fixes applied (the review wins over the spec where they differ):
- Tickets and workspaces per state: done. The ports' module-level ticket and per-call torch.empty workspaces are gone. The ws is preallocated per state, which the review flagged as a layout change that should be neutral under a graph.
- Self-resetting: the last arriver zeroes its ticket (both levels). The self-check proves it: every replay of the alternation check asserts the state's ticket sum is 0.
- `__threadfence` before the ticket atomic: present in both levels (all threads fence, `__syncthreads`, thread 0 draws). I also switched the single-level draw from `atomicAdd` to the same `atom.acq_rel.gpu` helper the two-level path uses.
- prepare() zeroes every ticket before returning: a final `_sk_zero` plus sync, with the sum printed.
- Stream-K covers [0, seqlen): the kernel masks positions >= seq and uses ceil(seq/64) tiles. Self-checks cover seq%64 in 1..7, lengths 1/7/33/63/64/65, agentic B=2 (ctx 80000), B=5 (20000), B=33 (8000), and kv_scale 1 and 0.25 against the fp32 reference. New: a planted-key probe (k = 6 x that kv head's head-0 q at the first / 63 / 64 / last position, so a dropped or doubled key changes the output by O(1), not 1e-4).
- Norm-gate fp32 internal: `finalize_store` computes o/L*v_scale, the sum of squares, rsqrt, sigmoid and the product in fp32 and rounds to bf16 only at the store. The incumbent rounded o to bf16 before the norm.
- launch_dependents: the incumbent kernels A/B and FA3 execute no griddepcontrol (grep: only the dense-chain TU has any), so the sk kernels issue none; this answers the review's open question.
- Workspace sized from config/hardware only (2 x SM + 64 slots; batch-1 group slots = 32).
- PATH CHOSEN lines (per state, three lines: B=1, B<=23, B>=24) carry the selector timing, `w2_mode=<state["w2_mode"]>` and `prepare_wall_cum=<s>`. A `[prepare] this call ..., cumulative ...` line prints at the end of every prepare().
- Gates (LESSONS 2026-10-25): a path is chosen only if the graph-timed incumbent/new ratio is >= 1.02 (cold L2 flush, interleaved, 15 reps), at B = 1, 16, 32 on agentic ragged lengths (ctx 80000 +/- 7.5%). The incumbent for B=1 is FA3 + combine + norm-gate, for 16/32 it is kernels A + B. Class 16 covers B <= 23, class 32 covers B >= 24.

Two-state proof: the first prepare() runs the full validation once with temporary buffer sets. Its alternation check (B=1 at 80000, B=2 at 80000, B=5 at 20000, B=33 at 8000) builds two independent buffer sets, captures one graph on each, and replays them alternately six times with refilled q/KV/page table/kv scale/lengths, NaN-poisoning the output each time. Every prepare() (including the second) also runs a graph-replay check on that state's own buffers.

Deviations: (1) the worktree and the ports are on different bases, so the self-check was rewritten around per-state buffers rather than copied; (2) the selector uses graph timing with the 1.02 margin (ports used eager timing and 0.99); (3) `launch_dependents` was not added because the incumbent has none.

Biggest risk: first-ever compile of the combined module (never built as one TU: the two ports built different single instantiations) and the longer prepare() (two builds plus about 40 reference checks and graph captures). Reading `[sk]`/PATH CHOSEN lines in the log is the first step; the build line must show 9.0a.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: `_full_decode_sk` only allocates `out` via torch.empty (as master's kernel B did) and launches; the tickets/workspaces are preallocated per state; no `.item()`/`.cpu()`/sync in kernel_fn (every sync is in prepare-time helpers); the capture itself is exercised in the self-check.
- [x] Nothing derived from `data` is cached across calls: `state["sk"]` holds only zeroed tickets and scratch sized from SM count and kernel constants, plus bool flags from synthetic data.
- [x] Weights: n/a: no weights touched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: q is cast to E4M3 exactly as in the split-KV path, output is bf16, softmax and merge in fp32 (per TASK.md).
- [x] Attention sees every key: stream-K covers [0, seqlen) exactly, with masked tails, verified against the fp32 reference including a planted-key probe at tile edges and the last position.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: grep for subprocess/multiprocessing/fork/Popen is empty; the build and self-check are in-process.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Two-state rule: no device pointer, workspace, ticket or descriptor in a module/static global; everything comes from `state["sk"]`.
- [x] sm_90a: `TORCH_CUDA_ARCH_LIST=9.0a` around load_inline and restored; arch flags printed; cubin version 90 asserted; no sm_90a-only PTX is used.

okc_lint output: `OK with warnings: submission.py` (only prepare-time host-sync warnings, all outside kernel_fn's call graph). `python3 -I -m py_compile` passed; `__pycache__` removed.

## Risks / what could fail
medium
Land candidate if the three full decode rows reproduce: the kernels are unchanged from runs that passed every test; the new risk is only the per-state ticket move. Judge on the three full decode rows and the sk kernel profile rows, not the headline (sum of row gains ~+0.008-0.012).

## Result (after the run)
- job `9c99379067084f6e` (check), status done, passed True, tests 13/13
- speedup 1.2847x, 95% CI [1.2654, 1.3040], score 17607.434433862647 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2725637226942819, "ci": [1.2534752668538676, 1.291652178534696], "job": "3fd3fa1f96d045dd"}; base result: muse_decode_layer/hypotheses/done/neutral/w2-marlin-atomic-add-t1-port.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode-sk-ticket-restack-b1-b64.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode-sk-ticket-restack-b1-b64.patch`; branch `hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64`
