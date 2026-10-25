---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
created: 2026-10-25T044131Z
base: master
area: "wgmma NVFP4 grouped w13+silu/w2 at 128<=T<=512: per-state tensor maps, graph-replay safety"
change: >
  Fix the s3 judge failure (job 3bdec0e58d684c17: verify 16 bs=128 "recheck failed on repetition
  2: relative L2 error 1.211e+00"). CAUSE (s3 branch submission.py): the 8 TMA maps live in ONE
  process-global `Maps g_maps[2]` inside the shared load_inline module (:972). wg_set_maps encodes
  them over the CALLING state's wg_xs/wg_hs (:1026-1045; called from prepare at :2122-2124), and
  every launch passes g_maps[which] by value (:1087), which the graph then freezes. The judge
  prepares several copies (setup record `copies: 2`) and rotates them across repetitions. The last
  prepare() re-points g_maps at its own scratch, so on every other copy: the gather writes that
  copy's state["wg_xs"] (:1364) and w13 writes its state["wg_hs"] (:1368), but the w13 TMA reads
  the OTHER copy's xs and the w2 TMA reads the OTHER copy's hs. Those hold rows from whatever the
  other copy last ran. Reps 0/1 pass (the other copy's rows still match those inputs); rep 2 is
  the first replay where the other copy has run different inputs. It is not a rules violation:
  n_padded, chunks and buckets are read on the device (:836-889), and kernel_fn branches on shape
  only (:2364). The s3 self-check could not see this. It used one state, captured per case and
  replayed the SAME inputs 3 times (:1897-1900). FIX. Base: master 9080a71 (w2 atomic-add landed).
  T<128 and T>512 run master's Marlin unchanged. Port the s3 kernels with variant A
  (wgmma_wait<0>) ONLY in every bucket. Drop variant B, the vmask argument and the A/B canary
  timing. (1) Delete the g_maps/g_maps_ok globals. Add wg_make_maps(xs, hs), which returns a CPU
  uint8 tensor [2, sizeof(Maps)] stored in state["wg_maps"]. wg_gemm takes that tensor plus the
  B-source scratch tensor, memcpys row `which` into the __grid_constant__ Maps argument, and
  TORCH_CHECKs that the scratch data_ptr equals the one recorded when the maps were encoded (keep
  a per-blob int64 ptr pair in state). Each copy's graph then freezes maps over its OWN scratch.
  (2) Keep wg_xs/wg_hs/wg_err per state (no module-level scratch, no static device pointers
  anywhere). (3) Audit the module for any other static/global that holds a device pointer, and
  print the result. MANDATORY self-check before PATH CHOSEN=WGMMA. Any failure prints one `OFF:
  <reason>` line and master's Marlin runs. (a) Refill replay: for M in (128, 256, 257), allocate
  static x / topk_ids / topk_weights, capture kernel_fn ONCE, then for >=3 different routings
  (random, skewed 4-hot, all-one-expert, mixed local+remote) copy_ the new values into the SAME
  buffers, replay, and compare each output to eager Marlin on the same values (rel-L2 <= 1e-2, row
  bound 5x, no NaN/Inf). (b) Cross-copy: build a second state B with its own scratch and maps
  (call prepare() again, or the wgmma setup alone). Capture GA on state A and GB on state B, then
  run GA, GB, GA, GB, GA with a fresh routing refilled before each replay, and compare each to
  Marlin. (c) Print one line per replay: routing tag, rel-L2, and the map scratch ptr vs that
  state's buffer ptr. Keep the s3 cold timing lines (M=128/256 wgmma vs Marlin) unchanged. The
  first measurement is an okc check, never a submit.
expected:
  - shapes: "*load=verify 16*"
    direction: faster
    why: bs=128 passes every recheck; prepare cold wgmma 669 vs Marlin 921 us at M=128
  - shapes: "*load=verify 32*"
    direction: faster
    why: bs=256 cold wgmma 838 vs Marlin 1251 us at M=256
  - shapes: "*load=decode*,*load=verify 1*"
    direction: neutral
    why: T<128 runs master 9080a71 Marlin unchanged
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: Marlin w13+silu+w2 at 128<=T<=512
review: proceed
status: running
review_at: 2026-10-25T044714Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T050458Z
base_sha: 8261c0480931bf1fdcb44c0d04d1dbf675b0c06a
base_blob: bf65a3afdf6ebe0d9b73a817ffd2be1240f91870
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T051658Z
impl_commit: 928151c3ea4e903fc4a05ae7ac59672291b71d0d
submission_sha256: 35e3210cd344650481fe6a611622bad9d900e699997f66ec43683dee2c770e62
norm_sha256: 48fff635d9984c58be53c056b405dad35e3e0f6ae62a6e96e516a3c4503c7eeb
submission_blob: a13d5fea2e79f85df573670cbd55a8a10fbc3d2c
lint_warnings: 60
attempt_file: 2026-10-25T051658Z-check-cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.md
prepared_at: 2026-10-25T051658Z
job: 4e7f8be895a145ba
dispatched_at: 2026-10-25T051707Z
---

# cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe

## Change

Fix the s3 judge failure (job 3bdec0e58d684c17: verify 16 bs=128 "recheck failed on repetition 2: relative L2 error 1.211e+00"). CAUSE (s3 branch submission.py): the 8 TMA maps live in ONE process-global `Maps g_maps[2]` inside the shared load_inline module (:972). wg_set_maps encodes them over the CALLING state's wg_xs/wg_hs (:1026-1045; called from prepare at :2122-2124), and every launch passes g_maps[which] by value (:1087), which the graph then freezes. The judge prepares several copies (setup record `copies: 2`) and rotates them across repetitions. The last prepare() re-points g_maps at its own scratch, so on every other copy: the gather writes that copy's state["wg_xs"] (:1364) and w13 writes its state["wg_hs"] (:1368), but the w13 TMA reads the OTHER copy's xs and the w2 TMA reads the OTHER copy's hs. Those hold rows from whatever the other copy last ran. Reps 0/1 pass (the other copy's rows still match those inputs); rep 2 is the first replay where the other copy has run different inputs. It is not a rules violation: n_padded, chunks and buckets are read on the device (:836-889), and kernel_fn branches on shape only (:2364). The s3 self-check could not see this. It used one state, captured per case and replayed the SAME inputs 3 times (:1897-1900).

FIX. Base: master 9080a71 (w2 atomic-add landed). T<128 and T>512 run master's Marlin unchanged. Port the s3 kernels with variant A (wgmma_wait<0>) ONLY in every bucket. Drop variant B, the vmask argument and the A/B canary timing. (1) Delete the g_maps/g_maps_ok globals. Add wg_make_maps(xs, hs), which returns a CPU uint8 tensor [2, sizeof(Maps)] stored in state["wg_maps"]. wg_gemm takes that tensor plus the B-source scratch tensor, memcpys row `which` into the __grid_constant__ Maps argument, and TORCH_CHECKs that the scratch data_ptr equals the one recorded when the maps were encoded (keep a per-blob int64 ptr pair in state). Each copy's graph then freezes maps over its OWN scratch. (2) Keep wg_xs/wg_hs/wg_err per state (no module-level scratch, no static device pointers anywhere). (3) Audit the module for any other static/global that holds a device pointer, and print the result.

MANDATORY self-check before PATH CHOSEN=WGMMA. Any failure prints one `OFF: <reason>` line and master's Marlin runs. (a) Refill replay: for M in (128, 256, 257), allocate static x / topk_ids / topk_weights, capture kernel_fn ONCE, then for >=3 different routings (random, skewed 4-hot, all-one-expert, mixed local+remote) copy_ the new values into the SAME buffers, replay, and compare each output to eager Marlin on the same values (rel-L2 <= 1e-2, row bound 5x, no NaN/Inf). (b) Cross-copy: build a second state B with its own scratch and maps (call prepare() again, or the wgmma setup alone). Capture GA on state A and GB on state B, then run GA, GB, GA, GB, GA with a fresh routing refilled before each replay, and compare each to Marlin. (c) Print one line per replay: routing tag, rel-L2, and the map scratch ptr vs that state's buffer ptr. Keep the s3 cold timing lines (M=128/256 wgmma vs Marlin) unchanged. The first measurement is an okc check, never a submit.

## Expected

- `*load=verify 16*` faster: bs=128 passes every recheck; prepare cold wgmma 669 vs Marlin 921 us at M=128
- `*load=verify 32*` faster: bs=256 cold wgmma 838 vs Marlin 1251 us at M=256
- `*load=decode*,*load=verify 1*` neutral: T<128 runs master 9080a71 Marlin unchanged

## Risk

Stage 4 of the cuda-nvfp4-wgmma-rega-grouped rewrite. It fixes s3 (3bdec0e58d684c17): every bucket canary, the self-check and the 5 judge tests passed, then a benchmark recheck failed on the first cross-copy replay. The held moe cuda-nvfp4-wgmma-rega-midm-bucket-cap32 and -w2-only-midm hypotheses and the prefill experts-turn1-cuda-wgmma-rega-port-bucket32 port all reuse the same _wg_* module, including the global g_maps. They must rebuild on this stage's commit (per-state maps plus the refill and cross-copy self-check), not on s2 or s3.

## Milestones

Rewrite stage 4 of `Marlin w13+silu+w2 at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T044714Z by an adversarial subagent (model opus).

**Verdict: proceed.** Diagnosis confirmed from the s3 branch (hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix; its worktree no longer exists). The process-global Maps g_maps[2] / g_maps_ok / g_map_rows[2] (s3 submission.py:972-974) are re-encoded by every prepare() over that state's wg_xs/wg_hs (:1026-1047, called at :2122-2124) and passed by value at launch (:1087), so a graph captured on one judge copy TMA-reads the other copy's scratch. No other module-global holds a device pointer in the s3 module, so the planned per-state map blob is enough. Proceed; it is a rewrite stage with a sound fix and the right self-check.

### Justification
A: The mechanism holds. The gather (:1364) and the w13 output (:1368) use state['wg_xs'] and state['wg_hs'], but both TMA reads go through g_maps[which], which encodes the LAST prepared state's buffers. The last-prepared copy is self-consistent. The other copy reads stale rows, which matches 'passes tests and reps 0/1, fails recheck rep 2 at rel-L2 1.21'. The full audit of the s3 C++ found nothing else that holds a pointer. The other C++ statics are constexpr or static_assert (Smem<G>::BYTES and the like); get_encode re-queries every time (no static cache); cudaFuncSetAttribute is per-function and idempotent. On the Python side, WG_BUILD_INFO (:1135) is diagnostic only. wg_err (:2135), wg_banks (:2130), wg_vmask (:2146) and wg_xs/wg_hs are already per state. The load_inline module is shared across prepare() calls in one process (same name, so the same .so), and that is exactly why any C++ global is cross-copy state. The g_maps_ok flip in wg_set_maps (:1030) is a second hazard: if a later prepare's encode fails, every copy's next launch dies on TORCH_CHECK, not only that copy's. The per-state blob removes this too. Weakest link: the expected speedups are the s3 prepare-time cold timings (669 vs 921 us at M=128, 838 vs 1251 us at M=256). No judge benchmark or profile of the wgmma path exists yet, because s3 failed before timing. Against floors of 101.3 / 108.9 us (TASK.md), both paths are far off-floor, so milestone 2 (at or below the incumbent) is plausible but unproven under judge timing, and milestone 3 is not expected this stage.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71, the base): changes only the T<=32 w2 Marlin call; the wgmma path covers 128<=T<=512, so it is disjoint. Neutral control shapes stay neutral.
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: the wgmma path calls its own moe_align_block_size at block 16 (s3 :1356) instead of master's align, so the align change is part of the replaced subsystem and is not separable from the kernel time. Read the profile per kernel (align + gather + 2x wg_kernel + norm vs align + 2x Marlin + act_and_mul + norm), not the headline.
- marlin-block-m-two-smaller-on-postnorm: tunes Marlin's block_size_m at T>=128, i.e. the incumbent being replaced. The comparison is against this tuned Marlin, which is correct and is not a confounder.

### Interference and A/B needs
No companions. This is a rewrite stage whose purpose is to be correct under the judge's multi-copy rotation. The control is built in: T<128 and T>512 run master Marlin, and the prepare-time cold A/B against Marlin at M=128/256 is kept. An ab-revert of a landed hypothesis would not change whether the fix works.

### Open questions
- Does the judge capture each copy's graph after ALL prepares, or right after its own prepare? The per-state fix is correct either way, but it decides whether s3's failure could ever have shown on rep 0. The self-check should cover both orders: capture GA before and after building B.
- Do the judge's copies hold distinct weight tensors (different data_ptr / values)? That does not matter for moe (wg_banks is per state), but it decides whether the bf16 deepgemm g_tb global (below) is latent or live.


## Implementation notes

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

## Rules check

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


## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only review plus offline emulation; no GPU, no nvcc on this node).

**Verdict: dispatch** (no fixes needed; the worktree file is unchanged by the audit).

- Master port: an AST comparison against `master:muse_moe_w4a16/submission.py` (9080a71) shows every master definition byte-identical (`_align_*`, `_post_norm_combine_kernel`, `_marlin_layer`, `W2_ATOMIC_MAX_TOKENS`, `_w2_atomic_self_check`) except `prepare`, `kernel_fn` and `_forward`. `_moe_marlin` is the body of master's `_forward` with only the argument plumbing and the tuple return changed, and `prepare()` runs master's w2 self-check before `_wg_setup`. T<128 and T>512 (and 128..512 with the path off) launch exactly master's kernels with `state["w2_mode"]`.
- Globals: `_wg_audit_cpp` run offline on the real `_WG_CPP + _wg_src()` reports no findings. With `Maps g_maps[2]`, `bool g_ok` and a `static int` injected, it reports all three, so the scanner works. On the C++ side only constexpr values, static_asserts, types and functions remain. On the Python side, no module global holds a tensor or graph (WG_BUILD_INFO holds strings). Everything a launch reads comes from `state` (wg_ext, wg_maps, wg_mptr, wg_xs/wg_hs, wg_err, wg_grid, wg_banks).
- Two-state check: GA is captured on A before B exists. B is a real second state (own xs/hs/err/maps blob and a cloned bank). GB and GA2 are captured after B, and the sequence GA,GB,GA,GB,GA,GA2,GB refills x/ids/weights in place before every replay and compares each output to eager Marlin. GA2 is the case that would have caught s3's global (the maps re-pointed at B before capture). Every raise lands in `_wg_setup`'s try, which prints one OFF line and PATH CHOSEN = MARLIN. s3's own numbers (out rel ~6e-3, judge worst row ~5e-3) leave wide margin under the 1e-2 / 5e-2 gates, so no gate always fails.
- kernel_fn: it branches on shape, the prepare flag and a host python-int pointer compare (`_wg_state_live`). The C++ reads only the CPU blob and the CPU pointer record. It has no sync, no allocation on values, no prints and no try/except. Lint finds no host-sync in kernel_fn's call graph.
- CUDA: `Maps` (4 x CUtensorMap, 512 B, align 64) is memcpy'd from the blob row into a local and passed by value to `const __grid_constant__ Maps`, the same kernel signature as s3, which compiled and ran. The C++ checks `sizeof(Maps)` against the blob width and checks `mptr[which] == bsrc.data_ptr()`, with bsrc being xs for w13 and hs for w2. The row bound now uses `bsrc.size(0)`: 8192 >= 4576 at M=512, and 286 blocks <= MAXB 384. mbarrier waits are bounded (2^26 cycles, then error word 1). The variant-A consume loop is s3's verbatim. Note: the C++ was not compiled here.
- Prepare time: s3 setup took 38 s per prepare (judge prepare_s 38.4). The replay check adds about 12 captures and 36 replays per prepare, far inside the 300 s budget and the 15 min run.

Residual risk: the new C++ (wg_make_maps, the blob-based wg_gemm) is uncompiled, and a build error silently lands on Marlin (neutral run, OFF line printed).
