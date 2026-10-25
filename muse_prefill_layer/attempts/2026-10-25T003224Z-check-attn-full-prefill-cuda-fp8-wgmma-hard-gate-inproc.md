# muse_prefill_layer · 2026-10-25T003224Z · check · attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T001824Z-attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc.md` (branch `hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc`).

- **Job:** `306cbde58aff4fa4` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `227ec48`

## Goal
BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its attention code is unchanged from the old base): start from master's submission.py and hand-port the attention path from done/regression/attn-full-prefill-cuda-fp8-exp-offload.patch (branch hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload: _ATTN_CUDA, attn_fwd_kernel<SW>, attn_combine_kernel, _attn_splits, _full_attn_gate, _full_attn, _attn_prepare), WGMMA BACKEND ONLY. That run (job f5c30b15, 0.881) never built wgmma (same ptxas '.target sm_90' bug as the expert rewrite), and routed its mma.sync fallback, which was 3.07-3.21x slower than FA3 on all five full shapes, because its timing was print-only (REWRITE_READ_2026-10-25 #3). Changes vs that patch, and only these: (1) BUILD: os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately before load_inline (restore after), only -gencode arch=compute_90a,code=sm_90a; wrap all wgmma/fence/commit/wait asm in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that writes an error flag and returns; log the nvcc arch flags and compile seconds. (2) DELETE THE mma.sync BACKEND: build only the USE_WGMMA=1 module; there is no second CUDA kernel to fall back to; the only fallback is FA3 exactly as master calls it. (3) NO EXP OFFLOAD: compile attn_fwd_kernel<0> only (EXP_SW_NB = 0, every exponential on MUFU ex2); keep exp2_sw and its unit test out of this stage; sweep the fraction only in a later stage after wgmma parity is shown (the old run measured offload 2/8 as 3% slower than 0/8). (4) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for attn_fwd_kernel<0> and attn_combine_kernel; anything else turns the path OFF); read back the error flag from (1) after the canary (non-zero -> OFF); then a tiny synthetic canary launch of attn_fwd_kernel<0> (both desc_swap orders) and attn_combine_kernel on synthetic FP8 caches (b1, tiny ctx), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; any failure prints '[cuda full attn] OFF: <reason>' and FA3 runs. Then run the self-check cases in-process (b1 ctx3000 L753, b3 ctx5000 L200 outliers, b4 {0,777,5000,80000} L753, b1 ctx0 L16384, b2 {37,1000} L211) with the existing pass rule. (5) HARD TIMING GATE in prepare(), replacing the print-only timing: for each full-layer shape class the judge scores (turn_1: b1 L753 ctx~80000; turn_4: b4 L753 ctx~80000; first_turn: b1 L16384 at ctx 0, 32768, 65536), CUDA-graph capture both the FA3 call (+ its combine) and the new kernel (+ attn_combine_kernel when split) on synthetic inputs of that exact config, flush L2 between replays (write a >= 2x L2-size buffer), take the median of >= 20 replays, and route the new kernel for that shape class only if new <= 0.92 x FA3. The decision is a dict keyed by config-only fields (batch, q_len, page_table width bucket) stored in state and read by _full_attn_gate in kernel_fn; any shape not timed, and every sliding / page-64 / kv-scaled case, calls FA3 exactly as master. Log one line per class: '[cuda full attn] GATE <class>: FA3 X us, new Y us, ratio R -> CUDA|FA3', plus a final 'PATH CHOSEN' summary. Everything else in the layer (q cast, o_norm_gate, experts, router) is master's. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## What changed, and where
Worktree `muse_prefill_layer/submission.py` only (diff: 1 file, +1026 / -18). Master's code is unchanged except for the FA3 call site in `kernel_fn`, a timing line in `prepare()`, and three stdlib imports (`os`, `statistics`, `time`).

- **Port.** The attention block from `done/regression/attn-full-prefill-cuda-fp8-exp-offload.patch` sits before `prepare()`: `_ATTN_CUDA`, `attn_fwd_kernel`, `attn_combine_kernel`, `_attn_splits`, `_full_attn`, `_fa3_attn`, `_attn_case`, `_attn_reference`, `_attn_errs`, `_attn_check_data`, `_attn_self_check` and `_attn_prepare`. The wgmma core, the cp.async ring, the V transpose, the softmax, the split and the combine are byte-for-byte the patch's USE_WGMMA=1 code.
- **(1) Build.** `_build_attn()` runs at import. It sets `os.environ["TORCH_CUDA_ARCH_LIST"]="9.0a"` right before `load_inline` and restores the old value in `finally`. The cflags are `-O3 -lineinfo`. `-arch=sm_90a` is gone, so the only gencode comes from torch's list. The build logs torch's `_get_cuda_arch_flags()` output (the list the JIT ninja path uses) and the compile seconds. The module name is new (`muse_prefill_full_attn_wgmma_sm90a_v2`), so a stale cached build is not reused. All wgmma, fence, commit and wait helpers, plus the kernel body, are under `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`. The `#else` branch does `atomicExch(err, 1)` and returns. The error word is a new int32 `[1]` arg, kept in `state["attn_err"]`.
- **(2)** The mma.sync backend is deleted, along with `USE_WGMMA` and `_ATTN_BACKENDS`. Only one module is built.
- **(3)** There is no exp offload. `exp2_sw`, `sw_col`, `exp2_test` and `_check_exp2` are removed, and every exponential uses `ex2.approx`. *Deviation:* the template parameter is dropped too, so the kernel is the plain `attn_fwd_kernel`, the same as `<0>`. The profile row will read `attn_fwd_kernel`.
- **(4) In-process guard.** It runs in this order:
  - `_attn_image_check`: the device capability must be (9,0); the arch flags must be only compute_90a/sm_90a, and unknown flags count as a failure; binaryVersion must be 90 for both kernels. ptxVersion is logged but not gated, because it is not a reliable sm_90a signal.
  - `_attn_canary` (b1, ctx 100, L 40): fwd at desc_swap 0 and then 1 (nsplit 1), then fwd with nsplit 2, then combine. Each step gets `torch.cuda.synchronize()` and a print, then the error word is read.
  - The 5 self-check cases with the patch's pass rule, at desc_swap 0 and then 1. The error word is read again afterwards.
  - Any failure prints `[cuda full attn] OFF: <reason>` and FA3 runs.
  - No process of any kind is started.
- **(5) Hard gate.** `_attn_gate` covers 5 classes (turn_1, turn_4, first_turn ctx 0/32768/65536). For each it CUDA-graphs `_fa3_attn` (FA3 with its internal combine) and `_full_attn` (with `attn_combine_kernel` when split). Both get the same pre-cast FP8 q, so no cast is timed on either side. Before every replay it rewrites a buffer of max(2xL2, 128 MB). The two graphs alternate and their order swaps every repetition. It takes the median of 25 replays and routes a class only if new <= 0.92x FA3. Each class logs `GATE <class>: FA3 X us, new Y us, ratio R -> CUDA|FA3`, followed by one `PATH CHOSEN` summary line. The decision is `state["attn_route"]`, keyed by `(batch, q_len, round((width - q_len)/16384))`. That key is 5 for the turn shapes and 0/2/4 for first_turn. `_full_attn_route` reads it in kernel_fn. It also prints one `PATH CHOSEN batch=.. q_len=.. width=..` line the first time each full-layer shape is seen. A module-level set de-dups those prints, and the set is never used for routing. Untimed shapes, and sliding, page-64 and kv-scaled cases, call FA3 exactly as master does.
- **Open questions.**
  - turn_4 is timed with a mild ragged spread (74000/78500/81500/86000), because the judge's spread is not in the repo. It falls in the same width bucket as fixed 80000. If the judge uses a lognormal (4x width) spread, the buckets miss and FA3 runs: safe and flat.
  - Build, canary, self-check, gate and total prepare seconds are all logged.
  - The ported core does NOT overlap K/V loads with the MMA beyond the 4-stage cp.async ring. Every wgmma is followed by wait_group 0, and there is no warp specialization.
- **Uncertainty.**
  - This was not compiled or run locally (no GPU or nvcc here).
  - `torch.utils.cpp_extension._get_cuda_arch_flags` is private. If the judge's torch lacks it, the path turns OFF for "unverified flags".

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc diff`:

- [x] CUDA-graph capturable. kernel_fn adds only `_full_attn_route` (config and page_table shape) and `_full_attn` (torch.empty plus two extension launches). Every `.item()` and `synchronize()` call is in prepare-only functions (lint: all host-sync findings are outside kernel_fn's call graph). Nothing branches on input values.
- [x] Nothing derived from `data` is cached. `state["attn_route"]` comes from synthetic prepare() inputs and is keyed by config and shapes. The `_ATTN_ROUTE_LOGGED` set holds shape keys for print de-dup only.
- [x] Weights untouched: the change is attention only.
- [x] Activations: q is rounded to E4M3 as before (FA3's own query precision). The KV cache stays FP8. P is rounded to E4M3, which is FA3's P precision. The output is BF16.
- [x] Every key is attended: k in [0, position] for each token, no skipping. Splits partition the keys and the combine merges all of them.
- [x] No timing patches, harness access, module edits, reference import or network. No subprocess, multiprocessing or child process (no matches in the diff). `os.environ` is set and restored around `load_inline` only.
- [x] All work happens inside the timed call. prepare() runs only synthetic checks and timing, and nothing is replayed or cached.
- [x] The first-line marker `# okc: problem=muse_prefill_layer` is kept.
- [x] n/a: no triton.autotune (no Triton added).

Lint (`okc_lint.py --problem muse_prefill_layer`): **OK with warnings**. There are 29 `host-sync` warnings, all "outside kernel_fn's call graph; fine in prepare()". They are master's existing ones plus the new prepare-only syncs and `.item()` reads (L471-L1605). There are no errors. `py_compile` OK; `__pycache__` removed.

## Risks / what could fail
high
supersedes attn-full-prefill-cuda-fp8-wgmma-hard-gate, held for a child-process self-check

## Result (after the run)
- job `306cbde58aff4fa4` (check), status done, passed True, tests 5/5
- speedup 1.0679x, 95% CI [1.0519, 1.0839], score 362768.4447441995 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc.result.json`, `muse_prefill_layer/hypotheses/done/neutral/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc.patch`; branch `hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc`
