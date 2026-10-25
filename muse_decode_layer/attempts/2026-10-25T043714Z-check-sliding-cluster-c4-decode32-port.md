# muse_decode_layer · 2026-10-25T043714Z · check · sliding-cluster-c4-decode32-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T042121Z-sliding-cluster-c4-decode32-port.md` (branch `hyp/muse_decode_layer/sliding-cluster-c4-decode32-port`).

- **Job:** `0fb2c9451c5d4283` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `a3136ce`

## Goal
Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's sliding_cluster_kernel (fp8qk TU, module muse_sliding_cluster_v2) from branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc, instantiating ONLY cluster size C=4 and gating it to sliding layer, q_len 1, page 1, 17 <= batch <= 32 (config only), i.e. sliding decode_32. C=8 lost at B=16 in attention and C=16 was kernel-flat at B=1, so neither is built; batch 1 keeps decode_layer's _sliding_fused_kernel and 2..16 keeps master. The kernel takes the qkv GEMM output, does q/k RMSNorm + RoPE + FP8 (FA3-equivalent), stores the new k/v rows into the FP8 cache, gathers the window per rank, merges the 4 ranks over DSMEM and applies the RMSNorm x sigmoid(gate) epilogue, writing the bf16 tensor decode_layer's o GEMM reads; it replaces decode_layer's sliding chain between the qkv GEMM and the o GEMM at that band. Map decode_layer's qkv column layout, norm weights, rope base and window to the kernel arguments; the window covers every key (no skipping). At import call cudaOccupancyMaxActiveClusters for C=4 and disable the path if < 32. Self-check at B in {17, 32} (lens < window and > window, kv_scale 1 and 0.25): output vs master's chain, cache rows equal, graph replay. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree (+1096/-82; about 80 of the deleted lines are a move).

What changed:
- `_master_attn(data, state, qkv)`: master's whole chain between the qkv GEMM and the o GEMM (everything that was kernel_fn's `else:` branch) moved into a function, byte-for-byte apart from indent. kernel_fn calls it, and the self-check and selector call the same function, so the reference is master's exact kv_scale division and q FP8 cast.
- kernel_fn: `elif _slide_gate(cfg): o = _slide_forward(data, qkv)` sits right after the batch-1 sliding branch, where a batch 2..16 branch would go. The gate reads config only (q_len 1, window, rope_theta, 17 <= batch <= 32, 16/2/64 heads, page 1) plus the module constant `_SLIDE_PATH` that prepare() sets. There is no try/except, timing, sync or occupancy query in kernel_fn.
- New section before `prepare()`: `sliding_cluster_kernel` ported verbatim from the attention branch (module `muse_sliding_cluster_c4_v1`, fp8qk only, one load_inline, no f16 fallback build). Only the host side was trimmed: `slide_setup` reports occupancy for C=4 only, `slide_attn` requires C == 4 and B <= 32. The kernel stores every cache_loc row, slot 0 included.
- Build: `TORCH_CUDA_ARCH_LIST=9.0a` set around load_inline and restored, flags `-O3 -std=c++20 -lineinfo`, the arch flags printed, compile seconds printed. The kernel needs no sm_90a-only PTX (mma.sync e4m3, clusters, cp.async), so no `__CUDA_ARCH_FEAT_SM90_ALL` guard or error flag. There are no mbarrier or spin waits (cluster barriers and cp.async groups only).
- `_sliding_forward` gained an optional `splits` argument (default unchanged), used for the Triton 16-split candidate (sibling hypothesis sliding-batched-triton-b16-band) at B=17..32.
- `_slide_prepare` (called last in prepare(), never raises):
  1. Prints regs / local spill / dynamic smem / binaryVersion / ptxVersion and `cudaOccupancyMaxActiveClusters` for C=4. It disables the cluster path if binaryVersion != 90 or occupancy < 32.
  2. Runs a tiny B=1 canary (flushed `launching ... ok` prints around a synchronize), then a self-check at B=17 (kv_scale 0.25), B=32 (kv_scale 1) and B=32 (kv_scale 0.25), with lens below, at and past the window and window-edge keys planted. Request 0's new row is slot 0.
  3. The self-check compares master's chain, the cluster and Triton-16 against an fp32 reference. The cluster must match its own-rounding reference (rel L2 < 2e-2) and sit within 2.5x of master's error against the unrounded fp32 reference plus 5e-3. The Triton candidate must be < 2e-2. v cache rows must be byte-exact against master's, k rows may differ by one E4M3 code in < 6% of bytes (master rounds through bf16 first), and no other cache row may change. The cluster launch is also captured and replayed as a graph and must equal the eager output.
  4. Times the WHOLE master chain, the cluster launch and the Triton-16 launch as CUDA graphs at B=32, interleaved, 30 reps after 5 warm-ups. A 320 MB write flushes L2 before every replay. It prints all medians and sets `_SLIDE_PATH` to the fastest candidate only if it is strictly faster than the master chain; otherwise master. Every outcome prints `PATH CHOSEN` (with `(fallback: <reason>)` on a fallback).

Review fixes applied (the review wins where it differs from the spec): no donor try/except; every cache_loc stored including slot 0 (and tested); donor q rounding order kept (q * qk_scale_factor then E4M3, not FA3's round-then-scale; both are E4M3, so within the rules, and the self-check is calibrated against master's own error rather than a fixed bound); no timing, sync or occupancy query in kernel_fn; selector flushes L2 and times the whole replaced chain, plus the Triton 16-split candidate; one instantiation (~41 s extra compile expected, printed per module); the decision comes from the printed medians.

Deviations from the spec: the selector times at B=32 only and applies the choice to 17..32; the canary is B=1, which the host check allows.

How to judge (per the review): compare the `sliding_cluster_kernel` row against the SUMMED rows it replaces (FA3 main 23.7 + combine 2.8 + prepare 2.2 + FusedQKNorm, the FP8 casts, store_cache and FusedRMSNormGate rows), not only the 236 us shape line. Expect -5..-10 us on the shape. Read the `PATH CHOSEN` and median lines first: if prepare chose master or triton16 the profile will not show the cluster row. Prepare-time timings ranked paths but undercount the judged time by 20-25% (LESSONS).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn's new branch is one `slide_attn` launch (or the existing Triton launch) with torch.empty outputs; no `.item()`/sync/print/data-dependent control flow; the cluster launch and the master chain are captured and replayed in prepare(). The sync, `int(...)` and `.nonzero()` calls are all in prepare-only helpers.
- [x] Nothing derived from `data` is cached across calls: only `_inv_freq` (config constant, existing helper), the Triton ticket counters (zeroed, left at 0 by every launch, existing mechanism) and the module/path constants set in prepare().
- [x] Weights: n/a: no weight is touched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3 (q rounded to E4M3 once, as the review keeps).
- [x] Attention sees every key: window 2048 covers the token and the 2047 before it; every key of every rank's chunk is scored, nothing skipped; the self-check plants window-edge keys.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: no process creation anywhere (load_inline builds in-process, as the two existing modules do); the CUDA-event timing is in prepare() only.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_decode_layer/submission.py --problem muse_decode_layer`): `OK with warnings`, 0 errors. All warnings are `host-sync` outside kernel_fn's call graph (prepare-only), plus one `dtype` warning for the uint8 L2-flush scratch buffer in prepare() (never an activation or weight). `python3 -I -m py_compile` passes and `__pycache__` was deleted.

## Risks / what could fail
medium
124 regs + 32 B spill in attention; 103 KB smem/CTA. Check the spill line before trusting.

## Result (after the run)
- job `0fb2c9451c5d4283` (check), status done, passed True, tests 13/13
- speedup 1.2693x, 95% CI [1.2503, 1.2883], score 17820.971063118075 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2855726646854595, "ci": [1.2662890747151776, 1.3048562546557412], "job": "7a9f5de5aecc4aba"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/sliding-cluster-c4-decode32-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/sliding-cluster-c4-decode32-port.patch`; branch `hyp/muse_decode_layer/sliding-cluster-c4-decode32-port`
