# muse_attention_decode · 2026-10-25T043825Z · check · full-decode1-sk-pdl-from-preattn

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T042335Z-full-decode1-sk-pdl-from-preattn.md` (branch `hyp/muse_attention_decode/full-decode1-sk-pdl-from-preattn`).

- **Job:** `c0f1945a108c4aa2` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `d4a9f92`

## Goal
Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is 22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize() (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup, the tile-range computation, page_table index loads, and cp.async issue of the prologue ring stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket writes stay after the wait. Batch > 1 unchanged (the running full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1 cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN: full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of pre-attn+kernel with and without PDL and assert PDL is not slower.

## What changed, and where
Edited only `muse_attention_decode/submission.py` in the worktree (194 insertions, 20 deletions). No GPU on this node: nothing was compiled or run, only `py_compile` and `okc_lint`.

- Kernel: new template flag `PDL` on `splitkv_attn_sk_kernel` (variant 4 = `<4, ticket, two-level, PDL>`; variants 0-3 compile to the same code as before). Launched via `cudaLaunchKernelEx` with `cudaLaunchAttributeProgrammaticStreamSerialization = 1`. Before `griddepcontrol.wait` the kernel does only `load_prefix` (cache_seqlens), the tile-range math, `issue_meta` (page_table) and the cp.async of ring tiles that are not the last tile of a request. The primary writes only q and K/V row seq_b-1, which lives in the request's last tile.
- Review fix applied (the review wins over the spec): q is no longer in the first tile's cp.async group. `issue_tile(nn, withq)`: the PDL prologue passes `withq=false`; after the wait, threads `tid < 64` copy the prologue segment heads' q with plain 16 B loads into the q slots, published by the `__syncthreads` at the top of the main loop. Main-loop tiles (after the wait) keep the cp.async q. The wait sits just before the first prologue tile that is last-of-request (found with `find_req` on smem values, so it is CTA-uniform), else right after the prologue; tiles after it are issued after the wait. Workspace, ticket, gticket and output writes are all later than the wait.
- Variant (b): `_pre_attention_kernel` gets `TRIGGER: tl.constexpr = False`; when True it runs `tl.inline_asm_elementwise("{ griddepcontrol.launch_dependents; mov.u32 $0, $1; }", ...)` (is_pure=False) on `tl.arange` at the top and uses the result as `i`, so the asm is kept and ordered first. `kernel_fn` passes `TRIGGER=True` only when `splitkv and batch == 1 and _B1_PDL == 2`; every other shape compiles the TRIGGER=False specialization, identical to before.
- Selection (`_sk_pdl_select`, import time, after the B=1 validation): variant 4 attributes (cubin 90); 7 cases x trigger {off, on} with the two kernels back to back and no sync (B=1 80000 / 4097 / 64 / 63 / 1 keys, grids SM / 7 / 3, plus B=16 ragged); then `kernel_fn` captured into three CUDA graphs (mode 0 landed plain launch, 1 PDL attribute only, 2 PDL + trigger), each replayed twice with NaN-poisoned output against the FP32 reference; then 40 interleaved cold-L2 (256 MB flush) graph replays per mode at synthetic B=1 x 80000. The fastest mode is chosen only if its median beats plain by more than 0.05 us; otherwise, and on any error, mode 0 (landed). Prints per-mode median/p10 and `PATH CHOSEN: full decode_1 streamk two-level + PDL ...` or `... no PDL (...)`, plus the final `... launch plain / (a) / (b)` line.
- Deviation: the prepare timing is graph-replay of pre-attention + kernel (as the brief asks), not eager CUDA events. The hypothesis asked to "assert PDL is not slower": implemented as a selector (never ships a PDL mode that is not faster), not a hard assert.
- Not guarded by `__CUDA_ARCH_FEAT_SM90_ALL`: `griddepcontrol` is plain sm_90 PTX, and a guard with an `#else` that skipped the wait would be a hazard. The module still builds with `TORCH_CUDA_ARCH_LIST=9.0a` (existing `_build_sk`).
- Biggest risks: (1) first compile of the Triton `inline_asm_elementwise` with a no-op `griddepcontrol.launch_dependents` (falls back to mode 0/1 only via the exception handler if compile raises at warm-up; the selector catches that and returns 0); (2) PDL edges captured in a CUDA graph need CUDA >= 12.3, else capture raises and the landed path is kept; (3) the gain may be below the 1 us kill bar (see open questions: record as a stacking item for sk4 if only the implicit variant helps).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()` in kernel_fn; the diff adds a constexpr kwarg and a variant index in kernel_fn/_full_decode_sk only; syncs are in prepare-time `_sk_pdl_select`.
- [x] Nothing derived from `data` is cached across calls (`_B1_PDL` is a global chosen from synthetic data at import; tickets are device-only as before).
- [x] Weights: n/a: attention problem, no weights touched.
- [x] Activations BF16 / KV cache FP8 / queries at most FP8 E4M3: unchanged (the q is the same FP8 query).
- [x] Attention sees every key (full context): unchanged; the deferred last tile is still read in full after the wait.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess: checked the diff; none. No child process anywhere.
- [x] All work happens inside the timed call; no output caching or replay: the PDL launch is still part of kernel_fn; import-time work is self-check and timing selection on synthetic data only.
- [x] sm_90a: n/a for arch guard (griddepcontrol is sm_90); build env unchanged (TORCH_CUDA_ARCH_LIST=9.0a around load_inline, restored).
- [x] Failure handling: any build error, self-check miss, graph capture error or raise lands on mode 0 (landed path) with a PATH CHOSEN line; never raises.

okc_lint:
```
warn  host-sync lines (18, all `torch.cuda.synchronize()` / `.item()` in prepare-time self-checks and diagnostics, "outside kernel_fn's call graph; fine in prepare()")
OK with warnings: submission.py
```

## Risks / what could fail
medium
Kill: decode_1 full shape not >= 1 us faster. splitkv-pdl-attn-to-combine (A->B) was neutral; this boundary differs because the dependent here can do useful gather work before the wait.

## Result (after the run)
- job `c0f1945a108c4aa2` (check), status done, passed True, tests 16/16
- speedup 1.4571x, 95% CI [1.4317, 1.4790], score 2584.704643280593 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-pdl-from-preattn.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-pdl-from-preattn.patch`; branch `hyp/muse_attention_decode/full-decode1-sk-pdl-from-preattn`
