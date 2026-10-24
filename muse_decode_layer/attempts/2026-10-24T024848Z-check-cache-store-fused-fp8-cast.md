# muse_decode_layer · 2026-10-24T024848Z · check · cache-store-fused-fp8-cast

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-23T225338Z-cache-store-fused-fp8-cast.md` (branch `hyp/muse_decode_layer/cache-store-fused-fp8-cast`).

- **Job:** `7264526cf9f94c37` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `4630a4d`

## Goal
In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(tokens, hq, d).to(FP8), with ONE Triton kernel, one program per token row: it reads the normed+roped k row (bf16 [T, 128]) and the v slice of qkv (bf16 [T, 128], strided view of the [T, 2304] GEMM output, read with explicit strides, no .contiguous()), divides both by k_scale / v_scale only when cfg['kv_scale'] != 1.0 (a shape/config branch, decided on the host), rounds to float8_e4m3fn with round-to-nearest-even and writes them to k_cache.view(-1, 128)[cache_loc[t]] and v_cache.view(-1, 128)[cache_loc[t]]; the same program also reads the 1024-wide q row and writes it to a preallocated fp8 [T, 16, 64] tensor that is then passed to FA3 as q. k/v stay bf16 before the cast exactly as SGLang keeps them. reserved_skip_index=-1 semantics are preserved by never skipping any row. Everything else in the attention path (FusedQKNorm, FA3 call, FusedRMSNormGate, o GEMM) is unchanged. Attribution: 4 launches become 1 on every shape; it is only visible on the launch-bound T<=8 shapes.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/cache-store-fused-fp8-cast`, only `muse_decode_layer/submission.py` (uncommitted; diff stat: 1 file, +48 / -11).

- New `@triton.jit _cache_store_q_fp8_kernel`, grid `(T,)`, fixed `num_warps=4`, no autotune. Per token: loads the q row as a `[16, 64]` tile from FusedQKNorm's q via explicit strides, casts bf16 -> fp32 -> `tl.float8e4nv` and stores into a contiguous fp8 `[T, 16, 64]` buffer; loads k (`[2, 64]`, explicit strides) and v (the strided `qkv[:, 1152:1280]` view via `v.stride(0)/stride(1)`, no `.contiguous()`), upcasts to fp32, divides by `k_scale`/`v_scale` only under `SCALED` (constexpr = host-side `cfg["kv_scale"] != 1.0`), casts to fp8 and stores at `k_cache.view(-1,128)[cache_loc[t]]` / `v_cache.view(-1,128)[cache_loc[t]]`. No row is skipped (reserved_skip_index=-1 semantics); no value-dependent masking or branching.
- The fp8 q buffer is `torch.empty` inside `kernel_fn` on every call (no per-shape buffer in `state`), passed to FA3 as `q=q_fp8`.
- Removed: `k.to(FP8)`, `v.to(FP8)`, the scaled-path `k / k_scale`, `v / v_scale` elementwise ops, `store_cache(...)` (import dropped), `q.view(...).to(FP8)`.
- `k = k.reshape(tokens, hkv, d)` (master also reshaped; a view when FusedQKNorm's k is contiguous); `q.view(tokens, hq, d)` as master. A host-side assert checks the two cache row views share strides with unit last stride (shape-only, graph-safe).
- FusedQKNorm, the FA3 call, FusedRMSNormGate, the o GEMM and everything downstream are untouched. The module docstring line for the cache-store stage is updated.
- Numerics: the unscaled path is one RNE rounding bf16 -> fp8, as torch's cast. The scaled path divides in fp32 then rounds once, where master rounded the quotient to bf16 first (double rounding); the difference is far inside the 8% cache bound.
- Uncertainty: Triton's fp32 -> fp8e4nv cast on sm90 saturates at +/-448 (cvt.rn.satfinite) where torch gives NaN above 448; this can only remove NaNs, not add them. Not run on GPU here.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/cache-store-fused-fp8-cast diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn; branching only on `cfg["kv_scale"]` (config) and strides (shape); the fp8 q `torch.empty` is a shape-sized allocation as master's `.to(FP8)` already made, not allocation on input values.
- [x] Nothing derived from `data` is cached across calls: the q buffer is allocated per call, nothing new in `state`.
- [x] Weights: n/a: no weight or `prepare()` change.
- [x] Activations BF16 where SGLang keeps BF16 (k/v stay bf16 until the cast); KV cache FP8 E4M3; queries FP8 E4M3, no coarser.
- [x] Attention sees every key: n/a for this change; the FA3 call (window, causal, page table, seqlens) is unchanged.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: only `triton` / `triton.language` imports added.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept.
- [x] Review rules notes followed: no `triton.autotune`; fixed constexpr config; v read with explicit strides, no `.contiguous()`; kv_scale divide only under the host-side config branch, in fp32 then cast; no row skipped; no value masking or branching.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L276: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```

The one warning is pre-existing (in `prepare()`), not from this diff. `python3 -m py_compile` passes.

## Risks / what could fail
low
Risk: the FP8 rounding must match torch's .to(float8_e4m3fn) (RNE, non-saturating: values above 448 become NaN in torch; Triton's fp8e4nv conversion saturates or NaNs depending on version). The cache check tolerance is 8% relative L2 and no-NaN, so a saturation difference on an outlier value could trip the NaN check in the dist=outliers test (batch 3, ctx 6000, q_len 8) or the kv_scale=1/64 test: implement the scaled path and test both. The v slice is a strided view of the qkv GEMM output; passing strides to Triton avoids a hidden copy kernel that would give back one of the launches. The q fp8 output must be contiguous [T,16,64] so FA3 takes it without a cast. Confounder: FA3 may itself cast q if given bf16; after this change q is already fp8, so any FA3-side cast kernel disappears too, which belongs to the same mechanism. Also note the second cache-related kernel SGLang runs, store_cache, is a JIT op that may already be a single fused scatter; the saving is then 3 launches, not 4.

## Result (after the run)
- job `7264526cf9f94c37` (check), status done, passed True, tests 13/13
- speedup 1.0303x, 95% CI [1.0149, 1.0458], score 21954.33589134917 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.020038644953596, "ci": [1.004738065279292, 1.0353392246278998], "job": "5755a412dda244a1"}; base result: muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/cache-store-fused-fp8-cast.result.json`, `muse_decode_layer/hypotheses/done/neutral/cache-store-fused-fp8-cast.patch`; branch `hyp/muse_decode_layer/cache-store-fused-fp8-cast`
