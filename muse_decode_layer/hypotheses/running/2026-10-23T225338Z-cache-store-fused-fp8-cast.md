---
problem: muse_decode_layer
slug: cache-store-fused-fp8-cast
created: 2026-10-23T225338Z
base: master
area: cache store
change: >
  In kernel_fn, replace the four separate launches between FusedQKNorm and
  flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(tokens, hq,
  d).to(FP8), with ONE Triton kernel, one program per token row: it reads the normed+roped k row
  (bf16 [T, 128]) and the v slice of qkv (bf16 [T, 128], strided view of the [T, 2304] GEMM
  output, read with explicit strides, no .contiguous()), divides both by k_scale / v_scale only
  when cfg['kv_scale'] != 1.0 (a shape/config branch, decided on the host), rounds to
  float8_e4m3fn with round-to-nearest-even and writes them to k_cache.view(-1, 128)[cache_loc[t]]
  and v_cache.view(-1, 128)[cache_loc[t]]; the same program also reads the 1024-wide q row and
  writes it to a preallocated fp8 [T, 16, 64] tensor that is then passed to FA3 as q. k/v stay
  bf16 before the cast exactly as SGLang keeps them. reserved_skip_index=-1 semantics are
  preserved by never skipping any row. Everything else in the attention path (FusedQKNorm, FA3
  call, FusedRMSNormGate, o GEMM) is unchanged. Attribution: 4 launches become 1 on every shape;
  it is only visible on the launch-bound T<=8 shapes.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "T=1 is launch-bound (SGLang ~172 us against a 40 us floor, ~30 launches): three fewer ~2-3 us launches is ~4-5% of the shape"
  - shapes: "*load=verify_1*"
    direction: faster
    why: "T=8, same launch-bound regime: ~2-3% of ~268 us"
  - shapes: "*load=decode_16*"
    direction: flat
    why: same 3 launches saved but the shape is dominated by GEMM/Marlin/FA3 bytes; at or inside the CI, control
  - shapes: "*load=verify_32*"
    direction: flat
    why: "T=256: the saved launches are below 1% of ~700-1050 us; control"
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T024344Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T024344Z
base_sha: 54112d428da559b67789e07fa8f2e586fe5fb45e
base_blob: 6884d26a928d1de1b07bf8d28e1be4e53dfdcfc5
branch: hyp/muse_decode_layer/cache-store-fused-fp8-cast
worktree: .worktrees/muse_decode_layer/cache-store-fused-fp8-cast
mode: null
implemented_at: 2026-10-24T024848Z
impl_commit: 4630a4dc3abbef94b3293018d7490b59ebeea37d
submission_sha256: d0de623c11e01c92a025fbdfafb41aa60a68f774b3a8f166ac03dc6ec571881b
norm_sha256: 1ff1758a3aea755c91e1cd7b5126fd32e66575d15a58a4257c207d81d165e4d1
submission_blob: 03d45c741603c0d133b48ff4e65bbbf0cf86980f
lint_warnings: 1
---

# cache-store-fused-fp8-cast

## Change

In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(tokens, hq, d).to(FP8), with ONE Triton kernel, one program per token row: it reads the normed+roped k row (bf16 [T, 128]) and the v slice of qkv (bf16 [T, 128], strided view of the [T, 2304] GEMM output, read with explicit strides, no .contiguous()), divides both by k_scale / v_scale only when cfg['kv_scale'] != 1.0 (a shape/config branch, decided on the host), rounds to float8_e4m3fn with round-to-nearest-even and writes them to k_cache.view(-1, 128)[cache_loc[t]] and v_cache.view(-1, 128)[cache_loc[t]]; the same program also reads the 1024-wide q row and writes it to a preallocated fp8 [T, 16, 64] tensor that is then passed to FA3 as q. k/v stay bf16 before the cast exactly as SGLang keeps them. reserved_skip_index=-1 semantics are preserved by never skipping any row. Everything else in the attention path (FusedQKNorm, FA3 call, FusedRMSNormGate, o GEMM) is unchanged. Attribution: 4 launches become 1 on every shape; it is only visible on the launch-bound T<=8 shapes.

## Expected

- `*load=decode_1*` faster: T=1 is launch-bound (SGLang ~172 us against a 40 us floor, ~30 launches): three fewer ~2-3 us launches is ~4-5% of the shape
- `*load=verify_1*` faster: T=8, same launch-bound regime: ~2-3% of ~268 us
- `*load=decode_16*` flat: same 3 launches saved but the shape is dominated by GEMM/Marlin/FA3 bytes; at or inside the CI, control
- `*load=verify_32*` flat: T=256: the saved launches are below 1% of ~700-1050 us; control

## Risk

Risk: the FP8 rounding must match torch's .to(float8_e4m3fn) (RNE, non-saturating: values above 448 become NaN in torch; Triton's fp8e4nv conversion saturates or NaNs depending on version). The cache check tolerance is 8% relative L2 and no-NaN, so a saturation difference on an outlier value could trip the NaN check in the dist=outliers test (batch 3, ctx 6000, q_len 8) or the kv_scale=1/64 test: implement the scaled path and test both. The v slice is a strided view of the qkv GEMM output; passing strides to Triton avoids a hidden copy kernel that would give back one of the launches. The q fp8 output must be contiguous [T,16,64] so FA3 takes it without a cast. Confounder: FA3 may itself cast q if given bf16; after this change q is already fp8, so any FA3-side cast kernel disappears too, which belongs to the same mechanism. Also note the second cache-related kernel SGLang runs, store_cache, is a JIT op that may already be a single fused scatter; the saving is then 3 launches, not 4.

## Adversarial review

Reviewed 2026-10-24T024344Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** The mechanism is real and already demonstrated end-to-end by the sibling check dfee6f2939974bad on muse_attention_decode (same four launches plus FusedQKNorm fused into one Triton kernel, 16/16 tests, fused kernel 1.3-1.5 us, 3.5-6.4 us saved per shape). Nothing landed interacts with the pre-attention stage, so one run is interpretable without companions; the expected decode_1 gain is ~2%, not 4-5%, which sits at the churn's 1.5% per-shape noise threshold and must be read from the per-shape rows against the merged base, not from the overall CI.

### Justification
Master (muse_decode_layer/submission.py:298-311) runs k.to(FP8), v.to(FP8), store_cache and q.to(FP8) serially on the main stream before FA3. Removing 3 of those 4 launches shortens the critical path on every shape. Sibling evidence (muse_attention_decode/profiles/submission.dfee6f2939974bad.txt): the 5->1 fusion saved 3.5-6.4 us per shape, ~1.0-1.6 us per removed launch under graph replay. Scaling to 3 removed: ~3-4.5 us against the merged base rows (decode_1 sliding 168.73 us, full 180.65; verify_1 sliding 233.99, full 250.89), i.e. ~1.8-2.0% on decode_1 and ~1.5-2.0% on verify_1; T>=16 shapes move <1%, correctly flat. Overall geomean moves ~+0.6%, inside the CI, so the signal lives in per-shape medians (err_ns at T=1 ~0.37%). Weakest link: the hypothesis's own numbers are inflated; the expected ~2% per-shape gain is only marginally above the 1.5% attribution noise, and the parity run showed a ~1% unexplained per-run offset at T=1. A true gain may be scored neutral by threshold; a flat result would not refute the mechanism.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8 (only landed item): changes ROUTER_FP32_MAX_TOKENS 8->1 on the side-stream router after the o projection; shares no launch, memory traffic or numerical path with the pre-attention cast/store stage. Additive, not interacting; its parity port fixes the base rows this run must be compared against.

### Interference and A/B needs
No companion. ab-revert of the router constant only perturbs T=8 rows via an unrelated path; nothing to re-tune; a control splitting the q-cast from the k/v cast+store is unnecessary since the sibling kernel already shows all pieces fused correctly, and any implementation loss is visible as extra or long kernels in the run's own profile. What the run needs is a reading rule: judge per shape against the merged base rows with err_ns.

### Open questions
- Expected decode_1 is ~1.8-2.0% (3-4 us), not 4-5%; treat a 2-4 us per-shape delta on the four T<=8 rows as success, against the merged base rows with err_ns.
- store_cache is already a single kernel, so the saving is 3 launches.
- A follow-up should port the sibling's _pre_attention_kernel (FusedQKNorm fused too) onto this base for a further ~1-1.5 us per shape.
- Verify no at::native copy/elementwise kernel remains between FusedQKNorm and FA3 in the profile.
- Confirm the fp8 q buffer handed to FA3 is [T,16,64] with unit last-dim stride and FA3 inserts no extra cast/copy.


## Implementation notes

Worktree `.worktrees/muse_decode_layer/cache-store-fused-fp8-cast`, only `muse_decode_layer/submission.py` (uncommitted; diff stat: 1 file, +48 / -11).

- New `@triton.jit _cache_store_q_fp8_kernel`, grid `(T,)`, fixed `num_warps=4`, no autotune. Per token: loads the q row as a `[16, 64]` tile from FusedQKNorm's q via explicit strides, casts bf16 -> fp32 -> `tl.float8e4nv` and stores into a contiguous fp8 `[T, 16, 64]` buffer; loads k (`[2, 64]`, explicit strides) and v (the strided `qkv[:, 1152:1280]` view via `v.stride(0)/stride(1)`, no `.contiguous()`), upcasts to fp32, divides by `k_scale`/`v_scale` only under `SCALED` (constexpr = host-side `cfg["kv_scale"] != 1.0`), casts to fp8 and stores at `k_cache.view(-1,128)[cache_loc[t]]` / `v_cache.view(-1,128)[cache_loc[t]]`. No row is skipped (reserved_skip_index=-1 semantics); no value-dependent masking or branching.
- The fp8 q buffer is `torch.empty` inside `kernel_fn` on every call (no per-shape buffer in `state`), passed to FA3 as `q=q_fp8`.
- Removed: `k.to(FP8)`, `v.to(FP8)`, the scaled-path `k / k_scale`, `v / v_scale` elementwise ops, `store_cache(...)` (import dropped), `q.view(...).to(FP8)`.
- `k = k.reshape(tokens, hkv, d)` (master also reshaped; a view when FusedQKNorm's k is contiguous); `q.view(tokens, hq, d)` as master. A host-side assert checks the two cache row views share strides with unit last stride (shape-only, graph-safe).
- FusedQKNorm, the FA3 call, FusedRMSNormGate, the o GEMM and everything downstream are untouched. The module docstring line for the cache-store stage is updated.
- Numerics: the unscaled path is one RNE rounding bf16 -> fp8, as torch's cast. The scaled path divides in fp32 then rounds once, where master rounded the quotient to bf16 first (double rounding); the difference is far inside the 8% cache bound.
- Uncertainty: Triton's fp32 -> fp8e4nv cast on sm90 saturates at +/-448 (cvt.rn.satfinite) where torch gives NaN above 448; this can only remove NaNs, not add them. Not run on GPU here.

## Rules check

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
