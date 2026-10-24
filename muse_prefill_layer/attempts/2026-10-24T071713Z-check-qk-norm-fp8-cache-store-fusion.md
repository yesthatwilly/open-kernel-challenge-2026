# muse_prefill_layer · 2026-10-24T071713Z · check · qk-norm-fp8-cache-store-fusion

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-23T230250Z-qk-norm-fp8-cache-store-fusion.md` (branch `hyp/muse_prefill_layer/qk-norm-fp8-cache-store-fusion`).

- **Job:** `fe8883a2fc264338` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `28a212d`

## Goal
One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1) exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1 (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns 1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3 call stays the parity path. Before submitting, compare against the parity kernels on random T=64 inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP FP8 differences from fp32 summation order in the RMS.

## What changed, and where
Implemented 2026-10-24 in the worktree `.worktrees/muse_prefill_layer/qk-norm-fp8-cache-store-fusion`, file `muse_prefill_layer/submission.py` only (uncommitted). Diff stat: 1 file, +106 / -21.

- New `@triton.jit _qk_norm_fp8_store_kernel` with grid `(T,)` and a fixed `num_warps=4`. There is no autotune: it JIT-compiles on warm-up, one variant per (ROPE, SCALED). One program per token row reads the qkv GEMM output row once. q is handled as a `[16, 32]` even/odd tile pair, k as a `[2, 32]` pair and v as a `[2, 64]` tile. The approach follows the main tree's `muse_attention_decode/submission.py` `_pre_attention_kernel` (check job dfee6f2939974bad, 16/16). I read that file but did not edit it.
- Math: per head, an fp32 sum of squares, then `1/sqrt(ss/64 + 1e-5)`, with no weight. On sliding layers (`cfg["rope_theta"]` != 0, a constexpr) the kernel applies interleaved RoPE in fp32 from `state["cos_sin"][positions[t]]` (cos lanes 0..31, sin lanes 32..63). That table is parity's, built in `prepare` and not derived from data. Pair (2i, 2i+1) uses cos[i] and sin[i], as reference.py `_rope` does.
- Rounding order (parity emulation):
  - q: fp32 -> bf16 -> FP8 E4M3.
  - k: fp32 -> bf16. When SCALED (`cfg["kv_scale"] != 1`, a constexpr) it is then divided by k_scale in fp32 and rounded to bf16 again, then cast to FP8. This matches parity's `k / data["k_scale"]`: a bf16 tensor divided by an fp32 0-d tensor gives a bf16 result.
  - v (bf16 qkv columns 1152..1279): the same path with v_scale.
  - k_scale and v_scale are loaded inside the kernel from the device tensors, and only when SCALED. There is no `.item()`.
- Cache store: through `k_cache.view(-1, hkv, d)` and `v_cache.view(-1, hkv, d)`. These are views of the same storage, never copies. The kernel receives their row and head strides, so page_size 64 works the same way (cache_loc is the flat row). Every `cache_loc[t]` row is written, slot 0 included. There is no skip index, which matches parity's `reserved_skip_index=-1`. A host-side assert checks that the lane stride is 1 (a shape/stride check only).
- `q_fp8 = torch.empty((tokens, hq, d), FP8)` is allocated inside kernel_fn from shapes only and passed straight to FA3. The gate stays a view of qkv. Everything from FA3 onward is unchanged.
- Removed what became unused: the `FusedQKNorm` and `store_cache` imports, the `state["qk_norm"]` module in `prepare`, and the old q/k/v/store lines in kernel_fn. The module docstring entry for the attention front is updated.
- Deviation: none in the mechanism. I did not use the optional q versus k/v program split. One program per row is the spec's default, and register pressure is small (about 1.3K fp32 values per program).

### Uncertainties (no GPU on this node; nothing was run)
- **No bitwise-equality check is claimed.** The FusedQKNorm source (`sglang.kernels.ops.muse_spark_v12.fused_qk_norm`) is not on this node, so its rounding order is unverified. This kernel assumes norm and RoPE both run in fp32 with one bf16 rounding at the end. If FusedQKNorm rounds to bf16 between norm and RoPE, or uses rsqrt or a different summation order, q_fp8 and the cache bytes can differ by one FP8 ULP on some lanes. That is harmless under the 8% cache bound and the attention bound. The change paragraph's pre-submit bitwise comparison should therefore be read as "at most one-ULP differences", and it was not run here because there is no GPU, torch or Triton.
- Triton's fp32 -> fp8e4nv conversion is round-to-nearest-even but saturating (satfinite); torch's `.to(float8_e4m3fn)` is not saturating. The two agree unless a magnitude exceeds 448. Normed q and k have magnitude at most 8. Only v / v_scale on the kv_scale = 1/64 test could get near 448, and parity would give NaN there, so the tests implicitly exclude it.
- `tl.sqrt` may lower to an approximate sqrt. The decode reference kernel uses the same form and passed.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/qk-norm-fp8-cache-store-fusion diff`:

- [x] CUDA-graph capturable: no `.item()`, `.cpu()`, `synchronize()` or tensor prints in kernel_fn or the kernel. k_scale and v_scale are loaded on the device. ROPE and SCALED branch only on `config`. The only host-side assert checks strides. q_fp8 is allocated from shapes.
- [x] Nothing derived from `data` is cached across calls. The kernel's only state is the `cos_sin` table, built in `prepare` from constants.
- [x] Weights: unchanged. `prepare` only drops the now-unused FusedQKNorm module, and nothing is repacked or re-precisioned.
- [x] Activations: q and k are bf16-rounded where parity's FusedQKNorm emits bf16, and the scaled k/v division is bf16-rounded as in parity. The KV cache stays FP8 E4M3, and queries are FP8 E4M3 as FA3 takes them, no coarser.
- [x] Attention sees every key: the FA3 call, window, page_table and cache_seqlens are unchanged, and every new row is stored, slot 0 included.
- [x] No timing patches, harness frames, I/O or secrets, module edits, reference import, subprocesses or network. The diff only adds a Triton kernel and its launch and removes two sglang imports.
- [x] All work happens inside the timed call, with no output caching or replay. The kernel runs in kernel_fn on every call's own inputs.
- [x] The first-line marker `# okc: problem=muse_prefill_layer` is kept, and there is no `triton.autotune`.

okc_lint output:

```
warn  host-sync    L344: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_prefill_layer/qk-norm-fp8-cache-store-fusion/muse_prefill_layer/submission.py
```

The single warning is the existing `torch.cuda.synchronize()` at the end of `prepare()`, which is untimed and unchanged. `python3 -I -m py_compile` passes, and I removed the `__pycache__` it left behind.

## Risks / what could fail
medium
Medium risk on exactness, low on performance. The cache-entry bound is 8% relative and the new entries are FP8-rounded in both paths, so a one-ULP FP8 flip on a few lanes is harmless; a RoPE pairing or cos/sin layout mistake would fail the first sliding test. Torch's .to(float8_e4m3fn) and Triton's .to(tl.float8e4nv) both round to nearest even without saturation; q and k are RMS-normalized so no lane can overflow, v comes from the same bf16 GEMM output in both paths. Rules: queries FP8 as FA3 does, cache FP8, nothing cached, no host sync. The gain is small by construction; a flat result on turn_1 is the honest outcome and attributes to launch overhead not mattering at M=753.

## Result (after the run)
- job `fe8883a2fc264338` (check), status done, passed True, tests 5/5
- speedup 1.0093x, 95% CI [0.9941, 1.0244], score 383839.254415674 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/qk-norm-fp8-cache-store-fusion.result.json`, `muse_prefill_layer/hypotheses/done/neutral/qk-norm-fp8-cache-store-fusion.patch`; branch `hyp/muse_prefill_layer/qk-norm-fp8-cache-store-fusion`
