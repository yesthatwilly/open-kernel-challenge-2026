---
problem: muse_prefill_layer
slug: qk-norm-fp8-cache-store-fusion
created: 2026-10-23T230250Z
base: master
area: pre-attention norm+rope + cache store
change: >
  One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8),
  v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q
  and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM
  output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with
  eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on
  sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin
  table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1)
  exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching
  parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k
  rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1
  (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns
  1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of
  v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3
  call stays the parity path. Before submitting, compare against the parity kernels on random T=64
  inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP
  FP8 differences from fp32 summation order in the RMS.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "four fewer launches (~10-15 us) and three fewer passes over the [753, 1152] bf16 q/k slices; ~1-1.5% of 1120-1465 us, the shapes where launch count still matters"
  - shapes: "*load=turn_4*"
    direction: flat
    why: same launches saved, ~0.3% of 3.4-4.9 ms, inside the interval
  - shapes: "*load=first_turn*"
    direction: flat
    why: ~0.1% of 17-26 ms
risk: medium
depends_on: sglang-pipeline-parity
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T071251Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T071251Z
base_sha: f526f70d8a581a7fa0a7df17ee7a297eb2e2000b
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/qk-norm-fp8-cache-store-fusion
worktree: .worktrees/muse_prefill_layer/qk-norm-fp8-cache-store-fusion
mode: check
implemented_at: 2026-10-24T071713Z
impl_commit: 28a212da340b37c2c639d847cc19b3f5682bb48a
submission_sha256: 53d5ff81af849ac6e1bf25fb7652483edd110b2033cbe396f6520c92c597647a
norm_sha256: aca709ca28d3a4f7dcd31e0ba072a909b990cc9875ff7cc71c59f807a2dc8f1c
submission_blob: 0a23d0c3affb72b5ba2b8e554d5681065494a214
lint_warnings: 1
attempt_file: 2026-10-24T071713Z-check-qk-norm-fp8-cache-store-fusion.md
prepared_at: 2026-10-24T071713Z
---

# qk-norm-fp8-cache-store-fusion

## Change

One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1) exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1 (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns 1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3 call stays the parity path. Before submitting, compare against the parity kernels on random T=64 inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP FP8 differences from fp32 summation order in the RMS.

## Expected

- `*load=turn_1*` faster: four fewer launches (~10-15 us) and three fewer passes over the [753, 1152] bf16 q/k slices; ~1-1.5% of 1120-1465 us, the shapes where launch count still matters
- `*load=turn_4*` flat: same launches saved, ~0.3% of 3.4-4.9 ms, inside the interval
- `*load=first_turn*` flat: ~0.1% of 17-26 ms

## Risk

Medium risk on exactness, low on performance. The cache-entry bound is 8% relative and the new entries are FP8-rounded in both paths, so a one-ULP FP8 flip on a few lanes is harmless; a RoPE pairing or cos/sin layout mistake would fail the first sliding test. Torch's .to(float8_e4m3fn) and Triton's .to(tl.float8e4nv) both round to nearest even without saturation; q and k are RMS-normalized so no lane can overflow, v comes from the same bf16 GEMM output in both paths. Rules: queries FP8 as FA3 does, cache FP8, nothing cached, no host sync. The gain is small by construction; a flat result on turn_1 is the honest outcome and attributes to launch overhead not mattering at M=753.

## Adversarial review

Reviewed 2026-10-24T071251Z by an adversarial subagent (model opus).

**Verdict: proceed.** Correct and within the rules, but the expected gain is about 0.3% on the geomean against a CI of at least +/-1.5%, so a neutral headline is what the run will most likely show. Run it once with no companions, and judge it from the profile's kernel table (the five replaced kernels against the one fused kernel), not from the speedup alone.

### Justification
Mechanism: on master the attention front runs five launches between the qkv GEMM and FA3 in the scored (kv_scale == 1) path: FusedQKNorm, k.to(FP8), v.to(FP8) on a strided slice of qkv, store_cache and q.view(T,16,64).to(FP8). Fusing them into one Triton kernel is plausible, and it is the same shape of fusion that gave attention_decode 1.127x. Size of the prize: at T=753 these kernels move only about 1.7 MB of q/k plus 0.2 MB of v in and about 1 MB of FP8 out. That is roughly 1 us of DRAM time, so nearly all the saving is launch and inter-kernel gaps inside the graph, about 1.5-3 us per kernel removed, or 5-10 us per shape. The hypothesis says 10-15 us, which is on the optimistic side. FINDINGS gives only the top three kernels per shape for master (Marlin MoE 45-66%, the nvjet GEMMs, FA3 on full layers). The only full per-kernel profile on disk (8a2df1fde4f54a7e) is the starter calibration and does not apply. So no measured number backs the 10-15 us. Direction and shapes: the expected directions agree with TASK.md. turn_1 (46x sliding at ~1080 us, 16x full at ~1435 us) gains about 0.5-1% on its load, turn_4 about 0.2%, first_turn about 0.05%. Each load counts equally in the geomean, so the problem-level gain is about 0.25-0.4%, well inside the +/-1.5% interval. Even turn_1's per-shape gain sits at or under the per-shape noise. The sister decode_layer cache-store-fused-fp8-cast was neutral on every shape, which fits launch savings not showing at this scale. Weakest link: the payoff is close to the resolution floor of the measurement. Exact numerical parity is the other weak spot: the hypothesis assumes FusedQKNorm applies norm and then RoPE in fp32 with a single bf16 rounding at the end. If FusedQKNorm rounds to bf16 between norm and rope, the fused kernel will differ by one ULP in places. That is harmless under the 8% cache bound and the attention bound, but it would break the pre-submit 'bitwise equal' check the change itself specifies, so read the FusedQKNorm source before trusting that check.

### Confounders on the current kernel
- sglang-pipeline-parity (the only landed hypothesis): this change replaces part of its attention front, so the interaction is the change itself, not a confounder. Nothing else landed touches the pre-FA3 launch sequence. moe-align-side-stream is neutral and not on master, and its side-stream fork comes after attention, so it cannot overlap or compete with the new kernel.

### Interference and A/B needs
No companion is warranted. The problem is resolution, not attribution. An ab-revert has nothing to revert: the only landed item is the base whose kernels are being replaced. An ab-tune of the parity path would not change these five launches. A control that isolates one sub-change, such as fusing only the casts and store_cache while keeping FusedQKNorm, splits an effect of about 0.3% that is already below the interval into two smaller ones. Both arms would come back neutral, which spends quota without making the result easier to read. The result can be read without companions as follows. In the okc profile for the run, compare the new kernel's time and the attention-front span against the five parity kernels' summed time on the turn_1 shapes. If the fused kernel is shorter by a few us, the mechanism is confirmed even with a flat headline. If the fused kernel is slower than FusedQKNorm alone (per-row programs with poor occupancy at T=753), that explains a regression. Grab the parity kernel times from master's profile, or from the parent job's .txt if it has more than the top three, so the comparison is like for like.

### Open questions
- Does FusedQKNorm round to bf16 once after norm+rope, or once after norm and again after rope? This decides whether the pre-submit check can be bitwise.
- What are master's actual per-kernel times for FusedQKNorm, the three casts and store_cache at T=753? FINDINGS lists only the top three kernels, so the 10-15 us estimate has no measurement behind it.
- fa3-contiguous-kv-gather (in flight) changes how the FA3 call reads the cache. If it lands first, this change needs a rebase so the fused store feeds whatever layout that branch expects, and the two should not be credited together without a rerun.

## Implementation notes

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

## Rules check

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

