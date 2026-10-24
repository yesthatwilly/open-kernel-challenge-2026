---
problem: muse_decode_layer
slug: sliding-fused-triton-decode1-port
created: 2026-10-24T140135Z
base: master
area: FA3 call
change: >
  Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding
  decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from
  muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers
  (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its
  _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In
  kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if
  cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch
  (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE =
  bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed),
  data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'],
  data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated
  attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That
  one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way
  split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x
  sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q
  FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape
  (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path
  byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula,
  as in the attention problem (decode's cos_sin table stays for the other shapes).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: attention measured 16.6 -> 13.1 us for the same replaced launches; here the master path has 4-5 more pre-attention launches (separate casts and store_cache), so expect 4-8 us off 172.8 us, +2.5-4.5% on the shape and ~+2-3% on the decode_1 load
  - shapes: "*layer=full*"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: gated out
  - shapes: "*layer=sliding load=decode_32*"
    direction: flat
    why: gated out
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: gated out
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164533Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T164533Z
base_sha: d170c1fd1af3e71b7de6f970d623da8b2a86af59
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/sliding-fused-triton-decode1-port
worktree: .worktrees/muse_decode_layer/sliding-fused-triton-decode1-port
mode: submit
implemented_at: 2026-10-24T164837Z
impl_commit: 2569d7e0d6f208921d432fd4237dc483601b39ca
submission_sha256: a7fdce245795b5c4fb76c25616180ec6564090ab39265a09e8c2f7f4da569c8a
norm_sha256: e28f6354345920a15ce2904cfeec7c25a5a572b2d1373ffea4eeea50111a5d87
submission_blob: e33a32aa5580fbd5b38764ea287a6364c865432e
lint_warnings: 1
attempt_file: 2026-10-24T164838Z-submit-sliding-fused-triton-decode1-port.md
prepared_at: 2026-10-24T164838Z
job: 12198572363b49a4
dispatched_at: 2026-10-24T164858Z
---

# sliding-fused-triton-decode1-port

## Change

Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE = bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed), data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'], data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula, as in the attention problem (decode's cos_sin table stays for the other shapes).

## Expected

- `*layer=sliding load=decode_1 *` faster: attention measured 16.6 -> 13.1 us for the same replaced launches; here the master path has 4-5 more pre-attention launches (separate casts and store_cache), so expect 4-8 us off 172.8 us, +2.5-4.5% on the shape and ~+2-3% on the decode_1 load
- `*layer=full*` flat: untouched
- `*layer=sliding load=decode_16*` flat: gated out
- `*layer=sliding load=decode_32*` flat: gated out
- `*layer=sliding load=verify_*` flat: gated out

## Risk

Single-shape gain (~+0.4-0.6% geomean) but on the heaviest-weighted shape of a launch-bound load, and the kernel is proven correct and faster on identical inputs in the sibling. Caveat: decode's fused-preattention-triton (neutral) showed that fusing pre-attention launches alone did not move sliding decode_1 here, so the gain may be only the attention+norm-gate part (~3.5 us, +2%). Correctness risk is in the I/O mapping: decode's qkv is [T, 2304] contiguous from the GEMM (attention's was [B, L, 2304]); the fused kernel's RoPE must match FusedQKNorm's interleaved convention (it passed the attention problem's tests with the same SGLang reference). The 13 tests include batch 1 ctx 10 sliding (q_len 1), which takes this path at small ctx and its page size. Composes with post-expert-norm-triton-small-t and the bf16 GEMM port (disjoint kernels on the same shape).

## Adversarial review

Reviewed 2026-10-24T164533Z by an adversarial subagent (model opus).

**Verdict: proceed.** A single-shape, config-gated port of a kernel that is proven correct and faster on the same inputs in muse_attention_decode. It does not interact with anything that has landed. The gain is real but small: about 2-4% on sliding decode_1 and roughly +0.3-0.5% on the geomean, which falls inside the CI. So judge it by the sliding decode_1 row and top_kernels, not by the headline.

### Justification
Mechanism: on sliding decode_1 master runs, after the qkv GEMM, FusedQKNorm, the k and v FP8 casts, store_cache, the q FP8 cast, FA3 (no scheduler_metadata on sliding, so FA3 schedules internally, then runs main and combine) and FusedRMSNormGate. That is about 8-9 serialized graph nodes, and the change collapses them into one 32-CTA launch (1 x 2 x 16). FA3 main alone is 10.2 us of the 145.8 us shape (FINDINGS). The attention problem measured exactly these replaced launches going from 16.56 to 13.12 us, although its baseline already had the fused pre-attention kernel. The I/O mapping checks out against muse_attention_decode/submission.py on master. The qkv row layout (q 16x64, k 2x64, v 2x64, gate 16x64) is identical. decode's qkv is the [1, 2304] contiguous GEMM output, so passing stride(0)=2304 is correct. The positions, cache_loc, seq_lens, page_table, k_scale and v_scale fields have the same names, dtypes and semantics. The caches are [pages, page_size, 2, 64] in both problems. The window is lo = seq - 2047 old keys plus the new key, which equals FA3's (2047, 0). The output is the [T, 1024] bf16 that _gemm(o, dense['o']) expects. inv_freq is built from the same formula as decode's prepare() cos_sin table (theta 500000, arange(0,64,2)/64). Numerically the fused path is finer than master: the query is bf16 rather than E4M3, and the norm-gate is computed on fp32 o before any bf16 rounding. Numbers: the hypothesis quotes 172.8 us for the shape, but that is stale. Master now runs sliding decode_1 at 145.8 us, so 4-8 us would be +2.7-5.5% on the shape. The decode_1 load is 46*145.8 + 16*157.0 = 9219 us-weight, and 4 us there is about 2% of that load, or about 0.33% on the 6-load geomean. That is consistent with the stated direction but below the stated +0.4-0.6%, and well below the ±1.5% CI and the 1.5-2.5% per-shape run drift (LESSONS). Weakest link: the decode_layer finding fused-preattention-triton (neutral) showed that removing the five pre-attention launches did not move sliding decode_1 here. So the claim that the extra pre-attention launches are worth 4-8 us is unsupported. The defensible gain is only the attention-plus-norm-gate part, about 3-3.5 us (+2-2.4% on the shape), which is barely above drift on one row. The run is still readable from the profile: the cutlass flash_fwd row (10.2 us), the FA3 combine, FusedQKNorm, store_cache, the cast kernels and FusedRMSNormGate should disappear from sliding decode_1's top_kernels, replaced by one _sliding_fused_kernel row. That row should be at or below the attention problem's ~9-10 us kernel time.

### Confounders on the current kernel
- None of the four landed hypotheses confounds this change.
- fa3-hdim128-head-pairing-batched: gated to window == 0 and batch > 1, so it never runs on sliding decode_1. The new gate is checked before the paired/unpaired branch and returns o before reaching it.
- post-expert-norm-triton-small-t: active at T=1 on the same shape, but it acts on downstream, disjoint kernels (the MoE combine). Every replay starts with a cold L2, so no cache interaction carries across. The effects add.
- marlin-block-m-cap-16: inert at T=1 (block_size_m 8 is chosen either way).
- sglang-pipeline-parity-router-hilo-m8: T=1 still takes the fp32 F.linear router path, so it is inert for this shape.

### Interference and A/B needs
No companion is warranted. The change is a config-only branch touching exactly one scored shape, and no landed hypothesis acts on that shape's attention block. A gain or a loss is therefore attributable from the sliding decode_1 row's paired ratio together with its top_kernels. Every other row is byte-identical code and works as a drift control: if they move uniformly by +/-2%, that is run drift. A pre-attention-only control (fused attention kernel plus master's FusedQKNorm/store_cache) would not change the decision, because fused-preattention-triton already measured that half as flat here. An ab-revert of any landed item would be uninformative, since none of them overlaps. The headline will be within CI no matter what, so the decision to merge should rest on the sliding decode_1 row being above about +2.5% and the replaced kernels being gone from the profile, not on CI separation.

### Open questions
- Scored-submit correctness. The only test that takes the new path is 'batch 1, ctx 10, layer sliding', at the test default page size (probably 64, which the PAGE constexpr handles), plus the scored batch-1 ctx-80000 sliding decode on every timed call. Splits 1..15 are empty at ctx 10 and are masked to -inf partials; the combine gives them weight 0. Can the implementer confirm the attention problem's test list contains the same batch-1 ctx-10 sliding case, so that this exact path is already proven at small ctx?
- The hypothesis's 172.8 us baseline and +0.4-0.6% geomean estimate are stale against master's 145.8 us row. Expect about +0.3% on the geomean. Should the harvest classify on the per-shape row rather than the headline? A neutral classification here would be expected even if the change is a real win.
- bf16-shape-gated-triton-gemms-small-t (queued) replaces the qkv GEMM at m=1. If it lands first, its output must stay a [1, 2304] bf16 tensor with stride(1)=1, and the gate must pass its actual stride(0). The two changes compose because they touch disjoint kernels on the same shape.
- The gate is batch == 1 only. muse_attention_decode later measured the fused kernel at 16 splits on decode_16 as +33% (sliding-fused-triton-decode-batched-deep-splits). Sliding decode_16 and decode_32 are a larger lever here (weight 46 each), and that should be a follow-up hypothesis once this one is proven.


## Implementation notes

Implemented 2026-10-24 in the worktree's `muse_decode_layer/submission.py` only (uncommitted). `git diff --stat` shows 327+/69-. `git diff -w --stat` shows 258+/0-: the 69 removed lines are master's attention block re-indented under `else:` and nothing else.

- **Ported verbatim** from `git show HEAD:muse_attention_decode/submission.py`, inserted before `prepare()`: `_INV_FREQ`/`_inv_freq` (attention lines 60-69) and `LOG2E`, `_COUNTERS`/`_counters`, `_num_splits`, `_norm_rope`, `_gpu_fence`, `_online_update`, `_gate_epilogue` and `_sliding_fused_kernel` (attention lines 142-340). A `diff` against those line ranges came back empty. The `_norm_rope` docstring still mentions `_pre_attention_kernel`, which exists only in the sibling. I left it because the copy is verbatim.
- **Adapted `_sliding_forward(data, qkv)`.** It takes the qkv GEMM output `[T, 2304]` directly instead of `data["qkv"].reshape`, passes its own `qkv.stride(0)` (2304), and allocates `out` as `[T, 1024]` bf16, the shape `_gemm(o, dense["o"])` expects, instead of `[B, L, 1024]`. Otherwise the body is the sibling's. At B=1, Hkv=2, `_num_splits` gives 16, so the grid is (1, 2, 16), with M=16, LN=16, BLOCK_N 64, num_warps 4, num_stages 2, ROPE=bool(rope_theta), EPS=RMS_EPS and PAGE=cfg["page_size"].
- **Added the gate in `kernel_fn`** right after `qkv = _gemm(...)`: `if cfg["window"] and batch == 1 and length == 1: o = _sliding_forward(data, qkv)`. Otherwise master's whole attention block runs unchanged under `else:`, from FusedQKNorm through `state["o_norm_gate"]`. Both paths rejoin at `attn_output = _gemm(o, dense["o"])`. The gate reads config only.
- **Module docstring:** added one bullet under "Differs from SGLang".
- **Indexing self-review, done twice:**
  - Offsets: q at head*64, k at (16+g)*64 = 1024+64g, v at (18+g)*64 = 1152+64g, gate at 1280+64h. These match master's slices `v = qkv[:, 1152:1280]` and `gate = qkv[:, 1280:]`.
  - Cache rows: `(loc*2+g)*64+d` on the uint8 view, the same as master's `k_cache.view(-1, 128)` at `cache_loc`.
  - Positions: `positions[tok]` with tok = b*L + j.
  - Window: old keys [seq-2047, seq-1] plus the new key, which equals the reference's `first = seq - window + 1` and FA3's (2047, 0).
  - Scales: q is multiplied by qk_scale_factor and then rounded to bf16, as the reference does. Scores use sm_scale*log2e*k_scale with exp2.
  - inv_freq: same formula as the reference's `_rope` and as prepare()'s cos_sin table.
  - k/v scale: the kernel always divides by k_scale/v_scale, while master divides only when kv_scale != 1. The two are identical at scale 1, and the kv_scale test is q_len 8, so it never reaches this path.
- **No deviations from the spec.**
- **Open question answered:** muse_attention_decode's TASK test list does contain `batch 1, ctx 10, heads tp8, layer sliding` (line 57), so this exact small-ctx path (splits 1..15 empty, masked to -inf) is already proven there.
- **bf16-shape-gated-triton-gemms-small-t:** not landed, so qkv still comes from `_bf16_gemm_dispatch_impl` (contiguous, stride(1)=1). The gate passes the real stride(0).
- **Assumption:** `page_table.stride(1) == 1`. The kernel indexes `pt_ptr + b*pt_stride + t//PAGE`, the same assumption the sibling makes, and the input is a freshly generated int32 [B, max_pages].

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/sliding-fused-triton-decode1-port diff` (and `diff -w`).

- [x] CUDA-graph capturable. The new path has no `.item()`, `.cpu()`, `synchronize()` or tensor prints; a grep of the added lines finds none. It allocates only `torch.empty` for out and workspaces, of sizes fixed by config. `_counters` does a one-time `torch.zeros` keyed by (B, Hkv, device). Control flow reads `cfg` only.
- [x] Nothing derived from `data` is cached across calls. `_INV_FREQ` is keyed on (rope_theta, head_dim, device), which is config. `_COUNTERS` holds ticket counters that every launch returns to 0, so they carry no data.
- [x] Weights: n/a, no change. prepare() and the weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16. The qkv input and the [T, 1024] output are bf16, as before. The KV cache entries are stored as E4M3. Queries are bf16 operands, finer than FA3's E4M3. Internals are fp32.
- [x] Attention sees every key. All 2047 old keys plus the new key on every call, split 16 ways with no skipping. Masked or empty splits carry weight 0 in the combine, which is not value skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds only Triton kernels and a Python launcher, and no new imports.
- [x] All work happens inside the timed call, with no output caching or replay. The K/V store, attention and epilogue all run in kernel_fn on each call's fresh inputs.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept. No `triton.autotune`.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L530: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_decode_layer/sliding-fused-triton-decode1-port/muse_decode_layer/submission.py
```

That warning is master's existing `torch.cuda.synchronize()` at the end of prepare(). `python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Implementation notes

Implemented 2026-10-24 in the worktree's `muse_decode_layer/submission.py` only (uncommitted). `git diff --stat` shows 327+/69-. `git diff -w --stat` shows 258+/0-: the 69 removed lines are master's attention block re-indented under `else:` and nothing else.

- **Ported verbatim** from `git show HEAD:muse_attention_decode/submission.py`, inserted before `prepare()`: `_INV_FREQ`/`_inv_freq` (attention lines 60-69) and `LOG2E`, `_COUNTERS`/`_counters`, `_num_splits`, `_norm_rope`, `_gpu_fence`, `_online_update`, `_gate_epilogue` and `_sliding_fused_kernel` (attention lines 142-340). A `diff` against those line ranges came back empty. The `_norm_rope` docstring still mentions `_pre_attention_kernel`, which exists only in the sibling. I left it because the copy is verbatim.
- **Adapted `_sliding_forward(data, qkv)`.** It takes the qkv GEMM output `[T, 2304]` directly instead of `data["qkv"].reshape`, passes its own `qkv.stride(0)` (2304), and allocates `out` as `[T, 1024]` bf16, the shape `_gemm(o, dense["o"])` expects, instead of `[B, L, 1024]`. Otherwise the body is the sibling's. At B=1, Hkv=2, `_num_splits` gives 16, so the grid is (1, 2, 16), with M=16, LN=16, BLOCK_N 64, num_warps 4, num_stages 2, ROPE=bool(rope_theta), EPS=RMS_EPS and PAGE=cfg["page_size"].
- **Added the gate in `kernel_fn`** right after `qkv = _gemm(...)`: `if cfg["window"] and batch == 1 and length == 1: o = _sliding_forward(data, qkv)`. Otherwise master's whole attention block runs unchanged under `else:`, from FusedQKNorm through `state["o_norm_gate"]`. Both paths rejoin at `attn_output = _gemm(o, dense["o"])`. The gate reads config only.
- **Module docstring:** added one bullet under "Differs from SGLang".
- **Indexing self-review, done twice:**
  - Offsets: q at head*64, k at (16+g)*64 = 1024+64g, v at (18+g)*64 = 1152+64g, gate at 1280+64h. These match master's slices `v = qkv[:, 1152:1280]` and `gate = qkv[:, 1280:]`.
  - Cache rows: `(loc*2+g)*64+d` on the uint8 view, the same as master's `k_cache.view(-1, 128)` at `cache_loc`.
  - Positions: `positions[tok]` with tok = b*L + j.
  - Window: old keys [seq-2047, seq-1] plus the new key, which equals the reference's `first = seq - window + 1` and FA3's (2047, 0).
  - Scales: q is multiplied by qk_scale_factor and then rounded to bf16, as the reference does. Scores use sm_scale*log2e*k_scale with exp2.
  - inv_freq: same formula as the reference's `_rope` and as prepare()'s cos_sin table.
  - k/v scale: the kernel always divides by k_scale/v_scale, while master divides only when kv_scale != 1. The two are identical at scale 1, and the kv_scale test is q_len 8, so it never reaches this path.
- **No deviations from the spec.**
- **Open question answered:** muse_attention_decode's TASK test list does contain `batch 1, ctx 10, heads tp8, layer sliding` (its TASK.md line 57), so this exact small-ctx path (splits 1..15 empty, masked to -inf) is already proven there.
- **bf16-shape-gated-triton-gemms-small-t:** not landed, so qkv still comes from `_bf16_gemm_dispatch_impl` (contiguous, stride(1)=1). The gate passes the real stride(0).
- **Assumption:** `page_table.stride(1) == 1`. The kernel indexes `pt_ptr + b*pt_stride + t//PAGE`, the same assumption the sibling makes.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/sliding-fused-triton-decode1-port diff` (and `diff -w`).

- [x] CUDA-graph capturable. The new path has no `.item()`, `.cpu()`, `synchronize()` or tensor prints; a grep of the added lines finds none. It allocates only `torch.empty` for out and workspaces, of sizes fixed by config. `_counters` does a one-time `torch.zeros` keyed by (B, Hkv, device). Control flow reads `cfg` only.
- [x] Nothing derived from `data` is cached across calls. `_INV_FREQ` is keyed on (rope_theta, head_dim, device), which is config. `_COUNTERS` holds ticket counters that every launch returns to 0, so they carry no data.
- [x] Weights: n/a, no change. prepare() and the weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16. The qkv input and the [T, 1024] output are bf16, as before. The KV cache entries are stored as E4M3. Queries are bf16 operands, finer than FA3's E4M3. Internals are fp32.
- [x] Attention sees every key. All 2047 old keys plus the new key on every call, split 16 ways with no skipping. Masked or empty splits carry weight 0 in the combine, which is not value skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds only Triton kernels and a Python launcher, and no new imports.
- [x] All work happens inside the timed call, with no output caching or replay. The K/V store, attention and epilogue all run in kernel_fn on each call's fresh inputs.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept. No `triton.autotune`.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L530: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_decode_layer/sliding-fused-triton-decode1-port/muse_decode_layer/submission.py
```

That warning is master's existing `torch.cuda.synchronize()` at the end of prepare(). `python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.
