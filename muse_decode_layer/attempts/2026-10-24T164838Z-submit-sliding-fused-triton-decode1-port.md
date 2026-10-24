# muse_decode_layer · 2026-10-24T164838Z · submit · sliding-fused-triton-decode1-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T140135Z-sliding-fused-triton-decode1-port.md` (branch `hyp/muse_decode_layer/sliding-fused-triton-decode1-port`).

- **Job:** `12198572363b49a4` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `2569d7e`

## Goal
Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE = bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed), data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'], data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula, as in the attention problem (decode's cos_sin table stays for the other shapes).

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
Single-shape gain (~+0.4-0.6% geomean) but on the heaviest-weighted shape of a launch-bound load, and the kernel is proven correct and faster on identical inputs in the sibling. Caveat: decode's fused-preattention-triton (neutral) showed that fusing pre-attention launches alone did not move sliding decode_1 here, so the gain may be only the attention+norm-gate part (~3.5 us, +2%). Correctness risk is in the I/O mapping: decode's qkv is [T, 2304] contiguous from the GEMM (attention's was [B, L, 2304]); the fused kernel's RoPE must match FusedQKNorm's interleaved convention (it passed the attention problem's tests with the same SGLang reference). The 13 tests include batch 1 ctx 10 sliding (q_len 1), which takes this path at small ctx and its page size. Composes with post-expert-norm-triton-small-t and the bf16 GEMM port (disjoint kernels on the same shape).

## Result (after the run)
- job `12198572363b49a4` (submit), status done, passed True, tests 13/13
- speedup 1.1690x, 95% CI [1.1515, 1.1866], score 19349.38768103388 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.1664628806987323, "ci": [1.1489659374882513, 1.183959823909213], "job": "a017e10aec544c55"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/sliding-fused-triton-decode1-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/sliding-fused-triton-decode1-port.patch`; branch `hyp/muse_decode_layer/sliding-fused-triton-decode1-port`
