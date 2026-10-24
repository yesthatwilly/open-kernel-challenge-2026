# muse_decode_layer · 2026-10-24T124616Z · check · fused-preattention-triton

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T075400Z-fused-preattention-triton.md` (branch `hyp/muse_decode_layer/fused-preattention-triton`).

- **Job:** `c7c26e2fd8b142a7` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `fc6abac`

## Goal
Port muse_attention_decode's fused pre-attention Triton kernel (main tree muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x there) into kernel_fn, replacing on every shape the five launches between the qkv GEMM and flash_attn_with_kvcache: state['qk_norm'](...) (FusedQKNorm, with RoPE on sliding layers), the optional k/v divide by the cache scales, k.to(FP8), v.to(FP8), store_cache(...) and q.view(T,16,64).to(FP8). One launch, grid (T, 16+2), num_warps=1, one program per (token, head): heads 0-15 RMS-normalize (scaleless, eps 1e-5) the q head read from the qkv row (stride qkv.stride(0)), apply interleaved RoPE when cfg['rope_theta'] is set and store FP8 into a fresh q buffer [T, 16, 64] float8_e4m3fn; heads 16-17 normalize (+RoPE) the k head, store (k / k_scale) as FP8 into k_cache.view(-1, 128) row cache_loc[t] at column (h-16)*64, and store the matching v head (v / v_scale, no norm or RoPE) into v_cache at the same row. One deliberate difference from the sibling: RoPE cos/sin are read from master's state['cos_sin'] table (row positions[t], cos in columns 0-31, sin in 32-63) instead of recomputed in-kernel, so the rotation is the table SGLang's FusedQKNorm uses. The FP8 q goes to flash_attn_with_kvcache unchanged otherwise; gate and v are still slices of qkv. FA3, the output norm-gate and everything downstream stay as on master.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/fused-preattention-triton`, file `muse_decode_layer/submission.py` only (uncommitted; diff stat `1 file changed, 83 insertions(+), 21 deletions(-)`).

- New `@triton.jit _pre_attention_kernel` (placed after `prepare()`), ported from `muse_attention_decode/hypotheses/done/gain/fused-preattention-triton.patch`. Grid `(T, hq + hkv)` = `(T, 18)`, `num_warps=1`, one program per (token, head). Heads 0-15: scaleless RMSNorm (eps `RMS_EPS` = 1e-5) of the q head read from the qkv row at `t * qkv.stride(0) + h*64`, optional interleaved RoPE, stored as `tl.float8e4nv` into a fresh `q = torch.empty((T, 16, 64), FP8)`. Heads 16-17: same norm (+RoPE) on the k head, stored as `(k / k_scale)` FP8 into the cache at flat offset `(cache_loc[t] * 2 + kh) * 64`, plus the matching v head (no norm/RoPE) as `(v / v_scale)` FP8 into `v_cache` at the same offset.
- Spec deviation from the sibling (as the spec requires): RoPE reads `state["cos_sin"]` (row `positions[t]`, cos columns 0-31, sin 32-63, row stride D=64) instead of computing `inv_freq`/`tl.cos`/`tl.sin` in-kernel. `ROPE` is a constexpr from `bool(cfg["rope_theta"])`, so the table pointer is passed but unread on full layers.
- kernel_fn: removed `state["qk_norm"](...)`, the `v` slice, the `k.reshape`, the `if scaled:` k/v divides, the k/v `.to(FP8)` casts and `store_cache(...)`; the unpaired branch now passes the kernel's FP8 `q` straight to FA3 (the `q.view(...).to(FP8)` cast is gone). The k/v divide now always runs in-kernel (dividing by 1.0 on scored shapes, exact). `scaled` is kept: it still drives FA3's `k_descale`/`v_descale`.
- Paired branch (full layers, batch>1), per the brief and the review's open question: master's `torch.zeros` + two strided pad copies are kept exactly as they were; they now read the kernel's FP8 `q` [T,16,64] (FP8 -> FP8 copies instead of BF16 -> FP8). No in-kernel paired write. Comment updated to say so.
- prepare(): removed the `"qk_norm": FusedQKNorm(...)` entry and the `FusedQKNorm` / `store_cache` imports (now unused); added `import triton`, `import triton.language as tl`. `state["cos_sin"]` is unchanged. The module docstring's pipeline table is updated for the new launch.
- Open question "same RoPE convention as FusedQKNorm": sglang's `fused_qk_norm` source is not on this node, so it could not be read directly. Indirect evidence: the reference `_rope` is interleaved (pairs 2i, 2i+1) with `inv_freq = theta^(-2i/64)` and angle `pos * inv_freq`; `prepare()` builds `cos_sin` as `cat(cos(pos*inv_freq), sin(pos*inv_freq))` with the same inv_freq; master (FusedQKNorm + this table) passes the sliding tests; baseline.py documents it as "interleaved RoPE". So the kernel's pairing `(2i, 2i+1)` with `cos[i]`, `sin[i]` matches the reference whatever FusedQKNorm does internally. A mismatch would fail the sliding tests loudly.
- Numerics: q and k go fp32 -> FP8 directly (master: fp32 -> bf16 -> FP8), allowed ("queries may be rounded to FP8"); v is bf16 -> fp32 / v_scale -> FP8.
- Not verifiable here (no GPU/Triton): compile, correctness, and the verify_32 (T=256, 4,608 programs) regression check the review asks for; read those from the run and its profile. Count the removed kernel rows in the decode_1 profile (FusedQKNorm, casts, store_cache, q cast) for LESSONS.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/fused-preattention-triton diff`:

- [x] CUDA-graph capturable: the new launch has no `.item()`/`.cpu()`/`synchronize()`/tensor prints; scales, positions and cache_loc are read in-kernel as device pointers; the grid and constexprs come from shapes and `config` only (`tokens`, `hq`, `hkv`, `d`, `cfg["rope_theta"]`). The one `torch.empty` for q has a shape-only size, as master's allocations do. No `triton.autotune`.
- [x] Nothing derived from `data` is cached across calls: q is a fresh per-call buffer; `state` gains nothing (it loses `qk_norm`); `cos_sin` is built in `prepare()` from constants only.
- [x] Weights: n/a: untouched; no new weight repack.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (same rows/offsets as store_cache wrote); queries FP8 E4M3 (fp32 -> E4M3 directly), not coarser. Gate, attention output and everything downstream unchanged.
- [x] Attention sees every key: FA3 call, window and page table unchanged; nothing skipped.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: only `triton`/`triton.language` imports added; two sglang imports removed.
- [x] All work happens inside the timed call: the norm, RoPE, casts and cache writes all run in kernel_fn each call; no output caching or replay.
- [x] First line still `# okc: problem=muse_decode_layer`.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L249: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/fused-preattention-triton/muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed; the `__pycache__` it left was removed.

## Risks / what could fail
medium
This is the superset of cache-store-fused-fp8-cast (3 of 4 cast/store launches removed, came back flat at 0.996/1.000 on decode_1): it additionally folds FusedQKNorm, the launch the earlier run left in, which is what the muse_attention_decode sibling fused to get its 3.5-6.4 us. Built on master, not on that neutral branch (the fused kernel replaces its code entirely). Expected per-shape gains (~2-3%) sit near the 1.5% drift, so read the T=1 and T=8 rows individually. Correctness: q is rounded fp32 -> FP8 directly instead of fp32 -> bf16 -> FP8 (allowed: queries may be rounded to FP8); k likewise; the kv_scale 0.015625 test and page-64 tests exercise the scale divide and the flat-row addressing (loc * 2 + kh) * 64. If triton-flash-decode-full-paired-kv lands first, its q input is this kernel's FP8 q buffer.

## Result (after the run)
- job `c7c26e2fd8b142a7` (check), status done, passed True, tests 13/13
- speedup 1.0970x, 95% CI [1.0806, 1.1135], score 20619.694652128608 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.0873503247934666, "ci": [1.0710400699215645, 1.1036605796653685], "job": "fb07a3bd3aa54baf"}; base result: muse_decode_layer/hypotheses/done/neutral/fa3-hdim128-head-pairing-batched.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.result.json`, `muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.patch`; branch `hyp/muse_decode_layer/fused-preattention-triton`
