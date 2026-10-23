# muse_attention_decode · 2026-10-23T2111Z · check · fused-preattention-triton

*Backfilled 2026-10-23T2135Z from `~/.config/okc/jobs.json` and the working-tree diff; no
explanation was written before this run.*

- **Job:** `dfee6f2939974bad`
- **File:** `muse_attention_decode/submission.py`, uncommitted (diff against `ab21d70`)

## Goal
On sliding-window decode the attention itself moves under a megabyte, so SGLang's five
pre-attention launches (fused QK-norm + RoPE, k→FP8, v→FP8, cache store, q→FP8) dominate the
step. Collapse them into one Triton launch and keep FA3 and SGLang's `rms_norm_gate` for the
rest, aiming at the sliding-layer shapes (weight 46 each, floors 0.1–3.7 µs).

## What changed, and where
- Replaced the plain-PyTorch `kernel_fn` with: one Triton kernel `_pre_attention_kernel`,
  grid `(tokens, Hq + Hkv)`, one program per (token, head).
- Query heads: scaleless RMSNorm, interleaved RoPE when `rope_theta` is set, store as FP8
  E4M3 into a fresh `[tokens, Hq, D]` buffer for FA3. The 5.473 factor is folded into
  `softmax_scale` instead of being applied to q.
- Key heads: same norm + RoPE, divided by `k_scale`, stored as FP8 straight into `k_cache`
  at `cache_loc[t]`; the same program also stores the value head (scale + cast, no norm).
- Attention: `sgl_kernel.flash_attn.flash_attn_with_kvcache` with `page_table`,
  `cache_seqlens`, `cu_seqlens_q`, `causal=True`, `window_size=(window-1, 0)` on sliding
  layers, `scheduler_metadata` passed through, k/v descale only when `kv_scale != 1`.
- Output: `rms_norm_gate(out, gate, has_norm=True)` from the SGLang muse_spark_v12 ops.
- `_INV_FREQ`: a per-(theta, head_dim, device) cache of the RoPE inverse frequencies.

## Rules check (reconstructed)
- [x] Graph-capturable: no host syncs; `torch.empty` for q is a fixed-shape allocation
      (shape-only). `_INV_FREQ` is keyed on `config`, not on `data`.
- [x] Nothing derived from `data` is cached (`_INV_FREQ` derives from `config["rope_theta"]`).
- [x] No weights on this problem.
- [x] Queries rounded to FP8 E4M3 (allowed, "as FA3 does"); cache stays FP8; output BF16.
- [x] Full window / full context via FA3's `window_size` and `cache_seqlens`.
- [x] No harness access, subprocesses or network.
- [x] All work inside the timed call.

## Risks / what could fail
- `_INV_FREQ` is filled lazily on first call; if the judge captures the graph on the very
  first call, that allocation happens under capture. Pre-building it per theta outside
  `kernel_fn` (or in a warm-up path) would remove the doubt.
- `num_warps=1` with `HALF = D//2 = 32` lanes: fine for D=64, revisit if head_dim changes.
- FA3 with `page_size=64` tests and the 1,048,576-token test: correct but the win is only on
  the small sliding shapes; full-layer shapes should land near SGLang's time.

## Result (after the run)
Job was still queued when this file was written; see `okc status dfee6f2939974bad`.
