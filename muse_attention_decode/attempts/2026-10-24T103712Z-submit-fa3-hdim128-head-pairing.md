# muse_attention_decode · 2026-10-24T103712Z · submit · fa3-hdim128-head-pairing

Hypothesis: `muse_attention_decode/hypotheses/done/gain/2026-10-23T225840Z-fa3-hdim128-head-pairing.md` (branch `hyp/muse_attention_decode/fa3-hdim128-head-pairing`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `7f00f9c`

## Goal
Promote check-proven gain fa3-hdim128-head-pairing (1.1706x [1.1530,1.1924], check job facc479992e24a35) to the board.

On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV heads are fetched as ONE 128-byte row instead of two 64-byte rows. With G = Hq // Hkv and S = Hkv // 2 (Hkv is even: 2 under tp8, 16 under dp; if odd, fall back to the unpaired call): k_cache.view(pages, page_size, S, 128) and v_cache likewise (contiguous views: kv heads 2s and 2s+1 of a token are adjacent 64-byte halves of the same row). Queries: q_pad = torch.zeros((tokens, Hq, 128), dtype=float8_e4m3fn) each call (fixed shape, no value dependence); view q_pad as [tokens, S, 2, G, 128] and the parity pipeline's FP8 q as [tokens, S, 2, G, 64]; q_pad[:, :, 0, :, :64] = q[:, :, 0] and q_pad[:, :, 1, :, 64:] = q[:, :, 1]. Because FA3 maps query head h to kv head h // (2G), query heads s*2G .. s*2G+2G-1 land on super-head s, the first G in the low half (original kv head 2s) and the next G in the high half (2s+1), which is exactly the original grouping; the zero half makes q_pad . [k_2s | k_2s+1] equal q . k_g exactly, so the softmax over keys is unchanged and FA3's output out[tokens, Hq, 128] holds the correct 64-wide result in the same half as the query (the other half is the other head's P@V and is discarded). Un-pad: attn = torch.empty((tokens, Hq, 64), bf16); a = attn.view(tokens, S, 2, G, 64); o = out.view(tokens, S, 2, G, 128); a[:, :, 0] = o[:, :, 0, :, :64]; a[:, :, 1] = o[:, :, 1, :, 64:]; then rms_norm_gate on attn.reshape(tokens, Hq*64) exactly as before. FA3 arguments otherwise identical (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(-1,-1)); descale when kv_scale != 1 becomes scale.reshape(1,1).expand(B, S). The judge's scheduler_metadata was built for (Hq, Hkv=2, headdim 64) and must NOT be passed with the paired geometry: on full decode steps rebuild it in kernel_fn with sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=page_table.shape[1]*page_size (a shape, never cache_seqlens.max()), num_heads=Hq, num_heads_k=S, headdim=128, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) (device side, one small launch); on verify blocks pass None as before. Sliding layers keep the unpaired FA3 call untouched. The pre-attention stages and the cache store are untouched.

## What changed, and where
Worktree `.worktrees/muse_attention_decode/fa3-hdim128-head-pairing`. Only `kernel_fn` in `muse_attention_decode/submission.py` changed, and only the FA3 call region. The pre-attention Triton kernel, the cache store and `rms_norm_gate` are untouched. Diff stat: 1 file, +38/-6.

- `paired = window == 0 and hkv % 2 == 0` is a config-only branch. Sliding layers and odd Hkv keep the original unpaired call byte for byte, and still pass the judge's `scheduler_metadata`.
- Paired path: `k_cache` and `v_cache` are viewed as `[pages, page_size, S, 128]` (contiguous views of the same storage, read in place). `q_attn = torch.zeros((tokens, Hq, 128), FP8)` is allocated each call. The FP8 q from the pre-attention kernel, viewed as `[tokens, S, 2, G, 64]`, is copied into the low half (parity 0) and high half (parity 1) of `q_attn.view(tokens, S, 2, G, 128)`.
- When kv_scale != 1, the descale expands to `(B, S)` instead of `(B, Hkv)`.
- After FA3, `attn = torch.empty((tokens, Hq, 64), bf16)` gets each query's own half of `out.view(tokens, S, 2, G, 128)`. That feeds the same `rms_norm_gate` call as before. The existing `.to(torch.bfloat16)` is kept and does nothing here.
- **Deviation (following the reviewer):** on full decode the spec rebuilds `scheduler_metadata` with `get_scheduler_metadata(...)`. The adversarial review recommends passing `None` on decode as on verify, because it is functionally the same: FA3 runs prepare_varlen_num_blocks internally with the same page-table-derived max_seqlen_k. I pass `metadata = None` on every paired call, so there is no rebuild code and no extra launch. The judge's metadata is never used with the paired geometry.
- Not verified locally, since this node has no GPU, torch or sgl_kernel. Open items:
  - whether FA3 in the image has e4m3 hdim-128 paged forward with pack_gqa at 16 q heads per kv head;
  - whether `torch.zeros` and slice `copy_` on float8_e4m3fn CUDA tensors work in the image's torch (both are supported in recent PyTorch).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/fa3-hdim128-head-pairing diff`.

- [x] CUDA-graph capturable: no `.item()`, `.cpu()`, `synchronize()` or tensor prints were added. The new `zeros`/`empty` have fixed shapes from config (tokens, hq, d). The branch is on `cfg["window"]` and `num_kv_heads` only. No `cache_seqlens.max()` is used.
- [x] Nothing derived from `data` is cached across calls. `q_attn`, `attn` and the cache views are rebuilt every call, and no new module-level state was added.
- [x] Weights: n/a. This problem has no weights, and nothing is repacked.
- [x] Activations BF16 where SGLang keeps BF16; the KV cache stays FP8 and is read in place through views; queries stay FP8 E4M3 with the same values, zero-padded; the un-padded attention output is BF16 into the same norm-gate.
- [x] Attention sees every key: causal, window_size=(-1,-1) on full layers and the same page_table/cache_seqlens, so the whole context is attended. There is no top-k, sparse, block-skip or weight-based skip. The discarded half is the other head's P@V, not skipped keys.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. No new imports.
- [x] All work happens inside the timed call (pad, FA3, un-pad, all in `kernel_fn`), with no output caching or replay.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept. No `triton.autotune` was added.

okc_lint output:

```
CLEAN: muse_attention_decode/submission.py
```

`python3 -I -m py_compile` OK. The resulting `__pycache__` was removed from the worktree.

## Risks / what could fail
medium

## Result (after the run)
(pending)
