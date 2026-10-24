# muse_decode_layer · 2026-10-24T104412Z · check · fa3-hdim128-head-pairing-batched

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T103909Z-fa3-hdim128-head-pairing-batched.md` (branch `hyp/muse_decode_layer/fa3-hdim128-head-pairing-batched`).

- **Job:** `fb07a3bd3aa54baf` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `b694721`

## Goal
Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attention layers with batch > 1. Gate, config only: paired = cfg["window"] == 0 and cfg["batch"] > 1 and cfg["num_kv_heads"] % 2 == 0 (never a tensor read; hkv is 2 under tp8). When not paired (all sliding layers, decode_1 and verify_1 full, odd hkv) the existing call stays byte for byte, including q.view(tokens, hq, d).to(FP8) and the judge's data["scheduler_metadata"]. Paired path, with G = hq // hkv and S = hkv // 2: k_attn = k_cache.view(pages, page_size, S, 2*d) and v_attn = v_cache.view(...) (contiguous in-place views of the FP8 cache AFTER store_cache has written the step's rows; works for any page_size, the tests use page sizes other than 1). Queries: in decode_layer q comes out of state["qk_norm"] as BF16 [tokens, hq*d] and is cast to FP8 inline in the FA3 call, so fold that cast into the pad: q_attn = torch.zeros((tokens, hq, 2*d), dtype=FP8, device); qp = q_attn.view(tokens, S, 2, G, 2*d); qv = q.view(tokens, S, 2, G, d); qp[:, :, 0, :, :d] = qv[:, :, 0]; qp[:, :, 1, :, d:] = qv[:, :, 1] (copy_ does the BF16->E4M3 cast with the same round-to-nearest as .to(FP8), so the FP8 values are identical; the separate cast launch disappears). FA3 then maps query head h to super-head h // (2G): heads 0..G-1 sit in the low half (kv head 2s) and G..2G-1 in the high half (2s+1), the zero half adds exactly 0 to every logit. descale when scaled: expand(batch, S) instead of (batch, hkv). scheduler_metadata=None on the paired call (the judge's metadata was built for hdim 64 / Hkv 2 and must not be reused; attention_decode's run passed None and FA3 built its own schedule). softmax_scale passed explicitly as today, cfg["sm_scale"] * cfg["qk_scale_factor"] (it must not default to 1/sqrt(128)). All other FA3 args unchanged (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=length, causal=True, window_size=(-1,-1)). Un-pad: attn = torch.empty((tokens, hq, d), bf16); a = attn.view(tokens, S, 2, G, d); o = out.view(tokens, S, 2, G, 2*d); a[:, :, 0] = o[:, :, 0, :, :d]; a[:, :, 1] = o[:, :, 1, :, d:]; then state["o_norm_gate"](attn.reshape(tokens, hq*d), gate) exactly as before. QKV GEMM, qk_norm/rope, store_cache, o GEMM and the whole MoE block are untouched. No autotune, no host sync, no new imports.

## What changed, and where
All edits are in `muse_decode_layer/submission.py` `kernel_fn`, around the single `flash_attn_with_kvcache` call between `store_cache` and `state["o_norm_gate"]` (39 insertions, 6 deletions). This is a port of `muse_attention_decode/hypotheses/done/gain/fa3-hdim128-head-pairing.patch` with the batch gate added.

- Gate, config only: `paired = window == 0 and batch > 1 and hkv % 2 == 0`, where `batch` is `cfg["batch"]` and `window` is `cfg["window"]`.
- Paired path: `g, s = hq // hkv, hkv // 2`. K and V become in-place views of the FP8 cache, `k_cache.view(pages, page_size, s, 2*d)` and likewise for V. These are taken after `store_cache` and work for any page_size.
  - `q_attn = torch.zeros((tokens, hq, 2*d), FP8, device=q.device)`, filled by two strided copies from the BF16 `q` view `(tokens, s, 2, g, d)`. These copies also do the BF16 to E4M3 cast, so there is no separate `.to(FP8)`.
  - `attn_heads = s`, and `scheduler_metadata=None`.
- Descale expands to `(batch, attn_heads)`. It is still `(batch, hkv)` on the unpaired path.
- `softmax_scale` stays explicit as `cfg["sm_scale"] * cfg["qk_scale_factor"]`. page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, causal and window_size are unchanged.
- Un-pad: `attn = torch.empty((tokens, hq, d), bf16)` is filled by two strided copies, each taking a query's own half of FA3's `(tokens, hq, 2d)` output. Then `out = attn` goes into the unchanged `state["o_norm_gate"](out.reshape(tokens, hq*d).to(torch.bfloat16), gate)`. The `.to(bf16)` is a no-op on the paired path.
- Unpaired path (all sliding layers, full layers at B=1, odd hkv): the same ops with the same arguments. These are `q.view(tokens, hq, d).to(FP8)`, `k_cache`, `v_cache`, `data["scheduler_metadata"]` and descale `(batch, hkv)`.
  - Deviation (textual only): the FP8 cast is now written in the `else` branch as `q_attn = ...` instead of inline in the call's argument list. It is the same single cast kernel and still the first launch after `store_cache`, because building the descale views launches nothing. The kernel sequence is the same as master.
- The QKV GEMM, qk_norm/rope, store_cache, o GEMM, router and MoE are untouched. There are no new imports, no autotune, no host syncs, and nothing is cached in `state`. Local names `g`, `s`, `a`, `pages` and `page_size` do not collide with later code (`o` is reassigned by `o_norm_gate` afterwards, as before).
- Could not be run here (no GPU). Unverified on this problem: the paired hdim-128 path with page_size > 1 at batch > 1. That combination is exercised by the test `batch 3, ctx 6000, layer full, q_len 8, dist outliers` (default page size). attention_decode passed the same geometry.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/fa3-hdim128-head-pairing-batched diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The `paired` branch reads only `cfg` and tensor `.shape`, never tensor values. The `zeros`/`empty` allocations have shape-only sizes and are made in the caching allocator during capture, as master's `.to(FP8)` already does.
- [x] Nothing derived from `data` is cached across calls: q_attn, attn and the cache views are per-call locals, and `state` is not written.
- [x] Weights: n/a. prepare() and the weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16: the attention output is un-padded into BF16 as before. The KV cache stays FP8 and is read in place through a view. Queries are FP8 E4M3 with the same round-to-nearest values as `.to(FP8)`, and the zero pad is exact.
- [x] Attention sees every key: still `causal=True`, `window_size=(-1, -1)` on full layers, and the full `cache_seqlens`/page_table. The sliding call is unchanged. There is no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds no imports and no such calls.
- [x] All work happens inside the timed call: padding, FA3 and un-padding all run in kernel_fn, with no output caching or replay.
- [x] First line is still `# okc: problem=muse_decode_layer`. There is no triton.autotune.

okc_lint output:

```
warn  host-sync    L247: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/fa3-hdim128-head-pairing-batched/muse_decode_layer/submission.py
```

The warning is the existing synchronize in prepare() and is not part of this diff. `python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Risks / what could fail
medium
Medium risk, but the mechanism is already proven on this exact FA3 build and cache layout (attention_decode, job facc479992e24a35, 16/16 tests incl. kv_scale and odd page sizes): same hq=16, hkv=2, d=64 FP8 e4m3 cache, so the hdim-128 e4m3 paged instantiation and pack_gqa at 16 q heads per super-head are known to work. Differences from attention_decode: (1) q arrives as BF16 from FusedQKNorm rather than FP8 from a fused pre-attention kernel, so the pad copies also do the cast (net extra launches: zeros + 2 un-pad copies, ~1 us against 30-75 us saved); (2) the full layer here is dist=soft rather than coherent, which changes softmax values but not the gather pattern; (3) FA3 is a smaller fraction of the shape (36-60%), so the per-load gain is diluted: estimated ~1.5-3% per batch-16/32 load on the geomean (~+2% headline, 4 of 6 loads), small enough that the harvester must read the FA3 kernel row and per-shape full rows, not the headline. The batch gate costs decode_1/verify_1 nothing. Confounders: queued triton-flash-decode-full-* and fa3-num-splits-full-no-metadata touch the same call; dropping the judge metadata on paired calls also changes the split schedule (LESSONS: num_splits is only a cap under dynamic split, so expected small). Rules: every key attended (causal, full window), FP8 KV read in place, queries E4M3 with identical values, no host syncs, no autotune, no caching across calls.

## Result (after the run)
- job `fb07a3bd3aa54baf` (check), status done, passed True, tests 13/13
- speedup 1.0874x, 95% CI [1.0710, 1.1037], score 20803.139047479057 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.0634521974439426, "ci": [1.0475004144822835, 1.0794039804056017], "job": "2b1f82408ab64116"}; base result: muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/fa3-hdim128-head-pairing-batched.result.json`, `muse_decode_layer/hypotheses/done/neutral/fa3-hdim128-head-pairing-batched.patch`; branch `hyp/muse_decode_layer/fa3-hdim128-head-pairing-batched`
