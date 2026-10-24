---
problem: muse_attention_decode
slug: fa3-hdim128-head-pairing
created: 2026-10-23T225840Z
base: master
area: FA3 call (head packing)
change: >
  On full-attention layers only (config window == 0, decode and verify) call
  flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV heads are
  fetched as ONE 128-byte row instead of two 64-byte rows. With G = Hq // Hkv and S = Hkv // 2
  (Hkv is even: 2 under tp8, 16 under dp; if odd, fall back to the unpaired call):
  k_cache.view(pages, page_size, S, 128) and v_cache likewise (contiguous views: kv heads 2s and
  2s+1 of a token are adjacent 64-byte halves of the same row). Queries: q_pad =
  torch.zeros((tokens, Hq, 128), dtype=float8_e4m3fn) each call (fixed shape, no value
  dependence); view q_pad as [tokens, S, 2, G, 128] and the parity pipeline's FP8 q as [tokens, S,
  2, G, 64]; q_pad[:, :, 0, :, :64] = q[:, :, 0] and q_pad[:, :, 1, :, 64:] = q[:, :, 1]. Because
  FA3 maps query head h to kv head h // (2G), query heads s*2G .. s*2G+2G-1 land on super-head s,
  the first G in the low half (original kv head 2s) and the next G in the high half (2s+1), which
  is exactly the original grouping; the zero half makes q_pad . [k_2s | k_2s+1] equal q . k_g
  exactly, so the softmax over keys is unchanged and FA3's output out[tokens, Hq, 128] holds the
  correct 64-wide result in the same half as the query (the other half is the other head's P@V and
  is discarded). Un-pad: attn = torch.empty((tokens, Hq, 64), bf16); a = attn.view(tokens, S, 2,
  G, 64); o = out.view(tokens, S, 2, G, 128); a[:, :, 0] = o[:, :, 0, :, :64]; a[:, :, 1] = o[:,
  :, 1, :, 64:]; then rms_norm_gate on attn.reshape(tokens, Hq*64) exactly as before. FA3
  arguments otherwise identical (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=L,
  softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(-1,-1)); descale when kv_scale
  != 1 becomes scale.reshape(1,1).expand(B, S). The judge's scheduler_metadata was built for (Hq,
  Hkv=2, headdim 64) and must NOT be passed with the paired geometry: on full decode steps rebuild
  it in kernel_fn with sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=1,
  max_seqlen_k=page_table.shape[1]*page_size (a shape, never cache_seqlens.max()), num_heads=Hq,
  num_heads_k=S, headdim=128, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn,
  cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) (device side, one
  small launch); on verify blocks pass None as before. Sliding layers keep the unpaired FA3 call
  untouched. The pre-attention stages and the cache store are untouched.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "388 us vs 126 us floor at ~1.7 TB/s effective: FA3 with page_size 1 gathers each (token, kv head) as a separate 64-byte row; paired rows halve the number of random gathers, make every fetch a full 128-byte line, and double the bytes per pipeline stage of the hdim-128 FP8 kernel (16 KB K blocks instead of 8 KB) so each SM keeps twice as many bytes in flight"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 201 us vs 63 us floor, same mechanism; 16 tiles of (request, super-head) instead of 32
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: 390 us vs 127 us floor, same bytes; pack_gqa packs 2G x 8 = 128 query rows per super-head tile, one M tile
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: 205 us vs 63 us, same mechanism
  - shapes: "*layer=full load=decode_1 page*"
    direction: slower
    why: six small extra launches (zeros, two pad copies, two un-pad copies, the metadata rebuild) on a 30.8 us shape with only 20 MB of KV; the gather gain cannot pay for ~12 us of launches; the cleanup child removes them
  - shapes: "*layer=full load=verify_1 page*"
    direction: slower
    why: five extra launches on a 35.6 us shape, same reason
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control that the window gate is right
risk: medium
depends_on: fused-preattention-triton
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T092245Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T092246Z
base_sha: e8c54b5ac52496d46ca232957689f65e6c729c2a
base_blob: f14211b4b0e88f8b3f683d9f9275b76a64b66e02
branch: hyp/muse_attention_decode/fa3-hdim128-head-pairing
worktree: .worktrees/muse_attention_decode/fa3-hdim128-head-pairing
mode: check
implemented_at: 2026-10-24T092444Z
impl_commit: aaa550bdabc89c81bbdaa382ce4bb68430021afc
submission_sha256: eb6b0498ec6e6b03dbaee6c28df0f25db3807e50a13c3b103e6151ddd99a4eb2
norm_sha256: 6bcb7977f4733b3f2b1e5d56ec31e4538e74e2d0507b7f84794a5715b24564ce
submission_blob: bc007e643ac0b440147a732071f23d9b3563174c
lint_warnings: 0
attempt_file: 2026-10-24T092444Z-check-fa3-hdim128-head-pairing.md
prepared_at: 2026-10-24T092444Z
job: facc479992e24a35
dispatched_at: 2026-10-24T092453Z
outcome: gain
verdict_detail: gain
speedup: 1.1705932092473075
speedup_ci: [1.1530343111085979, 1.1923944446416679]
harvested_at: 2026-10-24T103710Z
best_at_harvest: 1.1284429079326423
promotion: merged
promotion_commit: 86dc7133cb6594c6738fddf31911079e71700bfc
---

# fa3-hdim128-head-pairing

## Change

On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV heads are fetched as ONE 128-byte row instead of two 64-byte rows. With G = Hq // Hkv and S = Hkv // 2 (Hkv is even: 2 under tp8, 16 under dp; if odd, fall back to the unpaired call): k_cache.view(pages, page_size, S, 128) and v_cache likewise (contiguous views: kv heads 2s and 2s+1 of a token are adjacent 64-byte halves of the same row). Queries: q_pad = torch.zeros((tokens, Hq, 128), dtype=float8_e4m3fn) each call (fixed shape, no value dependence); view q_pad as [tokens, S, 2, G, 128] and the parity pipeline's FP8 q as [tokens, S, 2, G, 64]; q_pad[:, :, 0, :, :64] = q[:, :, 0] and q_pad[:, :, 1, :, 64:] = q[:, :, 1]. Because FA3 maps query head h to kv head h // (2G), query heads s*2G .. s*2G+2G-1 land on super-head s, the first G in the low half (original kv head 2s) and the next G in the high half (2s+1), which is exactly the original grouping; the zero half makes q_pad . [k_2s | k_2s+1] equal q . k_g exactly, so the softmax over keys is unchanged and FA3's output out[tokens, Hq, 128] holds the correct 64-wide result in the same half as the query (the other half is the other head's P@V and is discarded). Un-pad: attn = torch.empty((tokens, Hq, 64), bf16); a = attn.view(tokens, S, 2, G, 64); o = out.view(tokens, S, 2, G, 128); a[:, :, 0] = o[:, :, 0, :, :64]; a[:, :, 1] = o[:, :, 1, :, 64:]; then rms_norm_gate on attn.reshape(tokens, Hq*64) exactly as before. FA3 arguments otherwise identical (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(-1,-1)); descale when kv_scale != 1 becomes scale.reshape(1,1).expand(B, S). The judge's scheduler_metadata was built for (Hq, Hkv=2, headdim 64) and must NOT be passed with the paired geometry: on full decode steps rebuild it in kernel_fn with sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=page_table.shape[1]*page_size (a shape, never cache_seqlens.max()), num_heads=Hq, num_heads_k=S, headdim=128, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) (device side, one small launch); on verify blocks pass None as before. Sliding layers keep the unpaired FA3 call untouched. The pre-attention stages and the cache store are untouched.

## Expected

- `*layer=full load=decode_32*` faster: 388 us vs 126 us floor at ~1.7 TB/s effective: FA3 with page_size 1 gathers each (token, kv head) as a separate 64-byte row; paired rows halve the number of random gathers, make every fetch a full 128-byte line, and double the bytes per pipeline stage of the hdim-128 FP8 kernel (16 KB K blocks instead of 8 KB) so each SM keeps twice as many bytes in flight
- `*layer=full load=decode_16*` faster: 201 us vs 63 us floor, same mechanism; 16 tiles of (request, super-head) instead of 32
- `*layer=full load=verify_32*` faster: 390 us vs 127 us floor, same bytes; pack_gqa packs 2G x 8 = 128 query rows per super-head tile, one M tile
- `*layer=full load=verify_16*` faster: 205 us vs 63 us, same mechanism
- `*layer=full load=decode_1 page*` slower: six small extra launches (zeros, two pad copies, two un-pad copies, the metadata rebuild) on a 30.8 us shape with only 20 MB of KV; the gather gain cannot pay for ~12 us of launches; the cleanup child removes them
- `*layer=full load=verify_1 page*` slower: five extra launches on a 35.6 us shape, same reason
- `*layer=sliding*` flat: untouched; control that the window gate is right

## Risk

Medium risk. Correctness: the zero half contributes exactly 0 to every logit, the FP8 rounding of q is the same values in the same E4M3 format, the descale is per super-head with the same scalar, so the output is bit-for-bit the same math as FA3 unpaired up to accumulation order; nothing is coarser than FP8 queries. Rule edge: every key is still attended; queries stay E4M3; the cache is read in place. Hazards: FA3 must accept FP8 hdim 128 with page_size 1 (the non-TMA paged path) and Hq / S = 2G = 16 query heads per kv head under tp8 (32 under dp: 128 heads / 8 super-heads is 16 too); if FA3 caps qhead_per_khead for pack_gqa, the dp tests fail and the hypothesis is failed cleanly. The judge metadata must not be reused (built for headdim 64 / Hkv 2): rebuilding it is one launch and also changes the split schedule, which confounds decode_16/32 with fa3-num-splits-full-decode; the reviewer may want an A/B that passes None on decode instead of rebuilding. Extra traffic: FA3 writes a [tokens, Hq, 128] bf16 output (double) and reads a [tokens, Hq, 128] q: negligible against 328-655 MB of KV. The batch-1 full shapes lose ~10-12 us from launches: with weight 16 against the sliding shape's 46 x 21 us, decode_1 load gets ~10% slower while decode_16/32 and verify_16/32 loads, dominated by the full shapes, decide the geometric mean.

## Adversarial review

Reviewed 2026-10-24T092245Z by an adversarial subagent (model opus).

**Verdict: proceed.** The idea is plausible and the profile can attribute it: full-layer FA3 at B=16/32 is 94-98% of the shape, about 3x its floor, and the measured 1.3-1.7 TB/s is blamed on the page-size-1 row gather. Pairing heads into one contiguous 128-byte row attacks that directly, and the FA3 kernel's own time in top_kernels separates the mechanism from the extra pad, un-pad and metadata launches, so no companion is needed.

### Justification
Plausible on H200. The cache is [pages, 1, Hkv=2, 64] FP8, so the two KV heads of a token sit in one aligned 128-byte line. Today two different (request, kv-head) tiles each fetch a 64-byte half of that line at different times, so each pays the per-row page_table indirection and cp.async issue cost. If L2 fills at more than 64 bytes on a miss, both tiles may also pull the whole line, which would double DRAM traffic. That would explain why the 'effective' rate counted on necessary bytes sits at about 1/3 of the floor bandwidth. The profile supports the target: full decode_16 has FA3 at 186.2 of 194.4 us (floor 63.2), decode_32 is 377.9 us against a floor of 126.4, and verify_16/32 are 94-96% FA3. The head mapping is right for both geometries. Under tp8, Hq=16, Hkv=2, G=8, S=1, so FA3 puts all 16 q heads on super-head 0, heads 0-7 in the low half (kv0) and heads 8-15 in the high half (kv1). Under dp, Hq=128, Hkv=16, S=8, giving 16 q heads per super-head with the same grouping. The zero half adds exactly 0 to each logit, and the discarded output half is the other head's P@V. The expected directions agree with TASK.md weights. Each B=16/32 load is about 75-84% full-layer time (16 x 194-383 us against 46 x 21-29 us), so a 20-35% FA3 gain moves those loads about 15-28%. The B=1 loads pay roughly 10-12 us x 16 on loads of about 1.2-1.3 ms, so they get about 12-15% slower; the brief's ~10% is slightly optimistic. The geomean improves only if the FA3 gain at B=16/32 is above about 8-10%. Weakest link: nobody has measured that the 64-byte gather granularity, rather than per-row index and latency limits that pairing leaves unchanged per byte, is what caps FA3. Doubling headdim also doubles the bytes per K/V stage in shared memory. For the hdim-128 FP8 paged non-TMA instantiation, that can cut pipeline stages or change kBlockN, which could offset the larger per-row fetch. Second weakest: this needs an hdim-128 e4m3 FA3 build in the image's sgl_kernel, with pack_gqa working at 16 q heads per kv head. Otherwise it fails cleanly, which is acceptable.

### Confounders on the current kernel
- fused-preattention-triton: no confounding on the target shapes. It is about 1-2 us (about 1%) of full B=16/32 and is untouched; the full-layer FA3 kernel time is read directly. It does interact on B=1: because it writes q as [tokens, Hq, 64], the pad needs zeros plus 2 copies, which is where the expected decode_1/verify_1 loss comes from. If it were absent the same pad launches would follow the parity q cast, so the loss attaches to this change's layering and not to the landed kernel. The follow-up (paired-heads-launch-cleanup) should have this kernel write q_pad directly.
- judge scheduler_metadata (not a landed hypothesis but an inherited input): the rebuild uses max_seqlen_k = page_table.shape[1]*page_size, while the judge's used cache_seqlens.max(), so the split cap may differ on decode_16/32. This is intrinsic, because the paired geometry cannot use the old metadata and cache_seqlens.max() would be a host sync. It is also small: LESSONS records that num_splits is only a cap under dynamic varlen split. Any schedule effect would show in the FlashAttnFwdCombine row.

### Interference and A/B needs
No companion. A gain means the full-layer FA3 kernel row drops on decode/verify_16/32 with sliding flat, which can only come from the paired geometry. A loss splits three ways and the profile separates them: the paired FA3 kernel itself getting slower (the hdim-128 smem/stage trade-off), more combine time (the split schedule, its own row), or pad/un-pad/metadata launch overhead (their own small rows plus the shape-minus-kernel-sum residual, which matters on B=1). An ab-revert of fused-preattention-triton tells us nothing, because it is not on the measured path. A 'pass None instead of rebuilding metadata' control is functionally the same thing: with None, FA3 runs prepare_varlen_num_blocks internally using the same page-table-derived max_seqlen_k, so the launch count and schedule are the same. Recommend the builder simply pass None on decode as on verify, which makes the code simpler and leaves nothing to A/B. A num_splits control would duplicate the queued fa3-num-splits-full-decode. Readout instruction for the harvester: judge on the FA3 kernel's own us on the four full B=16/32 shapes and on per-shape ratios, not on the headline, since B=1 losses can mask a real mechanism gain in the geomean.

### Open questions
- Does the image's sgl_kernel FA3 include the e4m3 hdim-128 forward with paged KV (page_size 1, non-TMA) and pack_gqa at qhead_per_khead=16? If not, every full test fails at dispatch. Harvest as failed, not regression.
- For hdim-128 FP8 paged non-TMA, what kBlockN and stage count does FA3 pick against hdim 64? If the stages halve, bytes in flight per SM may not grow as the brief claims.
- Is the ~1.7 TB/s really caused by double L2/DRAM fills of each 128-byte line (two heads read separately)? An ncu dram__bytes_read on full decode_32 before and after would confirm or refute the mechanism directly.
- B=1 loads: is the loss about 12-15% as estimated, or does the paired FA3 at B=1 (one super-head, so all parallelism comes from splits) also slow down? Read the full decode_1 FA3 row on its own.


## Implementation notes

Worktree `.worktrees/muse_attention_decode/fa3-hdim128-head-pairing`. Only `kernel_fn` in `muse_attention_decode/submission.py` changed, and only the FA3 call region. The pre-attention Triton kernel, the cache store and `rms_norm_gate` are untouched. Diff stat: 1 file, +38/-6.

- `paired = window == 0 and hkv % 2 == 0` is a config-only branch. Sliding layers and odd Hkv keep the original unpaired call byte for byte, and still pass the judge's `scheduler_metadata`.
- Paired path: `k_cache` and `v_cache` are viewed as `[pages, page_size, S, 128]` (contiguous views of the same storage, read in place). `q_attn = torch.zeros((tokens, Hq, 128), FP8)` is allocated each call. The FP8 q from the pre-attention kernel, viewed as `[tokens, S, 2, G, 64]`, is copied into the low half (parity 0) and high half (parity 1) of `q_attn.view(tokens, S, 2, G, 128)`.
- When kv_scale != 1, the descale expands to `(B, S)` instead of `(B, Hkv)`.
- After FA3, `attn = torch.empty((tokens, Hq, 64), bf16)` gets each query's own half of `out.view(tokens, S, 2, G, 128)`. That feeds the same `rms_norm_gate` call as before. The existing `.to(torch.bfloat16)` is kept and does nothing here.
- **Deviation (following the reviewer):** on full decode the spec rebuilds `scheduler_metadata` with `get_scheduler_metadata(...)`. The adversarial review recommends passing `None` on decode as on verify, because it is functionally the same: FA3 runs prepare_varlen_num_blocks internally with the same page-table-derived max_seqlen_k. I pass `metadata = None` on every paired call, so there is no rebuild code and no extra launch. The judge's metadata is never used with the paired geometry.
- Not verified locally, since this node has no GPU, torch or sgl_kernel. Open items:
  - whether FA3 in the image has e4m3 hdim-128 paged forward with pack_gqa at 16 q heads per kv head;
  - whether `torch.zeros` and slice `copy_` on float8_e4m3fn CUDA tensors work in the image's torch (both are supported in recent PyTorch).

## Rules check

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

## Result

- job `facc479992e24a35` (check), status done, passed True, tests 16/16
- speedup 1.1706x, 95% CI [1.1530, 1.1924], score 3217.4285398611996 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.1284429079326423, "ci": [1.1115162643136527, 1.1453695515516318], "job": "c297440e53334f34"}; base result: muse_attention_decode/hypotheses/done/gain/fused-preattention-triton.result.json
- **outcome: gain** (gain)
- files: `muse_attention_decode/hypotheses/done/gain/fa3-hdim128-head-pairing.result.json`, `muse_attention_decode/hypotheses/done/gain/fa3-hdim128-head-pairing.patch`; branch `hyp/muse_attention_decode/fa3-hdim128-head-pairing`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 16.77 | 16.6 | 1.0024 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.87 | 38.75 | 0.7261 | lost |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 20.8 | 20.77 | 0.9947 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 194.45 | 165.21 | 1.1764 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 25.14 | 25.54 | 0.9938 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 377.92 | 306.99 | 1.2282 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.12 | 17.43 | 0.9744 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.02 | 45.98 | 0.6518 | lost |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.5 | 22.78 | 0.9828 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 198.32 | 172.95 | 1.143 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.72 | 28.85 | 0.9929 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 383.11 | 309.37 | 1.238 | gained |

gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. Expected hit: 7/7. Noise ±1.7%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

