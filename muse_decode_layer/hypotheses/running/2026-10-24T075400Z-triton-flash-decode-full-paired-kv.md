---
problem: muse_decode_layer
slug: triton-flash-decode-full-paired-kv
created: 2026-10-24T075400Z
base: master
area: FA3 call
change: >
  On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace
  flash_attn_with_kvcache with a two-launch Triton flash-decoding pair built for the page-size-1
  layout: k_cache/v_cache are [pages, page_size, 2, 64] FP8, so one token's two KV heads are ONE
  contiguous 128-byte row (view as [pages*page_size, 128]), while FA3 gathers each head's 64 bytes
  per (request, kv-head) tile. Kernel 1, grid (B, S) with S = clamp(ceil(2*132/B), 1, 128) from
  shapes only (B=1: 128, B=16: 17, B=32: 9; ~2 waves of 132 SMs), each program owns request b and
  a contiguous token range of length ceil(page_table.shape[1]*page_size / S) rounded up to
  BLOCK_N=64, masked by t < cache_seqlens[b]. Per BLOCK_N tile it loads the int32 rows
  page_table[b, t // P] * P + t % P (P = page_size, so page 64 tests work too), then K and V as
  [BLOCK_N, 128] fp8 tiles (contiguous 128-byte segments), converts them to bf16 (exact) and runs
  the online softmax for all 16 query heads at once with a block-diagonal trick: Q2 is a [16, 128]
  bf16 tile whose rows 0-7 hold heads 0-7 in columns 0-63 (zeros in 64-127) and rows 8-15 hold
  heads 8-15 in columns 64-127 (zeros in 0-63), so S = Q2 @ K^T gives each head's scores against
  its own kv head; acc[16, 128] += P @ V and only each row-group's own 64 columns are kept.
  Queries are master's FP8 q (q.view(T,16,64).to(FP8), as FA3 receives) up-converted to bf16
  in-kernel; scores scaled by sm_scale*qk_scale_factor, K/V times k_scale/v_scale (scalars), fp32
  m/l/acc, num_warps=4, num_stages=3. It writes fp32 partials acc [B, S, 16, 64] and (m, l) [B, S,
  16] to fresh scratch. Kernel 2, grid (B, 16), combines the S partials (log-sum-exp) into a bf16
  [B, 16, 64] tensor that goes into the unchanged state['o_norm_gate'](out.reshape(T, 1024),
  gate). scheduler_metadata is ignored on this path. Sliding layers, verify blocks (q_len=8,
  including full verify) and everything else keep master's code. Same design as
  muse_attention_decode's queued triton-flash-decode-full-both-heads, applied here where full
  decode FA3 is 189.1 us of 442 us (decode_16 full, 1.73 TB/s) and 491.6 us of 623 us (decode_32
  full, 1.33 TB/s against a 137 us KV floor).
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "655 MB of FP8 KV read as 128-byte rows by 288 CTAs; at ~3 TB/s FA3's 491.6 us becomes ~220 us, ~40% of the shape and ~20% of the decode_32 load"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 328 MB of KV, FA3 189.1 us at 1.73 TB/s; 272 CTAs at ~3 TB/s is ~110 us
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: B=1 is latency-bound (FA3 18 us + 6.9 us combine against a 4.3 us byte floor); 128 CTAs of ~5 tiles plus a combine is about the same
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 unchanged; control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len=8 keeps FA3 (full verify included); control
risk: high
depends_on: null
priority: 3
kind: rewrite
controls_for: null
review: proceed
status: running
stage: dispatched
replaces: "FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine"
review_at: 2026-10-24T083747Z
review_companions: null
claimed_at: 2026-10-24T083747Z
base_sha: 5785b27b7b785ce157d4372280668438549e2c34
base_blob: de98030423860cfc9507b61d19f82a416223a23d
branch: hyp/muse_decode_layer/triton-flash-decode-full-paired-kv
worktree: .worktrees/muse_decode_layer/triton-flash-decode-full-paired-kv
mode: check
implemented_at: 2026-10-24T084126Z
impl_commit: bbbf3168de76ebc524cb23723b2ebf734b893072
submission_sha256: 25f223ef917c96cffb2a5965c921b85e5c5aa0ddd099b70cc4e00b9d30a62c7f
norm_sha256: 17685f9ed93efa351a1232036b6b85bc10c7ec8db305f4d0668059c225536acc
submission_blob: cf09a9196c4d7a20540bb82c9d79bdebaeea2a43
lint_warnings: 1
attempt_file: 2026-10-24T084126Z-check-triton-flash-decode-full-paired-kv.md
prepared_at: 2026-10-24T084126Z
job: e82b11f9d9f64b23
dispatched_at: 2026-10-24T084136Z
---

# triton-flash-decode-full-paired-kv

## Change

On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair built for the page-size-1 layout: k_cache/v_cache are [pages, page_size, 2, 64] FP8, so one token's two KV heads are ONE contiguous 128-byte row (view as [pages*page_size, 128]), while FA3 gathers each head's 64 bytes per (request, kv-head) tile. Kernel 1, grid (B, S) with S = clamp(ceil(2*132/B), 1, 128) from shapes only (B=1: 128, B=16: 17, B=32: 9; ~2 waves of 132 SMs), each program owns request b and a contiguous token range of length ceil(page_table.shape[1]*page_size / S) rounded up to BLOCK_N=64, masked by t < cache_seqlens[b]. Per BLOCK_N tile it loads the int32 rows page_table[b, t // P] * P + t % P (P = page_size, so page 64 tests work too), then K and V as [BLOCK_N, 128] fp8 tiles (contiguous 128-byte segments), converts them to bf16 (exact) and runs the online softmax for all 16 query heads at once with a block-diagonal trick: Q2 is a [16, 128] bf16 tile whose rows 0-7 hold heads 0-7 in columns 0-63 (zeros in 64-127) and rows 8-15 hold heads 8-15 in columns 64-127 (zeros in 0-63), so S = Q2 @ K^T gives each head's scores against its own kv head; acc[16, 128] += P @ V and only each row-group's own 64 columns are kept. Queries are master's FP8 q (q.view(T,16,64).to(FP8), as FA3 receives) up-converted to bf16 in-kernel; scores scaled by sm_scale*qk_scale_factor, K/V times k_scale/v_scale (scalars), fp32 m/l/acc, num_warps=4, num_stages=3. It writes fp32 partials acc [B, S, 16, 64] and (m, l) [B, S, 16] to fresh scratch. Kernel 2, grid (B, 16), combines the S partials (log-sum-exp) into a bf16 [B, 16, 64] tensor that goes into the unchanged state['o_norm_gate'](out.reshape(T, 1024), gate). scheduler_metadata is ignored on this path. Sliding layers, verify blocks (q_len=8, including full verify) and everything else keep master's code. Same design as muse_attention_decode's queued triton-flash-decode-full-both-heads, applied here where full decode FA3 is 189.1 us of 442 us (decode_16 full, 1.73 TB/s) and 491.6 us of 623 us (decode_32 full, 1.33 TB/s against a 137 us KV floor).

## Expected

- `*layer=full load=decode_32*` faster: 655 MB of FP8 KV read as 128-byte rows by 288 CTAs; at ~3 TB/s FA3's 491.6 us becomes ~220 us, ~40% of the shape and ~20% of the decode_32 load
- `*layer=full load=decode_16*` faster: 328 MB of KV, FA3 189.1 us at 1.73 TB/s; 272 CTAs at ~3 TB/s is ~110 us
- `*layer=full load=decode_1*` flat: B=1 is latency-bound (FA3 18 us + 6.9 us combine against a 4.3 us byte floor); 128 CTAs of ~5 tiles plus a combine is about the same
- `*layer=sliding*` flat: FA3 unchanged; control
- `*load=verify_*` flat: q_len=8 keeps FA3 (full verify included); control

## Risk

Risk: Triton's gather loads through a page table may not reach 3 TB/s; check fa3-full-single-split-probe's result first if it has landed in done/ (if FA3 is gather-pattern-bound the 128-byte-row access is the lever, if MLP-bound the 2-wave grid is). Competes with trtllm-decode-full-attention (queued, same shapes, same call site): whichever lands first is the other's base, so read this one against the base it was built on. Correctness: the block-diagonal Q2 makes the off-group dot products exactly zero contributions only because the zero columns multiply real K values; keep the zero padding explicit, never masked loads of q only. Tests through this path: batch 1 ctx 3000 full, batch 33 ctx 20000 full spread mixed page 1, batch 1 ctx 80000 full page 1, batch 16 ctx 20000 full page 1 (plus any page-64 full decode). Every key in the context is attended; no block skipping. Graph-safe: S and the split length come from B and page_table.shape only; scratch is allocated per call.

## Adversarial review

Reviewed 2026-10-24T083747Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite of the most expensive kernel on the full-decode shapes, and it breaks no rule. Its decode_32 baseline is wrong, though: on this problem FA3 takes 374.4 us of a 601.7 us shape, not 491.6 of 623. Judge it on the replaced kernels' own time against 374.4/188.8/25.2 us, and look hard at whether Triton pipelines the page-table-indirect loads.

### Justification
The mechanism is plausible. Full-decode FA3 is the largest single kernel here. Master's profile (0730135cdcac43b7) shows flash_fwd at 188.8 us of 433.0 us on decode_16 full (42.9%) and 374.4 us of 601.7 us on decode_32 full (59.5%). Against about 328 MB and 655 MB of FP8 KV, that is roughly 1.74-1.75 TB/s, which matches LESSONS ('full-attention decode reaches only 1.3-1.7 TB/s: the page-size-1 row gather is the limit'). Shape floors are 137.5 and 208.1 us, and the KV-only floor is about 137 us at decode_32, so there is real headroom. Reading each token's two KV heads as one contiguous 128-byte row, with about 270-290 CTAs that stay resident (48 KB of smem each, so about 13.8 MB in flight), is a reasonable way to raise achieved bandwidth.

The expected numbers are inflated. The 491.6/623 us figure is the overlap-inflated muse_attention_decode row (493.3 us 'GPU time' on a 377.9 us shape, which includes PDL wait). It is not this problem's profile. Corrected, 3 TB/s would take decode_32 full from about 374 to 220 us, saving about 155 us per shape. Times weight 16 that is about 2.5 ms of a 21.3 ms decode_32 load, or about 1.13x on that load, not '~20%'. decode_16 full would go from about 189 to 110 us, about 1.07x on its load. Geomean over six loads is about 1.03x, which a single run can still resolve.

The decode_1 'flat' is right in direction but the arithmetic is off. 80,000/128 = 625 tokens is about 10 BLOCK_N=64 tiles, not about 5. The incumbent is FA3 18.4 us plus combine 6.8 us = 25.2 us. A latency-bound 10-tile chain of dependent gathers could land anywhere from about 15 to 30 us, so a few us worse is possible and costs at most about 0.7% on that load.

The weakest link is software pipelining of a doubly indirect load. Each tile loads page_table rows first, and the K/V addresses depend on them. If Triton's pipeliner does not overlap the next tile's index load and gather with the current tile's math (num_stages=3 is no guarantee for indirect loads), each CTA serialises two DRAM latencies per 16 KB tile. That would land at about 2 TB/s, a small gain over FA3 instead of a large one. A secondary uncertainty is that the 128-byte-row argument assumes FA3 wastes or duplicates traffic by reading 64-byte halves per kv-head CTA. With 32-byte sector granularity it may not waste bytes, only request count. The lever may then be grid shape and occupancy rather than row width. fa3-full-single-split-probe would separate the two, but one run of this rewrite cannot.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: this is the base pipeline. It defines the FA3 call site, the FP8 q cast, o_norm_gate and scheduler_metadata usage. It does not confound the result. The change replaces exactly the call it introduced and keeps its q.to(FP8) rounding, so the numerics stay comparable. One caveat: FA3 may currently overlap its neighbours via PDL ('GPU busy' < kernel sum on decode_32: 604.3 vs 629.3 us). Plain Triton launches lose that overlap, which costs a few us that is not the kernel's own fault. Read kernel time, not only shape time.
- marlin-block-m-cap-16: no interaction. At T<=32 (all decode shapes) the cap leaves block_size_m at 8/16 as before, and the change touches only the attention path. The decode shapes this hypothesis targets were flat under that landing (0.996-1.002x).

### Interference and A/B needs
No companion is warranted. This is kind: rewrite and is judged on the replaced kernels' own time (flash_fwd + FlashAttnFwdCombine on full decode) against the incumbent and the floor, and both are readable from one profile. Neither landed hypothesis touches attention memory traffic or numerics in a way an ab-revert would disambiguate. The one sub-change worth isolating, contiguous 128-byte rows versus the grid/occupancy change, is already covered by the queued fa3-full-single-split-probe on the FA3 side. The natural follow-up is stage 2 (index prefetch, BLOCK_N/num_stages tuning), not a companion. Ordering caveat: trtllm-decode-full-attention and the queued fused-preattention-triton touch the same call site and the q production. Whichever lands first becomes the base, so compare this run's kernel time against the FA3 time in its own run's profile.

### Open questions
- The hypothesis is kind: rewrite but has no ## Milestones section. What is stage 1's pass bar? I suggest replaced-kernel time at or below 0.8x FA3 (<=300 us decode_32, <=150 us decode_16) and achieved KV bandwidth reported against the 137 us floor.
- Does the Triton build on the judge image pipeline the in-loop page_table index load ahead of the dependent K/V gather? If not, stage 2 should prefetch the next tile's row indices explicitly (load indices for tile i+1 before computing tile i) or stage all of a split's indices up front.
- Is the combine kernel (B=16/32, S=17/9) cheap enough? FA3's own combine is about 3 us there, and a second Triton launch adds about 0.3 us plus the partials round-trip of about 1 MB.
- Tests with page 64 and full decode (e.g. batch 1 ctx 3000 full, if its default page is 64) exercise the t//P, t%P path. Confirm which full-decode tests use page 64.
- If fa3-full-single-split-probe lands first and shows FA3 is grid- or occupancy-bound rather than gather-pattern-bound, the 128-byte-row framing in this hypothesis's justification should be revised, though the kernel design still stands.


## Implementation notes

Worktree `.worktrees/muse_decode_layer/triton-flash-decode-full-paired-kv`, file `muse_decode_layer/submission.py` only (191 insertions, 21 deletions; the deletions are the FA3 call re-indented into an `else:` branch, otherwise unchanged).

- **Call site (`kernel_fn`).** `if window == 0 and length == 1:` calls the new `_full_decode_attention(q.view(T,16,64).to(FP8), k_cache, v_cache, page_table, cache_seqlens, k_scale, v_scale, sm_scale*qk_scale_factor)`. Otherwise it runs master's FA3 call byte for byte: sliding layers, every verify block including full verify, and both descale branches. `scheduler_metadata` is not read on the new path. The store_cache path, the FP8 q cast and `state['o_norm_gate']` are unchanged. The output is bf16 [B,16,64] and takes the same `.reshape(tokens, 1024).to(bf16)` (a no-op) as before.
- **Kernel 1, `_full_decode_split_kernel`.** Grid (B, S) with S = clamp(ceil(264/B), 1, 128) (`_full_decode_splits`, so B=1: 128, B=16: 17, B=32: 9, B=33: 8). split_len = ceil(page_table.shape[1]*P / S) rounded up to BLOCK_N=64. Both come from shapes only. Each program handles keys [s*split_len, min((s+1)*split_len, cache_seqlens[b])). The loop bound is in-kernel from the loaded length: every key t < cache_seqlens[b] is visited exactly once across the splits, and only tiles beyond the length are not iterated. Each token's two KV heads are read as one 128-byte row: `row = page_table[b, t//P] * P + t%P`, at offset row*128 in the uint8 view of the cache. Both the page_table load and the K/V gathers are masked by `t < end` (end <= cache_seqlens[b]) and `t//P < page_table.shape[1]`, with other=0, so -1 entries are never dereferenced. The query tile is block-diagonal [16,128]: the FP8 q is loaded, upconverted, then `tl.where(own, q, 0.0)` writes explicit zeros (not a masked q load). K and V are converted fp8 to bf16 (exact). S = Q2 @ K^T in fp32, scaled by sm_scale*qk_scale_factor*k_scale*log2(e). The online softmax runs in the exp2 domain with fp32 m/l/acc, P goes to bf16 for P @ V, and a guard makes m_new == -inf use 0 (no -inf minus -inf). The masked store keeps only each head's own kv-head 64 columns of acc[16,128]. It writes unnormalized fp32 acc [B,S,16,64] and m, l [B,S,16] to fresh scratch sized from B, S and the head shape only. An empty split (spread=mixed short requests) runs zero iterations and writes m=-inf, l=0, acc=0. num_warps=4, num_stages=3.
- **Explicit prefetch of the indirect indices (review requirement).** The first tile's page-table rows are loaded before the loop. Inside the loop, the current tile's K and V loads are issued first. The *next* tile's page_table indices and rows are loaded next, before the current tile's QK^T, softmax and PV, and are carried to the next iteration. The kernel does not rely on Triton's pipeliner for the doubly indirect load.
- **Kernel 2, `_full_decode_combine_kernel`.** Grid (B, 16), BLOCK_S = next_pow2(S). m_max is the max over finite partials only (all -inf gives 0), and w_s = exp2(m_s - m_max) for finite m_s, 0 otherwise. out = sum(w*acc) / sum(w*l) * v_scale, with the denominator guarded > 0, stored as bf16. No NaN or Inf path exists.
- **k_scale / v_scale.** Both 0-d device tensors are passed as pointers and loaded in-kernel (k_scale in kernel 1's score scale, v_scale in kernel 2's output). There is no `.item()` and no branch on them. The path is the same for kv_scale 1 and 1/64.
- **Page size.** PAGE is a constexpr taken from k_cache.shape[1], so page 1 (scored) and page 64 (for example the `batch 1, ctx 3000, layer full` test, default page 64) both use the Triton kernel. There is no FA3 fallback.
- **No autotune, no host syncs.** The runtime ints (pt_stride, max_pages, split_len, num_splits) are `do_not_specialize`, so there is one compile per constexpr set. `_warm_full_decode(device)` in `prepare()` compiles kernel 1 for PAGE in {1, 64} (through the host wrapper, which also compiles kernel 2 at BLOCK_S=128) and kernel 2 for every BLOCK_S in {1, 2, ..., 128}.
- **Deviations from the spec.** (1) The K/V caches and q are passed as uint8 views and bitcast to fp8e4nv in-kernel, which keeps masked loads with other=0 well defined. (2) Tiles past a request's length are not iterated, instead of being iterated fully masked. This is not block skipping, because every key below the length is attended. (3) v_scale is applied in the combine, not on V in kernel 1, which is mathematically identical. (4) The SM count is a 132 constant (H200), not queried.

### Incumbent and stage-1 pass bar (corrected numbers, master profile 0730135cdcac43b7)
- Incumbent FA3 (flash_fwd main + FlashAttnFwdCombine) on full decode: **374.4 us of 601.7 us on decode_32 full**, **188.8 us of 433.0 us on decode_16 full**, and **18.4 + 6.8 = 25.2 us on decode_1 full**. Do not use the 491.6/623 us figures in the Change paragraph: those are from muse_attention_decode and overlap-inflated.
- **Stage-1 pass bar:** the replaced kernels' own time (`_full_decode_split_kernel` + `_full_decode_combine_kernel`) must be <= 0.8x FA3: **<= 300 us on decode_32 full** and **<= 150 us on decode_16 full**. decode_1 full should stay at or below about 25 us; a few us worse is tolerable, at most about 0.7% on that load.
- **Report achieved KV bandwidth** against the about 137 us KV-only floor on decode_32 (about 655 MB FP8 KV; about 328 MB on decode_16): bandwidth = KV bytes / kernel-1 time, alongside FA3's about 1.75 TB/s. Read kernel time, not only shape time: the plain Triton launches lose FA3's PDL overlap, which costs a few us.

### Uncertainties (cannot run here)
- Nothing has been compiled or run (no GPU, Triton or torch on this node). Triton API points to confirm on the judge: `do_not_specialize` by argument name, `tl.arange(0, 1)` for BLOCK_S=1 (used only for B >= 264; warmed anyway), and the dynamic `range(start, end, BLOCK_N)` with loop-carried `valid`/`rows` under num_stages=3.
- P is rounded to bf16 for P @ V, which is finer than FA3's FP8 P, so error should be at or below master's.
- Achieved bandwidth with one 64-row tile per iteration may still be latency-bound. A stage-2 follow-up is deeper K/V prefetch or a larger BLOCK_N.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/triton-flash-decode-full-paired-kv diff`:

- [x] CUDA-graph capturable. The new path has no `.item()`, `.cpu()`, `synchronize()` or tensor prints. Grid, S, split_len, BLOCK_S and scratch sizes come from B, page_table.shape and k_cache.shape only. The branch is on `cfg['window']` and `cfg['q_len']` (config). Per-request lengths are read in-kernel. `torch.cuda.synchronize()` exists only in `prepare()` (lint warning, fine).
- [x] Nothing derived from `data` is cached across calls. Scratch (acc, m, l, out) is allocated per call. `_warm_full_decode` uses dummy tensors in `prepare()` and keeps nothing in state.
- [x] Weights: n/a: no weight handling changed. `prepare()` only adds the kernel warm-up.
- [x] Activations: the attention output is bf16 as from FA3. The KV cache is read as FP8 and upconverted exactly. Queries are master's FP8 E4M3 cast (no coarser). Softmax statistics are fp32 and P is bf16 (finer than FA3's FP8 P).
- [x] Attention sees every key: each key t < cache_seqlens[b] falls in exactly one split and is attended. There is no top-k, sparsity, block skipping or weight-based skipping. Only positions at or beyond the length (which FA3 also excludes) are not iterated.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only new imports are `triton` and `triton.language`.
- [x] All work happens inside the timed call. There is no output caching or replay. Warm-up compiles kernels on dummy data only.
- [x] No `triton.autotune`. The configs are fixed (num_warps=4, num_stages=3, BLOCK_N=64).
- [x] The first-line marker `# okc: problem=muse_decode_layer` is kept.

Lint output (`okc_lint.py ... --problem muse_decode_layer`):
```
warn  host-sync    L260: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```
`python3 -I -m py_compile` passed. The `__pycache__` it left was removed.
