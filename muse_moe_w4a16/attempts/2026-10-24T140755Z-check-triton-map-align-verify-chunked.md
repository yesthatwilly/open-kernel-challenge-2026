# muse_moe_w4a16 · 2026-10-24T140755Z · check · triton-map-align-verify-chunked

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T140229Z-triton-map-align-verify-chunked.md` (branch `hyp/muse_moe_w4a16/triton-map-align-verify-chunked`).

- **Job:** `3aa4d7e193244255` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `eb5d866`

## Goal
Extend master's single-program map+align (_align_small_kernel, triton-map-align-small-m-port) to num_tokens > 32 (bs=128/256, and the bs=33/257/skewed-128 tests) with a chunked variant _align_chunked_kernel, so the expert_map gather (at::native index_elementwise ~3.2 us), the align wrapper's fill (unrolled_elementwise ~2.8 us), moe_align_block_size_kernel (~2.7 us) and count_and_sort_expert_tokens_kernel (1.8-2.6 us) become one launch. One program, num_warps=8, CHUNK=512 routes (constexpr), NUM_E=32. Pass 1 loops over chunks of the n=M*8 flat ids: it computes local = id - local_start (masked to [0, count), else -1), stores local_ids, and accumulates counts[NUM_E] += sum(onehot [CHUNK, NUM_E]). Then compute padded / seg_start / total exactly as the small kernel does. Pass 2 loops over the chunks again, keeping a running base[NUM_E] (init 0): pos = seg_start[e] + base[e] + (within-chunk cumsum rank), it stores sorted_ids[pos] = flat index for local routes, and does base += chunk counts. Then a loop writes each expert's padding slots with the sentinel n, a CHUNK-strided loop fills slots [total, cap), and a loop over m-blocks in CHUNK-sized strides writes block_experts (-1 past the used blocks) and n_padded. n, cap and n_blocks are runtime (do_not_specialize), so the only constexpr per call is BSM. prepare() compiles it once per block_size_m candidate (8..64) on dummy ids and extends _align_self_check to tokens (33, 128, 256, 257) with the same per-expert comparison against moe_align_block_size(ref_local, bsm, 32). kernel_fn: replace num_tokens <= ALIGN_FUSED_MAX_TOKENS with expert_map is not None, using _align_small up to 32 tokens (unchanged) and _align_chunked above. block_size_m picks, Marlin calls, silu and post-norm are unchanged.

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/triton-map-align-verify-chunked`, `muse_moe_w4a16/submission.py` only (uncommitted; diff stat: 1 file, +124 / -16).

- **New `_align_chunked_kernel`** (Triton, one program, `num_warps=8`, `CHUNK=ALIGN_CHUNK=512` constexpr, `NUM_E=next_pow2(32)`, `do_not_specialize` on local_start/local_count/n/cap/n_blocks, so BSM is the only per-call constexpr).
  - Pass 1 loops over the 512-route chunks of the flat ids. It maps `local = id - local_start` (masked to [0, count), else -1), stores local_ids and accumulates `counts += sum(onehot[CHUNK, NUM_E], axis=0)`.
  - padded / seg_end / seg_start / total are computed exactly as in `_align_small_kernel`.
  - Pass 2 (as the review REQUIRES) **recomputes** local from topk_ids. It never reloads local_ids, which avoids the intra-CTA store-to-load race. The running `base` starts at seg_start, `pos = base[e] + within-chunk cumsum rank`, it stores `sorted_ids[pos] = flat index` for local routes, and does `base += chunk counts`.
  - Padding slots: a single `[NUM_E, BSM]` sentinel tile, identical to the small kernel's. **Deviation:** the spec says "a loop". One tile covers all 32 experts at any BSM up to 64 (2K int32), so no loop is needed.
  - A CHUNK-strided loop fills slots [total, cap) with n. Another CHUNK-strided loop over m-blocks writes block_experts (-1 past the used blocks), then n_padded.
  - The store sets into sorted_ids are disjoint (real positions, per-expert padding, tail), so their order does not matter.
- **New `_align_chunked` wrapper**: the same sizes as `_align_small` (cap = n + count*(bsm-1), n_blocks = cdiv(cap, bsm)).
- **kernel_fn**: `expert_map is not None and num_tokens <= ALIGN_FUSED_MAX_TOKENS` became `expert_map is not None`, choosing `_align_small` at <=32 tokens and `_align_chunked` above (a shape-only branch). The fallback `else` branch (expert_map None) is untouched. The block_size_m picks (including the two-smaller pick at >=128), the Marlin calls, silu and the post-norm are unchanged. The bs<=32 code path is byte-identical.
- **prepare()**: after the existing `_align_small` warm loop, `_align_chunked` is compiled once per BLOCK_SIZE_M_CANDIDATES entry (8..64) on 33x8 dummy ids. Nothing sees any input.
- **`_align_self_check`** is extended to tokens (33, 128, 256, 257) x **every** block_size_m candidate (not just SGLang's pick, so kernel_fn's two-smaller picks 16/32 at bs>=128 are covered). These cases use `_align_chunked`, with the same per-expert comparison against `moe_align_block_size(ref_local, bsm, 32)`. The <=32 cases run first with the same seed, so they are unchanged.
  - Pad lines for the >32 cases print only when they DIFF. This keeps the log under the 16K-char tail, since there are 300 new cases.
  - Master's existing summary counts the known capacity difference (SGLang's wrapper allocates more: e.g. T=8 296/37 vs 288/36) as a "mismatch" in every case, so a separate `[triton-map-align-verify-chunked]` line reports the >32 mismatches split into capacity-only vs other and prints up to 10 "other" ones. **Read that line: "0 other" is the pass criterion.**
- Open questions from the review: CHUNK=256 and a 2-CTA variant are not tried (follow-ups). BSM=64/128 for swap-AB is not added (that hypothesis has not landed). BSM=64 is already in the compile list and the self-check.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/triton-map-align-verify-chunked diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints were added in kernel_fn's path. `_align_chunked` only allocates with sizes from shapes/count/bsm (host ints) and launches one kernel. Its loop bounds n/cap/n_blocks are host ints from shapes. The branch `num_tokens <= 32` is on shape. In-kernel loops are device-side. The new syncs/`.tolist()` are only in the prepare()-time self-check.
- [x] Nothing derived from `data` is cached across calls: local_ids/sorted_ids/block_experts/n_padded are fresh `torch.empty` each call. The state is unchanged.
- [x] Weights: untouched (no change to the repack; NVFP4 stays NVFP4).
- [x] Activations BF16 where SGLang keeps BF16: untouched. The new kernel only handles int32 routing metadata.
- [x] n/a: attention sees every key. No attention in this problem, and there is no routing skipping: every local route still gets a slot.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff only adds a Triton kernel, a wrapper, prepare-time warmup/self-check and a dispatch line.
- [x] All work happens inside the timed call; no output caching or replay. The align work moved from SGLang ops into one Triton launch inside kernel_fn. prepare() only warms compile and self-checks on synthetic ids.
- [x] No triton.autotune. The first-line marker `# okc: problem=muse_moe_w4a16` is kept.

okc_lint output:

```
warn  host-sync    L300: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L302: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L304: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L322: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L327: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L328: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L425: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L427: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
exit 0
```

`python3 -I -m py_compile`: OK (the __pycache__ was removed).

## Risks / what could fail
medium
Expected ~1.5-2% on each verify shape, ~0.5-0.7% headline, inside one run's CI. Judge it by the removed rows in top_kernels and the new kernel's own time. It is a launch-count change on 330-470 us shapes, which LESSONS says rarely moves the headline. It is still worth a run because it also gives the bs>=128 path a Triton align that a two-class block-m (see marlin-two-class-block-m-decode32) or the swap-AB rewrite can reuse. Risk: the serial chunk loops in one CTA (5 chunks at bs=257) could take 5+ us and leave little gain. If the kernel row exceeds ~6 us, the result is flat. Confounding: swapab-nvfp4-gemm-verify-rewrite replaces Marlin at the same shapes and keeps SGLang's align, so if both land the chunked align must be called with BSM=BLOCK_T (64/128): add those to the prepare compile and self-check.

## Result (after the run)
- job `3aa4d7e193244255` (check), status done, passed False, tests 0/0
- best at harvest: {"speedup": 1.3116740840932535, "ci": [1.2919989728318546, 1.3313491953546521], "job": "bc56182a80444d16"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: failed** (test:failed)
- files: `muse_moe_w4a16/hypotheses/done/failed/triton-map-align-verify-chunked.result.json`, `muse_moe_w4a16/hypotheses/done/failed/triton-map-align-verify-chunked.patch`; branch `hyp/muse_moe_w4a16/triton-map-align-verify-chunked`
