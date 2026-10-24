# muse_moe_w4a16 · 2026-10-24T112901Z · check · triton-map-align-small-m-port

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T112155Z-triton-map-align-small-m-port.md` (branch `hyp/muse_moe_w4a16/triton-map-align-small-m-port`).

- **Job:** `c2feb68aafb44aeb` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `4adf4e1`

## Goal
Port muse_decode_layer moe-align-single-kernel-small-t (patch: muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch; its _align_small_kernel passed its prepare() self-check, 60 cases, 0 mismatches) to muse_moe_w4a16/submission.py. For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test; bs=33/128/256/257 keep master's path), replace the three launches master issues before w13 - the expert_map[topk_ids] gather (at::native index_elementwise_kernel, 1.7-2.4 us), the unrolled_elementwise_kernel that follows it (1.4-1.9 us, present at every shape, most likely inside the moe_align_block_size wrapper) and moe_align_block_size_small_batch_expert_kernel (4.4-6.5 us) - with one single-program Triton kernel (grid (1,), num_warps=4) that reads the M*8 int32 topk_ids and writes everything downstream consumes: local_ids [M, 8] int32 (id - local_start when 0 <= id - local_start < local_count, else -1; local_start/local_count are host ints from config, no mod-256 wrap since the bank is the contiguous slice [local_start, local_start+32)), sorted_ids (route indices grouped by ascending local expert, each expert padded to a multiple of block_size_m with the sentinel M*8, every slot past the used length also M*8, capacity M*8 + 32*(block_size_m-1) as master's 32-expert align allocates), block_experts (each m-block's expert, -1 past the used blocks) and n_padded (int32 [1]). Use the decode kernel's counting sort as is: one-hot [BLOCK_N, 32], per-expert counts, padded cumsum for segment starts, rank via cumsum of the one-hot, three disjoint store sets into sorted_ids. Remote routes get no bucket (as in decode, where the self-check matched moe_align_block_size's per-expert blocks and n_padded exactly). block_size_m pick unchanged (master's rule; the moe_align_single_token branch is unreachable with an expert_map and stays). local_ids still feed silu_and_mul(expert_ids=...), _post_norm_combine_kernel and, at bs=32, fused_post_expert_norm_weighted_sum(slot_expert_ids=...). Both Marlin calls unchanged. prepare(): warm-compile the kernel for every num_tokens 1..32 on dummy ids and run the decode self-check against moe_align_block_size(ref_local, bsm, 32) (32, matching master's call), untimed, printing the one-line summary.

## What changed, and where
Worktree file: `.worktrees/muse_moe_w4a16/triton-map-align-small-m-port/muse_moe_w4a16/submission.py` (uncommitted; diff stat `1 file changed, 203 insertions(+), 16 deletions(-)`).

- Added `ALIGN_FUSED_MAX_TOKENS = 32` and `_align_small_kernel`, ported from the muse_decode_layer patch (`moe-align-single-kernel-small-t.patch`) with the counting sort unchanged: one-hot `[BLOCK_N, NUM_E]`, per-expert counts, padded cumsum for segment starts, rank by cumsum of the one-hot, three disjoint store sets into sorted_ids, block_experts by segment test, n_padded `[1]` int32. Grid `(1,)`, `num_warps=4`.
- Deviations from the decode kernel, as the spec and review ask: (a) `local_start` and `local_count` are host-int kernel args (in `do_not_specialize` with `n`, `cap`, `n_blocks`), not a device-tensor load; (b) the map is `0 <= id - local_start < local_count`, with no `& 255` wrap; (c) `topk_ids` is cast to int32 inside the kernel (`tl.load(...).to(tl.int32)`), and the host does no `.to(torch.int32)` on this path; (d) the one-hot width is a constexpr `NUM_E = next_power_of_2(local_count)` (= 32) rather than a literal 32.
- `_block_size_m(num_tokens, count)` factors out master's rule unchanged. `_align_small(topk_ids, local_start, local_count, bsm)` allocates local_ids `[M, 8]` int32 contiguous and sorted_ids with capacity `M*8 + count*(bsm-1)`, with `cdiv(cap, bsm)` block_experts, the same as master's `moe_align_block_size(..., 32)`.
- `prepare()`: the bank dict gains `"start"`. For a bank with an expert_map, `prepare()` warm-compiles `_align_small` for every M 1..32 using the bank's own start/count (the same values kernel_fn passes) on dummy ids `full(start)`, then runs `_align_self_check(device, count)`. That self-check is the decode check (60 cases = T in {1,8,16,32} x local_start in {0, 32, 224} x 5 routings) compared against `moe_align_block_size(ref_local, bsm, 32)`. Here `ref_local` comes from a reference expert_map gather (contiguous bank, no wrap). The check also verifies local_ids dtype/contiguity and that the sorted_ids/block_experts capacities match SGLang's. Per the review, it prints one `pad ... n_pad=X r_pad=Y [DIFF]` line per case, and the summary counts the cases where n_pad != r_pad (info only, never a hard mismatch).
- `kernel_fn`: when `expert_map is not None and num_tokens <= 32` (a shape-only branch), one `_align_small` call replaces the `expert_map[topk_ids]` gather, the int32 cast and `moe_align_block_size`. Everything else runs master's path unchanged: the `moe_align_single_token` branch (unreachable with an expert_map) and the gather + `moe_align_block_size(..., 32)` at M > 32. Both Marlin calls, silu_and_mul(expert_ids=local_ids), `_post_norm_combine_kernel` and the bs=32 `fused_post_expert_norm_weighted_sum(slot_expert_ids=local_ids)` are unchanged and consume the new local_ids.
- When reading the result, do what the review asks: confirm in the profile that index_elementwise, unrolled_elementwise and small_batch_expert_kernel are gone at bs<=32, record `_align_small_kernel` time at bs=1/8/16/32, and read the `pad` lines in the log for whether SGLang buckets -1 routes (r_pad > n_pad).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/triton-map-align-small-m-port diff`:

- [x] CUDA-graph capturable: kernel_fn's new path is one Triton launch plus `torch.empty` allocations sized from shapes and bank host ints only. It has no `.item()`/`.cpu()`/`synchronize()`/`.tolist()`/tensor prints, and it branches only on `num_tokens` (a shape) and `expert_map is not None` (prepare-time state). All syncs, `.tolist()` and `.item()` in the diff are in `_align_self_check`/`prepare()` (lint confirms they are outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls: the new state entry `bank["start"]` is a prepare-time int from the bank key. The warm compile and self-check use dummy or synthetic ids only.
- [x] Weights: untouched. The Marlin repack in `prepare()` is unchanged, and NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: x, cache1/2/3 and the outputs are unchanged. The kernel handles only int32 routing metadata.
- [x] n/a: attention (none in this problem). No expert or route skipping beyond SGLang's own EP semantics: remote routes are excluded exactly as expert_map = -1 excludes them, and every local route is computed.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds only a Triton kernel, Python helpers and prints in prepare().
- [x] All work happens inside the timed call: the per-call map and align run in kernel_fn on fresh topk_ids. Nothing is cached or replayed.
- [x] First line `# okc: problem=muse_moe_w4a16` kept. No `triton.autotune`.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`):

```
warn  host-sync    L206: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L208: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L210: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L229: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L234: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L235: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L319: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L321: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK (the __pycache__ it left in the worktree was removed).

## Risks / what could fail
medium
Why this pays here when it was neutral on decode_layer: the saving is the same ~5 us in absolute terms, but decode_layer shapes are 170-620 us (decode_1 still moved 171 -> 166 us, 1.03x, there), while this problem's decode_1 is 55 us, so the same microseconds are 7-10% of a load. Judge it per shape and by the replaced kernels' own times in top_kernels (align/index/unrolled rows gone, _align_small_kernel added), not by Marlin time, which drifts 20-30% on identical code (align-256-experts-below-128 review). If the unrolled_elementwise row stays, it is not from the align wrapper; record which. Independent of marlin-block-m-two-smaller-on-postnorm (disjoint shapes: <=32 here, >=128 there) and can run from master in parallel. Overlap with the held align-remote-sentinel-ids: both drop remote routes from sorted_ids; this one also removes launches, so if both run, read the sentinel probe first for whether -1 handling cost Marlin anything. Risk: Marlin reads sorted_ids/block_experts layout exactly; the self-check guards it, and a wrong layout fails the 5 tests rather than slipping through.

## Result (after the run)
- job `c2feb68aafb44aeb` (check), status done, passed True, tests 5/5
- speedup 1.1850x, 95% CI [1.1672, 1.2028], score 11349.095591113803 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.2535643764852882, "ci": [1.2347609108380089, 1.2723678421325675], "job": "c00ef7295c154867"}; base result: muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/triton-map-align-small-m-port.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/triton-map-align-small-m-port.patch`; branch `hyp/muse_moe_w4a16/triton-map-align-small-m-port`
