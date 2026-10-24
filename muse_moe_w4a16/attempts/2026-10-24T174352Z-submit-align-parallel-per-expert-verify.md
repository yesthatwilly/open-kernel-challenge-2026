# muse_moe_w4a16 · 2026-10-24T174352Z · submit · align-parallel-per-expert-verify

Hypothesis: `muse_moe_w4a16/hypotheses/done/neutral/2026-10-24T164817Z-align-parallel-per-expert-verify.md` (branch `hyp/muse_moe_w4a16/align-parallel-per-expert-verify`).

- **Job:** `1947e3ec2abb4bdf` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `89b2252`

## Goal
Promote check-proven gain align-parallel-per-expert-verify (1.3249x [1.3050,1.3448], check job 45eacf4ca2e849d2) to the board.

For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when expert_map is not None, replace master's else-branch (topk_ids.to(int32), the expert_map[local_ids] gather = at::native index_elementwise 3.2 us, the at::native unrolled_elementwise 2.8 us, and moe_align_block_size's moe_align_block_size_kernel + count_and_sort_expert_tokens_kernel: ~10.5 us of kernels and 4 graph nodes in master's bs=128/256 profile) with ONE launch of a new multi-program Triton kernel _align_parallel_kernel, grid (NUM_E + 1,) = 33 programs, num_warps 4. Every program loops over the n = M*8 flat route ids in chunks of CHUNK = 1024 (tl.arange(0, CHUNK), masked), maps each id to local = id - local_start if 0 <= local < local_count else -1 (as _align_small_kernel does), and accumulates per-expert counts for all 32 local experts (one-hot [CHUNK, 32] summed over axis 0). Program e < 32 then computes padded = ceil(count/BSM)*BSM for all experts, its own exclusive seg_start (cumsum of padded, minus its own), and in a second pass over the chunks stores route index i at sorted_ids[seg_start + running_rank + in-chunk rank] for every route with local == e (in-chunk rank = tl.cumsum of the match mask minus 1; running_rank += the chunk's match count). It stores the sentinel n into its padding slots [seg_start+count, seg_start+padded) using tl.arange(0, PAD_BLOCK) with PAD_BLOCK = next_power_of_2(BSM), masked to padded-count (never tl.arange(0, BSM): BSM 48 occurs at M 116..127 and is what broke triton-map-align-verify-chunked), and writes block_experts[b] = e for b in [seg_start/BSM, (seg_start+padded)/BSM). Program 32 writes local_ids for all n routes (int32, same values as master), the sentinel n into every slot in [total, cap), -1 into block_experts[b] for b in [total/BSM, n_blocks), and n_padded = total. Capacities exactly as master's 32-expert moe_align_block_size allocates: cap = n + 32*(BSM-1), n_blocks = cdiv(cap, BSM). The store sets are disjoint across programs, so no atomics or ordering are needed. BSM is a constexpr; n, cap, n_blocks, local_start, local_count are do_not_specialize. The block_size_m pick is unchanged (two-smaller at >= 128). In prepare(), compile the kernel for BSM in (16, 32, 48, 64) on dummy ids and extend _align_self_check with T in (33, 57, 120, 128, 256, 257) using the same per-expert-set / block-count / sentinel / layout checks against moe_align_block_size(ref_local, bsm, 32). num_tokens <= 32 keeps _align_small_kernel; Marlin, silu_and_mul and post-norm are unchanged.

## What changed, and where
All in the worktree's muse_moe_w4a16/submission.py (uncommitted). Diff stat: 1 file, +154 / -16.

- New `_align_parallel_kernel` (Triton, `do_not_specialize` = local_start, local_count, n, cap, n_blocks; constexprs CHUNK, PAD_BLOCK, BLK, BSM, NUM_E). Grid `(local_count + 1,)` = 33 programs. Every program does pass 1: stream the n flat ids in CHUNK chunks, map to local (id - local_start if in [0, local_count) else -1), sum a [CHUNK, 32] int32 one-hot into counts. Then padded / seg_end = cumsum / seg_start / total as in `_align_small_kernel`.
  - Program e < local_count: its count, padded and seg_start pulled out with where+sum over the expert axis (review note). Pass 2 per chunk: 1-D int32 match mask `(mask & (id - local_start == e))`, rank = `tl.cumsum(match) - 1`, store route index at `base + rank`, `base += sum(match)` (base starts at seg_start). Padding: `tl.arange(0, PAD_BLOCK)`, PAD_BLOCK = next_power_of_2(BSM), mask `j < padded - count`. block_experts: a runtime loop over `[seg_start/BSM, seg_start/BSM + padded/BSM)` in BLK = 64 tiles (review open question: one expert can own up to cdiv(M, BSM) blocks, e.g. skewed cases).
  - Program local_count: local_ids for all n routes (int32, -1 for remote, same values as master), sentinel n into `[total, cap)` (loop over cap in CHUNK tiles, masked), -1 into block_experts `[total/BSM, n_blocks)` (BLK tiles), and n_padded = total.
  - No program reads another's stores (local ids are recomputed from topk_ids), store sets are disjoint, no atomics.
- New host wrapper `_align_parallel(topk_ids, local_start, local_count, block_size_m)`: same allocations as `_align_small` (cap = n + 32*(BSM-1), n_blocks = cdiv(cap, BSM), sentinel n).
- kernel_fn: when expert_map is not None, `_align_small` for num_tokens <= 32 (unchanged), `_align_parallel` above it (branch on shape only). The master else-branch (gather + moe_align_block_size) now only runs when expert_map is None (never with the EP8 bank). block_size_m pick unchanged.
- prepare(): compiles `_align_parallel_kernel` for BSM in (16, 32, 48, 64) on dummy ids (n is runtime, so one variant per BSM).
- `_align_self_check`: now loops over (tokens, bsm) shapes: master's T in (1, 8, 16, 32) with SGLang's pick via `_align_small`, plus T in (33, 57, 120, 128, 256, 257) via `_align_parallel` with SGLang's pick and, at T >= 128, also kernel_fn's two-smaller pick (so BSM 16/32/48/64 are all exercised: 33->16, 57->16, 120->48, 128->48 and 16, 256->64 and 32, 257->64 and 32). Same per-expert-set / block-count / n_padded / block_experts layout / sentinel checks against `moe_align_block_size(ref_local, bsm, 32)`. Pad lines for T > 32 print only when they differ. Added a separate summary line `[align-parallel-per-expert-verify] self-check tokens > 32: <cases> cases, <k> mismatches, <c> capacity-only, <o> other` (pattern borrowed from the chunked attempt, since SGLang's wrapper may allocate larger capacity at big n; "0 other" is the pass criterion).

Deviations / choices:
- CHUNK = 512 with num_warps = 8 (the review's REQUIRED fix for the [1024,32] one-hot at 4 warps: 64 int32 per thread per temporary). Constants ALIGN_PAR_CHUNK / ALIGN_PAR_WARPS / ALIGN_PAR_BLK at module level.
- Grid is `local_count + 1` rather than literal `NUM_E + 1`; identical (33) for this bank.
- tl.histogram not used (cannot verify mask support on this node without Triton).
- All tl.arange sizes are powers of two: CHUNK 512, NUM_E 32, PAD_BLOCK next_pow2(BSM) (64 for BSM 48), BLK 64. BSM itself appears only in scalar arithmetic.

Uncertainties: nothing could be compiled or run here (no GPU/Triton). Kernel time at bs=256 (each program: 4 chunks pass 1 + 4 chunks pass 2) is the open number; read the `_align_parallel_kernel` row against the four removed rows in top_kernels and confirm the self-check "0 other" line and that all 4 BSM variants compiled.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/align-parallel-per-expert-verify diff`.

- [x] CUDA-graph capturable: the new path in kernel_fn is torch.empty allocations sized from shapes/bank ints plus one Triton launch; no `.item()`/`.cpu()`/`synchronize()`/tensor prints; loop bounds inside the kernel are device-side; host branch is on num_tokens (shape) only. Syncs added are in prepare()/_align_self_check only.
- [x] Nothing derived from `data` is cached across calls: outputs are fresh per call; prepare-time state is unchanged (no new state entries).
- [x] Weights: untouched by this diff; repack in prepare() as before, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: the diff only touches int32 routing metadata; Marlin/silu_and_mul/post-norm unchanged.
- [x] n/a: attention item does not apply (MoE problem, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds a Triton kernel, a wrapper, self-check lines and prints only.
- [x] All work happens inside the timed call; no output caching or replay: the align runs every call inside kernel_fn; the prepare-time compile uses dummy ids only.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune; every tl.arange size a power of 2.

okc_lint output:

```
warn  host-sync    L329: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L331: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L333: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L351: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L356: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L357: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L455: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L457: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK (worktree __pycache__ removed).

## Risks / what could fail
medium

## Result (after the run)
Promotion passed: 1.3278x [1.3079,1.3477], board: {"rank": 410, "team": "yesthatwilly", "speedup": 1.3287, "first": false, "ahead": 409, "to_first": 1.8986}
