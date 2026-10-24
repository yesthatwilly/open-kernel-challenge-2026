# muse_moe_w4a16 · 2026-10-24T181733Z · submit · marlin-two-stream-expert-halves-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T164857Z-marlin-two-stream-expert-halves-small-m.md` (branch `hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m`).

- **Job:** `5d630369b1634f77` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `ea34588`

## Goal
At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4 test and num_tokens > 32 run master unchanged), split the expert work into two independent halves by local expert (A = local experts 0-15, B = 16-31) and run them on two CUDA streams so one half's Marlin ramp/tail, the silu_and_mul launch and the inter-kernel graph gaps overlap the other half's Marlin. Align: launch _align_small_kernel's logic with grid (2,) in a new _align_small_halves_kernel (same code; program h restricts the one-hot/counts/segments/blocks to experts [16h, 16h+16), every other route is treated like a remote route): each program writes its own sorted_ids_h (capacity n + 16*(bsm-1), sentinel n in every unused slot), block_experts_h (-1 past its used blocks; expert ids stay global-local, i.e. 16h + j) and n_padded_h, plus local_ids_h = local id if in its half else -1; program 0 also writes the full local_ids. Buffers are one stacked allocation per kind. block_size_m stays master's pick per M (8 at bs<=16, 16 at bs=32). kernel_fn: allocate cache1/cache2/cache13 and out on the current stream as now, then side.wait_stream(current); half A on the current stream and half B under torch.cuda.stream(side) each run moe_wna16_marlin_gemm w13 (its own triple, writing only its routes' rows of the shared cache1), silu_and_mul(cache1, cache2, expert_ids=local_ids_h, expert_step=1) (rows whose id is -1 are skipped, so each stream touches only its own rows), and w2 into the shared cache3; then current.wait_stream(side) and master's post-norm on the full local_ids (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32). Half B uses a second Marlin lock workspace (marlin_make_workspace(device, max_blocks_per_sm=4) made in prepare) so the two concurrent Marlin calls never share locks; the side stream is created once in prepare(). Same Marlin flags. prepare() compiles the halves kernel for tokens 8..32 and extends _align_self_check: for each half, per-expert route sets and block counts must equal moe_align_block_size(ref_local with the other half set to -1, bsm, 32).

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m`, `muse_moe_w4a16/submission.py` only (uncommitted; diff stat: 1 file, +229 / -1).

- New constants `TWO_STREAM_MIN_TOKENS = 8`, `TWO_STREAM_MAX_TOKENS = 32`, `HALF_SORTED_ALIGN = 64`.
- New Triton `_align_small_halves_kernel`, launched with grid (2,): `_align_small_kernel`'s code, program h restricted to local experts [16h, 16h+16) (routes outside the half treated as remote). Writes its own sorted_ids row (capacity n + 16*(bsm-1), sentinel n in every unused slot), block_experts row (bank-local ids 16h+j, -1 past used blocks), n_padded[h], and local_ids_h[h] (local id if in half, else -1). Program 0 also writes the full local_ids. One stacked allocation per kind (`[2, cap_stride]`, `[2, n_blocks]`, `[2]`, `[2, n]`).
- Wrapper `_align_small_halves` returns per-half triples as views (`sorted_ids[h, :cap]`, `block_experts[h]`, `n_padded[h:h+1]`).
- New `_experts_two_stream`: per the review's REQUIRED fix, each half has its own cache1 and cache2 (stacked `[2, M*8, 2I]` / `[2, M*8, I]`), and cache3 is a separate buffer `[M*8, H]` (no cache13 aliasing). Both halves write disjoint rows (their own routes) of the shared cache3, which the post-norm reads. All buffers and the align are issued on the current stream before `side.wait_stream(current)`. Half A runs on the current stream with `state["workspace"]`, half B on `state["side_stream"]` with `state["workspace_b"]`. Each half runs w13 Marlin, then `silu_and_mul(..., expert_ids=local_ids_h[h], expert_step=1)`, then w2 Marlin, with master's Marlin flags (is_ep=True). Then `current.wait_stream(side)`.
- kernel_fn: shape-only branch `expert_map is not None and count == 32 and 8 <= num_tokens <= 32` goes to `_experts_two_stream`. block_size_m is master's pick (8 at bs<=16, 16 at bs=32). Everything else (bs=1, bs=4, bs>=33) runs master's code unchanged. Master's post-norm tail was moved verbatim into `_post_norm(...)`, which both paths call. The only edit inside it is `x.device` to a `device` argument.
- prepare(): creates `workspace_b = marlin_make_workspace(device, max_blocks_per_sm=4)` and `side_stream = torch.cuda.Stream(device)` once, eagerly. Compiles `_align_small_halves` for tokens 8..32 on dummy ids. `_align_self_check` now ends by calling the new `_align_halves_self_check` (tokens 8/16/24/32 x three local_start values x 6 routings, including half-A-only and half-B-only). For each half it checks the full and per-half local_ids, and that the per-expert route sets and block counts equal `moe_align_block_size(ref_local with the other half set to -1, bsm, 32)`. It also checks that block_experts are in-half, ascending, and -1 past the used blocks, and checks the sentinel fill. It prints one summary line tagged `[marlin-two-stream-expert-halves-small-m]`.

Deviations and choices:
- The spec had one shared cache1. Following the review ("each expert half must get its own cache buffers"), cache1 and cache2 are now per half. cache3 is shared, with disjoint rows, because the post-norm needs one y. It is no longer aliased with cache1.
- Each half's sorted_ids row stride is rounded up to a multiple of 64 int32. This keeps half B's slice 256-byte aligned in case Marlin vector-loads sorted_ids. The capacity passed (the view length) is still n + 16*(bsm-1).
- No `record_stream`. Every cross-stream tensor is allocated on the current stream and stays referenced until after `current.wait_stream(side)` is enqueued. Marlin's internal side-stream temporaries live in the side stream's pool.

Uncertainties (cannot run here):
- This assumes `moe_wna16_marlin_gemm` writes into and returns the passed `c` (master already relies on this for cache1). cache3 is used directly as y.
- This assumes Marlin does no host-side size check of `sorted_token_ids` that would reject a 16-expert capacity (n + 16*(bsm-1)), which is smaller than moe_align_block_size(..., 32) would allocate.
- This assumes silu_and_mul skips rows whose expert_id is -1 (SGLang's EP path, as baseline relies on). If it does not, it only computes garbage rows in that half's private cache2, which w2 never reads, so this is a perf concern, not a correctness one.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's call graph (`_experts_two_stream`, `_align_small_halves`, `_post_norm`). The new host syncs are in `_align_halves_self_check`, which runs from prepare() only. Allocation sizes and the two-stream branch depend only on M, the bank's count and block_size_m (shape/config). Stream fork/join uses `wait_stream`, which is capturable. The side stream is created in prepare().
- [x] Nothing derived from `data` is cached across calls. State gains only a workspace and a stream, both made in prepare() from the device.
- [x] Weights: unchanged. The same Marlin-repacked NVFP4 bank from prepare() is read by both halves. No new weight storage.
- [x] Activations BF16 where SGLang keeps BF16: cache1/cache2/cache3 are x.dtype (bf16) as in master, with the same Marlin flags (use_fp32_reduce=True). KV cache / query precision: n/a, no attention in this problem.
- [x] Attention sees every key: n/a, no attention in this problem. Every local route is still computed: each route is in exactly one half, and the self-check verifies this.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. Only new torch/triton calls and SGLang ops already imported.
- [x] All work happens inside the timed call; no output caching or replay. The align, both Marlin chains and the post-norm all run in kernel_fn on fresh inputs.
- [x] First line `# okc: problem=muse_moe_w4a16` kept. No triton.autotune.

okc_lint output:

```
warn  host-sync    L294/L296/L298/L302: torch.randperm(...).tolist() in _align_halves_self_check  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L325: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L326/L327: .tolist() / .item() in _align_halves_self_check  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L375/L377/L379/L398/L403/L404: master's _align_self_check  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L495/L497: torch.cuda.synchronize() in prepare()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m/muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK (worktree __pycache__ removed).

## Risks / what could fail
high
High risk. Costs: two extra Marlin launches, one extra silu_and_mul, a stream fork/join (two event nodes) and a 2-CTA align, ~5-8 us in total; LESSONS records side-stream losses elsewhere (FA3 schedule on a side stream, moe-align-side-stream on prefill). It pays only if Marlin at 8-32 tokens leaves SMs idle in ramp/tail long enough for the other stream's CTAs to use them: Marlin's grid is persistent (sms x blocks_per_sm) and smem-heavy, so the second kernel's CTAs land only as the first one's retire. Distinct from the held marlin-two-class-block-m-decode32, which splits by block size on one stream (serial launches, no overlap). Judge on the bs=8/16/32 paired ratios; the per-kernel profile rows will sum higher (four Marlin rows), so read shape time, not the kernel sum. Correctness hazards for the reviewer: silu_and_mul must skip rows whose expert id is -1 (it is SGLang's EP path, which relies on that); each stream writes disjoint rows of cache1/cache2/cache3; cache buffers allocated before the fork. If it loses at bs=8 but wins at 32, an ab-tune gating to bs >= 16 follows.

## Result (after the run)
- job `5d630369b1634f77` (submit), status done, passed True, tests 5/5
- speedup 1.3024x, 95% CI [1.2829, 1.3220], score 10325.63799826287 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/marlin-two-stream-expert-halves-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/marlin-two-stream-expert-halves-small-m.patch`; branch `hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m`
