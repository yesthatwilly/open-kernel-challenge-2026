---
problem: muse_moe_w4a16
slug: marlin-two-stream-expert-halves-small-m
created: 2026-10-24T164857Z
base: master
area: launch overhead
change: >
  At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4
  test and num_tokens > 32 run master unchanged), split the expert work into two independent
  halves by local expert (A = local experts 0-15, B = 16-31) and run them on two CUDA streams so
  one half's Marlin ramp/tail, the silu_and_mul launch and the inter-kernel graph gaps overlap the
  other half's Marlin. Align: launch _align_small_kernel's logic with grid (2,) in a new
  _align_small_halves_kernel (same code; program h restricts the one-hot/counts/segments/blocks to
  experts [16h, 16h+16), every other route is treated like a remote route): each program writes
  its own sorted_ids_h (capacity n + 16*(bsm-1), sentinel n in every unused slot), block_experts_h
  (-1 past its used blocks; expert ids stay global-local, i.e. 16h + j) and n_padded_h, plus
  local_ids_h = local id if in its half else -1; program 0 also writes the full local_ids. Buffers
  are one stacked allocation per kind. block_size_m stays master's pick per M (8 at bs<=16, 16 at
  bs=32). kernel_fn: allocate cache1/cache2/cache13 and out on the current stream as now, then
  side.wait_stream(current); half A on the current stream and half B under torch.cuda.stream(side)
  each run moe_wna16_marlin_gemm w13 (its own triple, writing only its routes' rows of the shared
  cache1), silu_and_mul(cache1, cache2, expert_ids=local_ids_h, expert_step=1) (rows whose id is
  -1 are skipped, so each stream touches only its own rows), and w2 into the shared cache3; then
  current.wait_stream(side) and master's post-norm on the full local_ids
  (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32). Half B
  uses a second Marlin lock workspace (marlin_make_workspace(device, max_blocks_per_sm=4) made in
  prepare) so the two concurrent Marlin calls never share locks; the side stream is created once
  in prepare(). Same Marlin flags. prepare() compiles the halves kernel for tokens 8..32 and
  extends _align_self_check: for each half, per-expert route sets and block counts must equal
  moe_align_block_size(ref_local with the other half set to -1, bsm, 32).
expected:
  - shapes: bs=32 *
    direction: faster
    why: "master weakest shape (1.033x; Marlin w13+w2 187.6 us vs a 70.8 us floor, ~20 distinct local experts): two concurrent half-size Marlin chains fill each other's tails and hide the w13->silu->w2 dependency gaps; expect 4-8%"
  - shapes: bs=16 *
    direction: faster
    why: same mechanism on 116 us of Marlin over ~12 experts
  - shapes: bs=8 *
    direction: faster
    why: same mechanism on 105.8 us of Marlin over ~7 experts; smallest expected effect
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched (about one local expert, nothing to overlap)"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T165207Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T165338Z
base_sha: a928067ca1fead9b9f3da0998e11f654e387579f
base_blob: d9a9375ef4721ab9b28cc9d6364c92e67c7ccc59
branch: hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m
worktree: .worktrees/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m
mode: submit
implemented_at: 2026-10-24T165728Z
impl_commit: ea345883a3f48ce1b31b2198648961ff034fc764
submission_sha256: f9e813ee29cbf3a31bc6fca81c79cae37f8068070204751507caec392dbdd99e
norm_sha256: e14b6ce4971370f6e2e4020d4604ad411719c3aa6d9f59f8f097e088b4e144d9
submission_blob: bf2d61e93501721a0e00508157ae60ae4b279d10
lint_warnings: 15
attempt_file: 2026-10-24T181733Z-submit-marlin-two-stream-expert-halves-small-m.md
prepared_at: 2026-10-24T181733Z
job: 5d630369b1634f77
dispatched_at: 2026-10-24T181917Z
---

# marlin-two-stream-expert-halves-small-m

## Change

At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4 test and num_tokens > 32 run master unchanged), split the expert work into two independent halves by local expert (A = local experts 0-15, B = 16-31) and run them on two CUDA streams so one half's Marlin ramp/tail, the silu_and_mul launch and the inter-kernel graph gaps overlap the other half's Marlin. Align: launch _align_small_kernel's logic with grid (2,) in a new _align_small_halves_kernel (same code; program h restricts the one-hot/counts/segments/blocks to experts [16h, 16h+16), every other route is treated like a remote route): each program writes its own sorted_ids_h (capacity n + 16*(bsm-1), sentinel n in every unused slot), block_experts_h (-1 past its used blocks; expert ids stay global-local, i.e. 16h + j) and n_padded_h, plus local_ids_h = local id if in its half else -1; program 0 also writes the full local_ids. Buffers are one stacked allocation per kind. block_size_m stays master's pick per M (8 at bs<=16, 16 at bs=32). kernel_fn: allocate cache1/cache2/cache13 and out on the current stream as now, then side.wait_stream(current); half A on the current stream and half B under torch.cuda.stream(side) each run moe_wna16_marlin_gemm w13 (its own triple, writing only its routes' rows of the shared cache1), silu_and_mul(cache1, cache2, expert_ids=local_ids_h, expert_step=1) (rows whose id is -1 are skipped, so each stream touches only its own rows), and w2 into the shared cache3; then current.wait_stream(side) and master's post-norm on the full local_ids (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32). Half B uses a second Marlin lock workspace (marlin_make_workspace(device, max_blocks_per_sm=4) made in prepare) so the two concurrent Marlin calls never share locks; the side stream is created once in prepare(). Same Marlin flags. prepare() compiles the halves kernel for tokens 8..32 and extends _align_self_check: for each half, per-expert route sets and block counts must equal moe_align_block_size(ref_local with the other half set to -1, bsm, 32).

## Expected

- `bs=32 *` faster: master weakest shape (1.033x; Marlin w13+w2 187.6 us vs a 70.8 us floor, ~20 distinct local experts): two concurrent half-size Marlin chains fill each other's tails and hide the w13->silu->w2 dependency gaps; expect 4-8%
- `bs=16 *` faster: same mechanism on 116 us of Marlin over ~12 experts
- `bs=8 *` faster: same mechanism on 105.8 us of Marlin over ~7 experts; smallest expected effect
- `bs=1 *` flat: control: untouched (about one local expert, nothing to overlap)
- `bs=128 *` flat: control: untouched

## Risk

High risk. Costs: two extra Marlin launches, one extra silu_and_mul, a stream fork/join (two event nodes) and a 2-CTA align, ~5-8 us in total; LESSONS records side-stream losses elsewhere (FA3 schedule on a side stream, moe-align-side-stream on prefill). It pays only if Marlin at 8-32 tokens leaves SMs idle in ramp/tail long enough for the other stream's CTAs to use them: Marlin's grid is persistent (sms x blocks_per_sm) and smem-heavy, so the second kernel's CTAs land only as the first one's retire. Distinct from the held marlin-two-class-block-m-decode32, which splits by block size on one stream (serial launches, no overlap). Judge on the bs=8/16/32 paired ratios; the per-kernel profile rows will sum higher (four Marlin rows), so read shape time, not the kernel sum. Correctness hazards for the reviewer: silu_and_mul must skip rows whose expert id is -1 (it is SGLang's EP path, which relies on that); each stream writes disjoint rows of cache1/cache2/cache3; cache buffers allocated before the fork. If it loses at bs=8 but wins at 32, an ab-tune gating to bs >= 16 follows.

## Adversarial review

Reviewed 2026-10-24T165207Z by an adversarial subagent (model opus).

**Verdict: proceed.** Speculative but cleanly attributable: the change is gated to bs=8/16/32, and bs=1/128/256 act as untouched in-run controls, so one run reads either way. Fix one build bug first. master's cache1 and cache3 are both views at offset 0 of cache13, so half A's w2 writes would race half B's w13/silu rows. Expect flat to +2%, not 4-8%.

### Justification
Profile: Marlin is 92.8-94.6% of bs=8/16/32 (105.8/116.0/187.6 us vs floors 37.3/52.5/70.8 us), so it is the only lever there. The per-expert cost is far above the weight-streaming floor: about 20 us/expert at bs=1 and about 15 us/expert at bs=8, against about 5 us/expert of NVFP4 bytes. Marlin is latency- or bytes-in-flight-bound at small M, so more concurrent memory traffic could in principle help. The stated mechanism (ramp/tail plus inter-kernel gaps) is mostly the wrong one. Shape time minus kernel sum at bs=32 is only about 3 us (200.7 vs about 197.5), so hiding the silu launch and graph gaps is worth at most about 1.5%. Marlin's stripe partitioning also spreads every m-block across all SMs, so there is little per-expert tail to fill. The gain can only come from the two Marlin kernels' CTAs being co-resident on the same SMs. Weakest link: Marlin launches one CTA per SM and, as far as I recall, requests the device's max dynamic smem (in vLLM's marlin, launched with max_shared_mem). If so, the side stream's CTAs land only as the first kernel's retire, and the overlap collapses to the two kernel boundaries. The cost is certain: two extra Marlin launches (about 2-4 us each per the marlin-two-class review), an extra silu, a fork/join and a 2-CTA align, about 5-8 us. LESSONS also lists several flat or losing side-stream attempts (moe-align-side-stream, fa3-scheduler-metadata-side-stream). Two more weak points. (1) A fixed split by local id 0-15/16-31 under skewed popularity can leave one half with most of the routes, which shrinks any concurrency gain at bs=8 (about 7 experts). (2) Each half now spreads half the work over the same persistent grid, so fixed per-kernel cost (prologue, k-stripe lock reduction) doubles. The expected directions are consistent with the gating: bs=1/128/256 flat (code paths byte-identical), bs=8 the smallest effect. The 4-8% at bs=32 is optimistic; I expect flat to +2% at bs=32 and flat-to-slower at bs=8.

### Confounders on the current kernel
- sglang-marlin-path-parity: supplies the cache13 buffer with cache1 = cache13[:n*8*2I] and cache3 = cache13[:n*8*H], both at offset 0. This is safe serially but a cross-stream data race here: cache3 row r overlaps cache1 row r//2, which may belong to the other half and be unread by its silu. Allocate cache3 (or the y buffer) separately. Otherwise a correctness failure, or a silently different result, comes from the interaction, not the idea.
- triton-map-align-on-block-m-restack: the halves kernel is a re-parameterisation of its _align_small_kernel and keeps SGLang's block_size_m pick at bs<=32 (8 / 16). Any change in align cost (2-CTA grid, two stacked outputs) lands in the same 2.7-4.4 us row and is part of this hypothesis's measured cost, not a separate effect. Read the align row separately.
- post-expert-norm-triton-small-m: runs after the join on the full local_ids, which program 0 must write in full (remote routes as -1, both halves' routes as local ids). If program 0 writes only its half, the post-norm silently drops half B. Not a performance confounder.
- marlin-block-m-two-smaller-on-postnorm: touches only num_tokens >= 128, which this change leaves untouched. No confounding; its bs=128/256 rows are the in-run flat control.

### Interference and A/B needs
No companion warranted. The change is single-area and shape-gated, and bs=1/128/256 run master's code byte-for-byte in the same run, so a per-shape move at bs=8/16/32 beyond the paired-ratio noise is attributable to the two-stream split. A gain is unambiguous: splitting the halves serially on one stream can only add launches, so any net gain must come from overlap. A loss is mildly ambiguous between 'no overlap' and 'overlap smaller than the 5-8 us overhead'. A serialized-halves control (the same split on one stream) would separate the two, but it cannot change the decision: either way the follow-up is to drop the idea or to gate it to bs=32. With 2 moe checks left it is not worth a check. Noise: DECISIONS records unchanged bs=32 rows spanning 147-265 us across runs, so read only the paired ratio against SGLang within the run and the bs=8/16/32 rows, never the absolute Marlin row sum. The four Marlin rows will sum higher by construction, as the hypothesis notes.

### Open questions
- Does SGLang's moe_wna16_marlin_gemm launch with the device's max dynamic smem (one CTA per SM)? If so, two concurrent Marlin kernels cannot co-reside and the expected gain is about the kernel boundaries only. The profile's per-kernel durations will show it: if each half's Marlin row is about half of master's and they do not overlap in time, there was no concurrency.
- Split balance: under skewed routing a fixed 0-15/16-31 split can be lopsided. A follow-up could have the align kernel pick the halves on device: a greedy by-count assignment computed from the one-hot counts, which needs no host branch and so is legal. This is only worth trying if this run shows any overlap gain.
- If bs=32 gains and bs=8 loses, the hypothesis already plans an ab-tune gating to bs >= 16. Gating to bs=32 alone may be the right cut, since bs=16 has only about 12 experts.
- risk: high means this cannot be a scored run, and moe has 2 checks left today. Spend one check on it only if nothing better is queued (swapab-w13-silu-epilogue-verify is queued and unheld; compare priorities).


## Implementation notes

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

## Rules check

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
