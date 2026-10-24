---
problem: muse_moe_w4a16
slug: triton-map-align-verify-chunked
created: 2026-10-24T140229Z
base: master
area: MoE align/dispatch
change: >
  Extend master's single-program map+align (_align_small_kernel, triton-map-align-small-m-port) to
  num_tokens > 32 (bs=128/256, and the bs=33/257/skewed-128 tests) with a chunked variant
  _align_chunked_kernel, so the expert_map gather (at::native index_elementwise ~3.2 us), the
  align wrapper's fill (unrolled_elementwise ~2.8 us), moe_align_block_size_kernel (~2.7 us) and
  count_and_sort_expert_tokens_kernel (1.8-2.6 us) become one launch. One program, num_warps=8,
  CHUNK=512 routes (constexpr), NUM_E=32. Pass 1 loops over chunks of the n=M*8 flat ids: it
  computes local = id - local_start (masked to [0, count), else -1), stores local_ids, and
  accumulates counts[NUM_E] += sum(onehot [CHUNK, NUM_E]). Then compute padded / seg_start / total
  exactly as the small kernel does. Pass 2 loops over the chunks again, keeping a running
  base[NUM_E] (init 0): pos = seg_start[e] + base[e] + (within-chunk cumsum rank), it stores
  sorted_ids[pos] = flat index for local routes, and does base += chunk counts. Then a loop writes
  each expert's padding slots with the sentinel n, a CHUNK-strided loop fills slots [total, cap),
  and a loop over m-blocks in CHUNK-sized strides writes block_experts (-1 past the used blocks)
  and n_padded. n, cap and n_blocks are runtime (do_not_specialize), so the only constexpr per
  call is BSM. prepare() compiles it once per block_size_m candidate (8..64) on dummy ids and
  extends _align_self_check to tokens (33, 128, 256, 257) with the same per-expert comparison
  against moe_align_block_size(ref_local, bsm, 32). kernel_fn: replace num_tokens <=
  ALIGN_FUSED_MAX_TOKENS with expert_map is not None, using _align_small up to 32 tokens
  (unchanged) and _align_chunked above. block_size_m picks, Marlin calls, silu and post-norm are
  unchanged.
expected:
  - shapes: bs=128 *
    direction: faster
    why: 4 align/gather launches totalling ~10.5 us replaced by one ~3-5 us launch, out of 333 us
  - shapes: bs=256 *
    direction: faster
    why: 4 launches totalling ~11.5 us replaced by one, out of 470 us
  - shapes: bs=32 *
    direction: flat
    why: "control: _align_small unchanged"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T140447Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T140447Z
base_sha: 001aa1c251231981cfc4254d1cf8ce63b737a236
base_blob: d9a9375ef4721ab9b28cc9d6364c92e67c7ccc59
branch: hyp/muse_moe_w4a16/triton-map-align-verify-chunked
worktree: .worktrees/muse_moe_w4a16/triton-map-align-verify-chunked
mode: check
implemented_at: 2026-10-24T140755Z
impl_commit: eb5d866f75bc4c23be16cdeedc1948ad59abf2d2
submission_sha256: 218d5b05ef911cba6b2b4069b9bee0d0fd87ed8271be4e0c7b55eb62088bad48
norm_sha256: a95d879e69161f8fb99d50100cad723f96c72e0b19c830cf6150a714aa34357f
submission_blob: dd6c55be3fb9c7ca9e51737d697a4edabe866330
lint_warnings: 8
attempt_file: 2026-10-24T140755Z-check-triton-map-align-verify-chunked.md
prepared_at: 2026-10-24T140755Z
job: 3aa4d7e193244255
dispatched_at: 2026-10-24T140804Z
outcome: failed
verdict_detail: "test:failed"
speedup: null
speedup_ci: null
harvested_at: 2026-10-24T144810Z
best_at_harvest: 1.3116740840932535
promotion: null
---

# triton-map-align-verify-chunked

## Change

Extend master's single-program map+align (_align_small_kernel, triton-map-align-small-m-port) to num_tokens > 32 (bs=128/256, and the bs=33/257/skewed-128 tests) with a chunked variant _align_chunked_kernel, so the expert_map gather (at::native index_elementwise ~3.2 us), the align wrapper's fill (unrolled_elementwise ~2.8 us), moe_align_block_size_kernel (~2.7 us) and count_and_sort_expert_tokens_kernel (1.8-2.6 us) become one launch. One program, num_warps=8, CHUNK=512 routes (constexpr), NUM_E=32. Pass 1 loops over chunks of the n=M*8 flat ids: it computes local = id - local_start (masked to [0, count), else -1), stores local_ids, and accumulates counts[NUM_E] += sum(onehot [CHUNK, NUM_E]). Then compute padded / seg_start / total exactly as the small kernel does. Pass 2 loops over the chunks again, keeping a running base[NUM_E] (init 0): pos = seg_start[e] + base[e] + (within-chunk cumsum rank), it stores sorted_ids[pos] = flat index for local routes, and does base += chunk counts. Then a loop writes each expert's padding slots with the sentinel n, a CHUNK-strided loop fills slots [total, cap), and a loop over m-blocks in CHUNK-sized strides writes block_experts (-1 past the used blocks) and n_padded. n, cap and n_blocks are runtime (do_not_specialize), so the only constexpr per call is BSM. prepare() compiles it once per block_size_m candidate (8..64) on dummy ids and extends _align_self_check to tokens (33, 128, 256, 257) with the same per-expert comparison against moe_align_block_size(ref_local, bsm, 32). kernel_fn: replace num_tokens <= ALIGN_FUSED_MAX_TOKENS with expert_map is not None, using _align_small up to 32 tokens (unchanged) and _align_chunked above. block_size_m picks, Marlin calls, silu and post-norm are unchanged.

## Expected

- `bs=128 *` faster: 4 align/gather launches totalling ~10.5 us replaced by one ~3-5 us launch, out of 333 us
- `bs=256 *` faster: 4 launches totalling ~11.5 us replaced by one, out of 470 us
- `bs=32 *` flat: control: _align_small unchanged
- `bs=1 *` flat: control: untouched

## Risk

Expected ~1.5-2% on each verify shape, ~0.5-0.7% headline, inside one run's CI. Judge it by the removed rows in top_kernels and the new kernel's own time. It is a launch-count change on 330-470 us shapes, which LESSONS says rarely moves the headline. It is still worth a run because it also gives the bs>=128 path a Triton align that a two-class block-m (see marlin-two-class-block-m-decode32) or the swap-AB rewrite can reuse. Risk: the serial chunk loops in one CTA (5 chunks at bs=257) could take 5+ us and leave little gain. If the kernel row exceeds ~6 us, the result is flat. Confounding: swapab-nvfp4-gemm-verify-rewrite replaces Marlin at the same shapes and keeps SGLang's align, so if both land the chunked align must be called with BSM=BLOCK_T (64/128): add those to the prepare compile and self-check.

## Adversarial review

Reviewed 2026-10-24T140447Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. This is a sound, rules-clean extension of the landed single-program align to bs>32. One run can be read from the per-shape top_kernels rows: 4 rows removed, 1 new row added, and the code at bs<=32 is byte-identical, so those shapes act as a drift control. The real risk is that a serial two-pass loop in one CTA at n=2048 ends up as slow as the ~11.5 us it replaces.

### Justification
The profile of the current best (bc56182a80444d16) confirms the target rows exactly. At bs=128: index_elementwise 3.2, unrolled_elementwise (sorted_ids fill) 2.8, moe_align_block_size_kernel 2.7 and count_and_sort 1.8 us, 10.5 us in all. At bs=256: 3.3 + 2.8 + 2.8 + 2.6 = 11.5 us. Against 333/470 us, a full win is ~2-2.5% per shape and a realistic win is 1.5-2%, ~0.5% on the headline: inside one run's CI, as the spec admits. TASK.md gives every load equal weight, so 2 of the 6 loads moving ~1.5-2% is consistent with the expected directions. bs=1/8/16/32 are correctly predicted flat, because they keep _align_small. Weakest link: the new kernel's own time. The landed _align_small scales with n: 1.6 us (n=8), 2.7 (n=64), 2.9 (n=128), 4.4 (n=256, 4 warps, one [256,32] tile with cumsum). Extrapolating, the per-route cost is ~1-1.5 us per 128 routes, which mostly comes from the onehot cumsum/rank in pass 2. The chunked kernel at n=1024 (bs=128) and n=2048 (bs=256, 4 chunks of 512) runs pass 1 plus pass 2 serially in one CTA, with a loop-carried base[] dependency. 8 warps roughly halves the per-element cost, but a plausible range is ~5-8 us at bs=128 and ~8-14 us at bs=256. bs=128 should gain. bs=256 could come out flat or even slightly slower. The speed-of-light framing does not apply: this is launch and latency work, far from any bandwidth floor. Minor: the graph-node gap saved by removing 3 launches is only ~0.3-1 us in total (LESSONS).

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: sets BSM=16 at bs=128 and BSM=32 at bs=256/257, which fixes cap (1504 / 3040), the padding-slot count and n_blocks (94 / 95). The chunked kernel's padding, cap-fill and block loops therefore cost what they cost at those BSMs. At SGLang's 48/64 the padding loops would be ~3x larger. This shifts the new kernel's own time somewhat, but not what Marlin receives, so it does not confound attribution of the align rows.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: same area and same kernel design. They gate on ALIGN_FUSED_MAX_TOKENS=32, so the bs>32 path this hypothesis replaces was never touched by them. They are the in-run control (bs<=32 code unchanged), not a confounder.
- post-expert-norm-triton-small-m: no confound at the target shapes. bs>=32 still uses SGLang's fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids, which needs remote slots to be exactly -1. The new kernel must reproduce that, as _align_small does.
- sglang-marlin-path-parity: the base pipeline. The 32-expert align capacity (M*8 + 32*(bsm-1)) that the chunked kernel must match comes from it. No interaction beyond that.

### Interference and A/B needs
No companion is warranted. Attribution needs no A/B: the change is visible directly in top_kernels at bs=128/256 (the four at::native/align rows disappear and _align_chunked_kernel appears with its own time). The four bs<=32 shapes are byte-identical code and serve as the drift reference. Reverting block-m-two-smaller (ab-revert) would only change BSM, which changes the kernel's padding-loop cost by about a microsecond and none of Marlin's work. It cannot flip the interpretation, and spending a check on it would be wasted with moe at 4 checks left. A CHUNK/num_warps ab-tune is a follow-up, not a companion: if the kernel row shows bs=128 winning and bs=256 losing, the next step is a gate at n<=1024 or a CHUNK=256 / 2-CTA variant, chosen from this run's rows. Read the result per shape: success means kernel row < ~(sum of removed rows - 1 us) at each verify shape. The headline is expected to stay inside the CI either way.

### Open questions
- What is the new kernel's own time at bs=256 (n=2048, 4 serial chunks x 2 passes)? If it exceeds ~10 us, the bs=256 gain disappears. Would CHUNK=256 with num_warps=8 (lower register pressure for the [CHUNK,32] int32 onehot and cumsum: 512x32 = 16K elements, 64 per thread at 8 warps), or recomputing the onehot only in pass 2, be faster?
- Is unrolled_elementwise (2.8 us) really the wrapper's sorted_ids fill and not some other cast? topk_ids is already int32, so .to(int32) is a no-op. If it were something else that stays, the saving is smaller.
- If the swap-AB rewrite later needs BSM=64/128 from this kernel, the [NUM_E, BSM] padding tile grows to 4K int32. That is fine for one CTA, but add those BSMs to the prepare compile list and the self-check, as the spec notes.



## Implementation notes

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

## Rules check

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

## Result

- job `3aa4d7e193244255` (check), status done, passed False, tests 0/0
- best at harvest: {"speedup": 1.3116740840932535, "ci": [1.2919989728318546, 1.3313491953546521], "job": "bc56182a80444d16"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: failed** (test:failed)
- files: `muse_moe_w4a16/hypotheses/done/failed/triton-map-align-verify-chunked.result.json`, `muse_moe_w4a16/hypotheses/done/failed/triton-map-align-verify-chunked.patch`; branch `hyp/muse_moe_w4a16/triton-map-align-verify-chunked`

## Follow-ups

- none

