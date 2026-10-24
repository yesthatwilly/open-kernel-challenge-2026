---
problem: muse_moe_w4a16
slug: align-parallel-per-expert-verify
created: 2026-10-24T164817Z
base: master
area: MoE align/dispatch
change: >
  For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when
  expert_map is not None, replace master's else-branch (topk_ids.to(int32), the
  expert_map[local_ids] gather = at::native index_elementwise 3.2 us, the at::native
  unrolled_elementwise 2.8 us, and moe_align_block_size's moe_align_block_size_kernel +
  count_and_sort_expert_tokens_kernel: ~10.5 us of kernels and 4 graph nodes in master's
  bs=128/256 profile) with ONE launch of a new multi-program Triton kernel _align_parallel_kernel,
  grid (NUM_E + 1,) = 33 programs, num_warps 4. Every program loops over the n = M*8 flat route
  ids in chunks of CHUNK = 1024 (tl.arange(0, CHUNK), masked), maps each id to local = id -
  local_start if 0 <= local < local_count else -1 (as _align_small_kernel does), and accumulates
  per-expert counts for all 32 local experts (one-hot [CHUNK, 32] summed over axis 0). Program e <
  32 then computes padded = ceil(count/BSM)*BSM for all experts, its own exclusive seg_start
  (cumsum of padded, minus its own), and in a second pass over the chunks stores route index i at
  sorted_ids[seg_start + running_rank + in-chunk rank] for every route with local == e (in-chunk
  rank = tl.cumsum of the match mask minus 1; running_rank += the chunk's match count). It stores
  the sentinel n into its padding slots [seg_start+count, seg_start+padded) using tl.arange(0,
  PAD_BLOCK) with PAD_BLOCK = next_power_of_2(BSM), masked to padded-count (never tl.arange(0,
  BSM): BSM 48 occurs at M 116..127 and is what broke triton-map-align-verify-chunked), and writes
  block_experts[b] = e for b in [seg_start/BSM, (seg_start+padded)/BSM). Program 32 writes
  local_ids for all n routes (int32, same values as master), the sentinel n into every slot in
  [total, cap), -1 into block_experts[b] for b in [total/BSM, n_blocks), and n_padded = total.
  Capacities exactly as master's 32-expert moe_align_block_size allocates: cap = n + 32*(BSM-1),
  n_blocks = cdiv(cap, BSM). The store sets are disjoint across programs, so no atomics or
  ordering are needed. BSM is a constexpr; n, cap, n_blocks, local_start, local_count are
  do_not_specialize. The block_size_m pick is unchanged (two-smaller at >= 128). In prepare(),
  compile the kernel for BSM in (16, 32, 48, 64) on dummy ids and extend _align_self_check with T
  in (33, 57, 120, 128, 256, 257) using the same per-expert-set / block-count / sentinel / layout
  checks against moe_align_block_size(ref_local, bsm, 32). num_tokens <= 32 keeps
  _align_small_kernel; Marlin, silu_and_mul and post-norm are unchanged.
expected:
  - shapes: bs=128 *
    direction: faster
    why: 4 align-path launches (~10.5 us of kernels plus their graph nodes) become one ~3-4 us 33-CTA launch; the single-program chunked kernel (16.5 us) was already flat-to-better here (paired 1.403 vs 1.375), so the removed node overhead is real and a fast kernel should net ~7-12 us of 333 us
  - shapes: bs=256 *
    direction: faster
    why: same 4 launches (~11.5 us) replaced; the single-program chunked kernel took 29.2 us and lost only ~1%, so a 33-program kernel at ~4-5 us should net ~8-15 us of 470 us
  - shapes: bs=32 *
    direction: flat
    why: "control: _align_small path untouched"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T165146Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T165146Z
base_sha: 4a3b474ab343532548aeff7b0908beb14a4fae68
base_blob: d9a9375ef4721ab9b28cc9d6364c92e67c7ccc59
branch: hyp/muse_moe_w4a16/align-parallel-per-expert-verify
worktree: .worktrees/muse_moe_w4a16/align-parallel-per-expert-verify
mode: check
implemented_at: 2026-10-24T165513Z
impl_commit: 6c12cfed524c55040c6f5a306635414d6ee7b367
submission_sha256: d2687e7d6e1d8b529e5b45491756a4fc3769da4951f05ed31f15b5431c65033c
norm_sha256: af3d5c9f0c4ec4c9a67770ac2d78c0949c1a32203ab8362f4169efa81716ae0e
submission_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
lint_warnings: 8
attempt_file: 2026-10-24T165514Z-check-align-parallel-per-expert-verify.md
prepared_at: 2026-10-24T165514Z
job: 45eacf4ca2e849d2
dispatched_at: 2026-10-24T165523Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3249153782392282
speedup_ci: [1.30504164756564, 1.3447891089128166]
harvested_at: 2026-10-24T173234Z
best_at_harvest: 1.3075759186466844
promotion: merged
hand_landed: true
promotion_commit: 1a5da3d22ce48904de76852a1d585563f4c53048
---

# align-parallel-per-expert-verify

## Change

For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when expert_map is not None, replace master's else-branch (topk_ids.to(int32), the expert_map[local_ids] gather = at::native index_elementwise 3.2 us, the at::native unrolled_elementwise 2.8 us, and moe_align_block_size's moe_align_block_size_kernel + count_and_sort_expert_tokens_kernel: ~10.5 us of kernels and 4 graph nodes in master's bs=128/256 profile) with ONE launch of a new multi-program Triton kernel _align_parallel_kernel, grid (NUM_E + 1,) = 33 programs, num_warps 4. Every program loops over the n = M*8 flat route ids in chunks of CHUNK = 1024 (tl.arange(0, CHUNK), masked), maps each id to local = id - local_start if 0 <= local < local_count else -1 (as _align_small_kernel does), and accumulates per-expert counts for all 32 local experts (one-hot [CHUNK, 32] summed over axis 0). Program e < 32 then computes padded = ceil(count/BSM)*BSM for all experts, its own exclusive seg_start (cumsum of padded, minus its own), and in a second pass over the chunks stores route index i at sorted_ids[seg_start + running_rank + in-chunk rank] for every route with local == e (in-chunk rank = tl.cumsum of the match mask minus 1; running_rank += the chunk's match count). It stores the sentinel n into its padding slots [seg_start+count, seg_start+padded) using tl.arange(0, PAD_BLOCK) with PAD_BLOCK = next_power_of_2(BSM), masked to padded-count (never tl.arange(0, BSM): BSM 48 occurs at M 116..127 and is what broke triton-map-align-verify-chunked), and writes block_experts[b] = e for b in [seg_start/BSM, (seg_start+padded)/BSM). Program 32 writes local_ids for all n routes (int32, same values as master), the sentinel n into every slot in [total, cap), -1 into block_experts[b] for b in [total/BSM, n_blocks), and n_padded = total. Capacities exactly as master's 32-expert moe_align_block_size allocates: cap = n + 32*(BSM-1), n_blocks = cdiv(cap, BSM). The store sets are disjoint across programs, so no atomics or ordering are needed. BSM is a constexpr; n, cap, n_blocks, local_start, local_count are do_not_specialize. The block_size_m pick is unchanged (two-smaller at >= 128). In prepare(), compile the kernel for BSM in (16, 32, 48, 64) on dummy ids and extend _align_self_check with T in (33, 57, 120, 128, 256, 257) using the same per-expert-set / block-count / sentinel / layout checks against moe_align_block_size(ref_local, bsm, 32). num_tokens <= 32 keeps _align_small_kernel; Marlin, silu_and_mul and post-norm are unchanged.

## Expected

- `bs=128 *` faster: 4 align-path launches (~10.5 us of kernels plus their graph nodes) become one ~3-4 us 33-CTA launch; the single-program chunked kernel (16.5 us) was already flat-to-better here (paired 1.403 vs 1.375), so the removed node overhead is real and a fast kernel should net ~7-12 us of 333 us
- `bs=256 *` faster: same 4 launches (~11.5 us) replaced; the single-program chunked kernel took 29.2 us and lost only ~1%, so a 33-program kernel at ~4-5 us should net ~8-15 us of 470 us
- `bs=32 *` flat: control: _align_small path untouched
- `bs=1 *` flat: control: untouched

## Risk

Not a rerun of triton-map-align-verify-chunked(-pow2-fix) (neutral 1.311x): that kernel was ONE program streaming every route serially (16.5 us at bs=128, 29.2 us at bs=256, slower than the 4 kernels it replaced), yet both verify shapes held, so the launches it removed were worth roughly its extra kernel time. This design parallelises over experts so the kernel itself should be ~3-5 us and the node saving shows. Judge on the bs=128/256 paired ratios and on the new kernel row vs the four removed rows (top_kernels). Risks: each program re-reads all n ids (8 KB at bs=256, L2-resident after the first program) and builds a [1024, 32] one-hot per chunk; if the kernel exceeds ~6 us the gain vanishes. Must pass bs=33 (BSM 16), bs=257 (BSM 32) and skewed-128 tests; BSM 48 must never reach tl.arange.

## Adversarial review

Reviewed 2026-10-24T165146Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, rules-clean redesign of the chunked align. It parallelises over experts and removes the serial single-CTA bottleneck that made _align_chunked_kernel 16.5/29.2 us. The realistic gain is 1.5-2.5% per verify shape and about 0.5% on the headline, so the run will almost certainly be classified as neutral. Judge it on the new kernel row against the four removed rows in top_kernels, and fix the [1024,32] one-hot register blow-up before building.

### Justification
Mechanism: at bs=128/256, master's else-branch runs at::native index_elementwise (3.2/3.3 us), the align wrapper's unrolled_elementwise fill (~2.8 us), moe_align_block_size_kernel (~2.7 us) and count_and_sort (~1.8-2.6 us). That is about 10.5/11.5 us against 333/470 us of shape time. Marlin is 94-95% of each verify shape (316/490 us vs floors of 101/109 us), so this area is capped at ~3.1%/~2.4% per shape even for a free kernel. With a 3-5 us kernel and ~1 us of saved graph-node cost (LESSONS: ~0.3 us per single Triton launch node), expect ~6-8 us, i.e. ~2% at bs=128 and ~1.5% at bs=256. Over the 6 equally weighted loads that is ~0.5-0.6% on the geomean, inside the +-1.5% CI and the 1.5-2.5% per-shape drift. The expected shapes and directions are right: faster at bs=128/256, flat at bs<=32, where _align_small is byte-identical and serves as the in-run drift control. The design parallelism is plausible. Each of the 33 programs reads only 2-8 KB of ids, and pass 2 needs only a 1-D match mask and cumsum, unlike the chunked kernel's [CHUNK,32] cumsum. Weakest link 1 (the argument): the claim that 'the removed node overhead is real' rests on the chunked run's paired ratios, bs=128 1.403 vs 1.375 (+2.0%) and bs=256 1.301 vs 1.314 (-1.0%). Both are inside drift, and the Marlin row itself moved 316->337 and 490->462 us between those runs. So the hidden ~6 us node saving the expected section leans on is unproven; budget ~1 us. Weakest link 2 (the build): CHUNK=1024 with num_warps=4 makes the [1024,32] int32 one-hot 256 live values per thread per temporary, 4x the chunked kernel's 64. That will likely spill and could push the kernel past the ~6 us break-even. Use CHUNK=512 with num_warps=8, or CHUNK=256 with num_warps=4 (64 per thread), or tl.histogram over remote-remapped ids if this Triton version supports a mask. Also: each program's own block_experts range can span up to cdiv(M, BSM) blocks (8 at bs=128 BSM 16 and at bs=256 BSM 32; more at skewed-128), so it needs a loop or a power-of-2 arange sized for that bound. The spec leaves this unstated. Extract seg_start[e] and count[e] with a where+sum over the expert axis, and cast the bool match mask to int32 before tl.cumsum.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: sets BSM=16 at bs=128 and 32 at bs=256/257. That fixes cap, n_blocks, the padding width and each expert's block count, so it shifts the new kernel's own time slightly (well under 1 us). It does not change what Marlin receives, so it does not confound attribution of the removed align rows. Without it, BSM would be 48/64, and PAD_BLOCK=64 would matter more.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: same area and same design family, but gated at <=32 tokens. Those shapes are this run's byte-identical control, not a confounder. Latent hazard: _align_small_kernel still uses tl.arange(0, BSM) unpadded, which is safe only because its BSM is <=16.
- post-expert-norm-triton-small-m: no timing confound at the target shapes. bs>=32 still calls fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids, and silu_and_mul uses expert_ids=local_ids. Program 32's local_ids must be exactly -1 for remote routes and int32 contiguous, as on master. This is a correctness dependency, not an attribution one.
- sglang-marlin-path-parity: provides the base pipeline and the 32-expert capacity contract (cap = n + 32*(BSM-1), n_blocks = cdiv(cap, BSM), sentinel n). It does not interact beyond the sizes the kernel must reproduce.

### Interference and A/B needs
No companion is warranted. Nothing landed sits between the change and its measurement. The four removed rows and the one added row are directly visible in top_kernels, and Triton kernel times reproduce within ~1% between runs, so the kernel-level comparison is interpretable from this single run. The headline and per-shape ratios are dominated by Marlin drift, which no A/B companion fixes. An ab-revert of block-m-two-smaller would change only BSM and the padding cost and could not flip the reading. A CHUNK/num_warps ab-tune is a follow-up to choose from this run's kernel row, not a disambiguator. Decision rule for the harvest: merge by hand if the log shows the self-check with 0 mismatches for T in {33..257}, _align_parallel_kernel at or below (sum of removed rows - ~2 us) at both verify shapes, and the bs=128/256 paired ratios at or above master's, even if classify says neutral. If the kernel row exceeds ~8 us at bs=256, gate it to bs<=128 or shrink CHUNK in a follow-up.

### Open questions
- The new kernel's own time at bs=256 (n=2048, 2 chunks x 2 passes per program at CHUNK=1024). Check the compile log or register report for spills at [1024,32] int32 with 4 warps. If the build keeps CHUNK=1024, consider num_warps=8.
- skewed-128: the busiest expert can receive up to M routes (128, so 8 blocks at BSM 16). Confirm the per-program block_experts and route-store loops are bounded by a runtime loop, not a fixed small arange.
- Does this image's Triton support tl.histogram with a mask? It would replace the one-hot entirely in pass 1 (map remote routes to bin 32 of a 64-bin histogram if there is no mask).
- Prepare-time compile: four BSM variants plus 6 more self-check T values. Expected seconds, well inside 300 s. Confirm every BSM variant (16/32/48/64) compiled before trusting the timings, given the chunked kernel's arange history.
- With moe at 2 checks left today, the expected classify outcome is neutral (about 0.5% headline). Is the second check better kept for a follow-up tune of this kernel than for a held item?


## Implementation notes

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

## Rules check

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

## Result

- job `45eacf4ca2e849d2` (check), status done, passed True, tests 5/5
- speedup 1.3249x, 95% CI [1.3050, 1.3448], score 10150.459584726568 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3075759186466844, "ci": [1.287962279866984, 1.3271895574263846], "job": "3cabc739f1af493e"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch`; branch `hyp/muse_moe_w4a16/align-parallel-per-expert-verify`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.37 | 48.82 | 0.9817 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 147.87 | 148.37 | 1.0027 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 200.73 | 199.88 | 0.9996 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.77 | 107.74 | 1.003 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 333.39 | 318.26 | 1.0427 | gained |
| bs=256 load=verify_32 sharding=ep8 | 60 | 469.91 | 454.28 | 1.0321 | gained |

gained on verify_16 verify_16; verify_32 verify_32. Expected hit: 4/4. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-parallel-per-expert-verify`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

