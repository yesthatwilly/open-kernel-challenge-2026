---
problem: muse_prefill_layer
slug: experts-triton-first-turn-unmasked-loads-l2-hints
created: 2026-10-24T224437Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)"
change: >
  Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles,
  configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKED A LOADS: today a_lo/a_hi
  are tl.load(a_ptrs, mask=token_mask[None, :], other=0.0) every K step, where padding slots carry
  the sentinel id num_valid_tokens (row num_valid // TOP_K is out of bounds). Before the loop,
  compute safe_token = tl.where(token_mask, offs_token, 0) and build a_ptrs from safe_token //
  TOP_K, then load a_lo/a_hi with no mask (padding columns read row 0, a real row; their products
  land only in output columns the masked store already drops). This removes per-element predicates
  from the hot loop so the activation slabs load as unpredicated 16-byte cp.async. The output
  store keeps mask=token_mask. (2) L2 HINTS: weight codes b and scales s are read once per
  (expert, W tile) per token-tile group: load them with eviction_policy='evict_first'; the
  activation slabs are re-read by all N // BLOCK_W weight tiles of the group: load them with
  eviction_policy='evict_last'. Applies to every config (the kernel is shared), but only the T >
  4096 path runs it in the timed call. _check_expert_gemms runs unchanged (it covers padding
  slots, the -1 block and the past-n_padded block) and still logs regs/spills/smem per variant;
  compare them with master's log line.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel x2 at 8149-8873 us: unpredicated activation loads and keeping the 2 MB activation slabs resident in L2 across 64 (w13) / 32 (w2) weight tiles, est. 0-4% on the row, 0-2.5% per first_turn shape"
  - shapes: "*load=turn_*"
    direction: neutral
    why: Marlin path
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T034754Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T034755Z
base_sha: 62d28685d7829e2134746918148e95305d852b47
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-triton-first-turn-unmasked-loads-l2-hints
worktree: .worktrees/muse_prefill_layer/experts-triton-first-turn-unmasked-loads-l2-hints
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T034946Z
impl_commit: 9504288923f2bf57ab290450bdf2eb7e204a38eb
submission_sha256: cd367d224bbe6d7e8c670cb65df0c69236a004d25f9270596449f741260ec4e9
norm_sha256: c6d47ba7433f421ab2c6e3b1c9b2c33d6f8a7a0e20185351f9a26f13fff1ef43
submission_blob: 92d0b57884af6ce9d3a9770fd7e4f65fb49a7fd1
lint_warnings: 18
attempt_file: 2026-10-25T034946Z-check-experts-triton-first-turn-unmasked-loads-l2-hints.md
prepared_at: 2026-10-25T034946Z
job: aeda8b5a523642b1
dispatched_at: 2026-10-25T034953Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.0677050569173394
speedup_ci: [1.0516894810635793, 1.0837206327710993]
harvested_at: 2026-10-25T043404Z
best_at_harvest: 1.074386115190733
promotion: null
---

# experts-triton-first-turn-unmasked-loads-l2-hints

## Change

Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKED A LOADS: today a_lo/a_hi are tl.load(a_ptrs, mask=token_mask[None, :], other=0.0) every K step, where padding slots carry the sentinel id num_valid_tokens (row num_valid // TOP_K is out of bounds). Before the loop, compute safe_token = tl.where(token_mask, offs_token, 0) and build a_ptrs from safe_token // TOP_K, then load a_lo/a_hi with no mask (padding columns read row 0, a real row; their products land only in output columns the masked store already drops). This removes per-element predicates from the hot loop so the activation slabs load as unpredicated 16-byte cp.async. The output store keeps mask=token_mask. (2) L2 HINTS: weight codes b and scales s are read once per (expert, W tile) per token-tile group: load them with eviction_policy='evict_first'; the activation slabs are re-read by all N // BLOCK_W weight tiles of the group: load them with eviction_policy='evict_last'. Applies to every config (the kernel is shared), but only the T > 4096 path runs it in the timed call. _check_expert_gemms runs unchanged (it covers padding slots, the -1 block and the past-n_padded block) and still logs regs/spills/smem per variant; compare them with master's log line.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel x2 at 8149-8873 us: unpredicated activation loads and keeping the 2 MB activation slabs resident in L2 across 64 (w13) / 32 (w2) weight tiles, est. 0-4% on the row, 0-2.5% per first_turn shape
- `*load=turn_*` neutral: Marlin path

## Risk

Speculative tuning of the Triton first_turn engine, which experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) would replace; run only if that rewrite does not land or as a cheap hedge. Orthogonal to experts-triton-persistent-grid-first-turn (grid) and the dispatch hypotheses (what the grid contains), so it can stack with them. Packed asm dequant was flat (the kernel is not dequant-ALU bound), so if this is also flat the remaining cost is tl.dot's synchronous lowering, the rewrite's premise.

## Adversarial review

Reviewed 2026-10-25T034754Z by an adversarial subagent (model opus).

**Verdict: proceed.** Rules-clean, cheap and readable from the _moe_nvfp4_gemm_kernel rows, but both mechanisms are weak. The masked activation loads are already 16-byte vectorized, because the mask is constant along the contiguous K axis. DRAM traffic is about 0.2 TB/s, so L2 eviction priority has little to save. evict_first on weights also cuts against the kernel's own token-tile-fastest swizzle. Expect flat to a few %. Rebase on a81d19c and build an in-prepare() A/B of master / unmasked / unmasked+hints so that one check attributes the result.

### Justification
Profile (FINDINGS): _moe_nvfp4_gemm_kernel x2 is 8084-9174 us per first_turn shape (37-57% of the shape), and stays on Triton because the CUDA wgmma replacement is dead (LESSONS 2026-10-25: 1.8x slower, spills). The hedge condition in the risk note is therefore met. Persistent A/B (5d9ddbbaaffd4599 log): w13 3059 us and w2 1580 us at T=16384, BLOCK_T=256, with 74 live token tiles. The persistent grid was 2.9%/4.7% slower, so grid shape and dead CTAs are not the limiter, and the original launch order is worth something (L2 locality from the GROUP_SIZE_T swizzle). Rough roofline for w13: about 18.9K sorted rows x N 8192 x K 4096 x 2 is about 1.27 TFLOP in 3.06 ms, about 415 TFLOPS, which is tensor/issue-bound (synchronous tl.dot plus in-register dequant). DRAM is roughly 150 MB of activations plus the expert weights, about 0.2-0.3 TB/s, so L2 residency hints (which only change DRAM refetch) have little to save. L2-to-SM traffic is substantial (about 3 MB per CTA x 4736 CTAs), but evict hints do not reduce it. Mechanism (1): the a_lo/a_hi mask is token_mask[None, :], constant along dim 0, which is the K-contiguous axis (offs_h). Triton's axis analysis therefore already emits 16-byte cp.async with a zero-fill predicate, and dropping the mask saves only per-vector predicate setup, not vectorization. Mechanism (2): the kernel comment says 'token tiles fastest within a group, so the token tiles of one (expert, weight tile) run adjacent and the weight tile is re-read from L2'. evict_first on b/s therefore marks exactly the lines the swizzle is designed to re-hit, so the hint could cost rather than help. Activations already get a 64-way (w13) or 32-way (w2) reuse within a GROUP_SIZE_T=4 group of about 8 MB, which fits L2 without help. WEAKEST LINK: neither sub-change targets the measured limiter (tensor issue / dequant ALU interleave). The most plausible real effect is a side effect: the BLOCK_T=256 variant shows n_regs=254 with n_spills=4, and removing the mask and 'other' operands can change register allocation and remove those spills. That is a legitimate gain, but it is not the stated mechanism. The expected 0-2.5% per first_turn shape is below headline drift (CI ±1.5%), so judge only on the two kernel rows (Triton rows reproduce within ~1%).

### Confounders on the current kernel
- stack-tonight-five-prefill (landed, current master a81d19c): the spec says 'Built on master (635cfee)', but 635cfee is the submission blob of b92a0f4, one promotion behind. The stack changed the first_turn align (moe-align-32-experts-first-turn: different sorted_ids/padding layout and fewer -1 blocks) and turned w2 ZERO_NONLOCAL off. Both change what the K loop sees: the number of padding columns per tile, and early exits. Building on the stale base would measure a different padding mix, so it must rebase onto a81d19c.
- experts-triton-w4a16-first-turn-blockt256: it set BLOCK_T=256 at T > 4096 (254 regs, 4 spills). Any register change from removing the mask lands on a config that is at the register cap, so a gain may come from spill removal, not from load predication or L2. Read the n_regs/n_spills log line against master's (254/4) before attributing.
- experts-hybrid-marlin-turn-triton-first-turn: gates the Triton kernel to T > 4096 only; turn shapes stay on Marlin, so 'turn_* neutral' is structurally guaranteed.

### Interference and A/B needs
No separate companion run. The two sub-changes could have opposite signs (unmasked loads/spills help; evict_first on weights hurts the swizzle's reuse), so a flat result alone would be ambiguous. The cheap fix belongs inside this run, as the persistent-grid hypothesis already did: prepare() times master's kernel, unmasked-only and unmasked+hints at T=16384 for w13 and w2. Use graph replays with 7 alternations, print '[loads A/B] ... ratio' lines, ship the fastest variant only if its ratio is <= 0.98, and otherwise keep master's kernel. Prepare timings undercount the profile by 20-25% but rank correctly (LESSONS 2026-10-25). Also print n_regs/n_spills for each variant. That resolves attribution at zero extra checks; a companion run would only re-measure the same rows with more noise.

### Open questions
- Does Triton actually emit predicated 16-byte cp.async for the masked a loads today? Dump the TTGIR/PTX once in prepare() (ld/cp.async width) to settle mechanism (1).
- Do n_spills on the BLOCK_T=256 variant drop from 4 to 0 with unmasked loads? If yes, that is the likely source of any gain.
- Is evict_first on b/s harmful given the token-tile-fastest swizzle? The in-prepare three-way A/B answers this directly.


## Implementation notes

Edited only the worktree's muse_prefill_layer/submission.py (worktree HEAD 62d2868, the a81d19c lineage per the review).
- `_moe_nvfp4_gemm_kernel` gets two SEPARATE constexpr switches. `UNMASKED_A`: padding slots (sentinel id) are pointed at row 0 via `tl.where(token_mask, offs_token, 0) // TOP_K` and a_lo/a_hi load with no mask; the store keeps `mask=token_mask`. `L2_HINTS`: `eviction_policy="evict_last"` on the activation slabs only. Per the review, NO evict_first on the weights (b, s): it fights the token-tile-fastest swizzle.
- Variants are (UNMASKED_A, L2_HINTS) pairs: master (F,F), unmasked (T,F), unmasked+hints (T,T). `_nvfp4_moe_gemm` takes `loads=` and an optional `info` dict (n_regs/n_spills).
- `_check_expert_gemms` now iterates `_check_cases()`: every config with master loads, plus the T>4096 config with both new variants, so all are compiled and dequant-checked before capture.
- New `_loads_ab()` in prepare(), after the Marlin bank and before the e2e check (so the e2e check, whose real align has padding slots, runs the chosen variants): synthetic T=16384, real `moe_align_block_size`, first-turn config; for w13 and w2, CUDA-graph replays of the three variants, 7 alternations (plus a dropped warmup) with rotating order, 256 MB L2 flush before each timed replay, event-timed. Prints medians, ratio, n_regs, n_spills per variant and `PATH CHOSEN`. A variant is chosen per GEMM only if best and <= 0.98x master's median; else master. Any exception keeps master.
- `_experts_triton` passes `_loads_for(state, name, cfg)`: chosen variant on the first-turn config only, master loads otherwise. Marlin path untouched, so turn_* is structurally neutral.
- Deviations: decision is per GEMM (w13, w2 independently) rather than one joint pick. Hints are activations-only, as the review requires (spec's evict_first on weights dropped).
- Not run (no GPU): nothing verified at runtime. Padding-column garbage (row 0; for w2, a possibly unwritten cache2 row) stays in dropped output columns since acc[n, t] depends on A[t] only.

## Rules check

- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/sync/allocation in kernel_fn's path; `_loads_for` reads state and cfg identity only; all A/B syncs and graph captures are in prepare().
- [x] Nothing derived from `data` cached across calls: the choice comes from synthetic prepare() data; state holds only variant tuples.
- [x] Weights: unchanged, NVFP4 stays NVFP4 (dequant in kernel untouched).
- [x] Activations BF16 where SGLang keeps BF16: n/a to change, synthetic inputs are bf16; no dtype changes in the timed path.
- [x] Attention sees every key: n/a, no attention change.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff has none.
- [x] All work inside the timed call; no output caching or replay: kernels still run every call; the A/B only picks a constexpr variant in prepare().
- [x] No triton.autotune, no host sync in kernel_fn, first-line marker `# okc: problem=muse_prefill_layer` kept.

okc_lint (host-sync warnings in prepare() filtered; all are outside kernel_fn's call graph):
```
OK with warnings: .../experts-triton-first-turn-unmasked-loads-l2-hints/muse_prefill_layer/submission.py
```
py_compile OK; __pycache__ removed.

## Result

- job `aeda8b5a523642b1` (check), status done, passed True, tests 5/5
- speedup 1.0677x, 95% CI [1.0517, 1.0837], score 362837.936834828 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-first-turn-unmasked-loads-l2-hints.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-first-turn-unmasked-loads-l2-hints.patch`; branch `hyp/muse_prefill_layer/experts-triton-first-turn-unmasked-loads-l2-hints`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1101.29 | 1094.9 | 1.0016 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1529.55 | 1407.88 | 1.0018 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3318.71 | 3351.89 | 0.9991 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4849.27 | 4828.84 | 0.9981 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14705.32 | 14224.95 | 1.0013 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14264.64 | 14583.1 | 0.9902 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 17746.69 | 18299.63 | 0.9892 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21733.75 | 21712.71 | 0.9952 | flat |

no shape moved outside noise. Expected hit: 0/2. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-first-turn-unmasked-loads-l2-hints`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

