# muse_moe_w4a16 · 2026-10-25T054935Z · check · marlin-block-m-8-verify16

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T054545Z-marlin-block-m-8-verify16.md` (branch `hyp/muse_moe_w4a16/marlin-block-m-8-verify16`).

- **Job:** `d0811c27500c416a` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `5e32dd5`

## Goal
In _forward on master 9080a71, for 128 <= num_tokens <= 255 only (verify_16 bs=128 and the skewed bs=128 test), set block_size_m = 8 instead of master's two-smaller pick of 16; num_tokens >= 256 keeps 32 and T < 128 keeps SGLang's pick. Add a constant VERIFY16_BLOCK_SIZE_M = 8 and branch on the shape only. The same block size feeds _align_parallel (BSM=8, PAD_BLOCK=8) and the moe_block_size kwarg of both Marlin calls. prepare() must warm the _align_parallel BSM=8 variant (master warms one variant per block_size_m it can see above 32 tokens; add 8) and _align_self_check must cover (128, 8) against moe_align_block_size(local_ids, 8, 32) per expert route sets and block counts (capacity-only differences allowed, as master's check already logs). Mechanism: at bs=128 the measured trend is monotonic in block size: 48 -> 462 us (SGLang pick), 32 -> 382 us (next-smaller), 16 -> 340-344 us (two-smaller / cap-16), so Marlin's per-m-block cost is still falling at 16; with ~32 routes per local expert on average, 8-row blocks pad less on the cold experts. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## What changed, and where
Worktree base 1f8d1d2 equals master's submission.py (no diff vs master 0a3cc8a), so no merge was needed. Changes in muse_moe_w4a16/submission.py only (+11/-1):
- Added constants VERIFY16_BLOCK_SIZE_M = 8 and VERIFY16_MAX_TOKENS = 255.
- _forward: inside the `num_tokens >= 128` branch, after the two-smaller pick, `if num_tokens <= VERIFY16_MAX_TOKENS: block_size_m = 8`. Branches on num_tokens only. 256/257 keep 32; T < 128 untouched. The same block_size_m feeds _align_parallel (BSM=8, PAD_BLOCK=8) and both Marlin moe_block_size kwargs.
- prepare(): the _align_parallel warm loop is now (8, 16, 32, 48, 64).
- _align_self_check: adds shape (128, 8) (for tokens <= 255 with tokens >= 128). Mismatch classification is unchanged (": capacity " counted as capacity-only), and big_cases counts it automatically.
- Review items: no spec corrections; the open questions (w13/w2 split in the profile, do-not-retry note on a flat result, queue priority) are for the profile/closing stage, not the build. No deviations.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only host-side Python int comparisons in _forward; no sync/print/.item in kernel_fn.
- [x] Nothing derived from `data` is cached: only num_tokens (shape) branch; no new state.
- [x] Weights untouched: n/a, no weight changes; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] Attention: n/a, MoE; every route is still processed (block size only changes padding).
- [x] No timing patches, harness access, module edits, reference import, or child processes in the diff.
- [x] All work inside the timed call; no output caching or replay.
- [x] No module/static globals hold device pointers or buffers (only int constants added).

okc_lint: OK with warnings (host-sync warnings only in prepare()/self-checks, outside kernel_fn). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Risk: at bs=32 a uniform 8 (vs 16) was flat-to-worse (215 vs 208), so the trend may stop at 16 for hot experts that now need 2x the m-blocks. Independent of the wgmma line: if wgmma lands at B128 this becomes moot there.

## Result (after the run)
- job `d0811c27500c416a` (check), status done, passed True, tests 5/5
- speedup 1.2992x, 95% CI [1.2797, 1.3187], score 10351.220587921769 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.result.json
- **outcome: regression** (regression)
- files: `muse_moe_w4a16/hypotheses/done/regression/marlin-block-m-8-verify16.result.json`, `muse_moe_w4a16/hypotheses/done/regression/marlin-block-m-8-verify16.patch`; branch `hyp/muse_moe_w4a16/marlin-block-m-8-verify16`
