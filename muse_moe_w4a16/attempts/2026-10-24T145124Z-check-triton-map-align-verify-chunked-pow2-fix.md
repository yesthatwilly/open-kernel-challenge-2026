# muse_moe_w4a16 · 2026-10-24T145124Z · check · triton-map-align-verify-chunked-pow2-fix

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T144931Z-triton-map-align-verify-chunked-pow2-fix.md` (branch `hyp/muse_moe_w4a16/triton-map-align-verify-chunked-pow2-fix`).

- **Job:** `c1c72d5d19ea444f` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `a453496`

## Goal
Re-run triton-map-align-verify-chunked (failed at compile: tl.arange(0, BSM) with BSM=48 in prepare's warm-up raised 'arange's range must be a power of 2') with the one fix: _align_chunked_kernel takes a new constexpr BSM_P2 = triton.next_power_of_2(block_size_m) and builds the per-expert padding tile over tl.arange(0, BSM_P2) / [NUM_E, BSM_P2], keeping the existing mask j < padded - counts (always < BSM), so the stored slots are identical. Everything else is that branch's file byte for byte (chunked map+align for num_tokens > 32 replacing the expert_map gather and moe_align_block_size; the <=32 path is master's _align_small; self-check over 33/128/256/257 tokens).

## What changed, and where
Built by churnsubs: hyp/muse_moe_w4a16/triton-map-align-verify-chunked's submission.py (same base blob as current
master, d9a9375) plus the 4-line fix: `_align_chunked_kernel` gains constexpr `BSM_P2`, the padding tile uses
`tl.arange(0, BSM_P2)` and `[NUM_E, BSM_P2]` with the unchanged mask, and the launch passes
`BSM_P2=triton.next_power_of_2(block_size_m)`. The reviewer confirmed no other non-power-of-2 arange remains.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/triton-map-align-verify-chunked-pow2-fix diff` (the reviewed chunked-align diff plus the 4-line fix).

- [x] CUDA-graph capturable: branches on num_tokens/expert_map only; no `.item()`/`.cpu()`/`synchronize()` in kernel_fn.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: NVFP4 unchanged; Marlin repack untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV/query items n/a.
- [x] Attention sees every key: n/a; every local route aligned, remote routes dropped as on master.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First line marker kept; no triton.autotune.

okc_lint: OK with warnings (host-syncs only in prepare()/the self-check).

## Risks / what could fail
low
Minted by churnsubs to fix the compile error in a reviewed hypothesis; the fix is mechanical (power-of-2 tile + unchanged mask). The original review (proceed) is in the failed md.

## Result (after the run)
- job `c1c72d5d19ea444f` (check), status done, passed True, tests 5/5
- speedup 1.3107x, 95% CI [1.2910, 1.3303], score 10260.881311806997 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3116740840932535, "ci": [1.2919989728318546, 1.3313491953546521], "job": "bc56182a80444d16"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/triton-map-align-verify-chunked-pow2-fix.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/triton-map-align-verify-chunked-pow2-fix.patch`; branch `hyp/muse_moe_w4a16/triton-map-align-verify-chunked-pow2-fix`
