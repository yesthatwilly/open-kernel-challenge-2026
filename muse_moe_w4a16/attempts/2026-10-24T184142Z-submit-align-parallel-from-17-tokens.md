# muse_moe_w4a16 · 2026-10-24T184142Z · submit · align-parallel-from-17-tokens

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T183702Z-align-parallel-from-17-tokens.md` (branch `hyp/muse_moe_w4a16/align-parallel-from-17-tokens`).

- **Job:** `ed6d884240084d9c` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `c9c2db8`

## Goal
One constant plus its warm-up: lower ALIGN_FUSED_MAX_TOKENS from 32 to 16, so 17 <= num_tokens <= 32 (decode_32 bs=32) runs master's existing 33-program _align_parallel_kernel instead of the single-program _align_small_kernel; num_tokens <= 16 (bs=1, 4, 8, 16) keeps _align_small and num_tokens > 32 keeps _align_parallel, both unchanged. block_size_m at 17..32 is SGLang's pick (16), and _align_parallel already compiles a BSM=16 variant in prepare() (bsm loop 16/32/48/64), so no new variant is needed; the _align_small warm-up loop range(1, ALIGN_FUSED_MAX_TOKENS + 1) shrinks to 1..16 automatically. _align_self_check needs no edit: it routes tokens <= ALIGN_FUSED_MAX_TOKENS to _align_small and the rest to _align_parallel, so T=32 is now checked against moe_align_block_size on the parallel kernel; its tokens > ALIGN_FUSED_MAX_TOKENS summary line will then also count T=32 cases (expected 0 mismatches apart from the known capacity-only lines; never gate on the self-check). Update the comment next to the constant. Mechanism: at bs=32 _align_small does a [256, 32] one-hot cumsum, a [64, 32] block scan and a 1024-wide tail store in ONE program on one SM (4.2-4.4 us, 3x its 1.4 us at bs=1); _align_parallel spreads the placement over 33 programs and at n=256 each does a single CHUNK=512 pass, so it should run at or below its 4.0 us at n=1024 (bs=128).

## What changed, and where
In the worktree's muse_moe_w4a16/submission.py: ALIGN_FUSED_MAX_TOKENS 32 -> 16 with its comment extended; _align_parallel warm-up bsm loop in prepare() now (8, 16, 32, 48, 64), per the adversarial review (17..28 tokens get block_size_m 8), comment updated. Nothing else touched (stale "tokens <= 32" wording in the _align_self_check docstring and the "tokens > 32" comment left as is to keep the diff minimal; the self-check logic keys off the constant). Dummy warm-up ids use ALIGN_FUSED_MAX_TOKENS + 1 = 17 tokens; n is a runtime arg so the compiled variant is the same.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff touches only a constant and the prepare() warm-up tuple; no syncs/allocs added to kernel_fn.
- [x] Nothing derived from `data` is cached across calls: no state added.
- [x] Weights: n/a: no weight change.
- [x] Activations BF16 / KV FP8 / queries: n/a: untouched.
- [x] Attention sees every key: n/a: untouched.
- [x] No timing patches, harness I/O, module edits, reference import, subprocesses or network: diff has none.
- [x] All work inside the timed call; no output caching or replay: align kernels run per call as before.

Lint (okc_lint.py): OK with warnings (8 host-sync warnings, all in prepare()/self-check, outside kernel_fn's call graph). py_compile OK; __pycache__ removed.

## Risks / what could fail
low
A pure kernel-choice A/B at one shape: compare _align_parallel_kernel's bs=32 row with _align_small_kernel's 4.2-4.4 us in master runs. Headline effect is under noise and decode_32 Marlin rows are seed-noisy, so the kernel row is the evidence. If it wins, try 9..16 next. Overlap: the running cuda-dispatch-combine-pdl-verify replaces _align_parallel only at >32 (host branch on num_tokens > 32, independent of this constant? the implementer must keep that rewrite's gate untouched; here it is not present on master). The queued midm single-launch rewrite would replace the align at 2..32. The running swap-AB changes block_size_m at bs<=32 to 8*NT (32 at bs=32), so if it lands this would need a BSM=32 parallel variant (already compiled).

## Result (after the run)
- job `ed6d884240084d9c` (submit), status done, passed True, tests 5/5
- speedup 1.3250x, 95% CI [1.3051, 1.3449], score 10149.82427479356 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-parallel-from-17-tokens.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-parallel-from-17-tokens.patch`; branch `hyp/muse_moe_w4a16/align-parallel-from-17-tokens`
