---
problem: muse_moe_w4a16
slug: align-parallel-from-17-tokens
created: 2026-10-24T183702Z
base: master
area: MoE align/dispatch
change: >
  One constant plus its warm-up: lower ALIGN_FUSED_MAX_TOKENS from 32 to 16, so 17 <= num_tokens
  <= 32 (decode_32 bs=32) runs master's existing 33-program _align_parallel_kernel instead of the
  single-program _align_small_kernel; num_tokens <= 16 (bs=1, 4, 8, 16) keeps _align_small and
  num_tokens > 32 keeps _align_parallel, both unchanged. block_size_m at 17..32 is SGLang's pick
  (16), and _align_parallel already compiles a BSM=16 variant in prepare() (bsm loop 16/32/48/64),
  so no new variant is needed; the _align_small warm-up loop range(1, ALIGN_FUSED_MAX_TOKENS + 1)
  shrinks to 1..16 automatically. _align_self_check needs no edit: it routes tokens <=
  ALIGN_FUSED_MAX_TOKENS to _align_small and the rest to _align_parallel, so T=32 is now checked
  against moe_align_block_size on the parallel kernel; its tokens > ALIGN_FUSED_MAX_TOKENS summary
  line will then also count T=32 cases (expected 0 mismatches apart from the known capacity-only
  lines; never gate on the self-check). Update the comment next to the constant. Mechanism: at
  bs=32 _align_small does a [256, 32] one-hot cumsum, a [64, 32] block scan and a 1024-wide tail
  store in ONE program on one SM (4.2-4.4 us, 3x its 1.4 us at bs=1); _align_parallel spreads the
  placement over 33 programs and at n=256 each does a single CHUNK=512 pass, so it should run at
  or below its 4.0 us at n=1024 (bs=128).
expected:
  - shapes: bs=32 *
    direction: faster
    why: align 4.2-4.4 us (_align_small_kernel) -> ~2.5-3 us (_align_parallel_kernel at one chunk), ~0.7% of 200 us; judge on the align row
  - shapes: bs=1 *
    direction: flat
    why: untouched
  - shapes: bs=8 *
    direction: flat
    why: untouched
  - shapes: bs=16 *
    direction: flat
    why: untouched (could be a follow-up if bs=32 gains)
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T183956Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T183957Z
base_sha: 52cd6c64bc16f41452b2a54313347fee80b36ffe
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/align-parallel-from-17-tokens
worktree: .worktrees/muse_moe_w4a16/align-parallel-from-17-tokens
mode: submit
implemented_at: 2026-10-24T184142Z
impl_commit: c9c2db8910c72640ae840940ba5b1aa8b1d990ea
submission_sha256: d2afd64da35c3aa435424ca897931a56fc95a5f2700e80c1dbd379a28c12f03b
norm_sha256: 093861823c70c0d3f9345734853027090106d860ec4f44091652f707c1d52be0
submission_blob: b91c76ef847585a79561eb75a6c240987c726531
lint_warnings: 8
attempt_file: 2026-10-24T184142Z-submit-align-parallel-from-17-tokens.md
prepared_at: 2026-10-24T184142Z
job: ed6d884240084d9c
dispatched_at: 2026-10-24T184157Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3249983089262432
speedup_ci: [1.3051233342923494, 1.3448732835601367]
harvested_at: 2026-10-24T184719Z
best_at_harvest: 1.3249153782392282
promotion: null
---

# align-parallel-from-17-tokens

## Change

One constant plus its warm-up: lower ALIGN_FUSED_MAX_TOKENS from 32 to 16, so 17 <= num_tokens <= 32 (decode_32 bs=32) runs master's existing 33-program _align_parallel_kernel instead of the single-program _align_small_kernel; num_tokens <= 16 (bs=1, 4, 8, 16) keeps _align_small and num_tokens > 32 keeps _align_parallel, both unchanged. block_size_m at 17..32 is SGLang's pick (16), and _align_parallel already compiles a BSM=16 variant in prepare() (bsm loop 16/32/48/64), so no new variant is needed; the _align_small warm-up loop range(1, ALIGN_FUSED_MAX_TOKENS + 1) shrinks to 1..16 automatically. _align_self_check needs no edit: it routes tokens <= ALIGN_FUSED_MAX_TOKENS to _align_small and the rest to _align_parallel, so T=32 is now checked against moe_align_block_size on the parallel kernel; its tokens > ALIGN_FUSED_MAX_TOKENS summary line will then also count T=32 cases (expected 0 mismatches apart from the known capacity-only lines; never gate on the self-check). Update the comment next to the constant. Mechanism: at bs=32 _align_small does a [256, 32] one-hot cumsum, a [64, 32] block scan and a 1024-wide tail store in ONE program on one SM (4.2-4.4 us, 3x its 1.4 us at bs=1); _align_parallel spreads the placement over 33 programs and at n=256 each does a single CHUNK=512 pass, so it should run at or below its 4.0 us at n=1024 (bs=128).

## Expected

- `bs=32 *` faster: align 4.2-4.4 us (_align_small_kernel) -> ~2.5-3 us (_align_parallel_kernel at one chunk), ~0.7% of 200 us; judge on the align row
- `bs=1 *` flat: untouched
- `bs=8 *` flat: untouched
- `bs=16 *` flat: untouched (could be a follow-up if bs=32 gains)
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

A pure kernel-choice A/B at one shape: compare _align_parallel_kernel's bs=32 row with _align_small_kernel's 4.2-4.4 us in master runs. Headline effect is under noise and decode_32 Marlin rows are seed-noisy, so the kernel row is the evidence. If it wins, try 9..16 next. Overlap: the running cuda-dispatch-combine-pdl-verify replaces _align_parallel only at >32 (host branch on num_tokens > 32, independent of this constant? the implementer must keep that rewrite's gate untouched; here it is not present on master). The queued midm single-launch rewrite would replace the align at 2..32. The running swap-AB changes block_size_m at bs<=32 to 8*NT (32 at bs=32), so if it lands this would need a BSM=32 parallel variant (already compiled).

## Adversarial review

Reviewed 2026-10-24T183956Z by an adversarial subagent (model opus).

**Verdict: proceed.** A clean, shape-only kernel-choice A/B at bs=32, with bitwise-identical outputs and no confounders; the headline will sit inside noise, so judge it on the bs=32 align kernel row (_align_small 4.2-4.4 us vs _align_parallel). Before building, add BSM=8 to _align_parallel's warm-up, because the hypothesis is wrong that 17..32 tokens always get block_size_m 16.

### Justification
Plausible. Master's profiles show _align_small_kernel at 1.4 us (bs=1), 2.7 us (bs=8), 2.8-2.9 us (bs=16) and 4.2-4.4 us (bs=32), so it grows with n on one program. _align_parallel_kernel measured 4.0 us at n=1024 (bs=128) and 6.9 us at n=2048 (bs=256), about 1.45 us per 512-route chunk on top of a fixed cost. With one chunk at n=256 that extrapolates to about 2.5 us, which matches the hypothesis's 2.5-3 us. The saving is about 1.5-1.9 us of a 200 us shape (about 0.8%, against a 1.5% noise band), so `expected` is honest that the align row is the evidence. Outputs are identical: both kernels place each expert's routes in ascending route order, with the same capacity, sentinel fill and block_experts layout, so Marlin, silu and post-norm see byte-identical inputs and should not move. Weakest link: the extrapolation assumes per-chunk cost is linear and the parallel kernel's fixed cost stays well below _align_small's at bs=32. The parallel kernel runs two dependent passes per program (count, then place) plus a 736-slot tail loop of 2 chunks on the last program, with half of each 512-lane chunk masked, so a result of about 3.5-4 us (a wash) is a real possibility. A smaller factual error: the change paragraph says block_size_m at 17..32 is SGLang's pick of 16, but _block_size_m(M, 32) returns 8 for M <= 28 (28*8/32/8 = 0.875 < 0.9) and 16 only for 29..32. The prepare() warm-up compiles _align_parallel only for BSM 16/32/48/64, so any M in 17..28 would JIT-compile BSM=8 on first call. No scored shape or test (1, 4, 8, 16, 32, 33, 128, 256, 257) hits 17..28, so this is harmless for the run but is a latent first-call compile.

### Confounders on the current kernel
- triton-map-align-small-m-port / triton-map-align-on-block-m-restack: owns _align_small_kernel, which is the control arm of this A/B, not a confounder. Removing it would make the comparison moot (SGLang's align would run instead).
- align-parallel-per-expert-verify: owns _align_parallel_kernel, the treatment arm. Its ALIGN_PAR_CHUNK=512 / 8 warps tuning was chosen for n >= 1024. At n=256 half of every chunk is masked, so the result measures 'parallel kernel as tuned for verify' and not the best a parallel align could do at bs=32. Gain or loss is still unambiguous for this constant.
- marlin-block-m-two-smaller-on-postnorm: no interaction. Its pick applies only at num_tokens >= 128, so bs=32 keeps SGLang's 16.
- post-expert-norm-triton-small-m: no interaction. bs=32 is at SMALL_M_POST_NORM and runs SGLang's post-norm either way, and local_ids is identical from both align kernels.
- sglang-marlin-path-parity: base pipeline. Marlin consumes identical sorted_ids/block_experts/n_padded, so Marlin time should not change; any Marlin-row movement at bs=32 is seed noise.

### Interference and A/B needs
No companion needed. Both arms already exist on master and are profiled at bs=32 in the same harness: _align_small 4.2-4.4 us in jobs 45eacf4ca2e849d2 and 3cabc739f1af493e. The per-kernel row in this run gives a direct, unconfounded A/B: if _align_parallel_kernel at bs=32 is below about 3.8 us it is a gain, and above about 4.4 us it is a loss. Neither reading depends on a landed hypothesis being present or tuned differently. A revert companion would not help (reverting either align hypothesis removes one arm), and a tune companion (for example CHUNK=256 at small n) is a follow-up hypothesis, not a disambiguator. The headline cannot resolve 0.8%, which the hypothesis already accepts.

### Open questions
- If _align_parallel at n=256 comes in at about 3.5-4 us (a wash), is the fixed cost the two-pass structure or the half-masked 512-wide chunk? A follow-up could pick CHUNK = next_pow2(n) capped at 512 (256 here) and/or 4 warps at n <= 256.
- Do 29..32 (bsm 16) and 17..28 (bsm 8) behave differently on the parallel kernel? Only bs=32 is scored, so bsm=8 timing stays unmeasured.
- If marlin-two-class-block-m-decode32 or the swap-AB rewrites land, block_size_m at bs=32 changes (to 32, or 8*NT). The BSM=32 parallel variant is already compiled, but the win should be re-read on the new base.


## Implementation notes

In the worktree's muse_moe_w4a16/submission.py: ALIGN_FUSED_MAX_TOKENS 32 -> 16 with its comment extended; _align_parallel warm-up bsm loop in prepare() now (8, 16, 32, 48, 64), per the adversarial review (17..28 tokens get block_size_m 8), comment updated. Nothing else touched (stale "tokens <= 32" wording in the _align_self_check docstring and the "tokens > 32" comment left as is to keep the diff minimal; the self-check logic keys off the constant). Dummy warm-up ids use ALIGN_FUSED_MAX_TOKENS + 1 = 17 tokens; n is a runtime arg so the compiled variant is the same.

## Rules check

- [x] CUDA-graph capturable: diff touches only a constant and the prepare() warm-up tuple; no syncs/allocs added to kernel_fn.
- [x] Nothing derived from `data` is cached across calls: no state added.
- [x] Weights: n/a: no weight change.
- [x] Activations BF16 / KV FP8 / queries: n/a: untouched.
- [x] Attention sees every key: n/a: untouched.
- [x] No timing patches, harness I/O, module edits, reference import, subprocesses or network: diff has none.
- [x] All work inside the timed call; no output caching or replay: align kernels run per call as before.

Lint (okc_lint.py): OK with warnings (8 host-sync warnings, all in prepare()/self-check, outside kernel_fn's call graph). py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only; no GPU on node). **Verdict: dispatch.** No fixes.

- Diff is exactly the spec: ALIGN_FUSED_MAX_TOKENS 32 -> 16 plus BSM=8 added to the _align_parallel warm-up (the review's request). kernel_fn and the self-check key off the constant; no other area touched.
- Gates: none. The self-check only prints (never gates), so the new path cannot be silently disabled. At bs=32, _block_size_m(32, 32) = 16, a variant compiled in prepare(); topk_ids is int32 per Inputs, matching the int32 warm-up dummies, and n/cap/n_blocks are do_not_specialize, so no first-call JIT at any scored/test shape (1, 4, 8, 16, 32, 33, 128, 256, 257). The only new user is bs=32.
- Correctness: _align_parallel at n=256, bsm=16 is the same code path and same variant already exercised at T=33 (n=264, bsm=16, one CHUNK=512 pass); cap = 736 gives a 2-chunk tail loop on the last program, masked to [total, cap). Output buffers are sized identically by both wrappers. The self-check now also verifies T=32 on the parallel kernel.
- Rules: unchanged call graph shape (host branch on num_tokens only, no syncs/allocs added in kernel_fn, nothing cached from data). okc_lint OK with warnings (prepare()/self-check only); py_compile OK; __pycache__ removed.
- Fallback: none added; like master's >32 path, a compile error in prepare() would raise rather than fall back (warm-up of BSM 16 already runs on master, so no new build risk beyond the BSM=8 variant, which uses PAD_BLOCK=8 and is unused at scored shapes).

Residual risk: the bs=32 gain (~1.5 us of ~70 us) is below headline noise; only the per-kernel align row can read it.

## Result

- job `ed6d884240084d9c` (submit), status done, passed True, tests 5/5
- speedup 1.3250x, 95% CI [1.3051, 1.3449], score 10149.82427479356 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-parallel-from-17-tokens.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-parallel-from-17-tokens.patch`; branch `hyp/muse_moe_w4a16/align-parallel-from-17-tokens`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.82 | 48.0 | 1.0113 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 148.37 | 148.32 | 0.9976 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 199.88 | 196.35 | 1.0082 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.74 | 106.08 | 1.0051 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 318.26 | 323.0 | 0.9876 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 454.28 | 456.64 | 0.9907 | flat |

no shape moved outside noise. Expected hit: 5/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-parallel-from-17-tokens`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

