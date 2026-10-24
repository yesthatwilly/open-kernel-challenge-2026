---
problem: muse_moe_w4a16
slug: marlin-block-m-two-smaller-on-postnorm
created: 2026-10-24T112137Z
base: master
area: w13 marlin
change: >
  Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the
  pre-post-norm base, job ed837a4a46d44a95; patch at
  muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current
  master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection
  loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with
  num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it
  stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx -
  2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32
  (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the
  in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master;
  bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both
  moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at
  SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's
  fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul,
  the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no
  sync, nothing input-dependent.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "same change gained 36% on verify_16 on the old base (468.8 -> 343.7 us, Marlin 435 -> 275 us); post-norm did not touch bs=128 (SGLang combine kept), so the Marlin saving carries over: expect ~340-350 us vs master 475.9"
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 32 gained 32% on verify_32 on the old base (618.6 -> 468.7 us); same untouched combine on master, expect ~470 us vs 622.2
  - shapes: bs=1 *
    direction: flat
    why: "control: unchanged pick (8) and unchanged Triton post-norm"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: "control: pick stays 16 (16 -> 8 lost 3.6% in marlin-block-size-m-next-smaller), SGLang combine kept"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T112356Z
review_companions: null
hold_reason: null
stage: harvested
claimed_at: 2026-10-24T112420Z
base_sha: 2a10183f4da4f215e94343a98324a39e52be5cb6
base_blob: b81d47abe0c2e53a3ecc180884ae8d9671ba4e14
branch: hyp/muse_moe_w4a16/marlin-block-m-two-smaller-on-postnorm
worktree: .worktrees/muse_moe_w4a16/marlin-block-m-two-smaller-on-postnorm
mode: check
implemented_at: 2026-10-24T112434Z
impl_commit: 954a81bbc3a7e99edc92ef0f8a158fc85485cf9f
submission_sha256: aef6eb6a3d1c70bca1474c9d052fab0fb3f7b815048d9f8d58541638188cb38a
norm_sha256: 31f36fd3f81ea51151244b03a4d50c0b42ab67f76c207852ed8f1d9702be3376
submission_blob: c7bf9ffedc48b757f56765004e5d65f6ad5ba841
lint_warnings: 1
attempt_file: 2026-10-24T112434Z-check-marlin-block-m-two-smaller-on-postnorm.md
prepared_at: 2026-10-24T112434Z
job: c00ef7295c154867
dispatched_at: 2026-10-24T112447Z
outcome: gain
verdict_detail: gain
speedup: 1.2535643764852882
speedup_ci: [1.2347609108380089, 1.2723678421325675]
harvested_at: 2026-10-24T124010Z
best_at_harvest: 1.1297159216968498
promotion: merged
promotion_commit: cefc282ebdcb22f4a4e60d2acf151b862ee831ad
promotion_attempt: 2026-10-24T132213Z-submit-marlin-block-m-two-smaller-on-postnorm.md
---

# marlin-block-m-two-smaller-on-postnorm

## Change

Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the pre-post-norm base, job ed837a4a46d44a95; patch at muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master; bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul, the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no sync, nothing input-dependent.

## Expected

- `bs=128 *` faster: same change gained 36% on verify_16 on the old base (468.8 -> 343.7 us, Marlin 435 -> 275 us); post-norm did not touch bs=128 (SGLang combine kept), so the Marlin saving carries over: expect ~340-350 us vs master 475.9
- `bs=256 *` faster: 64 -> 32 gained 32% on verify_32 on the old base (618.6 -> 468.7 us); same untouched combine on master, expect ~470 us vs 622.2
- `bs=1 *` flat: control: unchanged pick (8) and unchanged Triton post-norm
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control: pick stays 16 (16 -> 8 lost 3.6% in marlin-block-size-m-next-smaller), SGLang combine kept

## Risk

Pure restack of two orthogonal measured gains: the block-m change touches only bs>=128, the post-norm change only bs<32, so the expected headline is roughly 1.130 x 1.092 = ~1.23x, and the bs<128 rows must reproduce master's (55/156/207/121 us) within drift. If marlin-block-m-cap-16-port (in the judge, bs=128 48->16 and bs=256 64->16) harvests better than verify-two-smaller at bs=256, the bs=256 pick here should follow it (16 instead of 32) - either retune this before claim or queue an ab-tune companion; bs=128 is 16 in both. Risk is low: the identical selection code already passed all 5 tests (incl. bs=257) on the old base; the only interaction is that bs=128/256 still use SGLang's combine, which is unchanged.

## Adversarial review

Reviewed 2026-10-24T112356Z by an adversarial subagent (model opus).

**Verdict: proceed.** A clean restack of two measured gains on disjoint shapes: block-m two-smaller touches only bs>=128, the landed Triton post-norm only bs<32, and the code is the same one-line pick that already passed all 5 tests including bs=257. One run can be read. The only open item is the bs=256 pick (32 here, 16 in marlin-block-m-cap-16-port, which is still running), and that run's harvest answers it, not a companion.

### Justification
The mechanism is measured on this exact problem. Per-m-block Marlin cost grows with thread_m_blocks. next-larger lost 18-28%, next-smaller gained (bs=128 1.194, bs=256 1.151), and verify-two-smaller gained more (bs=128 468.8->343.7 us, 1.364; bs=256 618.6->468.7 us, 1.32). All of these ran on base fc28840b9d, whose bs>=128 path is byte-identical to master's: the post-norm patch only adds a num_tokens < 32 branch, and bs=128/256 still call SGLang's fused_post_expert_norm_weighted_sum. Master's bs=128/256 rows (475.9/622.2 us, Marlin 95-96%, 4.7x/5.7x SoL of 101/109 us) match the old base's 468.8/618.6 within drift, so the expected ~345/~470 us are well grounded. Headline check: each load is one shape and the score is a geomean over 6 loads, so swapping master's bs=128/256 per-shape speedups (1.001/1.000) for 1.364/1.32 multiplies the headline by (1.364*1.32)^(1/6) ~= 1.10, which predicts about 1.24x. The proposer's ~1.23x is consistent. The controls are right: bs=1/8/16 keep pick 8 plus the Triton post-norm, and bs=32 keeps 16 plus SGLang's combine, so they should reproduce master's 55/121/156/207 us. Weakest link: the bs=256 choice of 32 is not known to be the best point. Decode_layer's cap-16 gained at T=256, and cap-16-port (64->16 at bs=256) is in the judge right now. If 16 beats 32 at bs=256, this run lands a result that is already sub-optimal at one of six loads. Minor: the old-base profile shows bs=256 Marlin at 490.8 us against a 468.7 us shape median (profile and timed passes disagree), so judge this run by the timed median and the paired SGLang ratio, not the kernel sum.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (192f225d8f): not a confounder for the target shapes, since its branch is num_tokens < 32 only and bs>=128 keeps SGLang's combine unchanged. Its rows (bs=1/8/16) are the in-run control and should reproduce master. A drift of more than ~2.5% there means run noise, not interaction. If it were reverted, bs>=128 would be unaffected.
- sglang-marlin-path-parity (a646cb4151): the foundation. block_size_m feeds moe_align_block_size over the 32-expert bank, so padding is 32*(bm-1) rather than 256*(bm-1), plus both Marlin calls. Smaller bm also cuts padded rows, but the old-base measurement had the same local-count align, so the measured gain already includes this interaction and carries over unchanged.

### Interference and A/B needs
No companion is needed. A gain at bs=128/256 can only come from the block-m pick, because nothing else in the timed path changes for those shapes, and the bs<128 rows are a built-in control on the same base. A loss would also be unambiguous: it would mean a bad restack or drift, and the controls would show which. The one real question is bs=256 at 32 versus 16. An ab-tune companion (this branch with the bs>=128 pick replaced by min(pick, 16)) would be the same code as cap-16-port restacked onto post-norm. That is a tuning choice settled by cap-16-port's harvest, which is already in flight. It is not needed to read this run. Recommended sequencing: hold dispatch until cap-16-port (base 150297e862) harvests. If its bs=256 row beats verify-two-smaller's 1.32 by clearly more than drift (more than ~3%), retune this spec's bs=256 pick to 16 before claiming, so that only one restack is spent. Otherwise dispatch as written. bs=128 is 16 in both, so nothing there depends on it.

### Open questions
- Does cap-16-port's bs=256 row (64->16) beat verify-two-smaller's 1.32 (64->32) by more than drift? If yes, change the bs=256 pick to 16 before dispatch, and do not run both restacks.
- The bs=257 test now picks 32 on master's lineage (it passed on the old base). Its post-norm path is SGLang's at bs>=32, which is unchanged, so correctness risk is minimal, but confirm the 5/5 tests pass.
- For the implementer: use BLOCK_SIZE_M_CANDIDATES.index(block_size_m) after the loop, as in the landed patch, so the loop-exhausted case (64) maps to index 4. Do not track idx inside the loop in a way that misses the exhaust case.


## Implementation notes

Implemented inline by churnsubs on master (which has post-expert-norm-triton-small-m): the same 5-line hunk
as marlin-block-size-m-verify-two-smaller, placed right after SGLang's unchanged block_size_m loop in
kernel_fn: at num_tokens >= 128 the pick moves two candidates down (bs=128 48 -> 16, bs=256/257 64 -> 32).
The bs=256 pick stays 32: marlin-block-m-cap-16-port's bs=256 row (1.329 vs SGLang) was within 3% of
two-smaller's (1.320), the reviewer's threshold for switching to 16. The post-norm path (bs<32) is untouched.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-m-two-smaller-on-postnorm diff` (+5 lines in kernel_fn).

- [x] CUDA-graph capturable: a Python index pick on `num_tokens` (a shape); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: prepare() and the Marlin repack untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged; KV cache / query items n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention; alignment still covers every local route.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

okc_lint: OK with warnings (only the existing `torch.cuda.synchronize()` in prepare()).

## Result

- job `c00ef7295c154867` (check), status done, passed True, tests 5/5
- speedup 1.2536x, 95% CI [1.2348, 1.2724], score 10728.208500712633 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.1297159216968498, "ci": [1.112770182871397, 1.1466616605223023], "job": "bacd1f502f4e46fa"}; base result: muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/marlin-block-m-two-smaller-on-postnorm.result.json`, `muse_moe_w4a16/hypotheses/done/gain/marlin-block-m-two-smaller-on-postnorm.patch`; branch `hyp/muse_moe_w4a16/marlin-block-m-two-smaller-on-postnorm`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 55.3 | 55.52 | 1.0027 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 155.66 | 155.89 | 1.0022 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 206.92 | 209.65 | 1.0027 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 121.24 | 120.38 | 1.0046 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 475.92 | 339.69 | 1.398 | gained |
| bs=256 load=verify_32 sharding=ep8 | 60 | 622.17 | 466.86 | 1.319 | gained |

gained on verify_16 verify_16; verify_32 verify_32. Expected hit: 6/6. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

