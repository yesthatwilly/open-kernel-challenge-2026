---
problem: muse_moe_w4a16
slug: marlin-block-size-m-next-smaller
created: 2026-10-24T075232Z
base: master
area: w13 marlin
change: >
  Change only the block_size_m pick in kernel_fn: keep SGLang's loop over BLOCK_SIZE_M_CANDIDATES
  = (8, 16, 32, 48, 64) (first candidate with num_tokens*8/32/block_size_m < 0.9) but take the
  candidate one index BELOW it, floored at index 0: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx
  - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)] with a module-level BLOCK_SIZE_M_STEP_DOWN = lambda
  num_tokens: 1 so a per-shape partial adoption is a one-line follow-up. Resulting picks, SGLang
  -> new: bs=1/4/8/16: 8 -> 8 (unchanged, already the smallest, the in-run control); bs=32/33: 16
  -> 8; bs=128: 48 -> 32; bs=256/257: 64 -> 48. The same block_size_m feeds moe_align_block_size
  (still over the bank's 32 experts, as on master) and the moe_block_size kwarg of both
  moe_wna16_marlin_gemm calls; nothing else changes. Mechanism: this is the mirror image of
  marlin-block-size-m-next-larger (regression 0.918x), which moved Marlin to the next
  thread_m_blocks config and lost 21% at bs=32 (16 -> 32, thread_m_blocks 1 -> 2), 14% at bs=128
  (48 -> 64, 3 -> 4) and ~2% at bs=1 (8 -> 16, the 8-row special path to the 16-row path; inside
  noise), even though the larger block collapsed m-blocks. So at these M the per-m-block cost
  (tile shape, registers, smem stages) dominates over the number of weight passes; a smaller block
  (8-row path at bs=32; thread_m_blocks 2 at bs=128; 3 at bs=256) may be cheaper per block by more
  than the extra m-blocks cost (experts with more routes than block_size_m now get two m-blocks,
  each streaming that expert's weights, partly from L2).
expected:
  - shapes: bs=32 *
    direction: faster
    why: 16 -> 8 moves to the 8-row Marlin path that bs=1/8/16 use; 16->32 lost 21% here, so the per-block cost gradient points down
  - shapes: bs=128 *
    direction: faster
    why: 48 -> 32 (thread_m_blocks 3 -> 2); 48 -> 64 lost 14% here, Marlin is 96% of the shape
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 48 (thread_m_blocks 4 -> 3); Marlin 96% of 621 us at ~1/6 of SoL
  - shapes: bs=1 *
    direction: flat
    why: "control: 8 before and after"
  - shapes: bs=8 *
    direction: flat
    why: "control: 8 before and after"
  - shapes: bs=16 *
    direction: flat
    why: "control: 8 before and after"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T100105Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T100106Z
base_sha: 38f7ef0c3079751ea38f770d5bf9c79242b606ef
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/marlin-block-size-m-next-smaller
worktree: .worktrees/muse_moe_w4a16/marlin-block-size-m-next-smaller
mode: check
implemented_at: 2026-10-24T100227Z
impl_commit: fa264a1be6ccfd5b5ece1f9a22b89c21d38c1c97
submission_sha256: 132ed645c5ca1b4ff06144d4a6d4031b1819fb985c504e3d0811fb875d2560a1
norm_sha256: e6c8aa1ba3511b86b59de78362bf48d851b5b6a6f6b813b93b1beb2beac3c40f
submission_blob: 12f7f3441aa7dfa758d69dd9147422a0c972317a
lint_warnings: 1
attempt_file: 2026-10-24T100227Z-check-marlin-block-size-m-next-smaller.md
prepared_at: 2026-10-24T100227Z
job: 198f10f446a847cb
dispatched_at: 2026-10-24T100235Z
---

# marlin-block-size-m-next-smaller

## Change

Change only the block_size_m pick in kernel_fn: keep SGLang's loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) (first candidate with num_tokens*8/32/block_size_m < 0.9) but take the candidate one index BELOW it, floored at index 0: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)] with a module-level BLOCK_SIZE_M_STEP_DOWN = lambda num_tokens: 1 so a per-shape partial adoption is a one-line follow-up. Resulting picks, SGLang -> new: bs=1/4/8/16: 8 -> 8 (unchanged, already the smallest, the in-run control); bs=32/33: 16 -> 8; bs=128: 48 -> 32; bs=256/257: 64 -> 48. The same block_size_m feeds moe_align_block_size (still over the bank's 32 experts, as on master) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls; nothing else changes. Mechanism: this is the mirror image of marlin-block-size-m-next-larger (regression 0.918x), which moved Marlin to the next thread_m_blocks config and lost 21% at bs=32 (16 -> 32, thread_m_blocks 1 -> 2), 14% at bs=128 (48 -> 64, 3 -> 4) and ~2% at bs=1 (8 -> 16, the 8-row special path to the 16-row path; inside noise), even though the larger block collapsed m-blocks. So at these M the per-m-block cost (tile shape, registers, smem stages) dominates over the number of weight passes; a smaller block (8-row path at bs=32; thread_m_blocks 2 at bs=128; 3 at bs=256) may be cheaper per block by more than the extra m-blocks cost (experts with more routes than block_size_m now get two m-blocks, each streaming that expert's weights, partly from L2).

## Expected

- `bs=32 *` faster: 16 -> 8 moves to the 8-row Marlin path that bs=1/8/16 use; 16->32 lost 21% here, so the per-block cost gradient points down
- `bs=128 *` faster: 48 -> 32 (thread_m_blocks 3 -> 2); 48 -> 64 lost 14% here, Marlin is 96% of the shape
- `bs=256 *` faster: 64 -> 48 (thread_m_blocks 4 -> 3); Marlin 96% of 621 us at ~1/6 of SoL
- `bs=1 *` flat: control: 8 before and after
- `bs=8 *` flat: control: 8 before and after
- `bs=16 *` flat: control: 8 before and after

## Risk

Risk: an expert with more local routes than block_size_m now needs an extra m-block, and every m-block streams that expert's full w13 (16 MB) and w2 (8 MB) slice; under skewed popularity at bs=256 the hot experts may cross 48 rows, so a weight-bandwidth loss can outweigh the config gain; read per shape. Correctness is unaffected (pure dispatch knob; the bs=33/257 and skewed bs=128 tests run the new picks). marlin-block-size-m-verify-two-smaller runs the two-step-down picks at bs=128/256 independently from master so the two runs together give the gradient; if all three lose here, SGLang's pick is a local optimum at bs>=32 and the block-size knob is closed. muse_decode_layer is testing a cap of 16 on the same Marlin call; do not merge their result into this one, the shapes and routing differ.

## Adversarial review

Reviewed 2026-10-24T100105Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a rule-clean change to one knob, and each scored load is its own shape, so a single run can be read per shape with no companions. The direction at bs=128/256 is already backed by muse_decode_layer's landed marlin-block-m-cap-16 (block 16 at T=128/256 cut Marlin by about 25-30%). The bs=32 leg (16 -> 8) rests on a false analogy. At bs=128/256 it overlaps marlin-block-size-m-verify-two-smaller, but the two do not duplicate each other: they measure different points on the same curve. If only one check can be spent, run verify-two-smaller first.

### Justification
Picks checked against master submission.py. The rule is num_tokens*8/32/bsm < 0.9, and when no candidate passes the loop runs out on the last one. That gives bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48 and bs=256/257 -> 64, so the new picks are 8/8/32/48 as the hypothesis states. Profile of the parity best (fb2cc736613747df): Marlin is 92% of bs=32 (212.7 us), 96% of bs=128 (495 us) and 96% of bs=256 (612 us). Floors are 70.8/101.3/108.9 us, so Marlin runs at 3-6x SoL and about 0.85-1.3 TB/s, which is latency/compute-per-block bound, not HBM bound. Each of the three moving shapes carries 1/6 of the geomean, so a 10-25% Marlin change there is easily resolved. bs=128/256 is plausible and has evidence. The sister next-larger run measured thread_m_blocks 3->4 at bs=128 as 0.86x with the m-block count barely changed. The landed decode_layer marlin-block-m-cap-16 (0730135cdcac43b7) went 48->16 / 64->16 and gained 1.10-1.15x per verify shape, with Marlin at verify_16 sliding falling from about 425 to 293 us. The curve below SGLang's pick therefore keeps falling at least down to 16. The one-step picks (32 at bs=128, 48 at bs=256) are probably intermediate and smaller gains than the two-step or cap-16 picks. Expected direction faster is consistent with that. The hypothesis's Risk text is stale: it says decode_layer is 'testing' cap 16, but that has landed as a gain. Its m-block worry is overstated. SGLang's rule overcounts local routes per expert 8x under EP8 (LESSONS), so at bs=128 the busiest GPU has roughly 4-8 routes per expert on average. Only the hot experts cross 32 or 48 rows, so the number of m-blocks hardly changes. Weakest link: bs=32, 16 -> 8. 8 and 16 both run thread_m_blocks=1; 8 only switches to the m_block_size_8 special path. The 21% loss for 16->32 came from thread_m_blocks 1->2, so it says nothing about the gradient below 16. The only direct evidence about the 8-row versus 16-row path is next-larger's bs=1/8/16 rows: 0.98/0.97/0.984, i.e. the 8-row path is about 2-3% cheaper per block. At bs=32 that saving is partly offset by an extra m-block for any expert with 9-16 local routes. The honest expectation for bs=32 is flat to slightly faster, perhaps 0-4%, which is close to the ±1.5% noise floor. It is not the confident 'faster' the expected table claims.

### Confounders on the current kernel
- sglang-marlin-path-parity (align over the bank's 32 experts, not 256): sorted_ids capacity is M*8 + 32*(bsm-1) and shrinks with a smaller block, so the align kernel's padding work drops by a few hundred slots. That is about 1 us or less, against 200-600 us of Marlin, so it cannot flip a sign. The decode_layer cap-16 evidence was measured with 256-expert align, so its magnitude transfers only approximately. The direction is unaffected. Reverting this landed change would bring back the 0.006x starter, so there is no meaningful counterfactual.

### Interference and A/B needs
No companions. The only diff is the block_size_m pick, a host int computed from num_tokens. bs=32, 128 and 256 are separate scored loads, each with the same-run SGLang column as its paired control, and bs=1/8/16 are in-run controls whose pick does not change. Per shape, a gain means adopt this pick and a loss means keep SGLang's, so neither sign is ambiguous. An ab-revert of parity is meaningless (it gives the starter), and a 256-expert-align ab-tune would move only about 1 us. On whether the two should both run: next-smaller and verify-two-smaller turn the same knob from the same base in the same direction at bs=128 (32 vs 16) and bs=256 (48 vs 32). That overlap is strong, but they measure different points, so neither duplicates the other, and run in parallel they map the curve. If the checks are limited, only one should run first, and it should be verify-two-smaller (or a direct cap-16 port), because the decode_layer result says the optimum at bs=128/256 is at or below 16-32. A next-smaller run is then needed for bs=128/256 only if two-step loses, to bracket the optimum. Its unique contribution is bs=32 (16 -> 8), and that leg could be run on its own or folded into the two-step run, since verify-two-smaller leaves bs=32 untouched. Running both now is acceptable when the quota allows: the results stay interpretable, but some of the information is redundant.

### Open questions
- The decode_layer cap-16 result (landed, 1.062x, with Marlin at verify_16 down about 30%) has not been ported to this problem. A direct cap at 16 for bs>=128 (bs=256 would go 64->16, three steps down, which neither queued hypothesis tests), plus 8 at bs=32, may be the single best run and would make most of this hypothesis's bs=128/256 points redundant.
- Update the Risk text: muse_decode_layer's cap-16 is no longer 'testing'. It landed as a gain (0730135cdcac43b7, promoted 2b1f82408ab64116).
- Implement the step-down as an idx table, as the next-larger patch did with BLOCK_SIZE_M_STEP, so that per-shape partial adoption (for example keeping 16 at bs=32 if that leg is flat) is a one-line change.
- Read bs=32 against a ±1.5% noise band. A 1-3% move there does not resolve the 8-row versus 16-row question.
- Two in-flight runs touch this path: small-m-grouped-gemv-splitk-rewrite (w13 at small M) and align-256-experts-below-128 (align for num_tokens < 128). Both could change bs=32's base if they land before this one is claimed. Re-check the base at claim time, and do not stack them blindly.
- If bs=256 at 48 gains but by less than bs=128 at 32, read that as hot-expert m-block growth under skew (the busiest-GPU routing). The next probe would then be 32 or 16 at bs=256 only.


## Implementation notes

- `muse_moe_w4a16/submission.py`, module level: added `BLOCK_SIZE_M_STEP_DOWN = lambda num_tokens: 1` (with a `# noqa: E731`) right after `BLOCK_SIZE_M_CANDIDATES`, so a per-shape partial adoption (for example keeping 16 at bs=32) is a one-line edit. This is the idx-table form the review asked for, matching next-larger's `BLOCK_SIZE_M_STEP`.
- `kernel_fn`, the block_size_m pick: SGLang's loop now records the index `idx` of the first candidate with `num_tokens*TOP_K/bank["count"]/candidate < 0.9`. If no candidate passes, `idx` defaults to the last index, which matches the old loop running out on 64. The pick is then `BLOCK_SIZE_M_CANDIDATES[max(idx - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)]`. This is a host int that depends only on the shape and the bank count, so it adds no sync and no branching on input values.
- The same `block_size_m` feeds `moe_align_block_size` (still over `bank["count"]` = 32) and the `moe_block_size` kwarg shared by both `moe_wna16_marlin_gemm` calls. The `num_tokens == 1 and expert_map is None` single-token path is unchanged; it gets 8 before and after. Nothing else is touched.
- Picks checked offline with the same rule (count = 32), old -> new: bs 1/4/8/16 8->8, bs 32/33 16->8, bs 128 48->32, bs 256/257 64->48. These match the spec.
- No deviation from the spec. I did not apply the review's open questions (port cap-16, edit the Risk text) because they are new ideas or documentation, not implementation details.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-size-m-next-smaller diff` (1 file, +11/-2):

- [x] CUDA-graph capturable: the diff adds only host-int arithmetic on `num_tokens` (a shape) and `bank["count"]` (config). It adds no `.item()`, `.cpu()`, `synchronize()` or tensor print, and no allocation or control flow that depends on input values.
- [x] Nothing derived from `data` is cached across calls: the diff writes no new state. `BLOCK_SIZE_M_STEP_DOWN` is a constant function of the shape.
- [x] Weights: untouched by the diff. `prepare()` is unchanged, NVFP4 stays NVFP4 and the repack is SGLang's own.
- [x] Activations: the diff leaves every dtype unchanged, so they stay BF16 where SGLang keeps BF16 (x, cache1/2/3 and the output).
- [x] Attention sees every key: n/a, this is an MoE kernel with no attention. Also no expert or route is skipped: every local route is still computed, and only the m-tile size changes.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none appear in the diff.
- [x] All work happens inside the timed call, with no output caching or replay: the diff only changes which tile size the in-call Marlin launches use.
- [x] The first-line marker `# okc: problem=muse_moe_w4a16` is kept, and the diff adds no triton.autotune.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`):

```
warn  host-sync    L114: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

The warning is the existing `torch.cuda.synchronize()` at the end of `prepare()` on master, not something the diff added. `python3 -I -m py_compile` passed, and I removed the `__pycache__` it left.
