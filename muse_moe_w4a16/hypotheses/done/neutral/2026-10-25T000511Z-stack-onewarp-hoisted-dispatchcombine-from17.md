---
problem: muse_moe_w4a16
slug: stack-onewarp-hoisted-dispatchcombine-from17
created: 2026-10-25T000511Z
base: master
area: MoE align (bs<=8 one warp; 17..32 parallel; bs>=128 CUDA) + post-expert combine
change: >
  Built on master. The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches,
  unchanged) plus align-parallel-from-17-tokens (ALIGN_FUSED_MAX_TOKENS 32 -> 16, BSM=8 in the
  _align_parallel warm-up; scored 1.3250), so tokens 17..32 use the 33-program parallel align.
  That range sits under the CUDA dispatch rewrite's (> 32 tokens) and above the one-warp change
  (<= 8), so all four are run-time disjoint.
expected:
  - shapes: "*"
    direction: faster
    why: the three-way stack plus the bs=32 parallel align
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-25T000512Z
base_sha: 0da67366bd6284dd7d133b480bf79f5076c2bb43
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine-from17
worktree: .worktrees/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine-from17
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T000539Z
impl_commit: 8f0cad735ca39177be03bacd801d482b6e4ff89e
submission_sha256: 4273d5e3f291160bcf6935d8afffa919cbaa54383898a8673f3cec885a64aa34
norm_sha256: 25bed6bc1543861dbb03557d71a595f30620c79656908074ab2f96c97d50fe20
submission_blob: c597d7762b0f4911a4f1725d6e018dc17ff92f6c
lint_warnings: 19
attempt_file: 2026-10-25T000539Z-check-stack-onewarp-hoisted-dispatchcombine-from17.md
prepared_at: 2026-10-25T000539Z
job: 0fde9013de97404d
dispatched_at: 2026-10-25T000557Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.3237465826611308
speedup_ci: [1.3038903839212137, 1.3436027814010476]
harvested_at: 2026-10-25T003211Z
best_at_harvest: 1.3278240305297715
promotion: null
---

# stack-onewarp-hoisted-dispatchcombine-from17

## Change

Built on master. The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches, unchanged) plus align-parallel-from-17-tokens (ALIGN_FUSED_MAX_TOKENS 32 -> 16, BSM=8 in the _align_parallel warm-up; scored 1.3250), so tokens 17..32 use the 33-program parallel align. That range sits under the CUDA dispatch rewrite's (> 32 tokens) and above the one-warp change (<= 8), so all four are run-time disjoint.

## Expected

- `*` faster: the three-way stack plus the bs=32 parallel align

## Risk

Reset priority 4 variant (user-requested), claimed with --skip-review on that instruction.

## Implementation notes

Built in the main thread 2026-10-25 ~00:12Z (reset priority 4 variant, user-specified; self-reviewed). `git apply -3` of the four done/neutral patches, all clean. ONE EDIT beyond the patches: the CUDA dispatch path's gate was `ALIGN_FUSED_MAX_TOKENS < num_tokens <= CDC_MAX_TOKENS`, so lowering ALIGN_FUSED_MAX_TOKENS to 16 (align-parallel-from-17-tokens) would have moved 17..32 tokens onto the CUDA kernels, which their self-check never covered (33/128/256/257). Added `CDC_MIN_TOKENS = 32` and gated the CUDA path on it, so 17..32 tokens run the Triton `_align_parallel` as align-parallel-from-17-tokens intended and the CUDA path keeps exactly its audited range. The CUDA path's PATH CHOSEN text still prints ALIGN_FUSED_MAX_TOKENS as its lower bound (cosmetic).

## Rules check

- [x] CUDA-graph capturable: unchanged from the four parents; the gate edit is a config-only comparison.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 unchanged.
- [x] Activations BF16.
- [x] n/a: no attention in this problem.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Result

- job `0fde9013de97404d` (check), status done, passed True, tests 5/5
- speedup 1.3237x, 95% CI [1.3039, 1.3436], score 10159.421883427603 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/stack-onewarp-hoisted-dispatchcombine-from17.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/stack-onewarp-hoisted-dispatchcombine-from17.patch`; branch `hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine-from17`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.82 | 51.23 | 0.9868 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 148.37 | 148.42 | 0.9975 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 199.88 | 204.11 | 1.0074 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.74 | 108.45 | 0.9889 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 318.26 | 317.68 | 1.0077 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 454.28 | 451.34 | 1.0067 | flat |

no shape moved outside noise. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: stack-onewarp-hoisted-dispatchcombine-from17`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

