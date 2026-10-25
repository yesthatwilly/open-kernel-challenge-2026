---
problem: muse_decode_layer
slug: stack-align-par-sidestream-sliding1
created: 2026-10-24T233721Z
base: master
area: T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention
change: >
  Built on master (dense-chain PDL). moe-align-parallel-large-t-port +
  moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build
  moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side
  stream and passes the result to _experts via dispatch=, which would bypass the align port's
  branch; merged so _expert_dispatch calls _align_parallel(shifted ids, 0, LOCAL_EXPERTS, bsm)
  when state['align_par_ok'] (one launch on the side stream), else master's gather + align. The
  sliding batch-1 restack is disjoint.
expected:
  - shapes: "*"
    direction: faster
    why: parallel align off the critical path at T>32, plus sliding decode_1
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T233722Z
base_sha: 90c5720f9cb1f796aeb6a987abdd676cba19ef5f
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-align-par-sidestream-sliding1
worktree: .worktrees/muse_decode_layer/stack-align-par-sidestream-sliding1
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233750Z
impl_commit: 511013c59c339b0537277be2638fdab4e28fe0fe
submission_sha256: af794c2403236844b6251445a0b6d93f20e4e5dcbc7b55bd8ecf4eac1db73fee
norm_sha256: f1839f83287cb09ca68022b5ab9082c485a17f447bd5f892662edeba2963485e
submission_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
lint_warnings: 28
attempt_file: 2026-10-24T233750Z-check-stack-align-par-sidestream-sliding1.md
prepared_at: 2026-10-24T233750Z
job: cc92097b64784c0f
dispatched_at: 2026-10-24T233759Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2495238726339006
speedup_ci: [1.230781014544392, 1.268266730723409]
harvested_at: 2026-10-25T000221Z
best_at_harvest: 1.228328123558382
promotion: submitted
promotion_job: 39bcf05afc4e4dd3
promotion_at: 2026-10-25T000254Z
---

# stack-align-par-sidestream-sliding1

## Change

Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side stream and passes the result to _experts via dispatch=, which would bypass the align port's branch; merged so _expert_dispatch calls _align_parallel(shifted ids, 0, LOCAL_EXPERTS, bsm) when state['align_par_ok'] (one launch on the side stream), else master's gather + align. The sliding batch-1 restack is disjoint.

## Expected

- `*` faster: parallel align off the critical path at T>32, plus sliding decode_1

## Risk

Main-thread hand merge before the 00:00Z reset, user-approved self-review.

## Implementation notes

Hand merge in the main thread 2026-10-24 ~23:37Z with user-approved self-review. Worktree on master (dense-chain PDL); `git apply -3` of moe-align-parallel-large-t-port's and moe-dispatch-prep-on-side-stream's done/ patches, then the sliding batch-1 restack as `git diff c9bcd4b..b85563b`.
- One conflict in `_experts`: resolved to `if dispatch is not None` (side-stream result) / `elif` the align port's T >= 33 branch / `elif align is None` master's path.
- `_expert_dispatch` (side stream, T > 32): when `state["align_par_ok"]`, `_align_parallel(shifted, 0, LOCAL_EXPERTS, _block_size_m(T))` replaces the expert_map gather + moe_align_block_size (the same call and inputs prepare() self-checked); otherwise master's two calls. Without this the align port would be bypassed whenever dispatch= is passed.
- kernel_fn's existing record_stream loop already covers local_ids and the align tensors produced on the side stream.

## Rules check

- [x] CUDA-graph capturable: side-stream fork/join and allocations as in the parents; prepare()-set gate; no host sync in kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; queries FP8 E4M3 as the parents.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Result

- job `cc92097b64784c0f` (check), status done, passed True, tests 13/13
- speedup 1.2495x, 95% CI [1.2308, 1.2683], score 18103.13551858608 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.patch`; branch `hyp/muse_decode_layer/stack-align-par-sidestream-sliding1`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 132.74 | 127.48 | 1.0355 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.57 | 142.93 | 0.9942 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 217.06 | 204.98 | 1.0061 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 347.51 | 349.47 | 0.9912 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 246.99 | 245.77 | 0.9994 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 534.98 | 529.71 | 1.0081 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 186.48 | 190.8 | 0.9992 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 195.92 | 192.44 | 1.0111 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 447.55 | 429.61 | 1.0615 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 599.37 | 563.65 | 1.0473 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 626.79 | 592.52 | 1.0216 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 910.91 | 861.3 | 1.01 | flat |

gained on sliding decode_1, verify_16; full verify_16. Expected hit: 1/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-align-par-sidestream-sliding1`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

