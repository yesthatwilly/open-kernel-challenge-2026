---
problem: muse_prefill_layer
slug: align-local-parallel-first-turn
created: 2026-10-24T232955Z
base: master
area: MoE align on the Triton path (T > 4096, first_turn)
change: >
  Built on moe-align-local-routes-parallel-port (running, its implementation commit). In
  _experts_triton, when state['align_par_ok'], replace moe_align_block_size(local_ids,
  cfg['BLOCK_T'], N_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T'])
  (local routes only), else keep master's call. Extend _align_par_setup's self-check with T=16384
  at BLOCK_T 256 (the first_turn block) so the gate covers this path. Measures whether the
  kernel's serial per-program chunk loop (n/512 = 256 iterations at 131K routes) beats SGLang's
  count_and_sort + align (87 + 50 us at first_turn) and drops the ~448 remote token tiles and
  their w2 zero-stores.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: no remote token tiles in the Triton grouped GEMMs; one align launch
risk: medium
depends_on: moe-align-local-routes-parallel-port
priority: 2
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T232955Z
base_sha: 67f7473c299ffe95b9bb8701cdaf2079f00e3cab
base_blob: 8780061c8886bf3d70fe3c433fbbfade9bdf8507
branch: hyp/muse_prefill_layer/align-local-parallel-first-turn
worktree: .worktrees/muse_prefill_layer/align-local-parallel-first-turn
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233023Z
impl_commit: 0b61b2b6f3bb4a07433e7bea0e169e70d46d0894
submission_sha256: 8177399e276e2bedb4909471f4c254855cefe2f038c518e9d9a63b93fc8f34cb
norm_sha256: 35c68a631fd3932f72e0d779eeda2bc5b8a3e654c1334fa62fc41f2fd1058792
submission_blob: 32d9f1e7e5c9c472ea3b8c05d97fa7de566e6135
lint_warnings: 12
attempt_file: 2026-10-24T233023Z-check-align-local-parallel-first-turn.md
prepared_at: 2026-10-24T233023Z
job: c2f3e9d8297f4a18
dispatched_at: 2026-10-24T233033Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.0625839850302201
speedup_ci: [1.0466452252547669, 1.0785227448056733]
harvested_at: 2026-10-25T000223Z
best_at_harvest: 1.0591912383762452
promotion: null
---

# align-local-parallel-first-turn

## Change

Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']) (local routes only), else keep master's call. Extend _align_par_setup's self-check with T=16384 at BLOCK_T 256 (the first_turn block) so the gate covers this path. Measures whether the kernel's serial per-program chunk loop (n/512 = 256 iterations at 131K routes) beats SGLang's count_and_sort + align (87 + 50 us at first_turn) and drops the ~448 remote token tiles and their w2 zero-stores.

## Expected

- `*load=first_turn*` faster: no remote token tiles in the Triton grouped GEMMs; one align launch

## Risk

Main-thread build before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:30Z with user-approved self-review, on moe-align-local-routes-parallel-port's implementation commit (claim based it there: parent still running). `_experts_triton`: `_align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg["BLOCK_T"])` when `state["align_par_ok"]`, else master's call. `_align_par_setup` now also tests T=16384 at the Triton path's BLOCK_T (256), which also compiles that variant before capture; a mismatch there turns the whole parallel align OFF (both paths), never just one.

## Rules check

- [x] CUDA-graph capturable: shape-only allocations, prepare()-set gate, no host sync in kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Result

- job `c2f3e9d8297f4a18` (check), status done, passed True, tests 5/5
- speedup 1.0626x, 95% CI [1.0466, 1.0785], score 364586.61664186686 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/neutral/moe-align-local-routes-parallel-port.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/align-local-parallel-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/align-local-parallel-first-turn.patch`; branch `hyp/muse_prefill_layer/align-local-parallel-first-turn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1049.13 | 1012.65 | 1.0019 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1472.72 | 1425.42 | 1.0002 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3289.65 | 3357.07 | 0.9999 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4933.23 | 4952.67 | 1.0003 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14603.47 | 14832.8 | 0.9768 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14010.24 | 14230.65 | 0.9906 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18449.67 | 18516.9 | 0.9852 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22182.6 | 22089.62 | 0.9963 | flat |

no shape moved outside noise. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: align-local-parallel-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

