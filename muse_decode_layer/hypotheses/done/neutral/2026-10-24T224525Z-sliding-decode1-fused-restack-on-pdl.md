---
problem: muse_decode_layer
slug: sliding-decode1-fused-restack-on-pdl
created: 2026-10-24T224525Z
base: master
area: FA3 call (sliding decode, batch 1)
change: >
  Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4
  us, +4.4%, every other shape flat; it also carried the +8.75% sliding decode_1 row of
  decode-restack-sliding-port-paired-pad-gemms-t1) onto current master (f1f2f35, dense-chain PDL).
  3-way merge branch hyp/muse_decode_layer/sliding-fused-triton-decode1-port (patch
  done/neutral/sliding-fused-triton-decode1-port.patch) unchanged: its Triton helpers and
  _sliding_fused_kernel, the config gate (window != 0 and q_len == 1 and batch == 1), its
  prepare() warm-up and fallback; on that branch the one fused launch replaces FusedQKNorm, the
  FP8 casts, store_cache, flash_attn_with_kvcache + combine and o_norm_gate, and its output feeds
  the unchanged o GEMM. Bring nothing from the restack parent (no T=1 Triton GEMMs, no paired
  pad). If sliding-fused-batched-decode-port (running, batch > 1) has landed by claim time, its
  copies of the same helpers are already on master: reuse them, only add batch == 1 to the gate
  with that parent decode_1 split count, never two copies. Everything from the o GEMM on is
  master.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "attention chain FA3 10.4 + combine 4.0 + qk_norm/casts/store/norm-gate nodes -> one fused launch: measured +4.4% on old base, ~5-6 us of 132.7"
  - shapes: "*layer=full *"
    direction: flat
    why: outside gate
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: outside gate
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: outside gate
risk: low
depends_on: sliding-fused-triton-decode1-port
priority: 3
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T231559Z
base_sha: 2569d7e0d6f208921d432fd4237dc483601b39ca
base_blob: e33a32aa5580fbd5b38764ea287a6364c865432e
branch: hyp/muse_decode_layer/sliding-decode1-fused-restack-on-pdl
worktree: .worktrees/muse_decode_layer/sliding-decode1-fused-restack-on-pdl
mode: check
implemented_at: 2026-10-24T231719Z
impl_commit: b85563b6593313a0cbabdf1acf5f57e9d4ed02a7
submission_sha256: 9aca5bba7f7575c8840f3e7d8100a979ee214b744cdf4db9a590e01990d5cd99
norm_sha256: e796e3016e09bfe67dc431bb28642d866430cd13ad6cc9e8034410752a4bebb7
submission_blob: 54c2a04b2a56aa2b7315c6ccc78d2573fc28a3ce
lint_warnings: 18
attempt_file: 2026-10-24T231719Z-check-sliding-decode1-fused-restack-on-pdl.md
prepared_at: 2026-10-24T231719Z
job: 30027c28212946fd
dispatched_at: 2026-10-24T231729Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.237368754815135
speedup_ci: [1.218808223492908, 1.255929286137362]
harvested_at: 2026-10-24T232735Z
best_at_harvest: 1.228328123558382
promotion: null
---

# sliding-decode1-fused-restack-on-pdl

## Change

Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4 us, +4.4%, every other shape flat; it also carried the +8.75% sliding decode_1 row of decode-restack-sliding-port-paired-pad-gemms-t1) onto current master (f1f2f35, dense-chain PDL). 3-way merge branch hyp/muse_decode_layer/sliding-fused-triton-decode1-port (patch done/neutral/sliding-fused-triton-decode1-port.patch) unchanged: its Triton helpers and _sliding_fused_kernel, the config gate (window != 0 and q_len == 1 and batch == 1), its prepare() warm-up and fallback; on that branch the one fused launch replaces FusedQKNorm, the FP8 casts, store_cache, flash_attn_with_kvcache + combine and o_norm_gate, and its output feeds the unchanged o GEMM. Bring nothing from the restack parent (no T=1 Triton GEMMs, no paired pad). If sliding-fused-batched-decode-port (running, batch > 1) has landed by claim time, its copies of the same helpers are already on master: reuse them, only add batch == 1 to the gate with that parent decode_1 split count, never two copies. Everything from the o GEMM on is master.

## Expected

- `*layer=sliding load=decode_1 *` faster: attention chain FA3 10.4 + combine 4.0 + qk_norm/casts/store/norm-gate nodes -> one fused launch: measured +4.4% on old base, ~5-6 us of 132.7
- `*layer=full *` flat: outside gate
- `*layer=sliding load=decode_16 *` flat: outside gate
- `*layer=sliding load=verify_*` flat: outside gate

## Risk

Low: parent passed tests; one shape. Conflicts with dense-chain-stage2-attn-side-pdl on sliding decode_1 (both replace the attention-side chain there); whichever lands first, the other gates out batch-1 sliding decode.

## Implementation notes

Built in the main thread on 2026-10-24 ~23:30Z; the user approved self-review in place of the separate adversarial review ("go") because no new agents could be spawned before the 00:00Z reset.
- Worktree reset to master c9bcd4b (claim had based it on the parent branch, which predates small-t-launch-minimal-dense-chain-pdl), then `git apply -3` of done/neutral/sliding-fused-triton-decode1-port.patch.
- Two conflicts, both "both sides added": the module docstring bullet list and the section insertion point (dense-chain section vs the sliding section). Resolved by keeping both, dense chain first. No line of either side was edited.
- Wiring checked: the fused launch replaces only the attention block at window != 0, batch == 1, q_len == 1, and its `o` feeds the unchanged o GEMM; the dense chain starts after the o GEMM at ffn_input, so the two are disjoint.

## Rules check

- [x] CUDA-graph capturable: the parent's kernel is unchanged and passed the judge's tests in this layer; no host sync, allocation or value-dependent control flow added in kernel_fn (gate is config-only).
- [x] Fresh inputs: nothing derived from `data` is cached.
- [x] Weights: unchanged (no new repacks).
- [x] Activations BF16 where SGLang keeps BF16; KV cache stays FP8; queries FP8 E4M3 as in the parent.
- [x] Every key attended: full 2048-token window, as in the parent.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: OK with warnings (prepare-only host syncs, pre-existing).

## Result

- job `30027c28212946fd` (check), status done, passed True, tests 13/13
- speedup 1.2374x, 95% CI [1.2188, 1.2559], score 18280.96912256323 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/neutral/sliding-fused-triton-decode1-port.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/sliding-decode1-fused-restack-on-pdl.result.json`, `muse_decode_layer/hypotheses/done/neutral/sliding-decode1-fused-restack-on-pdl.patch`; branch `hyp/muse_decode_layer/sliding-decode1-fused-restack-on-pdl`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 139.39 | 125.55 | 1.1133 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 156.81 | 142.71 | 1.1006 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 223.79 | 211.66 | 1.0789 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 364.77 | 359.57 | 1.0427 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 255.26 | 242.04 | 1.0788 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 542.94 | 534.72 | 1.0416 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 197.56 | 183.57 | 1.0893 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 213.42 | 202.19 | 1.0804 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 467.52 | 444.31 | 1.0459 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 593.54 | 602.37 | 1.0086 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 618.95 | 631.22 | 0.9906 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 897.56 | 909.95 | 1.0054 | flat |

gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16; full decode_1, decode_16, decode_32, verify_1. Expected hit: 1/4. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-decode1-fused-restack-on-pdl`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

