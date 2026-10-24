# muse_decode_layer · 2026-10-24T231719Z · check · sliding-decode1-fused-restack-on-pdl

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T224525Z-sliding-decode1-fused-restack-on-pdl.md` (branch `hyp/muse_decode_layer/sliding-decode1-fused-restack-on-pdl`).

- **Job:** `30027c28212946fd` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `b85563b`

## Goal
Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4 us, +4.4%, every other shape flat; it also carried the +8.75% sliding decode_1 row of decode-restack-sliding-port-paired-pad-gemms-t1) onto current master (f1f2f35, dense-chain PDL). 3-way merge branch hyp/muse_decode_layer/sliding-fused-triton-decode1-port (patch done/neutral/sliding-fused-triton-decode1-port.patch) unchanged: its Triton helpers and _sliding_fused_kernel, the config gate (window != 0 and q_len == 1 and batch == 1), its prepare() warm-up and fallback; on that branch the one fused launch replaces FusedQKNorm, the FP8 casts, store_cache, flash_attn_with_kvcache + combine and o_norm_gate, and its output feeds the unchanged o GEMM. Bring nothing from the restack parent (no T=1 Triton GEMMs, no paired pad). If sliding-fused-batched-decode-port (running, batch > 1) has landed by claim time, its copies of the same helpers are already on master: reuse them, only add batch == 1 to the gate with that parent decode_1 split count, never two copies. Everything from the o GEMM on is master.

## What changed, and where
Built in the main thread on 2026-10-24 ~23:30Z; the user approved self-review in place of the separate adversarial review ("go") because no new agents could be spawned before the 00:00Z reset.
- Worktree reset to master c9bcd4b (claim had based it on the parent branch, which predates small-t-launch-minimal-dense-chain-pdl), then `git apply -3` of done/neutral/sliding-fused-triton-decode1-port.patch.
- Two conflicts, both "both sides added": the module docstring bullet list and the section insertion point (dense-chain section vs the sliding section). Resolved by keeping both, dense chain first. No line of either side was edited.
- Wiring checked: the fused launch replaces only the attention block at window != 0, batch == 1, q_len == 1, and its `o` feeds the unchanged o GEMM; the dense chain starts after the o GEMM at ffn_input, so the two are disjoint.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the parent's kernel is unchanged and passed the judge's tests in this layer; no host sync, allocation or value-dependent control flow added in kernel_fn (gate is config-only).
- [x] Fresh inputs: nothing derived from `data` is cached.
- [x] Weights: unchanged (no new repacks).
- [x] Activations BF16 where SGLang keeps BF16; KV cache stays FP8; queries FP8 E4M3 as in the parent.
- [x] Every key attended: full 2048-token window, as in the parent.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: OK with warnings (prepare-only host syncs, pre-existing).

## Risks / what could fail
low
Low: parent passed tests; one shape. Conflicts with dense-chain-stage2-attn-side-pdl on sliding decode_1 (both replace the attention-side chain there); whichever lands first, the other gates out batch-1 sliding decode.

## Result (after the run)
- job `30027c28212946fd` (check), status done, passed True, tests 13/13
- speedup 1.2374x, 95% CI [1.2188, 1.2559], score 18280.96912256323 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/neutral/sliding-fused-triton-decode1-port.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/sliding-decode1-fused-restack-on-pdl.result.json`, `muse_decode_layer/hypotheses/done/neutral/sliding-decode1-fused-restack-on-pdl.patch`; branch `hyp/muse_decode_layer/sliding-decode1-fused-restack-on-pdl`
