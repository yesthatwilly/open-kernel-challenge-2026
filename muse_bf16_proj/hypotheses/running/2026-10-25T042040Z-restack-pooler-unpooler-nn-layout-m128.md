---
problem: muse_bf16_proj
slug: restack-pooler-unpooler-nn-layout-m128
created: 2026-10-25T042040Z
base: master
area: pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)
change: >
  Built on master (b99b1dd, the landed restack). Apply the exact change of branch
  hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the restack: in
  prepare(), store state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] =
  W_unpooler.t().contiguous() (lossless [K, N] bf16 copies; the originals stay). In kernel_fn add
  one branch just before the `if layer != 'router'` cuBLAS fallback: if layer in {pooler,
  unpooler} and 65 <= cfg m <= 128 and cfg n is the full width -> return torch.mm(x, state[layer +
  '_t']). m >= 129 keeps TN; m <= 32 keeps the restack's swap-AB paths. Nothing else changes.
  Record the nvjet NN kernel names for the two target rows.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: TN nvjet 64x64 25.7 us (row 25.55) -> NN splitK 20.4 + reduce 3.0, row 22.98 on master (-10.1%, z -29)
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: row 19.79 -> 19.20 on master (-3%, z -9); small
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: m>=129 keeps TN
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: m>=129 keeps TN
  - shapes: layer=* load=decode_*
    direction: flat
    why: m<=32 untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T042254Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T042254Z
base_sha: b2ff4b18564626ccd3f3e253cc2c61aae1c11131
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/restack-pooler-unpooler-nn-layout-m128
worktree: .worktrees/muse_bf16_proj/restack-pooler-unpooler-nn-layout-m128
mode: null
restack_of: null
stale_base: null
---

# restack-pooler-unpooler-nn-layout-m128

## Change

Built on master (b99b1dd, the landed restack). Apply the exact change of branch hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the restack: in prepare(), store state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() (lossless [K, N] bf16 copies; the originals stay). In kernel_fn add one branch just before the `if layer != 'router'` cuBLAS fallback: if layer in {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n is the full width -> return torch.mm(x, state[layer + '_t']). m >= 129 keeps TN; m <= 32 keeps the restack's swap-AB paths. Nothing else changes. Record the nvjet NN kernel names for the two target rows.

## Expected

- `layer=pooler load=verify_16 m=128` faster: TN nvjet 64x64 25.7 us (row 25.55) -> NN splitK 20.4 + reduce 3.0, row 22.98 on master (-10.1%, z -29)
- `layer=unpooler load=verify_16 m=128` faster: row 19.79 -> 19.20 on master (-3%, z -9); small
- `layer=pooler load=verify_32 m=256` flat: m>=129 keeps TN
- `layer=unpooler load=verify_32 m=256` flat: m>=129 keeps TN
- `layer=* load=decode_*` flat: m<=32 untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Kill: pooler verify_16 row not <= 24.0 us (6% under restack) with the NN kernel in the profile. +134 MB device memory, untimed. Estimated headline +0.003 over the restack.

## Adversarial review

Reviewed 2026-10-25T042254Z by an adversarial subagent (model opus).

**Verdict: proceed.** A mechanical restack of a dispatch-only per-shape win that has already replicated twice. Pooler m=128 was 1.092x and 1.097x paired vs SGLang, with the deterministic nvjet ..._splitK_NNT pick. Unpooler m=128 was 1.051x and 1.047x. On current master (b99b1dd) these rows still fall through to the TN cuBLAS fallback at about 1.0x, so nothing landed shares the path and no companion is needed. Judge on the two rows' paired ratios and kernel names, not on the headline.

### Justification
Plausible, and now measured rather than inferred. Three H200 runs (probe 2dc5dce0 on dba4ffd, gated 0daf3faa on a8bb42a, master 1e90292c on b99b1dd) give: pooler verify_16 m=128 TN nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 24.5 us kernel, row 25.00 vs SGLang 24.86 (0.995x). NN picks nvjet_sm90_tst_128x64_64x8_1x2_h_bz_splitK_NNT 20.4-21.5 us plus splitKreduce 3.0 us, rows 22.86/22.98 at paired 1.092x/1.097x (err about 50 ns). Unpooler verify_16 m=128: TN nvjet 128x64_..._TNT 20.3 us, row 19.96 (1.000x). NN nvjet 128x64_..._NNT 19.2 us, rows 19.15/19.20 at 1.051x/1.047x. The same kernel names appeared in both NN runs, so this is a deterministic heuristic pick, not run-to-run cuBLAS drift. On master b99b1dd (blob 970b415), kernel_fn sends pooler/unpooler at m=65..128 past every Triton gate: _pooler_shape_ok m<=16, _unpool_shape_ok n==1024 and m<=16, swapab16 m<=16, swapab 17..32. They then reach `if layer != 'router': return _bf16_gemm_dispatch_impl(...)`, so the planned insertion point exists unchanged. The expected directions agree with TASK.md: one load of six (verify_16) improves by about 2.0 + 0.8 us of a roughly 140 us verify_16 load time (all weights 62), which is about +0.3-0.5% on the geomean. That is well inside the restack's own CI half-width (about 1.5%). The stated +0.003 headline is right, and the headline cannot confirm or refute it. WEAKEST LINK: the kill criterion is in absolute us ('pooler row <= 24.0 us'). LESSONS document cross-run nvjet swings on this row (23.9 vs 25.7 us TN), so an absolute threshold can be passed or failed by drift. Use the same-run paired ratio vs SGLang instead (pooler >= 1.06x and the _splitK_NNT kernel present), plus unpooler >= 1.02x with _NNT present. Second weakest: the harvest classifier will probably label it neutral/unproven-vs-best on the headline, as it labelled the 0daf3faa run 'regression' (1.1953 vs best 1.2215). That run was flat against its own base (8b9ad43 at 1.1984) and lost only on untouched m<=32 rows, which is drift. Promotion must be decided on the per-shape evidence. Otherwise this third replication is wasted.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (b99b1dd, the base): it gates m<=32 (swapab16 m<=16, swapab 17..32) and the router only. The m=128 pooler/unpooler rows in its own run are TN nvjet at 0.995x/1.000x, the same incumbent the a8bb42a run displaced. It cannot interact: each shape is a separate cold-L2 graph.
- bf16-m32-swapab-wgmma-triton, bf16-shape-gated-triton-winners-combo: m<=32 gates only. They do not touch m=65..128. Not a confounder.
- sglang-gemm-dispatch-parity: it made the TN incumbent the same _bf16_gemm_dispatch_impl call SGLang makes. That is why the paired ratio isolates TN to NN cleanly. Reverting it would change the reference comparison, not this change's mechanism. Not a confounder.
- router-triton-hilo-fused-control-splitk-all-m, router-hilo-at-m8: router only. Not a confounder.

### Interference and A/B needs
No companion. Both a gain and a loss on the two target rows are unambiguous without a second run. The only code difference on those rows is TN to NN on the same cuBLAS, measured paired against SGLang's TN in the same run, and the kernel name in top_kernels shows which heuristic entry ran. No landed hypothesis shares the m=65..128 pooler/unpooler path, so an ab-revert has nothing to revert. An ab-tune of a landed item would not touch these rows. A control would just be master, which was measured in 1e90292c. The effect has already replicated twice on two different bases, so a third paired read on the current base is the whole value of this run. Movement on any other row is drift (LESSONS: 1.5-2.5% per shape) and says nothing about this change.

### Open questions
- Builder: insert the branch exactly at master line ~1069, inside/just before `if layer != 'router'` and after every Triton gate, gated on cfg layer in {pooler, unpooler}, 65 <= cfg m <= 128 and cfg n == state[layer+'_t'].shape[1]. Use own constants (NN_GATED_LAYERS/NN_GATED_MIN_M/NN_GATED_MAX_M) as in the 0daf3faa patch. Apply the patch done/regression/pooler-unpooler-m128-nn-layout-gated.patch by 3-way merge. Only the docstring hunk and the ROUTER_WARM_MS anchor should need adjustment. The state[...+'_t'] = state[name].t().contiguous() copy must be a real contiguous [K, N] tensor, not a view. Add nothing else.
- Harvest: decide on the paired ratio of pooler/unpooler verify_16 m=128 and the presence of ..._splitK_NNT / ..._NNT in top_kernels, not on the absolute-us kill threshold and not on the headline verdict label. Three consistent reads (1.09x/1.05x) would be strong per-shape evidence for landing even if the headline reads 'unproven-vs-best'.
- Correctness coverage: no TASK test exercises pooler/unpooler in 65..128 (the tests use m=130 and m=1000 for unpooler, both outside the gate and still TN). Only the timed m=128 calls are checked against the reference. Both earlier NN runs passed them, and the math is identical cuBLAS fp32-accumulate, so the risk is low. Note it in the attempt file.
- Stacking hazard: queued bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (m=65..256) and qkv-gate-o-m128-wgmma-splitk-semaphore target nearby rows. Whichever lands second must order its branch against this one's gate explicitly.
- Follow-up, not this run: the probe measured pooler verify_32 m=256 NN at 1.034x paired (29.73 vs 30.74 us, nvjet 128x64_NNT 29.5 vs TN 30.7). Widening the pooler gate to 129..256 is a separate one-constant hypothesis. Keep unpooler m=256 (1.012x) and qkv_gate m=256 (0.943x loss) excluded.


## Implementation notes

- Applied done/regression/pooler-unpooler-m128-nn-layout-gated.patch with `git apply -3`. One conflict (prepare(): the restack replaced the ROUTER_WARM_MS warm-up loop with `_router_prepare(state)`); hand-merged by keeping `_router_prepare(state)` and adding the `_t` copy loop before it. The docstring and constants hunks merged cleanly. Own constants kept: NN_GATED_LAYERS, NN_GATED_MIN_M=65, NN_GATED_MAX_M=128.
- prepare(): `state[name + "_t"] = state[name].t().contiguous()` for pooler and unpooler, a real [K, N] contiguous copy (not a view). Originals stay.
- kernel_fn: branch inside `if layer != "router":`, after every Triton gate (swapab16, pooler, qkv, unpooler, o, swapab 17..32) and immediately before the `_bf16_gemm_dispatch_impl` fallback. Gate: layer in NN_GATED_LAYERS, 65 <= cfg m <= 128, cfg n == state[layer+"_t"].shape[1]; returns `torch.mm(x, state[layer+"_t"])`. m>=129 and m<=32 paths unchanged.
- Review fixes applied: branch placement, constants, real contiguous copy as above. The review's kill-criterion correction wins over the spec's absolute-us kill: judge on paired ratios of the two rows (pooler >= 1.06x with `_splitK_NNT` in the profile, unpooler >= 1.02x with `_NNT`), not on absolute us or the headline label.
- No TASK test covers m=65..128 for pooler/unpooler (tests use m=130 and m=1000 for unpooler, outside the gate); only the timed m=128 calls are checked against the reference. Record this in the attempt file.
- Stacking: the queued m=65..256 rewrites (bf16-verify-swapab-wgmma-cluster-dsmem-rewrite, qkv-gate-o-m128-wgmma-splitk-semaphore) must order their branch against this gate explicitly; whichever lands second decides precedence.
- Diff: 1 file, +17/-1. Not run (no GPU on this node).

## Rules check

- [x] CUDA-graph capturable: torch.mm on cfg-only gate, no syncs/allocation on input values (diff checked).
- [x] Nothing derived from `data` cached: `_t` copies come from weights in prepare().
- [x] Weights repacked losslessly in prepare() only: bf16 transpose copy.
- [x] Activations BF16 unchanged; KV cache n/a: not touched.
- [x] Attention n/a: projection layers only.
- [x] No timing patches, harness access, module edits, subprocesses or network (diff checked).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `CLEAN: .../restack-pooler-unpooler-nn-layout-m128/muse_bf16_proj/submission.py`; py_compile OK; __pycache__ removed.
