---
problem: muse_bf16_proj
slug: pooler-nn-layout-widen-m256
created: 2026-10-25T145403Z
base: master
area: pooler GEMM (m=129..256, cuBLAS NN layout)
change: >
  Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128,
  submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous()
  ([8192, 4096] bf16, lossless repack) in prepare() and routes pooler/unpooler at
  NN_GATED_MIN_M=65 <= cfg m <= NN_GATED_MAX_M=128 to torch.mm(x, state[layer + '_t']) in
  kernel_fn. Widen that gate for the POOLER ONLY to cfg m <= 256: add a new constant
  NN_POOLER_MAX_M = 256 (keep NN_GATED_MAX_M = 128 for the unpooler) and change the kernel_fn
  condition so that layer == 'pooler' uses NN_POOLER_MAX_M as its upper bound while layer ==
  'unpooler' keeps 128. Mirror the same per-layer bound in _ab_component (so the in-process A/B
  self-check attributes a pooler m in 129..256 cell to 'nn_ok' and can switch it off) and, if
  cheap, add m=256 pooler to the self-check's captured cells. No new weight copy (pooler_t already
  exists), no new kernel, no other layer or m changes: qkv_gate and o stay TN at every m (the
  probe LOST qkv_gate m=256, 0.943x), unpooler m=256 stays TN (probe 1.012x, noise), and every m
  <= 128 path is byte-identical to master. The gate branches on cfg layer and m only. Evidence:
  bf16-m128-nn-weight-layout-probe (job 2dc5dce0) ran pooler m=256 through NN at 29.73 us row vs
  SGLang 30.74 (paired 1.034x, nvjet 128x64_64x8_1x2 NNT 29.5 us); the TN incumbent on the same
  row read 0.986x (restack 3-cell, nvjet TNT 30.4 us), 0.999x (router-cluster-s2, 28.5 us) and
  0.9998x (pooler-unpooler-m128-nn-layout-gated) in three other runs. The review of
  pooler-unpooler-m128-nn-layout-gated named this widening as its follow-up. The attempt file
  records the pooler verify_32 nvjet kernel name (must carry the NN suffix, e.g. _NNT) and us from
  the profile; if the name is still TNT the run is void.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: "NN-layout nvjet read 1.034x paired vs SGLang in the probe against 0.986-1.000x for the TN incumbent in three runs: about -0.8 to -1.4 us on a 31.3 us row (floor 17.4)"
  - shapes: layer=pooler load=verify_16 m=128
    direction: flat
    why: already NN on master; unchanged
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched (stays TN; NN lost at m=256)
  - shapes: layer=unpooler *
    direction: flat
    why: untouched; unpooler keeps the 128 bound
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
review_at: 2026-10-25T150319Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T150319Z
base_sha: b7d110b74193aba420325d572c4db747ff74118c
base_blob: 353e3b19a7edd5cb879e202d78097f474bcce207
branch: hyp/muse_bf16_proj/pooler-nn-layout-widen-m256
worktree: .worktrees/muse_bf16_proj/pooler-nn-layout-widen-m256
mode: null
restack_of: null
stale_base: null
---

# pooler-nn-layout-widen-m256

## Change

Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous() ([8192, 4096] bf16, lossless repack) in prepare() and routes pooler/unpooler at NN_GATED_MIN_M=65 <= cfg m <= NN_GATED_MAX_M=128 to torch.mm(x, state[layer + '_t']) in kernel_fn. Widen that gate for the POOLER ONLY to cfg m <= 256: add a new constant NN_POOLER_MAX_M = 256 (keep NN_GATED_MAX_M = 128 for the unpooler) and change the kernel_fn condition so that layer == 'pooler' uses NN_POOLER_MAX_M as its upper bound while layer == 'unpooler' keeps 128. Mirror the same per-layer bound in _ab_component (so the in-process A/B self-check attributes a pooler m in 129..256 cell to 'nn_ok' and can switch it off) and, if cheap, add m=256 pooler to the self-check's captured cells. No new weight copy (pooler_t already exists), no new kernel, no other layer or m changes: qkv_gate and o stay TN at every m (the probe LOST qkv_gate m=256, 0.943x), unpooler m=256 stays TN (probe 1.012x, noise), and every m <= 128 path is byte-identical to master. The gate branches on cfg layer and m only. Evidence: bf16-m128-nn-weight-layout-probe (job 2dc5dce0) ran pooler m=256 through NN at 29.73 us row vs SGLang 30.74 (paired 1.034x, nvjet 128x64_64x8_1x2 NNT 29.5 us); the TN incumbent on the same row read 0.986x (restack 3-cell, nvjet TNT 30.4 us), 0.999x (router-cluster-s2, 28.5 us) and 0.9998x (pooler-unpooler-m128-nn-layout-gated) in three other runs. The review of pooler-unpooler-m128-nn-layout-gated named this widening as its follow-up. The attempt file records the pooler verify_32 nvjet kernel name (must carry the NN suffix, e.g. _NNT) and us from the profile; if the name is still TNT the run is void.

## Expected

- `layer=pooler load=verify_32 m=256` faster: NN-layout nvjet read 1.034x paired vs SGLang in the probe against 0.986-1.000x for the TN incumbent in three runs: about -0.8 to -1.4 us on a 31.3 us row (floor 17.4)
- `layer=pooler load=verify_16 m=128` flat: already NN on master; unchanged
- `layer=qkv_gate *` flat: untouched (stays TN; NN lost at m=256)
- `layer=unpooler *` flat: untouched; unpooler keeps the 128 bound
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Expected headline effect is small (verify_32 is the weakest load at 1.075x; ~1 us of 94.5 us there is ~+0.2% geomean), so judge on the paired pooler verify_32 ratio, not the headline. Risk: the probe was one run on an older master; cuBLAS NN may pick the same 128x64 tile at m=256 and the 1.034x could be partly drift. Downside is bounded to one row (worst case ~0.97x, the qkv m=256 NN loss magnitude). Shape coverage: the test m=130 unpooler is unaffected (unpooler bound stays 128); no pooler test sits in 129..256, so nearby-shape behaviour equals cuBLAS NN, the same math. Overlap: queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 also targets pooler m=256; if both ever stack, branch order decides, so this keeps its own gate constant NN_POOLER_MAX_M.

## Adversarial review

Reviewed 2026-10-25T150319Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a low-risk change gated on config only. It reuses the existing lossless pooler_t copy for one row, pooler verify_32 m=256. The headline cannot show a gain this small (about +0.2% geomean against a CI of at least ±1.5%), so judge it on the row's paired ratio and the nvjet NNT kernel name. No companion is needed.

### Justification
Mechanism: on master, pooler m=256 falls through every Triton gate (pooler split-K and swap-AB16 are m<=16/32, swap-AB is 17..32) to _bf16_gemm_dispatch_impl. That path runs cuBLAS nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT at 30.4 us on a 31.3 us row, against SGLang's 30.9 and a 17.4 floor, so it is about 1.8x off the floor and roughly at parity with SGLang. The same NN copy already wins at m=128 on master (pooler verify_16 22.9 vs 25.5, splitK_NNT). So the [K,N] layout changing cuBLAS's heuristic and tile/split choice for this K-heavy (k=8192, n=4096) weight is plausible on H200. The expected shapes and directions agree with TASK.md: only pooler m=256 should move, and every other row is byte-identical because the gate branches on cfg layer/m/n only. The predicted -0.8 to -1.4 us on one verify_32 shape is about 1% of that load (94.5 us), so roughly +0.2% headline, well inside the CI. Weakest link: the evidence is ONE probe run (2dc5dce0) on an old master (dba4ffd), and it chose the SAME tile (128x64_64x8_1x2) as the TN incumbent, at 29.5 us NNT vs 28.5-30.4 us TNT across runs. The raw kernel-us gap is inside cross-run drift. Only the within-run paired ratio (1.034 vs 0.986/0.999/0.9998) separates them, and a single 3.5% paired-ratio gap on one row has never been replicated. The m=128 win also came with a splitK NNT variant, while the m=256 probe kernel is non-splitK, so the m=128 mechanism (a different split choice) does not obviously carry over.

### Confounders on the current kernel
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128: introduced pooler_t, the NN gate and the shared nn_ok flag. Widening reuses that flag, so if the proposed new m=256 A/B self-check cell MISSes, nn_ok turns off for BOTH layers at m=128 as well. pooler/unpooler verify_16 would then regress and confound the read. The attempt must log the final flags (nn_ok=True) or the run is void.
- sglang-gemm-dispatch-parity: the incumbent is _bf16_gemm_dispatch_impl and the new path is plain torch.mm. The measured delta therefore mixes the NN layout with whatever the dispatch impl does differently (cuBLASLt vs cuBLAS handle, workspace, heuristic). That does not change the decision, since either cause is a legitimate lossless gain, but the gain should not be credited to 'layout' alone in FINDINGS without that caveat.
- Other landed items (router hilo/cluster, swap-AB m<=32, unpooler-slice cluster) touch different (layer, m) cells and do not share this launch or its memory traffic. Every timed replay starts with a cold L2, so the extra 67 MB pooler_t residency has no cross-row effect.

### Interference and A/B needs
No companion. An ab-revert of the restack would remove pooler_t itself, so the change could not exist. An ab-tune of the gate is the change itself. A control isolating torch.mm-TN vs dispatch-impl-TN would only split the credit between layout and call path, and both outcomes lead to the same keep/drop decision. The single run is already interpretable from its own row. The incumbent's paired ratio on pooler verify_32 has three prior reads (0.986, 0.999, 0.9998), which gives a usable baseline distribution, and the kernel-name check (must be *_NNT) confirms the route actually changed. Decision rule: call it a gain if pooler verify_32 paired >= ~1.02 with an NNT kernel, call it a loss if <= ~0.97, and treat anything between as noise to log as neutral. Spending a second scored run to firm up a ~0.2% headline item is not justified.

### Open questions
- Should the optional m=256 A/B self-check cell get its own flag (e.g. nn_pooler256_ok) so a miss there cannot switch off the proven m=128 NN cells? Prefer either a separate flag or skipping the extra cell. AB_BF16_REL_L2=6e-3 vs ~2e-3 bf16 rounding makes a false miss unlikely but not impossible at k=8192.
- Does nvjet pick a different algorithm for NN at m=256 on the current driver/master than in the probe (128x64_64x8_1x2 NNT)? Record the exact kernel name and us. If it is the same tile as the TNT incumbent, any gain is weakly explained.
- Ordering with the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2, which also targets pooler m=256: whichever lands second must re-baseline this row against the other, not against the current master.

