---
problem: muse_bf16_proj
slug: pooler-m32-tma-splitk-extend
created: 2026-10-24T140257Z
base: master
area: pooler GEMM (m=17..32)
change: >
  Built on master (658d63b). Extend the combo's pooler Triton path to 17 <= cfg m <= 32 (the
  scored pooler decode_32 m=32 row) using the ONE m=32 Triton kernel that has measured a win:
  _bf16_splitk_tma_kernel copied verbatim from branch
  hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream (job 4e89311eefdc4bc3: pooler decode_32 20.8
  vs 22.1 us, 1.061x; kernel 21.0 us vs master's nvjet 20.2 + splitKreduce 2.0 us). Keep its
  configuration at BLOCK_M=32: grid (64, 2) = 128 CTAs, BLOCK_N 64, SPLIT_K 2, BLOCK_K 256, the
  weight via a Triton 3.8 TensorDescriptor.from_tensor(W, [64, 256]) built once in prepare() from
  the pooler weight, x a masked tl.load, num_stages 4 (with that branch's OutOfResources retry at
  3, recorded as a constant), and its ticket-semaphore reduce. Master's m <= 16 pooler path
  (_bf16_splitk_kernel, cp.async BLOCK_K 256, from the BLOCK_K=256 control, the combo's pick) is
  unchanged; only a new gate `layer == 'pooler' and 17 <= cfg m <= 32` plus the existing shape
  preconditions routes to the TMA kernel, placed before the cuBLAS fallback. Own state keys:
  state['pooler_tma_desc'], state['pooler_tma_ws'] fp32 [2, 32, 4096], state['pooler_tma_cnt']
  int32 zeros [64]; warm m in 17..32 in prepare() on synthetic x and re-zero the counter after.
  Nothing else changes (qkv_gate/o/unpooler at m=32 stay cuBLAS, their Triton m=32 rows all lost:
  0.858x, 0.907x, 0.91-0.97x).
expected:
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: master 21.6 us (nvjet 20.2 + 2.0 us reduce node) vs floor 14.1; the TMA kernel measured 20.8 us on this row (1.061x paired) in 4e89311eefdc4bc3; Triton reproduces within ~1%
  - shapes: layer=pooler load=decode_1 m=1
    direction: flat
    why: master m<=16 path unchanged
  - shapes: layer=pooler load=decode_16 m=16
    direction: flat
    why: unchanged
  - shapes: layer=pooler load=verify_1 m=8
    direction: flat
    why: unchanged
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-24T164414Z
review_companions: null
hold_reason: "adversarial review: Moot under master 8b9ad43 (bf16-m32-swapab-wgmma-triton, 1.198x): pooler decode_32 m=32 already runs _bf16_swapab_kernel at 20.0 us row / 20.5 us kernel vs SGLa"
outcome: expired
verdict_detail: "hold expired after 28 h: adversarial review: Moot under master 8b9ad43 (bf16-m32-swapab-wgmma-triton, 1.198x): pooler decode_32 m=32 already runs"
expired_at: 2026-10-25T211328Z
---

# pooler-m32-tma-splitk-extend

## Change

Built on master (658d63b). Extend the combo's pooler Triton path to 17 <= cfg m <= 32 (the scored pooler decode_32 m=32 row) using the ONE m=32 Triton kernel that has measured a win: _bf16_splitk_tma_kernel copied verbatim from branch hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream (job 4e89311eefdc4bc3: pooler decode_32 20.8 vs 22.1 us, 1.061x; kernel 21.0 us vs master's nvjet 20.2 + splitKreduce 2.0 us). Keep its configuration at BLOCK_M=32: grid (64, 2) = 128 CTAs, BLOCK_N 64, SPLIT_K 2, BLOCK_K 256, the weight via a Triton 3.8 TensorDescriptor.from_tensor(W, [64, 256]) built once in prepare() from the pooler weight, x a masked tl.load, num_stages 4 (with that branch's OutOfResources retry at 3, recorded as a constant), and its ticket-semaphore reduce. Master's m <= 16 pooler path (_bf16_splitk_kernel, cp.async BLOCK_K 256, from the BLOCK_K=256 control, the combo's pick) is unchanged; only a new gate `layer == 'pooler' and 17 <= cfg m <= 32` plus the existing shape preconditions routes to the TMA kernel, placed before the cuBLAS fallback. Own state keys: state['pooler_tma_desc'], state['pooler_tma_ws'] fp32 [2, 32, 4096], state['pooler_tma_cnt'] int32 zeros [64]; warm m in 17..32 in prepare() on synthetic x and re-zero the counter after. Nothing else changes (qkv_gate/o/unpooler at m=32 stay cuBLAS, their Triton m=32 rows all lost: 0.858x, 0.907x, 0.91-0.97x).

## Expected

- `layer=pooler load=decode_32 m=32` faster: master 21.6 us (nvjet 20.2 + 2.0 us reduce node) vs floor 14.1; the TMA kernel measured 20.8 us on this row (1.061x paired) in 4e89311eefdc4bc3; Triton reproduces within ~1%
- `layer=pooler load=decode_1 m=1` flat: master m<=16 path unchanged
- `layer=pooler load=decode_16 m=16` flat: unchanged
- `layer=pooler load=verify_1 m=8` flat: unchanged
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=unpooler *` flat: untouched
- `layer=router *` flat: untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched

## Risk

The only m in 17..32 Triton row with a measured win (the combo dropped it because it took the cp.async BLOCK_K 256 kernel, which read 1.014x at m=32). Small: ~1.2 us x 62 on one of 30 shapes, ~+0.3% headline, inside the CI, so it is judged by the pooler decode_32 row and kernel time, exactly as the combo was. Cheap to implement (verbatim kernel). Overlaps bf16-m32-swapab-wgmma-triton on this one row; if both run, the faster pooler m=32 row goes into the next combo. Its BLOCK_M=32 compile was 255 regs / mma.sync, so it is the floor for a Triton mma.sync m=32 pooler, not a stepping stone.

## Adversarial review

Reviewed 2026-10-24T164414Z by an adversarial subagent (model opus).

**Verdict: hold.** Moot under master 8b9ad43 (bf16-m32-swapab-wgmma-triton, 1.198x): pooler decode_32 m=32 already runs _bf16_swapab_kernel at 20.0 us row / 20.5 us kernel vs SGLang 22.0, which beats the TMA split-K kernel this hypothesis would install (20.8 us row / 21.0 us kernel in 4e89311eefdc4bc3). Rebased on master, the change would replace a faster wgmma kernel with a slower mma.sync one on the only row it targets.

### Justification
The mechanism (TMA weight stream, split-K 2, 128 CTAs, ticket reduce) was plausible against the base it names, 658d63b, where pooler m=32 was cuBLAS nvjet 20.2 + splitKreduce 2.0 us (21.6 us row) against a 14.1 us floor; a 1.061x paired gain on that row was measured. But the base is stale. FINDINGS (regenerated on master 8b9ad43) shows layer=pooler load=decode_32 m=32 at 20.0 us with _bf16_swapab_kernel 20.5 us (wgmma m64n32k16, same grid (64,2), split-K 2, BLOCK_K 256, 4 stages). The TMA kernel's own record is 20.8 us row / 21.0 us kernel and, per the hypothesis's Risk section, it compiled to 255 registers on mma.sync, which it calls 'the floor for a Triton mma.sync m=32 pooler'. So the expected 'faster' on pooler decode_32 has no support against the current incumbent; the expected direction is flat-to-slower (~+0.5-1 us). Weakest link: the 'expected' line compares to master 21.6 us, a number master no longer has. The headline effect would be ~0.3% at most even in its favour, well inside the CI, so it cannot be read from a scored run either.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (landed, master 8b9ad43): it already owns pooler at 17 <= m <= 32 with a faster kernel, so this change, rebased, would compete with it rather than with cuBLAS. A gate placed 'before the cuBLAS fallback' and after the swap-AB route never fires; placed before the swap-AB route it displaces the faster kernel. Either way any measured delta is an interaction with the swap-AB route, not the TMA kernel's own merit.
- bf16-shape-gated-triton-winners-combo (landed, 658d63b): the hypothesis's stated base; it deliberately left pooler m=32 on cuBLAS. That premise was removed by the swap-AB promotion.

### Interference and A/B needs
No companion makes this interpretable worth a scored submit. The only informative comparison is TMA split-K vs swap-AB wgmma on pooler decode_32, which existing records already answer (20.8/21.0 us vs 20.0/20.5 us). An ab-revert of bf16-m32-swapab-wgmma-triton would only reproduce job 4e89311eefdc4bc3's already-measured result on a worse base. The run is also a scored submit with checks spent, so spending it on an expected flat-to-slower, sub-CI change is not justified.

### Open questions
- If anyone wants more out of pooler m=32, the next lever is inside _bf16_swapab_kernel (wgmma path, 20.5 us vs 14.1 us floor), e.g. TMA loads of the weight into the swap-AB kernel. That would be a new hypothesis stacked on bf16-m32-swapab-wgmma-triton, not this one.
- Is there a measured paired comparison of the swap-AB and TMA split-K kernels on the same job? FINDINGS numbers come from different jobs (1602a4e947e64090 vs 4e89311eefdc4bc3). The gap (~0.5-0.8 us) is consistent but not paired.

