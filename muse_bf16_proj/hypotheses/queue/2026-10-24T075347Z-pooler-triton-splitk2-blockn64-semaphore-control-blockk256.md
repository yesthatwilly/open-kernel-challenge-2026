---
problem: muse_bf16_proj
slug: pooler-triton-splitk2-blockn64-semaphore-control-blockk256
created: 2026-10-24T075347Z
base: master
area: pooler GEMM
change: >
  From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one
  constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each CTA's K-slice of 4096 is
  16 K-steps instead of 32. Keep num_stages=4, num_warps=4, BLOCK_N=64, SPLIT_K=2, grid (64, 2) =
  128 CTAs, BLOCK_M=16 for m<=16 and 32 for 17..32. Keep the direct transposed weight addressing,
  the fp32 workspace, the barrier + acq_rel ticket + last-CTA fixed-order reduce and atomic_xchg
  reset, the routing (pooler m<=32 only), and the warm-up plus re-zero in prepare(). Smem becomes
  4 x (64x256 + BLOCK_M x256) x 2 B = 160 KB at BLOCK_M=16 and 192 KB at BLOCK_M=32. This is
  exactly the unpooler-full-triton-skinny-m32 per-CTA program, and it fits H200's 227 KB. If
  Triton rejects 192 KB at BLOCK_M=32, use num_stages=3 for the BLOCK_M=32 specialization only and
  record that.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: 128 KB of weight in flight per CTA, the 3.73 TB/s unpooler config. Expect about 18-18.5 us plus the reduce tail, below the parent if the parent is stage-bound at about 20.3 us
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: same as decode_1, BLOCK_M=16 binary
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: the exact m=16 unpooler program that measured 17.97 us on the same 67.1 MB
  - shapes: layer=pooler load=decode_32 *
    direction: flat
    why: "unpooler BLOCK_M=32 at this tile measured 19.27 us (paired 0.973); expect about 20 us vs nvjet's 20.2+1.9, roughly flat to slightly faster"
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 stays on cuBLAS
  - shapes: layer=pooler load=verify_32 *
    direction: flat
    why: m=256 stays on cuBLAS
  - shapes: "*layer=router*"
    direction: flat
    why: master router untouched
risk: low
depends_on: pooler-triton-splitk2-blockn64-semaphore
priority: 1
kind: control
controls_for: pooler-triton-splitk2-blockn64-semaphore
review: proceed
status: queued
hold_reason: null
review_at: 2026-10-24T104248Z
review_companions: null
---

# pooler-triton-splitk2-blockn64-semaphore-control-blockk256

## Change

From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each CTA's K-slice of 4096 is 16 K-steps instead of 32. Keep num_stages=4, num_warps=4, BLOCK_N=64, SPLIT_K=2, grid (64, 2) = 128 CTAs, BLOCK_M=16 for m<=16 and 32 for 17..32. Keep the direct transposed weight addressing, the fp32 workspace, the barrier + acq_rel ticket + last-CTA fixed-order reduce and atomic_xchg reset, the routing (pooler m<=32 only), and the warm-up plus re-zero in prepare(). Smem becomes 4 x (64x256 + BLOCK_M x256) x 2 B = 160 KB at BLOCK_M=16 and 192 KB at BLOCK_M=32. This is exactly the unpooler-full-triton-skinny-m32 per-CTA program, and it fits H200's 227 KB. If Triton rejects 192 KB at BLOCK_M=32, use num_stages=3 for the BLOCK_M=32 specialization only and record that.

## Expected

- `layer=pooler load=decode_1 *` faster: 128 KB of weight in flight per CTA, the 3.73 TB/s unpooler config. Expect about 18-18.5 us plus the reduce tail, below the parent if the parent is stage-bound at about 20.3 us
- `layer=pooler load=verify_1 *` faster: same as decode_1, BLOCK_M=16 binary
- `layer=pooler load=decode_16 *` faster: the exact m=16 unpooler program that measured 17.97 us on the same 67.1 MB
- `layer=pooler load=decode_32 *` flat: unpooler BLOCK_M=32 at this tile measured 19.27 us (paired 0.973); expect about 20 us vs nvjet's 20.2+1.9, roughly flat to slightly faster
- `layer=pooler load=verify_16 *` flat: m=128 stays on cuBLAS
- `layer=pooler load=verify_32 *` flat: m=256 stays on cuBLAS
- `*layer=router*` flat: master router untouched

## Risk

Separates stage depth (64 KB vs 128 KB of weight in flight per CTA, the only measured difference between the 3.3 and 3.73 TB/s Triton configs) from the parent's own design (split-K 2 + BLOCK_N=64 + semaphore reduce). Parent about 20.3 us and control about 18.5 us: the parent's spec, not its design, was at fault, and the control is the one to promote. Both about 20.3 us: split-K/semaphore costs as much as it saves. Both about 18.5 us: BLOCK_N=64 + split-K is what matters.

## Adversarial review

Reviewed 2026-10-24T104248Z by an adversarial subagent (model opus).

**Verdict: proceed.** Run it as specified. It is a one-constant (BLOCK_K 128->256) control of the neutral parent, and it is the cp.async arm that pooler-splitk2-tma-weight-stream needs before any gain can be credited to TMA. The evidence supports it: every 64 KB-in-flight Triton pooler stream ran at about 20 us, and the one 128 KB config ran at 18.0 us twice. No landed hypothesis confounds it and it raises no rules issue, so no companion is needed.

### Justification
Mechanism. At m<=32 the pooler is a pure 67.1 MB bf16 weight stream with a 14.0-14.1 us floor. The parent's _bf16_splitk_kernel (BLOCK_N=64, BLOCK_K=128, 4 stages = 64 KB of weight in flight per CTA) took 19.5/20.0/19.9/22.7 us of kernel time at m=1/8/16/32, with shape times of 20.19/19.87/20.67/22.20 us. pooler-triton-skinny-m32 (BLOCK_N=32, BLOCK_K=256, 4 stages, also 64 KB in flight, full K) took 20.2/20.2/20.5 us at m=1/8/16. Two different geometries with the same bytes in flight land at the same ~3.3-3.4 TB/s. The only 128 KB-in-flight config (BLOCK_N=64, BLOCK_K=256, 4 stages, 64 rows x 4096 K per CTA, 128 CTAs) ran the unpooler's 67.1 MB in 18.0 us of kernel time at m=16, and it reproduced: 17.97 us shape / 18.0 us kernel in unpooler-full-triton-skinny-m32, and 18.37 / 18.0 us in its BLOCK_M=16 control. So 'bytes in flight is the limiter' is the best-supported explanation and the control tests it directly. Predicted savings are about 1.5-2 us of kernel time on decode_1, verify_1 and decode_16. Each of those rows is about 31-38% of its load at equal weight 62. decode_32 marked flat is conservative: the unpooler's BLOCK_M=32 version of this tile ran 19.8 us of kernel time (19.27 us shape) against the parent's 22.7 us here, so 'faster' is at least as likely. verify_16/32 (cuBLAS) and the router rows are valid in-run flat controls. Smem is 160/192 KB, the same as the unpooler kernel, which compiled. 4096/256 = 16 K-steps, so there are no remainder issues. WEAKEST LINK: the spec calls this 'exactly the unpooler-full-triton-skinny-m32 per-CTA program', and it is not exact in two ways. (1) Weight-tile addressing: the unpooler kernel (_pooler_skinny_kernel) loads w as [BLOCK_N, BLOCK_K] as stored and uses tl.dot(x, tl.trans(w)). The parent and this control load [BLOCK_K, BLOCK_N] directly through transposed pointers (w_ptr + offs_k[:,None] + offs_n[None,:]*stride_wn). Both are K-contiguous in memory, so the global loads should coalesce the same way, but Triton may choose a different shared layout or ldmatrix path. (2) Problem geometry: the pooler weight's row stride is 16 KB vs the unpooler's 8 KB, x is [m, 8192], and there is a split-K semaphore reduce tail of about 0.3-0.5 us. The 18.0 us precedent therefore predicts roughly 18.3-18.8 us here, not 18.0. If the control comes back near 20 us, the in-flight-bytes story is falsified for this kernel, and the difference must be in addressing layout or row stride, not stage depth.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m (master 1.151x/1.143x best): different layer, no shared workspace or counters, cold L2 per replay, so it cannot move the pooler rows. The router rows are a reproducibility check only. The parent's router verify_1/verify_16 rows read 'lost' (0.962/0.952) with untouched code, which shows that headline and router rows carry run drift.
- sglang-gemm-dispatch-parity: defines the cuBLAS path that pooler m>=128 and every other layer keep. Those rows are in-run flat controls, not an interaction.
- router-hilo-at-m8: router-only and subsumed by the Triton router. It shares no path with the pooler and is not a confounder.
- Structural, not landed: the base is the neutral parent branch (1.166x), not master (1.143x), so a headline delta against BEST includes the parent's own ~+5-7% pooler-row effect. Read this run per pooler row (top_kernels us) against the parent's 19.5/20.0/19.9/22.7 us, and pair it with the TMA arm's rows, not with the headline.

### Interference and A/B needs
No companion is warranted. No landed hypothesis touches the pooler kernel, its memory traffic or its numerics, so ab-revert and ab-tune have nothing to isolate. The run is itself a control, of the parent and of the TMA hypothesis, and changes one constant, so its readout is unambiguous either way. Near 18.3-18.8 us at m<=16: stage depth was the parent's limiter, this variant is the promotable one, and TMA must beat this run's rows, not the parent's, to earn credit. Near 19.5-20 us: in-flight bytes is not the limiter in this kernel, and any TMA gain belongs to the copy mechanism or the [BLOCK_N, BLOCK_K] layout. A further control (the unpooler's tl.trans addressing at BLOCK_K=256) would only matter in that second case, and it can be queued after this result, not run alongside. Because the TMA arm differs from this control in both copy mechanism and tile addressing, compare the two runs' kernel-us rows directly. A difference under ~0.3 us (about 1.5%, close to Triton's ~1% reproducibility) is not attributable.

### Open questions
- If Triton rejects 192 KB at BLOCK_M=32 and the implementer drops to num_stages=3 for that specialization, decode_32 becomes a two-constant change (96 KB in flight). Record it, read decode_32 separately, and do not pool it with m<=16.
- Have prepare() print the compiled kernel's n_regs, n_spills and shared bytes for both BLOCK_M specializations, and grep the PTX for ldmatrix(.trans) and cp.async.cg counts. That shows whether the [BLOCK_K, BLOCK_N] direct addressing lowers the same way as the unpooler's [BLOCK_N, BLOCK_K] + tl.trans, the one non-identity with the 18.0 us precedent.
- Does the doubled per-step x tile ([BLOCK_M, 256], L2-resident) add measurable latency at BLOCK_M=32? The unpooler's 19.8 us there says probably not, but the pooler's x row is twice as long.
- Readout rule for churnsubs: if this control and the TMA arm both land at about 18.5 us, promote this control (simpler, no Triton-version risk), and do not port TMA to qkv_gate or unpooler.

