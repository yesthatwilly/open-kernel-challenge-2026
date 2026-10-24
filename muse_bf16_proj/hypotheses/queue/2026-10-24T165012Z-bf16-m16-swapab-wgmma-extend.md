---
problem: muse_bf16_proj
slug: bf16-m16-swapab-wgmma-extend
created: 2026-10-24T165012Z
base: master
area: bf16 GEMM dispatch (m<=16, swap-AB Triton)
change: >
  Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the
  tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate
  _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024,
  m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync;
  at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and
  +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's
  existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K,
  BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8
  run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's
  swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K
  256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36,
  4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K
  4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8,
  BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new
  prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n //
  64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE
  master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k,
  or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full
  weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in
  1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per
  layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX
  counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents
  are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3
  (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for
  these shapes.
expected:
  - shapes: layer=pooler load=decode_* m=1
    direction: faster
    why: incumbent _bf16_splitk_kernel 19.0 us (3.5 TB/s) vs floor 14.0; swap-AB wgmma at m=32 took pooler 9.3% faster with the identical grid; target <= 18 us
  - shapes: layer=pooler m=16
    direction: faster
    why: 19.5 us kernel vs 14.1 floor; same mechanism
  - shapes: layer=pooler m=8
    direction: faster
    why: 19.3 us kernel vs 14.0 floor
  - shapes: layer=qkv_gate m=1
    direction: faster
    why: _qkv_splitk_sem_kernel 11.8 us (3.2 TB/s) vs floor 7.9; m=32 swap-AB qkv gained 3.6%
  - shapes: layer=qkv_gate m=8
    direction: faster
    why: 12.0 us kernel vs floor 7.9
  - shapes: layer=qkv_gate m=16
    direction: faster
    why: 12.3 us kernel vs floor 7.9
  - shapes: layer=o m=1
    direction: faster
    why: _pooler_skinny_kernel 4.6 us vs floor 3.5; m=32 swap-AB o gained 4.1%
  - shapes: layer=o m=8
    direction: faster
    why: 5.2 us kernel vs floor 3.5
  - shapes: layer=o m=16
    direction: faster
    why: 4.8 us kernel vs floor 3.6
  - shapes: layer=unpooler m=1
    direction: faster
    why: slice _bf16_splitk_kernel 4.5 us vs floor 1.7 (8.4 MB at 1.9 TB/s), furthest from floor of the four
  - shapes: layer=unpooler m=8
    direction: faster
    why: slice kernel 4.6 us vs floor 1.8
  - shapes: layer=unpooler m=16
    direction: flat
    why: full weight stays cuBLAS nvjet
  - shapes: "* m=32"
    direction: flat
    why: master swap-AB path untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T165449Z
review_companions: null
---

# bf16-m16-swapab-wgmma-extend

## Change

Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024, m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync; at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K, BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8 run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K 256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36, 4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K 4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8, BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n // 64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k, or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in 1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3 (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for these shapes.

## Expected

- `layer=pooler load=decode_* m=1` faster: incumbent _bf16_splitk_kernel 19.0 us (3.5 TB/s) vs floor 14.0; swap-AB wgmma at m=32 took pooler 9.3% faster with the identical grid; target <= 18 us
- `layer=pooler m=16` faster: 19.5 us kernel vs 14.1 floor; same mechanism
- `layer=pooler m=8` faster: 19.3 us kernel vs 14.0 floor
- `layer=qkv_gate m=1` faster: _qkv_splitk_sem_kernel 11.8 us (3.2 TB/s) vs floor 7.9; m=32 swap-AB qkv gained 3.6%
- `layer=qkv_gate m=8` faster: 12.0 us kernel vs floor 7.9
- `layer=qkv_gate m=16` faster: 12.3 us kernel vs floor 7.9
- `layer=o m=1` faster: _pooler_skinny_kernel 4.6 us vs floor 3.5; m=32 swap-AB o gained 4.1%
- `layer=o m=8` faster: 5.2 us kernel vs floor 3.5
- `layer=o m=16` faster: 4.8 us kernel vs floor 3.6
- `layer=unpooler m=1` faster: slice _bf16_splitk_kernel 4.5 us vs floor 1.7 (8.4 MB at 1.9 TB/s), furthest from floor of the four
- `layer=unpooler m=8` faster: slice kernel 4.6 us vs floor 1.8
- `layer=unpooler m=16` flat: full weight stays cuBLAS nvjet
- `* m=32` flat: master swap-AB path untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched
- `layer=router *` flat: untouched

## Risk

At m<=16 these shapes are HBM-bound, not MMA-bound, so the m=32 gain (which removed 255-reg tokens-on-M spills) may not transfer: m=16 incumbents already run 96-ish regs. The plausible win is the async wgmma letting more loads stay in flight and fewer registers per CTA; the prepare() PTX/regs print attributes it either way. wgmma n16 wastes nothing at m=16 but computes 15 dead columns at m=1 (free under the BW bound). Per-shape judging: if only some layers gain, a combo keeps those (as with bf16-shape-gated-triton-winners-combo).

## Adversarial review

Reviewed 2026-10-24T165449Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean, attributable A/B. Each of the four m<=16 incumbents already uses the same grid, BLOCK_K, stage count and cp.async depth as the proposed swap-AB config, so tl.dot orientation (mma.sync+ldmatrix vs wgmma) is the only variable, and the 9 tests exercise the new path (m=1 qkv, m=7 pooler, m=5 unpooler slice). The archived evidence predicts flat for pooler and qkv_gate, with o at m=8 and the unpooler slice as the only real candidates, so judge it per shape on top_kernels, not on the headline.

### Justification
The mechanism is plausible in form: Triton gives tokens-on-M at BLOCK_M=16 mma.sync, and a 64-row weight tile on M with 16 tokens on N can lower to wgmma m64n16k16. Triton's MMAv3 rule (N%8==0, num_warps=4) allows that. But the evidence is weaker than the hypothesis claims, on three points. (1) The archived m<=16 incumbent diagnostics (pooler diag log) show n_regs=96, 0 spills, 120 KB smem, cp.async=90. Those are the same registers and bytes in flight as the swap-AB m=32 kernel (n_regs=96, cp.async=106, log 1602a4e947e64090). The 255-register pathology the m=32 win removed was the TMA variant only, and it does not exist at m<=16, as the Risk section itself concedes. (2) '+9.3% with the identical grid' was measured against SGLang's nvjet+splitKreduce at m=32 (21.97 -> 20.00 us shape time), not against the same-grid tokens-on-M Triton kernel at m<=16. (3) The cross-m kernel times from the same profiles predict flat. The swap-AB pooler ran 20.5 us of kernel at m=32 against the incumbent's 19.0/19.3/19.5 at m=1/8/16, and shape medians were 20.00 vs 19.85 at m=16. qkv_gate ran 12.1 us swap-AB at m=32 vs 11.8-12.3 incumbent. o ran 4.5 us vs 4.6/5.2/4.8. x bytes at m=32 are about 0.5 MB on a 67 MB stream, so halving tokens should change little. The pooler 'target <= 18 us' (3.7 TB/s) and the qkv/pooler 'faster' rows are therefore not supported; the most likely result is flat, within about 1 us of the incumbent. The only real candidates are o at m=8, whose 5.2 us is an outlier against 4.6 at m=1, and the unpooler slice. The slice is the furthest from its floor (4.5 us for 8.4 MB), but there the cost looks like fixed latency plus the 8-way split-K reduce rather than MMA issue, so wgmma may not help either. The expected directions agree with TASK.md: every changed shape is m in {1,8,16}, weight 62 in decode_1/decode_16/verify_1, and unpooler m=16 (full width, cuBLAS) and m>=17 are untouched. WEAKEST LINK: the assumption that the mma.sync+ldmatrix issue path, rather than HBM delivery at about 3.5 TB/s with 128 KB in flight per CTA, binds these streams at m<=16. The m=32 kernel times argue against it for the pooler and qkv_gate.

### Confounders on the current kernel
- bf16-shape-gated-triton-winners-combo (658d63b): this is the incumbent being replaced on exactly these shapes (pooler/qkv_gate/o at m<=16, unpooler slice). It is the control, not a confounder. Its per-layer constants (grid, BLOCK_K, stages, SPLIT_K) are copied into SWAPAB16_CFG, so a difference in top_kernels times comes from the orientation/MMA path. If that combo had used other tiles, the comparison would mix tile and orientation effects; as written it does not.
- bf16-m32-swapab-wgmma-triton (8b9ad43): this supplies the kernel and is the evidence base. Shared-state risk only: the new path must use its own ws/cnt keys and its own WS_M=16. Master's _swapab_forward hardcodes WS_M=SWAPAB_MAX_M (32) and BLOCK_T=SWAPAB_BLOCK_T (32), and _swapab_route_ok requires 17<=m and w.shape==(n,k). If the build reuses those helpers unchanged, the [SPLIT_K,16,n] workspace gets addressed with a 32-row stride, which is an out-of-bounds/overlapping write and a correctness failure, and the unpooler slice (full 8192x4096 weight, n=1024) fails the shape check. These are not measurement confounders, but they must be parametrized.
- sglang-gemm-dispatch-parity / router hypotheses: these cover a different layer or different shapes (m>=17 bf16 fallback, router), and the comparison is a paired ratio against SGLang in the same run. Not a confounder.

### Interference and A/B needs
No companion is warranted. (i) ab-revert of the shape-gated combo would compare swap-AB against cuBLAS, which the combo's own run already measured (5-17% Triton wins), so it adds nothing. (ii) ab-tune: the swap-AB config deliberately equals the incumbent's per layer, and tuning BLOCK_K or stages is a follow-up to a win, not something needed to read this result. (iii) control: the one sub-question, whether wgmma was actually emitted at n16 and at what register count, is answered by the prepare() PTX/regs print for both the new and the incumbent kernels in the same log. The shapes need no second run to read: Triton kernel times reproduce within about 1% (LESSONS), so per-shape top_kernels against the incumbent's archived times (pooler 19.0/19.3/19.5, qkv 11.8/12.0/12.3, o 4.6/5.2/4.8, unpooler slice 4.5/4.6 us) decide each layer independently. Faster rows go into a combo, slower rows revert to master's kernel. The headline will probably sit inside the ±1.5% CI and should not be the deciding signal. With bf16 checks spent and this going out as a scored submit, spending a second scored run on a companion is not justified.

### Open questions
- Does Triton emit wgmma for N=16 (m64n16k16) with the [BLOCK_K,16] K-major B tile, or fall back to mma.sync? The m=32 log only proves n32. If the PTX shows mma.sync, the run is a no-op that only says Triton will not reach wgmma at n16.
- Unpooler slice: SPLIT_K=8 with K_SLICE=512 and BLOCK_K=128 is only 4 K-steps per CTA at 3 stages, so pipeline fill and the transposed fp32 reduce dominate. Will the wgmma path's prologue and the epilogue layout conversion through smem make it slower than the 4.5 us incumbent?
- o at m=8 (5.2 us incumbent vs 4.6 at m=1) is the most likely real gain. Is that m=8 bump an incumbent artifact (for example BLOCK_M=16 masked load behaviour) that any non-M-specialized compile would remove? If so, attribute it to the compile, not to wgmma per se.
- With do_not_specialize=['M'], m=1..16 share one compile per layer, but Triton still specializes stride_xm/stride_om by divisibility. Confirm the warm-up at m=1 and at m=16 hits the same cache entry, so the timed call never compiles.
- If pooler and qkv_gate read flat or slower, as the m=32 kernel times predict, keep master's kernels for them in the follow-up combo and route only the layers that gained, with no rerun of identical code.

