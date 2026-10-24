---
problem: muse_bf16_proj
slug: unpooler-full-triton-skinny-m32-control-blockm16-all-m
created: 2026-10-24T032454Z
base: master
area: unpooler GEMM
change: >
  From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid
  (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num_warps=4, fp32 accumulate
  and the bf16 store exactly as the parent, and change only the M tiling: BLOCK_M is 16 for every
  m in 9..32. For m > 16 the kernel keeps two fp32 accumulators [16, 64] and per K-step loads the
  one [64, 256] weight tile once and two [16, 256] x tiles (rows 0..15 and 16..31, rows >= m
  masked to zero), doing two tl.dot(x_i, tl.trans(w_tile), acc_i); for m <= 16 it is the parent's
  single-tile path (so unpooler decode_16 is byte-for-byte the parent's kernel and serves as the
  within-run control). Smem per stage drops to 32 KB weight + 2 x 8 KB x = 48 KB, 192 KB at 4
  stages (same as the parent at m=32; if the parent had to drop to 3 stages, this one may keep 4).
  Two stores, one per 16-row accumulator, masked on rows < m. Warm m in 9..32 in prepare() on
  synthetic zeros. Pooler and every other layer unchanged from the parent.
expected:
  - shapes: layer=unpooler load=decode_32 *
    direction: faster
    why: "than the parent's m=32 row if the pooler's 25.2-vs-20.2 us m=32 penalty is the BLOCK_M=32 tl.dot tile; equal to the parent's row if it is bandwidth"
  - shapes: layer=unpooler load=decode_16 *
    direction: flat
    why: identical kernel binary to the parent at m<=16; within-run control
  - shapes: layer=unpooler load=decode_1 *
    direction: flat
    why: slice path untouched
  - shapes: layer=unpooler load=verify_* *
    direction: flat
    why: m=8 slice and m>=128 stay on cuBLAS
risk: low
depends_on: unpooler-full-triton-skinny-m32
priority: 3
kind: control
controls_for: unpooler-full-triton-skinny-m32
review: proceed
status: running
review_at: 2026-10-24T052937Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T052937Z
base_sha: fbfb795472c2ae79a1b093a08fb5e42d1867da77
base_blob: 29884ed7c43020beb1ad7c469eafaa8b9f362470
branch: hyp/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m
worktree: .worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m
mode: null
---

# unpooler-full-triton-skinny-m32-control-blockm16-all-m

## Change

From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num_warps=4, fp32 accumulate and the bf16 store exactly as the parent, and change only the M tiling: BLOCK_M is 16 for every m in 9..32. For m > 16 the kernel keeps two fp32 accumulators [16, 64] and per K-step loads the one [64, 256] weight tile once and two [16, 256] x tiles (rows 0..15 and 16..31, rows >= m masked to zero), doing two tl.dot(x_i, tl.trans(w_tile), acc_i); for m <= 16 it is the parent's single-tile path (so unpooler decode_16 is byte-for-byte the parent's kernel and serves as the within-run control). Smem per stage drops to 32 KB weight + 2 x 8 KB x = 48 KB, 192 KB at 4 stages (same as the parent at m=32; if the parent had to drop to 3 stages, this one may keep 4). Two stores, one per 16-row accumulator, masked on rows < m. Warm m in 9..32 in prepare() on synthetic zeros. Pooler and every other layer unchanged from the parent.

## Expected

- `layer=unpooler load=decode_32 *` faster: than the parent's m=32 row if the pooler's 25.2-vs-20.2 us m=32 penalty is the BLOCK_M=32 tl.dot tile; equal to the parent's row if it is bandwidth
- `layer=unpooler load=decode_16 *` flat: identical kernel binary to the parent at m<=16; within-run control
- `layer=unpooler load=decode_1 *` flat: slice path untouched
- `layer=unpooler load=verify_* *` flat: m=8 slice and m>=128 stay on cuBLAS

## Risk

Separates the m=32 row's result into tile (BLOCK_M=32 wgmma tile / smem pipeline) versus streaming bandwidth: if the parent loses at m=32 and this control does not, the family's m=32 problem is the 32-row tile and o-/qkv-gate-triton-skinny should adopt BLOCK_M=16 everywhere; if both lose equally, the plain Triton stream's ceiling (~3.3 TB/s on the pooler) is confirmed on a second shape and the whole skinny-Triton family should be withdrawn.

## Adversarial review

Reviewed 2026-10-24T052937Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** One run resolves the control's question if it is read from the Triton kernel's own us and the paired-vs-SGLang ratio on unpooler decode_32, calibrated by the byte-identical decode_16 row in the same run, not from the +-4% cross-run attribution: Triton-kernel rows reproduce within ~1% across runs (pooler rows 20.2/20.4, 20.5/20.5, 25.2/25.4 us; paired ratios within 0.6%), so the 10% m=32 penalty the parent measured (19.8 us vs 18.0 us for identical weight bytes) is far above noise. A clear gain is unambiguous; a flat result is ambiguous but no cheap companion disambiguates it better than running this first, so no companions.

### Justification
Mechanism, corrected against the parent's actual rows: the parent did NOT lose at m=16. Its BLOCK_N=64 / 16-step kernel streamed the 67.1 MB unpooler in 18.0 us (3.73 TB/s, paired speedup 1.033 vs nvjet 18.3-19.1 us) where the pooler's BLOCK_N=32 / 32-step kernel does 20.2-20.5 us (3.3 TB/s) on the same bytes; so the plain-Triton 'ceiling' the earlier review assumed is already broken by this tile, and the only unpooler problem left is the m=32 row: 19.8 us kernel, paired 0.973, a +10% penalty over m=16 for the same weight traffic (the pooler shows +24%: 25.4 vs 20.5). The control changes exactly one thing, the M tiling at m>16 (two [16,64] accumulators and two tl.dot per K-step against one weight tile), with smem, FLOPs, x bytes and the K-step order unchanged, so numerics are row-for-row identical to the parent and the diff is attributable to codegen of the 32-row dot. Expected rows are consistent with TASK.md: unpooler decode_32 has weight 62 and is ~30% of the decode_32 load, floor 14.1 us; a full fix (19.8 -> 18.0) is ~+2.8% on that load, ~+0.9% on the geomean, inside the +-1.5% CI, so the headline cannot be the readout, the per-row kernel us and paired ratio must be. Weakest link: the hypothesis frames a dichotomy (32-row tl.dot tile vs streaming bandwidth) that does not exhaust the mechanisms, and the control removes less than it claims. At the ISA level a [32,64] mma.sync dot is two [16,64] dots sharing B fragments; the two-dot form keeps the same mma count, the same ldmatrix of x, the same cp.async fill (48 KB/stage) and the same 2x x re-read from L2 (128 programs x 256 KB = 32 MB at m=32 vs 16 MB at m=16; pooler: 64 MB vs 32 MB, matching the ordering of the penalties 24% > 10%). What it can change is Triton's warp layout ([2,2] -> [1,4]-style, which halves per-warp duplicate B-fragment reads from smem) and whether the compiler shares the w ldmatrix across the two dots (not guaranteed; the x operand of each dot is distinct, the B operand the same). So a flat result does not 'confirm the bandwidth ceiling on a second shape' as the Risk section says; it says the penalty is not the dot's M-shape, leaving x-traffic amplification, issue-rate (4 warps doing 2x mma+ldmatrix per step) and pipeline depth all open. The hypothesis's withdrawal clause for the whole family on a flat result is an over-inference and should not be acted on: the m=16 row already shows this kernel beating nvjet.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (landed, in the build base): inert on the two rows that can move; the parent already replaces _bf16_gemm_dispatch_impl on unpooler 9<=m<=32 and the control keeps that routing, so the measured delta vs the parent is independent of it.
- router-hilo-at-m8 and router-triton-hilo-fused-control-splitk-all-m (landed on master, ABSENT from the build base fbfb795 which descends from d2840d0 on the parity base): different layer, own CUDA graphs, no shared kernel or memory; they cannot move the unpooler rows, but the headline will sit near 1.00 like the parent's 1.0053 and is not comparable to BEST 1.151x. Promotion of a gain needs a rebase onto master and a fresh run.
- pooler-triton-skinny-m32 (parent of the parent, unpromoted, in the build base): shares @triton.jit _pooler_skinny_kernel. If the implementer adds a constexpr (e.g. TWO_TILES) to that shared kernel, every pooler specialization and the unpooler m<=16 specialization recompile under a new cache key; same code, but the 'byte-for-byte identical binary' claim for the decode_16 in-run control weakens. Its pooler decode_32 loss (0.888) stays in the headline either way and must be attributed away.
- Within the parent itself: unpooler decode_16 and decode_32 are different graphs and different m; the control's decode_16 row is a drift calibrator (expect ~18.0 us), not a same-shape A/B, so the m=32 readout is kernel_us(m=32)/kernel_us(m=16) in the same run (parent: 19.8/18.0 = 1.10) against the parent's ratio.

### Interference and A/B needs
Noise: the +-4% cross-run swings the prompt cites are on cuBLAS split-K rows (qkv_gate nvjet 10.8 -> 9.5 us between runs while SGLang's own time moved too); Triton-kernel rows reproduce within ~1% across the pooler run and the parent run (20.2/20.4, 20.5/20.5, 25.2/25.4 us; paired ratios 1.059/1.058, 1.063/1.057, 0.894/0.888), and err_ns on these rows is 40-60 ns (0.3%). The effect under test is 19.8 vs 18.0 us (10%), so one run resolves it: decision rule, m=32 kernel us <= 18.6 (ratio to the in-run m=16 row <= 1.03) -> the 32-row dot codegen was the penalty, adopt BLOCK_M=16 tiles across the skinny family; >= 19.4 (ratio >= 1.08) -> not the tile; 18.6-19.4 -> partial, UNSURE. ab-revert: not applicable, no landed hypothesis touches the unpooler 9..32 path and the router changes are absent from the build base anyway. ab-tune: the pooler tile has no effect on unpooler rows. control: a gain is unambiguous on its own, so by the stated criterion no companion is warranted now. The one control that would disambiguate a FLAT result (issue-rate vs bandwidth) is num_warps=8 at the m=32 specialization only, same smem and tiles; it is a one-constant change and should be queued only if this run comes back flat, not spent in parallel, because its interpretation depends on this run's answer.

### Open questions
- Implement the two-tile path as a separate @triton.jit kernel (not a new constexpr on _pooler_skinny_kernel) so the pooler specializations and the unpooler m<=16 specialization are literally the parent's binaries and the decode_16 row is a true in-run control.
- Readout: judge on top_kernels us for unpooler decode_32 vs decode_16 in the same run (parent 19.8 vs 18.0) and on the paired speedup (parent 0.973 vs 1.033), not on the attribution delta vs the parent's medians (19.27/17.97) which carries the +-4% cross-run noise; record 67.1 MB / kernel us as the family's bandwidth datapoint (the m=16 row is 3.73 TB/s, already above nvjet).
- If flat: the penalty is not the dot's M-shape; the remaining candidates are x re-read amplification from L2 (32 MB at m=32 vs 16 MB at m=16 for 128 programs), 4-warp issue rate on 2x mma+ldmatrix per step, and pipeline depth; queue a num_warps=8-at-m=32 control next. Do not withdraw the o-/qkv-gate siblings on a flat result here: the qkv-gate regression (BLOCK_N=16, 22-26 us vs 14 us cuBLAS) is a BLOCK_N / program-count problem, not an m=32 tile problem.
- If gain: unpooler decode_16 and decode_32 both beat nvjet by ~3%; the gain only reaches the board after rebasing the parent + control onto master (router-triton-hilo-fused-control-splitk-all-m is not in the build base) and a fresh run; also revisit the pooler with BLOCK_N=64 (64 programs) or BLOCK_M=16 tiles, since the parent's data say the pooler's 3.3 TB/s is the BLOCK_N=32 / 32-step tile, not a Triton ceiling.
- Frontmatter says base: master while depends_on names the parent; confirm churn.py bases on fbfb795 (the parent's impl_commit) as the prompt states, otherwise the comparison rows are wrong.

