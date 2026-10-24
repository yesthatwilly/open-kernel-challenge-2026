---
problem: muse_bf16_proj
slug: router-swapab-wgmma-all-m
created: 2026-10-24T165036Z
base: master
area: router GEMM
change: >
  Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M
  side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32,
  16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @
  hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the
  16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run
  measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight
  tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new
  _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t,
  pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi
  / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T]
  (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc =
  tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on
  tokens < m), out zeroed right before the launch exactly as master. Static configs from m only
  (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256;
  m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once
  instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice
  512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if
  OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split,
  _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32
  fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket,
  and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new
  kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation
  change is what moved the MMA path.
expected:
  - shapes: layer=router m=1
    direction: faster
    why: _router_hilo_kernel 3.2 us + 0.7 us memset vs floor 1.8 (8.4 MB hi+lo); wgmma m64n16 instead of mma.sync with 128 CTAs and 2 K-steps each
  - shapes: layer=router m=8
    direction: faster
    why: 3.2 us kernel, same as m=1
  - shapes: layer=router m=16
    direction: faster
    why: 3.5 us kernel vs floor 1.8
  - shapes: layer=router m=32
    direction: faster
    why: 4.1 us kernel; the BLOCK_M=32 tokens-on-M tile is the one the bf16 m=32 run measured as wgmma=0
  - shapes: layer=router m=128
    direction: faster
    why: 4.9 us kernel vs floor 2.2; grid M dim 2 re-reads each weight tile twice; swap-AB with 128 tokens on N reads it once (8.4 MB instead of 16.8 MB of L2/HBM weight traffic)
  - shapes: layer=router m=256
    direction: faster
    why: 6.9 us kernel vs floor 2.7; weight read 4x today, 2x with BLOCK_T 128
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T165424Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T165424Z
base_sha: 278c4441e3dddcc35e9478bc678c6d596a9d3428
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/router-swapab-wgmma-all-m
worktree: .worktrees/muse_bf16_proj/router-swapab-wgmma-all-m
mode: null
---

# router-swapab-wgmma-all-m

## Change

Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32, 16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @ hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the 16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t, pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T] (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc = tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on tokens < m), out zeroed right before the launch exactly as master. Static configs from m only (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256; m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice 512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split, _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32 fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket, and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation change is what moved the MMA path.

## Expected

- `layer=router m=1` faster: _router_hilo_kernel 3.2 us + 0.7 us memset vs floor 1.8 (8.4 MB hi+lo); wgmma m64n16 instead of mma.sync with 128 CTAs and 2 K-steps each
- `layer=router m=8` faster: 3.2 us kernel, same as m=1
- `layer=router m=16` faster: 3.5 us kernel vs floor 1.8
- `layer=router m=32` faster: 4.1 us kernel; the BLOCK_M=32 tokens-on-M tile is the one the bf16 m=32 run measured as wgmma=0
- `layer=router m=128` faster: 4.9 us kernel vs floor 2.2; grid M dim 2 re-reads each weight tile twice; swap-AB with 128 tokens on N reads it once (8.4 MB instead of 16.8 MB of L2/HBM weight traffic)
- `layer=router m=256` faster: 6.9 us kernel vs floor 2.7; weight read 4x today, 2x with BLOCK_T 128
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

LESSONS says the router's small-m grid depth is settled (SPLIT_K 32 and stage counts flat); this changes the MMA orientation and the weight re-read, not the grid depth, but at m<=16 it also moves SPLIT_K 16 -> 32 and K-slice 512 -> 256, which the split-K-32 runs found flat, so a small-m loss would point at the orientation. The 0.7 us vectorized_elementwise memset stays (removing it was flat: router-splitk-semaphore-no-memset). Router accuracy bound is 0.01%: the hi+lo fp32 accumulation is unchanged, only operand order and atomic order differ (atomic order is already nondeterministic on master). At m=128/256 the x^T tile [128, 128] bf16 is 32 KB per stage, acc 64x128 fp32 = 64 regs/thread: watch spills in the print.

## Adversarial review

Reviewed 2026-10-24T165424Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound port of master's proven swap-AB orientation to the router, and no rule is at issue. The expected win is weaker than stated, though. At m<=16 the MMA path is probably not the binding cost, and at m=128/256 the 'weight re-read' is mostly an L2 hit, while the new configs double the fp32 atomic count (SPLIT_K 16->32 at m=128, 8->16 at m=256). Run it once and read it from the router rows' _router_*_kernel times and the prepare() PTX/regs print, not from the headline.

### Justification
A. The mechanism is plausible, but it is a smaller lever than claimed. (1) MMA path. Triton's wgmma lowering needs 64 rows on tl.dot's M per warpgroup. Master's _bf16_swapab_kernel showed that weight[64, BLOCK_K] as the LHS and x^T[BLOCK_K, BLOCK_T] as the RHS emits wgmma with no spills (archive/bf16-m32-swapab-wgmma-triton/1602a4e947e64090.log: wgmma=16, mma.sync=0, 56-96 regs). So the new kernel will very likely move the router at m<=32 from mma.sync to wgmma. At m=128/256, master's router already uses BLOCK_M=64, so it probably already emits wgmma; there the change is about tile shape and traffic, not the MMA path. The print will confirm either way. (2) The router is bandwidth/latency bound, not MMA bound. 8.4 MB of hi+lo at m<=32 is ~0.03 GFLOP. Master's kernel takes 3.2-4.1 us against a 1.8 us floor because each of 128 CTAs streams only 64 KB in 4 short K-steps, plus the fp32 atomics. Swapping mma.sync for wgmma on a [16..32 x 32] tile saves almost nothing when the MMA is not the bottleneck. The m<=16 config, with SPLIT_K 32, K-slice 256 and 2 K-steps holding all 64 KB in flight, is essentially the bytes-in-flight change that router-splitk32-small-m-all-loads-in-flight (1.143x) and its stages3 control (1.152x) already measured flat against 1.151x. Expect m=1/8/16 flat, possibly +0.2-0.4 us from a cheaper register footprint. m=32 is the best small-m candidate, if master's BLOCK_M=32 router tile has high register pressure as the bf16 BLOCK_M=32 tile did (255 regs). (3) m=128/256 traffic. The hypothesis counts 16.8 MB of 'L2/HBM' weight traffic at m=128. But all 128 CTAs are co-resident (128 <= 132 SMs) and the 8.4 MB weight fits easily in the 50 MB L2, so the second M-block's read of a tile is mostly an L2 hit, not HBM. HBM traffic is ~8.4 MB on both sides, and the saving is L2 bandwidth only. Meanwhile the proposed configs raise the atomic count: m=128 goes from 128*256*16 = 524K to 128*256*32 = 1.05M fp32 red ops, and m=256 goes from 256*256*8 = 524K to 256*256*16 = 1.05M. That is 4 MB of reductions into a 128-256 KB output, competing with the L2 bandwidth the change was meant to save. The swap also makes the per-CTA x^T tile 128 tokens wide (32 KB/stage), so x is read 4x (once per W-block) on both sides. Shared memory at m=128 is 3 stages x (16 + 16 + 32 KB) = 192 KB, plus up to 32 KB of convert_layout scratch for the transposed atomic epilogue if it does not alias the dead pipeline buffers. That is close to the 227 KB limit, so the 2-stage OutOfResources fallback is likely to be exercised. (4) Expected directions vs TASK.md: router rows are 62-weight shapes in every load at ~6-9 us of a ~50-60 us load, so a 0.3-1 us router saving is ~0.5-1.5% per load. That is inside the +/-1.5% CI, and the combo run showed byte-identical router rows drifting 0.92-0.94 on verify_1/verify_16. 'flat' for the four bf16 layers is correct by construction. WEAKEST LINK: the m=128/256 'faster' claim rests on removing a re-read that is mostly L2-resident, while the same config doubles the atomic traffic. A secondary weak link is the m<=16 claim, since the MMA instruction is not what bounds a 64 KB-per-CTA, 2-K-step stream.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: this is the kernel being replaced. Its SPLIT_K / atomic_add / torch.zeros design fixes the baseline atomic count (SPLIT_K 16 at m<=128, 8 above). The new configs double SPLIT_K at m=128/256, so any m=128/256 delta mixes orientation and weight re-read with 2x atomics. This is the one material confound.
- router-hilo-at-m8: fully superseded on master (the Triton router runs at every m and the fp32 F.linear path is only a fallback). No interaction.
- sglang-gemm-dispatch-parity: touches only the four bf16 layers on master now. Router graphs are separate per-shape captures. No interaction.
- bf16-shape-gated-triton-winners-combo: bf16 layers at m<=16 only, and separate graphs. No interaction with router rows. Its run showed ~6-8% drift on byte-identical router rows, which is a measurement caveat, not a confound.
- bf16-m32-swapab-wgmma-triton: the design template (same operand orientation and masked x^T load), but a different kernel, weights and graphs, so it has no runtime interaction. Its success makes the wgmma emission very likely, but it does not show that wgmma helps a bandwidth/latency-bound 8.4 MB router stream.

### Interference and A/B needs
No companion is warranted under the rule that a companion is needed only when both a gain and a loss would be ambiguous. m<=32 is readable from one run. The prepare() print says whether wgmma replaced mma.sync. The m<=16 grid-depth change (SPLIT_K 16->32, 2 K-steps) was already measured flat by router-splitk32-small-m-all-loads-in-flight and its stages3 control, so a small-m gain is attributable to orientation, and a flat or loss result says orientation does not matter for the router. m=128/256 is readable on a gain (net of doubled atomics, the swap-AB tile pays). A loss there would be ambiguous between the doubled atomic count and the tile change. Since a gain is not ambiguous, the rule does not justify a companion. bf16 checks are spent and every run is a scored submit, so the cheaper fix is a build-time choice, not an extra run. If the implementer wants the m=128/256 rows isolated, give the >32 buckets master's SPLIT_K (m<=128: BLOCK_T 64, grid (2, 4, 16), i.e. master's re-read with no extra atomics; m>128: BLOCK_T 128, grid (2, 4, 8) = 64 CTAs, or BLOCK_T 64, grid (4, 4, 8) = 128 CTAs). That is a design change, not a companion, and it should not go in without the author's agreement because it drops the re-read claim at m=128. Otherwise run as written and judge m=128/256 by the kernel's own top_kernels time against master's 4.9 / 6.9 us, which reproduces within ~1%.

### Open questions
- Does master's _router_hilo_kernel already emit wgmma at m=128/256 (BLOCK_M=64)? If so, the 'orientation moves the MMA path' story applies only at m<=32, and the m=128/256 rows test re-read vs atomics only. The planned side-by-side print answers this; make sure it covers all four buckets.
- What are master's router n_regs at BLOCK_M=32 (m=32 bucket)? If it is near 255 like the bf16 tokens-on-M tile, m=32 is the most likely real gain. If it is modest, expect m=32 flat as well.
- Does Triton lower tl.atomic_add on the transposed [64, BLOCK_T] accumulator as vectorized red.global.add (v2/v4.f32, after a coalescing convert_layout) or as scalar strided atomics? Add a red/atom count to the PTX print. Scalar uncoalesced atomics at 1M ops (m=128/256) would likely erase the gain.
- Does the m=128 bucket compile at 3 stages (192 KB pipeline + epilogue convert scratch) or fall back to 2? With only 2 K-steps per CTA (K-slice 256 / BLOCK_K 128), 3 stages is moot anyway. Consider num_stages 2 for that bucket by design.
- At m=1, BLOCK_T=16 wastes 15/16 of the RHS columns and loads a 16-row masked x^T tile. That is harmless for bandwidth, but confirm the masked load stays cp.async with zero-fill (count cp.async in the print) rather than turning into synchronous loads that stall the wgmma pipeline.

