---
problem: muse_moe_w4a16
slug: swapab-nvfp4-gemm-verify-rewrite
created: 2026-10-24T140229Z
base: master
area: w13 marlin + w2 marlin
change: >
  Verify shapes only: num_tokens >= 128 (bs=128, bs=256, the bs=257 and skewed bs=128 tests).
  num_tokens < 128 keeps master's path byte for byte. Port muse_prefill_layer master's Triton
  NVFP4 swap-AB grouped GEMM (master commit b92a0f4, experts-hybrid-marlin-turn-triton-first-turn)
  into muse_moe_w4a16/submission.py verbatim: _e2m1_bits_to_bf16, _moe_nvfp4_gemm_kernel,
  _nvfp4_moe_gemm, _reorder_nibbles, _expert_bank, the constants EXPERT_BLOCK_K=128, SF_BLOCK,
  SCALE_BIAS, EPILOGUE_BIAS and ZERO_NONLOCAL_W2=True, and its prepare-time tile check
  _check_expert_gemms. In prepare(), also build the Triton bank from the weights sliced to the
  bank's [start, start+count) = [0, 32): nibble-reordered w13/w2 codes (a lossless permutation,
  NVFP4 stays NVFP4) and w13_sf/w2_sf/w13_gs/w2_gs contiguous as stored. This adds ~0.9 GiB. The
  Marlin bank stays for num_tokens < 128. Compile both configs and run the tile check, untimed. In
  kernel_fn at num_tokens >= 128: drop master's two-smaller block_size_m pick. Keep local_ids as
  master builds them (expert_map gather). Then sorted_ids, block_experts, n_padded =
  moe_align_block_size(local_ids, cfg BLOCK_T, bank count 32). Run _nvfp4_moe_gemm(x, w13 bank,
  top_k=8, zero_nonlocal=False) into cache1, then silu_and_mul(cache1, cache2,
  expert_ids=local_ids.view(-1), expert_step=1) unchanged. Then _nvfp4_moe_gemm(cache2, w2 bank,
  top_k=1, zero_nonlocal=True) into cache3, viewed [M, 8, H], and SGLang's
  fused_post_expert_norm_weighted_sum unchanged. Configs are chosen by shape and are prefill's
  measured ones: num_tokens <= 128 gets BLOCK_T=64, BLOCK_W=128, BLOCK_K=128, GROUP_SIZE_T=8,
  num_warps=4, num_stages=3; above that, BLOCK_T=128, BLOCK_W=128, BLOCK_K=128, GROUP_SIZE_T=8,
  num_warps=8, num_stages=3. Print n_regs/n_spills per config in prepare, as prefill does. The
  mechanism: under rank=busiest with Student-t popularity, a few local experts carry most of the
  routes. Marlin cuts each of them into 16/32-row m-blocks (block 16 at bs=128, 32 at bs=256) and
  re-dequantizes and re-streams that expert's full ~28 MB weight once per m-block. One 64/128-row
  swap-AB tile dequantizes each weight tile once per expert in registers as the wgmma LHS, and
  only the bf16 activation slab goes through smem.
expected:
  - shapes: bs=128 *
    direction: faster
    why: incumbent Marlin w13+w2 316.1 us vs a 101.3 us floor (3.1x). Hot experts stop paying one full weight dequant+stream per 16-row m-block
  - shapes: bs=256 *
    direction: faster
    why: incumbent Marlin w13+w2 490.2 us vs a 108.9 us floor (4.5x), with 32-row m-blocks; BLOCK_T=128 covers a hot expert in 1-2 tiles
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens < 128 is master code"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at num_tokens >= 128
review: proceed
status: hold
review_at: 2026-10-24T140647Z
review_companions: null
hold_reason: "reviewer: stage 1 likely 1.5-2x slower than Marlin on the replaced kernels (Marlin runs only 15-18% more blocks; prefill's Triton tile costs ~25 us vs Marlin's 11.6-16.7 us/block). Unhold after prefill's packed-asm dequant stage (experts-triton-nvfp4-packed-asm-dequant) harvests and fold that dequant in, with BLOCK_T=64 at bs=256"
---

# swapab-nvfp4-gemm-verify-rewrite

## Change

Verify shapes only: num_tokens >= 128 (bs=128, bs=256, the bs=257 and skewed bs=128 tests). num_tokens < 128 keeps master's path byte for byte. Port muse_prefill_layer master's Triton NVFP4 swap-AB grouped GEMM (master commit b92a0f4, experts-hybrid-marlin-turn-triton-first-turn) into muse_moe_w4a16/submission.py verbatim: _e2m1_bits_to_bf16, _moe_nvfp4_gemm_kernel, _nvfp4_moe_gemm, _reorder_nibbles, _expert_bank, the constants EXPERT_BLOCK_K=128, SF_BLOCK, SCALE_BIAS, EPILOGUE_BIAS and ZERO_NONLOCAL_W2=True, and its prepare-time tile check _check_expert_gemms. In prepare(), also build the Triton bank from the weights sliced to the bank's [start, start+count) = [0, 32): nibble-reordered w13/w2 codes (a lossless permutation, NVFP4 stays NVFP4) and w13_sf/w2_sf/w13_gs/w2_gs contiguous as stored. This adds ~0.9 GiB. The Marlin bank stays for num_tokens < 128. Compile both configs and run the tile check, untimed. In kernel_fn at num_tokens >= 128: drop master's two-smaller block_size_m pick. Keep local_ids as master builds them (expert_map gather). Then sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, cfg BLOCK_T, bank count 32). Run _nvfp4_moe_gemm(x, w13 bank, top_k=8, zero_nonlocal=False) into cache1, then silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) unchanged. Then _nvfp4_moe_gemm(cache2, w2 bank, top_k=1, zero_nonlocal=True) into cache3, viewed [M, 8, H], and SGLang's fused_post_expert_norm_weighted_sum unchanged. Configs are chosen by shape and are prefill's measured ones: num_tokens <= 128 gets BLOCK_T=64, BLOCK_W=128, BLOCK_K=128, GROUP_SIZE_T=8, num_warps=4, num_stages=3; above that, BLOCK_T=128, BLOCK_W=128, BLOCK_K=128, GROUP_SIZE_T=8, num_warps=8, num_stages=3. Print n_regs/n_spills per config in prepare, as prefill does. The mechanism: under rank=busiest with Student-t popularity, a few local experts carry most of the routes. Marlin cuts each of them into 16/32-row m-blocks (block 16 at bs=128, 32 at bs=256) and re-dequantizes and re-streams that expert's full ~28 MB weight once per m-block. One 64/128-row swap-AB tile dequantizes each weight tile once per expert in registers as the wgmma LHS, and only the bf16 activation slab goes through smem.

## Expected

- `bs=128 *` faster: incumbent Marlin w13+w2 316.1 us vs a 101.3 us floor (3.1x). Hot experts stop paying one full weight dequant+stream per 16-row m-block
- `bs=256 *` faster: incumbent Marlin w13+w2 490.2 us vs a 108.9 us floor (4.5x), with 32-row m-blocks; BLOCK_T=128 covers a hot expert in 1-2 tiles
- `bs=32 *` flat: control: num_tokens < 128 is master code
- `bs=1 *` flat: control: untouched

## Risk

Milestones (judged on the two _moe_nvfp4_gemm_kernel rows vs Marlin's rows at bs=128/256): (1) correct on all 5 tests; (2) parity: w13+w2 <= 316/490 us; (3) target <= 250/340 us, which is ~+20-30% on those two loads, ~+7-9% headline. Main risk, from the same weights in prefill (H=I=4096, 32 local experts): at turn_1 (~24 rows/expert, one tile per expert, Marlin block 64) this kernel took 796-823 us vs Marlin's 652-754 us, ~25 us per expert-tile against Marlin's ~21-23 us. So it wins only if the hot-expert multi-block redundancy is large. If nearly all 32 experts are touched with one tile each, expect a ~10-20% loss on the replaced kernels. A loss still sets the per-tile cost to beat. Stage 2 options: the silu epilogue (minted as swapab-w13-silu-epilogue-verify), a cheaper packed dequant (8 codes per int32 via lop3/prmt inline asm, as Marlin does), and a BLOCK_T tune per shape. Overlaps triton-map-align-verify-chunked (same shapes, align path): this one keeps SGLang's moe_align_block_size at BLOCK_T.

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T140647Z by an adversarial subagent (model opus).

**Verdict: proceed.** No rules problem, and the port is mechanically sound: it is the same kernel and tile check as prefill master, with both configs compiled before capture. But the mechanism is quantitatively weak. On this routing generator Marlin runs only about 15-18% more m-blocks than the swap-AB kernel runs tiles, and prefill already measured this kernel's per-tile cost at about 2x Marlin's per-m-block cost here. Expect stage 1 to be about 1.5-2x slower on the replaced kernels, not faster. Run it as a rewrite stage, with BLOCK_T=64 at bs=256 too, and treat milestone 2 as unlikely until the dequant gets cheaper.

### Justification
Shapes. I re-simulated reference.generate_input (numpy, normal dist, rank=busiest, 200 seeds). bs=128 gives 22.8 distinct local experts and about 192 local routes (max expert about 56 rows, the rest mostly 3-15 rows). bs=256 gives 24.7 distinct experts and about 388 routes (max about 116). This agrees with the TASK floors: 101.3/108.9 us means 0.75 x distinct x 27.4 MB, so about 23.6/25.3 distinct. Tile counts: Marlin block 16 at bs=128 runs 27.2 m-blocks vs 23.1 swap-AB tiles at BLOCK_T=64. Marlin block 32 at bs=256 runs 29.4 m-blocks vs 25.1 tiles at BLOCK_T=128. So the redundancy this hypothesis removes is about 15-18% of tile count, not the multi-x the 'few hot experts carry most routes' story implies. Student-t popularity at scale 0.03 is mild, and most touched experts get a single small block either way. Per-unit cost: master's profile gives Marlin about 316/27.2 = 11.6 us per 16-row m-block at bs=128 and 490/29.4 = 16.7 us per 32-row m-block at bs=256. Prefill measured this exact kernel and config (BLOCK_T=64, 4 warps, 3 stages, same H=I=4096 NVFP4 weights) at about 25 us per expert-tile at turn_1, with a full grid (32 experts x 64 W-tiles, similar to the ~1500 CTAs here). The BLOCK_T=128 / 8-warp config cost more per tile at turn_4 (about 2.1 ms total). Projection: bs=128 is about 23 x 25 = 575 us vs 316 (about 1.8x slower). bs=256 at BLOCK_T=128 is about 25 tiles at more than 25 us each, so 650-1000 us vs 490. Each tile's dequant work is fixed by BLOCK_W x K and does not shrink with real rows: masked activation rows save HBM bytes, not dequant ALU or MMA. Parity at bs=128 needs per-tile cost of 13.7 us or less, a 1.8x improvement. Milestone 3 (1.5x floor, about 150 us) needs about 6.5 us per tile, close to the 5.7 us HBM time for one expert. That is reachable in principle only with Marlin-class packed dequant (lop3/prmt, 8 codes per int32), which the Risk section lists as a stage-2 option. Weakest link: the claim that Marlin's per-m-block re-dequant/re-stream is the dominant waste. The routing data shows few extra m-blocks, and the per-tile dequant cost (where Triton is about 2x behind Marlin) dominates instead. Second weak point: the bs>128 config is 'prefill's measured one' only nominally. BLOCK_T=128 was tuned for about 94 rows per expert, while here a typical expert has about 15 rows, so most of each 128-column MMA is padding. BLOCK_T=64 at bs=256 gives 26.2 tiles vs 25.1 (+4%) with half the MMA and the cheaper 4-warp config, so it is the better stage-1 pick. The expected 'faster' directions for bs=128 and bs=256 should read 'slower (stage 1), per-tile cost is the readout'. The flat controls at bs<128 are correct: those shapes are byte-identical to master.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: the incumbent this is judged against is the tuned Marlin (block 16/32 at bs=128/256, about 9-11% faster than SGLang's 48/64 pick). That raises the bar but does not confound attribution, because the milestone compares the _moe_nvfp4_gemm_kernel rows against master's 316.1/490.2 us Marlin rows directly. Note that against SGLang's untuned 48/64 Marlin the redundancy story would look stronger (bs=128 at block 48 is about 23-24 m-blocks of a more expensive class), so do not reuse SGLang's verify numbers as the incumbent.
- sglang-marlin-path-parity: it supplies moe_align_block_size(local_ids, bsm, 32). With count 32 instead of prefill's 256, the static capacity, and so the grid of early-exit CTAs, is about 8x smaller at BLOCK_T=64/128. That helps Triton slightly vs prefill's setup, so the prefill per-tile cost is if anything a slight overestimate. No attribution problem.
- post-expert-norm-triton-small-m and triton-map-align-on-block-m-restack: both gate on num_tokens <= 32 or < 32, so they are not on the bs>=128 path. No interaction.

### Interference and A/B needs
No companions. This is kind: rewrite, judged on the replaced kernels' own profile rows at bs=128/256. Those rows are directly readable (two _moe_nvfp4_gemm_kernel rows vs master's Marlin rows), and the bs<128 shapes are byte-identical controls. No landed hypothesis shares the bs>=128 GEMM path except the block-m pick, which this change removes. An ab-revert of that pick would only change the incumbent, not this kernel's time. Per-tile cost is the stage readout, and the next stage (packed dequant) is the follow-up. A BLOCK_T control is better folded into stage 1 as a spec edit (BLOCK_T=64 at all num_tokens >= 128) than spent as a separate check while moe has 4 checks left.

### Open questions
- Should stage 1 use BLOCK_T=64 at bs=256/257 too? The simulation gives 26.2 tiles vs 25.1 at BLOCK_T=128, with half the padding MMA and the cheaper 4-warp config. I recommend editing the spec before claim. bs=257 then has a capacity of 2056+32*63 slots, which is fine.
- Is a check worth spending when the per-tile cost is already known from prefill (same kernel, config and weights)? The new information is only the low-rows-per-expert regime, where the per-tile cost should be the same because the dequant is per weight tile. If checks are scarce, consider minting stage 1 with the packed lop3/prmt-style dequant already included. That is the only lever that addresses the dominant cost.
- Does SGLang's moe_align_block_size(..., 32) give the -1 routes blocks of their own (block_experts == -1)? If not, ZERO_NONLOCAL_W2 never fires and the remote cache3 rows stay stale. That is fine only because fused_post_expert_norm_weighted_sum skips slot_expert_ids == -1, as it does on master.

