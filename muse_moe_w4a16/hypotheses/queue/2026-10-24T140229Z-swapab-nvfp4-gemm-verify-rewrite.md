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
review: required
status: queued
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
