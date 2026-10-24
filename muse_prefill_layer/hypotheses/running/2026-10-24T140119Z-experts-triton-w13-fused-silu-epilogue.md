---
problem: muse_prefill_layer
slug: experts-triton-w13-fused-silu-epilogue
created: 2026-10-24T140119Z
base: master
area: silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096)
change: >
  Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T >
  4096 path (_experts_triton, first_turn only), built on master. (1) prepare(): build the Triton
  bank's w13 codes and w13 block scales with their N rows permuted (lossless, a new tensor; the
  weights stay untouched and still NVFP4) so that every 128-row W tile holds 64 gate rows followed
  by the matching 64 up rows: new row 128j + r = old row 64j + r (gate) for r < 64, and old row
  4096 + 64j + (r - 64) (up) for r >= 64, j = 0..63. Apply the same permutation to w13_sf
  (row-wise); w13_gs is per expert and unchanged. The Marlin bank is built from the original
  weights as today, so T <= 4096 is untouched. _check_expert_gemms compares the w13 variants
  against nvfp4.dequantize of the original rows taken in the permuted order (an index_select of
  ref by the same permutation) and keeps every other assertion. (2) _moe_nvfp4_gemm_kernel gets
  constexpr FUSE_SILU (False for w2 and for the check of the unfused output layout): when True,
  after acc * gs, round to bf16 and back to fp32 (so numerics match today's bf16 cache1 feeding
  silu_and_mul), split acc [BLOCK_W=128, BLOCK_T] into g = rows 0..63 and u = rows 64..127
  (tl.reshape to (2, 64, BLOCK_T), tl.permute, tl.split), compute g * sigmoid(g) * u in fp32 as
  SGLang's act_and_mul does, and store bf16 into the [num_slots, 4096] output at columns pid_w *
  64 + arange(64), rows = sorted ids, masked by token_mask as today. Blocks with expert -1 return
  without writing, as the w13 call does today (silu_and_mul also skips -1 slots, so cache2's -1
  rows were never written before either). (3) _experts_triton: allocate cache2 [num_slots, 4096]
  and pass it as the w13 call's output with FUSE_SILU=True; drop cache1 and the silu_and_mul call;
  cache3 for the w2 output is unchanged (it may now be its own [num_slots, 4096] allocation). The
  w2 call, the post-expert norm, EXPERT_CONFIGS, the shape gate and kernel_fn are unchanged. Add a
  FUSE_SILU_W13 = True module constant so a control can restore the unfused path. No BF16 weight
  copy, no autotune, no host sync.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "removes the act_and_mul_kernel (286.6-297.2 us per first_turn shape in the master profile) and the w13 GEMM's bf16 store of gate+up (16384 local rows x 8192 x 2 B = 268 MB written then re-read by silu, ~110 us of traffic) in favour of a 134 MB store of the activated half: ~300-400 us off a 14.3-22.3 ms shape, 2-2.5% on the 138-weight sliding first_turn and the first_turn load"
  - shapes: "*load=turn_*"
    direction: neutral
    why: "T <= 4096 runs Marlin with the unpermuted Marlin bank and SGLang's silu_and_mul; untouched"
risk: low
depends_on: experts-hybrid-marlin-turn-triton-first-turn
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164610Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T164610Z
base_sha: 721aada38118e1641389dda5e890b5a18175f8d8
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/experts-triton-w13-fused-silu-epilogue
worktree: .worktrees/muse_prefill_layer/experts-triton-w13-fused-silu-epilogue
mode: null
---

# experts-triton-w13-fused-silu-epilogue

## Change

Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T > 4096 path (_experts_triton, first_turn only), built on master. (1) prepare(): build the Triton bank's w13 codes and w13 block scales with their N rows permuted (lossless, a new tensor; the weights stay untouched and still NVFP4) so that every 128-row W tile holds 64 gate rows followed by the matching 64 up rows: new row 128j + r = old row 64j + r (gate) for r < 64, and old row 4096 + 64j + (r - 64) (up) for r >= 64, j = 0..63. Apply the same permutation to w13_sf (row-wise); w13_gs is per expert and unchanged. The Marlin bank is built from the original weights as today, so T <= 4096 is untouched. _check_expert_gemms compares the w13 variants against nvfp4.dequantize of the original rows taken in the permuted order (an index_select of ref by the same permutation) and keeps every other assertion. (2) _moe_nvfp4_gemm_kernel gets constexpr FUSE_SILU (False for w2 and for the check of the unfused output layout): when True, after acc * gs, round to bf16 and back to fp32 (so numerics match today's bf16 cache1 feeding silu_and_mul), split acc [BLOCK_W=128, BLOCK_T] into g = rows 0..63 and u = rows 64..127 (tl.reshape to (2, 64, BLOCK_T), tl.permute, tl.split), compute g * sigmoid(g) * u in fp32 as SGLang's act_and_mul does, and store bf16 into the [num_slots, 4096] output at columns pid_w * 64 + arange(64), rows = sorted ids, masked by token_mask as today. Blocks with expert -1 return without writing, as the w13 call does today (silu_and_mul also skips -1 slots, so cache2's -1 rows were never written before either). (3) _experts_triton: allocate cache2 [num_slots, 4096] and pass it as the w13 call's output with FUSE_SILU=True; drop cache1 and the silu_and_mul call; cache3 for the w2 output is unchanged (it may now be its own [num_slots, 4096] allocation). The w2 call, the post-expert norm, EXPERT_CONFIGS, the shape gate and kernel_fn are unchanged. Add a FUSE_SILU_W13 = True module constant so a control can restore the unfused path. No BF16 weight copy, no autotune, no host sync.

## Expected

- `*load=first_turn*` faster: removes the act_and_mul_kernel (286.6-297.2 us per first_turn shape in the master profile) and the w13 GEMM's bf16 store of gate+up (16384 local rows x 8192 x 2 B = 268 MB written then re-read by silu, ~110 us of traffic) in favour of a 134 MB store of the activated half: ~300-400 us off a 14.3-22.3 ms shape, 2-2.5% on the 138-weight sliding first_turn and the first_turn load
- `*load=turn_*` neutral: T <= 4096 runs Marlin with the unpermuted Marlin bank and SGLang's silu_and_mul; untouched

## Risk

Risk: the epilogue split doubles the live fp32 values briefly and may raise register pressure at BLOCK_T 256 (8 warps, acc 128x256); check n_spills in the prepare log. Pairing gate/up in one 128-row tile is required (two separate 128-row accumulators would double acc registers). The gain is ~2%, near the ~1.5-2.5% drift floor, so judge it on the act_and_mul row disappearing and the _moe_nvfp4_gemm_kernel row not growing. It touches disjoint lines from experts-triton-nvfp4-packed-asm-dequant (dequant inside the K loop vs epilogue/prepare), so the two should 3-way merge.

## Adversarial review

Reviewed 2026-10-24T164610Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, rules-clean fusion that removes a measured 286.6-297.2 us act_and_mul_kernel per first_turn shape plus about 270 MB of round-trip traffic, and prepare()'s existing Triton-vs-Marlin e2e check gates its correctness. The headline gain is about 0.6-0.7%, below the drift floor, so judge it on the act_and_mul row disappearing and the _moe_nvfp4_gemm_kernel row (8149-8873 us) not growing. The main risk is register and shared-memory pressure in the BLOCK_T 256 w13 epilogue, which already spills.

### Justification
Mechanism: in master's profile (fb1e52d2119d4175) act_and_mul_kernel is 297.2 / 287.3 / 286.6 / 287.4 us on the four first_turn shapes (1.3-2.0% of each), and it runs at 14-53 us on the turn shapes, which stay on Marlin and are untouched. Fusing it into the w13 epilogue drops that launch. It also drops the bf16 write and re-read of the [~16K local slots, 8192] gate|up tensor (about 268 MB each way) and halves the w13 store to 134 MB. The claim of about 300-400 us off each first_turn shape is plausible. The permutation is lossless and keeps the codes NVFP4: whole rows of codes and block scales move, and the per-expert global scale is unchanged. Numerics: rounding acc*gs to bf16 before silu reproduces today's bf16 cache1 path, so the result matches SGLang's act_and_mul to within its own fp32 silu ordering. Expected directions are consistent with TASK.md. first_turn carries 186 of the load's shape weight (138 sliding plus 3 x 16 full), about 2.86 s per load. 300 us x 186 is about 56 ms, about 1.9-2.0% on the first_turn load. The headline is a geomean over three loads with turn_1 and turn_4 flat, so it moves only about 0.6-0.7% (1.060 to about 1.067), inside the CI [1.044, 1.076] and under LESSONS' 1.5-2.5% drift floor. The hypothesis says this itself and correctly says to read the kernel rows instead. Weakest link: the epilogue cost at BLOCK_T 256. The prepare log already shows the w13 BLOCK_T=256 variant at n_regs=254 with n_spills=4 and shared=140288 B. With 8 warps and BLOCK_W=128, Triton's wgmma layout most likely puts rows 0-63 (gate) in warpgroup 0 and rows 64-127 (up) in warpgroup 1. The reshape(2,64,BLOCK_T)/permute/split would then force a convert_layout of the 128x256 fp32 accumulator (128 KB) through shared memory, plus sigmoid temporaries on top of an already-spilling register file. The w13 GEMM is about 2/3 of the 8.1-8.9 ms row, about 5.5 ms. A 5% slowdown from extra spills or the smem round-trip would cancel the whole ~300 us. This cannot cause a correctness failure, only a null or negative result.

### Confounders on the current kernel
- experts-triton-w4a16-first-turn-blockt256 (landed, stage 3): it set BLOCK_T=256, 8 warps and 2 stages, the config whose 254 regs and 4 spills the fused epilogue must fit in. A gain or loss in the _moe_nvfp4_gemm_kernel row is partly an interaction with that tile choice. At BLOCK_T 128 the same epilogue would be far cheaper but the GEMM slower. This shapes the result's magnitude but does not make it ambiguous: the act_and_mul row and the GEMM row are read separately.
- experts-hybrid-marlin-turn-triton-first-turn (landed, stage 4): its shape gate confines the change to T > 4096, so the turn shapes are a built-in null control. If the gate were absent, the fused path would also run at turn_1 and turn_4, at BLOCK_T 64 and 128 with much smaller acc. No interaction on the shapes this run touches.
- sglang-pipeline-parity (landed): it supplies the silu_and_mul semantics being replaced (gate first, then up, bf16 in and out, -1 slots skipped). No timing interaction beyond defining the reference behaviour.

### Interference and A/B needs
No companion is warranted. Both outcomes can be read from this one run's own profile. Gain: act_and_mul_kernel disappears from the first_turn shapes and the _moe_nvfp4_gemm_kernel row grows by less than about 290 us. Loss: that row grows by more than the saved time, which the prepare log's n_regs, n_spills and shared lines for the FUSE_SILU w13 variant will explain. A FUSE_SILU_W13=False control with the permuted bank would be master's kernel with a lossless row permutation and teaches nothing beyond the profile. An ab-revert of stage 3 (BLOCK_T 128 at first_turn) would ask a different tuning question. Checks are spent and this is a scored submit, so spending more runs on an expected ~0.65% headline is not justified.

### Open questions
- Correctness gating for a scored submit: none of the 5 judge tests exceeds 4096 tokens, so the fused path runs only in the timed first_turn calls. Make sure _check_experts_e2e (T=8192, rel L2 <= 2e-2 vs Marlin) actually runs the FUSE_SILU=True path. Also consider adding a direct fused-epilogue assertion in _check_expert_gemms: silu(bf16(ref_g)) * bf16(ref_u) at the permuted rows, checking the column mapping pid_w*64 + arange(64), the -1 block untouched and the past-n_padded block untouched. A layout bug then fails in prepare with a precise message rather than through the e2e tolerance.
- The output tensor is [num_slots, 4096], but the kernel uses N (8192 for w13) for both the weight row count and the C row stride. The fused store needs its own output stride (N // 2) and column base pid_w * (BLOCK_W // 2). Easy to miss.
- Could a finer gate/up interleave avoid the cross-warpgroup convert_layout? For example, 8 gate rows then 8 up rows in each 16-row slab, so each thread's wgmma accumulator pair (rows r, r+8) holds g and u. Check the TTGIR or the shared/n_spills numbers: if the 128-row/64-row split forces a 128 KB smem shuffle and more spills, that is the first thing to change.
- Merge order with the queued p1 experts-triton-nvfp4-packed-asm-dequant: both edit _moe_nvfp4_gemm_kernel, and the register budget the epilogue sees depends on which lands first. Re-read n_spills after any 3-way merge.
- silu_and_mul skips -1 slots and the fused path never writes them. cache2's -1 rows stay uninitialized, as today, and w2's -1 blocks never read A. Confirm the new separate cache2 allocation is not read anywhere else, e.g. by the post-expert norm, which reads y with slot_expert_ids.

