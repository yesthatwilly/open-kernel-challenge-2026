---
problem: muse_prefill_layer
slug: experts-dequant-triton-moe-16k
created: 2026-10-23T230250Z
base: master
area: w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens)
change: >
  Shape-gated replacement of the two Marlin W4A16 GEMMs for tokens >= DEQUANT_MIN_TOKENS = 4096 (a
  module constant), i.e. only the four first-turn shapes (T=16384) and the q_len 4096 test; below
  the gate _experts stays exactly the parity Marlin path. prepare() keeps the raw NVFP4 tensors
  from weights (w13 uint8 [32, 8192, 2048], w13_sf e4m3 [32, 8192, 256], w13_gs fp32 [32], w2 [32,
  4096, 2048], w2_sf, w2_gs) contiguous beside the Marlin bank; no BF16 copy is stored in state.
  On the gated path, inside the timed call every call: (a) torch.empty two BF16 buffers w13_bf16
  [32, 8192, 4096] and w2_bf16 [32, 4096, 4096] and fill them with one Triton dequant kernel (grid
  over expert x row-block): per packed byte the low nibble is element 2i and the high nibble 2i+1
  (nvfp4.unpack_e2m1), magnitude from the E2M1 grid (0, .5, 1, 1.5, 2, 3, 4, 6) with bit 3 the
  sign, times sf[e, n, k//16] (e4m3 read as fp32) times gs[e], rounded once to bf16, the same
  arithmetic as nvfp4.dequantize; assert equality to nvfp4.dequantize on one expert row-slice in
  prepare() (untimed). (b) moe_align_block_size(local_ids, 128, N_EXPERTS) for BLOCK_SIZE_M=128
  (the parity path keeps its 64). (c) two calls of
  sglang.srt.layers.moe.fused_moe_triton.fused_moe.invoke_fused_moe_kernel exactly as
  fused_experts_impl uses them for plain BF16 weights: A=x [T, 4096] bf16, B=w13_bf16, C=cache1
  [T*8, 8192], mul_routed_weight=False, top_k=8, config {BLOCK_SIZE_M: 128, BLOCK_SIZE_N: 128,
  BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 8, num_stages: 4}, compute_type=tl.bfloat16, every
  quant flag False, no scales; then the parity silu_and_mul(cache1, cache2, expert_ids=local_ids,
  expert_step=1); then the second call with A=cache2, B=w2_bf16, C=cache3 [T*8, 4096], top_k=1.
  The kernel's off_experts == -1 guard skips the non-local blocks that Marlin skips via is_ep. (d)
  the parity fused_post_expert_norm_weighted_sum on cache3.view(T, 8, 4096). If
  invoke_fused_moe_kernel's keyword names differ at SGLang a55ab7d, adapt the names only, never
  the math.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: at 4096 rows per local expert Marlin re-streams each expert 4-bit weight set once per 64-row m-block (64 passes, dequant in the MMA loop); the BF16 path pays one 0.77 GB read + 3.2 GB write + 3.2 GB read (~1.5 ms) and then a tensor-core GEMM at 128-row tiles with L2-swizzled reuse on 1.65 TFLOP of work; expected several ms off 17-26 ms, most visible on ctx=0 and sliding where attention is small
  - shapes: "*load=turn_1*"
    direction: flat
    why: "below the gate: Marlin path unchanged (control)"
  - shapes: "*load=turn_4*"
    direction: flat
    why: "below the gate: Marlin path unchanged (control)"
risk: high
depends_on: sglang-pipeline-parity
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T044949Z
review_companions: null
hold_reason: "needs human: rules check. The change materializes a full 3.2 GB BF16 copy of the NVFP4 expert weights in global memory on every call, then runs BF16 grouped GEMMs on it. TASK.md allows 'dequantizing inside your kernels' but forbids storing weights at a different precision; a per-call BF16 weight tensor is the strongest reading of that clause (reviewer's words) and may not survive the organizers' manual review. Proceed only if the user confirms it is within the rules; the adversarial review otherwise says proceed (~1.08x geomean upside)."
outcome: expired
verdict_detail: "retired: ruled out by the user's rules ruling (no per-call BF16 dequant of NVFP4 weights)"
expired_at: 2026-10-25T211931Z
---

# experts-dequant-triton-moe-16k

## Change

Shape-gated replacement of the two Marlin W4A16 GEMMs for tokens >= DEQUANT_MIN_TOKENS = 4096 (a module constant), i.e. only the four first-turn shapes (T=16384) and the q_len 4096 test; below the gate _experts stays exactly the parity Marlin path. prepare() keeps the raw NVFP4 tensors from weights (w13 uint8 [32, 8192, 2048], w13_sf e4m3 [32, 8192, 256], w13_gs fp32 [32], w2 [32, 4096, 2048], w2_sf, w2_gs) contiguous beside the Marlin bank; no BF16 copy is stored in state. On the gated path, inside the timed call every call: (a) torch.empty two BF16 buffers w13_bf16 [32, 8192, 4096] and w2_bf16 [32, 4096, 4096] and fill them with one Triton dequant kernel (grid over expert x row-block): per packed byte the low nibble is element 2i and the high nibble 2i+1 (nvfp4.unpack_e2m1), magnitude from the E2M1 grid (0, .5, 1, 1.5, 2, 3, 4, 6) with bit 3 the sign, times sf[e, n, k//16] (e4m3 read as fp32) times gs[e], rounded once to bf16, the same arithmetic as nvfp4.dequantize; assert equality to nvfp4.dequantize on one expert row-slice in prepare() (untimed). (b) moe_align_block_size(local_ids, 128, N_EXPERTS) for BLOCK_SIZE_M=128 (the parity path keeps its 64). (c) two calls of sglang.srt.layers.moe.fused_moe_triton.fused_moe.invoke_fused_moe_kernel exactly as fused_experts_impl uses them for plain BF16 weights: A=x [T, 4096] bf16, B=w13_bf16, C=cache1 [T*8, 8192], mul_routed_weight=False, top_k=8, config {BLOCK_SIZE_M: 128, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 8, num_warps: 8, num_stages: 4}, compute_type=tl.bfloat16, every quant flag False, no scales; then the parity silu_and_mul(cache1, cache2, expert_ids=local_ids, expert_step=1); then the second call with A=cache2, B=w2_bf16, C=cache3 [T*8, 4096], top_k=1. The kernel's off_experts == -1 guard skips the non-local blocks that Marlin skips via is_ep. (d) the parity fused_post_expert_norm_weighted_sum on cache3.view(T, 8, 4096). If invoke_fused_moe_kernel's keyword names differ at SGLang a55ab7d, adapt the names only, never the math.

## Expected

- `*load=first_turn*` faster: at 4096 rows per local expert Marlin re-streams each expert 4-bit weight set once per 64-row m-block (64 passes, dequant in the MMA loop); the BF16 path pays one 0.77 GB read + 3.2 GB write + 3.2 GB read (~1.5 ms) and then a tensor-core GEMM at 128-row tiles with L2-swizzled reuse on 1.65 TFLOP of work; expected several ms off 17-26 ms, most visible on ctx=0 and sliding where attention is small
- `*load=turn_1*` flat: below the gate: Marlin path unchanged (control)
- `*load=turn_4*` flat: below the gate: Marlin path unchanged (control)

## Risk

High risk. Correctness: a nibble/scale layout error shows up as a failed test at q_len 4096 (that test exercises the gate), so it fails before timing; the assert in prepare() against nvfp4.dequantize catches it earlier. API: invoke_fused_moe_kernel's signature and its -1 expert guard at a55ab7d are assumed from SGLang's fused_experts_impl; a signature mismatch costs one failed check run. Memory: 3.2 GB + 1 GB BF16 scratch plus the 2 GB cache13 per call, graph-pooled, fine on 141 GB. Rules: state holds only NVFP4 (Marlin repack and raw); the BF16 copy is produced inside the timed call from NVFP4 and never persists, which TASK.md names as allowed (dequantizing inside your kernels is fine); activations stay BF16 (A operands and outputs bf16, fp32 accumulate) as in Marlin; no host sync, allocation sizes depend only on config. The result attributes to the expert GEMM engine alone; the dequant cost is a known fixed ~1.5 ms that a dependent hypothesis can hide.

## Adversarial review

Reviewed 2026-10-24T044949Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** The mechanism (Marlin W4A16 is a small-M engine running at ~137 TFLOP/s on 1.65 TFLOP of expert work, 11-12 ms of a 17-25 ms shape; a BF16 tensor-core grouped GEMM should do it in half) is plausible and large enough that a single run is interpretable in either direction, with the per-kernel profile separating the dequant kernel, moe_align and the two fused_moe calls from the Marlin numbers already in the parity profile. The hypothesis's own arithmetic is wrong by 8x (it is ~512 rows per local expert and 8 Marlin m-blocks, not 4096 and 64) but the direction survives the correction.

### Justification
Profile support (parity job 62025c368c3c4e8e): Marlin takes 11995/11106/10240/10912 us on the four first-turn shapes, 44-64% of 18.3/17.0/21.5/25.3 ms, and the SoL floors are 5.5-9.8 ms. The expert work at T=16384 is 16384*8/256*32 = 16384 local (token,expert) rows x 2 x (4096*8192 + 4096*4096) = 1.65 TFLOP (this matches the hypothesis's own 1.65 TFLOP and the ctx=0 floor of 5518 us), so Marlin runs at ~137 TFLOP/s, 14% of peak. That is the real headroom. The hypothesis's stated cause is wrong: with 256 experts and random routing each local expert sees ~512 rows, not 4096, so Marlin makes ~8 passes per expert with block_size_m=64, not 64, and each expert's packed w13 (16.8 MB) fits in H200's 50 MB L2, so the re-streaming is mostly L2 traffic, not HBM. Marlin's cost is its small-M design (64-row tiles, dequant in the MMA loop, few MMA warps), which is a sound reason to expect a BF16 tensor-core engine to win at M=512/expert, just not the reason given. Budget with the corrected numbers: dequant 0.9 GB read + 3.2 GB write ~1.0-1.5 ms if the Triton dequant kernel is bandwidth-efficient; two fused_moe calls on 1.65 TFLOP at 300 TFLOP/s ~5.5 ms (weights re-read 4x per expert at BLOCK_M=128 = ~13 GB, 2.7 ms, under the compute time); total ~7 ms vs 12 ms, i.e. ~5 ms off each first-turn shape, first-turn load ~1.25x, geomean ~1.08x. Break-even needs the Triton kernels to fall below ~160 TFLOP/s, which is possible for BLOCK_N=128/BLOCK_K=64 (SGLang's large-M BF16 configs usually use BLOCK_N=256) and with 128-row blocks on ~512-row experts the average padding waste is ~12% and up to 25%. Expected table is consistent with TASK.md: the 138-weight ctx=32768 sliding shape has the largest Marlin share (64%), so first_turn (one of three equal loads) carries the gain; turn_1/turn_4 stay on Marlin and are true controls. Weakest link: the gain rests entirely on an unmeasured assumption that SGLang's generic Triton fused_moe_kernel reaches >~200 TFLOP/s at these shapes with the hand-picked config; the sister problem (moe_w4a16) found the Triton MoE kernel 2-2.6x slower than Marlin, though at small M with in-kernel dequant, so it is a caution, not a refutation. Second: the GEMM-level numbers in the change paragraph and in the two queued dependents (threshold-turn4 says '~753 rows per local expert' for T=3012; it is ~94) propagate the same 8x error.

### Confounders on the current kernel
- sglang-pipeline-parity (the base): it supplies the -1 local_ids via the Triton remap, moe_align_block_size with N_EXPERTS=256 and -1 ids, silu_and_mul(expert_ids=local_ids) and the slot-masked weighted sum that the new path reuses; the Triton fused_moe kernel relies on moe_align emitting expert_ids == -1 for the non-local blocks exactly as Marlin's is_ep path does, which is SGLang's own EP contract, so this is a dependency, not a confounder. Its own effect is below noise (all shapes 0.987-1.009x), and the only shape it moved, ctx=32768 full at 0.9865x, has err_ns 186 us from FA3 noise, far below the multi-ms effect claimed here. Nothing else has landed; nothing to revert or retune.

### Interference and A/B needs
No companion is warranted. The change has three sub-parts (per-call dequant, align at BLOCK_M=128, two Triton BF16 fused_moe calls), but the dequant can only add time and the GEMM swap is the only possible source of a gain, so the direction of any net result is unambiguous. The magnitude is attributable from the run's own profile: the dequant kernel, moe_align_block_size (block 128) and the two fused_moe_kernel calls appear as separate rows, and the parity profile already gives Marlin's 10.2-12.0 ms on the same shapes, so a loss splits cleanly into 'Triton GEMM slower than Marlin outright' versus 'GEMM faster but dequant round trip ate it' without a second run. turn_1 and turn_4 stay on the untouched Marlin path and act as built-in flat controls. The queued dependents already cover the tuning branches a companion would otherwise supply: experts-dequant-overlap-attention (hide the dequant), experts-dequant-grouped-mm-cutlass (engine swap), experts-dequant-threshold-turn4 (gate); an ab-tune on the Triton tile config (BLOCK_N=256, BLOCK_M=64) is the one follow-up not yet queued, but it is only worth a run after this result shows the fused_moe rows, so it is not a companion.

### Open questions
- invoke_fused_moe_kernel at SGLang a55ab7d expects C as [M, top_k, N] (it reads C.stride(1) and C.stride(2)) and B as [E, N, K]; pass cache1 as cache1.view(T, 8, 8192) and cache3 as cache3.view(T*8, 1, 4096), w13_bf16 [32, 8192, 4096] and w2_bf16 [32, 4096, 4096] as built. The hypothesis text says 'C=cache1 [T*8, 8192]', which will fail on stride(2) if passed 2-D; a view is a name/shape adaptation, not a math change.
- Does SGLang's moe_align_block_size place the ~114688 non-local (-1) slots into blocks that the Triton kernel then zero-fills (write_zeros_to_output), or drop them? If placed, GEMM1 zero-writes ~896 blocks x 8192 bf16 (~1.9 GB, ~0.4 ms) and GEMM2 ~1 GB, a fixed cost missing from the budget; the parity Marlin path skips those blocks without writing. Check in the profile whether fused_moe_kernel time is consistent with 16384 live rows.
- Rows per local expert at T=16384 is ~512 (uniform random routing over 256 experts), not 4096: correct the change paragraph's '64 passes' to '8 m-blocks' and expect ~4 m-blocks per expert at BLOCK_M=128 with ~12% average padding; the same 8x error is in experts-dequant-threshold-turn4 ('753 rows per expert' is ~94).
- If the fused_moe rows come in above ~8 ms (below ~200 TFLOP/s), the tile config is the first suspect: queue an ab-tune with BLOCK_N=256, BLOCK_M=64, num_stages=3 before concluding the engine loses; if they come in near 5 ms and the dequant kernel is above 2 ms, the dequant kernel is not bandwidth-efficient (byte loads, LUT gather) and experts-dequant-overlap-attention only hides it rather than fixes it.
- Marlin applies the e4m3 block scale in bf16 and the global scale at the end; the single fp32 product rounded once to bf16 is at least as accurate, so the 2.5x-reference bound at q_len 4096 (the one test on the gated path, soft dist only) should pass; note the gated path is never exercised with dist=outliers.

