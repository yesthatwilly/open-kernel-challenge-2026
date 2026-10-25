---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-producer-scale-expand
created: 2026-10-25T125512Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Consumer instruction count only, built on
  cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate's commit (branch
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, job
  7be125e9e1ea42e3), rebased onto master. MECHANISM: the b128-n32-deep-ring run (job
  6759e36bbd4648fc) showed that weight bytes in flight do not limit wg_kernel: w13 ran at
  1.83-2.13 TB/s at 37, 49 and 64.5 KB per SM, ~1150 clk per 9216-byte stage per SM. The consumer
  warpgroups are the suspect. In s5's dq_stage, every k16 step of every consumer thread spends ~7
  of its ~24 ALU instructions turning two E4M3 block scales into bf16x2 broadcasts:
  cvt.rn.f16x2.e4m3x2 (amortised), __half22float2, two fmul by 2^119, two __float2bfloat162_rn.
  Only ~16 go to the dequant8 of the 8 codes. Meanwhile the producer warpgroup has 128 threads, of
  which one issues copies. CHANGE: move the scale conversion to the idle producer warps 1-3 (96
  threads, still at setmaxnreg.dec 40). (1) Add a scale buffer per ring stage in dynamic smem,
  SCL_STAGE = 2 records x 1024 B = 2048 B, 4 stages = 8192 B. Budget: w2 8192 static + 203776 +
  8192 = 220160 B, w13 203776 B, both <= 227 KB. Add STAGES more mbarriers sfull[s], each
  initialised with count 3 (lane 0 of each helper warp arrives after __syncwarp). (2) Producer
  warps 1-3 walk the same item/k-stage sequence as tid 0, so they share its stage counter n and
  parity. For each fill they: mbar_wait(full[s], parity); read the stage's two 512-B scale blocks
  (st + r*REC + REC_CODES); write each E4M3 byte b as the bf16 of e4m3(b) * 2^119 into sc[s] +
  r*1024 + cw*128 + g*16 + 2*(2*step + rh). Use the same arithmetic as s5 (cvt.rn.f16x2.e4m3x2,
  __half2float, * 2^119, __float2bfloat16_rn), so every value is bit-identical to what s5's
  consumers build: e4m3 has 3 mantissa bits, and x 2^119 stays inside bf16 range for normals and
  subnormals. Then __syncwarp and arrive(sfull[s]). (3) In consume_item, wait full[s] then
  sfull[s], both by the same parity. In dq_stage, replace the per-record scale load +
  e4m3x4_to_f16x2x2 + per-k convert with ONE uint4 load per record from sc (8 bf16 = rows g and
  g+8 for its 4 k16 steps). Per k, build sg = prmt(w_k, w_k, 0x1010) and sg8 = prmt(w_k, w_k,
  0x3232): 2 instructions instead of ~7. dequant8 and everything after it are unchanged. Reuse is
  safe: the helpers write sc[s] for fill f only after full[s] phase f, which follows the
  consumers' empty[s] arrive for fill f-1, and the consumers read sc[s] in dq_stage before that
  arrive. The dead/timeout path must still arrive on sfull. Nothing else moves: weight bank bytes
  (NVFP4 stays NVFP4, scales stay E4M3 in global memory; the expansion lives only in shared memory
  per stage), ring depth 4, buckets, gather, the variant-A wgmma_wait<0> schedule, epilogues,
  per-call tensor maps, s5's selector and the kernel_fn gate. BUILD: keep s5's wg_kernel as
  variant `s5` and add the new one as template flag SX. Bump the load_inline module name;
  TORCH_CUDA_ARCH_LIST=9.0a; -std=c++20 -O3 -lineinfo; print regs/spill/smem per instantiation;
  producer warps must stay within 40 registers with 0 B spill. PREPARE: s5's bucket canary
  (N16-N128) and three-state copy check (A, deepcopy B, tensor clone C; refilled graph replays;
  launch marker) on SX at M=128, 255, 256, 257 x four routings, against fp32 dequant with a bound
  of <=1.5x Marlin's own error. Outputs must be bitwise identical to s5, so print max |SX - s5|
  and require 0. SELECTOR: s5's graph-replay selector with variant candidates {s5, SX} vs Marlin
  per bucket (margin 1.02x on the judge-like zipf a=0.3 routing). Print the deep-ring per-kernel
  diagnostic line (touched experts, max routes per expert, w13/w2 us and TB/s) for both variants
  at T=128 on zipf a=0.3 and on the 17-touched judge-like routing, and at T=256. Assert the chosen
  variant is the printed winner. Hard rules: CUDA-graph capturable; kernel_fn does no host sync,
  allocation or branching on input values, no prints, no try/except, no per-call mutable state;
  nothing cached from inputs (the scale expansion is per launch, in smem); activations BF16;
  attend every route; no child processes. No module or static global may hold device pointers, TMA
  maps or buffers, because the judge clones TWO state copies (pointer-equality gates fail).
expected:
  - shapes: bs=256 *
    direction: faster
    why: "B256 is always wgmma. s5's judged rows were w13+silu 239-300 us and w2 119-148 us. If consumer ALU issue is the limiter, cutting ~20-25% of dequant instructions per k16 step should take ~10-15% off each kernel row: target w13 <=215, w2 <=105, shape row <=370 us (from 405)"
  - shapes: bs=128 *
    direction: faster
    why: B128 (Marlin pair 306 us judged) flips only if the zipf a=0.3 selector ratio reaches >=1.02x (now 0.80x), so most likely it stays MARLIN on this change alone. The printed per-kernel TB/s at T=128 is the attribution either way
  - shapes: bs=32 *
    direction: flat
    why: "control: T<128 runs master's Marlin path"
  - shapes: bs=1 *
    direction: flat
    why: control
risk: low
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-producer-scale-expand

## Change

Consumer instruction count only, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate's commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, job 7be125e9e1ea42e3), rebased onto master. MECHANISM: the b128-n32-deep-ring run (job 6759e36bbd4648fc) showed that weight bytes in flight do not limit wg_kernel: w13 ran at 1.83-2.13 TB/s at 37, 49 and 64.5 KB per SM, ~1150 clk per 9216-byte stage per SM. The consumer warpgroups are the suspect. In s5's dq_stage, every k16 step of every consumer thread spends ~7 of its ~24 ALU instructions turning two E4M3 block scales into bf16x2 broadcasts: cvt.rn.f16x2.e4m3x2 (amortised), __half22float2, two fmul by 2^119, two __float2bfloat162_rn. Only ~16 go to the dequant8 of the 8 codes. Meanwhile the producer warpgroup has 128 threads, of which one issues copies. CHANGE: move the scale conversion to the idle producer warps 1-3 (96 threads, still at setmaxnreg.dec 40). (1) Add a scale buffer per ring stage in dynamic smem, SCL_STAGE = 2 records x 1024 B = 2048 B, 4 stages = 8192 B. Budget: w2 8192 static + 203776 + 8192 = 220160 B, w13 203776 B, both <= 227 KB. Add STAGES more mbarriers sfull[s], each initialised with count 3 (lane 0 of each helper warp arrives after __syncwarp). (2) Producer warps 1-3 walk the same item/k-stage sequence as tid 0, so they share its stage counter n and parity. For each fill they: mbar_wait(full[s], parity); read the stage's two 512-B scale blocks (st + r*REC + REC_CODES); write each E4M3 byte b as the bf16 of e4m3(b) * 2^119 into sc[s] + r*1024 + cw*128 + g*16 + 2*(2*step + rh). Use the same arithmetic as s5 (cvt.rn.f16x2.e4m3x2, __half2float, * 2^119, __float2bfloat16_rn), so every value is bit-identical to what s5's consumers build: e4m3 has 3 mantissa bits, and x 2^119 stays inside bf16 range for normals and subnormals. Then __syncwarp and arrive(sfull[s]). (3) In consume_item, wait full[s] then sfull[s], both by the same parity. In dq_stage, replace the per-record scale load + e4m3x4_to_f16x2x2 + per-k convert with ONE uint4 load per record from sc (8 bf16 = rows g and g+8 for its 4 k16 steps). Per k, build sg = prmt(w_k, w_k, 0x1010) and sg8 = prmt(w_k, w_k, 0x3232): 2 instructions instead of ~7. dequant8 and everything after it are unchanged. Reuse is safe: the helpers write sc[s] for fill f only after full[s] phase f, which follows the consumers' empty[s] arrive for fill f-1, and the consumers read sc[s] in dq_stage before that arrive. The dead/timeout path must still arrive on sfull. Nothing else moves: weight bank bytes (NVFP4 stays NVFP4, scales stay E4M3 in global memory; the expansion lives only in shared memory per stage), ring depth 4, buckets, gather, the variant-A wgmma_wait<0> schedule, epilogues, per-call tensor maps, s5's selector and the kernel_fn gate. BUILD: keep s5's wg_kernel as variant `s5` and add the new one as template flag SX. Bump the load_inline module name; TORCH_CUDA_ARCH_LIST=9.0a; -std=c++20 -O3 -lineinfo; print regs/spill/smem per instantiation; producer warps must stay within 40 registers with 0 B spill. PREPARE: s5's bucket canary (N16-N128) and three-state copy check (A, deepcopy B, tensor clone C; refilled graph replays; launch marker) on SX at M=128, 255, 256, 257 x four routings, against fp32 dequant with a bound of <=1.5x Marlin's own error. Outputs must be bitwise identical to s5, so print max |SX - s5| and require 0. SELECTOR: s5's graph-replay selector with variant candidates {s5, SX} vs Marlin per bucket (margin 1.02x on the judge-like zipf a=0.3 routing). Print the deep-ring per-kernel diagnostic line (touched experts, max routes per expert, w13/w2 us and TB/s) for both variants at T=128 on zipf a=0.3 and on the 17-touched judge-like routing, and at T=256. Assert the chosen variant is the printed winner. Hard rules: CUDA-graph capturable; kernel_fn does no host sync, allocation or branching on input values, no prints, no try/except, no per-call mutable state; nothing cached from inputs (the scale expansion is per launch, in smem); activations BF16; attend every route; no child processes. No module or static global may hold device pointers, TMA maps or buffers, because the judge clones TWO state copies (pointer-equality gates fail).

## Expected

- `bs=256 *` faster: B256 is always wgmma. s5's judged rows were w13+silu 239-300 us and w2 119-148 us. If consumer ALU issue is the limiter, cutting ~20-25% of dequant instructions per k16 step should take ~10-15% off each kernel row: target w13 <=215, w2 <=105, shape row <=370 us (from 405)
- `bs=128 *` faster: B128 (Marlin pair 306 us judged) flips only if the zipf a=0.3 selector ratio reaches >=1.02x (now 0.80x), so most likely it stays MARLIN on this change alone. The printed per-kernel TB/s at T=128 is the attribution either way
- `bs=32 *` flat: control: T<128 runs master's Marlin path
- `bs=1 *` flat: control

## Risk

Low-medium risk: no new MMA form and no register-A hazard. The new sync is an mbarrier (it has a timeout and an error word), not a named barrier, and outputs are bit-identical to s5, so correctness reduces to an exact compare. Rules: global weights and scales stay NVFP4/E4M3 bytes. The bf16 scale values exist only in per-stage shared memory inside the launch, exactly as s5 already builds them in registers, so this is not a BF16 weight copy. Main risk: the extra hop (helper waits full, converts, arrives sfull) adds latency in front of every stage. With 4 stages it should hide, but if the ring runs dry the kernel could be flat or slower. The printed TB/s against s5 on the same routing shows it. Sibling cuda-nvfp4-wgmma-rega-pingpong-consumers tests the other consumer hypothesis (dequant and MMA serialized, not ALU-bound). Each is attributable on the wg_kernel rows alone: if both win, stack them next; if both are flat, the limiter is neither and the next look is the per-expert tail (17 touched experts x 32 w2 tiles = 544 items over 132 CTAs = 4.12 waves). Judge on the bs=256 wg_kernel rows and the prepare TB/s lines, not the headline.

## Milestones

Rewrite stage 6 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
