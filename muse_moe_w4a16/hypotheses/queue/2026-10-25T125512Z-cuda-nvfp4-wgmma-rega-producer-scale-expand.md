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
review: hold
status: hold
review_at: 2026-10-25T212523Z
review_companions: null
hold_reason: "adversarial review: This design has already been measured and refuted. muse_prefill_layer's experts-rega-producer-scale-expand-all-t (job 6be2a8d3, written as an explicit 'port of "
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

## Adversarial review

Reviewed 2026-10-25T212523Z by an adversarial subagent (model opus).

**Verdict: hold.** This design has already been measured and refuted. muse_prefill_layer's experts-rega-producer-scale-expand-all-t (job 6be2a8d3, written as an explicit 'port of muse_moe_w4a16's queued cuda-nvfp4-wgmma-rega-producer-scale-expand') built the same helper-warp scale buffer, sfull barriers and prmt broadcast on the same s5-derived rega kernel. It was bit-identical but slower in every class: low T=753 w13 sx/master 1.135, w2 1.079, block 1.099, gate OFF. Its incumbent ran in the same ~2 TB/s regime as moe's wg_kernel. The hypothesis has no `prior` field and does not cite that run.

### Justification
A. The instruction count is plausible. Master's dq_stage (submission.py L1123-1140) spends a scale load, 0.5 cvt.e4m3x2, __half22float2, two fmul by 2^119 and two bf16 broadcasts per k16 step, about 7 instructions against roughly 22-24 for dequant8, so cutting it to 2 prmt removes about 20% of consumer ALU. The smem budget is right: Smem<false>::BYTES 203776 + 8192 + 8192 static = 220160 <= 232448. The new layout (r*1024 + cw*128 + g*16, a uint4 per record, 8 consumer warps) is consistent and conflict-free. Bit-exactness holds. The weakest link is the premise that consumer ALU issue limits the kernel. It is only inferred from a negative result: b128-n32-deep-ring showed that bytes in flight are not the limiter, which does not show that scale-conversion ALU is. No ncu issue-slot or stall data is cited. The decisive evidence is the prefill twin. LESSONS L182 and archive/experts-rega-producer-scale-expand-all-t/6be2a8d3cf7d489f.log record the same mechanism (helpers wait full[s], convert, arrive sfull with count 3; consumers wait full then sfull and use prmt 0x1010/0x3232) on prefill's rega_kernel, itself a port of moe s5 (same REC 4608, same 203776 -> 211968 smem). It gated OFF in all classes. At the low class (T=753, 751 routes over 32 experts, buckets 32/64) master's w13 was 305.2 us at 1.98 TB/s and sx was 346.3 us at 1.74 TB/s. That is the same small-N, ~2 TB/s weight-streaming regime this hypothesis targets at B256 (s5/wg2 w13 ~1.8-2.1 TB/s), so the 'prefill was activation-ingest-bound' explanation does not cover the low-class loss. The hop itself costs time: helpers wait on full[s] before the consumers can proceed, so it adds latency in front of every stage. That points to consumers waiting on stage arrival or a serial dequant -> wgmma -> wait<0> chain, not to ALU throughput. The `expected` bs=256 target (w13 <= 215 us, row <= 370) is therefore contradicted by direct measurement, and the bs=128 'faster' expectation (selector at 0.80x vs Marlin, needing >= 1.02x) was already unlikely by the file's own admission. bs=1/32 flat is consistent (Marlin path). The 'Do not try' lists do not name this slug, but LESSONS L182 is a measured refutation of the same code on a sibling kernel, and a missing prior when that obvious attempt exists is grounds for hold. A materially different follow-up the prefill file itself lists would not be a re-run: the consumer-only bit-trick scale conversion, with no extra barrier hop, built from the e4m3 bits by integer ops and folded into dequant8's existing mul. That is the version worth minting.

### Confounders on the current kernel
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (on master at 40e2e860): the base kernel. The hypothesis says to rebase s5's branch onto master, but master already contains s5 plus PDL and align2. Grafting the s5 branch again risks dropping wg-pdl-gather-w13-w2-chain's griddep_wait placement. Any build must start from master's wg_kernel as it is.
- wg-pdl-gather-w13-w2-chain (f7eac2c3): the producer tid 0 calls griddep_wait before the first activation TMA, and consumers call griddep_wait before the loop. Helper warps 1-3 waiting on full[s] inherit that wait. Under PDL the w2 helpers' first conversions overlap w13's tail, so a small w2 delta could come from PDL overlap shifting rather than from consumer ALU.
- combo-wg-occ2-v64-hot-interleave-align2 (the board, not on master; DIAGNOSIS action 1 restacks it tomorrow): if the restack lands first, B256 runs wg2 at 2 CTAs/SM (setmaxnreg 32/104, STG 3). There an extra 6-8 KB of scale buffer per CTA may break the 2-CTA fit, helpers drop to 32 registers, and TLP already hides consumer latency (0.955). The change would then measure something different from the s5 premise, and DIAGNOSIS action 3 explicitly asks for consumer work 'rebased on the restacked wg2 master'.

### Interference and A/B needs
No companions. The run would already contain an exact in-run A/B: s5 vs SX in the same graph-replay selector on the same routings, bit-identical outputs, printed per-kernel us and TB/s. Interference with landed items is not the problem. The problem is that the identical mechanism was measured slower (1.08-1.14x per kernel) on a near-identical kernel in the same TB/s regime, so a re-run would most likely gate SX off and re-measure master. If the author re-mints, make it materially different: (a) the consumer-only integer bit-trick scale conversion with no helper hop or extra barrier, or (b) wait for cuda-nvfp4-wgmma-rega-pingpong-consumers (sibling, p1) to show whether dequant/MMA serialisation is the limiter. The prefill pingpong port also lost (1.18-1.47x), but there the cause was activation ingest, which does not apply at moe small N. Build either on the post-restack wg2 master. Per the new lesson, use two interleaved selector rounds, both >= 1.03, with an IQR bound, not the single 1.02 margin.

### Open questions
- Why did the prefill twin get slower rather than flat at T=753 (w13 1.135)? If the consumers were ALU-bound with a full ring, the helper hop would be hidden. A slowdown suggests the consumers usually wait on full[s], so the ring is near-empty or the stage-arrival latency is exposed. That would also contradict the 'consumer ALU' reading of b128-n32-deep-ring. Has anyone printed per-stage mbar_wait spin counts or ncu stall reasons for wg_kernel?
- Is the restack of wg2 (DIAGNOSIS action 1, queue restack-occ2-wg2-b256-two-stream-halves-on-master) going first tomorrow? If so, should every consumer-side wg stage, including pingpong-consumers, be re-based on wg2 before dispatch?
- Should the consumer-only bit-trick (e4m3 -> bf16 via integer shift/or plus exponent rebias, fused into dequant8's existing fma.rn.bf16x2 multiplier) be minted as the replacement? It saves the same ALU with no extra barrier hop or smem round trip.

