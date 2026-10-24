---
problem: muse_attention_decode
slug: full-decode-cuda-fp8-qk-mma
created: 2026-10-24T165302Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel
  (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the
  native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of
  widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row
  quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1
  = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers
  (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 +
  quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at
  dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR
  (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and
  the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4
  mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick
  and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 =
  row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened
  exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte
  for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached
  build is not reused; keep the import-time self-check and its thresholds as they are (it now
  validates the fp8 QK path).
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: kernel A 111.4 us at 2.95 TB/s vs the 63.2 us floor; per lane per tile it issues 16 cvt.f16x2.e4m3x2 for K+Q-side work plus 16 for V and 8+16 mma; dropping the K widening halves the cvt count and halves QK mma issue (the 8 zero rows waste half of either op). The review of stage 1 estimated ~40 us of cvt ALU at decode_16 that must hide under the gather; if A is partly issue-bound this is ~5-10% on A (100-106 us), shape ~113-118 us
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: kernel A 245.5 us at 2.67 TB/s, lower per-byte efficiency than decode_16, which points at issue/ALU pressure rather than pure DRAM; target 225-235 us
  - shapes: "*layer=sliding* *load=verify_* *layer=full load=decode_1*"
    direction: flat
    why: outside the CUDA decode path; untouched
risk: medium
depends_on: full-decode-cuda-gather-splitkv
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T165634Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T165634Z
base_sha: 700fed761b57b25cbae687d07371c4020bf38af9
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/full-decode-cuda-fp8-qk-mma
worktree: .worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma
mode: null
---

# full-decode-cuda-fp8-qk-mma

## Change

Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1 = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 + quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4 mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 = row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached build is not reused; keep the import-time self-check and its thresholds as they are (it now validates the fp8 QK path).

## Expected

- `*layer=full load=decode_16*` faster: kernel A 111.4 us at 2.95 TB/s vs the 63.2 us floor; per lane per tile it issues 16 cvt.f16x2.e4m3x2 for K+Q-side work plus 16 for V and 8+16 mma; dropping the K widening halves the cvt count and halves QK mma issue (the 8 zero rows waste half of either op). The review of stage 1 estimated ~40 us of cvt ALU at decode_16 that must hide under the gather; if A is partly issue-bound this is ~5-10% on A (100-106 us), shape ~113-118 us
- `*layer=full load=decode_32*` faster: kernel A 245.5 us at 2.67 TB/s, lower per-byte efficiency than decode_16, which points at issue/ALU pressure rather than pure DRAM; target 225-235 us
- `*layer=sliding* *load=verify_* *layer=full load=decode_1*` flat: outside the CUDA decode path; untouched

## Risk

Single attributable change on the measured kernel: read kernel A time in top_kernels at decode_16/32 against 111.4/245.5 us. If flat, kernel A is DRAM/TLB-bound on the random 128-byte rows, not issue-bound, and further ALU tuning of the decode kernel should stop (next levers would be bytes in flight: stages / CTAs per SM). Numerics: e4m3 x e4m3 products are exact in fp32 and the f16 path also widened them exactly, so only the tensor-core accumulation order/precision differs (Hopper fp8 MMA accumulates with reduced internal precision; for 64-term logits that is ~1e-4 relative, far inside the 3% bound; the self-check rel L2 should stay ~1.7e-3). PTX m16n8k32 e4m3 needs sm_89+, the build already uses -arch=sm_90a. Overlaps in source with full-verify-cuda-streamk-splitkv (which adds new kernels to the same CUDA string and uses the same fp8 QK op in its own kernel); whichever lands second rebases with a textual merge, shapes are disjoint.

## Adversarial review

Reviewed 2026-10-24T165634Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is one change you can attribute, inside the kernel being measured. The PTX m16n8k32 e4m3 fragment mapping in the spec is correct, and the numerics are within the rules. Two things are uncertain. ptxas may lower FP8 mma.sync on sm_90a to F2FP+HMMA instead of a native FP8 MMA. And the decode_32 'issue-bound' argument does not hold up. A flat result is still informative, so proceed as a medium-risk scored submit, but build in a fallback to the f16 v1 kernel.

### Justification
Mechanism. Kernel A at decode_16 is 111.4 us for ~329 MB (2.95 TB/s), against a 63.2 us floor. Per lane per 64-token tile it issues 16 cvt for K (8 calls of e4m3x4_to_f16x2x2), 16 cvt plus 8 prmt for V, and 8+8 mma.sync. It also does the softmax ex2/shfl work. The spec writes '8+16 mma'; the P@V count is actually 8 (2 hw x 4 j). The stage-1 review estimated ~40 us of cvt ALU that has to hide under the gather. Removing the K widening halves the cvt count, halves the QK mma count and frees 4 Q registers. If A is partly issue-bound, 5-10% on A is a fair upper bound. I checked the layout against the PTX m16n8k32 .e4m3 figures. A: a0 = row groupID, k 4*qi..+3; a1 = row +8; a2 = row groupID, k 16+4*qi..+3; a3 = row +8. B: b0 = k 4*qi..+3, b1 = k 16+4*qi..+3, col = groupID. C has the same f32 layout as m16n8k16. The proposed operands (a1 = a3 = 0, chunk index g*4 + ks*2 + {0,1} XOR (tt&7), byte offset 4*qi) match this. The K reads stay conflict-free: 8 tokens with distinct tt&7 x 4 qi words give 32 distinct banks. The once-per-item Q load is now a 4-way bank conflict, which costs almost nothing. Weakest link 1: native support on sm_90. FP8 mma.sync m16n8k32 is specified for sm_89+, but Hopper's documented FP8 tensor-core path is wgmma. ptxas may lower mma.sync e4m3 on sm_90a to F2FP conversions plus HMMA.16816. In that case the K cvt does not go away, and the Q operand could be re-converted every tile instead of once per item, which is flat or slower. I could not verify this here: there is no CUDA toolkit or GPU on this node. A flat A time therefore has two explanations: 'A is DRAM/TLB-bound', or 'the FP8 MMA was emulated'. Weakest link 2: decode_32. It runs at a lower bandwidth per byte (2.67 vs 2.95 TB/s), but its ALU work per byte is identical to decode_16. An issue-bound kernel would run at the same TB/s at both shapes. The drop as the footprint doubles (~350 -> ~700 MB of randomly permuted rows) points to TLB reach or DRAM row locality, not ALU. Expect decode_32 to gain no more than decode_16 in relative terms, likely less. The 225-235 us target is optimistic. Headline arithmetic: about 123 -> 116 us at decode_16 (load 2925 -> ~2813, 1.039x) and about 257.7 -> 240 us at decode_32 (load 5282 -> ~5000, 1.056x). That works out to only ~+1.6% on the 6-load geomean (1.347 -> ~1.37x), inside the +-1.5% CI and ~2% drift. Judge the run on kernel A's own top_kernels time against 111.4/245.5 us, not on the headline. Directions in `expected` are consistent with TASK.md: sliding, full decode_1 and every verify shape stay outside the gate and are byte-identical.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv (direct parent, same kernel): no confound. Only the QK operand path inside kernel A changes, and A's own time is read directly. Its pipeline depth (4 stages, 2 CTAs/SM, 128 KB in flight per SM) does decide whether A is issue-bound at all, so a flat result settles the question only for this pipeline configuration.
- fused-preattention-triton: it produces the FP8 q bytes that now feed the MMA directly instead of being widened. It is the same numeric input as today, since widening e4m3 to f16 is exact, so there is no confound beyond tensor-core accumulation order.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / -ab-tune-batch-gate / sliding-fused-triton-decode1-gate: the gated path runs with PAIRED=False and never touches FA3 or the sliding kernel. They cannot interact.

### Interference and A/B needs
No companion is warranted. The result can be read directly from kernel A's profile row at decode_16/32 against the parent's 111.4/245.5 us; the shapes outside the gate are byte-identical. An ab-revert of the parent is impossible (the kernel is the parent). A control that splits 'drop Q cvt' from 'drop K cvt' does not make sense, because both come from one MMA instruction change. The one ambiguity a companion cannot fix is native vs emulated FP8 mma.sync on sm_90a. That needs a SASS check: before claiming, compile the CUDA string offline with nvcc -cubin -arch=sm_90a on any machine with CUDA 12.4+ (no GPU needed) and grep cuobjdump -sass for QMMA/HMMA and F2FP counts. Do not spawn a process from submission.py: that is a rules violation. If you cannot do the SASS check, record a flat A time as 'emulated-or-memory-bound' (undetermined), not as proof that A is DRAM-bound. Do not stop the ALU line of work on that basis alone.

### Open questions
- Does ptxas for sm_90a emit native FP8 MMA for mma.sync.m16n8k32.e4m3, or F2FP+HMMA emulation? This decides how to read a flat result. Check the SASS offline before or alongside the run.
- Why is decode_32 at 2.67 TB/s vs decode_16 at 2.95 TB/s when its ALU work per byte is the same? That points to TLB reach or DRAM locality over the ~700 MB random footprint. If confirmed, the next lever is bytes in flight or access order (stages, CTAs/SM, chunk ordering for locality), not ALU.
- Register count after the change (qa[4][2] -> 4 regs). If it drops well below the 128-reg cap of __launch_bounds__(256,2), would 3 CTAs/SM become possible? That is a separate follow-up, not part of this run.
- Overlap with full-verify-cuda-streamk-splitkv (same FP8 op, same CUDA string). Whichever lands second rebases. This run's SASS and timing evidence on FP8 mma.sync should feed into that hypothesis's design.

