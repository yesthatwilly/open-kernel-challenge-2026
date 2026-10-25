---
problem: muse_decode_layer
slug: full-verify-cuda-splitkv-stage2
created: 2026-10-24T165224Z
base: master
area: FA3 call
change: >
  Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify
  blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16,
  296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build
  on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page
  64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched
  so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work
  items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query:
  master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as
  f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv
  heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query
  tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA
  merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert
  each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per
  warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t
  only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never
  dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @
  V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64]
  and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine:
  splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that
  request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16
  into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged
  lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else
  changes.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: "incumbent paired FA3 153.3 us + combine + pad/unpad glue (~6.8 us x3) + norm-gate ~170 us of 604 us; the gather is the same as decode_16 where stage 1 kernel A ran ~111 us (attention), plus ~41 GFLOP of mma.sync that overlaps it: target <=130 us A+B, ~30-40 us off, +5-7% on the shape, ~+1.5-2% on the verify_16 load"
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: "incumbent FA3 296.3 us + glue ~315 us of 933 us; target <=265 us A+B: ~40-50 us off, +4-6% on the shape, ~+1.5% on the load"
  - shapes: "*layer=full load=decode_*"
    direction: flat
    why: stage 1 kernels unchanged; control
  - shapes: "*load=verify_1 *"
    direction: flat
    why: batch 1 outside the gate
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
risk: high
depends_on: full-decode-cuda-splitkv-port
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32
review: proceed
status: running
review_at: 2026-10-24T165634Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T003235Z
base_sha: f6cc1452343c4d93911ad7303c4a5d7878afd2fe
base_blob: b83f5ec1976c82d5b45dcd92b6c0818ce04eda9f
branch: hyp/muse_decode_layer/full-verify-cuda-splitkv-stage2
worktree: .worktrees/muse_decode_layer/full-verify-cuda-splitkv-stage2
mode: null
restack_of: null
stale_base: true
---

# full-verify-cuda-splitkv-stage2

## Change

Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16, 296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page 64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query: master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @ V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64] and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine: splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16 into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else changes.

## Expected

- `*layer=full load=verify_16 *` faster: incumbent paired FA3 153.3 us + combine + pad/unpad glue (~6.8 us x3) + norm-gate ~170 us of 604 us; the gather is the same as decode_16 where stage 1 kernel A ran ~111 us (attention), plus ~41 GFLOP of mma.sync that overlaps it: target <=130 us A+B, ~30-40 us off, +5-7% on the shape, ~+1.5-2% on the verify_16 load
- `*layer=full load=verify_32 *` faster: incumbent FA3 296.3 us + glue ~315 us of 933 us; target <=265 us A+B: ~40-50 us off, +4-6% on the shape, ~+1.5% on the load
- `*layer=full load=decode_*` flat: stage 1 kernels unchanged; control
- `*load=verify_1 *` flat: batch 1 outside the gate
- `*layer=sliding*` flat: untouched

## Risk

High: new causal-in-block masking and a 4x larger query tile; mma.sync compute (~41/82 GFLOP at verify_16/32) is no longer negligible against the gather, so if A exceeds ~150 us at verify_16 the fix is fp8 conversion once per stage in smem, not more splits. Run only after stage 1 lands or is judged on its replaced kernels. Not attempted in muse_attention_decode yet; a win here is portable back there.

## Milestones

Rewrite stage 2 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T165634Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound stage-2 design. Warps own disjoint rows, the causal mask is right and the self-check is adequate, so it is worth one run as a rewrite. Its real risk is that A becomes ALU- or register-bound: four warps convert the same FP8 K/V tile four times, plus 4x mma.sync. Partials traffic is also 8x decode's, which stresses the <=130 us A+B target. Dispatch only after stage 1 (full-decode-cuda-splitkv-port) has run and its log shows the kernel compiled and self-checked, and on a check after the reset, not as a scored submit (risk high).

### Justification
Mechanism: at full verify_16/32 paired FA3 (153.3/296.3 us) streams the same page-1 KV gather as decode (146.8/289.0 us). FA3 verify costs only ~4% more than decode, so it is memory-bound, and LESSONS says this gather tops out at 1.3-1.7 TB/s. The attention port of the CUDA split-KV kernel A ran 111.4/245.5 us at decode_16/32 for the same bytes, so a verify kernel near that gather time would beat FA3 by ~30-45 us. The shape and load arithmetic checks out: verify_16 load is 46*462.4+16*603.9 = 30.9 ms, and 16*35 us = 0.56 ms is ~1.8%. verify_32: 16*45 of 44.3 ms is ~1.6%. The headline geomean moves ~0.5%, inside noise, so as the hypothesis says it must be judged on top_kernels per its Milestones. Decode rows (stage 1 kernel untouched), verify_1 (batch 1) and sliding rows are valid in-run controls. The causal bound t < cache_seqlens-7+i is correct (cache_seqlens = seq_lens+8, so token i sees keys <= seq_lens+i). Ragged lengths 1025/4097 already make chunks where tokens 0..6 are fully masked, which exercises the -inf partial path in the combine. Design soundness: the row-ownership layout (warp = kv head x 2 tokens x 8 heads, all 64 tokens of the tile) is the only one that keeps the accumulators in registers, and it uses all 16 mma rows (decode wasted 8). Weakest link 1, ALU: each tile's K and V are now converted e4m3->f16 by 4 warps per head instead of each warp converting a 16-token slice. That is 4x the cvt.rn.f16x2.e4m3x2 plus byte_perm work, about 4x2.5 MB of FP8 per SM at verify_16, on top of 4x the mma.sync (~41 GFLOP, ~70-100 us even at a good mma.sync rate). Compute that was hidden under the gather in decode may now be on the critical path. The stated fallback (an f16 smem tile per stage) costs ~32 KB per stage-slot and likely drops occupancy from 2 to 1 CTA/SM (stage 1 SMEM_A is ~93 KB, x2 = 186 of 228 KB). A better next-stage lever is QK^T with mma.sync m16n8k32 e4m3 (sm_89+/sm_90). Q and K are both already FP8, so it is exact, needs no K conversion and halves the QK mma count. Only V needs widening. Weakest link 2, registers: S 16x64 fp32 (32) + acc 16x64 (32) + Q A-frags for 4 k-steps (16) + P frags (16) + addressing is ~110-125 regs against the 128 cap of __launch_bounds__(256,2). Spills are likely. Weakest link 3, partials: 8 tokens x 16 heads x 64 fp32 = 32 KB per 1024-token item versus 4 KB in decode. That is ~40 MB written by A and re-read by the combine at verify_16 (~80 MB at verify_32), about 12% of the KV bytes, plausibly 10-20 us that the <=130 us A+B budget does not account for.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched: it defines the incumbent (paired FA3 + pad/unpad glue) that stage 2 replaces at verify_16/32. The saving is measured against the paired path, which is right, but it is conditional on pairing staying on master. If pairing were absent, the incumbent would be the unpaired FA3 (~7-13% slower on these shapes) and the gain would look larger. Not a confound that a companion needs to resolve.
- paired-heads-pad-unpad-triton (in the judge now, not landed): it shrinks exactly the pad/unpad glue (~6.8 us x3) that this hypothesis counts in its verify_16/32 saving. If it lands before stage 2's base is cut, the expected saving drops by ~10-20 us per shape. Re-baseline the incumbent kernel sum from whichever master the branch is cut from, and merge carefully because both edit the same paired branch of kernel_fn.
- marlin-block-m-cap-16: no mechanistic interaction (it is MoE, not attention). Marlin is still 48-51% of full verify_16/32 (284/501 us), and its run-to-run drift can mask a 30-45 us attention change in the per-shape row, so read the replaced kernels' own us in top_kernels.
- post-expert-norm-triton-small-t, sglang-pipeline-parity-router-hilo-m8: no interaction with the target shapes. The post-norm path is gated to small T and does not run at T=128/256. The router is unrelated.

### Interference and A/B needs
No companion is warranted. This is a rewrite stage judged on the replaced kernels' own time against the incumbent and the floor. The decisive evidence is in the profile: splitkv_attn_verify_kernel + splitkv_combine_verify_kernel us at full verify_16/32, compared with paired FA3 main + combine + pad/unpad + FusedRMSNormGate from master. The run carries its own controls: decode rows run the unchanged stage 1 kernel, and verify_1, decode_1 and every sliding row run master code. An ab-revert of pairing would only re-measure FA3 unpaired, which is already known. The useful follow-ups (e4m3 mma for QK, a larger verify chunk to cut partials traffic, launch_bounds adjustments) belong in stage 3, not here. Sequencing note: stage 1 has not run on muse_decode_layer yet. Building on unharvested stage-1 code risks spending a check on a fallback run, so dispatch only after stage 1 harvests with '[splitkv] load_inline compiled' and no 'SELF-CHECK FAILED' in its log.

### Open questions
- What are the register count and spills of splitkv_attn_verify_kernel under __launch_bounds__(256,2)? Add -Xptxas -v (or print the cudaFuncGetAttributes numRegs/localSizeBytes) to the log so the harvest can tell ALU-bound from spill-bound.
- How does combine time scale with 8x larger partials (~40/80 MB at verify_16/32)? Would a 2048-token verify chunk (halving partials, still ~600 items vs 264 CTAs) be the cheaper stage-3 lever compared with conversion work?
- If A exceeds ~150 us at verify_16, is QK^T via mma.sync m16n8k32 e4m3 (no K conversion, exact) preferable to the f16 smem tile, which likely costs the second CTA per SM?
- Which incumbent glue is on the base master: does paired-heads-pad-unpad-triton land first? The expected verify savings should be restated against that base.
- Is the verify_16/32 'agentic' spread's max page_table width large enough that the workspace (B*max_chunks*8*16*64*4 bytes) inside graph capture is a concern? At verify_32 it is roughly 32*max_chunks*32 KB.

