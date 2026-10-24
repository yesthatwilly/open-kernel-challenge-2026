---
problem: muse_attention_decode
slug: full-cuda-onelaunch-warpspec-bulk-gather
created: 2026-10-24T172110Z
base: master
area: full-layer attention (whole-path rewrite, warp-specialized one launch)
change: >
  Stage 1 of a rewrite of the FULL-layer path, built on master (stage 1 of the first CUDA kernel,
  full-decode-cuda-gather-splitkv, has landed; no depends_on). Why: full layers are 67-78% of the
  batch-16/32 loads. Master's splitkv_attn_kernel streams the page-1 gather at only 2.94 TB/s at
  decode_16 (327.7 MB in 111.4 us) and 2.67 TB/s at decode_32 (655 MB in 245.5 us). Against floors
  of 63.2 / 126.4 us, the shape costs 123.0 / 257.7 us with 3 launches: _pre_attention_kernel 2.1,
  kernel A, splitkv_combine 7.9-8.6, plus about 1.6 us of gaps. A 2.7x leader must run full layers
  near 1.15x floor. That takes three structural changes at once: higher achieved gather bandwidth,
  no tail imbalance, and no pre/post launches. The new kernel full_ws_kernel<L> (L = 1 in this
  stage) lives in its OWN load_inline module, muse_full_ws_v1, so a build failure leaves master's
  splitkv module and path intact. (1) ONE LAUNCH, fused pre/post. Each CTA, on entering a request
  segment, loads that request's q (16 x 64 bf16) from qkv, applies scaleless RMSNorm, times
  qk_scale_factor and rounds to E4M3 in smem (there is no RoPE on full layers). The CTA whose
  range contains a request's FIRST tile also builds the new token: it normalizes k, divides k and
  v by their scales, rounds to E4M3, stores them at cache_loc[b] as master does, and computes the
  new key's contribution from registers (s = q . k_new per head, so m = s, l = 1, acc = v_new
  dequantized) into a dedicated partial slot grid + b. The gather covers only positions [0,
  seq_lens[b]), so no CTA ever reads the row being written. (2) STREAM-K over 64-token tiles:
  tiles_b = ceil(seq_lens[b] / 64) with prefix TP in smem (B <= 64). CTA c owns the flat range
  [c*Ttot/G, (c+1)*Ttot/G), using int64, and emits one partial per request segment, at slot c + b,
  as in full-verify-cuda-streamk-splitkv. This removes the 1024-token-chunk tail: about 1250
  chunks over 264 CTAs is 4.7 vs 5 items, a ~5% loss. (3) WARP-SPECIALIZED PRODUCER: G = SM count,
  1 CTA per SM, 288 threads. Warp 0 is the producer and warps 1-8 are consumers. The ring has 8
  stages of 64 tokens x (128 B K + 128 B V); rows are padded to a 144 B smem stride against bank
  conflicts, about 147 KB in all, with full[s]/empty[s] mbarriers. The producer prefetches page
  indices 16 tiles ahead into a double-buffered smem block. For each tile it waits on empty[s],
  arms full[s] with expect_tx for the valid rows' bytes, and each lane issues
  cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes of 128 B for the K row and the
  V row of 2 tokens. That is one instruction per row instead of master's 8 cp.async16, and there
  is no CTA-wide __syncthreads per tile. Masked rows are not issued, and their smem is zeroed
  once. Compile-time switch BULK = 0 swaps in 16-byte cp.async, coalesced to full lines (lane ->
  token lane/8, chunk lane%8), with cp.async.mbarrier.arrive.noinc. Stage 1 ships with BULK = 1.
  Consumers keep master's math: warp w-1 takes kv head (w-1) & 1 and token slice ((w-1) >> 1) *
  16. QK uses mma.sync m16n8k32 e4m3 on raw fp8 bytes (as full-decode-cuda-fp8-qk-mma, whether or
  not that primary lands). PV uses m16n8k16 f16 with V widened exactly. Softmax is online in exp2,
  -inf safe. Each consumer arrives on empty[s] after its PV. At segment end the 4 warps per head
  merge in smem, and the CTA stores fp32 (acc [16, 64], m [16], l [16]) to its slot. (4) IN-KERNEL
  COMBINE + EPILOGUE: after the slot store, __threadfence, CTA barrier, then one
  atomicAdd(acq_rel, gpu) on a per-request ticket cnt[b]. The expected count is the number of CTAs
  whose range overlaps request b, computed from TP, plus 1 for the new-token slot. Persistent
  int32 counters are zeroed once at first eager use, and the winner resets them, as master's
  _counters does. The CTA drawing the last ticket reloads that request's slots with .cg, does the
  log-sum-exp merge (about 17 slots at B = 16, 9 at B = 32), times v_scale, then rmsnorm and
  sigmoid(gate), bf16 into [B, 1, 1024]. GATE (config only): window == 0, q_len == 1, 1 < batch <=
  64, hq 16, hkv 2, head_dim 64, page_size 1, and _FULL_WS_OK. When it is false, master's path
  (pre-attention + splitkv A + B, or FA3) runs byte for byte. SELF-CHECK at import, on synthetic
  data only, extending _splitkv_synthetic: lens [9000, 1025, 70, 4097, 1023, 1, 64], kv_scale 1
  and 0.25, grid = device and grid = 3. The grid of 3 makes ranges span several requests and start
  or end mid-request. It checks rel L2 < 2e-2 against the FP32 reference, all finite, the new
  cache rows equal to master _pre_attention_kernel's on a cloned cache, and every counter back to
  0 after the call. On failure it prints why, sets _FULL_WS_OK = False, and the master path runs.
  STAGES: (1) this one, full decode at 1 < B <= 64. (2) L = 8 verify in the same kernel. It uses
  64 rows per kv head on fp8 QK and once-per-CTA V widening, as the A8 spec does; that replaces
  A8/B8 if full-verify-cuda-streamk-splitkv lands, else FA3. (3) Batch 1, decode and verify. 80K
  keys over 132 CTAs makes about 132 partials for one request, too many for one CTA to merge from
  L2 in under ~4 us, so this stage adds a cluster-of-8 DSMEM pre-merge before the ticket, giving
  about 17 partials. It replaces FA3 / full-fused-triton-decode1-gate at full decode_1 and
  verify_1. MILESTONES from the profile and log: on full decode_16 and decode_32, full_ws_kernel
  is the ONLY kernel. The log prints the build seconds, BULK, the grid, and the self-check rel L2.
  Report achieved bandwidth = sum(seq_lens) * 256 B / kernel time: M1 >= 2.94 TB/s, master's
  level; M2 >= 3.3 TB/s (decode_16 <= 100 us, decode_32 <= 200 us); M3 >= 3.6 TB/s.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: incumbent 123.0 us shape = splitkv_attn_kernel 111.4 + splitkv_combine 7.9 + _pre_attention_kernel 2.1 + ~1.6 gaps, floor 63.2 us; one launch removes ~11.6 us, stream-K ~5% tail, warp-specialized bulk-row producer lifts 2.94 -> ~3.5 TB/s; target kernel ~92 us, shape ~95 us; full is 67% of the decode_16 load, ~1.18x on it
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: incumbent 257.7 us = 245.5 + 8.6 + 2.3 + gaps, floor 126.4 us; master achieves only 2.67 TB/s here; target kernel ~185 us (3.5 TB/s), shape ~190 us; full is 78% of the load, ~1.26x on it. Stage 1 headline ~1.07x (1.349x -> ~1.44x); stages 2-3 (verify, batch 1) add ~1.15-1.2x more
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: batch 1 outside the stage-1 gate (stage 3)
  - shapes: "*layer=full load=verify_*"
    direction: flat
    why: verify outside the stage-1 gate (stage 2)
  - shapes: "*layer=sliding*"
    direction: flat
    why: "control: sliding path untouched"
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "full layer at tp8/page 1: _pre_attention_kernel + splitkv_attn_kernel + splitkv_combine_kernel at decode 1<B<=64 (stage 1); verify A8/B8 or FA3 (stage 2); batch-1 FA3/Triton (stage 3)"
review: required
status: queued
---

# full-cuda-onelaunch-warpspec-bulk-gather

## Change

Stage 1 of a rewrite of the FULL-layer path, built on master (stage 1 of the first CUDA kernel, full-decode-cuda-gather-splitkv, has landed; no depends_on). Why: full layers are 67-78% of the batch-16/32 loads. Master's splitkv_attn_kernel streams the page-1 gather at only 2.94 TB/s at decode_16 (327.7 MB in 111.4 us) and 2.67 TB/s at decode_32 (655 MB in 245.5 us). Against floors of 63.2 / 126.4 us, the shape costs 123.0 / 257.7 us with 3 launches: _pre_attention_kernel 2.1, kernel A, splitkv_combine 7.9-8.6, plus about 1.6 us of gaps. A 2.7x leader must run full layers near 1.15x floor. That takes three structural changes at once: higher achieved gather bandwidth, no tail imbalance, and no pre/post launches. The new kernel full_ws_kernel<L> (L = 1 in this stage) lives in its OWN load_inline module, muse_full_ws_v1, so a build failure leaves master's splitkv module and path intact. (1) ONE LAUNCH, fused pre/post. Each CTA, on entering a request segment, loads that request's q (16 x 64 bf16) from qkv, applies scaleless RMSNorm, times qk_scale_factor and rounds to E4M3 in smem (there is no RoPE on full layers). The CTA whose range contains a request's FIRST tile also builds the new token: it normalizes k, divides k and v by their scales, rounds to E4M3, stores them at cache_loc[b] as master does, and computes the new key's contribution from registers (s = q . k_new per head, so m = s, l = 1, acc = v_new dequantized) into a dedicated partial slot grid + b. The gather covers only positions [0, seq_lens[b]), so no CTA ever reads the row being written. (2) STREAM-K over 64-token tiles: tiles_b = ceil(seq_lens[b] / 64) with prefix TP in smem (B <= 64). CTA c owns the flat range [c*Ttot/G, (c+1)*Ttot/G), using int64, and emits one partial per request segment, at slot c + b, as in full-verify-cuda-streamk-splitkv. This removes the 1024-token-chunk tail: about 1250 chunks over 264 CTAs is 4.7 vs 5 items, a ~5% loss. (3) WARP-SPECIALIZED PRODUCER: G = SM count, 1 CTA per SM, 288 threads. Warp 0 is the producer and warps 1-8 are consumers. The ring has 8 stages of 64 tokens x (128 B K + 128 B V); rows are padded to a 144 B smem stride against bank conflicts, about 147 KB in all, with full[s]/empty[s] mbarriers. The producer prefetches page indices 16 tiles ahead into a double-buffered smem block. For each tile it waits on empty[s], arms full[s] with expect_tx for the valid rows' bytes, and each lane issues cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes of 128 B for the K row and the V row of 2 tokens. That is one instruction per row instead of master's 8 cp.async16, and there is no CTA-wide __syncthreads per tile. Masked rows are not issued, and their smem is zeroed once. Compile-time switch BULK = 0 swaps in 16-byte cp.async, coalesced to full lines (lane -> token lane/8, chunk lane%8), with cp.async.mbarrier.arrive.noinc. Stage 1 ships with BULK = 1. Consumers keep master's math: warp w-1 takes kv head (w-1) & 1 and token slice ((w-1) >> 1) * 16. QK uses mma.sync m16n8k32 e4m3 on raw fp8 bytes (as full-decode-cuda-fp8-qk-mma, whether or not that primary lands). PV uses m16n8k16 f16 with V widened exactly. Softmax is online in exp2, -inf safe. Each consumer arrives on empty[s] after its PV. At segment end the 4 warps per head merge in smem, and the CTA stores fp32 (acc [16, 64], m [16], l [16]) to its slot. (4) IN-KERNEL COMBINE + EPILOGUE: after the slot store, __threadfence, CTA barrier, then one atomicAdd(acq_rel, gpu) on a per-request ticket cnt[b]. The expected count is the number of CTAs whose range overlaps request b, computed from TP, plus 1 for the new-token slot. Persistent int32 counters are zeroed once at first eager use, and the winner resets them, as master's _counters does. The CTA drawing the last ticket reloads that request's slots with .cg, does the log-sum-exp merge (about 17 slots at B = 16, 9 at B = 32), times v_scale, then rmsnorm and sigmoid(gate), bf16 into [B, 1, 1024]. GATE (config only): window == 0, q_len == 1, 1 < batch <= 64, hq 16, hkv 2, head_dim 64, page_size 1, and _FULL_WS_OK. When it is false, master's path (pre-attention + splitkv A + B, or FA3) runs byte for byte. SELF-CHECK at import, on synthetic data only, extending _splitkv_synthetic: lens [9000, 1025, 70, 4097, 1023, 1, 64], kv_scale 1 and 0.25, grid = device and grid = 3. The grid of 3 makes ranges span several requests and start or end mid-request. It checks rel L2 < 2e-2 against the FP32 reference, all finite, the new cache rows equal to master _pre_attention_kernel's on a cloned cache, and every counter back to 0 after the call. On failure it prints why, sets _FULL_WS_OK = False, and the master path runs. STAGES: (1) this one, full decode at 1 < B <= 64. (2) L = 8 verify in the same kernel. It uses 64 rows per kv head on fp8 QK and once-per-CTA V widening, as the A8 spec does; that replaces A8/B8 if full-verify-cuda-streamk-splitkv lands, else FA3. (3) Batch 1, decode and verify. 80K keys over 132 CTAs makes about 132 partials for one request, too many for one CTA to merge from L2 in under ~4 us, so this stage adds a cluster-of-8 DSMEM pre-merge before the ticket, giving about 17 partials. It replaces FA3 / full-fused-triton-decode1-gate at full decode_1 and verify_1. MILESTONES from the profile and log: on full decode_16 and decode_32, full_ws_kernel is the ONLY kernel. The log prints the build seconds, BULK, the grid, and the self-check rel L2. Report achieved bandwidth = sum(seq_lens) * 256 B / kernel time: M1 >= 2.94 TB/s, master's level; M2 >= 3.3 TB/s (decode_16 <= 100 us, decode_32 <= 200 us); M3 >= 3.6 TB/s.

## Expected

- `*layer=full load=decode_16 *` faster: incumbent 123.0 us shape = splitkv_attn_kernel 111.4 + splitkv_combine 7.9 + _pre_attention_kernel 2.1 + ~1.6 gaps, floor 63.2 us; one launch removes ~11.6 us, stream-K ~5% tail, warp-specialized bulk-row producer lifts 2.94 -> ~3.5 TB/s; target kernel ~92 us, shape ~95 us; full is 67% of the decode_16 load, ~1.18x on it
- `*layer=full load=decode_32 *` faster: incumbent 257.7 us = 245.5 + 8.6 + 2.3 + gaps, floor 126.4 us; master achieves only 2.67 TB/s here; target kernel ~185 us (3.5 TB/s), shape ~190 us; full is 78% of the load, ~1.26x on it. Stage 1 headline ~1.07x (1.349x -> ~1.44x); stages 2-3 (verify, batch 1) add ~1.15-1.2x more
- `*layer=full load=decode_1 *` flat: batch 1 outside the stage-1 gate (stage 3)
- `*layer=full load=verify_*` flat: verify outside the stage-1 gate (stage 2)
- `*layer=sliding*` flat: control: sliding path untouched

## Risk

Biggest risk: random 128-byte row gathers from a ~330-655 MB pool may have a physical ceiling well below 4.8 TB/s, from HBM row-buffer misses and TLB reach. The drop in master's own bandwidth from decode_16 to decode_32 (2.94 to 2.67 TB/s, at the same per-CTA work) hints that TLB or DRAM locality and not issue is part of the limit. If so, M2 fails while one launch and stream-K still give ~10-15 us. That reading must be taken from the achieved-TB/s line, not the headline. Second: cp.async.bulk at 128 B per op may be TMA-issue-bound. About 13 B/clk per SM is needed, so one 128 B op every ~9.5 cycles. If it is under master's 2.94 TB/s, flip BULK=0 in stage 2 before blaming the design. Third: the ticket combine needs the release/acquire pattern master uses in the sliding Triton kernel (fence, barrier, acq_rel atomic, winner .cg reloads), and the per-request expected count must include CTAs whose range only grazes the request. An off-by-one deadlocks nothing but drops a partial, which the self-check with grid 3 catches. Fourth: 1 CTA per SM with 8 stages trades master's 2-CTA latency hiding for depth. Read the stall reasons if M1 fails. Interaction: full-decode-cuda-fp8-qk-mma (running) edits master's kernel A. Judge this rewrite against whichever decode kernel is master at dispatch.

## Milestones

Rewrite stage 1 of `full layer at tp8/page 1: _pre_attention_kernel + splitkv_attn_kernel + splitkv_combine_kernel at decode 1<B<=64 (stage 1); verify A8/B8 or FA3 (stage 2); batch-1 FA3/Triton (stage 3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
