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
review: proceed
status: running
review_at: 2026-10-24T172607Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T172705Z
base_sha: c0821b3a0b916336d23e7bbe2a454636be5b21cc
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/full-cuda-onelaunch-warpspec-bulk-gather
worktree: .worktrees/muse_attention_decode/full-cuda-onelaunch-warpspec-bulk-gather
mode: null
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

## Adversarial review

Reviewed 2026-10-24T172607Z by an adversarial subagent (model opus).

**Verdict: proceed.** The rewrite targets the right place: full decode_16/32 are 92-96% one gather kernel running at 2.67-2.94 TB/s against a 4.8 TB/s floor. Fusing to one launch and using stream-K are sound and should be worth about 10-15 us on their own. The spec has one definite indexing bug (new-token slots collide with segment slots) that will trip the self-check and silently fall back to master, so fix it before dispatch. Ship both BULK variants in the module so a miss on the bandwidth milestones can be attributed from a single run.

### Justification
Mechanism: FINDINGS shows full decode_16 = splitkv_attn 111.4 + combine 7.9 + pre 2.1 us (123.0 total, floor 63.2) and decode_32 = 245.5 + 8.6 + 2.3 (257.7, floor 126.4). Removing the pre and combine launches and the gaps (~11-12 us at B=16, ~12.5 us at B=32) is close to certain if the fused combine tail stays short. Stream-K over 64-token tiles removes the 1024-chunk quantization: about 1250 chunks over 264 CTAs is 4.73 vs 5 items, ~5%, which is plausible. The expected directions match TASK.md: the full shapes have weight 16 and are 67-78% of the decode_16/32 loads, while sliding, decode_1 and verify sit outside the config-only gate and should be flat. Numeric nits: with G = SM count = 132 (not 264), each request at B=16 spans ~8-9 CTAs, so ~10 slots, and ~5 at B=32, not the 17/9 stated. That is fewer, so the winner's merge is cheaper than budgeted. reference.generate_input fully permutes the pages (randperm over the whole pool), so every token costs two independent random 128-B row reads (K and V are separate tensors). Bytes in flight are not the limit: 8 stages x 16 KB per SM is ~17 MB GPU-wide against a Little's-law need of ~5 MB, and master already holds ~96 KB/SM in flight. So the move from 2.94 to 3.5 TB/s has to come from issue efficiency, sync removal or DRAM behaviour, not depth. WEAKEST LINK: the consumer side, more than the producer. One CTA per SM leaves 8 math warps per SM (2 per SMSP), where master has 16 (2 CTAs x 8 warps). At 3.5 TB/s each SM must retire one 16 KB tile every ~1200 cycles. Each consumer warp spends roughly 16 mma plus ~150-250 ALU instructions per tile (cvt e4m3->f16, byte_perm, ex2, shuffles, masks), all latency-bound chains. At 2 warps/SMSP, achieved IPC can drop far enough to make math, not memory, the limiter. The FP8 m16n8k32 QK halves the QK mma and removes the K converts, which helps, but V widening stays. Second weakest: cp.async.bulk at 128 B per op. TMA small-copy throughput on Hopper is the usual reason FA3/FlashInfer use cp.async for page-1 gathers, and the BULK=0 fallback correctly exists for this. Third: random-row DRAM/TLB efficiency (the 2.94 -> 2.67 drop between B=16 and B=32 at equal per-CTA work). This may cap even a perfect kernel near ~3.3-3.6 TB/s, so M3 is a stretch but M2 is physically reachable. The design can approach its floor in principle, and the correctness plan (synthetic self-check with grid 3, mid-request range boundaries, kv_scale 0.25, cache-row equality, counters back to zero) is adequate apart from the slot bug below.

### Confounders on the current kernel
- The landed list for master's submission.py is empty, so no landed hypothesis confounds. The incumbent being replaced is master's own splitkv path (full-decode-cuda-gather-splitkv), and the milestones compare against its kernel times directly.
- In flight, not landed: full-decode-cuda-fp8-qk-mma edits master's kernel A and uses the same FP8 QK math this rewrite adopts. If it lands before dispatch, the incumbent baseline shifts. Judge M1-M3 against whichever kernel A is master at dispatch (the hypothesis already says so), and read the rewrite's gain as structural (one launch, stream-K, producer), not as QK math.
- In flight: sliding-deep-splits-on-cuda-splitkv may share or extend the splitkv module and its gating. The separate muse_full_ws_v1 module isolates the build, but the kernel_fn gate order (full_ws before splitkv) must be rechecked if that lands first.
- Queued/held: fused-full-combine-normgate overlaps sub-change (4), the in-kernel combine plus epilogue. If this rewrite lands, that hypothesis is superseded for 1<B<=64 full decode and should be retired, not run on top.

### Interference and A/B needs
With nothing landed there is no ab-revert target, and for a kind: rewrite the next stage is the follow-up. The one real ambiguity in a single run is a miss on M1/M2: it could come from TMA small-copy issue (BULK=1) or from consumer latency at 1 CTA/SM. An extra check is not needed to resolve that. Instantiate the kernel as a template over BULK and build both into the one muse_full_ws_v1 module (one compile, ~same seconds). Gate on BULK=1 as planned. At import, after the self-check, run an eager synthetic timing of both instantiations (e.g. B=16 x 80K lens, cuda events, a few reps) and print the achieved TB/s for each, as a diagnostic only that does not select the path. That attributes the result from a single run, and stage 2 can flip BULK without a rebuild. A separate control companion would duplicate that information at the cost of a check, so none is proposed.

### Open questions
- SLOT COLLISION (fix before dispatch): segment slots are c + b for c in [0, G) and b in [0, B), so up to G + B - 2, and the new-token slots are 'grid + b', i.e. G .. G + B - 1. The last CTA always covers request B-1 at slot G + B - 2, which equals the new-token slot of request B-2. The same happens for other (c, b) near the end, so the self-check (grid 3, 7 requests) will fail and the run will silently measure master. Either offset the new-token slots to G + B - 1 + b, or, simpler and cheaper, fold the new token into the first-tile CTA's running state (initialize that warp's m = s_new, l = 1, acc = v_new), which removes the extra slot and the '+1' from the expected count.
- Consumer occupancy: why 1 CTA/SM with 8 consumer warps rather than 2 CTAs/SM at ~100 KB each (e.g. 5-6 stages) or 12-16 consumer warps in one CTA? Master's 16 math warps per SM may be part of why it reaches 2.94 TB/s at all. If M1 fails, read the stall reasons (long scoreboard vs math pipe vs barrier) before changing the producer.
- Slot counts per request: with G = 132 the stated ~17 (B=16) / ~9 (B=32) look like G = 264 numbers. Confirm the grid. If G = 264 (2 per SM) was intended, it contradicts '1 CTA per SM, ~147 KB'.
- Zero-length requests: the judge's seq_lens are clamped to >= 1 (agentic) and >= 16 (mixed), but tiles_b = 0 leaves the new token with no owner. Define the owner as the CTA containing flat position TP_b (or the next nonempty one) so the kernel is total.
- Masked rows 'zeroed once' do not stay zero once the ring is reused, so stale rows from earlier tiles remain. That is harmless only because the scores are masked to -inf (P = 0) and stale V is finite. State that the -inf mask, not zeroing, is the correctness mechanism.
- Combine tail: the winner merges ~5-10 slots of 16x64 fp32 from L2 with .cg plus the epilogue, ~1-3 us on the critical path after the last CTA. Report the kernel's own time minus a no-combine variant only if M2 is missed narrowly.

