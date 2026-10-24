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
stage: implemented
claimed_at: 2026-10-24T172705Z
base_sha: c0821b3a0b916336d23e7bbe2a454636be5b21cc
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/full-cuda-onelaunch-warpspec-bulk-gather
worktree: .worktrees/muse_attention_decode/full-cuda-onelaunch-warpspec-bulk-gather
mode: null
implemented_at: 2026-10-24T175215Z
impl_commit: 0785778cf4276e171a84f2a4c7d3ffc25893c346
submission_sha256: 34c696904f50319c35239749cf44711e4f93131f11c63d5588ade541a9712916
norm_sha256: c89f844774678ac0733b25b3098234d13b3e86133f22dd0501a3ea0aed197f80
submission_blob: a5c7945998f7e658cae185cc448230a0eb6c9bed
lint_warnings: 3
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


## Implementation notes

Worktree file `.worktrees/muse_attention_decode/full-cuda-onelaunch-warpspec-bulk-gather/muse_attention_decode/submission.py`, additions only (+850/-2; the -2 is the module docstring). Master's code (pre-attention, sliding, splitkv A/B, FA3, `_splitkv_self_check`) is unchanged byte for byte and stays the fallback.

- **New module `muse_full_ws_v1`** (`_FULL_WS_CPP` / `_FULL_WS_CUDA`, `_build_full_ws`), built after master's splitkv module, with `-O3 -arch=sm_90a`. Its kernel is `template <int BULK> full_ws_kernel` and the host entry is `full_ws(..., bulk)`. Both instantiations are compiled into the one module. The log prints the build seconds.
- **Gate** `_full_ws_gate(cfg)` (config only): module built, `_FULL_WS_OK`, window 0, q_len 1, 1 < batch <= 64, hq 16, hkv 2, d 64, page 1. In `kernel_fn` it is checked right after the sliding batch-1 branch and before `_pre_attention_kernel`, so on the gated path the one launch is the whole step. `_full_ws_forward` allocates `ws_acc [G+B,16,64]`, `ws_ml [G+B,2,16]` and `out [B,1,1024]` from shapes and the device. It uses persistent `_ws_counters(B)` int32 `[B+1]` (zeroed once, reset by the winners; `cnt[B]` is an error word).
- **Kernel**, as specced: G = SM count, 1 CTA/SM, 288 threads (warp 0 producer, warps 1-8 consumers). 8-stage ring of 64 tokens x (K 128 B + V 128 B) at a 144 B smem row stride (bank analysis: K word reads are `quad*4+qi` distinct; 8-byte V reads are conflict-free per half-warp). full[s]/empty[s] mbarriers, ~170 KB smem in all. Stream-K over 64-token tiles with prefix TP in smem and int64 range math. The producer prefetches page indices PF=12 tiles ahead with 4-byte cp.async into a 16-slot smem block. BULK=1 issues one `cp.async.bulk...complete_tx::bytes` per valid 128 B K/V row (2 tokens/lane) after `arrive.expect_tx(nvalid*256)`. BULK=0 issues 16 B cp.async (lane -> token lane/8, chunk lane%8, zero-filled) plus `cp.async.mbarrier.arrive.noinc` (full count 32). Consumers use master's math with the FP8 `mma.sync m16n8k32 e4m3` QK, using the fragment mapping of the reviewed full-decode-cuda-fp8-qk-mma, then m16n8k16 f16 PV with exact V widening and exp2 online softmax. Each consumer warp arrives on empty[s] after its PV. At segment end it does master's 4-warp merge, then stores the slot at c + b.
- **Review fixes (required):**
  1. *Slot collision:* there are no new-token slots. The new token is folded into the first-tile CTA's slice-0 warp of each kv head (m = q8.k8*scale, l = 1 on the qi==0 lane only since l is quad-reduced later, acc = v8 in master's acc column layout).
  2. *Ticket:* instead of a CTA count +1, each segment atomically adds its **tile count** (acq_rel, gpu) and the add that reaches tiles_b wins. This needs no counting of grazing or empty CTAs, so the off-by-one risk is gone. The winner reloads slots c+b for c in [cta_of(TP_b), cta_of(TP_{b+1}-1)], with `cta_of(t) = ((t+1)G-1)/T`, skipping empty-range CTAs.
  3. *BULK template:* both are built. After the self-check, `_full_ws_diagnostics` checks BULK=0 once (grid 3, kv 0.25). It then times BULK=1, BULK=0 and master's path eagerly on B=16 x 80000 and prints us/call and achieved TB/s (sum(seq)*256 B / time). This is a diagnostic only and never selects the path.
  4. *New token from rounded fp8:* the score uses the rounded e4m3 key times k_scale (inside `scale`), and acc uses the raw e4m3 value, with v_scale applied once in the combine.
  5. *q rounding order:* the kernel rounds the normalized q to E4M3 **before** qk_scale_factor, which is folded into the score scale. This matches FA3, reference.py (`(q / qk_scale_factor).to(e4m3) * qk_scale_factor`) and master's `_splitkv_reference`, so the self-check reference matches the kernel.
  6. *Zero-length requests:* tiles_b = max(1, ceil(seq/64)), so a zero-length request gets one fully masked tile (expect_tx 0) whose owner builds and folds its new token.
  7. *Empty ranges never arrive:* a CTA with s_c == e_c returns before the role split and touches no slot or ticket. A segment's tile count is always >= 1. A zero add is skipped and sets `cnt[B] |= 1`, which the self-check requires to be 0.
- **Self-check** `_full_ws_self_check` uses synthetic data from `_splitkv_synthetic` with lens [9000, 1025, 70, **0**, 4097, 1023, 1, 64] (0 added to exercise item 6) and (kv_scale, grid) in (1, device), (0.25, device), (0.25, 3), (1, 3). It requires: rel L2 < 2e-2 vs FP32, all finite, new cache rows vs master's `_pre_attention_kernel` on a cloned cache, and every counter, including the error word, back to 0. On failure it prints `[full_ws] SELF-CHECK FAILED`, `_FULL_WS_OK = False`, and master runs. The log also prints `[full_ws] PATH CHOSEN ...` at import, and once per config on first use after import (per-config lines are suppressed during the import checks so they don't mislead).
- **Defensive:** mbarrier waits are bounded (2^31 cycles). A timeout sets `cnt[B] |= 2` and the thread stops waiting, so a pipeline bug fails the self-check instead of hanging the 15-minute run. An illegal address would still be a sticky CUDA error that fails the whole run (unavoidable).
- **Deviations:**
  - (a) q is rounded before qk_scale_factor, not after (item 5 above, to match the reference and the review).
  - (b) The ticket counts tiles instead of CTAs (simpler and total; same release/acquire pattern).
  - (c) "Zeroed once" covers only the V region of the ring, with a `fence.proxy.async` before any bulk write. Per the review, the -inf score mask is the correctness mechanism and the zeroing only keeps stale V finite (P = 0 times NaN would be NaN). K garbage is removed by select, not by multiplication.
  - (d) Page-index prefetch is a 16-slot per-tile ring 12 tiles ahead, not a double-buffered 16-tile block (same effect, simpler group bookkeeping: wait_group PF for BULK=1, 2*PF for BULK=0).
  - (e) New-row check is "≥ 99% of bytes identical and dequantized rel L2 < 1e-2", not exact equality. Triton's sqrt/div may differ from CUDA's IEEE ones by an ulp, which could flip a rare E4M3 rounding. The judge allows 8%.
  - (f) The q for each segment is built at segment entry by the 256 consumer threads (a ~1 us stall per extra segment, 1-2 segments per CTA at B=16/32), not prefetched.
- **Uncertainties (built blind, nothing compiled):**
  - A host g++ `-fsyntax-only` pass over the CUDA source with stub headers is clean, with both instantiations parsed. The asm strings were checked by hand against the PTX ISA and CUTLASS usage only, and nvcc/ptxas has not seen them.
  - The likeliest breakages are the PTX forms `mbarrier.arrive.shared::cta.b64 _`, `cp.async.mbarrier.arrive.noinc.shared::cta.b64` and `atom.acq_rel.gpu.add.s32`, and the byte order of `cvt.rn.satfinite.e4m3x2.f32` (first source = upper byte, so pairs are passed high first). A byte-order swap would show up as new-rows-equal ~0 and a large rel L2 in the self-check, which falls back to master.
  - A build error also falls back cleanly to master.
- **Harvest reading:** look for `[full_ws] load_inline compiled`, the four `self-check` lines, the `diag BULK=1/0` and `diag master` TB/s lines, and `PATH CHOSEN`. If you see `SELF-CHECK FAILED` or `load_inline failed`, the run measured master.

### Pre-flight fixes (2026-10-24, after an Opus audit)
The audit verified every flagged PTX form (against CUTLASS) and emulated 3108 stream-K/ticket mixes. Two Python-only changes: the self-check stops at the first failure, and the diagnostics skip timing BULK=0 when its own check failed, so a broken variant costs ~1 s, not ~30-60 s. No kernel-side short-circuit was added; the audit explained why one would be unsafe here.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/full-cuda-onelaunch-warpspec-bulk-gather diff` (additions only, apart from the docstring):

- [x] CUDA-graph capturable. The gated path is `_full_ws_gate` (config plus import-time flags) -> `_full_ws_forward`: `torch.empty` from shapes, a persistent counter buffer, one extension launch. It has no `.item()`/`.cpu()`/`synchronize()`/tensor prints. `_log_path` prints config ints only, host-side. The `torch.cuda.synchronize()` calls the lint warns about are in the import-time self-check and diagnostics only.
- [x] Nothing derived from `data` is cached across calls. New state is `_WS_EXT` (the compiled module), the flags `_FULL_WS_OK/_BULK/_GRID`, `_ws_counters` (zeros, reset in-kernel, no data) and `_PATH_LOGGED` (config tuples). The workspace is fresh `torch.empty` per call.
- [x] n/a for weights: this problem has no weights to repack, and no `prepare()` is added.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. qkv and gate are read as bf16 and the output is bf16. New K/V are stored as E4M3 at cache_loc. q is rounded to E4M3 after RMSNorm (FA3's rounding), with fp32 accumulation, and nothing coarser.
- [x] Attention sees every key. Every position [0, seq_lens[b]) of every request is in exactly one tile of exactly one CTA, plus the new token folded in once. There is no top-k, skip, or weight-based pruning, and masked rows are only those past the length.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. `load_inline` is the same mechanism master uses. The lint finding on the docstring word "requests" (a false positive in the network regex) was reworded.
- [x] All work happens inside the timed call, with no output caching or replay. Pre-attention, the cache store, attention, combine and epilogue are all in the one launch per call. Import-time work is the build, the self-check and the timing diagnostics on synthetic data only.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept. No `triton.autotune` was added.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
warn  host-sync    L1832: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1869: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1888: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_attention_decode/submission.py
```

`python3 -I -m py_compile` passed, and the `__pycache__` it left in the worktree was removed.
