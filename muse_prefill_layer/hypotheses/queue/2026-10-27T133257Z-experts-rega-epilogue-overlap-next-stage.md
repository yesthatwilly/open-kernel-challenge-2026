---
problem: muse_prefill_layer
slug: experts-rega-epilogue-overlap-next-stage
created: 2026-10-27T133257Z
base: master
area: "w13 + w2 expert GEMMs (rega consumer schedule across item boundaries: epilogue under the next item's first wgmma)"
change: >
  Built on master (blob 42ded471). Producer, ring, barriers, k loop body (dq_stage -> 8 wgmma ->
  commit -> wait<0> -> arrive), tables and every value stay master's (with fsilu's w13 math).
  MECHANISM: each consumer runs item i's k loop, then its epilogue (gs scale, silu/pack, stmatrix
  staging, two named barriers, store loop) while the tensor core sits idle; the ring is already
  full for item i+1 (the producer runs up to 4 stages ahead), so item i+1's first wgmma could run
  during that epilogue. CHANGE, in a NEW module muse_prefill_rega_ovl = master's _rega_src()
  through fsilu's asserted edit plus asserted edits that split consume_item into mainloop<NB>(acc,
  kb0) and epilogue<NB, G>(m, acc) (both forceinline, code unchanged) and replace the consumer
  item loop: for consecutive items of THIS CTA (it, it + gridDim.x) with the SAME N bucket, after
  item i's k loop: wait full(n); dq_stage into a0; zero a second accumulator set nxt[NA];
  fence_acc(nxt); wgmma_fence; the 8 wgmmas of item i+1's first stage into nxt; commit; NO wait;
  run item i's epilogue on acc (registers disjoint from nxt); then wgmma_wait<0>; fence_acc(nxt);
  arrive(empty n); ++n; continue item i+1's k loop from kb = 1 on nxt; alternate acc/nxt by
  unrolling the loop body twice (no register copies). When the bucket changes, or at the last
  item, run master's serial order. The stage order, n bookkeeping and per-item accumulation order
  are master's, so hs and y are bit-identical. Register risk: NB = 128 needs 2 x 64 accumulators +
  32 A registers; compile both rega_kernel_ovl<G, BIG, DMAX> with DMAX = 128 (overlap in every
  bucket) and DMAX = 64 (buckets <= 64 overlap, 128 serial), 8 instantiations in one build; the
  attrs check (local 0, regs, smem, occ >= 1) disables any spilling instantiation, and the build
  log prints ptxas wgmma-serialization warnings if any. PREPARE BUDGET: retire the rega-lo phase
  (skip _rega_lo_phase; rega_low_variant stays 0, its choice in 10 of 10 archived logs, guard
  1.030-1.037) to free 39.6 s for this one build (~50-60 s with 8 instantiations); phase after
  fsilu's, start deadline 225 s, step deadline 270 s. GATE per class (low 753, high 3012, ft
  16384) and per GEMM: hs vs fsilu's w13 and y vs master's w2 torch.equal on random, skewed, one
  and pop, error word 0, markers; then alone graphs, incumbent vs DMAX 128 vs DMAX 64, 2 rounds x
  11 alternations, cold-L2 and warm-L2, calibrated judge-like pop and random; ON (fastest passing
  DMAX) when <= 0.99 in every round, mode and routing; two-state replay check with the path ON.
  state['rega_ovl'][cls] = {'w13': DMAX or 0, 'w2': DMAX or 0}; '[rega-ovl] PATH CHOSEN' lines
  last.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "ft chunks are nearly all bucket 128, so every item boundary overlaps: the epilogue (~2-3K clk after fsilu) hides under one stage of wgmma (~1K clk at N=128) of a ~50K clk item: w13 3388 us and w2 1740 us (sliding, ec3bd018) x0.97-0.99, -50 to -150 us at sliding first_turn (weight 138)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class, mostly bucket 128: w13 643.7 / w2 327.3 us x0.97-0.99"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low class buckets 32/64 change between experts, so fewer boundaries overlap and one stage of wgmma at N=32/64 is only ~250-500 clk: w13 255.5 / w2 136.4 us x0.98-1.00"
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "experts-rega-pingpong-consumers-all-t: that overlapped dequant with wgmma inside the k loop by offsetting the warpgroups (1.18-1.47x slower), and the s3 double-buffered register-A variant was no faster; this keeps the k loop as master's lockstep variant A and overlaps only the item epilogue with the next item's first-stage wgmma on a second accumulator set"
review: required
status: queued
---

# experts-rega-epilogue-overlap-next-stage

## Change

Built on master (blob 42ded471). Producer, ring, barriers, k loop body (dq_stage -> 8 wgmma -> commit -> wait<0> -> arrive), tables and every value stay master's (with fsilu's w13 math). MECHANISM: each consumer runs item i's k loop, then its epilogue (gs scale, silu/pack, stmatrix staging, two named barriers, store loop) while the tensor core sits idle; the ring is already full for item i+1 (the producer runs up to 4 stages ahead), so item i+1's first wgmma could run during that epilogue. CHANGE, in a NEW module muse_prefill_rega_ovl = master's _rega_src() through fsilu's asserted edit plus asserted edits that split consume_item into mainloop<NB>(acc, kb0) and epilogue<NB, G>(m, acc) (both forceinline, code unchanged) and replace the consumer item loop: for consecutive items of THIS CTA (it, it + gridDim.x) with the SAME N bucket, after item i's k loop: wait full(n); dq_stage into a0; zero a second accumulator set nxt[NA]; fence_acc(nxt); wgmma_fence; the 8 wgmmas of item i+1's first stage into nxt; commit; NO wait; run item i's epilogue on acc (registers disjoint from nxt); then wgmma_wait<0>; fence_acc(nxt); arrive(empty n); ++n; continue item i+1's k loop from kb = 1 on nxt; alternate acc/nxt by unrolling the loop body twice (no register copies). When the bucket changes, or at the last item, run master's serial order. The stage order, n bookkeeping and per-item accumulation order are master's, so hs and y are bit-identical. Register risk: NB = 128 needs 2 x 64 accumulators + 32 A registers; compile both rega_kernel_ovl<G, BIG, DMAX> with DMAX = 128 (overlap in every bucket) and DMAX = 64 (buckets <= 64 overlap, 128 serial), 8 instantiations in one build; the attrs check (local 0, regs, smem, occ >= 1) disables any spilling instantiation, and the build log prints ptxas wgmma-serialization warnings if any. PREPARE BUDGET: retire the rega-lo phase (skip _rega_lo_phase; rega_low_variant stays 0, its choice in 10 of 10 archived logs, guard 1.030-1.037) to free 39.6 s for this one build (~50-60 s with 8 instantiations); phase after fsilu's, start deadline 225 s, step deadline 270 s. GATE per class (low 753, high 3012, ft 16384) and per GEMM: hs vs fsilu's w13 and y vs master's w2 torch.equal on random, skewed, one and pop, error word 0, markers; then alone graphs, incumbent vs DMAX 128 vs DMAX 64, 2 rounds x 11 alternations, cold-L2 and warm-L2, calibrated judge-like pop and random; ON (fastest passing DMAX) when <= 0.99 in every round, mode and routing; two-state replay check with the path ON. state['rega_ovl'][cls] = {'w13': DMAX or 0, 'w2': DMAX or 0}; '[rega-ovl] PATH CHOSEN' lines last.

## Expected

- `*load=first_turn*` faster: ft chunks are nearly all bucket 128, so every item boundary overlaps: the epilogue (~2-3K clk after fsilu) hides under one stage of wgmma (~1K clk at N=128) of a ~50K clk item: w13 3388 us and w2 1740 us (sliding, ec3bd018) x0.97-0.99, -50 to -150 us at sliding first_turn (weight 138)
- `*load=turn_4*` faster: high class, mostly bucket 128: w13 643.7 / w2 327.3 us x0.97-0.99
- `*load=turn_1*` faster: low class buckets 32/64 change between experts, so fewer boundaries overlap and one stage of wgmma at N=32/64 is only ~250-500 clk: w13 255.5 / w2 136.4 us x0.98-1.00

## Risk

An alternative mechanism to experts-rega-epilogue-wg-local-bulk-store for the same idle-tensor-core epilogue; if that lands, the remaining hideable epilogue shrinks, so restack this on it rather than run both blind. Headline +0.5-1.2%: near the margin, sized to stack. High risk: two 64-float accumulator sets near the 232-register consumer budget (ptxas may spill or serialize wgmmas; DMAX 64 is the fallback), and the bucket-change path doubles the consumer code. A mismatch in the n/stage bookkeeping hangs the pipeline (the bounded mbar wait turns it into error word 1 and the path OFF). Read the scored rega rows, not only the gate.
