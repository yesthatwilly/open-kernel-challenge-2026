---
problem: muse_prefill_layer
slug: experts-rega-w2-epilogue-prefetch-bulk-store
created: 2026-10-27T133154Z
base: master
area: "w13 + w2 expert GEMMs (rega w2 consumer epilogue: loads and stores)"
change: >
  Built on master (blob 42ded471, experts-rega-w13-fast-silu-epilogue landed). NO NEW nvcc BUILD:
  master's muse_prefill_rega_fsilu module already compiles rega_kernel<false, *> (w2) but never
  dispatches it (_experts_rega always launches w2 from state['rega_ext'], and the fsilu gate's
  pair graphs use master's w2 too). MECHANISM: rega's epilogue is serialized (tensor core idle),
  and the fast-silu run showed it costs 5-10x what instruction counts predict. In master's w2
  epilogue (consume_item, the !G branch) (1) gsc = __ldg(gs + m.e) is issued only after the last
  wgmma_wait (the asm memory clobbers keep it there), and (2) the scatter loop 'for (v = ctid; v <
  nvalid * 16; v += 256)' issues, per 16-byte chunk, an __ldg of sorted_ids[slot0 + t] that is an
  L2 round trip on fresh slots, then the shared load and the global store: 8 iterations per thread
  at N=128. CHANGE, as further asserted text edits in _rega_fsilu_src() (each target must occur
  exactly once; every edit sits inside 'if constexpr (!G)' so the w13 instantiation the fsilu gate
  measures is textually unchanged): (a) at the top of consume_item, before the k loop, w2 only:
  load gsc and the thread's route id sid_r = (ctid < nvalid) ? __ldg(sorted_ids + m.slot0 + ctid)
  : -1 into registers; (b) after the stmatrix staging every consumer thread executes
  fence.proxy.async.shared::cta, then the existing named_bar(1, 256); then each thread with ctid <
  nvalid and 0 <= sid_r < num_routes issues ONE cp.async.bulk.global.shared::cta.bulk_group of 256
  B from ep + ctid * PITCH2 (272 = 17 x 16, aligned) to out + sid_r * out_ld + m.j * 128, then
  cp.async.bulk.commit_group; (c) before the FIRST named_bar(1, 256) of the next epilogue (and
  once after the item loop) the issuing threads run cp.async.bulk.wait_group.read 0 (staging
  reuse), and cp.async.bulk.wait_group 0 before the consumer exits. Values, addresses and rounding
  are master's: y is bit-identical. Rename the module (e.g. muse_prefill_rega_fsilu_w2st) so the
  cache cannot serve the old build. _rega_fsilu_attrs: fsilu's ON criteria read only the two w13
  rows; the w2 rows (regs, local 0, smem, occ >= 1) gate only the new w2 path. GATE per class (low
  753 when rega_low_variant == 0, high 3012, ft 16384), after the fsilu gates, reusing their
  scaffolding (_rega_fsilu_tables, _rega_fsilu_bench, the calibrated judge-like pop routing and
  random): exactness first, y from master's w2 vs the module's w2 on the same hs (from fsilu's
  w13) torch.equal on random, skewed, one and pop, error word 0 and the w2 launch marker; then
  w2-alone one-launch graphs, master vs new, 2 rounds x 11 alternations, cold-L2 (256 MB read
  flush) AND warm-L2, both routings; ON only when new/master <= 0.99 in every round, mode and
  routing; then the two-state replay check with the path ON. state['rega_w2st_ok'][cls];
  _experts_rega launches w2 from state['rega_fsilu_ext'] when that class is ON (variant 0 and
  fsilu ext present), else master's. Print per class the w2 us both ways and '[rega-w2st] PATH
  CHOSEN' lines after fsilu's (16 KB tail). Budget: no build; ~3-8 s of gating after fsilu
  (prepare ~216 -> ~224 s); a step deadline of 270 s leaves master's w2.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "w2 row rega_kernel<false,false> 136.4 us (sliding) / 144.7 us (full) in ec3bd018: removing 1-2 serialized L2 round trips (unrolled) or up to 8 (not unrolled) plus the synchronous store loop per item is ~0.8-4K clk of a ~19-54K clk item: w2 x0.94-0.985, -2 to -8 us per shape"
  - shapes: "*load=turn_4*"
    direction: faster
    why: w2 row 327.3 / 336.1 us -> ~310-322 us
  - shapes: "*load=first_turn*"
    direction: faster
    why: w2 row rega_kernel<false,true> 1576-1858 us -> x0.94-0.985, -25 to -100 us at sliding first_turn (weight 138)
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-w13-fast-silu-epilogue: that cut w13's epilogue math (accurate expf + IEEE divide); this leaves w13 alone and removes w2's epilogue latency (post-k-loop gs load, per-chunk sorted_ids loads, synchronous scatter loop), inside the fsilu module whose w2 is already compiled but unused, so it costs no nvcc build"
review: proceed
status: queued
review_at: 2026-10-27T133940Z
review_companions: null
---

# experts-rega-w2-epilogue-prefetch-bulk-store

## Change

Built on master (blob 42ded471, experts-rega-w13-fast-silu-epilogue landed). NO NEW nvcc BUILD: master's muse_prefill_rega_fsilu module already compiles rega_kernel<false, *> (w2) but never dispatches it (_experts_rega always launches w2 from state['rega_ext'], and the fsilu gate's pair graphs use master's w2 too). MECHANISM: rega's epilogue is serialized (tensor core idle), and the fast-silu run showed it costs 5-10x what instruction counts predict. In master's w2 epilogue (consume_item, the !G branch) (1) gsc = __ldg(gs + m.e) is issued only after the last wgmma_wait (the asm memory clobbers keep it there), and (2) the scatter loop 'for (v = ctid; v < nvalid * 16; v += 256)' issues, per 16-byte chunk, an __ldg of sorted_ids[slot0 + t] that is an L2 round trip on fresh slots, then the shared load and the global store: 8 iterations per thread at N=128. CHANGE, as further asserted text edits in _rega_fsilu_src() (each target must occur exactly once; every edit sits inside 'if constexpr (!G)' so the w13 instantiation the fsilu gate measures is textually unchanged): (a) at the top of consume_item, before the k loop, w2 only: load gsc and the thread's route id sid_r = (ctid < nvalid) ? __ldg(sorted_ids + m.slot0 + ctid) : -1 into registers; (b) after the stmatrix staging every consumer thread executes fence.proxy.async.shared::cta, then the existing named_bar(1, 256); then each thread with ctid < nvalid and 0 <= sid_r < num_routes issues ONE cp.async.bulk.global.shared::cta.bulk_group of 256 B from ep + ctid * PITCH2 (272 = 17 x 16, aligned) to out + sid_r * out_ld + m.j * 128, then cp.async.bulk.commit_group; (c) before the FIRST named_bar(1, 256) of the next epilogue (and once after the item loop) the issuing threads run cp.async.bulk.wait_group.read 0 (staging reuse), and cp.async.bulk.wait_group 0 before the consumer exits. Values, addresses and rounding are master's: y is bit-identical. Rename the module (e.g. muse_prefill_rega_fsilu_w2st) so the cache cannot serve the old build. _rega_fsilu_attrs: fsilu's ON criteria read only the two w13 rows; the w2 rows (regs, local 0, smem, occ >= 1) gate only the new w2 path. GATE per class (low 753 when rega_low_variant == 0, high 3012, ft 16384), after the fsilu gates, reusing their scaffolding (_rega_fsilu_tables, _rega_fsilu_bench, the calibrated judge-like pop routing and random): exactness first, y from master's w2 vs the module's w2 on the same hs (from fsilu's w13) torch.equal on random, skewed, one and pop, error word 0 and the w2 launch marker; then w2-alone one-launch graphs, master vs new, 2 rounds x 11 alternations, cold-L2 (256 MB read flush) AND warm-L2, both routings; ON only when new/master <= 0.99 in every round, mode and routing; then the two-state replay check with the path ON. state['rega_w2st_ok'][cls]; _experts_rega launches w2 from state['rega_fsilu_ext'] when that class is ON (variant 0 and fsilu ext present), else master's. Print per class the w2 us both ways and '[rega-w2st] PATH CHOSEN' lines after fsilu's (16 KB tail). Budget: no build; ~3-8 s of gating after fsilu (prepare ~216 -> ~224 s); a step deadline of 270 s leaves master's w2.

## Expected

- `*load=turn_1*` faster: w2 row rega_kernel<false,false> 136.4 us (sliding) / 144.7 us (full) in ec3bd018: removing 1-2 serialized L2 round trips (unrolled) or up to 8 (not unrolled) plus the synchronous store loop per item is ~0.8-4K clk of a ~19-54K clk item: w2 x0.94-0.985, -2 to -8 us per shape
- `*load=turn_4*` faster: w2 row 327.3 / 336.1 us -> ~310-322 us
- `*load=first_turn*` faster: w2 row rega_kernel<false,true> 1576-1858 us -> x0.94-0.985, -25 to -100 us at sliding first_turn (weight 138)

## Risk

Headline +0.3-1.0% if the w2 rows move as expected: likely below the 1.2% judge margin alone, so it is sized to stack with another epilogue gain (experts-rega-epilogue-wg-local-bulk-store or experts-rega-epilogue-overlap-next-stage); read the scored rega_kernel<false,*> rows against ec3bd018, not only the gate (LESSONS 2026-10-27: cold-L2 gates can show gains the warm judged graph does not). Risks: async-proxy ordering (missing fence.proxy.async gives stale rows; the torch.equal check catches it), and the fsilu attrs rule (today a bad w2 row would turn fsilu OFF; the split in (attrs) prevents that). If nvcc already unrolls the scatter loop and hoists its loads, expect the low end.

## Adversarial review

Reviewed 2026-10-27T133940Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is real for the consumer-bound classes (high, ft): the w2 epilogue has a gs load after the last wgmma_wait plus a scatter loop doing an L1-missing sorted_ids __ldg each pass, while the tensor core is idle. Low is at the HBM byte floor and should be flat. Headline ~0.15-0.6%, far below readability: the only readable evidence is the in-prepare interleaved warm+cold gate. This is the w2 subset of the queued experts-rega-epilogue-wg-local-bulk-store and does not stack with it; comparing scored rows to ec3bd018 is not readable.

### Justification
gsc = __ldg(gs + m.e) sits after wgmma_wait<0>/fence_acc; the scatter loop loads sorted_ids per 16-B chunk, each a new L1 line. Both consumer warpgroups run in lockstep, so exposed epilogue cycles are lost tensor time (fast silu's w13 x0.91-0.94 supports exposure). ft: w2 item ~30K clk; gs + 2-8 scatter round trips is 5-15% of it; x0.94-0.985 plausible. high: 6-8 scatter passes, plausible. low: w2 weights 604 MB in 136.4 us = 4.43 TB/s (92% of spec), HBM-bound, 1-3 passes, hidden by the 4-stage ring: expect flat. Headline ~0.15-0.6%. Weakest link: whether ptxas already unrolls/batches the scatter loads, and TMA contention of 128 one-row bulk stores with the producer's loads at ft. Prior (fast-silu) fair. Sibling experts-rega-epilogue-wg-local-bulk-store contains this change's w2 content: re-baseline, do not add. Same-code jobs 21911eb6 vs ec3bd018 differ by 1.4-14% on w2 rows, so judge on the same-process interleaved gate (warm-L2 and judge-like pop first; cold-only is not evidence).

### Confounders on the current kernel
- experts-rega-w13-fast-silu-epilogue: the new w2 compiles into and dispatches from the fsilu module; compare fsilu's PATH CHOSEN lines with ec3bd018 before attributing; w2st is unavailable when fsilu ext is None.
- experts-turn-rega-s4-*: the 4-stage ring hides part of the epilogue (low flat by construction) and the producer shares the TMA unit with the bulk stores; a reading caveat.
- align-two-phase-all-t-rega + combo-a normgather set nbk/nvalid (scatter passes) and keep sorted_ids L2-hot.
- stack-tonight-five-prefill (zero_nonlocal_w2): keep the sid < 0 / >= num_routes skip predicate exactly.

### Interference and A/B needs
No companion run. Fold the control into the gate: compile w2 with a template bool STORE_BULK in the same build; false = only the early gs/sid load (write sid into a 128-int __shared__ table before the staging named_bar and have master's 16-B st.global scatter loop read s_sid[t]). Time it alongside master and the full change in the same 2 x 11 alternations, cold and warm, both routings.

### Open questions
- Does ptxas already unroll and batch the scatter loop's sorted_ids loads (cuobjdump -sass)?
- Do 128 separate 256-B cp.async.bulk stores per item delay the producer at ft? The STORE_BULK=false arm answers it.
- Skip the low class (print only) to save prepare time.
- Confirm what 'r' counts in the gate lines before trusting epilogue-fraction estimates.
- Run this before the siblings; if it lands, the sibling's w2 half must gate against this w2.

