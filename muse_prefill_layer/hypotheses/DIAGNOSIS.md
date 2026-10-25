---
diagnosed_at: 2026-10-25T203436Z
problem: muse_prefill_layer
best: 1.4677x [1.4457,1.4898], job 73b40cd13fdf4bb0 (submit), combo-b-normgather-two-cta-low-regate. Its new path gated OFF, so this is combo A's code (9bfc488c36c84924, 1.4605) re-measured.
---

# muse_prefill_layer diagnosis

Per-load speedups recomputed from the best profile: turn_1 1.403, turn_4 1.491, first_turn 1.511. Each load counts equally, so turn_1 is the weakest. rega (w13 + w2) takes 49% / 43% / 45% of the weighted load time in each. Full-layer FA3 takes 12.8% / 16.4% / 9.1%. At first_turn the bf16 dense nvjet GEMMs take 38% of the sliding shape, at about 78% of the 989 TFLOP/s peak, so they have little headroom. The two boundary norm kernels take 813 µs, about 1.2x their memory floor.

## Diagnosis
- **Wrong bottleneck (4).** The last three rega runs attacked the dequant/MMA overlap on the consumer side. Those were `experts-rega-pingpong-consumers-all-t`, `experts-rega-producer-scale-expand-all-t` and `experts-rega-low-class-two-ctas-per-sm`. The pingpong log measures master's rega at 5.80 TB/s (w13) and 6.05 TB/s (w2) of L2->SM traffic at T=3012. Pingpong dropped that to 4.76 TB/s and ran 1.18-1.22x slower. The 2-CTA V64 low class plateaued at about 6.4 TB/s (2.29 TB/s of weights, 25.6 KB per stage). So in every class, rega's 128-row x ≤128-token tile is capped by L2->SM ingest. Most of that ingest is the bf16 activation slab, which every weight tile re-reads.
- **Local minimum (2).** The last eight results all sit within 1.4458-1.4677. combo A and combo B ran the same code and differed by +0.5%, inside the margin.
- **Repetition loop (1).** Align has had 8 hypotheses, from `moe-align-side-stream` through `align-two-phase-all-t-rega`, and it now costs only 7.5-27 µs. Four restacks of the same small 10-24 changes ran (`restack-router-hilo-align32-zerostore`, `stack-tonight-four/five/six-prefill`).
- **Missing fundamental (5).** No prefill hypothesis has used TMA multicast of the activation slab across a CTA cluster, and prefill has no PDL at all: `griddepcontrol` appears 0 times in submission.py, while muse_moe_w4a16 has it.
- **Ignored recommendations (6).** The pingpong reviewer said a flat result would hand the problem to `experts-rega-s4-bm256-wide-tile-high-ft`, which is still sitting in the queue. LESSONS's "needs the full 128-route slab under skew" (from the 2-CTA run) was never minted.
- **Not applicable.** Build churn (3) is cleared: since the sm_90a fix every load_inline build has compiled, and done/failed and done/rejected are empty. Unstacked wins (7): none; every per-shape gain is either stacked already (combo A) or gated off.

## Strategy
**targeted**: the CUDA rega engine is the right ceiling at about 1.75x its floor. The work is to fix the one limiter its logs name, L2->SM activation ingest, rather than to replace it or keep tuning its consumer loop.

## Actions (priority ordered; /hypothesize mints these first)
1. **What:** TMA multicast of rega's activation slab across a 2-CTA cluster (cluster dims (2,1,1)), in all three rega classes (low T=753, high T=3012, ft T=16384), as a new `rega_mc_kernel` behind a per-class prepare gate.
   - **Pairing.** The two CTAs of a cluster take adjacent weight tiles j=2J and 2J+1 of the same chunk. n_tiles is even: 64 for w13, 32 for w2.
   - **Loads.** Each producer loads one SLAB_HALF (k 0-63 by rank 0, k 64-127 by rank 1) with `.multicast::cluster` mask 0b11. The full barrier expects WST + 2·slab bytes.
   - **Empty barrier.** It counts 16 arrivals. Each consumer warp arrives on both CTAs' empty barriers (mapa plus a cluster-scope arrive).
   - **Unchanged parts.** consume_item, the bank, the align and the epilogue stay as they are, so the outputs are bit-identical to master and the exact-compare gate is the right one.
   - **Grid.** Size the persistent grid from cudaOccupancyMaxActiveClusters, not 132.
   - **Precedent.** muse_bf16_proj `bf16-verify-swapab-wgmma-cluster-dsmem-rewrite` already ran multicast::cluster on the judge.
   - **Prepare budget.** Prepare runs 140 s of 300 s today. Free budget in the same change by deleting the dead ssdq build/canary and the OFF 2-CTA low gate (33.2 s).

   **Why:** all three classes run at about 5.8-6.4 TB/s L2->SM.
   - At first_turn a stage moves 41 KB per CTA (9.2 KB of weight plus 32 KB of slab) every ~1530 clk, against 1024 clk of MMA. That is why w13 runs at 3669 µs (558 TFLOP/s) against a ~2070 µs floor.
   - Multicast cuts per-CTA ingest to 25.6 KB per stage at bucket 128 and to 17.4 KB at bucket 64, with no new register pressure.

   **Impact (upper bound):**
   - ft: w13 + w2 5504 -> ~4000-4400 µs on sliding first_turn (weight 138), about -10% of the first_turn load.
   - turn_4: 1098 -> ~800 µs, about -11% of the load.
   - turn_1: 418 -> ~350-380 µs, about -5%, because the low class is partly latency-bound (V32 at 4.4 TB/s was no faster than V64).
   - Headline: up to about x1.08-1.10 if ingest is the only cap.

   **Prior:** `experts-rega-pingpong-consumers-all-t` changed only the consumer and lost. This changes only the producer and barriers, and removes ingest.
2. **What:** run the queued `experts-rega-s4-bm256-wide-tile-high-ft` now, as is (review first, as its status says), in the same batch as action 1. It is the in-CTA way to halve activation bytes per FLOP. If both win, stack them: multicast plus a 256-row tile is a 4x cut in slab ingest per FLOP. **Why:** it is the reviewer's named successor once pingpong came out flat, and pingpong did. **Impact:** its own target is first_turn w13 ≤2900 µs and w2 ≤1450 µs, plus turn_4 w13+w2 1060 -> ≤880 µs. **Prior:** master's 128-row tile. The register risk (two accumulator sets near the setmaxnreg 232 ceiling) is the difference from action 1.
3. **What:** a low-class (200 < T ≤ 1024) variant that combines 2 CTAs/SM (setmaxnreg 32/104, as in V64) with the **full 128-route slab**, multicast across the cluster so the smem fits (2 stages per CTA). Popular experts then never split into extra chunks. Gate on the worse of the random and the 1.86x-skew guard routings, as combo B did. **Why:** turn_1 is the weakest load (1.403). V64 won 12% on random routing (0.878) but lost on the guard (1.035) because chunks grew to 4 per expert. The full slab keeps one chunk per expert at the judge's ~1356 routes (max 228 per expert). **Impact:** turn_1 rega 418 -> ~370 µs, roughly 5-7% of the turn_1 load and about +2% headline. **Prior:** `experts-rega-low-class-two-ctas-per-sm` / `combo-b-normgather-two-cta-low-regate`, which used a narrower slab and had no multicast.
4. **What:** a PDL chain over the expert block: `_norm_gather_kernel` -> rega w13 -> rega w2 -> `_post_expert_norm_weighted_sum_kernel`.
   - Each rega producer bulk-copies its first STAGES weight records and inits its barriers before `griddepcontrol.wait`. Weights do not depend on the previous kernel; only the activation TMA must wait.
   - Port the wrapper from muse_moe_w4a16's submission.py (line ~1029).

   **Why:** the persistent rega grid has a ±1-item tail, about 16-19 items per CTA at turn_1. muse_moe_w4a16 runs a PDL w13->w2 chain. Prefill has none. **Impact:** about 10-25 µs per shape, about 2-3% on turn_1 and under 1% elsewhere. **Prior:** none in prefill. In moe, the PDL chain did not interfere with front-end and epilogue wins (LESSONS, combo-toprow).
5. **What:** stage 2 of the full-layer FP8 attention rewrite (queued `attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload`). For the exp, consider packed `ex2.approx.f16x2` instead of FMA-polynomial offload: P is rounded to E4M3 anyway, and a 2-wide MUFU halves the exp cost. Get a rules check on the P precision before minting. **Why:** FA3 is about 1.6x its exp floor (turn_1 full 424 µs vs ~260 µs). The in-house wgmma kernel is 1.53-1.67x slower than FA3 (`attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc`). **Impact:** at most about 25% of FA3, which is about 3% / 4% / 2% of the three loads. That ceiling is lower than actions 1-3, so this comes after them. **Prior:** the hard-gate inproc run, whose design lacked warp specialisation and a TMA pipeline.

## Do not try
- `experts-rega-pingpong-consumers-all-t`, `experts-rega-producer-scale-expand-all-t`: consumer-side dequant/MMA overlap on the 128x128 tile. Their own gates measured them 1.18-1.47x and 1.03-1.10x slower, because the tile is L2->SM ingest-capped (5.8-6.05 TB/s). They are dead only until actions 1 or 2 remove the cap; retry them after that.
- `experts-rega-low-class-two-ctas-per-sm` with a V64/V32 narrow slab: it lost under judge-like skew (V64 1.035, V32 1.321) because experts split into extra chunks.
- Every queued item whose engine no longer runs on a scored shape (rega owns 200 < T ≤ 16384):
  - Triton engine: `experts-triton-first-turn-per-gemm-config-pick`, `experts-triton-turn4-persistent-local-align`, `first-turn-moe-dead-work-stack`.
  - Marlin: `marlin-w2-atomic-add-turn`, `experts-marlin-block-m-48-turn4`, `turn-moe-local-align-sorted-silu`.
  - ssdq: `experts-first-turn-ssdq-s2-large-m`, `experts-turn1-ssdq-s2-m64-tile`.
  - Already superseded by rega s4: `experts-turn-rega-s4-w2-only-t4096`, `experts-turn1-cuda-wgmma-rega-port-bucket32`, `experts-cuda-wgmma-s2-setmaxnreg-pingpong-first-turn`.
  - Ruled out: `experts-dequant-*`, under the user's rules ruling.
- The Triton NVFP4 expert engine: packed asm dequant, unmasked loads/L2 hints, persistent grid and BLOCK_T 32 measured flat, 17% slower, 3-5% slower and 1.6x slower respectively. rega beat it 0.63x at T=16384.
- MoE align (8 runs): align2 is 7.5-27 µs, under 1% of any shape.
- `norm-residual-boundaries-triton-row-kernel`, `moe-pre-post-norm-triton-multitoken`: SGLang's fused boundary is within 2% of a persistent multi-row variant (`norm-boundaries-persistent-multirow`).
- `dense-gemm-cublaslt-algo-pick-prefill-t`: algo picks timed in prepare did not carry over to the judge (bf16-cublaslt-algo-pick-in-prepare-m128-m256). The dense GEMMs already run at about 78% of peak.
- `router-triton-hilo-splitk-turn`, `router-pooler-concat-gemm-turn`, `router-hilo-one-gemm-n512`: the side-stream router is already hidden.
- `sliding-attn-block-triton-gqa-fused-normgate`: it was 1.26-2.6x slower than FA3 and the prize is about 1%. The FA3 contiguous-KV gather at turn shapes also stays out (gate ratio 0.974-0.984, path A chosen).

## LESSONS edits (proposed; churnsubs applies them)
- correct: `Knobs that did nothing: workspace blocks per SM, use_fp32_reduce=False, use_atomic_add.` -> "use_atomic_add did nothing at large M. On w2 at M≈1 it helps (moe bs=1 48.8 -> 44.9 µs; decode_1 −2.3/−3.5 µs)". The line as written contradicts the `w2-marlin-atomic-add-small-m` and `w2-marlin-atomic-add-t1-port` lines.
- retire: `2026-10-25 03:41Z harvest (prefill turn 1/4): the Triton NVFP4 expert GEMM loses to Marlin at T=753 ... Keep Marlin below first-turn sizes.` -> rega now beats Marlin at T=753 (gate 0.65x) and at T=3012, so the advice is obsolete (`experts-turn-rega-s4-port-t4096`).
- merge: the two prefill Marlin block-m lines (`On prefill, Marlin block_size_m from local rows...`) and the three prefill Triton-engine lines (packed asm dequant, unmasked loads/L2 hints, persistent grid at 74 blocks) -> one line: "prefill Marlin and Triton NVFP4 knobs are closed; rega replaced both on every scored shape".
- correct: `2026-10-25 prefill rega inner-loop tries: ... never went live; their own prepare gates measured them SLOWER` -> add the cause: "master's 128x128 rega tile already moves 5.8-6.05 TB/s L2->SM (pingpong log, T=3012), and pingpong cut that to 4.76 TB/s. The limiter is activation ingest, not exposed dequant, so cut activation bytes per FLOP (multicast or a wider weight tile) before any overlap change".
- correct: `... TLP helps; it needs the full 128-route slab (smem) to pay under skew.` -> add: "V64 already reaches ~6.4 TB/s L2->SM at bucket 64, so a full slab at 2 CTAs/SM needs multicast to fit both the smem and the ingest budget".
