---
problem: muse_attention_decode
slug: sliding-decode1-ph-push-merge
created: 2026-10-27T132706Z
base: master
area: "sliding-layer attention (batch-1 decode, PERHEAD cluster DSMEM merge: push instead of pull)"
change: >
  Built on master 2eb03f76 (no depends_on). Sliding decode_1 (weight 46, ~1.03%/us, the
  highest-rate row on the board) runs sliding_cluster_kernel<PERHEAD=true> in _SLIDE_PH_CUDA as 2
  clusters x C=16, 9.6 us kernel in a 10.82 us shape. Its merge tail today: every rank writes its
  8-row partial (P_ACC / P_M / P_L) to its own smem, cluster.sync() #1, owner ranks (rank r owns
  row r, NR = max(1, G/C) = 1; ranks 8..15 own none) PULL all C peers with two fully unrolled
  DSMEM load passes (m, then l + float4 acc), run the epilogue, then cluster.sync() #2 only so
  peers smem stays live. NEW template flag bool PUSH = false on sliding_cluster_kernel,
  instantiated only as <true, true> in the PH module (the <true,false> incumbent and <false,false>
  stay byte-identical). PUSH flow: (1) at entry thread 0 inits a receive mbarrier rbar (count 1)
  in a NEW dedicated region RX placed after the current SMEM_BYTES (C_MAX=16 slots x (D floats + m
  + l) = 4224 B + 16 B barrier, so the instance smem is ~107.7 KB, still under the 113 KB
  static_assert; print occupancy_of for the PUSH instance and require >= 2 co-resident C=16
  clusters as today), then fence.mbarrier_init.release.cluster and
  barrier.cluster.arrive.relaxed.aligned by all threads; (2) prologue, gather, MMA and the
  intra-CTA 8-warp merge unchanged; (3) in place of cluster.sync #1: barrier.cluster.wait.aligned
  (pairs with the entry arrive, so every owner rbar is initialised), then each owner thread 0 does
  mbarrier.arrive.expect_tx(rbar, C * (D + 2) * 4), and EVERY rank (owners included) pushes each
  row hl it holds into owner (hl / NR) slot RX[rank] with
  st.async.shared::cluster.mbarrier::complete_tx::bytes (.v4.f32 for the float4 of 16 threads per
  row, .v2.f32 for (m, l) from thread (tid & 15) == 0), addressed with mapa; (4) non-owner ranks
  are then done and exit, with NO second cluster barrier (nobody reads their smem); (5) owners
  wait on rbar parity 0 with a bounded try_wait loop (as the sw module WAIT_LIMIT: on timeout set
  an error word and fall through), then do a ONE-pass online-LSE over their C local slots and the
  UNCHANGED epilogue arithmetic (o / Ls * vs, 1/sqrtf RMSNorm, expf sigmoid), bf16 store. Host:
  slide_ph_attn gets a trailing int64 push (default 0 = today launch). In-prepare gate in
  _slide_ph_init after PERHEAD has routed: the PUSH instance passes the same B=1 self-check as
  PERHEAD today (lens 1 / 10 / 2047 / 2048 / 2049 / 80000 at kv 0.25 and 2049 at 1.0, vs the FP32
  reference and vs PERHEAD pull, new cache rows compared), the two-state replay through kernel_fn,
  and an error-word read of 0; then interleaved cold-L2 graph replays at B=1 x 80000 (median of
  31, two rounds, empty graph subtracted) vs PERHEAD pull; route PUSH only if >= 1.02x in BOTH
  rounds, freeze _SLIDE_PH_PUSH at import (config-only use in _slide_forward), print PATH CHOSEN:
  sliding B=1 PERHEAD merge push|pull (t_pull vs t_push both rounds). Every other band and path
  unchanged. Precision unchanged: E4M3 q, FP8 KV, fp32 merge, every window key attended.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "removes one 16-CTA cluster barrier and two DSMEM pull round trips from the end of a latency-bound 9.6 us kernel: -0.4..-0.8 us at 1.03%/us = +0.4..0.8% headline"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: Triton incumbent, untouched
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: sd stream-K route, untouched
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "sliding-decode1-cluster-per-kvhead: same PERHEAD kernel and layout kept byte-for-byte; what differs is only the merge tail (push with st.async + one receive mbarrier per owner instead of two cluster.sync and a DSMEM pull). full-decode1-sk4-cluster-dsmem-merge (neutral) and sliding-cuda-cluster-dsmem-onelaunch (regression) both used the pull merge with two cluster barriers; the push form has never been built on this problem"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.6%, high risk"
---

# sliding-decode1-ph-push-merge

## Change

Built on master 2eb03f76 (no depends_on). Sliding decode_1 (weight 46, ~1.03%/us, the highest-rate row on the board) runs sliding_cluster_kernel<PERHEAD=true> in _SLIDE_PH_CUDA as 2 clusters x C=16, 9.6 us kernel in a 10.82 us shape. Its merge tail today: every rank writes its 8-row partial (P_ACC / P_M / P_L) to its own smem, cluster.sync() #1, owner ranks (rank r owns row r, NR = max(1, G/C) = 1; ranks 8..15 own none) PULL all C peers with two fully unrolled DSMEM load passes (m, then l + float4 acc), run the epilogue, then cluster.sync() #2 only so peers smem stays live. NEW template flag bool PUSH = false on sliding_cluster_kernel, instantiated only as <true, true> in the PH module (the <true,false> incumbent and <false,false> stay byte-identical). PUSH flow: (1) at entry thread 0 inits a receive mbarrier rbar (count 1) in a NEW dedicated region RX placed after the current SMEM_BYTES (C_MAX=16 slots x (D floats + m + l) = 4224 B + 16 B barrier, so the instance smem is ~107.7 KB, still under the 113 KB static_assert; print occupancy_of for the PUSH instance and require >= 2 co-resident C=16 clusters as today), then fence.mbarrier_init.release.cluster and barrier.cluster.arrive.relaxed.aligned by all threads; (2) prologue, gather, MMA and the intra-CTA 8-warp merge unchanged; (3) in place of cluster.sync #1: barrier.cluster.wait.aligned (pairs with the entry arrive, so every owner rbar is initialised), then each owner thread 0 does mbarrier.arrive.expect_tx(rbar, C * (D + 2) * 4), and EVERY rank (owners included) pushes each row hl it holds into owner (hl / NR) slot RX[rank] with st.async.shared::cluster.mbarrier::complete_tx::bytes (.v4.f32 for the float4 of 16 threads per row, .v2.f32 for (m, l) from thread (tid & 15) == 0), addressed with mapa; (4) non-owner ranks are then done and exit, with NO second cluster barrier (nobody reads their smem); (5) owners wait on rbar parity 0 with a bounded try_wait loop (as the sw module WAIT_LIMIT: on timeout set an error word and fall through), then do a ONE-pass online-LSE over their C local slots and the UNCHANGED epilogue arithmetic (o / Ls * vs, 1/sqrtf RMSNorm, expf sigmoid), bf16 store. Host: slide_ph_attn gets a trailing int64 push (default 0 = today launch). In-prepare gate in _slide_ph_init after PERHEAD has routed: the PUSH instance passes the same B=1 self-check as PERHEAD today (lens 1 / 10 / 2047 / 2048 / 2049 / 80000 at kv 0.25 and 2049 at 1.0, vs the FP32 reference and vs PERHEAD pull, new cache rows compared), the two-state replay through kernel_fn, and an error-word read of 0; then interleaved cold-L2 graph replays at B=1 x 80000 (median of 31, two rounds, empty graph subtracted) vs PERHEAD pull; route PUSH only if >= 1.02x in BOTH rounds, freeze _SLIDE_PH_PUSH at import (config-only use in _slide_forward), print PATH CHOSEN: sliding B=1 PERHEAD merge push|pull (t_pull vs t_push both rounds). Every other band and path unchanged. Precision unchanged: E4M3 q, FP8 KV, fp32 merge, every window key attended.

## Expected

- `*layer=sliding load=decode_1 *` faster: removes one 16-CTA cluster barrier and two DSMEM pull round trips from the end of a latency-bound 9.6 us kernel: -0.4..-0.8 us at 1.03%/us = +0.4..0.8% headline
- `*layer=sliding load=decode_16 *` flat: Triton incumbent, untouched
- `*layer=sliding load=decode_32 *` flat: sd stream-K route, untouched
- `*load=verify_* *` flat: untouched
- `*layer=full *` flat: untouched

## Risk

Single-row idea at +0.4..0.8%: below the ~1.2% margin unless it hits the top of its range, so it lands alone only with luck and otherwise stacked (combo-tail-onepass-push-fastepi). Risks: (a) a hang if the expected tx bytes and the pushed bytes disagree: the bounded wait, the error word read in prepare and the eager self-check before any graph capture are mandatory, and any nonzero error word keeps PERHEAD pull; (b) a non-owner exiting with st.async in flight: the targets are owners that cannot exit before rbar completes, so this is allowed, but if any replay flakes, add barrier.cluster.arrive.release + wait before non-owner exit (keeps the pull-round-trip saving, gives back the barrier); (c) +4.2 KB smem per CTA must not drop co-residency of the two clusters (print it). Composes with the queued sliding-decode1-perkvhead-launch-select (C=8 / spread policy: launch configuration, not the merge); whichever lands second re-gates. A gain ports to muse_decode_layer d1s, which carries the same PERHEAD kernel (campaign W6 port).
