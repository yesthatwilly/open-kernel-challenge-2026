---
problem: muse_prefill_layer
slug: experts-rega-epilogue-wg-local-bulk-store
created: 2026-10-27T133216Z
base: master
area: "w13 + w2 expert GEMMs (rega consumer epilogue: barriers and store path, both GEMMs)"
change: >
  Built on master (blob 42ded471). The k loop, producer, ring, tables and every value stay
  master's (with fsilu's w13 math); only the item epilogue changes, in BOTH rega GEMMs. MECHANISM:
  after each item both consumer warpgroups meet at named_bar(1, 256) twice (before staging, after
  staging) and then run a synchronous 16-byte store loop (4 iterations/thread for w13 at N=128, 8
  for w2 with an L2 sorted_ids load each), with the tensor core idle; the fast-silu win (w13
  x0.92-0.94, 5-10x the predicted cost) says epilogue time is fully exposed. The staging tile is
  already split by warpgroup: in w13 warp cw writes bytes 16cw..16cw+15 of each 144-B row (WG0 =
  cw 0-3 = bytes 0-63, WG1 = bytes 64-127); in w2 warp cw writes bytes 32cw..32cw+31 of each 272-B
  row (WG0 bytes 0-127, WG1 128-255). CHANGE, in a NEW module muse_prefill_rega_wgep = master's
  _rega_src() through fsilu's asserted edit (REGA_FSILU_OLD -> NEW) plus asserted edits of
  consume_item's epilogue: (a) gsc and (w2) the thread's sorted_ids entry for row t = ctid % 128
  are loaded at item start, before the k loop; (b) the two named_bar(1, 256) become
  warpgroup-local named_bar(2 + wg, 128) (ids 2 and 3; id 1 unused); (c) after stmatrix each
  thread executes fence.proxy.async.shared::cta, then the WG barrier, then each thread with t =
  ctid % 128 < nvalid issues one cp.async.bulk.global.shared::cta.bulk_group of its warpgroup's
  half row: w13 64 B from ep + t*144 + 64*wg to out + (slot0 + t) * out_ld + j*64 + 32*wg; w2 128
  B from ep + t*272 + 128*wg to out + sid * out_ld + j*128 + 64*wg when 0 <= sid < num_routes;
  then commit_group; (d) before its next staging the issuing threads run
  cp.async.bulk.wait_group.read 0, and wait_group 0 before exit. The two warpgroups now couple
  only through the stage barriers (empty needs both, so neither can lead by STAGES). Outputs are
  bit-identical to the incumbents. PREPARE BUDGET: retire the rega-lo phase (skip the
  _rega_lo_phase call; rega_low_variant stays 0, which it chose in 10 of 10 archived logs: guard
  ratio 1.030-1.037, never <= 1.00) to free its 39.6 s for this module's one ~40 s build, so
  prepare stays ~216-225 s. Phase after fsilu's, start deadline 230 s, step deadline 270 s. GATE
  per class and per GEMM: attrs (regs, local 0, smem, occ >= 1) per instantiation; hs (vs fsilu's
  w13) and y (vs master's w2) torch.equal on random, skewed, one and pop, error word 0, both
  markers; then w13-alone and w2-alone graphs, incumbent vs new, 2 rounds x 11 alternations,
  cold-L2 and warm-L2, calibrated judge-like pop and random; ON when <= 0.99 in every round, mode
  and routing; the two-state replay check with the path ON. state['rega_wgep_ok'][cls] = {'w13':
  bool, 'w2': bool}; _experts_rega dispatches each launch to the new module when ON, else today's
  choice. '[rega-wgep] PATH CHOSEN' lines last.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "rows rega_kernel<true,false> 255.5/278.0 and <false,false> 136.4/144.7 us (ec3bd018): the epilogue loses two CTA-wide syncs and the store loop; one warpgroup can run the next item while the other finishes its epilogue. Target w13 x0.97-0.99, w2 x0.95-0.98"
  - shapes: "*load=turn_4*"
    direction: faster
    why: w13 643.7/656.1 and w2 327.3/336.1 us, same ratios
  - shapes: "*load=first_turn*"
    direction: faster
    why: "w13 2995-3511 and w2 1576-1858 us, same ratios: -80 to -200 us at sliding first_turn (weight 138); headline +0.6-1.5% if the scored rows move"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-rega-pingpong-consumers-all-t: that forced a half-stage offset between the two warpgroups inside the k loop with per-stage named barriers and ran 1.18-1.47x slower; this keeps master's k loop (lockstep, variant A) and only drops the CTA-wide epilogue syncs and the synchronous store loop, so the warpgroups decouple only at item ends and only as far as the stage ring allows"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: w2st arm a landed; bulk store is arm b which lost (665c32dd)"
---

# experts-rega-epilogue-wg-local-bulk-store

## Change

Built on master (blob 42ded471). The k loop, producer, ring, tables and every value stay master's (with fsilu's w13 math); only the item epilogue changes, in BOTH rega GEMMs. MECHANISM: after each item both consumer warpgroups meet at named_bar(1, 256) twice (before staging, after staging) and then run a synchronous 16-byte store loop (4 iterations/thread for w13 at N=128, 8 for w2 with an L2 sorted_ids load each), with the tensor core idle; the fast-silu win (w13 x0.92-0.94, 5-10x the predicted cost) says epilogue time is fully exposed. The staging tile is already split by warpgroup: in w13 warp cw writes bytes 16cw..16cw+15 of each 144-B row (WG0 = cw 0-3 = bytes 0-63, WG1 = bytes 64-127); in w2 warp cw writes bytes 32cw..32cw+31 of each 272-B row (WG0 bytes 0-127, WG1 128-255). CHANGE, in a NEW module muse_prefill_rega_wgep = master's _rega_src() through fsilu's asserted edit (REGA_FSILU_OLD -> NEW) plus asserted edits of consume_item's epilogue: (a) gsc and (w2) the thread's sorted_ids entry for row t = ctid % 128 are loaded at item start, before the k loop; (b) the two named_bar(1, 256) become warpgroup-local named_bar(2 + wg, 128) (ids 2 and 3; id 1 unused); (c) after stmatrix each thread executes fence.proxy.async.shared::cta, then the WG barrier, then each thread with t = ctid % 128 < nvalid issues one cp.async.bulk.global.shared::cta.bulk_group of its warpgroup's half row: w13 64 B from ep + t*144 + 64*wg to out + (slot0 + t) * out_ld + j*64 + 32*wg; w2 128 B from ep + t*272 + 128*wg to out + sid * out_ld + j*128 + 64*wg when 0 <= sid < num_routes; then commit_group; (d) before its next staging the issuing threads run cp.async.bulk.wait_group.read 0, and wait_group 0 before exit. The two warpgroups now couple only through the stage barriers (empty needs both, so neither can lead by STAGES). Outputs are bit-identical to the incumbents. PREPARE BUDGET: retire the rega-lo phase (skip the _rega_lo_phase call; rega_low_variant stays 0, which it chose in 10 of 10 archived logs: guard ratio 1.030-1.037, never <= 1.00) to free its 39.6 s for this module's one ~40 s build, so prepare stays ~216-225 s. Phase after fsilu's, start deadline 230 s, step deadline 270 s. GATE per class and per GEMM: attrs (regs, local 0, smem, occ >= 1) per instantiation; hs (vs fsilu's w13) and y (vs master's w2) torch.equal on random, skewed, one and pop, error word 0, both markers; then w13-alone and w2-alone graphs, incumbent vs new, 2 rounds x 11 alternations, cold-L2 and warm-L2, calibrated judge-like pop and random; ON when <= 0.99 in every round, mode and routing; the two-state replay check with the path ON. state['rega_wgep_ok'][cls] = {'w13': bool, 'w2': bool}; _experts_rega dispatches each launch to the new module when ON, else today's choice. '[rega-wgep] PATH CHOSEN' lines last.

## Expected

- `*load=turn_1*` faster: rows rega_kernel<true,false> 255.5/278.0 and <false,false> 136.4/144.7 us (ec3bd018): the epilogue loses two CTA-wide syncs and the store loop; one warpgroup can run the next item while the other finishes its epilogue. Target w13 x0.97-0.99, w2 x0.95-0.98
- `*load=turn_4*` faster: w13 643.7/656.1 and w2 327.3/336.1 us, same ratios
- `*load=first_turn*` faster: w13 2995-3511 and w2 1576-1858 us, same ratios: -80 to -200 us at sliding first_turn (weight 138); headline +0.6-1.5% if the scored rows move

## Risk

Overlaps experts-rega-w2-epilogue-prefetch-bulk-store on w2 (its (a)+(b) are a subset of this); if that lands first, restack this on it and read the w13 rows plus the w2 increment. Risks: async-proxy ordering (fence before the WG barrier; torch.equal catches stale rows); a warpgroup-local barrier count mismatch hangs (both WGs execute the same epilogues; the mbar 'dead' path must still hit both WG barriers); the pingpong result means warpgroup drift may help nothing, so most of the gain is the store loop and the second sync. Retiring rega-lo removes the V64/V32 candidates from every later run; they never routed, and experts-rega-low-2cta-fullslab-mc2 (queued) brings its own module. Read the scored rega rows, not only the gate.
