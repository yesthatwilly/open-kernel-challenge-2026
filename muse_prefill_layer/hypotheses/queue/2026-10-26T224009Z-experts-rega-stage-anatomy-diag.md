---
problem: muse_prefill_layer
slug: experts-rega-stage-anatomy-diag
created: 2026-10-26T224009Z
base: master
area: "w13 + w2 expert GEMMs (prepare-only diagnostic: where a rega stage goes)"
change: >
  Built on master (blob f4751fc0). A prepare-only diagnostic; kernel_fn and every timed path stay
  byte-for-byte master's. QUESTION: DIAGNOSIS bans consumer-side dequant/MMA overlap on the
  128x128 tile until the L2->SM ingest cap is removed, but both cap-removal attempts ran slower
  (experts-rega-mc2-slab-multicast-all-t cut issued L2->SM bytes ~40% and was 1.31-1.81x slower in
  its own gate; experts-rega-s4-bm256-wide-tile-high-ft halved slab bytes per FLOP and was
  1.16-1.20x slower), and stage times fit dequant then MMA in series (~0.60 us per stage at N =
  32, ~0.93 us at N = 128). This run measures where a stage goes. Build one derived module
  muse_prefill_rega_anat (asserted edits of master's source) with an int argument MODE read by the
  consumers: 0 = master; 1 = no dequant (a0 filled once per item from constant bf16 pairs instead
  of dq_stage; codes and scales still land through the producer); 2 = no MMA (dq_stage runs and
  every a0 register is folded into acc by one dependent add so the compiler keeps the dequant;
  wgmma calls skipped); 3 = no epilogue math (pack zeros, keep the barriers and stores); 4 =
  producer only (consumers just wait and release stages). Time each mode with master's routed
  inputs at T = 753, 3012 and 16384 on the pop routing (_rega_pop_case with the alpha
  _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase
  calibrates) and on random, cold L2, 2 rounds x 11 alternations, and print per class and mode:
  us, max items per CTA, per-stage us (us / (max items per CTA x 32)), and the model check t0 vs
  t1 + t2 - t4 (serial) vs max(t1, t2) (overlapped). Outputs of modes 1-4 are garbage and never
  leave prepare. End with a '[rega-anat] VERDICT' line: serial, overlapped or ingest-bound (t4
  close to t0).
expected:
  - shapes: "*"
    direction: flat
    why: "diagnostic only: the timed path is master's, so every row is master re-measured; the value is the per-mode stage times in the log"
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
prior: "experts-rega-pingpong-consumers-all-t: that changed the consumer schedule and measured only end-to-end kernel time; this changes nothing on the timed path and times each stage component separately"
review: required
status: queued
---

# experts-rega-stage-anatomy-diag

## Change

Built on master (blob f4751fc0). A prepare-only diagnostic; kernel_fn and every timed path stay byte-for-byte master's. QUESTION: DIAGNOSIS bans consumer-side dequant/MMA overlap on the 128x128 tile until the L2->SM ingest cap is removed, but both cap-removal attempts ran slower (experts-rega-mc2-slab-multicast-all-t cut issued L2->SM bytes ~40% and was 1.31-1.81x slower in its own gate; experts-rega-s4-bm256-wide-tile-high-ft halved slab bytes per FLOP and was 1.16-1.20x slower), and stage times fit dequant then MMA in series (~0.60 us per stage at N = 32, ~0.93 us at N = 128). This run measures where a stage goes. Build one derived module muse_prefill_rega_anat (asserted edits of master's source) with an int argument MODE read by the consumers: 0 = master; 1 = no dequant (a0 filled once per item from constant bf16 pairs instead of dq_stage; codes and scales still land through the producer); 2 = no MMA (dq_stage runs and every a0 register is folded into acc by one dependent add so the compiler keeps the dequant; wgmma calls skipped); 3 = no epilogue math (pack zeros, keep the barriers and stores); 4 = producer only (consumers just wait and release stages). Time each mode with master's routed inputs at T = 753, 3012 and 16384 on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random, cold L2, 2 rounds x 11 alternations, and print per class and mode: us, max items per CTA, per-stage us (us / (max items per CTA x 32)), and the model check t0 vs t1 + t2 - t4 (serial) vs max(t1, t2) (overlapped). Outputs of modes 1-4 are garbage and never leave prepare. End with a '[rega-anat] VERDICT' line: serial, overlapped or ingest-bound (t4 close to t0).

## Expected

- `*` flat: diagnostic only: the timed path is master's, so every row is master re-measured; the value is the per-mode stage times in the log

## Risk

Ships master: the headline is noise; run it only with checks to spare. It decides whether DIAGNOSIS's ban on intra-warpgroup dequant/MMA overlap should be lifted (serial verdict) and bounds what the schedule ideas queued beside it can win (mode 4 is the ingest floor). Mode 2's sink must keep the dequant live: compare the derived kernel's register count with master's.
