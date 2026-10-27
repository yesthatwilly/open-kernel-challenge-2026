---
problem: muse_attention_decode
slug: combo-tail-onepass-push-fastepi
created: 2026-10-27T132814Z
base: master
area: "combination (stream-K ticket merge onepass + PERHEAD push merge + fast-math epilogues: the serialized merge/epilogue tails)"
change: >
  COMBINATION for a same-day scored candidate in phase C: each of its parts sizes below the ~1.2%
  one-run margin alone, and the solo checks run in parallel, so no stacked round fits before
  24:00Z. Built on master 2eb03f76 (no depends_on). Apply EXACTLY the changes specified in three
  queued hypotheses and nothing else, copying the diff from hyp/muse_attention_decode/<slug> if a
  solo has already been implemented: (1) sk-ticket-merge-onepass (merge_range_onepass, MB = 8,
  behind template flag ONEPASS in splitkv_attn_sk_kernel); (2) sliding-decode1-ph-push-merge
  (sliding_cluster_kernel PUSH: entry mbarrier init + split cluster arrive / wait, st.async
  complete_tx push to owner RX slots, non-owners exit without the second cluster barrier, bounded
  owner wait with error word); (3) attn-fast-math-epilogues (rcp.approx o/L, rsqrtf RMSNorm,
  rcp.approx(1 + ex2(-g log2e)) sigmoid, in the routed epilogues only). To bound compile time,
  build each kernel ONE twin carrying every applicable flag instead of one twin per flag: sk
  variants 8 / 9 / 10 = <ONEPASS, FE> twins of 0 / 4 / 6; sliding_cluster_kernel<true, PUSH, FE>;
  c320_combine_kernel<320, 16, true, true, FE>; sw_combine_kernel<true, FE>; incumbents stay
  byte-identical. Gates exactly as the solos specify, per cell, each against that cell incumbent:
  sk cells (full decode B=16 and B=32 bands, PSEUDO sliding verify B=1, sd sliding decode band 3)
  route their twin at >= 1.01x in both interleaved rounds with the IQR bound; sliding decode_1
  routes the PUSH+FE twin at >= 1.02x in both rounds after the B=1 self-check, two-state replay
  and a zero error word; c320 full decode_1 and sw sliding verify B=2..16 / B=17..32 route their
  FE twin unless slower than 0.995x in either round. Correctness per twin as in the solos
  (incumbent cases, two-state replays, rel L2 vs incumbent printed). Every phase behind the
  import-clock guard (start only if clock + its deadline <= 430 s; else the incumbent). Print one
  PATH CHOSEN line per cell naming twin vs incumbent with both round ratios, so a gain is
  attributable cell by cell; the solos carry the per-mechanism attribution.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "push merge + fast epilogue on the PERHEAD tail: -0.45..-0.95 us at 1.03%/us = +0.45..1.0%"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "onepass ticket merge + fast finalize on PSEUDO: -0.55..-1.1 us at 0.68%/us = +0.4..0.75%"
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "onepass ticket merge: -1..-2 us at 0.112%/us = +0.1..0.2%"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "onepass ticket merge: -0.5..-1 us at 0.074%/us (small)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: "~4 slots per request at p4: inside noise"
  - shapes: "*layer=full load=verify_* *"
    direction: flat
    why: FA3 paths untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "combo-sk-full-line-c320-tail: the same apply-exactly-N-queued-changes pattern for a same-day candidate; this one stacks three new phase-C tail hypotheses (sk-ticket-merge-onepass, sliding-decode1-ph-push-merge, attn-fast-math-epilogues) on master after sk-full-line-gather and c320-combine-onepass landed, with one multi-flag twin per kernel to bound compile time"
review: required
status: queued
---

# combo-tail-onepass-push-fastepi

## Change

COMBINATION for a same-day scored candidate in phase C: each of its parts sizes below the ~1.2% one-run margin alone, and the solo checks run in parallel, so no stacked round fits before 24:00Z. Built on master 2eb03f76 (no depends_on). Apply EXACTLY the changes specified in three queued hypotheses and nothing else, copying the diff from hyp/muse_attention_decode/<slug> if a solo has already been implemented: (1) sk-ticket-merge-onepass (merge_range_onepass, MB = 8, behind template flag ONEPASS in splitkv_attn_sk_kernel); (2) sliding-decode1-ph-push-merge (sliding_cluster_kernel PUSH: entry mbarrier init + split cluster arrive / wait, st.async complete_tx push to owner RX slots, non-owners exit without the second cluster barrier, bounded owner wait with error word); (3) attn-fast-math-epilogues (rcp.approx o/L, rsqrtf RMSNorm, rcp.approx(1 + ex2(-g log2e)) sigmoid, in the routed epilogues only). To bound compile time, build each kernel ONE twin carrying every applicable flag instead of one twin per flag: sk variants 8 / 9 / 10 = <ONEPASS, FE> twins of 0 / 4 / 6; sliding_cluster_kernel<true, PUSH, FE>; c320_combine_kernel<320, 16, true, true, FE>; sw_combine_kernel<true, FE>; incumbents stay byte-identical. Gates exactly as the solos specify, per cell, each against that cell incumbent: sk cells (full decode B=16 and B=32 bands, PSEUDO sliding verify B=1, sd sliding decode band 3) route their twin at >= 1.01x in both interleaved rounds with the IQR bound; sliding decode_1 routes the PUSH+FE twin at >= 1.02x in both rounds after the B=1 self-check, two-state replay and a zero error word; c320 full decode_1 and sw sliding verify B=2..16 / B=17..32 route their FE twin unless slower than 0.995x in either round. Correctness per twin as in the solos (incumbent cases, two-state replays, rel L2 vs incumbent printed). Every phase behind the import-clock guard (start only if clock + its deadline <= 430 s; else the incumbent). Print one PATH CHOSEN line per cell naming twin vs incumbent with both round ratios, so a gain is attributable cell by cell; the solos carry the per-mechanism attribution.

## Expected

- `*layer=sliding load=decode_1 *` faster: push merge + fast epilogue on the PERHEAD tail: -0.45..-0.95 us at 1.03%/us = +0.45..1.0%
- `*layer=sliding load=verify_1 *` faster: onepass ticket merge + fast finalize on PSEUDO: -0.55..-1.1 us at 0.68%/us = +0.4..0.75%
- `*layer=full load=decode_16 *` faster: onepass ticket merge: -1..-2 us at 0.112%/us = +0.1..0.2%
- `*layer=full load=decode_32 *` faster: onepass ticket merge: -0.5..-1 us at 0.074%/us (small)
- `*layer=sliding load=decode_32 *` flat: ~4 slots per request at p4: inside noise
- `*layer=full load=verify_* *` flat: FA3 paths untouched

## Risk

Expected sum ~+1.0..1.9% headline, so it clears the ~1.2% margin in the upper half of its range; this is the one of the four meant for a scored submit today if its check beats 1.6340 by the margin (churn.py margin()). If a solo lands first, rebase on it and drop that part (the per-cell gates make the remainder independent). Risks are the union of the solos: PUSH hang (bounded wait + error word + eager self-check, mandatory), sk twin spills (MB = 4 fallback), and compile time (4 big twins + 2 small: print every [module] compiled-in line and the import clock at each gate; master reaches ~290 s with the sw phase). A twin that fails a check keeps that cell on its incumbent; it must never fall back further.
