---
problem: muse_bf16_proj
slug: router-pdl-plus-khalf-pooled-iqr-gate
created: 2026-10-27T143409Z
base: master
area: router m=9..256 PDL (restack) + pooler m=128 khalf gate rule
change: >
  Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all
  4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the
  read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean)
  on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128,
  whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6
  and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held)
  showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master
  IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to
  _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >=
  1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across
  the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)).
  To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict
  (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs,
  the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the
  DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.
expected:
  - shapes: layer=router *
    direction: faster
    why: "router PDL routed at m=16/32/128/256 in a20f8048: -0.2 to -0.45 us per router row (x1.03-1.10 gate)"
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "khalf routes on most runs under the pooled rule (b2 won every one of 8 recorded rounds at x1.025-1.054): keeps the ~-1.2 us judged pooler m=128 row"
  - shapes: "*"
    direction: flat
    why: every other cell unchanged
risk: low
depends_on: router-zero-pdl-overlap-m9-m256
priority: 1
kind: primary
controls_for: null
prior: "router-zero-pdl-overlap-m9-m256: same router code, which routed everywhere but lost the headline to a khalf gate miss in the same run; the khalf part is the pooled-IQR rule the khalf-gate-three-rounds-pooled adversary prescribed (that parent was held because its rule kept the per-round IQR test)"
review: required
status: queued
---

# router-pdl-plus-khalf-pooled-iqr-gate

## Change

Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all 4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean) on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128, whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6 and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held) showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >= 1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)). To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs, the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.

## Expected

- `layer=router *` faster: router PDL routed at m=16/32/128/256 in a20f8048: -0.2 to -0.45 us per router row (x1.03-1.10 gate)
- `layer=pooler load=verify_16 m=128` faster: khalf routes on most runs under the pooled rule (b2 won every one of 8 recorded rounds at x1.025-1.054): keeps the ~-1.2 us judged pooler m=128 row
- `*` flat: every other cell unchanged

## Risk

Headline effect ~+0.3-0.6% combined (router ~0.3% + khalf ~0.27% when it would otherwise miss): near the single-run readability line, so read the R-PDL and K PATH CHOSEN lines. The 1.2735 board run had khalf OFF, so a run with both ON can plausibly beat it; a check that beats the board gets a scored run.
