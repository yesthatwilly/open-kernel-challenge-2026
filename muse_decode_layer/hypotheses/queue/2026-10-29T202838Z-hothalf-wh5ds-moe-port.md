---
problem: muse_decode_layer
slug: hothalf-wh5ds-moe-port
created: 2026-10-29T202838Z
base: master
mint_blob: "315119e26cb18e755adf23c3336af2c15c72aafc"
best_at_mint: 1.6904
area: moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256
change: >
  Port moe's landed wh5ds hot half back to decode: take the module from
  muse_moe_w4a16/submission.py on master (landed hothalf-wh-a5b-port, scored a5318a1f: arms D0=0,
  a3=1, wh=7, wh5ds=14 = A3 + A5b code re-encoding / 7-bit packed scales + per-stage descriptor +
  split commit on the wh exact-N/ring kernel; isolated-pair route gate <= 0.97 vs master and D0
  every round, wh5ds also <= 0.98 vs wh; production no-loss gate) and decode's own routed wh
  module (branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wh routed T=128
  and T=256). Replace decode's wh-only arm set with D0/a3/wh/wh5ds, route per T class (128 and
  256) by the isolated pair, plus a no-loss gate on decode's layer timing at each class (wh/m
  layer ratio <= 1.00 every round, as decode's 44f30364 FORK lines measured). Keep decode's fork
  setting (CAMPAIGN_HCF False), routing, cold half and PDL unchanged. Prints as in 44f30364
  ([hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK/PATH CHOSEN, <= 1.5 KB at the end of prepare). Size:
  decode is 955,413 B on master; 44f30364's wh module brought it to 984,021 B; the extra wh5ds arm
  must still fit under 1,000,000 B (trim a3nx and dead diag text).
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "moe judge: wh5ds pair 92.8 vs wh 98.9 vs master 115.1 us; decode 44f30364 wh verify_32 sliding hot pair 153.5 -> 122.8 us; H100 hh3 wh5ds vs wh x0.986 forked"
  - shapes: load=verify_16*
    direction: faster
    why: decode T=128 wh pair .90 (44f30364)
  - shapes: load=decode_*
    direction: same
    why: T<=32 do not use the hot half
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "hothalf-a3-exactn-ring-wh: same routed wh plus the wh5ds arm (A5b+D+SC) that landed on moe, and a no-loss layer gate"
target: check
value_gain: 1.53
value_p: 0.5
value_ev: 0.77
value_cost: 300k tokens, 1 judge run
evidence: >
  moe landed a5318a1f 1.9748 (check 54b70249 1.9827: verify_32 291.1 -> 274.4 us, pair wh5ds 92.8
  vs master 115.1); decode 44f30364 wh routed T=128/256, verify_32 sliding hot pair 153.5 -> 122.8
  us, row 427.7 -> 403.2; H100 hh3: decode +1.32% (wh) -> +1.53% (wh5ds)
user_requested: true
review: required
status: queued
---

# hothalf-wh5ds-moe-port

## Change

Port moe's landed wh5ds hot half back to decode: take the module from muse_moe_w4a16/submission.py on master (landed hothalf-wh-a5b-port, scored a5318a1f: arms D0=0, a3=1, wh=7, wh5ds=14 = A3 + A5b code re-encoding / 7-bit packed scales + per-stage descriptor + split commit on the wh exact-N/ring kernel; isolated-pair route gate <= 0.97 vs master and D0 every round, wh5ds also <= 0.98 vs wh; production no-loss gate) and decode's own routed wh module (branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wh routed T=128 and T=256). Replace decode's wh-only arm set with D0/a3/wh/wh5ds, route per T class (128 and 256) by the isolated pair, plus a no-loss gate on decode's layer timing at each class (wh/m layer ratio <= 1.00 every round, as decode's 44f30364 FORK lines measured). Keep decode's fork setting (CAMPAIGN_HCF False), routing, cold half and PDL unchanged. Prints as in 44f30364 ([hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK/PATH CHOSEN, <= 1.5 KB at the end of prepare). Size: decode is 955,413 B on master; 44f30364's wh module brought it to 984,021 B; the extra wh5ds arm must still fit under 1,000,000 B (trim a3nx and dead diag text).

## Expected

- `load=verify_32*` faster: moe judge: wh5ds pair 92.8 vs wh 98.9 vs master 115.1 us; decode 44f30364 wh verify_32 sliding hot pair 153.5 -> 122.8 us; H100 hh3 wh5ds vs wh x0.986 forked
- `load=verify_16*` faster: decode T=128 wh pair .90 (44f30364)
- `load=decode_*` same: T<=32 do not use the hot half

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** moe landed a5318a1f 1.9748 (check 54b70249 1.9827: verify_32 291.1 -> 274.4 us, pair wh5ds 92.8 vs master 115.1); decode 44f30364 wh routed T=128/256, verify_32 sliding hot pair 153.5 -> 122.8 us, row 427.7 -> 403.2; H100 hh3: decode +1.32% (wh) -> +1.53% (wh5ds)
- **Gain if it works:** 1.53% headline; **p** 0.5; **EV** 0.77% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.53% < 2.5%; EV 0.77% < 1.5%), minted because the user described it.
- **Cost:** 300k tokens, 1 judge run.

## Risk

Decode's wh check 44f30364 routed and moved the verify_32 sliding hot pair x0.80, but the headline read flat (1.6794 vs 1.6904) on a noisy draw; moe's wh5ds then landed (54b70249 1.9827, scored a5318a1f 1.9748). Read this on the [hh-wh] lines and the wg_kernel<G,14> profile rows first, headline last.
