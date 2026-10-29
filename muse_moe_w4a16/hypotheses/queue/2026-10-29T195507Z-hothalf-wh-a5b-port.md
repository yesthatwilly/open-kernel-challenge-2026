---
problem: muse_moe_w4a16
slug: hothalf-wh-a5b-port
created: 2026-10-29T195507Z
base: master
mint_blob: "691018b5844d1a985d98f59021313b0c1a3ef3b7"
best_at_mint: 1.9521
area: moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256
change: >
  Port decode's routed wh hot half (module muse_decode_wg_wh from branch
  hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wg_kernel<G,V> with D0 control
  V=0 and wh V=7 = A3 dequant on a word-permuted whole bank + exact-N tiles + R4 ring wait<2> for
  N>=48; background-thread build; isolated-pair route gate <= 0.97 vs master and D0 every round,
  exact, 16-replay) into moe's hot half (muse_moe_w4a16/submission.py wg_kernel at :867-1700,
  dispatched at T=128/256 by the hybrid). Add one more arm wh5ds = wh + A5b+D+SC from takemyjob
  hopper/hh3-1029 6a33dbf (A4b code re-encoding of the same permuted bank, 7-bit packed (b-8)
  scale fields, per-stage wgmma descriptor, split commit; landed on prefill as
  experts-rega-a5b-desc-splitcommit, scored 9f934846), raced as a candidate against wh: route
  wh5ds only if it also beats wh by <= 0.98 every round, else wh. Scale bytes scanned in
  [0x08,0x7E] (else master); banks from weights only. Keep moe's fork, routing, cold half and PDL
  unchanged. Print the [hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK lines (<= 1.5 KB) at the end of
  prepare. Size: moe is 964,183 B; the cap is 1,000,000 B (build.py strips comments above 990 KB):
  measure and trim dead diag text first if needed.
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "H100 hh3 moe judge draw forked: wh5ds x0.982 vs wh, wh ~-15.8 us; moe +0.92% (wh) -> +1.29% (wh5ds)"
  - shapes: load=verify_16*
    direction: faster
    why: "T=128 hot half: wh isolated pair .90 on the judge (decode 44f30364)"
  - shapes: load=decode_*
    direction: same
    why: T<=32 do not use the hot half
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: none
target: check
value_gain: 1.29
value_p: 0.5
value_ev: 0.65
value_cost: 350k tokens, 1 judge run
evidence: >
  decode judge 44f30364: wh routed T=128/256, pair .898-.902 / .838-.842 vs master, exact 16/16,
  profile hot pair x0.80 at verify_32 sliding; H100 hh2/hh3 (research/h100-hothalf2-1029.md,
  h100-hothalf3-1029.md): moe forked every wh variant ~-15.8 us (~+0.9%), wh5ds x0.982 vs wh on
  the moe draw -> +1.29% moe headline; A5b landed on prefill (9f934846 1.5721)
user_requested: true
review: required
status: queued
---

# hothalf-wh-a5b-port

## Change

Port decode's routed wh hot half (module muse_decode_wg_wh from branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wg_kernel<G,V> with D0 control V=0 and wh V=7 = A3 dequant on a word-permuted whole bank + exact-N tiles + R4 ring wait<2> for N>=48; background-thread build; isolated-pair route gate <= 0.97 vs master and D0 every round, exact, 16-replay) into moe's hot half (muse_moe_w4a16/submission.py wg_kernel at :867-1700, dispatched at T=128/256 by the hybrid). Add one more arm wh5ds = wh + A5b+D+SC from takemyjob hopper/hh3-1029 6a33dbf (A4b code re-encoding of the same permuted bank, 7-bit packed (b-8) scale fields, per-stage wgmma descriptor, split commit; landed on prefill as experts-rega-a5b-desc-splitcommit, scored 9f934846), raced as a candidate against wh: route wh5ds only if it also beats wh by <= 0.98 every round, else wh. Scale bytes scanned in [0x08,0x7E] (else master); banks from weights only. Keep moe's fork, routing, cold half and PDL unchanged. Print the [hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK lines (<= 1.5 KB) at the end of prepare. Size: moe is 964,183 B; the cap is 1,000,000 B (build.py strips comments above 990 KB): measure and trim dead diag text first if needed.

## Expected

- `load=verify_32*` faster: H100 hh3 moe judge draw forked: wh5ds x0.982 vs wh, wh ~-15.8 us; moe +0.92% (wh) -> +1.29% (wh5ds)
- `load=verify_16*` faster: T=128 hot half: wh isolated pair .90 on the judge (decode 44f30364)
- `load=decode_*` same: T<=32 do not use the hot half

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** decode judge 44f30364: wh routed T=128/256, pair .898-.902 / .838-.842 vs master, exact 16/16, profile hot pair x0.80 at verify_32 sliding; H100 hh2/hh3 (research/h100-hothalf2-1029.md, h100-hothalf3-1029.md): moe forked every wh variant ~-15.8 us (~+0.9%), wh5ds x0.982 vs wh on the moe draw -> +1.29% moe headline; A5b landed on prefill (9f934846 1.5721)
- **Gain if it works:** 1.29% headline; **p** 0.5; **EV** 0.65% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.29% < 2.5%; EV 0.65% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge run.

## Risk

Decode's wh check 44f30364 read flat on the headline (1.6794 vs 1.6904) but its verify_32 sliding hot pair fell 153.5 -> 122.8 us (x0.80) and the row 427.7 -> 403.2 us; a single run cannot resolve ~1%. Read this one on the profile's wg_kernel<.,7|8> rows and the [hh-wh] ROUTE/FORK lines, not the headline alone.
