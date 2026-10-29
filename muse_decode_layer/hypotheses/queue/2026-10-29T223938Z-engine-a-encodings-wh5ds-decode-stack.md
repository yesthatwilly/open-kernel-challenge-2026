---
problem: muse_decode_layer
slug: engine-a-encodings-wh5ds-decode-stack
created: 2026-10-29T223938Z
base: master
mint_blob: "315119e26cb18e755adf23c3336af2c15c72aafc"
best_at_mint: 1.6904
area: "decode expert block: Engine A dequant (ea4x/v4x/ea6) at T=1..32 and the cold half, plus the wh5ds hot half"
change: >
  Port moe's two landed expert-block changes into decode as one stack, each routed by its own
  in-prepare exact race: (1) Engine A A4b/A5b encodings exactly as landed in
  muse_moe_w4a16/submission.py (engine-a-a4b-a5b-encodings-race, scored 24f183ff 2.0201): E4/E5
  arms on the v4x twins and v6 K1/K2 (T=1 K1 x K2 independent), per-T race through kernel_fn
  (decode's T classes: 1, 8, 16, 32 and the cold half at 128/256), pick only if exact (h, y,
  output torch.equal, err 0, tickets 0) and <= 0.98 vs master every round; E5 only if every scale
  byte is in [0x08,0x7E]; (2) the wh5ds hot half from decode's own hothalf-wh5ds-moe-port branch
  (check 35cc220c: routed both classes, layer veto .933-.952) with its isolated-pair pick + layer
  veto. Also pin decode's T=16 ea4x pooled race to the board's draw 's' (v4cxs) as a deterministic
  route (it flipped to 'd' in 44f30364 and 35cc220c, ~-0.38% headline). Keep everything else (fork
  guard, routing, PDL) unchanged. Size: decode is 955,413 B on master; use build.py strip_comments
  (AST-identical, CUDA strings byte-identical) if needed; report wc -c. Prints: [ea-enc] and
  [hh-wh] summaries as in the moe/decode builds, the [ea4x-dl] route line, all at the end of
  prepare.
expected:
  - shapes: load=decode_*
    direction: faster
    why: "moe judge 702f1004: Engine A E4 T1 0.967, T8 0.960, T16 0.953, T32 0.951 per class"
  - shapes: load=verify_16*
    direction: faster
    why: cold half E5 T128 0.955 + wh5ds T=128 layer .933-.940
  - shapes: load=verify_32*
    direction: faster
    why: cold half E4 T256 0.974 (iso 0.960) + wh5ds T=256 layer .936-.952
  - shapes: load=verify_1*
    direction: faster
    why: T=8 class E4 0.960
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "hothalf-wh5ds-moe-port: that ported only the hot half (flat headline, +0.38%); this adds the Engine A encodings that moved moe +4.55% on the judge, across every decode T class, plus the T=16 pin"
target: check
value_gain: 2.8
value_p: 0.55
value_ev: 1.54
value_cost: 500k tokens, 1 judge run
evidence: >
  moe judge 702f1004 2.0646 / scored 24f183ff 2.0201: Engine A encodings E/m 0.951-0.974 per T
  class, exact; decode 35cc220c 1.6968: wh5ds routed both classes, layer gate .933-.952; decode
  board 7cc034f9 anatomy: expert block 57 us (S d1/v1), 98 us (S d16/d32), 253 us (S v16/v32) =
  48-68% of the rows
review: required
status: queued
---

# engine-a-encodings-wh5ds-decode-stack

## Change

Port moe's two landed expert-block changes into decode as one stack, each routed by its own in-prepare exact race: (1) Engine A A4b/A5b encodings exactly as landed in muse_moe_w4a16/submission.py (engine-a-a4b-a5b-encodings-race, scored 24f183ff 2.0201): E4/E5 arms on the v4x twins and v6 K1/K2 (T=1 K1 x K2 independent), per-T race through kernel_fn (decode's T classes: 1, 8, 16, 32 and the cold half at 128/256), pick only if exact (h, y, output torch.equal, err 0, tickets 0) and <= 0.98 vs master every round; E5 only if every scale byte is in [0x08,0x7E]; (2) the wh5ds hot half from decode's own hothalf-wh5ds-moe-port branch (check 35cc220c: routed both classes, layer veto .933-.952) with its isolated-pair pick + layer veto. Also pin decode's T=16 ea4x pooled race to the board's draw 's' (v4cxs) as a deterministic route (it flipped to 'd' in 44f30364 and 35cc220c, ~-0.38% headline). Keep everything else (fork guard, routing, PDL) unchanged. Size: decode is 955,413 B on master; use build.py strip_comments (AST-identical, CUDA strings byte-identical) if needed; report wc -c. Prints: [ea-enc] and [hh-wh] summaries as in the moe/decode builds, the [ea4x-dl] route line, all at the end of prepare.

## Expected

- `load=decode_*` faster: moe judge 702f1004: Engine A E4 T1 0.967, T8 0.960, T16 0.953, T32 0.951 per class
- `load=verify_16*` faster: cold half E5 T128 0.955 + wh5ds T=128 layer .933-.940
- `load=verify_32*` faster: cold half E4 T256 0.974 (iso 0.960) + wh5ds T=256 layer .936-.952
- `load=verify_1*` faster: T=8 class E4 0.960

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** moe judge 702f1004 2.0646 / scored 24f183ff 2.0201: Engine A encodings E/m 0.951-0.974 per T class, exact; decode 35cc220c 1.6968: wh5ds routed both classes, layer gate .933-.952; decode board 7cc034f9 anatomy: expert block 57 us (S d1/v1), 98 us (S d16/d32), 253 us (S v16/v32) = 48-68% of the rows
- **Gain if it works:** 2.8% headline; **p** 0.55; **EV** 1.54% (bar: gain >= 2.5%, EV >= 1.5%).
- **Cost:** 500k tokens, 1 judge run.

## Risk

The expert block is 48-68% of every sliding decode row (1749Z anatomy); moe's per-class E/m 0.95-0.97 on the judge applied to that share is ~+2-3% decode, plus the wh5ds rider (~+0.4-0.8%) and the T=16 pin (~+0.38% recovered vs the last two runs). The user, 22:40Z: stay focused on the expert block.
