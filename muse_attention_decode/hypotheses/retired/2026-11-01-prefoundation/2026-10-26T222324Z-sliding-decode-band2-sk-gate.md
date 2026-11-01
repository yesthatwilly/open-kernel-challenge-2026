---
problem: muse_attention_decode
slug: sliding-decode-band2-sk-gate
created: 2026-10-26T222324Z
base: master
area: sliding-layer attention (decode B=2..16, stream-K window route gate)
change: >
  Stacked on sk-full-line-gather (build on its code; claim restacks it). master's _sd_setup_body
  gates only band 3 (B=17..32) for the [pre-attention + stream-K window TOK=1] pair: its loop is
  `for band, batch in ((3, 32),):` and its comment says B=2..16 keeps master. At B=16 the pair
  tied the incumbent Triton _sliding_fused_kernel (pair 16.2 vs Triton 16.3 us net, sk kernel
  alone 13.1 us, per the sliding-decode-sk-s3-fused-preattn-rope notes), so with a faster sk
  gather it may now clear the bar. CHANGE (host side only): extend that loop to ((2, 16), (3,
  32)). _sd_gate_band is already generic: it times every variant x grid spec (6 / 7 x p2..2sm) and
  re-times the best pair head to head against the band's real incumbent (Triton here, so it needs
  >= 1.05x); band 2 then gets the same _sd_replay_check two-state replay and PATH CHOSEN line as
  band 3, and a route lands in the frozen _SD_ROUTE table, which _sd_gate already looks up by
  _ps_band(batch) for 2 <= B <= 32. Band 3 and every other path unchanged.
expected:
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: "only if the pair beats Triton by >= 1.05x after the sk remap (tied before: 16.2 vs 16.3 us net); 15.26 us shape at 0.31%/us"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: band 3 decided as in the parent
  - shapes: "*layer=full *"
    direction: flat
    why: as the parent
risk: low
depends_on: sk-full-line-gather
priority: 4
kind: primary
controls_for: null
prior: "restack-sliding-decode-sk-window-on-cluster-master: it gated only band 3 because band 2's pair tied Triton (16.2 vs 16.3 us net); this gates band 2 only on top of the whole-line remap, under the existing 1.05x-vs-Triton rule"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.5% (p~0.4), cheap stacker at most"
---

# sliding-decode-band2-sk-gate

## Change

Stacked on sk-full-line-gather (build on its code; claim restacks it). master's _sd_setup_body gates only band 3 (B=17..32) for the [pre-attention + stream-K window TOK=1] pair: its loop is `for band, batch in ((3, 32),):` and its comment says B=2..16 keeps master. At B=16 the pair tied the incumbent Triton _sliding_fused_kernel (pair 16.2 vs Triton 16.3 us net, sk kernel alone 13.1 us, per the sliding-decode-sk-s3-fused-preattn-rope notes), so with a faster sk gather it may now clear the bar. CHANGE (host side only): extend that loop to ((2, 16), (3, 32)). _sd_gate_band is already generic: it times every variant x grid spec (6 / 7 x p2..2sm) and re-times the best pair head to head against the band's real incumbent (Triton here, so it needs >= 1.05x); band 2 then gets the same _sd_replay_check two-state replay and PATH CHOSEN line as band 3, and a route lands in the frozen _SD_ROUTE table, which _sd_gate already looks up by _ps_band(batch) for 2 <= B <= 32. Band 3 and every other path unchanged.

## Expected

- `*layer=sliding load=decode_16 *` faster: only if the pair beats Triton by >= 1.05x after the sk remap (tied before: 16.2 vs 16.3 us net); 15.26 us shape at 0.31%/us
- `*layer=sliding load=decode_32 *` flat: band 3 decided as in the parent
- `*layer=full *` flat: as the parent

## Risk

Dispatch only after sk-full-line-gather harvests (gain or neutral) and only if its PSEUDO TOK=1 rows did not slow down; the 1.05x Triton bar is strict, so the likeliest result is 'master kept' with a printed pair-vs-Triton ratio at B=16. The queued sliding-decode-sk-s3-fused-preattn-rope removes the pair's second node and is the bigger lever on the same band; this only asks whether the remap alone tips it.
