---
problem: muse_prefill_layer
slug: experts-rega-pdl-chain-regate-iqr-slack
created: 2026-10-26T224004Z
base: master
area: "launch overhead (PDL chain _norm_gather_kernel -> rega w13 -> rega w2: gate rule only)"
change: >
  Built on master (blob f4751fc0, the same base experts-rega-pdl-chain-gather-w13-w2 was built
  on). Apply done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch unchanged (it adds the
  muse_prefill_rega_pdl module: w13 programmatically chained to _norm_gather_kernel and w2 to w13,
  each producer bulk-copying its first STAGES weight records before griddepcontrol.wait; canary,
  bit-exact replays and a per-class gate; master's kernels untouched). Change only the gate rule
  in _rega_pdl_class. That run measured pdl / master 0.988 and 0.991 (low) and 0.988 and 0.990
  (high), bit-identical, but its IQR veto (candidate IQR <= 1.5 x max(master IQR, 1 us) in both
  rounds) failed one round per class (low IQR 2.2 vs 1.3 us, high 2.9 vs 1.2 us), so both stayed
  OFF, while ft passed at 1.001 / 0.996 and went ON for no gain. New rule: ON when the ratio is <=
  0.995 in both rounds (a win, not a tie: LESSONS 169) and the candidate IQR <= 1.5 x max(master
  IQR, 1 us) + 1% of master's median (the slack mc2's audit used). Expected outcome: low and high
  ON, ft OFF. Print the rule's inputs per round and PATH CHOSEN as the patch does.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "the low-class gate measured the routed expert block 476 -> 471 us: ~-5 us per turn_1 shape (~0.6% sliding)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class 727 -> 719 us: ~-8 us per turn_4 shape (~0.35% sliding)"
  - shapes: "*load=first_turn*"
    direction: flat
    why: ft measured 0.996-1.001 and stays OFF under the <= 0.995 rule
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "experts-rega-pdl-chain-gather-w13-w2: same module, kernels and chain byte for byte; only the gate changes (1% of master's median added as IQR slack, <= 0.995 required in both rounds), which is what kept low and high OFF at measured ratios 0.988-0.991"
review: required
status: queued
---

# experts-rega-pdl-chain-regate-iqr-slack

## Change

Built on master (blob f4751fc0, the same base experts-rega-pdl-chain-gather-w13-w2 was built on). Apply done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch unchanged (it adds the muse_prefill_rega_pdl module: w13 programmatically chained to _norm_gather_kernel and w2 to w13, each producer bulk-copying its first STAGES weight records before griddepcontrol.wait; canary, bit-exact replays and a per-class gate; master's kernels untouched). Change only the gate rule in _rega_pdl_class. That run measured pdl / master 0.988 and 0.991 (low) and 0.988 and 0.990 (high), bit-identical, but its IQR veto (candidate IQR <= 1.5 x max(master IQR, 1 us) in both rounds) failed one round per class (low IQR 2.2 vs 1.3 us, high 2.9 vs 1.2 us), so both stayed OFF, while ft passed at 1.001 / 0.996 and went ON for no gain. New rule: ON when the ratio is <= 0.995 in both rounds (a win, not a tie: LESSONS 169) and the candidate IQR <= 1.5 x max(master IQR, 1 us) + 1% of master's median (the slack mc2's audit used). Expected outcome: low and high ON, ft OFF. Print the rule's inputs per round and PATH CHOSEN as the patch does.

## Expected

- `*load=turn_1*` faster: the low-class gate measured the routed expert block 476 -> 471 us: ~-5 us per turn_1 shape (~0.6% sliding)
- `*load=turn_4*` faster: high class 727 -> 719 us: ~-8 us per turn_4 shape (~0.35% sliding)
- `*load=first_turn*` flat: ft measured 0.996-1.001 and stays OFF under the <= 0.995 rule

## Risk

Size ~+0.3% headline: lands only stacked (front-routing-pdl-launch-diet-stack carries it). The patch is 543 lines but applies as is; the edit is the gate rule.
