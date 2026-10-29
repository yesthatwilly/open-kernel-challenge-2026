---
problem: muse_moe_w4a16
slug: engine-a-a4b-a5b-encodings-race
created: 2026-10-29T215527Z
base: master
mint_blob: "37ea90fc306c0ed8fb3d49094bf5f4acf6202307"
best_at_mint: 1.9748
area: Engine A dequant (dq2) in ea4x/v4x/ea6 kernels at T=1..32 and the T=128/256 cold half
change: >
  Add two lossless dequant arms to moe's Engine A kernels (ea6 K1/K2 at T=1; ea4x / v4x-family w13
  and w2 at T=8/16/32; the cold-half ea4x at T=128/256), ported from takemyjob hopper/ea-enc-1029
  d6f55c5 (probe njenc; research/h100-ea-enc-1029.md): E4 = A4b code re-encoding of the 4-bit
  codes in a prepare-time permuted copy of the bank (nibble->bf16x2 9 ops/word vs 11), E5 = E4 +
  A5b packed 7-bit (b-8) scale fields (one LOP3 per step; requires every scale byte in
  [0x08,0x7E], scanned in prepare). Both are value-identical to master's dq2 (bit-exact). Compile
  them as template variants of the existing kernels in a separate background-built extension (or
  as template args in the existing ones if the build budget allows), and route PER CLASS (T=1, 8,
  16, 32, 128-cold, 256-cold) by an in-prepare interleaved race through kernel_fn (read-flushed,
  like the existing class races): pick E4 or E5 only if torch.equal to master on every draw, <=
  0.98 vs master in every round, and replay-exact; else master. Expected picks from the node: E4
  at T<=32 and T=1, E5 at the cold half. Keep master's routing, fork, hot half (wh5ds) and PDL
  unchanged. Print an [ea-enc] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill per
  arm, scale scan, per-class per-round ratios E4/m and E5/m, exactness, PATH CHOSEN. Size: moe is
  993,167 B (cap 1,000,000): trim dead diagnostics first; report wc -c.
expected:
  - shapes: load=decode_16
    direction: faster
    why: node consume-only B/E4 T16 w13 1.067-1.090, w2 1.05-1.07
  - shapes: load=decode_32
    direction: faster
    why: T32 w13 1.079-1.086
  - shapes: load=decode_1
    direction: faster
    why: ea6 K1 consume 1.076-1.097
  - shapes: load=verify_*
    direction: faster
    why: cold half E5 1.05-1.07 consume; T8 1.05-1.08
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "cuda-nvfp4-swapab-mma-persistent-small-m-childcheck: the mma.sync small-m lineage Engine A grew from; this changes only the dequant's instruction form (lossless code/scale re-encoding, bit-exact), not the schedule, ring, prefetch or engine that the matched refuted claims tried"
target: check
value_gain: 1.5
value_p: 0.4
value_ev: 0.6
value_cost: 450k tokens, 1 judge run
evidence: >
  H100 gpunode06 GPU 3 (user-approved; runs/20261029-144741-gpunode06), 2 replicates x 10 rounds,
  consume-only: B/E4 T8 1.05-1.08, T16 w13 1.067-1.090, T32 w13 1.079-1.086, cold 1.04-1.05, ea6
  K1 1.076-1.097; B/E5 cold 1.05-1.07; exact at every draw; SASS -9..-12% issued/k16
  (research/h100-ea-enc-1029.md)
user_requested: true
review: required
status: queued
---

# engine-a-a4b-a5b-encodings-race

## Change

Add two lossless dequant arms to moe's Engine A kernels (ea6 K1/K2 at T=1; ea4x / v4x-family w13 and w2 at T=8/16/32; the cold-half ea4x at T=128/256), ported from takemyjob hopper/ea-enc-1029 d6f55c5 (probe njenc; research/h100-ea-enc-1029.md): E4 = A4b code re-encoding of the 4-bit codes in a prepare-time permuted copy of the bank (nibble->bf16x2 9 ops/word vs 11), E5 = E4 + A5b packed 7-bit (b-8) scale fields (one LOP3 per step; requires every scale byte in [0x08,0x7E], scanned in prepare). Both are value-identical to master's dq2 (bit-exact). Compile them as template variants of the existing kernels in a separate background-built extension (or as template args in the existing ones if the build budget allows), and route PER CLASS (T=1, 8, 16, 32, 128-cold, 256-cold) by an in-prepare interleaved race through kernel_fn (read-flushed, like the existing class races): pick E4 or E5 only if torch.equal to master on every draw, <= 0.98 vs master in every round, and replay-exact; else master. Expected picks from the node: E4 at T<=32 and T=1, E5 at the cold half. Keep master's routing, fork, hot half (wh5ds) and PDL unchanged. Print an [ea-enc] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill per arm, scale scan, per-class per-round ratios E4/m and E5/m, exactness, PATH CHOSEN. Size: moe is 993,167 B (cap 1,000,000): trim dead diagnostics first; report wc -c.

## Expected

- `load=decode_16` faster: node consume-only B/E4 T16 w13 1.067-1.090, w2 1.05-1.07
- `load=decode_32` faster: T32 w13 1.079-1.086
- `load=decode_1` faster: ea6 K1 consume 1.076-1.097
- `load=verify_*` faster: cold half E5 1.05-1.07 consume; T8 1.05-1.08

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 gpunode06 GPU 3 (user-approved; runs/20261029-144741-gpunode06), 2 replicates x 10 rounds, consume-only: B/E4 T8 1.05-1.08, T16 w13 1.067-1.090, T32 w13 1.079-1.086, cold 1.04-1.05, ea6 K1 1.076-1.097; B/E5 cold 1.05-1.07; exact at every draw; SASS -9..-12% issued/k16 (research/h100-ea-enc-1029.md)
- **Gain if it works:** 1.5% headline; **p** 0.4; **EV** 0.6% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.5% < 2.5%; EV 0.6% < 1.5%), minted because the user described it.
- **Cost:** 450k tokens, 1 judge run.

## Risk

On the H100 every Engine A kernel is at the HBM wall in normal mode (ratios ~1.00), so only the judge (H200, estimated consumer-bound: consumer-alone 0.91-0.97x of the HBM floor) shows whether the 4-10% consume gain surfaces. The race decides per class; a class that reads level stays on master. Size is the binding constraint (993 KB). The matched refuted claims are other mechanisms (Engine S cold path, ea6 w2 prefetch, ea6 ring depth); t1-deep-ring's reopen ('the per-record consume rate is raised first') is exactly what this raises.
