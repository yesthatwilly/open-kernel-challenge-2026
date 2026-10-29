---
problem: muse_attention_decode
slug: full-verify-paged-gather-gate-kernel
created: 2026-10-29T215944Z
base: master
mint_blob: "2eb03f76ceefa9bf81e28616ced3b33be3a70443"
best_at_mint: 1.634
area: full-layer verify attention (FA3 call at verify_16 / verify_32)
change: >
  Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, paged FP8
  KV, ctx ~80000) with the H100-validated gate kernel from takemyjob hopper/fa3gate 4b2a169
  (probes/fa3gate; research/h100-fa3gate-1029.md): a persistent paged-gather attention kernel, 3
  stages x 2 CTAs/SM = 64 KB in flight per SM, page indices prefetched a ring ahead, FP8 QK^T by
  wgmma straight from the swizzled cp.async ring, f16 PV, online softmax, grid-barrier distributed
  combine. Build it as a separate background-built load_inline extension; route per row (verify_16
  full, verify_32 full only; verify_1 and every sliding/decode row keep master) by an in-prepare
  interleaved race through kernel_fn (read-flushed, cold L2) against master's FA3 path: on only if
  within the problem tolerance on every draw (fp64 relL2 and the judge metric, which the node
  measured at 2.5-2.8e-2 vs bounds 0.083/0.085) and <= 0.95 vs master in every round; else master.
  Inputs fresh per call (page table, lengths); no caching of anything input-derived; no host sync
  in kernel_fn. Print an [fvg] summary (<= 1.5 KB) at the end of prepare: build s,
  regs/spill/smem/occupancy, per-row per-round ratios and errors, PATH CHOSEN.
expected:
  - shapes: layer=full load=verify_16*
    direction: faster
    why: node e2e 0.83-0.84 of master FA3 (v16)
  - shapes: layer=full load=verify_32*
    direction: faster
    why: node e2e 0.82 (v32); headline ~ 0.256 share x ln(1/0.85) ~ +4.2%
  - shapes: layer=sliding*
    direction: same
    why: not routed
  - shapes: layer=full load=verify_1*
    direction: same
    why: not routed (gate loses at v1, 1.42x)
risk: high
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "fa3-metadata-fold-preattn-full-verify: that removed FA3's prepare node and kept FA3; this replaces FA3's main kernel on v16/v32 full with a node-measured faster paged-gather kernel (memory parallelism 64 KB/SM vs FA3's ~24-32 KB)"
target: check
value_gain: 4.2
value_p: 0.4
value_ev: 1.68
value_cost: 500k tokens, 1 judge run
evidence: >
  H100 gpunode06 fa3gate (takemyjob hopper/fa3gate 4b2a169; research/h100-fa3gate-1029.md), 2 seeds,
  5-7 interleaved rounds, cold L2: e2e 272 vs master 331 us (v32, 0.82), 147-151 vs 176-181 (v16,
  0.83-0.84); error vs fp32 1.67e-3 (FA3 6.3e-3..3.9e-2), judge metric 2.46-2.79e-2 vs bounds
  0.076-0.085; attention triage shares verify_16 full 12.4%, verify_32 full 13.2% (board 4e386647)
review: required
status: queued
---

# full-verify-paged-gather-gate-kernel

## Change

Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, paged FP8 KV, ctx ~80000) with the H100-validated gate kernel from takemyjob hopper/fa3gate 4b2a169 (probes/fa3gate; research/h100-fa3gate-1029.md): a persistent paged-gather attention kernel, 3 stages x 2 CTAs/SM = 64 KB in flight per SM, page indices prefetched a ring ahead, FP8 QK^T by wgmma straight from the swizzled cp.async ring, f16 PV, online softmax, grid-barrier distributed combine. Build it as a separate background-built load_inline extension; route per row (verify_16 full, verify_32 full only; verify_1 and every sliding/decode row keep master) by an in-prepare interleaved race through kernel_fn (read-flushed, cold L2) against master's FA3 path: on only if within the problem tolerance on every draw (fp64 relL2 and the judge metric, which the node measured at 2.5-2.8e-2 vs bounds 0.083/0.085) and <= 0.95 vs master in every round; else master. Inputs fresh per call (page table, lengths); no caching of anything input-derived; no host sync in kernel_fn. Print an [fvg] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill/smem/occupancy, per-row per-round ratios and errors, PATH CHOSEN.

## Expected

- `layer=full load=verify_16*` faster: node e2e 0.83-0.84 of master FA3 (v16)
- `layer=full load=verify_32*` faster: node e2e 0.82 (v32); headline ~ 0.256 share x ln(1/0.85) ~ +4.2%
- `layer=sliding*` same: not routed
- `layer=full load=verify_1*` same: not routed (gate loses at v1, 1.42x)

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 gpunode06 fa3gate (takemyjob hopper/fa3gate 4b2a169; research/h100-fa3gate-1029.md), 2 seeds, 5-7 interleaved rounds, cold L2: e2e 272 vs master 331 us (v32, 0.82), 147-151 vs 176-181 (v16, 0.83-0.84); error vs fp32 1.67e-3 (FA3 6.3e-3..3.9e-2), judge metric 2.46-2.79e-2 vs bounds 0.076-0.085; attention triage shares verify_16 full 12.4%, verify_32 full 13.2% (board 4e386647)
- **Gain if it works:** 4.2% headline; **p** 0.4; **EV** 1.68% (bar: gain >= 2.5%, EV >= 1.5%).
- **Cost:** 500k tokens, 1 judge run.

## Risk

H100 -> H200: the kernel is math/latency-bound at the ceiling on the H100 (math-only 233 us at v32 = memory-ceiling time); per-SM structure transfers, absolute TB/s does not; projected H200 e2e 0.82-0.87 (untested; the H200 is held). Every earlier custom full-verify loop lost to FA3 on the judge (CLAIMS); this one is the first measured faster on the same SM. mma.sync e4m3 is emulated on sm_90a (the kernel uses wgmma for QK).
