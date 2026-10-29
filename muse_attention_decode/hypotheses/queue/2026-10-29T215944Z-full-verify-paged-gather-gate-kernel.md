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
review: proceed
status: queued
review_at: 2026-10-29T220405Z
review_companions: null
review_best: 1.6339575246416982
review_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
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

## Adversarial review

Reviewed 2026-10-29T220405Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed: E passes (fresh at 1.6340 / blob 2eb03f76; this is the judge check that CLAIMS.md:86 [lead] names as its reopen), and the in-prepare race with a <=0.95 bar caps the downside at the cost of one check. Expect the H100 edge to shrink on the H200: realistic e2e about 0.86-0.92 and a 2-3.5% headline, not 4.2%. The build has to remove a grid-barrier hang risk and gate on page_size/kv_scale.

### Justification
The mechanism holds up on this SM. FA3's full-verify main loop is limited by memory parallelism per SM (CLAIMS.md:84: one CTA/SM, 24-32 KB in flight, 2.04 TB/s, DRAM 62.5%). The gate keeps 64 KB/SM in flight, and in the same process with interleaved cold-L2 rounds over 2 seeds its mainloop beat FA3's main kernel 1.22-1.24x (259 vs 322 us at v32, 137-139 vs 167-171 at v16). End to end it was 0.82 / 0.83-0.84 of master's call, with tolerance held and 23x more accurate than FA3 vs fp32. It differs from the earlier custom full-verify loops that lost on the judge in four measured ways (s3-s7, FINDINGS:177-189, CLAIMS.md:63; s6's DIAG was 179/367 us vs FA3 148.6/285.9). (1) FP8 QK uses wgmma. mma.sync e4m3 is emulated on sm_90a (F2FP unpack + 2 HMMA; v1 of the lineage ran 467 us). (2) Lean load issue, about 140 fewer instr/warp/tile and no spills (v4: 292->258 us). (3) The combine is distributed after a grid barrier and costs 3-4 us. The ticket combine re-read partials already evicted from L2 by the KV stream and cost +47-84 us on the critical path. (4) It was raced against master's actual FA3 call on the same SM, which no earlier loop was. So it is not the same mechanism again, and prior is set correctly: the metadata fold kept FA3, this replaces it. WEAKEST LINK, the H100->H200 transfer. On the H100 the gate is effectively math-bound: math-only 233 us (v32) / 119-121 (v16) equals the loads-only time, and mainloop = math + about 26 us / 18 us of issue-slot and barrier contention from load-issue instructions. Both terms are per-SM and do not shrink on the H200 (same GH100 SM count and clocks). The H200's 4.8 vs 3.35 TB/s helps FA3, which is memory-parallelism-bound but still latency/HBM-sensitive: its main kernel already gains 1.13-1.15x (322->285.9, 170.9->148.6). It barely helps the gate. Estimate: gate mainloop on the H200 about 240-259 (v32) / 125-139 (v16), plus about 11-13 us combine/launch, against master's call of about 295 / 158. That gives e2e 0.86-0.92 (v32) and 0.87-0.95 (v16). v16 can miss the 0.95 bar. The H100's 0.82 is therefore an upper bound on the gain, not the expected value. Rough headline at r=0.875 on both rows (full verify about 0.79 / 0.74 of those loads, inferred from the 13.2% / 12.4% shares): about +1.7% (v32) + 1.5% (v16) = about 3.2%. Pessimistic, with only v32 routed at 0.92: about 1.3%. p(>=2.5% headline) is nearer 0.35 than 0.4, putting EV at about 1.1-1.4. That is marginal against the 1.5 bar, but the race makes the loss case free apart from the check. The pre-registered fa3gate rule (mainloop >= 2.6 TB/s) was NOT met (2.52). Proceeding rests on CLAIMS.md:86, after WS was closed (fa3ws/fa3math, 0.78-0.83x of the gate's math). Not a defect, but the build log must make the H200 reading interpretable (see open_questions). The expected shapes are consistent: only v16/v32 full move, verify_1 is excluded (1.42x loss, latency-bound combine), and sliding is untouched.

### Confounders on the current kernel
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup (+ ab-tune-batch-gate, patches in done/gain and done/neutral): on full verify B>1, master's fused pre-attention writes q in the PADDED/PAIRED layout for FA3's head-paired call, and _paired_norm_gate_kernel reads FA3's paired output layout. The gate kernel writes the plain [B,8,16,64] bf16 layout (out[((b*8+t)*HQ + hh*8+gq)*D]). The build must either read master's paired q and write the layout the paired norm-gate expects, or switch both endpoints for the routed rows. Otherwise there is an extra pad/unpad launch or a wrong-layout epilogue. That hits correctness, or eats several us of the margin, which the node e2e (call-to-call) did not include. Print which q/out layouts and which norm-gate kernel are used on routed rows.
- fused-preattention-triton (and the v3 fused pre-attention prologue): this kernel writes the new K/V at cache_loc before attention. The gate reads the cache right after, so ordering must stay in-stream (no PDL early trigger into the gate launch, see rules_flags). Otherwise the gain is independent.
- Not confounding: the sliding, decode and full decode_1/verify_1 landings (stream-K, cluster, onepass, s2b/sw17). Their rows are untouched and their kernels are not routed through this path. The sw17 gate 'flips between runs' (CLAIMS.md:60), so sliding verify_32 can move about 1-3% on its own. Read the full verify_16/32 rows from the profile, not the headline.

### Interference and A/B needs
No companion needed. The change is routed per row by its own in-prepare race against master's FA3 path, and the build prints per-row, per-round ratios and errors. The judge profile then attributes any movement directly to the full verify_16 / verify_32 rows, and the race itself is the control. If the race does not route, the run still yields the first H200 measurement of the gate's mainloop and of FA3 at v16/v32 (provided the diagnostics below are printed), which is the information the next decision needs. An ab-revert of the pairing landings would not make the single result any less ambiguous; it is a layout-integration question for the build, not an interaction effect.

### Open questions
- What the build MUST print (the [fvg] block plus PATH CHOSEN lines that survive the 16 KB log tail). Build s and whether it finished before the race. From cudaFuncGetAttributes: numRegs, localSizeBytes (spill, must be 0), static+dynamic smem. Occupancy blocks/SM (must be 2), multiProcessorCount and G. Per row and round: master call us, gate e2e us, ratio, fp64 relL2 vs fp32, judge-style metric and worst row vs bound. The route decision with the reason. ALSO, if build time allows, a one-shot H200 diagnostic: gate mainloop-only (MODE 0), math-only and loads-only us with the implied TB/s at v16/v32. That is the H200 gather ceiling and the math floor, the two numbers the node could not supply, and they decide whether a v16-fails/v32-passes outcome is the math floor (stop) or contention (tunable).
- Does master's paired q/output layout on full verify B>1 match what the gate reads and writes? If not, which side is adapted, and is its cost inside the race timing? (The race runs through kernel_fn, so it is, but the node's 0.82 did not include it.)
- Is the 0.95 bar applied per row independently? v16 could plausibly sit at 0.90-0.96 on the H200. Routing v32 alone is still worth about 1.3-1.7%.
- The per-SM CTA priority imbalance (lower-id CTA finishes about 20% earlier) should be the same on the H200 scheduler. A priority-aware static split (<= 2% modelled) is the cheapest follow-up if the gate routes but barely.
- Value estimate: H200 transfer likely gives r about 0.86-0.92, not 0.82-0.85. Re-derive value_gain to about 2-3.5% when logging the result, so the outcome is not scored as a miss against an inflated 4.2%.

