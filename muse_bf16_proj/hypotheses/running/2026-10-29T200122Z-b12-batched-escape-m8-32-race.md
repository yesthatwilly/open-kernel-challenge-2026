---
problem: muse_bf16_proj
slug: b12-batched-escape-m8-32-race
created: 2026-10-29T200122Z
base: master
mint_blob: "a36f33927c06500cbba267d7e0be2235eb8cd716"
best_at_mint: 1.2794
area: "bf16 dense GEMM cells at m=8..32 (pooler, qkv_gate, o): bulk12 kernels"
change: >
  Add bulk12 kernels for the m=8/16/32 cells of pooler, qkv_gate (and o if it fits), each raced
  per cell in prepare against master's incumbent (swap-AB Triton / cuBLAS paths), routed only if
  bit-exact to its own fp32-accumulate contract within the problem tolerance as master's b12 m=1
  cells are, and <= 0.97x the incumbent in every round of a read-flushed interleaved race (reuse
  master's bulk12 read-flush gate machinery, CLAIMS:59). Kernels from takemyjob hopper/b12tma-1029
  d95798e (probes/b12tma; research/h100-b12tma-1029.md): (a) the cp.async bridge b12x with the
  end-of-kernel serial escape-x gather replaced by batched loads, and (b) uc1 = TMA-box b12
  without clusters, same batched escape gather, plain mbarrier.arrive.shared::cluster (no
  arrive.release.cluster / MEMBAR.ALL.GPU). Reuse master's lossless 12-bit codes where R matches
  (pooler R=32, qkv R=16 are already encoded for m=1) to save prepare time and memory. Print
  [b12m] RACE/ROUTE lines per cell (incumbent us, each arm us and ratio per round) in a <= 2 KB
  summary at the end of prepare.
expected:
  - shapes: layer=pooler load=verify_1*
    direction: faster
    why: "node: batched bridge 21.7 / uc1 20.7 us vs node cuBLAS 26.1 at pooler m8; projected judge 0.89-0.94"
  - shapes: layer=qkv_gate load=verify_1*
    direction: faster
    why: node qkv m8 bridge 22.9->17.6 us batched; uc1 0.922 of the bridge
  - shapes: layer=pooler load=decode_16*
    direction: faster
    why: pooler m16 bridge 44.0->27.4 batched; uc1 0.887
  - shapes: layer=* load=decode_32*
    direction: faster
    why: "m32 cells, same escape-gather fix (untested on node; N>=32 mma.sync trap per DISTILL 3.4: route only on the race)"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "b12mc-multicast-x-pooler-qkv-m8-readflush-diag: that measured multicast on a TMA build stalled by MEMBAR.ALL.GPU and kept the serial escape gather; this fixes both and races per cell with routing"
target: check
value_gain: 2.0
value_p: 0.4
value_ev: 0.8
value_cost: 400k tokens, 1 judge run
evidence: >
  H100 b12tma (takemyjob hopper/b12tma-1029 d95798e; research/h100-b12tma-1029.md): serial escape
  gather ~10 us/8 tokens; batched bridge 29.2->21.7 us (pooler m8), 44.0->27.4 (m16), 22.9->17.6
  (qkv m8); uc1 20.7 us vs cuBLAS 26.1 (pooler m8); judge b12x qkv m8 x0.769 (CLAIMS), board
  1.2794 (3e3ffc5a); triage shares pooler/qkv m8-32 ~25%
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T200826Z
review_companions: null
review_best: 1.279360358306445
review_blob: a36f33927c06500cbba267d7e0be2235eb8cd716
stage: claimed
claimed_at: 2026-10-29T202329Z
base_sha: 69eaf6972f9cd230611dddd4bb025a1dbf631d8a
base_blob: a36f33927c06500cbba267d7e0be2235eb8cd716
branch: hyp/muse_bf16_proj/b12-batched-escape-m8-32-race
worktree: .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race
mode: null
restack_of: null
stale_base: null
---

# b12-batched-escape-m8-32-race

## Change

Add bulk12 kernels for the m=8/16/32 cells of pooler, qkv_gate (and o if it fits), each raced per cell in prepare against master's incumbent (swap-AB Triton / cuBLAS paths), routed only if bit-exact to its own fp32-accumulate contract within the problem tolerance as master's b12 m=1 cells are, and <= 0.97x the incumbent in every round of a read-flushed interleaved race (reuse master's bulk12 read-flush gate machinery, CLAIMS:59). Kernels from takemyjob hopper/b12tma-1029 d95798e (probes/b12tma; research/h100-b12tma-1029.md): (a) the cp.async bridge b12x with the end-of-kernel serial escape-x gather replaced by batched loads, and (b) uc1 = TMA-box b12 without clusters, same batched escape gather, plain mbarrier.arrive.shared::cluster (no arrive.release.cluster / MEMBAR.ALL.GPU). Reuse master's lossless 12-bit codes where R matches (pooler R=32, qkv R=16 are already encoded for m=1) to save prepare time and memory. Print [b12m] RACE/ROUTE lines per cell (incumbent us, each arm us and ratio per round) in a <= 2 KB summary at the end of prepare.

## Expected

- `layer=pooler load=verify_1*` faster: node: batched bridge 21.7 / uc1 20.7 us vs node cuBLAS 26.1 at pooler m8; projected judge 0.89-0.94
- `layer=qkv_gate load=verify_1*` faster: node qkv m8 bridge 22.9->17.6 us batched; uc1 0.922 of the bridge
- `layer=pooler load=decode_16*` faster: pooler m16 bridge 44.0->27.4 batched; uc1 0.887
- `layer=* load=decode_32*` faster: m32 cells, same escape-gather fix (untested on node; N>=32 mma.sync trap per DISTILL 3.4: route only on the race)

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 b12tma (takemyjob hopper/b12tma-1029 d95798e; research/h100-b12tma-1029.md): serial escape gather ~10 us/8 tokens; batched bridge 29.2->21.7 us (pooler m8), 44.0->27.4 (m16), 22.9->17.6 (qkv m8); uc1 20.7 us vs cuBLAS 26.1 (pooler m8); judge b12x qkv m8 x0.769 (CLAIMS), board 1.2794 (3e3ffc5a); triage shares pooler/qkv m8-32 ~25%
- **Gain if it works:** 2.0% headline; **p** 0.4; **EV** 0.8% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 2.0% < 2.5%; EV 0.8% < 1.5%), minted because the user described it.
- **Cost:** 400k tokens, 1 judge run.

## Risk

Earlier judged b12x at m=8 (x0.769 qkv, x0.984 pooler) carried the serial escape gather; the H100 lane found it costs ~10 us per 8 tokens (25-38% of the kernel) and matches the judge's unexplained ~1 us per token row. Rows touched sum to ~25% of the headline; ~8% per cell is ~2%. HBM rates do not transfer H100->H200, so the race decides; a cell that misses keeps the incumbent.

## Adversarial review

Reviewed 2026-10-29T200826Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes. The newest b12x line is the [lead] at CLAIMS.md:75, and this build is the judge race its reopen asks for. The refuted L2 hot-spot story (CLAIMS.md:50) is replaced by a mechanism the H100 actually measured. Proceed as a user-requested diag/race, with four fixes: route at master's 0.95 (not 0.97); drop or diag-only m=32 and o; time the old serial-gather bridge (GA0) as a non-routing arm; encode codes when m=1 state is absent. Expect at most pooler m=8 to route, about +0.3-0.6% headline (inside the ±1.24% CI), not +2.0%. The probe's synthetic codes carry 1.9-3.2x the judge's escapes per CTA, and the gather-free b12r already judged pooler m=8 at only 0.98.

### Justification
A. The mechanism is real, and the difference from the refuted b12x claim is real. Phase stamps on the H100 (research/h100-b12tma-1029.md) show the consumer loop ending at 18.6 us and the serial escape-x gather ending at 28.5 us (pooler m8, bridge). Batched loads remove most of that tail: bridge 29.18 -> 21.74 us, m16 44.05 -> 27.38, qkv m8 22.90 -> 17.63. The MEMBAR.ALL.GPU diagnosis is also solid: SASS, instr and ncu agree, and v0's 3.25x on the H100 reproduces the judge's 3.0x. This is a code-path change, not a re-roll. The size of the win on the judge is where the argument is weak, in three places.
(1) WEAKEST LINK: escape density. The probe used synthetic codes with 96 escapes per CTA (base 0x38). The judge's weights carry 4687 escapes in pooler (max 51 per CTA, mean ~37 at R=32) and 2491 in qkv_gate (max 30 per CTA, mean ~17 at R=16): runs/muse_bf16_proj.7e8be36a77534aa7.log:11,28, and master's header says <= 58 per CTA. The gather cost is loads per lane x ~0.43 us (24 loads/lane = 10.4 us at m8). It scales with the worst CTA's escapes x m, so the judge-side tail is ~5.5 us at pooler m8, not 10.4, and ~3.3 us at qkv m8. The note's projection assumes 'the H100 absolute savings carry over', which overstates the saving ~1.9x (pooler) to ~3.2x (qkv). The 'matches the judge's ~1 us per token row' claim is also only half right. On the judge the gather explains ~5.5 of the ~9.6 us per 8 tokens slope (0acdca48: 25.3 / 34.9 / 52.8). The H100's own non-gather slope (bridge m8 -> m16 +14.9 us, of which 10.2 is gather) leaves ~4.7 us per 8 tokens of other per-token cost. Re-projected: bridge 24.3 (5f35efd5, read flush) - ~4.5 = ~19.8 us vs incumbent 17.9, ratio ~1.1. uc1 at ~0.95 of that gives ~18.8 us, ratio ~1.05.
(2) An unnamed prior bounds the gather-free ceiling. b12r (6617f94d, CLAIMS.md:51) keeps x resident in smem and reads escapes from smem (no global gather tail, per its audit). It judged pooler m=8 at 18.68 us vs 19.07 (x1.016, i.e. ratio 0.98). It ran a 6-slot ring (S=8 would have fit), so it was somewhat handicapped. Even so, a batched-gather bridge or uc1 is unlikely to land far below ~0.93-0.98 at pooler m8. The hypothesis does not cite b12r.
(3) Cell by cell against what master serves today (3e3ffc5a; all _bf16_swapab_kernel, Triton swap-AB, CLAIMS.md:3):
- pooler m8: row 19.0 us, read-flush incumbent 17.9, SoL 14.0. The only plausible route, ~0.93-1.0.
- pooler m16: row 19.4, kernel 20.6, read-flush incumbent 18.1. H100 uc1 m16 is +4.6 us over m8, so the judge lands at ~21-22, ratio ~1.15-1.2. No route.
- qkv m8: row 12.8, kernel 9.8, read-flush incumbent 11.6. The judged rotated b12x was 15.63 vs 12.86. The gather fix is worth ~2.5 us at 30 escapes/CTA, giving ~13, ratio ~1.0-1.05. Unlikely.
- m32 cells: pooler row 19.9, qkv 13.5, both swap-AB. b12x pooler m32 judged 52.8 vs 24.1 (write flush). The m32 gather at real density is ~22 us serial, so the batched fix leaves ~30+ us, ratio ~1.6. Tokens on mma.sync N=32 per A fragment is also the DISTILL 3.4 'wide mma.sync, tensor-issue bound' trap. The expected 'decode_32 faster' rests on no evidence: the H100 never ran m32.
- o m8/16/32: kernels 4.1-4.4 us on a 12.6 MB stream. b12x o m8 was 8.0 vs 6.6. Launch- and latency-bound; drop it.
(4) H100 -> H200 transfer. The H100 ratios vs cuBLAS (0.79) do not transfer. cuBLAS on the H100 is a proxy, while the judge's incumbent is Triton swap-AB at ~3.75 TB/s read-flushed. On the H200 the stream part is ~30% shorter (4.8 vs 3.35 TB/s), so every latency-bound residual (gather leftover, x from L2, ramp) is a larger share against an incumbent that does not pay them. Only the GA0 -> GA1 cycle delta transfers, scaled by real escapes.
(5) Value case. Load shares on today's rows: pooler is 39.5% of verify_1 and 31% of decode_16/32; qkv is 26.6% / 20.5%. A 0.93 pooler m8 route alone moves verify_1 by ~2.7%, which is ~+0.45% headline. Realistic gain ~0.3-0.6%, p of any route ~0.3, EV ~0.15%, versus the stated 2.0% / 0.4 / 0.8%. It is below the bar either way, but the user's steer allows it.
(6) Prior field. It names only b12mc, but the bridge arm is literally the a7da6fa9 rotated b12x (2980f57a) with the gather batched, and b12r is the gather-free m8 precedent. It is not 'prior: none', so this is not a hold, but the build note should cite both.

### Confounders on the current kernel
- bulk12 m=1 landing (5325ea40, CLAIMS.md:56): the m>=8 cells plan to reuse its codes, but master pops b12_wc_/b12_esc_/b12_base_/b12_inst_ for any layer whose m=1 cell does not route, and keeps only the winning R (submission.py:4068-4074). qkv m=1 is marginal (judged ~1.00, read gate 0.953 in a3dcd4cb). If it flips, or the R=32 challenger wins, the qkv m>=8 cell silently loses its R=16 codes. The cell must encode for itself (keyed by its own state names) when the m=1 state is absent or has the wrong R. Also, if the new kernels go into _B12_CUDA / B12_EXT_NAME, the m=1 kernels recompile in the same unit. Keep them in a separate extension, or print the m=1 b12 RACE lines to show they are unchanged.
- read-flush gate machinery (8f233e88, CLAIMS.md:59) and the pooled-IQR rule (f10f2752, CLAIMS.md:62): this is the race's protection. Master's own B12_MAX_RATIO_M8 = 0.95 at m>=8 (submission.py:3442, the b12x/b12r lesson), and the refuted claim's reopen also demands <= 0.95. The hypothesis's '<= 0.97 in every round' is looser than both, and at that threshold a gate-vs-judge gap (b12r 0.938 -> 0.984; b12x qkv 0.976 -> 1.30 under the old write flush) could route a loser.
- swap-AB Triton incumbents (8b9ad430 / b99b1dd3, CLAIMS.md:3) are the race baseline. Under the read flush in prepare they time ~6-9% faster than their graph-judged rows (pooler m8 17.9 vs 19.0; qkv 11.6 vs 12.8). The ratio is like-for-like only if each b12 arm is timed through the same kernel_fn on/off _cell_gate path, not as a bare kernel against a graph-timed incumbent.
- No interaction with the router zero-fill PDL (router-only), the qkv m=128 Lt plan, or khalf m=128: different cells, different graphs.

### Interference and A/B needs
No companion run is warranted. Each routed cell's effect reads directly off the judge's own per-row times for (pooler|qkv_gate, m), and an unrouted cell leaves master's bytes on that row. So a headline gain or loss is interpretable from the [b12m] ROUTE lines plus the row times, and a second run would only pool the ±1-5% row noise (CLAIMS.md:66). The one isolation the mechanism needs is cheap and belongs inside this run's prepare, not in a companion. Add the judged serial-gather bridge (GA0, a7da6fa9 code) as a NON-routable diag arm beside GA1/GA2 bridge and uc1, at pooler m8 / m16 and qkv m8. GA0 - GA1 is then the gather's cost on the judge's real escape counts, which is exactly what CLAIMS.md:75's reopen asks the judge to read. Optional, outside the judge: if the H100 lane is still authorized, rerun probes/b12tma with 51 (pooler) and 30 (qkv) escapes per CTA, or the real _b12_encode of N(0,1/k) weights, plus a b12r-style resident-x arm at m8. That is ~20 min of node time and would tell whether the check is worth spending before it is spent.

### Open questions
- Build scope. Raced, routable cells: pooler m8, pooler m16, qkv_gate m8 (plus qkv m16 only if cheap). m=32 cells and o: drop them, or keep them as one non-routable diag arm at pooler m32 to record the N=32 mma.sync and gather behaviour. Is the user fine with dropping them? It saves compile time and prepare.
- Arms per cell: GA0 bridge (diag, non-routable), bridge GA1 (m8) / GA2 (m16, qkv), uc1_g1/g2 (TMA boxes, no cluster, plain mbarrier.arrive.shared::cluster). Does the a7da6fa9 bridge consume master's _b12_encode tile layout at the same R (pooler R=32, qkv R=16 CW4 'B12XQkv4')? If not, it encodes its own, and the build must print the extra MB.
- Routing rule. Proposed: read-flush cand/incumbent via _cell_gate (kernel_fn on vs off), >= 3 rounds, route at _b12_max_ratio(m) = 0.95 on the pooled median with every round <= 0.97 and the pooled-IQR rule, then a profiler kernel-name assertion (exactly one b12m kernel). Does the builder agree to replace the hypothesis's 0.97?
- What the build must print (<= 2 KB [b12m] summary at the very end of prepare, so it survives the 16,000-char log tail):
- the compile seconds of the new extension and total prepare seconds (limits: prepare 300 s, whole run 10 min);
- per cell, the encoding source (reused m=1 state or own encode), R, base, escapes total / max per CTA, and extra state MB;
- per arm: occupancy (CTAs/SM, grid, smem, regs, spill), fp64 relL2, power-of-two-row error, err word;
- per round: incumbent us, each arm's us, ratio, IQR, read flush plus write flush beside it;
- GA0 - GA1 us (the judge-side gather cost);
- the ROUTE / NO-ROUTE line with its reason, plus the PATH CHOSEN per (layer, m).
- Is the H100 lane still open for a 20-minute pre-check at the judge's real escape density (51/30 per CTA) with a b12r-style resident-x m8 arm? If yes, run it first. A projection of <= 0.95 at pooler m8 is what makes the judge check worth more than a diag.
- For the decode bulk12 shelf (CLAIMS.md:97 per the LOG): the escape-density correction applies there too. Re-project with the real escapes per CTA before that shelf reopens on this result.

