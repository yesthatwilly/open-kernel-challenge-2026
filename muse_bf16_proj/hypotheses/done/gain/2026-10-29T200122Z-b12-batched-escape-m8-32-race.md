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
status: done
review_at: 2026-10-29T200826Z
review_companions: null
review_best: 1.279360358306445
review_blob: a36f33927c06500cbba267d7e0be2235eb8cd716
stage: harvested
claimed_at: 2026-10-29T202329Z
base_sha: 69eaf6972f9cd230611dddd4bb025a1dbf631d8a
base_blob: a36f33927c06500cbba267d7e0be2235eb8cd716
branch: hyp/muse_bf16_proj/b12-batched-escape-m8-32-race
worktree: .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T212209Z
impl_commit: 84995d9e32799d8c940b71c9d279bedaad12e6d1
submission_sha256: 7837d1e422852f91a995c3bc960614a411f0cb4aabf584aa9c70fe09983ed64b
norm_sha256: 6a001bc2811e29072474f4cb3f5160bc7340a3492911325721a6db4baeaac7af
submission_blob: 5ef62a2078f6922e02a5c5c4e9aa9c7a1b3e31ab
lint_warnings: 58
attempt_file: 2026-10-29T212210Z-check-b12-batched-escape-m8-32-race.md
prepared_at: 2026-10-29T212210Z
job: bced57b42b87444d
dispatched_at: 2026-10-29T212219Z
outcome: gain
verdict_detail: gain-pooled
speedup: 1.2973637641537636
speedup_ci: [1.2836027160087515, 1.3111248122987758]
harvested_at: 2026-10-29T212710Z
best_at_harvest: 1.279360358306445
promotion: pending
mechanism: miss
confirm_queued: b12-batched-escape-m8-32-race-cf
confirm_job: c0f8969e76cf4cb7
confirm_outcome: gain
pooled_speedup: 1.2973637641537636
runs: 2
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


## Implementation notes

**Mechanism:** pooler and qkv_gate at exactly m=8 run a new 12-bit row-owner kernel (b12m_tma_kernel, uc1_g1: TMA-box x without a cluster, local mbarrier arrives, batched escape-x gather). It lives in its own extension and routes only behind a read-flush race against master's swap-AB incumbent.

- **Where:** worktree `muse_bf16_proj/submission.py`, new section (12) after section (10) (+1049 lines, nothing removed). New extension `muse_bf16_b12m_m8_tma_v1` (TORCH_CUDA_ARCH_LIST=9.0a, `-O3 -lineinfo -std=c++20`). The m=1 `_B12_CUDA` unit is not touched or recompiled. Device code comes from takemyjob `hopper/b12tma-1029` d95798e / `hopper/b12dens-1029` 6d80717 `b12t_kernel` at CL=1 (cluster, multicast, AR and INS code stripped), and from a7da6fa9 / 2980f57a `b12x_kernel` for the bridge (GA0 is byte-for-byte the judged gather). Both kernels share one epilogue helper.
- **Arms:**
  - Routable: pooler m8 `uc1_g1<R32,S6,MINB2,CW8,NT8>`, qkv m8 `uc1_g1<R16,S4,MINB2,CW4,NT8>`.
  - Non-routable diagnostics: bridge GA0 and GA1 at pooler m8 and qkv m8 (extra `_cell_gate` arms, which print GA0-GA1), and pooler m16 `uc1_s8_g1` as a timed-only cell.
  - Dropped: m=32 cells and o (review fix).
- **Routing:** exact shape only (`_b12m_route`). It requires `(layer, cfg m)` in B12M_CELLS, `x.shape[0] == m`, bf16, contiguous, 16 B aligned and a row stride divisible by 16 B. The `uc1` tensor map is encoded inside `b12m_run` from `x.data_ptr()` on every call. The encode is a host-only `cuTensorMapEncodeTiled` resolved through `cudaGetDriverEntryPointByVersion`, it is never cached, and only the driver function pointer is static.
- **Gate (review fix: 0.95, not 0.97):** each arm runs `_cell_check` (fp64 rel L2, probe row, repeat bit-identity, error word), then `_b12m_rows` (every row's rel L2 <= 1.5e-2). After that come 3 read-flush `_cell_gate` rounds of kernel_fn with the flag on vs off (master's routed path, like for like) and 1 write-flush round as a diagnostic. A cell routes only if the pooled median ratio is <= B12_MAX_RATIO_M8 (0.95), every round is <= 0.97, and the pooled cand IQR is <= 2x max(incumbent IQR, 0.5 us). It must then pass a strict profiler assertion: exactly one `b12m_tma_kernel` (no profile means no route).
- **Codes (review fix):** a cell reuses the m=1 state's `b12_wc_/esc_/base_` only when that instance has the same R. Otherwise it encodes its own codes through `_b12_encode_checked` (host + device round trip) under `b12m_*` keys. The source, R, base, escapes total / max per CTA and extra MB are printed. Codes are popped when no cell of the layer routes.
- **Error word:** prepare-only. kernel_fn passes it and never reads it. Bounded mbarrier waits (0.5 s), and every launch is announced and synced in the checks.
- **A/B and summary:** qkv_gate m=8 was added to AB_CASES. `_ab_component` maps (pooler|qkv_gate, 8) to its own b12m flag, and the second state reloads codes. A <= 2 KB `[b12m]` SUMMARY prints after the DECISIONS block, as the last lines of prepare. It carries build s, prepare s, codes, per-arm regs/spill/smem/occupancy/worst-row, per-round inc/arm/ratio/IQR, write flush, pooled ratio, GA0-GA1, ROUTE/NO-ROUTE + reason, and PATH CHOSEN.
- **Size:** 309,225 B (cap 1,000,000).
- **Deviation:** none from the narrowed spec. The m=1 RACE lines are not re-printed, because the m=1 unit is a separate extension and is unchanged.

### Node validation (gpunode06 GPU 6, H100 SXM, nodejudge venv torch 2.14.1+cu130, 2026-10-29 20:35-20:39Z)
The same bytes (sha256 7837d1e422852f91) went through guarded `hopper-probe` as the operator, with node-local staging that was unstaged after the run. Run dir: takemyjob `~/git/.takemyjob-wt/b12mval/prototypes/hopper-probe/runs/20261029-133523-gpunode06/` (branch hopper/b12mval-1029, uncommitted).

- **Build and prepare:** the b12m extension compiled in 37.0 s. Import with every load_inline unit took 191 s, and prepare took 9.9 s. Register and spill counts: uc1 r48 (both cells), uc1_s8 r55, bridge r36-48, all with 0 spill and the needed occupancy (2/1 CTA/SM).
- **Correctness:** every arm passed (rel L2 1.64-1.66e-3, worst row 1.7e-3, probe rows exact, error word 0). The judge's `check_implementation` passed on 10 shapes through graph replay with refilled inputs, outliers included. The A/B check passed at pooler and qkv m8.
- **Codes:** pooler reused the m=1 codes (escapes 4559, max 51 per CTA). qkv encoded its own (escapes 2661, max 30 per CTA, +28.6 MB), because its m=1 cell picked another R.
- **Race on the H100** (read flush, net us, inc/arm):
  - P8: 24.0/18.6-18.75, pooled ratio **0.774**, GA0-GA1 3.90 us. ROUTE.
  - Q8: 15.1/11.5, pooled ratio **0.764**, GA0-GA1 1.95 us. ROUTE.
  - P16 (diag): pooled 0.863.
  These absolute numbers do not transfer to the H200. The b12dens projection is 0.939 / 0.906. The judge's own gate decides.

## Rules check
Each item was checked against `git -C .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race diff`.
- [x] CUDA-graph capturable: kernel_fn adds only `_b12m_route` (cfg, shapes, dtype, contiguity, `data_ptr() % 16` and state flags) and `_b12m_forward` (`torch.empty` plus one extension launch whose host-side tensor-map encode is not a stream op). Every `.item()` and synchronize sits in prepare-only helpers (`_b12m_rows`, `_b12m_prepare`). The graph capture and replay were validated on the H100.
- [x] Nothing derived from `data` is cached: the tensor map is a stack-local built on each call from that call's x, and `_B12M_SUM` and `_B12M_BUILD` hold only prepare text and build state.
- [x] Weights: a lossless 12-bit codec (section (10)'s `_b12_encode`, host + device round trip bit-exact), with escapes holding the true bf16 bits. Accumulation is fp32, followed by one bf16 rounding.
- [x] Activations stay BF16 (x is read as bf16, the output is bf16). The KV-cache / query item is n/a: this is not attention.
- [x] Attention full window: n/a, not attention.
- [x] No timing patches, harness frames or I/O, module edits, reference import, subprocesses or network: `load_inline` only (as in master), and the reference is imported only by the node driver outside the submission.
- [x] All work happens inside the timed call: one kernel per m=8 call, with no output caching or replay.
- [x] The first line `# okc: problem=muse_bf16_proj` is kept. py_compile is OK and __pycache__ was removed.

okc_lint: `OK with warnings: .../muse_bf16_proj/submission.py`. All 58 warnings are host-sync/dtype items "outside kernel_fn's call graph; fine in prepare()". There are no errors.

## Result

- job `bced57b42b87444d` (check), status done, passed True, tests 9/9
- speedup 1.2871x, 95% CI [1.2678, 1.3064], score 3753.1322673646223 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.279360358306445, "ci": [1.2601699529318484, 1.2985507636810416], "job": "3e3ffc5afc57433b"}; base result: muse_bf16_proj/hypotheses/BEST.result.json
- **outcome: neutral** (unproven-vs-best); mechanism: miss (2/10 predicted rows past ±2%)
- files: `muse_bf16_proj/hypotheses/done/neutral/b12-batched-escape-m8-32-race.result.json`, `muse_bf16_proj/hypotheses/done/neutral/b12-batched-escape-m8-32-race.patch`; branch `hyp/muse_bf16_proj/b12-batched-escape-m8-32-race`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.39 | 10.58 | 0.9947 | flat |
| layer=o load=decode_1 m=1 | 62 | 5.7 | 5.55 | 1.005 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 15.02 | 14.97 | 1.0158 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 4.97 | 4.99 | 1.0348 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.11 | 5.26 | 0.9795 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.93 | 12.89 | 0.9876 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.58 | 6.31 | 1.0283 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.35 | 19.58 | 1.0065 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.62 | 18.71 | 0.9871 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.31 | 5.25 | 1.0075 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.47 | 14.38 | 0.959 | lost |
| layer=o load=decode_32 m=32 | 62 | 6.34 | 6.7 | 0.9688 | lost |
| layer=pooler load=decode_32 m=32 | 62 | 19.93 | 20.21 | 0.9932 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.3 | 18.68 | 0.9893 | flat |
| layer=router load=decode_32 m=32 | 62 | 5.74 | 5.85 | 0.9906 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.75 | 11.68 | 1.0463 | gained |
| layer=o load=verify_1 m=8 | 62 | 6.11 | 6.23 | 0.9926 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.02 | 16.01 | 1.203 | gained |
| layer=unpooler load=verify_1 m=8 | 62 | 5.19 | 5.61 | 0.9686 | lost |
| layer=router load=verify_1 m=8 | 62 | 4.99 | 5.34 | 0.9396 | lost |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.07 | 15.74 | 1.0055 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.23 | 7.41 | 0.9723 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 21.93 | 22.34 | 0.9859 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 18.6 | 19.32 | 0.9739 | flat |
| layer=router load=verify_16 m=128 | 62 | 6.98 | 7.57 | 0.9448 | lost |
| layer=qkv_gate load=verify_32 m=256 | 62 | 20.01 | 20.0 | 0.976 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.38 | 9.66 | 0.9646 | lost |
| layer=pooler load=verify_32 m=256 | 62 | 30.25 | 30.15 | 0.9965 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.34 | 24.22 | 0.9968 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.61 | 9.28 | 1.0384 | gained |

gained on unpooler decode_1; qkv_gate verify_1; pooler verify_1; router verify_32; lost on qkv_gate decode_32; o decode_32, verify_32; unpooler verify_1; router verify_1, verify_16. Expected hit: 2/4. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

