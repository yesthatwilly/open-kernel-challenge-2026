# Log: campaign 2026-10-26-expert-bandwidth

The "## Now" block is rewritten by every loop tick (read it first; `head -40 LOG.md`). Entries below it are
append-only, newest last.

## Now

- **Quota policy (the user, 2026-10-26 ~19:05Z, PLAN §6 top and §11; keep this bullet first):** never idle until a
  reset, and never pre-build hours ahead of one. When a problem's checks run out, write its reset plan, then work on the
  highest-gain problem that still has checks today, with immediate check feedback. Order: decode > moe > bf16_proj >
  attention_decode > prefill (last). On bf16 / attention, favour kernels that port into decode, and submit when a check
  beats that board. Every check must answer a new question; using every check is not required; no identical bytes
  re-rolled.
- updated: 2026-10-26T21:45Z
- **Sprint (the user's 21:40Z steer, this window only):** spend bf16 / attention / prefill quota before 00:00Z on ~30
  hypotheses via /hypothesize + /churnsubs with ~8-10 agents. No dispatch after ~23:45Z; harvest and land winners;
  at 00:00Z back to the reset plan and the 19:05Z policy.
- step: **bf16_proj**.
  - Done: bulk12 m=1 scored at 1.2664 (board 1.2566 -> 1.2664, rank 381 -> 355) and landed on master.
  - Building: bulk12 for m = 8 / 16 / 32 (branch campaign/expert-bandwidth after 3acb748e, uncommitted). b12x_kernel
    streams the NT token rows of x with each chunk.
    - Cells: pooler / qkv / o at m=8, then pooler / o / full unpooler at m=16 / 32.
    - Each layer's codec stats, instance gates and PATH CHOSEN lines are re-printed at the end of prepare.
  - Check 22ddaf4a (1.2683): no m >= 8 cell routed. Per-row x TMA copies cost ~30-50 ns each.
  - Fixed in d466a692 (x via cp.async from all producer lanes): check 0acdca48 still loses at m >= 8. The next b12 step
    is cluster multicast of x (a later bf16 window).
  - In flight: router ticket check 7e8be36a (no memset at m >= 9). Submit if it beats 1.2664.
- in flight: bf16 check 7e8be36a77534aa7 (router ticket)
- at 00:00Z (10-27), decode / moe, from the drafts in the session scratchpad (submit_attempt.md, dqab_attempt.md,
  anat_attempt.md):
  1. `okc submit` of `submission.edoff-27ef83df.py` (check f59cb301, 1.5347). Verify that the content hash is
     613b6c2d..., then land it on master.
  2. moe check at eacf0722:
     - the ladder dq A/B (dq2 / dq3 / dq4);
     - H=-8;
     - v4c.
  3. decode check at eacf0722: H=-8, v4c, and the anatomies.
  4. Then build, for the next decode check: the bulk12 codec in decode's dense GEMMs at T=1.
     - P = dl_pooler_router: the pooler's 4096 rows of W_pr on 12-bit; the router's hi / lo rows stay raw.
     - qkv and o, from bf16's b12_kernel (judge-proven: pooler m=1 17.7 -> 14.8 us in bf16).
     - Gate each against today's route, as in bf16. At T >= 8 the decision waits on the bf16 cp.async-x check
       (0acdca48).
- quota left 10-26: decode 0 checks / 0 scored, moe 0 / 4, bf16 8 / 9, attention 14 / 10, prefill 16 / 10
- boards: decode **1.4956** (rank 347 of 740; 1.5347 would be about 318), moe **1.6851** (rank 370 of 844),
  bf16 **1.2664** (rank 355 of 1090; rank 300 = 1.2865, 200 = 1.3503), overall 347 of 1228

## 2026-10-26T15:35Z: campaign opened

- Approved plan saved as PLAN.md; user decisions: single agent + one Opus audit per new CUDA module; fully
  autonomous (self-paced loop, scored submits on improvement, commit / merge / next campaign without asking).
- Predecessor 2026-10-26-decode-first-rewrite closed (RETRO.md). Code continues on branch campaign/expert-bandwidth
  forked from campaign/decode-first-rewrite ae91f55d.
- Biggest scoring gain: the NVFP4 expert block (FINDINGS.md 1-3); T=128/256 first, T <= 32 consumer second.

## 2026-10-26T16:05Z: X0a (judge-shaped gate inputs) dispatched; moe harvested

- Code c8ac9beb on campaign/expert-bandwidth.
  - decode: `_cb_data`'s stream is judge-shaped on our own calibrated basis (`_cb_basis`: 8 draws x 6 betas, nearest the
    judge's per-load distinct experts), and attn_input = rms(stream).
  - moe: the v3 big-T and T=256 hybrid selectors time on reference-style p10/p50/p90 draws (`_ea_judge_routings`).
- A CPU model of reference.py (scratchpad sim_routing.py) shows how far the old gate inputs were from the judge:

  | | T=128 max | T=256 max | T=256 distinct |
  |---|---|---|---|
  | judge | 45-87 | 85-172 | ~20-21 |
  | old gate inputs | 25-28 | 43-45 | 27-29 |

  Its judge distinct counts match the backed-out targets within 8-20%. The surrogate basis at beta 0.3 matches within
  6-15%.
- moe 68216f9dfd9741ec (check): passed, **1.5849** [1.561, 1.609], noise vs board 1.5762; selections unchanged:
  - T=128 v3 (worst x1.057 on the judge draws).
  - T=256 master: v3 x1.007 at p50 and x0.928 at p90 (424 routes, max 123).
  - the hybrid wins only at p10 (H-32 x1.064) and loses at p50/p90 (x0.954-0.958).
  - **T=256 needs a new engine (X2).** Not submitted: same timed kernels as the board run.

## 2026-10-26T15:51Z: X0a decode harvested, scored and landed (decode 1.4466, rank 370; overall 375)

- Correction: the 16:05Z stamp on the entry above was wrong. That entry was written at about 15:40Z.
- decode 7fff4a8b616946c6 (check, c8ac9beb): passed 13/13, **1.4524** [1.4307, 1.4742].
  - Basis draw 2 beta 0.3, worst distinct error 9.7%, 0.7 s. Gate inputs on the busiest range
    (routes/distinct/max/hot): T=16 32/8.1/10/0; T=32 64/9.5/20/1.5; T=128 213/18.3/55/4.1; T=256 420/20.3/105/7.5
    (judge distinct 7.5 / 9.0 / 18.4 / 21.3).
  - One flip: Engine A's T=9..16 class v2 -> v3. T=128 stays v3; T=256 stays the hybrid H=-48 (542 vs 606 us
    master). The hot/cold gate at T=128 keeps master (H64 x1.010).
  - Rows vs board run 0d84f58c: d16 s 185.1 -> 177.3, f 287.4 -> 277.1 (x1.04, the flip). Others within -2.6..+4.6%
    drift (d32 s 226.4 -> 216.5 with no selection change).
- Scored a03e596d3aaa4f69 (same bytes, extracted as submission.x0a-c8ac9beb.py): **1.4466** [1.4249, 1.4683], passed.
  - Board 1.4422 -> **1.4466**; decode rank 373 -> **370 of 733**; overall 380 -> **375 of 1223**.
  - Landed on master (muse_decode_layer/submission.py sha256 850e9557...).
- Next: X0b (R no-align + combine prefetch) awaits its audit. A g++ -fsyntax-only CUDA checker (rewrite/bin/
  cuda_syntax.py with cuda_stub/) now parses every judge-compiled source and caught a planted arity error. Run it
  before every dispatch.

## 2026-10-26T16:05Z: X0b dispatched; S1 (v4) built

- X0b (d25a987d): R skips its Marlin align where Engine A runs; ea_combine issues all slot loads before one barrier.
  - Read-only Opus audit: no BLOCKER or MAJOR. Three MINOR fixes applied: the no-align check runs before the layer
    checks, the ticket counter is asserted 0, and the combine's regs/spill are printed.
  - Checks: decode 523252767fc84a26, moe 6eb226e9db4f4876.
- S1 = Engine A **v4** (rewrite/lib/cuda/ea4_kernels.cuh, ea4_core.py): v3 on a CTA of 8 consumer warps plus a
  producer warp (288 threads).
  - A stage is two group records (groups 2sg and 2sg+1, two 18432 B bulk copies on one barrier) plus the token rows
    once. D=2. The work unit is (pass, 128-row super-group, k-chunk).
  - build_map runs on 9 warps, its scratch aliased into the ring. The Map is 5324 B, smem 112364 B, 2 CTAs/SM.
  - The module builds with v4 and falls back to a build without v4 on failure.
  - Candidates: every T<=32 class (EA_ENGINES); decode big-T at T=128 only (T=256 keeps v3 then the hybrid gate);
    moe big-T at 128 and 256.
  - cuda_syntax.py passes for both builds (with and without v4).

## 2026-10-26T16:13Z: X0b harvested; scored run below the board (noise); end-game note

- decode 523252767fc84a26 (check, d25a987d): passed, **1.4569** [1.4351, 1.4788].
  - Rows X0b changes vs a03e596d: d1 s 105.1 -> 103.0, f 112.5 -> 110.5; d32 s 215.3 -> 210.9.
  - Near-tie gates flipped: Engine A T=2..8 v2 -> v3, T=9..16 back to v2; dense swap-AB unpooler at 9..16 on, o at
    17..32 off. v1 rows +4% (152.8 -> 159.4 s).
- Scored a009b6c8a9df4e3f (same bytes): **1.4401** [1.4185, 1.4617], below the board's 1.4466. The board keeps
  a03e596d (X0a) and master keeps that file. The scored spread for the same code is about +-1.2%.
- moe 6eb226e9db4f4876 (check): passed, **1.5879** [1.5641, 1.6117]. Combine: 187 regs, 0 B spill. Combine rows
  3.5 -> 3.1 us (T=8) and about 4.3 -> 2.7 us (T=32). Row drift is up to +-6% against the X0a check (T=256 master
  402.8 vs 387.4 with no code change). Not submitted: moe does not move our overall rank.
- **End-game note (FINDINGS):** final standings come from post-close reruns of each team's board submission (technical
  rules: "we'll read the top submissions ... rerun them several times ... medians of these reruns").
  - The board keeps the best-scoring run, so a noise-high older run can hold the board row with worse code.
  - Before the close, make sure the final code holds the board row: scored runs of the final bytes, and if an older
    run still leads, consider `okc withdraw` of the older run (standing rules allow withdraw).

## 2026-10-26T16:22Z: v4 dispatched; v5 (D1) built and in audit

- v4 audit (read-only Opus): no BLOCKER.
  - MAJOR fixed: moe's kernel-only rates timed v3's kernels for every engine past v2.
  - MINORs fixed: async-proxy fence after build_map's ring-scratch writes; fallback-build deadline; labels.
  - Engine A gate margins ("worst ratio" lines) now go into the final printed block (16 KB tail).
  - v4 checks: decode ff2bd9182fa04636, moe 54f63952015c4d3b (666eb1d7).
- **v5 = D1** (de1be708; rewrite/lib/cuda/ea5_kernels.cuh, ea5_core.py): v4 with passes of up to 32 routes (1/2/4 n8
  tiles per A fragment) on 256-k chunks.
  - Records are repacked on the device from v3's (a permutation) into super-group-contiguous order, so a stage is one
    18432 B copy plus up to 32 token rows of 512 B. D=3, smem 110972 B, 2 CTAs/SM.
  - Hot experts stream ceil(n/32) times instead of ceil(n/16); x traffic per weight byte is half of v4's.
  - Candidate at T<=32 classes and T=128/256 in both problems.
  - decode kernel_fn now takes the hot/cold hook first, so its gate times H against Engine A's choice; that is why v5
    may also compete at T=256.
  - Engine gates prefer the earlier engine within 1% (`_rt_select(prefer_within)`) against near-tie flips.
  - The build chain is v4+v5, then v4, then base; all three variants pass cuda_syntax.py.

## 2026-10-26T16:30Z: v4 harvested: moe landed (1.6389, rank 395); decode check faulted in the copy check

- moe 54f63952015c4d3b (check, 666eb1d7): passed, **1.6412** [1.6166, 1.6658].
  - v4 won T=2..32, T=128 (worst x1.159 vs master; v3 x1.064) and T=256 (x1.023 vs the wg path; v3 x0.930).
  - Kernel-only w13: T=8 3.07 -> 3.23 TB/s, T=16 2.88 -> 3.09, T=32 2.66 -> 2.92 (v3 -> v4). That is +5-10%, short of
    the 3.4-3.7 the research expected.
  - Rows vs the X0b check: T=8 +3.9%, T=16 +3.3%, T=32 +3.6%, T=128 +5.5%, T=256 +1.3% (the gate's p50 said x1.12; the
    judge's T=256 draws may sit nearer our p90). Copy check T [1, 2, 256] PASS.
- moe scored 35b55c74ee964187: **1.6389** [1.6144, 1.6635]. Board 1.5762 -> **1.6389**, rank 428 -> **395 of 843**.
  Landed on master.
- decode ff2bd9182fa04636 (check, 666eb1d7): **failed**.
  - The gates ran: v4 won T=8 (x1.095; v3 1.068), T=16 (1.127; 1.074), T=32 (1.070; 1.040) and T=128 (1.090; 1.029).
  - Then "illegal memory access" in `_ead_copy_check` (picks T 1 / 2 / 128) poisoned the context: every path fell back
    to master, the run failed.
  - Not found by reading: v4's bounds, smem layout and sync match v3, and moe's copy check passed with v4 at T=2.
  - The decode-only factor at T=2 is K1 launched with PDL after kernel R. The PDL gate kept PDL at x0.998-1.000 (no gain).
  - Next decode check: v4 / v5 K1 plain; an eager kernel_fn sweep at T 1,2,3,5,8,9,12,16,17,24,32,128,256 before the
    copy check; step prints in the copy check, so a repeat names its T and phase.

## 2026-10-26T16:55Z: v4 decode passes (1.4950) and is scored; v5 loses everywhere (off); X2 turns to wgmma-hot hybrids

- decode 5573e05423304324 (check, 4f530b95: v4 / v5 K1 without PDL, eager sweep, copy-check prints): passed 13/13,
  **1.4950** [1.4725, 1.5174].
  - Engine A: T=1..8 v1 (v4 within 1%: 1.047 vs 1.050); T=9..32 v4 (x1.116 / 1.088; v3 1.087 / 1.065); T=128 v4
    (x1.094; v3 1.035); T=256 v4 (x1.089 over master).
  - The hybrid (Marlin hot block 48 + v3 cold) beat v4 at T=256: 483 vs 516 us. Its gate now times against Engine A's
    choice (kernel_fn takes the hybrid hook first).
  - Rows vs board run a03e596d: d16 +2.9 / +4.4%, d32 +4.5 / +1.9%, v16 +6.1 / +4.9%, v32 +5.2 / +2.1%, d1 / v1 ~+1%.
  - No fault; the copy check passed. But T=1..8 went to v1, so v4 at T=2 with PDL was not exercised; K1 of v4 stays
    plain.
  - Scored dc51d8e8f58e4cb7 (same bytes) in flight.
- moe e403a42a80e6478f (check, 4f530b95): passed, 1.6441; **v5 lost every gate**.
  - Kernel-only w13 at T=8 / 16 / 32: v5 2.90 / 2.82 / 2.39 TB/s vs v4 3.21 / 3.12 / 2.92.
  - Big gates: T=128 v5 x0.899 vs v4 1.155; T=256 v5 x0.863 vs v4 1.019, so T=256 fell back to master this run.
  - Cause (inferred): the 4-tile mma.sync path runs about 0.25 HMMA/clk/SM, below the measured 0.31 floor (tensor
    issue), and half-size stages double the per-stage overhead with less in flight.
  - **D1 as designed is dead. v5 is off (EA_BUILD_V5 = False).**
- X2 next (4848aadc): hot experts need wide-N tensor work (wgmma).
  - moe's judge-proven wgmma path forms per-expert chunks of up to 128 routes (N up to 128), so it reads a hot expert
    once per chunk.
  - The new moe hybrid H=-16 sends hot experts to the wgmma path and cold experts to v4. It runs at T=128 / 256,
    gated against kernel_fn's own choice.
  - The decode hybrid's cold half moves to v4.
  - Checks: moe 0f4605dca83744c7, decode 212cdf5dcb844d54.

## 2026-10-26T17:05Z: decode v4 scored 1.4956 (rank 342); moe wgmma-hot hybrid scored 1.6851 (rank 370); decode port out

- Scored decode dc51d8e8f58e4cb7 (4f530b95): **1.4956** [1.4731, 1.5180]. Board 1.4466 -> 1.4956; decode rank 370 -> **342
  of 737**; overall 376 -> **347 of 1228**. Landed on master.
- moe 0f4605dca83744c7 (check, 4848aadc): passed, **1.6845**.
  - The hybrid H=-16 (hot experts > 16 routes on moe's wgmma path, cold on v4) won T=256: worst x1.120 over master
    (p10 1.198, p50 1.136, p90 1.120). Layer check vs fp32 ok, copy check PASS. T=256 row 391.1 -> 339.7 us.
  - At T=128 it lost (x0.970): few hot experts at p50/p90, and the split costs.
  - Scored e854db4683dc495f: **1.6851**. Board 1.6389 -> 1.6851, moe rank 395 -> **370 of 844**. Landed on master.
- decode 212cdf5dcb844d54 (check, 4848aadc: hybrid cold half on v4, v5 off): passed, **1.5007**.
  - v32 rows 528.7 -> 517.2 (s) and 796.4 -> 787.9 (f). Prepare 59 s.
  - The T=8 class flipped to v4 (x1.093 vs v1 1.072) and the v1 sliding row read +5.3%. But the profile of that row
    shows v4 faster (GPU busy 149 vs 187 us), so I take it as run-to-run variance. Watch it.
- Decode port (3bcb685b):
  - moe's wgmma path moved verbatim to `rewrite/lib/wg_core.py`; moe's generated code is byte-identical apart from
    the include markers.
  - decode includes it. Its hybrid gains H=-16: `_hcw_build` builds the module, checks attributes, attaches scratch
    and retiles 32 experts; a miss turns only H=-16 off.
  - Check a1bc1756256b4c91.

## 2026-10-26T17:50Z: decode wg-hot hybrid check 1.5097; selector tie-break reverted; ladder-2 verdict (v4 is consumer-bound)

- decode a1bc1756256b4c91 (check, 3bcb685b: wgmma-hot hybrid H=-16 at T=128/256): passed, **1.5097** vs board 1.4956.
  - v32 rows +15.5% (s 539.4 -> 466.9) and +10.8% (f 807.2 -> 728.5); v16 +1.7%.
  - T <= 32 rows were 1-9% slower. Cause: `EA_PREFER_WITHIN = 0.01` put T=9..16 on v3, where v4 leads by 3-5% in
    every run (d16 s 169.1 -> 180.7).
  - Not scored: decode had no scored runs left today. Bytes kept as `muse_decode_layer/submission.wghyb-3bcb685b.py`.
- 8b098599: `EA_PREFER_WITHIN = 0.0` again (the best worst ratio wins). At the 00:00Z reset, check this build first
  (expected about 1.52: the hybrid gain plus the T <= 32 rows back) and score whichever of the two beats the board.
- moe 5eafb39e194d4d8b (check, 1cec46df: ladder-2, diagnostic only): passed. Rung ceilings in TB/s, v4's geometry
  (8 consumer warps, two 18 KB records per stage):

  | rung | TB/s | drop |
  |---|---|---|
  | C-R1 ring alone, D=2, 2 CTAs/SM | 4.37 | (4-warp ring 4.46) |
  | C-R2 + exact dequant | 3.98 | -8.9% |
  | C-R3 + MMA (const B, 1 tile) | 3.75 | -5.8% |
  | C-R4 + x from smem, 8 tokens | 3.45 | -8.0% |
  | C-R4 + x from smem, 16 tokens (2 tiles) | 2.94 | -14.8% |
  | C-R1 / C-R4 at D=3, 1 CTA/SM | 4.34 / 3.44 | same as D=2 |

  - 8 vs 4 consumer warps at the same ring work: x1.024 (8 tokens), x1.043 (16 tokens). So v4 is neither ring- nor
    latency-bound: **the consumer's instructions per weight element are the limit.** Each addition costs throughput
    in proportion to its work. The B fragments (ldmatrix of x) and the MMAs scale with the token tiles.
  - v4's real kernels (3.21 / 3.12 / 2.92 at T=8/16/32) sit about 7% under C-R4 (maps, scatter, fix-ups).
  - Next lever for v4: each warp takes a k-slice of all 128 rows instead of 16 rows of all k. One B fragment then
    feeds 8 m-tiles instead of 1 (B loads / 8), at the cost of one cross-warp reduction per unit. A cheaper dequant
    is second.
- Ladder runs archived (profile files). No checks left today for either problem (decode 0/0, moe 0/4 scored).
- Now building Engine D (`rewrite/lib/cuda/ed_kernels.cuh`): a dense bf16 skinny GEMM on v4's CTA for the decode
  layer's P (pooler + router), qkv_gate, o and the unpooler at T <= 32.
  - These rows today: P 22-27 us at 2.8-3.4 TB/s; qkv_gate 11-13 us; o 5-5.4 us; unpooler at T >= 9 16.7 us (cuBLAS).
  - Engine D's ring alone reaches about 4.4 TB/s. Without dequant its consumer is 4x lighter per byte than v4's.
  - Expected savings per row: about 9 us at d1, 13 at d16, 20 at d32 (headline about x1.04).

## 2026-10-26T18:16Z: Engine D audited and dispatched (decode f59cb301895240e6)

- Engine D (27ef83df) is a dense bf16 skinny GEMM on v4's CTA. It covers P (pooler + router hi/lo pairs, fp32
  logits), qkv_gate, o, and the unpooler at T <= 32.
  - Kernel: 288 threads, a 2-stage ring of 32 KB, 2 CTAs/SM, stream-K over (128-row group, 128-k chunk) units, and
    last-arriver tickets.
  - Records are a lossless permutation in mma A-fragment order. On CPU, the roundtrip and the fragment order were
    checked exhaustively with numpy.
  - Each (GEMM, T class) has its own checks vs fp64, a GEMM selector against its incumbent, and a whole-layer gate
    and layer check, plus a copy check. Any miss keeps the incumbent.
- Audit (read-only Opus): no BLOCKER or MAJOR. The auditor emulated the fragment path end to end in numpy at 9 values
  of T and simulated stream-K over every grid from 1 to 299.
  - Fixed: its MINOR on the last arriver's sum, which ran one dependent L2 load and a division per contributor (about
    15 contributors per qkv_gate group). It now batches 8 / NT loads, picks the slot without a division (simulated
    clean) and uses the template NT.
  - Noted, not fixed: the whole-layer gates time full attention only at T=2..8; a timed-out mbarrier wait yields
    garbage (as in Engine A).
- The check also prints an anatomy: the expert block alone at T=1/8/16/32 next to the same launches with no local
  route. The d1 profile suggests about 45 us of expert block, against about 17 us at the ring's rate.
- Next lever, pending the anatomy: X3 glue folds, or dq3. dq3 is a dequant with fewer INT32 ops. Ladder-2 puts it at
  about 15 INT32 per k16 step at half rate, which is the consumer's limit. The best code layout saves about 15%.

## 2026-10-26T18:45Z: decode f59cb301 passed at 1.5347 (to be scored at 00:00Z); Engine D killed; dq3 built

- decode f59cb301895240e6 (check, 27ef83df): passed 13/13, **1.5347** vs board 1.4956. Content hash 613b6c2d...; the
  bytes are kept as `submission.edoff-27ef83df.py` for the 00:00Z scored run.
  - Against board run dc51d8e8 (each run vs its own SGLang time):
    - T <= 32 rows x0.958-x1.003, i.e. noise (v1 full's SGLang time moved 7% too);
    - v16 x1.075 / x1.034 and v32 x1.204 / x1.118 (the wgmma-hot hybrid).
  - Selections: v1 at T=1; v4 at T=2..32, 128 and 256 (the tie-break revert); H=-16 at T=128 / 256.
- **Engine D lost every GEMM selector**, so every cell kept its incumbent. Graph-timed, cold L2, incumbent / Engine D
  at T=1:

  | GEMM | incumbent us | Engine D us | ratio | incumbent rate |
  |---|---|---|---|---|
  | P | 19.6 | 21.1 | x0.923 | 3.85 TB/s |
  | qkv_gate | 11.2 | 13.9 | x0.804 | |
  | o | 4.85 | 9.1 | x0.530 | |

  - It gets worse with T (T=32: P x0.873, qkv x0.649, o x0.403, unpooler x0.686).
  - The incumbents already run at 3.4-3.9 TB/s. The 22-27 us P rows I planned from were profile rows that include
    PDL waits.
  - Engine D pays several us per launch: ramp, a fix-up per partial group, and T x-copies per stage. o is all
    overhead.
  - Speed-of-light headroom left in these four GEMMs: about 6 us per row in total. **Killed** (CAMPAIGN_ENGINE_D off,
    0a3d3f27).
  - Lesson: compare kernels only graph-timed in isolation, never on PDL-launched profile rows (PLAN §2 item 5, which
    I did not apply to the dense GEMMs).
- The expert-block anatomy print raised (`_ead_inputs` takes a host generator). It now runs from Engine A's decode
  setup and comes with the next decode check.
- dq3 (uncommitted, in audit) re-permutes each code word so that every pair comes out with a funnel-shift rotation
  plus AND; pair 3 takes two of each. Ladder-2's limiter is the consumer's INT32 work, and this cuts about 11 ALU ops
  per word to 9.
  - Found by an exhaustive search: four pure rotations cannot tile the word, and 9 is the minimum.
  - Verified on CPU over all codes with crosstalk.
  - One moe check at 00:00Z re-measures the ladder rungs and dq_exact on the device.

## 2026-10-26T20:20Z: user steer acknowledged (19:05Z quota policy); work moves to bf16_proj until 00:00Z

- Read at 20:05Z: the user's steer (relayed from the openclaw session) in INDEX.md, PLAN.md §6 / §11 and the campaign
  CLAUDE.md. My 18:49Z and 19:05Z Now rewrites had overwritten earlier copies of it unread.
  - The policy is now the first bullet of the Now block.
  - From now on, every Now rewrite starts from the current LOG.md block, not from a scratch copy.
- Applied: decode / moe have no checks until 00:00Z and their reset plan is written. Work moves to bf16_proj
  (12 checks left), then attention_decode.
- bf16 board 741072dc 1.2566 per load:

  | load | ours us | SGLang us | speedup |
  |---|---|---|---|
  | d1 | 44.8 | 59.5 | x1.33 |
  | d16 | 62.6 | 76.3 | x1.22 |
  | d32 | 64.5 | 76.9 | x1.19 |
  | v1 | 48.6 | 75.2 | x1.55 |
  | v16 | 72.6 | 86.5 | x1.19 |
  | v32 | 90.8 | 100.5 | x1.11 |

  - Today's 8 churn checks were all neutral (1.242-1.268). The only repeatable row win is pooler k-halves at m=128
    (22.9 -> 21.4 / 21.8 us).
  - The raw-byte floor caps us at about x2.03. The leaders (1.52-1.64) sit at about 80% of it. Lossless compression is
    the one lever that goes below the raw floors.
- Chosen: "bulk12". The weights are N(0, 1/k) bf16, so the high byte (sign + exponent bits 7..1) needs about 4 bits
  over a 16-binade window, and the low byte (exponent bit 0 + mantissa) stays raw. That is 12 bits per element.
  - Escapes are |w| < 2^-19 (about 0.014%). They decode to the window floor and are corrected exactly in the epilogue
    (two fp32 FMAs per token).
  - Decode costs about 2 integer ops per element, below Engine A's NVFP4 dequant.
  - The earlier lossless line (exp4, killed at 1.4-1.6 TB/s) ran 4 warps at 1 CTA/SM with escape groups.
- First target: the judge-proven m=1 row-owner bulk kernel (pooler A, qkv A2, o A3) with the codec in its consumer,
  behind the existing cell gate (canary, fp64 checks, routed only at <= 0.98 of master).

## 2026-10-26T20:57Z: bf16 bulk12 (lossless 12-bit weights) m=1: check 1.2699, scored 1.2664 (rank 381 -> 355), landed

- Code 4b79139c on campaign/expert-bandwidth, audited (no BLOCKER / MAJOR, verdict ship; MINORs applied).
  - Codec: each bf16 weight is L (raw low byte: exponent bit 0 + mantissa) plus a 4-bit code for H (sign + exponent
    bits 7..1) over the 8 most-used values. Escapes (~1.2e-4, at most 58 per CTA) decode to the window floor and are
    corrected exactly in the epilogue.
  - The m=1 row-owner bulk kernel decodes it in the consumer: 2 integer ops per element (IMAD + byte_perm), x resident
    in smem.
- check 474490c8258d4521: passed 9/9, **1.2699** [1.2508, 1.2889]. All three m=1 cells routed.

  | row | board us | check us |
  |---|---|---|
  | pooler m=1 | 17.72 | **14.8** |
  | o m=1 | 6.32 | 5.9 |
  | qkv m=1 | 10.54 | 10.7 (flat; 144 CTAs at 2/SM) |

  - pooler streams 50.3 MB at ~3.6 TB/s, i.e. **~4.8 TB/s raw-equivalent**. For the first time a bf16 row runs below
    its raw-byte floor rate.
- scored 3834ceeca7c0410d: **1.2664** [1.2474, 1.2854], same content hash. The bf16 board moves 1.2566 -> **1.2664**,
  rank 381 -> **355**. Landed on master.
- The 16 KB tail cut the per-instance gate lines. The next build re-prints them at the end of prepare.
- Port value for decode: the codec and decode step drop into decode's P (pooler + router), qkv and o kernels. pooler
  alone is ~67 MB per layer row at every T. Campaign #2 material once bf16 shows the m=8..32 rates.

## 2026-10-26T21:20Z: bf16 bulk12 at m=8/16/32: check 22ddaf4a 1.2683, no m >= 8 cell routed (per-row x TMA copies)

- check 22ddaf4a563b40cb (code 97aaf5a6, audited; MAJOR epilogue bank conflicts fixed before the run): passed 9/9 at
  1.2683.
- The m=1 cells routed again (gate: pooler 19.1 vs 22.0, qkv 12.2 vs 13.7, o 5.8 vs 6.7 net us).
- No m >= 8 cell routed:

  | cell | b12x vs incumbent (net us) | ratio |
  |---|---|---|
  | pooler m=8 | 26.4 vs 23.7 | 1.11 |
  | pooler m=16 | 40.4 vs 23.8 | 1.69 |
  | pooler m=32 | 70.6 vs 24.6 | 2.87 |
  | qkv m=8 | 16.7 vs 14.9 | 1.12 |
  | o m=8 / 16 / 32 | | 1.13 / 1.66 / 2.83 |
  | unpooler m=16 / 32 | | 1.63 / 2.89 |

- Diagnosis:
  - The extra time grows linearly with the number of per-row x bulk copies: m per chunk, 256 / 512 / 1024 per CTA.
  - That is about 30-50 ns per 512 B TMA copy, serialized in the copy engine.
  - CW 4 vs 8 makes no difference, so the kernel is not consumer-bound.
  - Lesson: never stream many small rows through cp.async.bulk. One TMA copy per chunk, LSU cp.async for small pieces.
- Fix built (d466a692, in audit): x pieces as 16 B cp.async from all 32 producer lanes, with
  cp.async.mbarrier.arrive.noinc on the slot's full barrier (count 1 + 32). The weight chunk stays on the TMA.

## 2026-10-26T21:40Z: bf16 cp.async x path: check 0acdca48 1.2670, m >= 8 still loses; router ticket check dispatched

- check 0acdca481ebc4b47 (03e8197d, audited: ship): passed 9/9 at 1.2670. The m=1 cells routed. No m >= 8 cell
  routed:

  | cell | b12x (net us) | incumbent (net us) | before cp.async (net us) |
  |---|---|---|---|
  | pooler m=8 | 25.3 | 23.4 | 26.4 |
  | pooler m=16 | 34.9 | 23.4 | 40.4 |
  | pooler m=32 | 52.8 | 24.1 | 70.6 |
  | unpooler m=16 / 32 | 28.5 / 41.1 | 22.4 | |
  | qkv m=8 | 17.1 | 14.8 | |
  | o m=8 | 8.0 | 6.6 | |

  - The per-row TMA cost is gone. ~1 us per token row remains, probably L2 hot-spotting: 128 CTAs read the same x lines
    in lockstep.
  - Next b12 step (a later bf16 window): x once per cluster (2D TMA tensor map + multicast), or a cluster-pair K split.
- Dispatched 7e8be36a (c891dc84): the router ticket path at m >= 9.
  - It drops the torch.zeros memset launch on the d16 / d32 / v16 / v32 router rows.
  - Self-check plus a graph gate per kernel.
  - Expected ~1 us per row, about +0.9% on the headline.

## 2026-10-26T21:45Z: user steer acknowledged (21:40Z expiring-quota sprint until 00:00Z)

- Read at 21:40Z in the campaign CLAUDE.md (relayed from the openclaw session): use as much of today's checks and scored
  runs as possible before 00:00Z.
  - Scope: ~30 hypotheses, bf16_proj first, then attention_decode, then prefill.
  - Run in parallel through the revived /hypothesize + /churnsubs pipeline, with ~8-10 concurrent agents.
  - For this window only it overrides "churn paused", "prefill idle", "<= 3 agents" and "one Opus audit per module".
  - Hard stop: dispatch nothing after ~23:45Z. Decode / moe code, quota and the 00:00Z reset plan stay untouched.
- Quota at 21:40Z:

  | problem | checks | scored |
  |---|---|---|
  | bf16 | 8 | 9 |
  | attention | 14 | 10 |
  | prefill | 16 | 10 |
  | moe | 0 | 4 |

- Started: three /hypothesize minting agents (bf16 6, attention 12, prefill 12). My bulk12 line continues alongside:
  - router-ticket check 7e8be36a is in flight;
  - the b12x chunk-rotation build a7da6fa9 is audited (ship) and waits for that result.
