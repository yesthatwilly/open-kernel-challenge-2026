# Log: campaign 2026-10-26-expert-bandwidth

The "## Now" block is rewritten by every loop tick (read it first; `head -40 LOG.md`). Entries below it are
append-only, newest last.

## Now

- **RESUMED by the user by hand at 2026-10-27T0003Z** ("run DISTILL.md", fresh session 860eee0a); the ~23:00Z STOP is
  retired.
- updated: 2026-10-27T0145Z (date -u); phase A; /loop /3to1 running
- **Operating policy: the 3:1 daily rhythm** (the user, 2026-10-27T0011Z; `campaigns/DAILY-RHYTHM.md`, `/3to1`, `/loop /3to1`):
  - 00:00Z until decode's reasonable checks are spent: phase A, the biggest decode rewrites on Opus 5.5 ultracode
    workflows (moe as testbed), built and checked;
  - then until 18:00Z: phase B, the other problems' minor gains (moe > bf16 > attention > prefill);
  - 18:00Z-24:00Z: phase C, use up every remaining check; `/hypothesize`; write tomorrow's phase-A plan by ~23:00Z;
  - always: a scored submit whenever a check beats a board; never re-roll identical bytes.
- **Today (10-27), phase A** (updated 2026-10-27T0145Z):
  - Boards: decode **1.5275** (rank 343; was 1.4956 / 365), moe **1.7076** (384), bf16 1.2664, attention 1.5347,
    prefill 1.4677.
  - Killed today: dq3 / dq4, H=-8, the k-slice consumer (lever #1), wg2 (the big-T hot-half re-split). Kept: v4c and
    b12p (P at T=1, layer x1.040).
  - **HOLD:** no decode scored run that carries b12p until the intermittent illegal address of 8e3c26d6 is
    diagnosed (agent running; fix branch campaign/eb-b12p-fix).
  - In flight: workflow wf_9ffcb970-fa3 (eat18 = T <= 8 expert-block streaming; dchain-pdl = qkv / o PDL weight
    prefetch), both on 8cfd91ce (b12p included).
  - Next: dispatch wave-2 checks when they land (one new cell per GEMM per check); the fault fix decides what can
    be scored. Phase B (bf16 readflush-gate, attention sk-full-line-gather) starts once decode's reasonable checks
    are spent. Phase C from 18:00Z.
  - Quota 01:45Z: decode 17 checks / 7 scored; moe 17 / 9; bf16, attention and prefill 20 / 10.

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

## 2026-10-26T21:58Z: user steer acknowledged (~22:00Z: the sprint is ended; keep what is in flight, then go quiet)

- Read at ~21:57Z in the campaign CLAUDE.md / PLAN / INDEX: "stop any more new sprints. Keep what's in flight."
- Starting nothing new: no minting, reviews, implementations, worktrees or subagents.
- In flight, finishing:
  - three /hypothesize minting agents (their files stay queued, unimplemented);
  - the adversarial review of attention `sliding-decode-sk-s3-fused-preattn-rope` (recorded, not implemented);
  - prefill confirm check 13cfaf1e25c341af;
  - my bf16 b12r check 6617f94dd1ce459f (resident-x m=8, audited, dispatched 21:56Z).
- Sprint so far (21:40-21:58Z): one sprint check used (prefill confirm). Reviews: prefill
  `experts-rega-mc4-cluster-high-ft` hold (its mc2 premise falsified); bf16 `combo-verify-cells-...` held by me
  (its only routable cell, K-halves kind (a), rounds fp32 partials to bf16: fails our fp32-accumulation rules gate).
  The moe submit picks were skipped (steer: moe quota untouched). Daily diagnoses skipped.
- bf16 BEST.json registered by hand to the landed bulk12 run (3834ceec 1.2664, blob 2bcfe7d4), noted in DECISIONS.
  Churn could not see it: it was submitted from a worktree copy.

## 2026-10-26T22:05Z: bf16 bulk12 beyond m=1 and the router ticket: three more checks, none beat the board

- Router ticket check 7e8be36a (c891dc84): 1.2650. Both gates lost (swap-AB m=16 x1.073, hilo m=128 x1.103 net): under
  graph replay the memset node is cheaper than a serialized last-arriver epilogue. **Killed.**
- Chunk rotation check 2980f57a (a7da6fa9, audited): 1.2690, same routes.
  - Rotation partly confirms the L2 hot-spot theory: qkv m=8 17.1 -> 14.5 net us (0.983 vs incumbent), pooler m=8 25.3
    -> 24.5 (1.02).
  - m=16/32 stay far behind.
- Resident-x m=8 check 6617f94d (b12r_kernel, audited): **1.2528**, below the board.
  - b12r pooler m=8 routed (gate 0.938) but moved the judged row only x1.016 (19.07 -> 18.68 us).
  - The rotated b12x qkv m=8 routed by a hair (0.976) and lost **x0.769** in the judge (12.86 -> 15.63 us).
  - **Lesson (gate vs judge):** _cell_gate flushes with a 256 MB write (flush.zero_()); the judge reads a buffer larger
    than L2. At m=1 the gate predicted the judge well (pooler 0.85 / 0.84). At m=8 it does not: the incumbent reads
    ~2 us faster in the judge and b12x ~1 us slower.
  - Next window: re-gate with a read flush and a stricter margin before routing anything at m >= 8.
- The bf16 board stays at the landed bulk12 m=1 (3834ceec 1.2664, rank 355). bf16 checks used today: 13 of 20 (8 by
  the churn before the campaign, 5 bulk12 + 1 router + 1 resident-x after 20:00Z).

## 2026-10-26T22:42Z: sprint summary (21:40Z start, ended by the user at ~22:00Z)

- Checks used by the sprint pipeline: 1 (prefill confirm 13cfaf1e, 1.4711, neutral). No wins, nothing landed, no scored
  runs.
- My bf16 line, which kept running alongside: 3 checks after 21:40Z (router ticket 7e8be36a 1.2650 killed; rotation
  2980f57a 1.2690; resident-x 6617f94d 1.2528). None beat the 1.2664 board. bf16 used 14 of 20 checks today (8 by the
  pre-campaign churn, 6 by bulk12 / router).
- Reviews (Opus):

  | hypothesis | problem | verdict |
  |---|---|---|
  | experts-rega-mc4-cluster-high-ft | prefill | hold (mc2 premise falsified) |
  | combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16 | bf16 | proceed narrowed; held by me (kind (a) bf16 partials fail the fp32-accumulation gate) |
  | sliding-decode-sk-s3-fused-preattn-rope | attention | proceed with builder notes; queued, not built |

- Minted and committed to the queues (churn intake 73a62c81), none implemented: 30 hypotheses.
  - bf16 6: pooler K-halves fp32 + Triton cast-add, b12 qkv m=1 R32, swapab16 5-stage, b12 unpooler slice, Lt NN
    cells, Lt m=256 config sweep.
  - attention 12: sk full-line gather, sliding-verify wgmma s2 split-combine, c320 one-pass combine and PDL trigger,
    combos.
  - prefill 12: rega rotated schedule, fine N buckets, fast silu epilogue, combos, front fusion, PDL regate, dynamic
    scheduler, routing fusion, LPT, anatomy diagnostic.
  - The minters flag the three queued prefill mc items as dead (mc2 gated off at 1.31-1.81x).
- Agent tokens, approximate: 3 minting agents ~1.6M, 3 reviews ~0.77M, 5 audits ~0.65M. About 3.0M in total, with
  only one check spent.
- Quota left at 22:42Z: bf16 6 / 9, attention 14 / 10, prefill 15 / 10, moe 0 / 4 (untouched).

## 2026-10-26T23:58Z: user steer acknowledged (~23:00Z: distil the context, then stop the loop); campaign paused

- Read at 23:41Z (date -u) in the campaign CLAUDE.md, PLAN.md and INDEX.md banners, relayed from the openclaw session.
  - The user's goals: capture the value in this session's context; the user then resumes the campaign by hand in a
    fresh session.
  - No judge actions: the 00:00Z reset plan is on hold.
- Done, with no okc check, submit or withdraw, no build and no hypothesis work (the three minting agents had finished
  and been queued at 22:42Z):
  - **DISTILL.md:**
    - the resume block: boards, ranks, quota, branches, HEADs, the kept bytes with their sha256 and source commits,
      and the reset plan as exact commands;
    - the killed lines and what would reopen them;
    - where the judge disagreed with the gates;
    - levers ranked by gain, effort and kill rule;
    - traps; what I would do differently;
    - the plan branches by outcome; the bulk12 decode port design; verdicts on the 33 queued sprint hypotheses;
    - operating lessons; a run ledger.
  - **`scratchpad/`:** the reset drafts, LOG tooling, run comparers, the bulk12 models, the Engine A emulators, the
    dq3 exhaustive check and the three sprint reviews, copied from /tmp (README.md lists them).
  - **FINDINGS.md:** ladder-2 rungs and the consumer-bound verdict, dq3 / dq4 / v4c / H=-8 rationale, the
    judge-weighted time shares by area (from `research/openclaw-relay-2026-10-26.md`), bulk12, gates vs the judge,
    and the tooling state.
- Boards read at 23:46Z (read-only):
  - decode 1.4956, rank 365 of 754 (1.5347 would be about 332);
  - moe 1.6851, 400 of 871;
  - bf16 1.2664, 370 of 1102;
  - attention 1.5347, 451 of 828;
  - prefill 1.4677, 179 of 486;
  - overall 370 of 1265.
- `okc problems` gives the close as Nov 4 11:59 pm PT, which is 11-05 07:59Z after DST ends on 11-01. The working
  deadline stays 06:59Z.
- The loop is stopped (ScheduleWakeup stop) and is not restarted.

## 2026-10-27T0003Z: manual resume; reset plan steps 1-3 dispatched

- The user resumed by hand ("run DISTILL.md"). Rules version 2026-10-22.1 unchanged; 20/10 quota fresh on every problem.
- decode scored 2bd373c9e23c4667: `submission.edoff-27ef83df.py` (sha256 9aad293f... verified vs git 27ef83df).
- moe check 9f8a518ea6c04d20 and decode check 0b085ef420c64ffb at eacf0722 (bytes = HEAD; cuda_syntax ok).
- archive.py SCRATCH_RUNS repointed to session 860eee0a's scratchpad (campaign fe240969).

## 2026-10-27T0011Z: the user's new standing policy, the 3:1 daily rhythm

- The text is in `campaigns/DAILY-RHYTHM.md`. Run it with the `/3to1` skill (`.claude/skills/3to1/`), or with
  `/loop /3to1` for an autonomous loop.
- The STOP banners in INDEX / PLAN / CLAUDE.md are replaced, and the 19:05Z quota steer is marked superseded.
- Today's phase-A plan is in the Now block.

## 2026-10-27T0013Z: day start 10-27 (phase A); /loop /3to1 started

- Boards 00:12Z: decode 1.4956 (367), moe 1.6851 (402), bf16 1.2664 (371), attention 1.5347 (451), prefill 1.4677 (179).
- Ultracode workflow wf_59d9138b-356 launched: the v4k rung (lever #1, rung first) and bulk12 P@T=1 (lever #5).
  Each is designed, built, audited twice and fixed if needed; the main thread dispatches the checks.
- X3 glue vs the dq layout waits on the harvest of 0b085ef4 / 9f8a518e.

## 2026-10-27T0031Z: reset-plan harvest (headlines); decode landed; two scored runs sent

- decode scored 2bd373c9e23c4667 (27ef83df bytes): **1.5218** (CI 1.499-1.545) > board 1.4956. Landed on master
  (4adf7b75, sha256 9aad293f...).
- decode check 0b085ef420c64ffb (eacf0722): **1.5484** (CI 1.525-1.572), +1.75% vs 1.5218 -> scored run 42b3e39478474bcd.
- moe check 9f8a518ea6c04d20 (eacf0722): **1.7112** (CI 1.686-1.737), +1.55% vs 1.6851 -> scored run cf937dd3656c4193.
  The moe T=128 gate kept master (H-16 x0.916, H-8 x0.867).
- The per-question harvest (VERDICT-dq, H=-8, v4c, anatomy) is with a harvest agent; its branch choice goes in the
  next entry.

## 2026-10-27T0035Z: reset harvest answers (agent, campaign 0ecf440e); DISTILL §4.1 branches chosen

- VERDICT-dq **fails**: best is dq3 x1.015 at C-R4 tok16 (dq4 x0.86-0.89). dq_exact clean (0 / 121,856; rotation counters
  0 / 0). Dequant stays dq2.
- H=-8 **dropped**: decode T=128 x0.982 vs H-16 x1.064, T=256 x1.157 vs x1.190; moe T=128 master, T=256 H-16. The
  threshold sweep is exhausted.
- v4c **kept**: decode big T +2% (T=128 1.093 -> 1.116, T=256 1.100 -> 1.123), ties at T=16 / 32, v4 kept at T=8. The
  moe per-class lines fell outside the 16 KB tail (ea4 at T=128 went 299 -> 251 us).
- Anatomy, T=1 / 8 / 16 / 32: block 27.4 / 61.8 / 135.7 / 152.4 us; no-route 5.2 / 8.8 / 9.3 / 9.4 us (19 / 14 / 7 / 6%).
  -> **consumer work before X3 glue.** T=1 / 8 under-stream at 2.1-2.7 TB/s (look at CTAs per expert).
- Anatomy, big T (H=-16): T=128 hot 86.9 us at 1.30 TB/s, cold 130.8 us at 2.81 (59%); T=256 hot 151.8 us at 1.31 TB/s,
  cold 134.2 (46%). -> **the wgmma hot path is the big-T lever**: phase-A build 3, launched now.
- Engine D off: rows within noise of f59cb301. Tooling: archive.py misses the profile glob of kept-file submits;
  rowcmp2 keys moe rows by batch.

## 2026-10-27T0047Z: phase-A workflow 1 built and checked

- Workflow wf_59d9138b-356 finished: 9 agents, ~1.24M tokens. Both builds audited twice; b12p's two MAJORs were fixed.
- moe check **e1406e5b05ba4cf7**: the k-slice rung (branch campaign/eb-v4k-rung f4280bd5; rungs ks8 / ks4, VERDICT-kslice,
  kslice_exact). Audit MINOR: a "KILL" with no exact, same-grid rung behind it is not a negative result, so read the
  per-entry /INEXACT and /GRID-DIFF tags first.
- decode check **dc3d5614d5e047c8**: bulk12 P at T=1 (branch campaign/eb-b12p-t1 e033fb35), gated with a read flush;
  the incumbent stays on any miss.

## 2026-10-27T0052Z: scored runs of eacf0722 land on both boards

- decode scored 42b3e39478474bcd: **1.5275** (CI 1.505-1.550), the new board (rank 342, was 367).
- moe scored cf937dd3656c4193: **1.7076** (CI 1.682-1.733), the new board (rank 384, was 402).
- master now holds the eacf0722 bytes for both (decode f97dded7..., moe 78e6ef0c...). Runs archived in runs/.

## 2026-10-27T0112Z: phase-A build 3 (wg2 hot half) checked

- Workflow wf_65a54a25-018 finished: 6 agents, ~0.86M tokens. Branch campaign/eb-wghot 3c450bed (code c17be13f) adds
  wg2, raced as H=-17 against H=-16 and master in both hybrids. Audit MAJORs fixed: the moe build deadline is now
  170 s, the log filter keeps 'worst ratio', and the canary poisons scratch first.
- Accepted risk: a first-launch device fault (no GPU or nvcc on this host).
- Open MINORs:
  - N bucket 16 is not hit by the canary;
  - no PDL at wg2's w13 -> w2 boundary;
  - H-17 must also clear RT_MARGIN over H-16 (a tie goes to the older path).
- moe check **c2224c010ea7454d** and decode check **74201da03ab64f40** dispatched.

## 2026-10-27T0114Z: k-slice rung killed; bulk12 P at T=1 wins its gate; scored run sent

- moe e1406e5b05ba4cf7: 1.6990 (diagnostic). VERDICT-kslice: ks8 tok8 x0.974, ks4 tok8 **x1.003** (0 spill, occ2), ks8
  tok16 x0.869 (72 B spill), ks4 tok16 x0.945 (56 B spill). Every rung is /INEXACT: kslice_exact is all NaN, a bug in
  the rung's out path (timing runs with out = null).
  -> **Lever #1 (k-slice consumer) is killed on timing.** Even the clean ks4 tok8 rung ties C-R4, so B-fragment loads
  are not the consumer's limiter. VERDICT-dq repeats: dq3 x1.000 / x1.019.
- decode dc3d5614d5e047c8: **1.5332** [1.510, 1.556] vs board 1.5275.
  - b12p<12,8,4>: P 19.62 -> 15.34 us (x1.277, read flush); T=1 layer 109.1 -> 104.6 us (x1.040, pdl=1).
  - Stream 58.98 MB vs 75.50 raw; escapes 1.4e-4; round trips bit-exact; canary, layer check and copy check pass.
  - The rest are unchanged (v4c at T=9..32 / 128 / 256).
- Scored run of dc3d5614's bytes: **8e3c26d66284496a** (a +0.37% gain, inside the spread; scored runs are plentiful).
- campaign/eb-b12p-t1 merged into campaign/expert-bandwidth (8cfd91ce). The next decode builds carry b12p.
- Next: harvest wg2 (c2224c01 moe, 74201da0 decode), then re-rank phase A. Lever #1 is gone; the candidates are T=1 / 8
  under-streaming (CTAs per expert), bulk12 qkv at T=1, and decode-chain.

## 2026-10-27T0131Z: wg2 killed on moe; phase-A wave 2 launched

- moe check c2224c010ea7454d: 1.6986. wg2 (H=-17) is correct (canary clean) but slow: T=128 x0.405, T=256 x0.882
  against H-16's x1.069. The gates kept H-16 / master, so wg2 is killed in this form (its work split; it has 6 units
  per CTA).
- decode wg2 check 74201da0 and b12p scored run 8e3c26d6 are still in flight.
- Workflow wf_9ffcb970-fa3 (base 8cfd91ce) has two builds:
  - **eat18**: T=1 / 2..8 expert-block streaming, from 2.55 / 2.67 TB/s net toward >= 3.3;
  - **dchain-pdl**: PDL static-weight prefetch for qkv_gate / o in the T <= 32 chain.
- Quota 01:30Z: decode 17 checks / 7 scored; moe 17 / 9; bf16, attention and prefill full.

## 2026-10-27T0136Z: wg2 killed on decode too

- decode check 74201da03ab64f40: 1.5176, below the 1.5275 board. H-17 (wg2) x0.920 at T=128 and x1.017 at T=256, against
  H-16 x1.050 / x1.156 and H-8 x1.052 / x1.161. The gate chose H-8 by a hair: an H-8 / H-16 near tie, flipping between
  runs. **wg2 is killed in this form.** The big-T hot half needs a different idea, not a re-split.

## 2026-10-27T0143Z: b12p scored run FAILED (intermittent illegal address); decode scored runs on hold

- 8e3c26d66284496a (the bytes of check dc3d5614, which passed): returncode 112. A sticky CUDA_ERROR_ILLEGAL_ADDRESS in
  prepare, reported asynchronously at _hc_anatomy's torch.Generator, after b12p PATH CHOSEN. Board unchanged (1.5275,
  rank 343).
- **Rule until it is diagnosed:** no decode scored run that carries b12p. Wave 2 builds on 8cfd91ce (b12p included),
  so harvest it with this in mind.
- A diagnosis agent is running (fix branch campaign/eb-b12p-fix if one is clear).
- End-game note: the final standings rerun the board bytes. Consider turning off the prepare-time diagnostics
  (anatomies, ladder) in the final bytes, so a diagnostic can never fail the rerun.

## 2026-10-27T0152Z: 8e3c26d6 diagnosed: a use-after-free in the _ead_anatomy diagnostic, not b12p; fix merged; check sent

- Cause (agent; static analysis):
  - `_ead_anatomy`'s "zero" iteration rebinds the "judge" inputs, and the graph keeps only raw addresses.
  - The next capture's empty_cache() can release that segment, so the judge replay reads unmapped memory.
  - Whether it faults depends on allocator layout, so it is random per prepare. b12p only changed the layout.
  - b12p is cleared: its final sync was clean in the failing run.
- **The board / master decode bytes (eacf0722) carry the same latent fault**, and so does the post-close rerun.
- Fix 7f74c074 (campaign/eb-b12p-fix): `keep` holds the inputs; each anatomy stage is synced and named. kernel_fn is
  unchanged. Merged into campaign/expert-bandwidth (7cdbcf17). Wave-2 branches (8cfd91ce) must merge it before any
  scored run.
- decode check **7844b6c903794ee7** (b12p + fix). If it is clean and beats 1.5275, send a scored run; that also makes
  the board bytes fault-free.
- **End game:** the final board bytes must carry this fix, or EAD_ANATOMY = False.

## 2026-10-27T0207Z: the anatomy fix runs clean on the judge

- decode check 7844b6c903794ee7 (7f74c074, b12p + fix): **1.5276** [1.505, 1.550], 13/13. Both anatomy stages printed
  "synced"; b12p routed (layer x1.038). The headline ties the board, so no scored run.
- The b12p HOLD is lifted for bytes that carry 7f74c074. The next decode scored run should come from a wave-2 build with
  the fix merged.

## 2026-10-27T0211Z: wave 2 built and checked (eat18, dchain-pdl)

- Workflow wf_9ffcb970-fa3: 10 agents, ~1.49M tokens. Both branches merged the anatomy fix 7cdbcf17. All audit
  findings were fixed.
- decode check **5b4e5b8f49d642c8**: eat18 (campaign/eb-eat18 86d64015), a new T <= 8 expert engine "eaf" raced in the
  per-T gates. The canary now covers the PDL overlap with a resident predecessor stand-in, and a final whole-layer
  check falls back to the incumbent at T=1 only on a miss.
- decode check **655304067cd74dfe**: dchain-pdl (campaign/eb-dchain-pdl eb690a25 bytes), o_pdl cells keyed by
  (predecessor path, batch, q_len). Only sliding T=1 can route in this check. Audit: the real headline gain is about
  half the builder's +0.6-0.8%.

## 2026-10-27T0217Z: phase B started alongside (two decode-portable items)

- attention check **00acb91008824b1e**: sk-full-line-gather (hyp/attn-skfull e4daaa4b, base master + v3chain patch,
  b4b20425's bytes at 1.5360). Full-line issue_tile under SK_FULLLINE, with canary + _sk_validate, else the half-line
  incumbent. No in-run timing A/B, so judge it on the sk profile rows (108.3 / 226.3 us at decode_16 / 32). The board
  entry is a78f29ae (1.5347); attention BEST.json is stale.
- bf16 check **a3dcd4cb46a84648**: the read-flush gate (hyp/bf16-readflush d7bf1c76, base 4b79139c). Every bulk12 gate
  routes on the read flush (0.98 at m=1, 0.95 at m >= 8), qkv gets R=32 one-CTA-per-SM variants, and a DECISIONS block is
  re-printed. Read it for the read vs write ratios (pooler / qkv / o) and the qkv decode_1 row (10.54 us).

## 2026-10-27T0233Z: wave 2 killed (eat18, dchain-pdl o); scored run of b12p + fix sent

- 5b4e5b8f (eat18): 1.5311. eaf at T=1 x0.991 vs v1 (margin 1.02), so v1 kept. Killed in this form.
- 65530406 (dchain-pdl): 1.5326. The isolated o_pdl ran > 1.3x the incumbent in every cell, so all were killed: the
  Triton swap-AB o at 4.5-5 us cannot be beaten by a CUDA PDL rewrite of this shape.
- In both, the anatomy stages synced clean. The T=128 hybrid fell back to master in both (H-16 x0.969 / x1.004; it won
  at x1.05-1.06 earlier), so the H-16 gate at T=128 is a near tie.
- **Scored run ba57a64f66864219** of 7844b6c9's bytes (b12p + fix; judged 1.5276 / 1.5311 / 1.5326 with inert extras).
  Above 1.5275, it also makes the board bytes fault-free.
- Phase A status: 4 of 6 levers killed today (k-slice, wg2, eaf, o_pdl); b12p landed. Next: a design panel re-ranks
  decode levers on today's evidence before any more builds.

## 2026-10-27T0244Z: attention +3.9% (full-line gather), scored run sent; bf16 read-flush calibration

- attention check 00acb91008824b1e: **1.5952** [1.571, 1.626] vs board 1.5347. The full-line gather routed (canary and
  self-check pass), on the v3chain base (b4b20425 1.5360). **Scored run bbf762ed52934bde.** decode_layer runs the same
  splitkv_attn_sk_kernel at decode_16 / 32, so this is a decode port candidate (wave 3).
- bf16 check a3dcd4cb46a84648: 1.2625, below 1.2664, so no submit. Read-flush vs write-flush gates at m=1: pooler
  0.833 / 0.828, qkv R16 **0.953** / 0.884, o 0.929 / 0.873. qkv R=32 fails (1.016). The write gate overstated qkv by
  7 points. A bulk12 qkv port to decode at T=1 is worth about 0.45 us (~0.05% headline): not worth a decode check alone.

## 2026-10-27T0252Z: design panel re-ranked decode levers; wave 3 launched

- Panel wf_1e5a2d61-92a (4 agents, ~0.62M tokens); verdict in research/wave3-panel-2026-10-27.md. Ranking (gain x p / h):
  1. SK_FULLLINE decode port (+0.7-1.0%, p 0.75; judged d32f sk 228 -> 190 us on attention);
  2. a wg hot-consumer diagnostic, then variant B (+1.1-2% if the hot half reaches x1.15+, p 0.25);
  3. x-copy rungs (+0.4-0.7%, p 0.35).
  Dropped: an eaf expert-major T=8 port. Parked: the hot / cold SM partition and SS warp specialization (too big today).
- Workflow wf_03f049d9-24a (base 7cdbcf17):
  - **W3-A** w3a-skfull, a decode check;
  - **W3-B** w3b-wgdiag, a moe diagnostic;
  - **W3-C** w3c-xcopy, a moe ladder diagnostic.
- d1s cluster port, norm fold, P T-scaling and the no-route fixed cost are held for tomorrow's phase A.

## 2026-10-27T0253Z: decode 1.5441 (b12p + anatomy fix) lands

- Scored run ba57a64f66864219: **1.5441** [1.521, 1.567], the new board (rank 331, was 343). master now holds 7f74c074's
  bytes (sha256 32c323be...), so the board bytes no longer carry the _ead_anatomy fault.

## 2026-10-27T0301Z: attention 1.5902 lands

- Scored run bbf762ed52934bde: **1.5902** [1.566, 1.614], the new attention board (rank 407, was 451). sk-full-line-gather
  + v3chain is landed on master; attention BEST.json was re-registered by hand (it was stale).

## 2026-10-27T0312Z: attention c320 pair checked

- attention check **508cfca7650848ec**: c320-combine-onepass + c320-pdl-early-trigger (hyp/attn-c320 0453ca70, base
  0d421faf at 1.5902). Each change has its own canary and an in-run A/B gate (head to head >= 1.01). Read the V3 / V4
  readouts and the 'compiled in' line (a build overrun drops FD1 to V0). attn-s2 is still building.
- 0313Z: attention check **becb6967be9940bf**: sliding-verify s2 split-combine (hyp/attn-s2 7cc48bef, base
  0d421faf). The bands B=2..16 / 17..32 route only at >= 1.05x in both rounds plus a replay check; B=1 is unchanged.

## 2026-10-27T0322Z: wave 3 built and checked

- Workflow wf_03f049d9-24a: 8 agents, ~1.2M tokens. No BLOCKER. W3-B has one MAJOR (no ptxas serialization report;
  diagnostic only).
- decode check **cc7d142af3fb4308**: W3-A, SK_FULLLINE in decode (campaign/eb-w3a-skfull, submission 4a9b6c2b...).
  Before dispatch I fixed two audit MINORs: the ticket guard now reads before zeroing, and the canary can no longer
  pass on a stale half-line buffer. Kill rule: judged d32f sk above ~205 us, or a d32f layer gate below 1.03.
- moe check **8263c7bca04a406a**: W3-B, the wg hot-consumer DIAG rungs (nothing routes). Build variant B only if it
  reaches >= x1.15 and is bit-exact.
- moe check **e775df93c6d64491**: W3-C, the x-copy rungs (nothing routes). Kill if x-static < x1.04 and x-cp.async
  < x1.03 vs C-R4 tok8.
- Attention checks 508cfca7 (c320) and becb6967 (s2) are in flight.
- 0327Z: attention c320 check 508cfca7: **1.6279** [1.604, 1.658] vs board 1.5902. ONEPASS combine routed
  (V4 19.20 -> 17.02 us, x1.128); the early trigger was not (readout x0.996). **Scored run 276ff0cda0404ef2.** This is
  the d1f decode port candidate: decode's c320 chain at T=1.
- 0334Z: attention s2 check becb6967: 1.5906, correct, but no band routed (B=16 x1.017 / x1.007 vs the 1.05
  bar; B=32 x0.978). The window kernel alone is 14.8 us against master's 24.4 us step, but the combine costs 7.5-10 us.
  s2 killed; the queued s2b fast-merge is built next (phase B).

## 2026-10-27T0335Z: wave 4 launched (c320 decode port at d1f)

- Decode has no c320 path. Its d1f still uses two-level attention. On the attention board, c320 V4 + ONEPASS runs
  decode_1 at 17.0 us against two-level's 23.8 us, so the port is worth ~6 us on the d1f row (~+0.5% headline).
- Workflow wf_40d4513d-fdc (base 7cdbcf17): 2 designs plus a judge at effort max, then build, 2 audits and fix at
  effort high (the user's new per-role effort). Branch campaign/eb-w4-c320.
- 0345Z: attention check **af08448e82c04fdf**: s2b fast merge (hyp/attn-s2b 9f428796 on c320 0453ca70). It races a PDL combine against an in-kernel ticket merge per band, at >= 1.05x in both rounds; the merge cost is printed per grid.

## 2026-10-27T0347Z: wave-3 results: W3-A real gain (scored run sent), W3-C killed, W3-B points to N=64

- **W3-A** cc7d142af3fb4308: 1.5403 (noise-low). Full-line routed at d16f (layer x1.019) and d32f (layer x1.088). sk
  alone at d32f 219 -> 182 us. Rows vs ba57a64f on their own SGLang: d32f **x1.077**, d16f x0.993, the rest x0.963-1.017
  (SGLang drift). Merged into campaign/expert-bandwidth (75bb6bf8). **Scored run 973509ebe8df4766.**
- **W3-C** e775df93: x-static x1.031, x-cp.async x1.003, so x-copy is **killed**. The compute-only ceiling at tok8 is
  x1.153.
- **W3-B** 8263c7bc: variant B x0.80, but ptxas serialized it (C7512 register shortage), so the overlap question is
  open. The ring does not bind. **VERDICT-N (forced N, INEXACT): hot T=128 102.5 -> 69.7 us at N64, T=256 160.1 -> 135.1
  us**, so N=64 hot tiles may be the big-T lever (exact-N buckets, panel P1 L3). This is a wave-5 candidate.
- 0347Z: wave 5 launched: workflow wf_df13b9f4-24c, exact N=64 / exact-N-bucket wgmma hot tiles for both hybrids (base 75bb6bf8, campaign/eb-w5-n64). Design and judge at max, build and audit at high. The judge may downgrade it to a diagnostic if N64 is not reachable exactly.
- 0357Z: attention scored 276ff0cda0404ef2 (c320 onepass): **1.6095**, the new board (rank 399). Landed on master; BEST.json registered.
- 0404Z: attention s2b check af08448e: **1.6247**. B=2..16 routed the PDL merge (x1.14 / x1.15; merge +2.9 us); B=17..32 just missed (x1.050 / x1.047). **Scored run a76ee9b03a9f4db4.** Phase-C idea: a B=17..32 bar of 1.03 with the IQR bound.
- 0407Z: decode scored 973509eb (W3-A full-line): 1.5272, below the 1.5441 board (rank now 334 through others'
  gains). The b12p + fix bytes have scored 1.5276-1.5441 across runs, so 1.5441 was likely a high draw. The d32f row gain
  (x1.077) is real. Not re-rolled; W3-A rides wave 4's decode build, and the next decode scored run should stack W3-A +
  c320 (+ N64 if it lands).
- 0411Z: bf16 check **670ad7c6b0da41ec**: pooler khalf m=128, b2 only (fp32 partials via bmm out_dtype=float32 + Triton add-cast; hyp/bf16-khalf 83e65c27 on the read-flush branch). Read-flush gate >= 1.03 in both rounds; judge it on the verify_16 pooler row (22.91 us).
