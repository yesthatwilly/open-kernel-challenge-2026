# Log: campaign 2026-10-26-expert-bandwidth

The "## Now" block is rewritten by every loop tick (read it first; `head -40 LOG.md`). Entries below it are
append-only, newest last.

## Now

- updated: 2026-10-26T16:22Z
- step: S1 (v4) checks in flight; X2/D1 (v5) built (de1be708), syntax-clean on all three build variants, Opus audit
  running
- in flight: decode check ff2bd9182fa04636 and moe check 54f63952015c4d3b (v4, 666eb1d7); v5 audit
- next action: harvest v4 (score on improvement); apply v5 audit fixes; v5 attempt files + decode/moe checks
- quota left 10-26: decode 4 checks / 1 scored, moe 3 / 6
- boards: decode **1.4466** (rank 370 of 734), moe 1.5762, overall 376 of 1225

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
