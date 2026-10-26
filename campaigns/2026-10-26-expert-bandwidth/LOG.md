# Log: campaign 2026-10-26-expert-bandwidth

The "## Now" block is rewritten by every loop tick (read it first; `head -40 LOG.md`). Entries below it are
append-only, newest last.

## Now

- updated: 2026-10-26T16:05Z
- step: X0b in flight; S1 (Engine A v4: v3 on 8 consumer warps, 2-record stages, D=2) built and syntax-clean,
  Opus audit running
- in flight: decode check 523252767fc84a26 and moe check 6eb226e9db4f4876 (X0b, d25a987d); v4 audit
- next action: harvest X0b (score on improvement); apply v4 audit fixes; v4 attempt files + decode/moe checks
- quota left 10-26: decode 5 checks / 2 scored, moe 4 / 6
- boards: decode **1.4466** (rank 370 of 733), moe 1.5762 (428 of 840), overall 375 of 1223

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
