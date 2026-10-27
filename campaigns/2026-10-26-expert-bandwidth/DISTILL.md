---
campaign: 2026-10-26-expert-bandwidth
status: paused for a manual resume by the user
written: 2026-10-26T23:55Z (date -u; state read at 23:46Z)
by: session 3734a126, which ran the campaign from 15:35Z until the user's ~23:00Z stop steer
---

# DISTILL: the expert-bandwidth campaign, handed to a fresh session

At ~23:00Z the user asked, through the openclaw relay banners, for this session's context to be captured. The
campaign then pauses until the user resumes it by hand in a new session. Everything below was checked against the
repo, git and okc while writing. Tags follow FINDINGS.md: **[M]** judge-measured, **[P]** prepare-time, **[I]**
inference.

**Read order for a cold session:**
1. §1 (state and the first actions);
2. LOG.md (the entries from 15:35Z to 22:42Z; the Now block says PAUSED);
3. FINDINGS.md;
4. §3 to §5 of this file when choosing work;
5. PLAN.md §4 and §9 for the original design.

## 1. Resume block

### 1.1 State at 2026-10-26T23:46Z

- **The loop is stopped. Nothing is in flight:** no okc job, no agent. The last judge action was bf16 check 6617f94d,
  dispatched about 21:56Z and harvested.
- **Boards** (`okc leaderboard <p> --json`; our row is the `you` field, team `yesthatwilly`):

  | board | ours | rank | leader | rank 300 / 200 / 100 | our board run | master's file |
  |---|---|---|---|---|---|---|
  | **decode (KPI)** | 1.4956 | 365 of 754 | 2.3797 | 1.5704 / 1.7331 / 1.8851 | dc51d8e8f58e4cb7 | bytes of 4f530b95 (sha256 bd64be82...) |
  | moe | 1.6851 | 400 of 871 | 2.6196 | 1.8075 / 1.9335 / 2.1292 | e854db4683dc495f | bytes of 4848aadc (7f8280f0...) |
  | bf16 | 1.2664 | 370 of 1102 | 1.6367 | 1.2963 / 1.3658 / 1.5086 | 3834ceeca7c0410d | bytes of 4b79139c (blob 2bcfe7d4) |
  | attention | 1.5347 | 451 of 828 | 3.0316 | 1.762 / 1.9098 / 2.1224 | (churn) | |
  | prefill | 1.4677 | 179 of 486 | 2.0252 | 1.2702 / 1.4373 / 1.5961 | (churn) | |
  | **overall** | | **370 of 1265** | | | | |

- **Decode speedup to decode rank today:**

  | speedup | 1.5097 | 1.5347 | 1.55 | 1.60 | 1.65 | 1.70 | 1.80 | 1.90 | 2.0 |
  |---|---|---|---|---|---|---|---|---|---|
  | rank | 353 | 332 | 316 | 286 | 260 | 222 | 141 | 96 | 54 |

  The overall board orders teams with a decode run by decode speedup (FINDINGS §1), so only decode moves our overall
  rank. At a fixed score we lose about 10 places a day.
- **Quota:**
  - The 10-26 leftovers expire unused at 10-27 00:00Z: bf16 6 checks / 9 scored, attention 14 / 10, prefill 15 / 10,
    moe 0 / 4, decode 0 / 0.
  - Every UTC day from 10-27 gives each problem 20 checks and 10 scored runs.
- **Close:** `okc problems` says "Kernel submissions close Nov 4 at 11:59 pm PT".
  - US DST ends on 11-01, so that is 11-05 07:59Z (PST).
  - The campaign has planned for 06:59Z (PDT). Keep 06:59Z as the working deadline, with the last dispatch at
    ~06:30Z.
  - There is no freeze: the user rejected one.

### 1.2 Code: branches, worktrees, HEADs

- **master** is 30a78b4a plus the distill commit. It holds the scored decode / moe / bf16 files above, the campaign
  folder and the churn queues.
- **`campaign/expert-bandwidth`** is at HEAD 796997d9 in `/home/user/git/musespark/.worktrees/expert-bandwidth`. The
  worktree is clean apart from 8 untracked kept-bytes files (below). Commits:

  | commit | content | judged |
  |---|---|---|
  | c8ac9beb | X0a: judge-shaped gate inputs | decode check 1.4524, scored a03e596d 1.4466 |
  | d25a987d | X0b: R skips its align; combine prefetch | decode check 1.4569, scored 1.4401 (noise) |
  | 666eb1d7 | Engine A v4 (8 consumer warps) | moe scored 35b55c74 1.6389; decode check **faulted** (ff2bd918) |
  | 4f530b95 | v4/v5 diagnostics; K1 plain | **decode board dc51d8e8 1.4956 = master's decode** |
  | 4848aadc | moe wgmma-hot hybrid H=-16; v5 off | **moe board e854db46 1.6851 = master's moe** |
  | 3bcb685b | decode wgmma-hot hybrid H=-16 | decode check a1bc1756 1.5097 |
  | 8b098599 | engine gates back to the best worst ratio (`prefer_within` 0) | |
  | 1cec46df | ladder-2 | moe check 5eafb39e (diagnostic) |
  | **27ef83df** | Engine D (it lost every selector, so it is inert) on top of the tie-break revert | **decode check f59cb301 1.5347, to be scored** |
  | 0a3d3f27 | Engine D off; anatomy print fix | |
  | 338cf9fe | dq A/B rungs (dq3 rotation, dq4 IMAD.WIDE); expert-block anatomies | |
  | 48e90d60 / 731d149e | hybrid H=-8, decode / moe | |
  | **eacf0722** | v4c (cost-weighted stream-K) | **not yet checked: the reset's moe and decode checks** |
  | e589eac2..796997d9 | bf16 bulk12 only, plus one line in the torch stub | bf16 checks (Appendix A) |

- **Bytes of the two reset checks.** The generated decode and moe files are byte-identical at eacf0722 and at HEAD:
  decode sha256 f97dded787370379..., moe 78e6ef0cae087073.... The steer's "731d149e" is the moe H=-8 commit. The build
  that carries dq A/B + H=-8 + v4c is eacf0722, and HEAD has the same bytes for these two problems.
- **Kept bytes.** These files are untracked in the worktree. Each one is reproducible with
  `git show <commit>:<problem>/submission.py`; every sha256 was verified between 23:41Z and 23:46Z.

  | file | commit | sha256 (16) | judged |
  |---|---|---|---|
  | `muse_decode_layer/submission.edoff-27ef83df.py` | 27ef83df | 9aad293f04305aca | check f59cb301 1.5347; okc content hash 613b6c2d... |
  | `muse_decode_layer/submission.wghyb-3bcb685b.py` | 3bcb685b | f69caddd6650a5fc | check a1bc1756 1.5097 |
  | `muse_decode_layer/submission.pw0-8b098599.py` | 8b098599 | 493a4604c306ddb2 | never checked (superseded by 27ef83df) |
  | `muse_decode_layer/submission.v4v5-4f530b95.py` | 4f530b95 | bd64be8285c42489 | board dc51d8e8 1.4956 |
  | `muse_decode_layer/submission.x0a-c8ac9beb.py` | c8ac9beb | 850e9557e9212b8a | scored a03e596d 1.4466 |
  | `muse_decode_layer/submission.x0b-d25a987d.py` | d25a987d | 3915255389592f6f | scored a009b6c8 1.4401 |
  | `muse_moe_w4a16/submission.hmwg-4848aadc.py` | 4848aadc | 7f8280f07a1b10be | board e854db46 1.6851 |
  | `muse_moe_w4a16/submission.v4-666eb1d7.py` | 666eb1d7 | e191903424ccf439 | scored 35b55c74 1.6389 |

- **Other worktrees stay** (PLAN §11: nothing is deleted): decode-first-rewrite, engine-b, engine-b2, ea-v3, ea-v3big,
  dense-port, hotcold, moe-hybrid, and the churn worktrees `hyp/*`.

### 1.3 First actions, in order

> **2026-10-27:** steps 1-3 were dispatched at 00:02-00:03Z. From now on, work follows `campaigns/DAILY-RHYTHM.md`
> (`/3to1`), and §4.1's build order runs as phase-A ultracode workflows. (the 00:00Z reset plan, on hold until the new session)

**One-time setup in the new session:**
- `rewrite/bin/archive.py` hardcodes `SCRATCH_RUNS` to this session's /tmp scratchpad
  (`/tmp/claude-1000/-home-user-git-musespark/3734a126-.../scratchpad/runs`). Point it at the new session's
  scratchpad before archiving. `dispatch.py` has no such path.
- The attempt templates are in `scratchpad/` (see `scratchpad/README.md`). Fill STAMP with
  `date -u +%Y-%m-%dT%H%MZ` and keep the ``**Job:** `<job id>` `` placeholder.
- Re-read the boards and `okc problems` (both read-only).

**1. Decode: the scored run of check f59cb301's bytes.** This is pre-approved: the check beat the board by 2.6%.
- Verify the bytes: `git -C .worktrees/expert-bandwidth show 27ef83df:muse_decode_layer/submission.py | sha256sum`
  must give 9aad293f04305aca..., and the untracked `submission.edoff-27ef83df.py` must match it.
- Attempt file: `muse_decode_layer/attempts/<STAMP>-submit-campaign-wghyb-pw0-decode.md` in the worktree, from
  `scratchpad/submit_attempt.md`.
- Send it with
  `okc submit /home/user/git/musespark/.worktrees/expert-bandwidth/muse_decode_layer/submission.edoff-27ef83df.py --no-wait --json`.
  `dispatch.py` only sends `<problem>/submission.py`, so submit the kept file directly and write the job id into the
  attempt by hand. okc should report content hash 613b6c2d....
- Expected result: 1.51-1.55. The same bytes score within about ±1.2%.
  - **Above 1.4956:** copy the file to master as `muse_decode_layer/submission.py`, add LOG and DECISIONS notes,
    re-read the board (about rank 332 at 1.5347), and archive the run.
  - **Below the board:** do not re-roll the same bytes. The decode check in step 3 becomes the scored candidate.
- These bytes still build Engine D. Its selectors re-run in prepare; they lost by 7.7% or more, so a flip is very
  unlikely.

**2. moe check at eacf0722**, the worktree HEAD's generated `muse_moe_w4a16/submission.py`.
- Attempt from `scratchpad/dqab_attempt.md` with COMMIT = eacf0722, then run
  `python3 rewrite/bin/dispatch.py check muse_moe_w4a16 <attempt>` from the worktree.
- Questions:
  - VERDICT-dq: dq3 / dq4 against dq2 at C-R2 and at C-R4 with 8 and 16 tokens.
  - dq_exact with crosstalk filler: production must show 0 mismatches and the rotation counters 0 / 0.
  - H=-8 against H=-16 and master at T=128 / 256.
  - v4c per class.
- `CAMPAIGN_LADDER` is True at eacf0722, so moe prepare runs the ladder (about +10 s).

**3. Decode check at eacf0722.**
- Attempt from `scratchpad/anat_attempt.md`, with BOARD = the decode board after step 1.
- Questions:
  - H=-8 against H=-16 at T=128 / 256;
  - v4c per class;
  - the anatomy: at T=1 / 8 / 16 / 32, the block against the same launches with no local route; at T=128 / 256, the
    hybrid whole, its hot half and its cold half;
  - with Engine D off, rows stay within noise of f59cb301.
- If its headline beats the (new) board with real row gains, send a scored run of the same bytes.

**4. Harvest.** Use `rewrite/bin/report.py RUN --base BASE` for the PATH / VERDICT lines and `scratchpad/rowcmp2.py`
for rows against each run's own SGLang time. Branch per §4.1.

**5. Then §4.2 (the bulk12 decode port) and §4.1's consumer work.** bf16 checks are the fast testbed for anything
bulk12.

## 2. Scratchpad files copied into the repo

They are in `scratchpad/` here, and `scratchpad/README.md` lists each one with its use:
- the three reset drafts (submit / dqab / anat);
- the LOG tooling (`now.md`, `log_now.py`, with the trap noted);
- the run comparers (`rowcmp2.py`, `bfcmp.py`), `hypsum.py` and `sim_routing.py` (a CPU model of the judge's routing);
- the bulk12 section source and its numpy models (`bulk12/`);
- the Engine A layout and map emulators (`emu/`);
- the exhaustive dq3 check (`dqcheck/`);
- the three sprint reviews (`reviews/`).

Everything else in the old /tmp scratchpad is in git (applied edit scripts, diffs, generated files), in `runs/`
(archived run records) or stale (leaderboard snapshots).

## 3. Judgment that existed only in this session's context

### 3.1 Killed or parked lines: why each died, and what would reopen it

| line | died because | evidence | reopen only if |
|---|---|---|---|
| **v5 = D1** (32-route mma.sync passes on 256-k records) | tensor-issue bound (~0.25 HMMA/clk/SM against a 0.31 floor); half-size stages double the per-stage overhead with less in flight | [M] moe e403a42a: kernel-only 2.39-2.90 vs v4 2.92-3.21 TB/s; big gates x0.86-0.90 | never on mma.sync. Wide N belongs on wgmma, which the hybrid's hot half already does. |
| **Engine D** (dense bf16 skinny GEMM on v4's CTA, stream-K) | the incumbents already run at 3.4-3.9 TB/s graph-timed; D pays a ramp, a fix-up per partial group and T x-copies per stage; o is all overhead | [M] f59cb301: P x0.923, qkv x0.804, o x0.530 at T=1, worse with T | not as a raw-byte engine. Past these incumbents the lever is fewer bytes (bulk12, §4.2), or a launch hidden behind its predecessor in decode-chain's PDL chain. |
| **router ticket** (bf16, no memset at m >= 9) | under graph replay, a memset node costs less than a serialized last-arriver epilogue | [M] 7e8be36a: swap-AB m=16 x1.073, hilo m=128 x1.103 slower (net gate) | no |
| **b12x at m >= 8** (x streamed with each chunk) | first, per-row TMA copies (30-50 ns each, serialized); after the cp.async fix, ~1 us per token row remains, from L2 hot-spotting (every CTA reads the same x lines in lockstep) | [M] 22ddaf4a, 0acdca48, 2980f57a; rotation took qkv m=8 to gate 0.983 | x is read once per cluster (TMA multicast over a 2-4 CTA cluster: 2D tensor map, one box per chunk), **and** it passes a read-flush gate at <= 0.95 |
| **b12r** (resident x, m=2..8) | pooler m=8 routed (gate 0.938) but judged only x1.016 (19.07 -> 18.68 us). The check as a whole lost because b12x qkv m=8 routed in the same run. | [M] 6617f94d 1.2528 | only inside a stack, alone among the m=8 cells, under a read-flush gate. Worth under 0.1% on bf16. |
| **`prefer_within` 0.01** (earliest engine within 1%) | put T=9..16 on v3, which is 3-5% slower than v4 in every judged run | [M] a1bc1756: d16 -4..-6% | no. The best worst ratio wins (`EA_PREFER_WITHIN = 0.0`). |
| **moe hybrid H=-16 at T=128** | few hot experts at T=128 | [M] 0f4605dc x0.970 | H=-8 is the test (reset step 2) |
| **Marlin-hot hybrid at T=256** (H=-48 / -32) | superseded by the wgmma-hot half (H=-16) | [M] decode v32 rows +11-16% with wgmma | no |
| earlier: **Engine B** v1/v2, **exp4** (the earlier lossless line: 1.4-1.6 TB/s at 4 warps and 1 CTA/SM, with escape groups), K2 on 1 CTA/SM, R -> K1 PDL | see the decode-first RETRO.md | | not in their old form. bulk12 is the working 12-bit form. |
| **sprint:** prefill mc4 / mc2 / bm256 slab multicast | the ingest-cap premise is falsified: mc2 cut issued L2 bytes 24-39% and ran 1.31-1.81x slower | [M] 0ca60bb7, a6f60630; review hold | a stage anatomy (prefill `experts-rega-stage-anatomy-diag`) shows ingest-bound stages |
| **sprint:** bf16 combo (khalf + qkv row-owner m8/16 + Lt) | its qkv cells never routed; its one robust cell, K-halves kind (a), rounds fp32 partials to bf16 (lower precision than SGLang's fp32 split-K reduce: a rules gray area) | review | superseded by `pooler-khalf-m128-fp32-partials-triton-castadd`, candidate (b2) only |

### 3.2 Where the judge disagrees with prepare-time gates

1. **bulk12's write-flush gate (the biggest disagreement).**
   - The bf16 cell gates flush L2 by writing: `flush.zero_()` on 256 MB in `muse_bf16_proj/submission.py`
     (campaign branch), lines 1608, 1647, 2786 and 2928; `_cell_gate` is at 2768.
   - The judge reads a buffer larger than L2. A write flush leaves ~50 MB of dirty lines, the timed kernel pays to
     write them back, and a byte-light kernel evicts fewer of them. So the gate flatters bulk12, worst at m >= 8 and on
     weights of L2-like size:

     | cell | gate | judge |
     |---|---|---|
     | pooler m=1 (67 MB raw) | 0.82-0.85 | 0.84 |
     | qkv m=1 (37.7 MB) | 0.88 | ~1.00 |
     | o m=1 (16.8 MB) | 0.87 | 0.97 |
     | b12x qkv m=8 | 0.976 | **x0.769** (12.86 -> 15.63 us) |
     | b12r pooler m=8 | 0.938 | 0.984 |

   - **Rule:** every gate flushes with a READ. `rewrite/lib/rt.py` `_rt_time(flush="read")` already does, so the
     decode and moe gates are fine. The bf16 gates need `flush_mode="read"`, which the queued
     `b12-qkv-m1-r32-one-cta-per-sm-readflush-gate` adds, and a margin of <= 0.95 at m >= 8.
   - **My mistake:** the same lesson is in decode-first FINDINGS 3.2 and in rt.py's header. I copied section 9's
     write-flush gate into bulk12 without applying it.
2. **Engine D: planning numbers versus the gate.**
   - I planned from decode profile rows (P 22-27 us at 2.8-3.4 TB/s) that include PDL waits. Graph-timed in
     isolation, the incumbent P was 19.6 us at 3.85 TB/s.
   - The gates were right and the plan was wrong.
   - **Rule:** plan only from graph-timed, isolated kernel times (FINDINGS §3.6, PLAN §2.5).
3. **The tie-break.** Prepare gates at T <= 32 carry ±1-3% noise. A deliberate 1% preference for the earlier engine
   picked the engine that is 3-5% slower in every judged run. **Rule:** no preference toward older engines; the best
   worst ratio wins.
4. **Big-T draws.**
   - moe v4 at T=256: the gate's p50 said x1.12, but the judged row moved only +1.3%. The judge's T=256 routing sits
     near our p90 draw.
   - **Rule:** decide big-T gates on the worst (p90) draw, as `_rt_select` does over every routing.
5. **Near ties at T=2..8.**
   - v1 / v2 / v3 / v4 sit within ~3% at T=2..8 and flip between runs.
   - v1 sliding rows vary ±5% with the same engine (212cdf5d: T=8 flipped to v4 and the row read +5.3% while the
     profile showed v4 faster).
   - Do not attribute row moves under 3% at T <= 8.
6. **The scored spread.** Identical bytes score within about ±1.2% (X0b: check 1.4569, scored 1.4401). A check under
   ~1.2% above the board can score below it.
7. **Two routable novelties in one check.** If one of them loses in the judge, the check cannot be submitted even when
   the other wins (6617f94d). Give each new routable cell its own check, or gate each so tightly that it cannot route
   on noise.

### 3.3 Levers ranked by expected decode gain, effort and kill rule

Judge-weighted decode time shares, from check f59cb301 (`research/openclaw-relay-2026-10-26.md` §1):

| area | headline share | range by load |
|---|---|---|
| moe expert block | 53% | 35% d1 to 62% v16 |
| dense GEMMs | 26% | 44% d1 to 18% v16 / v32 |
| attention | 15% | |
| small ops | 3% | |
| profiler "... N more" | ~2% | |

The headline moves by about share x cut, i.e. `1 / (1 - share x cut)`.

| # | lever | expected headline | effort | kill rule |
|---|---|---|---|---|
| 1 | **v4 consumer: k-slice B-fragment reuse.** Each warp takes a k-slice of all 128 rows, so one ldmatrix B fragment feeds 8 m-tiles (B loads / 8), with one cross-warp reduction per unit. It attacks ladder-2's C-R4 drop (-8% at 8 tokens, -14.8% at 16). | x1.02-1.04 (v4 at T >= 16 and the cold halves +5-12%) | 0.5 day for a ladder rung (one moe check), then 1-2 days for engine "v4k" + audit | the rung gains < 1.05x over C-R4, or the whole-layer gate < 1.02 after 2 checks |
| 2 | **Anatomy-driven glue at T <= 8** (X3: combine -> K2 epilogue, R -> K1 prologue, tighter PDL chain). The d1 profile suggests ~45 us of expert block against ~17 us at ring rate; the reset anatomy measures it. | x1.01-1.03 if the no-route launches are >= 10 us at T=1 / 8 | 2-4 days (decode-chain) | each piece must win its whole-layer selector |
| 3 | **Big T:** H=-8 and v4c (built; answered at the reset), then whichever hybrid half the anatomy names | x1.01-1.03 | 1-2 days after the reset | per-class gates; a class whose whole-layer gate stays under 1.03 keeps its current path |
| 4 | **dq3 / dq4 dequant**, only if VERDICT-dq shows >= 5% at C-R4 | x1.005-1.015 (dequant is ~9% of the consumer's rung cost) | 0.5-1 day (repack permutation + consume) + audit | VERDICT-dq < 5%, or any dq_exact mismatch |
| 5 | **bulk12 in decode's dense GEMMs.** P at T=1 first, then qkv R32 at T=1. T=16 / 32 only after multicast x wins on bf16. | P@T=1 about +0.3-0.5%; all dense at T=1 ~+0.7%; dense at T <= 32 +1.5-3% only if m >= 8 works | P@T=1: 1 day + audit; m >= 8: 2-3 days | the bf16 read-flush gate must show <= 0.95 first; decode gate < 1.02 turns it off |
| 6 | **Attention ports** (sk full-line gather at decode_16/32; c320 combine and PDL for d1f; sliding-verify wgmma split-combine for v16/v32; the d1s cluster) | x1.005-1.015 | from the attention-board hypotheses (§4.4) | per attention check |
| 7 | small ops (norm_residual and the rest), halved | x1.017 ceiling | | |

**Ceilings** (relay): moe cut to x0.6-0.7 gives x1.19-1.27; everything outside moe stacked gives about x1.09;
everything stacked gives about x1.30, i.e. ~2.0 decode. These are ceilings, not forecasts. Realistic for the next
3-4 days: 1.53 at the reset (rank ~335) -> 1.58-1.65 (rank ~300 -> ~260).

### 3.4 Traps and dead ends not to retry

- **Many small cp.async.bulk (TMA) copies** cost about 30-50 ns each, serialized in the copy engine. Use one TMA copy
  per chunk; move small pieces with LSU `cp.async` plus `cp.async.mbarrier.arrive.noinc` (full barrier count 1 + 32).
- **Write-flush gates** (§3.2.1).
- **Planning from PDL-inflated profile rows.** Combine shows 22-41 us in profiles against 3-4.5 us real; P shows
  22-27 against 19.6.
- **Ticket epilogues that only remove a memset**, under graph replay.
- **Wide mma.sync passes**, N >= 32 per A fragment: tensor-issue bound.
- **Raw-byte dense engines** against incumbents at 3.4-3.9 TB/s. The four T <= 32 GEMMs have ~6 us of speed-of-light
  headroom per row in total.
- **bf16 partials in K-splits.** The rules require fp32 accumulation, and SGLang's split-K keeps fp32.
- **Tie-break preferences in selectors**, and **re-rolling identical bytes** for noise.
- **PDL on v4's K1 at T=2.**
  - It caused the unexplained illegal-address fault in ff2bd918's copy check, which poisoned the context and failed
    the run.
  - K1 of v4 runs plain since then; the PDL gate measured no gain there anyway.
  - Any new kernel's first run is a check, with a canary and per-launch prints.
- **The 16 KB judge log tail.** Every decision line must be re-printed at the end of prepare. Decode and bf16 do this;
  keep it for every new section.
- **Agent sprints when checks are the bottleneck.** 21:40-22:42Z spent ~3.0M agent tokens for one check.
- **Churn's BEST.json for decode and moe is stale** (decode 1.3142 a903b9d0, moe 1.4021 31c27e4b). Campaign runs
  submitted from worktree copies are invisible to churn. Register them by hand before any churn run on those
  problems, as was done for bf16 at ~22:00Z.
- **Untracked kept bytes and /tmp drafts** are lost with the session. Commit them, or record the regeneration command
  at once.

### 3.5 What I would do differently

- **Run the diagnostic first.**
  - Ladder rungs and graph-timed isolated anatomies cost one moe check each.
  - I built v5 before the ladder measured the HMMA rate that killed it.
  - I built Engine D (~2 h plus a decode check) before timing the incumbents in isolation.
  - The next big build (v4k) should go rung first.
- **Use read-flush gates from the first cell.** The lesson was already written down in rt.py.
- **Delegate execution.** From 21:00 to 21:46Z this context read 83M cache tokens at ~627k tokens per call, nearly all
  of it for builds, greps over 300-900 KB generated files, status polls and edits. Brief subagents for builds,
  audits and harvests, and keep the main thread for decisions and the LOG. Start each day in a fresh session from
  LOG / DISTILL.
- **Read the Now block before rewriting it.** I templated it from /tmp and overwrote a user steer twice (18:49Z,
  19:05:50Z).
- **Keep drafts and kept bytes in the repo from the start.**
- **One new routable cell per GEMM per check.** That keeps attribution clean and keeps the check submittable.
- **Keep:** checks that each answer 3-4 questions through self-gated candidates (the reset checks do), PATH CHOSEN
  re-prints, attempt files before dispatch, and the per-row compare against each run's own SGLang time.

## 4. Hardened plan for the next work

### 4.1 The rest of expert-bandwidth: branches by outcome

After reset steps 1-3:

- **VERDICT-dq**
  - dq3 or dq4 >= 5% over dq2 at C-R4 (8 and 16 tokens) **and** dq_exact 0 / 0: build the production layout change
    (the repack permutation in Engine A's core plus consume() in ea4, and in v1-v3 if shared) behind a flag, audit
    it, then check moe, then decode.
  - < 5%: dequant stays dq2; go to the k-slice consumer.
  - Any dq_exact mismatch: that rung's extraction is wrong (production is unaffected). Note it and drop dq.
- **H=-8 (moe T=128 / 256; decode T=128 / 256)**
  - If it wins, the gates select it. If the moe check then beats 1.6851, send a moe scored run; it is a testbed board,
    but cheap.
  - If it loses everywhere, keep H=-16. The threshold sweep (-8 / -16 / -32 / -48) is then exhausted.
- **v4c**
  - If it wins classes at T >= 16 or big T, the gates keep it.
  - If it ties, the two-tile tail is not the limiter. Leave the bits off to save prepare time.
- **Anatomy (decode)**
  - **T=1 / 8:** if the no-route launches are >= 40% of the block (or >= 10 us at T=1), do X3 glue before consumer
    work. If the streaming part is below ~3 TB/s, the K1 / K2 split at T=1 under-streams; look at CTAs per expert.
  - **T=16 / 32:** if the block already streams >= 3.2 TB/s, consumer gains are capped near 10%; weigh them against
    glue and dense.
  - **T=128 / 256:** if the cold half (v4) is >= 60% of the hybrid, the k-slice consumer helps big T too. If the hot
    wgmma half dominates, tune the wgmma path (chunks at N up to 256, two CTAs per hot expert, or the threshold).
- **Build order after the harvest:**
  1. The **k-slice rung** in the ladder (`ea_ladder.cuh` / `.py`), as one moe diagnostic check.
  2. If it clears 1.05x: engine **v4k**, raced in every gate (T classes and the cold halves), audited; moe first, then
     decode.
  3. Use the spare decode / moe checks on the dq layout if VERDICT-dq passed.
- **Close the campaign** when X1 / X2 are landed or killed (target ~10-29/30):
  - merge `campaign/expert-bandwidth` into master with a merge commit;
  - write RETRO.md;
  - update INDEX.md;
  - open `decode-chain` (PLAN §9).

### 4.2 The bulk12 decode port (dense GEMMs)

**Facts:**
- decode's P is `dl_pooler_router_kernel<NF>` (`rewrite/src/muse_decode_layer.py`: header ~278, kernel ~401, host
  ~1553).
  - It is a swap-AB GEMV over W_pr bf16 [4608, 8192]: rows 0..4095 are the pooler, then 8 router tiles of
    2 x [16 hi rows; 16 lo rows], with hi = bf16(router) and lo = bf16(router - hi).
  - SPLIT_K is 3, giving 216 CTAs in one wave at 2 CTAs/SM. NF = 1 / 2 / 4 for T <= 8 / 16 / 32.
  - PDL: a CTA issues only the cp.async of its first 3 weight stages before `griddepcontrol.wait`; x comes after.
  - A fixed-order ticket sums the split-K partials and writes ei bf16 and logits fp32 (= sum(hi) + sum(lo)).
- At T=1, P is graph-timed at 19.6 us (~75 MB at 3.85 TB/s).
- On bf16, the b12 pooler at m=1 judged 17.7 -> 14.8 us (50.3 MB of 12-bit at ~3.4 TB/s net).

**Design for P at T=1:**
- **Rows.**
  - Encode the 4096 pooler rows with the bulk12 codec.
  - Keep the 512 router rows raw: the lo rows are residuals about 8 binades lower and would all escape.
- **One launch covers both.** The b12 row-owner shape (R=32 rows per CTA, whole K, 1 CTA/SM) gives 128 CTAs for the
  pooler. Give each CTA 2 experts' hi and lo rows (4 raw rows, 64 KB) as a tail of its stream (+17% bytes per CTA)
  and compute the 2 logits in fp32 in a fixed order. Do not use a second launch or extra CTAs: the 4 spare SMs cannot
  stream 8.4 MB in time.
- **PDL.** Keep the incumbent's weight prefetch: the producer warp issues the first S weight stages before
  `griddepcontrol.wait`, and x (copied once into smem) comes after the wait.
- **Prepare.** Port `_b12_encode`, `_b12_encode_into` (the device round trip, asserted bit-exact) and the escape
  blocks. Gate with rt.py's read flush in the whole layer; any miss keeps the incumbent.
  `scratchpad/bulk12/b12_emul.py` verifies the codec, the fragment maps and the epilogue on the CPU.
- **Expected:** P at T=1 19.6 -> ~17.3 us. That is +2.2% on d1 rows (~105 us) and about **+0.37% on the headline**:
  sub-margin, but real. Bundle it into a decode check that also carries another question.

**Then, in order:**
1. qkv R32 at T=1. Calibrate it on bf16 first with `b12-qkv-m1-r32-one-cta-per-sm-readflush-gate`: 11.2 -> ~9.5 us
   if it wins, i.e. +0.25%.
2. o at T=1: judged 0.97 on bf16, so skip it unless it rides along.
3. T=16 / 32 (d16 / d32, dense share 22-27%): only after bf16 shows multicast x winning at m=16 / 32 under a read
   flush. That is the only route to the +1.5-3% dense figure.
4. T >= 128 stays raw. The verify T=128 pooler K-halves (bf16 hypothesis (b2)) is the decode-portable item there.

### 4.3 decode-chain (campaign #2, PLAN §9)

- **Content:** the T <= 32 own-kernel chain, i.e. the X3 folds plus CUDA qkv / o / sliding kernels with PDL
  static-weight prefetch, plus the P rewrite (now bulk12 P with PDL prefetch, §4.2). Expected x1.047-1.106 (PLAN).
- **Re-rank it** against the anatomy numbers before opening. If the T=1 fixed cost is large, decode-chain moves ahead
  of the remaining X1 work.
- **Weights are prepare-time constants.** Every kernel in the chain can stream its first stages before
  `griddepcontrol.wait` (P already does). That is how a dense GEMM gains without fewer bytes.

### 4.4 The queued sprint hypotheses (churn queues on master), with verdicts

**Policy context:**
- The priority is decode > moe > bf16 > attention > prefill (the user, 19:05Z). Use bf16 and attention checks for
  decode-portable work; prefill only when nothing above has a useful question.
- These run through churn (an Opus review, then claim, implement, check), which costs agent tokens. Run the top one or
  two per problem per day, and register stale BEST.json files first (§3.4).
- `scratchpad/hypsum.py` prints a compact summary of all of them.

**bf16:**

| slug | verdict | why |
|---|---|---|
| `b12-qkv-m1-r32-one-cta-per-sm-readflush-gate` | **run first** | adds `flush_mode="read"` to `_cell_gate`, which every later bulk12 cell and the decode port's calibration need, and tests qkv R=32 at 72 CTAs. Python only; it also prints read-flush ratios for pooler and o. |
| `pooler-khalf-m128-fp32-partials-triton-castadd` | run, **candidate (b2) only** | fp32 partials plus one Triton add-cast has cuBLAS split-K numerics. Drop candidate (a), the bf16 partials (rules). Decode-portable (verify T=128 P). Gate with the read flush once it exists. |
| `b12-unpooler-slice-m1-r16-rowowner` | run after the read-flush gate | +0.2-0.4%; new instances of the existing template; decode-portable at T=1 |
| `qkv-swapab16-five-stage-ring-m8-m16` | cheap; run stacked | +0.2-0.5%; decode's qkv at T=8 / 16 uses the same swap-AB Triton kernel |
| `lt-config-sweep-m256-beyond-heuristic` | later | +0.4-0.7% on the weakest bf16 load (verify_32); ~8 s of prepare |
| `lt-nn-layout-cells-o-m128-qkv-m128` | stack only | +0.07-0.34% |
| `combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16` (hold) | **retire** | built on the pre-bulk12 base; its qkv cells never routed; superseded by the khalf (b2) item |

**attention** (board 1.5347, rank 451):

| slug | verdict | why |
|---|---|---|
| `sk-full-line-gather` | **run first** | one mapping change with a measured precedent (kernel A -4.7% / -14% at decode_16 / 32); decode_layer runs the same `splitkv_attn_sk_kernel` at decode_16 / 32 |
| `sliding-verify-wgmma-window-s2-split-combine` (rewrite s2) | run second | sliding verify_16 / 32 are weight-46 rows, and decode's weakest loads (v16 / v32) run the same FA3 chain; ~1.1x on the row |
| `sliding-decode-sk-s3-fused-preattn-rope` (reviewed: proceed) | run third, **with the review's two blocking fixes** (`load_prefix_pseudo` is empty in the skF TU; set the dynamic-smem attribute for both skF instantiations) plus an edge-key self-check | the review expects x1.0-1.15 per row, +0.3-1.0%, not the minter's +1-2% |
| `c320-combine-onepass` + `c320-pdl-early-trigger` | one check together (separate V3 / V4 readouts) | d1f is the planned decode port; +0.5-1% on attention |
| `combo-sk-full-line-c320-tail` | the submit candidate once its parts show gains | |
| `sliding-verify-wgmma-split-combine-on-v3chain` | only if s2 wins | the board twin of s2 |
| `sliding-verify-wgmma-window-s2b-fast-merge` | only if s2 fails | |
| `full-verify-wgmma-s7-split-combine` | later | measures the s5 kernel without its ticket tail; a long line |
| `c320-attn-full-line-gather` | stack with the c320 pair | +0.1-0.3% |
| `sliding-decode1-perkvhead-launch-select` | low | low prior |
| `sk-full-line-5stage-select`, `sliding-decode-band2-sk-gate` | conditional on `sk-full-line-gather` | |

**prefill** (last priority, no decode value; board rank 179, our best):

| slug | verdict | why |
|---|---|---|
| `experts-rega-stage-anatomy-diag` | first, if any prefill checks are spent | decides whether DIAGNOSIS's dequant/MMA-overlap ban stands, and bounds the schedule ideas |
| `experts-rega-round-rotated-schedule` | run | bit-exact; +1-2% if the modeled CTA-parity imbalance holds |
| `experts-rega-fine-n-buckets` | run | +1-1.7%; new wgmma N shapes (a canary keeps master on a miss) |
| `experts-rega-rotation-fastsilu-combo` | the stack to trust | +1.5-2.8% |
| `experts-rega-rotate-finebucket-fastsilu-combo` | after its parts | +2.5-4% |
| `experts-rega-dynamic-item-scheduler` | after rotation and LPT answer | high risk: a new producer/consumer handoff |
| `front-qknorm-rope-fp8-store-fusion-restack` -> `front-routing-pdl-launch-diet-stack` | as a stack | about +1% together |
| `experts-rega-w13-fast-silu-epilogue`, `experts-rega-lpt-group-order`, `experts-rega-pdl-chain-regate-iqr-slack`, `routing-triton-hilo-sum-topk-localids` | stack only | +0.2-0.8% each |
| `experts-rega-mc4-cluster-high-ft` (hold), `experts-rega-mc2-bm256-stack-high-ft`, `experts-rega-low-2cta-fullslab-mc2` | **retire** | premise falsified (§3.1) |

### 4.5 End game (before the close)

- Final standings come from post-close reruns of each team's **board** submission (FINDINGS "End game"). The board
  keeps the best-scoring run, so a noise-high older run can hold the row with worse code.
- In the last 24 h:
  - send a scored run of the final bytes;
  - if an older run still leads, consider `okc withdraw` of the older run (the standing rules allow it);
  - run the full rules checklist on each final file;
  - check that `okc_lint` is clean (via `rewrite/build.py`) and that the rules version is still 2026-10-22.1;
  - run the nearby-shape sweep: T=129..255 must map cleanly onto the 128 / 256 classes.
  - **(added 10-27)** the final decode bytes must carry the `_ead_anatomy` use-after-free fix (7f74c074) or run with
    `EAD_ANATOMY = False`. Before 7f74c074, eacf0722-era bytes can fail prepare at random (8e3c26d6). Prefer turning
    off every prepare-time diagnostic (anatomies, ladder) in the final bytes.

## 5. Operating lessons for the next loop

1. **Stamps come from `date -u`.** Never estimate: the 16:05Z entry and the 19:30Z / 19:50Z Now stamps were wrong.
2. **Read the current Now block before rewriting it.**
   - Carry forward every bullet marked as the user's.
   - Never template Now from a file outside the repo.
   - Rewrite from LOG.md itself (`log_now.py` only after reading).
3. **The user's steers arrive as blockquote banners:**
   - at the top of `campaigns/INDEX.md`;
   - after PLAN.md's frontmatter (and in §6 / §11);
   - in the campaign folder's `CLAUDE.md`.

   The user's openclaw session writes them. At the start of every tick, check `git status` / `git diff` on those three
   files, and acknowledge each new banner in LOG with a dated entry. A bullet placed only in LOG's Now block, or only
   in PLAN §6 / §11, did not reach the loop.
4. **Heartbeats and the prompt cache.**
   - The cache TTL is 1 hour, and an idle wake re-reads the whole context.
   - With a large context, idle heartbeats of 3600 s risk expiry and a full cache write.
   - Keep the context small (fresh session per day) and use 1800-3000 s idle heartbeats.
   - While a check is in flight, wake on its completion (`okc status <job> --wait` in the background) with a
     1200-1800 s fallback.
5. **Keep the main context lean.** Delegate execution with narrow briefs: builds and syntax checks, greps over the
   generated files, audits (one read-only Opus audit per new or changed CUDA module), and harvest summaries. The main
   thread keeps decisions, LOG and DISTILL.
6. **Agent budget:** at most 3 concurrent agents (the user's token quota binds). Do not mint more hypotheses than the
   day's checks can run.
7. **Checks:**
   - every check answers new questions;
   - never re-roll identical bytes;
   - submit only bytes that beat the board;
   - an attempt file comes before every check or submit (`**Job:**` placeholder, absolute paths);
   - run `rewrite/bin/cuda_syntax.py` on every generated file before dispatch.
8. **Out of checks:** when a problem's checks run out, write its reset plan **in the repo**, then switch to the
   highest-gain problem with checks left. Never pre-build hours ahead of a reset.
9. **Logging:** archive every run and log at most 10 lines per run. Commit on the campaign branch at every working
   step; land scored files on master immediately.

## Appendix A. Run ledger of the campaign (10-26, 15:35Z-22:42Z)

Every campaign run is archived in `runs/`. The prefill confirm is in
`muse_prefill_layer/hypotheses/archive/experts-rega-s4-bm256-wide-tile-high-ft-cf/`.

| job | kind | code | result |
|---|---|---|---|
| 7fff4a8b616946c6 | decode check | c8ac9beb X0a | 1.4524 |
| a03e596d3aaa4f69 | decode scored | c8ac9beb | 1.4466 (board then) |
| 68216f9dfd9741ec | moe check | c8ac9beb | 1.5849 |
| 523252767fc84a26 | decode check | d25a987d X0b | 1.4569 |
| a009b6c8a9df4e3f | decode scored | d25a987d | 1.4401 (below board) |
| 6eb226e9db4f4876 | moe check | d25a987d | 1.5879 |
| ff2bd9182fa04636 | decode check | 666eb1d7 v4 | **failed** (copy-check illegal address) |
| 54f63952015c4d3b | moe check | 666eb1d7 | 1.6412 |
| 35b55c74ee964187 | moe scored | 666eb1d7 | 1.6389 (board then) |
| 5573e05423304324 | decode check | 4f530b95 | 1.4950 |
| dc51d8e8f58e4cb7 | decode scored | 4f530b95 | **1.4956 (board)** |
| e403a42a80e6478f | moe check | 4f530b95 v5 | 1.6441 (v5 lost every gate) |
| 0f4605dca83744c7 | moe check | 4848aadc | 1.6845 |
| e854db4683dc495f | moe scored | 4848aadc | **1.6851 (board)** |
| 212cdf5dcb844d54 | decode check | 4848aadc | 1.5007 |
| a1bc1756256b4c91 | decode check | 3bcb685b | 1.5097 |
| 5eafb39e194d4d8b | moe check | 1cec46df ladder-2 | diagnostic, passed |
| f59cb301895240e6 | decode check | 27ef83df | **1.5347 (to score)** |
| 474490c8258d4521 | bf16 check | 4b79139c bulk12 m=1 | 1.2699 |
| 3834ceeca7c0410d | bf16 scored | 4b79139c | **1.2664 (board)** |
| 22ddaf4a563b40cb | bf16 check | 97aaf5a6 m >= 8 | 1.2683 |
| 0acdca481ebc4b47 | bf16 check | 03e8197d cp.async x | 1.2670 |
| 7e8be36a77534aa7 | bf16 check | c891dc84 router ticket | 1.2650 (killed) |
| 2980f57a8b9e4b1d | bf16 check | a7da6fa9 rotation | 1.2690 |
| 6617f94dd1ce459f | bf16 check | 419d2459 b12r | 1.2528 |
| 13cfaf1e25c341af | prefill check | churn confirm | 1.4711 (neutral) |
