---
id: 2026-10-26-expert-bandwidth
status: active
objective: overall kernel board rank (via muse_decode_layer; overall orders decode teams by decode speedup)
kpi: muse_decode_layer headline; testbed muse_moe_w4a16
baseline: decode 1.4422 (0d84f58ca33b424b, rank 373 of 731), moe 1.5762 (f0fc89031659499c), overall rank 380 of 1222 at 15:30Z
branch: campaign/expert-bandwidth (forked from campaign/decode-first-rewrite ae91f55d)
worktree: .worktrees/expert-bandwidth
start: 2026-10-26T15:30Z
deadline: 2026-11-05T06:59Z (no freeze, by the user; last runs dispatched by ~06:30Z so they finish in time)
approved: 2026-10-26 by the user (plan approved; "fully autonomous today, no choices unless presented now")
mode: autonomous self-paced /loop, single agent + one Opus audit per new CUDA module
---

> **STOP — NEWEST USER STEER, 2026-10-26 ~22:00Z, relayed from the user's openclaw session: THE SPRINT IS ENDED. No new work. Act on it this tick.**
>
> The user said: "stop any more new sprints. Keep what's in flight." This replaces the ~21:40Z expiring-quota sprint banner.
> - **Start nothing new:** no new /hypothesize minting, no new adversarial reviews, no new implementations, no new
>   worktrees, no new subagents of any kind. Minted hypotheses not yet implemented stay queued; do not implement them.
> - **Keep what is in flight:** let agents already running finish; harvest checks already dispatched; submit the exact
>   bytes of any harvested check that beats its board (rules checklist as usual). A hypothesis already implemented
>   and committed when you read this may be dispatched; anything earlier than that stops.
> - Your own bf16 line likewise: finish and check what is already built/audited (b12r); do not start another build.
> - **Then go quiet** until the 00:00Z reset plan (decode scored run, moe + decode checks). Long heartbeats while idle.
> - Log what the sprint did (checks used, wins, landed, what was left queued) in LOG.md.


> **STOP — USER STEER, 2026-10-26 ~19:05Z, relayed from the user's openclaw session. Not yet acknowledged. Read before acting on this tick.**
>
> **Do not idle until 00:00Z, and do not pre-build hours ahead of a reset.** Decode and moe quota is spent for 10-26. Their
> 00:00Z plan is already written (submit 1.5347, moe dq A/B + H=-8 + v4c, decode H=-8 + v4c + anatomy). Stop adding to it.
> **Now:** move to the highest-gain problem that still has checks today and work there with immediate check feedback.
> Priority by gain for decode: decode > moe (~53% of weighted decode time, x1.19-1.27) > **bf16_proj** (~26%, x1.03-1.04)
> > **attention_decode** (~15%, x1.01-1.06) > prefill (no decode value; last). So bf16_proj now, then attention_decode,
> favouring kernels that port into decode (campaign #2/#3: CUDA qkv/o/P on bf16, B=1 attention ports). Submit when a
> check beats that board. Return to decode/moe at 00:00Z. Apply the same rule every day: when a problem's checks run
> out, write its reset plan, then switch to a problem with checks left.
> Using every check is not required; never spend checks on low-value questions. Unchanged: every check answers a new
> question, no identical bytes re-rolled, challenge rules apply. Full text: PLAN.md section 6 (top) and section 11.
> **Acknowledge in LOG.md with a dated entry and keep this policy in the Now block.** Your Now rewrites at 18:49Z and
> 19:05:50Z overwrote earlier copies of this note unread.


# Campaign `2026-10-26-expert-bandwidth`: the NVFP4 expert block at full bandwidth

Durable home: `campaigns/2026-10-26-expert-bandwidth/` on master.

| | |
|---|---|
| Branch | `campaign/expert-bandwidth`, forked from `campaign/decode-first-rewrite` HEAD ae91f55d (all rewrite/ tooling lives there) |
| Worktree | `.worktrees/expert-bandwidth` |
| KPI | muse_decode_layer headline |
| Testbed | muse_moe_w4a16 |
| Close | 2026-11-05 06:59Z; no freeze (the user, 10-26); last runs dispatched by ~06:30Z |

## Context

The decode-first campaign (10-26, 04:38–08:26Z) moved the boards as follows (ranks re-read at 12:30Z):

| board | start of 10-26 | after campaign |
|---|---|---|
| decode | 1.316 | **1.4422** (rank 365 of 728) |
| moe | 1.402 | **1.5762** (rank 422 of 828) |
| overall | rank 480 | **rank 372 of 1210** |

At a fixed score we lose about 10 places a day. The user asked for another campaign of the same scale, built around the problem with the biggest scoring gain.

**User decisions (10-26)**
- **One agent writes all code, plus one read-only Opus audit per new or changed CUDA module before its first check.** No builder subagents.
- **Run autonomously ("single agent, slower token spend, same trajectory").**
  - Build the rewrites, check them, and use up the day's checks.
  - Submit scored runs when a check improves on the board (pre-approved).
  - When an ROI line proves out: commit, merge, close the campaign, and start the next-highest-ROI campaign without waiting for approval (§6, §10).
- Earlier, still in force: churn paused (all quota to campaigns); at most 3 concurrent agents; plans saved in `campaigns/`; never re-roll identical bytes for noise.

## 1. The biggest scoring gain: the expert block (moe-tested, decode-cashed)

**1.1 The overall board is the decode board for us.** Live fit over 1210 rows:
- Every team with a decode run sits in decode-speedup order.
- Teams without a decode run are slotted in by a composite of moe/bf16/attention. Prefill is not counted.
- So our overall rank moves only through decode. Today's thresholds:

| decode speedup | 1.53 | 1.69 | 1.86 | 1.97 |
|---|---|---|---|---|
| decode rank | 302 | 202 | 96 | 49 |

**1.2 The headline formula.** Reproduced exactly from the judge's JSON:
- headline = geomean over 6 loads of (46·sglang_sliding + 16·sglang_full) / (46·ours_sliding + 16·ours_full).
- Loads now: d1 1.608, d16 1.465, d32 1.314, v1 1.665, **v16 1.307, v32 1.336**.
- A µs saved on a sliding row is worth 3× one saved on a full row.

**1.3 Ranked levers.** My model plus two research agents; all are judge-formula multipliers on 1.4422.

| lever | headline | effort |
|---|---|---|
| **Expert block at T=128/256 (verify 16/32)**: read every expert once | ×1.08–1.12 → 1.56–1.62 | 3–7 days |
| **Expert consumer at T≤32**: 8 warps per CTA, 2.6–3.0 → 3.4–3.7 TB/s | ×1.05 → 1.51 | 2 days |
| both together | → ~1.64–1.70 | |
| T≤32 glue folds (combine → K2, R → K1, boundaries) + PDL prefetch | ×1.03–1.07 | 2–4 days |
| quick wins (R's unused align tail, combine slot prefetch) | ×1.008–1.016 | hours |
| full-layer attention, every row at speed of light | ≤×1.061; realistic ×1.014–1.032 | XL |
| dense m=128/256 GEMM | ×1.009–1.012 | L |

- The expert block is 49–75% of the verify 16/32 rows. Those rows are 3.0–3.8× above speed of light, the furthest of any row.
- moe isolates exactly this block. It goes 1.58 → about 1.9–2.0 if both expert levers land.
- The attention board runs **the same** full-layer kernels as decode at B ≥ 16 (sk 107/225 µs, FA3 152/289 µs), so it offers nothing to port.

## 2. Root causes (measured; research in the campaign's `research/`)

1. **v3 re-streams hot experts.**
   - Passes cover ≤16 routes, so an expert with n routes is streamed and dequantized ceil(n/16) times: 1.17–1.7× extra bytes.
   - Two-tile passes run 2.84 vs 3.40 TB/s.
   - Activations are re-fetched from L2 per 64-row group: n_local × 1.5 MB, i.e. 0.4–0.9× the weight bytes.
   - At decode T=256, the Marlin hot half of the hybrid runs serially at 0.9 TB/s (207 µs).
2. **The consumer is latency-bound at every T.**
   - Each warp issues only 0.22–0.32 instructions per clock; no pipe is above ~50%.
   - smem caps the CTA at 2/SM, so there are only 2 consumer warps per SMSP.
   - Throughput scales with warps: at 1 CTA/SM, R4 collapses to 1.37 TB/s while the ring alone holds 4.48.
3. **Engine B serialized dequant and MMA in the same warps.**
   - Its parts summed to 1.55× the whole ("OVERLAP").
   - Its token box was re-read by all 8 warps (smem caps at ~3.4 TB/s).
   - Runs of ≤32 routes still re-streamed hot experts.
   - Never tried: dequant/MMA warp specialization, mixing hot and cold work per SM, an 8-warp CTA on Engine A's consume().
4. **Gates don't see the judge's routing.**
   - Decode `_cb_data` (`rewrite/src/muse_decode_layer.py:6157-6199`) uses an isotropic `randn*3` stream: T=128 touches 23–26 experts vs the judge's ~18.4, with about half the judge's hot rows.
   - moe's v3 T=128/256 gate timed only zipf/random. It never timed the judge-style draw that `eb_moe.py:44-55` (`_eb_judge_draw`) already has. v3 won ×1.32 on random but lost ×0.614 on the 1154-route zipf.
   - Selections at T=16/32/128/256 may be wrong today.
5. **Profile rows include PDL waits** (ea_combine 22–41 µs, real 3–4.5 µs). Compare engines only on graph-timed kernel-only runs or whole-layer gates.

## 3. Milestones (decode targets; ranks at today's board)

| milestone | content | decode (rank today) | moe |
|---|---|---|---|
| M0 (10-26/27) | X0: judge-faithful gates, quick wins, ladder-2 verdict | ≥1.46 (358) | ≥1.59 |
| M1 (10-28/29) | X1: 8-warp consumer live at T≤32 | ≥1.51 (328) | ≥1.70 |
| M2 (10-31) | X2: large-T engine live at T=128/256 | ≥1.62 (252) | ≥1.90 |
| M3 (11-02) | X3: T≤32 glue folds | ≥1.68 (212) | — |
| Stretch (11-03) | X4 ports | 1.72–1.78 (182–135) | 2.0 |

Ranks drift down about 10 places a day as other teams submit, so these are upper bounds.

## 4. Workstreams (ROI order; each has a metric, a target and a kill rule)

### X0: Measure right, then choose the engine (10-26 → 10-27)

**X0a. Judge-faithful gate inputs.**
- In `_cb_data`, reimplement the judge's stream formula inside the submission (never import reference.py): a 0.7 mean direction plus rank-4 shared structure, router rows with a mean component; see `reference.py:140-148, 274-284`.
- Print routes / distinct / max / hot rows over 8 draws at T=16/32/128/256. The target is distinct ≈ 7.5 / 9.0 / 18.4 / 21.3 (backed out of the judge's speed-of-light bounds).
- Re-run the existing selectors on those inputs: v2 vs v3 at T=16/32, and hybrid H at T=256.
- In moe, add `_eb_judge_draw` at p10/p50/p90 to the v3 T=128/256 gates. This may move moe T=256 off master's wg path.

**X0b. Quick wins.**
- Drop R's unused Marlin align tail (`muse_decode_layer.py:655-708`).
- Issue all 8 slot loads before the reductions in `ea_combine.cuh:47-74`.

**X0c. Ladder-2: one moe check.**
- Extend `rewrite/lib/cuda/ea_ladder.cuh` and `rewrite/lib/ea_ladder.py` (behind moe's existing ladder flag; ~45 s compile, start deadline 170 s). Use a bank of real repacked records; graph-timed, read flush, empty-graph baseline subtracted.
- Each rung prints µs, TB/s, el/clk/SM, clk/stage, and regs/smem/occupancy.

| rung | what it measures |
|---|---|
| P0 | ring ceiling; L2→SM probe |
| P1 | mma.sync HMMA/clk/SM with NT 1/4/8/16 |
| P2 | wgmma RS clk per m64k16 at N 16/64/128/256 with a dequant warpgroup alongside |
| P3 | wgmma SS at the same N |
| C1 | Engine A consume() at 4 warps (repeats 3.40 / 2.84) |
| C2 | 8 consumer warps, 36 KB stages, D=2 |
| C3 | 3 CTAs/SM |
| C4 | D1 cell, N=32/64 |
| C5 | D2 hot cell alone, then mixed 1:3 with C2 cold units in one launch |
| C6 | D3 cell |

- It ends with `dq_exact` and a printed **VERDICT**:
  - **S1** if C2 ≥ 1.1 × C1.
  - **D1** if P1 ≥ 0.7 HMMA/clk/SM and C4(N=32) ≥ 3.0 TB/s.
  - **D2** if P2(N=128) ≤ 1.2 × 64 clk and the C5 overlap ≥ 1.25.
  - **D3** only if P3 ≤ 0.6 × P2 and C6 beats C4/C5.

| | |
|---|---|
| Metric | selector predictions within 5% of judge rows |
| Kill | none (prerequisite) |

### X1: 8-warp consumer (S1): T≤32, and the cold half of X2 (10-27 → 10-29)

**Design.**
- consume() and the repack stay unchanged.
- The CTA becomes 1 producer warp + 8 consumer warps (288 threads).
- A stage is two Engine A group records of the same expert and k-chunk (2 × 18,432 B on one full barrier), plus the pass's x rows **once**, shared by 128 rows.
- D=2, ~93–110 KB smem → 2 CTAs/SM, 16 consumer warps/SM.
- Template the CTA shape over the v1/v2/v3 bodies in `ea_kernels.cuh`, `ea2_kernels.cuh`, `ea3_kernels.cuh`; shrink v3's map to fit.
- Also: when w2's aligned split would idle ≥10% of the grid, fall back to stream-K (T=1 idles 72 CTAs today).

| | |
|---|---|
| Metric | kernel-only TB/s per class |
| Target | 3.4–3.7 TB/s (from 2.6–3.0) |
| Kill a class | C2 < 1.1 × C1 in the ladder, or its whole-layer gate < 1.03 after 2 checks |

### X2: Large-T engine (T=128/256), D1 or D2 by the ladder verdict (10-29 → 10-31)

**D1 (default under single-agent capacity): Engine A v4 on X1's 8-warp CTA.**
- A new lossless record: [expert][128-row group][256-k chunk][8 warps][16 rows × 256 k], 2,048 B codes + 256 B scales. It halves activation re-fetch.
- Passes cover up to 32 routes (64 if P1 allows), with 1/2/4/8 n8 tiles chosen per pass on the device.
- Work order: hot-first LPT under cost-weighted stream-K, reusing Engine B's `first_unit`/owner logic (`eb_kernels.cuh:208-305`).
- Keep the last-arriver fix-up and the PDL chain.
- Retire the Marlin hybrid at T=256.
- Expected: decode v16 expert block 280 → ~190 µs, v32 410 → ~265 µs; moe T=128 296 → ~210, T=256 398 → ~290.

**D2 (only if the verdict shows ≥1.25 hot/cold overlap and leaves ≥5 days before the close):**
- 1 CTA/SM, 4 warpgroups: producer; 2 cold mma.sync warpgroups; 1 hot wgmma RS warpgroup that reads each hot expert once, N ≤ 128.
- setmaxnreg 40/112/112/232.
- Expected: v16 → ~170 µs, v32 → ~190 µs.

**D3 (dequant to smem + wgmma SS)** is the fallback only if P3 ≪ P2.

| | |
|---|---|
| Metric | expert-block µs on judge-faithful T=128/256 routings |
| Target | ≥1.4× faster than the incumbent per class |
| Kill a class | whole-layer gate < 1.05 after 3 checks; that class keeps its current path |

### X3: T≤32 glue folds (11-01 → 11-02)

- **Combine into K2's epilogue:** a per-token last-arriver post-norm, removing the combine launch.
- **R into K1:** at T≤8, routing plus the pre-expert RMSNorm move into K1's prologue, computed redundantly per CTA.
- **Boundaries:** fold the attention boundary into a CUDA o-GEMM epilogue, and the FFN boundary into `dl_unpool`'s epilogue at T≤8. Fused only: a standalone own boundary lost before (3.9 vs 2.8 µs).

| | |
|---|---|
| Target | ×1.02–1.05 |
| Kill a piece | it loses its whole-layer selector |

### X4: Stretch, only if M2 lands by 11-01

- B=1 attention ports from the attention branches:
  - d1s: `sliding_cluster_kernel`, 9.7 vs 13.1 µs;
  - d1f: c320 inside piece A, 15.4 vs 20.3 µs;
  - together ×1.0035–1.009.
- Dense m=128/256 (W5 design in `research/dense_gemm_forensics.md`): ×1.009–1.012.
- Full-layer attention stays parked: at most ×1.03 for XL effort.

## 5. Schedule (UTC; quota resets 00:00Z; today's remainder is decode 7 checks / 3 scored, moe 6 / 6)

| day | work | checks (decode / moe) | scored |
|---|---|---|---|
| 10-26 | campaign home; X0a + X0b code; ladder-2 build + audit | 2 / 2 | if a gate fix gains |
| 10-27 | ladder verdict; X1 build + audit; moe checks | 2 / 4 | moe if gains |
| 10-28 | X1 into decode T≤32; X2 build starts | 4 / 2 | decode M1 |
| 10-29 | X2 build + audit | 1 / 3 | — |
| 10-30 | X2 iterate on moe | 2 / 5 | moe |
| 10-31 | X2 into decode T=128/256 | 5 / 2 | decode M2 |
| 11-01 | X3 combine → K2, R → K1 | 5 / 1 | — |
| 11-02 | X3 boundaries; nearby-shape sweep | 5 / 1 | decode M3 |
| 11-03 | X4 or buffer | 4 / 1 | as earned |
| 11-04 → 11-05 06:59Z | keep building to the close; rules audit on each final file; last runs dispatched by ~06:30Z; land on master | as earned | all |

Slip order: drop X4, then X3's boundary piece. X1 and X2 are protected; D1 is chosen over D2 whenever the schedule is tight.

## 6. Autonomous operation (single agent + audits)

> **STOP — NEWEST USER STEER, 2026-10-26 ~22:00Z, relayed from the user's openclaw session: THE SPRINT IS ENDED. No new work. Act on it this tick.**
>
> The user said: "stop any more new sprints. Keep what's in flight." This replaces the ~21:40Z expiring-quota sprint banner.
> - **Start nothing new:** no new /hypothesize minting, no new adversarial reviews, no new implementations, no new
>   worktrees, no new subagents of any kind. Minted hypotheses not yet implemented stay queued; do not implement them.
> - **Keep what is in flight:** let agents already running finish; harvest checks already dispatched; submit the exact
>   bytes of any harvested check that beats its board (rules checklist as usual). A hypothesis already implemented
>   and committed when you read this may be dispatched; anything earlier than that stops.
> - Your own bf16 line likewise: finish and check what is already built/audited (b12r); do not start another build.
> - **Then go quiet** until the 00:00Z reset plan (decode scored run, moe + decode checks). Long heartbeats while idle.
> - Log what the sprint did (checks used, wins, landed, what was left queued) in LOG.md.

- **the user, 2026-10-26 ~19:05Z (relayed from the openclaw session): quota policy; replaces the ~18:50Z note.**
  - Prioritise real gains for decode. Order by gain potential: **decode** (the KPI) > **moe** (the expert block,
    ~53% of judge-weighted decode time in f59cb301; optimistic x1.19-1.27) > **bf16_proj** (dense GEMMs, ~26%;
    x1.03-1.04, incumbents already at 3.4-3.9 TB/s) > **attention_decode** (~15%; x1.01-1.06) > **prefill** (no
    decode value; last, only when nothing above has a useful question).
  - Even use of quota is not required. Do not spend checks on low-value questions just to use them up.
  - Do not let time go to work that waits hours for feedback. When a problem's checks are exhausted, do not build
    for it hours ahead of the 00:00Z reset: write the reset plan (what to build and check first at reset; finishing
    an audit already under way is fine), then move to the highest-priority problem that still has checks today and
    work there with immediate check feedback. Come back to decode/moe at the reset.
  - On bf16_proj and attention_decode, favour work that can be ported into decode (campaign #2/#3 kernels); submit
    their bytes when they beat that board.
  - Unchanged: every check answers a new question, no identical bytes re-rolled, the challenge rules apply.
    Supersedes section 11's "bf16 / attention quota only as testbeds" and "prefill stays idle", in the order above.
  - Read this before every tick. Your Now rewrites at 18:49Z and 19:05:50Z overwrote it in LOG.md unread; it also stands in section 11.

**Driver: a self-paced `/loop` in this session.**
- Started after approval with the tick prompt `okc campaign tick: continue the active campaign (campaigns/INDEX.md -> its PLAN.md §6)`.
- It runs until the user says stop or the kernel track closes.
- It lives in this Claude Code session: if the session or the machine stops, the loop stops. Everything needed to resume is in files.

**Each tick does the following, in order.**
1. **Resume.**
   - Read the campaign's `LOG.md` "## Now" block: current step, jobs in flight, next action, today's remaining quota.
   - Read `okc problems`.
   - Never re-derive state from memory.
2. **Harvest.**
   - For each finished job: archive it, summarize it in ≤10 lines in `LOG.md`, and judge it per-row against its stated question.
   - Then: submit the same bytes if improved (rules below), land on master, or record the kill.
3. **Act.** Advance the current workstream step:
   - design note → edit → build/lint → CPU dry run → audit (new CUDA modules only) → attempt file → `rewrite/bin/dispatch.py`.
   - Dispatch checks with `okc status <job> --wait` in the background, so their completion wakes the loop.
4. **Rewrite the "## Now" block, then end the turn.**
   - Use `ScheduleWakeup` as a fallback heartbeat: 1200–1800 s while a check is in flight (its completion wakes the loop sooner), and 1800 s when waiting on nothing.
   - While actively building, keep working within the tick instead of sleeping.

**Quota use: spend the day's checks, never on noise.**
- Each UTC day, aim to spend the active campaign's decode and moe checks (20 each). Every check must answer a new question: a new kernel, a variant, or a diagnostic rung.
- While the main line is mid-build, put spare checks on queued smaller items: selector variants, quick wins, the next campaign's diagnostics.
- From 22:00Z, any checks still unspent go to queued variants or diagnostics before the 00:00Z reset.
- The bf16 and attention boards are used only when a workstream uses them as its testbed (dense GEMM → bf16; attention ports → attention). Prefill stays idle.

**Scored runs (pre-approved).**
- Submit the exact bytes of a check whose headline beats our board row and whose changed rows improved. The board keeps the best run, so sub-margin real gains are fine.
- At most 10 per problem per day. Never identical bytes re-rolled.
- Then commit the generated file to master with a DECISIONS/LOG note and re-read the board.

**Commit and merge.**
- Commit on the campaign branch at every working step.
- Land scored files on master immediately.
- When a campaign closes: merge its branch into master (merge commit), write `RETRO.md`, and mark it closed in `INDEX.md`.
- No git remote exists, so nothing is pushed.

**Token pacing ("slower spend").**
- One agent; one Opus audit per new CUDA module, never concurrent.
- grep and line ranges only on the 300–400 KB generated files; ≤10-line run summaries.
- Long heartbeats when idle; no speculative reading.

**Stop or pause conditions** (log the reason and stop the loop with `ScheduleWakeup stop`):
- the user says stop;
- the kernel track closes (11-05 06:59Z) and the final runs are landed;
- a scored run found to break the rules: withdraw it (`okc withdraw <job>`) without asking, log it, and continue; nothing else outward-facing is ever done (only okc check/submit/status/leaderboard/withdraw);
- 3 consecutive infrastructure failures (okc errors or judge outages): back off to 3600 s heartbeats and log it;
- any action outside `~/git/musespark` or outside okc check/submit/status/leaderboard (never cluster hosts or shared storage).

**Loop per change.** Same as decode-first PLAN §6:
1. Design note in the campaign `LOG.md`.
2. Edit `rewrite/src` or `rewrite/lib`.
3. `rewrite/build.py` + `okc_lint`.
4. CPU dry run.
5. **One read-only Opus audit per new or changed CUDA module** (the churn AUDIT_PROMPT focus list), at most one audit agent at a time.
6. Attempt file in `<problem>/attempts/` with the rules checklist and a `**Job:**` placeholder.
7. Check via `rewrite/bin/dispatch.py`.
8. Archive via `rewrite/bin/archive.py`, then log.

**Every check answers a stated question.** It prints PATH CHOSEN lines, and decode re-prints them at the end of prepare (the judge keeps a 16 KB tail).

**Gates.**
- Whole-layer, graph replay, read flush, on judge-faithful routing (from X0a).
- New kernels are checked against fp32 references with their own routing, never only against master.

**Scored runs.**
- Submit when a check beats the board beyond 1.24%, or a confirmed sub-margin gain has real kernel rows behind it.
- Never re-roll identical bytes.
- Land each scored file on master with a DECISIONS note.

**Rules gate, unchanged.**
- Graph-capturable; device-built maps and classes; no host branch on routing.
- Lossless permutation-only repacks; NVFP4/BF16 stay; bf16 activations; fp32 accumulation.
- No reference import.
- Prepare ≤300 s and the run ≤15 min. Decode prepare is ~68 s today and each module adds ~40 s.
- Final review reruns nearby shapes: T=129..255 must map cleanly onto the 128/256 classes.

**Token discipline.** grep and line ranges on the 300–400 KB generated files; state lives in `LOG.md`.

## 7. Verification

- **X0:**
  - Gate-input stats match distinct ≈ 7.5 / 9.0 / 18.4 / 21.3.
  - Ladder-2 prints every rung plus VERDICT.
  - `dq_exact` 0 mismatches.
- **Each engine stage:**
  - okc check passes 13/13 (decode) and 5/5 (moe).
  - The canary is clean on forced buckets (one expert, all local, zero local, skewed, every N bucket) and on 2-state alternation.
  - fp32 reference error ≤ master's.
  - PATH CHOSEN shows the new kernel live.
  - Target rows improve; untouched rows stay within drift.
  - The prepare sweep over T=2..7, 9..15, 17..31, 33, 129..255 is correct with no cliffs.
- **Milestones:** scored submit, `okc leaderboard muse_decode_layer --json` and the overall rank logged in `LOG.md`.
- **Final day:** full rules checklist on each final file; `okc_lint` clean; rules version still 2026-10-22.1.

## 8. Risks

| risk | mitigation |
|---|---|
| mma.sync tensor rate on H200 caps D1 at N=32 | P1 rung decides pass width; D2/D3 fallbacks |
| D2 too large for one agent before the close | D1 is the default; D2 only with ≥5 days left |
| New repack not lossless | prepare-time roundtrip check + `dq_exact`; canary before any timing |
| Gate fix changes today's selections for the worse | X0a re-runs every selector; the PATH lines show each flip; the board keeps the best run |
| Compile/prepare budget | ≤1 new module per check; ladder deadline-gated |
| Illegal instruction poisons the context | proven instruction forms only; canary first; first run of any module is a check |

## 9. Campaign chain (autonomous, in ROI order; re-ranked before each start)

Each campaign is opened by the loop itself in `campaigns/<date>-<slug>/`, with PLAN, FINDINGS and LOG. Before it starts, the loop re-runs the lever model on the latest scored decode profile, so a measured surprise can reorder the list.

| # | campaign | content | expected | entry / exit |
|---|---|---|---|---|
| 1 | `expert-bandwidth` (this plan) | X0–X2 | ×1.12–1.18 | now; closes when X2 lands or is killed (~10-31) |
| 2 | `decode-chain` | T≤32 own-kernel chain: X3's folds; CUDA qkv/o/sliding kernels with PDL static-weight prefetch; P (pooler+router) rewrite | ×1.047–1.106 | after #1; target 11-03 |
| 3 | `ports` | B=1 attention ports (d1s cluster, d1f c320); dense m=128/256 (W5); T≥128 glue | ×1.02–1.03 | only if ≥2 days remain before the close |

- X3 moves to campaign #2 if campaign #1 runs past 11-01.
- Full-layer attention (≤×1.03, XL effort) stays parked unless a re-ranking puts it first.

## 10. First actions after approval

1. **Close decode-first.**
   - Write `campaigns/2026-10-26-decode-first-rewrite/RETRO.md`: results, what worked, what failed, carry-overs.
   - Mark it closed in `campaigns/INDEX.md`.
2. **Save this campaign.**
   - Write `campaigns/2026-10-26-expert-bandwidth/` on master: `PLAN.md` (this plan, expanded) and `FINDINGS.md` (the board fit, headline formula, lever table, root causes).
   - `research/`: both agent reports verbatim from the workflow journal, plus the board-fit and lever-model scripts with their outputs and today's leaderboard snapshots.
   - `LOG.md`.
   - Commit.
3. **Memory.**
   - New `okc-campaign-expert-bandwidth` (active; single agent + audits).
   - Update `okc-standing-autonomy`: autonomous campaign chain via a self-paced `/loop`, scored submits on improvement, commit/merge/next campaign without asking, until the kernel track closes or the user says stop.
   - Mark `okc-campaign-decode-first` closed.
   - Update the index.
4. **Branch.** `git worktree add .worktrees/expert-bandwidth -b campaign/expert-bandwidth campaign/decode-first-rewrite`.
5. **Write the LOG's "## Now" block, then start the loop** (`/loop` skill, self-paced, with the tick prompt in §6).
6. **First ticks:**
   - Build X0a + X0b, then dispatch decode and moe checks on today's quota (decode 7 / 3, moe 6 / 6 left at 15:00Z).
   - Build ladder-2, run its audit, and dispatch the moe ladder check before 00:00Z.

## 11. Standing answers (the user asked for no further questions, 2026-10-26 ~15:20Z)

- **the user, 2026-10-26 ~19:05Z (relayed from the openclaw session): quota policy; replaces the ~18:50Z note.**
  - Prioritise real gains for decode. Order by gain potential: **decode** (the KPI) > **moe** (the expert block,
    ~53% of judge-weighted decode time in f59cb301; optimistic x1.19-1.27) > **bf16_proj** (dense GEMMs, ~26%;
    x1.03-1.04, incumbents already at 3.4-3.9 TB/s) > **attention_decode** (~15%; x1.01-1.06) > **prefill** (no
    decode value; last, only when nothing above has a useful question).
  - Even use of quota is not required. Do not spend checks on low-value questions just to use them up.
  - Do not let time go to work that waits hours for feedback. When a problem's checks are exhausted, do not build
    for it hours ahead of the 00:00Z reset: write the reset plan (what to build and check first at reset; finishing
    an audit already under way is fine), then move to the highest-priority problem that still has checks today and
    work there with immediate check feedback. Come back to decode/moe at the reset.
  - On bf16_proj and attention_decode, favour work that can be ported into decode (campaign #2/#3 kernels); submit
    their bytes when they beat that board.
  - Unchanged: every check answers a new question, no identical bytes re-rolled, the challenge rules apply.
    Supersedes section 11's "bf16 / attention quota only as testbeds" and "prefill stays idle", in the order above.
  - Keep this bullet at the top of the Now block when rewriting it (the 18:49Z rewrite dropped the first version).

- Scored runs, commits, merges into master, closing a campaign and opening the next one: done without asking.
- A rule-breaking scored run is withdrawn without asking; no other outward-facing action is taken.
- Nothing is deleted: the merged builder worktrees (.worktrees/engine-b, engine-b2, ea-v3, ea-v3big, dense-port,
  hotcold, moe-hybrid) stay.
- Churn stays paused; prefill stays idle; bf16 / attention quota only as testbeds of an active workstream.
- No freeze (the user, 10-26: "i dont want that"): work continues to the close at 11-05 06:59Z, with the last runs dispatched by ~06:30Z; the loop stops after they land; the serving track is not touched.
- If the session ends, the loop ends; "resume campaign" restarts it from LOG.md's "## Now" block.
