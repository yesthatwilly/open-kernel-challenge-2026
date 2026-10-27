---
id: X5 (workstream of campaign 2026-10-26-expert-bandwidth)
title: the single-pass expert block ("Engine S")
status: planned; execution starts only when the user says go
written: 2026-10-27T1613Z (date -u) by session 860eee0a, in plan mode, at the user's request ("write this plan as a durable file"; "populated well enough so it can be run COLD")
inputs: three code / tooling / evidence explorers; an architecture design agent; an adversarial reviewer; two external advisors' notes (pasted by the user, judged in section 10); ~/git/takemyjob/prototypes/hopper-probe/HANDOFF.md
supersedes: PLAN.md workstreams X1 / X2 (X1 landed as Engine A v4; X2 became the hybrid, whose hot half this replaces)
---

# X5: the single-pass expert block (Engine S): a cold-start plan

The expert block is 48.9% of the decode headline and all of moe, and it is where our gap to the leaders is
structural. This file is self-contained: a fresh session with no memory of 10-27 should be able to run it from
section 0. Every number cites its source; paths are absolute or repo-relative to `/home/user/git/musespark`.

## 0. Cold start

### 0.1 Read order

1. **This file**, sections 0-5 (the rest is reference).
2. `campaigns/2026-10-26-expert-bandwidth/LOG.md`: the `## Now` block (read it before you ever rewrite it), then the
   2026-10-27 entries from 0003Z.
3. `campaigns/DAILY-RHYTHM.md`: the user's standing 3:1 daily rhythm, with phases A/B/C and per-role effort.
4. `campaigns/2026-10-26-expert-bandwidth/FINDINGS.md` and `DISTILL.md` §3 (killed lines, gate-vs-judge
   disagreements, traps).
5. `campaigns/2026-10-26-expert-bandwidth/research/expert_engine_forensics.md` (designs D1/D2/D3, findings F1-F12).
6. `~/git/takemyjob/prototypes/hopper-probe/HANDOFF.md` and `NOTES.md`: the opportunistic Hopper-node mandate and its
   tool.

### 0.2 State when this was written (2026-10-27T1613Z)

**Boards** (`okc leaderboard <p> --json`; our team is `yesthatwilly`):

| board | ours | rank | leader | board run |
|---|---|---|---|---|
| decode (KPI) | 1.5920 | 327/794 | 2.3986 | c1daf58bf192471e |
| moe (testbed) | 1.7187 | 426/914 | 2.6196 | 2a7b5d2030894c4a |
| attention | 1.6340 | 417/868 | 3.0316 | |
| bf16 | 1.2740 | 388/1152 | 1.6367 | |
| prefill | 1.4897 | 177/516 | 2.0252 | |

- **The loop is stopped** (the user, 1505Z: "lets stop whatever is in flight"). Nothing was in flight. Restart it
  only when the user says so.
- **Code:**
  - master = 1a5563e2 at writing.
  - **`campaign/expert-bandwidth` HEAD 4c0f0253** (worktree `.worktrees/expert-bandwidth`) holds the code of both
    boards: decode's W6-G stack and moe's W6-M fork. It is the base for X5.
- **Quota at 1505Z on 10-27** (resets 00:00Z; 20 checks + 10 scored per problem per day):

  | problem | checks | scored |
  |---|---|---|
  | decode | 3 | 0 |
  | moe | 12 | 8 |
  | attention | 12 | 5 |
  | bf16 | 13 | 8 |
  | prefill | 13 | 8 |

- **Close:** `okc problems` says "Kernel submissions close Nov 4 at 11:59 pm PT". The working deadline is
  2026-11-05 06:59Z, with the last dispatch at ~06:30Z. DISTILL §4.5's end game applies on 11-04.

### 0.3 When the user says go: the first hour

1. `date -u` (never estimate a stamp). Read the LOG Now block, check the steer banners (top of
   `campaigns/INDEX.md`, the campaign `PLAN.md` and `CLAUDE.md`), then run `okc problems` and the leaderboards.
2. Create the branch and worktree:
   `git -C /home/user/git/musespark worktree add .worktrees/eb-sp -b campaign/eb-sp 4c0f0253`
   - If campaign/expert-bandwidth has moved past 4c0f0253 with landed board code, base on its HEAD instead and say
     so in the LOG.
3. Check Hopper availability: `cd ~/git/takemyjob/prototypes/hopper-probe && bin/hopper-probe status`. Available
   means `IDLE+DRAIN` with zero jobs (§9).
   - **A node is available:** run Gate 0 (§3), then build the stage-0 ladder.
   - **None is:** build Gate 0's variants into one moe judge check instead (≤ 20 kernel symbols; §4.2's constraints).
4. Write a dated LOG entry (X5 started, base commit, node used), and keep the Now block's user bullets.

### 0.4 Operating rules

**Dispatch and scoring:**
- **Attempt gate (hook `.claude/hooks/okc-attempt-gate.py`, on every Bash `okc check|submit`).** It needs an attempt
  file in the attempts/ directory of the checkout that owns the submission (a worktree's own attempts/), with these
  properties:
  - named `<YYYY-MM-DDTHHMMZ>-<check|submit>-<slug>.md`;
  - modified less than 2 h ago;
  - still holding the literal ``- **Job:** `<job id>` (`okc status <job>`)`` line;
  - ticking the rules checklist honestly against the diff.

  Templates are in `campaigns/2026-10-26-expert-bandwidth/scratchpad/` (`dqab_attempt.md` is the richest). The okc
  path must be absolute; `$PWD` is rejected.
- **Dispatch:** from the worktree, `python3 rewrite/bin/dispatch.py check|submit <problem> <attempt.md>`. It sends
  the generated `<problem>/submission.py`, refuses anything over 1,000,000 B, and writes the job id into the attempt
  file.
- **Wait and save:** `okc status <job> --wait --json > $SCRATCHPAD/runs/<job>.json`.
- **Archive:** `python3 rewrite/bin/archive.py <job>`. It hardcodes `SCRATCH_RUNS` to session 860eee0a's scratchpad,
  so point it at your own scratchpad's `runs/` first (an edit on the branch, or copy the json there).
- **Compare** rows against each run's own SGLang time:
  `python3 -I campaigns/2026-10-26-expert-bandwidth/scratchpad/rowcmp2.py OLD.json[.gz] NEW.json[.gz]` (decode)
  and `rowcmp_moe.py` (moe).
- **Never re-roll identical bytes** (okc dedups them anyway).
- **A check that beats its board gets a scored run of the same bytes,** with a new attempt file. A scored run that
  beats the board is landed on master as `<problem>/submission.py` and registered in
  `<problem>/hypotheses/BEST.json`.
- **Identical bytes score within about ±1.2%.** Judge changes on rows and on gate lines, not on the headline.

**Building:**
- `python3 rewrite/build.py [problem]` generates the submission from `rewrite/src/*.py` and `rewrite/lib`.
- `python3 rewrite/build.py --check` checks that it is current.
- `python3 rewrite/bin/cuda_syntax.py <submission>` does g++ -fsyntax-only on module-level CUDA strings. It
  **skips** names ending in `_HEAD` / `_BODY`, so name new sources `_ES_*_SRC`.
- Decode is at 975,200 B against okc's 1,000,000 cap. build.py strips Python comments above 990,000, but comments
  inside CUDA strings still count.

**Agents:**
- Phase-A work runs as ultracode Workflows. Design-panel and judge agents run at `effort:'max'`; builders, auditors
  and fixers at `effort:'high'`. Load `workflow-authoring` first.
- Template script: `~/.claude/projects/-home-user-git-musespark/860eee0a-753a-41f3-9c96-e9691adfb63e/workflows/scripts/phasea-wave5-2026-10-27-wf_df13b9f4-24c.js`
  (design x2 + judge -> build in a worktree -> two adversarial audits -> fix).
- Outside workflows, at most 3 concurrent agents.

**Log and judge-side traps:**
- Every decision line is re-printed at the end of prepare (the judge keeps only a 16,000-character log tail, which
  is already overflowing).
- Only read-flush gates (`rt.py _rt_time(flush="read")`).
- The worst draw decides big-T gates.
- One new routable cell per GEMM per check.
- Every new kernel's first judge launch happens in a **moe** check, with a named, synced canary.

**Cluster:** §9's rules are binding. Never change admin-host or Jumphost (Jumphost is transit only). Never
write under /newhome, /scratch, /phxscratch, /data, /packages or /sysadmin. Send nothing to external services.

### 0.5 Where results go

- A dated LOG entry per harvest (at most ~10 lines per run).
- Run archives in `campaigns/2026-10-26-expert-bandwidth/runs/`.
- The Gate-0 / stage-0 write-up in `campaigns/2026-10-26-expert-bandwidth/research/kstep-schedule-<date>.md`.
- Node runs in `~/git/takemyjob/prototypes/hopper-probe/runs/<ts>-<node>/out/`, indexed in `runs/INDEX.tsv`; copy
  the relevant `es-kstep.txt` into the research note.
- Lessons in `hypotheses/LESSONS.md`, one line each with its evidence.
- Commit on `campaign/eb-sp` at every working step. Land scored files on master at once.

## 1. Why

**The gap is structural.**
- The top 25 teams took a median of 10-17 scored runs; we took 21-37. Several top-10 rows took one run.
- The decode board runs smoothly from 2.40 (#1) to 2.06 (#50) and 1.63 (#300), so no single trick explains it.
- Gains under ~1.2% cannot be read on one judge run.
- Our own all-levers ceiling (~2.0) sits below the decode leader, so the leaders changed something structural.

**moe matches a bandwidth-rate engine.** An engine that streams each touched expert's weights once at ~4.2 TB/s,
with 6-10 us of fixed cost, models moe at 2.600 against the leader's 2.6196. The expert block is 48.9% of the
decode headline (`research/wave6-panel2-2026-10-27.md`).

**Two cautions:**
- A 4.2 TB/s expert block models decode at only ~2.006. The decode leaders are also ahead outside this block.
- At T<=32, Engine A already reads each expert once (streamed ≈ unique bytes; forensics F1). There, the lever is
  consumer rate and fixed costs.

**What runs where today** (races, so picks can flip between runs; code map 2026-10-27):

| T | decode | moe |
|---|---|---|
| 1 | Engine A v1 (K1 plain, K2 PDL, CUDA combine) | v1 |
| 2..8 | v4 | v4 |
| 9..32 | v4c | v4 / v4c |
| 33..127 (tests only) | Marlin | Marlin |
| 128 | H-16 hybrid: `wg_gather` -> wg w13 -> wg w2 -> ea4 K1 -> K2, serial, no PDL across the halves; plus W6-B cells; W6-G confirm race | Engine A v4c alone (hybrid 292 vs master 270 us) |
| 129..255 | hybrid (class 128) | Marlin |
| 256 | hybrid | forked hybrid (W6-M: hot on a side stream) |
| 257+ (tests) | n/a | `_moe_wgmma` |

Decode keeps three resident copies of the expert weights (Marlin, Engine A records, the wg bank).

**Measured rates and costs:**
- **Cold half (Engine A v4/v4c):** 2.57-3.03 TB/s in-layer.
  - The ladder (5eafb39e, e775df93): ring-only 4.46-4.61 TB/s; compute-only C-R4 3.98 TB/s-equivalent (≈ 27
    el/clk/SM) at 1 CTA/SM but 3.78 at 2/SM. More warps did not help.
- **Hot half** (`rewrite/lib/wg_core.py`, T=128/256): 0.96-1.54 TB/s, serial before the cold half (all = hot + cold
  + 2.5-5.7 us; `_hc_anatomy` lines in every decode log since 10-27 0030Z).
- **T<=32:** net streaming 2.57 / 2.71 / 3.12 / 3.17 TB/s at T=1/8/16/32, with no-route fixed costs of 5.5 / 9.2 /
  9.3 / 9.2 us (`_ead_anatomy`, c1daf58b).
  - T=1 phases (5b4e5b8f): ramp 3.9, w13 8.5, gap 3.8, w2 5.9, tail 2.3 us.
- **Bimodal replays.** In 2a7b5d20's `runs_ns`, moe T=1 splits 50/50 at ≤ 34.3 vs ≥ 44.6 us, and T=8 splits 47/53
  at ≤ 88.4 vs ≥ 97.7 us. The decode d1 rows have a tail of +8-9 us. Never investigated.

**W3-B hot-half decomposition** (moe check 8263c7bca04a406a, worktree `.worktrees/w3b-wgdiag`):

| variant | T=128 hot (us) | T=256 hot (us) | cold-dominated full draws (clk/stage) |
|---|---|---|---|
| D0, today's loop | 82.2 | 127.3 | 1192 |
| dequant, no MMA | 41.4 | 72.1 | 637 |
| MMA, constant A | 63.0 | 84.2 | 744 |
| ring only | 41.2 | 72.7 | 639 |

- The serial loop costs both components. Constant-A is ~93 clk per wgmma per warpgroup at N<=32, which reproduces
  Engine B's "64-98 clk whatever N". The cause is open.
- Variant B (double-buffered A + `wait<1>`) was serialized by ptxas (C7512), so overlap was never really tested.

**Why the hot half is slow** (code): `consume_item` (`wg_core.py:292-374`) runs dequant -> 8 wgmma -> commit ->
`wait<0>` -> release per stage. Both consumer WGs wait on the same full barrier and run in lockstep. Items use a
static stride (`:466`, `:492`), so 36-72 CTAs idle on few hot experts; W5 (48919184) showed the "N64 gain" was this
balance. Chunks are capped at 128 routes; there is no split-K.

**Killed in this campaign (do not retry as is):**

| line | evidence |
|---|---|
| v5/D1 wide mma.sync | tensor-issue bound, e403a42a |
| Engine B v1/v2 | serialized dequant and MMA in the same warps |
| wg2 re-split | x0.405 / 0.882 |
| variant B | not refuted, only never compiled as intended |
| wgx / exact-N | 48919184 |
| k-slice | e1406e5b |
| x-copy | e775df93 |
| dq3/dq4 | |
| the H-threshold sweep | |
| eat18 / eaf | |
| decode two-stream fork | block x0.930, layer x1.003 |
| W6-A last-arriver tails | |
| static SM splits | x1.10-1.64 slower |

**Prefill evidence on "ingest"** (`muse_prefill_layer/hypotheses/archive/*/*.log`). Cutting activation ingest made
the same register-A wgmma tile slower:

| change | L2->SM | result |
|---|---|---|
| `[rega-wide]` BM256 | 5.66 -> 2.89 TB/s | x1.07-1.20 slower |
| `[rega-mc]` 2-CTA multicast | | x1.31-1.81 slower |
| `[rega-pp]` ping-pong | | x1.18-1.47 slower |

The tile was consumer-limited (427-546 TFLOP/s). **No L2->SM ceiling has ever been measured.**

## 2. Value and ceilings

**Value** (judge formula; decode rows c1daf58b, moe rows 2a7b5d20; the expert µs from the 7ceab934 split; distinct
local experts from SoL inversion: decode 2.5 / 7.5 / 7.5 / 9.0 / 18.4 / 21.3, moe 2.58 / 8.43 / 11.87 / 16.0 / 22.9 /
24.6; 5-10 us of fixed cost; 28.31 MB per expert).

| scope / unique-byte rate | 2.8 TB/s | 3.2 TB/s | 3.6 TB/s (stretch) |
|---|---|---|---|
| T>=128 only (S1), decode | 1.703 (x1.069) | 1.744 (x1.096) | 1.780 (x1.118) |
| T>=128 only (S1), moe | 1.809 | 1.888 | 1.960 |
| + T=9..32 (S2), decode / moe | | 1.782 / 1.925 | 1.842 / 2.074 |
| all T at 3.8 / 4.2 TB/s, decode | 1.936 / 2.006 | | |
| all T at 3.8 / 4.2 TB/s, moe | 2.376 / 2.600 | | |

- **T<=32 rows are conditional on Gate 0** (§3). If it fails, X5 is S1a alone: decode about x1.04-1.07.
- **If every stage-0 row passes:** moe T=256 comes to ≈ max(HBM 145, consumer 136) + 8 ≈ 153 us (today 322.7).
- **Decode rank** (10-27 board): 1.63 ≈ #300, 1.76 ≈ #200, 1.92 ≈ #100, 2.06 ≈ #50.

**Per-SM rates at 1980 MHz:**
- HBM at 4.5 TB/s (the measured ring; 4.8 is the spec and the SoL convention) = 17.2 B/clk = 30.6 NVFP4 el/clk.
- Tensor m64nNk16 = N/2 clk, i.e. 2048/N el/clk (32 at N=64, 16 at 128, 8 at 256).
- dq2 dequant: ≈ 19.5 instructions/warp/k16 (13 on the 16-lane ALU pipe) + ≈ 3.5 of wgmma bookkeeping. That puts
  the ALU ceiling at ≈ 39.4 el/clk/SM, and HBM rate needs ≥ 78% ALU use (Engine A reaches 69%). wg_core's own
  dequant costs ≈ 27.5 instructions.

**Floors** (us; HBM at 4.5; tensor at 80%; consumer at 27 / 36 el/clk/SM):

| class | HBM | tensor | consumer 27 / 36 |
|---|---|---|---|
| decode d1 | 15.7 | 0.3 | 17.8 / 13.4 |
| decode v16 | 115.8 | 24.8 | 131.2 / 98.4 |
| decode v32 | 134.0 | 48.9 | 151.9 / 113.9 |
| moe v32 | 154.8 | 48.9 | 175.4 / 131.6 |

- **Measured ceilings beneath them:** wg_core's MMA path at small N (constant A, 724-808 clk per 16K-element stage ≈
  20-23 el/clk/SM ≈ 3.0-3.3 TB/s), and wg ring-only at 639 clk/stage (≈ 3.5-3.8 TB/s of weights).
- **B ingest per weight byte** = 3.56·N8/BM (N8/36 at BM=128); its ceiling is unmeasured (K7).

**Routing** (CPU sim of the judge's generators, `scratchpad/sim_routing.py` structure):
- At T=256, 68% of weight bytes are experts with N<=16, while 59% of routes are in N>=33.
- Max routes per expert, p50/p90:

  | | T=128 | T=256 |
  |---|---|---|
  | decode | 40/44 | 75/81 |
  | moe | 42/105 | 85/242 |

- moe's `_ea_judge_routings` draws in 2a7b5d20, as touched / max / routes:

  | | p10 | p50 | p90 |
  |---|---|---|---|
  | T=128 | 16/122/201 | 23/28/194 | 27/49/162 |
  | T=256 | 22/155/333 | 23/82/374 | 25/143/467 |

- Timed shapes never exceed N=256.

## 3. Gate 0: the experiment that decides the scope (~2-4 h on a drained Hopper node; no judge check)

**The question** (both advisors and the reviewer, narrowed to our swap-AB kernel): what is the minimum sustainable
cycles per K-step when a consumer warpgroup unpacks and scales NVFP4 into its own BF16 A registers while RS wgmma
consumes the previous fragments? Which resource stops overlap first as N, accumulator independence, A-ring depth,
wait depth, issuing warpgroups and the register split change? **The biggest risk it retires:** the wgmma consumer
cannot carry cold (small-N) units faster than Engine A, in which case the T<=32 value disappears.

### 3.1 What hopper-probe already showed

`hc` on gpunode06, 6 GPUs, 2026-10-27 09:01 (`~/git/takemyjob/prototypes/hopper-probe/runs/20261027-090137-gpunode06/out/hc.txt`):
- FULL_RATE_READY, eta 0.97-1.01, hot slowdown x1.083-1.086, cold ≤ x1.0.
- The hot warpgroup: packed FP4 -> unpack x scale -> FP16 smem tiles (no swizzle) -> 4x SS wgmma m64n64k16 ->
  `wait 0`.
- The cold warpgroup: a 4 x 16 KiB `cp.async.bulk` ring at 3.15 TB/s (94% of HBM3).
- **So a hot dequant+wgmma warpgroup and an HBM stream share an SM with almost no interference.** This is a strong
  prior for mixing. It does not test cold units that also dequantize.
- Not yet built in hc_probe: tensor-map TMA, a 3-warpgroup producer/consumer split, setmaxnreg, pipelined
  (non-wait-0) wgmma. Those are this experiment.

### 3.2 Steps

1. **Calibrate.** Port W3-B's DIAG kernels unchanged (`.worktrees/w3b-wgdiag/rewrite/lib/wg_core.py`: DIAG template
   lines ~1106-1320, variant B ~1151-1185) and the ladder's C-R1 ring and compute-only rungs
   (`.worktrees/w3c-xcopy/rewrite/lib/ea_ladder.py`, `rewrite/lib/cuda/ea_ladder.cuh`). Run them on the node.
   - The consumer-bound kernels must reproduce in clocks on an H100 SXM: D0 1192, constant-A 744 clk/stage.
   - Memory-bound ones differ: the H100's HBM3 reads 3.11 TB/s against the judge's ~4.5.
   - Record the SM count, clocks, power, MIG state and driver.
2. **Toolchain.** hopper-probe builds with `nvhpc/26.9-cuda-13.3-nompi`. The judge logs torch 2.14.1+cu130
   (LESSONS:132), so also build with a CUDA 13.0 module if Sol has one (the RC docs list `cuda-13.0.1`; check with
   `module avail` on the node). Every judge probe prints `.text` sha256 for the bridge (§4.2).
3. **The small-N kernel** (`es_g0`, one symbol per variant x N x tiles):
   - **Shape:** 384 threads = 3 warpgroups, `__launch_bounds__(384,1)`.
     - WG0 is the producer, `setmaxnreg.dec.sync.aligned.u32 40`. In constant mode it only fills smem once and
       idles.
     - WG1/WG2 are consumers, `setmaxnreg.inc.sync.aligned.u32 232`.
   - **B:** resident in smem in wg_core's layout (K-major, SWIZZLE_128B, two 64-k halves; `make_desc` with LBO 1,
     SBO 1024, +32 B per k16, as `wg_core.py:193-200, 314`). N in {8, 16, 32, 64}.
   - **A:** a constant bf16 pattern in registers (4 regs per k16 per thread), except variant (e).
   - **Loop:** ≥ 4096 k16 steps per warpgroup. clock64 (and %globaltimer) at loop entry and exit on lane 0 of each
     consumer warp. Accumulators are summed to global at the end, so the result is observable.
   - **Variants:**
     - (a) W3-B's pattern: 8 wgmma per 128-k stage -> commit -> `wait<0>`.
     - (b) `wait<1>`: commit per stage and per k16, keeping one group outstanding.
     - (c) 2 and 4 independent accumulator tiles (own A patterns) sharing one B descriptor per k16.
     - (d) the best of (a)-(c) with WG2 idle vs both WGs issuing.
     - (e) dq2 (`rewrite/lib/cuda/ea_common.cuh:64-127`, `ext_prop` / `scale_prop_rt`) producing A per k16 from a
       smem-resident ring of real-format records (Engine B 128-k layout, synthetic codes, scale bytes in
       [0x08, 0x7E]). It uses a k16 A ring of R=2 and `wait<1>`, combined with the better of (b)/(c).
     - (f) the "M-cell": Engine A's mma.sync consume on the same records, B fragments by ldmatrix from the swizzled
       box (pattern: `rewrite/lib/cuda/eb_kernels.cuh:341-399` cell m).
   - **Build:** `-O3 -lineinfo -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v`. Fail the build on any
     C75xx warning or local memory > 0. Dump `cuobjdump -res-usage` and `nvdisasm -gp -plr` per symbol.
   - **Output:** one line per variant:
     `KV v=<a..f> N=<n> tiles=<1|2|4> wgs=<1|2> clk_k16=<..> elclk=<..> regs=<..> spill=<..> c75xx=<0|1> ok=<0|1>`
     - el/clk/SM = 64 rows x 16 k x tiles x issuing WGs / clk_k16.
   - **Correctness:**
     - (a) vs (b) bitwise.
     - (c) per tile bitwise against (a) with the same A pattern.
     - (e) against a host dequant + fp32 GEMM reference at rel ≤ 1e-5.
     - (f) against (e) within tolerance.
     - compute-sanitizer memcheck and synccheck once per variant family.
4. **The register question.** From nvdisasm's life ranges, read the highest register index in the consumer region.
   Did ptxas size it at 232 (the setmaxnreg value) or at the 168 launch budget? (cudaFuncGetAttributes reads 168 for
   every W3-B variant, so it cannot tell.)
5. **Verdict** (record it in the LOG and in `research/kstep-schedule-<date>.md`):

   | (e) at N=16, device-wide | X5 scope |
   |---|---|
   | ≥ 34 el/clk/SM | cold in Engine S at all T: S1b, S2 and S3 in scope |
   | 30-34 | cold in Engine S at T>=128 only (S1b); T<=32 stays Engine A |
   | < 30 | cold stays on Engine A (or the M-cell if it wins). X5 = S1a + the T<=8 fixed-cost work (§5) |

## 4. Stage 0: the K-step ladder and the judge confirmation

**Deliverable:** the explained schedule per N regime (live / outstanding / waited / overlapped), with SASS and the
register map, written to `research/kstep-schedule-<date>.md`. A faster variant is not the deliverable.

### 4.1 Harness and rungs (Hopper node first)

**Harness:**
- Shared kernels in musespark: `rewrite/lib/cuda/es_common.cuh` (RS wrappers n8..n256, dq2 per k16, stamps) and
  `es_probe.cuh` (the rungs).
- Node driver as a new hopper-probe probe: `~/git/takemyjob/prototypes/hopper-probe/probes/es_kstep.cu` (host main,
  no torch, never shipped), with:
  - a synthetic Engine-B-layout bank;
  - driver-API tensor maps (`cuTensorMapEncodeTiled`, as `wg_core.py:537-559`);
  - a 256 MB read flush;
  - clock64.

**Wiring** (small commits in the takemyjob repo):
- A sync step copies the two `.cuh` from the musespark worktree into the probe build.
- `node/build.sh` and the probe list in `node/run.sh` learn `es-kstep`.
- `bin/hopper-probe run <node> es-kstep` then runs under the existing guard and audit.

**The judge build** is a moe-only `rewrite/lib/es_probe.py` including the same `.cuh`. Decode never includes it.

**Modes:** single-CTA long-running (≥ 4096 k16 per warpgroup, re-armed resident stages), then device-wide (≥ 2 GB
streamed).

| rung | content |
|---|---|
| K0 build | ptxas -v (regs, local memory, C75xx); SASS counts of HGMMA / WARPGROUP.DEPBAR (a `DEPBAR.LE gsb0,0x0` after every HGMMA = serialized); life ranges |
| K1 MMA-only | constant A, N=8..256: issue-to-completion latency; steady cadence committing every 1 and every 8; 2 WGs at once; A rewritten + fence each step; dependent vs independent accumulators; an SS control; W3-B D2-constA unchanged (must explain 744 clk/stage) |
| K2 dequant-only | 1-4 WGs: dq2, a load-only control (same LDS, trivial convert), an arithmetic-only control (register-resident words, same unpack/scale), the master form, dq4 |
| K3 serial | today's consume_item, the reference |
| K4 one-consumer pipeline | stage double-buffer vs k16 ring R=2 / R=4, on configurations S2 and S4/D2 (§6) |
| K5 ping-pong | own A / accumulators / tile per WG with an order barrier. Prior negative: it halves dequant concurrency, and prefill's lost |
| K6 hot/cold mixture | S3h: hot alone vs cold alone vs mixed, with each role's own slowdown |
| K7 ingest | an L2-resident stream (the ceiling); BM 128 vs 256 at N=128; an HBM stream at N=8 |
| K8 D3 | dequant WG -> smem bf16 -> SS wgmma (smem-capped near 24 el/clk at N=8). Hot-unit fallback only |
| K9 small-T fixed cost | the T=1/8 block vs no-route (v1's 5.2 us); a 1000-replay histogram of the moe T=1 / T=8 graphs under the read flush (the bimodal slow mode) |

**The K4 schedule to explain** (R=2, k16 step t, j = t mod 8):
1. If j=0, wait for the stage's full barrier.
2. If j mod 4 = 0, load the next record.
3. Dequantize step t into A[t&1]. Its last reader, group t-2, was retired by the previous wait.
4. Fence, wgmma, commit.
5. `wait_group<1>`, which retires t-1. Arrive on the stage's empty barrier when its last step retires.

Expected cost per k16 ≈ max(dequant, wgmma latency L(N), C·N/2 clk of shared tensor time), with C the number of
issuing warpgroups.

**Method.**
- **Overlap:** eta = (T_K1 + T_K2 - T_K4) / min(T_K1, T_K2). It is a diagnostic only, biased when the ring clips a
  component: no-MMA 637 = ring 639, while the true dequant cost is ≈ 448 clk (D0 - constA).
- **Stamped builds** are separate and must time within 5% of the unstamped build (volatile reads constrain ptxas).
  Lane 0 of each consumer warp records 4 events per k16 (full-returned, A-ready, committed, wait-returned) as u32
  deltas over 64 steps on 4 CTAs.
- **Correctness:**
  - bitwise across pure-pipelining schedules;
  - rel ≤ 1e-5 against a naive fp32 GEMM for reassociated variants (independent accumulators, stream-K);
  - `eal_dq_exact`;
  - compute-sanitizer memcheck / synccheck (racecheck is weak on TMA/wgmma async).
- **Recorded per variant:** a register-lifetime table (R_acc = N/2 per m64nN bank + A ring + packed + scales +
  addresses; producer + 2 consumers obey R_p + 2R_c ≤ 512), ptxas -v, life ranges, and useful vs padded MMA work.
  Report consumer-bound results in el/clk/SM, and node device-wide results as ratios against the incumbent on the
  same node.

**Decision table** (absolute, against Engine A: compute-only 3.98 TB/s-equivalent ≈ 27 el/clk/SM, C-R4 3.45 TB/s;
single-CTA unless noted):

| question | measure | go if | otherwise |
|---|---|---|---|
| small-N RS cost | K1 n8/n16 | latency ≤ 40, cadence ≤ 16 clk | M-cell for N ≤ 16 |
| tensor cadence | K1 N ≥ 64 | ≤ 1.1·N/2 clk | chunk hot units to 64 |
| dequant ceiling | K2 dq2 | ≥ 36 el/clk/SM | 31-36: ≥ 3 consumer WGs; < 31: M-cell for cold |
| overlap | K4 | eta ≥ 0.7 and C75xx-free | iterate on the node |
| cold K-step | K4 N=8/16, device-wide | ≥ 34 el/clk/SM | §3.2 step 5 |
| hot K-step | K4 N=64/128 | ≥ 25.6 / 12.8 el/clk/SM | cap hot N |
| ping-pong | K5 vs K4 | ≥ x1.05 | drop |
| mixing | K6 | mixed ≤ 0.85 x (hot + cold), each role ≤ x1.25 | keep halves separate (S2 pairs only; fork and SM-split forms lost before) |
| ingest | K7 on the judge | ceiling < 1.25x demand and BM256 ≥ x1.1 | BM=128 |

- The fastest legal variant wins; rule order never decides.
- **Fallback if RS overlap fails:** forensics D2 (a hot RS warpgroup plus cold mma.sync warpgroups in one CTA). Not
  K8 and not an SM partition.

### 4.2 The judge confirmation check (one moe check, at most two)

- **Contents:**
  - moe's ladder (`CAMPAIGN_LADDER`, `rewrite/src/muse_moe_w4a16.py:3216`, -39 s of prepare), Engine B and the
    other diagnostics off;
  - ≤ 12-20 kernel symbols (the chosen variants + runners-up) in threaded `load_inline` modules (threaded builds
    work on the judge: 3 modules in 23.8 s), with a build deadline;
  - output ≤ 40 lines of ≤ 110 characters, re-printed at the end of prepare.
- **R0 lines:**
  - `-Xptxas -v` and C75xx text, captured with W3-B's fd-1 redirect (`_wg_diag_build` in `.worktrees/w3b-wgdiag`;
    the main branch has only the `-Werror` probe trick, `wg_core.py:867-901`);
  - regs and local memory (cudaFuncGetAttributes);
  - torch.version.cuda and `CUDA_HOME/version.json` (read the file; no subprocess);
  - each kernel's `.text` sha256 plus an HGMMA / WARPGROUP.DEPBAR digest from W3-C's in-prepare cubin reader
    (`_eal_elf_sections` / `_eal_lz4` / `_eal_cubins` / `_eal_sass`, `.worktrees/w3c-xcopy/rewrite/lib/ea_ladder.py:388-538`).
    Its opcode table uses sm_8x names; calibrate the sm_90 HGMMA / WARPGROUP / SYNCS opcodes on the node first.
- **The bridge:** when the judge's hashes match the node's CUDA 13.0 builds, the node's SASS and life-range analysis
  applies to the judge's binaries.
- **Runs:** device-wide K4 calibration at N=8/16/128, then K7; an fp32 canary on `_ea_judge_routings` draws; the copy
  check.

## 5. Stages after the verdict

**Every stage gates per class:**
- read-flush graph timing over p10/p50/p90 judge-like draws, with the worst draw deciding;
- margin 1.03 (1.02 with a tight IQR; `rt.py:15-18, 121-148`);
- canary vs fp32 at half the judge bound (`_rt_check`);
- copy check (cloned states, alternating replays), two-state alternation, and the layer check in decode;
- the incumbent kept on any miss.

**S1a: hot list only** (the cold half stays Engine A inside the hybrid).
- Engine S kernels on hot experts. Raced as a new H code (e.g. `H-16s`) against H-16 in `HC_HOT_BSMS`
  (`rewrite/lib/hotcold_decode.py:20-26`) and `HM_BSMS` (`hotcold_moe.py:22`), with an `_hc_experts` branch (:248-329)
  and decode's W6-G race (:956-1059).
- **Contents:**
  - the C75xx-clean A ring;
  - stream-K balance over hot items;
  - N ≤ 256 in one pass (no chunk re-streaming);
  - dq2 on the `_eb_repack128` bank (`rewrite/lib/eb_core.py:122-141`; +906 MB GPU memory);
  - separate weight and B rings;
  - the `eb_prep` front end.
- **Target:** hot half ≤ 0.65 x D0 on judge-like draws. **Kill:** < x1.10 on the hot half after 2 judge checks.

**S1b: hot + cold lists in one launch** (only if Gate 0 ≥ 30 and K6 passes). It replaces the hybrid at T=128/256.
- **Target:** block ≥ x1.25 at T=256 and ≥ x1.15 at T=128, worst draw.
- **Calendar kill:** ≥ x1.10 block-level on the Hopper node (as a ratio to the incumbent on the same node) by the end
  of D2 and routed on the judge by D3, or X5 stops at S1a.

**S2: T=9..32** (only if Gate 0 ≥ 34 and S1b routed). Gate ≥ x1.05 per class.

**S3: T<=8.**
- K9 first, whatever the verdicts: the fixed cost and the bimodal slow mode.
- Then Engine S or the M-cell at N ≤ 8, if it wins.
- A fused w13 -> w2 spin only with a judge `runs_ns` bimodality check (precedent: 40% of bs=1 replays in a +23 us
  slow mode, LESSONS:156).

**S4: decode cash-in.**
- **A size-trim check comes first,** with identical routing. Decode is at 975,200 B; Engine S needs ≈ 45 KB.
  Candidates:
  - fallback-only `_DL1_*` 26 KB, `_WG_MASTER_REVERT` 7.7 KB, `_SK1_*` 2.6 KB;
  - print-only diagnostics 13 KB;
  - ea5 stubs;
  - v2/v3-only kernels ≈ 32 KB (v4 still needs `eak3::consume2`, `ea2_check`, `eak2::build_map`,
    `_ea3_forward_core`).
- Once S covers their classes, retire Engine A v1/v2 (46 KB) and decode's wg_core copy (53.5 KB).
- A scored run whenever a check beats the board.

## 6. Engine S design (to be finalised by the D0 design panel on Gate-0 evidence)

**Data path (swap-AB RS, as wg_core):**
- Weights are the register A operand: e2m1 unpacked and scaled (e4m3 block x fp32 global) to BF16 by the issuing
  warps. Hopper wgmma has no FP4 input.
- Activations are the smem B operand via TMA 2-D boxes, SWIZZLE_128B.
- The dequant is dq2 on Engine B's 128-k bank (a lossless permutation of Engine A's).

**CTA configurations** (setmaxnreg split; registers per consumer thread):

| config | split | consumer warps/SM | use |
|---|---|---|---|
| S2 | P40 + 2xC232 (as wg_core) | 8 | default |
| S3h | P32 + hot 216 + 2x cold 128 | 12 | if K6 says mix |
| S3u | P24 + 3xC160 | 12 | |
| S4 | P24 + 4xC112 | 16 | |
| D2 | 2x(P32 + 2xC104) per SM, proven on the judge | 16 | fallback |

**Live registers** (accumulator N/2 per m64nN bank + A + ≈ 38 other, the estimate to replace with nvdisasm):

| A schedule | N=16 | 64 | 128 | 256 |
|---|---|---|---|---|
| k16 ring R=2 (8 A regs) | 54 | 78 | 110 | 174 |
| stage double-buffer (variant B, 64 A regs) | 110 | 134 | 166 (C7512 seen at a 232 budget) | 230 |

- The model under-counts stage-granular buffers by ≥ 30 regs, so stay under 85% of budget.
- Two accumulator banks are impossible at N=256.

**C75xx-clean rules:**
- A is buffered per k16, not per stage.
- The descriptor advances +2 per k16; no descriptor arrays.
- No non-wgmma write to A or the accumulator between fence and wait: use scale-d=0 on the first wgmma, not a zero
  fill.
- No calls inside that window.
- One kernel symbol per variant and N bucket, since ptxas may serialize a whole function for one bad path.

**smem (S2):** a weight ring of 8 x 9.2 KB = 72 KB, a variable B arena of 96 KB (1 KB granules), 34 KB of epilogue
(N=256 in two halves) and an 8 KB map: 210 KB of 227. That holds about 2x the ≈ 34 KB in flight per SM that HBM
rate needs. Separate rings are required: wg_core's combined stage leaves only 2 stages at N=256.

**N handling:**
- Ingest exactly N8 = 8⌈N/8⌉ rows (box heights 8-128, ≤ 4 boxes per 64-k half). The wgmma shape is the pow2 bucket ≥
  N8, and padded columns are never stored.
- N ≤ 256 runs in one pass (n256 = 128 accumulator registers: legal on S2 and S3h-hot).
- N > 256 (the bs 257 test) is chunked; correctness only, not timed.
- N ≤ 8 is dequant-bound. Keep the M-cell only if it beats K4 by ≥ 5%.

**Ingest:** default BM=128. BM=256 only for 64 < N8 ≤ 128, and only if K7 says so. Multicast deferred (prefill
x1.31-1.81 slower; MIG pairing).

**Scheduler:**
- Unit = (GEMM, expert, N-chunk ≤ 256, 128-row tile, 128-k stage), with stream-K for hot and cold.
- Per-stage cost in SM-clocks = max(9216/β_mem, 16384/β_dq, 8·N8, (9216 + 256·N8)/β_ing). β_mem = 17.2; β_dq and
  β_ing come from stage 0. Costs are packed into Engine B's cost bytes (`eb_core.py:27, 69-70`).
- Hot (N8 > 64) and cold lists are each cut into 132 equal-cost ranges with Engine B's `first_unit` / `owner`
  (`rewrite/lib/cuda/eb_kernels.cuh:286-305, 579-582`). CTA b takes hot range b and cold range b, so every SM gets
  equal tensor and ALU demand.
- `build_map` runs on the device per call (`eb_kernels.cuh:208-284`), then weight prefetch, then
  `griddepcontrol.wait`. Graph-capturable.
- Engine B's last-arriver fix-up (`eb_kernels.cuh:463-509`).

**Front end and epilogues:**
- `eb_prep` gather (`eb_kernels.cuh:650-722`), one CTA per token, dense expert-sorted slots.
- w13: silu·up -> h[slot] (wg_core `consume_item` epilogue, `:325-374`).
- w2: rows -> y[route].
- `ea_combine` with PDL (`rewrite/lib/cuda/ea_combine.cuh`).
- Chain: eb_prep -> S<w13> -> S<w2> -> combine.

**Later (S3):** fused w13 -> w2 with `ready[e]` counters. It is deadlock-free because:
- the grid is ≤ resident capacity, checked in prepare;
- each CTA lists all its w13 units before any w2 unit;
- only the producer waits, after issuing its CTA's w13 loads;
- the spin is bounded and writes an error word.

The combine fold needs per-token tickets, gated per T (W6-A tails lost at T=1).

**Size:**
- **New:** `es_common` ≈ 6 KB, `es_probe` ≈ 18 KB (moe only), `es_kernels` ≈ 28 KB, `es_core.py` ≈ 20 KB.
- **Reused** ≈ 34 KB: `ea_common.cuh`; Engine B's wgmma fence/commit/wait, `make_desc`, `tma_2d`, `build_map`,
  `first_unit`/`owner`, fix-up and `eb_prep`; wg_core's epilogues and bounded waits; `ea_combine`.
- moe: ≈ 70 KB of 348 KB headroom. decode: ≈ 45 KB of 24.8 KB headroom, hence S4's trim first.

## 7. Schedule (D0 = the first day after the user's go)

| day | work | judge checks (decode / moe) |
|---|---|---|
| D0 | Hopper session 1: calibration, Gate 0, the register question (~4 h). Then the design panel (effort max) on that evidence, the ladder build (high) + audit (high), node sweeps, and `research/kstep-schedule-<date>.md` | 0 / 0-1 |
| D1 | the judge confirmation check (§4.2); S1a built and tested on the node (sanitizer, correctness, clocks); S1a moe checks | 0 / 2-4 |
| D2 | the decode trim check; S1a into decode; S1b on the node if in scope; calendar review at day end | 2-3 / 3-4 |
| D3 | S1b on the judge, or S3's fixed-cost work; S2 if in scope | 3 / 3 |
| D4+ | S2 / S3; the decode stack; scored runs; the end game on 11-04 | as earned |

- **Daily rhythm:** this is phase-A work. Phase B/C spend the leftover checks as usual.
- **Fallback queue:** gate-scale decode items stay queued for any day X5 stalls:
  - PERHEAD d1s;
  - s2b at v16s;
  - the T=1/8 glue folds;
  - SKW full-line + b16;
  - the size trim.
- **Slip order:** S3's fusion, then S2, then S1b. Gate 0 and S1a are protected.
- **Node time is opportunistic.** Run `bin/hopper-probe status` before each session. With no drained node, run the
  day's rungs on the judge (≤ 20 symbols per check).

## 8. Verification

**On a drained Hopper node, before every judge check:**
1. Build, failing on any C75xx or local memory > 0.
2. Read the SASS and life ranges.
3. Check correctness on judge-like routings, from a `scratchpad/sim_routing.py` port that writes `.bin` inputs at
   T ∈ {1,4,8,16,32,33,128,256,257}, plus the extremes: all-local, one expert at N=T, no local route, the moe p10
   draw with N=122.
4. Run compute-sanitizer.
5. Take graph timings in clocks.

**Locally (no GPU on this workstation, by the user's choice):**
- `rewrite/build.py --check`, `okc_lint` (via build.py), `py_compile`, `rewrite/bin/cuda_syntax.py`.
- CPU emulators following `rewrite/bin/b12p_emul.py`, `scratchpad/emu/ea2_map_emu.py` and `scratchpad/dqcheck/`, with
  the real repack loaded via AST:
  - the scheduler: unit coverage, the stream-K partition and tickets;
  - the bank: exact per-lane fragments;
  - the ring: expect_tx bytes per N and a deadlock-free mbarrier model;
  - align + gather against a Python reference.

**On the judge, in every check:**
- a named, synced first launch (`_rt_tracer`);
- bounded mbarrier waits writing an error word, and a launch marker;
- scratch poisoned (NaN / `HC_JUNK`);
- canary vs fp32 on judge-like and extreme routings, with exact zeros for tokens with no local route;
- the copy check and two-state alternation; the layer check in decode;
- the R0 lines;
- decisions re-printed last.

**Harvest:**
- rows against their own SGLang time, never the headline;
- the scored profile rows too (a cold-L2 gate once showed a gain the judged warm run lacked);
- `runs_ns` checked for bimodality;
- an attempt file and rules checklist before every dispatch.

**End game:** the nearby-shape sweep (T=129..255; the bs 4 / 33 / 257 / skewed tests), diagnostics off in the final
bytes (including `_ead_anatomy`), and the rules version re-checked (2026-10-22.1).

## 9. Hopper probing: the mandate, the rules and the tool

**Source:** `~/git/takemyjob/prototypes/hopper-probe/HANDOFF.md` and `NOTES.md` (2026-10-27). The standing rule is
also in the takemyjob project's Claude memory (`hopper-opportunistic-probes`).

**Mandate (operator the operator's words):** "all available h100/h200 time can be used outside slurm so long as no user
jobs are impacted." "drain == available." In this session (2026-10-27) the user also granted gpunode26 while it drains
("you have full freedom while it is draining to test probes"; per-job approval).

**Rules:**
- **Available** = Slurm `IDLE+DRAIN` (any extra flags) and zero jobs. Not `DRAINING` (`MIXED+DRAIN` /
  `ALLOCATED+DRAIN`), `MIXED`, `ALLOCATED` or `PLANNED`, and not a free MIG slice on a live node.
- **User jobs always win.** Kill our probes the moment a node goes live or a foreign GPU process appears. Never
  cancel, requeue, hold or slow a user job. Never resume or drain a node: that is a Slurm change needing the operator's
  explicit per-action approval.
- Run as the operator, never root, with no Slurm allocation.
- Node-local only: `/tmp/USER-hopper/`, removed after each run. Nothing under /home, /scratch, /data, /packages or
  /sysadmin; no uploads to external services.
- On nodes drained for "hardware troubleshooting": thermal guard on, stay off GPUs hot at idle, no GPU resets or
  reconfiguration, no reboots.
- Not planned without the user's approval: installing a cu130 torch into a node-local venv to run moe's `prepare()`
  / `reference.py` on the node (it pulls packages onto a compute node).

**Hardware** (verified 2026-10-27; all Sol, driver 615.71.09, compute capability 9.0, so one sm_90a build runs
everywhere):

| node | IP | GPUs | state then |
|---|---|---|---|
| gpunode06 | 10.x.x.x | 8x H100 80GB HBM3 SXM, 700 W, NVSwitch | drained by root 10-26 15:36 "hardware troubleshooting"; GPU0 78-79 °C and GPU2 62 °C at idle, excluded by `--gpus auto` |
| gpunode26 | 10.x.x.x | 4x H200 NVL, all MIG: 1x 7g.141gb, 2x (4g.71gb + 3g.71gb), 3x 2g.35gb | production per the HANDOFF; draining per the user on 10-27; MIG stays on (turning it off is an admin change); best rubric there is MECHANISM_READY |
| gpunode16 / gpunode20 | 10.x.x.x / .50 | H100 SXM | production |
| gpunode21 / gpunode24 / gpunode28 | 10.x.x.x / .74 / .88 | H100 NVL | production |

The user called gpunode26 "an h100 node"; per the HANDOFF the drained H100 SXM node is gpunode06. The guard's state check
governs either way. Phoenix and Dev have no Hopper GPUs, and there is no B200/B300 anywhere.

**What transfers to the judge (H200 SXM, 700 W, 1980 MHz):**
- **H100 SXM (gpunode06):** the same GH100 SM, 132 SMs, power and clock class. Per-SM cycles transfer; HBM does not
  (3.11 vs ~4.5 TB/s).
- **H200 NVL (gpunode26):** the judge's HBM3e class, but under MIG and at 600 W.
- **Absolute rates** come from the judge confirmation check.

**Access:**
- `ssh <IP>`: `~/.ssh/config` routes 10.x.* through the jumphost jump (transit only). Host keys are in
  `~/.ssh/known_hosts` (added 10-27; one image-wide ed25519 key). Bare node names do not resolve on the workstation.
- Login works with no job (`pam_slurm_allowed`). An ssh session is not cgroup-confined (it sees every GPU), so the
  guard is what keeps us off users.
- Read-only Slurm queries: `~/git/takemyjob/bin/slurm-ro.sh sol '<query>'`. Every probe run goes through
  `~/git/takemyjob/bin/audit-run.sh` (logged).

**The tool:**
```bash
cd ~/git/takemyjob/prototypes/hopper-probe
bin/hopper-probe status                 # all 7 nodes: state, job count, AVAILABLE|busy
bin/hopper-probe build [host]           # compile on a node (CPU only, ~17 s; nvhpc/26.9-cuda-13.3) -> build/bundle.tgz
bin/hopper-probe run <node> [--gpus auto|0,3|MIG-uuid,...] [--tmax 85] [probe ...]
```
- **`run`:** availability check -> ships `build/bundle.tgz` + `node/` to `/tmp/USER-hopper/run-<ts>/` -> runs
  `audit-run.sh <IP> "bash .../node/run.sh ..."` -> pulls `out/` to `runs/<ts>-<node>/out/` -> `rm -rf` the node dir
  -> appends `runs/INDEX.tsv`.
- **The guard** (`node/run.sh`, before the first probe and every 2 s):
  - `scontrol` state `IDLE*` + `DRAIN`;
  - an empty `squeue -w <node>`, which must succeed;
  - no GPU compute process owned by anyone but USER;
  - every used GPU under `--tmax` °C.

  Any miss kills the probe process group and writes `out/STATUS = ABORTED <reason>`; so does an ssh HUP.
- **Overhead:** ~25 s of ssh hops per run (a ControlMaster for 10.x.x.x would cut it). The smallest useful
  window is `run <node> hc-quick`, about 1 s of GPU time.
- **Already measured on gpunode06:**
  - hc FULL_RATE_READY (above);
  - HBM read 3.11 TB/s, L2 cliff after 50 MiB;
  - BF16 cuBLASLt 660-800 TFLOPS;
  - p2p 397 / 782 GB/s.
- **Not yet run:** any H200 or the MIG path, H100 NVL nodes, GPU0/2 on gpunode06.
- **X5 adds:** the `es-kstep` probe and a CUDA 13.0 build option. Per session: `status`, then `run <node> es-kstep`
  with the variant list. State each session's first run to the user, and log the session in the work record
  (`~/git/takemyjob/private/history/<week>/<date>.md`, with `tags: node:<name>`).

## 10. The two external advisors, judged for our kernel

The key fact: wg_core is swap-AB. Weights are the register A operand, dequantized by the warps that issue the wgmma;
activations are the smem B operand.

| point | advisor 1 | advisor 2 | verdict |
|---|---|---|---|
| deliverable | an explained K-step schedule | the first resource that blocks overlap | agree; adopted |
| overlap designs | RS independent consumers vs an SS dequant producer are different experiments; no cross-WG register handoff | "one WG unpacks while another runs RS WGMMA" | **advisor 1 right.** Advisor 2's design exists only as SS (K8) |
| operand roles | (none) | weights = smem B; bottleneck = expanded-weight stores vs wgmma reads | **wrong for us**; applies only to K8 |
| no FP4 in Hopper wgmma | implicit | explicit (bf16 m64nNk16; fp8 k32) | correct; already how the dequant works |
| "64-98 clk whatever N" | not a constant: separate completion latency, steady cadence and K-step time | same, plus H800 priors (128/64/32/16/13/13 clk at N=256..8) | right in principle. W3-B reproduces ~93 clk per wgmma at N<=32 with constant A, 5-7x the H800 prior; the cause is open (Gate 0 / K1) |
| registers | lifetime is central; pipelining needs no second accumulator bank; setmaxnreg only redistributes | R_p + 2R_c ≤ 512; two banks impossible at N=256, "explains variant B" | arithmetic right. **The variant-B explanation is wrong**: it had one bank and doubled A (W3-B wg_core.py:1151-1185). Likeliest cause: stage-granular A + temporaries. ptxas sizing (232 vs 168) is settled in Gate 0 step 4 |
| ladder | MMA / dq / serial / 1-consumer / ping-pong / mixture | R0 SASS, R1, R2 + controls, R3 eta, R4, R5 per-role | merged. eta is a diagnostic; decisions are absolute vs Engine A. Ping-pong's prior is negative (prefill) |
| hot TB/s vs floor | hot work may be compute-limited | (none) | right; per-unit floors |
| HBM | 4.8 TB/s spec | 4.8 TB/s spec | adopted; 4.5 is labelled as the measured ring |
| SASS | offline | offline with exact flags | on a drained Hopper node via hopper-probe with CUDA 13.0, tied to the judge by `.text` hashes |

**What both miss,** and the plan covers:
- load balance (W5): the scheduler;
- the cold-path ceiling, which carries 68% of weight bytes even at T=256: Gate 0 / K2;
- the unmeasured ingest ceiling (prefill shows cutting ingest made things slower): K7;
- the bimodal T=1/8 replays: K9.

## 11. Risks and fallbacks

1. **The wgmma consumer cannot carry cold units faster than v4c** (the biggest risk). Gate 0 retires it in ~2-4 h. If
   it fails: X5 = S1a + T<=8 fixed costs (decode ≈ x1.04-1.07).
2. **No C75xx-clean overlap at wide N.** Follow §6's rules, with one symbol per variant. Fallback: D2 in one CTA.
3. **Node-to-judge transfer.** Compare engines by ratio on the same node, keep consumer-bound results in el/clk/SM,
   bridge SASS by hash, and let the judge confirmation settle absolute rates.
4. **First-launch faults on the judge** (cf. b12p 8e3c26d6, a diagnostic's use-after-free). Every kernel runs clean on
   the node under compute-sanitizer first. Bounded waits and synced canaries on the judge; first launches go in moe
   checks.
5. **Budgets:**
   - prepare 300 s (moe 164 s, decode 121 s today; each module ≈ 34-61 s to compile; ≤ 20 probe symbols, threaded);
   - the 16 KB log tail (≤ 40 short lines);
   - decode size (trim first).
6. **Schedule.** Builders overestimated 5 of 8 builds on 10-27. Calendar kills and the slip order apply.
7. **Hopper availability.** The guard kills our probes when a node goes live; with no drained node, rungs move to
   judge checks.
8. **Rules:**
   - lossless retiles only (a 4th resident weight copy until old engines retire);
   - fp32 accumulation and fp32 split partials; h rounded to bf16 as SGLang does;
   - device-built maps and in-kernel N dispatch (no host branching on input values);
   - nothing cached across calls; CUDA-graph capturable;
   - no subprocess, network or reference import in submissions (the lint rejects them).

## 12. Reference index (code at `campaign/expert-bandwidth` 4c0f0253, paths under `.worktrees/expert-bandwidth/`)

| what | where |
|---|---|
| wgmma hot path | `rewrite/lib/wg_core.py`: header 1-31; constants 33-53; `make_desc` 193-200; `dq_stage` 270-287; `consume_item` 292-374; `wg_kernel` 376-513 (producer 461-484, consumers 486-505); `wg_gather` 515-535; tensor maps 537-559, 641-655; RS wrappers 691-715; ptxas `-Werror` probe 867-901; `_wg_build` 884-945; bank 959-1026; scratch 1095-1102 |
| Engine A | `rewrite/lib/cuda/ea_common.cuh` (mbarrier 10-42, bulk copy 44-47, PDL 48/51, dequant 64-127); `ea4_kernels.cuh` (CTA 26-32, map 58-127, stream-K 158-209, producer 221-248, fix-up 301-347, epilogue 352-376); `rewrite/lib/ea_core.py` (`_ea_pack` / `_ea_repack` 30-85, `_ea_scales_ok` 51-57) |
| Engine B (unrouted; reusable) | `rewrite/lib/cuda/eb_kernels.cuh` (`make_desc` / EbWg 76-168, `build_map` 206-284, `first_unit` / `owner` 286-305 and 579-582, cell m 341-399, cell w 400-446, fix-up 463-509, `eb_prep` 650-722, tensor maps 729-747 and 856-871); `rewrite/lib/eb_core.py` (`EB_COSTS` 27 and 69-70, `_eb_repack128` 122-141, `_eb_ref_ids` 202-217); `rewrite/lib/eb_moe.py` (`_eb_judge_draw` 44-55) |
| hybrids | `rewrite/lib/hotcold_decode.py` (H codes 20-26, `_hc_experts` 248-329, `_hc_anatomy` 343-384, kernel check 507-588, layer gate 591-618, race 763-803, W6-G 956-1059); `rewrite/lib/hotcold_moe.py` (`HM_BSMS` 22, `_hm_gate` 450-477, fork 775-893) |
| timing / race / checks | `rewrite/lib/rt.py` (`_rt_time` 80-118, `_rt_select` 121-148, `_rt_check` 169-181, `_rt_tracer` 184-189, `_rt_build` 192-218) |
| kernel_fn dispatch | decode `rewrite/src/muse_decode_layer.py` 10084-10140 (FINAL PATHS 9750-9754); moe `rewrite/src/muse_moe_w4a16.py` 3176-3212 (`CAMPAIGN_LADDER` 3216, `CAMPAIGN_ENGINE_B` 3227) |
| judge-like routings | moe `_ea_judge_routings` (`rewrite/lib/ea_moe.py:62-77`); decode `_cb_basis` / `_cb_data` (`muse_decode_layer.py:6782-6924`), `_ead_cases` (`ea_decode.py:367-382`) |
| W3-B DIAG / fd-1 ptxas capture | `.worktrees/w3b-wgdiag/rewrite/lib/wg_core.py` (DIAG ~1106-1320; variant B ~1151-1185; `_wg_diag_build`) |
| in-prepare cubin reader | `.worktrees/w3c-xcopy/rewrite/lib/ea_ladder.py:388-538` |
| ladder harness | `rewrite/lib/ea_ladder.py` + `rewrite/lib/cuda/ea_ladder.cuh` (`EAL_RUNGS`, X-macro `EAL_KERNELS`) |
| key runs (`campaigns/2026-10-26-expert-bandwidth/runs/`) | c1daf58bf192471e decode board; 2a7b5d2030894c4a moe board; 7ceab934eea24099 decode (expert split); 8263c7bca04a406a W3-B; 5eafb39e194d4d8b ladder-2; e775df93* W3-C; 48919184* W5; e403a42a* v5 |
| research | `research/expert_engine_forensics.md`, `wave3-panel-2026-10-27.md`, `wave6-panel2-2026-10-27.md`, `secondary_levers.md`, `openclaw-relay-2026-10-26.md` |
| lessons cited | `hypotheses/LESSONS.md:72` (small-M kernels; bimodal spins), `:132` (judge torch 2.14.1+cu130), `:156` (bimodal bs=1 replays), `:179-199` (prefill rega lessons, fork, fast silu) |

## 13. Open decisions for the user (record answers in the LOG)

1. Go / no-go and the start time.
2. Whether a cu130 torch may be installed in a node-local venv on a drained Hopper node, to run moe's `prepare()` /
   `reference.py` there.
3. Whether this file should also be linked from the campaign `PLAN.md` banner and the LOG Now block. Not done at
   writing; the user asked only for this durable file.
