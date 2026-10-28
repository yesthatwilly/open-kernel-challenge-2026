# H100 node ledger (gpunode06): X5 / x5-h100-program

This is the H100 node ledger of the X5 campaign and its x5-h100-program. Every record carries the MANIFEST sha256 of
the bytes it ran, the GPU, cold or warm caches, and the SM clock under load. A run below 1800 MHz under load, or one
that throttled, is void.

Conventions for the records below:
- Node: gpunode06, 8x H100 80GB HBM3 SXM, driver 615.71.09. Root drained it on 2026-10-26 15:36 for "hardware
  troubleshooting". Runs go through takemyjob `prototypes/hopper-probe` (`bin/hopper-queue` and the on-node guard), as
  USER, outside Slurm.
- Run dirs live under `~/git/takemyjob/prototypes/hopper-probe/runs/` and are named in local time (MST = UTC-7).
  `queue/HISTORY.tsv` uses UTC.
- Caches: `warm` means the env's build cache (`fresh_build` false), so prepare time and the deadline-driven prepare
  phases are not judge-like. `cold` means empty per-run build dirs. Every nodejudge timed replay runs after a 256 MiB L2
  read flush, so L2 is cold for every replay.
- SM clock: nodejudge takes an nvidia-smi snapshot before and after a run (1830 MHz both times on every record below).
  It does not sample during the timed loop, so the 1800 MHz rule is **unverified** for nodejudge timings: they are not
  void and not proven valid either. Each esl KV line records the clock it measured under load (`mhz`); lines below
  1800 MHz are marked void.
- Harness: the 10-27 nodejudge runs used the committed harness, takemyjob HEAD 0c0bc56. The tracebacks in the 24-* logs
  match its 1009-line `nodejudge.py` line for line.
- **Bytes: every nodejudge run below ran moe `board.py` sha256 ab46fbb722c2 (641,473 B).** That is the board from before
  a4d1c81d (the W6-M fork, scored 1.7187). It is **not** the current board, master `muse_moe_w4a16/submission.py`
  498afbd8e73d (906,398 B, 1.7225, landed 10-27 21:57Z).

## Node state, 2026-10-28 00:48Z (read-only check before the redeploy)

- Slurm `IDLE+DRAIN`, 0 jobs, Reason `hardware troubleshooting [root@2026-10-26T15:36:35]`. No compute process on any
  GPU. nv-hostengine and dcgm-exporter run as services; no DCGM diag (`nvvs`) was running.
- Idle temperatures: GPU0 86 C (229 W at 0% utilisation, 1815 MHz, **SW thermal slowdown active**, hotter than the
  78-79 C seen on 10-27), GPU2 65 C, GPUs 1/3/4/5/6/7 29-37 C at about 120 W. The runner's 55 C idle limit already
  excludes GPU0 and GPU2. GPU 7 is reserved (`queue/reserved`).
- `/tmp/USER-hopper` is absent: `clean gpunode06 nodejudge` removed the env on 10-27 at 21:25Z.
- The redeploy was **held at about 01:00Z**. `setup` would ship the uncommitted nodejudge edits (E1/E2, mid-build) and
  run them. The runner would ship `build/nodejudge.tgz`, which was rebuilt at 21:23:42Z with an earlier partial E1
  `nodejudge.py` (sha256 c2caa74f50ce, neither HEAD nor the working tree) and still packs board ab46fbb722c2. No
  hopper-probe GPU run has happened since 21:23Z (`runs/INDEX.tsv`).

## Harvest: queue jobs that finished 2026-10-27 19:38Z-21:23Z

The 23 jobs in `queue/done/`, plus the 8 fuzz-filler runs (2 of them finished but were never reaped). Fault counts
cover test failures, plus miscompares on timed replays (each timed replay is checked against the fp32 reference),
plus sanitizer errors. In every nodejudge run, bytes = board ab46fbb722c2 and GPU = H100 SXM. Clock "snap" = 1830 MHz
before and after, with nothing sampled under load.

| job | run dir | GPU | caches | clock | verdict | faults | key numbers |
|---|---|---|---|---|---|---|---|
| 30-h100-nj-moe-board-flips-1 | 20261027-123454-gpunode06 | 1 | cold | snap | PASS | 0: tests 5/5, 0/600 replays | headline x1.3929 [1.3720, 1.4138] vs SGLang; prepare 164.4 s; the one gate flip (Engine A v1 at T=1..8) |
| 30-h100-nj-moe-board-flips-2 | 20261027-123457-gpunode06 | 3 | cold | snap | PASS | 0: 5/5, 0/600 | x1.3893 [1.3685, 1.4101]; prepare 163.0 s |
| 30-h100-nj-moe-board-flips-3 | 20261027-123459-gpunode06 | 4 | cold | snap | PASS | 0: 5/5, 0/600 | x1.3908 [1.3699, 1.4117]; prepare 163.0 s |
| 30-h100-nj-moe-board-flips-4 | 20261027-123502-gpunode06 | 5 | cold | snap | PASS | 0: 5/5, 0/600 | x1.3921 [1.3712, 1.4130]; prepare 161.5 s |
| 25-h100-nj-moe-sweep | 20261027-125905-gpunode06 | 1 | warm | snap | PASS | 0: tests 37/37, 0/120 | 32 held-out shapes (27 bs in 2..255, 5 skewed) plus the judge 5, all within tolerance; x1.3835 |
| 26-h100-nj-moe-soak-1 | 20261027-125907-gpunode06 | 3 | warm | snap | PASS | 0: 5/5, 0/6000 | 1000 checked reps per shape x 6 shapes; x1.3900 [1.3691, 1.4108] |
| 26-h100-nj-moe-soak-2 | 20261027-125910-gpunode06 | 4 | warm | snap | PASS | 0: 5/5, 0/6000 | x1.3862 [1.3654, 1.4070] |
| 22-h100-nj-moe-bigt-1 | 20261027-141313-gpunode06 | 1 | warm | snap | PASS | 0: 5/5, 0/2000 | T=128 / T=256 raw median 297.9 / 349.8 us; 8.61 / 8.59 us per distinct local expert (R2 0.84 / 0.93) |
| 22-h100-nj-moe-bigt-2 | 20261027-141316-gpunode06 | 3 | warm | snap | PASS | 0: 5/5, 0/2000 | 299.5 / 348.4 us; 8.67 / 8.51 us per expert (R2 0.84 / 0.92) |
| 22-h100-nj-moe-k9-1 | 20261027-141318-gpunode06 | 4 | warm | snap | PASS | 0: 5/5, 0/6000 | per-expert OLS slope T=1/8/16: 11.34 / 9.89 / 9.56 us (K9 section) |
| 22-h100-nj-moe-k9-2 | 20261027-141321-gpunode06 | 5 | warm | snap | PASS | 0: 5/5, 0/6000 | 11.63 / 10.12 / 9.63 us |
| 24-h100-nj-moe-san-ea4 | 20261027-141326-gpunode06 | 6 | warm | before 1830 | **VOID** (OOM fallbacks) | memcheck 0 errors; harness OOM: tests 3/5, no timing | kernel filter `kns=ea4_`; prepare fell back under memcheck (79.2 GiB of 80) |
| 24-h100-nj-moe-san-ea_combine | 20261027-141530-gpunode06 | 1 | warm | - | **VOID** (OOM) | memcheck 0 errors; prepare aborted (OOM), 0 tests | `kns=ea_combine` |
| 24-h100-nj-moe-san-ea_w | 20261027-141533-gpunode06 | 3 | warm | - | **VOID** (OOM) | memcheck 0 errors; prepare aborted (OOM), 0 tests | `kns=ea_w` |
| 24-h100-nj-moe-san-wg | 20261027-141535-gpunode06 | 4 | warm | before 1830 | **VOID** (OOM fallbacks) | memcheck 0 errors; tests 2/5 (3 OOM), no timing | `kns=wg_` |
| 20-h100-esl-cold-rep1 | 20261027-133349-gpunode06 | 1,3 | n/a | 1810-1830 | PASS | 0: KV 200/200 ok, 4 sanitizer runs 0 errors | P1: es streamed N=16 23.2-24.6 el/clk/SM; e devres ceilings N=16 27.2-30.4 |
| 20-h100-esl-cold-rep2 | 20261027-133352-gpunode06 | 4,5 | n/a | 1802-1830 | PASS | 0: 200/200, 0 | matches rep1: clk_k16 median diff 0.04%, p90 0.86%, max 5.9% |
| 20-h100-esl-k4-rep1 | 20261027-133552-gpunode06 | 1,3 | n/a | 1679-1830; **30/192 lines void** | PASS | 0: 192/192, 6 sanitizer runs 0 errors | cta r2 N=64: S2 14.2-14.4, D2 14.5-14.6, S4 18.4 (target 25.6); N=128 S2 12.5-12.6 (target 12.8); N=256 7.9 |
| 20-h100-esl-k4-rep2 | 20261027-133555-gpunode06 | 4,5 | n/a | 1665-1830; **24/192 void** | PASS | 0: 192/192, 0 | matches rep1: median 0.03%, p90 0.38%, max 5.6% |
| 21-h100-esl-k1-rep1 | 20261027-133757-gpunode06 | 1,3 | n/a | 1803-1830 | PASS | 0: 240/240, 10 sanitizer runs 0 errors | ring-free K1 (cta, tc130) cad8: N=8/16 78.7, N=32 64.0, N=64 32.0, N=128 16.0, N=256 8.0 el/clk/SM |
| 21-h100-esl-k2-rep1 | 20261027-133800-gpunode06 | 4,5 | n/a | 1830 | PASS | 0: 280/280; K2CHECK 8 x 65,536 cases, 0 mismatches; 4 sanitizer runs 0 errors | dq2 best 34.6-35.5 el/clk/SM (imad / sts sinks, 3-4 WGs) vs the ~39.4 ALU ceiling; ar fold4 about 30 |
| 21-h100-esl-k6-rep1 | 20261027-133959-gpunode06 | 1,3 | n/a | 1830 (timed lines) | PASS, **K6 NO-GO** | 0: 64/64, 2 sanitizer runs 0 errors | K6GO 0/16: mixed/(hot+cold) 0.83-0.93, but hot slowdown 1.66-1.87 and cold 1.14-1.52 against the x1.25 limit |
| 21-h100-esl-k7-rep1 | 20261027-134002-gpunode06 | 4,5 | n/a | 1830 | PASS | 0: 96/96, 2 sanitizer runs 0 errors | ingest N=128 hbm-own 90-101 B/clk/SM, i.e. 39.5-45.0 el/clk/SM at BM=128; l2-shared 105.7-107.4 B/clk/SM; N=256 hbm-own 83-89, i.e. 19.4-20.9 el/clk/SM |
| 90-h100-nj-moe-fuzz-odd x4 | 20261027-141634 / -141819 / -142000 / -142210 (the last unreaped) | 5 / 1 / 1 / 1 | warm | snap | PASS | 0: 53/53 tests in each run (212 cases), 0/40 replays | 48 odd/prime bs plus the judge 5, a fresh seed each run |
| 90-h100-nj-moe-fuzz-skew x4 | 20261027-141637 / -141822 / -142002 / -142213 (the last unreaped) | 6 / 3 / 3 / 3 | warm | snap | PASS | 0: 27/27 in each run (108 cases), 0/40 | 22 skewed-routing bs plus the judge 5 |

The esl bundle stamp is 20261027T202132Z-src9081729ad153, with binary hashes esl_cold 130/133 b480f8b7f55250cf /
e1c34525c7dc3aba, esl_hot (k4/k6/k7) 2c4defd247220ed3 / 3d5c466d4971232a, esl_k1 9fc5d14e04b08777 / 76cbf6f3b1a3c45f,
and esl_k2 146a70f383a3eb18 / 2517440ff1035e84. The 20:23-20:29Z validation runs used the same stamp, and the
replicates' SASS `text=` hashes match them line for line.

### S3 / K9: replay time against distinct local experts at T=1/8/16 (22-h100-nj-moe-k9-1/2)

2 x 2000 reps per shape, raw replay medians in us. The empty-graph fixed cost (3.6-3.9 us) is not subtracted and does
not affect the slopes. Each cell gives the median and n, pooled over both runs, for cells with n >= 10.

| distinct local experts | T=1 (decode 1) | T=8 (verify 1) | T=16 (decode 16) |
|---|---|---|---|
| 1 | 28.1 (16) | | |
| 2 | 37.2 (2097) | | |
| 3 | 50.5 (1566) | | |
| 4 | 57.2 (300) | 58.8 (55) | |
| 5 | 72.4 (19) | 75.0 (154) | |
| 6 | | 83.2 (336) | 86.6 (32) |
| 7 | | 92.6 (630) | 95.7 (92) |
| 8 | | 101.4 (825) | 103.8 (186) |
| 9 | | 114.0 (851) | 117.1 (312) |
| 10 | | 123.6 (636) | 126.6 (476) |
| 11 | | 131.3 (320) | 133.3 (605) |
| 12 | | 142.1 (124) | 143.7 (648) |
| 13 | | 151.8 (42) | 153.1 (585) |
| 14 | | | 163.3 (472) |
| 15 | | | 172.8 (304) |
| 16 | | | 182.4 (155) |
| 17 | | | 191.9 (87) |
| 18 | | | 202.8 (30) |

Per-expert slope, in us per additional distinct local expert. Each run's OLS fit has a 95% bootstrap interval over reps:

| T | k9-1 (GPU 4) | k9-2 (GPU 5) | with local routes as a 2nd regressor | adjacent-step medians (n >= 30) |
|---|---|---|---|---|
| 1 | 11.34 [11.20, 11.52] | 11.63 [11.47, 11.79] | routes = distinct at T=1 | 2->3 +13.3, 3->4 +6.8 (both runs); pooled 1->2 +9.1, 4->5 +15.2 |
| 8 | 9.89 [9.84, 9.93] | 10.12 [10.07, 10.17] | 9.89 / 10.07 per expert, -0.01 / +0.15 per route | median 9.6, range 6.8-15.4; 4->5 +15.4 (pooled +16.2); 8->9 +11.2 / +14.1 |
| 16 | 9.56 [9.51, 9.61] | 9.63 [9.58, 9.68] | 9.51 / 9.55 per expert, +0.37 / +0.39 per route | median 9.5, range 6.4-13.4; 8->9 +13.4 / +13.2; 10->11 +6.4 / +6.9 |

Reading:
- The spread between replicates (0.29 / 0.23 / 0.07 us at T=1/8/16) is wider than each run's bootstrap interval, so
  quote the replicate range: **T=8 9.9-10.1, T=16 9.56-9.63 us per expert.** At T=1 a single slope does not describe
  the data: the steps alternate 13.3 / 6.8 us. The two-step mean from d=2 to d=4 is 10.0 us per expert.
- The staircase is not uniform. The largest steps are into d=3 and d=5 at T=1 (+13.3, +15.2), into d=5 at T=8
  (+15.4 / +16.2) and into d=9 at T=8 and T=16 (+11.2 to +14.1). The smallest are into d=4 at T=1 (+6.8) and into d=11
  at T=8 and T=16 (+6.4 to +8.1). That looks like a grouping or boundary cost in the board's expert loop, and it is
  worth a look in S3's design.
- Time follows distinct experts, not routes: at most 0.4 us per local route at T=16, about 0 at T=8.
- Fit of t = a + b*k (k = distinct local experts, natural routing), with a_net = the intercept minus the 3.6-3.9 us
  replay fixed cost:

  | T | b, OLS (us per expert) | b, fit to group medians (n >= 30) | a_net, OLS | a_net, group medians |
  |---|---|---|---|---|
  | 1 | 11.34 / 11.63 | 10.00 / 10.10 (k=2..4) | 11.2 / 10.8 | 14.3 / 14.3 |
  | 8 | 9.89 / 10.12 | 10.09 / 9.74 | 19.9 / 18.0 | 17.7 / 21.4 |
  | 16 | 9.56 / 9.63 | 9.56 / 9.62 | 26.0 / 25.5 | 25.4 / 24.9 |

- Against the program's M08 rule (`research/h100-program-2026-10-27.md` §3):
  - b is above the 9.28 us "streaming engine" bar at every T (1.03-1.25x). The rule also needs that time concentrated
    in one kernel, and these runs do not profile.
  - a(T=1) straddles the 12 us "fixed-cost work" bar (10.8-14.3 us). The estimate depends on the estimator because of
    the sawtooth and the extrapolation to k=0.
  - Both questions stay unresolved here. M08's forced k in {0..8} (E1b) settles them.
- Bandwidth, as an assumption: the reference NVFP4 layout is 28.3 MB per expert (w13 16 MiB + 2 MiB scales, w2 8 MiB +
  1 MiB scales). At that size, b = 9.6-10.1 us is 2.8-2.95 TB/s, against the node's 3.11 TB/s read rate (HANDOFF §6).
  The board's packed bytes may differ; the program's 9.28 us bar implies a smaller per-expert byte count.
- Transfer: R8 (program §1) scales b by x0.69-0.97 on the judge, i.e. about 6.6-9.8 us per expert there. The judge's
  own T=1 mode gap was 10.3 us (record 2a7b5d20), against 13.3 us for d=2->3 here (x0.77). Settle any absolute number
  with a judge check.

### S1a draws: T=128/256 against routing features (22-h100-nj-moe-bigt-1/2)

| T | run | raw median (p10 / p90) | corr with distinct / routes / busiest expert | OLS fit (us) | R2 |
|---|---|---|---|---|---|
| 128 | bigt-1 (GPU 1) | 297.9 (264.1 / 330.3) | +0.75 / +0.41 / +0.01 | 20.2 + 0.399 routes + 8.61 distinct + 0.105 max | 0.84 |
| 128 | bigt-2 (GPU 3) | 299.5 (264.3 / 335.2) | +0.76 / +0.40 / +0.01 | 23.0 + 0.391 routes + 8.67 distinct + 0.069 max | 0.84 |
| 256 | bigt-1 | 349.8 (317.3 / 378.9) | +0.87 / +0.25 / -0.17 | 72.4 + 0.174 routes + 8.59 distinct - 0.023 max | 0.93 |
| 256 | bigt-2 | 348.4 (317.9 / 377.9) | +0.87 / +0.24 / -0.14 | 77.4 + 0.159 routes + 8.51 distinct - 0.015 max | 0.92 |

- **The slow draws are the wide ones, not the hot ones.** The slowest 10% of draws average 25.7 distinct local experts
  at T=128 (fastest 10%: 18.1-18.4) and 28.0 at T=256 (19.9-20.2). Their busiest expert is the same or less loaded
  (T=128: 67-71 rows in both deciles; T=256: 113-122 rows in the slow decile vs 149-169 in the fast one).
- So the worst-draw class S1a must beat is many distinct experts: T=128 with d >= 26 runs 318.9-332.6 us, and T=256 with
  d >= 28 runs 370.0-382.2 us.
- 8.5-8.7 us per distinct expert at T >= 128 corresponds to more than the node's read rate (3.3 TB/s on 28.3 MB). At
  these sizes the per-expert term is partly overlapped with other work, or the fit splits it with the route term.

### Memcheck per kernel family (24-h100-nj-moe-san-{ea4, ea_combine, ea_w, wg})

- compute-sanitizer memcheck reported **0 errors** in all four runs (`--kernel-name kns=<family>`). All four are still
  **VOID, not a pass**: the program counts an out-of-memory fallback as voiding the run (program §2, E2).
- Under memcheck, the board's prepare() filled the 80 GB H100 (79.2 GiB in use), even with the kernel filter. As a
  result:
  - The ladder, the combo phases (F1, F2, C1a, C1b), Engine A ("v3 not built") and the wgmma bucket selector raised OOM
    and fell back to master or Marlin.
  - ea4 then failed 2 of its 5 tests on OOM, and wg failed 3 of 5. Neither got a timed replay.
  - ea_combine and ea_w aborted in the harness's weight checksum after prepare, so they ran no tests.
- The filtered families therefore never ran on their judged paths. This repeats the 10-27 18:3xZ finding in the
  nodejudge README.
- Ways to make memcheck conclusive: a lighter prepare under the sanitizer (diagnostics off, fewer state copies), or
  running on the 141 GB H200 when it is available.

### Board gate flips across 4 cold prepares (30-h100-nj-moe-board-flips-1..4)

- Each run logged 25 `PATH CHOSEN` lines, and 21 are identical across all four runs. Two F2 B256 lines differ only in
  which routing was worse; their decision (align2 chunk 64) is the same.
- **One gate flipped, in 1 of 4 runs: Engine A's v1/v4 split (the remaining two lines).** Run 1 chose v1 for T=1..8 and
  v4 for T=9..16; runs 2-4 chose v1 for T=1 only and v4 for T=2..16. Program §0 records run 1's pick as a tie (1.059 vs
  1.059) and counts class 8 going v1 in 3 of 7 board prepares, warm runs included.
- Cost of the flip: verify 1 (T=8) paired x1.2010 in run 1 against x1.2132 / 1.2179 / 1.2155, about -1.2%. decode 16
  is identical (x1.1804-1.1818).
- The other rows vary 0.5-1% between runs whose decisions matched (verify 16 x1.5573-1.5824, verify 32 x1.7999-1.8157).
- Headline spread across the 4 runs: 0.26% (x1.3893-1.3929). Cold prepare took 161.5-164.4 s (judge record 2a7b5d20:
  163.8 s).

### Held-out sweep and soak (25-h100-nj-moe-sweep, 26-h100-nj-moe-soak-1/2)

- Sweep: 37/37 tests pass, i.e. the judge 5 plus 32 held-out shapes (27 bs in 2..255 and 5 skewed). No held-out shape
  in that range failed on board ab46fbb7.
- Soak: 12,000 timed replays across two seeds, every one checked against the fp32 reference on fresh inputs: 0
  miscompares, and no fault or hang. Every row's paired ratio lies within 0.5% of the four cold runs' range.

### Stage-0 ladder replicates (20-h100-esl-*, 21-h100-esl-*)

- All 8 runs are correct: every KV line has ok=1, SUMMARY bad=0, every memcheck and synccheck run reports 0 errors,
  and K2CHECK has 0 mismatches.
- Replicates on different GPUs agree closely. For the cold and k4 replicate pairs, and for k1 and k2 against their
  validation runs, the median clk_k16 difference is 0.03-0.07% and the p90 at most 0.9%. Single lines differ by up to
  3.8-5.9%, and one k2 line by 25.7%. k7 matches its validation within 0.9% median and 3.2% max (B/clk), and k6 gives
  the same verdict.
- Against the 20:2xZ "quick" validation runs, esl-cold differs more (median 0.23%, max 12.5%), and k4 differs by a 1.3%
  median over its 32 shared lines. The quick forms are noisier, and their clocks dipped to 1696-1788 MHz.
- **Void lines: esl-k4 had 30 (rep1) and 24 (rep2) of 192 lines measured at 1665-1795 MHz.** All of them are mode=dev
  device-wide streamed lines at 1.0-3.0 TB/s. The cta lines that carry the K4 targets were at 1830 MHz and stand.
- K4 against its targets, from the cta lines: the ring K-step reaches 12.5-12.6 el/clk/SM at N=128 (target 12.8, 98%)
  but only 14.2-18.4 at N=64 (target 25.6).
- K6 is a NO-GO on both toolchains, the same verdict as the validation run.

### Not part of the harvest

- `runs/20261027-142513-gpunode06` (nodejudge-e1-validate, GPU 7) started at 21:25:27Z, after the 21:22Z hold. It died at
  once with `python: command not found`, because the concurrent `clean` was removing the env. `INDEX.tsv` lists it as
  DONE rc=0, but no GPU work ran.

## 2026-10-28T0139Z: re-proving of the CURRENT moe board 498afbd8 (10-28, committed harness 0c0bc56, bundle nodejudge 01:23:22Z)

Queued by the main thread after the runner restart (runner pid 1400704 from 01:24Z), gpunode06, warm caches unless noted.

| job | verdict | headline vs SGLang on this GPU | run dir (`runs/…-gpunode06`) |
|---|---|---|---|
| 25-sweep-b498 (32 held-out shapes) | board passed=True | x1.3705 [1.3499, 1.4237] | 20261027-182831 |
| 26-soak-b498-1 / -2 (1000 reps/shape, every replay checked) | passed=True / passed=True | x1.3927 / x1.3922 | -182834 / -182836 |
| 30-board-flips-b498-1..4 (cold judge-like prepare) | all passed=True | x1.4028 / 1.3907 / 1.3932 / 1.3966 (spread 0.87%) | -182918 / -182921 / -183032 / -183114 |
| 90-fuzz-odd / -skew (fillers) | passed=True / passed=True | (fuzz shapes) | -182439 / -182442 |

- Races on the node, cold runs: `[hybrid-moe] T=128 = MASTER` (master 314.5-319.0 us vs H-16 343.6-344.1), `T=256 = HYBRID
  FORKED H=-16` (H-16 390-392 vs master 401-402). The T=128 pick is the opposite of the judge's (H-16): the known
  node / judge disagreement on that race. The full flip table and the replay-check counts are left to E4's parser.
- **Read:** the current board passes every proving job the old board passed (sweep, soaks, fuzz, cold runs). No new
  end-game correctness risk found for the final held-out rerun.

## 2026-10-28T0206Z: K9 on the CURRENT moe board 498afbd8 (S3 baseline; 2 x 2000 reps, warm, GPUs 1 / 3)

Runs `runs/20261027-184017-gpunode06` and `-184020`; fitted with the harvest scripts (`/tmp/h100-harvest/k9.py`, `k9_fit.py`).

| T | OLS slope, us per distinct local expert (run 1 / run 2) | median raw replay by distinct experts (pooled) |
|---|---|---|
| 1 | 11.72 [11.57, 11.87] / 11.56 [11.42, 11.72] | d2 37.1, d3 50.7, d4 57.4, d5 72.5 |
| 8 | 9.86 [9.82, 9.90] / 10.09 [10.05, 10.14] | d4 59.8, d5 74.4, d6 83.5, d7 92.9, d8 101.9, d9 114.6, d10 123.8 |
| 16 | 9.61 [9.55, 9.66] / 9.55 [9.49, 9.61] | d8 104.0, d9 118.0, d10 126.9, d11 133.7, d12 144.0 |

- Same staircase as the old board (the T<=8 path did not change between ab46fbb7 and 498afbd8).
- **The steps are not uniform:** at T=1 they alternate (+13.6, +6.7, +15.1 from d2 to d5); at T=8 and T=16 the
  **8 -> 9 distinct-expert step is +14.5 / +13.8 us** against a median step of 8.6 / 9.5. That points to how experts are
  partitioned over SMs or CTA groups (a quantization of the per-expert SM share), not a flat streaming rate. S3
  design input: balance the per-expert split so the makespan does not jump at odd counts and at 9+ experts.
- Each extra local route costs 0.11 (T=8) to 0.40 (T=16) us: routes barely matter; distinct experts do.

## 2026-10-28T0234Z: S3 node validation, PRE-REGISTERED (R2) before queuing

- **Bytes:** campaign/eb-s3 8dcdf0fb (the S3 builder's commit, before its two audits; Engine A v6 / v6s raced at moe
  T=1 and T=8), moe file sha256 f4050fc3f0ac (958,245 B), pinned as `problems/moe/s3.py` (a copy, not a link). Bundle
  nj-s3: harness nodejudge.py 02df40ec (takemyjob 0c0bc56), board 498afbd8e73d. The judge check later uses the
  post-audit commit; if its bytes differ, Q1 reruns on them.
- **Q1, correctness:** does s3 pass fuzz-odd (53 odd / prime bs), fuzz-skew (22 skewed bs) and the 32-shape held-out
  sweep, every replay checked? **Rule:** any failure blocks a scored run of these bytes and goes back to the S3 fix.
- **Q2, the staircase on the same instrument:** K9 x2 on s3 against the 0141Z board K9 (11.6 / 10.0 / 9.55 us per
  distinct local expert at T=1 / 8 / 16, with the +14 us 8 -> 9 step). Expected if v6 routes at T=1: a smaller T=1
  slope (the panel estimates 7.2-8.0 us on the judge). Absolute node times run higher where memory-bound (node HBM
  ~0.69x the judge's). **Rule:** if v6 routes and the T=1 slope is not below the board's, the node says the re-split
  did not help. Only the judge check applies the kill rule.
- **Q3, isolated A/B:** 2 cold moe-s3-iso runs (board vs s3) and 1 moe-null-iso (board vs board, the noise floor).
  Report the moe headline ratio, per-load ratios, the [ea6] PATH CHOSEN lines and the prepare-budget line.
  **Rule:** reported, not decisive (HBM-bound arms do not transfer node -> judge).

## 2026-10-28T0242Z: S3 node results, Q1 + Q2 (bytes f4050fc3 = eb-s3 8dcdf0fb, pre-audit; bundle nj-s3; warm)

- **Q1 correctness: PASS.** fuzz-odd (runs/20261027-193732), fuzz-skew (-193735) and the 32-shape sweep (-193745): s3
  passed=True. The v6 canary was clean (36 inputs, NaN-poisoned h/y; v6 T1..8 worst 0.00396 vs master 0.00843; 18
  bitwise repeats). Copy checks at T=1 / T=8: PASS.
- **Routing on the node** (`[ea6]` lines): T=1 v6 vs v1 weighted 1.045-1.057 in every run, but the per-n floor at
  n=4 is marginal (+0.007, +0.005, **-0.002**): in the fuzz-odd run v6 missed its n=4 floor and T=1 stayed on v1. T=8
  v6 vs v4/v1 1.037-1.051: routed in every run. v6s (no co-resident K2) lost, 0.964-0.968, so the co-resident K2
  earns the gain (panel decision (c): keep it for the decode port).
- **Q2, K9 on s3** (runs -193737 / -193743; 2 x 2000 reps; v6 routed at T=1 and T=8 in both):

  | T | board 0141Z: OLS us/expert; medians | s3: OLS us/expert; medians |
  |---|---|---|
  | 1 | 11.64; d2 37.1, d3 50.7, d4 57.4, d5 72.5 | 11.84; d2 35.3, d3 47.9, d4 58.3, d5 68.4 |
  | 8 | 9.98; d4 59.8, d8 101.9, d9 114.6 | 9.74; d4 59.0, d8 98.0, d9 109.7 |
  | 16 | 9.58 | 9.70 (v4c, outside S3) |

  - At T=1, v6 removes the alternation (steps now 12.6 / 10.4 / 10.1) and the fixed cost (intercept net 8.1-8.5 vs
    10.85 us), saving 1.8-2.8 us at n=2-3 and 4.1 us at n=5. It does **not** lower the per-expert slope on the
    node, and it is flat at n=4 (58.3 vs 57.4).
  - **Pre-registered Q2 rule: the re-split did not cut the per-expert slope on the node.** The judge check
    decides the panel's kill rule (2) (slope >= 9.0 us on the judge).
  - K9-config headline x1.3310 / 1.3384 vs the board's x1.2950 / 1.3004 (+2.8%).
- **Prepare:** warm prepare 120-141 s vs the board's ~34 s (the v6 build is 32 s plus its canary, race and copy check).
  Cold behaviour and deadline effects are read from the Q3 cold isolated runs.

## 2026-10-28T0328Z: the moe-track program, PRE-REGISTERED (R2) and queued

- Harness: takemyjob 3d1dfd2 (merge of hopper/nj-moe 5f53c70: E1 + E1b, E3, E4); bundle nodejudge, nodejudge.py 8866e311; moe board.py PINNED to 498afbd8 (a copy, not the master link: master moved to the S3 bytes b464bd4b at 2abc122e, which route v6 at T=8 on the node).
- **Queued** (the user approved, ~03:25Z): m00 r1/r2, m456 c8/c16/c32/c128 r1/r2, m08t8 r1/r2, m1 b0..b6, p3. m456 and m08t8 wait for queue/gates/m00-accepted (written by hand after reading M00).
- **Held, not queued:** m07 and m08t1 (T=1 is now Engine A v6 on the board, so v1-at-T=1 arms are superseded), m03c (X0b: fork-knob arms vs master flip sign node -> judge; needs a judge carrier).

The staged registrations follow verbatim (queue-staged/registrations.md):


### m00 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M00 [10 GPU-min, needs E1+E3, judge: no]: is the forced-arm A/B exact and sensitive? A = as raced (the base), B = every timed-shape flag re-set to its raced value, C = serial H-16 at T=256, D = Engine A v4 at T=128; same/same2 = identity arms (two, to tell a base-position bias from arm noise)
- **Expected effect:** B (raced) vs A: 12/12 row points (6 shapes x 2 replicates) within +-0.3% and >= 11/12 CIs covering 1; the same, separately, for same vs A, for same2 vs A, and for same2 vs same (abstat.py inproc --base same --arm same2); C verify 32 in x0.88-0.96 (races predict x0.91-0.94; GPU 7 integration run x0.922); D verify 16 below 1 (~-1.5%; GPU 7 x0.973); identical kernel lists for B/same/same2 vs A. Integration history: before 10-28 02:05Z the base was the only party captured without a prior eager call and an identity arm read x1.0102 RESOLVED at verify 1 (run 20261027-185629); with the base given the same eager call the identity rows read x0.998-1.000, headline x0.9993 (run 20261027-190516); same2 checks that the fix holds at 300 reps
- **Decision rule:** accept E1 (the in-process instrument) only if all hold, every party ran with 0 check failures and no R1 void, and both replicates exited rc 0; then (and only then) the orchestrator writes queue/gates/m00-accepted, which releases 31-34; the B/identity spread sets the in-process MDE (R6); a resolved identity row voids in-process arm readings on that row until explained
- **Metric:** R3a per row + headline vs the base; also same2 vs same (abstat.py inproc --base same --arm same2); MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 300, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm same2** (of board):  -- second identity arm
- **Arm raced** (of board): post `{"ea.on.1": {"$expr": "base['ea']['on'][1]"}, "ea.on.8": {"$expr": "base['ea']['on'][8]"}, "ea.on.16": {"$expr": "base['ea']['on'][16]"}, "ea.on.32": {"$expr": "base['ea']['on'][32]"}, "ea.on.128": {"$expr": "base['ea']['on'][128]"}, "ea.on.256": {"$expr": "base['ea']['on'][256]"}, "hm.on.128": {"$expr": "base['hm']['on'][128]"}, "hm.on.256": {"$expr": "base['hm']['on'][256]"}, "hm.fon.128": {"$expr": "base['hm']['fon'][128]"}, "hm.fon.256": {"$expr": "base['hm']['fon'][256]"}, "hm.cold": {"$expr": "base['hm']['cold']"}, "hm.cold_bits.128": {"$expr": "base['hm']['cold_bits'][128]"}, "hm.cold_bits.256": {"$expr": "base['hm']['cold_bits'][256]"}, "wg_on_128": {"$expr": "base['wg_on_128']"}, "wg_on_256": {"$expr": "base['wg_on_256']"}, "wg_pdl": {"$expr": "base['wg_pdl']"}, "w2_mode": {"$expr": "base['w2_mode']"}, "w13_mode": {"$expr": "base['w13_mode']"}, "d32_align": {"$expr": "base['d32_align']"}, "d32_post_norm": {"$expr": "base['d32_post_norm']"}, "ea.g2_t": {"$expr": "dict(base['ea']['g2_t'])"}, "ea.cmb_t": {"$expr": "dict(base['ea']['cmb_t'])"}}` -- M00-B: every timed-shape flag set to its raced value
- **Arm h256serial** (of board): post `{"hm.on.256": -16, "hm.fon.256": false}` -- M00-C: serial H-16 at T=256
- **Arm v4t128** (of board): post `{"hm.on.128": 0, "hm.fon.128": false, "ea.on.128": "v4"}` -- M00-D: Engine A v4 at T=128
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m00-r1.json`, seed 4545804740698577694, job `queue/pending/30-h100-m00-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m00-r2.json`, seed 3568204243879638241, job `queue/pending/30-h100-m00-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m07 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M07 [5 GPU-min, needs E1, judge: J0 carrier]: does K1's w2 prefetch cut decode 1? pf_t[1] is a max-active-experts threshold (prefetch only when n_act <= pf_t[1]), so at k in {2, 3} the program's pf_t {2,3,4,8} are two behaviours: pf2 (prefetch at k=2 only) and pf3 (= pf4 = pf8, prefetch at both). Controls pdl=0 and v4; split by 2 vs 3 distinct local experts (E1b k_local [2, 3])
- **Expected effect:** trade-off arm (R8: changes prefetch / L2 use): pf arms within +-2% of v1; a gain, if any, larger at k=2
- **Decision rule:** +2% or more in both replicates: send to the J0 carrier, pin only at +1.5% or more judge-side; every pf arm at +0.5% or less: close prefetch for moe and decode
- **Metric:** R3a decode 1 row per arm vs v1 (abstat.py inproc --base v1) and vs the base, strata k = 2 / 3; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 400, shapes decode 1; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm v1** (of board): post `{"ea.on.1": "v1"}` -- v1 pinned (the raced class-1 pick, 23/23 node)
- **Arm pf2** (of board): post `{"ea.on.1": "v1", "new:ea.pf_t": {"$expr": "{1: 2}"}}` -- v1 + K1 w2 prefetch when n_act <= 2 (pf_t[1] = 2 is a max-active threshold): on at k=2, OFF at k=3 (its k=3 stratum is a pre-registered null vs v1)
- **Arm pf3** (of board): post `{"ea.on.1": "v1", "new:ea.pf_t": {"$expr": "{1: 3}"}}` -- v1 + K1 w2 prefetch when n_act <= 3: on at k=2 and k=3 (stands for pf3/pf4/pf8, identical at k<=3)
- **Arm pdl0** (of board): post `{"ea.on.1": "v1", "ea.pdl": 0}` -- control: v1 with K2's PDL off
- **Arm v4** (of board): post `{"ea.on.1": "v4"}` -- control: Engine A v4 at T=1
- **Routing control decode 1:** {"k_local": [2, 3], "method": "construct"}
- **Pre-registered:** pf2's k=3 stratum is a null (prefetch off there): it must read within the MDE of v1, else the arm reading is void until explained; pf2's effect is read on its k=2 stratum, pf3's on both
- **Pre-registered:** pf4 and pf8 are not run: at k <= 3 they are the same kernel arguments' behaviour as pf3 (threshold met)
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m07-r1.json`, seed 4079379381259440351, job `queue/pending/31-h100-m07-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m07-r2.json`, seed 2852634887404032162, job `queue/pending/31-h100-m07-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m456-c8 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M456 [16 GPU-min for all four classes, needs E1, judge: carrier for arms above 1%]: which engine, cw, K2 grid and combine win the Engine A class of T=8 (verify 1)? every arm forces the whole class
- **Expected effect:** same-bytes arms (R8: the sign transfers): spread within ~+-1.5% of the raced pick; X0a/X0b: v4 >= v1 and v4 ~ v4c on the node at T=8 (the judge favours v4 by ~5% at T=8), v4 >= v4c at T=16 by ~0.7%
- **Decision rule:** pin a class's best arm at +0.5% or more in both replicates; ties: R10 (pin v4 at classes 8 and 16); change K2 or the combine only at +0.8% or more; arms above 1% go to the carrier
- **Metric:** R3a verify 1 row per arm vs the base and arm vs arm (abstat.py inproc); MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 300, shapes verify 1; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm c8v1** (of board): post `{"ea.on.2..8": "v1"}` -- T=8 class: v1
- **Arm c8v4** (of board): post `{"ea.on.2..8": "v4"}` -- T=8 class: v4
- **Arm c8v4c7** (of board): post `{"ea.on.2..8": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=8 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112)
- **Arm c8v4c9** (of board): post `{"ea.on.2..8": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 144}` -- T=8 class: v4c cw9 (cap EA_PDL_BITS.v4c = 144)
- **Arm c8v4k2** (of board): post `{"ea.on.2..8": "v4", "new:ea.g2_t.8": {"$expr": "base['ea']['grid'] // EA_CTAS_PER_SM"}}` -- T=8 class: v4, K2 on one CTA per SM
- **Arm c8v4tri** (of board): post `{"ea.on.2..8": "v4", "new:ea.cmb_t.8": 0}` -- T=8 class: v4, Triton combine
- **Pre-registered:** the base's class-8 engine is raced at prepare and can differ between replicates (class 8 flips v1/v4 in ~40% of prepares): the decision is arm vs arm (every arm forces the class), and the base's PATH CHOSEN / FLAGS line is stated per replicate
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m456-c8-r1.json`, seed 2032779984247594520, job `queue/pending/32-h100-m456-c8-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m456-c8-r2.json`, seed 934202439948931335, job `queue/pending/32-h100-m456-c8-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m456-c16 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M456 [16 GPU-min for all four classes, needs E1, judge: carrier for arms above 1%]: which engine, cw, K2 grid and combine win the Engine A class of T=16 (decode 16)? every arm forces the whole class
- **Expected effect:** same-bytes arms (R8: the sign transfers): spread within ~+-1.5% of the raced pick; X0a/X0b: v4 >= v1 and v4 ~ v4c on the node at T=8 (the judge favours v4 by ~5% at T=8), v4 >= v4c at T=16 by ~0.7%
- **Decision rule:** pin a class's best arm at +0.5% or more in both replicates; ties: R10 (pin v4 at classes 8 and 16); change K2 or the combine only at +0.8% or more; arms above 1% go to the carrier
- **Metric:** R3a decode 16 row per arm vs the base and arm vs arm (abstat.py inproc); MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 300, shapes decode 16; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm c16v4** (of board): post `{"ea.on.9..16": "v4"}` -- T=16 class: v4
- **Arm c16v4c7** (of board): post `{"ea.on.9..16": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=16 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112)
- **Arm c16v4c8** (of board): post `{"ea.on.9..16": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 128}` -- T=16 class: v4c cw8 (cap EA_PDL_BITS.v4c = 128)
- **Arm c16v4c9** (of board): post `{"ea.on.9..16": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 144}` -- T=16 class: v4c cw9 (cap EA_PDL_BITS.v4c = 144)
- **Arm c16v4c11** (of board): post `{"ea.on.9..16": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 176}` -- T=16 class: v4c cw11 (cap EA_PDL_BITS.v4c = 176)
- **Arm c16v3** (of board): post `{"ea.on.9..16": "v3"}` -- T=16 class: v3
- **Arm c16v4k2** (of board): post `{"ea.on.9..16": "v4", "new:ea.g2_t.16": {"$expr": "base['ea']['grid'] // EA_CTAS_PER_SM"}}` -- T=16 class: v4, K2 on one CTA per SM
- **Pre-registered:** the base's class-16 engine is raced at prepare and can differ between replicates (class 8 flips v1/v4 in ~40% of prepares): the decision is arm vs arm (every arm forces the class), and the base's PATH CHOSEN / FLAGS line is stated per replicate
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m456-c16-r1.json`, seed 1394946176265187079, job `queue/pending/32-h100-m456-c16-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m456-c16-r2.json`, seed 3920564090368528973, job `queue/pending/32-h100-m456-c16-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m456-c32 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M456 [16 GPU-min for all four classes, needs E1, judge: carrier for arms above 1%]: which engine, cw, K2 grid and combine win the Engine A class of T=32 (decode 32)? every arm forces the whole class
- **Expected effect:** same-bytes arms (R8: the sign transfers): spread within ~+-1.5% of the raced pick; X0a/X0b: v4 >= v1 and v4 ~ v4c on the node at T=8 (the judge favours v4 by ~5% at T=8), v4 >= v4c at T=16 by ~0.7%
- **Decision rule:** pin a class's best arm at +0.5% or more in both replicates; ties: R10 (pin v4 at classes 8 and 16); change K2 or the combine only at +0.8% or more; arms above 1% go to the carrier
- **Metric:** R3a decode 32 row per arm vs the base and arm vs arm (abstat.py inproc); MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 300, shapes decode 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm c32v4c6** (of board): post `{"ea.on.17..32": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 96}` -- T=32 class: v4c cw6 (cap EA_PDL_BITS.v4c = 96)
- **Arm c32v4c7** (of board): post `{"ea.on.17..32": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=32 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112)
- **Arm c32v4c9** (of board): post `{"ea.on.17..32": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 144}` -- T=32 class: v4c cw9 (cap EA_PDL_BITS.v4c = 144)
- **Arm c32v4c11** (of board): post `{"ea.on.17..32": "v4c"}`; cap `{"EA_PDL_BITS.v4c": 176}` -- T=32 class: v4c cw11 (cap EA_PDL_BITS.v4c = 176)
- **Arm c32v4** (of board): post `{"ea.on.17..32": "v4"}` -- T=32 class: v4
- **Arm c32v4ck2** (of board): post `{"ea.on.17..32": "v4c", "new:ea.g2_t.32": {"$expr": "base['ea']['grid'] // EA_CTAS_PER_SM"}}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=32 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112), K2 on one CTA per SM
- **Arm c32v4ctri** (of board): post `{"ea.on.17..32": "v4c", "new:ea.cmb_t.32": 0}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=32 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112), Triton combine
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m456-c32-r1.json`, seed 3439729908768725824, job `queue/pending/32-h100-m456-c32-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m456-c32-r2.json`, seed 2956738619624274360, job `queue/pending/32-h100-m456-c32-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m456-c128 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M456 [16 GPU-min for all four classes, needs E1, judge: carrier for arms above 1%]: which engine, cw, K2 grid and combine win the Engine A class of T=128 (verify 16)? every arm forces the whole class
- **Expected effect:** same-bytes arms (R8: the sign transfers): spread within ~+-1.5% of the raced pick; X0a/X0b: v4 >= v1 and v4 ~ v4c on the node at T=8 (the judge favours v4 by ~5% at T=8), v4 >= v4c at T=16 by ~0.7%
- **Decision rule:** pin a class's best arm at +0.5% or more in both replicates; ties: R10 (pin v4 at classes 8 and 16); change K2 or the combine only at +0.8% or more; arms above 1% go to the carrier
- **Metric:** R3a verify 16 row per arm vs the base and arm vs arm (abstat.py inproc); MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 300, shapes verify 16; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm c128v4c7** (of board): post `{"ea.on.128": "v4c", "hm.on.128": 0, "hm.fon.128": false}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=128 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112)
- **Arm c128v4c9** (of board): post `{"ea.on.128": "v4c", "hm.on.128": 0, "hm.fon.128": false}`; cap `{"EA_PDL_BITS.v4c": 144}` -- T=128 class: v4c cw9 (cap EA_PDL_BITS.v4c = 144)
- **Arm c128v4c11** (of board): post `{"ea.on.128": "v4c", "hm.on.128": 0, "hm.fon.128": false}`; cap `{"EA_PDL_BITS.v4c": 176}` -- T=128 class: v4c cw11 (cap EA_PDL_BITS.v4c = 176)
- **Arm c128v4** (of board): post `{"ea.on.128": "v4", "hm.on.128": 0, "hm.fon.128": false}` -- T=128 class: v4
- **Arm c128v4ck2** (of board): post `{"ea.on.128": "v4c", "hm.on.128": 0, "hm.fon.128": false, "new:ea.g2_t.128": {"$expr": "base['ea']['grid'] // EA_CTAS_PER_SM"}}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=128 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112), K2 on one CTA per SM
- **Arm c128v4ccuda** (of board): post `{"ea.on.128": "v4c", "hm.on.128": 0, "hm.fon.128": false, "new:ea.cmb_t.128": 1}`; cap `{"EA_PDL_BITS.v4c": 112}` -- T=128 class: v4c cw7 (cap EA_PDL_BITS.v4c = 112), CUDA combine (cmb_t=1)
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m456-c128-r1.json`, seed 2947383060488994142, job `queue/pending/32-h100-m456-c128-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m456-c128-r2.json`, seed 2391619555393727555, job `queue/pending/32-h100-m456-c128-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m03c (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M03c [10 GPU-min, needs E1+E3, judge: carrier]: which cold-half knob shortens the T=256 fork? H-16f with cold_bits[256] = cw<<4 for cw 7/9/11, cold v3, WG PDL off; controls H-16 serial and H-8f; nsys pass gives each half's time and the overlap
- **Expected effect:** trade-off arms (R8: the fork's SM partition / bytes): cw arms within +-1.5% of H-16f; E3 (moe-prof) shows the cold half holding the SMs for 163 of 343 us of the hot path, so cold-half knobs can move the row; X0b: the node overstates the fork's gain (x1.084 node vs x1.036 judge)
- **Decision rule:** pin at +0.8% or more in both replicates, and only if the cold half is on the critical path (E3 diff); a judge carrier reading before any pin (trade-off arm)
- **Metric:** R3a verify 32 row per arm vs the base (H-16f) + E3 critical path per arm (nj_profile.py diff); MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 300, shapes verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm h16fcw7** (of board): post `{"hm.on.256": -16, "hm.fon.256": true, "hm.cold_bits.256": 112}` -- H-16f, cold half cost-weighted cw7 (cold_bits[256] = 112)
- **Arm h16fcw9** (of board): post `{"hm.on.256": -16, "hm.fon.256": true, "hm.cold_bits.256": 144}` -- H-16f, cold half cost-weighted cw9 (cold_bits[256] = 144)
- **Arm h16fcw11** (of board): post `{"hm.on.256": -16, "hm.fon.256": true, "hm.cold_bits.256": 176}` -- H-16f, cold half cost-weighted cw11 (cold_bits[256] = 176)
- **Arm h16fv3** (of board): post `{"hm.on.256": -16, "hm.fon.256": true, "hm.cold": "v3"}` -- H-16f, cold half on Engine A v3
- **Arm h16fwgpdl0** (of board): post `{"hm.on.256": -16, "hm.fon.256": true, "wg_pdl": false}` -- H-16f, WG PDL off
- **Arm h16** (of board): post `{"hm.on.256": -16, "hm.fon.256": false}` -- control: serial H-16
- **Arm h8f** (of board): post `{"hm.on.256": -8, "hm.fon.256": true}` -- control: H-8f
- **Pre-registered:** the base runs hm.cold_bits.256 = 0 (plain v4 cold split, cw0) at T=256 in every non-ncu board run (only ncu prepares read 112): h16fcw7/9/11 each CHANGE it (0 -> 112/144/176); none equals the board
- **Pre-registered:** an E3 PROFILE FIDELITY FAIL (tol 3%) on a party marks that party's critical path unusable; the 'cold half on the critical path' clause is then read from the parties that pass, and no pin rests on a failing party's split
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m03c-r1.json`, seed 2340456597447584653, job `queue/pending/33-h100-m03c-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m03c-r2.json`, seed 2884765605473563054, job `queue/pending/33-h100-m03c-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m08t1 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M08+P2 T=1 [26 GPU-min with m08t8, needs E1b+E3, judge: no]: is the T=1 expert cost fixed (a) or per expert (b)? k in {0,1,2,3,4,6,8} local experts, ~200 reps per k; arms v1, v1+pf8 (prefetch at every k), v4, v3; nsys pass over every k (K1, K2, combine, gaps)
- **Expected effect:** E1 GPU 7 reading: board T=1 fit t = 11.3 + 10.81 k us (integration run k 1..4: 14.6 + 9.97 k); a should transfer ~1:1 and b at x0.69-0.97 (judge 10.3 us per expert)
- **Decision rule:** a(T=1) >= 12 us: S3 becomes fixed-cost work (fusion or a persistent T<=8 kernel, >= 1% headline); b >= 9.28 us per expert (1.3x the ring rate) concentrated in one kernel: S3 becomes a streaming engine; the arm with the smallest b is the class-1 pin candidate
- **Metric:** BY_K fit t = a + b*k per party; R3a per k stratum; E3 kernel split per k; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 1400, shapes decode 1; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm t1v1** (of board): post `{"ea.on.1": "v1"}`
- **Arm t1v1pf8** (of board): post `{"ea.on.1": "v1", "new:ea.pf_t": {"$expr": "{1: 8}"}}`
- **Arm t1v4** (of board): post `{"ea.on.1": "v4"}`
- **Arm t1v3** (of board): post `{"ea.on.1": "v3"}`
- **Routing control decode 1:** {"k_local": [0, 1, 2, 3, 4, 6, 8], "method": "construct"}
- **Pre-registered:** BY_K fit t = a + b*k is reported twice, on all k and on k >= 1 (k=0 has no local route: never drawn by the judge, all-zero reference, high leverage on a); the decision reads the k >= 1 fit
- **Pre-registered:** t1v1pf8 sets pf_t[1] = 8 (a max-active threshold), so its prefetch is on at every k in the range and its fit spans one regime (audit 10-28: pf_t[1] = 4 would switch it off at k = 6, 8)
- **Pre-registered:** E3 fidelity: T=1 rows are ~11-100 us and nsys adds 1.5-5% on short rows. A PROFILE FIDELITY FAIL (tol 3%) marks that party x k kernel split unusable; the 'b concentrated in one kernel' clause is read only from k strata whose split passes, and with fewer than 3 passing k >= 1 strata it is undecided and the decision rests on BY_K (a, b) alone
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m08t1-r1.json`, seed 24378362869959566, job `queue/pending/34-h100-m08t1-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m08t1-r2.json`, seed 3000658969699471195, job `queue/pending/34-h100-m08t1-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m08t8 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M08+P2 T=8 [26 GPU-min with m08t1, needs E1b+E3, judge: no]: is the T=8 expert cost fixed or per expert? k in {0,2,...,14} local experts, 13 local routes for k 2..12 (the harness clamps L to [k, T*min(topk, k)]: L = 0 at k=0 and L = 14 at k=14), ~200 reps per k; arms v1, v4, v4c (whole class 2..8); nsys pass over every k
- **Expected effect:** E1 GPU 7 reading: board T=8 fit t = 15.9 + 10.21 k us at 13 routes; b at x0.69-0.97 on the judge
- **Decision rule:** as m08t1: a >= 12 us -> fixed-cost work; b >= 9.28 us per expert in one kernel -> streaming engine; the smallest-b arm is the class-8 pin candidate (with M456)
- **Metric:** BY_K fit t = a + b*k per party; R3a per k stratum; E3 kernel split per k; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 1600, shapes verify 1; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm same** (of board):  -- in-process null (R5): identity arm, no overrides
- **Arm c8v1** (of board): post `{"ea.on.2..8": "v1"}`
- **Arm c8v4** (of board): post `{"ea.on.2..8": "v4"}`
- **Arm c8v4c** (of board): post `{"ea.on.2..8": "v4c"}`
- **Routing control verify 1:** {"k_local": [0, 2, 4, 6, 8, 10, 12, 14], "local_routes": 13, "method": "construct"}
- **Pre-registered:** the decision reads the BY_K fit on k 2..12 (fixed L = 13); k=0 (L=0, never drawn by the judge, all-zero reference) and k=14 (L=14) are reported apart, and the all-k fit is shown for reference
- **Pre-registered:** E3 fidelity: as m08t1 (a FAIL marks that party x k split unusable; the one-kernel clause needs >= 3 passing strata in k 2..12, else the decision rests on BY_K alone)
- **Pre-registered:** the in-process MDE per row is at least M00's identity spread on that row (max of |same-1|, |same2-1|, |same2/same-1|): an arm reading inside it is unresolved whatever its interval
- **Pre-registered:** every arm is read against the identity arm `same` (captured like every arm) as well as against the base: a pin or a close needs both readings to agree in sign and clear the MDE (GPU 7 smoke 20261027-194606: `same` read x1.0072 RESOLVED vs the base at verify 1 on 16 pairs, so a base-capture offset is not excluded)
- **Pre-registered:** R11/R1: a row with any check failure on the arm or its base, or a shape the clock monitor voided, is VOID (arms.txt / abstat inproc print VOID); a headline with a void row is VOID
- **Pre-registered:** R4: the runner cannot pin a GPU (gpus=1 takes any free pool GPU); each replicate's GPU is read from its CUDA_VISIBLE_DEVICES line, and a replicate that landed on its sibling's GPU is re-run (new job, same config)
- **Replicate 1:** config `configs/m08t8-r1.json`, seed 4499152488654165744, job `queue/pending/34-h100-m08t8-r1.job` (staged in queue-staged/)
- **Replicate 2:** config `configs/m08t8-r2.json`, seed 4058921836810187547, job `queue/pending/34-h100-m08t8-r2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b0 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 warm-up (not a block, never analysed): builds board, copy and SGLang into the env's warm caches so b1-b3 are WARM for both processes
- **Expected effect:** both processes pass; the copy's prepare may be cold (~110 s)
- **Decision rule:** none (prerequisite of b1)
- **Metric:** none; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches WARM, reps 4, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Replicate 1:** config `configs/m1-b0.json`, seed 202610281000, job `queue/pending/35-h100-m1-b0.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b1 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 block 1/6 [25 GPU-min for 6 blocks, needs none, judge: no]: how noisy are isolated runs and how often do races flip? board vs a byte-identical copy, one judge-shaped process each (SGLang + it), order AB, caches WARM
- **Expected effect:** null: block log-ratio ~0; with E4's X0a, class 8 flips ~36-43% of prepares, class 16 ~7-9%, the rest 0
- **Decision rule:** the 6 blocks set the isolated MDE (R6); an order or GPU effect above 0.2% makes AB/BA + GPU balancing mandatory; flip rates reported with the 7 earlier runs and X0a's 23
- **Metric:** R3b mean block log-ratio, t-interval, over the 6 m1 blocks (abstat.py iso --blocks-from ... --metric paired, and --metric raw: each process's raw replay median); PATH CHOSEN per process; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches WARM, reps 100, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Pre-registered:** the block table records each process's prepare_s and PATH CHOSEN; a WARM block in which either process prepared for more than 60 s is not warm and is reported apart, not pooled with b1-b3
- **Pre-registered:** blocks run strictly in sequence b0 -> b1 -> ... -> b6 (requires= on the previous block's done file), so no two prepares overlap; b0 only warms the caches
- **Pre-registered:** deviation: the program's GPU sequence 1,3,4,5,6,1 cannot be pinned through the runner (gpus=1 takes any pool GPU); each block's GPU is read from its CUDA_VISIBLE_DEVICES line
- **Replicate 1:** config `configs/m1-b1.json`, seed 202610281001, job `queue/pending/35-h100-m1-b1.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b2 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 block 2/6 [25 GPU-min for 6 blocks, needs none, judge: no]: how noisy are isolated runs and how often do races flip? board vs a byte-identical copy, one judge-shaped process each (SGLang + it), order BA, caches WARM
- **Expected effect:** null: block log-ratio ~0; with E4's X0a, class 8 flips ~36-43% of prepares, class 16 ~7-9%, the rest 0
- **Decision rule:** the 6 blocks set the isolated MDE (R6); an order or GPU effect above 0.2% makes AB/BA + GPU balancing mandatory; flip rates reported with the 7 earlier runs and X0a's 23
- **Metric:** R3b mean block log-ratio, t-interval, over the 6 m1 blocks (abstat.py iso --blocks-from ... --metric paired, and --metric raw: each process's raw replay median); PATH CHOSEN per process; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches WARM, reps 100, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Pre-registered:** the block table records each process's prepare_s and PATH CHOSEN; a WARM block in which either process prepared for more than 60 s is not warm and is reported apart, not pooled with b1-b3
- **Pre-registered:** blocks run strictly in sequence b0 -> b1 -> ... -> b6 (requires= on the previous block's done file), so no two prepares overlap; b0 only warms the caches
- **Pre-registered:** deviation: the program's GPU sequence 1,3,4,5,6,1 cannot be pinned through the runner (gpus=1 takes any pool GPU); each block's GPU is read from its CUDA_VISIBLE_DEVICES line
- **Replicate 1:** config `configs/m1-b2.json`, seed 202610281002, job `queue/pending/35-h100-m1-b2.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b3 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 block 3/6 [25 GPU-min for 6 blocks, needs none, judge: no]: how noisy are isolated runs and how often do races flip? board vs a byte-identical copy, one judge-shaped process each (SGLang + it), order AB, caches WARM
- **Expected effect:** null: block log-ratio ~0; with E4's X0a, class 8 flips ~36-43% of prepares, class 16 ~7-9%, the rest 0
- **Decision rule:** the 6 blocks set the isolated MDE (R6); an order or GPU effect above 0.2% makes AB/BA + GPU balancing mandatory; flip rates reported with the 7 earlier runs and X0a's 23
- **Metric:** R3b mean block log-ratio, t-interval, over the 6 m1 blocks (abstat.py iso --blocks-from ... --metric paired, and --metric raw: each process's raw replay median); PATH CHOSEN per process; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches WARM, reps 100, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Pre-registered:** the block table records each process's prepare_s and PATH CHOSEN; a WARM block in which either process prepared for more than 60 s is not warm and is reported apart, not pooled with b1-b3
- **Pre-registered:** blocks run strictly in sequence b0 -> b1 -> ... -> b6 (requires= on the previous block's done file), so no two prepares overlap; b0 only warms the caches
- **Pre-registered:** deviation: the program's GPU sequence 1,3,4,5,6,1 cannot be pinned through the runner (gpus=1 takes any pool GPU); each block's GPU is read from its CUDA_VISIBLE_DEVICES line
- **Replicate 1:** config `configs/m1-b3.json`, seed 202610281003, job `queue/pending/35-h100-m1-b3.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b4 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 block 4/6 [25 GPU-min for 6 blocks, needs none, judge: no]: how noisy are isolated runs and how often do races flip? board vs a byte-identical copy, one judge-shaped process each (SGLang + it), order BA, caches COLD
- **Expected effect:** null: block log-ratio ~0; with E4's X0a, class 8 flips ~36-43% of prepares, class 16 ~7-9%, the rest 0
- **Decision rule:** the 6 blocks set the isolated MDE (R6); an order or GPU effect above 0.2% makes AB/BA + GPU balancing mandatory; flip rates reported with the 7 earlier runs and X0a's 23
- **Metric:** R3b mean block log-ratio, t-interval, over the 6 m1 blocks (abstat.py iso --blocks-from ... --metric paired, and --metric raw: each process's raw replay median); PATH CHOSEN per process; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches COLD (fresh_build), reps 100, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Pre-registered:** the block table records each process's prepare_s and PATH CHOSEN; a WARM block in which either process prepared for more than 60 s is not warm and is reported apart, not pooled with b1-b3
- **Pre-registered:** blocks run strictly in sequence b0 -> b1 -> ... -> b6 (requires= on the previous block's done file), so no two prepares overlap; b0 only warms the caches
- **Pre-registered:** deviation: the program's GPU sequence 1,3,4,5,6,1 cannot be pinned through the runner (gpus=1 takes any pool GPU); each block's GPU is read from its CUDA_VISIBLE_DEVICES line
- **Replicate 1:** config `configs/m1-b4.json`, seed 202610281004, job `queue/pending/35-h100-m1-b4.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b5 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 block 5/6 [25 GPU-min for 6 blocks, needs none, judge: no]: how noisy are isolated runs and how often do races flip? board vs a byte-identical copy, one judge-shaped process each (SGLang + it), order AB, caches COLD
- **Expected effect:** null: block log-ratio ~0; with E4's X0a, class 8 flips ~36-43% of prepares, class 16 ~7-9%, the rest 0
- **Decision rule:** the 6 blocks set the isolated MDE (R6); an order or GPU effect above 0.2% makes AB/BA + GPU balancing mandatory; flip rates reported with the 7 earlier runs and X0a's 23
- **Metric:** R3b mean block log-ratio, t-interval, over the 6 m1 blocks (abstat.py iso --blocks-from ... --metric paired, and --metric raw: each process's raw replay median); PATH CHOSEN per process; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches COLD (fresh_build), reps 100, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Pre-registered:** the block table records each process's prepare_s and PATH CHOSEN; a WARM block in which either process prepared for more than 60 s is not warm and is reported apart, not pooled with b1-b3
- **Pre-registered:** blocks run strictly in sequence b0 -> b1 -> ... -> b6 (requires= on the previous block's done file), so no two prepares overlap; b0 only warms the caches
- **Pre-registered:** deviation: the program's GPU sequence 1,3,4,5,6,1 cannot be pinned through the runner (gpus=1 takes any pool GPU); each block's GPU is read from its CUDA_VISIBLE_DEVICES line
- **Replicate 1:** config `configs/m1-b5.json`, seed 202610281005, job `queue/pending/35-h100-m1-b5.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### m1-b6 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** M1 block 6/6 [25 GPU-min for 6 blocks, needs none, judge: no]: how noisy are isolated runs and how often do races flip? board vs a byte-identical copy, one judge-shaped process each (SGLang + it), order BA, caches COLD
- **Expected effect:** null: block log-ratio ~0; with E4's X0a, class 8 flips ~36-43% of prepares, class 16 ~7-9%, the rest 0
- **Decision rule:** the 6 blocks set the isolated MDE (R6); an order or GPU effect above 0.2% makes AB/BA + GPU balancing mandatory; flip rates reported with the 7 earlier runs and X0a's 23
- **Metric:** R3b mean block log-ratio, t-interval, over the 6 m1 blocks (abstat.py iso --blocks-from ... --metric paired, and --metric raw: each process's raw replay median); PATH CHOSEN per process; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, isolated processes, caches COLD (fresh_build), reps 100, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes copy:** `snap/copy-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Bytes sglang:** `baseline.py` sha256 `4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1`
- **Pre-registered:** the block table records each process's prepare_s and PATH CHOSEN; a WARM block in which either process prepared for more than 60 s is not warm and is reported apart, not pooled with b1-b3
- **Pre-registered:** blocks run strictly in sequence b0 -> b1 -> ... -> b6 (requires= on the previous block's done file), so no two prepares overlap; b0 only warms the caches
- **Pre-registered:** deviation: the program's GPU sequence 1,3,4,5,6,1 cannot be pinned through the runner (gpus=1 takes any pool GPU); each block's GPU is read from its CUDA_VISIBLE_DEVICES line
- **Replicate 1:** config `configs/m1-b6.json`, seed 202610281006, job `queue/pending/35-h100-m1-b6.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


### p3 (registered 2026-10-28T02:57Z; status: STAGED, not queued)

- **Question:** P3 [40 GPU-min, needs P1 PASS (runs/20261027-182246-gpunode06), judge: no]: what limits each production kernel (Engine A, hot, align, combine, post-norm)? ncu SOL, memory, occupancy, launch, warp-state and scheduler sections, --clock-control none, on an arm PINNED to the node/judge majority paths (under ncu the board's own prepare races pick differently)
- **Expected effect:** Engine A K1/K2 and the wgmma hot half memory-bound at the larger T; combine / post-norm / align latency-bound
- **Decision rule:** classify each kernel memory-, consumer- or latency-bound; one ledger finding per kernel naming its lever
- **Metric:** ncu section metrics per kernel per shape (prof/ncu.csv), reps 0..5 of each shape's seeded draws; MDE 0.30% per row, 0.25% headline (R6, provisional); strata: distinct_local
- **Instrument:** gpunode06, 1 GPU per replicate, one process, caches WARM, reps 20, shapes decode 1, decode 16, decode 32, verify 1, verify 16, verify 32; void below 1800 MHz or on throttle (R1)
- **Bytes board:** `snap/board-498afbd8e73d.py` sha256 `498afbd8e73de786e57b31d96cba41e2d04eeb3772cec3017bb5d2994b94c2fe` (frozen from /home/user/git/musespark/muse_moe_w4a16/submission.py)
- **Arm pinned** (of board): post `{"ea.on.1": "v1", "ea.on.2..8": "v4", "ea.on.9..16": "v4", "ea.on.17..32": "v4c", "hm.on.128": 0, "hm.fon.128": false, "ea.on.128": "v4c", "hm.on.256": -16, "hm.fon.256": true, "hm.cold": "v4", "hm.cold_bits.128": 112, "hm.cold_bits.256": 0, "wg_on_128": false, "wg_on_256": true, "wg_pdl": true, "w2_mode": "atomic", "w13_mode": "atomic", "d32_align": 1, "d32_post_norm": 1, "ea.cmb_t.1": 1, "ea.cmb_t.8": 1, "ea.cmb_t.16": 1, "ea.cmb_t.32": 1, "ea.on.256": false}` -- node / judge majority paths, literal (X0a: class 8 tie-pinned v4)
- **Pre-registered:** audit 10-28 blocker, resolved: every prepare-time flag is pinned to production's literal value; GPU 7 ncu validation runs/20261027-195015-gpunode06 (all 6 shapes) shows the pinned arm's 6 FLAGS lines and 6 GRAPH lines (kernel lists, topology, pdl_edges) byte-identical to the production board's in runs/20261027-191006-gpunode06; the first P3 job re-checks this (its FLAGS / GRAPH lines are compared with production before any ncu.csv reading enters the ledger)
- **Pre-registered:** deviation: one replicate, because this is characterisation, not an A/B
- **Replicate 1:** config `configs/p3.json`, seed 2186850503301014679, job `queue/pending/36-h100-p3.job` (staged in queue-staged/)
- **Results:** (append below with run dirs, PATH CHOSEN of every party, clock, R3a / R3b tables; never edit the lines above)


## 2026-10-28T03:41Z: M00 result: E1 (in-process A/B) NOT accepted; m456 / m08t8 stay gated

- Run dirs: r1 runs/20261027-203851-gpunode06 (GPU 1), r2 runs/20261027-203854-gpunode06 (GPU 3). Both rc 0; all parties
  passed. Clock 1830 MHz flat, no throttle.
- **Fails the registered rule on CI coverage.** CIs covering 1, out of 12 row points: raced 8/12, same 7/12, same2 7/12
  (the rule needs >= 11/12). Points within +-0.3%: raced 12/12, same 10/12, same2 11/12.
- **Identity spread per row:** d1 0.31, d16 0.18, d32 0.28, v1 0.42, v16 0.43, v32 0.13 (%). That puts the in-process
  per-row MDE at about 0.45%, not 0.30%. The headline identity reads x0.9996-1.0006, so the 0.25% headline MDE holds.
  The bootstrap CI is within-capture only.
- **Sensitivity holds.** C verify 32 x0.919 / x0.921 (predicted 0.88-0.96). D verify 16 x0.968 / x0.964 (below 1;
  larger than the -1.5% predicted).
- **Gate not written** (takemyjob 04e8ace). m1 and p3 run as queued. Re-registering m456 / m08t8 at a 0.45% per-row MDE
  is the user's call.

## 2026-10-28T05:16Z: amendment (the user's decision, before any m456 / m08t8 data): E1 accepted at a 0.45% per-row MDE; gate opened

- **Amendment.** The per-row MDE for in-process readings is 0.45%, the M00 identity spread, replacing R6's provisional
  0.30%. E1 is accepted for this program at that MDE despite M00's CI-coverage miss (7-8/12), which stays on record.
  M00's sensitivity arms resolved as predicted, and the identity headline held within 0.25%.
- **Unchanged decision rules.** m456: pin at +0.5% or more in both replicates; K2 / combine only at +0.8%; ties go to R10.
  Readings must agree in sign with the reading against the identity arm and clear 0.45%. m08t8: the fit thresholds
  a >= 12 us and b >= 9.28 us per expert.
- **Released.** queue/gates/m00-accepted written; 32-m456 x8 and 34-m08t8 x2 are eligible. The amendments are also in
  takemyjob queue-staged/registrations.md.

## 2026-10-28T05:18Z: M1 result: the isolated A/A null holds; isolated MDE ~0.3% per row; no order effect; 0/12 race flips

- Six blocks (board vs a byte-identical copy, one judge-shaped process each), all on GPU 1 at 1830 MHz, rc 0. b1-b3 WARM,
  b4-b6 COLD; orders alternated AB/BA.
- **R3b paired:** rows x0.9968-x1.0011, all unresolved; headline x0.9991 [0.9974, 1.0008]. **Raw:** headline x0.9978
  [0.9956, 1.0001].
- **Isolated MDE (R6):** the 95% half-width is ~0.3% per row (v1 0.26 to d32 0.32), except decode 1 at ~0.70%;
  headline ~0.17%. That is tighter than the in-process instrument's 0.45% per row (M00).
- **Order effect:** AB x0.9989 vs BA x0.9993 (0.04% < 0.2%), so balancing is not mandatory. The GPU effect was not
  testable (all blocks on GPU 1).
- **Race flips:** 0/12 prepares flipped at any class (T=1 v1, 2..16 v4, 17..32 v4c, 128 v4c, 256 MASTER). X0a's 36-43%
  class-8 flip rate is not reproduced on the node in this state.
- Recorded in takemyjob 4f1dabb (registrations.md, m1-b6).

## 2026-10-28T05:32Z: M456 + M08t8 results (10 jobs, 05:20-05:24Z, all rc 0, replicates on distinct GPUs)

- **Class 8 (verify 1):** a tie. The best arm, v4, read +0.24% / +0.93% vs the board, which was not +0.5% in both
  replicates. R10 applies: **pin v4 at class 8**. The board raced class 8 to **v1** in the m456-c8 and m08t8 processes
  (4/4) but to v4 in all 14 other prepares (M1 12, c16 2). The race outcome depends on the process, and the pin removes
  that variance.
- **Class 16 (decode 16):** a tie (v4 +0.15% / -0.12%). R10: **pin v4**. Every v4c variant is resolved slower
  (-0.9 to -1.5%); v3 is -3%.
- **Class 32 (decode 32):** **no change**. v4c7 and v4c9 tie with the board; v4 -0.9%; the triton combine -1.2%.
- **Class 128 (verify 16):** **switch the combine to CUDA**: x1.0112 / x1.0096 vs the board, +1.04% / +0.85% vs the
  identity arm. That clears the +0.8% combine bar in both replicates, so it goes to the judge carrier.
- **Every K2-grid variant** is -13 to -14% in all four classes.
- **M08t8, the T=8 cost per extra local expert** (fit over k 2..12): v1 a 15.3-15.8 us, b 10.26-10.33 us/expert. v4 a
  19.4 us, b 9.85 us/expert. v4c a 18.2-18.5 us, b 10.1 us/expert. v1 and v4 cross at k ~8.7.
  - a >= 12 us in every arm, so fixed-cost work at T=8 is indicated. b >= 9.28 in every arm, so a streaming engine at T=8
    is indicated.
  - On the H100, b sits near the HBM3 floor (~9.4 us per 28.3-MB expert), so b does not transfer to the H200; a does
    better.
- **Carrier candidates for the judge** (not built): pin v4 at classes 8/16, and the CUDA combine at class 128. Details in
  takemyjob 52dd7c5.

## 2026-10-28T05:44Z: carrier check dispatched; m07, m03c, m08t1 released (registered 02:57Z, unchanged)

- **Moe carrier** (pin v4 at classes 8/16, the CUDA combine at T=128, on the S3 board): build ff6b525b, audit PASS with
  two minor findings, both applied (aa9bcb9b: skip v6's T=8 race only when the class-8 pin took). Judge check
  be4440e24e244103 dispatched 05:43Z from campaign/eb-carrier (962,450 B, sha256 d01f5b75...).
- **Released** to keep the H100s busy while P3b and the consumer microbenchmark are prepared. Gate m00-accepted; the
  registrations and decision rules are unchanged.
  - **m03c** (T=256 fork, cold-half knobs): the S3 board runs the same T=256 kernels and gate code, so a pin carries.
  - **m07** (K1 w2 prefetch at T=1) and **m08t1** (T=1 cost per extra local expert, nsys per k): moe's T=1 is v6 on
    S3, so their moe class-1 pin clauses are moot. They are read for decode's T=1 (board e1fd3c00 runs v1) and as the
    pre-S3 baseline for the consumer program.

## 2026-10-28T05:52Z: the M456 carrier on the judge: the T=128 combine transfers; landed (moe 1.7345)

- Check be4440e2 1.7457; scored 12d17a46 1.7345 (CI 1.7085-1.7605) vs the 1.7340 board: a tie on the headline, landed
  by the "beats" rule (master 81c9802e).
- **Transfer reading (R8):** the T=128 CUDA combine read +1.12% / +0.96% on the node (vs the board) and x1.023 (check) /
  x1.018 (scored) on the judge's verify 16 row: same sign, larger on the H200 (consumer-bound, so a fixed per-call
  saving is a larger share). The class 8 / 16 pins (node ties) read x1.015 / x1.007 at verify 1 and x1.010 / x1.001 at
  decode 16: inside the judge spread, as R10 predicts.

## 2026-10-28T06:04Z: P3 audit done; P3b registered (R2), its GPU-7 validation queued

- **P3 audit** (workflow wf_01ce0f10-ce0; research/p3-audit-2026-10-28.md). Engine A runs 24.9-27.1 HW warp
  instructions per 256-weight step (not ~17); the step normalization holds (HMMA/step 1.000 at q=1). v4 issues on
  42-55% of SMSP-active cycles, with warps stalling and waking together. The ALU-pipe floor is 1,720-1,810 busy cycles
  per 36,864-B record, the same for v1 and v4 at every T: **71-75% of the H200 consumer-alone 2,421 cycles and 79-83%
  of the 2,190 target**, so the target needs ALU >= ~80% busy and issue >= ~73%. Resources: v4 91 regs, 0 spills,
  113,008 B smem, 2 CTAs/SM (cap 112 regs at 288 threads). x loads add no excess wavefronts (1.9-2.2 per extra tile vs
  2.0 ideal). Waits (long_scoreboard ~1,100-1,180 cycles/record) are H100 memory-coupled and do not transfer.
- **P3b registration** (full text: takemyjob queue-staged/registrations.md `### p3b`, appended after the validation).
  - **Question:** where do the 25-27 instructions per step and the SM-active cycles per record go, per PC, at T=1 (v1),
    T=8 and T=16 (v4)? Same bytes (snap 498afbd8e73d), pinned arm and seed as P3, so reps 0..5 pair with P3's launches.
  - **Expected:** consume / consume2 carry >= 90% of executed instructions at T=8/16; ALU-class ~13-14 per step; the
    ~4.2/step double pipe charge is one opcode class (HFMA2.BF16 or IMAD); long_scoreboard on the consumers'
    full-barrier wait (smaller share on the epilogue's gs[le] LDG); ldmatrix bank conflicts ~0; per-PC shared
    wavefronts actual/ideal <= 1.1.
  - **Decision rule:** update the audit's evidence rows (a)-(e) by the registered thresholds (>= 95% of instructions
    attributed to named regions; a pipe saturated only at >= 80% of SM-active; H4 reopens at > 0.2 bank conflicts per
    two-tile step or > 1.1x ideal wavefronts on ldmatrix PCs). No kernel change follows from P3b alone.
  - **Metric / instrument:** ncu 2026.2.1, --clock-control none; SourceCounters, InstructionStats,
    ComputeWorkloadAnalysis, MemoryWorkloadAnalysis_Tables/_Chart + P3's six sections + 38 metrics, preflight-checked;
    exports per-instance and per-PC SASS; .ncu-rep kept. gpunode06, one pool GPU, reps 20, 6 profiled per shape (decode 1,
    decode 16, verify 1). VOID below 1800 MHz, on a check failure, or if I_step / tensor-busy per record miss P3's same
    rep by > 3%.
  - **Harness:** nj_profile.py d05ce8ba -> 635e9830 (the P3b patch, sha 52ae7432; selftest PASS; without the new keys it
    builds P3's ncu command byte for byte). configs/p3b.json from nj-exp make (identical to the audited draft but the
    registered stamp). Cost ~6 GPU-min.
  - **Validation (R1):** 37-h100-p3b-val on GPU 7 (<= 5 min): rc 0, preflight stop false with the required items
    listed, per-PC SASS rows for ea4_w13/ea4_w2, opcode-keyed instance values, .ncu-rep kept. FAIL: P3b is not queued.

## 2026-10-28T06:08Z: M07 + M03c + M08t1 results

- 6 jobs (05:45-05:49Z), rc 0, 1830 MHz, 0 check failures; replicate GPUs 1/3, 4/5, 6/1. m07-r2 and m03c-r1 share
  runs/20261027-224533 (same launch second), but both outputs are complete and self-consistent, so neither is VOID.
- **M07 (K1 w2 prefetch, T=1):** **close prefetch for moe and decode.** pf3 x0.973 / x0.971 vs v1 (x0.978 / x0.976 vs
  identity); pf2 at k=2 x0.981 / x0.974, k=3 null holds. +0.7-1.4 us/call; nothing for J0. PDL off -12%, v4 -13 to -15%.
- **M03c (T=256 fork, cold half):** **no pin**, no S3 carrier. Best: WG PDL off +0.39% / +0.38% (inside the MDE); cw9
  ties; cw7, cw11 and cold v3 are 0.5-1.6% slower. E3: the cold half blocks the hot path for 167-172 of ~342 us, and
  every knob trades that blocking for path time. Serial H-16 -7.7% (the fork is worth x1.083 on the node).
- **M08t1 (T=1 cost per local expert, fit over k >= 1):** v1 a 13.5-13.9 us, b 10.40-10.45 us/expert; v4 a 21.2-21.4,
  b 9.35; v3 a 16.6-16.8, b 10.29; pf8 a 12.6-12.8, b 11.7 (all-k v1 11.1-11.4 + 10.9 k, as on GPU 7). a >= 12 in all:
  **fixed-cost work at T=1** (v1 by 1.1-1.5 se). b >= 9.28 in all, but split w13 64-75% / w2 23-34%: **no streaming
  engine.** v4 never beats v1 (ties at k=8). r2's identity row resolves (board capture slow at k <= 1): its R3a vs board
  is void until explained; fits and E3 unaffected. m08t8 (T=8) splits ~2:1 too; its 05:27Z call skipped this clause.
- Recorded in takemyjob 1f96029 (registrations.md).

## 2026-10-28T06:09Z: correction to the 05:32Z M08t8 line (streaming engine at T=8: not indicated)

- The registered rule is "b >= 9.28 us per expert **in one kernel** -> streaming engine". The 05:27Z harvest read b
  (9.85-10.33) against 9.28 without the one-kernel clause. The per-expert cost splits between w13 and w2 roughly 2:1, as
  their bytes do (8192 vs 4096 rows), so no single kernel carries 9.28 us per expert. **Corrected decision for T=8:
  fixed-cost work indicated (a >= 12 us, unchanged); streaming engine not indicated.** The v4 class-8 pin is unaffected
  (it rests on m456-c8 and R10). The 05:32Z entry stays as written; this line supersedes its streaming-engine clause.

## 2026-10-28T06:11Z: M09 registered (R2): T=1 cost per local expert on the current board (v6 vs v1 vs v4)

- **Question:** on the current moe board (81c9802e = S3 + carrier pins; bytes frozen at make time), what are Engine A
  v6's fixed cost a and per-expert cost b at T=1 (fit over k >= 1), against v1 and v4 forced, and where does a sit
  (K1, K2, combine, launch gaps, PDL waits; nsys per k)? k in {0,1,2,3,4,6,8}. Arms: same, t1v6 (forces v6), t1v1, t1v4.
- **Expected:** b(v6) ~9.3-9.8 us/expert (the H100 HBM floor, ~9.4 us per 28.3-MB expert, hides v6's judge-side slope
  gain of 8.5-9.0 vs ~10); a(v6) ~10-12.5 us (co-resident K2 + PDL cut 1-4 us vs v1's 13.5-13.9); most of a in K1's
  prologue, the K1->K2 edge and the combine.
- **Decision rule:** a(v6, k >= 1) >= 12 us: a fixed-cost T<=8 path (fused K1/K2/combine or a persistent kernel) is the
  T=1 lever for the next phase-A plan, sized by the E3 split. a(v6) < 12 us: fixed-cost work at T=1 is closed for v6,
  and the T=1 lever is the consumer (the eac sweep). No pin (judge: no).
- **Metric / instrument:** as m08t1: BY_K per party, R3a per k at the 0.45% MDE vs the base and vs same, E3 split per
  k (fidelity tol 3%); gpunode06, one pool GPU per replicate (R4), reps 1400, 28 profiled; void below 1800 MHz or on a
  check failure. Jobs 39-h100-m09t1v6-r1/r2, prio after P3b; spec nj-exp check rc 0. ~8 GPU-min in total.

## 2026-10-28T06:19Z: M09 result (T=1 on the current board) and P3b validation PASS

- **M09** (r1 GPU 1 / r2 GPU 3, rc 0, 1830 MHz): the node routes v6 at T=1, as the judge does. k >= 1 fits: **v6 a 12.5
  us (se ~0.95) + ~10.2 us/expert**; v1 13.8 + ~10.4; v4 21.3 + ~9.35 (v1 and v4 reproduce m08t1 within 0.1-0.3 us).
  v6 is 3-8% faster than v1 per k except k=4. **Decision: a >= 12 (marginal, ~0.5 se): a fixed-cost T<=8 path is the
  T=1 lever for the next phase-A plan.** Sizing: v6 cut a by only ~1.3 us; between the k=0 floor (7.0 us) and a (12.5)
  sit ~5.5 us of pipeline fill / K1->K2 hand-off; the combine is on the critical path for 2.5-3.8 us at every k (fuse it
  into K2's epilogue). v1's fixed cost splits K1 70% / K2 13% / combine 17%. takemyjob e3b7f04.
- **P3b validation PASS** (hand-run on GPU 7, 06:15-06:17Z; runs/20261027-231530-gpunode06): preflight lists every
  required item (one optional metric dropped), per-PC SASS rows and opcode-keyed instance values present, .ncu-rep kept.
  First opcode mix (ea4_w13, verify 1): per HMMA step LOP3 6.4, IMAD 4.3, HMUL2 4.0, SHF 2.3, LEA 2.2, PRMT 2.1, LDSM 1,
  HMMA 1, BRA 0.9, ISETP 0.8, LDS 0.45 (27.9 per step at kernel level). P3b queued at 06:20Z (37-h100-p3b, one pool GPU).
- Ops note: a bare gpus=7 in a job file is a count (seven pool GPUs) and GPU 7 is reserved, so GPU-7 validations are
  hand-run (queue/README.md, takemyjob 19815b9). The staged p3b-val was pulled before it ran.

## 2026-10-28T06:20Z: P3c registered (R2): per-PC anatomy of the current board's T=1 fixed cost

- **Question:** where does the T=1 fixed cost go, per PC? M09: v6 a 12.5 us + ~10.2 us/expert; k=0 floor 7.0 us; the
  combine on the critical path for 2.5-3.8 us. ncu with P3b's sections, metrics and exports on v6 K1 / K2 (ea6_kernel)
  and the CUDA combine (ea_combine_kernel) at decode 1, k_local 1 and 4 (4 profiled reps each); arm pinned to the
  board's own T=1 picks (ea.on.1 v6, cmb_t.1 = 1); bytes = master's moe file 81c9802e (snap board-d01f5b757bfd).
- **Expected:** at k=1, K1's SM-active cycles are dominated by the ramp (prologue, staging, first-record fill) rather
  than the steady-state loop; K2's waits sit on griddepcontrol.wait and the K1->K2 hand-off; the combine is
  latency-bound at ~2.5-3.0 us; at k=4 the consumer loop carries >= 80% of executed instructions (25-27 per step).
- **Decision rule:** characterisation (no pin): name and size, in cycles per launch, the PCs that hold K1's and K2's
  non-loop cycles at k=1 (prologue, staging, barrier waits, griddepcontrol.wait, epilogue). That ranks the fixed-cost
  levers for the next phase-A plan. VOID on a dropped required item, a check failure or a clock below 1800 MHz.
- **Instrument:** gpunode06, one pool GPU, reps 20, ncu 2026.2.1 via nj_profile.py 635e9830; validation
  nj-moe-val-p3c hand-run on GPU 7 first (2 profiled reps, launch count 6; <= 5 min). ~4 GPU-min.

## 2026-10-28T06:39Z: decode track: E2 merged into hopper-probe master; DEC-01 re-run and DEC-02 registered (R2)

- **E2** (decode shim + judge-like inputs) is merged into master (fast-forward 06:25Z; takemyjob 2207160, staging
  69fcce6 / 37ea89e). It needed two decode-only fixes: E3's profiled pass checks decode replays with written=static,
  and the E1 eager check uses the pristine-copy pattern. The moe path is unchanged: moe dry runs are byte-identical, all
  119 moe bundle members are identical, and the tests pass.
- **dec01a (DEC-01 re-run; hand-run on GPU 7, ~255 s):** do the decode shims match the reference math and SGLang's API?
  It runs on the frozen decode board (sha256 2fd3d945, e1fd3c00's bytes). PASS: rc 0, the shim self-test PASS at the
  DEC-01 thresholds, 'API board-...: 38 call sites bound ... ok', the FA3 and GEMM lines present, no VOID, and GPU 7 clean
  (no overlapping run). PASS writes queue/gates/dec01-pass; any miss blocks decode node work. The earlier dec01a and 11
  more E2 runs overlapped other GPU-7 runs; their verdicts stand, their timings do not.
- **DEC-02 x4** (c1daf58b, ced20a71, 8e3b9874, 5a33db71 = the board): is the decode node faithful to the judge?
  - Design: one cold judge-shaped process per bytes, with the 13 tests, the 12 timed shapes at the judge's reps, and a
    20-rep nsys pass at T=1 and T=32 (**the decode-layer T=1 timeline**).
  - **V2:** >= 12/13 tests inside the judge's err/bound envelope per test index, and the PATH and canary lines inside
    the judge's spread. **V3:** shim kernels within +-1 us of the judge profile, and the GEMM / FA3 names match.
  - Decision: all four must pass. V2 fail -> stop decode node work. V3 fail -> arms next to a shimmed op stay
    unresolved.
  - ~12 min each on four pool GPUs, gated on dec01-pass. Full text: takemyjob queue-staged/registrations.md.

## 2026-10-28T06:44Z: DEC-01 re-run PASS on a clean GPU 7; DEC-02 x4 released

- dec01a (runs/20261027-233921-gpunode06, 06:39:39-06:43:51Z, rc 0; no other run on GPU 7 in that window): shim self-test
  PASS (q/k 0.50 ulp, RMSNormGate 0.50 ulp, boundary stream rel 2.6e-8, routing weights rel <= 6.7e-8, 0 id-set
  mismatches); 38 API call sites bound; FA3 / GEMM lines present; no VOID. Gate dec01-pass written; the four DEC-02 jobs
  were moved into pending together (06:46Z) so they land on four GPUs.

## 2026-10-28T07:15Z: P3b + P3c results (P3b VOID by its registered rule; P3c VALID)

- **P3b** (runs/20261027-232220-gpunode06) is **VOID** by its registered pairing check: 12/36 launches miss 3% on pct x 4 x
  132 x gpc__cycles_elapsed (max 6.47%). Cause: the exported gpc elapsed comes from another of the 37 replay passes.
  Post hoc, the registered raw counters reproduce P3 within 0.02%. No amendment: P3b's readings are provisional, and
  P3c (VALID) reproduces (a)-(d) independently on v6.
- **Consume body** (the same in v1/v4/v6), per 256-weight step: ext_prop 11.0 (6 LOP3, 3 IMAD.SHL, 2 SHF) +
  scale_prop_rt 4.0 (2 PRMT, 2 LEA) + mul_bf16x2 4.0 (HMUL2) + LDSM 1 + HMMA 1 + LDS 0.375 + ~0.25 = 21.6-21.75.
  Per-record control adds 1.6-2.1. The ~4.2/step double charge is HMUL2.BF16 (fmaheavy + fmalite). No pipe reaches 80% of
  SM-active (ALU <= 55.7%). long_scoreboard sits 65-83% on the consumers' full-barrier wait. Every LDS/LDSM PC is at
  1.000x ideal wavefronts, so H4 is closed.
- **P3c** (runs/20261027-232407-gpunode06), T=1, standalone under ncu: K1 7.96 + 6.81k, K2 7.49 + 3.58k, combine 4.61 +
  0.09k us. Non-loop at k=1: ~2 us launch/teardown per kernel; 1.6-1.8 us prologue; K1 x staging 1.2 us; first-record
  fill 1.4 us. The combine is latency-bound on one SM (2.05 us boundary + 2.66 us).
- **Implications.** The fragment-order repack removes <= 2.1/step (pair 3 only = dq3's floor, measured x1.000-1.019).
  m-tile reuse r=2/4 removes 0.5/0.75 LDSM per step, with LSU at 9-12%. **Neither touches the 17/step per-weight
  dequant.** Fixed-cost levers at T=1 [E]: persistent T<=8 kernel ~3-4.5 us > fused K2+combine epilogue ~1.5-3 us > K1's
  x copy before build_map <= 1.2 us > prologue trims 0.3-0.8 us.
- Report: research/p3b-p3c-2026-10-28.md (script p3bc_analysis.py). Registrations: takemyjob 764a2ce.

## 2026-10-28T07:19Z: DEC-02 results: FAIL (V2 and V3 on all four bytes), so decode node work stops

- 4 runs, rc 0, 1830 MHz, 13/13 tests: w6g 234719 (GPU 4), trim 234724 (GPU 5), pin128 234716 (GPU 3), board 234714
  (GPU 1). Node headlines x1.357-1.391 vs SGLang; the judge read 1.572-1.592.
- **V2 FAIL.** err/bound inside the judge envelope: w6g 12/13, trim 13/13, pin128 11/13, board 13/13. PATH / canary
  fails on all four (11-14 of 20 races outside the judge's picks). Causes:
  1. Under E3's nsys wrapper the bytes' in-prepare profiler checks count 0 kernels (B12P and c320d forced to the
     incumbent).
  2. The T=1 Engine A gate falls to MASTER on 2/4 draws (x1.016-1.019 vs the judge's x1.050), and its atomic w2 then
     fails the T=1 canaries and copy checks.
  3. H100 timing races.
- **V3 FAIL.** The norm_residual shim is +1.24..+1.76 us per call at T=1. At T=1 cuBLAS picks a different qkv_gate
  kernel (64x8 vs the judge's 128x8). w6g decode-1 fidelity FAIL (x1.053).
- **Decode-layer timeline** (board, node; shares moe / dense / attention / small, 46 sliding : 16 full): decode 1 node
  38/43/12/7 vs judge 40/42/11/7; decode 32 node 47/26/23/3 vs judge 56/22/20/3. Decode 1 is dense + moe back to back,
  with almost no gaps (the pooler / router GEMV alone is ~29 us node, 17-22 us judge). Per-kernel transfer: dense
  x1.23-1.38, long-context attention x1.12-1.34, small ops ~1:1, T=32 expert block x0.90.
- **Reopening** needs a new registration (the user's call): forced judge picks via E1 arms, a profiled pass whose
  prepare does not run under nsys, and a faster boundary shim. Tool defects: records.py skips dec02-* configs;
  nj_profile truncates kernel names and merges the sliding and full cases per load. takemyjob 703f45f.

## 2026-10-28T08:01Z: eac registered (R2): consumer-only microbenchmark (Engine A dq2 loop, records resident in smem)

- **Question:** cycles per 16-row x 4096-k record per SM of Engine A's dq2 consumer loop at T = 1 / 8 / 16, at the
  production CTA shapes (S6 = v6 (320,2), S6W = (320,1), S4 = v4 2 x (288,2)). What changes them:
  - Q1, m-tile reuse r = 2 / 4 (K_s = 512/r sub-blocks or SB = 1);
  - Q2, the fragment-order scale repack F (L1) and the paired scale decode (L2 = F + paired; L3 = paired on today's
    layout; L4 = L3 with the add on IMAD);
  - Q3, x in registers (X2: v6 T <= 8, 64 registers; X1: v4 T=8 per-kq preload).
  Every arm is bitwise equal to its baseline.
- **Build-time facts:** the twins match P3b's production body within 0.13 instr/step. The paired decode folds (-2.0
  instr/step, all ALU; l3c NOFOLD). The register cap at (288,2) / (320,2) is 96, so no reuse variant fits v4 at T=16
  (SPILL is the port answer there).
- **Expected (tc130, baseline / variant):** L3 s4 t8 1.12 [1.05, 1.18], s6 t1 1.09 [1.03, 1.15], s4 t16 1.07
  [1.02, 1.13]. L1 1.000. R2 / R4 0.97-1.08 (SPILL at t16). X1 1.01, X2 1.04, X2L3 1.14. SC0 (uniform-scale ceiling)
  1.25.
- **Decision rule** (both replicates; tc133 must agree in sign): port L3 into eak::consume<2> if l3 >= 1.03 at s4 t8
  with its CI clear of 1.01 and s6 t1 >= 0.99 (into consume2 if >= 1.03 at s4 t16). F only if >= 1.02. Reuse becomes
  a port candidate only at >= 1.05, spill-free, with port_regs <= 96, else it is closed. X2 only at >= 1.05. A combo
  must beat its better arm by >= 1%. VOID on ok=0, mhz_min < 1800, an occupancy or smid mismatch, |ss_dev| > 1%, or a
  REF null out of band.
- **Metric / instrument:** marginal clock64 cycles per record per SM (256 vs 1,024 chunk-records), the median of 11
  shuffled reps with a bootstrap CI, MDE 1%. Bundle build/eac.tgz sha256 637d2e3c (eac_130 7ef7fbee, eac_133
  735b202b). Validation hand-run on GPU 7 (~3 min, Tier A+, DQX exhaustive dequant check, memcheck), then 41-r1 / 41-r2
  (one pool GPU each) and 42-ncu on gate eac-val-ok. Full text: takemyjob registrations.md `### eac` (a62947b).
- The L3 production port (workflow wf_5e31da3c-b6f) runs in parallel. Its judge check is the transfer test, and this
  sweep is the per-cell attribution.
