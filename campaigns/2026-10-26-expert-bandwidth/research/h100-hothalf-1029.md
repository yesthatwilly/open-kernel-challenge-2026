# H100 lane: what bounds the moe/decode wgmma HOT HALF at T=128/256? (2026-10-29, gpunode06, 18:06Z-18:35Z)

Owner: an H100 lane agent in the 10-29 18:04Z window ("use the h100 for anything ... next 20 hours"). Rules as in
`okc-hopper-probing`: gpunode06 only, USER, no Slurm, node-local `/tmp/USER-hopper`, every run through guarded
`bin/hopper-probe` / audit-run. GPUs **1, 3, 4** (timing replicates: one process per GPU, all three at once; ncu on GPU 1).
The node was clean at the end: no hh process, no hh files. A pre-existing orphan `decs` fuzz under compute-sanitizer
(pids 2789298 / 2789304, CUDA_VISIBLE_DEVICES=4, 19 h old, no GPU context listed) was left untouched. GPU 4's timings
agree with GPUs 1 and 3 to within the round spread.

**Transfer key.** `[cyc]` marks per-SM cycles, instruction counts, stall shares, pipe-utilisation shares and A/B ratios of
this consumer-bound kernel. These carry over to the judge's H200 (same GH100 SM, 132 SMs). `[hbm]` marks absolute us,
DRAM % and the ring-only (D3) times. These do not carry over: the H200 has about 1.5x the HBM bandwidth, so its ring
is *less* binding and the kernel is *more* consumer-bound than here.

## Verdict (pre-registered decision rule)

- **Rule 1 (port A3): the outcome holds; the stated mechanism does not.**
  - A3 is bit-exact and makes the hot half (w13 + w2, PDL pair) **x0.902-0.923 at decode-like T=256** (8.3-10.9%
    faster) and **x0.894-0.898 at T=128** (10.2-10.6% faster). On moe's judge draws it reads x0.866 at T=256 and x0.890
    at T=128 [cyc].
  - The rule's premise is not met. The tensor pipe is 53-61% active at T=256 (37-42% at T=128). Issue slots are only
    40-52% used and the ALU pipe 27-42%: nothing is saturated.
  - A3 pays because the dequant sits on a **serial per-stage chain**: dq_stage, then 8 wgmma, then `wgmma.wait_group 0`,
    then the release. Fewer instructions shorten that chain (a latency effect), not a saturated pipe.
  - Action: the A3 port to moe + decode is the 10-30 phase-A build. Expected headline is below.
- **Rule 2 (producer/ring): no.**
  - The top named stall is `barrier` (19-24% of samples), but per-PC it sits on `WARPGROUP.DEPBAR`: the consumer waiting
    on its **own** wgmma, not on the ring.
  - Consumers wait on the full mbarrier only ~4-5% of samples (the branch after `SYNCS.PHASECHK`).
  - The producer spends 45-58% of its time waiting on the *empty* barriers: it waits for the consumers.
  - D3 ring-only runs in 0.57x of dq0 at T=256 [hbm; smaller still on the H200].
- **Rule 3 (scheduling): borderline, a second lever.**
  - Tail and wave quantization is **22% of the stamped pair span at T=256 a/b, 27% at c**, 14-22% at decode-like
    T=128, 11% / 40% on moe's judge draws.
  - w2 alone is 29-36% idle at T=256: 192 items on 132 CTAs is 1.45 waves.
  - In production the hot half runs forked next to the cold half, which backfills part of that idle time. The
    standalone 22-27% is therefore an upper bound on what a work split recovers.
- **Rule 4 (near floor): no.**
  - The K-step is 1.36x the padded-N tensor floor at N128, 2.1x at N64 and 4x at N32.
  - The N buckets compute 576 token columns for 386 real routes at T=256 a (67% useful).

## Instrument

- **Package.** takemyjob `prototypes/hopper-probe/probes/hh` (branch `hopper/hh-1029`).
  - `gen.py` reads **master 10a3e862** `muse_moe_w4a16/submission.py` (blob 691018b5) from the git object and runs its own
    `_wg_src(True)`. Host-only cuts are as in escal.
  - Variants are asserted edits: `dq0` (master's kernel, unchanged), `a1`, `a3`, `st` / `stw` (clock64 stamps), and
    `diag` (escal's W3-B DIAG text with master's fast silu, giving D0 / D1-noMMA / D2-constA / D3-ring). `gen/MANIFEST.txt`
    has every hash.
- **Build.** cu130 = `cuda-13.0.1-gcc-12.1.0` (V13.0.88), with the judge's load_inline flags. Every kernel: 168 regs,
  0 spill, 0 stack, no C75xx. The dq0 `wg_kernel<true>` has 3376 SASS instructions, the same as ncu's PC count (ncu
  profiled this binary).
- **Data.**
  - Synthetic bank in the exact record layout: random E2M1 codes, E4M3 scale bytes uniform over every positive normal
    [0x08, 0x7E], 32 local experts, plus a3's word-permuted copy.
  - x ~ N(0,1) bf16. The hot class is `_hm_align` H=-16 (experts with > 16 routes, block 16).
- **Routings.**
  - `dec-T256-{a,b,c}`: 6 hot experts, 386 routes (the decode board's T=256 anatomy, 7cc034f9 log). Counts
    a [98,80,66,56,46,40], b flat [65x2,64x4], c [150,70,50,46,40,30].
  - `dec-T128-{a,b,c}`: 4 hot experts, 169 routes. Counts a [62,45,35,27], b [43,42,42,42], c [90,30,26,23].
  - `jd-T256` / `jd-T128`: moe's exact W3-B judge draw (8 / 3 hot experts).
- **Timing.** escal's `rt_time`: one kernel (or the w13 -> PDL w2 pair) per CUDA graph, a 256 MB read flush before
  every replay, interleaved and rotated, empty-graph baseline subtracted, 61 reps x 3 rounds on each of GPUs 1/3/4
  (9 rounds per cell). The SM clock read 1830 MHz throughout (NVML).
- **Exactness.** Every exact variant (a1, a3, st, stw, D0) is **bit-identical to dq0** on w13 and w2 in all 8
  routings, compared over the full output buffers. The a3 bank permutation was also self-checked against ext_master on
  every 16-bit half pattern.
- **Runs.** hh + hh-ncu `runs/20261029-112341-gpunode06` (`out/hh/gpu{1,3,4}.txt`, stamps tsv, `out/hh-ncu/` with the
  .ncu-rep, raw / details / source csv, build SASS in `build/bin/`); quick validation `runs/20261029-112141-gpunode06`.
  Reducers are in `probes/hh/tools/` (`time_reduce.py`, `stamp_reduce.py`, `ncu_reduce.py`, `ncu_source.py`).

## 1. ncu (2026.2.1, `--clock-control none`, kernels serialized; dq0 unless noted) [cyc except DRAM]

| kernel (routing) | us | tensor active (of SM-active / elapsed) | issue active | ALU / FMA pipe | per-SM active min/avg/max (k cyc) vs elapsed | avg/elapsed | max/avg | warp inst / gmma | DRAM % |
|---|---|---|---|---|---|---|---|---|---|
| w13 T256-a | 83.7 | 55.9% / 50.4% | 46.1% | 37.0 / 19.6% | 89.7 / 127.9 / 138.1 of 141.8 | 0.90 | 1.08 | 39.6 | 44 |
| w2 T256-a | 53.4 | 53.1% / 37.8% | 42.8% | 30.8 / 20.3% | 42.8 / 67.4 / 91.1 of 94.5 | **0.71** | **1.35** | 38.2 | 36 |
| w13 T128-a | 46.7 | 37.2% / 33.3% | 52.1% | 42.0 / 21.8% | 41.4 / 74.8 / 80.3 of 83.6 | 0.90 | 1.07 | 39.1 | 52 |
| w2 T128-a | 26.2 | 34.8% / 29.8% | 47.5% | 34.4 / 22.7% | 4.1 / 40.0 / 42.8 of 46.6 | 0.86 | 1.07 | 38.2 | 48 |
| a3 w13 T256-a | 77.4 | 61.1% / 54.6% | 40.1% | 30.9 / 16.1% | 83.0 / 117.1 / 127.2 of 130.8 | 0.90 | 1.09 | 31.8 | 47 |
| a3 w2 T256-a | 48.8 | 57.8% / 41.3% | 37.0% | 26.5 / 16.2% | 38.5 / 61.8 / 83.7 of 86.5 | 0.72 | 1.36 | 30.4 | 39 |

- **Occupancy and launch.** 1 CTA/SM, grid 132, 384 threads (setmaxnreg 40 / 232), 168 regs, dynamic smem
  187,392 B (w13) / 203,776 B (w2). Warps active is 14.2% (9 of 64: 8 consumer warps plus the producer).
- **Waves.** T256-a: w13 has 384 items (2.91 waves), w2 has 192 (1.45). T128-a: 256 (1.94) / 128 (0.97).
- **Tensor-active check.** Tensor active at T256-a w13 (55.9%) equals the padded-N MMA floor:
  64 tiles x 32 stages x (3 x N128 + 3 x N64) x 128 x 128 MAC at 2048 MAC/clk/SM over 132 SMs = 71.5 k cyc, against
  127.9 k active.
- **Warp stalls** (ncu ratios; `selected` excluded): `barrier` 20-24%, `wait` 17-23%, `dispatch_stall` 6-12%,
  `long_scoreboard` 9-16%, `short_scoreboard` 4-6%, `gmma` 2-5%, `math_pipe_throttle` 2-3%.

**Per-PC attribution** (`ncu_source.py`, SourceCounters samples, dq0 T256-a w13 / w2):

| region | share of samples | dominant reasons |
|---|---|---|
| K-loop int-ALU (LOP3 / SHF / IMAD / PRMT) | 26% / 22% | selected, dispatch, wait |
| `WARPGROUP.DEPBAR` (wgmma wait_group 0) | **19% / 18%** | barrier (97%) |
| K-loop uniform datapath (descriptors, loop) | 13% / 14% | wait, selected |
| K-loop dq-FP (HMUL2 / HADD2.F32 / FMUL / F2FP) | 10.5% / 12% | selected, dispatch, wait |
| K-loop branch on the full-mbarrier try_wait (ring wait) | 3.8% / 5.3% | long_scoreboard (90%+) |
| HGMMA issue | 3.7% / 3.3% | wait, dispatch |
| everything else (setup, epilogue, producer, prologue) | 17% / 20% | long_sb, wait, barrier |

**SASS per stage (2 records = 8 k16 steps) per consumer warp**, from the ncu per-PC dynamic counts of the cu130 build
(identical for the N64 and N128 instantiations) [cyc]:

| | dq0 (master) | a1 | a3 |
|---|---|---|---|
| K-loop body, warp-inst per stage (per k16 step) | **287 (35.9)** | 255 (31.9) | **223 (27.9)** |
| LOP3 / IMAD.SHL / SHF / LEA.HI | 65 / 27 / 22 / 8 | 65 / 27 / 22 / 8 | 49 / 16 / 25 / 0 |
| HMUL2.BF16 (scale multiply) | 32 | 32 | 32 |
| scale path: HADD2.F32 + FMUL + F2FP (unpack e4m3 / pack bf16) | 16 + 16 + 16 | 0 (PRMT 9 + IMAD 10) | 0 (PRMT 9 + IMAD 10) |
| uniform datapath (ULOP3 / UMOV / UIADD3 / USHF: wgmma descriptors, loop) | 43 | 43 | 43 |
| HGMMA / LDS / SYNCS / WARPGROUP / control | 8 / 4 / 2 / 2 / 3 | same | same |

- Each SMSP holds 2 consumer warps: 574 issue slots per stage at dq0, against a measured N128 K-step of about 1390
  cycles. That is ~41% issue use, matching ncu.
- The 43 uniform-datapath instructions per stage (15% of the body) are the wgmma descriptor builds. Hoisting them is
  a further cheap cut (not measured).

## 2. Decomposition of the hot half (clock64 stamps, `st` variant)

**Stamp variants.**
- `st` stamps per CTA and per item (entry, setup end, first stage ready, K-loop end, epilogue end) plus the producer's
  empty-barrier wait. It costs +2-6% at T=256 and up to +15% on some T=128 instantiations.
- `stw` also times every full-barrier wait. It costs +7-20%, so stw is used only for the wait share.
- Cycles are `clock64`; the SM clock under load read 1.72-1.79 GHz from cycles over globaltimer.

**K-step** (cycles per 128-k stage, every routing) [cyc]:
- N128: **1390 ± 5** (tensor floor 1024, x1.36)
- N64: **1090-1210** (floor 512, x2.1)
- N32: **1020-1035** (floor 256, x4.0)
- With stw, waits inside the K-loop are 9-14% of it.
- Reading: the chain costs ~1000 cycles per stage below N64, and MMA adds on top of it. That is the signature of a
  serial dequant -> wgmma -> wait<0> chain with partial overlap between the two consumer warpgroups.

**Pair span (w13 + w2), stamped** [cyc shares]:

| routing | stamped span (us) | last CTA: K-loops | epilogues | setup | fill (first-stage waits) | other | balanced span | tail / wave share |
|---|---|---|---|---|---|---|---|---|
| dec-T256-a | 139.4 | 121.6 (87%) | 7.2 (5%) | 4.5 (3%) | 3.8 (3%) | 2.3 | 108.3 | **22%** (w13 13%, w2 36%) |
| dec-T256-b | 131.6 | 116.4 (88%) | 5.4 | 4.2 | 3.7 | 2.2 | 103.1 | **22%** (w13 12%, w2 36%) |
| dec-T256-c | 155.3 | 136.5 (88%) | 7.3 | 5.0 | 3.3 | 2.6 | 114.0 | **27%** (w13 25%, w2 29%) |
| dec-T128-a | 76.7 | 65.7 (86%) | 2.9 | 3.7 | 2.8 | 1.6 | 64.0 | 16% |
| dec-T128-b | 76.3 | 65.4 (86%) | 2.9 | 3.5 | 2.8 | 1.6 | 65.3 | 14% |
| dec-T128-c | 81.4 | 69.0 (85%) | 3.7 | 4.0 | 3.0 | 1.6 | 63.1 | 22% |
| jd-T256 (moe) | 130.3 | 115.1 (88%) | 5.4 | 3.7 | 3.4 | 2.5 | 116.4 | 11% |
| jd-T128 (moe) | 80.8 | 68.5 (85%) | 3.7 | 4.0 | 2.9 | 1.6 | 48.5 | 40% |

How to read the T=256 hot half (dec-T256-a; dq0 graph time 128.7 us on this H100):
- **Steady K-step x steps: ~87% of the critical path.** Inside it, MMA at the padded-N floor is ~62% (1024 / 1392 at
  N128, 512 / 1093 at N64) and the exposed dequant / sync chain is ~38%.
- **Fixed costs: ~13%.**
  - Setup (the chunk-table build): ~3.9k cyc per kernel.
  - Fill: entry -> first stage ready ~7.7k cyc per kernel, i.e. setup plus the first TMA / bulk latency.
  - Epilogues: 1.5k (N64) / 2.7k (N128) cyc per item.
  - PDL already hides some of w2's prologue in production.
- **Tail / wave quantization (GPU-wide idle): 22%,** mostly w2 (192 items, 1.45 waves, CTA busy 69%).
- **N-bucket padding** (not in the split above): 576 token columns are computed for 386 routes.
- CLAIMS:62's "gap is fixed costs" is therefore not what this measures. Fixed costs are ~13%. The gap is
  per-stage chain overhead (x1.36-4 over the MMA floor), N padding (x1.5) and the w2 tail.

**W3-B DIAG rungs on master's text** (fraction of dq0, 9 rounds) [cyc for noMMA / constA; hbm for ring]:

| routing | w13 noMMA / constA / ring | w2 noMMA / constA / ring |
|---|---|---|
| dec-T256-a | 0.574 / 0.767 / 0.578 | 0.568 / 0.768 / 0.567 |
| dec-T256-c | 0.500 / 0.725 / 0.502 | 0.541 / 0.761 / 0.542 |
| dec-T128-a | 0.647 / 0.745 / 0.649 | 0.718 / 0.794 / 0.714 |
| jd-T256 | 0.672 / 0.726 / 0.674 | 0.737 / 0.806 / 0.739 |

- In every cell, dequant-only with the ring (noMMA) costs exactly the ring alone: the dequant hides completely under
  the ring.
- MMA with the ring (constA) is 0.73-0.81 of dq0. Putting the dequant back costs +24-38%. That is the serialization
  the A1 / A3 cuts attack. Prefill's rega reads the same way (dequant-only ~= MMA-only, contention).

## 3. A/B: a1 / a3 vs dq0, interleaved, GPUs 1 / 3 / 4 x 3 rounds (mean [min, max] over the 9 rounds) [cyc]

| routing | w13 a1 | w13 a3 | w2 a1 | w2 a3 | **pair (w13 -> PDL w2) a1** | **pair a3** | dq0 pair us |
|---|---|---|---|---|---|---|---|
| dec-T256-a | x0.972 | x0.923 | x0.961 | x0.907 | x0.972 [0.971, 0.973] | **x0.923** [0.922, 0.925] | 128.7 |
| dec-T256-b | x0.961 | x0.906 | x0.962 | x0.905 | x0.970 [0.969, 0.973] | **x0.916** [0.914, 0.918] | 122.1 |
| dec-T256-c | x0.954 | x0.902 | x0.959 | x0.909 | x0.955 [0.953, 0.956] | **x0.902** [0.901, 0.903] | 145.7 |
| dec-T128-a | x0.943 | x0.880 | x0.967 | x0.895 | x0.962 [0.961, 0.966] | **x0.895** [0.891, 0.898] | 67.4 |
| dec-T128-b | x0.947 | x0.886 | x0.962 | x0.885 | x0.965 [0.963, 0.967] | **x0.898** [0.896, 0.900] | 68.2 |
| dec-T128-c | x0.931 | x0.874 | x0.978 | x0.933 | x0.940 [0.937, 0.943] | **x0.894** [0.891, 0.895] | 77.6 |
| jd-T256 (moe) | x0.894 | x0.828 | x0.959 | x0.891 | x0.925 [0.923, 0.927] | **x0.866** [0.864, 0.868] | 122.7 |
| jd-T128 (moe) | x0.926 | x0.870 | x0.970 | x0.932 | x0.941 [0.940, 0.943] | **x0.890** [0.889, 0.892] | 76.7 |

- Controls: D0 (the diag TU's master path) is x0.990-1.006 of dq0 in every cell. Round spreads are <= ±0.4%.
- The gain grows as N shrinks (more N32 / N64 items): the exposed chain is a larger share of the K-step there.

## Expected headline (decode elasticity; estimate, judge-unverified)

- **Assumption: the fork is zero-sum.** Decode's hot and cold halves share the SMs: decode's fork gate is only
  x1.013-1.026 over serial, and h100-expert-1028 found the halves zero-sum. So a hot-half SM-time cut of (1 - r)
  shortens the row by about (1 - r) x the hot half alone.
- **Hot half alone (judge anatomy, 7cc034f9):** 134.5 us at T=256, 74.4 us at T=128. Gather excluded, w13 + w2 ~131 /
  ~71 us.
- **A3 at the H100 decode-like ratios** (x0.914 T=256, x0.896 T=128):
  - verify 32: -11.3 us per row. Sliding 427.7 -> 416.4, full 700.4 -> 689.1; load x1.023.
  - verify 16: -7.4 us per row. Load x1.021.
  - Decode headline over 6 loads: **x1.0074, about 1.690 -> 1.703 (+0.7%)**.
- **Range:** 0 if the cold half sets both rows' critical path, ~+0.9% if the H200's less-binding ring lets the ratio
  improve.
- **moe:** only verify 32 runs the hybrid (H-16f; the T=128 hybrid is refuted on moe). jd-T256 x0.866 on a ~110 us
  judge-standalone hot half is -14 us on the 297.9 us row (board 0069fec4): load x1.05, **moe headline about +0.8%**,
  same caveat.
- A1 alone (no bank change) gives about a third of that: x0.955-0.972 at T=256.

## Levers, ranked

1. **A3 port to wg_core's dq_stage** (moe + decode share it: moe submission.py:1128 / decode :15330).
   - Bit-exact, same registers (168), no spill.
   - Cost: a word-permuted copy of the hot experts' code bytes in prepare (the bank is already a lossless retiling; a3
     only permutes bits inside each code word), plus the scale-range precondition [0x08, 0x7E], which the judge's
     [8, 448] clamp meets.
   - Expected +0.7% decode, +0.8% moe (above).
2. **w2 work split / persistent w13 + w2 (rule 3, borderline).**
   - Standalone, w2 idles 29-36% at T=256 (1.45 waves); the pair's tail / wave share is 22-27%.
   - The fork's cold-half backfill already recovers part of it, so size it on a forked anatomy before building.
3. **Descriptor hoist.** 43 uniform-datapath instructions per stage (15% of the K-loop body) rebuild the 8 wgmma
   descriptors every stage. Unmeasured; a cheap follow-on to A3.
4. **Not levers:**
   - The producer / ring (consumers wait on it ~4% of samples).
   - More stages.
   - Overlap via wait<1> / ping-pong. Refuted before (D4-varB serialized C7512, wg2 / ping-pong x0.4-0.9). Here the
     time is consumer issue plus MMA, already partly overlapped between the two consumer warpgroups.

## Caveats

- The decode-like counts are synthetic. They match the anatomy's expert and route totals, not the judge's exact
  per-expert counts. The a / b / c spread (x0.902-0.923 at T=256) brackets the shape effect.
- `st` perturbs the timing (see section 2). K-step and share numbers come from st (item-level stamps only); wait
  shares come from stw.
- Standalone hot half, not forked: in production the cold half shares the SMs, which changes the tail reading (rule 3)
  more than the A3 ratio.
- Ratios are cu130 SASS. The judge's nvcc is inferred to be 13.0, not logged.
