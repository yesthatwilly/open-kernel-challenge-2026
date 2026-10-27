# Gate 0 and calibration on an H100 SXM node (2026-10-27)

X5 Gate 0 (`SINGLE-PASS.md` §3) on gpunode06 (8x H100 80GB HBM3 SXM, drained), run by session 0605b7cc through
takemyjob hopper-probe under the guard:
- runs `20261027-112303-gpunode06` (es-cal, GPUs 1 and 3) and `20261027-112308-gpunode06` (es-kstep, GPUs 4-6);
- raw copies: `../runs/hopper/20261027-1123-gpunode06/` (`es-cal.txt.gz`, `es-kstep.txt.gz`);
- code: musespark `campaign/eb-sp` c492091d (`rewrite/lib/cuda/es_common.cuh`, `es_probe.cuh`); takemyjob 9b50ccf
  (`probes/es`) and 6014007 (`probes/escal`).

All SM clocks held at 1830 MHz (min 1829) for the whole run, with no throttle reasons. GPUs 0 and 2 were not used,
because they idle hot.

## 1. Calibration: node clocks reproduce the judge's consumer-bound clocks

`es-cal` reruns the judge's own CUDA text, byte for byte: W3-B's DIAG kernels, the production `wg_kernel` and the
expert-engine ladder, with the judge's exact W3-B routing draw. It covers both toolchains on both GPUs.

| class | result (cu130, GPU 1) | median ratio to the judge |
|---|---|---|
| consumer-bound | 37/37 within 5% | 0.986 |
| mixed | 13/14 within 7% | 0.999 |
| memory-bound | 35/40 within 7% | 1.050 (after the HBM ratio) |

- The HBM ring reads 3.05 TB/s on the node against the judge's 4.45 TB/s, a ratio of 0.687.
- GPU 3 and cu133 agree to within one row.

**Consequence:** consumer-bound results in el/clk/SM from this node transfer to the judge. Memory-bound results must
be scaled by the ring ratio, or measured on the judge.

## 2. Gate-0 K-step table

The table gives el/clk/SM for the cu130 build at the best tiles x WGs. Notes:
- `devres` runs device-wide with the records resident in smem, which is the consumer ceiling.
- `dev` runs device-wide streaming 4.75 GB from HBM.
- 424 KV lines are all ok=1 (bitwise or host-checked), with c75xx 0, spill 0 and lmem 0. SASS shows no serialized
  HGMMA beyond (a)'s design.

| variant | N=8 | N=16 | N=32 | N=64 | N=128 | N=256 |
|---|---|---|---|---|---|---|
| (a) 8 wgmma -> wait<0>, constant A | 78.7 | 78.7 | 63.9 | 32.0 | | |
| (b) wait<1> per k16 | 73.4 | 70.8 | 61.1 | 32.0 | 16.0 | 8.0 |
| (b8) 8 wgmma, wait<1> per unit | 78.7 | 78.7 | 63.9 | 32.0 | 16.0 | 8.0 |
| (c) 4 independent tiles | 78.7 | 78.7 | 63.9 | 32.0 | | |
| **(e) dq2 + RS, R=2 ring, devres** | **32.3** (t2w2) | **29.8** (t4w2) | 27.0 | 21.0 | | |
| (es) streamed from HBM, dev | 24.4 | 25.8 | 23.9 | 20.2 | | |
| (f) M-cell (Engine A mma.sync) | 30.8 | 27.4 | | | | |

For reference, Engine A's compute-only C-R4 runs at ~27 el/clk/SM on the judge.

**Readings:**
- **The RS wgmma itself is not the small-N problem.** With constant A, two WGs issue one m64nNk16 per ~13 clk per SM
  at N=8/16 (26 clk per k16 per WG). From N=64 up it runs at the tensor rate of exactly N/2 clk.
  - W3-B's "~93 clk per wgmma at N<=32 whatever N" was therefore its loop: serial `wait<0>`, lockstep consumers and
    dequant in the same window. It is not a hardware latency.
- **The cold path is dequant-bound.** (e) peaks at 29.8 el/clk/SM at N=16, which is 76% of the dq2 ALU ceiling of
  ~39.4 (§2). That is x1.10 over Engine A's 27 and x1.09 over the M-cell (27.4). At N=8 it reaches 32.3 against the
  M-cell's 30.8 (x1.05).
  - More issuing work per WG (tiles 2-4) adds 10-19% over t1.
  - The second consumer WG adds 15-40%.
- **(es) on the node is HBM-bound:** 25.8 el/clk/SM x 132 x 1.83 GHz x 0.5 B/el is ~3.1 TB/s, the H100 ring. It cannot
  show the streamed consumer rate here. The judge (4.45 TB/s) can.
- **Register question (§3.2 step 4):** ptxas sizes the consumer region at the setmaxnreg value, not at the 168 launch
  budget (maxR 191-212 in e_n8/16/32_t4). cudaFuncGetAttributes keeps reporting 168. So setmaxnreg works as designed,
  and variant B's C7512 was not a 168-register artifact.
- **Sanitizer:** memcheck reports 0 errors in every family (a, b, c, e, f, g, s). Synccheck reports "Barrier error
  detected. Divergent thread(s) in block" in every family. This is likely the warp-specialized `bar.sync 0` reached
  from different producer and consumer paths. **Open: before Engine S production code, use named barriers with
  explicit counts and get synccheck clean.**

## 3. Verdict (§3.2 step 5)

**(e) at N=16, device-wide = 29.8 el/clk/SM (both toolchains), in the "< 30" band, at its edge.**
- The cold path (T<=32 and the cold half at T>=128) stays on Engine A for now.
- X5 = **S1a** (Engine S on the hot list inside the hybrid) **+ the T<=8 fixed-cost work (K9)**.
- **S1b is not ruled out** (hot + cold in one launch at T>=128). The cold-unit gain over Engine A here is x1.05-1.10,
  under the x1.10 block-level calendar kill. So S1b needs K4/K6 evidence (the streamed one-consumer pipeline and the
  mixing test) and a judge reading of (es), not this table alone.
- **The hot path is the clear win.** At N>=64 the RS K-step can run at the tensor rate (32 / 16 / 8 el/clk/SM at
  N=64 / 128 / 256), where wg_core's measured MMA path ran 20-23 el/clk/SM.
  - The hot design should keep the A ring per k16 with wait<1>, and push dequant off the critical path. At N>=64, the
    (e)-style per-k16 dequant (21.0 at N=64) is below the tensor rate.
  - The next rungs to build are K4 (one-consumer pipeline) and K6 (mixing).

## 4. Next (X5, from the plan)

1. The Gate-0 judge check (moe carrier, §4.2). It gives absolute judge rates, (es) at 4.45 TB/s, the judge's
   nvcc/.text hashes for the cu130 bridge, and the K9 replay histogram if it is added.
2. S1a: an Engine S hot-unit kernel (stream-K over hot items, N<=256 in one pass, separate weight and B rings),
   raced as a new H code in both hybrids.
3. The decode trim (`campaign/eb-sp-trim`), then the decode cash-in.

## 5. Judge confirmation: moe check 5a64104026ab4f91 (2026-10-27 19:02Z; passed 1.7199, the board's timed path)

The carrier is `campaign/eb-sp` 7997607e: 20 symbols in prepare(), 22.6 s, with prepare ending at 147 s.
`runs/5a64104026ab4f91.json.gz` keeps 112 `[es-g0]` lines.

**The bridge holds.**
- R0: torch 2.14.1+cu130, CUDA 13.0, H200 with 132 SMs.
- All 20 kernels' `.text` sha256 values on the judge equal the node's cuda-13.0.1 builds (node run 20261027-115801-gpunode06).
- So SASS and life-range readings from the node are readings of the judge's binaries.
- The `cuda-13.0.1-gcc-12.1.0` module is the judge's compiler for our purposes.

The judge's SM clock under this load reads 1769 MHz (clock64 / globaltimer), against the node's 1830.

**Consumer-bound el/clk/SM matches the node to the decimal.**

| kernel | judge | node |
|---|---|---|
| e16.4.2 | 29.8 | 29.8 |
| e16.2.2 | 29.7 | 29.7 |
| e8.1.2 | 28.5 | 28.5 |
| f16 | 27.4 | 27.4 |
| a16 / b8_16 / c16 | 77.9 / 77.9 / 78.0 | 78.7 |
| a64 / b8_64 | 31.9 | 32.0 |

**The streamed (es) form at the judge's HBM** (dev, el/clk/SM, TB/s of 0.5 B/el):

| kernel | el/clk/SM | TB/s |
|---|---|---|
| es16.4.2 | **27.5** | 3.60 |
| es16.2.2 | 25.8 | 3.37 |
| es16.1.2 | 22.7 | 2.93 |
| es8.1.2 | 22.1 | 2.87 |
| es32.1.2 | 20.5 | 2.65 |
| es64.1.2 | 17.9 | 2.30 |

- es16.4.2 is 92% of its resident ceiling and x1.02 over Engine A's ~27.
- (es) is bitwise equal to (e) on every symbol.

**Verdict, confirmed on the judge:** cold stays on Engine A / the M-cell; X5 = S1a + the T<=8 fixed costs.
- The only route back to a cold Engine S is the ladder's P1: 3-4 consumer WGs (S3u / S4 / D2) lifting the dequant
  rate.
- No scored run: the timed path is the board's, so 1.7199 vs 1.7187 is noise, and the bytes carry diagnostics.

## 6. Stage-0 ladder on gpunode06 (probes/esl; musespark eb-sp 9b11171b, takemyjob 357ff8c; validation runs 20261027-1323..1329-gpunode06)

All 1,072 KV lines are ok. memcheck and synccheck are clean (named barriers). 170 symbols with 0 C75xx, 0 spills and
0 local memory. Values are el/clk/SM on tc130.

**P1, cold with more consumer WGs** (resident ceiling, best tiles; dev):

| N | S2 (2 WGs) | S3u (3) | S4 (4) | D2 |
|---|---|---|---|---|
| 8 | 32.9 | 33.9 | 33.0 | 32.7 |
| 16 | 29.6 | 29.8 | **30.4** | 29.2 |
| 32 | 25.9 | 27.2 | 27.3 | 26.1 |

- More consumer WGs add at most +3% at N=16. The cold path stays capped at ~30 against Engine A's ~27, below the 34
  that S1b / S2 need. **Cold stays on Engine A.**
- K2's dequant-only ceiling at 4 WGs is 34.7-35.4 (the fold4 sink had understated it by 7-14%), but the combined K-step
  does not reach it.

**P2, the hot K4 pipeline** (streamed weights, separate B ring, the §4.1 schedule):

| N | best | target / tensor | read |
|---|---|---|---|
| 256 (S2 r2) | 7.92 | tensor 8 | eta 1.00: **go**, tensor-bound |
| 128 (S2 r4) | 12.64 | target 12.8 | at target |
| 64 (S4) | 18.2 cta / 17.9 dev | target 25.6 | **missed** |
| 64 (S2 r2) | 14.1 | | |
| 64 (D2) | 17.0 dev | | |
| 64 (S3h) | 9.2 | | |

- Weight-ring depth (SW 8 / 12 / 16) does not matter.
- At N=64 the overhead sits per unit: c1 = 1.34x K1 (ring-free), c2 = 1.54x K2. Stamps put a per-unit barrier, the
  next-unit LDS and the release step at ~2x a normal step.
- **S1a input:** hot units at N >= 128 run at or near the tensor rate. Units around N=64 (most hot experts carry 17-80
  routes) need longer units or more WGs (S4) to amortize per-unit costs.

**Other rungs:**
- K6 mixing (S3h): no go. The hot role runs x1.57-1.82 slower when mixed, so keep the halves separate.
- K7: B ingest ~47.8 clk per k16, flat for N <= 64, once the producer threads are split.

## 7. Proving on the node (moe board, master bytes)

- **Held-out sweep** (nodejudge moe-sweep, run 20261027-125905): **37/37 tests pass**, the judge's 5 plus bs 2..255
  and 5 skewed.
- **Soaks** (moe-soak x2, 1000 reps per shape, every replay checked vs the fp32 reference, fresh inputs): **12,000
  checked replays, 0 faults, 0 miscompares**. Headlines x1.3900 / x1.3862.
- **Flip rates:** 4 cold-prepare runs routed every moe race identically; headlines 1.3893-1.3929.
- **Replicates** (8 queued jobs, 20:30-20:40Z, different GPUs): the cold N=16 rows reproduce within 0.16% (S4 30.35 /
  30.40). The K4 rows' median spread is 0.55%. The ladder numbers are stable to well under the effects they decide.
