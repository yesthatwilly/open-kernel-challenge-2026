# Expert engine forensics (read-only Plan agent, Opus, 2026-10-26)

Verbatim structured output of workflow run wf_7532bad1-294 (next-campaign-research). Do not edit; corrections go in ../FINDINGS.md.

## Summary

The biggest scoring gain is the NVFP4 expert block at T=128/256, i.e. the decode verify_16 and verify_32 rows. The expert block takes about 280 us of v16 sliding's 411 us. It takes about 410 us of v32 sliding's 546 us, and 207 us of that is the serial Marlin hot half of the hybrid (hotcold_decode.py:209-248). Reading every expert once with an engine whose consumer is not starved for warps would take about 110 us off v16 rows and 215 us off v32 rows. That moves the decode headline from 1.442 to about 1.62 (inferred, about ±0.03). Adding a fix to the consumer's warp count at T<=32 brings it to about 1.70. moe would go from 1.576 to about 1.9-2.0.

Three corrections to the premise:

(1) v3 does not run at 1.1-1.4 TB/s at T=128. That figure divides by the speed-of-light bound's (SoL) touched count, 0.75 x router_distinct. Backing the judge's mean distinct experts out of each case's SoL gives decode 18.4 at T=128 and 21.3 at T=256, and moe 22.8 and 24.4. With those, v3 runs 2.22 TB/s on unique bytes in moe T=128 and 1.5-1.8 in decode, where the profile times include PDL waits. On streamed bytes it runs about 2.6-2.7 TB/s, the same rate as moe T=32 (2.56-2.61). The T=128/256 losses are:
- re-streaming and re-dequantizing hot experts once per 16 routes: 1.17x more bytes in moe T=128, about 1.3x in decode T=128, 1.5-1.7x at T=256;
- passes with two n8 tiles: 2.84 vs 3.40 TB/s in the ladder;
- activation rows re-fetched from L2 for every 64-row group: about n_local x 1.5 MB, which is 0.4x the weight bytes at T=128 and about 0.9x at T=256;
- at decode T=256, the Marlin hot half running at 0.9 TB/s.

(2) At every T the shared limiter is a latency-bound mma.sync consumer. Each warp issues only 0.22-0.32 instructions per clock in every ladder rung. No issue, ALU or smem pipe is above about 50%. smem caps v1/v2/v3 at 2 CTAs/SM, which is only 2 consumer warps per SMSP.

(3) The decode gates do not see the judge's routing. `_cb_data` builds an isotropic stream, so its T=128 inputs touch 23-26 experts against the judge's ~18-19, with roughly half the judge's hot rows. The moe v3 gate for T=128/256 never timed the reference-style judge draw that eb_moe.py already has.

Recommended order:
1. One moe ladder check to pin three unknowns: the mma.sync tensor rate on H200, the per-instruction cost of wgmma RS vs SS as N grows, and whether hot and cold work on the same SM actually overlap. Fix the decode gate routing in the same window.
2. S1: 8 consumer warps per CTA, keeping Engine A's consume() unchanged. This is the shared consumer for cold experts at every T.
3. Then, chosen by the ladder, either D1 (mma.sync passes covering up to 32-64 routes, simplest) or D2 (hot experts read once on a wgmma warpgroup that shares the CTA with mma.sync cold warpgroups, largest gain). D3 (dequantize once into smem, then wgmma SS) is the fallback if SS beats RS by a wide margin.

Critical files:
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/lib/cuda/ea3_kernels.cuh
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/lib/cuda/ea_kernels.cuh
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/lib/cuda/eb_kernels.cuh
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/lib/cuda/ea_ladder.cuh and ea_ladder.py
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/lib/ea_core.py (_ea_pack/_ea_repack)
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/lib/hotcold_decode.py
- /home/user/git/musespark/.worktrees/decode-first-rewrite/rewrite/src/muse_decode_layer.py:6157-6199 (_cb_data)

## Findings

### F1

The '1.1-1.4 TB/s at T=128/256' figure is an artefact of dividing by the SoL's 0.75 x router_distinct and of PDL-inflated profile times. On streamed bytes v3 runs the same at T=128 as at T=32. Per expert: w13 is 18,874,368 B (16 MiB codes + 2 MiB scales), i.e. 1024 stages of 18,432 B (128 groups x 8 chunks of 512 k); w2 is 9,437,184 B (512 stages); together 28.31 MB.

**Evidence.** Decode SoL formula muse_decode_layer/reference.py:681-697 and moe SoL muse_moe_w4a16/reference.py:125-136, inverted with the per-case sol of 0d84f58ca33b424b and f0fc89031659499c. Implied router_distinct: decode d1 2.5, v1 7.5, d16 7.5, d32 9.0, v16 18.4, v32 21.3; moe 2.6 / 8.4 / 11.9 / 16.0 / 22.8 / 24.4. moe profile f0fc89031659499c: T=128 ea3 w13 188.9 + w2 101.3 us gives 645 MB unique -> 2.22 TB/s, and 27.3 passes x 28.31 MB = 773 MB streamed -> 2.66 TB/s. T=32: 114.1 + 62.5 us gives 453 MB -> 2.56 TB/s unique, 2.61 streamed. Decode v16 (0d84f58ca33b424b profile 8): 195.6 + 159.1 us for 521 MB unique -> 1.47 TB/s on profile time; about 1.76 if w2's exclusive time is about 100 us.

**Measured or inferred.** Kernel times measured. Distinct counts derived exactly from the SoL formula. moe pass counts from a numpy re-simulation of generate_input (reference.py:53-78), which reproduces the SoL distinct counts within 0.4. The w2 PDL-wait share is inferred.

### F2

Why v3 loses at T=128/256: every pass covers at most 16 routes, so an expert with n routes is streamed and dequantized ceil(n/16) times.
- Passes with 9-16 routes run consume2 (one ldmatrix.x4 plus 2 mma.sync per k16 per warp); the ladder measured 2.84 TB/s for it vs 3.40 TB/s for one tile.
- Judge-like pass/distinct ratios: moe T=128 27/23 (1.17x), moe T=256 38/25 (1.54x); decode T=128 about 24/19 (about 1.3x), decode T=256 about 38/22.
- At moe T=256 the median hot expert has 93 routes (p90: 256), i.e. 6-16 passes.

**Evidence.** ea3_kernels.cuh:23 (CAP 16), 41-113 (consume2), 106-109 (ldsm_x4 + two mma16816), 219-222 (one-tile vs two-tile dispatch). Ladder a997f2587a654f7c.profile.log:126-127: R4 nt1 tok8 3.40, nt2 tok16 2.84. numpy moe simulation (2000 draws): T=128 p50 187 routes, 23 distinct, max 47, passes16 27; T=256 p50 369 routes, 25 distinct, max 93 (p90 256), passes16 38.

**Measured or inferred.** Code and ladder measured; routing percentiles simulated.

### F3

How the B operand is gathered:
- Per stage, producer lanes 0..v_n-1 each issue a 1 KB bulk copy of one x row (w13; gathered via pos_r) or one h row (w2) into 1040 B padded smem rows, on the same mbarrier as the 18,432 B weight record.
- So every (64-row group, 512-k chunk) re-fetches the pass's activations from L2: v_n MB per w13 pass and v_n/2 MB per w2 pass, i.e. n_local x 1.5 MB per call. That is about 270-280 MB at T=128 (0.4x the unique weight bytes) and 555-600 MB at T=256 (about 0.9x).
- Inside the CTA, each of the 4 consumer warps re-reads the same B slice with ldmatrix: about 64 KB of B reads per 18 KB weight stage at 16 routes.

**Evidence.** ea3_kernels.cuh:162-180 (weight_copy / x_copy, 16 x 1 KB copies), 27 (XS = 16 x 1040), 44-46 and 107 (per-warp ldmatrix addressing). The x/h buffers are L2-resident: x is T x 8 KB, h is n_local x 8 KB.

**Measured or inferred.** Derived from code. The L2->SM ceiling itself is unmeasured: v3 at moe T=256 'random' pushed about 1.67 GB in 345.6 us (about 4.9 TB/s) per 8589aad509334dff.profile.log:65.

### F4

Work map, split, occupancy:
- Grid 264 = 2 CTAs/SM x 132; 160 threads (4 consumer warps + 1 producer warp).
- v3 dq2 uses 93-96 regs and 105-111 KB smem, so occupancy is smem-limited (registers would allow 4 CTAs/SM). That gives 8 consumer warps/SM, 2 per SMSP. Ring D=3, so at most 110 KB/SM of weight slots.
- At T=128 the w13 grid has about 27.6k stages, about 105 per CTA (w2 about 52); n_v x NG > 264, so it runs plain stream-K.
- Each CTA has at most 2 partial groups: a 4 KB fp32 partial, a ticket, and a fixed-order sum by the last arriver. This costs under 2%.
- Every CTA rebuilds the 9,420 B map (64 chunks, thread 0 builds the pass list serially), an estimated 2-4 us at kernel start that is mostly hidden behind PDL.
- The epilogue does scattered 2-byte bf16 stores.

**Evidence.** ea_decode.py:467-475 (grid = cps*sms); ea3_kernels.cuh:143-157 (aligned / stream-K split), 238-279 (fix-up), 280-307 (epilogue), 329-341 (D=3, smem 114,684 B, 2-CTA static_assert); ea2_kernels.cuh:28-42 and 95-109 (MapT, serial pass list). Log lines 'kernel v3 w13/dq2: 93/96 regs, 105/111 KB smem, 2 CTAs/SM' in 22ac0cb836e34645 and 8589aad509334dff.

**Measured or inferred.** Measured (attributes and code). The fix-up and map costs are inferred.

### F5

Decode T=256 is dominated by the serial hybrid:
- Marlin hot w13+w2 takes 206.9 us on about 6.5 hot experts (about 184 MB, 0.9 TB/s).
- Then v3 cold takes 108.3 + 86.1 us on about 14.8 experts (about 419 MB, 2.2 TB/s), plus _hc_align 9.7 us.
- That is about 410 us of the 546.5 us v32 sliding row (and of the 816 us full row).
- The moe T=256 wg path is 246.2 + 152.8 us for 691 MB (1.73 TB/s).

**Evidence.** 0d84f58ca33b424b.profile.txt profiles 10 and 11; hotcold_decode.py:209-248 (Marlin hot w13, silu_and_mul, Marlin hot w2, then ea3_w13/ea3_w2 cold); 0d84f58ca33b424b.profile.log:91 (PATH T=256 = HOT/COLD H=-48); f0fc89031659499c.profile.txt profile 5.

**Measured or inferred.** Times measured. The hot/cold byte split uses the simulated judge routing (inferred).

### F6

At T<=32 the consumer is latency-bound, not pipe-bound. Per warp per k16 step (256 elements, dq2) it issues about 21.5-23.5 warp-instructions:
- 0.5 LDS
- 4 scale ops (2 PRMT + 2 IMAD)
- about 10.5-11.5 code-extraction ops (pair0 1, pair1/pair2 2 each, pair3 about 6)
- 4 HFMA2
- 1 ldmatrix
- 1 HMMA (2 in consume2)
- about 0.6 for barriers and the loop
At the ladder's R4 tok8 rate (23.1 el/clk/SM) that is about 48% of issue, about 43% of the ALU pipe and about 42% of smem bandwidth. Each warp sustains only 0.22-0.32 instructions per clock in every rung, at both 1 and 2 consumer warps per SMSP, so throughput scales with warp count. At 1 CTA/SM, R4 nt4 tok32 collapses to 1.37 TB/s while the ring alone stays at 4.48.

**Evidence.** a997f2587a654f7c.profile.log:114-130, clk_stage_cta: R2 2385, R3 nt1 2562, R4 tok8 2835, nt2 tok16 3388 (occ2), nt4 tok32 3519 (occ1). Divided by 32 k16 steps that is 74-110 clk per k16 step per warp. Source: ea_kernels.cuh:49-119; ea_common.cuh:105-127 (ext_prop, scale_prop_rt), 75-79 (fma.rn.bf16x2).

**Measured or inferred.** Rates and clocks measured. Instruction counts come from source, not SASS (inferred).

### F7

Production kernels lose a further 10-20% against the ladder:
- Kernel-only rates are T=8 w13 3.00 / w2 2.78; T=16 v2 2.99 / 2.78; T=32 v2 2.72 / 2.59, v3 2.65 / 2.51 TB/s.
- T=1 is latency-bound at w13 14.5 us and w2 13.3 us (1.42 TB/s), about 1.34 us per stage per CTA at the tok1 rate.
- w2's aligned split leaves 72 of 264 CTAs idle when 3 experts are touched (G=192, 8 stages per CTA vs 5.8 under stream-K), costing about 3 us.

**Evidence.** f0fc89031659499c.profile.log:92-101; ladder R4 tok1 clk_stage_cta 2645 (line 125); ea3_kernels.cuh:145-150 and ea2_kernels.cuh:149-154 (aligned split rule).

**Measured or inferred.** Measured, except the per-CTA stage accounting at T=1, which is inferred.

### F8

Engine B v1 (check 3f402ec33b094c2e) was a 384-thread CTA with 3 warpgroups:
- a producer warpgroup and 2 consumer warpgroups, with setmaxnreg 32/104 (2 CTAs/SM) or 40/232 (1 CTA/SM);
- wgmma m64nNk16 RS with double-buffered register A and wait_group<1>;
- 36 KB weight stages plus 64-k SWIZZLE_128B TMA token slabs;
- runs of at most 64 or 128 routes; LPT order with cost-weighted stream-K.
It reached c2 w13 1.83-1.95 TB/s and c1 1.51-1.53 (c1 was read-once at T=256 judge-like, max 111 routes, yet still slow); its gates came out 0.63-0.98 against master.
What serialized: each consumer warpgroup both dequantized and issued wgmma, with only 4 wgmma per 64-k sub-step between waits. The header records about 64-98 SM clocks per m64xk16 wgmma whatever N <= 128 is, plus about 400 clk per commit group.

**Evidence.** git show 66b6f501:rewrite/lib/cuda/eb_kernels.cuh header lines 1-21 and Cell<> constants 41-50; 3f402ec33b094c2e.profile.log:83, 91-92, 93-104; current eb_kernels.cuh:7-10.

**Measured or inferred.** Measured. Note the 64-98 clk figure only comes from designs that serialized dequant and wgmma (master s5, wg2, EB v1); no clean probe of the wgmma RS instruction cost exists.

### F9

Engine B v2 (check 1bcc06feac904641) had 288 threads (8 consumer warps + 1 producer, 112 regs, no setmaxnreg):
- unified 128-k stages: a 9,216 B weight record plus two 4 KB token boxes;
- runs of at most 32 routes;
- cell m: mma.sync with N of 8/16/32;
- cell w: wgmma RS with N of 16/32 and independent accumulator chains.
It reached m 1.51-1.76 and w 1.54-1.71 TB/s. Diagnostics at T=128 w13: ring alone 4.04, no-dequant 3.51, no-MMA 2.90, and the three parts sum to 1.55x the full kernel ('OVERLAP').
What serialized:
- 8 k16 steps per warp per stage vs Engine A's 32, so 4x the barrier waits and releases per byte;
- all 8 warps re-read the token box through ldmatrix: 4 B/el of B reads at N=32, which by itself caps smem at about 3.4 TB/s;
- a non-unrolled runtime sub-step loop with 2x Engine A's LDS count, so its dequant-only rate (2.90) sits far below Engine A's R2 (4.04);
- runs of at most 32 routes still re-stream hot experts (ring-only drops to 3.57 at T=256 on unique bytes).

**Evidence.** eb_kernels.cuh:1-21, 25-40, 173-204 (dq_sub, uint2 loads), 341-399 (cell m runtime nsub loop, per-k16 ldsm_x2 per n-tile), 401-446 (cell w); 1bcc06feac904641.profile.log:77-80, 81-92; LOG.md:171-173.

**Measured or inferred.** Rates measured. The attribution of the serialization to those causes is inferred from code plus smem arithmetic.

### F10

What Engine B (and v3) did NOT try:
1. Dequant warps writing bf16 A tiles to smem that feed wgmma SS: never tried; both versions used RS.
2. Separating dequant and MMA into different warpgroups linked by named barriers or mbarriers: never tried. v1 had 3 warpgroups but each consumer warpgroup did both jobs, and named_bar was used only in the fix-up.
3. Mixing hot (compute-heavy) and cold (bandwidth-heavy) tiles on one SM: never tried. v2's LPT order runs the hot runs first on all CTAs, then the cold ones.
4. Register-A double buffering or software pipelining for mma.sync: never tried (it was used only for wgmma cells).
5. Read-once hot passes with N >= 128 at 2 CTAs/SM: never tried.
6. An 8-consumer-warp CTA built on Engine A's unrolled 512-k consume(): never tried.
7. One B tile shared across 128+ rows with mma.sync: never tried.
8. Any gate on judge-faithful routing for the v3 T=128/256 classes: never tried.

**Evidence.** eb_kernels.cuh:365-446 and 472-485; git 66b6f501 eb header lines 11-17; build_map LPT at eb_kernels.cuh:206-284; 8589aad509334dff.profile.log:59-66 (gate kinds zipf and random only).

**Measured or inferred.** From code and logs.

### F11

The decode gate inputs are not judge-like:
- `_cb_data` sets stream = randn(T, 8192) * 3, isotropic.
- The judge's stream has a 0.7 mean direction plus rank-4 shared structure, and the router rows carry a component along that mean (reference.py:140-148, 274-284). So the judge's routing is concentrated.
- Gate inputs: T=128 161-171 routes, 23-26 distinct, max 25-28, 48-67 hot rows; T=256 318-325 routes, 27-29 distinct, max 43-45.
- A numpy replica of the judge's stream gives median distinct 8.0 / 9.5 / 18.8 / 21.8 at T=16 / 32 / 128 / 256 (SoL-implied: 7.5 / 9.0 / 18.4 / 21.3). At T=128 it gives about 180 routes, max 33-66, 3-5 hot experts with 85-140 rows; at T=256 about 400 routes, max 64-123, 4-9 hot with 290-500 rows.
- moe: v3 at T=256 was rejected at x0.614 on a zipf routing with about 1154 local routes (LOG.md:228-229), while winning x1.32 on 'random'. The judge median is about 370 routes and was never timed for v3.

**Evidence.** rewrite/src/muse_decode_layer.py:6184 and 6157-6199; 0d84f58ca33b424b.profile.log:55-56, 62-63; 8589aad509334dff.profile.log:63-66; LOG.md:179-182; eb_moe.py:44-55 (_eb_judge_draw, reference-style draws, already exists).

**Measured or inferred.** Code and logs measured. Judge routing statistics simulated (inferred); the model matches the SoL-implied distinct counts within 0.5 at 4 T values.

### F12

Resource constants at the judge clock:
- The judge H200 reports clocks.max.sm 1980 MHz. At that clock, 4 TB/s is 15.3 B/clk/SM = 27.2 NVFP4 el/clk/SM including scales (30.6 counting code bytes only). The task's 34.5 el/clk corresponds to code bytes at about 1.75 GHz.
- Dequant alone needs about 1.8 issue slots/clk and 1.0 ALU slot/clk per SM at 4 TB/s.
- wgmma m64nNk16 costs N/2 clk at peak, so hot experts with N=128/256 are tensor-bound at 16/8 el/clk/SM in any engine.
- Total MoE FLOP at T=256 (about 37 GFLOP) is only about 35 us at wgmma peak against about 155-175 us of memory time, so the whole-GPU problem is memory-bound if hot tensor work overlaps cold ALU work.

**Evidence.** 0d84f58ca33b424b.json.gz result.machine.gpu_after (clocks.max.sm 1980); FLOP = routes x 2 x 3 x 4096^2.

**Measured or inferred.** Clock measured; the rest is arithmetic (inferred).

## Options

### O1: D2 (T=33..256, largest gain): hot/cold warp-specialized CTA, a wgmma RS hot warpgroup sharing the CTA with mma.sync cold warpgroups

**Design.** Layout: 1 CTA/SM, 4 warpgroups, 512 threads; setmaxnreg 40/112/112/232 (pool = 64K registers exactly).
- WG0: producer warps (cold weight ring; hot weight ring; TMA B boxes).
- WG1 + WG2: cold consumers, i.e. S1/D1 consume() on experts with at most 32 routes.
- WG3: hot consumer, one pass per hot expert covering all its routes, N = 8*ceil(n/8) <= 128 (two halves above 128). BM = 128: two m64nNk16 RS instructions per k16 share one B sub-tile.
Hot path details:
- The A fragments come from Engine A's dequant on the same 512-k records, consumed as 8 x 64-k sub-steps (eb dq_sub shows the per-warp 16x16 layout is shared by mma.sync and wgmma RS).
- Double-buffered register A with wait_group<1> (Engine B v1's loop).
- B = 64-k SWIZZLE_128B boxes of the gathered x (eb_prep-style gather for w13) or the sorted h (w2).
Per call, the device map:
- sorts experts by count and sets hot = more than 32 routes;
- computes class costs and splits each class over 132 CTAs (stream-K within the class, last-arriver tickets per class);
- lets the hot WG take cold units once its list empties.
smem about 200-220 KB: cold D=2 x 35 KB; hot 2 x 18.4 KB weights + 2-3 x 16 KB B; maps.
Throughput model per SM:
- hot WG: tensor-bound at N/2 clk per m64k16, i.e. 16 el/clk at N=128 or 32 at N=64; its dequant costs about 0.6 ALU slots/clk and 1.2 issue slots/clk;
- cold WGs: 8 warps x 0.24 IPC, about 1.9 instructions/clk, about 22 el/clk;
- combined demand 38-54 el/clk exceeds memory's 27-30, so the SM is memory-bound if the overlap holds; ALU about 73%, issue about 85%.

**Expected gain.** Every expert read and dequantized once; hot tensor time hidden behind cold streaming. Inferred:
- decode v16 expert block about 280 -> about 170 us; v32 about 410 -> about 190 us.
- moe T=128 296 -> about 205 us; T=256 398 -> about 200-215 us.
- Decode headline about 1.62 from T=128/256 alone (v32 alone gives x1.074, v16 alone x1.048); about 1.70 with S1.
- moe about 2.0.

**Effort.** High: about 5-7 days. New CTA shape, wgmma hot path with per-call TMA maps, setmaxnreg pools, per-class balance and fix-ups.

**Risks.** - The cost of a wgmma RS instruction at N=64-128 while another warpgroup dequantizes is unmeasured, and EB v1 c1 read-once at 1 CTA/SM reached only 1.51 TB/s.
- One CTA/SM leaves no latency hiding across CTAs.
- A 256-route expert (moe T=256 p90) makes the hot WG the bottleneck.
- The smem budget is tight.
- The setmaxnreg pool must equal launch registers x threads.
- This is the largest code change before the 2026-11-05 06:59Z close (PLAN.md:21).

**Reuse.** - Engine A consume/dequant/ring: ea_kernels.cuh:49-119, ea3_kernels.cuh:117-309.
- Engine B v1's wgmma RS loop and setmaxnreg pools: git 66b6f501 eb_kernels.cuh.
- Current eb_kernels.cuh: make_desc/EbWg (76-168), cell w double buffering (401-446), eb_prep gather (650-722), cost-weighted stream-K (208-305).
- Per-call TMA map encoding in the eb host code.

### O2: D1 (T=33..256, lowest risk): Engine A v4, mma.sync passes covering all of an expert's routes on an 8-consumer-warp CTA

**Design.** Built on S1's CTA (producer warp + 8 consumer warps, 2 CTAs/SM, 16 consumer warps/SM).
Lossless repack: [expert][128-row group][256-k chunk][8 warps][16 rows x 256 k = 2,048 B codes + 256 B scales]. A stage stays 18,432 B, but its B tile (N x 512 B) now serves 128 rows, halving v3's L2 activation traffic.
Passes = (expert, up to 32 routes), or up to 64 if the ladder shows at least 0.7 HMMA/clk/SM. The number of n8 tiles (1/2/4/8) is chosen per pass on the device.
Consume = Engine A's dequant + one ldmatrix.x4 per 2 tiles + NT mma.sync. D=3 at N <= 32: 3 x (18.4 + 16.6) KB = 105 KB, so the map moves to global memory or shrinks.
Work order: hot passes first (LPT) under cost-weighted stream-K (cost per stage about max(1, NT/2)), with the existing last-arriver fix-up and PDL chain. Marlin is removed at T=256 and the hybrid retired.
Throughput model at 4 TB/s (27.2 el/clk/SM):
- 21.5 / 22.5 / 25.5 instructions per 256 el at N = 8 / 16 / 32;
- 16 warps x 0.24-0.3 IPC gives 3.8-4 instructions/clk, i.e. 38-45 el/clk of demand, so N <= 16 passes are memory-capped;
- N=32 passes are smem-capped by per-warp ldmatrix: 4 B/el of B reads plus about 1.5 B/el of A reads and writes gives about 23 el/clk, i.e. 3.4 TB/s-equivalent;
- HMMA demand at N=32 is 0.43/clk/SM (the ladder's measured lower bound is 0.31).

**Expected gain.** Expected rates (inferred):
- cold streaming 3.2-3.6 TB/s; N=32 passes about 3.0;
- hot experts above 32 routes are still read ceil(n/32) times (re-reads partly from L2).
Expected times (inferred):
- decode v16 about 280 -> about 190 us; v32 about 410 -> about 265 us;
- moe T=128 296 -> about 210 us; T=256 398 -> about 290 us.
Decode headline about 1.56 from T=128/256 alone; about 1.64 with S1.

**Effort.** Medium: about 3-4 days on top of S1. Repack variant, consume with 4 and 8 tiles, N-class map, retire the hybrid.

**Risks.** - mma.sync's tensor rate on H200 is unknown; below 0.5 HMMA/clk/SM, N=32 passes fall to about 2.5 TB/s.
- The new repack must be proven lossless (prepare-time roundtrip).
- smem sits at the 2-CTA limit.
- Hot experts are still dequantized ceil(n/32) times, which at T=256 leaves about 1.3-1.4x more streamed bytes.

**Reuse.** ea_kernels.cuh consume and ea_common.cuh dequant; ea3 producer/ring/fix-up/epilogue (ea3_kernels.cuh:117-309); eak2::build_map (ea2_kernels.cuh:49-118) plus a per-pass N-class; Engine B first_unit/owner/LPT (eb_kernels.cuh:208-305); _ea_pack (ea_core.py:30-49) with nk = K/256 and 8 warps; ea3_core forward and the existing adapters, canary, copy check and gates.

### O3: D3 (T=33..256, alternative): dequantize once into smem as bf16, then wgmma SS, with dequant and MMA in separate warps

**Design.** 1 CTA/SM.
- WG0: producer.
- WG1 + WG2 (8 dequant warps): read the NVFP4 ring and write bf16 A tiles (128 rows x 64 k = 16 KB) in the canonical 128B-swizzled K-major layout via stmatrix into a 3-slot A ring, handing off through full/empty mbarriers.
- WG3: issues SS wgmma m64nNk16 for both row halves, with N = all routes of the expert (up to 128; up to 256 at BM 64) and B from TMA boxes.
One structure serves cold and hot experts, and dequant (ALU) never shares a warp with MMA (tensor).
Throughput model: smem traffic is about 0.56 (ring read) + 2 (bf16 write) + 2 (wgmma A read) + N/32 (B) + fills, i.e. about 5.2 + N/32 B/el. That makes it smem-bound at 22 / 17.7 / 13.9 el/clk for N = 16 / 64 / 128 (3.3 / 2.6 / 2.05 TB/s-equivalent). The dequant warpgroups need about 1.9 issue slots/clk/SM at 23 el/clk.

**Expected gain.** Inferred: decode v32 about 410 -> about 250 us, v16 about 280 -> about 205 us; headline about 1.57-1.58 from T=128/256 alone. Choose it only if the ladder shows SS far cheaper than RS and the C6 rung beats D1/D2.

**Effort.** High: about 5-6 days.

**Risks.** - smem bandwidth caps it below D2.
- The bf16 write and read double smem traffic.
- The handoff between roles adds a stage of latency.
- Rules: never spill dequantized weights to global memory.

**Reuse.** Engine A dequant and ring; master's make_desc/SWIZZLE_128B and stmatrix (judge-proven forms); Engine B TMA maps and stream-K.

### O4: S1 (T<=32, and the cold half of D1/D2): fix the consumer's warp starvation with 8 consumer warps per CTA, 2-record stages, D=2

**Design.** consume() and the repack stay unchanged. The CTA becomes a producer warp + 8 consumer warps (288 threads).
- A stage = two Engine A group records of the same expert and k-chunk (groups 2j and 2j+1: two 18,432 B bulk copies on one full barrier) plus the pass's x rows once, shared by 128 rows.
- Empty barrier count 8; D=2. smem is about 2 x (36.9 + 8.3 or 16.6) KB plus the v2 map (2.6 KB), i.e. 93-110 KB, so 2 CTAs/SM fit and 16 consumer warps/SM run (4 per SMSP).
- The work unit becomes a 128-row group x 512-k chunk; map, fix-up (partials x2), epilogue and PDL are otherwise the same.
Also folded in for T=1:
- w2's aligned split falls back to stream-K when the aligned grid is under 0.9 x the grid (with 3 experts touched it idles 72 CTAs);
- optionally a per-token last-arriver post-norm inside K2, which removes the combine launch.

**Expected gain.** Model: each warp sustains 0.22-0.32 instructions/clk in every ladder rung, so SM issue rate is about 0.24 x the number of consumer warps.
- 16 warps give about 3.8 instructions/clk, about 44 el/clk of demand at about 22 instructions per 256 el.
- That is memory-capped at 27-30 el/clk (4.0-4.5 TB/s), with issue about 58%, ALU about 50% and smem 55-75% at 4 TB/s.
Expected (inferred):
- production kernels 3.4-3.7 TB/s vs 2.6-3.0 now, +15-25% on the expert kernels;
- decode d1 -3 to -6 us, d16 -12 to -18, d32 -15 to -22, v1 -12 to -16;
- headline about 1.51 from S1 alone (x1.05), plus its share inside D1/D2;
- moe decode16/32 and verify1 each about -12%.

**Effort.** Low to medium: about 1.5-2 days, by templating the CTA shape on the v1/v2/v3 bodies.

**Risks.** - D=2 halves the ring depth per CTA, but bytes in flight stay about 72 KB/SM, the same as the ladder's D3 occ2 rung that reached 4.56 TB/s on the pure stream.
- The slowest of 8 warps gates every stage release.
- v3's 9.4 KB map plus D=2 with 16 rows is about 1 KB over the 2-CTA limit; shrink the map or use a swizzled XROW of 1,024 B.

**Reuse.** ea_kernels.cuh / ea2 / ea3 bodies nearly verbatim; the ladder's PATH 0 for the probe rung.

### O5: S2 (T<=32, alternative or add-on): more ILP or occupancy without the 8-warp CTA

**Design.** (a) Two row tiles per warp: each warp owns 32 rows and loads each B fragment once for both tiles. That halves ldmatrix per HMMA and gives two independent dequant -> HMMA chains per k16 step; it needs a new 32-row lossless record.
(b) Three CTAs/SM: keep the 160-thread CTA at D=2 and move the work map to global memory (built once by K1, read by K2), about 56-60 KB per CTA, for 12 consumer warps/SM.
(c) The zero-layout-change sub-variant: register-A double buffering for mma.sync, dequantizing k16 step j+1 before issuing step j's MMAs.

**Expected gain.** Inferred: (a) about 1.5x per-warp ILP, about 3.3-3.6 TB/s; (b) 12 warps x 0.24 = 2.9 instructions/clk, about 33 el/clk of demand, about 3.4-3.8 TB/s; (c) +5-10%. Comparable to S1, but S1 is preferred because it also builds D1/D2's cold half.

**Effort.** Low to medium: 1-2 days each.

**Risks.** (a) needs a new repack; (b) the global map adds a K1 -> K2 dependency (K2 already waits on K1 through PDL); (c) its gain depends on what ptxas schedules, which cannot be inspected on the judge.

**Reuse.** Engine A kernels; the existing ladder harness.

## Diagnostics

- Ladder-2 (one moe check; module '[ladder2]', about 45 s compile, start deadline 170 s; 226 MB bank of real repacked w13 records from 12 experts; graph-timed with a read flush, 21 reps x 2 rounds, empty-graph baseline subtracted). Each rung prints t_us, weight TB/s, el/clk/SM, clk per stage per SM, regs/spill/smem/occ.
- P0, ceilings: R1 ring D3 occ2 (repeats 4.5 TB/s). An L2->SM probe streams a 16 MB L2-resident buffer with 1-D bulk copies at 264 CTAs, which bounds the activation re-fetch budget at T=256.
- P1, mma.sync tensor rate: constant A and B in registers, NT in {1,4,8,16} independent m16n8k16 accumulators, at 8 and 16 warps/SM. Reports HMMA/clk/SM; the ladder's current lower bound is 0.31 (R3 nt4). This decides how wide D1's passes can go.
- P2, wgmma RS: constant register A, B in SWIZZLE_128B smem via make_desc, m64nNk16 with N in {16,64,128,256}, wait_group<1> with 2 A buffers, 1/2/4 warpgroups per SM. Reports clk per wgmma per SM against the N/2 peak, and tests whether eb_kernels.cuh:7-10's '64-98 clk whatever N' is an intrinsic floor or a serialization artefact.
- P3, wgmma SS: A in bf16 in smem, same N values. P3 vs P2 decides whether D3 is viable.
- C1: Engine A consume(), 4 warps, D3/D4 occ2, N=8/16 (repeats R4's 3.40 / 2.84).
- C2 (S1): 8 consumer warps, 2-record 36 KB stages, D=2, occ2, N=8/16.
- C3 (S2b): 3 CTAs/SM, D=2, N=8.
- C4 (D1): 8 warps, 128-row x 256-k stages, N=32 (4 tiles) and N=64 (8 tiles).
- C5 (D2 hot cell): wgmma RS BM128, N=64/128, on 512-k records with a B sub-ring, run alone. Then the same hot units are mixed 1:3 by cost with C2 cold units inside one launch (hot and cold warpgroups in the same CTA); the overlap factor is (t_hot + t_cold) / t_mix.
- C6 (D3 cell): 8 dequant warps write bf16 A via stmatrix, then wgmma SS at N=16/64/128.
- VERDICT line:
- Build S1 if C2 >= 1.1 x C1.
- Pick D1 if P1 >= 0.7 HMMA/clk/SM and C4(N=32) >= 3.0 TB/s-equivalent.
- Pick D2 if P2(N=128) <= 1.2 x 64 clk and the C5 mix overlap >= 1.25.
- Pick D3 only if P3 <= 0.6 x P2 at N <= 64 and C6 >= max(C4, C5).
- Also print the L2->SM ceiling, and dq_exact as a regression guard.
- Gate fidelity fix (do it first; costs one decode check):
- Replace the isotropic stream in _cb_data (rewrite/src/muse_decode_layer.py:6184) with the judge's formula: reference.py:274-284 with the STREAM_SEED basis. Compute it in the submission, without importing reference.py.
- Print routes, distinct, max and hot rows over 8 draws at T = 16 / 32 / 128 / 256; they should land near distinct 7.5 / 9.0 / 18.4 / 21.3.
- Re-run the existing selectors on those inputs: v2 vs v3 at T=16/32, hybrid H at T=256.
- In moe, add _eb_judge_draw (eb_moe.py:44-55) at p10, p50 and p90 to the v3 T=128/256 gate, which currently uses only zipf and random (8589aad509334dff.profile.log:59-66).
- Reading profiles: kernels launched with PDL report GPU time that includes their wait, e.g. decode v16 ea3_w2 at 159.1 us against moe's 101.3 us. Compare engines on kernel-only graph timings or whole-layer gates, not on profile rows.
- Critical files: rewrite/lib/cuda/ea3_kernels.cuh; rewrite/lib/cuda/ea_kernels.cuh; rewrite/lib/cuda/eb_kernels.cuh (and git 66b6f501 for v1); rewrite/lib/cuda/ea_ladder.cuh + rewrite/lib/ea_ladder.py; rewrite/lib/ea_core.py (_ea_pack/_ea_repack); rewrite/lib/hotcold_decode.py; rewrite/src/muse_decode_layer.py:6157-6199. All paths are under /home/user/git/musespark/.worktrees/decode-first-rewrite/.
- Rules check per design:
- All designs keep NVFP4 codes and scale bytes as lossless permutations, fp32 accumulation (mma.sync, wgmma and stream-K partials), and bf16 x and h between the GEMMs (keep eak::silu_mul's rounding of gate and up to bf16).
- Work maps, hot/cold classes and N-classes are built on the device per call: graph-capturable and never cached across calls. Any host-side branch on hot-expert counts would break the rule against branching on input values.
- D3's bf16 A tiles may exist only in smem. Writing dequantized weights to global memory, even per call, risks the 'NVFP4 stays NVFP4' rule and doubles HBM traffic.
- Never use FP8 wgmma for hot experts: activations must stay bf16.
- Fusing K1+K2 into one kernel would need a grid-wide wait (a spin), so keep two kernels linked by PDL.
- A second weight layout is allowed only as another lossless permutation (+906 MB and extra prepare time); prefer replacing the layout.
- Judge-faithful gate generators must be reimplemented formulas used only in prepare; never import reference.py.
- Watch prepare's 300 s limit and the 15-minute run budget: each new module costs about 40 s to compile, and decode prepare is already at about 68 s.
- Final review reruns nearby shapes: T=129..255 must map cleanly onto the 128/256 classes, and moe bs 257 stays on master.

## Open questions

- What is the sustained mma.sync m16n8k16 bf16 HMMA/clk/SM on the judge's H200? The ladder only bounds it from below (0.31 with dequant in R3 nt4). The marginal cost from nt1 to nt4 (about 1 SM clock per extra HMMA) hints it may be close to tensor peak. This decides D1's maximum pass width.
- Is the '64-98 SM clocks per m64xk16 wgmma whatever N <= 128' (eb_kernels.cuh:7-10) intrinsic to RS mode, or the product of designs that serialized dequant and wgmma? D2's hot path at N=64 depends on it.
- What is the practical L2->SM ceiling? v3 already pushed about 4.9 TB/s at moe T=256 random, and the activation re-fetch is n_local x 1.5 MB per call at BM=64.
- What router_load does the judge use for decode T=128/256? Only router_distinct can be backed out of the SoL. The numpy stream model gives about 180 / 400 local routes and max 33-66 / 64-123, but it is a model: confirm with the faithful _cb_data.
- Does D=2 with 36 KB stages at 2 CTAs/SM keep the ring at 4+ TB/s, and does the slowest of 8 warps gating each stage release cost more than the extra warps gain?
- How much do hot (tensor-bound) and cold (ALU-bound) work actually overlap when they share a CTA (C5 mix rung)? D2's 170-190 us estimates assume at least 1.25x.
- What is the true exclusive time of decode T=128 ea3_w2? The 159.1 us profile time includes a PDL wait on K1's tail; about 100 us is inferred from the moe T=128 analogue.
