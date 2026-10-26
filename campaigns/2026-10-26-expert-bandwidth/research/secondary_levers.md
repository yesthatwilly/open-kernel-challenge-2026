# Secondary decode levers: attention, dense, glue (read-only Explore agent, Opus, 2026-10-26)

Verbatim structured output of workflow run wf_7532bad1-294 (next-campaign-research). Do not edit; corrections go in ../FINDINGS.md.

## Summary

Answer: none of the secondary levers matches the expert engine on its own. Your model puts experts at T=128/256 x0.5 at x1.145 and experts at T<=32 x0.7 at x1.094. My ranking is computed with the judge's formula on the 0d84f58ca33b424b rows (base 1.4422). Each range below is a low/high estimate of microseconds saved per case.

1. **Own-kernel T<=32 chain (the PLAN's W4 widened): x1.047-1.106.** This is the biggest secondary lever, about the size of the T<=32 expert scenario. It has four parts:
   - glue folds: R, combine and both boundaries, x1.020-1.049
   - PDL static-weight prefetch, which needs CUDA versions of today's Triton qkv/o/sliding kernels: x1.010-1.020
   - a rewrite of P (the pooler+router kernel, dl_pooler_router, which falls from 3.54 to 2.75 TB/s as T grows): x1.009-1.016
   - qkv/o/unpool tuning: x1.005-1.011
   Two S-effort quick wins are worth about x1.008-1.016 together:
   - R still builds a Marlin align that Engine A ignores.
   - ea_combine does one L2 round trip and one __syncthreads per slot, serially.
2. **Full-layer attention rewrites: x1.014-1.032 all-in.**
   - verify at B=16/32: x1.0065-1.014
   - decode at B=16/32: x1.005-1.012
   - B=1 ports: x1.003-1.005
   Even every full row at the 4.8 TB/s speed of light only gives x1.061, because full rows weigh 16 of 62.
3. **P rewrite alone: x1.009-1.016** (already counted in 1).
4. **Dense m=128/256 custom GEMM (W5): x1.0085-1.012.** That is -11..-16 us at T=128 and -16..-22 us at T=256 per case. Main-stream dense GEMMs run 70-72 / 88-90 us, against a bf16 speed of light of 41.6 / 49.1 us.
5. **Sliding attention: x1.005-1.011.**
6. **T>=128 glue: x1.005-1.010.**

All secondary levers together come to x1.08-1.19.

**Attention board versus decode.** At B>=16 the attention board runs exactly decode's kernels:
- decode full: sk 107.2 vs 107.6 us (B=16) and 225.4 vs 225.5 us (B=32)
- verify full: FA3 152.7 vs 151.6 us (B=16) and 289.2 vs 287.7 us (B=32)

So there is nothing to port there. Only two B=1 kernels differ:
- d1s: sliding_cluster 9.7 us vs our Triton 13.1 us
- d1f: c320 11.0+4.4 us vs sk two-level 20.3 us

The board's own 1.535 comes from the same 3.0 TB/s decode and 2.1 TB/s FA3 verify kernels. Working backwards from the board scores (inferred, not measured), a 2.37 (rank 50) needs full rows at about 3.5 TB/s, and the leader's ~3.0 needs about 4.4 TB/s plus sliding rows at about 0.5x of ours.

**The "do not try" verdict should be reopened, at medium priority.** Its premises are weaker than they look:
- "sk is at 1.75x its KV floor" is measured against 4.8 TB/s. Against the measured gather-only ceiling (3.97 TB/s at B=16, about 3.77 at B=32) sk is 1.31x: 107.6 vs 81.9 us and 225.5 vs 172.5 us.
- "Every verify kernel lost to FA3" is true, but the losses are diagnosed and fixable rather than hardware limits:
  - The s6 diag shows the ticket epilogue alone adds about 107 us at B=16 (188.7 vs 81.9 us gather-only), and unoverlapped compute adds about 97 us more.
  - ws-s1 died on producer width: its single-warpgroup gather reached only 2.28 TB/s.
  - mma.sync verify kernels are compute-bound because the 64 query rows per kv head serialise the math.

Glue real costs at T<=32 per case:
- R: 6.5-7.1 us (about 3.5-4.5 us of it on the critical path)
- the two boundaries: 3.3-4.1 us each
- combine: 2.5-4.5 us of real work. Its profile rows of 22-41 us are mostly the PDL wait.
- P: 21-27.5 us against a 19.4 us floor

Profile GPU-busy versus median varies by ±20 us in both directions. The profile is a separate replay and fresh routings spread the replays by ±10%, so that gap does not measure launch gaps.

## Findings

### F1

At B>=16 the attention board runs the same full-layer kernels as decode, so there is nothing to port. Only the two B=1 kernels differ.

**Evidence.** Decode run 0d84f58ca33b424b.profile.txt against attention run a78f29aeab9b47cb.profile.txt (us):
- d16f: sk 107.6 + pre 1.7 (lines 74, 84) vs sk 107.2 + pre 2.0 (lines 44-45)
- d32f: 225.5 + 1.8 (lines 108, 118) vs 225.4 + 2.0 (lines 56-57)
- v16f: FA3 151.6 + combine 6.5 (lines 182, 187) vs 152.7 + 6.1 + 3.3 + 2.7 + 2.5 (lines 90-94)
- v32f: 287.7 + 6.9 (lines 218, 229) vs 289.2 + 7.1 + 4.1 + 3.8 + 3.5 (lines 112-116)
- d32s: skw 14.6 + 2.5 vs sk-window 13.9 + 2.2
- v1s: skw 10.6 + 2.2 vs sk PSEUDO 9.9 + 1.6
- d16s: the same _sliding_fused kernel, 14.6 vs 12.5
- d1s: our Triton _sliding_fused 13.1 (decode line 31) vs per-kvhead sliding_cluster_kernel 9.7 (attention line 27, shape 11.05)
- d1f: our sk two-level 20.3, with pre-attention fused into dl_qkv 13.1 (decode lines 44, 47), vs splitkv_attn<320> 11.0 + combine<320> 4.4 + pre 1.4 (attention lines 32-34); the best c320 chain 8713665f is 11.4 + 4.2, shape 17.4 (attention FINDINGS.md:12)

**Measured or inferred.** measured (judge profiles)

### F2

The same FA3 call runs 4-5 us slower inside decode than on the attention board for sliding verify and full verify_1. Full verify at B=16/32 shows no difference.

**Evidence.** FA3 main kernel, decode vs attention board (us):
- v1f: 19.0 (decode 0d84 line 148) vs 15.0 (a78f29ae line 68)
- v16s: 15.1 (line 166) vs 10.7 (line 79)
- v32s: 19.8 (line 205) vs 14.5 (line 101)
The whole attention path (qkv GEMM included), timed in prepare (69db8aeb profile.log:61-77):
- full decode B=16 / B=32: 121.76 / 230.02
- full verify B=16 / B=32: 177.50 / 318.53
- full verify B=1: 44.10
- sliding verify B=16 / B=32: 38.98 / 48.80

**Measured or inferred.** measured; cause unknown (L2 state or clock suspected, unverified)

### F3

The 'do not try full-layer attention' verdict rests on two premises that do not hold up. The KV gather itself runs at about 4 TB/s, so sk sits 1.31x above the measured gather ceiling, not 1.75x. The verify kernels lost because of a fixable epilogue and unoverlapped compute.

**Evidence.** The verdict: PLAN.md:132, FINDINGS.md:181, decode_layer_read.md:248.
Gather-only measurements from the s6 run (muse_attention_decode/hypotheses/archive/full-verify-wgmma-s6-split-ring-2cta/4899ab49d74f4df7.log):
- line 60: all-thread cp.async, 4x16 KB stages, 2 CTAs/SM, no epilogue: 81.9 us = 3.97 TB/s on 325 MB at B=16
- line 84 (fit): B=32 172.5 us, about 3.77 TB/s
- line 59: the same gather with the ticket epilogue kept: 188.7 us; the fit puts its fixed cost at a = 174.5 us (line 83)
- line 65: full kernels at B=16: 285.7 / 320.1 / 346.3 us
- lines 10-21: gates 0.46-0.57x of FA3
The decode sk, by contrast:
- no ticket: 106.9 us (3.07 TB/s) at B=16 and 217.7 us at B=32
- with ticket: 113.5 / 221.5 us (prepare-time diag, full-decode1-sk-pdl-from-preattn/c0f1945a108c4aa2.log:128-141)
- in the judge: 107.6 / 225.5 us
The 1.75x figure is measured against 4.8 TB/s. Against the measured gather it is 107.6/81.9 = 1.31x and 225.5/172.5 = 1.31x.

**Measured or inferred.** measured prepare-time diag (attention board, older flush); conclusion inferred

### F4

Each earlier attempt failed for a specific, identifiable reason: correctness bugs, a serial per-tile chain, a costly merge epilogue, too narrow a producer, or mma.sync running out of compute. FP8 QK MMA alone was neutral.

**Evidence.** - wgmma verify s3/s4: the Q register fragment was carried across tiles wrongly, so multi-tile segments were wrong (REWRITE_READ_2026-10-25.md:394).
- s5: correct but slow: 288.7 vs FA3 164.4 us at B=16 and 429.5 vs 302.4 at B=32, with a serial S -> transpose -> softmax -> PV chain and a CTA-wide rendezvous (DIAGNOSIS.md:35-38).
- s6 split ring: 0.46-0.57x of FA3 (4899ab49 log:10-21).
- ws-s1 producer split: ws64 0.604x, ws128 0.531x; its gather alone ran at only 2.28 / 1.81 TB/s (e701c51d7b944a13.log:97-102).
- One-launch bulk-gather kernel: per-row cp.async.bulk 0.49 TB/s; one producer warp 1.77 TB/s (REWRITE_READ:172-175).
- mma.sync verify A8: 1.44 TB/s, scaling linearly with bytes (REWRITE_READ:60-74).
- Sliding wgmma window: 43.8 vs 24.4, 45.2 vs 31.6 and 40.9 vs 15.1 us (cd01da3909cc4f4c.log:65,87,109).
- FP8 m16n8k32 QK on the decode stage-1 kernel: neutral, job b939681d3ab549d7 (done/neutral/2026-10-24T165302Z-full-decode-cuda-fp8-qk-mma.md). Removing K cvt alone does not close the sk gap.

**Measured or inferred.** measured

### F5

The decode sk kernel is limited by its consumer, not by the gather. Every tile ends in a CTA-wide sync and a cp.async wait. Half the MMA rows are zero padding. Each lane converts 32 FP8 values per tile.

**Evidence.** rewrite/src/muse_decode_layer.py:
- 4085-4092: 256 threads, 64-token tiles, 16 KB stages
- 4233-4234: 4 stages at 2 CTAs/SM
- 4358-4359: cp_wait plus __syncthreads per tile
- 4399: mma16816(s, qa0, 0u, qa1, 0u, ...), i.e. rows 8-15 are zero
- 4398 and 4435-4438: cvt.f16x2.e4m3x2 for both K and V
Back-of-envelope at 1.755 GHz: 3.0 TB/s means about 1260 clk per 16 KB tile per SM, against a budget of about 945 at 4 TB/s. The cvt pipe alone takes about 512 clk per tile per SM.

**Measured or inferred.** code read (measured); clock budget inferred

### F6

Full rows are worth little to the headline because they weigh 16 of 62 per load. A microsecond on a T<=8 sliding row is worth 4-16x one on a T=128/256 full row.

**Evidence.** Headline percent per us saved (sliding row / full row), from the 0d84 rows:
- d1: 0.116 / 0.040
- d16: 0.059 / 0.020
- d32: 0.044 / 0.015
- v1: 0.079 / 0.027
- v16: 0.028 / 0.010
- v32: 0.020 / 0.007
Every full-layer attention row at the 4.8 TB/s speed of light (d16f -44, d32f -99, v16f -103, v32f -179, d1f -12, v1f -22 us) gives only x1.061.

**Measured or inferred.** computed with the judge formula

### F7

Working backwards from the attention board's scores, the top teams must run full-layer kernels at about 3.5-4.4 TB/s, so such kernels are buildable.

**Evidence.** Recomputed from the a78f29ae rows (ours 1.5347):
- full rows x0.6 with sliding unchanged gives only 2.08
- full B>=16 at 3.5 TB/s with sliding x0.7 gives 2.31; with sliding x0.5, 2.60
- full at 4.4 TB/s with sliding x0.5 gives 2.93
r50 is 2.37 and the leader 2.96-3.03.

**Measured or inferred.** inferred from board math; no per-shape data exists for other teams (the leaderboard JSON is headline-only)

### F8

The dense GEMMs at T=128/256 run 1.7-1.8x above the bf16 speed of light and 5-9 us above the bf16 board's own rows for the same kernels. The router runs on a side stream.

**Evidence.** Decode in-profile, T=128 (v16s/v16f):
- qkv: nvjet 96x64 splitK TNN 15.1-15.4, plus splitKreduce (4.0 over 2 calls)
- o + unpooler: nvjet 128x64 TNT, 29.4-29.9 over 2 calls
- pooler: nvjet 64x64 TNT 23.1-25.2
- router hi/lo: tss splitK 10.6 over 2 calls, side stream
Decode in-profile, T=256 (v32s/v32f):
- qkv: 96x64 TNN 22.6-23.1
- o + unpooler: 128x128 TNT 33.8-34.3
- pooler: 128x64 31.1-32.4
- router: 13.0-14.5
Source lines: 0d84 profile.txt 163-172, 183-186, 202-206, 222-225.
Main-stream total (decode / bf16 board judge rows / bf16 speed of light / custom estimate):
- T=128: about 70-72 / 65.0 / 41.6 / 54.5-57.5
- T=256: about 88-90 / 81.1 / 49.1 / 67-71
Board rows and custom estimates: dense_gemm_forensics.md:104-113 and 246-257.
What the forensics recommends:
- one M-tile, x multicast, static split-K <=132 CTAs, last-arriver reduce, no memset node, tile-contiguous repack (lines 206-244)
- a one-check Nondeterministic CUTLASS diagnostic first (lines 261-264)

**Measured or inferred.** measured rows; o/unpooler split of the 2-call row inferred

### F9

Glue at T<=32 has a real cost of roughly 13-19 us per case on top of P. Several pieces are cheap to remove.

**Evidence.** 0d84 profile rows, T<=32 cases, in us:
- R: 6.5-7.1. The P+R pair takes 26.69/27.71/30.37 at T<=8/16/32 (69db8aeb log:63-65), so R adds about 3.5-4.5 on the critical path.
- R's last block builds a sorted_ids/block_experts align (muse_decode_layer.py:655-708) that Engine A never reads ('R's align goes unused', :8711-8712).
- Boundaries: SGLang norm_residual 3.3-4.1 each. They take no part in PDL, and an own standalone boundary lost 3.9 vs 2.8 (LESSONS.md:162).
- Combine: its profile rows read 22-41 at T>=8, but the moe board, where it runs alone, shows 3.1 (T=1), 3.5 (T=8), about 3.6 (T=16) and about 4.3 (T=32) once the PDL overlap is removed (f0fc89031659499c profile). ea_combine does one L2 round trip plus a __syncthreads per slot, serially (ea_combine.cuh:47-74).
- P: 21.3/22.2/24.1/27.5 at T=1/8/16/32, i.e. 3.54 to 2.75 TB/s on 75.5 MB. Its mma.sync consumer works on 1, 2 or 4 n8 token tiles, and its 216 CTAs leave SM load uneven at 2 CTAs/SM (muse_decode_layer.py:278-293).

**Measured or inferred.** measured; critical-path shares inferred

### F10

Comparing profile GPU-busy with the case median says nothing reliable about launch gaps. The gap swings ±20 us both ways because each replay draws fresh inputs and routing.

**Evidence.** Median vs GPU busy, us (0d84 profile.json.gz):
- d1s 105.0 vs 117.4
- d16s 185.1 vs 207.0
- d16f 287.4 vs 268.3
- d32s 226.4 vs 224.2
- d32f 431.8 vs 444.3
- v1s 152.3 vs 145.8
- v1f 172.8 vs 181.5
- v16s 411.1 vs 430.4
- v16f 560.8 vs 556.8
- v32s 546.5 vs 550.8
- v32f 816.1 vs 827.5
Replay spreads p10-p90 are wide: d16f 274-306, d32s 202-254, v1s 139-168.

**Measured or inferred.** measured

### F11

PDL chains of our own kernels pay off, but not every link does. Triton kernels (swap-AB qkv/o, _sliding_fused) still break the chain at T<=32.

**Evidence.** Positive:
- own chain Q -> sk -> O: 57.25 -> 48.96 us in prepare, and full d1 -7.8 us in the judge (FINDINGS.md:167)
- small-t dense chain with PDL: 1.164 -> 1.228x (LESSONS.md:38)
- c320 chain: 0.791 of master (LESSONS.md:191)
Negative:
- R -> K1 PDL neutral (LOG.md:137)
- L2 prefetch of K2's w2 lost (LOG.md:154)
- side-stream L2 prefetch +21% at d16s (FINDINGS.md:168)
_bf16_swapab_kernel is a Triton kernel (dense_swapab.py:37).

**Measured or inferred.** measured

## Options

### O1: W4+ own-kernel T<=32 chain: glue folds, PDL weight prefetch, P rewrite, qkv/o tuning

**Design.** Four parts, ending with every link a PDL link:
(a) R: drop the unused align tail now. At T<=8, fold routing and pre-expert RMSNorm into the w13 (K1) prologue, computed redundantly per CTA.
(b) Combine: fold it into the w2 (K2) epilogue with a per-token last-arriver ticket, or first just issue all 8 slot loads before the reductions.
(c) Fold attn_boundary into a CUDA o-GEMM epilogue (per-token last arriver), and ffn_boundary into dl_unpool's epilogue at T<=8 or a custom unpooler at T=16/32.
(d) Replace the Triton swap-AB qkv/o and the Triton sliding attention with CUDA kernels that trigger launch_dependents early. Prefetch o's 16.8 MB into smem/L2 during latency-bound sliding attention, and extend P's prefetch of its 75.5 MB beyond the 10 MB it takes today.
(e) P: wgmma or swap-AB with tile-contiguous repacked bulk loads and a balanced grid.

**Expected gain.** x1.047-1.106 in total:
- glue folds: x1.020-1.049 (boundaries x1.011-1.019, R x1.004-1.017, combine x1.005-1.011)
- PDL prefetch: x1.010-1.020
- P rewrite: x1.009-1.016
- qkv/o/unpool: x1.005-1.011
Quick wins (R without align plus combine slot prefetch) give x1.008-1.016 for S effort.

**Effort.** L overall: quick wins S; R/combine folds M; boundaries and CUDA qkv/o M-L; P rewrite M

**Risks.** - A standalone own boundary lost before (3.9 vs 2.8 us), so only fused boundaries are worth trying.
- R -> K1 PDL was neutral.
- Fused epilogues add last-arriver tails.
- At T=16/32, a redundant per-CTA norm or routing in K1 costs 35-70 MB of L2 traffic.
- fp32 residual math must match the reference tolerance.

**Reuse.** Existing pieces:
- piece A dl_qkv/dl_o (fused norm/RoPE/FP8 store, PDL)
- dl_unpool's pre-wait weight ring
- P's ticket split-K
- Engine A's last-arriver tickets
- the bf16 bulk row-owner kernel (3.97 TB/s with repack)
- the per-kvhead sliding cluster kernel and c320 (attention branches)

### O2: Full-layer verify kernel for B=16/32 (replacing FA3 + combine + prepare + norm-gate)

**Design.** - Gather: the proven all-thread 16 B cp.async gather at 2 CTAs/SM (512 issuers per SM, 4 or more 16 KB stages), with stages released per warpgroup through mbarriers instead of per-tile __syncthreads.
- QK: one consumer warpgroup per kv head (64 rows = 8 tokens x 8 heads) on wgmma m64nNk32 FP8, with K in its natural layout.
- PV: wgmma RS with V converted to f16 (MN-major, so no transpose), or FP8 with an in-smem V transpose.
- Overlap: issue S(n+1) before softmax(n), with wait_group 1.
- Merge: cheap. Use a PDL-launched combine, as FA3 does (6.5 us), or a two-level ticket with unrolled slot loads. The s6 merge is what cost about 107 us.
- Budget at B=16: about 82 us of gather, 40-44 us of ex2, about 33 us of softmax FMAs and about 21 us of FP8 MMA. They fit only if overlapped.

**Expected gain.** v16f about 162-167 -> 96-128 us; v32f about 299-306 -> 190-254 us. x1.0065-1.014.

**Effort.** XL (the attention board has six failed or slow iterations of this line)

**Risks.** - A fourth large wgmma attempt could still lose to FA3; gate it against the whole FA3 chain.
- Rules question on P in FP8.
- Register and smem budget at 2 CTAs/SM.

**Reuse.** - s5/s6 kernels: correct, with 19 bisect cases passing
- gather-only diag variants
- S3 paired pre-attention
- the sk epilogue (RMSNorm x sigmoid(gate))
Build and prove it on the attention board first, where it is worth +5-6.5% on that board by the attention DIAGNOSIS estimate.

### O3: Full-layer decode B=16/32: sk consumer rewrite

**Design.** - Keep the 2 CTAs/SM all-thread cp.async ring.
- Release stages per warp through mbarriers, with no per-tile CTA sync.
- QK: swap-AB FP8 m16n8k32 (tokens on M, 8 queries on N): no K cvt and no padded rows.
- PV: exact integer e4m3 -> f16 bit shift with 2^-8 folded into v_scale, or cvt.
- Each warp owns a token slice through all tiles and merges in smem at segment end.

**Expected gain.** d16f 107.6 -> 85-98 us, d32f 225.5 -> 180-205 us. x1.005-1.012.

**Effort.** L

**Risks.** - FP8 QK alone was neutral, so the win depends on removing the per-tile sync.
- Bandwidth for the ~650 MB random footprint at B=32 is unproven above 3.77 TB/s.

**Reuse.** The sk kernel (muse_decode_layer.py:4233-4543), its ticket and two-level merge, and the gather-only diag as the ceiling probe.

### O4: B=1 attention ports (d1s per-kvhead cluster, d1f c320 in piece A, v1f grid-SM two-level verify)

**Design.** - d1s: port sliding_cluster_kernel<true> (9.7 us on the attention board) in place of _sliding_fused (13.1 us in decode).
- d1f: put c320 inside piece A, chained by PDL from dl_qkv.
- v1f: a grid-SM stream-K two-level kernel for q_len 8 (64 rows per kv head) in place of FA3 + combine + prepare (about 31-32 us).

**Expected gain.** x1.0035-1.009: d1s -0.7..-3.4, d1f -2..-4, v1f -7..-13 us

**Effort.** d1s/d1f: S-M (the code exists on attention branches); v1f: L

**Risks.** In decode the d1s Triton kernel runs 2.7 us slower than on the attention board, for reasons unknown, so the port's gain there is uncertain.

**Reuse.** Branches merge-perkvhead-c320-sv32-tiled and combo-b1-chain-c320-fused-pre-pdl; the sk two-level ticket.

### O5: Dense m=128/256 custom GEMM (W5)

**Design.** - one M-tile x 128 weight rows, BK 64
- TMA with x multicast across a 2-CTA cluster
- static split-K <=132 CTAs
- deterministic last-arriver fp32 reduce, no memset node
- tile-contiguous weight repack
Start with the one-check CUTLASS Nondeterministic diagnostic. Also fold the pre-expert RMSNorm and measure the router's side-stream join slack.

**Expected gain.** -11..-16 us per case at T=128 and -16..-22 at T=256: x1.0085-1.012. bf16 board +3.5-4.5%.

**Effort.** L

**Risks.** - In decode, qkv m=256 runs 22.8 vs 19.5 us standalone (clocks or L2), so gains may shrink.
- Spin reductions need grid <=132 and no early launch_dependents.

**Reuse.** CUTLASS 4.5 builds on the judge (49 s); bf16_proj is the testbed.

### O6: Sliding verify B=16/32 kernel plus fused pre-attention at v1s/d32s

**Design.** - One CUDA kernel for v16s/v32s (fused pre-attention, window gather, norm-gate) in place of FA3 + combine + prepare + pre + norm-gate (about 25 / 30 us in decode).
- Fold skw_pre into skw_attn at v1s/d32s.

**Expected gain.** x1.004-1.007 (v1s/d32s -1.5..-2.5, v16s -4..-8, v32s -5..-10 us)

**Effort.** L

**Risks.** The earlier sliding wgmma window kernel lost 0.35-0.70x, and mma.sync pseudo-decode lost at B=16/32.

**Reuse.** SKW PSEUDO kernels and the S3 pre-attention.

### O7: T>=128 glue

**Design.** - Fold pre-expert RMSNorm (vectorized_layer_norm 6.0 us) into Engine A v3's K1 prologue or the pooler epilogue.
- Concatenate router and pooler GEMMs.
- Trim the 9-10 hidden small kernels (24-37 us per case, partly on the side stream).
- Partly fold the boundaries (4.6-8.8 us each).

**Expected gain.** -8..-15 us per case: x1.005-1.010

**Effort.** M

**Risks.** Unknown share on the critical path; the router join slack has never been measured.

**Reuse.** The R-style norm and routing code, and the hot/cold align kernel.

## Diagnostics

- Profile kernel rows include PDL wait time: ea_combine (22-41 us at T>=8), dl_unpool at v1f (20.4 us), w13 at d1s. Real costs came from the moe board (f0fc89031659499c), where the combine runs alone, and from prepare pair timings.
- The attention-board diagnostics (s6, ws-s1, sk-diag) are prepare-time numbers taken before W0 and probably with the zero_() write flush. That inflates large streams by up to about 20%, so the gather ceilings may be conservative.
- All multipliers come from one run's medians (0d84f58ca33b424b, base 1.4422) and per-case microsecond savings applied with the judge formula. Opportunities that touch the same load are roughly, not exactly, multiplicative.
- The decode profile shows only the top 12 kernels per case. The hidden rows at T=128/256 (24-37 us) were obtained by subtraction.

## Open questions

- May P be quantized to FP8 e4m3 for the PV product? FA3's FP8 path, which is the SGLang baseline, does it, but TASK.md only names queries. This decides whether FP8 PV with a V transpose or f16 PV with V conversion is the right verify design.
- Why does the same FA3 call run 4-5 us slower in decode than on the attention board (v1f 19.0 vs 15.0, v16s 15.1 vs 10.7, v32s 19.8 vs 14.5 us)? The same question applies to the 2.7 us gap on d1s's Triton kernel (13.1 vs 10.4 us).
- How much of R, the two boundaries and the combine is really on the critical path? Measure with a prepare-time whole-layer A/B using no-op variants, for example R without the align tail and combine with prefetched slots.
- What is the root cause of the s6 ticket epilogue's ~107-175 us fixed cost? Suspect: merge_head reading ~16 slots serially with dependent L2 loads (s6 branch submission.py ~6034-6084).
- Does 2 CTAs/SM all-thread cp.async with per-warp mbarrier stage release close sk's gap to the gather-only ceiling (106.9 vs 81.9 us at B=16)? One prepare-time variant answers it.
- Is the T>=128 side-stream router (10.6-14.5 us) on the critical path?
