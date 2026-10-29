# muse_decode_layer findings

Regenerated 2026-10-29T213652Z by `churn.py index`. Read this before writing a hypothesis for muse_decode_layer.

## Claims (from CLAIMS.md, newest per mechanism; tag meanings at the top of CLAIMS.md)

### lead

- [lead] engineA-v4x-s1 decode: Engine A v4 stream-K fix-up D_dec=+5.12 us/kernel (K1 +5.58, K2 +4.66), T=128 cold-half onset E2c +4.35 on the H200; v4x S1 BUILD | rows: load=decode_16 load=decode_32 load=verify_1 load=verify_16 load=verify_32 | evidence: TC-B check b47f8f7382d04d47 [pcd-dfx] lines (phasec/dec-fx-1028 b0a3762e); the run failed a verify_32 routing recheck at repetition 8 (cause under investigation) | board: 1.6551 | blob: b04cdf5c | reopen: - | 2026-10-28 campaign (re-measure: measured on 1.6551, best now 1.6904)
- [lead] dq2 A3 dequant in the wgmma hot half (wg_kernel dq_stage): 35.9 -> ~27.9 instr per k16 step per warp, bit-exact; shortens the serial dequant->wgmma->wait chain | rows: load=verify_16, load=verify_32 | evidence: H100 gpunode06 hh probe (takemyjob hopper/hh-1029 e942c40; research/h100-hothalf-1029.md): hot-half pair A3/master x0.902-0.923 T=256, x0.894-0.898 T=128, moe judge draws x0.866/0.890; est decode +0.7%, moe +0.8% headline (not judge-checked); also w2 1.45 waves (tail 22-27% of T=256 pair, upper bound) and 576 token columns computed for 386 routes | board: 1.690 | blob: 315119e2 | reopen: stack with a w2 wave/tail fix and per-expert N-tile (padding) fix to >= 2.5%; judge check of the stack | 2026-10-29 h100

### shelf

- [shelf] port of attention's fused Triton sliding batched decode (sliding-fused-batched-decode-port): within-ci but hit 5/5 predicted rows (listed Do-not-try at FINDINGS:172 in error) | rows: layer=sliding load=decode_16, layer=sliding load=decode_32 | evidence: b01fd2df41a040b3 1.2248 vs 1.2283 (hypotheses/FINDINGS-MODEL-REVIEW-2026-10-28.md:56); sliding B=17..32 has since moved to the stream-K window kernel (DECISIONS:426) | board: 1.2283 | blob: f99bbe8e | reopen: re-measure B=2..16 against master's sliding route before stacking | 2026-10-24 churn (re-measure: measured on 1.2283, best now 1.6904)
- [shelf] wg2 OCC2 B256 decode port (wg2-occ2-b256-decode-port), est ~0.4%; its Marlin premise is gone and wg2 is refuted above | rows: load=verify_32 | evidence: DECISIONS:432 | board: 1.6551 | blob: - | reopen: - | 2026-10-28 screen (re-measure: measured on 1.6551, best now 1.6904)
- [shelf] v4xd-forkguard: decode side_stream fork on _rt_fork_stream at priority 0 (moe v4x256c guard) | rows: load=decode_* load=verify_* | evidence: check 3390f868d49d4ea3 1.663 (board 1.661), a tie; campaign/eb-v4xd-forkguard 91e09235; fork vs serial T=128 x1.020-1.023, T=256 x1.013-1.014; board-eq distinct bcap 0 | board: 1.661 | blob: 124f158f | reopen: ride on the next decode candidate (insurance against a capture-stream collision) | 2026-10-29 campaign (re-measure: measured on 1.6610, best now 1.6904)
- [shelf] bulk12 in decode's dense GEMMs at T=8..32 (P + unpooler + qkv/o), est +2.8% if all three capture the 25% byte cut, P alone ~1.3% | rows: load=verify_1, load=decode_16, load=decode_32 | evidence: 7cc034f917274439 profile: P 22.0-26.9 us, _bf16_swapab 19.1-22.0 us, unpooler nvjet 18.6-19.5 us; bf16 judge m>=8: b12x qkv m=8 x0.769, b12r pooler m=8 x0.984 (campaign FINDINGS bulk12) | board: 1.6904 | blob: 315119e2 | reopen: a judged m>=8 bulk12 cell at <= 0.85x its raw incumbent (e.g. x multicast per cluster) | 2026-10-29 screen
- [shelf] wh hot half (A3 + exact-N + R4 ring N>=48) in wg_kernel<G,7>, routed by isolated-pair race (hothalf-a3-exactn-ring-wh) | rows: load=verify_16, load=verify_32 | evidence: check 44f303647100481b 1.6794 (board 1.6904; flat): routed T=128/256, pair wh/m .898-.902 / .838-.842, layer fork .973-.976 / .960-.979, exact 16/16; profile verify_32 sliding hot pair 153.5 -> 122.8 us (x0.80), row 427.7 -> 403.2 us; verify_32 full 130.6 -> 124.7; branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh | board: 1.6904 | blob: 315119e2 | reopen: ride as a bit-exact rider on the next decode candidate (with A5b+D+SC per hh3, ~+0.2%) | 2026-10-29 churn

### unresolved

- [unresolved] decode T=1 sliding row 93.8 vs ~96 us: a lucky input draw (2 vs 3 distinct local experts), not a race | rows: layer=sliding load=decode_1 | evidence: race table over 15 judged runs; c1daf58b's d1s paired 1.831 vs 1.760-1.786 in 11 other PERHEAD-era runs (LOG:1503-1505); K9 staircase H100:79-121 | board: 1.5926 | blob: - | reopen: - | 2026-10-28 campaign (re-measure: measured on 1.5926, best now 1.6904)
- [unresolved] b12p PDL race at T=1 (pdl 8:2 across today's runs): neither pin gains | rows: load=decode_1 | evidence: dec-b12p-pdl0 1.6345, pdl1 1.6404 vs board 1.6470 (LOG:2077-2078, 2098) | board: 1.6470 | blob: - | reopen: - | 2026-10-28 campaign (re-measure: measured on 1.6470, best now 1.6904)
- [unresolved] final-rerun routing tail: FP8-q full attention vs the judge's BF16-q reference flips a clear-margin expert at T=256 (KEEP_TOL 0.03), ~1.6%/run (1 of 61 runs; 95% upper ~8%); board 83f0c00e exposed | rows: load=verify_32 | evidence: settled by dec-rt-tail 0635f30d0dc84455: INHERENT (keep slack p0.1 board 0.0179 = FA3 0.0179; fp64-attn FP8-q emu 0.0187; BF16-q 0.0296); q x32 pre-scale no lead (+0.0013 < 0.005); 0 violations in 16384 tokens | board: 1.6551 | blob: b04cdf5c | reopen: only a BF16-q full-attention path at T=256 removes it (speed cost unmeasured) | 2026-10-28 campaign (re-measure: measured on 1.6551, best now 1.6904)

### node-only

- [node-only] decode node judge (DEC-02): the H100 node cannot stand in for the judge on decode as built (V2 and V3 fail on all four bytes: prepare under nsys forces incumbents, T=1 Engine A gate falls to MASTER on 2/4 draws, H100 timing races, norm_residual shim +1.2-1.8 us); transfer: failed | rows: all | evidence: H100:887-906; LOG:1726-1735; DEC-02 runs 234714 / 234716 / 234719 / 234724 (gpunode06) | board: 1.6262 | blob: - | reopen: a new registration (the user's call): forced judge picks via E1 arms, a profiled pass whose prepare is not under nsys, a faster boundary shim | 2026-10-28 h100
- [node-only] decode-layer anatomy on the node matches the judge's shares within a few points (decode 1: moe 38-40%, dense 42-43%, attention 11-12%, small 7%); the dense pooler / router GEMV (~17-22 us judge) is decode 1's biggest block; transfer: held for shares, not per-kernel times (dense x1.23-1.38) | rows: load=decode_1, load=decode_32 | evidence: H100:900-903; LOG:1731-1732 | board: 1.6262 | blob: - | reopen: - | 2026-10-28 h100
- [node-only] K1 w2 prefetch at T=1 (m07): 2-3% slower at k=2 and 3 (+0.7-1.4 us/call); closed for moe and decode; transfer: untested | rows: load=decode_1 | evidence: H100:775-776; LOG:1676-1677 | board: 1.6262 | blob: - | reopen: - | 2026-10-28 h100

### landed

- [landed] Engine A (CUDA W4A16 expert engine, v1 / v2) replaces Marlin for the T<=32 expert block (decode-first campaign) | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: ec719761 (decode scored 1.3746); later rounds c8b77a59, 249f6ef3, 2fc609f3, 2196b906, 3cd54072 | board: 1.3746 | blob: 459835a2 | reopen: - | 2026-10-26 campaign
- [landed] Engine A v4 (8 consumer warps, K1 plain without PDL) at T=2..32 | rows: load=decode_16, load=decode_32, load=verify_1 | evidence: check 5573e05423304324 1.4950, scored dc51d8e8f58e4cb7 1.4956 (LOG:330, 356-357); land 49142841 | board: 1.4956 | blob: e008c521 | reopen: - | 2026-10-26 campaign
- [landed] wgmma-hot / Engine A-cold hybrid (H=-16) at T=128/256, Engine D inert (27ef83df bytes) | rows: load=verify_16, load=verify_32 | evidence: check a1bc1756256b4c91 1.5097, f59cb301 1.5347; scored 2bd373c9e23c4667 1.5218 (LOG:376, 433, 686, 706); land 4adf7b75 | board: 1.5218 | blob: a11e55ef | reopen: - | 2026-10-27 campaign
- [landed] v4c: Engine A v4 with a cost-weighted stream-K split, raced against v4 in every gate (big T +2%) | rows: load=verify_16, load=verify_32, load=decode_16, load=decode_32 | evidence: eacf0722; check 0b085ef420c64ffb 1.5484, scored 42b3e39478474bcd 1.5275 (LOG:708, 720, 740); land 76815f82 | board: 1.5275 | blob: de9ac2d8 | reopen: - | 2026-10-27 campaign
- [landed] b12p: bulk12 P (pooler + router) at T=1 under a read-flush gate, with the _ead_anatomy use-after-free fix | rows: load=decode_1 | evidence: check dc3d5614d5e047c8 (P 19.62 -> 15.34 us, layer x1.040), 7844b6c903794ee7 1.5276; scored ba57a64f66864219 1.5441 (LOG:763-767, 800-818, 879); land 07b71687 | board: 1.5441 | blob: 32e717e7 | reopen: - | 2026-10-27 campaign
- [landed] W3-A SK_FULLLINE (full-line K/V gather) at decode 16 / 32 full + W4 c320 at decode_1 full (TRIG lost its head to head) | rows: layer=full load=decode_16, layer=full load=decode_32, layer=full load=decode_1 | evidence: cc7d142af3fb4308 (d32f x1.077), d08b11fe295f4acf 1.5484 (d1f x1.038, d32f x1.095), scored 7ceab934eea24099 1.5507 (LOG:924-926, 957-959, 976); land d5eeef58 | board: 1.5507 | blob: 66a30065 | reopen: - | 2026-10-27 campaign
- [landed] W6-B T>=128 diet cells (norm, comb, pool, hilo, unpool) | rows: load=verify_16, load=verify_32 | evidence: check 27a008fa 1.5554, scored 4a7a49d0ae82440f 1.5697: v16s x1.063, v16f x1.051, v32s x1.063, v32f x1.031 (LOG:1014-1025); land dca993b1 | board: 1.5697 | blob: 5b2e4078 | reopen: - | 2026-10-27 campaign
- [landed] W6-G stack: PERHEAD sliding decode_1 port + c320t at bar 1.003 + fast silu in Engine A / wg + a T=128 confirm race that can promote H-16+comb with the cells on | rows: layer=sliding load=decode_1, layer=full load=decode_1, load=verify_16 | evidence: fff7d3e7250042f1 (PERHEAD d1s x1.028), f2aece517e8543e1 1.5783, check 41975d1ce6334756 1.5749, scored c1daf58bf192471e 1.5920 (LOG:1041-1049, 1200, 1215-1224); land 3e176f6c | board: 1.5920 | blob: a259ca2f | reopen: - | 2026-10-27 campaign
- [landed] pinned trim: H-16 (+comb) pinned at T=128 and T=256 (each hybrid race decides ~20 us per row; prepare-time margins do not track judged rows) | rows: load=verify_16, load=verify_32 | evidence: ced20a71e99b4798, 8e3b9874d56e4657 1.5722, 5a33db7194c24282 1.5847 (LOG:1349-1353, 1399-1413); scored e1fd3c00c83045dc 1.5926 (LOG:1490-1496); land d6c1a9b1 | board: 1.5926 | blob: dc0b246a | reopen: - | 2026-10-28 campaign
- [landed] pins round 2 (e8f8134b): Engine A v4 at T=2..8, unpooler master at T=9..16, qkv_gate / o swap-AB at T=17..32, pool cell at T=256 (determinism, not score) | rows: load=verify_1, load=decode_16, load=decode_32, load=verify_32 | evidence: 314cca3c51604c23 1.5883, all four pins routed, rows x0.964-1.018 (LOG:1506-1508, 1517-1521); carried to master through eb-s3d -> eb-l3d-s3d (LOG:1597, 622b97bb) | board: 1.5926 | blob: f34ed172 | reopen: - | 2026-10-28 campaign
- [landed] L3 paired scale decode in Engine A's dq2 consumers (bit-identical, -2 ALU instr/step) + Engine A v6 (S3: whole-record ownership, co-resident K2) at T=1 | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: 31e0aad5 1.637 (L3 + v6), 7f9712c4 1.624 (L3 alone): decode 32 sliding x1.07, verify 16 sliding x1.05 (LOG:1767-1769); scored adb9c3436aad4e1d 1.6262 (LOG:1777-1780); land 622b97bb | board: 1.6262 | blob: f34ed172 | reopen: - | 2026-10-28 campaign
- [landed] L4: L3's pair add as an IMAD on the FMA pipe (run-time 16 in inline PTX), bit-identical | rows: load=decode_16, load=decode_32, load=verify_1 | evidence: check 0576191d3d544bf9 1.638 (flat over L3 but 9 of 12 rows faster), scored 3a68058a0f0a4eab 1.6407 (LOG:1800-1812); land f3828162 | board: 1.6407 | blob: df18f12b | reopen: - | 2026-10-28 campaign
- [landed] fc: Engine A v6 T=1 fixed-cost arms (v6pe = consumer-staged B + early combine), raced vs v6, bitwise v6 | rows: load=decode_1 | evidence: check 5f1047f58069453b 1.647 (v6pe 1.074 / 1.072 over v6), scored 9afcccc328974350 1.6428 (LOG:1883-1906); land f8cf2476 | board: 1.6428 | blob: 770edae9 | reopen: - | 2026-10-28 campaign
- [landed] X1 restricted to the v6 family (T=1, on fc); v1-v4 keep L4's code | rows: load=decode_1 | evidence: check cfdf6df3 1.646 (decode 1 x1.010 / x1.013), scored 7352add6920f40e7 1.6470 (LOG:1947-1953); land d0cdde04 | board: 1.6470 | blob: 3b23ccde | reopen: - | 2026-10-28 campaign
- [landed] Engine A v4c pinned at T=9..16 (the race flipped v4 / v4c 6:4; the layer prefers v4c though v4's block is 1.8-2.5% faster in isolation) | rows: load=decode_16 | evidence: check 86edd970 1.6599 ((16,1) rows 162.8 / 250.0 vs 165.1 / 261.8 us with v4 pinned; dec-pin-all 1.6470, dec-pin16-v4 1.6509), scored 83f0c00eb3644e1c 1.6551 (LOG:2088-2090, 2104-2105); land ec3c4dcb | board: 1.6551 | blob: b04cdf5c | reopen: - | 2026-10-28 campaign
- [landed] Engine A v4x decode port: warp-scan pass list, gs in smem, 32-bit split, arm d/s ticket rounds; routed 8 v4xs, 16 v4xd, 32 v4cxs, T=128 cold half v4cxs | rows: load=decode_16, load=decode_32, load=verify_1, load=verify_16, load=verify_32 | evidence: job 31e9a69aac344675 (okc status 31e9a69aac344675), check 1fd2c931cf874a07, master d1fb10ab8a | board: 1.6606 | blob: 124f158f | reopen: - | 2026-10-28 campaign
- [landed] v4xd-stack: v6pe2 two-wave ea6 K2 at T=1 + v4x-routes (each twin qualifies alone, pooled hysteresis; T=128 cold half alone) + priority-0 side-stream fork guard | rows: load=decode_* load=verify_* | evidence: job 7cc034f917274439 (okc status 7cc034f917274439), check 264355099fc74226, master d28b23d729 | board: 1.6904 | blob: 315119e2 | reopen: - | 2026-10-29 campaign

### refuted

- [refuted] pooler / router GEMM retunes on the old dense chain: pooler split-K autotune, kernel P wide tile at T=16/32 | rows: load=decode_16, load=decode_32 | evidence: 2e1a43ab85324776 1.0189, 9ad8e2cf869c4b4f 1.2457 vs 1.2521 (FINDINGS:169, 173) | board: 1.2521 | blob: 930f0e60 | reopen: - | 2026-10-24 churn
- [refuted] single-kernel MoE align at small T; L2 prefetch of o / pooler weights during sliding attention | rows: load=decode_1, load=decode_16, load=decode_32 | evidence: 0a0af8669ffe4cb0 1.0431 vs 1.0621 (neutral), 96ee63e15edb438c 1.1670 vs 1.2283 (regression) (FINDINGS:170, 177) | board: 1.2283 | blob: 7377122b | reopen: - | 2026-10-24 churn
- [refuted] paired-heads pad / unpad in Triton around the output norm-gate | rows: layer=full | evidence: a92b64be4c6547d3 1.1654 vs 1.1665 (FINDINGS:171) | board: 1.1665 | blob: 408bdcba | reopen: - | 2026-10-24 churn
- [refuted] routing top-k + PDL into the unpooler at T<=8 (stack-route-topk-pdl-unpool-nf1) | rows: load=decode_1, load=verify_1 | evidence: 87211e1362b64c4c 1.2671 vs 1.2680 (FINDINGS:174) | board: 1.2680 | blob: 20289b4c | reopen: - | 2026-10-25 churn
- [refuted] v5 = D1: 32-route mma.sync passes on 256-k records (wide mma.sync, N >= 32 per A fragment) | rows: load=decode_16, load=decode_32, load=verify_1 | evidence: moe e403a42a kernel-only 2.39-2.90 vs v4 2.92-3.21 TB/s, big gates x0.86-0.90 (DISTILL:176, 269) | board: 1.4956 | blob: - | reopen: never on mma.sync; wide N belongs on wgmma | 2026-10-26 campaign
- [refuted] Engine D: dense bf16 skinny GEMM (P, qkv, o) on v4's CTA with stream-K, a raw-byte engine | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: f59cb301: P x0.923, qkv x0.804, o x0.530 at T=1, worse with T (DISTILL:177, 270) | board: 1.4956 | blob: - | reopen: fewer bytes (bulk12) or a launch hidden behind its predecessor in the PDL chain | 2026-10-26 campaign
- [refuted] selector tie-breaks: prefer_within 0.01 (earliest engine within 1%), and re-rolling identical bytes for noise | rows: load=decode_16 | evidence: a1bc1756 put T=9..16 on v3, d16 -4..-6% (DISTILL:181, 273) | board: 1.4956 | blob: - | reopen: - | 2026-10-26 campaign
- [refuted] PDL on Engine A v4's K1 at T=2 (after kernel R) | rows: load=decode_1, load=verify_1 | evidence: ff2bd918 illegal address in the copy check; the PDL gate measured x0.998-1.000 (LOG:318-327; DISTILL:274-278) | board: 1.4956 | blob: - | reopen: - | 2026-10-26 campaign
- [refuted] earlier engines and launch forms: Engine B v1/v2, exp4 lossless line, K2 at 1 CTA/SM, R -> K1 PDL | rows: load=decode_1, load=verify_1 | evidence: DISTILL:184 (see the decode-first RETRO.md) | board: 1.4956 | blob: - | reopen: not in their old form; bulk12 is the working 12-bit form | 2026-10-26 campaign
- [refuted] sizing levers from PDL-inflated profile rows (combine shows 22-41 us vs 3-4.5 us real; P 22-27 vs 19.6 us) | rows: all | evidence: DISTILL:266 | board: 1.4956 | blob: - | reopen: - | 2026-10-26 campaign
- [refuted] dq3 / dq4 dequant variants in Engine A's consumer (VERDICT-dq) | rows: load=decode_16, load=decode_32, load=verify_1 | evidence: best dq3 x1.015 at C-R4 tok16, dq4 x0.86-0.89 (LOG:716-717); repeated x1.000 / x1.019 in e1406e5b (LOG:762) | board: 1.5275 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] hybrid threshold H=-8 at T=128/256 (threshold sweep exhausted) | rows: load=verify_16, load=verify_32 | evidence: decode T=128 x0.982 vs H-16 x1.064, T=256 x1.157 vs x1.190 (LOG:718-719) | board: 1.5275 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] k-slice consumer rung (lever #1): B-fragment loads are not the consumer's limiter | rows: load=decode_16, load=decode_32, load=verify_1 | evidence: moe e1406e5b05ba4cf7: ks4 tok8 x1.003, ks8 tok8 x0.974, tok16 x0.869-0.945 (LOG:758-762) | board: 1.5275 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] wg2: the big-T wgmma hot-half re-split | rows: load=verify_16, load=verify_32 | evidence: 74201da03ab64f40 1.5176: H-17 x0.920 at T=128, x1.017 at T=256 vs H-16 x1.050 / x1.156 (LOG:785-787); moe x0.405 / x0.882 (LOG:774-776) | board: 1.5275 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] eat18 / eaf: a new T<=8 expert-block streaming engine (fused w13 -> w2) | rows: load=decode_1, load=verify_1 | evidence: 5b4e5b8f49d642c8 1.5311: eaf x0.991 vs v1 at T=1 (LOG:845) | board: 1.5275 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] dchain-pdl: CUDA PDL rewrite of the o GEMM (o_pdl) in the T<=32 chain | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: 655304067cd74dfe 1.5326: isolated o_pdl > 1.3x the incumbent in every cell (Triton swap-AB o at 4.5-5 us) (LOG:846-847) | board: 1.5275 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] W6-A glue folds: FFN boundary folded into the unpooler (G1) and R's T=1 tail on b12p's last CTA (G2), i.e. last-arriver single-CTA tails | rows: load=decode_1 | evidence: ea65cba7 1.5484: G1 x0.974 (+2.2 us), G2 x0.981 (LOG:1007-1013) | board: 1.5697 | blob: - | reopen: a fold that keeps the work spread over CTAs (no single-CTA tails) | 2026-10-27 campaign
- [refuted] trigger-first boundary PDL at T=1 (the incumbent chain already hides it), and the hot / cold fork inside decode's layer (T=128 x0.93 block gain does not survive; T=256 x0.975; static SM splits x1.10-1.64 slower) | rows: load=decode_1, load=verify_16, load=verify_32 | evidence: 1fc3b48974874a17 (LOG:1071-1075); ee9142eb7dd64f8a 1.5645, H-16f x1.003 vs master (LOG:1123-1124) | board: 1.5697 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] S1a: Engine S hot kernel in the hybrid's hot half (the hot half already runs at the K4 K-step rate; its gap is fixed costs) | rows: load=verify_16, load=verify_32 | evidence: moe judge 2157c43bb41f4d43: S x0.890 vs master (below wg's 0.916); H100 S/wg x0.89-0.95 (LOG:1457-1463) | board: 1.5926 | blob: - | reopen: - | 2026-10-28 campaign
- [refuted] X1 for v4 (x preloaded per 128-k kq with one ldmatrix.x4): the bench's x1.037 flipped in production | rows: load=verify_1 | evidence: cbec03d56e8e4da3 1.639: decode (1,8) full x0.937 vs the L4 check (LOG:1853-1861); eac x1.037 on the H100 (H100:954) | board: 1.6407 | blob: - | reopen: - | 2026-10-28 campaign
- [refuted] sized out: full-attention split-KV rewrite in decode (decode 32 full at its floor is only x1.013 on the headline; attention is 48% of that row, splitkv_attn_sk_kernel 187.5 us) | rows: layer=full load=decode_32 | evidence: LOG:2095-2097, 2159-2160 | board: 1.6551 | blob: - | reopen: a measured floor gap that moves the headline > 2.5% | 2026-10-28 campaign
- [refuted] sized out: full-layer verify attention rewrite (FA3 -> decode sk / c320 widened to q_len 8) | rows: layer=full load=verify_1, layer=full load=verify_16, layer=full load=verify_32 | evidence: 7cc034f917274439 profile: FA3 286.4 / 153.6 / 19.6+7.5+3.3 us vs decode kernels on the same KV 191.3 / 106.7 / 15.1 us = +1.9% headline at parity, ~3.2% at the KV floor; full-verify-cuda-splitkv-stage2 (mma.sync) 506 vs 296 us | board: 1.6904 | blob: 315119e2 | reopen: a measured q_len-8 kernel at <= 0.75x FA3 at B=32, or a stack with another full-row lever to >= 2.5% | 2026-10-29 screen

### superseded

- [superseded] attention-problem ports into decode: full decode sk ticket restack at B=1..64, sliding sk window at decode_32 / verify_1 | rows: layer=full load=decode_16, layer=full load=decode_32, layer=sliding load=decode_32, layer=sliding load=verify_1 | evidence: 9c99379067084f6e 1.2847, 3117f2e423d44cb7 1.3117 (FINDINGS:175, 176); decode now runs splitkv_attn_sk with the full-line gather (d5eeef58) and the window kernel at B=17..32 (DECISIONS:426) | board: 1.3142 | blob: 2e0687a9 | reopen: - | 2026-10-25 churn
- [superseded] Marlin-hot hybrid at T=256 (H=-48 / -32) | rows: layer=sliding load=verify_32, layer=full load=verify_32 | evidence: DISTILL:183; decode v32 rows +11-16% with the wgmma hot half | board: 1.4956 | blob: - | reopen: - | 2026-10-26 campaign
- [superseded] W6-C BN128 sliding decode_1 attention cell | rows: layer=sliding load=decode_1 | evidence: 71741a4f2cdc4cc5 1.5499: attention side x1.012 (bar 1.02) (LOG:1001-1002); PERHEAD dominates it (x1.10 vs x1.02, LOG:1053) | board: 1.5697 | blob: - | reopen: - | 2026-10-27 campaign
- [superseded] Engine A v6 (S3) port at decode T=1 alone (678f6e9c 1.5860, d1 block per expert 10.03 -> 8.98 us) | rows: load=decode_1 | evidence: LOG:1615-1624; carried into the L3 + v6 landing 622b97bb | board: 1.5926 | blob: - | reopen: - | 2026-10-28 campaign
- [superseded] sliding cluster C=4 s3 satfinite on the sk ticket kernel (sliding-cluster-c4-s3-satfinite-on-sk-ticket) | rows: layer=sliding load=decode_32 | evidence: DECISIONS:426: its base never landed; sliding B=17..32 already runs the stream-K window kernel | board: 1.6551 | blob: - | reopen: - | 2026-10-28 screen
- [superseded] T=128/256 FFN front concat route (ffn-front-t128-t256-concat-route) | rows: load=verify_16, load=verify_32 | evidence: DECISIONS:427: the W6-B T>=128 diet cells landed (dca993b1) | board: 1.6551 | blob: - | reopen: - | 2026-10-28 screen
- [superseded] fused split-I MoE (Marlin-era) ports at 1<=T<=8 and 9<=T<=32 (fsi-moe-tiny-t-decode-port, fsi-moe-midt-decode-port) | rows: load=decode_1, load=verify_1, load=decode_16, load=decode_32 | evidence: DECISIONS:428-429: Engine A replaced Marlin at T<=8; Engine A v4c runs T=9..32; moe's fsi-midt measured x0.58-0.81 (LOG:1422-1423) | board: 1.6551 | blob: - | reopen: - | 2026-10-28 screen
- [superseded] Marlin call knobs (block_size_m 8 at T=17..32; w13 atomic add at T=1) and per-class Marlin block_size_m at T=128/256 | rows: load=decode_1, load=decode_32, load=verify_16, load=verify_32 | evidence: DECISIONS:430-431: Marlin is no longer on the T<=32 path; the hot/cold hybrid (H=-16) landed | board: 1.6551 | blob: - | reopen: - | 2026-10-28 screen

## Current best

1.6904x [1.6650,1.7158] from job `7cc034f917274439` (submit), master commit `d28b23d729`, hypothesis v4xd-stack

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60: yours 89.1µs vs SGLang 171.4µs (SoL 40.0µs) · void eak6::ea6_kernel<true, 3, 2, true>(__nv_bfloat16 con... 20.3µs 19.6%, _bf16_swapab_kernel 19.9µs 19.2%, void b12p::b12p_kernel<12, 8, 4>(unsigned char const*, un... 18.8µs 18.2%
- batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60: yours 92.4µs vs SGLang 179.3µs (SoL 44.2µs) · void eak6::ea6_kernel<true, 3, 2, true>(__nv_bfloat16 con... 20.8µs 19.3%, void b12p::b12p_kernel<12, 8, 4>(unsigned char const*, un... 17.1µs 15.9%, void eak6::ea6_kernel<false, 2, 2, true>(__nv_bfloat16 co... 13.8µs 12.8%
- batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic: yours 161.1µs vs SGLang 260.3µs (SoL 76.7µs) · void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 50.9µs 25.9%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 42.9µs 21.8%, void (anonymous namespace)::dl_pooler_router_kernel<2>(__... 23.7µs 12.1%
- batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic: yours 254.3µs vs SGLang 443.1µs (SoL 137.5µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 106.7µs 37.3%, void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 49.9µs 17.5%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 41.0µs 14.4%
- batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic: yours 195.9µs vs SGLang 266.9µs (SoL 84.5µs) · void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 44.0µs 22.7%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 40.1µs 20.7%, void (anonymous namespace)::dl_pooler_router_kernel<4>(__... 26.6µs 13.7%
- batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic: yours 369.9µs vs SGLang 631.7µs (SoL 208.1µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 191.3µs 46.6%, void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 66.8µs 16.3%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 60.9µs 14.8%
- batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8: yours 139.1µs vs SGLang 259.2µs (SoL 62.3µs) · void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 43.3µs 27.5%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 34.6µs 21.9%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 22.0µs 14.0%
- batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8: yours 159.3µs vs SGLang 277.1µs (SoL 66.4µs) · void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 39.2µs 22.5%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 34.3µs 19.6%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 23.9µs 13.7%
- batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 321.1µs vs SGLang 532.0µs (SoL 126.6µs) · void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 107.6µs 26.7%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 89.5µs 22.2%, void (anonymous namespace)::wg_kernel<true>((anonymous na... 45.1µs 11.2%
- batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 460.2µs vs SGLang 720.0µs (SoL 188.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 153.6µs 28.4%, void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 84.6µs 15.6%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 69.6µs 12.9%
- batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 427.7µs vs SGLang 723.9µs (SoL 143.8µs) · void (anonymous namespace)::wg_kernel<true>((anonymous na... 102.1µs 22.0%, void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 78.5µs 16.9%, void eak4x::ea4x_w2_kernel<2, false>(__nv_bfloat16 const*... 62.4µs 13.4%
- batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 700.4µs vs SGLang 1076.2µs (SoL 267.2µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 286.4µs 39.5%, void eak4x::ea4x_w13_kernel<2, false>(__nv_bfloat16 const... 81.4µs 11.2%, void (anonymous namespace)::wg_kernel<true>((anonymous na... 80.2µs 11.1%

Diagnosis of 2026-10-25T203237Z, written on board 1.3142 (today 1.6904): historical, re-diagnose before minting from it (`muse_decode_layer/hypotheses/DIAGNOSIS.md`).

## By area

### FA3 call
- **triton-flash-decode-full-paired-kv** → neutral (unproven-vs-best), 1.0702x, mechanism flat 0/2 — gained on sliding verify_16; full verify_16. On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair buil
- **triton-flash-decode-full-oversub-kv-prefetch** → regression (regression), 1.0413x, mechanism miss 0/2 — lost on full decode_16, decode_32; sliding verify_16. Build on stage 1 (branch hyp/muse_decode_layer/triton-flash-decode-full-paired-kv, muse_decode_layer/submission.py) and change only _full_decode_splits and the 
- **fa3-hdim128-head-pairing-batched** → neutral (unproven-vs-best), 1.0874x, mechanism hit 4/4 — gained on full decode_16, decode_32, verify_16, verify_32. Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attentio
- **sliding-fused-triton-decode1-port** → neutral (unproven-vs-best), 1.1690x, mechanism hit 1/1 — gained on sliding decode_1. Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy ve
- **full-decode-cuda-splitkv-port** → neutral (unproven-vs-best), 1.1862x, mechanism hit 2/2 — gained on full decode_16, decode_32. Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e
- **full-verify-cuda-splitkv-stage2** → regression (regression), 1.1559x, mechanism miss 0/2 — lost on full verify_16, verify_32. Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify blocks (q_len 8) at batch 16/32, where master still runs paired FA3

### FA3 call (full decode, all batches): split-KV A/B at 1<B<=64 and FA3 chain at B=1
- **full-decode-sk-ticket-restack-b1-b64** → neutral (within-ci), 1.2847x, mechanism hit 3/3 — gained on full decode_16, decode_32; lost on sliding verify_32. Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention-side stream-K ports that each won their rows on the previous master, 

### FA3 call (full decode, batch 1)
- **full-decode1-two-level-ticket-port** → neutral (within-ci), 1.2746x, mechanism hit 1/1 — gained on sliding verify_32. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_

### FA3 call (full decode, batch > 1)
- **full-decode-splitkv-restack-on-pdl** → neutral (unproven-vs-best), 1.2680x, mechanism hit 2/2 — gained on full decode_16, decode_32. Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decod

### FA3 call (full decode, batch > 1): splitkv_attn_kernel + splitkv_combine_kernel
- **full-decode-streamk-ticket-port** → neutral (within-ci), 1.2775x, mechanism hit 2/2 — gained on full decode_16, decode_32. Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed restack). Built on master, which now carries the landed full-deco

### FA3 call (full decode, batch > 1): splitkv_attn_kernel bytes in flight
- **full-decode-splitkv-kv-bw-3p5** → neutral (within-ci), 1.2663x, mechanism miss 0/2 — lost on full decode_16. Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on the restacked code from full-decode-splitkv-restack-on-pdl. Kerne

### FA3 call (sliding decode B=17..32 and sliding verify B=1: windowed stream-K)
- **sliding-sk-window-port-decode32-verify1** → neutral (within-ci), 1.3117x, mechanism flat 1/2 — gained on sliding decode_16; lost on full verify_16; sliding verify_32. Port muse_attention_decode master's windowed stream-K sliding routes (commit 7e12891512, best 1.5017x check 252abb7f11ec48c3; hypotheses restack-sliding-decode-

### FA3 call (sliding decode, batch 1)
- **sliding-decode1-fused-restack-on-pdl** → neutral (unproven-vs-best), 1.2374x, mechanism hit 1/1 — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16; full decode_1, decode_16, decode_32, verify_1. Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4 us, +4.4%, every other shape flat; it also carried the +8.75% slid

### FA3 call (sliding decode, batch 17..32)
- **sliding-cluster-c4-decode32-port** → neutral (within-ci), 1.2693x, mechanism flat 0/1 — no shape moved outside noise. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_
- **sliding-cluster-c4-decode32-s2-k-encode-match** → neutral (unproven-vs-best), 1.2803x, mechanism flat 0/1 — gained on full decode_1; sliding verify_32. Stage 1 (run 0fb2c945) was NEVER MEASURED: sliding_cluster_kernel C=4 built (124 regs, 32 B spill, 103 KB smem, 62 active clusters), ran without fault and its a

### FA3 call (sliding decode, batch 2..16)
- **sliding-batched-triton-b16-band** → neutral (unproven-vs-best), 1.2743x, mechanism hit 1/1 — gained on full decode_32. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Re-apply decode's ha

### FA3 call (sliding decode, batch > 1)
- **sliding-fused-batched-decode-port** → neutral (within-ci), 1.2248x, mechanism hit 2/2 — gained on sliding decode_16, decode_32; lost on sliding verify_32; full verify_32. Port muse_attention_decode master batched sliding-decode path (landed sliding-deep-splits-on-cuda-splitkv, job 698d4cd3838c4a63: sliding decode_16 20.7 -> 15.3 

### MoE align (T>32) + sliding batch-1 attention
- **stack-align-par-sliding1** → neutral (unproven-vs-best), 1.2444x, mechanism flat 3/12 — gained on sliding decode_1, verify_32. Built on master (dense-chain PDL). Stack the two decode checks of 2026-10-24 night that each beat the 1.228 best, applied unchanged from their done/ patches: mo

### MoE align/dispatch
- **expert-id-map-single-kernel** → neutral (unproven-vs-best), 1.0374x, mechanism flat 2/8 — no shape moved outside noise. Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_i
- **moe-align-single-kernel-small-t** → neutral (within-ci), 1.0431x, mechanism flat 0/10 — no shape moved outside noise. At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one 

### MoE align/dispatch (T>32)
- **moe-align-parallel-large-t-port** → neutral (unproven-vs-best), 1.2433x, mechanism flat 3/4 — gained on sliding verify_16; full verify_16. Port muse_moe_w4a16 align-parallel-per-expert-verify (patch muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check 45eacf4ca2e

### MoE align/dispatch (stream placement)
- **moe-dispatch-prep-on-side-stream** → neutral (unproven-vs-best), 1.2294x, mechanism flat 0/12 — no shape moved outside noise. Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto state["side_stream"], after state["routing"](...), so it overlaps 

### T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention
- **stack-align-par-sidestream-sliding1** → neutral (unproven-vs-best), 1.2495x, mechanism flat 4/12 — gained on sliding decode_1, verify_16; full verify_16. Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream bu

### T>32 MoE dispatch on the side stream + sliding batch-1 fused attention
- **stack-sidestream-sliding1** → neutral (unproven-vs-best), 1.2352x, mechanism flat 1/12 — gained on sliding decode_1. Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 r

### T>32 MoE dispatch on the side stream + sliding batch>1 fused attention
- **stack-sidestream-sliding-batched** → neutral (unproven-vs-best), 1.2443x, mechanism flat 2/12 — gained on sliding decode_16, decode_32. Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 r

### T>32 parallel align + sliding batch>1 fused attention
- **stack-align-par-sliding-batched** → failed (test:failed), mechanism flat 2/7 — gained on sliding decode_16, decode_32; lost on full verify_1. Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 r

### cache store
- **cache-store-fused-fp8-cast** → neutral (unproven-vs-best), 1.0303x, mechanism flat 0/8 — no shape moved outside noise. In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(token

### combination: C1 measured stack + w13/w2 rega and align2 at 128<=T<=512
- **combo-measured-rows-plus-rega-align2-large-t** → neutral (unproven-vs-best), 1.3164x, mechanism miss 1/9 — lost on sliding verify_16; full verify_16. COMBINATION (measured stack + cross-problem MoE at large T): C1's three decode-measured sources (S1 Triton sliding batched 2..32 from sliding-fused-batched-deco

### combination: sliding attention B=2..32 + pre-attention feeds (T>1) + routing R / unpooler (2<=T<=32 / 2<=T<=8)
- **combo-measured-rows-slide-batched-preattn-t8-tail** → neutral (unproven-vs-best), 1.3142x, mechanism flat 1/6 — gained on full verify_1, verify_16. COMBINATION (naive top-row stack): the decode-measured per-shape winners that never reached master, each routed only to the shape class where it won, on master 

### full decode_1 attention side (dl3 piece A enable)
- **dl3-piece-a-gate-stride-check-fix** → neutral (unproven-vs-best), 1.3080x, mechanism hit 1/1 — gained on full decode_1. Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + 

### moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256
- **hothalf-a3-exactn-ring-wh** → neutral (within-ci), 1.6794x — no shape moved outside noise. Port the H100-validated 'wh' stack into decode's wgmma hot half (wg_kernel<true>/<false> and its dq_stage, muse_decode_layer/submission.py ~:15330; moe twin mus
- **hothalf-wh5ds-moe-port** → neutral (unproven-vs-best), 1.6968x — gained on sliding verify_16, verify_32; full verify_32. Port moe's landed wh5ds hot half back to decode: take the module from muse_moe_w4a16/submission.py on master (landed hothalf-wh-a5b-port, scored a5318a1f: arms 

### o GEMM + pooler GEMM (L2 warm-up overlapped with attention)
- **l2-prefetch-o-pooler-during-sliding-attention** → regression (regression), 1.1670x, mechanism miss 0/3 — lost on sliding decode_16, decode_32, verify_1. On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.e. sliding decode_16, decode_32 and verify_1), overlap an L2 warm-up o

### output norm-gate
- **paired-heads-pad-unpad-triton** → neutral (within-ci), 1.1654x, mechanism flat 1/4 — gained on full verify_16. Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master'

### pooler + router GEMM (kernel P, T=9..32)
- **dl-p-wide-tile-t16-t32** → neutral (within-ci), 1.2457x, mechanism flat 0/4 — no shape moved outside noise. Kernel P (dl_pooler_router_kernel<NF>) streams only 2.85 TB/s at T=32 and 3.41 at T=8 (prepare timing in 81ee6714b7614a99.log: 26.4 us vs 20.7 at T=1) because e

### pooler GEMM
- **pooler-gemm-splitk-autotune** → neutral (within-ci), 1.0189x, mechanism flat 0/4 — no shape moved outside noise. For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in pre
- **bf16-shape-gated-triton-gemms-small-t** → neutral (unproven-vs-best), 1.1688x, mechanism miss 2/6 — gained on sliding decode_1; full decode_1, decode_32, verify_16; lost on sliding decode_16, verify_1; full verify_1. Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel,

### post-MoE unpooler + residual boundaries (T<=32)
- **boundaries-unpooler-pdl-small-t** → neutral (unproven-vs-best), 1.2559x, mechanism flat 1/8 — no shape moved outside noise. Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte

### post-expert norm+combine
- **post-expert-norm-triton-small-t** → gain (gain), 1.1635x, mechanism hit 6/6 — gained on sliding decode_1, decode_16, verify_1; full decode_1, decode_16, verify_1. Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-tri

### pre-attention norm+rope
- **fused-preattention-triton** → neutral (unproven-vs-best), 1.0970x, mechanism flat 1/8 — gained on sliding decode_32. Port muse_attention_decode's fused pre-attention Triton kernel (main tree muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x the
- **fused-preattention-paired-pad-restack** → neutral (unproven-vs-best), 1.2745x, mechanism flat 4/9 — gained on full decode_1, decode_16, decode_32; sliding verify_1. Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.

### qkv GEMM + o GEMM (2 <= T <= 16)
- **qkv-o-swapab16-t2-t16-no-fork** → neutral (unproven-vs-best), 1.3056x, mechanism flat 1/4 — gained on sliding decode_1; full decode_1, decode_16, decode_32; lost on sliding verify_32. Port muse_bf16_proj master's m<=16 swap-AB wgmma path (hypothesis bf16-m16-swapab-wgmma-extend: qkv 18% faster, o 10-19% at m<=16 there) to exactly two call sit

### qkv GEMM + o GEMM + attention-side small kernels (T<=32)
- **dense-chain-stage2-attn-side-pdl** → regression (regression), 1.2253x, mechanism miss 1/8 — gained on full decode_1; lost on sliding decode_16, decode_32; full decode_32. Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["

### qkv GEMM + o GEMM + attention-side small kernels (T=1 only)
- **dense-chain-stage2-attn-side-pdl-t1-gate** → neutral (unproven-vs-best), 1.2717x, mechanism hit 1/1 — gained on full decode_1, decode_32, verify_32; sliding verify_32. Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its 

### restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs)
- **decode-restack-sliding-port-paired-pad-gemms-t1** → neutral (beats-base-not-best), 1.1937x, mechanism flat 3/5 — gained on sliding decode_1, verify_32; full decode_1, verify_16. Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t) by a 3-way merge of their branches, resolving conflicts by hand 

### router GEMM
- **sglang-pipeline-parity-router-hilo-m8** → gain (gain), 1.0216x, mechanism hit 4/4 — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_

### routing top-k (kernel R) + post-MoE unpooler (T<=8)
- **stack-route-topk-pdl-unpool-nf1** → neutral (within-ci), 1.2671x, mechanism flat 3/4 — gained on sliding decode_1, verify_1; full decode_1. Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim restacks dl-route-warp-topk-pdl onto master from its patch (depends_on)

### routing top-k + MoE align (kernel R, T<=32)
- **dl-route-warp-topk-pdl** → neutral (unproven-vs-best), 1.2599x, mechanism miss 2/8 — no shape moved outside noise. Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32 (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes,

### w13 marlin
- **marlin-block-m-cap-16** → gain (gain), 1.0621x, mechanism hit 4/4 — gained on sliding verify_16, verify_32; full verify_16, verify_32. Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first siz

### w13 marlin + silu-mul + w2 marlin
- **moe-chain-cuda-swapab-expert-once-large-t** → failed (test:failed). WHY: at T=128/256 the two Marlin MoE GEMMs are the single largest block of the whole problem: 254/271 us (sliding/full verify_16) and 443/427 us (verify_32) in 

### w13 marlin + silu-mul + w2 marlin (T>32)
- **moe-chain-cuda-swapab-expert-once-fix-inproc** → regression (regression), 1.1651x, mechanism no-base — gained on sliding decode_1, decode_16, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2 still comes after this) of failed moe-chain-cuda-swapab-expert-on
- **moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound** → regression (regression), 1.0637x, mechanism miss 0/4 — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_1, decode_16, decode_32, verify_1; lost on sliding verify_16, verify_32; full verify_16, verify_32. Stage 3 of the expert-once swap-AB MoE chain (moe_route_list + moe_w13_silu_swapab + moe_w2_swapab replacing moe_align + Marlin w13 + act_and_mul + Marlin w2 at

### w2 marlin (T=1)
- **w2-marlin-atomic-add-t1-port** → neutral (unproven-vs-best), 1.2856x, mechanism hit 2/2 — gained on sliding verify_32. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16'

### whole-layer fusion
- **small-t-launch-minimal-dense-chain-pdl** → gain (gain), 1.2283x, mechanism hit 8/8 — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_1, decode_16, verify_1. WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.

### whole-layer fusion (T=1 dense chain: qkv/o GEMM side + routing R + unpooler)
- **dense-chain-s3-t1-q-sk2-o-route-unpool** → neutral (unproven-vs-best), 1.3018x, mechanism hit 2/2 — gained on sliding decode_1; full decode_1, decode_16, decode_32. Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2 dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on 

## Open follow-ups

- triton-flash-decode-full-paired-kv: rewrite stage 1 of `FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- full-decode-cuda-splitkv-port: rewrite stage 1 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32`: read the replaced kernels' own us per shape (profile top_kernels) against the 
- full-decode-splitkv-kv-bw-3p5: rewrite stage 2 of `paired FA3 flash_fwd + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full decode, 1 < batch <= 64 (now splitkv_attn_kernel + splitkv_combine_kernel)`: read the replaced kernels' own us 
- full-decode-streamk-ticket-port: rewrite stage 3 of `splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- full-decode1-two-level-ticket-port: rewrite stage 1 of `FA3 main + FlashAttnFwdCombine + pad/unpad elementwise + FusedRMSNormGate on full decode batch 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor
- sliding-cluster-c4-decode32-port: rewrite stage 1 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`: read the replaced kernels' own us per 
- sliding-cluster-c4-decode32-s2-k-encode-match: rewrite stage 2 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`: read the replaced kernels' own us per 
- dense-chain-s3-t1-q-sk2-o-route-unpool: rewrite stage 3 of `T=1 dense chain: nvjet qkv + FusedQKNorm/rope + FP8 casts + store_cache + nvjet o (full decode_1), routing kernel R and unpooler GEMM+pad at NF=1 (both decode_1 rows)`: read the replaced kernels' own 
- moe-chain-cuda-swapab-expert-once-large-t: rewrite stage 1 of `moe_align_block_size + Marlin w13 + silu_and_mul (act_and_mul_kernel) + Marlin w2 at T > 32 (verify_16 T=128, verify_32 T=256)`: read the replaced kernels' own us per shape (profile top_kernels) again
- stack-align-par-sliding-batched: none

## Shelf (row wins below the headline bar; stack them)

Neutral results whose predicted rows all moved past ±2% (one run each: the rows drift 1.5-2.5%, so re-measure on today's master before stacking).

- **sliding-fused-triton-decode1-port** [FA3 call] unproven-vs-best, 1.1690x, rows 1/1 hit, job `12198572363b49a4`: Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, slidi
- **sliding-fused-batched-decode-port** [FA3 call (sliding decode, batch > 1)] within-ci, 1.2248x, rows 2/2 hit, job `b01fd2df41a040b3`: Port muse_attention_decode master batched sliding-decode path (landed sliding-deep-splits-
- **sliding-decode1-fused-restack-on-pdl** [FA3 call (sliding decode, batch 1)] unproven-vs-best, 1.2374x, rows 1/1 hit, job `30027c28212946fd`: Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 13
- **dense-chain-stage2-attn-side-pdl-t1-gate** [qkv GEMM + o GEMM + attention-side small kernels (T=1 only)] unproven-vs-best, 1.2717x, rows 1/1 hit, job `c6cb53d8ebc2478a`: Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on)
- **sliding-batched-triton-b16-band** [FA3 call (sliding decode, batch 2..16)] unproven-vs-best, 1.2743x, rows 1/1 hit, job `02bf417257e14a3d`: Re-apply decode's harvested sliding-fused-batched-decode-port (done/neutral patch: the Tri
- **full-decode-sk-ticket-restack-b1-b64** [FA3 call (full decode, all batches): split-KV A/B at 1<B<=64 and FA3 chain at B=1] within-ci, 1.2847x, rows 3/3 hit, job `9c99379067084f6e`: Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end. Neutrals whose predicted rows all moved are on the Shelf instead.

- **pooler-gemm-splitk-autotune** [pooler GEMM] neutral, within-ci: For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB wei
- **moe-align-single-kernel-small-t** [MoE align/dispatch] neutral, within-ci: At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call A
- **paired-heads-pad-unpad-triton** [output norm-gate] neutral, within-ci: Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layer
- **dl-p-wide-tile-t16-t32** [pooler + router GEMM (kernel P, T=9..32)] neutral, within-ci: Kernel P (dl_pooler_router_kernel<NF>) streams only 2.85 TB/s at T=32 and 3.41 at T=8 (pre
- **stack-route-topk-pdl-unpool-nf1** [routing top-k (kernel R) + post-MoE unpooler (T<=8)] neutral, within-ci: Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim r
- **sliding-sk-window-port-decode32-verify1** [FA3 call (sliding decode B=17..32 and sliding verify B=1: windowed stream-K)] neutral, within-ci: Port muse_attention_decode master's windowed stream-K sliding routes (commit 7e12891512, b
- **hothalf-a3-exactn-ring-wh** [moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256] neutral, within-ci: Port the H100-validated 'wh' stack into decode's wgmma hot half (wg_kernel<true>/<false> a
- **l2-prefetch-o-pooler-during-sliding-attention** [o GEMM + pooler GEMM (L2 warm-up overlapped with attention)] regression, regression: On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.

## Queue and in flight

- queue: bf16-m32-swapab-wgmma-port [pooler GEMM] p3 hold
- queue: moe-small-t-cuda-swapab-port [w13 marlin + silu-mul + w2 marlin (T<=32)] p4 hold
- queue: moe-bs1-fused-persistent-port [w13 marlin + silu-mul + w2 marlin + post-expert combine (T=1)] p5 hold
- queue: moe-chain-cuda-swapab-expert-once-fix [w13 marlin + silu-mul + w2 marlin (T>32)] p2 hold
- queue: sliding-cluster-c4-s3-satfinite-on-sk-ticket [FA3 call (sliding decode, batch 17..32)] p1 hold
- queue: dl3-unpool-nf1-t2-t8 [unpooler GEMM (T=2..8, dl3 piece C widened)] p2 hold
- queue: dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32 [qkv GEMM + o GEMM + unpooler GEMM (T <= 32)] p1 hold
- queue: ffn-front-t128-t256-concat-route [pooler GEMM + router GEMM + routing top-k + MoE align + pre-expert norm (T > 32)] p2 hold
- queue: moe-rega-bucket-gate-port-t128-t512 [w13 marlin + silu-mul + w2 marlin (128 <= T <= 512)] p1 hold
- queue: preattn-fused-t-gt1-sk-and-fa3-feeds [pre-attention norm+rope + cache store (T > 1, every shape not routed to _sliding_forward or dl3 piece A)] p3 hold
- queue: sliding-cluster-c16-decode1-port [FA3 call (sliding decode, batch 1)] p4 hold
- queue: combo-attn-master-sliding-race-skw-c16 [combination: sliding attention per-band race (B=1, 2..16, 17..32, verify B=1) + routing R / unpooler tail] p1 hold
- queue: swapab16-t2-t16-port-on-master [qkv GEMM + o GEMM (2 <= T <= 16)] p1 hold
- queue: fsi-moe-midt-decode-port [w13 marlin + silu-mul + w2 marlin + post-expert combine (9 <= T <= 32)] p2 hold
- queue: fsi-moe-tiny-t-decode-port [w13 marlin + silu-mul + w2 marlin + post-expert combine (1 <= T <= 8)] p1 hold
- queue: marlin-bsm8-t17-t32-w13-atomic-t1 [w13 marlin + w2 marlin (call knobs: block_size_m 8 at 17 <= T <= 32; w13 use_atomic_add at T = 1)] p2 hold
- queue: marlin-hot-cold-two-class-bsm-t128-t256 [w13 marlin + w2 marlin (block_size_m per expert class, 128 <= T <= 256)] p2 hold
- queue: wg2-occ2-b256-decode-port [w13 marlin + silu-mul + w2 marlin (T = 256; T = 128 candidate)] p4 hold
- 12 stale holds expired to `done/expired/` (not run)
