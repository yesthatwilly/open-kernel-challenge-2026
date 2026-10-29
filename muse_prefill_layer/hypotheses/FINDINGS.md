# muse_prefill_layer findings

Regenerated 2026-10-29T222816Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Claims (from CLAIMS.md, newest per mechanism; tag meanings at the top of CLAIMS.md)

### lead

- [lead] rega stages run super-serial: t0 exceeds even dequant+MMA serial (t1+t2-t5) by 1.22-1.29x, and 18-34% of rega time sits above max(dequant-only, MMA-only) in every class; overlap/TLP levers (K-pair, ping-pong-like schedules) reopen | rows: load=turn_1, load=turn_4, load=first_turn | evidence: check cb22e6104be142ee [rega-anat] (see previous line); CEILINGS RE-SCALED per class rows (adversary of experts-rega-low-kpair-four-consumer-wgs): low ~5.2% headline (all of g at sliding+full turn_1), high ~3%, not 13.3/8.4; the super-serial excess may partly be the producer floor in modes 1/2 (ft t2 = t4 per stage), so a TLP lever's floor is ~t4, a pure-overlap lever's ~max(t1,t2). Candidate siblings: anti-phase of master's two consumer WGs (FA3-style named barriers, 232 regs, bit-exact); deferred wgmma_wait<1> with SASS check for C7510/C7512 | board: 1.5008 | blob: c63801e7 | reopen: - | 2026-10-29 churn (re-measure: measured on 1.5008, best now 1.5721)
- [lead] dq2 A3 dequant (paired-scale prmt+IMAD + ext_prop nibbles on a word-permuted bank) in the rega tile's dq_stage: 26 -> 18 dequant SASS/k16 step, bit-exact | rows: load=turn_1, load=turn_4, load=first_turn | evidence: H100 gpunode06 rgx probe (takemyjob hopper/rgx 1a17a00; research/h100-rega-1029.md): A3/master full-mode pair x0.857 low L2, x0.888 low HBM, x0.835 low random routing, x0.923/0.908 high; 7 interleaved rounds; per-SM ratios transfer; tensor-active full 25.5% vs MMA-only 49.0% at low (issue-serialized, super-serial was a model artifact) | board: 1.5008 | blob: c63801e7 | reopen: judge read: in-flight diag experts-rega-dq2-dequant-cost-diag (routes A3 per class at <=0.97) | 2026-10-29 h100 (re-measure: measured on 1.5008, best now 1.5721)

### shelf

- [shelf] pre-attention q/k norm + RoPE + FP8 cache store fused into one Triton kernel (front-qknorm-rope-fp8-store-fusion-restack), est ~0.5%, stack-only | rows: all | evidence: DECISIONS:418 | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen (re-measure: measured on 1.5008, best now 1.5721)
- [shelf] prepare-only rega stage anatomy (experts-rega-stage-anatomy-diag), 0% headline by itself; it is the revival test for mc2 / bm256 | rows: all | evidence: DECISIONS:421; DISTILL:185, 431 | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen (re-measure: measured on 1.5008, best now 1.5721)
- [shelf] front fusion + routing fusion + PDL chain re-gate stack (front-routing-pdl-launch-diet-stack), est ~1%; its parent never landed | rows: all | evidence: DECISIONS:422 | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen (re-measure: measured on 1.5008, best now 1.5721)
- [shelf] side-stream logits sum + routing top-k + local-id remap in one Triton kernel (routing-triton-hilo-sum-topk-localids), est ~0.2%, stack-only | rows: all | evidence: DECISIONS:423 | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen (re-measure: measured on 1.5008, best now 1.5721)
- [shelf] rega epilogue overlapped with the next item's first wgmma (experts-rega-epilogue-overlap-next-stage), est <= 1.2%, high risk (232-register ceiling) | rows: all | evidence: DECISIONS:425 | board: 1.5008 | blob: - | reopen: re-aim at the fsilu_w2a module | 2026-10-28 screen (re-measure: measured on 1.5008, best now 1.5721)
- [shelf] full-layer FA3 with a fraction f of exp2 on the FMA pipe (Cody-Waite + polynomial) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: H100 pfexp (takemyjob hopper/pfexp-1029 405fbe2; research/h100-pfexp-1029.md, GPU 5): within its own FA3 build f=0.1-0.2 gives 1.07-1.10x (turn_1/turn_4 full), collapsing above f=0.3 (6 extra issue slots per exp); but the rebuilt FA3 (sgl-flash-attn 141fa063 + pinned cutlass/flags) is 1.31-1.34x slower than sgl_kernel 0.4.9's binary and not bit-identical, so every arm vs master is 0.67-0.85x; error holds at every f; value <= ~1-1.5% headline even at 10% | board: 1.5721 | blob: 7569df17 | reopen: a FA3 build that reproduces sgl_kernel's binary speed (or patching the shipped kernel), then f ~0.15, stacked with another prefill full-row gain to >= 2.5% | 2026-10-29 h100

### node-only

- [node-only] prefill turn_1 non-expert buckets have no >=10% lever: dense bf16 GEMMs at cuBLAS's own ceiling (650-820 TF/s; Triton 0.53-0.86x); FA3 knobs 0.998-1.002x (ns=1 0.53x; sliding contiguous window 1.077x but 0.97x with its gather); f16x2/bf16x2 exp no faster than f32 (0.989x). Correction to CLAIMS:39 / LOG:1238: the 39 us kernel is qkv_gate, not o (o 19.5 us alone, 20.8 after a producer) | rows: load=turn_1 | evidence: H100 pf-t1 probe (takemyjob hopper/pf-t1-1029 5df5ba1; research/h100-pf-t1-1029.md): anatomy turn_1 sliding rega 48.6%, dense 32.5%, FA3 5.0%, norm 6.5%, glue 7.5%; full FA3 410 us 41%; ncu FA3 t1f exp unit 70% busy, pure-exp floor 266 vs 401 us, tensor ~32% | board: 1.5721 | blob: 7569df17 | reopen: full-layer FA3 with part of the exps on the FMA pipe (measured ~1.48x exp-throughput ceiling, ~2-3% headline; a rewrite; the last attempt lost x0.881) | 2026-10-29 h100

### landed

- [landed] rega: the register-A swap-AB wgmma NVFP4 grouped GEMM (moe s4 port) for w13 + w2 at T<=4096, then at first_turn with N=128 | rows: load=turn_1, load=turn_4, load=first_turn | evidence: promotes 23c1d5fa (2910b2fb4b744865 1.3275), a1a5497d (07a069bc69ba42a8 1.4060) | board: 1.4060 | blob: c2e3e595 | reopen: - | 2026-10-25 churn
- [landed] two-phase (chunk-count + prefix-place) MoE align on every rega class, including first_turn | rows: load=turn_1, load=turn_4, load=first_turn | evidence: promote 1932a9c4 (4e6f815a8dd945ba 1.4458) | board: 1.4458 | blob: a2fe2a75 | reopen: - | 2026-10-25 churn
- [landed] FA3 on full-attention layers fed by a gated contiguous-KV Triton gather (per-class timing gate) | rows: layer=full | evidence: promote 020566aa (1cc92a43b8be40f8 1.4587) | board: 1.4587 | blob: 9fb1f7c4 | reopen: - | 2026-10-25 churn
- [landed] pre-expert norm fused into the rega row gather (combo-a), then combo-b: norm-gather + two-CTA/SM low-class re-gate (the low-class rega-lo arm never routes on the judge; its phase is skipped since 350343bc) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: promotes 33aa4083 (9bfc488c36c84924 1.4605), 61ce79bb (73b40cd13fdf4bb0 1.4677, hand-landed) | board: 1.4677 | blob: f4751fc0 | reopen: - | 2026-10-25 churn
- [landed] fast silu in the rega w13 epilogue (__fdividef; w13 alone x0.917-0.944, output error x0.996-1.003 of master's vs fp64) | rows: all | evidence: check 21911eb694ce467c 1.4949, scored ec3bd01859574bf1 1.4889 (LOG:1179-1188); promote 9fbde72c | board: 1.4889 | blob: 42ded471 | reopen: - | 2026-10-27 churn
- [landed] rega w2 early gs / sid loads (arm a) routed at the low class (pop / random x0.975; rega-lo guard x0.966-0.968) | rows: load=turn_1 | evidence: 59eb665f162a4ac9 1.4986, confirm 8716d98c405f417d 1.4994, scored 99767cf374bb4688 1.4897 (LOG:1266-1285; FINDINGS:237 is this line); land 95af1c5f | board: 1.4897 | blob: 11a08b3f | reopen: - | 2026-10-27 campaign
- [landed] final near-ties pinned: fa3c contig-varlen at turn_4 / turn_1, w2st arm a at high / ft, the never-routed LO phase skipped (~40 s shorter prepare); deterministic final-rerun bytes | rows: load=turn_1, load=turn_4, load=first_turn | evidence: check 7db6fa1a 1.5055 (vs same-code checks 1.4986 / 1.4994), scored b2941ea23dc743a3 1.5008 (LOG:2039-2042, 2056-2070); land 350343bc | board: 1.5008 | blob: c63801e7 | reopen: - | 2026-10-28 campaign
- [landed] dq2 A3 dequant in the rega tile's dq_stage, routed per class by an in-prepare race (experts-rega-dq2-dequant-cost-diag) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: check 835e1a443a604eb2 1.5556 (pair low x0.854, high x0.909, ft x0.922 vs master, A3 routed all classes, bit-exact); scored 27f0b9af04604842 1.5362 [1.513, 1.559]; H100 transfer held (research/h100-rega-1029.md) | board: 1.5362 | blob: d8ce4cbf | reopen: - | 2026-10-29 churn
- [landed] A5b+D+SC rega dequant (A4b code re-encoding + 7-bit packed scales + per-stage descriptor + split commit) on A3, routed per class (experts-rega-a5b-desc-splitcommit) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: check da80fdb9bed749a3 1.5792 (A5b/A3 pair low .943 high .936 ft .933, all classes routed, bit-exact); scored 9f93484606cd4f9e 1.5721 [1.549, 1.596]; H100 rgx2 (research/h100-rega2-1029.md) | board: 1.5721 | blob: 7569df17 | reopen: - | 2026-10-29 churn

### refuted

- [refuted] persistent multi-row norm boundaries and multi-token Triton MoE pre / post norm | rows: all | evidence: 46afe489e2754c83 1.0716, 3fb4e7237a604fa3 1.0714 vs 1.0744 (FINDINGS:222, 223) | board: 1.0744 | blob: cab1f9ab | reopen: - | 2026-10-25 churn
- [refuted] rega kernel schedule rewrites: two-warpgroup ping-pong consumers, E4M3 scale expansion moved to the producer warps | rows: load=turn_1, load=turn_4, load=first_turn | evidence: 5fbaec9a075543d4 1.4478, 6be2a8d3cf7d489f 1.4538 vs 1.4587 (FINDINGS:230, 231); ping-pong x1.18-1.53 slower (DECISIONS:401) | board: 1.4587 | blob: 762afd24 | reopen: - | 2026-10-25 churn
- [refuted] qkv_gate output-gate columns split onto the side stream under attention | rows: load=turn_1, load=turn_4 | evidence: b8b8536c4ed74d78 1.4512 vs 1.4587 (FINDINGS:232) | board: 1.4587 | blob: 19661e64 | reopen: - | 2026-10-25 churn
- [refuted] PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2) | rows: load=turn_1, load=turn_4 | evidence: 8e4685f303c143a8 1.4622 vs 1.4677 (FINDINGS:233) | board: 1.4677 | blob: 1834ae36 | reopen: - | 2026-10-26 churn
- [refuted] slab multicast across a 2-CTA cluster in rega (mc4 / mc2 / bm256): the ingest-cap premise is false (mc2 cut issued L2 bytes 24-39% and ran 1.31-1.81x slower) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: 0ca60bb7, a6f60630 (DISTILL:185, 439); LOG:599, 645 | board: 1.4677 | blob: - | reopen: a stage anatomy (experts-rega-stage-anatomy-diag) shows ingest-bound stages | 2026-10-26 campaign
- [refuted] rega persistent schedule rebalancing: round-rotated item order, expert groups heaviest first, an atomic work counter (the ingest cap is partly shared; stages / CTA max/min 928/896) | rows: load=turn_1, load=turn_4 | evidence: d00770b6caef4445 1.4664: bit-exact, all classes OFF, span cycles max/mean 1.06-1.13 (LOG:1148-1150; FINDINGS:234); DECISIONS:419-420 | board: 1.4677 | blob: 7625a6d5 | reopen: - | 2026-10-27 churn
- [refuted] rega wgmma N buckets at 16-row granularity (fine-N): ON at the high class in-prepare (pop 0.961 / 0.969, random 0.920), but the judge's warm-L2 profile shows 1096.6 vs 1098.5 us at sliding turn_4 | rows: load=turn_4, load=first_turn | evidence: c951d9168cd546bb 1.4573 (LOG:1169-1173; FINDINGS:235); fine-N-high + fast-silu held (LOG:1177) | board: 1.4677 | blob: 33168720 | reopen: - | 2026-10-27 churn
- [refuted] Triton o-proj GEMM at 200 < T <= 4096 (cuBLAS o is 20 us in isolation; the 39.7 us in the layer profile is not the kernel choice) | rows: load=turn_1, load=turn_4 | evidence: af3b0d0d84834ee9 1.4863, every Triton tile x1.15-1.45 slower (LOG:1279-1280; FINDINGS:236) | board: 1.4889 | blob: ffa750b4 | reopen: - | 2026-10-27 campaign
- [refuted] rega w2 epilogue with cp.async.bulk stores (arm b; TMA contention) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: 665c32dd8c794c0b 1.5010, arm b slower everywhere x1.00-1.05 (LOG:1240-1241); DECISIONS:424 | board: 1.4889 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] rega-lo V64 at the low class pinned on the judge's own routing (finding erased from FINDINGS by c3c14a4d, restored here) | rows: load=turn_1 | evidence: 546f6386; check 56752c8d (lo-pin-v64-fsilu) 1.4776 vs 1.4897 with V64 + fast silu ON [PINNED]; the gate alone kept master (guard 1.009) (LOG:2061) | board: 1.4897 | blob: - | reopen: a different low-class kernel, not V64 / V32 | 2026-10-28 campaign
- [refuted] slab multicast at the low class with the full 128-route slab (experts-rega-low-2cta-fullslab-mc2) | rows: load=turn_1 | evidence: DECISIONS:415: mc2 never landed (1.31-1.81x slower), DISTILL 3.1 retire, REGA_LO_SKIP on master | board: 1.5008 | blob: - | reopen: a stage anatomy shows ingest-bound stages | 2026-10-28 screen
- [refuted] mc2 x bm256 wide tile stack at high / ft (experts-rega-mc2-bm256-stack-high-ft) | rows: load=turn_4, load=first_turn | evidence: DECISIONS:416: mc2 and bm256 both slower, DISTILL retire | board: 1.5008 | blob: - | reopen: a stage anatomy shows ingest-bound stages | 2026-10-28 screen
- [refuted] PDL chain re-gate with IQR slack (experts-rega-pdl-chain-regate-iqr-slack) | rows: load=turn_1, load=turn_4 | evidence: DECISIONS:417: parent on Do-not-try (8e4685f3); the module predates fsilu / w2st | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen
- [refuted] expert groups heaviest first in the rega persistent schedule (experts-rega-lpt-group-order) | rows: load=turn_1, load=turn_4 | evidence: DECISIONS:419: rotation cut the imbalance but missed its gate; round-rotated is Do-not-try | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen
- [refuted] atomic work-counter item scheduler for rega (experts-rega-dynamic-item-scheduler) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: DECISIONS:420: stages per CTA max/min 928/896, the imbalance premise fails | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen
- [refuted] rega epilogue with warpgroup-local barriers and bulk stores (experts-rega-epilogue-wg-local-bulk-store) | rows: all | evidence: DECISIONS:424: w2st arm a landed; the bulk store is arm b, which lost (665c32dd) | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen
- [refuted] pf-rega-low-stream: low-class rega dequant/wgmma register ring (wait<1>) + cp.async.bulk.prefetch.L2 weight prefetch | rows: load=turn_1 | evidence: check 6edf1cb002bf42b9 1.493 (master routed): best arm P8 cold gm x1.075 SLOWER, gate missed; attrs 168 regs 0 local | board: 1.5008 | blob: c63801e7 | reopen: a ring that ptxas does not serialize (check SASS for C7510/C7512) | 2026-10-28 campaign
- [refuted] rega low class K-pair: four consumer warpgroups split each stage's two 64-k records over one 128-route slab (experts-rega-low-kpair-four-consumer-wgs), stagger 0 and 1 | rows: load=turn_1, load=turn_4 | evidence: check bbca0105449c4217 [rega-kp]: -Werror probe clean (no C7510/C7512), r96 l0 o1, accuracy x0.70-0.74 of master's error; low T=753 kp/master w13 x1.110-1.140, w2 x1.104-1.120 (pop/random, cold/warm, both staggers within 0.003); high T=3012 x1.087-1.133; gate kept master in both classes. More WGs at 112 regs + smem split-K reduction loses 10-14% against the super-serial gap (cb22e610) | board: 1.5008 | blob: c63801e7 | reopen: - | 2026-10-29 churn
- [refuted] rega stage anti-phase via ordered wgmma issue between master's two consumer WGs (experts-rega-stage-antiphase-ordered-issue); closes dequant/MMA overlap levers on the current rega tile | rows: load=turn_1, load=turn_4, load=first_turn | evidence: check 17d3fdf1a6594d33 [rega-ap]: -Werror clean, r168 l0, bit-exact (torch.equal) all classes; anti-phase FORMED (retire skew 0.44-0.47 of the stage period at N128, the dominant bucket; 0.18-0.20 at N16-64); yet ap/master pop x1.159-1.164 low, x1.066-1.096 high, x1.102-1.109 ft (w13 and w2 both slower); with K-pair (bbca0105, x1.09-1.14) and ping-pong (CLAIMS:44): the super-serial gap of cb22e610 is not recoverable by overlap; likely per-SMSP dequant/MMA contention (issue slots, register file, smem) rather than idle latency | board: 1.5008 | blob: c63801e7 | reopen: a rega tile that changes the per-SMSP dequant/MMA co-issue cost (e.g. dequant moved off the consumer warps, or a cheaper dequant) | 2026-10-29 churn

### superseded

- [superseded] Marlin / Triton-NVFP4-era expert and align variants (Marlin block_size_m 48 at turn_1 and local-rows block-m; Triton w13 fused silu epilogue, persistent grid and unmasked loads at first_turn; Triton turn_4 persistent + local align; silu-mul local rows / sorted; local parallel align at first_turn; align remap on the side stream; SGLang-32 align + pre-norm restack; FA3 contiguous KV gather) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: 0606f849d2f441ad, 108231906a30407e, b2259479e57e4196, 5d9ddbbaaffd4599, aeda8b5a523642b1, c2f3e9d8297f4a18, 9528eed511b04f33, 374efc72c4d14856, 40f21883155b47eb, 9cfd4f58d24044b2 (neutral), 75bbf278dd9e430e, 8e9943ec1b004b0f (regressions) (FINDINGS:218-221, 224-229, 238, 239); the rega engine and the two-phase align own these rows since 23c1d5fa / a1a5497d / 1932a9c4 | board: 1.0744 | blob: 8507f9e9 | reopen: - | 2026-10-25 churn

## Current best

1.5721x [1.5485,1.5957] from job `9f93484606cd4f9e` (submit), master commit `d9c22d138d`, hypothesis experts-rega-a5b-desc-splitcommit

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6: yours 661.9µs vs SGLang 1073.6µs (SoL 257.0µs) · void (anonymous namespace)::rega_kernel<true, false, 5>((... 224.7µs 33.0%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 156.4µs 23.0%, void (anonymous namespace)::rega_kernel<false, false, 5>(... 120.3µs 17.7%
- batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6: yours 1030.8µs vs SGLang 1436.2µs (SoL 496.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 404.2µs 36.0%, void (anonymous namespace)::rega_kernel<true, false, 5>((... 215.2µs 19.2%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 163.7µs 14.6%
- batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6: yours 1936.1µs vs SGLang 3307.5µs (SoL 1036.3µs) · void (anonymous namespace)::rega_kernel<true, false, 5>((... 578.8µs 28.4%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 324.5µs 15.9%, void (anonymous namespace)::rega_kernel<false, false, 5>(... 302.9µs 14.8%
- batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6: yours 3491.5µs vs SGLang 4922.7µs (SoL 1992.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 1542.1µs 44.1%, void (anonymous namespace)::rega_kernel<true, false, 5>((... 587.1µs 16.8%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 321.8µs 9.2%
- batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 11160.7µs vs SGLang 18523.5µs (SoL 5518.4µs) · void (anonymous namespace)::rega_kernel<true, true, 5>((a... 3394.5µs 29.8%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2517.5µs 22.1%, void (anonymous namespace)::rega_kernel<false, true, 5>((... 1739.7µs 15.3%
- batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6: yours 10326.3µs vs SGLang 17127.3µs (SoL 5663.4µs) · void (anonymous namespace)::rega_kernel<true, true, 5>((a... 3164.5µs 30.1%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2516.1µs 23.9%, void (anonymous namespace)::rega_kernel<false, true, 5>((... 1632.8µs 15.5%
- batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 14556.8µs vs SGLang 21479.5µs (SoL 7710.0µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 4262.0µs 29.5%, void (anonymous namespace)::rega_kernel<true, true, 5>((a... 3150.9µs 21.8%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2522.7µs 17.5%
- batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 17600.9µs vs SGLang 24539.5µs (SoL 9847.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7654.6µs 42.8%, void (anonymous namespace)::rega_kernel<true, true, 5>((a... 3166.9µs 17.7%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2523.9µs 14.1%

Diagnosis of 2026-10-25T203436Z, written on board 1.4677 (today 1.5721): historical, re-diagnose before minting from it (`muse_prefill_layer/hypotheses/DIAGNOSIS.md`).

## By area

### FA3 call
- **fa3-contiguous-kv-gather** → regression (regression), 0.9879x, mechanism miss 1/3 — lost on full turn_1, turn_4. Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache

### FA3 call (CUDA sm_90a FP8 prefill flash attention with software-exp offload, stage 1: full layers, page size 1)
- **attn-full-prefill-cuda-fp8-exp-offload** → regression (regression), 0.8813x, mechanism miss 0/5 — lost on full first_turn, turn_1, turn_4. Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill layers with a hand-written sm_90a FP8 flash-attention forward kernel

### FA3 call on full-attention layers (KV source: contiguous TMA copy of the paged page-size-1 rows, per-class timing gate)
- **fa3-full-contiguous-kv-triton-gather-gated** → neutral (unproven-vs-best), 1.4587x, mechanism hit 2/2 — gained on full first_turn. Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 ch

### FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 1b: fix of attn-full-prefill-cuda-fp8-exp-offload with build fix, hard timing gate, no exp offload)
- **attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc** → neutral (within-ci), 1.0679x, mechanism flat 0/5 — no shape moved outside noise. BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its attention code is unchanged from the old base): start from master's 

### FFN boundary + unpooler GEMM (boundary chunks overlapped under unpooler chunks via a slice-writing boundary kernel)
- **boundary-overlap-s1-slice-kernel-ffn-tail** → regression (regression), 1.4438x, mechanism flat 0/4 — lost on sliding turn_1. Rewrite stage 1, built on current master (blob a2fe2a75). Goal of the line: take the two FusedPostNormResidualPreNorm boundaries (muse_spark_norm_residual_kerne

### MoE align on the Marlin path (T <= 4096) + pre-expert norm: restack of two measured row wins
- **turn-align-sglang32-prenorm-triton-restack** → neutral (within-ci), 1.0694x, mechanism flat 0/4 — no shape moved outside noise. Built on master (a81d19c). Two row-level wins already measured in the archive, applied together, shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: maste

### MoE align on the Triton path (T > 4096, first_turn)
- **align-local-parallel-first-turn** → neutral (within-ci), 1.0626x, mechanism miss 0/4 — no shape moved outside noise. Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size

### MoE align/dispatch
- **moe-align-side-stream** → neutral (unproven-vs-best), 1.0076x, mechanism flat 0/4 — no shape moved outside noise. Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.

### MoE align/dispatch (align over the 32 local experts at T > 4096)
- **moe-align-32-experts-first-turn** → neutral (unproven-vs-best), 1.0604x, mechanism flat 0/4 — no shape moved outside noise. Built on master (635cfee). One argument in _experts_triton (T > 4096 only): moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes moe_align_block_s

### MoE align/dispatch (local-routes-only align, port of muse_moe_w4a16 align-parallel-per-expert-verify)
- **moe-align-local-routes-parallel-port** → neutral (unproven-vs-best), 1.0658x, mechanism flat 1/8 — no shape moved outside noise. Built on master (635cfee). Port muse_moe_w4a16's measured align (master there: _align_parallel_kernel and its host wrapper _align_parallel in muse_moe_w4a16/sub

### MoE align/dispatch on T <= 4096 (Marlin turn_1 + ssdq turn_4): _align_parallel replaced by a two-launch chunk-count + prefix-place align
- **turn-align-two-phase-chunk-prefix** → neutral (unproven-vs-best), 1.3369x, mechanism flat 1/4 — gained on sliding turn_4. Built on master (ssdq s1b landed). Only _align_parallel's implementation changes, at its two call sites on T <= 4096: _experts_marlin (turn_1 T=753, block_size_

### MoE align/dispatch on the rega path (200 < T <= 16384): _align_parallel replaced by the two-phase chunk-count + prefix-place align, including the first_turn class
- **align-two-phase-all-t-rega** → neutral (unproven-vs-best), 1.4458x, mechanism flat 3/8 — gained on sliding turn_1, turn_4; full first_turn. Restack of turn-align-two-phase-chunk-prefix (job d792e5c9d6d94f02, neutral 1.3369 on the pre-ft master; patch muse_prefill_layer/hypotheses/done/neutral/turn-a

### MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM
- **turn-align-remap-on-side-stream-under-pooler** → neutral (within-ci), 1.0683x, mechanism flat 0/4 — no shape moved outside noise. Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on the main stream after the router join: 16.9 us at turn_1 (1.7%) 

### attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)
- **norm-boundaries-persistent-multirow** → neutral (within-ci), 1.0716x, mechanism flat 0/8 — no shape moved outside noise. Built on master (635cfee). The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program

### experts rega tile dq_stage / consumer stage loop (w13, w2)
- **experts-rega-a5b-desc-splitcommit** → gain (gain), 1.5792x — gained on full first_turn. Stack the H100-validated A5b+D+SC form on master's landed A3 rega dequant (muse_prefill_layer/submission.py master 1b14174f, the rega dq module from experts-reg

### launch overhead: PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2)
- **experts-rega-pdl-chain-gather-w13-w2** → neutral (within-ci), 1.4622x, mechanism flat 0/4 — no shape moved outside noise. DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no programmatic dependent launch (griddepcontrol appears 0 times i

### o GEMM (attention output projection at 200 < T <= 4096)
- **o-proj-triton-gemm-turn-gated** → neutral (within-ci), 1.4863x, mechanism flat 0/2 — lost on full first_turn. Built on master (blob 42ded471). Only the call attn_output = _gemm(o, dense['o']) in kernel_fn changes, and only at 200 < T <= 4096. MECHANISM: the o projection

### pre-attention norm+rope + cache store
- **qk-norm-fp8-cache-store-fusion** → neutral (unproven-vs-best), 1.0093x, mechanism flat 0/2 — no shape moved outside noise. One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per t

### pre-expert norm (MuseSpark1xRMSNorm) fused into the rega row gather
- **pre-expert-norm-fused-rega-gather** → neutral (unproven-vs-best), 1.4600x, mechanism flat 1/6 — gained on sliding first_turn. Built on current master (submission blob a2fe2a75, align-two-phase-all-t-rega landed; no depends_on). On the rega expert path only (_experts dispatches to _expe

### pre-expert norm + post-expert norm/combine (the MoE elementwise ends)
- **moe-pre-post-norm-triton-multitoken** → neutral (within-ci), 1.0714x, mechanism flat 0/8 — no shape moved outside noise. Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1

### qkv GEMM (output-gate columns split onto the side stream, overlapping qk_norm + cache store + FA3)
- **qkv-gate-columns-side-stream-under-attention** → neutral (within-ci), 1.4512x, mechanism flat 1/6 — no shape moved outside noise. Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection is one GEMM today (qkv = _gemm(attn_input, dense['qkv_gate']), N = 23

### rega w2 epilogue (early gs/sid loads, arm a) routed at the LOW class
- **experts-rega-w2-early-loads-low-class** → neutral (unproven-vs-best), 1.4986x, mechanism flat 0/2 — no shape moved outside noise. Restack of experts-rega-w2-epilogue-prefetch-bulk-store (neutral, check 665c32dd; patch in muse_prefill_layer/hypotheses/done/neutral/) on current master, with 
- **experts-rega-w2-early-loads-low-class-cf** → neutral (unproven-vs-best), 1.4994x — no shape moved outside noise. Confirmation rerun of experts-rega-w2-early-loads-low-class: its implementation commit 52169f1783ec72c5ee8b2e97b0648a0e91426c87 unchanged except one trailing co

### restack: pre-expert norm + rega gather fused (all rega classes) on master 1.4587; naive per-shape winner stack
- **combo-a-master-normgather-naive-stack** → neutral (unproven-vs-best), 1.4605x, mechanism miss 0/8 — no shape moved outside noise. COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved 3-way merge /home/user/git/musespark/muse_prefill_layer/hypothes

### router GEMM (hi and lo halves as one N=512 cuBLAS GEMM)
- **router-hilo-one-gemm-n512** → neutral (unproven-vs-best), 1.0607x, mechanism flat 0/4 — no shape moved outside noise. Built on master (635cfee). prepare(): add state['router_hilo_t'] = torch.cat((hi.t(), lo.t()), dim=1).contiguous(), bf16 [8192, 512] (the same BF16 hi/lo split 

### router GEMM + MoE align (T>4096) + w2 zero-store (T>4096)
- **restack-router-hilo-align32-zerostore** → neutral (unproven-vs-best), 1.0608x, mechanism miss 0/8 — lost on full first_turn. Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one

### router GEMM + MoE align (both paths) + w2 zero-store
- **stack-tonight-four-prefill** → neutral (unproven-vs-best), 1.0675x, mechanism flat 0/8 — no shape moved outside noise. Built on master. Stack all four prefill changes built on 2026-10-24 night, each applied unchanged: router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w

### router GEMM + MoE align (both paths) + w2 zero-store + Marlin silu-mul
- **stack-tonight-five-prefill** → neutral (unproven-vs-best), 1.0705x, mechanism flat 0/8 — no shape moved outside noise. Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-

### router GEMM + MoE align (both paths, parallel) + w2 zero-store + Marlin silu-mul
- **stack-tonight-six-prefill** → neutral (unproven-vs-best), 1.0642x, mechanism miss 0/8 — no shape moved outside noise. Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel local-routes align on the Triton path too, self-checked at T=1638

### silu-mul (Marlin path, T <= 4096)
- **silu-mul-triton-local-rows-marlin-turn** → neutral (within-ci), 1.0512x, mechanism flat 0/4 — no shape moved outside noise. Built on master (635cfee). In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS = 4096: turn_1 T=753, turn_4 T=3012 and every test), replace silu_an

### silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096)
- **experts-triton-w13-fused-silu-epilogue** → neutral (within-ci), 1.0525x, mechanism miss 0/4 — lost on full first_turn. Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T > 4096 path (_experts_triton, first_turn only), built on master. (1) 

### silu-mul (rega w13 epilogue)
- **experts-rega-w13-fast-silu-epilogue** → gain (gain), 1.4949x, mechanism flat 3/8 [gain not attributable to the mechanism (race/draw?)] — gained on sliding turn_1. Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32 accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.

### silu-mul on the Triton path (T > 4096, first_turn)
- **silu-mul-sorted-first-turn** → neutral (within-ci), 1.0636x, mechanism miss 0/4 — no shape moved outside noise. Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, 

### sliding-window attention block (q FP8 cast + FA3 + FlashAttnFwdCombine + FusedRMSNormGate) on sliding layers
- **sliding-attn-block-triton-gqa-fused-normgate** → neutral (within-ci), 1.0507x, mechanism miss 0/3 — no shape moved outside noise. Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing the two in-flight prefill runs touch (full-layer FA3 stays FA3, e

### w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid
- **experts-triton-turn4-persistent-local-align-control-no-persi** → neutral (within-ci), 1.0557x, mechanism flat 0/4 — no shape moved outside noise. Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 10

### w13 + w2 expert GEMMs (prepare-only diagnostic: where a rega stage goes)
- **experts-rega-stage-anatomy-diag-r2** → neutral (within-ci), 1.4939x — gained on full first_turn. Built on master (submission.py blob c63801e7, board 1.5008: fast-silu w13 and w2st arm a landed, LO phase pinned off by REGA_LO_SKIP). A prepare-only diagnostic

### w13 + w2 expert GEMMs (rega consumer schedule: ordered wgmma issue / stage anti-phase of master's two consumer warpgroups), low / high / ft classes
- **experts-rega-stage-antiphase-ordered-issue** → neutral (within-ci), 1.4857x, mechanism miss 0/8 — no shape moved outside noise. Built on master (submission.py blob c63801e7, board 1.5008; master dispatches the muse_prefill_rega_fsilu_w2a module: fast-silu w13 in every class, w2st arm a w

### w13 + w2 expert GEMMs (rega dq_stage instruction cost: prepare-only diag of cheaper exact dequants, gated route)
- **experts-rega-dq2-dequant-cost-diag** → gain (gain), 1.5556x, mechanism flat 7/8 [gain not attributable to the mechanism (race/draw?)] — gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_1. Built on master (submission.py blob c63801e7, board 1.5008). Changes the INSTRUCTION COST of the rega dequant, not the stage schedule. A prepare-only diagnostic

### w13 + w2 expert GEMMs (rega persistent item->CTA schedule)
- **experts-rega-round-rotated-schedule** → neutral (within-ci), 1.4664x, mechanism flat 0/8 — no shape moved outside noise. Built on master (blob f4751fc0). Only the order in which rega_kernel's persistent CTAs walk items changes; every item is computed by master's code, so outputs a

### w13 + w2 expert GEMMs (rega w2 consumer epilogue: loads and stores)
- **experts-rega-w2-epilogue-prefetch-bulk-store** → neutral (unproven-vs-best), 1.5010x, mechanism flat 1/6 — no shape moved outside noise. Built on master (blob 42ded471, experts-rega-w13-fast-silu-epilogue landed). NO NEW nvcc BUILD: master's muse_prefill_rega_fsilu module already compiles rega_ke

### w13 + w2 expert GEMMs (rega wgmma N buckets at 16-row granularity)
- **experts-rega-fine-n-buckets** → neutral (within-ci), 1.4573x, mechanism flat 0/4 — no shape moved outside noise. Built on master (blob f4751fc0). rega sizes each item's wgmma N (tokens) and its slab TMA box from the chunk's 16-route block count nbk (1..8) with four buckets

### w13 + w2 expert GEMMs (rega_kernel consumer schedule: two-warpgroup ping-pong), every rega class
- **experts-rega-pingpong-consumers-all-t** → neutral (within-ci), 1.4478x, mechanism miss 4/8 — gained on full first_turn; lost on sliding turn_1. Built on master (1932a9c4, align-two-phase-all-t-rega). Consumer schedule of rega_kernel only. ARITHMETIC FIRST (the judge's router seed fixes the busiest GPU's

### w13 + w2 expert GEMMs (rega_kernel dequant: E4M3 scale expansion moved to idle producer warps), every rega class
- **experts-rega-producer-scale-expand-all-t** → neutral (within-ci), 1.4538x, mechanism miss 5/8 — gained on sliding first_turn; full first_turn; lost on sliding turn_1. Built on master (1932a9c4). Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1 (~1.36K local routes, ~42 per expert, buckets 32/64) streams w13'

### w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft classes): 256-weight-row CTA tile with in-warpgroup dequant/MMA overlap
- **experts-rega-s4-bm256-wide-tile-high-ft** → neutral (unproven-vs-best), 1.4758x, mechanism hit 6/6 — gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_4. Rewrite stage 4 of the rega line, built on master (1932a9c4; stage 2 experts-turn-rega-s4-first-turn-n128 landed, stage 3 experts-rega-low-class-two-ctas-per-sm
- **experts-rega-s4-bm256-wide-tile-high-ft-cf** → neutral (within-ci), 1.4711x — no shape moved outside noise. Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing 

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a dequant-producer warpgroup + SS wgmma grouped GEMM, stage 1b: rebuild with the undefined-symbol fix
- **experts-turn4-ssdq-s1b-dataptr-symbol-fix** → neutral (unproven-vs-best), 1.1004x, mechanism hit 2/2 — gained on sliding turn_4; full turn_4. Re-run of experts-turn4-cuda-dequant-producer-ss-wgmma (job 40541db65f1b4ec6, neutral 1.0714), which NEVER RAN: the log reads '[ssdq] load_inline failed after 3

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a warp-specialized grouped GEMM with an in-kernel dequant-producer warpgroup and SS wgmma
- **experts-turn4-cuda-dequant-producer-ss-wgmma** → neutral (within-ci), 1.0714x, mechanism flat 0/2 — no shape moved outside noise. Built on master (a81d19c). At turn_4 the Marlin pair is 2148/2115 us (64%/43% of the sliding/full turn_4 shapes) against a 306 us math floor (~3012 local routes

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): ssdq producer weight feed, TMA raw-code smem ring instead of register __ldg prefetch
- **experts-turn4-ssdq-s2-tma-raw-weight-ring** → neutral (unproven-vs-best), 1.1293x, mechanism hit 2/2 — gained on sliding turn_4; full turn_4. Stage 2 of the ssdq line, built on master (experts-turn4-ssdq-s1b-dataptr-symbol-fix landed: ssdq_kernel serves 1024 < T <= 4096). Only the PRODUCER's weight fe

### w13 + w2 expert GEMMs at 200 < T <= 1024 (rega low class, turn_1): K-pair rega kernel, four consumer warpgroups in one CTA splitting each stage's two 64-k records over master's full 128-route slab
- **experts-rega-low-kpair-four-consumer-wgs** → neutral (within-ci), 1.4926x, mechanism miss 0/4 — no shape moved outside noise. Built on master (submission.py blob c63801e7, board 1.5008; fast-silu w13 and w2st arm a are what master dispatches at the low class, the LO phase is pinned off

### w13 + w2 expert GEMMs at 200 < T <= 1024 (turn_1, rega low class): two resident CTAs per SM with a 64- or 32-route slab, to hide consumer dequant latency
- **experts-rega-low-class-two-ctas-per-sm** → neutral (unproven-vs-best), 1.4173x, mechanism flat 1/2 — gained on sliding turn_1. Rewrite stage 3 of the rega line, built on master a1a5497d (experts-turn-rega-s4-first-turn-n128). It touches ONLY the 'low' class (200 < T <= REGA_LOW_MAX = 10

### w13 + w2 expert GEMMs at 200 < T <= 1024 (two CTAs/SM V64/V32 port, re-gated on master's random routing with a skew guard) stacked on combo A
- **combo-b-normgather-two-cta-low-regate** → neutral (unproven-vs-best), 1.4677x, mechanism flat 0/2 — no shape moved outside noise. COMBINATION B (conjectural: re-gate the two-CTA/SM low class on master's own gate routing, stacked on A). Start from the same merge as combination A (/home/user

### w13 + w2 expert GEMMs at T <= 4096 (turn_1 T=753, turn_4 T=3012): port of muse_moe_w4a16's replay-safe register-A swap-AB wgmma NVFP4 grouped GEMM (s4), all four N buckets
- **experts-turn-rega-s4-port-t4096** → gain (gain), 1.3275x, mechanism hit 4/4 — gained on sliding turn_1, turn_4; full turn_1, turn_4. Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_

### w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 1b: re-measure of experts-cuda-wgmma-mixed-input-grouped-gemm with the build fix, on the new master)
- **experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc** → regression (regression), 0.9401x, mechanism miss 0/4 — lost on full first_turn; sliding first_turn. BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored), not on the old branch: start from master's submission.py and hand-port 

### w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ported s4 register-A wgmma engine past Marlin's range, replacing the Triton NVFP4 GEMM
- **experts-turn-rega-s4-first-turn-n128** → gain (gain), 1.4060x, mechanism hit 4/4 — gained on full first_turn; sliding first_turn; lost on sliding turn_1. Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries the ported _wg_* module and its per-state scratch). First_turn is

### w13 + w2 expert GEMMs, every rega class: activation slab loaded once per 2-CTA cluster and shared by TMA multicast
- **experts-rega-mc2-slab-multicast-all-t** → neutral (within-ci), 1.4630x, mechanism flat 0/8 — no shape moved outside noise. DIAGNOSIS action 1. Built on master (combo-b-normgather-two-cta-low-regate code, blob f4751fc0). New kernel rega_mc_kernel<G, BIG> appended to the rega module s

### w13 marlin + w2 marlin (CUDA sm_90a warp-specialized wgmma W4A16 grouped GEMM, stage 1: first_turn)
- **experts-cuda-wgmma-mixed-input-grouped-gemm** → neutral (unproven-vs-best), 1.0637x, mechanism flat 0/4 — no shape moved outside noise. Replace the expert GEMM engine with a hand-written Hopper (sm_90a) mixed-input grouped GEMM in CUDA, compiled with torch.utils.cpp_extension.load_inline at impo

### w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)
- **experts-triton-first-turn-unmasked-loads-l2-hints** → neutral (within-ci), 1.0677x, mechanism flat 0/4 — no shape moved outside noise. Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKE

### w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)
- **experts-triton-persistent-grid-first-turn** → neutral (within-ci), 1.0726x, mechanism flat 0/4 — no shape moved outside noise. Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (pointer setup, K loop, in-register dequant, t

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 5: packed inline-PTX dequant)
- **experts-triton-nvfp4-packed-asm-dequant** → neutral (within-ci), 1.0556x, mechanism miss 0/4 — no shape moved outside noise. Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4, experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_ker

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 6: take over turn_4)
- **experts-triton-packed-dequant-turn4** → neutral (within-ci), 1.0522x, mechanism flat 0/2 — no shape moved outside noise. Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only t

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 7: take over turn_1 with a weight-stream config)
- **experts-triton-turn1-blockt32-stream** → regression (regression), 0.9425x, mechanism miss 0/2 — lost on sliding turn_1; full turn_1. Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits 

### w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256)
- **experts-triton-w4a16-first-turn-blockt256** → neutral (unproven-vs-best), 1.0229x, mechanism hit 4/4 — gained on full first_turn; sliding first_turn; lost on sliding turn_4. Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and EXPERT_

### w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes)
- **experts-triton-w4a16-inkernel-fp4** → regression (regression), 0.6891x, mechanism miss 0/8 — lost on sliding first_turn, turn_1, turn_4; full first_turn, turn_1, turn_4. A Triton grouped-GEMM kernel with in-register NVFP4 dequant replaces both moe_wna16_marlin_gemm calls wherever master still runs Marlin (if a dequant-to-BF16 pa

### w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, stage 2: swap-AB register-LHS dot)
- **experts-triton-w4a16-swapab-reglhs** → neutral (unproven-vs-best), 1.0068x, mechanism miss 4/6 — gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_1, turn_4. Stage 2 of the Triton NVFP4 grouped GEMM, built on stage 1 (branch hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4, _moe_nvfp4_gemm_kernel in muse_pref

### w13 marlin + w2 marlin (block_size_m pick)
- **experts-marlin-block-m-local-rows** → regression (regression), 0.9836x, mechanism miss 0/2 — lost on sliding turn_1; full turn_1. Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks o
- **experts-marlin-block-m-48-turn1** → neutral (within-ci), 1.0030x, mechanism flat 0/2 — no shape moved outside noise. Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 < num_tokens <= 1024 use block_size_m = 48; every other token count k

### w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above)
- **experts-hybrid-marlin-turn-triton-first-turn** → gain (gain), 1.0599x, mechanism hit 4/4 — gained on sliding turn_1, turn_4; full turn_1. Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3 experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in muse_

### w2 (Triton NVFP4 engine at T > 4096: remote-block zero store)
- **w2-nonlocal-zero-store-off-first-turn** → neutral (unproven-vs-best), 1.0609x, mechanism flat 0/4 — no shape moved outside noise. Built on master (635cfee). One flag and one prepare() guard. Today ZERO_NONLOCAL_W2 = True makes the second _nvfp4_moe_gemm call in _experts_triton (T > 4096 on

### whole-layer parity (SGLang pipeline) + routing remap
- **sglang-pipeline-parity** → gain (gain), 1.0039x, mechanism no-base — no shape moved outside noise. Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline module for module as baseline.py does: _bf16_gemm_dispatch_impl f

## Open follow-ups

- experts-triton-w4a16-swapab-reglhs: rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-w4a16-first-turn-blockt256: rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-nvfp4-packed-asm-dequant: rewrite stage 5 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-packed-dequant-turn4: rewrite stage 6 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-cuda-wgmma-mixed-input-grouped-gemm: rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 in stage 1; moe_wna16_marlin_gemm x2 at T <= 4096 in stages 2-3`: read the replaced kernels' own us per shape (profile top_kernels) against the incumben
- sliding-attn-block-triton-gqa-fused-normgate: rewrite stage 1 of `q.view(T,16,64).to(FP8) + flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) + state['o_norm_gate'] (FusedRMSNormGate) on sliding layers (cfg['window'] == 2048)`: r
- attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc: rewrite stage 1 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`: read the replaced kernels' own us per shape (profile top_kernels) again
- experts-turn4-cuda-dequant-producer-ss-wgmma, experts-turn4-ssdq-s1b-dataptr-symbol-fix: rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- experts-turn4-ssdq-s2-tma-raw-weight-ring: rewrite stage 2 of `ssdq_kernel x2 (w13, w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reac
- experts-rega-low-class-two-ctas-per-sm: rewrite stage 3 of `rega_kernel<true,false> + rega_kernel<false,false> (NMAX 128, 4 stages, 1 CTA/SM) in the low class 200 < T <= 1024`: read the replaced kernels' own us per shape (profile top_kernels) against the incum
- experts-rega-s4-bm256-wide-tile-high-ft: rewrite stage 4 of `rega_kernel<true,*> + rega_kernel<false,*> (128-row tile) in the high (T=3012) and ft (T=16384) classes`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and th
- experts-rega-mc2-slab-multicast-all-t: rewrite stage 1 of `rega_kernel<true,BIG> + rega_kernel<false,BIG> (w13, w2) in all three rega classes: low T<=1024, high 1024<T<=4096, ft 4096<T<=16384`: read the replaced kernels' own us per shape (profile top_kernels)

## Shelf (row wins below the headline bar; stack them)

Neutral results whose predicted rows all moved past ±2% (one run each: the rows drift 1.5-2.5%, so re-measure on today's master before stacking).

- none

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end. Neutrals whose predicted rows all moved are on the Shelf instead.

- **experts-marlin-block-m-48-turn1** [w13 marlin + w2 marlin (block_size_m pick)] neutral, within-ci: Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512
- **experts-triton-w13-fused-silu-epilogue** [silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096)] neutral, within-ci: Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T
- **silu-mul-triton-local-rows-marlin-turn** [silu-mul (Marlin path, T <= 4096)] neutral, within-ci: In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS = 4096: turn_1 T=753, tur
- **experts-triton-persistent-grid-first-turn** [w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)] neutral, within-ci: Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (po
- **norm-boundaries-persistent-multirow** [attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)] neutral, within-ci: The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-ro
- **moe-pre-post-norm-triton-multitoken** [pre-expert norm + post-expert norm/combine (the MoE elementwise ends)] neutral, within-ci: Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['
- **experts-triton-first-turn-unmasked-loads-l2-hints** [w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)] neutral, within-ci: Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, d
- **align-local-parallel-first-turn** [MoE align on the Triton path (T > 4096, first_turn)] neutral, within-ci: Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _ex
- **silu-mul-sorted-first-turn** [silu-mul on the Triton path (T > 4096, first_turn)] neutral, within-ci: Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_trito
- **experts-triton-turn4-persistent-local-align-control-no-persi** [w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid] neutral, within-ci: Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to 
- **turn-align-remap-on-side-stream-under-pooler** [MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM] neutral, within-ci: In the best profile master's _align_parallel_kernel runs serially on the main stream after
- **turn-align-sglang32-prenorm-triton-restack** [MoE align on the Marlin path (T <= 4096) + pre-expert norm: restack of two measured row wins] neutral, within-ci: Two row-level wins already measured in the archive, applied together, shapes untouched oth
- **experts-rega-pingpong-consumers-all-t** [w13 + w2 expert GEMMs (rega_kernel consumer schedule: two-warpgroup ping-pong), every rega class] neutral, within-ci: Consumer schedule of rega_kernel only. ARITHMETIC FIRST (the judge's router seed fixes the
- **experts-rega-producer-scale-expand-all-t** [w13 + w2 expert GEMMs (rega_kernel dequant: E4M3 scale expansion moved to idle producer warps), every rega class] neutral, within-ci: Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1 (~1.36K local routes, ~
- **qkv-gate-columns-side-stream-under-attention** [qkv GEMM (output-gate columns split onto the side stream, overlapping qk_norm + cache store + FA3)] neutral, within-ci: Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection
- **experts-rega-pdl-chain-gather-w13-w2** [launch overhead: PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2)] neutral, within-ci: DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill 
- **experts-rega-round-rotated-schedule** [w13 + w2 expert GEMMs (rega persistent item->CTA schedule)] neutral, within-ci: Only the order in which rega_kernel's persistent CTAs walk items changes; every item is co
- **experts-rega-fine-n-buckets** [w13 + w2 expert GEMMs (rega wgmma N buckets at 16-row granularity)] neutral, within-ci: rega sizes each item's wgmma N (tokens) and its slab TMA box from the chunk's 16-route blo
- **o-proj-triton-gemm-turn-gated** [o GEMM (attention output projection at 200 < T <= 4096)] neutral, within-ci: Only the call attn_output = _gemm(o, dense['o']) in kernel_fn changes, and only at 200 < T
- **experts-rega-w2-early-loads-low-class** [rega w2 epilogue (early gs/sid loads, arm a) routed at the LOW class] neutral, not confirmed (pooled 1.499021779478406): Restack of experts-rega-w2-epilogue-prefetch-bulk-store (neutral, check 665c32dd; patch in
- **experts-rega-stage-anatomy-diag-r2** [w13 + w2 expert GEMMs (prepare-only diagnostic: where a rega stage goes)] neutral, within-ci: A prepare-only diagnostic: kernel_fn, _experts, _experts_rega and every function the timed
- **experts-rega-low-kpair-four-consumer-wgs** [w13 + w2 expert GEMMs at 200 < T <= 1024 (rega low class, turn_1): K-pair rega kernel, four consumer warpgroups in one CTA splitting each stage's two 64-k records over master's full 128-route slab] neutral, within-ci: GOAL: the ~10% TLP effect that two CTAs/SM showed at the low class (V64 occ2 0.887x vs its
- **experts-rega-stage-antiphase-ordered-issue** [w13 + w2 expert GEMMs (rega consumer schedule: ordered wgmma issue / stage anti-phase of master's two consumer warpgroups), low / high / ft classes] neutral, within-ci: Consumer schedule only: FA3-style ordered anti-phase of master's TWO existing consumer war
- **fa3-contiguous-kv-gather** [FA3 call] regression, regression: Replace the paged FA3 call with a contiguous one; everything before and after stays the pa
- **experts-marlin-block-m-local-rows** [w13 marlin + w2 marlin (block_size_m pick)] regression, regression: Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The pari

## Queue and in flight

- queue: attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload [FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 2: warp-specialized producer, 128-key tiles, intra-WG overlap + ping-pong, gated exp offload)] p2 hold
- queue: fa3-num-splits-cap-gated-turn [FA3 call (num_splits cap on the paged path at q_len <= 4096, per-class timing gate)] p3 hold
- queue: experts-rega-low-2cta-fullslab-mc2 [w13 + w2 expert GEMMs at 200 < T <= 1024 (rega low class, turn_1): 2 CTAs/SM with the full 128-route slab, slab multicast across a 2-CTA cluster] p4 hold
- queue: experts-rega-mc2-bm256-stack-high-ft [w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft): 256-row wide tile x 2-CTA slab multicast, stacked] p3 hold
- queue: experts-rega-mc4-cluster-high-ft [w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft): activation slab shared across a 4-CTA cluster by TMA multicast] p5 hold
- queue: experts-rega-rotate-finebucket-fastsilu-combo [w13 + w2 expert GEMMs (rega: rotated schedule + 16-row N buckets + fast silu, one module)] p2 hold
- queue: experts-rega-rotation-fastsilu-combo [w13 + w2 expert GEMMs (rega: rotated schedule + fast silu, one module)] p2 hold
- queue: experts-rega-pdl-chain-regate-iqr-slack [launch overhead (PDL chain _norm_gather_kernel -> rega w13 -> rega w2: gate rule only)] p3 hold
- queue: front-qknorm-rope-fp8-store-fusion-restack [pre-attention norm+rope + cache store (five launches -> one Triton kernel)] p2 hold
- queue: experts-rega-lpt-group-order [w13 + w2 expert GEMMs (rega persistent schedule: expert groups heaviest first)] p4 hold
- queue: experts-rega-dynamic-item-scheduler [w13 + w2 expert GEMMs (rega persistent schedule: atomic work counter)] p3 hold
- queue: experts-rega-stage-anatomy-diag [w13 + w2 expert GEMMs (prepare-only diagnostic: where a rega stage goes)] p5 hold
- queue: front-routing-pdl-launch-diet-stack [launch overhead (front fusion + routing fusion + PDL chain regate, stacked)] p4 hold
- queue: routing-triton-hilo-sum-topk-localids [routing top-k (side-stream logits sum + FusedMuseSparkRouting + local-id remap in one Triton kernel)] p3 hold
- queue: experts-rega-fine-n-high-plus-fast-silu [rega experts kernel (w13/w2 N buckets at the high class + w13 silu epilogue)] p1 hold
- queue: experts-rega-epilogue-wg-local-bulk-store [w13 + w2 expert GEMMs (rega consumer epilogue: barriers and store path, both GEMMs)] p2 hold
- queue: experts-rega-epilogue-overlap-next-stage [w13 + w2 expert GEMMs (rega consumer schedule across item boundaries: epilogue under the next item's first wgmma)] p3 hold
- queue: experts-rega-w2-epilogue-prefetch-bulk-store-cf [w13 + w2 expert GEMMs (rega w2 consumer epilogue: loads and stores)] p1 hold
- 22 stale holds expired to `done/expired/` (not run)
