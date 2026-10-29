# muse_bf16_proj findings

Regenerated 2026-10-29T222816Z by `churn.py index`. Read this before writing a hypothesis for muse_bf16_proj.

## Claims (from CLAIMS.md, newest per mechanism; tag meanings at the top of CLAIMS.md)

### lead

- [lead] bulk12 m>=8 kernels (b12x / cp.async bridge / TMA-box uc1) with the end-of-kernel serial escape-x gather replaced by batched loads; plain mbarrier.arrive.shared::cluster instead of arrive.release.cluster (MEMBAR.ALL.GPU) in the TMA build | rows: layer=pooler load=verify_1, layer=pooler load=decode_16, layer=pooler load=decode_32, layer=qkv_gate load=verify_1, layer=qkv_gate load=decode_16, layer=qkv_gate load=decode_32 | evidence: H100 b12tma probe (takemyjob hopper/b12tma-1029 d95798e; research/h100-b12tma-1029.md, GPU 2): the serial escape gather costs ~10 us per 8 tokens (25-38% of the kernel); batched: bridge 29.2->21.7 us pooler m8, 44.0->27.4 m16, 22.9->17.6 qkv m8; uc1 TMA boxes 20.7 us pooler m8 vs node cuBLAS bf16 26.1; the 3x TMA stall was 4 x MEMBAR.ALL.GPU per chunk (61-74% of stall samples); mc/uc 0.97-1.02 (multicast adds nothing); projected judge pooler m8 ~0.89-0.94 of the incumbent (untested) | board: 1.2794 | blob: a36f3392 | reopen: judge diag/race of the batched-gather b12 kernels per m>=8 cell | 2026-10-29 h100

### shelf

- [shelf] b12r: resident x at m=2..8 (pooler m=8 judged x1.016, 19.07 -> 18.68 us) | rows: layer=pooler load=verify_1 | evidence: 6617f94d 1.2528 (DISTILL:180) | board: 1.2664 | blob: - | reopen: only inside a stack, alone among the m=8 cells, under a read-flush gate (worth < 0.1%) | 2026-10-26 campaign (re-measure: measured on 1.2664, best now 1.2839)
- [shelf] unpooler slice m=1 as a bulk12 row-owner cell (U12s) | rows: layer=unpooler load=decode_1 | evidence: a642424734f54365 1.2737: gated read x1.024 vs ouns, not routed (LOG:1147); its rows never ran on the judge | board: 1.2735 | blob: - | reopen: - | 2026-10-27 churn
- [shelf] tile-contiguous weight repack for the swap-AB m<=32 GEMMs (+2.1-3.4% kernel under the read flush, 3.72 -> 3.86 TB/s), decode-portable | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: pc-diag-pattern bf87ac81 tiled / orig 0.966 / 0.979 / 0.978 at m = 8 / 16 / 32 (LOG:2004-2006) | board: 1.2740 | blob: - | reopen: - | 2026-10-28 campaign
- [shelf] cuBLASLt config sweep at m=256 beyond the heuristic (lt-config-sweep-m256-beyond-heuristic), est ~0.5% (own Risk +0.4-0.7%) | rows: load=verify_32 | evidence: DECISIONS:412 | board: 1.2794 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] pooler m=128/256 K-halves on a prepare-picked cuBLASLt batched plan (pooler-khalf-lt-batched-plan-m128-m256), est ~0.5% | rows: layer=pooler load=verify_16, layer=pooler load=verify_32 | evidence: DECISIONS:413 | board: 1.2794 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] router hilo launch config per m from a read-flush sweep at m=65..256 (router-hilo-m128-m256-readflush-config-sweep), est ~0.2%; its predecessor tall-tile is refuted above | rows: layer=router load=verify_16, layer=router load=verify_32 | evidence: DECISIONS:414 | board: 1.2794 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] qkv_gate swap-AB m=8..32 runs 0.9-1.6 us above the o/pooler swap-AB latency+bandwidth fit (t = 2.3 us + bytes / 3.93 TB/s: qkv predicted 11.9 us vs 12.8 / 12.9 / 13.5 us at m=8 / 16 / 32); closing it is worth <= ~1.0% headline | rows: layer=qkv_gate load=verify_1, layer=qkv_gate load=decode_16, layer=qkv_gate load=decode_32 | evidence: board 3e3ffc5afc57433b row times (o d16 6.6 us / 16.8 MB, pooler d16 19.4 us / 67.1 MB) | board: 1.2794 | blob: a36f3392 | reopen: only inside a stack; swap-AB split-K/stage retunes are refuted (CLAIMS:57) | 2026-10-29 screen

### unresolved

- [unresolved] bf16 board level: the 1.2740 board is the top of a ~1.267-1.274 spread; per-row noise between identical bytes is 1-5% | rows: all | evidence: pc-pin-rpdl-all-khalf bcc4babe 1.2671, pc-diag-pattern bf87ac81 1.2692, pc-ouns-cl4 5ff48999 1.2713 (LOG:1987, 2002-2003) | board: 1.2740 | blob: - | reopen: pin flipping races only for determinism; judge single-row effects under ~3-5% on >= 2 runs | 2026-10-28 campaign
- [unresolved] CW=4 b12 runner-ups for pooler / o (pc-b12-cw4-pooler-o), inside the board-routing spread | rows: layer=pooler load=decode_1, layer=o load=decode_1 | evidence: phase-C check 1.2786 vs the 1.274 board (LOG:2012-2013) | board: 1.2794 | blob: - | reopen: stack with the qkv m=128 plan and judge on >= 2 runs | 2026-10-28 campaign
- [unresolved] cuBLASLt NN-layout plan for qkv_gate at m=128 (the untried half of lt-nn-layout-cells-o-m128-qkv-m128), at most ~0.35% headline | rows: layer=qkv_gate load=verify_16 | evidence: same review; the 62da2d9e plan already re-ranks qkv m=128 TN under the read flush | board: 1.2794 | blob: - | reopen: a prepare-only NN vs TN ranking under the read flush shows a >= 10% kernel cut | 2026-10-28 churn
- [unresolved] bf16-shelf-stack: tile-contiguous repack + L/KL Lt plans + b12 CW=4 cells, each gated (routed: tile pooler 8/16/32, o_16, unpooler_32; L m=256 qkv_gate, pooler; KL pooler m128; b12 o) | rows: load=verify_1 load=verify_16 load=verify_32 | evidence: check f37f01e7884e46b1 1.2810 vs board 1.2794 (+0.1%, inside the 1.24% margin) | board: 1.2794 | blob: a36f3392 | reopen: only as part of a bigger bf16 stack; a second run would only pool noise | 2026-10-28 campaign

### landed

- [landed] router GEMM as a fused Triton hi/lo kernel with split-K control at every m | rows: layer=router | evidence: promote dba4ffd5 (job bd9fa14151644cee 1.1510) | board: 1.1510 | blob: da55301d | reopen: - | 2026-10-23 churn
- [landed] shape-gated Triton winners combo, then swap-AB wgmma Triton GEMMs at m<=32 and the m=16 swap-AB / router swap-AB m=32 restack | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: promotes 658d63bc (875f3e0172ee48a1 1.1760), 8b9ad430 (1602a4e947e64090 1.1984), b99b1dd3 (1e90292c7b954d96 1.2215) | board: 1.2215 | blob: 970b415b | reopen: - | 2026-10-24 churn
- [landed] three-cell restack: router and unpooler-slice cluster DSMEM kernels at m<=8 and the cuBLAS NN layout for pooler / unpooler at m=65..128, then the pooler NN layout widened to m=256 (pooler-nn-layout-widen-m256 also sits on the Do-not-try list; it is on master) | rows: layer=router load=decode_1, layer=router load=verify_1, layer=unpooler load=decode_1, layer=unpooler load=verify_1, layer=pooler load=verify_16, layer=unpooler load=verify_16, layer=pooler load=verify_32 | evidence: promotes 040fc7e1 (84f17adc3f074a33 1.2366), 67b46a57 (85c91578cc184287 1.2394; FINDINGS:211); earlier cluster restack d323ce8ef6a54fde and NN gate 0daf3faa83ab4e81 (FINDINGS:209, 227) are owned by these | board: 1.2394 | blob: f919748b | reopen: - | 2026-10-25 churn
- [landed] bulk row-owner kernel at m=1 (reduce-free) widened to qkv_gate / o, on the naive stack of pooler m=1 row-owner + qkv_gate m=128 cuBLASLt single node | rows: layer=pooler load=decode_1, layer=qkv_gate load=decode_1, layer=o load=decode_1, layer=qkv_gate load=verify_16 | evidence: promote 36c26199 (741072dca5fc4c9d 1.2566); combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt 0ee2055590aa4e84 (FINDINGS:215) is owned by this and 62da2d9e | board: 1.2566 | blob: ad8d7254 | reopen: - | 2026-10-25 churn
- [landed] bulk12: m=1 row-owner kernel streaming a lossless 12-bit weight code (pooler 17.72 -> 14.8 us, ~4.8 TB/s raw-equivalent) | rows: layer=pooler load=decode_1, layer=o load=decode_1, layer=qkv_gate load=decode_1 | evidence: check 474490c8258d4521 1.2699, scored 3834ceeca7c0410d 1.2664 (LOG:500-517); land 5325ea40 | board: 1.2664 | blob: 2bcfe7d4 | reopen: - | 2026-10-26 campaign
- [landed] read-flush gates for every bulk12 cell + pooler m=128 K-halves with fp32 partials (khalf b2: pooler m=128 22.05 -> 21.12 us) | rows: layer=pooler load=verify_16 | evidence: check 670ad7c6b0da41ec 1.2726, scored bc147c5b87bc465d 1.2735 (LOG:939-942); land 8f233e88; pooler-k-halves-fp32-partials-restack de95f81782374124 (FINDINGS:216) is this line | board: 1.2735 | blob: d51c745a | reopen: - | 2026-10-27 campaign
- [landed] router zero-fill PDL overlap at m=9..256 (all four classes) + the pooled khalf gate rule (3 rounds, 2.0x pooled IQR) | rows: layer=router load=decode_16, layer=router load=decode_32, layer=router load=verify_16, layer=router load=verify_32, layer=pooler load=verify_16 | evidence: a20f80486c5b4bc1 (router PDL x1.031-1.101 routed, FINDINGS:218), check 023f4260e86e423e 1.2723, scored 254c83ae55834807 1.2740 (LOG:1283-1302; FINDINGS:219 is this line); land f10f2752 | board: 1.2740 | blob: ada0503d | reopen: - | 2026-10-27 campaign
- [landed] qkv_gate m=128 cuBLASLt plan ranked and gated under the judge's read flush | rows: layer=qkv_gate load=verify_16 | evidence: check 0c9095e2 1.2844, scored 3e3ffc5afc57433b 1.2794 (LOG:2011-2012, 2034); land 62da2d9e | board: 1.2794 | blob: a36f3392 | reopen: - | 2026-10-28 campaign
- [landed] bulk12 m=8 uc1_g1 (TMA boxes, no cluster, batched escape gather, plain mbarrier arrive) at pooler m8 and qkv_gate m8, raced per cell at <= 0.95 (b12-batched-escape-m8-32-race) | rows: layer=pooler load=verify_1, layer=qkv_gate load=verify_1 | evidence: check bced57b42b87444d 1.2871 + confirm c0f8969e76cf4cb7 1.3076 (pooled 1.2974): pooler m8 0.837 / 0.827, qkv m8 0.928; judge gather cost GA0-GA1 3.84-4.16 us (pooler m8), 2.43 (qkv m8); scored 14ea7be493204308 1.2839 [1.265, 1.303]; pooler m16 diag 0.923-0.936 (routable next) | board: 1.2839 | blob: 5ef62a20 | reopen: - | 2026-10-29 churn

### refuted

- [refuted] skinny Triton GEMM / GEMV replacements at m<=32 (pooler, o, unpooler full and slice, qkv_gate skinny m32; block-m 16 control) | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: 9f0ad00bbc874d31 0.9983, d75669d46acb42a5 1.0057, 635c2693bd4147a4 1.0053, 09abdd472faa4dce 0.9990 (neutral); abcd45776f244c8c 0.9238, 4c2b113f81434570 0.8801 (regressions) (FINDINGS:199, 201-203, 221, 222) | board: 1.1510 | blob: 72d10599 | reopen: - | 2026-10-24 churn
- [refuted] split-K Triton GEMMs with a semaphore reduce (qkv_gate split-K 4, pooler split-K 2 BLOCK_K 256 control, pooler/unpooler m=128 wgmma split-K) | rows: layer=qkv_gate, layer=pooler, layer=unpooler | evidence: 135ef6499875416a 1.1431, 75d595a29a4c4b57 1.1620 (neutral), d34fd76f9c444728 1.1158 (regression) (FINDINGS:204, 206, 223) | board: 1.1433 | blob: 0b015466 | reopen: - | 2026-10-24 churn
- [refuted] router GEMM retiles beyond the landed hilo / swap-AB / cluster routes: addmm epilogue, split-K 32 with all loads in flight, swap-AB wgmma at every m, tall hilo tile at m=65..256 | rows: layer=router | evidence: 743020b6b43d458c 1.0256, 1e339074b1844a43 1.1427, 587e340476944b46 1.1951, 864e8094b5bc4c1b 1.2539 vs 1.2566 (FINDINGS:198, 205, 208, 213); triton-hilo-fused without split-K control c2a8de5969374dff 0.9402 (FINDINGS:220) | board: 1.2566 | blob: 07996bb0 | reopen: - | 2026-10-25 churn
- [refuted] K-split as a batched cuBLAS GEMM for qkv and unpooler at m=65..256 (the pooler m=128 form landed as khalf b2) | rows: layer=qkv_gate load=verify_16, layer=qkv_gate load=verify_32, layer=unpooler load=verify_16 | evidence: e82feba3dd8d4efb 1.2452, 9d93c3d0cc184226 1.2410 (regression), pooler m=65..256 ce808e5a055f41b4 1.2470 (FINDINGS:212, 214, 230) | board: 1.2566 | blob: 76386bd8 | reopen: - | 2026-10-25 churn
- [refuted] cuBLASLt algo / plan selection timed under the write flush (backend toggle; in-prepare algo pick at m=128/256; per-cell plan selector for o, qkv m=256, unpooler) | rows: load=verify_16, load=verify_32 | evidence: 5ed9ace10d9a4f68 1.1505, 26a8be34a5eb4426 1.1879, 37ed8722bc9b402d 1.2525 vs 1.2566 (FINDINGS:200, 207, 217) | board: 1.2566 | blob: 1950f143 | reopen: rank per cell under the read flush, as the qkv m=128 plan did in 62da2d9e | 2026-10-26 churn
- [refuted] router ticket: no zero-fill memset at m>=9, a last-arriver epilogue instead | rows: layer=router load=decode_16, layer=router load=decode_32, layer=router load=verify_16, layer=router load=verify_32 | evidence: 7e8be36a: swap-AB m=16 x1.073, hilo m=128 x1.103 slower (DISTILL:178, 268) | board: 1.2664 | blob: - | reopen: - | 2026-10-26 campaign
- [refuted] b12x at m>=8: x streamed with each 12-bit chunk (per-row TMA copies, then L2 hot-spotting) | rows: load=verify_1, load=decode_16, load=decode_32 | evidence: 22ddaf4a, 0acdca48, 2980f57a; rotation took qkv m=8 only to gate 0.983 (DISTILL:179; LOG:529-560); many small cp.async.bulk copies cost 30-50 ns each, serialized (DISTILL:263) | board: 1.2664 | blob: - | reopen: x read once per cluster (TMA multicast over 2-4 CTAs, one box per chunk) and a read-flush gate at <= 0.95 | 2026-10-26 campaign
- [refuted] bf16 partials in K-splits (rules: fp32 accumulation; SGLang's split-K keeps fp32) | rows: all | evidence: DISTILL:272, 186 | board: 1.2664 | blob: - | reopen: - | 2026-10-26 campaign
- [refuted] prepare-time gates timed under the write flush (they overstate small-GEMM times 20-30%; qkv by 7 points) | rows: all | evidence: a3dcd4cb46a84648 read vs write at m=1: pooler 0.833 / 0.828, qkv 0.953 / 0.884, o 0.929 / 0.873 (LOG:860-862); LOG:2016; DISTILL:265 | board: 1.2664 | blob: - | reopen: - | 2026-10-26 campaign
- [refuted] raw-byte dense engines against incumbents already at 3.4-3.9 TB/s graph-timed (Engine D style) | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: f59cb301 P x0.923, qkv x0.804, o x0.530 at T=1 in decode (DISTILL:177, 270) | board: 1.2664 | blob: - | reopen: fewer bytes (bulk12) or a launch hidden behind its predecessor's PDL chain | 2026-10-26 campaign
- [refuted] swap-AB split-K variants at m=8..32: qkv one-wave split-K 7, o split-K 2 at two CTAs/SM, qkv split-K 8 with 3 stages, a 5-stage qkv ring at m=2..16, unpooler full m=16 swap-AB route | rows: layer=qkv_gate, layer=o, layer=unpooler | evidence: c7138fe05598428e 1.2229 (neutral), 75f09fa68f9c4415 1.1860, a61778a08ec04ea6 1.1963, 05782267384343eb 1.1885 (regressions), c65ffcb6d9ab4f95 1.2507 (S5 never routed, read 0.95-0.97; its drop was drift, LOG:1165-1167, 1194) (FINDINGS:210, 224-226, 231) | board: 1.2735 | blob: 40dfcaff | reopen: - | 2026-10-27 churn
- [refuted] DeepGEMM bf16 kernels at m=65..256 (NT route; sm90 s3 tile-count variants) | rows: load=verify_16, load=verify_32 | evidence: 2d522d47ca5c4fc9 1.2027 (regression, FINDINGS:229); 6407497bec88470e 1.2741: every cell lost or tied cuBLAS under the read flush (pooler m=256 30.2 vs 28.7 us, qkv m=256 x0.47-0.77) (LOG:1255-1257) | board: 1.2740 | blob: 94b37346 | reopen: - | 2026-10-27 churn
- [refuted] qkv m=1 bulk12 at R=32, one CTA per SM | rows: layer=qkv_gate load=decode_1 | evidence: a3dcd4cb46a84648 R=32 read gate 1.016 (LOG:861); lost again in a642424734f54365 (LOG:1155) | board: 1.2735 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] cuBLASLt NN-layout plan for o at m=128 (lt-nn-layout-cells-o-m128-qkv-m128, o half) | rows: layer=o load=verify_16 | evidence: review hold 2026-10-28T145555Z (queue/2026-10-26T221237Z-lt-nn-layout-cells-o-m128-qkv-m128.md, Adversarial review): NN 128x64 ran 5.5 us in 37ed8722 vs the TN plan's 5.4 us on 3e3ffc5a; the claimed 7.68 -> 7.24 us was SGLang drift (7.33-7.68 us) | board: 1.2794 | blob: - | reopen: - | 2026-10-28 churn
- [refuted] b12mc: bulk12 row-owner with x as 2D tensor-map TMA boxes in 4-CTA clusters, multicast vs per-CTA (pooler m8/m16, qkv m8) | rows: layer=pooler load=verify_1, layer=pooler load=decode_16, layer=qkv_gate load=verify_1 | evidence: check 5f35efd59d054392 (read flush, 2 rounds, 2 CTA/SM instances with 62 clusters, 0 spill): pooler m8 inc 17.9 / mc 72.8 / uc 72.7 / cp.async bridge 24.3 us; pooler m16 inc 18.1 / mc,uc 78.5; qkv m8 inc 11.6 / mc,uc 35.3; mc/uc 0.995-1.007 (multicast adds nothing); pre-registered REFUTE (b8 4.06 >= 1.0). Caveat: the TMA arms ran 3x slower than the cp.async bridge, so the build is stalled somewhere and mc/uc was read on a stalled kernel | board: 1.2794 | blob: a36f3392 | reopen: a TMA-box build at <= the cp.async bridge (24.3 us at pooler m8) that then shows mc/uc <= 0.9 | 2026-10-29 churn

### superseded

- [superseded] lossless exp4 compressed-weight stream kernel (4 warps, 1 CTA/SM, escape groups) | rows: load=decode_1 | evidence: 9148142cee4441ad 1.1904 (FINDINGS:228); DISTILL:184 (1.4-1.6 TB/s); bulk12 is the working 12-bit form (5325ea40) | board: 1.2215 | blob: 11c9d817 | reopen: - | 2026-10-26 campaign
- [superseded] bf16 sprint combo (khalf + qkv row-owner m=8/16 + Lt) | rows: layer=pooler load=verify_16, layer=qkv_gate load=verify_1 | evidence: DISTILL:186 (its qkv cells never routed; khalf kind (a) rounds fp32 partials to bf16); superseded by khalf b2 (8f233e88) | board: 1.2664 | blob: - | reopen: - | 2026-10-26 campaign

## Current best

1.2839x [1.2646,1.3031] from job `14ea7be493204308` (submit), master commit `51c4c3d8c8`, hypothesis b12-batched-escape-m8-32-race

Where the time goes (per scored shape, top kernels):

- layer=qkv_gate load=decode_1 m=1: yours 10.6µs vs SGLang 14.5µs (SoL 7.9µs) · void b12_kernel<16, 8192, 6, 2, 4>(unsigned char const*, ... 9.2µs 100.0%
- layer=o load=decode_1 m=1: yours 5.5µs vs SGLang 6.6µs (SoL 3.5µs) · void b12_kernel<64, 1024, 8, 1, 8>(unsigned char const*, ... 4.2µs 100.0%
- layer=pooler load=decode_1 m=1: yours 15.0µs vs SGLang 21.8µs (SoL 14.0µs) · void b12_kernel<32, 8192, 8, 1, 8>(unsigned char const*, ... 13.9µs 100.0%
- layer=unpooler load=decode_1 m=1: yours 5.0µs vs SGLang 7.0µs (SoL 1.7µs) · void (anonymous namespace)::ouns_kernel<8, 64, 512, 8, 4>... 4.5µs 100.0%
- layer=router load=decode_1 m=1: yours 5.3µs vs SGLang 10.6µs (SoL 1.8µs) · void (anonymous namespace)::router_kernel<8, 8>(__nv_bflo... 3.7µs 100.0%
- layer=qkv_gate load=decode_16 m=16: yours 12.9µs vs SGLang 13.9µs (SoL 7.9µs) · _bf16_swapab_kernel 10.1µs 100.0%
- layer=o load=decode_16 m=16: yours 6.3µs vs SGLang 7.0µs (SoL 3.6µs) · _bf16_swapab_kernel 4.3µs 100.0%
- layer=pooler load=decode_16 m=16: yours 19.6µs vs SGLang 21.7µs (SoL 14.1µs) · _bf16_swapab_kernel 19.6µs 100.0%
- layer=unpooler load=decode_16 m=16: yours 18.7µs vs SGLang 18.5µs (SoL 14.1µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 18.5µs 100.0%
- layer=router load=decode_16 m=16: yours 5.3µs vs SGLang 14.7µs (SoL 1.8µs) · _router_swapab_kernel 3.3µs 79.8%, _router_zero_pdl_kernel 0.8µs 20.2%
- layer=qkv_gate load=decode_32 m=32: yours 14.4µs vs SGLang 14.5µs (SoL 8.0µs) · _bf16_swapab_kernel 11.7µs 100.0%
- layer=o load=decode_32 m=32: yours 6.7µs vs SGLang 6.9µs (SoL 3.6µs) · _bf16_swapab_kernel 4.6µs 100.0%
- layer=pooler load=decode_32 m=32: yours 20.2µs vs SGLang 22.1µs (SoL 14.1µs) · _bf16_swapab_kernel 20.4µs 100.0%
- layer=unpooler load=decode_32 m=32: yours 18.7µs vs SGLang 18.8µs (SoL 14.1µs) · _bf16_swapab_kernel 19.0µs 100.0%
- layer=router load=decode_32 m=32: yours 5.9µs vs SGLang 15.4µs (SoL 1.9µs) · _router_swapab_kernel 3.8µs 82.2%, _router_zero_pdl_kernel 0.8µs 17.8%
- layer=qkv_gate load=verify_1 m=8: yours 11.7µs vs SGLang 13.9µs (SoL 7.9µs) · void (anonymous namespace)::b12m_tma_kernel<16, 8192, 4, ... 11.5µs 100.0%
- layer=o load=verify_1 m=8: yours 6.2µs vs SGLang 6.8µs (SoL 3.5µs) · _bf16_swapab_kernel 4.1µs 100.0%
- layer=pooler load=verify_1 m=8: yours 16.0µs vs SGLang 21.7µs (SoL 14.0µs) · void (anonymous namespace)::b12m_tma_kernel<32, 8192, 6, ... 16.1µs 100.0%
- layer=unpooler load=verify_1 m=8: yours 5.6µs vs SGLang 7.3µs (SoL 1.8µs) · void (anonymous namespace)::ouns_kernel<8, 64, 512, 8, 4>... 4.7µs 100.0%
- layer=router load=verify_1 m=8: yours 5.3µs vs SGLang 26.3µs (SoL 1.8µs) · void (anonymous namespace)::router_kernel<8, 8>(__nv_bflo... 4.0µs 100.0%
- layer=qkv_gate load=verify_16 m=128: yours 15.7µs vs SGLang 17.1µs (SoL 8.4µs) · nvjet_sm90_tst_40x64_64x16_4x2_h_bz_TNN 16.1µs 100.0%
- layer=o load=verify_16 m=128: yours 7.4µs vs SGLang 7.3µs (SoL 4.0µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 5.4µs 100.0%
- layer=pooler load=verify_16 m=128: yours 22.3µs vs SGLang 25.2µs (SoL 14.6µs) · nvjet_sm90_tss_128x64_64x8_1x2_h_bz_NNT 21.7µs 91.9%, _add2_cast_kernel 1.9µs 8.1%
- layer=unpooler load=verify_16 m=128: yours 19.3µs vs SGLang 19.8µs (SoL 14.6µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT 20.2µs 100.0%
- layer=router load=verify_16 m=128: yours 7.6µs vs SGLang 16.5µs (SoL 2.2µs) · _router_hilo_kernel 5.2µs 85.8%, _router_zero_pdl_kernel 0.9µs 14.2%
- layer=qkv_gate load=verify_32 m=256: yours 20.0µs vs SGLang 19.6µs (SoL 9.8µs) · nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN 20.1µs 100.0%
- layer=o load=verify_32 m=256: yours 9.7µs vs SGLang 9.4µs (SoL 4.5µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 8.3µs 100.0%
- layer=pooler load=verify_32 m=256: yours 30.1µs vs SGLang 30.5µs (SoL 17.4µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT 30.0µs 100.0%
- layer=unpooler load=verify_32 m=256: yours 24.2µs vs SGLang 24.0µs (SoL 17.4µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 25.6µs 100.0%
- layer=router load=verify_32 m=256: yours 9.3µs vs SGLang 17.8µs (SoL 2.7µs) · _router_hilo_kernel 7.0µs 88.7%, _router_zero_pdl_kernel 0.9µs 11.3%

Diagnosis of 2026-10-25T203424Z, written on board 1.2566 (today 1.2839): historical, re-diagnose before minting from it (`muse_bf16_proj/hypotheses/DIAGNOSIS.md`).

## By area

### bf16 GEMM dispatch
- **sglang-gemm-dispatch-parity** → gain (gain), 1.0022x, mechanism hit 4/4 — gained on router decode_16, decode_32, verify_16, verify_32; o verify_32. Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], 
- **bf16-cublaslt-backend-toggle** → neutral (within-ci), 1.1505x, mechanism flat 1/4 — gained on qkv_gate decode_1, verify_16; pooler decode_32; lost on o decode_1, verify_1; unpooler decode_1, verify_1; router verify_1; qkv_gate verify_32. A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only, call torch.backends.cuda.preferred_blas_library('cublaslt') imme

### bf16 GEMM dispatch (m<=16, swap-AB Triton)
- **bf16-m16-swapab-wgmma-extend** → neutral (unproven-vs-best), 1.2115x, mechanism hit 1/1 — gained on qkv_gate decode_1; o decode_1, decode_16, verify_1; pooler decode_1; unpooler decode_1, verify_1; lost on router decode_16. Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kerne

### bf16 GEMM dispatch (m<=32, lossless-compressed weight stream)
- **bf16-lossless-exp4-weight-stream-rewrite** → regression (regression), 0.7071x, mechanism miss 0/16 — gained on router decode_16; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1, verify_16; pooler decode_1, decode_16, decode_32, verify_1; unpooler decode_1, decode_16, decode_32, verify_1; router decode_1, verify_16. Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: every m<=32 bf16 shape is a pure weight stream (FINDINGS: pooler 19.0-20.5 us kernel

### bf16 GEMM dispatch (m<=8, hand-written CUDA weight stream)
- **bf16-small-m-cuda-gemv-rewrite** → regression (regression), 1.1667x — gained on pooler verify_16; lost on o decode_1, verify_1; router decode_1; qkv_gate verify_1; pooler verify_1; unpooler verify_1. Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; at

### bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection)
- **bf16-cublaslt-algo-pick-in-prepare-m128-m256** → neutral (within-ci), 1.1879x, mechanism flat 0/3 — lost on qkv_gate decode_1, decode_32, verify_1, verify_32; router verify_1. Built on master (658d63b). For the four bf16 layers at exactly cfg m == 128 and cfg m == 256 (loads verify_16 and verify_32; 8 shapes), replace _bf16_gemm_dispa

### bf16 GEMM dispatch (m=128/256, per-cell cuBLASLt plan selector)
- **bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole** → neutral (within-ci), 1.2525x, mechanism miss 3/7 — gained on unpooler verify_1; router verify_1, verify_32; lost on o decode_16; router decode_16; qkv_gate verify_32. DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise master's cell-B machinery (_LT_CPP candidates/run with MuseLtDesc(n, k, m

### bf16 GEMM dispatch (m=129..256, CUTLASS sm90 cooperative split-K)
- **bf16-verify32-cutlass-coop-256x128-splitk-stage2** → neutral (within-ci), 1.2483x, mechanism no-base — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Stage 2 of bf16-verify-streamk-cutlass-wgmma-rewrite (job f5b1361aaa0e48b5), but ported onto CURRENT master (blob 353e3b19, restack-three-cell-wins-router-unpoo

### bf16 GEMM dispatch (m=129..256, persistent stream-K TMA/wgmma)
- **bf16-verify-streamk-cutlass-wgmma-rewrite** → regression (regression), 1.1635x, mechanism miss 0/4 — gained on qkv_gate decode_16, decode_32; router decode_16, decode_32, verify_1; o decode_32; pooler verify_16; lost on qkv_gate decode_1, verify_1, verify_32; o verify_32; pooler verify_32; unpooler verify_32. Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: at m=256 (verify_32, the load furthest from its floor: 92.9 us vs 51.8 floor) all fo

### bf16 GEMM dispatch (m=17..32, swap-AB Triton)
- **bf16-m32-swapab-wgmma-triton** → neutral (unproven-vs-best), 1.1984x, mechanism flat 3/4 — gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_32, verify_1, verify_16; pooler decode_32, verify_1; router verify_1, verify_16. Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-

### bf16 GEMM dispatch (m=65..128, CUTLASS sm90 one-M-tile split-K)
- **bf16-verify16-cutlass-m128-onetile-splitk-rewrite** → neutral (within-ci), 1.2423x, mechanism miss 0/4 — gained on router verify_1; lost on qkv_gate decode_1, decode_16, verify_32; o verify_1, verify_16; pooler verify_32. Built on current master (blob 353e3b19). This is a new line for the m=128 band, which the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (m=129..256 on

### bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)
- **bf16-verify-deepgemm-bf16-gemm-nt-route** → regression (regression), 1.2027x, mechanism flat 1/5 — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; router decode_16, decode_32. Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=1

### bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)
- **bf16-verify-deepgemm-sm90-kernel-asserted-rewrite** → neutral (unproven-vs-best), 1.2221x, mechanism flat 2/4 — gained on unpooler verify_1; router verify_1; lost on router decode_16, verify_16; o decode_32. Built on master. bf16-verify-deepgemm-bf16-gemm-nt-route (2d522d47) printed PATH CHOSEN deepgemm for all 16 (layer, m) pairs, every self-check at 0.0 s (no JIT 
- **bf16-verify-deepgemm-sm90-explicit-instantiation-stage2** → neutral (unproven-vs-best), 1.2606x, mechanism miss 4/6 — gained on qkv_gate decode_1, decode_32, verify_16; pooler decode_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32; o decode_32; lost on o verify_16, verify_32. DIAGNOSIS Action 5. Built on master 36c26199, porting stage 1's [dg] scaffold (patch done/neutral/bf16-verify-deepgemm-sm90-kernel-asserted-rewrite.patch: TORCH
- **bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush** → neutral (unproven-vs-best), 1.2741x, mechanism flat 1/3 — gained on pooler decode_1, verify_16; qkv_gate verify_16; o verify_16; router verify_16, verify_32; lost on unpooler decode_1; router decode_1; o decode_16, decode_32; pooler decode_16. Stage 2 (job 566a79e4) measured NOTHING: its generator filled DeepGEMM's sm90_bf16_gemm_impl template by parameter NAME and stopped at 'unknown template paramet

### bf16 GEMM dispatch (m=65..256, swap-AB wgmma + cluster DSMEM split-K)
- **bf16-verify-swapab-wgmma-cluster-dsmem-rewrite** → neutral (within-ci), 1.2155x, mechanism flat 2/5 — gained on router decode_32, verify_32; lost on o decode_1, decode_16, decode_32. Built on master (blob a8bb42a). WHY: the m=128/256 rows are the last ones on cuBLAS and the furthest from their floors in absolute us (master profile 1602a4e947

### bf16 GEMM dispatch (m>=128, weight layout)
- **bf16-m128-nn-weight-layout-probe** → neutral (unproven-vs-best), 1.1580x, mechanism flat 2/3 — gained on pooler verify_16; unpooler verify_16; lost on o decode_1; router verify_1, verify_16; qkv_gate verify_32. Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are where cuBLAS sits furthest from the floor (pooler verify_16 25.2 

### bf16 GEMM dispatch (shape-gated Triton routing, m<=16)
- **bf16-shape-gated-triton-winners-combo** → neutral (unproven-vs-best), 1.1760x, mechanism flat 8/10 — gained on qkv_gate decode_1; o decode_1, decode_16; pooler decode_1, decode_16, verify_1; unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). Combine, behind one shape gate in kernel_fn, the four individually-neu

### bf16 dense GEMM cells at m=8..32 (pooler, qkv_gate, o): bulk12 kernels
- **b12-batched-escape-m8-32-race** → gain (gain-pooled), 1.2974x, mechanism miss 2/10 — gained on unpooler decode_1; qkv_gate verify_1; pooler verify_1; router verify_32; lost on qkv_gate decode_32; o decode_32, verify_32; unpooler verify_1; router verify_1, verify_16. Add bulk12 kernels for the m=8/16/32 cells of pooler, qkv_gate (and o if it fits), each raced per cell in prepare against master's incumbent (swap-AB Triton / c
- **b12-batched-escape-m8-32-race-cf** → gain (gain), 1.3076x — gained on unpooler decode_1, decode_32; router decode_1, decode_16, decode_32, verify_1, verify_16; qkv_gate decode_32, verify_1; o decode_32. Confirmation rerun of b12-batched-escape-m8-32-race: its implementation commit 84995d9e32799d8c940b71c9d279bedaad12e6d1 unchanged except one trailing comment, r

### bulk12 row-owner kernel at m>=8 (x delivery: TMA multicast per 4-CTA cluster)
- **b12mc-multicast-x-pooler-qkv-m8-readflush-diag** → neutral (within-ci), 1.2727x, mechanism flat 0/5 — gained on unpooler decode_1; router decode_1, verify_32; pooler verify_32; lost on qkv_gate decode_32, verify_1; o verify_1; unpooler verify_1; router verify_1. Prepare-only diagnostic. kernel_fn, every routing flag and every state key the timed path reads stay byte-identical to master (62da2d9e, blob a36f3392); the dia

### combination (naive stack + bulk row-owner widened to qkv_gate/o m=1)
- **combo-rowowner-reduce-free-m1-qkv-o-plus-naive** → neutral (unproven-vs-best), 1.2566x, mechanism hit 3/3 — gained on qkv_gate decode_1, verify_16; pooler decode_1, verify_16. Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gate

### combination (pooler m=1 bulk row-owner + qkv_gate m=128 cuBLASLt single-node)
- **combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt** → neutral (within-ci), 1.2448x, mechanism hit 2/2 — gained on pooler decode_1; router decode_32; qkv_gate verify_16; lost on o decode_16. Built on current master (commit 67b46a57, blob f919748b: restack-three-cell-wins + pooler-nn-layout-widen-m256). NAIVE TOP-ROW STACK: the only two cells where a

### lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)
- **bf16-lossless-exp4-occupancy-dequant-once-stage2** → neutral (beats-base-not-best), 0.8870x, mechanism hit 10/10 — gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1; unpooler decode_1, decode_16, decode_32, verify_1; router verify_1. Build on the stage-1 code of bf16-lossless-exp4-weight-stream-rewrite (job 542f2b169c934706; codec 12.17 bits/elem, ratio 0.761, bit-exact; load_inline + per-ro
- **bf16-lossless-exp4-decode-only-throughput-probe** → regression (regression), 1.1904x — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1. A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn routes NOTHING new (byte-identical master dispatch, so the headline shoul

### o GEMM
- **o-triton-skinny-m32** → neutral (within-ci), 1.0057x, mechanism miss 3/4 — gained on qkv_gate decode_1, decode_16; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1; lost on o decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'o' at m <= 32 (n=8192, k=1024, 16.8 MB): in kernel_fn route o at m <= 32 to the same ke

### o GEMM (m=17..32, swap-AB)
- **o-swapab-m32-splitk2-two-ctas-per-sm** → regression (regression), 1.1860x, mechanism miss 0/1 — lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (1

### o GEMM + unpooler GEMM (latency-bound small streams)
- **o-unpooler-slice-cuda-cluster-dsmem-rewrite** → regression (regression), 1.1710x, mechanism miss 2/5 — gained on unpooler decode_1, verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB, floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are th

### pooler GEMM
- **pooler-triton-skinny-m32** → neutral (within-ci), 0.9983x, mechanism miss 3/4 — gained on pooler decode_1, decode_16, verify_1, verify_16; lost on qkv_gate decode_1; unpooler decode_1; pooler decode_32. For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams t
- **pooler-triton-splitk2-blockn64-semaphore** → neutral (unproven-vs-best), 1.1664x, mechanism hit 3/3 — gained on pooler decode_1, decode_16, verify_1, verify_16; qkv_gate decode_32; lost on router verify_1, verify_16; o verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf
- **pooler-triton-splitk2-blockn64-semaphore-control-blockk256** → neutral (within-ci), 1.1620x, mechanism hit 3/3 — gained on pooler decode_1; lost on qkv_gate decode_16, verify_1; router verify_1, verify_16, verify_32. From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each C
- **pooler-splitk2-tma-weight-stream** → neutral (unproven-vs-best), 1.1693x, mechanism hit 3/3 — gained on pooler decode_1, decode_16, decode_32, verify_1; router decode_16; lost on qkv_gate verify_16. From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32: _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2,

### pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)
- **pooler-k-halves-fp32-partials-restack-m128-m256** → neutral (unproven-vs-best), 1.2683x, mechanism flat 1/2 — gained on unpooler decode_1; router decode_1, verify_32; qkv_gate decode_16, decode_32; pooler verify_16; lost on router decode_32. DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256 (done/n
- **pooler-k-halves-fp32-partials-restack-m128-m256-cf** → neutral (within-ci), 1.2590x — gained on o decode_32; router decode_32, verify_1; pooler verify_32; lost on o decode_1; unpooler decode_1; router decode_1; qkv_gate decode_16, decode_32, verify_16. Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one t

### pooler GEMM (m=129..256, cuBLAS NN layout)
- **pooler-nn-layout-widen-m256** → neutral (within-ci), 1.2394x, mechanism hit 1/1 — gained on unpooler decode_32; pooler verify_32; lost on router decode_32, verify_1; unpooler verify_1. Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = 

### pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM)
- **pooler-k-halves-batched-gemm-m65-m256** → neutral (within-ci), 1.2470x, mechanism hit 2/2 — gained on qkv_gate decode_1, decode_16; router verify_1; pooler verify_16, verify_32; lost on router decode_16. Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only (scored rows verify_16 m=128 and verify_32 m=256; no test uses this ba

### pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel)
- **pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite** → regression (regression), 1.1828x, mechanism miss 0/6 — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16; pooler decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32. Built on master (blob a8bb42a). WHY: pooler (n=4096, k=8192) and the full unpooler (n=8192, k=4096) are 67.1 MB pure weight streams at m<=32 with a 14.0-14.1 us
- **pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild** → regression (regression), 1.0331x, mechanism miss 1/6 — gained on qkv_gate decode_1; o decode_1; pooler decode_1; lost on pooler decode_16, decode_32; unpooler decode_16, decode_32, verify_1; router verify_1, verify_16; o verify_16. Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built on its commit (8ba8624, claim bases a rewrite-on-rewrite on the p

### pooler GEMM + unpooler GEMM (m=128)
- **pooler-unpooler-m128-wgmma-splitk-semaphore** → regression (regression), 1.1158x, mechanism miss 0/2 — gained on router decode_16; lost on qkv_gate verify_1; o verify_1; unpooler verify_1, verify_16; router verify_1, verify_16; pooler verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 
- **pooler-unpooler-m128-tma-wspec-rewrite** → regression (regression), 1.1639x, mechanism miss 0/2 — gained on router verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; pooler verify_16; unpooler verify_16; router verify_16. Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler} at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m ma

### pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)
- **pooler-unpooler-m128-nn-layout-gated** → regression (regression), 1.1953x, mechanism hit 2/2 — gained on pooler verify_16; unpooler verify_16; router verify_32; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1; pooler decode_16. Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master) to the only rows it won. Measured per
- **restack-pooler-unpooler-nn-layout-m128** → neutral (unproven-vs-best), 1.2282x, mechanism hit 2/2 — gained on unpooler decode_1, verify_16; router verify_1; pooler verify_16; lost on router decode_16, verify_16; o decode_32. Built on master (b99b1dd, the landed restack). Apply the exact change of branch hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the re

### qkv GEMM
- **qkv-gate-triton-skinny-m32** → regression (regression), 0.9238x, mechanism miss 0/4 — lost on qkv_gate decode_1, decode_16, decode_32, verify_1. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <= 32 (n=2304, k=8192, a 37.7 MB weight): in kernel_fn route qkv_gate at
- **qkv-gate-triton-splitk4-blockn64-semaphore** → neutral (within-ci), 1.1431x, mechanism hit 3/3 — gained on qkv_gate decode_1, decode_16, verify_1; router decode_16; pooler verify_16; lost on qkv_gate decode_32; o verify_1; unpooler verify_1; router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl (nvje

### qkv GEMM (m<=16, swap-AB Triton)
- **qkv-swapab16-five-stage-ring-m8-m16** → regression (regression), 1.2507x, mechanism no-base — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Built on master blob 2bcfe7d4. qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99 us in 3834ceec) runs _bf16_swapab_kernel through _swapab16_forwa

### qkv GEMM (m<=32, swap-AB)
- **qkv-swapab-w128-splitk7-one-wave-m32** → neutral (within-ci), 1.2229x, mechanism miss 0/4 — gained on o verify_32; lost on router decode_16, decode_32, verify_1; qkv_gate decode_32, verify_1; o decode_32. Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, check 84f17adc3f074a33, 1.2366x). WHY: at m<=32 qkv_gate (n=2304, k=

### qkv GEMM (m=17..32, swap-AB)
- **qkv-swapab-m32-splitk8-stages3** → regression (regression), 1.1963x, mechanism miss 0/1 — gained on pooler verify_16; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a). qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 st

### qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM)
- **qkv-k-split-batched-gemm-m65-m256** → regression (regression), 1.2410x, mechanism miss 0/2 — gained on unpooler verify_1; router verify_1; lost on qkv_gate decode_1, decode_16, verify_32; o decode_16; router decode_16, decode_32; pooler verify_32. Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only, which covers the scored rows verify_16 m=128 and verify_32 m=256. T

### qkv GEMM (m=8/16, bulk row-owner kernel)
- **qkv-rowowner-a2-widen-m8-m16** → neutral (unproven-vs-best), 1.2603x, mechanism flat 1/2 — gained on router decode_1, verify_1, verify_32; unpooler verify_1; lost on router decode_16. DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; de

### restack (m<=16 swap-AB GEMMs + router swap-AB)
- **bf16-restack-m16-swapab-router-swapab-m32** → neutral (unproven-vs-best), 1.2215x, mechanism hit 3/3 — gained on o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1. Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a 3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma

### router GEMM
- **router-hilo-at-m8** → gain (gain), 1.0219x, mechanism hit 1/1 — gained on unpooler verify_1; router verify_1. In kernel_fn's router branch, lower the threshold at which the router switches from F.linear(x.float(), router) to the BF16 hi + lo pair: change ROUTER_FP32_MAX
- **router-hilo-addmm-epilogue** → neutral (within-ci), 1.0256x, mechanism flat 0/4 — gained on o verify_16; pooler verify_32. In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc +=
- **router-triton-hilo-fused** → regression (regression), 0.9402x, mechanism miss 4/6 — gained on router decode_1, decode_16, decode_32, verify_1; qkv_gate verify_1; o verify_1, verify_16; unpooler verify_1; lost on router verify_16, verify_32. Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch 
- **router-triton-hilo-fused-control-splitk-all-m** → gain (gain), 1.1510x, mechanism hit 6/6 — gained on router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; lost on qkv_gate decode_1. Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Tr
- **router-splitk32-small-m-all-loads-in-flight** → neutral (within-ci), 1.1427x, mechanism miss 0/4 — gained on pooler verify_16; lost on router decode_1, decode_16, decode_32, verify_1. Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the num_stages they launch with; the kernel body, the torch.zeros output + 
- **router-splitk-semaphore-no-memset** → neutral (unproven-vs-best), 1.1516x, mechanism miss 0/3 — gained on qkv_gate decode_1, verify_1; pooler verify_16; lost on router decode_1, decode_16, decode_32, verify_16. Built on master (dba4ffd). For the router at m <= 32 only (buckets (16,32,16) and (32,32,16): grid (1, 8, 16) = 128 CTAs, unchanged tiles, BLOCK_K=128, num_stag
- **router-splitk32-small-m-all-loads-in-flight-control-stages3** → neutral (unproven-vs-best), 1.1518x, mechanism hit 3/3 — gained on unpooler decode_1; router decode_1, decode_16, decode_32, verify_1; qkv_gate decode_16; lost on o verify_16; pooler verify_16; router verify_16. From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPL
- **router-swapab-wgmma-all-m** → neutral (within-ci), 1.1951x — gained on router decode_1, decode_16, decode_32, verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; router verify_16. Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M
- **router-cuda-cluster-dsmem-splitk-rewrite** → regression (regression), 1.1897x, mechanism miss 2/4 — gained on router decode_1, verify_1; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1, verify_16, verify_32; unpooler decode_1, verify_1; router decode_16, decode_32, verify_16. Built on master (8b9ad43, blob a8bb42a); the restack is not landed and is not needed. WHY: router rows at m<=32 are 5.4-6.2 us against a 1.8-1.9 us floor (8.4 M

### router GEMM (m<=32, cluster DSMEM kernel)
- **router-cluster-dsmem-s2-flat-in-m-occupancy** → neutral (within-ci), 1.2058x, mechanism miss 1/4 — gained on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1, verify_1; router decode_32; lost on router verify_1; pooler verify_16. Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4). Stage 1 won rows at m=1/8 only by deleting the memset node: its

### router GEMM (m<=8)
- **restack-router-cluster-dsmem-m8** → neutral (within-ci), 1.2199x, mechanism hit 2/2 — gained on router decode_1, verify_1; qkv_gate decode_16; lost on o decode_32, verify_16; router verify_16. Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel fr

### router GEMM (m=65..256)
- **router-m128-m256-cluster-dsmem-tiled-rewrite** → neutral (within-ci), 1.2189x, mechanism flat 1/2 — gained on unpooler verify_1; lost on o decode_32. Built on master (the router at m > 32 is identical on master and the restack). Router rows at m=128/256 are 7.3/9.6 us against floors of 2.2/2.7 (3.3-3.6x, the 

### router GEMM (m=65..256, CUDA wgmma swap-AB cluster kernel)
- **router-m128-m256-wgmma-swapab-tall-cluster-stage2** → neutral (unproven-vs-best), 1.2612x, mechanism miss 0/2 — gained on qkv_gate decode_1, decode_16, decode_32, verify_1, verify_16; pooler decode_1, verify_16, verify_32; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1; lost on router verify_32. Stage 2 of router-m128-m256-cluster-dsmem-tiled-rewrite (job b7b2b55d04544660, neutral). Port its load_inline scaffold, self-check, cold-L2 timing table and PAT

### router GEMM (m=65..256, hilo tile config)
- **router-hilo-tall-tile-m65-m256** → neutral (within-ci), 1.2539x, mechanism miss 0/2 — gained on unpooler decode_1; router decode_1, verify_1; lost on router decode_16, verify_32. Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound

### router GEMM (m=9..256, zero-fill node overlapped with the router kernel by PDL)
- **router-zero-pdl-overlap-m9-m256** → neutral (within-ci), 1.2715x, mechanism no-base — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Built on master (8f233e88 lineage). At m > 8 the router is two graph nodes: torch.zeros((m, 256), fp32) (profile: at::native vectorized_elementwise fill, 0.7 us

### router m=9..256 PDL (restack) + pooler m=128 khalf gate rule
- **router-pdl-plus-khalf-pooled-iqr-gate** → neutral (within-ci), 1.2723x, mechanism no-base — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all 4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059

### shape-gated restack: router GEMM m<=8 + unpooler-slice GEMM m<=8 + pooler/unpooler GEMM m=65..128
- **restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128** → neutral (unproven-vs-best), 1.2366x, mechanism hit 6/6 — gained on unpooler decode_1, verify_1; router decode_1, decode_32, verify_1; pooler verify_16; lost on o verify_32. Built on current master (b99b1dd, bf16-restack-m16-swapab-router-swapab-m32, check 1e90292c7b954d96, 1.2215). Stack the three row wins that were each checked al

### unpooler GEMM
- **unpooler-slice-triton-gemv** → regression (regression), 0.8801x, mechanism miss 0/2 — gained on o verify_1; lost on unpooler decode_1, verify_1. For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call i
- **unpooler-full-triton-skinny-m32** → neutral (within-ci), 1.0053x, mechanism miss 1/2 — gained on qkv_gate decode_1, decode_16, verify_1; unpooler decode_16, verify_1; o verify_1; lost on unpooler decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route
- **unpooler-full-triton-skinny-m32-control-blockm16-all-m** → neutral (within-ci), 0.9990x, mechanism miss 0/1 — gained on o verify_16; lost on qkv_gate decode_16; unpooler decode_32. From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num
- **unpooler-slice-triton-splitk8-semaphore** → neutral (unproven-vs-best), 1.1546x, mechanism hit 2/2 — gained on unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only,
- **unpooler-full-m16-swapab-route** → regression (regression), 1.1885x, mechanism flat 0/1 — gained on router verify_32; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; pooler decode_1, decode_16, verify_1; router decode_1, decode_16, decode_32, verify_1; unpooler verify_1. Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (c

### unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM)
- **unpooler-k-halves-batched-gemm-m65-m128** → neutral (within-ci), 1.2452x, mechanism flat 0/1 — gained on unpooler verify_1; router verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; router decode_16; pooler verify_16. Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192) and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=1

### unpooler GEMM (sliced n=1024, m<=8)
- **restack-unpooler-slice-cluster-dsmem-m8** → neutral (unproven-vs-best), 1.2260x, mechanism hit 2/2 — gained on unpooler decode_1, verify_1; router verify_1; lost on o verify_16; router verify_16. Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-

### unpooler GEMM (sliced n=1024, m=1, bulk12 12-bit row-owner)
- **b12-unpooler-slice-m1-r16-rowowner** → neutral (unproven-vs-best), 1.2737x, mechanism no-base — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024 rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. C

## Open follow-ups

- bf16-lossless-exp4-occupancy-dequant-once-stage2: rewrite stage 2 of `Triton swap-AB m<=32 kernels (bf16-m32-swapab-wgmma-triton) on pooler m 1-32, unpooler-full m 16/32, qkv_gate m 1-32`: read the replaced kernels' own us per shape (profile top_kernels) against the inc
- bf16-verify-swapab-wgmma-cluster-dsmem-rewrite: rewrite stage 1 of `cuBLAS nvjet at m=128/256 for qkv_gate (13.0+3.0 splitKreduce / 19.4 us), pooler (25.7 / 29.5 us) and unpooler m=128 (20.4 us)`: read the replaced kernels' own us per shape (profile top_kernels) again
- bf16-verify-deepgemm-sm90-kernel-asserted-rewrite: rewrite stage 1 of `cuBLAS nvjet (+cublasLt splitKreduce) for qkv_gate, pooler, unpooler at m=65..256`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check whi
- router-cluster-dsmem-s2-flat-in-m-occupancy: rewrite stage 2 of `_router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor
- router-m128-m256-cluster-dsmem-tiled-rewrite: rewrite stage 1 of `_router_hilo_kernel + torch.zeros memset at router m=65..256`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reach
- bf16-verify32-cutlass-coop-256x128-splitk-stage2: rewrite stage 2 of `cuBLAS nvjet at m=129..256 (verify_32) for qkv_gate (nvjet 96x64 2x4 TNN 19.4 us), o (nvjet 128x128 TNT 8.1 us), pooler (nvjet 128x64 TNT 29.7 us) and unpooler (nvjet 128x128 TNT 24.8 us)`: read the r
- bf16-verify16-cutlass-m128-onetile-splitk-rewrite: rewrite stage 1 of `cuBLAS at m=65..128 (verify_16): qkv_gate nvjet 96x64 splitK 12.7 + cublasLt splitKreduce 3.1 us, o nvjet 128x64 TNT 5.5 us, pooler NN nvjet 128x64 splitK 20.2 + splitKreduce 3.0 us, unpooler NN nvjet
- router-m128-m256-wgmma-swapab-tall-cluster-stage2: rewrite stage 2 of `_router_hilo_kernel + torch.zeros memset at router m=65..256 (hilo 5.0 + 0.8 us at m=128, 7.0 + 0.7 us at m=256)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbe
- bf16-verify-deepgemm-sm90-explicit-instantiation-stage2: rewrite stage 2 of `cuBLAS nvjet (+splitKreduce) for qkv_gate, pooler, unpooler at m=128/256`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milest
- bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush: rewrite stage 3 of `cuBLAS nvjet at m=128/256 for qkv_gate (Lt 40x64 15.7 us at m=128; 96x64 2x4 TNN 19.6 us at m=256), pooler m=256 (NN 128x64 29.0 us) and unpooler (NN 128x64 19.1 us at m=128; 128x128 TNT 24.7 us at m=

## Shelf (row wins below the headline bar; stack them)

Neutral results whose predicted rows all moved past ±2% (one run each: the rows drift 1.5-2.5%, so re-measure on today's master before stacking).

- **pooler-triton-splitk2-blockn64-semaphore** [pooler GEMM] unproven-vs-best, 1.1664x, rows 3/3 hit, job `08ec203e5717406e`: For layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf16), replace _bf16_g
- **qkv-gate-triton-splitk4-blockn64-semaphore** [qkv GEMM] within-ci, 1.1431x, rows 3/3 hit, job `135ef6499875416a`: For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16
- **unpooler-slice-triton-splitk8-semaphore** [unpooler GEMM] unproven-vs-best, 1.1546x, rows 2/2 hit, job `bb31bb36545a47bb`: For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf1
- **pooler-triton-splitk2-blockn64-semaphore-control-blockk256** [pooler GEMM] within-ci, 1.1620x, rows 3/3 hit, job `75d595a29a4c4b57`: From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one co
- **pooler-splitk2-tma-weight-stream** [pooler GEMM] unproven-vs-best, 1.1693x, rows 3/3 hit, job `4e89311eefdc4bc3`: From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32
- **router-splitk32-small-m-all-loads-in-flight-control-stages3** [router GEMM] unproven-vs-best, 1.1518x, rows 3/3 hit, job `4685bac7a662454d`: From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything ex
- **bf16-m16-swapab-wgmma-extend** [bf16 GEMM dispatch (m<=16, swap-AB Triton)] unproven-vs-best, 1.2115x, rows 1/1 hit, job `89fabef70bd745f7`: Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler 
- **restack-pooler-unpooler-nn-layout-m128** [pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)] unproven-vs-best, 1.2282x, rows 2/2 hit, job `92cd4e758a924ca0`: Apply the exact change of branch hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (
- **restack-router-cluster-dsmem-m8** [router GEMM (m<=8)] within-ci, 1.2199x, rows 2/2 hit, job `d323ce8ef6a54fde`: Port ONLY the router cluster kernel from branch hyp/muse_bf16_proj/router-cuda-cluster-dsm
- **restack-unpooler-slice-cluster-dsmem-m8** [unpooler GEMM (sliced n=1024, m<=8)] unproven-vs-best, 1.2260x, rows 2/2 hit, job `e557d92969864449`: Port ONLY the unpooler-slice half of the oun kernel from branch hyp/muse_bf16_proj/o-unpoo
- **pooler-k-halves-batched-gemm-m65-m256** [pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM)] within-ci, 1.2470x, rows 2/2 hit, job `ce808e5a055f41b4`: Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only 
- **combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt** [combination (pooler m=1 bulk row-owner + qkv_gate m=128 cuBLASLt single-node)] within-ci, 1.2448x, rows 2/2 hit, job `0ee2055590aa4e84`: Built on current master (commit 67b46a57, blob f919748b: restack-three-cell-wins + pooler-

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end. Neutrals whose predicted rows all moved are on the Shelf instead.

- **router-hilo-addmm-epilogue** [router GEMM] neutral, within-ci: In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accum
- **pooler-triton-skinny-m32** [pooler GEMM] neutral, within-ci: For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kerne
- **bf16-cublaslt-backend-toggle** [bf16 GEMM dispatch] neutral, within-ci: A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers 
- **o-triton-skinny-m32** [o GEMM] neutral, within-ci: Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'o' at m <= 32 (n=
- **unpooler-full-triton-skinny-m32** [unpooler GEMM] neutral, within-ci: Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in
- **unpooler-full-triton-skinny-m32-control-blockm16-all-m** [unpooler GEMM] neutral, within-ci: From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, gri
- **router-splitk32-small-m-all-loads-in-flight** [router GEMM] neutral, within-ci: Change only the two small-m buckets of _router_config and the num_stages they launch with;
- **bf16-cublaslt-algo-pick-in-prepare-m128-m256** [bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection)] neutral, within-ci: For the four bf16 layers at exactly cfg m == 128 and cfg m == 256 (loads verify_16 and ver
- **router-swapab-wgmma-all-m** [router GEMM] neutral, within-ci: Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, B
- **qkv-swapab-w128-splitk7-one-wave-m32** [qkv GEMM (m<=32, swap-AB)] neutral, within-ci: WHY: at m<=32 qkv_gate (n=2304, k=8192, a 37.7 MB weight, cold L2 every replay) is the one
- **pooler-nn-layout-widen-m256** [pooler GEMM (m=129..256, cuBLAS NN layout)] neutral, within-ci: Master already builds state['pooler_t'] = W_pooler.t().contiguous() ([8192, 4096] bf16, lo
- **router-hilo-tall-tile-m65-m256** [router GEMM (m=65..256, hilo tile config)] neutral, within-ci: Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo
- **unpooler-k-halves-batched-gemm-m65-m128** [unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM)] neutral, within-ci: Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8
- **pooler-k-halves-fp32-partials-restack-m128-m256** [pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)] neutral, not confirmed (pooled 1.2636332647989572): DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from b
- **bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole** [bf16 GEMM dispatch (m=128/256, per-cell cuBLASLt plan selector)] neutral, within-ci: DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise mas
- **router-zero-pdl-overlap-m9-m256** [router GEMM (m=9..256, zero-fill node overlapped with the router kernel by PDL)] neutral, within-ci: At m > 8 the router is two graph nodes: torch.zeros((m, 256), fp32) (profile: at::native v
- **router-pdl-plus-khalf-pooled-iqr-gate** [router m=9..256 PDL (restack) + pooler m=128 khalf gate rule] neutral, within-ci: Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED i
- **b12mc-multicast-x-pooler-qkv-m8-readflush-diag** [bulk12 row-owner kernel at m>=8 (x delivery: TMA multicast per 4-CTA cluster)] neutral, within-ci: Prepare-only diagnostic. kernel_fn, every routing flag and every state key the timed path 
- **router-triton-hilo-fused** [router GEMM] regression, regression: Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the 
- **unpooler-slice-triton-gemv** [unpooler GEMM] regression, regression: For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight
- **qkv-gate-triton-skinny-m32** [qkv GEMM] regression, regression: Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <=
- **pooler-unpooler-m128-wgmma-splitk-semaphore** [pooler GEMM + unpooler GEMM (m=128)] regression, regression: For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 65 <= m <= 128 may t
- **o-swapab-m32-splitk2-two-ctas-per-sm** [o GEMM (m=17..32, swap-AB)] regression, regression: o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BL
- **qkv-swapab-m32-splitk8-stages3** [qkv GEMM (m=17..32, swap-AB)] regression, regression: qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192,
- **unpooler-full-m16-swapab-route** [unpooler GEMM] regression, regression: On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through t
- **pooler-unpooler-m128-nn-layout-gated** [pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)] regression, regression: Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master)
- **bf16-lossless-exp4-decode-only-throughput-probe** [lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)] regression, regression: A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn rou
- **bf16-verify-deepgemm-bf16-gemm-nt-route** [bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)] regression, regression: At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master p
- **qkv-k-split-batched-gemm-m65-m256** [qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM)] regression, regression: Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 onl
- **qkv-swapab16-five-stage-ring-m8-m16** [qkv GEMM (m<=16, swap-AB Triton)] regression, regression: qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99 us in 3834ceec) runs _bf1

## Queue and in flight

- queue: combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16 [combination (pooler K-halves m=128/256 + qkv row-owner m=8/16 + per-cell Lt plans m=128/256)] p2 hold
- queue: pooler-khalf-m128-fp32-partials-triton-castadd [pooler GEMM (m=128, K-split as a 2-batch cuBLAS GEMM)] p1 hold
- queue: b12-qkv-m1-r32-one-cta-per-sm-readflush-gate [qkv GEMM (m=1, bulk12 12-bit row-owner CTA shape)] p1 hold
- queue: lt-nn-layout-cells-o-m128-qkv-m128 [o GEMM + qkv GEMM (m=128, cuBLASLt plan on an NN weight copy)] p2 hold
- queue: lt-config-sweep-m256-beyond-heuristic [bf16 GEMM dispatch (m=256, cuBLASLt configs beyond the heuristic)] p3 hold
- queue: khalf-gate-three-rounds-pooled [pooler m=128 khalf (b2) cell gate (_khalf prepare gate only)] p1 hold
- queue: khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr [pooler m=128 khalf (b2) cell gate (_khalf_prepare gate only)] p1 hold
- queue: pooler-khalf-lt-batched-plan-m128-m256 [pooler GEMM (m=128/256, K-halves batched GEMM on a prepare-picked cuBLASLt plan)] p2 hold
- queue: router-hilo-m128-m256-readflush-config-sweep [router GEMM (m=65..256, hilo launch config chosen per m by a read-flush sweep)] p3 hold
- 8 stale holds expired to `done/expired/` (not run)
