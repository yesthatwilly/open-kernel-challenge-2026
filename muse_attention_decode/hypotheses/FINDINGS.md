# muse_attention_decode findings

Regenerated 2026-10-29T224335Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Claims (from CLAIMS.md, newest per mechanism; tag meanings at the top of CLAIMS.md)

### lead

- [lead] full-verify paged-gather attention gate kernel (non-WS; 64 KB in flight/SM; FP8 QK wgmma + f16 PV; grid-barrier combine) routed at v16/v32 full only; warp specialization closed (C7514 trigger found and fixed, still 0.78-0.83x the gate's math speed) | rows: layer=full load=verify_16, layer=full load=verify_32 | evidence: H100 fa3gate (4b2a169) e2e 0.82 (v32) / 0.83-0.84 (v16) of master's FA3 call, judge-metric error 2.5-2.8e-2 vs 0.083/0.085 bounds; fa3ws (16c06a0) and fa3math (399b9bd): WS e2e 1.02-1.16x (C7513/C7514: unequal pending wgmma groups across paths; fixed by an empty commit_group + uniform wait<1>); research/h100-fa3{gate,ws,math}-1029.md | board: 1.634 | blob: 2eb03f76 | reopen: judge check of the gate kernel at v16/v32 full (projected H200 e2e 0.82-0.87, untested), stacked with the FA3 metadata fold (CLAIMS:82) for verify_1 | 2026-10-29 h100

### shelf

- [shelf] full-line K/V gather inside the c320 kernel A at full decode_1 (c320-attn-full-line-gather), est ~0.2% headline | rows: layer=full load=decode_1 | evidence: DECISIONS:435 (screen estimate, not measured) | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] sliding decode_1 per-kv-head cluster launch-config select (sliding-decode1-perkvhead-launch-select), est ~0.5% | rows: layer=sliding load=decode_1 | evidence: DECISIONS:436 | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] stream-K ring depth 5 after the full-line remap (sk-full-line-5stage-select), est ~0.2% | rows: layer=full load=decode_16, layer=full load=decode_32 | evidence: DECISIONS:437 | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] stream-K window route gate for sliding decode B=2..16 (sliding-decode-band2-sk-gate), est ~0.5% at p~0.4, a cheap stacker | rows: layer=sliding load=decode_16 | evidence: DECISIONS:438 | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] PERHEAD cluster DSMEM merge by push instead of pull at sliding decode_1 (sliding-decode1-ph-push-merge), est ~0.6%, high risk | rows: layer=sliding load=decode_1 | evidence: DECISIONS:439 | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] fast-math epilogues (o/L, rsqrt, sigmoid) in the routed attention kernels (attn-fast-math-epilogues), est ~0.2% | rows: layer=sliding load=decode_1, layer=sliding load=verify_1 | evidence: DECISIONS:440 | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] stack of onepass merge + PERHEAD push merge + fast-math epilogues (combo-tail-onepass-push-fastepi), est ~0.6% | rows: layer=sliding load=decode_1, layer=sliding load=verify_1, layer=full load=decode_16, layer=full load=decode_32 | evidence: DECISIONS:441; its onepass part is [unresolved] above, not Do-not-try | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen
- [shelf] FA3 scheduler metadata folded into the pre-attention kernel (one META program), FA3 prepare node removed on full verify rows (fa3-metadata-fold-preattn-full-verify) | rows: layer=full load=verify_1, layer=full load=verify_16, layer=full load=verify_32 | evidence: check 6d1082a6c0dd4f80: metadata bitwise == library on 156-162 samples/row (caps 57/8/4 binding, stable tie), output/FA3/k_cache/v_cache torch.equal, all 3 rows routed; in-run race step 32.45->30.06 (1.085/1.074x), 160.98->158.66 (1.016/1.013x), 301.30->298.62 us (1.009x); judge rows verify_1 full 27.8 us (board 30.1), verify_16 full 161.4 (162.7), verify_32 full 299.8 (302.1); headline 1.6336 vs board 1.6340 (lucky-draw board; flat within the sliding-gate flips); ~+0.7% expected; branch hyp/muse_attention_decode/fa3-metadata-fold-preattn-full-verify | board: 1.6340 | blob: 2eb03f76 | reopen: stack with another attention row gain to >= 1.24% over the board (e.g. the [unresolved] onepass verify_1) | 2026-10-29 churn
- [shelf] dense_q full-verify FA3 call (q reshaped [B,8,H,D], cu_seqlens_q dropped; bitwise-equal output), stacks with the metadata fold | rows: layer=full load=verify_1 | evidence: H100 fa3v probe (takemyjob hopper/fa3v c67d880; research/h100-fa3-1029.md): dense_q 1.03-1.04 at v1, 1.00 v16/v32; with precomputed metadata 1.05-1.06 / 1.006 / 1.007; 7 interleaved rounds, 2 seeds | board: 1.634 | blob: 2eb03f76 | reopen: stack with the fold (CLAIMS:82) and another row gain to >= 1.24% | 2026-10-29 h100

### unresolved

- [unresolved] attention board level: 1.6340 drew both flipping gates (s2b B=17..32 routed only on the board, verify_32 sliding 27.10 vs 27.79-28.34 us; the B=2..16 grid fell back to 8b once, verify_16 sliding 20.10 vs 19.40-19.75 us); the code's level is ~1.61-1.62 | rows: layer=sliding load=verify_16, layer=sliding load=verify_32 | evidence: logs of 4e386647, ba8c8a61, bfb0457c, aa9fc860 (LOG:1970-1973); pins sw-pin-verify-all 1306b615 1.6154, sw-pin-b2-sm 8a87e3ea 1.6094, sw-pin-b17-on cb4fe7c4 1.6073, sw-pin-b17-off 2ab18645 1.6029 (LOG:1993-1995) | board: 1.6340 | blob: 2eb03f76 | reopen: pinning buys variance, not score; sw-pin-verify-all is the deterministic final-rerun candidate (~1.63) | 2026-10-28 campaign
- [unresolved] one-pass ticket merge at sliding verify B=1 (sk-onepass-verify1-timing-fix; overrides its Do-not-try entry): routed x1.026 / x1.024 in-prepare (14.9 -> 14.6 us), but the judged row sits inside the board's own 12.66-13.90 us spread | rows: layer=sliding load=verify_1 | evidence: bfb0457c9ae84322 1.6219, scored aa9fc86078f543fb 1.6183 (DECISIONS:399-400; LOG:1303-1311); pins-onepass 4cfddb59 1.6265 vs a base level ~1.623 (LOG:2049-2052, 2081); FINDINGS:228 | board: 1.6340 | blob: 21166a14 | reopen: >= 3 runs per arm on pinned routes, or stacked with a measured verify_1 fixed-cost cut | 2026-10-28 campaign
- [unresolved] swf stage 1 (one cluster launch per sliding-verify step): canary FAIL at B=2 lens 3/2100 C=4 (rel L2 4.91e-3 vs incumbent 5.80e-3, worst/mean row 6.8); all bands fell back | rows: load=verify_1 load=verify_16 load=verify_32 | evidence: check 109a387b34a3441a 1.624 (incumbent routes); attrs 124-125 regs, 0 B local; rungs 296ec797 chain+merge clean | board: 1.6340 | blob: 2eb03f76 | reopen: fix of the short-lens canary case (in progress for 00:00Z) | 2026-10-28 campaign
- [unresolved] swf S1b: one cluster launch per sliding-verify step (C=16/8/4), canary rule fixed | rows: load=verify_1, load=verify_16, load=verify_32 | evidence: check 89add1cc5468453d 1.626 [1.602, 1.656] vs board 1.6340; level ~1.62, neutral | board: 1.6340 | blob: 2eb03f76 | reopen: the expert-block focus ends, or a per-row swf vs master read shows >= 2.5% headline | 2026-10-28 campaign

### node-only

- [node-only] full-verify FA3 main loop is bound by memory parallelism per SM (one CTA/SM, ~24-32 KB in flight), not HBM or tensor: a paged gather of the same rows reaches 2.72/2.79 TB/s vs FA3 172/324 us against a 123/236 us ceiling (v16/v32), ~1.37-1.40x headroom; num_splits=1 is 3-21x slower on full verify (LOG:2043 does not hold there) | rows: layer=full load=verify_16, layer=full load=verify_32 | evidence: H100 fa3v ncu: DRAM 59.5/62.5%, issue 34%, tensor ~26%, L2 hit 6-8%, stalls long_scoreboard/barrier/wait; research/h100-fa3-1029.md | board: 1.634 | blob: 2eb03f76 | reopen: a mainloop-only stream-K paged-gather diag at >= 2.6 TB/s at v32 on the node, then a judge check (H200 ceiling untested) | 2026-10-29 h100
- [node-only] custom full-verify paged-gather attention (wgmma FP8 QK from the swizzled cp.async ring, f16 PV; 3 stages x 2 CTAs/SM = 64 KB in flight; page indices a ring ahead; grid-barrier combine): e2e 0.82 (v32) / 0.83-0.84 (v16) of master's FA3 call on the H100, error 1.67e-3 vs fp32 (FA3 6.3e-3..3.9e-2), judge-metric 2.5-2.8e-2 vs 0.076-0.085 bounds; v1 loses (1.42, latency-bound combine) | rows: layer=full load=verify_16, layer=full load=verify_32 | evidence: H100 fa3gate (takemyjob hopper/fa3gate 4b2a169; research/h100-fa3gate-1029.md, GPU 6): mainloop 2.52-2.54 TB/s at v32 (0.90-0.91 of the 2.79 gather ceiling), 2.38-2.40 at v16; math-off 2.80/2.76 TB/s; math-only 233 us at v32 = the memory-ceiling rate, latency-bound (issue ~52%); mma.sync e4m3 is emulated on sm_90a; projected H200 e2e ~0.82-0.87 (untested) | board: 1.634 | blob: 2eb03f76 | reopen: FA3-style warp specialization (producer WG + 2 setmaxnreg consumers, QK/softmax overlap) making math ~1.3x faster than the ceiling rate, then a judge check routed at v16/v32 only | 2026-10-29 h100

### landed

- [landed] fused pre-attention Triton kernel (q/k norm + RoPE + cache store, one launch); every routed path's "pre-attn" stage | rows: all | evidence: promote 3e98bd00, job 9e8fcb3fe4b04458 | board: 1.1274 | blob: f14211b4 | reopen: - | 2026-10-24 churn
- [landed] FA3 head pairing on full-attention layers (each token's two KV heads fetched as one 128-byte row), with the paired-heads launch cleanup and batch-gated tuning; FA3 now runs only at full verify | rows: layer=full load=verify_1, layer=full load=verify_16, layer=full load=verify_32 | evidence: promotes 86dc7133 (job facc479992e24a35), fa95f6d9 (2846122f764043da), bfb7bac3 (bd64bf9f9e904f02) | board: 1.2686 | blob: efc54410 | reopen: - | 2026-10-24 churn
- [landed] CUDA split-KV attention for full decode B>1, now the stream-K + ticket v2 kernel (splitkv_attn_sk_kernel) | rows: layer=full load=decode_16, layer=full load=decode_32 | evidence: promote 3693fb89 (job 82085a4e82154393); PATH CHOSEN "full decode streamk+ticket v2" in campaigns/2026-10-26-expert-bandwidth/runs/4e386647edd845f7.json.gz | board: 1.3468 | blob: 1c0f5fa1 | reopen: - | 2026-10-24 churn
- [landed] sliding verify_1 stream-K pseudo-decode kernel (PSEUDO, 4 stages, ticket, grid p8) | rows: layer=sliding load=verify_1 | evidence: promote eb1b68ba (job cbd49dc94be548e7, 1.4920); PATH CHOSEN "sliding verify B=1: stream-K pseudo-decode" in 4e386647 | board: 1.4920 | blob: 84b54ae2 | reopen: - | 2026-10-25 churn
- [landed] sliding decode B=17..32 stream-K window kernel (PSEUDO TOK=1, grid p4) | rows: layer=sliding load=decode_32 | evidence: promote 7e128915 (job 252abb7f11ec48c3, 1.5017); 4e386647 head-to-head 18.8 us vs cluster C=4 21.7 us | board: 1.5017 | blob: 5cf8fae3 | reopen: - | 2026-10-25 churn
- [landed] sliding decode_1 cluster kernel per kv head (PERHEAD, 2 clusters x C=16, fp8 qk) | rows: layer=sliding load=decode_1 | evidence: promote 5d528bf3 (job 4b608df595b046fb, 1.5120) | board: 1.5120 | blob: 6d6afa47 | reopen: - | 2026-10-25 churn
- [landed] merge of per-kv-head sliding decode_1 + full decode_1 chunk-320 V4 chain (A+C+D) + SV32 tiled pre-attention at sliding verify_32 (merge-perkvhead-v3chain-sv32-tiled; the base of 0d421faf) | rows: layer=full load=decode_1, layer=sliding load=verify_32 | evidence: job b4b20425eaff4942 1.5360 (churn 406b997e); LOG:857-858; PATH CHOSEN "[chain]: V4 (A+C+D)" and "SV32 [B tiled]: route" in 4e386647 | board: 1.5360 | blob: 219c08d9 | reopen: - | 2026-10-26 churn
- [landed] sk-full-line-gather: full-line K/V issue in the stream-K kernel | rows: layer=full load=decode_16, layer=full load=decode_32 | evidence: check 00acb91008824b1e 1.5952, scored bbf762ed52934bde 1.5902 (LOG:857-858, 884); land 0d421faf | board: 1.5902 | blob: 3e89dd89 | reopen: - | 2026-10-27 campaign
- [landed] c320 one-pass combine at full decode_1 (ONEPASS on the V4 chain) | rows: layer=full load=decode_1 | evidence: check 508cfca7650848ec (V4 19.20 -> 17.02 us, x1.128), scored 276ff0cda0404ef2 1.6095 (LOG:907-908, 933); land 09f04175 | board: 1.6095 | blob: 09d8594a | reopen: - | 2026-10-27 campaign
- [landed] s2b sliding verify B=2..16: window kernel + PDL sw_combine_kernel (grid sm) | rows: layer=sliding load=verify_16 | evidence: check af08448e82c04fdf (x1.14 / x1.15, merge +2.9 us), scored a76ee9b03a9f4db4 1.6114 (LOG:934, 941); land ea24b53c | board: 1.6114 | blob: 376da166 | reopen: - | 2026-10-27 campaign
- [landed] sw17: s2b at sliding verify B=17..32 with the gate bar lowered to 1.03 (this gate flips between runs; see the board-level line below) | rows: layer=sliding load=verify_32 | evidence: check 4e695fa0 1.6129 (x1.032 / x1.037), scored 4e386647edd845f7 1.6340 (LOG:983-985); land 31f4f6e4 | board: 1.6340 | blob: 2eb03f76 | reopen: - | 2026-10-27 campaign

### refuted

- [refuted] regridding or tiling the pre-attention norm+RoPE kernel (group-tiled programs; tiled gate at large token counts) | rows: all | evidence: 877731a3b2214ddb 1.2614 vs 1.2686, c2ebf6ed05824692 1.3863 vs 1.3949 (FINDINGS:214, 220) | board: 1.3949 | blob: 1e5e94e8 | reopen: a profile shows the pre-attention kernel above ~2 us on a heavy row | 2026-10-24 churn
- [refuted] split-KV kernel micro-changes at full decode B>1: FP8 QK mma, 2048-token chunks at B>=32, wide-MLP combine, PDL attn->combine | rows: layer=full load=decode_16, layer=full load=decode_32 | evidence: b939681d3ab549d7, 7cfc8c5be3a04369, 47967b6c4d98454b, 113cbf3093dd40b9, all 1.3439-1.3929 vs 1.3949 (FINDINGS:216, 218, 222, 223) | board: 1.3949 | blob: 0850f42d | reopen: - | 2026-10-24 churn
- [refuted] token-tiled paired output norm-gate kernel | rows: layer=full | evidence: 5e01430a8dfe4977 1.3911 vs 1.3949, expected hit 0/3 (FINDINGS:219) | board: 1.3949 | blob: 7745b9a3 | reopen: - | 2026-10-24 churn
- [refuted] folding the pre-attention kernel into the full-decode split-KV prologue (FUSED_PRE / fused-preattn stages) | rows: layer=full load=decode_1, layer=full load=decode_16, layer=full load=decode_32 | evidence: 825baa03798f431a 1.4283 (FINDINGS:89), 2038a530edd04309 1.4411 (FINDINGS:70); cbd49dc94be548e7 target rows 1.0013 / 0.9973 (hypotheses/FINDINGS-MODEL-REVIEW-2026-10-28.md:55); FUSED_PRE not >= 1.02x at B=2..64 in the logs of 4e386647 and 7284659f | board: 1.4920 | blob: 84b54ae2 | reopen: an in-prepare A/B shows the fused prologue shortening the kernel's critical path | 2026-10-25 churn
- [refuted] alternatives to the stream-K kernel at full decode_1: Triton fused decode1 gate, split-KV chunk 640, deeper two-level ring | rows: layer=full load=decode_1 | evidence: af3c6a46badf4e50 1.3417, 91d541778b8f45f3 1.3805, 26c049edfdd842b3 1.5109 vs 1.5295 (FINDINGS:215, 217, 224) | board: 1.5295 | blob: 246f5494 | reopen: - | 2026-10-25 churn
- [refuted] sliding batched-decode Triton kernel tweaks: BLOCK_N 128 single tile, hoisted first old-key tile gather, paired kv-head row gathers | rows: layer=sliding load=decode_16, layer=sliding load=decode_32 | evidence: 88c62334775e48e4 1.3834 (neutral), ead13d58c6c843e9 1.3449 and 48ffbe5690cc4ab4 1.2865 (regressions) (FINDINGS:221, 231, 232) | board: 1.3949 | blob: af33f955 | reopen: - | 2026-10-25 churn
- [refuted] FA3 call-site tweaks: window-trimmed page table, scheduler metadata on a side stream, head pairing at sliding verify, num_splits band gate at full verify | rows: layer=sliding load=verify_16, layer=sliding load=verify_32, layer=full load=verify_16, layer=full load=verify_32 | evidence: 1f63fcbbc0394bed 1.1411, 42964d938c5842f4 1.2245, 031731b5db294a92 1.3475 (regressions), 849f2733f8ea4888 1.5216 vs 1.5295 (FINDINGS:229, 230, 233, 225) | board: 1.5295 | blob: f171866b | reopen: only inside the attn-fixed rewrite: LOG:2084 measures num_splits=1 and precomputed scheduler metadata at ~2.2 us each of FA3's ~15 us fixed cost at full verify | 2026-10-26 churn
- [refuted] c320 PDL early trigger at full decode_1 | rows: layer=full load=decode_1 | evidence: 508cfca7650848ec readout x0.996 (LOG:907-908); 4e386647 kept the incumbent trigger, readout ratio 0.990 (PATH CHOSEN); DECISIONS:434 | board: 1.6095 | blob: 09d8594a | reopen: - | 2026-10-27 campaign
- [refuted] sliding decode s3: pre-attention + RoPE fused into the sliding decode kernel | rows: layer=sliding load=decode_16, layer=sliding load=decode_32 | evidence: check 877a6ba556e142d4 1.6209: B=2..16 failed its correctness runs, B=17..32 x0.99-1.005 (LOG:960) | board: 1.6114 | blob: - | reopen: - | 2026-10-27 campaign
- [refuted] in-kernel ticket merge at full decode B>1 (merge_range one-pass twins in splitkv_attn_sk_kernel) | rows: layer=full load=decode_16, layer=full load=decode_32 | evidence: check ba8c8a6103124f71 1.6177: full decode x1.002-1.005, the merge is not the bottleneck (LOG:1249-1251; FINDINGS:227) | board: 1.6340 | blob: faed6a62 | reopen: - | 2026-10-27 campaign
- [refuted] in-kernel ticket merge (tk) for the sliding-verify window kernel (s2b fast-merge); finding erased from FINDINGS by c3c14a4d, restored here | rows: layer=sliding load=verify_16, layer=sliding load=verify_32 | evidence: c18c73f7; merge_head +17.6-45.0 us and ticket-only tail +1.5-2.7 us vs the PDL combine's +2.9-5.2 us (log of bfb0457c9ae84322; check af08448e82c04fdf); 4e386647 tk merge +22.4 / +15.5 us vs pdl +3.2 / +5.0 us; LOG:1917-1918 | board: 1.6340 | blob: 21166a14 | reopen: the window kernel itself gets faster first (14.6 us alone at B=16 vs a 1.9 us floor; 22.4 us at B=32 vs 3.7 us) | 2026-10-28 churn
- [refuted] full verify s7: wgmma split-combine kernel replacing FA3 at B=16/32 | rows: layer=full load=verify_16, layer=full load=verify_32 | evidence: review hold DECISIONS:405-406; bounded by s6's DIAG, the no-ticket kernel would be ~179 / 367 us vs FA3 main 148.6 / 285.9 us (LOG:1919-1920) | board: 1.6340 | blob: - | reopen: an isolated DIAG beats FA3's main loop at B=16 and B=32 first | 2026-10-28 churn
- [refuted] pre-attention folded into the PSEUDO verify-1 kernel (sliding-verify1-pseudo-fused-preattn-rope, muse_sv1_fused_v1) | rows: layer=sliding load=verify_1 | evidence: job 7284659f6c4d4035 1.6191 vs 1.6340 (66e589b9): its own gate read FUSED p8 16.4 us vs pre-attn + PSEUDO 15.3 us (x0.934; best FUSED grid sm 15.8 us), never routed (archive/sliding-verify1-pseudo-fused-preattn-rope/7284659f6c4d4035.log); the verify_1 sliding row 13.4 vs board 13.9 us is the incumbent inside the 12.66-13.90 spread (LOG:2051-2052); full decode_32 190.6 vs 177.9 us and verify_32 sliding 27.8 us are gate draws | board: 1.6340 | blob: 7e7938b3 | reopen: - | 2026-10-28 churn
- [refuted] sized out: a rewrite of the full-decode split-KV attention at B=16/32 (attn-full-splitkv) | rows: layer=full load=decode_16, layer=full load=decode_32 | evidence: those rows run at 1.1-1.3x their floor on the scoring formula (LOG:2201; hyp.py triage after 4b45b873); lever dropped LOG:2159-2160 | board: 1.6340 | blob: - | reopen: a measured floor gap above 1.5x on those rows | 2026-10-28 campaign
- [refuted] fvs: full verify B=1 on the sliding window kernel at window 0 + per-row combine (attn-fixed design 2 / swf stage 4) | rows: layer=full load=verify_1 | evidence: diag-sw-chain-cost 664a6da6321a45c9 log:106: kernel alone at 80008 keys 24.6 us grid 132 / 22.3 us grid 264 vs FA3 step 32.4 and FA3 main 14.7 us; combine alone 43.6 us; PDL pair + pre-attn 66.5 us; its own rule (> 17 us: drop fvs) | board: 1.6340 | blob: 2eb03f76 | reopen: - | 2026-10-29 screen
- [refuted] full verify B=1 on the stream-K decode kernel with token groups (full-verify1-sk-two-level-token-group, FULLV TG 1/2/4/8) | rows: layer=full load=verify_1 | evidence: job 7d23e8776c194344 log:80-96: best FULLV TG=2 grid 2xSM 51.0 us vs FA3 32.7 us raw (0.641x), TG=8 65.8 us, TG=1 70.4 us; never routed; LESSONS:126 | board: 1.4920 | blob: 2eb03f76 | reopen: - | 2026-10-29 screen
- [refuted] swf S1b: one cluster launch per sliding-verify step, per-row read | rows: layer=sliding load=verify_1, layer=sliding load=verify_16, layer=sliding load=verify_32 | evidence: check 89add1cc5468453d log:53-67: B=1 rounds 1.004-1.027x vs incumbent 14.6 us, B=2..16 quick 0.997x, B=17..32 quick 0.989x, no band routed; lens-0 swf step 10.8-11.4 us gross (null 2.2) bounds B=1 at >= ~11 us | board: 1.6340 | blob: 2eb03f76 | reopen: a prepare A/B shows the swf lens-0 step below ~7 us | 2026-10-29 screen

### superseded

- [superseded] naive stack of full decode_1 chunk-320 with tiled verify pre-attention (combo-naive-c320-fd1-tiled-preattn-verify) | rows: layer=full load=decode_1, layer=sliding load=verify_32 | evidence: b9d062aaa94c4cbf 1.5094 vs 1.5120, expected hit 5/6 (FINDINGS:226); both parts landed through b4b20425 and 0d421faf | board: 1.5120 | blob: a1f1eed6 | reopen: - | 2026-10-25 churn
- [superseded] sliding verify s2 split-combine (window kernel + separate combine, no PDL) | rows: layer=sliding load=verify_16, layer=sliding load=verify_32 | evidence: check becb6967be9940bf 1.5906, no band routed (B=16 x1.017, B=32 x0.978; window kernel 14.8 us but combine 7.5-10 us) (LOG:910-912); s2b owns the bands | board: 1.5902 | blob: - | reopen: - | 2026-10-27 campaign
- [superseded] FA3 num_splits band gate at sliding verify (sliding-verify-fa3-num-splits-band-gate) | rows: layer=sliding load=verify_16, layer=sliding load=verify_32 | evidence: DECISIONS:433: sliding verify B=2..32 runs on the s2b kernels, FA3 is not called | board: 1.6340 | blob: - | reopen: s2b loses both bands on master | 2026-10-28 screen
- [superseded] stack of full-line gather + c320 tail + early trigger (combo-sk-full-line-c320-tail) | rows: layer=full load=decode_1, layer=full load=decode_16, layer=full load=decode_32 | evidence: DECISIONS:434: full-line gather and c320 onepass landed (0d421faf, 09f04175); the early trigger was measured and not routed (LOG:908) | board: 1.6340 | blob: - | reopen: - | 2026-10-28 screen

## Current best

1.6340x [1.6094,1.6585] from job `4e386647edd845f7` (submit), master commit `26498226d8`, hypothesis sliding-verify-s2b-b17-32-bar103

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 10.8µs vs SGLang 20.8µs (SoL 0.1µs) · void (anonymous namespace)::sliding_cluster_kernel<true>(... 9.6µs 100.0%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 15.6µs vs SGLang 31.4µs (SoL 4.3µs) · void (anonymous namespace)::c320_attn_kernel<320, true, f... 11.2µs 80.6%, void (anonymous namespace)::c320_combine_kernel<320, 16, ... 2.7µs 19.4%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 15.4µs vs SGLang 26.7µs (SoL 1.6µs) · _sliding_fused_kernel 12.7µs 100.0%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 104.8µs vs SGLang 201.0µs (SoL 63.2µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 104.1µs 97.9%, _pre_attention_kernel 2.3µs 2.1%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 16.8µs vs SGLang 30.8µs (SoL 3.3µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 13.0µs 86.2%, _pre_attention_kernel 2.1µs 13.8%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 177.9µs vs SGLang 386.0µs (SoL 126.4µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 191.0µs 98.8%, _pre_attention_kernel 2.3µs 1.2%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 13.9µs vs SGLang 23.2µs (SoL 0.1µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 9.9µs 85.8%, _pre_attention_kernel 1.6µs 14.2%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 30.1µs vs SGLang 36.3µs (SoL 4.3µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 14.8µs 54.7%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.8µs 25.3%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 2.8µs 10.3%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 19.5µs vs SGLang 29.2µs (SoL 1.9µs) · void (anonymous namespace)::slidewin_verify_wgmma_kernel<... 12.7µs 58.4%, void (anonymous namespace)::sw_combine_kernel<true>(int c... 6.3µs 28.8%, _pre_attention_kernel 2.8µs 12.8%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 162.7µs vs SGLang 205.0µs (SoL 63.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 154.1µs 91.4%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.1µs 3.6%, void flash::prepare_varlen_num_blocks_kernel<1, true>(int... 3.1µs 1.8%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 27.1µs vs SGLang 34.5µs (SoL 3.7µs) · void (anonymous namespace)::slidewin_verify_wgmma_kernel<... 20.5µs 52.8%, void (anonymous namespace)::sw_combine_kernel<true>(int c... 15.8µs 40.7%, _pre_attention_tiled_kernel 2.5µs 6.4%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 302.1µs vs SGLang 389.6µs (SoL 126.8µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 284.6µs 94.0%, void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu... 6.6µs 2.2%, _pre_attention_kernel 4.2µs 1.4%

Diagnosis of 2026-10-25T203356Z, written on board 1.5295 (today 1.6340): historical, re-diagnose before minting from it (`muse_attention_decode/hypotheses/DIAGNOSIS.md`).

## By area

### FA3 call (full verify split count)
- **full-verify-fa3-num-splits-band-gate** → neutral (within-ci), 1.5216x, mechanism flat 0/1 — gained on sliding decode_1; lost on full decode_1. Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 main 14.9 + combine 7.3 + prepare 2.8 + norm-gate 1.1 us, shape 30.5) and

### FA3 call (full-layer decode gather)
- **full-decode-cuda-gather-splitkv** → gain (gain), 1.3468x, mechanism hit 2/2 — gained on full decode_16, decode_32; lost on sliding decode_1. Stage 1 of a rewrite of full-attention decode, built on master (sliding-fused-triton-decode1-gate). Gate (config/shape only): cfg window == 0 and q_len == 1 and
- **full-decode-cuda-fp8-qk-mma** → neutral (within-ci), 1.3439x, mechanism flat 0/2 — gained on sliding decode_1. Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kernel touched): comput
- **splitkv-full-line-gather-mapping** → neutral (unproven-vs-best), 1.4160x — gained on full decode_16, decode_32; lost on sliding decode_1. One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64): remap threads so that each warp-wide cp.async instruction cover
- **splitkv-chunk2048-batch32** → neutral (within-ci), 1.3838x, mechanism flat 0/1 — no shape moved outside noise. Shape-gated chunk size for master's CUDA split-KV path: at batch >= 32 use 2048-token work items instead of 1024, halving the partials written by kernel A and m

### FA3 call (full-layer verify gather)
- **full-verify-cuda-streamk-splitkv** → regression (regression), 1.2314x, mechanism miss 0/2 — gained on full decode_32; lost on full verify_16, verify_32. Stage 2 of the full-attention rewrite, built on master (stage 1 full-decode-cuda-gather-splitkv landed). Extend the CUDA split-KV path to full-layer VERIFY (q_l

### FA3 call (head packing)
- **fa3-hdim128-head-pairing** → gain (gain), 1.1706x, mechanism hit 6/6 — gained on full decode_16, decode_32, verify_16, verify_32; lost on full decode_1, verify_1. On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV hea

### FA3 call (head packing) gate
- **paired-heads-launch-cleanup-ab-tune-batch-gate** → neutral (unproven-vs-best), 1.2686x, mechanism hit 2/2 — gained on full decode_1, verify_1. From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 

### FA3 call (head packing, sliding verify)
- **sliding-verify-fa3-head-pairing** → regression (regression), 1.3475x, mechanism miss 0/2 — lost on full decode_32; sliding verify_1, verify_16, verify_32. Config-only change in kernel_fn: extend master head-pairing predicate to sliding verify. Today paired = window == 0 and hkv % 2 == 0 and batch > 1 and not split

### FA3 call (schedule, off critical path)
- **fa3-scheduler-metadata-side-stream** → regression (regression), 1.2245x, mechanism miss 0/11 — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full verify_1. Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash

### FA3 call (scheduler metadata folded into pre-attention, full verify)
- **fa3-metadata-fold-preattn-full-verify** → neutral (within-ci), 1.6336x, mechanism flat 1/3 — gained on sliding verify_1; full verify_1; lost on full decode_32. Fold FA3's varlen scheduler metadata (what flash::prepare_varlen_num_blocks_kernel<NumWarps, Sort> writes, 2.8-3.4 us per call on the board profile) into master

### attention boundary
- **paired-heads-launch-cleanup** → gain (gain), 1.2436x, mechanism hit 5/5 — gained on full decode_1, decode_16, decode_32, verify_1, verify_16. Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing 

### combination (conjectural: full decode_1 chunk320 + fused pre-attention prologue + PDL to combine)
- **combo-b1-chain-c320-fused-pre-pdl** → neutral (unproven-vs-best), 1.5295x, mechanism hit 1/1 — gained on full decode_1. COMBINATION 2 of 3, a conjectural B=1 launch-chain pairing (not the top rows). Built on current master 7e12891 (blob 5cf8fae). Full decode_1 is the highest-leve

### combination (conjectural: tiled pre-attention as every pair's first node, sd B=16 and skF selectors re-decided, + chunk320)
- **combo-tiled-preattn-neighbour-regate-c320** → neutral (unproven-vs-best), 1.5199x, mechanism flat 2/4 — gained on full decode_1; sliding verify_32. COMBINATION 3 of 3, a conjectural neighbour pairing: the pre-attention kernel as the shared first node of every two-node pair, with the downstream selectors re-

### combination (naive top-row stack: full decode_1 chunk320 + verify tiled pre-attention)
- **combo-naive-c320-fd1-tiled-preattn-verify** → neutral (within-ci), 1.5094x, mechanism hit 2/2 — gained on full decode_1; sliding verify_32; lost on sliding decode_1. COMBINATION 1 of 3, the naive top-row stack. Built on current master 7e12891 (blob 5cf8fae; restack-sliding-decode-sk-window-on-cluster-master, 1.5017 scored). 

### full + sliding decode attention (stack of two pending kernels)
- **stack-streamk-ticket-cluster-c16-c4** → neutral (beats-base-not-best), 1.4266x, mechanism hit 4/4 — gained on full decode_16, decode_32; sliding decode_32; lost on sliding verify_1. Stack the two pending (user-held, not on master) decode wins on current master; they touch disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harve

### full decode_1 split-KV + sliding verify_32 pre-attention
- **merge-perkvhead-c320-sv32-tiled** → neutral (unproven-vs-best), 1.5347x, mechanism miss 1/3 — gained on sliding decode_1, verify_32; lost on full decode_1. Hand-merge two gated pieces of combo-tiled-preattn-neighbour-regate-c320 (branch hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320, impl 5add0

### full decode_1 split-KV launch chain (merge onto per-kvhead master, with SV32 tiled pre-attention)
- **merge-perkvhead-v3chain-sv32-tiled** → neutral (unproven-vs-best), 1.5360x, mechanism miss 1/3 — gained on sliding decode_1. DIAGNOSIS action 3(i), the PENDING_USER hand-merge. Start from branch hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled (built on master 5d528bf3 = per-

### full-layer attention (batch-1 decode gate)
- **full-fused-triton-decode1-gate** → neutral (within-ci), 1.3417x, mechanism miss 0/2 — lost on full decode_1. Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) through master _sliding_fused_kernel via _sliding_forwar
- **full-decode1-cuda-splitkv-chunk320** → neutral (unproven-vs-best), 1.3954x, mechanism hit 1/1 — gained on full decode_1; lost on full decode_32; sliding verify_1. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's existing CUDA split-KV path with a smaller work chunk, instead of 
- **full-decode1-cuda-splitkv-chunk640** → neutral (within-ci), 1.3805x, mechanism hit 1/1 — gained on full decode_1; lost on full decode_32; sliding verify_1. Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) thr

### full-layer attention (batch-1 decode, splitkv stream-K kernel)
- **full-decode1-streamk-s3-two-level-ticket** → neutral (unproven-vs-best), 1.4322x, mechanism hit 1/1 — gained on full decode_1; lost on sliding verify_1. Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: 

### full-layer attention (batch-1 decode, stream-K kernel merge)
- **full-decode1-sk4-cluster-dsmem-merge** → neutral (unproven-vs-best), 1.4388x, mechanism flat 0/1 — gained on sliding verify_1; lost on sliding decode_1. Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01). Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) 

### full-layer attention (batch-1 decode, stream-K prologue fusion)
- **full-decode1-sk-two-level-fused-preattn** → neutral (within-ci), 1.4411x, mechanism hit 1/1 — gained on full decode_1. Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel: ~2.6 us of the row is the second graph node and its launch gap. PD

### full-layer attention (batch-1 decode, stream-K ring depth)
- **full-decode1-sk-two-level-deep-ring** → neutral (within-ci), 1.5109x, mechanism miss 0/1 — gained on sliding decode_1; lost on full decode_1. Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stages, two-level ticket> at grid = SM count: 19.0 us kernel + pre-attn 1.4

### full-layer attention (batch-1 verify, stream-K two-level kernel, token grouping)
- **full-verify1-sk-two-level-token-group** → neutral (beats-base-not-best), 1.4714x, mechanism flat 0/1 — gained on sliding decode_1, verify_1. On the token-group branch (sliding-verify-sk-pseudo-s2-token-group). Full verify_1 is 30.5 us shape: FA3 15.4 + combine 7.0 + prepare 2.8 + pre-attn 1.5 + norm-

### full-layer attention (decode 1<B<=64 prologue) on top of the sliding restack
- **restack-sk-pseudo-cluster-plus-v3-fused-preattn** → gain (gain), 1.4920x, mechanism flat 2/4 [gain not attributable to the mechanism (race/draw?)] — gained on sliding decode_1, decode_32, verify_1. Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the st

### full-layer attention (decode 1<B<=64, stream-K ticket epilogue)
- **full-decode-sk-s4-deferred-ticket-arrive** → failed (test:{\"batch\": 1, \"ctx\": 10, \"heads\": \"tp8\", \"layer\": \"sliding\", \"seed\": 6893170448766904337}). Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs 106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.
- **full-decode-sk-s5-deferred-ticket-fault-fix** → neutral (within-ci), 1.5162x, mechanism flat 0/2 — gained on sliding decode_1; lost on full decode_1. DIAGNOSIS action 4: re-measure the deferred ticket arrive, which has never been measured because it faulted. Re-apply done/failed/full-decode-sk-s4-deferred-tic

### full-layer attention (decode, splitkv kernel A/B)
- **full-decode-splitkv-s2-streamk-ticket-inproc** → neutral (unproven-vs-best), 1.4171x, mechanism hit 2/2 — gained on full decode_16, decode_32. Stage 2 of the full-decode rewrite, built on master (stage 1 kernels splitkv_attn_kernel A + splitkv_combine_kernel B in load_inline module muse_full_decode_spl

### full-layer attention (decode, splitkv stream-K kernel prologue)
- **full-decode-splitkv-s3-fused-preattn** → neutral (within-ci), 1.4283x, mechanism flat 1/2 — no shape moved outside noise. Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token t

### full-layer attention (verify, splitkv A8/B8)
- **full-verify-wgmma-s3-warpgroup-per-kvhead-inproc** → neutral (beats-base-not-best), 1.3801x, mechanism hit 2/2 — gained on sliding decode_1, decode_16, decode_32; full verify_16, verify_32; lost on full decode_32. Stage 3 of the full-attention rewrite, built on the stage-2 code (branch hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv, impl_commit ee7dd20, rebase

### full-layer attention (verify, wgmma stream-K kernel, multi-tile Q-fragment fix)
- **full-verify-wgmma-s5-qfrag-reload** → neutral (beats-base-not-best), 1.4345x, mechanism flat 0/2 — gained on full decode_1, decode_16, decode_32. Stage 5 on branch hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix (job 51b0d5c110f3459c). s4's bisect: every multi-tile segment (>=2 tiles, even B1

### full-layer attention (verify, wgmma stream-K kernel, multi-tile correctness)
- **full-verify-wgmma-s4-multitile-fix** → neutral (within-ci), 1.3758x, mechanism flat 0/2 — no shape moved outside noise. Stage 4 (correctness fix) of the full-verify wgmma rewrite, built on the stage-3 commit (hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inp

### full-layer attention (verify, wgmma stream-K kernel, per-warpgroup ring)
- **full-verify-wgmma-s6-split-ring-2cta** → neutral (unproven-vs-best), 1.5331x, mechanism flat 0/2 — gained on sliding decode_1; lost on full decode_1. DIAGNOSIS action 1(a)+(b): the sibling of the queued full-verify-wgmma-ws-s1-producer-split that KEEPS s5's proven occupancy (2 CTAs/SM, every thread issues loa

### full-layer attention (verify, wgmma stream-K kernel, warp-specialized pipeline)
- **full-verify-wgmma-ws-s1-producer-split** → neutral (within-ci), 1.5157x, mechanism flat 0/2 — gained on sliding decode_1; lost on full decode_1. Fresh stage 1 on CURRENT master (no depends_on). Why now: the s3/s4 wgmma full-verify items are held because the stage-3 kernel failed its self-check; that hold

### full-layer attention (whole-path rewrite, warp-specialized one launch)
- **full-cuda-onelaunch-warpspec-bulk-gather** → regression (regression), 0.8238x, mechanism miss 0/2 — lost on full decode_16, decode_32; sliding verify_1. Stage 1 of a rewrite of the FULL-layer path, built on master (stage 1 of the first CUDA kernel, full-decode-cuda-gather-splitkv, has landed; no depends_on). Why

### full-layer verify attention (FA3 call at verify_16 / verify_32)
- **full-verify-paged-gather-gate-kernel** → neutral (within-ci), 1.6154x — gained on sliding verify_1; lost on full decode_32; sliding verify_16. Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, paged FP8 KV, ctx ~80000) with the H100-validated gate kernel from takemy
- **full-verify-paged-gather-gate-kernel-r2** → neutral (unproven-vs-best), 1.6422x — gained on full verify_16, verify_32; lost on sliding decode_1; full decode_32. Same change as full-verify-paged-gather-gate-kernel (restacked from its patch), with the one build fix its judge run needed: _FVG_FLAGS used -std=c++17, and the

### launch overhead
- **splitkv-pdl-attn-to-combine** → neutral (within-ci), 1.3929x — lost on full decode_32. Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv

### launch overhead (full decode_1, pre-attention -> stream-K kernel)
- **full-decode1-sk-pdl-from-preattn** → neutral (unproven-vs-best), 1.4571x, mechanism flat 0/1 — gained on full decode_32; sliding verify_1. Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is 22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, s

### output norm-gate
- **paired-norm-gate-token-tiled** → neutral (within-ci), 1.3911x, mechanism flat 0/2 — lost on full decode_32. Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at full verify_16/32) from one 1-warp program per (token, query head

### output norm-gate (gate operand prefetch in the CUDA epilogues)
- **epilogue-gate-l2-prefetch-cuda** → neutral (unproven-vs-best), 1.5020x, mechanism flat 0/4 — no shape moved outside noise. Built on current master eb1b68b (no depends_on). Mechanism: both landed CUDA kernels read the output gate logits (qkv row r, bf16 columns (HQ + 2*HKV)*D .. +HQ*

### page table (sliding)
- **fa3-window-trimmed-page-table** → regression (regression), 1.1411x, mechanism miss 0/4 — lost on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each request instead of the whole 80K-token context. Let W = window - 1 + L

### pre-attention norm+rope
- **fused-preattention-triton** → gain (gain), 1.1274x, mechanism hit 8/8 — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, gr
- **preattention-group-tiled-programs** → neutral (within-ci), 1.2614x, mechanism flat 1/8 — gained on sliding verify_32; lost on sliding decode_1. Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (
- **preattention-tiled-gate-large-tokens** → neutral (within-ci), 1.3863x, mechanism flat 1/2 — gained on sliding verify_32; lost on full decode_32. Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is muse_attention_decode/hypotheses/done/neutral/preattention-group-t

### sliding attention, batched decode, split depth
- **sliding-deep-splits-on-cuda-splitkv** → gain (gain), 1.3949x, mechanism hit 2/2 — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_32. Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto 

### sliding-layer attention (batch-1 decode gate)
- **sliding-fused-triton-decode1-gate** → neutral (unproven-vs-best), 1.2931x, mechanism hit 1/1 — gained on sliding decode_1. Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim fr

### sliding-layer attention (batch-1 decode, cluster kernel work split)
- **sliding-decode1-cluster-per-kvhead** → neutral (unproven-vs-best), 1.5120x, mechanism hit 1/1 — gained on sliding decode_1. Built on current master (no depends_on). Sliding decode_1 (weight 46, 60% of the decode_1 load) runs sliding_cluster_kernel at C=16: ONE cluster of 16 CTAs for 

### sliding-layer attention (batch-1 decode, split depth)
- **sliding-fused-triton-decode1-splits-32** → neutral (unproven-vs-best), 1.3520x, mechanism flat 1/2 — no shape moved outside noise. Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only shape master routes to _sliding_fused_kernel) use S = 32 splits in

### sliding-layer attention (batch-1 verify, PSEUDO stream-K kernel, pre-attention fold)
- **sliding-verify1-pseudo-fused-preattn-rope** → neutral (within-ci), 1.6191x, mechanism no-base — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. DIAGNOSIS action 3(ii). Stacked on merge-perkvhead-v3chain-sv32-tiled (build on its branch, whatever its score; the change touches only the sliding verify B=1 b

### sliding-layer attention (batched decode, KV row gather)
- **sliding-decode-paired-kvhead-rows** → regression (regression), 1.2865x, mechanism miss 0/2 — lost on sliding decode_16, decode_32, verify_1; full decode_32. One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding decode at batch > 1 only (q_len 1; config-only predicate pair = lengt

### sliding-layer attention (batched decode, gather prologue)
- **sliding-fused-hoist-first-tile** → regression (regression), 1.3449x, mechanism miss 0/1 — lost on sliding decode_1, decode_16, decode_32; full decode_32. In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so the KV round trip overlaps the q load, RMSNorm and RoPE. Move th

### sliding-layer attention (batched decode, split depth)
- **sliding-fused-triton-decode-batched-deep-splits** → neutral (unproven-vs-best), 1.3087x, mechanism hit 2/2 — gained on sliding decode_16. Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window and batch == 1 and q_len == 1) to (window and q_len == 
- **sliding-decode32-splits-8** → neutral (unproven-vs-best), 1.3973x, mechanism hit 1/1 — gained on sliding decode_32; lost on full decode_32. Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8 (today 16 at every batch). At batch 32 that is 512 programs of 256 

### sliding-layer attention (batched decode, tile size)
- **sliding-decode-blockn128-single-tile** → neutral (within-ci), 1.3834x, mechanism hit 1/1 — gained on sliding decode_1; lost on full decode_32. Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split c

### sliding-layer attention (decode, cluster kernel)
- **sliding-cluster-s2-decode-c4-pdl-inproc** → neutral (unproven-vs-best), 1.4001x, mechanism miss 1/3 — gained on sliding decode_32, verify_1, verify_16, verify_32; lost on sliding decode_1, decode_16. Stage 2 of the sliding-layer cluster rewrite, built on the stage-1 code (branch hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch, impl_commit 5938

### sliding-layer attention (decode, stream-K windowed kernel)
- **sliding-decode-sk-window-route** → neutral (unproven-vs-best), 1.4973x, mechanism flat 1/2 — gained on sliding decode_32; full decode_32. On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is already a windowed stream-K decode kernel (per-row lo = max(0, sp -

### sliding-layer attention (decode, stream-K windowed kernel) restack
- **restack-sliding-decode-sk-window-on-cluster-master** → neutral (unproven-vs-best), 1.5017x, mechanism hit 1/1 — gained on sliding decode_32. Restack sliding-decode-sk-window-route (done/neutral 1.4973x [1.475,1.520], job 118c6de9704146f3; built on the unlanded s1 branch, base blob 76492911) onto curr

### sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4)
- **restack-sk-pseudo-b1-cluster-c16-c4** → neutral (unproven-vs-best), 1.4870x, mechanism hit 3/3 — gained on sliding decode_1, decode_32, verify_1. Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at B=1) the two measured sliding-row wins that touch disjoint shapes

### sliding-layer attention (verify, stream-K pseudo-decode kernel)
- **sliding-verify-sk-pseudo-decode-s1** → neutral (unproven-vs-best), 1.4705x, mechanism flat 1/3 — gained on sliding verify_1; lost on sliding decode_1. Stage 1 of a new sliding-verify line, built on master (which now carries the full-decode stream-K kernels: splitkv_attn_sk_kernel single-level ticket at 1<B<=64

### sliding-layer attention (verify, stream-K pseudo-decode kernel, token grouping)
- **sliding-verify-sk-pseudo-s2-token-group** → regression (regression), 1.4251x, mechanism miss 0/3 — lost on sliding verify_1. Stage 2 of the pseudo-decode line, on branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. Measured: the PSEUDO kernel won B=1 (15.4 vs FA3 19.6

### sliding-layer attention (verify, wgmma stream-K windowed kernel)
- **sliding-verify-wgmma-window-s1** → neutral (within-ci), 1.5164x, mechanism flat 0/2 — gained on sliding decode_1; lost on full decode_1. Fresh stage 1 on CURRENT master (no depends_on). This is the held full-verify-wgmma-s4-sliding-window idea with its hold reason fixed: it was held because the s

### sliding-layer attention (whole-path rewrite)
- **sliding-single-launch-fused-attention** → regression (regression), 0.9452x, mechanism miss 1/6 — gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (

### sliding-layer attention (whole-path rewrite, CUDA cluster)
- **sliding-cuda-cluster-dsmem-onelaunch** → regression (regression), 1.2206x, mechanism miss 2/6 — gained on sliding decode_1, decode_16; lost on sliding verify_1, verify_16, verify_32. Stage 1 of a rewrite of the whole SLIDING-layer path, built on master (no depends_on). Why: the sliding layer runs 46x per step in every load, and it costs 13.9

### split-KV combine
- **splitkv-combine-wide-mlp** → neutral (within-ci), 1.3836x — lost on full decode_32. Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight; grid (B, 16) and the math stay. Today each (request, head) block h

### stream-K ticket merge (splitkv_attn_sk_kernel merge_range: full decode 1<B<=64, PSEUDO sliding verify_1, sk-window sliding decode_32)
- **sk-ticket-merge-onepass** → neutral (within-ci), 1.6177x, mechanism no-base — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Built on master 2eb03f76 (no depends_on). ONE mechanism, the load structure of the in-kernel ticket merge in _SK_CUDA splitkv_attn_sk_kernel. Today merge_range 

### stream-K ticket merge onepass twin at sliding verify B=1 (the gate's timing inputs)
- **sk-onepass-verify1-timing-fix** → neutral (within-ci), 1.6219x, mechanism no-base — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Restack of sk-ticket-merge-onepass (neutral 1.6177 in check ba8c8a61; twins compiled, merge canary exact, full decode measured only x1.002-1.005) with ONE fix: 

## Open follow-ups

- full-decode-splitkv-s2-streamk-ticket-inproc: rewrite stage 2 of `full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the f
- full-verify-wgmma-s3-warpgroup-per-kvhead-inproc: rewrite stage 3 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master); stage-2 splitkv_verify_kernel A8 + splitkv_verify_combine B8`: read the replaced kern
- sliding-cluster-s2-decode-c4-pdl-inproc: rewrite stage 2 of `sliding layer decode at tp8/page 1, 1<=batch<=32: master Triton _sliding_fused_kernel (stage-1 sliding_cluster_kernel<1>)`: read the replaced kernels' own us per shape (profile top_kernels) against th
- full-decode-splitkv-s3-fused-preattn: rewrite stage 3 of `_pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and ch
- full-decode1-streamk-s3-two-level-ticket: rewrite stage 3 of `FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- full-verify-wgmma-s4-multitile-fix: rewrite stage 4 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`: read the replaced kernels' own us per shape (profile top_kernels) against
- full-decode1-sk4-cluster-dsmem-merge: rewrite stage 4 of `full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check 
- sliding-verify-sk-pseudo-decode-s1: rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8, 1<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`: read the replaced kernels' own us per shape (profile top_kernels) a
- full-decode1-sk-two-level-fused-preattn: rewrite stage 4 of `full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the
- full-verify-wgmma-s5-qfrag-reload: rewrite stage 5 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`: read the replaced kernels' own us per shape (profile top_kernels) against
- full-verify1-sk-two-level-token-group: rewrite stage 3 of `full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + norm-gate`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the 
- sliding-decode-sk-window-route: rewrite stage 2 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)`: read the replaced kernels' own us per shape (profile top_kernels) against th
- full-verify-wgmma-ws-s1-producer-split: rewrite stage 1 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)`: read the replaced kernels' own us per shape (profile top_k
- sliding-verify-wgmma-window-s1: rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8) at 1<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate (B=16/32), and the PSEUDO sk route (B=1)`: read the replaced kernels' o
- full-verify-wgmma-s6-split-ring-2cta: rewrite stage 6 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)`: read the replaced kernels' own us per shape (profile top_k
- full-decode-sk-s5-deferred-ticket-fault-fix: rewrite stage 5 of `full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the f
- full-decode-sk-s4-deferred-ticket-arrive: rewrite stage 4 of `full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the f

## Shelf (row wins below the headline bar; stack them)

Neutral results whose predicted rows all moved past ±2% (one run each: the rows drift 1.5-2.5%, so re-measure on today's master before stacking).

- **sliding-fused-triton-decode-batched-deep-splits** [sliding-layer attention (batched decode, split depth)] unproven-vs-best, 1.3087x, rows 2/2 hit, job `179dd013d3a347e4`: Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window an
- **full-decode1-cuda-splitkv-chunk320** [full-layer attention (batch-1 decode gate)] unproven-vs-best, 1.3954x, rows 1/1 hit, job `8d4771feca9b4e18`: Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through maste
- **full-decode1-cuda-splitkv-chunk640** [full-layer attention (batch-1 decode gate)] within-ci, 1.3805x, rows 1/1 hit, job `91d541778b8f45f3`: Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL
- **sliding-decode-blockn128-single-tile** [sliding-layer attention (batched decode, tile size)] within-ci, 1.3834x, rows 1/1 hit, job `88c62334775e48e4`: Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): 
- **sliding-decode32-splits-8** [sliding-layer attention (batched decode, split depth)] unproven-vs-best, 1.3973x, rows 1/1 hit, job `16d3fb5cae3f4141`: Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else
- **stack-streamk-ticket-cluster-c16-c4** [full + sliding decode attention (stack of two pending kernels)] beats-base-not-best, 1.4266x, rows 4/4 hit, job `195dcc1209b6457b`: Stack the two pending (user-held, not on master) decode wins on current master; they touch
- **restack-sk-pseudo-b1-cluster-c16-c4** [sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4)] unproven-vs-best, 1.4870x, rows 3/3 hit, job `eb6ff0cb701d4def`: Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticke
- **combo-b1-chain-c320-fused-pre-pdl** [combination (conjectural: full decode_1 chunk320 + fused pre-attention prologue + PDL to combine)] unproven-vs-best, 1.5295x, rows 1/1 hit, job `8713665f3064462c`: COMBINATION 2 of 3, a conjectural B=1 launch-chain pairing (not the top rows). Built on cu
- **combo-naive-c320-fd1-tiled-preattn-verify** [combination (naive top-row stack: full decode_1 chunk320 + verify tiled pre-attention)] within-ci, 1.5094x, rows 2/2 hit, job `b9d062aaa94c4cbf`: COMBINATION 1 of 3, the naive top-row stack. Built on current master 7e12891 (blob 5cf8fae

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end. Neutrals whose predicted rows all moved are on the Shelf instead.

- **preattention-group-tiled-programs** [pre-attention norm+rope] neutral, within-ci: Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq
- **full-fused-triton-decode1-gate** [full-layer attention (batch-1 decode gate)] neutral, within-ci: Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) thr
- **full-decode-cuda-fp8-qk-mma** [FA3 call (full-layer decode gather)] neutral, within-ci: One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kern
- **splitkv-combine-wide-mlp** [split-KV combine] neutral, within-ci: Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flig
- **paired-norm-gate-token-tiled** [output norm-gate] neutral, within-ci: Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs onl
- **preattention-tiled-gate-large-tokens** [pre-attention norm+rope] neutral, within-ci: Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch
- **splitkv-chunk2048-batch32** [FA3 call (full-layer decode gather)] neutral, within-ci: Shape-gated chunk size for master's CUDA split-KV path: at batch >= 32 use 2048-token work
- **splitkv-pdl-attn-to-combine** [launch overhead] neutral, within-ci: Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B
- **full-decode1-sk-two-level-deep-ring** [full-layer attention (batch-1 decode, stream-K ring depth)] neutral, within-ci: Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stage
- **full-verify-fa3-num-splits-band-gate** [FA3 call (full verify split count)] neutral, within-ci: Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 mai
- **sliding-verify1-pseudo-fused-preattn-rope** [sliding-layer attention (batch-1 verify, PSEUDO stream-K kernel, pre-attention fold)] neutral, within-ci: DIAGNOSIS action 3(ii). Stacked on merge-perkvhead-v3chain-sv32-tiled (build on its branch
- **sk-ticket-merge-onepass** [stream-K ticket merge (splitkv_attn_sk_kernel merge_range: full decode 1<B<=64, PSEUDO sliding verify_1, sk-window sliding decode_32)] neutral, within-ci: ONE mechanism, the load structure of the in-kernel ticket merge in _SK_CUDA splitkv_attn_s
- **sk-onepass-verify1-timing-fix** [stream-K ticket merge onepass twin at sliding verify B=1 (the gate's timing inputs)] neutral, within-ci: Restack of sk-ticket-merge-onepass (neutral 1.6177 in check ba8c8a61; twins compiled, merg
- **fa3-metadata-fold-preattn-full-verify** [FA3 call (scheduler metadata folded into pre-attention, full verify)] neutral, within-ci: Fold FA3's varlen scheduler metadata (what flash::prepare_varlen_num_blocks_kernel<NumWarp
- **full-verify-paged-gather-gate-kernel** [full-layer verify attention (FA3 call at verify_16 / verify_32)] neutral, within-ci: Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, pag
- **fa3-window-trimmed-page-table** [page table (sliding)] regression, regression: On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each 
- **fa3-scheduler-metadata-side-stream** [FA3 call (schedule, off critical path)] regression, regression: Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding 
- **sliding-fused-hoist-first-tile** [sliding-layer attention (batched decode, gather prologue)] regression, regression: In master _sliding_fused_kernel, issue the first old-key tile's gather before the query wo
- **sliding-decode-paired-kvhead-rows** [sliding-layer attention (batched decode, KV row gather)] regression, regression: One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding
- **sliding-verify-fa3-head-pairing** [FA3 call (head packing, sliding verify)] regression, regression: Config-only change in kernel_fn: extend master head-pairing predicate to sliding verify. T

## Queue and in flight

- queue: triton-window-attention-sliding [FA3 call] p5 hold
- queue: fa3-drop-cu-seqlens-decode [FA3 call] p5 hold
- queue: fused-full-combine-normgate [output norm-gate] p5 hold
- queue: full-decode-splitkv-s2-streamk-ticket [full-layer attention (decode, splitkv kernel A/B)] p1 hold
- queue: full-verify-wgmma-s3-warpgroup-per-kvhead [full-layer attention (verify, splitkv A8/B8)] p2 hold
- queue: sliding-cluster-s2-decode-c4-pdl [sliding-layer attention (decode, cluster kernel)] p2 hold
- queue: full-verify-wgmma-s4-sliding-window [sliding-layer attention (verify, wgmma stream-K kernel)] p2 hold
- queue: full-verify1-wgmma-s4-batch1-grid-sm [full-layer attention (batch-1 verify, wgmma stream-K kernel)] p2 hold
- queue: sliding-cluster-s3-verify-attn-only-phase-timed [sliding-layer attention (verify, cluster kernel)] p2 hold
- queue: sliding-verify1-triton-pseudo-decode [sliding-layer attention (batch-1 verify, Triton fused decode kernel)] p2 hold
- queue: full-verify1-sk-pseudo-decode [full-layer attention (batch-1 verify, stream-K pseudo-decode)] p3 hold
- queue: sliding-decode-sk-s3-fused-preattn-rope [sliding-layer attention (decode, stream-K windowed kernel, pre-attention fold)] p2 hold
- queue: preattention-tiled-every-verify-block [pre-attention norm+rope] p2 hold
- queue: sliding-verify-fa3-num-splits-band-gate [FA3 call (sliding verify split count)] p2 hold
- queue: full-verify1-wgmma-s7-grid-sm-two-level [full-layer attention (batch-1 verify, wgmma stream-K kernel, two-level ticket)] p2 hold
- queue: sk-full-line-gather [full-layer attention (decode 1<B<=64, stream-K kernel K/V gather mapping)] p1 hold
- queue: sliding-verify-wgmma-window-s2-split-combine [sliding-layer attention (verify B=2..32, wgmma window kernel merge)] p1 hold
- queue: c320-combine-onepass [full decode_1 split-KV combine (c320 chain)] p2 hold
- queue: c320-pdl-early-trigger [launch overhead (full decode_1 c320 kernel A -> combine PDL)] p2 hold
- queue: combo-sk-full-line-c320-tail [combination (full decode B>=2 gather mapping + full decode_1 combine tail)] p2 hold
- queue: sliding-verify-wgmma-window-s2b-fast-merge [sliding-layer attention (verify B=2..32, wgmma window kernel merge)] p3 hold
- queue: c320-attn-full-line-gather [full-layer attention (batch-1 decode, c320 kernel A K/V gather mapping)] p3 hold
- queue: sliding-verify-wgmma-split-combine-on-v3chain [sliding-layer attention (verify B=2..32, wgmma window kernel merge) on the v3chain base] p3 hold
- queue: sliding-decode1-perkvhead-launch-select [sliding-layer attention (batch-1 decode, cluster kernel launch configuration)] p4 hold
- queue: sk-full-line-5stage-select [full-layer attention (decode 1<B<=64, stream-K ring depth after the whole-line remap)] p4 hold
- queue: sliding-decode-band2-sk-gate [sliding-layer attention (decode B=2..16, stream-K window route gate)] p4 hold
- queue: full-verify-wgmma-s7-split-combine [full-layer attention (verify B=2..64, wgmma stream-K kernel merge)] p4 hold
- queue: sliding-decode1-ph-push-merge [sliding-layer attention (batch-1 decode, PERHEAD cluster DSMEM merge: push instead of pull)] p1 hold
- queue: attn-fast-math-epilogues [output norm-gate (CUDA epilogue math: o/L, RMSNorm reciprocal sqrt, sigmoid) in the routed attention kernels] p3 hold
- queue: combo-tail-onepass-push-fastepi [combination (stream-K ticket merge onepass + PERHEAD push merge + fast-math epilogues: the serialized merge/epilogue tails)] p2 hold
- queue: full-verify-paged-gather-gate-kernel-r2-cf [full-layer verify attention (FA3 call at verify_16 / verify_32)] p1 queued
- 17 stale holds expired to `done/expired/` (not run)
