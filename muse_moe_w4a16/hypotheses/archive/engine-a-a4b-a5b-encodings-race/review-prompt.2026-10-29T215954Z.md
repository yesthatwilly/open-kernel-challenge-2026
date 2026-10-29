You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_moe_w4a16/hypotheses/queue/2026-10-29T215527Z-engine-a-a4b-a5b-encodings-race.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_moe_w4a16/hypotheses/CLAIMS.md — the curated claims, one line per finding, tagged [landed] [lead] [shelf] [unresolved]
   [node-only] [refuted] [superseded]; the newest line per mechanism is the current one. Claims that share a
   distinctive word with this hypothesis (a pre-filter; read the whole file anyway):
   - muse_moe_w4a16/hypotheses/CLAIMS.md:60 (dq2): - [refuted] Engine S streamed cold path (es) replacing Engine A / the M-cell: Gate 0 (e) dq2+RS 29.8 el/clk/SM sits in the "< 30" band; judge es16.4.2 27.5 el/clk/SM is x1.02 over Engine A's ~27 | rows: load=decode_16, load=decode_32, load=verify_1, load=verify_16, load=verify_32 | evidence: H100 run 20261027-112303/112308-gpunode06 (KSTEP:70-80); judge check 5a64104026ab4f91 1.7199, 20/20 .text hashes equal the node's (KSTEP:92-133; LOG:1364-1374) | board: 1.7187 | blob: - | reopen: the ladder's P1 (3-4 consumer WGs: S3u / S4 / D2) lifts the dequant rate | 2026-10-27 h100
   - muse_moe_w4a16/hypotheses/CLAIMS.md:74 (dq2): - [landed] L3: paired scale decode in Engine A's dq2 consumers (2-source prmt + HMUL2 half selectors; bit-identical; -2.0 ALU instr/step, from the eac SASS census) | rows: all | evidence: eac bench x1.119 (T=8) / x1.097 (T=16) (H100:948-960); check f2100f49 1.8286 (every row +2.3-7.1%), scored 261cfe56537241f6 1.8249 (LOG:1748-1762); land 869cd463 | board: 1.8249 | blob: 223f43ab | reopen: - | 2026-10-28 campaign
   - muse_moe_w4a16/hypotheses/CLAIMS.md:79 (ea6): - [landed] X1 restricted to the v6 family at T=1 (eak::consume<2, 1> only at ea6's call site; v1-v4 unchanged) | rows: load=decode_1 | evidence: check 2e78be8b 1.906 (decode 1 29.6 -> 29.0 us, verify 1 79.3 -> 78.8 us), scored 0b49b1aae6a04acb 1.8941 (LOG:1930-1946, 1952); land 2ec8090f | board: 1.8941 | blob: 54068229 | reopen: - | 2026-10-28 campaign
   - muse_moe_w4a16/hypotheses/CLAIMS.md:90 (ea6): - [refuted] T=1 w2 prefetch to L2 before griddepcontrol.wait in ea6 K2 | rows: load=decode_1 | evidence: gpunode06 t1pf r1/r2 (runs 20261028-220821/-221010): every arm slower at k3 (+1.3..3.0 us) and k4 (+5..7.5); 2 records are already resident before the wait on 99-100% of CTAs; the K2 tail is bound by consume rate, not by the PDL wait (h100-t1-1029) | board: 1.950 | blob: a90ff05c | reopen: - | 2026-10-29 h100-t1-1029
   - muse_moe_w4a16/hypotheses/CLAIMS.md:92 (ea6): - [refuted] t1-deep-ring: ea6 record ring at depth 4/5 (1 CTA/SM) for T=1 (EP1 / v6si gate R0) | rows: load=decode_1 | evidence: check d1b5d75d30b141ee (campaign/eb-v6deep 1394cc63) [r0] VERDICT KILL D=4/5: rec5 1.327 vs rec3 1.306 us (d5/d3 1.016), K1d5 n2 14.24 (bar 13.2); gpunode06 D5/D3 1.044 (h100-t1-1029) | board: 1.950 | blob: a90ff05c | reopen: the per-record consume rate (~1.3 us/36,864-B record, 8 consumer warps) is raised first | 2026-10-29 campaign
   - muse_moe_w4a16/hypotheses/CLAIMS.md:97 (ea6): - [landed] stack2: v6pe2 two-wave ea6 K2 at T=1 + EPLF stream-K fix-up with fixed roles (routed at T16) + v4x-routes robust class-128 routing | rows: load=decode_1 load=decode_16 load=verify_16 | evidence: job 8418677e467d4fb7 (okc status 8418677e467d4fb7), check 456d90673f4346d1, master 9d66b13def | board: 1.9521 | blob: 691018b5 | reopen: - | 2026-10-29 campaign
   - muse_moe_w4a16/hypotheses/CLAIMS.md:104 (dequant, dq2): - [lead] dq2 A3 dequant in the wgmma hot half (wg_kernel dq_stage): 35.9 -> ~27.9 instr per k16 step per warp, bit-exact; shortens the serial dequant->wgmma->wait chain | rows: load=verify_16, load=verify_32 | evidence: H100 gpunode06 hh probe (takemyjob hopper/hh-1029 e942c40; research/h100-hothalf-1029.md): hot-half pair A3/master x0.902-0.923 T=256, x0.894-0.898 T=128, moe judge draws x0.866/0.890; est decode +0.7%, moe +0.8% headline (not judge-checked); also w2 1.45 waves (tail 22-27% of T=256 pair, upper bound) and 576 token columns computed for 386 routes | board: 1.952 | blob: 691018b5 | reopen: stack with a w2 wave/tail fix and per-expert N-tile (padding) fix to >= 2.5%; judge check of the stack | 2026-10-29 h100
   - muse_moe_w4a16/hypotheses/CLAIMS.md:105 (a5b, race, wh5ds): - [landed] hot half wh5ds at T=256 (A3 + A5b codes/scales + per-stage descriptor + split commit, wh exact-N/ring) in wg_kernel<G,14>, routed by isolated-pair race + fork no-loss gate (hothalf-wh-a5b-port) | rows: load=verify_32 | evidence: check 54b70249868f468b 1.9827 (verify_32 291.1 -> 274.4 us; pair m 115.1 -> wh5ds 92.8 us; forked kfn .887-.953); scored a5318a1f98984a5f 1.9748 [1.945, 2.004]; H100 hh2/hh3 | board: 1.9748 | blob: 37ea90fc | reopen: - | 2026-10-29 churn
   - muse_moe_w4a16/hypotheses/CLAIMS.md:106 (a4b, a5b, dq2, ea6): - [lead] Engine A dq2 with A4b code re-encoding (E4) / A4b + A5b packed scales (E5): lossless, bit-exact; per-kernel arm choice (E4 at T<=32 and ea6, E5 at the T=128/256 cold half) | rows: load=decode_1, load=decode_8, load=decode_16, load=decode_32, load=verify_16, load=verify_32 | evidence: H100 gpunode06 GPU 3 (user-approved run, takemyjob hopper/ea-enc-1029 d6f55c5; runs/20261029-144741-gpunode06), 2 replicates x 10 rounds: consume-only B/E4 T8 1.05-1.08, T16 w13 1.067-1.090, T32 w13 1.079-1.086, cold half 1.04-1.05, ea6 K1 1.076-1.097; B/E5 T8-32 1.03-1.07, cold half 1.05-1.07; normal mode ~1.00 (HBM wall on the H100); exact at every draw, err 0, tickets 0; SASS -9..-12% issued/k16; H200 estimated consumer-bound (consumer-alone 0.91-0.97x of HBM floor) | board: 1.9748 | blob: 37ea90fc | reopen: judge check with an in-prepare exact race per kernel class (E4 vs E5 vs master): the judge decides whether the consume gain surfaces on the H200 | 2026-10-29 h100
3. muse_moe_w4a16/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
4. The active campaign's findings: campaigns/2026-10-26-expert-bandwidth/FINDINGS.md, campaigns/2026-10-26-expert-bandwidth/DISTILL.md, and the top 150 lines of campaigns/2026-10-26-expert-bandwidth/LOG.md (the Now block). Its landings, kills and H100 transfer readings are not in
   churn's own files (most of master's recent submission.py commits are campaign landings).
5. muse_moe_w4a16/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI). The hypothesis's freshness:
   mint 2026-10-29T215527Z on board 1.9748, blob 37ea90fc; BEST now 1.9748 (job a5318a1f98984a5f), master blob 37ea90fc; moved 0.00% against a 1.24% margin; freshness gate: fresh; value case: EV 0.6%.
6. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_moe_w4a16",
 "landed": [
  {
   "slug": "sglang-marlin-path-parity",
   "area": "MoE align/dispatch",
   "outcome": "gain",
   "speedup": 0.9887547325362586,
   "promotion_commit": "a646cb4151c0a45f926a49b68de21789e11a8102",
   "patch": "muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.patch",
   "md": "muse_moe_w4a16/hypotheses/done/gain/2026-10-23T225501Z-sglang-marlin-path-parity.md",
   "change": "Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's Marlin pipeline from baseline.py: prepare() builds the Marlin bank with prepare_moe_nvfp4_layer_for_marlin over experts [local_start, local_s"
  },
  {
   "slug": "post-expert-norm-triton-small-m",
   "area": "post-expert norm+combine",
   "outcome": "gain",
   "speedup": 1.1297159216968498,
   "promotion_commit": "192f225d8fbfa65bb838ce448c7d6b5e2c522ecd",
   "patch": "muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.patch",
   "md": "muse_moe_w4a16/hypotheses/done/gain/2026-10-24T075126Z-post-expert-norm-triton-small-m.md",
   "change": "For num_tokens < 32 only (bs=1, 8, 16; the bs=4 test too), replace the call to SGLang's fused_post_expert_norm_weighted_sum at the end of kernel_fn with one launch of a new Triton kernel _post_norm_combine_kernel defined in submission.py; bs>=32 keeps SGLang's call untouched as the in-run control. E"
  },
  {
   "slug": "marlin-block-m-two-smaller-on-postnorm",
   "area": "w13 marlin",
   "outcome": "gain",
   "speedup": 1.2535643764852882,
   "promotion_commit": "cefc282ebdcb22f4a4e60d2acf151b862ee831ad",
   "patch": "muse_moe_w4a16/hypotheses/done/gain/marlin-block-m-two-smaller-on-postnorm.patch",
   "md": "muse_moe_w4a16/hypotheses/done/gain/2026-10-24T112137Z-marlin-block-m-two-smaller-on-postnorm.md",
   "change": "Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the pre-post-norm base, job ed837a4a46d44a95; patch at muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn o"
  },
  {
   "slug": "triton-map-align-on-block-m-restack",
   "area": "MoE align/dispatch + block_size_m pick",
   "outcome": "gain",
   "speedup": 1.3116740840932535,
   "promotion_commit": "8440ae5baaeda52f365611b273035480a61b2325",
   "patch": "muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.patch",
   "md": "muse_moe_w4a16/hypotheses/done/gain/2026-10-24T132314Z-triton-map-align-on-block-m-restack.md",
   "change": "Stack the two measured, shape-disjoint moe wins: start from master (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_tokens >= 128) and add triton-map-align-small-m-port (1.185x on the older post-norm base, gained 4.5-12.6% at bs=1/8/16/32, flat at bs=128/256) "
  },
  {
   "slug": "align-parallel-per-expert-verify",
   "area": "MoE align/dispatch",
   "outcome": "neutral",
   "speedup": 1.3249153782392282,
   "promotion_commit": "1a5da3d22ce48904de76852a1d585563f4c53048",
   "patch": "muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch",
   "md": "muse_moe_w4a16/hypotheses/done/neutral/2026-10-24T164817Z-align-parallel-per-expert-verify.md",
   "change": "For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when expert_map is not None, replace master's else-branch (topk_ids.to(int32), the expert_map[local_ids] gather = at::native index_elementwise 3.2 us, the at::native unrolled_elementwise 2.8 us, and moe_align_bl"
  },
  {
   "slug": "w2-marlin-atomic-add-small-m",
   "area": "w2 marlin",
   "outcome": "neutral",
   "speedup": 1.344525539017997,
   "promotion_commit": "9080a71e9183df77bc71f28ecfa0f635b7d0aff0",
   "patch": "muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.patch",
   "md": "muse_moe_w4a16/hypotheses/done/neutral/2026-10-25T011401Z-w2-marlin-atomic-add-small-m.md",
   "change": "For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_wna16_marlin_gemm call (w2) only, by building a separate kwargs dict from master's common for that call; the w13 call and every num_tokens > "
  },
  {
   "slug": "align-two-phase-port-from17",
   "area": "MoE align/dispatch",
   "outcome": "neutral",
   "speedup": 1.3700229697096111,
   "promotion_commit": "a0e5264057e1b24beb5aa7c2b04a7c2e1f14b7e5",
   "patch": "muse_moe_w4a16/hypotheses/done/neutral/align-two-phase-port-from17.patch",
   "md": "muse_moe_w4a16/hypotheses/done/neutral/2026-10-25T145952Z-align-two-phase-port-from17.md",
   "change": "Built on master (9080a71, submission_blob 60dfe35d); MoE align only, at 17 <= num_tokens <= 512 (decode_32 bs=32, verify_16 bs=128, verify_32 bs=256, and the bs=33, bs=257 and skewed bs=128 tests). num_tokens <= 16 keeps master's _align_small byte for byte, and Marlin, silu_and_mul, the w2 atomic pa"
  },
  {
   "slug": "wg-pdl-gather-w13-w2-chain",
   "area": "launch overhead",
   "outcome": "neutral",
   "speedup": 1.3752355517671193,
   "promotion_commit": "f7eac2c3ca666ac6dcd24425163ae410a5ca60f5",
   "patch": "muse_moe_w4a16/hypotheses/done/neutral/wg-pdl-gather-w13-w2-chain.patch",
   "md": "muse_moe_w4a16/hypotheses/done/neutral/2026-10-25T153417Z-wg-pdl-gather-w13-w2-chain.md",
   "change": "Programmatic dependent launch across master's wgmma chain at 128 <= T <= 512 (verify_32 bs=256 runs it; verify_16 only if its bucket flag is on). Kernels, tiling, numerics, align and post-norm are unchanged. MECHANISM: wg_kernel<w13+silu> runs ~15.5 waves of ~19.5 us items at bs=256 (303 us / 2048 i"
  },
  {
   "slug": "Engine A (CUDA W4A16 expert engine, v1 / v2) replaces Marlin for T<=32 (decode-first campaign)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5269,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:41",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1; evidence: ec719761 (moe scored 1.5269); later rounds c8b77a59, 2fc609f3"
  },
  {
   "slug": "Engine A v4 (8 consumer warps) at T=2..32, T=128 and the T=256 cold half",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6389,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:43",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_16, load=decode_32, load=verify_1, load=verify_16; evidence: check 54f63952015c4d3b 1.6412 (w13 kernel-only +5-10% TB/s), scored 35b55c74ee964187 1.6389 (LOG:309-317); land 20d55811"
  },
  {
   "slug": "X2: wgmma-hot / Engine A v4-cold hybrid at T=256 (H=-16: experts with > 16 routes on wgmma)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6851,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:44",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_32; evidence: check 0f4605dca83744c7 1.6845 (T=256 row 391.1 -> 339.7 us), scored e854db4683dc495f 1.6851 (LOG:358-362); land 72e6bd6d"
  },
  {
   "slug": "v4c: Engine A v4 with a cost-weighted stream-K split, raced against v4 in every gate (ea4 at T=128 299 -> 251 us)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.7076,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:49",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_16, load=verify_32, load=decode_32; evidence: eacf0722; check 9f8a518ea6c04d20 1.7112, scored cf937dd3656c4193 1.7076 (LOG:709-710, 720-721, 741); land 76815f82"
  },
  {
   "slug": "W6-M: the hot half on a fork stream concurrent with the cold half at M=256 (H-16f, fork gate 1.01 every round)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.7187,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:55",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_32, load=verify_16; evidence: check a80dda1ab8c14f48 1.6978 (fork x1.024-1.060 over serial H-16), scored 2a7b5d2030894c4a 1.7187 (verify 32 x1.023, verify 16 x1.016) (LOG:1118-1121, 1132-1135); land 86e4a4d9"
  },
  {
   "slug": "S3: Engine A v6 at T=1 (grid = SMs with whole-record ownership, per-expert K2 blocks, K2 co-resident with K1); per extra expert 8.53 / 8.71 vs 10.2 us on the judge",
   "area": null,
   "outcome": "landed",
   "speedup": 1.734,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:66",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1; evidence: check fd0e92717ec0400e 1.7377 (decode 1 35.4 -> 33.3 us), scored 426218dc889d45d0 1.7340 (LOG:1566-1569, 1590-1596, 1604-1605); land 2abc122e; on the node v6 removed the T=1 alternation but not the slope (H100:286-290)"
  },
  {
   "slug": "carrier: CUDA combine at class 128 (M456: +1.04% / +0.85% vs the identity arm on the node; judge verify 16 x1.023 check / x1.018 scored, so the transfer held) + pins v4 at classes 8 / 16",
   "area": null,
   "outcome": "landed",
   "speedup": 1.7345,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:69",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_16, load=verify_1, load=decode_16; evidence: check be4440e24e244103 1.7457, scored 12d17a4696844630 1.7345 (LOG:1655-1671; H100:732-739); land 81c9802e"
  },
  {
   "slug": "L3: paired scale decode in Engine A's dq2 consumers (2-source prmt + HMUL2 half selectors; bit-identical; -2.0 ALU instr/step, from the eac SASS census)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.8249,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:74",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: all; evidence: eac bench x1.119 (T=8) / x1.097 (T=16) (H100:948-960); check f2100f49 1.8286 (every row +2.3-7.1%), scored 261cfe56537241f6 1.8249 (LOG:1748-1762); land 869cd463"
  },
  {
   "slug": "L4: L3's pair add as an IMAD on the FMA pipe (run-time 16 in inline PTX; ALU 10.44 -> 9.56 per step), bit-identical",
   "area": null,
   "outcome": "landed",
   "speedup": 1.8719,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:76",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: all; evidence: eac l4 x1.193 / x1.114 (H100:953); check 7607a1c39e134a4f 1.887, scored 427a6f289e6745ac 1.8719 (LOG:1783-1799, 1809); land bfb3214b"
  },
  {
   "slug": "fc: v6 T=1 fixed-cost arms raced vs v6, v6pe routes (consumer warps stage B with __ldcg; early combine kernel with w / ids / tw loaded before the PDL wait), bitwise v6",
   "area": null,
   "outcome": "landed",
   "speedup": 1.8822,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:78",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1; evidence: research/fc-spec-2026-10-28.md; check dec43c9cd1ac4235 1.880 (decode 1 32.0 -> 29.6 us, x1.084), scored e2827c9e800549cd 1.8822 (decode 1 30.7 us, about its run-to-run spread) (LOG:1817-1830, 1863-1879); land 52ab6fde"
  },
  {
   "slug": "X1 restricted to the v6 family at T=1 (eak::consume<2, 1> only at ea6's call site; v1-v4 unchanged)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.8941,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:79",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1; evidence: check 2e78be8b 1.906 (decode 1 29.6 -> 29.0 us, verify 1 79.3 -> 78.8 us), scored 0b49b1aae6a04acb 1.8941 (LOG:1930-1946, 1952); land 2ec8090f"
  },
  {
   "slug": "engineA-v4x-s1: Engine A v4x twins routed per class (d first, then s upgrade): T=2..16 v4xs, T=17..32 v4cxs, T=128 v4cxs+combine",
   "area": null,
   "outcome": "landed",
   "speedup": 1.925,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:84",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_16 load=decode_32 load=verify_1 load=verify_16; evidence: job af154932df744959 (okc status af154932df744959), master 37a5921a77"
  },
  {
   "slug": "v4x256: the T=256 hybrid cold half routed to the v4x twins (v4cxd, then v4cxs)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.9253,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:85",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_32; evidence: job 787a28152682498d (okc status 787a28152682498d), check c04363bbe0ae4e0a, master a72163730f"
  },
  {
   "slug": "dedicated priority=-1 W6-M fork stream (_rt_fork_stream) instead of the torch pool stream",
   "area": null,
   "outcome": "landed",
   "speedup": 1.95,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:89",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_32; evidence: reopened at priority 0 as v4x256c: check 03d2c99e 1.938 (fork x1.034-1.069, alt 2), scored dceed39d 1.950; gpunode06 eb-forkprio prio 0 337-338 vs -1 356-358 us"
  },
  {
   "slug": "v4x256c: W6-M fork stream at priority 0, never the capture stream (_rt_fork_stream/_rt_fork_pick), plus v4cxs qualifying alone on the T=256 cold half",
   "area": null,
   "outcome": "landed",
   "speedup": 1.9502,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:88",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_32; evidence: job dceed39d08a14c73 (okc status dceed39d08a14c73), check 03d2c99e043a4a6a, master a96fae0582"
  },
  {
   "slug": "stack2: v6pe2 two-wave ea6 K2 at T=1 + EPLF stream-K fix-up with fixed roles (routed at T16) + v4x-routes robust class-128 routing",
   "area": null,
   "outcome": "landed",
   "speedup": 1.9521,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_moe_w4a16/hypotheses/CLAIMS.md:97",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1 load=decode_16 load=verify_16; evidence: job 8418677e467d4fb7 (okc status 8418677e467d4fb7), check 456d90673f4346d1, master 9d66b13def"
  },
  {
   "slug": "hothalf-wh-a5b-port",
   "area": "moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256",
   "outcome": "gain",
   "speedup": 1.9826789715234687,
   "promotion_commit": "7da461db12b3c802a605fcdcaf2208314fe6dc60",
   "patch": "muse_moe_w4a16/hypotheses/done/gain/hothalf-wh-a5b-port.patch",
   "md": "muse_moe_w4a16/hypotheses/done/gain/2026-10-29T195507Z-hothalf-wh-a5b-port.md",
   "change": "Port decode's routed wh hot half (module muse_decode_wg_wh from branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wg_kernel<G,V> with D0 control V=0 and wh V=7 = A3 dequant on a word-permuted whole bank + exact-N tiles + R4 ring wait<2> for N>=48; background-thread build; isola"
  }
 ]
}
7. muse_moe_w4a16/submission.py (master) — the kernel the hypothesis will be built on (its base).
8. muse_moe_w4a16/baseline.py — SGLang's own pipeline (the named parts), muse_moe_w4a16/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
9. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): arlin-exec-config-autotune-mid-m, g-producer-direct-x-rows-no-gather, g2-pingpong-consumers-occ2-b256, g2-stream-k-tail-floor-g-over-r, g2-w13-occ2-nmax128-stg2-hot, mbf-combine-fused-w2-epilogue-t8-32-cf, ngine-a-a4b-a5b-encodings-race, uda-fused-moe-midm-single-launch-routed, uda-fused-split-i-moe-tiny-t, uda-nvfp4-wgmma-rega-grouped-sm90a-rebuild, uda-nvfp4-wgmma-rega-producer-scale-expand, uda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix

Answer these, adversarially:
A. **Justification.** Is the stated mechanism plausible on this hardware for these shapes? Does
   the profile data in FINDINGS support it (which kernels, how many µs, how far from the
   speed-of-light floor)? Are the `expected` shapes and directions consistent with the shape
   weights and floors in TASK.md? Name the weakest link in the argument. Check the `prior`
   field (the closest earlier attempt and what differs) against that slug's file and outcome,
   and FINDINGS' "Do not try" list: is the difference real, or is this the same mechanism again?
   A `prior: none` that has an obvious earlier attempt is grounds for hold.
B. **Confounding by what already landed.** For each landed hypothesis that touches the same
   area, the same launch sequence, the same memory traffic, or the same numerical path: could a
   gain or loss measured for this hypothesis be caused by the interaction rather than by the
   change itself? Would the change behave differently if that landed hypothesis were absent or
   tuned differently?
C. **Interference and A/B.** Decide specifically whether an additional run is warranted to make
   *this* hypothesis's single result meaningful: (i) this change with landed hypothesis X
   reverted (`ab-revert`), (ii) this change with X tuned differently (`ab-tune`), or (iii) a
   control that isolates one of the hypothesis's own sub-changes (`control`). A companion is
   warranted only when, without it, both a gain and a loss would be ambiguous. At most two.
   Each must be implementable from the parent's branch: say exactly what to revert (the patch
   path from the landed list, applied in reverse) or what to tune, and what stays.
D. **Rules.** Flag anything in the change that could break the competition rules (precision,
   caching derived from inputs, every-key attention, host syncs, work outside the timed call).
E. **Claims and freshness (a hold, not a note).** Hold when (i) the mechanism's newest claim is
   [refuted] or [superseded] and the hypothesis does not show that claim's `reopen` condition is met;
   (ii) a [landed] claim (or a campaign landing in step 4) already owns the rows it targets with a
   different mechanism, so it is built on a base that no longer exists; (iii) the board moved past the
   margin since `best_at_mint` / the last review (the freshness line in step 5) and the value case
   (`value_gain`, `expected`) was not re-derived on today's rows. A [shelf] mechanism proceeds only as part
   of a stack whose rows sum to >= 2.5% headline; an [unresolved] one only with the extra runs or pinned
   routes its `reopen` names; a [node-only] one only when the change is per-SM work (per-SM cycles transfer
   to the judge H200; HBM rates do not). Cite the CLAIMS.md line numbers you relied on.

If the hypothesis has `kind: rewrite`, it replaces a whole subsystem over several staged runs and
is judged on the replaced kernels' own time against the incumbent and the speed-of-light floor (its
`## Milestones`), not on the headline. For a rewrite, weigh instead: is the design sound for this
hardware and these shapes (parallelism, bytes in flight, dequant cost, tensor-core path), is the
correctness plan adequate, is the stage's milestone readable from the profile, and what is the
single biggest risk to fix in the next stage. Do not hold a rewrite because one run cannot
attribute a headline change or because the first stage may be slower than the incumbent; hold it
only for a rules problem or a design that cannot reach its floor. Ambition is not a defect: the
user keeps 40% of the day's checks for rewrites, so a large overhaul is judged on whether its
target is physically reachable and its plan sound, not on how much it changes. Companions are rarely useful for
a rewrite: the next stage is the follow-up.

Return ONLY a JSON object, no prose around it, with this schema:
{
 "model": "opus",
 "verdict": "proceed" | "hold" | "proceed-with-companions",
 "summary": "one or two sentences",
 "justification": "your assessment of A, including the weakest link",
 "confounders": ["one line per landed hypothesis that confounds, with why", ...],
 "interference": "your reasoning for C: why companions are or are not needed",
 "rules_flags": ["...", ...],
 "freshness": "E: the claims (CLAIMS.md:N) and board/blob movement you checked, and what they decided",
 "companions": [
   {"slug": "<parent-slug>-ab-revert-<x>", "kind": "ab-revert" | "ab-tune" | "control",
    "area": "...", "change": "a paragraph implementable from the parent's branch without asking",
    "expected": [{"shapes": "<glob over shape keys>", "direction": "faster|slower|flat", "why": "..."}],
    "reason": "what this companion disambiguates", "priority": 1-5, "risk": "low|medium|high"}
 ],
 "open_questions": ["...", ...]
}
`hold` only when the hypothesis is unjustifiable, duplicates a landed/do-not-retry item, fails E, or
would break a rule. Prefer `proceed` with sharp notes over a timid `hold`. Put a one-line reason in
`summary` that starts with the E clause when E decided it (e.g. "E(i): CLAIMS.md:23 refuted ...").
