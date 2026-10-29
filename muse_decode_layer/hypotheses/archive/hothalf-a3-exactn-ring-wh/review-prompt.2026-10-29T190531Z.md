You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_decode_layer/hypotheses/queue/2026-10-29T190020Z-hothalf-a3-exactn-ring-wh.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_decode_layer/hypotheses/CLAIMS.md — the curated claims, one line per finding, tagged [landed] [lead] [shelf] [unresolved]
   [node-only] [refuted] [superseded]; the newest line per mechanism is the current one. Claims that share a
   distinctive word with this hypothesis (a pre-filter; read the whole file anyway):
   - muse_decode_layer/hypotheses/CLAIMS.md:74 (dq2): - [landed] L3 paired scale decode in Engine A's dq2 consumers (bit-identical, -2 ALU instr/step) + Engine A v6 (S3: whole-record ownership, co-resident K2) at T=1 | rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1 | evidence: 31e0aad5 1.637 (L3 + v6), 7f9712c4 1.624 (L3 alone): decode 32 sliding x1.07, verify 16 sliding x1.05 (LOG:1767-1769); scored adb9c3436aad4e1d 1.6262 (LOG:1777-1780); land 622b97bb | board: 1.6262 | blob: f34ed172 | reopen: - | 2026-10-28 campaign
   - muse_decode_layer/hypotheses/CLAIMS.md:98 (dq2, stage): - [lead] dq2 A3 dequant in the wgmma hot half (wg_kernel dq_stage): 35.9 -> ~27.9 instr per k16 step per warp, bit-exact; shortens the serial dequant->wgmma->wait chain | rows: load=verify_16, load=verify_32 | evidence: H100 gpunode06 hh probe (takemyjob hopper/hh-1029 e942c40; research/h100-hothalf-1029.md): hot-half pair A3/master x0.902-0.923 T=256, x0.894-0.898 T=128, moe judge draws x0.866/0.890; est decode +0.7%, moe +0.8% headline (not judge-checked); also w2 1.45 waves (tail 22-27% of T=256 pair, upper bound) and 576 token columns computed for 386 routes | board: 1.690 | blob: 315119e2 | reopen: stack with a w2 wave/tail fix and per-expert N-tile (padding) fix to >= 2.5%; judge check of the stack | 2026-10-29 h100
3. muse_decode_layer/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
4. The active campaign's findings: campaigns/2026-10-26-expert-bandwidth/FINDINGS.md, campaigns/2026-10-26-expert-bandwidth/DISTILL.md, and the top 150 lines of campaigns/2026-10-26-expert-bandwidth/LOG.md (the Now block). Its landings, kills and H100 transfer readings are not in
   churn's own files (most of master's recent submission.py commits are campaign landings).
5. muse_decode_layer/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI). The hypothesis's freshness:
   mint 2026-10-29T190020Z on board 1.6904, blob 315119e2; BEST now 1.6904 (job 7cc034f917274439), master blob 315119e2; moved 0.00% against a 1.24% margin; freshness gate: fresh; value case: EV 0.72%.
6. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_decode_layer",
 "landed": [
  {
   "slug": "sglang-pipeline-parity-router-hilo-m8",
   "area": "router GEMM",
   "outcome": "gain",
   "speedup": 1.021632132299037,
   "promotion_commit": "164c1d27648cffdd4d7f5a04f0d6f582e1f9ac56",
   "patch": "muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-23T225318Z-sglang-pipeline-parity-router-hilo-m8.md",
   "change": "Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules, the cos|sin table, the "
  },
  {
   "slug": "marlin-block-m-cap-16",
   "area": "w13 marlin",
   "outcome": "gain",
   "speedup": 1.0620874943175542,
   "promotion_commit": "ef7dea6cac43678c09ae91b20ece2c40e0670c4b",
   "patch": "muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-23T225545Z-marlin-block-m-cap-16.md",
   "change": "Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9, which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as chosen and clamps the"
  },
  {
   "slug": "fa3-hdim128-head-pairing-batched",
   "area": "FA3 call",
   "outcome": "neutral",
   "speedup": 1.0873503247934666,
   "promotion_commit": "09da1e0d71f2fca87bd222e2507f638ddb81d38e",
   "patch": "muse_decode_layer/hypotheses/done/neutral/fa3-hdim128-head-pairing-batched.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-24T103909Z-fa3-hdim128-head-pairing-batched.md",
   "change": "Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attention layers with batch > 1. Gate, config only: paired = cfg[\"window\"] == 0 and cfg[\"batch\"] > 1 and cfg[\"num_kv_heads\"] % 2 == 0 (never a tenso"
  },
  {
   "slug": "post-expert-norm-triton-small-t",
   "area": "post-expert norm+combine",
   "outcome": "gain",
   "speedup": 1.1635238875350442,
   "promotion_commit": "ea896254423db20bfb9c584010c4b8ca73228a23",
   "patch": "muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-24T140057Z-post-expert-norm-triton-small-t.md",
   "change": "Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.patch) into muse_decode_layer/submission.py _experts. Today _experts ends with SGLang's fused_post_expert_norm_weighted_sum(y, t"
  },
  {
   "slug": "small-t-launch-minimal-dense-chain-pdl",
   "area": "whole-layer fusion",
   "outcome": "gain",
   "speedup": 1.228328123558382,
   "promotion_commit": "f1f2f35cf76890e924a0d0ae1576b1050ed9020f",
   "patch": "muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-24T172138Z-small-t-launch-minimal-dense-chain-pdl.md",
   "change": "WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.7 us, the dense GEMMs ~55 us against a 28.8 us weight floor (qkv 37.7 + o 16.8 + router 8.4 + pooler 67.1 + unpooler slice 8.4 MB), FA3 ~14 "
  },
  {
   "slug": "stack-align-par-sidestream-sliding1",
   "area": "T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention",
   "outcome": "neutral",
   "speedup": 1.2495238726339006,
   "promotion_commit": "8fa8fbf0c89a2b6323c511c25b8616c0290dde23",
   "patch": "muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-24T233721Z-stack-align-par-sidestream-sliding1.md",
   "change": "Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side stream and passes the result to _experts via"
  },
  {
   "slug": "full-decode-splitkv-restack-on-pdl",
   "area": "FA3 call (full decode, batch > 1)",
   "outcome": "neutral",
   "speedup": 1.2680313318618024,
   "promotion_commit": "b047510f64cf1919b8313cb88b2f6fe17c1c2305",
   "patch": "muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-24T224525Z-full-decode-splitkv-restack-on-pdl.md",
   "change": "Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decode_32 535.8 -> 490.2 (+10.3%), every other shape flat) onto current master (small-t-launch-minimal-dense-chain-pdl, f1f2f35), the way moe sta"
  },
  {
   "slug": "w2-marlin-atomic-add-t1-port",
   "area": "w2 marlin (T=1)",
   "outcome": "neutral",
   "speedup": 1.2855726646854595,
   "promotion_commit": "00e36e3a74224d6c0cc4f10708dfd9e759d8ee84",
   "patch": "muse_decode_layer/hypotheses/done/neutral/w2-marlin-atomic-add-t1-port.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-25T042121Z-w2-marlin-atomic-add-t1-port.md",
   "change": "Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16's w2-marlin-atomic-add-small-m (branch hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m; bs=1 48.8 -> 44.9 us, slow mode 44/100 -> 10/100 rep"
  },
  {
   "slug": "dense-chain-s3-t1-q-sk2-o-route-unpool",
   "area": "whole-layer fusion (T=1 dense chain: qkv/o GEMM side + routing R + unpooler)",
   "outcome": "neutral",
   "speedup": 1.3017519911209459,
   "promotion_commit": "d5595769062af2701dafcd15e4a410b3684e1760",
   "patch": "muse_decode_layer/hypotheses/done/neutral/dense-chain-s3-t1-q-sk2-o-route-unpool.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-25T054152Z-dense-chain-s3-t1-q-sk2-o-route-unpool.md",
   "change": "Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2 dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on an old base). Built on full-decode-sk-ticket-restack-b1-b64, whose two-level stream-K kernel replaces FA3+combine+pad/unpad+FusedRMSNormGate"
  },
  {
   "slug": "dl3-piece-a-gate-stride-check-fix",
   "area": "full decode_1 attention side (dl3 piece A enable)",
   "outcome": "neutral",
   "speedup": 1.3079569800910293,
   "promotion_commit": "6773ec5cf56207a62cd84d77607d9c11cabca6f9",
   "patch": "muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-25T145938Z-dl3-piece-a-gate-stride-check-fix.md",
   "change": "Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + kernel O dl2_o, replacing nvjet qkv + FusedQKNorm + FP8 casts + store_cache + nvjet o at full decode_1) never ran in the landed job 34a9be06"
  },
  {
   "slug": "combo-measured-rows-slide-batched-preattn-t8-tail",
   "area": "combination: sliding attention B=2..32 + pre-attention feeds (T>1) + routing R / unpooler (2<=T<=32 / 2<=T<=8)",
   "outcome": "neutral",
   "speedup": 1.3142361220457452,
   "promotion_commit": "ec89382beb4990b340cef17f37a963295ee61c0f",
   "patch": "muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-slide-batched-preattn-t8-tail.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-25T161651Z-combo-measured-rows-slide-batched-preattn-t8-tail.md",
   "change": "COMBINATION (naive top-row stack): the decode-measured per-shape winners that never reached master, each routed only to the shape class where it won, on master 6773ec5cf5 (dense-chain-s3 + dl3 piece A, 1.3080 job 933d3f825a044d36). No new load_inline module (all three sources are Triton or already-b"
  },
  {
   "slug": "Engine A (CUDA W4A16 expert engine, v1 / v2) replaces Marlin for the T<=32 expert block (decode-first campaign)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.3746,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:45",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1; evidence: ec719761 (decode scored 1.3746); later rounds c8b77a59, 249f6ef3, 2fc609f3, 2196b906, 3cd54072"
  },
  {
   "slug": "Engine A v4 (8 consumer warps, K1 plain without PDL) at T=2..32",
   "area": null,
   "outcome": "landed",
   "speedup": 1.4956,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:46",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_16, load=decode_32, load=verify_1; evidence: check 5573e05423304324 1.4950, scored dc51d8e8f58e4cb7 1.4956 (LOG:330, 356-357); land 49142841"
  },
  {
   "slug": "wgmma-hot / Engine A-cold hybrid (H=-16) at T=128/256, Engine D inert (27ef83df bytes)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5218,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:54",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_16, load=verify_32; evidence: check a1bc1756256b4c91 1.5097, f59cb301 1.5347; scored 2bd373c9e23c4667 1.5218 (LOG:376, 433, 686, 706); land 4adf7b75"
  },
  {
   "slug": "v4c: Engine A v4 with a cost-weighted stream-K split, raced against v4 in every gate (big T +2%)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5275,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:55",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_16, load=verify_32, load=decode_16, load=decode_32; evidence: eacf0722; check 0b085ef420c64ffb 1.5484, scored 42b3e39478474bcd 1.5275 (LOG:708, 720, 740); land 76815f82"
  },
  {
   "slug": "b12p: bulk12 P (pooler + router) at T=1 under a read-flush gate, with the _ead_anatomy use-after-free fix",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5441,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:62",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1; evidence: check dc3d5614d5e047c8 (P 19.62 -> 15.34 us, layer x1.040), 7844b6c903794ee7 1.5276; scored ba57a64f66864219 1.5441 (LOG:763-767, 800-818, 879); land 07b71687"
  },
  {
   "slug": "W3-A SK_FULLLINE (full-line K/V gather) at decode 16 / 32 full + W4 c320 at decode_1 full (TRIG lost its head to head)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5507,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:63",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: layer=full load=decode_16, layer=full load=decode_32, layer=full load=decode_1; evidence: cc7d142af3fb4308 (d32f x1.077), d08b11fe295f4acf 1.5484 (d1f x1.038, d32f x1.095), scored 7ceab934eea24099 1.5507 (LOG:924-926, 957-959, 976); land d5eeef58"
  },
  {
   "slug": "W6-B T>=128 diet cells (norm, comb, pool, hilo, unpool)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5697,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:64",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_16, load=verify_32; evidence: check 27a008fa 1.5554, scored 4a7a49d0ae82440f 1.5697: v16s x1.063, v16f x1.051, v32s x1.063, v32f x1.031 (LOG:1014-1025); land dca993b1"
  },
  {
   "slug": "W6-G stack: PERHEAD sliding decode_1 port + c320t at bar 1.003 + fast silu in Engine A / wg + a T=128 confirm race that can promote H-16+comb with the cells on",
   "area": null,
   "outcome": "landed",
   "speedup": 1.592,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:68",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: layer=sliding load=decode_1, layer=full load=decode_1, load=verify_16; evidence: fff7d3e7250042f1 (PERHEAD d1s x1.028), f2aece517e8543e1 1.5783, check 41975d1ce6334756 1.5749, scored c1daf58bf192471e 1.5920 (LOG:1041-1049, 1200, 1215-1224); land 3e176f6c"
  },
  {
   "slug": "pinned trim: H-16 (+comb) pinned at T=128 and T=256 (each hybrid race decides ~20 us per row; prepare-time margins do not track judged rows)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5926,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:69",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_16, load=verify_32; evidence: ced20a71e99b4798, 8e3b9874d56e4657 1.5722, 5a33db7194c24282 1.5847 (LOG:1349-1353, 1399-1413); scored e1fd3c00c83045dc 1.5926 (LOG:1490-1496); land d6c1a9b1"
  },
  {
   "slug": "pins round 2 (e8f8134b): Engine A v4 at T=2..8, unpooler master at T=9..16, qkv_gate / o swap-AB at T=17..32, pool cell at T=256 (determinism, not score)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5926,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:71",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=verify_1, load=decode_16, load=decode_32, load=verify_32; evidence: 314cca3c51604c23 1.5883, all four pins routed, rows x0.964-1.018 (LOG:1506-1508, 1517-1521); carried to master through eb-s3d -> eb-l3d-s3d (LOG:1597, 622b97bb)"
  },
  {
   "slug": "L3 paired scale decode in Engine A's dq2 consumers (bit-identical, -2 ALU instr/step) + Engine A v6 (S3: whole-record ownership, co-resident K2) at T=1",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6262,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:74",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1, load=decode_16, load=decode_32, load=verify_1; evidence: 31e0aad5 1.637 (L3 + v6), 7f9712c4 1.624 (L3 alone): decode 32 sliding x1.07, verify 16 sliding x1.05 (LOG:1767-1769); scored adb9c3436aad4e1d 1.6262 (LOG:1777-1780); land 622b97bb"
  },
  {
   "slug": "L4: L3's pair add as an IMAD on the FMA pipe (run-time 16 in inline PTX), bit-identical",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6407,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:78",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_16, load=decode_32, load=verify_1; evidence: check 0576191d3d544bf9 1.638 (flat over L3 but 9 of 12 rows faster), scored 3a68058a0f0a4eab 1.6407 (LOG:1800-1812); land f3828162"
  },
  {
   "slug": "fc: Engine A v6 T=1 fixed-cost arms (v6pe = consumer-staged B + early combine), raced vs v6, bitwise v6",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6428,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:80",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1; evidence: check 5f1047f58069453b 1.647 (v6pe 1.074 / 1.072 over v6), scored 9afcccc328974350 1.6428 (LOG:1883-1906); land f8cf2476"
  },
  {
   "slug": "X1 restricted to the v6 family (T=1, on fc); v1-v4 keep L4's code",
   "area": null,
   "outcome": "landed",
   "speedup": 1.647,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:81",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_1; evidence: check cfdf6df3 1.646 (decode 1 x1.010 / x1.013), scored 7352add6920f40e7 1.6470 (LOG:1947-1953); land d0cdde04"
  },
  {
   "slug": "Engine A v4c pinned at T=9..16 (the race flipped v4 / v4c 6:4; the layer prefers v4c though v4's block is 1.8-2.5% faster in isolation)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6551,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:83",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_16; evidence: check 86edd970 1.6599 ((16,1) rows 162.8 / 250.0 vs 165.1 / 261.8 us with v4 pinned; dec-pin-all 1.6470, dec-pin16-v4 1.6509), scored 83f0c00eb3644e1c 1.6551 (LOG:2088-2090, 2104-2105); land ec3c4dcb"
  },
  {
   "slug": "Engine A v4x decode port: warp-scan pass list, gs in smem, 32-bit split, arm d/s ticket rounds; routed 8 v4xs, 16 v4xd, 32 v4cxs, T=128 cold half v4cxs",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6606,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:93",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_16, load=decode_32, load=verify_1, load=verify_16, load=verify_32; evidence: job 31e9a69aac344675 (okc status 31e9a69aac344675), check 1fd2c931cf874a07, master d1fb10ab8a"
  },
  {
   "slug": "v4xd-stack: v6pe2 two-wave ea6 K2 at T=1 + v4x-routes (each twin qualifies alone, pooled hysteresis; T=128 cold half alone) + priority-0 side-stream fork guard",
   "area": null,
   "outcome": "landed",
   "speedup": 1.6904,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_decode_layer/hypotheses/CLAIMS.md:95",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=decode_* load=verify_*; evidence: job 7cc034f917274439 (okc status 7cc034f917274439), check 264355099fc74226, master d28b23d729"
  }
 ]
}
7. muse_decode_layer/submission.py (master) — the kernel the hypothesis will be built on (its base).
8. muse_decode_layer/baseline.py — SGLang's own pipeline (the named parts), muse_decode_layer/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
9. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): arlin-bsm8-t17-t32-w13-atomic-t1, arlin-hot-cold-two-class-bsm-t128-t256, ense-qkv-o-unpool-swapab-bf16-master-port-t1-t32, f16-m32-swapab-wgmma-port, fn-front-t128-t256-concat-route, g2-occ2-b256-decode-port, l3-unpool-nf1-t2-t8, liding-cluster-c16-decode1-port, liding-cluster-c4-s3-satfinite-on-sk-ticket, oe-bs1-fused-persistent-port, oe-chain-cuda-swapab-expert-once-fix, oe-rega-bucket-gate-port-t128-t512, oe-small-t-cuda-swapab-port, ombo-attn-master-sliding-race-skw-c16, othalf-a3-exactn-ring-wh, reattn-fused-t-gt1-sk-and-fa3-feeds, si-moe-midt-decode-port, si-moe-tiny-t-decode-port, wapab16-t2-t16-port-on-master

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
