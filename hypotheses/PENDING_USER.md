# Waiting on the user

churnsubs never blocks on these: the loop keeps dispatching everything else and lists open
items here and in each tick's summary. Resolved items move to the log at the bottom.

## Open

- 2026-10-25 01:36Z attention **full-decode-splitkv-s2-streamk-ticket-inproc** checked **1.4171 [1.396, 1.443]** vs attention's best 1.3949 [1.374, 1.416] (job e592b1f5d3db4f83). Per-shape read (dd73acf): full decode_16 122.8 -> 110.9 us (-9.7%), full decode_32 241.9 -> 226.8 us (-6.2%), kernel time -9-11% in the profile; every other shape within +-1%. All self-checks passed on sm_90a. Planned if you approve: land its branch on master, then `okc submit muse_attention_decode <worktree>/muse_attention_decode/submission.py --no-wait --json`. Meanwhile churn mints a stack of it with the sliding cluster (C=8 off), estimated ~1.44x, which also re-measures it. Default until you answer: not landed.

- 2026-10-25 01:04Z decode **full-decode-splitkv-restack-on-pdl** (CUDA split-KV hand-merged onto master) checked **1.268 [1.249, 1.287]** vs decode's best 1.2521 scored / 1.2495 check (job c243a29f26dc4725). Classified unproven-vs-best (CI low 1.249 < 1.2521). Supporting evidence: the independent split-KV variant full-decode-splitkv-kv-bw-3p5 (older base) also ran 1.266 [1.247, 1.285]. Planned if you approve landing + scored run: land its branch on master, then `okc submit muse_decode_layer <worktree>/muse_decode_layer/submission.py --no-wait --json`. Meanwhile churn mints a stack (restack + kv-bw kernel configs) as a second measurement. Default until you answer: not landed.
  - Per-shape read (01:04Z, REWRITE_READ d41c833): the gain is concentrated and well beyond noise: full decode_16 306.7 vs 349.5 us (-12.2%, z -8.1, paired 1.407 vs 1.229) and full decode_32 479.6 vs 529.7 us (-9.5%, z -5.0, paired 1.301 vs 1.178); the other 10 shapes are noise (|z| <= 1.8). The kv-bw configs are dead (bandwidth capped ~3.0 TB/s), so the restack itself is the candidate. Recommendation: land and run it scored.

- 2026-10-25 bf16 **bf16-restack-m16-swapab-router-swapab-m32** was NOT landed on master: its scored confirmation 60b20fd0bf924267 came back `duplicate` of 1e90292c7b954d96 (okc dedups identical code, so no second measurement and no scored run spent). Single run: 1.2215 [1.203, 1.240] vs master lineage 1.1959 [1.178, 1.214]; it is already bf16's board entry. Land it on one run (later bf16 hypotheses then build on it), or wait for a stack built on it to measure it again? Default until you answer: not landed; new bf16 work stays on master.
- 2026-10-24 rules heads-up (not blocking): bf16 **bf16-cublaslt-algo-pick-in-prepare-m128-m256** times cuBLASLt algorithm candidates on synthetic data inside the untimed prepare() and replays the winner in kernel_fn (no timing or autotune in kernel_fn). The Opus reviewer read the rules as allowing it, and decode's pooler-gemm-splitk-autotune used the same pattern. It runs on tomorrow's quota unless you say to hold it.

## Standing answers

- Scored runs for experiments: when a problem's checks are spent, churnsubs may dispatch reviewed hypotheses as scored submits (never risk: high), since the board shows only the best run. (2026-10-24, the user: "yes, do all three")

- The 40% rule: 40% of each problem's daily checks go to ambitious rewrites (whole-subsystem or
  whole-layer overhauls), the rest to incremental hypotheses. Unused shares are released so
  nothing idles. Mint big rewrites so the share is always fed. (2026-10-24, the user: "yes, add
  the 40% rule and mint the big rewrites")

- Scored submits (promotions with CI low above 1.0): ask each time, but never hold the loop for
  the answer. Leave the planned `okc submit` command here and keep going.
- Prefill expert-dequant family (per-call BF16 copy of NVFP4 expert weights): keep held. Pursue
  in-kernel dequant only. (2026-10-24)
- Attention-decode base: fused-preattention-triton with no parent; the 12 parity dependents
  re-pointed to it. (2026-10-24)
- Adversarial review model: Opus, not Fable. (2026-10-24)
- Autonomy: run the loop all day, every day. Mint hypotheses when queues run low, and mint next
  rewrite stages. Submits are still asked as above. (2026-10-24)

## Log

- 2026-10-24: bf16 router-triton-hilo-fused-control-splitk-all-m (check 1.151x) submitted by
  Claude with the user's approval, job 240eb22ea0b6487e.
- 2026-10-24: decode marlin-block-m-cap-16 (check 1.062x) submitted at the user's request, job 2b1f82408ab64116.
- 2026-10-24: attention_decode fused-preattention-triton (check 1.127x) submitted by the user, job c297440e53334f34.
- 2026-10-24: the user said "submit all five promotions"; Claude ran them: attention full-decode-cuda-gather-splitkv (check 1.347x) de486093311d4bb9, decode post-expert-norm-triton-small-t (1.164x) a017e10aec544c55, moe triton-map-align-on-block-m-restack (1.312x) 3cabc739f1af493e, prefill experts-hybrid-marlin-turn-triton-first-turn (1.060x) ba9c16b9cc6b4605, bf16 bf16-m32-swapab-wgmma-triton (1.198x) eeb9985396764ac2.
- 2026-10-24: bf16-lossless-exp4-weight-stream-rewrite heads-up closed: TASK.md says weights may be compressed losslessly ("BF16 weights stay BF16, or smaller, losslessly"). The review also had cuobjdump removed from its diagnostics, since it starts a process.
- 2026-10-24: moe align-parallel-per-expert-verify (check 1.3249x, hand-landed on per-shape rows) submitted at the user's request ("sure promote moe i guess"), job 1947e3ec2abb4bdf.
