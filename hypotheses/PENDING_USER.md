# Waiting on the user

churnsubs never blocks on these: the loop keeps dispatching everything else and lists open
items here and in each tick's summary. Resolved items move to the log at the bottom.

## Open

All run `okc submit <p> <p>/submission.py --no-wait --json` on master, then `churn.py record-submit <p> <slug> <job>`.

- 2026-10-24 attention_decode: master is now **sliding-fused-triton-decode1-gate** (check 1.293x, CI 1.274-1.321; board 1.128x). Supersedes the earlier head-pairing ask.
  `okc submit muse_attention_decode /home/user/git/musespark/muse_attention_decode/submission.py --no-wait --json`
- 2026-10-24 moe_w4a16: master is now **triton-map-align-on-block-m-restack** (check 1.312x, CI 1.292-1.331; no board result yet).
  `okc submit muse_moe_w4a16 /home/user/git/musespark/muse_moe_w4a16/submission.py --no-wait --json`
- 2026-10-24 decode_layer: master is now **fa3-hdim128-head-pairing-batched** (check 1.087x; board 1.063x).
  `okc submit muse_decode_layer /home/user/git/musespark/muse_decode_layer/submission.py --no-wait --json`

- 2026-10-24 prefill_layer: master is now **experts-hybrid-marlin-turn-triton-first-turn** (Marlin at T<=4096, Triton NVFP4 above; check 1.060x, CI 1.044-1.076; board: none yet).
  `okc submit muse_prefill_layer /home/user/git/musespark/muse_prefill_layer/submission.py --no-wait --json`

- 2026-10-24 bf16_proj: master is now **bf16-shape-gated-triton-winners-combo** (check 1.176x, CI 1.158-1.194; board 1.143x).
  `okc submit muse_bf16_proj /home/user/git/musespark/muse_bf16_proj/submission.py --no-wait --json`

- 2026-10-24 rules heads-up (not blocking): bf16 **bf16-cublaslt-algo-pick-in-prepare-m128-m256** times cuBLASLt algorithm candidates on synthetic data inside the untimed prepare() and replays the winner in kernel_fn (no timing or autotune in kernel_fn). The Opus reviewer read the rules as allowing it, and decode's pooler-gemm-splitk-autotune used the same pattern. It runs on tomorrow's quota unless you say to hold it.

## Standing answers

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
