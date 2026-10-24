# Waiting on the user

churnsubs never blocks on these: the loop keeps dispatching everything else and lists open
items here and in each tick's summary. Resolved items move to the log at the bottom.

## Open

- 2026-10-24 attention_decode **fa3-hdim128-head-pairing** gained (check 1.171x, CI 1.153-1.192; board best is 1.128x) and is merged to master. Scored submit, if you approve:
  `okc submit muse_attention_decode /home/user/git/musespark/muse_attention_decode/submission.py --no-wait --json`
  then `churn.py record-submit muse_attention_decode fa3-hdim128-head-pairing <job>`. A batch-gated follow-up (pair only at batch > 1, where it lost 27-35%) is being minted and may beat it.

- 2026-10-24 moe_w4a16 **post-expert-norm-triton-small-m** gained (check 1.130x, CI 1.113-1.147; no moe scored result yet) and is merged to master. Scored submit, if you approve:
  `okc submit muse_moe_w4a16 /home/user/git/musespark/muse_moe_w4a16/submission.py --no-wait --json`
  then `churn.py record-submit muse_moe_w4a16 post-expert-norm-triton-small-m <job>`. The Marlin block-m sweep (two-smaller 1.092x on the old base) is being restacked on it and may beat it.

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
