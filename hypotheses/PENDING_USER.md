# Waiting on the user

churnsubs never blocks on these: the loop keeps dispatching everything else and lists open
items here and in each tick's summary. Resolved items move to the log at the bottom.

## Open

- **Scored submit, decode marlin-block-m-cap-16** (check 1.062x [1.046, 1.078], merged on master;
  current board 1.020x). Run: `okc submit muse_decode_layer /home/user/git/musespark/muse_decode_layer/submission.py --no-wait --json`
  then `churn.py record-submit muse_decode_layer marlin-block-m-cap-16 <job>`. (2026-10-24)

## Standing answers

- Scored submits (promotions with CI low above 1.0): ask each time, but never hold the loop for
  the answer. Leave the planned `okc submit` command here and keep going.
- Prefill expert-dequant family (per-call BF16 copy of NVFP4 expert weights): keep held. Pursue
  in-kernel dequant only. (2026-10-24)
- Attention-decode base: fused-preattention-triton with no parent; the 12 parity dependents
  re-pointed to it. (2026-10-24)
- Adversarial review model: Opus, not Fable. (2026-10-24)

## Log

- 2026-10-24: bf16 router-triton-hilo-fused-control-splitk-all-m (check 1.151x) submitted by
  Claude with the user's approval, job 240eb22ea0b6487e.
