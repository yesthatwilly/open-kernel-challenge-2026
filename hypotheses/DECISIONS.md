# Decisions

Append-only log of churnsubs judgement calls (holds, deferrals, merges by hand,
overrides of a reviewer) with the reason, newest last. Written by `churn.py note` and `hold`.

## Backfilled 2026-10-24 (calls made before the log existed)

- 2026-10-23 `muse_attention_decode` **sglang-path-parity**: held by the reviewer. As specified it is baseline.py line for line, so commit-impl would reject it as a duplicate. Gaming the dedup with a cosmetic edit was refused.
- 2026-10-23 `muse_bf16_proj` **router-triton-hilo-fused / pooler-triton-skinny-m32**: re-filed from done/gain to regression (0.940x) and neutral (0.998x). The harvest had compared against a stale BEST.json (0.902x starter) instead of the merged parity base (1.0022x). classify() is fixed to use the higher of BEST and base (aee1194).
- 2026-10-23 `muse_moe_w4a16` **sglang-marlin-path-parity**: BEST.json set by hand to the merged parity result (0.9888x). Merge-without-submit had not updated BEST. _land is fixed.
- 2026-10-23 `muse_bf16_proj` **router-hilo-addmm-epilogue**: re-claimed on master. claim had based it on its landed parent's impl commit, which lacks router-hilo-at-m8. claim is fixed to base on master once the parent has landed (c257eca).
- 2026-10-24 `muse_bf16_proj` **router-triton-hilo-fused-control-splitk-all-m**: merged to master by hand (1.151x). Its file is byte-identical to the checked one, and its router path replaces master's router-hilo-at-m8 change wholesale. promote's self-supersede bug is fixed (e51ac52).
- 2026-10-24 `muse_bf16_proj` **router-triton-hilo-fused-control-hybrid-cublas-above-m32, router-hilo-addmm-epilogue-control-inplace-out**: held as superseded. Master's router no longer uses cuBLAS.
- 2026-10-24 `muse_bf16_proj` **o-triton-skinny-m32, pooler-skinny-tiled-weight-layout**: deferred until unpooler-full-triton-skinny-m32 harvested (same tile, larger weight). Afterwards o was run (neutral 1.006x). pooler-tiled was held: the family is settled and the reviewer expected ~zero.
- 2026-10-24 `muse_moe_w4a16` **marlin-workspace-blocks-per-sm-8**: dispatched although the reviewer expected inert. Quota was plentiful and a flat result settles the knob (it was flat).
- 2026-10-24 `muse_moe_w4a16` **w13-marlin-atomic-add-small-m**: implemented with use_fp32_reduce=True, deviating from the spec on the reviewer's advice, to keep it disjoint from w13-marlin-bf16-reduce-verify.
- 2026-10-24 `muse_prefill_layer` **experts-dequant-triton-moe-16k (+3 dependents)**: held for a rules call (per-call 3.2 GB BF16 copy of NVFP4 weights). The user ruled: keep held, in-kernel dequant only.
- 2026-10-24 `muse_decode_layer` **fa3-num-splits-sliding, fa3-num-splits-full-no-metadata**: held by the reviewer. FA3 sizes splits on the device; a static num_splits is only a cap.
- 2026-10-24 `muse_prefill_layer` **fa3-batched-mode**: held by the reviewer. FA3 already runs at ~600 TFLOP/s whatever the CTA count.
- 2026-10-24 `muse_decode_layer` **moe-align-local-expert-count**: held. The identical one-argument change was already measured in moe_w4a16 (flat at verify, -4% at decode_1).
- 2026-10-24 `(all)` **adversarial review model**: switched from Fable to Opus at the user's request after Fable rate limits (590760a).
- 2026-10-24 `muse_attention_decode` **fused-preattention-triton**: at the user's direction, no parent and the base for the 12 parity dependents. It was committed verbatim from the main tree's uncommitted file (sha256 fde179d5…) so the 1.127x result (dfee6f2939974bad) stands. The reviewer's hardening fixes (stride asserts, eager _INV_FREQ) were deferred because they would change measured code.
- 2026-10-24 `(all)` **rewrite kind**: added at the user's request (a0cd0b9). triton-flash-decode-full-paired-kv (decode) and experts-triton-w4a16-inkernel-fp4 (prefill) were converted to rewrite stage 1, and triton-flash-decode-full-verify-paired-kv to stage 2.
- 2026-10-24 `muse_bf16_proj` **router-triton-hilo-fused-control-splitk-all-m**: scored submit 240eb22ea0b6487e, run by Claude with the user's approval.
- 2026-10-24 `muse_decode_layer` **marlin-block-m-cap-16**: scored submit 2b1f82408ab64116, run by Claude at the user's request.

## Log
