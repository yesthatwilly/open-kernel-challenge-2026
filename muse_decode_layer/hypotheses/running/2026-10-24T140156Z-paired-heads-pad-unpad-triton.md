---
problem: muse_decode_layer
slug: paired-heads-pad-unpad-triton
created: 2026-10-24T140156Z
base: master
area: output norm-gate
change: >
  Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers
  there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master's paired path in
  muse_decode_layer kernel_fn (full layers, batch > 1). Master's pairing adds five glue launches
  around FA3, visible in its profile (fb07a3bd3aa54baf) as elementwise_kernel<128,4> x3 (6.4 us)
  plus x2 (4.7 us) at full decode_32 and x3 6.4-6.8 us at full decode_16/verify_16:
  torch.zeros(q_attn) and the two strided qp[...] = qv[...] copies (which also cast to FP8), then
  torch.empty(attn) and the two a[...] = o[...] un-pad copies, followed by SGLang's
  FusedRMSNormGate. (1) Un-pad in the epilogue: copy _paired_norm_gate_kernel verbatim from
  muse_attention_decode/submission.py on master (grid (tokens, Hq), one program per head: loads
  its 64 bf16 values from FA3's [tokens, Hq, 128] output at half (h // G) % 2, fp32 scaleless
  RMSNorm with eps 1e-5, times sigmoid of the gate logits read from qkv row t at (Hq + 2*Hkv)*64 +
  h*64 with qkv.stride(0), bf16 store into a fresh [tokens, Hq*64] output) and use it in place of
  the attn/a/o un-pad block and state['o_norm_gate'] when paired; that output goes to the o GEMM.
  (2) Pad in one launch: since decode_layer has no fused pre-attention kernel on master
  (fused-preattention-triton was neutral, not merged), add a small Triton kernel
  _pair_pad_q_kernel, grid (tokens, Hq), that reads head h's 64 bf16 q values from FusedQKNorm's
  output (pass its strides), casts to FP8 E4M3 and writes them at column half*64 of the [tokens,
  Hq, 128] FP8 buffer (allocated torch.empty) and 64 FP8 zeros in the other half, replacing
  torch.zeros plus the two slice copies. FA3's call (paired k/v views, scheduler_metadata None),
  the unpaired path (batch 1 and sliding layers), the cache store and everything after the o GEMM
  are unchanged.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "5 glue launches + FusedRMSNormGate (~11-14 us) become 2 launches (~3-4 us): ~8-10 us off 407 us, +2%"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "elementwise<128,4> 6.4 + 4.7 us and the norm-gate become 2 launches: ~8-10 us off 544 us, +1.5-2%"
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: "same glue at T=128 (6.8 us x3 row): ~8-12 us off 587 us, +1.5-2%"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "same glue at T=256, larger copies: ~10-14 us off 907 us, ~+1-1.5%, near drift"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: batch 1 unpaired, untouched
  - shapes: "*layer=full load=verify_1 *"
    direction: flat
    why: batch 1 unpaired, untouched
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: low
depends_on: fa3-hdim128-head-pairing-batched
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164538Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T164752Z
base_sha: c6b167670ed7835c8c0a38fc7064ea60ab8726dc
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/paired-heads-pad-unpad-triton
worktree: .worktrees/muse_decode_layer/paired-heads-pad-unpad-triton
mode: null
---

# paired-heads-pad-unpad-triton

## Change

Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master's paired path in muse_decode_layer kernel_fn (full layers, batch > 1). Master's pairing adds five glue launches around FA3, visible in its profile (fb07a3bd3aa54baf) as elementwise_kernel<128,4> x3 (6.4 us) plus x2 (4.7 us) at full decode_32 and x3 6.4-6.8 us at full decode_16/verify_16: torch.zeros(q_attn) and the two strided qp[...] = qv[...] copies (which also cast to FP8), then torch.empty(attn) and the two a[...] = o[...] un-pad copies, followed by SGLang's FusedRMSNormGate. (1) Un-pad in the epilogue: copy _paired_norm_gate_kernel verbatim from muse_attention_decode/submission.py on master (grid (tokens, Hq), one program per head: loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at half (h // G) % 2, fp32 scaleless RMSNorm with eps 1e-5, times sigmoid of the gate logits read from qkv row t at (Hq + 2*Hkv)*64 + h*64 with qkv.stride(0), bf16 store into a fresh [tokens, Hq*64] output) and use it in place of the attn/a/o un-pad block and state['o_norm_gate'] when paired; that output goes to the o GEMM. (2) Pad in one launch: since decode_layer has no fused pre-attention kernel on master (fused-preattention-triton was neutral, not merged), add a small Triton kernel _pair_pad_q_kernel, grid (tokens, Hq), that reads head h's 64 bf16 q values from FusedQKNorm's output (pass its strides), casts to FP8 E4M3 and writes them at column half*64 of the [tokens, Hq, 128] FP8 buffer (allocated torch.empty) and 64 FP8 zeros in the other half, replacing torch.zeros plus the two slice copies. FA3's call (paired k/v views, scheduler_metadata None), the unpaired path (batch 1 and sliding layers), the cache store and everything after the o GEMM are unchanged.

## Expected

- `*layer=full load=decode_16*` faster: 5 glue launches + FusedRMSNormGate (~11-14 us) become 2 launches (~3-4 us): ~8-10 us off 407 us, +2%
- `*layer=full load=decode_32*` faster: elementwise<128,4> 6.4 + 4.7 us and the norm-gate become 2 launches: ~8-10 us off 544 us, +1.5-2%
- `*layer=full load=verify_16*` faster: same glue at T=128 (6.8 us x3 row): ~8-12 us off 587 us, +1.5-2%
- `*layer=full load=verify_32*` faster: same glue at T=256, larger copies: ~10-14 us off 907 us, ~+1-1.5%, near drift
- `*layer=full load=decode_1 *` flat: batch 1 unpaired, untouched
- `*layer=full load=verify_1 *` flat: batch 1 unpaired, untouched
- `*layer=sliding*` flat: untouched; control

## Risk

Smallest of the four (~+0.3-0.5% geomean, on the 16-weight full shapes only), picked over a re-run of the MoE map+align port because that idea already ran here (moe-align-single-kernel-small-t, every shape within +0.2..+1.9%, neutral) and FINDINGS give no reason to expect more on master. Risk is low: the epilogue kernel is verbatim and correct in the sibling; the pad kernel is new but trivial (cast+scatter). Check that FusedQKNorm's q layout ([tokens, Hq*64], possibly a strided view of qkv) is read through its real strides. Gains are per-shape 1.5-2%, so read the elementwise_kernel<128,4> rows disappearing in top_kernels, not the headline.

## Adversarial review

Reviewed 2026-10-24T164538Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, low-risk port of the sibling's proven paired-path launch cleanup onto glue that the current master profile (9dcaec316b3045a7) shows is still there on every full batch>1 shape. The expected effect is about +1-2.5% on those four shapes but only about +0.4% on the headline geomean, inside the CI, so it has to be judged per shape and per kernel row, not on the headline. No companion is needed.

### Justification
Mechanism checks out. In master kernel_fn (submission.py lines 353-399) the paired path runs torch.zeros(q_attn), two strided cast-copies, torch.empty plus two un-pad copies, and then FusedRMSNormGate. The current master profile shows the copies as elementwise_kernel<128,4> x3 plus x2: 6.2+4.4 us at full decode_16, 6.7+4.8 at decode_32, 7.0+4.8 at verify_16, and 7.0 x3 plus a second row below the top-12 at verify_32. The zero-fill and the norm-gate fall in the '... more' tail, about 2-3 us each. Replacing them with two one-program-per-(token, head) Triton kernels that each move 128-256 B per program should cost about 2-4 us total, so about 8-12 us saved per full batch>1 shape. Against current master totals (364.9/535.8/603.9/932.9 us, not the stale 407/544/587/907 the hypothesis quotes from fb07a3bd) that is about +2.5%/+1.7%/+1.5%/+1.1%, consistent with the stated directions. decode_1/verify_1 (unpaired, batch 1) and every sliding shape run identical code, so flat there is a genuine control. Weakest link: magnitude against the scoring. Full shapes carry weight 16 against 46 for sliding in each load, so the load-level gains are about 0.9% (decode_16), 0.7% (decode_32), 0.5% (verify_16) and 0.4% (verify_32). That makes about +0.4% geomean over the six loads, well inside the current CI half-width (about 1.5%). The scored result will almost certainly classify neutral or within-ci, and its evidence is the elementwise_kernel<128,4> rows vanishing plus the two new kernels' own us in top_kernels. The second weakest link is that the sibling's +3.8-6.4% came from an attention-only problem with much smaller per-shape totals, where the pad was also fused into a pre-attention kernel. Its percentages do not transfer; only the absolute us do. Correctness: the epilogue is verbatim from a kernel that passed muse_attention_decode's tests, and its gate offset (Hq+2*Hkv)*64 + h*64 with qkv.stride(0) matches master's gate = qkv[:, hq*d+2*hkv*d:]. Its fp32 scaleless RMSNorm (eps 1e-5) times sigmoid(gate) matches FusedRMSNormGate(MuseSpark1xRMSNorm(64, has_weight=False)) up to bf16 rounding order, far inside the 3-15% bound. The pad kernel's bf16 to E4M3 cast is round-to-nearest in both torch and Triton; Triton saturates and torch does not, but post-QK-norm q magnitudes (at most about 8 times the norm weight) are far below 448, so the two cast paths should match bit for bit. The one real hazard is FusedQKNorm's q layout. The kernel must index q through its real strides (q.stride(0), and the head stride if q is 3-D), not assume [T, 1024] contiguous. Master's q.view(tokens, s, 2, g, d) proves only that the last two dims are dense.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched (depends_on): this change exists only on the paired path that hypothesis added, so it cannot be measured without it, and an ab-revert would delete the code under test. The interaction is by construction, not a confounder. A gain is attributable to removing that hypothesis's glue, read directly from the elementwise rows.
- post-expert-norm-triton-small-t: different area (MoE combine) and gated at T<32, so it does not touch the attention glue. It changes only the decode_16 totals that the percentages are computed against. No confounding.
- marlin-block-m-cap-16 and sglang-pipeline-parity-router-hilo-m8: different areas (Marlin MoE GEMMs and router GEMM) with no shared launches or buffers with the q pad or o un-pad. No confounding beyond run-to-run drift on the totals.

### Interference and A/B needs
No companion is warranted. An ab-revert of the pairing is meaningless because it removes the code path being changed. An ab-tune of the pairing gate (for example batch>1 versus some other gate) does not affect what this hypothesis measures on the shapes where pairing is on. A control splitting sub-change (1), the epilogue, from (2), the pad kernel, is unnecessary, because the profile names both new kernels (_paired_norm_gate_kernel, _pair_pad_q_kernel) and the removed elementwise rows per shape. Either a gain or a loss can therefore be attributed to the sub-change responsible from the single run's top_kernels. The ambiguity this result does carry is statistical: the headline effect is below the CI, and a second run would not fix that. Read the per-shape us on the four full batch>1 shapes against the flat controls (sliding, full decode_1/verify_1) in the same run.

### Open questions
- What is FusedQKNorm's q return: a fresh contiguous [T, 16, 64] tensor or a strided view of qkv? The pad kernel must take both the token and head strides. Implement it as reading q[t, h, j] through q.stride(0) and q.stride(-2) (or q.view(tokens, hq, d).stride()), and assert the last-dim stride is 1.
- FA3's paired output is [tokens, hq, 128] bf16. Pass out.stride(0) and out.stride(1), as the sibling does, rather than assuming contiguity; on varlen verify the output can be [T, H, D] from a [B, L, ...] view.
- Given about +0.4% expected geomean and a scored submit, does the operator accept that the classification will very likely be neutral/within-ci, so that promotion has to be by hand on per-shape evidence, as was done for paired-heads-launch-cleanup-ab-tune-batch-gate in DECISIONS?
- Follow-up rather than blocker: the unpaired path's q.view(...).to(FP8) and the k/v FP8 casts are still separate launches on every shape. Folding the k/v casts into _pair_pad_q_kernel would stack on cache-store-fused-fp8-cast's open follow-up, but keep this run's scope as written for attribution.

