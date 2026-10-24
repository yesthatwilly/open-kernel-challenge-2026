---
problem: muse_attention_decode
slug: fa3-batched-q-verify
created: 2026-10-23T225858Z
base: master
area: FA3 call (q layout)
change: >
  On verify blocks only (config q_len == 8), call flash_attn_with_kvcache with q as the batched
  4-D tensor q.view(B, 8, Hq, 64) and cu_seqlens_q=None, max_seqlen_q=None, instead of the varlen
  form q.view(B*8, Hq, 64) + cu_seqlens_q + max_seqlen_q=8 that the parity pipeline (and SGLang)
  uses. Every request has exactly 8 queries, so the two forms describe the same batch; the varlen
  form, though, makes FA3 take its varlen-q path (cu_seqlens reads per tile, per-request offset
  arithmetic, non-TMA Q loads, the varlen tile scheduler) while the batched form lets FA3 load the
  [8 x Hq] query block per request through its TMA path and use the static batched scheduler.
  page_table, cache_seqlens, softmax_scale, causal=True, window_size, scheduler_metadata (already
  None on verify blocks) and descale stay as they are. The output is [B, 8, Hq, 64]; reshape it to
  (B*8, Hq*64) for rms_norm_gate as before. Decode steps (q_len 1) are untouched; this is the
  verify-block sibling of the queued fa3-drop-cu-seqlens-decode and never shares a shape with it.
expected:
  - shapes: "*layer=sliding load=verify_1 page*"
    direction: faster
    why: "23 us launch-bound shape: varlen bookkeeping is a per-call fixed cost; small, possibly inside the CI"
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: same, 16 requests x 8 queries
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: same
  - shapes: "*layer=full load=verify_16*"
    direction: flat
    why: 205 us FA3-bound over 16 x 80K tokens; the Q-load path is not the bottleneck, so a change here would attribute to the scheduler difference, not the Q load
  - shapes: "*layer=full load=verify_32*"
    direction: flat
    why: same, 390 us
  - shapes: "*layer=full load=verify_1 page*"
    direction: faster
    why: "35.6 us shape with 2 tiles: the same fixed cost as the sliding verify_1"
  - shapes: "*load=decode_*"
    direction: flat
    why: untouched; control
risk: low
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T120202Z
review_companions: null
hold_reason: "adversarial review: Master has not made this moot, since verify still passes cu_seqlens_q. The mechanism doesn't hold up, though: cache_seqlens (seqused_k) already puts FA3 on its "
---

# fa3-batched-q-verify

## Change

On verify blocks only (config q_len == 8), call flash_attn_with_kvcache with q as the batched 4-D tensor q.view(B, 8, Hq, 64) and cu_seqlens_q=None, max_seqlen_q=None, instead of the varlen form q.view(B*8, Hq, 64) + cu_seqlens_q + max_seqlen_q=8 that the parity pipeline (and SGLang) uses. Every request has exactly 8 queries, so the two forms describe the same batch; the varlen form, though, makes FA3 take its varlen-q path (cu_seqlens reads per tile, per-request offset arithmetic, non-TMA Q loads, the varlen tile scheduler) while the batched form lets FA3 load the [8 x Hq] query block per request through its TMA path and use the static batched scheduler. page_table, cache_seqlens, softmax_scale, causal=True, window_size, scheduler_metadata (already None on verify blocks) and descale stay as they are. The output is [B, 8, Hq, 64]; reshape it to (B*8, Hq*64) for rms_norm_gate as before. Decode steps (q_len 1) are untouched; this is the verify-block sibling of the queued fa3-drop-cu-seqlens-decode and never shares a shape with it.

## Expected

- `*layer=sliding load=verify_1 page*` faster: 23 us launch-bound shape: varlen bookkeeping is a per-call fixed cost; small, possibly inside the CI
- `*layer=sliding load=verify_16*` faster: same, 16 requests x 8 queries
- `*layer=sliding load=verify_32*` faster: same
- `*layer=full load=verify_16*` flat: 205 us FA3-bound over 16 x 80K tokens; the Q-load path is not the bottleneck, so a change here would attribute to the scheduler difference, not the Q load
- `*layer=full load=verify_32*` flat: same, 390 us
- `*layer=full load=verify_1 page*` faster: 35.6 us shape with 2 tiles: the same fixed cost as the sliding verify_1
- `*load=decode_*` flat: untouched; control

## Risk

Low risk, low expected magnitude: a one-line change that establishes whether the varlen-q form costs anything at L = 8. Correctness unaffected (same math); the batched form requires FA3 to accept paged KV with a 4-D q and cache_seqlens, which is its documented decode API. If FA3 rejects the batched q together with page_size 1 or the dp head counts, the tests fail and the hypothesis is failed, which is itself the answer. Companion to fa3-drop-cu-seqlens-decode: if both land, every shape uses the batched form.

## Adversarial review

Reviewed 2026-10-24T120202Z by an adversarial subagent (model opus).

**Verdict: hold.** Master has not made this moot, since verify still passes cu_seqlens_q. The mechanism doesn't hold up, though: cache_seqlens (seqused_k) already puts FA3 on its Varlen instantiation with the varlen tile scheduler and prepare_varlen_num_blocks, and page-size-1 paged KV forces PackGQA with cp.async Q loads. Dropping cu_seqlens_q only swaps a few runtime offset reads inside the same kernel binary, so the expected effect is well under the ~1.5% per-shape drift and one run cannot resolve it.

### Justification
Not moot: master's kernel_fn still calls flash_attn_with_kvcache with cu_seqlens_q=data['cu_seqlens_q'] and max_seqlen_q=length on every shape, including the paired full-layer path at batch > 1 (q is [tokens, Hq, 128], heads=1, metadata=None) and the unpaired path elsewhere. The mechanism is where it fails. (1) Upstream FA3 hopper picks its Varlen template when any of cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k or leftpad_k is set. cache_seqlens is seqused_k, so the batched form still runs Varlen=true: the same varlen tile scheduler and the same prepare_varlen_num_blocks_kernel. FINDINGS shows that kernel on every shape, decode q_len 1 included (1.8-3.4 us). There is no 'static batched scheduler' to switch to. (2) FA3 forces pack_gqa whenever page_table is set and paged KV is not TMA-able, and page_size 1 is the non-TMA case. Q is then loaded by the PackGQA cp.async path in both forms, so the 'TMA Q load' gain does not exist. (3) Only is_varlen_q changes at runtime: a cu_seqlens_q load and an offset per tile, against 7.7-18 us of FA3 main kernel plus 2.5-12 us combine on the batch-1 verify shapes. The expected sliding verify_1/16/32 and full verify_1 'faster' rows are therefore well under 1% of shape time. The judge's drift is 1.5-2.5% per shape (LESSONS), so a gain or a loss would both be noise. The flat expectations for full verify_16/32 and decode are fine. Weakest link: the claim that the batched 4-D q takes a different FA3 code path (TMA Q plus a non-varlen scheduler). With seqused_k and page-size-1 paging it does not. The one residual unknown is whether the sgl_kernel FA3 fork departs from upstream here; a profile showing the same kernel names (prepare_varlen_num_blocks present, same flash fwd template) would confirm it.

### Confounders on the current kernel
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup: at batch 16/32 on full layers q is the padded [tokens, Hq, 128] tensor with Hkv'=1 super-head. The batched view must be q.view(B, 8, Hq, 2*D), not (B, 8, Hq, 64) as the spec says. The output is then [B, 8, Hq, 128] and must reach _paired_norm_gate_kernel as view(tokens, Hq, 2D) with the same strides. This is a correctness trap in the spec, not a measurement confounder.
- paired-heads-launch-cleanup-ab-tune-batch-gate: the full verify_1 shape the spec expects to speed up is now the unpaired path (batch 1 is gated out), and the full verify_16/32 shapes are the paired path. The two paths have different qhead_per_khead (8 vs 16) but both force pack_gqa, so the gate does not change the conclusion.
- fused-preattention-triton: it writes q contiguous as [tokens, Hq, D or 2D], so the 4-D view is free. It does not interact with the change.

### Interference and A/B needs
No companion would help. Both the change and the landed paths run the same FA3 instantiation, and the predicted effect is below what one run can resolve. An ab-revert of the pairing or the batch gate would only change which q shape is viewed, not whether FA3 leaves its varlen path, and a control cannot isolate a sub-change of a one-argument edit. The cheap way to settle it is offline: read sgl_kernel's flash_api.cpp, or inspect the kernel names in an existing profile. If the fork matches upstream (Varlen switch includes seqused_k; pack_gqa is forced for non-TMA paged KV), the hypothesis is inert. If the fork really does dispatch a non-varlen kernel when cu_seqlens_q is None, it can be unheld and run as written, with the paired q view fixed.

### Open questions
- Does the sgl_kernel FA3 fork's run_mha_fwd still include seqused_k in its VARLEN_SWITCH, and does get_pack_gqa still force packing when page_table is set without pagedkv_tma? If both hold, this hypothesis and its sibling fa3-drop-cu-seqlens-decode share the same false premise and both should be held.
- If FA3 does take a non-varlen path without cu_seqlens_q, does it still launch prepare_varlen_num_blocks? Removing that 1.8-3.2 us launch would be the only effect worth measuring, and it would show in top_kernels.

