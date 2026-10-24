---
problem: muse_attention_decode
slug: fa3-drop-cu-seqlens-decode
created: 2026-10-23T223722Z
base: master
area: FA3 call
change: >
  On decode steps (config q_len == 1), call flash_attn_with_kvcache with q as [B, 1, Hq, 64] and
  cu_seqlens_q=None (max_seqlen_q unset) instead of the varlen form q [B*L, Hq, 64] + cu_seqlens_q
  + max_seqlen_q that the parity pipeline (and SGLang) uses; since L = 1 the two describe the same
  batch, but the varlen form makes FA3 take its varlen-q code path (cu_seqlens reads, per-request
  offset arithmetic, varlen tile scheduler). Everything else in the call (page_table,
  cache_seqlens, causal, window_size, softmax_scale, scheduler_metadata, descale) stays. Verify
  blocks (q_len 8) are untouched. The output is reshaped to [B*1, Hq*64] for rms_norm_gate as
  before.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "21 us launch-bound shape: the varlen bookkeeping is a fixed cost per call; small, inside the CI if nothing else"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: same, 16 requests
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: same
  - shapes: "*layer=full load=decode_*"
    direction: flat
    why: FA3-bound at batch 16/32; decode_1 may gain the same fixed cost. Also a check that the judge-built scheduler_metadata (made with cu_seqlens_q) is still accepted on the non-varlen call
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched
risk: low
depends_on: fused-preattention-triton
priority: 5
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T132442Z
review_companions: null
hold_reason: "adversarial review: Moot and unresolvable under master 1.293x. Its headline target (sliding decode_1) no longer calls FA3 at all since sliding-fused-triton-decode1-gate landed. On "
---

# fa3-drop-cu-seqlens-decode

## Change

On decode steps (config q_len == 1), call flash_attn_with_kvcache with q as [B, 1, Hq, 64] and cu_seqlens_q=None (max_seqlen_q unset) instead of the varlen form q [B*L, Hq, 64] + cu_seqlens_q + max_seqlen_q that the parity pipeline (and SGLang) uses; since L = 1 the two describe the same batch, but the varlen form makes FA3 take its varlen-q code path (cu_seqlens reads, per-request offset arithmetic, varlen tile scheduler). Everything else in the call (page_table, cache_seqlens, causal, window_size, softmax_scale, scheduler_metadata, descale) stays. Verify blocks (q_len 8) are untouched. The output is reshaped to [B*1, Hq*64] for rms_norm_gate as before.

## Expected

- `*layer=sliding load=decode_1*` faster: 21 us launch-bound shape: the varlen bookkeeping is a fixed cost per call; small, inside the CI if nothing else
- `*layer=sliding load=decode_16*` faster: same, 16 requests
- `*layer=sliding load=decode_32*` faster: same
- `*layer=full load=decode_*` flat: FA3-bound at batch 16/32; decode_1 may gain the same fixed cost. Also a check that the judge-built scheduler_metadata (made with cu_seqlens_q) is still accepted on the non-varlen call
- `*load=verify_*` flat: untouched

## Risk

Low risk, low expected magnitude: a one-line change meant to establish whether the varlen-q form costs anything at L=1. The scheduler_metadata was built by the judge with cu_seqlens_q set; if FA3 rejects or mis-schedules the non-varlen call with that metadata on full decode layers, the tests fail and the hypothesis is 'failed', which is itself the answer (then try with scheduler_metadata=None on the sliding layers only, which never had metadata). Likely outcome: inside the CI on every shape.

## Adversarial review

Reviewed 2026-10-24T132442Z by an adversarial subagent (model opus).

**Verdict: hold.** Moot and unresolvable under master 1.293x. Its headline target (sliding decode_1) no longer calls FA3 at all since sliding-fused-triton-decode1-gate landed. On the remaining shapes FA3 still compiles its Varlen template because cache_seqlens (seqused_k) is passed, so dropping cu_seqlens_q saves at most some per-CTA offset arithmetic. That is far below the ~1.5-2.5% per-shape run-to-run drift, and attention has only 2 checks left.

### Justification
A: The claimed mechanism is that varlen-q makes FA3 take a 'varlen-q code path (cu_seqlens reads, per-request offset arithmetic, varlen tile scheduler)'. That is mostly wrong for this call. In FA3 (hopper flash_api), is_varlen is true when any of cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k or leftpad_k is set. kernel_fn always passes cache_seqlens, which is seqused_k, so the Varlen kernel instantiation, the varlen tile scheduler and prepare_varlen_num_blocks (when no scheduler_metadata is given) all stay in place. The only removed work is one int32 load of cu_seqlens_q[b], [b+1] per CTA in place of b*seqlen_q. That is nanoseconds inside a 10-13 us FA3 kernel, and the loads are L2-resident. The profile in FINDINGS shows no varlen-q-specific kernel to remove. The prepare_varlen_num_blocks rows (1.7-3.5 us) come from seqused_k plus metadata=None on the paired and sliding paths, and this change does not touch them. Shape coverage: (1) sliding decode_1, the 'launch-bound 21 us' shape the hypothesis leads with, now routes to _sliding_forward (13.1 us, no FA3), so its expected 'faster' is moot. (2) Sliding decode_16/32 (20.9/25.0 us, FA3 9.9/13.3 us) are the only real targets, and any plausible gain is <<1% per shape. (3) Full decode_1 (unpaired, judge metadata) and full decode_16/32 (paired, metadata=None) are FA3-bound, and the expected 'flat' is right. The verify 'flat' is right too. The hypothesis itself says 'Likely outcome: inside the CI on every shape'. Weakest link: the premise that cu_seqlens_q=None leaves the varlen path. With seqused_k set, it does not.

### Confounders on the current kernel
- sliding-fused-triton-decode1-gate: moves sliding batch-1 decode off FA3 entirely, which removes the hypothesis's lead shape. Any 'sliding decode_1' movement in the run would be drift, not this change.
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup(-ab-tune-batch-gate): full decode at batch>1 now calls FA3 on a paired [tokens, hq, 128] q with scheduler_metadata=None. The change would also reshape that q to [B,1,hq,128], a different call geometry from the one the hypothesis reasoned about. Full decode_1 stays unpaired with the judge's metadata, which was built with cu_seqlens_q, so the metadata-compatibility check applies only to that one shape.
- fused-preattention-triton: writes q as [tokens, hq, d or 2d] contiguous, so a [B,1,...] view is free. No numerical interaction.

### Interference and A/B needs
C: No companion helps. The expected effect is below what one run, or a pair of runs, can resolve: LESSONS says identical code drifts 1.5-2.5% per shape, and the predicted change is <<1% on two sliding shapes weighted into a geomean of 12. An ab-revert of sliding-fused-triton-decode1-gate would bring FA3 back on sliding decode_1, but that undoes a +25.6% win to measure a sub-1% knob and cannot change any decision. The only readable signal would be the FA3 kernel's own time in top_kernels on sliding decode_16/32, and FA3 rows are not shown to reproduce within the needed precision. With 2 attention checks left, the cost is not justified.

### Open questions
- If anyone wants the answer anyway, it can be settled for free without a check: read sgl_kernel/FA3 flash_api.cpp on a GPU-less clone to confirm is_varlen = ... || seqused_k.has_value(). If so, the hypothesis is a no-op at the template level and can be closed as moot.
- Is fa3-scheduler-metadata-side-stream (running) going to change which path passes metadata on sliding decode_16/32? If it lands, any revisit of this idea should be re-based on it.

