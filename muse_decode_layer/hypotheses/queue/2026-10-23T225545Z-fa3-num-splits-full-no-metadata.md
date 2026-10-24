---
problem: muse_decode_layer
slug: fa3-num-splits-full-no-metadata
created: 2026-10-23T225545Z
base: master
area: FA3 call
change: >
  On full-attention layers only (cfg['window'] == 0), stop passing the judge-supplied
  scheduler_metadata to flash_attn_with_kvcache (pass None) and instead pass an explicit
  num_splits chosen on the host from the shape: with 2 kv heads and 132 SMs, num_splits =
  clamp(ceil(132 * 2 / (B * 2)), 1, 128) so B=1 gets 128, B=16 gets 9, B=32 gets 5 (more CTAs than
  SMs by ~2x to cover the latency of page-size-1 gathers over an 80K-row context). q_len=8 verify
  blocks on full layers use the same formula (their metadata is already None). The sliding branch
  and everything outside the FA3 call are unchanged. The gap this targets: decode_32 full costs
  SGLang 623 us against 261 us for the sliding twin, i.e. ~360 us for 655 MB of FP8 KV that the
  4.8 TB/s floor moves in 137 us; decode_16 full is 165 us over its sliding twin for a 68 us floor
  delta. Attribution: full shapes only (weight 16 per load); sliding shapes are the control.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "largest absolute gap: ~360 us of FA3 over a 137 us floor; a split count that oversubscribes the SMs ~2x should raise achieved bandwidth"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 165 us over the sliding twin for a 68 us floor delta
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: same KV volume with 8 queries; metadata was already None here so only num_splits changes
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: B=1 is already split heavily by the metadata heuristic; 128 vs its pick likely a wash
  - shapes: "*layer=sliding*"
    direction: flat
    why: sliding branch untouched; control
risk: medium
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 3
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T041234Z
review_companions: null
hold_reason: "adversarial review: An explicit num_splits cannot raise FA3's split count on this call: every flash_attn_with_kvcache call with cache_seqlens is varlen, so FA3 runs its device-side"
---

# fa3-num-splits-full-no-metadata

## Change

On full-attention layers only (cfg['window'] == 0), stop passing the judge-supplied scheduler_metadata to flash_attn_with_kvcache (pass None) and instead pass an explicit num_splits chosen on the host from the shape: with 2 kv heads and 132 SMs, num_splits = clamp(ceil(132 * 2 / (B * 2)), 1, 128) so B=1 gets 128, B=16 gets 9, B=32 gets 5 (more CTAs than SMs by ~2x to cover the latency of page-size-1 gathers over an 80K-row context). q_len=8 verify blocks on full layers use the same formula (their metadata is already None). The sliding branch and everything outside the FA3 call are unchanged. The gap this targets: decode_32 full costs SGLang 623 us against 261 us for the sliding twin, i.e. ~360 us for 655 MB of FP8 KV that the 4.8 TB/s floor moves in 137 us; decode_16 full is 165 us over its sliding twin for a 68 us floor delta. Attribution: full shapes only (weight 16 per load); sliding shapes are the control.

## Expected

- `*layer=full load=decode_32*` faster: largest absolute gap: ~360 us of FA3 over a 137 us floor; a split count that oversubscribes the SMs ~2x should raise achieved bandwidth
- `*layer=full load=decode_16*` faster: 165 us over the sliding twin for a 68 us floor delta
- `*layer=full load=verify_32*` faster: same KV volume with 8 queries; metadata was already None here so only num_splits changes
- `*layer=full load=decode_1*` flat: B=1 is already split heavily by the metadata heuristic; 128 vs its pick likely a wash
- `*layer=sliding*` flat: sliding branch untouched; control

## Risk

Risk: SGLang's scheduler_metadata is FA3's own tile schedule for this exact step and may already be near-optimal; dropping it can make B=16/32 decode SLOWER if the heuristic FA3 then runs on the host sees seqlen_k as the page table extent and picks worse. This run will be read either way: the per-shape result tells whether the metadata or the split count is what limits FA3 at page size 1. The remaining gap (page-size-1 row gathers through the page table) is not touched by num_splits; if full shapes come back flat, the next hypothesis is a different decode kernel (see trtllm-decode-full-attention). Correctness: only the fp32 combine order changes. Graph-safe: num_splits from config only. The batch 1 ctx 1048576 full q_len=8 test still runs through this branch: 128 splits over 8192 tiles is fine, but check its 15-minute budget.

## Adversarial review

Reviewed 2026-10-24T041234Z by an adversarial Fable subagent (model fable).

**Verdict: hold.** An explicit num_splits cannot raise FA3's split count on this call: every flash_attn_with_kvcache call with cache_seqlens is varlen, so FA3 runs its device-side dynamic split (prepare_varlen_num_blocks) and uses the static num_splits only as a cap (num_splits_dynamic[b] = min(max(1, ceil(n_blocks/blocks_per_sm)), num_splits_static)); the dynamic pick already targets about one wave (roughly 2 splits at B=32, 4 at B=16, ~57 at B=1 for 625 key tiles), so the proposed caps 5/9/128 never bind and the 'oversubscribe the SMs 2x' mechanism is unreachable through this parameter. The only measurable change the run would make is moving the prepare kernel into the timed call on the full decode shapes (one extra ~2 us launch, a <=1.5% loss on decode_1 full, inside the 1.5% drift), while verify full rows are byte-identical in kernel sequence. Re-mint as a sensitivity probe that caps BELOW the dynamic pick (num_splits=1, optionally 2) on full layers instead.

### Justification
A. The gap is real: profile 5755a412dda244a1 shows FA3 at 491.6 us of 623 us on full decode_32 (655 MB of FP8 KV, 137 us at 4.8 TB/s, i.e. 1.33 TB/s achieved) and 189.1 us of 442 us on full decode_16 (328 MB, 68 us floor, 1.73 TB/s); verify_16/32 full FA3 is 192/375 us for the same bytes, and B=1 full is 18.0 us FA3 + 6.9 us FlashAttnFwdCombine against a 4.3 us byte floor (latency, not bandwidth). But the lever is wrong for the API. In FA3 hopper mha_fwd, is_varlen is true whenever cu_seqlens_q or seqused_k (cache_seqlens) is passed, use_dynamic_split = is_varlen && b <= 992 is set before num_splits is resolved ('temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it'), and an explicit num_splits only replaces the static value get_num_splits would have produced; the per-batch split count is then computed on device by prepare_varlen_num_blocks as min(max(1, ceil_div(num_n_blocks, blocks_per_sm)), num_splits_static) with blocks_per_sm ~ ceil(1.1 * total_blocks * h_k / num_sm), i.e. sized for about one wave. The judge's scheduler_metadata is exactly that prepare kernel's output built with num_splits=0 (reference.py:_fa3_scheduler_metadata), so the existence of get_scheduler_metadata in the judge's sgl_kernel confirms the dynamic-split build. Worked at 80K (625 tiles of 128, packgqa m-blocks=1 for q_len 1 and 8): B=32 -> blocks_per_sm ~334, 2 splits (128 CTAs); B=16 -> ~167, 4 splits (128 CTAs); B=1 -> ~11, ~57 splits (114 CTAs). Even if the 1.1*h_k factor is absent, the picks are 5/9/63, again at or below the proposed caps 5/9/128. In every variant the cap never binds, so the kernel's grid, partials and combine are unchanged on all full shapes. What does change: on full decode shapes the judge's precomputed schedule is replaced by an in-call prepare launch (plus an in-call torch::empty for the semaphore buffer), costing ~2 us per replay: ~1.1% on decode_1 full (183 us), ~0.3-0.5% on decode_16/32 full, all inside the ~1.5% per-shape drift. The expected table is therefore inconsistent with the mechanism: 'faster' on the three full shapes is unreachable, 'flat' on decode_1 full should be 'slower by one launch', and 'metadata was already None so only num_splits changes' on verify_32 full is true but the change is a no-op. The B=32 < B=16 bandwidth (1.33 vs 1.73 TB/s) with 128 CTAs in both cases also says the limiter is the page-size-1 gather pattern (two 64-byte head rows per token through the page table, non-TMA path since page_size % kBlockN != 0), not SM fill; the combine is a fixed ~4.1-4.4 us wherever it appears. Weakest link: the premise that num_splits can be raised above FA3's own device-side pick at all; it can only be lowered.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8 (only landed item; patch muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.patch): its FA3 call is byte-identical to baseline.py and its one structural change (ROUTER_FP32_MAX_TOKENS 8->1) runs on the side stream after the o projection; no shared launch, traffic or numerical path with the attention call, so it cannot cause or mask a gain or loss here.
- in flight, not landed: cache-store-fused-fp8-cast rewrites the launches immediately before the FA3 call and hands FA3 a preallocated fp8 q; it does not touch num_splits or scheduler_metadata, but if it lands first the T<=8 base rows shift by ~3-4 us, the same order as the ~2 us prepare-launch effect this run would show on decode_1 full, so any reading must be against the correct merged base rows, not FINDINGS' current ones.

### Interference and A/B needs
No ab-revert or ab-tune is warranted: nothing landed interacts with the FA3 call. A control is not needed for interpretability either, because the run self-disambiguates: the verify full rows carry only sub-change (b) (the explicit cap, metadata already None) and the decode full rows carry (a) drop metadata + (b); flat verify rows plus a small decode_1 full loss reads as 'cap did not bind, prepare launch cost', and moving verify rows would mean the cap bound (contradicting the dynamic formula) in the slower direction. The problem is not ambiguity but that the predicted outcome is null by construction and inside the 1.5% drift, so a run spends quota to confirm an API semantic that the source already settles. The run that would actually measure FA3's sensitivity to the split count on these shapes caps below the dynamic pick: num_splits=1 on full layers (64 CTAs at B=32, 32 at B=16, 2 at B=1; no partials, no combine, no prepare contribution to the grid), optionally num_splits=2 as a second point. If decode_32 full stays near 492 us with half the CTAs, the gather pattern is the limiter and a replacement kernel must change the access pattern (fetch both heads' 128-byte row per token, deeper per-CTA pipelining), not add parallelism; if it roughly doubles, in-flight parallelism matters and a custom kernel with more MLP per SM is the right follow-up. That is a different hypothesis from this one (opposite direction, different expected table) and should be minted, not run as a companion of a primary whose own effect is nil.

### Open questions
- Settle the API semantics offline before any FA3 num_splits run on this problem: in the judge image's sgl_kernel FA3 (hopper flash_api.cpp), confirm (1) is_varlen includes seqused_k so every kvcache decode call uses dynamic split, (2) prepare_varlen_num_blocks computes num_splits_dynamic = min(max(1, ceil_div(num_n_blocks, blocks_per_sm)), num_splits_static) with blocks_per_sm ~ ceil(1.1 * total_blocks * h_k / num_sm), (3) an explicit num_splits only sets num_splits_static. A print of sgl_kernel.__version__ from prepare() in any queued run returns in the log tail at no quota cost.
- Suggested re-mint (one run, full layers only, decode drops metadata and verify keeps None): num_splits=1 on cfg['window']==0; expected: decode_32/16 full and verify_32/16 full move by a large, readable amount in one direction or the other (either ~flat, meaning gather-pattern-bound, or ~2x slower, meaning MLP-bound), decode_1/verify_1 full slower (2 CTAs over 625 tiles, ~10x FA3 time but it is 18 us), sliding flat as control. Both outcomes directly steer trtllm-decode-full-attention and any Triton decode kernel (how many CTAs and how deep a pipeline they need, and whether head-pairing the 128-byte row is the real lever).
- Is FlashAttnFwdCombine present on full decode_16 (it is below the profile's 4.0 us cutoff there while visible at 4.1-4.2 us on decode_32 and verify_16)? If the combine is absent at B=16 the dynamic pick there is 1, which would make the no-op argument even stronger.
- If a hypothesis still wants more CTAs than FA3's one-wave schedule on full decode, the only route inside flash_attn_with_kvcache is non-varlen batched mode without cache_seqlens, which a paged KV cache cannot use; so 'more splits' is only reachable in a custom kernel, as triton-flash-decode-full-both-heads on muse_attention_decode already assumes with its own clamp(ceil(2*132/B), 1, 128) grid.

