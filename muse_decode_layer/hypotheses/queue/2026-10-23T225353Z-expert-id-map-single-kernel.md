---
problem: muse_decode_layer
slug: expert-id-map-single-kernel
created: 2026-10-23T225353Z
base: master
area: MoE align/dispatch
change: >
  Replace the three small kernels that turn the routing output into Marlin's local expert ids with
  one Triton kernel. Today kernel_fn computes local_ids = (topk_ids - data['local_start']) %
  N_EXPERTS (a sub and a remainder launch) and _experts then gathers bank['expert_map'][topk_ids]
  (an index launch) to get int32 ids in 0..31 or -1. The new kernel (one program over the T*8
  int32 ids, no host-side values) reads topk_ids and the local_start device scalar and writes a
  single int32 [T, 8] tensor with ((id - local_start) mod 256) where that is < 32 and -1
  otherwise, exactly what expert_map produced; that tensor is passed unchanged to
  moe_align_block_size, silu_and_mul (expert_ids) and fused_post_expert_norm_weighted_sum
  (slot_expert_ids). topk_weights handling, moe_align_block_size, the Marlin calls and the
  returned topk_ids are untouched; the expert_map table is no longer needed in the bank.
  Attribution: 3 launches become 1 on every shape; only measurable on the launch-bound T<=8
  shapes.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "two ~2-3 us launches removed from a ~172 us launch-bound shape: ~2-3%"
  - shapes: "*load=verify_1*"
    direction: faster
    why: "same two launches on a ~268 us shape: ~1.5-2%, near the CI edge"
  - shapes: "*load=decode_32*"
    direction: flat
    why: the saved launches are under 1% of ~260-620 us; control
  - shapes: "*load=verify_16*"
    direction: flat
    why: same; control
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T032546Z
review_companions: null
---

# expert-id-map-single-kernel

## Change

Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_ids - data['local_start']) % N_EXPERTS (a sub and a remainder launch) and _experts then gathers bank['expert_map'][topk_ids] (an index launch) to get int32 ids in 0..31 or -1. The new kernel (one program over the T*8 int32 ids, no host-side values) reads topk_ids and the local_start device scalar and writes a single int32 [T, 8] tensor with ((id - local_start) mod 256) where that is < 32 and -1 otherwise, exactly what expert_map produced; that tensor is passed unchanged to moe_align_block_size, silu_and_mul (expert_ids) and fused_post_expert_norm_weighted_sum (slot_expert_ids). topk_weights handling, moe_align_block_size, the Marlin calls and the returned topk_ids are untouched; the expert_map table is no longer needed in the bank. Attribution: 3 launches become 1 on every shape; only measurable on the launch-bound T<=8 shapes.

## Expected

- `*load=decode_1*` faster: two ~2-3 us launches removed from a ~172 us launch-bound shape: ~2-3%
- `*load=verify_1*` faster: same two launches on a ~268 us shape: ~1.5-2%, near the CI edge
- `*load=decode_32*` flat: the saved launches are under 1% of ~260-620 us; control
- `*load=verify_16*` flat: same; control

## Risk

Risk: correctness is exact integer arithmetic, so the only failure modes are dtype (moe_align_block_size and silu_and_mul expect int32 ids; FusedMuseSparkRouting already returns int32 since kernel_fn returns topk_ids as the int32 output) and the modulo of a negative difference (use a Python-style non-negative mod, as torch's % does). The effect size is small: if decode_1 moves less than the +-1.5% CI, record it as flat rather than a gain and keep the kernel only if the stacked moe-align-single-kernel-small-t (which subsumes it) lands. Confounder: if the adversarial review finds that torch already fuses sub+remainder into one kernel on this image, the saving is 1 launch, not 2.

## Adversarial review

Reviewed 2026-10-24T032546Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** Fusing the sub / remainder / expert_map-gather launches into one Triton kernel is a real, exactly-attributable launch-count reduction on the launch-bound T<=8 shapes (decode_1 runs ~43 kernels in 172 us, nearly serial), nothing landed interacts with it, and no companion run would make a flat result more interpretable; proceed, with the implementation cautions on signed modulo and Triton specialization/compile timing below.

### Justification
A. Mechanism: master (sglang-pipeline-parity) computes local_ids = (topk_ids - data['local_start']) % N_EXPERTS in kernel_fn (submission.py:349; eager torch, so a broadcast-sub launch and a remainder launch) and then _experts gathers bank['expert_map'][topk_ids] (submission.py:255; an index launch, and very likely a preceding int32->int64 index-cast copy launch, since torch's advanced-indexing make_info converts kInt indices to kLong before index_kernel). All of these sit on the main stream after main_stream.wait_stream(side_stream), i.e. on the serial critical path, on every shape. The decode_1 sliding profile (profiles/submission.5755a412dda244a1.txt, profile 0) shows 175.9 us total GPU time vs 174.0 us busy (kernels essentially serial), 12 listed kernel names plus '24 more' rows, i.e. ~43 launches, with the small at::native elementwise kernels at 1.6-2.8 us each; speed-of-light is 40.0 us, so the shape is launch/latency-bound and each removed launch is worth ~2-3 us (kernel plus inter-node gap). Removing 2 (or 3) launches is therefore ~4-8 us = 2.5-4.5% on decode_1 (172 us) and ~1.5-3% on verify_1 (229-243 us), consistent with the hypothesis's 2-3% / 1.5-2% and with the TASK.md weights (decode_1 and verify_1 are two of six equally weighted loads; the overall geomean moves only ~0.4-0.7%, inside the +-1.5% run CI, so attribution must come from the per-shape rows). T>=16 flat is right: the same launches are <1% of 262-1048 us. Numerics are exact integer arithmetic; ((id - local_start) mod 256) < 32 ? that : -1 is identical to expert_map[(id - local_start) mod 256] because expert_map is arange(32) then -1. Weakest link: effect size versus run-to-run per-shape noise. The same master code scored decode_1 sliding at 1.0145x in check e260fde413e448c7 and 0.999x in submit 5755a412dda244a1, so the per-shape ratio drifts ~1.5% (~2.6 us) between runs even though the within-run err is ~0.4%; a single run's decode_1 row at +2% is only ~1.3 sigma against that drift, so 'small gain' and 'flat' may not be separable from one run. The hypothesis already accepts flat as an outcome and the stacked moe-align-single-kernel-small-t measures a larger version of the same mechanism, so this is acceptable. Second weakest: the '3 launches become 1' count is not verified from a profile (the relevant kernels are hidden in the '24 more' rows); if the gather is really cast+index it is 4->1 (stronger), and there is no eager-mode fusion in torch that would make it fewer than 3->1, so the direction of the uncertainty favours the hypothesis.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8 (the only landed item, and the base itself): its one structural change (ROUTER_FP32_MAX_TOKENS 8->1) alters only the side-stream router GEMMs at T=8; the id-map launches run on the main stream after the side-stream join regardless of the router path, so the number of serial launches saved is the same with the constant at 8 or 1. The expert_map gather being inside _experts is SGLang's own code copied by that landing, not an interaction. Not a confounder.

### Interference and A/B needs
No companion is warranted. The change replaces one code path (three or four eager launches) with one Triton launch and touches nothing else; a gain on decode_1/verify_1 with decode_16/32 and verify_16/32 flat is attributable only to this change, and a loss could only mean the Triton kernel costs more than the launches it replaces (a single program over <=2048 int32 is ~2 us) or a compile/specialization problem, neither of which an ab-revert or ab-tune of the landed router constant could disambiguate (the router change is on the side stream before the join and is orthogonal). A 'control' isolating a sub-change is impossible because there is only one sub-change. The genuinely ambiguous outcome is 'flat', and that ambiguity is statistical (run-to-run per-shape drift ~1.5% vs an expected ~2-3% effect), which another single run would not cure and which the queued moe-align-single-kernel-small-t (which subsumes this kernel at T<=32 and removes 4-5 launches) will resolve with a larger signal. In-flight cache-store-fused-fp8-cast is in the cache-store area before attention and is additive and independent; if it lands first the base shifts but nothing about this mechanism changes.

### Open questions
- Exact current launch count of the id-map path: is bank['expert_map'][topk_ids] with an int32 index one launch (index kernel) or two (int32->int64 cast copy then index)? The profile's '24 more' rows hide the small kernels; a profile with the full kernel list, or nsys on the judge image, would pin the saved count at 2 or 3.
- Does the judge call kernel_fn eagerly for each timed shape before CUDA-graph capture (so an uncovered Triton specialization compiles outside capture)? If not, the fixed-BLOCK design above is mandatory rather than advisable.
- Can one run's decode_1 row separate a +2-3% gain from flat, given the observed 1.014x vs 0.999x drift of identical code across e260fde413e448c7 and 5755a412dda244a1? Record a sub-CI move as flat per the hypothesis's own rule and let moe-align-single-kernel-small-t carry the decision.
- Minor: SGLang's moe_align_block_size with num_experts=256 and -1 ids is unchanged behaviour here, but the queued moe-align-local-expert-count (32 experts) will stack on this kernel's output; the -1 sentinel must survive both.

