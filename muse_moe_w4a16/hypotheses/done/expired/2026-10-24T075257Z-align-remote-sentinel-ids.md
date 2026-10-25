---
problem: muse_moe_w4a16
slug: align-remote-sentinel-ids
created: 2026-10-24T075257Z
base: master
area: MoE align/dispatch
change: >
  Give moe_align_block_size ids in which remote routes are the out-of-range value num_experts (32)
  instead of -1, so its bounds check drops them from sorted_ids entirely instead of whatever it
  does with -1. prepare(): next to expert_map (remote -> -1), build align_map = torch.full((256,),
  32, int32) with align_map[start:start+count] = arange(count), stored in the bank. kernel_fn,
  every shape: align_ids = align_map[topk_ids].contiguous() (one extra small gather launch, ~1-2
  us) is passed ONLY to moe_align_block_size(align_ids, block_size_m, bank['count']); local_ids
  (with -1) still feed silu_and_mul(expert_ids=...) and
  fused_post_expert_norm_weighted_sum(slot_expert_ids=...) and the Marlin topk_weights path
  exactly as on master. Marlin keeps is_ep=True. Purpose: the mechanism probe for why master's
  32-expert align is 3.2 us slower than SGLang's 256-expert align at bs=1 (80.2 vs 77.0 us). If
  the align treats -1 as a valid signed index at 32 experts (e.g. counted into a local or spare
  bucket), remote routes form m-blocks that both Marlin GEMMs either compute against a local
  expert or skip; sentinel ids remove them, shrinking n_padded and Marlin's m-block loop at every
  shape.
expected:
  - shapes: bs=1 *
    direction: faster
    why: if -1 routes form a phantom block at 32 experts, dropping them saves one expert weight pass (~24 MB, ~4-5 us) net of the extra ~1-2 us gather
  - shapes: bs=8 *
    direction: faster
    why: same, ~56 remote routes per call
  - shapes: bs=16 *
    direction: faster
    why: same
  - shapes: bs=32 *
    direction: faster
    why: same
  - shapes: bs=128 *
    direction: flat
    why: remote blocks would be skipped by Marlin (is_ep) at small per-block cost; any gain is a smaller n_padded, under the CI at 462 us
  - shapes: bs=256 *
    direction: flat
    why: as bs=128
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: done
hold_reason: "adversarial review: Moot on master and overlapped by the running triton-map-align-small-m-port. The 3.2 us bs=1 gap it was written to explain (80.2 vs 77.0 us) no longer exists: ma"
review_at: 2026-10-24T120146Z
review_companions: null
outcome: expired
verdict_detail: "hold expired after 33 h: adversarial review: Moot on master and overlapped by the running triton-map-align-small-m-port. The 3.2 us bs=1 gap it w"
expired_at: 2026-10-25T211329Z
---

# align-remote-sentinel-ids

## Change

Give moe_align_block_size ids in which remote routes are the out-of-range value num_experts (32) instead of -1, so its bounds check drops them from sorted_ids entirely instead of whatever it does with -1. prepare(): next to expert_map (remote -> -1), build align_map = torch.full((256,), 32, int32) with align_map[start:start+count] = arange(count), stored in the bank. kernel_fn, every shape: align_ids = align_map[topk_ids].contiguous() (one extra small gather launch, ~1-2 us) is passed ONLY to moe_align_block_size(align_ids, block_size_m, bank['count']); local_ids (with -1) still feed silu_and_mul(expert_ids=...) and fused_post_expert_norm_weighted_sum(slot_expert_ids=...) and the Marlin topk_weights path exactly as on master. Marlin keeps is_ep=True. Purpose: the mechanism probe for why master's 32-expert align is 3.2 us slower than SGLang's 256-expert align at bs=1 (80.2 vs 77.0 us). If the align treats -1 as a valid signed index at 32 experts (e.g. counted into a local or spare bucket), remote routes form m-blocks that both Marlin GEMMs either compute against a local expert or skip; sentinel ids remove them, shrinking n_padded and Marlin's m-block loop at every shape.

## Expected

- `bs=1 *` faster: if -1 routes form a phantom block at 32 experts, dropping them saves one expert weight pass (~24 MB, ~4-5 us) net of the extra ~1-2 us gather
- `bs=8 *` faster: same, ~56 remote routes per call
- `bs=16 *` faster: same
- `bs=32 *` faster: same
- `bs=128 *` flat: remote blocks would be skipped by Marlin (is_ep) at small per-block cost; any gain is a smaller n_padded, under the CI at 462 us
- `bs=256 *` flat: as bs=128

## Risk

Risk: if this sgl_kernel align does not bounds-check ids >= num_experts it writes or counts out of range (a spare (num_experts+1)-th bucket would make block_experts = 32 and Marlin would read past the 32-expert bank): expect a test failure or a fault, which reads as 'the align has no >= num_experts guard', not as a Marlin problem. Output correctness otherwise is guarded by the -1 masks in silu_and_mul and the post-norm, which are unchanged. The extra gather is a known ~1-2 us cost on every shape, so a flat bs=1 means the -1 handling is not the 3.2 us. Read beside align-256-experts-below-128: if that one restores decode_1 parity and this one also gains at bs=1, the -1 bucket is the mechanism and this variant (fewer blocks than SGLang) is the one to keep; if only the 256 revert helps, the cost is in the 32-expert kernel itself.

## Adversarial review

Reviewed 2026-10-24T120146Z by an adversarial subagent (model opus).

**Verdict: hold.** Moot on master and overlapped by the running triton-map-align-small-m-port. The 3.2 us bs=1 gap it was written to explain (80.2 vs 77.0 us) no longer exists: master's bs=1 is 55.3 vs SGLang 77.3 us, and the align-256 review already traced the gap to the small-batch align kernel. At bs<=32 the port replaces this align call entirely, drops remote routes itself, and its prepare() self-check logs SGLang's r_pad against the true n_pad, which answers this probe's question deterministically and costs no timing run.

### Justification
A: The mechanism is weak and the motivation is stale. (1) The premise is gone. The 80.2 vs 77.0 us bs=1 comparison is from before post-expert-norm-triton-small-m landed. In bacd1f502f4e46fa master runs bs=1 at 55.3 us against SGLang's 77.3, and bs=8 and bs=16 also beat SGLang clearly. Only bs=32 trails (206.9 vs 204.2). (2) The 3.2 us was already explained by a different mechanism. The align-256-experts-below-128 review found that at numel<1024 with 32 experts, sgl-kernel picks moe_align_block_size_small_batch_expert_kernel (4.4-6.5 us), while the 256-expert call and bs>=128 use the regular kernel (2.6-2.8 us). align-single-token-ep's 1.2 us kernel recovered that 3.2 us almost exactly, and its Marlin time rose rather than fell, which argues against a phantom m-block. (3) The cost is understated. align_map[topk_ids] is a second int32-indexed gather, which in master's profile is index_elementwise_kernel (1.7-3.3 us) plus, very likely, an unrolled_elementwise int64 index cast (1.4-3.2 us). That is ~3-6 us per shape, not 1-2. At bs=1, if the phantom block does not exist, the result is a ~5-10% loss on a 55 us shape. (4) Risk: if sgl-kernel's align has no >=num_experts guard, the sentinel 32 indexes past its 32 counters or produces block_experts=32 beyond the bank, which is a fault, not a measurement. Weakest link: the whole expected gain depends on an unverified phantom-block behaviour that the existing evidence (align-single-token-ep's Marlin time, the small-batch kernel explanation) already points against. The expected directions are consistent with TASK.md's equal weights, but they would be read on top of ~3-6 us of added launches at every shape.

### Confounders on the current kernel
- sglang-marlin-path-parity: it defines the 32-expert moe_align_block_size call and the -1 expert_map that this probe modifies. The question is a sub-question of that patch's one deliberate difference from baseline.py, and align-256-experts-below-128 (neutral, 1.000x) already A/B'd that difference at M<128 without finding a Marlin-side effect.
- post-expert-norm-triton-small-m: no timing confound, because it consumes local_ids, which stay unchanged. But it moved bs=1/8/16 far ahead of SGLang, so this probe's motivating gap is no longer present on master.

### Interference and A/B needs
Overlap with triton-map-align-small-m-port (running, job c2feb68aafb44aeb): direct, and it covers 4 of 6 shapes. For num_tokens<=32 the port deletes the moe_align_block_size call this hypothesis would modify and replaces it with a Triton counting sort in which remote routes get no bucket. That is exactly the end state this probe wants (remote routes dropped from sorted_ids), plus removal of the gather and cast launches. If the port lands, this change at bs<=32 is dead code. Whether or not it lands, its prepare() self-check prints n_pad vs SGLang r_pad per case (T in {1,8,16,32}, 5 routings, 3 local_starts), which settles whether the 32-expert small-batch kernel buckets -1 routes. It does so deterministically and for free, which a noisy Marlin timing cannot. The remaining slice, bs=128/256 on the regular align kernel, is expected flat by the hypothesis itself, and profiled Marlin time drifts 20-30% on identical code there, so a check spent on it would not be interpretable. No companion would rescue it: an ab-revert of parity is meaningless, there is nothing to tune, and the only sub-change (the extra gather) is pure cost. Re-queue only if the port's pad lines show r_pad > n_pad (SGLang buckets -1 routes). Even then, re-scope it to num_tokens>=128 only, use a non-gather sentinel construction, and add an untimed prepare() check that r_pad shrinks on the regular kernel before spending a check.

### Open questions
- From triton-map-align-small-m-port's log: do any self-check cases print r_pad > n_pad [DIFF]? If none do, the 32-expert small-batch align does not bucket -1 routes at T<=32, and this hypothesis's mechanism is refuted for bs<=32.
- If DIFF lines appear, does the regular moe_align_block_size_kernel (numel>=1024, bs=128/256) bucket -1 routes as well? An untimed prepare() comparison of n_padded on synthetic ids with and without remote routes answers this without a check.
- Does sgl-kernel's align bounds-check ids >= num_experts at all? That needs to be known before any sentinel variant is built.

