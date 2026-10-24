---
problem: muse_decode_layer
slug: moe-align-small-t-restack-on-master
created: 2026-10-24T165302Z
base: master
area: MoE align/dispatch
change: >
  Restack decode own moe-align-single-kernel-small-t (neutral at 1.043x on the old expert-id-map
  base, but every T<=32 row moved the right way: sliding decode_16 262.8 -> 254.6 us, sliding
  decode_32 261.1 -> 251.1, full verify_1 240.8 -> 235.3) onto master, the way muse_moe_w4a16
  restacked its own neutral triton-map-align-small-m-port onto its block-m gain and got 1.312x
  with both parents per-shape gains reproduced. Take the patch
  muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch (branch
  hyp/muse_decode_layer/moe-align-single-kernel-small-t) and 3-way merge its T <= 32 additions
  onto master unchanged: ALIGN_FUSED_MAX_TOKENS = 32, _align_small_kernel (one program: local =
  (id - local_start) & 255 if < 32 else -1, counting sort over the 32 local experts, padding to
  block_size_m with the numel sentinel, block_experts, n_padded), _align_small, _align_self_check
  and the prepare() warm-up over T 1..32 for each block_size_m the rule can pick. In _experts /
  kernel_fn at tokens <= 32, the single _align_small launch replaces master (topk_ids -
  local_start) % N_EXPERTS, the bank["expert_map"][...] gather, .contiguous() and
  moe_align_block_size(local_ids, block_size_m, N_EXPERTS); its local_ids feed silu_and_mul and
  master _post_norm_combine_kernel (which already skips -1 slots) and the returned topk_ids output
  is unchanged. Do NOT bring the expert-id-map-single-kernel parent: at tokens > 32 keep master
  path byte for byte (remap, expert_map gather, moe_align_block_size over 256, block_size_m capped
  at 16). Resolve conflicts in favour of master everywhere outside the tokens <= 32 align branch.
expected:
  - shapes: "*load=decode_16 *"
    direction: faster
    why: 3-4 align/id-map launches become one; the parent measured -8 us sliding, -7 us full on this shape (1.3-1.4%); on master the post-norm is one Triton program, so the align nodes are a larger share of the remaining small-kernel chain
  - shapes: "*load=decode_32 *"
    direction: faster
    why: parent -10 us on sliding decode_32 (1.2%), the T=32 post-norm on master still runs SGLang fused_post_expert_norm (32 is not < 32), unchanged
  - shapes: "*load=verify_1 *"
    direction: faster
    why: parent -2 to -5.5 us (0.5-1.3%); master profile shows _moe_align_small_numel_kernel 3.6 us plus the remap/gather launches at T=8
  - shapes: "*load=decode_1 *"
    direction: faster
    why: parent only -1.2 us here (0.2-1.9%), near drift
  - shapes: "*load=verify_16 *"
    direction: flat
    why: T=128 keeps master path; control
  - shapes: "*load=verify_32 *"
    direction: flat
    why: T=256 keeps master path; control
risk: low
depends_on: moe-align-single-kernel-small-t
priority: 4
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T165707Z
review_companions: null
---

# moe-align-small-t-restack-on-master

## Change

Restack decode own moe-align-single-kernel-small-t (neutral at 1.043x on the old expert-id-map base, but every T<=32 row moved the right way: sliding decode_16 262.8 -> 254.6 us, sliding decode_32 261.1 -> 251.1, full verify_1 240.8 -> 235.3) onto master, the way muse_moe_w4a16 restacked its own neutral triton-map-align-small-m-port onto its block-m gain and got 1.312x with both parents per-shape gains reproduced. Take the patch muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch (branch hyp/muse_decode_layer/moe-align-single-kernel-small-t) and 3-way merge its T <= 32 additions onto master unchanged: ALIGN_FUSED_MAX_TOKENS = 32, _align_small_kernel (one program: local = (id - local_start) & 255 if < 32 else -1, counting sort over the 32 local experts, padding to block_size_m with the numel sentinel, block_experts, n_padded), _align_small, _align_self_check and the prepare() warm-up over T 1..32 for each block_size_m the rule can pick. In _experts / kernel_fn at tokens <= 32, the single _align_small launch replaces master (topk_ids - local_start) % N_EXPERTS, the bank["expert_map"][...] gather, .contiguous() and moe_align_block_size(local_ids, block_size_m, N_EXPERTS); its local_ids feed silu_and_mul and master _post_norm_combine_kernel (which already skips -1 slots) and the returned topk_ids output is unchanged. Do NOT bring the expert-id-map-single-kernel parent: at tokens > 32 keep master path byte for byte (remap, expert_map gather, moe_align_block_size over 256, block_size_m capped at 16). Resolve conflicts in favour of master everywhere outside the tokens <= 32 align branch.

## Expected

- `*load=decode_16 *` faster: 3-4 align/id-map launches become one; the parent measured -8 us sliding, -7 us full on this shape (1.3-1.4%); on master the post-norm is one Triton program, so the align nodes are a larger share of the remaining small-kernel chain
- `*load=decode_32 *` faster: parent -10 us on sliding decode_32 (1.2%), the T=32 post-norm on master still runs SGLang fused_post_expert_norm (32 is not < 32), unchanged
- `*load=verify_1 *` faster: parent -2 to -5.5 us (0.5-1.3%); master profile shows _moe_align_small_numel_kernel 3.6 us plus the remap/gather launches at T=8
- `*load=decode_1 *` faster: parent only -1.2 us here (0.2-1.9%), near drift
- `*load=verify_16 *` flat: T=128 keeps master path; control
- `*load=verify_32 *` flat: T=256 keeps master path; control

## Risk

Low risk (the kernel already passed all 13 tests and its self-check), low expected value (~+1% on the headline). It is the stacked follow-up FINDINGS asks for on moe-align-single-kernel-small-t; the difference from the neutral run is the base (master now carries cap-16, Triton post-norm, head pairing), not the code, so a reviewer may prefer to merge-without-submit if the parent rows are judged sufficient. Re-check against moe_w4a16 done: moe two-smaller block-m at T>=128 (bs=128 339.7, bs=256 466.9 us) equals its cap-16 port (337.8 / 465.3 us), so decode cap-16 at verify_16/32 is not retuned here. moe-align-single-kernel-small-t-control-capacity-256 stays held.

## Adversarial review

Reviewed 2026-10-24T165707Z by an adversarial subagent (model opus).

**Verdict: proceed.** This restacks an already-correct, low-risk fusion that moved every T<=32 row the right way in its first run (paired ratio +0.5-1.4%, verify_16/32 flat). On master it should save the same ~4-6 us per T<=32 shape, about +0.5-0.9% on the headline. A single run will almost certainly classify neutral, so the merge has to be decided per shape, read together with the parent run, and not from the headline. No companion changes that decision.

### Justification
Mechanism. On master, at T<=32, the align path issues these launches: (topk_ids - local_start) (elementwise), % 256 (elementwise), the bank['expert_map'][...] gather (index kernel), .contiguous() (a no-op on the gather output), then moe_align_block_size(local_ids, bsm, 256). The profile shows that last one as a single Triton _moe_align_small_numel_kernel at 3.5-3.6 us (verify_1, master 9dcaec316b3045a7), plus its host-side buffer allocations. So there are about 4 launches today, not 'the 3-4 launches moe_align_block_size issues' the parent assumed: SGLang's align is already one kernel at this numel. The replacement is not free. In the parent profile (0a0af8669ffe4cb0, decode_32 sliding), _align_small_kernel itself costs 4.3 us. The net saving is therefore the three elementwise/gather nodes, about 1.5-2.5 us each as graph nodes, plus the difference between SGLang's align and the new kernel. Call it 3-6 us per shape, which matches what the parent measured (decode_16 sliding -8 us, decode_32 sliding -10 us, verify_1 full -5.5 us; some of that is drift). The paired ratios against SGLang in the parent run, versus its base expert-id-map-single-kernel, support a small real effect: decode_16 sliding 1.016 -> 1.030, decode_16 full 1.009 -> 1.022, decode_32 sliding 1.023 -> 1.035, verify_1 full 1.140 -> 1.154, decode_1 about flat (1.028 -> 1.030). The controls verify_16/32, whose code was byte-identical, stayed flat (1.017 -> 1.017, 1.015 -> 1.011). On master the absolute saving should be the same, because Marlin sees the same bsm (8 at T<=28, 16 at T=32) and the same align outputs as in the parent run. The shapes are now shorter (decode_16 sliding 263 -> 226 us, verify_1 sliding 227 -> 201 us), so the relative effect grows slightly, as the hypothesis says. Floors: these shapes are 3-4x above SoL (decode_16 sliding 226 vs 77 us), and the gap is launch/latency-bound small kernels plus Marlin. Removing nodes is a legitimate lever, but a small one. Expected directions match TASK.md. The score is a geomean over 6 equally weighted loads; 4 loads (decode_1/16/32, verify_1) are touched and verify_16/32 are true controls on master's byte-identical T>32 path. Estimated headline: (+0.2 +1.3 +1.0 +0.8 +0 +0)/6, about +0.5-0.9%, so roughly 1.170-1.174 against 1.164 [1.146, 1.181]. That is inside master's CI. Weakest link: the effect per shape (0.5-1.5% paired) is at or below the documented run drift (1.5-2.5% per shape, LESSONS Measurement), and the profile cannot attribute it. Marlin's own profiled time swings 20-40% between runs on the same code because routing seeds differ (decode_32 sliding Marlin 104-148 us, verify_1 66-105 us across the archived runs), and the replaced align/gather rows mostly sit in the '... N more' tail of the top-12 report. Attribution must come from (a) the sign and consistency of the paired ratio across all 8 T<=32 rows, together with the parent run, and (b) the _align_small_kernel row versus master's _moe_align_small_numel_kernel and at::native elementwise rows, wherever the profile shows them. Second weakest: LESSONS already lists this parent among the 'fusing a handful of tiny kernels was flat on decode-sized shapes' results. This restack is the stacked follow-up FINDINGS asks for, not a new mechanism.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: it fixes moe_align_block_size's num_experts at 256, so master's sorted_ids capacity is T*8 + 256*(bsm-1) (T=32: 4096 slots / 256 m-blocks) and remote routes get skipped -1 blocks. The restack silently shrinks this to 32-expert capacity with no remote bucket (T=32: 736 slots / 46 m-blocks). If Marlin sizes its grid or m-block loop from sorted_ids.size(0) on the host, part of any T<=32 change is a Marlin-partition effect (fewer early-exit blocks) and not launch removal. This is the same bundled sub-change the parent's reviewer flagged, and its control (moe-align-single-kernel-small-t-control-capacity-256) is held. It does not make a gain ambiguous for the merge decision, because both sub-changes are legal and ship together. It only limits what LESSONS can claim about why.
- marlin-block-m-cap-16: at T<=32 min(bsm, 16) is a no-op (8 or 16), so it does not confound the T<=32 rows. Risk at merge: the parent's _block_size_m helper has no cap. If the restack routes the T>32 path through that helper, verify_16/32 lose cap-16 and regress 8-15%, which would wreck the control rows. The cap must stay on the T>32 path byte for byte.
- post-expert-norm-triton-small-t: it consumes local_ids at T<32 (skipping -1 slots), and SGLang's fused_post_expert_norm uses them via slot_expert_ids at T=32. The new kernel's int32 [T,8] local_ids with -1 for remote routes match what master's expert_map gather produces, so the numerical path is unchanged. It reduced the post-expert kernel chain, so align nodes are a larger share of what remains, which is consistent with the expected directions and not a confound. The T=32 boundary differs: decode_32 runs the new align but SGLang's post-norm, so decode_32 isolates the align change slightly more cleanly than decode_16.
- fa3-hdim128-head-pairing-batched: it touches only attention on full layers with batch > 1. It is disjoint from the MoE align path and changes only the denominators of the full decode_16/32 rows. No interaction.

### Interference and A/B needs
No companion is warranted. (i) ab-revert: none of the landed hypotheses interacts with the T<=32 align branch in a way that would flip the sign. cap-16 is a no-op at T<=32, post-norm consumes an identical local_ids tensor, and head pairing is in a different subsystem. Reverting any of them would not change what this result means. (ii) ab-tune: there is nothing to tune. bsm at T<=32 is fixed by SGLang's rule, and the kernel has no knobs that matter at <=256 routes. (iii) control: the one real ambiguity is capacity (32 vs 256 experts) versus launch fusion, which is moe-align-single-kernel-small-t-control-capacity-256. A gain is not ambiguous for the merge decision, because the whole T<=32 branch ships as a unit. Only a flat or negative result would be ambiguous: fusion could have helped while the capacity change hurt Marlin. Even then, the parent run already showed the combined change moving the right way on every T<=32 row, so a flat restack would most likely be read as drift, not as a hidden capacity loss. Decode checks are spent, the expected value is ~+0.5-0.9%, and the control's held reason still applies, so a second run is not worth its slot. If the restack comes back flat or negative on the T<=32 rows while verify_16/32 stay flat, unhold the capacity-256 control then, rebased on this branch, instead of pre-paying for it now. How to read the run: judge it per shape against the parent. Expect the paired ratio vs SGLang to rise on most of the 8 T<=32 rows (decode_16/32 and verify_1 most, decode_1 about flat) and verify_16/32 to be unchanged within ~1.5%. Do not use the classify verdict alone; it will very likely say neutral/within-ci, as it did for the parent.

### Open questions
- Does the full per-kernel profile (not the top-12 report) record _align_small_kernel and master's _moe_align_small_numel_kernel plus the elementwise/gather rows at decode_1/16/32? If not, attribution relies only on paired ratios. Consider printing the full kernel list for the T<=32 shapes in the harvest.
- Is a scored submit the right spend at ~+0.5-0.9% expected? The base change does not alter what Marlin sees at T<=32, so merge-without-submit after a careful 3-way merge is defensible. That would only be acceptable if the next scored run (any hypothesis) carries it, and its T<=32 rows are then checked against this review's expectations.
- Does moe_wna16_marlin_gemm size its grid from sorted_ids.size(0) / moe_block_size on the host? If it does, part of the parent's gain is the 256 -> 32 capacity shrink (fewer skipped -1 blocks), and that lesson would carry to the T>32 path. moe-align-local-expert-count, held on moe_w4a16's flat result, probes that path.
- Parent-run baseline drift: at T>32 (byte-identical code) the parent's SGLang reference times moved +7% (verify_32 sliding 681 -> 733 us) while the paired ratio held. Confirm this run also reads ratios and not absolute microseconds when comparing against the parent.

