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
review: required
status: queued
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
