---
problem: muse_moe_w4a16
slug: post-norm-triton-hoisted-decode32
created: 2026-10-24T183635Z
base: master
area: post-expert norm+combine
change: >
  num_tokens == 32 only (decode_32); bs<32 keeps master's _post_norm_combine_kernel unchanged and
  bs>=33 (incl. the bs=33 test) keeps SGLang's fused_post_expert_norm_weighted_sum. Add a new
  Triton kernel _post_norm_combine_hoisted_kernel with master _post_norm_combine_kernel's
  signature and math (out[t] = sum over local slots of g * rmsnorm(y[t,s] * (1 + w)), fp32, remote
  slots never read, zero row for a token with no local slot) but with every slot load issued
  before any reduction: load the 8 ids and 8 weights of the token as vectors, then in
  tl.static_range(TOP_K_C) extract eid_s/g_s with tl.sum(tl.where(arange==s, vec, 0)), load v_s =
  tl.load(y + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no 'if', and accumulate
  tl.where(eid_s >= 0, g_s, 0.0) * v_s*nw * rsqrt(sum((v_s*nw)^2)/H + eps). Launch grid (32,),
  BLOCK_H = 4096, num_warps=16. In kernel_fn, at the end: if num_tokens < SMALL_M_POST_NORM:
  master's path unchanged; elif num_tokens == 32: y = y.contiguous(), out = torch.empty((32, H),
  bf16) and launch the new kernel with state['post_norm_w'] (as master's small-M path does); else
  SGLang's call unchanged. Compile it in prepare() once on a dummy y/ids (untimed) so the first
  call does not JIT.
expected:
  - shapes: bs=32 *
    direction: faster
    why: _post_expert_norm_weighted_sum_kernel 4.5 us -> ~2.5 us of hoisted Triton; ~1% of 200 us. The old serial-load Triton kernel measured 3.8 us here (post-norm-triton-all-m), so hoisting is the new ingredient
  - shapes: bs=1 *
    direction: flat
    why: untouched
  - shapes: bs=8 *
    direction: flat
    why: untouched
  - shapes: bs=16 *
    direction: flat
    why: untouched
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T183956Z
review_companions: null
---

# post-norm-triton-hoisted-decode32

## Change

num_tokens == 32 only (decode_32); bs<32 keeps master's _post_norm_combine_kernel unchanged and bs>=33 (incl. the bs=33 test) keeps SGLang's fused_post_expert_norm_weighted_sum. Add a new Triton kernel _post_norm_combine_hoisted_kernel with master _post_norm_combine_kernel's signature and math (out[t] = sum over local slots of g * rmsnorm(y[t,s] * (1 + w)), fp32, remote slots never read, zero row for a token with no local slot) but with every slot load issued before any reduction: load the 8 ids and 8 weights of the token as vectors, then in tl.static_range(TOP_K_C) extract eid_s/g_s with tl.sum(tl.where(arange==s, vec, 0)), load v_s = tl.load(y + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no 'if', and accumulate tl.where(eid_s >= 0, g_s, 0.0) * v_s*nw * rsqrt(sum((v_s*nw)^2)/H + eps). Launch grid (32,), BLOCK_H = 4096, num_warps=16. In kernel_fn, at the end: if num_tokens < SMALL_M_POST_NORM: master's path unchanged; elif num_tokens == 32: y = y.contiguous(), out = torch.empty((32, H), bf16) and launch the new kernel with state['post_norm_w'] (as master's small-M path does); else SGLang's call unchanged. Compile it in prepare() once on a dummy y/ids (untimed) so the first call does not JIT.

## Expected

- `bs=32 *` faster: _post_expert_norm_weighted_sum_kernel 4.5 us -> ~2.5 us of hoisted Triton; ~1% of 200 us. The old serial-load Triton kernel measured 3.8 us here (post-norm-triton-all-m), so hoisting is the new ingredient
- `bs=1 *` flat: untouched
- `bs=8 *` flat: untouched
- `bs=16 *` flat: untouched
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Shape-disjoint sibling of post-norm-combine-hoisted-slot-loads (bs<32): the two restack by 3-way merge into one hoisted kernel for bs<=32. decode_32 Marlin rows are seed-noisy (147-265 us), so judge only on the post-norm kernel row in top_kernels. post-norm-triton-all-m (old serial kernel at every M) was within CI; this differs by the hoisted loads and by bs=32 only. Overlap: the queued cuda-fused-moe-midm-single-launch-routed would replace it; the running swap-AB and two-stream keep this post-norm, so it stacks.

## Adversarial review

Reviewed 2026-10-24T183956Z by an adversarial subagent (model opus).

**Verdict: proceed.** A cheap, shape-disjoint and rules-clean test of load hoisting at decode_32 that can only be read from the post-norm kernel row (its headline effect is about 0.15% of the geomean), so proceed without companions, ideally after its bs<32 sibling post-norm-combine-hoisted-slot-loads has shown whether hoisting moves that kernel at all.

### Justification
Mechanism: master's _post_norm_combine_kernel puts each of the 8 slots under a scalar 'if eid >= 0', which serialises up to 8 dependent L2 round trips per program. y was just written by the w2 Marlin, is about 2 MB at bs=32 and sits in L2, so the kernel is latency-bound rather than bandwidth-bound, and issuing every masked load up front plausibly cuts it to one or two round trips. Profile support: at bs=32 SGLang's _post_expert_norm_weighted_sum_kernel is 3.6-4.5 us across jobs (3.6 in 3cabc739 profile 2, 4.0 in FINDINGS, 4.5 in the hypothesis), and the old serial Triton kernel measured 3.8 us there (post-norm-triton-all-m, job 2ec5387f). So the new ingredient really is the hoisting, and about 2.5 us is a reasonable target. The expected directions are consistent: only decode_32 moves, and every other shape's code path is byte-identical. Weakest link: the payoff is about 1-2 us on a roughly 200 us shape that carries 1/6 of the geomean, about 0.15% of the headline, far inside the ±1.5% CI. decode_32's Marlin time is seed-noisy (169-190 us), and the SGLang post-norm row itself drifts about 0.9 us between jobs, which is as large as the expected saving. A result is therefore readable only as the new kernel's row against the 3.6-4.5 us band, and anything above about 3 us is ambiguous. Two smaller doubts: ptxas may already have if-converted and hoisted part of the serial kernel, and at bs=32 under skewed routing the mean number of local slots per token may be only about 2-4, which shrinks the serial chain that hoisting removes. Grid (32,) also leaves about 100 SMs idle, so per-program latency still dominates after the change.

### Confounders on the current kernel
- post-expert-norm-triton-small-m: defines the math and the bs<32 kernel that this one copies; decode_32 still runs SGLang's kernel on master, so it does not confound the decode_32 measurement, it only sets the code template.
- triton-map-align-on-block-m-restack: _align_small_kernel produces the local_ids the new kernel reads at bs=32; it is upstream in the same graph but its time is a separate profile row, so it cannot leak into the post-norm row.
- marlin-block-m-two-smaller-on-postnorm and align-parallel-per-expert-verify: act only at num_tokens >= 128 / > 32 and do not touch decode_32; no interaction.
- sglang-marlin-path-parity: fixes the y layout [M, 8, H] with uninitialised remote rows, which the masked loads must respect; needed, but not a confounder of timing.

### Interference and A/B needs
No companion is warranted. The change is a separate kernel used only at bs=32, so its own profile row attributes it. The control already exists: SGLang's kernel row at bs=32 in this job's own baseline profile and in 3cabc739, plus the old serial Triton kernel at 3.8 us in post-norm-triton-all-m. Reverting any landed hypothesis would not change what the decode_32 post-norm kernel sees. A serial-vs-hoisted control at bs=32 would duplicate post-norm-triton-all-m. The cheapest way to tighten the read is an in-run, untimed microbenchmark in prepare(): time the new kernel, master's serial _post_norm_combine_kernel and SGLang's fused_post_expert_norm_weighted_sum on one synthetic [32, 8, 4096] y with a fixed synthetic routing (CUDA events, about 200 iterations each), and print the three medians to the log. That gives a same-GPU A/B without spending a check. Sequencing matters more than companions: run this after post-norm-combine-hoisted-slot-loads. If the sibling's _post_norm_combine_kernel row does not drop, the mechanism has failed and this run can be dropped.

### Open questions
- At bs=32 under the skewed ep8 routing, how many local slots does a token have on average? If it is about 2, the serial chain is short and hoisting buys little.
- Did ptxas already if-convert the serial kernel's loads? Check its SASS for predicated LDGs ahead of the reductions. If so, the 3.8 us serial time is already 'hoisted' and the floor is elsewhere (launch, grid of 32).
- Would a 2-D grid (32 tokens x 2-4 H-chunks, with sum-of-squares needing the full row) or one program per (token, slot) with an atomic combine beat hoisting, given that only 32 of 132 SMs are busy?
- If both this and the sibling win, restack them as a single hoisted kernel for bs<=32 and retire the separate decode32 branch.

