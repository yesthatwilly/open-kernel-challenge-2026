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
review: required
status: queued
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
