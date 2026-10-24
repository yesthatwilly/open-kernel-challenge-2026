---
problem: muse_moe_w4a16
slug: post-norm-combine-hoisted-slot-loads
created: 2026-10-24T183618Z
base: master
area: post-expert norm+combine
change: >
  num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16 and the bs=4 test) only; bs>=32 and every other
  kernel stay master byte for byte. Rewrite master's _post_norm_combine_kernel body so all
  local-slot row loads are issued before any reduction. Today each of the 8 slots sits under 'if
  eid >= 0:', so the compiler emits 8 serial (load ids -> branch -> load 8 KB row ->
  sum-of-squares) chains and the program pays ~8 DRAM round trips. New body: load the 8 ids at
  once (ids = tl.load(ids_ptr + t*TOP_K_C + tl.arange(0, TOP_K_C))) and the 8 weights g likewise;
  then in a tl.static_range(TOP_K_C) loop, extract eid_s and g_s with
  tl.sum(tl.where(tl.arange(0,TOP_K_C)==s, ids, 0)) (same for g), and load each row as v_s =
  tl.load(y_ptr + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no branch, so the 8
  loads are independent; then acc += tl.where(eid_s >= 0, g_s, 0.0) * v_s * nw *
  tl.rsqrt(tl.sum((v_s*nw)^2)/H + eps). Remote slots are still never read (masked load), so the
  uninitialised y rows of remote slots are never touched and a fully masked row contributes
  exactly 0 (g forced to 0, so rsqrt(eps) of a zero row times 0 is 0, no NaN). fp32 math, eps, nw
  = w + 1 in fp32, output bf16 and the launch (grid (num_tokens,), BLOCK_H = 4096) are unchanged
  except num_warps 8 -> 16 so each thread holds 8 elements per row (64 live fp32 values for 8
  rows). Keep the function name so the profile row stays comparable.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "_post_norm_combine_kernel 3.1 us -> ~1.6-2 us: one round trip for the <=8 local rows instead of 8 serial ones; ~2-3% of a 49 us shape"
  - shapes: bs=8 *
    direction: faster
    why: 3.8 us -> ~2-2.5 us, ~1-1.5% of 108 us
  - shapes: bs=16 *
    direction: faster
    why: 4.3 us -> ~2.5 us, ~1% of 148 us
  - shapes: bs=32 *
    direction: flat
    why: untouched (SGLang fused_post_expert_norm_weighted_sum)
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# post-norm-combine-hoisted-slot-loads

## Change

num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16 and the bs=4 test) only; bs>=32 and every other kernel stay master byte for byte. Rewrite master's _post_norm_combine_kernel body so all local-slot row loads are issued before any reduction. Today each of the 8 slots sits under 'if eid >= 0:', so the compiler emits 8 serial (load ids -> branch -> load 8 KB row -> sum-of-squares) chains and the program pays ~8 DRAM round trips. New body: load the 8 ids at once (ids = tl.load(ids_ptr + t*TOP_K_C + tl.arange(0, TOP_K_C))) and the 8 weights g likewise; then in a tl.static_range(TOP_K_C) loop, extract eid_s and g_s with tl.sum(tl.where(tl.arange(0,TOP_K_C)==s, ids, 0)) (same for g), and load each row as v_s = tl.load(y_ptr + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no branch, so the 8 loads are independent; then acc += tl.where(eid_s >= 0, g_s, 0.0) * v_s * nw * tl.rsqrt(tl.sum((v_s*nw)^2)/H + eps). Remote slots are still never read (masked load), so the uninitialised y rows of remote slots are never touched and a fully masked row contributes exactly 0 (g forced to 0, so rsqrt(eps) of a zero row times 0 is 0, no NaN). fp32 math, eps, nw = w + 1 in fp32, output bf16 and the launch (grid (num_tokens,), BLOCK_H = 4096) are unchanged except num_warps 8 -> 16 so each thread holds 8 elements per row (64 live fp32 values for 8 rows). Keep the function name so the profile row stays comparable.

## Expected

- `bs=1 *` faster: _post_norm_combine_kernel 3.1 us -> ~1.6-2 us: one round trip for the <=8 local rows instead of 8 serial ones; ~2-3% of a 49 us shape
- `bs=8 *` faster: 3.8 us -> ~2-2.5 us, ~1-1.5% of 108 us
- `bs=16 *` faster: 4.3 us -> ~2.5 us, ~1% of 148 us
- `bs=32 *` flat: untouched (SGLang fused_post_expert_norm_weighted_sum)
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Judge on the _post_norm_combine_kernel row in top_kernels (Triton times reproduce within ~1%), not the headline: +0.8-1.0% geomean is under one run's resolution. Overlap: cuda-fused-moe-bs1-persistent-w13-silu-w2 (running) replaces this kernel at bs=1 and the queued cuda-fused-moe-midm-single-launch-routed replaces it at 2..32; cuda-nvfp4-swapab-mma-persistent-small-m and marlin-two-stream keep it, so a gain stacks on them. Same mechanism as the running cuda-dispatch-combine-pdl-verify combine (issue all slot loads first) but on disjoint shapes (<32 vs >32). Risk: register pressure at 16 warps is low (64 fp32 per thread); a correctness slip would show as a failed test, not a slow run.
