---
problem: muse_moe_w4a16
slug: post-norm-pdl-launch-small-m
created: 2026-10-24T183649Z
base: master
area: launch overhead
change: >
  num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel
  body is otherwise unchanged and every other launch is master's. Launch the post-norm with
  programmatic dependent launch so its launch latency and prologue overlap the tail of the w2
  Marlin grid inside the CUDA graph. (1) In _post_norm_combine_kernel add a constexpr USE_PDL;
  after the static loads that do not depend on Marlin (offs, mask, nw = norm weight + 1) and
  before the first load of ids/topk_weights/y, do 'if USE_PDL: tl.extra.cuda.gdc_wait()'.
  (topk_weights and local_ids are written before Marlin, but keep every data-dependent load after
  the wait for simplicity.) No gdc_launch_dependents call (nothing after it in the graph needs
  it). (2) Launch with launch_pdl=True (Triton launcher kwarg) when USE_PDL. (3) prepare(): set
  state['pdl_ok'] by a self-test: build a tiny y/ids/tw case, capture in a torch.cuda.CUDAGraph a
  plain elementwise kernel writing y followed by the PDL post-norm launch, replay 3 times, and
  compare with the USE_PDL=False launch (torch.equal). Any exception (kwarg unknown to this
  Triton, gdc_wait missing, capture error) or mismatch -> pdl_ok False. Also warm up/compile both
  variants for every num_tokens < 32 there. kernel_fn passes USE_PDL=state['pdl_ok']. Print one
  line '[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)' in prepare so the run log says which
  path was measured.
expected:
  - shapes: bs=1 *
    direction: faster
    why: removes ~1-2 us of launch gap + prologue between w2 Marlin and the post-norm, ~2-3% of a 49 us shape
  - shapes: bs=8 *
    direction: faster
    why: same absolute saving, ~1% of 108 us
  - shapes: bs=16 *
    direction: faster
    why: same, ~0.7-1% of 148 us
  - shapes: bs=32 *
    direction: flat
    why: untouched
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# post-norm-pdl-launch-small-m

## Change

num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel body is otherwise unchanged and every other launch is master's. Launch the post-norm with programmatic dependent launch so its launch latency and prologue overlap the tail of the w2 Marlin grid inside the CUDA graph. (1) In _post_norm_combine_kernel add a constexpr USE_PDL; after the static loads that do not depend on Marlin (offs, mask, nw = norm weight + 1) and before the first load of ids/topk_weights/y, do 'if USE_PDL: tl.extra.cuda.gdc_wait()'. (topk_weights and local_ids are written before Marlin, but keep every data-dependent load after the wait for simplicity.) No gdc_launch_dependents call (nothing after it in the graph needs it). (2) Launch with launch_pdl=True (Triton launcher kwarg) when USE_PDL. (3) prepare(): set state['pdl_ok'] by a self-test: build a tiny y/ids/tw case, capture in a torch.cuda.CUDAGraph a plain elementwise kernel writing y followed by the PDL post-norm launch, replay 3 times, and compare with the USE_PDL=False launch (torch.equal). Any exception (kwarg unknown to this Triton, gdc_wait missing, capture error) or mismatch -> pdl_ok False. Also warm up/compile both variants for every num_tokens < 32 there. kernel_fn passes USE_PDL=state['pdl_ok']. Print one line '[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)' in prepare so the run log says which path was measured.

## Expected

- `bs=1 *` faster: removes ~1-2 us of launch gap + prologue between w2 Marlin and the post-norm, ~2-3% of a 49 us shape
- `bs=8 *` faster: same absolute saving, ~1% of 108 us
- `bs=16 *` faster: same, ~0.7-1% of 148 us
- `bs=32 *` flat: untouched
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Mechanism parity with the running cuda-dispatch-combine-pdl-verify (PDL on its CUDA combine at >32), here on the Triton combine at <32: shape-disjoint. Marlin never triggers, so the implicit trigger at its block exits applies; the win is the launch/prologue overlap of the last w2 wave only. Must read the PATH CHOSEN line before judging: pdl off means the run measured master. Risk medium: Triton version on the judge (torch 2.14.1) should have launch_pdl and tl.extra.cuda.gdc_wait, but if not the fallback keeps it a master-equivalent run, not a failure. Kernel time in the profile will not move; judge on the shape median vs master's bs=1/8/16 rows. Overlaps post-norm-combine-hoisted-slot-loads (same kernel, different mechanism); keep them as separate runs for attribution, then restack. Replaced at bs=1 by the running bs1 fused rewrite if it lands.
