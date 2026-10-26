---
problem: muse_bf16_proj
slug: qkv-swapab16-five-stage-ring-m8-m16
created: 2026-10-26T221149Z
base: master
area: qkv GEMM (m<=16, swap-AB Triton)
change: >
  Built on master blob 2bcfe7d4. qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99
  us in 3834ceec) runs _bf16_swapab_kernel through _swapab16_forward with SWAPAB16_CFG qkv_gate =
  (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages): 144 CTAs, 48 regs, ~80 KB smem, 2 CTAs/SM, so
  only ~3 x 16 KB of weight is in flight per CTA, while the pooler's swapab16 path keeps ~3 x 32
  KB in flight and streams ~3.4 TB/s against qkv's ~2.9 TB/s (row basis). Give qkv a 5-stage ring
  at the same split, tile and occupancy. In _swapab16_prepare, compile a num_stages=5 qkv_gate
  variant on synthetic x, print its n_regs and compiled metadata.shared, and keep it only if
  shared <= 113 KB (2 CTAs/SM still fit); then warm every m in 1..16 at 5 stages so the timed call
  never compiles. New flag qkv16_s5_ok (False by default, appended to NEW_CELL_FLAGS): kernel_fn
  passes num_stages=5 instead of state['swapab16_stages']['qkv_gate'] when the flag is on and
  key16 == 'qkv_gate' (a cfg / flag-only choice; everything else in the call is identical). Gate
  with master's _cell_gate on that flag at m=8 and at m=16 (cand = 5 stages, master = 4 stages,
  same workspace and tickets), using a judge-like read flush because the 37.7 MB weight fits in
  L2: add flush_mode='read' to _cell_gate (flush.view(torch.int32).sum() in place of
  flush.zero_(); default unchanged for every other caller). Two rounds per m; route only if all
  four master/cand ratios are >= 1.03. State B inherits the flag (and re-warms 5 stages); add
  AB_CASES (qkv_gate, 2304, 8192, 8) and (qkv_gate, 2304, 8192, 16), mapped to qkv16_s5_ok in
  _ab_component. One PATH CHOSEN line with the four ratios and the printed smem. m=17..32 (24 KB
  per stage, where a fifth stage would break 2 CTAs/SM), m=1 and every other layer stay untouched.
expected:
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: "~4 x 16 KB instead of ~3 x 16 KB of weight in flight per CTA at the same 2 CTAs/SM: 12.86 -> ~12.0 us if the stream is in-flight-bound"
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: "same kernel and config as m=8: 12.99 -> ~12.1 us"
  - shapes: layer=qkv_gate load=decode_32 m=32
    direction: flat
    why: m=17..32 swap-AB config untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "qkv-swapab-m32-splitk8-stages3: that run doubled the split (4 -> 8) and cut the ring to 3 stages at m=17..32 and lost in the reduce tail; this keeps split-K 4, the tile and 2 CTAs/SM and only deepens the m<=16 ring from 4 to 5 stages (+33% weight bytes in flight), gated per m"
review: required
status: queued
---

# qkv-swapab16-five-stage-ring-m8-m16

## Change

Built on master blob 2bcfe7d4. qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99 us in 3834ceec) runs _bf16_swapab_kernel through _swapab16_forward with SWAPAB16_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages): 144 CTAs, 48 regs, ~80 KB smem, 2 CTAs/SM, so only ~3 x 16 KB of weight is in flight per CTA, while the pooler's swapab16 path keeps ~3 x 32 KB in flight and streams ~3.4 TB/s against qkv's ~2.9 TB/s (row basis). Give qkv a 5-stage ring at the same split, tile and occupancy. In _swapab16_prepare, compile a num_stages=5 qkv_gate variant on synthetic x, print its n_regs and compiled metadata.shared, and keep it only if shared <= 113 KB (2 CTAs/SM still fit); then warm every m in 1..16 at 5 stages so the timed call never compiles. New flag qkv16_s5_ok (False by default, appended to NEW_CELL_FLAGS): kernel_fn passes num_stages=5 instead of state['swapab16_stages']['qkv_gate'] when the flag is on and key16 == 'qkv_gate' (a cfg / flag-only choice; everything else in the call is identical). Gate with master's _cell_gate on that flag at m=8 and at m=16 (cand = 5 stages, master = 4 stages, same workspace and tickets), using a judge-like read flush because the 37.7 MB weight fits in L2: add flush_mode='read' to _cell_gate (flush.view(torch.int32).sum() in place of flush.zero_(); default unchanged for every other caller). Two rounds per m; route only if all four master/cand ratios are >= 1.03. State B inherits the flag (and re-warms 5 stages); add AB_CASES (qkv_gate, 2304, 8192, 8) and (qkv_gate, 2304, 8192, 16), mapped to qkv16_s5_ok in _ab_component. One PATH CHOSEN line with the four ratios and the printed smem. m=17..32 (24 KB per stage, where a fifth stage would break 2 CTAs/SM), m=1 and every other layer stay untouched.

## Expected

- `layer=qkv_gate load=verify_1 m=8` faster: ~4 x 16 KB instead of ~3 x 16 KB of weight in flight per CTA at the same 2 CTAs/SM: 12.86 -> ~12.0 us if the stream is in-flight-bound
- `layer=qkv_gate load=decode_16 m=16` faster: same kernel and config as m=8: 12.99 -> ~12.1 us
- `layer=qkv_gate load=decode_32 m=32` flat: m=17..32 swap-AB config untouched

## Risk

Low: one Triton launch parameter behind a flag and a two-m, two-round gate; no new module. If the m <= 16 qkv stream is DRAM-bound rather than bound by bytes in flight, the gate keeps 4 stages and the check answers that. The Do-not-try entries on qkv swap-AB concern deeper split-K or two CTAs per SM; this changes neither (split-K 4 and 2 CTAs/SM as today). Sizing: up to -0.8 us on verify_1 (0.34 %/us) and decode_16 (0.27 %/us) is +0.2-0.5%, below the 1.24% one-run margin alone. Decode-portable: the same knob applies to decode's qkv at T=8/16.
