---
problem: muse_moe_w4a16
slug: cuda-align-kernel-a-from17
created: 2026-10-25T011401Z
base: master
area: MoE align/dispatch
change: >
  Port ONLY kernel A (the CUDA align: expert_map gather + moe_align_block_size work in one
  1024-thread CTA) from
  muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch onto master;
  kernel B (its post-expert combine) is NOT ported, since it was slower than SGLang's combine at
  bs=256 (6.2 vs 6.0 us profile, 4.75 vs 3.72 us prepare timing). Use kernel A for 17 <=
  num_tokens <= 512 under its OWN bound constants CUDA_ALIGN_MIN_TOKENS = 17 and
  CUDA_ALIGN_MAX_TOKENS = 512 (LESSONS 2026-10-25: never derive a new path's gate from
  ALIGN_FUSED_MAX_TOKENS), replacing _align_small at 17..32 (bs=32 decode_32) and _align_parallel
  at 33..512 (bs=128, 256 and the 33/257/skewed-128 tests), with master's block_size_m pick
  unchanged (it is an input to kernel A). Launch it plainly on the current stream (no PDL; its
  consumer is Marlin, which cannot gdc_wait). Build with load_inline in prepare (plain sm_90; it
  uses no sm_90a instructions), log the compile seconds. Self-check in prepare: sorted_ids,
  block_experts and n_padded byte-identical to master's align (_align_small / _align_parallel) at
  T in {17, 24, 32, 33, 57, 128, 256, 257, 512} x routings
  random/skewed/all-remote/all-one-expert/all-local x local_start {0, 32, 224} at master's BSM for
  each T; do not gate on the capacity-vs-SGLang count (LESSONS: always differs). On any failure or
  build error fall back to master and print '[cuda-align-a] PATH CHOSEN = CUDA|TRITON for 17..512
  (<reason>)'. Everything else is master byte for byte.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "kernel A measured 3.9 us vs _align_parallel 6.9 us (cuda-dispatch-combine-pdl-verify profile): about -3 us of 454 us; judge on the align row"
  - shapes: bs=128 *
    direction: faster
    why: "3.3 vs 4.0 us: about -0.7 us; align row"
  - shapes: bs=32 *
    direction: faster
    why: _align_small is 4.1-4.3 us at bs=32 (one Triton program over 256 routes); kernel A should be near its bs=128 3.3 us or below; align row
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# cuda-align-kernel-a-from17

## Change

Port ONLY kernel A (the CUDA align: expert_map gather + moe_align_block_size work in one 1024-thread CTA) from muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch onto master; kernel B (its post-expert combine) is NOT ported, since it was slower than SGLang's combine at bs=256 (6.2 vs 6.0 us profile, 4.75 vs 3.72 us prepare timing). Use kernel A for 17 <= num_tokens <= 512 under its OWN bound constants CUDA_ALIGN_MIN_TOKENS = 17 and CUDA_ALIGN_MAX_TOKENS = 512 (LESSONS 2026-10-25: never derive a new path's gate from ALIGN_FUSED_MAX_TOKENS), replacing _align_small at 17..32 (bs=32 decode_32) and _align_parallel at 33..512 (bs=128, 256 and the 33/257/skewed-128 tests), with master's block_size_m pick unchanged (it is an input to kernel A). Launch it plainly on the current stream (no PDL; its consumer is Marlin, which cannot gdc_wait). Build with load_inline in prepare (plain sm_90; it uses no sm_90a instructions), log the compile seconds. Self-check in prepare: sorted_ids, block_experts and n_padded byte-identical to master's align (_align_small / _align_parallel) at T in {17, 24, 32, 33, 57, 128, 256, 257, 512} x routings random/skewed/all-remote/all-one-expert/all-local x local_start {0, 32, 224} at master's BSM for each T; do not gate on the capacity-vs-SGLang count (LESSONS: always differs). On any failure or build error fall back to master and print '[cuda-align-a] PATH CHOSEN = CUDA|TRITON for 17..512 (<reason>)'. Everything else is master byte for byte.

## Expected

- `bs=256 *` faster: kernel A measured 3.9 us vs _align_parallel 6.9 us (cuda-dispatch-combine-pdl-verify profile): about -3 us of 454 us; judge on the align row
- `bs=128 *` faster: 3.3 vs 4.0 us: about -0.7 us; align row
- `bs=32 *` faster: _align_small is 4.1-4.3 us at bs=32 (one Triton program over 256 routes); kernel A should be near its bs=128 3.3 us or below; align row
- `bs=1 *` flat: control
- `bs=8 *` flat: control
- `bs=16 *` flat: control

## Risk

Headline effect is below the 1.5% resolution (about 0.3-1%); the result is read from the align kernel rows. Value: a free part for any later bs>=17 path, and settles whether the one-CTA CUDA scan beats the single-program Triton align at bs=32.
