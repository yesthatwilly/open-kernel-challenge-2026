---
problem: muse_moe_w4a16
slug: triton-map-align-verify-chunked-pow2-fix
created: 2026-10-24T144931Z
base: master
area: MoE align/dispatch
change: >
  Re-run triton-map-align-verify-chunked (failed at compile: tl.arange(0, BSM) with BSM=48 in
  prepare's warm-up raised 'arange's range must be a power of 2') with the one fix:
  _align_chunked_kernel takes a new constexpr BSM_P2 = triton.next_power_of_2(block_size_m) and
  builds the per-expert padding tile over tl.arange(0, BSM_P2) / [NUM_E, BSM_P2], keeping the
  existing mask j < padded - counts (always < BSM), so the stored slots are identical. Everything
  else is that branch's file byte for byte (chunked map+align for num_tokens > 32 replacing the
  expert_map gather and moe_align_block_size; the <=32 path is master's _align_small; self-check
  over 33/128/256/257 tokens).
expected:
  - shapes: bs=128 *
    direction: faster
    why: one launch replaces the 4 gather/align launches (~10.5 us)
  - shapes: bs=256 *
    direction: faster
    why: same, ~11.5 us replaced; kernel time may eat part of it
  - shapes: bs=1 *
    direction: flat
    why: master path
  - shapes: bs=32 *
    direction: flat
    why: master path
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# triton-map-align-verify-chunked-pow2-fix

## Change

Re-run triton-map-align-verify-chunked (failed at compile: tl.arange(0, BSM) with BSM=48 in prepare's warm-up raised 'arange's range must be a power of 2') with the one fix: _align_chunked_kernel takes a new constexpr BSM_P2 = triton.next_power_of_2(block_size_m) and builds the per-expert padding tile over tl.arange(0, BSM_P2) / [NUM_E, BSM_P2], keeping the existing mask j < padded - counts (always < BSM), so the stored slots are identical. Everything else is that branch's file byte for byte (chunked map+align for num_tokens > 32 replacing the expert_map gather and moe_align_block_size; the <=32 path is master's _align_small; self-check over 33/128/256/257 tokens).

## Expected

- `bs=128 *` faster: one launch replaces the 4 gather/align launches (~10.5 us)
- `bs=256 *` faster: same, ~11.5 us replaced; kernel time may eat part of it
- `bs=1 *` flat: master path
- `bs=32 *` flat: master path

## Risk

Minted by churnsubs to fix the compile error in a reviewed hypothesis; the fix is mechanical (power-of-2 tile + unchanged mask). The original review (proceed) is in the failed md.
