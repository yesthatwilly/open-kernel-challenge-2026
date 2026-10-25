---
problem: muse_moe_w4a16
slug: triton-small-m-threshold-32
created: 2026-10-23T225502Z
base: master
area: MoE align/dispatch
change: >
  Raise the small-M dispatch threshold in kernel_fn from num_tokens <= 8 to num_tokens <= 32 so
  decode_16 and decode_32 also take the Triton grouped-GEMV path for both GEMMs. The only kernel
  change is BLOCK_M as a constexpr chosen from the shape: 16 for M <= 16, 32 for M <= 32 (cnt[e]
  <= M still fits one m-tile, so no m-loop), and compact_routes loads M*8 <= 256 ids per program.
  Everything else (prologue, dequant, epilogue, buffers) is the landed small-M code; bs=128/256
  stay on Marlin. Attribution: decode_16 and decode_32 move; bs=1/8 are unchanged code and
  bs=128/256 unchanged path.
expected:
  - shapes: bs=16 *
    direction: faster
    why: SGLang is 189 us vs a 52.5 us floor at bs=16 (~9 touched experts); the grouped GEMV reads each touched expert once and skips the rest, with 5 launches instead of ~9
  - shapes: bs=32 *
    direction: faster
    why: 207 us vs 70.8 us floor at bs=32; BLOCK_M=32 keeps one m-tile per expert
  - shapes: bs=1 *
    direction: flat
    why: "control: same code and constexpr as before"
  - shapes: bs=128 *
    direction: flat
    why: "control: Marlin path untouched"
risk: medium
depends_on: triton-w2-grouped-gemv-small-m
priority: 4
kind: primary
controls_for: null
review: required
status: done
hold_reason: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)
outcome: expired
verdict_detail: "hold expired after 31 h: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)"
expired_at: 2026-10-25T211329Z
---

# triton-small-m-threshold-32

## Change

Raise the small-M dispatch threshold in kernel_fn from num_tokens <= 8 to num_tokens <= 32 so decode_16 and decode_32 also take the Triton grouped-GEMV path for both GEMMs. The only kernel change is BLOCK_M as a constexpr chosen from the shape: 16 for M <= 16, 32 for M <= 32 (cnt[e] <= M still fits one m-tile, so no m-loop), and compact_routes loads M*8 <= 256 ids per program. Everything else (prologue, dequant, epilogue, buffers) is the landed small-M code; bs=128/256 stay on Marlin. Attribution: decode_16 and decode_32 move; bs=1/8 are unchanged code and bs=128/256 unchanged path.

## Expected

- `bs=16 *` faster: SGLang is 189 us vs a 52.5 us floor at bs=16 (~9 touched experts); the grouped GEMV reads each touched expert once and skips the rest, with 5 launches instead of ~9
- `bs=32 *` faster: 207 us vs 70.8 us floor at bs=32; BLOCK_M=32 keeps one m-tile per expert
- `bs=1 *` flat: control: same code and constexpr as before
- `bs=128 *` flat: control: Marlin path untouched

## Risk

Risks: at M=32 the per-program x gather is 32 x BLOCK_K bf16 per K step and the tl.dot is 32 x BLOCK_N x BLOCK_K, so the kernel may become compute/latency-bound rather than bandwidth-bound; if decode_32 comes out slower than Marlin, re-queue with threshold 16 (an ab-tune the reviewer may add). Nearby shape bs=33 then crosses to Marlin: both sides must be correct (bs 33 is a pre-timing test).
