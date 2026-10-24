---
problem: muse_moe_w4a16
slug: align-parallel-from-17-tokens
created: 2026-10-24T183702Z
base: master
area: MoE align/dispatch
change: >
  One constant plus its warm-up: lower ALIGN_FUSED_MAX_TOKENS from 32 to 16, so 17 <= num_tokens
  <= 32 (decode_32 bs=32) runs master's existing 33-program _align_parallel_kernel instead of the
  single-program _align_small_kernel; num_tokens <= 16 (bs=1, 4, 8, 16) keeps _align_small and
  num_tokens > 32 keeps _align_parallel, both unchanged. block_size_m at 17..32 is SGLang's pick
  (16), and _align_parallel already compiles a BSM=16 variant in prepare() (bsm loop 16/32/48/64),
  so no new variant is needed; the _align_small warm-up loop range(1, ALIGN_FUSED_MAX_TOKENS + 1)
  shrinks to 1..16 automatically. _align_self_check needs no edit: it routes tokens <=
  ALIGN_FUSED_MAX_TOKENS to _align_small and the rest to _align_parallel, so T=32 is now checked
  against moe_align_block_size on the parallel kernel; its tokens > ALIGN_FUSED_MAX_TOKENS summary
  line will then also count T=32 cases (expected 0 mismatches apart from the known capacity-only
  lines; never gate on the self-check). Update the comment next to the constant. Mechanism: at
  bs=32 _align_small does a [256, 32] one-hot cumsum, a [64, 32] block scan and a 1024-wide tail
  store in ONE program on one SM (4.2-4.4 us, 3x its 1.4 us at bs=1); _align_parallel spreads the
  placement over 33 programs and at n=256 each does a single CHUNK=512 pass, so it should run at
  or below its 4.0 us at n=1024 (bs=128).
expected:
  - shapes: bs=32 *
    direction: faster
    why: align 4.2-4.4 us (_align_small_kernel) -> ~2.5-3 us (_align_parallel_kernel at one chunk), ~0.7% of 200 us; judge on the align row
  - shapes: bs=1 *
    direction: flat
    why: untouched
  - shapes: bs=8 *
    direction: flat
    why: untouched
  - shapes: bs=16 *
    direction: flat
    why: untouched (could be a follow-up if bs=32 gains)
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

# align-parallel-from-17-tokens

## Change

One constant plus its warm-up: lower ALIGN_FUSED_MAX_TOKENS from 32 to 16, so 17 <= num_tokens <= 32 (decode_32 bs=32) runs master's existing 33-program _align_parallel_kernel instead of the single-program _align_small_kernel; num_tokens <= 16 (bs=1, 4, 8, 16) keeps _align_small and num_tokens > 32 keeps _align_parallel, both unchanged. block_size_m at 17..32 is SGLang's pick (16), and _align_parallel already compiles a BSM=16 variant in prepare() (bsm loop 16/32/48/64), so no new variant is needed; the _align_small warm-up loop range(1, ALIGN_FUSED_MAX_TOKENS + 1) shrinks to 1..16 automatically. _align_self_check needs no edit: it routes tokens <= ALIGN_FUSED_MAX_TOKENS to _align_small and the rest to _align_parallel, so T=32 is now checked against moe_align_block_size on the parallel kernel; its tokens > ALIGN_FUSED_MAX_TOKENS summary line will then also count T=32 cases (expected 0 mismatches apart from the known capacity-only lines; never gate on the self-check). Update the comment next to the constant. Mechanism: at bs=32 _align_small does a [256, 32] one-hot cumsum, a [64, 32] block scan and a 1024-wide tail store in ONE program on one SM (4.2-4.4 us, 3x its 1.4 us at bs=1); _align_parallel spreads the placement over 33 programs and at n=256 each does a single CHUNK=512 pass, so it should run at or below its 4.0 us at n=1024 (bs=128).

## Expected

- `bs=32 *` faster: align 4.2-4.4 us (_align_small_kernel) -> ~2.5-3 us (_align_parallel_kernel at one chunk), ~0.7% of 200 us; judge on the align row
- `bs=1 *` flat: untouched
- `bs=8 *` flat: untouched
- `bs=16 *` flat: untouched (could be a follow-up if bs=32 gains)
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

A pure kernel-choice A/B at one shape: compare _align_parallel_kernel's bs=32 row with _align_small_kernel's 4.2-4.4 us in master runs. Headline effect is under noise and decode_32 Marlin rows are seed-noisy, so the kernel row is the evidence. If it wins, try 9..16 next. Overlap: the running cuda-dispatch-combine-pdl-verify replaces _align_parallel only at >32 (host branch on num_tokens > 32, independent of this constant? the implementer must keep that rewrite's gate untouched; here it is not present on master). The queued midm single-launch rewrite would replace the align at 2..32. The running swap-AB changes block_size_m at bs<=32 to 8*NT (32 at bs=32), so if it lands this would need a BSM=32 parallel variant (already compiled).
