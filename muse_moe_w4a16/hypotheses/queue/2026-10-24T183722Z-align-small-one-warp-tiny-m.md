---
problem: muse_moe_w4a16
slug: align-small-one-warp-tiny-m
created: 2026-10-24T183722Z
base: master
area: MoE align/dispatch
change: >
  num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1/bs=4 tests); bs=16 and above keep
  master's launches exactly. In _align_small (host wrapper), replace the fixed num_warps=4 of the
  _align_small_kernel launch with num_warps = 1 if n = num_tokens*8 <= 64 else 4. Nothing in the
  kernel changes. The prepare() warm-up loop already calls _align_small for every tokens 1..32, so
  it compiles the 1-warp variants for 1..8 automatically; _align_self_check runs through the same
  wrapper and so checks the 1-warp variants too. Mechanism: at n <= 64 every tensor in the kernel
  is tiny (ids [<=64], one-hot [<=64, 32], CAP_BLOCK <= 512, MB_BLOCK <= 64), so 4 warps leave
  most threads idle while every tl.sum / tl.cumsum across the 4 warps goes through shared memory
  and a bar.sync; one warp does all of them with shuffles only, so the kernel's serial chain of ~8
  reductions/scans gets shorter. One-warp register use: the [64, 32] one-hot and its cumsum are 64
  int32 per thread per temporary, under the 255 limit (print n_regs/n_spills of the 1-warp bs=8
  variant in prepare via the compiled kernel's .n_regs/.n_spills).
expected:
  - shapes: bs=1 *
    direction: faster
    why: _align_small_kernel 1.4 us -> ~1.0-1.2 us, ~0.5% of 49 us
  - shapes: bs=8 *
    direction: faster
    why: 2.7 us -> ~1.8-2.2 us, ~0.5-0.8% of 108 us
  - shapes: bs=16 *
    direction: flat
    why: untouched
  - shapes: bs=32 *
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
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# align-small-one-warp-tiny-m

## Change

num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1/bs=4 tests); bs=16 and above keep master's launches exactly. In _align_small (host wrapper), replace the fixed num_warps=4 of the _align_small_kernel launch with num_warps = 1 if n = num_tokens*8 <= 64 else 4. Nothing in the kernel changes. The prepare() warm-up loop already calls _align_small for every tokens 1..32, so it compiles the 1-warp variants for 1..8 automatically; _align_self_check runs through the same wrapper and so checks the 1-warp variants too. Mechanism: at n <= 64 every tensor in the kernel is tiny (ids [<=64], one-hot [<=64, 32], CAP_BLOCK <= 512, MB_BLOCK <= 64), so 4 warps leave most threads idle while every tl.sum / tl.cumsum across the 4 warps goes through shared memory and a bar.sync; one warp does all of them with shuffles only, so the kernel's serial chain of ~8 reductions/scans gets shorter. One-warp register use: the [64, 32] one-hot and its cumsum are 64 int32 per thread per temporary, under the 255 limit (print n_regs/n_spills of the 1-warp bs=8 variant in prepare via the compiled kernel's .n_regs/.n_spills).

## Expected

- `bs=1 *` faster: _align_small_kernel 1.4 us -> ~1.0-1.2 us, ~0.5% of 49 us
- `bs=8 *` faster: 2.7 us -> ~1.8-2.2 us, ~0.5-0.8% of 108 us
- `bs=16 *` flat: untouched
- `bs=32 *` flat: untouched
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Low-gain, low-risk launch-latency tune; the headline cannot resolve it, the _align_small_kernel rows at bs=1/8 can (Triton rows reproduce within ~1%). If 1 warp spills at bs=8 (n_spills > 0 in the log) the expected direction there flips to slower; that is the main risk. Overlap: the running bs1 fused rewrite replaces this kernel at bs=1 and the queued midm rewrite at 2..32; the running swap-AB keeps _align_small but at a larger BSM (8*NT), which does not change the variant's size class at bs<=8 (BSM 8).
