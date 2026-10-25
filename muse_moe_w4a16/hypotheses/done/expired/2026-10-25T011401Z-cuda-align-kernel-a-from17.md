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
review: hold
status: done
review_at: 2026-10-25T011915Z
review_companions: null
hold_reason: "adversarial review: Hold. Kernel A's rows at bs=128/256 have already been measured three times, and a run cannot see a ~0.3% headline change. In the one new range (17..32), the com"
outcome: expired
verdict_detail: "retired: stale hold closed per the 2026-10-25 moe diagnosis"
expired_at: 2026-10-25T212024Z
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

## Adversarial review

Reviewed 2026-10-25T011915Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. Kernel A's rows at bs=128/256 have already been measured three times, and a run cannot see a ~0.3% headline change. In the one new range (17..32), the comparator that matters is _align_parallel at 2.6 us, not _align_small at 4.2 us, and kernel A is unlikely to beat it there.

### Justification
The mechanism is real and has already been measured. Kernel A is the (anonymous namespace)::cd... row in three runs: cuda-dispatch-combine-pdl-verify, stack-onewarp-hoisted-dispatchcombine and its -from17 variant. It read 3.2/3.2/3.2 us at bs=128 against _align_parallel's 4.0-4.1, and 3.9/3.8/4.4 us at bs=256 against 6.8-6.9. Re-running it on bs=128/256 can only reproduce those rows. The headline arithmetic: -0.7/318 us at verify_16 and -3/454 us at verify_32, plus perhaps -1/200 us at decode_32, comes to about 0.3% geomean over 6 equal-weight loads. That is far inside the +/-1.5% CI and inside the 1-3 us per-shape err_ns at those shapes. Both moe stacks that already carried kernel A at bs>=33 came back neutral and below best (1.318 and 1.324 vs 1.3287). The only unmeasured part is 17..32 (decode_32). WEAKEST LINK: the brief compares kernel A at bs=32 with _align_small (4.2 us), but align-parallel-from-17-tokens and stack-...-from17 both measured _align_parallel at bs=32 at 2.6 us. REWRITE_READ attributes kernel A's time to the single 1024-thread CTA's scan plus launch latency, about 2x its floor and roughly flat in T: 3.2 us at T=128 and 3.9 at T=256. At T=32 it therefore lands near 3 us, which beats master's _align_small but probably loses to the already-measured _align_parallel. So the 17..32 half picks the second-best known engine for decode_32. The decode_32 per-shape err (2.6-3.4 us) is also larger than the whole effect. The 'expected' directions are correct against master, but the result settles nothing that the existing rows do not.

### Confounders on the current kernel
- align-parallel-per-expert-verify: kernel A replaces its _align_parallel at 33..512. The patch was built on this same base, so it applies cleanly. The A/B is the align row, and that row is already known (4.0/6.9 -> 3.2/3.9 us).
- triton-map-align-on-block-m-restack: kernel A replaces its _align_small at 17..32. align-parallel-from-17-tokens is the unpromoted but measured better alternative there (2.6 us). The new range's verdict depends on which comparator is used.
- marlin-block-m-two-smaller-on-postnorm: kernel A takes block_size_m as an input (16 at bs=128, 32 at bs=256, SGLang's 16 at bs=32). Its self-check must use master's BSM per T, as the brief says. Otherwise the n_padded and block layout differ and Marlin's row moves for reasons unrelated to align.

### Interference and A/B needs
No companion would rescue interpretability. The hypothesis is judged on its own align row, and every companion (revert align-parallel, tune the 17 bound) would also be sub-resolution at the headline and already measured at the row level. The decisive comparison at bs=32 (kernel A vs _align_parallel) needs no extra run beyond a prepare-time CUDA-event timing of both engines at T=17/24/32, printed in the log. LESSONS warns that prepare timings undercount by 20-25%, but they rank variants correctly. That belongs inside a larger hypothesis, not a standalone check.

### Open questions
- If the coordinator still wants kernel A on master, the better spec routes each shape to its measured-fastest align: _align_parallel at 17..32 (per align-parallel-from-17-tokens) and kernel A at 33..512. Each gets its own constant, and the result is folded into a larger bs>=17 change rather than spent as its own check.
- Does kernel A at T=32 actually come in under 2.6 us? A prepare-time event timing of kernel A, _align_small and _align_parallel at T in {17, 24, 32} answers this for free in any run that builds kernel A.
- Without PDL, does kernel A keep its 3.2/3.9 us rows? Those rows were measured with PDL on, and the launch-latency share is exactly what PDL hides.

