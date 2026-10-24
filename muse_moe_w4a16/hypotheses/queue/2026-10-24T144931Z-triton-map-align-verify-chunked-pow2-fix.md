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
review: proceed
status: queued
review_at: 2026-10-24T145123Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T145123Z by an adversarial subagent (model opus).

**Verdict: proceed.** The fix is correct and minimal. Diffing the branch file against chunked_fixed.py shows exactly 4 changed lines: the BSM_P2 constexpr param, tl.arange(0, BSM_P2), the [NUM_E, BSM_P2] sentinel tile and the BSM_P2=triton.next_power_of_2(block_size_m) launch arg. The mask j < padded - counts (<= BSM-1) keeps the stored slots identical, and no other non-power-of-2 tl.arange or block shape remains in the kernel. This is a re-run of a reviewed (proceed) hypothesis whose only failure was this compile error, so no companions are needed.

### Justification
The mechanism is the same as the original reviewed hypothesis. At bs=128/256, one Triton launch replaces the expert_map gather (index_elementwise ~3.2-3.3 us), the align wrapper's fill (unrolled_elementwise ~2.8 us), moe_align_block_size_kernel (~2.7-2.8 us) and count_and_sort (1.8-2.6 us): ~10.5 / ~11.5 us out of ~333 / ~470 us. The realistic per-shape gain is 1.5-2%, ~0.5% on the headline, inside one run's CI, so read the result from top_kernels rows and not from the headline. bs=1/8/16/32 run byte-identical _align_small code and are a valid drift control. The fix itself: archive log 3aa4d7e193244255.log confirms the failure was 'arange's range must be a power of 2' at 'j = tl.arange(0, BSM)' during prepare's warm-up of BSM=48. The original's tl.zeros([NUM_E, BSM]) would have failed for the same reason, and the fix covers it too. Pass 1 and pass 2 use tl.arange(0, CHUNK=512) and tl.arange(0, NUM_E=next_pow2(32)=32), both powers of 2. The tail-fill and block_experts loops reuse offs0 (CHUNK). block_start = blocks * BSM is scalar arithmetic with a non-pow2 constexpr, which is fine. The loop-carried counts/base are int32 (tl.zeros int32 / cumsum of int32, plus tl.sum of int32), so their types are stable across iterations. Every runtime loop bound (n, cap, n_blocks) is a host int in do_not_specialize. BSM=48 can only reach the chunked kernel from prepare warm-up and the self-check (kernel_fn's picks at the test shapes are 16 at bs=33, 16 at 128, 32 at 256/257), so the fix matters for compile and not for timed shapes. Weakest link (unchanged from the original review): the kernel's own serial time in one CTA at n=2048 (bs=256: 4 chunks x 2 passes with a [512,32] cumsum, 64 int32 per thread per temporary at 8 warps, possible spills). A plausible 8-14 us could cancel the bs=256 gain. A secondary unknown is that the compile was never reached past the first error, so a later compile or runtime issue cannot be fully ruled out without a GPU. A read of the rest of the kernel found none.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: it sets BSM=16 at bs=128 and 32 at bs=256/257, which fixes cap, the padding tile's live width and n_blocks, so it shifts the new kernel's own time a little. It does not change what Marlin receives, so it does not confound attribution of the removed align rows.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: same area and design, but gated at <=32 tokens. Those shapes are this run's in-run control, not a confounder. Note that _align_small_kernel still uses tl.arange(0, BSM) unpadded. It is safe today because _block_size_m at <=32 tokens never exceeds 16, but it is a latent compile hazard for queued triton-small-m-threshold-32 or marlin-two-class-block-m-decode32 if they route BSM=48 to _align_small.
- post-expert-norm-triton-small-m: no confound at the target shapes. bs>=32 still uses fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids, which needs remote slots to be exactly -1, and the chunked pass 1 reproduces that.
- sglang-marlin-path-parity: the base pipeline and the 32-expert capacity contract (n + 32*(bsm-1)). No interaction beyond the sizes the kernel must match.

### Interference and A/B needs
No companion is warranted. The change is directly attributable per shape: at bs=128/256 the four at::native/align rows should disappear from top_kernels and one _align_chunked_kernel row should appear. bs<=32 code is byte-identical and acts as the drift reference. An ab-revert of block-m-two-smaller would change only BSM and the padding-tile cost (well under 1 us) and cannot flip the interpretation. A CHUNK or num_warps ab-tune is a follow-up chosen from this run's kernel row (e.g. gate at n<=1024 if bs=256 loses), not a disambiguator. Pass criteria: the '[triton-map-align-verify-chunked] ... 0 other' self-check line, then a kernel row below about (sum of removed rows - 1 us) at each verify shape.

### Open questions
- _align_chunked_kernel's own time at bs=256 (n=2048). If it is above ~10 us, the bs=256 gain disappears. CHUNK=256 or computing the onehot only in pass 2 would be the next try.
- The compile has never got past this line on a GPU. Confirm in the log that all 5 BSM variants (8..64) compile and that the self-check prints '0 other' before trusting the timings.
- Should _align_small_kernel get the same BSM_P2 hardening now, before triton-small-m-threshold-32 or marlin-two-class-block-m-decode32 can send BSM=48 to it? Doing it in this run would break byte-identity at bs<=32, so it belongs in a separate change.
- The extended self-check (360 cases with Python per-slot loops) adds prepare time. It is expected to take seconds, well inside the 300 s prepare budget and the 15 min run budget, but watch for it.

