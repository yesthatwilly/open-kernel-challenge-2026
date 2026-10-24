---
problem: muse_attention_decode
slug: splitkv-pdl-attn-to-combine
created: 2026-10-24T183855Z
base: master
area: launch overhead
change: >
  Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B
  (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv_combine (host), launch
  kernel B with cudaLaunchKernelEx: cudaLaunchConfig_t with gridDim (B, 16), blockDim 128, stream
  = current stream, and one attribute cudaLaunchAttributeProgrammaticStreamSerialization with
  programmaticStreamSerializationAllowed = 1. (2) First statement of splitkv_combine_kernel: asm
  volatile("griddepcontrol.wait;" ::: "memory"); this blocks until kernel A has completed and its
  writes are visible, so the combine's reads are as safe as today. (3) In splitkv_attn_kernel,
  after the main tile loop and before the final cp_wait<0>(), every thread executes asm
  volatile("griddepcontrol.launch_dependents;" :::); a CTA that exits early (bx >= total) triggers
  implicitly by exiting. Nothing else changes; CUDA graph capture records the programmatic edge
  (CUDA >= 12.3; the judge image is cu130). Rename the module muse_full_decode_splitkv_v2. If
  cudaLaunchKernelEx returns an error at the self-check, fall back to the plain <<<>>> launch and
  print '[splitkv] PDL off'; print '[splitkv] PDL on' otherwise.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: "the combine's launch and block ramp overlap kernel A's tail instead of following it; the shape-minus-kernel-sum gap is ~1-2 us per graph node, so expect -1 to -2 us (~1%) on each shape, readable in the per-shape median, not the headline"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: outside the split-KV path
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T184111Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T184125Z
base_sha: f6e63f1cb26d879530432b8f29b2946a17411828
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/splitkv-pdl-attn-to-combine
worktree: .worktrees/muse_attention_decode/splitkv-pdl-attn-to-combine
mode: null
---

# splitkv-pdl-attn-to-combine

## Change

Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv_combine (host), launch kernel B with cudaLaunchKernelEx: cudaLaunchConfig_t with gridDim (B, 16), blockDim 128, stream = current stream, and one attribute cudaLaunchAttributeProgrammaticStreamSerialization with programmaticStreamSerializationAllowed = 1. (2) First statement of splitkv_combine_kernel: asm volatile("griddepcontrol.wait;" ::: "memory"); this blocks until kernel A has completed and its writes are visible, so the combine's reads are as safe as today. (3) In splitkv_attn_kernel, after the main tile loop and before the final cp_wait<0>(), every thread executes asm volatile("griddepcontrol.launch_dependents;" :::); a CTA that exits early (bx >= total) triggers implicitly by exiting. Nothing else changes; CUDA graph capture records the programmatic edge (CUDA >= 12.3; the judge image is cu130). Rename the module muse_full_decode_splitkv_v2. If cudaLaunchKernelEx returns an error at the self-check, fall back to the plain <<<>>> launch and print '[splitkv] PDL off'; print '[splitkv] PDL on' otherwise.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: the combine's launch and block ramp overlap kernel A's tail instead of following it; the shape-minus-kernel-sum gap is ~1-2 us per graph node, so expect -1 to -2 us (~1%) on each shape, readable in the per-shape median, not the headline
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: outside the split-KV path

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). Below headline noise by design; judge on the full decode_16/32 medians against the paired SGLang ratio.

## Adversarial review

Reviewed 2026-10-24T184111Z by an adversarial subagent (model opus).

**Verdict: proceed.** PDL from splitkv_attn_kernel to splitkv_combine_kernel is rule-clean and needs no companion, but the most it can save is B's launch latency (sub-µs to ~1 µs on 122/242 µs shapes, about 0.2% headline), well under the per-shape drift, so it is effectively unreadable unless the run measures itself, for example with an import-time graph A/B printed to the log.

### Justification
Mechanism is real but tiny here. Kernel A is a persistent grid (2 CTAs/SM, 256 threads, launch_bounds(256,2) so up to 128 regs/thread = the whole register file at 2 CTAs, ~91 KB smem each), and the proposed trigger sits at the very end of each CTA's tile loop, so B can launch only once the last A CTA has nearly finished. That makes the saving B's launch/first-CTA ramp, not any real overlap of work. After griddepcontrol.wait, B still waits for A's completion and memory flush. Profile evidence for a '1-2 µs per graph node' gap is weak. At full decode_16 the shape median is 122.8 µs against a kernel sum of 111.8+8.2+2.1 = 122.1 µs, so about 0.7 µs covers BOTH inter-kernel gaps (pre_attn->A and A->B). At full decode_32 the kernel sum (256 µs) exceeds the shape median (241.9 µs), so the gap is not measurable there at all. A realistic expectation is -0.3 to -1 µs per shape (<=0.8% at decode_16, <=0.4% at decode_32). Weighted 16 against the sliding shape's 46, that is about 0.1-0.2% of each decode load's time and less on the geometric mean. That is far inside the ±1.5% CI and the hypothesis's own quoted 6% run-to-run drift on full decode_32 medians. The expected directions (full decode_16/32 faster, everything else flat) are consistent with the gate. WEAKEST LINK: the readout plan. 'Judge on the splitkv kernel rows in top_kernels' will mislead. With PDL, the combine kernel's recorded start moves earlier and its duration then includes time blocked in griddepcontrol.wait, so the splitkv_combine_kernel row can get LONGER even when end-to-end time improves. Per-shape medians cannot resolve under 1 µs either. Without an in-run measurement, any result will be read as noise.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: defines both kernels. Its persistent, register-saturating 2-CTA/SM design is what limits PDL to hiding launch latency. If the grid were not persistent, or A's CTAs had uneven tails, the overlap would differ, so the result says nothing about PDL in general, only about this kernel pair.
- sliding-deep-splits-on-cuda-splitkv: does not touch the full path, yet FINDINGS credits it with a full decode_32 'gain'. That is direct evidence that full decode_32 moves by more than this change's whole expected effect between runs with identical full-path code. It is a noise confounder, not an interaction.
- fused-preattention-triton: its Triton kernel is the predecessor of A and is launched normally, so it completes before A starts and before any early B. No interaction with the A->B edge, but its own pre_attn->A gap stays in the shape time, and this change does not address it.

### Interference and A/B needs
No companion. No landed hypothesis meaningfully interacts with the A->B launch edge, so an ab-revert or ab-tune of a landed patch would not disambiguate anything. A control (same code with programmaticStreamSerializationAllowed=0) would only remeasure the run-to-run noise, which has the same 6% drift and so cannot settle a ~0.5% effect. The right fix is inside this run rather than a second run. At import, after the self-check, capture A+B on synthetic batch-16 and batch-32 data (80K lengths) into two CUDA graphs, one with PDL and one with the plain launch. Replay each about 200 times, alternating, with torch.cuda.Event timing, and print '[splitkv] pdl A/B b16: X us vs Y us; b32: ...'. That is a paired, same-GPU, in-process A/B at sub-µs resolution, readable from the log_tail, and it stays far inside the 15-minute budget. Judge the hypothesis on that line plus the full decode_16/32 medians. If the import-time delta is under 0.3 µs, call it neutral regardless of the headline. Also note full-cuda-onelaunch-warpspec-bulk-gather (running) replaces A and B outright. If it lands first, this idea is moot as stated and should be re-based or dropped.

### Open questions
- How many registers does splitkv_attn_kernel actually use (ptxas -v)? If fewer than about 120, B's 128-thread CTAs can sit on SMs beside live A CTAs (smem leaves about 45 KB free), and B launches as soon as the last A CTA triggers. Otherwise B lands only on SMs A has vacated. Either way the saving is launch latency only.
- Should the trigger move earlier, for example at the start of each CTA's LAST item rather than after the loop? It is still safe, because B's wait covers full completion. It does not change much because A is persistent, but it costs nothing.
- B could load its A-independent inputs (seqlens[b], v_scale, the gate values from qkv) BEFORE griddepcontrol.wait to overlap them with A's tail. These are step inputs not written by A, so this is safe. Is that within scope, or a separate hypothesis?
- Does the same PDL treatment on the pre_attention(Triton)->A edge belong in this or a sibling hypothesis? Triton cannot trigger launch_dependents, so only A's side (launch with the attribute plus griddepcontrol.wait at A's entry) is possible, and A's seq_s/pre_s prologue reads only inputs, so it could run before the wait.

