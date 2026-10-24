---
problem: muse_moe_w4a16
slug: w13-marlin-bf16-reduce-verify
created: 2026-10-23T225502Z
base: master
area: w13 marlin
change: >
  For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13
  moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all decode shapes keep
  use_fp32_reduce=True). With fp32 reduce, Marlin writes each k-slice's partial C tile to an fp32
  scratch and the last slice reads them back: at bs=256 the w13 output is M*8 x 2I = 2048 x 8192,
  so each slice pass moves 67 MB of fp32 scratch (~14 us at HBM speed per pass) on a 620 us shape;
  without it partials are combined in bf16 in the output tile. Nothing else changes.
expected:
  - shapes: bs=256 *
    direction: faster
    why: 67 MB fp32 scratch per k-slice pass removed from a 620 us shape; small (2-4%) but above the +/-1.5% CI if the kernel is traffic-bound there
  - shapes: bs=128 *
    direction: faster
    why: "bs=128: 33 MB per pass on a 467 us shape; smaller effect"
  - shapes: bs=32 *
    direction: flat
    why: "control: decode shapes keep fp32 reduce"
risk: medium
depends_on: sglang-marlin-path-parity
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T040540Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T040540Z
base_sha: 58b624e7f1438bb67b466c3fa63d9278bcff3406
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/w13-marlin-bf16-reduce-verify
worktree: .worktrees/muse_moe_w4a16/w13-marlin-bf16-reduce-verify
mode: null
---

# w13-marlin-bf16-reduce-verify

## Change

For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13 moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all decode shapes keep use_fp32_reduce=True). With fp32 reduce, Marlin writes each k-slice's partial C tile to an fp32 scratch and the last slice reads them back: at bs=256 the w13 output is M*8 x 2I = 2048 x 8192, so each slice pass moves 67 MB of fp32 scratch (~14 us at HBM speed per pass) on a 620 us shape; without it partials are combined in bf16 in the output tile. Nothing else changes.

## Expected

- `bs=256 *` faster: 67 MB fp32 scratch per k-slice pass removed from a 620 us shape; small (2-4%) but above the +/-1.5% CI if the kernel is traffic-bound there
- `bs=128 *` faster: bs=128: 33 MB per pass on a 467 us shape; smaller effect
- `bs=32 *` flat: control: decode shapes keep fp32 reduce

## Risk

Risks: bf16 combination of k-slice partials adds ~2^-8 relative error per slice (2-4 slices): well inside the 1.5% rel-L2 bound, but it is a precision knob, so the attempt file must show the per-row margin from the run; it does not lower activation precision below bf16, which SGLang keeps. The expected gain is small; priority 5. If flat, retire: Marlin is then not traffic-bound at bs=256 and the verify gap (467/620 us vs 101/109 us floors) needs a different kernel, not a knob.

## Adversarial review

Reviewed 2026-10-24T040540Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A one-bool knob on master's w13 Marlin call for bs>=128 only; a single run is attributable in every direction, but the mechanism is over-stated (bf16 reduce still round-trips partials through global memory, and only tiles whose column straddles a CTA boundary reduce at all), so the most likely outcome is flat inside the +/-1.5% CI. Proceed, with the retirement conclusion narrowed.

### Justification
Plausibility: moe_wna16_marlin_gemm takes use_fp32_reduce as a runtime bool on the same Marlin template instantiation (the profile shows one Marlin<__nv_bfloat16, 56...> symbol, 2 launches per shape), so nothing recompiles and the only code-path difference is inside the k-parallel global reduce. Three things weaken the stated mechanism. (1) The reduce only runs for tiles with slice_count > 1, i.e. (m-block, n-tile) columns split between two CTAs. Marlin partitions work as iters = ceil(k_tiles * n_tiles * m_blocks / gridDim); at bs=256 (block 64, ~40 m-blocks, N=8192, K=4096) the work units number in the tens of thousands against a grid of ~528 CTAs (132 SMs x max_blocks_per_sm=4), so iters >> k_tiles and only the column at each CTA boundary is split: a minority of tiles, not 'each k-slice pass' over the whole 2048x8192 output. (2) With use_fp32_reduce=False and use_atomic_add=False the kernel does not 'combine partials in the output tile' for free: it takes the bf16 global-reduce path, which still writes the partial tile to C in global memory and reads it back under the same lock serialization, so bytes halve rather than vanish. (3) c_tmp is sized ~sms*4*block_m*max_thread_n fp32 (~8-9 MB), written and re-read within microseconds, so much of that traffic is L2-resident on H200 anyway; the 67 MB / 14 us figure is an upper bound on HBM traffic. Profile support: at bs=256 Marlin is 611.8 us of 621 us (two launches, w13 roughly 2/3 of it by N) against a 108.9 us floor, and at bs=128 495 us vs 101 us; the kernel is 5-6x off the bandwidth floor, so it is not traffic-bound, which the hypothesis itself concedes ('if the kernel is traffic-bound there'). The strongest counter-evidence is marlin-block-size-m-next-larger: at bs=128 block 48->64 added ~20% padded rows and cost 18% (462->545 us) with bs=256 unchanged as a control; time scales with padded rows, i.e. per-row MMA/dequant work, so a few microseconds of hidden scratch traffic will mostly be absorbed. Realistic ceiling: ~0.5-1% at bs=256, less at bs=128, inside the CI. Expected directions and the equal 60-weights are consistent with TASK.md (each load counts equally; the verify_16/32 loads are where the absolute microseconds are). Weakest link: the assumption that the fp32 scratch write/read is on the critical path and disappears, when it (a) only affects boundary-split columns, (b) halves rather than disappears, and (c) is dominated by compute-shaped time per the block_size_m regression.

### Confounders on the current kernel
- sglang-marlin-path-parity (the only landed item, and this hypothesis's base): it introduced use_fp32_reduce=True as SGLang parity, so both sides of the judge's A/B (SGLang's kernel and ours) reduce in fp32 today; flipping only our w13 call is a clean delta. Its one structural deviation (moe_align_block_size over 32 experts) shrinks the sorted_ids buffer but not num_tokens_post_padded, so Marlin's m-block count, iteration split and hence the set of tiles that reduce are identical to SGLang's. No confound.

### Interference and A/B needs
No companion is warranted. A gain can only come from the one changed bool (same template, same grid, same align, same workspace; the skipped c_tmp torch::empty inside the op is captured by the CUDA graph and costs nothing at replay), so a gain is attributable. A loss would mean the bf16 global-reduce path is slower than the fp32 one (plausible: vLLM/SGLang made fp32 reduce the default for accuracy at near-zero cost), also attributable. Flat is the likely outcome and is interpretable as 'the knob is worth less than the CI on master', but NOT as 'Marlin is not traffic-bound at bs=256': flat is equally consistent with few split columns at this grid size. No cheap run separates those two readings without kernel-side instrumentation, so none is proposed. Cross-hypothesis note: in-flight marlin-workspace-blocks-per-sm-8 doubles the w13 grid at exactly these shapes, which shrinks iters and raises the number of boundary-split columns and therefore the reduce cost; if it lands, this knob's effect grows and should be re-measured on top of it rather than inferred from this run. The queued w13-marlin-atomic-add-small-m covers bs<=32 with the same bool plus atomic add: disjoint shapes, not a duplicate, and both should keep use_atomic_add=False here (atomic add into a bf16 output would require a zeroed cache1).

### Open questions
- Does sgl_kernel's MoE moe_wna16_marlin_gemm actually implement the non-fp32, non-atomic (bf16 global-reduce) path, or does the binding assert use_fp32_reduce || use_atomic_add? If it raises, record as rejected, not failed. (No sglang/sgl_kernel source is checked out on this node; the mechanism above is from the vLLM/SGLang Marlin template as remembered.)
- What fraction of w13 tiles at bs=128/256 have slice_count > 1 under the max_blocks_per_sm=4 grid? If it is small, the knob cannot move the needle regardless of traffic-boundness; the retirement text should say 'knob below CI on master', not 'Marlin not traffic-bound'.
- If marlin-workspace-blocks-per-sm-8 lands (more CTAs, more split columns), re-run this knob on top of it; its value is grid-dependent.
- The parity profile cannot split the 611.8 us Marlin total between w13 and w2 (one symbol, 2 calls); the hypothesis's 'w13 only' scoping leaves w2's reduce (output 2048x4096, half the bytes) untouched, which is fine as scoping but the expected effect should be sized against ~2/3 of the Marlin time, not all of it.

