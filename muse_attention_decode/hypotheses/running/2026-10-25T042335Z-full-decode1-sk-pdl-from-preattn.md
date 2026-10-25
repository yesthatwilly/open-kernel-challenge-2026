---
problem: muse_attention_decode
slug: full-decode1-sk-pdl-from-preattn
created: 2026-10-25T042335Z
base: master
area: launch overhead (full decode_1, pre-attention -> stream-K kernel)
change: >
  Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is
  22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the
  serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with
  cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master
  Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent
  may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize()
  (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup,
  the tile-range computation, page_table index loads, and cp.async issue of the prologue ring
  stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only
  key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key
  seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket
  writes stay after the wait. Batch > 1 unchanged (the running
  full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1
  cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN:
  full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of
  pre-attn+kernel with and without PDL and assert PDL is not slower.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: overlaps the sk kernel launch, index loads and ring fill with the 1.4 us pre-attention kernel; shape 22.6 -> ~20.5 us (-1 to -2.5 us)
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32* *layer=sliding* *load=verify_*"
    direction: flat
    why: outside the batch-1 gate
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T043143Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T043143Z
base_sha: df59fe97803906261a926deeb694b808107ab115
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/full-decode1-sk-pdl-from-preattn
worktree: .worktrees/muse_attention_decode/full-decode1-sk-pdl-from-preattn
mode: null
restack_of: null
stale_base: null
---

# full-decode1-sk-pdl-from-preattn

## Change

Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is 22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize() (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup, the tile-range computation, page_table index loads, and cp.async issue of the prologue ring stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket writes stay after the wait. Batch > 1 unchanged (the running full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1 cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN: full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of pre-attn+kernel with and without PDL and assert PDL is not slower.

## Expected

- `*layer=full load=decode_1 *` faster: overlaps the sk kernel launch, index loads and ring fill with the 1.4 us pre-attention kernel; shape 22.6 -> ~20.5 us (-1 to -2.5 us)
- `*layer=full load=decode_16* *layer=full load=decode_32* *layer=sliding* *load=verify_*` flat: outside the batch-1 gate

## Risk

Kill: decode_1 full shape not >= 1 us faster. splitkv-pdl-attn-to-combine (A->B) was neutral; this boundary differs because the dependent here can do useful gather work before the wait.

## Adversarial review

Reviewed 2026-10-25T043143Z by an adversarial subagent (model opus).

**Verdict: proceed.** PDL from the Triton pre-attention kernel into the B=1 stream-K kernel is rules-safe: only key seq_b-1 is written by the primary, and the hypothesis defers that tile plus q until after griddepcontrol.wait. With no explicit trigger in the primary, though, the overlap is limited to the launch gap and roughly meets the 1 us kill bar. Time an explicit early-trigger variant too, and split the q load out of the first tile's cp.async group.

### Justification
Mechanism: full decode_1 is 22.6 us of shape against 1.4 + 18.6 us of kernels, leaving ~2.6 us of launch and dependency gap. The sk kernel's prologue is a serial chain: load_prefix(seqlens) -> issue_meta (page_table) -> cp.async of K/V tiles, two dependent DRAM round trips (~1-1.5 us) before the first byte of KV arrives. Running that chain under pre-attention is a real saving. WEAKEST LINK: without griddepcontrol.launch_dependents in the primary, the dependent is released only when every primary CTA has exited. The overlap is then the dependent's launch latency and the primary's tail/flush (~0.5-1 us), not the 1.4 us body. splitkv-pdl-attn-to-combine (neutral, same environment) is the precedent for a sub-noise PDL gain. An explicit trigger at the top of _pre_attention_kernel (Triton gdc_launch_dependents if this Triton has it, else tl.inline_asm_elementwise('griddepcontrol.launch_dependents;')) lets the index chain overlap the body. It is a no-op for any dependent launched without the PDL attribute, so every other shape is unaffected.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (landed c3c1b01): the dependent is its variant-3 kernel at grid SM. The change touches only the B=1 launch, so read the full decode_1 row and the sk kernel row. The kernel row should not change; the shape row should drop.
- fused-preattention-triton (landed): this is the primary. If an explicit trigger is added, the kernel changes for every shape (benign no-op). Check that sliding decode rows (which do not run it) and the other full rows stay flat.
- In flight: full-decode1-sk4-cluster-dsmem-merge (queued) changes the same B=1 launch to a cluster launch. Run against master each is attributable on its own; a later stack needs both attributes on one cudaLaunchKernelEx. full-decode-splitkv-s3-fused-preattn (running) removes pre-attn at B>1 only, so it does not overlap.

### Interference and A/B needs
No companion. The prepare prints pre-attn+kernel with and without PDL (and with the explicit-trigger variant, see open_questions) on the same input. PATH CHOSEN names the variant, so a gain or flat on full decode_1 is attributable. A separate control check would only repeat the prepare.

### Open questions
- BUILDER: add the variant 'PDL + explicit griddepcontrol.launch_dependents at the top of _pre_attention_kernel' to the prepare timing, alongside no-PDL and PDL-implicit. Print all three and choose the fastest that is not slower than no-PDL.
- BUILDER: report the shape-row delta against the 1 us kill bar from the profile. If only the implicit variant is available and the delta is < 1 us, record it as a stacking item for sk4 rather than a dead end.

