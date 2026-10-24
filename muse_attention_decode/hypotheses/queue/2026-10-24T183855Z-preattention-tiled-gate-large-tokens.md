---
problem: muse_attention_decode
slug: preattention-tiled-gate-large-tokens
created: 2026-10-24T183855Z
base: master
area: pre-attention norm+rope
change: >
  Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is
  muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.patch) behind a
  token-count gate. Add that patch's group-tiled kernel as a NEW Triton kernel
  _pre_attention_tiled_kernel beside master's _pre_attention_kernel (do not modify the old one):
  grid (tokens, Hkv), num_warps 4, program (t, g) normalizes the 8 query heads of group g as one
  [8, 64] tile plus key head g, applies RoPE when ROPE, writes FP8 q in the PAIRED or unpaired
  layout exactly as master (padded half zeroed when PAIRED), and stores the FP8 key and value rows
  divided by k_scale / v_scale at cache_loc[t]. In kernel_fn, launch the tiled kernel when tokens
  >= 64 (verify_16 and verify_32, sliding and full) and master's _pre_attention_kernel otherwise
  (decode at every batch and verify_1, where the tiled version was 0.1 us slower). Same arguments,
  same outputs, same rounding points.
expected:
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "measured in the parent run: _pre_attention_kernel 4.1 -> 2.7 us, sliding verify_32 28.7 -> 27.4 us (-4.5%)"
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: "parent: 2.7 -> 2.0 us, about -3% on the shape"
  - shapes: "*layer=full load=verify_16 * *layer=full load=verify_32 *"
    direction: faster
    why: "same kernel saving, ~1 us out of 162 / 301 us: within noise"
  - shapes: "*load=decode_* *load=verify_1 *"
    direction: flat
    why: "below the 64-token gate, master's kernel runs"
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T184124Z
review_companions: null
---

# preattention-tiled-gate-large-tokens

## Change

Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.patch) behind a token-count gate. Add that patch's group-tiled kernel as a NEW Triton kernel _pre_attention_tiled_kernel beside master's _pre_attention_kernel (do not modify the old one): grid (tokens, Hkv), num_warps 4, program (t, g) normalizes the 8 query heads of group g as one [8, 64] tile plus key head g, applies RoPE when ROPE, writes FP8 q in the PAIRED or unpaired layout exactly as master (padded half zeroed when PAIRED), and stores the FP8 key and value rows divided by k_scale / v_scale at cache_loc[t]. In kernel_fn, launch the tiled kernel when tokens >= 64 (verify_16 and verify_32, sliding and full) and master's _pre_attention_kernel otherwise (decode at every batch and verify_1, where the tiled version was 0.1 us slower). Same arguments, same outputs, same rounding points.

## Expected

- `*layer=sliding load=verify_32 *` faster: measured in the parent run: _pre_attention_kernel 4.1 -> 2.7 us, sliding verify_32 28.7 -> 27.4 us (-4.5%)
- `*layer=sliding load=verify_16 *` faster: parent: 2.7 -> 2.0 us, about -3% on the shape
- `*layer=full load=verify_16 * *layer=full load=verify_32 *` faster: same kernel saving, ~1 us out of 162 / 301 us: within noise
- `*load=decode_* *load=verify_1 *` flat: below the 64-token gate, master's kernel runs

## Risk

The sliding verify gain is replaced if sliding-cuda-cluster-dsmem-onelaunch lands (it fuses pre-attention into its one launch). full-verify-cuda-streamk-splitkv keeps master's _pre_attention_kernel (PAIRED=False) in front of its A8 kernel, so the full verify_16/32 part survives that rewrite, and the tiled kernel stacks there cleanly. Judge on the _pre_attention_tiled_kernel row.

## Adversarial review

Reviewed 2026-10-24T184124Z by an adversarial subagent (model opus).

**Verdict: proceed.** A gated restack of a kernel already measured on the same verify path (parent job 877731a3: _pre_attention_kernel 4.2->2.7 us at verify_32, 2.8->2.0 us at sliding verify_16), with no landed change since then touching verify pre-attention, so one run with its per-kernel row can be read; expect the headline to stay inside the CI (~+0.4%) and judge it on the _pre_attention_tiled_kernel row and sliding verify_16/32.

### Justification
Mechanism is plausible: the kernel moves <1.2 MB (~0.25 us at 4.8 TB/s) but takes 4.1 us at verify_32 on current best (job 698d4cd3) because it issues 256*18 = 4608 single-warp CTAs, each a dependent load-reduce-store chain; 512 four-warp CTAs cut issue and tail overhead. The parent run measured exactly this against its own base (paired-heads-launch-cleanup-ab-tune-batch-gate, job bd64bf9f): sliding verify_32 pre-attn 4.2->2.7 us and shape 28.9->27.4 us (speedup 1.198->1.249), sliding verify_16 2.8->2.0 us (22.9->22.5 us), full verify_16 2.8->2.7 us, full verify_32 pre-attn fell out of the top-4. Current master still shows 4.1 / 2.7 us on those rows, so the opportunity is intact. Expected directions are consistent with TASK.md: verify_32 load = 46*28.7 + 16*300.8 ~ 6133 us, saving ~1.4 us on both layers ~ 86 us (~1.4%); verify_16 load ~ 3636 us, saving ~34 us (~0.9%); geomean over 6 loads ~ +0.4%, well inside the >=1.5% CI, so the headline will read neutral by construction and the verdict must come from the kernel row and the sliding verify_16/32 shape ratios. Weakest link: the stated reason for the 64-token gate ('tiled was 0.1 us slower at decode and verify_1') is not supported by the parent profile, where the tiled kernel was equal or 0.1 us faster on every small shape (sliding verify_1 1.7->1.6, full verify_1 1.5->1.4, full decode_1 1.4->1.4); the gate is harmless (it only forgoes ~0.1 us) but it means the 'flat' expectation on small shapes is trivially true and carries no information. Secondary weak link: full verify_16 barely moved in the parent (2.8->2.7) despite the same token count as sliding verify_16 (2.8->2.0); the PAIRED path writes 2x q bytes plus a zero tile, so do not expect the full-layer half of the claim.

### Confounders on the current kernel
- fused-preattention-triton: the kernel being regridded; the new kernel is added beside it and arithmetic/rounding are copied, so it is the baseline of the comparison, not a confounder of attribution.
- paired-heads-launch-cleanup / paired-heads-launch-cleanup-ab-tune-batch-gate: on full verify_16/32 (batch > 1, not splitkv) paired=True, so the tiled kernel writes the zero-padded 128-byte query; the full-layer saving is specific to the PAIRED layout (parent: only 0.1 us at full verify_16). Without pairing the full rows would likely gain more, but that is a different base and does not make this run ambiguous.
- full-decode-cuda-gather-splitkv, sliding-fused-triton-decode1-gate, sliding-deep-splits-on-cuda-splitkv: touch decode only (sliding decode bypasses _pre_attention_kernel entirely; full decode >1 batch has <=32 tokens, below the gate) — no interaction with the gated verify path, which is byte-identical to the parent's base outside the pre-attention launch.
- fa3-hdim128-head-pairing: same as the pairing note above; FA3 consumes the same q bytes, so FA3 time should be unchanged.

### Interference and A/B needs
No companion warranted. The parent run already is the A/B: the identical tiled kernel versus master's kernel on a verify path that has not changed since (git diff f51cac0..HEAD on submission.py only adds the CUDA split-KV decode path and the sliding fused decode path, neither reached at q_len 8). A gain on the kernel row would be attributable; a loss would be equally attributable since the gated code is the only difference on verify shapes. An ab-revert of any landed hypothesis would not change the verify pre-attention launch, and a gate-off control is just master. If anything, a later follow-up could drop the gate (parent data says tiled is never slower), but that is a new hypothesis, not a companion.

### Open questions
- Is the 64-token gate needed at all? Parent profile shows the tiled kernel equal or faster on every small shape; dropping it would simplify (one kernel) at a gain of ~0.1 us on full decode_1/verify_1.
- Why did full verify_16 gain only 0.1 us versus 0.8 us on sliding verify_16 at the same token count? The PAIRED zero-tile store with a runtime half offset may serialize stores; a [G, 2D] single store with the half selected by a mask could recover it.
- Obsolescence risk: sliding-cuda-cluster-dsmem-onelaunch (running) and fused-sliding-preattn-into-window-kernel / sliding-verify-cuda-streamk-window (queued) may absorb sliding verify pre-attention; stream-fork-preattention would overlap the same launch and should be evaluated on top of this if both proceed.
- Could tiling several tokens per program (e.g. 4 tokens x 1 group per CTA at verify_32) push the kernel further toward ~1 us? Worth a follow-up hypothesis if this one lands.

