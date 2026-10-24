---
problem: muse_attention_decode
slug: paired-norm-gate-token-tiled
created: 2026-10-24T183855Z
base: master
area: output norm-gate
change: >
  Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at
  full verify_16/32) from one 1-warp program per (token, query head) to one program per token:
  grid (tokens,), num_warps 4. Program t loads the [16, 64] tile of head h's own half (h // G) % 2
  of FA3's [tokens, 16, 128] output (column offset h * attn_stride_h + half * D + j, rows h =
  arange(16)), computes the scaleless RMSNorm per row (sum over axis 1, fp32, eps inside the sqrt
  of the mean of 64 squares), times sigmoid of the [16, 64] gate tile read from qkv, and stores
  bf16 into out[t, h * 64 + j]. Arithmetic and rounding are identical; only the decomposition
  changes. kernel_fn launches it with grid (tokens,) instead of (tokens, hq).
expected:
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: _paired_norm_gate_kernel is 3.8 us at verify_32 for 4096 one-warp programs moving 1 MB; 256 four-warp programs should be ~1.5 us, ~-0.8% on the 300 us shape
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: same, smaller (2048 -> 128 programs)
  - shapes: "*layer=sliding* *load=decode_* *load=verify_1 *"
    direction: flat
    why: these shapes do not run the paired epilogue
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T184058Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T184059Z
base_sha: 49da3d4776ed7ee46334a320b804bc460500870c
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/paired-norm-gate-token-tiled
worktree: .worktrees/muse_attention_decode/paired-norm-gate-token-tiled
mode: null
---

# paired-norm-gate-token-tiled

## Change

Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at full verify_16/32) from one 1-warp program per (token, query head) to one program per token: grid (tokens,), num_warps 4. Program t loads the [16, 64] tile of head h's own half (h // G) % 2 of FA3's [tokens, 16, 128] output (column offset h * attn_stride_h + half * D + j, rows h = arange(16)), computes the scaleless RMSNorm per row (sum over axis 1, fp32, eps inside the sqrt of the mean of 64 squares), times sigmoid of the [16, 64] gate tile read from qkv, and stores bf16 into out[t, h * 64 + j]. Arithmetic and rounding are identical; only the decomposition changes. kernel_fn launches it with grid (tokens,) instead of (tokens, hq).

## Expected

- `*layer=full load=verify_32 *` faster: _paired_norm_gate_kernel is 3.8 us at verify_32 for 4096 one-warp programs moving 1 MB; 256 four-warp programs should be ~1.5 us, ~-0.8% on the 300 us shape
- `*layer=full load=verify_16 *` faster: same, smaller (2048 -> 128 programs)
- `*layer=sliding* *load=decode_* *load=verify_1 *` flat: these shapes do not run the paired epilogue

## Risk

full-verify-cuda-streamk-splitkv (in the judge now) replaces the paired FA3 path and this epilogue at full verify_16/32; if it lands, this is moot and should be held. Small by construction: judge on the kernel row.

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: proceed.** A low-risk, rules-clean regrid of a 2.8/4.0 us epilogue (full verify_16/32 only) whose best case is about 0.2% on the headline, so it can only be read from the _paired_norm_gate_kernel profile row; run it only if full-verify-cuda-streamk-splitkv (running now, which removes this kernel) does not land first.

### Justification
Mechanism: on master, _paired_norm_gate_kernel runs only on the paired path (window == 0, batch > 1, not splitkv), which means full verify_16 and verify_32 only. The decode shapes went to the CUDA split-KV path, and batch 1 is unpaired. Profile de486093311d4bb9 measures it at 2.8 us (verify_16, 1.6% of 169 us) and 4.0 us (verify_32, 1.3% of 300.6 us). It moves about 1.5 MB at verify_32 (1 MB of the 128-wide FA3 output, of which only half is read, plus 0.25 MB of gate and 0.25 MB written), which is about 0.3 us at 4.8 TB/s. So it sits 10x or more above its bandwidth floor and is bound by latency and per-CTA overhead (4096 one-warp CTAs, each doing a serial load -> reduce -> gate load -> store chain on 128 B rows), not by bandwidth. Moving to 256 four-warp programs, each loading a [16,64] tile (8 elements per thread, with the attn and gate loads independent so both can be in flight), is a plausible way to get to about 1.5 us. Expected directions match the routing: faster at full verify_16/32, flat elsewhere, because no other shape runs this kernel. Magnitude against TASK weights: verify_32 load = 46*28.7 + 16*300.8 = about 6130 us, and saving 16*2.5 us is about -0.65%. verify_16 load is about 3640 us, and saving 16*1.3 us is about -0.6%. The geomean over 6 loads is about +0.2%, far inside the +-1.5% CI. The weakest link is the claim that CTA count is the bottleneck. With 4096 one-warp CTAs, every CTA is resident in one wave (132 SMs x 32 CTAs), so the 4 us may mostly be kernel-launch/ramp latency plus the dependent-load chain, which the regrid does not remove. The sibling precedent preattention-group-tiled-programs (the same one-warp -> tiled regrid on _pre_attention_kernel) came out neutral within CI, and its full verify rows were flat. The realistic win is 1-2 us on the kernel row, possibly less. There is also no PDL on this Triton launch, so the gap between FA3 and the epilogue stays either way.

### Confounders on the current kernel
- fa3-hdim128-head-pairing: created the [tokens,16,128] paired output this kernel reads, including the half = (h // G) % 2 indexing; without pairing the kernel does not exist and the unpaired path uses sglang rms_norm_gate. This is the base, not a confounder of the measured delta.
- paired-heads-launch-cleanup: introduced _paired_norm_gate_kernel itself; the hypothesis edits only its grid and tiling, so the measured delta is attributable to the change.
- paired-heads-launch-cleanup-ab-tune-batch-gate: limits pairing to batch > 1, which is why verify_1 is unaffected. This is consistent with the expected 'flat' rows.
- full-decode-cuda-gather-splitkv: moved full decode_16/32 off the paired path, so the kernel now runs only at verify_16/32. This shrinks the hypothesis's reach to 2 of 12 shapes but does not confound the measurement.
- Not landed but decisive: full-verify-cuda-streamk-splitkv (running) replaces the paired FA3 path and this epilogue at full verify_16/32. If it lands, this hypothesis is moot. If this one is measured first, its result does not carry over.

### Interference and A/B needs
No companion is warranted. The change has no sub-changes to isolate: it is one regrid with identical arithmetic. No landed hypothesis could plausibly flip its sign through interaction, because the kernel reads FA3's output and the gate, and FA3 and the pre-attention kernel are unchanged. Neither an ab-revert nor an ab-tune would make the headline readable, because the effect (about 0.2% geomean) is below the noise floor whatever runs next to it. The interpretable readout is the _paired_norm_gate_kernel row in the profile at verify_16/32 (base 2.8/4.0 us), together with the full verify_16/32 shape times. Judge it as a gain if the kernel row drops by at least 1 us and no shape regresses outside CI. Sequencing matters more than companions: hold dispatch until full-verify-cuda-streamk-splitkv is harvested, and drop this hypothesis if that one lands.

### Open questions
- Did full-verify-cuda-streamk-splitkv land? If it did, this kernel no longer runs on any scored shape and the hypothesis should be retired, not run.
- Is the 4.0 us kernel time launch/ramp-bound (in which case a regrid gains little and PDL or folding the epilogue into FA3's combine, as fused-full-combine-normgate proposes, is the real lever) or per-CTA-overhead-bound (in which case the regrid helps)? An nsys/ncu look at achieved occupancy and the CTA count per SM would settle it before spending a check.
- Correctness detail for the implementer: the per-row column offset depends on the row (half = (h // G) % 2 varies with h), so the [16,64] load needs a 2-D pointer block with a per-row offset, not a single base plus arange. The tl.sum over axis 1 must stay in fp32.

