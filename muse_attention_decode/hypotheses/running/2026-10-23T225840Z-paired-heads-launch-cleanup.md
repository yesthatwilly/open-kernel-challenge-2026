---
problem: muse_attention_decode
slug: paired-heads-launch-cleanup
created: 2026-10-23T225840Z
base: master
area: attention boundary
change: >
  Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must
  have landed too): remove the five pad/un-pad launches the pairing added on full layers. (1) Pad
  at the source: the fused pre-attention Triton kernel's query programs (one per (token, head))
  write their 64 normalized FP8 values straight into the padded [tokens, Hq, 128] FP8 buffer at
  column offset half*64 with half = (h // G) % 2, and write 64 FP8 zeros into the other half, so
  torch.zeros and the two slice copies disappear and FA3 reads the buffer directly. (2) Un-pad in
  the epilogue: replace rms_norm_gate on full layers with a Triton norm-gate kernel, grid (tokens,
  Hq), one program per head: it loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at
  [t, h, half*64 : half*64+64] (the garbage half is never read), computes r = rsqrt(mean(x^2) +
  1e-5) in fp32, loads the 64 gate logits from qkv row t at (Hq + 2*Hkv)*64 + h*64, and stores
  bf16 (x * r * sigmoid(gate)) into the final [B, L, Hq*64] output; the two un-pad copies and
  SGLang's rms_norm_gate disappear. The metadata rebuild on decode steps remains the only extra
  launch over the fused-preattention path. Sliding layers keep whatever the landed base does
  (fused pre-attention writing the [tokens, Hq, 64] buffer, unpaired FA3, SGLang rms_norm_gate).
expected:
  - shapes: "*layer=full load=decode_1 page*"
    direction: faster
    why: five launches (~10-12 us) removed from a ~40 us launch-bound shape
  - shapes: "*layer=full load=verify_1 page*"
    direction: faster
    why: same five launches on a ~45 us shape
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: same launches removed but they are under 6% of 150-200 us; expect a small gain, possibly inside the CI
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: launches are under 3% of this shape
  - shapes: "*layer=full load=verify_*"
    direction: faster
    why: same launches, same magnitudes as the decode shapes
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: fa3-hdim128-head-pairing
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T104027Z
review_companions: [paired-heads-launch-cleanup-ab-tune-batch-gate]
stage: implemented
claimed_at: 2026-10-24T104027Z
base_sha: 81028ae9b9d5098940f55070c9174a5674a3aaaf
base_blob: bc007e643ac0b440147a732071f23d9b3563174c
branch: hyp/muse_attention_decode/paired-heads-launch-cleanup
worktree: .worktrees/muse_attention_decode/paired-heads-launch-cleanup
mode: null
implemented_at: 2026-10-24T104221Z
impl_commit: 12230f154820fa60ff8dc5f1e9efca0e1d05c4c9
submission_sha256: 9a39092406218bc28179ef0a494d95f6575165f73da1dba63e44b168aa5a82ff
norm_sha256: 53b59c7f32fce92796ec3a97e3c5c5231248249bd61512719e8cb1fe4279cbd9
submission_blob: 60fc66432a00f0c6a1e2228f6a0775cc08f492fc
lint_warnings: 0
---

# paired-heads-launch-cleanup

## Change

Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing added on full layers. (1) Pad at the source: the fused pre-attention Triton kernel's query programs (one per (token, head)) write their 64 normalized FP8 values straight into the padded [tokens, Hq, 128] FP8 buffer at column offset half*64 with half = (h // G) % 2, and write 64 FP8 zeros into the other half, so torch.zeros and the two slice copies disappear and FA3 reads the buffer directly. (2) Un-pad in the epilogue: replace rms_norm_gate on full layers with a Triton norm-gate kernel, grid (tokens, Hq), one program per head: it loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at [t, h, half*64 : half*64+64] (the garbage half is never read), computes r = rsqrt(mean(x^2) + 1e-5) in fp32, loads the 64 gate logits from qkv row t at (Hq + 2*Hkv)*64 + h*64, and stores bf16 (x * r * sigmoid(gate)) into the final [B, L, Hq*64] output; the two un-pad copies and SGLang's rms_norm_gate disappear. The metadata rebuild on decode steps remains the only extra launch over the fused-preattention path. Sliding layers keep whatever the landed base does (fused pre-attention writing the [tokens, Hq, 64] buffer, unpaired FA3, SGLang rms_norm_gate).

## Expected

- `*layer=full load=decode_1 page*` faster: five launches (~10-12 us) removed from a ~40 us launch-bound shape
- `*layer=full load=verify_1 page*` faster: same five launches on a ~45 us shape
- `*layer=full load=decode_16*` faster: same launches removed but they are under 6% of 150-200 us; expect a small gain, possibly inside the CI
- `*layer=full load=decode_32*` flat: launches are under 3% of this shape
- `*layer=full load=verify_*` faster: same launches, same magnitudes as the decode shapes
- `*layer=sliding*` flat: untouched; control

## Risk

Medium risk: two kernels touched (pre-attention and the epilogue) but both edits are mechanical placement changes with no new math; the attributed quantity is launch count on full layers. Correctness: the norm-gate epilogue must match reference._rms (eps 1e-5 inside the sqrt over the mean of 64 squares) and sigmoid(gate.float()); SGLang's rms_norm_gate took a bf16 attention output and this reads FA3's bf16 output too, so the rounding points are identical. The pad write must cover the full 128 columns every call (fresh zeros each step, not relying on a buffer from an earlier call: fresh inputs rule). Needs both fa3-hdim128-head-pairing and fused-preattention-triton landed; the single depends_on names the pairing because its buffer layout is what this edits. If fa3-hdim128-head-pairing is rejected this is moot.

## Adversarial review

Reviewed 2026-10-24T104027Z by an adversarial subagent (model opus).

**Verdict: proceed-with-companions.** The cleanup is sound and should gain on full B=16/32, where the four-plus strided pad/un-pad elementwise kernels cost 3.3-4.9 us each, but it cannot recover batch 1. Most of the B=1 loss comes from the paired FA3 and combine kernels and the dropped judge metadata, not from launches. One companion is warranted: the same cleanup with pairing gated to batch > 1.

### Justification
The mechanism is plausible and the profile supports it at B>=16. On master (pairing), full decode_16 has two at::native elementwise rows of 4.4 us each and decode_32 4.6+4.5 us, while verify_16 shows 4.9+4.8 us in its top 4 alone. Add the remaining copies and the zeros fill and pad/un-pad is roughly 13-20 us per full shape. Removing it is a ~6-10% gain on decode_16/verify_16 and ~5-6% on decode_32/verify_32. That is well outside the +/-1.7% noise, so the 'decode_32 flat' expectation is probably wrong in direction (faster), which is harmless. The B=1 claim is the weakest link. Its arithmetic charges the whole 27.9->38.7 us (decode_1) and 30.0->46.0 us (verify_1) loss to the five launches, and the top_kernels contradict that. decode_1 base: FA3 14.6 + combine 7.1 + pre 1.3 + norm_gate 1.1 = 24.1 us of kernels, 27.9 us median. Paired: FA3 13.2 (-1.4) and combine 10.3 (+3.2; the combine now reduces 128-wide partials and a split schedule FA3 chose itself). Because the landed pairing passes scheduler_metadata=None on decode too (an implementation deviation from its spec), FA3 also launches prepare_varlen_num_blocks (~1.8-2.8 us) that the judge metadata used to save. Floor after cleanup is about 13.2+10.3+~2+1.3+~1.1 = ~28 us of kernels, ~31-32 us median against 27.9 base: still ~10-15% slower. verify_1 base: FA3 14.9 + combine 7.3 + prepare 2.8 + pre 1.4 = ~26.4 of 30.0. Paired: FA3 18.3 (+3.4) and combine 12.5 (+5.2), because with S=1 a B=1 verify has a single (request, super-head) tile and all parallelism comes from splits. Floor after cleanup is ~36 us of kernels, ~39-40 us median against 30.0: still ~25-30% slower. So the paired geometry is intrinsically worse at B=1, which matches LESSONS line 71 ('Gate it on batch'). The stated '~40/45 us launch-bound shape, five launches ~10-12 us' over-attributes. The hypothesis text is also stale: 'the metadata rebuild remains the only extra launch' is wrong, since master has no rebuild and instead pays FA3's internal prepare launch on decode. Design details check out. half = (h // G) % 2 is the parity of kv head h//G. Each (t,h) program owns its whole 128-byte q_pad row, so every byte is written each call. The gate offset (Hq+2Hkv)*64 + h*64 matches the qkv layout. Reading only the own half of FA3's output is what the un-pad copies did.

### Confounders on the current kernel
- fa3-hdim128-head-pairing (direct parent, defines the buffer layout): its B=1 loss is mostly intrinsic (combine +3.2/+5.2 us, verify FA3 +3.4 us, plus FA3's internal prepare launch now that metadata=None on decode), so the cleanup's B=1 readout measures pairing+cleanup against unpaired, not launch removal alone. A B=1 result that stays slower would be misread as 'the cleanup failed' when the launches really did go away. Read the elementwise rows disappearing, not the B=1 headline.
- fused-preattention-triton (the kernel being edited): it is not a confounder on timing. Its row is 1.3-4.1 us, and writing 128 instead of 64 FP8 bytes per q program changes that negligibly. But on full layers the edit makes its q output layout depend on the pairing branch, so the two must share one config-only predicate (window == 0 and hkv % 2 == 0 [and batch > 1 in the companion]), or FA3 reads the wrong layout.
- SGLang rms_norm_gate replaced on full layers only: if the Triton norm-gate is faster or slower than SGLang's (1.1 us at B=1), that shows up as part of the 'launch cleanup' gain. It is small and visible as its own row, so it does not change the decision.

### Interference and A/B needs
One companion is warranted. Without it, a B=1 result from the cleanup is ambiguous in the way that decides promotion. If decode_1/verify_1 stay below the pre-pairing base (27.9/30.0 us), we cannot tell from this run whether gating would beat cleanup-everywhere at B=1. That needs a measured run, because the gated B=1 path changes FA3's split schedule back to the judge metadata and the combine back to 64-wide, and that is not exactly a splice of old numbers into a new run (drift, the CUDA-graph composition). By my estimate (decode_1 ~31-32 us and verify_1 ~39-40 us after cleanup, against 27.9/30.0 unpaired), the gate wins at B=1 by ~10-25%. Restoring those two shapes at weight 16 is worth roughly +4-5% on the geomean (decode_1 load ~1.38 -> ~1.21-1.26 ms, verify_1 ~1.54 -> ~1.28-1.40 ms). The gated companion is then the likely promotion candidate, and the ungated parent is the attribution run for launch removal at B>=16. Both runs are identical at B>=16 (same code path), so a B>=16 mismatch between them doubles as a noise check. Nothing else is needed. An ab-revert of fused-preattention-triton is meaningless (the cleanup is written into it). A metadata-rebuild control is the same as the gate's B=1 path for practical purposes, and the gate covers it.

### Companions queued
- `paired-heads-launch-cleanup-ab-tune-batch-gate`: Decides whether pairing must be gated to batch > 1 or whether the launch cleanup alone makes batch 1 acceptable; also the likely promotion candidate (~+4-5% geomean over the ungated cleanup if the estimates hold).

### Open questions
- Why does each strided pad/un-pad copy cost 3.3-4.9 us even at 16-256 tokens x 16 heads? If it is a graph-replay artifact of at::native elementwise kernels on strided FP8 views, the B>=16 cleanup gain could be larger than estimated. Sum the elementwise rows on full decode_16/32 in the full profile before and after.
- With metadata=None on paired decode, does FA3 choose more splits at B=1 (the reason the combine grew 7.1 -> 10.3 us)? If the companion's B=1 path still loses to base, check that the judge metadata really is passed on that path.
- Is the threshold batch > 1 right, or would some batch in 2..15 (only 1/16/32 are scored) prefer unpaired? Only the scored shapes matter for the score; the tests at other batches only need correctness.
- Does the Triton norm-gate match SGLang's rms_norm_gate within the 2.5x-reference-error bound on the page_size 64 / kv_scale 1/64 tests, where the paired path is also taken?


## Implementation notes

Worktree `.worktrees/muse_attention_decode/paired-heads-launch-cleanup`, `muse_attention_decode/submission.py` only (61+/25-).

- **Pad at the source.** `_pre_attention_kernel` gained two constexprs, `PAIRED` and `G`. When `PAIRED`, each query program (t, h < HQ) computes `half = (h // G) % 2` and writes its 64 normalized FP8 values to `q_out[t, h, half*64 : half*64+64]` of a `[tokens, Hq, 128]` FP8 buffer. It also stores 64 FP8 zeros into the other half. The program owns its whole 128-byte row and writes every byte on every call, so the buffer is a plain `torch.empty` and nothing is reused across calls. The key/value programs are unchanged. `torch.zeros` and the two strided slice copies are gone, and FA3 gets the buffer directly (`q_attn = q`).
- **Un-pad in the epilogue.** New `_paired_norm_gate_kernel` with grid (tokens, Hq) and `num_warps=1`. It reads only head h's own half of FA3's output (`[t, h, half*64 : +64]`, using the output's real strides from `out.view(tokens, hq, 128)`). It computes `r = 1/sqrt(sum(x^2)/64 + 1e-5)` in fp32, which matches `reference._rms`, loads the gate logits from qkv row t at `(Hq + 2*Hkv)*64 + h*64`, and stores `bf16(x * r * sigmoid(gate))` into a fresh `[tokens, Hq*64]` output. The paired branch returns from there. The two un-pad copies and SGLang's `rms_norm_gate` are gone on full layers.
- **One predicate.** I moved `paired = window == 0 and hkv % 2 == 0` above the pre-attention launch. It now decides both the query layout (the PAIRED constexpr and the q buffer width) and the FA3/epilogue branch, as the review requires. It depends on config only. `g = hq // hkv` is computed for both paths (it is unused when PAIRED is False).
- **Sliding layers** are unchanged: 64-wide q buffer, unpaired FA3 with the judge's `scheduler_metadata`, SGLang `rms_norm_gate`.
- **Deviations.** None from the change paragraph. The hypothesis text says "the metadata rebuild remains the only extra launch". As the review notes, master has no rebuild: the paired path still passes `scheduler_metadata=None`, so FA3 does its internal prepare launch on decode. I left that alone because it is out of scope and the batch gate is the companion's job. I added no batch gate.
- **Uncertainties (nothing could be run on this node).**
  - I am assuming FA3's output for the paired call is (or views as) `[tokens, Hq, 128]`, as the old un-pad code also assumed via `out.view`.
  - The epilogue rounds once, to bf16 at the end. SGLang's `rms_norm_gate` might round intermediates differently. Either way the result is compared against an fp32 reference, so this should be within the bound, but the page-64 and kv_scale 1/64 paired tests are the ones to watch.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/paired-heads-launch-cleanup diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints. The new allocations (`torch.empty` for q and the output) are shape-only. The `paired`/`PAIRED` branch depends only on `config` (window, num_kv_heads).
- [x] Nothing derived from `data` is cached across calls. The padded q is fully rewritten (data plus zeros) each call. Only the existing per-theta `_inv_freq` cache remains, and it is config-derived.
- [x] n/a: weights. This problem has no weights and no `prepare()`.
- [x] Activations BF16 where SGLang keeps BF16 (FA3 bf16 output in, bf16 out, fp32 math in between). The KV cache stays FP8. Queries stay FP8 E4M3 (unchanged rounding, and the zero half is exact).
- [x] Attention sees every key. FA3 call parameters (window, lengths, page table) are unchanged, and there is no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. No new imports were added.
- [x] All work happens inside the timed call, with no output caching or replay.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept. No `triton.autotune`.

Lint (`okc_lint.py ... --problem muse_attention_decode`):

```
CLEAN: muse_attention_decode/submission.py
```

`python3 -I -m py_compile` OK. I removed the resulting `muse_attention_decode/__pycache__` from the worktree.
