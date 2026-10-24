# muse_attention_decode · 2026-10-24T115932Z · submit · paired-heads-launch-cleanup

Hypothesis: `muse_attention_decode/hypotheses/done/gain/2026-10-23T225840Z-paired-heads-launch-cleanup.md` (branch `hyp/muse_attention_decode/paired-heads-launch-cleanup`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `25689c3`

## Goal
Promote check-proven gain paired-heads-launch-cleanup (1.2436x [1.2250,1.2623], check job 2846122f764043da) to the board.

Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing added on full layers. (1) Pad at the source: the fused pre-attention Triton kernel's query programs (one per (token, head)) write their 64 normalized FP8 values straight into the padded [tokens, Hq, 128] FP8 buffer at column offset half*64 with half = (h // G) % 2, and write 64 FP8 zeros into the other half, so torch.zeros and the two slice copies disappear and FA3 reads the buffer directly. (2) Un-pad in the epilogue: replace rms_norm_gate on full layers with a Triton norm-gate kernel, grid (tokens, Hq), one program per head: it loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at [t, h, half*64 : half*64+64] (the garbage half is never read), computes r = rsqrt(mean(x^2) + 1e-5) in fp32, loads the 64 gate logits from qkv row t at (Hq + 2*Hkv)*64 + h*64, and stores bf16 (x * r * sigmoid(gate)) into the final [B, L, Hq*64] output; the two un-pad copies and SGLang's rms_norm_gate disappear. The metadata rebuild on decode steps remains the only extra launch over the fused-preattention path. Sliding layers keep whatever the landed base does (fused pre-attention writing the [tokens, Hq, 64] buffer, unpaired FA3, SGLang rms_norm_gate).

## What changed, and where
Worktree `.worktrees/muse_attention_decode/paired-heads-launch-cleanup`, `muse_attention_decode/submission.py` only (61+/25-).

- **Pad at the source.** `_pre_attention_kernel` gained two constexprs, `PAIRED` and `G`. When `PAIRED`, each query program (t, h < HQ) computes `half = (h // G) % 2` and writes its 64 normalized FP8 values to `q_out[t, h, half*64 : half*64+64]` of a `[tokens, Hq, 128]` FP8 buffer. It also stores 64 FP8 zeros into the other half. The program owns its whole 128-byte row and writes every byte on every call, so the buffer is a plain `torch.empty` and nothing is reused across calls. The key/value programs are unchanged. `torch.zeros` and the two strided slice copies are gone, and FA3 gets the buffer directly (`q_attn = q`).
- **Un-pad in the epilogue.** New `_paired_norm_gate_kernel` with grid (tokens, Hq) and `num_warps=1`. It reads only head h's own half of FA3's output (`[t, h, half*64 : +64]`, using the output's real strides from `out.view(tokens, hq, 128)`). It computes `r = 1/sqrt(sum(x^2)/64 + 1e-5)` in fp32, which matches `reference._rms`, loads the gate logits from qkv row t at `(Hq + 2*Hkv)*64 + h*64`, and stores `bf16(x * r * sigmoid(gate))` into a fresh `[tokens, Hq*64]` output. The paired branch returns from there. The two un-pad copies and SGLang's `rms_norm_gate` are gone on full layers.
- **One predicate.** I moved `paired = window == 0 and hkv % 2 == 0` above the pre-attention launch. It now decides both the query layout (the PAIRED constexpr and the q buffer width) and the FA3/epilogue branch, as the review requires. It depends on config only. `g = hq // hkv` is computed for both paths (it is unused when PAIRED is False).
- **Sliding layers** are unchanged: 64-wide q buffer, unpaired FA3 with the judge's `scheduler_metadata`, SGLang `rms_norm_gate`.
- **Deviations.** None from the change paragraph. The hypothesis text says "the metadata rebuild remains the only extra launch". As the review notes, master has no rebuild: the paired path still passes `scheduler_metadata=None`, so FA3 does its internal prepare launch on decode. I left that alone because it is out of scope and the batch gate is the companion's job. I added no batch gate.
- **Uncertainties (nothing could be run on this node).**
  - I am assuming FA3's output for the paired call is (or views as) `[tokens, Hq, 128]`, as the old un-pad code also assumed via `out.view`.
  - The epilogue rounds once, to bf16 at the end. SGLang's `rms_norm_gate` might round intermediates differently. Either way the result is compared against an fp32 reference, so this should be within the bound, but the page-64 and kv_scale 1/64 paired tests are the ones to watch.

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium

## Result (after the run)
(pending)
