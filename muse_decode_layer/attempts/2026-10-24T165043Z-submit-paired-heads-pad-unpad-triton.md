# muse_decode_layer · 2026-10-24T165043Z · submit · paired-heads-pad-unpad-triton

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T140156Z-paired-heads-pad-unpad-triton.md` (branch `hyp/muse_decode_layer/paired-heads-pad-unpad-triton`).

- **Job:** `a92b64be4c6547d3` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `77b5def`

## Goal
Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master's paired path in muse_decode_layer kernel_fn (full layers, batch > 1). Master's pairing adds five glue launches around FA3, visible in its profile (fb07a3bd3aa54baf) as elementwise_kernel<128,4> x3 (6.4 us) plus x2 (4.7 us) at full decode_32 and x3 6.4-6.8 us at full decode_16/verify_16: torch.zeros(q_attn) and the two strided qp[...] = qv[...] copies (which also cast to FP8), then torch.empty(attn) and the two a[...] = o[...] un-pad copies, followed by SGLang's FusedRMSNormGate. (1) Un-pad in the epilogue: copy _paired_norm_gate_kernel verbatim from muse_attention_decode/submission.py on master (grid (tokens, Hq), one program per head: loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at half (h // G) % 2, fp32 scaleless RMSNorm with eps 1e-5, times sigmoid of the gate logits read from qkv row t at (Hq + 2*Hkv)*64 + h*64 with qkv.stride(0), bf16 store into a fresh [tokens, Hq*64] output) and use it in place of the attn/a/o un-pad block and state['o_norm_gate'] when paired; that output goes to the o GEMM. (2) Pad in one launch: since decode_layer has no fused pre-attention kernel on master (fused-preattention-triton was neutral, not merged), add a small Triton kernel _pair_pad_q_kernel, grid (tokens, Hq), that reads head h's 64 bf16 q values from FusedQKNorm's output (pass its strides), casts to FP8 E4M3 and writes them at column half*64 of the [tokens, Hq, 128] FP8 buffer (allocated torch.empty) and 64 FP8 zeros in the other half, replacing torch.zeros plus the two slice copies. FA3's call (paired k/v views, scheduler_metadata None), the unpaired path (batch 1 and sliding layers), the cache store and everything after the o GEMM are unchanged.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/paired-heads-pad-unpad-triton`, file `muse_decode_layer/submission.py` only (1 file, +60/-14).

- Added `_pair_pad_q_kernel` (new Triton kernel, before `prepare`), grid `(tokens, hq)`, `num_warps=1`: loads head h's 64 values of FusedQKNorm's q at `t*q_stride_t + h*q_stride_h + j`, widens to fp32 (exact from bf16, so the E4M3 rounding matches torch's bf16 -> fp8 path), stores `float8e4nv` at column `((h // G) % 2) * 64` of the `[tokens, hq, 128]` FP8 row and 64 FP8 zeros in the other half. Every byte of the buffer is written, so it is allocated with `torch.empty`.
- Added `_paired_norm_gate_kernel`, copied verbatim from `muse_attention_decode/submission.py` on master (same signature, body and docstring).
- kernel_fn paired branch, pad: `torch.zeros` and the two `qp[...] = qv[...]` slice copies are replaced by `qh = q.view(tokens, hq, d)`, `assert qh.stride(2) == 1` (stride metadata only, no host sync), `torch.empty` and one `_pair_pad_q_kernel` launch that takes `qh.stride(0)` and `qh.stride(1)`. This follows the review's open question: FusedQKNorm's q is read through its real token and head strides, whether it is a fresh tensor or a strided view of qkv.
- kernel_fn paired branch, epilogue: `torch.empty(attn)`, the two un-pad copies and `state["o_norm_gate"]` are replaced by `attn = out.view(tokens, hq, 2*d)`, a fresh bf16 `[tokens, hq*d]` output and one `_paired_norm_gate_kernel` launch with `attn.stride(0)`, `attn.stride(1)` and `qkv.stride(0)`, called as the sibling calls it (EPS=RMS_EPS=1e-5, num_warps=1). Its output goes straight to the o GEMM.
- Unpaired path (batch 1, sliding layers) unchanged: it still calls `state["o_norm_gate"]`, now in an `else:`. Also unchanged: the FA3 call, the paired k/v views, `scheduler_metadata=None`, the cache store, and everything from the o GEMM on. `state` is unchanged (`o_norm_gate` is still used by the unpaired path).
- Deviations: none in substance. Assumptions: the pad kernel assumes FusedQKNorm's q last-dim stride is 1 (asserted). The epilogue assumes FA3's output and qkv each have a last-dim stride of 1, as the sibling does. `out.view(tokens, hq, 2*d)` is the same reshape master's `out.view(tokens, s, 2, g, 2*d)` already relied on. Nothing could be run on this node (no GPU, torch or Triton), so correctness is unverified until okc check/submit.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/paired-heads-pad-unpad-triton diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`/`.cpu()`/`synchronize()`/tensor prints. The two allocations are `torch.empty` sized by `tokens`/`hq`/`d` (config), the grids are `(tokens, hq)`, and the only branch is the existing config-only `paired` gate. The `assert` reads a stride (shape metadata), not a value.
- [x] Nothing derived from `data` is cached across calls: no state writes, and both buffers are fresh every call.
- [x] Weights: n/a: prepare() and the weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16: the epilogue reads FA3's bf16 output, computes in fp32 and stores bf16 for the o GEMM, as FusedRMSNormGate does. KV cache FP8 (store unchanged). Queries are E4M3 (`tl.float8e4nv` == `torch.float8_e4m3fn`), the same precision as master's cast copy and no coarser.
- [x] Attention sees every key: the FA3 call, causal flag, window and lengths are unchanged. The zero half of the query adds exactly 0 to each logit, as on master.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds two `@triton.jit` kernels and their launches only.
- [x] All work happens inside the timed call; no output caching or replay: both kernels launch inside kernel_fn on every paired call.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept; no `triton.autotune` added.

okc_lint output:

```
warn  host-sync    L319: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed; its `__pycache__` was removed from the worktree.

## Risks / what could fail
low
Smallest of the four (~+0.3-0.5% geomean, on the 16-weight full shapes only), picked over a re-run of the MoE map+align port because that idea already ran here (moe-align-single-kernel-small-t, every shape within +0.2..+1.9%, neutral) and FINDINGS give no reason to expect more on master. Risk is low: the epilogue kernel is verbatim and correct in the sibling; the pad kernel is new but trivial (cast+scatter). Check that FusedQKNorm's q layout ([tokens, Hq*64], possibly a strided view of qkv) is read through its real strides. Gains are per-shape 1.5-2%, so read the elementwise_kernel<128,4> rows disappearing in top_kernels, not the headline.

## Result (after the run)
- job `a92b64be4c6547d3` (submit), status done, passed True, tests 13/13
- speedup 1.1654x, 95% CI [1.1479, 1.1829], score 19409.883750919944 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.1664628806987323, "ci": [1.1489659374882513, 1.183959823909213], "job": "a017e10aec544c55"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/paired-heads-pad-unpad-triton.result.json`, `muse_decode_layer/hypotheses/done/neutral/paired-heads-pad-unpad-triton.patch`; branch `hyp/muse_decode_layer/paired-heads-pad-unpad-triton`
