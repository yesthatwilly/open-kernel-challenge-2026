---
problem: muse_attention_decode
slug: full-decode-cuda-fp8-qk-mma
created: 2026-10-24T165302Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel
  (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the
  native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of
  widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row
  quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1
  = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers
  (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 +
  quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at
  dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR
  (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and
  the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4
  mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick
  and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 =
  row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened
  exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte
  for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached
  build is not reused; keep the import-time self-check and its thresholds as they are (it now
  validates the fp8 QK path).
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: kernel A 111.4 us at 2.95 TB/s vs the 63.2 us floor; per lane per tile it issues 16 cvt.f16x2.e4m3x2 for K+Q-side work plus 16 for V and 8+16 mma; dropping the K widening halves the cvt count and halves QK mma issue (the 8 zero rows waste half of either op). The review of stage 1 estimated ~40 us of cvt ALU at decode_16 that must hide under the gather; if A is partly issue-bound this is ~5-10% on A (100-106 us), shape ~113-118 us
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: kernel A 245.5 us at 2.67 TB/s, lower per-byte efficiency than decode_16, which points at issue/ALU pressure rather than pure DRAM; target 225-235 us
  - shapes: "*layer=sliding* *load=verify_* *layer=full load=decode_1*"
    direction: flat
    why: outside the CUDA decode path; untouched
risk: medium
depends_on: full-decode-cuda-gather-splitkv
priority: 3
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T165634Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T165634Z
base_sha: 700fed761b57b25cbae687d07371c4020bf38af9
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/full-decode-cuda-fp8-qk-mma
worktree: .worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma
mode: submit
implemented_at: 2026-10-24T165945Z
impl_commit: 3a7b503a24dd5f850b506b560784c7dac2db844e
submission_sha256: 6b2885c6eef2261cfb2ebf17a3786cadd6c859db5b106c8a84e1e9db2a238dc4
norm_sha256: 52d730f65a802a1c715435b6042d13b9b15c625560aa34484f993e18db2286af
submission_blob: 2ee93584612efa3b3320db386889110c4f2c939b
lint_warnings: 0
attempt_file: 2026-10-24T165946Z-submit-full-decode-cuda-fp8-qk-mma.md
prepared_at: 2026-10-24T165946Z
job: b939681d3ab549d7
dispatched_at: 2026-10-24T170002Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.3438993089368396
speedup_ci: [1.323740819302787, 1.3640577985708922]
harvested_at: 2026-10-24T173816Z
best_at_harvest: 1.3948738389144513
promotion: null
---

# full-decode-cuda-fp8-qk-mma

## Change

Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1 = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 + quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4 mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 = row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached build is not reused; keep the import-time self-check and its thresholds as they are (it now validates the fp8 QK path).

## Expected

- `*layer=full load=decode_16*` faster: kernel A 111.4 us at 2.95 TB/s vs the 63.2 us floor; per lane per tile it issues 16 cvt.f16x2.e4m3x2 for K+Q-side work plus 16 for V and 8+16 mma; dropping the K widening halves the cvt count and halves QK mma issue (the 8 zero rows waste half of either op). The review of stage 1 estimated ~40 us of cvt ALU at decode_16 that must hide under the gather; if A is partly issue-bound this is ~5-10% on A (100-106 us), shape ~113-118 us
- `*layer=full load=decode_32*` faster: kernel A 245.5 us at 2.67 TB/s, lower per-byte efficiency than decode_16, which points at issue/ALU pressure rather than pure DRAM; target 225-235 us
- `*layer=sliding* *load=verify_* *layer=full load=decode_1*` flat: outside the CUDA decode path; untouched

## Risk

Single attributable change on the measured kernel: read kernel A time in top_kernels at decode_16/32 against 111.4/245.5 us. If flat, kernel A is DRAM/TLB-bound on the random 128-byte rows, not issue-bound, and further ALU tuning of the decode kernel should stop (next levers would be bytes in flight: stages / CTAs per SM). Numerics: e4m3 x e4m3 products are exact in fp32 and the f16 path also widened them exactly, so only the tensor-core accumulation order/precision differs (Hopper fp8 MMA accumulates with reduced internal precision; for 64-term logits that is ~1e-4 relative, far inside the 3% bound; the self-check rel L2 should stay ~1.7e-3). PTX m16n8k32 e4m3 needs sm_89+, the build already uses -arch=sm_90a. Overlaps in source with full-verify-cuda-streamk-splitkv (which adds new kernels to the same CUDA string and uses the same fp8 QK op in its own kernel); whichever lands second rebases with a textual merge, shapes are disjoint.

## Adversarial review

Reviewed 2026-10-24T165634Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is one change you can attribute, inside the kernel being measured. The PTX m16n8k32 e4m3 fragment mapping in the spec is correct, and the numerics are within the rules. Two things are uncertain. ptxas may lower FP8 mma.sync on sm_90a to F2FP+HMMA instead of a native FP8 MMA. And the decode_32 'issue-bound' argument does not hold up. A flat result is still informative, so proceed as a medium-risk scored submit, but build in a fallback to the f16 v1 kernel.

### Justification
Mechanism. Kernel A at decode_16 is 111.4 us for ~329 MB (2.95 TB/s), against a 63.2 us floor. Per lane per 64-token tile it issues 16 cvt for K (8 calls of e4m3x4_to_f16x2x2), 16 cvt plus 8 prmt for V, and 8+8 mma.sync. It also does the softmax ex2/shfl work. The spec writes '8+16 mma'; the P@V count is actually 8 (2 hw x 4 j). The stage-1 review estimated ~40 us of cvt ALU that has to hide under the gather. Removing the K widening halves the cvt count, halves the QK mma count and frees 4 Q registers. If A is partly issue-bound, 5-10% on A is a fair upper bound. I checked the layout against the PTX m16n8k32 .e4m3 figures. A: a0 = row groupID, k 4*qi..+3; a1 = row +8; a2 = row groupID, k 16+4*qi..+3; a3 = row +8. B: b0 = k 4*qi..+3, b1 = k 16+4*qi..+3, col = groupID. C has the same f32 layout as m16n8k16. The proposed operands (a1 = a3 = 0, chunk index g*4 + ks*2 + {0,1} XOR (tt&7), byte offset 4*qi) match this. The K reads stay conflict-free: 8 tokens with distinct tt&7 x 4 qi words give 32 distinct banks. The once-per-item Q load is now a 4-way bank conflict, which costs almost nothing. Weakest link 1: native support on sm_90. FP8 mma.sync m16n8k32 is specified for sm_89+, but Hopper's documented FP8 tensor-core path is wgmma. ptxas may lower mma.sync e4m3 on sm_90a to F2FP conversions plus HMMA.16816. In that case the K cvt does not go away, and the Q operand could be re-converted every tile instead of once per item, which is flat or slower. I could not verify this here: there is no CUDA toolkit or GPU on this node. A flat A time therefore has two explanations: 'A is DRAM/TLB-bound', or 'the FP8 MMA was emulated'. Weakest link 2: decode_32. It runs at a lower bandwidth per byte (2.67 vs 2.95 TB/s), but its ALU work per byte is identical to decode_16. An issue-bound kernel would run at the same TB/s at both shapes. The drop as the footprint doubles (~350 -> ~700 MB of randomly permuted rows) points to TLB reach or DRAM row locality, not ALU. Expect decode_32 to gain no more than decode_16 in relative terms, likely less. The 225-235 us target is optimistic. Headline arithmetic: about 123 -> 116 us at decode_16 (load 2925 -> ~2813, 1.039x) and about 257.7 -> 240 us at decode_32 (load 5282 -> ~5000, 1.056x). That works out to only ~+1.6% on the 6-load geomean (1.347 -> ~1.37x), inside the +-1.5% CI and ~2% drift. Judge the run on kernel A's own top_kernels time against 111.4/245.5 us, not on the headline. Directions in `expected` are consistent with TASK.md: sliding, full decode_1 and every verify shape stay outside the gate and are byte-identical.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv (direct parent, same kernel): no confound. Only the QK operand path inside kernel A changes, and A's own time is read directly. Its pipeline depth (4 stages, 2 CTAs/SM, 128 KB in flight per SM) does decide whether A is issue-bound at all, so a flat result settles the question only for this pipeline configuration.
- fused-preattention-triton: it produces the FP8 q bytes that now feed the MMA directly instead of being widened. It is the same numeric input as today, since widening e4m3 to f16 is exact, so there is no confound beyond tensor-core accumulation order.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / -ab-tune-batch-gate / sliding-fused-triton-decode1-gate: the gated path runs with PAIRED=False and never touches FA3 or the sliding kernel. They cannot interact.

### Interference and A/B needs
No companion is warranted. The result can be read directly from kernel A's profile row at decode_16/32 against the parent's 111.4/245.5 us; the shapes outside the gate are byte-identical. An ab-revert of the parent is impossible (the kernel is the parent). A control that splits 'drop Q cvt' from 'drop K cvt' does not make sense, because both come from one MMA instruction change. The one ambiguity a companion cannot fix is native vs emulated FP8 mma.sync on sm_90a. That needs a SASS check: before claiming, compile the CUDA string offline with nvcc -cubin -arch=sm_90a on any machine with CUDA 12.4+ (no GPU needed) and grep cuobjdump -sass for QMMA/HMMA and F2FP counts. Do not spawn a process from submission.py: that is a rules violation. If you cannot do the SASS check, record a flat A time as 'emulated-or-memory-bound' (undetermined), not as proof that A is DRAM-bound. Do not stop the ALU line of work on that basis alone.

### Open questions
- Does ptxas for sm_90a emit native FP8 MMA for mma.sync.m16n8k32.e4m3, or F2FP+HMMA emulation? This decides how to read a flat result. Check the SASS offline before or alongside the run.
- Why is decode_32 at 2.67 TB/s vs decode_16 at 2.95 TB/s when its ALU work per byte is the same? That points to TLB reach or DRAM locality over the ~700 MB random footprint. If confirmed, the next lever is bytes in flight or access order (stages, CTAs/SM, chunk ordering for locality), not ALU.
- Register count after the change (qa[4][2] -> 4 regs). If it drops well below the 128-reg cap of __launch_bounds__(256,2), would 3 CTAs/SM become possible? That is a separate follow-up, not part of this run.
- Overlap with full-verify-cuda-streamk-splitkv (same FP8 op, same CUDA string). Whichever lands second rebases. This run's SASS and timing evidence on FP8 mma.sync should feed into that hypothesis's design.


## Implementation notes

Worktree: `.worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma`, file `muse_attention_decode/submission.py` only (diff: 85 insertions, 15 deletions; uncommitted).

- **Kernel A QK (`splitkv_attn_kernel`) under `#if SPLITKV_FP8QK`:** a new `mma16832_e4m3` helper wraps `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. `qa` is `uint32_t qa[2][2]` (4 regs). It is loaded at `t == 0` straight from `q_s` with no cvt: `qa[s][0]` = word at dims `s*32 + 4*qi`, `qa[s][1]` = word at `s*32 + 16 + 4*qi`, row = query `g*G + quad`, and a1 = a3 = 0. K B operand per k-step `ks` (2 of them): `b0` = word at chunk `(g*4 + ks*2) ^ (tt&7)`, `b1` = chunk `(g*4 + ks*2 + 1) ^ (tt&7)`, byte offset `4*qi`. That is 4 mma16832 per (warp, tile) and 0 K cvt. Natural dim order. Everything else in kernel A, kernel B, the grid, smem, the pipeline, the masking, the softmax and P@V is byte-identical.
- **Fallback (REQUIRED by the review):** the same CUDA string is compiled with `#define SPLITKV_FP8QK 1` or `0` prepended. `0` reproduces master's f16 QK code exactly: the old `qa[4][2]`, the cvt and m16n8k16, with the k-permutation comment kept in the `#else` branch. At import the loop tries fp8 first (module `muse_full_decode_splitkv_v1fp8qk`). If the build or the unchanged self-check (`err < 2e-2`, finite, same 3 configs vs FP32 ref) fails, it builds and self-checks the f16 kernel (module `muse_full_decode_splitkv_v1`). Only if that also fails does it drop to FA3. It prints `[splitkv] active path: fp8 QK (mma.sync m16n8k32 e4m3)` or `... f16 QK fallback ...`, and `SELF-CHECK FAILED (fp8qk)` when the fp8 kernel builds but is wrong.
- **Build evidence:** a new host function `splitkv_info()` (C++ `cudaFuncGetAttributes` on kernel A, in-process, no subprocess) is printed after each build. It shows SPLITKV_FP8QK, regs/thread, local (spill) bytes/thread, SASS binaryVersion and PTX version. **Deviation / limit:** native QMMA vs F2FP+HMMA emulation cannot be read in-process, because that needs `cuobjdump -sass`, which is a subprocess and a rules violation from submission.py. This node has no nvcc/cuobjdump either, so the offline SASS check was not done. Per the review, a flat kernel-A time must be recorded as "emulated-or-memory-bound (undetermined)". The regs/thread line gives a partial hint: emulation would likely raise register use relative to the f16 build (the f16 build prints its own count only when the fallback is taken).
- Comments: the kernel-A header block and the mma comments are updated. The k-permutation comment survives only in the f16 `#else` branch.
- The self-check thresholds and synthetic data are unchanged. In the fp8 build they now validate the fp8 QK path.
- Compile-time cost: the fallback build runs only on failure. The worst case is two nvcc builds at import, well inside the 15-minute budget if master's single build was ~1 min.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma diff`:

- [x] CUDA-graph capturable: the diff touches no host code in `kernel_fn` / `_full_decode_splitkv`. The new `splitkv_info()` and prints run only at import. No `.item()`/`.cpu()`/sync, no allocation or branching on input values. The fp8/f16 choice is an import-time build flag, not data.
- [x] Nothing derived from `data` is cached across calls: no new state. `_EXT` is a compiled module, and the self-check uses synthetic data only.
- [x] Weights: n/a: this problem has no weights; no prepare() repack.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. Q stays the same E4M3 bytes (now consumed natively, not widened). K and V stay FP8, accumulation is fp32 and the output is BF16. Nothing is coarser than E4M3.
- [x] Attention sees every key: the tile/item loop, masking (`pos < cseq`) and grid are unchanged, and every token's full 64-dim dot product is computed (2 k-steps x 32 dims). No skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. `load_inline` (already on master) is the only build mechanism. The SASS check via cuobjdump was deliberately NOT added.
- [x] All work happens inside the timed call; no output caching or replay. Only the build and self-check run at import, as on master.
- [x] First line `# okc: problem=muse_attention_decode` kept; no triton.autotune added.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK (its `__pycache__` was removed from the worktree).

## Result

- job `b939681d3ab549d7` (submit), status done, passed True, tests 16/16
- speedup 1.3439x, 95% CI [1.3237, 1.3641], score 2802.516509201515 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode-cuda-fp8-qk-mma.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode-cuda-fp8-qk-mma.patch`; branch `hyp/muse_attention_decode/full-decode-cuda-fp8-qk-mma`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.88 | 13.04 | 1.0384 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.93 | 28.06 | 1.0128 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 20.84 | 20.68 | 0.9986 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 123.02 | 124.53 | 0.9899 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 25.2 | 24.89 | 1.0187 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 257.75 | 261.16 | 0.986 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.63 | 17.74 | 0.9727 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.29 | 30.3 | 0.9826 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.77 | 23.34 | 0.9907 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.83 | 163.76 | 1.0014 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.35 | 29.27 | 1.0104 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 301.91 | 302.06 | 1.0007 | flat |

gained on sliding decode_1. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-cuda-fp8-qk-mma`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

