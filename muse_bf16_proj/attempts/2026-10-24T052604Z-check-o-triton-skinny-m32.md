# muse_bf16_proj · 2026-10-24T052604Z · check · o-triton-skinny-m32

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225303Z-o-triton-skinny-m32.md` (branch `hyp/muse_bf16_proj/o-triton-skinny-m32`).

- **Job:** `d75669d46acb42a5` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `7dec83c`

## Goal
Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'o' at m <= 32 (n=8192, k=1024, 16.8 MB): in kernel_fn route o at m <= 32 to the same kernel with BLOCK_N=64 (128 programs), BLOCK_K=256 (only 4 K-steps, so num_stages=4 covers the whole loop), BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 8192), bf16). o at m >= 128 and every other layer keep what they have. Kernel moved: nvjet_sm90_tst_64x8/64x16/64x32_64x16_2x1_v_bz_TNT, 4.5-5.1 us of GPU time inside a 6.6-7.0 us reported shape against a 3.5 us floor; a single launch already, so like unpooler-full this isolates bandwidth on a short-K weight where each program's stream is only 128 KB.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/o-triton-skinny-m32`, file `muse_bf16_proj/submission.py` only (uncommitted, base d2840d0). Diff: 1 file, +51 / -2. Follows the sibling unpooler-full-triton-skinny-m32 pattern exactly.

- New constants `O_SKINNY_MAX_M=32, O_N=8192, O_K=1024, O_BLOCK_N=64` (128 programs), `O_BLOCK_K=256` (4 K steps), `O_NUM_WARPS=4`, `O_NUM_STAGES=4`. Smem: 4 x (64x256 + BLOCK_M x 256) x 2 B = 160 KB at BLOCK_M 16, 192 KB at 32 (same footprint as the sibling's unpooler tile, under H200's 227 KB).
- `_o_skinny_applies(x, cfg)`: guards on cfg m <= 32, cfg n == 8192, cfg k == 1024, plus x 2-D, x.shape == (m, k), x.stride(1) == 1, x bf16. Never keys on x.shape[0] alone; the m=33 o test and m=128/256 stay on `_bf16_gemm_dispatch_impl`.
- `_o_skinny_gemm`: reuses the unchanged `_pooler_skinny_kernel` (no kernel edits), BLOCK_M 16 for m <= 16 else 32, fp32 accumulate, bf16 store into `torch.empty((m, 8192), bf16)`.
- `_o_skinny_warmup`: runs every m in 1..32 on synthetic zeros in prepare(); no try/except, so a compile or shared-memory failure raises in prepare() (no silent cuBLAS fallback). The only fallback is the static weight-shape check `state["o_skinny"] = o.shape == (8192, 1024)`, decided in prepare() from the weight, mirroring the sibling.
- kernel_fn: `layer == "o" and state["o_skinny"] and _o_skinny_applies(x, cfg)` routes to the Triton path; everything else unchanged.
- Deviation: none. All four rows m=1/8/16/32 are routed as specified; the review's suggestion to narrow to m=1 was conditional on the unpooler-full harvest and is a new scope decision, not an implementation detail, so it was not applied.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/o-triton-skinny-m32 diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in the diff; branches only on cfg m/n/k, x shape/stride/dtype and a prepare()-time bool; `torch.empty` of a fixed shape-derived size, same as parent's pooler path.
- [x] Nothing derived from `data` is cached across calls (state gains only `o_skinny`, a bool from the weight shape).
- [x] Weights: o used as stored (bf16, no repack, no precision change); router split unchanged.
- [x] Activations BF16 where SGLang keeps BF16 (x bf16 in, bf16 out, fp32 accumulate); KV cache / queries: n/a: no attention in this problem.
- [x] Attention sees every key: n/a: dense GEMM problem, no attention; full K reduction (4 x 256 = 1024).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network; no new imports, no triton.autotune.
- [x] All work happens inside the timed call; warmup outputs in prepare() are synthetic zeros and dropped; no output caching or replay.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` OK; __pycache__ removed.

## Risks / what could fail
medium
Smallest absolute upside of the family (at most ~1.5 us per shape, four shapes), and the reported-minus-kernel gap of ~2 us is graph-node latency this change cannot touch, so expect a small effect that may sit inside the load CI; read the per-shape o rows. Short K (4 steps of 256) means a program's pipeline barely fills; BLOCK_K=128 with num_stages=8 is the ab-tune. Tests touched: m=33 o (stays on cuBLAS since 33 > 32). Same correctness and graph properties as the parent.

## Result (after the run)
- job `d75669d46acb42a5` (check), status done, passed True, tests 9/9
- speedup 1.0057x, 95% CI [0.9906, 1.0208], score 4803.459598340808 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1510295903072447, "ci": [1.133764146452636, 1.1682950341618532], "job": "bd9fa14151644cee"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-triton-skinny-m32.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/o-triton-skinny-m32.result.json`, `muse_bf16_proj/hypotheses/done/neutral/o-triton-skinny-m32.patch`; branch `hyp/muse_bf16_proj/o-triton-skinny-m32`
