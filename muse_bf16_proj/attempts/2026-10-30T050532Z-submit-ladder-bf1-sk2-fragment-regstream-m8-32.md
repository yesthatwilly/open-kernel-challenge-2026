# muse_bf16_proj · 2026-10-30T050532Z · submit · ladder-bf1-sk2-fragment-regstream-m8-32

Hypothesis: `muse_bf16_proj/hypotheses/queue/2026-10-30T033003Z-ladder-bf1-sk2-fragment-regstream-m8-32.md` (branch `ladder/bf1`, worktree `.worktrees/ladder-bf1`; rung file on branch `campaign/public-ladder`).

- **Job:** `83cf0c487f4a45bf` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `f9593fbf` (343,871 B, sha256 5e739f0c3761796c...)

## Goal
Campaign 2026-10-30-public-ladder rung BF1 (one mechanism; user-requested): does the public sk2 data path (github.com/alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_bf16_proj.py), re-implemented on uncompressed BF16 in our file, move the m=8..32 rows by >= 1/3 of the pre-registered delta (rung file Expected: pooler m16 -4.1, unpooler m16 -3.5, pooler m32 -4.2, unpooler m32 -2.7, qkv m16/m32 -1.8/-2.1, o -0.6..-0.8, m8 same)? Read per row; the in-prepare race on the H200 is the deciding measurement.

## What changed, and where
- **Mechanism:** new section (13) `bf1_kernel` in its own extension `muse_bf16_bf1_sk_v1` (TORCH_CUDA_ARCH_LIST=9.0a, `-O3 -lineinfo -std=c++20`). prepare() permutes each raced layer's [N, K] weight losslessly into mma.m16n8k16 fragment order, stage-major (one contiguous block per CTA stage). One CTA owns R rows over all of K; a producer warp moves each stage into a 3-4 stage smem ring with one 1-D cp.async.bulk (weights) and one 3-D TMA box (x, SWIZZLE_128B, rows >= m zero-filled out of bounds); consumer warps split rows and K (intra-CTA split-K, smem FP32 reduction; not the cross-CTA split-K/stream-K of CLAIMS:42/:57), mma.sync with tokens on M. FP32 accumulate, one BF16 rounding.
- **Cells (11, own flags `bf1_*_ok`):** pooler m 8 / 9-16 / 17-32; unpooler 9-16 / 17-32; qkv_gate 8 / 9-16 / 17-32; o 8 / 9-16 / 17-32. Geometries: pooler R32 128 CTAs, unpooler R64 128 CTAs, qkv R24 96 CTAs (public NEG decode #8), o R64.
- **Routing:** `_bf1_route` in kernel_fn right after the ouns route: cfg layer/m range, n/k, x bf16 contiguous 16 B aligned, cell flag True, layer stream in state. Shapes and state only.
- **Race (prepare):** per cell `_cell_check` at m_lo / m_hi / raced m (fp64 rel L2, probe row, repeat bit-identity, error word) + per-row check, 3 read-flush `_cell_gate` rounds of kernel_fn flag on vs off; route iff pooled <= 0.97, every round <= 0.99, pooled cand IQR <= 3.0x max(inc IQR, 0.5); torch.profiler asserts exactly one `bf1_kernel`. Streams of layers with no routed cell are dropped. Deadline 200 s into the state build. `[bf1] SUMMARY` block printed after the b12m summary (per cell inc/bf1 per round, pooled, ROUTE/NO-ROUTE + reason).
- **A/B self-check:** 9 bf1 cases added to AB_CASES; `_ab_component` maps a bf1 cell to its own flag (checked first); the second state rebuilds its streams.
- **Untouched:** every other section; m=1, m>=128, router.

## Node validation (H100 gpunode06 GPU 4, nodejudge venv, takemyjob hopper/ladder-bf1 8f2a72d probes/bf1)
Run `runs/20261029-214640-gpunode06` (bytes identical but for BF1_POOLED_IQR_MAX 2.0; replicates of the final bytes: queue job 50-h100-ladder-bf1-val-replicate).
- Build 34.8 s; import of all units 227 s; prepare 13.0 s (bf1 2.4 s); regs <= 100, 0 spill, 1 CTA/SM.
- check_implementation: 22 shapes x 3 inputs (outliers incl.) through graph replay, all pass, probe rows exact, error word 0.
- A/B bf1/inc pooled (identity inc2 / bf1B): pooler m16 0.941 (0.997/0.945), m32 0.929 (0.997/0.928), unpooler m16 0.956 (1.000/0.955), m32 0.978, qkv m16 0.973, m32 0.832 (1.000/0.836); losses: pooler m8 1.229, qkv m8 1.333 (12-bit b12m keeps them), o 1.03-1.08.
- In-prepare race routed pooler 9-16, unpooler 9-16, qkv 17-32 (pooler 17-32 refused at IQR 2.0 -> rule set to 3.0 in the commit).

## Rules check (each must be true before the run; checked against `git -C .worktrees/ladder-bf1 diff ae98caf0 f9593fbf`)
- [x] CUDA-graph capturable: kernel_fn adds only `_bf1_route` (cfg, shapes, dtype, contiguity, `data_ptr() % 16`, state flags) and `_bf1_forward` (`torch.empty` + one extension launch; the x tensor map is a host-only cuTensorMapEncodeTiled on this call's x, stack-local, no stream op). Every `.item()`/synchronize is in prepare-only helpers (`_bf1_prepare`, `_cell_*`, `_b12m_rows`). Capture + replay validated on the H100.
- [x] Nothing derived from `data` is cached: the tensor map is rebuilt per call; `_BF1_SUM` / `_BF1_BUILD` hold prepare text and build state only.
- [x] Weights: a lossless permutation of the BF16 weights (same values, same precision); FP32 accumulation, one BF16 rounding.
- [x] Activations stay BF16 (x read as bf16, output bf16). KV-cache/query item n/a.
- [x] Attention items n/a.
- [x] No timing patches, harness frames/I/O, module edits, reference import, subprocesses or network: `load_inline` only.
- [x] All work inside the timed call: one kernel per routed call, no output caching or replay.
- [x] First line `# okc: problem=muse_bf16_proj` kept; py_compile OK, __pycache__ removed.
- [x] Size 343,871 B < 1,000,000; prepare 13 s on the H100 (master ~10 s), compile +35 s (import 227 s vs 191-197 s; 10-minute run budget).
- [x] No third-party code copied; design cited as github.com/alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_bf16_proj.py; no Nathen Fernandes (MIT) code.

okc_lint: `OK with warnings` (all warnings pre-existing prepare-side host-sync/dtype items; no errors).

## Risks / what could fail
medium.
- H200 transfer: the H100 incumbents sit near the 3.1 TB/s read ceiling (max ~10% gain there); the H200 decides each cell. A cell that does not clear 0.97 there keeps its incumbent (no loss beyond race noise).
- Memory layout (public NEG #11): routed layers keep an extra stream (pooler/unpooler 67 MB each, qkv 38 MB) in the state; unrouted cells' streams are freed. Watch SASS-identical rows (m=1, m>=128) for 0.5-1 us drift.
- Compile time +35 s toward the 10-minute run limit (import was 227 s on the H100 node).
- A/B self-check adds 9 cases (~0.1 s).


## Why a scored run
Check 8c915678ba4f4b1b scored 1.312x [1.293, 1.332] against the board 1.2839 (+2.2%, past the 1.2% margin); same bytes (f9593fbf). The user approved scored submit + promotion for ladder checks that beat the board (2026-10-30 ~05:05Z).

## Result (after the run)
- job `83cf0c487f4a45bf` (submit), passed, tests 9/9, **1.310x [1.291, 1.330]** vs board 1.2839 -> landed on master.
