# muse_bf16_proj · 2026-10-29T213649Z · submit · b12-batched-escape-m8-32-race

Hypothesis: `muse_bf16_proj/hypotheses/done/gain/2026-10-29T200122Z-b12-batched-escape-m8-32-race.md` (branch `hyp/muse_bf16_proj/b12-batched-escape-m8-32-race`).

- **Job:** `14ea7be493204308` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `0684d0c2`

## Goal
Promote check-proven gain b12-batched-escape-m8-32-race (1.2974x [1.2836,1.3111], check job bced57b42b87444d) to the board.

Add bulk12 kernels for the m=8/16/32 cells of pooler, qkv_gate (and o if it fits), each raced per cell in prepare against master's incumbent (swap-AB Triton / cuBLAS paths), routed only if bit-exact to its own fp32-accumulate contract within the problem tolerance as master's b12 m=1 cells are, and <= 0.97x the incumbent in every round of a read-flushed interleaved race (reuse master's bulk12 read-flush gate machinery, CLAIMS:59). Kernels from takemyjob hopper/b12tma-1029 d95798e (probes/b12tma; research/h100-b12tma-1029.md): (a) the cp.async bridge b12x with the end-of-kernel serial escape-x gather replaced by batched loads, and (b) uc1 = TMA-box b12 without clusters, same batched escape gather, plain mbarrier.arrive.shared::cluster (no arrive.release.cluster / MEMBAR.ALL.GPU). Reuse master's lossless 12-bit codes where R matches (pooler R=32, qkv R=16 are already encoded for m=1) to save prepare time and memory. Print [b12m] RACE/ROUTE lines per cell (incumbent us, each arm us and ratio per round) in a <= 2 KB summary at the end of prepare.

## What changed, and where
**Mechanism:** pooler and qkv_gate at exactly m=8 run a new 12-bit row-owner kernel (b12m_tma_kernel, uc1_g1: TMA-box x without a cluster, local mbarrier arrives, batched escape-x gather). It lives in its own extension and routes only behind a read-flush race against master's swap-AB incumbent.

- **Where:** worktree `muse_bf16_proj/submission.py`, new section (12) after section (10) (+1049 lines, nothing removed). New extension `muse_bf16_b12m_m8_tma_v1` (TORCH_CUDA_ARCH_LIST=9.0a, `-O3 -lineinfo -std=c++20`). The m=1 `_B12_CUDA` unit is not touched or recompiled. Device code comes from takemyjob `hopper/b12tma-1029` d95798e / `hopper/b12dens-1029` 6d80717 `b12t_kernel` at CL=1 (cluster, multicast, AR and INS code stripped), and from a7da6fa9 / 2980f57a `b12x_kernel` for the bridge (GA0 is byte-for-byte the judged gather). Both kernels share one epilogue helper.
- **Arms:**
  - Routable: pooler m8 `uc1_g1<R32,S6,MINB2,CW8,NT8>`, qkv m8 `uc1_g1<R16,S4,MINB2,CW4,NT8>`.
  - Non-routable diagnostics: bridge GA0 and GA1 at pooler m8 and qkv m8 (extra `_cell_gate` arms, which print GA0-GA1), and pooler m16 `uc1_s8_g1` as a timed-only cell.
  - Dropped: m=32 cells and o (review fix).
- **Routing:** exact shape only (`_b12m_route`). It requires `(layer, cfg m)` in B12M_CELLS, `x.shape[0] == m`, bf16, contiguous, 16 B aligned and a row stride divisible by 16 B. The `uc1` tensor map is encoded inside `b12m_run` from `x.data_ptr()` on every call. The encode is a host-only `cuTensorMapEncodeTiled` resolved through `cudaGetDriverEntryPointByVersion`, it is never cached, and only the driver function pointer is static.
- **Gate (review fix: 0.95, not 0.97):** each arm runs `_cell_check` (fp64 rel L2, probe row, repeat bit-identity, error word), then `_b12m_rows` (every row's rel L2 <= 1.5e-2). After that come 3 read-flush `_cell_gate` rounds of kernel_fn with the flag on vs off (master's routed path, like for like) and 1 write-flush round as a diagnostic. A cell routes only if the pooled median ratio is <= B12_MAX_RATIO_M8 (0.95), every round is <= 0.97, and the pooled cand IQR is <= 2x max(incumbent IQR, 0.5 us). It must then pass a strict profiler assertion: exactly one `b12m_tma_kernel` (no profile means no route).
- **Codes (review fix):** a cell reuses the m=1 state's `b12_wc_/esc_/base_` only when that instance has the same R. Otherwise it encodes its own codes through `_b12_encode_checked` (host + device round trip) under `b12m_*` keys. The source, R, base, escapes total / max per CTA and extra MB are printed. Codes are popped when no cell of the layer routes.
- **Error word:** prepare-only. kernel_fn passes it and never reads it. Bounded mbarrier waits (0.5 s), and every launch is announced and synced in the checks.
- **A/B and summary:** qkv_gate m=8 was added to AB_CASES. `_ab_component` maps (pooler|qkv_gate, 8) to its own b12m flag, and the second state reloads codes. A <= 2 KB `[b12m]` SUMMARY prints after the DECISIONS block, as the last lines of prepare. It carries build s, prepare s, codes, per-arm regs/spill/smem/occupancy/worst-row, per-round inc/arm/ratio/IQR, write flush, pooled ratio, GA0-GA1, ROUTE/NO-ROUTE + reason, and PATH CHOSEN.
- **Size:** 309,225 B (cap 1,000,000).
- **Deviation:** none from the narrowed spec. The m=1 RACE lines are not re-printed, because the m=1 unit is a separate extension and is unchanged.

### Node validation (gpunode06 GPU 6, H100 SXM, nodejudge venv torch 2.14.1+cu130, 2026-10-29 20:35-20:39Z)
The same bytes (sha256 7837d1e422852f91) went through guarded `hopper-probe` as the operator, with node-local staging that was unstaged after the run. Run dir: takemyjob `~/git/.takemyjob-wt/b12mval/prototypes/hopper-probe/runs/20261029-133523-gpunode06/` (branch hopper/b12mval-1029, uncommitted).

- **Build and prepare:** the b12m extension compiled in 37.0 s. Import with every load_inline unit took 191 s, and prepare took 9.9 s. Register and spill counts: uc1 r48 (both cells), uc1_s8 r55, bridge r36-48, all with 0 spill and the needed occupancy (2/1 CTA/SM).
- **Correctness:** every arm passed (rel L2 1.64-1.66e-3, worst row 1.7e-3, probe rows exact, error word 0). The judge's `check_implementation` passed on 10 shapes through graph replay with refilled inputs, outliers included. The A/B check passed at pooler and qkv m8.
- **Codes:** pooler reused the m=1 codes (escapes 4559, max 51 per CTA). qkv encoded its own (escapes 2661, max 30 per CTA, +28.6 MB), because its m=1 cell picked another R.
- **Race on the H100** (read flush, net us, inc/arm):
  - P8: 24.0/18.6-18.75, pooled ratio **0.774**, GA0-GA1 3.90 us. ROUTE.
  - Q8: 15.1/11.5, pooled ratio **0.764**, GA0-GA1 1.95 us. ROUTE.
  - P16 (diag): pooled 0.863.
  These absolute numbers do not transfer to the H200. The b12dens projection is 0.939 / 0.906. The judge's own gate decides.

## Rules check (each must be true before the run)
Each item was checked against `git -C .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race diff`.
- [x] CUDA-graph capturable: kernel_fn adds only `_b12m_route` (cfg, shapes, dtype, contiguity, `data_ptr() % 16` and state flags) and `_b12m_forward` (`torch.empty` plus one extension launch whose host-side tensor-map encode is not a stream op). Every `.item()` and synchronize sits in prepare-only helpers (`_b12m_rows`, `_b12m_prepare`). The graph capture and replay were validated on the H100.
- [x] Nothing derived from `data` is cached: the tensor map is a stack-local built on each call from that call's x, and `_B12M_SUM` and `_B12M_BUILD` hold only prepare text and build state.
- [x] Weights: a lossless 12-bit codec (section (10)'s `_b12_encode`, host + device round trip bit-exact), with escapes holding the true bf16 bits. Accumulation is fp32, followed by one bf16 rounding.
- [x] Activations stay BF16 (x is read as bf16, the output is bf16). The KV-cache / query item is n/a: this is not attention.
- [x] Attention full window: n/a, not attention.
- [x] No timing patches, harness frames or I/O, module edits, reference import, subprocesses or network: `load_inline` only (as in master), and the reference is imported only by the node driver outside the submission.
- [x] All work happens inside the timed call: one kernel per m=8 call, with no output caching or replay.
- [x] The first line `# okc: problem=muse_bf16_proj` is kept. py_compile is OK and __pycache__ was removed.

okc_lint: `OK with warnings: .../muse_bf16_proj/submission.py`. All 58 warnings are host-sync/dtype items "outside kernel_fn's call graph; fine in prepare()". There are no errors.

## Risks / what could fail
medium

## Result (after the run)
(pending)
