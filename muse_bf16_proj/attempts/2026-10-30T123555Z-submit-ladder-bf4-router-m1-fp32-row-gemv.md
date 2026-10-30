# muse_bf16_proj · 2026-10-30T123555Z · submit · ladder-bf4-router-m1-fp32-row-gemv


- **Job:** `89d3505130b443ea` (`okc status <job>`)
Hypothesis: `muse_bf16_proj/hypotheses/queue/2026-10-30T094751Z-ladder-bf4-router-m1-fp32-row-gemv.md` (branch `ladder/bf4`).

- **Job:** `<job>` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `aad820f8` (369,323 B, sha256 f686c6926b552df0...), on master 8f6c8d92 (BF5 landed, board 1.318)

## Goal
Campaign 2026-10-30-public-ladder rung BF4 (one mechanism; user-requested). Does a non-codec, row-per-CTA FP32 register GEMV for the router at m=1 move the decode_1 router row by at least 1/3 of the pre-registered -1.2 us? The mechanism follows the public rgemv (github.com/alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_bf16_proj.py), re-implemented here, not copied. The public reports 2.5 us on the H200 for this kernel, against 3.7 us for our router_kernel<8,8> (FINDINGS:73). The hit bar is a decode_1 router delta of at least -0.4 us. Read the [rv] line for the in-prepare race on the H200.

## What changed, and where
- **Mechanism:** new section (15), `router_rv_kernel<512>`, in its own extension `muse_bf16_router_rv_m1_v1` (9.0a, -O3, compiled at import in ~32 s). Grid: 256 CTAs x 512 threads, one weight row per CTA. Each thread issues 4 x 16 B ld.global.nc.L1::no_allocate loads of the FP32 weight before any FMA. x is upcast exactly. Accumulation is an FP32 FMA chain in k order, then a warp xor butterfly, then a fixed-order cross-warp sum in smem, so the result is deterministic. It reads the FP32 router weight as given: no hi/lo split, no repack, no codec. One plain launch: no cluster, no memset.
- **Routing:** `_rv_route` sits in kernel_fn right before the router-cluster route. It checks the flag `router_rv_ok`, that cfg m == 1, n == 256 and k == 8192, and that x is bf16 [1, 8192] with stride(1) == 1 and 16 B alignment. It branches on shapes and state only.
- **Race (prepare):** `_cell_check` through kernel_fn (fp64 rel L2 <= 2e-5, one-hot probe row, run 2 bit-identical). Then 3 read-flush `_cell_gate` rounds with the flag on vs off, judged by `_bf1_rule` (pooled <= 0.97, every round <= 0.99, IQR). A torch.profiler check asserts exactly one router_rv_kernel. It falls back to router_kernel<8,8>. `[rv]` summary line is printed after the bf5 block.
- **A/B self-check:** the router m=1 case (already in AB_CASES) maps to `router_rv_ok` via `_ab_component`, and the second state inherits the flag.
- **Untouched:** router m=2..8 (cluster), m>=9, and every bf16 layer.

## Node validation (H100 gpunode06 GPU 5, takemyjob hopper/ladder-bf4 probes/bf4)
- Dev sweep `runs/20261030-024437-gpunode06` and `-024657-gpunode06`, rv/inc pooled: 256 thr 0.77, 512 thr 0.65-0.67, 1024 thr 0.67. m=8 variants ran 1.66-2.2x slower, so they are not routed.
- Pre-rebase bytes `runs/20261030-025033-gpunode06` and final bytes `runs/20261030-052313-gpunode06`. Both: 22 shapes x 3 inputs (outliers included) pass check_implementation through graph replay, bad=0. The race routed in both runs (pooled 0.630 / 0.664). Prepare took 12.4 / 13.5 s and import 258 / 263 s.
- A/B on the final bytes, pooled net us: seed 3000 inc 4.93 / inc2 4.83 (0.981) / rv 3.14 (0.636) / rvB 3.26 (0.662); seed 9001 inc 4.80 / inc2 4.77 (0.993) / rv 3.10 (0.647) / rvB 3.17 (0.660).
- Precision: max |out - fp64| at m=1 is 4.8e-7, against 2.4-3.6e-7 for F.linear fp32 and 7.8e-6 for today's hi/lo route. The output is NOT bitwise today's route. It is within SGLang's own FP32 error, and more accurate than the incumbent.

## Rules check (against `git diff 8f6c8d92 aad820f8`)
- [x] CUDA-graph capturable: kernel_fn adds `_rv_route` (cfg, shapes, dtype, strides, `data_ptr() % 16`, state flag) and `_rv_forward` (`torch.empty` plus one launch). Every sync and `.item()` is in prepare-only helpers. Capture and replay were validated (graph == direct, bitwise).
- [x] Nothing derived from `data` is cached.
- [x] Router weight: the FP32 weight runs as given. This is SGLang's own m <= 8 math (F.linear FP32), with FP32 accumulation (rules :107-108, baseline.py :37-64). No lossy change and no codec.
- [x] Activations: x is BF16 input, upcast exactly. Output is fp32 as in the reference.
- [x] Attention items n/a.
- [x] No timing patches, harness frames, module edits, reference import, subprocesses or network: `load_inline` only.
- [x] All work happens inside the timed call: one kernel per call, no output caching.
- [x] First line `# okc: problem=muse_bf16_proj` kept. py_compile OK, __pycache__ removed. okc_lint: `OK with warnings` (all pre-existing prepare-side items).
- [x] Size 369,323 B < 1,000,000. Prepare 13.5 s on the H100. Import +32 s (263 s).
- [x] No third-party code copied. The design is cited as github.com/alygg77/okc-muse-spark-kernels@6cae49a kernels/muse_bf16_proj.py rgemv. No Nathen Fernandes (MIT) code.

## Risks / what could fail
Low.
- H200 transfer: the H100 measured -1.7 us (0.64x). The H200 prediction (-1.2 us) comes from the public's measured 2.5 us vs our 3.7 us, not from the H100 ratio. No codec is involved, so the ALU cost is 1 FMA per 4 bytes. If the race does not clear 0.97 on the judge, the incumbent stays.
- Import time is +32 s toward the run limit.
- Rows with identical SASS may drift by about 0.5 us (layout).


## Why a scored run
Check e9975aa1 1.332x beats the board 1.318, same bytes aad820f8. Standing rule: scored submit + promotion.

## Result (after the run)
- job `89d3505130b443ea` (submit), passed, **1.321x [1.301, 1.341]** vs board 1.318 -> landed on master (bytes aad820f8).
