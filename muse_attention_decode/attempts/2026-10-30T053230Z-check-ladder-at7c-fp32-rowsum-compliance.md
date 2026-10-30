# muse_attention_decode · 2026-10-30T053230Z · check · ladder-at7c-fp32-rowsum-compliance

- **Job:** `fefe31b8de02492b` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `1b5b1346` (branch ladder/at7c = AT7 98be5aac + AT7c; base 91d0594e = fvg r2-cf aaae1769)

## Goal
This supersedes the AT7 98be5aac check text for compliance. AT7c adds the user-required FP32 softmax denominator from
the unrounded P (as FA3). On the H100 that is also faster than AT7: FP8-PV mode -3.5..-4.1 us at v16f and
-4.9..-5.1 us at v32f against AT7, and against fvg 0.936-0.942 / 0.954-0.955 (run 20261029-222131, 2 seeds; math-only
-10%). The end-to-end import routes mode 5 on both rows: 0.766 / 0.773-0.776 of master (run -222616). Error per arm is
equal to or lower than AT7's: judge rule 2.808e-2 / 2.959e-2 against 0.083 / 0.085.

The original AT7 goal follows.

Campaign 2026-10-30-public-ladder, rung AT7. One mechanism: FP8 PV. The fvg full-verify kernel computes P·V with a
wgmma m64n64k32 e4m3 instead of f16. P is rounded to E4M3 exactly as SGLang's FA3 FP8 path does it, and V stays its FP8
cache bytes. Rows: `layer=full load=verify_16` and `load=verify_32`.

H100 results (gpunode06 GPU 3, 15 rounds x 2 seeds), measured against fvg:
- v16f: -5.0 / -5.5 us (0.965)
- v32f: -7.1 / -7.4 us (0.973)
- math-only: -6%
- The worst round (0.977) is clear of the identity arm (best round 0.992).

Expected on the judge: v16f about 148 -> 143 us and v32f about 278 -> 270 us. Against master that is about
0.87-0.88 / 0.89-0.90 per row, against fvg's own 0.903-0.916 / 0.918-0.924.

**Attribution.** This check ships fvg itself (the [shelf] route, CLAIMS:87) plus FP8 PV. That is fine for attribution:
fvg's own judge ratio is already known from two checks (184991095dff4d50 and a47caf0724de40ad: v16 0.903-0.916, v32
0.918-0.924). Any per-row ratio clearly below those ranges is FP8 PV's contribution. The [fvg] import summary prints
which mode each row routed: 1 = f16 PV, 5 = FP8 PV.

## What changed, and where
- `_FVG_CUDA`: `fvg_kernel<MODE, PV8>`. Launch mode bit 4 selects PV8; mode 1 is unchanged fvg (bitwise equal on the
  node).
  - PV8 computes P with fvg's f16 exp2 and FA3's FP8 Max_offset 8.
  - It packs P with `cvt.rn.satfinite.e4m3x2.f16x2` into the k32 A fragment, keys permuted per 16-key group.
  - Each warpgroup stages its head's V^T (K-major, no-swizzle, LBO 128 / SBO 528) from the cp.async ring with
    LDS.64 + PRMT + STS.32.
  - It issues 2 x `wgmma m64n64k32.f32.e4m3.e4m3` per tile per head.
  - Row sums stay on the unrounded P.
  - Mode 5 uses 115 regs (mode 1: 128), 0 B local, 2 CTAs/SM, the same smem.
- `kfn` / `set_smem` dispatch: 8 template instances.
- `_FVG_ROUTE` maps batch -> launch mode. `_fvg_forward` launches the row's mode.
- `_fvg_row` (in-prepare race) now runs per arm (modes 1 and 5):
  - the 3 synthetic draws, with the judge-rule error check against shared fp64 references;
  - the replay check;
  - interleaved graph rounds against master.
  - Mode 5 is taken only if it is <= 0.95 of master AND faster than mode 1 in every round. Any miss falls back to
    mode 1, then to master.
- `_fvg_setup`: attributes checked for mode 5 (no spill, 2 CTAs/SM), or the arm is dropped.
- `_fvg_diag`: adds the FP8 PV kernel, mainloop and math-only lines (H200 transfer data).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()` on the routed path. The mode is a frozen
      per-batch dict set in prepare, and the launch is config/shape-only. (The race's syncs are import-time only, as in
      fvg.)
- [x] Nothing derived from `data` is cached across calls. Lens and page table are read on device every call; the
      partial buffers are allocated from config and SM count only.
- [x] Weights: n/a (muse_attention_decode has no weights).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3.
  - **P precision: E4M3 for the PV product (mode 5), as FA3.** In mode 1 (fallback) P is f16, which is finer.
  - **Softmax denominator precision: fp32, summed from the unrounded fp32 P, as FA3** (sgl-flash-attn 141fa063
    softmax.h:181-184 `reduce_sum` over the fp32 scores before mainloop:1360-1361 rounds P). P is computed with
    `ex2.approx.ftz.f32` per score (FA3: `exp2f`); FA3's Max_offset 8 applies in mode 5.
  - The combine and the output are fp32, then the bf16 output (as before).
  - P is rounded to E4M3 for the PV product, which is exactly what SGLang's kernel does: sgl-flash-attn 141fa063
    (SGLang's FA3 pin) mainloop_fwd_sm90_tma_gmma_ws.hpp:1360-1361 `tOrP = make_tensor_like<Element>(tOrP_acc);
    convert_type_out(...)` with Element = float_e4m3_t when q is e4m3 (flash_api.cpp:63, :322), then the PV GEMM at
    :1386.
  - V is used as its stored FP8 bytes, and the scale is 1 on the gated rows.
  - The output is bf16, as before.
- [x] Attention sees every key. The flat partition is unchanged; every tile of every request is consumed.
- [x] No top-k, sparse, block-skip or weight-based skipping. P is never thresholded; E4M3 underflow of tiny weights is
      FA3's own behaviour, with the same Max_offset 8.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.

## Risks / what could fail
- **Precision.** Judge-rule error 2.81e-2 / 2.96e-2 against bounds 0.083 / 0.085 on the judge's own generator (fvg:
  2.72e-2 / 2.78e-2). That is a large margin.
- **Race.** The end-to-end import on gpunode06 (AT7 run 20261029-221206; AT7c run 20261029-222616: 0.765-0.766 / 0.773-0.776 of master) routed mode 5 on both rows: 0.783-0.790 / 0.785-0.787
  of master in every round, against mode 1 at 0.813-0.819 / 0.806-0.810. reference.check_implementation passes on both
  seeds. The race falls back to mode 1, then to master, on any miss.
- **H200 transfer of the gain is untested.** The mechanism lowers math-only by 6% on the H100. The H200 is math-bound
  at v32 (diag: math-only 233 / mainloop 265 us).
- **Import time.** One more arm adds about 3 eager launches, 1 replay check and 1 graph per row (a few seconds). The
  build is unchanged in kind: 8 instances instead of 4, about 36 s in the background thread on the H100. The import clock read 268.9 s on the H100 (the build joins after master's modules, as before).

## Result (after the run)
Pass/fail, first failing case if any, speedup with interval, profile file, what to try next.

## Result
- job `fefe31b8de02492b` (check), passed 16/16, **1.676x [1.651, 1.702]** vs board 1.634 (+2.6%): v16f 138.3 us, v32f 254.7 us.
