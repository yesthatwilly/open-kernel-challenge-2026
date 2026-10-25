---
problem: muse_bf16_proj
slug: bf16-verify-deepgemm-bf16-gemm-nt-route
created: 2026-10-25T011408Z
base: master
area: bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)
change: >
  Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit
  at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=128 nvjet 96x64 splitK 13.0 +
  cublasLt splitKreduce 3.0 us (two nodes, row 16.81, floor 8.4), m=256 nvjet 96x64 19.4 (row
  19.62, 498 TFLOP/s, floor 9.8); pooler m=128 25.7 (row 25.55, floor 14.6), m=256 29.5 (583
  TFLOP/s, floor 17.4); o m=128 5.8, m=256 8.3; unpooler m=128 20.4, m=256 24.6 (~700 TFLOP/s).
  Our own kernels have lost here three ways (Triton 128x128 split-K, Triton TMA+warp-spec, CUTLASS
  stream-K: qkv m=256 28.5 us, every m=256 row worse), and prepare-time algo picking is a dead
  knob. Untried: DeepGEMM, which the image ships (its include dir was found by the stream-K build:
  deep_gemm/include with CUTLASS 4.2.1) and whose recent releases have dense BF16 GEMMs
  (deep_gemm.bf16_gemm_nt and friends) with analytic block/multicast/SM-count heuristics tuned for
  exactly these skinny-M, deep-K Hopper shapes, no timing search. Change: at import, try `import
  deep_gemm`; print deep_gemm.__version__ (or __file__) and whether hasattr(deep_gemm,
  'bf16_gemm_nt'); if the module or the function is missing print '[deepgemm] unavailable, master
  kept' and change nothing. In prepare(), for each of qkv_gate, o, pooler, unpooler and m in {128,
  256}: allocate out = torch.empty(m, n, bf16), call deep_gemm.bf16_gemm_nt(x, W, out) on a
  synthetic bf16 x (this also triggers DeepGEMM's JIT compile, untimed; print the seconds),
  torch.cuda.synchronize(), then self-check against x.double() @ W.double().T: rel L2 <= 3e-3, a
  one-hot power-of-two probe row within 1e-4, finite. Wrap each layer in try/except; a failure
  prints '[deepgemm] <layer> DISABLED: <reason>' and that layer keeps master. Also run m=130 for
  unpooler and m=200 for each layer through the self-check so the 65..256 route is covered. Record
  state['dg_ok'][layer]. In kernel_fn add one branch just before master's cuBLAS fallback: if
  state['dg_ok'][layer] and 65 <= cfg m <= 256 and cfg n is the full width and x is contiguous
  bf16 [m, k], out = torch.empty(m, n, bf16); deep_gemm.bf16_gemm_nt(x, W, out); return out. If
  the signature differs in the installed version (e.g. it takes (a, b, d, c=None) or tuples),
  adapt to it and print the signature used. Weights are read in place (W [n, k] row-major is
  DeepGEMM's NT 'b'), no repack, no precision change. m <= 32 paths, the router, and the
  m=512/1000 tests are unchanged. Print '[deepgemm] PATH CHOSEN <layer>: deepgemm|master'. At
  harvest, read the PATH CHOSEN lines and confirm a deep_gemm kernel in top_kernels on the
  m=128/256 rows before reading any number. Check DeepGEMM's launch is CUDA-graph capturable (no
  host sync, no allocation inside it once JIT-compiled): run one torch.cuda.graph capture + replay
  per layer in prepare and disable the layer if capture raises.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: one node instead of nvjet splitK 13.0 + splitKreduce 3.0 us; row 16.81 vs floor 8.4
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: nvjet 19.4 us at 498 TFLOP/s (36-72 tiles on 132 SMs); floor 9.8
  - shapes: layer=pooler load=verify_*
    direction: faster
    why: m=128 nvjet 25.7 at ~2.6 TB/s (floor 14.6); m=256 29.5 at 583 TFLOP/s (floor 17.4)
  - shapes: layer=o load=verify_*
    direction: flat
    why: o is short-K (1024); a gain is possible at m=256 (8.3 us vs 4.5 floor) but not expected
  - shapes: layer=unpooler load=verify_*
    direction: flat
    why: m=256 nvjet already ~700 TFLOP/s; m=128 20.4 vs 14.6 floor may move
  - shapes: layer=* load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# bf16-verify-deepgemm-bf16-gemm-nt-route

## Change

Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=128 nvjet 96x64 splitK 13.0 + cublasLt splitKreduce 3.0 us (two nodes, row 16.81, floor 8.4), m=256 nvjet 96x64 19.4 (row 19.62, 498 TFLOP/s, floor 9.8); pooler m=128 25.7 (row 25.55, floor 14.6), m=256 29.5 (583 TFLOP/s, floor 17.4); o m=128 5.8, m=256 8.3; unpooler m=128 20.4, m=256 24.6 (~700 TFLOP/s). Our own kernels have lost here three ways (Triton 128x128 split-K, Triton TMA+warp-spec, CUTLASS stream-K: qkv m=256 28.5 us, every m=256 row worse), and prepare-time algo picking is a dead knob. Untried: DeepGEMM, which the image ships (its include dir was found by the stream-K build: deep_gemm/include with CUTLASS 4.2.1) and whose recent releases have dense BF16 GEMMs (deep_gemm.bf16_gemm_nt and friends) with analytic block/multicast/SM-count heuristics tuned for exactly these skinny-M, deep-K Hopper shapes, no timing search. Change: at import, try `import deep_gemm`; print deep_gemm.__version__ (or __file__) and whether hasattr(deep_gemm, 'bf16_gemm_nt'); if the module or the function is missing print '[deepgemm] unavailable, master kept' and change nothing. In prepare(), for each of qkv_gate, o, pooler, unpooler and m in {128, 256}: allocate out = torch.empty(m, n, bf16), call deep_gemm.bf16_gemm_nt(x, W, out) on a synthetic bf16 x (this also triggers DeepGEMM's JIT compile, untimed; print the seconds), torch.cuda.synchronize(), then self-check against x.double() @ W.double().T: rel L2 <= 3e-3, a one-hot power-of-two probe row within 1e-4, finite. Wrap each layer in try/except; a failure prints '[deepgemm] <layer> DISABLED: <reason>' and that layer keeps master. Also run m=130 for unpooler and m=200 for each layer through the self-check so the 65..256 route is covered. Record state['dg_ok'][layer]. In kernel_fn add one branch just before master's cuBLAS fallback: if state['dg_ok'][layer] and 65 <= cfg m <= 256 and cfg n is the full width and x is contiguous bf16 [m, k], out = torch.empty(m, n, bf16); deep_gemm.bf16_gemm_nt(x, W, out); return out. If the signature differs in the installed version (e.g. it takes (a, b, d, c=None) or tuples), adapt to it and print the signature used. Weights are read in place (W [n, k] row-major is DeepGEMM's NT 'b'), no repack, no precision change. m <= 32 paths, the router, and the m=512/1000 tests are unchanged. Print '[deepgemm] PATH CHOSEN <layer>: deepgemm|master'. At harvest, read the PATH CHOSEN lines and confirm a deep_gemm kernel in top_kernels on the m=128/256 rows before reading any number. Check DeepGEMM's launch is CUDA-graph capturable (no host sync, no allocation inside it once JIT-compiled): run one torch.cuda.graph capture + replay per layer in prepare and disable the layer if capture raises.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: one node instead of nvjet splitK 13.0 + splitKreduce 3.0 us; row 16.81 vs floor 8.4
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 19.4 us at 498 TFLOP/s (36-72 tiles on 132 SMs); floor 9.8
- `layer=pooler load=verify_*` faster: m=128 nvjet 25.7 at ~2.6 TB/s (floor 14.6); m=256 29.5 at 583 TFLOP/s (floor 17.4)
- `layer=o load=verify_*` flat: o is short-K (1024); a gain is possible at m=256 (8.3 us vs 4.5 floor) but not expected
- `layer=unpooler load=verify_*` flat: m=256 nvjet already ~700 TFLOP/s; m=128 20.4 vs 14.6 floor may move
- `layer=* load=decode_*` flat: m <= 32 untouched
- `layer=* load=verify_1 m=8` flat: m <= 32 untouched
- `layer=router *` flat: untouched

## Risk

Medium. (a) The installed DeepGEMM may predate the BF16 GEMMs (then the run measures master: read the [deepgemm] lines first). (b) Its JIT needs nvcc and a writable cache dir at prepare time; load_inline builds have run on the judge, so nvcc exists, but print the JIT seconds against the 10-minute budget. (c) DeepGEMM's heuristics favour large M; at m=128/256 with M on the token side it may choose block_m 64/128 and lose like CUTLASS did; per-shape rows decide, gate per layer at harvest. (d) If it allocates or syncs inside the call, graph capture fails in prepare and the layer is disabled. Rules: an in-image library (rules list DeepGEMM explicitly), no timing-based selection.
