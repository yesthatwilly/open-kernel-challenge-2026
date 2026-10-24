# muse_bf16_proj · 2026-10-24T181732Z · submit · bf16-small-m-cuda-gemv-rewrite

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T165102Z-bf16-small-m-cuda-gemv-rewrite.md` (branch `hyp/muse_bf16_proj/bf16-small-m-cuda-gemv-rewrite`).

- **Job:** `b76aca7ec1a048e7` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `6fa230c`

## Goal
Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; attention's only >Triton bandwidth came from a load_inline CUDA kernel (full-decode-cuda-gather-splitkv, compiled at import in 41 s). Stage 1: one hand-written CUDA kernel, built with torch.utils.cpp_extension.load_inline at import (no network, cached in the run's own build dir; on any compile/load failure print '[gemv] load_inline failed' and keep master's routing), for layer in {qkv_gate, o, pooler} at 1 <= cfg m <= 8 and the unpooler slice (cfg n == 1024, m <= 8, the first 1024 rows of the full weight read in place, row stride 4096). Design: swap-AB mma.sync.aligned.m16n8k16 bf16 -> fp32 with the WEIGHT as the A operand (16 weight rows per MMA) and the <= 8 tokens as B (n8 exactly covers m <= 8; columns >= m zero-filled), so no MMA work is wasted on padding rows. Grid: persistent, 2 CTAs per SM (264), each CTA owns a contiguous range of (64-weight-row x K-chunk) work items; split-K only where N/64 < 264 work items need it (qkv 36 n-tiles x K-split 8; pooler 64 x 4; o 128 x 2; unpooler slice 16 x 16), chosen statically per layer. Weight bytes in flight: a 4-stage cp.async.cg (16-byte, L1-bypass) or cp.async.bulk ring in shared memory of >= 32 KB per stage per CTA (>= 256 KB in flight per SM, double master's Triton 128 KB per CTA at 1 CTA/SM), consumed by 4 warps each owning 16 weight rows, ldmatrix from a swizzled tile to avoid bank conflicts. x (<= 8 x 8192 bf16, 128 KB) is loaded once per CTA's K-range into shared memory. Reduce: fp32 partials into a prepare()-allocated workspace [SPLIT_K, 8, n] with master's ticket semaphore (store, __threadfence, one atomicAdd ticket per n-tile, last CTA sums in fixed order, writes bf16, resets its counter). prepare(): compile check, allocate workspaces/counters (zeroed), warm m in 1..8 per layer, run a self-check against x @ W.T on synthetic x (rel L2 <= 1e-3, else print 'SELF-CHECK FAILED' and disable the path; a constant fixed in prepare, never input-derived), and print per layer the warm-L2 cudaEvent kernel time and effective TB/s for m=1 and m=8. kernel_fn routes only those shapes; m >= 9 and every router shape keep master.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-small-m-cuda-gemv-rewrite`, `muse_bf16_proj/submission.py` only. The diff only adds lines (+403, -0). Every existing path (the Triton m<=16 gates, swap-AB at 17..32, cuBLAS, router) is unchanged.

- **New section (6)**, placed before `prepare()`. It holds `_GEMV_CPP` / `_GEMV_CUDA`, one translation unit with one non-templated kernel `gemv_m8_kernel` and the host wrappers `gemv_m8`, `gemv_occupancy` and `gemv_smem_bytes`. `_build_gemv()` builds it with `load_inline(name="muse_bf16_gemv_m8_v1", -O3 -arch=sm_90a)` **at import**, the same pattern as the attention precedent. If the build fails it prints `[gemv] load_inline failed ...` and sets `_GEMV_EXT = None`, so master's routing stays everywhere.
- **Kernel.** Swap-AB `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32` with the weight as A (16 rows per warp, 4 warps = 64 rows per CTA) and the <=8 tokens as B. Token rows >= m are zero-filled by `cp.async.cg` with src-size 0. Both operands come through `ldmatrix.x4` from XOR-swizzled tiles: 16-B chunk c of row r is stored at c ^ (r & 7), which is conflict-free for the 8-row ldmatrix phases. Two alternating accumulators break the MMA dependency chain. The weight is read in place with `stride_w = w.stride(0)`, so the unpooler slice uses the first 1024 rows of the [8192, 4096] weight with row stride 4096.
- **Shared memory (review fix 1).** BLOCK_K = 128 and a 6-stage ring. Each stage holds a 16 KB weight tile and a 2 KB x tile; x is staged per stage, not per K-range. That is **110592 B (108 KB) dynamic smem per CTA**. At **2 CTAs/SM** (`__launch_bounds__(128, 2)`) it is 216 KB + 2 x 1 KB reserved = 218 KB <= 228 KB. Up to 5 stages (80 KB of weight) are in flight per CTA, 160 KB per SM. `cudaFuncAttributeMaxDynamicSharedMemorySize` and the max-shared carveout are set explicitly. `prepare()` prints the smem size and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
- **Decomposition (review fix 2).** Each CTA gets one item, and all items run in one wave of <= 264 = 2 x 132:

  | layer | items | K per item |
  |---|---|---|
  | pooler | 64 x 4 = 256 | 2048 |
  | o | 128 x 2 = 256 | 512 |
  | unpooler slice | 16 x 16 = 256 | 256 |
  | qkv_gate | **36 x 7 = 252** | uneven chunks of 9 or 10 x 128, chunk s = K-blocks [s*64/7, (s+1)*64/7) |

  The qkv split replaces the 288-item plan.
- **Deviation: the grid is not persistent.** The hypothesis asked for a persistent grid with contiguous item ranges. Since every layer now has <= 264 items, one item per CTA in one resident wave is equivalent and simpler.
- **Reduce (review fix 3).** fp32 partials are stored only for token rows < m, into this path's own `gemv_ws_<layer>` [SPLIT_K, 8, N] and `gemv_cnt_<layer>` buffers, separate from the Triton paths' buffers. Every thread runs `__threadfence`, then `__syncthreads`, then one `atomicAdd` ticket per n-tile. The last CTA fences again, sums the partials with `__ldcg` in fixed order 0..SPLIT_K-1, stores bf16, and resets its counter with `atomicExch(..., 0)`.
- **Routing.** `kernel_fn` checks `_gemv_route_ok` first, ahead of every m<=16 Triton gate. It requires all of:
  - `1 <= cfg m <= 8`
  - the layer's own n and k (the unpooler only at n == 1024)
  - bf16 rows with 16-B alignment
  - `state["gemv_ok"][layer]` set

  `gemv_ok` is a per-layer constant fixed in `prepare()`. m >= 9 and the router never reach this path.
- **Self-check (`_gemv_check_layer`, every routed (layer, m)).**
  - Cases: synthetic x for every m in 1..8, plus an extra m=1 case. For m >= 2, and in the extra m=1 case, the last row is a one-hot 2.0.
  - Each case launches **twice**. The two outputs must be bitwise equal, which catches a counter-reset bug.
  - Outputs must be finite.
  - Rel L2 must be <= 1e-3, and the worst row <= 5e-3 (5x, with near-zero rows measured against 0.1 x the mean row).
  - The one-hot row must be exact.
  - Every ticket counter must be 0 at the end.
  - On any failure or exception it prints `[gemv] SELF-CHECK FAILED layer=...: <why>` and keeps master for that layer only.
- **Deviation: the self-check reference.** The reference is fp64 (`x.double() @ W.double().T`, answering the open question), and the 1e-3 bound is taken against that reference **rounded to bf16**. bf16 rounding of the output alone gives a rel L2 of about 1.5e-3 against the unrounded fp64 value, so the spec's literal "rel L2 <= 1e-3 against x @ W.T" would fail even a perfect kernel.
- **Diagnostics.** For each passing layer, `prepare()` prints the warm-L2 cudaEvent time per launch and the effective TB/s at m=1 and m=8. This is eager and includes launch overhead, so it is an upper bound on kernel time.
- **Not compiled here.** There is no nvcc or GPU on this node, so the CUDA source has never been compiled. The first check run is its first compile. The import-time fallback and the per-layer self-check are the safety net, but a device-side fault (an illegal address) would be a sticky CUDA error that no Python fallback can recover from.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-small-m-cuda-gemv-rewrite diff` (+403/-0, submission.py only):

- [x] CUDA-graph capturable.
  - The timed path (`_gemv_route_ok` -> `_gemv_forward` -> `_GEMV_EXT.gemv_m8`) has no `.item()`, `.cpu()`, synchronize or tensor prints.
  - It branches only on cfg, shapes, strides, data_ptr alignment and the prepare()-fixed `gemv_ok`.
  - The launch goes to `getCurrentCUDAStream()`. `cudaFuncSetAttribute` runs at the first launch, which is in the prepare() self-check, not during capture.
  - The output is a `torch.empty` sized from shapes only.
  - The ticket counters self-reset inside the kernel, so every replay starts from 0.
  - The syncs (`torch.cuda.synchronize`, `e1.synchronize`, `float()`/`int()` reductions) are all in prepare()-only helpers; lint reports them as warnings only.
- [x] Nothing derived from `data` is cached across calls. `ws`/`cnt` are prepare()-allocated scratch that each launch overwrites, and the reducer reads only rows < m written in the same launch. `gemv_ok` depends on the weights and synthetic x only.
- [x] Weights: BF16 stays BF16, unchanged and read in place, with no repack. The router is untouched.
- [x] Activations: x is bf16, accumulation is fp32 (mma), and the output is bf16, as in SGLang's bf16 GEMM.
- [x] n/a: attention-only rule (this problem has no attention or KV cache).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import or network. The only build step is `torch.utils.cpp_extension.load_inline` at import, the same mechanism the attention submission already passed the judge with. No process or network modules are imported in this file.
- [x] All work happens inside the timed call. Every output is computed by the launch, with no output caching or replay. The self-check and timing run on synthetic x in prepare().
- [x] Shape routing is limited to 1 <= cfg m <= 8 on the four bf16 layers. m=9..16, 17..32, >=33 and the router keep master's routing byte-identical, since the diff only adds code.

Lint (`okc_lint.py --problem muse_bf16_proj`), exit 0:

```
warn  host-sync    L968: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1002: e1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1049: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed; the `__pycache__` it left was removed.

## Risks / what could fail
high
High risk: a hand-written kernel, judged on the replaced kernels' times against its milestones, not the headline. Compile time counts toward the 10-minute run (attention's load_inline took 41 s); keep one translation unit, one template instantiated per layer. Overlaps bf16-m16-swapab-wgmma-extend on the m=1/m=8 shapes: build on master as written, and if that Triton swap-AB lands first, measure stage 2 against it. The read rules allow the bf16 weights unchanged (no repack needed); any swizzled repack in prepare() must stay bf16 and lossless. Later stages: extend to m=16 with n16 (two m16n8 B fragments), and try TMA 1-D bulk copies if cp.async stalls below 4 TB/s.

## Result (after the run)
- job `b76aca7ec1a048e7` (submit), status done, passed True, tests 9/9
- speedup 1.1667x, 95% CI [1.1492, 1.1842], score 4140.618113293016 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/bf16-small-m-cuda-gemv-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/regression/bf16-small-m-cuda-gemv-rewrite.patch`; branch `hyp/muse_bf16_proj/bf16-small-m-cuda-gemv-rewrite`
