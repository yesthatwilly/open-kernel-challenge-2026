---
problem: muse_bf16_proj
slug: bf16-small-m-cuda-gemv-rewrite
created: 2026-10-24T165102Z
base: master
area: bf16 GEMM dispatch (m<=8, hand-written CUDA weight stream)
change: >
  Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the
  incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; attention's only >Triton
  bandwidth came from a load_inline CUDA kernel (full-decode-cuda-gather-splitkv, compiled at
  import in 41 s). Stage 1: one hand-written CUDA kernel, built with
  torch.utils.cpp_extension.load_inline at import (no network, cached in the run's own build dir;
  on any compile/load failure print '[gemv] load_inline failed' and keep master's routing), for
  layer in {qkv_gate, o, pooler} at 1 <= cfg m <= 8 and the unpooler slice (cfg n == 1024, m <= 8,
  the first 1024 rows of the full weight read in place, row stride 4096). Design: swap-AB
  mma.sync.aligned.m16n8k16 bf16 -> fp32 with the WEIGHT as the A operand (16 weight rows per MMA)
  and the <= 8 tokens as B (n8 exactly covers m <= 8; columns >= m zero-filled), so no MMA work is
  wasted on padding rows. Grid: persistent, 2 CTAs per SM (264), each CTA owns a contiguous range
  of (64-weight-row x K-chunk) work items; split-K only where N/64 < 264 work items need it (qkv
  36 n-tiles x K-split 8; pooler 64 x 4; o 128 x 2; unpooler slice 16 x 16), chosen statically per
  layer. Weight bytes in flight: a 4-stage cp.async.cg (16-byte, L1-bypass) or cp.async.bulk ring
  in shared memory of >= 32 KB per stage per CTA (>= 256 KB in flight per SM, double master's
  Triton 128 KB per CTA at 1 CTA/SM), consumed by 4 warps each owning 16 weight rows, ldmatrix
  from a swizzled tile to avoid bank conflicts. x (<= 8 x 8192 bf16, 128 KB) is loaded once per
  CTA's K-range into shared memory. Reduce: fp32 partials into a prepare()-allocated workspace
  [SPLIT_K, 8, n] with master's ticket semaphore (store, __threadfence, one atomicAdd ticket per
  n-tile, last CTA sums in fixed order, writes bf16, resets its counter). prepare(): compile
  check, allocate workspaces/counters (zeroed), warm m in 1..8 per layer, run a self-check against
  x @ W.T on synthetic x (rel L2 <= 1e-3, else print 'SELF-CHECK FAILED' and disable the path; a
  constant fixed in prepare, never input-derived), and print per layer the warm-L2 cudaEvent
  kernel time and effective TB/s for m=1 and m=8. kernel_fn routes only those shapes; m >= 9 and
  every router shape keep master.
expected:
  - shapes: layer=pooler m=1
    direction: faster
    why: "incumbent kernel 19.0 us (67.1 MB, 3.5 TB/s) vs floor 14.0. Milestones: M1 correct and self-check passes; M2 pooler m=1 kernel <= 17.5 us (>= 3.85 TB/s, past the Triton ceiling); M3 <= 16 us (4.2 TB/s)"
  - shapes: layer=pooler m=8
    direction: faster
    why: incumbent 19.3 us; same milestones (m8 MMA is free under the BW bound)
  - shapes: layer=qkv_gate m=1
    direction: faster
    why: incumbent 11.8 us (37.7 MB, 3.2 TB/s) vs floor 7.9; M2 <= 10.5 us
  - shapes: layer=qkv_gate m=8
    direction: faster
    why: incumbent 12.0 us; M2 <= 10.5 us
  - shapes: layer=o m=1
    direction: faster
    why: incumbent 4.6 us (16.8 MB, 3.65 TB/s) vs floor 3.5; M2 <= 4.2 us
  - shapes: layer=o m=8
    direction: faster
    why: incumbent 5.2 us; M2 <= 4.4 us
  - shapes: layer=unpooler m=1
    direction: faster
    why: slice incumbent 4.5 us (8.4 MB, 1.9 TB/s) vs floor 1.7; latency-bound, so 264 CTAs with all bytes issued up front is the lever; M2 <= 3.2 us
  - shapes: layer=unpooler m=8
    direction: faster
    why: slice incumbent 4.6 us; M2 <= 3.3 us
  - shapes: "* m=16"
    direction: flat
    why: master path untouched
  - shapes: "* m=32"
    direction: flat
    why: untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "Triton m<=8 bf16 paths: pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler-slice _bf16_splitk_kernel (decode_1 m=1 and verify_1 m=8)"
review: proceed
status: running
review_at: 2026-10-24T165448Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T165507Z
base_sha: ec7244a1361d3e15e9b974bed90d02b05aa34980
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/bf16-small-m-cuda-gemv-rewrite
worktree: .worktrees/muse_bf16_proj/bf16-small-m-cuda-gemv-rewrite
mode: null
implemented_at: 2026-10-24T170114Z
impl_commit: 6fa230cf03af3a914049ff88e1046984a01fc7df
submission_sha256: 7e4e64746de65a34075781899634e6dd890fd40c083728f34f1c38469168e20d
norm_sha256: b8c6df82a7fb561c0d7070007b266a51c52eac2a433bcff9cc1d0afa38359a43
submission_blob: 49423021596e590a05c4c5f2e83ac3555ce2028d
lint_warnings: 3
---

# bf16-small-m-cuda-gemv-rewrite

## Change

Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; attention's only >Triton bandwidth came from a load_inline CUDA kernel (full-decode-cuda-gather-splitkv, compiled at import in 41 s). Stage 1: one hand-written CUDA kernel, built with torch.utils.cpp_extension.load_inline at import (no network, cached in the run's own build dir; on any compile/load failure print '[gemv] load_inline failed' and keep master's routing), for layer in {qkv_gate, o, pooler} at 1 <= cfg m <= 8 and the unpooler slice (cfg n == 1024, m <= 8, the first 1024 rows of the full weight read in place, row stride 4096). Design: swap-AB mma.sync.aligned.m16n8k16 bf16 -> fp32 with the WEIGHT as the A operand (16 weight rows per MMA) and the <= 8 tokens as B (n8 exactly covers m <= 8; columns >= m zero-filled), so no MMA work is wasted on padding rows. Grid: persistent, 2 CTAs per SM (264), each CTA owns a contiguous range of (64-weight-row x K-chunk) work items; split-K only where N/64 < 264 work items need it (qkv 36 n-tiles x K-split 8; pooler 64 x 4; o 128 x 2; unpooler slice 16 x 16), chosen statically per layer. Weight bytes in flight: a 4-stage cp.async.cg (16-byte, L1-bypass) or cp.async.bulk ring in shared memory of >= 32 KB per stage per CTA (>= 256 KB in flight per SM, double master's Triton 128 KB per CTA at 1 CTA/SM), consumed by 4 warps each owning 16 weight rows, ldmatrix from a swizzled tile to avoid bank conflicts. x (<= 8 x 8192 bf16, 128 KB) is loaded once per CTA's K-range into shared memory. Reduce: fp32 partials into a prepare()-allocated workspace [SPLIT_K, 8, n] with master's ticket semaphore (store, __threadfence, one atomicAdd ticket per n-tile, last CTA sums in fixed order, writes bf16, resets its counter). prepare(): compile check, allocate workspaces/counters (zeroed), warm m in 1..8 per layer, run a self-check against x @ W.T on synthetic x (rel L2 <= 1e-3, else print 'SELF-CHECK FAILED' and disable the path; a constant fixed in prepare, never input-derived), and print per layer the warm-L2 cudaEvent kernel time and effective TB/s for m=1 and m=8. kernel_fn routes only those shapes; m >= 9 and every router shape keep master.

## Expected

- `layer=pooler m=1` faster: incumbent kernel 19.0 us (67.1 MB, 3.5 TB/s) vs floor 14.0. Milestones: M1 correct and self-check passes; M2 pooler m=1 kernel <= 17.5 us (>= 3.85 TB/s, past the Triton ceiling); M3 <= 16 us (4.2 TB/s)
- `layer=pooler m=8` faster: incumbent 19.3 us; same milestones (m8 MMA is free under the BW bound)
- `layer=qkv_gate m=1` faster: incumbent 11.8 us (37.7 MB, 3.2 TB/s) vs floor 7.9; M2 <= 10.5 us
- `layer=qkv_gate m=8` faster: incumbent 12.0 us; M2 <= 10.5 us
- `layer=o m=1` faster: incumbent 4.6 us (16.8 MB, 3.65 TB/s) vs floor 3.5; M2 <= 4.2 us
- `layer=o m=8` faster: incumbent 5.2 us; M2 <= 4.4 us
- `layer=unpooler m=1` faster: slice incumbent 4.5 us (8.4 MB, 1.9 TB/s) vs floor 1.7; latency-bound, so 264 CTAs with all bytes issued up front is the lever; M2 <= 3.2 us
- `layer=unpooler m=8` faster: slice incumbent 4.6 us; M2 <= 3.3 us
- `* m=16` flat: master path untouched
- `* m=32` flat: untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched
- `layer=router *` flat: untouched

## Risk

High risk: a hand-written kernel, judged on the replaced kernels' times against its milestones, not the headline. Compile time counts toward the 10-minute run (attention's load_inline took 41 s); keep one translation unit, one template instantiated per layer. Overlaps bf16-m16-swapab-wgmma-extend on the m=1/m=8 shapes: build on master as written, and if that Triton swap-AB lands first, measure stage 2 against it. The read rules allow the bf16 weights unchanged (no repack needed); any swizzled repack in prepare() must stay bf16 and lossless. Later stages: extend to m=16 with n16 (two m16n8 B fragments), and try TMA 1-D bulk copies if cp.async stalls below 4 TB/s.

## Milestones

Rewrite stage 1 of `Triton m<=8 bf16 paths: pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler-slice _bf16_splitk_kernel (decode_1 m=1 and verify_1 m=8)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T165448Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound rewrite target (8 shapes on 4 of 6 loads, 1.3-2.6x off their floors, and load_inline already has a precedent here), but the stage-1 design as written does not fit H200 shared memory (2 CTAs/SM x a 4 x 32 KB ring plus x staging is 272-320 KB against a 228 KB/SM limit) and leaves qkv_gate with 288 work items on 264 persistent CTAs. Fix both at build time. It is risk high, so it should wait for a check run and not go out as a scored submit.

### Justification
Target and mechanism: FINDINGS shows that every replaced kernel is the whole of its shape's time (100% of top_kernels). The kernels are pooler _bf16_splitk_kernel at 19.0/19.3 us (floor 14.0), qkv _qkv_splitk_sem_kernel at 11.8/12.0 (floor 7.9), o _pooler_skinny_kernel at 4.6/5.2 (floor 3.5) and the unpooler slice at 4.5/4.6 (floor 1.7/1.8), with effective bandwidth of 1.9-3.65 TB/s. These 8 shapes make up 4 of 5 shapes on decode_1 and verify_1, i.e. 2 of the 6 equally weighted loads, so a 10% kernel gain there moves the headline by about 2-3%. The expected directions match TASK.md: m=16/32/128/256 and the router are untouched, so they should be flat. The MMA plan is right for this layout. The weight [N,K] row-major is the A operand of mma.m16n8k16 .row, and x [m,K] row-major is exactly B .col, so ldmatrix works without a transpose. n8 covers m<=8, and the math is about 1k mma per SM for the pooler, which is negligible under the bandwidth bound. Correctness is low-risk numerically: bf16 weights at their own precision, fp32 accumulation, and a fixed-order split-K sum. One-hot power-of-two rows come out exact. WEAKEST LINK: the premise that bytes in flight are what cap Triton at about 3.5-3.7 TB/s. By Little's law, 4.8 TB/s at roughly 1 us of DRAM latency needs only about 36 KB in flight per SM, and master already has about 128 KB per CTA. The gap to the floor is therefore more plausibly fixed per-launch cost: launch and first-byte ramp, wave quantization, and the serialized split-K reduce tail. For example, 19.0 us for 67.1 MB is about 4.1 TB/s steady state plus about 2.5 us of fixed cost. A CUDA kernel can shave that fixed cost (TMA/cp.async.bulk issue, all bytes issued up front, a tighter reduce), but the 'double the bytes in flight' argument should not be relied on. M2 (pooler <=17.5 us, about 3.85 TB/s) is plausible. M3 (4.2 TB/s, within about 88% of the HBM3e peak including ramp) is a stretch. The unpooler slice (8.4 MB) is latency-bound: 3.2 us is reachable only if the SPLIT_K=16 reduce tail is kept small. DESIGN DEFECTS TO FIX IN STAGE 1: (1) Shared memory. H200 allows 228 KB/SM (227 KB per block). The '>= 32 KB/stage x 4 stages per CTA at 2 CTAs/SM (>= 256 KB in flight per SM)' plan does not fit, and staging x for the whole K-range adds 8 to 32 KB per CTA (the pooler K-chunk of 2048 needs 32 KB). As written, the launch fails with too many resources, or the occupancy silently drops to 1 CTA/SM. Use, for example, 2 CTAs x (3 stages x 24-32 KB) with x staged per stage (8 x BLOCK_K bf16, 2-4 KB) instead of per K-range, or 1 CTA/SM with 8 warps and a 6 x 32 KB ring. Set cudaFuncAttributeMaxDynamicSharedMemorySize explicitly. (2) Balance. qkv gives 36 n-tiles x SPLIT_K 8 = 288 items on 264 CTAs, so 24 CTAs run two items and the makespan is about 2x the mean. Pick a decomposition with items <= CTAs (e.g. 36 x 7 with uneven 128-multiple K-chunks, or 36 x 8 = 288 CTAs at 3 per SM if smem allows), or use finer stream-K items (64 rows x 256 K = 1152 units, about 4.4 per CTA). Pooler 64x4, o 128x2 and unpooler 16x16 all give 256 items and are fine. (3) Partials. At m=1, store and reduce only rows < m. Do not move all 8 rows of [SPLIT_K, 8, n] fp32, because at SPLIT_K=16 on the unpooler that is 512 KB of partial traffic against an 8.4 MB weight.

### Confounders on the current kernel
- bf16-shape-gated-triton-winners-combo (landed): this is the incumbent being replaced (pooler/qkv/o at m<=16 and the unpooler slice). The comparison is the intended one, but the m<=8 rows of qkv_gate/pooler/o share prepare()-allocated ws/cnt buffers sized for m<=16. Give the CUDA path its own workspaces and counters so a counter left non-zero by one path cannot corrupt the other, especially in the self-check and warm-up.
- unpooler-slice-triton-splitk8-semaphore (folded into the combo): the slice reads W[:1024] in place with row stride 4096. The CUDA kernel must take stride_wn as an argument and must not assume a contiguous [1024, 4096] tensor; otherwise the m=5 unpooler test fails.
- bf16-m32-swapab-wgmma-triton (landed) and sglang-gemm-dispatch-parity: there is no shared kernel, but kernel_fn's gate order matters. The new route must sit ahead of the existing m<=16 gates and be limited to 1<=cfg m<=8, with m=9..16 and the m=17..32 swap-AB path left byte-identical, so the 'flat' rows are truly untouched.
- router-triton-hilo-fused-control-splitk-all-m and router-hilo-at-m8: no interaction (the router is untouched). The router verify_1/verify_16 rows drifted 0.92-0.94 on byte-identical code in the combo run, so a router row moving here is drift, not this change.

### Interference and A/B needs
No companions. This is kind: rewrite, judged on the replaced kernels' own time per shape (top_kernels) against the incumbent and the floor, and the prepare() log prints per-layer warm-L2 kernel time and TB/s, which already isolates the kernel from the headline. No landed hypothesis shares a launch sequence or numerical path with the new kernel: it replaces the incumbent outright on disjoint shapes. An ab-revert would only re-measure master. The useful follow-up is stage 2, which should respond to whatever the stage-1 profile shows: (a) cp.async vs cp.async.bulk if bandwidth stalls below 4 TB/s, (b) extending to m=16, and (c) re-baselining against bf16-m16-swapab-wgmma-extend if that lands first, since it targets the same m=1/m=8 shapes with Triton wgmma.

### Open questions
- Which smem plan will be built, given that 2 CTAs/SM x 4 x 32 KB plus x does not fit 228 KB/SM? Prefer x staged per stage and 3 stages, or 1 CTA/SM with a deeper ring. Print the achieved occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor) in prepare().
- qkv decomposition: 288 items on 264 CTAs gives a 2x tail. Which split will be used instead?
- Self-check coverage: it should run every m in 1..8 for every layer, call twice in a row (a counter-reset bug shows only on the second launch), include a one-hot power-of-two row (must be exact), and apply a per-row bound (<= 5x the 1e-3 rel-L2 bound) as well as the global rel L2. A wrong token column at one m would slip under a whole-matrix 1e-3.
- Should the self-check reference be fp32 (x.float() @ W.float().T) rather than bf16 cuBLAS, so that the 1e-3 threshold measures the kernel and not cuBLAS's own rounding?
- Scheduling: risk is high and today's bf16 checks are spent. Per the standing policy (DECISIONS 2026-10-24T164610Z), this should wait for the check reset rather than go out as a scored submit.
- If bf16-m16-swapab-wgmma-extend lands first, the m=1/m=8 incumbent changes to Triton wgmma swap-AB. Rebase onto it and compare against its kernel times.


## Implementation notes

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

## Rules check

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
