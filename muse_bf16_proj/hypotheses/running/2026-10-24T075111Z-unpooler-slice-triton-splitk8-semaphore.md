---
problem: muse_bf16_proj
slug: unpooler-slice-triton-splitk8-semaphore
created: 2026-10-24T075111Z
base: master
area: unpooler GEMM
change: >
  Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the
  sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only, replace
  _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0 us +
  cublasLt::splitKreduce_kernel 1.5 us; 7.0-7.1 us reported vs a 1.7-1.8 us floor, the worst ratio
  to floor of any bf16 shape) with ONE Triton launch shaped like the landed router kernel: grid =
  (1024 // BLOCK_N, SPLIT_K) = (16, 8) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=8 (K-slice
  512, 4 K-steps), BLOCK_M=16 (rows >= m masked to 0), num_warps=4, num_stages=3, so each CTA
  reads 64 rows x 512 K x 2 B = 64 KB, the same bytes per CTA as the router's 32 rows x 512 K x
  (hi + lo); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [8192, 4096]
  weight's first 1024 rows (row stride 4096, no copy of the slice, no repack), fp32 acc =
  tl.dot(x_tile, w_tile, acc). Epilogue with a bf16 output and no memset: store the fp32 partial
  to persistent scratch state['unpool_ws'] = torch.empty((8, 16, 1024), fp32) from prepare();
  tl.debug_barrier(); scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel',
  scope='gpu') on state['unpool_cnt'] = torch.zeros(16, int32) from prepare(); the CTA drawing
  ticket == 7 reloads the 8 partials (cache_modifier='.cg'), sums in fixed order, casts once to
  bf16, masked-stores [m, 64] into out = torch.empty((m, 1024), bf16), and resets the counter with
  tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm the binary in prepare() on synthetic x (m in 1, 5, 8)
  and re-zero unpool_cnt afterwards. Unpooler with n == 8192 (m >= 16) and every other layer keep
  master's path.
expected:
  - shapes: layer=unpooler load=decode_1 *
    direction: faster
    why: "7.1 us row: the router kernel moves the same 8.4 MB with 128 CTAs of 64 KB each in 3.3 us; one node instead of nvjet + splitKreduce; target <= 5 us"
  - shapes: layer=unpooler load=verify_1 *
    direction: faster
    why: m=8, same binary, 7.0 us row
  - shapes: layer=unpooler load=decode_16 *
    direction: flat
    why: n=8192 keeps cuBLAS; control
  - shapes: layer=unpooler load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
  - shapes: "*layer=router*"
    direction: flat
    why: master router untouched; control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T083652Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T092007Z
base_sha: 4463015ebd7f2b7014e5f694aa241cc9df7a78c1
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore
worktree: .worktrees/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore
mode: null
---

# unpooler-slice-triton-splitk8-semaphore

## Change

Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only, replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0 us + cublasLt::splitKreduce_kernel 1.5 us; 7.0-7.1 us reported vs a 1.7-1.8 us floor, the worst ratio to floor of any bf16 shape) with ONE Triton launch shaped like the landed router kernel: grid = (1024 // BLOCK_N, SPLIT_K) = (16, 8) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=8 (K-slice 512, 4 K-steps), BLOCK_M=16 (rows >= m masked to 0), num_warps=4, num_stages=3, so each CTA reads 64 rows x 512 K x 2 B = 64 KB, the same bytes per CTA as the router's 32 rows x 512 K x (hi + lo); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [8192, 4096] weight's first 1024 rows (row stride 4096, no copy of the slice, no repack), fp32 acc = tl.dot(x_tile, w_tile, acc). Epilogue with a bf16 output and no memset: store the fp32 partial to persistent scratch state['unpool_ws'] = torch.empty((8, 16, 1024), fp32) from prepare(); tl.debug_barrier(); scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['unpool_cnt'] = torch.zeros(16, int32) from prepare(); the CTA drawing ticket == 7 reloads the 8 partials (cache_modifier='.cg'), sums in fixed order, casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 1024), bf16), and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm the binary in prepare() on synthetic x (m in 1, 5, 8) and re-zero unpool_cnt afterwards. Unpooler with n == 8192 (m >= 16) and every other layer keep master's path.

## Expected

- `layer=unpooler load=decode_1 *` faster: 7.1 us row: the router kernel moves the same 8.4 MB with 128 CTAs of 64 KB each in 3.3 us; one node instead of nvjet + splitKreduce; target <= 5 us
- `layer=unpooler load=verify_1 *` faster: m=8, same binary, 7.0 us row
- `layer=unpooler load=decode_16 *` flat: n=8192 keeps cuBLAS; control
- `layer=unpooler load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=router*` flat: master router untouched; control

## Risk

Not a retry of unpooler-slice-triton-gemv (0.880x): that one used an FMA broadcast GEMV without tl.dot and ran 70.8 us at m=8. This copies the only Triton design proven fast on an 8.4 MB read here (the landed router: tl.dot, BLOCK_K=128, K-slice 512, 128 CTAs, 3 stages, 3.3 us kernel) and changes only what the bf16 output needs (semaphore reduce instead of zeros + fp32 atomics + a cast kernel, which would be 3 nodes and lose). Per-shape ~30% on 2 shapes, each ~13% of its load: ~+4% on decode_1 and verify_1, ~+1.3% geomean. Risks: semaphore ordering correctness (the m=5 unpooler test and the m=7.. tests exercise it); the serial last-CTA reduce reads 8 x 16 x 64 x 4 B = 32 KB, ~0.3 us; the row stride is 4096 while the weight tensor is [8192, 4096], so the kernel must take stride_wn from the full weight, not assume a contiguous slice. If it lands, the same kernel is the shared base for pooler-triton-splitk2-blockn64-semaphore and qkv-gate-triton-splitk4-blockn64-semaphore.

## Adversarial review

Reviewed 2026-10-24T083652Z by an adversarial subagent (model opus).

**Verdict: proceed.** Well-grounded and attributable: it copies the router kernel's measured shape (same 8.4 MB, 128 CTAs, 64 KB per CTA, 48 KB in flight) onto the worst bf16 ratio-to-floor shape (unpooler n=1024, 7.0-7.1 us vs 1.7-1.8 us floor) and collapses two cuBLAS nodes into one. The other 28 shapes are built-in controls. A realistic outcome is ~5.0-5.5 us per shape, readable per shape but likely under the ~1.5% headline resolution.

### Justification
The mechanism is plausible. In the profile, nvjet splitK is 4.0-4.1 us plus splitKreduce 1.5-1.6 us, so 5.5-5.7 us of kernels against 7.0-7.1 us reported. The landed _router_hilo_kernel reads the same 8.4 MB (256x8192x2 tensors x2 B, the same as 1024x4096x2 B) in 3.3-3.5 us at m=1-16 with 128 CTAs, BLOCK_K=128, K-slice 512 and 3 stages. The proposed tile (64 N-rows x 256 B contiguous per k-step, 16 KB per stage, 48 KB in flight) matches the router's two 32-row tiles byte for byte, so the mainloop should land near 3.3 us. The serial last-CTA reduce is small: only m rows (at most 8) of 8x64 fp32 are needed, a few KB from L2. The expected directions agree with TASK.md. Every shape has weight 62 and each load has 5 shapes. decode_1 is 55.2 us (14.4+6.7+21.2+7.1+5.8) and verify_1 is 54.3 us. Saving about 2 us per shape gives about +3.6% on those two loads and about +1.2% on the geomean, so the hypothesis's +1.3% is right. That sits below the ~1.5% single-run headline resolution in LESSONS, but the per-shape effect (25-35%) is far above the 1.5-2.5% drift. Weakest link: the '<= 5 us target'. The reported-minus-kernel gap is not the ~0.3 us LESSONS quotes. The router's 1-kernel-plus-memset row is 5.8 us reported vs 4.1 us profiled, and the earlier unpooler-slice-triton-gemv single Triton node was 11.53 us reported vs 7.2 us profiled at m=1, a 4.3 us gap. So the result could plausibly land at 5.0-5.5 us, not <= 5, and the acq_rel ticket plus the dependent tail reload adds a latency step the router does not have. The second risk is semaphore correctness in Triton. Release needs every thread's partial stores ordered before the gpu-scope acq_rel atomic (bar.sync via debug_barrier, then the atomic, which is cumulative). Acquire must be followed by a CTA barrier before other threads issue the .cg reloads. Triton's scalar-atomic broadcast through shared memory normally provides that, but the code should not depend on it implicitly.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: no shared launch sequence, memory or numerical path with the unpooler (shapes are captured and timed separately with a cold L2). It is a reference design, not a confounder. Only the prepare() warm-up budget is shared, and adding three unpooler warm-ups is trivial against 300 s.
- sglang-gemm-dispatch-parity: this defines the incumbent (_bf16_gemm_dispatch_impl on W[:1024]) that the change replaces for n=1024 only. It is also SGLang's own path, so the paired ratio is a clean head-to-head. No interaction.
- router-hilo-at-m8: router-only, superseded by the Triton router. No interaction with unpooler shapes.

### Interference and A/B needs
No companion is warranted. A gain is unambiguous: only unpooler n=1024 (decode_1, verify_1) changes path, and decode_16/verify_16/router plus the other 24 shapes are untouched controls. No landed hypothesis touches this path, so ab-revert or ab-tune would only measure noise. A loss is mostly readable from the profile alone, because Triton kernel times reproduce within ~1% and the single kernel's own time can be compared directly with the router's 3.3 us on the same bytes. A kernel well above ~3.8 us points at the mainloop or tail. A kernel near 3.5 us with a large reported gap points at per-node cost. The one ambiguity is mainloop vs semaphore tail. The queued router-splitk-semaphore-no-memset puts the same ticket epilogue on a kernel with a known baseline, and the in-flight pooler/qkv semaphore runs measure the epilogue too, so they serve as the cross-check. A dedicated control here (a separate reduce kernel) would add a node and could not win, so it is not worth a run.

### Open questions
- Does the semaphore pattern provide gpu-scope ordering as written? Specifically: an explicit tl.debug_barrier() after the ticket atomic and before the .cg reloads, so non-leader threads are ordered after the acquire. If the in-flight pooler/qkv semaphore runs fail correctness, fix this before dispatching.
- Should the ticket==7 CTA sum its own register acc plus 7 reloaded partials instead of 8 reloads? That saves one 4 KB reload and keeps the fixed order if it is applied consistently.
- Store only the rows < m of each partial to unpool_ws (masked) and reload only those, which cuts the tail from 32 KB to <= 16 KB at m=8.
- The reported-minus-profiled gap for single Triton nodes varies widely: 4.3 us for the earlier slice GEMV vs ~1.7 us for the router's two nodes. When harvesting, judge on the per-shape paired ratio and the kernel's own time, not on the '<= 5 us' target.
- The prepare() warm-up must use the real unpooler weight slice (or the same strides) so the warmed binary matches the stride_wn=4096 specialization used at timed calls.

