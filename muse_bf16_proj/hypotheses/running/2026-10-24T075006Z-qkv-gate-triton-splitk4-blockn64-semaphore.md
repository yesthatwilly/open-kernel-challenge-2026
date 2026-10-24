---
problem: muse_bf16_proj
slug: qkv-gate-triton-splitk4-blockn64-semaphore
created: 2026-10-24T075006Z
base: master
area: qkv GEMM
change: >
  Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only
  (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl
  (nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT 8.1-10.8 us + cublasLt::splitKreduce_kernel
  1.7-2.0 us; reported 13.8-14.4 us vs a 7.9 us floor, the largest reported-minus-kernel gap of
  any layer) with ONE Triton launch of a split-K kernel with a bf16 semaphore epilogue: grid = (N
  // BLOCK_N, SPLIT_K) = (2304/64, 4) = 144 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=4 (K-slice
  2048, 16 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0),
  num_warps=4, num_stages=4 (24-32 KB smem per stage, so 2 CTAs fit per SM and all 144 CTAs are
  co-resident in one wave); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major
  [N, K] weight (the router's transposed addressing, no repack), fp32 acc via tl.dot(x_tile,
  w_tile, acc). Epilogue: store the fp32 partial to a persistent scratch state['qkv_ws'] =
  torch.empty((4, 32, 2304), fp32) from prepare(); tl.debug_barrier(); one scalar ticket =
  tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['qkv_cnt'] =
  torch.zeros(36, int32) from prepare(); the CTA drawing ticket == SPLIT_K-1 reloads the 4
  partials with cache_modifier='.cg', sums them in fixed order 0..3, casts once to bf16,
  masked-stores [m, 64] into out = torch.empty((m, 2304), bf16) and resets the counter with
  tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm both BLOCK_M binaries in prepare() on synthetic x (m in
  1, 8, 16, 32) then re-zero qkv_cnt. qkv_gate at m >= 128 (and the m=512 outlier test) and all
  other layers keep master's path. This is the router's 3-D split-K design (filled grid, short
  K-slices) applied to qkv_gate, NOT the regressed full-K skinny stream.
expected:
  - shapes: layer=qkv_gate load=decode_1 *
    direction: faster
    why: "14.4 us: 37.7 MB at 3.5-3.7 TB/s is 10.2-10.8 us in one node; the splitKreduce node and the ~3 us reported-minus-kernel gap go; target <= 12 us"
  - shapes: layer=qkv_gate load=verify_1 *
    direction: faster
    why: m=8, same BLOCK_M=16 binary, 14.0 us
  - shapes: layer=qkv_gate load=decode_16 *
    direction: faster
    why: 13.8 us; nvjet main kernel is 8.1 us here (4.66 TB/s) so the margin is the reduce node and gap only; expect a smaller gain
  - shapes: layer=qkv_gate load=decode_32 *
    direction: flat
    why: BLOCK_M=32 Triton binaries carried a ~2.4 us penalty on unpooler; flat-or-slower is the honest call
  - shapes: layer=qkv_gate load=verify_16 *
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
review_at: 2026-10-24T075337Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T075337Z
base_sha: 7bcb8092c1665b25485402c87868a1ebb583ee27
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore
worktree: .worktrees/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore
mode: null
---

# qkv-gate-triton-splitk4-blockn64-semaphore

## Change

Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT 8.1-10.8 us + cublasLt::splitKreduce_kernel 1.7-2.0 us; reported 13.8-14.4 us vs a 7.9 us floor, the largest reported-minus-kernel gap of any layer) with ONE Triton launch of a split-K kernel with a bf16 semaphore epilogue: grid = (N // BLOCK_N, SPLIT_K) = (2304/64, 4) = 144 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=4 (K-slice 2048, 16 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0), num_warps=4, num_stages=4 (24-32 KB smem per stage, so 2 CTAs fit per SM and all 144 CTAs are co-resident in one wave); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [N, K] weight (the router's transposed addressing, no repack), fp32 acc via tl.dot(x_tile, w_tile, acc). Epilogue: store the fp32 partial to a persistent scratch state['qkv_ws'] = torch.empty((4, 32, 2304), fp32) from prepare(); tl.debug_barrier(); one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['qkv_cnt'] = torch.zeros(36, int32) from prepare(); the CTA drawing ticket == SPLIT_K-1 reloads the 4 partials with cache_modifier='.cg', sums them in fixed order 0..3, casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 2304), bf16) and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm both BLOCK_M binaries in prepare() on synthetic x (m in 1, 8, 16, 32) then re-zero qkv_cnt. qkv_gate at m >= 128 (and the m=512 outlier test) and all other layers keep master's path. This is the router's 3-D split-K design (filled grid, short K-slices) applied to qkv_gate, NOT the regressed full-K skinny stream.

## Expected

- `layer=qkv_gate load=decode_1 *` faster: 14.4 us: 37.7 MB at 3.5-3.7 TB/s is 10.2-10.8 us in one node; the splitKreduce node and the ~3 us reported-minus-kernel gap go; target <= 12 us
- `layer=qkv_gate load=verify_1 *` faster: m=8, same BLOCK_M=16 binary, 14.0 us
- `layer=qkv_gate load=decode_16 *` faster: 13.8 us; nvjet main kernel is 8.1 us here (4.66 TB/s) so the margin is the reduce node and gap only; expect a smaller gain
- `layer=qkv_gate load=decode_32 *` flat: BLOCK_M=32 Triton binaries carried a ~2.4 us penalty on unpooler; flat-or-slower is the honest call
- `layer=qkv_gate load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=router*` flat: master router untouched; control

## Risk

Why this is not a retry of qkv-gate-triton-skinny-m32 (0.924x, 22-26 us, ~1.6 TB/s): that kernel was full-K with BLOCK_N=16 (each CTA streamed 16 rows x 8192 K with 32 KB in flight per SM). This one uses the two levers the landed router proved and the unpooler control measured: BLOCK_N=64 tiles (3.73 TB/s on unpooler) and a split-K grid with short slices that keeps 144 CTAs each with 4 x 16 KB weight stages in flight (~9 MB fleet-wide, above Little's law for 4.8 TB/s). Main risk: nvjet is unusually efficient on qkv_gate (8.1 us = 4.66 TB/s at m=16), so the bar is high; the gain must come mostly from removing the reduce node and the gap. 144 CTAs on 132 SMs means 12 SMs host two CTAs; if the profile shows a 2x tail, the one-axis retune is SPLIT_K=8/num_stages=3 (288 CTAs, ~2.2 per SM) or BLOCK_N=128/SPLIT_K=8 (144 CTAs, larger tiles). Semaphore correctness risk as in pooler-triton-splitk2-blockn64-semaphore; a broken ordering fails the m=1 qkv_gate test. If both land they should share one kernel; they are minted independently so each attributes on its own.

## Adversarial review

Reviewed 2026-10-24T075337Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible and not a retry. Done results show single-kernel Triton paths report about what their kernel takes, while the two-node nvjet+splitKreduce qkv_gate path reports 2-4 us more than its kernel sum. A one-node kernel at the measured BLOCK_N=64 Triton bandwidth (3.73 TB/s) lands around 10.5-12 us against today's 13.8-14.4 us. One run is interpretable on its own because only qkv_gate m<=32 changes and the profile names the new kernel's time.

### Justification
Mechanism: qkv_gate m<=32 is pure weight streaming (37.7 MB, floor 7.9 us). Today it is two graph nodes. Reported vs profiled kernel sums: decode_1 14.4 vs 10.8+1.7; decode_16 13.8 vs 8.1+1.7; decode_32 14.3 vs 10.2+1.8; verify_1 14.0 vs 9.2+2.0. Every single-kernel Triton result in done/ has reported close to profiled: unpooler-full-triton-skinny-m32 decode_16 18.0 vs 18.0, qkv-gate-triton-skinny-m32 decode_1 22.6 vs 22.0. So the 'reported-minus-kernel gap' is real and tied to the extra node, and folding the reduce into one launch should remove most of it. The bandwidth premise also has evidence: the BLOCK_N=64 Triton stream measured 3.73 TB/s on unpooler (67 MB in 18.0 us), and 37.7 MB at that rate is 10.1 us. The 144-CTA split-K grid answers the in-flight-bytes failure of the full-K BLOCK_N=16 kernel (0.924x, ~1.6 TB/s). Its ~9 MB in flight is above Little's law for HBM3e. The ticket semaphore does not spin, so it cannot deadlock even if CTAs are not co-resident. It is graph-safe: counters reset in-kernel, workspace and counters are allocated in prepare(), there are no host syncs, and it branches only on m. The fixed 0..3 summation order makes it deterministic. Shape expectations: faster at m=1/8/16 and flat-or-slower at m=32 matches the per-shape evidence. BLOCK_M=32 Triton tiles were slower than nvjet on unpooler decode_32: 19.8 and 21.9 us vs 18.5. m=128/256 untouched is correct. Payoff in score terms: about 2-3 us off a roughly 55 us decode_1 load is about 4-5% on that load, and four of six loads are affected. That is above the about 1.5% CI half-width and well above the qkv_gate decode_1 cell's run-to-run drift (0.981-1.025 across five master-equivalent runs). WEAKEST LINK: carrying the 3.73 TB/s unpooler number over to a kernel half its size with shorter per-CTA streams. Unpooler streamed 67 MB over 18 us. Here each CTA does only 16 K-steps of 256-byte row segments (BLOCK_K=128 vs unpooler's 256), so pipeline ramp, first-byte latency and the semaphore tail (last-arriving CTA per column reloads 4 partials and stores) are a larger share. The 12 SMs hosting two of the 144 CTAs may also set the tail. If effective bandwidth falls to about 3.0 TB/s plus about 1 us of epilogue and ramp, the kernel is about 13.5 us and the result is flat. Second weakest: the 'nvjet is at 4.66 TB/s' figure (8.1 us at m=16) is a profile number that varies 8.0-10.8 us run to run on identical code. Don't read it as the cold-L2 bar. The reported 13.8-14.4 us is the bar.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (965b046): this defines the qkv_gate path being replaced, and the baseline uses the same dispatch, so the measured delta is Triton vs nvjet with no interaction. qkv_gate m>=128 stays on it as a control. Not a confounder.
- router-triton-hilo-fused-control-splitk-all-m (dba4ffd): this is the design template only. The router shapes, kernel and state are untouched, so router cells are a clean control. Its own result logged a 'loss' on qkv_gate decode_1 (0.981) with unchanged qkv code, which shows that cell drifts by about ±2% from noise. Gains or losses on qkv_gate under about 3% should not be attributed. The expected about 15-20% cell effect is far outside that.
- router-hilo-at-m8 (f9dc05d): router-only threshold, superseded by the all-m Triton router. No shared launch sequence, memory or numerics with qkv_gate. Not a confounder.

### Interference and A/B needs
No companion is warranted. Only the qkv_gate m<=32 cells change, and each scored shape is timed as its own graph after an L2 flush, so a landed hypothesis cannot leak into these cells through cache or launch-sequence effects. Reverting or retuning a landed item would not change what this run measures. The run is still interpretable whichever way it goes. A gain is unambiguous, since only one new kernel is on the path. A loss or a split result can be read from the per-shape table plus the profile's single-kernel duration: kernel time well above 10 us means bandwidth or occupancy, and the hypothesis already pre-registers the one-axis retunes (SPLIT_K=8/num_stages=3 or BLOCK_N=128/SPLIT_K=8); a slowdown only at m=32 means the BLOCK_M=32 penalty and the follow-up is restricting to m<=16. The one ambiguity a companion could resolve is epilogue cost vs streaming cost. A fair control for that (atomic_add into a zeroed fp32 buffer) would add a memset and a cast node and reintroduce the very gap under test, so it would not isolate cleanly. It is better run only if the primary comes back flat with a kernel time near 11-12 us.

### Open questions
- Does the installed Triton lower the scalar atomic_add with sem/scope to a single-thread atom.acq_rel.gpu? Dump the PTX in the attempt file.
- Smem per CTA at num_stages=4: weight 16 KB + x 4 KB (BLOCK_M=16) or 8 KB (BLOCK_M=32) per stage is 80-96 KB, so 2 CTAs/SM fit in 228 KB only if Triton does not add epilogue scratch. Confirm the occupancy claim from the compiled kernel's shared-memory size.
- Is BLOCK_K=128 (256-byte row segments from the [N, K] weight) chosen on purpose over unpooler's BLOCK_K=256 (512-byte segments, 8 K-steps per 2048 slice)? If the result is flat, that is a third cheap retune axis.
- If m=32 regresses while m<=16 gains, restrict the route to m<=16 rather than retuning BLOCK_M=32. The per-shape table will show it directly.
- If both this and the pooler semaphore hypothesis gain, merge them into one parametrized kernel with per-layer state keys, as the hypothesis itself suggests.

