# muse_prefill_layer · 2026-10-24T181734Z · submit · attn-full-prefill-cuda-fp8-exp-offload

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T172146Z-attn-full-prefill-cuda-fp8-exp-offload.md` (branch `hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload`).

- **Job:** `f5c30b15fd284c15` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `5e83a89`

## Goal
Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill layers with a hand-written sm_90a FP8 flash-attention forward kernel in CUDA (torch.utils.cpp_extension.load_inline at import, as attention's full-decode-cuda-gather-splitkv did in 41 s), designed around the one thing that bounds FA3 here: the softmax pipe. WHY: after the experts, FA3 is the largest kernel on all five full shapes (415.2 us turn_1 full, 1634.0 turn_4 full, 892.2 first_turn ctx0, 4471.1 ctx32768 full, 8006.7 ctx65536 full; 24-37% of the three big full first_turn shapes) and it does not respond to occupancy (LESSONS: ~600 TFLOP/s on every full prefill shape whatever the CTA count; fa3-batched-mode held for that reason). The evidence is that it is bound by exponentials: score elements per shape are T x keys x 16 heads = 0.97G, 3.87G, 2.15G, 10.7G and 19.3G, which at sm_90's 16 ex2 per clock per SM (132 SMs, ~1.83 GHz, 3.87 T/s) take 250, 1000, 555, 2770 and 4990 us; FA3 measures 1.60-1.66x that on every one of the five shapes, while its FP8 MMA work (4 x 64 FLOP per element) would take only ~0.5x of it at FP8 tensor rate. With head_dim 64 there are only 256 MMA FLOP per exponential, so the tensor cores idle on MUFU and the surrounding FP32 softmax work. No amount of rescheduling FA3 changes that ratio; only moving exponentials off MUFU does. KERNEL DESIGN: persistent grid (132 CTAs), static work list from the config only (batch, q_len, heads: shapes, never values), longest-first so the causal tail balances; per-item key counts read on the device from cache_seqlens/cu_seqlens_q (no host sync, graph-safe). Item = 128 query rows of one KV head: 16 consecutive tokens x the 8 q heads that share it (GQA-packed, causal mask per token, so each gathered 64-byte K/V row feeds 8 heads), by default 2 consumer warpgroups x 64 rows (try 3 x 64 if registers allow) and 1 producer warpgroup. Producer: page_table lookup and cp.async 16 B gather of page-size-1 K and V rows (TMA cannot gather rows on sm_90) into a 2-3 stage K/V ring of 128-176 keys, mbarrier-signalled; V transposed in smem to K-major for the FP8 PV MMA (ldmatrix + byte prmt, as FA3's FP8 path does). Consumers: S = Q K^T by wgmma m64nNk32 e4m3 (Q FP8 in smem, queries rounded to E4M3 exactly as FA3 does today and no coarser), online softmax in fp32 registers with the scale folded into one FFMA, P converted to FP8 E4M3 in registers (FA3's own P precision) and permuted from the accumulator to the A-fragment layout, O += P V by wgmma RS, intra-warpgroup overlap of softmax(j) with QK^T(j+1) and inter-warpgroup ping-pong on named barriers. The new idea: a constexpr fraction of each row's exponentials (start at 1 in 3 columns, tune toward the MUFU/FMA balance point, ~0.7 MUFU) is computed on the FMA pipe by Cody-Waite range reduction plus a degree-3 minimax polynomial for 2^frac and an integer add into the exponent (the FlashAttention-4 software-exp trick; optionally as HFMA2 pairs, since P is rounded to E4M3 anyway). Every key is still attended with its exact softmax weight (no skipping, no block sparsity, the whole context on full layers). Epilogue: O / l to bf16 [T, 16, 64], the same tensor FA3 returns, so o_norm_gate and everything after are unchanged. STAGE PLAN: stage 1 (this file) = full layers only (window == 0), page_size == 1 and kv_scale == 1: the five scored full shapes and the batch 1 ctx 3000 full test; everything else (sliding, page_size 64, the scaled-cache test) keeps FA3 through a shape/config gate in kernel_fn. Stage 2 = fuse FusedRMSNormGate (per-head rmsnorm x sigmoid(gate), one 64-wide head row per thread quad) into the epilogue and take sliding layers too (window band: only the first and last tiles of each row's 2048-key band are masked; sliding first_turn FA3 is 304 us at weight 138). Stage 3 = take the query side in (QK-norm + RoPE + E4M3 cast in the prologue, dropping the q.to(FP8) launch), and the kv-scaled and page-64 cases. FALLBACK AND SELF-CHECK: module-level try around load_inline; any exception sets CUDA_FULL_ATTN_OK = False and kernel_fn calls FA3 exactly as master does. In prepare(): (a) a unit test of the software exp against exp2f over [-126, 0] (max rel err <= 2^-11) and (b) the kernel against flash_attn_with_kvcache on synthetic FP8 caches with a randomly permuted page-size-1 page table: batch 1 ctx 3000 L 753; batch 4 with seq_lens {0, 777, 5000, 80000} L 753; batch 1 ctx 0 L 16384; one length not a multiple of 16 or 128. Pass = rel L2 <= 1e-2 and worst row <= 5e-2 vs FA3, no NaN/Inf; else the flag goes off. Log one line "[cuda full attn] ON" or "[cuda full attn] OFF: <reason>", with compile seconds, registers and smem. MILESTONES (profile top_kernels: the new kernel's row replaces FA3's cutlass::device_kernel<flash::...> row on full shapes): M0 log says ON and the 5 tests pass; M1 the kernel beats FA3 on every full shape (turn_4 full under 1634 us, ctx65536 under 8007 us); M2 0.75x FA3 (turn_1 full ~310, turn_4 full ~1225, ctx0 ~670, ctx32768 ~3350, ctx65536 ~6000 us), i.e. ~1.2x the MUFU-only time; stretch 0.65x.

## What changed, and where
Only `muse_prefill_layer/submission.py` changed, in the worktree, uncommitted. The diff is +959/-17 lines.

- **New CUDA section** (before `prepare`). It is one source, `_ATTN_CUDA`, compiled at import by `load_inline` into two separate modules: `USE_WGMMA=1` ("wgmma") and `USE_WGMMA=0` ("mma"). A build error in one module cannot take down the other. Each build logs `[cuda full attn] <backend>: load_inline compiled in Xs`, or `FAILED` with the last 3000 characters of the error.
  - **`attn_fwd_kernel<SW>`**: 256 threads. One CTA per (16-token block x KV head x split), giving 128 GQA-packed rows (8 heads per KV row). Warp w owns tokens 2w and 2w+1, so the causal mask applies per token.
    - Tiles are 64 keys, loaded through a 4-stage cp.async ring of 16-byte page-size-1 gathers. The page-table entry for the next tile is prefetched into a register.
    - K lands in the no-swizzle core-matrix layout (LBO 128, SBO 512).
    - V lands row-major. All threads then transpose it into Vt in the same core-matrix layout, with keys permuted so that the S accumulator fragment of QK^T is the A fragment of PV directly: no P shuffles, only E4M3 packs. Store order is rotated per lane to spread banks.
    - S = QK^T and O += PV use FP8 E4M3 with fp32 accumulation. The wgmma module uses `wgmma.mma_async m64n64k32` RS, with Q and P from registers and B from a smem descriptor, plus `fence.proxy.async` and wgmma fence/commit/wait 0. The mma module uses `mma.sync m16n8k32 e4m3` with the same layouts and the same softmax.
    - Softmax is online in exp2, with FA3's FP8 max offset: P = 2^(x - m + 8), so P sits in [0, 256] in E4M3. O and l carry the same factor, so it cancels. l is summed in fp32.
  - **Software exp `exp2_sw`**: the review's required form, FP32 FFMA only, no HFMA2. Clamp to [-127, 0]. Floor with `__fadd_rd(x, 1.5*2^23 + 8)`, the FADD.RM magic, which also folds in the +8 offset. Then f = x - floor exactly, a degree-3 minimax polynomial (3 FFMA), and an integer add of `t << 23` into the exponent. The exponent field stays in [8, 135], so it never goes denormal or NaN.
    - The coefficients were fitted offline in numpy. A float32 emulation gives max rel err 7.5e-5 on [-126, 0] (limit 2^-11 = 4.9e-4). -inf, -200 and -127 give 2^-119; tiny positive inputs give 255.98.
    - The offload is constexpr `SW` columns of every 8, spread Bresenham-style. `EXP_SW_NB = 2` (25%) matches the reviewer's 25-30% balance estimate. It was not measured.
  - **Split-KV from config**: `_attn_splits` = ceil(8 x SMs / items), capped at 16 and at max_pages/256. That gives turn_1 full 96 items -> 11 splits, turn_4 -> 3, first_turn -> 1, the ctx 3000 test -> 11 and the b3 L200 test -> 14. Each split stores an unnormalized fp32 partial (O, m, l). `attn_combine_kernel` merges them by log-sum-exp into bf16. Every key is still attended; splits only partition the key range. Empty splits store m = -inf and weigh 0.
  - **Grid order**: the grid is ordered last token block first (longest-first), not persistent.
- **`kernel_fn`**: the config-only `_full_attn_gate` takes window 0, page_size 1, kv_scale 1 and tp8 heads, and requires `state["full_attn"]` to be set. The gated path calls `_full_attn` (CUDA kernel, plus combine when split). Everything else calls FA3 exactly as master. The FP8 query is master's `q.view(T,16,64).to(FP8)`.
- **`prepare`**: `_attn_prepare` does the following, then logs `[cuda full attn] ON: PATH CHOSEN: CUDA <backend> ...` or `[cuda full attn] OFF: <reasons>. PATH CHOSEN: FA3 (master)`.
  - Prints regs, local (spill) bytes and CTAs/SM for both SW instances, plus smem.
  - Runs the exp2 unit test over [-126, 0] (2M points) and the special inputs.
  - Runs the self-check against FA3 and an FP64 reference on the FP8 values. The cases are b1 ctx3000 L753 (judge test), b3 ctx5000 L200 with outlier Q/K (judge test), b4 seq {0, 777, 5000, 80000} L753, b1 ctx0 L16384, and b2 {37, 1000} L211. SW = 2 runs on all of them, and SW = 0 also on the first two.
  - Picks the first passing option, in this order: (wgmma, documented descriptor order), (wgmma, LBO/SBO swapped), mma.
  - Times FA3 against the chosen backend at sw = 0 and sw = 2 with CUDA events on synthetic turn_1 and turn_4 full shapes. This is print-only and never routes anything.

**Deviations, and why**
1. **Not warp-specialized or persistent.** There is no producer warpgroup, no mbarriers and no intra-WG overlap. All threads issue cp.async, with `__syncthreads`, and wgmma waits 0 after each GEMM. The grid is non-persistent, with longest-first blockIdx order relying on the hardware block scheduler. This keeps a blind-built stage 1 small enough to be right. Overlap, ping-pong and the persistent schedule are levers for stage 2. As a result M1 (beating FA3) is less likely in this stage than the md's M2 math assumes.
2. **Two backends instead of one.** The mma.sync module is a correctness backstop. It also diagnoses the wgmma pieces: if mma passes and wgmma fails, the fault is in the descriptor or fragment code. It is likely slower than FA3.
3. **The descriptor's LBO/SBO order is a runtime flag** (`desc_swap`). prepare() tries the swapped order only if the documented order (LBO = K-direction 128 B, SBO = N-direction 512 B, as in CUTLASS's INTERLEAVE K-major layout) fails. Both orders stay inside the CTA's smem, so neither can fault.
4. **Self-check pass rule.** A case passes if it is finite, and either (a) within rel L2 1e-2 and worst row 5e-2 of FA3, or (b) no worse than FA3 against the FP64 reference (1.25x + 2e-3, and 1.25x + 1e-2 for the worst row). Rule (b) protects against the case where FA3's own P handling differs, since FA3's max-offset behaviour on this build is unverified.
5. **The md's planned turn_4 timing case** is replaced by synthetic turn_1 and turn_4 full shapes, eager with warm L2, print-only.

**Uncertainties (nothing could be compiled or run here)**
- That mma.sync m16n8k32 .e4m3 compiles and runs on sm_90a (PTX lists it as sm_89+).
- The wgmma register-A fragment layout. It is assumed to equal mma.m16n8k32's for 8-bit types.
- The wgmma RS e4m3 syntax `d, {a}, desc, p, 1, 1`.
- Whether `fence.proxy.async` is sufficient for cp.async-written smem.
- A fault in the wgmma kernel would be a sticky CUDA error and would fail the whole run. The fallback cannot catch that.
- Read the log tail before judging. A `PATH CHOSEN: FA3` or `OFF` line means the run timed master.

### Pre-flight fix (2026-10-24, after an Opus audit)
The audit verified the RS wgmma e4m3 form, the 8-bit A layout, the LBO/SBO order (documented order correct; a swapped descriptor cannot read outside shared memory), the software exp and the split-KV combine. Added an absolute cap (rel L2 <= 0.1, worst row <= 0.5) to the 'no worse than FA3' pass rule, so a broken FA3 run in the self-check cannot let a garbage output through.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload diff`:

- [x] CUDA-graph capturable. The new path has no `.item()`, `.cpu()`, `synchronize()` or tensor prints. All of those are in prepare()-only helpers, and lint confirms none is reachable from kernel_fn. Splits, grid and workspace sizes come from config, `page_table.shape` and the SM count, which prepare() stores in state. Per-item key counts are read on the device. Allocations are shape-sized `torch.empty`.
- [x] Nothing derived from `data` is cached across calls. The state holds only the compiled module, the descriptor flag and the SM count.
- [x] Weights untouched. n/a for attention; no weight is repacked or changed in precision.
- [x] Activations are BF16 where SGLang keeps BF16. The KV cache stays FP8 and is read as stored. Queries are E4M3, exactly master's `q.to(FP8)`. P is E4M3, as FA3 does, with FA3's max offset 8. The exp is fp32 (rel err 7.5e-5 by emulation). The output is bf16 [T, 16, 64], as FA3 returns.
- [x] Attention sees every key: every key in [0, position] on full layers. Split-KV partitions the key range and combines it exactly by log-sum-exp. There is no skipping, top-k or weight-based dropping. Masked keys are only the causal ones beyond the position.
- [x] No timing patches, harness frames or I/O, secrets, module edits, reference import, subprocesses or network. `load_inline` builds the module in-process, the same precedent attention_decode used on the judge. CUDA events are used print-only in prepare().
- [x] All work happens inside the timed call; no output caching or replay. prepare() runs only self-checks and print-only timing on synthetic data.
- [x] First line `# okc: problem=muse_prefill_layer` kept; no `triton.autotune` added.

Lint output (`okc_lint.py ... --problem muse_prefill_layer`):
```
warn  host-sync    L468: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L510: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1350: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1365: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1396: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1431: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1444: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1490: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1496: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1503: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1506: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```
`python3 -I -m py_compile` passed; `__pycache__` was removed from the worktree.

## Risks / what could fail
high
Biggest risk: FA3 is a mature, tuned kernel, and beating it needs FA3-class engineering written blind (FP8 V transpose in smem, accumulator-to-A-fragment byte permutes for P, warp specialization with page-size-1 cp.async gathers). The exp-offload gain can be eaten by FP32 issue-slot pressure: Hopper has no FFMA2, so the polynomial costs ~6-7 issue slots per offloaded element, and the balance point must be tuned. If the kernel lands at FA3 parity, stage 1 still pays as the base for stage 2's norm-gate fusion and stage 3's query fusion. Independent of the expert rewrite: no shared code, so it can run in parallel and be restacked by a 3-way merge.

## Result (after the run)
- job `f5c30b15fd284c15` (submit), status done, passed True, tests 5/5
- speedup 0.8813x, 95% CI [0.8681, 0.8945], score 439568.92565084 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/attn-full-prefill-cuda-fp8-exp-offload.result.json`, `muse_prefill_layer/hypotheses/done/regression/attn-full-prefill-cuda-fp8-exp-offload.patch`; branch `hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload`
