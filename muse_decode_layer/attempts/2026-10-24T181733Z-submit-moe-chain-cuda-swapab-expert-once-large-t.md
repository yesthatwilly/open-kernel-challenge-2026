# muse_decode_layer · 2026-10-24T181733Z · submit · moe-chain-cuda-swapab-expert-once-large-t

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T172107Z-moe-chain-cuda-swapab-expert-once-large-t.md` (branch `hyp/muse_decode_layer/moe-chain-cuda-swapab-expert-once-large-t`).

- **Job:** `1bba1bab70da4f61` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `e65eda8`

## Goal
WHY: at T=128/256 the two Marlin MoE GEMMs are the single largest block of the whole problem: 254/271 us (sliding/full verify_16) and 443/427 us (verify_32) in a017e10aec544c55, 46-70% of those shapes, at ~1.1-1.2 TB/s, while the MoE-only floor is about 101/109 us (muse_moe_w4a16 SoL at bs=128/256, same weights and routing family). Under EP8 a local expert gets only ~4 (T=128) to ~8 (T=256) routes on average (hot ones 30-60), so the GEMM is a pure NVFP4 weight stream; Marlin re-streams an expert once per 16-row m-block (cap 16) and underfills. Nothing in the decode queue or in moe covers T>=128 with CUDA (moe swapab-nvfp4-gemm-verify-rewrite is Triton and held; moe cuda-nvfp4-swapab-mma-persistent-small-m, b65c9aabd7, is M<=32 only). STAGE PLAN: (1) this file: three new launches replace align + w13 + silu + w2 at tokens > 32; pre-expert F.rms_norm and fused_post_expert_norm_weighted_sum stay. (2) fold the pre-expert RMSNorm into the w13 B-operand load (per-token rsqrt from the route-list kernel) and the post-expert norm sum-of-squares into the w2 epilogue (fp32 atomics into a [T*8] buffer) so the combine becomes one cheap pass; and chain w13 -> w2 per expert with a device-side per-expert completion counter inside one persistent launch (no host sync) to remove the inter-kernel tail. (3) extend the same kernels down to T<=32 (n8 fragments) and retire Marlin, or adopt the moe small-m CUDA kernel if it lands. STAGE-1 SCOPE, gate tokens > 32 from config (covers verify_16/32 and the batch 33 and batch 5 x q_len 8 tests; T<=32 keeps master byte for byte). PREPARE (lossless, NVFP4 stays NVFP4): build a CUDA bank from w13/w13_sf/w13_gs, w2/w2_sf/w2_gs (local experts 0..31). w13 row order permuted so every 16-row MMA tile holds 8 gate rows i..i+7 then the 8 matching up rows I+i..I+i+7 (I=4096): in the m16n8k16 C fragment a lane then holds gate (c0,c1, row lane/4) and up (c2,c3, row lane/4+8) for the same intermediate column, so silu(gate)*up is computed in registers with no smem exchange. Codes and e4m3 group scales (16 k per scale) re-tiled so each lane fetches its A fragment for a (16 rows x 64 k) sub-tile with one 16-byte cp.async (reuse the permutation and dequant helpers from moe b65c9aabd7 _tile_codes/_tile_scales if they compile, else write them from this spec); fold w*_gs into the scale at dequant time (fp32 e4m3*gs -> bf16x2 multiply), never store a dequantized weight. Compile with torch.utils.cpp_extension.load_inline in prepare (one TU, -O3 -arch=sm_90a, print compile seconds; attention compiled a similar TU in 41 s). KERNEL 1 route_list (one CTA, 1024 threads): from local_ids [T,8] (master expert_map path) count routes per local expert in smem, exclusive prefix, write route_idx (t*8+s grouped by expert), expert_off[33], and a compacted work list of active (expert, n_tile, token_chunk) items for w13 and for w2, plus their lengths, and reset both kernels work counters; fixed grid, data-dependent counts stay on device. KERNEL 2 w13_silu (persistent, grid = #SMs, 1 CTA/SM, 8 warps): item = (expert e, 128-row w13 tile j of 64, chunk of <=64 routes of e); weight tile 128 rows x 4096 k streamed exactly once per chunk through a 4-stage cp.async ring of 256-k slices (16 KB codes + 2 KB scales per stage); B operand = the chunk routes rows of the normed expert input gathered by route_idx (bf16, 64 x 256 per stage, ldmatrix.trans from smem; rows past the chunk zero-filled with cp.async src-size 0); swap-AB mma.sync m16n8k16 bf16 (weight dequantized e2m1->bf16 in registers with the Marlin-style lop3/prmt conversion, 8 nibbles per u32) with tokens on N, fp32 acc 16 x 64 per warp; epilogue silu(g)*u in fp32 -> bf16 into act[route, 4096] (route-indexed rows, 64 cols per tile). Items grabbed by atomicAdd on the work counter; hot experts with > 64 routes take extra chunks. KERNEL 3 w2 (same template, not gated): item = (e, 128-row w2 tile of 32, chunk), K = 4096 from act rows, output bf16 y[t*8+s, 4096] written only at local route rows exactly where Marlin wrote them, so fused_post_expert_norm_weighted_sum(y.view(T,8,4096), slot_expert_ids=local_ids) is unchanged. FALLBACK and SELF-CHECK: in prepare, after compile, run synthetic cases at T = 40, 128, 256 with skewed routing (one expert with 96 routes to exercise multi-chunk, some experts empty, remote slots present) against the master Marlin path built from the same bank: per-row rel L2 of y local rows <= 1e-2, global <= 5e-3, and with y pre-filled with NaN the combine output must stay finite (no remote row read). Any compile or check failure sets state[moe_cuda_large_t]=False and keeps Marlin; print one line [moe-cuda-large-t] ON/OFF with compile s, max row rel, and reason. MILESTONES (profile top_kernels at verify_16/32): kernel names moe_w13_silu_swapab, moe_w2_swapab, moe_route_list replace both Marlin rows and act_and_mul; M1 correct and replaced-kernel sum <= Marlin+act_and_mul (254/443 us sliding); M2 <= 200 us at T=256 and <= 180 at T=128 (>=2.6 TB/s on touched experts); M3 <= 160 / 140 us (>=3.3 TB/s). If M1 fails with the stream below 2 TB/s, the stage-2 fix is the dequant path (ALU-bound like moe triton-w13-grouped-gemv-small-m), not more CTAs.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-large-t`, file `muse_decode_layer/submission.py` only. Uncommitted. Diff stat: 1 file, +787 / -3. It was built blind: nvcc is not available here. The CUDA source got a stubbed `g++ -fsyntax-only` pass, with CUDA builtins, torch and `<<<>>>` stubbed. That pass was clean. The PTX was not assembled.

- **Gate.** `_experts` now dispatches. With `x.shape[0] > MOE_CUDA_LARGE_T_MIN` (32), a shape-only branch, and `state["moe_cuda_large_t"]` set, it calls `_lt_experts`. Otherwise it calls `_marlin_experts`, which is master's old `_experts` body unchanged except that it returns `(out, y, cache2, local_ids)` so the self-check can use it. T <= 32 runs master's launches byte for byte.
- **The CUDA module** (`_LT_SRC`, `load_inline` name `muse_decode_moe_large_t_v1`, `-O3 -arch=sm_90a`, one TU) has three kernels. The names match the milestones:
  - `moe_route_list`: one CTA, 1024 threads. It reads the local-start-shifted ids (0..255). Its first pass writes `local_ids` (-1 for ids >= 32), so it folds in master's `expert_map[topk_ids]` gather, and counts routes per expert in smem. Thread 0 then builds the exclusive prefix and a chunk list `(expert, start, n <= 64)`, with `n_chunks` on device. The second pass scatters `route_idx` (route = t*8+s), grouped by expert, through smem atomic cursors.
  - `moe_w13_silu_swapab` / `moe_w2_swapab`: two `__global__` wrappers over one `swapab_body<GATED>`. The mainloop is ported from moe `cuda-nvfp4-swapab-mma-persistent-small-m` (f8b3f72) verbatim: the cp.async helpers, `ldmatrix` x4 with the c ^ (r & 7) swizzle, the e4m3x2 -> f16x2 scale conversion, the exact `dequant8` lop/fma.bf16x2 path, the mma.sync m16n8k16 bf16 form, and the 4-stage 256-k ring of 16 KB codes, 2 KB scales and a 64-row x 512 B B slab. `_tile_codes`, `_tile_scales` and `_code_dims`/`_scale_dims` are also copied verbatim, with lossless round-trip asserts. The global scale x 2^7 is applied in fp32 at the epilogue. Grid = SM count, 256 threads, 1 CTA/SM, dynamic smem 4 x 51200 + 16384 = 221184 B (<= 227 KB).
- **REQUIRED review fix (n8-fragment bound).** Every item reads its chunk's route count `n` and sets `nf = ceil(n/8)`, which is uniform over the CTA. Three things are bounded by it:
  - the cp.async B staging, `i < nf`. Rows n .. 8*nf-1 are zero-filled with src-size 0, and nothing past 8*nf is staged;
  - the `ldmatrix` loop, `f < me_nf`;
  - the `mma.sync` loop, `f < me_nf`.

  To keep registers low, the A fragments for the four k16 steps of a 64-k sub-tile are dequantized once (16 regs). The loop then goes over fragments, and each fragment loads its own 8 B regs.
- **act indexing (open question).** I chose t*8+s, Marlin's cache2 layout. w13 reads B rows `x[route >> 3]` and writes `act[route]`. w2 reads `act[route]` and writes `y[route]`. Both get the route from `route_idx[start + p]`, so they cannot disagree. Remote rows of act and y are never written or read, and `fused_post_expert_norm_weighted_sum(y.view(T,8,4096), slot_expert_ids=local_ids)` is unchanged.
- **Epilogue staged through smem (open question, done).** The bf16 tile (64 routes x 64 cols for w13, x 128 for w2) goes into a dedicated 16 KB smem region. After one `__syncthreads`, it is stored as 16-byte row pieces to `out[route, j*TC + ...]`. The region is next reused at least KS = 16 loop-top barriers later. For w13, gate and up are rounded to bf16 before silu, to match SGLang's cache1 -> silu_and_mul.
- **Self-check** (`_lt_self_check`, prepare only, fixed seed). It runs at T = 40, 128 and 256. Expert 5 takes 96 routes, so it spans 2 chunks. At T=256, expert 9 takes 150 routes (3 chunks, the last one partial). Experts 24..31 are empty, and remote slots are present. Act and y are prefilled with NaN. It checks only what the path consumes or produces:
  - local_ids equal to master's;
  - act and y per-row rel at local routes <= 1e-2;
  - the combine output's global rel <= 5e-3, and finite;
  - a second run identical at local rows;
  - at T=128, hot expert 5 against a local fp32 dequant <= 1e-2.

  There is no align or capacity comparison of any kind, following the moe sibling trap. Any failure or exception leaves `state["moe_cuda_large_t"] = False`, and Marlin is kept. The log lines are `[moe-cuda-large-t] load_inline compiled in Xs` (or FAILED), per-kernel numRegs/localSizeBytes/smem, retile time, per-T self-check lines, a print-only eager timing of CUDA vs Marlin at T=128/256, `[moe-cuda-large-t] ON/OFF: compile s, max row rel, reason`, and `PATH CHOSEN = CUDA|MARLIN`.
- **Deviations from the spec:**
  1. Items are walked statically (item bx + k*grid over n_chunks*n_tiles, read on device) instead of grabbed with an atomicAdd work counter. This is the reviewed sibling's flat-ring scheme: the producer has to know the next item 3 stages ahead. It also removes the counter-reset hazard. Kernel 1 therefore resets no counters.
  2. There is one chunk list, shared by w13 (64 tiles) and w2 (32 tiles), instead of two work lists. `expert_off[33]` is not materialized because nothing consumes it.
  3. Each scale is converted exactly (e4m3 -> f16 -> fp32 x 2^119 -> bf16), and gs is applied in fp32 at the epilogue, not folded into a bf16 scale. This is the sibling's exact scheme, and it removes the review's 2^-9 rounding concern.
  4. B uses non-trans `ldmatrix`. Activations are stored [token][k], which already gives the col-major B fragment, as in the sibling.
  5. The B stage is 64 x 256 k with 4 stages, not the optional 5-stage variant.
- **Uncertainties:**
  - The PTX was never assembled here. The forms are the sibling's (its forms are also untested on the judge).
  - The 2-byte smem epilogue writes have 4-way bank conflicts (minor).
  - The repeat-identical check assumes that tensor-core column results do not depend on column position. Route order within an expert is atomic-nondeterministic, and if that assumption were wrong the check would fail closed, to Marlin.
  - Compile time of the TU is unknown (attention's took 41 s).
  - A sticky CUDA fault inside the self-check would fail the whole run rather than fall back.

## Rules check (each must be true before the run)
Every box was ticked after reading `git -C .worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-large-t diff`.

- [x] CUDA-graph capturable. `_lt_experts` (reachable from kernel_fn) has no `.item()`, `.cpu()`, `synchronize()` or tensor prints. Every allocation is sized from T (shape) only. The CUDA branch is on `x.shape[0]` and a prepare-time flag. Data-dependent counts (`n_chunks`, chunk sizes, nf) stay on device, and the grid is fixed. All syncs and float()/bool() reads are in `_lt_self_check`, `_lt_timing` and `_lt_setup`, which only prepare() calls.
- [x] Nothing derived from `data` is cached across calls. route_idx, chunks, n_chunks, local_ids, act and y are allocated fresh every call. The state holds only the weight retiling, the compiled module, the grid and the flag.
- [x] Weights are repacked losslessly in prepare() only. `_lt_bank` permutes the same E2M1 nibbles and E4M3 scale bytes, asserting exact round trips. gs stays fp32 (x 128, exact). No dequantized weight is stored, and dequant is in registers.
- [x] Activations stay BF16 where SGLang keeps BF16: x, act (cache2) and y are bf16. fp32 accumulation and silu are in fp32 over bf16-rounded gate/up, as SGLang does.
- [x] n/a for attention: it is not touched by this diff. Every key is still attended (attention code unchanged).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. `load_inline` compiles in-process, which matches the precedent already promoted (attention).
- [x] All work happens inside the timed call, with no output caching or replay. The three launches run per call. The self-check and timing are synthetic and prepare-only.
- [x] First-line marker `# okc: problem=muse_decode_layer` is kept. There is no triton.autotune, and T <= 32 keeps master's launches byte for byte.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  dtype        L798: q13 = torch.empty((count, w13[0].numel()), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L799: s13 = torch.empty((count, weights["w13_sf"][0].numel()), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L800: q2 = torch.empty((count, w2[0].numel()), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  dtype        L801: s2 = torch.empty((count, weights["w2_sf"][0].numel()), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L899: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L956: t1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L985: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L992: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1046: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1050: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```

The uint8 warnings are the packed NVFP4 code and scale byte buffers (weights, not activations). The host-sync warnings are all in prepare-only code. `python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Risks / what could fail
high
Biggest risk: the in-register NVFP4 dequant makes the stream ALU-bound below Marlin (Triton versions were 2-2.6x slower at small M), or the mma fragment / scale permutation is wrong; the prepare self-check catches the latter and keeps Marlin. Compile time ~40-90 s fits the 15 min run. Confounding: none of the running decode runs touches T>32 MoE; marlin-block-m-cap-16 (landed) is the incumbent being replaced. Judge on the replaced kernels rows, not the headline.

## Result (after the run)
- job `1bba1bab70da4f61` (submit), status done, passed False, tests 0/0
- best at harvest: {"speedup": 1.1664628806987323, "ci": [1.1489659374882513, 1.183959823909213], "job": "a017e10aec544c55"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: failed** (test:failed)
- files: `muse_decode_layer/hypotheses/done/failed/moe-chain-cuda-swapab-expert-once-large-t.result.json`, `muse_decode_layer/hypotheses/done/failed/moe-chain-cuda-swapab-expert-once-large-t.patch`; branch `hyp/muse_decode_layer/moe-chain-cuda-swapab-expert-once-large-t`
