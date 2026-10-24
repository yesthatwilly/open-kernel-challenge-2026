---
problem: muse_bf16_proj
slug: pooler-triton-splitk2-blockn64-semaphore
created: 2026-10-24T074946Z
base: master
area: pooler GEMM
change: >
  Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For
  layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf16), replace
  _bf16_gemm_dispatch_impl (nvjet_sm90_tst_128x8/16/32_..._splitK_TNT 18.3-20.2 us +
  cublasLt::splitKreduce_kernel 1.6-1.9 us, two graph nodes) with ONE Triton launch of a new
  _bf16_splitk_kernel that applies the router's 2-D tile x K-slice grid to a bf16 output: grid =
  (N // BLOCK_N, SPLIT_K) = (4096/64, 2) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=2 (K-slice
  4096, 32 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0),
  num_warps=4, num_stages=4; weight tiles read as [BLOCK_K, BLOCK_N] straight from the row-major
  [N, K] weight with the router's direct transposed addressing (no repack), acc = tl.dot(x_tile,
  w_tile, acc) in fp32. Per-CTA work is therefore exactly the unpooler-full-triton-skinny-m32
  program (64 weight rows x 4096 K), the one Triton config measured at 3.73 TB/s at m=16. Epilogue
  (no memset, no second kernel): each CTA stores its fp32 partial to a persistent workspace
  state['pooler_ws'] = torch.empty((SPLIT_K, 32, 4096), fp32) allocated in prepare() (scratch
  fully overwritten each call, nothing derived from inputs survives), then tl.debug_barrier(),
  then one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on
  state['pooler_cnt'] = torch.zeros(64, int32) made in prepare(); the CTA that draws ticket ==
  SPLIT_K-1 reloads both partials (cache_modifier='.cg'), sums them in fixed order 0..SPLIT_K-1
  (deterministic), casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 4096),
  bf16), and resets its counter with tl.atomic_xchg(cnt_ptr + pid_n, 0) so the next graph replay
  starts at 0. Pre-compile both BLOCK_M specializations in prepare() on synthetic x (m in 1, 8,
  16, 32) and re-zero pooler_cnt after the warm-up. Pooler at m >= 128 and every other layer keep
  their master path.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: "21.2 us: 67.1 MB at the measured 3.73 TB/s is 18.0 us plus a 16 KB last-CTA reduce tail, and the splitKreduce node and its gap disappear; target <= 19 us"
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: m=8 on the BLOCK_M=16 binary, same mechanism, 21.3 us
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: 21.3 us; BLOCK_M=16 is the exact m=16 config that hit 3.73 TB/s on unpooler
  - shapes: layer=pooler load=decode_32 *
    direction: flat
    why: 22.0 us; the BLOCK_M=32 Triton path paid an unexplained ~2.4 us penalty on unpooler (20.8 vs 18.4 us), so flat is the honest call
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 stays on cuBLAS; control
  - shapes: layer=pooler load=verify_32 *
    direction: flat
    why: m=256 stays on cuBLAS; control
  - shapes: "*layer=router*"
    direction: flat
    why: master router untouched; reproducibility control for the 1.151x base
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T075347Z
review_companions: [pooler-triton-splitk2-blockn64-semaphore-control-blockk256]
stage: implemented
claimed_at: 2026-10-24T075347Z
base_sha: 3fe0dcca6e3cc2a8200f9881dba77770f6234d3e
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore
worktree: .worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore
mode: null
implemented_at: 2026-10-24T075600Z
impl_commit: cee590d58c6c34ded84fdb734c38ca129e327fee
submission_sha256: 22ee139ee4a6828742e6007d116c9ee05b3f2d0b5d635ecd28f519edc860d5e4
norm_sha256: 7a700bcc3c8849e9837636c4313ad6327417a3399a716308dfeedca8f2803ddc
submission_blob: 66ba1a39e2aaa5cedb2ce3ddfb809f67f86b37f9
lint_warnings: 0
---

# pooler-triton-splitk2-blockn64-semaphore

## Change

Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf16), replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_128x8/16/32_..._splitK_TNT 18.3-20.2 us + cublasLt::splitKreduce_kernel 1.6-1.9 us, two graph nodes) with ONE Triton launch of a new _bf16_splitk_kernel that applies the router's 2-D tile x K-slice grid to a bf16 output: grid = (N // BLOCK_N, SPLIT_K) = (4096/64, 2) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=2 (K-slice 4096, 32 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0), num_warps=4, num_stages=4; weight tiles read as [BLOCK_K, BLOCK_N] straight from the row-major [N, K] weight with the router's direct transposed addressing (no repack), acc = tl.dot(x_tile, w_tile, acc) in fp32. Per-CTA work is therefore exactly the unpooler-full-triton-skinny-m32 program (64 weight rows x 4096 K), the one Triton config measured at 3.73 TB/s at m=16. Epilogue (no memset, no second kernel): each CTA stores its fp32 partial to a persistent workspace state['pooler_ws'] = torch.empty((SPLIT_K, 32, 4096), fp32) allocated in prepare() (scratch fully overwritten each call, nothing derived from inputs survives), then tl.debug_barrier(), then one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['pooler_cnt'] = torch.zeros(64, int32) made in prepare(); the CTA that draws ticket == SPLIT_K-1 reloads both partials (cache_modifier='.cg'), sums them in fixed order 0..SPLIT_K-1 (deterministic), casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 4096), bf16), and resets its counter with tl.atomic_xchg(cnt_ptr + pid_n, 0) so the next graph replay starts at 0. Pre-compile both BLOCK_M specializations in prepare() on synthetic x (m in 1, 8, 16, 32) and re-zero pooler_cnt after the warm-up. Pooler at m >= 128 and every other layer keep their master path.

## Expected

- `layer=pooler load=decode_1 *` faster: 21.2 us: 67.1 MB at the measured 3.73 TB/s is 18.0 us plus a 16 KB last-CTA reduce tail, and the splitKreduce node and its gap disappear; target <= 19 us
- `layer=pooler load=verify_1 *` faster: m=8 on the BLOCK_M=16 binary, same mechanism, 21.3 us
- `layer=pooler load=decode_16 *` faster: 21.3 us; BLOCK_M=16 is the exact m=16 config that hit 3.73 TB/s on unpooler
- `layer=pooler load=decode_32 *` flat: 22.0 us; the BLOCK_M=32 Triton path paid an unexplained ~2.4 us penalty on unpooler (20.8 vs 18.4 us), so flat is the honest call
- `layer=pooler load=verify_16 *` flat: m=128 stays on cuBLAS; control
- `layer=pooler load=verify_32 *` flat: m=256 stays on cuBLAS; control
- `*layer=router*` flat: master router untouched; reproducibility control for the 1.151x base

## Risk

Differs from the failed/neutral skinny family (pooler-triton-skinny-m32: full-K, BLOCK_N=32, 128 programs, 3.3 TB/s, neutral 0.998x) in exactly the two things the evidence says matter: BLOCK_N=64 tiles (the only Triton config that beat nvjet's stream) and a split-K grid so the 64-row tile still fills 128 SMs; and unlike the router it emits bf16 without a memset or cast kernel, via a self-resetting per-tile semaphore. Risks: (1) the semaphore pattern (barrier + scalar acq_rel atomic, cumulativity of the release over the CTA's other threads' stores) must be correct under Triton; a wrong ordering shows up as a numeric failure on the m=7 pooler test or a timed-call check, not silently. (2) If the counter is ever left nonzero (e.g. the warm-up), every later call is wrong: re-zero after warm-up. (3) decode_32 may carry the BLOCK_M=32 penalty seen on unpooler; if decode_1/16/verify_1 gain and decode_32 loses, the follow-up is BLOCK_M=16 with two accumulators at m=32, not dropping the design. Readout: the kernel's own us per row vs nvjet 18.3/18.9/19.3/20.2 us; gain on 4 shapes that are 28-36% of four loads, ~2-2.5 us each, ~+3% geomean. Tests touched: m=7 pooler (masked rows, BLOCK_M=16). Correctness: fp32 accumulate, one bf16 rounding, same as cuBLAS splitK.

## Adversarial review

Reviewed 2026-10-24T075347Z by an adversarial subagent (model opus).

**Verdict: proceed-with-companions.** The idea is sound: get rid of cuBLAS's separate splitKreduce step and stream the weight with 64-wide tiles over a full 128-CTA grid. But the spec's BLOCK_K=128 does not match the unpooler kernel it claims to copy (that one used BLOCK_K=256). As written, each CTA keeps the same 64 KB of weight in flight as the 3.3 TB/s pooler kernel, not the 128 KB of the 3.73 TB/s unpooler kernel. Run it with one BLOCK_K=256 control so that a flat result can be read.

### Justification
Mechanism. Pooler at m<=32 is 67.1 MB of bf16 weight. Today it is nvjet splitK at 18.3-20.2 us plus cublasLt splitKreduce at 1.6-1.9 us, and the shape takes 21.2-22.0 us against a 14.0-14.1 us floor. Removing the second graph node and streaming nearer 3.7 TB/s is plausible on H200. There is direct prior evidence. pooler-triton-skinny-m32 (full-K, BLOCK_N=32, BLOCK_K=256, 4 stages, 128 programs) already measured 20.27-20.51 us at m=1/8/16, paired 1.057-1.063 vs SGLang. So a Triton pooler beats cuBLAS by about 1 us at m<=16 even at 3.3 TB/s. That run came out neutral overall (0.998x) only because decode_32 lost 11% (24.5 us vs 21.9). unpooler-full-triton-skinny-m32 moved the same 67.1 MB in 17.97 us at m=16 (3.73 TB/s, paired 1.033). The direction 'faster at decode_1/verify_1/decode_16' is well supported. The 'target <= 19 us' rests on reproducing the unpooler program. WEAKEST LINK: the claim that 'per-CTA work is therefore exactly the unpooler-full-triton-skinny-m32 program' is false as specified. That kernel used BLOCK_N=64, BLOCK_K=256, num_stages=4: 32 KB of weight per stage, 128 KB of weight in flight, 160 KB of smem at BLOCK_M=16 (UNPOOLER_NUM_STAGES comment in its patch). This hypothesis uses BLOCK_K=128 with 4 stages: 64x128x2 = 16 KB of weight per stage and 64 KB in flight. That is byte-for-byte the in-flight depth of the 3.3 TB/s pooler kernel (32x256x2 = 16 KB/stage, 64 KB in flight, per POOLER_NUM_STAGES). The only measured difference between the 3.3 and 3.73 TB/s configs that the family has is the tile and stage bytes, not split-K. So this spec may land at about 20.3 us, the old pooler's rate, and fail to tell 'split-K + BLOCK_N=64 does not help' apart from 'stage depth was the limit'. Secondary: decode_32 marked flat is optimistic. Both earlier Triton m=32 pooler rows lost 11% (24.5-24.6 us). The unpooler m=32 rows lost 3-9% (19.27 us with BLOCK_M=32, 20.8 us with the BLOCK_M=16 control). Split-K 2 halves per-CTA K, so per-CTA x re-read drops from the old pooler's 512 KB to 256 KB (32 MB from L2 in total, the same as the unpooler m=32 case). That makes the unpooler's 19.3 us, about a 3% loss, the right prior, not flat. A loss there is about 30% of the decode_32 load and could erase one shape's gain in the geomean. Expected magnitudes are consistent with TASK.md: equal weights of 62, pooler is 31-32% of each of the four loads, and about 2 us per shape gives about +3% on each of three loads. Verify_16/32 (m=128/256) and router are proper in-run controls. Semaphore design: deterministic, because the fixed 0..SPLIT_K-1 order is applied by whichever CTA arrives last. It is capture-safe if the counter is zero at capture and every replay completes, which the self-reset by atomic_xchg and the re-zero after warm-up provide. Ordering: store partials, bar.sync (tl.debug_barrier), then a gpu-scope acq_rel scalar atomic. This relies on the barrier making the other threads' stores cumulative with thread 0's release. That is accepted practice, but a fence on the release side (CUTLASS Semaphore uses __threadfence before the atomic) is cheap insurance. On the acquire side, the other threads in the last CTA must be ordered after thread 0's acquire. Triton's broadcast of the scalar result through smem normally inserts a barrier. The implementer should confirm this in the TTGIR/PTX or add tl.debug_barrier() after the ticket. The .cg reloads of the partials are correct, since L1 is not coherent.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m (master's 1.151x): different layer, its own CUDA graph per shape, cold L2 per replay, no shared workspace or counters. It cannot move the pooler rows. The router rows are a reproducibility control only. The headline will be judged against 1.151x, so read the gain from the per-shape pooler rows and paired ratios, not from the headline delta.
- sglang-gemm-dispatch-parity: defines the cuBLAS path being replaced, and keeps pooler m>=128 and all other layers unchanged. This is not an interaction: verify_16/verify_32 pooler rows stay on it and serve as in-run flat controls.
- router-hilo-at-m8: router-only threshold, now subsumed by the Triton router. No shared path with the pooler. Not a confounder.

### Interference and A/B needs
No landed hypothesis touches the pooler path, the pooler's memory traffic or its numerics, so ab-revert and ab-tune are not warranted. A gain at m<=16 would be attributable to the change on its own. The ambiguity is internal. The hypothesis says it reproduces the 3.73 TB/s unpooler program, but its BLOCK_K=128 / 4-stage pipeline holds the 64 KB in-flight depth of the 3.3 TB/s pooler program. Suppose the parent comes back about 20.3 us at m=1/8/16 (no better than pooler-triton-skinny-m32) or loses at decode_32. Then you cannot tell whether split-K + BLOCK_N=64 + in-kernel reduce fails, or whether only the stage bytes held it back. A modest gain (about 20 us) would also be indistinguishable from the old full-K kernel's gain. One control fixes this: set BLOCK_K=256 (16 K-steps per 4096 slice) with num_stages=4, which is the unpooler's per-CTA tile, stage depth and step count exactly, with everything else identical. Compare the two runs per row (kernel us at m=1/8/16 and at m=32). That separates the effect of stage depth from that of the split-K/semaphore design, and tells you which variant to promote. A BLOCK_M=16-two-accumulator variant at m=32 is not worth a parallel run: the unpooler control already showed it does not fix m=32 (20.8 us vs 19.3). If decode_32 regresses, the next follow-up is to route only m<=16, not to run another m=32 tile experiment.

### Companions queued
- `pooler-triton-splitk2-blockn64-semaphore-control-blockk256`: Separates stage depth (64 KB vs 128 KB of weight in flight per CTA, the only measured difference between the 3.3 and 3.73 TB/s Triton configs) from the parent's own design (split-K 2 + BLOCK_N=64 + semaphore reduce). Parent about 20.3 us and control about 18.5 us: the parent's spec, not its design, was at fault, and the control is the one to promote. Both about 20.3 us: split-K/semaphore costs as much as it saves. Both about 18.5 us: BLOCK_N=64 + split-K is what matters.

### Open questions
- Does Triton emit a barrier between the scalar atomic's result broadcast and the other threads' .cg reloads in the last CTA? If not, add tl.debug_barrier() after the ticket. Also consider a release-side fence before the atomic, as CUTLASS Semaphore does.
- Is the ws store masked to rows < m? At m=1 an unmasked BLOCK_M=16 store writes 16x more partial bytes than needed. This is harmless (about 128 KB total) but adds to the reduce tail.
- decode_32: given both earlier Triton m=32 pooler rows lost 11% and the unpooler m=32 rows lost 3-9%, should the promotable variant route only m<=16 (leaving decode_32 on cuBLAS) if decode_32 regresses? Per-shape rows from this run answer that without another companion.
- Readout should use the Triton kernel's own us per row (top_kernels) and the paired ratios, calibrated against pooler-triton-skinny-m32's 20.27-20.51 us rows at m=1/8/16. Headline deltas vs 1.151x will be inside the CI (expected about +2-3% geomean).


## Implementation notes

All changes are in the worktree's `muse_bf16_proj/submission.py`. The diff adds 113 lines and removes none. The router kernel, `_router_forward` and the router warm-up are untouched.

- **New `_bf16_splitk_kernel`** (`@triton.jit(do_not_specialize=["M"])`).
  - Grid is `(4096 // 64, 2)`, which is 128 CTAs. BLOCK_N=64, BLOCK_K=128 (as specified; the BLOCK_K=256 control is a separate companion), SPLIT_K=2, K_SLICE=4096 (32 K-steps), num_warps=4, num_stages=4.
  - BLOCK_M is 16 for m<=16 and 32 for 17..32. There is one M-block, and rows >= m are masked.
  - The weight is read as `[BLOCK_K, BLOCK_N]` directly from the row-major `[N, K]` weight, using the router's addressing. The product is `acc = tl.dot(x_tile, w_tile, acc)` in fp32.
- **Epilogue, in this order:**
  1. Masked (`offs_m < M`) fp32 partial store to `ws[pid_k]`.
  2. `tl.debug_barrier()`.
  3. One scalar `tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")` per CTA.
  4. `tl.debug_barrier()`, so the other threads' reloads are ordered after thread 0's acquire. This is the reviewer's open question, resolved by adding the barrier unconditionally.
  5. `if ticket == SPLIT_K - 1`: reload slices 0..SPLIT_K-1 with `cache_modifier=".cg"` and the same `offs_m < M` mask, sum them in fixed order, cast to bf16 once, and do a masked store of `[m, 64]` into `out = torch.empty((m, 4096), bf16)`. Then `tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")`.
  - The kernel has no early return. Every CTA stores its partial and takes exactly one ticket on every path.
- **Release side.** No separate fence instruction: Triton has no `tl` threadfence primitive, and `inline_asm_elementwise` with no tensor operand is untested here. The release comes from the acq_rel gpu-scope atomic, preceded by `bar.sync`. This matches the review's "release fence / acq_rel" alternative.
- **State and prepare().** New state keys `pooler_ws` (fp32 `[2, 32, 4096]`, `torch.empty`, pure scratch; every row < m is rewritten each call before it is read) and `pooler_cnt` (int32 `[64]`, zeros). prepare() warms every m in 1..32 on synthetic x, which compiles both BLOCK_M specializations. It then calls `pooler_cnt.zero_()` (stream-ordered).
- **Dispatch.** `kernel_fn` takes the new path only when `layer == "pooler"` and `_pooler_shape_ok` holds. That check requires cfg m <= 32, cfg n == 4096, cfg k == 8192, x 2-D bf16, x.stride(1) == 1, 1 <= x.shape[0] <= 32, x.shape[1] == 8192, and the weight to be `[4096, 8192]` with stride(1) == 1. Pooler at m >= 128 and every other bf16 layer fall through to the unchanged `_bf16_gemm_dispatch_impl`.
- **Deviations.** None from the spec's `change` paragraph. Two items follow the review rather than adding to the spec: the post-ticket barrier, and warming every m in 1..32 rather than only 1/8/16/32.
- **Uncertainties (cannot run on this node).**
  - Whether Triton accepts the scalar `atomic_add` with `sem`/`scope` and the `atomic_xchg` keyword args on the judge's Triton version. Both are documented in Triton 3.x.
  - Whether the scalar atomic is lowered to a single thread's predicated `atom`. Triton's usual lowering is a thread-0 predicate plus an smem broadcast.
  - The shared workspace assumes pooler calls are not run concurrently on two streams. The judge runs them sequentially.

## Rules check

Each box was checked against `git -C .worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore diff`.

- [x] **CUDA-graph capturable.** The diff adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints. It allocates nothing that depends on values: `out` and the grid depend only on x.shape. The branches are on shapes, `cfg` and dtype only. The counter is 0 at capture and self-resets on every replay.
- [x] **Nothing derived from `data` is cached across calls.** `pooler_ws` is scratch, fully rewritten for rows < m before it is read in the same launch, and rows >= m are never read. `pooler_cnt` holds only ticket counts and returns to 0.
- [x] **Weights.** The pooler bf16 weight is used as is: the same `state["pooler"]` tensor, with no repack and no precision change.
- [x] **Activations are BF16 where SGLang keeps BF16.** x is bf16 and the output is bf16. Accumulation is fp32 with one bf16 rounding, the same as cuBLAS splitK.
- [x] n/a: no KV cache or queries in this problem (dense projections).
- [x] n/a: no attention in this problem.
- [x] **No forbidden operations.** No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff imports nothing new.
- [x] **All work happens inside the timed call; no output caching or replay.** The prepare() warm-up uses synthetic x only, and its outputs are discarded.
- [x] **No `triton.autotune`.** The configs are static constants chosen from m.

okc_lint output:
```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore/muse_bf16_proj/submission.py
```
`python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.
