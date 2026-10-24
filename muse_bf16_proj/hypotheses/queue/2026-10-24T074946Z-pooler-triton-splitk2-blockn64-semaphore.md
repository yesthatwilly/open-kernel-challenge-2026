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
review: required
status: queued
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
