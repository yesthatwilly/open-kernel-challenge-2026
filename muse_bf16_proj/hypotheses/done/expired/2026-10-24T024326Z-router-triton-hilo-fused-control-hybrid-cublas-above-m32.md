---
problem: muse_bf16_proj
slug: router-triton-hilo-fused-control-hybrid-cublas-above-m32
created: 2026-10-24T024326Z
base: master
area: router GEMM
change: >
  From hyp/muse_bf16_proj/router-triton-hilo-fused (commit a392f0b6dc; rebase onto master f9dc05d
  first, the conflict is only the docstring/ROUTER_FP32_MAX_TOKENS hunks the parent deletes), keep
  _router_hilo_kernel, _router_forward and _router_config exactly as the parent for m <= 32
  (m<=16: BLOCK_M=16, BLOCK_N=64, SPLIT_K=8; 17..32: BLOCK_M=32, BLOCK_N=64, SPLIT_K=8; 32 CTAs;
  tl.atomic_add into torch.zeros((m,256), fp32) created right before the launch). For x.shape[0] >
  32 do not launch Triton at all: return the cuBLAS hi/lo pair exactly as master does, acc =
  torch.mm(x, state['router_hi_t'], out_dtype=torch.float32); acc += torch.mm(x,
  state['router_lo_t'], out_dtype=torch.float32). prepare() therefore stores both layouts, built
  from the weights alone: router_hi/router_lo as [256, 8192] bf16 row-major for the Triton path
  and router_hi_t/router_lo_t = hi.t().contiguous()/lo.t().contiguous() for torch.mm (8.4 MB extra
  state, lossless). Remove the parent's m<128 (BLOCK_M=64) and m>=128 (SPLIT_K=1) buckets so no m
  reaches an under-filled grid; ROUTER_WARM_MS becomes (1, 8, 16, 32). The threshold branches on
  x.shape[0] only. bf16 layers untouched on _bf16_gemm_dispatch_impl. The fp32 F.linear fallback
  for shapes outside the kernel's preconditions stays.
expected:
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: parent measured 7.7 us vs master 10.5 us with this exact config
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: "parent measured 7.6 us vs master's cuBLAS hi/lo 14.2 us (router-hilo-at-m8)"
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: parent measured 8.0 us vs master 14.5 us
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: parent measured 8.4 us vs master 15.0 us
  - shapes: layer=router load=verify_16 *
    direction: flat
    why: "master's own torch.mm hi/lo path, 16.2 us; control"
  - shapes: layer=router load=verify_32 *
    direction: flat
    why: "master's own torch.mm hi/lo path, 17.7 us; control"
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: bf16 layers untouched; control
risk: low
depends_on: router-triton-hilo-fused-control-splitk-all-m
priority: 1
kind: control
controls_for: router-triton-hilo-fused-control-splitk-all-m
review: required
status: done
hold_reason: "superseded: router-triton-hilo-fused-control-splitk-all-m landed (1.151x); its 3-D Triton grid beats cuBLAS hi/lo at every m (5.7-9.3 us vs 10.5-17.7 us), so a hybrid using cuBLAS above m=32 can only be slower"
outcome: expired
verdict_detail: "hold expired after 41 h: superseded: router-triton-hilo-fused-control-splitk-all-m landed (1.151x); its 3-D Triton grid beats cuBLAS hi/lo at eve"
expired_at: 2026-10-25T211328Z
---

# router-triton-hilo-fused-control-hybrid-cublas-above-m32

## Change

From hyp/muse_bf16_proj/router-triton-hilo-fused (commit a392f0b6dc; rebase onto master f9dc05d first, the conflict is only the docstring/ROUTER_FP32_MAX_TOKENS hunks the parent deletes), keep _router_hilo_kernel, _router_forward and _router_config exactly as the parent for m <= 32 (m<=16: BLOCK_M=16, BLOCK_N=64, SPLIT_K=8; 17..32: BLOCK_M=32, BLOCK_N=64, SPLIT_K=8; 32 CTAs; tl.atomic_add into torch.zeros((m,256), fp32) created right before the launch). For x.shape[0] > 32 do not launch Triton at all: return the cuBLAS hi/lo pair exactly as master does, acc = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32); acc += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32). prepare() therefore stores both layouts, built from the weights alone: router_hi/router_lo as [256, 8192] bf16 row-major for the Triton path and router_hi_t/router_lo_t = hi.t().contiguous()/lo.t().contiguous() for torch.mm (8.4 MB extra state, lossless). Remove the parent's m<128 (BLOCK_M=64) and m>=128 (SPLIT_K=1) buckets so no m reaches an under-filled grid; ROUTER_WARM_MS becomes (1, 8, 16, 32). The threshold branches on x.shape[0] only. bf16 layers untouched on _bf16_gemm_dispatch_impl. The fp32 F.linear fallback for shapes outside the kernel's preconditions stays.

## Expected

- `layer=router load=decode_1 *` faster: parent measured 7.7 us vs master 10.5 us with this exact config
- `layer=router load=verify_1 *` faster: parent measured 7.6 us vs master's cuBLAS hi/lo 14.2 us (router-hilo-at-m8)
- `layer=router load=decode_16 *` faster: parent measured 8.0 us vs master 14.5 us
- `layer=router load=decode_32 *` faster: parent measured 8.4 us vs master 15.0 us
- `layer=router load=verify_16 *` flat: master's own torch.mm hi/lo path, 16.2 us; control
- `layer=router load=verify_32 *` flat: master's own torch.mm hi/lo path, 17.7 us; control
- `*layer=qkv_gate*` flat: bf16 layers untouched; control

## Risk

The promotable composite of what has already been measured: if the control's m>=128 rows lose to cuBLAS, this run carries the parent's 2.6-6.6 us/shape small-m wins onto master (~1.05x over master's 1.0219x by load arithmetic). If the control wins everywhere it is superseded at the cost of one check slot. Its m<=32 rows are also the same-day baseline for the control's small-m config.
