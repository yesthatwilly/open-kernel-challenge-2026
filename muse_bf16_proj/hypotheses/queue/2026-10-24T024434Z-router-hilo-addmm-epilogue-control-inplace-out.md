---
problem: muse_bf16_proj
slug: router-hilo-addmm-epilogue-control-inplace-out
created: 2026-10-24T024434Z
base: master
area: router GEMM
change: >
  From the parent's branch (master with sglang-gemm-dispatch-parity and router-hilo-at-m8;
  ROUTER_FP32_MAX_TOKENS = 1), apply the parent's change with one difference: the lo GEMM
  accumulates in place into the hi result instead of allocating a new output. In kernel_fn's
  router branch above ROUTER_FP32_MAX_TOKENS replace 'acc = torch.mm(x, state["router_hi_t"],
  out_dtype=torch.float32); acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)'
  with 'acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32); torch.addmm(acc, x,
  state["router_lo_t"], beta=1.0, alpha=1.0, out_dtype=torch.float32, out=acc); return acc'.
  Because out is the same tensor as self, at::native::copy_(result, self) early-returns and no
  elementwise kernel is launched; cuBLAS applies beta=1 in its splitKreduce and writes D over C.
  In prepare(), after building the hi/lo tensors, probe once on synthetic CUDA tensors (a =
  torch.zeros(2, 256, fp32); torch.addmm(a, torch.zeros(2, 8192, bf16), state["router_lo_t"],
  beta=1.0, alpha=1.0, out_dtype=torch.float32, out=a)) inside try/except, set a module-level bool
  _ADDMM_INPLACE_OK, and print it; kernel_fn uses the in-place addmm when the flag is True and the
  parent's two-statement mm + add when False, so a missing overload never fails the run and the
  log_tail states which path was timed. Everything else stays: prepare()'s hi/lo split, the fp32
  F.linear path at m<=1, the bf16 layers on _bf16_gemm_dispatch_impl. Expected kernel sequence per
  router row with m>1: nvjet splitK, splitKreduce, nvjet splitK, splitKreduce, and no
  vectorized_elementwise node at all.
expected:
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: m=8 is on hi/lo on master (14.15 us); the add node and its ~0.4 us gap go away, ~1.3 us or ~9%
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: same, 14.51 us with a 0.9 us add node
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same, 15.03 us with a 1.0 us add node
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: same at m=128, 16.21 us with a 1.1 us add node
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: same at m=256, 17.66 us with a 1.1 us add node
  - shapes: layer=router load=decode_1 *
    direction: flat
    why: m=1 stays on F.linear fp32; control
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: bf16 layers untouched; control
risk: low
depends_on: router-hilo-addmm-epilogue
priority: 1
kind: control
controls_for: router-hilo-addmm-epilogue
review: required
status: queued
---

# router-hilo-addmm-epilogue-control-inplace-out

## Change

From the parent's branch (master with sglang-gemm-dispatch-parity and router-hilo-at-m8; ROUTER_FP32_MAX_TOKENS = 1), apply the parent's change with one difference: the lo GEMM accumulates in place into the hi result instead of allocating a new output. In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS replace 'acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32); acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)' with 'acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32); torch.addmm(acc, x, state["router_lo_t"], beta=1.0, alpha=1.0, out_dtype=torch.float32, out=acc); return acc'. Because out is the same tensor as self, at::native::copy_(result, self) early-returns and no elementwise kernel is launched; cuBLAS applies beta=1 in its splitKreduce and writes D over C. In prepare(), after building the hi/lo tensors, probe once on synthetic CUDA tensors (a = torch.zeros(2, 256, fp32); torch.addmm(a, torch.zeros(2, 8192, bf16), state["router_lo_t"], beta=1.0, alpha=1.0, out_dtype=torch.float32, out=a)) inside try/except, set a module-level bool _ADDMM_INPLACE_OK, and print it; kernel_fn uses the in-place addmm when the flag is True and the parent's two-statement mm + add when False, so a missing overload never fails the run and the log_tail states which path was timed. Everything else stays: prepare()'s hi/lo split, the fp32 F.linear path at m<=1, the bf16 layers on _bf16_gemm_dispatch_impl. Expected kernel sequence per router row with m>1: nvjet splitK, splitKreduce, nvjet splitK, splitKreduce, and no vectorized_elementwise node at all.

## Expected

- `layer=router load=verify_1 *` faster: m=8 is on hi/lo on master (14.15 us); the add node and its ~0.4 us gap go away, ~1.3 us or ~9%
- `layer=router load=decode_16 *` faster: same, 14.51 us with a 0.9 us add node
- `layer=router load=decode_32 *` faster: same, 15.03 us with a 1.0 us add node
- `layer=router load=verify_16 *` faster: same at m=128, 16.21 us with a 1.1 us add node
- `layer=router load=verify_32 *` faster: same at m=256, 17.66 us with a 1.1 us add node
- `layer=router load=decode_1 *` flat: m=1 stays on F.linear fp32; control
- `*layer=qkv_gate*` flat: bf16 layers untouched; control

## Risk

Separates 'cuBLAS's beta=1 epilogue is not cheaper than the separate add' from 'PyTorch's out-of-place addmm re-introduced an elementwise copy kernel'; this is the form to promote if it gains.
