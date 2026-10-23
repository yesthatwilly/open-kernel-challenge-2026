---
problem: muse_bf16_proj
slug: router-hilo-addmm-epilogue
created: 2026-10-23T225128Z
base: master
area: router GEMM
change: >
  In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate
  'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc += torch.mm(x, router_lo_t,
  out_dtype=torch.float32)' with 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc =
  torch.addmm(acc, x, router_lo_t, beta=1.0, alpha=1.0, out_dtype=torch.float32)' so the lo-GEMM's
  cuBLAS epilogue adds the hi accumulator (beta=1) instead of a separate at::native elementwise
  add kernel. Everything else stays: prepare(), the hi/lo split, the fp32 F.linear path at m<=8,
  and the bf16 layers. Kernel moved: the vectorized_elementwise_kernel (0.9-1.0 us GPU plus its
  ~1-2 us graph-node gap) on every router shape with m>8 disappears; the two nvjet splitK GEMMs
  and their splitKreduce stay (the reduce is where cuBLAS applies beta, so it should cost nothing
  extra).
expected:
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: one fewer launch of five (14.8 us reported, 12.5 us of kernels)
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same, 5 launches to 4
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: same at m=128
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: same at m=256; add kernel there is 1.0 us + gap
  - shapes: layer=router load=decode_1 *
    direction: flat
    why: m=1 stays on fp32 F.linear; control
  - shapes: layer=router load=verify_1 *
    direction: flat
    why: m=8 stays on fp32 F.linear unless router-hilo-at-m8 has landed first
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# router-hilo-addmm-epilogue

## Change

In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc += torch.mm(x, router_lo_t, out_dtype=torch.float32)' with 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc = torch.addmm(acc, x, router_lo_t, beta=1.0, alpha=1.0, out_dtype=torch.float32)' so the lo-GEMM's cuBLAS epilogue adds the hi accumulator (beta=1) instead of a separate at::native elementwise add kernel. Everything else stays: prepare(), the hi/lo split, the fp32 F.linear path at m<=8, and the bf16 layers. Kernel moved: the vectorized_elementwise_kernel (0.9-1.0 us GPU plus its ~1-2 us graph-node gap) on every router shape with m>8 disappears; the two nvjet splitK GEMMs and their splitKreduce stay (the reduce is where cuBLAS applies beta, so it should cost nothing extra).

## Expected

- `layer=router load=decode_16 *` faster: one fewer launch of five (14.8 us reported, 12.5 us of kernels)
- `layer=router load=decode_32 *` faster: same, 5 launches to 4
- `layer=router load=verify_16 *` faster: same at m=128
- `layer=router load=verify_32 *` faster: same at m=256; add kernel there is 1.0 us + gap
- `layer=router load=decode_1 *` flat: m=1 stays on fp32 F.linear; control
- `layer=router load=verify_1 *` flat: m=8 stays on fp32 F.linear unless router-hilo-at-m8 has landed first

## Risk

Risk: torch.addmm's out_dtype kwarg (fp32 accumulate from bf16 operands with an fp32 self) exists in torch >= 2.8 alongside torch.mm's; if the image's torch rejects it (TypeError or a dtype-mismatch check requiring self.dtype == out_dtype, which it is here, fp32) the run fails at the first router test, costing one check and nothing else. Fallback if rejected: torch.mm(..., out=acc) is not an accumulate, so there is no PyTorch-only substitute; the fused Triton kernel is the other route. Gain is small (~2 us of 15-17) and may sit inside the 1.5% CI on the load, so read the per-shape router rows, not the headline. If router-triton-hilo-fused lands first, this becomes moot; keep for attribution of the add-kernel cost.
