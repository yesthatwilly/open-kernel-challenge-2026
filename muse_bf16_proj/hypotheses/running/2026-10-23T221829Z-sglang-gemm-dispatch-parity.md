---
problem: muse_bf16_proj
slug: sglang-gemm-dispatch-parity
created: 2026-10-23T221829Z
base: master
area: bf16 GEMM dispatch
change: >
  Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses:
  sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four
  bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(),
  router) at m <= 8 but build the BF16 hi + lo split in prepare() and run two torch.mm(...,
  out_dtype=torch.float32) GEMMs above 8 tokens, as baseline.py does. No other change. This
  establishes parity (expected ~1.0x) as the base every later bf16_proj hypothesis stacks on, and
  attributes the starter's 10% loss to the GEMM dispatch path alone.
expected:
  - shapes: layer=router load=verify_*
    direction: faster
    why: "fp32 F.linear on m=128/256 is the starter's worst shapes; the hi/lo bf16 GEMMs are what SGLang runs there"
  - shapes: layer=qkv_gate *
    direction: faster
    why: cuBLAS/nvjet dispatch picks splitK kernels the plain matmul does not
  - shapes: layer=router load=decode_*
    direction: flat
    why: same F.linear path at m<=8
risk: low
depends_on: null
priority: 1
status: running
stage: claimed
claimed_at: 2026-10-23T221905Z
base_sha: 965a5a2c44b96f0f62ef2d7cd77e99c879e9d3df
base_blob: 6ce773a44036578450b1489df0030d65d2c67c0e
branch: hyp/muse_bf16_proj/sglang-gemm-dispatch-parity
worktree: .worktrees/muse_bf16_proj/sglang-gemm-dispatch-parity
mode: null
---

# sglang-gemm-dispatch-parity

## Change

Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(), router) at m <= 8 but build the BF16 hi + lo split in prepare() and run two torch.mm(..., out_dtype=torch.float32) GEMMs above 8 tokens, as baseline.py does. No other change. This establishes parity (expected ~1.0x) as the base every later bf16_proj hypothesis stacks on, and attributes the starter's 10% loss to the GEMM dispatch path alone.

## Expected

- `layer=router load=verify_*` faster: fp32 F.linear on m=128/256 is the starter's worst shapes; the hi/lo bf16 GEMMs are what SGLang runs there
- `layer=qkv_gate *` faster: cuBLAS/nvjet dispatch picks splitK kernels the plain matmul does not
- `layer=router load=decode_*` flat: same F.linear path at m<=8

## Risk

Low risk: it is SGLang's own code path. Only risk is an import path difference inside the judge image; baseline.py imports the same symbol, so it exists there.
