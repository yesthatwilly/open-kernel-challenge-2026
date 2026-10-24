---
problem: muse_decode_layer
slug: prefork-qkv-o-swapab-t16-t32
created: 2026-10-24T184451Z
base: master
area: qkv GEMM + o GEMM (before the router fork)
change: >
  At tokens == 16 and tokens == 32 only (decode_16 and decode_32, both layer types; config gate),
  route the two dense GEMMs that run BEFORE the router side-stream fork through the bf16 problem
  swap-AB Triton kernel: qkv = _gemm(data["attn_input"], dense["qkv_gate"]) (n 2304, k 8192) and
  attn_output = _gemm(o, dense["o"]) (n 8192, k 1024). The pooler, the router, the unpooler and
  every other T keep master exactly. Copy _bf16_swapab_kernel, SWAPAB_BLOCK_W, SWAPAB_NUM_WARPS,
  SWAPAB_FALLBACK_STAGES and the OutOfResources import guard from muse_bf16_proj/submission.py on
  master, with the qkv_gate and o rows of SWAPAB_CFG ((2304, 8192, 4, 128, 4) and (8192, 1024, 1,
  256, 4)). T=32 uses master BLOCK_T = WS_M = 32; T=16 uses BLOCK_T = WS_M = 16 with the qkv_gate
  and o rows of SWAPAB16_CFG from branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend (same
  kernel, smaller token tile, wgmma m64n16). For attribution give each layer its own kernel name:
  two thin @triton.jit wrappers _swapab_qkv_kernel and _swapab_o_kernel that call the shared body
  (or two copies of the kernel), so the profile shows qkv and o rows separately. prepare():
  allocate per-layer fp32 split-K workspaces and zeroed int32 ticket counters (as bf16
  _swapab_prepare does, qkv and o only, sized for WS_M 32), warm each (layer, T) once, and
  self-check each against torch.mm on random bf16 inputs (rel L2 <= 1e-2); any failure or
  OutOfResources at both stage counts sets that layer flag off and keeps _gemm. Print
  "[prefork-swapab] layer=.. T=.. stages=.. ON|OFF".
expected:
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "bf16 m16 swap-AB measured qkv 18% and o 10-19% faster; qkv (nvjet 64x16 4x1 splitK 12.2 us + splitKreduce ~2 us) -> ~10 us, o ~6 -> ~5 us: -3..-4 us of 230 / 385 us, +1-1.5%"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "m32 swap-AB measured qkv +3.6%, o +4.1%: -1 us, flat to +0.5%"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate (restack covers it)
  - shapes: "*load=verify_*"
    direction: flat
    why: T=8/128/256 outside the gate
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T185025Z
review_companions: null
hold_reason: "adversarial review: Its main purpose no longer holds on the current master: small-t-launch-minimal-dense-chain-pdl removed the router side stream at T<=32, so there is no fork for "
---

# prefork-qkv-o-swapab-t16-t32

## Change

At tokens == 16 and tokens == 32 only (decode_16 and decode_32, both layer types; config gate), route the two dense GEMMs that run BEFORE the router side-stream fork through the bf16 problem swap-AB Triton kernel: qkv = _gemm(data["attn_input"], dense["qkv_gate"]) (n 2304, k 8192) and attn_output = _gemm(o, dense["o"]) (n 8192, k 1024). The pooler, the router, the unpooler and every other T keep master exactly. Copy _bf16_swapab_kernel, SWAPAB_BLOCK_W, SWAPAB_NUM_WARPS, SWAPAB_FALLBACK_STAGES and the OutOfResources import guard from muse_bf16_proj/submission.py on master, with the qkv_gate and o rows of SWAPAB_CFG ((2304, 8192, 4, 128, 4) and (8192, 1024, 1, 256, 4)). T=32 uses master BLOCK_T = WS_M = 32; T=16 uses BLOCK_T = WS_M = 16 with the qkv_gate and o rows of SWAPAB16_CFG from branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend (same kernel, smaller token tile, wgmma m64n16). For attribution give each layer its own kernel name: two thin @triton.jit wrappers _swapab_qkv_kernel and _swapab_o_kernel that call the shared body (or two copies of the kernel), so the profile shows qkv and o rows separately. prepare(): allocate per-layer fp32 split-K workspaces and zeroed int32 ticket counters (as bf16 _swapab_prepare does, qkv and o only, sized for WS_M 32), warm each (layer, T) once, and self-check each against torch.mm on random bf16 inputs (rel L2 <= 1e-2); any failure or OutOfResources at both stage counts sets that layer flag off and keeps _gemm. Print "[prefork-swapab] layer=.. T=.. stages=.. ON|OFF".

## Expected

- `*load=decode_16 *` faster: bf16 m16 swap-AB measured qkv 18% and o 10-19% faster; qkv (nvjet 64x16 4x1 splitK 12.2 us + splitKreduce ~2 us) -> ~10 us, o ~6 -> ~5 us: -3..-4 us of 230 / 385 us, +1-1.5%
- `*load=decode_32 *` faster: m32 swap-AB measured qkv +3.6%, o +4.1%: -1 us, flat to +0.5%
- `*load=decode_1 *` flat: T=1 outside the gate (restack covers it)
- `*load=verify_*` flat: T=8/128/256 outside the gate

## Risk

Mostly an attribution probe for the open LESSONS line: bf16-shape-gated-triton-gemms-small-t lost 2-4.5% at T=8/16 despite faster kernel sums, "possibly interference with the side-stream router overlap". qkv and o run before the fork, so if their own rows get faster and the shape gets faster, the loss came from the pooler/unpooler side; if the rows get faster and the shape does not, the graph-node or ticket overhead is to blame. Overlaps the held bf16-m32-swapab-wgmma-port (qkv/o/pooler at T=32); this is narrower (pre-fork layers only) and adds T=16. The in-judge restack gates its Triton GEMMs to T=1, so no shape overlap.

## Adversarial review

Reviewed 2026-10-24T185025Z by an adversarial subagent (model opus).

**Verdict: hold.** Its main purpose no longer holds on the current master: small-t-launch-minimal-dense-chain-pdl removed the router side stream at T<=32, so there is no fork for qkv/o to sit before and no overlap interference left to attribute. The remaining kernel gain is about 1-4 us at T=16 and under 1 us at T=32, inside the ~1.5% shape noise (decode_16 sliding err ~4 us), and stage 2 of the landed rewrite plans to replace these same qkv and o GEMMs, so this work would be thrown away.

### Justification
Mechanism: swap-AB wgmma Triton for qkv_gate (n2304,k8192) and o (n8192,k1024) at m=16/32 is real and measured on the H200 judge. In bf16_proj bf16-m16-swapab-wgmma-extend, the qkv kernel is 10.1 us at m16 and 11.7 us at m32, and o is 4.4 / 4.6 us. Shape wall-clock against SGLang cuBLAS: qkv 13.28 vs 14.11 us and o 6.49 vs 7.17 us at m16; qkv 14.20 vs 14.38 us and o 6.73 vs 7.12 us at m32. The hypothesis's 'qkv 18%, o 10-19%' compares kernel time with kernel+splitKreduce and does not match those wall-clock deltas. qkv_gate decode_16 was graded flat (1.021) and decode_32 flat (0.973) there. A realistic in-graph saving is about 1.5-4 us at T=16 (217/348 us, under 1.8% sliding and 1.1% full) and about 0.6-1 us at T=32 (247/535 us, about 0.3%). The attribution noise is 0.015 per shape and decode_16 sliding err_ns is 4.1 us, so no headline or shape row is readable; only the kernel rows are. The numbers are also stale: the hypothesis quotes 230/385 us, but the current master is 217.1/347.5 us at decode_16. Weakest link: the stated reason for the run. Risk calls it an attribution probe for bf16-shape-gated-triton-gemms-small-t's T=16 loss ('possibly interference with the side-stream router overlap'), reasoning that 'qkv and o run before the fork'. On the current master (submission.py ~1353-1377, DL path at T<=DL_MAX_TOKENS) the side stream, wait_stream and record_stream are gone at T<=32, and pooler+router run as one PDL kernel (dl_pooler_router_kernel). There is no fork, so the experiment cannot answer its own question; the old loss was measured on a base that no longer exists. Overlap: at T=32 this is a strict subset of the held bf16-m32-swapab-wgmma-port (qkv/o/pooler/unpooler at T=32), which was held for being about 0.15% headline and inside drift. This hypothesis is smaller. Stage 2 of the landed rewrite (small-t-launch-minimal-dense-chain-pdl STAGES (2)) plans a qkv split-K GEMV with fused QK-norm/RoPE/FP8 cache-store epilogue and an o GEMV with fused norm-gate prologue at T<=32. Those replace exactly the two kernels this hypothesis swaps.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (base): removed the side-stream fork at T<=32, which makes the hypothesis's attribution question void. Its PDL kernel P follows attn_boundary, not the o GEMM, so swapping o should not change PDL overlap. Still, any change in o's tail or launch timing moves the graph node gap before attn_boundary->P. A small shape delta could come from that gap rather than from GEMM speed.
- post-expert-norm-triton-small-t: different area (post-MoE combine) and different memory traffic; no meaningful confound.
- marlin-block-m-cap-16: Marlin at T=16/32 is unchanged by the cap (it picks 8/16 anyway); no confound.
- fa3-hdim128-head-pairing-batched: the full-layer decode_16/32 FA3 path sits between qkv and o, so FA3 run-to-run variance (145.7 / 291.7 us rows) dominates the full-layer noise floor where this change is 1 us. That makes the full rows unreadable, not confounded.
- sglang-pipeline-parity-router-hilo-m8: superseded at T<=32 by the dense-chain kernel; no confound.

### Interference and A/B needs
A companion cannot rescue this. With the fork gone, there is nothing to revert that would recreate the side-stream condition being probed. An ab-revert of small-t-launch-minimal-dense-chain-pdl would restore the fork, but that tests an obsolete base and costs a check to learn about code that will not return. A per-GEMM control (qkv only, or o only) splits a 1-4 us effect into two smaller effects that are each further inside noise. The only readable signal is the kernel rows in the profile (_swapab_qkv_kernel / _swapab_o_kernel against the nvjet rows they replace), and the per-layer kernel names already give that. If it is ever run, judge it on those rows alone, not on the headline.

### Open questions
- Should the T=16 swap-AB tile config (BLOCK_W/stages for wgmma m64n16) instead go into the planned stage 2 of small-t-launch-minimal-dense-chain-pdl as the starting body for its qkv split-K GEMV and o GEMV? That puts the gain where it compounds with the fused epilogue and prologue.
- Does the old bf16-shape-gated-triton-gemms-small-t T=16 loss (faster kernel sums, slower shape) still need explaining now that the fork is gone? If so, the better probe is a pooler/unpooler-side change on the current master, since the pooler is now inside kernel P.
- If kept for the combined decode GEMM restack named in bf16-m32-swapab-wgmma-port's hold_reason, should T=16 qkv/o be merged into that one item rather than spending a separate check?

