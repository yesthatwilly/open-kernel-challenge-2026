---
problem: muse_decode_layer
slug: decode-restack-sliding-port-paired-pad-gemms-t1
created: 2026-10-24T174421Z
base: master
area: restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs)
change: >
  Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t)
  by a 3-way merge of their branches, resolving conflicts by hand (the code paths are disjoint).
  (1) sliding-fused-triton-decode1-port as is: sliding decode_1 +3.4-4.3%, _sliding_fused_kernel
  13.7 us replaces FA3 main + combine + glue. (2) paired-heads-pad-unpad-triton as is: full
  decode_16/32 and verify_16 +1-3%, elementwise glue launches gone. (3)
  bf16-shape-gated-triton-gemms-small-t, but gate its Triton split-K GEMMs to tokens == 1 ONLY. At
  T=8/16 they lost 2-4.5% despite faster kernel sums, so T>1 keeps master's GEMM path exactly.
  Known conflicts: (1) and (2) both insert helpers at the same spot and edit the kernel_fn
  attention block around qk_norm; (3) also touches the module top, the helper block and the qkv =
  _gemm(...) line where (1) hooks in. Keep each piece's own self-check. Nothing else changes.
expected:
  - shapes: "*layer=sliding*load=decode_1*"
    direction: faster
    why: sliding fused port plus the T=1 GEMMs
  - shapes: "*layer=full*load=decode_16*"
    direction: faster
    why: paired pad/unpad
  - shapes: "*load=verify_1*"
    direction: flat
    why: GEMMs gated off at T=8
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# decode-restack-sliding-port-paired-pad-gemms-t1

## Change

Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t) by a 3-way merge of their branches, resolving conflicts by hand (the code paths are disjoint). (1) sliding-fused-triton-decode1-port as is: sliding decode_1 +3.4-4.3%, _sliding_fused_kernel 13.7 us replaces FA3 main + combine + glue. (2) paired-heads-pad-unpad-triton as is: full decode_16/32 and verify_16 +1-3%, elementwise glue launches gone. (3) bf16-shape-gated-triton-gemms-small-t, but gate its Triton split-K GEMMs to tokens == 1 ONLY. At T=8/16 they lost 2-4.5% despite faster kernel sums, so T>1 keeps master's GEMM path exactly. Known conflicts: (1) and (2) both insert helpers at the same spot and edit the kernel_fn attention block around qk_norm; (3) also touches the module top, the helper block and the qkv = _gemm(...) line where (1) hooks in. Keep each piece's own self-check. Nothing else changes.

## Expected

- `*layer=sliding*load=decode_1*` faster: sliding fused port plus the T=1 GEMMs
- `*layer=full*load=decode_16*` faster: paired pad/unpad
- `*load=verify_1*` flat: GEMMs gated off at T=8

## Risk

Restack of measured pieces; expected headline about 1.18 vs 1.165 (harvest analysis). Run as a scored experiment.
