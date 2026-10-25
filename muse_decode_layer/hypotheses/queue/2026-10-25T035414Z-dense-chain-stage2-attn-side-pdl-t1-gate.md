---
problem: muse_decode_layer
slug: dense-chain-stage2-attn-side-pdl-t1-gate
created: 2026-10-25T035414Z
base: master
area: qkv GEMM + o GEMM + attention-side small kernels (T=1 only)
change: >
  Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and
  narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its run (b31b8453, check 1.2253x)
  won full decode_1 by -3.7 us (-2.6%, z-3.9; sliding decode_1 -1.4%, flat) but lost at T=16/32
  because dl_o_kernel does not scale past T=1 (O at T=32 40.8 us, 0.41 TB/s; sliding decode_16
  +9.2%, decode_32 +13.5%, full decode_32 +6.0%). Change only the predicate that sets/uses
  state["dl2_ok"] routing in kernel_fn: Q (dl_qkv_kernel<NF>) and O (dl_o_kernel<NF>) run only
  when tokens == 1; every T in 2..32 and above returns to master attention side (nvjet qkv +
  FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) byte for byte, with the
  landed stage-1 dense chain (P, R, Marlin) unchanged. Keep both kernels compiled at NF=1 only if
  the template allows (drop the NF=2/4 instantiations to cut build time; otherwise leave them
  unused). Self-check: keep the existing Q/O/FA3/graph-replay cases at T=1 for both the full and
  sliding branches; drop T>1 cases. Print PATH CHOSEN per T so the log shows Q/O only at T=1.
  First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: stage 2 measured -3.7 us (-2.6%, z-3.9) at full decode_1 (Q 8.4 us at 4.5 TB/s, O 6.4 us at T=1)
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: stage 2 read -1.4% here, within err
  - shapes: "*load=decode_16 * *load=decode_32 * *load=verify_*"
    direction: flat
    why: back on master attention side; removes the +6-13.5% dl_o_kernel losses
risk: medium
depends_on: dense-chain-stage2-attn-side-pdl
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# dense-chain-stage2-attn-side-pdl-t1-gate

## Change

Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its run (b31b8453, check 1.2253x) won full decode_1 by -3.7 us (-2.6%, z-3.9; sliding decode_1 -1.4%, flat) but lost at T=16/32 because dl_o_kernel does not scale past T=1 (O at T=32 40.8 us, 0.41 TB/s; sliding decode_16 +9.2%, decode_32 +13.5%, full decode_32 +6.0%). Change only the predicate that sets/uses state["dl2_ok"] routing in kernel_fn: Q (dl_qkv_kernel<NF>) and O (dl_o_kernel<NF>) run only when tokens == 1; every T in 2..32 and above returns to master attention side (nvjet qkv + FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) byte for byte, with the landed stage-1 dense chain (P, R, Marlin) unchanged. Keep both kernels compiled at NF=1 only if the template allows (drop the NF=2/4 instantiations to cut build time; otherwise leave them unused). Self-check: keep the existing Q/O/FA3/graph-replay cases at T=1 for both the full and sliding branches; drop T>1 cases. Print PATH CHOSEN per T so the log shows Q/O only at T=1. First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: stage 2 measured -3.7 us (-2.6%, z-3.9) at full decode_1 (Q 8.4 us at 4.5 TB/s, O 6.4 us at T=1)
- `*layer=sliding load=decode_1 *` flat: stage 2 read -1.4% here, within err
- `*load=decode_16 * *load=decode_32 * *load=verify_*` flat: back on master attention side; removes the +6-13.5% dl_o_kernel losses

## Risk

Small single-shape gain (~+0.003-0.005 headline) near check resolution; read the full decode_1 row (err ~0.6 us) and the dl_qkv/dl_o profile rows. The patch also renames the _DL_CUDA module; if stack-route-topk-pdl-unpool-nf1 lands first, rebase onto it and merge the TU by hand.
