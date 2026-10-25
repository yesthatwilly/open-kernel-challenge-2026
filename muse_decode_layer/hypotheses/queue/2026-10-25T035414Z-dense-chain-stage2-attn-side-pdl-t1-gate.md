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
review: proceed
status: queued
review_at: 2026-10-25T040253Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** The diagnosis is right: dl_o_kernel does not scale past T=1 (0.41 TB/s at T=32), and narrowing to T=1 keeps the one measured win (full decode_1 -3.7 us, z -3.9) while removing the +6 to +13.5% losses. Sharpen the gate to full-attention T=1 only (DL2_SLIDING1='nvjet'), so sliding decode_1 becomes an in-run drift control for the single target row.

### Justification
The source run used the current master blob (base_blob 4da9467), so its rows compare directly to BEST. At T=1, Q ran 8.4 us at 4.5 TB/s and O 6.4 us. Full decode_1 measured 139.3 vs BEST 142.9/145.9 and vs the control run's 144.1, so -3.6 to -4.8 us, about 6-8 err. Sliding decode_1 was 125.8 vs control 126.5 (-0.7 us, within err), and the 03:41Z harvest explicitly recommended full only. The patch already has the knobs: DL2_MAX_TOKENS (its own bound, also sizing ws_q/ws_o) and _dl2_applies(), which honours DL2_SLIDING1 != 'raw' to keep sliding decode_1 on master's nvjet + _sliding_forward. Setting DL2_MAX_TOKENS = 1 and DL2_SLIDING1 = 'nvjet' implements the narrowed gate without touching kernel code. One check is readable: a single target row at err ~0.6 us with an expected -3.7 us, plus the dl_qkv_kernel/dl_o_kernel profile rows. With sliding decode_1 off, that row reads batch drift in the same run. Weakest link: the gain is small (~+0.003-0.005 headline) and lives on one shape, so landing will need a submit-grade read either way.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (master): stage 2's O kernel PDL-chains into the stage-1 chain (boundary -> P). The T=1 gain partly depends on P's PDL pre-wait overlap, which is unchanged and landed.
- fa3-hdim128-head-pairing-batched (master): pairing is batch>1 only, so it does not apply at decode_1 (batch 1). Q writes the paired pad layout only where paired, so no interaction at T=1.
- stack-route-topk-pdl-unpool-nf1 (queued sibling): it also improves T=1 (R and unpool). Run both independently on master. The patches conflict in _DL_CUDA (verified: dense-chain-stage2 patch fails to apply after the route patch at ~line 295), so whichever lands second must hand-merge the TU with one module name. Their T=1 gains may be sub-additive through the shared PDL chain.

### Interference and A/B needs
No companion. With DL2_SLIDING1='nvjet' the sliding decode_1 row is master code in the same run and serves as the control. T>1 shapes return to master byte for byte. A separate ab run would only duplicate that.

### Open questions
- BUILDER: implement the gate with the existing constants: DL2_MAX_TOKENS = 1 (also shrinks ws_q/ws_o) and DL2_SLIDING1 = 'nvjet' (full-only at T=1), rename the module, and drop the NF=2/4 instantiations and T>1 self-check cases. Print PATH CHOSEN per T, showing Q/O only at full T=1.
- BUILDER: make the selector compare its printed T=1 timing (Q + FA3 + O vs master nvjet qkv + FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) and assert the custom path wins before enabling it (LESSONS 163).
- If the user prefers to keep sliding T=1 on (the spec as written), the run is still readable on the full decode_1 row, but then there is no in-run decode_1 drift control.

