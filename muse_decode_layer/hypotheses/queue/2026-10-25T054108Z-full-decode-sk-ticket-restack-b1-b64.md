---
problem: muse_decode_layer
slug: full-decode-sk-ticket-restack-b1-b64
created: 2026-10-25T054108Z
base: master
area: "FA3 call (full decode, all batches): split-KV A/B at 1<B<=64 and FA3 chain at B=1"
change: >
  Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention-side
  stream-K ports that each won their rows on the previous master, in ONE muse_decode_sk TU with
  two template instantiations and disjoint config gates: (a) from branch
  hyp/muse_decode_layer/full-decode-streamk-ticket-port, splitkv_attn_sk_kernel<4,true,false>
  (single-level ticket, grid 2 x SM) replacing master's splitkv_attn_kernel +
  splitkv_combine_kernel at full layer, q_len 1, page 1, 1 < batch <= 64 (run 6d48932b: selector
  B=16 107.6 vs 120.6 us, B=32 222.8 vs 257.4; profile 109.7 / 227.3 us; full decode_16
  306.7->296.4, decode_32 479.6->437.5); (b) from branch
  hyp/muse_decode_layer/full-decode1-two-level-ticket-port, splitkv_attn_sk_kernel<4,true,true>
  (two-level ticket, grid = SM count) replacing FA3 + FlashAttnFwdCombine + pad/unpad copies +
  FusedRMSNormGate at full decode batch == 1 (run 5e44c7ae: selector 24.8 vs 34.1 us; profile 20.1
  us; full decode_1 142.7->137.7). Both read the q prepared by master's split-KV pre-attention
  route (extend it to B=1 as (b) did) and write the bf16 tensor the o GEMM reads; master's kernel
  A/B and FA3 chain stay compiled as fallbacks. Sliding layers, verify and full decode B>64
  untouched. The previous ports kept the ticket buffer as a module-level import-time zero buffer:
  MOVE it (both ticket levels) and the fp32 partial workspace into the per-state dict, sized from
  config in prepare; the kernel resets tickets to 0 on exit, and the self-check must prove it
  across alternating states. Self-check per gate: B in {1,2,5,16,32,33} with agentic and uniform
  lengths, kv_scale 1 and 0.25, vs master kernels and fp32 reference; selectors at B=1,16,32. HARD
  RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except,
  allocation on input values, data-dependent control flow or per-call mutable state; gates read
  config only (layer, batch, q_len, window, page); nothing cached from inputs; precision per
  TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES:
  the judge prepares two state copies and rotates them, so every ticket buffer, partial workspace,
  device pointer and descriptor lives in the per-state dict built in prepare(), never in a
  module/static global (LESSONS moe g_maps). Self-check: capture once per state, then alternate
  graph replays of state 0 and state 1 with refilled inputs (new q/KV/lengths each replay) and
  compare each to master; any miss falls back to master path for that gate and prints PATH CHOSEN
  ... (fallback: <reason>). Build sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a, -std=c++20 if cflags
  overridden), print regs/smem/local/binaryVersion, first launch a tiny canary + synchronize with
  launching/ok prints. The prepare-time selector prints CUDA-event timing new vs incumbent per
  gated batch and keeps the incumbent if new is slower. First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced splitkv A+B 121.1 us -> sk single-level 109.7 (profile 6d48932b); shape on new master 308.7 -> ~297. Kill: sk row >= 116 us"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced 257.1 -> 227.3 us; shape 473.3 -> ~440 (z ~ -5 in the port run). Kill: sk row >= 245"
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "FA3+combine+pad+norm-gate ~30 us -> sk two-level 20.1 us; shape 139.2 -> ~134. Kill: not below 137 (z>=2)"
  - shapes: "*layer=sliding*"
    direction: flat
    why: outside every gate; sliding decode_1 is the drift control (123.9)
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# full-decode-sk-ticket-restack-b1-b64

## Change

Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention-side stream-K ports that each won their rows on the previous master, in ONE muse_decode_sk TU with two template instantiations and disjoint config gates: (a) from branch hyp/muse_decode_layer/full-decode-streamk-ticket-port, splitkv_attn_sk_kernel<4,true,false> (single-level ticket, grid 2 x SM) replacing master's splitkv_attn_kernel + splitkv_combine_kernel at full layer, q_len 1, page 1, 1 < batch <= 64 (run 6d48932b: selector B=16 107.6 vs 120.6 us, B=32 222.8 vs 257.4; profile 109.7 / 227.3 us; full decode_16 306.7->296.4, decode_32 479.6->437.5); (b) from branch hyp/muse_decode_layer/full-decode1-two-level-ticket-port, splitkv_attn_sk_kernel<4,true,true> (two-level ticket, grid = SM count) replacing FA3 + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate at full decode batch == 1 (run 5e44c7ae: selector 24.8 vs 34.1 us; profile 20.1 us; full decode_1 142.7->137.7). Both read the q prepared by master's split-KV pre-attention route (extend it to B=1 as (b) did) and write the bf16 tensor the o GEMM reads; master's kernel A/B and FA3 chain stay compiled as fallbacks. Sliding layers, verify and full decode B>64 untouched. The previous ports kept the ticket buffer as a module-level import-time zero buffer: MOVE it (both ticket levels) and the fp32 partial workspace into the per-state dict, sized from config in prepare; the kernel resets tickets to 0 on exit, and the self-check must prove it across alternating states. Self-check per gate: B in {1,2,5,16,32,33} with agentic and uniform lengths, kv_scale 1 and 0.25, vs master kernels and fp32 reference; selectors at B=1,16,32. HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on input values, data-dependent control flow or per-call mutable state; gates read config only (layer, batch, q_len, window, page); nothing cached from inputs; precision per TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES: the judge prepares two state copies and rotates them, so every ticket buffer, partial workspace, device pointer and descriptor lives in the per-state dict built in prepare(), never in a module/static global (LESSONS moe g_maps). Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with refilled inputs (new q/KV/lengths each replay) and compare each to master; any miss falls back to master path for that gate and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a, -std=c++20 if cflags overridden), print regs/smem/local/binaryVersion, first launch a tiny canary + synchronize with launching/ok prints. The prepare-time selector prints CUDA-event timing new vs incumbent per gated batch and keeps the incumbent if new is slower. First measurement is an okc check.

## Expected

- `*layer=full load=decode_16*` faster: replaced splitkv A+B 121.1 us -> sk single-level 109.7 (profile 6d48932b); shape on new master 308.7 -> ~297. Kill: sk row >= 116 us
- `*layer=full load=decode_32*` faster: replaced 257.1 -> 227.3 us; shape 473.3 -> ~440 (z ~ -5 in the port run). Kill: sk row >= 245
- `*layer=full load=decode_1 *` faster: FA3+combine+pad+norm-gate ~30 us -> sk two-level 20.1 us; shape 139.2 -> ~134. Kill: not below 137 (z>=2)
- `*layer=sliding*` flat: outside every gate; sliding decode_1 is the drift control (123.9)
- `*load=verify_*` flat: outside the gate

## Risk

Land candidate if the three full decode rows reproduce: the kernels are unchanged from runs that passed every test; the new risk is only the per-state ticket move. Judge on the three full decode rows and the sk kernel profile rows, not the headline (sum of row gains ~+0.008-0.012).
