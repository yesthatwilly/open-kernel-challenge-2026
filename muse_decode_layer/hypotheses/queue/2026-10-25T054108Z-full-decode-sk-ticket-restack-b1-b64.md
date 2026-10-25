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
review: proceed
status: queued
review_at: 2026-10-25T054426Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T054426Z by an adversarial subagent (model opus).

**Verdict: proceed.** This restacks two attention kernels, each of which already won its own gated rows (6d48932b on full decode_16/32, 5e44c7ae on full decode_1), onto master 00e36e3. The gates do not overlap, and the landed changes that share these rows are serial MoE or GEMM work in the same graph, so the three full-decode rows can each be read on their own. No companion run is needed; spend one check.

### Justification
A. The mechanism is plausible. On master, full decode_16 and decode_32 spend 112.4 µs (37.5%) and 248.1 µs (52%) in splitkv_attn_kernel, against shape floors of 137.5 and 208.1 µs. Their attention is a pure FP8 KV stream at ctx about 80K, so a stream-K single-level-ticket schedule that removes the tail imbalance of agentic-length chunks and folds the combine into the last CTA should save the measured 11 and 30 µs. The port profile shows sk rows of 109.7 and 227.3 µs against 121.1 and 257.1 for kernels A+B. At full decode_1, FA3 main (18.5 µs) plus FlashAttnFwdCombine, the pad/unpad copies and FusedRMSNormGate (about 30 µs in total) are replaced by one SM-count-grid two-level-ticket kernel at 20.1 µs. The expected directions agree with the weights: each full row has weight 16 against 46 for sliding. The load-level savings are about 1.0% (decode_1), 1.3% (decode_16) and 2.8% (decode_32), so the headline should rise by only about 0.8-0.9% (to about 1.296). That is inside the ±1.5% CI, so the hypothesis is rightly judged per row and per kernel. WEAKEST LINK: the decode_1 claim. In 5e44c7ae, full decode_1 went 142.7 -> 137.7, but churn did not class it as moved: that run's only gain was outside the gate, on sliding verify_32. The base here is 139.2 µs (w2-atomic already took Marlin T=1 from about 44 to 35.8 µs). The predicted 134 needs about 5 µs to reproduce on top of a different MoE tail, and the kill line of 137 sits only about 3 standard errors (err about 0.7 µs) below the current row. A secondary weak point is the per-state ticket and workspace move, which is new code, so the earlier 13/13 passes do not cover it. One more difference from the incumbent: master allocates ws_acc and ws_ml per call with torch.empty, while the restack preallocates them per state. Under a graph this should be neutral, but it changes the memory layout.

### Confounders on the current kernel
- w2-marlin-atomic-add-t1-port (current master, landed after both ports were measured): it shares the full decode_1 row with sub-change (b). The w2 path's mode (atomic / atomic_zero / master) is picked by a self-check in prepare(), and atomic_zero adds a cache3.zero_() memset. If the new sk setup in prepare (extra allocations, captures, refills) flips that choice, decode_1 moves by about 1-3 µs on both sliding and full, for reasons that have nothing to do with attention. Sliding decode_1 (123.9) is the control: if it moves, read the w2 line in the log before crediting (b). Otherwise the two are serial and roughly additive, because Marlin runs after attention in the graph.
- full-decode-splitkv-restack-on-pdl (landed; the incumbent kernels A/B that (a) replaces, and the pre-attention q route that both sub-changes read): the selector compares new against these very kernels, so the comparison has a clean incumbent. Any delta on decode_16/32 can only come from replacing A+B, or from the workspace moving out of per-call torch.empty.
- small-t-launch-minimal-dense-chain-pdl (landed; PDL dense chain at T<=32): the o GEMM and pooler/router kernels that follow attention may be launched with programmatic serialization. If the sk kernels never execute griddepcontrol.launch_dependents, the PDL overlap at the attention-to-o boundary is lost or changes compared with master, which ran plain FA3 or kernel B there. That can shift decode_1/16/32 by a few µs in either direction for reasons unrelated to the attention kernel itself. Read the attention row in top_kernels, not only the shape median.
- fa3-hdim128-head-pairing-batched and post-expert-norm-triton-small-t: these no longer touch the gated paths ((a) replaces paired FA3 territory that the splitkv path already took over; post-expert norm is downstream MoE). They do not confound.

### Interference and A/B needs
No companion is warranted. The two sub-changes are gated on batch==1 versus 1<batch<=64 and land on different scored rows (full decode_1 versus full decode_16/32), so each row isolates one kernel and a control would add nothing. Each kernel was already measured on its own, one master earlier. The only landed change since then (w2-atomic, T=1 MoE) is serial with attention, and sliding decode_1 detects any drift it causes. An ab-revert of w2-atomic would only re-measure what 5e44c7ae already measured. With 3 decode checks left today, one run is the right spend. If decode_1 comes in at 137-139 while decode_16/32 reproduce, the follow-up is not a companion but a narrower restack of (a) alone, plus a look at whether (b) loses its PDL edge into the o GEMM.

### Open questions
- Did 5e44c7ae's full decode_1 improvement (142.7 -> 137.7) fail churn's 'moved' test because of base drift? If so, how much of the predicted 5 µs is real, and is the 137 kill line too tight now that the base is 139.2?
- Do the sk kernels issue griddepcontrol.launch_dependents so that the PDL-launched kernels after attention keep their overlap, and was this the same in both port runs?
- Does the w2_mode line in the log stay 'atomic', as on master, after the new sk setup is added to prepare()?
- Are the two port branches' base_sha values (f6e653b, 08b1841) the same master tree? If not, each port's row delta was measured against a slightly different incumbent.

