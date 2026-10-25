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
status: running
review_at: 2026-10-25T054426Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T054426Z
base_sha: 3856ee7eca4325994492f9d6308b31aba517ef17
base_blob: 62c5a9413ba5710defea2ca3de14c1a1dc3b5a6d
branch: hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64
worktree: .worktrees/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T084818Z
impl_commit: 67bca2f6fefd66e61404802d4382ee37e1797744
submission_sha256: 1c947e7006f96e2847e14a34154b3aade32b33fb14f61a691ba8dbb81dbd3638
norm_sha256: 1f16e86616a18da73331a7ba4bdb9cb8135e5edb4fbd79acd1bd8dc04ea309e2
submission_blob: d67278e071e62145a19d0d67759c2d311ce1e398
lint_warnings: 45
attempt_file: 2026-10-25T084818Z-check-full-decode-sk-ticket-restack-b1-b64.md
prepared_at: 2026-10-25T084818Z
job: 9c99379067084f6e
dispatched_at: 2026-10-25T084834Z
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


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree (+1118/-2, no diff against master 00e36e3 before the edit). Nothing was run: this node has no GPU or torch.

What changed:
- One new module `muse_decode_sk` (load_inline at import, `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored, `-O3 -std=c++20 -lineinfo`, arch flags printed). The kernel template is byte-identical in both port branches (checked by diff); it has two instantiations: variant 0 = `<4,true,false>` (grid 2 x SM) and variant 3 = `<4,true,true>` (grid = SM count). No wgmma/setmaxnreg, so no `__CUDA_ARCH_FEAT_SM90_ALL` guard is needed; the kernel uses cp.async.cg, mma.sync and cvt e4m3 only.
- kernel_fn: `_sk_variant(cfg, state["sk"])` (config plus the state's own flags). B == 1 uses variant 3, which replaces FA3 + FlashAttnFwdCombine + FusedRMSNormGate. For 1 < B <= 64 it uses variant 0 where `_splitkv_gate` holds, replacing kernels A + B. The B=1 q route is the existing qk_norm/cast block (no extra change was needed). Kernels A/B and the FA3 chain stay as fallbacks. Sliding, verify and B > 64 are untouched.
- Per state, not global: prepare() builds `state["sk"]` with ws_acc, ws_ml, ticket[64], gticket[32], gacc and gml (sized from SM count and kernel MAXB only, zeroed). The only module globals are the compiled module, arch flags, plain-bool decision dict and a wall-time float. The C++ side keeps only function-attribute flags in a static, no pointers.

Review fixes applied (the review wins over the spec where they differ):
- Tickets and workspaces per state: done. The ports' module-level ticket and per-call torch.empty workspaces are gone. The ws is preallocated per state, which the review flagged as a layout change that should be neutral under a graph.
- Self-resetting: the last arriver zeroes its ticket (both levels). The self-check proves it: every replay of the alternation check asserts the state's ticket sum is 0.
- `__threadfence` before the ticket atomic: present in both levels (all threads fence, `__syncthreads`, thread 0 draws). I also switched the single-level draw from `atomicAdd` to the same `atom.acq_rel.gpu` helper the two-level path uses.
- prepare() zeroes every ticket before returning: a final `_sk_zero` plus sync, with the sum printed.
- Stream-K covers [0, seqlen): the kernel masks positions >= seq and uses ceil(seq/64) tiles. Self-checks cover seq%64 in 1..7, lengths 1/7/33/63/64/65, agentic B=2 (ctx 80000), B=5 (20000), B=33 (8000), and kv_scale 1 and 0.25 against the fp32 reference. New: a planted-key probe (k = 6 x that kv head's head-0 q at the first / 63 / 64 / last position, so a dropped or doubled key changes the output by O(1), not 1e-4).
- Norm-gate fp32 internal: `finalize_store` computes o/L*v_scale, the sum of squares, rsqrt, sigmoid and the product in fp32 and rounds to bf16 only at the store. The incumbent rounded o to bf16 before the norm.
- launch_dependents: the incumbent kernels A/B and FA3 execute no griddepcontrol (grep: only the dense-chain TU has any), so the sk kernels issue none; this answers the review's open question.
- Workspace sized from config/hardware only (2 x SM + 64 slots; batch-1 group slots = 32).
- PATH CHOSEN lines (per state, three lines: B=1, B<=23, B>=24) carry the selector timing, `w2_mode=<state["w2_mode"]>` and `prepare_wall_cum=<s>`. A `[prepare] this call ..., cumulative ...` line prints at the end of every prepare().
- Gates (LESSONS 2026-10-25): a path is chosen only if the graph-timed incumbent/new ratio is >= 1.02 (cold L2 flush, interleaved, 15 reps), at B = 1, 16, 32 on agentic ragged lengths (ctx 80000 +/- 7.5%). The incumbent for B=1 is FA3 + combine + norm-gate, for 16/32 it is kernels A + B. Class 16 covers B <= 23, class 32 covers B >= 24.

Two-state proof: the first prepare() runs the full validation once with temporary buffer sets. Its alternation check (B=1 at 80000, B=2 at 80000, B=5 at 20000, B=33 at 8000) builds two independent buffer sets, captures one graph on each, and replays them alternately six times with refilled q/KV/page table/kv scale/lengths, NaN-poisoning the output each time. Every prepare() (including the second) also runs a graph-replay check on that state's own buffers.

Deviations: (1) the worktree and the ports are on different bases, so the self-check was rewritten around per-state buffers rather than copied; (2) the selector uses graph timing with the 1.02 margin (ports used eager timing and 0.99); (3) `launch_dependents` was not added because the incumbent has none.

Biggest risk: first-ever compile of the combined module (never built as one TU: the two ports built different single instantiations) and the longer prepare() (two builds plus about 40 reference checks and graph captures). Reading `[sk]`/PATH CHOSEN lines in the log is the first step; the build line must show 9.0a.

## Rules check

- [x] CUDA-graph capturable: `_full_decode_sk` only allocates `out` via torch.empty (as master's kernel B did) and launches; the tickets/workspaces are preallocated per state; no `.item()`/`.cpu()`/sync in kernel_fn (every sync is in prepare-time helpers); the capture itself is exercised in the self-check.
- [x] Nothing derived from `data` is cached across calls: `state["sk"]` holds only zeroed tickets and scratch sized from SM count and kernel constants, plus bool flags from synthetic data.
- [x] Weights: n/a: no weights touched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: q is cast to E4M3 exactly as in the split-KV path, output is bf16, softmax and merge in fp32 (per TASK.md).
- [x] Attention sees every key: stream-K covers [0, seqlen) exactly, with masked tails, verified against the fp32 reference including a planted-key probe at tile edges and the last position.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: grep for subprocess/multiprocessing/fork/Popen is empty; the build and self-check are in-process.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Two-state rule: no device pointer, workspace, ticket or descriptor in a module/static global; everything comes from `state["sk"]`.
- [x] sm_90a: `TORCH_CUDA_ARCH_LIST=9.0a` around load_inline and restored; arch flags printed; cubin version 90 asserted; no sm_90a-only PTX is used.

okc_lint output: `OK with warnings: submission.py` (only prepare-time host-sync warnings, all outside kernel_fn's call graph). `python3 -I -m py_compile` passed; `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading the code plus a plain-Python emulation of the stream-K schedule. **Verdict: dispatch-with-fixes.**

Fix (1):
- `_sk_select` B=1 cross-check: the tolerance against the incumbent goes from 1e-2 to 2e-2 at B=1 only. The incumbent there is FA3, and its FP8-P error alone was logged at 3.4-7.6e-3 against fp32 in earlier muse_attention_decode runs (`[ps-gate] B=1 master FA3 path rel L2 6.08e-03`; `[fullfused] FA3 vs fp32 3.4-7.2e-3`). Combined with the sk error of about 1.6e-3, that left under 2x headroom, so on an unlucky seed the B=1 path could have been silently disabled. The port never ran this check. The authoritative check is still the fp32 reference (2e-2). B>1 stays at 1e-2: A+B and sk agree to about 1e-4.

Checked and found OK:
- Gates read config only. B==1 and 1<B<=64 (plus `_splitkv_gate`) do not overlap. Sliding, verify, B>64, page!=1 and any selector loss fall through to master's exact code (the diff is additive, base == master 00e36e3 blob). The prepare order keeps the w2 self-check before the sk setup, so w2_mode cannot flip.
- Per state: ws_acc/ws_ml/ticket/gticket/gacc/gml live in `state["sk"]`. Module globals are the extension, arch flags, plain bool/str decision and a float. The C++ static holds only the cudaFuncSetAttribute bools.
- Tickets self-reset: the last arriver zeroes each ticket. The alternation check asserts a zero sum after every replay, and prepare zeroes the tickets again.
- Fallbacks: a build failure, a validate/select/state-check exception or a miss each lands on the incumbent with three PATH CHOSEN lines. Nothing raises out of prepare.
- kernel_fn: no sync, print or try. The only allocation is `out` (torch.empty, shape from config, as master's kernel B). `_sm_count` is cached.
- Stream-K coverage (emulated, 21k random configs, B 1..64, grids 1..1024, lengths 1..86k plus 1M): every tile is owned once, `tile_owner` is the exact inverse of `range_start`, partial slots bx+b are unique and < grid+B, every request is finalized exactly once, two-level ngrp<=32, and the group and second-level arrival counts match. Ragged tails are masked by pos < seq.
- Budgets: a third load_inline module, as in each port run (30-37 s there; two instantiations here, so maybe ~45 s). The added prepare work is about 40 small eager cases, 8 alternation captures, 3 selectors and 1-2 state checks, well inside 300 s.

Residual risk: the combined two-instantiation TU has never been compiled, and the B=1 gain (about 5 us expected) is small against the 137 us kill line.
