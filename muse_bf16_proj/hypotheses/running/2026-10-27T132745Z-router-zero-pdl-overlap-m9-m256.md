---
problem: muse_bf16_proj
slug: router-zero-pdl-overlap-m9-m256
created: 2026-10-27T132745Z
base: master
area: router GEMM (m=9..256, zero-fill node overlapped with the router kernel by PDL)
change: >
  Built on master (8f233e88 lineage). At m > 8 the router is two graph nodes: torch.zeros((m,
  256), fp32) (profile: at::native vectorized_elementwise fill, 0.7 us at m=16/32, 0.8 at m=128,
  0.9 at m=256 in a6424247) followed by the Triton router kernel (_router_swapab_kernel at 9 <= m
  <= 32 via _router_forward, _router_hilo_kernel at m > 32 via _router_hilo_forward) whose fp32
  K-partials are tl.atomic_add'ed into the zeroed output. The router kernel's whole K-loop
  (hi/lo/x loads + both tl.dot) does not depend on the zeroed output; only the final atomic_add
  does. CHANGE: (1) new Triton kernel _router_zero_pdl_kernel(out_ptr, numel, BLOCK: tl.constexpr)
  whose FIRST statement is tl.extra.cuda.gdc_launch_dependents(), then a masked tl.store of zeros
  over its BLOCK (grid cdiv(numel, 4096), BLOCK 4096, 4 warps); out becomes torch.empty((m, 256),
  fp32). (2) Both router kernels get a constexpr PDL: tl.constexpr; when PDL is True,
  tl.extra.cuda.gdc_wait() is placed after the K-loop and immediately before tl.atomic_add
  (nothing else moves); the router launch passes launch_pdl=True. PDL=False compiles today's
  kernel byte-identically. So the router CTAs stream their weights and do all MMAs while the zero
  kernel runs, and gdc_wait (primary grid complete + its stores visible) orders the zeros before
  every atomic. Flags router_pdl_ok_<m> for m in (16, 32, 128, 256) in state, read by
  _router_forward/_router_hilo_forward through kernel_fn (cfg/shape/flag-only branching; m=9..15
  maps to the 16 flag, 17..31 to 32, 33..128 to 128, >128 to 256). The m <= 8 cluster path
  (router_kernel<8,8>, no memset) is untouched. Prepare, state A, new section after
  _router_prepare: capability probe (hasattr(tl.extra.cuda, 'gdc_wait') and
  'gdc_launch_dependents'; the launch accepts launch_pdl; compiled PTX of both kernels contains
  griddepcontrol.wait / griddepcontrol.launch_dependents) else print why and keep master; per m:
  _cell_check-style correctness vs an fp64 reference (rel L2 <= AB_ROUTER_REL_L2 5e-5, the
  power-of-two probe row) AND a race check: capture kernel_fn once, replay 32 times with refilled
  x, every replay vs master within 5e-5 (a zero landing after an atomic doubles or drops a
  partial: errors of order 1, never 1e-5); then two read-flush _cell_gate rounds
  (flush_mode='read', the khalf rule: master/cand >= 1.03 in both rounds and cand IQR <= 1.5x
  max(master IQR, 0.5 us)); then _cell_kernels asserting exactly [_router_zero_pdl_kernel, the
  router kernel] and no vectorized_elementwise. State B inherits the flags; AB_CASES gains router
  m=16/32/128/256 mapped to their flags. One PATH CHOSEN R-PDL line per m with both rounds' net
  us, re-printed at the end of prepare (16 KB tail).
expected:
  - shapes: layer=router load=decode_16 m=16
    direction: faster
    why: "the 0.7 us zero-fill node runs under the router K-loop instead of before it: row 5.48 -> ~5.0 us (-6..-9%, above the 1.7-3.6% drift)"
  - shapes: layer=router load=decode_32 m=32
    direction: faster
    why: "same: 5.95 -> ~5.4 us"
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "0.8 us fill hidden under the hilo K-loop: 7.95 -> ~7.3 us"
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "0.9 us fill hidden: 9.51 -> ~8.8 us"
  - shapes: layer=router load=*_1 m=*
    direction: flat
    why: m <= 8 keeps the cluster kernel (no memset)
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "router-splitk-semaphore-no-memset: that deleted the memset with a ticket-semaphore last-arriver reduce, a serialized epilogue that DISTILL 3.1 measured x1.07-1.10 slower than the memset under graph replay; this keeps the memset and the relaxed atomics exactly and only lets the router kernel's weight stream and MMAs run while the zero kernel executes (PDL), with no new reduce or serialization"
review: proceed
status: running
review_at: 2026-10-27T141019Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T141019Z
base_sha: 93f6f7fa2f41100d0f75cf209e2ff5e6c65f97e7
base_blob: d51c745a0e43d20ebf2cbb3d582f3f8ead72f039
branch: hyp/muse_bf16_proj/router-zero-pdl-overlap-m9-m256
worktree: .worktrees/muse_bf16_proj/router-zero-pdl-overlap-m9-m256
mode: null
restack_of: null
stale_base: null
---

# router-zero-pdl-overlap-m9-m256

## Change

Built on master (8f233e88 lineage). At m > 8 the router is two graph nodes: torch.zeros((m, 256), fp32) (profile: at::native vectorized_elementwise fill, 0.7 us at m=16/32, 0.8 at m=128, 0.9 at m=256 in a6424247) followed by the Triton router kernel (_router_swapab_kernel at 9 <= m <= 32 via _router_forward, _router_hilo_kernel at m > 32 via _router_hilo_forward) whose fp32 K-partials are tl.atomic_add'ed into the zeroed output. The router kernel's whole K-loop (hi/lo/x loads + both tl.dot) does not depend on the zeroed output; only the final atomic_add does. CHANGE: (1) new Triton kernel _router_zero_pdl_kernel(out_ptr, numel, BLOCK: tl.constexpr) whose FIRST statement is tl.extra.cuda.gdc_launch_dependents(), then a masked tl.store of zeros over its BLOCK (grid cdiv(numel, 4096), BLOCK 4096, 4 warps); out becomes torch.empty((m, 256), fp32). (2) Both router kernels get a constexpr PDL: tl.constexpr; when PDL is True, tl.extra.cuda.gdc_wait() is placed after the K-loop and immediately before tl.atomic_add (nothing else moves); the router launch passes launch_pdl=True. PDL=False compiles today's kernel byte-identically. So the router CTAs stream their weights and do all MMAs while the zero kernel runs, and gdc_wait (primary grid complete + its stores visible) orders the zeros before every atomic. Flags router_pdl_ok_<m> for m in (16, 32, 128, 256) in state, read by _router_forward/_router_hilo_forward through kernel_fn (cfg/shape/flag-only branching; m=9..15 maps to the 16 flag, 17..31 to 32, 33..128 to 128, >128 to 256). The m <= 8 cluster path (router_kernel<8,8>, no memset) is untouched. Prepare, state A, new section after _router_prepare: capability probe (hasattr(tl.extra.cuda, 'gdc_wait') and 'gdc_launch_dependents'; the launch accepts launch_pdl; compiled PTX of both kernels contains griddepcontrol.wait / griddepcontrol.launch_dependents) else print why and keep master; per m: _cell_check-style correctness vs an fp64 reference (rel L2 <= AB_ROUTER_REL_L2 5e-5, the power-of-two probe row) AND a race check: capture kernel_fn once, replay 32 times with refilled x, every replay vs master within 5e-5 (a zero landing after an atomic doubles or drops a partial: errors of order 1, never 1e-5); then two read-flush _cell_gate rounds (flush_mode='read', the khalf rule: master/cand >= 1.03 in both rounds and cand IQR <= 1.5x max(master IQR, 0.5 us)); then _cell_kernels asserting exactly [_router_zero_pdl_kernel, the router kernel] and no vectorized_elementwise. State B inherits the flags; AB_CASES gains router m=16/32/128/256 mapped to their flags. One PATH CHOSEN R-PDL line per m with both rounds' net us, re-printed at the end of prepare (16 KB tail).

## Expected

- `layer=router load=decode_16 m=16` faster: the 0.7 us zero-fill node runs under the router K-loop instead of before it: row 5.48 -> ~5.0 us (-6..-9%, above the 1.7-3.6% drift)
- `layer=router load=decode_32 m=32` faster: same: 5.95 -> ~5.4 us
- `layer=router load=verify_16 m=128` faster: 0.8 us fill hidden under the hilo K-loop: 7.95 -> ~7.3 us
- `layer=router load=verify_32 m=256` faster: 0.9 us fill hidden: 9.51 -> ~8.8 us
- `layer=router load=*_1 m=*` flat: m <= 8 keeps the cluster kernel (no memset)
- `layer=qkv_gate *` flat: untouched

## Risk

Headline ~+0.3-0.4% (four router rows x ~0.5 us; 0.18-0.23 %/us on the verify loads, ~0.6% of each decode_16/32 load): sub-margin alone, so judge it on the four router rows and the R-PDL gate lines, then stack. Risk 1 (VOID): the judge's Triton lacks tl.extra.cuda.gdc_wait / launch_pdl (added in Triton 3.4-era; torch 2.14.1+cu130 should ship newer). Before dispatch, grep a recent judge log or the sglang env for the Triton version; the probe prints it and keeps master otherwise. Risk 2: a programmatic edge must survive CUDA-graph capture (CUDA 12.3+ captures PDL edges; the judge is cu130); the race check catches a lost edge as large errors. Risk 3: gdc_wait must sit after the loop: if Triton hoists or sinks it, the PTX order check (griddepcontrol.wait after the last wgmma, before the red/atom) is the guard. Decode-portable: decode's T=128 'hilo router' cell (W6-B) is the same torch.zeros + _router_hilo_kernel pair. Do not combine with router-hilo-m128-m256-readflush-config-sweep in one check (two routable novelties on the same rows, DISTILL 3.2.7).

## Adversarial review

Reviewed 2026-10-27T141019Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible and a different lever from every earlier router attempt: it overlaps the graph's memset node with the router kernel (earlier attempts tuned the kernel or replaced the memset with a ticket epilogue). Deleting the memset at m<=8 saved 0.40-0.54 us, and PDL paid between our own kernels in decode. The in-prepare interleaved read-flush gate gives master vs candidate on identical surroundings; no companion. Tighten two guards: the PTX check must count exactly one griddepcontrol.wait, and the race check needs a slow-primary stand-in.

### Justification
Router rows at m=16/32/128/256 (5.48/5.95/7.95/9.51 us vs floors 1.8-2.7 us) are a fill node (0.7-0.9 us) plus the Triton split-K kernel with atomics. The m<=8 memset deletion (~0.4-0.5 us) is the realistic ceiling; m=128/256 predictions are ~0.2 us optimistic; a realistic 5-6% still clears drift and the 1.03 gate. Headline ~+0.3%, unreadable: judge on the four router rows and the R-PDL gate lines, not profile top_kernels (PDL-inflated rows). Prior router-splitk-semaphore-no-memset serialized the reduce; this keeps the relaxed atomics and the memset. Weakest link: whether Triton 3.8's launch_pdl under graph capture produces a real programmatic edge, and whether Triton auto-inserts a griddepcontrol.wait at entry; either gives a misleading flat result. Assert exactly one griddepcontrol.wait per router kernel (after the last mma, before the red/atom) and exactly one launch_dependents in the zero kernel at entry.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32: sets the K-loop available to hide the fill; a precondition shared by both arms.
- router-triton-hilo-fused-control-splitk-all-m: the split-K atomic + zeros design; a precondition.
- restack-three-cell-wins (router_kernel<8,8>, no memset at m<=8): m<=8 rows stay flat.
- Queued router-hilo-m128-m256-readflush-config-sweep: same rows; keep separate checks; whichever lands second re-gates.

### Interference and A/B needs
No companion. Add a third gate arm (Triton zero kernel + router PDL=False, no launch_pdl) and print its net us beside cand/master, to separate the fill change from PDL.

### Open questions
- Does Triton 3.8 insert griddepcontrol.wait at entry with launch_pdl=True? Count waits and check position.
- Race check: add a stand-in primary that spins ~10 us before storing zeros; a debug variant without the wait must fail.
- Does graph capture keep the programmatic edge on the judge (cu130)? Print the edge type if cheap.
- At m=256: 16 zero CTAs + 128 router CTAs at ~144 KB smem; confirm no placement delay.
- Check both rounds' IQRs before trusting a marginal 1.03 pass.

