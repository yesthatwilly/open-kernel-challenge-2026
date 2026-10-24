---
problem: muse_bf16_proj
slug: router-splitk-semaphore-no-memset
created: 2026-10-24T075041Z
base: master
area: router GEMM
change: >
  Built on master (dba4ffd). For the router at m <= 32 only (buckets (16,32,16) and (32,32,16):
  grid (1, 8, 16) = 128 CTAs, unchanged tiles, BLOCK_K=128, num_stages=3, num_warps=4), remove the
  torch.zeros((m, 256), fp32) fill kernel (at::native::vectorized_elementwise_kernel, 0.7-0.8 us
  plus its own graph node) and the relaxed atomic_add epilogue, and replace them with a serial
  split-K reduce inside the same launch: out = torch.empty((m, 256), fp32); each CTA stores its
  fp32 [BLOCK_M, 32] K-partial to a persistent scratch state['router_ws'] = torch.empty((16, 32,
  256), fp32) allocated in prepare() (fully overwritten each call, nothing input-derived kept),
  then tl.debug_barrier(), then one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1,
  sem='acq_rel', scope='gpu') on state['router_cnt'] = torch.zeros(8, int32) from prepare(); the
  CTA drawing ticket == 15 reloads the 16 partials (cache_modifier='.cg'), sums them in fixed
  order 0..15 in fp32, masked-stores to out, and resets the counter with tl.atomic_xchg(cnt_ptr +
  pid_n, 0). Make this a separate @triton.jit kernel (_router_hilo_sem_kernel) sharing the K loop
  so the m > 32 buckets keep master's kernel, zeros and atomics byte-for-byte. Re-zero router_cnt
  after the prepare() warm-up. (The simpler 'split 0 stores, the others atomic_add' is not used:
  nothing orders split 0's plain store before another split's add into torch.empty memory, so a
  late store can overwrite earlier adds.)
expected:
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: 5.8 us row = kernel 3.3 + memset 0.8 + node gap; dropping the memset node should save ~1 us; the last-CTA reduce reads only 16 x 128 B at m=1
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: m=8, 5.7 us, same; reduce tail 16 x 1 KB
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: 6.2 us row, same mechanism
  - shapes: layer=router load=decode_32 *
    direction: flat
    why: 6.5 us row; the last CTA of each tile reads 16 x 4 KB = 64 KB serially, which may eat most of the ~1 us saved
  - shapes: layer=router load=verify_16 *
    direction: flat
    why: m > 32 keeps master kernel, memset and atomics; control
  - shapes: layer=router load=verify_32 *
    direction: flat
    why: same; control
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T111911Z
review_companions: null
hold_reason: null
stage: claimed
claimed_at: 2026-10-24T120042Z
base_sha: caf555ec8435254c599ff4cf1e7741a96a88e5e1
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/router-splitk-semaphore-no-memset
worktree: .worktrees/muse_bf16_proj/router-splitk-semaphore-no-memset
mode: null
---

# router-splitk-semaphore-no-memset

## Change

Built on master (dba4ffd). For the router at m <= 32 only (buckets (16,32,16) and (32,32,16): grid (1, 8, 16) = 128 CTAs, unchanged tiles, BLOCK_K=128, num_stages=3, num_warps=4), remove the torch.zeros((m, 256), fp32) fill kernel (at::native::vectorized_elementwise_kernel, 0.7-0.8 us plus its own graph node) and the relaxed atomic_add epilogue, and replace them with a serial split-K reduce inside the same launch: out = torch.empty((m, 256), fp32); each CTA stores its fp32 [BLOCK_M, 32] K-partial to a persistent scratch state['router_ws'] = torch.empty((16, 32, 256), fp32) allocated in prepare() (fully overwritten each call, nothing input-derived kept), then tl.debug_barrier(), then one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['router_cnt'] = torch.zeros(8, int32) from prepare(); the CTA drawing ticket == 15 reloads the 16 partials (cache_modifier='.cg'), sums them in fixed order 0..15 in fp32, masked-stores to out, and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Make this a separate @triton.jit kernel (_router_hilo_sem_kernel) sharing the K loop so the m > 32 buckets keep master's kernel, zeros and atomics byte-for-byte. Re-zero router_cnt after the prepare() warm-up. (The simpler 'split 0 stores, the others atomic_add' is not used: nothing orders split 0's plain store before another split's add into torch.empty memory, so a late store can overwrite earlier adds.)

## Expected

- `layer=router load=decode_1 *` faster: 5.8 us row = kernel 3.3 + memset 0.8 + node gap; dropping the memset node should save ~1 us; the last-CTA reduce reads only 16 x 128 B at m=1
- `layer=router load=verify_1 *` faster: m=8, 5.7 us, same; reduce tail 16 x 1 KB
- `layer=router load=decode_16 *` faster: 6.2 us row, same mechanism
- `layer=router load=decode_32 *` flat: 6.5 us row; the last CTA of each tile reads 16 x 4 KB = 64 KB serially, which may eat most of the ~1 us saved
- `layer=router load=verify_16 *` flat: m > 32 keeps master kernel, memset and atomics; control
- `layer=router load=verify_32 *` flat: same; control

## Risk

Answers whether the memset node is a real cost on the router's small-m rows (master's profile: memset 0.7-0.8 us of a 4.1 us kernel sum, and a ~1.7 us reported-minus-kernel gap). The fp32 output also makes this deterministic (fixed-order sum instead of atomic order), a free side benefit for the router's 0.01% bound. Risks: (1) memory-ordering correctness of the barrier + scalar acq_rel ticket pattern in Triton; a bug shows as wrong outputs on the m=8 and m=9 router tests, not silently; (2) a counter left nonzero after warm-up corrupts every call, hence the re-zero; (3) the serial tail: the reducing CTA's 16 dependent-latency loads could cost ~0.5 us, more at m=32, which is why decode_32 is called flat. Independent of router-splitk32-small-m-all-loads-in-flight (that one changes grid/stages, this one the epilogue); if both gain, combine on SPLIT_K=32 only after re-checking the doubled reduce tail.

## Adversarial review

Reviewed 2026-10-24T111911Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, rules-clean, readable from top_kernels (the memset row vanishes and _router_hilo_sem_kernel's own time shows the tail cost), but it sits on master's small-m router config that the two running splitk32 runs may replace. Defer dispatch until router-splitk32-small-m-all-loads-in-flight and its stages3 control harvest, then build it on whichever small-m config wins, with SPLIT_K as a constexpr instead of a hardcoded 16.

### Justification
The mechanism holds up. FINDINGS shows the router m<=32 rows as _router_hilo_kernel 3.3-4.1 us plus at::native::vectorized_elementwise_kernel (the torch.zeros fill) 0.7-0.8 us in its own graph node, with 5.7-6.5 us reported. LESSONS puts a single Triton graph node at ~0.3 us of reported-minus-kernel, so removing the memset node can save at most ~0.8 + 0.3 = ~1.1 us per row, about 15-19% of the router row at m=1/8/16. The floor is 1.8-1.9 us, so there is room. The headline effect is small: four of 30 shapes at -10..-18% is about +1.2-2% on the geomean, near LESSONS' ~1.5% single-run limit. Judge it per shape; Triton rows reproduce within ~1%. The expected directions agree with TASK: m=1/8/16/32 take the changed buckets, and m=128/256 (verify_16/32) are untouched in-run controls. WEAKEST LINK: the hypothesis prices the new epilogue only by bytes. It ignores the serial latency added to the last-arriving CTA of each N-tile, which is the critical path. Master's relaxed fp32 atomic_add is fire-and-forget reds. The new tail is a dependent chain: partial store, then bar.sync, then a gpu-scope acq_rel atomic round trip to L2 (~0.5-1 us), then 16 .cg L2 loads (one more round trip if they are issued together, 16 if they are emitted as a dependent loop), then the final store and the xchg. That can be ~1-1.5 us, roughly the size of the saving, so 'faster' at m=1/8/16 is a toss-up, not a given, and decode_32's 'flat' could just as well be 'slower'. Implementation must-dos: load all 16 partials before the first add (tl.static_range so the loads are hoisted, or one [SPLIT_K, BLOCK_M, BLOCK_N] tile load), then add in fixed order 0..SPLIT_K-1. Make the partial store and reload use the same masked [BLOCK_M, BLOCK_N] footprint, so rows >= m in the workspace are never read unmasked. The ticket/barrier/xchg pattern itself is proven by the pooler and qkv-gate semaphore checks (LESSONS, DECISIONS 2026-10-24T090934Z).

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m (landed, master's router): this hypothesis edits that kernel's epilogue directly. The tail cost scales with its SPLIT_K (16 partials) and its grid (8 N-tiles x 16 = 128 CTAs, one wave), so a gain or loss is specific to master's (16|32, 32, 16) small-m buckets. That is intended and attributable as long as the base stays the same.
- router-splitk32-small-m-all-loads-in-flight and its stages3 control (running, not landed, but they would land on exactly these buckets): SPLIT_K=32 doubles the number of partials the winner reloads (32 x 4 KB = 128 KB at m=32) and grows the workspace to (32, 32, 256). It also doubles the atomics this change removes, and at 256 CTAs it can change which CTA finishes last. The semaphore's value therefore differs between SPLIT_K=16 and 32 in both directions: more is saved (no doubled atomics, same memset) and the tail is longer. A SPLIT_K=16 result does not carry over to a SPLIT_K=32 master.
- sglang-gemm-dispatch-parity and router-hilo-at-m8: no confound. The router path they changed is gone from master (replaced wholesale by the split-K Triton kernel), and the bf16 layers' dispatch does not touch the router's launch sequence or memory.

### Interference and A/B needs
No companion is needed for the run itself. The profile already separates the two sub-effects: the memset kernel either disappears from top_kernels or it does not, and the new kernel's own time versus master's 3.3-4.1 us gives the tail cost directly. A gain or a loss is interpretable without an A/B, and the verify_16/32 rows are in-run controls. The real interference is sequencing against the splitk32 pair already in the judge. Its harvest is imminent and it changes the exact buckets this kernel replaces. If this is dispatched now at SPLIT_K=16 and splitk32 then gains, the result is stale. It would measure an epilogue on a config master has abandoned, it cannot be combined without a re-run (the hypothesis says so itself), and rebuilding it on the new master would silently revert the splitk32 gain. Recommendation: wait for that harvest, which costs one tick and no quota. If splitk32 (num_stages=2) or its stages3 control is a gain, write this on the winning config: SPLIT_K=32, the winning num_stages, workspace (32, 32, 256), winner on ticket == SPLIT_K-1. In that case decode_32 should be expected flat-to-slower because of the 128 KB serial reload. If both are neutral or regress, run it as written at SPLIT_K=16. Either way, write SPLIT_K as a constexpr (ticket == SPLIT_K-1, workspace sized from _router_config) so the choice is one line.

### Open questions
- Which small-m config wins in the splitk32 pair (num_stages=2 vs 3, or neither)? That decides SPLIT_K=16 or 32 for this build.
- Does Triton emit the scalar acq_rel ticket by one thread with a broadcast (an extra smem barrier on the winner path)? Check the TTGIR/PTX in the implement step; it adds to the tail.
- If the tail eats the saving, the cheaper follow-up is to keep the relaxed atomics and remove only the memset by having split 0 zero-and-add under a ticket. That still needs ordering, so it is probably no better. Worth a line in FINDINGS whichever way this lands.

