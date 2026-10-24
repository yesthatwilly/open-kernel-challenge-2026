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
review: required
status: queued
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
