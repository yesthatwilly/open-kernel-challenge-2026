---
problem: muse_attention_decode
slug: full-decode1-sk4-cluster-dsmem-merge
created: 2026-10-25T042334Z
base: master
area: full-layer attention (batch-1 decode, stream-K kernel merge)
change: >
  Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01).
  Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) vs a 4.3 us floor; in stage 3
  prepare grid 2xSM was slower than grid SM (24.7 vs 23.1), so more CTAs do not help and the tail
  is the merge chain: each of 132 CTAs writes a 4.2 KB fp32 partial, __threadfence, group atomic,
  the group last arrival reads ~11 partials, fence, second atomic, last group merges ~12 group
  slots + norm-gate. Change, at batch 1 only (config): launch the kernel with a thread-block
  cluster of C CTAs (cudaLaunchKernelEx, cluster dims C, grid a multiple of C: C=4 -> grid 132,
  C=2 -> 132, C=8 -> 128) and replace the FIRST ticket level with an on-chip merge: each CTA
  writes (m[16], l[16], acc[16x64]) to its own smem, barrier.cluster.arrive.release +
  wait.acquire, rank 0 reads the C-1 peer partials through mapa + ld.shared::cluster and
  LSE-merges, then writes ONE cluster partial to global, __threadfence, atomicAdd on ticket[0];
  the last cluster merges the grid/C cluster partials, applies v_scale, the landed norm-gate
  epilogue, stores bf16 and resets the ticket (graph replay). No group tickets are used at B=1.
  Peers stay resident until rank 0 has read them (second barrier.cluster before exit). The
  stream-K tile split, ring and math are the landed code. Prepare: build C in {2,4,8}, time each
  against the landed two-level kernel (cold L2, median of 10), print all, choose the fastest and
  assert it beats the landed kernel by >= 3%, else keep the landed two-level path (PATH CHOSEN:
  full decode_1 cluster C=<c> or two-level (fallback: <timings>)). Self-check: the stage-3 B=1
  cases (80000/4097/63/1 keys, kv 1.0 and 0.25) for each C, two replays to prove the ticket
  resets. B > 1 unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process
  canary, never a child process). First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: removes one global fence/atomic/read level and cuts global partial traffic by C; kernel 18.6 -> ~15.5 us, shape 22.6 -> ~19.5. M1 tests pass; M2 kernel <= 16.5 us; M3 <= 14 us
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32*"
    direction: flat
    why: batch > 1 keeps the landed single-level kernel
  - shapes: "*layer=sliding* *load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)"
review: required
status: queued
---

# full-decode1-sk4-cluster-dsmem-merge

## Change

Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01). Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) vs a 4.3 us floor; in stage 3 prepare grid 2xSM was slower than grid SM (24.7 vs 23.1), so more CTAs do not help and the tail is the merge chain: each of 132 CTAs writes a 4.2 KB fp32 partial, __threadfence, group atomic, the group last arrival reads ~11 partials, fence, second atomic, last group merges ~12 group slots + norm-gate. Change, at batch 1 only (config): launch the kernel with a thread-block cluster of C CTAs (cudaLaunchKernelEx, cluster dims C, grid a multiple of C: C=4 -> grid 132, C=2 -> 132, C=8 -> 128) and replace the FIRST ticket level with an on-chip merge: each CTA writes (m[16], l[16], acc[16x64]) to its own smem, barrier.cluster.arrive.release + wait.acquire, rank 0 reads the C-1 peer partials through mapa + ld.shared::cluster and LSE-merges, then writes ONE cluster partial to global, __threadfence, atomicAdd on ticket[0]; the last cluster merges the grid/C cluster partials, applies v_scale, the landed norm-gate epilogue, stores bf16 and resets the ticket (graph replay). No group tickets are used at B=1. Peers stay resident until rank 0 has read them (second barrier.cluster before exit). The stream-K tile split, ring and math are the landed code. Prepare: build C in {2,4,8}, time each against the landed two-level kernel (cold L2, median of 10), print all, choose the fastest and assert it beats the landed kernel by >= 3%, else keep the landed two-level path (PATH CHOSEN: full decode_1 cluster C=<c> or two-level (fallback: <timings>)). Self-check: the stage-3 B=1 cases (80000/4097/63/1 keys, kv 1.0 and 0.25) for each C, two replays to prove the ticket resets. B > 1 unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: removes one global fence/atomic/read level and cuts global partial traffic by C; kernel 18.6 -> ~15.5 us, shape 22.6 -> ~19.5. M1 tests pass; M2 kernel <= 16.5 us; M3 <= 14 us
- `*layer=full load=decode_16* *layer=full load=decode_32*` flat: batch > 1 keeps the landed single-level kernel
- `*layer=sliding* *load=verify_*` flat: outside the gate

## Risk

Kill: if no C beats the landed two-level kernel in prepare by 3%, the merge chain is not the B=1 limiter; next stage would be per-phase clock64 stamps. Risk: cluster co-scheduling at 95 KB smem and 1 CTA/SM; peer smem lifetime.

## Milestones

Rewrite stage 4 of `full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
