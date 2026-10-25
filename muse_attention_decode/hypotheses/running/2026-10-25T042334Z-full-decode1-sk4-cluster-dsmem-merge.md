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
review: proceed
status: running
review_at: 2026-10-25T043142Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T043142Z
base_sha: aafeb8f4d969d4a791dafeca1ad8dda986c3e7c1
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge
worktree: .worktrees/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T043142Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound stage 4 for the full decode_1 rewrite: replacing the first global ticket level with a DSMEM cluster merge is physically cheaper and has no rules risk. As written, though, it only times grid ~SM, and at C=2 and C=4 its single global level is longer than the landed second level. Add grid-2xSM cluster variants and a two-level-over-clusters option so the prepare can find the real win.

### Justification
Mechanism: the stage-3 diagnostic (REWRITE_READ 03:41Z) prices the merge structure directly. Single-level at grid SM ran 34.8 us and two-level 23.1 us, so serial slot reads cost ~0.1 us per 4.2 KB slot (11.7 us over ~109 fewer serial reads). The landed chain is ~12 group-slot reads plus ~11 final reads plus two fence/atomic rounds. The proposal is C-1 DSMEM peer reads plus grid/C global reads plus one fence/atomic. That gives C=8 -> 16 global reads (a clear cut), C=4 -> 33 (about equal to today's 23 plus one fence) and C=2 -> 66 (worse). So only C=8 shortens the chain, and the expected 18.6 -> 15.5 us (-3 us) is more than the whole merge chain is worth at 0.1 us/slot (~2.3 us plus two fences). WEAKEST LINK: the claim that the merge chain is the B=1 limiter. 20.5 MB in 18.6 us at grid SM means 1 CTA/SM with a 4-stage 16 KB ring, half the bytes in flight of the 2-CTA/SM B=16 config that reaches 3 TB/s. Bytes in flight at 1 CTA/SM explains most of the gap to the 4.3 us floor. grid 2xSM lost in stage 3 (24.7 vs 23.1) because it doubled the partials the global merge chain reads, and a cluster merge removes exactly that cost. The design's real upside is therefore cluster merge plus grid 2xSM (264 CTAs, C=8 -> 33 cluster partials), which the hypothesis does not time. Co-scheduling: tickets are last-arrival and never spin, so non-resident clusters cannot deadlock. barrier.cluster is intra-cluster and co-scheduled by hardware, so the design is safe at any grid. Milestones are readable from the profile row (sk kernel time on full decode_1). M3 (<=1.5x of 4.3 us) is not reachable this stage and needs the bytes-in-flight fix.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (c3c1b01): this is the incumbent being replaced. B=1 grid and variant selection live in _full_decode_sk (variant 3, grid SM), and the change must leave B>1 (variant 0) byte-identical. Read the result only on the full decode_1 sk-kernel row.
- full-decode-splitkv-s2-streamk-ticket-inproc (landed inside c3c1b01): it shares splitkv_attn_sk_kernel and its smem layout (95264 B, 2 CTAs/SM). A cluster launch attribute on the B=1 instantiation must not change B>1 regs/smem, so print attrs for every variant and compare with the landed log (ticket 119-121 regs).
- In flight, not landed: full-decode-splitkv-s3-fused-preattn edits the same TU for B>1, and full-decode1-sk-pdl-from-preattn edits the same B=1 launch. Neither confounds a run against master, but sk4 + PDL must later share one cudaLaunchKernelEx (cluster dim + programmatic serialization attributes are compatible).

### Interference and A/B needs
No companion. This is a rewrite stage, its prepare already times every C against the landed two-level kernel on the same input, and PATH CHOSEN names what ran. A gain or flat on the full decode_1 row is attributable to the chosen variant. The grid/merge-shape ambiguity is better handled inside the prepare (extra variants) than with a second check.

### Open questions
- BUILDER: extend the prepare matrix to grid 2xSM (264 CTAs, C=4 and C=8, co-residency permitting; print cudaOccupancyMaxActiveClusters per C at 95 KB smem and 2 CTAs/SM). Also time, for C in {2,4}, a 'cluster merge + group ticket over cluster partials' variant (Gs = ceil(sqrt(grid/C))). A single global level over 33-66 cluster slots is no shorter than the landed 12+11 chain. Choose by the printed timings, with the 3% bar as written.
- BUILDER: confirm that a cluster launch keeps 2 CTAs/SM at 95264 B smem. If clusters force 1 CTA/SM at grid 2xSM, print it and say so in PATH CHOSEN.
- If no variant clears 3%, the kill note should record whether grid 2xSM + C=8 closed the gap. That tells whether stage 5 is bytes-in-flight (more stages or 2 CTAs/SM at B=1) rather than per-phase stamps.

