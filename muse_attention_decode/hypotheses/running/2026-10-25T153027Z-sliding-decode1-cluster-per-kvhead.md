---
problem: muse_attention_decode
slug: sliding-decode1-cluster-per-kvhead
created: 2026-10-25T153027Z
base: master
area: sliding-layer attention (batch-1 decode, cluster kernel work split)
change: >
  Built on current master (no depends_on). Sliding decode_1 (weight 46, 60% of the decode_1 load)
  runs sliding_cluster_kernel at C=16: ONE cluster of 16 CTAs for the whole step, 10.8 us kernel /
  11.9 us shape, floor 0.1 us. Each CTA owns 128 old keys of BOTH kv heads and, before it can
  compute, builds all 16 query heads (RMSNorm + RoPE with accurate sincosf, 32 B local stack)
  while its gather flies; the last rank also builds the new k/v rows. With only 16 CTAs on 132
  SMs, per-CTA latency is the whole shape. CHANGE (B=1 band only): a template flag PERHEAD (false
  = master, byte-identical) where the grid is 2 clusters of C=16 (32 CTAs), cluster k serving kv
  head k only: each CTA gathers only its head's 64-byte half of every K and V row (16-byte
  cp.async on the 4 chunks of that half; same SW pattern), builds only its 8 query heads, runs the
  same mma.sync QK/PV for one head with all 8 warps splitting the 128 keys (instead of warps
  alternating heads), and the DSMEM reduce-scatter covers 8 rows instead of 16. The new k/v build
  stays on the last rank of EACH cluster for its own head (each writes its half of the 128-byte
  cache row; the two halves are disjoint bytes). Output: rows of head k only. Gate in prepare: B=1
  x 80000, interleaved cold-L2 graph replay (empty graph subtracted, median of 31) PERHEAD vs
  master C=16; route at >= 1.03x and rel L2 < 1e-2 vs master; self-check on master's slide cases
  (lens 1/10/2047/2048/2049/80000, kv 0.25) with cache rows compared to master's, plus the
  two-state alternating replay; else master. B=17..32 and every other path untouched. RULES (every
  path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in
  kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32
  ticket arrays, zeroed once and returned to 0 by every launch, are fine).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "halve each CTA's q-build and compute on the latency chain (32 CTAs, one head each); 10.8 us kernel, target <= 9.5 us; floor 0.1 us"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: Triton route, untouched
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: sk window route, untouched
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T161935Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T161936Z
base_sha: a5f0c41fcf6d081ab9a5f2262b3505c796565fea
base_blob: 5cf8fae31e5b84c36efeeb4569763eb9b06e1c8a
branch: hyp/muse_attention_decode/sliding-decode1-cluster-per-kvhead
worktree: .worktrees/muse_attention_decode/sliding-decode1-cluster-per-kvhead
mode: null
restack_of: null
stale_base: null
---

# sliding-decode1-cluster-per-kvhead

## Change

Built on current master (no depends_on). Sliding decode_1 (weight 46, 60% of the decode_1 load) runs sliding_cluster_kernel at C=16: ONE cluster of 16 CTAs for the whole step, 10.8 us kernel / 11.9 us shape, floor 0.1 us. Each CTA owns 128 old keys of BOTH kv heads and, before it can compute, builds all 16 query heads (RMSNorm + RoPE with accurate sincosf, 32 B local stack) while its gather flies; the last rank also builds the new k/v rows. With only 16 CTAs on 132 SMs, per-CTA latency is the whole shape. CHANGE (B=1 band only): a template flag PERHEAD (false = master, byte-identical) where the grid is 2 clusters of C=16 (32 CTAs), cluster k serving kv head k only: each CTA gathers only its head's 64-byte half of every K and V row (16-byte cp.async on the 4 chunks of that half; same SW pattern), builds only its 8 query heads, runs the same mma.sync QK/PV for one head with all 8 warps splitting the 128 keys (instead of warps alternating heads), and the DSMEM reduce-scatter covers 8 rows instead of 16. The new k/v build stays on the last rank of EACH cluster for its own head (each writes its half of the 128-byte cache row; the two halves are disjoint bytes). Output: rows of head k only. Gate in prepare: B=1 x 80000, interleaved cold-L2 graph replay (empty graph subtracted, median of 31) PERHEAD vs master C=16; route at >= 1.03x and rel L2 < 1e-2 vs master; self-check on master's slide cases (lens 1/10/2047/2048/2049/80000, kv 0.25) with cache rows compared to master's, plus the two-state alternating replay; else master. B=17..32 and every other path untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=sliding load=decode_1 *` faster: halve each CTA's q-build and compute on the latency chain (32 CTAs, one head each); 10.8 us kernel, target <= 9.5 us; floor 0.1 us
- `*layer=sliding load=decode_16 *` flat: Triton route, untouched
- `*layer=sliding load=decode_32 *` flat: sk window route, untouched
- `*layer=full *` flat: untouched
- `*load=verify_* *` flat: untouched

## Risk

Risk: 64-byte half-row gathers halve sector use per request but the shape is latency-bound (0.5 MB window); if the gate keeps master the timing table still tells whether per-CTA work is part of the 10.8 us. Interacts with the running epilogue-gate-l2-prefetch-cuda (gate prefetch in the CUDA epilogues): disjoint lines, both may stack.

## Adversarial review

Reviewed 2026-10-25T161935Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a plausible, self-gated primary. It halves each CTA's q-build, MMA tiles and gather bytes on the only sliding decode_1 kernel (10.8 us) and touches no other shape. Run it with the fixes listed: the reduce-scatter row mapping (HQ/C goes to 0), a C=16 occupancy >= 2 check, and making PERHEAD conditional on C=16 having routed. The headline will sit inside the CI, so read the result from the [slide-gate] PERHEAD-vs-C=16 line and the sliding decode_1 row, not from the geomean.

### Justification
Mechanism: at B=1 the sliding layer is one 16-CTA cluster on 132 SMs, so the shape's time is one CTA's latency chain. Each CTA gathers 128 old keys (one SK=128 sub-chunk, about 32 KB of K+V). Each warp builds 2 query heads (loop h = warp, warp+8, each with a cold qkv load, a shfl reduce and sincosf; at pos*fr <= 80000 rad sincosf stays on the fast Cody-Waite path, so the cost is the load more than the math). Each warp then attends 2 of the 8 tiles for its head. PERHEAD gives each warp 1 head and 1 tile and halves the gather to 4 x 16 B chunks per row. If the compiler does not hoist the second qkv load over the first iteration's smem store, the change also removes one serialised DRAM round trip per warp, which is the most likely source of a real gain (~0.5-1 us). The profile supports a latency-bound kernel: 10.8 us for 0.5 MB is about 48 GB/s, roughly 1.5% of HBM, and the floor is 0.1 us. The expectations are consistent with TASK.md: only the sliding decode_1 row (weight 46) moves. That is about 46 x 1.3 us = 60 us of the 1578 us decode_1 load, around 0.6% on the six-load geomean, well inside the [1.479, 1.530] CI. WEAKEST LINK: nothing shows that per-CTA work is a meaningful part of the 10.8 us. The chain is dominated by things PERHEAD does not shorten: dependent DRAM round trips (seq_lens -> page_table indices -> K/V gather), cluster launch and scheduling of a 16-wide non-portable cluster, two cluster.sync, and the DSMEM reduce. Halving the bytes per row does not halve the latency of one gather round trip. The 9.5 us target is optimistic, a 3% gate is plausible, and 'flat' is the most likely alternative. The gate makes a loss impossible by construction (master is kept), so a non-routing result is still informative if the gate line prints both times. Co-schedulability (extra note): the only measured occupancy in profiles shows cudaOccupancyMaxActiveClusters C=16 -> 14, so 2 clusters fit. The two clusters also have no cross-cluster dependency: disjoint output rows, disjoint 64-byte halves of the new cache row, no shared ticket, and the new k/v is read from smem, never from the cache. So serialisation could only slow it down, never deadlock or corrupt it, and the timing gate would catch the slowdown.

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (landed C=16 at B=1): this is the incumbent PERHEAD is gated against. The comparison is clean only if C=16 itself routed in the same process. If C=16 fell back to Triton (canary, self-check or its own timing gate), PERHEAD must stay off or must be gated against what is actually routed. As written ('vs master C=16') it can route against a C=16 that was never enabled.
- restack-sliding-decode-sk-window-on-cluster-master (landed sk-window decode route): kernel_fn checks _sd_gate before _slide_gate. Today _SD_ROUTE only fills band 3 (B=17..32), so B=1 is not intercepted. The builder must keep that true; if band 1 can ever be filled, PERHEAD is dead code.
- fused-preattention-triton / full-decode stream-K landings: no shared launch or memory traffic with the sliding B=1 single-node graph; no confound.
- In-flight epilogue-gate-l2-prefetch-cuda (named by the hypothesis): it touches the CUDA epilogues, possibly including this kernel's gate read. If it lands first, the absolute times shift for both PERHEAD and C=16 alike; the interleaved gate ratio stays valid.

### Interference and A/B needs
No companion. The change is internally A/B'd: prepare times PERHEAD against master C=16 interleaved, with cold L2, the empty graph subtracted and the median of 31, and routes only at >= 1.03x. The printed gate line is therefore the attribution, independent of the headline. A scored companion could not resolve anything the gate line does not already resolve, and the headline delta (~0.6% expected) is below the CI either way. Requirement instead: the [slide-gate] PERHEAD line must print both times and the ratio whether or not it routes, and the profile's sliding decode_1 top kernel must show which template instance ran (e.g. by name or by the 32-CTA grid).

### Open questions
- Does master's q-build loop issue its two qkv loads per warp back to back, or serialised behind the first iteration's smem store and sincosf? If they already overlap, the q-build half of the argument is mostly gone. SASS or a quick ncu source-level view of the existing C=16 kernel would settle it before the build.
- Is the 10.8 us dominated by the seq_lens -> page_table -> gather dependency chain plus cluster launch? If so, a follow-up should attack that chain (for example, prefetching page indices for both clusters, or PDL from nothing). Neither more CTAs nor less work per CTA shortens it.
- Do the two 16-CTA clusters land on separate GPCs, or do they share SMs (2 CTAs/SM at 101 KB smem)? If 32 CTAs pack onto 16 SMs, per-SM gather bandwidth and issue are shared and the halving partly cancels. Printing %smid or %clusterid once in the canary would show it.
- Would C=8 per head (2 clusters x 8 CTAs = 16 CTAs, 256 keys each) be a cheaper variant to time in the same prepare gate? It needs no new occupancy assumption and keeps DSMEM fan-in at 8.

