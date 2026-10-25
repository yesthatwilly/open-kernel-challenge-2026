---
problem: muse_attention_decode
slug: full-verify-wgmma-s6-split-ring-2cta
created: 2026-10-25T212048Z
base: master
area: full-layer attention (verify, wgmma stream-K kernel, per-warpgroup ring)
change: >
  DIAGNOSIS action 1(a)+(b): the sibling of the queued full-verify-wgmma-ws-s1-producer-split that
  KEEPS s5's proven occupancy (2 CTAs/SM, every thread issues loads) and removes only the CTA-wide
  per-tile rendezvous. Fresh module on CURRENT master 5d528bf3 (no depends_on: s5's branch sits on
  a stale base, LESSONS: rebase before a stage judged on the headline). SOURCE =
  full-verify-wgmma-s5-qfrag-reload (branch
  hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl 69378d7, job 71e40388):
  splitkv_verify_wgmma_kernel<ticket>, regs 127, local 0, 2 CTAs/SM, dyn smem 99936 B, correct on
  all 19 bisect cases and two-state replays, but 288.7 us vs paired FA3 164.4 us at B=16 and 429.5
  vs 302.4 at B=32 (2x the bytes cost only 1.49x the time: latency-bound in the serial per-tile
  chain plus the EMPTY barrier that needs all 256 threads of BOTH warpgroups every tile). Port it
  verbatim into a new load_inline module muse_full_verify_wgmma_s6 and change ONLY pipeline
  ownership. (1) SPLIT RING: warpgroup w (w = kv head 0/1) owns a private ring holding only its
  head's 64-byte half of each K and V row (64 tokens x 64 B x {K,V} = 8 KB per stage), STAGES_WG =
  6 (2 x 6 x 8 KB = 96 KB, under s5's 99936 B, so 2 CTAs/SM holds). Its 128 threads issue that
  ring's 16-B cp.async.cg (4 per thread per tile) at prefetch distance STAGES_WG-2;
  FULL_w[st]/EMPTY_w[st] are per-warpgroup mbarriers with 128 arrivals
  (cp.async.mbarrier.arrive.noinc for FULL), so a refill waits only for its own warpgroup. The
  page-index block for a segment is filled once at segment start behind the one remaining CTA-wide
  barrier; the other CTA-wide barriers are only the ticket epilogue and the norm-gate. Smem
  layout: the 64-B half row is a K-major 64x64 e4m3 tile, use the 64B swizzle (descriptor layout
  SWIZZLE_64B) with s5's descriptor builder at the half-row stride; if that fails the self-check,
  fall back to s5's SW128 128-B-stride layout with each ring filling only its half (STAGES_WG 3)
  and say so in the log. (2) Template OVL. OVL=0: split ring + s5's serial S -> transpose ->
  softmax -> PV chain at __launch_bounds__(256,2), grid 2xSM: 4 independent warpgroups per SM hide
  each other's latency. OVL=1: split ring + FA3-style intra-warpgroup overlap: issue S(n+1)
  (FULL_w[n+1] waited) before softmax(n), wgmma.wait_group 1 instead of 0, which needs a second
  64x64 fp32 S accumulator (+32 regs over 127, so it cannot fit 2 CTAs/SM):
  __launch_bounds__(256,1), grid SM, STAGES_WG 12 (192 KB). Build both. (3) GATHER-ONLY DIAG
  (action 1a), prepare-only, never routed or timed in kernel_fn: a GATHER_ONLY instantiation of
  each ring design (s5's CTA-wide ring at 2 CTAs/SM, split ring OVL=0, split ring 1 CTA/SM 12
  stages) that runs the same cp.async/mbarrier/page-index schedule with every wgmma, transpose and
  softmax compiled out (consumers wait FULL, xor one word of the stage into a dummy output, arrive
  EMPTY), at B=16 x 80K agentic lengths; print '[s6-diag] gather-only <design>: X us, Y TB/s' next
  to each full variant's time, so the producer issue rate is measured before compute is blamed.
  Everything else verbatim from s5: FP8 QK wgmma m64n64k32 with Q staged in smem and reloaded
  before every tile's S (s5's fix), Max_offset-8 FP8 P, V^T transpose per warpgroup (now from its
  own ring), stream-K over 64-token tiles, in-kernel ticket combine + paired norm-gate, error
  word, bounded mbarrier waits. GATE: s5's harness (_wg_time_graph: own graph per path,
  interleaved, lengths 74K-86K, median of 40): OVL=0 and OVL=1 vs master paired FA3 path
  (pre-attention + prepare + FA3 + combine + paired norm-gate) at B=16 and B=32; route a band
  (2..24 timed at 16, 25..64 timed at 32) only if the best variant >= 1.05x AND rel L2 < 3e-2 vs
  master, finite, error word 0. SELF-CHECK first: s5's bisect table (19 cases incl. B1-2t..B1-34t,
  mid-request 1-tile segments, kv 0.25, grids 3/7/61/device, coherent data, never random keys) for
  each built variant, then the two-state alternating capture-once replay for any routed variant;
  any miss leaves that band on master paired FA3 with a PATH CHOSEN line naming the case. New
  CUDA/PTX with fault risk: the first measurement is an okc check. Import clock is 222 s over 4
  modules: give this build its own deadline and fall back on overrun. Decode, sliding and full
  verify_1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no
  module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every
  launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags
  are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print
  regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed
  interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only
  when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: "replaced: FA3 main 148.7 + combine 6.4 + prepare 3.1 + paired norm-gate ~2.5 us of a 162.5 us shape; floor 63.4 us; s5 288.7 us. M1 routed (>=1.05x, shape <=156); M2 shape <=135 (~2.4 TB/s); M3 <=115 (decode-level 3.0 TB/s). 0.073%/us: M3 is about +3.4% headline"
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: "replaced: FA3 288.7 + combine 6.7 + prepare ~3.6 + norm-gate ~3.9 of a 302.0 us shape; floor 126.8; s5 429.5. M1 routed <=290; M2 <=255; M3 <=225. 0.043%/us: M3 is about +3.3%"
  - shapes: "*layer=full load=verify_1 *"
    direction: flat
    why: band 1<B only
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
prior: "full-verify-wgmma-s5-qfrag-reload: same math and 2 CTAs/SM, but each warpgroup gets a private per-kv-head ring with 128-arrival barriers (no CTA-wide EMPTY rendezvous per tile), an OVL=1 variant issues S(n+1) before softmax(n) with wait_group 1, and a gather-only DIAG measures each ring's issue rate; differs from queued full-verify-wgmma-ws-s1-producer-split by keeping all 256 threads as issuers instead of a 128-thread producer at 1 CTA/SM"
rewrite_stage: 6
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)"
review: required
status: queued
---

# full-verify-wgmma-s6-split-ring-2cta

## Change

DIAGNOSIS action 1(a)+(b): the sibling of the queued full-verify-wgmma-ws-s1-producer-split that KEEPS s5's proven occupancy (2 CTAs/SM, every thread issues loads) and removes only the CTA-wide per-tile rendezvous. Fresh module on CURRENT master 5d528bf3 (no depends_on: s5's branch sits on a stale base, LESSONS: rebase before a stage judged on the headline). SOURCE = full-verify-wgmma-s5-qfrag-reload (branch hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl 69378d7, job 71e40388): splitkv_verify_wgmma_kernel<ticket>, regs 127, local 0, 2 CTAs/SM, dyn smem 99936 B, correct on all 19 bisect cases and two-state replays, but 288.7 us vs paired FA3 164.4 us at B=16 and 429.5 vs 302.4 at B=32 (2x the bytes cost only 1.49x the time: latency-bound in the serial per-tile chain plus the EMPTY barrier that needs all 256 threads of BOTH warpgroups every tile). Port it verbatim into a new load_inline module muse_full_verify_wgmma_s6 and change ONLY pipeline ownership. (1) SPLIT RING: warpgroup w (w = kv head 0/1) owns a private ring holding only its head's 64-byte half of each K and V row (64 tokens x 64 B x {K,V} = 8 KB per stage), STAGES_WG = 6 (2 x 6 x 8 KB = 96 KB, under s5's 99936 B, so 2 CTAs/SM holds). Its 128 threads issue that ring's 16-B cp.async.cg (4 per thread per tile) at prefetch distance STAGES_WG-2; FULL_w[st]/EMPTY_w[st] are per-warpgroup mbarriers with 128 arrivals (cp.async.mbarrier.arrive.noinc for FULL), so a refill waits only for its own warpgroup. The page-index block for a segment is filled once at segment start behind the one remaining CTA-wide barrier; the other CTA-wide barriers are only the ticket epilogue and the norm-gate. Smem layout: the 64-B half row is a K-major 64x64 e4m3 tile, use the 64B swizzle (descriptor layout SWIZZLE_64B) with s5's descriptor builder at the half-row stride; if that fails the self-check, fall back to s5's SW128 128-B-stride layout with each ring filling only its half (STAGES_WG 3) and say so in the log. (2) Template OVL. OVL=0: split ring + s5's serial S -> transpose -> softmax -> PV chain at __launch_bounds__(256,2), grid 2xSM: 4 independent warpgroups per SM hide each other's latency. OVL=1: split ring + FA3-style intra-warpgroup overlap: issue S(n+1) (FULL_w[n+1] waited) before softmax(n), wgmma.wait_group 1 instead of 0, which needs a second 64x64 fp32 S accumulator (+32 regs over 127, so it cannot fit 2 CTAs/SM): __launch_bounds__(256,1), grid SM, STAGES_WG 12 (192 KB). Build both. (3) GATHER-ONLY DIAG (action 1a), prepare-only, never routed or timed in kernel_fn: a GATHER_ONLY instantiation of each ring design (s5's CTA-wide ring at 2 CTAs/SM, split ring OVL=0, split ring 1 CTA/SM 12 stages) that runs the same cp.async/mbarrier/page-index schedule with every wgmma, transpose and softmax compiled out (consumers wait FULL, xor one word of the stage into a dummy output, arrive EMPTY), at B=16 x 80K agentic lengths; print '[s6-diag] gather-only <design>: X us, Y TB/s' next to each full variant's time, so the producer issue rate is measured before compute is blamed. Everything else verbatim from s5: FP8 QK wgmma m64n64k32 with Q staged in smem and reloaded before every tile's S (s5's fix), Max_offset-8 FP8 P, V^T transpose per warpgroup (now from its own ring), stream-K over 64-token tiles, in-kernel ticket combine + paired norm-gate, error word, bounded mbarrier waits. GATE: s5's harness (_wg_time_graph: own graph per path, interleaved, lengths 74K-86K, median of 40): OVL=0 and OVL=1 vs master paired FA3 path (pre-attention + prepare + FA3 + combine + paired norm-gate) at B=16 and B=32; route a band (2..24 timed at 16, 25..64 timed at 32) only if the best variant >= 1.05x AND rel L2 < 3e-2 vs master, finite, error word 0. SELF-CHECK first: s5's bisect table (19 cases incl. B1-2t..B1-34t, mid-request 1-tile segments, kv 0.25, grids 3/7/61/device, coherent data, never random keys) for each built variant, then the two-state alternating capture-once replay for any routed variant; any miss leaves that band on master paired FA3 with a PATH CHOSEN line naming the case. New CUDA/PTX with fault risk: the first measurement is an okc check. Import clock is 222 s over 4 modules: give this build its own deadline and fall back on overrun. Decode, sliding and full verify_1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=verify_16 *` faster: replaced: FA3 main 148.7 + combine 6.4 + prepare 3.1 + paired norm-gate ~2.5 us of a 162.5 us shape; floor 63.4 us; s5 288.7 us. M1 routed (>=1.05x, shape <=156); M2 shape <=135 (~2.4 TB/s); M3 <=115 (decode-level 3.0 TB/s). 0.073%/us: M3 is about +3.4% headline
- `*layer=full load=verify_32 *` faster: replaced: FA3 288.7 + combine 6.7 + prepare ~3.6 + norm-gate ~3.9 of a 302.0 us shape; floor 126.8; s5 429.5. M1 routed <=290; M2 <=255; M3 <=225. 0.043%/us: M3 is about +3.3%
- `*layer=full load=verify_1 *` flat: band 1<B only
- `*load=decode_* *` flat: untouched
- `*layer=sliding *` flat: untouched

## Risk

Fallback is free (hard >=1.05x gate keeps master paired FA3). Risks: SWIZZLE_64B half-row descriptors are new (fallback layout given); OVL=1 is 1 CTA/SM so it trades TLP for ILP and the gate decides; a fault in the self-check poisons the context, so check first, never a submit. Milestones on the replaced kernels: M1 routed, M2 verify_16 <=135 / verify_32 <=255 us, M3 <=115 / <=225 us. Next stage: full-verify1-wgmma-s7-grid-sm-two-level (B=1) on this code whatever its score.

## Milestones

Rewrite stage 6 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
