---
problem: muse_attention_decode
slug: full-verify-wgmma-ws-s1-producer-split
created: 2026-10-25T153027Z
base: master
area: full-layer attention (verify, wgmma stream-K kernel, warp-specialized pipeline)
change: >
  Fresh stage 1 on CURRENT master (no depends_on). Why now: the s3/s4 wgmma full-verify items are
  held because the stage-3 kernel failed its self-check; that hold reason is FIXED:
  full-verify-wgmma-s5-qfrag-reload (job 71e40388a1904937, branch
  hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl 69378d7) passed its whole
  bisect table (19 cases incl. B1-2t..B1-34t, mid-request 1-tile segments, kv 0.25, grids
  3/7/61/device, coherent data), the DIAG (tile-1 S max abs err 5.5e-3, S correct) and the
  two-state alternating graph replays (worst rel L2 5.3e-3). What still blocks the line is SPEED:
  its prepare gate measured wgmma 288.7 us vs master paired FA3 164.4 us at B=16 (0.570x) and
  429.5 vs 302.4 us at B=32 (0.704x), so it never routed. Reading the s5 kernel names why: (a) the
  256 threads that compute are also the only gather issuers, and each thread issues tile n+3 only
  after waiting the EMPTY barrier of tile n-1, which needs all 256 threads of BOTH warpgroups: a
  CTA-wide rendezvous per tile, so the slower kv head gates the ring and load issue rate equals
  compute rate; (b) the per-tile chain is fully serial: S wgmma commit -> V^T transpose ->
  wait_group 0 -> softmax -> PV wgmma commit -> wait_group 0, no overlap across tiles; at B=16
  that is ~76 tiles per CTA in 288 us = ~3.8 us per 16 KB tile, i.e. 1.14 TB/s. CHANGE: port the
  s5 kernel (verbatim math: SW128 ring, fp8 QK wgmma m64n64k32 with Q staged in smem and reloaded
  per tile, Max_offset-8 FP8 P, per-warpgroup V^T transpose, in-kernel ticket combine + norm-gate,
  stream-K over 64-token tiles, page-index prefetch) into a new load_inline module
  muse_full_verify_wgmma_ws_v1 and restructure ONLY the pipeline: 384 threads = 3 warpgroups,
  __launch_bounds__(384, 1), 1 CTA/SM, grid = SM count. WG0 = producer: setmaxnreg.dec to 40, its
  128 threads issue every 16-byte cp.async.cg of each K/V tile (8 per thread per tile) plus the
  page-index blocks, and arrive (cp.async.mbarrier.arrive.noinc) on FULL[st]; it waits EMPTY[st]
  (256 consumer arrivals) only to refill stage st; the consumers no longer issue loads, and the 8-stage ring absorbs
  skew between the two consumers instead of a per-tile rendezvous. Ring STAGES = 8 (8 x 16 KB = 128 KB) so up to 7 tiles are in flight per SM. WG1/WG2
  = consumers, kv head 0/1, setmaxnreg.inc to 232: never issue loads; per tile: wait FULL, issue
  S(n) wgmma, transpose V while it runs (as s5), then FA3-style intra-warpgroup overlap: keep two
  S accumulator sets, issue S(n+1) as soon as FULL[n+1] lands, before softmax(n), and wait PV(n)
  with wgmma.wait_group 1 rather than 0, so tensor-core work of tile n+1 overlaps the softmax of
  tile n. Arrive EMPTY[st] after the last wgmma reading stage st retires. Everything else (segment
  start/end, partial slots, ticket combine, norm-gate epilogue, error word, bounded mbarrier
  waits) as s5. Template variant TILE in {64, 128} (128 = m64n128k32 S, 4 PV k-steps, half the
  per-tile barriers and rescales); both built. PHASE DIAG (prepare-only DBG instantiation, never
  timed): clock64 stamps per consumer phase (FULL wait, S issue->ready, transpose, softmax, PV)
  and per producer phase (EMPTY wait, issue), CTA 0 at B=16 x 80K, printed as mean ns per tile, so
  whichever phase is still long is named by this run. GATE: in prepare, same harness as s5
  (_wg_time_graph: own CUDA graph per path, interleaved replays, 256 MB L2 flush, empty-graph
  baseline subtracted, lengths 74K-86K, median of 40) for each TILE variant vs master paired FA3
  path (pre-attention + prepare + FA3 + combine + paired norm-gate) at B=16 and B=32; route a band
  only if best variant >= 1.05x AND its 80K output matches master's by rel L2 < 3e-2, finite,
  error word 0. Self-check = s5's bisect table + two-state alternating replay for each routed
  variant. Any miss: that band stays on master paired FA3 with a PATH CHOSEN line naming the case.
  Gate predicate config-only: full layer, q_len 8, 1 < batch <= 64, tp8, head_dim 64, page 1.
  BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, all wgmma/mbarrier/setmaxnreg under
  __CUDA_ARCH_FEAT_SM90_ALL, -lineinfo, print regs/local bytes per variant and refuse a variant
  with local bytes > 0. Decode, sliding and full verify_1 paths untouched. RULES (every path):
  CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in
  kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32
  ticket arrays, zeroed once and returned to 0 by every launch, are fine).
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: "incumbent shape 163.0 us (FA3 main 154.5 + combine 5.9 + prepare 3.0 + paired norm-gate 2.5 + pre-attn 2.7); floor 63.4 us; s5 serial kernel 288.7 us. M1: gate routes (>=1.05x, <=156 us replaced); M2: kernel <=135 us (~2.4 TB/s, under FA3's ~2.2); M3: <=95 us (1.5x floor)"
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: incumbent shape 302.3 us (FA3 293.1 + combine 7.7 + prepare 3.6 + norm-gate 3.9); floor 126.8 us; s5 429.5 us. M1 routed <=290 us; M2 <=255 us; M3 <=190 us
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*layer=full load=verify_1 *"
    direction: flat
    why: band 1 < B only
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)"
review: required
status: queued
---

# full-verify-wgmma-ws-s1-producer-split

## Change

Fresh stage 1 on CURRENT master (no depends_on). Why now: the s3/s4 wgmma full-verify items are held because the stage-3 kernel failed its self-check; that hold reason is FIXED: full-verify-wgmma-s5-qfrag-reload (job 71e40388a1904937, branch hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl 69378d7) passed its whole bisect table (19 cases incl. B1-2t..B1-34t, mid-request 1-tile segments, kv 0.25, grids 3/7/61/device, coherent data), the DIAG (tile-1 S max abs err 5.5e-3, S correct) and the two-state alternating graph replays (worst rel L2 5.3e-3). What still blocks the line is SPEED: its prepare gate measured wgmma 288.7 us vs master paired FA3 164.4 us at B=16 (0.570x) and 429.5 vs 302.4 us at B=32 (0.704x), so it never routed. Reading the s5 kernel names why: (a) the 256 threads that compute are also the only gather issuers, and each thread issues tile n+3 only after waiting the EMPTY barrier of tile n-1, which needs all 256 threads of BOTH warpgroups: a CTA-wide rendezvous per tile, so the slower kv head gates the ring and load issue rate equals compute rate; (b) the per-tile chain is fully serial: S wgmma commit -> V^T transpose -> wait_group 0 -> softmax -> PV wgmma commit -> wait_group 0, no overlap across tiles; at B=16 that is ~76 tiles per CTA in 288 us = ~3.8 us per 16 KB tile, i.e. 1.14 TB/s. CHANGE: port the s5 kernel (verbatim math: SW128 ring, fp8 QK wgmma m64n64k32 with Q staged in smem and reloaded per tile, Max_offset-8 FP8 P, per-warpgroup V^T transpose, in-kernel ticket combine + norm-gate, stream-K over 64-token tiles, page-index prefetch) into a new load_inline module muse_full_verify_wgmma_ws_v1 and restructure ONLY the pipeline: 384 threads = 3 warpgroups, __launch_bounds__(384, 1), 1 CTA/SM, grid = SM count. WG0 = producer: setmaxnreg.dec to 40, its 128 threads issue every 16-byte cp.async.cg of each K/V tile (8 per thread per tile) plus the page-index blocks, and arrive (cp.async.mbarrier.arrive.noinc) on FULL[st]; it waits EMPTY[st] (256 consumer arrivals) only to refill stage st; the consumers no longer issue loads, and the 8-stage ring absorbs skew between the two consumers instead of a per-tile rendezvous. Ring STAGES = 8 (8 x 16 KB = 128 KB) so up to 7 tiles are in flight per SM. WG1/WG2 = consumers, kv head 0/1, setmaxnreg.inc to 232: never issue loads; per tile: wait FULL, issue S(n) wgmma, transpose V while it runs (as s5), then FA3-style intra-warpgroup overlap: keep two S accumulator sets, issue S(n+1) as soon as FULL[n+1] lands, before softmax(n), and wait PV(n) with wgmma.wait_group 1 rather than 0, so tensor-core work of tile n+1 overlaps the softmax of tile n. Arrive EMPTY[st] after the last wgmma reading stage st retires. Everything else (segment start/end, partial slots, ticket combine, norm-gate epilogue, error word, bounded mbarrier waits) as s5. Template variant TILE in {64, 128} (128 = m64n128k32 S, 4 PV k-steps, half the per-tile barriers and rescales); both built. PHASE DIAG (prepare-only DBG instantiation, never timed): clock64 stamps per consumer phase (FULL wait, S issue->ready, transpose, softmax, PV) and per producer phase (EMPTY wait, issue), CTA 0 at B=16 x 80K, printed as mean ns per tile, so whichever phase is still long is named by this run. GATE: in prepare, same harness as s5 (_wg_time_graph: own CUDA graph per path, interleaved replays, 256 MB L2 flush, empty-graph baseline subtracted, lengths 74K-86K, median of 40) for each TILE variant vs master paired FA3 path (pre-attention + prepare + FA3 + combine + paired norm-gate) at B=16 and B=32; route a band only if best variant >= 1.05x AND its 80K output matches master's by rel L2 < 3e-2, finite, error word 0. Self-check = s5's bisect table + two-state alternating replay for each routed variant. Any miss: that band stays on master paired FA3 with a PATH CHOSEN line naming the case. Gate predicate config-only: full layer, q_len 8, 1 < batch <= 64, tp8, head_dim 64, page 1. BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, all wgmma/mbarrier/setmaxnreg under __CUDA_ARCH_FEAT_SM90_ALL, -lineinfo, print regs/local bytes per variant and refuse a variant with local bytes > 0. Decode, sliding and full verify_1 paths untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=full load=verify_16 *` faster: incumbent shape 163.0 us (FA3 main 154.5 + combine 5.9 + prepare 3.0 + paired norm-gate 2.5 + pre-attn 2.7); floor 63.4 us; s5 serial kernel 288.7 us. M1: gate routes (>=1.05x, <=156 us replaced); M2: kernel <=135 us (~2.4 TB/s, under FA3's ~2.2); M3: <=95 us (1.5x floor)
- `*layer=full load=verify_32 *` faster: incumbent shape 302.3 us (FA3 293.1 + combine 7.7 + prepare 3.6 + norm-gate 3.9); floor 126.8 us; s5 429.5 us. M1 routed <=290 us; M2 <=255 us; M3 <=190 us
- `*load=decode_* *` flat: untouched
- `*layer=sliding *` flat: untouched
- `*layer=full load=verify_1 *` flat: band 1 < B only

## Risk

Fixes the hold reason of full-verify-wgmma-s3-warpgroup-per-kvhead / -s4-sliding-window / full-verify1-wgmma-s4-batch1-grid-sm (correctness, fixed by s5) and attacks the measured 0.57x/0.70x speed gap that s5 exposed; those held items stay held (they would extend a 0.57x kernel). Risks: setmaxnreg + 1 CTA/SM at 384 threads; LESSONS full_ws: a 1-CTA/SM single-producer-WARP bulk design was issue-bound at 0.5 TB/s; here the producer is a 128-thread warpgroup issuing the same 16 B cp.async as master's 2.9-3.0 TB/s decode kernel, and the PHASE DIAG names the producer if it is still the limiter. Fallback to master is free (hard gate). Floor: verify_16 63.4 us, verify_32 126.8 us (TASK).

## Milestones

Rewrite stage 1 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
