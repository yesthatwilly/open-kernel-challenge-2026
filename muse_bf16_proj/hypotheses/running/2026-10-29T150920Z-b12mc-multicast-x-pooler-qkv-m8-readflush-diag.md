---
problem: muse_bf16_proj
slug: b12mc-multicast-x-pooler-qkv-m8-readflush-diag
created: 2026-10-29T150920Z
base: master
mint_blob: "a36f33927c06500cbba267d7e0be2235eb8cd716"
best_at_mint: 1.2794
area: "bulk12 row-owner kernel at m>=8 (x delivery: TMA multicast per 4-CTA cluster)"
change: >
  Prepare-only diagnostic. kernel_fn, every routing flag and every state key the timed path reads
  stay byte-identical to master (62da2d9e, blob a36f3392); the diag adds no state key and frees
  all its buffers before prepare() returns. (1) KERNEL: a new load_inline module
  muse_bf16_b12mc_diag_v1, built at import right after _b12_build (same BULK_NVCC_FLAGS, 9.0a
  only; a build failure prints the VERDICT as n/a: build and changes nothing else). Its kernel
  b12mc_kernel<R, K, S, MINB, CW, NT, MC> is derived from the ROTATED b12x_kernel at commit
  a7da6fa9 (git show a7da6fa9:muse_bf16_proj/submission.py, from the comment "b12x smem below the
  ring" ~line 3327: keep its helpers ldsm_x2 / ldsm_x4 / b12x_esx, the NN = NT/8 n8 tiles per
  decoded A fragment, the producer's escape-x gather after the stream and the RP-padded epilogue);
  the cluster launch (cudaLaunchKernelEx + cudaLaunchAttributeClusterDimension) and the mapa
  helper come from master's section (7) ouns_kernel. Changes vs b12x: (a) cluster dims (4,1,1),
  grid N/R unchanged (pooler R=32: 128 CTAs = 32 clusters; qkv_gate R=16: 144 CTAs = 36 clusters);
  the chunk rotation becomes per cluster, jc = (j + blockIdx.x / 4) % NCHUNK, identical in the 4
  ranks (each CTA still streams its own weight chunk jc with the existing bulk_g2s). (b) x is no
  longer copied as 16 B cp.async pieces by the producer lanes: the host encodes a 2D CUtensorMap
  over x (cuTensorMapEncodeTiled fetched with cudaGetDriverEntryPoint, so no -lcuda; BFLOAT16,
  globalDim {K, m}, globalStride {2K} B, boxDim {64, NT}, SWIZZLE_128B, L2 promotion 128B, OOB
  fill none) passed as const __grid_constant__ CUtensorMap. A chunk's x segment is CKK/64 boxes
  (pooler 4, qkv 8; 64 k x NT rows, 1 KB at NT=8). With MC=true, producer lane 0 of cluster rank r
  issues the boxes b with b % 4 == r as
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
  [xslot + b*NT*128], [tmap, {jc*CKK + 64b, 0}], [full_s], ctaMask 0xF: the same CTA-relative
  offsets are written in all 4 CTAs and each CTA's full_s gets the bytes. MC=false is the control
  arm: the CTA issues all CKK/64 boxes itself without .multicast, everything else identical. (c)
  Slot layout: SLOT = WB + NT*CKK*2 (pooler NT=8 16384 B, NT=16 20480 B; qkv NT=8 20480 B); the
  ring offset is rounded to 1024 B (b12x_ring_off rounds to 128 today) so each x box is 1024 B
  aligned as SWIZZLE_128B requires. The consumers' ldmatrix row address for token tok and
  chunk-local k = kk16*16 + khalf*8 becomes xslot + (k/64)*NT*128 + tok*128 + ((((k%64)/8) ^ (tok
  & 7)) * 16), replacing the XS-padded address (conflict-free through the swizzle). (d) Barriers:
  full_s init count 1, armed by the local lane 0 with mbar_expect_tx(full_s, WB + NT*CKK*2) before
  it issues anything into slot s; a peer's multicast bytes may land before the local arm, which is
  legal because the phase cannot complete before the local arrival (CUTLASS's multicast pipeline
  relies on the same); the cp.async.mbarrier.arrive.noinc path and the 1 + 32 count are removed.
  empty_s becomes cluster-wide: init count CW*4; after a chunk each consumer warp's lane 0 arrives
  on slot s's empty barrier of every rank q = 0..3 with
  mbarrier.arrive.release.cluster.shared::cluster.b64 _, [mapa(empty_s, q)]; the producer waits
  its local empty_s with mbarrier.try_wait.parity.acquire.cluster, so no rank multicasts into slot
  s before all 4 ranks' consumers freed it. Every wait stays bounded (B12_WAIT_NS, error word).
  (e) Start: after mbar_init and fence.mbarrier_init.release.cluster,
  barrier.cluster.arrive.release + barrier.cluster.wait.acquire (all threads) replaces the first
  __syncthreads, so no peer signals an uninitialised barrier; end: one more cluster arrive/wait
  after the epilogue store, so no CTA exits while a peer may still arrive on its barriers.
  Instances (6): pooler <32, 8192, 8, 1, 8, NT, MC> at NT = 8 and 16, qkv_gate <16, 8192, 4, 2, 4,
  8, MC>, each with MC true and false. A case whose cudaOccupancyMaxActiveClusters is below its
  cluster count (32 pooler, 36 qkv) prints skipped. (2) HARNESS: _mcx_diag(state), called last in
  prepare() after the occupancy re-print and before the FINAL FLAGS line, skipped (VERDICT n/a:
  budget) when prepare has already run > 200 s. Per case (pooler m=8, pooler m=16, qkv_gate m=8):
  encode state[layer] with master's _b12_encode at the instance's R (the encoding b12x shared with
  the m=1 cell) and check it bit-exact with master's b12_unpack; run _cell_check (fp64 reference +
  canary) on mc and uc, and require mc and uc outputs bit-identical; any failure marks the case
  INVALID. Timing copies _cell_gate's replay loop without a routing flag: arms inc =
  kernel_fn(data, state) on the final state (master's routed incumbent for that layer and m), mc
  and uc, each captured with _cell_capture on the same synthetic x; the CELL_FLUSH_BYTES (256 MB)
  READ flush (flush_i32.sum(), as the judge reads) before every replay; CELL_GATE_REPS (41) reps
  in rotating interleaved order; empty-graph baseline subtracted; 2 rounds. Print per round each
  arm's net us and IQR, mc/inc and mc/uc, plus a reference arm kernel_fn at pooler m=1 (master's
  b12 cell). Then one _decision line, so it is re-printed in the 16 KB log tail: [mcx] VERDICT
  pooler m8 mc/inc=<worst round> mc/uc=<..> | pooler m16 mc/inc=<..> | qkv m8 mc/inc=<..> ->
  LEAD|SHELF|REFUTE|INVALID, per the decision rule. Free every diag tensor, graph and tensor map
  and call torch.cuda.empty_cache(). Churn runs it as a check and never submits it.
expected:
  - shapes: "*"
    direction: unchanged
    why: "prepare-only diagnostic: kernel_fn and routing are byte-identical to master, so the check should score the 1.2794 board within the same-bytes spread (~+-1.2%); it is never submitted"
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: "the lever, not this run: incumbent 19.0 us judged (triage of 3e3ffc5a, elasticity 6.6%); at the m=1 bulk12 judge ratio 0.84 it would be ~16.0 us. Diag target: mc/inc <= 0.85 under the read flush"
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: "lever: 19.4 us, elasticity 5.1%; the diag's pooler m=16 case (NN=2 n8 tiles per fragment) predicts it"
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: "lever: 19.9 us, elasticity 5.2%; extrapolated from m=8 / m=16, not measured by the diag"
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: "lever: 18.6 us, elasticity 4.9% (R=64, K=4096: 2 boxes per chunk); proxied by pooler m=16"
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: "lever: 18.3 us, elasticity 4.8%. Gain arithmetic: elasticities 6.6+5.1+5.2+4.9+4.8 = 26.6% x ln(1/0.84) = 0.174 -> +4.6% headline, discounted to ~4.2% because m=16/32 decode 2-4 n8 tiles per fragment and pay more x per chunk; qkv m=8 (4.4%) would add ~0.8% on top if its ratio matched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "qkv-rowowner-a2-widen-m8-m16: that put m=8/16 on the raw-bf16 row-owner with x resident; the closest bulk12 line, the campaign's b12x checks (22ddaf4a, 0acdca48, 2980f57a: per-row TMA, then 16 B cp.async, then per-CTA rotation), streamed x per CTA and lost to L2 hot-spotting. Here x is read ONCE per 4-CTA cluster by 2D tensor-map TMA multicast with per-cluster rotation, timed under the judge's read flush: exactly the reopen condition of the [refuted] b12x claim (CLAIMS:50)"
target: diag
value_gain: 4.2
value_p: 0.25
value_ev: 1.05
value_cost: 250k tokens (CUDA cluster/TMA-multicast build + Opus audit), 1 judge run
evidence: >
  b12x at m>=8 lost on the judge: qkv m=8 x0.769 (12.86 -> 15.63 us, check 6617f94d; gate 0.976
  under the old write flush). After the cp.async fix ~1 us per token row remained (0acdca48:
  pooler m=8/16/32 25.3/34.9/52.8 vs incumbent 23.4-24.1 net us), read as L2 hot-spotting (every
  CTA reads the same x lines); per-CTA chunk rotation cut qkv m=8 17.1 -> 14.5 net us (2980f57a),
  confirming part of it (FINDINGS bulk12; DISTILL:179). Resident x (b12r) judged pooler m=8 only
  x1.016, 19.07 -> 18.68 us (6617f94d). bulk12 at m=1 judges 0.84 of its incumbent (pooler 17.7 ->
  14.8 us, 3834ceec). Triage of 3e3ffc5a: pooler v1 m=8 19.0 us (6.6%), pooler d16/d32 19.4/19.9
  us (5.1/5.2%), unpooler d16/d32 18.6/18.3 us (4.9/4.8%).
decision: >
  Read the WORST of the 2 read-flush rounds of mc/inc at pooler m=8 (r8) and pooler m=16 (r16),
  and mc/uc at pooler m=8. Checked in this order. INVALID (build, occupancy, round-trip,
  _cell_check or mc-vs-uc bit-identity failure): fix and rerun the diag once; no claim. REFUTE: r8
  >= 1.0, or mc/uc >= 0.97 at pooler m=8 (multicast does not remove the per-row cost, so L2
  hot-spotting is not the bottleneck) -> CLAIMS [refuted] multicast x at m>=8; b12x stays closed.
  LEAD: r8 <= 0.85 and r16 <= 0.92 -> mint the check b12mc pooler m=8/16/32 + unpooler m=16/32 (+
  qkv m=8 if its mc/inc <= 0.90), each cell routed only at a read-flush gate <= 0.95
  (B12_MAX_RATIO_M8), with CLAIMS [lead] reopening b12x. SHELF: anything else (0.85 < r8 < 1.0, or
  r8 <= 0.85 with r16 > 0.92, where the m=8 cells alone are worth ~1.2-1.9%, under the bar) ->
  CLAIMS [shelf], minted only inside a bigger bf16 stack.
instrument: >
  prepare-time graph-replay timer copied from _cell_gate (no routing flag): 256 MB READ flush
  (flush_i32.sum(), as the judge reads) before every replay, 41 interleaved reps x 2 rounds,
  empty-graph baseline subtracted; arms inc = kernel_fn on master's final state, mc (multicast x),
  uc (same kernel, per-CTA TMA x, the multicast control), plus pooler m=1 as reference;
  correctness by _cell_check (fp64) and mc==uc bit-identity; the VERDICT goes through _decision
  into the 16 KB log tail; the judge's own rows of the unchanged timed path are the board-noise
  control
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T151743Z
review_companions: null
review_best: 1.279360358306445
review_blob: a36f33927c06500cbba267d7e0be2235eb8cd716
stage: claimed
claimed_at: 2026-10-29T151743Z
base_sha: c54503e25e6fe0056cea3feba5d4e4982d602c7d
base_blob: a36f33927c06500cbba267d7e0be2235eb8cd716
branch: hyp/muse_bf16_proj/b12mc-multicast-x-pooler-qkv-m8-readflush-diag
worktree: .worktrees/muse_bf16_proj/b12mc-multicast-x-pooler-qkv-m8-readflush-diag
mode: null
restack_of: null
stale_base: null
---

# b12mc-multicast-x-pooler-qkv-m8-readflush-diag

## Change

Prepare-only diagnostic. kernel_fn, every routing flag and every state key the timed path reads stay byte-identical to master (62da2d9e, blob a36f3392); the diag adds no state key and frees all its buffers before prepare() returns. (1) KERNEL: a new load_inline module muse_bf16_b12mc_diag_v1, built at import right after _b12_build (same BULK_NVCC_FLAGS, 9.0a only; a build failure prints the VERDICT as n/a: build and changes nothing else). Its kernel b12mc_kernel<R, K, S, MINB, CW, NT, MC> is derived from the ROTATED b12x_kernel at commit a7da6fa9 (git show a7da6fa9:muse_bf16_proj/submission.py, from the comment "b12x smem below the ring" ~line 3327: keep its helpers ldsm_x2 / ldsm_x4 / b12x_esx, the NN = NT/8 n8 tiles per decoded A fragment, the producer's escape-x gather after the stream and the RP-padded epilogue); the cluster launch (cudaLaunchKernelEx + cudaLaunchAttributeClusterDimension) and the mapa helper come from master's section (7) ouns_kernel. Changes vs b12x: (a) cluster dims (4,1,1), grid N/R unchanged (pooler R=32: 128 CTAs = 32 clusters; qkv_gate R=16: 144 CTAs = 36 clusters); the chunk rotation becomes per cluster, jc = (j + blockIdx.x / 4) % NCHUNK, identical in the 4 ranks (each CTA still streams its own weight chunk jc with the existing bulk_g2s). (b) x is no longer copied as 16 B cp.async pieces by the producer lanes: the host encodes a 2D CUtensorMap over x (cuTensorMapEncodeTiled fetched with cudaGetDriverEntryPoint, so no -lcuda; BFLOAT16, globalDim {K, m}, globalStride {2K} B, boxDim {64, NT}, SWIZZLE_128B, L2 promotion 128B, OOB fill none) passed as const __grid_constant__ CUtensorMap. A chunk's x segment is CKK/64 boxes (pooler 4, qkv 8; 64 k x NT rows, 1 KB at NT=8). With MC=true, producer lane 0 of cluster rank r issues the boxes b with b % 4 == r as cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [xslot + b*NT*128], [tmap, {jc*CKK + 64b, 0}], [full_s], ctaMask 0xF: the same CTA-relative offsets are written in all 4 CTAs and each CTA's full_s gets the bytes. MC=false is the control arm: the CTA issues all CKK/64 boxes itself without .multicast, everything else identical. (c) Slot layout: SLOT = WB + NT*CKK*2 (pooler NT=8 16384 B, NT=16 20480 B; qkv NT=8 20480 B); the ring offset is rounded to 1024 B (b12x_ring_off rounds to 128 today) so each x box is 1024 B aligned as SWIZZLE_128B requires. The consumers' ldmatrix row address for token tok and chunk-local k = kk16*16 + khalf*8 becomes xslot + (k/64)*NT*128 + tok*128 + ((((k%64)/8) ^ (tok & 7)) * 16), replacing the XS-padded address (conflict-free through the swizzle). (d) Barriers: full_s init count 1, armed by the local lane 0 with mbar_expect_tx(full_s, WB + NT*CKK*2) before it issues anything into slot s; a peer's multicast bytes may land before the local arm, which is legal because the phase cannot complete before the local arrival (CUTLASS's multicast pipeline relies on the same); the cp.async.mbarrier.arrive.noinc path and the 1 + 32 count are removed. empty_s becomes cluster-wide: init count CW*4; after a chunk each consumer warp's lane 0 arrives on slot s's empty barrier of every rank q = 0..3 with mbarrier.arrive.release.cluster.shared::cluster.b64 _, [mapa(empty_s, q)]; the producer waits its local empty_s with mbarrier.try_wait.parity.acquire.cluster, so no rank multicasts into slot s before all 4 ranks' consumers freed it. Every wait stays bounded (B12_WAIT_NS, error word). (e) Start: after mbar_init and fence.mbarrier_init.release.cluster, barrier.cluster.arrive.release + barrier.cluster.wait.acquire (all threads) replaces the first __syncthreads, so no peer signals an uninitialised barrier; end: one more cluster arrive/wait after the epilogue store, so no CTA exits while a peer may still arrive on its barriers. Instances (6): pooler <32, 8192, 8, 1, 8, NT, MC> at NT = 8 and 16, qkv_gate <16, 8192, 4, 2, 4, 8, MC>, each with MC true and false. A case whose cudaOccupancyMaxActiveClusters is below its cluster count (32 pooler, 36 qkv) prints skipped. (2) HARNESS: _mcx_diag(state), called last in prepare() after the occupancy re-print and before the FINAL FLAGS line, skipped (VERDICT n/a: budget) when prepare has already run > 200 s. Per case (pooler m=8, pooler m=16, qkv_gate m=8): encode state[layer] with master's _b12_encode at the instance's R (the encoding b12x shared with the m=1 cell) and check it bit-exact with master's b12_unpack; run _cell_check (fp64 reference + canary) on mc and uc, and require mc and uc outputs bit-identical; any failure marks the case INVALID. Timing copies _cell_gate's replay loop without a routing flag: arms inc = kernel_fn(data, state) on the final state (master's routed incumbent for that layer and m), mc and uc, each captured with _cell_capture on the same synthetic x; the CELL_FLUSH_BYTES (256 MB) READ flush (flush_i32.sum(), as the judge reads) before every replay; CELL_GATE_REPS (41) reps in rotating interleaved order; empty-graph baseline subtracted; 2 rounds. Print per round each arm's net us and IQR, mc/inc and mc/uc, plus a reference arm kernel_fn at pooler m=1 (master's b12 cell). Then one _decision line, so it is re-printed in the 16 KB log tail: [mcx] VERDICT pooler m8 mc/inc=<worst round> mc/uc=<..> | pooler m16 mc/inc=<..> | qkv m8 mc/inc=<..> -> LEAD|SHELF|REFUTE|INVALID, per the decision rule. Free every diag tensor, graph and tensor map and call torch.cuda.empty_cache(). Churn runs it as a check and never submits it.

## Expected

- `*` unchanged: prepare-only diagnostic: kernel_fn and routing are byte-identical to master, so the check should score the 1.2794 board within the same-bytes spread (~+-1.2%); it is never submitted
- `layer=pooler load=verify_1 m=8` faster: the lever, not this run: incumbent 19.0 us judged (triage of 3e3ffc5a, elasticity 6.6%); at the m=1 bulk12 judge ratio 0.84 it would be ~16.0 us. Diag target: mc/inc <= 0.85 under the read flush
- `layer=pooler load=decode_16 m=16` faster: lever: 19.4 us, elasticity 5.1%; the diag's pooler m=16 case (NN=2 n8 tiles per fragment) predicts it
- `layer=pooler load=decode_32 m=32` faster: lever: 19.9 us, elasticity 5.2%; extrapolated from m=8 / m=16, not measured by the diag
- `layer=unpooler load=decode_16 m=16` faster: lever: 18.6 us, elasticity 4.9% (R=64, K=4096: 2 boxes per chunk); proxied by pooler m=16
- `layer=unpooler load=decode_32 m=32` faster: lever: 18.3 us, elasticity 4.8%. Gain arithmetic: elasticities 6.6+5.1+5.2+4.9+4.8 = 26.6% x ln(1/0.84) = 0.174 -> +4.6% headline, discounted to ~4.2% because m=16/32 decode 2-4 n8 tiles per fragment and pay more x per chunk; qkv m=8 (4.4%) would add ~0.8% on top if its ratio matched

## Value case

- **Target:** diag: a prepare-only judge diagnostic that sizes a lever.
- **Evidence:** b12x at m>=8 lost on the judge: qkv m=8 x0.769 (12.86 -> 15.63 us, check 6617f94d; gate 0.976 under the old write flush). After the cp.async fix ~1 us per token row remained (0acdca48: pooler m=8/16/32 25.3/34.9/52.8 vs incumbent 23.4-24.1 net us), read as L2 hot-spotting (every CTA reads the same x lines); per-CTA chunk rotation cut qkv m=8 17.1 -> 14.5 net us (2980f57a), confirming part of it (FINDINGS bulk12; DISTILL:179). Resident x (b12r) judged pooler m=8 only x1.016, 19.07 -> 18.68 us (6617f94d). bulk12 at m=1 judges 0.84 of its incumbent (pooler 17.7 -> 14.8 us, 3834ceec). Triage of 3e3ffc5a: pooler v1 m=8 19.0 us (6.6%), pooler d16/d32 19.4/19.9 us (5.1/5.2%), unpooler d16/d32 18.6/18.3 us (4.9/4.8%).
- **Gain if it works:** 4.2% headline; **p** 0.25; **EV** 1.05% (bar: gain >= 3.0%, EV >= 1.5%); **below the bar** (EV 1.05% < 1.5%), minted because the user described it.
- **Cost:** 250k tokens (CUDA cluster/TMA-multicast build + Opus audit), 1 judge run.
- **Decision rule:** Read the WORST of the 2 read-flush rounds of mc/inc at pooler m=8 (r8) and pooler m=16 (r16), and mc/uc at pooler m=8. Checked in this order. INVALID (build, occupancy, round-trip, _cell_check or mc-vs-uc bit-identity failure): fix and rerun the diag once; no claim. REFUTE: r8 >= 1.0, or mc/uc >= 0.97 at pooler m=8 (multicast does not remove the per-row cost, so L2 hot-spotting is not the bottleneck) -> CLAIMS [refuted] multicast x at m>=8; b12x stays closed. LEAD: r8 <= 0.85 and r16 <= 0.92 -> mint the check b12mc pooler m=8/16/32 + unpooler m=16/32 (+ qkv m=8 if its mc/inc <= 0.90), each cell routed only at a read-flush gate <= 0.95 (B12_MAX_RATIO_M8), with CLAIMS [lead] reopening b12x. SHELF: anything else (0.85 < r8 < 1.0, or r8 <= 0.85 with r16 > 0.92, where the m=8 cells alone are worth ~1.2-1.9%, under the bar) -> CLAIMS [shelf], minted only inside a bigger bf16 stack.
- **Instrument:** prepare-time graph-replay timer copied from _cell_gate (no routing flag): 256 MB READ flush (flush_i32.sum(), as the judge reads) before every replay, 41 interleaved reps x 2 rounds, empty-graph baseline subtracted; arms inc = kernel_fn on master's final state, mc (multicast x), uc (same kernel, per-CTA TMA x, the multicast control), plus pooler m=1 as reference; correctness by _cell_check (fp64) and mc==uc bit-identity; the VERDICT goes through _decision into the 16 KB log tail; the judge's own rows of the unchanged timed path are the board-noise control

## Risk

Risk: cluster-wide empty barriers couple the 4 ranks, so the slowest CTA paces its cluster (the uc arm keeps that coupling, so mc/uc isolates multicast). The read-flush gate has under-predicted the judge at m=8 before (b12r gate 0.938 vs judge 0.984): a LEAD needs the judge check to confirm. The future check must build the tensor map per call (x's address varies) on the host, inside graph capture; this diag uses a fixed x buffer. Compile adds ~30-60 s to the 10-minute run budget.

## Adversarial review

Reviewed 2026-10-29T151743Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes: CLAIMS.md:50 [refuted] b12x names exactly this as its reopen (x multicast once per 2-4 CTA cluster, read-flush gate <= 0.95); board unmoved (a36f3392, 1.2794). Buildable, but as written the run is probably unreadable: the pooler cases (r8, r16) very likely cannot co-schedule 32 four-CTA clusters at 1 CTA/SM on the judge, and the rule checks REFUTE (mc/uc) before LEAD, killing a real win from the 2D TMA boxes alone. Fix both in the build; no extra runs.

### Justification
Plausible for qkv, weakly backed for pooler. b12 at m=1 judges 0.84 on pooler (17.7 -> 14.8 us, 3834ceec); the m=8 incumbent is swap-AB Triton at 19.0 us (board 3e3ffc5a, floor 14.0). WEAKEST LINK: the L2 hot-spotting story. Per-CTA rotation cut readers per x line from 128 to 4 and moved qkv m=8 17.1 -> 14.5 us but pooler m=8 only 25.3 -> 24.5 (0acdca48 -> 2980f57a), so most of pooler's ~5.8 us excess is not hot-spotting. The likelier lever is replacing 256 16-B cp.async pieces per chunk with tensor-map boxes, which BOTH mc and uc do. mc/uc also confounds fewer TMA issues (uc 128/CTA vs mc 32/CTA; DISTILL:263 30-50 ns each) with fewer L2 reads. m=32 rows (10 of the 26.6 elasticity points) are extrapolated and hit DISTILL 3.4's N>=32 mma.sync trap (b12x pooler m=32 55.6 vs 24.8 us); without them gain ~2.9%. Risk slip: b12r 0.938 vs 0.984 was a WRITE-flush gate; no read-flush gate-vs-judge pair exists at m>=8; at m=1 read flush under-predicted qkv/o by 4-5 points. DESIGN CHECK (must fix): pooler <32,8192,8,1> is 1 CTA/SM, 128 CTAs, cluster 4, needs 32 co-resident clusters; judge reports MaxActiveClusters 62 for cl=4 at 2/SM (ouns) and 30 for cl=8 at 2/SM (router), so 1/SM gives <= ~31 four-CTA clusters: pooler m=8/16 would print 'skipped' and the run ends INVALID. qkv (MINB=2, 36 clusters vs ~62) is fine. Barrier/box alignment sound (CUTLASS multicast pipeline pattern; SWIZZLE_128B limits inner box to 64 bf16, so 1-2 boxes per rank).

### Confounders on the current kernel
- swap-AB Triton at m<=32 (bf16-m32-swapab-wgmma-triton / bf16-restack-m16-swapab-router-swapab-m32): the inc comparator, not a confounder; if the shelved tile repack (CLAIMS.md:64) lands the bar moves.
- bulk12 m=1 landing (5325ea40, CLAIMS.md:56, :59): diag re-encodes pooler R=32 and qkv R=16, which master already holds; reuse state's codes when R matches (saves prepare time and ~80 MB).
- kernel_fn and routing byte-identical: board rows are the noise control.

### Interference and A/B needs
No companion run. All comparisons are interleaved in one process under the same read flush. Fix in the build: (1) co-residency: size a pooler instance per NT for 2 CTAs/SM (S=5-6 at NT=8, <= ~113 KB dynamic smem), MaxActiveClusters(4) ~62 >= 32; keep S=8 MINB=1 only if its query reports >= 32; print every instance's MaxActiveClusters in VERDICT. (2) Decision: REFUTE only if min(mc,uc)/inc >= 1.0 (or > 0.95); base LEAD/SHELF on best-arm/inc (b8, b16) and name the winning arm; keep mc/uc as attribution worded 'multicast (fewer L2 reads and 4x fewer TMA issues per CTA) vs per-CTA boxes'. (3) Optional: bridge arm with a7da6fa9's rotated b12x (cp.async x) at pooler m=8 to tie to 2980f57a.

### Open questions
- Judge MaxActiveClusters for 4-CTA at 1 CTA/SM with ~140-170 KB smem is likely <= 31 < 32; choose the 2/SM pooler instance.
- LEAD on the better of mc and uc; uc <= 0.85 means the lever is tensor-map delivery, not multicast.
- Add a pooler m=32 case (NT=32, S<=6) or drop pooler d32 / unpooler d32 from a LEAD's mint.
- VERDICT line must carry both rounds of mc/inc, uc/inc, mc/uc, net us and IQR per arm, MaxActiveClusters per instance, pooler m=1 reference (16 KB tail).
- Correct the Risk note (b12r 0.938 was write-flush); qkv m=8 threshold ~0.86-0.88.

