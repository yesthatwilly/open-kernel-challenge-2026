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
review: proceed
status: running
review_at: 2026-10-25T212521Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T212552Z
base_sha: b3b8b66c3106acbf5b6caf1c61d24bda67cd95d5
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/full-verify-wgmma-ws-s1-producer-split
worktree: .worktrees/muse_attention_decode/full-verify-wgmma-ws-s1-producer-split
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T011237Z
impl_commit: 198089eb208ddc0a0de64f2977173876834c57d0
submission_sha256: 25c1ac890e3b7d4fd618b7584e444f5aa865e2b41e7f80369d36069199a1acaa
norm_sha256: 9d9bde6a809544ca479046ad510696bb8fe5187c305360b44da7f47ec59ca75d
submission_blob: 680eba9fc5edd19dfbbeae4c15929ab6eca23fce
lint_warnings: 82
attempt_file: 2026-10-26T012538Z-check-full-verify-wgmma-ws-s1-producer-split.md
prepared_at: 2026-10-26T012538Z
job: e701c51d7b944a13
dispatched_at: 2026-10-26T012615Z
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

## Adversarial review

Reviewed 2026-10-25T212521Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound rewrite stage aimed at the problem's largest gap (full verify_16/32 at 2.2 TB/s, 2.3x floor; DIAGNOSIS action 1). The gate makes it fail-safe. Its weakest link is producer width at 1 CTA/SM, which looks like the BULK=0 layout of full-cuda-onelaunch-warpspec-bulk-gather (1.77 TB/s). Before dispatch, add DIAGNOSIS's gather-only roofline DIAG and its 2-CTA/SM per-warpgroup-ring variant as in-run variants, fix the TILE=128 smem arithmetic, and move the gate to LESSONS' two-round rule.

### Justification
Mechanism: plausible. The profile backs every number. Full verify_16 is 162.5 us (FA3 148.7 + combine 6.4 + prepare 3.1 + norm-gate + pre-attn) against a 63.4 us floor. Verify_32 is 302.0 us (FA3 288.7) against 126.8 us. Master's CUDA stream-K decode kernel moves the same 256 B/token gather at 3.0-3.07 TB/s (decode_16 108.2 us), so decode-level bandwidth on verify (about 110/225 us) is a demonstrated ceiling, not a guess. s5 is correct (19 bisect cases, two-state replays). Its 0.570x/0.704x with 1.49x time for 2x bytes says it is latency-bound in a serial per-tile chain. Fixing that chain (S(n+1) issued before softmax(n), wait_group 1) is the right target. The compute fits (about 21 us of FP8 wgmma, about 40 us of MUFU at B=16) if it overlaps with a 108 us gather. q_len 8 x G 8 = 64 rows per kv head maps exactly onto m64 wgmma, one consumer warpgroup per kv head. The register budget fits (232x256 + 40x128 = 64,512). Expected shapes and directions match TASK: only full verify_16/32 move, decode, sliding and full verify_1 stay flat, and the milestones (M2 <=135/<=255 us, M3 <=95/<=190 us) sit between the incumbent and 1.5x floor. Weakest link: producer issue width and TLP at 1 CTA/SM. s5 ran 2 CTAs/SM with 512 issuing threads per SM, and master decode also has about 512 threads per SM in flight. This design has 128 issuers and 8 consumer warps per SM. That is the shape of full-cuda-onelaunch-warpspec-bulk-gather (1 CTA/SM, 8-stage ring, warp-specialized producer), whose BULK=0 per-thread 16 B cp.async variant capped at 1.77 TB/s with one producer warp. 4 producer warps instead of 1 is a real difference: 7 x 16 KB in flight easily covers Little's law at 3 TB/s, and about 32 warp-instructions per tile is little issue pressure. But nothing has yet shown that 4 warps reach 3 TB/s on a random 128 B row gather with a dependent page-index load per row. If they do not, M1 fails for a reason the consumer-side PHASE DIAG may only show indirectly (as long FULL waits). Prior: the frontmatter has no `prior` field. This is not grounds for hold: the body names s5 and the full_ws lesson, and DIAGNOSIS names s5 as the prior. The mechanism is different from s5 (overlapped, warp-specialized pipeline versus a serial chain with a CTA-wide EMPTY rendezvous). It is different from the do-not-try onelaunch item (a 128-thread cp.async producer warpgroup, not single-warp cp.async.bulk per row). It is different from the mma.sync verify do-not-try (wgmma). The churn tooling should add `prior: full-verify-wgmma-s5-qfrag-reload`. Concrete defect: 'Ring STAGES = 8 (8 x 16 KB = 128 KB)' is a 64-token tile (K 8 KB + V 8 KB for both heads). A TILE=128 stage is 32 KB, so 8 stages need 256 KB, more than the 227 KB per CTA. The TILE=128 variant must say STAGES=4-5 (or count stages in 64-token half-tiles). Otherwise it fails to launch or the builder improvises. The local-bytes>0 refusal does not catch an smem overflow, so the variant also needs an smem-size refusal. The DIAGNOSIS additions are not in the change text: (a) a gather-only roofline DIAG (same ring and producer, no MMA) and (b) a 2-CTA/SM variant that keeps s5's all-threads-issue layout, splits the ring per warpgroup and overlaps S(n+1). Add both as template variants in the same module. They answer the weakest link inside this one run.

### Confounders on the current kernel
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / paired-heads-launch-cleanup-ab-tune-batch-gate: these define the incumbent (paired FA3 + prepare + combine + _paired_norm_gate_kernel at full verify B>1). They do not confound the measured kernel, but the prepare gate must time exactly master's paired path, including the paired pre-attention q layout and the norm-gate, or the >=1.05x routing decision is mis-anchored. When the wgmma path routes, the pre-attention kernel's q layout must be the one the wgmma kernel expects, not the paired-padded layout.
- fused-preattention-triton: the wgmma path keeps _pre_attention_kernel as its prologue. If the routed band changes the PAIRED constexpr passed to it, verify_16/32 pre-attention time (2.7/4.1 us) can move a few us either way and be misread as kernel gain or loss. Read the replaced kernels' own top_kernels, not the shape total.
- full-decode-cuda-gather-splitkv / full-decode-splitkv-s2 stream-K (master decode kernel): it shares the load_inline import clock (222 s across 4 modules) and its ticket arrays. A new module adds compile time and a self-check, but there is no timing interaction on decode shapes, because they are untouched and gated by config.

### Interference and A/B needs
No companions. This is kind: rewrite, judged on the replaced kernels' own us at verify_16/32 against 162.5/302.0 and the floor, and the hard prepare gate makes the decision fail-safe (no route means master is re-measured). No landed hypothesis is a plausible cause of a move on those rows except through the incumbent definition, and the gate already times that path. The real single-run ambiguity is why it fails to route: producer issue width at 1 CTA/SM, or consumer latency at 1 CTA/SM. Separate checks are the wrong fix. Resolve it inside the run by building (a) DIAGNOSIS's gather-only roofline instantiation (producer plus ring, consumers just wait FULL and arrive EMPTY; time it at B=16 x 80K and print achieved TB/s) and (b) the 2-CTA/SM per-warpgroup-ring overlapped variant, as extra template instantiations in the same muse_full_verify_wgmma_ws_v1 module. Both are timed in the same interleaved harness, and either may route. The extra compile cost is one module. The next stage is the follow-up.

### Open questions
- TILE=128: what is STAGES? 8 x 32 KB does not fit in 227 KB. Specify 4 (or 5 with no other large smem) and add a refusal when dynamic smem exceeds the device limit, alongside local bytes > 0.
- Does the 128-thread producer warpgroup sustain >= 2.8 TB/s on its own? The gather-only DIAG (DIAGNOSIS 1a) should answer this before the consumers are blamed. The BULK=0 single-warp precedent was 1.77 TB/s.
- Page-index prefetch: how far ahead does the producer load page indices relative to the K/V issue? A dependent int32 load per row on the producer's critical path can cap issue rate more than the width can.
- cp.async.mbarrier.arrive.noinc with 128 producer arrivals per FULL stage, plus EMPTY with 256 consumer arrivals: are expected counts initialised once and phase bits tracked per stage across stream-K segment boundaries (s4/s5's bug class was per-segment state)?
- With two S accumulator sets at TILE=128 (2 x 64 regs) plus O (32) plus softmax temps, does the consumer stay under 232 regs with no spill? Print it per variant, as specified.
- Should the gate also offer the 2-CTA/SM per-warpgroup-ring variant (DIAGNOSIS 1b), so that a 1-CTA/SM TLP loss does not leave the stage with nothing routed?


## Implementation notes

Edited only `muse_attention_decode/submission.py` in the worktree (+1733/-1). New load_inline module `muse_full_verify_wgmma_ws_v1`, kernel symbols `wsv1_*`, python names `_ws_*`/`_wsv_*`/`_WS_*`, own ticket array and error word (nothing shared with the sibling split-ring-2cta build). The s5 math was ported verbatim (SW128 ring, fp8 QK wgmma with per-tile Q A-fragment reload from smem, Max_offset-8 FP8 P, per-WG V^T transpose, ticket combine + norm-gate, stream-K, bounded mbarrier waits). Only the pipeline changed.

Variants (one module, runtime GO flag on all): 0 ws64 = 384 thr, 1 CTA/SM, WG0 producer (128 thr, 8 cp.async/thread/tile, FULL = 128 noinc arrivals, EMPTY = 256 consumer arrivals), WG1/2 consumers, 8 stages of 16 KB, two S sets (S(n+1) issued before softmax(n), PV(n-1) before it, waited with wait_group 1). 1 ws128 = TILE 128 (m64n128k32, 4 PV k-steps), STAGES=5 (160 KB ring, 205 KB smem, static_assert <= 227 KB), one S set. 2 c2 = 2 CTAs/SM, s5 layout, ring split per warpgroup, per-WG barriers (128 arrivals), no CTA rendezvous, 5 stages. 3 = ws64 PHASE DIAG (clock64 per consumer phase and producer phase, CTA 0), prepare only.

Review fixes applied: (1) gather-only roofline DIAG (`go` flag, all variants, printed TB/s at B=16 x 80K); (2) 2-CTA/SM per-warpgroup-ring variant, routable; (3) TILE=128 STAGES=5 with an smem static_assert plus host refusal when dynamic smem > device opt-in, plus refusal on local bytes > 0 / non-sm_90 image / occupancy; (4) two-round gate: master's ACTUALLY ROUTED path (kernel_fn, no variant forced, empty route), 2 interleaved rounds x 40 replays, 256 MB flush, empty-graph baseline subtracted, both rounds >= 1.05x, candidate IQR <= 1.5x master's, vs-master rel L2 < 3e-2, finite, error word 0; both rounds' medians/IQRs, kernel names and PATH CHOSEN printed per cell; compact summary repeated as the last lines.

Open questions answered: page-index prefetch = one 4-byte __ldg per tile, issued 4 tiles (a block) ahead in registers, never on the issue path (c2: one tile ahead). Barrier phases are tracked on the CTA-flat tile index n (FULL parity (n/STAGES)&1, EMPTY parity ((n/STAGES)-1)&1), not per segment. Register fit is printed per variant; a variant with local bytes > 0 is refused.

Deviations (documented): (a) producer setmaxnreg 56 / consumers 224 instead of 40 / 232 (128*56 + 256*224 = 64512) to avoid producer spills at 40; (b) ws128 and c2 use ONE S set with FA3 order (PV(n-1) overlaps softmax(n)): two 64-register S sets do not fit 224 at TILE 128 and 2 CTAs/SM caps c2 at 128 regs; (c) TICKET only (no B8 fallback kernel); (d) variants are timed after a quick canary check and the full bisect + two-state replay runs for the winning variant per band only; (e) deadlines (import clock, per the coordinator note against the 480 s start rule): build < 400 s, quick checks < 440, gate < 455, full checks < 465, DIAGs < 470, each prints "skipped: deadline". Verified only by a g++ -fsyntax-only pass with CUDA stubs (no nvcc/GPU here).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only a config-only predicate and torch.empty workspaces; no .item()/sync/prints in the call path (setup-only globals are final and None/frozen when timed).
- [x] Nothing derived from `data` is cached: only a zeroed-once self-resetting int32 ticket array and an error word, both device-only.
- [x] Weights: n/a (attention, no weights).
- [x] KV cache stays FP8 E4M3; query rounded to FP8 E4M3 by the existing pre-attention kernel; fp32 softmax/merges, bf16 out.
- [x] Every key attended (stream-K over all tiles; only the causal block mask), no skipping.
- [x] No subprocesses, network, harness access; arch list env restored after build.
- [x] All work inside the timed call; no output caching.
- [x] New path has its own per-variant route (no _ab_flags pattern exists in this problem); a miss falls back to master's exact paired FA3 path and touches no other flag.

okc_lint: `OK with warnings` (only host-sync warnings outside kernel_fn, as in master); py_compile OK; __pycache__ removed.

## Pre-flight audit

Verdict: dispatch-with-fixes. Audited by reading (no nvcc/GPU here); evidence: `/tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/ws/ws_audit_emu.py` (smem, registers, producer gather map, barrier-phase simulation, all pass) and the generated CUDA source `.../audits26/ws/wsv1.cu` (no unresolved template fields).

Checked and fine: (1) arrival counts: ws FULL 128 noinc (one per producer thread per tile), EMPTY 256 (both consumer WGs, once per tile after S retire + transpose); c2 per-WG FULL/EMPTY 128; parities on the CTA-flat tile index; simulated for N 1..257, no deadlock. smem: ws64 157344 B, ws128 206448 B, c2 108224 B, all <= 232448; c2 two CTAs 218496 B incl. 1 KB reserve <= 233472, 2 CTAs/SM also enforced by the occupancy check. Producer gather map covers every 16 B chunk once with s5's SW128 swizzle at TL 64 and 128. (4) Gate predicate is config-only: window 0, q_len 8, 1<batch<=64, Hq 16/Hkv 2 (dp is 128/16), head_dim 64, page_size 1, so page 64 and dp heads stay on master. Only `_IMPORT_T0` collides with the s6 sibling (identical meaning; trivial merge); module, kernel and python names are disjoint.

Fixes:
1. `_ws_gate_band` read the candidate and master outputs for the vs-master rel L2 / finiteness BEFORE any graph had been replayed (a captured graph has not run; the buffers are uninitialised), so the gate could never pass and nothing would route. Each graph is now replayed once and synchronized before those reads (s5 read them after the timing replays).
2. `_IMPORT_T0` was set after the torch/triton/sgl_kernel/sglang imports, so every start deadline ran late by the import time (tens of seconds) against the 480 s rule; moved to the first statement after `import time`.
3. IQR veto had no absolute floor (`max(master IQR, 1e-9)`); now `max(master IQR, 0.5 us)`.
4. setmaxnreg hang guard: 128*56 + 256*224 = 64512 only balances if the kernel launches with exactly 168 regs/thread (65536/384 rounded down to 8); with fewer, `setmaxnreg.inc` would block forever. Variants 0/1/3 are now refused (fallback, PATH CHOSEN line) when `numRegs < 168`.
5. Ticket buffer key mismatch: setup keyed `torch.device("cuda")` ("cuda") but kernel_fn keyed `qkv.device` ("cuda:0"), so two buffers existed and the setup re-zero hit the wrong one. Key normalised to cuda:N; tickets also re-zeroed after every self-check so a failed variant cannot dirty the next one.
6. A gate exception left `_WS_FORCE` set (kernel_fn would then force that variant for every verify shape at import end); `_ws_setup`'s finally now resets all setup-only globals.

Residual risk: no nvcc/GPU run, so compile time of four heavy instantiations, ptxas register/spill counts (producer at 56, consumers at 224) and the 128-thread producer's gather bandwidth are untested; the build has a start deadline (400 s) but no duration cap, so a slow compile can push the import past 480 s, though the later phases then skip cleanly to master.
