---
problem: muse_bf16_proj
slug: pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite
created: 2026-10-25T011408Z
base: master
area: pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel)
change: >
  Built on master (blob a8bb42a). WHY: pooler (n=4096, k=8192) and the full unpooler (n=8192,
  k=4096) are 67.1 MB pure weight streams at m<=32 with a 14.0-14.1 us floor; master runs them at
  19.0-20.7 us (pooler m=1/8/16 via _bf16_splitk_kernel tokens-on-M, m=32 via swap-AB) and
  18.1-18.5 (unpooler m=16 cuBLAS, m=32 swap-AB), i.e. ~3.4-3.6 TB/s. These are the m<=16 rows the
  1.2215 restack does NOT cover (it covers qkv, o and the unpooler slice; pooler was flat under
  swap-AB m16, LESSONS). Every attempt so far hit the same ~3.5 TB/s: Triton cp.async BLOCK_K 256
  and TMA tiles (pooler-splitk2-tma-weight-stream), the CUDA gemv with 2 CTAs/SM and a 6-stage
  cp.async 16 B ring (REWRITE_READ s2: 'the gap is fixed per-launch cost, not bytes in flight'),
  and all of them read each CTA's weight as 64 rows x 512 B segments at a 16 KB (pooler) / 8 KB
  (unpooler) row stride, then pay a split-K ticket tail. Untested: a stream where (a) every CTA
  reads ONE contiguous region of DRAM, (b) loads are issued as a few large bulk copies by one
  thread (no per-16 B instruction cost, no issue pressure on the consumer warps), and (c) there is
  NO split-K and no tail. STAGE 1: in prepare(), repack each of the two weights losslessly (bf16
  stays bf16) into a row-owner tiled layout Wt[c][kb][r][kk] with c = the owning CTA (pooler: 128
  CTAs x 32 rows; unpooler: 128 CTAs x 64 rows), kb = k/64 K-blocks, r = the CTA's rows, kk = 64,
  so each CTA's whole working set (512 KB for both layers) is one contiguous span, and inside it
  each [r x 64] K-block is one MMA-ready chunk (apply the ldmatrix-friendly XOR swizzle in the
  repack so smem needs no rearrangement). Kernel (load_inline CUDA, mma.sync only, valid on plain
  sm_90, so no arch hazard; still set TORCH_CUDA_ARCH_LIST=9.0a and -lineinfo per LESSONS): grid =
  128 CTAs (pooler: one CTA per 32-row unit; unpooler: one CTA per 64 rows = 2 units, 512 KB, so
  both layers stream 512 KB per CTA), 1 CTA/SM, 5 warps: warp 0 lane 0 is the producer, issuing
  cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes (1-D bulk copy) of 16 KB
  weight chunks (pooler 4 K-blocks = 256 of K; unpooler 2 K-blocks = 128 of K) plus, in the same
  slot and transaction count, the matching x chunk [m x chunk-K] bf16 from L2 (<= 16 KB at m=32),
  into a 6-slot ring of <= 32 KB slots (<= 192 KB smem, ~96 KB of weight in flight per SM;
  mbarrier full/empty pairs); warps 1-4 are consumers, each owning a quarter of each chunk's K,
  doing mma.sync m16n8k16 with the weight chunk as A (rows on M via ldmatrix) and x as B (tokens
  on N: NT = 8, 16 or 32 template instances; x is not staged whole, it rides in each slot as
  above). At the end the four consumer warps' partial fp32 accumulators (32 rows x NT per unit)
  are summed in smem in fixed order and stored once as bf16 to out[token, row]. No workspace, no
  ticket, no memset, one graph node. Route in kernel_fn before master's paths for these shapes:
  layer == pooler and 1 <= cfg m <= 32, or layer == unpooler with cfg n == 8192 and 9 <= cfg m <=
  32; the unpooler slice (n == 1024), qkv_gate, o, router and m >= 33 keep master; the m=7 pooler
  test takes the new path. Prepare: per-route self-check at m in {1, 7, 8, 16, 32} (pooler) and
  {9, 16, 32} (unpooler) vs x.double() @ W.double().T, rel L2 <= 3e-3, one-hot power-of-two probe
  row within 1e-4 (bit-exact column), finite; sync and print after each launch; DISABLED prints
  and fallback per layer. Print-only diagnostics: a PURE-READ variant of the same kernel (producer
  + mbarrier ring, consumers only XOR one word per chunk into a dummy store) timed cold-L2 with
  cudaEvents (flush 256 MB first), reported as TB/s, next to the full kernel's cold-L2 time at
  m=1/16/32 and master's path. MILESTONES: M1 correct + PATH CHOSEN; M2 pooler m=1 kernel <= 18.5
  us and unpooler m=16 <= 18.3 (incumbent 19.0 / 19.3); M3 pooler m=1/8/16 <= 16.5 us (>= 4.07
  TB/s). KILL GATE: if the pure-read diagnostic is below 3.8 TB/s cold, the contiguous bulk stream
  gives no headroom over the incumbents and the line is DEAD (record that H200's practical read
  rate for 67 MB is the ceiling).
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: incumbent row 19.0 (_bf16_splitk_kernel 19.0 at ~3.5 TB/s); floor 14.0; M3 <= 16.5
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: incumbent 19.14; floor 14.0
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: incumbent 19.85; floor 14.1
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: incumbent swap-AB 20.0 row / 20.7 kernel; floor 14.1
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: incumbent cuBLAS nvjet 64x16 19.3 us (row 18.61); floor 14.1
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: incumbent swap-AB 18.7 kernel (row 18.48); floor 14.1
  - shapes: layer=unpooler load=decode_1 m=1
    direction: flat
    why: slice (n=1024) keeps master
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: slice keeps master
  - shapes: layer=pooler load=verify_16 m=128
    direction: flat
    why: m >= 128 keeps cuBLAS
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: m >= 128 keeps cuBLAS
  - shapes: layer=unpooler load=verify_16 m=128
    direction: flat
    why: m >= 128 keeps cuBLAS
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: m >= 128 keeps cuBLAS
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: pooler m<=32 (_bf16_splitk_kernel m<=16 19.0-19.6 us, _bf16_swapab_kernel m=32 20.7 us) and unpooler-full m=16/32 (nvjet 64x16 19.3 us, _bf16_swapab_kernel 18.7 us)
review: proceed
status: done
review_at: 2026-10-25T012031Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T012032Z
base_sha: 0a2ef67eeb9be9dce45fed1c5dd5f4e4ba19537c
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite
worktree: .worktrees/muse_bf16_proj/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T013044Z
impl_commit: 8ba8624df9248fb986052e4a67ab8e9a96a9770d
submission_sha256: e3be58e99a878b9745064dffce48610469ac7a3514a4d4cb17fa29f220387e98
norm_sha256: 8351dff8dad39fc74cd23459227806d7ffece8ede107bd30fd917439f39f7225
submission_blob: d2ef0372de1f177e26ce1a1a4f578a993b4fa98c
lint_warnings: 9
attempt_file: 2026-10-25T013045Z-check-pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite.md
prepared_at: 2026-10-25T013045Z
job: cd39d36e31814087
dispatched_at: 2026-10-25T013053Z
outcome: regression
verdict_detail: regression
speedup: 1.1827696304198618
speedup_ci: [1.1650280859635638, 1.2005111748761597]
harvested_at: 2026-10-25T034134Z
best_at_harvest: 1.2215006063849365
promotion: null
---

# pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite

## Change

Built on master (blob a8bb42a). WHY: pooler (n=4096, k=8192) and the full unpooler (n=8192, k=4096) are 67.1 MB pure weight streams at m<=32 with a 14.0-14.1 us floor; master runs them at 19.0-20.7 us (pooler m=1/8/16 via _bf16_splitk_kernel tokens-on-M, m=32 via swap-AB) and 18.1-18.5 (unpooler m=16 cuBLAS, m=32 swap-AB), i.e. ~3.4-3.6 TB/s. These are the m<=16 rows the 1.2215 restack does NOT cover (it covers qkv, o and the unpooler slice; pooler was flat under swap-AB m16, LESSONS). Every attempt so far hit the same ~3.5 TB/s: Triton cp.async BLOCK_K 256 and TMA tiles (pooler-splitk2-tma-weight-stream), the CUDA gemv with 2 CTAs/SM and a 6-stage cp.async 16 B ring (REWRITE_READ s2: 'the gap is fixed per-launch cost, not bytes in flight'), and all of them read each CTA's weight as 64 rows x 512 B segments at a 16 KB (pooler) / 8 KB (unpooler) row stride, then pay a split-K ticket tail. Untested: a stream where (a) every CTA reads ONE contiguous region of DRAM, (b) loads are issued as a few large bulk copies by one thread (no per-16 B instruction cost, no issue pressure on the consumer warps), and (c) there is NO split-K and no tail. STAGE 1: in prepare(), repack each of the two weights losslessly (bf16 stays bf16) into a row-owner tiled layout Wt[c][kb][r][kk] with c = the owning CTA (pooler: 128 CTAs x 32 rows; unpooler: 128 CTAs x 64 rows), kb = k/64 K-blocks, r = the CTA's rows, kk = 64, so each CTA's whole working set (512 KB for both layers) is one contiguous span, and inside it each [r x 64] K-block is one MMA-ready chunk (apply the ldmatrix-friendly XOR swizzle in the repack so smem needs no rearrangement). Kernel (load_inline CUDA, mma.sync only, valid on plain sm_90, so no arch hazard; still set TORCH_CUDA_ARCH_LIST=9.0a and -lineinfo per LESSONS): grid = 128 CTAs (pooler: one CTA per 32-row unit; unpooler: one CTA per 64 rows = 2 units, 512 KB, so both layers stream 512 KB per CTA), 1 CTA/SM, 5 warps: warp 0 lane 0 is the producer, issuing cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes (1-D bulk copy) of 16 KB weight chunks (pooler 4 K-blocks = 256 of K; unpooler 2 K-blocks = 128 of K) plus, in the same slot and transaction count, the matching x chunk [m x chunk-K] bf16 from L2 (<= 16 KB at m=32), into a 6-slot ring of <= 32 KB slots (<= 192 KB smem, ~96 KB of weight in flight per SM; mbarrier full/empty pairs); warps 1-4 are consumers, each owning a quarter of each chunk's K, doing mma.sync m16n8k16 with the weight chunk as A (rows on M via ldmatrix) and x as B (tokens on N: NT = 8, 16 or 32 template instances; x is not staged whole, it rides in each slot as above). At the end the four consumer warps' partial fp32 accumulators (32 rows x NT per unit) are summed in smem in fixed order and stored once as bf16 to out[token, row]. No workspace, no ticket, no memset, one graph node. Route in kernel_fn before master's paths for these shapes: layer == pooler and 1 <= cfg m <= 32, or layer == unpooler with cfg n == 8192 and 9 <= cfg m <= 32; the unpooler slice (n == 1024), qkv_gate, o, router and m >= 33 keep master; the m=7 pooler test takes the new path. Prepare: per-route self-check at m in {1, 7, 8, 16, 32} (pooler) and {9, 16, 32} (unpooler) vs x.double() @ W.double().T, rel L2 <= 3e-3, one-hot power-of-two probe row within 1e-4 (bit-exact column), finite; sync and print after each launch; DISABLED prints and fallback per layer. Print-only diagnostics: a PURE-READ variant of the same kernel (producer + mbarrier ring, consumers only XOR one word per chunk into a dummy store) timed cold-L2 with cudaEvents (flush 256 MB first), reported as TB/s, next to the full kernel's cold-L2 time at m=1/16/32 and master's path. MILESTONES: M1 correct + PATH CHOSEN; M2 pooler m=1 kernel <= 18.5 us and unpooler m=16 <= 18.3 (incumbent 19.0 / 19.3); M3 pooler m=1/8/16 <= 16.5 us (>= 4.07 TB/s). KILL GATE: if the pure-read diagnostic is below 3.8 TB/s cold, the contiguous bulk stream gives no headroom over the incumbents and the line is DEAD (record that H200's practical read rate for 67 MB is the ceiling).

## Expected

- `layer=pooler load=decode_1 m=1` faster: incumbent row 19.0 (_bf16_splitk_kernel 19.0 at ~3.5 TB/s); floor 14.0; M3 <= 16.5
- `layer=pooler load=verify_1 m=8` faster: incumbent 19.14; floor 14.0
- `layer=pooler load=decode_16 m=16` faster: incumbent 19.85; floor 14.1
- `layer=pooler load=decode_32 m=32` faster: incumbent swap-AB 20.0 row / 20.7 kernel; floor 14.1
- `layer=unpooler load=decode_16 m=16` faster: incumbent cuBLAS nvjet 64x16 19.3 us (row 18.61); floor 14.1
- `layer=unpooler load=decode_32 m=32` faster: incumbent swap-AB 18.7 kernel (row 18.48); floor 14.1
- `layer=unpooler load=decode_1 m=1` flat: slice (n=1024) keeps master
- `layer=unpooler load=verify_1 m=8` flat: slice keeps master
- `layer=pooler load=verify_16 m=128` flat: m >= 128 keeps cuBLAS
- `layer=pooler load=verify_32 m=256` flat: m >= 128 keeps cuBLAS
- `layer=unpooler load=verify_16 m=128` flat: m >= 128 keeps cuBLAS
- `layer=unpooler load=verify_32 m=256` flat: m >= 128 keeps cuBLAS
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

High. The decisive unknown is H200's practical cold read rate for a 67 MB contiguous stream; the pure-read diag answers it in the same run, so even a loss is informative and cheap to kill. Overlap: unpooler-full-m16-swapab-route (running) moves unpooler m=16 off cuBLAS onto swap-AB (~18.5); judge unpooler rows against that harvest too. x rides from L2 beside each weight chunk; at m=32 that is 64 MB of L2->SM traffic for pooler (as much as the weight), at m<=16 half or less; if m=32 lags, stage 2 multicasts x across a cluster. Print the slot geometry per instance. Bulk copies need 16 B alignment and sizes that are multiples of 16 (all chunks are). A fault in the self-check poisons the context (LESSONS): run as a check, sync+print per launch.

## Milestones

Rewrite stage 1 of `pooler m<=32 (_bf16_splitk_kernel m<=16 19.0-19.6 us, _bf16_swapab_kernel m=32 20.7 us) and unpooler-full m=16/32 (nvjet 64x16 19.3 us, _bf16_swapab_kernel 18.7 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T012031Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound, cheap-to-kill test of whether the ~3.5 TB/s ceiling on the 67 MB streams is structural: a contiguous per-CTA region, one-thread bulk copies, and no split-K tail. The in-run pure-read kill gate makes even a loss decisive. Fix the incumbent numbers to master's (run 1602a4e947e64090), and expect m=32 to lag, because each CTA re-reads ALL of x from L2. It needs no cluster/DSMEM machinery, so it does not have to wait for the in-flight cluster checks.

### Justification
Shapes: pooler (4096x8192) and unpooler-full (8192x4096) are pure 67.1 MB weight streams at m<=32, with floors of 14.0-14.1 us. Master 1602a4e947e64090 incumbents (use these, not the restack's): pooler m=1/8/16 _bf16_splitk_kernel 19.0/19.3/19.5 us kernel (rows 19.00/19.14/19.85); pooler m=32 _bf16_swapab_kernel 20.5 (row 20.00); unpooler m=16 nvjet 64x16 18.5 (row 18.61); unpooler m=32 _bf16_swapab_kernel 18.9 (row 18.48). The hypothesis quotes the restack's 19.3 / 18.7 / 20.7 for unpooler m=16, unpooler m=32 and pooler m=32. That makes its unpooler M2 (<= 18.3 against '19.3') nearly a tie with master's 18.5. Restate M2 as pooler m=1 <= 18.5 and unpooler m=16 <= 18.0. Also, the restack did route pooler m<=16 to swap-AB (18.6/19.7/19.6), so 'the rows the restack does not cover' is loose wording, not a substantive error. Mechanism: the three claimed differences are real and untested together here. (c) No split-K is exact for both layers: 128 CTAs x 32 rows = 4096 and 128 x 64 = 8192, with K whole per CTA, so there is no ticket, workspace or tail. That directly removes the m-proportional cost that REWRITE_READ section 2 found in the gemv. (b) A one-thread bulk copy frees the consumer warps from issue. (a) Contiguity is the weakest of the three: HBM address hashing already spreads 512 B segments across channels, so expect at most a small row-buffer-locality gain. The in-run pure-read diagnostic is the right arbiter. Bytes in flight (~96 KB/SM x 128 ~ 12 MB) far exceed Little's-law needs (~5 MB at 4.8 TB/s x ~1 us), so depth is not the risk. Physics: practical large-stream read bandwidth on H200 is roughly 85-90% of 4.8 TB/s. With ~1-1.5 us of launch-to-first-byte ramp inside a ~16 us kernel, M3 (<= 16.5 us, 4.07 TB/s effective) is at the edge of physics; M2 (<= 18.5) is reachable. WEAKEST LINK: x traffic. Each CTA owns full K, so each CTA streams ALL of x through its ring: per CTA, m x 8192 x 2 B for pooler (512 KB at m=32, equal to its weight). Pooler's total L2->SM traffic is 67 MB of weight plus 128 x 512 KB = 67 MB of x at m=32 (twice the incumbent split-K=2's x traffic) and 33.5 MB at m=16. With L2->SM bandwidth in the ~5.5-7 TB/s class, m=32 pooler is likely L2-bound around ~19-24 us and m=16 marginal. m=1/8 (<= 16.8 MB of x) is where the mechanism is cleanly tested. The hypothesis already flags this and names multicast as stage 2, which is correct. A mechanical detail the change paragraph glosses: an x chunk [m x chunk-K] is NOT contiguous in global (row stride 16 KB pooler / 8 KB unpooler), so a 1-D bulk copy needs m separate row copies of 512 B / 256 B per slot, or a 2-D tensor map. Columns >= m in the slot are never written; that is harmless only because MMA columns are independent and never stored, so say so in the code, or zero-fill once at kernel start.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (landed): it is the incumbent on pooler m=32 and unpooler m=32. The new route runs before it, so there is no runtime interaction. Its tile is SPLIT_K=2, so it reads half the x traffic per CTA that this design does; an m=32 loss here is the x re-read, not the bulk stream. Read m=1/8 to judge the mechanism.
- bf16-shape-gated-triton-winners-combo (landed): owns pooler m<=16 (_bf16_splitk_kernel). It is the incumbent and is bypassed by the gate, so it does not confound. Keep the new gate on its own bound constants so the combo's m<=16 bounds cannot widen the route by accident (LESSONS stack lesson).
- sglang-gemm-dispatch-parity (landed): the unpooler m=16 incumbent is its cuBLAS nvjet 64x16 path. cuBLAS rows vary between runs, so compare against 1602's 18.5 kernel and this run's own prepare-time cold-L2 side-by-side.
- unpooler-full-m16-swapab-route (running, not landed): moves the same unpooler m=16 shape. Judge unpooler m=16 against both master's nvjet (18.5) and that harvest; the two do not stack, so whichever wins owns the shape.

### Interference and A/B needs
No companion is needed, and none would be readable sooner than the in-run diagnostics. The pure-read variant (same producer and ring, trivial consumers) already isolates the stream from MMA/x cost, and it is the kill gate. One cheap print-only addition makes the result fully attributable without a second check: a pure-read variant that issues the incumbent's access pattern (512 B row segments at a 16 KB stride from the ORIGINAL weight, same ring and bulk-copy issue). If contiguous and strided pure-read rates match, claim (a) is dead and any gain comes from (b)/(c). If contiguous is faster, the repack is the lever. The two in-flight cluster/DSMEM rewrites test cluster launch, mapa and barrier.cluster, none of which this kernel uses. cp.async.bulk with a .shared::cluster destination is valid under a non-cluster launch (an implicit 1-CTA cluster) and needs only sm_90. There is no reason to wait for them.

### Open questions
- What cold pure-read rate does a 67 MB contiguous stream reach on the judge's H200? Below 3.8 TB/s means DEAD, as the hypothesis states. Remember that prepare event times undercounted the judge profile by 20-25% in the split-KV case (LESSONS), so a marginal 3.8-3.9 in prepare is likely 3.5 on the judge.
- Is m=32 L2->SM bound by the full-x re-read? Compare full-kernel cold times at m=1/16/32 against pure-read. A widening gap with m confirms it and makes x multicast (cluster of 2-4 sharing x) the stage-2 lever.
- Does one producer lane issuing 1 weight copy + m x-row copies per slot keep up at m=32 (33 bulk ops per 16 KB of weight)? If the full kernel lags pure-read even at m=8, split x issue across a second producer lane.


## Implementation notes

Edited only `muse_bf16_proj/submission.py` in the worktree (+568 lines, nothing removed): one new block before `prepare()`, a `_bulk_prepare(state)` call at the end of `prepare()`, and a route at the top of `kernel_fn`'s shape gate. No GPU or nvcc on this node, so none of it has been compiled or run.

**Kernel** (`load_inline`, name `okc_bf16_bulk_rowown_s1`, `-O3 -lineinfo -std=c++17`; plain sm_90 instructions only: mma.sync, ldmatrix, cp.async.bulk, mbarrier, so no 90a-only PTX and no `__CUDA_ARCH_FEAT_SM90_ALL` guard; an `#else` sets error word 3):
- Grid 128 CTAs, 1 CTA/SM, 160 threads. Pooler R=32 rows/CTA (K=8192), unpooler R=64 rows/CTA (K=4096); both stream 512 KB/CTA as 32 chunks of 16 KB.
- Warp 0 is the producer: lane 0 waits on the empty barrier and arms `expect_tx`; one lane issues the 16 KB weight copy and lanes 0..m-1 each issue one x-row copy (512 B pooler / 256 B unpooler). The x issue is spread across lanes already (the review's open question on a second producer lane). 6-stage ring, slot = 16 KB + NT x (CKK*2+16) B; max 199,936 B dynamic smem (pooler NT=32).
- Warps 1-4 are consumers, each owning a quarter of every chunk's K (k16 steps), `mma.sync.m16n8k16` with weight as A (ldmatrix x4, swizzled) and x as B (ldmatrix x2/x4 from the 16 B-padded row stride). Final: the four partials are summed in fixed order via smem (aliasing the quiescent ring) and stored once as bf16. No split-K, workspace, ticket, memset: one graph node.
- NT = 8/16/32 for pooler, 16/32 for unpooler (m <= 8 unpooler is the n=1024 slice and is not routed).
- Every mbarrier wait is bounded (globaltimer, 0.5 s) and sets error word 1 (producer) or 2 (consumer), then the warp leaves its loop; no hang.
- Copies are plain `cp.async.bulk...mbarrier::complete_tx::bytes`; no createpolicy / L2::cache_hint.

**Repack** (prepare): `Wt[cta][kblock][row][64 k]`, 16 B segments swizzled `seg ^ (row & 7)`. A pure permutation of bf16 values; prepare verifies the inverse is bit-exact (`torch.equal` on int16 views) and raises (route off) otherwise.

**Review fixes applied (the review wins over the spec):**
- Incumbents restated to master run 1602a4e947e64090: pooler m=1/8/16 splitk kernel 19.0/19.3/19.5 us, pooler m=32 swapab 20.5, unpooler m=16 nvjet 18.5, unpooler m=32 swapab 18.9 (the spec's 19.3 / 18.7 / 20.7 are the restack's). **M2 restated: pooler m=1 <= 18.5 us and unpooler m=16 <= 18.0 us** (spec said <= 18.3 against 19.3). M3 (pooler m=1/8/16 <= 16.5) unchanged.
- In-run pure-read kill gate kept: `BULK_READ_GATE_TBS = 3.8`, cold-L2, per layer; below it that layer's route is DISABLED and prints `KILL-GATE ... MISS`.
- Added the strided pure-read diagnostic: MODE 2 reads the ORIGINAL `[N][K]` weight as R row segments of CKK*2 B at stride K*2 B per chunk (512 B at 16 KB pooler, 256 B at 8 KB unpooler), same ring and bulk issue. The log line prints both rates and "repack is the lever" vs "repack is NOT the lever".
- m=32 expected to lag (each CTA reads all of x from L2). prepare() prints, per shape, bulk cold time, pure-read time, the gap (a gap widening with m confirms the x re-read), master's path time on the same cold-L2 harness, and the run-1602 kernel time. Pooler m in 1/8/16/32; unpooler m in 16/32.
- x chunk is not contiguous in global: done as m row copies per slot (not a 2-D tensor map). Columns >= m in the slot are never written; commented in the code as harmless because MMA columns are independent and never stored.
- Own extension name, own gate constants (`BULK_POOLER_MIN_M/MAX_M = 1/32`, `BULK_UNPOOLER_MIN_M/MAX_M = 9/32`), not derived from the combo's or swap-AB's bounds.
- `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` in try/finally and restored (popped if it was unset); the arch flags torch passes are printed.
- Self-check launches announced (`LAUNCH ... about to run`) before, `torch.cuda.synchronize()` + `DONE ... error word` after, for each of a random launch and a probe launch per m (pooler m in 1,7,8,16,32; unpooler 9,16,32). Probe: x row 0 one-hot 0.5, row must equal the weight column within 1e-4; random rel L2 <= 3e-3 vs `x.double() @ W.double().T`; finite; error word 0.

**Deviations / choices:** (1) The kill gate is on each layer's own pure-read rate; there is no extra gate against master's time, so a slower-than-master kernel still routes (keeps the result readable; the printed TIME lines show it). (2) Prepare timings are eager cold-L2 events (flush enqueued before start event) and undercount the judge profile by ~20-25% (LESSONS); read them to rank, not as the judged number. (3) The pure-read modes use the NT=8 instance (x unused).

**Biggest risks:** first-ever compile of this kernel (not compiled here: template/PTX typos are possible, which lands on master via `_bulk_build` -> DISABLED); mbarrier/bulk-copy semantics faulting in the self-check (a fault poisons the context, so run as `okc check`); x L2 re-read at m=32; contiguity may not be the lever (the strided pure-read says).

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite diff` (additions only, +568 lines in muse_bf16_proj/submission.py):

- [x] CUDA-graph capturable: `kernel_fn` reaches only `_bulk_route_ok` (cfg/shape/alignment checks) and `_bulk_forward` (one `torch.empty` for the output and one launch on the current stream); all sync, `.item()`, events and prints are in `_bulk_prepare` / `_bulk_median_us`, which `kernel_fn` cannot reach. The kernel's error flag is written on device, never read in the timed call. No allocation or control flow on input values.
- [x] Nothing derived from `data` is cached across calls: state holds only weight-derived tensors (repack), a device error word and a dummy buffer; `prepare()` self-check inputs are synthetic.
- [x] Weights: repacked losslessly in `prepare()` only; the bf16 permutation is verified bit-exact in `_bulk_repack`; no precision change (BF16 stays BF16, fp32 accumulation inside the kernel).
- [x] Activations BF16 where SGLang keeps BF16: x and output bf16, fp32 accumulate.
- [x] n/a: attention rule; this problem is dense projections and every key/row is computed (no skipping).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess calls or network: only `load_inline` build code; the source text avoids the lint's process/network words; `TORCH_CUDA_ARCH_LIST` is set and restored around the build.
- [x] All work happens inside the timed call (one kernel launch computes the full GEMM); no output caching or replay.
- [x] First line marker `# okc: problem=muse_bf16_proj` kept; no triton.autotune added.
- [x] Fallback: any build error, self-check miss, kill-gate miss or unsupported shape leaves `state["bulk_*"]` False and master's paths run; loud `PATH CHOSEN` lines per layer.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_bf16_proj/submission.py --problem muse_bf16_proj`): `OK with warnings`, 0 errors; the warnings are 8 `host-sync` lines (torch.cuda.synchronize / .item() in `_bulk_prepare` and `_bulk_median_us`, "outside kernel_fn's call graph; fine in prepare()"). `py_compile` clean, `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 (model opus) by reading only; this node has no GPU or nvcc. **Verdict: dispatch-with-fixes.**

Checked and found sound:
- mbarrier protocol: the full barrier's count is 1, armed by lane 0's `arrive.expect_tx` before any copy is issued. tx = 16384 + m*CKK*2 matches exactly the bytes MODE 0 issues, and 16384 matches MODE 1/2. The empty barrier's count is 4, one arrive per consumer warp. The producer starts at parity 1 and the consumers at 0, flipping every 6 chunks. Consumers wait on every full phase before `__syncthreads`, so the ring is quiescent before the scratch alias reuses it. Read-then-bulk-overwrite needs no `fence.proxy.async`. Every wait is bounded.
- Alignment: every slot stride is a multiple of 128 B and the x row stride is 528/272 B. Every bulk size and offset is a multiple of 16. Max smem is 199,936 B, under 227 KB.
- Fragments: the A ldmatrix.x4 matrix order and the `seg ^ (row&7)` swizzle match the repack, whose inverse prepare verifies bit-exact. The B ldmatrix x2/x4 and the C accumulator mapping match m16n8k16 row.col.
- Routing: bulk runs ahead of master's paths (pooler split-K and swap-AB; unpooler swap-AB and cuBLAS), as the spec says. The unpooler slice (n=1024, m<=8, so the m=5 test), m>=33, and the m=130 and m=1000 tests keep master.
- kernel_fn: one `torch.empty` and one launch, no sync, nothing cached from `data`. The arch list is restored in `finally`. There is no ptxVersion gate.

Fixes:
1. Changed `mov.u64 %0, %globaltimer;` to `%%globaltimer`. In operand-form inline asm a lone `%` is an operand escape, so nvcc would reject it. The whole extension would fail to build, and the route would be DISABLED for a wasted run.
2. Made the kill gate print-only. The spec calls the pure-read a print-only diagnostic, and the milestones are judged on the profile. Disabling a self-checked route because of a marginal eager cold-L2 event time would leave the run with no kernel in the profile. That event time includes event overhead and is ~20-25% off the judge. The KILL-GATE PASS/MISS line with its line-DEAD verdict still prints.
3. Widened the self-check to an interior m in every routed NT bucket (pooler 12 and 24, unpooler 24). Every m in 1..32 (pooler) and 9..32 (unpooler) routes, not only the ones that were self-checked.
4. Pure-read timings now print and reset the device error word, so a pure-read timeout is visible and does not read as a slow rate.

okc_lint: OK with warnings (prepare-only host syncs). py_compile clean, __pycache__ removed.

Residual risk: this is the extension's first-ever nvcc compile, so a template or PTX typo would leave master's paths running via DISABLED. Pooler m=32 will likely lag because each CTA re-reads all of x from L2.

## Result

- job `cd39d36e31814087` (check), status done, passed True, tests 9/9
- speedup 1.1828x, 95% CI [1.1650, 1.2005], score 4084.311835336145 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite.patch`; branch `hyp/muse_bf16_proj/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.18 | 13.62 | 0.9137 | lost |
| layer=o load=decode_1 m=1 | 62 | 5.89 | 6.62 | 0.8991 | lost |
| layer=pooler load=decode_1 m=1 | 62 | 18.84 | 19.71 | 0.943 | lost |
| layer=unpooler load=decode_1 m=1 | 62 | 6.15 | 6.39 | 0.9492 | lost |
| layer=router load=decode_1 m=1 | 62 | 5.38 | 5.99 | 0.8986 | lost |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.15 | 13.97 | 0.9308 | lost |
| layer=o load=decode_16 m=16 | 62 | 6.23 | 6.86 | 0.934 | lost |
| layer=pooler load=decode_16 m=16 | 62 | 19.47 | 19.94 | 0.9555 | lost |
| layer=unpooler load=decode_16 m=16 | 62 | 18.51 | 18.72 | 0.9857 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.47 | 5.97 | 0.9256 | lost |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.19 | 14.21 | 0.9916 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.34 | 6.74 | 0.982 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.73 | 19.85 | 0.9774 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.08 | 18.84 | 0.9776 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.2 | 6.5 | 0.9474 | lost |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.5 | 13.36 | 0.9348 | lost |
| layer=o load=verify_1 m=8 | 62 | 6.13 | 6.41 | 0.9703 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 18.89 | 19.19 | 0.961 | lost |
| layer=unpooler load=verify_1 m=8 | 62 | 6.0 | 5.98 | 1.0265 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.47 | 5.48 | 0.9864 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.05 | 16.85 | 1.0051 | flat |
| layer=o load=verify_16 m=128 | 62 | 6.98 | 7.31 | 0.9997 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.0 | 25.55 | 0.9971 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.96 | 19.73 | 0.9939 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.28 | 7.11 | 1.0051 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.69 | 20.28 | 0.9788 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.26 | 9.49 | 0.9964 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.19 | 29.83 | 0.9867 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.42 | 24.53 | 0.9942 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.6 | 9.37 | 1.0195 | flat |

lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16; pooler decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32. Expected hit: 8/15. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `pooler m<=32 (_bf16_splitk_kernel m<=16 19.0-19.6 us, _bf16_swapab_kernel m=32 20.7 us) and unpooler-full m=16/32 (nvjet 64x16 19.3 us, _bf16_swapab_kernel 18.7 us)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_bf16_proj "<what to fix or tune>" --kind rewrite --depends-on pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.

