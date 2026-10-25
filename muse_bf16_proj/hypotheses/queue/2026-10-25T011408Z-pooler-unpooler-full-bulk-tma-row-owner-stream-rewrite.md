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
status: queued
review_at: 2026-10-25T012031Z
review_companions: null
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

