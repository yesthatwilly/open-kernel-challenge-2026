# Check-quota reset plan (2026-10-25 UTC day)

Written 2026-10-24 ~17:15Z by churnsubs, while about 18 runs were still in the judge. CORRECTION: the
real limits are 20 checks and 10 scored submits per problem per day (okc problems). Order is the dispatch
order at the reset.

**Rule for every slot marked "after harvest":** read the run's per-shape rows and its log
lines first (build OK, self-check passed, path chosen). If it gained, restack it onto master
by a 3-way merge before anything else in that problem. A per-shape win that came back
neutral on the headline is still a restack candidate.

## Pre-built, dispatch at reset (all risk high, so checks only)

| # | problem | hypothesis | state | why first |
|---|---|---|---|---|
| 1 | attention | full-verify-cuda-streamk-splitkv | implemented | Rewrite stage 2. Extends the CUDA split-KV kernel (attention 1.293x -> 1.347x) to full verify_16/32, the two largest shapes. Est. 1.12-1.18x on those loads. |
| 2 | decode | full-decode-cuda-splitkv-port | implemented | Ports attention's CUDA kernel to decode's full decode_16/32 (est. +4% on those loads). |
| 3 | moe | cuda-nvfp4-swapab-mma-persistent-small-m | implemented | The big moe lever. CUDA W4A16 swap-AB replaces Marlin w13+w2 at bs<=32 (Marlin runs at 1.2-2.2 TB/s, floors 2.6-4x lower). |
| 4 | moe | cuda-fused-moe-bs1-persistent-w13-silu-w2 | implemented | One persistent kernel for all of bs=1 (target 20 us vs 48 us). |
| 5 | bf16 | pooler-unpooler-m128-tma-wspec-rewrite | implemented | The largest board gap, pooler at m=128 (11.4 us over its floor). |
| 6 | bf16 | bf16-small-m-cuda-gemv-rewrite | implemented | CUDA weight stream at m<=8 (Triton tops out at ~3.7 TB/s). |
| 7 | moe | marlin-two-stream-expert-halves-small-m | implemented | Overlap the bs=8..32 Marlin chains (expect flat to +2%). |

**Pre-flight audits (Opus, read-only), 2026-10-24.** All 7 builds are cleared to dispatch.
- moe swap-AB: had a blocker, its gate always chose Marlin. Fixed in f8b3f72.
- moe fused bs1: three fallback-safety fixes, in 0efab7b.
- The other four CUDA builds: dispatch as-is.
- marlin-two-stream: Python streams only, not audited.

Also pre-building nine newly reviewed stage-1 rewrites under the 40% rule; each is audited when its build lands:
- attention: sliding cluster, full one-launch
- moe: wgmma at bs>=128
- decode: MoE chain at T>32, small-T PDL
- prefill: wgmma experts, FP8 attention
- bf16: lossless compression, CUTLASS stream-K

## Final dispatch list at the reset (all built, all audited by Opus, 2026-10-24 ~18:20Z)

16 high-risk checks, all with a fallback to master and loud PATH CHOSEN log lines. Read those lines before judging any shape.

| problem | check | build | audit result |
|---|---|---|---|
| attention | full-verify-cuda-streamk-splitkv | ee7dd20 | as-is (emulated A8/B8) |
| attention | sliding-cuda-cluster-dsmem-onelaunch | 593859d | fixed: cache byte-match gate always failed |
| attention | full-cuda-onelaunch-warpspec-bulk-gather | fa289e5 | fixed: fail fast, cache byte-match gate |
| bf16 | pooler-unpooler-m128-tma-wspec-rewrite | fd7d7ee | as-is (warp_specialize may hang) |
| bf16 | bf16-small-m-cuda-gemv-rewrite | 6fa230c | as-is |
| bf16 | bf16-lossless-exp4-weight-stream-rewrite | 7895e56 | as-is (codec bit-exact) |
| bf16 | bf16-verify-streamk-cutlass-wgmma-rewrite | 08a0e62 | +16 B pointer guard |
| decode | full-decode-cuda-splitkv-port | f6cc145 | as-is (byte-identical port) |
| decode | moe-chain-cuda-swapab-expert-once-large-t | e65eda8 | as-is |
| decode | small-t-launch-minimal-dense-chain-pdl | d1f9835 | fixed: xn bound always failed |
| moe | cuda-nvfp4-swapab-mma-persistent-small-m | f8b3f72 | fixed: align-capacity gate always failed |
| moe | cuda-fused-moe-bs1-persistent-w13-silu-w2 | 0efab7b | fixed: R=0 Marlin, spins, alignment |
| moe | cuda-nvfp4-wgmma-rega-grouped-verify | f50b333 | fixed: prepare overrun cascade, setmaxnreg |
| moe | marlin-two-stream-expert-halves-small-m | (implemented) | not audited (Python streams) |
| prefill | experts-cuda-wgmma-mixed-input-grouped-gemm | 746b976 | fixed: zero-store gate, producer drain |
| prefill | attn-full-prefill-cuda-fp8-exp-offload | 5e83a89 | +abs cap on the pass rule |

After these harvest:
- attention: sliding-verify-cuda-streamk-window and full-verify1-cuda-streamk-batch1, only if the stream-K parent compiled and self-checked.
- decode: full-verify-cuda-splitkv-stage2, only if the port compiled.
- moe: cuda-fused-moe-midm-single-launch-routed, only after the swap-AB sibling harvests; it reuses its mainloop.

Six audits found a gate that would have switched a correct kernel off, or a cascade that would have turned a fallback into a failed run. All were fixed before dispatch.

## Harvest results (2026-10-24 ~18:00Z, per-shape analysis by an Opus agent)
- attention: no restack. Drop full-fused-decode1 (lost 5% on its target); fp8-qk-mma was flat. Go straight to item 2.
- bf16: restack m16-swapab + router-swapab (router gated to m<=32), expected ~1.215-1.222. Drop cuBLASLt (it lost at m=256), and keep prefill's dense-gemm-cublaslt held.
- decode: restack sliding-port + paired-pad + small-T GEMMs (GEMMs gated to T==1), expected ~1.18. fused-preattention-paired-pad-restack can follow.
- moe: align-parallel (check 1.325) is a gain on current master: hand-merge and promote. w2-l2-prefetch is dead.
- prefill: the packed-dequant condition is NOT met (ON but flat). Stages 6/7 and moe's swapab-nvfp4-gemm-verify-rewrite stay held.

## After harvest (as planned before the harvest)

### attention (10 checks)
1. Restack the winners among the four scored runs onto master:
   - sliding-deep-splits-on-cuda-splitkv
   - full-fused-triton-decode1-gate
   - sliding-fused-triton-decode1-splits-32
   - full-decode-cuda-fp8-qk-mma

   They touch disjoint shapes, so they can be combined.
2. After #1 harvests: sliding-verify-cuda-streamk-window and full-verify1-cuda-streamk-batch1. Both are
   reviewed (proceed, no companions) and both depend on the parent compiling and passing its
   self-check: read the parent's log tail first, or the run silently measures FA3. Build notes from
   the reviews, for the implement briefs:
   - sliding: mask every tile that overlaps either edge band, not just the first and last. The
     parent's full-verify path has the same gap, so fix it there too. Give the sliding route its own
     self-check flag so a failure can't drop full decode_16/32 back to FA3. Make the window a
     constexpr, cap the grid at batch x 33 tiles, and keep the route in its own predicate.
     Expect verify_1 at ~11-13 us, down from 17.6.
   - batch1: the B8 combine is the weak link (~8-13 us, vs the 2-3 us assumed). Build it single-pass
     and wide, or as the two-level merge. Self-check batch-1 cases at the 132-CTA grid: ~80K, very
     short, and non-multiple-of-64 lengths. Expect about +2% on the headline; judge it on the
     verify_1 and A8/B8 rows.
3. Rewrite stage 3 from whichever of the CUDA kernels lands.

### bf16 (20 checks)
1. Combine the winners of the four scored runs into one shape-gated master. Each touches different (layer, m) keys.
   - bf16-cublaslt-algo-pick-in-prepare-m128-m256
   - router-swapab-wgmma-all-m
   - bf16-m16-swapab-wgmma-extend
   - bf16-m32-swapab-wgmma-triton is promoted: 1.196x on the board, rank 349.
2. After the bf16 cuBLASLt result: unhold prefill's dense-gemm-cublaslt-algo-pick-prefill-t.
3. If the m=128 rewrite lands: unhold qkv-gate-o-m128-wgmma-splitk-semaphore, its stage-2 extension.

### decode (10 checks)
1. Combine the winners of the three scored runs into a decode restack. Then remint
   bf16-m32-swapab-wgmma-port (held) on top, with per-layer kernel names.
   - sliding-fused-triton-decode1-port
   - bf16-shape-gated-triton-gemms-small-t
   - paired-heads-pad-unpad-triton
2. After #2 (the CUDA port) harvests compiled and self-checked: full-verify-cuda-splitkv-stage2 (pre-reviewed).
3. moe-align-small-t-restack-on-master (reviewed; ~+0.5-0.9%).
4. fused-preattention-paired-pad-restack: reviewed, proceed, est. +0.8-1.2%. Dispatch only after
   paired-heads-pad-unpad-triton (a92b64be4c6547d3) harvests and its base is fixed. Skip the spec's
   "prepare() warm-up": the parent patch has none.

### moe (20 checks)
- Harvest w2-l2-prefetch-decode1 and align-parallel-per-expert-verify.
- If the CUDA swap-AB (#3) works, its stage 2 should fold in the fused silu. That is the reviewer's
  follow-up; swapab-w13-silu-epilogue-verify stays tied to the held Triton rewrite.
- Unhold swapab-nvfp4-gemm-verify-rewrite (the Triton swap-AB at bs>=128) only if prefill's
  packed dequant (49ff8248b96547de) logs "packed dequant ON" and beats stage 3.

### prefill (10 checks)
1. If 49ff8248b96547de logged "packed dequant ON" and gained: experts-triton-packed-dequant-turn4 (stage 6, pre-reviewed), then experts-triton-turn1-blockt32-stream (stage 7, pre-reviewed).
2. Restack the winners of the packed dequant and the fused silu epilogue (108231906a30407e).
3. silu-mul-triton-local-rows-marlin-turn (reviewed, ~0.7%).
4. dense-gemm-cublaslt-algo-pick-prefill-t, held until the bf16 precedent harvests.

## Scored runs left today, for reference
bf16 4, attention 4, moe 9, decode 4, prefill 7. These go to restacks of winning scored
experiments as they harvest. High-risk rewrites always run as checks.

## Change of plan (2026-10-24 ~18:17Z, user request)
The user asked to spend the day's leftover scored submits on ambitious tests. 15 of the 16
audited builds went out **today as scored submits**, not as checks at the reset. They count
against today's scored quota (the board keeps only the best run, so a slow one costs no rank).
Only `bf16-verify-streamk-cutlass-wgmma-rewrite` is still unrun: bf16 had 3 scored slots left, and
this rewrite overlaps pooler-m128 on the pooler shapes. It runs as a check at the reset.
The reset tick harvests these, then dispatches the "after these harvest" follow-ups as checks.
