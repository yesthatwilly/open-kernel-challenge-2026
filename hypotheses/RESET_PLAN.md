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

## Reset priorities for 2026-10-25 (user, 2026-10-24 ~23:50Z), in order

Run these before ordinary queue dispatch. Harvest first (tonight's decode stacks and the prefill scored runs).

1. **Scored confirmations, then land.**
   - decode `stack-align-par-sidestream-sliding1`: check `cc92097b64784c0f` 1.2495 [1.231, 1.268], the best decode result (board best 1.2283). One scored submit; if it again beats 1.2283, land it on master (it already contains the dense-chain PDL master).
   - bf16 `bf16-restack-m16-swapab-router-swapab-m32`: scored `1e90292c7b954d96` 1.2215 [1.203, 1.240] vs master lineage 1.1959. One more scored submit of the same code to confirm (user-approved exception to "identical code is never re-run"), then land it on master.
2. **Prefill master = the five-way stack.** `stack-tonight-five-prefill` measured 1.0705 (check `c4d14d62f11e4aaf`) and 1.0744 (scored `502f2eab95164781`, now on the board, rank 273). Land it on master (user-approved). Do NOT carry `align-local-parallel-first-turn` (six-way stack 1.0642 / 1.0672: it costs ~0.6%). `silu-mul-sorted-first-turn` (built, unrun) is now a stage on the five-stack lineage: re-base it on the new master before running.
3. **Per-shape read of today's 15 rewrites before any stage 2.** For each rewrite in done/ from 2026-10-24 (bf16 x3, attention x3, moe x4, decode x2 incl. the failed moe-chain, prefill x2 + the sliding-attn and the moe/prefill night rewrites): read the replaced kernels' per-shape rows and PATH CHOSEN lines from the archived profiles, against the incumbent and floor in its `## Milestones`. Record per rewrite: path actually taken, target-shape speedup, verdict (promising stage 2 / dead). Mint stage 2 only for the promising ones. One Opus agent per problem is fine.
4. **moe: stack the sub-noise winners**, the way prefill's stack showed accumulation. Candidates at 1.328-1.329 vs best 1.3278: `align-small-one-warp-tiny-m`, `post-norm-combine-hoisted-slot-loads`, `cuda-dispatch-combine-pdl-verify` (bs >= 128 align+combine, disjoint from the other two). Mint one stack of all three (+ `align-parallel-from-17-tokens` as a separate variant), restack via the fixed claim, run as checks.

Decode note: `stack-align-par-sliding-batched` (`b94e797333fb4a27`) FAILED its tests: the parallel align and the batched sliding port do not compose as built; do not stack them again without a diagnosis.

### Progress (reset tick, 2026-10-25 ~00:15Z)

1. Scored confirmations SUBMITTED: decode stack-align-par-sidestream-sliding1 -> `39bcf05afc4e4dd3`, bf16 restack -> `60b20fd0bf924267` (from `<slug>-confirm` worktrees). NEXT TICK: when they finish, if each beats its best again (decode 1.2283, bf16 1.1959), `churn.py land <p> <slug> --reason ...` (both are in done/neutral), then remove the `-confirm` worktrees.
2. DONE: prefill stack-tonight-five-prefill landed on master (5bdc653, BEST 1.0705), promotion recorded as scored `502f2eab95164781`. silu-mul-sorted-first-turn re-based onto it and dispatched as check `9528eed511b04f33`.
3. IN FLIGHT: five Opus per-shape readers write `<p>/hypotheses/REWRITE_READ_2026-10-25.md` (uncommitted). NEXT TICK: commit them, then mint/unhold stage 2 only for PROMISING verdicts; hold stage-2 picks (full-verify-cuda-splitkv-stage2, cuda-fused-moe-midm-single-launch-routed, experts-triton-packed-dequant-turn4) until their problem's read is in.
4. DONE: moe stacks dispatched as checks: stack-onewarp-hoisted-dispatchcombine `7e5ab95a413f4ebe`, and the +from17 variant `0fde9013de97404d` (with a CDC_MIN_TOKENS gate fix, see its notes).
