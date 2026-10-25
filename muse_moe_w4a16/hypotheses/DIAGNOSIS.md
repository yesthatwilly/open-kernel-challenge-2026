---
diagnosed_at: 2026-10-25T203317Z
problem: muse_moe_w4a16
best: 1.4021x [1.3811,1.4231], job 31c27e4b8fc84bd2 (submit). This is branch code (combo-wg-occ2-v64-hot-interleave-align2) and is not on master. Master is combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 at 1.3866x (b395f5b6).
---

# muse_moe_w4a16 diagnosis

## Diagnosis
Four checklist items apply: **7 unstacked wins**, **4 wrong bottleneck**, **2 local minimum** and **6 ignored recommendations**.

- **Unstacked wins.** The board's 1.4021 (combo-wg-occ2-v64-hot-interleave-align2, wg2 V64 live at B256, verify_32 404.9 -> 358.1 µs) was never merged onto master. Neither was combo-smallm-two-stream-halves-x-w13-splitk (0.968-0.980 at T 8-32). DECISIONS 2026-10-25T190615Z and T192122Z record 6-7 merge conflicts and no restack minted. Every child built on master is therefore judged against a board that master does not contain.
- **Wrong bottleneck.** Every shape is memory-bound and sits 2.8-4.1x off its floor: decode_1 46.4/11.4, decode_16 150.9/52.5, decode_32 199.1/70.8, verify_1 104.5/37.3, verify_16 323.2/101.3, verify_32 358.1/108.9. In every case the binding cost is the NVFP4 GEMM mainloop running at only 1.5-2.1 TB/s:
  - Marlin at bs≤32
  - wg_kernel at B128/B256, which b128-n32-deep-ring showed is consumer/dequant-bound, not bound by bytes in flight
  - fused bs1, which reached 1.46 TB/s

  muse_bf16_proj streams similar m≤32 weights at ~3.5 TB/s, so the DRAM is not the limit. Yet 7 of the 9 queued (non-held) items target the two verify loads, and several of them are front-end or bytes-in-flight work (wg-producer-direct-x-rows-no-gather, cuda-nvfp4-wgmma-rega-b256-n64-deep-ring). The four small-M loads, bs 1/8/16/32, carry 2/3 of the geomean. decode_1 is the furthest from its floor (4.1x) and decode_32 is only 1.03x over SGLang.
- **Local minimum.** On Marlin's small-M knobs and on the ~3 µs align and post-norm kernels, about 20 slugs landed within the 1.24% margin. On the knob side:
  - w13-marlin-atomic-add-small-m, -on-master-small-m and -t1-only
  - w2-marlin-atomic-add-verify, marlin-workspace-blocks-per-sm-8, w13-marlin-bf16-reduce-verify
  - marlin-two-stream-expert-halves-small-m

  On the align and post-norm side:
  - align-small-one-warp-tiny-m, align-parallel-from-17-tokens
  - post-norm-combine-hoisted-slot-loads, post-norm-pdl-launch-small-m, post-norm-triton-hoisted-decode32
  - stack-onewarp-hoisted-dispatchcombine and its -from17 variant
  - cuda-dispatch-combine-pdl-verify

  decode_16, decode_32 and verify_1 have stayed in 146-152, 198-205 and 104-112 µs since 2026-10-24.
- **Ignored recommendations and build churn.** Several holds have outlived their conditions:
  - midm-bucket-cap32 and w2-only-midm still wait on s3/s4, both long harvested.
  - The occ2 reviewer's B256 hot-expert follow-up (w13 at OCC2, NMAX 128, STG 2) was never minted.

  Item 3 applies in part: 6 CUDA runs measured master instead of the new kernel.
  - cuda-nvfp4-wgmma-rega-grouped-verify: gencode
  - sm90a-rebuild-inproc: canary non-finite
  - s2-finite-fix: bucket NaN
  - s3-wide-bucket-fix: g_maps
  - cuda-nvfp4-swapab-mma-persistent-small-m: 715
  - cuda-nvfp4-wgmma-rega-tiny-m-bucket16: PATH CHOSEN = MASTER

  INDEX credits tiny-m-bucket16 with "gained decode_1", but its log shows the tiny path never ran.

## Strategy
**Pivot.** The incumbent's ceiling at small M is spent: every Marlin knob is flat, and Marlin runs at ~1.9 TB/s. The wg_kernel consumer is capped near 2 TB/s at every bucket. A small-M kernel without padding or smem staging that streams at ≥3 TB/s, plus a consumer fix shared by all wgmma lines, has a far higher ceiling than more front-end, align or B128 tuning. First, restack the board onto master so every measurement starts from 1.40.

## Actions (priority ordered; /hypothesize mints these first)
1. **What:** Restack onto master (toprow, bdbbc97b) in two parts, as one combination with a per-cell gate.
   - The C2 piece of combo-wg-occ2-v64-hot-interleave-align2: the `muse_moe_w4a16_wgmma_occ2_v1` module, `wg2_kernel<..,64,3,2,..>`, `_c2_checks`, and `_c2_select` gating on `_wg_judge_ids` plus zipf.
   - The two-stream expert halves at 8 ≤ T ≤ 32 from combo-smallm-two-stream-halves-x-w13-splitk, behind its own prepare gate and its own bound constant.

   Leave out F1 and F2: master already has them. Scored submit.

   **Why:** Board 1.4021 vs master 1.3866. Master's verify_32 row is 402.4 µs vs the board's 358.1 (wg2 0.955 vs single-CTA in its selector). Two-stream gated at 0.968-0.980 on T 8-32.

   **Impact:** verify_32 -44 µs (+2.0% headline). verify_1, decode_16 and decode_32 each -2-3% where the gate goes live (+~1%). Expect ~1.41-1.43.

   **Prior:** restack-bucket-gate-on-master-align-atomic (held as redundant). That restack was a different branch; this one is the board code itself.
2. **What:** Port the bulk row-owner, reduce-free M=1 stream from muse_bf16_proj (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, 741072dc) to NVFP4 at T ≤ 4 (decode_1 and the bs=4 test). Design:
   - **Repack in prepare (lossless).** Give each CTA one contiguous span of an expert's rows over the full K: codes plus e4m3 block scales. Interleave the gate and up rows of w13 so silu fuses into the w13 epilogue.
   - **Load path.** One producer lane issues 1-D `cp.async.bulk` copies into a deep ring, as in the bf16 kernel.
   - **Consumers.** Dequantize the codes and FMA on CUDA cores (bf16x2 products, fp32 accumulate), or use mma.sync m16n8k16. Pick between them in the prepare gate.
   - **No split-K, no tickets, no spins.** w13 writes bf16 h; w2 writes each route's bf16 y row over the full K = I.
   - **Device-side work assignment.** The route-to-CTA map is computed on the device from topk_ids, so there is no host branching.
   - **Gate.** Require ≥ 1.02x over master on skewed and `_wg_judge_ids` routings, with cold-L2 graph replays. Print kernel-only TB/s and a PATH CHOSEN line.
   - **Instructions.** Use none that are sm_90a-only. Measure first with `okc check`.

   **Why:** decode_1 is the load furthest from its floor (46.4 vs 11.4 µs; the Marlin pair is 35.4 µs on ~56 MB = 1.6 TB/s). bf16's row-owner stream reaches ~3.5 TB/s on m ≤ 32 weight streams and won qkv_gate decode_1 1.16x and pooler decode_1 1.07x. The bs=1 slow mode came from cross-CTA serial reduction (w2 atomic cut it from 44/100 to 10/100), and this design has no reduction at all. Every earlier small-M design kept padding, smem staging, split-K or grid spins:
   - tiny-m-bucket16: wgmma N16, 46.0 vs 45.5 µs at T=1
   - swapab mma.sync: 1.7-2x slower
   - fused bs1: bimodal from spins

   **Impact:** The decode_1 GEMMs drop from 35 to 17-22 µs at 2.6-3.3 TB/s. The shape lands at ~28-33 µs, worth +5.5-8% on the headline.

   **Prior:** cuda-fused-split-i-moe-tiny-t (queued). It keeps the wgmma rega consumer, which is the measured ~2 TB/s limit and only tied Marlin at T=1. This design changes the consumer and the load path, not the phase structure.
3. **What:** Fix the consumer of the shared wg_kernel before any further B128, B256 or split-I wgmma stage. Dispatch cuda-nvfp4-wgmma-rega-pingpong-consumers (p1), then cuda-nvfp4-wgmma-rega-producer-scale-expand, both rebased on the restacked wg2 master. Both must print kernel-only TB/s per bucket (N16 through N128) at B128 and B256.

   **Why:** b128-n32-deep-ring showed that bytes in flight do not limit wg_kernel (1.83-2.13 TB/s at 37-64 KB/SM). The dequant → 8 wgmma → wait<0> chain is serial per warpgroup. wg2's 2-CTA gain of 0.955 is the same effect seen through TLP.

   **Impact:** verify_32 wg2 rows total 366.8 µs; at 2.8 TB/s they would be ~240 µs, so verify_32 goes from ~358 to ~240 µs (+6-7% headline). B128 could flip off Marlin (260.6 µs in the Marlin rows), worth +2-3%.

   **Prior:** wg-two-ctas-per-sm-port-b128-n32 (TLP rather than the ILP inside a warpgroup). cuda-fused-split-i-moe-tiny-t and cuda-fused-split-i-moe-midt-cluster-reduce inherit this consumer, so hold them until this harvests.
4. **What:** After action 3, dispatch cuda-fused-split-i-moe-midt-cluster-reduce for 9 ≤ T ≤ 32 on the fixed consumer. If thread-block clusters are refused, drop to CL = 2 or to plain `red.global.add`.

   **Why:** decode_32 is the weakest load: 1.03x over SGLang, with Marlin at 176.8 µs against a 70.8 µs floor. Every Marlin knob there was flat (local minimum above). This is the only queued design that reads each weight byte once per call regardless of routing.

   **Impact:** decode_32 and decode_16 at 2.5 TB/s land at ~120 and ~95 µs, worth up to +8% on the headline.

   **Prior:** cuda-nvfp4-wgmma-rega-midm-bucket-cap32 (held). That design keeps the w13 → w2 boundary and the hs round trip, a structure the tiny path showed losing 1.15-1.5x at T = 4-7.
5. **What:** At B256, run w13 alone at OCC2 with NMAX 128 and STG 2 (~110 KB/CTA, so 2 fit per SM). Keep w2 at the wg2 V64 configuration. Gate on `_wg_judge_ids`.

   **Why:** This is the occ2 reviewer's unminted follow-up. With NMAX 64, the judge's hot experts (~130 routes at T=256) are cut into 3 chunks, so their weights are read twice.

   **Impact:** w13 at verify_32 is 225.3 µs; -5-10% on it is -11 to -22 µs, worth +0.5-1%. That is small, so mint it only after action 3, or fold it into action 3.

   **Prior:** combo-wg-occ2-v64-hot-interleave-align2, which ran NMAX 64 / STG 3 on both kernels.

## Do not try
- `triton-w13-grouped-gemv-small-m`, `small-m-grouped-gemv-splitk-rewrite`: Triton small-M W4A16 GEMV. Triton did not pipeline the K loop (smem 8 KB, 211 regs) and the kernel was dequant-ALU bound: 2-3.2x slower than Marlin at bs 1/8/16. The held children `triton-w13-silu-epilogue`, `triton-w2-grouped-gemv-small-m` and `triton-small-m-threshold-32` should be closed.
- `cuda-nvfp4-swapab-mma-persistent-small-m-inproc`: ran correctly once the fault was fixed and was 1.7-2x slower than Marlin at T ≤ 32 (75.9 vs 44.7 µs at bs=1). Close `cuda-fused-moe-midm-single-launch-routed` and `-childcheck` (child processes are rejected by the rules).
- `cuda-fused-moe-bs1-persistent-*` (3 stages): grid-wide phase-2 flag spins left the judge's timed loop bimodal (41-59% of replays at +23 µs) even though the prepare DIAG was unimodal. Grid-wide spin dataflow is dead at bs=1. `-nospin-bandwidth` (child process) is held and should be closed.
- `cuda-nvfp4-wgmma-rega-tiny-m-bucket16`: the N16 wgmma path at T ≤ 7 lost its own gate (46.0 vs 45.5 µs at T=1 skewed, 1.15-1.5x at T=1 random, T=4 and T=7) and never ran. Do not run its consumer at T ≤ 7 again until action 3 lifts the consumer's TB/s. A stage built on a new consumer is not a re-run.
- Marlin knobs at small M are exhausted: `marlin-workspace-blocks-per-sm-8`, `w13-marlin-bf16-reduce-verify`, `w13-marlin-atomic-add-*` (routing-dependent; only T=1 is live), `w2-marlin-atomic-add-verify` (+0.3-0.4% at T ≥ 33), `w2-l2-prefetch-decode1` (0.79x at bs=1), `marlin-block-size-m-next-larger`, `marlin-block-m-8-verify16`. Close the held `marlin-two-class-block-m-decode32` and `marlin-exec-config-autotune-mid-m`: the reviewer expected both flat, and there is no evidence the kwargs exist.
- Align, dispatch and post-norm micro-work on kernels of 1-6 µs is below resolution (≤0.3% headline per REWRITE_READ §1): `cuda-dispatch-combine-pdl-verify`, `post-norm-pdl-launch-small-m`, `post-norm-triton-hoisted-decode32`, `post-norm-combine-hoisted-slot-loads`, `align-small-one-warp-tiny-m`, `align-parallel-from-17-tokens`, `stack-onewarp-hoisted-dispatchcombine(-from17)`. Close the held `cuda-align-kernel-a-from17`, `cuda-combine-kernel-b-bs128-only` and `align-remote-sentinel-ids`. Also do not spend a check on `wg-producer-direct-x-rows-no-gather` alone: the gather is 4.7 µs on one load, ~0.2% headline. Fold it into a consumer stage if at all.
- `cuda-nvfp4-wgmma-rega-b128-n32-deep-ring` and the queued `cuda-nvfp4-wgmma-rega-b256-n64-deep-ring`: deeper weight rings were flat at B128 (1.83 → 1.87 TB/s), which refutes the b256 file's "capped by bytes in flight" premise. Drop it.
- `swapab-nvfp4-gemm-verify-rewrite`, `swapab-w13-silu-epilogue-verify`: the Triton verify rewrite has been overtaken by the wgmma B256 path (358 µs). Prefill's Triton expert GEMM also lost to Marlin below first-turn sizes. Close both.
- Holds whose parent stage has been superseded (s3, s4 and s5 all harvested): `cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild`, `-s2-fix-gate-from-8`, `-s2-per-kernel-enable`, `-midm-bucket-cap32`, `-w2-only-midm`, `restack-bucket-gate-on-master-align-atomic`, `silu-mul-triton-row-tiled-gated` (silu is fused in the wgmma path; Triton act_and_mul ran 2x slower in prefill). Close them, not unhold.

## LESSONS edits (proposed; churnsubs applies them)
- correct (add a line next to 202): "moe cuda-nvfp4-wgmma-rega-tiny-m-bucket16 (3df5c697) never went live. Its gate printed PATH CHOSEN = MASTER for T=1 (46.0 vs 45.5 µs skewed, 123.9 vs 107.4 random-local) and for T=2-7 (1.14-1.5x slower). Its 'gained decode_1' row is master's w2 atomic measured against an older base. The wgmma rega consumer does not beat Marlin at T ≤ 7." Evidence: archive/cuda-nvfp4-wgmma-rega-tiny-m-bucket16/3df5c69750884600.log.
- merge: "moe small-m-grouped-gemv-splitk-rewrite stage 1 (split-K grouped GEMV) lost 1.7-2.6x at bs 1/8/16 (0.664x)." with "A Triton W4A16 grouped GEMV with arithmetic FP4 dequant was 2-2.6x slower than Marlin at small M (triton-w13-grouped-gemv-small-m)..." and line 154 (swap-AB mma.sync 1.7-2x slower). Replace them with one line: "Small-M NVFP4 kernels that lost to Marlin: Triton GEMV x2 (unpipelined, dequant-bound), swap-AB mma.sync (1.7-2x), wgmma N16 tiny (1.0-1.5x), fused bs1 persistent (bimodal spins). None exceeded ~2 TB/s." They are three lines about one finding, and the tiny result above completes it.
- correct: "Knobs that did nothing: workspace blocks per SM, use_fp32_reduce=False, use_atomic_add." → add "except use_atomic_add on w2 at T=1, which removed the bs=1 slow mode (48.8 → 44.9 µs, w2-marlin-atomic-add-small-m), and on w13 at T=1, which depends on the routing (0.750 random, 0.997 skewed, combo-wg-occ2-v64-hot-interleave-align2)". As written, the line contradicts lines 167 and 202.
- add: "moe vs bf16: the NVFP4 GEMMs here (Marlin, wg_kernel, fused bs1) stream at 1.5-2.1 TB/s, while bf16_proj's m ≤ 32 weight streams reach ~3.5 TB/s. The small-M limit is dequant/consumer issue, not DRAM. Judge new small-M kernels by printed kernel-only TB/s." Evidence: REWRITE_READ_2026-10-25 §4, b128-n32-deep-ring, muse_bf16_proj pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite.
