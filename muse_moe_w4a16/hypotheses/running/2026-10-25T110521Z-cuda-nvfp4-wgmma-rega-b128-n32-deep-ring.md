---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-b128-n32-deep-ring
created: 2026-10-25T110521Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2764050).
  Changes the bucket where wgmma LOSES (B128 = 128<=T<=255: verify_16 and the skewed-128 test) and
  nothing else: B256 (256<=T<=512) keeps s5's wg_kernel byte for byte and stays forced on, as an
  in-run control; T<128 and T>512 stay master. MECHANISM: s5's wg_kernel streams NVFP4 weight at
  ~2.2 TB/s whatever the route count (w13 238.4 us judged at bs=128 vs 271.0 at bs=256; prepare
  cold w13 303 vs 308 us at 241 vs 464 local routes), i.e. it is capped by bytes in flight, not by
  MMA or dequant (s3's double-buffered register-A variant B was no faster than variant A in every
  N bucket, so the consumer is not the limiter). Each stage is 9216 B of weight plus a token slab
  sized for N=128 (2 x SLAB_HALF = 2 x 16384 B), so STAGES=4 fits and only 4 x 9216 = 37 KB of
  weight per SM is in flight; LESSONS (bf16): 64 KB in flight per CTA reached 3.3 TB/s, 128 KB
  reached 3.73 TB/s. At B128 the judge-like routing has ~7.5 routes per local expert, so nearly
  every chunk is the N16 bucket and the 32 KB slab space is idle. CHANGE: add a second
  instantiation wg_kernel<G, NMAX=32> used only for B128: (1) on-device chunking cuts chunks every
  2 blocks of 16 rows (N <= 32) instead of every 8, so only the N16/N32 buckets and TMA maps are
  used (an expert with more than 32 routes is split into several chunks and re-streams its weight
  per chunk: correct, slower only for skewed tests); (2) SLAB_HALF becomes 32 x 128 = 4096 B and
  STAGE_BYTES = 9216 + 8192 = 17408 (1024-B aligned for SWIZZLE_128B, as are the weight record and
  each slab half); make_desc offsets use the template SLAB_HALF; (3) STAGES = 10 for both w13 and
  w2 (dyn smem w13 1024 + 174080 + 18432 = 193536 B, w2 1024 + 174080 + 34816 = 209920 B, each +
  <=8192 static under 227 KB), so 92 KB of weight is in flight per SM, 2.5x s5; the mbarrier
  arrays and phase arithmetic take STAGES as the template constant. Nothing else moves: same
  weight bank, gather, variant A (wgmma_wait<0>), epilogues, per-call cuTensorMapEncodeTiled from
  bsrc, shape+flag kernel_fn gate (state['wgmma_ok'] and 128<=T<=255 picks the NMAX=32 launch;
  256<=T<=512 the s5 launch). Bump the load_inline module name. prepare: build-time register/spill
  print for both new instantiations (must be <=232 regs, 0 B spill, else B128 falls back to the s5
  kernel and the log says why); the s5 per-bucket canary for N16/N32 on the new kernel; s5's
  three-state copy check (A, deepcopy B, tree-cloned C) and device launch marker at T=128 with a
  forced >32-routes-per-expert routing (all-one-expert, skewed-4hot) so multi-chunk re-streaming
  is exercised; error vs the fp32-dequant reference <=1.5x Marlin's own. No timing gate (forced,
  as s5, so the judged profile names the new kernel rows); print the s5-style Zipf cold-L2 timing
  at T=128 for NMAX=32 vs s5 kernel vs Marlin, compact and last in the log. Hard rules: CUDA-graph
  capturable; kernel_fn does no host sync, no allocation or branching on input values, no prints,
  no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4
  (dequant in registers), activations BF16; attend every route; no child processes; no
  module/static global holds device pointers, TMA maps or buffers (judge rotates TWO state
  copies); TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20 as s5.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "s5 judged wgmma rows at bs=128: w13+silu 238.4 + w2 110.9 + gather 3.7 = 353.0 us vs master Marlin pair 305.9 + act_and_mul 4.0 = 309.9 us (shape 364.8 vs master 317.8). 2.5x weight bytes in flight should lift ~2.2 TB/s toward 3.3-3.7 TB/s: target w13 <=170, w2 <=80, pair <=0.9x Marlin (<=279 us), shape row <=300 us"
  - shapes: bs=256 *
    direction: flat
    why: "in-run control: s5 kernel unchanged at B256 (404.7 us shape, wg rows 271.0 + 126.1 + 4.8)"
  - shapes: bs=1 *
    direction: flat
    why: "control: master below 128 tokens"
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=255
review: proceed
status: running
review_at: 2026-10-25T120207Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T120207Z
base_sha: 57e77548ba31c2145d575df62752f62684b2f6ad
base_blob: 77f34503659c90a5953f91a93ba6e1a2b02e4613
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T121529Z
impl_commit: c30fd0cc3d4a310d54f63abf868723b75b3f1210
submission_sha256: f56d66ed56172aea69bf8f2b46153a13aca09da79762057579db1c791a27b1fb
norm_sha256: 64b78e769b3c8135c0204f0801805499ad9bb6cfaadf5040fb58dffd503b5750
submission_blob: 0ff43a7c6e340a04dce17ace8dc8e7b37aca61ac
lint_warnings: 78
attempt_file: 2026-10-25T121529Z-check-cuda-nvfp4-wgmma-rega-b128-n32-deep-ring.md
prepared_at: 2026-10-25T121529Z
job: 6759e36bbd4648fc
dispatched_at: 2026-10-25T121537Z
---

# cuda-nvfp4-wgmma-rega-b128-n32-deep-ring

## Change

Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2764050). Changes the bucket where wgmma LOSES (B128 = 128<=T<=255: verify_16 and the skewed-128 test) and nothing else: B256 (256<=T<=512) keeps s5's wg_kernel byte for byte and stays forced on, as an in-run control; T<128 and T>512 stay master. MECHANISM: s5's wg_kernel streams NVFP4 weight at ~2.2 TB/s whatever the route count (w13 238.4 us judged at bs=128 vs 271.0 at bs=256; prepare cold w13 303 vs 308 us at 241 vs 464 local routes), i.e. it is capped by bytes in flight, not by MMA or dequant (s3's double-buffered register-A variant B was no faster than variant A in every N bucket, so the consumer is not the limiter). Each stage is 9216 B of weight plus a token slab sized for N=128 (2 x SLAB_HALF = 2 x 16384 B), so STAGES=4 fits and only 4 x 9216 = 37 KB of weight per SM is in flight; LESSONS (bf16): 64 KB in flight per CTA reached 3.3 TB/s, 128 KB reached 3.73 TB/s. At B128 the judge-like routing has ~7.5 routes per local expert, so nearly every chunk is the N16 bucket and the 32 KB slab space is idle. CHANGE: add a second instantiation wg_kernel<G, NMAX=32> used only for B128: (1) on-device chunking cuts chunks every 2 blocks of 16 rows (N <= 32) instead of every 8, so only the N16/N32 buckets and TMA maps are used (an expert with more than 32 routes is split into several chunks and re-streams its weight per chunk: correct, slower only for skewed tests); (2) SLAB_HALF becomes 32 x 128 = 4096 B and STAGE_BYTES = 9216 + 8192 = 17408 (1024-B aligned for SWIZZLE_128B, as are the weight record and each slab half); make_desc offsets use the template SLAB_HALF; (3) STAGES = 10 for both w13 and w2 (dyn smem w13 1024 + 174080 + 18432 = 193536 B, w2 1024 + 174080 + 34816 = 209920 B, each + <=8192 static under 227 KB), so 92 KB of weight is in flight per SM, 2.5x s5; the mbarrier arrays and phase arithmetic take STAGES as the template constant. Nothing else moves: same weight bank, gather, variant A (wgmma_wait<0>), epilogues, per-call cuTensorMapEncodeTiled from bsrc, shape+flag kernel_fn gate (state['wgmma_ok'] and 128<=T<=255 picks the NMAX=32 launch; 256<=T<=512 the s5 launch). Bump the load_inline module name. prepare: build-time register/spill print for both new instantiations (must be <=232 regs, 0 B spill, else B128 falls back to the s5 kernel and the log says why); the s5 per-bucket canary for N16/N32 on the new kernel; s5's three-state copy check (A, deepcopy B, tree-cloned C) and device launch marker at T=128 with a forced >32-routes-per-expert routing (all-one-expert, skewed-4hot) so multi-chunk re-streaming is exercised; error vs the fp32-dequant reference <=1.5x Marlin's own. No timing gate (forced, as s5, so the judged profile names the new kernel rows); print the s5-style Zipf cold-L2 timing at T=128 for NMAX=32 vs s5 kernel vs Marlin, compact and last in the log. Hard rules: CUDA-graph capturable; kernel_fn does no host sync, no allocation or branching on input values, no prints, no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16; attend every route; no child processes; no module/static global holds device pointers, TMA maps or buffers (judge rotates TWO state copies); TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20 as s5.

## Expected

- `bs=128 *` faster: s5 judged wgmma rows at bs=128: w13+silu 238.4 + w2 110.9 + gather 3.7 = 353.0 us vs master Marlin pair 305.9 + act_and_mul 4.0 = 309.9 us (shape 364.8 vs master 317.8). 2.5x weight bytes in flight should lift ~2.2 TB/s toward 3.3-3.7 TB/s: target w13 <=170, w2 <=80, pair <=0.9x Marlin (<=279 us), shape row <=300 us
- `bs=256 *` flat: in-run control: s5 kernel unchanged at B256 (404.7 us shape, wg rows 271.0 + 126.1 + 4.8)
- `bs=1 *` flat: control: master below 128 tokens

## Risk

Risk: if bandwidth is limited by something other than bytes in flight (per-item epilogue bubbles, the single producer thread, L2 slab traffic), w13 stays near 238 us; the s3 variant-B null already rules out consumer dequant/MMA overlap. Second risk: smem/regs for STAGES=10 plus a second instantiation pair grows build time; s5 prepare was ~40 s with a 200 s setup budget. Read the judged bs=128 wg_kernel rows against s5's 238.4/110.9, not the headline (bs=128 Marlin rows drift 306-350 us across runs). Milestones: M1 new kernel rows appear in the bs=128 profile; M2 w13+w2+gather <=0.9x master's Marlin pair+act (<=279 us); M3 B128 enabled behind the LESSONS-173 selector in a later stage.

## Milestones

Rewrite stage 6 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=255`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T120207Z by an adversarial subagent (model opus).

**Verdict: proceed.** Yes, spend moe's last check on this, but not as written. Rebase it onto the bucket-gate build (hyp/.../s5-copy-safe-bucket-gate). Keep B256 byte-identical to that build, and put the new NMAX=32 kernel at B128 behind the same >=1.02x graph-replay selector against Marlin, instead of forcing it on. One check then either beats 1.3744 with a landable build or re-measures the pending land candidate, and the prepare log still shows the ring-depth effect.

### Justification
MECHANISM: plausible for the routing it was derived from, overstated for the judged one. w13 is 8192x4096 NVFP4, about 18.9 MB per expert, and a whole expert is about 28.3 MB. The prepare zipf a=0.3 routing touches all 32 local experts, about 604 MB of w13, which s5 moves in 303-308 us at about 2.0 TB/s. With 132 CTAs x 37 KB in flight, that matches Little's law at the about 2.5 us loaded latency implied by the bf16 LESSONS number (64 KB/CTA -> 3.3 TB/s). So 'capped by bytes in flight' is reasonable there, and STAGES=10 (92 KB) could plausibly reach about 3 TB/s. JUDGED REGIME: reference.py's floor counts touched experts. 101.3 us x 4.8 TB/s gives about 17 touched experts at bs=128, and about 18 at bs=256. The judge's Student-t popularity plus the busiest-GPU rotation concentrates routes. The judged s5 w13 row of 238.4 us is therefore about 321 MB at only about 1.35 TB/s, and the pair is about 484 MB in 349 us, about 1.39 TB/s, against Marlin's 1.58 TB/s (306 us). In the scored regime a second limiter is visible: tail imbalance (about 17 chunks x 64 tiles = about 1088 items over 132 CTAs is about 8.2 rounds), and per-expert route counts large enough to need N32/N64/N128 consumers. A deeper ring cannot fix either. WEAKEST LINK: flat time across N is equally consistent with a dequant-bound consumer, since dequant work scales with weight bytes, not N. The s3 variant-B null only rules out a gain from overlapping MMA with dequant, not dequant ALU throughput. Also, '7.5 routes per local expert' is the prepare routing. At about 17 touched experts and about 14 routes per expert on average with a heavy tail, the hot experts exceed 32 routes, so NMAX=32 splits them and re-streams 18.9 MB of w13 (plus 9.4 MB of w2) per extra chunk. In the judged regime that cuts directly into the gain, and the prepare zipf a=0.3 routing will never trigger it. TARGETS: w13 <=170 / w2 <=80 needs 1.9-2.0 TB/s on the judged bytes. That is reachable only if the ring is really the limit and re-streaming stays below about 10%. A pair of about 280-300 us (beating Marlin's 306 by a little) is the realistic upper outcome, not a sure one. Upside if it lands: bs=128 from 314 to about 285-290 us scored, about +1.4-1.6% on the headline (about 1.39-1.40). Shape mapping is consistent with TASK.md: verify_16 = bs128 is the only scored B128 shape, and the skewed-128 test is a correctness test only. Q1, BEST USE OF THE LAST CHECK: yes among the queued moe items. B128 is the only scored verify bucket still on Marlin, the change isolates one variable (ring depth via slab size), and its prepare A/B against the unchanged s5 kernel is interleaved in-process. s6-per-kernel-bucket-mix is weaker: the forced profile shows wgmma losing both cells at B128 (238.4+gather vs Marlin call 1, 110.9 vs Marlin call 2, against a Marlin pair of 306), so it can only reshuffle a known loss. b256-n64-deep-ring widens a bucket that already wins, with the same mechanism risk and a smaller relative gap. But only the rebased/gated form makes the check landable. The forced form puts a probable loser on bs=128 (s5-forced scored 364.8 vs 314.3 Marlin), so the headline cannot be promoted. Its only extra yield, judged kernel rows for the new kernel, matters less now: the judged-vs-prepare gap is calibrated (selector predicted direction correctly at B128 0.80x and B256 1.05x), and the forced rows already exist for s5. Q2, BASE ON BUCKET-GATE: yes. FINDINGS says that build is code-identical to s5-forced apart from the selector, and judged B256 rows match (239.2+119.0+4.8 vs 271.0+126.1+4.8; scored 404.85 vs 404.67). Basing there keeps the B256 win, makes bs=256 an in-run control against job 7be125e9 (404.85 us), and lets M3 (the selector) merge into this stage, which saves a future check.

### Confounders on the current kernel
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate / s5-forced (parent kernel, not landed): every B128 comparison is against s5's wg_kernel or Marlin, so read deltas against the bucket-gate job 7be125e9 (bs=128 scored 314.26, Marlin path) and the forced job d4ad4f31 (wg rows 238.4/110.9), never against BEST.json's master row alone.
- marlin-block-m-two-smaller-on-postnorm (landed, block_size_m two candidates down at T>=128): sets Marlin's bsm 16 at bs=128, which is the incumbent the B128 selector must beat. If that pick were different, the Marlin bar (306 us judged) would move and so would the gate decision. No interaction with the wgmma kernel itself, which aligns at block 16 on its own.
- align-parallel-per-expert-verify (landed, align at T>32): the wgmma path uses master's moe_align_block_size at block 16, not this align, but the gate compares full paths. Align cost (2.5-4 us) differs between the two paths and is inside the selector's 'full path' timing, so it is correctly included. Not a confounder for the kernel rows.
- post-expert-norm / w2-atomic-add / triton-map-align (landed, all T<128 or downstream norm unchanged): untouched by this change. The norm runs identically after either path. No confound.

### Interference and A/B needs
No companion. Moe has one check left, so a companion is unaffordable, and none is needed if the build is rebased as recommended. The prepare log then holds the clean A/B the result needs: s5 kernel vs NMAX=32 vs Marlin at T=128, interleaved, cold L2, 41 reps, empty-graph baseline subtracted, on the same state. That separates 'deeper ring helps' from 'judge drift', and the bs=256 row (byte-identical to bucket-gate) is the in-run drift control. Two additions make one run interpretable either way. (1) The selector and the timing print must include a judge-like heavy-tailed routing: Student-t per-expert popularity plus busiest-GPU rotation, as reference.generate_input does, synthesized in prepare, never taken from inputs. Print touched experts, max routes per expert, chunk count and re-streamed weight bytes. Gate on the WORSE ratio of zipf a=0.3 and that routing, because zipf a=0.3 touches all 32 experts at about 7.5 routes each and never exercises NMAX=32's re-streaming, which the judge's about-17-touched-expert routing will. (2) Print achieved TB/s per kernel (touched-expert bytes / time) for s5 and NMAX=32 at both routings. If NMAX=32 is flat vs s5 at the all-touched routing, the ring is not the limiter, and the consumer (dequant + wgmma_wait<0>) is the next stage's target, not the b256-n64 sibling.

### Open questions
- Rebase detail: confirm with the user that basing on the unlanded bucket-gate branch is acceptable while its landing is pending. If bucket-gate is then landed, this result restacks with no conflict, because B256 is byte-identical.
- If the user insists on the forced form (to get judged NMAX=32 rows), accept that bs=128 will likely score worse than 314 us and the run is not promotable. Then the run is worth the check only as a rewrite-stage measurement.
- Consider NMAX=64 / STAGES=7 at B128 instead of NMAX=32 / STAGES=10. With about 17 touched experts and a heavy tail, the 2-block chunk cut may re-stream more weight than the deeper ring saves. A prepare print of max-routes-per-expert under the judge-like routing decides this, and building both and letting the selector pick is cheap (two more template instantiations).
- Build time: two extra instantiations (w13/w2 at NMAX=32) plus the selector add perhaps 20-40 s to s5's 37.5 s. That is well under WG_SETUP_BUDGET_S=200, but the build-failure fallback must leave B128 on Marlin (not s5) and B256 on the bucket-gate decision.
- STAGES=10 is not a power of two: n % STAGES and (n / STAGES) & 1 stay correct, but recheck any code that assumed STAGES=4 (bit masks, unrolled init loops, barrier array sizing tied to 2*STAGES in static smem).


## Implementation notes

Base: `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate:muse_moe_w4a16/submission.py` copied over the worktree's submission.py (per the user's instruction and the binding review; B256 route ON in that build, checked 1.3744). That file already contains master's landed align-parallel and w2-atomic changes, so the stale_base 3-way merge was not run (replacing with the bucket-gate file was the explicit instruction). Edit only the worktree's `muse_moe_w4a16/submission.py`.

Review items applied (the review wins over the spec where they differ):
- Rebase onto bucket-gate, B256 byte-identical: B256 always runs variant 0 = `wg_kernel<G,128,4>`, which is the s5 kernel with the same code (the ring geometry became template constants; for (128,4) every constant, chunk cut `& 7`, N16..N128 consumers and smem is unchanged). The B256 selector body, the calibrated-zipf gate and `wg_on_256` are untouched.
- Not forced: B128 goes through the existing graph-replay selector at T=128. `_wg_select_b128` times Marlin full path vs s5 kernel vs NMAX32/STAGES10 vs NMAX64/STAGES7 (full-path graphs, cold L2 flush, 41 interleaved reps, empty-graph baseline subtracted) and enables the best variant only if its worse ratio is >= 1.02x.
- Build/self-check failure leaves B128 on Marlin: `wg_gemm`/`wg_init` take a variant; the module is built with `WG_NEW_VARIANTS=1`; if that build fails the s5-only module (`WG_NEW_VARIANTS=0`, name `..._s5only`) is built, B256 keeps the bucket-gate decision and B128 has no candidates (Marlin, not s5). A new variant that fails its attribute check (regs/occupancy/smem/bin, and 0 B spill), `_wg_variant_check` or the copy check only drops itself.
- Judge-like routing: `_wg_judge_ids` rebuilds reference.generate_input (dist normal, rank busiest: rand scores + 0.03 * Student-t popularity, top-8, busiest-GPU rotation) in prepare from a fixed CPU generator, 8 draws, keeps the draw whose touched-expert count is nearest 17 (the review's floor estimate); never from inputs. Gate = worse ratio of zipf a=0.3 and that routing (B128 only; random routing dropped from B128, B256 prints as before).
- Prints per variant and routing: touched experts, max routes on one expert, chunk count (and experts split), weight read twice (MB; a k-chunk expert re-streams k-1 times its w13+w2 bytes), w13 and w2 kernel-graph us and achieved TB/s (touched-expert bytes / kernel-graph time, empty-graph baseline subtracted).
- STAGES=10 not a power of two: all stage indexing is `n % STG` and `(n / STG) & 1`; barrier array is `2*STG` (160 B static at 10, the 8192 B static bound still holds: 5x384x4 + 160 + 48 + 4 = 7892); the init loop is `#pragma unroll` over STG; the only `& (CH-1)` is the chunk cut, CH = NMAX/16 in {2,4,8} (power of two, static_assert'ed). Nothing else assumed 4 or a power of two.
- NMAX=64/STAGES=7 added as the review's open-question suggestion (stage 25600 B, dyn smem w13 197632 B / w2 215040 B + <=8192 static < 227 KB).
- Open question on the unlanded base: user chose it. Open question on forced form: user chose gated.

What changed (all in submission.py):
- CUDA: `Geo<NMAX,STG>` (SLAB_HALF = NMAX*128, STAGE_BYTES = 9216 + 2*SLAB_HALF, 1024-B alignment static_asserts), `Smem<G,NMAX,STG>`; `consume_item<NB,G,NMAX,STG>` and `wg_kernel<G,NMAX,STG>` use the template constants (make_desc offsets use the template SLAB_HALF); chunk cut every CH blocks; consumer buckets above NMAX are not instantiated (`if constexpr`); `wg_init` returns 11 ints per instantiation (variant first); `wg_gemm` takes `variant` (0 s5, 1 NMAX32/10, 2 NMAX64/7). Ext name bumped to `muse_moe_w4a16_wgmma_rega_v6`.
- Python: `WG_VARIANTS`; `variant=` threaded through `_wg_gemm`, `_moe_wgmma`, `_wg_eval_case`, `_wg_chunks(cut)`; new `_wg_judge_ids`, `_wg_route_stats`, `_wg_variant_stats`, `_wg_variant_check`, `_wg_select_b128`; `_wg_select` takes `cands128` and returns `(decisions, lines, v128)`; `_wg_copy_check(plan)` runs the three-state (A, deepcopy B, tree-clone C) refilled-replay check once per variant (new variants at M 128 and 255) and returns per-variant results; `_wg_setup` builds/falls back, parses the variant attribute table, runs variant checks and the copy check, picks candidates, sets `state['wg_on_128']`, `state['wg_v128']`; PATH CHOSEN lines name the variant.
- kernel_fn: shape + `wg_on_128` / `wg_on_256` / int `wg_v128` only; no try/except, prints or per-call state; B256 call unchanged.

Not run (no GPU/nvcc on this node): nothing was compiled or executed. Biggest risk: the CUDA template edits compile only in the judge's nvcc (first build of the 6 instantiations; build time maybe 2x s5's, inside WG_SETUP_BUDGET_S=200 with the s5-only fallback if it fails) and the ring-depth effect may be flat if bytes in flight is not the limiter (the printed TB/s per variant answers that either way).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only a Python int read (`state.get("wg_v128")`) and one extra kernel argument; the variant is chosen on host before capture; per-call tensor-map encode, no sync/allocation/branching on input values (checked in `git diff`: the only kernel_fn change is the B128/B256 branch).
- [x] Nothing derived from `data` is cached across calls: the judge-like and zipf routings are built in prepare from fixed seeds; state holds only flags/ints.
- [x] Weights: same lossless retiled NVFP4 bank in prepare; NVFP4 stays NVFP4 (dequant in registers); no new weight format.
- [x] Activations BF16 (gathered BF16 slab, BF16 outputs); no KV cache/queries in this problem (n/a for those clauses).
- [x] Attends every route: chunks of an expert with more than NMAX routes are split into several chunks (each re-streams the weight); the variant checks (c=64/128 one expert, skewed-4hot, all-one-expert at T=255) and copy check compare against the fp32-dequant reference; no skipping.
- [x] No timing patches, harness access, module edits, reference import (reference.py was only read to mirror generate_input; nothing imported), subprocesses (load_inline is the existing in-process build) or network.
- [x] All work inside the timed call; no output caching or replay; no module/static device state (audit `_wg_audit_cpp` scans the new source; only the host driver function pointer remains); per-call TMA encode, launch markers, LAUNCHED and PATH CHOSEN lines kept; the judge's two copied states are covered by the copy check on every enabled variant.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a around load_inline and `__CUDA_ARCH_FEAT_SM90_ALL` guards unchanged; every mbarrier wait stays bounded (2^26 cycles -> error word 1).

Lint (okc_lint, `python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`): `OK with warnings` (only host-sync/dtype warnings at prepare()-only lines, outside kernel_fn's call graph). undef.py: `[]` (no undefined names). py_compile: ok. __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus), against the bucket-gate base (hyp/.../s5-copy-safe-bucket-gate, checked 1.3744). **Verdict: dispatch-with-fixes.**

Checked and found sound: variant 0 = `wg_kernel<G,128,4>` reproduces the s5 kernel exactly (CH=8 gives the same `& 7` / `cn < 8` chunk cut; SLAB_HALF 16384, STAGE_BYTES 41984, 4 stages; stage index `n % STG` and parity `(n / STG) & 1` give the same values; tx bytes, TMA offsets, make_desc and epilogue are unchanged), so the B256 kernel cannot move. For the new instantiations, smem with the 8192 B static bound is 218112 B (n32s10 w2) and 223232 B (n64s7 w2), both at most 232448 B; stages, slab halves and weight records are all 1024-B aligned; `cudaFuncSetAttribute` runs once per instantiation in `wg_init`. Compile: no global STAGES/SLAB_HALF/STAGE_BYTES is left anywhere; buckets above NMAX are behind `if constexpr` inside the template, so they are not instantiated; the `#if WG_NEW_VARIANTS` launch chain is well formed for both 0 and 1; the C++ declaration and the definition of `wg_gemm(which, variant, ...)` match the Python call. The `_wg_audit_cpp` emulation finds no globals in either source. kernel_fn branches only on shape, two bools and the int `wg_v128`. It has no pointer gate, and module globals hold no device state.

Fixes:
1. Copy-check order: the new variants now run before variant 0. Before the fix, a stage-6 variant that left an error word in state A's `wg_err` (for example an mbarrier timeout) leaked into the path-wide `err_word` read after the copy check. That turned the whole wgmma path off, B256 included, which is a fallback cascade. Variant 0 zeroes the word before each of its own replays, so its result decides the path-wide flags again.
2. Build fallback: if the -Werror probe fails with no ptxas line (a front-end compile error), the module is no longer rebuilt with the stage-6 variants without -Werror, since that rebuild would hit the same error. The s5-only module is built at once, which saves budget so B256 keeps its bucket-gate decision.

Residual risk: nvcc has not compiled the new template instantiations (this node has no nvcc), and a hard device fault in a new variant during prepare would poison the context. B256's selector now draws different random inputs than the base (the generator is consumed differently at T=128), so its decision, about 1.06x against a 1.02x margin, carries ordinary noise.
