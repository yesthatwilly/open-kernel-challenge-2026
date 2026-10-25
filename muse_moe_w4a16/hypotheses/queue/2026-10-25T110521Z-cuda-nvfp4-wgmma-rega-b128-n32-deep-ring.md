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
status: queued
review_at: 2026-10-25T120207Z
review_companions: null
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

