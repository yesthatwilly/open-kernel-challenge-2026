---
problem: muse_prefill_layer
slug: router-triton-hilo-splitk-turn
created: 2026-10-24T075454Z
base: master
area: router GEMM
change: >
  For tokens <= 4096 only, replace the router's two cuBLAS calls on the side stream (logits =
  torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t,
  out_dtype=fp32)) with one Triton launch ported from master's muse_bf16_proj/submission.py
  (_router_hilo_kernel / _router_forward): grid (cdiv(M, BLOCK_M), 256 // BLOCK_N, SPLIT_K), each
  program loads its x tile once per BLOCK_K step and does acc = tl.dot(x, hi, acc); acc =
  tl.dot(x, lo, acc) in fp32, then tl.atomic_add (sem relaxed) of its K-partial into an fp32 [M,
  256] output zeroed with torch.zeros right before the launch, all inside the existing
  torch.cuda.stream(side_stream) block. prepare() additionally keeps hi and lo as [256, 8192]
  row-major bf16 (state router_hi / router_lo, the same split as today, untransposed). Config for
  this M range, from tokens only: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64, num_warps 8, num_stages 3,
  SPLIT_K 16 for M <= 1024 (6 x 2 x 16 = 192 CTAs at M=753, K-slice 512) and SPLIT_K 4 for 1024 <
  M <= 4096 (24 x 2 x 4 = 192 CTAs at M=3012, K-slice 2048); do_not_specialize M; prepare()
  pre-compiles both buckets on synthetic bf16 x so the timed call never compiles. For tokens >
  4096 (first_turn) the cuBLAS pair stays: there it runs as nvjet 128x256 at ~485 TFLOP/s (283 us
  for 2 calls on 137 GFLOP, profile 62025c368c3c4e8e) on a side branch far shorter than the 1.4 ms
  pooler, so nothing is gained. FusedMuseSparkRouting, the stream fork/join and everything else
  are unchanged. Why at these M: cuBLAS is inefficient here: at M=753 the pair runs as split-K
  nvjet 64x96 (33-40 us) plus cublasLt splitKreduce (45.6 us on turn_1 full) for 6.3 GFLOP (~80
  TFLOP/s), and the moe-align-side-stream review found this side branch (~80-90 us incl. routing)
  exceeds the main branch it overlaps (pooler ~40-50 us + pre-expert norm 7 us), so it is on the
  critical path; at M=3012 it is nvjet 64x104 at 132 us for 25 GFLOP (~190 TFLOP/s), concurrent
  with the pooler and sharing its SMs.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "side branch drops from ~80-90 us to ~25-30 us (one launch + memset, ~6 GFLOP at ~300 TFLOP/s), below the ~50 us main branch, so up to ~40 us leaves the critical path: ~2-3% of 1079/1435 us; the bound is P+N-S, smaller where cuBLAS skipped the splitKreduce"
  - shapes: "*load=turn_4*"
    direction: faster
    why: router 132 us -> ~80 us running beside the pooler; fewer SM-cycles stolen from it; ~0.5-1.5%, likely inside noise
  - shapes: "*load=first_turn*"
    direction: flat
    why: cuBLAS pair kept above 4096 tokens (control)
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: hold
review_at: 2026-10-24T104017Z
review_companions: null
hold_reason: "reviewer: the side-stream router (~40-45 us) already fits under the main branch (the 39.3 us nvjet is qkv_gate, not the pooler), expected 0.3-1% inside noise; prefill has 2 checks and the rewrite's next stage needs them"
---

# router-triton-hilo-splitk-turn

## Change

For tokens <= 4096 only, replace the router's two cuBLAS calls on the side stream (logits = torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t, out_dtype=fp32)) with one Triton launch ported from master's muse_bf16_proj/submission.py (_router_hilo_kernel / _router_forward): grid (cdiv(M, BLOCK_M), 256 // BLOCK_N, SPLIT_K), each program loads its x tile once per BLOCK_K step and does acc = tl.dot(x, hi, acc); acc = tl.dot(x, lo, acc) in fp32, then tl.atomic_add (sem relaxed) of its K-partial into an fp32 [M, 256] output zeroed with torch.zeros right before the launch, all inside the existing torch.cuda.stream(side_stream) block. prepare() additionally keeps hi and lo as [256, 8192] row-major bf16 (state router_hi / router_lo, the same split as today, untransposed). Config for this M range, from tokens only: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64, num_warps 8, num_stages 3, SPLIT_K 16 for M <= 1024 (6 x 2 x 16 = 192 CTAs at M=753, K-slice 512) and SPLIT_K 4 for 1024 < M <= 4096 (24 x 2 x 4 = 192 CTAs at M=3012, K-slice 2048); do_not_specialize M; prepare() pre-compiles both buckets on synthetic bf16 x so the timed call never compiles. For tokens > 4096 (first_turn) the cuBLAS pair stays: there it runs as nvjet 128x256 at ~485 TFLOP/s (283 us for 2 calls on 137 GFLOP, profile 62025c368c3c4e8e) on a side branch far shorter than the 1.4 ms pooler, so nothing is gained. FusedMuseSparkRouting, the stream fork/join and everything else are unchanged. Why at these M: cuBLAS is inefficient here: at M=753 the pair runs as split-K nvjet 64x96 (33-40 us) plus cublasLt splitKreduce (45.6 us on turn_1 full) for 6.3 GFLOP (~80 TFLOP/s), and the moe-align-side-stream review found this side branch (~80-90 us incl. routing) exceeds the main branch it overlaps (pooler ~40-50 us + pre-expert norm 7 us), so it is on the critical path; at M=3012 it is nvjet 64x104 at 132 us for 25 GFLOP (~190 TFLOP/s), concurrent with the pooler and sharing its SMs.

## Expected

- `*load=turn_1*` faster: side branch drops from ~80-90 us to ~25-30 us (one launch + memset, ~6 GFLOP at ~300 TFLOP/s), below the ~50 us main branch, so up to ~40 us leaves the critical path: ~2-3% of 1079/1435 us; the bound is P+N-S, smaller where cuBLAS skipped the splitKreduce
- `*load=turn_4*` faster: router 132 us -> ~80 us running beside the pooler; fewer SM-cycles stolen from it; ~0.5-1.5%, likely inside noise
- `*load=first_turn*` flat: cuBLAS pair kept above 4096 tokens (control)

## Risk

Expected effect is small (2-3% on turn_1 shapes, at the edge of the +-1.5% interval); queued because it is the one lever on the turn_1 side-branch limiter the moe-align-side-stream review named, and it reuses the one big proven kernel (bf16_proj 1.151x). Atomic fp32 accumulation order is nondeterministic run to run; routing tolerance (0.05 of the 8th best, near-tie slack) absorbs fp32-ulp differences, as it did on bf16_proj and decode. If experts-marlin-block-m-* lands first and shrinks Marlin, turn_1 becomes more sensitive to this branch, so the gain may grow when stacked. torch.zeros adds one memset node on the side stream.

## Adversarial review

Reviewed 2026-10-24T104017Z by an adversarial subagent (model opus).

**Verdict: proceed.** Rule-clean, well-scoped port of the proven bf16_proj split-K hi/lo router, but the premise that the router branch is on the turn_1 critical path rests on misidentifying the pooler as the 39 us nvjet; the pooler is in the 128x192 group (~55-70 us), so the side branch most likely already fits under it and the realistic headline effect is SM-contention relief of ~0.3-1%, inside the +/-1.5% interval. Run it for the router kernel's own time (readable from top_kernels), not for the headline, and given 2 checks left it should not displace the in-flight rewrite's next stage.

### Justification
Mechanism: a 3-D split-K Triton GEMM with fp32 atomic_add into a zeroed output and hi+lo dots on the same x tile is sound on H200 and was the one large bf16 win (1.151x in muse_bf16_proj, LESSONS). On cuBLAS's own time the case is plausible: at M=753 the pair runs as nvjet_tss_64x96 splitK (33.2 us sliding / 39.6 us full, 2 calls) for 6.3 GFLOP (~160-190 TFLOP/s), and at M=3012 as nvjet_tss_64x104 (132.4 / 134.9 us, 2 calls) for 25.3 GFLOP (~190 TFLOP/s), so a Triton kernel at 300-450 TFLOP/s would cut the router to ~15-25 us and ~55-85 us respectively. The critical-path argument is the weak part. (1) The pooler is not the 39 us kernel: nvjet_112x128 is a single call and 50.5 GFLOP (753x4096x8192x2) in 39.3 us would be 1.28 PFLOP/s, above the 989 TFLOP/s peak; it is the qkv_gate GEMM (N=2304, 28.4 GFLOP, ~720 TFLOP/s). The pooler is one of the three nvjet_128x192 calls (o, pooler, unpooler: 113.6 GFLOP in 157-162 us), i.e. ~60-70 us while sharing SMs with the router, plus the 6.6 us pre-expert norm: a main branch of ~65-80 us. (2) The side branch on turn_1 sliding is 33.2 us of router GEMM plus a splitKreduce and FusedMuseSparkRouting that are both below the 5.1 us top-12 cut-off: ~40-45 us. So the side branch is very likely already shorter than the main branch, which is the opposite of the moe-align-side-stream review's premise that this hypothesis inherits. (3) The 45.6 us splitKreduce on turn_1 full (absent from the sliding top-12 for identical M and identical router code) is almost certainly wait time folded into an overlapping kernel's row (the profile says so explicitly), not real reduce work; it cannot be used as the size of the prize. What remains is contention: both branches are wave-filling (pooler 128x192 tiles: 6x22=132 CTAs at M=753; the Triton config is 192 CTAs at ~144 KB smem, 1 CTA/SM, two partial waves), so the headline moves only by the SM-time the router stops stealing from the pooler, roughly the difference in router SM-time: ~10-20 us at turn_1 (~1% of 1079/1435 us) and ~50-80 us at turn_4 (~1-1.5% of 3367/4950 us). Note turn_4 may actually be the better load, not turn_1, contrary to the expected block. first_turn flat is right (cuBLAS kept above 4096). Directions are consistent with TASK.md weights (turn_1 and turn_4 sliding carry w46 each, so a ~1% per-shape move is ~0.3-0.5% on the geomean after the first_turn load's flat 1/3), but magnitudes land inside the interval. Second weak link: the config (128x128x64, 8 warps, 3 stages, SPLIT_K 16/4) is not the measured one (bf16_proj ran 64x64x128 at m<=256, ~128 CTAs). At SPLIT_K 16 each CTA has only 8 BLOCK_K steps of a 3-stage pipeline, so prologue/epilogue and 16 x 753 x 256 fp32 atomics (~12 MB of L2 atomic traffic) are a real fraction of the work; the 'ported, proven' claim covers the kernel skeleton, not this operating point. Weakest link overall: the pooler misidentification, which turns 'removes ~40 us from the critical path' into 'reduces SM contention by ~10-20 us'.

### Confounders on the current kernel
- sglang-pipeline-parity: it is the base (the whole file against the plain-PyTorch starter), not a separable confounder. Its side-stream fork placing the router beside the pooler is exactly what decides whether a faster router shows in the headline (critical path vs contention). The change would behave the same with any base that keeps the fork; reverting the patch removes the fork, Marlin and the router path together, so there is nothing to A/B against.

### Interference and A/B needs
No companion is warranted. ab-revert has no target: the only landed hypothesis is the parity base, and reversing muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.patch returns to the starter with no side stream. ab-tune has no meaningful knob on the landed side (moving the router to the main stream is the sibling router-pooler-concat-gemm-turn's territory and would change the comparison point). A control is also not needed because the result is attributable without one: the replaced kernel's own time is directly readable in top_kernels (the two nvjet_tss splitK rows and the splitKreduce disappear and one _router_hilo_kernel row appears; Triton times reproduce within ~1%), and the 4096-token gate leaves first_turn as a built-in control. A per-shape gain on turn_1/turn_4 with a faster router row is the change; a loss with a faster router row is contention of a 192-CTA, 1-CTA/SM Triton kernel against the pooler (read the pooler's 128x192 row for inflation); a loss with a slower router row is the untested config. With 2 checks left today, spending a second on a companion that can only resolve sub-1% effects is not justified.

### Open questions
- Which nvjet_128x192 call is the pooler and how long is it while the router runs beside it? A per-call breakdown (or the profile's json) would settle whether the side branch is on the turn_1 critical path at all; if it is not, the expected block's turn_1 'up to ~40 us' should be read as at most ~10-20 us.
- Is the 45.6 us splitKreduce on turn_1 full real work or wait time? It is absent from turn_1 sliding's top 12 for identical M and identical router code, which points to wait.
- Is 128x128x64 / 8 warps / 3 stages with SPLIT_K 16 (8 K-steps per CTA) better than the measured bf16_proj shape scaled up (64x64x128 or 64x128x128, ~128-192 CTAs)? Only the router row's own time answers it; if the row is not under ~25 us at M=753, the config, not the idea, is the problem.
- Should this run before router-pooler-concat-gemm-turn? They target the same branch and do not stack; this one is the cheaper, lower-risk read of whether the router branch matters at all, and its router-row time sizes the concat's best case. Given prefill's 2 remaining checks today and the in-flight swapab rewrite, consider deferring both router hypotheses to tomorrow's quota unless the rewrite's stage 2 is not ready.

