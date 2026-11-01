---
problem: muse_decode_layer
slug: ffn-front-t128-t256-concat-route
created: 2026-10-25T153205Z
base: master
area: pooler GEMM + router GEMM + routing top-k + MoE align + pre-expert norm (T > 32)
change: >
  Stage 1 of a rewrite of the FFN front at T > DL_MAX_TOKENS (the dense chain already owns T <= 32
  with kernels P and R). Master at T=128/256 runs: the pooler GEMM (cuBLAS nvjet, main stream)
  overlapped with a side-stream branch of two router GEMMs (torch.mm hi and lo, out fp32: the 'tss
  ... splitK_NNT' row, 10.8 us at sliding verify_16, 13.5 us sliding verify_32 and 26.5 us at full
  verify_32, 2 calls) + their splitKreduce nodes + an fp32 add + FusedMuseSparkRouting +
  _align_parallel_kernel (6.2 us at T=256), then F.rms_norm on the main stream
  (vectorized_layer_norm 5.6-5.9 us) and the join. The router branch is as long as or longer than
  the pooler at T=256 (and it steals bandwidth from it), so the MoE block waits on it. Floor for
  the whole front: 75.5 MB of pooler + router hi/lo weights -> 15.7 us at 4.8 TB/s, or
  2*256*4608*8192 FLOP -> 19.5 us at T=256. NEW PATH (gate: config tokens > DL_MAX_TOKENS and
  tokens <= 512 and this state's flag state['ff1_on_<bucket>']): (1) prepare() builds a row-major
  bf16 W_cat [4608, 8192] = rows 0..4095 the pooler, 4096..4351 router_hi (bf16(router)),
  4352..4607 router_lo (bf16(router - hi)); a lossless repack (the rules allow the router as a
  BF16 hi + lo pair with fp32 accumulation, as SGLang does above 8 tokens). (2) ONE GEMM: cat =
  torch.mm(ffn_input, W_cat.t(), out_dtype=torch.float32) -> [T, 4608] fp32 on the main stream (no
  side stream, no fork). (3) ONE Triton kernel 'ff1_route_norm', one program per token: logits[e]
  = cat[t, 4096+e] + cat[t, 4352+e] (fp32, the same hi+lo sum master does), scores =
  sigmoid(0.1767766952966369 * logits) and top-8 by (score + router_bias) with exactly
  FusedMuseSparkRouting's renormalisation/route_scale and tie order (copy dense-chain kernel R's
  top-k semantics, which already passed bit-identity against FusedMuseSparkRouting); write
  topk_ids int32, topk_weights fp32, local ids ((id - local_start) % 256 if < 32 else -1); and the
  pre-expert RMSNorm of the pooler slice rounded to bf16 FIRST (bf16(cat[t, :4096]) then
  MuseSpark1xRMSNorm with pre_expert_norm weight, eps 1e-5, fp32 math, bf16 out), so the
  activations stay BF16 exactly where SGLang keeps them. (4) The align: prefill master's two-phase
  align (hypothesis align-two-phase-all-t-rega: count + prefix-place, bit-identical to
  moe_align_block_size, 69 -> 9 us at T=3012) at bsm = _block_size_m(T) over the 32 local experts,
  replacing _align_parallel; then master's _experts(xn, None, None, state, dispatch=(local_ids,
  tw, align)) unchanged. Launches: GEMM + route_norm + 2 align = 4 nodes on one stream vs ~10
  nodes on two. SELF-CHECK in prepare at T in {33, 40, 128, 256, 512}: ids/weights/local ids/align
  bit-identical to master's side-stream path on the same inputs except at near ties (allowed set
  as TASK.md: picked scores within 0.05 of the 8th best), xn rel-L2 <= 1e-3 vs master's
  pre_expert_norm(expert_input), and a whole-layer comparison under alternating-state replays.
  SELECTOR per bucket (T=128 for 33..255, T=256 for 256..512): the whole FFN front graph (boundary
  output -> xn + dispatch ready) new vs master (with its fork/join) by the standard interleaved
  graph timing; also print each piece (cat GEMM, route_norm, align; master pooler, router branch,
  norm) in CUDA events. STAGE PLAN: s1 (this) cuBLAS cat GEMM + fused route/norm + two-phase
  align; s2 fold the post-attention boundary (FusedPostNormResidualPreNorm, 7.4-8.9 us at T=256)
  into a prologue of the route kernel's producer, or the cat GEMM's A-operand producer; s3 hand
  rega (rewrite moe-rega-bucket-gate-port-t128-t512) its gathered wg_xs rows directly from
  ff1_route_norm. Every stage falls back per bucket to master's path through the prepare-time
  gate. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no
  prints, no try/except, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow and no per-call mutable Python state; gates read config/shape
  and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4
  stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key
  attended; no child processes. The judge prepares TWO state copies and rotates them: keep every
  device pointer, TMA map, ticket array and workspace per state, never in a module global. Every
  new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode
  arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an
  in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any
  miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check
  by capturing once and replaying with refilled inputs while alternating two states. Selectors
  time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps,
  empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS
  by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per
  module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First
  measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: "replaced: router hi/lo 13.5 us + 2 splitKreduce + add + routing + align_par 6.2 + layer_norm 5.9 + fork/join, the router branch on the critical path; expect -10..-16 us on 600 us. M1 new front (cat GEMM + route_norm + align) <= master front in the prepare graph timing; M2 front <= 30 us at T=256 (floor 19.5)"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: router row is 26.5 us here (2 calls); expect -12..-20 us on 881 us
  - shapes: "*load=verify_16*"
    direction: faster
    why: "T=128: router 10.8-11.0 us + reduces + routing + align + layer_norm 5.6-5.9; expect -8..-12 us on 427/572 us; M2 front <= 24 us (floor 15.7)"
  - shapes: "*load=decode_*"
    direction: flat
    why: "T <= 32: dense chain P/R, untouched"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "T=8: dense chain, untouched"
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: pooler nvjet + side-stream router hi/lo tss splitK x2 + splitKreduce x2 + fp32 add + FusedMuseSparkRouting + _align_parallel + vectorized_layer_norm (pre-expert RMSNorm) + the side-stream fork/join, at 33 <= T <= 512 (verify_16 T=128, verify_32 T=256)
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: w6b-t128diet cells landed"
---

# ffn-front-t128-t256-concat-route

## Change

Stage 1 of a rewrite of the FFN front at T > DL_MAX_TOKENS (the dense chain already owns T <= 32 with kernels P and R). Master at T=128/256 runs: the pooler GEMM (cuBLAS nvjet, main stream) overlapped with a side-stream branch of two router GEMMs (torch.mm hi and lo, out fp32: the 'tss ... splitK_NNT' row, 10.8 us at sliding verify_16, 13.5 us sliding verify_32 and 26.5 us at full verify_32, 2 calls) + their splitKreduce nodes + an fp32 add + FusedMuseSparkRouting + _align_parallel_kernel (6.2 us at T=256), then F.rms_norm on the main stream (vectorized_layer_norm 5.6-5.9 us) and the join. The router branch is as long as or longer than the pooler at T=256 (and it steals bandwidth from it), so the MoE block waits on it. Floor for the whole front: 75.5 MB of pooler + router hi/lo weights -> 15.7 us at 4.8 TB/s, or 2*256*4608*8192 FLOP -> 19.5 us at T=256. NEW PATH (gate: config tokens > DL_MAX_TOKENS and tokens <= 512 and this state's flag state['ff1_on_<bucket>']): (1) prepare() builds a row-major bf16 W_cat [4608, 8192] = rows 0..4095 the pooler, 4096..4351 router_hi (bf16(router)), 4352..4607 router_lo (bf16(router - hi)); a lossless repack (the rules allow the router as a BF16 hi + lo pair with fp32 accumulation, as SGLang does above 8 tokens). (2) ONE GEMM: cat = torch.mm(ffn_input, W_cat.t(), out_dtype=torch.float32) -> [T, 4608] fp32 on the main stream (no side stream, no fork). (3) ONE Triton kernel 'ff1_route_norm', one program per token: logits[e] = cat[t, 4096+e] + cat[t, 4352+e] (fp32, the same hi+lo sum master does), scores = sigmoid(0.1767766952966369 * logits) and top-8 by (score + router_bias) with exactly FusedMuseSparkRouting's renormalisation/route_scale and tie order (copy dense-chain kernel R's top-k semantics, which already passed bit-identity against FusedMuseSparkRouting); write topk_ids int32, topk_weights fp32, local ids ((id - local_start) % 256 if < 32 else -1); and the pre-expert RMSNorm of the pooler slice rounded to bf16 FIRST (bf16(cat[t, :4096]) then MuseSpark1xRMSNorm with pre_expert_norm weight, eps 1e-5, fp32 math, bf16 out), so the activations stay BF16 exactly where SGLang keeps them. (4) The align: prefill master's two-phase align (hypothesis align-two-phase-all-t-rega: count + prefix-place, bit-identical to moe_align_block_size, 69 -> 9 us at T=3012) at bsm = _block_size_m(T) over the 32 local experts, replacing _align_parallel; then master's _experts(xn, None, None, state, dispatch=(local_ids, tw, align)) unchanged. Launches: GEMM + route_norm + 2 align = 4 nodes on one stream vs ~10 nodes on two. SELF-CHECK in prepare at T in {33, 40, 128, 256, 512}: ids/weights/local ids/align bit-identical to master's side-stream path on the same inputs except at near ties (allowed set as TASK.md: picked scores within 0.05 of the 8th best), xn rel-L2 <= 1e-3 vs master's pre_expert_norm(expert_input), and a whole-layer comparison under alternating-state replays. SELECTOR per bucket (T=128 for 33..255, T=256 for 256..512): the whole FFN front graph (boundary output -> xn + dispatch ready) new vs master (with its fork/join) by the standard interleaved graph timing; also print each piece (cat GEMM, route_norm, align; master pooler, router branch, norm) in CUDA events. STAGE PLAN: s1 (this) cuBLAS cat GEMM + fused route/norm + two-phase align; s2 fold the post-attention boundary (FusedPostNormResidualPreNorm, 7.4-8.9 us at T=256) into a prologue of the route kernel's producer, or the cat GEMM's A-operand producer; s3 hand rega (rewrite moe-rega-bucket-gate-port-t128-t512) its gathered wg_xs rows directly from ff1_route_norm. Every stage falls back per bucket to master's path through the prepare-time gate. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=verify_32*` faster: replaced: router hi/lo 13.5 us + 2 splitKreduce + add + routing + align_par 6.2 + layer_norm 5.9 + fork/join, the router branch on the critical path; expect -10..-16 us on 600 us. M1 new front (cat GEMM + route_norm + align) <= master front in the prepare graph timing; M2 front <= 30 us at T=256 (floor 19.5)
- `*layer=full load=verify_32*` faster: router row is 26.5 us here (2 calls); expect -12..-20 us on 881 us
- `*load=verify_16*` faster: T=128: router 10.8-11.0 us + reduces + routing + align + layer_norm 5.6-5.9; expect -8..-12 us on 427/572 us; M2 front <= 24 us (floor 15.7)
- `*load=decode_*` flat: T <= 32: dense chain P/R, untouched
- `*load=verify_1 *` flat: T=8: dense chain, untouched

## Risk

Weakest link: cuBLAS on the [T, 8192] x [8192, 4608] fp32-out GEMM may pick a slower kernel than the bf16-out pooler alone (fp32 epilogue writes 4.7 MB at T=256, negligible, but the algo choice can differ); the per-piece CUDA-event print separates this from the route kernel, and the gate keeps master if the front loses. Routing tie order must match FusedMuseSparkRouting closely enough for TASK.md's routing checks; reuse kernel R's verified top-k semantics rather than writing new ones. Not a duplicate of router-triton-splitk-side-stream (held: at T <= 32 the router was off the critical path; this removes the side stream at T > 32, where the router row is 10.8-26.5 us) nor of pre-expert-rmsnorm-triton (held: stale at T <= 32; here the norm is fused into the routing kernel at T > 32 only).

## Milestones

Rewrite stage 1 of `pooler nvjet + side-stream router hi/lo tss splitK x2 + splitKreduce x2 + fp32 add + FusedMuseSparkRouting + _align_parallel + vectorized_layer_norm (pre-expert RMSNorm) + the side-stream fork/join, at 33 <= T <= 512 (verify_16 T=128, verify_32 T=256)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
