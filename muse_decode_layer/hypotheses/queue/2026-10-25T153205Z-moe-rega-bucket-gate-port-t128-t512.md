---
problem: muse_decode_layer
slug: moe-rega-bucket-gate-port-t128-t512
created: 2026-10-25T153205Z
base: master
area: w13 marlin + silu-mul + w2 marlin (128 <= T <= 512)
change: >
  Stage 1 of a rewrite of decode's expert GEMMs at large T. At T=128/256 (verify_16/verify_32) the
  two Marlin calls are the single largest block of the problem: 275.8/297.2 us (sliding/full
  verify_16) and 442.6/425.8 us (sliding/full verify_32) in the master profile (job
  34a9be06918d448b), 50-70% of those shapes, vs an expert-block floor of ~101/109 us
  (muse_moe_w4a16 verify_16/verify_32 SoL, same 32 local experts at width 4096). Port the swap-AB
  register-A wgmma NVFP4 grouped expert path from muse_moe_w4a16 master (commit 40e2e860c4,
  hypothesis cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, submit 4e4a0e26c85944f2,
  1.3727x; it won the B256 bucket 402 vs 457 us there and ported to prefill at 0.63-0.65x Marlin's
  time). Copy verbatim: the WG_* constants (WG_BSM 16, WG_THREADS 384, WG_STAGES 4, WG_BK 128,
  WG_BUCKETS 16/32/64/128, WG_CAP_ROWS, WG_MAX_BLOCKS), the CUDA TU (wg_gather, wg_kernel<true> =
  w13 + silu into h_sorted, wg_kernel<false> = w2 into cache3 at the route index, the per-CALL
  tensor-map encode into a stack-local grid constant with the driver entry point resolved once in
  prepare; no stored map or pointer record), _wg_attach_scratch (per-state wg_xs / wg_hs /
  wg_err), the bank retile (q13/gs13/q2/gs2), the per-bucket canary that FORCES each N bucket
  (16/32/64/128 routes on one expert) and the three-state copy-safety check (prepared state,
  copy.deepcopy, recursive tensor-cloning walk; refilled graph replays vs an fp32-dequant
  reference; block-0 launch marker in wg_err[1]). Rename the module muse_decode_layer_rega_v1.
  DECODE ADAPTATION: (1) build the rega bank from the ORIGINAL
  weights['w13'/'w13_sf'/'w13_gs'/'w2'/'w2_sf'/'w2_gs'] BEFORE _marlin_bank runs (its loader step
  may repack the same storage in place; or retile from .clone()); the local experts are always
  weights' 32 (local ids 0..31 after the (id - local_start) % 256 shift, -1 remote), so one bank,
  key (0, 32). (2) In kernel_fn's T > DL_MAX_TOKENS branch, keep master's side-stream router,
  routing and _expert_dispatch (align_par at bsm = _block_size_m(T), which is 16 at T >= 128 under
  the cap: assert this in prepare). After main_stream.wait_stream(side_stream), when 128 <= tokens
  <= 512 and this state's flag for the bucket is set (state['rega_on_128'] for 128..255,
  state['rega_on_256'] for 256..512), call a new _experts_rega(xn, tw, local_ids, (sorted_ids,
  block_experts, n_padded), state): wg_gather of xn into wg_xs, wg_kernel<w13+silu> into wg_hs,
  wg_kernel<w2> into cache3 [T*8, 4096], then exactly master's T >= 32 finish
  (fused_post_expert_norm_weighted_sum with post_expert_w, slot_expert_ids=local_ids). Every other
  T and every unset bucket runs master's _experts unchanged. (3) In prepare, verify once that rega
  fed decode's align_par output equals rega fed moe_align_block_size(local_ids, 16, 32) bit for
  bit on skewed, random, all-one-expert and no-local routings at T=128/256 (the two aligns differ
  in capacity only); if not, the rega path calls moe_align_block_size(local_ids, 16, 32) on the
  side stream instead and prints why. (4) Correctness: rel-L2 of the post-combine output against
  an fp32-dequant reference, with master's own (Marlin) error printed beside it; bound max(2x
  master's error, 1e-2); force every N bucket and multi-chunk experts; name the first non-finite
  tensor. (5) SELECTOR (decode's own; do not inherit moe's flags): per bucket at T=128 and T=256,
  time decode's whole expert block (master: align_par + Marlin w13 + act_and_mul + Marlin w2 +
  post-norm, cap-16 bsm) vs the rega block (same align + gather + w13silu + w2 + post-norm) on two
  routings: (a) judge-like, produced by the real router weights and bias through
  FusedMuseSparkRouting on synthetic rmsnormed ffn inputs with local_start in {0, 96, 224}
  (decode's 'dist soft' EP8 routing gives fewer local routes than moe's busiest-GPU routing, so
  moe's B128 verdict does not transfer), and (b) moe's zipf a=0.3 skew; enable a bucket only if
  rega wins by >= 1.02x on both. Print '[rega] PATH CHOSEN B128|B256 = REGA|MARLIN [rega X us vs
  marlin Y us on soft / skew]' with both numbers and the local-route count per routing. (6)
  Budget: decode prepare already builds the dense chain, split-KV and sk modules; run the rega
  setup right after _align_par_setup with its own deadline measured from prepare()'s t_prepare0
  (skip with a fallback line if more than 120 s have elapsed at its start; skip its timing if past
  230 s), and print its compile seconds and the cumulative prepare wall time. Memory: one extra
  NVFP4 bank (~0.5-0.9 GiB) plus T-sized scratch per state. STAGE PLAN: s1 (this) port + decode's
  own bucket gate; s2 bring in whichever of moe's queued rega consumer-side changes lands
  (cuda-nvfp4-wgmma-rega-pingpong-consumers, cuda-nvfp4-wgmma-rega-producer-scale-expand; moe's
  deep-ring run showed the B128 limiter is the consumer/dequant side, not bytes in flight) to flip
  B128 (verify_16); s3 extend down to T=33..127 (the bs=33/40 tests and nothing scored) only if s2
  wins B128; s4 emit the gathered wg_xs rows from the FFN-front norm kernel (with
  ffn-front-t128-t256-concat-route) so wg_gather disappears. Every stage falls back to master per
  bucket through the prepare-time gate. HARD RULES (every path): CUDA-graph capturable; kernel_fn
  has no host sync, no timing, no prints, no try/except, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow and no per-call mutable Python state;
  gates read config/shape and this state's own prepare() flags only; nothing cached from inputs;
  weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at
  most E4M3; every key attended; no child processes. The judge prepares TWO state copies and
  rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in
  a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode
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
    why: incumbent Marlin pair 442.6 us + act_and_mul 6.1 us (profile 34a9be06918d448b); moe measured rega 402 vs Marlin 457 us at T=256 (0.88x); expert-block floor ~109 us. M1 rega block <= incumbent; M2 <= 390 us (shape 600 -> ~550 us); M3 <= 300 us (2.75x floor)
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "same T=256 bucket: Marlin pair 425.8 us; expect -40..-55 us on 881 us"
  - shapes: "*load=verify_16*"
    direction: flat
    why: "B128: moe's selector kept Marlin there (rega 0.80-0.81x); decode's own gate decides on decode-like routing (Marlin pair 275.8/297.2 us). A REGA choice with a lower verify_16 Marlin+act row is a bonus"
  - shapes: "*load=decode_*"
    direction: flat
    why: "T <= 32: outside the gate (dense chain path)"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "T=8: outside the gate"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13 + act_and_mul + Marlin w2 at 128 <= T <= 512 (verify_16 T=128, verify_32 T=256), both layers
review: required
status: hold
hold_reason: "superseded: decode's own selector switched rega OFF at T=128 (384.8 vs Marlin 327.2 us) and T=256 (454.6 vs 440.0) on 2026-10-25; dead in decode (decode minter, 2026-10-25)"
---

# moe-rega-bucket-gate-port-t128-t512

## Change

Stage 1 of a rewrite of decode's expert GEMMs at large T. At T=128/256 (verify_16/verify_32) the two Marlin calls are the single largest block of the problem: 275.8/297.2 us (sliding/full verify_16) and 442.6/425.8 us (sliding/full verify_32) in the master profile (job 34a9be06918d448b), 50-70% of those shapes, vs an expert-block floor of ~101/109 us (muse_moe_w4a16 verify_16/verify_32 SoL, same 32 local experts at width 4096). Port the swap-AB register-A wgmma NVFP4 grouped expert path from muse_moe_w4a16 master (commit 40e2e860c4, hypothesis cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, submit 4e4a0e26c85944f2, 1.3727x; it won the B256 bucket 402 vs 457 us there and ported to prefill at 0.63-0.65x Marlin's time). Copy verbatim: the WG_* constants (WG_BSM 16, WG_THREADS 384, WG_STAGES 4, WG_BK 128, WG_BUCKETS 16/32/64/128, WG_CAP_ROWS, WG_MAX_BLOCKS), the CUDA TU (wg_gather, wg_kernel<true> = w13 + silu into h_sorted, wg_kernel<false> = w2 into cache3 at the route index, the per-CALL tensor-map encode into a stack-local grid constant with the driver entry point resolved once in prepare; no stored map or pointer record), _wg_attach_scratch (per-state wg_xs / wg_hs / wg_err), the bank retile (q13/gs13/q2/gs2), the per-bucket canary that FORCES each N bucket (16/32/64/128 routes on one expert) and the three-state copy-safety check (prepared state, copy.deepcopy, recursive tensor-cloning walk; refilled graph replays vs an fp32-dequant reference; block-0 launch marker in wg_err[1]). Rename the module muse_decode_layer_rega_v1. DECODE ADAPTATION: (1) build the rega bank from the ORIGINAL weights['w13'/'w13_sf'/'w13_gs'/'w2'/'w2_sf'/'w2_gs'] BEFORE _marlin_bank runs (its loader step may repack the same storage in place; or retile from .clone()); the local experts are always weights' 32 (local ids 0..31 after the (id - local_start) % 256 shift, -1 remote), so one bank, key (0, 32). (2) In kernel_fn's T > DL_MAX_TOKENS branch, keep master's side-stream router, routing and _expert_dispatch (align_par at bsm = _block_size_m(T), which is 16 at T >= 128 under the cap: assert this in prepare). After main_stream.wait_stream(side_stream), when 128 <= tokens <= 512 and this state's flag for the bucket is set (state['rega_on_128'] for 128..255, state['rega_on_256'] for 256..512), call a new _experts_rega(xn, tw, local_ids, (sorted_ids, block_experts, n_padded), state): wg_gather of xn into wg_xs, wg_kernel<w13+silu> into wg_hs, wg_kernel<w2> into cache3 [T*8, 4096], then exactly master's T >= 32 finish (fused_post_expert_norm_weighted_sum with post_expert_w, slot_expert_ids=local_ids). Every other T and every unset bucket runs master's _experts unchanged. (3) In prepare, verify once that rega fed decode's align_par output equals rega fed moe_align_block_size(local_ids, 16, 32) bit for bit on skewed, random, all-one-expert and no-local routings at T=128/256 (the two aligns differ in capacity only); if not, the rega path calls moe_align_block_size(local_ids, 16, 32) on the side stream instead and prints why. (4) Correctness: rel-L2 of the post-combine output against an fp32-dequant reference, with master's own (Marlin) error printed beside it; bound max(2x master's error, 1e-2); force every N bucket and multi-chunk experts; name the first non-finite tensor. (5) SELECTOR (decode's own; do not inherit moe's flags): per bucket at T=128 and T=256, time decode's whole expert block (master: align_par + Marlin w13 + act_and_mul + Marlin w2 + post-norm, cap-16 bsm) vs the rega block (same align + gather + w13silu + w2 + post-norm) on two routings: (a) judge-like, produced by the real router weights and bias through FusedMuseSparkRouting on synthetic rmsnormed ffn inputs with local_start in {0, 96, 224} (decode's 'dist soft' EP8 routing gives fewer local routes than moe's busiest-GPU routing, so moe's B128 verdict does not transfer), and (b) moe's zipf a=0.3 skew; enable a bucket only if rega wins by >= 1.02x on both. Print '[rega] PATH CHOSEN B128|B256 = REGA|MARLIN [rega X us vs marlin Y us on soft / skew]' with both numbers and the local-route count per routing. (6) Budget: decode prepare already builds the dense chain, split-KV and sk modules; run the rega setup right after _align_par_setup with its own deadline measured from prepare()'s t_prepare0 (skip with a fallback line if more than 120 s have elapsed at its start; skip its timing if past 230 s), and print its compile seconds and the cumulative prepare wall time. Memory: one extra NVFP4 bank (~0.5-0.9 GiB) plus T-sized scratch per state. STAGE PLAN: s1 (this) port + decode's own bucket gate; s2 bring in whichever of moe's queued rega consumer-side changes lands (cuda-nvfp4-wgmma-rega-pingpong-consumers, cuda-nvfp4-wgmma-rega-producer-scale-expand; moe's deep-ring run showed the B128 limiter is the consumer/dequant side, not bytes in flight) to flip B128 (verify_16); s3 extend down to T=33..127 (the bs=33/40 tests and nothing scored) only if s2 wins B128; s4 emit the gathered wg_xs rows from the FFN-front norm kernel (with ffn-front-t128-t256-concat-route) so wg_gather disappears. Every stage falls back to master per bucket through the prepare-time gate. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=verify_32*` faster: incumbent Marlin pair 442.6 us + act_and_mul 6.1 us (profile 34a9be06918d448b); moe measured rega 402 vs Marlin 457 us at T=256 (0.88x); expert-block floor ~109 us. M1 rega block <= incumbent; M2 <= 390 us (shape 600 -> ~550 us); M3 <= 300 us (2.75x floor)
- `*layer=full load=verify_32*` faster: same T=256 bucket: Marlin pair 425.8 us; expect -40..-55 us on 881 us
- `*load=verify_16*` flat: B128: moe's selector kept Marlin there (rega 0.80-0.81x); decode's own gate decides on decode-like routing (Marlin pair 275.8/297.2 us). A REGA choice with a lower verify_16 Marlin+act row is a bonus
- `*load=decode_*` flat: T <= 32: outside the gate (dense chain path)
- `*load=verify_1 *` flat: T=8: outside the gate

## Risk

Weakest link: decode's local routes per T are fewer than moe's (dist soft, not busiest-GPU), so the B256 win may shrink; the gate then keeps Marlin and the run re-measures master plus a dead path (still informative: the PATH CHOSEN timings give the per-bucket ratio on decode routing). Second risk: the prepare budget (four load_inline builds); the deadline guard turns rega off rather than time out. Third: the in-place Marlin repack aliasing the original NVFP4 storage; build the rega bank first. Check the log for the build line, '[rega] PATH CHOSEN' per bucket, and the wg_kernel rows in the verify_32 profile before reading any number; a log without PATH CHOSEN = REGA for B256 means the bucket is unmeasured.

## Milestones

Rewrite stage 1 of `Marlin w13 + act_and_mul + Marlin w2 at 128 <= T <= 512 (verify_16 T=128, verify_32 T=256), both layers`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
