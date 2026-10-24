---
problem: muse_decode_layer
slug: bf16-m32-swapab-wgmma-port
created: 2026-10-24T165242Z
base: master
area: pooler GEMM
change: >
  Port muse_bf16_proj bf16-m32-swapab-wgmma-triton (job 1602a4e947e64090, 1.198x there; per layer
  at m=32: pooler 21.55 -> 20.00 us, qkv_gate 14.13 -> 13.69, o 7.04 -> 6.67, unpooler 18.89 ->
  18.48) to decode_layer dense GEMMs at 17 <= T <= 32, which in the scored set is decode_32 (T=32)
  only. Copy verbatim from muse_bf16_proj/submission.py on master:
  SWAPAB_MIN_M/MAX_M/BLOCK_W/BLOCK_T/NUM_WARPS, SWAPAB_CFG (pooler (4096, 8192, split-K 2, BLOCK_K
  256, 4 stages), qkv_gate (2304, 8192, 4, 128, 4), unpooler (8192, 4096, 1, 256, 4), o (8192,
  1024, 1, 256, 4)), SWAPAB_FALLBACK_STAGES, the OutOfResources import, _bf16_swapab_kernel (64
  weight rows on tl.dot M, tokens on N so Triton emits wgmma m64n32k16; fp32 partials + one
  gpu-scope acq_rel ticket per N-tile, last ticket sums in fixed order and self-resets),
  _swapab_forward and _swapab_prepare (adapted only to read weights from state["dense"][name] and
  keep its workspaces/tickets under its own state keys). In kernel_fn route, by tokens and weight
  shape only: qkv = _swapab_forward("qkv_gate", ...) for attn_input, o GEMM = _swapab_forward("o",
  ...), pooler (main stream, still overlapping the side-stream router) = _swapab_forward("pooler",
  ...), and the full-width unpooler (T > 8) = _swapab_forward("unpooler", ...), each when 17 <=
  tokens <= 32, else master _gemm. prepare() calls _swapab_prepare, which allocates the pooler ws
  [2,32,4096] / cnt [64] and qkv ws [4,32,2304] / cnt [36], picks 4 or 3 stages per layer, warms
  every T in 17..32 for each layer on synthetic bf16 inputs, synchronises and zero_()s every cnt.
  The router (side stream, hi/lo torch.mm), all T <= 16 and T >= 128 GEMMs, attention and MoE are
  untouched.
expected:
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "bf16_proj measured -2.8 us of kernels across the four layers at m=32, and each replaced cuBLAS split-K GEMM drops its splitKreduce node (decode_32 profile: splitKreduce 4.0 + 3.8 us rows, ~2-4 us per nvjet+reduce pair in LESSONS), so ~4-7 us off 252 us (sliding) and 536 us (full): +1.5-2.5% sliding, ~+1% full, ~+1.5% on the decode_32 load"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate
  - shapes: "*load=decode_16 *"
    direction: flat
    why: T=16 outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: T=8/128/256 outside the gate; control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T165641Z
review_companions: null
---

# bf16-m32-swapab-wgmma-port

## Change

Port muse_bf16_proj bf16-m32-swapab-wgmma-triton (job 1602a4e947e64090, 1.198x there; per layer at m=32: pooler 21.55 -> 20.00 us, qkv_gate 14.13 -> 13.69, o 7.04 -> 6.67, unpooler 18.89 -> 18.48) to decode_layer dense GEMMs at 17 <= T <= 32, which in the scored set is decode_32 (T=32) only. Copy verbatim from muse_bf16_proj/submission.py on master: SWAPAB_MIN_M/MAX_M/BLOCK_W/BLOCK_T/NUM_WARPS, SWAPAB_CFG (pooler (4096, 8192, split-K 2, BLOCK_K 256, 4 stages), qkv_gate (2304, 8192, 4, 128, 4), unpooler (8192, 4096, 1, 256, 4), o (8192, 1024, 1, 256, 4)), SWAPAB_FALLBACK_STAGES, the OutOfResources import, _bf16_swapab_kernel (64 weight rows on tl.dot M, tokens on N so Triton emits wgmma m64n32k16; fp32 partials + one gpu-scope acq_rel ticket per N-tile, last ticket sums in fixed order and self-resets), _swapab_forward and _swapab_prepare (adapted only to read weights from state["dense"][name] and keep its workspaces/tickets under its own state keys). In kernel_fn route, by tokens and weight shape only: qkv = _swapab_forward("qkv_gate", ...) for attn_input, o GEMM = _swapab_forward("o", ...), pooler (main stream, still overlapping the side-stream router) = _swapab_forward("pooler", ...), and the full-width unpooler (T > 8) = _swapab_forward("unpooler", ...), each when 17 <= tokens <= 32, else master _gemm. prepare() calls _swapab_prepare, which allocates the pooler ws [2,32,4096] / cnt [64] and qkv ws [4,32,2304] / cnt [36], picks 4 or 3 stages per layer, warms every T in 17..32 for each layer on synthetic bf16 inputs, synchronises and zero_()s every cnt. The router (side stream, hi/lo torch.mm), all T <= 16 and T >= 128 GEMMs, attention and MoE are untouched.

## Expected

- `*load=decode_32 *` faster: bf16_proj measured -2.8 us of kernels across the four layers at m=32, and each replaced cuBLAS split-K GEMM drops its splitKreduce node (decode_32 profile: splitKreduce 4.0 + 3.8 us rows, ~2-4 us per nvjet+reduce pair in LESSONS), so ~4-7 us off 252 us (sliding) and 536 us (full): +1.5-2.5% sliding, ~+1% full, ~+1.5% on the decode_32 load
- `*load=decode_1 *` flat: T=1 outside the gate
- `*load=decode_16 *` flat: T=16 outside the gate
- `*load=verify_*` flat: T=8/128/256 outside the gate; control

## Risk

Medium: one load only and an expected effect near the 1.5% drift, so judge it on the decode_32 rows and the replaced kernels own us (nvjet 64x32 / 128x32 splitK rows and their splitKreduce), not the headline. Shape-disjoint from bf16-shape-gated-triton-gemms-small-t (running, T<=16) and touches the same four call sites, so the two combine by 3-way merge whatever each scores. The router is deliberately excluded (swap-AB router read 0.984x in bf16_proj).

## Adversarial review

Reviewed 2026-10-24T165641Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, rules-clean port of a measured m=32 wgmma win into the four dense GEMMs at T=32, but the expected size is overstated (about 3-5 us, not 4-7) and sits inside run drift on the decode_32 rows. Judge it on the replaced kernels' own time, which needs one profile fix: give the swap-AB kernels distinct names per layer. No companion run is needed.

### Justification
The mechanism is plausible and has been measured on the same H200 judge at the same shapes. In bf16_proj at m=32, putting the weight on tl.dot's LHS and the tokens on N made Triton emit wgmma with no spills, and pooler/o/qkv gained 9.3/4.1/3.6% (bf16-m32-swapab-wgmma-triton, 1.198x). The decode_layer GEMMs have exactly those [N,K] shapes. The master decode_32 sliding profile maps cleanly onto the four replaced layers: nvjet 128x32 splitK 20.5 us (1 call) is the pooler; nvjet 64x32 4x1 splitK 12.7 us is qkv_gate; nvjet 64x32 2x1 27.0 us (2 calls) is o plus unpooler. Two of the four splitKreduce launches (4.0 + 3.8 us over 4 calls) belong to the pooler and qkv; the other two belong to the router hi/lo, which is untouched. In-layer that is about 64-66 us of GEMM kernels against 59.9 us for the bf16_proj swap-AB kernels, so a realistic saving is about 3-5 us of kernel time. The hypothesis's 4-7 us double counts: bf16_proj's per-shape times already include the splitKreduce node, so dropping it is not extra on top of the -2.8 us. Against 252 us sliding and 536 us full, that is about +1.2-2% sliding and +0.6-0.9% full, which is under 1% on the decode_32 load and about 0.15% on the six-load geomean. The verify, decode_1 and decode_16 flats are right: T=8/16/128/256 all fall outside 17..32. One upside: sliding decode_32 is master's worst shape (1.003x against SGLang), so even a small real gain there is worth landing. Weakest link: the pooler is the largest share of the saving (about 1.5-2.4 us), and in bf16_proj it was measured alone. Here it runs concurrently with the side-stream router (2 cuBLAS splitK calls, 2 reduces, the routing kernel). At 4 stages x 48 KB = 192 KB of shared memory per CTA, the swap-AB pooler runs 1 CTA per SM on 128 of 132 SMs, and its split-K ticket reduce finishes only when the last CTA does. Any pooler CTA that cannot land because router CTAs hold an SM's shared memory becomes a tail of roughly the router's length. That could erase the pooler's gain in context, an interaction that bf16_proj could not see.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: it owns the side-stream router that overlaps the pooler. At T=32 its router path is SGLang's own hi/lo torch.mm (that patch changes kernels only at T=8), so it is not a tunable confounder. But the overlap it composes is the in-context difference from bf16_proj's standalone pooler: SM and shared-memory contention could make the swap-AB pooler slower here than in bf16_proj. Reverting it is not meaningful (it is the whole pipeline).
- post-expert-norm-triton-small-t: gated to T < 32, so at T=32 the unpooler input still comes from SGLang's fused_post_expert_norm_weighted_sum. The two do not interact numerically or in time. The only implementation point is that the swap-AB route guard must still check stride(1)==1 and the shape of that output (keep _swapab_route_ok's shape and stride checks; they read metadata, not values).
- fa3-hdim128-head-pairing-batched: it changes only attention on full batch>1 layers. It shrank full decode_32 (608 to 536 us), which makes a fixed microsecond saving a slightly larger percentage there, but it does not interact with the GEMMs. Not a confounder.
- marlin-block-m-cap-16: at T=32 the pick is already 16 (32*8/32/16 = 0.5 < 0.9), so the cap is inert at the gated shape. Not a confounder.

### Interference and A/B needs
No companion makes this single result more interpretable than its own profile can. The expected effect (under ~2% on the one affected row) is inside the 1.5-2.5% per-shape drift whichever companion runs, so the headline is not the readout for either. The readout is (a) the replaced kernels' own microseconds against the removed nvjet and splitKreduce rows at decode_32 sliding and full, and (b) the decode_32 rows of the concurrently running bf16-shape-gated-triton-gemms-small-t, whose T=32 path is byte-identical to master and so gives a same-day drift reference. A pooler-excluded control (swap-AB on qkv, o and unpooler only) would isolate the side-stream contention risk. But per-layer kernel rows answer the same question for free, provided the profile can tell the layers apart. The profile aggregates by kernel name, and all four layers would launch one function named _bf16_swapab_kernel, so they would collapse into one 4-call row. Build note instead of a companion: give each layer its own kernel name, for example by defining the same body under four @triton.jit names (_swapab_pooler_kernel, _swapab_qkv_kernel, _swapab_o_kernel, _swapab_unpooler_kernel), or at least split-K versus full-K. The math and launches stay identical, only the names change, and that makes the pooler's in-context time readable against 20.5 + ~2 us of nvjet plus reduce. Run a pooler-excluded control later only if that row shows the swap-AB pooler slower than about 20.5 us in context.

### Open questions
- Does the pooler swap-AB kernel (128 CTAs at ~192 KB shared memory each, so 1 per SM) get delayed by the concurrent side-stream router CTAs? The per-layer kernel names in the profile answer this.
- Merge: this edits the same four kernel_fn call sites as the running bf16-shape-gated-triton-gemms-small-t (T<=16). The gates are disjoint, but the edits touch the same lines (qkv = _gemm(...), etc.), so expect a hand-resolved textual conflict rather than a clean 3-way merge. Settle a single dispatcher helper, for example _dense(name, x, state, tokens), if both land.
- The prepared state stores weights under state['dense'][name], while bf16_proj reads state[layer]. Confirm the adapted _swapab_prepare reads state['dense'], and that its ws and cnt keys do not collide with the shape-gated port's pooler_ws/qkv_ws keys (use distinct swapab_* keys as specified).

