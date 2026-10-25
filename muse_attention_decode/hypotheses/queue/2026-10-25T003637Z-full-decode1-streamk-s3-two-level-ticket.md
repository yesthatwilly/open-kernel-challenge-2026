---
problem: muse_attention_decode
slug: full-decode1-streamk-s3-two-level-ticket
created: 2026-10-25T003637Z
base: master
area: full-layer attention (batch-1 decode, splitkv stream-K kernel)
change: >
  Stage 3 of the full-decode rewrite, built on the stage-2 branch
  hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel:
  stream-K over 64-token tiles with an in-kernel per-request ticket combine + norm-gate), whatever
  its score. Full decode_1 is 27.3 us (FA3 main 14.6 + FA3 combine 7.1 + pre-attn 1.3 + norm-gate)
  against a 4.3 us floor and weight 16 in the decode_1 load (42% of that load); stage-1
  chunk320/chunk640 gated batch 1 onto splitkv and gained on full decode_1 but paid a separate
  combine over many partials. Change: (1) widen the gate to 1 <= batch <= 64 (config only), so
  full decode_1 runs master _pre_attention_kernel (PAIRED=False, as master batch 1 already does) +
  the stage-2 kernel. (2) At batch == 1 only (config-only), grid = SM count (one CTA per SM, ~9.5
  tiles each for 80000 keys) instead of 2 x SM count, passed through the existing grid argument.
  (3) Two-level ticket for a request covered by many CTAs: CTAs whose ranges intersect request b
  are grouped by c // G with G = ceil(sqrt(nseg_b)) (from TP and grid only); each CTA writes its
  fp32 partial (acc[16,64], m[16], l[16]), __threadfence(), atomicAdd on group ticket gticket[b*32
  + g]; the group last arrival LSE-merges its group partials into a group slot, __threadfence(),
  atomicAdd on ticket[b]; the last group LSE-merges the <= G group slots, applies v_scale, the
  stage-2 norm-gate epilogue and stores bf16, and resets both its tickets to 0 for graph replay.
  Batch > 1 keeps the stage-2 single-level ticket (nseg_b <= ~17 there). gticket is a module-level
  int32[64*32] zero buffer allocated at import. BUILD/SAFETY exactly as stage 2
  (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary +
  self-check, each kernel followed by torch.cuda.synchronize() and a print; never a child
  process); add batch-1 self-check cases at 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids
  3/7/SM count, and two consecutive replays to prove both ticket levels reset. Fallback on
  failure: batch 1 returns to master FA3 (PATH CHOSEN: full decode_1 FA3 (fallback: <reason>));
  success prints PATH CHOSEN: full decode_1 streamk two-level. Every other shape keeps the stage-2
  code. First measurement is an okc check, never a scored submit.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: replaced FA3 main 14.6 + combine 7.1 + norm-gate (~23 us of a 27.3 us shape), floor 4.3. 20.5 MB at the stage-1 2.7-2.9 TB/s is ~7.5 us + two merges ~2 us. M1 tests pass and shape < 27.3; M2 shape <= 16 us; M3 <= 12 us
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32*"
    direction: flat
    why: batch > 1 keeps stage-2 grid and single-level ticket
  - shapes: "*layer=sliding* *load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: full-decode-splitkv-s2-streamk-ticket-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1
review: required
status: queued
---

# full-decode1-streamk-s3-two-level-ticket

## Change

Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: stream-K over 64-token tiles with an in-kernel per-request ticket combine + norm-gate), whatever its score. Full decode_1 is 27.3 us (FA3 main 14.6 + FA3 combine 7.1 + pre-attn 1.3 + norm-gate) against a 4.3 us floor and weight 16 in the decode_1 load (42% of that load); stage-1 chunk320/chunk640 gated batch 1 onto splitkv and gained on full decode_1 but paid a separate combine over many partials. Change: (1) widen the gate to 1 <= batch <= 64 (config only), so full decode_1 runs master _pre_attention_kernel (PAIRED=False, as master batch 1 already does) + the stage-2 kernel. (2) At batch == 1 only (config-only), grid = SM count (one CTA per SM, ~9.5 tiles each for 80000 keys) instead of 2 x SM count, passed through the existing grid argument. (3) Two-level ticket for a request covered by many CTAs: CTAs whose ranges intersect request b are grouped by c // G with G = ceil(sqrt(nseg_b)) (from TP and grid only); each CTA writes its fp32 partial (acc[16,64], m[16], l[16]), __threadfence(), atomicAdd on group ticket gticket[b*32 + g]; the group last arrival LSE-merges its group partials into a group slot, __threadfence(), atomicAdd on ticket[b]; the last group LSE-merges the <= G group slots, applies v_scale, the stage-2 norm-gate epilogue and stores bf16, and resets both its tickets to 0 for graph replay. Batch > 1 keeps the stage-2 single-level ticket (nseg_b <= ~17 there). gticket is a module-level int32[64*32] zero buffer allocated at import. BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary + self-check, each kernel followed by torch.cuda.synchronize() and a print; never a child process); add batch-1 self-check cases at 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids 3/7/SM count, and two consecutive replays to prove both ticket levels reset. Fallback on failure: batch 1 returns to master FA3 (PATH CHOSEN: full decode_1 FA3 (fallback: <reason>)); success prints PATH CHOSEN: full decode_1 streamk two-level. Every other shape keeps the stage-2 code. First measurement is an okc check, never a scored submit.

## Expected

- `*layer=full load=decode_1 *` faster: replaced FA3 main 14.6 + combine 7.1 + norm-gate (~23 us of a 27.3 us shape), floor 4.3. 20.5 MB at the stage-1 2.7-2.9 TB/s is ~7.5 us + two merges ~2 us. M1 tests pass and shape < 27.3; M2 shape <= 16 us; M3 <= 12 us
- `*layer=full load=decode_16* *layer=full load=decode_32*` flat: batch > 1 keeps stage-2 grid and single-level ticket
- `*layer=sliding* *load=verify_*` flat: outside the gate

## Risk

Risk: a single CTA LSE-merging 132 partials would be ~1 MB serial read, so the two-level merge is the design point; if M2 is missed, the per-phase eager diagnostic should name whether the gather or the merge dominates.

## Milestones

Rewrite stage 3 of `FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full decode batch 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
