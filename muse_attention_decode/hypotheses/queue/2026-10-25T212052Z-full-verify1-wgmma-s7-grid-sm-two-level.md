---
problem: muse_attention_decode
slug: full-verify1-wgmma-s7-grid-sm-two-level
created: 2026-10-25T212052Z
base: master
area: full-layer attention (batch-1 verify, wgmma stream-K kernel, two-level ticket)
change: >
  DIAGNOSIS action 5. Stage 7 on the full-verify-wgmma-s6-split-ring-2cta branch, whatever its
  score (its kernel is correct by its self-check; the B=1 regime needs per-request parallelism,
  not per-tile throughput). Full verify_1 (B=1, q_len 8, 80K keys, weight 16 at ~0.24%/us) is 30.3
  us of shape: FA3 main 15.2 + FlashAttnFwdCombine 7.2 + prepare_varlen_num_blocks 2.8 + pre-attn
  + rms_norm_gate; floor 4.3 us. At B=1 one request has 1250 64-token tiles and 64 query rows per
  kv head (8 tokens x 8 q heads), exactly one m64 wgmma tile per warpgroup. Add the B=1 band: the
  s6 kernel variant that measured fastest at B=16 in s6's printed gate (OVL=0 or OVL=1) gains a
  TWO template flag with the two-level ticket of full-decode1-streamk-s3-two-level-ticket (group
  size Gs = ceil(sqrt(nseg)), group slots in gticket[c_b*32 + gi], last group merges + norm-gate;
  that change took full decode_1 27.3 -> 22.6 us) at grid = SM count (and 2xSM for OVL=0), so the
  CTAs of the one request merge in two levels instead of one serial merge of ~132-264 slots.
  Pre-attention unchanged (unpaired, as master's batch-1 branch). Route: full layer, q_len 8, B ==
  1, config-only. GATE at B=1 x 80000: interleaved [pre-attn + s7 TWO, each grid] vs master's B=1
  path [pre-attn + prepare + FA3 + combine + rms_norm_gate]; route only at >= 1.05x and rel L2 <
  3e-2 vs master. SELF-CHECK: s5's bisect B=1 cases (B1-2t..B1-34t, mid-request 1-tile segments),
  lens 1, 8, 63, 4097, 80000, kv 0.25, grids 3/7/sm/2sm, coherent data, then the two-state
  alternating replay. Supersedes the held full-verify1-wgmma-s4-batch1-grid-sm (built on the s3
  kernel that failed its self-check). If s6 never built, this stage builds s6's OVL=0 variant
  itself first. New CUDA: first measurement is an okc check. RULES (every path): CUDA-graph
  capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config
  only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable
  Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16
  out; every key in each window attended; no child processes, no timing in kernel_fn. The judge
  prepares TWO states and rotates them: no module/static global may hold a device pointer into a
  state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed
  once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only,
  -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print
  after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local
  bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty
  graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints
  both numbers.
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: replaced FA3 15.2 + combine 7.2 + prepare 2.8 + norm-gate ~1 us of a 30.3 us shape (floor 4.3). M1 routed (>=1.05x); M2 shape <=22 us (about -8 us at 0.24%/us = +2%); M3 <=15 us
  - shapes: "*layer=full load=verify_16 *"
    direction: flat
    why: "as s6 (the B>1 bands are s6's routing)"
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: high
depends_on: full-verify-wgmma-s6-split-ring-2cta
priority: 2
kind: rewrite
controls_for: null
prior: "full-verify1-wgmma-s4-batch1-grid-sm: same idea, held because its s3 base kernel failed its self-check; this builds on the correct s5-derived s6 kernel. Differs from full-verify1-sk-two-level-token-group (mma.sync, 0.596x, Do-not-try) by using wgmma for the 64 query rows per kv head"
rewrite_stage: 7
replaces: "full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + rms_norm_gate"
review: required
status: queued
---

# full-verify1-wgmma-s7-grid-sm-two-level

## Change

DIAGNOSIS action 5. Stage 7 on the full-verify-wgmma-s6-split-ring-2cta branch, whatever its score (its kernel is correct by its self-check; the B=1 regime needs per-request parallelism, not per-tile throughput). Full verify_1 (B=1, q_len 8, 80K keys, weight 16 at ~0.24%/us) is 30.3 us of shape: FA3 main 15.2 + FlashAttnFwdCombine 7.2 + prepare_varlen_num_blocks 2.8 + pre-attn + rms_norm_gate; floor 4.3 us. At B=1 one request has 1250 64-token tiles and 64 query rows per kv head (8 tokens x 8 q heads), exactly one m64 wgmma tile per warpgroup. Add the B=1 band: the s6 kernel variant that measured fastest at B=16 in s6's printed gate (OVL=0 or OVL=1) gains a TWO template flag with the two-level ticket of full-decode1-streamk-s3-two-level-ticket (group size Gs = ceil(sqrt(nseg)), group slots in gticket[c_b*32 + gi], last group merges + norm-gate; that change took full decode_1 27.3 -> 22.6 us) at grid = SM count (and 2xSM for OVL=0), so the CTAs of the one request merge in two levels instead of one serial merge of ~132-264 slots. Pre-attention unchanged (unpaired, as master's batch-1 branch). Route: full layer, q_len 8, B == 1, config-only. GATE at B=1 x 80000: interleaved [pre-attn + s7 TWO, each grid] vs master's B=1 path [pre-attn + prepare + FA3 + combine + rms_norm_gate]; route only at >= 1.05x and rel L2 < 3e-2 vs master. SELF-CHECK: s5's bisect B=1 cases (B1-2t..B1-34t, mid-request 1-tile segments), lens 1, 8, 63, 4097, 80000, kv 0.25, grids 3/7/sm/2sm, coherent data, then the two-state alternating replay. Supersedes the held full-verify1-wgmma-s4-batch1-grid-sm (built on the s3 kernel that failed its self-check). If s6 never built, this stage builds s6's OVL=0 variant itself first. New CUDA: first measurement is an okc check. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=verify_1 *` faster: replaced FA3 15.2 + combine 7.2 + prepare 2.8 + norm-gate ~1 us of a 30.3 us shape (floor 4.3). M1 routed (>=1.05x); M2 shape <=22 us (about -8 us at 0.24%/us = +2%); M3 <=15 us
- `*layer=full load=verify_16 *` flat: as s6 (the B>1 bands are s6's routing)
- `*layer=sliding *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Runs after s6's check (depends_on). Weighted ~+2% if M2 is reached, above the margin. If s6's gate shows the wgmma kernel still >1.5x slower per tile than FA3, read s6's gather-only DIAG before spending this check.

## Milestones

Rewrite stage 7 of `full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
