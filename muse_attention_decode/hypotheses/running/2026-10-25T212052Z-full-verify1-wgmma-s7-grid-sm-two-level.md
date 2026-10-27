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
review: proceed
status: running
review_at: 2026-10-25T212855Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T212857Z
base_sha: d708cbde0daeb3705686792cd6d030c115406b9b
base_blob: 3cd54814f4dcbb30997660d9de6dfacd05908e07
branch: hyp/muse_attention_decode/full-verify1-wgmma-s7-grid-sm-two-level
worktree: .worktrees/muse_attention_decode/full-verify1-wgmma-s7-grid-sm-two-level
mode: null
restack_of: null
stale_base: true
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

## Adversarial review

Reviewed 2026-10-25T212855Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a rewrite stage, but only after s6 is harvested with its self-check passing. The 'whatever its score' clause must not cover a failed self-check. Gate BOTH OVL variants at B=1 rather than taking s6's B=16 winner, and print a merge-only DIAG so that a gate miss can be traced to the tile chain or to the 8x-larger two-level merge.

### Justification
Full verify_1 is 30.3 us (FA3 15.2 + combine 7.2 + prepare 2.8 + norm-gate ~1 + pre-attn 1.5) against a 4.3 us floor, at weight 16 / ~0.24%/us, so about 26 us is replaceable. FINDINGS and the token-group attribution table confirm these numbers. The mechanism is plausible. At B=1, q_len 8 gives 64 query rows per kv head, exactly one m64 wgmma per warpgroup. That removes the mma.sync compute wall that sank full-verify1-sk-two-level-token-group (0.596x gate, row flat 30.0->30.07). For window=0 all 8 tokens share the key range, so the 20 MB are read once. The two-level ticket is proven at decode_1 (27.3->22.6 us; kernel 18.6 us at grid SM). The prior field is honest. s4-batch1 was held on a broken s3 base. The token-group design is mma.sync and is on Do-not-try; the difference here (wgmma, correct s5-derived kernel) is real and is not a retry. DIAGNOSIS action 5 says to do this 'once action 1 routes'. The hypothesis relaxes that to 'whatever its score'. That is acceptable for a rewrite because B=1 is a different regime, but only if s6's kernel passed its self-check. WEAKEST LINK: the latency of each CTA's serial chain, multiplied by too few tiles per CTA. s5 cost about 3.8 us per tile per CTA at B=16 (288.7 us over ~76 tiles/CTA at 2xSM) and about 2.8 us at B=32. At grid SM each CTA owns ~9.5 tiles. Even at a reduced uncontended ~1.5-2 us/tile that is 14-19 us before the merge. And the ring prologue (prefetch distance STAGES_WG-2 = 4, or 10 for OVL=1's 12-stage ring) is about half to all of a CTA's 9.5 tiles, so pipeline fill is never amortised. SECOND: each CTA's partial is 8 tok x 16 heads x 64 fp32 = 32 KB, so ~4.2 MB of partials pass through two merge levels at grid SM (8.4 MB at 2xSM). This is the same 8x-larger merge the token-group reviewer flagged as able to eat the margin, and that design lost. The gate threshold is about 25 us of replaced work at 1.05x. M2 (shape <=22 us) needs a kernel of about 19-20 us, i.e. decode_1's kernel time with 8x the rows and 8x the merge bytes. That is reachable but optimistic. M3 as written (<=15 us) does not match the template milestone (within 1.5x of the 4.3 us floor = ~6.5 us). That floor is unreachable for any design at this tile count; treat M2 as the realistic stage target. The expected flats (verify_16/32 as s6, sliding, decode untouched) match a config-only B==1 route.

### Confounders on the current kernel
- full-verify-wgmma-s6-split-ring-2cta (depends_on, not landed, not yet run): this branch inherits s6's B>1 routing, so verify_16/32 and the headline reflect s6, not s7. Judge s7 only on the full verify_1 row's top_kernels and its PATH CHOSEN line. Judge the B>1 rows against s6's own result, not master or BEST.
- full-decode1-streamk-s3-two-level-ticket (landed): it is the source of the two-level ticket. If s7 reuses master's module-level gticket/ticket int32 arrays rather than its own, decode_1 and verify_1 kernels share self-resetting counters. That is safe only while launches are strictly stream-ordered (no PDL overlap between them). Give s7 its own ticket buffers so a future PDL chain cannot corrupt both.
- paired-heads-launch-cleanup-ab-tune-batch-gate (landed): it defines the B=1 incumbent as UNPAIRED FA3 + unpaired pre-attention. s7's unpaired-query route matches this, and so does s5's kernel, which takes the unpaired FP8 query. The gate must time exactly master's B=1 path (prepare + FA3 + combine + rms_norm_gate), not the paired path s6 gates against. Otherwise the 1.05x decision is mis-anchored.
- fused-preattention-triton (landed): pre-attention stays a separate 1.5 us launch plus a graph-node gap on both sides of the gate, so it cancels. The not-yet-landed V3 chain (fused pre-attn + PDL) showed this gap compounds, so it is headroom left for a later stage, not a confounder.

### Interference and A/B needs
No companions. This is rewrite stage 7. Its readout is the B=1 printed gate (s7 TWO vs master's unpaired B=1 path), the self-check table and the verify_1 top_kernels, all produced inside one run. The B>1 rows belong to s6 and are judged against s6's result. An ab-revert of any landed hypothesis would not change the B=1 route this stage builds. The one ambiguity a single run can leave is which part of a gate miss (tile chain vs merge) cost the time. A companion run cannot fix that cheaply. An in-run DIAG fixes it: print the kernel time with the merge compiled out (partials written, no ticket/merge) next to the full TWO time, and both OVL variants at grid SM and 2xSM. Also do not select the OVL variant from s6's B=16 winner: B=1 at 9.5 tiles/CTA is latency- and prologue-dominated, so the B=16 ranking does not transfer. Gate both at B=1, use two interleaved rounds at >= 1.05x with an IQR bound (the day's lesson on single-round selectors), and print both.

### Open questions
- Has s6 run and passed its self-check? If s6 failed its self-check, s7 must wait for an s6 fix rather than build on it. That is the exact failure that held s4-batch1.
- What is s5/s6's per-tile chain latency on an idle SM (B=1, grid SM)? A one-line DIAG at B=1 with GATHER_ONLY vs full compute would show whether the 9.5-tile chain alone exceeds the ~25 us gate threshold before the merge is added.
- Should the last-group merge spread the 8 tokens x 16 heads over all 256 threads, or over multiple finalizer CTAs (one per token), to keep the 4.2 MB merge off a single CTA's critical path?
- Is grid SM really optimal at B=1 for a 64-row wgmma tile? With prologue depth ~4-10 tiles, a 2xSM or 3xSM grid with a shallower ring may beat SM. Consider adding grid 2xSM for OVL=1 with STAGES_WG reduced as well.

