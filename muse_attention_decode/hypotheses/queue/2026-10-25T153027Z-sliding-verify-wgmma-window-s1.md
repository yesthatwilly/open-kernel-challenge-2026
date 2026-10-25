---
problem: muse_attention_decode
slug: sliding-verify-wgmma-window-s1
created: 2026-10-25T153027Z
base: master
area: sliding-layer attention (verify, wgmma stream-K windowed kernel)
change: >
  Fresh stage 1 on CURRENT master (no depends_on). This is the held
  full-verify-wgmma-s4-sliding-window idea with its hold reason fixed: it was held because the
  stage-3 kernel it builds on failed its self-check; full-verify-wgmma-s5-qfrag-reload (job
  71e40388a1904937, branch hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl
  69378d7) made that kernel correct (all 19 bisect cases, DIAG S correct, two-state replays
  5.3e-3). It is slow on FULL verify (0.57x of FA3 at B=16: ~3.8 us per tile per CTA, serial
  per-tile chain), but sliding verify is a different regime: B=16 has only 16 x 33 = 528 tiles, so
  each CTA owns 1-4 tiles and the shape is decided by launch count and one dependent round trip,
  not by per-tile throughput. Master pays 4 launches after pre-attention there (prepare 1.9 + FA3
  10.7 + combine 4.0 + norm-gate 1.4 us at verify_16; 1.8 + 15.0 + 4.2 + 1.6 at verify_32).
  CHANGE: port the s5 kernel splitkv_verify_wgmma_kernel<ticket> verbatim (math, ring, ticket
  combine + norm-gate, error word) into its own load_inline module muse_sliding_verify_wgmma_v1
  and add ONE int argument window (0 = full behaviour, must compile identically). With window > 0:
  per request seq_b = cache_seqlens[b], key range [lo_b, seq_b) with lo_b = max(0, seq_b - 8 -
  (window - 1)), tiles_b = ceil((seq_b - lo_b) / 64), tile k covers keys lo_b + 64k .. +63 (page
  1, no alignment needed); the tile prefix (tile_prefix) and the ticket's per-request tile count
  use tiles_b; page-index prefetch offsets by lo_b. Mask adds the lower edge: key t valid for
  block token j iff t <= seq_b - 8 + j AND t >= seq_b - 8 + j - (window - 1), applied on any tile
  overlapping either edge (matches FA3 window_size=(window - 1, 0) as master passes it; every
  window key attended). Route: sliding verify (window != 0, q_len 8, page 1, hq 16, hkv 2,
  head_dim 64, rope on, 1 <= B <= 32; config-only) runs master _pre_attention_kernel unchanged
  (ROPE on, PAIRED=False: q FP8 [tokens, 16, 64], new rows stored) then this kernel; nothing else.
  Grid per band chosen in prepare from {2B, 4B, 8B, SM, 2xSM} (capped at 2xSM; config-only), 2
  CTAs/SM as s5. GATE per band (B=1, B=2..16 timed at 16, B=17..32 timed at 32) at batch x 80000
  agentic lengths: interleaved cold-L2 graph replay, empty graph subtracted, median of 31,
  [pre-attn + wgmma window] vs the band's master incumbent (B=1: pre-attn + PSEUDO sk route as
  kernel_fn takes it; B=16/32: pre-attn + FA3 path + rms_norm_gate); route only at >= 1.05x and
  output rel L2 < 3e-2 vs master; print every grid's time. SELF-CHECK vs an fp32 windowed causal
  reference: window 2048, lens < 8, < window, window + 7, window + 8, 80000, B = 1/2/5/16/32
  ragged, kv_scale 1 and 0.25, grids 3/7/device, coherent data, plus the two-state alternating
  graph replay; also run window=0 cases to prove the full instantiation is unchanged. Any miss:
  band stays on master with a PATH CHOSEN line. BUILD as s5 (TORCH_CUDA_ARCH_LIST=9.0a,
  __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo, regs/local printed). Full verify, decode and page-64
  tests untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no
  module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every
  launch, are fine).
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: incumbent shape 22.8 us; replaced kernels prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 = 18.0 us; floor 1.9 us. ~2 tiles per CTA at grid 264. M1 routed; M2 replaced kernels <= 12 us (shape <= 17); M3 <= 8 us
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: incumbent shape 29.2 us; replaced prepare 1.8 + FA3 15.0 + combine 4.2 + norm-gate 1.6 = 22.6 us; floor 3.7 us. M2 replaced <= 15 us
  - shapes: "*layer=sliding load=verify_1 *"
    direction: flat
    why: incumbent PSEUDO sk route 10.0 us + pre-attn 1.7 (13.2 shape); route only if the gate shows >= 1.05x, else flat
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "sliding verify (q_len 8, page 1, tp8) at 1<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate (B=16/32), and the PSEUDO sk route (B=1)"
review: proceed
status: queued
review_at: 2026-10-25T212855Z
review_companions: null
---

# sliding-verify-wgmma-window-s1

## Change

Fresh stage 1 on CURRENT master (no depends_on). This is the held full-verify-wgmma-s4-sliding-window idea with its hold reason fixed: it was held because the stage-3 kernel it builds on failed its self-check; full-verify-wgmma-s5-qfrag-reload (job 71e40388a1904937, branch hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl 69378d7) made that kernel correct (all 19 bisect cases, DIAG S correct, two-state replays 5.3e-3). It is slow on FULL verify (0.57x of FA3 at B=16: ~3.8 us per tile per CTA, serial per-tile chain), but sliding verify is a different regime: B=16 has only 16 x 33 = 528 tiles, so each CTA owns 1-4 tiles and the shape is decided by launch count and one dependent round trip, not by per-tile throughput. Master pays 4 launches after pre-attention there (prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 us at verify_16; 1.8 + 15.0 + 4.2 + 1.6 at verify_32). CHANGE: port the s5 kernel splitkv_verify_wgmma_kernel<ticket> verbatim (math, ring, ticket combine + norm-gate, error word) into its own load_inline module muse_sliding_verify_wgmma_v1 and add ONE int argument window (0 = full behaviour, must compile identically). With window > 0: per request seq_b = cache_seqlens[b], key range [lo_b, seq_b) with lo_b = max(0, seq_b - 8 - (window - 1)), tiles_b = ceil((seq_b - lo_b) / 64), tile k covers keys lo_b + 64k .. +63 (page 1, no alignment needed); the tile prefix (tile_prefix) and the ticket's per-request tile count use tiles_b; page-index prefetch offsets by lo_b. Mask adds the lower edge: key t valid for block token j iff t <= seq_b - 8 + j AND t >= seq_b - 8 + j - (window - 1), applied on any tile overlapping either edge (matches FA3 window_size=(window - 1, 0) as master passes it; every window key attended). Route: sliding verify (window != 0, q_len 8, page 1, hq 16, hkv 2, head_dim 64, rope on, 1 <= B <= 32; config-only) runs master _pre_attention_kernel unchanged (ROPE on, PAIRED=False: q FP8 [tokens, 16, 64], new rows stored) then this kernel; nothing else. Grid per band chosen in prepare from {2B, 4B, 8B, SM, 2xSM} (capped at 2xSM; config-only), 2 CTAs/SM as s5. GATE per band (B=1, B=2..16 timed at 16, B=17..32 timed at 32) at batch x 80000 agentic lengths: interleaved cold-L2 graph replay, empty graph subtracted, median of 31, [pre-attn + wgmma window] vs the band's master incumbent (B=1: pre-attn + PSEUDO sk route as kernel_fn takes it; B=16/32: pre-attn + FA3 path + rms_norm_gate); route only at >= 1.05x and output rel L2 < 3e-2 vs master; print every grid's time. SELF-CHECK vs an fp32 windowed causal reference: window 2048, lens < 8, < window, window + 7, window + 8, 80000, B = 1/2/5/16/32 ragged, kv_scale 1 and 0.25, grids 3/7/device, coherent data, plus the two-state alternating graph replay; also run window=0 cases to prove the full instantiation is unchanged. Any miss: band stays on master with a PATH CHOSEN line. BUILD as s5 (TORCH_CUDA_ARCH_LIST=9.0a, __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo, regs/local printed). Full verify, decode and page-64 tests untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=sliding load=verify_16 *` faster: incumbent shape 22.8 us; replaced kernels prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 = 18.0 us; floor 1.9 us. ~2 tiles per CTA at grid 264. M1 routed; M2 replaced kernels <= 12 us (shape <= 17); M3 <= 8 us
- `*layer=sliding load=verify_32 *` faster: incumbent shape 29.2 us; replaced prepare 1.8 + FA3 15.0 + combine 4.2 + norm-gate 1.6 = 22.6 us; floor 3.7 us. M2 replaced <= 15 us
- `*layer=sliding load=verify_1 *` flat: incumbent PSEUDO sk route 10.0 us + pre-attn 1.7 (13.2 shape); route only if the gate shows >= 1.05x, else flat
- `*layer=full *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Weight 46 per sliding verify shape: verify_16 load 3657 us, sliding share 1049; -6 us there is ~7.5% of the load. Risk: the s5 per-tile chain (~3.8 us/tile at full) may also hurt here when a CTA owns 3-4 tiles; the per-grid timing table shows that and the gate keeps master. Disjoint from full-verify-wgmma-ws-s1-producer-split (different module, different shapes); if that one lands, a stage 2 here adopts its pipeline. Supersedes held full-verify-wgmma-s4-sliding-window (built on the broken s3 branch) and is not the stage-2 split-KV A8 route of held sliding-verify-cuda-streamk-window (mma.sync A8 was 1.8x slower than FA3 on verify, LESSONS).

## Milestones

Rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8) at 1<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate (B=16/32), and the PSEUDO sk route (B=1)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T212855Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage-1 rewrite. It reuses a kernel that is proven correct (s5, 71e40388: 19/19 bisect, coherent data, two-state replays) in a regime where its known weakness, the serial per-tile chain, matters least: about 2 tiles per CTA at sliding verify_16. A hard per-band gate and PATH CHOSEN lines make a single check readable without companions. The real risks are the unmeasured low-occupancy latency plus the ticket-merge tail, and the run's wall clock.

### Justification
Mechanism. In the board profile, sliding verify_16 is 22.6 us: FA3 main 10.7, combine 3.8, pre-attn 2.7, plus prepare and rms_norm_gate outside the top 3. Sliding verify_32 is 29.0 us: FA3 14.8, combine 4.3, pre-attn 4.1. The floors are 1.9 and 3.7 us, so roughly 18 and 22.6 us of replaced launches sit 5-10x above the floor. That makes the target launch count and dependent round trips, not bandwidth. A full window is 2055 keys, which is 33 tiles of 64 per request. 16 x 33 = 528 tiles at grid 264 is about 2 tiles per CTA, so most of s5's measured 3.8 us/tile is chain latency, not repeated serially. The expected directions match TASK.md: weight 46 per sliding verify row, verify_16/32 faster, verify_1 flat unless gated in, full/decode untouched. The milestones are readable from top_kernels, since the new kernel replaces four named kernels with one. Prior. The frontmatter has no `prior:` field, but the body names every relevant ancestor: held full-verify-wgmma-s4-sliding-window (same idea, built on the broken s3 branch), sliding-verify-cuda-streamk-window and sliding-verify-sk-pseudo-decode-s1 (mma.sync, 0.653x/0.482x at B=16/32). The FINDINGS/DIAGNOSIS do-not-try entry is 'mma.sync for q_len-8 verify at B>=16', and DIAGNOSIS says 'the wgmma line replaces this approach'. The difference (FP8 wgmma tensor-core path vs mma.sync) is real, and this is DIAGNOSIS action 2. So it is not a duplicate. WEAKEST LINK: s5 has never been timed at low occupancy, and the ticket merge is a serial tail. The only s5 timing is full verify at B=16/32 (288.7/429.5 us, ~76 tiles per CTA), where latency overlapped across many tiles. Here every CTA's first tile has a fully exposed cold-L2 gather plus S, transpose, softmax and PV, then a partial write and an atomic. The last-arriving CTA per request then merges about 16-17 fp32 partials (8 tokens x 16 heads x 64 plus m/l, about 33 KB each, about 0.55 MB per request) and runs norm-gate from one SM. Merge plus norm-gate alone could cost several us, so the M2 target (<= 12 us replaced) is plausible but not assured. Fan-in vs per-CTA tile count is exactly what the {2B, 4B, 8B, SM, 2xSM} per-grid table resolves, so the design can find the right point. At verify_1 (33 tiles), any grid either fans in 33 partials to one CTA or serialises 4-8 tiles per CTA, against a 9.9 us PSEUDO incumbent. Flat via the gate is the realistic outcome there.

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: owns the sliding verify_1 incumbent (PSEUDO sk route). The B=1 gate compares against it, not against FA3. A verify_1 move is attributable only via PATH CHOSEN; if the band stays on master, verify_1 is a re-measure.
- fused-preattention-triton: the new path keeps _pre_attention_kernel unchanged (ROPE on, PAIRED=False), the same node master runs before FA3. Its 2.7/4.1 us is common to both sides and is not a confounder, as long as the builder does not touch it.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup: their predicate is window == 0, so sliding verify is unpaired on master and on this branch. No interaction, but the route predicate must stay disjoint from `paired` and `pseudo` (checked before or instead of _ps_gate) so B=1 does not take two routes.
- All four existing load_inline modules (splitkv, sk/PSEUDO, skF, slide): no numerical interaction. They share the run's wall clock, though. DIAGNOSIS puts import at 222 s across 4 modules, s5's module compiled in 33.9 s, and s5's whole check took ~430 s. A fifth module plus a 3-band x 5-grid x 31-median gate and the full self-check suite can approach the 15-minute run limit, and a timeout is charged.

### Interference and A/B needs
No companions. This is kind: rewrite, judged on the replaced kernels' own time per shape. The per-band hard gate prints every grid's time against the band's master incumbent, measured in the same process with cold-L2 graph replay. PATH CHOSEN lines say which path ran on each scored row. Together these make a gain, a loss and 'gated off' all unambiguous from the one log plus top_kernels. An ab-revert of any landed hypothesis would not change what this run measures, because the replaced FA3/combine/norm-gate chain has had no landed tuning on sliding verify. A control isolating a sub-change (window mask vs the s5 kernel) is meaningless: s5 at window=0 is already measured (0.570x/0.704x on full). One forward-looking caveat: queued sliding-verify-fa3-num-splits-band-gate changes this hypothesis's incumbent on the same rows. If both run on the same day, do not stack them blind. Whichever lands second must re-gate against the new master, and this run's gate numbers are valid only against the master it was built on.

### Open questions
- The self-check uses kv_scale 1 and 0.25, but the judge's own sliding page-1 verify test uses kv_scale 0.015625 (batch 3, ctx 4000). Add 1/64 to the self-check, or exclude kv_scale != 1 from the route config-only, so the first failing case is not a judge test.
- Per the new single-round selector lesson (1.02x selectors mis-routed bf16 today): the 1.05x threshold has margin, but a second interleaved gate round with an IQR bound for each band that would route is cheap insurance against a noisy route at verify_16/32.
- Print, per grid, kernel-only time and the ticket-merge tail separately (e.g. a no-merge DBG variant, or a clock64 stamp on the last CTA). If M2 is missed, stage 2 then knows whether to attack first-tile latency (PDL from pre-attention, prefetch) or merge fan-in (two-level or cluster DSMEM merge).
- Retire or mark superseded the held full-verify-wgmma-s4-sliding-window and sliding-verify-cuda-streamk-window queue items, so a later tick does not run the same idea again on the broken base.
- Does the gate's B=2..16 band (timed only at B=16) risk routing small-B tests (B=3/4/5 ragged, ctx 4000-6000) through a grid tuned for 528 tiles? Correctness is covered by the self-check; only speed at unscored B is affected, which is acceptable but worth noting.

