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
status: done
review_at: 2026-10-25T212855Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T212905Z
base_sha: 29d5f864b053aab03e8c54ff87dd01db6442070e
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/sliding-verify-wgmma-window-s1
worktree: .worktrees/muse_attention_decode/sliding-verify-wgmma-window-s1
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T004942Z
impl_commit: ae788b564701c5b041a7cf8d94efa567ae51eb0e
submission_sha256: 88cb856674238b072969154919eba5b619d2416a0ede6ce9cc001d433bed419e
norm_sha256: 2aa40cd291ecdbe482a93231c86c7267c92357d03183e1aa4b667bac32c4ec58
submission_blob: c06e0fbeba63badb368924c2a565fe5ba167d510
lint_warnings: 73
attempt_file: 2026-10-26T012537Z-check-sliding-verify-wgmma-window-s1.md
prepared_at: 2026-10-26T012537Z
job: cd01da3909cc4f4c
dispatched_at: 2026-10-26T012615Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.516417328039991
speedup_ci: [1.4936710681193912, 1.5391635879605907]
harvested_at: 2026-10-26T015634Z
best_at_harvest: 1.5295495462227864
promotion: null
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


## Implementation notes

Edited only `<worktree>/muse_attention_decode/submission.py` (+1215 lines, nothing removed). Base is master 29d5f864 (the claim base). The s5 kernel is NOT on master (branch `hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload`, 69378d7), so its CUDA source was taken from that branch's file and ported by script into a new `_SW_CUDA` string (new module `muse_sliding_verify_wgmma_v1`, kernel `slidewin_verify_wgmma_kernel<TICKET>`, entry points `sw_verify_attn` / `sw_verify_attrs`, so no symbol shares a name with another module).

What changed:
- Kernel: math, 4-stage ring, SW128/SW64 layouts, qa reload, ticket merge + norm-gate, bounded mbarrier waits and error word are verbatim. Dropped: the DBG instantiation and stage 2's B8 combine (the merge tail is timed with the `TICKET=false` instantiation instead). Added one int `window`: `lo_s[b] = window > 0 ? max(0, seq - 8 - (window - 1)) : 0`, `seq_s[b]` holds the window's key COUNT (seq - lo), so `tile_prefix`, the ticket tile count, the tile loop and the `edge` test are unchanged; page-index prefetch reads `page_table[b, lo + pos]` only when `pos < count`; mask adds `elo = window > 0 && lt*64 < count - window` and masks `tile token < lim - window` (equivalent to key t >= seq - 8 + j - (window - 1)). `window = 0` gives lo = 0, no lower mask (same math as s5).
- Route: `_sw_gate(cfg)` (config-only, window == 2048, q_len 8, page 1, hq 16, hkv 2, hd 64, rope on, 1 <= B <= 32, band routed) is tested in kernel_fn BEFORE the PSEUDO predicate and returns `_sw_forward` (master `_ps_prepare_q` = `_pre_attention_kernel` unchanged, then the one kernel). `paired` needs window == 0, so it is disjoint by construction; PSEUDO (B=1) is shadowed only while this route is on for that band.
- New state: `_SW_ROUTE` (frozen MappingProxyType, read-only after prepare), `_sw_buf` int32[129] (self-resetting tickets + error word, zeroed once from the device alone), per-launch workspace by `torch.empty` from shapes. No device pointer into a state's tensors, no TMA, no state-sized buffer. No per-state flag pattern (`_ab_flags` / `NEW_CELL_FLAGS`) exists in this file, so the band dict is the flag; a miss never touches another landed route.
- Prepare (`_sw_setup`, called last at import so master's routes are final): build (deadline), image check (regs, spill, CTAs/SM printed), self-check, three band gates (B=16, 32, then 1), two-state replay of each routed band through `kernel_fn`, ticket/error-word check, recap.
- `_say` now also records `PATH CHOSEN` lines in `_PATH_LOG`; `_sw_recap()` prints them plus this hypothesis' GATE/PATH lines as the last lines of prepare. `_IMPORT_T0` was added at the top for phase deadlines.

Review fixes applied:
1. Partial tail tile: prefetch predicated on `pos < count` (zero-fill, no read), tile starts at lo_b (not 64-aligned), page index also clamped to the cache as in s5.
2. kv_scale 1/64 (0.015625) added to the self-check (B=1 lens 1500 and 80000, B=16, B=32) and to the B=16 two-state replay; kv 0.25 and 1 kept. Route is not restricted by kv_scale.
3. Two interleaved gate rounds (31 reps, cold L2, empty graph subtracted) for the best two grids; a grid routes only if master/cand >= 1.05 in BOTH rounds, candidate IQR <= 1.5x master's in both, and rel L2 vs master's routed output < 3e-2 (plus < 2e-2 vs fp32 reference). Master is `kernel_fn(data)` with this route off, i.e. the ACTUALLY routed path (PSEUDO at B=1 if landed, FA3 + norm-gate otherwise), pre-attention on both sides.
4. Budget: quick ~7-rep interleaved pass over {2B, 4B, 8B, SM, 2xSM} (deduped, capped at 2xSM), finals only for the top 2 and only if the quick best is >= 1.0x. Elapsed per phase printed; deadlines on the import clock: build 360 s, self-check 420, quick/gate start 470, finals 520, replay 560, each prints `skipped: deadline` and keeps master. Constants are `_SW_*_BY`.
5. Per grid, printed separately: pre-attn + kernel, kernel only (no merge, partials only), kernel + ticket merge + norm-gate, and the tail (difference).
6. Route predicate disjoint from `paired` and the B=1 PSEUDO route (checked first, early return).

Answers to open questions: B=2..16 band is gated only at B=16 and the grid spec scales with B (2B/4B/8B) or is SM-based, so B=3..5 ragged get a sensible grid; speed at unscored B is unmeasured but correctness is covered by B=2/5 self-check cases (grid 7 / device). Retiring held queue items s4 / streamk-window is not done (the build is limited to submission.py).

Deviations: the 300 s prepare budget cannot be met if import is already ~222 s (DIAGNOSIS); the deadlines above are on the import clock and are my judgment, please adjust `_SW_*_BY` if the judge's prepare limit is hard. Nothing could be compiled or run on this node.

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only `_sw_gate` (config) and `_sw_forward` (pre-attn + one launch); `.item()`/synchronize only in prepare (lint: warnings outside kernel_fn's call graph); allocations are shape-only.
- [x] Nothing derived from `data` is cached: `_sw_buf` is device-only self-resetting tickets; `_SW_ROUTE` is config/band only.
- [x] Weights: n/a: attention problem, no weights touched.
- [x] KV FP8, q e4m3 (master pre-attention kernel), fp32 softmax/merge, bf16 out, P e4m3 as FA3's FP8 path (s5 verbatim).
- [x] Every key in each window attended: tiles cover [lo_b, seq_b); mask exactly FA3 window_size=(window-1, 0); no skipping.
- [x] No timing patches, harness access, module edits, subprocesses or network; load_inline only (no child process code added).
- [x] All work inside the timed call; no output caching or replay; timing code only in prepare.

okc_lint: `OK with warnings: muse_attention_decode/submission.py` (only host-sync warnings in prepare-only code, outside kernel_fn's call graph). `python3 -I -m py_compile` passes; __pycache__ removed.

## Pre-flight audit

Verdict: dispatch-with-fixes (auditor: opus; read only, nothing could be compiled or run).

Fixes:
- Phase deadlines `_SW_*_BY` tightened from 360/420/470/470/520/560 s to build 330, self-check 380, quick+gate start 430, finals 460, replay 480 s. Reason: the old values let the replay phase start at 560 s and finish near 600 s on a clock that starts after the torch/triton imports, which leaves the 16 tests and timing little room inside the 15 min limit. The new values match the 480 s import start rule used by the merge port. Master's import is about 222 s, so the build (about 34 s) still starts with about 100 s of slack.

Checked and sound:
- Tail-tile reads: `issue_meta` reads `page_table[b, lo + pos]` only when `f < s1` and `pos < count`, and zero-fills otherwise. `issue_tile` zero-fills K/V rows past the count, and the score mask removes them. lo + pos < seq, so it never passes the page-table row.
- Mask: upper edge unchanged. Lower edge `ct < lim - window` is equivalent to FA3 window_size=(window-1, 0), checked at seq = window+7, window+8 and ragged tails. `elo` is a correct tile-level guard.
- Route: `_sw_gate` is config-only and runs before the PSEUDO predicate. Sliding verify has window != 0, so it is never `paired` and never reaches splitkv/fused. The band dict is frozen after prepare, and the replay check asserts the gate.
- kernel_fn branches only on config. Allocations are shape-only, `_sw_buf` is device-only, and there are no syncs or prints.
- The self-check reference matches the kernel's seq convention (cache_seqlens = lens + 8). Shared memory is about 100 KB, so 2 CTAs/SM fits.

Residual risk: the speed is unmeasured at low occupancy (about 2 tiles per CTA), so the 1.05x two-round gate may leave a band on master; the 480 s start rule drops late phases to master with PATH CHOSEN lines rather than failing.

## Result

- job `cd01da3909cc4f4c` (check), status done, passed True, tests 16/16
- speedup 1.5164x, 95% CI [1.4937, 1.5392], score 2483.6830405176397 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-verify-wgmma-window-s1.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-verify-wgmma-window-s1.patch`; branch `hyp/muse_attention_decode/sliding-verify-wgmma-window-s1`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 12.1 | 10.85 | 1.1261 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 17.39 | 22.77 | 0.7571 | lost |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.46 | 15.56 | 1.0013 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 110.77 | 111.46 | 0.9895 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 17.46 | 17.86 | 0.9769 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.89 | 228.29 | 0.994 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.23 | 13.43 | 0.998 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.27 | 29.95 | 1.0016 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.62 | 22.93 | 0.9862 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.46 | 163.34 | 0.9946 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.01 | 28.54 | 1.0085 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 301.99 | 301.12 | 0.9991 | flat |

gained on sliding decode_1; lost on full decode_1. Expected hit: 2/5. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8) at 1<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate (B=16/32), and the PSEUDO sk route (B=1)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on sliding-verify-wgmma-window-s1 --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-verify-wgmma-window-s1`) that adds a further change in the same area is the only way to learn more.

