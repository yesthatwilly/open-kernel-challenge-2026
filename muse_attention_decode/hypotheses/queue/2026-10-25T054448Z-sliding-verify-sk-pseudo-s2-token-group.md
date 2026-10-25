---
problem: muse_attention_decode
slug: sliding-verify-sk-pseudo-s2-token-group
created: 2026-10-25T054448Z
base: master
area: sliding-layer attention (verify, stream-K pseudo-decode kernel, token grouping)
change: >
  Stage 2 of the pseudo-decode line, on branch
  hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. Measured: the PSEUDO kernel won
  B=1 (15.4 vs FA3 19.6 us in prepare, 9.8 us kernel in the profile) but lost B=16 (37.2 vs 24.3)
  and B=32 (62.6 vs 30.2) because each of the 8 tokens of a request re-reads the same ~2048-key
  window: 8x L2 read amplification (67/134 MB of L2 reads). Lever: token grouping. Add template
  int TG in {1,2,4} (TG=1 is s1's PSEUDO code, byte-identical): a pseudo-row block is now TG
  consecutive tokens j0..j0+TG-1 of one request (R = 8B/TG blocks). Its key range is the union
  [lo_{j0}, sp_{j0+TG-1}) (TG-1 keys longer than one window); each K/V tile is cp.async-loaded
  ONCE into the smem ring and consumed by TG row groups. Row layout keeps the gather-bound
  mma.sync shape per row group: row group g = token j0+g x (2 kv heads x 8 q heads) = 16 rows,
  exactly the decode tile, so MMA work per key per row is unchanged (LESSONS: 64-row verify tiles
  are MMA-bound; this stays at 16-row tiles, TG of them per K/V tile). Assign row groups to warps
  (warp w owns group w % TG and its q fragment, O accumulator and m/l), so per-thread registers
  stay at the s1 level; print regs/local/smem per TG and refuse a TG that spills. Masking is per
  row group: key t valid for token j iff lo_j <= t < sp_j (sp_j = cache_seqlens[b]-7+j, lo_j =
  max(0, sp_j-2048)), applied in the score mask; a fully masked tile for one group contributes
  nothing (guard m=-inf exactly as decode's empty tiles). Ticket/workspace: one slot per (CTA,
  block) holds TG row groups; the finalize writes out rows b*8+j for each group. Gate as s1 but
  per (band, TG, ticket variant, grid in {4R, 8R, SM, 2xSM}): print every time, route a band only
  if its best variant beats FA3 by >= 1.05x; B=1 also compares TG=2/4 against s1's TG=1 and keeps
  the faster. Self-check additionally covers tokens whose windows straddle a tile edge (lens
  2041..2055) and kv_scale 0.25 for each TG. RULES (every path): CUDA-graph capturable; kernel_fn
  does no host sync, no allocation sized by input values (shape/config only), no data-dependent
  Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached
  from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window
  attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them:
  no module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0
  by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare
  against an fp32 reference with two states and alternating graph replays with refilled inputs
  (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to
  master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed
  TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: incumbent FA3 path 22.68 us shape (main 10.7 + combine 4.0 + prepare 1.8 + norm-gate); TG=4 cuts L2 reads 67 -> 17 MB over 8.4 MB DRAM (floor 1.9 us); M2 shape <= 20 us
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: incumbent 28.34 us (main 14.8 + combine 4.2 + prepare 1.8); TG=4 at 34 MB L2 reads; M2 shape <= 26 us, else the gate keeps FA3 (flat)
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: s1 13.44 us; TG=2/4 halves/quarters the 4 MB of L2 reads and the per-pseudo-row merges; flat if TG=1 stays best
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: high
depends_on: sliding-verify-sk-pseudo-decode-s1
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "sliding verify (q_len 8, page 1, tp8, 2<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate"
review: proceed
status: queued
review_at: 2026-10-25T054704Z
review_companions: null
---

# sliding-verify-sk-pseudo-s2-token-group

## Change

Stage 2 of the pseudo-decode line, on branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. Measured: the PSEUDO kernel won B=1 (15.4 vs FA3 19.6 us in prepare, 9.8 us kernel in the profile) but lost B=16 (37.2 vs 24.3) and B=32 (62.6 vs 30.2) because each of the 8 tokens of a request re-reads the same ~2048-key window: 8x L2 read amplification (67/134 MB of L2 reads). Lever: token grouping. Add template int TG in {1,2,4} (TG=1 is s1's PSEUDO code, byte-identical): a pseudo-row block is now TG consecutive tokens j0..j0+TG-1 of one request (R = 8B/TG blocks). Its key range is the union [lo_{j0}, sp_{j0+TG-1}) (TG-1 keys longer than one window); each K/V tile is cp.async-loaded ONCE into the smem ring and consumed by TG row groups. Row layout keeps the gather-bound mma.sync shape per row group: row group g = token j0+g x (2 kv heads x 8 q heads) = 16 rows, exactly the decode tile, so MMA work per key per row is unchanged (LESSONS: 64-row verify tiles are MMA-bound; this stays at 16-row tiles, TG of them per K/V tile). Assign row groups to warps (warp w owns group w % TG and its q fragment, O accumulator and m/l), so per-thread registers stay at the s1 level; print regs/local/smem per TG and refuse a TG that spills. Masking is per row group: key t valid for token j iff lo_j <= t < sp_j (sp_j = cache_seqlens[b]-7+j, lo_j = max(0, sp_j-2048)), applied in the score mask; a fully masked tile for one group contributes nothing (guard m=-inf exactly as decode's empty tiles). Ticket/workspace: one slot per (CTA, block) holds TG row groups; the finalize writes out rows b*8+j for each group. Gate as s1 but per (band, TG, ticket variant, grid in {4R, 8R, SM, 2xSM}): print every time, route a band only if its best variant beats FA3 by >= 1.05x; B=1 also compares TG=2/4 against s1's TG=1 and keeps the faster. Self-check additionally covers tokens whose windows straddle a tile edge (lens 2041..2055) and kv_scale 0.25 for each TG. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=verify_16 *` faster: incumbent FA3 path 22.68 us shape (main 10.7 + combine 4.0 + prepare 1.8 + norm-gate); TG=4 cuts L2 reads 67 -> 17 MB over 8.4 MB DRAM (floor 1.9 us); M2 shape <= 20 us
- `*layer=sliding load=verify_32 *` faster: incumbent 28.34 us (main 14.8 + combine 4.2 + prepare 1.8); TG=4 at 34 MB L2 reads; M2 shape <= 26 us, else the gate keeps FA3 (flat)
- `*layer=sliding load=verify_1 *` faster: s1 13.44 us; TG=2/4 halves/quarters the 4 MB of L2 reads and the per-pseudo-row merges; flat if TG=1 stays best
- `*layer=full *` flat: untouched

## Risk

Kill: if every TG at B=16 is still slower than FA3 in the printed gate, the pseudo line is B=1-only and this row belongs to the wgmma verify line. Risk: union key range and per-group masks at window edges; warp-per-group mapping must not double the ring depth.

## Milestones

Rewrite stage 2 of `sliding verify (q_len 8, page 1, tp8, 2<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054704Z by an adversarial subagent (model opus).

**Verdict: proceed.** Token grouping aims at the cause s1 measured for its B=16/32 losses: 8x L2 re-reads of the same window. It keeps s1's 16-row mma.sync tile per token, and the per-band >=5% printed gate keeps every sliding verify row attributable. Proceed with no companion. Judge the run against s1's result (job 94523c3313444fd8), not master, and treat B=16 as a coin flip.

### Justification
The mechanism is plausible. s1 lost verify_16 (37.2 vs FA3 24.3 us in prepare) and verify_32 (62.6 vs 30.2) while DRAM bytes stayed about 1x (8.4 MB at B=16). That loss is consistent with an L2-bandwidth and per-key-gather bound: 67 MB of cp.async plus 8x page_table lookups at roughly 5-7 TB/s of L2 is 10-13 us before latency. TG=4 cuts both the K/V bytes and the page_table gathers 4x (to about 17 MB) and leaves total MMA work unchanged, since the union range adds only TG-1 keys. Each warp owns one 16-row group with its own O and m/l, so registers stay at the s1 level and no intra-CTA merge is added. The masking algebra extends s1's audited sp/lo correctly: groups g>0 mask only their first g keys of the union, so a fully masked tile occurs only at tiny lengths, where the m=-inf guard applies. The targets match the FINDINGS profile: sliding verify_16 is 22.7 us (FA3 main 10.7 + combine 4.0 + prepare 1.8 + norm-gate) against a 1.9 us floor, and verify_32 is 28.3 us against 3.7 us. These rows carry weight 46 each, so a real win would show in the headline. WEAKEST LINK: per-tile latency. With TG=4 each warp now runs all 64 keys of a tile against its group, about 4x the per-tile MMA, softmax and mask work of s1's decode split. The 4 groups also share one ring slot, so the slot is released only when the slowest group finishes. The kernel can therefore become issue/latency-bound before the L2 savings pay off. Going from 37 us to under 22.7 us needs a 1.65x drop, which is not guaranteed even at 4x fewer bytes. A second weak point: the B=2..16 and 17..32 bands are gated on uniform-80000 synthetic timing, but the scored shapes use spread=agentic, so the gate decision may not transfer exactly. The 5% margin covers most of that gap.

### Confounders on the current kernel
- sliding-verify-sk-pseudo-decode-s1 (base branch, NOT on master; 1.4705x check, verify_1 17.78 -> 13.44 us): a master-relative headline will include s1's whole verify_1 gain. Attribute TG only against s1's result json (verify_1 13.44, verify_16 22.39, verify_32 28.49), and confirm that PATH CHOSEN for B=1 names which TG won.
- full-decode1-streamk-s3-two-level-ticket (c3c1b01, the templated splitkv_attn_sk_kernel): the TG template adds instantiations to the same TU. The PSEUDO=false variants must stay byte-identical (121 regs / 0 local / 95264 B / 2 CTAs per SM printed IDENTICAL). Otherwise a full decode_1/16/32 move would be a confound, not noise.
- fused-preattention-triton (landed): provides the unpaired FP8 q rows b*8+j and the new K/V rows. It is unchanged and excluded from both sides of the gate timing, so it is not a confound unless TG reorders q reads. It does not.
- sliding-deep-splits-on-cuda-splitkv / sliding-fused-triton-decode1-gate (landed): sliding decode only, a disjoint Triton path. A sliding decode_1 move (s1 showed -3.9%, which is noise-level) is run-to-run variance and not caused by TG.

### Interference and A/B needs
No companion. Every sliding verify row is either the TG kernel, routed only when its printed best variant beats the printed FA3 time by >=1.05x and named with TG/ticket/grid in PATH CHOSEN, or byte-identical FA3. The TG=1 point is s1's code and is timed in the same gate at B=1. s1's own check result is the stage control, so the in-run variant matrix already does what a TG=1 control would. Full layers and sliding decode run untouched code. A gain or a loss is therefore readable from the PATH CHOSEN lines plus the profile top_kernels without an extra check, which matters with 8 checks for 8 queued hypotheses.

### Open questions
- BUILDER: print the per-TG regs/local/smem/CTAs-per-SM and the per-tile issue layout (which warp runs which group over which keys). If 4 warps x TG=4 means one warp per group over all 64 keys, also time TG=4 with 8 warps (2 warps per group, keys split as in decode) when the template allows it cheaply. That targets the per-tile latency weak link.
- BUILDER: also print the s1 TG=1 time in the B=16 and B=32 bands next to TG=2/4 and FA3. The ratio TG=1/TG=4 tests the L2-amplification diagnosis directly, even if the band falls back to FA3.
- Prepare budget: 3 TG x 2 tickets x 4 grids x 3 bands of cold-L2 median-of-10 replays, plus per-TG self-checks and the extra instantiations' compile time. Confirm the total stays well inside the import budget. Prune grids per TG (for example drop 2xSM for TG=4 at B=1, where R=2) if it gets close.
- The gate uses uniform 80000-length synthetic batches, but the scored B=16/32 shapes are spread=agentic. Consider timing the band candidates on a spread-like synthetic too, so a marginal (5-10%) route does not invert on the real shape.
- Kill criterion as written: if every TG at B=16 is still slower than FA3, retire the pseudo line to B=1 only and hand verify_16/32 to the wgmma or cluster verify lines. Never stack those lines on this one.

