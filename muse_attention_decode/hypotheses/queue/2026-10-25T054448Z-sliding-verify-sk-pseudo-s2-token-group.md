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
review: required
status: queued
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
