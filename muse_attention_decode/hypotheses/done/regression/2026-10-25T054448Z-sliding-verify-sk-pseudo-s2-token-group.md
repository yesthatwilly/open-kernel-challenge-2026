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
status: done
review_at: 2026-10-25T054704Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T054704Z
base_sha: 6f95ce71ba21682ba9cbb6d346abb2a93a862b2f
base_blob: 76492911dfd6685139a2ffac06c1d27083ecdb0a
branch: hyp/muse_attention_decode/sliding-verify-sk-pseudo-s2-token-group
worktree: .worktrees/muse_attention_decode/sliding-verify-sk-pseudo-s2-token-group
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T084910Z
impl_commit: 39fb54fe3f47f42f763160b2e09426e294bdbe7a
submission_sha256: 1926573a2bbdde9bebf692b1770bff845fac7977b2ec293f9d03ef6e03b6dd2c
norm_sha256: bf3c789fd6f7bc3061490ec1cb37abfdd4992d0e3b01d201e3fecd13d719a494
submission_blob: 7fd30a78be18303b7c935dc944c31ff33649bed0
lint_warnings: 27
attempt_file: 2026-10-25T084910Z-check-sliding-verify-sk-pseudo-s2-token-group.md
prepared_at: 2026-10-25T084910Z
job: bd0ddd413f7f4a94
dispatched_at: 2026-10-25T084919Z
outcome: regression
verdict_detail: regression
speedup: 1.4250766878672403
speedup_ci: [1.4037005375492317, 1.451856608658083]
harvested_at: 2026-10-25T091106Z
best_at_harvest: 1.444132502888963
promotion: null
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


## Implementation notes

Edited only `muse_attention_decode/submission.py` in the worktree (nothing could be compiled or run on this node: no GPU, no nvcc).

Kernel (`splitkv_attn_sk_kernel`): new template `int TG = 1`; the TG=1 body is the old code verbatim inside `if constexpr (TG == 1)`; `Lay` gained `TGM` (q slots x TG, SP0 table), `merge_range` gained default `sstride/soff`, `load_prefix_pseudo<TG>` builds union count / lo / raw sp0 per token block. Variants 6/7 = TG 2 (single/two-level), 8/9 = TG 4. A block = TG consecutive tokens, R = 8B/TG, union key range, K/V tile loaded once. Workspace/group slots index `(cta + block) * TG + token`; finalize writes rows `block*TG + token`. Per-token masks `lo_j <= t < sp_j` rebuilt per tile from lo_s/sp0_s; `m=-inf` guard kept per token row.

Deviation (review/spec said warp w owns group w%TG, one 16-row group per token): s1's mma.sync wastes mma rows 8-15 (a1/a3 = 0, acc[2..3] dead). TG>1 puts token a in rows 0-7 and token b in rows 8-15 of the SAME m16n8k16, so TG=2 costs s1's MMA count per warp and TG=4 is 2 token pairs x 2 warps per pair (32-key slices) - this is also the review's "TG=4 with 8 warps" ask, no extra template needed. q is kept as raw e4m3 words (8 regs) and converted per tile to stay near the 128-reg cap; the segment-end merge runs in two passes reusing acc_s/ml_s (smem stays inside 2 CTAs/SM by arithmetic: TG2 ~100.9 KB, TG4 ~108.1 KB).

Review fixes applied: per-TG regs/local/smem/CTAs-per-SM printed and the issue layout printed (`_ps_layout_lines`); a variant that spills (local>0) or cannot launch is refused, a variant that misses a self-check case is excluded alone with a PATH CHOSEN line naming the case; s1 TG=1 best, TG=2/4 best and FA3 printed together per band plus the TG1/TG4 ratio; gate grid pruned (`_ps_candidates`: effective grid, dup/<2-tiles-per-CTA drops, two-level only at >=4 CTAs/block); B=16/32 top-per-TG candidates re-timed with FA3 on an agentic synthetic (80000 +-7.5%) and a route needs >=1.05x on BOTH uniform and agentic; PSEUDO=false variants untouched (variant 0 identical check still printed in `_sk_validate`); workspace sized from batch/grid/TG only. New self-check cases per variant: tile-edge straddle lens 2041..2055 at kv 0.25, B=1 lens 0, B=3, B=16/32 ragged. Replay check replaced by `_ps_two_state_check`: two prepared states, a graph captured on each, alternating A,B,A,B replays each preceded by an in-place refill with new lens.

## Rules check

- [x] CUDA-graph capturable: kernel_fn path unchanged in form (pre-attention + one launch); workspace is torch.empty sized from batch/grid/TG only; no sync, no input-value control flow (diff checked).
- [x] Nothing derived from `data` is cached: only self-resetting int32 ticket arrays are module-level; no pointer/tensor map/workspace in a global.
- [x] n/a: no weights in this problem.
- [x] KV FP8, q e4m3 from the pre-attention kernel, fp32 softmax/merges, bf16 out.
- [x] Every key of each token's window attended (union range + per-token masks); no skipping.
- [x] No timing patches, harness access, subprocess or network (lint OK).
- [x] All work inside the timed call; sm_90a build flags unchanged (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, no setmaxnreg/wgmma used).

okc_lint: `OK with warnings` (only host-sync warnings in prepare-time code, outside kernel_fn). `py_compile` OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent, by reading the code and emulating the masks in plain Python. Nothing was compiled or run.

**Verdict: dispatch-with-fixes.**

Checked and found sound:
- PSEUDO=false and TG=1 are unchanged in source. The TG=1 body is verbatim inside `if constexpr (TG == 1)`. `Lay<..., TGM=1>` gives the earlier offsets. `merge_range` defaults to stride 1 and offset 0. The landed no-SK_PSEUDO fallback build still compiles the new helpers.
- Masks: a brute-force emulation covered lens 0..2099, 5000, 79999 and 80000 at TG 2 and 4. For every token, union range [lo_j0, sp_j0+TG-1), tile count and per-token mask `lo_j <= t < sp_j` select exactly the reference window max(0,p-2047)..p, with 0 mismatches.
- MMA fragments: token a sits in rows 0-7 (a0/a2, c0/c1) and token b in rows 8-15 (a1/a3, c2/c3). The QK and PV k-permutations match TG=1. The smem merge picks the warps hg + 2(p2 + NP j). Finalize rows are c_b*TG + k = b*8 + j. Workspace slots are (cta + block)*TG + k, and group slots are (c_b*32 + gi)*TG + k. Every host TORCH_CHECK holds for B <= 32.
- Smem: TG2 is 100896 B and TG4 is 108064 B, so both fit 2 CTAs per SM. The m=-inf guards match TG=1. Tickets reset at both levels.
- Rules: kernel_fn is unchanged in form, with a config-only gate, workspace sized from batch, grid and TG, and no sync, print or try/except. The only module state is self-resetting int32 tickets plus floats in _PS_INFO. The two-state check captures one graph per state, then refills and replays them in the order A, B, A, B.
- Fallbacks: a spill or a self-check miss in one variant excludes that variant alone, with a PATH CHOSEN line. Validate, gate and two-state exceptions are caught and the band goes to FA3.

Fix:
- The FA3-fallback PATH CHOSEN lines carried no numbers, and the per-band gate SUMMARY lines may fall out of the 16 KB log tail because the s2 gate prints about 17 candidates x 3 lines per band. The kill criterion for this run is "every TG at B=16 slower than FA3", so the fallback lines need the numbers. _PS_INFO is now recorded for every gated band, not only routed ones, and includes the best uniform time per TG. The fallback PATH CHOSEN now prints the best time per TG, FA3, and the agentic pair when present. Logging only; routing is unchanged.

okc_lint: OK with warnings (prepare-time host syncs only). py_compile: OK. __pycache__ removed.

Residual risk: the CUDA has never been compiled. A compile error in the TG>1 branch would drop full decode to a landed rebuild, which doubles compile time. A spill at TG=4 under the 128-reg cap refuses that variant loudly. About 59 self-check cases plus about 4 new instantiations push prepare above the 380 s that s1 took, though it should stay under 900 s. A device fault in a new variant would poison the context, which is the same exposure as s1.

## Result

- job `bd0ddd413f7f4a94` (check), status done, passed True, tests 16/16
- speedup 1.4251x, 95% CI [1.4037, 1.4519], score 2642.8753147570033 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/sliding-verify-sk-pseudo-decode-s1.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/sliding-verify-sk-pseudo-s2-token-group.result.json`, `muse_attention_decode/hypotheses/done/regression/sliding-verify-sk-pseudo-s2-token-group.patch`; branch `hyp/muse_attention_decode/sliding-verify-sk-pseudo-s2-token-group`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.33 | 13.5 | 0.9808 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 22.73 | 23.13 | 1.0004 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.72 | 15.16 | 1.0091 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 110.4 | 111.73 | 0.9852 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 23.85 | 24.31 | 0.9809 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.28 | 227.53 | 0.9935 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.44 | 17.49 | 0.7635 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.1 | 30.0 | 0.9996 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.39 | 22.12 | 1.0077 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.76 | 162.91 | 0.9978 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.49 | 28.72 | 1.0067 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.78 | 300.48 | 1.0008 | flat |

lost on sliding verify_1. Expected hit: 1/4. Noise ±1.7%.

## Follow-ups

- rewrite stage 2 of `sliding verify (q_len 8, page 1, tp8, 2<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on sliding-verify-sk-pseudo-s2-token-group --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.

