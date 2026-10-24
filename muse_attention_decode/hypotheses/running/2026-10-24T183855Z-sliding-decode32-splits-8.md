---
problem: muse_attention_decode
slug: sliding-decode32-splits-8
created: 2026-10-24T183855Z
base: master
area: sliding-layer attention (batched decode, split depth)
change: >
  Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8
  (today 16 at every batch). At batch 32 that is 512 programs of 256 old keys (4 BLOCK_N tiles)
  instead of 1024 programs of 128 keys, and half the partials and ticket merges. The workspace
  sizing and counters follow `splits` as now; nothing else changes.
expected:
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: _sliding_fused_kernel takes 22.1 us at decode_32 for 16.8 MB (0.76 TB/s) with 1024 four-warp programs, more than one resident wave plus 1024 partial stores and 64 ticket merges of 16 partials; 8 splits was never measured at B = 32 (S = 2 lost to S = 16 there). Expect 5-10% on the kernel
  - shapes: "*layer=sliding load=decode_1 * *layer=sliding load=decode_16 *"
    direction: flat
    why: batch <= 16 keeps 16 splits
  - shapes: "*layer=full* *load=verify_*"
    direction: flat
    why: not on this path
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T184104Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T184104Z
base_sha: 7636e41481d93a8dc77886fc9674dcdb6c52067e
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/sliding-decode32-splits-8
worktree: .worktrees/muse_attention_decode/sliding-decode32-splits-8
mode: null
---

# sliding-decode32-splits-8

## Change

Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8 (today 16 at every batch). At batch 32 that is 512 programs of 256 old keys (4 BLOCK_N tiles) instead of 1024 programs of 128 keys, and half the partials and ticket merges. The workspace sizing and counters follow `splits` as now; nothing else changes.

## Expected

- `*layer=sliding load=decode_32 *` faster: _sliding_fused_kernel takes 22.1 us at decode_32 for 16.8 MB (0.76 TB/s) with 1024 four-warp programs, more than one resident wave plus 1024 partial stores and 64 ticket merges of 16 partials; 8 splits was never measured at B = 32 (S = 2 lost to S = 16 there). Expect 5-10% on the kernel
- `*layer=sliding load=decode_1 * *layer=sliding load=decode_16 *` flat: batch <= 16 keeps 16 splits
- `*layer=full* *load=verify_*` flat: not on this path

## Risk

Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches).

## Adversarial review

Reviewed 2026-10-24T184104Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a cheap, config-only probe on one shape. A single S=8 run can be read on the sliding decode_32 _sliding_fused_kernel row either way, so it needs no companion. Its headline effect (~0.3-0.9%) will sit inside the CI, so judge it per shape only.

### Justification
The mechanism is plausible, and the profile argues for it more strongly than the hypothesis says. On master, _sliding_fused_kernel takes 12.9 us at B=16/S=16 (512 CTAs, 2 tiles each, 8.4 MB) and 22.1 us at B=32/S=16 (1024 CTAs, 2 tiles, 16.8 MB). The grid doubles with the same per-CTA work, and the time nearly doubles (1.71x). That fits two residency waves, or saturated per-SM concurrency. It does not fit a single resident wave, where 2 tiles per CTA would cost about the same at either batch. The B=16 points (S=4/8 tiles 23.9 us vs S=16/2 tiles 12.9 us) give about 1.8 us per extra serial tile with an intercept near 9 us. If roughly 512 CTAs fit in one wave on 132 SMs (about 4 resident per SM at 4 warps with num_stages=2), S=8 at B=32 is one wave of 4 tiles, about 9 + 4*1.8 = 16-17 us vs 22.1 us. num_stages=2 also gets real overlap on a 4-iteration loop instead of only a prologue on 2. The combine work halves too (64 merges of 8 x 4 KB partials instead of 16), which shortens the tail. So 5-10% is conservative, and up to about 25% on the kernel is possible. The expected shapes and directions match TASK.md: only sliding decode_32 (weight 46, floor 3.3 us) changes code path. decode_1 and decode_16 keep S=16 and are byte-identical. Full and verify shapes do not take this path. Headline arithmetic: the decode_32 load is 46*24.2 + 16*241.9 = 4984 us. Saving 2-5.5 us per sliding call is 1.8-5% on that load, about 0.3-0.9% geomean, which cannot clear the CI. Weakest link: the residency assumption. The earlier reviewer estimated ~7-8 CTAs/SM, which would mean 1024 CTAs already roughly fit in one wave. In that case the 1.71x scaling means the kernel is limited by per-SM memory concurrency, not waves. Halving the CTAs then halves bytes in flight per SM, and serial tiles double, so S=8 would read flat or slower (about +2 tiles * 1.8 us). The run settles this one way or the other. It cannot mislead.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: this landed hypothesis set the S=16 value being retuned, so it is the incumbent, not a confounder. Reverting it would send B=32 back to _num_splits (S=2), which is already measured at 38.5 us.
- sliding-fused-triton-decode1-gate: owns the kernel body and the B=1 path. B=1 keeps S=16, so it has no interaction. Its decode_1 row drifted 4.8% between runs with identical code, which sets the noise floor for reading sliding rows.
- fused-preattention-triton, fa3-hdim128-head-pairing, paired-heads-launch-cleanup, paired-heads-launch-cleanup-ab-tune-batch-gate, full-decode-cuda-gather-splitkv: none of these touch sliding decode (they are window == 0 only, or their pre-attention launch is not on the fused sliding path). Their shapes are separate CUDA graphs, so they share no launch sequence, memory traffic or numerics. Their rows serve as in-run drift controls.

### Interference and A/B needs
No companion is warranted. The change is one scalar on one shape's path, and every landed hypothesis on other shapes is disjoint. A gain on the decode_32 kernel row means S=8 beats S=16. A loss means it does not. Neither reading depends on any landed change. The only ambiguity is what a loss implies about the other direction (S=32, one tile per program, 2048 CTAs). That calls for a stacked follow-up, not a companion, because S=8 losing is a complete answer for this hypothesis. An ab-revert has nothing meaningful to revert. A control is pointless because the change has no sub-changes.

### Open questions
- Gate the split choice on cfg['batch'] (config), not on any tensor value. Keep the B=1 case explicitly at 16 so a 3-way merge with any B=1 split change stays clean. Example: splits = (16 if batch <= 16 else 8) if length == 1 else _num_splits(batch, hkv).
- Correctness at S=8: chunk = ceil(2047/8) rounded to 64 = 256, and 8*256 >= 2047, so the whole window is covered. Split S-1 holds the new keys and is never empty, so m_max is finite. Empty chunks on the short-context tests (batch 3 ctx 2047 page 64, batch 4 ctx 6000) get weight 0. Counters are keyed on (batch, hkv), not S, and reset by the last ticket, so changing S per batch is safe. All of this should pass, but confirm the test batches use S=16 or S=8 as intended (they are <= 16, so they keep 16 and are unchanged).
- Read occupancy from the profile if it is available (achieved CTAs/SM or waves for _sliding_fused_kernel at B=32). It decides whether to try S=32 next (if occupancy is high and per-SM concurrency is the limiter, S=8 loses and S=32 is unlikely to help either) or num_stages/num_warps tuning.
- Moot-risk: sliding-cuda-cluster-dsmem-onelaunch is running and would replace this kernel on every sliding shape. If it lands as a gain before this is dispatched, hold this, as the hypothesis itself says.
- Noise: sliding rows drift by 2-5% between runs with identical code. A kernel-row change under about 5% should be called flat, not a gain.

