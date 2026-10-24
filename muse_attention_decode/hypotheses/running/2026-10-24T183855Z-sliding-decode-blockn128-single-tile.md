---
problem: muse_attention_decode
slug: sliding-decode-blockn128-single-tile
created: 2026-10-24T183855Z
base: master
area: sliding-layer attention (batched decode, tile size)
change: >
  Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass
  BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split covers ceil(2047 / 16) = 128 old
  keys rounded up to BLOCK_N, so each program now runs its old-key loop once, issuing the
  page-index load and the whole chunk's K and V loads in one step instead of two dependent
  iterations. num_warps stays 4, num_stages 2, the new-key block, the masks, the ticket combine
  and verify (length 8, which keeps BLOCK_N 64) are unchanged.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "_sliding_fused_kernel is 10.4 us for 0.5 MB: a latency chain (q norm, idx load, KV load, dot, partial store, ticket, reload). One gather round trip instead of two per program should save ~0.5-1.5 us"
  - shapes: "*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *"
    direction: faster
    why: same chain, 12.9 / 22.1 us kernels; smaller relative gain
  - shapes: "*layer=full* *load=verify_*"
    direction: flat
    why: not on the sliding decode path
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T184058Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T184058Z
base_sha: 591234f46b4e7e6b5307dd6a14ed904220f58742
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/sliding-decode-blockn128-single-tile
worktree: .worktrees/muse_attention_decode/sliding-decode-blockn128-single-tile
mode: null
implemented_at: 2026-10-24T184519Z
impl_commit: e5892c6cdb4175b793ed05d56834089ccc8dc8d5
submission_sha256: df6470f19463c48cd61417096d3b2ce3199e3468ba523073f251b29814666f96
norm_sha256: 055479c9f389631efceec3a962f7417b4cc245cc3787c6db6be6018845e02271
submission_blob: 5118538da13178bac467b5449d0bdd25010ca10d
lint_warnings: 0
---

# sliding-decode-blockn128-single-tile

## Change

Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split covers ceil(2047 / 16) = 128 old keys rounded up to BLOCK_N, so each program now runs its old-key loop once, issuing the page-index load and the whole chunk's K and V loads in one step instead of two dependent iterations. num_warps stays 4, num_stages 2, the new-key block, the masks, the ticket combine and verify (length 8, which keeps BLOCK_N 64) are unchanged.

## Expected

- `*layer=sliding load=decode_1 *` faster: _sliding_fused_kernel is 10.4 us for 0.5 MB: a latency chain (q norm, idx load, KV load, dot, partial store, ticket, reload). One gather round trip instead of two per program should save ~0.5-1.5 us
- `*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *` faster: same chain, 12.9 / 22.1 us kernels; smaller relative gain
- `*layer=full* *load=verify_*` flat: not on the sliding decode path

## Risk

Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches). Sliding decode_1 carries weight 46 and is 58% of the decode_1 load, so ~1 us there is ~+4% on that load.

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: proceed.** A one-knob config change (BLOCK_N 64->128 on the sliding decode path, S=16) whose per-shape _sliding_fused_kernel rows read cleanly without a companion; the mechanism is plausible, but the gain is modest and may be partly pre-empted by num_stages=2 pipelining already overlapping the second tile's gather.

### Justification
Arithmetic checks out against master: kernel_fn routes every sliding q_len==1 shape to _sliding_forward with splits=16 and BLOCK_N=64; count = 2047 at an 80K context, so chunk = ceil(2047/16)=128 (already a multiple of 64), i.e. exactly two loop iterations per program, and BLOCK_N=128 gives one iteration (the last split gets 127 keys, masked). The profile backs a latency-bound picture: sliding decode_1 is 10.4 us for about 0.5 MB (floor 0.1 us), decode_16 is 12.9 us (floor 1.6), decode_32 is 22.1 us (floor 3.3), all with _sliding_fused_kernel at 100%, so removing a dependent page-index->KV gather round trip (roughly 1 us of HBM latency with a cold L2) is the right kind of lever. Register and shared-memory cost is small: M=16, D=64, a 128x64 FP8 tile becomes 64 bf16 regs/thread for K and for V at 4 warps, with about 32 KB staged at 2 stages. The expected directions agree with the weights: sliding decode_1, 16 and 32 each carry weight 46, and verify and full shapes do not touch this kernel (sliding verify goes through FA3 on master, so the 'verify keeps BLOCK_N 64' clause is moot but harmless). Weakest link: with num_stages=2, Triton's software pipeliner may already issue iteration 2's page-index and K/V loads during iteration 1's dot and softmax, so the 'two dependent round trips' may really be closer to 1.3 round trips, and the saving could be well under 0.5 us, inside the 2-3.5% drift the md itself reports for sliding decode_1. Secondary risk: at decode_32 the grid is 1024 programs (several waves on 132 SMs), so a fatter tile per program that lowers occupancy could make that shape flat or slightly slower. That runs against the 'faster' expectation, but the per-shape rows make it readable either way. Prior: sliding-fused-triton-decode1-splits-32 (also one tile per program at batch 1, reached by doubling S) was neutral, though that change also doubled the combine width, so it does not refute this one.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: set S=16 at every decode batch, which is exactly what makes the chunk 128 keys = 2 tiles of 64; the hypothesis is a tuning of that landed config, not an independent change. With S tuned differently (8 -> 256 keys/program, 32 -> 64) BLOCK_N=128 would mean 2 tiles or a half-empty tile. That is a stacking dependency, not a confound: the measured delta is attributable to BLOCK_N at S=16.
- sliding-fused-triton-decode1-gate: introduced _sliding_fused_kernel (num_warps=4, num_stages=2); the pipelining depth inherited from it decides how much of the second iteration's latency is already hidden, so the gain is conditional on that setting, but it is held fixed here.
- fused-preattention-triton / paired-heads / full-decode-cuda-gather-splitkv / FA3 pairing: not on the sliding decode path; no interaction expected beyond run-to-run drift on full shapes.

### Interference and A/B needs
No companion is warranted. The change is a single constexpr on one kernel that owns 100% of the three target shapes' time, so the _sliding_fused_kernel profile rows on sliding decode_1/16/32 attribute any gain or loss directly. A revert of sliding-deep-splits-on-cuda-splitkv would not isolate anything (it changes the tile count the hypothesis is premised on), and an ab-tune of num_stages (1 vs 2 at BLOCK_N=128) would only explain a null result, not disambiguate a gain or loss. If the result is null, the natural follow-up is a stacked num_stages/num_warps sweep, not a companion now. Judge on the kernel rows, as the md says, because sliding decode_1 headline drift (2-3.5%) is comparable to the expected effect.

### Open questions
- Does the compiled kernel at BLOCK_N=64, num_stages=2 actually pipeline the indirect (page-table -> K/V) loads across the 2-trip dynamic loop? If yes, the expected saving shrinks toward noise; a look at the TTGIR/PTX for cp.async or prefetch of iteration 2 would settle it before or after the run.
- Should num_warps go to 8 together with BLOCK_N=128 (the 128x64 bf16 conversion of K and V doubles per-thread work)? Kept out by design; a candidate follow-up if decode_32 regresses.
- If sliding-cuda-cluster-dsmem-onelaunch (running) lands and replaces this kernel on all sliding tp8/page-1 shapes, this result becomes moot; the scheduler should check that outcome before spending the run, or accept it as a cheap tuning of the fallback.


## Implementation notes

In `_sliding_forward` of muse_attention_decode/submission.py, the kernel launch now passes `BLOCK_N=128 if length == 1 else 64` (was 64) and the adjacent comment was updated. Nothing else changed: num_warps 4, num_stages 2, S=16, kernel body untouched. The file has no `prepare()` and no warm-up, so there is nothing to extend; the new BLOCK_N variant compiles on first call as the old one did (the same constexpr-specialised launch, no extra variant for verify since length 8 keeps 64). No deviation from the spec.

## Rules check

- [x] CUDA-graph capturable: diff only changes a constexpr from a config-derived value (`length`); no syncs, allocation or value-dependent control flow.
- [x] Nothing derived from `data` is cached across calls: diff adds no state.
- [x] Weights: n/a: no weight handling touched.
- [x] Activations/KV dtypes: n/a: unchanged by the diff.
- [x] Attention sees every key: 16 splits x 128 keys cover all 2047 old keys (last split masked); no skipping.
- [x] No timing patches, harness access, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune, no host syncs in kernel_fn.

okc_lint output: `CLEAN: submission.py`. py_compile passed; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only emulation; no GPU on this node).

**Verdict: dispatch.** The diff is exactly the spec: one constexpr (`BLOCK_N=128 if length == 1 else 64`) plus a comment, in `_sliding_forward` only. No gate, no new path, no fallback surface. The kernel body is generic in BLOCK_N: `chunk` rounds up to a BLOCK_N multiple so S*chunk >= count for every count (ctx 10 → split 0 covers all, splits 1-15 empty with m=-inf weighted 0 in the combine; ctx 2047 and the 80K scored shapes → 16 x 128 >= 2047, last split masked at t < end), page 1 and page 64 index the same way via `t // PAGE`, and tl.dot shapes [16,64]x[64,128] are valid. Verify (L=8) never reaches this kernel. Rules unchanged: no syncs, allocations or value-dependent host control flow added; no cached data. okc_lint CLEAN, py_compile OK, __pycache__ removed.

Fixes: none.

Residual risk: num_stages=2 may already pipeline the second 64-key tile, so the gain may sit inside sliding decode_1 drift; decode_32 could be flat from lower occupancy.
