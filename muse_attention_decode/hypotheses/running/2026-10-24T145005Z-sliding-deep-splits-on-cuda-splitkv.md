---
problem: muse_attention_decode
slug: sliding-deep-splits-on-cuda-splitkv
created: 2026-10-24T145005Z
base: master
area: sliding attention, batched decode, split depth
change: >
  Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding
  decode_16 gained 33%, decode_32 +3%, every other shape flat) onto master
  full-decode-cuda-gather-splitkv (1.347x). The result is a clean 3-way merge (git merge-file, no
  conflicts): the fused sliding kernel's gate widens from batch == 1 to every batch at q_len 1,
  and _sliding_forward uses 16 splits for decode (length 1). The CUDA split-KV path (full layers,
  1 < batch <= 64, q_len 1) and everything else are master's. Shapes are disjoint: the merge
  touches only sliding decode.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: +33% measured on the previous master
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: +3% measured; may read flat
  - shapes: "*layer=full*"
    direction: flat
    why: master CUDA/FA3 paths unchanged
  - shapes: "*load=verify*"
    direction: flat
    why: FA3, unchanged
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164510Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T164510Z
base_sha: b1c16f8f1b1394bbe7c86189ca2cccf026a288d2
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/sliding-deep-splits-on-cuda-splitkv
worktree: .worktrees/muse_attention_decode/sliding-deep-splits-on-cuda-splitkv
mode: null
---

# sliding-deep-splits-on-cuda-splitkv

## Change

Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto master full-decode-cuda-gather-splitkv (1.347x). The result is a clean 3-way merge (git merge-file, no conflicts): the fused sliding kernel's gate widens from batch == 1 to every batch at q_len 1, and _sliding_forward uses 16 splits for decode (length 1). The CUDA split-KV path (full layers, 1 < batch <= 64, q_len 1) and everything else are master's. Shapes are disjoint: the merge touches only sliding decode.

## Expected

- `*layer=sliding load=decode_16*` faster: +33% measured on the previous master
- `*layer=sliding load=decode_32*` faster: +3% measured; may read flat
- `*layer=full*` flat: master CUDA/FA3 paths unchanged
- `*load=verify*` flat: FA3, unchanged

## Risk

Minted by churnsubs to restack a harvested per-shape win onto the new master; merged file at /home/user/.claude/jobs/7156a0da/tmp/merge/a_merged.py.

## Adversarial review

Reviewed 2026-10-24T164510Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean restack of two shape-disjoint, already-measured changes. diff master vs a_merged.py is exactly the parent's two config-only edits (sliding gate widened to every batch at q_len 1, splits = 16 for decode), and both parents passed 16/16 with this exact sliding code and master's full-layer code. Expect about +1.4% headline (~1.366x), mostly from sliding decode_16. That is near the CI edge, but it is safe for a scored submit.

### Justification
The mechanism is already measured, not just inferred. In job 179dd013d3a347e4, sliding decode_16 went from 20.8 to 15.5 us (_sliding_fused_kernel 12.8 us vs FA3 9.9 + combine 3.3 + prepare 1.8 + pre-attn 1.7 + norm) and sliding decode_32 from 25.2 to 24.7 us (22.1 us kernel). decode_1 stayed byte-identical (_num_splits(1,2) == 16). Verify and full shapes were untouched. The merge leaves master's CUDA split-KV path (window == 0 only) and all of the q_len 8 FA3 path alone, so the per-shape expectations are consistent. Load arithmetic (TASK.md weights 46/16): decode_16 load goes from 46*20.8 + 16*123.0 = 2925 us to ~2681 us (~1.09x on that load). The decode_32 load is ~1.00-1.01x. The geomean over 6 loads is ~1.015x, so the headline is ~1.366x vs best 1.3468 [1.3266, 1.3670]. The CI low will probably still overlap best's CI high, so expect 'neutral/unproven-vs-best' again. Judge it per shape, as with the decode1-gate merge-by-hand precedent in DECISIONS.md. Weakest link: the decode_32 'faster' claim. The fused kernel at B=32 (1024 CTAs) costs 22.1 us of kernel time against FA3's 13.2 us main kernel. The shape gain was only ~2%, which is inside the ~1.5-2.5% per-shape drift in LESSONS, so decode_32 should be expected flat and could read slightly slower. The design is far from its floor at B=32 (3.3 us SoL): the serial tile count is no longer the limiter there, and occupancy or combine cost probably is. Master has not made this moot: the incumbent sliding decode_16/32 path is still FA3 (FINDINGS profile).

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: no confound. Its gate requires window == 0, and the sliding gate requires window != 0. The CUDA path uses its own workspace (no _COUNTERS), and its import-time self-check calls kernel_fn only on window-0 synthetic data, so the widened sliding gate never sees it. The full rows' numbers come from master's code, so only sliding decode_16/32 should move.
- sliding-fused-triton-decode1-gate: this is the code being widened, not a confounder. B=1 keeps S=16 and the same predicate result, so decode_1 is byte-identical.
- fused-preattention-triton: on sliding decode_16/32 its _pre_attention_kernel launch is removed (the fused kernel does norm/rope/cache store itself). That is part of the measured gain, not an interaction. Full and verify use it unchanged.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / -ab-tune-batch-gate: full layers only (window == 0), so they share no launch, memory traffic or numerics with the sliding decode path.

### Interference and A/B needs
No companion is needed. Each sub-change has already been isolated: the gate widened at _num_splits depth (sliding-single-launch-fused-attention: 23.9/38.5 us), S=16 at every batch (parent: 15.5/24.7 us), and the CUDA full path (master: 123.0/257.7 us). The shapes are disjoint, so the per-shape rows attribute any change directly. A gain or a loss on the sliding decode rows can only come from the two merged lines, and full/verify rows act as in-run controls against drift. If decode_32 reads slower, the follow-up is a stacked ab-tune (gate the fused kernel to batch <= 16 so B=32 returns to FA3, or S=32 at B=32), not a companion now. Checks are spent anyway.

### Open questions
- decode_32: is the fused kernel at B=32 (22.1 us vs FA3 main 13.2 us) worth keeping? If the scored run shows sliding decode_32 at or below master's 25.2 us within drift, consider a follow-up that gates the fused path to batch <= 16, or tries S=32 at B=32.
- Harvest: read the log tail for '[splitkv] load_inline compiled' and three self-check lines. A '[splitkv] SELF-CHECK FAILED' or load_inline failure would mean the full rows measured the FA3 fallback, which would muddle the headline but not the sliding rows.
- Compile budget: CUDA load_inline (~41 s) plus the Triton fused-kernel variants (PAGE 1/64, ROPE on, S=16) are the same set as the two parents compiled separately, which should be well inside 15 minutes.

