---
problem: muse_attention_decode
slug: paired-heads-launch-cleanup-ab-tune-batch-gate
created: 2026-10-24T104027Z
base: master
area: FA3 call (head packing) gate
change: >
  From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing
  predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 1), where batch = cfg['batch'],
  and use this single predicate both for the pre-attention kernel's q layout (pass it as a
  constexpr, e.g. PAIRED, plus the q buffer shape [tokens, Hq, 128] vs [tokens, Hq, 64]) and for
  the FA3/epilogue branch. When paired is false on a full layer (B=1 decode_1/verify_1, and odd
  Hkv), run exactly master's pre-pairing path. The pre-attention kernel writes the unpadded
  [tokens, Hq, 64] FP8 q. FA3 gets the original k_cache/v_cache views, descale expanded to (B,
  Hkv), and scheduler_metadata=data['scheduler_metadata'] (the judge's). The output goes through
  SGLang's rms_norm_gate on out.reshape(tokens, Hq*64), as before fa3-hdim128-head-pairing.
  Sliding layers are unchanged. B>=16 full layers are byte-identical to the parent.
expected:
  - shapes: "*layer=full load=decode_1 page*"
    direction: faster
    why: "back to the unpaired FA3 + judge metadata path, ~27.9 us (fused-preattention-triton result) against an estimated ~31-32 us for the ungated cleanup: no 128-wide combine (10.3 vs 7.1 us) and no internal prepare launch"
  - shapes: "*layer=full load=verify_1 page*"
    direction: faster
    why: "~30.0 us unpaired against an estimated ~39-40 us ungated: the paired single-super-head verify FA3 (18.3 vs 14.9 us) and combine (12.5 vs 7.3 us) go away"
  - shapes: "*layer=full load=*_16*"
    direction: flat
    why: same code path as the parent; a mismatch measures run-to-run drift
  - shapes: "*layer=full load=*_32*"
    direction: flat
    why: same code path as the parent
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
risk: low
depends_on: paired-heads-launch-cleanup
priority: 2
kind: ab-tune
controls_for: paired-heads-launch-cleanup
review: proceed
status: done
review_at: 2026-10-24T104320Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T104320Z
base_sha: 12230f154820fa60ff8dc5f1e9efca0e1d05c4c9
base_blob: 60fc66432a00f0c6a1e2228f6a0775cc08f492fc
branch: hyp/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate
worktree: .worktrees/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate
mode: check
implemented_at: 2026-10-24T104354Z
impl_commit: f6513e3feb35334cdc64baa24953ca46621839de
submission_sha256: c1520ef7b632076cf4a4f3e9b65ba5c859c2dd8802ddf7bbece8ab2230ba0c7d
norm_sha256: 4535a46c53b9f66d462750c869d90bcae1dec2ef9af0856e2f8a0587fc648433
submission_blob: efc544103e6d303d1f8602caa9439628043a17a7
lint_warnings: 0
attempt_file: 2026-10-24T104354Z-check-paired-heads-launch-cleanup-ab-tune-batch-gate.md
prepared_at: 2026-10-24T104354Z
job: bd64bf9f9e904f02
dispatched_at: 2026-10-24T104412Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2686474414890054
speedup_ci: [1.2496177298666704, 1.2876771531113405]
harvested_at: 2026-10-24T115930Z
best_at_harvest: 1.1705932092473075
promotion: merged
promotion_commit: bfb7bac32c93cb1602f4273c5d2c65f69e1c7d6f
---

# paired-heads-launch-cleanup-ab-tune-batch-gate

## Change

From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 1), where batch = cfg['batch'], and use this single predicate both for the pre-attention kernel's q layout (pass it as a constexpr, e.g. PAIRED, plus the q buffer shape [tokens, Hq, 128] vs [tokens, Hq, 64]) and for the FA3/epilogue branch. When paired is false on a full layer (B=1 decode_1/verify_1, and odd Hkv), run exactly master's pre-pairing path. The pre-attention kernel writes the unpadded [tokens, Hq, 64] FP8 q. FA3 gets the original k_cache/v_cache views, descale expanded to (B, Hkv), and scheduler_metadata=data['scheduler_metadata'] (the judge's). The output goes through SGLang's rms_norm_gate on out.reshape(tokens, Hq*64), as before fa3-hdim128-head-pairing. Sliding layers are unchanged. B>=16 full layers are byte-identical to the parent.

## Expected

- `*layer=full load=decode_1 page*` faster: back to the unpaired FA3 + judge metadata path, ~27.9 us (fused-preattention-triton result) against an estimated ~31-32 us for the ungated cleanup: no 128-wide combine (10.3 vs 7.1 us) and no internal prepare launch
- `*layer=full load=verify_1 page*` faster: ~30.0 us unpaired against an estimated ~39-40 us ungated: the paired single-super-head verify FA3 (18.3 vs 14.9 us) and combine (12.5 vs 7.3 us) go away
- `*layer=full load=*_16*` flat: same code path as the parent; a mismatch measures run-to-run drift
- `*layer=full load=*_32*` flat: same code path as the parent
- `*layer=sliding*` flat: untouched

## Risk

Decides whether pairing must be gated to batch > 1 or whether the launch cleanup alone makes batch 1 acceptable; also the likely promotion candidate (~+4-5% geomean over the ungated cleanup if the estimates hold).

## Adversarial review

Reviewed 2026-10-24T104320Z by an adversarial subagent (model opus).

**Verdict: proceed.** Implementable as written, as a one-predicate edit on the parent's worktree: add `and batch > 1` to `paired = window == 0 and hkv % 2 == 0`. The parent's else-branch already is master's pre-pairing path (64-wide q from the fused kernel, original cache views, descale expanded to (batch, hkv), judge scheduler_metadata, SGLang rms_norm_gate). It cleanly decides whether B=1 should be gated off the paired geometry.

### Justification
Mechanism is sound and well supported. On master the paired path costs B=1 full 38.7 us (decode_1) and 46.0 us (verify_1) against 27.9 / 30.0 us on the unpaired fused-preattention-triton lineage. The top kernels show the loss is mostly intrinsic to pairing at S=1: combine 7.1->10.3 us and 7.3->12.5 us, verify FA3 14.9->18.3 us, plus FA3's internal prepare launch because metadata=None. It is not only the pad/unpad launches, so the parent's cleanup cannot fully recover B=1, and LESSONS line 71 already says 'Gate it on batch'. The gated B=1 path is code that has already been measured (the fused-preattention-triton build), so the 'faster than parent at decode_1/verify_1' expectation is near-certain unless the parent's cleanup unexpectedly brings paired B=1 under ~28/30 us. B>=16 is byte-identical to the parent: predicate true, same constexprs and buffers. Shape weights: decode_1 and verify_1 are two of the twelve scored shapes, so restoring ~10-25% on each is a multi-percent geomean move. Weakest link: the companion's magnitude (and its value as the promotion candidate) rests on the parent's ungated B=1 estimate (~31-32 / ~39-40 us). That estimate is unmeasured. If the cleanup does better than estimated, the gate's margin shrinks toward flat. That result is still interpretable, because the parent and the companion differ only at B=1.

### Confounders on the current kernel
- fa3-hdim128-head-pairing: this is exactly the thing being gated. At B=1 the companion reverts to the pre-pairing path, so the B=1 delta between companion and parent is cleanly 'paired+cleanup vs unpaired'. That is the intended measurement, not a confound.
- fused-preattention-triton: the gated B=1 path uses its PAIRED=False, 64-wide q layout, the same as master's pre-pairing build. The one predicate drives both the constexpr and the FA3 branch in the parent's code, so no layout mismatch is possible as long as the batch term is added to that single `paired` variable and not to a second copy.

### Interference and A/B needs
No further companions (as instructed), and none needed. The parent/companion pair is a clean A/B. The two are identical at B>=16, so any B>=16 difference measures run-to-run drift. They differ only on full decode_1/verify_1, where the companion runs the previously measured unpaired path. Whichever direction B=1 moves, the comparison reads directly: gate wins, a tie, or cleanup-everywhere wins.

### Open questions
- The expected directions are stated relative to the parent, not to master. Against master's BEST, B>=16 full shapes should be faster (the cleanup's gain), not flat. Make sure the harvester compares this run with the parent's run and not with BEST when it scores the expectations.
- Implement it by editing only line 180 of the parent's submission.py (`paired = window == 0 and hkv % 2 == 0 and batch > 1`). Do not duplicate the predicate. Confirm the batch=1 correctness tests at page_size 64 and kv_scale != 1 now take the unpaired branch and still pass. They are master's validated path, but the tests are the check.
- If the parent has not finished yet, check out its committed branch head (hyp/muse_attention_decode/paired-heads-launch-cleanup) before applying the one-line change, so that B>=16 really is byte-identical.


## Implementation notes

Implemented inline by churnsubs on the parent's commit (12230f1): the single predicate in kernel_fn became `paired = window == 0 and hkv % 2 == 0 and batch > 1` (batch = cfg["batch"]). The parent already uses that one predicate for the q layout (PAIRED constexpr and buffer shape) and for the FA3/epilogue branch, and its unpaired branch is the pre-pairing path (unpadded FP8 q, original cache views, descale (B, Hkv), the judge's scheduler_metadata, SGLang rms_norm_gate). B>=16 full layers are byte-identical to the parent. No deviation.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate diff` (one predicate line plus a comment).

- [x] CUDA-graph capturable: the edit is on config values only (cfg window, num_kv_heads and batch); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 and queries FP8 E4M3: unchanged by this edit (both paths already exist on the parent and passed review).
- [x] Attention sees every key: unchanged: both branches attend every key (the paired branch reads the same KV rows two heads at a time).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line marker kept; no triton.autotune added.

okc_lint output:

```
CLEAN: .worktrees/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate/muse_attention_decode/submission.py
```

## Result

- job `bd64bf9f9e904f02` (check), status done, passed True, tests 16/16
- speedup 1.2686x, 95% CI [1.2496, 1.2877], score 2968.7522922676703 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.1705932092473075, "ci": [1.1530343111085979, 1.1923944446416679], "job": "facc479992e24a35"}; base result: muse_attention_decode/hypotheses/done/gain/paired-heads-launch-cleanup.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.result.json`, `muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.patch`; branch `hyp/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 16.33 | 16.56 | 0.9984 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 30.51 | 28.09 | 1.095 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 20.58 | 20.92 | 0.9904 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 155.95 | 156.19 | 0.9989 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 25.11 | 25.0 | 1.0123 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 297.97 | 298.45 | 0.996 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.16 | 17.15 | 1.0098 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 36.39 | 30.23 | 1.217 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.45 | 22.89 | 0.9952 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.48 | 163.33 | 1.0053 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.26 | 28.93 | 1.004 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.59 | 301.68 | 1.0025 | flat |

gained on full decode_1, verify_1. Expected hit: 5/5. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: paired-heads-launch-cleanup-ab-tune-batch-gate`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

