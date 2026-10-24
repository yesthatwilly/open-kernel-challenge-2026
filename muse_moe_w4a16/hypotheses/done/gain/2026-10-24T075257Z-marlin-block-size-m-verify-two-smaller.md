---
problem: muse_moe_w4a16
slug: marlin-block-size-m-verify-two-smaller
created: 2026-10-24T075257Z
base: master
area: w13 marlin
change: >
  Change only the block_size_m pick in kernel_fn, for num_tokens >= 128 only: take SGLang's pick
  (the first of BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) with num_tokens*8/32/block_size_m <
  0.9) and step TWO candidates down: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)].
  Resulting picks, SGLang -> new: bs=128: 48 -> 16 (thread_m_blocks 3 -> 1); bs=256/257: 64 -> 32
  (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps SGLang's pick as the in-run control.
  The same value feeds moe_align_block_size (bank count 32, as on master) and the moe_block_size
  kwarg of both Marlin calls; nothing else changes. Independent of
  marlin-block-size-m-next-smaller (one step down at bs=32/128/256) and run from master in
  parallel with it, so the two give the per-shape gradient of Marlin cost vs block size below
  SGLang's pick, the opposite direction of the next-larger regression (48 -> 64 lost 14% at
  bs=128).
expected:
  - shapes: bs=128 *
    direction: faster
    why: 48 -> 16 puts the verify_16 GEMMs on the thread_m_blocks=1 config; per-block cost fell steeply with smaller blocks in the next-larger run (14% at bs=128, 21% at bs=32), Marlin is 96% of 462 us
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 32 (thread_m_blocks 4 -> 2); Marlin 96% of 621 us, ~6x the SoL floor
  - shapes: bs=1 *
    direction: flat
    why: "control: unchanged pick"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T100034Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T100034Z
base_sha: fc28840b9d1e2741aebe085a82c80e31c12ca710
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller
worktree: .worktrees/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller
mode: check
implemented_at: 2026-10-24T100132Z
impl_commit: 4df94f38320bc45c5ba2fd3dfc548dbfa35251d6
submission_sha256: 53cecd0b198a0c4d1c6ed7271f683ce638be35f284ce62282dfb2c4e342ff066
norm_sha256: fa21ef8fce76f481ec6846b6f730dabce03124ee72750f5713bba4892b4c7daa
submission_blob: d046cd42e462a82dcefc683527d7d97f6953e470
lint_warnings: 1
attempt_file: 2026-10-24T100132Z-check-marlin-block-size-m-verify-two-smaller.md
prepared_at: 2026-10-24T100133Z
job: ed837a4a46d44a95
dispatched_at: 2026-10-24T100141Z
outcome: gain
verdict_detail: gain
speedup: 1.0916165083160805
speedup_ci: [1.0752422606913392, 1.1079907559408215]
harvested_at: 2026-10-24T111614Z
best_at_harvest: 0.9887547325362586
promotion: superseded
promotion_reason: best is now 1.3075759186466844
---

# marlin-block-size-m-verify-two-smaller

## Change

Change only the block_size_m pick in kernel_fn, for num_tokens >= 128 only: take SGLang's pick (the first of BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) with num_tokens*8/32/block_size_m < 0.9) and step TWO candidates down: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Resulting picks, SGLang -> new: bs=128: 48 -> 16 (thread_m_blocks 3 -> 1); bs=256/257: 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps SGLang's pick as the in-run control. The same value feeds moe_align_block_size (bank count 32, as on master) and the moe_block_size kwarg of both Marlin calls; nothing else changes. Independent of marlin-block-size-m-next-smaller (one step down at bs=32/128/256) and run from master in parallel with it, so the two give the per-shape gradient of Marlin cost vs block size below SGLang's pick, the opposite direction of the next-larger regression (48 -> 64 lost 14% at bs=128).

## Expected

- `bs=128 *` faster: 48 -> 16 puts the verify_16 GEMMs on the thread_m_blocks=1 config; per-block cost fell steeply with smaller blocks in the next-larger run (14% at bs=128, 21% at bs=32), Marlin is 96% of 462 us
- `bs=256 *` faster: 64 -> 32 (thread_m_blocks 4 -> 2); Marlin 96% of 621 us, ~6x the SoL floor
- `bs=1 *` flat: control: unchanged pick
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Risk: higher than one step down. At bs=128 the average local expert has ~32 routes (more on hot experts under skewed popularity), so 16-row blocks give 2-4+ m-blocks per expert, each streaming the expert's full NVFP4 w13/w2 slice; if Marlin at this M is weight-bandwidth-bound per m-block rather than per-block-latency-bound, this loses, possibly a lot. Read together with marlin-block-size-m-next-smaller: (one-step gain, two-step loss) brackets the optimum; (both lose) closes the knob at verify. Correctness unaffected; the bs=257 and skewed bs=128 tests run the new picks.

## Adversarial review

Reviewed 2026-10-24T100034Z by an adversarial subagent (model opus).

**Verdict: proceed.** A one-knob probe that is well supported: muse_decode_layer marlin-block-m-cap-16 (1.062x, 0730135cdcac43b7) already moved the same Marlin call 48->16 at T=128 and 64->16 at T=256 and gained 8-15% per shape, and next-larger here showed per-m-block cost rising with thread_m_blocks. It overlaps heavily with marlin-block-size-m-next-smaller on bs=128/256, so only one of the two needs to spend a check on those shapes. Run this one, and cut next-smaller down to its bs=32 (16->8) arm, which is its only part this run does not cover.

### Justification
A. The mechanism is plausible and has been measured. On master, Marlin is 495 of 462 us at bs=128 and 612 of 621 us at bs=256 (the kernel sum exceeds the shape time because of the graph-cost subtraction). Both are 4.5-5.7x their SoL floors (101.3 / 108.9 us), and the next-larger run measured 48->64 at bs=128 as 462->545 us (0.860x), with Marlin at 495->662 us. Going down was measured too, on the identical Marlin call in the sister problem: decode_layer cap-16 gave verify_16 1.10-1.15x and verify_32 1.08-1.13x. LESSONS records it. The hypothesis does not cite that result, and it should be cited, because it moves this from a speculative extrapolation to a near-replication at bs=128. The risk paragraph's m-block arithmetic is wrong by about 8x, and the error favours the change. It assumes ~32 routes per local expert at bs=128. That is SGLang's T*8/32 overcount. The true average is T*8/256 = 4 routes per expert, perhaps 2x that on the busiest GPU (TASK.md: each routing makes this GPU the busiest), so ~8 at bs=128 and ~16 at bs=256. Hot experts are a bit more. So block 16 adds an extra m-block, and with it another pass over the expert's weights, only for a few hot experts. Block 32 at bs=256 adds almost none. The weight-bandwidth failure mode named in the risk is unlikely. Expected directions are consistent with TASK.md: bs=128 and bs=256 are the only moving shapes, each weighted 60, as one load each. A 10-15% Marlin cut there would be about 1.5-2.5% on the geomean, near the edge of the +-1.5% CI, so the verdict must be read per shape. The bs<128 controls are genuine: their pick is unchanged, and the bs=4/33 tests as well as the bs=1 test run it too. The weakest link is bs=256. 64->32 has no direct measurement in this problem (next-larger left bs=256 at 64 because the loop is exhausted, so that shape was a control). The decode_layer evidence is for 64->16, not 32, under different routing (dist=soft vs skewed EP8 with a 32-expert align). The assumed mapping 16 -> thread_m_blocks 1, 32 -> 2 cannot be read from the truncated profile kernel names. So at bs=256 this run may land between the parity pick and the decode-proven best, rather than testing the best.

### Confounders on the current kernel
- sglang-marlin-path-parity (the only landed item) is the base Marlin path. Its one non-SGLang change is moe_align_block_size over 32 experts instead of 256. That changes sorted_ids capacity (M*8 + 32*(bsm-1)), which shrinks with the smaller block. The effect is ~1-2 us of align work against ~500-600 us of Marlin, so it cannot flip a sign. The pick behaves the same whether the align runs over 32 or 256 experts, as the decode_layer run (256-expert align) suggests. Not a confounder that needs an A/B.

### Interference and A/B needs
No companions are warranted. The diff is one host-side pick on num_tokens, applied only at bs>=128, and the four bs<128 shapes plus the same-run SGLang column are in-run controls. A gain at bs=128/256 is attributable to the tile size, and so is a loss. Reverting the landed parity hypothesis is meaningless (that would leave the dense starter). Tuning its align expert count is irrelevant at this magnitude. The running align-256-experts-below-128 touches only bs<128, so it is disjoint from this run's moving shapes. small-m-grouped-gemv-splitk-rewrite touches only bs<=16, also disjoint. Overlap with marlin-block-size-m-next-smaller: both edit the same line from master and touch the same bs=128/256 shapes with disjoint picks (next-smaller: 128->32, 256->48; this one: 128->16, 256->32). Given decode_layer's measured cap-16 gain and the ~4-16 routes per local expert, the cost curve below SGLang's pick is very likely monotone down to 16. Next-smaller's bs=128/256 points would mostly confirm a slope this run measures more strongly. Its unique information is bs=32 16->8 (the 8-row special path at decode_32). Recommendation: run only this one for bs=128/256. Either trim next-smaller to its bs=32 arm (bs=128/256 keep SGLang's pick as controls), or defer it until this result harvests. If both run unchanged from master, they cannot both land, because they edit the same line, and the loser becomes a stacked re-run anyway.

### Open questions
- Should bs=256 take 16 instead of 32? The sister problem measured 64->16 as a gain. A follow-up 'cap 16 at bs>=128' (or 8 at bs=128) is the natural next point if this run gains at both shapes, and it should be queued as a stacked hypothesis rather than folded into this run.
- Can the attempt file record, from the log or profile, the number of padded blocks (n_padded) at bs=128/256 under the old and new picks? That would confirm the m-block growth is small (the expected ~8-16 routes per local expert), not the ~32/64 the risk paragraph assumes.
- If bs=128 gains and bs=256 is flat or loses, read it as the 64->32 step being too short or the thread_m_blocks=2 template not being cheaper at that M, not as hot-expert weight re-streaming. A loss from re-streaming would need far more routes per expert than EP8 skew gives.


## Implementation notes

- One edit in `kernel_fn` (worktree `muse_moe_w4a16/submission.py`), right after SGLang's candidate loop: `if num_tokens >= 128: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(BLOCK_SIZE_M_CANDIDATES.index(block_size_m) - 2, 0)]`, with a 3-line comment. 5 insertions, nothing else touched.
- The same `block_size_m` still feeds `moe_align_block_size(local_ids, block_size_m, bank["count"])` (count 32, as on master) and `moe_block_size` in `common`, so both Marlin calls get it.
- Picks checked by evaluating the loop + new line in plain Python (SGLang -> new): bs 1/4/8/16 8->8, bs 32/33 16->16, bs 128 48->16, bs 256 64->32, bs 257 64->32 (256/257 never break the loop, so they end on 64, index 4 -> 32). Matches the spec.
- No deviation from the spec. Per the review's open question, the attempt file should try to record n_padded at bs=128/256 from the profile/log if visible; no print was added to kernel_fn (it would sync), so this is left to the profile.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller diff` (one 5-line hunk in kernel_fn).

- [x] CUDA-graph capturable: the new branch is on `num_tokens` (a shape) and a Python tuple index; no `.item()`/`.cpu()`/`synchronize()`/tensor prints added; no allocation or control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: prepare() and the repack are untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged (only the tile size of the same Marlin W4A16 calls changes); KV cache / query items n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention in this problem; no routes are skipped (alignment still covers every local route).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

okc_lint output:

```
warn  host-sync    L110: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile` passed; the `__pycache__` it left was removed. Not committed, staged or run.

## Result

- job `ed837a4a46d44a95` (check), status done, passed True, tests 5/5
- speedup 1.0916x, 95% CI [1.0752, 1.1080], score 12319.802694029935 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.result.json`, `muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch`; branch `hyp/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 80.23 | 80.25 | 1.007 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 191.41 | 197.43 | 1.0002 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 208.15 | 209.63 | 1.0003 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 152.04 | 151.59 | 0.9985 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 462.46 | 343.68 | 1.3641 | gained |
| bs=256 load=verify_32 sharding=ep8 | 60 | 621.28 | 468.68 | 1.3196 | gained |

gained on verify_16 verify_16; verify_32 verify_32. Expected hit: 6/6. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

