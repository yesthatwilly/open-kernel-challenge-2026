---
problem: muse_decode_layer
slug: marlin-block-m-cap-16
created: 2026-10-23T225545Z
base: master
area: w13 marlin
change: >
  Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks
  BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9,
  which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as
  chosen and clamps the T=128/256 picks to 16. Mechanism: routing is skewed (a few experts take
  most tokens), so with block 48/64 the many cold experts holding 1-4 tokens each are padded to a
  48/64-row MMA tile and the hot expert gets only 4-6 tiles to spread over 132 SMs; block 16 pads
  cold experts 3-4x less and gives the hot expert 3-4x more m-blocks for the same 25 MB weight
  stream. The same block_size_m goes to moe_align_block_size and both moe_wna16_marlin_gemm calls
  (moe_block_size), as it must. Nothing else changes. Attribution: only T=128 and T=256 shapes
  (load=verify_16 and verify_32) can move; every other shape keeps its current block size and is
  the control.
expected:
  - shapes: "*load=verify_32*"
    direction: faster
    why: "T=256: block 64 -> 16; the hot expert gains m-block parallelism and cold experts stop paying 64-row padding"
  - shapes: "*load=verify_16*"
    direction: faster
    why: "T=128: block 48 -> 16, same mechanism"
  - shapes: "*load=decode_1*"
    direction: flat
    why: keeps block 8; control
  - shapes: "*load=decode_32*"
    direction: flat
    why: keeps block 16; control
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 4
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T053040Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T053040Z
base_sha: 66e598a7966dd1ac8380100ad95bc3a0a4d21449
base_blob: 6884d26a928d1de1b07bf8d28e1be4e53dfdcfc5
branch: hyp/muse_decode_layer/marlin-block-m-cap-16
worktree: .worktrees/muse_decode_layer/marlin-block-m-cap-16
mode: check
implemented_at: 2026-10-24T053147Z
impl_commit: 9a0492ffc41d11d5f4dd958b1838e238df63cd49
submission_sha256: 90ab84928d8ef488a0d6cfd3efb112d1d09ebec5cb93035bd1d55e6c7e33fc67
norm_sha256: a0b7b726786a9cdcf0ceb09b25226d60ca943727f1bc26e5bab11fc8c6be3a11
submission_blob: de98030423860cfc9507b61d19f82a416223a23d
lint_warnings: 1
attempt_file: 2026-10-24T053147Z-check-marlin-block-m-cap-16.md
prepared_at: 2026-10-24T053147Z
job: 0730135cdcac43b7
dispatched_at: 2026-10-24T053154Z
outcome: gain
verdict_detail: gain
speedup: 1.0620874943175542
speedup_ci: [1.046156181902791, 1.0780188067323175]
harvested_at: 2026-10-24T075010Z
best_at_harvest: 1.020038644953596
promotion: submitted
promotion_commit: ef7dea6cac43678c09ae91b20ece2c40e0670c4b
promotion_attempt: 2026-10-24T075012Z-submit-marlin-block-m-cap-16.md
promotion_job: 2b1f82408ab64116
promotion_at: 2026-10-24T080343Z
---

# marlin-block-m-cap-16

## Change

Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9, which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as chosen and clamps the T=128/256 picks to 16. Mechanism: routing is skewed (a few experts take most tokens), so with block 48/64 the many cold experts holding 1-4 tokens each are padded to a 48/64-row MMA tile and the hot expert gets only 4-6 tiles to spread over 132 SMs; block 16 pads cold experts 3-4x less and gives the hot expert 3-4x more m-blocks for the same 25 MB weight stream. The same block_size_m goes to moe_align_block_size and both moe_wna16_marlin_gemm calls (moe_block_size), as it must. Nothing else changes. Attribution: only T=128 and T=256 shapes (load=verify_16 and verify_32) can move; every other shape keeps its current block size and is the control.

## Expected

- `*load=verify_32*` faster: T=256: block 64 -> 16; the hot expert gains m-block parallelism and cold experts stop paying 64-row padding
- `*load=verify_16*` faster: T=128: block 48 -> 16, same mechanism
- `*load=decode_1*` flat: keeps block 8; control
- `*load=decode_32*` flat: keeps block 16; control

## Risk

Risk: SGLang's 0.9 occupancy rule exists because at large per-expert M a bigger tile reuses each weight fragment across more rows from registers/smem; with 16-row tiles Marlin re-streams the hot expert's 25 MB weight once per m-block from L2 (an 80K-token-context H200 has 50 MB L2, so one expert's w13 fits, w2 fits), which could make verify_32 slower instead. Either sign is a clean attribution to the tile choice; if slower, queue the opposite (cap at 32) as an ab-tune. Correctness unaffected (same fp32 reduce, same math). If this lands, a follow-up should also try block 8 at T=32 (decode_32), which the 0.9 rule just misses.

## Adversarial review

Reviewed 2026-10-24T053040Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A one-knob tile-size probe whose only moving shapes are verify_16 (T=128, block 48->16) and verify_32 (T=256, block 64->16); the stated mechanism (padding bytes, m-block parallelism) is wrong, but the sister problem's measurements that larger thread_m_blocks cost 18-28% on Marlin at T=32/128 with the m-block count unchanged make the direction credible, and either sign is cleanly attributable from the single run. No companions.

### Justification
A. Arithmetic of the pick is verified against submission.py:257-259 (break keeps the first candidate with T*8/32/bsm < 0.9; T<=28 -> 8, 32 -> 16, 128 -> 48, 256 -> 64, the last by loop exhaustion). Profile (job 5755a412dda244a1): the two Marlin launches are 425/402 us of verify_16 (73%/54%) and 512/552 us of verify_32 (73%/49%) against SoL floors of 127-267 us for the whole layer, so a 25% Marlin change moves those shapes ~12-18% and the geomean ~4-6%, far outside the +-1.5% CI; the expected table is consistent with TASK.md (only the two verify_16/verify_32 loads carry T=128/256; all other scored shapes and all 13 tests have T<=33 and keep their block, so they are true controls, including decode_16 and verify_1 which the table omits but which also keep 8). The mechanism as written does not hold: Marlin parallelizes over n-tiles x k-tiles x m-blocks (8192/128 = 64 n-tiles per m-block already feed 132 SMs x 4 blocks), padding cold experts to 48/64 rows costs MMA issue, not bytes, and giving the hot expert 3-4x more m-blocks means 3-4x more weight passes and dequant for that expert (served from L2 only if the consecutive m-blocks of one expert stay resident; L2 bandwidth on Hopper is not much above the 4.8 TB/s HBM figure). What actually supports the direction is empirical: muse_moe_w4a16 marlin-block-size-m-next-larger (job 2639e3ca7f5549de) measured thread_m_blocks 1->2 at T=32 as 208 -> 261 us (~+28% on Marlin) and 3->4 at T=128 as 462 -> 545 us (~+18%), both with the m-block count essentially unchanged (its review counted ~32 local routes at T=32 and ~128 at T=128, i.e. one block per expert at either size), and its reviewer measured the kernel at ~1.3 TB/s, i.e. not bandwidth-bound. A per-m-block cost of roughly a*(1 + 0.4*thread_m_blocks) fits both points and, extrapolated down to thread_m_blocks=1, predicts Marlin 25-35% faster at T=128/256 before the extra m-blocks of hot experts (at T=256 a 100-route expert needs 7 blocks of 16 instead of 2 of 64, so total m-blocks grow maybe 30-40%), net perhaps 10-25%. Weakest link: the extrapolation below SGLang's own pick. The two sister points sit above the pick; nothing measured yet says the curve keeps falling at thread_m_blocks=1 for large padded M, and SGLang's 0.9 rule may have been tuned to exactly this crossover (though if the rule is vLLM's global-E heuristic applied to the 32 local experts, it picks 8x larger blocks than vLLM would at E=256, which would favour the hypothesis). A secondary weakness: the m-block count at T=256 depends on how skewed the local routing is, and Marlin time already varies 10-40% between the sliding and full inputs at the same T in the profile, so verify_32 sliding and full may move by different amounts; the paired per-shape ratio still isolates the change.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8 (the only landed item): its one non-parity change is ROUTER_FP32_MAX_TOKENS 8->1, which alters kernels only at T=8 (verify_1); at T=128/256 the router was already hi/lo in SGLang, routing/topk_ids feeding moe_align_block_size are produced identically, and the Marlin path is byte-for-byte SGLang's. No interaction with the block-size pick; the change would behave the same with it absent.

### Interference and A/B needs
Reverting the landed hypothesis is meaningless (master would be the 0.0045x starter) and nothing in it touches T=128/256. A gain on verify_16/verify_32 with the other ten shapes at ~1.0x says block 16 beats 48/64 at that M; a loss says it does not; both are unambiguous attributions to the tile choice because the diff is one pick and the same-run SGLang column is the control. The only bundled side effect is moe_align_block_size's host-sized capacity shrinking with block size (T*8 + 256*(bsm-1): 18176 -> 5888 slots at T=256), which changes its fill kernels by ~1-2 us against 500 us of Marlin, not enough to flip a sign. A cap-32 ab-tune or a block-8 probe would be a tuning sweep to find the optimum, not a disambiguation, so it belongs in the follow-up queue after this result, not as a companion. The in-flight expert-id-map-single-kernel and the queued moe-align-* hypotheses touch the align launch sequence but not block_size_m; they are not duplicates and gains may simply not add.

### Open questions
- Which Marlin template (thread_m_blocks) actually runs at T=128/256 cannot be read from the saved profile (names truncated at 'Marlin<__nv_bfloat16, 56...'); if the implementer can print the launched config or the padded block count in prepare()/log, it would confirm 48 -> 16 maps to thread_m_blocks 3 -> 1 and 64 -> 16 to 4 -> 1.
- If verify_16 and verify_32 move in opposite directions, read it as the m-block growth of hot experts at T=256 (block 16 multiplies their weight passes) outweighing the per-block saving; the next probe is cap 32 for T=256 only, and block 8 (the special m_block_size_8 path, which the sister data shows slightly ahead of 16 at T<=16) for T=32/128.
- Does the SGLang moe_align_block_size + Marlin m-block loop walk the device num_tokens_post_pad or the host-sized capacity (the question moe-align-local-expert-count raises)? If the latter, block 16 at T=256 walks 368 instead of 284 slots and part of any gain or loss belongs to that interaction; combined later with moe-align-local-expert-count the two gains will not simply add.
- The hypothesis's own expected magnitudes are unstated; a useful falsification threshold is: Marlin at T=128 should drop by more than the ~18% the sister measured for 48->64 if per-block cost is the driver; a change inside +-5% would say the per-block cost model saturates at thread_m_blocks<=3 and SGLang's pick is already near-optimal.

## Implementation notes

- `muse_decode_layer/submission.py`: new module constant `MARLIN_BLOCK_SIZE_M_CAP = 16` beside `BLOCK_SIZE_M_CANDIDATES`, and in `_experts` one line after SGLang's unchanged selection loop: `block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP)`. That single variable already feeds `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` and `common["moe_block_size"]`, which both `moe_wna16_marlin_gemm` calls (w13 and w2) take, so all three see the same capped size. Diff: 1 file, +3 lines.
- Branches only on `num_tokens` (a shape) and constants; no input values, no syncs, nothing cached.
- Picks per scored T (T*8/32/bsm < 0.9 rule, T/4 routes per local expert on average):
  | T | before | after |
  |---|---|---|
  | T <= 28 (decode_1, decode_16, verify_1, ...) | 8 | 8 |
  | 32 (decode_32) | 16 | 16 |
  | 128 (verify_16) | 48 | 16 |
  | 256 (verify_32; loop exhausts at 64) | 64 | 16 |
- Mechanism caveat (from the adversarial review, recorded so the result is read correctly): the reason to expect a gain is the per-m-block cost of larger thread_m_blocks Marlin templates, not padding bytes or SM parallelism. Sister problem muse_moe_w4a16 (job 2639e3ca7f5549de) measured going *up* in thread_m_blocks costs +18-28% on Marlin (1->2 at T=32: 208 -> 261 us; 3->4 at T=128: 462 -> 545 us) with the m-block count essentially unchanged. Block 16 does add m-blocks for hot experts (more weight passes), so the net is that per-block saving minus the m-block growth; opposite signs on verify_16 vs verify_32 would point at the latter at T=256.
- No deviation from the spec. The review's open question (log the launched thread_m_blocks) was not added: it would need a print/host-side side effect outside the one-knob change.
- moe_align_block_size's host-sized capacity shrinks with block size (review: 18176 -> 5888 slots at T=256), a bundled ~1-2 us side effect.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/marlin-block-m-cap-16 diff` (+3 lines, constant + `min()` on a host int).
- [x] CUDA-graph capturable: the cap is a Python `min` on a host int derived from `x.shape`; no `.item()`/`.cpu()`/sync/prints added; no allocation or control flow on input values.
- [x] Nothing derived from `data` is cached across calls (no state writes added).
- [x] Weights: n/a: weights and `prepare()` untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: unchanged, same dtypes into Marlin, same fp32 reduce.
- [x] Attention sees every key: n/a: attention path untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept.

okc_lint output:
```
warn  host-sync    L247: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/marlin-block-m-cap-16/muse_decode_layer/submission.py
```
`python3 -I -m py_compile` passed; `__pycache__` removed.

## Result

- job `0730135cdcac43b7` (check), status done, passed True, tests 13/13
- speedup 1.0621x, 95% CI [1.0462, 1.0780], score 21297.962852424607 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.020038644953596, "ci": [1.004738065279292, 1.0353392246278998], "job": "5755a412dda244a1"}; base result: muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.result.json
- **outcome: gain** (gain)
- files: `muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.result.json`, `muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch`; branch `hyp/muse_decode_layer/marlin-block-m-cap-16`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 168.73 | 171.1 | 0.9892 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 180.65 | 183.78 | 0.9879 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 269.71 | 259.9 | 1.0003 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 440.57 | 432.97 | 1.0021 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 265.68 | 253.85 | 0.9979 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 614.9 | 601.7 | 0.9997 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 233.99 | 234.75 | 1.011 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 250.89 | 240.09 | 1.0156 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 521.15 | 436.67 | 1.154 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 714.3 | 618.93 | 1.1023 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 713.04 | 631.95 | 1.1274 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 1049.76 | 969.88 | 1.0764 | gained |

gained on sliding verify_16, verify_32; full verify_16, verify_32. Expected hit: 4/4. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

