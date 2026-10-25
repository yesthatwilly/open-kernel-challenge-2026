---
problem: muse_moe_w4a16
slug: marlin-two-class-block-m-decode32
created: 2026-10-24T140229Z
base: master
area: MoE align/dispatch + block_size_m pick
change: >
  At num_tokens in (16, 32] only (decode_32 bs=32; no other scored or test shape; bs=33 is above
  32 and stays on master), give cold experts Marlin's 8-row block and keep SGLang's 16 for the
  rest, by splitting each GEMM into two Marlin launches over disjoint expert classes. Add
  _align_small_two_class_kernel, a copy of _align_small_kernel that computes counts as now, then
  small = counts <= 8. Class S (small experts) is padded to a multiple of 8; class L (the rest) is
  padded to a multiple of 16 (SGLang's pick at bs=32). Each class gets its own seg_start/cumsum
  over only its experts, and its own sorted_ids (capacity n + 32*7 for S, n + 32*15 for L,
  sentinel n everywhere unused), block_experts (-1 past used) and n_padded (0 if the class is
  empty). Each local route is stored only in its expert's class buffer; local_ids is written once.
  In kernel_fn at that shape, call moe_wna16_marlin_gemm for w13 twice into the same cache1: first
  moe_block_size=8 with class S's triple, then moe_block_size=16 with class L's. Rows are route
  indices, so the two calls write disjoint rows. Then silu_and_mul once, unchanged. Then w2 twice
  into the same cache3, the same way. Then master's post-norm (fused_post_expert_norm_weighted_sum
  at bs=32). Same workspace, same flags. prepare() compiles the kernel for tokens 17..32 and
  extends _align_self_check: for each class, the per-expert route sets and block counts must equal
  moe_align_block_size(ref_local restricted to that class's experts, bsm, 32). Mechanism:
  per-m-block Marlin cost grows with block_size_m (LESSONS). At bs=32 a uniform 8 lost (16->8
  multiplied the hot experts' blocks: 239 vs ~186 us in next-smaller) and a uniform 32 lost
  (next-larger). Most local experts receive <= 8 routes at bs=32 (~256 routes over 32 heavy-tailed
  experts), so giving only those the cheaper m_block_size_8 tile, while the hot experts keep 16,
  takes the per-expert best of the two measured uniform picks.
expected:
  - shapes: bs=32 *
    direction: faster
    why: "master weakest shape (1.033x, Marlin w13+w2 187.6 us vs a 70.8 us floor): cold experts drop from a 16-row to an 8-row Marlin tile, while hot experts keep 16 and do not multiply blocks"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens <= 16 untouched (SGLang already picks 8 there)"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T140520Z
review_companions: null
hold_reason: "reviewer expects flat: uniform block 8 cost only ~2.4% at bs=32, and the 239 vs 186 us premise is seed noise (unchanged bs=32 rows range 147-265 us); two extra Marlin launches (~2-4 us each) likely eat the rest"
outcome: expired
verdict_detail: "hold expired after 31 h: reviewer expects flat: uniform block 8 cost only ~2.4% at bs=32, and the 239 vs 186 us premise is seed noise (unchanged "
expired_at: 2026-10-25T211329Z
---

# marlin-two-class-block-m-decode32

## Change

At num_tokens in (16, 32] only (decode_32 bs=32; no other scored or test shape; bs=33 is above 32 and stays on master), give cold experts Marlin's 8-row block and keep SGLang's 16 for the rest, by splitting each GEMM into two Marlin launches over disjoint expert classes. Add _align_small_two_class_kernel, a copy of _align_small_kernel that computes counts as now, then small = counts <= 8. Class S (small experts) is padded to a multiple of 8; class L (the rest) is padded to a multiple of 16 (SGLang's pick at bs=32). Each class gets its own seg_start/cumsum over only its experts, and its own sorted_ids (capacity n + 32*7 for S, n + 32*15 for L, sentinel n everywhere unused), block_experts (-1 past used) and n_padded (0 if the class is empty). Each local route is stored only in its expert's class buffer; local_ids is written once. In kernel_fn at that shape, call moe_wna16_marlin_gemm for w13 twice into the same cache1: first moe_block_size=8 with class S's triple, then moe_block_size=16 with class L's. Rows are route indices, so the two calls write disjoint rows. Then silu_and_mul once, unchanged. Then w2 twice into the same cache3, the same way. Then master's post-norm (fused_post_expert_norm_weighted_sum at bs=32). Same workspace, same flags. prepare() compiles the kernel for tokens 17..32 and extends _align_self_check: for each class, the per-expert route sets and block counts must equal moe_align_block_size(ref_local restricted to that class's experts, bsm, 32). Mechanism: per-m-block Marlin cost grows with block_size_m (LESSONS). At bs=32 a uniform 8 lost (16->8 multiplied the hot experts' blocks: 239 vs ~186 us in next-smaller) and a uniform 32 lost (next-larger). Most local experts receive <= 8 routes at bs=32 (~256 routes over 32 heavy-tailed experts), so giving only those the cheaper m_block_size_8 tile, while the hot experts keep 16, takes the per-expert best of the two measured uniform picks.

## Expected

- `bs=32 *` faster: master weakest shape (1.033x, Marlin w13+w2 187.6 us vs a 70.8 us floor): cold experts drop from a 16-row to an 8-row Marlin tile, while hot experts keep 16 and do not multiply blocks
- `bs=16 *` flat: control: num_tokens <= 16 untouched (SGLang already picks 8 there)
- `bs=128 *` flat: control: untouched

## Risk

Risk: each extra Marlin launch costs a graph node plus a grid that exits fast when its class is empty or small (~2-4 us each, two extra). The saving must exceed ~5-8 us on 187.6 us. The 8-vs-16 per-block cost gap for 1-8 real rows is unmeasured directly. If it wins, extend it to bs=128/256 (S=8, L=master pick 16/32) on top of triton-map-align-verify-chunked, and ab-tune the hot class larger (32 at bs=32). Reviewer: check that Marlin accepts n_padded=0 (an empty class) without writing, and that the shared lock workspace is clean between the back-to-back calls (master already runs w13 then w2 on one workspace).

## Adversarial review

Reviewed 2026-10-24T140520Z by an adversarial subagent (model opus).

**Verdict: proceed.** The idea is legal, new and implementable, but the upside is small. Running uniform block 8 at bs=32 cost only about 2.4% on decode_32 (0.964x vs about 0.988x for parity-equivalent rows), so taking the per-expert best of 8 and 16 can win at most a few percent on one load (about 0.5% geomean), and two extra Marlin launches (about 4-8 us) could cancel that. Proceed only because nothing else useful is queued for decode_32, judge it on the bs=32 row alone, and expect flat.

### Justification
The mechanism is plausible: Marlin's per-m-block cost grows with block_size_m (next-larger at bs=32 lost hard, 0.785x), and a class split leaves every hot expert's block count unchanged. The supporting numbers are weaker than the hypothesis says. (1) The '239 vs ~186 us' Marlin comparison is profile noise and seed (routing) variation, not the effect of block 8. Runs that did not touch bs=32 at all show the bs=32 Marlin kernel sum anywhere from 147.5 to 264.6 us (cap-16-port 147.5, bf16-reduce-verify 158.5, align-single-token-ep 231.6, splitk-rewrite 264.6). In next-smaller the profiled 239 us is even larger than the 215 us median shape time. Each job draws a fresh routing seed, so top_kernels times cannot be compared across jobs. (2) The only clean evidence is the paired ratio against SGLang on the bs=32 row: uniform 8 gave 0.964x, against about 0.987-0.990x for unchanged bs=32 rows in the same period. That is a loss of about 2.4%, barely outside the ±1.5% floor. So the whole 8-vs-16 trade at bs=32 is worth only a few percent. This hypothesis can at best recover the part of that 2.4% that came from hot experts, plus whatever block 8 saves on cold experts, which is unmeasured and probably small. Marlin at 1-8 rows is mostly weight-streaming bound, and a cold expert is one m-block either way, so its weight bytes are the same at 8 and 16. (3) Costs: two extra Marlin launches, each a persistent full-SM grid with its own ramp and tail (the risk section's 2-4 us each), plus a heavier align kernel (two classes, two cumsums, two capacity fills; master's _align_small is already 4.4 us at bs=32). Weakest link: whether block 8 is any cheaper than block 16 for an expert with 1-8 rows. If it is not, the change is a pure loss of about 2-5%. The 'expected' entries are consistent with TASK.md: only bs=32 changes, bs<=16 and bs>=128 are untouched, and bs=33 (a test) stays on master. Be aware that even a 5% decode_32 gain moves the geomean by under 1%, which is inside one run's resolution. Judge it on the paired ratio of the bs=32 row, and do not read a headline change as signal.

### Confounders on the current kernel
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: the new _align_small_two_class_kernel replaces _align_small_kernel (4.4 us) at bs 17..32. Any change in align time adds directly to the bs=32 result. Compare the two align kernels' own times in top_kernels to separate align cost from the Marlin effect.
- post-expert-norm-triton-small-m: no confound at bs=32, which keeps SGLang's fused_post_expert_norm_weighted_sum. bs 17..31 would take _post_norm_combine_kernel through the two-class path, but no test or scored shape lands there, so that path is never exercised. Keep master's SMALL_M_POST_NORM branch exactly as it is.
- marlin-block-m-two-smaller-on-postnorm: does not confound, because it is active only at num_tokens >= 128.
- sglang-marlin-path-parity: sets the common Marlin flags (fp32 reduce, one shared lock workspace, is_ep). Master already runs two Marlin calls back to back on one workspace, and the split makes that four, so this is low risk.

### Interference and A/B needs
No companion. A gain is interpretable: splitting into two launches can only add cost, so a bs=32 gain must come from block 8 on the cold class. A flat or losing result is ambiguous, though: either block 8 has no per-block advantage, or the split overhead ate it. The control that separates the two is the same two-class split with moe_block_size=16 for both classes (class S padded to 16). It measures only the overhead of four Marlin launches and the two-class align. Because the gain side is clean, the expected effect is small, and moe has 4 checks left, mint that control as a follow-up only if the parent comes back flat and someone still wants this direction. Do not run it now.

### Open questions
- Is block 8 actually cheaper than block 16 for an expert with 1-8 rows on H200 Marlin (m_block_size_8 variant vs thread_m_blocks=1)? No run has measured this. It decides the outcome.
- How are local routes and per-expert counts distributed in the decode_32 loads (which TASK.md says are skewed to make this GPU the busiest)? If few experts go above 8 routes, uniform 8 should already have been close to free, and the 2.4% loss would point to the 8-row tile itself being slower. That would argue against this hypothesis.
- Marlin's per-launch fixed cost here (each launch is a persistent grid of SMs x 4 blocks): do two launches, each over half the m-blocks, lose more to ramp and tail than one combined launch?
- Can moe_wna16_marlin_gemm's host wrapper take a sorted_ids capacity that is not the one moe_align_block_size would produce? Check any host-side divisibility or size checks against S capacity 480 (60 blocks of 8) and L capacity 736 (46 blocks of 16).

