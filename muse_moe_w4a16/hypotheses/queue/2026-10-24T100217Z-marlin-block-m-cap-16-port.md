---
problem: muse_moe_w4a16
slug: marlin-block-m-cap-16-port
created: 2026-10-24T100217Z
base: master
area: w13 marlin
change: >
  Port muse_decode_layer marlin-block-m-cap-16 (landed gain 1.062x, job 0730135cdcac43b7; patch in
  muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch) to kernel_fn in
  muse_moe_w4a16/submission.py. Add a module constant MARLIN_BLOCK_SIZE_M_CAP = 16 beside
  BLOCK_SIZE_M_CANDIDATES, and immediately after SGLang unchanged selection loop (first candidate
  in (8, 16, 32, 48, 64) with num_tokens*8/bank["count"]/block_size_m < 0.9, loop exhausting at
  64) add the single line block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP). Master picks,
  verified from the rule with bank count 32: bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48,
  bs=256/257 -> 64 (not 32 at bs=128). After the cap: bs=128 48 -> 16 (thread_m_blocks 3 -> 1),
  bs=256/257 64 -> 16 (4 -> 1); every bs<=33 is unchanged (8 and 16 are already <= 16) and is the
  in-run control. bs=32 is deliberately left at 16 rather than 8: 8 and 16 both run
  thread_m_blocks=1 (8 only switches to the m_block_size_8 special path), the only evidence on
  that step is next-larger bs=1/8/16 rows at 2-3% (inside noise once extra m-blocks for 9-16-route
  experts are paid), decode_layer kept 16 at T=32 when it gained, and
  marlin-block-size-m-next-smaller already tests bs=32 16 -> 8 from master. The capped value feeds
  moe_align_block_size (still over the bank 32 experts, and moe_align_single_token at bs=1 which
  is unaffected) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2), as
  it must. Nothing else changes: no input-dependent branch, no sync, the branch is on num_tokens
  only. Mechanism (as measured, not as decode first argued): per-m-block Marlin cost grows with
  thread_m_blocks (next-larger here: 48 -> 64 at bs=128 took Marlin 495 -> 662 us; 16 -> 32 at
  bs=32 lost 21%), while SGLang T*8/32 rule overcounts routes per local expert 8x under EP8 (true
  ~4-8 routes/expert at bs=128, ~8-16 at bs=256 on the busiest GPU), so block 16 adds an extra
  m-block (another weight pass) only for a few hot experts.
expected:
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 16 (thread_m_blocks 4 -> 1), the point neither in-flight sibling tests; decode_layer gained 1.08-1.13x on its T=256 shapes with this exact change; Marlin is 612 of 621 us here at 5.7x SoL
  - shapes: bs=128 *
    direction: faster
    why: 48 -> 16 (3 -> 1); decode_layer gained 1.10-1.15x at T=128 (Marlin ~425 -> 293 us); next-larger showed the opposite step costing 14% here; Marlin 96% of 462 us
  - shapes: bs=32 *
    direction: flat
    why: "control: keeps 16 (bs=32 16 -> 8 is next-smaller own arm)"
  - shapes: bs=16 *
    direction: flat
    why: "control: keeps 8"
  - shapes: bs=8 *
    direction: flat
    why: "control: keeps 8"
  - shapes: bs=1 *
    direction: flat
    why: "control: keeps 8"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T103919Z
review_companions: null
---

# marlin-block-m-cap-16-port

## Change

Port muse_decode_layer marlin-block-m-cap-16 (landed gain 1.062x, job 0730135cdcac43b7; patch in muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch) to kernel_fn in muse_moe_w4a16/submission.py. Add a module constant MARLIN_BLOCK_SIZE_M_CAP = 16 beside BLOCK_SIZE_M_CANDIDATES, and immediately after SGLang unchanged selection loop (first candidate in (8, 16, 32, 48, 64) with num_tokens*8/bank["count"]/block_size_m < 0.9, loop exhausting at 64) add the single line block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP). Master picks, verified from the rule with bank count 32: bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48, bs=256/257 -> 64 (not 32 at bs=128). After the cap: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 16 (4 -> 1); every bs<=33 is unchanged (8 and 16 are already <= 16) and is the in-run control. bs=32 is deliberately left at 16 rather than 8: 8 and 16 both run thread_m_blocks=1 (8 only switches to the m_block_size_8 special path), the only evidence on that step is next-larger bs=1/8/16 rows at 2-3% (inside noise once extra m-blocks for 9-16-route experts are paid), decode_layer kept 16 at T=32 when it gained, and marlin-block-size-m-next-smaller already tests bs=32 16 -> 8 from master. The capped value feeds moe_align_block_size (still over the bank 32 experts, and moe_align_single_token at bs=1 which is unaffected) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2), as it must. Nothing else changes: no input-dependent branch, no sync, the branch is on num_tokens only. Mechanism (as measured, not as decode first argued): per-m-block Marlin cost grows with thread_m_blocks (next-larger here: 48 -> 64 at bs=128 took Marlin 495 -> 662 us; 16 -> 32 at bs=32 lost 21%), while SGLang T*8/32 rule overcounts routes per local expert 8x under EP8 (true ~4-8 routes/expert at bs=128, ~8-16 at bs=256 on the busiest GPU), so block 16 adds an extra m-block (another weight pass) only for a few hot experts.

## Expected

- `bs=256 *` faster: 64 -> 16 (thread_m_blocks 4 -> 1), the point neither in-flight sibling tests; decode_layer gained 1.08-1.13x on its T=256 shapes with this exact change; Marlin is 612 of 621 us here at 5.7x SoL
- `bs=128 *` faster: 48 -> 16 (3 -> 1); decode_layer gained 1.10-1.15x at T=128 (Marlin ~425 -> 293 us); next-larger showed the opposite step costing 14% here; Marlin 96% of 462 us
- `bs=32 *` flat: control: keeps 16 (bs=32 16 -> 8 is next-smaller own arm)
- `bs=16 *` flat: control: keeps 8
- `bs=8 *` flat: control: keeps 8
- `bs=1 *` flat: control: keeps 8

## Risk

Risk: the decode_layer evidence was measured with 256-expert align and dist=soft routing; here the align is over 32 experts and routing is skewed so this GPU is the busiest, so hot experts may hold more than 16 local routes at bs=256 and pay 2+ m-blocks (each a full pass over that expert 16 MB w13 / 8 MB w2 slice, partly from L2). If bs=256 is flat or loses while bs=128 gains, read it as hot-expert re-streaming and prefer the verify-two-smaller pick (32) at bs=256. Correctness unaffected (same fp32 reduce, same math); the bs=257 and skewed bs=128 tests run the new picks. Sweep: with marlin-block-size-m-next-smaller (one step down: bs=128 -> 32, bs=256 -> 48, bs=32 -> 8) and marlin-block-size-m-verify-two-smaller (two steps down: bs=128 -> 16, bs=256 -> 32), this run (cap 16: bs=256 -> 16) completes a 3-point sweep at bs=256 (48/32/16) against SGLang 64. At bs=128 this pick (16) coincides with verify-two-smaller; that overlap is intended for the direct port but means only one of the two needs to spend a check on bs=128, and the bs=128 rows of whichever runs second are a reproducibility read, not new information. All three edit the same line from master, so at most one lands; the others become stacked re-runs or are closed by this result. Run this one first if checks are limited: it is the replicated design.

## Adversarial review

Reviewed 2026-10-24T103919Z by an adversarial subagent (model opus).

**Verdict: proceed.** A rule-clean one-line cap on a host int derived from num_tokens. It is backed by the landed decode_layer cap-16 gain and by this problem's next-larger regression. It adds the one new point neither in-flight sibling covers (bs=256 at 16), and each moving shape is its own scored load, so one run reads per shape with no companions.

### Justification
I checked the picks against master's kernel_fn. The rule is num_tokens*8/32/bsm < 0.9, and the loop exhausts at 64. That gives bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48 (32/48 = 0.67) and bs=256/257 -> 64 (64/64 = 1.0 is not < 0.9, so the loop exhausts). After min(.,16), bs=128 goes 48->16 and bs=256/257 go 64->16, and every bs<=33 is unchanged, as the hypothesis states. Profile of the parity best (fb2cc736613747df): Marlin is 495 us (96%) of bs=128 against a 101.3 us floor, and 612 us (96%) of bs=256 against 108.9 us, so both run at about 5-6x SoL. That is the per-m-block compute/latency regime where thread_m_blocks dominates, not an HBM-bound one. Evidence for the direction: (1) next-larger here took bs=128 from 48 to 64 and Marlin went 495 -> 662 us, a 14% shape loss; at bs=32 16->32 lost 21%; (2) decode_layer marlin-block-m-cap-16 (0730135cdcac43b7) made this same min() change and gained 1.10-1.15x at T=128 and 1.08-1.13x at T=256. Each of bs=128 and bs=256 is 1/6 of the geomean, so a 10%+ Marlin change is well above the ~1.5-2.5% drift. Expected directions match: bs=128/256 faster, bs<=32 flat as in-run controls. Weakest link: bs=256 under this problem's skewed busiest-GPU routing. About 2048 routes spread over 8 GPUs gives roughly 256-400 local routes over 32 experts, so the mean is about 8-12 per expert, but the hot experts may hold 30-60. With block 16, each hot expert pays 2-4 m-blocks, each another pass over its w13/w2 slice. decode_layer's T=256 gain came under different routing (dist=soft, 256-expert align), so its magnitude at bs=256 transfers only loosely. bs=128 is on firmer ground because two reasons point the same way: the m-block counts there are smaller, and the next-larger gradient was measured on this problem.

### Confounders on the current kernel
- sglang-marlin-path-parity (landed; align over the bank's 32 experts instead of 256): sorted_ids capacity is M*8 + 32*(bsm-1), so a smaller block slightly shrinks the align padding work (<~1 us against 500-600 us of Marlin). That cannot flip a sign. It does mean decode_layer's 256-expert-align magnitude transfers only approximately. Reverting it would restore the 0.006x dense starter, so there is no meaningful counterfactual and no ab-revert.

### Interference and A/B needs
No companions. The only diff is one host-side int, and the shapes it moves (bs=128, bs=256) are separate scored loads, each paired with SGLang in the same run. bs=1/8/16/32 keep their pick and act as in-run drift controls. A gain or a loss at either shape reads directly as 'cap 16 beats / loses to SGLang's pick at that shape', so neither sign is ambiguous. The sweep the hypothesis wants is already covered by the in-flight siblings from the same master: next-smaller gives bs=128 at 32 and bs=256 at 48, and verify-two-smaller gives bs=128 at 16 and bs=256 at 32. This run adds bs=256 at 16, which completes 48/32/16 at bs=256. Its bs=128 row repeats verify-two-smaller's exact pick (16), so it costs nothing extra and serves as a reproducibility read for that sibling. An ab-tune of the cap (e.g. 32) would duplicate verify-two-smaller at bs=256, and a control isolating w13 vs w2 cannot change the adopt/reject decision for a single shared block size. One base caveat: if a sibling lands first and master's pick code changes (e.g. to an idx-step table), claim on the new master and put the min() after whatever the final pick is. On top of verify-two-smaller, this run would then move only bs=256 (32 -> 16), which is still readable.

### Open questions
- At bs=256, compare against verify-two-smaller (32) and next-smaller (48). If 16 loses to 32 there while bs=128 at 16 gains, read it as hot-expert m-block re-streaming under skew. In that case land a per-shape pick (16 at bs<=128, 32 at bs>=256) rather than a uniform cap.
- bs=257 and the skewed bs=128 correctness tests run block 16. Marlin supports moe_block_size 16 at any M, and the math and fp32 reduce are unchanged, so correctness risk is minimal. Still, confirm that the run's test list passes before reading timing.
- Since this run and verify-two-smaller share the bs=128 pick exactly, if both harvest as gains at bs=128 the promotion should take only one of them. They edit the same line, so a by-hand pick is needed: prefer whichever bs=256 point wins.
- If bs=128 gains here, a natural follow-up is whether the w13 and w2 Marlin calls want different block sizes (w2 has size_m = M*8 rows of K=size_n). That is out of scope for this run.

