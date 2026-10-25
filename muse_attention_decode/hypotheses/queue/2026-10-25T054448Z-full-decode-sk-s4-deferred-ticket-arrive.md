---
problem: muse_attention_decode
slug: full-decode-sk-s4-deferred-ticket-arrive
created: 2026-10-25T054448Z
base: master
area: full-layer attention (decode 1<B<=64, stream-K ticket epilogue)
change: >
  Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs
  106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.09 TB/s), and 221.5 vs 218.2
  us at B=32 ([sk-diag] in c0f1945a108c4aa2 and f519261b7c2b435f logs): the in-kernel combine
  costs ~6 us at B=16 against the no-ticket kernel, before the separate combine it saves. Where:
  at every segment end inside the tile loop the CTA does a __threadfence + 2 __syncthreads +
  atomicAdd round trip (and the last arrival a serial merge_slots of ~16 slots) while its next
  segment's tiles wait. Change (new template flag DEFER on the single-level ticket path, B>1 only;
  DEFER=false byte-identical): inside the loop a finished non-whole segment only stores its
  partial (ws_acc/ws_ml slot bx + c_b) and records c_b in a small smem list (<= 4 entries; at grid
  264 a CTA holds <= 3 segments for B<=64, assert it); the loop continues streaming with no fence
  or atomic. After the last tile, ONE __threadfence + __syncthreads, then for each recorded
  request draw the ticket (atomicAdd == c_hi - c_lo) and the last arrival merges + finalize_store
  + resets the ticket, exactly as today. Whole segments still finalize in place. Correctness
  argument: the ticket draw still follows the release fence of every partial it counts. Variant
  matrix in prepare: DEFER on/off x grid 2xSM/SM at B=16, B=32, B=64; route DEFER only if it beats
  master's variant 0 by >= 1.02x in interleaved graph replay; B=1 two-level unchanged. RULES
  (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input
  values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes. The judge
  prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (master's
  self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data
  and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference
  with two states and alternating graph replays with refilled inputs (capture once, refill A,
  replay, refill B, replay, compare each), and on any miss falls back to master's path for that
  band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: ticket kernel 110.1 us in profile vs the no-ticket kernel ~6 us faster in prepare; target kernel <= 106 us (floor 63.2)
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: 225.6 us kernel; no-ticket gap ~3 us; target <= 222 us
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: two-level B=1 path untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: medium
depends_on: full-decode1-streamk-s3-two-level-ticket
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue"
review: proceed
status: queued
review_at: 2026-10-25T054724Z
review_companions: null
---

# full-decode-sk-s4-deferred-ticket-arrive

## Change

Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs 106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.09 TB/s), and 221.5 vs 218.2 us at B=32 ([sk-diag] in c0f1945a108c4aa2 and f519261b7c2b435f logs): the in-kernel combine costs ~6 us at B=16 against the no-ticket kernel, before the separate combine it saves. Where: at every segment end inside the tile loop the CTA does a __threadfence + 2 __syncthreads + atomicAdd round trip (and the last arrival a serial merge_slots of ~16 slots) while its next segment's tiles wait. Change (new template flag DEFER on the single-level ticket path, B>1 only; DEFER=false byte-identical): inside the loop a finished non-whole segment only stores its partial (ws_acc/ws_ml slot bx + c_b) and records c_b in a small smem list (<= 4 entries; at grid 264 a CTA holds <= 3 segments for B<=64, assert it); the loop continues streaming with no fence or atomic. After the last tile, ONE __threadfence + __syncthreads, then for each recorded request draw the ticket (atomicAdd == c_hi - c_lo) and the last arrival merges + finalize_store + resets the ticket, exactly as today. Whole segments still finalize in place. Correctness argument: the ticket draw still follows the release fence of every partial it counts. Variant matrix in prepare: DEFER on/off x grid 2xSM/SM at B=16, B=32, B=64; route DEFER only if it beats master's variant 0 by >= 1.02x in interleaved graph replay; B=1 two-level unchanged. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_16 *` faster: ticket kernel 110.1 us in profile vs the no-ticket kernel ~6 us faster in prepare; target kernel <= 106 us (floor 63.2)
- `*layer=full load=decode_32 *` faster: 225.6 us kernel; no-ticket gap ~3 us; target <= 222 us
- `*layer=full load=decode_1 *` flat: two-level B=1 path untouched
- `*layer=sliding *` flat: untouched

## Risk

Small (~0.3-0.5% headline) but attributable on two weight-16 rows via the kernel row. Composes with v3 fused-preattn later (both are template flags on the same kernel). Risk: deferring arrivals lengthens the time the last request waits for its earliest segments' owners; the gate decides.

## Milestones

Rewrite stage 4 of `full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054724Z by an adversarial subagent (model opus).

**Verdict: proceed.** Safe, self-gated rewrite stage with no rules problems. But the mechanism is probably misattributed: the ~6 us ticket-vs-no-ticket gap at B=16 most likely comes from the end-of-kernel serial merge tail, which DEFER keeps and may make worse, not from the in-loop round trip. Expect a neutral result unless prepare's variant matrix shows otherwise. No companions.

### Justification
The target is real. Full decode_16 is 111.1 us (kernel 110.1 us, floor 63.2 us) and decode_32 is 226.4 us (kernel 225.6 us, floor 126.4 us). Both are weight-16 rows. The [sk-diag] gap of ~6 us at B=16 and ~3 us at B=32 between <4,ticket> and <4,no ticket> is cited from logs. The 'expected' directions agree with TASK.md: faster on full decode_16/32, flat elsewhere, and B=1 TWO stays untouched. Now the structure. In master's single-level path (submission.py ~1338-1402), a CTA's segments are its head segment (the tail of request b, from c_hi's view), any number of whole requests (finalized in place, no ticket), and its tail segment (the start of request b+1). So each CTA makes at most ONE in-loop non-whole arrival: the head segment, which ends mid-loop. The tail segment already ends at n == N-1. All CTAs start together with near-equal tile counts. For a request spanning c_lo..c_hi, c_lo and the interior CTAs finish at loop end, while c_hi's head segment ends early. So c_hi's mid-loop arrival is essentially never the last one: mid-loop work is one __threadfence + 2 __syncthreads + atomicAdd, roughly sub-us, and merge_slots never runs mid-loop in practice. The 'serial merge_slots of ~16 slots while the next segment's tiles wait' that the hypothesis describes therefore does not happen on the critical path. The merge runs at the END of the last-finishing CTA, where it forms the kernel tail: merge_range walks ~16 slots serially through L2, then finalize_store. DEFER does not remove that tail. It can lengthen it: c_hi now arrives at the end too, so it becomes a candidate last arriver for BOTH b (head) and b+1 (tail) and could run two serial merges back-to-back. WEAKEST LINK: the 6 us gap is attributed to in-loop fences with no measurement that separates the fence/atomic cost from the end-merge tail. Realistic gain if the hypothesis is right: about 0.5-1 us per CTA's single mid-loop fence, below the 1.02x (~2.2 us at B=16) routing margin. So the most likely outcome is PATH CHOSEN = master, i.e. flat. That is cheap and still informative, but only if prepare also prints the numbers. Design nit that can break correctness/routing: the claim 'a CTA holds <= 3 segments for B<=64, assert it' holds only for the scored ~80K-context shapes. The landed self-check cases (B=64 mixed at grid 7, B=2 [20000,2] at grid 7, B=32 ragged at grid 3) and test shapes such as batch 4 spread mixed page 1 give one CTA many segments. The smem list only ever needs the NON-WHOLE segments, which are <= 1 deferred by construction (<= 2 counting the tail). Assert or size on that count, not on total segments, or the self-check falls back or traps on cases that are fine.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (master, c3c1b01): shares the splitkv_attn_sk_kernel template. DEFER is only on the single-level TICKET && !TWO branch, so the B=1 two-level path must stay byte-identical. Any change to the shared prologue or template instantiation would move full decode_1 and contaminate the read. Check that the kernel row for full decode_1 stays at ~18.6 us.
- full-decode-cuda-gather-splitkv (stage-1 A+B, landed): only the [sk-diag] baseline uses it now, not the timed path. It does not confound the kernel-row read, but its A+B number is a useful sanity anchor in the prepare log.
- fused-preattention-triton / paired-heads-launch-cleanup: _pre_attention_kernel (2.0-2.2 us) runs before the sk kernel on the same rows and stays unchanged. The headline shape time includes it, so read the sk kernel's own row, not the shape total.
- Headline noise: recent full-decode stages repeatedly showed 'lost/gained on sliding verify_1/decode_1' with no code change there. A sliding-row move on this run is noise, not interaction.

### Interference and A/B needs
No companion is warranted. This is a kind: rewrite stage judged on the sk kernel's own top_kernels row, and the hypothesis already carries an in-prepare A/B: DEFER on/off x grid 2xSM/SM at B=16/32/64, interleaved graph replay, with both numbers printed in PATH CHOSEN. That A/B is the disambiguator, and it costs no check. A loss cannot reach the headline because the gate falls back to master's variant 0. A gain is attributable on the kernel row. Attention has 8 checks for 8 queued hypotheses, so spending one on a companion is not justified. Better than a companion, at zero check cost: add one extra [sk-diag] row timing <4,ticket> with the merge+finalize skipped by the last arriver (diag only, never routed). The prepare log then splits the 6 us gap into in-loop fence cost versus end-merge tail, which tells the next stage whether to attack the tail (parallel or wider merge_range loads, or distributing the final merge across the arriving CTAs) instead of the fences.

### Open questions
- Is the [sk-diag] 'no ticket' number kernel-only, and was it measured on the same cold-L2 interleaved replay as the ticket variant? If not, part of the 6 us may be measurement asymmetry.
- How long does merge_slots + finalize_store take for ~16 slots, measured as the tail of the last CTA? That is the likely real cost, and a stage that parallelizes it may be the better stage 4.
- Under DEFER, how often does one CTA become last arriver for two requests (head and tail) and run two serial merges at the end? Print the count in the self-check if cheap.

