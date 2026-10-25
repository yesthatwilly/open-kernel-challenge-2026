---
problem: muse_attention_decode
slug: sliding-decode-sk-window-route
created: 2026-10-25T054449Z
base: master
area: sliding-layer attention (decode, stream-K windowed kernel)
change: >
  On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is
  already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over
  rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b],
  token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK
  = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB
  at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6
  us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel
  (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is
  _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes;
  the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for
  the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of
  [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >=
  1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv
  0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The
  judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
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
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: 15.33 us shape, Triton 12.7 us kernel; 8.4 MB at ~3 TB/s is ~3 us + pre-attn 2.0 us; neither cluster C=8 (13.8) nor paired rows (16.8) beat Triton here; M2 shape <= 12 us
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: 24.33 us (Triton 22.4; cluster C=4 18.0 off-master); M2 shape <= 18 us
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: 32 tiles only; expected to lose to Triton 10.2 / cluster C=16 10.6, gate keeps master
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: medium
depends_on: sliding-verify-sk-pseudo-decode-s1
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)"
review: proceed
status: queued
review_at: 2026-10-25T054724Z
review_companions: null
---

# sliding-decode-sk-window-route

## Change

On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b], token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6 us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes; the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >= 1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv 0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=decode_16 *` faster: 15.33 us shape, Triton 12.7 us kernel; 8.4 MB at ~3 TB/s is ~3 us + pre-attn 2.0 us; neither cluster C=8 (13.8) nor paired rows (16.8) beat Triton here; M2 shape <= 12 us
- `*layer=sliding load=decode_32 *` faster: 24.33 us (Triton 22.4; cluster C=4 18.0 off-master); M2 shape <= 18 us
- `*layer=sliding load=decode_1 *` flat: 32 tiles only; expected to lose to Triton 10.2 / cluster C=16 10.6, gate keeps master
- `*layer=full *` flat: untouched

## Risk

Kill: if B=16 [pre-attn + sk] is not >= 1.05x faster than Triton in the printed gate, sliding decode stays Triton/cluster. If B=17..32 wins, compare its kernel row with the cluster C=4 row (18.0 us) before a restack picks one. Weight 46 rows: worth ~+1-2% headline per band if it reaches M2.

## Milestones

Rewrite stage 2 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054724Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design is sound and every band is either routed by printed paired timings or left as byte-identical Triton, so each row can be attributed without a companion. The ~3 TB/s, 3-6 us kernel estimate is optimistic: on this kernel, small workloads run latency-bound at about 0.9-1.1 TB/s. Expect B=32 to win clearly, while B=16 is a coin flip against the 1.05x gate.

### Justification
Mechanism: sliding decode at TOK=1 has no L2 read amplification, so it differs from s1's B=16/32 verify bands, which lost 0.65x/0.48x because of 67/134 MB of L2 traffic. That makes the work 512 tiles / 8.4 MB at B=16 and 1024 tiles / 16.8 MB at B=32. The incumbent Triton _sliding_fused_kernel is far from the floor: 12.7 us for 8.4 MB (0.66 TB/s, floor 1.6 us) and 22.4 us for 16.8 MB (0.75 TB/s, floor 3.3 us). These rows weigh 46 each in their loads, so the target is worth pursuing. WEAKEST LINK: the '~3 TB/s, ~3-6 us' figure. This kernel reaches 2.9-3.0 TB/s only on 330-660 MB full-decode jobs (sk-diag). On small jobs it is latency- and merge-bound: full decode_1 moves 20 MB in 18.6 us (0.89-1.1 TB/s), and s1's verify_1 (256 tiles, 0.5 MB DRAM) costs 9.8 us of kernel. A realistic B=16 estimate is a gather round-trip plus a ~16-32-slot ticket merge, about 6-9 us, plus a separate pre-attention node of 2.0 us and its graph-node boundary. That puts the pair at about 9-11.5 us against 12.7 us, which is near the 1.05x gate. B=32 (about 9-12 us of kernel plus 2.2 us of pre-attention, against 22.4 us) has real margin, and the off-master cluster C=4 row (18.0 us) gives a cross-check. Expected directions are consistent with TASK weights and floors: decode_1 flat because it is gate-kept, full layers flat because the code is untouched. The M2 targets (12 us / 18 us shape) are reachable at B=32 and marginal at B=16.

### Confounders on the current kernel
- sliding-verify-sk-pseudo-decode-s1 (base branch, NOT on master): the build starts from this branch, so the headline carries s1's sliding verify_1 route (1.4705x check vs master 1.4441x). Judge against s1's job 94523c3313444fd8 rows, not BEST.json. The TOK template must leave the TOK=8 instantiation byte-identical (print regs/local/smem/CTAs for it next to s1's). Otherwise a verify_1 shift gets mixed into this run.
- full-decode1-streamk-s3-two-level-ticket / full-decode-splitkv-s2 (landed, c3c1b01): this is the kernel being templated. The PSEUDO=false instantiations serving full decode must stay identical. If the full decode_1/16/32 rows move outside noise, that comes from a changed instantiation, not from this route.
- sliding-deep-splits-on-cuda-splitkv (landed): the comparator is its 16-split Triton _sliding_fused_kernel, which does norm, RoPE and the cache store in-kernel. The new route pays a separate _pre_attention_kernel node (about 2.0 us at B=16/32) that the incumbent folds in. That is why the gate must time the pair [pre-attn + sk] against Triton alone in one graph, as specified.
- fused-preattention-triton (landed): it now runs on sliding DECODE in its PAIRED=False, ROPE-on configuration, which until now was used only at verify. It writes the new K/V row that the sk window [sp-2048, sp) must include. Correctness depends on it storing the row before the sk kernel reads it, which stream order guarantees.

### Interference and A/B needs
No companion. This is a rewrite stage with a per-band hard gate: each sliding decode band runs either the sk pair, named in PATH CHOSEN together with both printed numbers, or byte-identical master Triton. A gain or loss on a row is therefore readable from the log and the profile's top_kernels without an A/B run. The one headline confound is s1's verify_1 route, and comparing against s1's own job resolves it. A revert or tune run would add nothing that the printed per-variant gate matrix does not already show. Attention has 8 checks for 8 queued items, which supports spending none on companions.

### Open questions
- BUILDER: subtract the empty-graph baseline as stated. s1's gate numbers evidently included about 5 us of fixed graph cost (gate 15.4 us vs profile 9.8 us at verify_1), which compresses ratios toward 1 and can wrongly reject a real 1.05x win at B=16.
- BUILDER: time the pre-attn + sk pair as two nodes in ONE captured graph (not two graphs summed), so the inter-node gap that Triton's single launch avoids is charged.
- BUILDER: at B=16 add a grid of 2R-4R (32-64 CTAs, 8-16 tiles each) so each request's merge is about 2-4 slots. s1's B=1 data showed that small merges won (p8 15.4 us vs 2xSM 19.5 us).
- If B=17..32 routes, print its kernel time next to the cluster C=4 figure (18.0 us, off-master) so the restack picks one. If B=2..16 falls back, the next stage should fold pre-attention into the sk prologue (as full-decode-splitkv-s3-fused-preattn tried) rather than retune the grid.

