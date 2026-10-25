---
problem: muse_attention_decode
slug: restack-sk-pseudo-b1-cluster-c16-c4
created: 2026-10-25T054448Z
base: master
area: "sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4)"
change: >
  Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at
  B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A)
  From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job
  94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as
  is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup
  gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed
  only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at
  B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route.
  (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job
  195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU
  (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode
  (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on
  master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's
  stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS
  2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit
  (B) for module/static globals holding device pointers or state-sized buffers and move any into
  per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state
  refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills;
  extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no
  host sync, no allocation sized by input values (shape/config only), no data-dependent Python
  control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from
  inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window
  attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them:
  no module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0
  by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare
  against an fp32 reference with two states and alternating graph replays with refilled inputs
  (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to
  master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed
  TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "s1 pseudo-decode route: 17.78 -> 13.44 us measured (kernel 9.8 us replaces FA3 7.9 + combine 3.9 + prepare 1.8 + norm-gate)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "cluster C=4: 24.33 -> 20.05 us measured (kernel 18.0 vs Triton 22.4)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "cluster C=16: 13.15 -> 12.57 us measured (-4.4%; near drift, read the kernel row)"
  - shapes: "*layer=full *"
    direction: flat
    why: "full paths are master's byte-identical code (PSEUDO=false layout check prints IDENTICAL)"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: B=2..16 stays on master Triton (C=8 gated off)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T054818Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T054819Z
base_sha: edf4bf29fe8d509a2574c0becee65138cb7705f2
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/restack-sk-pseudo-b1-cluster-c16-c4
worktree: .worktrees/muse_attention_decode/restack-sk-pseudo-b1-cluster-c16-c4
mode: null
restack_of: null
stale_base: null
---

# restack-sk-pseudo-b1-cluster-c16-c4

## Change

Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A) From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job 94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route. (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job 195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS 2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit (B) for module/static globals holding device pointers or state-sized buffers and move any into per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills; extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=verify_1 *` faster: s1 pseudo-decode route: 17.78 -> 13.44 us measured (kernel 9.8 us replaces FA3 7.9 + combine 3.9 + prepare 1.8 + norm-gate)
- `*layer=sliding load=decode_32 *` faster: cluster C=4: 24.33 -> 20.05 us measured (kernel 18.0 vs Triton 22.4)
- `*layer=sliding load=decode_1 *` faster: cluster C=16: 13.15 -> 12.57 us measured (-4.4%; near drift, read the kernel row)
- `*layer=full *` flat: full paths are master's byte-identical code (PSEUDO=false layout check prints IDENTICAL)
- `*layer=sliding load=decode_16 *` flat: B=2..16 stays on master Triton (C=8 gated off)

## Risk

Landing candidate. Substituting the two cluster rows into s1's measured loads gives ~1.489x (s1 alone 1.4705; master re-measured 1.432-1.457 across three master-equivalent checks today: 2a3241780c7b4c46, f519261b7c2b435f and c0f1945a108c4aa2 all fell back to master on every row). Do not add sk-pdl-from-preattn: its gate chose plain launch (PDL +0.54 / +3.84 us slower), so that run measured master; its full decode_32 199 us row is below its own 229.6 us kernel row, i.e. a timing artefact, not a gain. Risk: the cluster TU was written before the two-state lesson; the audit above is mandatory.

## Adversarial review

Reviewed 2026-10-25T054818Z by an adversarial subagent (model opus).

**Verdict: proceed.** Clean restack of two separately measured sliding-row wins on disjoint shapes. (A) applies to master's own blob d8769ef and (B) is gated away from every shape (A) touches, so each row is attributable from the profile and the PATH CHOSEN lines. Before the build: strip (B)'s kernel_fn try/except, prints and mutable set, and state the C=16 margin explicitly so the B=1 band is not silently gated off.

### Justification
A: The mechanism is plausible and was measured. Sliding verify_1 on master is FA3 main 7.9 + combine 3.9 + prepare_varlen 1.8 us + norm-gate (17.8 us shape). The s1 pseudo-decode kernel (grid p8, ticket) replaced that chain at 13.44 us in job 94523c33 on the exact master blob (d8769ef = BEST.submission_blob), so (A) is a re-run of measured code. Sliding decode_32 on master is _sliding_fused_kernel at 22.4 us (24.3 us shape, floor 3.3). Cluster C=4 measured 20.05 us in job 195dcc12; the Triton kernel it replaces is the same deep-splits kernel that is on master now. Each of the three rows has weight 46, and they sit in three different loads (decode 1, decode 32, verify 1), so the ~1.49x projection from substituting rows is arithmetically sound. The expected directions match the gates: full rows flat (byte-identical PSEUDO=false instantiations), sliding decode_16 flat (B=2..16 stays on Triton). Weakest link: the sliding decode_1 C=16 row. Its gain is 0.58 us (13.15 -> 12.57, 4.4%), and master's own sliding decode_1 has read 13.1-13.33 us across today's runs, so one run cannot confirm it. Worse, the hypothesis says gates route 'only when the printed interleaved graph-replay timing shows the new path winning by the stated margin' but never states (B)'s margin. If the builder reuses s1's >=1.05x, C=16 (about 1.046x) will usually fall back to Triton and the run silently measures master on that row. Second weakest: everything sits far above the 0.1-3.3 us floors, so there is headroom; the risk lies in the port, not the physics.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: the cluster C=16/C=4 wins were measured against this exact Triton _sliding_fused_kernel, which is unchanged on master, so the comparison carries over. No confound beyond drift on the near-noise decode_1 row.
- sliding-fused-triton-decode1-gate: the incumbent at sliding decode B=1 that C=16 replaces. Same code as in the stack run, so no interaction; the C=16 row stays near noise.
- full-decode1-streamk-s3-two-level-ticket: (A) adds a PSEUDO template flag and Lay<STAGES,256,true> to the same splitkv_attn_sk_kernel and builds with -DSK_PSEUDO=1. Full decode rows stay flat only if the landed instantiations are byte-identical (the sk_attrs IDENTICAL print must appear) and the landed-only rebuild fallback did not fire. A full-row move without IDENTICAL means the build changed, not the sliding work.
- fused-preattention-triton: (A) keeps _pre_attention_kernel ahead of the pseudo kernel at verify B=1. (B) folds norm/RoPE/FP8 store into the cluster kernel at decode B=1 and B=17..32, with a K-byte encode that only matches master within rel 5e-2 (byte match is info only). That is inside the 8% cache bound, but a row failure on a new-cache test would come from this encode, not the attention.

### Interference and A/B needs
No companion is warranted. The two sub-changes route disjoint shapes, (A) to sliding verify B=1 and (B) to sliding decode B=1 and B=17..32, and each run is a separate graph replay. Each row's gain or loss therefore belongs to exactly one sub-change, readable from the profile's top kernel (splitkv_attn_sk_kernel<...PSEUDO> vs sliding_cluster_kernel) and the PATH CHOSEN lines. An ab-revert of a landed hypothesis would not change that: neither route depends on a landed tuning other than the incumbent each replaces, and those incumbents are unchanged since the source measurements. The only ambiguous row is sliding decode_1, and the in-run interleaved timing that the gate prints (cluster C=16 vs Triton, cold L2, empty-graph subtracted) already serves as its control at no check cost. The A/B partner restack-sk-pseudo-cluster-plus-v3-fused-preattn is already minted and covers the full decode_16/32 question, so no duplicate is queued. With 8 checks left and 8 queued, spending one on a control would be waste.

### Open questions
- What margin gates (B)'s bands? Recommend keeping (B) config-only as it ran in 195dcc12 (C=16 at B=1, C=4 at B=17..32) and only printing the interleaved timing. Failing that, set C=16's margin at <=1.02x, since 1.05x would drop the B=1 band at its measured 1.046x.
- Compile budget: splitkv 49 s + sk 37 s + slide (fp8qk) 41 s + the -DSK_PSEUDO=1 rebuild, plus a possible landed-only rebuild fallback and an f16qk fallback build, plus all the gating timings. That is likely 3-5 min of the 15-min run. Print per-phase wall times. Is the torch extension cache warm on the judge?
- Does the s1 IDENTICAL check (sk_attrs for landed variants) still print IDENTICAL when (B)'s TU is also loaded in the same process? Register pressure and smem attributes should not change, but the line must be in the log to read the full rows as flat.
- The (B) self-check's cache tolerance is rel < 5e-2 with byte match as info only. Does the new-cache correctness bound (8% of the reference's) still pass on the kv_scale 0.015625 page-1 test? That test routes to master today (q_len 8 or page 64), so check that no cluster-gated test case has kv_scale != 1.

