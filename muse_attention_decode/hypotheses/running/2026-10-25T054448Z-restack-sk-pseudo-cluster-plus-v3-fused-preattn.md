---
problem: muse_attention_decode
slug: restack-sk-pseudo-cluster-plus-v3-fused-preattn
created: 2026-10-25T054448Z
base: master
area: full-layer attention (decode 1<B<=64 prologue) on top of the sliding restack
change: >
  Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster
  C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the stage-3 fused pre-attention
  prologue from branch hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn (job
  825baa03798f431a): the v3 instantiation of splitkv_attn_sk_kernel that builds the FP8 q (RMSNorm
  + RoPE) and stores the new K/V row in-kernel, removing the _pre_attention_kernel launch, gated
  to full decode 1<B<=64 only. Full decode B=1 keeps master's two-level ticket with the separate
  pre-attention kernel (v3 has no two-level variant). Port v3 as a NEW template flag (FUSED_PRE,
  default false) on master's kernel so variants 0-3 and PSEUDO stay byte-identical (print sk_attrs
  IDENTICAL for the landed variants, as s1 does); v3 keeps its own self-check (cache rows
  bit-exact vs the pre-attention kernel, rel L2 vs fp32 reference) and its own bound constant. Add
  a gate: route v3 per band (B=2..16 from the B=16 timing, B=17..64 from B=32) only if the printed
  interleaved graph-replay time of [v3] beats [pre-attn + v2 ticket] by >= 1.02x (v3 measured
  115.1 vs 118.2 us at B=16 and 220.5 vs 227.7 at B=32 in its diag). This run is the A/B partner
  of restack-sk-pseudo-b1-cluster-c16-c4 dispatched in the same batch: the only difference is v3,
  so the full decode_16/32 rows pair directly. RULES (every path): CUDA-graph capturable;
  kernel_fn does no host sync, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state;
  nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key
  in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and
  rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA
  map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and
  returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route
  self-checks in prepare against an fp32 reference with two states and alternating graph replays
  with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on
  any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing
  case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary,
  sync+print after each self-check launch). Gates route a band only when the printed interleaved
  graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path
  winning by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "v3 measured 110.91 -> 109.55 us (-1.2%) with _pre_attention_kernel gone; must beat the partner run's row"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: v3 measured 226.81 -> 223.11 us (-1.6%); diag saved 7.2 us at B=32
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: same as partner (pseudo-decode B=1)
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: same as partner (cluster C=4)
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: B=1 keeps master two-level ticket
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T054723Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T054723Z
base_sha: 6f1f3a99d4b2849b4e653b5b3b88ff6f21aefaab
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/restack-sk-pseudo-cluster-plus-v3-fused-preattn
worktree: .worktrees/muse_attention_decode/restack-sk-pseudo-cluster-plus-v3-fused-preattn
mode: null
restack_of: null
stale_base: null
---

# restack-sk-pseudo-cluster-plus-v3-fused-preattn

## Change

Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the stage-3 fused pre-attention prologue from branch hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn (job 825baa03798f431a): the v3 instantiation of splitkv_attn_sk_kernel that builds the FP8 q (RMSNorm + RoPE) and stores the new K/V row in-kernel, removing the _pre_attention_kernel launch, gated to full decode 1<B<=64 only. Full decode B=1 keeps master's two-level ticket with the separate pre-attention kernel (v3 has no two-level variant). Port v3 as a NEW template flag (FUSED_PRE, default false) on master's kernel so variants 0-3 and PSEUDO stay byte-identical (print sk_attrs IDENTICAL for the landed variants, as s1 does); v3 keeps its own self-check (cache rows bit-exact vs the pre-attention kernel, rel L2 vs fp32 reference) and its own bound constant. Add a gate: route v3 per band (B=2..16 from the B=16 timing, B=17..64 from B=32) only if the printed interleaved graph-replay time of [v3] beats [pre-attn + v2 ticket] by >= 1.02x (v3 measured 115.1 vs 118.2 us at B=16 and 220.5 vs 227.7 at B=32 in its diag). This run is the A/B partner of restack-sk-pseudo-b1-cluster-c16-c4 dispatched in the same batch: the only difference is v3, so the full decode_16/32 rows pair directly. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_16 *` faster: v3 measured 110.91 -> 109.55 us (-1.2%) with _pre_attention_kernel gone; must beat the partner run's row
- `*layer=full load=decode_32 *` faster: v3 measured 226.81 -> 223.11 us (-1.6%); diag saved 7.2 us at B=32
- `*layer=sliding load=verify_1 *` faster: same as partner (pseudo-decode B=1)
- `*layer=sliding load=decode_32 *` faster: same as partner (cluster C=4)
- `*layer=full load=decode_1 *` flat: B=1 keeps master two-level ticket

## Risk

If both runs land, the partner is the cleaner submit unless this one's full decode_16/32 rows beat it by more than the 1.5% drift band on the kernel rows. Risk: three edits to one TU (PSEUDO + FUSED_PRE on the same kernel); keep each behind if constexpr and print attrs for every instantiation.

## Adversarial review

Reviewed 2026-10-25T054723Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound A/B arm: it adds only the measured v3 fused pre-attention (from job 825baa03798f431a) to the partner restack on full decode 1<B<=64, and it is rules-clean. The expected gain is about 1.4 us at decode_16 and 3.7 us at decode_32. That is at the cross-job drift band, so read the result from the in-process interleaved gate timing and the kernel rows, not from the headline or a row-vs-row comparison with the partner run.

### Justification
The mechanism is plausible. On master's profile, full decode_16/32 spend 2.0 and 2.2 us in _pre_attention_kernel next to splitkv_attn_sk_kernel<4,true> (110.1 and 225.6 us). Floors are 63.2 and 126.4 us, so the attention kernel, not pre-attn, is what dominates the distance to the floor. v3 already ran on the stage-2 kernel, which is master's variant 0 at 1<B<=64. It passed 16/16 tests, measured 110.91->109.55 and 226.81->223.11 us, and its in-process diag showed 118.2->115.1 (1.027x) and 227.7->220.5 (1.033x). The direction (faster on full decode_16/32, flat everywhere else beyond the partner's own moves) matches TASK.md: weight 16 on both, full decode_1 unchanged on the two-level path. Headline impact is about +0.9% on load decode_16 and +1.25% on load decode_32, roughly 0.35% geomean, which is far inside the ±1.5% CI. The s3 run labelled its own 1.75% and 2.4% row moves 'flat'. WEAKEST LINK: one okc run cannot show that v3 beats the partner on the full rows, because the expected per-row delta (1.2-1.6%) equals the cross-job noise band. The only clean attribution is the gate's interleaved graph-replay pair ([pre-attn + v2] vs [v3], same process, same GPU, cold L2), printed in PATH CHOSEN. Second weakest: the 1.02x gate margin sits close to v3's measured 1.027x at B=16, so prepare-time noise may gate the B=2..16 band off. That is acceptable, but then that band of this run just replicates the partner. Third: the port is a re-implementation, not the measured code. s3 built v3 as a separate kernel (splitkv_attn_sk_fused_kernel) in its own module, muse_full_decode_splitkv_v3. This spec instead folds it in as a FUSED_PRE template flag on master's splitkv_attn_sk_kernel, which also carries the partner's PSEUDO flag and the TWO flag. The kernel already uses 121 of 128 registers at __launch_bounds__(THREADS,2), and s3's audit had to restructure the q build into two smem passes to avoid spills. If the template port changes register allocation or inlining, the measured 3-7 us can disappear. Print regs/local for the FUSED_PRE instantiation and compare them with s3's [sk3] attrs line before reading timing.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (master c3c1b01): it owns template variant 3 (<4,true,true>, the B=1 two-level) and the existing '[sk3]' log tag (submission.py:1435, 2043). The s3 branch also called its fused kernel 'v3' with an '[sk3]' tag. Name the new instantiation and its log lines distinctly (e.g. variant 4 / '[skF]'), or the 'variants 0-3 IDENTICAL' check and PATH CHOSEN grep will point at the wrong kernel. Its self-resetting ticket arrays are shared, so the v3 self-check must re-zero them in finally (as s3 did) before the timed path runs.
- full-decode-cuda-gather-splitkv / s2 stream-K v2 (master variant 0): this is v3's comparison base. v3 was measured against exactly this kernel, so that pairing holds. But if the FUSED_PRE port perturbs codegen for variant 0 (shared device helpers, inlining changes), the [pre-attn + v2] side of the gate moves too, and both runs' full rows drift. The IDENTICAL attrs print is the guard.
- fused-preattention-triton (+ paired-heads-launch-cleanup / ab-tune-batch-gate): _pre_attention_kernel is the reference math for v3, run with ROPE=False on full layers and an unpaired q layout on the gated path. A mismatch makes the self-check fall back silently to v2, and the run then measures the partner again. Grep PATH CHOSEN first.
- partner restack-sk-pseudo-b1-cluster-c16-c4 (not landed, same batch): its PSEUDO edits the same kernel template and its build ladder (-DSK_PSEUDO=1 with a landed-only rebuild fallback). If this run's combined build fails and falls back to landed-only, it drops PSEUDO and the cluster routes as well as v3, so the sliding rows stop pairing. Make the fallbacks independent: drop FUSED_PRE first, then PSEUDO.
- sliding-* landed hypotheses: v3 gates these shapes out. Sliding rows should match the partner's within noise, and any larger difference there is build/fallback confounding, not v3.

### Interference and A/B needs
No companion. The A/B partner (restack-sk-pseudo-b1-cluster-c16-c4) is already this change minus v3, and the instructions forbid a duplicate. Reverting a landed hypothesis isolates nothing, because no landed change shares v3's prologue; s3 already measured it against the same v2 base. The residual ambiguity is that a ~1.5% full-row difference between two jobs is noise-level. The fix is in-process, not another okc run. PATH CHOSEN must print, for each band, the interleaved graph-replay times of [pre-attn + v2] and [FUSED_PRE], plus the FUSED_PRE kernel-alone vs v2 kernel-alone time. Read the profile's top_kernels per job: whether _pre_attention_kernel disappears on full decode_16/32, and the attention kernel's own µs vs the partner's. If the gate rejects both bands, treat this run as a free replicate of the partner (a noise estimate), not as evidence against v3.

### Open questions
- Template flag vs separate module: s3's measured kernel lived in its own module. Folding it into master's template is cleaner long-term but is untested code on a 121/128-register kernel. Is the template port worth the risk compared with porting s3's module verbatim and keeping its own [skF] attrs line?
- Should the gate margin for B=2..16 be 1.01x instead of 1.02x, given that v3's diag showed only 1.027x at B=16 and prepare timings undercount the judge by 20-25%? As written, the band may flip on noise.
- If both runs pass, which one to submit should be decided by the in-process gate numbers and the full-row kernel µs, not by the two headlines, which differ by ~0.35% expected.
- The larger remaining tail on this kernel is the ~6 us serialized ticket merge (s2 diag), not pre-attn. A fused-pre + two-level merge at B=16 (nseg ~17) is the natural next stage.

