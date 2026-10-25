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
status: done
review_at: 2026-10-25T054723Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T054723Z
base_sha: 6f1f3a99d4b2849b4e653b5b3b88ff6f21aefaab
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/restack-sk-pseudo-cluster-plus-v3-fused-preattn
worktree: .worktrees/muse_attention_decode/restack-sk-pseudo-cluster-plus-v3-fused-preattn
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T084824Z
impl_commit: 7028a6bb7cc8dd826e6e3e0b26f5bf99614fe803
submission_sha256: 5589f1c4b89b8129c4d3f69d494c90f83ef283186028446d936efec0b2669ae1
norm_sha256: a0ca656e3fb25b2fa0a62203136188fdcdc618a45219b4d6c1429c2f670d2a1b
submission_blob: 84b54ae2c9800a7b9bf96692557dd390904ed09b
lint_warnings: 40
attempt_file: 2026-10-25T084824Z-check-restack-sk-pseudo-cluster-plus-v3-fused-preattn.md
prepared_at: 2026-10-25T084824Z
job: cbd49dc94be548e7
dispatched_at: 2026-10-25T084834Z
outcome: gain
verdict_detail: gain
speedup: 1.4919706177473366
speedup_ci: [1.4695910584811265, 1.5143501770135466]
harvested_at: 2026-10-25T091106Z
best_at_harvest: 1.444132502888963
promotion: submitted
promotion_commit: eb1b68bad24c7457aaa0606c1df0fa7733c03365
promotion_attempt: 2026-10-25T091112Z-submit-restack-sk-pseudo-cluster-plus-v3-fused-preattn.md
promotion_job: d672f69ea69b4396
promotion_at: 2026-10-25T110014Z
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


## Implementation notes

Built standalone on master (c3c1b01 submission blob d8769ef) in the worktree's submission.py (no GPU/nvcc here: nothing compiled or run).
- (A) PSEUDO: s1's whole change (6f95ce7, whose base blob is master's) taken verbatim, including the -DSK_PSEUDO build with the landed-only rebuild fallback and its 1.05x gate. Its `_ps_replay_check` now goes through the new two-state `_alt_check`.
- (B) Cluster: `sliding_cluster_kernel` TU (module muse_sliding_cluster_v2, C=16 at B=1, C=4 at B=17..32, C=8 off) ported from stack-streamk-ticket-cluster-c16-c4; its v2 stream-K not taken. Audit: no module global holds a device pointer or buffer (only module handles and tags). Removed from kernel_fn the try/except and the `_slide_note` prints (spec rules); hook is `if _slide_gate(cfg): return _slide_forward(...)`. Added `_slide_route_checks`: two-state alternating graph replay through kernel_fn, then an interleaved cold-L2 timing gate [master Triton] vs [cluster] (routed if master/cluster >= 1.00x; the spec states no margin). Own bound constants (`_SLIDE_*`), nothing shared with PSEUDO.
- (C) FUSED_PRE: s3's measured fused kernel ported as its OWN module (muse_decode_skf_v1, kernel splitkv_attn_sk_fused_kernel, tag [skF], functions skf_*), built from this file's helper head. DEVIATION from the spec's template flag: this answers the review's open question in favour of the measured module; variants 0-5 are untouched by construction (so no 'IDENTICAL' risk beyond the existing print) and fallbacks are independent (FUSED_PRE module off first; PSEUDO has its own landed-only rebuild).
- Review fixes applied: distinct names/tags ([skF], not variant 4 since 4-5 are PSEUDO; not [sk3]); regs/local printed next to s3's [sk3] line (125 regs, 0 B local, 99360 B, 2 CTAs/SM) with IDENTICAL/DIFFERS; gate prints interleaved [pre-attn + v2] vs [FUSED_PRE] (and v2 alone) per band, B=16 -> B=2..16, B=32 -> B=17..64, margin 1.02x kept (open question on 1.01x not changed); shared tickets re-zeroed (`_rezero_tickets`) in a finally after the self-check and after diagnostics; ownership assertion cache_loc[b] == page_table[b, seq_b-1] plus one-owner-CTA check kept in `_skf_case`; full decode B=1 keeps the two-level path.
- Two-state check: `_alt_check` builds states A and B, captures a graph on each, alternates replays with in-place refills, compares each against the fp32 reference; used for PSEUDO, FUSED_PRE (B=16, B=32) and cluster (C=16, C=4). Gates use `_interleaved_us` (256 MB flush, empty-graph baseline subtracted if capturable).
- Small helper edits: `max_pages` kwarg on `_splitkv_synthetic` and `_slide_synthetic`.

## Rules check

- [x] CUDA-graph capturable: kernel_fn additions are launches only (`_slide_forward`, `_full_decode_skf`); syncs/.item() only in import-time checks (lint warns, outside kernel_fn).
- [x] Nothing derived from data cached across calls: tickets are self-resetting and data-free; `_SLIDE_MOD`/`_EXTF` are module handles.
- [x] n/a: weights (attention problem, no weights).
- [x] KV FP8, queries at most e4m3, fp32 softmax/merges, bf16 out (diff checked).
- [x] Every key attended in each window; fused kernel and cluster tile all keys.
- [x] No timing patches, harness access or reference import in the diff.
- [x] All work in the timed call; no output caching or replay; no child processes (grep: 0 matches).
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a around each load_inline and restored, -lineinfo, arch flags printed, cubin version check.

okc_lint: OK with warnings (only host-sync warnings, all outside kernel_fn: prepare-time synchronize()/item()). py_compile OK, __pycache__ deleted.


## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading and by emulating in Python. No GPU was used, nothing was compiled, and the file was not edited.

**Verdict: dispatch.** No fixes were needed. okc_lint reports OK with warnings: every host sync it flags is import-time, outside kernel_fn. py_compile passes, and the __pycache__ it left was deleted.

What I checked:
- FUSED_PRE kernel. Its body is byte-identical to s3 825baa03 (`splitkv_attn_sk_fused_kernel`). Only the host-side names changed, sk3_* to skf_*. The helper head it is built on differs from s3 in one way: merge_slots now calls merge_range with the same arithmetic, plus the inert PSEUDO additions. Every symbol it uses (finalize_store, merge_slots, range_start, tile_owner, load_prefix, cp_async*, mma16816) is defined before the `template <int STAGES, bool TICKET` cut point. Its q math (scaleless RMSNorm, then e4m3, with 5.473 folded into the score scale) and its k/v stores (rms/k_scale, v/v_scale, both e4m3) match master's `_pre_attention_kernel` with ROPE=False. Every tile of every request is attended.
- Gates that can never pass. reference.generate_input sets cache_loc = page_table[b, seq_lens[b]] = page_table[b, cache_seqlens - 1], so the ownership assertion holds on the judge's data. `_skf_owner_check` was emulated on 80k random cases (B=2..64; grids 3, 7, 264 and random) with 0 failures: it is the exact inverse of range_start/tile_owner. The tolerances (out vs fp32 < 2e-2, vs v2 < 5e-3, new rows < 2e-2) are the ones s3 passed. Every synthetic builder returns all of `_ALT_KEYS`, and `_alt_sets` keeps the shapes fixed through max_pages.
- Fallbacks. The build, validate, per-band gate, per-band two-state replay and the whole `_skf_setup` are each wrapped. Every miss prints a PATH CHOSEN line and lands on pre-attn + v2. The shared tickets are re-zeroed in a finally block. PSEUDO and the cluster route fall back independently.
- Rules. The kernel_fn additions are config-only predicates plus launches (`_fused_gate`, `_slide_gate`/`_slide_clusters`, `_ps_gate`). kernel_fn has no sync, print or try, and allocates only from shapes. No module global or C++ static holds a device pointer, tensor map or state buffer: only module handles, route dicts, the theta-keyed `_inv_freq`, and the self-resetting int32 tickets. Outside the gated shapes, kernel_fn is master's code path unchanged.
- Budget. There are four nvcc TUs (splitkv v1, stage-2 + PSEUDO, skF, sliding fp8qk), at about 30-45 s each in recent logs, so about 3 min of compiles. The checks and interleaved timings take seconds each. This fits the 15-min run.
- Fidelity. Two deviations, both noted but not changed. (1) FUSED_PRE was built as its own module, not as a template flag; the review's open question allowed this. (2) The sliding cluster's Python glue comes from stack-streamk-ticket-cluster-c16-c4, not from the partner, and adds a >= 1.00x interleaved timing gate. The partner routes config-only. Its _SLIDE_CUDA text is identical to the partner's. The partner's `_phase` timing prints and its sk_attrs re-check with both TUs loaded were dropped.

Residual risk: the 1.02x FUSED gate (s3 measured 1.027x at B=16) and the 1.00x cluster gate can flip on prepare-time noise. If either does, this run's sliding or full rows stop pairing cleanly with the partner's, so grep the PATH CHOSEN lines before comparing the two runs.

## Result

- job `cbd49dc94be548e7` (check), status done, passed True, tests 16/16
- speedup 1.4920x, 95% CI [1.4696, 1.5144], score 2524.379471820013 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json
- **outcome: gain** (gain)
- files: `muse_attention_decode/hypotheses/done/gain/restack-sk-pseudo-cluster-plus-v3-fused-preattn.result.json`, `muse_attention_decode/hypotheses/done/gain/restack-sk-pseudo-cluster-plus-v3-fused-preattn.patch`; branch `hyp/muse_attention_decode/restack-sk-pseudo-cluster-plus-v3-fused-preattn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.15 | 11.89 | 1.1026 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 22.6 | 22.48 | 0.9973 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.33 | 15.41 | 0.9949 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 111.14 | 111.34 | 1.0013 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.33 | 20.64 | 1.1846 | gained |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.42 | 227.88 | 0.9973 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.78 | 13.33 | 1.3305 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.51 | 30.27 | 0.999 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.68 | 23.1 | 0.989 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.8 | 163.37 | 0.9899 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.34 | 28.93 | 0.992 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.2 | 303.64 | 0.9929 | flat |

gained on sliding decode_1, decode_32, verify_1. Expected hit: 3/5. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

