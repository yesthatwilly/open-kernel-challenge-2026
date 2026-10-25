# muse_attention_decode · 2026-10-25T091112Z · submit · restack-sk-pseudo-cluster-plus-v3-fused-preattn

Hypothesis: `muse_attention_decode/hypotheses/done/gain/2026-10-25T054448Z-restack-sk-pseudo-cluster-plus-v3-fused-preattn.md` (branch `hyp/muse_attention_decode/restack-sk-pseudo-cluster-plus-v3-fused-preattn`).

- **Job:** `d672f69ea69b4396` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `00f7debf`

## Goal
Promote check-proven gain restack-sk-pseudo-cluster-plus-v3-fused-preattn (1.4920x [1.4696,1.5144], check job cbd49dc94be548e7) to the board.

Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the stage-3 fused pre-attention prologue from branch hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn (job 825baa03798f431a): the v3 instantiation of splitkv_attn_sk_kernel that builds the FP8 q (RMSNorm + RoPE) and stores the new K/V row in-kernel, removing the _pre_attention_kernel launch, gated to full decode 1<B<=64 only. Full decode B=1 keeps master's two-level ticket with the separate pre-attention kernel (v3 has no two-level variant). Port v3 as a NEW template flag (FUSED_PRE, default false) on master's kernel so variants 0-3 and PSEUDO stay byte-identical (print sk_attrs IDENTICAL for the landed variants, as s1 does); v3 keeps its own self-check (cache rows bit-exact vs the pre-attention kernel, rel L2 vs fp32 reference) and its own bound constant. Add a gate: route v3 per band (B=2..16 from the B=16 timing, B=17..64 from B=32) only if the printed interleaved graph-replay time of [v3] beats [pre-attn + v2 ticket] by >= 1.02x (v3 measured 115.1 vs 118.2 us at B=16 and 220.5 vs 227.7 at B=32 in its diag). This run is the A/B partner of restack-sk-pseudo-b1-cluster-c16-c4 dispatched in the same batch: the only difference is v3, so the full decode_16/32 rows pair directly. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
Built standalone on master (c3c1b01 submission blob d8769ef) in the worktree's submission.py (no GPU/nvcc here: nothing compiled or run).
- (A) PSEUDO: s1's whole change (6f95ce7, whose base blob is master's) taken verbatim, including the -DSK_PSEUDO build with the landed-only rebuild fallback and its 1.05x gate. Its `_ps_replay_check` now goes through the new two-state `_alt_check`.
- (B) Cluster: `sliding_cluster_kernel` TU (module muse_sliding_cluster_v2, C=16 at B=1, C=4 at B=17..32, C=8 off) ported from stack-streamk-ticket-cluster-c16-c4; its v2 stream-K not taken. Audit: no module global holds a device pointer or buffer (only module handles and tags). Removed from kernel_fn the try/except and the `_slide_note` prints (spec rules); hook is `if _slide_gate(cfg): return _slide_forward(...)`. Added `_slide_route_checks`: two-state alternating graph replay through kernel_fn, then an interleaved cold-L2 timing gate [master Triton] vs [cluster] (routed if master/cluster >= 1.00x; the spec states no margin). Own bound constants (`_SLIDE_*`), nothing shared with PSEUDO.
- (C) FUSED_PRE: s3's measured fused kernel ported as its OWN module (muse_decode_skf_v1, kernel splitkv_attn_sk_fused_kernel, tag [skF], functions skf_*), built from this file's helper head. DEVIATION from the spec's template flag: this answers the review's open question in favour of the measured module; variants 0-5 are untouched by construction (so no 'IDENTICAL' risk beyond the existing print) and fallbacks are independent (FUSED_PRE module off first; PSEUDO has its own landed-only rebuild).
- Review fixes applied: distinct names/tags ([skF], not variant 4 since 4-5 are PSEUDO; not [sk3]); regs/local printed next to s3's [sk3] line (125 regs, 0 B local, 99360 B, 2 CTAs/SM) with IDENTICAL/DIFFERS; gate prints interleaved [pre-attn + v2] vs [FUSED_PRE] (and v2 alone) per band, B=16 -> B=2..16, B=32 -> B=17..64, margin 1.02x kept (open question on 1.01x not changed); shared tickets re-zeroed (`_rezero_tickets`) in a finally after the self-check and after diagnostics; ownership assertion cache_loc[b] == page_table[b, seq_b-1] plus one-owner-CTA check kept in `_skf_case`; full decode B=1 keeps the two-level path.
- Two-state check: `_alt_check` builds states A and B, captures a graph on each, alternates replays with in-place refills, compares each against the fp32 reference; used for PSEUDO, FUSED_PRE (B=16, B=32) and cluster (C=16, C=4). Gates use `_interleaved_us` (256 MB flush, empty-graph baseline subtracted if capturable).
- Small helper edits: `max_pages` kwarg on `_splitkv_synthetic` and `_slide_synthetic`.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn additions are launches only (`_slide_forward`, `_full_decode_skf`); syncs/.item() only in import-time checks (lint warns, outside kernel_fn).
- [x] Nothing derived from data cached across calls: tickets are self-resetting and data-free; `_SLIDE_MOD`/`_EXTF` are module handles.
- [x] n/a: weights (attention problem, no weights).
- [x] KV FP8, queries at most e4m3, fp32 softmax/merges, bf16 out (diff checked).
- [x] Every key attended in each window; fused kernel and cluster tile all keys.
- [x] No timing patches, harness access or reference import in the diff.
- [x] All work in the timed call; no output caching or replay; no child processes (grep: 0 matches).
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a around each load_inline and restored, -lineinfo, arch flags printed, cubin version check.

okc_lint: OK with warnings (only host-sync warnings, all outside kernel_fn: prepare-time synchronize()/item()). py_compile OK, __pycache__ deleted.

## Risks / what could fail
medium

## Result (after the run)
(pending)
