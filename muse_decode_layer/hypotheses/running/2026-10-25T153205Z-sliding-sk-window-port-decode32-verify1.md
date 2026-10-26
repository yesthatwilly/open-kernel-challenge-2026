---
problem: muse_decode_layer
slug: sliding-sk-window-port-decode32-verify1
created: 2026-10-25T153205Z
base: master
area: "FA3 call (sliding decode B=17..32 and sliding verify B=1: windowed stream-K)"
change: >
  Port muse_attention_decode master's windowed stream-K sliding routes (commit 7e12891512, best
  1.5017x check 252abb7f11ec48c3; hypotheses restack-sliding-decode-sk-window-on-cluster-master
  and restack-sk-pseudo-cluster-plus-v3-fused-preattn) for exactly two sliding bands: (a) sliding
  decode 17 <= batch <= 32 (q_len 1): _pre_attention_kernel (PAIRED=False, RoPE on, writes k/v FP8
  into the cache and the FP8 q) + splitkv_attn_sk_kernel PSEUDO TOK=1 single-level ticket (variant
  6), which in attention took sliding decode_32 from 23.85 to 17.38 us shape (sk 13.9 + pre 1.9)
  and beat the cluster C=4 kernel there (20.6); (b) sliding verify batch 1 (q_len 8):
  _pre_attention_kernel + the PSEUDO TOK=8 two-level kernel (variant 5), attention's sliding
  verify_1 = 13.2 us shape (sk 10.0 + pre 1.7) vs SGLang 22.6. Decode master runs both bands
  through FusedQKNorm + k/v/q FP8 casts + store_cache + prepare_varlen_num_blocks + FA3 +
  FlashAttnFwdCombine + FusedRMSNormGate (sliding decode_32: FA3 18.9 + combine 3.2 + prepare ~2.4
  + 5-6 small launches; sliding verify_1: FA3 9.7 + combine 4.1 + prepare + the same small
  launches). Build attention master's sk CUDA TU as a NEW load_inline module
  (muse_decode_sk_window_v1) next to decode's existing _SK module, which stays byte-identical for
  the full decode paths; copy _pre_attention_kernel, the variant table and the PSEUDO prefix
  loader verbatim; the kernels read the unpaired FP8 q and write the normed, gated bf16 [T, Hq*D]
  input of the o GEMM (gate read from qkv at SK_QKV_GATE_OFF as decode's sk does; assert the qkv
  row layout matches attention's). Route in _parent_attention FIRST, before the existing branches,
  on config only (window != 0, page 1, and (q_len 1 and 17 <= batch <= 32) or (q_len 8 and batch
  1)) plus this state's per-band flag; kv_scale != 1 must be handled exactly as attention handles
  it (descale inside the kernel) or the band falls back. Self-check vs an fp32 windowed reference:
  lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv_scale 0.015625, both bands, two alternating
  states with refilled inputs; cache bytes compared by rel-L2 against an accurate-rope reference
  (LESSONS: Triton sin/cos differ in 1-6% of E4M3 bytes at large positions), not byte-exact
  against torch's cast. SELECTOR per band (B=32 for 17..32, B=1 q_len 8): the whole attention side
  from qkv to the o-GEMM input, new [pre-attn + sk] vs master's chain, standard interleaved graph
  timing, route at >= 1.02x. Print '[skw] PATH CHOSEN <band>: new X us vs master Y us'. If
  sliding-cluster-c4-s3-satfinite-on-sk-ticket (queued, same B=17..32 band, cluster C=4 kernel)
  lands first, this band's selector must time against that incumbent instead. HARD RULES (every
  path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except,
  no allocation sized by input values (shape/config only), no data-dependent Python control flow
  and no per-call mutable Python state; gates read config/shape and this state's own prepare()
  flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in
  registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child
  processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA
  map, ticket array and workspace per state, never in a module global. Every new CUDA build is
  sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with
  -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize +
  'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback:
  <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with
  refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved
  CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted),
  print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the
  enabled choice equals the printed winner. Print compile seconds per module and the cumulative
  prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "attention side ~30 us (FA3 18.9 + combine 3.2 + prepare + 5-6 launches) -> ~16-18 us; 236.4 -> ~224 us on a weight-46 row (master's worst ratio, 1.075x)"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: ~22-25 us attention chain -> ~12-14 us; 189.9 -> ~180 us (weight 46)
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: "B=2..16 band stays FA3 (attention: neither cluster nor Triton nor sk beats it there)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: keeps _sliding_fused_kernel
  - shapes: "*layer=full *"
    direction: flat
    why: "untouched; decode's _SK module unchanged"
  - shapes: "*layer=sliding load=verify_16*"
    direction: flat
    why: B=16 verify keeps FA3
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T203233Z
review_companions: null
hold_reason: null
stage: implemented
claimed_at: 2026-10-25T213134Z
base_sha: 4b5ab758064dec781322c9326848d77ac8ab9b96
base_blob: 3fc72e97304e132db432fcc87872be997f842929
branch: hyp/muse_decode_layer/sliding-sk-window-port-decode32-verify1
worktree: .worktrees/muse_decode_layer/sliding-sk-window-port-decode32-verify1
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T005253Z
impl_commit: f798110f78661e2bd739fe420dff86ec8795b0f4
submission_sha256: fa63b1fb5cbcf0af88c7c1db7264f9f0f273cdf968b31845ea92289bf99e7336
norm_sha256: 12c87fd68d38764b1d0d66d37a797148eb347d4ede0c1ce35e47ccd2ad2b8c4b
submission_blob: 2e0687a9bdbdcede1b35478fb931d2a37e0dbbc4
lint_warnings: 78
attempt_file: 2026-10-26T012541Z-check-sliding-sk-window-port-decode32-verify1.md
prepared_at: 2026-10-26T012541Z
---

# sliding-sk-window-port-decode32-verify1

## Change

Port muse_attention_decode master's windowed stream-K sliding routes (commit 7e12891512, best 1.5017x check 252abb7f11ec48c3; hypotheses restack-sliding-decode-sk-window-on-cluster-master and restack-sk-pseudo-cluster-plus-v3-fused-preattn) for exactly two sliding bands: (a) sliding decode 17 <= batch <= 32 (q_len 1): _pre_attention_kernel (PAIRED=False, RoPE on, writes k/v FP8 into the cache and the FP8 q) + splitkv_attn_sk_kernel PSEUDO TOK=1 single-level ticket (variant 6), which in attention took sliding decode_32 from 23.85 to 17.38 us shape (sk 13.9 + pre 1.9) and beat the cluster C=4 kernel there (20.6); (b) sliding verify batch 1 (q_len 8): _pre_attention_kernel + the PSEUDO TOK=8 two-level kernel (variant 5), attention's sliding verify_1 = 13.2 us shape (sk 10.0 + pre 1.7) vs SGLang 22.6. Decode master runs both bands through FusedQKNorm + k/v/q FP8 casts + store_cache + prepare_varlen_num_blocks + FA3 + FlashAttnFwdCombine + FusedRMSNormGate (sliding decode_32: FA3 18.9 + combine 3.2 + prepare ~2.4 + 5-6 small launches; sliding verify_1: FA3 9.7 + combine 4.1 + prepare + the same small launches). Build attention master's sk CUDA TU as a NEW load_inline module (muse_decode_sk_window_v1) next to decode's existing _SK module, which stays byte-identical for the full decode paths; copy _pre_attention_kernel, the variant table and the PSEUDO prefix loader verbatim; the kernels read the unpaired FP8 q and write the normed, gated bf16 [T, Hq*D] input of the o GEMM (gate read from qkv at SK_QKV_GATE_OFF as decode's sk does; assert the qkv row layout matches attention's). Route in _parent_attention FIRST, before the existing branches, on config only (window != 0, page 1, and (q_len 1 and 17 <= batch <= 32) or (q_len 8 and batch 1)) plus this state's per-band flag; kv_scale != 1 must be handled exactly as attention handles it (descale inside the kernel) or the band falls back. Self-check vs an fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv_scale 0.015625, both bands, two alternating states with refilled inputs; cache bytes compared by rel-L2 against an accurate-rope reference (LESSONS: Triton sin/cos differ in 1-6% of E4M3 bytes at large positions), not byte-exact against torch's cast. SELECTOR per band (B=32 for 17..32, B=1 q_len 8): the whole attention side from qkv to the o-GEMM input, new [pre-attn + sk] vs master's chain, standard interleaved graph timing, route at >= 1.02x. Print '[skw] PATH CHOSEN <band>: new X us vs master Y us'. If sliding-cluster-c4-s3-satfinite-on-sk-ticket (queued, same B=17..32 band, cluster C=4 kernel) lands first, this band's selector must time against that incumbent instead. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_32*` faster: attention side ~30 us (FA3 18.9 + combine 3.2 + prepare + 5-6 launches) -> ~16-18 us; 236.4 -> ~224 us on a weight-46 row (master's worst ratio, 1.075x)
- `*layer=sliding load=verify_1 *` faster: ~22-25 us attention chain -> ~12-14 us; 189.9 -> ~180 us (weight 46)
- `*layer=sliding load=decode_16*` flat: B=2..16 band stays FA3 (attention: neither cluster nor Triton nor sk beats it there)
- `*layer=sliding load=decode_1 *` flat: keeps _sliding_fused_kernel
- `*layer=full *` flat: untouched; decode's _SK module unchanged
- `*layer=sliding load=verify_16*` flat: B=16 verify keeps FA3

## Risk

Weakest links: (1) the extra load_inline build (~40 s) on top of four modules; print compile seconds and fall back if prepare is past its deadline. (2) Decode's attention input differs from attention's: here qkv comes out of a GEMM with the gate columns interleaved in decode's layout, so the norm-gate epilogue's gate stride must be asserted (dl3 piece A failed its own self-check on exactly this, 'gate stride 2304'). (3) The selector compares against a longer chain than attention did, so the margin should be larger here; if PATH CHOSEN keeps master for a band, that band is unmeasured.

## Adversarial review

Reviewed 2026-10-25T203233Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, no companion: the sanctioned A1 split of the held combo-attn-master-sliding-race-skw-c16. On master both bands still run the FA3 chain (Sd32 FA3 18.6 us, Sv1 10.0 us). The risks are reading and safety: a scored submit returns no prepare log, and a sticky CUDA fault in an unchecked multi-thousand-line port cannot fall back.

### Justification
Collapsing each band to pre + one sk kernel could save ~8-14 us per row (weight 46 each; Sd32 is master's worst ratio 1.087x); headline ~1-1.5%, inside the CI, read per row. Weakest link: in-layer transfer on B=17..32 (sliding-cluster-c4-decode32-port and S1 class B both beat FA3 standalone and moved nothing in-layer; no griddepcontrol in the new kernels). Variant choice: attention's logs chose variant 4 at grid p8 for SV1, variant 6 at p4 for B=17..32. Prior missing: sliding-cluster-c4-decode32-port/-s2, sliding-fused-batched-decode-port, combo-attn-master-sliding-race-skw-c16 (A1).

### Confounders on the current kernel
- combo-measured-rows-slide-batched-preattn-t8-tail: the selector must time against _attn_core as each state routes it; route skw before S1/S3.
- small-t-launch-minimal-dense-chain-pdl / dense-chain-s3: new kernels lack griddepcontrol; time qkv GEMM through o-GEMM input.
- full-decode-sk-ticket-restack-b1-b64 / dl3-piece-a: per-state tickets; give the new kernels distinct symbol names.
- stack-align-par-sidestream-sliding1: Sd1 is the drift control.

### Interference and A/B needs
No companion; attribution per row (only Sd32 and Sv1 can change). Distinct kernel symbols (skw_attn_kernel, skw_pre_kernel) and an in-prepare variant x grid sub-race replace a second run.

### Open questions
- SV1 variant/grid: sub-race {4,5} x {p4,p8,sm}; SB32 {6} x {p2,p4,p8}.
- Selector graph must include qkv GEMM through o-GEMM input with PDL neighbours.
- Distinct symbols so top_kernels shows which path ran.
- Wait for checks, per its own text?
- Serialize with sliding-cluster-c4-s3-satfinite-on-sk-ticket.


## Implementation notes

Edited only muse_decode_layer/submission.py (+1348/-28). Nothing could be compiled or run here (no GPU, torch or nvcc).

What changed
- New load_inline module `muse_decode_sk_window_v1` (built at import, TORCH_CUDA_ARCH_LIST=9.0a, -O3 -std=c++20 -lineinfo, compile seconds printed). Its TU is attention master's sk TU, kernel body verbatim, reduced to the PSEUDO instantiations 4-7. The CUDA kernel is renamed `skw_attn_kernel`, the exports `skw_attn` / `skw_attrs`, and the combine and non-PSEUDO variants are removed. The host code checks the qkv row layout (2304 bf16 columns, gate at 1280, matching master's qkv GEMM). Decode's `_SK` module is untouched.
- Triton `skw_pre_kernel` is `_pre_attention_kernel` copied under its own symbol. Both new kernels therefore have distinct names in top_kernels.
- Routing: `_attn_core` calls `_skw_route` right after the qkv GEMM and before the S1 / S3 / FA3 branches. The route is config plus the state's own per-band flag (b32 = sliding decode 17..32, v1 = sliding verify B=1). kv_scale is descaled inside the kernel. Per-state tickets and workspaces live in `state["skw"]`.
- Phase SKW runs after T8 and has its own start deadline (150 s), which prints "skipped: deadline". It runs a canary (a tiny first launch of each variant, with launching/ok prints and sync), the pre-kernel front checks, the kernel suite with plants, the variant x grid sub-race, a two-state alternation and a whole-layer check.
- Kernel suite and plants: lens 1/63/2047/2048/2049/80000 plus ragged, kv_scale 1 and 1/64, plants at the window edges. The reference is fp64 fed the kernel's own q bytes.
- Sub-race: b32 {6} x {p2,p4,p8}; v1 {4,5} x {p4,p8,sm}. Each candidate is timed against master's routed path (qkv GEMM through the o GEMM, other flags as applied) in two interleaved rounds. A candidate must win by >= 1.03x in both rounds with IQR <= 1.5x master's (master's IQR floored at 0.5 us). All medians and IQRs are printed.
- The last lines of prepare are compact `[summary] PATH CHOSEN` lines, written after `_prepare_done`.

Review fixes applied: per-state tickets and buffers; distinct kernel symbols; the variant sub-race; selector timed against `_parent_attention` as each state routes it, with skw routed first; fallback lines carry a reason.

Deviations
- Shared helpers were generalized without changing their defaults: `_cb_flags` (skw), `_cb_cmp_attn` (row_agg), `_cb_cmp_layer` (loose, k_gap_max), `_s3_front_check` (pre_fn, gap_gate), `_cb_attn_ref` (q_override). The skw layer check also uses loose bounds, because both sides round q to FP8 independently (the earlier S1/S3 gates failed on that noise).
- The existing "start deadline" message now reads "skipped: deadline (...)".
- The SKW layer check is run inside the SKW phase and not in the T8 check, so a skw miss cannot disable unrelated flags.
- Hypothesis text says route at >= 1.02x; the brief's standing 1.03x was used.
- Pre-kernel front checks use rel L2 only, no byte-gap gate (LESSONS).
- Open questions answered: the sub-race is as the review lists; the selector includes the o GEMM; symbols are distinct. If sliding-cluster-c4-s3-satfinite lands, the b32 selector automatically times against whatever S1/S3 flags the state holds. No serialization with that hypothesis was done.

## Rules check
- [x] CUDA-graph capturable: kernel_fn path adds only torch.empty, Triton launch and skw_attn launch (shape-based); no sync, item or value-dependent control flow.
- [x] Nothing derived from data is cached across calls; the race results held in `_SKW_RES` are plain floats from synthetic prepare data.
- [x] Weights: unchanged (NVFP4 stays NVFP4, BF16 stays BF16).
- [x] Activations BF16 as master; KV FP8; q FP8 E4M3.
- [x] Attention sees every key: window 2048 = positions p-2047..p, no skipping (plants test the edges).
- [x] No subprocess, network, timing or harness patching (nvcc via torch load_inline, as the other modules).
- [x] All work inside the timed call; no output caching or replay.
- [x] Per-state flags, a miss falls back to master's exact path and no unrelated flag is touched.
- [x] Bounded waits: the kernel has no spin or mbarrier (ticket arrival only).

okc_lint:
```
OK with warnings: .worktrees/muse_decode_layer/sliding-sk-window-port-decode32-verify1/muse_decode_layer/submission.py
```
py_compile ok; __pycache__ removed.

## Pre-flight audit

Verdict: dispatch (no edits made). Checked: (1) every generalized helper (_cb_flags skw, _cb_cmp_attn row_agg, _cb_cmp_layer loose/k_gap_max, _s3_front_check pre_fn/gap_gate/label, _cb_attn_ref q_override) keeps its old behavior at its existing master call sites (defaults reproduce the old code and tag text; _cb_off now also forces skw routes off, a no-op while state["skw"] is absent or its route is all None, which holds through T8); (2) the SKW layer check is loose only on the layer residual, while the kernel-level suite (fp64 reference fed the kernel's own q bytes, window-edge plants, <= 1.5e-2 total / 5e-2 per row) and the row-aggregated attention alternation still fail an O(1) wrong row; the new path is fp16-MMA so it is closer to the fp64 reference than master's FP8-P FA3, so the "new <= 1.25x master" gate does not trip; (3) tickets, workspaces and routes live in state["skw"], cloned per state by _cb_second_state, no module-global device pointer (_SKW_RES holds floats/tuples, the C++ static is a host bool array); (4) SKW starts at <= 150 s into prepare() (prior master logs show prepare at ~7 s), the cell/case deadlines (240/260 s) bound the rest below 300 s, and the compile adds ~45 s to the import clock. The CUDA TU diffs against attention master as rename + PSEUDO-only + qkv layout TORCH_CHECK; skw_pre_kernel is byte-identical to _pre_attention_kernel.
Residual risk: the race's IQR gate (candidate <= 1.5x master's, master floored at 0.5 us) can reject a real win on 0.5 us event quantization, which would leave both bands unmeasured on master's path.
