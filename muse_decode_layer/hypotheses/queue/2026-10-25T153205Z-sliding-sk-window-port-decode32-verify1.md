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
status: hold
review_at: 2026-10-25T203233Z
review_companions: null
hold_reason: "own condition: first measurement is an okc check (unchecked multi-thousand-line CUDA port; a sticky fault fails the run with no prepare log); only scored submits remained on 2026-10-25. Unhold when checks reset; apply the review's per-state tickets, distinct kernel symbols and variant sub-race"
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

