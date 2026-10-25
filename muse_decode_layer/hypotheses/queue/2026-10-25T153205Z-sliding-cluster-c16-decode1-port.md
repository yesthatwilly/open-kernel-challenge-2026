---
problem: muse_decode_layer
slug: sliding-cluster-c16-decode1-port
created: 2026-10-25T153205Z
base: master
area: FA3 call (sliding decode, batch 1)
change: >
  Port muse_attention_decode master's sliding_cluster_kernel at cluster size C=16 for sliding
  decode batch 1 only (q_len 1, page 1, tp8), replacing decode's _sliding_fused_kernel (12.8 us,
  profile 34a9be06918d448b, sliding decode_1, weight 46). In attention the C=16 cluster took
  sliding decode_1 from 13.15 to 12.57 us (stack-streamk-ticket-cluster-c16-c4) and attention
  master now runs it at 10.8 us kernel / 11.9 us shape vs SGLang 20.3; like _sliding_fused_kernel
  it does the QK-norm, RoPE, FP8 k/v cache store, windowed attention over 2,048 keys, the DSMEM
  merge across the cluster and the norm-gate epilogue in one launch. Copy the cluster TU and its
  host wrapper verbatim from attention master into a new load_inline module
  (muse_decode_slide_c16_v1; attention builds it as muse_sliding_cluster_v2, with the non-portable
  cluster-size attribute set) and launch it only at cluster size C=16; feed it decode's qkv (bf16
  [1, 2304]) and write the bf16 [1, Hq*D] o-GEMM input. Route in _parent_attention's existing
  'window and batch == 1 and length == 1' branch on this state's flag only. Self-check: use
  attention master's acceptance rules, NOT decode's stage-2 byte-exact torch-cast probe
  (sliding-cluster-c4-decode32-s2 fell back on that probe and was never measured): o rel-L2 vs an
  fp32 windowed reference with accurate sincosf rope, cache k bytes within one E4M3 code by
  sign-magnitude ordinal on < 6% of bytes and v byte-exact, the satfinite cvt accepted (the s2
  probe showed torch's cast saturates on this image); lens 1, 63, 2047, 2048, 2049, 80000,
  kv_scale 0.015625, positions up to 2^20; two alternating states with refilled inputs. Selector
  at B=1: cluster vs _sliding_fused_kernel by the standard interleaved graph timing, >= 1.02x.
  HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no
  prints, no try/except, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow and no per-call mutable Python state; gates read config/shape
  and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4
  stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key
  attended; no child processes. The judge prepares TWO state copies and rotates them: keep every
  device pointer, TMA map, ticket array and workspace per state, never in a module global. Every
  new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode
  arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an
  in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any
  miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check
  by capturing once and replaying with refilled inputs while alternating two states. Selectors
  time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps,
  empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS
  by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per
  module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First
  measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "_sliding_fused_kernel 12.8 us -> ~10.8 us (attention master's C=16 kernel row); ~-2 us on 117.8 us, the heaviest-ratio shape (2.9x its 40 us floor)"
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: batch > 1 untouched
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: q_len 8 untouched
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-25T203413Z
review_companions: null
hold_reason: "adversarial review: Plausible but small (~2 us on the 118.5 us sliding decode_1 row, ~0.25% headline); its own first measurement must be an okc check, and it is an unchecked cross-"
---

# sliding-cluster-c16-decode1-port

## Change

Port muse_attention_decode master's sliding_cluster_kernel at cluster size C=16 for sliding decode batch 1 only (q_len 1, page 1, tp8), replacing decode's _sliding_fused_kernel (12.8 us, profile 34a9be06918d448b, sliding decode_1, weight 46). In attention the C=16 cluster took sliding decode_1 from 13.15 to 12.57 us (stack-streamk-ticket-cluster-c16-c4) and attention master now runs it at 10.8 us kernel / 11.9 us shape vs SGLang 20.3; like _sliding_fused_kernel it does the QK-norm, RoPE, FP8 k/v cache store, windowed attention over 2,048 keys, the DSMEM merge across the cluster and the norm-gate epilogue in one launch. Copy the cluster TU and its host wrapper verbatim from attention master into a new load_inline module (muse_decode_slide_c16_v1; attention builds it as muse_sliding_cluster_v2, with the non-portable cluster-size attribute set) and launch it only at cluster size C=16; feed it decode's qkv (bf16 [1, 2304]) and write the bf16 [1, Hq*D] o-GEMM input. Route in _parent_attention's existing 'window and batch == 1 and length == 1' branch on this state's flag only. Self-check: use attention master's acceptance rules, NOT decode's stage-2 byte-exact torch-cast probe (sliding-cluster-c4-decode32-s2 fell back on that probe and was never measured): o rel-L2 vs an fp32 windowed reference with accurate sincosf rope, cache k bytes within one E4M3 code by sign-magnitude ordinal on < 6% of bytes and v byte-exact, the satfinite cvt accepted (the s2 probe showed torch's cast saturates on this image); lens 1, 63, 2047, 2048, 2049, 80000, kv_scale 0.015625, positions up to 2^20; two alternating states with refilled inputs. Selector at B=1: cluster vs _sliding_fused_kernel by the standard interleaved graph timing, >= 1.02x. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_1 *` faster: _sliding_fused_kernel 12.8 us -> ~10.8 us (attention master's C=16 kernel row); ~-2 us on 117.8 us, the heaviest-ratio shape (2.9x its 40 us floor)
- `*layer=full *` flat: untouched
- `*layer=sliding load=decode_16*` flat: batch > 1 untouched
- `*layer=sliding load=verify_*` flat: q_len 8 untouched

## Risk

Small (~1.5% on one weight-46 row, ~0.25% headline) but attributable to one kernel row, and it is the batch-1 half of the cluster port whose batch-17..32 half is queued as sliding-cluster-c4-s3-satfinite-on-sk-ticket (that branch is unlanded, so this is on master and B=1 only; if both land, one cluster module with C=4 and C=16 instantiations). Risk: a cluster launch faulting in prepare poisons the CUDA context (LESSONS), so run the 1-CTA-cluster canary first and print after each launch; the extra build adds ~40 s to prepare.

## Adversarial review

Reviewed 2026-10-25T203413Z by an adversarial subagent (model opus).

**Verdict: hold.** Plausible but small (~2 us on the 118.5 us sliding decode_1 row, ~0.25% headline); its own first measurement must be an okc check, and it is an unchecked cross-problem port of a ~1,300-line cluster CUDA module that cannot be compiled here. Hold until checks reset, like sliding-sk-window-port-decode32-verify1 and combo-attn-master-sliding-race-skw-c16 (its A2).

### Justification
BEST a903b9d0: _sliding_fused_kernel 13.5 us (10.9%) of 118.5 us vs SoL 40.0; attention's C=16 cluster kernel does the same fused work in 10.8 us (~1.2x). The branch is in _attn_core (submission.py:7216), not _parent_attention. Prior (missing): sliding-cluster-c4-decode32-port / -s2-k-encode-match (never measured: s2 fell back on the byte-exact probe); difference is real (C=16, B=1, attention's acceptance rules). Weakest link: a cluster-16 launch or DSMEM/ticket fault in prepare poisons the CUDA context and fails the whole scored run with no prepare log.

### Confounders on the current kernel
- sliding-decode1-fused-restack-on-pdl: installed the incumbent; the selector compares directly.
- small-t-launch-minimal-dense-chain-pdl / dense-chain-s3: PDL around attention; time attention + o GEMM in one graph.
- combo-measured-rows-slide-batched-preattn-t8-tail: disjoint (B>=2).

### Interference and A/B needs
No companions: one gated shape class with its own in-prepare race; time it as attention kernel + o GEMM.

### Open questions
- Module-level statics in attention's cluster host wrapper?
- Query cudaOccupancyMaxActiveClusters for C=16 before the first launch.
- Fold into sliding-cluster-c4-s3-satfinite-on-sk-ticket as one module with C=4 and C=16?
- Quote savings against a903b9d0 (13.5 us).

