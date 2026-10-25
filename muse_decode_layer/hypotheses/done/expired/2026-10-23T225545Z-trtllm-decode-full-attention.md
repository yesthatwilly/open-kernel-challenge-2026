---
problem: muse_decode_layer
slug: trtllm-decode-full-attention
created: 2026-10-23T225545Z
base: master
area: FA3 call
change: >
  On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1), replace the
  flash_attn_with_kvcache call with FlashInfer's plan-free TensorRT-LLM decode kernel,
  flashinfer.decode.trtllm_batch_decode_with_kv_cache, which takes the FP8 paged k/v caches, the
  int32 block table ([B, max_pages], page size 1 as given), seq_lens as a device tensor and
  max_seq_len as a host int from the table extent, so it is CUDA-graph safe without any host-side
  plan. q is the same FP8 [T, 16, 64] FA3 receives (or bf16 if the kernel requires bf16 queries;
  rules allow either), softmax scale sm_scale * qk_scale_factor, bf16 output into the same
  FusedRMSNormGate. Sliding layers and verify blocks (q_len=8, which trtllm decode does not cover)
  keep FA3 exactly as before. The motivation is the full-decode gap in FINDINGS: decode_32 full is
  623 us for SGLang against 261 us on the sliding twin, i.e. ~360 us for 655 MB of KV (137 us at
  the 4.8 TB/s floor), and decode_16 full 165 us over its twin for a 68 us delta; FA3 at page size
  1 gathers each 128-byte KV row through the page table, and the trtllm-gen decode kernel is built
  for exactly that block-table layout. Attribution: only layer=full decode shapes move; every
  other shape is a control.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: largest absolute gap (~220 us above the KV floor) on a kernel built for block-table decode
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: ~100 us above its KV floor
  - shapes: "*layer=full load=decode_1*"
    direction: faster
    why: "B=1: one request over 80K rows, split-KV inside the trtllm kernel; direction uncertain but the shape is small"
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 unchanged; control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len=8 keeps FA3; control
risk: high
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 5
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-24T132507Z
review_companions: null
hold_reason: "adversarial review: Hold. The spec was written on 2026-10-23 against the parity base. Since then master has landed head-paired FA3 for full layers at batch>1, so as written this ch"
outcome: expired
verdict_detail: "hold expired after 32 h: adversarial review: Hold. The spec was written on 2026-10-23 against the parity base. Since then master has landed head-"
expired_at: 2026-10-25T211328Z
---

# trtllm-decode-full-attention

## Change

On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1), replace the flash_attn_with_kvcache call with FlashInfer's plan-free TensorRT-LLM decode kernel, flashinfer.decode.trtllm_batch_decode_with_kv_cache, which takes the FP8 paged k/v caches, the int32 block table ([B, max_pages], page size 1 as given), seq_lens as a device tensor and max_seq_len as a host int from the table extent, so it is CUDA-graph safe without any host-side plan. q is the same FP8 [T, 16, 64] FA3 receives (or bf16 if the kernel requires bf16 queries; rules allow either), softmax scale sm_scale * qk_scale_factor, bf16 output into the same FusedRMSNormGate. Sliding layers and verify blocks (q_len=8, which trtllm decode does not cover) keep FA3 exactly as before. The motivation is the full-decode gap in FINDINGS: decode_32 full is 623 us for SGLang against 261 us on the sliding twin, i.e. ~360 us for 655 MB of KV (137 us at the 4.8 TB/s floor), and decode_16 full 165 us over its twin for a 68 us delta; FA3 at page size 1 gathers each 128-byte KV row through the page table, and the trtllm-gen decode kernel is built for exactly that block-table layout. Attribution: only layer=full decode shapes move; every other shape is a control.

## Expected

- `*layer=full load=decode_32*` faster: largest absolute gap (~220 us above the KV floor) on a kernel built for block-table decode
- `*layer=full load=decode_16*` faster: ~100 us above its KV floor
- `*layer=full load=decode_1*` faster: B=1: one request over 80K rows, split-KV inside the trtllm kernel; direction uncertain but the shape is small
- `*layer=sliding*` flat: FA3 unchanged; control
- `*load=verify_*` flat: q_len=8 keeps FA3; control

## Risk

Risk: HIGH on feasibility, not on rules. Before spending a run, the implementer must verify in the image that trtllm_batch_decode_with_kv_cache supports head_dim 64, 2 kv heads with GQA 8, page size 1 (the trtllm-gen kernels may require page size >= 16 or head_dim 128: if either is unsupported, mark this hypothesis rejected with the exact error, spend no run) and the FP8 e4m3 cache with a unit scale; the k_scale/v_scale test (1/64) needs the kernel's bmm1/bmm2 scale arguments to carry the descale. The kernel must attend over every key up to seq_lens (no sparsity: it does not, so the every-key rule holds). Output must be bf16 and the -1 entries past seq_lens in the page table must never be read. The ctx 1048576 q_len=8 test stays on FA3. If it lands, the same swap for verify blocks is a separate hypothesis.

## Adversarial review

Reviewed 2026-10-24T132507Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. The spec was written on 2026-10-23 against the parity base. Since then master has landed head-paired FA3 for full layers at batch>1, so as written this change would quietly drop the pairing win on decode_16/32. Whether the kernel can run at all on an H200 at page size 1 and head_dim 64 cannot be checked from this node, and decode has one check left with fused-preattention-triton still in the judge.

### Justification
A. The motivation numbers are stale. The spec cites SGLang's 623 us at decode_32 full and a 360 us gap, and it argues that FA3 'gathers each 128-byte KV row'. On master (fb07a3bd3aa54baf) the full-layer FA3 kernel is already paired. It reads one 128-byte row per token for both KV heads and costs 152.3 us at decode_16 and 290.2 us at decode_32. That is about 2.26 TB/s at decode_32 over its ~655 MB of KV, whose 4.8 TB/s floor is ~137 us. The kernel the spec would replace is therefore already the access pattern the spec says trtllm fixes. The spec still feeds trtllm an unpaired FP8 [T,16,64] query, so at batch>1 it races an unpaired kernel against paired FA3. Headroom is real but modest. If attention fell from 290 to ~200 us at decode_32 and from 152 to ~100 us at decode_16, the decode_32 load would gain ~7%, the decode_16 load ~4.5%, and the 6-load geomean ~1.9%. Full decode_1 has 18.8 us of FA3 over ~20 MB of KV (floor ~4.3 us), worth roughly 1-2% on that load. The expected directions agree with TASK's weights (full shapes weight 16, sliding and verify are clean controls), but the decode_16/32 'faster' entries are measured against the wrong incumbent. Weakest link: feasibility on this hardware. flashinfer.decode.trtllm_batch_decode_with_kv_cache is the TRT-LLM-gen FMHA path, and its cubins target Blackwell (SM100-class). On SM90 only the XQA backend of newer FlashInfer can serve it, and XQA's supported page sizes (typically >= 16 tokens) and head dims (128 is the common one) probably exclude page size 1 with head_dim 64. Nobody can verify this here: there is no GPU, and FlashInfer is not installed locally. Verifying in the image means spending the decode problem's last check, and the likely result is an import, support or JIT failure rather than a timing. This lineage also has a track record: a hand-written Triton paired-row decode ran 5-7% behind unpaired FA3 and further behind paired FA3, and the stage-2 rewrite was paused for the same reason ('no stage 3 unless a new design beats paired FA3'). The prior that a generic kernel beats paired FA3 at page size 1 is weak.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched: direct conflict. It owns the exact call this hypothesis replaces (full layers, batch>1). Swapping in an unpaired trtllm call removes the 7-13% pairing gain on full decode_16/32. A loss there could just be pairing lost, and a gain would understate trtllm against an unpaired FA3. The spec does not say whether the `paired` gate is kept, bypassed or ported to trtllm (feeding trtllm the 128-wide padded query over the [pages,1,1,128] view), so neither outcome is interpretable as written.
- fused-preattention-triton (in the judge, not landed): it rewrites the q production that feeds the attention call. If it lands first, it becomes this hypothesis's base and changes what q looks like (dtype, layout, possibly a padded paired query). The spec would have to be rebuilt on it.
- marlin-block-m-cap-16, sglang-pipeline-parity-router-hilo-m8: no confound. They touch the MoE and router after attention, and their kernels are identical on every shape this change moves.

### Interference and A/B needs
No companion fixes this. The problem is not attribution between this change and a sub-change. The spec's base and its incumbent are both out of date, and its feasibility is unknown. An ab-revert of fa3-hdim128-head-pairing-batched would compare trtllm against the old unpaired FA3, which answers a question master no longer asks. With one decode check left, the only sensible next step is a re-mint built on current master (after fused-preattention-triton harvests). The re-mint should state explicitly: (1) the gate (full and q_len==1, with batch>1 also replacing the paired path, or only batch==1, where FA3 is unpaired and 18.8 us); (2) which FlashInfer backend runs on SM90 and its proven support for page_size=1 and head_dim=64, or a paired 128-wide head_dim variant over the existing [pages,1,1,128] view; (3) that cubins come from the image with no runtime download. A batch==1-only variant is the cheapest and cleanest test, because it never competes with pairing and the batch>1 shapes become controls.

### Open questions
- Does the lmsysorg/sglang:dev-muse-spark-open image's FlashInfer expose an SM90 backend for trtllm_batch_decode_with_kv_cache (XQA), and does that backend accept page_size=1, head_dim=64, GQA 8 and an FP8 e4m3 KV cache?
- Are the trtllm-gen or XQA cubins bundled in the image, or fetched over the network at first use?
- Should the re-mint target batch==1 only (unpaired FA3, 18.8 us over a ~4.3 us floor), keep paired FA3 at batch>1, or port the pairing to trtllm with head_dim 128?
- Does fused-preattention-triton land first, and if so what q layout does it hand to the attention call?

