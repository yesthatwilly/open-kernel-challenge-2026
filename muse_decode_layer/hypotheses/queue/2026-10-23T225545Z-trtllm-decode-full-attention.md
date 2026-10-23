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
review: required
status: queued
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
