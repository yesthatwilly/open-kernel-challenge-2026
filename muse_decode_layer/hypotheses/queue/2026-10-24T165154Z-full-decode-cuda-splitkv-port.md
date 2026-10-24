---
problem: muse_decode_layer
slug: full-decode-cuda-splitkv-port
created: 2026-10-24T165154Z
base: master
area: FA3 call
change: >
  Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from
  muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e82154393: full
  decode_16 156.8 -> 123.0 us, decode_32 298.5 -> 257.8 us; attention 1.293x -> 1.347x). Copy
  verbatim from muse_attention_decode/submission.py on master: LOG2E, _SPLITKV_CHUNK,
  _SPLITKV_CPP, _SPLITKV_CUDA (splitkv_attn_kernel = persistent 2xSM grid over 1024-token chunks,
  page indices prefetched to smem, 4-stage cp.async ring of 128-byte K/V rows, fp8->f16 in
  registers, mma.sync m16n8k16, online softmax; splitkv_combine_kernel = LSE merge x v_scale +
  scaleless RMSNorm(64, eps 1e-5) x sigmoid(gate read from qkv at column (Hq+2Hkv)*64 + h*64 with
  qkv.stride(0))), _build_splitkv (load_inline, -O3 -arch=sm_90a, module name
  muse_full_decode_splitkv_v1), _EXT, _SPLITKV_GRID, _SMS/_sm_count, _splitkv_gate and
  _full_decode_splitkv. Compile at import exactly as attention does (guarded by
  torch.cuda.is_available(); on failure _EXT stays None and master path runs everywhere).
  Self-check: attention calls its own kernel_fn on synthetic data; decode cannot (kernel_fn is a
  whole layer), so add _splitkv_self_check() run once in prepare() that builds synthetic inputs
  directly (FP8 q [B,16,64], random FP8 k_cache/v_cache [pages,1,2,64], a shuffled page_table with
  ragged lengths [9000, 1025, 70, 4097, 1023], cache_seqlens int32, k_scale/v_scale 1.0 and 0.25,
  a bf16 qkv [B,2304] supplying the gate) and compares _full_decode_splitkv against an FP32
  reference (gather rows, softmax(q k^T * sm_scale*qk_scale_factor*k_scale), @ v*v_scale, rms-norm
  over 64 eps 1e-5, x sigmoid(gate)) at the device grid and at _SPLITKV_GRID=3; rel L2 < 2e-2 and
  finite or _EXT = None. In decode kernel_fn: compute splitkv = _splitkv_gate(cfg) (config only:
  window 0, q_len 1, 1 < batch <= 64, hq 16, hkv 2, d 64, page_size 1); make paired = window == 0
  and batch > 1 and hkv % 2 == 0 and not splitkv. On the splitkv branch keep master qkv GEMM,
  FusedQKNorm (no RoPE on full layers), the FP8 casts and store_cache byte for byte, take q_fp8 =
  q.view(tokens, hq, d).to(FP8) (contiguous), and replace flash_attn_with_kvcache plus the paired
  pad/unpad copies plus state["o_norm_gate"] with o = _full_decode_splitkv(data, q_fp8,
  qkv).reshape(tokens, hq * d), feeding the unchanged o GEMM. scheduler_metadata is ignored on
  this path. Every other shape (all sliding, full decode_1, every verify block including the
  paired full verify_16/32) keeps master code byte for byte.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "incumbent paired FA3 main 146.8 us + FlashAttnFwdCombine 4.3 + 5 pad/unpad elementwise copies (6.2 + 4.4 us) + FA3 prepare + FusedRMSNormGate, ~165 us of 365 us, floor for the KV gather ~63 us; attention measured kernel A 111.4 + kernel B 7.9 us for the same KV at dist=coherent, so ~40-45 us off: ~320 us, +12-14% on the shape, ~+4% on the decode_16 load"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "incumbent FA3 main 289.0 us + combine + glue ~305 us of 536 us; attention kernel A 245.5 + B 8.6 us: ~45-55 us off, ~485 us, +9-11% on the shape, ~+4% on the decode_32 load"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: batch 1 outside the gate, unpaired FA3 kept
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 outside the gate; paired FA3 kept (stage 2 extends to verify)
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched control
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32
review: required
status: queued
---

# full-decode-cuda-splitkv-port

## Change

Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e82154393: full decode_16 156.8 -> 123.0 us, decode_32 298.5 -> 257.8 us; attention 1.293x -> 1.347x). Copy verbatim from muse_attention_decode/submission.py on master: LOG2E, _SPLITKV_CHUNK, _SPLITKV_CPP, _SPLITKV_CUDA (splitkv_attn_kernel = persistent 2xSM grid over 1024-token chunks, page indices prefetched to smem, 4-stage cp.async ring of 128-byte K/V rows, fp8->f16 in registers, mma.sync m16n8k16, online softmax; splitkv_combine_kernel = LSE merge x v_scale + scaleless RMSNorm(64, eps 1e-5) x sigmoid(gate read from qkv at column (Hq+2Hkv)*64 + h*64 with qkv.stride(0))), _build_splitkv (load_inline, -O3 -arch=sm_90a, module name muse_full_decode_splitkv_v1), _EXT, _SPLITKV_GRID, _SMS/_sm_count, _splitkv_gate and _full_decode_splitkv. Compile at import exactly as attention does (guarded by torch.cuda.is_available(); on failure _EXT stays None and master path runs everywhere). Self-check: attention calls its own kernel_fn on synthetic data; decode cannot (kernel_fn is a whole layer), so add _splitkv_self_check() run once in prepare() that builds synthetic inputs directly (FP8 q [B,16,64], random FP8 k_cache/v_cache [pages,1,2,64], a shuffled page_table with ragged lengths [9000, 1025, 70, 4097, 1023], cache_seqlens int32, k_scale/v_scale 1.0 and 0.25, a bf16 qkv [B,2304] supplying the gate) and compares _full_decode_splitkv against an FP32 reference (gather rows, softmax(q k^T * sm_scale*qk_scale_factor*k_scale), @ v*v_scale, rms-norm over 64 eps 1e-5, x sigmoid(gate)) at the device grid and at _SPLITKV_GRID=3; rel L2 < 2e-2 and finite or _EXT = None. In decode kernel_fn: compute splitkv = _splitkv_gate(cfg) (config only: window 0, q_len 1, 1 < batch <= 64, hq 16, hkv 2, d 64, page_size 1); make paired = window == 0 and batch > 1 and hkv % 2 == 0 and not splitkv. On the splitkv branch keep master qkv GEMM, FusedQKNorm (no RoPE on full layers), the FP8 casts and store_cache byte for byte, take q_fp8 = q.view(tokens, hq, d).to(FP8) (contiguous), and replace flash_attn_with_kvcache plus the paired pad/unpad copies plus state["o_norm_gate"] with o = _full_decode_splitkv(data, q_fp8, qkv).reshape(tokens, hq * d), feeding the unchanged o GEMM. scheduler_metadata is ignored on this path. Every other shape (all sliding, full decode_1, every verify block including the paired full verify_16/32) keeps master code byte for byte.

## Expected

- `*layer=full load=decode_16 *` faster: incumbent paired FA3 main 146.8 us + FlashAttnFwdCombine 4.3 + 5 pad/unpad elementwise copies (6.2 + 4.4 us) + FA3 prepare + FusedRMSNormGate, ~165 us of 365 us, floor for the KV gather ~63 us; attention measured kernel A 111.4 + kernel B 7.9 us for the same KV at dist=coherent, so ~40-45 us off: ~320 us, +12-14% on the shape, ~+4% on the decode_16 load
- `*layer=full load=decode_32 *` faster: incumbent FA3 main 289.0 us + combine + glue ~305 us of 536 us; attention kernel A 245.5 + B 8.6 us: ~45-55 us off, ~485 us, +9-11% on the shape, ~+4% on the decode_32 load
- `*layer=full load=decode_1 *` flat: batch 1 outside the gate, unpaired FA3 kept
- `*load=verify_*` flat: q_len 8 outside the gate; paired FA3 kept (stage 2 extends to verify)
- `*layer=sliding*` flat: untouched control

## Risk

High: 41 s load_inline compile at import (the judge accepted it in attention), a decode-specific self-check that must be written rather than copied, and decode dist=soft vs attention dist=coherent (the kernel is data-independent, but per-shape lengths differ). Overlaps paired-heads-pad-unpad-triton (running) on full decode_16/32 only: that change still applies to full verify_16/32; at decode_16/32 whichever lands second supersedes the paired glue there. Verify is untouched, so the verify rows are the in-run control. Do not stack on triton-flash-decode-full-* (paused rewrite).

## Milestones

Rewrite stage 1 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
