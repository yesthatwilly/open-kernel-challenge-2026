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
review: proceed
status: running
review_at: 2026-10-24T165606Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T165606Z
base_sha: f3620a153798c4af3feb47a7fc26977635b8a6f3
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/full-decode-cuda-splitkv-port
worktree: .worktrees/muse_decode_layer/full-decode-cuda-splitkv-port
mode: null
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

## Adversarial review

Reviewed 2026-10-24T165606Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite. It ports a kernel the judge has already accepted and that passed its self-check (attention 82085a4e82154393: compiled in 41.0 s, rel L2 1.6-1.7e-3), against the same KV geometry (decode_16 full floor 63 us in both problems), and it replaces ~165/~305 us of paired-FA3 work at decode_16/32. Judge it on top_kernels, not the headline: the expected headline move is only ~+1.4%, which is inside run drift.

### Justification
The mechanism is plausible. Full-layer decode at page size 1 is limited by the row gather (LESSONS: 1.3-1.7 TB/s). In muse_attention_decode the hand-written CUDA split-KV kernel beat paired FA3 on this same gather (decode_16 156.8 -> 123.0 us shape, kernel A 111.4 + B 7.9 us). The KV geometry is identical: 16 q heads, 2 KV heads, d 64, page 1, ctx 80k, spread agentic. The attention floor for full decode_16 is 63.2 us, the figure the hypothesis cites. The kernel does not depend on the data, so dist soft vs coherent only changes values, not bytes. In decode_layer's profile, paired FA3 main is 146.8 us (38%) at decode_16 and 289.0 us (52.4%) at decode_32, about what attention's paired FA3 took. That makes the incumbent comparable, and the ~40-55 us saving per full shape is credible. It removes FA3 main, FlashAttnFwdCombine, the paired zero-fill, the pad and unpad copies, the bf16 cast and FusedRMSNormGate, and adds two launches. Load arithmetic from TASK weights: decode_16 load = 46*225.8 + 16*364.9 = 16225 us, and saving ~45 us x16 gives ~4.4%. decode_32 load = 46*251.9 + 16*535.8 = 20160 us, and ~50 us x16 gives ~4.0%. The other four loads are flat, so the geomean is ~+1.4%. The per-shape and per-load expectations are consistent; the headline is below the ~1.5-2.5% drift (LESSONS). Weakest link: it assumes kernel A's attention-measured time carries over unchanged into the layer. In the layer it runs right after the qkv GEMM and FusedQKNorm on a cold L2 (irrelevant at 327 MB of KV). It also runs with 2 CTAs/SM of ~64 KB+ smem, which matches attention's setup. A smaller risk is that the decode-specific self-check is newly written rather than copied: it must build data['config'] with sm_scale, qk_scale_factor, batch, num_q_heads and head_dim for _full_decode_splitkv, and it must reset _SPLITKV_GRID to None in a finally. Run time: the decode run took ~412 s; adding 41 s of compile plus the self-check stays well under the 15-minute limit.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched (landed): this is the incumbent being replaced on full decode_16/32. If it were absent, the baseline would be slower unpaired FA3 and the measured gain larger. It is not a confounder for attribution, because the change replaces that path outright on the target shapes, and the paired full verify_16/32 rows keep it as an in-run control.
- post-expert-norm-triton-small-t (landed): it touches T<32 (decode_16), but in the MoE tail after the o GEMM, so the kernels and stream position are disjoint. It does not confound the attention top_kernels rows; it only shifts the decode_16 shape total used to compute the percentage.
- marlin-block-m-cap-16 and sglang-pipeline-parity-router-hilo-m8 (landed): disjoint areas (Marlin block_m does not change at T<=32, router on the side stream after attention). No interaction.
- paired-heads-pad-unpad-triton (in flight, not landed): if it lands first, master's paired glue on full decode_16/32 shrinks by ~10 us. This change's apparent saving against the new master then falls to ~30-45 us. The kernel's own time against FA3 main stays readable either way. A merge must keep `paired = ... and not splitkv`.

### Interference and A/B needs
No companions. This is kind: rewrite, and it is judged on the replaced kernels' own time per shape (splitkv_attn_kernel + splitkv_combine_kernel against FA3 main + combine + pad/unpad + RMSNormGate in top_kernels) against the 63/126 us gather floors. That attribution is direct, and an A/B revert of head pairing would only re-measure a known FA3 variant. The in-run controls are the untouched shapes: every sliding row, full decode_1, and the paired full verify_16/32 rows, which keep the exact incumbent FA3 path. At harvest, check the log tail first: '[splitkv] load_inline failed' or a self-check failure means the run measured the master fallback (DECISIONS 2026-10-24T141733Z).

### Open questions
- Does FusedQKNorm's q come out row-contiguous so that q.view(tokens, hq, d).to(FP8) satisfies the kernel's q.is_contiguous() TORCH_CHECK? (.to() makes a contiguous copy, so this should hold.) Is the k_cache/v_cache from the decode judge contiguous [pages,1,2,64] as the TORCH_CHECK requires?
- The gated tests are 'batch 33, ctx 20000, full, spread mixed, page 1' and 'batch 16, ctx 20000, full, dist soft, page 1'. Both are within 1 < batch <= 64, and they are the correctness gate for this path. Should the self-check also cover batch 33 with a non-power-of-two chunk count?
- If prepare() can be called more than once per process, is the self-check guarded so it runs once, and is _EXT set to None globally on failure before any graph capture?

