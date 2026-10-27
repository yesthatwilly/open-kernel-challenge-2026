---
problem: muse_attention_decode
slug: sliding-verify-wgmma-window-s2-split-combine
created: 2026-10-26T222100Z
base: master
area: sliding-layer attention (verify B=2..32, wgmma window kernel merge)
change: >
  Stage 2 on the stage-1 branch (hyp/muse_attention_decode/sliding-verify-wgmma-window-s1, job
  cd01da3909cc4f4c), whatever its score. s1's own gate at B=16 x 80K timed
  slidewin_verify_wgmma_kernel WITHOUT its in-kernel ticket merge at 14.9-16.1 us (grids 128 / 132
  / 264, 'kernel only (no merge)', partials written) against master's whole routed path (pre-attn
  + prepare + FA3 + combine + norm-gate) at 24.4 us net, but with the merge at 40-97 us: the
  printed 'merge tail' was 14.9 / 17.6 / 24.0 / 32.0 / 81.7 us at grids 32 / 64 / 128 / 132 / 264,
  so no grid cleared the 1.05x gate. CHANGE: take the in-kernel ticket out of the routed path.
  Route slidewin_verify_wgmma_kernel<false> (TICKET = false: every segment stores its fp32 partial
  at slot bx + cb; no fence, atomic or merge) followed by ONE new kernel in the same module
  (rename muse_sliding_verify_wgmma_v1 -> _v2): sw_combine_kernel, grid (B * QL, HKV), 128
  threads, one CTA per (request b, verify token j, kv head g); thread t serves query head h = g *
  G + t / 16, dims d0 = (t % 16) * 4 .. +3 (float4). Factor the kernel prologue's per-request key
  counts (seq_s, window-limited exactly as now) and tile_prefix into a __device__ helper shared by
  both kernels, so the combine recomputes pre_s, T and, with the launch grid gx passed in, c_lo =
  owner_of(pre[b]), c_hi = owner_of(pre[b+1] - 1). Merge slots c + b for c in [c_lo, c_hi] whose
  CTA range overlaps the request (start_of computed incrementally: one int64 division before the
  loop) in ONE pass: loads for 4 slots at a time (m and l at row j * HQ + h, float4 of acc) issued
  before use, online LSE (running max M, rescale L and o by ex2(M_old - M_new); a slot with m =
  -inf weighs 0). Then merge_head's epilogue exactly: o / L * v_scale, scaleless RMSNorm over the
  head's 64 dims (16-lane xor shuffle, eps inside the sqrt), times sigmoid(gate) from qkv row b *
  QL + j, bf16 into out [B * QL, HQ * 64]. Host: _sw_forward and _sw_forward_grid launch
  kernel<false> then sw_combine_kernel on the same stream (plain launches, workspace and grid from
  shapes only). Gate and checks unchanged: s1's eager correctness runs vs fp32 and vs master per
  grid, two-state capture-once replay, quick pass then two final rounds needing >= 1.05x vs master
  in both with the IQR bound, PATH CHOSEN with both numbers; add the combine's own time to the
  quick-pass line ('kernel only', 'combine', 'pre-attn + kernel + combine'). Bands B=2..16 (timed
  at 16) and B=17..32 (timed at 32) as in s1; B=1 stays on the PSEUDO route. RULES as s1:
  graph-capturable, no host sync or input-sized allocation in kernel_fn, nothing cached from
  inputs, KV FP8 / q E4M3 / fp32 merges / bf16 out, every key in the window, two states rotated
  (tickets / slots are workspace, never module-global device pointers), TORCH_CUDA_ARCH_LIST=9.0a
  with -std=c++20, sync + print after each self-check launch.
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: "s1 measured kernel-only 14.9-16.1 us vs master's path 24.4 us net at B=16: pre-attn ~3 + kernel ~15-16 + combine ~2-3 us = ~21-22 us (1.1-1.16x); shape ~22.8 -> ~20 us at 0.21%/us"
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "same mechanism in band B=17..32 (s1's B=32 gate lines were cut from the 15 KB log tail); ~29 us shape at 0.124%/us"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: flat
    why: B=1 stays on the PSEUDO route
  - shapes: "*layer=full *"
    direction: flat
    why: full layers untouched
risk: medium
depends_on: sliding-verify-wgmma-window-s1
priority: 1
kind: rewrite
controls_for: null
prior: "sliding-verify-wgmma-window-s1: s1 routed nothing because its in-kernel ticket merge tail (14.9-81.7 us by grid) dwarfed the 15-16 us kernel; this stage leaves the kernel as is and replaces only the merge, with a TICKET=false launch plus one parallel one-pass combine kernel, so the tail cost goes away"
rewrite_stage: 2
replaces: "sliding verify (q_len 8, page 1, tp8) at 2<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate"
review: required
status: hold
hold_reason: "superseded 2026-10-27: s2 was checked and killed by the campaign; its successor s2b landed on master (scored a76ee9b0, then sw17 4e386647 1.6340)"
---

# sliding-verify-wgmma-window-s2-split-combine

## Change

Stage 2 on the stage-1 branch (hyp/muse_attention_decode/sliding-verify-wgmma-window-s1, job cd01da3909cc4f4c), whatever its score. s1's own gate at B=16 x 80K timed slidewin_verify_wgmma_kernel WITHOUT its in-kernel ticket merge at 14.9-16.1 us (grids 128 / 132 / 264, 'kernel only (no merge)', partials written) against master's whole routed path (pre-attn + prepare + FA3 + combine + norm-gate) at 24.4 us net, but with the merge at 40-97 us: the printed 'merge tail' was 14.9 / 17.6 / 24.0 / 32.0 / 81.7 us at grids 32 / 64 / 128 / 132 / 264, so no grid cleared the 1.05x gate. CHANGE: take the in-kernel ticket out of the routed path. Route slidewin_verify_wgmma_kernel<false> (TICKET = false: every segment stores its fp32 partial at slot bx + cb; no fence, atomic or merge) followed by ONE new kernel in the same module (rename muse_sliding_verify_wgmma_v1 -> _v2): sw_combine_kernel, grid (B * QL, HKV), 128 threads, one CTA per (request b, verify token j, kv head g); thread t serves query head h = g * G + t / 16, dims d0 = (t % 16) * 4 .. +3 (float4). Factor the kernel prologue's per-request key counts (seq_s, window-limited exactly as now) and tile_prefix into a __device__ helper shared by both kernels, so the combine recomputes pre_s, T and, with the launch grid gx passed in, c_lo = owner_of(pre[b]), c_hi = owner_of(pre[b+1] - 1). Merge slots c + b for c in [c_lo, c_hi] whose CTA range overlaps the request (start_of computed incrementally: one int64 division before the loop) in ONE pass: loads for 4 slots at a time (m and l at row j * HQ + h, float4 of acc) issued before use, online LSE (running max M, rescale L and o by ex2(M_old - M_new); a slot with m = -inf weighs 0). Then merge_head's epilogue exactly: o / L * v_scale, scaleless RMSNorm over the head's 64 dims (16-lane xor shuffle, eps inside the sqrt), times sigmoid(gate) from qkv row b * QL + j, bf16 into out [B * QL, HQ * 64]. Host: _sw_forward and _sw_forward_grid launch kernel<false> then sw_combine_kernel on the same stream (plain launches, workspace and grid from shapes only). Gate and checks unchanged: s1's eager correctness runs vs fp32 and vs master per grid, two-state capture-once replay, quick pass then two final rounds needing >= 1.05x vs master in both with the IQR bound, PATH CHOSEN with both numbers; add the combine's own time to the quick-pass line ('kernel only', 'combine', 'pre-attn + kernel + combine'). Bands B=2..16 (timed at 16) and B=17..32 (timed at 32) as in s1; B=1 stays on the PSEUDO route. RULES as s1: graph-capturable, no host sync or input-sized allocation in kernel_fn, nothing cached from inputs, KV FP8 / q E4M3 / fp32 merges / bf16 out, every key in the window, two states rotated (tickets / slots are workspace, never module-global device pointers), TORCH_CUDA_ARCH_LIST=9.0a with -std=c++20, sync + print after each self-check launch.

## Expected

- `*layer=sliding load=verify_16 *` faster: s1 measured kernel-only 14.9-16.1 us vs master's path 24.4 us net at B=16: pre-attn ~3 + kernel ~15-16 + combine ~2-3 us = ~21-22 us (1.1-1.16x); shape ~22.8 -> ~20 us at 0.21%/us
- `*layer=sliding load=verify_32 *` faster: same mechanism in band B=17..32 (s1's B=32 gate lines were cut from the 15 KB log tail); ~29 us shape at 0.124%/us
- `*layer=sliding load=verify_1 *` flat: B=1 stays on the PSEUDO route
- `*layer=full *` flat: full layers untouched

## Risk

Built on s1's branch (per-kvhead master 5d528bf3 + the sw module), so the headline lacks v3chain's c320 chain (~1.52-1.53 even if it routes): read the SV16/SV32 rows, the PATH CHOSEN lines and the quick-pass 'combine' time. If it routes and wins, sliding-verify-wgmma-split-combine-on-v3chain is the board candidate. Risk: the combine must reproduce the kernel's window-limited tile prefix exactly (a mismatch merges wrong slots; the eager fp32 check and two-state replay catch it). Decode-portable: decode_layer's sliding verify_16/32 run the same FA3 chain.

## Milestones

Rewrite stage 2 of `sliding verify (q_len 8, page 1, tp8) at 2<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
