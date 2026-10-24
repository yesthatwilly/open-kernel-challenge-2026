---
problem: muse_prefill_layer
slug: attn-full-prefill-cuda-fp8-exp-offload
created: 2026-10-24T172146Z
base: master
area: "FA3 call (CUDA sm_90a FP8 prefill flash attention with software-exp offload, stage 1: full layers, page size 1)"
change: >
  Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill
  layers with a hand-written sm_90a FP8 flash-attention forward kernel in CUDA
  (torch.utils.cpp_extension.load_inline at import, as attention's full-decode-cuda-gather-splitkv
  did in 41 s), designed around the one thing that bounds FA3 here: the softmax pipe. WHY: after
  the experts, FA3 is the largest kernel on all five full shapes (415.2 us turn_1 full, 1634.0
  turn_4 full, 892.2 first_turn ctx0, 4471.1 ctx32768 full, 8006.7 ctx65536 full; 24-37% of the
  three big full first_turn shapes) and it does not respond to occupancy (LESSONS: ~600 TFLOP/s on
  every full prefill shape whatever the CTA count; fa3-batched-mode held for that reason). The
  evidence is that it is bound by exponentials: score elements per shape are T x keys x 16 heads =
  0.97G, 3.87G, 2.15G, 10.7G and 19.3G, which at sm_90's 16 ex2 per clock per SM (132 SMs, ~1.83
  GHz, 3.87 T/s) take 250, 1000, 555, 2770 and 4990 us; FA3 measures 1.60-1.66x that on every one
  of the five shapes, while its FP8 MMA work (4 x 64 FLOP per element) would take only ~0.5x of it
  at FP8 tensor rate. With head_dim 64 there are only 256 MMA FLOP per exponential, so the tensor
  cores idle on MUFU and the surrounding FP32 softmax work. No amount of rescheduling FA3 changes
  that ratio; only moving exponentials off MUFU does. KERNEL DESIGN: persistent grid (132 CTAs),
  static work list from the config only (batch, q_len, heads: shapes, never values), longest-first
  so the causal tail balances; per-item key counts read on the device from
  cache_seqlens/cu_seqlens_q (no host sync, graph-safe). Item = 128 query rows of one KV head: 16
  consecutive tokens x the 8 q heads that share it (GQA-packed, causal mask per token, so each
  gathered 64-byte K/V row feeds 8 heads), by default 2 consumer warpgroups x 64 rows (try 3 x 64
  if registers allow) and 1 producer warpgroup. Producer: page_table lookup and cp.async 16 B
  gather of page-size-1 K and V rows (TMA cannot gather rows on sm_90) into a 2-3 stage K/V ring
  of 128-176 keys, mbarrier-signalled; V transposed in smem to K-major for the FP8 PV MMA
  (ldmatrix + byte prmt, as FA3's FP8 path does). Consumers: S = Q K^T by wgmma m64nNk32 e4m3 (Q
  FP8 in smem, queries rounded to E4M3 exactly as FA3 does today and no coarser), online softmax
  in fp32 registers with the scale folded into one FFMA, P converted to FP8 E4M3 in registers
  (FA3's own P precision) and permuted from the accumulator to the A-fragment layout, O += P V by
  wgmma RS, intra-warpgroup overlap of softmax(j) with QK^T(j+1) and inter-warpgroup ping-pong on
  named barriers. The new idea: a constexpr fraction of each row's exponentials (start at 1 in 3
  columns, tune toward the MUFU/FMA balance point, ~0.7 MUFU) is computed on the FMA pipe by
  Cody-Waite range reduction plus a degree-3 minimax polynomial for 2^frac and an integer add into
  the exponent (the FlashAttention-4 software-exp trick; optionally as HFMA2 pairs, since P is
  rounded to E4M3 anyway). Every key is still attended with its exact softmax weight (no skipping,
  no block sparsity, the whole context on full layers). Epilogue: O / l to bf16 [T, 16, 64], the
  same tensor FA3 returns, so o_norm_gate and everything after are unchanged. STAGE PLAN: stage 1
  (this file) = full layers only (window == 0), page_size == 1 and kv_scale == 1: the five scored
  full shapes and the batch 1 ctx 3000 full test; everything else (sliding, page_size 64, the
  scaled-cache test) keeps FA3 through a shape/config gate in kernel_fn. Stage 2 = fuse
  FusedRMSNormGate (per-head rmsnorm x sigmoid(gate), one 64-wide head row per thread quad) into
  the epilogue and take sliding layers too (window band: only the first and last tiles of each
  row's 2048-key band are masked; sliding first_turn FA3 is 304 us at weight 138). Stage 3 = take
  the query side in (QK-norm + RoPE + E4M3 cast in the prologue, dropping the q.to(FP8) launch),
  and the kv-scaled and page-64 cases. FALLBACK AND SELF-CHECK: module-level try around
  load_inline; any exception sets CUDA_FULL_ATTN_OK = False and kernel_fn calls FA3 exactly as
  master does. In prepare(): (a) a unit test of the software exp against exp2f over [-126, 0] (max
  rel err <= 2^-11) and (b) the kernel against flash_attn_with_kvcache on synthetic FP8 caches
  with a randomly permuted page-size-1 page table: batch 1 ctx 3000 L 753; batch 4 with seq_lens
  {0, 777, 5000, 80000} L 753; batch 1 ctx 0 L 16384; one length not a multiple of 16 or 128. Pass
  = rel L2 <= 1e-2 and worst row <= 5e-2 vs FA3, no NaN/Inf; else the flag goes off. Log one line
  "[cuda full attn] ON" or "[cuda full attn] OFF: <reason>", with compile seconds, registers and
  smem. MILESTONES (profile top_kernels: the new kernel's row replaces FA3's
  cutlass::device_kernel<flash::...> row on full shapes): M0 log says ON and the 5 tests pass; M1
  the kernel beats FA3 on every full shape (turn_4 full under 1634 us, ctx65536 under 8007 us); M2
  0.75x FA3 (turn_1 full ~310, turn_4 full ~1225, ctx0 ~670, ctx32768 ~3350, ctx65536 ~6000 us),
  i.e. ~1.2x the MUFU-only time; stretch 0.65x.
expected:
  - shapes: "*layer=full*load=turn_1*"
    direction: faster
    why: "Replaced: FA3 415.2 us + combine 7.0 us of 1419 us. Floor is the softmax exponentials, 0.97G at 3.87 T/s = 250 us (the shape floor is 496.7 us); at M2 ~310 us the shape gains ~7-8%, ~1.5% on the turn_1 load."
  - shapes: "*layer=full*load=turn_4*"
    direction: faster
    why: "Replaced: FA3 1634 us of 4938 us; exp floor 1000 us; M2 ~1225 us gives ~8% on the shape, ~1.8% on the turn_4 load."
  - shapes: "*layer=full*load=first_turn*"
    direction: faster
    why: "Replaced: FA3 892 (ctx0), 4471 (ctx32768), 8007 us (ctx65536); exp floors 555/2770/4990 us; M2 ~670/3350/6000 us gives 1.5/6/9% on those shapes, ~2.3% on the first_turn load (they carry weight 16 each beside the sliding shape's 138). Headline about +2% on master (1.059 -> ~1.08); once the expert rewrite lands the same kernel is worth ~4-5%, since FA3 then becomes the largest kernel on every full shape."
  - shapes: "*layer=sliding*"
    direction: neutral
    why: stage 1 keeps FA3 on sliding layers (25.7 / 61.4 / 304.4 us there)
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1"
review: required
status: queued
---

# attn-full-prefill-cuda-fp8-exp-offload

## Change

Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill layers with a hand-written sm_90a FP8 flash-attention forward kernel in CUDA (torch.utils.cpp_extension.load_inline at import, as attention's full-decode-cuda-gather-splitkv did in 41 s), designed around the one thing that bounds FA3 here: the softmax pipe.
WHY: after the experts, FA3 is the largest kernel on all five full shapes (415.2 us turn_1 full, 1634.0 turn_4 full, 892.2 first_turn ctx0, 4471.1 ctx32768 full, 8006.7 ctx65536 full; 24-37% of the three big full first_turn shapes) and it does not respond to occupancy (LESSONS: ~600 TFLOP/s on every full prefill shape whatever the CTA count; fa3-batched-mode held for that reason). The evidence is that it is bound by exponentials: score elements per shape are T x keys x 16 heads = 0.97G, 3.87G, 2.15G, 10.7G and 19.3G, which at sm_90's 16 ex2 per clock per SM (132 SMs, ~1.83 GHz, 3.87 T/s) take 250, 1000, 555, 2770 and 4990 us; FA3 measures 1.60-1.66x that on every one of the five shapes, while its FP8 MMA work (4 x 64 FLOP per element) would take only ~0.5x of it at FP8 tensor rate. With head_dim 64 there are only 256 MMA FLOP per exponential, so the tensor cores idle on MUFU and the surrounding FP32 softmax work. No amount of rescheduling FA3 changes that ratio; only moving exponentials off MUFU does.
KERNEL DESIGN: persistent grid (132 CTAs), static work list from the config only (batch, q_len, heads: shapes, never values), longest-first so the causal tail balances; per-item key counts read on the device from cache_seqlens/cu_seqlens_q (no host sync, graph-safe). Item = 128 query rows of one KV head: 16 consecutive tokens x the 8 q heads that share it (GQA-packed, causal mask per token, so each gathered 64-byte K/V row feeds 8 heads), by default 2 consumer warpgroups x 64 rows (try 3 x 64 if registers allow) and 1 producer warpgroup. Producer: page_table lookup and cp.async 16 B gather of page-size-1 K and V rows (TMA cannot gather rows on sm_90) into a 2-3 stage K/V ring of 128-176 keys, mbarrier-signalled; V transposed in smem to K-major for the FP8 PV MMA (ldmatrix + byte prmt, as FA3's FP8 path does). Consumers: S = Q K^T by wgmma m64nNk32 e4m3 (Q FP8 in smem, queries rounded to E4M3 exactly as FA3 does today and no coarser), online softmax in fp32 registers with the scale folded into one FFMA, P converted to FP8 E4M3 in registers (FA3's own P precision) and permuted from the accumulator to the A-fragment layout, O += P V by wgmma RS, intra-warpgroup overlap of softmax(j) with QK^T(j+1) and inter-warpgroup ping-pong on named barriers. The new idea: a constexpr fraction of each row's exponentials (start at 1 in 3 columns, tune toward the MUFU/FMA balance point, ~0.7 MUFU) is computed on the FMA pipe by Cody-Waite range reduction plus a degree-3 minimax polynomial for 2^frac and an integer add into the exponent (the FlashAttention-4 software-exp trick; optionally as HFMA2 pairs, since P is rounded to E4M3 anyway). Every key is still attended with its exact softmax weight (no skipping, no block sparsity, the whole context on full layers). Epilogue: O / l to bf16 [T, 16, 64], the same tensor FA3 returns, so o_norm_gate and everything after are unchanged.
STAGE PLAN: stage 1 (this file) = full layers only (window == 0), page_size == 1 and kv_scale == 1: the five scored full shapes and the batch 1 ctx 3000 full test; everything else (sliding, page_size 64, the scaled-cache test) keeps FA3 through a shape/config gate in kernel_fn. Stage 2 = fuse FusedRMSNormGate (per-head rmsnorm x sigmoid(gate), one 64-wide head row per thread quad) into the epilogue and take sliding layers too (window band: only the first and last tiles of each row's 2048-key band are masked; sliding first_turn FA3 is 304 us at weight 138). Stage 3 = take the query side in (QK-norm + RoPE + E4M3 cast in the prologue, dropping the q.to(FP8) launch), and the kv-scaled and page-64 cases.
FALLBACK AND SELF-CHECK: module-level try around load_inline; any exception sets CUDA_FULL_ATTN_OK = False and kernel_fn calls FA3 exactly as master does. In prepare(): (a) a unit test of the software exp against exp2f over [-126, 0] (max rel err <= 2^-11) and (b) the kernel against flash_attn_with_kvcache on synthetic FP8 caches with a randomly permuted page-size-1 page table: batch 1 ctx 3000 L 753; batch 4 with seq_lens {0, 777, 5000, 80000} L 753; batch 1 ctx 0 L 16384; one length not a multiple of 16 or 128. Pass = rel L2 <= 1e-2 and worst row <= 5e-2 vs FA3, no NaN/Inf; else the flag goes off. Log one line "[cuda full attn] ON" or "[cuda full attn] OFF: <reason>", with compile seconds, registers and smem.
MILESTONES (profile top_kernels: the new kernel's row replaces FA3's cutlass::device_kernel<flash::...> row on full shapes): M0 log says ON and the 5 tests pass; M1 the kernel beats FA3 on every full shape (turn_4 full under 1634 us, ctx65536 under 8007 us); M2 0.75x FA3 (turn_1 full ~310, turn_4 full ~1225, ctx0 ~670, ctx32768 ~3350, ctx65536 ~6000 us), i.e. ~1.2x the MUFU-only time; stretch 0.65x.

## Expected

- `*layer=full*load=turn_1*` faster: Replaced: FA3 415.2 us + combine 7.0 us of 1419 us. Floor is the softmax exponentials, 0.97G at 3.87 T/s = 250 us (the shape floor is 496.7 us); at M2 ~310 us the shape gains ~7-8%, ~1.5% on the turn_1 load.
- `*layer=full*load=turn_4*` faster: Replaced: FA3 1634 us of 4938 us; exp floor 1000 us; M2 ~1225 us gives ~8% on the shape, ~1.8% on the turn_4 load.
- `*layer=full*load=first_turn*` faster: Replaced: FA3 892 (ctx0), 4471 (ctx32768), 8007 us (ctx65536); exp floors 555/2770/4990 us; M2 ~670/3350/6000 us gives 1.5/6/9% on those shapes, ~2.3% on the first_turn load (they carry weight 16 each beside the sliding shape's 138). Headline about +2% on master (1.059 -> ~1.08); once the expert rewrite lands the same kernel is worth ~4-5%, since FA3 then becomes the largest kernel on every full shape.
- `*layer=sliding*` neutral: stage 1 keeps FA3 on sliding layers (25.7 / 61.4 / 304.4 us there)

## Risk

Biggest risk: FA3 is a mature, tuned kernel, and beating it needs FA3-class engineering written blind (FP8 V transpose in smem, accumulator-to-A-fragment byte permutes for P, warp specialization with page-size-1 cp.async gathers). The exp-offload gain can be eaten by FP32 issue-slot pressure: Hopper has no FFMA2, so the polynomial costs ~6-7 issue slots per offloaded element, and the balance point must be tuned. If the kernel lands at FA3 parity, stage 1 still pays as the base for stage 2's norm-gate fusion and stage 3's query fusion. Independent of the expert rewrite: no shared code, so it can run in parallel and be restacked by a 3-way merge.

## Milestones

Rewrite stage 1 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
