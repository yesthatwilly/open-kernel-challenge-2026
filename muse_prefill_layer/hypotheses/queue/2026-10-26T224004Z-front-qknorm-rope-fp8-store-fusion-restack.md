---
problem: muse_prefill_layer
slug: front-qknorm-rope-fp8-store-fusion-restack
created: 2026-10-26T224004Z
base: master
area: pre-attention norm+rope + cache store (five launches -> one Triton kernel)
change: >
  Built on master (blob f4751fc0). Port _qk_norm_fp8_store_kernel from
  done/neutral/qk-norm-fp8-cache-store-fusion.patch (job fe8883a2, which passed every judge test)
  onto master's kernel_fn. After the qkv GEMM master runs five launches: FusedQKNorm (the Triton
  _qk_rms_norm_fwd_kernel: 42.2 us at T=16384, 10.8 us at T=3012), k.to(FP8), v.to(FP8) (a strided
  generic elementwise kernel), store_cache and q.view(T, 16, 64).to(FP8). The fused kernel reads
  each token's qkv row once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 (eps
  1e-5, no weight), on sliding layers (rope_theta != 0) the interleaved RoPE from
  state['cos_sin'][positions], bf16 rounding then E4M3 for q (into q8 [T, 16, 64]) and k, v
  columns 1152..1279 cast to E4M3, and the k and v rows stored at cache_loc[t] of the FP8 caches
  (flat rows, so any page size). Use it only when config kv_scale == 1 (scaled configs keep
  master's code). q8 then feeds master's fa3c / FA3 branch unchanged; gate stays a view of qkv.
  SELF-CHECK in prepare at T = 753, 3012 and 16384 for both layer types and page_size 1 and 64: q8
  bytes and the written cache rows against master's five-launch path, at most 1-ulp E4M3
  differences in at most 0.1% of bytes (fp32 sum order in the RMS) and rel L2 <= 1e-3 after
  dequant; else OFF. TIMING GATE: cold-L2 graph of the five launches vs the fused launch, 2 rounds
  x 11 alternations per T class; ON at <= 0.9 in both rounds; state['front_fused_ok'][T class]
  plain bool. Print both medians and PATH CHOSEN.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "four fewer launches and one pass over the [753, 1280] q/k/v columns: ~-6 to -10 us of 756.7 / 1150.3 us (~1% sliding, ~0.7% full)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "_qk_rms_norm_fwd_kernel 10.8 us plus four small kernels become one ~6 us kernel: ~-12 to -18 us of 2197.8 / 3744.8 us (~0.5-0.8%)"
  - shapes: "*load=first_turn*"
    direction: faster
    why: the norm kernel moves ~80 MB in 42.2 us (~1.9 TB/s) and the casts and store add ~25 us; fused ~-30 to -50 us (~0.3-0.4%)
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "qk-norm-fp8-cache-store-fusion: the same 5->1 kernel re-measured on the 1.468 base (its 1.009 run was on the 1.004 parity base, where the five launches were a smaller share of every shape), now behind a prepare self-check and timing gate so a loss falls back to master"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.5%, stack-only"
---

# front-qknorm-rope-fp8-store-fusion-restack

## Change

Built on master (blob f4751fc0). Port _qk_norm_fp8_store_kernel from done/neutral/qk-norm-fp8-cache-store-fusion.patch (job fe8883a2, which passed every judge test) onto master's kernel_fn. After the qkv GEMM master runs five launches: FusedQKNorm (the Triton _qk_rms_norm_fwd_kernel: 42.2 us at T=16384, 10.8 us at T=3012), k.to(FP8), v.to(FP8) (a strided generic elementwise kernel), store_cache and q.view(T, 16, 64).to(FP8). The fused kernel reads each token's qkv row once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 (eps 1e-5, no weight), on sliding layers (rope_theta != 0) the interleaved RoPE from state['cos_sin'][positions], bf16 rounding then E4M3 for q (into q8 [T, 16, 64]) and k, v columns 1152..1279 cast to E4M3, and the k and v rows stored at cache_loc[t] of the FP8 caches (flat rows, so any page size). Use it only when config kv_scale == 1 (scaled configs keep master's code). q8 then feeds master's fa3c / FA3 branch unchanged; gate stays a view of qkv. SELF-CHECK in prepare at T = 753, 3012 and 16384 for both layer types and page_size 1 and 64: q8 bytes and the written cache rows against master's five-launch path, at most 1-ulp E4M3 differences in at most 0.1% of bytes (fp32 sum order in the RMS) and rel L2 <= 1e-3 after dequant; else OFF. TIMING GATE: cold-L2 graph of the five launches vs the fused launch, 2 rounds x 11 alternations per T class; ON at <= 0.9 in both rounds; state['front_fused_ok'][T class] plain bool. Print both medians and PATH CHOSEN.

## Expected

- `*load=turn_1*` faster: four fewer launches and one pass over the [753, 1280] q/k/v columns: ~-6 to -10 us of 756.7 / 1150.3 us (~1% sliding, ~0.7% full)
- `*load=turn_4*` faster: _qk_rms_norm_fwd_kernel 10.8 us plus four small kernels become one ~6 us kernel: ~-12 to -18 us of 2197.8 / 3744.8 us (~0.5-0.8%)
- `*load=first_turn*` faster: the norm kernel moves ~80 MB in 42.2 us (~1.9 TB/s) and the casts and store add ~25 us; fused ~-30 to -50 us (~0.3-0.4%)

## Risk

Size ~+0.5% headline: below the judge margin alone; it lands through front-routing-pdl-launch-diet-stack. The turn_1 rows are the board's most sensitive (1 us at sliding turn_1 is ~0.03% headline).
