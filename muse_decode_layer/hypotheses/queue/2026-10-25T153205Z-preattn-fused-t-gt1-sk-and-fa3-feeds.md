---
problem: muse_decode_layer
slug: preattn-fused-t-gt1-sk-and-fa3-feeds
created: 2026-10-25T153205Z
base: master
area: pre-attention norm+rope + cache store (T > 1, every shape not routed to _sliding_forward or dl3 piece A)
change: >
  Replace the five pre-attention launches master still runs at T > 1 (FusedQKNorm with RoPE,
  k.to(FP8), v.to(FP8), store_cache, q.to(FP8); plus, on paired full verify, torch.zeros + two
  strided copies to build the padded paired FP8 query) with ONE launch of muse_attention_decode
  master's _pre_attention_kernel (commit 7e12891512; its first version gave attention 1.127x on
  every shape, and decode's own two ports moved in the right direction: fused-preattention-triton
  gained sliding decode_32, fused-preattention-paired-pad-restack gained full decode_1/16/32 and
  sliding verify_1, both unproven on since-moved bases). Copy the kernel verbatim with its PAIRED
  (writes the zero-padded 128-wide paired query straight into q_attn) and unpaired modes, its
  kv_scale handling (k / k_scale, v / v_scale before the E4M3 cast) and its rope_theta == 0
  branch; grid (tokens, hq + hkv) as in attention. Wire it in _parent_attention's else-branch
  (sliding T > 1, full T > 1, and full decode_1 when dl3 piece A is off): unpaired FP8 q feeds the
  sk / split-KV kernels on full decode 1 < B <= 64 and FA3 on sliding layers; PAIRED feeds FA3 on
  full verify (batch > 1). v and the gate still come from qkv views as now; FusedRMSNormGate and
  the paired un-pad are unchanged. RoPE uses state['cos_sin'] exactly as FusedQKNorm does (same
  table, fp32), and k/v cache rows are written in the same [page, slot, hkv*d] byte layout.
  Self-check in prepare at T in {2, 8, 16, 32, 40, 128, 256} x {sliding, full} x kv_scale {1,
  0.015625} with rope on: q (bf16 before cast) rel-L2 <= 1e-3 vs master, cache bytes by rel-L2 vs
  an accurate-rope fp32 reference (not byte-exact vs torch: Triton sin/cos), paired q halves zero
  exactly; any miss keeps master for that (layer kind, T bucket) with a PATH CHOSEN line. Selector
  per (layer kind, T in {16, 32, 128, 256}) on the whole attention side (qkv -> o-GEMM input),
  standard interleaved graph timing, >= 1.02x win. HARD RULES (every path): CUDA-graph capturable;
  kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input
  values (shape/config only), no data-dependent Python control flow and no per-call mutable Python
  state; gates read config/shape and this state's own prepare() flags only; nothing cached from
  inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV
  FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies
  and rotates them: keep every device pointer, TMA map, ticket array and workspace per state,
  never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only
  -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline
  module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch,
  and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell.
  Self-check by capturing once and replaying with refilled inputs while alternating two states.
  Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each,
  >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the
  new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print
  compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay
  under 270 s. First measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: "5 launches (~8-10 us at T=256) -> one ~4 us kernel (attention: _pre_attention_kernel 4.2 us at T=256); ~-4..-6 us on 600 us (weight 46)"
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: "same at T=128 (attention: 2.8 us kernel); ~-3..-5 us on 427 us"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "~-3 us on 209 us: 5 launches -> 1 on a launch-bound T=16"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "~-3 us on 236 us (decode's first port gained this row)"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: feeds sk directly; ~-3 us on 298 us
  - shapes: "*layer=full load=verify_*"
    direction: faster
    why: "paired q written directly: also drops torch.zeros + 2 copies (elementwise rows 6.9 us/3 calls at full verify_16); ~-5..-8 us"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: _sliding_forward does it in-kernel
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# preattn-fused-t-gt1-sk-and-fa3-feeds

## Change

Replace the five pre-attention launches master still runs at T > 1 (FusedQKNorm with RoPE, k.to(FP8), v.to(FP8), store_cache, q.to(FP8); plus, on paired full verify, torch.zeros + two strided copies to build the padded paired FP8 query) with ONE launch of muse_attention_decode master's _pre_attention_kernel (commit 7e12891512; its first version gave attention 1.127x on every shape, and decode's own two ports moved in the right direction: fused-preattention-triton gained sliding decode_32, fused-preattention-paired-pad-restack gained full decode_1/16/32 and sliding verify_1, both unproven on since-moved bases). Copy the kernel verbatim with its PAIRED (writes the zero-padded 128-wide paired query straight into q_attn) and unpaired modes, its kv_scale handling (k / k_scale, v / v_scale before the E4M3 cast) and its rope_theta == 0 branch; grid (tokens, hq + hkv) as in attention. Wire it in _parent_attention's else-branch (sliding T > 1, full T > 1, and full decode_1 when dl3 piece A is off): unpaired FP8 q feeds the sk / split-KV kernels on full decode 1 < B <= 64 and FA3 on sliding layers; PAIRED feeds FA3 on full verify (batch > 1). v and the gate still come from qkv views as now; FusedRMSNormGate and the paired un-pad are unchanged. RoPE uses state['cos_sin'] exactly as FusedQKNorm does (same table, fp32), and k/v cache rows are written in the same [page, slot, hkv*d] byte layout. Self-check in prepare at T in {2, 8, 16, 32, 40, 128, 256} x {sliding, full} x kv_scale {1, 0.015625} with rope on: q (bf16 before cast) rel-L2 <= 1e-3 vs master, cache bytes by rel-L2 vs an accurate-rope fp32 reference (not byte-exact vs torch: Triton sin/cos), paired q halves zero exactly; any miss keeps master for that (layer kind, T bucket) with a PATH CHOSEN line. Selector per (layer kind, T in {16, 32, 128, 256}) on the whole attention side (qkv -> o-GEMM input), standard interleaved graph timing, >= 1.02x win. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=verify_32*` faster: 5 launches (~8-10 us at T=256) -> one ~4 us kernel (attention: _pre_attention_kernel 4.2 us at T=256); ~-4..-6 us on 600 us (weight 46)
- `*layer=sliding load=verify_16*` faster: same at T=128 (attention: 2.8 us kernel); ~-3..-5 us on 427 us
- `*layer=sliding load=decode_16*` faster: ~-3 us on 209 us: 5 launches -> 1 on a launch-bound T=16
- `*layer=sliding load=decode_32*` faster: ~-3 us on 236 us (decode's first port gained this row)
- `*layer=full load=decode_16*` faster: feeds sk directly; ~-3 us on 298 us
- `*layer=full load=verify_*` faster: paired q written directly: also drops torch.zeros + 2 copies (elementwise rows 6.9 us/3 calls at full verify_16); ~-5..-8 us
- `*layer=sliding load=decode_1 *` flat: _sliding_forward does it in-kernel

## Risk

LESSONS: fusing a handful of tiny kernels was flat on decode-sized shapes in several runs, so the per-shape gain is small (~1%) and may sit inside drift on the headline; judge on the replaced rows (FusedQKNorm, the cast elementwise rows, store_cache, the pad zeros/copies) and the paired ratio per shape. Not a re-run of fused-preattention-triton / fused-preattention-paired-pad-restack: attention's newer kernel, current master (sk feeds, dense chain at T=1), and T=1 excluded (REWRITE_READ: superseded by the dense chain there). If sliding-sk-window-port-decode32-verify1 also lands, its two bands use its own route and the same kernel source; the merge keeps one copy.
