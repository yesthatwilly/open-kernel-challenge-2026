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
review: hold
status: hold
review_at: 2026-10-25T203413Z
review_companions: null
hold_reason: "adversarial review: Duplicate of master: the landed combo-measured-rows-slide-batched-preattn-t8-tail (ec89382b, BEST 1.3142x) carries _pre_attention_kernel as source S3 (PAIRED an"
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

## Adversarial review

Reviewed 2026-10-25T203413Z by an adversarial subagent (model opus).

**Verdict: hold.** Duplicate of master: the landed combo-measured-rows-slide-batched-preattn-t8-tail (ec89382b, BEST 1.3142x) carries _pre_attention_kernel as source S3 (PAIRED and unpaired, _paired_norm_gate_kernel, per-cell self-checks and >=1.02x cold-L2 selectors) over full decode 2<=B<=64 sk/split-KV feeds, every sliding verify batch, full verify paired FA3 and full verify B=1. Its brief also requires an okc check first.

### Justification
Written 15:32Z against 6773ec5cf5, before the combo landed (19:21Z). master _attn_core routes use_s3 -> _s3_pre_attention; _s3_cell maps c1-b16/b32, c2-t8/t128/t256, c3-t128/t256, c4-t8 (submission.py 3264-3400, 5960-6027). Only residual slices: sliding decode_16/32 when S1 loses, full decode_1 when dl3 piece A is off, and attention's newer kernel version (7e12891512 vs 963bcf0267 lineage). Missing prior: the landed combo S3. Re-mint as a stacked hypothesis on the combo that swaps in the newer kernel version and/or adds an S3 cell for sliding decode with S1 off.

### Confounders on the current kernel
- combo-measured-rows-slide-batched-preattn-t8-tail (landed): its S3 is this change; S1 pre-empts sliding decode B=2..32.
- dl3-piece-a-gate-stride-check-fix / dense-chain-s3-t1-q-sk2-o-route-unpool: kernel Q already fuses the front half at full decode_1.
- full-decode-splitkv-restack-on-pdl / sk tickets: S3 c1 selector already measures the remaining launch count.
- fa3-hdim128-head-pairing-batched: S3 c3 already writes paired zero-padded q and uses _paired_norm_gate_kernel.

### Interference and A/B needs
No companion: on master it either reproduces the S3 selectors or swaps kernel versions.

### Open questions
- Which S3 cells were enabled in a903b9d004d2452d (check the log)?
- How does attention's 7e12891512 kernel differ from master's?
- When does S1 lose for sliding decode B=2..32?

